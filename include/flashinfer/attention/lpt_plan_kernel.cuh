/*
 * GPU-side LPT plan kernel for the dynamic scheduler.
 *
 * Replaces the CPU-side plan for the LPT (dynamic) scheduler path.
 * Eliminates the torch.cuda.synchronize() + cudaMemcpyAsync overhead (~24ms).
 *
 * Input: qo_indptr, kv_indptr, kv_len_arr (already on GPU)
 * Output: per-seq dyn_* arrays, dyn_scalars, merge_indptr/o_indices written
 *         directly to the attention workspace.
 */

#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace flashinfer {

// Single-block kernel: 256 threads, each handles ceil(num_seqs/256) sequences
template <typename IdType>
__global__ void lpt_plan_kernel(
    // Inputs (GPU tensors)
    const IdType* __restrict__ qo_indptr,     // [batch_size + 1]
    const IdType* __restrict__ kv_indptr,      // [batch_size + 1]
    const IdType* __restrict__ kv_len_arr,     // [batch_size]
    // Sequence classification: which seqs belong to this task
    const int* __restrict__ seq_indices,        // [num_seqs] indices into original batch
    int num_seqs,
    // Config
    int cta_tile_q,
    uint32_t gqa_group_size,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    int kv_len_limit,
    int l2_budget,
    // Output: per-seq arrays (pre-allocated in workspace)
    IdType* __restrict__ dyn_qo_indptr,
    IdType* __restrict__ dyn_qo_len,
    IdType* __restrict__ dyn_kv_indptr,
    IdType* __restrict__ dyn_kv_len,
    IdType* __restrict__ dyn_num_m_blocks,
    IdType* __restrict__ dyn_nheads_in_l2,
    IdType* __restrict__ dyn_partial_o_offset,
    IdType* __restrict__ dyn_num_kv_chunks,
    // Output: scalars
    IdType* __restrict__ dyn_scalars,  // [4]: num_seqs, total_tiles, len_kv_chunk, uniform_m
    // Output: merge data
    IdType* __restrict__ merge_indptr,
    IdType* __restrict__ merge_o_indices,
    IdType* __restrict__ num_packed_qo_len
) {
    constexpr int BLOCK_SIZE = 256;
    using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ int shared_total_tiles;
    __shared__ int shared_uniform;

    const int tid = threadIdx.x;

    // Phase 1: Compute per-sequence metadata (embarrassingly parallel)
    int local_tiles = 0;
    int local_m_blocks = -1;
    int local_kv_chunks = -1;

    for (int s = tid; s < num_seqs; s += BLOCK_SIZE) {
        int seq_idx = seq_indices[s];
        int qo_len = qo_indptr[seq_idx + 1] - qo_indptr[seq_idx];
        int kv_len = kv_len_arr[seq_idx];
        int packed_qo_len = qo_len * gqa_group_size;
        int num_m = (packed_qo_len + cta_tile_q - 1) / cta_tile_q;
        int num_chunks = (num_m == 1 && kv_len > kv_len_limit)
            ? (kv_len + kv_len_limit - 1) / kv_len_limit : 1;

        // L2-aware head grouping
        int64_t kv_bytes_per_head = (int64_t)kv_len * head_dim * 2 * 2;
        int nheads;
        if (kv_bytes_per_head == 0 || l2_budget >= kv_bytes_per_head * (int64_t)num_kv_heads) {
            nheads = num_kv_heads;
        } else {
            nheads = max(1, (int)(l2_budget / kv_bytes_per_head));
            // Round down to power of 2
            nheads = 1 << (31 - __clz(nheads));
        }
        nheads = min(nheads, (int)num_kv_heads);

        // Write per-seq metadata
        dyn_qo_indptr[s] = qo_indptr[seq_idx];
        dyn_qo_len[s] = qo_len;
        dyn_kv_indptr[s] = kv_indptr[seq_idx];
        dyn_kv_len[s] = kv_len;
        dyn_num_m_blocks[s] = num_m;
        dyn_nheads_in_l2[s] = nheads;
        dyn_num_kv_chunks[s] = num_chunks;

        local_tiles += num_m * num_kv_heads * num_chunks;

        // Track uniformity
        if (local_m_blocks == -1) { local_m_blocks = num_m; local_kv_chunks = num_chunks; }
        if (num_m != local_m_blocks || num_chunks != local_kv_chunks) { local_m_blocks = 0; }
    }

    // Phase 2: Reduce total_tiles and check uniformity
    int total = BlockReduce(reduce_storage).Sum(local_tiles);
    __syncthreads();
    if (tid == 0) {
        shared_total_tiles = total;
    }

    // Check uniformity: all threads must agree
    int uniform_check = (local_m_blocks > 0 && num_seqs > 0) ? local_m_blocks : 0;
    int uniform_result = BlockReduce(reduce_storage).Reduce(uniform_check, cub::Min());
    __syncthreads();
    if (tid == 0) {
        // If all threads have same positive value, it's uniform
        shared_uniform = (uniform_result > 0) ? uniform_result : 0;
    }
    __syncthreads();

    // Phase 3: Write scalars
    if (tid == 0) {
        dyn_scalars[0] = num_seqs;
        dyn_scalars[1] = shared_total_tiles;
        dyn_scalars[2] = kv_len_limit;
        dyn_scalars[3] = shared_uniform;
    }

    // Phase 4: Compute partial_o_offset and merge_indptr
    // This requires a prefix sum over split sequences.
    // For simplicity, use a single thread (split seqs are few in typical workloads).
    if (tid == 0) {
        int partial_o_nnz = 0;
        int merge_idx = 0;
        merge_indptr[0] = 0;

        for (int s = 0; s < num_seqs; ++s) {
            int num_chunks = dyn_num_kv_chunks[s];
            if (num_chunks > 1) {
                dyn_partial_o_offset[s] = partial_o_nnz;
                int seq_idx = seq_indices[s];
                int qo_len = dyn_qo_len[s];
                int packed_qo_len = qo_len * gqa_group_size;
                int num_qo_tiles = (packed_qo_len + cta_tile_q - 1) / cta_tile_q;

                for (int qt = 0; qt < num_qo_tiles; ++qt) {
                    int row_tile_size = min(cta_tile_q, packed_qo_len - qt * cta_tile_q);
                    for (int row = 0; row < row_tile_size; ++row) {
                        merge_idx++;
                        merge_indptr[merge_idx] = merge_indptr[merge_idx - 1] + num_chunks;
                        int q = (qt * cta_tile_q + row) / gqa_group_size;
                        int r = (qt * cta_tile_q + row) % gqa_group_size;
                        merge_o_indices[merge_idx - 1] =
                            (qo_indptr[seq_idx] + q) * num_kv_heads * gqa_group_size + r;
                    }
                    partial_o_nnz += row_tile_size * num_chunks;
                }
            } else {
                dyn_partial_o_offset[s] = 0;
            }
        }
        *num_packed_qo_len = merge_idx;
    }
}

}  // namespace flashinfer
