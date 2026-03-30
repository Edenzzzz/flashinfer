/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_PERSISTENT_CUH_
#define FLASHINFER_PERSISTENT_CUH_

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../utils.cuh"
#include "mask.cuh"
#include "persistent_template.cuh"
#include "prefill.cuh"
#include "state.cuh"

namespace flashinfer {

using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

template <typename Params>
__device__ __forceinline__ auto get_block_coord(const Params& params, const uint32_t work_idx) {
  return std::tuple(params.q_indptr[work_idx], params.kv_indptr[work_idx],
                    params.partial_indptr[work_idx], params.q_len[work_idx],
                    params.kv_len[work_idx], params.q_start[work_idx], params.kv_start[work_idx],
                    params.kv_end[work_idx], params.kv_head_idx_arr[work_idx],
                    *params.len_kv_chunk);
}

/*!
 * \brief Warp-level inclusive prefix sum using shuffle instructions.
 * Each lane i holds value[i], returns sum of value[0..i].
 */
__device__ __forceinline__ int warp_prefix_sum(int value) {
  for (int delta = 1; delta < 32; delta <<= 1) {
    int n = __shfl_up_sync(0xffffffff, value, delta);
    if ((threadIdx.x % 32) >= delta) value += n;
  }
  return value;
}

/*!
 * \brief FA3-style dynamic tile scheduler: maps a global tile index to
 * (seq_idx, kv_head_idx, q_tile_idx) with L2-aware LPT ordering.
 *
 * Tile layout: batch (slowest) → head → query block (fastest).
 * Within each sequence, heads are grouped into L2 sections; within each section,
 * query blocks iterate first (inner), heads second (outer) for L2 locality.
 * Query blocks are reversed (LPT) for causal attention load balancing.
 *
 * \tparam CTA_TILE_Q The query tile size for this runner
 * \param params The persistent params containing per-sequence metadata
 * \param tile_idx The global tile index from atomicAdd
 * \param prev_bidb The batch index from the previous iteration (for warm start)
 * \param prev_group_start The group_start_tile from previous iteration
 *
 * Returns tuple of (q_indptr, kv_indptr, partial_o_offset, q_len, kv_len,
 *                    packed_qo_start, kv_start, kv_end, kv_head_idx, len_kv_chunk)
 * matching get_block_coord() signature, plus updated (bidb, group_start_tile).
 */
// Helper: read dynamic scheduler scalar from GPU buffer (CUDA graph) or frozen param.
// dyn_scalars layout per task: [num_seqs, total_tiles, len_kv_chunk, uniform_num_m_blocks]
template <typename Params, typename IdType>
__device__ __forceinline__ int dyn_scalar(const Params& params, int idx, int frozen_val) {
  return params.dyn_scalars ? params.dyn_scalars[idx] : frozen_val;
}

template <uint32_t CTA_TILE_Q, bool CAUSAL, typename Params, typename IdType>
__device__ __forceinline__ auto tile_idx_to_work_tile(
    const Params& params, int tile_idx, int prev_bidb, int prev_group_start) {

  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint_fastdiv& gqa_group_size = params.gqa_group_size;
  const int uniform_m = dyn_scalar<Params, IdType>(params, 3, params.dyn_uniform_num_m_blocks);

  int bidb, num_m_blocks, mh_block, group_start_tile;

  if (uniform_m > 0) {
    // --- Fast O(1) path: all sequences have same tiles_per_seq ---
    // uniform_m > 0 only when all seqs have same num_m_blocks AND same num_kv_chunks.
    // tiles_per_seq = uniform_m * num_kv_heads * uniform_chunks
    int uniform_chunks = params.dyn_num_kv_chunks ? params.dyn_num_kv_chunks[0] : 1;
    int tiles_per_seq = uniform_m * num_kv_heads * uniform_chunks;
    bidb = tile_idx / tiles_per_seq;
    mh_block = tile_idx - bidb * tiles_per_seq;
    num_m_blocks = uniform_m;
    group_start_tile = bidb * tiles_per_seq;
  } else {
    // --- General path: warp-level binary search for sequence ---
    const int lane = threadIdx.x % 32;
    const int dyn_ns = dyn_scalar<Params, IdType>(params, 0, params.dyn_num_seqs);

    auto get_num_tiles = [&](int seq_start) -> int {
      int seq_idx = lane + seq_start;
      if (seq_idx < dyn_ns && lane < 31) {
        int nm = params.dyn_num_m_blocks[seq_idx];
        int nc = params.dyn_num_kv_chunks ? params.dyn_num_kv_chunks[seq_idx] : 1;
        return nm * nc;
      }
      return 0;
    };

    bidb = prev_bidb;
    int num_tiles = get_num_tiles(bidb);
    int num_tiles_cumulative = warp_prefix_sum(num_tiles);
    int tiles_in_group = __shfl_sync(0xffffffff, num_tiles_cumulative, 31);
    int group_end_tile = prev_group_start + tiles_in_group * num_kv_heads;

    while (group_end_tile <= tile_idx) {
      bidb += 31;
      if (bidb >= dyn_ns) {
        return std::tuple((IdType)0, (IdType)0, (IdType)0, (IdType)0, (IdType)0,
                          (IdType)0, (IdType)0, (IdType)0, (IdType)0,
                          dyn_scalar<Params, IdType>(params, 2, params.dyn_len_kv_chunk), bidb, group_end_tile);
      }
      num_tiles = get_num_tiles(bidb);
      num_tiles_cumulative = warp_prefix_sum(num_tiles);
      tiles_in_group = __shfl_sync(0xffffffff, num_tiles_cumulative, 31);
      group_end_tile += tiles_in_group * num_kv_heads;
    }
    group_start_tile = group_end_tile - tiles_in_group * num_kv_heads;

    int batch_idx_in_group = __popc(__ballot_sync(0xffffffff,
        group_start_tile + num_tiles_cumulative * num_kv_heads <= tile_idx));
    bidb += batch_idx_in_group;
    num_m_blocks = __shfl_sync(0xffffffff, num_tiles, batch_idx_in_group);
    group_start_tile += (batch_idx_in_group == 0 ? 0 :
        __shfl_sync(0xffffffff, num_tiles_cumulative, batch_idx_in_group - 1)) * num_kv_heads;
    mh_block = tile_idx - group_start_tile;
  }

  // --- Step 2: Extract chunk index, then map (head, q_block) with L2 awareness ---
  int num_kv_chunks = params.dyn_num_kv_chunks ? params.dyn_num_kv_chunks[bidb] : 1;
  int mh_block_with_chunks = mh_block;
  int chunk_idx = 0;
  if (num_kv_chunks > 1) {
    // Layout: mhc_block = mh_block * num_kv_chunks + chunk_idx
    mh_block = mh_block_with_chunks / num_kv_chunks;
    chunk_idx = mh_block_with_chunks - mh_block * num_kv_chunks;
  }

  int nheads_in_l2 = params.dyn_nheads_in_l2[bidb];
  int mh_in_l2 = nheads_in_l2 * num_m_blocks;
  int section_idx = mh_block / mh_in_l2;
  int l2_mod = mh_block - section_idx * mh_in_l2;
  int nheads_remainder = num_kv_heads - section_idx * nheads_in_l2;
  int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
  int block = l2_mod / nheads_in_this_section;
  int bidh_residual = l2_mod - block * nheads_in_this_section;
  int kv_head_idx = section_idx * nheads_in_l2 + bidh_residual;

  // LPT: reverse query block order (largest first for causal)
  block = num_m_blocks - 1 - block;

  // --- Step 3: Compute the work coordinates ---
  int qo_len = params.dyn_qo_len[bidb];
  int packed_qo_len = qo_len * (uint32_t)gqa_group_size;
  int packed_qo_start = block * CTA_TILE_Q;
  int kv_len = params.dyn_kv_len[bidb];
  int len_kv_chunk_val = dyn_scalar<Params, IdType>(params, 2, params.dyn_len_kv_chunk);

  // Split-KV: compute kv_start/kv_end for this chunk
  int kv_start = chunk_idx * len_kv_chunk_val;
  int kv_end = min((int)kv_len, kv_start + len_kv_chunk_val);
  // Clamp kv_end for causal: avoid processing KV beyond the causal boundary
  if constexpr (CAUSAL) {
    uint32_t gqa_group_size = params.gqa_group_size;
    int causal_kv_end = (int)kv_len - (int)qo_len +
        (int)ceil_div((uint32_t)packed_qo_start + CTA_TILE_Q, gqa_group_size);
    kv_end = max(0, min(kv_end, causal_kv_end));
  }
  kv_start = min(kv_start, kv_end);  // ensure kv_start <= kv_end

  // For split KV: use per-sequence partial_o_offset from planner
  int partial_o_offset = params.dyn_partial_o_offset ? params.dyn_partial_o_offset[bidb] : 0;

  return std::tuple(
      (IdType)params.dyn_qo_indptr[bidb],
      (IdType)params.dyn_kv_indptr[bidb],
      (IdType)partial_o_offset,
      (IdType)qo_len,
      (IdType)kv_len,
      (IdType)packed_qo_start,
      (IdType)kv_start,
      (IdType)kv_end,
      (IdType)kv_head_idx,
      dyn_scalar<Params, IdType>(params, 2, params.dyn_len_kv_chunk),
      bidb,
      group_start_tile
  );
}

template <typename KTraits>
__device__ __forceinline__ void prefetch_offest(
    const uint32_t packed_block_iter_base, const uint32_t packed_kv_bound,
    const uint32_t kv_head_idx, const uint32_t kv_stride_page, const uint32_t kv_stride_h,
    const uint32_t kv_stride_n, const uint_fastdiv& block_size, typename KTraits::IdType* indices,
    size_t* kv_offset) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
  constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;

#pragma unroll
  for (uint32_t i = 0;
       i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
    uint32_t page_iter, entry_idx;
    uint32_t packed_block_iter = packed_block_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                 lane_idx / KV_THR_LAYOUT_COL +
                                 KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i;
    block_size.divmod(packed_block_iter, page_iter, entry_idx);
    kv_offset[i] = (packed_block_iter < packed_kv_bound ? indices[page_iter] : 0) * kv_stride_page +
                   entry_idx * kv_stride_n + kv_head_idx * kv_stride_h +
                   (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_(float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                         smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
                                         typename KTraits::DTypeO* o_ptr_base,
                                         const uint32_t o_packed_idx_base_warp,
                                         const uint32_t o_packed_idx_base_cta,
                                         const uint32_t qo_upper_bound, const uint32_t o_stride_n,
                                         const uint_fastdiv group_size, const uint32_t warp_idx,
                                         const uint32_t lane_idx, const dim3 tid) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y),
                 warp_idx_z = get_warp_idx_kv<KTraits>(tid.z);

  static_assert(sizeof(DTypeO) == 2);
  if (warp_idx_z == 0) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        uint32_t o_frag_f16[8 / 2];
        vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);

#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
            (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx % 16,
            mma_d * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
            (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * UPCAST_STRIDE_O))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_O))[lane_idx % 4] =
            o_frag_f16[3];
#endif
      }
    }

    uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        const uint32_t o_packed_idx = o_packed_idx_base_warp + lane_idx / 8 + mma_q * 16 + j * 4;
        group_size.divmod(o_packed_idx, q, r);

        const uint32_t o_idx = q;
        DTypeO* o_ptr = o_ptr_base + (o_packed_idx - o_packed_idx_base_cta) * o_stride_n +
                        (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_VO / 4; ++mma_do) {
          if (o_idx < qo_upper_bound) {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
          o_ptr += 8 * upcast_size<DTypeO>();
          o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, mma_do);
        }
        o_smem_offset_w =
            o_smem->template advance_offset_by_row<4, UPCAST_STRIDE_O>(o_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_VO;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d(float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2],
                                            float v_scale = 1.0f) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  if constexpr (AttentionVariant::use_softmax) {
    float d_rcp[KTraits::NUM_MMA_Q][2];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        d_rcp[mma_q][j] = (m[mma_q][j] != typename KTraits::DTypeQKAccum(-math::inf))
                              ? math::ptx_rcp(d[mma_q][j])
                              : 0.f;
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[mma_q][mma_d][reg_id] =
              o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][(reg_id >> 1) & 1];
          if (v_scale != 1.0f) {
            o_frag[mma_q][mma_d][reg_id] *= v_scale;
          }
        }
      }
    }
  }
}

template <typename KTraits_, typename Params_>
struct BlockBatchPagedAttentionPersistent {
  using KTraits = KTraits_;
  using Params = Params_;

  static __device__ __forceinline__ void Run(const Params& params,
                                             typename KTraits::SharedStorage* smem_storage
                                                 PROFILER_CLOSURE_FUNC_PARAMS) {
    using DTypeQ = typename Params::DTypeQ;
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr bool CAUSAL = KTraits::MASK_MODE == MaskMode::kCausal;
    [[maybe_unused]] constexpr uint32_t NUM_STAGES = KTraits::NUM_STAGES;

    DTypeQ* q = params.q;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    IdType* kv_indices = params.kv_indices;
    float* partial_lse = params.partial_lse;
    IdType* work_indptr = params.work_indptr;

    float s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    float m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];

    const uint_fastdiv& gqa_group_size = params.gqa_group_size;
    const uint32_t num_kv_heads = params.num_kv_heads;
    const uint_fastdiv& block_size = params.page_size;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_page = params.k_stride_page;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t v_stride_page = params.v_stride_page;
    const uint32_t v_stride_h = params.v_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;
    smem_t<SWIZZLE_MODE_Q> q_smem(smem_storage->q_smem);

    AttentionVariant variant(params, /*batch_idx=*/0, nullptr);

    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t warp_idx = threadIdx.x / 32;

    // threadIdx: [32, NUM_WARPS_Q, NUM_WARPS_KV]
    // remap to utilize tool function in FA2 prefill
    const dim3 tid = dim3(lane_idx, warp_idx % NUM_WARPS_Q, warp_idx / NUM_WARPS_Q);

    uint32_t q_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_Q, UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);
    uint32_t k_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16);
    uint32_t k_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
                 warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
                 lane_idx % KTraits::KV_THR_LAYOUT_COL),
             v_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
                 warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
                 lane_idx % KTraits::KV_THR_LAYOUT_COL);
    size_t thr_local_kv_offset[NUM_MMA_KV * KTraits::KV_THR_LAYOUT_COL / 2 / KTraits::NUM_WARPS_Q];

    // Choose scheduling mode: read from GPU buffer for CUDA graph, else from frozen param
    const int eff_dyn_num_seqs = dyn_scalar<Params, IdType>(params, 0, params.dyn_num_seqs);
    const bool use_dynamic = (eff_dyn_num_seqs > 0);

    // Dynamic scheduler state (for tile_idx_to_work_tile warm start)
    int dyn_bidb = 0, dyn_group_start = 0;

#pragma unroll 1
    for (IdType work_idx = use_dynamic ? 0 : work_indptr[blockIdx.y]; ; ) {
      IdType q_indptr_val, kv_indptr_val, o_indptr, q_len, kv_len;
      IdType packed_qo_start, kv_start, kv_end, kv_head_idx;
      int len_kv_chunk;

      if (use_dynamic) {
        // Dynamic scheduling: sequential atomicAdd (strict LPT order)
        __syncthreads();
        IdType tile_idx;
        if (threadIdx.x == 0) {
          tile_idx = atomicAdd(params.tile_counter, 1);
        }
        // Broadcast tile_idx to all threads via smem
        if (warp_idx == 0 && lane_idx == 0) {
          reinterpret_cast<volatile IdType*>(smem_storage)[0] = tile_idx;
        }
        __syncthreads();
        tile_idx = reinterpret_cast<volatile IdType*>(smem_storage)[0];
        if (tile_idx >= dyn_scalar<Params, IdType>(params, 1, params.dyn_total_tiles)) break;

        // Map tile_idx to work coordinates
        auto result = tile_idx_to_work_tile<CTA_TILE_Q, CAUSAL, Params, IdType>(
            params, tile_idx, dyn_bidb, dyn_group_start);
        q_indptr_val = std::get<0>(result);
        kv_indptr_val = std::get<1>(result);
        o_indptr = std::get<2>(result);
        q_len = std::get<3>(result);
        kv_len = std::get<4>(result);
        packed_qo_start = std::get<5>(result);
        kv_start = std::get<6>(result);
        kv_end = std::get<7>(result);
        kv_head_idx = std::get<8>(result);
        len_kv_chunk = std::get<9>(result);
        dyn_bidb = std::get<10>(result);
        dyn_group_start = std::get<11>(result);
        // tile mapping complete
        // Check for sentinel (past end of sequences)
        if (dyn_bidb >= eff_dyn_num_seqs) break;
      } else {
        // Static scheduling: iterate work_indptr range
        if (work_idx >= work_indptr[blockIdx.y + 1]) break;
        auto coords = get_block_coord(params, work_idx);
        q_indptr_val = std::get<0>(coords);
        kv_indptr_val = std::get<1>(coords);
        o_indptr = std::get<2>(coords);
        q_len = std::get<3>(coords);
        kv_len = std::get<4>(coords);
        packed_qo_start = std::get<5>(coords);
        kv_start = std::get<6>(coords);
        kv_end = std::get<7>(coords);
        kv_head_idx = std::get<8>(coords);
        len_kv_chunk = std::get<9>(coords);
      }

      // profile log
      if constexpr (CTA_TILE_Q > 64) {
        PROFILER_EVENT_START(profiler_closure, PersistentProfileEventType::kRunner1);
      } else {
        PROFILER_EVENT_START(profiler_closure, PersistentProfileEventType::kRunner2);
      }

      // Alias for compatibility with rest of the function
      const auto q_indptr = q_indptr_val;
      const auto kv_indptr = kv_indptr_val;

      const uint32_t kv_chunk_idx = kv_start / len_kv_chunk;
      const uint32_t num_kv_chunks = ceil_div(
          CAUSAL
              ? min((kv_len - q_len) + ceil_div(packed_qo_start + cluster_tile_q, gqa_group_size),
                    kv_len)
              : kv_len,
          len_kv_chunk);
      const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q +
                                          get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16;
      const uint32_t qo_upperbound =
          min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));

      init_states<KTraits>(variant, o_frag, m, d);

      DTypeQ* q_ptr_base = q + q_indptr * q_stride_n + (kv_head_idx * gqa_group_size) * q_stride_h;

      // load_q
      load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upperbound, q_ptr_base, q_stride_n,
                                  q_stride_h, gqa_group_size, &q_smem, tid);

      smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage->k_smem), v_smem(smem_storage->v_smem);
      int kv_tile_idx =
          ceil_div((CAUSAL ? min(kv_end,
                                 kv_len - q_len +
                                     ceil_div((packed_qo_start + cluster_tile_q), gqa_group_size))
                           : kv_end),
                   CTA_TILE_KV) -
          1 - (kv_start / CTA_TILE_KV);

      int mask_tile_idx =
          (CAUSAL ? min(kv_end, kv_len - q_len + ceil_div(packed_qo_start, gqa_group_size))
                  : kv_end) /
              CTA_TILE_KV -
          (kv_start / CTA_TILE_KV);

      uint32_t block_iter_base = kv_indptr * block_size + kv_start;
      // last kv tile
      __syncthreads();
      uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;

      prefetch_offest<KTraits>(block_iter_base + kv_tile_idx * CTA_TILE_KV, packed_kv_bound,
                               kv_head_idx, k_stride_page, k_stride_h, k_stride_n, block_size,
                               kv_indices, thr_local_kv_offset);
      page_produce_kv<false, KTraits>(smem_storage, &k_smem_offset_w, k,
                                      kv_start + kv_tile_idx * CTA_TILE_KV, thr_local_kv_offset,
                                      kv_end, warp_idx, lane_idx);
      cp_async::commit_group();
      page_produce_kv<true, KTraits>(smem_storage, &v_smem_offset_w, v,
                                     kv_start + kv_tile_idx * CTA_TILE_KV, thr_local_kv_offset,
                                     kv_end, warp_idx, lane_idx);
      cp_async::commit_group();

      // loop with mask
      LOOP_SPLIT_MASK(
          kv_tile_idx, kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0,
          kv_tile_idx + 1 > NUM_STAGES, {
            prefetch_offest<KTraits>(block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV,
                                     packed_kv_bound, kv_head_idx, k_stride_page, k_stride_h,
                                     k_stride_n, block_size, kv_indices, thr_local_kv_offset);
            cp_async::wait_group<1>();
            __syncthreads();

            compute_qk<KTraits>(&q_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
            if constexpr (AttentionVariant::use_logits_soft_cap) {
              logits_transform<KTraits>(
                  params, variant, /*batch_idx=*/0, qo_packed_idx_base,
                  kv_start + (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) *
                                 NUM_MMA_KV * 16,
                  q_len, kv_len, gqa_group_size, s_frag, tid, kv_head_idx);
            }
            if constexpr (WITH_MASK) {
              logits_mask<KTraits>(
                  params, variant, /*batch_idx=*/0, qo_packed_idx_base,
                  kv_start + (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) *
                                 NUM_MMA_KV * 16,
                  q_len, kv_len, kv_end, gqa_group_size, s_frag, tid, kv_head_idx);
            }
            update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

            __syncthreads();
            page_produce_kv<false, KTraits>(smem_storage, &k_smem_offset_w, k,
                                            kv_start + (kv_tile_idx - 1) * CTA_TILE_KV,
                                            thr_local_kv_offset, kv_end, warp_idx, lane_idx);
            cp_async::commit_group();
            cp_async::wait_group<1>();

            __syncthreads();
            compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
            __syncthreads();

            page_produce_kv<true, KTraits>(smem_storage, &v_smem_offset_w, v,
                                           kv_start + (kv_tile_idx - 1) * CTA_TILE_KV,
                                           thr_local_kv_offset, kv_end, warp_idx, lane_idx);
            cp_async::commit_group();
          });
      cp_async::wait_group<0>();
      __syncthreads();

#pragma unroll
      for (; kv_tile_idx >= 0; --kv_tile_idx) {
        compute_qk<KTraits>(&q_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
        if constexpr (AttentionVariant::use_logits_soft_cap) {
          logits_transform<KTraits>(
              params, variant, /*batch_idx=*/0, qo_packed_idx_base,
              kv_start +
                  (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
              q_len, kv_len, gqa_group_size, s_frag, tid, kv_head_idx);
        }
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/0, qo_packed_idx_base,
            kv_start +
                (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
            q_len, kv_len, kv_end, gqa_group_size, s_frag, tid, kv_head_idx);
        update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
      }

      __syncthreads();

      finalize_m<KTraits>(variant, m);

      // threadblock synchronization
      threadblock_sync_mdo_states<KTraits>(o_frag, smem_storage, m, d, warp_idx, lane_idx, tid);

      // normalize d
      normalize_d<KTraits>(o_frag, m, d, params.v_scale);

      // write back to global memory
      // o_indptr (partial_o): [packed_qo_len * num_kv_chunks, num_kv_heads, head_dim]
      // q_indpt (final_o): [qo_len, num_kv_heads, gqa_group_size, head_dim]
      if (num_kv_chunks > 1) {
        DTypeO* o_ptr_base = params.partial_o +
                             ((o_indptr + kv_chunk_idx) * num_kv_heads + kv_head_idx) * HEAD_DIM_VO;
        write_o_<KTraits>(o_frag, &q_smem, o_ptr_base, qo_packed_idx_base, packed_qo_start,
                          qo_upperbound, num_kv_chunks * num_kv_heads * HEAD_DIM_VO, gqa_group_size,
                          warp_idx, lane_idx, tid);
      } else {
        // write through
        // o_stride_n = num_qo_heads* head_dim
        const uint32_t o_stride_n = num_kv_heads * gqa_group_size * HEAD_DIM_VO,
                       o_stride_h = HEAD_DIM_VO;
        DTypeO* o_ptr_base =
            params.final_o + q_indptr * o_stride_n + (kv_head_idx * gqa_group_size) * o_stride_h;
        write_o_reg_gmem<KTraits>(o_frag, &q_smem, o_ptr_base, qo_packed_idx_base, q_len,
                                  o_stride_n, o_stride_h, gqa_group_size, tid);
      }

      if constexpr (variant.use_softmax) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              const uint32_t packed_qo_idx = qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16;
              gqa_group_size.divmod(packed_qo_idx, q, r);
              if (q < qo_upperbound) {
                if (num_kv_chunks > 1) {
                  partial_lse[(o_indptr + (packed_qo_idx - packed_qo_start) * num_kv_chunks +
                               kv_chunk_idx) *
                                  num_kv_heads +
                              kv_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else if (params.final_lse != nullptr) {
                  // write through
                  const uint32_t qo_head_idx = kv_head_idx * gqa_group_size + r;
                  params.final_lse[(q_indptr + q) * num_kv_heads * gqa_group_size + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }

      // profile
      if constexpr (CTA_TILE_Q > 64) {
        PROFILER_EVENT_END(profiler_closure, PersistentProfileEventType::kRunner1);
      } else {
        PROFILER_EVENT_END(profiler_closure, PersistentProfileEventType::kRunner2);
      }

      // Advance to next work item (for static path; dynamic uses atomic at loop top)
      if (!use_dynamic) {
        ++work_idx;
      }
    }
  }
};

template <uint32_t HEAD_DIM_VO_, uint32_t NUM_SMEM_STAGES_, uint32_t NUM_THREADS_,
          typename DTypeIn_, typename DTypeO_, typename IdType_>
struct StateReductionKernelTraits {
  using DTypeIn = DTypeIn_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  static constexpr uint32_t HEAD_DIM_VO = HEAD_DIM_VO_;
  static constexpr uint32_t NUM_SMEM_STAGES = NUM_SMEM_STAGES_;
  static constexpr uint32_t NUM_THREADS = NUM_THREADS_;
  static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;

  static constexpr uint32_t vec_size =
      std::max<uint32_t>(16U / static_cast<uint32_t>(sizeof(DTypeIn)), HEAD_DIM_VO / 32U);
  static constexpr uint32_t bdx = HEAD_DIM_VO / vec_size;

  // gridDim is accessed by runtime variable and should be set by core attention
  // workload layout [bdx, bdy, num_warps]
  static_assert(NUM_THREADS % bdx == 0);
  static constexpr uint32_t bdy = 32 / bdx;

  // pipeline load & reduction
  static constexpr size_t SMEM_SIZE =
      NUM_WARPS * NUM_SMEM_STAGES * bdy * HEAD_DIM_VO * sizeof(DTypeIn) +
      NUM_THREADS * sizeof(float);
};

template <typename KTraits_>
struct BlockBatchReductionPersistent {
  using KTraits = KTraits_;

  static __device__ __forceinline__ void Run(
      typename KTraits::DTypeIn* __restrict__ V, typename KTraits::DTypeO* __restrict__ v_merged,
      float* __restrict__ S, float* __restrict__ s_merged,
      const typename KTraits::IdType num_packed_qo_len, const uint_fastdiv gqa_group_size,
      const uint32_t num_kv_heads, const typename KTraits::IdType* indptr,
      const typename KTraits::IdType* o_indices, uint8_t* smem PROFILER_CLOSURE_FUNC_PARAMS) {
    __syncthreads();  // NOTE(Zihao): required for guarantee correctness on blackwell
    using DTypeIn = typename KTraits::DTypeIn;
    using DTypeO = typename KTraits::DTypeO;
    using IdType = typename KTraits::IdType;

    [[maybe_unused]] constexpr uint32_t bdx = KTraits::bdx;
    [[maybe_unused]] constexpr uint32_t bdy = KTraits::bdy;
    [[maybe_unused]] constexpr uint32_t num_warps = KTraits::NUM_WARPS;

    [[maybe_unused]] constexpr uint32_t vec_size = KTraits::vec_size;
    [[maybe_unused]] constexpr uint32_t head_dim = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t num_smem_stages = KTraits::NUM_SMEM_STAGES;
    [[maybe_unused]] constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;

    // control flow metadata
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t tx = (threadIdx.x % 32) % bdx, ty = (threadIdx.x % 32) / bdx;

    const uint32_t worker_id = blockIdx.y * num_warps + warp_idx;
    const uint32_t num_workers = gridDim.x * gridDim.y * gridDim.z * num_warps;

    DTypeIn* v_smem = (DTypeIn*)smem + warp_idx * num_smem_stages * bdy * head_dim;
    // FIXME: fix the offset calculation
    float* s_smem = (float*)(smem + num_warps * num_smem_stages * bdy * head_dim * sizeof(DTypeIn) +
                             warp_idx * 32 * sizeof(float));

    // V: [num_packed_qo_len x num_kv_tiles, num_kv_heads, head_dim]
    // v_merged: [qo_len, num_kv_heads, gqa_group_size, head_dim]
#pragma unroll 1
    for (uint32_t i = worker_id; i < num_packed_qo_len * num_kv_heads; i += num_workers) {
      __syncwarp();  // avoid data hazard due to reordering st.cast_store
      PROFILER_EVENT_START(profiler_closure, PersistentProfileEventType::kReduction);
      __syncwarp();  // avoid data hazard due to reordering st.cast_store
      // remap workload
      uint32_t packed_qo_idx = i / num_kv_heads;
      uint32_t kv_head_idx = i % num_kv_heads;
      const uint32_t num_index_sets = indptr[packed_qo_idx + 1] - indptr[packed_qo_idx];
      if (num_index_sets == 0 || num_index_sets == 1) {
        // already write through, bypass
        PROFILER_EVENT_END(profiler_closure, PersistentProfileEventType::kReduction);
        continue;
      }

      // index calculation
      auto partial_idx_to_offset = [&](uint32_t off) {
        return (indptr[packed_qo_idx] + off) * num_kv_heads + kv_head_idx;
      };
      auto merge_idx_to_offset = [&]() {
        // NOTE (Yilong): qo_head_idx has been calculated in schedule.plan
        return o_indices[packed_qo_idx] + kv_head_idx * gqa_group_size;
      };

      state_t<vec_size> st;

#pragma unroll
      for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
        cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
            v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
            V + partial_idx_to_offset(iter * bdy + ty) * head_dim + tx * vec_size,
            (iter * bdy + ty) < num_index_sets);
        cp_async::commit_group();
      }
#pragma unroll 4
      for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
        if (iter % bdx == 0) {
          s_smem[ty * bdx + tx] = iter * bdy + (ty * bdx + tx) < num_index_sets
                                      ? S[partial_idx_to_offset(iter * bdy + ty * bdx + tx)]
                                      : 0.f;
          __syncwarp();
        }
        cp_async::wait_group<num_smem_stages - 1>();
        __syncwarp();
        vec_t<float, vec_size> v;
        v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
        if (iter * bdy + ty < num_index_sets) {
          float s = s_smem[(iter % bdx) * bdy + ty];
          st.merge(v, s, 1);
        }
        __syncwarp();
        cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
            v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
            V + partial_idx_to_offset((iter + num_smem_stages) * bdy + ty) * head_dim +
                tx * vec_size,
            (iter + num_smem_stages) * bdy + ty < num_index_sets);
        cp_async::commit_group();
      }
      cp_async::wait_group<0>();
      __syncwarp();

      st.normalize();
      if constexpr (bdy > 1) {
        warp_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem, tx, ty);
        st.normalize();
      }

      st.o.cast_store(v_merged + merge_idx_to_offset() * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[merge_idx_to_offset()] = st.get_lse();
      }
      PROFILER_EVENT_END(profiler_closure, PersistentProfileEventType::kReduction);
    }
  }
};

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, typename AttentionVariant, bool FlippedSchedule, typename Params>
cudaError_t BatchPagedAttentionPersistent(const Params params_1, const Params params_2,
                                          const uint32_t num_blks_x, const uint32_t num_blks_y,
                                          const cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  constexpr uint32_t NUM_WARPS_Q_1 = get_num_warps_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_WARPS_KV_1 = get_num_warps_kv(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_Q_1 = get_num_mma_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_KV_1 = 4;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using KTraits1 = KernelTraits<MASK_MODE, CTA_TILE_Q_1, NUM_MMA_Q_1, NUM_MMA_KV_1, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_1, NUM_WARPS_KV_1, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;
  constexpr uint32_t NUM_WARPS_Q_2 = get_num_warps_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_WARPS_KV_2 = get_num_warps_kv(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_Q_2 = get_num_mma_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_KV_2 = 2;
  using KTraits2 = KernelTraits<MASK_MODE, CTA_TILE_Q_2, NUM_MMA_Q_2, NUM_MMA_KV_2, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_2, NUM_WARPS_KV_2, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;

  // Attention state reduction kernel
  constexpr uint32_t NUM_THREADS =
      KTraits1::NUM_THREADS > KTraits2::NUM_THREADS ? KTraits1::NUM_THREADS : KTraits2::NUM_THREADS;
  using ReductionKTraits =
      StateReductionKernelTraits<HEAD_DIM_VO, 4, NUM_THREADS, DTypeO, DTypeO, IdType>;
  size_t smem_size =
      max(sizeof(typename KTraits1::SharedStorage), sizeof(typename KTraits2::SharedStorage));
  smem_size = max(smem_size, ReductionKTraits::SMEM_SIZE);
  // Launch persistent kernel
  auto kernel =
      PersistentKernelTemplate<BlockBatchPagedAttentionPersistent<KTraits1, Params>,
                               BlockBatchPagedAttentionPersistent<KTraits2, Params>,
                               BlockBatchReductionPersistent<ReductionKTraits>, FlippedSchedule>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(NUM_THREADS);

  // CTA counter for prefill-decode overlap
  int num_sm = 0;
  int num_ctas_per_sm = 0;
  static int* cta_counter = nullptr;
  if constexpr (FlippedSchedule) {
    if (cta_counter == nullptr) {
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas_per_sm, kernel,
                                                                         NUM_THREADS, smem_size));
      FLASHINFER_CUDA_CALL(
          cudaMallocAsync(&cta_counter, sizeof(int) * num_sm * num_ctas_per_sm, stream));
    }
  }

  void* args[] = {(void*)&params_1, (void*)&params_2, (void*)&cta_counter};

  FLASHINFER_CUDA_CALL(
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

};  // namespace flashinfer

#endif  // FLASHINFER_PERSISTENT_CUH_
