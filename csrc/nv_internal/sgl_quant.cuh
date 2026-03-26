/*
 * Generalized SGL-style quantization kernel supporting MXFP8, NVFP4, and MXFP4.
 * Uses the same CuTe-based G2R/R2G/R2S copy pipeline as the original SGL mxfp8 kernel.
 * Reuses FI's conversion math from quantization_utils.cuh.
 */

#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/ptx>

#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

namespace sgl_quant {

using namespace cute;
using tensorrt_llm::BlockScaleQuantizationType;

// ---------------------------------------------------------------------------
// Conversion functors: operate on CuTe fragments, same interface as
// cvt_warp_fp16_to_mxfp8 from sgl_mxfp8_quant.cuh
// ---------------------------------------------------------------------------

// NVFP4: 16 input half/bf16 → 8 output uint8_t (packed fp4 pairs) + 1 e4m3 SF
template <typename FragmentS, typename FragmentD>
__inline__ __device__ uint8_t cvt_warp_fp16_to_nvfp4(FragmentS& fragment_s, FragmentD& fragment_d,
                                                     float SFScaleVal) {
  using namespace tensorrt_llm::kernels;
  constexpr int eles_per_thr = 16;
  using ValType = typename FragmentS::element_type;
  using VecType =
      std::conditional_t<std::is_same_v<ValType, __nv_bfloat16>, __nv_bfloat162, __half2>;

  // Pack into vec pairs for max computation
  VecType vec[8];
  vec[0].x = fragment_s(cute::Int<0>{});
  vec[0].y = fragment_s(cute::Int<1>{});
  vec[1].x = fragment_s(cute::Int<2>{});
  vec[1].y = fragment_s(cute::Int<3>{});
  vec[2].x = fragment_s(cute::Int<4>{});
  vec[2].y = fragment_s(cute::Int<5>{});
  vec[3].x = fragment_s(cute::Int<6>{});
  vec[3].y = fragment_s(cute::Int<7>{});
  vec[4].x = fragment_s(cute::Int<8>{});
  vec[4].y = fragment_s(cute::Int<9>{});
  vec[5].x = fragment_s(cute::Int<10>{});
  vec[5].y = fragment_s(cute::Int<11>{});
  vec[6].x = fragment_s(cute::Int<12>{});
  vec[6].y = fragment_s(cute::Int<13>{});
  vec[7].x = fragment_s(cute::Int<14>{});
  vec[7].y = fragment_s(cute::Int<15>{});

  // Local max (no cross-thread for SF_VEC_SIZE=16)
  auto local_max = __habs2(vec[0]);
  for (int i = 1; i < eles_per_thr / 2; i++) {
    local_max = __hmax2(__habs2(vec[i]), local_max);
  }

  float block_max(0.0f);
  if constexpr (std::is_same_v<ValType, __nv_bfloat16>) {
    block_max = __bfloat162float(__hmax(local_max.x, local_max.y));
  } else {
    block_max = __half2float(__hmax(local_max.x, local_max.y));
  }

  // Scale factor: e4m3 with global scale (same as FI's cvt_warp_fp16_to_fp4 UE8M0=false)
  float SFValue = SFScaleVal * (block_max * reciprocal_approximate_ftz(6.0f));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  uint8_t sf_out = tmp.__x;
  SFValue = static_cast<float>(tmp);
  float output_scale =
      block_max != 0.f
          ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
          : 0.0f;

  // Convert to float, scale, and pack to e2m1
  float2 fp2_vals[8];
#pragma unroll
  for (int i = 0; i < eles_per_thr / 2; i++) {
    if constexpr (std::is_same_v<ValType, __half>) {
      fp2_vals[i] = __half22float2(vec[i]);
    } else {
      fp2_vals[i] = __bfloat1622float2(vec[i]);
    }
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }

  // Convert to e2m1 (16 fp4 values → 8 bytes)
  uint64_t e2m1_val = fp32_vec_to_e2m1(fp2_vals);

  // Store into output fragment (8 uint8_t elements)
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_val);
  fragment_d(cute::Int<0>{}) = bytes[0];
  fragment_d(cute::Int<1>{}) = bytes[1];
  fragment_d(cute::Int<2>{}) = bytes[2];
  fragment_d(cute::Int<3>{}) = bytes[3];
  fragment_d(cute::Int<4>{}) = bytes[4];
  fragment_d(cute::Int<5>{}) = bytes[5];
  fragment_d(cute::Int<6>{}) = bytes[6];
  fragment_d(cute::Int<7>{}) = bytes[7];

  return sf_out;
}

// MXFP4: 16 input half/bf16 → 8 output uint8_t (packed fp4 pairs) + 1 e8m0 SF
template <typename FragmentS, typename FragmentD>
__inline__ __device__ uint8_t cvt_warp_fp16_to_mxfp4(FragmentS& fragment_s, FragmentD& fragment_d) {
  using namespace tensorrt_llm::kernels;
  constexpr int eles_per_thr = 16;
  using ValType = typename FragmentS::element_type;
  using VecType =
      std::conditional_t<std::is_same_v<ValType, __nv_bfloat16>, __nv_bfloat162, __half2>;

  VecType vec[8];
  vec[0].x = fragment_s(cute::Int<0>{});
  vec[0].y = fragment_s(cute::Int<1>{});
  vec[1].x = fragment_s(cute::Int<2>{});
  vec[1].y = fragment_s(cute::Int<3>{});
  vec[2].x = fragment_s(cute::Int<4>{});
  vec[2].y = fragment_s(cute::Int<5>{});
  vec[3].x = fragment_s(cute::Int<6>{});
  vec[3].y = fragment_s(cute::Int<7>{});
  vec[4].x = fragment_s(cute::Int<8>{});
  vec[4].y = fragment_s(cute::Int<9>{});
  vec[5].x = fragment_s(cute::Int<10>{});
  vec[5].y = fragment_s(cute::Int<11>{});
  vec[6].x = fragment_s(cute::Int<12>{});
  vec[6].y = fragment_s(cute::Int<13>{});
  vec[7].x = fragment_s(cute::Int<14>{});
  vec[7].y = fragment_s(cute::Int<15>{});

  // Cross-thread max for SF_VEC_SIZE=32
  auto local_max = __habs2(vec[0]);
  for (int i = 1; i < eles_per_thr / 2; i++) {
    local_max = __hmax2(__habs2(vec[i]), local_max);
  }
  local_max = __hmax2(__shfl_xor_sync(uint32_t(-1), local_max, 1), local_max);

  float block_max(0.0f);
  if constexpr (std::is_same_v<ValType, __nv_bfloat16>) {
    block_max = __bfloat162float(__hmax(local_max.x, local_max.y));
  } else {
    block_max = __half2float(__hmax(local_max.x, local_max.y));
  }

  // Scale factor: e8m0 format
  float vec_max = block_max * reciprocal_approximate_ftz(6.0f);
  __nv_fp8_e8m0 tmp_sf;
  tmp_sf.__x = __nv_cvt_float_to_e8m0(vec_max, __NV_SATFINITE, cudaRoundPosInf);
  uint8_t sf_out = tmp_sf.__x;
  float output_scale = block_max != 0.f ? exp2f_rcp(sf_out) : 0.0f;

  float2 fp2_vals[8];
#pragma unroll
  for (int i = 0; i < eles_per_thr / 2; i++) {
    if constexpr (std::is_same_v<ValType, __half>) {
      fp2_vals[i] = __half22float2(vec[i]);
    } else {
      fp2_vals[i] = __bfloat1622float2(vec[i]);
    }
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }

  uint64_t e2m1_val = fp32_vec_to_e2m1(fp2_vals);
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_val);
  fragment_d(cute::Int<0>{}) = bytes[0];
  fragment_d(cute::Int<1>{}) = bytes[1];
  fragment_d(cute::Int<2>{}) = bytes[2];
  fragment_d(cute::Int<3>{}) = bytes[3];
  fragment_d(cute::Int<4>{}) = bytes[4];
  fragment_d(cute::Int<5>{}) = bytes[5];
  fragment_d(cute::Int<6>{}) = bytes[6];
  fragment_d(cute::Int<7>{}) = bytes[7];

  return sf_out;
}

// ---------------------------------------------------------------------------
// Fast tile function: accumulates SFs in registers across all subtiles,
// then writes to smem once. Only 3 syncs per tile instead of 18.
// Works for both THREADS_PER_SF=1 (NVFP4) and THREADS_PER_SF=2 (MXFP4).
// ---------------------------------------------------------------------------
template <typename ConvertFn, typename TensorS, typename TensorP, typename TensorD,
          typename TensorSharedSF, typename TensorSF, typename TiledCopyG2R, typename TiledCopyR2G,
          typename TiledCopyR2S, int THREADS_PER_SF>
__inline__ __device__ void sgl_quant_tile(TensorS& tensor_s, TensorP& tensor_p, TensorD& tensor_d,
                                          TensorSharedSF& tensor_shared_sf, TensorSF& tensor_sf,
                                          int m, TiledCopyG2R& tiled_copy_g2r,
                                          TiledCopyR2G& tiled_copy_r2g,
                                          TiledCopyR2S& tiled_copy_r2s, ConvertFn convert_fn) {
  using Tiler_MN_G2R = typename TiledCopyG2R::Tiler_MN;
  auto tiler_mn_g2r = Tiler_MN_G2R{};

  auto tiled_tensor_s = tiled_divide(tensor_s, tiler_mn_g2r);
  auto tiled_tensor_p = tiled_divide(tensor_p, tiler_mn_g2r);
  auto tiled_tensor_d = tiled_divide(tensor_d, typename TiledCopyR2G::Tiler_MN{});

  using SF_Tiler_MN = typename TiledCopyR2S::Tiler_MN;
  auto tiled_tensor_shared_sf = tiled_divide(tensor_shared_sf, SF_Tiler_MN{});
  auto tiled_tensor_sf = tiled_divide(tensor_sf, SF_Tiler_MN{});
  auto squeeze_tiled_tensor_sf = take<0, 2>(tiled_tensor_sf);
  auto squeeze_tiled_tensor_shared_sf = take<0, 2>(tiled_tensor_shared_sf);

  constexpr int tile_loop_count = size<1>(tiled_tensor_s);
  constexpr int rows_in_tile = size<0>(tiler_mn_g2r);

  // Accumulate SFs in registers across all subtiles (max 8 subtiles)
  uint8_t sf_regs[tile_loop_count];

  // Phase 1: Load → Convert → Store output, accumulate SFs (0 syncs)
#pragma unroll
  for (int t = 0; t < tile_loop_count; t++) {
    if (t * rows_in_tile >= m) {
      sf_regs[t] = 0;
      continue;
    }

    auto current_s = tensor<0>(take<0, 2>(tiled_tensor_s)(_, t));
    auto current_p = tensor<0>(take<0, 2>(tiled_tensor_p)(_, t));
    auto current_d = tensor<0>(take<0, 2>(tiled_tensor_d)(_, t));

    auto thr_g2r = tiled_copy_g2r.get_thread_slice(threadIdx.x);
    auto thr_s = thr_g2r.partition_S(current_s);
    auto thr_p_g2r = thr_g2r.partition_S(current_p);
    auto input_frag = make_fragment_like(thr_s);

    auto thr_r2g = tiled_copy_r2g.get_thread_slice(threadIdx.x);
    auto thr_d = thr_r2g.partition_D(current_d);
    auto output_frag = make_fragment_like(thr_d);

    copy_if(tiled_copy_g2r, thr_p_g2r, thr_s, input_frag);
    sf_regs[t] = convert_fn(input_frag, output_frag);
    copy(tiled_copy_r2g, output_frag, thr_d);
  }

  // Phase 2: Wait for previous tile's TMA to finish, then write all SFs to smem
  if (threadIdx.x == 0) {
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
  }
  __syncthreads();  // sync 1: ensure TMA done

#pragma unroll
  for (int t = 0; t < tile_loop_count; t++) {
    auto current_shared_sf = tensor<0>(squeeze_tiled_tensor_shared_sf(_, t));
    auto thr_r2s = tiled_copy_r2s.get_thread_slice(threadIdx.x / THREADS_PER_SF);
    auto thr_shared_sf = thr_r2s.partition_D(current_shared_sf);
    auto sf_frag = make_fragment_like(thr_shared_sf);
    sf_frag[0] = sf_regs[t];
    if (threadIdx.x % THREADS_PER_SF == 0) {
      copy(tiled_copy_r2s, sf_frag, thr_shared_sf);
    }
  }

  // Phase 3: TMA bulk copy SFs from shared to global
  __syncthreads();  // sync 2: ensure smem writes visible
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

  constexpr int sf_smem_bytes = size(typename TensorSharedSF::layout_type{});
  if (threadIdx.x == 0) {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                             squeeze_tiled_tensor_sf.data().get(),
                             squeeze_tiled_tensor_shared_sf.data().get(), sf_smem_bytes);
    cuda::ptx::cp_async_bulk_commit_group();
  }
  // No final sync needed - next tile's phase 2 will wait for this TMA
}

// ---------------------------------------------------------------------------
// NVFP4 single-matrix kernel using full CuTe copies
// ---------------------------------------------------------------------------
template <typename T_IN, bool UE8M0_SF, typename TiledCopyG2R, typename TiledCopyR2G,
          typename TiledCopyR2S>
__global__ void nvfp4_single_quant_kernel(const T_IN* input, int m, int k, uint8_t* quant_output,
                                          uint8_t* scale_factor, float SFScaleVal,
                                          TiledCopyG2R tiled_copy_g2r, TiledCopyR2G tiled_copy_r2g,
                                          TiledCopyR2S tiled_copy_r2s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // SF_VEC_SIZE=16 for NVFP4: 128 cols / 16 = 8 SFs per row
  constexpr int SF_PER_ROW = 8;
  constexpr int SF_SMEM_BYTES = SGL_BLOCK_M * SF_PER_ROW;  // 1024
  __shared__ __align__(512) uint8_t shared_memory[SF_SMEM_BYTES];

  // Scale factor tile layout for SF_VEC_SIZE=16: (128, 8)
  // Use simple linear layout for shared memory
  auto sf_shared_layout = make_layout(make_shape(SGL_BLOCK_M, SF_PER_ROW), LayoutRight{});
  auto scale_factor_shared = make_tensor(make_smem_ptr(shared_memory), sf_shared_layout);

  // Input tensor: (m, k) of T_IN
  auto input_tensor =
      make_tensor(make_gmem_ptr(input), make_layout(make_shape(m, k), LayoutRight{}));

  // Output tensor: (m, k/2) of uint8_t (packed fp4)
  auto output_tensor =
      make_tensor(make_gmem_ptr(quant_output), make_layout(make_shape(m, k / 2), LayoutRight{}));

  // Scale factor tensor: (aligned_m, k/16) of uint8_t, linear layout
  int aligned_m = (m + 127) / 128 * 128;
  int sf_cols = k / 16;
  auto sf_tensor = make_tensor(make_gmem_ptr(scale_factor),
                               make_layout(make_shape(aligned_m, sf_cols), LayoutRight{}));

  auto input_shape = shape(input_tensor);
  auto identity_tensor = make_identity_tensor(input_shape);
  auto predict_tensor =
      cute::lazy::transform(identity_tensor, [&](auto c) { return elem_less(c, input_shape); });

  auto tiler_in = make_shape(cute::Int<SGL_BLOCK_M>{}, cute::Int<SGL_BLOCK_K>{});
  auto tiler_out = make_shape(cute::Int<SGL_BLOCK_M>{}, cute::Int<SGL_BLOCK_K / 2>{});
  auto tiler_sf = make_shape(cute::Int<SGL_BLOCK_M>{}, cute::Int<SF_PER_ROW>{});

  auto tiled_input = zipped_divide(input_tensor, tiler_in);
  auto tiled_output = zipped_divide(output_tensor, tiler_out);
  auto tiled_predict = zipped_divide(predict_tensor, tiler_in);
  auto tiled_sf = zipped_divide(sf_tensor, tiler_sf);

  auto total_tiles = size<1>(tiled_input);

  // Conversion lambda
  auto convert_fn = [&](auto& input_frag, auto& output_frag) -> uint8_t {
    if constexpr (UE8M0_SF) {
      return cvt_warp_fp16_to_mxfp4(input_frag, output_frag);
    } else {
      return cvt_warp_fp16_to_nvfp4(input_frag, output_frag, SFScaleVal);
    }
  };

  for (auto blk = blockIdx.x; blk < total_tiles; blk += gridDim.x) {
    auto current_in = tensor<0>(tiled_input(_, blk));
    auto current_out = tensor<0>(tiled_output(_, blk));
    auto current_pred = tensor<0>(tiled_predict(_, blk));
    auto current_sf = tensor<0>(tiled_sf(_, blk));

    constexpr int THREADS_PER_SF = UE8M0_SF ? 2 : 1;
    sgl_quant_tile<decltype(convert_fn), decltype(current_in), decltype(current_pred),
                   decltype(current_out), decltype(scale_factor_shared), decltype(current_sf),
                   decltype(tiled_copy_g2r), decltype(tiled_copy_r2g), decltype(tiled_copy_r2s),
                   THREADS_PER_SF>(current_in, current_pred, current_out, scale_factor_shared,
                                   current_sf, m, tiled_copy_g2r, tiled_copy_r2g, tiled_copy_r2s,
                                   convert_fn);
  }
#endif
}

// ---------------------------------------------------------------------------
// Launch: NVFP4/MXFP4
// ---------------------------------------------------------------------------
template <typename T_IN, bool UE8M0_SF>
void invokeNvfp4QuantizationSGL(int m, int k, const T_IN* input, uint8_t* quant_output,
                                uint8_t* scale_factor, float SFScaleVal, int multiProcessorCount,
                                cudaStream_t stream) {
  // G2R: 16 input elements (T_IN) per thread, 256-bit loads
  using G2RThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using G2RValLayout = Layout<Shape<_1, _16>>;
  G2RThrLayout g2r_thr{};
  G2RValLayout g2r_val{};
  using G2RCopyOp = UniversalCopy<cutlass::AlignedArray<T_IN, size(g2r_val)>>;
  using G2RCopyAtom = cute::Copy_Atom<G2RCopyOp, T_IN>;
  auto tiled_copy_g2r = cute::make_tiled_copy(G2RCopyAtom{}, g2r_thr, g2r_val);

  // R2G: 8 output bytes (uint8_t) per thread, 64-bit stores
  // FP4: 16 input elements → 8 output bytes (16 fp4 values packed)
  using R2GThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using R2GValLayout = Layout<Shape<_1, _8>>;
  R2GThrLayout r2g_thr{};
  R2GValLayout r2g_val{};
  using R2GCopyOp = UniversalCopy<cutlass::AlignedArray<uint8_t, size(r2g_val)>>;
  using R2GCopyAtom = cute::Copy_Atom<R2GCopyOp, uint8_t>;
  auto tiled_copy_r2g = cute::make_tiled_copy(R2GCopyAtom{}, r2g_thr, r2g_val);

  // R2S: scale factors
  constexpr int SF_PER_ROW = UE8M0_SF ? 4 : 8;  // SF_VEC_SIZE 32 or 16
  constexpr int THREADS_PER_SF = UE8M0_SF ? 2 : 1;
  // For SF_VEC_SIZE=16: 128 threads, each writes 1 SF → (16, 8) thr layout
  // For SF_VEC_SIZE=32: 64 threads, each writes 1 SF → (16, 4) thr layout
  using R2SThrLayout = std::conditional_t<UE8M0_SF, Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                          Layout<Shape<_16, _8>, Stride<_8, _1>>>;
  using R2SValLayout = Layout<Shape<_1, _1>>;
  R2SThrLayout r2s_thr{};
  R2SValLayout r2s_val{};
  using R2SCopyOp = UniversalCopy<cutlass::AlignedArray<uint8_t, size(r2s_val)>>;
  using R2SCopyAtom = cute::Copy_Atom<R2SCopyOp, uint8_t>;
  auto tiled_copy_r2s = cute::make_tiled_copy(R2SCopyAtom{}, r2s_thr, r2s_val);

  static int max_active_blocks_per_sm = -1;
  if (max_active_blocks_per_sm < 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm,
        nvfp4_single_quant_kernel<T_IN, UE8M0_SF, decltype(tiled_copy_g2r),
                                  decltype(tiled_copy_r2g), decltype(tiled_copy_r2s)>,
        SGL_THREAD_BLOCK_SIZE, 0);
  }

  dim3 grid(multiProcessorCount * max_active_blocks_per_sm, 1, 1);
  dim3 block(SGL_THREAD_BLOCK_SIZE, 1, 1);

  nvfp4_single_quant_kernel<T_IN, UE8M0_SF>
      <<<grid, block, 0, stream>>>(input, m, k, quant_output, scale_factor, SFScaleVal,
                                   tiled_copy_g2r, tiled_copy_r2g, tiled_copy_r2s);
}

}  // namespace sgl_quant
