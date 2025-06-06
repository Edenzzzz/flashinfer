/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass_extensions/gemm/threadblock/dq_mma_base.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Iterators over scales in global memory
    typename IteratorScale_,
    /// Iterators over scales in shared memory
    typename SmemIteratorScale_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Layout of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Converter for B matrix applied immediately after the LDS
    typename TransformBAfterLDS_,
    /// The quantization operator being used
    WeightOnlyQuantOp QuantOp_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
class DqMmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_,
                      CacheOpB, IteratorScale_, SmemIteratorScale_, ElementC_, LayoutC_, Policy_,
                      Stages, TransformBAfterLDS_, QuantOp_, SharedMemoryClear,
                      std::enable_if_t<isFinegrained(QuantOp_)>>
    : public DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages, QuantOp_> {
 public:
  ///< Base class
  using Base = DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages, QuantOp_>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using IteratorScale = IteratorScale_;
  using ElementScale = typename IteratorScale::Element;
  using LayoutScale = typename IteratorScale::Layout;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorScale = SmemIteratorScale_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using TransformBAfterLDS = TransformBAfterLDS_;

  static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  using Dequantizer = warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, Operand::kB,
                                                   ElementScale, LayoutScale, 32, QuantOp>;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static_assert(Base::SharedStorage::ShapeScale::kRow == Stages, "");
  static_assert(Base::SharedStorage::ShapeScale::kColumn == Shape::kN, "");

  /// Internal structure exposed for introspection.
  struct Detail {
    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA = IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB = IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;
  Dequantizer warp_dequantizer_;

  using ElementA = typename IteratorA::Element;
  using ElementB = typename IteratorB::Element;
  using LayoutDetailsForB = kernel::LayoutDetailsB<ElementA, ElementB, ArchTag>;

  static constexpr bool RequiresTileInterleave =
      layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;
  static_assert(!RequiresTileInterleave ||
                    (RequiresTileInterleave && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
                "Layout K must match threadblockK");

 private:
  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Iterator to write threadblock-scoped tile of scale and zero operand to shared memory
  SmemIteratorScale smem_iterator_scale_;

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  DqMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage& shared_storage,
      /// The group size for quantization
      int const group_size,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        warp_dequantizer_(
            {shared_storage.operand_scale.data(), LayoutScale(Shape::kN)},
            {shared_storage.operand_zero.data(), LayoutScale(Shape::kN)},
            (warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN)) / Base::WarpCount::kM,
            lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_iterator_scale_(LayoutScale(Shape::kN), shared_storage.operand_scale.data(),
                             shared_storage.operand_zero.data(), {Base::kStages, Shape::kN},
                             thread_idx, group_size) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterationsForB * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void copy_scales_and_advance(IteratorScale& iterator_scale, int stage = -1, int k_iter = -1) {
    static_assert(IteratorScale::Shape::kRow == 1, "Scale stride must be 1.");

    typename IteratorScale::AccessType* gmem_scale_ptr = iterator_scale.get_scale();
    typename IteratorScale::AccessType* gmem_zero_ptr = iterator_scale.get_zero();

    typename IteratorScale::AccessType* smem_scale_ptr =
        reinterpret_cast<typename IteratorScale::AccessType*>(
            this->smem_iterator_scale_.get_scale());
    typename IteratorScale::AccessType* smem_zero_ptr =
        reinterpret_cast<typename IteratorScale::AccessType*>(
            this->smem_iterator_scale_.get_zero());

    int const kSrcBytes =
        sizeof_bits<typename IteratorScale::Element>::value * IteratorScale::kAlignment / 8;

    cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(smem_scale_ptr, gmem_scale_ptr,
                                                  iterator_scale.valid());

    if (gmem_zero_ptr != nullptr) {
      cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(smem_zero_ptr, gmem_zero_ptr,
                                                    iterator_scale.valid());
    }

    if (iterator_scale.group_size_ == 64) {
      iterator_scale.add_tile_offset({1, 0});
    } else if (iterator_scale.group_size_ == 128) {
      if constexpr (Shape::kK == 128) {
        iterator_scale.add_tile_offset({1, 0});
      } else if constexpr (Shape::kK == 64) {
        if (iterator_scale.row_groupsize64_ & 0x1) {
          iterator_scale.add_tile_offset({1, 0});
        }
      } else {
        static_assert(Shape::kK == 0, "Unsupported k tile shape, can only be 64 or 128");
      }
    }

    iterator_scale.row_groupsize64_++;

    this->smem_iterator_scale_.add_tile_offset({1, 0});
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA& iterator_A, IteratorB& iterator_B, int group_start_A = 0,
                              int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr,
                                                                iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr,
                                                          iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr,
                                                                iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr,
                                                          iterator_B.valid());
          }

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC& accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< iterator over scale operand in global memory
      IteratorScale iterator_scale,
      ///< initial value of accumulator
      FragmentC const& src_accum) {
    //
    // Prologue
    //

    TransformBAfterLDS lds_converter;

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);
      iterator_scale.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, iterator_A.get(),
                                                              iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, iterator_B.get(),
                                                              iterator_B.valid());

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }

      copy_scales_and_advance(iterator_scale, stage, gemm_k_iterations);

      // Move to the next stage
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {
      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages -
    // 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    typename Dequantizer::FragmentScale warp_frag_scales;
    typename Dequantizer::FragmentZero warp_frag_zeros;

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    warp_dequantizer_.load(warp_frag_scales, warp_frag_zeros);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    warp_dequantizer_.add_pointer_offset(Shape::kN);

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);
    iterator_scale.clear_mask(gemm_k_iterations == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_A_;

        int const warp_tileB_k_compute_offset = warp_mma_k % Base::kNumKIterationsPerWarpBLoad;
        int const warp_tileB_k_load_offset = warp_mma_k / Base::kNumKIterationsPerWarpBLoad;
        if (warp_tileB_k_compute_offset == Base::kNumKIterationsPerWarpBLoad - 1) {
          this->warp_tile_iterator_B_.set_kgroup_index((warp_tileB_k_load_offset + 1) %
                                                       Base::kWarpGemmIterationsForB);
          this->warp_tile_iterator_B_.load(warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);
          ++this->warp_tile_iterator_B_;
        }

        typename TransformBAfterLDS::result_type converted_frag_B =
            lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2]);
        warp_dequantizer_.dequantize(converted_frag_B, warp_frag_scales, warp_frag_zeros);

        using FragmentOperandB = cutlass::Array<ElementA, Operator::FragmentB::kElements>;
        constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
        constexpr int ConversionVectorWidth = TransformBAfterLDS::result_type::kElements;
        static_assert(ConversionVectorWidth == FragmentOperandB::kElements);

        using Converter = cutlass::NumericArrayConverter<ElementA, ElementScale,
                                                         ConversionVectorWidth, RoundStyle>;

        FragmentOperandB converted_frag_B_operand = Converter::convert(converted_frag_B);
        warp_mma(accum, warp_frag_A[warp_mma_k % 2], converted_frag_B_operand, accum,
                 warp_tileB_k_compute_offset);

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations - 1) {
          int group_start_iteration_A, group_start_iteration_B;

          group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
          group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                                 group_start_iteration_B);

          // This is the first group of a given stage, so we issue the loads for the B scales
          // immediately.
          if (group_start_iteration_B == 0) {
            copy_scales_and_advance(iterator_scale);
          }
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          int group_start_iteration_A, group_start_iteration_B;
          group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                                 group_start_iteration_B);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Wait until we have at least one committed global fetch stage. (#uncommitted =
          // Base::kStages - 1 - #committed)
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A.add_tile_offset({0, 1});
          iterator_B.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            this->smem_iterator_scale_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterationsForB, 0});
            warp_dequantizer_.add_pointer_offset(-Base::kStages * Shape::kN);
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
          iterator_scale.clear_mask(gemm_k_iterations == 0);
        }
      }

      // Load the scale needed for the next tile iteration.
      warp_dequantizer_.load(warp_frag_scales, warp_frag_zeros);
      // Update internal pointer to set of scales in shared memory.
      warp_dequantizer_.add_pointer_offset(Shape::kN);
    }

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
