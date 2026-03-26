# NVFP4 quantize kernel: TMA vs non-TMA

## Implementations

Both live in the TensorRT-LLM–derived code and are used by the Python FP4 quantization APIs (SM100+).

**APIs that use these kernels:**

- **`fp4_quantize`** – Main API; can use TMA or non-TMA depending on shape and `sf_vec_size`.
- **`nvfp4_quantize`** – Wrapper around `fp4_quantize` with NVFP4 options (`sf_vec_size=16`). Can use the TMA path when `m >= 1024` and `k % 512 == 0`.
- **`mxfp4_quantize`** – Wrapper around `fp4_quantize` with MXFP4 options (`sf_vec_size=32`, `sf_use_ue8m0=True`). Always uses the **non-TMA** path (TMA is only enabled for `sf_vec_size == 16`).
- **`mxfp8_quantize`** – Separate API in `fp8_quantization.py`; calls `invokeMxFP8Quantization` in C++ (fixed `sf_vec_size=32`, FP16→MXFP8). Uses the **non-TMA** kernel only: `quantize_with_block_size<FP16_TO_MXFP8, ...>` from `quantization.cuh`. TMA is not implemented for MXFP8 (see comment in `quantization.cu`: “TMA quantization for MXFP8 is not supported yet because of SF_VEC_SIZE = 32”).

| | Non-TMA | TMA |
|--|--------|-----|
| **Kernel** | `quantize_with_block_size` | `quantize_with_block_size_tma` |
| **File** | `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh` | same |
| **Goal** | General-purpose | High throughput (large m, aligned n) |
| **Input load** | Direct global loads (packed vec per thread) | TMA 3D copy into shared memory |
| **Block** | Up to 512 threads, row-persistent | 288 threads (1 producer warp + 8 consumer warps) |

**Dispatch** (in `invokeFP4Quantization`, `csrc/nv_internal/cpp/kernels/quantization.cu`):

- **TMA** when: `sf_vec_size == 16` and `m >= 1024` and `n % 512 == 0`.
- **Non-TMA** otherwise (smaller `m`, or `sf_vec_size != 16`, or `n` not a multiple of 512).

### TMA kernel details

The TMA path aims for **high throughput** on large matrices by:

- **Producer–consumer pipeline:** One producer warp (warp 0) issues TMA copies from global memory into shared memory; 8 consumer warps (warps 1–8) read from shared memory, run the FP4 conversion, and write quantized output. They synchronize via `full_barriers` and `empty_barriers` with a **multi-stage** software pipeline (`NUM_STAGES` = 4 for half/bf16, 6 for FP8).
- **TMA load (not 256-bit LDG):** Input is loaded with **TMA** (`cute::SM90_TMA_LOAD_3D::copy`) using a `CUtensorMap`. For half/bf16 the descriptor uses `SWIZZLE_128B`, so TMA does bulk async 128-byte copies into shared memory. Consumers then read from shared memory using **128-bit (float4)** loads; the FP4 output is written to global with **64-bit** stores. So there are no 256-bit load/store instructions; the high throughput comes from TMA’s bulk copy and the producer–consumer overlap.
- **Tiles:** `TMA_ROW_TILE` = 16 (half/bf16) or 8 (FP8), `TMA_COL_TILE` = 64 elements (128 B) or 128 elements (128 B). Shared memory is 128-byte aligned for TMA + swizzle.

## Benchmarking

Use the `nvfp4_quantize` routine:

```bash
python benchmarks/flashinfer_benchmark.py --routine nvfp4_quantize --m <M> --k <K> \
  --input_dtype bfloat16 --global_scale 1.0 --sf_layout 128x4 -vv
```

- **TMA path:** e.g. `--m 1024 --k 4096` or `--m 2048 --k 8192` (m ≥ 1024, k multiple of 512).
- **Non-TMA path:** e.g. `--m 512 --k 4096` (m < 1024) or `--m 1024 --k 1024` (k % 512 ≠ 0).

Routine implementation: `benchmarks/routines/quantization.py` → `testNvfp4Quantize`. It reports median time and TB/s; no separate “TMA” vs “non-TMA” labels—compare by choosing shapes that trigger each path.

## TODO

Flashinfer already supports TMA and non-TMA for nvfp4_quantize. However, I suspect even the current TMA implementation is suboptimal: it uses a warp as TMA producer and only 64-bit stores and 128-bit loads, instead of the 256-bit load available on B200. SGLang has another quant implementation that uses 256-bit load and 128-bit store, and eliminates the producer warp. However, currently it only supports mxfp8, not mxfp4 and nvfp4. My understanding is that it only takes changing some dtype template params and the cvt util (cvt_warp_fp16_to_mxfp8 to cvt_fp16_to_fp4_expert, etc.). So I want to port this kernel to flashinfer, test precision and speed comprehensively and see if it surpasses both impl in flashinfer. Steps:

1. **Baseline benchmark**: Compare SGLang's mxfp8_group_quant (`/sgl-workspace/sglang/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh`) against FlashInfer's current `mxfp8_quantize` on shapes `(m, m)` for `m` in `[512, 1024, 2048, 4096, 8192, 16384, 32768]`. Verify the SGL kernel is actually faster before proceeding.
2. Port the mxfp8_group_quant from `/sgl-workspace/sglang/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh`, and add `launchFP4QuantizationSGL` in `csrc/nv_internal/cpp/kernels/quantization.cu`. Refer to `.claude/skills/add-cuda-kernel/SKILL.md` on how to add bindings.
3. Add python bindings to SGL impl (see **APIs that use these kernels:**), enabled via a `use_sgl` flag.
4. Templatize the SGL-style kernel into a general quant function supporting mxfp8, mxfp4, and nvfp4 — like the FI original impl. Reference the FI kernel (`quantization.cuh`) and reuse its quant conversion utilities (`cvt_fp16_to_fp4_expert`, etc.) instead of the mxfp8-only `cvt_warp_fp16_to_mxfp8`. The conversion function and output type should be template parameters so the same 256-bit-load / no-producer-warp tile loop serves all three formats.
5. Add comprehensive precision and benchmark testing in `benchmarks/routines/quantization.py` (see **Benchmarking**). Test nvfp4 specifically. Test shapes `(m, m)` for `m` in `[512, 1024, 2048, 4096, 8192, 16384, 32768]` plus 4D-reshaped shapes `(32760, 1536)` from `[1, 32760, 12, 128]` and `(16384, 1536)` from `[1, 16384, 12, 128]`. Make sure the SGL impl matches flashinfer precision and compare speedup on these shapes.
