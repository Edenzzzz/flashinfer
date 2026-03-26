"""
Benchmark: FlashInfer mxfp8_quantize (original vs SGL-ported) vs SGLang mxfp8_group_quant

Compares speed and precision on various shapes including:
- Square shapes (m, m) for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]
- LLM-style shapes: (32760, 1536), (16384, 1536) from [1, 32760, 12, 128], [1, 16384, 12, 128]
"""

import torch
import numpy as np

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

try:
    from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


def setup_sgl_inputs(m, k, input_dtype, device):
    """Prepare SGL kernel inputs for a single-expert (m, k) problem."""
    input_tensor = torch.randn(m, k, dtype=input_dtype, device=device)
    problem_sizes = torch.tensor([[m, k, k]], dtype=torch.int32, device=device)
    expert_offsets = torch.tensor([0], dtype=torch.int32, device=device)
    blockscale_offsets = torch.tensor([0], dtype=torch.int32, device=device)
    aligned_m = ((m + 127) // 128) * 128
    quant_output = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=device)
    scale_factor = torch.zeros(aligned_m, k // 32, dtype=torch.uint8, device=device)
    return (
        input_tensor,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )


def compare_quant_values(q1, q2, m, k):
    """Compare quantized values, return match percentage."""
    a = q1[:m, :k].contiguous().view(torch.uint8).flatten()
    b = q2[:m, :k].contiguous().view(torch.uint8).flatten()
    return 100.0 * (a == b).sum().item() / a.numel()


def main():
    device = "cuda"
    input_dtype = torch.bfloat16
    shapes_mk = [(m, m) for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]]
    shapes_mk += [(32760, 1536), (16384, 1536)]
    num_iters = 50
    dry_run = 10

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input dtype: {input_dtype}")
    print(f"Iterations: {num_iters} (dry run: {dry_run})")
    print()

    # Header
    cols = ["Shape", "FI orig (us)", "FI+SGL (us)", "Speedup"]
    if HAS_SGL_KERNEL:
        cols += ["SGL ext (us)", "Q match%"]
    header = " | ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))

    for m, k in shapes_mk:
        torch.cuda.empty_cache()

        # Shared input
        fi_input = torch.randn(m, k, dtype=input_dtype, device=device)

        # --- FlashInfer original ---
        fi_times = bench_gpu_time(
            fn=flashinfer.mxfp8_quantize,
            input_args=(fi_input,),
            input_kwargs={"is_sf_swizzled_layout": False},
            dry_run_iters=dry_run,
            repeat_iters=num_iters,
            enable_cupti=True,
        )
        fi_median = np.median(fi_times)

        # --- FlashInfer + SGL ported ---
        sgl_ported_times = bench_gpu_time(
            fn=flashinfer.mxfp8_quantize,
            input_args=(fi_input,),
            input_kwargs={"use_sgl": True},
            dry_run_iters=dry_run,
            repeat_iters=num_iters,
            enable_cupti=True,
        )
        sgl_ported_median = np.median(sgl_ported_times)

        speedup = fi_median / sgl_ported_median
        shape_str = f"({m},{k})"
        row = f"{shape_str:>14s} | {fi_median * 1e6:>14.1f} | {sgl_ported_median * 1e6:>14.1f} | {speedup:>13.2f}x"

        # --- External SGL kernel (if available) ---
        if HAS_SGL_KERNEL and k % 128 == 0:
            (
                sgl_input,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                quant_output,
                scale_factor,
            ) = setup_sgl_inputs(m, k, input_dtype, device)
            sgl_input.copy_(fi_input)

            def sgl_bench_fn(inp, ps, eo, bso, qo, sf):
                es_sm100_mxfp8_blockscaled_grouped_quant(inp, ps, eo, bso, qo, sf)

            sgl_ext_times = bench_gpu_time(
                fn=sgl_bench_fn,
                input_args=(
                    sgl_input,
                    problem_sizes,
                    expert_offsets,
                    blockscale_offsets,
                    quant_output,
                    scale_factor,
                ),
                dry_run_iters=dry_run,
                repeat_iters=num_iters,
                enable_cupti=True,
            )
            sgl_ext_median = np.median(sgl_ext_times)

            # Precision: compare ported vs external
            q_ported, _ = flashinfer.mxfp8_quantize(fi_input, use_sgl=True)
            es_sm100_mxfp8_blockscaled_grouped_quant(
                sgl_input,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                quant_output,
                scale_factor,
            )
            q_match = compare_quant_values(q_ported, quant_output, m, k)

            row += f" | {sgl_ext_median * 1e6:>14.1f} | {q_match:>13.1f}%"

        print(row)

    print()
    print("Speedup > 1.0 means FI+SGL (ported) is faster than FI original.")
    if HAS_SGL_KERNEL:
        print("Q match% shows agreement between ported and external SGL kernel.")


if __name__ == "__main__":
    main()
