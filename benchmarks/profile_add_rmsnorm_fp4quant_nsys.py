#!/usr/bin/env python3
"""
One-config profile driver for add_rmsnorm_fp4quant for nsys comparison (PDL off vs on).

Run with nsys to generate .nsys-rep files:
  nsys profile -o add_rmsnorm_fp4quant_pdl_off  python benchmarks/profile_add_rmsnorm_fp4quant_nsys.py
  nsys profile -o add_rmsnorm_fp4quant_pdl_on   python benchmarks/profile_add_rmsnorm_fp4quant_nsys.py

Toggle PDL in flashinfer/cute_dsl/add_rmsnorm_fp4quant.py (griddepcontrol_wait and
griddepcontrol_launch_dependents) before each run to get the two traces.
"""

import torch
from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_fp4quant


def main():
    device = "cuda"
    dtype = torch.float16
    batch_size = 16384
    hidden_size = 4096
    block_size = 16
    num_iters = 100

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)
    y_fp4 = torch.empty(
        batch_size, hidden_size // 2, device=device, dtype=torch.float4_e2m1fn_x2
    )
    block_scale = torch.empty(
        batch_size, hidden_size // block_size, device=device, dtype=torch.float8_e4m3fn
    )
    global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        add_rmsnorm_fp4quant(
            x,
            residual,
            weight,
            y_fp4=y_fp4,
            block_scale=block_scale,
            global_scale=global_scale,
            eps=1e-6,
            block_size=block_size,
            scale_format="e4m3",
        )
    torch.cuda.synchronize()
    print(f"Done: {num_iters} iterations, batch={batch_size}, hidden={hidden_size}")


if __name__ == "__main__":
    main()
