from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import time
import flashinfer
from flashinfer.testing.utils import bench_gpu_time
import matplotlib.pyplot as plt

NUM_LAYERS = 36  # QWen3 8b


def run_bench(
    decode_kv_lens: Sequence[int],
    decode_qo_lens: Sequence[int],
    prefill_kv_lens: Sequence[int],
    prefill_qo_lens: Sequence[int],
    *,
    page_block_size: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    device: int = 0,
    causal: bool = True,
    repeats: int = 50,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    kv_lens = list(decode_kv_lens) + list(prefill_kv_lens)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(
        list(decode_qo_lens) + list(prefill_qo_lens), dtype=torch.int32
    )
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_data = torch.randn(
        num_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # old
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1

    def old_plan():
        wrapper_old.plan(
            q_indptr.to(device),
            kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    old_plan()  # warmup module loading
    start_time = time.perf_counter()
    old_plan()
    end_time = time.perf_counter()
    measurements_old = bench_gpu_time(
        lambda: wrapper_old.run(q, kv_data), repeat_iters=repeats
    )
    ms_old = np.mean(measurements_old) + (end_time - start_time) * 1000 / NUM_LAYERS

    # Fused kernel
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")

    # Normal schedule
    def persistent_plan(flipped_schedule: bool):
        wrapper.plan(
            q_indptr.to(device),
            kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            seq_lens.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            flipped_schedule=flipped_schedule,
        )

    persistent_plan(False)  # warmup module loading
    start_time = time.perf_counter()
    persistent_plan(False)
    end_time = time.perf_counter()
    measurements_new_normal = bench_gpu_time(
        lambda: wrapper.run(q, kv_data),
        repeat_iters=repeats,
    )
    ms_new_normal = (
        np.mean(measurements_new_normal) + (end_time - start_time) * 1000 / NUM_LAYERS
    )
    o, _ = wrapper.run(q, kv_data)

    # Overlap schedule
    persistent_plan(True)  # warmup module loading
    start_time = time.perf_counter()
    persistent_plan(True)
    end_time = time.perf_counter()
    measurements_new_flipped = bench_gpu_time(
        lambda: wrapper.run(q, kv_data),
        repeat_iters=repeats,
    )
    ms_new_flipped = (
        np.mean(measurements_new_flipped) + (end_time - start_time) * 1000 / NUM_LAYERS
    )

    # Separate prefill and decode wrappers
    q_lens_d = torch.tensor(decode_qo_lens, dtype=torch.int32, device=device)
    q_indptr_d = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(q_lens_d, 0)], dim=0
    ).int()
    seq_lens_d = torch.tensor(decode_kv_lens, dtype=torch.int32)
    seq_lens_blocks_d = torch.ceil(seq_lens_d / page_block_size).int()
    kv_indptr_d = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks_d, 0)], dim=0
    ).int()
    num_blocks_d = kv_indptr_d[-1].item()
    if len(decode_qo_lens) > 0:
        wrapper_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
            use_tensor_cores=True,
        )

        last_page_len_d = (seq_lens_d - 1) % page_block_size + 1

        q_d = q[: q_indptr_d[-1].item()]
        kv_data_d = kv_data[:num_blocks_d]
        start_time_d = time.perf_counter()
        wrapper_decode.plan(
            kv_indptr_d.to(device),
            torch.arange(num_blocks_d, dtype=torch.int32, device=device),
            last_page_len_d.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        end_time_d = time.perf_counter()
        measurements_decode = bench_gpu_time(
            lambda: wrapper_decode.run(q_d, kv_data_d), repeat_iters=repeats
        )
        ms_decode = (
            np.mean(measurements_decode)
            + (end_time_d - start_time_d) * 1000 / NUM_LAYERS
        )
    else:
        ms_decode = 0

    wrapper_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    q_lens_p = torch.tensor(prefill_qo_lens, dtype=torch.int32, device=device)
    q_indptr_p = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(q_lens_p, 0)], dim=0
    ).int()
    seq_lens_p = torch.tensor(prefill_kv_lens, dtype=torch.int32, device=device)
    seq_lens_blocks_p = torch.ceil(seq_lens_p / page_block_size).int()
    kv_indptr_p = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(seq_lens_blocks_p, 0)],
        dim=0,
    ).int()
    num_blocks_p = kv_indptr_p[-1].item()
    q_p = q[q_indptr_d[-1].item() :]
    kv_data_p = kv_data[num_blocks_d:]
    last_page_len_p = (seq_lens_p - 1) % page_block_size + 1

    if len(prefill_qo_lens) > 0:
        start_time_p = time.perf_counter()
        wrapper_prefill.plan(
            q_indptr_p.to(device),
            kv_indptr_p.to(device),
            torch.arange(num_blocks_p, dtype=torch.int32, device=device),
            last_page_len_p.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        end_time_p = time.perf_counter()
        measurements_prefill = bench_gpu_time(
            lambda: wrapper_prefill.run(q_p, kv_data_p), repeat_iters=repeats
        )
        ms_prefill = (
            np.mean(measurements_prefill)
            + (end_time_p - start_time_p) * 1000 / NUM_LAYERS
        )
    else:
        ms_prefill = 0

    ms_separate = ms_prefill + ms_decode

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    mem_MB = total_bytes / 1024**2
    bw_old = total_bytes / (ms_old * 1e-3) / 1024**3
    bw_new_normal = total_bytes / (ms_new_normal * 1e-3) / 1024**3
    bw_new_flipped = total_bytes / (ms_new_flipped * 1e-3) / 1024**3
    bw_separate = total_bytes / (ms_separate * 1e-3) / 1024**3

    return (
        ms_old,
        ms_new_normal,
        ms_new_flipped,
        ms_separate,
        mem_MB,
        bw_old,
        bw_new_normal,
        bw_new_flipped,
        bw_separate,
    )  # type: ignore


def synthesize_seq_len_configs(
    decode_len, prefill_len, prefill_chunk_size, num_prefill_reqs, num_decode_reqs
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    decode_lens: List[List[Tuple[int, int]]] = [
        [(decode_len, 1)] * num_decode_reqs,
    ]
    prefill_lens: List[List[Tuple[int, int]]] = [
        [(prefill_len, prefill_chunk_size)] * num_prefill_reqs,
    ]

    return decode_lens, prefill_lens


def plot_per_model_results(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Generate per-model comparison plots with side-by-side bars of the same color."""
    # Get unique models and schedulers
    models = df["model_name"].unique()
    schedulers = df["scheduler"].unique()

    # Color mapping for schedulers
    scheduler_colors = {
        "BatchAttentionWrapper (flipped)": "#1f77b4",
        "BatchPrefillWithPagedKVCacheWrapper": "#ff7f0e",
        "Decode + Prefill": "#9467bd",
        "BatchAttentionWrapper": "#2ca02c",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.2
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    # Plot bars for each scheduler
    for i, scheduler in enumerate(schedulers):
        if scheduler in scheduler_colors:
            values = []
            for model in models:
                model_data = df[
                    (df["model_name"] == model) & (df["scheduler"] == scheduler)
                ]
                if not model_data.empty:
                    values.append(model_data["bandwidth_GB_s"].mean())
                else:
                    values.append(0)

            bars = ax.bar(
                x + offsets[i],
                values,
                width,
                label=scheduler,
                color=scheduler_colors[scheduler],
            )

            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(values) * 0.01,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    # Customize plot
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Bandwidth (GB/s)")
    ax.set_title(
        f"Per-Model Bandwidth Comparison ({args.repeats} repeats, "
        f"{args.prefill_chunk_size // 1024}k prefill, {args.decode_len // 1024}k decode)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(fontsize=8, loc="upper right")

    # Set y-axis limits
    max_value = df["bandwidth_GB_s"].max()
    ax.set_ylim(0, max_value * 1.1)

    plt.tight_layout()

    # Save plot
    plot_filename = f"per_model_comparison_{args.repeats}_repeats_{args.prefill_chunk_size // 1024}k_prefill_{args.decode_len // 1024}k_decode.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Per-model plot saved as: {plot_filename}")
    plt.show()


def main(args: argparse.Namespace) -> None:
    np.random.seed(42)
    torch.random.manual_seed(42)
    decode_len = args.decode_len
    prefill_len = args.prefill_len
    prefill_chunk_size = args.prefill_chunk_size
    num_prefill_reqs = args.num_prefill_reqs
    num_decode_reqs = args.num_decode_reqs

    decode_lens, prefill_lens = synthesize_seq_len_configs(
        decode_len, prefill_len, prefill_chunk_size, num_prefill_reqs, num_decode_reqs
    )
    if num_prefill_reqs == 0:
        prefill_chunk_size = 0
        prefill_len = 0
    if num_decode_reqs == 0:
        decode_len = 0

    combinations = [
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 32,
            "model_name": "Qwen-8B",
        },
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 64,
            "model_name": "Llama-3.1-70B",
        },
        {
            "page_block_size": 1,
            "head_dim": 64,
            "num_kv_heads": 4,
            "num_qo_heads": 64,
            "model_name": "Qwen-MoE-235B",
        },
    ]
    records_old = []
    records_new = []
    records_new_flipped = []
    records_separate = []
    for cfg_id, (decode_case, prefill_case) in enumerate(
        zip(decode_lens, prefill_lens), start=1
    ):
        prefill_kv_lens = [p[0] for p in prefill_case]
        prefill_qo_lens = [p[1] for p in prefill_case]
        decode_kv_lens = [p[0] for p in decode_case]
        decode_qo_lens = [p[1] for p in decode_case]
        for param in combinations:
            pbs, hd, n_kv, n_qo, model_name = (
                param["page_block_size"],  # type: ignore
                param["head_dim"],  # type: ignore
                param["num_kv_heads"],  # type: ignore
                param["num_qo_heads"],  # type: ignore
                param["model_name"],
            )
            (
                ms_old,
                ms_new_normal,
                ms_new_flipped,
                ms_separate,
                mem_MB,
                bw_old,
                bw_new_normal,
                bw_new_flipped,
                bw_separate,
            ) = run_bench(
                decode_kv_lens,
                decode_qo_lens,
                prefill_kv_lens,
                prefill_qo_lens,
                page_block_size=pbs,  # type: ignore
                num_kv_heads=n_kv,  # type: ignore
                num_qo_heads=n_qo,  # type: ignore
                head_dim=hd,  # type: ignore
                device=0,
                causal=True,
                repeats=args.repeats,
            )
            records_old.extend(
                [
                    {
                        "scheduler": "BatchPrefillWithPagedKVCacheWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_old,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_old,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                    },
                ]
            )
            records_new.extend(
                [
                    {
                        "scheduler": "BatchAttentionWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_new_normal,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_new_normal,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                    },
                ]
            )
            records_new_flipped.extend(
                [
                    {
                        "scheduler": "BatchAttentionWrapper (flipped)",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_new_flipped,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_new_flipped,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                    },
                ]
            )
            records_separate.extend(
                [
                    {
                        "scheduler": "Decode + Prefill",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_separate,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_separate,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                    },
                ]
            )
    df = pd.DataFrame(
        records_old + records_new + records_new_flipped + records_separate,
        columns=[
            "scheduler",
            "seq_cfg_id",
            "page_size",
            "head_dim",
            "num_kv_heads",
            "num_qo_heads",
            "model_name",
            "time_ms",
            "memory_MB",
            "bandwidth_GB_s",
            "num_repeats",
            "decode_len",
            "prefill_len",
            "prefill_chunk_size",
        ],
    )
    file_name = "bench_batch_attention.csv"
    if os.path.exists(file_name) and args.overwrite:
        os.remove(file_name)

    if os.path.exists(file_name):
        # don't save columns
        df_save = df.drop(columns=df.columns)
        df_save.to_csv(file_name, index=False, mode="a")
    else:
        df.to_csv(file_name, index=False)

    # remove last 4 columns
    df = df.iloc[:, :-4]
    # remove page_size column
    df = df.drop(columns=["page_size"])
    print(df.to_markdown(index=False, floatfmt=".2f"))

    # Generate per-model plots if requested
    if args.plot_per_model:
        plot_per_model_results(df, args)


if __name__ == "__main__":
    # Now running both normal and flipped schedules in a single run
    # Each configuration will be benchmarked with both scheduling strategies

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--num_prefill_reqs", type=int, default=1)
    parser.add_argument("--num_decode_reqs", type=int, default=128)
    parser.add_argument("--decode_len", type=int, default=8192)
    parser.add_argument("--prefill_len", type=int, default=8192)
    parser.add_argument(
        "--plot_per_model",
        action="store_true",
        help="Generate per-model comparison plots",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing CSV file"
    )
    args = parser.parse_args()
    main(args)
