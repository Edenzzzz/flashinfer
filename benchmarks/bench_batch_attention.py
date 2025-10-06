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
from torch.testing import assert_close

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
    flipped_schedule: bool = False,
    repeats: int = 50,
) -> Tuple[float, float, float, float, float, float, float]:
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
    if flipped_schedule:
        wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
        )
        last_page_len = (seq_lens - 1) % page_block_size + 1
        start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        measurements_old = bench_gpu_time(
            lambda: wrapper_old.run(q, kv_data), repeat_iters=repeats
        )
        ms_old = np.mean(measurements_old) + (end_time - start_time) * 1000 / NUM_LAYERS
    else:
        ms_old = 1

    # new
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    start_time = time.perf_counter()
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
    )
    end_time = time.perf_counter()
    measurements_new = bench_gpu_time(
        lambda: wrapper.run(q, kv_data, flipped_schedule=flipped_schedule),
        repeat_iters=repeats,
    )
    ms_new = np.mean(measurements_new) + (end_time - start_time) * 1000 / NUM_LAYERS
    o, _ = wrapper.run(q, kv_data, flipped_schedule=flipped_schedule)
    if flipped_schedule:
        # Separate prefill and decode wrappers
        wrapper_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
        )
        wrapper_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
            use_tensor_cores=True,
        )
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
            np.mean(measurements_prefill) + (end_time_p - start_time_p) / NUM_LAYERS
        )
        measurements_decode = bench_gpu_time(
            lambda: wrapper_decode.run(q_d, kv_data_d), repeat_iters=repeats
        )
        ms_decode = (
            np.mean(measurements_decode) + (end_time_d - start_time_d) / NUM_LAYERS
        )
        ms_separate = ms_prefill + ms_decode
        o_d = wrapper_decode.run(q_d, kv_data_d)
        o_p = wrapper_prefill.run(q_p, kv_data_p)
        o_separate = torch.cat([o_d, o_p], dim=0)
        assert_close(o, o_separate, rtol=1e-3, atol=1e-3)
    else:
        ms_separate = 1

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    mem_MB = total_bytes / 1024**2
    bw_old = total_bytes / (ms_old * 1e-3) / 1024**3
    bw_new = total_bytes / (ms_new * 1e-3) / 1024**3
    bw_separate = total_bytes / (ms_separate * 1e-3) / 1024**3

    return ms_old, ms_new, ms_separate, mem_MB, bw_old, bw_new, bw_separate  # type: ignore


def synthesize_seq_len_configs() -> Tuple[
    List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]
]:
    # cfgs: List[List[Tuple[int, int]]] = [
    #     # [(8192, 1)] * 128,  # decode-only
    #     # [(4096, 128)] * 4,  # prefill-only
    #     # [(600, 1)] * 122 + [(10_000, 17)] * 8,  # hybird
    #     # [(8192, 1)] * 127 * 2 + [(2048, 512)] * 1,  # hybrid (chunked-prefill)
    #     [(8192, 1)] * 127 * 2 + [(8192, 4096)] * 1,  # hybrid (chunked-prefill)
    # ]

    decode_lens: List[List[Tuple[int, int]]] = [
        [(8192, 1)] * 128,
    ]
    prefill_lens: List[List[Tuple[int, int]]] = [
        [(8192, 4096)] * 1,
    ]

    # def _rand_case(bsz: int, lo: int, hi: int) -> List[Tuple[int, int]]:
    #     stride, sparsity = 16, 0.05
    #     full = np.random.randint(lo, hi, size=bsz)
    #     out = []
    #     for i, kv_len in enumerate(full):
    #         if i % stride == 0:
    #             out.append((kv_len, stride + 1))
    #         else:
    #             out.append((int(kv_len * sparsity), 1))
    #     return out

    # cfgs.append(_rand_case(256, 1000, 8192))
    # cfgs.append(_rand_case(128, 2000, 16_000))
    return decode_lens, prefill_lens


def main(args: argparse.Namespace) -> None:
    np.random.seed(42)
    torch.random.manual_seed(42)

    decode_lens, prefill_lens = synthesize_seq_len_configs()

    # sweep = {
    #     "page_block_size": (1,),  # (1, 8, 16),
    #     "head_dim": (
    #         # 64,
    #         128,
    #     ),
    #     "num_kv_heads": (8,),
    #     "num_qo_heads": (32, 64),
    # }
    combinations = [
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 32,
        },  # Qwen-8B
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 64,
        },
        {
            "page_block_size": 1,
            "head_dim": 64,
            "num_kv_heads": 4,
            "num_qo_heads": 64,
        },  # Qwen-MoE-235B
    ]
    records_old = []
    records_new = []
    records_separate = []
    for cfg_id, (decode_case, prefill_case) in enumerate(
        zip(decode_lens, prefill_lens), start=1
    ):
        prefill_kv_lens = [p[0] for p in prefill_case]
        prefill_qo_lens = [p[1] for p in prefill_case]
        decode_kv_lens = [p[0] for p in decode_case]
        decode_qo_lens = [p[1] for p in decode_case]
        for param in combinations:
            pbs, hd, n_kv, n_qo = (
                param["page_block_size"],
                param["head_dim"],
                param["num_kv_heads"],
                param["num_qo_heads"],
            )
            ms_old, ms_new, ms_separate, mem_MB, bw_old, bw_new, bw_separate = (
                run_bench(
                    decode_kv_lens,
                    decode_qo_lens,
                    prefill_kv_lens,
                    prefill_qo_lens,
                    page_block_size=pbs,
                    num_kv_heads=n_kv,
                    num_qo_heads=n_qo,
                    head_dim=hd,
                    device=0,
                    causal=True,
                    flipped_schedule=args.flipped,
                    repeats=args.repeats,
                )
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
                        "time_ms": ms_old,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_old,
                        "num_repeats": args.repeats,
                    },
                ]
            )
            records_new.extend(
                [
                    {
                        "scheduler": "BatchAttentionWrapper"
                        + (" (flipped)" if args.flipped else ""),
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "time_ms": ms_new,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_new,
                        "num_repeats": args.repeats,
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
                        "time_ms": ms_separate,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_separate,
                        "num_repeats": args.repeats,
                    },
                ]
            )
    df = pd.DataFrame(
        records_old + records_new + records_separate if args.flipped else records_new,
        columns=[
            "scheduler",
            "seq_cfg_id",
            "page_size",
            "head_dim",
            "num_kv_heads",
            "num_qo_heads",
            "time_ms",
            "memory_MB",
            "bandwidth_GB_s",
            "num_repeats",
        ],
    )
    file_name = "bench_batch_attention_flipped.csv"
    if os.path.exists(file_name):
        # append to the existing csv without writing headers again
        df.to_csv(file_name, mode="a", index=False, header=False)
    else:
        df.to_csv(file_name, index=False)
    print(df.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    # (NOTE)Somehow running the flipped schedule together with the normal schedule in one run causes incorrect performance for both
    # (slightly decrease for flipped, slightly increase for normal)
    # Flushing the l2 cache doesn't help. So we split the runs for now

    parser = argparse.ArgumentParser()
    parser.add_argument("--flipped", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    main(args)
