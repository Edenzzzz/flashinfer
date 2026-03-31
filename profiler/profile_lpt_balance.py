"""
Profile LPT load-balance improvement via SM-level grid.sync() wait time.

Measures the gap between each SM's last attention event end and the reduction
start (= time spent waiting at grid.sync() barrier). Lower wait time = better
load balance.

Usage:
    python profiler/profile_lpt_balance.py
"""

import torch
import numpy as np
from collections import defaultdict

import flashinfer

# Import decode_tag and EventType directly to avoid tg4perfetto dependency
import sys
from enum import Enum

class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2

def decode_tag(tag, num_blocks, num_groups):
    sm_id = (tag >> 24) & 0xFF
    block_group_idx = (tag >> 12) & 0xFFF
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    block_idx = block_group_idx // num_groups
    group_idx = block_group_idx % num_groups
    return block_idx, group_idx, event_idx, event_type, sm_id


def extract_grid_sync_wait(profiler_buffer, event_names=None):
    """
    Extract per-SM grid.sync() wait time from profiler buffer.

    The wait time is: reduction_start - max(runner1_end, runner2_end) per SM.
    This measures how long each SM idles at the grid.sync() barrier before
    the split-KV reduction phase.

    Returns dict with keys: wait_times (list of per-SM waits in clock cycles),
    max, mean, std, attn_times (per-SM total attention time).
    """
    if event_names is None:
        event_names = ["prefill", "decode", "reduction", "grid_sync"]

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    # Collect all events
    # Key: (sm_id, block_idx) -> {event_idx: {begin, end}}
    sm_events = defaultdict(lambda: defaultdict(lambda: {"begin": None, "end": None}))

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, num_blocks, num_groups
        )
        if event_idx >= len(event_names):
            continue

        key = (sm_id, block_idx, event_idx)
        if event_type == EventType.kBegin.value:
            sm_events[sm_id][(block_idx, event_idx)]["begin"] = timestamp
        elif event_type == EventType.kEnd.value:
            sm_events[sm_id][(block_idx, event_idx)]["end"] = timestamp

    # For each SM, find:
    # - latest attention end (max of runner1_end, runner2_end across all blocks on this SM)
    # - grid.sync duration (kGridSync begin → end) = actual barrier wait time
    PREFILL_IDX = event_names.index("prefill")
    DECODE_IDX = event_names.index("decode")
    GRID_SYNC_IDX = event_names.index("grid_sync") if "grid_sync" in event_names else -1

    wait_times = []
    attn_times = []

    for sm_id in sorted(sm_events.keys()):
        events = sm_events[sm_id]
        latest_attn_end = 0
        earliest_attn_start = float('inf')
        grid_sync_duration = None

        for (block_idx, event_idx), times in events.items():
            if event_idx in (PREFILL_IDX, DECODE_IDX):
                if times["end"] is not None:
                    latest_attn_end = max(latest_attn_end, times["end"])
                if times["begin"] is not None:
                    earliest_attn_start = min(earliest_attn_start, times["begin"])
            elif event_idx == GRID_SYNC_IDX:
                if times["begin"] is not None and times["end"] is not None:
                    dur = times["end"] - times["begin"]
                    if grid_sync_duration is None or dur > grid_sync_duration:
                        grid_sync_duration = dur

        if grid_sync_duration is not None:
            wait_times.append(grid_sync_duration)
            if earliest_attn_start < float('inf') and latest_attn_end > 0:
                attn_times.append(latest_attn_end - earliest_attn_start)

    wait_arr = np.array(wait_times, dtype=np.float64)
    attn_arr = np.array(attn_times, dtype=np.float64)

    return {
        "wait_times": wait_arr,
        "attn_times": attn_arr,
        "max": float(wait_arr.max()) if len(wait_arr) > 0 else 0,
        "mean": float(wait_arr.mean()) if len(wait_arr) > 0 else 0,
        "std": float(wait_arr.std()) if len(wait_arr) > 0 else 0,
        "min": float(wait_arr.min()) if len(wait_arr) > 0 else 0,
        "num_sms": len(wait_arr),
    }


def run_profile(seq_len_config, flipped, label, num_kv_heads=8, num_qo_heads=32,
                head_dim=128, page_size=1, profiler_buffer_size=3048576):
    """Run one profiling pass and return wait time stats."""
    kv_lens = [p[0] for p in seq_len_config]
    qo_lens = [p[1] for p in seq_len_config]

    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()
    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_data = torch.randn(num_blocks, 2, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        q_indptr.to("cuda"), kv_indptr.to("cuda"),
        torch.arange(num_blocks).int().to("cuda"),
        seq_lens.to("cuda"),
        num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
        causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        use_profiler=True, flipped_schedule=flipped,
    )

    profiler_buffer = torch.zeros((profiler_buffer_size,), dtype=torch.uint64, device="cuda")

    # Warmup
    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
    profiler_buffer.zero_()

    # Profile
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
    end.record()
    end.synchronize()
    kernel_ms = start.elapsed_time(end)

    stats = extract_grid_sync_wait(profiler_buffer)
    stats["kernel_ms"] = kernel_ms
    stats["label"] = label

    return stats


def print_stats(stats):
    """Pretty-print profiling stats."""
    print(f"  {stats['label']}:")
    print(f"    Kernel time: {stats['kernel_ms']:.3f} ms")
    print(f"    grid.sync() wait (clock cycles): max={stats['max']:.0f}  mean={stats['mean']:.0f}  std={stats['std']:.0f}  min={stats['min']:.0f}")
    if len(stats['attn_times']) > 0:
        attn = stats['attn_times']
        print(f"    Attn compute (cycles): max={attn.max():.0f}  mean={attn.mean():.0f}  std={attn.std():.0f}")
        ratio = stats['max'] / attn.mean() * 100 if attn.mean() > 0 else 0
        print(f"    Max wait / mean attn = {ratio:.1f}%")
    print(f"    SMs reporting: {stats['num_sms']}")


if __name__ == "__main__":
    # Realistic LLM serving workloads based on WL1-6 and agent workload analysis.
    # Decode KV ≤16k, ≤100 decode requests. Mixed split/non-split KV.
    workloads = [
        # WL1-like: Short output, moderate prefill. Decode KV ~2k (no split expected).
        ("WL1: 60 decode@kv2k + 2 prefill@4k (no split)",
         [(2048, 1)] * 60 + [(4096, 4096)] * 2),

        # WL2-like: Medium context. Decode KV ~4k → may split depending on kv_limit.
        ("WL2: 40 decode@kv4k + 1 prefill@8k (borderline split)",
         [(4096, 1)] * 40 + [(8192, 8192)] * 1),

        # WL3-like: Long context decode. Decode KV 8k → split-KV triggered.
        ("WL3: 30 decode@kv8k + 1 prefill@12k (decode split)",
         [(8192, 1)] * 30 + [(12000, 12000)] * 1),

        # WL4-like: Very long decode context. Decode KV 12k → heavy split-KV.
        ("WL4: 20 decode@kv12k + 1 prefill@4k (heavy split)",
         [(12288, 1)] * 20 + [(4096, 4096)] * 1),

        # Agent-like: Many short decodes + multiple small prefills.
        ("Agent: 80 decode@kv1k + 3 prefill@2k (no split, many seqs)",
         [(1024, 1)] * 80 + [(2048, 2048)] * 3),

        # Mixed-KV: Varying decode KV lengths (realistic heterogeneous batch).
        ("Mixed: 50 decode@kv1k-8k + 2 prefill@4k (mixed split/non-split)",
         [(1024 + i * 140, 1) for i in range(50)] + [(4096, 4096)] * 2),

        # Long prefill + few decode: Chunked prefill dominant.
        ("Prefill-heavy: 10 decode@kv4k + 1 prefill@16k (prefill split + decode split)",
         [(4096, 1)] * 10 + [(16384, 16384)] * 1),
    ]

    for wl_name, config in workloads:
        print(f"\n{'='*60}")
        print(f"Workload: {wl_name}")
        print(f"  Sequences: {len(config)}, Total Q tokens: {sum(q for _, q in config)}")
        print(f"{'='*60}")

        static_stats = run_profile(config, flipped=False, label="Static (no LPT)")
        lpt_stats = run_profile(config, flipped=True, label="LPT (dynamic)")

        print_stats(static_stats)
        print_stats(lpt_stats)

        # Improvement
        if static_stats['max'] > 0:
            improvement = (static_stats['max'] - lpt_stats['max']) / static_stats['max'] * 100
            print(f"\n  Max wait reduction: {improvement:.1f}%")
        if static_stats['std'] > 0:
            std_improvement = (static_stats['std'] - lpt_stats['std']) / static_stats['std'] * 100
            print(f"  Std wait reduction: {std_improvement:.1f}%")

    print(f"\n{'='*60}")
    print("Done.")
