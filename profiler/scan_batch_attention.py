"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import pandas as pd
import random
import time
import torch
from tqdm import tqdm
import flashinfer
from flashinfer.profiler import export_to_perfetto_trace

page_size = 1
num_kv_heads = 8
num_qo_heads = 32
head_dim = 128
layout = "NHD"
test_dtype = torch.bfloat16
causal = True


def profile_persistent_batch_attention(
    kv_lens,
    qo_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    test_dtype,
    causal,
    profiler_buffer_size,
    device="cuda",
    flipped=False,
):
    """Profile batch attention and return SM-level dataframe with configuration info."""
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, device=device, dtype=test_dtype
    )
    if layout == "NHD":
        kv_data = torch.randn(
            num_blocks,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=test_dtype,
            device=device,
        )
    elif layout == "HND":
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=test_dtype,
            device=device,
        )

    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        seq_lens.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        use_profiler=True,
        flipped_schedule=flipped,
    )

    profiler_buffer = torch.zeros(
        (profiler_buffer_size,), dtype=torch.uint64, device=device
    )

    # warmup
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
    profiler_buffer.zero_()

    torch.cuda.synchronize()
    start_event.record()
    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
    end_event.record()
    end_event.synchronize()
    # print(f"Kernel execution time: {start_event.elapsed_time(end_event)} ms")

    trace_name = "batch_attention.perfetto-trace"
    events = ["prefill", "decode", "reduction"]
    export_to_perfetto_trace(profiler_buffer, events, trace_name)

    # print(f"Profile trace exported to {trace_name}")

    # Analyze SM performance if task_info is available
    from flashinfer.profiler import analyze_sm_performance

    df = analyze_sm_performance(profiler_buffer, wrapper._task_info, events)

    return df


def profile_training_data():
    p_lens = list(2**i for i in range(8, 14))
    d_lens = list(2**i for i in range(8, 15))
    seq_len_combinations = [(p, d) for p in p_lens for d in d_lens]
    # seq_len_config = [(8192, 1)] * 128 + [(4096, 4096)] * 1  # hybrid (chunked-prefill)

    # kv_lens = [p[0] for p in seq_len_config]
    # qo_lens = [p[1] for p in seq_len_config]
    datapoints_per_config = 10

    max_decode_tokens = (
        100 * 4000
    )  # The server and hold a max of bs = 100, avg kv len = 4000
    randgen = random.Random(args.seed)

    # Collect selected SMs (10 per run, grouped by unique configurations)
    all_selected_sms = []

    for p_len, d_len in tqdm(
        seq_len_combinations, desc="Profiling sequence length combinations"
    ):
        max_bs = max_decode_tokens // d_len
        batch_sizes = [
            max_bs,
            randgen.randint(int(max_bs * 0.2), int(max_bs * 0.7)),
            randgen.randint(int(max_bs * 0.5), int(max_bs * 0.9)),
        ]  # simulate below-max server load
        causal_list = [True, True, False]
        for causal, batch_size in zip(causal_list, batch_sizes):
            if causal:
                continue
            # non-causal simulates cache hit scenario, compute prefix attention
            p_kv_len = p_len if causal else randgen.randint(256, 4096)
            kv_lens = [d_len] * batch_size + [p_kv_len]
            qo_lens = [1] * batch_size + [p_len]

            df = profile_persistent_batch_attention(
                kv_lens=kv_lens,
                qo_lens=qo_lens,
                profiler_buffer_size=args.profiler_buffer_size,
                page_size=page_size,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                layout=layout,
                test_dtype=test_dtype,
                causal=causal,
                flipped=args.flipped,
            )

            if df is not None and len(df) > 0:
                # Group by unique four-tuple using existing column names from the dataframe
                # The dataframe already has: prefill_qo_len, prefill_kv_len, decode_qo_len, decode_kv_len
                grouped = df.groupby(
                    [
                        "prefill_qo_len",
                        "prefill_kv_len",
                        "decode_qo_len",
                        "decode_kv_len",
                    ]
                )
                unique_groups = list(grouped.groups.keys())

                # Create queues (lists) for each group - store the group_df and track current index
                group_queues = {}
                for group_key in unique_groups:
                    group_df = grouped.get_group(group_key)
                    group_queues[group_key] = group_df.copy()  # Copy the dataframe

                # Select from each unique group first, then fill with duplicates
                selected_rows = []

                # First pass: pop one SM from each unique group
                for i, group_key in enumerate(unique_groups):
                    if len(selected_rows) >= datapoints_per_config:
                        break
                    if len(group_queues[group_key]) > 0:
                        selected_rows.append(group_queues[group_key].iloc[[0]].copy())
                        group_queues[group_key] = (
                            group_queues[group_key].iloc[1:].reset_index(drop=True)
                        )
                        print(
                            f"Configuration {i}: prefill_qo_len={group_key[0]}, prefill_kv_len={group_key[1]}, decode_qo_len={group_key[2]}, decode_kv_len={group_key[3]}"
                        )

                # Second pass: if we have fewer than datapoints_per_config, pop more from existing groups
                group_idx = 0
                while (
                    len(selected_rows) < datapoints_per_config
                    and len(unique_groups) > 0
                ):
                    group_key = unique_groups[group_idx]
                    if len(group_queues[group_key]) > 0:
                        selected_rows.append(group_queues[group_key].iloc[[0]].copy())
                        group_queues[group_key] = (
                            group_queues[group_key].iloc[1:].reset_index(drop=True)
                        )
                        print(
                            f"  Filling duplicate from config {group_idx}: prefill_qo_len={group_key[0]}, prefill_kv_len={group_key[1]}, decode_qo_len={group_key[2]}, decode_kv_len={group_key[3]}"
                        )
                    group_idx = (group_idx + 1) % len(unique_groups)
                    # Stop if all queues are empty
                    if all(len(q) == 0 for q in group_queues.values()):
                        print(
                            f"  All queues empty, stopping at {len(selected_rows)} SMs"
                        )
                        break

                # Combine selected rows into dataframe
                if selected_rows:
                    selected_df = pd.concat(selected_rows, ignore_index=True)
                    all_selected_sms.append(selected_df)
                    print(
                        f"Run: Selected {len(selected_rows)} SMs from {len(unique_groups)} unique seqlen configurations"
                    )

    # Combine all selected SMs into final dataframe
    final_df = pd.concat(all_selected_sms, ignore_index=True)
    print(f"\nTotal selected SMs: {len(final_df)}")
    print("\nFinal SM Performance Data:")
    print(final_df.to_string())

    # Save final dataframe
    output_name = (
        ("sm_performance_final_flipped" if args.flipped else "sm_performance_final")
        + "_"
        + time.strftime("%Y%m%d%H%M%S")
    )
    final_df.to_csv(f"{output_name}.csv", index=False)
    print(f"\nFinal SM performance data saved to {output_name}.csv")


def profile_validation_data():
    """Generate 200 random validation data points within the specified ranges."""
    p_len_min = 256
    p_len_max = 8192
    d_len_min = 256
    d_len_max = 16384
    max_decode_tokens = (
        100 * 4000
    )  # The server can hold a max of bs = 100, avg kv len = 4000
    randgen = random.Random(args.seed)
    num_validation_points = 200

    # Collect all validation SMs
    all_validation_sms = []

    # Randomly generate 200 (p_len, d_len) pairs within the ranges
    for _ in tqdm(
        range(num_validation_points), desc="Profiling validation data points"
    ):
        # Randomly select p_len and d_len from their respective ranges
        p_len = randgen.randint(p_len_min, p_len_max)
        d_len = randgen.randint(d_len_min, d_len_max)
        max_bs = max_decode_tokens // d_len

        # Randomly select batch_size from the range
        batch_size = randgen.randint(int(max_bs * 0.2), int(max_bs))

        # Randomly select causal flag (80% True, 20% False)
        causal = randgen.choices([True, False], weights=[0.8, 0.2], k=1)[0]

        # For non-causal, use random p_kv_len; for causal, use p_len
        p_kv_len = p_len if causal else randgen.randint(256, 5000)
        kv_lens = [d_len] * batch_size + [p_kv_len]
        qo_lens = [1] * batch_size + [p_len]

        df = profile_persistent_batch_attention(
            kv_lens=kv_lens,
            qo_lens=qo_lens,
            profiler_buffer_size=args.profiler_buffer_size,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            layout=layout,
            test_dtype=test_dtype,
            causal=causal,
            flipped=args.flipped,
        )

        if df is not None and len(df) > 0:
            all_validation_sms.append(df)
            print(
                f"Validation point: p_len={p_len}, d_len={d_len}, batch_size={batch_size}, causal={causal}, SMs={len(df)}"
            )

    # Combine all validation SMs into final dataframe
    final_df = pd.concat(all_validation_sms, ignore_index=True)
    print(f"\nTotal validation SMs: {len(final_df)}")
    print("\nValidation SM Performance Data:")
    print(final_df.to_string())

    # Save final dataframe
    output_name = (
        (
            "sm_performance_validation_flipped"
            if args.flipped
            else "sm_performance_validation"
        )
        + "_"
        + time.strftime("%Y%m%d%H%M%S")
    )
    final_df.to_csv(f"{output_name}.csv", index=False)
    print(f"\nValidation SM performance data saved to {output_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler-buffer-size", type=int, default=3048576)
    parser.add_argument("--flipped", default=True, type=eval)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.val:
        profile_training_data()
    else:
        profile_validation_data()
