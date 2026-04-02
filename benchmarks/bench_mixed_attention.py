import argparse
import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def run_bench(
    p_qo_lens,
    p_kv_lens,
    d_qo_lens,
    d_kv_lens,
    # page_block_size=1,
    num_kv_heads=8,
    num_qo_heads=32,
    head_dim=128,
    device=0,
    causal=True,
):
    # POD Attention only supports page size = 1 due to use of single prefill kernel
    page_block_size = 1
    d_bs = len(d_kv_lens)
    seq_lens = torch.tensor(d_kv_lens + p_kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(d_qo_lens + p_qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
    p_seq_lens_blocks = torch.ceil(
        torch.tensor(p_kv_lens, dtype=torch.int32) / page_block_size
    ).int()
    d_seq_lens_blocks = torch.ceil(
        torch.tensor(d_kv_lens, dtype=torch.int32) / page_block_size
    ).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    p_q_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(p_qo_lens), 0)], dim=0
    ).int()
    # p_kv_indptr = torch.cat(
    #     [torch.tensor([0]), torch.cumsum(p_seq_lens_blocks, 0)], dim=0
    # ).int()

    d_q_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(d_qo_lens), 0)], dim=0
    ).int()
    d_kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(d_seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )
    kv_data = torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )

    workspace_buffer = torch.empty(156 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv_layout = "NHD"

    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout=kv_layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    wrapper_old.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    o = wrapper_old.run(q, kv_data)
    measurements = bench_gpu_time(lambda: wrapper_old.run(q, kv_data))
    ms_old = np.median(measurements)
    o_ref = o.clone()
    del wrapper_old, o; torch.cuda.empty_cache()

    # Helper to create BatchAttention, bench, and cleanup
    def _bench_persistent(flipped_schedule=None):
        w = flashinfer.BatchAttention(kv_layout="NHD")
        plan_kwargs = dict(
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, causal=causal,
        )
        if flipped_schedule is not None:
            plan_kwargs["flipped_schedule"] = flipped_schedule
        w.plan(
            q_indptr.to(device), kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            seq_lens.to(device),
            num_qo_heads, num_kv_heads, head_dim, head_dim, page_block_size,
            **plan_kwargs,
        )
        out, _ = w.run(q, kv_data)
        torch.testing.assert_close(out, o_ref, rtol=4e-3, atol=4e-3)
        ms = float(np.median(bench_gpu_time(lambda: w.run(q, kv_data))))
        del w, out; torch.cuda.empty_cache()
        return ms

    # Persistent with FlippedSchedule
    ms_persistent = _bench_persistent()

    # Persistent with static schedule
    ms_persistent_static = _bench_persistent(flipped_schedule=False)

    # Batched POD Attention
    q_d = q[: d_q_indptr[-1]]
    # kv_d = kv_data[: d_kv_indptr[-1]].unbind(1)
    q_p = q[d_q_indptr[-1] :]
    # kv_p = kv_data[d_kv_indptr[-1] :].unbind(1)
    # kv_indices_d = torch.arange(0, d_kv_indptr[-1], device=device, dtype=torch.int32)
    # kv_indices_p = torch.arange(0, p_kv_indptr[-1], device=device, dtype=torch.int32)

    last_page_len_d = (d_seq_lens_blocks - 1) % page_block_size + 1
    last_page_len_p = (p_seq_lens_blocks - 1) % page_block_size + 1
    wrapper_pod = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout=kv_layout,
    )
    wrapper_pod.plan(
        # Prefill params
        p_q_indptr.to(device),
        kv_indptr[d_bs:],
        kv_indices,
        last_page_len_p,
        # Decode params
        d_q_indptr.to(device),
        kv_indptr[: d_bs + 1],
        kv_indices,
        last_page_len_d,
        # Common params
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_block_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    (o_p_batch, o_d_batch), _ = wrapper_pod.run(
        q_p,
        kv_data,
        q_d,
        kv_data,
        causal_p=causal,
        return_lse=True,
    )
    o_batch_pod = torch.cat([o_d_batch, o_p_batch], dim=0)

    # Verify output matches (reference o is decode-first, then prefill)
    torch.testing.assert_close(
        o_batch_pod, o_ref, rtol=4e-3, atol=4e-3
    )
    measurements = bench_gpu_time(
        lambda: wrapper_pod.run(
            q_p,
            kv_data,
            q_d,
            kv_data,
            causal_p=causal,
        )
    )
    ms_batch_pod = np.median(measurements)

    if len(p_kv_lens) == 1:
        # Single POD attention
        q_d = q[: d_q_indptr[-1]]
        kv_d = kv_data[: d_kv_indptr[-1]].unbind(1)
        q_p = q[d_q_indptr[-1] :]
        k_p, v_p = kv_data[d_kv_indptr[-1] :].unbind(1)
        k_p, v_p = k_p.squeeze(1), v_p.squeeze(1)
        kv_indices_d = torch.arange(
            0, d_kv_indptr[-1], device=device, dtype=torch.int32
        )

        last_page_len_d = (d_seq_lens_blocks - 1) % page_block_size + 1
        wrapper_pod = flashinfer.PODWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout=kv_layout,
        )
        wrapper_pod.plan(
            d_kv_indptr.to(device),
            kv_indices_d.to(device),
            last_page_len=last_page_len_d,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_block_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        o_p, o_d = wrapper_pod.run(
            q_p,
            k_p,
            v_p,
            q_d,
            kv_data,
            causal_p=causal,
        )
        o_pod = torch.cat([o_d, o_p], dim=0)
        # Verify output matches
        torch.testing.assert_close(
            o_ref, o_pod, rtol=4e-3, atol=4e-3, msg="POD-Attention output mismatch!"
        )
        measurements = bench_gpu_time(
            lambda: wrapper_pod.run(
                q_p,
                k_p,
                v_p,
                q_d,
                kv_d,
                causal_p=causal,
                causal_d=causal,
            )
        )
        ms_pod = np.median(measurements)

        # Sequential two kernels: single prefill + batch decode (tensor cores)
        # Prefill using single_prefill_with_kv_cache
        def _run_single_prefill():
            return flashinfer.prefill.single_prefill_with_kv_cache(
                q_p,
                k_p,
                v_p,
                causal=causal,
                pos_encoding_mode="NONE",
                backend="fa2",
            )

        measurements_prefill = bench_gpu_time(lambda: _run_single_prefill())
        ms_prefill = np.median(measurements_prefill)

        # Batch decode using tensor cores
        wrapper_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout=kv_layout, use_tensor_cores=True
        )
        wrapper_decode.plan(
            d_kv_indptr.to(device),
            kv_indices_d.to(device),
            last_page_len_d,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
        )
        measurements_decode = bench_gpu_time(lambda: wrapper_decode.run(q_d, kv_d))
        ms_decode = np.median(measurements_decode)
        ms_seq_two_kernels = ms_prefill + ms_decode

    print(f"Elapsed time (Batched Prefill): {ms_old:.2f} ms")
    print(f"Elapsed time (Batched POD Attention): {ms_batch_pod:.2f} ms")
    if len(p_kv_lens) == 1:
        print(f"Elapsed time (POD Attention): {ms_pod:.2f} ms")
        print(f"Elapsed time (Sequential two kernels): {ms_seq_two_kernels:.2f} ms")
    print(f"Elapsed time (Persistent FlippedSchedule): {ms_persistent:.2f} ms")
    print(f"Elapsed time (Persistent Static): {ms_persistent_static:.2f} ms")
    print(
        f"Batch POD speedup over Persistent FlippedSchedule: {ms_persistent / ms_batch_pod:.2f}x"
    )
    print(
        f"Batch POD speedup over Persistent Static: {ms_persistent_static / ms_batch_pod:.2f}x"
    )

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    print(f"Loading memory size (MB): {total_bytes / (1024**2):.2f} MB")

    bandwidth_old_gb_s = total_bytes / (ms_old * 1e-3) / (1024**3)

    print(f"Memory bandwidth (Batched Prefill): {bandwidth_old_gb_s:.2f} GB/s")
    bandwidth_batch_pod_gb_s = total_bytes / (ms_batch_pod * 1e-3) / (1024**3)
    print(
        f"Memory bandwidth (Batched POD Attention): {bandwidth_batch_pod_gb_s:.2f} GB/s"
    )
    if len(p_kv_lens) == 1:
        bandwidth_pod_gb_s = total_bytes / (ms_pod * 1e-3) / (1024**3)
        print(f"Memory bandwidth (POD Attention): {bandwidth_pod_gb_s:.2f} GB/s")
        bandwidth_seq_gb_s = total_bytes / (ms_seq_two_kernels * 1e-3) / (1024**3)
        print(
            f"Memory bandwidth (Sequential two kernels): {bandwidth_seq_gb_s:.2f} GB/s"
        )
    bandwidth_persistent_gb_s = total_bytes / (ms_persistent * 1e-3) / (1024**3)
    print(
        f"Memory bandwidth (Persistent FlippedSchedule): {bandwidth_persistent_gb_s:.2f} GB/s"
    )
    bandwidth_persistent_static_gb_s = total_bytes / (ms_persistent_static * 1e-3) / (1024**3)
    print(
        f"Memory bandwidth (Persistent Static): {bandwidth_persistent_static_gb_s:.2f} GB/s"
    )

    # Free GPU memory between cases
    import gc
    del wrapper_pod, q, kv_data, workspace_buffer, o_ref
    gc.collect()
    torch.cuda.empty_cache()

    return ms_persistent / ms_batch_pod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-scaling",
        action="store_true",
        help="Plot scaling trend: use third case and scale proportionally",
    )
    args = parser.parse_args()

    np.random.seed(42)
    torch.random.manual_seed(42)

    page_block_size = 1
    num_kv_heads = 8
    num_qo_heads = 32
    head_dim = 128

    if args.plot_scaling:
        # Use the third case (index 2) as base
        # Base: p_q = [2048] * 2, p_kv = [2048] * 2, d_q = [1] * 100, d_kv = [2048] * 100
        base_p_q = [2048] * 2
        base_p_kv = [2048] * 2
        base_d_q = [1] * 100
        base_d_kv = [2048] * 100

        scales = [0.5, 1, 2, 4, 8]
        speedups = []
        scale_values = []

        for scale in scales:
            if scale == 0.5:
                # Special case for 0.5: p_q = p_kv = [2048], d_kv = [1024] * 100
                p_q_lens = [2048]
                p_kv_lens = [2048]
                d_q_lens = [1] * 100
                d_kv_lens = [1024] * 100
            else:
                # For 1, 2, 4, 8: p_q = p_kv = [2048 * r] * 2, d_kv = [2048 * r] * 100
                scaled_val = int(2048 * scale)
                p_q_lens = [scaled_val] * 2
                p_kv_lens = [scaled_val] * 2
                d_q_lens = [1] * 100
                d_kv_lens = [scaled_val] * 100

            print(f"===== Scaling factor: {scale} =====")
            print(
                f"Prefill: p_q={p_q_lens}, p_kv={p_kv_lens}, "
                f"Decode: d_q={len(d_q_lens)} requests, d_kv={d_kv_lens[0]}"
            )
            speedup = run_bench(
                p_q_lens,
                p_kv_lens,
                d_q_lens,
                d_kv_lens,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                device=0,
                causal=True,
            )
            speedups.append(speedup)
            scale_values.append(scale)
            print(f"Speedup: {speedup:.2f}x\n")

        # Plot the trend
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(scale_values, speedups, marker="o", linewidth=2, markersize=8)
            plt.xlabel("Scaling Factor", fontsize=12)
            plt.ylabel("Speedup over Persistent BatchAttention", fontsize=12)
            plt.title("Batch POD Speedup Scaling Trend", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xscale("log", base=2)
            plt.xticks(scale_values, scale_values)
            for x, y in zip(scale_values, speedups):
                plt.annotate(
                    f"{y:.2f}x",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
            plt.tight_layout()
            plt.savefig("scaling_speedup.png", dpi=150)
            print("Plot saved to scaling_speedup.png")
        except ImportError:
            print("matplotlib not available, skipping plot generation")
            print("Speedups:", dict(zip(scale_values, speedups)))

    else:
        # Irregular sequence lengths for prefill and decode (100 decode each)
        d_q_len_configs = [[1] * 100] * 9
        d_kv_len_configs = [
            [2048] * 100,                                    # Case 1
            [2048] * 100,                                    # Case 2
            [2048] * 100,                                    # Case 3
            [2048] * 100,                                    # Case 4
            [4096] * 100,                                    # Case 5
            [8192] * 100,                                    # Case 6
            [8192] * 100,                                    # Case 7
            [2048] * 95 + [24576] * 5,                       # Case 8: split-KV decode
            [2048] * 100,                                    # Case 9: split-KV prefill
        ]
        p_q_configs = [[512], [1536], [2048] * 2, [2048], [4096], [4096], [6000],
                       [2048],                               # Case 8
                       [4096]]                               # Case 9
        p_kv_configs = [[512], [1536], [2048] * 2, [2048], [4096], [4096], [7000],
                        [2048],                              # Case 8
                        [24576]]                             # Case 9: long KV prefill

        for idx, (p_q_lens, p_kv_lens, d_q_len, d_kv_len) in enumerate(
            zip(p_q_configs, p_kv_configs, d_q_len_configs, d_kv_len_configs)
        ):
            print(f"===== Benchmark {idx + 1}: (kv_len, qo_len) set =====")
            run_bench(
                p_q_lens,
                p_kv_lens,
                d_q_len,
                d_kv_len,
                # page_block_size=page_block_size,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                device=0,
                causal=True,
            )
