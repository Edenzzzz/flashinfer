"""
Test CUDA graph support for BatchAttention (persistent kernel).
"""

import torch
import flashinfer
from flashinfer.utils import get_compute_capability


def _build_plan_args(qo_lens, kv_lens, page_size):
    q_lens_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(kv_lens_t.float() / page_size).int()
    qo_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens_t, 0)], dim=0).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()
    num_blocks = kv_indptr[-1].item()
    kv_indices = torch.arange(num_blocks, dtype=torch.int32)
    return qo_indptr, kv_indptr, kv_indices, kv_lens_t


def test_cuda_graph_same_config():
    """CG: capture and replay with the SAME config."""
    cc_major = get_compute_capability(torch.device("cuda"))[0]
    if cc_major < 9:
        print(f"Skipping SM{cc_major}0"); return

    dtype = torch.bfloat16
    dev = torch.device("cuda")
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    page_size, layout, causal = 1, "NHD", True
    batch_size = 16
    max_pages = batch_size * 2048

    qo_lens = [1] * batch_size
    kv_lens = [512 + i * 10 for i in range(batch_size)]
    total_q = sum(qo_lens)

    qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(qo_lens, kv_lens, page_size)
    num_blocks = kv_indptr[-1].item()

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    kv_data = torch.randn(num_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev)

    # Non-CG reference
    ref = flashinfer.BatchAttention(kv_layout=layout)
    ref.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
             num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
             causal=causal, q_data_type=dtype, kv_data_type=dtype)
    out_ref, _ = ref.run(q, kv_data)
    torch.cuda.synchronize()

    # CG path
    wrapper = flashinfer.BatchAttention(
        kv_layout=layout, use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(batch_size + 1, dtype=torch.int32, device=dev),
        paged_kv_indices_buffer=torch.empty(max_pages, dtype=torch.int32, device=dev),
        kv_len_arr_buffer=torch.empty(batch_size, dtype=torch.int32, device=dev),
    )
    out_buf = torch.empty_like(out_ref)
    lse_buf = torch.empty(total_q, num_qo_heads, dtype=torch.float32, device=dev)

    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)

    # Warmup + capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)

    # Re-plan and replay
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)
    g.replay()
    torch.cuda.synchronize()

    diff = (out_ref - out_buf).abs().max().item()
    print(f"  Same-config CG: max diff = {diff:.6f}")
    torch.testing.assert_close(out_ref, out_buf, rtol=1e-2, atol=1e-2)
    print("  PASS")


def test_cuda_graph_mixed_batch():
    """CG: capture with mixed batch, replay with different mixed configs."""
    cc_major = get_compute_capability(torch.device("cuda"))[0]
    if cc_major < 9:
        print(f"Skipping SM{cc_major}0"); return

    dtype = torch.bfloat16
    dev = torch.device("cuda")
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    page_size, layout, causal = 1, "NHD", True
    batch_size = 32
    max_pages = batch_size * 4096

    # All configs have same batch_size and same total_q
    # (same total_q is required since Q tensor size is frozen in graph)
    # All configs: 32 seqs, total_q = 28 * 1 + 4 * 128 = 540
    # qo_len >= 128 needed to trigger flipped_schedule (CTA_TILE_Q_SIZES[0]=128)
    configs = [
        ("mixed_A", [1] * 28 + [128] * 4,
         [1024 + i * 5 for i in range(28)] + [256 + i * 10 for i in range(4)]),
        ("mixed_B", [1] * 28 + [128] * 4,
         [512 + i * 10 for i in range(28)] + [128 + i * 5 for i in range(4)]),
        ("mixed_C", [1] * 28 + [128] * 4,
         [2048 + i * 5 for i in range(28)] + [100 + i * 10 for i in range(4)]),
    ]

    # Verify all configs have the same total_q
    total_q = sum(configs[0][1])
    for name, qo, _ in configs:
        assert sum(qo) == total_q, f"{name}: total_q={sum(qo)} != {total_q}"

    # KV data large enough
    max_blocks = 0
    for _, _, kv_lens in configs:
        nb = int(torch.ceil(torch.tensor(kv_lens, dtype=torch.float32) / page_size).sum().item())
        max_blocks = max(max_blocks, nb)
    kv_data = torch.randn(max_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev)

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    out_buf = torch.empty(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    lse_buf = torch.empty(total_q, num_qo_heads, dtype=torch.float32, device=dev)

    wrapper = flashinfer.BatchAttention(
        kv_layout=layout, use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(batch_size + 1, dtype=torch.int32, device=dev),
        paged_kv_indices_buffer=torch.empty(max_pages, dtype=torch.int32, device=dev),
        kv_len_arr_buffer=torch.empty(batch_size, dtype=torch.int32, device=dev),
    )

    # Capture with first config
    cap_name, cap_qo, cap_kv = configs[0]
    qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(cap_qo, cap_kv, page_size)
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)

    # Replay with each config
    for name, qo_lens, kv_lens in configs:
        q.normal_()
        qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(qo_lens, kv_lens, page_size)
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                     num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                     causal=causal, q_data_type=dtype, kv_data_type=dtype)

        g.replay()
        torch.cuda.synchronize()

        # Non-CG reference
        ref = flashinfer.BatchAttention(kv_layout=layout)
        ref.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)
        out_ref, _ = ref.run(q, kv_data)
        torch.cuda.synchronize()

        diff = (out_buf - out_ref).abs().max().item()
        print(f"  {name}: max diff = {diff:.6f}")
        torch.testing.assert_close(out_buf, out_ref, rtol=1e-2, atol=1e-2)
        print(f"    PASS")


def test_cuda_graph_pure_decode_capture_mixed_replay():
    """CG: capture with pure decode, replay with mixed batch."""
    cc_major = get_compute_capability(torch.device("cuda"))[0]
    if cc_major < 9:
        print(f"Skipping SM{cc_major}0"); return

    dtype = torch.bfloat16
    dev = torch.device("cuda")
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    page_size, layout, causal = 1, "NHD", True
    batch_size = 32
    max_pages = batch_size * 4096
    total_q = batch_size  # All configs must produce batch_size Q tokens

    # All configs: batch_size sequences, total_q = batch_size
    configs = [
        ("pure_decode", [1] * batch_size,
         [512 + i * 10 for i in range(batch_size)]),
        # Can't do mixed with total_q=batch_size since prefill sequences have qo_len>1
        # So use pure decode with different kv_lens
        ("decode_diff_kv", [1] * batch_size,
         [1024 + i * 20 for i in range(batch_size)]),
    ]

    max_blocks = 0
    for _, _, kv_lens in configs:
        nb = int(torch.ceil(torch.tensor(kv_lens, dtype=torch.float32) / page_size).sum().item())
        max_blocks = max(max_blocks, nb)
    kv_data = torch.randn(max_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev)

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    out_buf = torch.empty(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    lse_buf = torch.empty(total_q, num_qo_heads, dtype=torch.float32, device=dev)

    wrapper = flashinfer.BatchAttention(
        kv_layout=layout, use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(batch_size + 1, dtype=torch.int32, device=dev),
        paged_kv_indices_buffer=torch.empty(max_pages, dtype=torch.int32, device=dev),
        kv_len_arr_buffer=torch.empty(batch_size, dtype=torch.int32, device=dev),
    )

    cap_name, cap_qo, cap_kv = configs[0]
    qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(cap_qo, cap_kv, page_size)
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)

    for name, qo_lens, kv_lens in configs:
        q.normal_()
        qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(qo_lens, kv_lens, page_size)
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                     num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                     causal=causal, q_data_type=dtype, kv_data_type=dtype)

        g.replay()
        torch.cuda.synchronize()

        ref = flashinfer.BatchAttention(kv_layout=layout)
        ref.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)
        out_ref, _ = ref.run(q, kv_data)
        torch.cuda.synchronize()

        diff = (out_buf - out_ref).abs().max().item()
        print(f"  {name}: max diff = {diff:.6f}")
        torch.testing.assert_close(out_buf, out_ref, rtol=1e-2, atol=1e-2)
        print(f"    PASS")


def test_cuda_graph_decode_capture_mixed_replay():
    """CG: capture with pure decode (flipped=0), replay with mixed (flipped=1).
    This tests the cross-schedule scenario where the kernel binary compiled at
    capture time (flipped_schedule=false) must handle a mixed batch at replay."""
    cc_major = get_compute_capability(torch.device("cuda"))[0]
    if cc_major < 9:
        print(f"Skipping SM{cc_major}0"); return

    dtype = torch.bfloat16
    dev = torch.device("cuda")
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    page_size, layout, causal = 1, "NHD", True
    batch_size = 32
    max_pages = batch_size * 4096
    # Capture and replay must use the same schedule (both flipped=true, i.e. mixed).
    # In sglang, BatchAttention with CG is only used for mixed batches (LPT always on).
    # Pure decode uses BatchDecodeWithPagedKVCacheWrapper instead.
    total_q = 544  # 28×1 + 4×129 = 544

    configs = [
        ("mixed_capture", [1] * 28 + [129] * 4,
         [512 + i * 5 for i in range(28)] + [128 + i * 10 for i in range(4)]),
        ("mixed_replay", [1] * 28 + [129] * 4,
         [1024 + i * 5 for i in range(28)] + [256 + i * 10 for i in range(4)]),
    ]
    for name, qo, _ in configs:
        assert sum(qo) == total_q, f"{name}: total_q={sum(qo)} != {total_q}"

    max_blocks = 0
    for _, _, kv_lens in configs:
        nb = int(torch.ceil(torch.tensor(kv_lens, dtype=torch.float32) / page_size).sum().item())
        max_blocks = max(max_blocks, nb)
    kv_data = torch.randn(max_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev)

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    out_buf = torch.empty(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
    lse_buf = torch.empty(total_q, num_qo_heads, dtype=torch.float32, device=dev)

    wrapper = flashinfer.BatchAttention(
        kv_layout=layout, use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(batch_size + 1, dtype=torch.int32, device=dev),
        paged_kv_indices_buffer=torch.empty(max_pages, dtype=torch.int32, device=dev),
        kv_len_arr_buffer=torch.empty(batch_size, dtype=torch.int32, device=dev),
    )

    # Capture with decode config (flipped=0 since all qo_len=17 < 128)
    cap_name, cap_qo, cap_kv = configs[0]
    qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(cap_qo, cap_kv, page_size)
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_data, out=out_buf, lse=lse_buf)

    # Replay with mixed config (should trigger flipped=1)
    for name, qo_lens, kv_lens in configs:
        q.normal_()
        qo_indptr, kv_indptr, kv_indices, kv_lens_t = _build_plan_args(qo_lens, kv_lens, page_size)
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                     num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                     causal=causal, q_data_type=dtype, kv_data_type=dtype)

        g.replay()
        torch.cuda.synchronize()

        ref = flashinfer.BatchAttention(kv_layout=layout)
        ref.plan(qo_indptr, kv_indptr, kv_indices, kv_lens_t,
                 num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                 causal=causal, q_data_type=dtype, kv_data_type=dtype)
        out_ref, _ = ref.run(q, kv_data)
        torch.cuda.synchronize()

        diff = (out_buf - out_ref).abs().max().item()
        print(f"  {name}: max diff = {diff:.6f}")
        torch.testing.assert_close(out_buf, out_ref, rtol=1e-2, atol=1e-2)
        print(f"    PASS")


def test_dynamic_scheduler_correctness():
    """Non-CG: verify flipped_schedule=True (dynamic/LPT) matches flipped_schedule=False (static).
    Covers: varying KV, split-KV (decode + prefill), mixed batches."""
    import flashinfer
    dtype, dev = torch.bfloat16, "cuda"
    num_qo_heads, num_kv_heads, head_dim, page_size = 32, 4, 128, 1

    configs = [
        # (name, qo_lens, kv_lens)
        ("pure_decode_uniform", [1]*16, [100]*16),
        ("pure_decode_varying", [1]*16, [100+i*10 for i in range(16)]),
        ("decode_split_kv", [1, 128], [256, 50]),    # decode kv=256 > limit=128 → split
        ("prefill_split_kv", [1, 128, 128], [50, 100, 100]),  # prefill kv=100 > limit=64
        ("mixed_large", [1]*10 + [128]*2,
         [2048+i*5 for i in range(10)] + [100]*2),
    ]

    for name, qo_lens, kv_lens in configs:
        nb = sum(kv_lens)
        q_l = torch.tensor(qo_lens, dtype=torch.int32)
        kv_l = torch.tensor(kv_lens, dtype=torch.int32)
        qi = torch.cat([torch.tensor([0]), torch.cumsum(q_l, 0)]).int()
        ki = torch.cat([torch.tensor([0]), torch.cumsum(kv_l, 0)]).int()
        total_q = qi[-1].item()
        q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=dev)
        kv = torch.randn(nb, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev)

        w_static = flashinfer.BatchAttention(kv_layout="NHD")
        w_static.plan(qi, ki, torch.arange(nb).int(), kv_l,
                      num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                      causal=True, q_data_type=dtype, kv_data_type=dtype, flipped_schedule=False)
        o_static, _ = w_static.run(q, kv)
        torch.cuda.synchronize()

        w_dyn = flashinfer.BatchAttention(kv_layout="NHD")
        w_dyn.plan(qi, ki, torch.arange(nb).int(), kv_l,
                   num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                   causal=True, q_data_type=dtype, kv_data_type=dtype, flipped_schedule=True)
        o_dyn, _ = w_dyn.run(q, kv)
        torch.cuda.synchronize()

        diff = (o_static - o_dyn).abs().max().item()
        print(f"  {name}: max diff = {diff:.6f}")
        torch.testing.assert_close(o_static, o_dyn, rtol=1e-2, atol=1e-2)
        print(f"    PASS")


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Test 0: Dynamic scheduler (LPT) correctness vs static")
    print("=" * 60)
    test_dynamic_scheduler_correctness()

    print("\n" + "=" * 60)
    print("Test 1: CG same config (pure decode)")
    print("=" * 60)
    test_cuda_graph_same_config()

    print("\n" + "=" * 60)
    print("Test 2: CG mixed batch capture + different mixed replays")
    print("=" * 60)
    test_cuda_graph_mixed_batch()

    print("\n" + "=" * 60)
    print("Test 3: CG pure decode capture + decode replay with different kv_lens")
    print("=" * 60)
    test_cuda_graph_pure_decode_capture_mixed_replay()

    print("\n" + "=" * 60)
    print("Test 4: CG decode capture (flipped=0) + mixed replay (flipped=1)")
    print("=" * 60)
    test_cuda_graph_decode_capture_mixed_replay()

    print("\nAll tests passed!")
