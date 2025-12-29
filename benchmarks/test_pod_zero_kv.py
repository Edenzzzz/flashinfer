import torch
import flashinfer


def test_batch_pod_skips_zero_kv_len_queries():
    """
    Regression test for BatchPODWithPagedKVCacheWrapper with zero-KV-length prefill queries.

    Construct a mixed batch where:
      - Prefill side has some requests with kv_len == 0 but non-zero q_len.
      - Decode side has normal non-zero kv_len and q_len == 1 per request.

    The POD kernel should not read out-of-bounds and should run without illegal
    memory access, effectively skipping prefill queries whose kv_len == 0.
    """
    device = "cuda"
    dtype = torch.bfloat16

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1  # POD currently requires page_size == 1

    # Prefill batch: 3 requests, where the first has kv_len == 0
    # q_lens_p sums to 10, but kv_lens_p sums to 6 (first request has no KV)
    q_lens_p = torch.tensor([4, 3, 3], dtype=torch.int32)
    kv_lens_p = torch.tensor([0, 3, 3], dtype=torch.int32)

    # Decode batch: 5 requests, each with q_len == 1 and some KV
    d_bs = 5
    q_lens_d = torch.ones(d_bs, dtype=torch.int32)
    kv_lens_d = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32)

    # Build indptr for queries (decode first, then prefill) – matches the
    # semantics in bench_mixed_attention.run_bench
    d_q_indptr = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens_d, 0)], dim=0
    ).int()
    p_q_indptr = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens_p, 0)], dim=0
    ).int()

    # Blocks for KV (page_size == 1 → lens == blocks)
    d_kv_blocks = kv_lens_d
    p_kv_blocks = kv_lens_p

    kv_lens_all = torch.cat([kv_lens_d, kv_lens_p], dim=0)
    kv_blocks_all = kv_lens_all  # page_size == 1

    kv_indptr_all = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_blocks_all, 0)], dim=0
    ).int()

    # Split global kv_indptr into decode and prefill segments the same way as
    # bench_mixed_attention.run_bench
    kv_indptr_d = kv_indptr_all[: d_bs + 1]
    kv_indptr_p = kv_indptr_all[d_bs:]

    num_blocks = kv_indptr_all[-1].item()

    # Allocate Q and KV buffers
    q_total = (d_q_indptr[-1] + p_q_indptr[-1]).item()
    q = torch.randn(q_total, num_qo_heads, head_dim, device=device, dtype=dtype)

    kv_data = torch.randn(
        num_blocks,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )

    # Simple contiguous page indices and last-page lens
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    last_page_len_d = (d_kv_blocks - 1) % page_size + 1
    last_page_len_p = (p_kv_blocks - 1) % page_size + 1

    # Slice q for decode / prefill (decode first, then prefill)
    q_d = q[: d_q_indptr[-1]]
    q_p = q[d_q_indptr[-1] :]

    # Workspace
    workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Construct POD wrapper and plan with some prefill kv_lens == 0
    wrapper_pod = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    # This is the critical call that used to trigger illegal memory access
    wrapper_pod.plan(
        # Prefill params
        p_q_indptr.to(device),
        kv_indptr_p.to(device),
        kv_indices.to(device),
        last_page_len_p.to(device),
        # Decode params
        d_q_indptr.to(device),
        kv_indptr_d.to(device),
        kv_indices.to(device),
        last_page_len_d.to(device),
        # Common params
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Run once to ensure the kernel does not crash or access illegal memory
    o, lse = wrapper_pod.run(
        q_p,
        kv_data,
        q_d,
        kv_data,
        causal_p=True,
        join_outputs=False,
        return_lse=True,
    )
    o_p, o_d = o
    o = torch.cat([o_p, o_d], dim=0)

    # Basic sanity checks
    assert isinstance(o, torch.Tensor)
    assert o.shape[0] == q_total
    assert o.shape[1] == num_qo_heads
    assert o.shape[2] == head_dim


def test_batch_pod_exact_case_from_sglang_log():
    """
    Regression test using the exact indptr/indices configuration observed in sglang logs
    (mixed prefill+decode batch that previously triggered a warp illegal address).
    """
    device = "cuda"
    dtype = torch.bfloat16

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1

    # From sglang log:
    # qo_indptr_p: tensor([   0,  491,  668, 1234, 1246, 2147, 2548, 2719], dtype=int32)
    qo_indptr_p = torch.tensor(
        [0, 491, 668, 1234, 1246, 2147, 2548, 2719], dtype=torch.int32, device=device
    )

    # qo_indptr_d: tensor([0, 1, 2, ..., 186], dtype=int32)
    qo_indptr_d = torch.arange(0, 187, dtype=torch.int32, device=device)

    # kv_indptr_p: tensor([  0, 509, 509, 510, 514, 515, 516, 516], dtype=int32)
    kv_indptr_p = torch.tensor(
        [0, 509, 509, 510, 514, 515, 516, 516], dtype=torch.int32, device=device
    )

    # kv_indptr_d: long list from log (length 187, starting at 516)
    kv_indptr_d = torch.tensor(
        [
            516,
            768,
            1528,
            1909,
            2427,
            2990,
            3892,
            4233,
            5002,
            5937,
            6660,
            7634,
            8608,
            8847,
            9447,
            10288,
            10585,
            11068,
            12096,
            12749,
            13486,
            14068,
            14374,
            14733,
            15505,
            16425,
            17294,
            17771,
            18260,
            19362,
            20415,
            21528,
            22626,
            22988,
            23911,
            24197,
            24740,
            24994,
            25737,
            26491,
            27231,
            28313,
            29297,
            30320,
            31491,
            32271,
            33234,
            33953,
            34371,
            34678,
            35786,
            37014,
            37479,
            38152,
            39138,
            39905,
            40586,
            41252,
            41978,
            42778,
            43973,
            44334,
            45261,
            45736,
            46025,
            47031,
            48211,
            48666,
            49227,
            50026,
            50429,
            50944,
            52084,
            53025,
            53389,
            54297,
            54823,
            55405,
            56463,
            57447,
            58660,
            59000,
            59493,
            60088,
            60658,
            61027,
            62174,
            62700,
            63509,
            63888,
            64744,
            65246,
            65878,
            66867,
            67360,
            68229,
            69194,
            69495,
            70593,
            71274,
            71504,
            71783,
            72194,
            72434,
            73016,
            73968,
            74654,
            75768,
            76264,
            77144,
            77975,
            78212,
            79204,
            80126,
            80884,
            81171,
            81774,
            82445,
            83578,
            84355,
            84884,
            85615,
            86795,
            87803,
            88062,
            88866,
            89108,
            90144,
            90696,
            91533,
            92579,
            93329,
            93859,
            94969,
            95502,
            95927,
            96407,
            97352,
            98222,
            99064,
            99712,
            100931,
            101781,
            102454,
            103233,
            104031,
            104464,
            105662,
            106878,
            107849,
            108568,
            109382,
            110464,
            110768,
            111127,
            112346,
            112957,
            113182,
            113719,
            114687,
            115414,
            116039,
            116539,
            116936,
            117452,
            117925,
            118501,
            119634,
            120321,
            121420,
            121890,
            122599,
            123483,
            124627,
            125244,
            125699,
            125991,
            126978,
            128103,
            129036,
            129901,
            130968,
            132148,
            132512,
            132861,
            133664,
            133922,
        ],
        dtype=torch.int32,
        device=device,
    )

    # Total number of KV pages
    num_blocks = int(kv_indptr_d[-1].item())

    # kv_indices: contiguous range [0, num_blocks)
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)

    # kv_last_page_len_p: tensor([1, 1, 1, 1, 1, 1, 1], dtype=int32)
    kv_last_page_len_p = torch.ones(7, dtype=torch.int32, device=device)
    # kv_last_page_len_d: many ones (length 186)
    kv_last_page_len_d = torch.ones(186, dtype=torch.int32, device=device)

    # q_reshaped.shape: torch.Size([2905, 32, 128]) from log
    total_q_tokens = 2905
    q = torch.randn(total_q_tokens, num_qo_heads, head_dim, device=device, dtype=dtype)

    # Split q into decode then prefill according to qo_indptr_d / qo_indptr_p
    q_d = q[: qo_indptr_d[-1].item()]
    q_p = q[qo_indptr_d[-1].item() :]

    # Allocate KV cache (paged) – arbitrary data, shape consistent with num_blocks
    kv_data = torch.randn(
        num_blocks,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )

    # Workspace for BatchPODWithPagedKVCacheWrapper
    workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    wrapper_pod = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    # Plan with the exact indptr/indices configuration
    wrapper_pod.plan(
        # Prefill params
        qo_indptr_p,
        kv_indptr_p,
        kv_indices,
        kv_last_page_len_p,
        # Decode params
        qo_indptr_d,
        kv_indptr_d,
        kv_indices,
        kv_last_page_len_d,
        # Common params
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Run once – this is where we previously saw warp illegal address
    out, lse = wrapper_pod.run(
        q_p,
        kv_data,
        q_d,
        kv_data,
        causal_p=True,
        join_outputs=False,
        return_lse=True,
    )
    out_p, out_d = out
    out = torch.cat([out_p, out_d], dim=0)
    # Sanity: shapes match expectations
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == total_q_tokens
    assert out.shape[1] == num_qo_heads
    assert out.shape[2] == head_dim


def test_batch_pod_zero_kv_prefill_with_decode():
    """
    Regression test loading exact parameters saved from sglang kernel call.
    Loads tensors saved before the kernel call to ensure valid content before
    memory corruption occurs.

    Saved files (from flashinfer_backend.py:832-847):
    - qo_indptr_p.pt, qo_indptr_d.pt
    - kv_indptr_p.pt, kv_indptr_d.pt
    - kv_indices.pt
    - kv_last_page_len_p.pt, kv_last_page_len_d.pt
    - q_p.pt, q_d.pt
    - kv_buffer_0.pt (K buffer only - V buffer might be missing!)

    Note: kv_buffer is a tuple (k_buffer, v_buffer), but only kv_buffer[0] (K) is saved.
    We reconstruct kv_data from the saved K buffer for the test.
    """
    import os

    device = "cuda"
    dtype = torch.bfloat16

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1

    # Load saved tensors
    saved_files = {
        "qo_indptr_p": "qo_indptr_p.pt",
        "qo_indptr_d": "qo_indptr_d.pt",
        "kv_indptr_p": "kv_indptr_p.pt",
        "kv_indptr_d": "kv_indptr_d.pt",
        "kv_indices": "kv_indices.pt",
        "kv_last_page_len_p": "kv_last_page_len_p.pt",
        "kv_last_page_len_d": "kv_last_page_len_d.pt",
        "q_p": "q_p.pt",
        "q_d": "q_d.pt",
        "kv_buffer_shape": "kv_buffer_0_shape.pt",
    }

    # Check which required files exist
    required_files = {k: v for k, v in saved_files.items() if k != "kv_buffer_1"}
    missing_files = []
    for name, filename in required_files.items():
        if not os.path.exists(filename):
            missing_files.append(f"{name} ({filename})")

    if missing_files:
        raise FileNotFoundError(
            f"Missing saved tensor files. Please run sglang first to generate them.\n"
            f"Missing: {', '.join(missing_files)}"
        )

    # Load all saved tensors
    qo_indptr_p = torch.load("qo_indptr_p.pt", map_location=device)
    qo_indptr_d = torch.load("qo_indptr_d.pt", map_location=device)
    kv_indptr_p = torch.load("kv_indptr_p.pt", map_location=device)
    kv_indptr_d = torch.load("kv_indptr_d.pt", map_location=device)
    kv_indices = torch.load("kv_indices.pt", map_location=device)
    kv_last_page_len_p = torch.load("kv_last_page_len_p.pt", map_location=device)
    kv_last_page_len_d = torch.load("kv_last_page_len_d.pt", map_location=device)
    # kv_buffer_0 = torch.load("kv_buffer_0.pt", map_location=device)
    # kv_buffer_1 = torch.load("kv_buffer_1.pt", map_location=device)
    kv_buffer_0_shape = torch.load("kv_buffer_0_shape.pt", map_location=device)
    kv_buffer_0 = torch.randn(kv_buffer_0_shape, dtype=dtype, device=device)
    kv_buffer_1 = torch.randn(kv_buffer_0_shape, dtype=dtype, device=device)

    # Verify shapes match expectations
    assert kv_buffer_0.shape == torch.Size([403196, 8, 128]), (
        f"Expected kv_buffer_0.shape [403196, 8, 128], got {kv_buffer_0.shape}"
    )

    # Split q: decode first (1 token), then prefill (236 tokens)
    q_p = torch.load("q_p.pt", map_location=device)
    q_d = torch.load("q_d.pt", map_location=device)
    total_q_tokens = q_p.shape[0] + q_d.shape[0]
    # Reconstruct kv_data from saved kv_buffer[0] (K buffer) and kv_buffer[1] (V buffer)
    # kv_buffer[0] and kv_buffer[1] have shape [403196, 8, 128] which is [num_blocks, num_kv_heads, head_dim]
    # FlashInfer POD expects kv_data with shape [num_blocks, 2, page_size, num_kv_heads, head_dim]
    # where 2 is for K and V
    num_blocks = kv_buffer_0.shape[0]
    k_buffer = kv_buffer_0.view(num_blocks, page_size, num_kv_heads, head_dim)
    v_buffer = kv_buffer_1.view(num_blocks, page_size, num_kv_heads, head_dim)
    kv_data = torch.stack(
        [k_buffer, v_buffer], dim=1
    )  # [num_blocks, 2, page_size, num_kv_heads, head_dim]

    # Workspace for BatchPODWithPagedKVCacheWrapper
    workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    wrapper_pod = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    # Plan with the exact indptr/indices configuration
    wrapper_pod.plan(
        # Prefill params
        qo_indptr_p,
        kv_indptr_p,
        kv_indices,
        kv_last_page_len_p,
        # Decode params
        qo_indptr_d,
        kv_indptr_d,
        kv_indices,
        kv_last_page_len_d,
        # Common params
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Run once – this should handle zero-KV prefill gracefully
    out, lse = wrapper_pod.run(
        q_p,
        kv_data,
        q_d,
        kv_data,
        causal_p=True,
        join_outputs=False,
        return_lse=True,
    )
    out_p, out_d = out
    out = torch.cat([out_p, out_d], dim=0)

    # Sanity: shapes match expectations
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == total_q_tokens
    assert out.shape[1] == num_qo_heads
    assert out.shape[2] == head_dim


if __name__ == "__main__":
    # Allow quick manual run without pytest
    test_batch_pod_skips_zero_kv_len_queries()
    test_batch_pod_exact_case_from_sglang_log()
    # test_batch_pod_zero_kv_prefill_with_decode()
    print("tests passed.")
