#!/usr/bin/env python3
"""Fixed 18-workload definitions for the 377_chunk_kda_fwd task benchmark.

This module mirrors, verbatim, the workload construction exposed by the task's
public fixed-benchmark contract (``test_cases.py`` workload table, per-case
``model_seed = 377300000 + case_index``, input dtypes/generator ordering,
per-case ``get_init_inputs()`` constants) so the candidate leg and the pinned
fast_impl baseline leg can construct byte-identical inputs in two clean
processes and the aggregate can reject any missing, extra, reordered, or
seed-mismatched row by fingerprint comparison.

The fixed protocol these workloads are timed with lives in
``bench_recurrent_kda_prefill_fixed_protocol.py``: warmup=5 and iters=10
count-based CUPTI samples (``bench_gpu_time(dry_run_iters=5, repeat_iters=10,
enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False)``) with the
MEDIAN as the per-shape latency (SKB/toolbox convention), eager execution,
inputs built once per workload and reused across all warmup/timed calls
(``output`` allocated by the caller, ``initial_state`` mutated in place by
every call, matching the fixed benchmark's forward contract), and an
independently owned persistent RecurrentKDAPrefillWorkspace per leg.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
LOWER_BOUND = -5.0
MODEL_SEED_BASE = 377300000


@dataclass(frozen=True)
class FixedWorkload:
    index: int
    sequence_lengths: tuple[int, ...]
    heads: int

    @property
    def packed(self) -> bool:
        return len(self.sequence_lengths) > 1

    @property
    def model_seed(self) -> int:
        return MODEL_SEED_BASE + self.index

    @property
    def name(self) -> str:
        tag = "packed" if self.packed else "fixed"
        return (
            f"case{self.index:02d}_{tag}_h{self.heads}_"
            f"{'x'.join(str(s) for s in self.sequence_lengths)}"
        )


# Ordered exactly as the fixed benchmark's merged workload table: the B200
# SGLANG shapes first, then the FlashInfer PR 4262 fixed/packed set (the
# duplicate [8192]xH* entries deduplicated), then the PR 4571 small-BH set.
FIXED_KDA_WORKLOADS: tuple[FixedWorkload, ...] = (
    FixedWorkload(0, (512,), 64),
    FixedWorkload(1, (1024,), 64),
    FixedWorkload(2, (2048,), 64),
    FixedWorkload(3, (4096,), 64),
    FixedWorkload(4, (8192,), 64),
    FixedWorkload(5, (512,), 96),
    FixedWorkload(6, (1024,), 96),
    FixedWorkload(7, (2048,), 96),
    FixedWorkload(8, (4096,), 96),
    FixedWorkload(9, (8192,), 96),
    FixedWorkload(10, (1300, 547, 2048, 963, 271, 3063), 96),
    FixedWorkload(11, (1024,) * 8, 96),
    FixedWorkload(12, (1300, 547, 2048, 963, 271, 3063), 64),
    FixedWorkload(13, (1024,) * 8, 64),
    FixedWorkload(14, (65536,), 8),
    FixedWorkload(15, (65536,), 4),
    FixedWorkload(16, (131072,), 1),
    FixedWorkload(17, (1048576,), 1),
)


def build_case_inputs(workload: FixedWorkload, device: str = "cuda") -> dict:
    """Reproduce the fixed benchmark's per-case ``get_inputs()`` exactly.

    Same CUDA generator seeding, same randn call order, same dtypes/scaling,
    same packed ``cu_seqlens``/``seq_order`` policy, same preallocated
    ``output = torch.empty_like(v)``. ``initial_state`` doubles as the
    in-place final-state write destination, exactly like the fixed harness.
    """

    batch = 1
    seq_len = sum(workload.sequence_lengths)
    state_count = len(workload.sequence_lengths) if workload.packed else 1
    generator = torch.Generator(device=device)
    generator.manual_seed(workload.model_seed)
    shape = (batch, seq_len, workload.heads, HEAD_DIM)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    g = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.25
    )
    beta = torch.randn(
        (batch, seq_len, workload.heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    a_log = (
        torch.randn(
            workload.heads, dtype=torch.float32, device=device, generator=generator
        )
        * 0.25
    )
    dt_bias = (
        torch.randn(
            (workload.heads, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    )
    initial_state = torch.randn(
        (state_count, workload.heads, HEAD_DIM, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output = torch.empty_like(v)
    cu_seqlens = None
    seq_order = None
    if workload.packed:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(list(workload.sequence_lengths)).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        )
        seq_order = torch.tensor(
            sorted(
                range(len(workload.sequence_lengths)),
                key=list(workload.sequence_lengths).__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device=device,
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "seq_order": seq_order,
        "output": output,
    }


def _tensor_fingerprint(tensor: torch.Tensor, label: str) -> dict:
    snapshot = tensor.detach().contiguous().cpu()
    raw_bytes = snapshot.view(torch.uint8).numpy().tobytes()
    return {
        "label": label,
        "shape": list(snapshot.shape),
        "dtype": str(snapshot.dtype).replace("torch.", ""),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def fingerprint_case_inputs(inputs: dict) -> dict:
    """Fingerprint every constructed input tensor the timed calls consume.

    ``output`` is preallocated but never read as input, so only its layout is
    recorded (the sentinel/full-write check separately proves it is fully
    overwritten). ``initial_state`` is fingerprinted from its pre-call
    contents; the timed region mutates it in place, like the fixed harness.
    """

    fingerprints = {}
    for label in ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "initial_state"):
        fingerprints[label] = _tensor_fingerprint(inputs[label], label)
    for label in ("cu_seqlens", "seq_order"):
        value = inputs[label]
        fingerprints[label] = (
            _tensor_fingerprint(value, label) if value is not None else None
        )
    output = inputs["output"]
    fingerprints["output"] = {
        "label": "output",
        "shape": list(output.shape),
        "dtype": str(output.dtype).replace("torch.", ""),
        "allocation": "torch.empty_like(v) per case, preallocated before timing",
    }
    return fingerprints


def workload_descriptor(workload: FixedWorkload) -> dict:
    return {
        "index": workload.index,
        "name": workload.name,
        "sequence_lengths": list(workload.sequence_lengths),
        "heads": workload.heads,
        "packed": workload.packed,
        "model_seed": workload.model_seed,
    }


def recurrent_kda_call_kwargs(inputs: dict) -> dict:
    """The exact public-API arguments both legs forward on every timed call."""

    return {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "g": inputs["g"],
        "beta": inputs["beta"],
        "A_log": inputs["a_log"],
        "dt_bias": inputs["dt_bias"],
        "scale": SCALE,
        "initial_state": inputs["initial_state"],
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": LOWER_BOUND,
        "cu_seqlens": inputs["cu_seqlens"],
        "output": inputs["output"],
        "seq_order": inputs["seq_order"],
        "beta_is_logit": True,
    }
