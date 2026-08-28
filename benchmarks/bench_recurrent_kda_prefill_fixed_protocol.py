#!/usr/bin/env python3
"""Fixed-protocol recurrent-KDA prefill comparison: ``vibecuda`` candidate vs
the task's configured fast_impl denominator (``flashinfer_b200``) executed
directly from its pinned FlashInfer source revision.

Unlike ``bench_recurrent_kda_prefill.py --compare-to-sota`` (the repository's
canonical SOTA benchmark over its own case sets), this script reproduces the
fixed task benchmark's protocol field by field so the measured comparison is a
direct denominator equivalence:

* Workloads: the exact 18 fixed workloads from
  ``kda_prefill_fixed_workloads.py`` (ordered sequence-length/head table,
  per-case ``model_seed = 377300000 + index``, packed ``cu_seqlens`` +
  descending-length ``seq_order``, ``output = torch.empty_like(v)``
  preallocated per case, ``initial_state`` mutated in place by every call).
* Timing: ``bench_gpu_time(dry_run_iters=5, repeat_iters=10,
  enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False)`` — count-based
  warmup=5 / iters=10 (the SKB metadata default) CUPTI samples with a
  cold-L2 flush before each iteration, eager execution, and the MEDIAN of
  the 10 samples as each leg's per-shape latency (the toolbox profiler's
  reported statistic). No adaptive 20ms/100ms windows.
* Lifecycle: inputs/output/state allocated once per case outside timing and
  byte-identical across legs; each leg receives its own independently owned
  persistent RecurrentKDAPrefillWorkspace (the frozen wrapper gives each
  implementation its own persistent workspace); only backend-internal GPU
  preprocessing lives inside the timed recurrent_kda call.
* Call: the exact public ``flashinfer.kda.recurrent_kda`` argument vector
  both legs forward on every call (``kda_prefill_fixed_workloads.
  recurrent_kda_call_kwargs``), with ``backend="vibecuda"`` for the candidate
  leg and the pinned source's default public path (the frozen CAKE family;
  this revision predates the CuTe-DSL backend) for the baseline leg. Both
  legs receive IDENTICAL public kwargs and physical inputs — the same
  tensors the fixed harness hands the candidate (cu_seqlens/seq_order None
  for non-packed workloads; packed tensors for packed workloads). No leg
  may be presented inputs the fixed wrapper does not construct.
* Isolation: candidate and baseline revisions cannot coexist in one process,
  so each leg runs in a clean subprocess with PYTHONPATH pinned to its own
  checkout and an import probe proving the resolved ``flashinfer`` tree.
* Aggregate: rejects any missing, extra, reordered, or seed-mismatched row by
  comparing ordered indices, workload descriptors, and per-tensor SHA-256
  fingerprints of every constructed input (including cu_seqlens/seq_order),
  then reports per-workload latency/speedup plus arithmetic-mean (the fixed
  benchmark's ``arithmetic_mean_all_shapes`` aggregation) and geometric-mean
  speedups.

Usage (one command runs both legs and aggregates):

    python benchmarks/bench_recurrent_kda_prefill_fixed_protocol.py \
        --fastimpl-source /path/to/pinned/flashinfer

Phases can be staged for wall-clock-limited shells via
``--only <candidate|baseline|aggregate>``; per-leg JSON artifacts under
``--results-dir`` are reused by later phases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from importlib import util as importlib_util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_MODULE_PATH = REPO_ROOT / "benchmarks" / "kda_prefill_fixed_workloads.py"
PINNED_REVISION = "ee3fda10e0b9d82b71bda534bc2954d3176e6ded"
WARMUP_ITERS = 5
# SKB metadata leaves the CUPTI repeat count at its default 10 and the
# toolbox profiler reports the MEDIAN of those samples; reproduce both.
REPEAT_ITERS = 10

# Executed-reference-symbol evidence (port reference audit). For every
# scheduled (family, variant) the table below pins the public path from the
# repository JIT registry down to the launched CUDA device kernel, derived
# from the csrc binding sources of the executed revision:
#   public flashinfer.kda.recurrent_kda
#     -> kda_prefill._get_flash_kda_prefill_module(variant, target)  (cake)
#        / kda_vibecuda._get_vibecuda_prefill_module(target)         (vibecuda)
#     -> JIT module URI (jit/flash_kda.py registry)
#     -> TVM-FFI typed export (`run` / `run_m64` / ...)
#     -> C++ launcher (flashinfer::flash_kda::Run* /
#        flashinfer::vibecuda_flashkda::Run*)
#     -> CUDA device kernel name launched by that binding TU
# Each leg additionally captures the device kernels that ACTUALLY executed in
# one untimed profiled call per workload and asserts the expected scheduled
# kernel is among them, so the artifacts prove — not just claim — which
# reference symbol was timed.
EXECUTED_SYMBOL_TABLE = {
    ("cake", "m64"): {
        "ffi_export": "run",
        "cxx_launcher": "flashinfer::flash_kda::RunM64",
        "binding_source": "csrc/kda/flashkda_bf16_fused_m64_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m64",
    },
    ("cake", "m128"): {
        "ffi_export": "run",
        "cxx_launcher": "flashinfer::flash_kda::RunM128",
        "binding_source": "csrc/kda/flashkda_bf16_fused_m128_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m128",
    },
    ("cake", "m128_n16"): {
        "ffi_export": "run",
        "cxx_launcher": "flashinfer::flash_kda::RunM128N16",
        "binding_source": "csrc/kda/cake_flashkda_bf16_fused_m128_n16_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m128",
    },
    ("cake", "persistent_m128"): {
        "ffi_export": "run",
        "cxx_launcher": "flashinfer::flash_kda::RunPersistentM128",
        "binding_source": "csrc/kda/cake_flashkda_bf16_persistent_m128_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_persistent_m128",
    },
    ("cake", "small_bh_m128"): {
        "ffi_export": "run",
        "cxx_launcher": "flashinfer::flash_kda::RunSmallBHM128",
        "binding_source": "csrc/kda/cake_flashkda_bf16_small_bh_m128_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_small_bh_m128",
    },
    ("vibecuda", "m64"): {
        "ffi_export": "run_m64",
        "cxx_launcher": "flashinfer::vibecuda_flashkda::RunM64",
        "binding_source": "csrc/kda/vibecuda_flashkda_bf16_fused_m64_binding.cu",
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m64",
    },
    ("vibecuda", "m128"): {
        "ffi_export": "run_m128",
        "cxx_launcher": "flashinfer::vibecuda_flashkda::RunM128",
        "binding_source": (
            "csrc/kda/vibecuda_flashkda_bf16_fused_m128_slab_binding.cu"
        ),
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m128",
    },
    ("vibecuda", "m128_split"): {
        "ffi_export": "run_m128_split",
        "cxx_launcher": "flashinfer::vibecuda_flashkda::RunM128Split",
        "binding_source": (
            "csrc/kda/vibecuda_flashkda_bf16_fused_m128_binding.cu + "
            "csrc/kda/vibecuda_flashkda_bf16_fused_m128_slab_binding.cu"
        ),
        "expected_device_kernel": "kernel_flashkda_bf16_fused_m128",
    },
    ("vibecuda", "persistent"): {
        "ffi_export": "run_persistent_m128",
        "cxx_launcher": "flashinfer::vibecuda_flashkda::RunPersistentM128",
        "binding_source": (
            "csrc/kda/vibecuda_flashkda_bf16_persistent_m128_binding.cu"
        ),
        "expected_device_kernel": "kernel_flashkda_bf16_persistent_m128",
    },
}


def _capture_device_kernels(call) -> list[str]:
    """Return the CUDA device kernel names executed by one untimed call."""

    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names: set[str] = set()
    cuda_device = getattr(torch.autograd, "DeviceType", None)
    for event in prof.events():
        device_type = getattr(event, "device_type", None)
        if cuda_device is not None and device_type == cuda_device.CUDA:
            names.add(event.name)
    if not names:
        # Kineto variants exposing only aggregated events.
        for event in prof.key_averages():
            if event.self_device_time_total > 0:
                names.add(event.key)
    return sorted(names)


def _executed_symbol_evidence(
    family: str, variant: str, target: str, call, jit_flash_kda
) -> dict:
    """Bind a scheduled route to its executed symbol chain and verify it.

    Raises if the expected scheduled device kernel was not observed executing
    in the profiled verification call — that mismatch is exactly the audit
    failure mode this evidence exists to catch.
    """

    key = (family, variant)
    if key not in EXECUTED_SYMBOL_TABLE:
        raise RuntimeError(
            f"no executed-symbol table entry for route {family}:{variant}; "
            "extend EXECUTED_SYMBOL_TABLE for this route before timing"
        )
    table = EXECUTED_SYMBOL_TABLE[key]
    if family == "cake":
        uri = (
            jit_flash_kda.get_flash_kda_uri(variant, target)
            if hasattr(jit_flash_kda, "get_flash_kda_uri")
            else None
        )
    else:
        uri = (
            jit_flash_kda.get_vibecuda_flash_kda_uri(target)
            if hasattr(jit_flash_kda, "get_vibecuda_flash_kda_uri")
            else None
        )
    observed = _capture_device_kernels(call)
    expected_kernel = table["expected_device_kernel"]
    found = any(expected_kernel in name for name in observed)
    evidence = {
        "family": family,
        "variant": variant,
        "target": target,
        "jit_module_uri": uri,
        "ffi_export": table["ffi_export"],
        "cxx_launcher": table["cxx_launcher"],
        "binding_source": table["binding_source"],
        "expected_device_kernel": expected_kernel,
        "device_kernels_observed_at_runtime": observed,
        "expected_kernel_observed_executing": found,
    }
    if not found:
        raise RuntimeError(
            f"executed-symbol audit failed for route {family}:{variant}: "
            f"expected scheduled device kernel {expected_kernel!r} was not "
            f"observed executing in the profiled call; observed={observed}"
        )
    return evidence

TIMING_POLICY = {
    "timer": "CUPTI activity tracing (flashinfer.testing.bench_gpu_time, "
    "enable_cupti=True)",
    "warmup": f"{WARMUP_ITERS} untimed iterations (dry_run_iters=5)",
    "repeats": f"{REPEAT_ITERS} timed iterations (repeat_iters=10, the SKB "
    "metadata default)",
    "cold_l2": "L2 flush before every timed iteration (cold_l2_cache=True)",
    "cuda_graph": "disabled (use_cuda_graph=False), eager execution",
    "sample_aggregation": "median of the 10 per-iteration samples (matching "
    "the toolbox profiler's reported statistic; arithmetic mean also "
    "recorded)",
    "aggregate_aggregation": "arithmetic mean over all shapes of the "
    "per-shape median-latency speedups (matches the fixed benchmark's "
    "arithmetic_mean_all_shapes); geometric mean also recorded",
}

LIFECYCLE_POLICY = {
    "callable": "flashinfer.kda.recurrent_kda public API per timed call",
    "inputs_per_case": "constructed once per workload from the case's CUDA "
    "generator seed (model_seed = 377300000 + index); the identical tensor "
    "objects feed all warmup and timed calls",
    "plan_setup": "no explicit plan call; first-call JIT/workspace setup is "
    "absorbed by the untimed observation call plus the 5 warmup iterations. "
    "Each leg receives its own independently owned persistent "
    "RecurrentKDAPrefillWorkspace (constructed inside that leg's clean "
    "subprocess), matching the frozen wrapper's per-implementation "
    "persistent-workspace lifecycle",
    "output": "torch.empty_like(v) preallocated per case before warmup, "
    "reused as the in-place output destination by every call",
    "mutable_state": "initial_state tensor is consumed as input and fully "
    "rewritten in place by every call; no reset inside the timed region",
    "preprocessing": "cu_seqlens/seq_order built once per case outside the "
    "timed closure; the timed closure is exactly one recurrent_kda call. "
    "Both legs consume identical public kwargs and identical physical "
    "inputs: the exact fixed-contract tensors (cu_seqlens/seq_order None "
    "for non-packed workloads, packed tensors for packed workloads); no "
    "leg-specific input presentation is applied.",
}


def _load_workload_module(path: Path):
    spec = importlib_util.spec_from_file_location(
        "kda_prefill_fixed_workloads", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload module {path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return f"unavailable({proc.stderr.strip()[:120]})"
    return proc.stdout.strip()


def _probe_flashinfer(expected_root: Path, env: dict) -> str:
    code = (
        "import os, sys, flashinfer\n"
        "print(os.path.realpath(flashinfer.__file__))\n"
        f"sys.exit(0 if os.path.realpath(flashinfer.__file__).startswith("
        f"{str(expected_root)!r}) else 3)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        env=env,
        cwd=str(expected_root),
        capture_output=True,
        text=True,
    )
    resolved = proc.stdout.strip()
    if proc.returncode != 0 or not resolved:
        raise RuntimeError(
            f"import probe failed for {expected_root}:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr[-1500:]}"
        )
    return resolved


def _parse_case_indices(expr: str | None) -> set[int] | None:
    if not expr:
        return None
    indices: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        elif part:
            indices.add(int(part))
    return indices


def run_leg(args: argparse.Namespace) -> None:
    import torch

    workload_module = _load_workload_module(Path(args.workload_module).resolve())
    workloads = workload_module.FIXED_KDA_WORKLOADS
    selected = _parse_case_indices(args.case_indices)
    workloads = tuple(w for w in workloads if selected is None or w.index in selected)

    import flashinfer

    resolved_pkg = str(Path(flashinfer.__file__).resolve())
    if args.expect_import and not resolved_pkg.startswith(args.expect_import):
        raise RuntimeError(
            f"flashinfer resolved to {resolved_pkg}; expected prefix "
            f"{args.expect_import!r}"
        )
    from flashinfer import kda_prefill as _kda_prefill
    from flashinfer.jit import flash_kda as _jit_flash_kda
    from flashinfer.testing import bench_gpu_time

    prefill_workspace = None
    if args.with_prefill_workspace:
        # The fixed benchmark's fast_impl denominator wraps the pinned public
        # prefill path with a persistent RecurrentKDAPrefillWorkspace (the
        # task contract's "RecurrentKDAPrefillWorkspace-equivalent persistent
        # workspace behavior"). Supplying it keeps the baseline on the
        # nonpersistent direct route family with a capacity-grown, reusable
        # descriptor workspace, instead of the dispatcher's stream workspace.
        prefill_workspace = _kda_prefill.RecurrentKDAPrefillWorkspace(
            torch.device("cuda")
        )
        print(f"[{args.run_leg}] persistent RecurrentKDAPrefillWorkspace enabled")

    print(f"[{args.run_leg}] flashinfer resolved: {resolved_pkg}")
    print(
        f"[{args.run_leg}] {len(workloads)} workloads, "
        f"warmup={WARMUP_ITERS} iters={REPEAT_ITERS} timer=CUPTI "
        "cold_l2=True cuda_graph=False"
    )

    original_get_module = _kda_prefill._get_flash_kda_prefill_module
    recorded_routes: list[tuple] = []

    def recording_get_module(variant, target):
        recorded_routes.append(("cake", variant, target))
        return original_get_module(variant, target)

    vibecuda_module = None
    original_vibecuda_get = None
    recorded_vibecuda: list[tuple] = []
    if importlib_util.find_spec("flashinfer.kda_vibecuda") is not None:
        vibecuda_module = importlib.import_module("flashinfer.kda_vibecuda")
        original_vibecuda_get = vibecuda_module._get_vibecuda_prefill_module

        def recording_vibecuda_get(target):
            module = original_vibecuda_get(target)

            class _Recorder:
                def __init__(self, inner):
                    self._inner = inner

                def __getattr__(self, name):
                    attribute = getattr(self._inner, name)
                    variants = {
                        "run_m64": "m64",
                        "run_m128": "m128",
                        "run_m128_split": "m128_split",
                        "run_persistent_m128": "persistent",
                    }
                    variant = variants.get(name)
                    if variant is None:
                        return attribute

                    def recorded(*a, **kw):
                        recorded_vibecuda.append(("vibecuda", variant, target))
                        return attribute(*a, **kw)

                    return recorded

            return _Recorder(module)

    rows = []
    route_table: dict[int, str] = {}
    for workload in workloads:
        inputs = workload_module.build_case_inputs(workload)
        call_kwargs = workload_module.recurrent_kda_call_kwargs(inputs)
        if args.backend is not None:
            call_kwargs["backend"] = args.backend
        if prefill_workspace is not None:
            call_kwargs["prefill_workspace"] = prefill_workspace

        # Fingerprint the pristine constructed inputs BEFORE any call mutates
        # initial_state in place — the aggregate's seed-match contract keys on
        # these pre-call contents.
        fingerprint = workload_module.fingerprint_case_inputs(inputs)

        # Untimed observation call with route recording active. It also
        # absorbs first-call JIT/workspace setup outside the timed region.
        recorded_routes.clear()
        recorded_vibecuda.clear()
        _kda_prefill._get_flash_kda_prefill_module = recording_get_module
        if vibecuda_module is not None:
            vibecuda_module._get_vibecuda_prefill_module = recording_vibecuda_get
        try:
            flashinfer.recurrent_kda(**call_kwargs)
            torch.cuda.synchronize()
        finally:
            _kda_prefill._get_flash_kda_prefill_module = original_get_module
            if vibecuda_module is not None:
                vibecuda_module._get_vibecuda_prefill_module = original_vibecuda_get

        observed = list(recorded_routes) + list(recorded_vibecuda)
        families = {route[0] for route in observed}
        if len(observed) != 1 or len(families) != 1:
            # The default public path may use the stream workspace without a
            # persistent-module route; only CAKE/VibeCUDA module routes count.
            raise RuntimeError(
                f"case {workload.index}: expected exactly one prefill module "
                f"route during the observation call, got {observed}"
            )
        family, variant, target = observed[0]
        route_table[workload.index] = f"{family}:{variant}({target})"

        def timed_call():
            flashinfer.recurrent_kda(**call_kwargs)

        samples_ms = [
            float(value)
            for value in bench_gpu_time(
                timed_call,
                dry_run_iters=WARMUP_ITERS,
                repeat_iters=REPEAT_ITERS,
                enable_cupti=True,
                cold_l2_cache=True,
                use_cuda_graph=False,
            )
        ]
        if not samples_ms:
            raise RuntimeError(f"case {workload.index}: no timed samples")
        ordered = sorted(samples_ms)
        mean_ms = sum(samples_ms) / len(samples_ms)
        if len(ordered) % 2:
            median_ms = ordered[len(ordered) // 2]
        else:
            half = len(ordered) // 2
            median_ms = 0.5 * (ordered[half - 1] + ordered[half])
        # Executed-reference-symbol evidence: one untimed profiled call that
        # records the device kernels which ACTUALLY executed for this
        # workload, and asserts the scheduled kernel of the recorded route is
        # among them. Untimed, so it cannot perturb the measured samples.
        asserted_evidence = _executed_symbol_evidence(
            family, variant, target, timed_call, _jit_flash_kda
        )
        row = {
            **workload_module.workload_descriptor(workload),
            "route": route_table[workload.index],
            "backend_argument": args.backend,
            "fingerprints": fingerprint,
            "executed_symbols": asserted_evidence,
            "num_samples": len(samples_ms),
            "statistic": f"median of {len(samples_ms)} CUPTI samples "
            "(SKB/toolbox convention; mean recorded alongside)",
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "samples_ms": samples_ms,
            "input_fingerprint_sha256": hashlib.sha256(
                json.dumps(fingerprint, sort_keys=True).encode()
            ).hexdigest(),
        }
        rows.append(row)
        print(
            f"[{args.run_leg}] #{workload.index:02d} {workload.name:<40} "
            f"route={row['route']:<24} median={median_ms * 1000.0:>10.3f}us "
            f"n={len(samples_ms)} "
            f"executed={asserted_evidence['expected_device_kernel']}"
        )
        del inputs, call_kwargs
        torch.cuda.empty_cache()

    result = {
        "leg": args.run_leg,
        "backend_argument": args.backend,
        "prefill_workspace": bool(args.with_prefill_workspace),
        "flashinfer_resolved": resolved_pkg,
        "timing_policy": TIMING_POLICY,
        "lifecycle_policy": LIFECYCLE_POLICY,
        "provenance": {
            "expect_import": args.expect_import,
            "git_head": _git(Path(args.expect_import), "rev-parse", "HEAD")
            if args.expect_import
            else None,
            "git_dirty": _git(
                Path(args.expect_import),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--ignore-submodules",
            )
            if args.expect_import
            else None,
        },
        "rows": rows,
    }
    out_path = Path(args.json)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.run_leg}] wrote {out_path} ({len(rows)} rows)")


def _assert_row_contract(candidate: list[dict], baseline: list[dict]) -> None:
    """Reject missing, extra, reordered, or seed-mismatched rows."""

    def explain(c_row, b_row):
        return (
            f"candidate={c_row.get('name') if c_row else '<missing>'} "
            f"baseline={b_row.get('name') if b_row else '<missing>'}"
        )

    if len(candidate) != len(baseline):
        raise RuntimeError(
            "row-count mismatch between legs: "
            f"candidate={len(candidate)} baseline={len(baseline)} — a missing "
            "or extra workload invalidates the aggregate"
        )
    for position, (c_row, b_row) in enumerate(zip(candidate, baseline)):
        if c_row["index"] != position or b_row["index"] != position:
            raise RuntimeError(
                f"reordered rows at position {position}: candidate index "
                f"{c_row['index']} baseline index {b_row['index']}"
            )
        for field in ("sequence_lengths", "heads", "packed", "model_seed", "name"):
            if c_row[field] != b_row[field]:
                raise RuntimeError(
                    f"workload descriptor mismatch at position {position} "
                    f"field {field}: {explain(c_row, b_row)}"
                )
        if (
            c_row["input_fingerprint_sha256"]
            != b_row["input_fingerprint_sha256"]
        ):
            raise RuntimeError(
                f"input fingerprint mismatch at position {position} — the "
                "legs did not consume byte-identical inputs (seed/construction "
                f"mismatch): {explain(c_row, b_row)}"
            )


def aggregate(args: argparse.Namespace) -> None:
    candidate = json.loads(Path(args.candidate_json).read_text())
    baseline = json.loads(Path(args.baseline_json).read_text())
    c_rows, b_rows = candidate["rows"], baseline["rows"]
    _assert_row_contract(c_rows, b_rows)

    gate_reference = None
    if args.gate_reference_json:
        gate_reference = {
            row["index"]: row
            for row in json.loads(Path(args.gate_reference_json).read_text())
        }

    rows = []
    speedups = []
    print(
        f"{'workload':<42} {'fast_impl':>11} {'vibecuda':>11} "
        f"{'speedup':>9}  routes"
    )
    for c_row, b_row in zip(c_rows, b_rows):
        cand_ms = float(c_row["median_ms"])
        base_ms = float(b_row["median_ms"])
        speedup = base_ms / cand_ms
        speedups.append(speedup)
        entry = {
            "index": c_row["index"],
            "name": c_row["name"],
            "model_seed": c_row["model_seed"],
            "per_shape_statistic": c_row["statistic"],
            "fastimpl_median_ms": base_ms,
            "fastimpl_mean_ms": float(b_row["mean_ms"]),
            "fastimpl_route": b_row["route"],
            "fastimpl_executed_symbols": b_row.get("executed_symbols"),
            "candidate_median_ms": cand_ms,
            "candidate_mean_ms": float(c_row["mean_ms"]),
            "candidate_route": c_row["route"],
            "candidate_executed_symbols": c_row.get("executed_symbols"),
            "speedup_vs_fastimpl": speedup,
        }
        if gate_reference is not None and c_row["index"] in gate_reference:
            gate = gate_reference[c_row["index"]]
            entry["gate_fastimpl_latency_ms"] = gate.get("fast_impl_latency_ms")
            entry["gate_candidate_latency_ms"] = gate.get("candidate_latency_ms")
            entry["baseline_over_gate_baseline"] = (
                base_ms / float(gate["fast_impl_latency_ms"])
                if gate.get("fast_impl_latency_ms")
                else None
            )
            entry["candidate_over_gate_candidate"] = (
                cand_ms / float(gate["candidate_latency_ms"])
                if gate.get("candidate_latency_ms")
                else None
            )
        rows.append(entry)
        print(
            f"{c_row['name']:<42} {base_ms * 1000.0:>9.3f}us "
            f"{cand_ms * 1000.0:>9.3f}us {speedup:>8.4f}x  "
            f"{b_row['route']} -> {c_row['route']}"
        )

    arith = sum(speedups) / len(speedups)
    geo = math.exp(sum(math.log(v) for v in speedups) / len(speedups))
    executed_symbol_audit = all(
        (b_row.get("executed_symbols") or {}).get(
            "expected_kernel_observed_executing"
        )
        and (c_row.get("executed_symbols") or {}).get(
            "expected_kernel_observed_executing"
        )
        for b_row, c_row in zip(b_rows, c_rows)
    )
    reference_provenance = {
        "configured_fast_impl": (
            "flashinfer_b200 (the task's configured fast_impl denominator), "
            "timed directly from its pinned FlashInfer source revision "
            f"{PINNED_REVISION} (PR #4571 squash merge, merged 2026-08-19)"
        ),
        "baseline_checkout": str(
            Path(baseline.get("flashinfer_resolved", "")).parent
        ),
        "baseline_git_head": (baseline.get("provenance") or {}).get("git_head"),
        "baseline_git_dirty": (baseline.get("provenance") or {}).get(
            "git_dirty"
        ),
        "executed_public_api": "flashinfer.kda.recurrent_kda(...)",
        "executed_symbol_chain": (
            "recurrent_kda -> kda_prefill._run_flash_kda_prefill -> "
            "_get_flash_kda_prefill_module(variant, target) -> JIT module "
            "per_workload[*].fastimpl_executed_symbols.jit_module_uri -> "
            "TVM-FFI export .ffi_export -> C++ launcher .cxx_launcher -> "
            "CUDA device kernel .expected_device_kernel, each OBSERVED "
            "executing in one untimed profiled call per workload "
            "(.device_kernels_observed_at_runtime, kineto/CUPTI capture); "
            "the leg aborts when the scheduled kernel is absent"
        ),
        "executed_symbol_audit_passed_all_workloads": executed_symbol_audit,
    }
    report = {
        "comparison": (
            "flashinfer recurrent_kda prefill, fixed 18-workload protocol: "
            "integrated backend='vibecuda' vs the task's configured fast_impl "
            "denominator (flashinfer_b200) timed directly from its pinned "
            "FlashInfer source revision "
            f"{PINNED_REVISION} (PR #4571 squash merge)"
        ),
        "workload_count": len(rows),
        "per_workload": rows,
        "aggregate": {
            "workloads": len(rows),
            "arithmetic_mean_speedup": float(arith),
            "geometric_mean_speedup": float(geo),
            "arithmetic_mean_aggregation": "arithmetic_mean_all_shapes "
            "(matches the fixed benchmark aggregate)",
        },
        "timing_policy": TIMING_POLICY,
        "lifecycle_policy": LIFECYCLE_POLICY,
        "reference_provenance": reference_provenance,
        "candidate_leg": {
            "flashinfer_resolved": candidate["flashinfer_resolved"],
            "prefill_workspace": candidate.get("prefill_workspace"),
            "provenance": candidate["provenance"],
        },
        "baseline_leg": {
            "flashinfer_resolved": baseline["flashinfer_resolved"],
            "prefill_workspace": baseline.get("prefill_workspace"),
            "provenance": baseline["provenance"],
        },
    }
    out_path = Path(args.results_dir) / "aggregate_fixed_protocol_vs_fastimpl.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"workloads measured: {len(rows)}")
    print(
        f"aggregate vs fast_impl denominator: arithmetic mean {arith:.4f}x, "
        f"geometric mean {geo:.4f}x over {len(rows)} workloads"
    )
    print(
        "executed-symbol audit: "
        f"{'PASS (scheduled kernel observed executing on every workload, both legs)' if executed_symbol_audit else 'FAIL'}"
    )
    print(
        "fast_impl executed reference: "
        f"{reference_provenance['baseline_git_head']} via "
        f"{reference_provenance['executed_public_api']}; per-workload chain "
        "in per_workload[*].fastimpl_executed_symbols"
    )
    print(f"candidate leg results: {args.candidate_json}")
    print(f"baseline leg results: {args.baseline_json}")
    print(f"aggregate results: {out_path}")
    if gate_reference is not None:
        deltas_b = [
            abs(row["baseline_over_gate_baseline"] - 1.0)
            for row in rows
            if row.get("baseline_over_gate_baseline")
        ]
        deltas_c = [
            abs(row["candidate_over_gate_candidate"] - 1.0)
            for row in rows
            if row.get("candidate_over_gate_candidate")
        ]
        print(
            "gate-latency calibration (|ratio-1|): baseline max "
            f"{max(deltas_b):.4f} mean {sum(deltas_b) / len(deltas_b):.4f}; "
            f"candidate max {max(deltas_c):.4f} "
            f"mean {sum(deltas_c) / len(deltas_c):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastimpl-source", type=Path, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "fixed_protocol_vs_fastimpl",
    )
    parser.add_argument(
        "--only",
        choices=("candidate", "baseline", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--force-rebench", action="store_true")
    parser.add_argument(
        "--gate-reference-json",
        default=None,
        help="Optional per-shape fixed-gate latency reference for calibration "
        "columns (list of {index, fast_impl_latency_ms, "
        "candidate_latency_ms}).",
    )
    parser.add_argument(
        "--run-leg",
        choices=("candidate", "baseline"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--expect-import", default=None)
    parser.add_argument(
        "--with-prefill-workspace",
        action="store_true",
        help="Pass a persistent RecurrentKDAPrefillWorkspace to every call "
        "(the pinned fast_impl denominator's workspace policy).",
    )
    parser.add_argument("--workload-module", default=str(WORKLOAD_MODULE_PATH))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.run_leg is not None:
        if args.json is None:
            parser.error("--run-leg requires --json")
        run_leg(args)
        return

    fastimpl_source = (
        args.fastimpl_source.resolve()
        if args.fastimpl_source
        else Path(
            os.environ.get("FLASHINFER_FASTIMPL_SOURCE", "")
        ).resolve()
        if os.environ.get("FLASHINFER_FASTIMPL_SOURCE")
        else None
    )
    if fastimpl_source is None:
        parser.error("--fastimpl-source (or FLASHINFER_FASTIMPL_SOURCE) is required")
    head = _git(fastimpl_source, "rev-parse", "HEAD")
    if head != PINNED_REVISION:
        raise RuntimeError(
            f"pinned fast_impl source must be at {PINNED_REVISION}, got {head}"
        )
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    candidate_json = args.results_dir / "candidate_vibecuda_fixed18.json"
    baseline_json = args.results_dir / "baseline_fastimpl_fixed18.json"

    def run_subprocess_leg(*, leg: str, backend, root: Path, out: Path) -> None:
        if out.exists() and not args.force_rebench:
            print(f"[{leg}] reusing {out}")
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        resolved = _probe_flashinfer(root, env)
        print(f"[{leg}] probe: flashinfer resolves to {resolved}")
        command = [
            sys.executable,
            "-P",
            str(Path(__file__).resolve()),
            "--run-leg",
            leg,
            "--expect-import",
            str(root),
            "--workload-module",
            str(WORKLOAD_MODULE_PATH),
            "--json",
            str(out),
        ]
        if backend is not None:
            command.extend(["--backend", backend])
        # Every leg (candidate and baseline) receives its own independently
        # owned persistent RecurrentKDAPrefillWorkspace, constructed inside
        # that leg's clean subprocess — matching the frozen wrapper, which
        # gives each implementation its own persistent workspace.
        command.append("--with-prefill-workspace")
        if args.case_indices:
            command.extend(["--case-indices", args.case_indices])
        print(f"[{leg}] running {' '.join(command)}")
        proc = subprocess.run(command, env=env, cwd=str(root), check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"[{leg}] leg subprocess exited {proc.returncode}")

    if args.only in ("candidate", "all"):
        run_subprocess_leg(
            leg="candidate",
            backend="vibecuda",
            root=REPO_ROOT,
            out=candidate_json,
        )
    if args.only in ("baseline", "all"):
        run_subprocess_leg(
            leg="baseline",
            backend=None,
            root=fastimpl_source,
            out=baseline_json,
        )
    if args.only in ("aggregate", "all"):
        args.candidate_json = str(candidate_json)
        args.baseline_json = str(baseline_json)
        if not candidate_json.is_file() or not baseline_json.is_file():
            raise RuntimeError("per-leg JSON artifacts missing")
        aggregate(args)


if __name__ == "__main__":
    main()
