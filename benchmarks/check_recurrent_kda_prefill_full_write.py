#!/usr/bin/env python3
"""Untimed sentinel-prefill / full-write check over the fixed 18-workload
protocol (``kda_prefill_fixed_workloads.py``).

The timed legs validate throughput but not write coverage: an output left
``torch.empty``-garbage, a final state that is only partially rewritten, or a
stale coordinate outside the validated region could all pass a numeric
comparison if the reference made the same omission. For each of the 18 fixed
workloads this check proves, without timing, that every location the
benchmark's validation reads is initialized and fully overwritten by each
timed leg (the ``vibecuda`` candidate and the pinned fast_impl baseline):

* output tensor: pre-filled with NaN before each of two consecutive calls;
  after every call every element must be non-NaN (full-write proof for every
  byte the output validation reads).
* initial_state (the in-place final-state output): prescribed with a
  distinctive *valid* constant bf16 pattern before call 1 (it is a real input,
  so it must stay finite); after the call every element must differ from the
  pattern. Call 2 must fully rewrite it again — every element must differ
  from call 1's final state. Rare bf16 value collisions are tolerated at
  0.05%.
  NOTE: a same-input re-call cannot prove this. The KDA gate decay is
  exp(gate) <= e^-5 per token (lower_bound=-5), so after hundreds of tokens
  the initial state's contribution underflows bf16 and the final state is
  legitimately input-state-independent; an unperturbed call 2 reproduces
  call 1's final state bit-for-bit by physics, not by staleness. Call 2
  therefore perturbs v at the sequence-final token(s) (which enter the
  final state with decay factor 1 as a dense beta*residual(x)k rank-1
  update touching every element), so any state element the kernel failed
  to rewrite stays bit-equal to call 1 and is detected.
* twin determinism: a second, identically constructed case with the same
  constant initial state must produce bitwise-identical final state and
  bitwise-identical output — any slot-carried staleness or per-call
  nondeterminism breaks this.
* out-of-bounds guards: output and initial_state are allocated as views over
  storage with a NaN guard tail; the guard region must survive every call
  bit-exactly, proving the kernels wrote only the validated regions.

Run per leg:

    # Candidate leg (this checkout's integrated backend):
    python benchmarks/check_recurrent_kda_prefill_full_write.py \
        --backend vibecuda --expect-import <this-checkout-root>
    # Pinned fast_impl baseline leg (default public path at the pinned rev):
    PYTHONPATH=<pinned-source> python benchmarks/check_recurrent_kda_prefill_full_write.py \
        --expect-import <pinned-source>
"""

from __future__ import annotations

import argparse
import sys
from importlib import util as importlib_util
from pathlib import Path

import torch

GUARD_ELEMENTS = 4096
STATE_SENTINEL = 0.001953125  # 2**-9, exact bf16 constant, valid input state
MIN_CHANGED_FRACTION = 0.9995
WORKLOAD_MODULE_PATH = (
    Path(__file__).resolve().with_name("kda_prefill_fixed_workloads.py")
)


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


def _guarded_like(tensor: torch.Tensor, fill: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tensor-shaped view filled with `fill`, NaN guard tail view)."""

    numel = tensor.numel()
    storage = torch.full(
        (numel + GUARD_ELEMENTS,), float("nan"), dtype=tensor.dtype, device=tensor.device
    )
    view = storage[:numel].view(tensor.shape)
    view.fill_(fill)
    return view, storage[numel:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--expect-import", default=None)
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--workload-module", default=str(WORKLOAD_MODULE_PATH))
    parser.add_argument(
        "--with-prefill-workspace",
        action="store_true",
        help="Pass a persistent RecurrentKDAPrefillWorkspace to every call "
        "(the pinned fast_impl baseline leg's timed-lifecycle policy).",
    )
    args = parser.parse_args()

    workload_module = _load_workload_module(Path(args.workload_module).resolve())
    import flashinfer

    resolved = str(Path(flashinfer.__file__).resolve())
    if args.expect_import and not resolved.startswith(args.expect_import):
        raise RuntimeError(
            f"flashinfer resolved to {resolved}; expected prefix "
            f"{args.expect_import!r}"
        )
    print(f"flashinfer resolved: {resolved}")

    prefill_workspace = None
    if args.with_prefill_workspace:
        from flashinfer import kda_prefill as _kda_prefill

        prefill_workspace = _kda_prefill.RecurrentKDAPrefillWorkspace(
            torch.device("cuda")
        )
        print("persistent RecurrentKDAPrefillWorkspace enabled")

    selected = None
    if args.case_indices:
        selected = set()
        for part in args.case_indices.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                selected.update(range(int(lo), int(hi) + 1))
            else:
                selected.add(int(part))
    workloads = [
        workload
        for workload in workload_module.FIXED_KDA_WORKLOADS
        if selected is None or workload.index in selected
    ]

    nan = float("nan")
    failures: list[str] = []
    print(
        f"{'workload':<42} {'out_nan c1/c2':>14} {'state_full c1/c2':>17} "
        f"{'twin_bits':>9} {'guards':>7}  status"
    )
    for workload in workloads:
        check = {"ok": True, "detail": {}}

        def build_twin_runs():
            """One fully checked two-call sequence on a fresh construction."""

            inputs = workload_module.build_case_inputs(workload)
            output, out_guard = _guarded_like(inputs["output"], nan)
            state, state_guard = _guarded_like(inputs["initial_state"], STATE_SENTINEL)
            inputs["output"] = output
            inputs["initial_state"] = state
            call_kwargs = workload_module.recurrent_kda_call_kwargs(inputs)
            if args.backend is not None:
                call_kwargs["backend"] = args.backend
            if prefill_workspace is not None:
                call_kwargs["prefill_workspace"] = prefill_workspace

            def perturb_sequence_final_tokens():
                # The final token of each sequence enters its final-state slot
                # with decay factor 1 as a dense beta*residual(x)k update; a
                # +2.0 shift forces every final-state element to change on a
                # faithfully recomputed call (see module docstring).
                if inputs["cu_seqlens"] is None:
                    idx = [inputs["v"].shape[1] - 1]
                else:
                    idx = (inputs["cu_seqlens"][1:] - 1).tolist()
                inputs["v"][:, idx, :, :] += 2.0

            def call(perturb: bool):
                output.fill_(nan)
                if perturb:
                    perturb_sequence_final_tokens()
                flashinfer.recurrent_kda(**call_kwargs)
                torch.cuda.synchronize()
                return {
                    "out_nan": int(torch.isnan(output).sum().item()),
                    "state": state.clone(),
                    "output": output.clone(),
                    "out_guard_intact": bool(torch.isnan(out_guard).all().item()),
                    "state_guard_intact": bool(
                        torch.isnan(state_guard).all().item()
                    ),
                }

            first = call(False)
            second = call(True)
            del inputs, call_kwargs
            return first, second

        first_a, second_a = build_twin_runs()
        first_b, second_b = build_twin_runs()

        numel_state = first_a["state"].numel()
        changed_c1 = (
            int((first_a["state"] != STATE_SENTINEL).sum().item()) / numel_state
        )
        changed_c2 = (
            int((second_a["state"] != first_a["state"]).sum().item()) / numel_state
        )
        out_nan = (first_a["out_nan"], second_a["out_nan"])
        # Twin runs consumed identical inputs (same seeds, same constant
        # sentinel state): both call-1 and call-2 results must be bitwise
        # equal across the twins.
        twin_ok = bool(
            torch.equal(first_a["state"], first_b["state"])
            and torch.equal(second_a["state"], second_b["state"])
            and torch.equal(first_a["output"], first_b["output"])
            and torch.equal(second_a["output"], second_b["output"])
        )
        guards_ok = bool(
            first_a["out_guard_intact"]
            and second_a["out_guard_intact"]
            and first_a["state_guard_intact"]
            and second_a["state_guard_intact"]
            and first_b["out_guard_intact"]
            and first_b["state_guard_intact"]
        )
        finite_ok = bool(
            torch.isfinite(second_a["output"]).all().item()
            and torch.isfinite(second_a["state"]).all().item()
        )
        ok = (
            out_nan == (0, 0)
            and changed_c1 >= MIN_CHANGED_FRACTION
            and changed_c2 >= MIN_CHANGED_FRACTION
            and twin_ok
            and guards_ok
            and finite_ok
        )
        if not ok:
            failures.append(workload.name)
        print(
            f"{workload.name:<42} {str(out_nan):>14} "
            f"{changed_c1:.5f}/{changed_c2:.5f}   {str(twin_ok):>9} "
            f"{str(guards_ok):>7}  {'PASS' if ok else 'FAIL'}"
        )
        del first_a, second_a, first_b, second_b
        torch.cuda.empty_cache()

    if failures:
        print(f"FAIL: {len(failures)}/{len(workloads)} workloads: {failures}")
        raise SystemExit(1)
    print(
        f"PASS: all {len(workloads)} fixed workloads prove fully initialized, "
        "fully overwritten, twin-deterministic output+final-state writes with "
        "intact out-of-bounds guards"
    )


if __name__ == "__main__":
    main()
