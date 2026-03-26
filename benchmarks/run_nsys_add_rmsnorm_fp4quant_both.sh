#!/usr/bin/env bash
# Generate both nsys reports (PDL off and PDL on) for add_rmsnorm_fp4quant.
# Run from repo root: bash benchmarks/run_nsys_add_rmsnorm_fp4quant_both.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_PY="$REPO_ROOT/flashinfer/cute_dsl/add_rmsnorm_fp4quant.py"
BENCH_DIR="$REPO_ROOT/benchmarks"

# Toggle PDL in kernel: 1 = comment out (PDL off), 0 = uncomment (PDL on)
toggle_pdl() {
  local off="$1"
  if [[ "$off" == "1" ]]; then
    sed -i 's/^        griddepcontrol_wait()$/        # griddepcontrol_wait()  # PDL off/' "$KERNEL_PY"
    sed -i 's/^        griddepcontrol_launch_dependents()$/        # griddepcontrol_launch_dependents()  # PDL off/' "$KERNEL_PY"
  else
    sed -i 's/^        # griddepcontrol_wait()  # PDL off$/        griddepcontrol_wait()/' "$KERNEL_PY"
    sed -i 's/^        # griddepcontrol_launch_dependents()  # PDL off$/        griddepcontrol_launch_dependents()/' "$KERNEL_PY"
  fi
}

cd "$REPO_ROOT"

echo "=== PDL OFF ==="
toggle_pdl 1
nsys profile -o "$BENCH_DIR/add_rmsnorm_fp4quant_pdl_off" -f true --cuda-trace-all-apis=true python "$BENCH_DIR/profile_add_rmsnorm_fp4quant_nsys.py"

echo "=== PDL ON ==="
toggle_pdl 0
nsys profile -o "$BENCH_DIR/add_rmsnorm_fp4quant_pdl_on" -f true --cuda-trace-all-apis=true python "$BENCH_DIR/profile_add_rmsnorm_fp4quant_nsys.py"

echo "Done. Reports:"
echo "  $BENCH_DIR/add_rmsnorm_fp4quant_pdl_off.nsys-rep"
echo "  $BENCH_DIR/add_rmsnorm_fp4quant_pdl_on.nsys-rep"
