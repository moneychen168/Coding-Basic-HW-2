#!/bin/bash
#SBATCH -J ggemm-planner-unit
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/run_planner_unit.slurm -- build + run tests/test_planner.cpp (host-only planner unit test).
# Submit from repo root:  sbatch scripts/run_planner_unit.slurm
# Builds from the CURRENT src/planner.cpp + src/ozaki2_numerics.cu (not build/libggemm.a), so it
# always tests the working-tree planner. COMPILE_ONLY=1 builds without running (login-node safe).
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
OUT="$ROOT/results/planner_unit_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
BLD="$OUT/build"; mkdir -p "$BLD"
INC=(-I"$ROOT/include" -I"$ROOT/src" -I"$CUDA_HOME/include")
set -e
"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 --expt-relaxed-constexpr \
    -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
    "${INC[@]}" -c "$ROOT/src/ozaki2_numerics.cu" -o "$BLD/ozaki2_numerics.o"
g++ -std=c++17 -O2 -Wall "${INC[@]}" -DGGEMM_CALIB_DIR="\"$ROOT/bench/calib\"" \
    -c "$ROOT/src/planner.cpp" -o "$BLD/planner.o"
g++ -std=c++17 -O2 -Wall "${INC[@]}" -c "$ROOT/tests/test_planner.cpp" -o "$BLD/test_planner.o"
g++ "$BLD/test_planner.o" "$BLD/planner.o" "$BLD/ozaki2_numerics.o" \
    -L"$CUDA_HOME/lib64" -lcudart -Wl,-rpath,"$CUDA_HOME/lib64" -o "$BLD/test_planner"
set +e
echo "built $BLD/test_planner"
[[ "${COMPILE_ONLY:-0}" == 1 ]] && exit 0
cd "$ROOT"
"$BLD/test_planner" "$ROOT/tests/testdata/planner" 2>&1 | tee "$OUT/test_planner.log"
rc=${PIPESTATUS[0]}
echo "test_planner exit: $rc  (results: $OUT)"
exit $rc
