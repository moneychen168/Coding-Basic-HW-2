#!/bin/bash
#SBATCH -J ggemm-test
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/run_tests.slurm -- build + correctness tests (owner: agent-B).
# Submit from repo root:  sbatch scripts/run_tests.slurm
# Env knobs: TEST_SET=test|quick  PATHS=all  PHIS=0,0.5,1,2  SKIP_BUILD=1
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/results/test_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
[[ "${SKIP_BUILD:-0}" == 1 ]] || TARGETS="test_grouped bench_grouped" BUILD_DIR="$ROOT/build" "$ROOT/scripts/build.sh" || exit 1
"$ROOT/build/test_grouped" --set "${TEST_SET:-test}" --paths "${PATHS:-all}" --phis "${PHIS:-0,0.5,1,2}" \
    --csv "$OUT/tests.csv" 2>&1 | tee "$OUT/tests.log"
rc=${PIPESTATUS[0]}
# compute-sanitizer pass on a small subset (out-of-bounds / race in the library kernels)
if [[ "${SANITIZE:-1}" == 1 ]]; then
  "$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 9 \
      "$ROOT/build/test_grouped" --set test --paths emu_grouped,auto --phis 1 --no-edges \
      > "$OUT/memcheck.log" 2>&1
  echo "memcheck exit: $? (log: $OUT/memcheck.log)"
fi
echo "test_grouped exit: $rc  (results: $OUT)"
exit $rc
