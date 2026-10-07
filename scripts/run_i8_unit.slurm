#!/bin/bash
#SBATCH -J ggemm-i8unit
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/run_i8_unit.slurm -- build + run tests/test_i8_grouped (agent-K's unit test) (owner: agent-B).
# Submit from repo root:  sbatch scripts/run_i8_unit.slurm      (extra args: I8_ARGS="...")
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/results/i8unit_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
[[ "${SKIP_BUILD:-0}" == 1 ]] || TARGETS="test_i8_grouped" BUILD_DIR="$ROOT/build" "$ROOT/scripts/build.sh" || exit 1
"$ROOT/build/test_i8_grouped" ${I8_ARGS:-} 2>&1 | tee "$OUT/test_i8_grouped.log"
rc=${PIPESTATUS[0]}
echo "test_i8_grouped exit: $rc  (results: $OUT)"
exit $rc
