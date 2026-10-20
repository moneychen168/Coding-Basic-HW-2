#!/bin/bash
#SBATCH -J ggemm-v0final
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH -o results/slurm-%x-%j.out
# scripts/final/starter_vs_final.slurm -- "starter" (our first working version, commit 5d8e2ad = v0)
# vs final (HEAD) under identical conditions: same job, same node, same GPU, same workload set,
# same reps / warmup, binaries built in this job. No official starter code was provided to us, so the
# starter is v0 (see docs/report.md). Output: results/v0final_<jobid>/{v0,final}.csv + compare.md
#
# Submit from the repo root:  sbatch scripts/final/starter_vs_final.slurm
# Env knobs: V0_REV=5d8e2ad REPS=20 WARMUP=5 BENCH_SET=bench WT_BASE=/work/$USER
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$ROOT/scripts/env.sh"
JOB="${SLURM_JOB_ID:-local}"
OUT="$ROOT/results/v0final_${JOB}"; mkdir -p "$OUT"
V0_REV="${V0_REV:-5d8e2ad}"; REPS="${REPS:-20}"; WARMUP="${WARMUP:-5}"; SET="${BENCH_SET:-bench}"
WT="${WT_BASE:-/work/$USER}/hw2_v0_${JOB}"
write_provenance "$OUT"
select_calib "$OUT"
echo "v0: $(git -C "$ROOT" rev-parse "$V0_REV")" >> "$OUT/provenance.txt"
cleanup() { git -C "$ROOT" worktree remove --force "$WT" 2>/dev/null; rm -rf "$WT" "$ROOT/build_v0final_$JOB"; }
trap cleanup EXIT

git -C "$ROOT" worktree add --detach "$WT" "$V0_REV" > "$OUT/worktree.log" 2>&1 || { echo "worktree failed"; exit 1; }
TARGETS="bench_grouped" BUILD_DIR="$WT/build" "$WT/scripts/build.sh" > "$OUT/build_v0.log" 2>&1 \
  || { echo "v0 build failed (see build_v0.log)"; exit 1; }
TARGETS="bench_grouped" BUILD_DIR="$ROOT/build_v0final_$JOB" "$ROOT/scripts/build.sh" > "$OUT/build_final.log" 2>&1 \
  || { echo "final build failed (see build_final.log)"; exit 1; }

# Alternate the two binaries twice (v0, final, v0, final) so clock / thermal drift hits both equally;
# the reported time per row is the median over the two runs of each binary.
for r in 1 2; do
  "$WT/build/bench_grouped" --set "$SET" --reps "$REPS" --warmup "$WARMUP" --csv "$OUT/v0_r$r.csv" \
      > "$OUT/v0_r$r.log" 2>&1; echo "v0 run $r rc=$?" | tee -a "$OUT/runs.txt"
  "$ROOT/build_v0final_$JOB/bench_grouped" --set "$SET" --reps "$REPS" --warmup "$WARMUP" --csv "$OUT/final_r$r.csv" \
      > "$OUT/final_r$r.log" 2>&1; echo "final run $r rc=$?" | tee -a "$OUT/runs.txt"
done

python3 -I "$ROOT/scripts/final/compare_v0_final.py" "$OUT" > "$OUT/compare.md" 2> "$OUT/compare.err"
echo "compare rc=$?"; cat "$OUT/compare.md" | head -40
echo "results: $OUT"
