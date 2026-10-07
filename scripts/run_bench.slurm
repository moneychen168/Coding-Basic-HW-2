#!/bin/bash
#SBATCH -J ggemm-bench
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/run_bench.slurm -- performance sweep + accuracy sweep + plots (owner: agent-B).
# Submit from repo root:  sbatch scripts/run_bench.slurm
# Env knobs: BENCH_SET=bench|quick  PATHS=all  REPS=20  WARMUP=5  ONLY=<substr>
#            ACC=1 (also run the phi x num_moduli accuracy sweep)  SKIP_BUILD=1
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"; cuda_preflight
OUT="$ROOT/results/bench_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
[[ "${SKIP_BUILD:-0}" == 1 ]] || TARGETS="test_grouped bench_grouped" BUILD_DIR="$ROOT/build" "$ROOT/scripts/build.sh" || exit 1
B="$ROOT/build/bench_grouped"

# 1) performance: every path, same workloads/data, phi=0.5, adaptive moduli
"$B" --set "${BENCH_SET:-bench}" --paths "${PATHS:-all}" --phis 0.5 --moduli 0 \
     --reps "${REPS:-20}" --warmup "${WARMUP:-5}" ${ONLY:+--only "$ONLY"} --csv "$OUT/bench.csv" \
     --dump-shapes "$OUT/shapes.csv" \
     2>&1 | tee "$OUT/bench.log"

# 2) accuracy vs phi and fixed moduli count (timing is a by-product; fewer reps)
if [[ "${ACC:-1}" == 1 ]]; then
  "$B" --set quick --paths native_loop,emu_grouped,auto --phis 0,0.5,1,2 --moduli 0,8,10,12,14,16,18,20 \
       --reps 3 --warmup 1 --csv "$OUT/accuracy.csv" 2>&1 | tee "$OUT/accuracy.log"
fi
plot_results "$OUT/bench.csv" "$OUT/plots"
[[ -f "$OUT/accuracy.csv" ]] && plot_results "$OUT/accuracy.csv" "$OUT/plots_acc"
echo "results: $OUT"
