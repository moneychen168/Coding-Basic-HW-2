#!/bin/bash
#SBATCH -J ggemm-repair
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/repair_sweep.slurm -- cost and frequency of the FP64 certificate/repair (owner: agent-B).
# Submit from repo root:  mkdir -p results && sbatch scripts/repair_sweep.slurm
#
# Part A (EMU_GROUPED): phi x N x repair_mode in {count,on} on the "repair" workload set
#   -> time (COUNT vs ON = cost of repair), uncertified / repaired fraction, err/gamma.
# Part B (AUTO): rho_budget in {4,16,64,256} (env GGEMM_RHO_BUDGET, read once per process,
#   so one bench process per value) -> time, planner N distribution (plan CSV), repaired fraction.
# Counters always come from a separate UNTIMED call on a handle created with
# GGEMM_STATS_REPAIR=1 (--count-call); the timed calls run with counters off.
# Env knobs: PHIS=0,0.5,1,2  MODULI=13,14,15,16,18,20  RHOS="4 16 64 256"  REPS=10  WARMUP=3  SKIP_BUILD=1
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"; cuda_preflight
OUT="$ROOT/results/repair_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
unset GGEMM_STATS_REPAIR GGEMM_RHO_BUDGET
job_build "bench_grouped" || exit 1
B="$BDIR/bench_grouped"
PHIS="${PHIS:-0,0.5,1,2}"; REPS="${REPS:-10}"; WARMUP="${WARMUP:-3}"
"$B" --set repair --dump-shapes "$OUT/shapes.csv" --dump-only

# ---- A: EMU_GROUPED, fixed N, COUNT vs ON
"$B" --set repair --paths emu_grouped --phis "$PHIS" --moduli "${MODULI:-13,14,15,16,18,20}" \
     --repair-modes count,on --count-call --reps "$REPS" --warmup "$WARMUP" \
     --csv "$OUT/emu_repair.csv" 2>&1 | tee "$OUT/emu_repair.log"
rcA=${PIPESTATUS[0]}

# ---- B: AUTO, adaptive N, rho_budget sweep (repair ON = AUTO default; native_loop for reference)
rcB=0
for rho in ${RHOS:-4 16 64 256}; do
  GGEMM_RHO_BUDGET="$rho" "$B" --set repair --paths native_loop,auto --phis "$PHIS" --moduli 0 \
     --repair-modes on --count-call --reps "$REPS" --warmup "$WARMUP" \
     --plan-csv "$OUT/auto_plan_rho${rho}.csv" --csv "$OUT/auto_rho${rho}.csv" 2>&1 | tee "$OUT/auto_rho${rho}.log"
  (( PIPESTATUS[0] != 0 )) && rcB=1
done
# merge (header once)
awk 'FNR==1 && NR!=1 {next} {print}' "$OUT"/auto_rho*.csv > "$OUT/auto_repair.csv"
awk 'FNR==1 && NR!=1 {next} {print}' "$OUT"/auto_plan_rho*.csv > "$OUT/auto_plan.csv"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-$USER}"
uv run --quiet --no-project --with matplotlib --with pandas python "$ROOT/scripts/plot_repair.py" \
   --emu "$OUT/emu_repair.csv" --auto "$OUT/auto_repair.csv" --plan "$OUT/auto_plan.csv" -o "$OUT/plots" \
   || echo "[plot] failed (non-fatal); rerun scripts/plot_repair.py later"
echo "results: $OUT"
exit $(( rcA || rcB ))
