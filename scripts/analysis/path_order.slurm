#!/bin/bash
#SBATCH -A GOV114009
#SBATCH -J ggemm-order
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH -o results/slurm-%x-%j.out
# scripts/analysis/path_order.slurm -- does the bench path ORDER change native_grouped time on MoE decode?
# Runs bench_grouped --only moe_E64_decode with 6 path orders x 2; result: results/order_<jobid>/summary.txt (job 590956).
set -uo pipefail
ROOT="$SLURM_SUBMIT_DIR"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/results/order_$SLURM_JOB_ID"; mkdir -p "$OUT"; write_provenance "$OUT"; select_calib "$OUT"
B="$ROOT/build_order_$SLURM_JOB_ID"; trap 'rm -rf "$B"' EXIT
TARGETS=bench_grouped BUILD_DIR="$B" "$ROOT/scripts/build.sh" > "$OUT/build.log" 2>&1 || exit 1
i=0
for order in native_loop,native_grouped,auto native_grouped,native_loop,auto auto,native_grouped native_grouped native_loop,auto,native_grouped native_grouped,native_grouped; do
  for r in 1 2; do i=$((i+1))
    "$B/bench_grouped" --set bench --only moe_E64_decode --paths "$order" --reps 20 --warmup 5 --csv "$OUT/o${i}.csv" > "$OUT/o${i}.log" 2>&1
    python3 -I -c "
import csv,sys
rs=list(csv.DictReader(open('$OUT/o${i}.csv')))
print('order=$order run=$r ::', '  '.join(f\"{x['workload'][15:]}:{x['path']}={float(x['time_ms']):.3f}\" for x in rs))" | tee -a "$OUT/summary.txt"
  done
done
echo "results: $OUT"
