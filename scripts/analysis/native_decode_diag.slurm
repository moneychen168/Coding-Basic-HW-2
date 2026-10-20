#!/bin/bash
#SBATCH -J ggemm-ndiag
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH -o results/slurm-%x-%j.out
# scripts/analysis/native_decode_diag.slurm -- NATIVE_GROUPED vs AUTO on MoE decode, committed revision
# (BASE_REV, merged same-shape groups) vs the working tree (size-1 groups unless all shapes equal).
# Same job / GPU. For each build: bench timing x3 (alternating) and one nsys trace of the decode workloads,
# then the cuBLAS kernel names / counts per NVTX path range. Output: results/ndiag_<jobid>/
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$ROOT/scripts/env.sh"
JOB="${SLURM_JOB_ID:-local}"
OUT="$ROOT/results/ndiag_${JOB}"; mkdir -p "$OUT"
BASE_REV="${BASE_REV:-HEAD}"
WT="${WT_BASE:-/work/$USER}/hw2_ndiag_${JOB}"
NSYS="${NSYS:-$CUDA_HOME/bin/nsys}"
ONLY="${ONLY:-moe_E}"   # substring match (bench --only)
write_provenance "$OUT"; select_calib "$OUT"
echo "base: $(git -C "$ROOT" rev-parse "$BASE_REV")  new: working tree (see dirty.patch)" >> "$OUT/provenance.txt"
git -C "$ROOT" diff > "$OUT/dirty.patch"
cleanup() { git -C "$ROOT" worktree remove --force "$WT" 2>/dev/null; rm -rf "$WT" "$ROOT/build_ndiag_$JOB"; }
trap cleanup EXIT
git -C "$ROOT" worktree add --detach "$WT" "$BASE_REV" > "$OUT/worktree.log" 2>&1 || exit 1
TARGETS="bench_grouped" BUILD_DIR="$WT/build" "$WT/scripts/build.sh" > "$OUT/build_base.log" 2>&1 || { echo "base build failed"; exit 1; }
TARGETS="bench_grouped" BUILD_DIR="$ROOT/build_ndiag_$JOB" "$ROOT/scripts/build.sh" > "$OUT/build_new.log" 2>&1 || { echo "new build failed"; exit 1; }
declare -A BIN=([base]="$WT/build/bench_grouped" [new]="$ROOT/build_ndiag_$JOB/bench_grouped")
for r in 1 2 3; do
  for v in base new; do
    "${BIN[$v]}" --set bench --only "$ONLY" --paths native_loop,native_grouped,auto --reps 20 --warmup 5 \
        --csv "$OUT/${v}_r$r.csv" > "$OUT/${v}_r$r.log" 2>&1; echo "$v r$r rc=$?" >> "$OUT/runs.txt"
  done
done
for v in base new; do
  "$NSYS" profile -t cuda,nvtx -f true -o "$OUT/nsys_$v" "${BIN[$v]}" --set bench --only "$ONLY" \
      --paths native_grouped,auto --reps 5 --warmup 2 --csv "$OUT/nsys_${v}_bench.csv" > "$OUT/nsys_$v.log" 2>&1
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o "$OUT/nsys_$v" "$OUT/nsys_$v.nsys-rep" >> "$OUT/nsys_$v.log" 2>&1
  python3 -I "$ROOT/scripts/analysis/nsys_report.py" "$OUT/nsys_$v.nsys-rep" > "$OUT/nsys_report_$v.txt" 2>&1
done
rm -f "$OUT"/*.sqlite
python3 -I - "$OUT" <<'EOF' > "$OUT/summary.md"
import csv, statistics, sys, os
out = sys.argv[1]; t = {}
for v in ("base", "new"):
    for r in (1, 2, 3):
        p = f"{out}/{v}_r{r}.csv"
        if not os.path.exists(p): continue
        for x in csv.DictReader(open(p)):
            t.setdefault((x["workload"], x["path"], v), []).append((float(x["time_ms"]), x["err_max"]))
print("| workload | path | base runs (ms) | new runs (ms) | base err_max | new err_max |")
print("|---|---|---|---|---|---|")
for w, p in sorted({(k[0], k[1]) for k in t}):
    b, n = t.get((w, p, "base"), []), t.get((w, p, "new"), [])
    print(f"| {w} | {p} | {' '.join(f'{x[0]:.4f}' for x in b)} | {' '.join(f'{x[0]:.4f}' for x in n)} "
          f"| {b[0][1] if b else '-'} | {n[0][1] if n else '-'} |")
EOF
cat "$OUT/summary.md"; for v in base new; do echo "== $v kernels"; head -12 "$OUT/nsys_${v}_cuda_gpu_kern_sum.csv"; done
echo "results: $OUT"
