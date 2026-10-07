#!/bin/bash
#SBATCH -J ggemm-i8cfg2sep
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o logs/i8cfg2sep_%j.log
# scripts/i8_cfg2sep.slurm -- split-K cfg2 as a SEPARATE kernel (tile_cfg 3) vs in-kernel cfg2
# (written by agent-K at the coordinator's request).
# Submit from repo root:  sbatch scripts/i8_cfg2sep.slurm      (ITERS=50 PASSES=2)
# One binary holds both paths: cfg2 = main kernel (round-1 SASS), cfg3 = split-K 2nd kernel,
# auto-sk = auto with 32x32 choices routed to cfg3 (2 launches on the "mix" workload).
# build/abl/test_r1 (round-1 binary) is also timed as a same-node reference for cfg0/auto.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/logs/i8cfg2sep_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
cat "$OUT/provenance.txt"
ITERS="${ITERS:-50}"; PASSES="${PASSES:-2}"
BIN="$ROOT/build/cfg2sep/test_i8_grouped"; mkdir -p "$(dirname "$BIN")"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
  -I"$ROOT/src" -Xptxas -v "$ROOT/tests/test_i8_grouped.cu" "$ROOT/src/int8_grouped_gemm.cu" -lcublasLt \
  -o "$BIN" > "$OUT/build.txt" 2>&1 || { cat "$OUT/build.txt"; exit 1; }
grep -E "Compiling entry|registers|spill" "$OUT/build.txt" | sed -E 's/_ZN[^ ]*(persistent|splitk)_kernel[^ ]*/\1_kernel/'
R1="$ROOT/build/abl/test_r1"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv

echo; echo "== correctness (--quick): cfg0..cfg3, auto, auto+splitK, mixed cfg (2 launches), split path relaunch, -128 edge"
"$BIN" --quick > "$OUT/quick.txt" 2>&1; rc=$?
cat "$OUT/quick.txt"
fail=$(( rc != 0 ))

for ((p = 1; p <= PASSES; ++p)); do
  if (( p % 2 == 1 )); then seq=(new r1); else seq=(r1 new); fi
  for k in "${seq[@]}"; do
    b=$BIN; [[ $k == r1 ]] && b=$R1
    [[ -x "$b" ]] || { echo "skip $k (missing $b)"; continue; }
    echo; echo "== perf pass $p binary $k  ($(date +%T), $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader))"
    "$b" --perf --iters "$ITERS" 2>&1 | tee "$OUT/perf_${k}_p$p.txt" | grep -E "^(set|random64|ozaki|small256|mix|single|  \(launches|host scratch)"
  done
done

echo; echo "== summary: TOPS (small256: TOPS(ms)); new binary unless noted"
printf "%-24s %-5s %10s %10s %10s %10s %10s %12s\n" workload pass cfg2 cfg3_sk auto auto_sk cfg0 r1_cfg0
for w in "random64" "ozaki" "small256" "mix" "single 8192" "single 4096"; do
  for ((p = 1; p <= PASSES; ++p)); do
    val() { [[ -f "$1" ]] || return; awk -v w="$w" -v v="$2" 'index($0,w)==1 && $(NF-3)==v { if (w=="small256") printf "%s(%s)", $(NF-1), $(NF-2); else print $(NF-1) }' "$1" | head -1; }
    f="$OUT/perf_new_p$p.txt"; r="$OUT/perf_r1_p$p.txt"
    printf "%-24s %-5s %10s %10s %10s %10s %10s %12s\n" "$w" "$p" "$(val "$f" cfg2)" "$(val "$f" cfg3)" \
      "$(val "$f" auto)" "$(val "$f" auto-sk)" "$(val "$f" cfg0)" "$(val "$r" cfg0)"
  done
done
grep -h "launches auto-sk" "$OUT"/perf_new_p1.txt | head -8
grep -l "FAIL" "$OUT"/perf_new_*.txt && { echo "perf-set check FAILED"; fail=1; }
echo "results: $OUT  overall: $([[ $fail == 0 ]] && echo PASS || echo FAIL)"
exit $fail
