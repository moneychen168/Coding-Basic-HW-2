#!/bin/bash
#SBATCH -J ggemm-i8cfg2
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o logs/i8cfg2_%j.log
# scripts/i8_cfg2.slurm -- cfg2 split-K (-DI8_CFG2_SPLITK) vs r1 cfg2, same node, one job
# (written by agent-K at the coordinator's request).
# Submit from repo root:  sbatch scripts/i8_cfg2.slurm
#   ITERS=50  PASSES=2 (pass 2 runs the binaries in reverse order)
# Reports: bit-exact --quick for both builds, then TOPS of cfg2 / auto / cfg0 per workload
# (cfg0 is included because the flag perturbs whole-kernel codegen, see the agent-K report).
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/logs/i8cfg2_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
cat "$OUT/provenance.txt"
ITERS="${ITERS:-50}"; PASSES="${PASSES:-2}"
B="$ROOT/build/cfg2"; mkdir -p "$B"
NV="$CUDA_HOME/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -I$ROOT/src -Xptxas -v"
declare -A FLAGS=([off]="" [splitk]="-DI8_CFG2_SPLITK")
ORDER=(off splitk)
for k in "${ORDER[@]}"; do
  $NV ${FLAGS[$k]} "$ROOT/tests/test_i8_grouped.cu" "$ROOT/src/int8_grouped_gemm.cu" -lcublasLt \
      -o "$B/test_$k" > "$OUT/build_$k.txt" 2>&1 &
done
wait
for k in "${ORDER[@]}"; do
  [[ -x "$B/test_$k" ]] || { echo "build failed: $k"; cat "$OUT/build_$k.txt"; exit 1; }
  echo "ptxas $k: $(grep -E 'registers|spill' "$OUT/build_$k.txt" | tr -s ' ' | paste -sd' ')"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv

echo; echo "== correctness (--quick: 19 hetero problems x {cfg0,cfg1,cfg2,auto} full CPU bit-exact, split path, -128 edge, vs cuBLASLt)"
fail=0
for k in "${ORDER[@]}"; do
  "$B/test_$k" --quick > "$OUT/quick_$k.txt" 2>&1; rc=$?
  [[ $rc == 0 ]] || fail=1
  echo "$k: exit=$rc $(tail -1 "$OUT/quick_$k.txt")"
  grep -E "FAIL|MISMATCH" "$OUT/quick_$k.txt" | head -5
done

for ((p = 1; p <= PASSES; ++p)); do
  if (( p % 2 == 1 )); then seq=(off splitk); else seq=(splitk off); fi
  for k in "${seq[@]}"; do
    echo; echo "== perf pass $p variant $k  ($(date +%T), $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader))"
    "$B/test_$k" --perf --iters "$ITERS" 2>&1 | tee "$OUT/perf_${k}_p$p.txt" | grep -E "^(set|random64|ozaki|small256|single)"
  done
done

echo; echo "== summary: TOPS (ms in parentheses for small256); check column must be ok"
printf "%-32s %-7s %-5s %10s %10s %10s\n" workload build pass cfg2 auto cfg0
for w in "random64" "ozaki" "small256" "single 8192" "single 4096"; do
  for k in "${ORDER[@]}"; do for ((p = 1; p <= PASSES; ++p)); do
    f="$OUT/perf_${k}_p$p.txt"
    val() { awk -v w="$w" -v v="$1" 'index($0,w)==1 && $(NF-3)==v { if (w=="small256") printf "%s(%s)", $(NF-1), $(NF-2); else print $(NF-1) }' "$f" | head -1; }
    printf "%-32s %-7s %-5s %10s %10s %10s\n" "$w" "$k" "$p" "$(val cfg2)" "$(val auto)" "$(val cfg0)"
  done; done
done
grep -l "FAIL" "$OUT"/perf_*.txt && { echo "perf-set check FAILED (see files above)"; fail=1; }
echo "results: $OUT  overall: $([[ $fail == 0 ]] && echo PASS || echo FAIL)"
exit $fail
