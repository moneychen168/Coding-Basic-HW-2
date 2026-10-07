#!/bin/bash
#SBATCH -J ggemm-i8abl
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:30:00
#SBATCH -o logs/i8abl_%j.log
# scripts/i8_ablation.slurm -- round-2 cfg0 regression ablation for the grouped INT8 GEMM
# (written by agent-K at the coordinator's request).
# Submit from repo root:  sbatch scripts/i8_ablation.slurm
#   SKIP_BUILD=1  reuse build/abl/test_* (prebuilt on the login node)
#   ITERS=50      timed iterations per measurement
#   PASSES=2      perf passes; pass 2 runs the binaries in reverse order (drift check)
#
# Variants (all in ONE job, same node):
#   r1          round-1 source (build/abl/round1_int8_grouped_gemm.cu)
#   r2          round-2 default
#   no_po       -DI8_NO_PROLOGUE_OVERLAP
#   zfill       -DI8_ZFILL
#   dec_end     -DI8_DECODE_AT_END
#   dec_every4  -DI8_DECODE_EVERY=4
#   no_splitk   -DI8_NO_SPLITK
#   minb1       -DI8_MINBLOCKS=1 (no 128-reg cap; 1 CTA/SM)
#   revert_all  no_po + zfill + dec_end + no_splitk
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"; cuda_preflight
OUT="$ROOT/logs/i8abl_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
cat "$OUT/provenance.txt"
ITERS="${ITERS:-50}"; PASSES="${PASSES:-2}"
B="$ROOT/build/abl"; mkdir -p "$B"
NV="$CUDA_HOME/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -I$ROOT/src"
declare -A FLAGS=(
  [r2]="" [no_po]="-DI8_NO_PROLOGUE_OVERLAP" [zfill]="-DI8_ZFILL" [dec_end]="-DI8_DECODE_AT_END"
  [dec_every4]="-DI8_DECODE_EVERY=4" [no_splitk]="-DI8_NO_SPLITK" [minb1]="-DI8_MINBLOCKS=1"
  [revert_all]="-DI8_NO_PROLOGUE_OVERLAP -DI8_ZFILL -DI8_DECODE_AT_END -DI8_NO_SPLITK" )
ORDER=(r1 r2 no_po zfill dec_end dec_every4 no_splitk minb1 revert_all)

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "[build] variants -> $B"
  for k in "${!FLAGS[@]}"; do
    $NV ${FLAGS[$k]} -Xptxas -v "$ROOT/tests/test_i8_grouped.cu" "$ROOT/src/int8_grouped_gemm.cu" \
        -lcublasLt -o "$B/test_$k" > "$OUT/build_$k.txt" 2>&1 &
  done
  $NV -Xptxas -v "$ROOT/tests/test_i8_grouped.cu" "$B/round1_int8_grouped_gemm.cu" -lcublasLt \
      -o "$B/test_r1" > "$OUT/build_r1.txt" 2>&1 &
  wait
fi
echo "== ptxas (registers / spills; first line sm_90, second sm_120)"
for k in "${ORDER[@]}"; do
  [[ -x "$B/test_$k" ]] || { echo "missing $B/test_$k"; exit 1; }
  [[ -f "$OUT/build_$k.txt" ]] && echo "$k: $(grep -E 'registers|spill' "$OUT/build_$k.txt" | tr -s ' ' | paste -sd' ')"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv

# 1) correctness for every round-2 variant (r1 is excluded: it rejects the kp=0 case by design)
echo; echo "== correctness (--quick)"
for k in "${ORDER[@]}"; do
  [[ $k == r1 ]] && continue
  "$B/test_$k" --quick > "$OUT/quick_$k.txt" 2>&1; rc=$?
  echo "$k: exit=$rc $(tail -1 "$OUT/quick_$k.txt")"
done

# 2) performance, PASSES passes, alternating order
for ((p = 1; p <= PASSES; ++p)); do
  if (( p % 2 == 1 )); then seq=("${ORDER[@]}"); else seq=(); for ((i=${#ORDER[@]}-1; i>=0; --i)); do seq+=("${ORDER[$i]}"); done; fi
  for k in "${seq[@]}"; do
    echo; echo "== perf pass $p variant $k  ($(date +%T), $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader))"
    "$B/test_$k" --perf --iters "$ITERS" 2>&1 | tee "$OUT/perf_${k}_p$p.txt" | grep -E "^(set|random64|ozaki|small256|single)"
  done
done

# 3) summary: cfg0 / auto TOPS (and small256 ms) per variant per pass
echo; echo "== summary (TOPS; small256 in ms)"
printf "%-11s %-5s %9s %9s %9s %9s %9s %9s %9s\n" variant pass rnd_cfg0 rnd_auto oz_cfg0 sq8k_cfg0 k16k_cfg0 sm256_auto sm256_cfg2
for k in "${ORDER[@]}"; do for ((p = 1; p <= PASSES; ++p)); do
  f="$OUT/perf_${k}_p$p.txt"
  r0=$(awk 'index($0,"random64")==1 && $(NF-3)=="cfg0"{print $(NF-1)}' "$f" | head -1)
  ra=$(awk 'index($0,"random64")==1 && $(NF-3)=="auto"{print $(NF-1)}' "$f" | head -1)
  o0=$(awk 'index($0,"ozaki")==1 && $(NF-3)=="cfg0"{print $(NF-1)}' "$f" | head -1)
  s0=$(awk 'index($0,"single 8192")==1 && $(NF-3)=="cfg0"{print $(NF-1)}' "$f" | head -1)
  k0=$(awk 'index($0,"single 4096")==1 && $(NF-3)=="cfg0"{print $(NF-1)}' "$f" | head -1)
  sa=$(awk 'index($0,"small256")==1 && $(NF-3)=="auto"{print $(NF-2)}' "$f" | head -1)
  s2=$(awk 'index($0,"small256")==1 && $(NF-3)=="cfg2"{print $(NF-2)}' "$f" | head -1)
  printf "%-11s %-5s %9s %9s %9s %9s %9s %9s %9s\n" "$k" "$p" "$r0" "$ra" "$o0" "$s0" "$k0" "$sa" "$s2"
done; done
echo "results: $OUT"
