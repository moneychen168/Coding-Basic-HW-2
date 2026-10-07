#!/bin/bash
# scripts/run_nano4.sh -- build + test + bench on nano4 (RTX 5090, sm_120) (owner: agent-B).
#
# Usage (from repo root on nano4):
#   scripts/run_nano4.sh            # submits itself through sbatch if SLURM exists, else runs directly
#   MODE=direct scripts/run_nano4.sh
#
# ---------------------------------------------------------------- TODO: fill in for nano4
NANO4_ACCOUNT="${NANO4_ACCOUNT:-TODO_ACCOUNT}"       # TODO: nano4 account / project code
NANO4_PARTITION="${NANO4_PARTITION:-TODO_PARTITION}" # TODO: partition that has the RTX 5090
NANO4_GPUS="${NANO4_GPUS:-1}"
NANO4_CPUS="${NANO4_CPUS:-12}"                      # TODO: check per-GPU CPU limit on nano4
NANO4_TIME="${NANO4_TIME:-04:00:00}"
NANO4_CUDA_HOME="${NANO4_CUDA_HOME:-/usr/local/cuda-13.0}"   # TODO: CUDA >= 12.8 needed for sm_120
NANO4_MODULES="${NANO4_MODULES:-}"                  # TODO: e.g. "cuda/13.0" if nano4 uses modules
# ----------------------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-auto}"

if [[ "$MODE" == auto && -z "${SLURM_JOB_ID:-}" ]] && command -v sbatch >/dev/null 2>&1; then
  if [[ "$NANO4_PARTITION" == TODO* || "$NANO4_ACCOUNT" == TODO* ]]; then
    echo "fill NANO4_ACCOUNT / NANO4_PARTITION first (TODO in $0)" >&2; exit 2
  fi
  mkdir -p "$ROOT/results"
  exec sbatch -J ggemm-nano4 -A "$NANO4_ACCOUNT" -p "$NANO4_PARTITION" --nodes=1 \
       --gpus-per-node="$NANO4_GPUS" --cpus-per-task="$NANO4_CPUS" --time="$NANO4_TIME" \
       -o "$ROOT/results/slurm-ggemm-nano4-%j.out" --chdir="$ROOT" --export=ALL,MODE=direct "$0"
fi

[[ -n "$NANO4_MODULES" ]] && module load $NANO4_MODULES
export CUDA_HOME="$NANO4_CUDA_HOME"
source "$ROOT/scripts/env.sh"
OUT="$ROOT/results/nano4_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
[[ "${SKIP_BUILD:-0}" == 1 ]] || TARGETS="test_grouped bench_grouped" ARCHS="120" BUILD_DIR="$ROOT/build_sm120" "$ROOT/scripts/build.sh" || exit 1
BIN="$ROOT/build_sm120"
"$BIN/test_grouped" --set test --csv "$OUT/tests.csv" 2>&1 | tee "$OUT/tests.log"
"$BIN/bench_grouped" --set "${BENCH_SET:-bench}" --paths all --phis 0.5 --moduli 0 \
     --reps "${REPS:-20}" --warmup "${WARMUP:-5}" --csv "$OUT/bench.csv" 2>&1 | tee "$OUT/bench.log"
"$BIN/bench_grouped" --set quick --paths native_loop,emu_grouped,auto --phis 0,0.5,1,2 \
     --moduli 0,8,10,12,14,16,18,20 --reps 3 --warmup 1 --csv "$OUT/accuracy.csv" 2>&1 | tee "$OUT/accuracy.log"
plot_results "$OUT/bench.csv" "$OUT/plots"
plot_results "$OUT/accuracy.csv" "$OUT/plots_acc"
echo "results: $OUT"
