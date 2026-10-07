#!/bin/bash
#SBATCH -J ggemm-app
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o results/slurm-%x-%j.out
# bench/app/run_app.slurm -- FP64 application-level benchmarks (owner: agent-E).
# Submit from repo root:  mkdir -p results && sbatch bench/app/run_app.slurm
# Env knobs: PATHS=all  REPS_MOE=10  REPS_BSR=5  SKIP_BUILD=1  QUICK=1 (one small config each)
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_MODULE_LOADING=EAGER
OUT="$ROOT/results/app_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
{ date -Is; hostname; git -C "$ROOT" rev-parse HEAD 2>/dev/null; nvcc --version | tail -1
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; } > "$OUT/provenance.txt" 2>&1
source "$ROOT/scripts/env.sh"; cuda_preflight
select_calib "$OUT" || true

# ---- build (library via scripts/build.sh conventions; app binaries compiled here)
BUILD="$ROOT/build"; ARCHS="${ARCHS:-90 120}"
GENCODE=""; for a in $ARCHS; do GENCODE+=" -gencode arch=compute_${a},code=sm_${a}"; done
NVFLAGS="-std=c++17 -O3 -lineinfo $GENCODE -I$ROOT/include -I$ROOT/src -I$ROOT/bench -Xcompiler -Wall"
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  TARGETS="none" BUILD_DIR="$BUILD" ARCHS="$ARCHS" "$ROOT/scripts/build.sh" || { echo "library build failed"; exit 1; }
  for app in moe_ffn_fp64 block_sparse_fp64; do
    nvcc $NVFLAGS "$ROOT/bench/app/$app.cu" "$BUILD/obj/sup_workloads.cu.o" "$BUILD/libggemm.a" \
         -lcublas -lcublasLt -o "$BUILD/$app" || { echo "build $app failed"; exit 1; }
  done
fi
P="${PATHS:-all}"
# run LOG CMD... : tee output to LOG but keep CMD's exit code (PIPESTATUS); any failure fails the job.
FAILED=0
run() {
  local log="$1"; shift
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then echo "[run_app] FAILED rc=$rc: $*" | tee -a "$log"; FAILED=1; fi
}

# ---- 1) MoE FFN (FP64). Base config + skew sweep + top-1 "per-element network" config.
M="$BUILD/moe_ffn_fp64"; MC="$OUT/moe_ffn_fp64.csv"; R="${REPS_MOE:-10}"
if [[ "${QUICK:-0}" == 1 ]]; then
  run "$OUT/moe.log" "$M" --tokens 1024 --experts 8 --d-model 256 --d-ff 512 --paths "$P" --reps 3 --warmup 1 --csv "$MC"
else
  for z in 0.0 1.0 1.5; do
    run "$OUT/moe.log" "$M" --tokens 4096 --experts 16 --topk 2 --d-model 1024 --d-ff 2048 --zipf $z --paths "$P" \
         --reps "$R" --warmup 3 --csv "$MC" --append
  done
  run "$OUT/moe.log" "$M" --tokens 16384 --experts 8 --topk 1 --d-model 256 --d-ff 256 --zipf 1.2 --paths "$P" \
       --reps "$R" --warmup 3 --csv "$MC" --append   # per-element MLIP-like
fi

# ---- 2) block-sparse BSR x dense (FP64), both formulations
B="$BUILD/block_sparse_fp64"; BC="$OUT/block_sparse_fp64.csv"; RB="${REPS_BSR:-5}"
if [[ "${QUICK:-0}" == 1 ]]; then
  run "$OUT/bsr.log" "$B" --atoms 128 --cutoff 5 --basis dz --nrhs 32 --paths "$P" --reps 3 --warmup 1 --csv "$BC"
else
  run "$OUT/bsr.log" "$B" --atoms 1024 --cutoff 6 --basis dz --nrhs 128 --mode both --paths "$P" --reps "$RB" --warmup 2 \
       --csv "$BC" --append
  run "$OUT/bsr.log" "$B" --atoms 512 --cutoff 6 --basis tz --nrhs 256 --mode both --paths "$P" --reps "$RB" --warmup 2 \
       --csv "$BC" --append
  run "$OUT/bsr.log" "$B" --atoms 256 --cutoff 6 --basis qz --nrhs 256 --mode both --paths "$P" --reps "$RB" --warmup 2 \
       --csv "$BC" --append
fi
echo "results: $OUT"
if (( FAILED )); then echo "[run_app] at least one app run failed"; exit 1; fi
