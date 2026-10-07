#!/bin/bash
#SBATCH -J ggemm_calib
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH -o bench/calib/calib_%j.log
# Submit from repo root: sbatch bench/calib/run_calib.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh; cuda_preflight
D=bench/calib
NVCC=/usr/local/cuda-13.0/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
$NVCC -O3 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
  $D/calib.cu -lcublas -lcublasLt -o $D/calib
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.limit --format=csv
CSV=$D/calib_${SLURM_JOB_ID:-local}.csv
$D/calib 4096 > "$CSV"
python3 $D/fit.py "$CSV" -o $D
