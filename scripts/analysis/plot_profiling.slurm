#!/bin/bash
#SBATCH -A GOV115094
#SBATCH -J ggemm-plotprof
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH -o results/slurm-%x-%j.out
# scripts/analysis/plot_profiling.slurm -- render docs/figures/fig_*.{png,svg} from results/final_590258/analysis CSVs (no GPU work).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-$USER}"
if command -v uv >/dev/null 2>&1; then
  uv run --quiet --no-project --with matplotlib --with pandas python scripts/analysis/plot_profiling.py results/final_590258 -o docs/figures
else
  python3 scripts/analysis/plot_profiling.py results/final_590258 -o docs/figures
fi
ls -la docs/figures
