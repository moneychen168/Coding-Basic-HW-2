#!/bin/bash
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH -J render_video
#SBATCH -o /home/money168/hw2_grouped_gemm/docs/video/render_%j.out
cd /home/money168/hw2_grouped_gemm/docs/video
.venv/bin/python make_video.py
