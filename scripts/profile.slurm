#!/bin/bash
#SBATCH -J ggemm-prof
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/profile.slurm -- nsys timelines for every path + ncu on the grouped pipeline (owner: agent-B).
# Submit from repo root:  sbatch scripts/profile.slurm
# Env knobs: WORKLOADS="uni_G1024_s32 moe_E64_prefill_z1.2_up het_G256 lora_G64_rmix_shrink"
#            NCU_PATHS="emu_grouped auto"  NCU_KERNELS='regex:^(?!k_fill)'  NCU_SET=full
#            NCU_COUNT=12  SKIP_NCU=1  SKIP_BUILD=1
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"
OUT="$ROOT/results/prof_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
[[ "${SKIP_BUILD:-0}" == 1 ]] || TARGETS="test_grouped bench_grouped" BUILD_DIR="$ROOT/build" "$ROOT/scripts/build.sh" || exit 1
B="$ROOT/build/bench_grouped"
NSYS="${NSYS:-$CUDA_HOME/bin/nsys}"; NCU="${NCU:-$CUDA_HOME/bin/ncu}"
WORKLOADS="${WORKLOADS:-uni_G1024_s32 moe_E64_prefill_z1.2_up het_G256 lora_G64_rmix_shrink}"

for w in $WORKLOADS; do
  # nsys: all paths in one process; NVTX-free, filter by time range in the GUI / stats
  "$NSYS" profile -t cuda,cublas,osrt --cuda-graph-trace=node -f true -o "$OUT/nsys_$w" \
      "$B" --set quick --only "$w" --paths all --phis 0.5 --reps 5 --warmup 2 --no-check \
      --csv "$OUT/nsys_$w.csv" > "$OUT/nsys_$w.log" 2>&1
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f csv -o "$OUT/nsys_$w" \
      "$OUT/nsys_$w.nsys-rep" >> "$OUT/nsys_$w.log" 2>&1
done

if [[ "${SKIP_NCU:-0}" != 1 ]]; then
  for w in $WORKLOADS; do
    for p in ${NCU_PATHS:-emu_grouped auto}; do
      # one warm-up + one measured call -> ncu sees the library kernels of 2 calls
      "$NCU" --set "${NCU_SET:-full}" -k "${NCU_KERNELS:-regex:^(?!k_fill)}" -c "${NCU_COUNT:-12}" \
          --import-source yes -f -o "$OUT/ncu_${w}_${p}" \
          "$B" --set quick --only "$w" --paths "$p" --phis 0.5 --reps 1 --warmup 1 --no-check \
          --csv "$OUT/ncu_${w}_${p}.csv" > "$OUT/ncu_${w}_${p}.log" 2>&1
      "$NCU" --import "$OUT/ncu_${w}_${p}.ncu-rep" --page details --csv > "$OUT/ncu_${w}_${p}_details.csv" 2>/dev/null
    done
  done
fi
echo "results: $OUT"
