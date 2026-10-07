#!/bin/bash
#SBATCH -J ggemm-final
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:50:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH -o results/slurm-%x-%j.out
# scripts/final/run_final.slurm -- final experiment matrix for the 10/14 hand-in (owner: agent-A).
# Everything (build, tests, bench, apps, profiles, analysis, tables) runs in ONE job on ONE node from
# ONE source tree, into results/final_<jobid>/. A failing step is recorded and the next step runs;
# summary.md lists PASS / FAIL / SKIP per step, tables.md is produced by scripts/final/aggregate.py.
#
# Submit from the repo root:   mkdir -p results && sbatch scripts/final/run_final.slurm
# Re-make the tables later:    python3 -I scripts/final/aggregate.py results/final_<jobid>
#
# Env knobs: REPS=20 WARMUP=5 PHI=0.5 BENCH_SET=bench REPS_MOE=10 REPS_BSR=5
#            PROF_WORKLOADS="uni_G1024_s32 moe_E64_prefill_z1.2_up het_G256_16-2048"
#            NCU_PATHS="emu_grouped auto"  NCU_SET=full  NCU_COUNT=12  SKIP_NCU=1  SKIP_NSYS=1
#            REQUIRE_CLEAN=1 (abort if the git tree is dirty)
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$ROOT/scripts/env.sh"
export ROOT
JOB="${SLURM_JOB_ID:-local}"
export OUT="$ROOT/results/final_${JOB}"
export BUILD="$OUT/build"            # fresh build dir: no stale binary from another commit can be run
export LOGS="$OUT/logs" PROF="$OUT/prof" ANA="$OUT/analysis"
mkdir -p "$OUT" "$LOGS" "$PROF" "$ANA"
STEPS="$OUT/steps.tsv"
printf 'step\tstatus\trc\tseconds\tlog\n' > "$STEPS"
T_START="$(date -Is)"

# ------------------------------------------------------------------ step bookkeeping
# run_step NAME TIMEOUT_S CMD  -- CMD runs in `bash -o pipefail -c`, stdout+stderr -> logs/NAME.log
run_step() {
  local name="$1" tmo="$2" cmd="$3" log="$LOGS/$1.log" t0=$SECONDS rc st
  echo "[$(date +%T)] >>> $name"
  { echo "# cmd: $cmd"; echo "# start: $(date -Is)"; } > "$log"
  timeout --kill-after=60 "$tmo" bash -o pipefail -c "$cmd" >> "$log" 2>&1; rc=$?
  if   (( rc == 0 )); then st=PASS
  elif (( rc == 124 || rc == 137 )); then st="FAIL(timeout ${tmo}s)"
  else st=FAIL; fi
  printf '%s\t%s\t%d\t%d\t%s\n' "$name" "$st" "$rc" "$((SECONDS - t0))" "logs/$name.log" >> "$STEPS"
  echo "[$(date +%T)] <<< $name: $st (rc=$rc, $((SECONDS - t0)) s)"
  return $rc
}
skip_step() {   # NAME REASON
  printf '%s\tSKIP\t-\t0\t%s\n' "$1" "$2" >> "$STEPS"
  echo "[$(date +%T)] --- $1: SKIP ($2)"
}
write_summary() {
  local s="$OUT/summary.md" np nf ns
  np=$(awk -F'\t' 'NR>1 && $2=="PASS"' "$STEPS" | wc -l)
  nf=$(awk -F'\t' 'NR>1 && $2 ~ /^FAIL/' "$STEPS" | wc -l)
  ns=$(awk -F'\t' 'NR>1 && $2=="SKIP"' "$STEPS" | wc -l)
  {
    echo "# Final experiment matrix: job $JOB"
    echo
    echo "- host: $(hostname)"
    echo "- start: $T_START  end: $(date -Is)"
    echo "- git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none)  dirty files: $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l) (see provenance_git.txt, dirty.patch)"
    echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0 2>/dev/null | head -1)"
    echo "- paths: $(awk -F'\t' '$2=="included"{printf "%s ", $1}' "$OUT/paths.tsv" 2>/dev/null)"
    echo "- steps: $np PASS, $nf FAIL, $ns SKIP"
    echo
    echo "| step | status | rc | seconds | log |"
    echo "|---|---|---:|---:|---|"
    awk -F'\t' 'NR>1 {printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5}' "$STEPS"
    echo
    echo "Tables: tables.md (scripts/final/aggregate.py). Paths: paths.tsv."
  } > "$s"
  echo "summary: $s"
}

# ------------------------------------------------------------------ 0) CUDA preflight
if ! ( cuda_preflight ) > "$LOGS/cuda_preflight.log" 2>&1; then
  printf 'cuda_preflight\tFAIL\t75\t0\tlogs/cuda_preflight.log\n' >> "$STEPS"
  cat "$LOGS/cuda_preflight.log"; write_summary; exit 75
fi
printf 'cuda_preflight\tPASS\t0\t0\tlogs/cuda_preflight.log\n' >> "$STEPS"

# ------------------------------------------------------------------ 1) provenance
prov() {
  write_provenance "$OUT"
  {
    echo "commit: $(git -C "$ROOT" rev-parse HEAD)"
    echo "branch: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
    git -C "$ROOT" log -1 --format='subject: %s%ndate: %cI'
    echo; echo "== git status --porcelain"; git -C "$ROOT" status --porcelain
    echo; echo "== git diff --stat HEAD"; git -C "$ROOT" diff --stat HEAD
  } > "$OUT/provenance_git.txt" 2>&1
  git -C "$ROOT" diff HEAD > "$OUT/dirty.patch" 2>/dev/null
  # untracked files are part of the tree that gets built: list them with their checksums
  git -C "$ROOT" ls-files --others --exclude-standard -z 2>/dev/null \
    | (cd "$ROOT" && xargs -0 -r sha256sum) > "$OUT/untracked.sha256" 2>/dev/null
  { nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; } > "$OUT/nvidia-smi.txt" 2>&1
  "$CUDA_HOME/bin/nvcc" --version > "$OUT/nvcc.txt" 2>&1
  select_calib "$OUT"
  cat "$OUT/provenance.txt" "$OUT/provenance_git.txt"
  if [[ "${REQUIRE_CLEAN:-0}" == 1 && -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    echo "REQUIRE_CLEAN=1 but the tree is dirty"; return 1
  fi
}
t0=$SECONDS
prov > "$LOGS/provenance.log" 2>&1; rc=$?
printf 'provenance\t%s\t%d\t%d\tlogs/provenance.log\n' "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" $rc $((SECONDS - t0)) >> "$STEPS"
if (( rc != 0 )); then write_summary; exit 1; fi
# (select_calib, run inside prov in this shell, exported GGEMM_CALIB_JSON if a calib JSON was found)

# ------------------------------------------------------------------ path list
# A path is run only if bench/workloads.{h,cu} know its name (parse_paths silently drops unknown names).
: > "$OUT/paths.tsv"
PATH_LIST=()
for p in native_loop native_grouped emu_loop emu_loop_batched emu_grouped emu_grouped_reuse auto; do
  ok=1; why=""
  grep -q "\"$p\"" "$ROOT/bench/workloads.h" "$ROOT/bench/workloads.cu" 2>/dev/null || { ok=0; why="name not in bench/workloads.h"; }
  if [[ $p == emu_loop_batched ]]; then
    grep -q 'GGEMM_PATH_EMU_LOOP_BATCHED' "$ROOT/include/ggemm.h" || { ok=0; why="GGEMM_PATH_EMU_LOOP_BATCHED not in ggemm.h"; }
    grep -q 'GGEMM_PATH_EMU_LOOP_BATCHED' "$ROOT/bench/workloads.cu" || { ok=0; why="${why:+$why; }not in parse_paths list (bench/workloads.cu)"; }
  fi
  if (( ok )); then PATH_LIST+=("$p"); printf '%s\tincluded\t\n' "$p" >> "$OUT/paths.tsv"
  else printf '%s\tskipped\t%s\n' "$p" "$why" >> "$OUT/paths.tsv"; skip_step "path_$p" "$why"; fi
done
export PATHS_CSV="$(IFS=,; echo "${PATH_LIST[*]}")"
echo "paths: $PATHS_CSV"

# ------------------------------------------------------------------ 2) build (library + tests + bench + apps)
GENCODE=""; for a in ${ARCHS:-90 120}; do GENCODE+=" -gencode arch=compute_${a},code=sm_${a}"; done
APP_FLAGS="-std=c++17 -O3 -lineinfo $GENCODE -I$ROOT/include -I$ROOT/src -I$ROOT/bench -Xcompiler -Wall"
run_step build 2400 "JOBS=12 BUILD_DIR='$BUILD' '$ROOT/scripts/build.sh'"
for app in moe_ffn_fp64 block_sparse_fp64; do
  run_step "build_$app" 1200 "'$CUDA_HOME/bin/nvcc' $APP_FLAGS '$ROOT/bench/app/$app.cu' '$BUILD/obj/sup_workloads.cu.o' \
      '$BUILD/libggemm.a' -lcublas -lcublasLt -o '$BUILD/$app'"
done
need() { [[ -x "$BUILD/$1" ]]; }   # binary built in THIS job?

# ------------------------------------------------------------------ 3) correctness tests
for t in test_i8_grouped test_numerics_standalone test_grouped test_graph test_reuse_e2e; do
  if [[ ! -f "$ROOT/tests/$t.cu" ]]; then skip_step "$t" "tests/$t.cu not present"; continue; fi
  if ! need "$t"; then
    printf '%s\tFAIL\t-\t0\t%s\n' "$t" "binary not built (see logs/build.log)" >> "$STEPS"; continue
  fi
  case $t in
    test_grouped)  run_step "$t" 3600 "'$BUILD/$t' --set test --paths '$PATHS_CSV' --phis 0,0.5,1,2 --csv '$OUT/tests.csv'" ;;
    test_reuse_e2e) run_step "$t" 1800 "GGEMM_STATS_REPAIR=1 '$BUILD/$t'" ;;   # fills stats.num_repaired
    *)             run_step "$t" 1800 "'$BUILD/$t'" ;;
  esac
done

# ------------------------------------------------------------------ 4) bench_grouped, full set
if need bench_grouped; then
  run_step bench_grouped 5400 "'$BUILD/bench_grouped' --set '${BENCH_SET:-bench}' --paths '$PATHS_CSV' --phis '${PHI:-0.5}' \
      --moduli 0 --reps '${REPS:-20}' --warmup '${WARMUP:-5}' --csv '$OUT/bench.csv' --dump-shapes '$OUT/shapes.csv'"
else
  printf 'bench_grouped\tFAIL\t-\t0\tbinary not built (see logs/build.log)\n' >> "$STEPS"
fi

# ------------------------------------------------------------------ 5) application benchmarks (configs = bench/app/run_app.slurm)
R="${REPS_MOE:-10}"; RB="${REPS_BSR:-5}"
if need moe_ffn_fp64; then
  run_step app_moe_ffn 2400 "rm -f '$OUT/moe_ffn_fp64.csv'
    for z in 0.0 1.0 1.5; do
      '$BUILD/moe_ffn_fp64' --tokens 4096 --experts 16 --topk 2 --d-model 1024 --d-ff 2048 --zipf \$z \
        --paths '$PATHS_CSV' --reps $R --warmup 3 --csv '$OUT/moe_ffn_fp64.csv' --append || exit 1
    done
    '$BUILD/moe_ffn_fp64' --tokens 16384 --experts 8 --topk 1 --d-model 256 --d-ff 256 --zipf 1.2 \
      --paths '$PATHS_CSV' --reps $R --warmup 3 --csv '$OUT/moe_ffn_fp64.csv' --append"
else
  printf 'app_moe_ffn\tFAIL\t-\t0\tbinary not built (see logs/build_moe_ffn_fp64.log)\n' >> "$STEPS"
fi
if need block_sparse_fp64; then
  run_step app_block_sparse 2400 "rm -f '$OUT/block_sparse_fp64.csv'
    for cfg in '1024 dz 128' '512 tz 256' '256 qz 256'; do
      set -- \$cfg
      '$BUILD/block_sparse_fp64' --atoms \$1 --cutoff 6 --basis \$2 --nrhs \$3 --mode both --paths '$PATHS_CSV' \
        --reps $RB --warmup 2 --csv '$OUT/block_sparse_fp64.csv' --append || exit 1
    done"
else
  printf 'app_block_sparse\tFAIL\t-\t0\tbinary not built (see logs/build_block_sparse_fp64.log)\n' >> "$STEPS"
fi

# ------------------------------------------------------------------ 6) profiles (nsys all paths, ncu on the grouped kernels)
NSYS="${NSYS:-$CUDA_HOME/bin/nsys}"; NCU="${NCU:-$CUDA_HOME/bin/ncu}"
PROF_WORKLOADS="${PROF_WORKLOADS:-uni_G1024_s32 moe_E64_prefill_z1.2_up het_G256_16-2048}"
NCU_PATHS_EFF=""
for p in ${NCU_PATHS:-emu_grouped auto}; do [[ ",$PATHS_CSV," == *",$p,"* ]] && NCU_PATHS_EFF+="$p "; done
for w in $PROF_WORKLOADS; do
  if ! need bench_grouped; then printf 'nsys_%s\tFAIL\t-\t0\tbench_grouped not built\n' "$w" >> "$STEPS"; continue; fi
  if [[ "${SKIP_NSYS:-0}" == 1 ]]; then skip_step "nsys_$w" "SKIP_NSYS=1"; continue; fi
  run_step "nsys_$w" 1800 "'$NSYS' profile -t cuda,cublas,osrt,nvtx --cuda-graph-trace=node -f true -o '$PROF/nsys_$w' \
      '$BUILD/bench_grouped' --set '${BENCH_SET:-bench}' --only '$w' --paths '$PATHS_CSV' --phis '${PHI:-0.5}' \
      --reps 5 --warmup 2 --no-check --csv '$PROF/nsys_$w.csv' \
    && '$NSYS' stats -r cuda_gpu_trace,cuda_api_sum,cuda_api_trace,nvtx_pushpop_trace -f csv -o '$PROF/nsys_$w' \
      '$PROF/nsys_$w.nsys-rep'"
done
if [[ "${SKIP_NCU:-0}" != 1 ]] && need bench_grouped; then
  for w in $PROF_WORKLOADS; do
    for p in $NCU_PATHS_EFF; do
      # one warm-up + one measured call -> ncu sees the library kernels of 2 calls
      run_step "ncu_${w}_${p}" 1800 "'$NCU' --set '${NCU_SET:-full}' -k 'regex:^(?!k_fill)' -c '${NCU_COUNT:-12}' \
          --import-source yes -f -o '$PROF/ncu_${w}_${p}' \
          '$BUILD/bench_grouped' --set '${BENCH_SET:-bench}' --only '$w' --paths '$p' --phis '${PHI:-0.5}' --reps 1 \
          --warmup 1 --no-check --csv '$PROF/ncu_${w}_${p}.csv' \
        && '$NCU' --import '$PROF/ncu_${w}_${p}.ncu-rep' --page details --csv > '$PROF/ncu_${w}_${p}_details.csv' \
        && '$NCU' --import '$PROF/ncu_${w}_${p}.ncu-rep' --page raw --csv > '$PROF/ncu_${w}_${p}_raw.csv'"
    done
  done
else
  skip_step ncu "SKIP_NCU=1 or bench_grouped not built"
fi

# ------------------------------------------------------------------ 7) analysis tools (scripts/analysis)
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-$USER}"
PYRUN="python3 -I"   # stdlib-only parsers; figures only if matplotlib is importable
if command -v uv >/dev/null 2>&1 && uv run --quiet --no-project --with matplotlib python -c 'import matplotlib' >/dev/null 2>&1; then
  PYRUN="uv run --quiet --no-project --with matplotlib python -I"
fi
A="$ROOT/scripts/analysis"
if [[ -s "$OUT/bench.csv" && -n "${GGEMM_CALIB_JSON:-}" && -f "${GGEMM_CALIB_JSON:-/nonexistent}" ]]; then
  run_step roofline 900 "$PYRUN '$A/roofline.py' '$OUT/bench.csv' --calib '$GGEMM_CALIB_JSON' --shapes '$OUT/shapes.csv' \
      -o '$ANA' --label final"
else
  printf 'roofline\tFAIL\t-\t0\tbench.csv empty or no calib JSON (see calib.txt)\n' >> "$STEPS"
fi
for w in $PROF_WORKLOADS; do
  if [[ -f "$PROF/nsys_${w}_cuda_gpu_trace.csv" ]]; then
    run_step "nsys_report_$w" 900 "$PYRUN '$A/nsys_report.py' '$PROF/nsys_$w' -o '$ANA' --paths '$PATHS_CSV' --label '$w'"
  else
    skip_step "nsys_report_$w" "no nsys export for $w"
  fi
  for p in $NCU_PATHS_EFF; do
    f="$PROF/ncu_${w}_${p}_details.csv"; r="$PROF/ncu_${w}_${p}_raw.csv"
    if [[ -s "$f" ]]; then
      run_step "ncu_report_${w}_${p}" 900 "$PYRUN '$A/ncu_report.py' '$f' $([[ -s $r ]] && echo "'$r'") -o '$ANA' --label '${w}_${p}'"
    else
      skip_step "ncu_report_${w}_${p}" "no ncu export"
    fi
  done
done
if [[ -s "$OUT/bench.csv" ]]; then
  run_step plots 900 "source '$ROOT/scripts/env.sh'; plot_results '$OUT/bench.csv' '$OUT/plots'"
fi

# ------------------------------------------------------------------ 8) tables + summary
run_step aggregate 600 "python3 -I '$ROOT/scripts/final/aggregate.py' '$OUT' -o '$OUT/tables.md'"
write_summary
cat "$OUT/summary.md"
echo "results: $OUT"
awk -F'\t' 'NR>1 && $2 ~ /^FAIL/ {f=1} END {exit f}' "$STEPS"
