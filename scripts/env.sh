# scripts/env.sh -- shared environment for job scripts (sourced, owner: agent-B).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=EAGER     # load all kernels up front -> no lazy-load cost in warm-up/timing
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT
# record provenance next to the results (reproducibility)
write_provenance() {   # $1 = output dir
  {
    echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "job: ${SLURM_JOB_ID:-none}"
    echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none) dirty=$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l)"
    "$CUDA_HOME/bin/nvcc" --version | tail -1
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv,noheader
  } > "$1/provenance.txt" 2>&1
}
plot_results() {      # $1 = bench csv, $2 = out dir  (matplotlib via uv, cache in /tmp)
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-$USER}"
  if command -v uv >/dev/null 2>&1; then
    uv run --quiet --no-project --with matplotlib --with pandas python "$ROOT/scripts/plot.py" "$1" -o "$2" \
      || echo "[plot] failed (non-fatal)"
  elif python3 -c "import matplotlib, pandas" 2>/dev/null; then
    python3 "$ROOT/scripts/plot.py" "$1" -o "$2"
  else
    echo "[plot] matplotlib/pandas unavailable; run scripts/plot.py $1 later"
  fi
}
# Planner calibration: honour GGEMM_CALIB_JSON if set, else bench/calib/params_<slug>.json where
# slug follows bench/calib/fit.py: re.sub("[^A-Za-z0-9]+","_",gpu).strip("_").lower().
select_calib() {      # $1 = output dir (logs the choice into $1/calib.txt and stdout)
  local gpu slug f
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
  slug="$(printf '%s' "$gpu" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//' | tr 'A-Z' 'a-z')"
  if [[ -n "${GGEMM_CALIB_JSON:-}" ]]; then
    f="$GGEMM_CALIB_JSON (from env)"
  elif [[ -f "$ROOT/bench/calib/params_${slug}.json" ]]; then
    export GGEMM_CALIB_JSON="$ROOT/bench/calib/params_${slug}.json"; f="$GGEMM_CALIB_JSON (auto, gpu='$gpu')"
  else
    f="none: bench/calib/params_${slug}.json missing (gpu='$gpu'), planner built-in defaults"
  fi
  echo "[calib] GGEMM_CALIB_JSON = $f" | tee "$1/calib.txt"
}
