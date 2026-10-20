#!/bin/bash
#SBATCH -J ggemm-sanitize
#SBATCH -A GOV114009
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH -o results/slurm-%x-%j.out
# scripts/sanitize.slurm -- compute-sanitizer memcheck/racecheck/initcheck/synccheck over the
# test binaries; one log per (program, tool) + summary table (owner: agent-B).
# Submit from repo root:  mkdir -p results && sbatch scripts/sanitize.slurm
# Env knobs: TOOLS="memcheck racecheck initcheck synccheck"
#            PROGS="test_grouped test_reuse_e2e test_graph test_i8_grouped test_operand_reuse"
#            RUN_TIMEOUT=900 (s, per program x tool; TIMEOUT is reported, not hidden)  SKIP_BUILD=1
#
# Workload choice: sanitizer tools instrument every memory access (memcheck/synccheck ~10x,
# racecheck/initcheck up to ~100x slower), so test_grouped runs the "tiny" set (uniform 8x16^3,
# MoE decode with empty experts, heterogeneous 6 problems), all paths, edge cases with --small;
# racecheck/initcheck additionally use a single phi and skip nothing else.
# test_reuse_e2e / test_graph under racecheck use --small (same paths/edge cases, shrunken shapes,
# no timing loops): the full-size run hit the 900 s limit in job 511062.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; source "$ROOT/scripts/env.sh"; cuda_preflight
OUT="$ROOT/results/sanitize_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
write_provenance "$OUT"
select_calib "$OUT"
PROGS="${PROGS:-test_grouped test_reuse_e2e test_graph test_i8_grouped test_operand_reuse}"
# host-only programs: they never call the CUDA API, so compute-sanitizer cannot attach at all
# ("Target application terminated before first instrumented API call", exit 255) -> every tool
# is n/a; the program is run once without the sanitizer and its exit code is recorded instead.
HOST_ONLY="${HOST_ONLY:-test_operand_reuse}"
TOOLS="${TOOLS:-memcheck racecheck initcheck synccheck}"
job_build "$PROGS" || exit 1
CS="$CUDA_HOME/bin/compute-sanitizer"
"$CS" --version | tail -1 | tee -a "$OUT/provenance.txt"

args_for() {   # $1 = program, $2 = tool
  case "$1" in
    test_grouped)
      if [[ "$2" == racecheck || "$2" == initcheck ]]; then echo "--set tiny --small --phis 0.5"
      else echo "--set tiny --small --phis 1"; fi ;;   # phi=1 also runs alpha=1.5,beta=-0.75
    test_reuse_e2e|test_graph)   # racecheck is ~100x slower: shrunken workloads, no timing
      if [[ "$2" == racecheck ]]; then echo "--reps 1 --small"; else echo "--reps 1"; fi ;;
    test_i8_grouped) echo "--quick" ;;
    *) echo "" ;;
  esac
}
tool_flags() {
  case "$1" in
    memcheck)  echo "--leak-check full" ;;
    racecheck) echo "--racecheck-report all" ;;
    *)         echo "" ;;
  esac
}

echo "program,tool,errors,sanitizer_exit,status,seconds,log" > "$OUT/summary.csv"
for p in $PROGS; do
  bin="$BDIR/$p"
  for t in $TOOLS; do
    if [[ " $HOST_ONLY " == *" $p "* ]]; then
      echo "$p,$t,,,n/a_host_only,0," >> "$OUT/summary.csv"; continue
    fi
    log="$OUT/${p}_${t}.log"
    if [[ ! -x "$bin" ]]; then echo "$p,$t,,,missing_binary,0," >> "$OUT/summary.csv"; continue; fi
    t0=$(date +%s)
    timeout --signal=INT --kill-after=60 "${RUN_TIMEOUT:-900}" \
      "$CS" --tool "$t" $(tool_flags "$t") --print-limit 100 --error-exitcode 99 --target-processes all \
      "$bin" $(args_for "$p" "$t") > "$log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t0 ))
    # "========= ERROR SUMMARY: N errors"; racecheck: "RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)"
    errs=$(grep -Eo 'ERROR SUMMARY: [0-9]+ error' "$log" | tail -1 | grep -Eo '[0-9]+')
    if [[ "$t" == racecheck ]]; then
      rerr=$(grep -Eo 'RACECHECK SUMMARY: .*\(([0-9]+) error' "$log" | tail -1 | grep -Eo '\(([0-9]+)' | tr -d '(')
      # racecheck prints no "ERROR SUMMARY" line (only one RACECHECK SUMMARY per process), so
      # errs is usually empty here: take rerr then; otherwise the larger of the two counts.
      if [[ -n "$rerr" ]] && [[ -z "$errs" ]] || (( ${rerr:-0} > ${errs:-0} )); then errs=$rerr; fi
    fi
    if   (( rc == 124 || rc == 137 )); then st="TIMEOUT"
    elif [[ -z "$errs" ]];             then st="no_summary(rc=$rc)"
    elif (( errs > 0 ));               then st="ERRORS"
    elif (( rc != 0 ));                then st="prog_fail(rc=$rc)"   # sanitizer clean, test itself failed
    else st="clean"; fi
    echo "$p,$t,${errs:-},$rc,$st,$dt,$(basename "$log")" >> "$OUT/summary.csv"
    echo "[sanitize] $p / $t -> errors=${errs:-?} status=$st (${dt}s)"
  done
done

# host-only programs: plain run (no sanitizer), exit code into summary.csv as tool "plain"
for p in $PROGS; do
  [[ " $HOST_ONLY " == *" $p "* ]] || continue
  if [[ -x "$BDIR/$p" ]]; then
    "$BDIR/$p" > "$OUT/${p}_plain.log" 2>&1; rc=$?
    st=$([[ $rc == 0 ]] && echo clean || echo "prog_fail(rc=$rc)")
    echo "$p,plain,0,$rc,$st,0,${p}_plain.log" >> "$OUT/summary.csv"
    echo "[sanitize] $p (host-only, no sanitizer) exit=$rc"
  else echo "$p,plain,,,missing_binary,0," >> "$OUT/summary.csv"; fi
done

# program x tool table of error counts (status appended when not clean)
python3 - "$OUT/summary.csv" "$TOOLS" > "$OUT/summary.md" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tools = sys.argv[2].split() + (["plain"] if any(r["tool"] == "plain" for r in rows) else [])
progs = list(dict.fromkeys(r["program"] for r in rows))
cell = {(r["program"], r["tool"]): (r["errors"] or "?") + ("" if r["status"] == "clean" else f" ({r['status']})") for r in rows}
print("errors per program x compute-sanitizer tool (clean = 0 errors and test exit 0)\n")
print("| program | " + " | ".join(tools) + " |")
print("|---|" + "---|" * len(tools))
for p in progs:
    print(f"| {p} | " + " | ".join(cell.get((p, t), "-") for t in tools) + " |")
PY
cat "$OUT/summary.md"
echo "results: $OUT"
# job state reflects the outcome: non-zero if any run is not clean
! grep -q -v -E ',clean,|,n/a_host_only,|^program,' "$OUT/summary.csv"
