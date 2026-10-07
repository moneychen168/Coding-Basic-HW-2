#!/usr/bin/env python3
"""scripts/analysis/nsys_report.py -- per-path launch/busy/gap breakdown from an nsys capture (owner: agent-F).

Input: either a .nsys-rep (converted with `nsys stats`, CPU-only post-processing -- run it inside a
SLURM job like everything else) or the prefix of already-exported CSVs:
    <prefix>_cuda_gpu_trace.csv   (required)  every kernel / memcpy with GPU start+duration+CorrId
    <prefix>_cuda_api_trace.csv   (required)  every CUDA API call with host start+duration+CorrID+TID
    <prefix>_cuda_api_sum.csv     (optional)  whole-process API summary (copied into the report)
    <prefix>_nvtx_pushpop_trace.csv (optional) NVTX push/pop ranges -> exact path/stage attribution

Usage:
    python3 scripts/analysis/nsys_report.py results/prof_<job>/nsys_<w>.nsys-rep -o results/prof_<job>/analysis
    python3 scripts/analysis/nsys_report.py results/prof_<job>/analysis/nsys_<w>      # CSV prefix
    options: --paths native_loop,native_grouped,emu_loop,emu_grouped,auto  (order of --paths all)
             --nsys /path/to/nsys   --no-plot   --label <name used in output files>

How a "call" (one timed ggemm_grouped_dgemm) is found
  * NVTX mode (preferred, if the trace has ranges whose name contains `ggemm:<path>`): one call =
    one such range; kernels belong to it if their launching API call lies inside the range on the
    same thread. If ranges named `*timed*` exist, only calls nested in them are kept (no warm-up).
    Stage ranges named prep / i8_gemm / reconstruct / cublas / h2d override the name-based stage.
  * Fallback (no NVTX; matches bench/bench_grouped.cu's timed loop): a timed call is bracketed by
    the bench's cudaEventRecord(e0) -- the first Record right after a host *Synchronize -- and
    cudaEventRecord(e1) -- the Record immediately followed by cudaEventSynchronize. Warm-up and
    the untimed check call have no such bracket and are excluded. Calls are grouped into paths by
    the ggemm_destroy() cudaFree burst between handles, then named in --paths order; a kernel-
    signature check warns if e.g. a "native" segment contains INT8 kernels.

Per-call metrics (median over the timed calls of each path)
    kernels     GPU kernels launched by the call (memcpy/memset counted separately as memops)
    span_us     first GPU op start -> last GPU op end
    busy_us     union of GPU op intervals inside the span
    gap_us      span - busy  (GPU idle between ops = launch gaps / host-bound stalls)
    host_api_us sum of CUDA API durations issued by the call (launch + memcpyAsync + sync ...)
    host_us     host time from e0 record end to e1 record start (whole host side of the call)
    <stage>_pct share of busy_us per stage (prep, i8_gemm, reconstruct, cublas, memops, other)
Outputs (in -o): nsys_summary_<label>.md / .csv, nsys_timeline_<label>.png (if matplotlib).
"""
import argparse
import bisect
import csv
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (GAP_COLOR, INK, INK2, PATH_ORDER, STAGE_COLOR, STAGE_LABEL, STAGE_ORDER,  # noqa: E402
                     Table, classify_stage, ensure_dir, fmt, get_pyplot, is_bench_kernel, md_table, median,
                     read_csv_skip_preamble, style_axes, union_length)

REPORTS = ["cuda_gpu_trace", "cuda_api_sum", "cuda_api_trace", "nvtx_pushpop_trace"]
_RE_PATH_RANGE = re.compile(r"ggemm[:/]([a-z][a-z0-9_]*)")   # ggemm:<path name>; stage ranges carry no prefix
_NVTX_STAGE = [(re.compile(r"\bprep\b", re.I), "prep"), (re.compile(r"i8_gemm|int8", re.I), "i8_gemm"),
               (re.compile(r"reconstruct|crt", re.I), "reconstruct"), (re.compile(r"cublas|native", re.I), "cublas"),
               (re.compile(r"h2d|desc|memcpy", re.I), "memops")]


# ---------------------------------------------------------------- nsys invocation
def find_nsys(explicit=None):
    cands = [explicit, os.environ.get("NSYS"), "/usr/local/cuda-13.0/bin/nsys"]
    cands += sorted(glob.glob("/usr/local/cuda-13.0/nsight-systems-*/target-linux-x64/nsys"), reverse=True)
    cands += sorted(glob.glob("/opt/nvidia/nsight-systems/*/target-linux-x64/nsys"), reverse=True)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise SystemExit("nsys not found (looked in /usr/local/cuda-13.0 and /opt/nvidia/nsight-systems); use --nsys")


def export_csv(rep, out_dir, nsys):
    """`nsys stats` -> <out_dir>/<stem>_<report>.csv. Returns the prefix."""
    stem = os.path.join(out_dir, os.path.basename(rep)[: -len(".nsys-rep")])
    cmd = [nsys, "stats", "--report", ",".join(REPORTS), "--format", "csv", "--output", stem,
           "--force-export", "true", "--force-overwrite", "true", rep]
    print("[nsys]", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:   # a missing report (e.g. no NVTX) is a warning, not a failure, in recent nsys
        print(r.stdout[-2000:], file=sys.stderr)
    return stem


# ---------------------------------------------------------------- loading
def load(prefix):
    gt = Table(*read_csv_skip_preamble(prefix + "_cuda_gpu_trace.csv", ["start", "duration", "name"]))
    at = Table(*read_csv_skip_preamble(prefix + "_cuda_api_trace.csv", ["start", "duration", "name"]))
    ops = []
    for r in gt.rows:
        name = gt.get_str(r, "name")
        s = gt.get_ns(r, "start")
        d = gt.get_ns(r, "duration")
        if s != s or d != d:
            continue
        corr = gt.get_str(r, "corrid", "correlationid") if gt.has("corrid", "correlationid") else ""
        ops.append({"name": name, "start": s, "end": s + d, "corr": corr.strip(), "stage": classify_stage(name)})
    ops.sort(key=lambda o: o["start"])
    apis = []
    for r in at.rows:
        s = at.get_ns(r, "start")
        d = at.get_ns(r, "duration")
        if s != s:
            continue
        apis.append({"name": at.get_str(r, "name").strip(), "start": s, "end": s + (d if d == d else 0.0),
                     "corr": at.get_str(r, "corrid", "correlationid").strip() if at.has("corrid", "correlationid") else "",
                     "tid": at.get_str(r, "tid").strip() if at.has("tid") else "0"})
    apis.sort(key=lambda a: a["start"])
    nvtx = []
    p = prefix + "_nvtx_pushpop_trace.csv"
    if os.path.isfile(p):
        try:
            nt = Table(*read_csv_skip_preamble(p, ["start", "end", "name"]))
            for r in nt.rows:
                nvtx.append({"name": nt.get_str(r, "name"), "start": nt.get_ns(r, "start"), "end": nt.get_ns(r, "end"),
                             "tid": nt.get_str(r, "tid").strip() if nt.has("tid") else "0"})
        except ValueError:
            pass
    api_sum = None
    p = prefix + "_cuda_api_sum.csv"
    if os.path.isfile(p):
        try:
            api_sum = Table(*read_csv_skip_preamble(p, ["name"]))
        except ValueError:
            api_sum = None
    return ops, apis, nvtx, api_sum


# ---------------------------------------------------------------- call windows
def _is_sync(n):
    return "Synchronize" in n


def windows_from_events(apis, paths):
    """Fallback: bench e0/e1 brackets, segmented into paths by cudaFree bursts."""
    by_tid = {}
    for a in apis:
        by_tid.setdefault(a["tid"], []).append(a)
    calls = []
    for tid, seq in by_tid.items():
        seg = 0
        e0 = None
        prev = None
        seg_has_call = False
        for i, a in enumerate(seq):
            n = a["name"]
            if n.startswith("cudaFree"):
                e0 = None
                if seg_has_call:   # ggemm_destroy / free_workload -> next handle
                    seg += 1
                    seg_has_call = False
            if n.startswith("cudaEventRecord"):
                nxt = seq[i + 1]["name"] if i + 1 < len(seq) else ""
                if e0 is not None and nxt.startswith("cudaEventSynchronize"):
                    calls.append({"tid": tid, "seg": seg, "t0": e0["end"], "t1": a["start"], "path": None})
                    seg_has_call = True
                    e0 = None
                elif prev is not None and _is_sync(prev["name"]):
                    e0 = a   # latest Record-after-Synchronize before e1 = bench e0
            prev = a
    segs = sorted({c["seg"] for c in calls})
    seg_name = {s: (paths[i] if i < len(paths) else f"segment{i}") for i, s in enumerate(segs)}
    for c in calls:
        c["path"] = seg_name[c["seg"]]
    return calls


def windows_from_nvtx(nvtx):
    calls = []
    timed = [r for r in nvtx if "timed" in r["name"].lower()]
    for r in nvtx:
        m = _RE_PATH_RANGE.search(r["name"])
        if not m:
            continue
        if timed and not any(t["tid"] == r["tid"] and t["start"] <= r["start"] and r["end"] <= t["end"] for t in timed):
            continue
        calls.append({"tid": r["tid"], "seg": m.group(1), "t0": r["start"], "t1": r["end"], "path": m.group(1)})
    return calls


def _stage_of_range(name):
    if _RE_PATH_RANGE.search(name):
        return None
    for rx, st in _NVTX_STAGE:
        if rx.search(name):
            return st
    return None


def nvtx_stage_sweep(nvtx, launches):
    """Innermost NVTX stage range enclosing each launch time. launches: [(t, tid, op)] sorted by t.
    Push/pop ranges are nested per thread, so a per-thread stack sweep is exact and O(n log n)."""
    out = {}
    by_tid = {}
    for r in nvtx:
        st = _stage_of_range(r["name"])
        if st:
            by_tid.setdefault(r["tid"], []).append((r["start"], r["end"], st))
    for tid in by_tid:
        by_tid[tid].sort()
    idx = {tid: 0 for tid in by_tid}
    stack = {tid: [] for tid in by_tid}
    for (t, tid, o) in launches:
        rs = by_tid.get(tid)
        if not rs:
            continue
        i, stk = idx[tid], stack[tid]
        while i < len(rs) and rs[i][0] <= t:
            stk.append(rs[i])
            i += 1
        idx[tid] = i
        while stk and stk[-1][1] < t:      # innermost first; ranges are nested
            stk.pop()
        live = [r for r in stk if r[1] >= t]
        stack[tid] = live
        if live:
            out[id(o)] = live[-1][2]
    return out


# ---------------------------------------------------------------- analysis
def analyze(ops, apis, nvtx, paths):
    use_nvtx = any(_RE_PATH_RANGE.search(r["name"]) for r in nvtx)
    calls = windows_from_nvtx(nvtx) if use_nvtx else windows_from_events(apis, paths)
    api_by_corr = {a["corr"]: a for a in apis if a["corr"]}
    ops_launch = []
    for o in ops:
        a = api_by_corr.get(o["corr"])
        if a is None or is_bench_kernel(o["name"]):
            continue
        ops_launch.append((a["start"], a["tid"], o))
    ops_launch.sort(key=lambda x: x[0])
    if use_nvtx:
        st_map = nvtx_stage_sweep(nvtx, ops_launch)
        for (_, _, o) in ops_launch:
            st = st_map.get(id(o))
            if st and o["stage"] in ("other", "memops", st):
                o["stage"] = st
    by_tid_ops, by_tid_api = {}, {}
    for x in ops_launch:
        by_tid_ops.setdefault(x[1], []).append(x)
    for a in apis:
        by_tid_api.setdefault(a["tid"], []).append(a)
    t_ops = {k: [x[0] for x in v] for k, v in by_tid_ops.items()}
    t_api = {k: [a["start"] for a in v] for k, v in by_tid_api.items()}
    # host time inside NVTX "plan" ranges (CPU-only planning / descriptor packing)
    plan_r = sorted((r["start"], r["end"], r["tid"]) for r in nvtx if r["name"].split(":")[-1].strip() == "plan")
    plan_t = [r[0] for r in plan_r]
    for c in calls:
        tid = c["tid"]
        lo = bisect.bisect_left(t_ops.get(tid, []), c["t0"])
        hi = bisect.bisect_right(t_ops.get(tid, []), c["t1"])
        mine = [x[2] for x in by_tid_ops.get(tid, [])[lo:hi]]
        c["ops"] = mine
        lo = bisect.bisect_left(t_api.get(tid, []), c["t0"])
        hi = bisect.bisect_right(t_api.get(tid, []), c["t1"])
        c["host_api_ns"] = sum(a["end"] - a["start"] for a in by_tid_api.get(tid, [])[lo:hi] if a["end"] <= c["t1"])
        c["host_ns"] = c["t1"] - c["t0"]
        lo = bisect.bisect_left(plan_t, c["t0"])
        hi = bisect.bisect_right(plan_t, c["t1"])
        c["plan_ns"] = sum(e - s for (s, e, t) in plan_r[lo:hi] if t == tid and e <= c["t1"])
        if mine:
            s0, s1 = min(o["start"] for o in mine), max(o["end"] for o in mine)
            c["span_ns"] = s1 - s0
            c["busy_ns"] = union_length([(o["start"], o["end"]) for o in mine])
        else:
            c["span_ns"] = c["busy_ns"] = 0.0
        c["gap_ns"] = c["span_ns"] - c["busy_ns"]
        c["kernels"] = sum(1 for o in mine if o["stage"] != "memops")
        c["memops"] = sum(1 for o in mine if o["stage"] == "memops")
        c["stage_ns"] = {s: sum(o["end"] - o["start"] for o in mine if o["stage"] == s) for s in STAGE_ORDER}
    return calls, use_nvtx


def summarize(calls):
    order = [p for p in PATH_ORDER if any(c["path"] == p for c in calls)]
    order += sorted({c["path"] for c in calls} - set(order))
    rows, warn = [], []
    for p in order:
        cs = [c for c in calls if c["path"] == p]
        r = {"path": p, "calls": len(cs)}
        for k in ("kernels", "memops", "span_ns", "busy_ns", "gap_ns", "host_api_ns", "host_ns", "plan_ns"):
            r[k] = median([c[k] for c in cs])
        for s in STAGE_ORDER:
            r[s + "_ns"] = median([c["stage_ns"][s] for c in cs])
        tot = sum(r[s + "_ns"] for s in STAGE_ORDER)
        for s in STAGE_ORDER:
            r[s + "_pct"] = 100.0 * r[s + "_ns"] / tot if tot > 0 else float("nan")
        # median call = the call whose span is closest to the median span (used for the timeline)
        r["rep_call"] = min(cs, key=lambda c: abs(c["span_ns"] - r["span_ns"])) if cs else None
        has_i8 = r["i8_gemm_ns"] > 0
        if p.startswith("native") and has_i8:
            warn.append(f"{p}: contains INT8 kernels -> path order (--paths) probably wrong")
        if p in ("emu_loop", "emu_grouped") and not has_i8 and r["cublas_ns"] > 0:
            warn.append(f"{p}: only cuBLAS kernels -> path order (--paths) probably wrong")
        rows.append(r)
    return rows, warn


# ---------------------------------------------------------------- outputs
def write_outputs(rows, warn, use_nvtx, api_sum, out, label):
    us = lambda ns: ns / 1e3  # noqa: E731
    hdr = ["path", "timed calls", "kernels/call", "memops/call", "GPU span (us)", "GPU busy (us)", "launch gaps (us)",
           "gap % of span", "host API (us)", "host plan (us)", "host total (us)"] + [STAGE_LABEL[s] + " %" for s in STAGE_ORDER]
    tab = []
    for r in rows:
        gp = 100.0 * r["gap_ns"] / r["span_ns"] if r["span_ns"] > 0 else float("nan")
        tab.append([r["path"], r["calls"], fmt(r["kernels"], 0), fmt(r["memops"], 0), fmt(us(r["span_ns"]), 1),
                    fmt(us(r["busy_ns"]), 1), fmt(us(r["gap_ns"]), 1), fmt(gp, 1), fmt(us(r["host_api_ns"]), 1), fmt(us(r["plan_ns"]), 1),
                    fmt(us(r["host_ns"]), 1)] + [fmt(r[s + "_pct"], 1) for s in STAGE_ORDER])
    md = [f"# nsys summary: {label}\n",
          f"Call detection: {'NVTX ranges ggemm:<path>' if use_nvtx else 'bench e0/e1 event brackets (no NVTX)'}; "
          "every value is the median over the timed calls of that path. Stage % = share of GPU busy time.\n\n",
          md_table(hdr, tab)]
    if warn:
        md.append("\nWarnings:\n" + "".join(f"- {w}\n" for w in warn))
    if api_sum is not None:
        top = []
        for r in api_sum.rows[:15]:
            top.append([api_sum.get_str(r, "name"), fmt(api_sum.get_num(r, "num calls"), 0),
                        fmt(api_sum.get_ns(r, "total time") / 1e3, 1), fmt(api_sum.get_ns(r, "avg") / 1e3, 2),
                        fmt(api_sum.get_num(r, "time"), 1)])
        md.append("\n## CUDA API summary (whole process, incl. setup)\n\n")
        md.append(md_table(["API", "calls", "total (us)", "avg (us)", "% of API time"], top))
    p_md = os.path.join(out, f"nsys_summary_{label}.md")
    with open(p_md, "w") as f:
        f.write("".join(md))
    p_csv = os.path.join(out, f"nsys_summary_{label}.csv")
    keys = ["path", "calls", "kernels", "memops", "span_ns", "busy_ns", "gap_ns", "host_api_ns", "plan_ns", "host_ns"] + \
           [s + "_ns" for s in STAGE_ORDER] + [s + "_pct" for s in STAGE_ORDER]
    with open(p_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    return p_md, p_csv


def plot_timeline(rows, out, label):
    plt = get_pyplot()
    if plt is None:
        print("[nsys_report] matplotlib unavailable -> no figure", file=sys.stderr)
        return None
    rows = [r for r in rows if r["rep_call"] and r["rep_call"]["ops"]]
    if not rows:
        return None
    n = len(rows)
    fig, axes = plt.subplots(n + 1, 1, figsize=(10, 1.0 * n + 2.6),
                             gridspec_kw={"height_ratios": [1] * n + [max(1.2, 0.45 * n)]})
    # (a) one representative (median-span) call per path; each row has its own time scale
    for ax, r in zip(axes[:n], rows):
        ops = r["rep_call"]["ops"]
        t0 = min(o["start"] for o in ops)
        for s in STAGE_ORDER:
            bars = [((o["start"] - t0) / 1e3, max((o["end"] - o["start"]) / 1e3, 1e-3)) for o in ops if o["stage"] == s]
            if bars:
                ax.broken_barh(bars, (0, 1), facecolors=STAGE_COLOR[s],   # no white edge when dense
                              edgecolor="white" if len(ops) <= 200 else "none", linewidth=0.3 if len(ops) <= 200 else 0)
        span = (max(o["end"] for o in ops) - t0) / 1e3
        ax.set_xlim(0, span * 1.01 if span > 0 else 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([r["path"]], color=INK)
        ax.set_ylim(0, 1)
        style_axes(ax)
        ax.text(1.0, 1.05, f"span {span:,.1f} us, {int(r['kernels'])} kernels, gaps {r['gap_ns'] / 1e3:,.1f} us",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    axes[n - 1].set_xlabel("time within one timed call (us; per-row scale)", color=INK2)
    # (b) composition of the span: busy per stage + idle gaps, as % of span
    ax = axes[n]
    names = [r["path"] for r in rows]
    left = [0.0] * n
    for s in STAGE_ORDER + ["gap"]:
        vals = [(100.0 * (r[s + "_ns"] if s != "gap" else r["gap_ns"]) / r["span_ns"]) if r["span_ns"] > 0 else 0
                for r in rows]
        if not any(v > 0 for v in vals):
            continue
        ax.barh(names, vals, left=left, height=0.6, color=STAGE_COLOR.get(s, GAP_COLOR),
                edgecolor="white", linewidth=1.0, label=STAGE_LABEL.get(s, "idle gap (launch / host)"))
        left = [a + b for a, b in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    style_axes(ax, xlabel="% of GPU span of one call (stage busy time + idle gaps)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38 - 0.25 / max(1.2, 0.45 * n)), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"nsys timeline summary: {label}", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    p = os.path.join(out, f"nsys_timeline_{label}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help=".nsys-rep file, or prefix of <prefix>_cuda_gpu_trace.csv etc.")
    ap.add_argument("-o", "--out", default=None, help="output dir (default: next to the input)")
    ap.add_argument("--paths", default=",".join(PATH_ORDER), help="path order of the bench run (fallback mode)")
    ap.add_argument("--nsys", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    inp = a.input
    out = ensure_dir(a.out or os.path.dirname(os.path.abspath(inp)))
    if inp.endswith(".nsys-rep"):
        prefix = export_csv(inp, out, find_nsys(a.nsys))
    else:
        prefix = inp[: -len("_cuda_gpu_trace.csv")] if inp.endswith("_cuda_gpu_trace.csv") else inp
    label = a.label or os.path.basename(prefix)
    ops, apis, nvtx, api_sum = load(prefix)
    calls, use_nvtx = analyze(ops, apis, nvtx, [p.strip() for p in a.paths.split(",") if p.strip()])
    if not calls:
        raise SystemExit("no timed calls found (neither NVTX ggemm:<path> ranges nor bench e0/e1 brackets)")
    rows, warn = summarize(calls)
    p_md, p_csv = write_outputs(rows, warn, use_nvtx, api_sum, out, label)
    p_png = None if a.no_plot else plot_timeline(rows, out, label)
    for w in warn:
        print("[warn]", w, file=sys.stderr)
    print(open(p_md).read())
    print("wrote:", p_md, p_csv, p_png or "(no figure)", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
