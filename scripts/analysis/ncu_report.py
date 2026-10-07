#!/usr/bin/env python3
"""scripts/analysis/ncu_report.py -- per-kernel hardware metrics table from Nsight Compute CSV (owner: agent-F).

Input: one or more CSV exports of the same .ncu-rep, merged by the report's kernel "ID":
    ncu --import X.ncu-rep --page details --csv > X_details.csv     (what scripts/profile.slurm writes)
    ncu --import X.ncu-rep --page raw     --csv > X_raw.csv         (recommended in addition: the details
        page only carries section *header* metrics; INT8 tensor-pipe and per-reason stall metrics live in
        section bodies and are only guaranteed in the raw page)
Both layouts are auto-detected: details = long format (Section Name / Metric Name / Metric Unit /
Metric Value), raw = wide format (one column per metric, 2nd row = units).

Usage:
    python3 scripts/analysis/ncu_report.py results/prof_<job>/ncu_<w>_<path>_details.csv [..._raw.csv]
            [-o out_dir] [--label name] [--all-kernels] [--list-metrics]

Kernels reported (by name): INT8 grouped GEMM (i8_grouped*), prep (oz2_prep*), reconstruct
(oz2_reconstruct*); --all-kernels adds cuBLAS/other. Several launches of one kernel -> median.

Metric selection (names checked against Nsight Compute 2025.3 ComputeWorkloadAnalysis.section):
  INT8 tensor pipe, % of peak sustained active:
    CC 7.5-9.0 (sm_90 H100/H200)  sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active
    CC 8.9                         sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active
    CC >= 10.0 (sm_120 RTX 5090)   sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_active
    fallbacks: same with .pct_of_peak_sustained_elapsed, then sm__inst_executed_pipe_tensor_{op,subpipe}_imma,
    then all tensor pipes sm__pipe_tensor_cycles_active (flagged "(all tensor)").
  DRAM throughput   dram__throughput.avg.pct_of_peak_sustained_elapsed   (details: "DRAM Throughput")
  occupancy         sm__warps_active.avg.pct_of_peak_sustained_active    (details: "Achieved Occupancy")
  registers         launch__registers_per_thread                         (details: "Registers Per Thread")
  smem/block        launch__shared_mem_per_block_static + _dynamic       (details: "... Shared Memory Per Block")
  stalls            smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio (cycles per issued
                    instruction; "selected" is not a stall and is excluded); fallback: PC-sampling counts
                    smsp__pcsamp_warps_issue_stalled_<reason> (share of samples).
Outputs: ncu_summary_<label>.md / .csv in -o (default: next to the first input).
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import classify_stage, ensure_dir, fmt, md_table, median, to_float  # noqa: E402

FOCUS = ["i8_gemm", "prep", "reconstruct"]
STAGE_NAME = {"i8_gemm": "INT8 grouped GEMM", "prep": "prep", "reconstruct": "reconstruct",
              "cublas": "cuBLAS", "other": "other"}

IMMA = {  # arch -> ordered candidates
    "sm90": ["sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
             "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
             "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
             "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_elapsed"],
    "sm89": ["sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active",
             "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed",
             "sm__inst_executed_pipe_tensor_op_imma_v2.avg.pct_of_peak_sustained_active"],
    "sm100": ["sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
              "sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_active",
              "sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_elapsed"],
}
TENSOR_ALL = ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active",
              "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
SIMPLE = {  # key -> (raw names..., details labels...)
    "duration_us": (["gpu__time_duration.sum"], ["Duration"]),
    "dram_pct": (["dram__throughput.avg.pct_of_peak_sustained_elapsed",
                  "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"], ["DRAM Throughput"]),
    "dram_GBps": (["dram__bytes.sum.per_second"], ["DRAM Bandwidth"]),
    "sm_pct": (["sm__throughput.avg.pct_of_peak_sustained_elapsed"], ["Compute (SM) Throughput", "SM Throughput"]),
    "occ_pct": (["sm__warps_active.avg.pct_of_peak_sustained_active"], ["Achieved Occupancy"]),
    "occ_theo_pct": (["sm__maximum_warps_per_active_cycle_pct"], ["Theoretical Occupancy"]),
    "regs": (["launch__registers_per_thread"], ["Registers Per Thread"]),
    "smem_static_B": (["launch__shared_mem_per_block_static"], ["Static Shared Memory Per Block"]),
    "smem_dyn_B": (["launch__shared_mem_per_block_dynamic"], ["Dynamic Shared Memory Per Block"]),
    "lim_regs": (["launch__occupancy_limit_registers"], ["Block Limit Registers"]),
    "lim_smem": (["launch__occupancy_limit_shared_mem"], ["Block Limit Shared Mem"]),
    "lim_warps": (["launch__occupancy_limit_warps"], ["Block Limit Warps"]),
}
_RE_STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
_RE_STALL_PC = re.compile(r"^smsp__pcsamp_warps_issue_stalled_(\w+)$")
_RE_STALL_LABEL = re.compile(r"^Stall (.+)$")
_UNIT_SCALE = {  # -> canonical unit of the key (us for time, bytes for memory, GB/s for bandwidth)
    "nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6,
    "byte": 1.0, "kbyte": 1e3, "mbyte": 1e6, "kib": 1024.0, "mib": 1048576.0, "b": 1.0, "kb": 1e3,
    "byte/second": 1e-9, "kbyte/second": 1e-6, "mbyte/second": 1e-3, "gbyte/second": 1.0, "tbyte/second": 1e3,
}


def _scaled(v, unit, key):
    u = (unit or "").strip().lower()
    if u.endswith("/block"):          # ncu 2025: "Kbyte/block", "byte/block"
        u = u[: -len("/block")]
    if key in ("duration_us", "smem_static_B", "smem_dyn_B", "dram_GBps") and u in _UNIT_SCALE:
        return v * _UNIT_SCALE[u]
    if key == "duration_us" and not u:   # raw page without unit row: gpu__time_duration is in ns
        return v * 1e-3
    return v


# ---------------------------------------------------------------- parsing
def _read(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = [r for r in csv.reader(f)]
    for i, r in enumerate(rows):  # skip ==PROF== banner lines
        if r and r[0].strip().strip('"') == "ID":
            return r, rows[i + 1:]
    raise ValueError(f"{path}: no ncu CSV header (first column 'ID') found")


def parse(paths):
    """-> {id: {"name", "cc", "m": {metric_or_label: (value, unit)}}}"""
    kern = {}
    for p in paths:
        hdr, rows = _read(p)
        h = [c.strip() for c in hdr]
        ix = {c: i for i, c in enumerate(h)}
        if "Metric Name" in ix:  # details (long)
            for r in rows:
                if len(r) < len(h):
                    continue
                k = kern.setdefault(r[ix["ID"]], {"name": r[ix["Kernel Name"]], "cc": r[ix["CC"]] if "CC" in ix else "", "m": {}})
                unit = r[ix["Metric Unit"]] if "Metric Unit" in ix else ""
                k["m"][r[ix["Metric Name"]].strip()] = (to_float(r[ix["Metric Value"]]), unit)
        else:  # raw (wide); second row holds units when present
            units = [""] * len(h)
            if rows and (not rows[0] or rows[0][0].strip() == ""):
                units, rows = rows[0], rows[1:]
            for r in rows:
                if not r or not r[0].strip():
                    continue
                k = kern.setdefault(r[ix["ID"]], {"name": r[ix["Kernel Name"]], "cc": r[ix["CC"]] if "CC" in ix else "", "m": {}})
                for i, c in enumerate(h):
                    if "__" in c and i < len(r):
                        k["m"][c] = (to_float(r[i]), units[i] if i < len(units) else "")
    return kern


def _arch(cc):
    try:
        v = float(cc)
    except ValueError:
        return "sm90"
    if v >= 10.0:
        return "sm100"
    if abs(v - 8.9) < 1e-6:
        return "sm89"
    return "sm90"


def _first(m, names):
    for n in names:
        if n in m and m[n][0] == m[n][0]:
            return n, m[n]
    return None, (float("nan"), "")


def extract(k):
    m = k["m"]
    out = {"name": k["name"], "cc": k["cc"], "stage": classify_stage(k["name"])}
    n, (v, _) = _first(m, IMMA[_arch(k["cc"])])
    if n is None:
        n, (v, _) = _first(m, TENSOR_ALL)
        out["imma_metric"] = (n + " (all tensor)") if n else "missing"
    else:
        out["imma_metric"] = n
    out["imma_pct"] = v
    for key, (raw, labels) in SIMPLE.items():
        n, (v, u) = _first(m, raw + labels)
        out[key] = _scaled(v, u, key) if n else float("nan")
    st = {}
    for name, (v, _) in m.items():
        mm = _RE_STALL.match(name) or _RE_STALL_LABEL.match(name)
        if mm and v == v:
            reason = mm.group(1).strip().lower().replace(" ", "_")
            if reason not in ("selected", "not_selected_total"):
                st[reason] = v
    src = "cycles/issue"
    if not st:
        for name, (v, _) in m.items():
            mm = _RE_STALL_PC.match(name)
            if mm and v == v and not mm.group(1).endswith("not_issued") and mm.group(1) != "selected":
                st[mm.group(1)] = v
        src = "pc-samples"
    tot = sum(st.values())
    out["stalls"] = sorted(((r, v, 100.0 * v / tot if tot > 0 else float("nan")) for r, v in st.items()),
                           key=lambda x: -x[1])
    out["stall_src"] = src if st else "missing"
    return out


def aggregate(kern, all_kernels):
    groups = {}
    for kid, k in sorted(kern.items(), key=lambda x: int(to_float(x[0])) if to_float(x[0]) == to_float(x[0]) else 0):
        e = extract(k)
        if not all_kernels and e["stage"] not in FOCUS:
            continue
        groups.setdefault((e["stage"], e["name"]), []).append(e)
    rows = []
    for (stage, name), es in groups.items():
        r = {"stage": stage, "kernel": name, "launches": len(es), "cc": es[0]["cc"], "imma_metric": es[0]["imma_metric"],
             "stall_src": es[0]["stall_src"]}
        for key in ["imma_pct", "duration_us"] + list(SIMPLE):
            r[key] = median([e[key] for e in es])
        reasons = {}
        for e in es:
            for (rs, v, pct) in e["stalls"]:
                reasons.setdefault(rs, []).append(pct)
        r["stalls_top"] = sorted(((rs, median(p)) for rs, p in reasons.items()), key=lambda x: -x[1])[:3]
        rows.append(r)
    order = {s: i for i, s in enumerate(FOCUS + ["cublas", "other", "memops"])}
    rows.sort(key=lambda r: (order.get(r["stage"], 9), r["kernel"]))
    return rows


def short(name, n=48):
    base = re.sub(r"^(void\s+)?((<unnamed>|\(anonymous namespace\))::)?", "", name.strip())
    base = re.split(r"[(<]", base, 1)[0].strip() or name.strip()
    return base if len(base) <= n else base[: n - 1] + "~"


def write(rows, out, label):
    hdr = ["stage", "kernel", "launches", "duration (us)", "INT8 tensor pipe %", "DRAM thru %", "SM thru %",
           "achieved occ %", "theor. occ %", "regs/thread", "smem/block (B)", "block limit reg/smem/warp", "top-3 stalls (% of stall)"]
    tab = []
    for r in rows:
        smem = (r["smem_static_B"] if r["smem_static_B"] == r["smem_static_B"] else 0) + \
               (r["smem_dyn_B"] if r["smem_dyn_B"] == r["smem_dyn_B"] else 0)
        smem_s = fmt(smem, 0) if (r["smem_static_B"] == r["smem_static_B"] or r["smem_dyn_B"] == r["smem_dyn_B"]) else "n/a"
        st = ", ".join(f"{rs} {p:.0f}%" for rs, p in r["stalls_top"]) or "n/a"
        imma = fmt(r["imma_pct"], 1) + (" (all)" if "(all tensor)" in r["imma_metric"] else "")
        tab.append([STAGE_NAME.get(r["stage"], r["stage"]), "`" + short(r["kernel"]) + "`", r["launches"],
                    fmt(r["duration_us"], 1), imma, fmt(r["dram_pct"], 1), fmt(r["sm_pct"], 1), fmt(r["occ_pct"], 1),
                    (fmt(r["occ_theo_pct"], 1) + ("*" if r["occ_theo_pct"] == 0 and r["occ_pct"] > 0 else "")), fmt(r["regs"], 0), smem_s,
                    "/".join(fmt(r[k], 0) for k in ("lim_regs", "lim_smem", "lim_warps")), st])
    used = sorted({r["imma_metric"] for r in rows})
    srcs = sorted({r["stall_src"] for r in rows})
    md = (f"# ncu summary: {label}\n\nMedian over launches of each kernel. INT8 tensor pipe metric: "
          + ", ".join(f"`{u}`" for u in used) + f". Stall source: {', '.join(srcs)}"
          " (share of all stall reasons, `selected` excluded).\n\n" + md_table(hdr, tab, "ll" + "r" * 10 + "l"))
    if any(r["occ_theo_pct"] == 0 and r["occ_pct"] > 0 for r in rows):
        md += ("\n\\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while "
               "achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.\n")
    if any(r["imma_metric"] == "missing" for r in rows) or any(r["stall_src"] == "missing" for r in rows):
        md += ("\nNote: tensor-pipe or stall metrics missing -> also export the raw page: "
               "`ncu --import X.ncu-rep --page raw --csv > X_raw.csv` and pass both CSVs.\n")
    p_md = os.path.join(out, f"ncu_summary_{label}.md")
    with open(p_md, "w") as f:
        f.write(md)
    p_csv = os.path.join(out, f"ncu_summary_{label}.csv")
    keys = ["stage", "kernel", "launches", "cc", "duration_us", "imma_pct", "imma_metric", "dram_pct", "dram_GBps",
            "sm_pct", "occ_pct", "occ_theo_pct", "regs", "smem_static_B", "smem_dyn_B", "lim_regs", "lim_smem", "lim_warps", "stall_src"]
    with open(p_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["stall1", "stall1_pct", "stall2", "stall2_pct", "stall3", "stall3_pct"])
        for r in rows:
            st = [x for rs, p in r["stalls_top"] for x in (rs, f"{p:.2f}")]
            w.writerow([r[k] for k in keys] + st + [""] * (6 - len(st)))
    return p_md, p_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", help="ncu --csv exports (details and/or raw page) of ONE report")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--all-kernels", action="store_true")
    ap.add_argument("--list-metrics", action="store_true", help="print metric names found for the first kernel")
    a = ap.parse_args(argv)
    kern = parse(a.csv)
    if not kern:
        raise SystemExit("no kernels in input")
    if a.list_metrics:
        k = next(iter(kern.values()))
        print(k["name"])
        for n in sorted(k["m"]):
            print("  ", n, k["m"][n])
        return None
    out = ensure_dir(a.out or os.path.dirname(os.path.abspath(a.csv[0])))
    label = a.label or re.sub(r"_(details|raw)$", "", os.path.splitext(os.path.basename(a.csv[0]))[0])
    rows = aggregate(kern, a.all_kernels)
    p_md, p_csv = write(rows, out, label)
    print(open(p_md).read())
    print("wrote:", p_md, p_csv, file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
