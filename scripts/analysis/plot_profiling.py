#!/usr/bin/env python3
"""scripts/analysis/plot_profiling.py -- report figures from the final profiling CSVs (job 590258).

Usage:
    python3 scripts/analysis/plot_profiling.py results/final_590258 -o docs/figures

Reads ONLY existing CSVs (no GPU work, no re-analysis) from <results_dir>/analysis/ (or from
<results_dir> itself if it has no analysis/ subdir):
    nsys_summary_<wl>.csv                 per-path nsys summary (nsys_report.py)
    ncu_summary_<wl>_emu_grouped.csv      per-kernel ncu summary (ncu_report.py)
    ncu_summary_<wl>_auto.csv             same for AUTO (header-only when every problem fell back)
    roofline_final.csv                    roofline points (roofline.py)
Writes <out_dir>/fig_*.png (200 dpi) and fig_*.svg; see docs/figures/README.md.

Units (from nsys_report.py): every nsys *_ns value is the MEDIAN PER CALL over the timed calls
of that path (`calls` = number of timed calls), so nothing is divided by `calls`.
*_pct = share of GPU busy time of that call (%).
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WORKLOADS = ["uni_G1024_s32", "moe_E64_prefill_z1.2_up", "het_G256_16-2048"]
WL_SHORT = {"uni_G1024_s32": "uni", "moe_E64_prefill_z1.2_up": "moe", "het_G256_16-2048": "het"}

# Reference palette (dataviz/references/palette.md), light-mode categorical slots in order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# One color per entity, identical in every figure: paths take slots 1-5, stages slots 6-8.
PATHS = ["native_loop", "native_grouped", "emu_loop", "emu_grouped", "auto"]
PATH_COLOR = dict(zip(PATHS, SLOTS[:5]))
PATH_MARKER = {"native_loop": "o", "native_grouped": "s", "emu_loop": "^", "emu_grouped": "D", "auto": "v"}
STAGES = ["prep", "i8_gemm", "reconstruct", "other"]
STAGE_COLOR = {"prep": SLOTS[5], "i8_gemm": SLOTS[6], "reconstruct": SLOTS[7], "other": "#9ca3af"}
STAGE_LABEL = {"prep": "prep (scale + residue)", "i8_gemm": "INT8 grouped GEMM",
               "reconstruct": "reconstruct (CRT)", "other": "other + memops"}
NEUTRAL = "#9ca3af"            # host-side bars (not a path, not a stage)

INK, INK2 = "#1f2937", "#4b5563"
GRID, AXIS = "#e5e7eb", "#d1d5db"
EDGE = dict(edgecolor="white", linewidth=2)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "svg.fonttype": "none",
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, which="major", color=GRID, linewidth=0.6, linestyle="-")
    ax.tick_params(length=0 if grid_axis == "y" else 3, which="both")


def text_on(fill):
    """Label color for text drawn inside a filled mark: dark gray or white, whichever contrasts more."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    lf, ld = lum(fill), lum(INK)
    return INK if (lf + 0.05) / (ld + 0.05) >= 1.05 / (lf + 0.05) else "white"


def fmt3(v):
    if not np.isfinite(v):
        return "n/a"
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def footnote(fig, src_rel, files, job, extra=""):
    names = files if isinstance(files, str) else ", ".join(files)
    fig.text(0.01, 0.01, f"source: {src_rel}/{names} (job {job}){extra}", fontsize=7, color=INK2,
             ha="left", va="bottom")


def save(fig, out, name):
    for ext, kw in (("png", {"dpi": 200}), ("svg", {})):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, **kw)
        print(f"[plot_profiling] wrote {p}")
    plt.close(fig)


def read_csv(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        print(f"[plot_profiling] missing {p}", file=sys.stderr)
        return None
    return pd.read_csv(p)


def load_nsys(d):
    out = {}
    for w in WORKLOADS:
        df = read_csv(d, f"nsys_summary_{w}.csv")
        if df is not None:
            out[w] = df.set_index("path")
    return out


# ------------------------------------------------------------------------------- figure 1
def fig_stage_breakdown(nsys, out, src, job):
    rows = []
    for w in WORKLOADS:
        for p in ("emu_loop", "emu_grouped"):
            if w in nsys and p in nsys[w].index:
                r = nsys[w].loc[p]
                other = sum(float(r.get(c, 0.0)) for c in ("memops_pct", "other_pct", "cublas_pct"))
                rows.append((f"{w}  |  {p}", {"prep": float(r["prep_pct"]), "i8_gemm": float(r["i8_gemm_pct"]),
                                               "reconstruct": float(r["reconstruct_pct"]), "other": other}))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ys = np.arange(len(rows))[::-1]
    h = 0.5
    for y, (_, sh) in zip(ys, rows):
        tot = sum(sh.values()) or 1.0
        left = 0.0
        for s in STAGES:
            v = 100.0 * sh[s] / tot      # renormalise to exactly 100 (rounding in the CSV)
            ax.barh(y, v, left=left, height=h, color=STAGE_COLOR[s], **EDGE,
                    label=STAGE_LABEL[s] if y == ys[0] else None)
            if v >= 8.0:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=text_on(STAGE_COLOR[s]))
            left += v
    ax.set_yticks(ys, [lbl for lbl, _ in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)])
    ax.set_xlabel("share of GPU busy time per call (stage_ns / busy_ns, median of timed calls)")
    style(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_title("GPU busy time by stage, emulated paths", pad=24)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1.2,
              borderaxespad=0.3, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    footnote(fig, src, "nsys_summary_<workload>.csv", job)
    save(fig, out, "fig_stage_breakdown")


# ------------------------------------------------------------------------------- figure 2
def fig_gpu_span(nsys, out, src, job):
    paths = ["native_loop", "native_grouped", "emu_loop", "emu_grouped"]
    wls = [w for w in WORKLOADS if w in nsys]
    if not wls:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bw = 0.17
    x = np.arange(len(wls))
    for i, p in enumerate(paths):
        vals = [float(nsys[w].loc[p, "span_ns"]) / 1e6 if p in nsys[w].index else np.nan for w in wls]
        xs = x + (i - (len(paths) - 1) / 2) * bw
        ax.bar(xs, vals, width=bw, color=PATH_COLOR[p], label=p, **EDGE)
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xi, v * 1.12, fmt3(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    lo = np.nanmin([float(nsys[w].loc[p, "span_ns"]) / 1e6 for w in wls for p in paths if p in nsys[w].index])
    hi = np.nanmax([float(nsys[w].loc[p, "span_ns"]) / 1e6 for w in wls for p in paths if p in nsys[w].index])
    ax.set_ylim(10 ** np.floor(np.log10(lo)), 10 ** (np.ceil(np.log10(hi)) + 0.3))
    ax.set_xticks(x, wls, color=INK)
    ax.set_ylabel("GPU span per call (ms, log)\nfirst GPU op start to last GPU op end, median")
    style(ax)
    ax.set_title("GPU span per grouped-GEMM call", pad=24)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1.2,
              borderaxespad=0.3, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    footnote(fig, src, "nsys_summary_<workload>.csv", job, extra="; column span_ns (already per call)")
    save(fig, out, "fig_gpu_span")


# ------------------------------------------------------------------------------- figure 3
def fig_host_vs_gpu(nsys, out, src, job):
    paths = [p for p in ("emu_grouped", "auto") if any(p in nsys[w].index for w in nsys)]
    wls = [w for w in WORKLOADS if w in nsys]
    if not paths or not wls:
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(4.2 * len(paths) + 0.6, 4.4), sharey=True, squeeze=False)
    allv = []
    for ax, p in zip(axes[0], paths):
        x = np.arange(len(wls))
        bw = 0.3
        plan = [float(nsys[w].loc[p, "plan_ns"]) / 1e3 if p in nsys[w].index else np.nan for w in wls]
        busy = [float(nsys[w].loc[p, "busy_ns"]) / 1e3 if p in nsys[w].index else np.nan for w in wls]
        allv += [v for v in plan + busy if np.isfinite(v) and v > 0]
        for off, vals, col, lbl in ((-bw / 2, plan, NEUTRAL, "host plan (CPU, plan_ns)"),
                                    (bw / 2, busy, PATH_COLOR[p], f"GPU busy ({p}, busy_ns)")):
            ax.bar(x + off, vals, width=bw, color=col, label=lbl, **EDGE)
            for xi, v in zip(x + off, vals):
                if np.isfinite(v):
                    ax.text(xi, v * 1.12, fmt3(v), ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(x, [WL_SHORT[w] for w in wls], color=INK)
        style(ax)
        ax.set_title(p, fontsize=10, pad=22)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5, handlelength=1.2,
                  borderaxespad=0.2, labelcolor=INK)
    if allv:
        axes[0][0].set_ylim(10 ** np.floor(np.log10(min(allv))), 10 ** (np.ceil(np.log10(max(allv))) + 0.3))
    axes[0][0].set_ylabel("time per call (µs, log), median of timed calls")
    fig.suptitle("Host plan time vs GPU busy time per call", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    footnote(fig, src, "nsys_summary_<workload>.csv", job,
             extra="; uni = uni_G1024_s32, moe = moe_E64_prefill_z1.2_up, het = het_G256_16-2048")
    save(fig, out, "fig_host_vs_gpu")


# ------------------------------------------------------------------------------- figure 4
def fig_roofline(d, out, src, job):
    df = read_csv(d, "roofline_final.csv")
    if df is None:
        return
    df["path_key"] = df["path"].str.replace(r"^auto.*$", "auto", regex=True)
    # Roofs derived from the CSV itself (roofline.py: roof = min(peak, AI * BW)).
    mem = df[(df["bound"].str.strip() == "memory") & (df["ai"] > 0)]
    bw_all = mem["roof_gflops"] * 1e9 / mem["ai"]
    bw = float(bw_all.median()) if len(bw_all) else np.nan
    if len(bw_all) and (bw_all.max() - bw_all.min()) / bw > 1e-3:
        print(f"[plot_profiling] warning: DRAM BW implied by memory-bound rows varies "
              f"{bw_all.min():.4g}..{bw_all.max():.4g} B/s", file=sys.stderr)
    comp_nat = df[(df["bound"].str.strip() == "compute") & ~df["path"].str.startswith("emu")]
    f64 = float(comp_nat["roof_gflops"].median()) * 1e9 if len(comp_nat) else np.nan
    comp_emu = df[(df["bound"].str.strip() == "compute") & df["path"].str.startswith("emu")]
    emu_peaks = comp_emu.groupby("N")["roof_gflops"].median().to_dict()   # empty if all emu memory-bound

    sel = df[df["workload"].isin(WORKLOADS) & df["path_key"].isin(PATHS)]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    xlo = min(0.2, sel["ai"].min() / 1.5)
    xhi = max(100.0, sel["ai"].max() * 1.5)
    xs = np.logspace(np.log10(xlo), np.log10(xhi), 200)
    if np.isfinite(bw) and np.isfinite(f64):
        ax.plot(xs, np.minimum(f64, xs * bw) / 1e9, color=INK, linewidth=1.4,
                label=f"FP64 native roof: min({f64 / 1e12:.1f} TFLOPS, AI x {bw / 1e12:.2f} TB/s)")
        ax.plot(xs[xs * bw > f64], xs[xs * bw > f64] * bw / 1e9, color=INK2, linewidth=1.0, linestyle="--",
                label="DRAM roof AI x BW (emulated; compute cap INT8 peak/N not in CSV)"
                if not emu_peaks else "DRAM roof AI x BW (emulated)")
    elif np.isfinite(bw):
        ax.plot(xs, xs * bw / 1e9, color=INK2, linewidth=1.0, linestyle="--", label=f"DRAM roof AI x {bw / 1e12:.2f} TB/s")
    for N, pk in sorted(emu_peaks.items()):
        ax.plot(xs, np.minimum(pk * 1e9, xs * bw) / 1e9, color=INK2, linewidth=1.0, linestyle=":",
                label=f"emulated roof N={N}: {pk / 1e3:.1f} TFLOPS-eq")
    # only these paths get point labels (native_grouped / auto sit on top of each other)
    offs = {"native_loop": (4, -12, "left"), "emu_loop": (6, -11, "left"), "emu_grouped": (-6, 6, "right")}
    for p in PATHS:
        sp = sel[sel["path_key"] == p]
        if sp.empty:
            continue
        hollow = p == "auto"
        ax.scatter(sp["ai"], sp["gflops"], marker=PATH_MARKER[p], s=46, zorder=3,
                   facecolors="none" if hollow else PATH_COLOR[p],
                   edgecolors=PATH_COLOR[p] if hollow else "white", linewidths=1.6 if hollow else 0.8,
                   label=p if p != "auto" else "auto (all fallback, = native_grouped)")
        if p not in offs:
            continue
        dx, dy, ha = offs[p]
        for _, r in sp.iterrows():
            ax.annotate(f"{WL_SHORT[r['workload']]} {r['pct_roof']:.1f}%", (r["ai"], r["gflops"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ylo = 10 ** np.floor(np.log10(sel["gflops"].min()))
    yhi = max(sel["gflops"].max(), f64 / 1e9 if np.isfinite(f64) else 0) * 4
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("arithmetic intensity (FP64-equivalent flop / modeled DRAM byte, column ai)")
    ax.set_ylabel("achieved FP64-equivalent GFLOPS (column gflops, log)")
    style(ax, grid_axis="both")
    ax.set_title("Roofline, label = workload and % of the path's own roof (pct_roof)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, labelcolor=INK, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    footnote(fig, src, "roofline_final.csv", job,
             extra="; roofs derived from roof_gflops of memory-/compute-bound rows")
    save(fig, out, "fig_roofline")


# ------------------------------------------------------------------------------- figure 5
def fig_ncu_kernels(d, out, src, job):
    metrics = [("imma_pct", "tensor (IMMA)\npipe %"), ("dram_pct", "DRAM\nthroughput %"),
               ("sm_pct", "SM\nthroughput %")]
    stages = ["prep", "i8_gemm", "reconstruct"]
    panels, empty = [], []
    for p in ("emu_grouped", "auto"):
        for w in WORKLOADS:
            df = read_csv(d, f"ncu_summary_{w}_{p}.csv")
            if df is None:
                continue
            df = df[df["stage"].isin(stages)]
            if len(df):
                panels.append((p, w, df.groupby("stage")[[m for m, _ in metrics]].median()))
            else:
                empty.append(f"{w}_{p}")
    if not panels:
        return
    rows = sorted({p for p, _, _ in panels}, key=["emu_grouped", "auto"].index)
    fig, axes = plt.subplots(len(rows), len(WORKLOADS), figsize=(10.0, 3.9 * len(rows) + 0.4),
                             sharey=True, squeeze=False)
    bw = 0.24
    for ri, p in enumerate(rows):
        for ci, w in enumerate(WORKLOADS):
            ax = axes[ri][ci]
            hit = [t for t in panels if t[0] == p and t[1] == w]
            style(ax)
            ax.set_ylim(0, 100)
            ax.set_title(f"{w}  |  {p}", fontsize=9, pad=4)
            if not hit:
                ax.text(0.5, 0.5, "no kernel rows", transform=ax.transAxes, ha="center", color=INK2)
                continue
            g = hit[0][2]
            x = np.arange(len(metrics))
            for si, s in enumerate(stages):
                if s not in g.index:
                    continue
                vals = [float(g.loc[s, m]) for m, _ in metrics]
                xs = x + (si - 1) * bw
                ax.bar(xs, vals, width=bw, color=STAGE_COLOR[s], label=STAGE_LABEL[s], **EDGE)
                for xi, v in zip(xs, vals):
                    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
            ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=7.5, color=INK)
        axes[ri][0].set_ylabel("% of peak (ncu, median of launches)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=8,
               handlelength=1.2, labelcolor=INK)
    fig.suptitle("Nsight Compute utilisation per emulation kernel", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    extra = ""
    if empty:
        extra = (f"\nauto CSVs empty: AUTO fell back to cuBLAS on all {len(empty)} workloads"
                 if len(empty) == len(WORKLOADS) else
                 "\nno kernel rows (AUTO fell back to cuBLAS): " + ", ".join(e.split("_auto")[0] for e in empty))
    footnote(fig, src, "ncu_summary_<workload>_emu_grouped.csv", job, extra=extra)
    save(fig, out, "fig_ncu_kernels")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_dir", help="e.g. results/final_590258 (or its analysis/ subdir)")
    ap.add_argument("-o", "--out", default="docs/figures", help="output directory")
    a = ap.parse_args()
    d = a.results_dir.rstrip("/")
    if os.path.isdir(os.path.join(d, "analysis")):
        d = os.path.join(d, "analysis")
    if not os.path.isdir(d):
        sys.exit(f"[plot_profiling] no such directory: {d}")
    os.makedirs(a.out, exist_ok=True)
    m = re.search(r"_(\d{5,})", os.path.abspath(d))
    job = m.group(1) if m else "?"
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ad = os.path.abspath(d)
    src = os.path.relpath(ad, repo) if ad.startswith(repo + os.sep) else d

    nsys = load_nsys(d)
    fig_stage_breakdown(nsys, a.out, src, job)
    fig_gpu_span(nsys, a.out, src, job)
    fig_host_vs_gpu(nsys, a.out, src, job)
    fig_roofline(d, a.out, src, job)
    fig_ncu_kernels(d, a.out, src, job)


if __name__ == "__main__":
    main()
