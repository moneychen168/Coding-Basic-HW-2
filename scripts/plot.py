#!/usr/bin/env python3
"""scripts/plot.py -- figures + speedup table from bench_grouped CSV (owner: agent-B).

Usage:  python scripts/plot.py results/bench_<job>/bench.csv -o results/bench_<job>/plots
        (in jobs this runs via `uv run --with matplotlib --with pandas`, see scripts/env.sh)

Outputs (one per family where applicable):
  gflops_<family>.png   effective FP64-equivalent GFLOPS per workload, bars per path (log y)
  speedup_<family>.png  time(native_loop) / time(path), reference line at 1x
  launches_<family>.png kernel launches per call (log y)
  accuracy_<workload>.png  err/gamma vs fixed num_moduli, one line per phi (if CSV has a sweep)
  speedup_table.md      median times and speedups, for the report
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (color follows the path, never its rank).
PATH_ORDER = ["native_loop", "native_grouped", "emu_loop", "emu_grouped", "auto"]
PATH_LABEL = {"native_loop": "native loop (cuBLAS DGEMM)", "native_grouped": "native grouped (cuBLAS)",
              "emu_loop": "emu loop (starter)", "emu_grouped": "emu grouped (ours)", "auto": "auto (ours+planner)"}
PATH_COLOR = {"native_loop": "#2a78d6", "native_grouped": "#eb6834", "emu_loop": "#1baf7a",
              "emu_grouped": "#eda100", "auto": "#e87ba4"}
PHI_COLOR = {0.0: "#9ec5f4", 0.5: "#5598e7", 1.0: "#256abf", 2.0: "#0d366b"}  # sequential, one hue
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e4e3dc"


def style(ax, ylabel, logy=False):
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def grouped_bars(df, value, fname, title, ylabel, logy, ref_line=None):
    wls = list(dict.fromkeys(df["workload"]))
    paths = [p for p in PATH_ORDER if p in set(df["path"])]
    if not wls or not paths:
        return
    width = 0.8 / len(paths)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(wls) * len(paths) / 2 + 2), 4.2))
    for i, p in enumerate(paths):
        sub = df[df["path"] == p].set_index("workload")
        xs = [j + (i - (len(paths) - 1) / 2) * width for j in range(len(wls))]
        ys = [sub[value].get(w, float("nan")) for w in wls]
        ax.bar(xs, ys, width=width * 0.92, color=PATH_COLOR[p], label=PATH_LABEL[p], edgecolor="white", linewidth=0.5)
    if ref_line is not None:
        ax.axhline(ref_line, color=INK2, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels(wls, rotation=40, ha="right", fontsize=7)
    style(ax, ylabel, logy)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=min(3, len(paths)))
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv)
    df = df[df["status"] == "ok"].copy()
    gpu = df["gpu"].iloc[0] if len(df) else "?"

    # performance figures: adaptive moduli (num_moduli_cfg == 0), one phi (the smallest present)
    perf = df[df["num_moduli_cfg"] == 0]
    if len(perf):
        perf = perf[perf["phi"] == perf["phi"].min()]
        base = perf[perf["path"] == "native_loop"].set_index("workload")["time_ms"]
        perf = perf.assign(speedup=[base.get(w, float("nan")) / t for w, t in zip(perf["workload"], perf["time_ms"])])
        for fam, sub in perf.groupby("family"):
            grouped_bars(sub, "effective_GFLOPS", f"{a.out}/gflops_{fam}.png",
                         f"{fam}: effective FP64-equivalent GFLOPS ({gpu})", "GFLOPS (sum 2mnk / time)", True)
            grouped_bars(sub, "speedup", f"{a.out}/speedup_{fam}.png",
                         f"{fam}: speedup over native loop ({gpu})", "x native_loop", True, ref_line=1.0)
            grouped_bars(sub, "launches", f"{a.out}/launches_{fam}.png",
                         f"{fam}: kernel launches per call", "launches", True)
        # markdown table for the report
        piv = perf.pivot_table(index="workload", columns="path", values="time_ms", aggfunc="first")
        cols = [p for p in PATH_ORDER if p in piv.columns]
        with open(f"{a.out}/speedup_table.md", "w") as f:
            f.write(f"GPU: {gpu}. time_ms = median of reps; speedup = native_loop time / path time.\n\n")
            f.write("| workload | " + " | ".join(f"{c} ms" for c in cols) + " | " +
                    " | ".join(f"{c} x" for c in cols if c != "native_loop") + " |\n")
            f.write("|" + "---|" * (1 + 2 * len(cols) - ("native_loop" in cols)) + "\n")
            for w, r in piv.iterrows():
                nl = r.get("native_loop", float("nan"))
                f.write(f"| {w} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " | " +
                        " | ".join(f"{nl / r[c]:.2f}" for c in cols if c != "native_loop") + " |\n")

    # accuracy sweep: err/gamma vs fixed moduli count, per phi
    acc = df[(df["num_moduli_cfg"] > 0) & df["err_over_gamma"].notna()]
    for (w, p), sub in acc.groupby(["workload", "path"]):
        if sub["num_moduli_cfg"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for phi in sorted(sub["phi"].unique()):
            s = sub[sub["phi"] == phi].sort_values("num_moduli_cfg")
            c = PHI_COLOR.get(float(phi), "#256abf")
            ax.plot(s["num_moduli_cfg"], s["err_over_gamma"], color=c, linewidth=2, marker="o", markersize=4,
                    label=f"phi={phi:g}")
            ax.annotate(f"phi={phi:g}", (s["num_moduli_cfg"].iloc[-1], s["err_over_gamma"].iloc[-1]),
                        textcoords="offset points", xytext=(4, 0), fontsize=7, color=INK2, va="center")
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
        ax.text(sub["num_moduli_cfg"].min(), 1.0, " FP64 bound (gamma_n)", fontsize=7, color=INK2, va="bottom")
        style(ax, "max componentwise err / gamma_n", True)
        ax.set_xlabel("number of moduli N", color=INK2)
        ax.set_title(f"{w} / {p}", color=INK, fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{a.out}/accuracy_{w}_{p}.png", dpi=150)
        plt.close(fig)
    print(f"[plot] wrote {a.out}")


if __name__ == "__main__":
    main()
