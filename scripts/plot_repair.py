#!/usr/bin/env python3
"""scripts/plot_repair.py -- figures + tables for scripts/repair_sweep.slurm (owner: agent-B).

Usage: python scripts/plot_repair.py --emu emu_repair.csv --auto auto_repair.csv --plan auto_plan.csv -o plots

Definitions (all from bench_grouped CSV columns):
  uncert_frac   = num_uncertified / total_entries   (COUNT: entries whose certificate failed, value kept)
  repaired_frac = num_repaired    / total_entries   (ON: entries recomputed as FP64 dot products)
  repair_cost   = time_ms(ON) / time_ms(COUNT), same workload/phi/N (both evaluate the certificate,
                  so the ratio isolates the FP64 recomputation)
  err/gamma     = max componentwise error / gamma_n  (<= 1: FP64 contract met everywhere)
Outputs: emu_cost.png, emu_uncert.png, emu_err.png, auto_time.png, auto_repaired.png, auto_N.png,
         repair_summary.md
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PHI_COLOR = {0.0: "#9ec5f4", 0.5: "#5598e7", 1.0: "#256abf", 2.0: "#0d366b"}   # sequential, one hue
MODE_STYLE = {"count": "-", "on": "--"}   # line style = secondary encoding for repair mode
INK, INK2, GRID, NEUTRAL = "#1a1a19", "#5f5e58", "#e4e3dc", "#8a8983"


def style(ax, ylabel=None, xlabel=None, logy=False, symlog=None):
    if logy:
        ax.set_yscale("log")
    if symlog:
        ax.set_yscale("symlog", linthresh=symlog)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7)


def panels(n):
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    return fig, [a for row in axes for a in row]


def save(fig, path, title):
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    for c in ("num_uncertified", "num_repaired"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = float("nan")   # -1 = not measured
    df["uncert_frac"] = df["num_uncertified"] / df["total_entries"]
    df["repaired_frac"] = df["num_repaired"] / df["total_entries"]
    return df


def phi_lines(ax, sub, x, y, mode=None, label_suffix=""):
    for phi in sorted(sub["phi"].unique()):
        s = sub[sub["phi"] == phi].sort_values(x)
        if s[y].notna().any():
            ax.plot(s[x], s[y], color=PHI_COLOR.get(float(phi), "#256abf"), linewidth=2, marker="o", markersize=4,
                    linestyle=MODE_STYLE.get(mode, "-"), label=f"phi={phi:g}{label_suffix}")


def emu_figures(df, out, md):
    wls = list(dict.fromkeys(df["workload"]))
    piv = df.pivot_table(index=["workload", "phi", "num_moduli_cfg"], columns="repair_mode", values="time_ms",
                         aggfunc="first").reset_index()
    if {"count", "on"} <= set(piv.columns):
        piv["repair_cost"] = piv["on"] / piv["count"]
        fig, axes = panels(len(wls))
        for ax, w in zip(axes, wls):
            phi_lines(ax, piv[piv["workload"] == w], "num_moduli_cfg", "repair_cost")
            ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
            style(ax, "time ON / time COUNT", "N (moduli)")
            ax.set_title(w, fontsize=8, color=INK, loc="left")
        axes[0].legend(fontsize=7, frameon=False)
        for ax in axes[len(wls):]:
            ax.set_visible(False)
        save(fig, f"{out}/emu_cost.png", "EMU_GROUPED: cost of FP64 repair (ON vs COUNT)")
        md.append("## EMU_GROUPED: repair cost (time ON / time COUNT) and uncertified fraction (COUNT)\n")
        cnt = df[df["repair_mode"] == "count"].set_index(["workload", "phi", "num_moduli_cfg"])
        md.append("| workload | phi | N | t COUNT ms | t ON ms | ON/COUNT | uncert frac | err/gamma COUNT | err/gamma ON |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        on = df[df["repair_mode"] == "on"].set_index(["workload", "phi", "num_moduli_cfg"])
        for _, r in piv.iterrows():
            key = (r["workload"], r["phi"], r["num_moduli_cfg"])
            uf = cnt["uncert_frac"].get(key, float("nan"))
            ec = cnt["err_over_gamma"].get(key, float("nan"))
            eo = on["err_over_gamma"].get(key, float("nan"))
            md.append(f"| {r['workload']} | {r['phi']:g} | {int(r['num_moduli_cfg'])} | {r['count']:.4f} | "
                      f"{r['on']:.4f} | {r['repair_cost']:.3f} | {uf:.3e} | {ec:.3e} | {eo:.3e} |")
        md.append("")

    fig, axes = panels(len(wls))
    for ax, w in zip(axes, wls):
        phi_lines(ax, df[(df["workload"] == w) & (df["repair_mode"] == "count")], "num_moduli_cfg", "uncert_frac")
        style(ax, "uncertified / entries", "N (moduli)", symlog=1e-7)
        ax.set_title(w, fontsize=8, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[len(wls):]:
        ax.set_visible(False)
    save(fig, f"{out}/emu_uncert.png", "EMU_GROUPED (COUNT): fraction of entries failing the certificate")

    fig, axes = panels(len(wls))
    for ax, w in zip(axes, wls):
        for mode in ("count", "on"):
            phi_lines(ax, df[(df["workload"] == w) & (df["repair_mode"] == mode)], "num_moduli_cfg",
                      "err_over_gamma", mode=mode, label_suffix=f" {mode.upper()}")
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
        style(ax, "max err / gamma_n", "N (moduli)", logy=True)
        ax.set_title(w, fontsize=8, color=INK, loc="left")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    for ax in axes[len(wls):]:
        ax.set_visible(False)
    save(fig, f"{out}/emu_err.png", "EMU_GROUPED: accuracy (solid COUNT, dashed ON; dotted = FP64 bound)")


def auto_figures(df, plan, out, md):
    df = df.copy()
    df["rho"] = pd.to_numeric(df["rho_budget"], errors="coerce")
    auto = df[df["path"] == "auto"]
    nat = df[df["path"] == "native_loop"]
    wls = list(dict.fromkeys(auto["workload"]))
    fig, axes = panels(len(wls))
    for ax, w in zip(axes, wls):
        phi_lines(ax, auto[auto["workload"] == w], "rho", "time_ms")
        nt = nat[nat["workload"] == w]["time_ms"].median()
        if pd.notna(nt):
            ax.axhline(nt, color=NEUTRAL, linewidth=1, linestyle="--")
            ax.annotate("native loop", (auto["rho"].min(), nt), fontsize=7, color=INK2, va="bottom")
        ax.set_xscale("log", base=2)
        style(ax, "time ms (median)", "rho_budget")
        ax.set_title(w, fontsize=8, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[len(wls):]:
        ax.set_visible(False)
    save(fig, f"{out}/auto_time.png", "AUTO: time vs rho_budget")

    fig, axes = panels(len(wls))
    for ax, w in zip(axes, wls):
        phi_lines(ax, auto[auto["workload"] == w], "rho", "repaired_frac")
        ax.set_xscale("log", base=2)
        style(ax, "repaired / entries", "rho_budget", symlog=1e-7)
        ax.set_title(w, fontsize=8, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[len(wls):]:
        ax.set_visible(False)
    save(fig, f"{out}/auto_repaired.png", "AUTO (repair ON): fraction of entries recomputed in FP64")

    md.append("## AUTO: rho_budget sweep\n")
    md.append("| workload | phi | rho | time ms | native_loop ms | repaired frac | moduli [min,max] | fallback | err/gamma |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in auto.sort_values(["workload", "phi", "rho"]).iterrows():
        nt = nat[(nat["workload"] == r["workload"]) & (nat["phi"] == r["phi"])]["time_ms"].median()
        md.append(f"| {r['workload']} | {r['phi']:g} | {r['rho']:g} | {r['time_ms']:.4f} | {nt:.4f} | "
                  f"{r['repaired_frac']:.3e} | [{r['moduli_min']},{r['moduli_max']}] | {r['num_fallback']} | "
                  f"{r['err_over_gamma']:.3e} |")
    md.append("")

    if plan is not None and len(plan):
        plan = plan[plan["path"] == "auto"].copy()
        plan["rho"] = pd.to_numeric(plan["rho_budget"], errors="coerce")
        plan["bucket"] = [f"N={n}" if r == 0 else "native" for r, n in zip(plan["route"], plan["num_moduli"])]
        ns = sorted({int(n) for r, n in zip(plan["route"], plan["num_moduli"]) if r == 0})
        ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b"]
        col = {f"N={n}": ramp[min(i, len(ramp) - 1)] for i, n in enumerate(ns)}
        col["native"] = NEUTRAL
        order = [f"N={n}" for n in ns] + ["native"]
        pw = list(dict.fromkeys(plan["workload"]))
        fig, axes = panels(len(pw))
        for ax, w in zip(axes, pw):
            sub = plan[plan["workload"] == w].groupby(["rho", "bucket"])["problems"].sum().unstack(fill_value=0)
            sub = sub.div(sub.sum(axis=1), axis=0)
            rhos = list(sub.index)
            bottom = [0.0] * len(rhos)
            for b in order:
                if b not in sub.columns:
                    continue
                vals = list(sub[b])
                ax.bar([str(int(r)) for r in rhos], vals, bottom=bottom, color=col[b], label=b,
                       edgecolor="white", linewidth=1)
                bottom = [x + y for x, y in zip(bottom, vals)]
            style(ax, "share of problems", "rho_budget")
            ax.set_title(w, fontsize=8, color=INK, loc="left")
        axes[0].legend(fontsize=6, frameon=False, ncol=3)
        for ax in axes[len(pw):]:
            ax.set_visible(False)
        save(fig, f"{out}/auto_N.png", "AUTO planner: moduli count N chosen per problem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emu")
    ap.add_argument("--auto")
    ap.add_argument("--plan")
    ap.add_argument("-o", "--out", default="plots_repair")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# Repair-mode sweep\n",
          "uncert/repaired frac = count / total C entries; counters from a separate untimed call.\n"]
    emu = load(a.emu)
    if emu is not None and len(emu):
        md.insert(1, f"GPU: {emu['gpu'].iloc[0]}\n")
        emu_figures(emu, a.out, md)
    auto = load(a.auto)
    plan = pd.read_csv(a.plan) if a.plan and os.path.exists(a.plan) else None
    if auto is not None and len(auto):
        auto_figures(auto, plan, a.out, md)
    with open(f"{a.out}/repair_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[plot_repair] wrote {a.out}")


if __name__ == "__main__":
    main()
