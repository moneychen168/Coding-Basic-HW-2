#!/usr/bin/env python3
"""scripts/analysis/roofline.py -- roofline of every (workload, path) from bench_grouped CSV (owner: agent-F).

Usage:
    python3 scripts/analysis/roofline.py results/bench_<job>/bench.csv \
        --calib bench/calib/params_nvidia_h200.json [--shapes shapes.csv] [-o out_dir] [--label name] [--no-plot]

x = arithmetic intensity  AI = FP64-equivalent flops / modeled DRAM bytes   (flop/byte)
y = effective FP64-equivalent GFLOPS (bench column effective_GFLOPS = sum 2mnk / median time)

Bytes model (beta = 0 in bench_grouped):
    native_*          sum_g 8*(m*k + k*n + m*n)
    emu_* / auto      calib JSON model.bytes_emu evaluated per problem (same model the planner uses),
                      N = num_moduli_cfg if > 0 else moduli_max of the row (for auto: upper bound
                      on bytes, i.e. a lower bound on AI, since per-problem N is not in the CSV;
                      fallback problems are still counted with the emulated byte model)
Roofs (from the calib JSON, measured on that GPU):
    FP64 roof            min(f64_peak_flops, AI * bw_Bps)
    emulated roof (N)    min(i8_peak_ops / N, AI * bw_Bps)   -- FP64-equivalent rate when every FP64
                         flop costs N INT8 ops; drawn for the N values present in the CSV
Shapes: bench.csv has no per-problem shapes. Give --shapes (CSV: workload,m,n,k one row per problem,
optional count column) or the script derives them for uniform names uni_G<G>_s<s> / uni_G<G>_<m>x<n>x<k>;
other rows are skipped with a warning.
Outputs: roofline_<label>.png, roofline_<label>.md / .csv (AI, GFLOPS, % of the path's roof).
"""
import argparse
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import INK, INK2, PATH_COLOR, PATH_ORDER, ensure_dir, fmt, get_pyplot, md_table, style_axes, to_float  # noqa: E402

_RE_UNI_S = re.compile(r"^uni_G(\d+)_s(\d+)$")
_RE_UNI_MNK = re.compile(r"^uni_G(\d+)_(\d+)x(\d+)x(\d+)$")


def load_shapes(path):
    sh = {}
    if not path:
        return sh
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c = int(to_float(r.get("count", 1))) if r.get("count") else 1
            sh.setdefault(r["workload"], []).extend([(int(r["m"]), int(r["n"]), int(r["k"]))] * c)
    return sh


def shapes_for(w, given):
    if w in given:
        return given[w]
    m = _RE_UNI_S.match(w)
    if m:
        return [(int(m.group(2)),) * 3] * int(m.group(1))
    m = _RE_UNI_MNK.match(w)
    if m:
        return [(int(m.group(2)), int(m.group(3)), int(m.group(4)))] * int(m.group(1))
    return None


def _safe_eval(expr, env):
    expr = re.sub(r"\(2 if beta!=0 else 1\)", "(2 if beta != 0 else 1)", expr)
    code = compile(expr, "<bytes_emu>", "eval")
    for n in code.co_names:
        if n not in env:
            raise ValueError(f"bytes_emu uses unknown name {n!r}")
    return eval(code, {"__builtins__": {}}, env)  # names restricted to m,n,k,N,Kp,beta


def bytes_for(path, shapes, N, model_expr):
    tot = 0.0
    for (m, n, k) in shapes:
        if path.startswith("native"):
            tot += 8.0 * (m * k + k * n + m * n)
        else:
            kp = (k + 31) // 32 * 32
            tot += _safe_eval(model_expr, {"m": m, "n": n, "k": k, "N": N, "Kp": kp, "beta": 0.0})
    return tot


def build(bench_csv, calib, shapes_given):
    model = calib.get("model", {}).get("bytes_emu",
                                       "8*(m*k+k*n) + N*(m+n)*Kp + 8*N*m*n + 8*m*n*(2 if beta!=0 else 1)")
    pts, warn = [], []
    with open(bench_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            w, p = r["workload"], r["path"]
            sh = shapes_for(w, shapes_given)
            if sh is None:
                warn.append(f"{w}: no shapes (pass --shapes) -> skipped")
                continue
            flops = to_float(r["total_flops"])
            gfl = to_float(r["effective_GFLOPS"])
            ncfg = int(to_float(r.get("num_moduli_cfg", 0)) or 0)
            nmax = int(to_float(r.get("moduli_max", 0)) or 0)
            N = ncfg if ncfg > 0 else nmax
            if not p.startswith("native") and N <= 0:
                warn.append(f"{w}/{p}: moduli count unknown -> skipped")
                continue
            by = bytes_for(p, sh, N, model)
            ai = flops / by if by > 0 else float("nan")
            bw = calib["bw_Bps"]
            if p.startswith("native"):
                roof = min(calib["f64_peak_flops"], ai * bw)
            else:
                roof = min(calib["i8_peak_ops"] / N, ai * bw)
            pts.append({"workload": w, "path": p, "N": N if not p.startswith("native") else 0, "phi": r.get("phi", ""),
                        "flops": flops, "bytes": by, "ai": ai, "gflops": gfl, "roof_gflops": roof / 1e9,
                        "pct_roof": 100.0 * gfl * 1e9 / roof if roof > 0 else float("nan"),
                        "bound": "memory" if ai * bw < (calib["f64_peak_flops"] if p.startswith("native")
                                                        else calib["i8_peak_ops"] / N) else "compute"})
    return pts, sorted(set(warn))


def plot(pts, calib, out, label):
    plt = get_pyplot()
    if plt is None or not pts:
        print("[roofline] matplotlib unavailable or no points -> no figure", file=sys.stderr)
        return None
    bw, f64 = calib["bw_Bps"], calib["f64_peak_flops"]
    ais = [p["ai"] for p in pts if p["ai"] > 0]
    lo, hi = min(ais) / 3, max(ais) * 3
    lo = min(lo, f64 / bw / 10)
    hi = max(hi, calib["i8_peak_ops"] / max(2, min((p["N"] for p in pts if p["N"] > 0), default=8)) / bw * 3)
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(xs, [min(f64, x * bw) / 1e9 for x in xs], color=INK, linewidth=2, label=f"FP64 roof ({f64 / 1e12:.1f} TFLOPS)")
    Ns = sorted({p["N"] for p in pts if p["N"] > 0})
    for N in ([Ns[0], Ns[-1]] if len(Ns) > 1 else Ns):
        peak = calib["i8_peak_ops"] / N
        ax.plot(xs, [min(peak, x * bw) / 1e9 for x in xs], color=INK2, linewidth=1.2, linestyle="--")
        ax.text(hi, peak / 1e9, f" INT8/N, N={N}", color=INK2, fontsize=8, va="center", ha="left")
    ax.text(lo * 1.3, lo * 1.3 * bw / 1e9 * 1.25, f"DRAM {bw / 1e12:.2f} TB/s", color=INK2, fontsize=8, rotation=0)
    for path in PATH_ORDER + sorted({p["path"] for p in pts} - set(PATH_ORDER)):
        sel = [p for p in pts if p["path"] == path]
        if not sel:
            continue
        ax.scatter([p["ai"] for p in sel], [p["gflops"] for p in sel], s=42, color=PATH_COLOR.get(path, "#8a8983"),
                   edgecolor="white", linewidth=1.5, zorder=3, label=path)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, xlabel="arithmetic intensity (FP64-equiv. flop / modeled DRAM byte)",
               ylabel="effective FP64-equivalent GFLOPS", grid_axis="both")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Roofline: {label} ({calib.get('gpu', '?')})", color=INK, loc="left")
    fig.tight_layout()
    p = os.path.join(out, f"roofline_{label}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def write(pts, warn, out, label, calib):
    hdr = ["workload", "path", "N", "AI (flop/B)", "GFLOPS", "roof GFLOPS", "% of roof", "bound"]
    order = {p: i for i, p in enumerate(PATH_ORDER)}
    pts = sorted(pts, key=lambda p: (p["workload"], order.get(p["path"], 9)))
    tab = [[p["workload"], p["path"], p["N"] or "-", fmt(p["ai"], 2), fmt(p["gflops"], 1), fmt(p["roof_gflops"], 0),
            fmt(p["pct_roof"], 1), p["bound"]] for p in pts]
    md = (f"# Roofline: {label}\n\nGPU {calib.get('gpu', '?')}: FP64 peak {calib['f64_peak_flops'] / 1e12:.2f} TFLOPS, "
          f"INT8 peak {calib['i8_peak_ops'] / 1e12:.0f} TOPS, DRAM {calib['bw_Bps'] / 1e12:.2f} TB/s (calib JSON). "
          "Roof of a path = FP64 roof (native) or INT8 peak / N (emulated) capped by DRAM bandwidth x AI.\n\n"
          + md_table(hdr, tab, "ll" + "r" * 5 + "l"))
    if warn:
        md += "\nWarnings:\n" + "".join(f"- {w}\n" for w in warn)
    p_md = os.path.join(out, f"roofline_{label}.md")
    with open(p_md, "w") as f:
        f.write(md)
    p_csv = os.path.join(out, f"roofline_{label}.csv")
    keys = ["workload", "path", "phi", "N", "flops", "bytes", "ai", "gflops", "roof_gflops", "pct_roof", "bound"]
    with open(p_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for p in pts:
            wr.writerow([p[k] for k in keys])
    return p_md, p_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bench_csv")
    ap.add_argument("--calib", required=True, help="bench/calib/params_<gpu>.json")
    ap.add_argument("--shapes", default=None, help="CSV workload,m,n,k[,count] (one row per problem)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    calib = json.load(open(a.calib))
    for key in ("bw_Bps", "f64_peak_flops", "i8_peak_ops"):
        if not (key in calib and calib[key] > 0 and math.isfinite(calib[key])):
            raise SystemExit(f"calib JSON lacks {key}")
    out = ensure_dir(a.out or os.path.dirname(os.path.abspath(a.bench_csv)))
    label = a.label or os.path.splitext(os.path.basename(a.bench_csv))[0]
    pts, warn = build(a.bench_csv, calib, load_shapes(a.shapes))
    p_md, p_csv = write(pts, warn, out, label, calib)
    p_png = None if a.no_plot else plot(pts, calib, out, label)
    print(open(p_md).read())
    print("wrote:", p_md, p_csv, p_png or "(no figure)", file=sys.stderr)
    return pts


if __name__ == "__main__":
    main()
