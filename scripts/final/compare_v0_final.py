"""Compare starter (v0) and final bench CSVs from scripts/final/starter_vs_final.slurm.

Usage: python3 -I compare_v0_final.py results/v0final_<jobid>
Reads v0_r{1,2}.csv and final_r{1,2}.csv, takes the median time_ms of the two runs per
(workload, path), and prints a markdown table of rows present in both versions plus geomeans.
"""
import csv
import math
import os
import statistics
import sys


def load(out, tag):
    rows = {}
    for r in (1, 2):
        p = os.path.join(out, f"{tag}_r{r}.csv")
        if not os.path.exists(p):
            continue
        for x in csv.DictReader(open(p)):
            if x.get("status", "ok") not in ("ok", ""):
                continue
            k = (x["workload"], x["path"])
            rows.setdefault(k, {"t": [], "fb": x.get("num_fallback", ""), "err": x.get("err_max", ""),
                                "launches": x.get("launches", "")})["t"].append(float(x["time_ms"]))
    return {k: dict(v, t=statistics.median(v["t"])) for k, v in rows.items() if v["t"]}


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(out):
    v0, fi = load(out, "v0"), load(out, "final")
    keys = sorted(set(v0) & set(fi))
    print(f"# Starter (v0) vs final, same job / node / GPU / workloads (`{os.path.basename(out)}`)\n")
    print("time_ms = E2E median of 2 runs per binary (each run: warmup + reps, mean per call as bench reports).")
    print("speedup = v0 / final (>1 means final is faster). Only rows present in both versions are shown.\n")
    print("| workload | path | v0 ms | final ms | speedup | v0 launches | final launches | v0 fb | final fb | v0 err_max | final err_max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_path = {}
    for k in keys:
        a, b = v0[k], fi[k]
        s = a["t"] / b["t"]
        by_path.setdefault(k[1], []).append(s)
        print(f"| {k[0]} | {k[1]} | {a['t']:.4f} | {b['t']:.4f} | {s:.2f}x | {a['launches']} | {b['launches']} "
              f"| {a['fb']} | {b['fb']} | {a['err']} | {b['err']} |")
    print("\n## Geomean speedup per path (v0 / final)\n")
    print("| path | rows | geomean | min | max |")
    print("|---|---:|---:|---:|---:|")
    for p, ss in sorted(by_path.items()):
        print(f"| {p} | {len(ss)} | {gm(ss):.2f}x | {min(ss):.2f}x | {max(ss):.2f}x |")
    only0, only1 = sorted(set(v0) - set(fi)), sorted(set(fi) - set(v0))
    print(f"\nRows only in v0: {len(only0)}; rows only in final: {len(only1)} (new paths/workloads are not compared).")
    print("\n## Starter vs final on the same problem: v0 EMU_LOOP (starter-like) and v0 EMU_GROUPED vs final EMU_GROUPED\n")
    print("| workload | v0 emu_loop ms | v0 emu_grouped ms | final emu_grouped ms | final native_grouped ms |")
    print("|---|---:|---:|---:|---:|")
    for w in sorted({k[0] for k in fi}):
        g = lambda d, p: f"{d[(w, p)]['t']:.4f}" if (w, p) in d else "-"
        print(f"| {w} | {g(v0, 'emu_loop')} | {g(v0, 'emu_grouped')} | {g(fi, 'emu_grouped')} | {g(fi, 'native_grouped')} |")


if __name__ == "__main__":
    main(sys.argv[1])
