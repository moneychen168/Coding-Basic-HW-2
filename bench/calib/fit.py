#!/usr/bin/env python3
"""bench/calib/fit.py — fit AUTO-planner cost-model parameters from calib.cu CSV (agent-C).

Usage: python3 fit.py calib_<jobid>.csv [-o OUTDIR]
Writes OUTDIR/params_<gpu_slug>.json (default OUTDIR = this script's directory).

Cost model (seconds):
  emu(m,n,k,N)    = 3*t_launch + bytes_emu(m,n,k,N)/bw + N*2mnk / i8_peak
  native(m,n,k)   = t_launch + 2mnk / f64_peak
  bytes_emu       = prep:  8(mk+kn) read + N(m+n)Kp written
                  + recon: 4Nmn read + 8mn read(C, if beta!=0) + 8mn written
                  (GEMM's own C32 write 4Nmn is folded in as 4Nmn extra)
Peaks: a linear fit t = t0 + work/peak over the large sizes (work-bound regime);
also the per-size measured rate tables so the planner can interpolate small sizes.
"""
import argparse, csv, json, math, os, re, sys


def read_csv(path):
    meta, rows = {}, []
    with open(path) as f:
        lines = f.read().splitlines()
    body = []
    for ln in lines:
        if ln.startswith("# "):
            kv = ln[2:].split(",", 1)
            if len(kv) == 2:
                meta[kv[0]] = kv[1]
        elif ln.strip():
            body.append(ln)
    for r in csv.DictReader(body):
        rows.append(r)
    return meta, rows


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def linfit(xs, ys):
    """least squares y = a + b x; returns (a, b)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def fit_gemm(rows, kind):
    pts = [(int(r["m"]), float(r["time_us"]) * 1e-6, float(r["value"])) for r in rows
           if r["kind"] == kind and float(r["time_us"]) > 0]
    if not pts:
        return None
    pts.sort()
    table = [{"mnk": d, "time_s": t, "rate_ops": 2.0 * d ** 3 / t} for d, t, _ in pts]
    best = max(p["rate_ops"] for p in table)
    big = [p for p in table if p["mnk"] >= 1024] or table[-3:]
    a, b = linfit([2.0 * p["mnk"] ** 3 for p in big], [p["time_s"] for p in big])
    peak_fit = 1.0 / b if b > 0 else best
    return {"peak_ops": best, "peak_ops_linfit": peak_fit, "t0_linfit_s": a, "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    meta, rows = read_csv(args.csv)
    gpu = meta.get("gpu", "unknown")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_").lower()

    def sel(kind, **kw):
        return [r for r in rows if r["kind"] == kind and all(r[k] == v for k, v in kw.items())]

    t_launch = median([float(r["time_us"]) for r in sel("launch_stream")]) * 1e-6
    t_launch_sync = median([float(r["time_us"]) for r in sel("launch_sync")])
    h2d = sorted(((float(r["bytes"]), float(r["time_us"])) for r in sel("h2d_pinned")))
    h2d_small = min(t for _, t in h2d) * 1e-6 if h2d else None
    h2d_lat, h2d_inv = linfit([b for b, _ in h2d], [t * 1e-6 for _, t in h2d]) if len(h2d) > 1 else (h2d_small, 0)

    bwc = [float(r["value"]) * 1e9 for r in sel("bw_copy")]
    bwr = [float(r["value"]) * 1e9 for r in sel("bw_read")]
    bw = max(bwc) if bwc else None

    f64 = fit_gemm(rows, "dgemm")
    i8 = fit_gemm(rows, "i8gemm")

    out = {
        "schema": "ggemm-calib-v1",
        "gpu": gpu,
        "sm_count": int(meta.get("sm_count", 0)),
        "cc": meta.get("cc", ""),
        "source_csv": os.path.basename(args.csv),
        # --- the four cost-model parameters (SI units) ---
        "t_launch_s": t_launch,
        "bw_Bps": bw,
        "f64_peak_flops": f64["peak_ops_linfit"] if f64 else None,
        "i8_peak_ops": i8["peak_ops_linfit"] if i8 else None,
        # --- extras ---
        "extra": {
            "t_launch_sync_us": t_launch_sync,
            "h2d_pinned_small_s": h2d_small,
            "h2d_pinned_latency_s": h2d_lat,
            "h2d_pinned_Bps": (1.0 / h2d_inv) if h2d_inv and h2d_inv > 0 else None,
            "bw_read_Bps": max(bwr) if bwr else None,
            "f64_peak_flops_max": f64["peak_ops"] if f64 else None,
            "i8_peak_ops_max": i8["peak_ops"] if i8 else None,
            "f64_t0_s": f64["t0_linfit_s"] if f64 else None,
            "i8_t0_s": i8["t0_linfit_s"] if i8 else None,
        },
        # per-size measured rates (square m=n=k) for interpolation at small sizes
        "f64_table": f64["table"] if f64 else [],
        "i8_table": i8["table"] if i8 else [],
        "model": {
            "emu": "3*t_launch_s + bytes_emu/bw_Bps + N*2*m*n*k/i8_peak_ops",
            "native": "t_launch_s + 2*m*n*k/f64_peak_flops",
            "bytes_emu": "8*(m*k+k*n) + N*(m+n)*Kp + 8*N*m*n + 8*m*n*(2 if beta!=0 else 1)",
            "Kp": "round_up(k,32)",
        },
    }
    # crossover sample (informational): smallest square size where emu(N) < native
    if out["f64_peak_flops"] and out["i8_peak_ops"] and bw:
        def emu(d, N):
            kp = (d + 31) // 32 * 32
            by = 8 * 2 * d * d + N * 2 * d * kp + 8 * N * d * d + 8 * d * d
            return 3 * t_launch + by / bw + N * 2.0 * d ** 3 / out["i8_peak_ops"]
        def nat(d):
            return t_launch + 2.0 * d ** 3 / out["f64_peak_flops"]
        cross = {}
        for N in range(6, 21):
            c = next((d for d in [2 ** i for i in range(4, 16)] if emu(d, N) < nat(d)), None)
            cross[str(N)] = c
        out["crossover_square_pow2_by_N"] = cross

    path = os.path.join(args.outdir, f"params_{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(path)
    print(json.dumps({k: out[k] for k in ("gpu", "sm_count", "t_launch_s", "bw_Bps",
                                           "f64_peak_flops", "i8_peak_ops")}, indent=2))


if __name__ == "__main__":
    main()
