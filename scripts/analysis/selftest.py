#!/usr/bin/env python3
"""scripts/analysis/selftest.py -- parser self-test on the synthetic CSVs in testdata/ (owner: agent-F).

Pure python, no GPU, < 1 s:   python3 -I scripts/analysis/selftest.py [-o out_dir]
Regenerates testdata/ first, runs nsys_report / ncu_report / roofline on it and checks the numbers
the generator built in. Figures are produced only if matplotlib is importable, e.g.
    uv run --no-project --with matplotlib python scripts/analysis/selftest.py
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "testdata"))
import gen_testdata  # noqa: E402
import ncu_report  # noqa: E402
import nsys_report  # noqa: E402
import roofline  # noqa: E402

TD = os.path.join(HERE, "testdata")
REPO = os.path.dirname(os.path.dirname(HERE))


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def test_nsys(out, prefix, mode, extra=()):
    print(f"[nsys_report] {mode}")
    rows = {r["path"]: r for r in nsys_report.main([os.path.join(TD, prefix), "-o", out, *extra])}
    D = gen_testdata.DUR
    nl, eg = rows.get("native_loop"), rows.get("emu_grouped")
    check(nl is not None and eg is not None and len(rows) == 2, f"paths found: {sorted(rows)}")
    if not (nl and eg):
        return
    check(nl["calls"] == 2 and eg["calls"] == 2, "2 timed calls per path (warm-up excluded)")
    check(nl["kernels"] == 3 and nl["memops"] == 0, "native_loop: 3 kernels, no memops")
    check(eg["kernels"] == 3 and eg["memops"] == 1, "emu_grouped: 3 kernels + 1 H2D")
    check(close(nl["busy_ns"], 3 * D["gemm"]) and close(nl["gap_ns"], 0), "native_loop busy 30 us, no gap")
    busy = D["h2d"] + D["prep"] + D["i8"] + D["rec"]
    check(close(eg["busy_ns"], busy), f"emu_grouped busy {busy / 1e3} us")
    check(eg["gap_ns"] > 0 and close(eg["span_ns"], eg["busy_ns"] + eg["gap_ns"]), "emu_grouped has a launch gap, span = busy + gap")
    check(close(eg["i8_gemm_pct"], 100.0 * D["i8"] / busy) and close(nl["cublas_pct"], 100.0), "stage shares")
    check(eg["host_api_ns"] > 0 and eg["host_ns"] >= eg["host_api_ns"], "host API time inside host window")


def test_ncu(out):
    print("[ncu_report] details + raw (sm_90)")
    rows = ncu_report.main([os.path.join(TD, "ncu_synth_details.csv"), os.path.join(TD, "ncu_synth_raw.csv"),
                            "-o", out, "--label", "synth"])
    by = {r["stage"]: r for r in rows}
    check(set(by) == {"i8_gemm", "prep", "reconstruct"}, f"focus kernels only: {sorted(by)}")
    i8 = by["i8_gemm"]
    check(i8["launches"] == 2, "two launches merged by name")
    check(close(i8["imma_pct"], 78.5) and "op_imma" in i8["imma_metric"], "sm_90 picks sm__pipe_tensor_op_imma")
    check(close(i8["duration_us"], 41.0), "duration from raw ns -> us (details usecond agrees)")
    check(close(i8["regs"], 128) and close(i8["smem_dyn_B"], 65536, 1e-3), "regs / dynamic smem (Kbyte -> B)")
    check(close(i8["occ_pct"], 24.5) and close(i8["dram_pct"], 22.0), "occupancy / DRAM %")
    check([s for s, _ in i8["stalls_top"]] == ["long_scoreboard", "math_pipe_throttle", "short_scoreboard"],
          f"top-3 stalls, 'selected' excluded: {i8['stalls_top']}")
    check(close(sum(p for _, p in i8["stalls_top"]), 100 * 7.0 / 7.2), "stall shares")
    print("[ncu_report] raw only (sm_120)")
    rows = ncu_report.main([os.path.join(TD, "ncu_synth120_raw.csv"), "-o", out, "--label", "synth120"])
    i8 = {r["stage"]: r for r in rows}["i8_gemm"]
    check("subpipe_imma" in i8["imma_metric"] and close(i8["imma_pct"], 78.5), "sm_120 picks sm__pipe_tensor_subpipe_imma (decoy ignored)")


def test_roofline(out):
    print("[roofline]")
    calib = os.path.join(REPO, "bench", "calib", "params_nvidia_h200.json")
    pts = roofline.main([os.path.join(TD, "bench_synth.csv"), "--calib", calib, "--shapes", os.path.join(TD, "shapes_synth.csv"),
                         "-o", out, "--label", "synth"])
    by = {(p["workload"], p["path"]): p for p in pts}
    check(len(pts) == 8, f"8 points (failed row and shape-less workload skipped): {len(pts)}")
    p = by[("uni_G64_s1024", "native_loop")]
    s = 1024
    check(close(p["ai"], 2 * s ** 3 / (8 * 3 * s * s)), "native AI = 2s^3 / 24 s^2")
    p = by[("uni_G64_s1024", "emu_grouped")]
    N, kp = 14, 1024
    by_exp = 8 * 2 * s * s + N * 2 * s * kp + 8 * N * s * s + 8 * s * s
    check(close(p["ai"], 2 * s ** 3 / by_exp), "emu AI from calib bytes_emu model, N=14")
    check(by[("moe_E3_synth", "auto")]["N"] == 15, "auto uses moduli_max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="ggemm_analysis_selftest_")
    gen_testdata.main()
    test_nsys(out, "synth", "fallback (no NVTX)", ["--paths", "native_loop,emu_grouped"])
    test_nsys(out, "synthnvtx", "NVTX ranges")
    test_ncu(out)
    test_roofline(out)
    print(f"outputs in {out}")
    print("SELFTEST", "PASSED" if check.failed == 0 else f"FAILED ({check.failed})")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
