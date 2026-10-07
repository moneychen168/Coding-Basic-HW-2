#!/usr/bin/env python3
"""scripts/final/testdata/gen_testdata.py -- synthetic run directory for aggregate.py (owner: agent-A).
Writes testdata/final_synth/ with the same file names / CSV headers run_final.slurm produces.
Numbers are made up (round values so selftest.py can check them); they are NOT measurements.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "final_synth")
PATHS = ["native_loop", "native_grouped", "emu_loop", "emu_loop_batched", "emu_grouped", "emu_grouped_reuse", "auto"]
# per-path time (ms) for workload uni_G64_s32; moe workload = 2x these
T = {"native_loop": 2.0, "native_grouped": 1.0, "emu_loop": 4.0, "emu_loop_batched": 0.5, "emu_grouped": 0.25,
     "emu_grouped_reuse": 0.2, "auto": 0.125}
L = {"native_loop": 64, "native_grouped": 1, "emu_loop": 64 * 30, "emu_loop_batched": 64 * 3, "emu_grouped": 3,
     "emu_grouped_reuse": 3, "auto": 4}
EMU = {"emu_loop", "emu_loop_batched", "emu_grouped", "emu_grouped_reuse", "auto"}


def w(name, header, rows, delim=","):
    with open(os.path.join(D, name), "w", newline="") as f:
        wr = csv.writer(f, delimiter=delim)
        wr.writerow(header)
        wr.writerows(rows)


def main():
    os.makedirs(os.path.join(D, "logs"), exist_ok=True)
    os.makedirs(os.path.join(D, "analysis"), exist_ok=True)
    with open(os.path.join(D, "provenance.txt"), "w") as f:
        f.write("date: 2026-10-10T00:00:00+08:00\nhost: synth-node\njob: 999\ngit: deadbeef dirty=0\n"
                "Build cuda_13.0\nNVIDIA H200, 570.00, 143771 MiB, 1980 MHz, 700.00 W\n")
    w("paths.tsv", [], [], "\t")
    with open(os.path.join(D, "paths.tsv"), "w") as f:
        f.writelines(f"{p}\tincluded\t\n" for p in PATHS)
    w("steps.tsv", ["step", "status", "rc", "seconds", "log"], [
        ["cuda_preflight", "PASS", 0, 0, "logs/cuda_preflight.log"],
        ["build", "PASS", 0, 120, "logs/build.log"],
        ["test_grouped", "FAIL", 1, 30, "logs/test_grouped.log"],
        ["test_graph", "PASS", 0, 10, "logs/test_graph.log"],
        ["test_numerics_standalone", "PASS", 0, 5, "logs/test_numerics_standalone.log"],
        ["test_i8_grouped", "PASS", 0, 40, "logs/test_i8_grouped.log"],
        ["test_reuse_e2e", "PASS", 0, 3, "logs/test_reuse_e2e.log"],
        ["bench_grouped", "PASS", 0, 900, "logs/bench_grouped.log"],
        ["ncu_x_auto", "FAIL(timeout 1800s)", 124, 1800, "logs/ncu_x_auto.log"]], "\t")

    # tests.csv: 3 cases per path; emu_grouped has one FAIL with err/gamma 1.5
    hdr = ["case", "path", "G", "err_cw", "err_over_gamma", "err_vs_native", "rel_max", "nonfinite", "num_fallback",
           "launches", "moduli_min", "moduli_max", "status"]
    rows = []
    for p in PATHS:
        for i, g in enumerate([0.1, 0.2, 0.3]):
            bad = p == "emu_grouped" and i == 2
            rows.append([f"case{i}", p, 8, 1e-15, 1.5 if bad else g, 2.0 if bad else 1.0, 1e-16, 0,
                         1 if p == "auto" else 0, L[p], 14 if p in EMU else 0, 16 if p in EMU else 0,
                         "FAIL" if bad else "ok"])
    w("tests.csv", hdr, rows)

    hdr = ["workload", "path", "G", "total_flops", "time_ms", "effective_GFLOPS", "launches", "err_max", "num_fallback",
           "family", "phi", "num_moduli_cfg", "time_ms_min", "wall_ms", "err_over_gamma", "err_vs_native", "rel_max",
           "moduli_min", "moduli_max", "reps", "gpu", "status"]
    rows = []
    for wl, fam, G, k in [("uni_G64_s32", "uniform", 64, 1.0), ("moe_E8_prefill_z1.2_up", "moe", 8, 2.0)]:
        for p in PATHS:
            t = T[p] * k
            st = "call_failed" if (wl.startswith("moe") and p == "emu_loop") else "ok"
            rows.append([wl, p, G, 1e9, t, 1e9 / (t * 1e-3) / 1e9, L[p], 1e-15, 2 if p == "auto" else 0, fam, 0.5, 0,
                         t, t * 1.1, 0.25 if p in EMU else 0.5, 1.0, 1e-16, 14 if p in EMU else 0,
                         17 if p in EMU else 0, 20, "NVIDIA H200", st])
    rows.append(["uni_G64_s32", "native_grouped", 64, 1e9, 99.0, 1, 1, 1e-15, 0, "uniform", 1.0, 0, 99, 99, 0.5,
                 1.0, 1e-16, 0, 0, 20, "NVIDIA H200", "ok"])   # other phi: must be filtered out
    w("bench.csv", hdr, rows)

    hdr = ("app,path,T,E,topk,d_model,d_ff,zipf,phi,act,seed,num_moduli_cfg,experts_active,load_max,load_min,"
           "gemm_flops,total_ms_median,total_ms_mean,wall_ms_median,route_ms,gather_ms,up_ms,act_ms,down_ms,scatter_ms,"
           "route_pct,gather_pct,up_pct,act_pct,down_pct,scatter_pct,gemm_pct,eff_GFLOPS,"
           "launches_up,launches_down,fallback_up,fallback_down,moduli_min,moduli_max,"
           "err_up_cw,err_up_over_bound,err_y_cw,err_y_over_bound,rel_fro_y,nonfinite,reps,gpu,status").split(",")
    rows = []
    for p in PATHS:
        r = {h: 0 for h in hdr}
        r.update(app="moe_ffn_fp64", path=p, T=4096, E=16, topk=2, d_model=1024, d_ff=2048, zipf=1.0, phi=0.5,
                 act="tanh", total_ms_median=10 * T[p], gemm_pct=80.0, launches_up=L[p], launches_down=L[p],
                 err_y_over_bound=0.01, reps=10, gpu="NVIDIA H200", status="ok")
        rows.append([r[h] for h in hdr])
    w("moe_ffn_fp64.csv", hdr, rows)

    hdr = ("app,mode,path,atoms,basis,N,nrhs,blocks,G,nnz,max_K,cutoff,density,phi,seed,num_moduli_cfg,flops,"
           "total_ms_median,total_ms_mean,wall_ms_median,pre_ms,gemm_ms,post_ms,pre_pct,gemm_pct,post_pct,"
           "eff_GFLOPS,launches,num_fallback,moduli_min,moduli_max,err_cw,err_over_bound,rel_fro,nonfinite,"
           "reps,gpu,status").split(",")
    rows = []
    for mode in ["block", "row"]:
        for p in ["native_grouped", "emu_loop_batched", "auto"]:
            r = {h: 0 for h in hdr}
            r.update(app="block_sparse_fp64", mode=mode, path=p, atoms=1024, basis="dz", nrhs=128, G=500,
                     total_ms_median=3 * T[p], gemm_ms=2 * T[p], launches=L[p], err_over_bound=0.02, reps=5,
                     gpu="NVIDIA H200", status="ok")
            rows.append([r[h] for h in hdr])
    w("block_sparse_fp64.csv", hdr, rows)

    with open(os.path.join(D, "logs", "test_grouped.log"), "w") as f:
        f.write("device: synth\n  FAIL x\n\n==== 20 passed, 1 failed ====\n")
    with open(os.path.join(D, "logs", "test_graph.log"), "w") as f:
        f.write("\ntest_graph: 12 passed, 0 failed\n")
    with open(os.path.join(D, "logs", "test_i8_grouped.log"), "w") as f:
        f.write("device: synth\nALL PASS\n")
    with open(os.path.join(D, "logs", "test_reuse_e2e.log"), "w") as f:
        for tag, rep in [("EMU_LOOP_BATCHED", 3), ("EMU_GROUPED", 3), ("EMU_GROUPED_REUSE", 3), ("AUTO", -1),
                         ("EMU_GROUPED", 4)]:
            f.write(f"  {tag:<18} stats: probs=16 fallback=0 launches=3 moduli=[14,14] reuse=0 uA=16 uB=16 "
                    f"promoted=0 repaired={rep}\n")
    with open(os.path.join(D, "logs", "test_numerics_standalone.log"), "w") as f:
        f.write("\nshape m=64 n=64 k=256 phi=0.5   (naive FP64 dot rel_cw=1.0e-16)\n"
                " N   rel_cw      rel_abs     rel_nrm     bound       gate        rep     ok\n"
                "14  1.000e-16  1.000e-17  1.000e-17  1.000e-15  2.000e-01  5       ok\n"
                "16  1.000e-16  1.000e-17  1.000e-17  1.000e-15  7.000e-01  0       ok\n"
                "\ngrouped launch (4 problems, 1 prep + 1 recon launch), total repaired entries=11\n"
                "\nPASSED (0 failures)\n")

    keys = ["path", "calls", "kernels", "memops", "span_ns", "busy_ns", "gap_ns", "host_api_ns", "host_ns"] + \
           [s + "_ns" for s in ["prep", "i8_gemm", "reconstruct", "cublas", "memops", "other"]] + \
           [s + "_pct" for s in ["prep", "i8_gemm", "reconstruct", "cublas", "memops", "other"]]
    rows = [["native_grouped", 5, 1, 1, 100000, 90000, 10000, 5000, 20000, 0, 0, 0, 90000, 0, 0, 0, 0, 0, 100, 0, 0],
            ["emu_grouped", 5, 3, 1, 50000, 40000, 10000, 3000, 8000, 10000, 25000, 5000, 0, 0, 0, 25, 62.5, 12.5, 0, 0, 0]]
    w("analysis/nsys_summary_uni_G64_s32.csv", keys, rows)
    with open(os.path.join(D, "analysis", "nsys_summary_uni_G64_s32.md"), "w") as f:
        f.write("# nsys summary: uni_G64_s32\n")
    keys = ["stage", "kernel", "launches", "cc", "duration_us", "imma_pct", "imma_metric", "dram_pct", "dram_GBps",
            "sm_pct", "occ_pct", "occ_theo_pct", "regs", "smem_static_B", "smem_dyn_B", "stall_src",
            "stall1", "stall1_pct", "stall2", "stall2_pct", "stall3", "stall3_pct"]
    w("analysis/ncu_summary_uni_G64_s32_emu_grouped.csv", keys, [
        ["i8_gemm", "i8_grouped_gemm_kernel<128,128>", 1, "9.0", 25.0, 55.5, "sm__pipe", 40.0, 1000, 60.0, 30.0, 50.0,
         128, 0, 49152, "metrics", "long_scoreboard", "35.00", "wait", "20.00", "", ""]])
    keys = ["workload", "path", "phi", "N", "flops", "bytes", "ai", "gflops", "roof_gflops", "pct_roof", "bound"]
    w("analysis/roofline_final.csv", keys, [
        ["uni_G64_s32", "emu_grouped", 0.5, 16, 1e9, 1e8, 10, 4000, 8000, 50.0, "memory"],
        ["moe_E8_prefill_z1.2_up", "emu_grouped", 0.5, 16, 1e9, 1e7, 100, 7000, 10000, 70.0, "compute"],
        ["uni_G64_s32", "native_grouped", 0.5, "", 1e9, 1e8, 10, 1000, 2000, 50.0, "compute"]])
    print("wrote", D)


if __name__ == "__main__":
    main()
