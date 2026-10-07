#!/usr/bin/env python3
"""Generate the synthetic CSVs in this directory (owner: agent-F). Pure python, no GPU, milliseconds.

The numbers are invented and only exercise the parsers; they are NOT measurements.
Writes: synth_*_cuda_gpu_trace.csv / _cuda_api_trace.csv / _cuda_api_sum.csv (fallback mode),
        synthnvtx_* (same + _nvtx_pushpop_trace.csv), ncu_synth_details.csv, ncu_synth_raw.csv,
        ncu_synth120_raw.csv, bench_synth.csv, shapes_synth.csv.  Returns the expected values.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TID = 4242
KERN = {"gemm": "sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x64x16_cublas", "prep": "oz2_prep_kernel(const oz2_problem *, int)",
        "i8": "void i8_grouped_persistent_kernel<128, 128>(const i8_problem *, const sched_item *, int)",
        "rec": "oz2_reconstruct_kernel(const oz2_problem *)", "fill": "k_fill_ozaki(double *, unsigned long, double, unsigned long)",
        "h2d": "[CUDA memcpy Host-to-Device]"}
DUR = {"gemm": 10000, "prep": 5000, "i8": 20000, "rec": 8000, "fill": 3000, "h2d": 2000}
CALLS = {"native_loop": ["gemm", "gemm", "gemm"], "emu_grouped": ["h2d", "prep", "i8", "rec"]}


class Sim:
    def __init__(self):
        self.t = 1_000_000          # host clock (ns)
        self.gpu_free = 0           # GPU stream busy until
        self.corr = 1
        self.api, self.gpu, self.nvtx = [], [], []

    def call_api(self, name, dur=1500, op=None):
        start = self.t
        corr = self.corr
        self.corr += 1
        self.api.append([start, dur, name, 0, corr, 77, TID, 0, "bench_grouped"])
        self.t += dur + 200
        if op:
            gs = max(start + dur + 3000, self.gpu_free)   # queue latency 3 us after launch returns
            self.gpu.append([gs, DUR[op], corr, KERN[op]])
            self.gpu_free = gs + DUR[op]
        return start

    def lib_call(self, path):
        if path == "emu_grouped":
            self.call_api("cudaEventSynchronize", 800)                  # acquire_slot
        for op in CALLS[path]:
            # the i8 launch gets a long host API (12 us) so the GPU sees a real launch gap before it
            dur = {"h2d": 2500, "i8": 12000}.get(op, 1500)
            self.call_api("cudaMemcpyAsync" if op == "h2d" else "cudaLaunchKernel", dur, op)
        if path == "emu_grouped":
            self.call_api("cudaEventRecord", 600)                       # release_slot

    def drain(self):
        self.t = max(self.t, self.gpu_free + 500)

    def nvtx_range(self, name, s, e, lvl):
        self.nvtx.append([s, e, e - s, 0, e - s, name, 77, TID, lvl, 0, len(self.nvtx) + 1, "", "", name])


def simulate(with_nvtx):
    s = Sim()
    s.call_api("cudaMalloc", 50000)
    s.call_api("cudaLaunchKernel", 1500, "fill")       # bench data fill: must be ignored
    s.call_api("cudaStreamSynchronize", 4000); s.drain()
    for path in ("native_loop", "emu_grouped"):
        for _ in range(3):
            s.call_api("cudaMalloc", 30000)            # ggemm_create
        s.call_api("cudaStreamSynchronize", 2000); s.drain()
        t0 = s.t; s.lib_call(path)                     # warm-up (untimed)
        if with_nvtx: s.nvtx_range(f"ggemm:{path}", t0, s.t, 0)
        s.drain(); s.call_api("cudaStreamSynchronize", 3000)
        for _ in range(2):                             # timed reps
            tb = s.t
            s.call_api("cudaEventRecord", 700)
            t0 = s.t; s.lib_call(path)
            if with_nvtx: s.nvtx_range(f"ggemm:{path}", t0, s.t, 1)
            s.call_api("cudaEventRecord", 700)
            s.drain(); s.call_api("cudaEventSynchronize", 2000)
            if with_nvtx: s.nvtx_range("bench:timed", tb, s.t, 0)
        for _ in range(3):
            s.call_api("cudaFree", 20000)              # ggemm_destroy
    return s


def write_nsys(prefix, s):
    with open(prefix + "_cuda_gpu_trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "CorrId", "GrdX", "GrdY", "GrdZ", "BlkX", "BlkY", "BlkZ", "Reg/Trd",
                    "StcSMem (MB)", "DymSMem (MB)", "Bytes (MB)", "Throughput (MBps)", "SrcMemKd", "DstMemKd", "Device",
                    "Ctx", "GreenCtx", "Strm", "Name"])
        for (st, d, c, n) in sorted(s.gpu):
            w.writerow([st, d, c, 132, 1, 1, 256, 1, 1, 64, 0, 0, "", "", "", "", "NVIDIA H200 (0)", 1, "", 13, n])
    with open(prefix + "_cuda_api_trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "Name", "Result", "CorrID", "Pid", "Tid", "T-Pri", "Thread Name"])
        w.writerows(s.api)
    agg = {}
    for r in s.api:
        agg.setdefault(r[2], []).append(r[1])
    tot = sum(sum(v) for v in agg.values())
    with open(prefix + "_cuda_api_sum.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time (%)", "Total Time (ns)", "Num Calls", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)",
                    "StdDev (ns)", "Name"])
        for n, v in sorted(agg.items(), key=lambda x: -sum(x[1])):
            w.writerow([round(100 * sum(v) / tot, 1), sum(v), len(v), sum(v) / len(v), sorted(v)[len(v) // 2],
                        min(v), max(v), 0, n])
    if s.nvtx:
        with open(prefix + "_nvtx_pushpop_trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Start (ns)", "End (ns)", "Duration (ns)", "DurChild (ns)", "DurNonChild (ns)", "Name", "PID",
                        "TID", "Lvl", "NumChild", "RangeId", "ParentId", "RangeStack", "NameTree"])
            w.writerows(s.nvtx)


NCU_HDR = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
           "Grid Size", "Device", "CC"]
NCU_K = [("oz2_prep_kernel", 6.1, 0.4, 71.0, 35.0, 32, 0, 0),
         ("i8_grouped_persistent_kernel", 41.0, 78.5, 22.0, 24.5, 128, 0, 65536),
         ("oz2_reconstruct_kernel", 12.3, 1.2, 64.0, 48.0, 40, 0, 0),
         ("sm90_xmma_gemm_f64f64_f64f64_f64_nn_n", 30.0, 0.0, 40.0, 12.0, 168, 0, 98304)]
# name, duration us, imma %, dram %, occ %, regs, static smem, dyn smem
STALLS = {"oz2_prep_kernel": {"long_scoreboard": 6.0, "wait": 1.0, "barrier": 0.5, "selected": 1.0},
          "i8_grouped_persistent_kernel": {"short_scoreboard": 1.5, "math_pipe_throttle": 2.5, "long_scoreboard": 3.0,
                                           "barrier": 0.2, "selected": 1.0},
          "oz2_reconstruct_kernel": {"long_scoreboard": 9.0, "lg_throttle": 2.0, "wait": 1.5, "selected": 1.0},
          "sm90_xmma_gemm_f64f64_f64f64_f64_nn_n": {"wait": 1.0, "selected": 1.0}}


def write_ncu():
    with open(os.path.join(HERE, "ncu_synth_details.csv"), "w", newline="") as f:
        f.write('==PROF== Disconnected from process 77\n')
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(NCU_HDR + ["Section Name", "Metric Name", "Metric Unit", "Metric Value"])
        for call in range(2):
            for j, (n, dur, imma, dram, occ, regs, ss, ds) in enumerate(NCU_K):
                kid = call * len(NCU_K) + j
                base = [kid, 77, "bench_grouped", "127.0.0.1", n, 1, 13, "(256, 1, 1)", "(132, 1, 1)", 0, "9.0"]
                for sec, mn, unit, v in [("GPU Speed Of Light Throughput", "Duration", "usecond", dur + call * 0.2),
                                         ("GPU Speed Of Light Throughput", "DRAM Throughput", "%", dram),
                                         ("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", 50.0),
                                         ("Launch Statistics", "Registers Per Thread", "register/thread", regs),
                                         ("Launch Statistics", "Static Shared Memory Per Block", "byte", ss),
                                         ("Launch Statistics", "Dynamic Shared Memory Per Block", "Kbyte", ds / 1000),
                                         ("Occupancy", "Achieved Occupancy", "%", occ),
                                         ("Occupancy", "Theoretical Occupancy", "%", 50.0)]:
                    w.writerow(base + [sec, mn, unit, f"{v:,.2f}"])
    for fname, cc, imma_name in [("ncu_synth_raw.csv", "9.0", "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"),
                                 ("ncu_synth120_raw.csv", "12.0", "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_active")]:
        reasons = sorted({r for d in STALLS.values() for r in d})
        cols = [imma_name, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "gpu__time_duration.sum"] + \
               [f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio" for r in reasons]
        if cc == "12.0":   # decoy: the sm_90 name must NOT be picked on CC 12.0
            cols.append("sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active")
        with open(os.path.join(HERE, fname), "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(NCU_HDR + cols)
            w.writerow([""] * len(NCU_HDR) + ["%", "%", "nsecond"] + ["inst"] * len(reasons) + (["%"] if cc == "12.0" else []))
            for call in range(2):
                for j, (n, dur, imma, *_r) in enumerate(NCU_K):
                    kid = call * len(NCU_K) + j
                    row = [kid, 77, "bench_grouped", "127.0.0.1", n, 1, 13, "(256, 1, 1)", "(132, 1, 1)", 0, cc]
                    row += [imma, imma + 1.0, f"{dur * 1000:,.0f}"] + [STALLS[n].get(r, 0.0) for r in reasons]
                    if cc == "12.0":
                        row.append(99.0)
                    w.writerow(row)


def write_bench():
    hdr = ("workload,path,G,total_flops,time_ms,effective_GFLOPS,launches,err_max,num_fallback,family,phi,"
           "num_moduli_cfg,time_ms_min,wall_ms,err_over_gamma,err_vs_native,rel_max,moduli_min,moduli_max,reps,gpu,status")
    rows = []
    def add(w, fam, flops, path, t_ms, nmin, nmax, status="ok"):
        rows.append([w, path, 64, f"{flops:.6e}", t_ms, f"{flops / (t_ms * 1e-3) / 1e9:.3f}", 3, 1e-15, 0, fam, 0.5, 0,
                     t_ms, t_ms * 1.1, 0.5, 1.0, 1e-16, nmin, nmax, 5, "NVIDIA H200", status])
    f_uni = 64 * 2 * 1024 ** 3
    for p, t, n in [("native_loop", 0.30, 0), ("native_grouped", 0.25, 0), ("emu_loop", 0.60, 14),
                    ("emu_grouped", 0.12, 14), ("auto", 0.11, 12)]:
        add("uni_G64_s1024", "uniform", f_uni, p, t, n, n)
    f_moe = 2 * (100 + 300 + 50) * 2048 * 1024
    for p, t, n in [("native_loop", 0.05, 0), ("emu_grouped", 0.03, 15), ("auto", 0.03, 15)]:
        add("moe_E3_synth", "moe", f_moe, p, t, n, n)
    add("het_G4_noshape", "hetero", 1e9, "native_loop", 0.1, 0, 0)
    add("uni_G64_s1024", "uniform", f_uni, "emu_grouped", 0.12, 14, 14, status="call_failed")
    with open(os.path.join(HERE, "bench_synth.csv"), "w", newline="") as f:
        f.write(hdr + "\n")
        csv.writer(f).writerows(rows)
    with open(os.path.join(HERE, "shapes_synth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "m", "n", "k", "count"])
        for m in (100, 300, 50):
            w.writerow(["moe_E3_synth", m, 2048, 1024, 1])


def main():
    write_nsys(os.path.join(HERE, "synth"), simulate(False))
    write_nsys(os.path.join(HERE, "synthnvtx"), simulate(True))
    write_ncu()
    write_bench()


if __name__ == "__main__":
    main()
