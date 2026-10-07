#!/usr/bin/env python3
"""scripts/final/selftest.py -- check aggregate.py on the synthetic run dir (owner: agent-A).
Pure python, no GPU:   python3 -I scripts/final/selftest.py
Regenerates testdata/final_synth/, runs aggregate.py, checks numbers the generator built in.
Also runs aggregate.py on an empty directory (every input missing must not crash).
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "testdata"))
import aggregate  # noqa: E402
import gen_testdata  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("  ok   " if cond else "  FAIL ") + msg)
    fails += 0 if cond else 1


def row(md, first_cell, after=None):
    """First markdown table row whose first cell == first_cell (searching after the heading `after`)."""
    if after is not None:
        md = md[md.index(after):]
    for ln in md.splitlines():
        if ln.startswith("| "):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if cells[0] == first_cell:
                return cells
    return None


def main():
    gen_testdata.main()
    d = gen_testdata.D
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "tables.md")
        md = aggregate.main([d, "-o", out])
        check(os.path.isfile(out) and open(out).read() == md, "tables.md written")

        r = row(md, "emu_grouped", "### test_grouped")
        check(r is not None and r[1:5] == ["3", "2", "1", "1.500"], f"tests.csv emu_grouped pass/fail/worst: {r}")
        r = row(md, "auto", "### test_grouped")
        check(r is not None and r[7] == "3", f"tests.csv auto fallback sum: {r}")
        r = row(md, "test_grouped", "### Test suites")
        check(r is not None and r[1:5] == ["FAIL", "1", "20", "1"], f"suite test_grouped: {r}")
        r = row(md, "test_i8_grouped", "### Test suites")
        check(r is not None and r[4] == "0", f"suite test_i8_grouped: {r}")
        r = row(md, "emu_grouped", "### Certificate repairs: test_reuse_e2e")
        check(r is not None and r[1:4] == ["2", "2", "7"], f"reuse repairs emu_grouped: {r}")
        r = row(md, "auto", "### Certificate repairs: test_reuse_e2e")
        check(r is not None and r[2:4] == ["0", "n/a"], f"reuse repairs auto unmeasured: {r}")
        r = row(md, "per-shape x N sweep")
        check(r is not None and r[1:6] == ["2", "2", "0", "0.7000", "5"], f"numerics sweep: {r}")
        r = row(md, "grouped launch")
        check(r is not None and r[5] == "11", f"numerics grouped repairs: {r}")

        r = row(md, "uni_G64_s32", "### 2a.")
        check(r is not None and r[3:10] == ["2.000", "1.000", "4.000", "0.5000", "0.2500", "0.2000", "0.1250"],
              f"time row (phi=1.0 duplicate filtered): {r}")
        r = row(md, "moe_E8_prefill_z1.2_up", "### 2a.")
        check(r is not None and r[5] == "call_failed", f"failed cell shows status: {r}")
        r = row(md, "uni_G64_s32", "### 2b.")   # cols: native_loop emu_loop elb emu_grouped reuse auto
        check(r is not None and r[1:] == ["0.50x", "0.25x", "2.00x", "4.00x", "5.00x", "8.00x"], f"speedup vs NG: {r}")
        r = row(md, "moe_E8_prefill_z1.2_up", "### 2b.")
        check(r is not None and r[2] == "n/a", f"speedup of failed cell is n/a: {r}")
        r = row(md, "**geomean all** (2)", "### 2b.")
        check(r is not None and r[4] == "4.00x", f"geomean vs NG: {r}")
        r = row(md, "uni_G64_s32", "### 2c.")   # cols: native_loop native_grouped emu_loop emu_grouped reuse auto
        check(r is not None and r[1:] == ["0.25x", "0.50x", "0.12x", "2.00x", "2.50x", "4.00x"], f"speedup vs ELB: {r}")
        r = row(md, "uni_G64_s32", "## 3.")
        check(r is not None and r[1] == "64" and r[5] == "192" and r[9:] == ["2", "14-17"], f"launch row: {r}")

        r = row(md, "T=4096 E=16 topk=2 d_model=1024 d_ff=2048 zipf=1.0", "### MoE FFN")
        check(r is not None and r[1] == "native_loop" and r[3:5] == ["0.50x", "0.25x"], f"moe first row: {r}")
        lines = [ln for ln in md[md.index("### MoE FFN"):].splitlines() if "| auto |" in ln]
        check(bool(lines) and "| 8.00x | 4.00x |" in lines[0], f"moe auto speedups: {lines[:1]}")
        lines = [ln for ln in md[md.index("### Block-sparse"):].splitlines() if "mode=row" in ln and "| auto |" in ln]
        check(bool(lines) and "| 8.00x | 4.00x |" in lines[0], f"bsr row-mode auto speedups: {lines[:1]}")

        r = row(md, "uni_G64_s32", "### nsys")
        check(r is not None and r[1:9] == ["native_grouped", "5", "1.00", "100.0", "90.00", "10.0", "20.00", "cublas 100%"],
              f"nsys row: {r}")
        lines = [ln for ln in md[md.index("### nsys"):].splitlines() if "| emu_grouped |" in ln]
        check(bool(lines) and "i8_gemm 62.5%" in lines[0] and "| 20.0 |" in lines[0], f"nsys emu row: {lines[:1]}")
        r = row(md, "uni_G64_s32_emu_grouped", "### ncu")
        check(r is not None and r[5] == "55.5" and r[10] == "long_scoreboard 35.00%", f"ncu row: {r}")
        lines = [ln for ln in md[md.index("### Roofline"):].splitlines() if "| emu_grouped |" in ln]
        check(bool(lines) and "| 2 | 60.0 | 70.0 | compute 1, memory 1 |" in lines[0], f"roofline row: {lines[:1]}")
        check("logs/ncu_x_auto.log" in md, "failed steps listed")

        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        try:
            md2 = aggregate.main([empty])
            check("not available" in md2, "empty run dir handled")
        except Exception as e:  # noqa: BLE001
            check(False, f"empty run dir crashed: {e!r}")
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
