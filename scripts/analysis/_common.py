"""scripts/analysis/_common.py -- shared helpers for the profiling-analysis scripts (owner: agent-F).

Pure stdlib (csv/re/statistics). matplotlib is imported lazily by the plotting functions only,
so every parser runs on a bare python3 (login node, no GPU, no third-party packages).
"""
import csv
import os
import re
import statistics

# ---------------------------------------------------------------- kernel / stage classification
# Stage of one GPU op, decided from the kernel name (NVTX can override, see nsys_report.py).
STAGE_ORDER = ["prep", "i8_gemm", "reconstruct", "cublas", "memops", "other"]
STAGE_LABEL = {"prep": "prep (scale+residue)", "i8_gemm": "INT8 grouped GEMM", "reconstruct": "reconstruct (CRT)",
               "cublas": "cuBLAS FP64", "memops": "memcpy/memset", "other": "other kernels"}
# Fixed categorical slots (same validated palette as scripts/plot.py; color follows the stage).
STAGE_COLOR = {"prep": "#2a78d6", "i8_gemm": "#eb6834", "reconstruct": "#1baf7a", "cublas": "#eda100",
               "memops": "#e87ba4", "other": "#8a8983"}
GAP_COLOR = "#d6d5ce"
PATH_ORDER = ["native_loop", "native_grouped", "emu_loop", "emu_grouped", "auto",
              "emu_grouped_reuse", "emu_loop_batched"]   # == bench `--paths all`
PATH_EXTRA = ["graph"]   # newer paths: sorted after, drawn in neutral gray
PATH_COLOR = {"native_loop": "#2a78d6", "native_grouped": "#eb6834", "emu_loop": "#1baf7a",
              "emu_grouped": "#eda100", "auto": "#e87ba4",
              "emu_grouped_reuse": "#008300",   # slot 6 (green)
              "emu_loop_batched": "#4a3aa7"}    # slot 7 (violet)
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e4e3dc"

_RE_PREP = re.compile(r"oz2_prep", re.I)
_RE_I8 = re.compile(r"i8_grouped|int8_grouped|i8_.*gemm", re.I)
_RE_REC = re.compile(r"oz2_reconstruct|reconstruct", re.I)
_RE_CUBLAS = re.compile(r"gemm|cublas|cutlass|xmma|gemv|sm\d+_|ampere_|hopper_|blackwell_|nvjet", re.I)
_RE_MEM = re.compile(r"^\[CUDA (memcpy|memset)", re.I)
_RE_BENCH = re.compile(r"k_fill|fill_ozaki|fill_const|ref_|dd_gemm|compare", re.I)


def classify_stage(name):
    """Stage of one GPU op from its (demangled) name."""
    if _RE_MEM.search(name):
        return "memops"
    if _RE_PREP.search(name):
        return "prep"
    if _RE_I8.search(name):
        return "i8_gemm"
    if _RE_REC.search(name):
        return "reconstruct"
    if _RE_CUBLAS.search(name):
        return "cublas"
    return "other"


def is_bench_kernel(name):
    """Kernels launched by bench/tests helpers (data fill, reference), never by the library."""
    return bool(_RE_BENCH.search(name))


# ---------------------------------------------------------------- CSV with unit-annotated headers
_UNIT_NS = {"ns": 1.0, "nsec": 1.0, "nsecond": 1.0, "us": 1e3, "usec": 1e3, "usecond": 1e3, "μs": 1e3,
            "µs": 1e3, "ms": 1e6, "msec": 1e6, "msecond": 1e6, "s": 1e9, "sec": 1e9, "second": 1e9}


def _split_header(h):
    """'Start (ns)' -> ('start', 'ns'); 'Start:ts_ns' -> ('start', 'ns'); 'Name' -> ('name', None)."""
    h = h.strip().strip('"')
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", h)
    if m:
        return m.group(1).strip().lower(), m.group(2).strip()
    m = re.match(r"^(.*?):(?:ts|dur)_(\w+)$", h)
    if m:
        return m.group(1).strip().lower(), m.group(2)
    m = re.match(r"^(.*?):\w+$", h)
    if m:
        return m.group(1).strip().lower(), None
    return h.lower(), None


def to_float(s):
    if s is None:
        return float("nan")
    s = str(s).strip().strip('"').replace(",", "")
    if s in ("", "n/a", "N/A", "nan", "None"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_csv_skip_preamble(path, must_have):
    """Read a CSV whose real header row contains every string in `must_have` (case-insensitive).

    nsys / ncu sometimes print banner lines ("==PROF==", "Generating ...") before the table.
    Returns (header list, list of row lists)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    for i, r in enumerate(rows):
        low = [c.strip().lower() for c in r]
        if all(any(c.startswith(m.lower()) for c in low) for m in must_have):
            return r, [x for x in rows[i + 1:] if x and any(c.strip() for c in x)]
    raise ValueError(f"{path}: no header row containing {must_have}")


class Table:
    """Column access by base name with automatic time-unit conversion to ns."""

    def __init__(self, header, rows):
        self.header = header
        self.rows = rows
        self._idx = {}
        for i, h in enumerate(header):
            base, unit = _split_header(h)
            self._idx.setdefault(base, (i, unit))

    def has(self, *names):
        return any(n.lower() in self._idx for n in names)

    def col(self, *names):
        for n in names:
            if n.lower() in self._idx:
                return self._idx[n.lower()]
        raise KeyError(f"none of {names} in header {self.header}")

    def get_str(self, row, *names):
        i, _ = self.col(*names)
        return row[i] if i < len(row) else ""

    def get_ns(self, row, *names):
        i, unit = self.col(*names)
        v = to_float(row[i] if i < len(row) else "")
        return v * _UNIT_NS.get((unit or "ns").lower(), 1.0)

    def get_num(self, row, *names):
        i, _ = self.col(*names)
        return to_float(row[i] if i < len(row) else "")


def median(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def union_length(intervals):
    """Total length covered by [(start, end), ...] (GPU busy time; robust to overlap/multi-stream)."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def md_table(header, rows, align=None):
    """GitHub markdown table. `align`: string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---" if a == "l" else "---:" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def fmt(x, nd=1):
    if x is None or x != x:
        return "n/a"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)
    return d


def get_pyplot():
    """matplotlib.pyplot with the Agg backend, or None (caller then skips figures)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def style_axes(ax, xlabel=None, ylabel=None, grid_axis="x"):
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
