# ggemm: Grouped FP64 GEMM emulated on INT8 Tensor Cores (Ozaki scheme II)

ggemm computes, for a group of G independent problems with different shapes,

    C_g = alpha_g * A_g (m_g x k_g) * B_g (k_g x n_g) + beta_g * C_g

in FP64 (column-major, no transpose in v1). The emulation path uses INT8 Tensor
Cores with the Ozaki scheme II (CRT) and fuses the whole group into a fixed number
of kernel launches. Every output entry is certified against a componentwise FP64 error
bound; entries that fail the certificate (or touch NaN/Inf) are recomputed in FP64
inside the reconstruct kernel. Implementation is original code; GEMMul8 and the papers
were only read for the algorithm (see docs/report.md, AI / external code section).

Platform: development and all measurements run on nano4 (Taiwan National Center for
High-performance Computing, NCHC), whose GPUs are **H200 (sm_90)**. This is not an RTX 5090
machine. The build still compiles for sm_90 and sm_120 so the same code can target
consumer Blackwell (weak FP64), but sm_120 is not what the SLURM jobs run on.

## Setup

Requirements: CUDA 13.0 toolkit, CMake >= 3.24, a C++17 compiler, cuBLAS / cuBLASLt.

```bash
# login node: compile only, never run GPU programs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     # defaults: CUDA 13.0 nvcc, archs 90;120
cmake --build build -j 12
# or without CMake (nvcc directly, what the SLURM scripts use):
scripts/build.sh                                   # ARCHS="90 120" BUILD_DIR=build TARGETS="..."
```

CMake builds the static library `ggemm` and one executable per `tests/test_*.cu` and
`bench/bench_*.cu`. Compile-time option: `-DGGEMM_NVTX=0` removes the NVTX ranges
(default 1, header-only nvtx3, used by nsys attribution; see Profiling).

## Platform notes (nano4)

- SLURM: account GOV114009, partition dev, `--gpus-per-node=1`, at most 12 CPUs per GPU.
- Do not run any GPU program on a login node; use the `.slurm` scripts.
- Known-bad nodes are excluded in every script with
  `#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085` (GPU passes NVML but CUDA
  init fails). Keep this list when writing new job scripts.
- `scripts/env.sh` provides `cuda_preflight` (compiles and runs `scripts/cuda_preflight.cu`:
  device count, context, trivial kernel). On failure it prints host and `nvidia-smi` and exits
  75, so a bad node shows up as FAILED instead of an empty COMPLETED job. Every job script
  calls it right after sourcing env.sh. If a new bad node appears, add it to the exclude list.
- `scripts/env.sh` also sets `CUDA_MODULE_LOADING=EAGER`, writes provenance (git hash, host,
  GPU) next to results, and `select_calib` picks the cost-model calibration (below).
- `scripts/run_nano4.sh` is an older RTX 5090 template with TODO placeholders; it is stale.
  Use the `.slurm` scripts.

## API

Header: `include/ggemm.h`. All pointers inside problem descriptors are device pointers;
the descriptor array itself is on the host.

```c
#include "ggemm.h"

ggemm_handle_t h;
// Allocate ALL workspace here, never inside timed calls.
// Upper bounds on what later calls may use:
ggemm_create(&h, /*max_problems*/ G, total_A_elems, total_B_elems, total_C_elems, stream);

ggemm_problem_t probs[G];
for (int g = 0; g < G; ++g)
  probs[g] = { m[g], n[g], k[g], dA[g], lda[g], dB[g], ldb[g], dC[g], ldc[g], alpha[g], beta[g] };

ggemm_config_t cfg = { /*num_moduli*/ 0,          // 0 = adaptive per problem
                       /*target_rel_err*/ 1e-15,
                       /*allow_fallback*/ 1 };
ggemm_stats_t st;
int rc = ggemm_grouped_dgemm(h, GGEMM_PATH_AUTO, &cfg, probs, G, &st);
if (rc != 0) { /* GGEMM_ERR_* code, see below; no hidden allocation */ }
ggemm_destroy(h);
```

Config: `num_moduli` = 0 for adaptive N per problem, otherwise fixed N in
[2, GGEMM_MAX_MODULI = 20]. `target_rel_err` is used when `num_moduli == 0`
(default 1e-15). `allow_fallback = 1` routes unsafe or unprofitable problems to native FP64
and reports the count in `stats.num_fallback`. `repair_mode` selects the certificate behavior
(see Repair mode); 0 / omitted = path default.

`ggemm_create_ex(..., min_k)` is `ggemm_create` plus the smallest k of any emulated problem the
handle will see (`ggemm_create` assumes 32). Residues are stored with `Kp = round_up(k, 32)`, so
for k < 32 the residue workspace inflates; size the handle with the right `min_k`, otherwise a
call returns `GGEMM_ERR_WORKSPACE` and prints one diagnostic line on stderr.

Problems with `alpha == 0` (planner route `GGEMM_ROUTE_ALPHA_ZERO`) are always sent to cuBLAS to
compute `beta*C`, on every emulation path. `GGEMM_PATH_NATIVE_GROUPED` merges problems of
identical shape into one group.

Stats (`ggemm_stats_t`): `num_problems`, `num_fallback`, `kernel_launches`,
`moduli_min/max`, and the operand-reuse fields `reuse_enabled`, `num_unique_A`,
`num_unique_B`, `num_promoted`, plus `num_repaired`, `num_uncertified`, `num_safety_fallback` (see Repair mode;
each is -1 unless `GGEMM_STATS_REPAIR=1`).

Error codes (internal, `GGEMM_ERR_*`, returned as nonzero ints): INVALID (bad args, null handle,
`num_problems > max_problems`), WORKSPACE (call exceeds what `ggemm_create` sized), UNSAFE
(K too large for emulation and `allow_fallback == 0`), CUDA, CUBLAS, NUMERICS (oz2_*
failed), I8GEMM (INT8 grouped GEMM failed), ALLOC (only from create).

### Plan / execute (CUDA Graph)

Header: `src/ggemm_graph.h` (needs `-I src`, not part of the public `include/` yet).
For repeated calls with identical shapes, planning, the descriptor H2D and launch overhead
are paid once:

```c
ggemm_plan_t plan = ggemm_make_plan(h, GGEMM_PATH_AUTO, &cfg, probs, G);  // or _ex(..., &err)
ggemm_execute_plan(plan, stream);                  // cudaGraphLaunch only
ggemm_plan_update_pointers(plan, A_ptrs, B_ptrs, C_ptrs);  // same shapes, new device pointers
ggemm_plan_get_stats(plan, &st);
ggemm_destroy_plan(plan);                          // BEFORE ggemm_destroy(h)
```

- `ggemm_make_plan` allocates plan-owned memory and synchronizes: call it outside timed regions.
  It captures the launch sequence into an instantiated graph.
- Only `GGEMM_PATH_EMU_GROUPED` and `GGEMM_PATH_AUTO` can be planned (other paths return NULL).
- A graph does **not** do operand reuse: prep runs once per problem even if A or B is shared.
- Shapes, leading dimensions, alpha and beta are frozen in the plan; only the A/B/C pointers
  can change (`ggemm_plan_update_pointers`, any of the three arrays may be NULL = keep).
- A plan must be destroyed before its handle. Plans and handles are not thread-safe.
- All users of one handle (plain calls, any plan, any stream) are ordered into one total
  order on the handle's stream, because residue/INT32 scratch is shared.

## The seven paths (`ggemm_path_t`, values 0 to 6)

| # | Path | What it does | Role in the ablation |
|---|---|---|---|
| 0 | `GGEMM_PATH_NATIVE_LOOP` | `cublasDgemm` once per problem | native FP64 baseline |
| 1 | `GGEMM_PATH_NATIVE_GROUPED` | `cublasDgemmGroupedBatched` | native FP64 grouped baseline |
| 2 | `GGEMM_PATH_EMU_LOOP` | single-GEMM INT8 Ozaki-II emulation called per problem; the INT8 GEMM is issued once per modulus plane (O(G*N) launches) | naive lower bound of emulation (no batching of any kind) |
| 3 | `GGEMM_PATH_EMU_GROUPED` | fused pipeline: prep -> persistent grouped INT8 GEMM (LPT tile scheduling, fixed 128x128 tile) -> reconstruct (3 kernels, one descriptor H2D; problems with k = 0, alpha = 0 go to cuBLAS as one extra launch) | ours: isolates the launch-merging / grouping gain |
| 4 | `GGEMM_PATH_AUTO` | EMU_GROUPED plus planner: per-problem moduli count N (componentwise, `GGEMM_RHO_BUDGET`), per-problem tile config (128x128 / 64x64 / 32x32), fallback to native FP64 for unsafe (k too large), small-k (k <= 16, route `SMALL_K`), tiny or unprofitable problems; repair ON. LPT tile scheduling is the same as in every grouped path. **Operand reuse is off.** | ours (default); isolates the planner gain |
| 5 | `GGEMM_PATH_EMU_GROUPED_REUSE` | EMU_GROUPED plus operand reuse: prep runs once per unique A and once per unique B in the group | ours: isolates the operand-reuse gain |
| 6 | `GGEMM_PATH_EMU_LOOP_BATCHED` | GEMMul8-style strided-batched emulation: per problem 3 launches (prep, ONE INT8 GEMM over all its planes, reconstruct) | main starter-like baseline (what a per-problem GEMMul8 call loop does) |

Reading the ablation: 2 -> 6 shows what batching planes inside one problem buys; 6 -> 3 shows
what merging across problems buys; 3 -> 4 adds the planner; 3 -> 5 adds operand reuse.
Paths 2 and 6 use the same planner decisions as the per-problem loop (they plan like EMU_LOOP).

## Environment variables

Read at `ggemm_create` (planner / handle) unless noted.

| Variable | Meaning |
|---|---|
| `GGEMM_CALIB_JSON` | path to a calibration JSON (`f64_peak_flops`, `i8_peak_ops`, `bw_Bps` scalars, written by `bench/calib/fit.py`). If unset, the planner loads `bench/calib/params_<gpu_slug>.json` (slug = non-alphanumeric runs to `_`, lower-case, e.g. `nvidia_h200`); if that is missing, nominal built-in defaults. `scripts/env.sh: select_calib` logs the choice into `<out>/calib.txt`. The chosen source is also printed once on stderr as `[ggemm] calib=...`, and the loaded values are range-checked. |
| `GGEMM_FP64_FLOPS`, `GGEMM_INT8_OPS`, `GGEMM_MEM_BW` | override the cost-model peaks (FP64 flop/s, INT8 op/s, memory bytes/s) for the planner; take precedence over the JSON. Positive numbers only. |
| `GGEMM_TINY_MNK` | m*n*k threshold under which a problem is treated as tiny by the planner (default 32^3); `0` disables the tiny route. |
| `GGEMM_RHO_BUDGET` | rho budget used only by `AUTO` with adaptive N (`num_moduli = 0`) to choose N componentwise (`oz2_choose_num_moduli_cw`). Read once per process (function-static, at the first plan), not per handle. The built-in default (16) is a placeholder and has **not been tuned**. |
| `GGEMM_REUSE_PROMOTE` | `1` = path 5 shares operands regardless of N and promotes every problem to the max N of its connected component (more sharing, more moduli for some problems; counted in `stats.num_promoted`). Default `0` = share only between problems with equal N. |
| `GGEMM_STATS_REPAIR` | `1` = fill `stats.num_repaired`, `num_uncertified`, `num_safety_fallback` (reads device counters, **synchronizes the stream**; keep off in timed regions). Otherwise they are -1. |
| `GGEMM_NVTX` | **compile-time macro**, not an env var: `-DGGEMM_NVTX=0` compiles NVTX ranges out. Default on. |

Job-script knobs (not library variables): `SKIP_BUILD`, `TEST_SET`, `BENCH_SET`, `PATHS`, `PHIS`,
`REPS`, `WARMUP`, `ONLY`, `ACC`, `SANITIZE`, `ARCHS`, `BUILD_DIR`; see the header of each script.

## Repair mode

`ggemm_config_t.repair_mode` controls the componentwise FP64 certificate in the reconstruct
kernel (values in `include/ggemm.h`):

| Value | Name | Meaning |
|---|---|---|
| 0 | `GGEMM_REPAIR_DEFAULT` | path default: `AUTO` uses ON; the other EMU paths (LOOP, LOOP_BATCHED, GROUPED, GROUPED_REUSE) use COUNT |
| 1 | `GGEMM_REPAIR_OFF` | no certificate |
| 2 | `GGEMM_REPAIR_COUNT` | certificate is evaluated and failures are counted (`num_uncertified`); the emulated value is kept |
| 3 | `GGEMM_REPAIR_ON` | entries failing the certificate are recomputed as FP64 dot products (`num_repaired`) |

In every mode, entries whose inputs contain NaN/Inf take the safety fallback (computed in FP64;
counted in `num_safety_fallback`).

Relation to the assignment rules: native FP64 is forbidden in the timed path, but a safety
fallback is required. The design therefore keeps the safety fallback unconditional (NaN/Inf), while
the certificate-driven FP64 repair is a mode: the EMU paths default to COUNT (pure emulation in the
timed region, with uncertified entries reported rather than silently hidden), and only `AUTO`
repairs by default. Use `GGEMM_STATS_REPAIR=1` to read the counters (synchronizes the stream).

## Known limitations (v1)

- No transpose: column-major, A is m x k, B is k x n, op = N only.
- Emulation needs kp = round_up(k, 32) <= 131040 (INT32 accumulation limit of the INT8 GEMM,
  `OZ2_MAX_K`); larger K falls back to native FP64 (`allow_fallback = 1`) or returns an error.
- `oz2_prep_grouped` handles at most 65535 emulated problems per call (`num <= 65535`); the
  graph path rejects more with `GGEMM_ERR_INVALID`.
- Plan/execute (CUDA Graph) does not do operand reuse; plan only for paths 3 and 4; destroy
  plans before the handle.
- Multi-GPU: the CRT constant tables are uploaded (`oz2_init_constants`) only to the device that
  is current when the handle is created; a handle must be created and used on that one device.
- Workspace is fixed at `ggemm_create`: calls exceeding `max_problems` or the total
  A/B/C element bounds return a nonzero error code; there is no allocation fallback.
- Descriptor array must be on the host; data pointers must be device pointers.
- Accuracy is controlled by N (moduli count, at most 20) and the input dynamic range;
  with repair ON, uncertified entries are repaired in FP64, so very wide dynamic range costs speed, not
  correctness (quantification: see docs/report.md, sections 5.3 and 9.5).
- Small or unprofitable problems may be routed to native FP64 by the planner.
- INT8 split-K (tile config 3) cannot be enabled through the ggemm API: the auto split-K rule
  (`i8_set_auto_splitk_min_kp`) only applies to problems submitted with `tile_cfg < 0`
  (`src/int8_grouped_gemm.cu`), and the ggemm planner always passes tile config 0..2. It is
  reachable only by calling `i8_grouped_gemm` directly (tests). A 4th launch in `EMU_GROUPED`
  comes from problems with k = 0 / alpha = 0 (or native fallbacks) that go to cuBLAS.
- `GGEMM_RHO_BUDGET` default is untuned.
- Single stream per handle; handles and plans are not thread-safe.
- Only FP64 real GEMM; no FP32 / complex types.

## Tests, benchmarks, profiling

All are SLURM scripts (submit with sbatch from the repo root, never run on login nodes);
outputs land in `results/<kind>_<jobid>/` with provenance and calibration logs.

```bash
sbatch scripts/run_tests.slurm      # correctness vs high-precision reference, edge cases, compute-sanitizer
sbatch scripts/run_bench.slurm      # timing of all paths on the workload families + accuracy sweep + plots
sbatch scripts/profile.slurm        # nsys / ncu profiling
sbatch scripts/sanitize.slurm       # compute-sanitizer runs
sbatch scripts/repair_sweep.slurm   # repair-mode sweep (plot with scripts/plot_repair.py)
sbatch scripts/final/run_final.slurm  # final benchmark run; aggregate with scripts/final/aggregate.py
sbatch scripts/i8_ablation.slurm    # INT8 grouped GEMM kernel ablation (also i8_cfg2.slurm, i8_cfg2sep.slurm, run_i8_unit.slurm)
sbatch bench/app/run_app.slurm      # application benchmarks: MoE FFN, block-sparse (FP64)
```

Binaries: `test_grouped`, `bench_grouped` (and `test_graph`, `test_i8_grouped`,
`test_operand_reuse` (renamed source: `tests/test_operand_reuse.cu`), `test_reuse_e2e`, `test_numerics_standalone`).
Workload families: uniform small, MoE (shared n,k; varying m), random hetero, LoRA (low rank).

Post-processing (CPU only, `scripts/analysis/`, run inside a job like everything else):
`nsys_report.py` (launch / busy / gap breakdown per path, uses the NVTX ranges
`ggemm:<path>`, stage ranges prep / i8_gemm / reconstruct), `ncu_report.py`,
`roofline.py` (roofline from bench CSV + calibration JSON), `selftest.py` (runs them on
`scripts/analysis/testdata`). `scripts/plot.py` plots bench CSVs.

Measured results are in `results/` (one directory per job) and the write-up with numbers
and discussion is `docs/report.md` (final report, Traditional Chinese). This README intentionally
quotes no measured numbers.

Not committed to git (size): per-job build dirs, `*.ncu-rep`, nsys `*.sqlite`, and the per-event
nsys trace exports `*_trace.csv`. The `.nsys-rep` files are committed; regenerate the traces with
`nsys stats -r cuda_gpu_trace,cuda_api_trace,nvtx_pushpop_trace -f csv -o <prefix> <file>.nsys-rep`
or pass the `.nsys-rep` directly to `scripts/analysis/nsys_report.py`. ncu summaries are committed
as `*_details.csv` / `*_raw.csv`. The final run is `results/final_590258/`.

## Layout

```
include/ggemm.h          public API (paths, config, stats, handle)
src/ggemm.cu             handle, dispatch of the 7 paths
src/ggemm_graph.{h,cu}   plan / execute (CUDA Graph)
src/planner.{h,cpp}      cost model, per-problem N / tile config / fallback
src/operand_reuse.{h,cpp} shared-operand detection (path 5)
src/ozaki2_numerics.*    scaling, residues, CRT reconstruct, certificate (agent-N)
src/int8_grouped_gemm.*  persistent grouped INT8 GEMM (agent-K)
src/internal.h           error codes, NVTX macros
tests/ bench/ scripts/   tests, benchmarks (bench/app = application benchmarks), job scripts, analysis
docs/                    related_work.md, discussion_1007.md, report.md (final report)
DESIGN.md                internal design and contracts
```
