# ggemm: Grouped FP64 GEMM emulated on INT8 Tensor Cores (Ozaki scheme II)

ggemm computes, for a group of G independent problems with different shapes,

    C_g = alpha_g * A_g (m_g x k_g) * B_g (k_g x n_g) + beta_g * C_g

in FP64 (column-major, no transpose in v1). The emulation path uses INT8 Tensor
Cores with the Ozaki scheme II (CRT) and fuses the whole group into a fixed number
of kernel launches. Target GPU: RTX 5090 (sm_120, weak FP64); development GPU:
H100/H200 (sm_90). Implementation is original code; GEMMul8 and the papers were
only read for the algorithm (see docs/report.md, AI / external code section).

## Setup

Requirements: CUDA 13.0 toolkit, CMake, a C++17 compiler, cuBLAS.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="90;120"
cmake --build build -j
```

Equivalent nvcc flags: `-gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120`.
Note: the build system lives in CMakeLists.txt (maintained by agent-P); flag names above
follow standard CMake and may be adjusted there (TODO: confirm once CMakeLists.txt lands).

Do not run GPU programs on login nodes. Use the SLURM scripts in `scripts/`
(account GOV114009, partition dev, `--gpus-per-node=1`).

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
if (rc != 0) { /* workspace too small or invalid args; no hidden allocation */ }
// st.num_fallback, st.kernel_launches, st.moduli_min/max
ggemm_destroy(h);
```

Config: `num_moduli` = 0 for adaptive N per problem, otherwise fixed N in
[2, GGEMM_MAX_MODULI = 20]. `target_rel_err` is used when `num_moduli == 0`
(default 1e-15). `allow_fallback = 1` routes unsafe or unprofitable problems to native FP64
and reports the count in `stats.num_fallback`.

## The five paths (`ggemm_path_t`)

| Path | What it does | Role |
|---|---|---|
| `GGEMM_PATH_NATIVE_LOOP` | `cublasDgemm` once per problem | native FP64 baseline |
| `GGEMM_PATH_NATIVE_GROUPED` | `cublasDgemmGroupedBatched` | native FP64 grouped baseline |
| `GGEMM_PATH_EMU_LOOP` | single-GEMM INT8 Ozaki-II emulation called per problem (O(G*N) launches) | starter-like baseline, isolates the launch-merging gain |
| `GGEMM_PATH_EMU_GROUPED` | fused pipeline: prep -> persistent grouped INT8 GEMM -> reconstruct (fixed 3 kernels, one descriptor H2D) | ours |
| `GGEMM_PATH_AUTO` | EMU_GROUPED plus planner: per-problem moduli count N, per-problem tile config (128x128 / 64x64 / 32x32), LPT tile scheduling, fallback to native FP64 | ours (default) |

## Known limitations (v1)

- No transpose: column-major, A is m x k, B is k x n, op = N only.
- K <= 131072 per problem (INT32 accumulation limit of the INT8 GEMM); larger K must
  fall back to native FP64 or be rejected.
- Workspace is fixed at `ggemm_create`: calls exceeding `max_problems` or the total
  A/B/C element bounds return a nonzero error code; there is no allocation fallback.
- Descriptor array must be on the host; data pointers must be device pointers.
- Accuracy is controlled by N (moduli count, at most 20) and the input dynamic range;
  very wide dynamic range may force fallback or lower accuracy (TODO: quantify in report).
- Small or unprofitable problems may be routed to native FP64 by the planner.
- Single stream per handle; a handle is not thread-safe (TODO: confirm against implementation).
- Only the paths above are supported; no FP32/complex types.

## Tests and benchmarks

Scripts are written by agent-B under `scripts/`. File names below are placeholders
(TODO: update to the real names). Each is a SLURM script; submit with sbatch, never run on login nodes.

```bash
sbatch scripts/run_tests.sbatch     # correctness: all paths vs high-precision reference, edge cases
sbatch scripts/run_bench.sbatch     # timing of 5 paths on the test families
sbatch scripts/run_profile.sbatch   # nsys / ncu profiling
sbatch scripts/run_sanitizer.sbatch # compute-sanitizer memcheck
```

Test families: uniform small, MoE (shared n,k; varying m), random hetero, LoRA (low rank).

## Layout

```
include/ggemm.h          public API
src/                     numerics (agent-N), INT8 grouped GEMM (agent-K), pipeline/planner (agent-P)
tests/ bench/ scripts/   agent-B
docs/                    related_work.md, discussion_1007.md, report.md
DESIGN.md                internal design and contracts
```
