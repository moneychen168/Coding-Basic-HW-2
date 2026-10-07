// ggemm public API (agent-P): workspace management + the 5 execution paths.
#include "internal.h"
#include "operand_reuse.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

// --------------------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------------------
#define CU(x) do { if ((x) != cudaSuccess) return GGEMM_ERR_CUDA; } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) return GGEMM_ERR_CUBLAS; } while (0)

static const size_t kCublasWsBytes = 64ull << 20;   // fixed cuBLAS workspace (no lazy alloc in timed calls)
static const int kDescSlots = 2;                    // == size of ggemm_handle_s::desc_done

// Descriptor blob layout: see ggemm_blob_layout in internal.h.
ggemm_blob_layout ggemm_layout_for(size_t num_emu, size_t num_prep, size_t num_i8,
                                   size_t sched_bytes, size_t num_nat) {
  ggemm_blob_layout L;
  L.off_oz    = 0;
  L.off_prep  = num_prep ? ggemm_align_up(L.off_oz + num_emu * sizeof(oz2_problem), GGEMM_BUF_ALIGN)
                         : L.off_oz;
  L.off_i8    = ggemm_align_up((num_prep ? L.off_prep + num_prep * sizeof(oz2_problem)
                                         : L.off_oz + num_emu * sizeof(oz2_problem)),
                               GGEMM_BUF_ALIGN);
  L.off_sched = ggemm_align_up(L.off_i8 + num_i8 * sizeof(i8_problem), GGEMM_BUF_ALIGN);
  L.off_ptr   = ggemm_align_up(L.off_sched + sched_bytes, GGEMM_BUF_ALIGN);
  L.total     = L.off_ptr + 3 * num_nat * sizeof(void*);
  return L;
}

static void* host_slot(ggemm_handle_t h, int s) {
  return static_cast<char*>(h->h_desc) + (size_t)s * h->desc_bytes;
}

// Acquire a pinned slot whose previous call has fully drained on the stream.
static int acquire_slot(ggemm_handle_t h, int* slot) {
  int s = h->desc_next; h->desc_next = (s + 1) % kDescSlots;
  CU(cudaEventSynchronize(h->desc_done[s]));
  *slot = s;
  return GGEMM_OK;
}

static int release_slot(ggemm_handle_t h, int slot) {
  CU(cudaEventRecord(h->desc_done[slot], h->stream));
  return GGEMM_OK;
}

int ggemm_validate(ggemm_handle_t h, const ggemm_problem_t* p, int num) {
  if (num < 0 || num > h->max_problems || (num > 0 && !p)) return GGEMM_ERR_INVALID;
  size_t ta = 0, tb = 0, tc = 0;
  for (int g = 0; g < num; ++g) {
    const ggemm_problem_t& q = p[g];
    if (q.m < 0 || q.n < 0 || q.k < 0) return GGEMM_ERR_INVALID;
    if (q.lda < std::max(1, q.m) || q.ldb < std::max(1, q.k) || q.ldc < std::max(1, q.m))
      return GGEMM_ERR_INVALID;
    if (q.m > 0 && q.n > 0) {
      if (!q.C) return GGEMM_ERR_INVALID;
      if (q.k > 0 && (!q.A || !q.B)) return GGEMM_ERR_INVALID;
    }
    ta += (size_t)q.m * q.k; tb += (size_t)q.k * q.n; tc += (size_t)q.m * q.n;
  }
  // Contract of ggemm_create: the call may not exceed the element budgets (all paths).
  if (ta > h->max_A_elems || tb > h->max_B_elems || tc > h->max_C_elems) return GGEMM_ERR_WORKSPACE;
  return GGEMM_OK;
}

static int dgemm_one(ggemm_handle_t h, const ggemm_problem_t& p) {
  CB(cublasDgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, p.m, p.n, p.k, &p.alpha, p.A, p.lda,
                 p.B, p.ldb, &p.beta, p.C, p.ldc));
  return GGEMM_OK;
}

// Native FP64 for problems idx[0..cnt): cublasDgemmGroupedBatched (one group per problem,
// group_size = 1) when cnt >= 2 and `grouped`, else one cublasDgemm each.
// d_ptrs: device array [A x cnt | B x cnt | C x cnt], already uploaded.
static int run_native(ggemm_handle_t h, const ggemm_problem_t* probs, const int* idx, int cnt,
                      bool grouped, void* d_ptrs, int* launches) {
  if (cnt == 0) return GGEMM_OK;
  if (!grouped || cnt == 1) {
    for (int i = 0; i < cnt; ++i) {
      int r = dgemm_one(h, probs[idx[i]]);
      if (r) return r;
      ++*launches;
    }
    return GGEMM_OK;
  }
  for (int i = 0; i < cnt; ++i) {
    const ggemm_problem_t& p = probs[idx[i]];
    h->nat_ta[i] = CUBLAS_OP_N; h->nat_tb[i] = CUBLAS_OP_N;
    h->nat_m[i] = p.m; h->nat_n[i] = p.n; h->nat_k[i] = p.k;
    h->nat_lda[i] = p.lda; h->nat_ldb[i] = p.ldb; h->nat_ldc[i] = p.ldc;
    h->nat_alpha[i] = p.alpha; h->nat_beta[i] = p.beta; h->nat_gsize[i] = 1;
  }
  void* const* base = static_cast<void* const*>(d_ptrs);
  const double* const* dA = reinterpret_cast<const double* const*>(base);
  const double* const* dB = reinterpret_cast<const double* const*>(base + cnt);
  double* const* dC = reinterpret_cast<double* const*>(base + 2 * cnt);
  // Signature (cuBLAS >= 12.5; checked against cuda-13.0 cublas_api.h + docs): pointer arrays
  // Aarray/Barray/Carray in DEVICE memory, every other array in HOST memory.
  CB(cublasDgemmGroupedBatched(h->cublas, h->nat_ta, h->nat_tb, h->nat_m, h->nat_n, h->nat_k,
                               h->nat_alpha, dA, h->nat_lda, dB, h->nat_ldb, h->nat_beta,
                               dC, h->nat_ldc, cnt, h->nat_gsize));
  ++*launches;
  return GGEMM_OK;
}

static void write_ptrs(char* hb, const ggemm_blob_layout& L, const ggemm_problem_t* probs,
                       const int* idx, int cnt) {
  const void** P = reinterpret_cast<const void**>(hb + L.off_ptr);
  for (int i = 0; i < cnt; ++i) {
    P[i] = probs[idx[i]].A; P[cnt + i] = probs[idx[i]].B; P[2 * cnt + i] = probs[idx[i]].C;
  }
}

// --------------------------------------------------------------------------------------
// create / destroy
// --------------------------------------------------------------------------------------
extern "C" void ggemm_destroy(ggemm_handle_t h) {
  if (!h) return;
  cudaStreamSynchronize(h->stream);
  for (int s = 0; s < kDescSlots; ++s) if (h->desc_done[s]) cudaEventDestroy(h->desc_done[s]);
  if (h->cublas) cublasDestroy(h->cublas);
  cudaFree(h->cublas_ws); cudaFree(h->A8); cudaFree(h->B8); cudaFree(h->C32);
  cudaFree(h->row_shift); cudaFree(h->col_shift); cudaFree(h->d_desc);
  if (h->h_desc) cudaFreeHost(h->h_desc);
  delete[] h->plan; delete[] h->emu_idx; delete[] h->nat_idx; delete[] h->loop_tiles;
  delete[] h->nat_ta; delete[] h->nat_tb;
  delete[] h->nat_m; delete[] h->nat_n; delete[] h->nat_k;
  delete[] h->nat_lda; delete[] h->nat_ldb; delete[] h->nat_ldc; delete[] h->nat_gsize;
  delete[] h->nat_alpha; delete[] h->nat_beta;
  delete h->reuse; delete[] h->reuse_nmod;
  delete[] h->ua_off8; delete[] h->ua_offs; delete[] h->ub_off8; delete[] h->ub_offs;
  ::operator delete(h);
}

extern "C" int ggemm_create(ggemm_handle_t* out, int max_problems, size_t max_A, size_t max_B,
                            size_t max_C, cudaStream_t stream) {
  if (!out || max_problems <= 0) return GGEMM_ERR_INVALID;
  *out = nullptr;
  void* mem = ::operator new(sizeof(ggemm_handle_s), std::nothrow);
  if (!mem) return GGEMM_ERR_ALLOC;
  std::memset(mem, 0, sizeof(ggemm_handle_s));
  ggemm_handle_t h = static_cast<ggemm_handle_t>(mem);
  h->stream = stream;
  h->max_problems = max_problems;
  h->max_A_elems = max_A; h->max_B_elems = max_B; h->max_C_elems = max_C;
  const size_t P = (size_t)max_problems;
  // agent-N: a problem with N moduli uses oz2_num_planes(N) = N + OZ2_EXTRA_PLANES planes in
  // A8/B8/C32 (residues + abs plane), and OZ2_ROW_WS_PER_ROW int32 per row/column of shifts.
  const size_t M = (size_t)oz2_num_planes(GGEMM_MAX_MODULI);

#define FAIL(code) do { ggemm_destroy(h); return (code); } while (0)
#define CUA(x) do { if ((x) != cudaSuccess) FAIL(GGEMM_ERR_ALLOC); } while (0)

  cudaDeviceProp prop;
  if (cudaGetDevice(&h->device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, h->device) != cudaSuccess) FAIL(GGEMM_ERR_CUDA);
  h->num_sms = prop.multiProcessorCount;
  ggemm_device_model_init(&h->dm, prop.name, prop.major, prop.minor, h->num_sms);
  // agent-N: upload CRT tables to __constant__ now, so no H2D happens inside timed calls.
  if (oz2_init_constants() != 0) FAIL(GGEMM_ERR_NUMERICS);

  // cuBLAS: fixed stream, host pointer mode, plain FP64 math (no cuBLAS-side emulation),
  // user-provided workspace so no allocation happens inside timed calls.
  if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) FAIL(GGEMM_ERR_CUBLAS);
  h->cublas_ws_bytes = kCublasWsBytes;
  CUA(cudaMalloc(&h->cublas_ws, h->cublas_ws_bytes));
  if (cublasSetStream(h->cublas, stream) != CUBLAS_STATUS_SUCCESS ||
      cublasSetWorkspace(h->cublas, h->cublas_ws, h->cublas_ws_bytes) != CUBLAS_STATUS_SUCCESS ||
      cublasSetPointerMode(h->cublas, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS ||
      cublasSetMathMode(h->cublas, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
    FAIL(GGEMM_ERR_CUBLAS);

  // Residues: Kp = round_up(k,32) inflates m*k by <= 2x when k >= 32 (k < 32 can exceed it
  // and then the call returns GGEMM_ERR_WORKSPACE); + per-problem 256 B alignment slack.
  h->A8_bytes  = M * (2 * max_A + P * GGEMM_BUF_ALIGN);
  h->B8_bytes  = M * (2 * max_B + P * GGEMM_BUF_ALIGN);
  h->C32_bytes = M * (max_C + P * (GGEMM_BUF_ALIGN / 4)) * sizeof(int32_t);
  // sum(m) <= sum(m*k) for k >= 1 (same for n); + 64-int alignment per problem
  h->row_shift_elems = OZ2_ROW_WS_PER_ROW * max_A + P * 64;
  h->col_shift_elems = OZ2_ROW_WS_PER_ROW * max_B + P * 64;
  CUA(cudaMalloc(&h->A8, h->A8_bytes));
  CUA(cudaMalloc(&h->B8, h->B8_bytes));
  CUA(cudaMalloc(&h->C32, h->C32_bytes));
  CUA(cudaMalloc(&h->row_shift, h->row_shift_elems * sizeof(int32_t)));
  CUA(cudaMalloc(&h->col_shift, h->col_shift_elems * sizeof(int32_t)));

  // Descriptors: worst case = all problems emulated with MAX moduli (+ schedule image(s):
  // one grouped image, or one image per (prob, mod) for EMU_LOOP), plus any native pointers.
  h->sched_img_bytes_1 = ggemm_align_up(i8_grouped_gemm_sched_ws_bytes(1, 0), GGEMM_BUF_ALIGN);
  h->sched_img_bytes_P = ggemm_align_up(i8_grouped_gemm_sched_ws_bytes((int)M, 0), GGEMM_BUF_ALIGN);
  const size_t sched_max = std::max({i8_grouped_gemm_sched_ws_bytes((int)(P * M), 0),
                                     P * M * h->sched_img_bytes_1, P * h->sched_img_bytes_P});
  // oz2 area: P recon + up to 2P prep descriptors (operand reuse: UA + UB <= 2 * num_emu).
  ggemm_blob_layout Lw = ggemm_layout_for(P, 2 * P, P * M, sched_max, P);
  h->desc_bytes = ggemm_align_up(Lw.total, GGEMM_BUF_ALIGN);
  CUA(cudaMalloc(&h->d_desc, h->desc_bytes));
  CUA(cudaHostAlloc(&h->h_desc, h->desc_bytes * kDescSlots, cudaHostAllocDefault));
  for (int s = 0; s < kDescSlots; ++s) {
    CUA(cudaEventCreateWithFlags(&h->desc_done[s], cudaEventDisableTiming));
    CUA(cudaEventRecord(h->desc_done[s], stream));
  }

  h->plan = new (std::nothrow) ggemm_plan_entry_t[P];
  h->emu_idx = new (std::nothrow) int[P];
  h->nat_idx = new (std::nothrow) int[P];
  h->loop_tiles = new (std::nothrow) int[P * M];
  h->nat_ta = new (std::nothrow) cublasOperation_t[P];
  h->nat_tb = new (std::nothrow) cublasOperation_t[P];
  h->nat_m = new (std::nothrow) int[P]; h->nat_n = new (std::nothrow) int[P];
  h->nat_k = new (std::nothrow) int[P]; h->nat_lda = new (std::nothrow) int[P];
  h->nat_ldb = new (std::nothrow) int[P]; h->nat_ldc = new (std::nothrow) int[P];
  h->nat_gsize = new (std::nothrow) int[P];
  h->nat_alpha = new (std::nothrow) double[P]; h->nat_beta = new (std::nothrow) double[P];
  if (!h->plan || !h->loop_tiles || !h->emu_idx || !h->nat_idx || !h->nat_ta || !h->nat_tb || !h->nat_m ||
      !h->nat_n || !h->nat_k || !h->nat_lda || !h->nat_ldb || !h->nat_ldc || !h->nat_gsize ||
      !h->nat_alpha || !h->nat_beta) FAIL(GGEMM_ERR_ALLOC);

  // operand reuse: host arena + offset scratch, all sized here (no allocation inside calls)
  h->reuse = new (std::nothrow) reuse_arena();
  if (!h->reuse) FAIL(GGEMM_ERR_ALLOC);
  try {
    if (h->reuse->reserve(max_problems) != REUSE_OK) FAIL(GGEMM_ERR_INVALID);
  } catch (...) { FAIL(GGEMM_ERR_ALLOC); }
  h->reuse_nmod = new (std::nothrow) int[P];
  h->ua_off8 = new (std::nothrow) size_t[P]; h->ua_offs = new (std::nothrow) size_t[P];
  h->ub_off8 = new (std::nothrow) size_t[P]; h->ub_offs = new (std::nothrow) size_t[P];
  if (!h->reuse_nmod || !h->ua_off8 || !h->ua_offs || !h->ub_off8 || !h->ub_offs) FAIL(GGEMM_ERR_ALLOC);
  {
    const char* e = std::getenv("GGEMM_REUSE_PROMOTE");
    h->reuse_promote = (e && std::atoi(e) == 1) ? 1 : 0;
    e = std::getenv("GGEMM_STATS_REPAIR");
    h->stats_repair = (e && std::atoi(e) == 1) ? 1 : 0;
  }

  if (cudaStreamSynchronize(stream) != cudaSuccess) FAIL(GGEMM_ERR_CUDA);
#undef CUA
#undef FAIL
  *out = h;
  return GGEMM_OK;
}

// --------------------------------------------------------------------------------------
// native paths
// --------------------------------------------------------------------------------------
static int path_native(ggemm_handle_t h, bool grouped, const ggemm_problem_t* probs, int num,
                       ggemm_stats_t* st) {
  int cnt = 0;
  for (int g = 0; g < num; ++g)
    if (probs[g].m > 0 && probs[g].n > 0) h->nat_idx[cnt++] = g;
  if (!grouped || cnt < 2) {
    GGEMM_NVTX_RANGE(nv_cublas, "cublas");
    return run_native(h, probs, h->nat_idx, cnt, false, nullptr, &st->kernel_launches);
  }
  int slot, r;
  if ((r = acquire_slot(h, &slot))) return r;
  char* hb = static_cast<char*>(host_slot(h, slot));
  ggemm_blob_layout L = ggemm_layout_for(0, 0, 0, 0, cnt);
  write_ptrs(hb, L, probs, h->nat_idx, cnt);
  {
    GGEMM_NVTX_RANGE(nv_h2d, "h2d");
    CU(cudaMemcpyAsync(h->d_desc, hb, L.total, cudaMemcpyHostToDevice, h->stream));
  }
  GGEMM_NVTX_RANGE(nv_cublas, "cublas");
  r = run_native(h, probs, h->nat_idx, cnt, true, static_cast<char*>(h->d_desc) + L.off_ptr,
                 &st->kernel_launches);
  int r2 = release_slot(h, slot);
  return r ? r : r2;
}

// --------------------------------------------------------------------------------------
// emulation paths (EMU_LOOP, EMU_GROUPED, AUTO)
// --------------------------------------------------------------------------------------
static int path_emu(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                    const ggemm_problem_t* probs, int num, ggemm_stats_t* st) {
  const bool loop = (path == GGEMM_PATH_EMU_LOOP);
  const bool lbat = (path == GGEMM_PATH_EMU_LOOP_BATCHED);   // per problem, planes batched
  // Operand reuse only on its own path (clean ablation: EMU_GROUPED -> +reuse = path 5).
  const bool reuse = (path == GGEMM_PATH_EMU_GROUPED_REUSE);
  // The planner knows EMU_LOOP / EMU_GROUPED / AUTO; REUSE plans exactly like EMU_GROUPED.
  // EMU_LOOP_BATCHED plans exactly like EMU_LOOP.
  const ggemm_path_t plan_path = (path == GGEMM_PATH_EMU_GROUPED_REUSE) ? GGEMM_PATH_EMU_GROUPED
                               : (path == GGEMM_PATH_EMU_LOOP_BATCHED) ? GGEMM_PATH_EMU_LOOP : path;
  GGEMM_NVTX_RANGE(nv_plan, "plan");   // planning + reuse + descriptors + LPT schedule
  int r = ggemm_plan(probs, num, cfg, plan_path, &h->dm, h->plan);
  if (r) return r;

  // ---- operand reuse (host, O(G), no allocation): after the plan, before any sizing.
  reuse_plan rp{};
  if (reuse) {
    for (int g = 0; g < num; ++g)
      h->reuse_nmod[g] = h->plan[g].route == GGEMM_ROUTE_EMU ? h->plan[g].num_moduli : 0;
    rp = build_operand_reuse(*h->reuse, probs, num, h->reuse_nmod,
                             h->reuse_promote ? REUSE_PROMOTE : REUSE_EXACT_N);
    if (rp.status != REUSE_OK)
      return rp.status == REUSE_ERR_CAPACITY ? GGEMM_ERR_WORKSPACE : GGEMM_ERR_INVALID;
    if (h->reuse_promote)   // PROMOTE: every problem runs with its component's max N
      for (int g = 0; g < num; ++g)
        if (h->plan[g].route == GGEMM_ROUTE_EMU) h->plan[g].num_moduli = rp.eff_nmod[g];
  }

  // ---- partition + workspace offsets ([plane][prob] slabs; slab stride shared by all problems)
  // Without reuse every emulated problem owns its A8/B8/shift blocks; with reuse the blocks are
  // per unique operand (offsets in ua_off* / ub_off*), so SA / SB / rows / cols shrink.
  int ne = 0, nn = 0, nmax = 0, pmax = 0;
  size_t SA = 0, SB = 0, SC = 0, rows = 0, cols = 0, n_i8 = 0;
  for (int g = 0; g < num; ++g) {
    const ggemm_plan_entry_t& e = h->plan[g];
    if (e.route == GGEMM_ROUTE_EMPTY) continue;
    if (e.route != GGEMM_ROUTE_EMU) { h->nat_idx[nn++] = g; continue; }
    const ggemm_problem_t& p = probs[g];
    const size_t kp = ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
    h->emu_idx[ne++] = g;
    if (!reuse) {
      SA += ggemm_align_up((size_t)p.m * kp, GGEMM_BUF_ALIGN);
      SB += ggemm_align_up((size_t)p.n * kp, GGEMM_BUF_ALIGN);
      rows += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.m, 64);
      cols += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.n, 64);
    }
    SC += ggemm_align_up((size_t)p.m * p.n, GGEMM_BUF_ALIGN / 4);
    const int planes = oz2_num_planes(e.num_moduli);
    n_i8 += planes;
    nmax = std::max(nmax, e.num_moduli);
    pmax = std::max(pmax, planes);
  }
  int pmax_m = 0, pmax_n = 0;   // prep grid maxima (reuse: over unique operands)
  if (reuse) {
    for (int u = 0; u < rp.num_ua; ++u) {
      const ggemm_problem_t& p = probs[rp.ua_rep[u]];
      const size_t kp = ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
      h->ua_off8[u] = SA; h->ua_offs[u] = rows;
      SA += ggemm_align_up((size_t)p.m * kp, GGEMM_BUF_ALIGN);
      rows += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.m, 64);
      pmax = std::max(pmax, oz2_num_planes(rp.ua_nmod[u]));
      pmax_m = std::max(pmax_m, p.m);
    }
    for (int u = 0; u < rp.num_ub; ++u) {
      const ggemm_problem_t& p = probs[rp.ub_rep[u]];
      const size_t kp = ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
      h->ub_off8[u] = SB; h->ub_offs[u] = cols;
      SB += ggemm_align_up((size_t)p.n * kp, GGEMM_BUF_ALIGN);
      cols += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.n, 64);
      pmax = std::max(pmax, oz2_num_planes(rp.ub_nmod[u]));
      pmax_n = std::max(pmax_n, p.n);
    }
  }
  if ((size_t)pmax * SA > h->A8_bytes || (size_t)pmax * SB > h->B8_bytes ||
      (size_t)pmax * SC * sizeof(int32_t) > h->C32_bytes || rows > h->row_shift_elems ||
      cols > h->col_shift_elems)
    return GGEMM_ERR_WORKSPACE;
  if (ne > 65535) return GGEMM_ERR_INVALID;   // agent-N prep/reconstruct grid limit
  const int nprep = reuse ? rp.num_ua + rp.num_ub : 0;
  if (nprep > 65535) return GGEMM_ERR_INVALID;

  // ---- stats. num_fallback counts precision/profit reroutes (K too large, target
  // unreachable, tiny, unprofitable). k==0 problems (C = beta*C, exact) also run on cuBLAS
  // but are not a "fallback": nothing was emulated-or-not. They do count in kernel_launches.
  st->num_fallback = 0;
  for (int i = 0; i < nn; ++i) st->num_fallback += (h->plan[h->nat_idx[i]].route != GGEMM_ROUTE_K_ZERO);
  st->moduli_min = ne ? GGEMM_MAX_MODULI : 0;
  st->moduli_max = nmax;
  for (int j = 0; j < ne; ++j) st->moduli_min = std::min(st->moduli_min, h->plan[h->emu_idx[j]].num_moduli);
  st->reuse_enabled = reuse ? 1 : 0;
  st->num_unique_A = reuse ? rp.num_ua : ne;
  st->num_unique_B = reuse ? rp.num_ub : ne;
  st->num_promoted = reuse ? rp.num_promoted : 0;

  // ---- fill descriptors in the pinned slot (oz2 recon per problem; i8 per (prob, plane),
  // problem-major; with reuse also UA A-only + UB B-only prep descriptors)
  int slot;
  if ((r = acquire_slot(h, &slot))) return r;
  char* hb = static_cast<char*>(host_slot(h, slot));
  char* db = static_cast<char*>(h->d_desc);
  const bool nat_grouped = (nn >= 2);
  const size_t sched_bytes = ne == 0 ? 0
      : (loop ? n_i8 * h->sched_img_bytes_1
         : lbat ? (size_t)ne * h->sched_img_bytes_P
                : i8_grouped_gemm_sched_ws_bytes((int)n_i8, 0));
  ggemm_blob_layout L = ggemm_layout_for(ne, nprep, n_i8, sched_bytes, nat_grouped ? nn : 0);
  oz2_problem* hoz = reinterpret_cast<oz2_problem*>(hb + L.off_oz);
  oz2_problem* hpr = reinterpret_cast<oz2_problem*>(hb + L.off_prep);
  i8_problem* hi8 = reinterpret_cast<i8_problem*>(hb + L.off_i8);
  const oz2_problem* doz = reinterpret_cast<const oz2_problem*>(db + L.off_oz);
  const oz2_problem* dpr = reinterpret_cast<const oz2_problem*>(db + L.off_prep);
  const i8_problem* di8 = reinterpret_cast<const i8_problem*>(db + L.off_i8);

  int max_m = 0, max_n = 0;
  size_t oa = 0, ob = 0, oc = 0, orow = 0, ocol = 0, q = 0;
  for (int j = 0; j < ne; ++j) {
    const int g = h->emu_idx[j];
    const ggemm_problem_t& p = probs[g];
    const ggemm_plan_entry_t& e = h->plan[g];
    const size_t kp = ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
    oz2_problem& o = hoz[j];
    o.m = p.m; o.n = p.n; o.k = p.k; o.kp = (int)kp; o.num_moduli = e.num_moduli;
    o.lda = p.lda; o.ldb = p.ldb; o.ldc = p.ldc;
    o.A = p.A; o.B = p.B; o.C = p.C; o.alpha = p.alpha; o.beta = p.beta;
    if (reuse) {
      const int ua = rp.prob_a[g], ub = rp.prob_b[g];
      // M2: prep of the shared A / B and this problem's GEMM + reconstruct must use the same N
      // (same T_N shifts and the same planes 0..N). EXACT_N: by key; PROMOTE: eff_nmod was
      // written back into plan[g] above and equals ua_nmod / ub_nmod of its component.
      const bool n_ok = rp.ua_nmod[ua] == e.num_moduli && rp.ub_nmod[ub] == e.num_moduli;
      assert(n_ok);
      if (!n_ok) { release_slot(h, slot); return GGEMM_ERR_NUMERICS; }
      o.A8 = h->A8 + h->ua_off8[ua]; o.row_shift = h->row_shift + h->ua_offs[ua];
      o.B8 = h->B8 + h->ub_off8[ub]; o.col_shift = h->col_shift + h->ub_offs[ub];
    } else {
      o.A8 = h->A8 + oa; o.row_shift = h->row_shift + orow;
      o.B8 = h->B8 + ob; o.col_shift = h->col_shift + ocol;
    }
    o.a8_mod_stride = SA;   // bytes == int8 elements
    o.b8_mod_stride = SB;
    o.C32 = h->C32 + oc; o.c32_mod_stride = SC;   // int32 elements
    const int planes = oz2_num_planes(e.num_moduli);
    for (int i = 0; i < planes; ++i, ++q) {
      i8_problem& d = hi8[q];
      d.m = p.m; d.n = p.n; d.kp = (int)kp;
      d.A = o.A8 + (size_t)i * SA; d.B = o.B8 + (size_t)i * SB; d.C = o.C32 + (size_t)i * SC;
      d.tile_cfg = e.tile_cfg;
    }
    oa += ggemm_align_up((size_t)p.m * kp, GGEMM_BUF_ALIGN);
    ob += ggemm_align_up((size_t)p.n * kp, GGEMM_BUF_ALIGN);
    oc += ggemm_align_up((size_t)p.m * p.n, GGEMM_BUF_ALIGN / 4);
    orow += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.m, 64);
    ocol += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * p.n, 64);
    max_m = std::max(max_m, p.m); max_n = std::max(max_n, p.n);
  }
  if (reuse) {
    // Prep descriptors: [0, UA) A-only (n = 0: agent-N's prep skips the B blocks),
    // [UA, UA+UB) B-only (m = 0: skips the A blocks). Shift T_N comes from num_moduli.
    for (int u = 0; u < rp.num_ua; ++u) {
      const ggemm_problem_t& p = probs[rp.ua_rep[u]];
      oz2_problem& o = hpr[u];
      std::memset(&o, 0, sizeof o);
      o.m = p.m; o.n = 0; o.k = p.k; o.kp = (int)ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
      o.num_moduli = rp.ua_nmod[u]; o.lda = p.lda; o.ldb = p.ldb; o.ldc = std::max(1, p.m);
      o.A = p.A; o.alpha = 1.0;
      o.A8 = h->A8 + h->ua_off8[u]; o.a8_mod_stride = SA; o.b8_mod_stride = SB;
      o.c32_mod_stride = SC;
      o.row_shift = h->row_shift + h->ua_offs[u];
    }
    for (int u = 0; u < rp.num_ub; ++u) {
      const ggemm_problem_t& p = probs[rp.ub_rep[u]];
      oz2_problem& o = hpr[rp.num_ua + u];
      std::memset(&o, 0, sizeof o);
      o.m = 0; o.n = p.n; o.k = p.k; o.kp = (int)ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
      o.num_moduli = rp.ub_nmod[u]; o.lda = 1; o.ldb = p.ldb; o.ldc = 1;
      o.B = p.B; o.alpha = 1.0;
      o.B8 = h->B8 + h->ub_off8[u]; o.a8_mod_stride = SA; o.b8_mod_stride = SB;
      o.c32_mod_stride = SC;
      o.col_shift = h->col_shift + h->ub_offs[u];
    }
  } else {
    pmax_m = max_m; pmax_n = max_n;
  }
  if (nat_grouped) write_ptrs(hb, L, probs, h->nat_idx, nn);

  // ---- agent-K LPT schedule image(s), built on host straight into the pinned slot so they
  // ride along with the descriptors in the single H2D (avoids K's pageable-memcpy wrapper).
  int grp_tiles = 0;
  if (ne > 0) {
    if (loop) {
      for (size_t i = 0; i < n_i8; ++i)
        if (i8_grouped_gemm_build_schedule(hi8 + i, 1, hb + L.off_sched + i * h->sched_img_bytes_1,
                                           h->sched_img_bytes_1, &h->loop_tiles[i])) {
          release_slot(h, slot); return GGEMM_ERR_I8GEMM;
        }
    } else if (lbat) {
      size_t qq = 0;   // image j covers problem j's planes (consecutive in hi8)
      for (int j = 0; j < ne; ++j) {
        const int planes = oz2_num_planes(hoz[j].num_moduli);
        if (i8_grouped_gemm_build_schedule(hi8 + qq, planes, hb + L.off_sched + j * h->sched_img_bytes_P,
                                           h->sched_img_bytes_P, &h->loop_tiles[j])) {
          release_slot(h, slot); return GGEMM_ERR_I8GEMM;
        }
        qq += planes;
      }
    } else if (i8_grouped_gemm_build_schedule(hi8, (int)n_i8, hb + L.off_sched, sched_bytes,
                                              &grp_tiles)) {
      release_slot(h, slot); return GGEMM_ERR_I8GEMM;
    }
  }

  // ---- single descriptor H2D (pinned, async). EMU_LOOP also uploads once up front: the
  // baseline is then penalized only by its launch structure, not by extra memcpys.
  GGEMM_NVTX_END(nv_plan);
  {
    GGEMM_NVTX_RANGE(nv_h2d, "h2d");
    if (L.total) CU(cudaMemcpyAsync(db, hb, L.total, cudaMemcpyHostToDevice, h->stream));
  }

  int& nl = st->kernel_launches;
  if (ne > 0) {
    if (loop) {
      // Starter-like baseline (no starter source available to us, so we model the usual
      // single-GEMM emulation): per problem 1 prep launch, then ONE INT8 GEMM launch PER
      // PLANE (the single-GEMM emulator calls its INT8 GEMM once per modulus; + agent-N's abs
      // plane), then 1 reconstruct launch => planes_g + 2 launches per problem.
      size_t qq = 0;
      for (int j = 0; j < ne; ++j) {
        const ggemm_problem_t& p = probs[h->emu_idx[j]];
        {
          GGEMM_NVTX_RANGE(nv_prep, "prep");
          if (oz2_prep_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        }
        ++nl;
        const int planes = oz2_num_planes(hoz[j].num_moduli);
        for (int i = 0; i < planes; ++i, ++qq) {
          GGEMM_NVTX_RANGE(nv_i8, "i8_gemm");
          if (i8_grouped_gemm_launch(di8 + qq, 1, db + L.off_sched + qq * h->sched_img_bytes_1,
                                     h->loop_tiles[qq], h->stream)) {
            r = GGEMM_ERR_I8GEMM; break;
          }
          if (h->loop_tiles[qq] > 0) ++nl;   // K launches nothing when total_tiles == 0
        }
        if (r) break;
        GGEMM_NVTX_RANGE(nv_rec, "reconstruct");
        if (oz2_reconstruct_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        ++nl;
      }
    } else if (lbat) {
      // GEMMul8-like baseline: per problem prep, ONE INT8 GEMM launch for all its planes (the
      // planes are batched like a strided-batched GEMM), reconstruct => 3 launches per problem.
      size_t qq = 0;
      for (int j = 0; j < ne; ++j) {
        const ggemm_problem_t& p = probs[h->emu_idx[j]];
        const int planes = oz2_num_planes(hoz[j].num_moduli);
        {
          GGEMM_NVTX_RANGE(nv_prep, "prep");
          if (oz2_prep_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        }
        ++nl;
        {
          GGEMM_NVTX_RANGE(nv_i8, "i8_gemm");
          if (i8_grouped_gemm_launch(di8 + qq, planes, db + L.off_sched + j * h->sched_img_bytes_P,
                                     h->loop_tiles[j], h->stream)) { r = GGEMM_ERR_I8GEMM; break; }
          if (h->loop_tiles[j] > 0) ++nl;
        }
        qq += planes;
        GGEMM_NVTX_RANGE(nv_rec, "reconstruct");
        if (oz2_reconstruct_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        ++nl;
      }
    } else {
      // Fused grouped pipeline: exactly 3 launches for all emulated problems. With reuse the
      // prep grid runs over UA + UB unique operands instead of ne problems.
      GGEMM_NVTX_RANGE(nv_prep, "prep");
      const int r_prep = reuse ? oz2_prep_grouped(dpr, nprep, pmax_m, pmax_n, h->stream)
                               : oz2_prep_grouped(doz, ne, max_m, max_n, h->stream);
      GGEMM_NVTX_END(nv_prep);
      if (r_prep) r = GGEMM_ERR_NUMERICS;
      else {
        ++nl;
        GGEMM_NVTX_RANGE(nv_i8, "i8_gemm");
        const int r_i8 = i8_grouped_gemm_launch(di8, (int)n_i8, db + L.off_sched, grp_tiles, h->stream);
        GGEMM_NVTX_END(nv_i8);
        if (r_i8)
          r = GGEMM_ERR_I8GEMM;
        else {
          if (grp_tiles > 0) ++nl;
          GGEMM_NVTX_RANGE(nv_rec, "reconstruct");
          if (oz2_reconstruct_grouped(doz, ne, max_m, max_n, h->stream)) r = GGEMM_ERR_NUMERICS;
          else ++nl;
        }
      }
    }
  }
  // Native FP64 for k==0 / K too large / tiny / unprofitable (one grouped call if >= 2).
  if (!r && nn > 0) {
    GGEMM_NVTX_RANGE(nv_cublas, "cublas");
    r = run_native(h, probs, h->nat_idx, nn, nat_grouped, nat_grouped ? db + L.off_ptr : nullptr, &nl);
  }
  int r2 = release_slot(h, slot);
  return r ? r : r2;
}

// --------------------------------------------------------------------------------------
extern "C" int ggemm_grouped_dgemm(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                                   const ggemm_problem_t* problems, int num_problems,
                                   ggemm_stats_t* stats) {
  ggemm_stats_t local;
  ggemm_stats_t* st = stats ? stats : &local;
  std::memset(st, 0, sizeof(*st));
  st->num_problems = num_problems;
  st->num_repaired = -1;
  if (!h) return GGEMM_ERR_INVALID;
  int r = ggemm_validate(h, problems, num_problems);
  if (r) return r;

#if GGEMM_NVTX
  static const char* const kNvtxName[] = {
      "ggemm:native_loop", "ggemm:native_grouped", "ggemm:emu_loop", "ggemm:emu_grouped",
      "ggemm:auto", "ggemm:emu_grouped_reuse", "ggemm:emu_loop_batched"};
  GGEMM_NVTX_RANGE(nv_call, ((unsigned)path < sizeof kNvtxName / sizeof kNvtxName[0])
                                ? kNvtxName[path] : "ggemm:invalid");
#endif
  static const ggemm_config_t kDefault = {0, 1e-15, 1};
  if (!cfg) cfg = &kDefault;
  if (cfg->num_moduli < 0 || cfg->num_moduli == 1 || cfg->num_moduli > GGEMM_MAX_MODULI)
    return GGEMM_ERR_INVALID;

  switch (path) {
    case GGEMM_PATH_NATIVE_LOOP:    return path_native(h, false, problems, num_problems, st);
    case GGEMM_PATH_NATIVE_GROUPED: return path_native(h, true, problems, num_problems, st);
    case GGEMM_PATH_EMU_LOOP:
    case GGEMM_PATH_EMU_GROUPED:
    case GGEMM_PATH_AUTO:
    case GGEMM_PATH_EMU_GROUPED_REUSE:
    case GGEMM_PATH_EMU_LOOP_BATCHED: {
      // Repair counter: agent-N's counters are sticky and reading them synchronizes the
      // stream, so only on explicit request (never in timed runs): reset before, read after.
      if (h->stats_repair && oz2_read_counters(nullptr, nullptr, 1, h->stream)) return GGEMM_ERR_NUMERICS;
      r = path_emu(h, path, cfg, problems, num_problems, st);
      if (!r && h->stats_repair) {
        unsigned long long rep = 0;
        if (oz2_read_counters(nullptr, &rep, 1, h->stream)) return GGEMM_ERR_NUMERICS;
        st->num_repaired = (long long)rep;
      }
      return r;
    }
    default:                        return GGEMM_ERR_INVALID;
  }
}
