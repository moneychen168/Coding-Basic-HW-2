// ggemm_graph.cu -- plan / execute split on top of the EMU_GROUPED / AUTO pipeline (owner: agent-G).
//
// The descriptor build below mirrors path_emu() in ggemm.cu (EMU_GROUPED / AUTO branch) field
// by field, so the graph runs the very same kernels on the very same descriptor contents as a
// plain ggemm_grouped_dgemm call (=> bit-identical results). ggemm.cu keeps those helpers
// static, hence the copy; see the agent-G report for the optional dedup diff.
//
// Why update_pointers patches device descriptors instead of using graph-exec update:
//   the user A/B/C pointers are NOT kernel parameters of any captured node. prep/reconstruct
//   read them from oz2_problem[] and cublasDgemmGroupedBatched from a device pointer array,
//   both inside the plan's device blob; the captured kernel parameters are only blob addresses,
//   which never change. cudaGraphExecKernelNodeSetParams therefore has nothing to update; one
//   O(num) host patch + one small async H2D of the pointer-bearing ranges is enough (no
//   capture, no instantiate). Single exception: exactly one native-fallback problem runs as a
//   plain cublasDgemm whose pointers are opaque cuBLAS kernel arguments; only then we re-capture
//   and apply cudaGraphExecUpdate (same topology -> in-place update, no re-instantiate).
#include "internal.h"
#include "ggemm_graph.h"

#include <algorithm>
#include <cstring>
#include <new>

#define CU(x) do { if ((x) != cudaSuccess) return GGEMM_ERR_CUDA; } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) return GGEMM_ERR_CUBLAS; } while (0)

// Blob layout / argument validation are shared with ggemm.cu (internal.h: ggemm_layout_for,
// ggemm_validate). The graph path never uses operand reuse (num_prep = 0): update_pointers may
// change which problems share an operand, which would change the prep structure of the graph.

struct ggemm_plan_s {
  ggemm_handle_t h;
  int num;
  ggemm_problem_t* probs;          // host copy (pointers patched by update_pointers)
  ggemm_plan_entry_t* entries;
  int *emu_idx, *nat_idx;
  int ne, nn, max_m, max_n, grp_tiles;
  size_t n_i8;
  ggemm_blob_layout L;
  char* h_blob;                    // pinned mirror of d_blob (also the source of update H2Ds)
  char* d_blob;                    // plan-owned descriptors + schedule image + native ptrs
  // plan-owned host metadata for cublasDgemmGroupedBatched (must outlive the graph)
  cublasOperation_t *nat_ta, *nat_tb;
  int *nat_m, *nat_n, *nat_k, *nat_lda, *nat_ldb, *nat_ldc, *nat_gsize;
  double *nat_alpha, *nat_beta;
  cudaStream_t cap;                // private capture stream
  cudaGraph_t graph;
  cudaGraphExec_t exec;            // nullptr when the plan has no work
  cudaEvent_t ev_h, ev_p, ev_upd;
  ggemm_stats_t st;
};

static void free_plan(ggemm_plan_t p) {
  if (!p) return;
  if (p->h) cudaStreamSynchronize(p->h->stream);   // every execute is joined into h->stream
  if (p->exec) cudaGraphExecDestroy(p->exec);
  if (p->graph) cudaGraphDestroy(p->graph);
  if (p->cap) cudaStreamDestroy(p->cap);
  if (p->ev_h) cudaEventDestroy(p->ev_h);
  if (p->ev_p) cudaEventDestroy(p->ev_p);
  if (p->ev_upd) { cudaEventSynchronize(p->ev_upd); cudaEventDestroy(p->ev_upd); }
  cudaFree(p->d_blob);
  if (p->h_blob) cudaFreeHost(p->h_blob);
  delete[] p->probs; delete[] p->entries; delete[] p->emu_idx; delete[] p->nat_idx;
  delete[] p->nat_ta; delete[] p->nat_tb; delete[] p->nat_m; delete[] p->nat_n; delete[] p->nat_k;
  delete[] p->nat_lda; delete[] p->nat_ldb; delete[] p->nat_ldc; delete[] p->nat_gsize;
  delete[] p->nat_alpha; delete[] p->nat_beta;
  delete p;
}

// Enqueue the whole launch sequence on s (only ever called while s is being captured).
static int enqueue(ggemm_plan_t p, cudaStream_t s, int* launches) {
  ggemm_handle_t h = p->h;
  const oz2_problem* doz = reinterpret_cast<const oz2_problem*>(p->d_blob + p->L.off_oz);
  const i8_problem* di8 = reinterpret_cast<const i8_problem*>(p->d_blob + p->L.off_i8);
  if (p->ne > 0) {
    if (oz2_prep_grouped(doz, p->ne, p->max_m, p->max_n, s)) return GGEMM_ERR_NUMERICS;
    ++*launches;
    if (i8_grouped_gemm_launch(di8, (int)p->n_i8, p->d_blob + p->L.off_sched, p->grp_tiles, s))
      return GGEMM_ERR_I8GEMM;
    if (p->grp_tiles > 0) ++*launches;
    if (oz2_reconstruct_grouped(doz, p->ne, p->max_m, p->max_n, s)) return GGEMM_ERR_NUMERICS;
    ++*launches;
  }
  if (p->nn == 1) {
    const ggemm_problem_t& q = p->probs[p->nat_idx[0]];
    CB(cublasDgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, q.m, q.n, q.k, &q.alpha, q.A, q.lda,
                   q.B, q.ldb, &q.beta, q.C, q.ldc));
    ++*launches;
  } else if (p->nn >= 2) {
    void* const* base = reinterpret_cast<void* const*>(p->d_blob + p->L.off_ptr);
    const int c = p->nn;
    CB(cublasDgemmGroupedBatched(h->cublas, p->nat_ta, p->nat_tb, p->nat_m, p->nat_n, p->nat_k,
                                 p->nat_alpha, reinterpret_cast<const double* const*>(base),
                                 p->nat_lda, reinterpret_cast<const double* const*>(base + c),
                                 p->nat_ldb, p->nat_beta, reinterpret_cast<double* const*>(base + 2 * c),
                                 p->nat_ldc, c, p->nat_gsize));
    ++*launches;
  }
  return GGEMM_OK;
}

// Capture enqueue() on the private stream. cuBLAS is temporarily bound to it; cublasSetStream
// resets the workspace to cuBLAS's default pool, so the handle's fixed workspace is re-attached
// (otherwise cuBLAS could allocate during capture) and restored afterwards.
static int capture(ggemm_plan_t p, cudaGraph_t* out, int* launches) {
  ggemm_handle_t h = p->h;
  *out = nullptr;
  const bool uses_cublas = p->nn > 0;
  // Relaxed: agent-K's first launch on a device does cudaFuncSetAttribute / occupancy queries.
  CU(cudaStreamBeginCapture(p->cap, cudaStreamCaptureModeRelaxed));
  int r = GGEMM_OK;
  if (uses_cublas && (cublasSetStream(h->cublas, p->cap) != CUBLAS_STATUS_SUCCESS ||
                      cublasSetWorkspace(h->cublas, h->cublas_ws, h->cublas_ws_bytes) !=
                          CUBLAS_STATUS_SUCCESS))
    r = GGEMM_ERR_CUBLAS;
  if (!r) r = enqueue(p, p->cap, launches);
  cudaGraph_t g = nullptr;
  cudaError_t ce = cudaStreamEndCapture(p->cap, &g);   // always end the capture
  if (uses_cublas && (cublasSetStream(h->cublas, h->stream) != CUBLAS_STATUS_SUCCESS ||
                      cublasSetWorkspace(h->cublas, h->cublas_ws, h->cublas_ws_bytes) !=
                          CUBLAS_STATUS_SUCCESS) && !r)
    r = GGEMM_ERR_CUBLAS;
  if (!r && ce != cudaSuccess) r = GGEMM_ERR_CUDA;
  if (r) { if (g) cudaGraphDestroy(g); return r; }
  *out = g;
  return GGEMM_OK;
}

static void write_ptrs(ggemm_plan_t p) {
  if (p->nn < 2) return;
  const void** P = reinterpret_cast<const void**>(p->h_blob + p->L.off_ptr);
  const int c = p->nn;
  for (int i = 0; i < c; ++i) {
    const ggemm_problem_t& q = p->probs[p->nat_idx[i]];
    P[i] = q.A; P[c + i] = q.B; P[2 * c + i] = q.C;
  }
}

static int build(ggemm_plan_t p, ggemm_path_t path, const ggemm_config_t* cfg) {
  ggemm_handle_t h = p->h;
  const int num = p->num;
  int r = ggemm_plan(p->probs, num, cfg, path, &h->dm, p->entries);
  if (r) return r;

  // ---- partition + workspace offsets (identical to ggemm.cu path_emu)
  size_t SA = 0, SB = 0, SC = 0, rows = 0, cols = 0;
  int nmax = 0;
  p->ne = p->nn = 0; p->n_i8 = 0;
  for (int g = 0; g < num; ++g) {
    const ggemm_plan_entry_t& e = p->entries[g];
    if (e.route == GGEMM_ROUTE_EMPTY) continue;
    if (e.route != GGEMM_ROUTE_EMU) { p->nat_idx[p->nn++] = g; continue; }
    const ggemm_problem_t& q = p->probs[g];
    const size_t kp = ggemm_align_up((size_t)q.k, GGEMM_KP_ALIGN);
    p->emu_idx[p->ne++] = g;
    SA += ggemm_align_up((size_t)q.m * kp, GGEMM_BUF_ALIGN);
    SB += ggemm_align_up((size_t)q.n * kp, GGEMM_BUF_ALIGN);
    SC += ggemm_align_up((size_t)q.m * q.n, GGEMM_BUF_ALIGN / 4);
    rows += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * q.m, 64);
    cols += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * q.n, 64);
    p->n_i8 += oz2_num_planes(e.num_moduli);   // residue planes + agent-N abs plane
    nmax = std::max(nmax, e.num_moduli);
  }
  const size_t pmax = nmax ? (size_t)oz2_num_planes(nmax) : 0;
  if (pmax * SA > h->A8_bytes || pmax * SB > h->B8_bytes ||
      pmax * SC * sizeof(int32_t) > h->C32_bytes || rows > h->row_shift_elems ||
      cols > h->col_shift_elems)
    return GGEMM_ERR_WORKSPACE;
  if (p->ne > 65535) return GGEMM_ERR_INVALID;

  ggemm_stats_t& st = p->st;
  st.num_problems = num;
  st.num_fallback = 0;
  for (int i = 0; i < p->nn; ++i) st.num_fallback += (p->entries[p->nat_idx[i]].route != GGEMM_ROUTE_K_ZERO);
  st.moduli_min = p->ne ? GGEMM_MAX_MODULI : 0;
  st.moduli_max = nmax;
  for (int j = 0; j < p->ne; ++j) st.moduli_min = std::min(st.moduli_min, p->entries[p->emu_idx[j]].num_moduli);
  st.reuse_enabled = 0;                 // graph path: no operand reuse (see top of file)
  st.num_unique_A = st.num_unique_B = p->ne;
  st.num_promoted = 0;
  st.num_repaired = -1;                 // never read here (would synchronize)

  // ---- plan-owned blob (allocation is fine here: make_plan is setup, never timed)
  const size_t sched_bytes = p->ne == 0 ? 0 : i8_grouped_gemm_sched_ws_bytes((int)p->n_i8, 0);
  p->L = ggemm_layout_for(p->ne, 0, p->n_i8, sched_bytes, p->nn >= 2 ? p->nn : 0);
  const size_t bytes = ggemm_align_up(std::max<size_t>(p->L.total, 1), GGEMM_BUF_ALIGN);
  CU(cudaMalloc(&p->d_blob, bytes));
  CU(cudaHostAlloc(&p->h_blob, bytes, cudaHostAllocDefault));
  std::memset(p->h_blob, 0, bytes);

  char* hb = p->h_blob;
  oz2_problem* hoz = reinterpret_cast<oz2_problem*>(hb + p->L.off_oz);
  i8_problem* hi8 = reinterpret_cast<i8_problem*>(hb + p->L.off_i8);
  p->max_m = p->max_n = 0;
  size_t oa = 0, ob = 0, oc = 0, orow = 0, ocol = 0, q8 = 0;
  for (int j = 0; j < p->ne; ++j) {
    const int g = p->emu_idx[j];
    const ggemm_problem_t& q = p->probs[g];
    const ggemm_plan_entry_t& e = p->entries[g];
    const size_t kp = ggemm_align_up((size_t)q.k, GGEMM_KP_ALIGN);
    oz2_problem& o = hoz[j];
    o.m = q.m; o.n = q.n; o.k = q.k; o.kp = (int)kp; o.num_moduli = e.num_moduli;
    o.lda = q.lda; o.ldb = q.ldb; o.ldc = q.ldc;
    o.A = q.A; o.B = q.B; o.C = q.C; o.alpha = q.alpha; o.beta = q.beta;
    o.A8 = h->A8 + oa;  o.a8_mod_stride = SA;
    o.B8 = h->B8 + ob;  o.b8_mod_stride = SB;
    o.C32 = h->C32 + oc; o.c32_mod_stride = SC;
    o.row_shift = h->row_shift + orow; o.col_shift = h->col_shift + ocol;
    for (int i = 0; i < oz2_num_planes(e.num_moduli); ++i, ++q8) {
      i8_problem& d = hi8[q8];
      d.m = q.m; d.n = q.n; d.kp = (int)kp;
      d.A = o.A8 + (size_t)i * SA; d.B = o.B8 + (size_t)i * SB; d.C = o.C32 + (size_t)i * SC;
      d.tile_cfg = e.tile_cfg;
    }
    oa += ggemm_align_up((size_t)q.m * kp, GGEMM_BUF_ALIGN);
    ob += ggemm_align_up((size_t)q.n * kp, GGEMM_BUF_ALIGN);
    oc += ggemm_align_up((size_t)q.m * q.n, GGEMM_BUF_ALIGN / 4);
    orow += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * q.m, 64);
    ocol += ggemm_align_up((size_t)OZ2_ROW_WS_PER_ROW * q.n, 64);
    p->max_m = std::max(p->max_m, q.m); p->max_n = std::max(p->max_n, q.n);
  }
  p->grp_tiles = 0;
  if (p->ne > 0 && i8_grouped_gemm_build_schedule(hi8, (int)p->n_i8, hb + p->L.off_sched,
                                                  sched_bytes, &p->grp_tiles))
    return GGEMM_ERR_I8GEMM;

  // native metadata (plan-owned, so the captured cuBLAS call never references handle scratch)
  for (int i = 0; i < p->nn; ++i) {
    const ggemm_problem_t& q = p->probs[p->nat_idx[i]];
    p->nat_ta[i] = CUBLAS_OP_N; p->nat_tb[i] = CUBLAS_OP_N;
    p->nat_m[i] = q.m; p->nat_n[i] = q.n; p->nat_k[i] = q.k;
    p->nat_lda[i] = q.lda; p->nat_ldb[i] = q.ldb; p->nat_ldc[i] = q.ldc;
    p->nat_alpha[i] = q.alpha; p->nat_beta[i] = q.beta; p->nat_gsize[i] = 1;
  }
  write_ptrs(p);

  // ---- the one H2D, done here (not inside the graph). Synchronous: setup only.
  if (p->L.total) CU(cudaMemcpyAsync(p->d_blob, hb, p->L.total, cudaMemcpyHostToDevice, h->stream));
  CU(cudaEventRecord(p->ev_upd, h->stream));
  CU(cudaStreamSynchronize(h->stream));

  // ---- capture + instantiate
  if (p->ne == 0 && p->nn == 0) return GGEMM_OK;   // nothing to run: exec stays null
  int launches = 0;
  if ((r = capture(p, &p->graph, &launches))) return r;
  st.kernel_launches = launches;
  CU(cudaGraphInstantiate(&p->exec, p->graph, 0));
  // Upload once now so the first execute does not pay the graph upload.
  CU(cudaGraphUpload(p->exec, h->stream));
  CU(cudaStreamSynchronize(h->stream));
  return GGEMM_OK;
}

extern "C" ggemm_plan_t ggemm_make_plan_ex(ggemm_handle_t h, ggemm_path_t path,
                                           const ggemm_config_t* cfg,
                                           const ggemm_problem_t* problems, int num, int* err) {
  int dummy; int& e = err ? *err : dummy;
  e = GGEMM_ERR_INVALID;
  if (!h || (path != GGEMM_PATH_EMU_GROUPED && path != GGEMM_PATH_AUTO)) return nullptr;
  if ((e = ggemm_validate(h, problems, num))) return nullptr;
  static const ggemm_config_t kDefault = {0, 1e-15, 1};   // == ggemm.cu default
  if (!cfg) cfg = &kDefault;
  if (cfg->num_moduli < 0 || cfg->num_moduli == 1 || cfg->num_moduli > GGEMM_MAX_MODULI) {
    e = GGEMM_ERR_INVALID; return nullptr;
  }

  ggemm_plan_t p = new (std::nothrow) ggemm_plan_s();   // value-initialised (all null / 0)
  if (!p) { e = GGEMM_ERR_ALLOC; return nullptr; }
  p->h = h; p->num = num;
  const size_t P = (size_t)std::max(num, 1);
  p->probs = new (std::nothrow) ggemm_problem_t[P];
  p->entries = new (std::nothrow) ggemm_plan_entry_t[P];
  p->emu_idx = new (std::nothrow) int[P]; p->nat_idx = new (std::nothrow) int[P];
  p->nat_ta = new (std::nothrow) cublasOperation_t[P]; p->nat_tb = new (std::nothrow) cublasOperation_t[P];
  p->nat_m = new (std::nothrow) int[P]; p->nat_n = new (std::nothrow) int[P];
  p->nat_k = new (std::nothrow) int[P]; p->nat_lda = new (std::nothrow) int[P];
  p->nat_ldb = new (std::nothrow) int[P]; p->nat_ldc = new (std::nothrow) int[P];
  p->nat_gsize = new (std::nothrow) int[P];
  p->nat_alpha = new (std::nothrow) double[P]; p->nat_beta = new (std::nothrow) double[P];
  if (!p->probs || !p->entries || !p->emu_idx || !p->nat_idx || !p->nat_ta || !p->nat_tb ||
      !p->nat_m || !p->nat_n || !p->nat_k || !p->nat_lda || !p->nat_ldb || !p->nat_ldc ||
      !p->nat_gsize || !p->nat_alpha || !p->nat_beta) {
    free_plan(p); e = GGEMM_ERR_ALLOC; return nullptr;
  }
  if (num > 0) std::memcpy(p->probs, problems, sizeof(ggemm_problem_t) * num);
  if (cudaStreamCreateWithFlags(&p->cap, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&p->ev_h, cudaEventDisableTiming) != cudaSuccess ||
      cudaEventCreateWithFlags(&p->ev_p, cudaEventDisableTiming) != cudaSuccess ||
      cudaEventCreateWithFlags(&p->ev_upd, cudaEventDisableTiming) != cudaSuccess) {
    free_plan(p); e = GGEMM_ERR_CUDA; return nullptr;
  }
  if ((e = build(p, path, cfg))) { free_plan(p); return nullptr; }
  return p;
}

extern "C" ggemm_plan_t ggemm_make_plan(ggemm_handle_t h, ggemm_path_t path,
                                        const ggemm_config_t* cfg,
                                        const ggemm_problem_t* problems, int num) {
  return ggemm_make_plan_ex(h, path, cfg, problems, num, nullptr);
}

extern "C" int ggemm_execute_plan(ggemm_plan_t p, cudaStream_t stream) {
  if (!p) return GGEMM_ERR_INVALID;
  GGEMM_NVTX_RANGE(nv_graph, "ggemm:graph");
  if (!p->exec) return GGEMM_OK;
  cudaStream_t hs = p->h->stream;
  if (stream == hs) {                      // already ordered with every other handle user
    CU(cudaGraphLaunch(p->exec, stream));
    return GGEMM_OK;
  }
  // Join with the handle's stream on both sides: shared scratch (A8/B8/C32/shifts, cuBLAS ws)
  // and the plan's self-resetting LPT counter are then never used by two launches at once.
  CU(cudaEventRecord(p->ev_h, hs));
  CU(cudaStreamWaitEvent(stream, p->ev_h, 0));
  CU(cudaGraphLaunch(p->exec, stream));
  CU(cudaEventRecord(p->ev_p, stream));
  CU(cudaStreamWaitEvent(hs, p->ev_p, 0));
  return GGEMM_OK;
}

extern "C" int ggemm_plan_update_pointers(ggemm_plan_t p, const double* const* A,
                                          const double* const* B, double* const* C) {
  if (!p) return GGEMM_ERR_INVALID;
  // validate first so a bad call leaves the plan untouched
  for (int g = 0; g < p->num; ++g) {
    const ggemm_problem_t& q = p->probs[g];
    if (q.m == 0 || q.n == 0) continue;
    if (C && !C[g]) return GGEMM_ERR_INVALID;
    if (q.k > 0 && ((A && !A[g]) || (B && !B[g]))) return GGEMM_ERR_INVALID;
  }
  for (int g = 0; g < p->num; ++g) {
    if (A) p->probs[g].A = A[g];
    if (B) p->probs[g].B = B[g];
    if (C) p->probs[g].C = C[g];
  }
  ggemm_handle_t h = p->h;
  // The previous patch H2D must have finished reading the pinned mirror before we rewrite it.
  CU(cudaEventSynchronize(p->ev_upd));
  oz2_problem* hoz = reinterpret_cast<oz2_problem*>(p->h_blob + p->L.off_oz);
  for (int j = 0; j < p->ne; ++j) {
    const ggemm_problem_t& q = p->probs[p->emu_idx[j]];
    hoz[j].A = q.A; hoz[j].B = q.B; hoz[j].C = q.C;
  }
  write_ptrs(p);
  // Enqueued on h->stream: every execute is joined into h->stream, so this patch lands after
  // all previously issued launches and before all later ones (which wait on h->stream).
  if (p->ne > 0)
    CU(cudaMemcpyAsync(p->d_blob + p->L.off_oz, p->h_blob + p->L.off_oz,
                       (size_t)p->ne * sizeof(oz2_problem), cudaMemcpyHostToDevice, h->stream));
  if (p->nn >= 2)
    CU(cudaMemcpyAsync(p->d_blob + p->L.off_ptr, p->h_blob + p->L.off_ptr,
                       3 * (size_t)p->nn * sizeof(void*), cudaMemcpyHostToDevice, h->stream));
  CU(cudaEventRecord(p->ev_upd, h->stream));

  if (p->nn == 1) {   // pointers are cuBLAS kernel args: re-capture + in-place exec update
    cudaGraph_t g = nullptr;
    int launches = 0;
    int r = capture(p, &g, &launches);
    if (r) return r;
    cudaGraphExecUpdateResultInfo info;
    if (cudaGraphExecUpdate(p->exec, g, &info) != cudaSuccess) {
      (void)cudaGetLastError();
      // Topology changed (should not happen): re-instantiate after in-flight launches drain.
      if (cudaStreamSynchronize(h->stream) != cudaSuccess) { cudaGraphDestroy(g); return GGEMM_ERR_CUDA; }
      cudaGraphExecDestroy(p->exec); p->exec = nullptr;
      if (cudaGraphInstantiate(&p->exec, g, 0) != cudaSuccess) { cudaGraphDestroy(g); return GGEMM_ERR_CUDA; }
    }
    cudaGraphDestroy(p->graph);
    p->graph = g;
  }
  return GGEMM_OK;
}

extern "C" int ggemm_plan_get_stats(ggemm_plan_t p, ggemm_stats_t* st) {
  if (!p || !st) return GGEMM_ERR_INVALID;
  *st = p->st;
  return GGEMM_OK;
}

extern "C" void ggemm_destroy_plan(ggemm_plan_t p) { free_plan(p); }
