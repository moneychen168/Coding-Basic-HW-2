// ggemm public API (agent-P): workspace management + the 5 execution paths.
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

// --------------------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------------------
#define CU(x) do { if ((x) != cudaSuccess) return GGEMM_ERR_CUDA; } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) return GGEMM_ERR_CUBLAS; } while (0)

static const size_t kCublasWsBytes = 64ull << 20;   // fixed cuBLAS workspace (no lazy alloc in timed calls)
static const int kDescSlots = 2;                    // == size of ggemm_handle_s::desc_done

// Per-call descriptor blob layout (identical in the pinned slot and in the device blob):
//   [oz2_problem x num_emu] [i8_problem x sum(N_g)] [agent-K LPT schedule image(s)]
//   [A ptrs | B ptrs | C ptrs  x num_nat]
struct blob_layout { size_t off_oz, off_i8, off_sched, off_ptr, total; };

static blob_layout layout_for(size_t num_emu, size_t num_i8, size_t sched_bytes, size_t num_nat) {
  blob_layout L;
  L.off_oz    = 0;
  L.off_i8    = ggemm_align_up(L.off_oz + num_emu * sizeof(oz2_problem), GGEMM_BUF_ALIGN);
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

static int validate(ggemm_handle_t h, const ggemm_problem_t* p, int num) {
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

static void write_ptrs(char* hb, const blob_layout& L, const ggemm_problem_t* probs,
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
  const size_t P = (size_t)max_problems, M = GGEMM_MAX_MODULI;

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
  h->row_shift_elems = max_A + P * 64;
  h->col_shift_elems = max_B + P * 64;
  CUA(cudaMalloc(&h->A8, h->A8_bytes));
  CUA(cudaMalloc(&h->B8, h->B8_bytes));
  CUA(cudaMalloc(&h->C32, h->C32_bytes));
  CUA(cudaMalloc(&h->row_shift, h->row_shift_elems * sizeof(int32_t)));
  CUA(cudaMalloc(&h->col_shift, h->col_shift_elems * sizeof(int32_t)));

  // Descriptors: worst case = all problems emulated with MAX moduli (+ schedule image(s):
  // one grouped image, or one image per (prob, mod) for EMU_LOOP), plus any native pointers.
  h->sched_img_bytes_1 = ggemm_align_up(i8_grouped_gemm_sched_ws_bytes(1, 0), GGEMM_BUF_ALIGN);
  const size_t sched_max = std::max(i8_grouped_gemm_sched_ws_bytes((int)(P * M), 0),
                                    P * M * h->sched_img_bytes_1);
  blob_layout Lw = layout_for(P, P * M, sched_max, P);
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
  if (!grouped || cnt < 2) return run_native(h, probs, h->nat_idx, cnt, false, nullptr, &st->kernel_launches);
  int slot, r;
  if ((r = acquire_slot(h, &slot))) return r;
  char* hb = static_cast<char*>(host_slot(h, slot));
  blob_layout L = layout_for(0, 0, 0, cnt);
  write_ptrs(hb, L, probs, h->nat_idx, cnt);
  CU(cudaMemcpyAsync(h->d_desc, hb, L.total, cudaMemcpyHostToDevice, h->stream));
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
  int r = ggemm_plan(probs, num, cfg, path, &h->dm, h->plan);
  if (r) return r;

  // ---- partition + workspace offsets ([mod][prob] slabs; slab stride shared by all problems)
  int ne = 0, nn = 0, nmax = 0;
  size_t SA = 0, SB = 0, SC = 0, rows = 0, cols = 0, n_i8 = 0;
  for (int g = 0; g < num; ++g) {
    const ggemm_plan_entry_t& e = h->plan[g];
    if (e.route == GGEMM_ROUTE_EMPTY) continue;
    if (e.route != GGEMM_ROUTE_EMU) { h->nat_idx[nn++] = g; continue; }
    const ggemm_problem_t& p = probs[g];
    const size_t kp = ggemm_align_up((size_t)p.k, GGEMM_KP_ALIGN);
    h->emu_idx[ne++] = g;
    SA += ggemm_align_up((size_t)p.m * kp, GGEMM_BUF_ALIGN);
    SB += ggemm_align_up((size_t)p.n * kp, GGEMM_BUF_ALIGN);
    SC += ggemm_align_up((size_t)p.m * p.n, GGEMM_BUF_ALIGN / 4);
    rows += ggemm_align_up((size_t)p.m, 64);
    cols += ggemm_align_up((size_t)p.n, 64);
    n_i8 += e.num_moduli;
    nmax = std::max(nmax, e.num_moduli);
  }
  if ((size_t)nmax * SA > h->A8_bytes || (size_t)nmax * SB > h->B8_bytes ||
      (size_t)nmax * SC * sizeof(int32_t) > h->C32_bytes || rows > h->row_shift_elems ||
      cols > h->col_shift_elems)
    return GGEMM_ERR_WORKSPACE;
  if (ne > 65535) return GGEMM_ERR_INVALID;   // agent-N prep/reconstruct grid limit

  // ---- stats. num_fallback counts precision/profit reroutes (K too large, target
  // unreachable, tiny, unprofitable). k==0 problems (C = beta*C, exact) also run on cuBLAS
  // but are not a "fallback": nothing was emulated-or-not. They do count in kernel_launches.
  st->num_fallback = 0;
  for (int i = 0; i < nn; ++i) st->num_fallback += (h->plan[h->nat_idx[i]].route != GGEMM_ROUTE_K_ZERO);
  st->moduli_min = ne ? GGEMM_MAX_MODULI : 0;
  st->moduli_max = nmax;
  for (int j = 0; j < ne; ++j) st->moduli_min = std::min(st->moduli_min, h->plan[h->emu_idx[j]].num_moduli);

  // ---- fill descriptors in the pinned slot (oz2 per problem; i8 per (prob, mod), problem-major)
  int slot;
  if ((r = acquire_slot(h, &slot))) return r;
  char* hb = static_cast<char*>(host_slot(h, slot));
  char* db = static_cast<char*>(h->d_desc);
  const bool nat_grouped = (nn >= 2);
  const bool loop = (path == GGEMM_PATH_EMU_LOOP);
  const size_t sched_bytes = ne == 0 ? 0
      : (loop ? n_i8 * h->sched_img_bytes_1 : i8_grouped_gemm_sched_ws_bytes((int)n_i8, 0));
  blob_layout L = layout_for(ne, n_i8, sched_bytes, nat_grouped ? nn : 0);
  oz2_problem* hoz = reinterpret_cast<oz2_problem*>(hb + L.off_oz);
  i8_problem* hi8 = reinterpret_cast<i8_problem*>(hb + L.off_i8);
  const oz2_problem* doz = reinterpret_cast<const oz2_problem*>(db + L.off_oz);
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
    o.A8 = h->A8 + oa;  o.a8_mod_stride = SA;     // bytes == int8 elements
    o.B8 = h->B8 + ob;  o.b8_mod_stride = SB;
    o.C32 = h->C32 + oc; o.c32_mod_stride = SC;   // int32 elements
    o.row_shift = h->row_shift + orow; o.col_shift = h->col_shift + ocol;
    for (int i = 0; i < e.num_moduli; ++i, ++q) {
      i8_problem& d = hi8[q];
      d.m = p.m; d.n = p.n; d.kp = (int)kp;
      d.A = o.A8 + (size_t)i * SA; d.B = o.B8 + (size_t)i * SB; d.C = o.C32 + (size_t)i * SC;
      d.tile_cfg = e.tile_cfg;
    }
    oa += ggemm_align_up((size_t)p.m * kp, GGEMM_BUF_ALIGN);
    ob += ggemm_align_up((size_t)p.n * kp, GGEMM_BUF_ALIGN);
    oc += ggemm_align_up((size_t)p.m * p.n, GGEMM_BUF_ALIGN / 4);
    orow += ggemm_align_up((size_t)p.m, 64);
    ocol += ggemm_align_up((size_t)p.n, 64);
    max_m = std::max(max_m, p.m); max_n = std::max(max_n, p.n);
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
    } else if (i8_grouped_gemm_build_schedule(hi8, (int)n_i8, hb + L.off_sched, sched_bytes,
                                              &grp_tiles)) {
      release_slot(h, slot); return GGEMM_ERR_I8GEMM;
    }
  }

  // ---- single descriptor H2D (pinned, async). EMU_LOOP also uploads once up front: the
  // baseline is then penalized only by its launch structure, not by extra memcpys.
  if (L.total) CU(cudaMemcpyAsync(db, hb, L.total, cudaMemcpyHostToDevice, h->stream));

  int& nl = st->kernel_launches;
  if (ne > 0) {
    if (loop) {
      // Starter-like baseline (no starter source available to us, so we model the usual
      // single-GEMM emulation): per problem 1 prep launch, then ONE INT8 GEMM launch PER
      // MODULUS (the single-GEMM emulator calls its INT8 GEMM N times), then 1 reconstruct
      // launch => N_g + 2 launches per problem.
      size_t qq = 0;
      for (int j = 0; j < ne; ++j) {
        const ggemm_problem_t& p = probs[h->emu_idx[j]];
        if (oz2_prep_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        ++nl;
        for (int i = 0; i < hoz[j].num_moduli; ++i, ++qq) {
          if (i8_grouped_gemm_launch(di8 + qq, 1, db + L.off_sched + qq * h->sched_img_bytes_1,
                                     h->loop_tiles[qq], h->stream)) {
            r = GGEMM_ERR_I8GEMM; break;
          }
          if (h->loop_tiles[qq] > 0) ++nl;   // K launches nothing when total_tiles == 0
        }
        if (r) break;
        if (oz2_reconstruct_grouped(doz + j, 1, p.m, p.n, h->stream)) { r = GGEMM_ERR_NUMERICS; break; }
        ++nl;
      }
    } else {
      // Fused grouped pipeline: exactly 3 launches for all emulated problems.
      if (oz2_prep_grouped(doz, ne, max_m, max_n, h->stream)) r = GGEMM_ERR_NUMERICS;
      else {
        ++nl;
        if (i8_grouped_gemm_launch(di8, (int)n_i8, db + L.off_sched, grp_tiles, h->stream))
          r = GGEMM_ERR_I8GEMM;
        else {
          if (grp_tiles > 0) ++nl;
          if (oz2_reconstruct_grouped(doz, ne, max_m, max_n, h->stream)) r = GGEMM_ERR_NUMERICS;
          else ++nl;
        }
      }
    }
  }
  // Native FP64 for k==0 / K too large / tiny / unprofitable (one grouped call if >= 2).
  if (!r) r = run_native(h, probs, h->nat_idx, nn, nat_grouped,
                         nat_grouped ? db + L.off_ptr : nullptr, &nl);
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
  if (!h) return GGEMM_ERR_INVALID;
  int r = validate(h, problems, num_problems);
  if (r) return r;

  static const ggemm_config_t kDefault = {0, 1e-15, 1};
  if (!cfg) cfg = &kDefault;
  if (cfg->num_moduli < 0 || cfg->num_moduli == 1 || cfg->num_moduli > GGEMM_MAX_MODULI)
    return GGEMM_ERR_INVALID;

  switch (path) {
    case GGEMM_PATH_NATIVE_LOOP:    return path_native(h, false, problems, num_problems, st);
    case GGEMM_PATH_NATIVE_GROUPED: return path_native(h, true, problems, num_problems, st);
    case GGEMM_PATH_EMU_LOOP:
    case GGEMM_PATH_EMU_GROUPED:
    case GGEMM_PATH_AUTO:           return path_emu(h, path, cfg, problems, num_problems, st);
    default:                        return GGEMM_ERR_INVALID;
  }
}
