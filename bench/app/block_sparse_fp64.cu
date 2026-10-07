// bench/app/block_sparse_fp64.cu -- application-level benchmark: FP64 block-sparse (BSR, variable
// block sizes) matrix x dense multi-vector product  Y = A X  (owner: agent-E).
// All GEMMs go through ggemm_grouped_dgemm(); no other GEMM library is called.
//
// ---------------------------------------------------------------------------------------------
// Scenario: atom-blocked sparse matrices of linear-scaling electronic-structure codes
//   In Gaussian-basis DFT / Hartree-Fock (e.g. the DBCSR library used by CP2K) the Kohn-Sham,
//   overlap and density matrices are stored block-sparse with one block per atom pair: block
//   (I,J) has size nbf(I) x nbf(J), where nbf is the number of basis functions of the atom's
//   element, and it is kept only if the atoms are within a cutoff radius (matrix elements decay
//   exponentially with distance). Products such as  F * C_occ  (Fock matrix times occupied MO
//   coefficients, nrhs = #occupied orbitals or a chunk of them) are exactly "a huge number of
//   small dense GEMMs of non-uniform size, then a reduction", and they are done in FP64
//   because SCF energies are converged to ~1e-10 Hartree and smaller errors feed back into the
//   next SCF iteration. Same structure arises in FEM with mixed element orders (block size =
//   dofs per node/element) and in block-Jacobi / multigrid smoothers.
//
// Synthetic but physically shaped instance (fixed seed, fully deterministic):
//   * N_atoms atoms uniform in a periodic cube, number density rho (default 0.1 / A^3,
//     roughly that of liquid water / organic matter including H).
//   * Elements: 50% H, 35% C, 15% O. nbf per element from the spherical Dunning basis:
//       --basis dz : cc-pVDZ  H 5,  C 14, O 14
//       --basis tz : cc-pVTZ  H 14, C 30, O 30
//       --basis qz : cc-pVQZ  H 30, C 55, O 55
//   * Block (I,J) present iff minimum-image distance r_IJ <= cutoff (default 6 A).
//   * Entries: Ozaki-style phi distribution, block scaled by exp(-r_IJ / 2 A) so the matrix has
//     the distance-decay dynamic range real KS/density matrices have.
//   * X: N x nrhs dense, phi distribution.
// The sparsity pattern is static (as across SCF iterations), so descriptor construction is done
// once outside the timed region; ggemm's own descriptor upload stays inside it.
//
// Two formulations (--mode block|panel|both):
//   block : one grouped-GEMM problem per stored block:  T_b (r_I x nrhs) = A_IJ (r_I x c_J) X_J
//           (B operand is a row slice of X, ldb = N: no gather), then a deterministic segmented
//           reduction  Y_I = sum_J T_b.   Stages: gemm, reduce.  Problems = #blocks (tiny m,k).
//   panel : one problem per block row. Because blocks are column-major with ld = r_I and stored
//           consecutively per block row, A_I = [A_IJ1 A_IJ2 ...] is already an r_I x K_I
//           column-major panel. X rows of the row's block columns are gathered into a packed
//           K_I x nrhs panel, then  Y_I = A_I Xp_I  directly (reduction happens inside the
//           GEMM's K loop).  Stages: gather, gemm.  Problems = #block rows (large K).
// Both are reported for all 5 ggemm paths; same data, same stream, cudaEvent timing; all
// allocation happens before the timed region.
//
// Correctness: compared with the same formulation run through NATIVE_LOOP:
//   err_cw = max |Y - Y_ref| / (|A| |X|)   (den computed natively from |A|, |X|, same formulation)
//   err_over_bound = err_cw / (2 gamma_n),  n = max_I K_I + max_I (#blocks in row I) + 1
//   (both path and reference obey gamma_n * den; <= 1 means FP64-level). Diagnostic, not proof.
//
// Usage: block_sparse_fp64 [--atoms 1024] [--density 0.1] [--cutoff 6] [--basis dz|tz|qz]
//                          [--nrhs 128] [--mode both] [--phi 0.5] [--seed 4321] [--moduli 0]
//                          [--paths all] [--reps 5] [--warmup 2] [--csv block_sparse_fp64.csv] [--append]
// ---------------------------------------------------------------------------------------------
#include "app_common.h"

// scale every element of block b by sc[b]; one CUDA block per BSR block (grid-stride over blocks)
__global__ void k_block_scale(double* vals, const size_t* voff, const int* vlen, const double* sc, int nb) {
  for (int b = blockIdx.x; b < nb; b += gridDim.x)
    for (int i = threadIdx.x; i < vlen[b]; i += blockDim.x) vals[voff[b] + i] *= sc[b];
}
// block mode: Y[row, c] = sum_{b in row I} T[toff_b + loc + c * r_I]
__global__ void k_reduce(int N, int nrhs, const int* rowI, const int* rowloc, const int* bsz,
                         const int* rowptr, const size_t* toff, const double* Tb, double* Y) {
  size_t n = (size_t)N * nrhs;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
    int row = (int)(idx % N); size_t c = idx / N;
    int I = rowI[row], loc = rowloc[row], r = bsz[I];
    double acc = 0;
    for (int b = rowptr[I]; b < rowptr[I + 1]; ++b) acc += Tb[toff[b] + loc + c * r];
    Y[idx] = acc;
  }
}
// panel mode: Xp[poff_I + p + c * K_I] = X[gsrc + c * N] for every packed row
__global__ void k_gather_panel(int N, int nrhs, size_t total_rows, const int* prow_I, const int* prow_p,
                               const int* prow_src, const size_t* poff, const int* Kp, const double* X, double* Xp) {
  size_t n = total_rows * nrhs;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
    size_t q = idx % total_rows, c = idx / total_rows;
    int I = prow_I[q];
    Xp[poff[I] + prow_p[q] + c * (size_t)Kp[I]] = X[prow_src[q] + c * (size_t)N];
  }
}

template <class T> static T* dcopy(const std::vector<T>& v) {
  T* p; APP_CUDA(cudaMalloc(&p, std::max<size_t>(v.size(), 1) * sizeof(T)));
  if (!v.empty()) APP_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

int main(int argc, char** argv) {
  int atoms = 1024, nrhs = 128, reps = 5, warmup = 2, nmod = 0;
  double rho = 0.1, cutoff = 6.0, phi = 0.5; uint64_t seed = 4321;
  std::string basis = "dz", mode_s = "both", paths_s = "all", csv = "block_sparse_fp64.csv";
  bool append = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--atoms") atoms = atoi(nxt().c_str());
    else if (a == "--density") rho = atof(nxt().c_str());
    else if (a == "--cutoff") cutoff = atof(nxt().c_str());
    else if (a == "--basis") basis = nxt();
    else if (a == "--nrhs") nrhs = atoi(nxt().c_str());
    else if (a == "--mode") mode_s = nxt();
    else if (a == "--phi") phi = atof(nxt().c_str());
    else if (a == "--seed") seed = strtoull(nxt().c_str(), nullptr, 10);
    else if (a == "--moduli") nmod = atoi(nxt().c_str());
    else if (a == "--paths") paths_s = nxt();
    else if (a == "--reps") reps = atoi(nxt().c_str());
    else if (a == "--warmup") warmup = atoi(nxt().c_str());
    else if (a == "--csv") csv = nxt();
    else if (a == "--append") append = true;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (atoms <= 0 || nrhs <= 0) { fprintf(stderr, "--atoms and --nrhs must be > 0\n"); return 2; }
  int nbf[3];   // H, C, O
  if (basis == "dz") { nbf[0] = 5; nbf[1] = 14; nbf[2] = 14; }
  else if (basis == "tz") { nbf[0] = 14; nbf[1] = 30; nbf[2] = 30; }
  else if (basis == "qz") { nbf[0] = 30; nbf[1] = 55; nbf[2] = 55; }
  else { fprintf(stderr, "--basis dz|tz|qz\n"); return 2; }
  std::vector<std::string> modes;
  if (mode_s == "both" || mode_s == "block") modes.push_back("block");
  if (mode_s == "both" || mode_s == "panel") modes.push_back("panel");
  auto paths = app_parse_paths(paths_s);
  const std::string envs = app_env_str();
  std::string gpu = app_gpu_name();

  // ------------------------------------------------ structure (host, deterministic)
  SplitMix64 rng(seed);
  const double Lbox = std::cbrt(atoms / rho);
  std::vector<double> pos(3 * (size_t)atoms);
  std::vector<int> bsz(atoms), rstart(atoms + 1, 0);
  for (int a = 0; a < atoms; ++a) {
    for (int k = 0; k < 3; ++k) pos[3 * a + k] = rng.uniform() * Lbox;
    double u = rng.uniform();
    bsz[a] = nbf[u < 0.50 ? 0 : (u < 0.85 ? 1 : 2)];
    rstart[a + 1] = rstart[a] + bsz[a];
  }
  const int N = rstart[atoms];
  std::vector<int> rowptr(atoms + 1, 0), bcol, Kp(atoms, 0);
  std::vector<double> bscale;
  for (int I = 0; I < atoms; ++I) {
    for (int J = 0; J < atoms; ++J) {
      double r2 = 0;
      for (int k = 0; k < 3; ++k) {
        double dx = pos[3 * I + k] - pos[3 * J + k];
        dx -= Lbox * std::nearbyint(dx / Lbox);
        r2 += dx * dx;
      }
      double r = std::sqrt(r2);
      if (r <= cutoff) { bcol.push_back(J); bscale.push_back(std::exp(-r / 2.0)); Kp[I] += bsz[J]; }
    }
    rowptr[I + 1] = (int)bcol.size();
  }
  const int nb = (int)bcol.size();
  std::vector<size_t> voff(nb + 1, 0), toff(nb + 1, 0), poff(atoms + 1, 0);
  std::vector<int> vlen(nb);
  int maxK = 0, maxBlkRow = 0;
  for (int I = 0; I < atoms; ++I) {
    for (int b = rowptr[I]; b < rowptr[I + 1]; ++b) {
      vlen[b] = bsz[I] * bsz[bcol[b]];
      voff[b + 1] = voff[b] + vlen[b];
      toff[b + 1] = toff[b] + (size_t)bsz[I] * nrhs;
    }
    poff[I + 1] = poff[I] + (size_t)Kp[I] * nrhs;
    maxK = std::max(maxK, Kp[I]); maxBlkRow = std::max(maxBlkRow, rowptr[I + 1] - rowptr[I]);
  }
  const size_t nnz = voff[nb], nT = toff[nb], nXp = poff[atoms], packed_rows = nXp / nrhs;
  std::vector<int> rowI(N), rowloc(N), prow_I(packed_rows), prow_p(packed_rows), prow_src(packed_rows);
  for (int I = 0; I < atoms; ++I) for (int l = 0; l < bsz[I]; ++l) { rowI[rstart[I] + l] = I; rowloc[rstart[I] + l] = l; }
  {
    size_t q = 0;
    for (int I = 0; I < atoms; ++I) {
      int p = 0;
      for (int b = rowptr[I]; b < rowptr[I + 1]; ++b)
        for (int l = 0; l < bsz[bcol[b]]; ++l, ++q, ++p) { prow_I[q] = I; prow_p[q] = p; prow_src[q] = rstart[bcol[b]] + l; }
    }
  }
  const double flops = 2.0 * (double)nnz * nrhs;
  fprintf(stderr, "device: %s  atoms=%d basis=%s N=%d blocks=%d (%.1f/row) nnz=%zu (%.3f%%) maxK=%d nrhs=%d\n",
          gpu.c_str(), atoms, basis.c_str(), N, nb, (double)nb / atoms, nnz, 100.0 * nnz / ((double)N * N), maxK, nrhs);

  // ------------------------------------------------ device data (outside every timed region)
  cudaStream_t s; APP_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  auto dalloc = [](size_t n) { double* p; APP_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(double))); return p; };
  double *Av = dalloc(nnz), *X = dalloc((size_t)N * nrhs), *Y = dalloc((size_t)N * nrhs), *Tb = dalloc(nT), *Xp = dalloc(nXp);
  double *aAv = dalloc(nnz), *aX = dalloc((size_t)N * nrhs), *Yref = dalloc((size_t)N * nrhs), *den = dalloc((size_t)N * nrhs);
  size_t* d_voff = dcopy(voff); int* d_vlen = dcopy(vlen); double* d_bscale = dcopy(bscale);
  int *d_rowI = dcopy(rowI), *d_rowloc = dcopy(rowloc), *d_bsz = dcopy(bsz), *d_rowptr = dcopy(rowptr);
  size_t *d_toff = dcopy(toff), *d_poff = dcopy(poff);
  int *d_pI = dcopy(prow_I), *d_pp = dcopy(prow_p), *d_psrc = dcopy(prow_src), *d_Kp = dcopy(Kp);
  APP_CUDA(cudaDeviceSynchronize());   // legacy-stream copies complete before use on non-blocking s
  unsigned long long* d_u; double* d_s;
  APP_CUDA(cudaMalloc(&d_u, 2 * sizeof(unsigned long long))); APP_CUDA(cudaMalloc(&d_s, 2 * sizeof(double)));
  cudaEvent_t ev[3]; for (auto& e : ev) APP_CUDA(cudaEventCreate(&e));

  fill_ozaki_device(Av, nnz, phi, seed ^ 0xA1, s);
  fill_ozaki_device(X, (size_t)N * nrhs, phi, seed ^ 0xB2, s);
  if (nb) k_block_scale<<<std::min(nb, 65535), 128, 0, s>>>(Av, d_voff, d_vlen, d_bscale, nb);
  app_abs(Av, aAv, nnz, s); app_abs(X, aX, (size_t)N * nrhs, s);
  APP_CUDA(cudaStreamSynchronize(s));

  // descriptors for a given (A values, X, output) set; built once per mode (static pattern)
  auto make_probs = [&](const std::string& mode, const double* Ap, const double* Xs, double* Xpk, double* Tout,
                        double* Yout) {
    std::vector<ggemm_problem_t> v;
    if (mode == "block") {
      v.reserve(nb);
      for (int I = 0; I < atoms; ++I)
        for (int b = rowptr[I]; b < rowptr[I + 1]; ++b)
          v.push_back({bsz[I], nrhs, bsz[bcol[b]], Ap + voff[b], bsz[I], Xs + rstart[bcol[b]], N,
                       Tout + toff[b], bsz[I], 1.0, 0.0});
    } else {
      v.reserve(atoms);
      for (int I = 0; I < atoms; ++I)
        v.push_back({bsz[I], nrhs, Kp[I], Ap + voff[rowptr[I]], bsz[I], Xpk + poff[I], std::max(Kp[I], 1),
                     Yout + rstart[I], N, 1.0, 0.0});
    }
    return v;
  };

  FILE* fo = fopen(csv.c_str(), append ? "a" : "w");
  if (!fo) { perror(csv.c_str()); return 1; }
  fseek(fo, 0, SEEK_END);
  if (ftell(fo) == 0)
    fprintf(fo, "app,mode,path,atoms,basis,N,nrhs,blocks,G,nnz,max_K,cutoff,density,phi,seed,num_moduli_cfg,flops,"
                "total_ms_median,total_ms_mean,wall_ms_median,pre_ms,gemm_ms,post_ms,pre_pct,gemm_pct,post_pct,"
                "eff_GFLOPS,launches,num_fallback,moduli_min,moduli_max,err_cw,err_over_bound,rel_fro,nonfinite,"
                "reps,gpu,env,status\n");
  const double bnd = 2.0 * app_gamma((long long)maxK + maxBlkRow + 1);

  for (const auto& mode : modes) {
    const bool blk = (mode == "block");
    auto probs = make_probs(mode, Av, X, Xp, Tb, Y);
    auto aprobs = make_probs(mode, aAv, aX, Xp, Tb, den);   // |A||X| through the same formulation
    const int G = (int)probs.size();
    // element budgets: sum m*k = nnz and sum k*n = (packed rows)*nrhs in both modes
    const size_t mA = nnz + 1, mB2 = nXp + 1, mC = (blk ? nT : (size_t)N * nrhs) + 1;

    // one application step (pre stage, grouped GEMM, post stage)
    auto step = [&](ggemm_handle_t h, ggemm_path_t p, const ggemm_config_t& cfg, const std::vector<ggemm_problem_t>& pr,
                    const double* Xs, double* Yout, cudaEvent_t* e, ggemm_stats_t* st) -> int {
      if (e) APP_CUDA(cudaEventRecord(e[0], s));
      if (!blk && packed_rows)
        k_gather_panel<<<app_grid(nXp), 256, 0, s>>>(N, nrhs, packed_rows, d_pI, d_pp, d_psrc, d_poff, d_Kp, Xs, Xp);
      if (e) APP_CUDA(cudaEventRecord(e[1], s));
      int rc = ggemm_grouped_dgemm(h, p, &cfg, pr.data(), (int)pr.size(), st);
      if (e) APP_CUDA(cudaEventRecord(e[2], s));
      if (blk)
        k_reduce<<<app_grid((size_t)N * nrhs), 256, 0, s>>>(N, nrhs, d_rowI, d_rowloc, d_bsz, d_rowptr, d_toff, Tb, Yout);
      return rc;
    };
    cudaEvent_t evs[4] = {ev[0], ev[1], ev[2], nullptr};
    cudaEvent_t e_end; APP_CUDA(cudaEventCreate(&e_end)); evs[3] = e_end;

    // reference + denominator with NATIVE_LOOP
    int ref_ok = 0;
    {
      ggemm_handle_t h = nullptr; ggemm_stats_t st;
      ggemm_config_t cfg = app_cfg(GGEMM_PATH_NATIVE_LOOP, 0);
      if (ggemm_create(&h, std::max(G, 1), mA, mB2, mC, s) == 0 && h) {
        int rc = step(h, GGEMM_PATH_NATIVE_LOOP, cfg, probs, X, Y, nullptr, &st);
        APP_CUDA(cudaMemcpyAsync(Yref, Y, (size_t)N * nrhs * sizeof(double), cudaMemcpyDeviceToDevice, s));
        rc |= step(h, GGEMM_PATH_NATIVE_LOOP, cfg, aprobs, aX, den, nullptr, &st);
        APP_CUDA(cudaStreamSynchronize(s));
        ref_ok = (rc == 0);
        ggemm_destroy(h);
      }
    }
    if (!ref_ok) fprintf(stderr, "WARNING: [%s] NATIVE_LOOP reference failed; err columns will be NaN\n", mode.c_str());

    for (auto p : paths) {
      ggemm_config_t cfg = app_cfg(p, nmod);
      ggemm_handle_t h = nullptr; ggemm_stats_t st{};
      const char* status = "ok";
      if (ggemm_create(&h, std::max(G, 1), mA, mB2, mC, s) != 0 || !h) status = "create_failed";
      int rc = 0;
      for (int r = 0; h && rc == 0 && r < warmup; ++r) rc = step(h, p, cfg, probs, X, Y, nullptr, &st);
      if (h) APP_CUDA(cudaStreamSynchronize(s));
      std::vector<double> tot, wall, sp[3];
      for (int r = 0; h && rc == 0 && r < reps; ++r) {
        auto w0 = std::chrono::steady_clock::now();
        rc = step(h, p, cfg, probs, X, Y, evs, &st);
        APP_CUDA(cudaEventRecord(e_end, s));
        APP_CUDA(cudaEventSynchronize(e_end));
        auto w1 = std::chrono::steady_clock::now();
        if (rc != 0) break;   // failed iteration is not timed
        float ms;
        APP_CUDA(cudaEventElapsedTime(&ms, evs[0], e_end)); tot.push_back(ms);
        for (int k = 0; k < 3; ++k) { APP_CUDA(cudaEventElapsedTime(&ms, evs[k], evs[k + 1])); sp[k].push_back(ms); }
        wall.push_back(std::chrono::duration<double, std::milli>(w1 - w0).count());
      }
      if (h && rc != 0) status = "call_failed";
      if (cudaGetLastError() != cudaSuccess) status = "cuda_error";
      AppErr er;
      if (h && rc == 0 && ref_ok) {
        fill_const_device(Y, (size_t)N * nrhs, NAN, s);
        if (blk) fill_const_device(Tb, nT, NAN, s);
        if (step(h, p, cfg, probs, X, Y, nullptr, &st) == 0) er = app_compare(Y, Yref, den, (size_t)N * nrhs, d_u, d_s, s);
        else status = "check_failed";
      }
      double tmed = app_median(tot), tmean = app_mean(tot), sm[3];
      for (int k = 0; k < 3; ++k) sm[k] = app_mean(sp[k]);
      fprintf(fo, "block_sparse,%s,%s,%d,%s,%d,%d,%d,%d,%zu,%d,%.2f,%.3f,%.2f,%llu,%d,%.6e,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%.6e,%.6e,%.6e,%llu,%d,%s,%s,%s\n",
              mode.c_str(), app_path_name(p), atoms, basis.c_str(), N, nrhs, nb, G, nnz, maxK, cutoff, rho, phi,
              (unsigned long long)seed, nmod, flops, tmed, tmean, app_median(wall), sm[0], sm[1], sm[2],
              100.0 * sm[0] / tmean, 100.0 * sm[1] / tmean, 100.0 * sm[2] / tmean, flops / (tmed * 1e-3) / 1e9,
              st.kernel_launches, st.num_fallback, st.moduli_min, st.moduli_max, er.cw, er.cw / bnd, er.rel_fro,
              er.bad, (int)tot.size(), gpu.c_str(), envs.c_str(), status);
      fflush(fo);
      if (!strcmp(status, "cuda_error")) { fprintf(stderr, "CUDA error (sticky) -> abort\n"); fclose(fo); exit(1); }
      fprintf(stderr, "[%s] %-15s G=%-6d total %9.3f ms (%8.1f GFLOPS) pre %4.1f%% gemm %4.1f%% post %4.1f%% "
                      "launches=%d fb=%d err=%.2e (x%.2g bound) %s\n",
              mode.c_str(), app_path_name(p), G, tmed, flops / (tmed * 1e-3) / 1e9, 100.0 * sm[0] / tmean,
              100.0 * sm[1] / tmean, 100.0 * sm[2] / tmean, st.kernel_launches, st.num_fallback, er.cw, er.cw / bnd, status);
      if (h) ggemm_destroy(h);
    }
    cudaEventDestroy(e_end);
  }
  fclose(fo);
  return 0;
}
