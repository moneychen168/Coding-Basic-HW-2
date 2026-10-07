// bench/workloads.h -- shape sets + deterministic data generation for ggemm tests/bench.
// Owner: agent-B. Depends only on include/ggemm.h.
//
// Every workload is fully determined by (family, parameters, seed): shapes come from a
// counter-based RNG (splitmix64) on the host, matrix entries from the same RNG evaluated
// on the GPU per element index, so the bytes are identical on H100/H200 and RTX 5090 and
// independent of thread scheduling.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "ggemm.h"

struct Shape { int m, n, k; };

struct Workload {
  std::string name;     // unique id, used as CSV key, e.g. "uni_G1024_s64"
  std::string family;   // uniform | moe | hetero | lora | edge
  std::vector<Shape> shapes;
  uint64_t seed;        // shape seed (data seed is derived from it)
  double total_flops() const;          // sum 2*m*n*k (FP64-equivalent flops)
  size_t elems_A() const, elems_B() const, elems_C() const;   // tight (lda=m, ldb=k, ldc=m)
  int max_k() const;
};

// Ozaki-paper test-matrix convention: a_ij = (U(0,1) - 0.5) * exp(phi * N(0,1)).
// phi controls the exponent spread: phi=0 -> uniform in [-0.5,0.5), phi=2 -> ~17 binades.
static const double kPhis[] = {0.0, 0.5, 1.0, 2.0};

// Set selector:
//   "bench" : full performance sweep (all four families, sized for a 32 GB GPU)
//   "test"  : smaller instances of every family, for correctness (dd reference is O(mnk))
//   "quick" : a handful of workloads for smoke runs / profiling
std::vector<Workload> make_workloads(const std::string& set);

// Optional: real MoE routing counts (one integer per expert per line, '#' comments).
// Shapes become m_e=count_e with (n,k) given. Returns empty workload on read error.
Workload make_moe_from_counts(const std::string& path, int n, int k, const std::string& name);

// Fill d_x[0..count) on the GPU with the phi-distribution; element i depends only on (seed, i).
void fill_ozaki_device(double* d_x, size_t count, double phi, uint64_t seed, cudaStream_t s = 0);
// Fill with a constant (e.g. NaN to check beta==0 semantics).
void fill_const_device(double* d_x, size_t count, double v, cudaStream_t s = 0);

// Device buffers for one workload, laid out contiguously problem after problem.
// ld padding: lda = m + pad, ldb = k + pad, ldc = m + pad (pad=0 by default).
struct WorkloadBuffers {
  double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC0 = nullptr;
  size_t nA = 0, nB = 0, nC = 0;
  std::vector<ggemm_problem_t> probs;  // host descriptors (device pointers)
  std::vector<size_t> offC;            // element offset of each problem's C in dC
};
// Allocates (outside any timed region), fills A,B,C0 with phi data, copies C0->C.
// Returns 0 on success.
int alloc_workload(const Workload& w, double phi, double alpha, double beta, int ld_pad,
                   WorkloadBuffers* out);
void reset_C(WorkloadBuffers* b, cudaStream_t s = 0);   // C <- C0 (device copy)
void free_workload(WorkloadBuffers* b);

// Deterministic host RNG used for shapes.
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * 0x1.0p-53; }   // [0,1)
};

inline const char* ggemm_path_name(ggemm_path_t p) {
  switch (p) {
    case GGEMM_PATH_NATIVE_LOOP: return "native_loop";
    case GGEMM_PATH_NATIVE_GROUPED: return "native_grouped";
    case GGEMM_PATH_EMU_LOOP: return "emu_loop";
    case GGEMM_PATH_EMU_GROUPED: return "emu_grouped";
    case GGEMM_PATH_AUTO: return "auto";
  }
  return "?";
}
// Parses "native_loop,emu_grouped,..." or "all" -> path list. Unknown names are skipped.
std::vector<ggemm_path_t> parse_paths(const std::string& s);
std::vector<double> parse_doubles(const std::string& s);
std::vector<int> parse_ints(const std::string& s);
