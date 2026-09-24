// ============================================================================================
// L0 MINIAPP -- the expensive step of the Gamma_1 vertex, extracted.
//
// WHY THIS KERNEL. Measured on the Si 4^3 production runs (notes/vertex_perf_plan.md, 2026-09-24):
// a chain iteration with Gamma_1 in both P and Sigma spends ~97 % of its wall in the Sigma-side
// dynamic-vertex "solve", and the solver's own timers put 95.9 % of THAT in t_l0 -- the pair-pole
// applications (rung applications are 2.4 %, refits 0.6 %). The solve does ONE operator
// application per RHS block (every unit reports "1 applications, converged true"), so there is no
// Krylov iteration count to cut: l0_apply IS the cost.
//
// WHAT IS REPRODUCED. dynbse.hpp::l0_apply_shift_cols (the inu != 0 twisted {U,T} path, which all
// but one of the 79 bosonic nodes take), per k-point:
//   A. pack the input into Vt(nc, ncomp, nR, nc), ncomp = 1 + 2 np components (constant + U_a + T_a);
//   B. for each G pole j: three skinny gemms  Pj = gj^T Vt,  Qj = Pj Ghat_j,  Bj = Pj gkq_j^T
//      and for each l: two more                Pl = Gtil_l Vt,  Rl = Pl gkq_l^T;
//   C. mulU / mulT: for every (pole, component) scatter-accumulate the (nc, nR, nc) block into the
//      five node-resolved accumulators AU, AT, M2, A1, A3 of shape (2, np, nc, nR, nc);
//   D. assemble AU/AT (+ the confluent M2/A1/A3 through the D^2 / D^3 tables) into F and Fsum.
// Phase B is BLAS3 but tall-and-skinny (K = N = nc = 8); phase C is a pure read-modify-write
// scatter with no reuse. Production shape per L0 application: ~1.1 TFLOP of gemm and ~0.3 TB of
// scatter traffic -- which is why it is the target for a GPU port and for CPU re-blocking.
//
// The arrays are filled with deterministic pseudo-random data of the right shapes and sparsity;
// the miniapp checks kernels against each other, never against physics.
// ============================================================================================
#pragma once
#include <complex>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace l0mini {

using cplx = std::complex<double>;

// production dimensions: Si 4^3, C = [0,8), DLR prec high, the Sigma-side UNION grid
// (79 vertex nodes + 80 G nodes = 159), RHS blocked at 32 of N_m = 156
struct dims {
  long nk = 64;      // k-points (the OpenMP axis)
  long nc = 8;       // C-window size
  long nR = 32;      // right-hand-side block width
  long np = 159;     // pole nodes (union grid)
  long ng = 80;      // G poles
  long np_fit = 79;  // DLR fit nodes (confluent terms must land below this)
  long ncomp() const { return 1 + 2 * np; }
  long nc2() const { return nc * nc; }
  void parse(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      auto eq = std::string(argv[i]);
      auto set = [&](const char *k, long &v) {
        if (eq.rfind(std::string("--") + k + "=", 0) == 0) v = std::atol(eq.c_str() + std::strlen(k) + 3);
      };
      set("nk", nk); set("nc", nc); set("nR", nR); set("np", np); set("ng", ng); set("np_fit", np_fit);
    }
    if (np_fit > np) np_fit = np;
  }
  void print(const char *tag) const {
    std::printf("[%s] nk %ld, nc %ld, nR %ld, np %ld (fit %ld), ng %ld, ncomp %ld\n",
                tag, nk, nc, nR, np, np_fit, ng, ncomp());
  }
};

// xorshift64*: identical streams on host and device, no <random> divergence
struct rng {
  std::uint64_t s;
  explicit rng(std::uint64_t seed) : s(seed ? seed : 88172645463325252ull) {}
  double next() {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return double((s * 2685821657736338717ull) >> 11) / double(1ull << 53) - 0.5;
  }
  cplx nextc() { const double a = next(), b = next(); return cplx(a, b); }
};

// everything the kernel reads, in the production layouts
struct inputs {
  dims d;
  cplx inu{0.0, 0.137};                       // a generic bosonic node (the inu != 0 path)
  std::vector<cplx> gk, gkq;                  // (ng, nk, nc, nc) pole residues at k and k+q
  std::vector<cplx> Ghat, Gtil;               // (nk, ng, nc, nc) the per-k pole matrices
  std::vector<cplx> Xfam;                     // (2, np, nk, nc, nc, nR) the input vector's pole families
  std::vector<cplx> Xcst;                     // (nk, nc, nc, nR) its constant part
  std::vector<double> eps, epsG, fhalf, fd1;  // pole energies / the half-sum and derivative weights
  std::vector<long> gnode;                    // G pole -> vertex node
  std::vector<cplx> Dsq, Dcb;                 // (np, np) the confluent D^2 / D^3 tables

  void build(dims const &dd, std::uint64_t seed = 20260924ull) {
    d = dd;
    rng r(seed);
    auto fill = [&](std::vector<cplx> &v, long n) { v.resize(size_t(n)); for (auto &z : v) z = r.nextc(); };
    fill(gk, d.ng * d.nk * d.nc * d.nc);
    fill(gkq, d.ng * d.nk * d.nc * d.nc);
    fill(Ghat, d.nk * d.ng * d.nc * d.nc);
    fill(Gtil, d.nk * d.ng * d.nc * d.nc);
    fill(Xfam, 2 * d.np * d.nk * d.nc * d.nc * d.nR);
    fill(Xcst, d.nk * d.nc * d.nc * d.nR);
    eps.resize(size_t(d.np)); for (long a = 0; a < d.np; ++a) eps[size_t(a)] = 0.05113 + 0.31071 * double(a);
    epsG.resize(size_t(d.ng)); for (long j = 0; j < d.ng; ++j) epsG[size_t(j)] = 0.07021 + 0.29037 * double(j);
    // the kernel divides by (e_j - e_a): a coincidence makes it NaN, and the production pole sets never collide
    for (long j = 0; j < d.ng; ++j)
      for (long a = 0; a < d.np; ++a)
        if (std::abs(epsG[size_t(j)] - eps[size_t(a)]) < 1e-9) { std::printf("pole collision j %ld a %ld\n", j, a); std::abort(); }
    fhalf.resize(size_t(d.np)); fd1.resize(size_t(d.np));
    for (long a = 0; a < d.np; ++a) { fhalf[size_t(a)] = 0.5 * r.next(); fd1[size_t(a)] = 0.25 * r.next(); }
    gnode.resize(size_t(d.ng));
    for (long j = 0; j < d.ng; ++j) gnode[size_t(j)] = j % d.np_fit;   // every G pole maps into the fit set
    // the confluent tables are SPARSE in practice (a DLR fit matrix): keep ~8 nonzeros per row
    Dsq.assign(size_t(d.np * d.np), cplx(0.0));
    Dcb.assign(size_t(d.np * d.np), cplx(0.0));
    for (long n = 0; n < d.np_fit; ++n)
      for (int t = 0; t < 8; ++t) {
        const long c = (n * 7 + t * 13) % d.np;
        Dsq[size_t(n * d.np + c)] = r.nextc();
        Dcb[size_t(n * d.np + c)] = r.nextc();
      }
  }
};

// the kernel's output
struct outputs {
  std::vector<cplx> Ffam;   // (2, np, nk, nc, nc, nR)
  std::vector<cplx> Fsum;   // (nk, nc, nc, nR)
  void alloc(dims const &d) {
    Ffam.assign(size_t(2 * d.np * d.nk * d.nc * d.nc * d.nR), cplx(0.0));
    Fsum.assign(size_t(d.nk * d.nc * d.nc * d.nR), cplx(0.0));
  }
};

inline double rel_diff(std::vector<cplx> const &a, std::vector<cplx> const &b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) { num += std::norm(a[i] - b[i]); den += std::norm(b[i]); }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// the arithmetic the kernel performs, for a FLOP/byte report
inline void cost_model(dims const &d, double &gflop, double &gbytes) {
  const double C = double(d.ncomp()) * double(d.nR) * double(d.nc);
  const double gemm_macs = double(d.nk) * double(d.ng) * 5.0 * double(d.nc) * double(d.nc) * C;
  gflop = 8.0 * gemm_macs * 1e-9;                                   // 8 real FLOP per complex MAC
  const double scatter = double(d.nk) * double(d.ng) * double(d.ncomp()) * double(d.nc * d.nR * d.nc) * 4.0;
  gbytes = scatter * 3.0 * 16.0 * 1e-9;                             // read V, read-modify-write an accumulator
}

} // namespace l0mini
