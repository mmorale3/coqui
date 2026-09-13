/**
 * scGW-tilde Tier 2 full frequency, increment D1 (notes/dynbse_plan.md): the two-family DLR
 * algebra of the resummed DYNAMIC-rung BSE (methods/vertex/dynbse.hpp) against a DENSE
 * Matsubara BSE oracle on an exact rational toy.
 *
 * Toy: nk = 2 (k and k+q on a two-point mesh), nc = 2 bands, exact G from the Schur embedding
 * (toy_poles.hpp), an explicit pair-space rung K(q'; inu') = Z(q') + sum_p M_p(q')
 * [1/(inu'-e_p) - 1/(inu'+e_p)] with two bosonic poles. The oracle solves
 *   Gamma(k,n) = D(k) + (1/beta) sum_{k',m} K(k-k'; nu_{n-m}) L0(k',m) Gamma(k',m)
 * on an explicit fermionic grid |n| < N by dense LU, with the kernel tail |m| >= N supplied
 * EXACTLY by the closed-form one-rung convolution (Gamma -> D there; the residual truncation
 * error is O(1/N^2) and is removed by Richardson in N), and the bare tail of the readout sum
 * taken from the closed-form T-sums. Gates:
 *   (0) the closed-form rung convolution vs a dense sum (the oracle's own ingredient);
 *   (A) static limit: the solver's y = 0 vertex (the closed-form static resolvent) vs the
 *       oracle with the constant kernel K0 -- machine class;
 *   (C) resummed: the solver's converged vertex vs the oracle with the full dynamic kernel, at
 *       inu = 0 and inu != 0 -- Richardson class (~1e-8);
 *   (C1) the first iterate Gamma_1 vs the oracle's first-order Dyson term (L_s D + L_s K_d L_s D);
 *   (D) the same toy through the FITTED, shared aux grid (the production pathway: G refit on
 *       the vertex's nodes => confluent products through the double-pole table) -- reported
 *       and gated at fit class.
 */

#undef NDEBUG

#include <cmath>
#include <optional>
#include <cstdio>
#include <string>
#include <complex>
#include <random>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
#include "methods/vertex/ward_legs.hpp"
#include "methods/vertex/dynbse.hpp"
#include "toy_poles.hpp"

namespace bdft_tests {

  namespace wl = methods::solvers::ward_legs;
  namespace db = methods::solvers::dynbse;

  // ---------------- the toy unit ----------------------------------------------------------
  struct dyn_toy {
    long nk = 2, nc = 2, nc2 = 4, nR = 8, npole = 2;
    double beta = 10.0;
    std::vector<toy_k> t;                       // t[k]
    nda::array<long, 2> kmk;                    // (nk, nk) q' = k - k'
    std::vector<long> kpq;                      // k + q
    nda::array<double, 1> epsW;                 // (npole) bosonic poles (> 0)
    nda::array<cplx, 3> Z;                      // (nq, nc2, nc2)
    nda::array<cplx, 4> M;                      // (npole, nq, nc2, nc2)
    nda::array<cplx, 4> Dc;                     // (nk, nc, nc, nR) the external legs

    nda::array<cplx, 2> W(long iq, cplx inup) const {   // K(q'; inu') = Z + W_dyn
      nda::array<cplx, 2> w(nc2, nc2);
      w() = Z(iq, all_, all_);
      for (long p = 0; p < npole; ++p)
        w += M(p, iq, all_, all_) * (1.0 / (inup - epsW(p)) - 1.0 / (inup + epsW(p)));
      return w;
    }
    nda::array<cplx, 2> L0(long ik, cplx z, cplx inu) const {   // (nc2 x nc2), rows (p1' nc + p3), cols (a nc + b)
      auto Gk = t[size_t(ik)].G(z);
      auto Gq = t[size_t(kpq[size_t(ik)])].G(z + inu);
      nda::array<cplx, 2> L(nc2, nc2);
      for (long p1p = 0; p1p < nc; ++p1p)
        for (long p3 = 0; p3 < nc; ++p3)
          for (long a = 0; a < nc; ++a)
            for (long b = 0; b < nc; ++b) L(p1p * nc + p3, a * nc + b) = Gk(a, p1p) * Gq(p3, b);
      return L;
    }
  };

  inline dyn_toy make_dyn_toy(double zscale, double mscale) {
    dyn_toy T;
    T.t = {make_toy(T.nc, {-1.5, 1.5}, {0.3, 0.3}, 41u), make_toy(T.nc, {-1.5, 1.5}, {0.3, 0.3}, 43u)};
    T.kmk = nda::array<long, 2>(T.nk, T.nk);
    for (long k = 0; k < T.nk; ++k)
      for (long kp = 0; kp < T.nk; ++kp) T.kmk(k, kp) = (k - kp + T.nk) % T.nk;
    T.kpq = {1, 0};                                     // q = the non-zero vector of the 2-point mesh
    T.epsW = nda::array<double, 1>(T.npole);
    T.epsW(0) = 0.8;
    T.epsW(1) = 1.6;
    std::mt19937 gen(59u);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto herm = [&](double sc) {
      nda::array<cplx, 2> A(T.nc2, T.nc2), H(T.nc2, T.nc2);
      for (auto &v : A) v = sc * cplx(nd(gen), nd(gen));
      for (long i = 0; i < T.nc2; ++i)
        for (long j = 0; j < T.nc2; ++j) H(i, j) = 0.5 * (A(i, j) + std::conj(A(j, i)));
      return H;
    };
    T.Z = nda::array<cplx, 3>(T.nk, T.nc2, T.nc2);
    T.M = nda::array<cplx, 4>(T.npole, T.nk, T.nc2, T.nc2);
    for (long iq = 0; iq < T.nk; ++iq) {
      T.Z(iq, all_, all_) = herm(zscale);
      for (long p = 0; p < T.npole; ++p) T.M(p, iq, all_, all_) = herm(mscale);
    }
    T.Dc = nda::array<cplx, 4>(T.nk, T.nc, T.nc, T.nR);
    T.Dc() = cplx(0.0);
    for (long k = 0; k < T.nk; ++k)
      for (long a = 0; a < T.nc; ++a)
        for (long b = 0; b < T.nc; ++b) T.Dc(k, a, b, k * T.nc2 + a * T.nc + b) = cplx(1.0);
    return T;
  }

  // closed-form (1/beta) sum_m [1/(iw_n - iw_m - e) - 1/(iw_n - iw_m + e)] / (iw_m - ej)
  inline cplx conv_pole(double beta, cplx iwn, double ej, double e) {
    const double fj = wl::nF(beta, ej), nb = db::nB(beta, e);
    return -(1.0 - fj + nb) / (iwn - ej - e) - (fj + nb) / (iwn - ej + e);
  }

  // ---------------- the dense oracle -------------------------------------------------------
  // returns the readout block P (nR x nR) = sum_k D^dag sum_iw L0 Gamma at explicit-grid size N
  // mode: 0 = static kernel K0 at every inu' (the static ladder), 1 = full dynamic kernel
  inline nda::array<cplx, 2> dense_bse(dyn_toy const &T, cplx inu, long N, int mode) {
    const long nk = T.nk, nc = T.nc, nc2 = T.nc2, nR = T.nR, twoN = 2 * N;
    const double beta = T.beta;
    auto iw = [&](long n) { return I_ * cplx((2.0 * double(n) + 1.0) * M_PI / beta); };
    auto inuj = [&](long j) { return I_ * cplx(2.0 * M_PI * double(j) / beta); };
    // pair-propagator blocks and kernel tables
    std::vector<nda::array<cplx, 2>> L0(size_t(nk * twoN));
    for (long k = 0; k < nk; ++k)
      for (long n = -N; n < N; ++n) L0[size_t(k * twoN + n + N)] = T.L0(k, iw(n), inu);
    std::vector<nda::array<cplx, 2>> Wt(size_t(nk * (2 * twoN)));    // (q', j + twoN), j in [-(twoN-1), twoN-1]
    nda::array<cplx, 3> K0(nk, nc2, nc2);
    for (long iq = 0; iq < nk; ++iq) {
      K0(iq, all_, all_) = T.W(iq, cplx(0.0));
      for (long j = -twoN + 1; j < twoN; ++j) {
        auto w = (mode == 0) ? nda::array<cplx, 2>(K0(iq, all_, all_)) : T.W(iq, inuj(j));
        Wt[size_t(iq * 2 * twoN + j + twoN)] = w;
      }
    }
    auto Wof = [&](long iq, long j) -> nda::array<cplx, 2> const & { return Wt[size_t(iq * 2 * twoN + j + twoN)]; };
    // the exact kernel-tail correction: sum_{|m| >= N} K(n-m) L0(m) V for a CONSTANT V(k')
    // = [closed-form full convolution] - [explicit |m| < N part]
    auto rung_full = [&](long k, long n, nda::array<cplx, 4> const &V, long r) {
      nda::array<cplx, 1> out(nc2), a(nc2);
      out() = cplx(0.0);
      for (long kp = 0; kp < nk; ++kp) {
        const long iq = T.kmk(k, kp);
        auto const &tk = T.t[size_t(kp)];
        auto const &tq = T.t[size_t(T.kpq[size_t(kp)])];
        for (long j = 0; j < tk.ng; ++j)
          for (long l = 0; l < tq.ng; ++l) {
            const double ej = tk.lam(j), el = tq.lam(l);
            const cplx Djl = cplx(ej - el) + inu;
            // pair vector A_jl = pack(g_j(k')^T V g_l(k'+q)^T)
            for (long p1p = 0; p1p < nc; ++p1p)
              for (long p3 = 0; p3 < nc; ++p3) {
                cplx s(0.0);
                for (long aa = 0; aa < nc; ++aa)
                  for (long bb = 0; bb < nc; ++bb)
                    s += tk.g(j, aa, p1p) * V(kp, aa, bb, r) * tq.g(l, p3, bb);
                a(p1p * nc + p3) = s / Djl;
              }
            // scalar convolution factors: Z part = f_j - f_l (n-independent); dynamic per pole
            const double fj = wl::nF(beta, ej), fl = wl::nF(beta, el);
            nda::array<cplx, 2> Keff(nc2, nc2);
            Keff() = T.Z(iq, all_, all_) * cplx(fj - fl);
            if (mode != 0)
              for (long p = 0; p < T.npole; ++p) {
                const cplx cu = conv_pole(beta, iw(n), ej, T.epsW(p));
                const cplx cs = conv_pole(beta, iw(n) + inu, el, T.epsW(p));
                Keff += T.M(p, iq, all_, all_) * (cu - cs);
              }
            else {
              // constant kernel K0 = Z + W_dyn(0): the dynamic part also multiplies (f_j - f_l)
              nda::array<cplx, 2> Wd0(nc2, nc2);
              Wd0() = K0(iq, all_, all_) - T.Z(iq, all_, all_);
              Keff += Wd0 * cplx(fj - fl);
            }
            for (long p = 0; p < nc2; ++p) {
              cplx s(0.0);
              for (long pp = 0; pp < nc2; ++pp) s += Keff(p, pp) * a(pp);
              out(p) += s;
            }
          }
      }
      return out;
    };
    auto rung_partial = [&](long k, long n, nda::array<cplx, 4> const &V, long r) {
      nda::array<cplx, 1> out(nc2), v(nc2), lv(nc2);
      out() = cplx(0.0);
      for (long kp = 0; kp < nk; ++kp) {
        const long iq = T.kmk(k, kp);
        for (long a = 0; a < nc; ++a)
          for (long b = 0; b < nc; ++b) v(a * nc + b) = V(kp, a, b, r);
        for (long m = -N; m < N; ++m) {
          auto const &L = L0[size_t(kp * twoN + m + N)];
          for (long p = 0; p < nc2; ++p) {
            cplx s(0.0);
            for (long pp = 0; pp < nc2; ++pp) s += L(p, pp) * v(pp);
            lv(p) = s;
          }
          auto const &Wn = Wof(iq, n - m);
          for (long p = 0; p < nc2; ++p) {
            cplx s(0.0);
            for (long pp = 0; pp < nc2; ++pp) s += Wn(p, pp) * lv(pp);
            out(p) += s / beta;
          }
        }
      }
      return out;
    };
    // the tail's vertex: the STATIC vertex Gamma_s = (1 - K0 Cb)^-1 D (exact for mode 0; the
    // large-frequency limit of the dynamic vertex otherwise), in the (k, pair) space
    nda::array<cplx, 4> Gs(nk, nc, nc, nR);
    {
      const long Dp = nk * nc2;
      nda::array<cplx, 2> Cb(Dp, Dp), Ksm(Dp, Dp), KC(Dp, Dp);
      Cb() = cplx(0.0);
      Ksm() = cplx(0.0);
      for (long k = 0; k < nk; ++k) {
        auto const &tk = T.t[size_t(k)];
        auto const &tq = T.t[size_t(T.kpq[size_t(k)])];
        for (long j = 0; j < tk.ng; ++j)
          for (long l = 0; l < tq.ng; ++l) {
            const cplx Tjl = wl::two_pole(wl::fermi_all(beta, tk.lam(j)), tk.lam(j),
                                          wl::fermi_all(beta, tq.lam(l)), tq.lam(l), inu, false).T;
            for (long p1p = 0; p1p < nc; ++p1p)
              for (long p3 = 0; p3 < nc; ++p3)
                for (long a = 0; a < nc; ++a)
                  for (long b = 0; b < nc; ++b)
                    Cb(k * nc2 + p1p * nc + p3, k * nc2 + a * nc + b) += Tjl * tk.g(j, a, p1p) * tq.g(l, p3, b);
          }
        for (long kp = 0; kp < nk; ++kp)
          for (long p = 0; p < nc2; ++p)
            for (long pp = 0; pp < nc2; ++pp) Ksm(k * nc2 + p, kp * nc2 + pp) = K0(T.kmk(k, kp), p, pp);
      }
      nda::blas::gemm(Ksm, Cb, KC);
      nda::matrix<cplx> Mm(Dp, Dp);
      for (long i = 0; i < Dp; ++i)
        for (long j = 0; j < Dp; ++j) Mm(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) - KC(i, j);
      nda::inverse_in_place(Mm);
      for (long k = 0; k < nk; ++k)
        for (long p = 0; p < nc2; ++p)
          for (long r = 0; r < nR; ++r) {
            cplx v(0.0);
            for (long kp = 0; kp < nk; ++kp)
              for (long pp = 0; pp < nc2; ++pp) v += Mm(k * nc2 + p, kp * nc2 + pp) * T.Dc(kp, pp / nc, pp % nc, r);
            Gs(k, p / nc, p % nc, r) = v;
          }
    }
    // the linear system A x = b, A = I - (1/beta) K L0
    const long Dd = nk * twoN * nc2;
    nda::matrix<cplx> A(Dd, Dd);
    A() = cplx(0.0);
    for (long k = 0; k < nk; ++k)
      for (long n = -N; n < N; ++n)
        for (long kp = 0; kp < nk; ++kp) {
          const long iq = T.kmk(k, kp);
          for (long m = -N; m < N; ++m) {
            auto const &Wn = Wof(iq, n - m);
            auto const &L = L0[size_t(kp * twoN + m + N)];
            nda::array<cplx, 2> WL(nc2, nc2);
            nda::blas::gemm(Wn, L, WL);
            const long row0 = (k * twoN + n + N) * nc2, col0 = (kp * twoN + m + N) * nc2;
            for (long p = 0; p < nc2; ++p)
              for (long pp = 0; pp < nc2; ++pp) A(row0 + p, col0 + pp) -= WL(p, pp) / beta;
          }
        }
    for (long i = 0; i < Dd; ++i) A(i, i) += cplx(1.0);
    // RHS: the external leg + the exact kernel tail
    nda::array<cplx, 2> Aa(Dd, Dd);
    {
      nda::matrix<cplx> Ai(Dd, Dd);
      Ai() = A;
      nda::inverse_in_place(Ai);
      Aa() = Ai;
    }
    auto solve_with = [&](nda::array<cplx, 2> const &B) {
      nda::array<cplx, 2> X(Dd, nR);
      nda::blas::gemm(Aa, B, X);
      return X;
    };
    auto readout = [&](nda::array<cplx, 2> const &X, nda::array<cplx, 4> const &V) {
      // Gsum(k,p,r) = (1/beta) sum_{|n|<N} [L0 X]_p + bare tail (Cb_exact - Cb_partial) V
      nda::array<cplx, 4> Gsum(nk, nc, nc, nR);
      Gsum() = cplx(0.0);
      for (long k = 0; k < nk; ++k)
        for (long r = 0; r < nR; ++r) {
          for (long n = -N; n < N; ++n) {
            auto const &L = L0[size_t(k * twoN + n + N)];
            for (long p = 0; p < nc2; ++p) {
              cplx s(0.0);
              for (long pp = 0; pp < nc2; ++pp) s += L(p, pp) * X((k * twoN + n + N) * nc2 + pp, r);
              Gsum(k, p / nc, p % nc, r) += s / beta;
            }
          }
          // bare tail: (Cb_exact - Cb_partial) D
          auto const &tk = T.t[size_t(k)];
          auto const &tq = T.t[size_t(T.kpq[size_t(k)])];
          nda::array<cplx, 2> Cb(nc2, nc2);
          Cb() = cplx(0.0);
          for (long j = 0; j < tk.ng; ++j)
            for (long l = 0; l < tq.ng; ++l) {
              const cplx Tjl = wl::two_pole(wl::fermi_all(beta, tk.lam(j)), tk.lam(j),
                                            wl::fermi_all(beta, tq.lam(l)), tq.lam(l), inu, false).T;
              for (long p1p = 0; p1p < nc; ++p1p)
                for (long p3 = 0; p3 < nc; ++p3)
                  for (long a = 0; a < nc; ++a)
                    for (long b = 0; b < nc; ++b)
                      Cb(p1p * nc + p3, a * nc + b) += Tjl * tk.g(j, a, p1p) * tq.g(l, p3, b);
            }
          for (long n = -N; n < N; ++n) Cb -= L0[size_t(k * twoN + n + N)] / beta;
          for (long p = 0; p < nc2; ++p) {
            cplx s(0.0);
            for (long pp = 0; pp < nc2; ++pp) s += Cb(p, pp) * V(k, pp / nc, pp % nc, r);
            Gsum(k, p / nc, p % nc, r) += s;
          }
        }
      return Gsum;
    };
    // The tail's vertex V: the large-frequency limit Gamma(i inf) = D + K(i inf) Gsum with
    // K(i inf) = K0 (static kernel: V = Gamma_s exactly, one pass) or Z (dynamic kernel: the
    // dynamic part of the rung vanishes at large transfer; V is made self-consistent with the
    // solved Gsum in a few passes, and the residual tail error is then O(1/N^3)). First pass:
    // V = Gamma_s.
    nda::array<cplx, 4> V(Gs), Gsum(nk, nc, nc, nR);
    for (int pass = 0; pass < ((mode == 0) ? 1 : 8); ++pass) {
      nda::array<cplx, 2> B(Dd, nR);
      for (long k = 0; k < nk; ++k)
        for (long r = 0; r < nR; ++r)
          for (long n = -N; n < N; ++n) {
            auto full = rung_full(k, n, V, r);
            auto part = rung_partial(k, n, V, r);
            for (long p = 0; p < nc2; ++p)
              B((k * twoN + n + N) * nc2 + p, r) = T.Dc(k, p / nc, p % nc, r) + full(p) - part(p);
          }
      auto X = solve_with(B);
      Gsum = readout(X, V);
      if (mode == 0) break;
      nda::array<cplx, 4> Vn(nk, nc, nc, nR);
      double dv = 0.0, sv = 0.0;
      for (long k = 0; k < nk; ++k)
        for (long p = 0; p < nc2; ++p)
          for (long r = 0; r < nR; ++r) {
            cplx v = T.Dc(k, p / nc, p % nc, r);
            for (long kp = 0; kp < nk; ++kp) {
              const long iq = T.kmk(k, kp);
              for (long pp = 0; pp < nc2; ++pp) v += T.Z(iq, p, pp) * Gsum(kp, pp / nc, pp % nc, r);
            }
            Vn(k, p / nc, p % nc, r) = v;
            dv = std::max(dv, std::abs(v - V(k, p / nc, p % nc, r)));
            sv = std::max(sv, std::abs(v));
          }
      V = Vn;
      if (dv < 1e-14 * sv) break;
    }
    return db::collapse(T.Dc, Gsum);
  }

  // ---------------- the two-family solver on the toy ---------------------------------------
  struct solver_out {
    nda::array<cplx, 2> P, P0, P1;
    db::dyson_result res;
    double dsq_err = 0.0, shift_fit_err = 0.0;
  };

  inline solver_out run_solver(dyn_toy const &T, imag_axes_ft::IAFT const &ft, cplx inu,
                               int fitted, double tol, long maxit, long gmres_m = 0, int table_mode = 0) {
    solver_out o;
    auto b = db::build_freq_basis(ft);
    o.dsq_err = b.dsq_fit_err;
    const long nk = T.nk, nc = T.nc, nc2 = T.nc2, nt = b.nt;
    db::pair_poles P;
    if (fitted == 2) {
      // the PRODUCTION union scheme: G refit on the half-spacing-shifted grid (no node coincides
      // with a vertex node), the basis extended by those nodes
      const long np_fit = b.np;
      auto gnodes = db::shifted_nodes(b);
      db::node_pole_fit npf;
      npf.build(T.beta, b.s, gnodes, 1e-8);
      const long ng = gnodes.shape(0);
      nda::array<cplx, 4> gk(ng, nk, nc, nc), gkq(ng, nk, nc, nc);
      double fe = 0.0;
      for (long k = 0; k < nk; ++k) {
        nda::array<cplx, 2> F(nt, nc * nc);
        F() = cplx(0.0);
        auto const &tk = T.t[size_t(k)];
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < tk.ng; ++j) {
            const double K = imag_axes_ft::dlr_kF(T.beta, b.s(i), tk.lam(j));
            for (long a = 0; a < nc; ++a)
              for (long bb = 0; bb < nc; ++bb) F(i, a * nc + bb) += tk.g(j, a, bb) * K;
          }
        auto c = npf.coeffs(F);
        fe = std::max(fe, npf.fit_error(F, c));
        for (long p = 0; p < ng; ++p)
          for (long a = 0; a < nc; ++a)
            for (long bb = 0; bb < nc; ++bb) gk(p, k, a, bb) = c(p, a * nc + bb);
      }
      app_log(1, "dynbse union/fitted: G refit on {} shifted nodes, fit_error {:.3e}, kept {} of {} (s_min/s_max {:.2e})",
              ng, fe, npf.n_kept, ng, npf.s_min_kept / npf.s_max);
      for (long k = 0; k < nk; ++k) gkq(all_, k, all_, all_) = gk(all_, T.kpq[size_t(k)], all_, all_);
      db::extend_freq_basis(b, gnodes);
      P = db::make_pair_poles(T.beta, gnodes, gk, gkq, np_fit);
    } else if (fitted == 0) {
      // the union of the two toys' exact poles (zero residues cross-wise), appended to the
      // vertex node set (union basis: the refit still targets the DLR part)
      auto set = assemble({T.t[0], T.t[1]});
      const long ng = set.epsG.shape(0);
      nda::array<cplx, 4> gk(ng, nk, nc, nc), gkq(ng, nk, nc, nc);
      for (long k = 0; k < nk; ++k) {
        gk(all_, k, all_, all_) = set.g(0, all_, k, all_, all_);
        gkq(all_, k, all_, all_) = set.g(0, all_, T.kpq[size_t(k)], all_, all_);
      }
      const long np_fit = b.np;
      db::extend_freq_basis(b, set.epsG);
      P = db::make_pair_poles(T.beta, set.epsG, gk, gkq, np_fit);
    } else {
      // G refit on the vertex's aux grid from its exact tau samples (the production pathway)
      const long np = b.np;
      nda::array<cplx, 4> gk(np, nk, nc, nc), gkq(np, nk, nc, nc);
      for (long k = 0; k < nk; ++k) {
        nda::array<cplx, 2> F(nt, nc * nc);
        F() = cplx(0.0);
        auto const &tk = T.t[size_t(k)];
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < tk.ng; ++j) {
            const double K = imag_axes_ft::dlr_kF(T.beta, b.s(i), tk.lam(j));
            for (long a = 0; a < nc; ++a)
              for (long bb = 0; bb < nc; ++bb) F(i, a * nc + bb) += tk.g(j, a, bb) * K;
          }
        auto c = b.pf.coeffs(F);
        app_log(1, "dynbse fitted G(k = {}): fit_error {:.3e}, residue ratio {:.3g}", k,
                b.pf.fit_error(F, c), b.pf.residue_ratio(F, c));
        for (long p = 0; p < np; ++p)
          for (long a = 0; a < nc; ++a)
            for (long bb = 0; bb < nc; ++bb) gk(p, k, a, bb) = c(p, a * nc + bb);
      }
      for (long k = 0; k < nk; ++k) gkq(all_, k, all_, all_) = gk(all_, T.kpq[size_t(k)], all_, all_);
      P = db::make_pair_poles(T.beta, b.eps, gk, gkq);
    }
    db::pair_rung R;
    R.nq = nk; R.nt = nt; R.nc2 = nc2;
    R.kmk = T.kmk;
    R.Wd_s = nda::array<cplx, 4>(nk, nt, nc2, nc2);
    R.Wd0 = nda::array<cplx, 3>(nk, nc2, nc2);
    R.K0 = nda::array<cplx, 3>(nk, nc2, nc2);
    R.Wd_s() = cplx(0.0);
    R.Wd0() = cplx(0.0);
    for (long iq = 0; iq < nk; ++iq) {
      for (long p = 0; p < T.npole; ++p) {
        R.Wd0(iq, all_, all_) += T.M(p, iq, all_, all_) * cplx(-2.0 / T.epsW(p));
        for (long i = 0; i < nt; ++i)
          R.Wd_s(iq, i, all_, all_) += T.M(p, iq, all_, all_) *
                                        cplx(db::KB(T.beta, b.s(i), T.epsW(p)) - db::KB(T.beta, b.s(i), -T.epsW(p)));
      }
      R.K0(iq, all_, all_) = T.Z(iq, all_, all_) + R.Wd0(iq, all_, all_);
    }
    const bool shared = (fitted == 1);
    auto S = db::build_static_resolvent(b, P, R, inu, shared);
    std::optional<db::shift_tables> stab;
    db::shift_tables const *stp = nullptr;
    if (inu != cplx(0.0)) {
      stab.emplace(db::build_shift_tables(b, inu, 1e-11, table_mode));
      stp = std::addressof(stab.value());
      o.shift_fit_err = stab->fit_err;
    }
    if (gmres_m > 0) {
      auto kd = [&](db::tf_vector const &F, nda::array<cplx, 4> const &Fsum, db::tf_vector &y) {
        return db::kd_apply(b, R, F, Fsum, y, inu);
      };
      o.res = db::solve_dyson_gmres(b, P, kd, S, inu, shared, T.Dc, tol, maxit, gmres_m, nullptr, stp);
    } else
      o.res = db::solve_dyson(b, P, R, S, inu, shared, T.Dc, tol, maxit, false, stp);
    o.P = db::collapse(T.Dc, o.res.Gsum);
    o.P0 = db::collapse(T.Dc, o.res.Gsum0);
    o.P1 = db::collapse(T.Dc, o.res.Gsum1);
    return o;
  }

  inline double rel_diff(nda::array<cplx, 2> const &A, nda::array<cplx, 2> const &B) {
    double d = 0.0, s = 0.0;
    for (long i = 0; i < A.shape(0); ++i)
      for (long j = 0; j < A.shape(1); ++j) {
        d = std::max(d, std::abs(A(i, j) - B(i, j)));
        s = std::max(s, std::abs(B(i, j)));
      }
    return (s > 0.0) ? d / s : d;
  }

  // ======================================================================================
  TEST_CASE("dynbse_oracle", "[methods][vertex][scgwt][dynbse]") {
#ifndef ENABLE_DLR
    SUCCEED("dynbse_oracle skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;
    auto T = make_dyn_toy(0.10, 0.02);
    const double beta = T.beta;

    // (0) the closed-form rung convolution vs a dense sum
    {
      const double ej = -0.37, e = 0.8;
      const cplx iwn = I_ * cplx(7.0 * M_PI / beta);
      const cplx ref = richardson_fsum(beta, 20000, 3, [&](cplx z) {
        return (1.0 / (iwn - z - e) - 1.0 / (iwn - z + e)) / (z - ej);
      });
      const cplx val = conv_pole(beta, iwn, ej, e);
      app_log(1, "dynbse (0): rung convolution closed form vs dense: |diff| = {:.3e} (|ref| = {:.3e})",
              std::abs(val - ref), std::abs(ref));
      REQUIRE(std::abs(val - ref) < 1e-9 * std::max(1.0, std::abs(ref)));
    }

    imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
    // ---- (L)/(K): the two operations pointwise on the exact node set ----------------------
    for (cplx inu : {cplx(0.0), I_ * cplx(2.0 * M_PI * 2.0 / beta)}) {
      auto b = db::build_freq_basis(ft);
      auto set = assemble({T.t[0], T.t[1]});
      const long ng = set.epsG.shape(0), nk = T.nk, nc = T.nc, nc2 = T.nc2, nt = b.nt;
      nda::array<cplx, 4> gk(ng, nk, nc, nc), gkq(ng, nk, nc, nc);
      for (long k = 0; k < nk; ++k) {
        gk(all_, k, all_, all_) = set.g(0, all_, k, all_, all_);
        gkq(all_, k, all_, all_) = set.g(0, all_, T.kpq[size_t(k)], all_, all_);
      }
      const long np_fit = b.np;
      db::extend_freq_basis(b, set.epsG);
      const long np = b.np;
      auto P = db::make_pair_poles(beta, set.epsG, gk, gkq, np_fit);
      const bool nu0 = (inu == cplx(0.0));
      // family 1 is the twisted pair T_c = U_c S_c at inu != 0 (D2e); at inu = 0 the families coincide
      auto eval_tf = [&](db::tf_vector const &V, long k, cplx z, long r, nda::array<cplx, 2> &out) {
        out() = cplx(0.0);
        for (long c = 0; c < np; ++c) {
          const cplx u = 1.0 / (z - b.eps(c)), sh = nu0 ? u : u / (z + inu - b.eps(c));
          for (long i = 0; i < nc; ++i)
            for (long jj = 0; jj < nc; ++jj)
              out(i, jj) += V.fam(0, c, k, i, jj, r) * u + V.fam(1, c, k, i, jj, r) * sh;
        }
        for (long i = 0; i < nc; ++i)
          for (long jj = 0; jj < nc; ++jj) out(i, jj) += V.cst(k, i, jj, r);
      };
      // (L): X = U_a e_(0,1) at k = 0 (fam 0) and S_a' e_(1,0) at k = 1 (fam 1), a few nodes
      db::tf_vector X(np, nk, nc, 1), F(np, nk, nc, 1);
      nda::array<cplx, 4> Fsum(nk, nc, nc, 1);
      X.fam(0, 3, 0, 0, 1, 0) = cplx(1.0);
      X.fam(1, 7, 1, 1, 0, 0) = cplx(0.7, -0.2);
      X.cst(0, 1, 1, 0) = cplx(0.3);
      std::optional<db::shift_tables> stL;
      db::shift_tables const *stLp = nullptr;
      if (not nu0) { stL.emplace(db::build_shift_tables(b, inu)); stLp = std::addressof(stL.value()); }
      if (stLp) app_log(1, "dynbse shift tables (inu = {:.3f}i): fit err {:.3e}, kept condition {:.3e}", inu.imag(), stLp->fit_err, stLp->cond_kept);
      db::l0_apply(b, P, inu, false, X, F, Fsum, nullptr, stLp);
      double errL = 0.0, scL = 0.0;
      for (long k = 0; k < nk; ++k)
        for (long n : {0l, 3l, 17l, -6l}) {
          const cplx z = I_ * cplx((2.0 * n + 1.0) * M_PI / beta);
          nda::array<cplx, 2> got(nc, nc), xin(nc, nc), ref(nc, nc);
          eval_tf(F, k, z, 0, got);
          eval_tf(X, k, z, 0, xin);
          auto L = T.L0(k, z, inu);
          for (long p = 0; p < nc2; ++p) {
            cplx v(0.0);
            for (long pp = 0; pp < nc2; ++pp) v += L(p, pp) * xin(pp / nc, pp % nc);
            ref(p / nc, p % nc) = v;
          }
          for (long i = 0; i < nc; ++i)
            for (long jj = 0; jj < nc; ++jj) {
              errL = std::max(errL, std::abs(got(i, jj) - ref(i, jj)));
              scL = std::max(scL, std::abs(ref(i, jj)));
            }
        }
      // and the exact frequency sum of F vs a dense sum
      double errS = 0.0, scS = 0.0;
      for (long k = 0; k < nk; ++k)
        for (long i = 0; i < nc; ++i)
          for (long jj = 0; jj < nc; ++jj) {
            const cplx ref = richardson_fsum(beta, 4000, 3, [&](cplx z) {
              nda::array<cplx, 2> g(nc, nc);
              eval_tf(F, k, z, 0, g);
              return g(i, jj);
            });
            errS = std::max(errS, std::abs(Fsum(k, i, jj, 0) - ref));
            scS = std::max(scS, std::abs(ref));
          }
      app_log(1, "dynbse (L) inu = {:.3f}i: l0_apply pointwise |err| {:.3e} (scale {:.3e}); frequency sum |err| "
                 "{:.3e} (scale {:.3e})", inu.imag(), errL, scL, errS, scS);
      REQUIRE(errL < 1e-9 * scL);
      REQUIRE(errS < 1e-7 * std::max(scS, 1e-3));
      // (K): F = (U_a - U_b) e_(0,1) at k' = 0 -> y = K_d F, pointwise vs conv_pole
      db::pair_rung R;
      R.nq = nk; R.nt = nt; R.nc2 = nc2; R.kmk = T.kmk;
      R.Wd_s = nda::array<cplx, 4>(nk, nt, nc2, nc2); R.Wd_s() = cplx(0.0);
      R.Wd0 = nda::array<cplx, 3>(nk, nc2, nc2); R.Wd0() = cplx(0.0);
      R.K0 = nda::array<cplx, 3>(nk, nc2, nc2);
      for (long iq = 0; iq < nk; ++iq) {
        for (long pp = 0; pp < T.npole; ++pp) {
          R.Wd0(iq, all_, all_) += T.M(pp, iq, all_, all_) * cplx(-2.0 / T.epsW(pp));
          for (long i = 0; i < nt; ++i)
            R.Wd_s(iq, i, all_, all_) += T.M(pp, iq, all_, all_) *
                                          cplx(db::KB(beta, b.s(i), T.epsW(pp)) - db::KB(beta, b.s(i), -T.epsW(pp)));
        }
        R.K0(iq, all_, all_) = T.Z(iq, all_, all_) + R.Wd0(iq, all_, all_);
      }
      const long ia = 5, ib = 11;
      db::tf_vector Fin(np, nk, nc, 1), y(np, nk, nc, 1);
      Fin.fam(0, ia, 0, 0, 1, 0) = cplx(1.0);
      Fin.fam(0, ib, 0, 0, 1, 0) = cplx(-1.0);
      nda::array<cplx, 4> Fs(nk, nc, nc, 1);
      Fs() = cplx(0.0);
      Fs(0, 0, 1, 0) = cplx(b.fd[size_t(ia)].f - b.fd[size_t(ib)].f);
      const double fe = db::kd_apply(b, R, Fin, Fs, y, inu);
      double errK = 0.0, scK = 0.0;
      for (long k = 0; k < nk; ++k)
        for (long n : {0l, 3l, 17l, -6l}) {
          const cplx z = I_ * cplx((2.0 * n + 1.0) * M_PI / beta);
          nda::array<cplx, 2> got(nc, nc);
          eval_tf(y, k, z, 0, got);
          const long iq = T.kmk(k, 0);
          nda::array<cplx, 1> e(nc2);
          e() = cplx(0.0);
          e(0 * nc + 1) = cplx(1.0);
          for (long p = 0; p < nc2; ++p) {
            cplx v(0.0);
            for (long pp = 0; pp < T.npole; ++pp) {
              const cplx cf = conv_pole(beta, z, b.eps(ia), T.epsW(pp)) - conv_pole(beta, z, b.eps(ib), T.epsW(pp));
              for (long q2 = 0; q2 < nc2; ++q2) v += T.M(pp, iq, p, q2) * cf * e(q2);
            }
            for (long q2 = 0; q2 < nc2; ++q2) v -= R.Wd0(iq, p, q2) * Fs(0, 0, 1, 0) * e(q2);
            errK = std::max(errK, std::abs(got(p / nc, p % nc) - v));
            scK = std::max(scK, std::abs(v));
          }
        }
      app_log(1, "dynbse (K) inu = {:.3f}i: kd_apply pointwise |err| {:.3e} (scale {:.3e}); tau refit err {:.3e}",
              inu.imag(), errK, scK, fe);
      REQUIRE(errK < 1e-6 * scK);
      // (G): the grouped l0_apply vs the per-(j,l) reference on random two-family inputs, on the
      // union set (no confluence) and on a shared set (G refit on the vertex grid: confluent
      // products through the Dsq/Dcb tables), with and without the Cb_cst override
      {
        std::mt19937 rng(7 + long(inu.imag() * 10));
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        auto rand_tf = [&](db::tf_vector &V) {
          for (auto &v : V.fam) v = cplx(u(rng), u(rng));
          for (auto &v : V.cst) v = cplx(u(rng), u(rng));
        };
        auto compare = [&](db::freq_basis const &bb, db::pair_poles const &PP, bool sh, const char *tag) {
          const long npp = bb.np;
          db::tf_vector Xr(npp, nk, nc, 3), Fa(npp, nk, nc, 3), Fb(npp, nk, nc, 3);
          nda::array<cplx, 4> Sa(nk, nc, nc, 3), Sb(nk, nc, nc, 3);
          rand_tf(Xr);
          // no components on the union extension (the exact G poles): a physical vector never
          // carries them (the refit targets the DLR part), and a confluent product there has no table
          for (long a = bb.np_fit; a < npp; ++a) Xr.fam(all_, a, all_, all_, all_, all_) = cplx(0.0);
          std::optional<db::shift_tables> stG;
          db::shift_tables const *stGp = nullptr;
          if (not nu0) { stG.emplace(db::build_shift_tables(bb, inu)); stGp = std::addressof(stG.value()); }
          // the reference is in the {U, S} basis: at inu != 0 its family-1 input means S_a, the
          // grouped code's means T_a = U_a S_a -- feed the reference the SAME FUNCTION by
          // converting T -> S: T_a = [U_a - S_a]/(i nu)  =>  x^T T_a = (x^T/(i nu)) U_a - (x^T/(i nu)) S_a
          db::tf_vector Xs(npp, nk, nc, 3);
          Xs.cst() = Xr.cst;
          Xs.fam() = Xr.fam;
          if (not nu0) {
            for (long a = 0; a < npp; ++a)
              for (long k = 0; k < nk; ++k)
                for (long x = 0; x < nc; ++x)
                  for (long y = 0; y < nc; ++y)
                    for (long rr = 0; rr < 3; ++rr) {
                      const cplx t = Xr.fam(1, a, k, x, y, rr) / inu;
                      Xs.fam(0, a, k, x, y, rr) = Xr.fam(0, a, k, x, y, rr) + t;
                      Xs.fam(1, a, k, x, y, rr) = -t;
                    }
          }
          db::l0_apply_ref(bb, PP, inu, sh, Xs, Fa, Sa);
          db::l0_apply(bb, PP, inu, sh, Xr, Fb, Sb, nullptr, stGp);
          double df = 0.0, sf = 0.0, ds = 0.0, ss = 0.0;
          if (nu0) {
            for (long i = 0; i < long(Fa.fam.size()); ++i) {
              df = std::max(df, std::abs(Fa.fam.data()[i] - Fb.fam.data()[i]));
              sf = std::max(sf, std::abs(Fa.fam.data()[i]));
            }
          } else {
            // pointwise values at a few fermionic frequencies (each output in its own basis)
            for (long k = 0; k < nk; ++k)
              for (long n : {0l, 2l, 9l, -5l, 40l})
                for (long rr = 0; rr < 3; ++rr) {
                  const cplx z = I_ * cplx((2.0 * n + 1.0) * M_PI / beta);
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) {
                      cplx va(0.0), vb(0.0);
                      for (long c = 0; c < npp; ++c) {
                        const cplx u = 1.0 / (z - bb.eps(c)), sS = 1.0 / (z + inu - bb.eps(c));
                        va += Fa.fam(0, c, k, x, y, rr) * u + Fa.fam(1, c, k, x, y, rr) * sS;
                        vb += Fb.fam(0, c, k, x, y, rr) * u + Fb.fam(1, c, k, x, y, rr) * u * sS;
                      }
                      df = std::max(df, std::abs(va - vb));
                      sf = std::max(sf, std::abs(va));
                    }
                }
          }
          for (long i = 0; i < long(Sa.size()); ++i) {
            ds = std::max(ds, std::abs(Sa.data()[i] - Sb.data()[i]));
            ss = std::max(ss, std::abs(Sa.data()[i]));
          }
          // the Cb_cst override: with Cb_cst = the pole-route Cb the sums must agree too
          nda::array<cplx, 3> Cbk(nk, nc2, nc2);
          Cbk() = cplx(0.0);
          for (long k = 0; k < nk; ++k)
            for (long j = 0; j < PP.ng; ++j)
              for (long l = 0; l < PP.ng; ++l) {
                const cplx Tjl = wl::two_pole(PP.fdG[size_t(j)], PP.epsG(j), PP.fdG[size_t(l)], PP.epsG(l), inu, j == l).T;
                for (long p1p = 0; p1p < nc; ++p1p)
                  for (long p3 = 0; p3 < nc; ++p3)
                    for (long a = 0; a < nc; ++a)
                      for (long bq = 0; bq < nc; ++bq)
                        Cbk(k, p1p * nc + p3, a * nc + bq) += Tjl * PP.gk(j, k, a, p1p) * PP.gkq(l, k, p3, bq);
              }
          db::tf_vector Fc(npp, nk, nc, 3);
          nda::array<cplx, 4> Sc(nk, nc, nc, 3);
          db::l0_apply(bb, PP, inu, sh, Xr, Fc, Sc, &Cbk, stGp);
          double dc = 0.0, dfc = 0.0;
          for (long i = 0; i < long(Sa.size()); ++i) dc = std::max(dc, std::abs(Sa.data()[i] - Sc.data()[i]));
          for (long i = 0; i < long(Fa.fam.size()); ++i) dfc = std::max(dfc, std::abs(Fb.fam.data()[i] - Fc.fam.data()[i]));
          app_log(1, "dynbse (G) inu = {:.3f}i [{}]: grouped vs ref |dF| {:.3e} (scale {:.3e}), |dFsum| {:.3e} "
                     "(scale {:.3e}); with Cb_cst: |dFsum| {:.3e}, |dF| {:.3e}", inu.imag(), tag, df, sf, ds, ss, dc, dfc);
          REQUIRE(df < (nu0 ? 1e-11 : 1e-7) * sf);           // inu != 0: the tables' fit class
          REQUIRE(ds < (nu0 ? 1e-11 : 1e-7) * std::max(ss, 1e-3));
          REQUIRE(dc < 1e-9 * std::max(ss, 1e-3));
          REQUIRE(dfc == 0.0);
        };
        compare(b, P, false, "union set");
        // the shared set: G refit on the aux grid (as run_solver's fitted path)
        {
          auto b2 = db::build_freq_basis(ft);
          const long np2 = b2.np;
          nda::array<cplx, 4> gk2(np2, nk, nc, nc), gkq2(np2, nk, nc, nc);
          for (long k = 0; k < nk; ++k) {
            nda::array<cplx, 2> Fq(nt, nc * nc);
            Fq() = cplx(0.0);
            auto const &tk = T.t[size_t(k)];
            for (long i = 0; i < nt; ++i)
              for (long j = 0; j < tk.ng; ++j) {
                const double Kf = imag_axes_ft::dlr_kF(beta, b2.s(i), tk.lam(j));
                for (long a = 0; a < nc; ++a)
                  for (long bq = 0; bq < nc; ++bq) Fq(i, a * nc + bq) += tk.g(j, a, bq) * Kf;
              }
            auto c = b2.pf.coeffs(Fq);
            for (long pq = 0; pq < np2; ++pq)
              for (long a = 0; a < nc; ++a)
                for (long bq = 0; bq < nc; ++bq) gk2(pq, k, a, bq) = c(pq, a * nc + bq);
          }
          for (long k = 0; k < nk; ++k) gkq2(all_, k, all_, all_) = gk2(all_, T.kpq[size_t(k)], all_, all_);
          auto P2 = db::make_pair_poles(beta, b2.eps, gk2, gkq2, 0);
          compare(b2, P2, true, "shared set");
        }
      }
    }
    for (cplx inu : {cplx(0.0), I_ * cplx(2.0 * M_PI * 2.0 / beta)}) {
      app_log(1, "dynbse: ---- inu = {:.4f} i ----", inu.imag());
      // the solver on the exact node set (fast) first
      // tolerance above the tau-refit noise floor (~1e-9 relative, the regularized pole fit)
      auto so = run_solver(T, ft, inu, false, 1e-8, 60);
      std::string hist;
      for (double h : so.res.history) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.1e", h);
        hist += buf;
      }
      app_log(1, "dynbse solver (exact nodes): np = {}, Dsq fit err {:.3e}, iterations {}, converged {}, "
                 "contraction {:.3f}, residual {:.3e}, tau refit err max {:.3e}; history:{}",
              db::build_freq_basis(ft).np, so.dsq_err, so.res.iterations, so.res.converged, so.res.contraction,
              so.res.residual, so.res.fit_err_max, hist);
      // the oracle's convergence in N (static kernel), then two sizes for the Richardson step
      {
        auto s128 = dense_bse(T, inu, 128, 0), s256 = dense_bse(T, inu, 256, 0);
        app_log(1, "dynbse oracle convergence (static): |P(128) - P0_solver| {:.3e}, |P(256) - P0_solver| {:.3e}, "
                   "|P(128) - P(256)| {:.3e}", rel_diff(s128, so.P0), rel_diff(s256, so.P0), rel_diff(s128, s256));
      }
      auto Pst_a = dense_bse(T, inu, 256, 0), Pst_b = dense_bse(T, inu, 512, 0);
      auto Pdy_a = dense_bse(T, inu, 256, 1), Pdy_b = dense_bse(T, inu, 512, 1);
      nda::array<cplx, 2> Pst(T.nR, T.nR), Pdy(T.nR, T.nR);
      Pst = (Pst_b * cplx(4.0) - Pst_a) / cplx(3.0);
      Pdy = (Pdy_b * cplx(4.0) - Pdy_a) / cplx(3.0);
      app_log(1, "dynbse oracle: static N=256 vs 512 rel diff {:.3e} (512 vs P0_solver {:.3e}); dynamic {:.3e}; "
                 "|P_dyn| {:.3e}, |P_dyn - P_static|/|P_static| {:.3e}",
              rel_diff(Pst_a, Pst_b), rel_diff(Pst_b, so.P0), rel_diff(Pdy_a, Pdy_b), max_abs(Pdy),
              rel_diff(Pdy, Pst));
      const double dA = rel_diff(so.P0, Pst), dC = rel_diff(so.P, Pdy), dC1 = rel_diff(so.P1, Pdy);
      app_log(1, "dynbse (A) static limit: solver vs oracle rel diff {:.3e}", dA);
      app_log(1, "dynbse (C) resummed:     solver vs oracle rel diff {:.3e}   (first iterate vs oracle {:.3e})",
              dC, dC1);
      REQUIRE(dA < 1e-9);
      REQUIRE(dC < 1e-7);
      REQUIRE(so.res.converged);
      REQUIRE(so.res.contraction < 1.0);
      // (E) GMRES(4) on the same problem: same answer, fewer operator applications
      {
        auto sg = run_solver(T, ft, inu, false, 1e-8, 40, 4);
        std::string hg;
        for (double h : sg.res.history) {
          char buf[32];
          std::snprintf(buf, sizeof(buf), " %.1e", h);
          hg += buf;
        }
        const double dG = rel_diff(sg.P, Pdy), dG1 = rel_diff(sg.P1, so.P1), dG0 = rel_diff(sg.P0, so.P0);
        app_log(1, "dynbse (E) GMRES(4): vs oracle {:.3e}; first iterate vs Neumann's {:.3e}, static {:.3e}; "
                   "operator applications {}, converged {}, max |Ritz(K_d L_s)| {:.3f}; cycle residuals:{}",
                dG, dG1, dG0, sg.res.iterations, sg.res.converged, sg.res.contraction, hg);
        REQUIRE(dG < 1e-7);
        REQUIRE(dG1 < 1e-12);
        REQUIRE(dG0 == 0.0);
        REQUIRE(sg.res.converged);
      }
      // (D) the fitted, shared-grid pathway
      auto sf = run_solver(T, ft, inu, true, 1e-8, 60);
      const double dAf = rel_diff(sf.P0, Pst), dCf = rel_diff(sf.P, Pdy);
      app_log(1, "dynbse (D) fitted/shared grid: static {:.3e}, resummed {:.3e}; iterations {}, tau refit "
                 "err max {:.3e}", dAf, dCf, sf.res.iterations, sf.res.fit_err_max);
      REQUIRE(dAf < 1e-6);
      REQUIRE(dCf < 1e-5);
    }
#endif
  }


  // ---------------------------------------------------------------------------------------
  // SMALL nu: the two-family bookkeeping at the smallest bosonic node of a long-beta grid.
  // The index-confluent pairs carry coefficients +-1/(i nu) that cancel only in VALUES; this
  // measures whether the solver (union set: no confluence; shared set: G refit on the vertex
  // grid) stays exact against the dense oracle when 1/nu = beta/(2 pi) is large.
  // ---------------------------------------------------------------------------------------
  TEST_CASE("dynbse_small_nu", "[methods][vertex][scgwt][dynbse]") {
    auto T = make_dyn_toy(0.10, 0.02);
    T.beta = 100.0;
    const double beta = T.beta;
    imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
    const cplx inu = I_ * cplx(2.0 * M_PI / beta);          // the first bosonic node, 1/nu = 15.9
    app_log(1, "dynbse small-nu: beta {}, inu = {:.4f}i (1/nu = {:.1f}), aux nodes {}", beta, inu.imag(),
            1.0 / inu.imag(), db::build_freq_basis(ft).np);
    auto Pdy_a = dense_bse(T, inu, 384, 1), Pdy_b = dense_bse(T, inu, 768, 1);
    nda::array<cplx, 2> Pdy(T.nR, T.nR);
    Pdy = (Pdy_b * cplx(4.0) - Pdy_a) / cplx(3.0);
    app_log(1, "dynbse small-nu oracle: N = 384 vs 768 rel diff {:.3e}, |P| {:.3e}", rel_diff(Pdy_a, Pdy_b), max_abs(Pdy));
    auto hist_of = [](db::dyson_result const &r) {
      std::string h;
      for (double x : r.history) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.1e", x);
        h += buf;
      }
      return h;
    };
    for (int fitted = 0; fitted < 2; ++fitted) {
      auto so = run_solver(T, ft, inu, bool(fitted), 1e-8, 40, 0);
      auto sg = run_solver(T, ft, inu, bool(fitted), 1e-8, 40, 4);
      app_log(1, "dynbse small-nu [{}]: Neumann vs oracle {:.3e} (Gamma1 {:.3e}), {} it, converged {}, contraction "
                 "{:.3e}, refit {:.3e}; history:{}", fitted ? "shared/fitted" : "union/exact", rel_diff(so.P, Pdy),
              rel_diff(so.P1, Pdy), so.res.iterations, so.res.converged, so.res.contraction, so.res.fit_err_max, hist_of(so.res));
      app_log(1, "dynbse small-nu [{}]: GMRES(4) vs oracle {:.3e}, {} applications, converged {}, max |Ritz| {:.3e}; "
                 "shift-table fit {:.2e}; history:{}", fitted ? "shared/fitted" : "union/exact", rel_diff(sg.P, Pdy),
              sg.res.iterations, sg.res.converged, sg.res.contraction, sg.shift_fit_err, hist_of(sg.res));
      // family scales of the first iterate's y: the 1/nu cancellation hazard made visible
      {
        auto b = db::build_freq_basis(ft);
        (void)b;
      }
      CHECK(rel_diff(sg.P, Pdy) < 1e-6);
    }
  }


  TEST_CASE("dynbse_small_nu_1000", "[methods][vertex][scgwt][dynbse][slow]") {
    auto T = make_dyn_toy(0.10, 0.02);
    T.beta = 1000.0;
    const double beta = T.beta;
    const cplx inu = I_ * cplx(2.0 * M_PI / beta);          // 1/nu = 159
    auto Pdy_a = dense_bse(T, inu, 1024, 1), Pdy_b = dense_bse(T, inu, 1536, 1);
    nda::array<cplx, 2> Pdy(T.nR, T.nR);
    // 1/N^3 tail: Richardson with (1536/1024)^3 = 3.375
    Pdy = (Pdy_b * cplx(3.375) - Pdy_a) / cplx(2.375);
    app_log(1, "dynbse small-nu(1000) oracle: N = 1024 vs 1536 rel diff {:.3e}, |P| {:.3e}", rel_diff(Pdy_a, Pdy_b), max_abs(Pdy));
    auto hist_of = [](db::dyson_result const &r) {
      std::string h;
      for (double x : r.history) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.1e", x);
        h += buf;
      }
      return h;
    };
    for (std::string prec : {"high", "medium", "low"}) {
      imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, prec);
      for (int fitted = 0; fitted < 2; ++fitted) {
        auto so = run_solver(T, ft, inu, bool(fitted), 1e-8, 30, 0);
        auto sg = run_solver(T, ft, inu, bool(fitted), 1e-8, 30, 4);
        app_log(1, "dynbse small-nu(1000) prec {} [{}]: Neumann vs oracle {:.3e}, {} it, converged {}, contraction "
                   "{:.3e}, refit {:.3e}; history:{}", prec, fitted ? "shared/fitted" : "union/exact",
                rel_diff(so.P, Pdy), so.res.iterations, so.res.converged, so.res.contraction, so.res.fit_err_max, hist_of(so.res));
        app_log(1, "dynbse small-nu(1000) prec {} [{}]: GMRES(4) vs oracle {:.3e}, {} applications, converged {}, "
                   "max |Ritz| {:.3e}; history:{}", prec, fitted ? "shared/fitted" : "union/exact", rel_diff(sg.P, Pdy),
                sg.res.iterations, sg.res.converged, sg.res.contraction, hist_of(sg.res));
      }
    }
    SUCCEED("measured");
  }


  // ---------------------------------------------------------------------------------------
  // (T) the twisted-pair tables: R1_j = U_j^2 S_j and R3_j = U_j^2 S_j^2 evaluated at Matsubara
  // points from the tables vs their closed forms, at beta 10 / 100 / 1000 (nu = 2pi/beta).
  // ---------------------------------------------------------------------------------------
  TEST_CASE("dynbse_shift_tables", "[methods][vertex][scgwt][dynbse]") {
    for (double beta : {10.0, 100.0, 1000.0}) {
      imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
      auto b = db::build_freq_basis(ft);
      const cplx inu = I_ * cplx(2.0 * M_PI / beta);
      auto st = db::build_shift_tables(b, inu);
      const long np = b.np;
      double e1 = 0.0, e3 = 0.0, c1 = 0.0, c3 = 0.0, x1 = 0.0, x3 = 0.0;
      long jworst1 = -1, jworst3 = -1;
      for (long j = 0; j < np; ++j) {
        const double ej = b.eps(j);
        double s1 = 0.0, s3 = 0.0, d1 = 0.0, d3 = 0.0;
        for (long n : {0l, 1l, 3l, 10l, 40l, -2l, -20l}) {
          const cplx z = I_ * cplx((2.0 * n + 1.0) * M_PI / beta);
          const cplx u = 1.0 / (z - ej), sS = 1.0 / (z + inu - ej);
          const cplx r1 = u * u * sS, r3 = u * u * sS * sS;
          cplx t1 = st.r1u(j) * u, t3 = st.r3u(j) * u + st.r3t(j) * u * sS;
          for (long c = 0; c < np; ++c) {
            const cplx uc = 1.0 / (z - b.eps(c)), tc = uc / (z + inu - b.eps(c));
            t1 += st.R1U(j, c) * uc + st.R1T(j, c) * tc;
            t3 += st.R3U(j, c) * uc + st.R3T(j, c) * tc;
          }
          d1 = std::max(d1, std::abs(t1 - r1)); s1 = std::max(s1, std::abs(r1));
          d3 = std::max(d3, std::abs(t3 - r3)); s3 = std::max(s3, std::abs(r3));
        }
        if (d1 / s1 > e1) { e1 = d1 / s1; jworst1 = j; }
        if (d3 / s3 > e3) { e3 = d3 / s3; jworst3 = j; }
        for (long c = 0; c < np; ++c) {
          c1 = std::max({c1, std::abs(st.R1U(j, c)), std::abs(st.R1T(j, c))});
          c3 = std::max({c3, std::abs(st.R3U(j, c)), std::abs(st.R3T(j, c))});
        }
        x1 = std::max(x1, std::abs(st.r1u(j)));
        x3 = std::max({x3, std::abs(st.r3u(j)), std::abs(st.r3t(j))});
      }
      app_log(1, "dynbse (T) beta {}: np {}, tables fit err {:.2e} (kept cond {:.2e}); R1 max rel err {:.2e} (node {} eps {:.3e}), "
                 "R3 {:.2e} (node {} eps {:.3e}); max |exact| R1 {:.2e} R3 {:.2e}; max |fit coeff| R1 {:.2e} R3 {:.2e}",
              beta, np, st.fit_err, st.cond_kept, e1, jworst1, b.eps(jworst1), e3, jworst3, b.eps(jworst3), x1, x3, c1, c3);
      CHECK(e1 < 1e-6);
      CHECK(e3 < 1e-3);      // the order-4 object at the far nodes: 8e-5 at beta 1000 (tiny in absolute terms)
    }
  }


  // shared vs union at small nu WITHOUT the oracle: the union solution is the reference (it has no
  // confluent product at all); the shared path with the fitted tables (mode 0) and with the exact
  // 1/nu-amplified Dsq tables (mode 1).
  TEST_CASE("dynbse_small_nu_shared", "[methods][vertex][scgwt][dynbse]") {
    for (double beta : {100.0, 1000.0}) {
      auto T = make_dyn_toy(0.10, 0.02);
      T.beta = beta;
      const cplx inu = I_ * cplx(2.0 * M_PI / beta);
      imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
      auto hist_of = [](db::dyson_result const &r) {
        std::string h;
        for (double x : r.history) { char buf[32]; std::snprintf(buf, sizeof(buf), " %.1e", x); h += buf; }
        return h;
      };
      auto su = run_solver(T, ft, inu, false, 1e-9, 40, 4, 0);
      app_log(1, "dynbse shared-vs-union beta {}: union GMRES(4) {} applications, converged {}, Ritz {:.3e}; history:{}",
              beta, su.res.iterations, su.res.converged, su.res.contraction, hist_of(su.res));
      for (int mode = 0; mode < 2; ++mode) {
        auto sg = run_solver(T, ft, inu, true, 1e-9, 40, 4, mode);
        auto sn = run_solver(T, ft, inu, true, 1e-9, 40, 0, mode);
        app_log(1, "dynbse shared-vs-union beta {} tables mode {} ({}): GMRES vs union {:.3e} (static {:.3e}, Gamma1 {:.3e}), "
                   "{} applications, converged {}, stagnated {}, Ritz {:.3e}, table err {:.2e}; Neumann vs union {:.3e}, "
                   "contraction {:.3e}; GMRES history:{}",
                beta, mode, mode ? "exact/Dsq" : "fitted", rel_diff(sg.P, su.P), rel_diff(sg.P0, su.P0), rel_diff(sg.P1, su.P1),
                sg.res.iterations, sg.res.converged, sg.res.stagnated, sg.res.contraction, sg.shift_fit_err,
                rel_diff(sn.P, su.P), sn.res.contraction, hist_of(sg.res));
      }
    }
    SUCCEED("measured");
  }


  // the PRODUCTION union scheme on the toy: G refit on the shifted grid vs the exact-pole union
  // reference, at beta 100 / 1000 and inu = 0 / 2pi/beta (no oracle: the exact union IS the reference)
  TEST_CASE("dynbse_union_fitted", "[methods][vertex][scgwt][dynbse]") {
    for (double beta : {100.0, 1000.0}) {
      auto T = make_dyn_toy(0.10, 0.02);
      T.beta = beta;
      imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
      auto hist_of = [](db::dyson_result const &r) {
        std::string h;
        for (double x : r.history) { char buf[32]; std::snprintf(buf, sizeof(buf), " %.1e", x); h += buf; }
        return h;
      };
      for (cplx inu : {cplx(0.0), I_ * cplx(2.0 * M_PI / beta)}) {
        auto su = run_solver(T, ft, inu, 0, 1e-9, 40, 4);
        auto sf = run_solver(T, ft, inu, 2, 1e-9, 40, 4);
        auto sn = run_solver(T, ft, inu, 2, 1e-9, 40, 0);
        app_log(1, "dynbse union/fitted beta {} inu {:.4f}i: vs exact union -- static {:.3e}, Gamma1 {:.3e}, resummed "
                   "{:.3e} (Neumann {:.3e}); GMRES {} applications, converged {}, stagnated {}, Ritz {:.3e} (exact union "
                   "Ritz {:.3e}, {} applications); refit {:.2e}; history:{}",
                beta, inu.imag(), rel_diff(sf.P0, su.P0), rel_diff(sf.P1, su.P1), rel_diff(sf.P, su.P), rel_diff(sn.P, su.P),
                sf.res.iterations, sf.res.converged, sf.res.stagnated, sf.res.contraction, su.res.contraction,
                su.res.iterations, sf.res.fit_err_max, hist_of(sf.res));
        CHECK(rel_diff(sf.P, su.P) < 1e-5);
      }
    }
  }

} // namespace bdft_tests
