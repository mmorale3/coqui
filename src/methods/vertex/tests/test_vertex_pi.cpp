/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

/**
 * Tests for the ISDF-Vertex Pi^C kernel (Phase 1d), see notes/pi_c_kernel_design.md.
 *
 *  - residue_calibration: pins the Matsubara residue algebra of the dense reference
 *    against explicit truncated sums (self-calibration of the arbiter).
 *  - pin_rpa_bubble: reproduces the code's RPA tau-Hadamard bubble (rpa_pi.icc:305-336
 *    formula) through the Pi^C frequency machinery -- pins the code-Pi sign/index/mirror
 *    mapping chain of the design doc (section 2).
 *  - dense_cross_check: brute-force orbital/pole-space evaluation of the verified Eq. 13
 *    (prefactor -1) with analytic bosonic rung sums and an explicit fermionic frequency
 *    window, vs the THC kernel. The arbiter test.
 *  - structure: chi(beta-tau) = chi^T(tau) on the kernel output; hermiticity reported.
 *  - lih_smoke: one 2-iteration THC-scGW run on LiH-222 (nosym) with the vertex active:
 *    finite results, stable W-Dyson, vertex visibly changes e_corr.
 */

#undef NDEBUG

#include <complex>
#include <cmath>
#include <random>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_pi.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  namespace vertex_pi = methods::solvers::vertex_pi;
  using vertex_pi::iaft_tools;
  using cplx = ComplexType;
  static auto const t_all = nda::range::all;

  namespace toy {

    constexpr double beta = 20.0;
    constexpr double wmax = 8.0;
    constexpr double Omega = 0.61;   // toy W_dyn pole
    constexpr long nk = 3, nbnd = 3, Np = 4, ns = 1;
    inline nda::range C() { return nda::range(0, 2); }

    struct rng_t {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dist{-0.5, 0.5};
      explicit rng_t(unsigned seed) : gen(seed) {}
      cplx operator()() { return cplx(dist(gen), dist(gen)); }
    };

    // unitary from two Householder reflections
    inline nda::array<cplx, 2> unitary(long n, rng_t& rng) {
      auto refl = [&](nda::array<cplx, 1> const& v) {
        nda::array<cplx, 2> H(n, n);
        double nv = 0;
        for (long i = 0; i < n; ++i) nv += std::norm(v(i));
        for (long i = 0; i < n; ++i)
          for (long j = 0; j < n; ++j)
            H(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) - 2.0 * v(i) * std::conj(v(j)) / nv;
        return H;
      };
      nda::array<cplx, 1> v1(n), v2(n);
      for (long i = 0; i < n; ++i) { v1(i) = rng(); v2(i) = rng(); }
      nda::array<cplx, 2> U(n, n);
      nda::blas::gemm(refl(v1), refl(v2), U);
      return U;
    }

    inline double nF(double x) { return 1.0 / (std::exp(beta * x) + 1.0); }

    // stable -e^{-eps*tau} (1 - nF(eps)) for tau in [0,beta]
    inline double g_tau(double eps, double tau) {
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-beta * eps));
      return -std::exp(eps * (beta - tau)) / (1.0 + std::exp(beta * eps));
    }

    struct model_t {
      nda::array<double, 2> eps;       // (nk, nbnd)
      nda::array<cplx, 4> Pr;          // (nk, nbnd[r], nbnd, nbnd) pole matrices U_r U_r^dag
      nda::array<cplx, 4> X_skPa;      // (ns, nk, Np, nbnd)
      nda::array<cplx, 3> Z_qPQ;       // (nk, Np, Np)
      nda::array<cplx, 3> M_qPQ;       // (nk, Np, Np)  W_dyn(q, inu) = M_q * s(nu)
      nda::array<long, 2> kmq, kpq;    // (nq, nk)

      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd),
                  X_skPa(ns, nk, Np, nbnd), Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np),
                  kmq(nk, nk), kpq(nk, nk) {
        rng_t rng(17);
        const double base[3] = {-0.71, -0.13, 0.47};
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.07 * double(k);
          auto U = unitary(nbnd, rng);
          for (long r = 0; r < nbnd; ++r)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                Pr(k, r, i, j) = U(i, r) * std::conj(U(j, r));
        }
        for (long k = 0; k < nk; ++k)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < nbnd; ++a) X_skPa(0, k, P, a) = rng();
        for (long q = 0; q < nk; ++q) {
          nda::array<cplx, 2> Y(Np, Np), V(Np, 2);
          for (auto& y : Y) y = rng();
          for (auto& v : V) v = rng();
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              cplx zy = 0, zv = 0;
              for (long r = 0; r < Np; ++r) zy += Y(P, r) * std::conj(Y(Q, r));
              for (long r = 0; r < 2; ++r) zv += V(P, r) * std::conj(V(Q, r));
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.3) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        for (long q = 0; q < nk; ++q)
          for (long k = 0; k < nk; ++k) {
            kmq(q, k) = (k - q + nk) % nk;
            kpq(q, k) = (k + q) % nk;
          }
      }

      // G(k, tau) on the IAFT tau mesh: (nt, ns, nk, nbnd, nbnd)
      nda::array<cplx, 5> G_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f();
        auto xm = ft.tau_mesh();
        nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
        G() = cplx(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * beta;
          for (long k = 0; k < nk; ++k)
            for (long r = 0; r < nbnd; ++r) {
              double g = g_tau(eps(k, r), tau);
              for (long i = 0; i < nbnd; ++i)
                for (long j = 0; j < nbnd; ++j)
                  G(it, 0, k, i, j) += g * Pr(k, r, i, j);
            }
        }
        return G;
      }

      // W_dyn on the full bosonic mesh
      nda::array<cplx, 4> Wdyn(iaft_tools const& tools) const {
        nda::array<cplx, 4> W(nk, tools.nw_b, Np, Np);
        for (long q = 0; q < nk; ++q)
          for (long l = 0; l < tools.nw_b; ++l) {
            double nu = double(tools.wn_b(l)) * M_PI / beta;
            double s = 2.0 * Omega / (Omega * Omega + nu * nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) W(q, l, P, Q) = M_qPQ(q, P, Q) * s;
          }
        return W;
      }
    };

    // ---------------- residue algebra for the dense reference ----------------

    // pole with real part `re`, Matsubara-integer imaginary part of parity `odd`,
    // full complex value `val`
    struct pole_t { cplx val; double re; bool odd; };

    inline cplx nB_pole(pole_t const& p) {
      double s = p.odd ? -1.0 : 1.0;
      return cplx(1.0) / (s * std::exp(beta * p.re) - 1.0);
    }
    inline cplx nF_pole(pole_t const& p) {
      double s = p.odd ? -1.0 : 1.0;   // parity of the *fermionic* label offset
      // fermionic occupation at p: 1/(e^{beta p}+1) with e^{i beta Im p} = -s ... for the
      // poles used below the label offsets are always bosonic (even), so s = +1.
      (void)s;
      return cplx(1.0) / (std::exp(beta * p.re) + 1.0);
    }

    // (1/beta) sum_{bosonic nu} prod_i 1/(i nu - p_i) = -sum_i nB(p_i) prod_{j!=i} 1/(p_i-p_j)
    inline cplx msum_boson(std::vector<pole_t> const& p) {
      cplx acc = 0;
      for (size_t i = 0; i < p.size(); ++i) {
        cplx res = nB_pole(p[i]);
        for (size_t j = 0; j < p.size(); ++j)
          if (j != i) res /= (p[i].val - p[j].val);
        acc -= res;
      }
      return acc;
    }

    // (1/beta) sum_{fermionic w} prod_i 1/(i w - p_i) = +sum_i nF(p_i) prod_{j!=i} ...
    inline cplx msum_fermi(std::vector<pole_t> const& p) {
      cplx acc = 0;
      for (size_t i = 0; i < p.size(); ++i) {
        cplx res = nF_pole(p[i]);
        for (size_t j = 0; j < p.size(); ++j)
          if (j != i) res /= (p[i].val - p[j].val);
        acc += res;
      }
      return acc;
    }

  } // namespace toy

  TEST_CASE("vertex_pi_toy", "[methods][vertex][pi_c]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_pi_toy skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace toy;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-11);
    iaft_tools tools(ft);
    model_t mdl;
    auto G = mdl.G_tau(ft);
    auto Wdyn = mdl.Wdyn(tools);
    const long nw_b = tools.nw_b, nt = tools.nt_f;
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    const long m0 = tools.m0;
    const long Nb = C().size();
    const cplx I(0.0, 1.0);

    SECTION("residue_calibration") {
      // scalar poles typical of the reference
      pole_t P{I * double(15) * M_PI / beta - (-0.13), 0.13, true};
      pole_t Q{I * double(15 + 4) * M_PI / beta - 0.47, -0.47, true};
      pole_t Wp{cplx(Omega), Omega, false};
      pole_t Wm{cplx(-Omega), -Omega, false};

      auto brute_boson = [&](std::vector<pole_t> const& p, long L) {
        cplx acc = 0;
        for (long l = -L; l <= L; ++l) {
          cplx z = I * double(2 * l) * M_PI / beta;
          cplx t = 1.0;
          for (auto const& pp : p) t /= (z - pp.val);
          acc += t;
        }
        return acc / beta;
      };
      // 3-pole: absolutely convergent ~1/nu^3
      {
        auto exact = msum_boson({P, Q, Wp});
        auto brute = brute_boson({P, Q, Wp}, 40000);
        REQUIRE(std::abs(exact - brute) < 1e-5 * std::max(1.0, std::abs(exact)));
      }
      {
        auto exact = msum_boson({P, Q, Wm});
        auto brute = brute_boson({P, Q, Wm}, 40000);
        REQUIRE(std::abs(exact - brute) < 1e-5 * std::max(1.0, std::abs(exact)));
      }
      // 2-pole: ~1/nu^2, Richardson in 1/L
      {
        auto exact = msum_boson({P, Q});
        auto s1 = brute_boson({P, Q}, 50000);
        auto s2 = brute_boson({P, Q}, 100000);
        auto extrap = 2.0 * s2 - s1;
        REQUIRE(std::abs(exact - extrap) < 1e-5 * std::max(1.0, std::abs(exact)));
      }
      // fermionic 2-pole vs brute
      {
        pole_t a{cplx(-0.71), -0.71, false};
        pole_t b{cplx(0.47) - I * 2.0 * M_PI / beta, 0.47, false};
        auto exact = msum_fermi({a, b});
        cplx s1 = 0, s2 = 0;
        for (long j = -100000; j < 100000; ++j) {
          cplx z = I * double(2 * j + 1) * M_PI / beta;
          cplx t = 1.0 / ((z - a.val) * (z - b.val));
          if (std::abs(j) < 50000) s1 += t;
          s2 += t;
        }
        s1 /= beta; s2 /= beta;
        auto extrap = 2.0 * s2 - s1;
        REQUIRE(std::abs(exact - extrap) < 1e-5 * std::max(1.0, std::abs(exact)));
      }
      // consistency: bosonic sum over shifted fermionic pair == fermionic bubble
      {
        long nw = 15;   // fermionic label
        long m = 4;     // bosonic label
        cplx z = I * double(nw) * M_PI / beta;
        cplx inu = I * double(m) * M_PI / beta;
        double e3 = -0.13, e2 = 0.47;
        pole_t Pp{z - e3, -e3, true};
        pole_t Qp{z + inu - e2, -e2, true};
        auto via_boson = msum_boson({Pp, Qp});
        pole_t a{cplx(e3), e3, false};
        pole_t b{cplx(e2) - inu, e2, false};
        auto via_fermi = msum_fermi({a, b});
        REQUIRE(std::abs(via_boson - via_fermi) < 1e-12);
      }
    }

    SECTION("pin_rpa_bubble") {
      // aux-dressed G on tau
      nda::array<cplx, 4> Gt_kPQ(nk, nt, Np, Np);
      nda::array<cplx, 2> tmp(Np, nbnd);
      for (long k = 0; k < nk; ++k) {
        auto X = mdl.X_skPa(0, k, t_all, t_all);
        for (long it = 0; it < nt; ++it) {
          auto Gv = G(it, 0, k, t_all, t_all);
          auto out = Gt_kPQ(k, it, t_all, t_all);
          nda::blas::gemm(X, Gv, tmp);
          nda::blas::gemm(tmp, nda::dagger(X), out);
        }
      }

      // (a) the code's tau-Hadamard bubble (rpa_pi.icc:305-336 with conj(Gtilde)=Gtilde^T)
      nda::array<cplx, 4> Pi_code(nt_half, nk, Np, Np);
      Pi_code() = cplx(0.0);
      for (long it = 0; it < nt_half; ++it)
        for (long q = 0; q < nk; ++q)
          for (long k = 0; k < nk; ++k) {
            long ikmq = mdl.kmq(q, k);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q)
                Pi_code(it, q, P, Q) += (-2.0 / double(nk)) *
                    std::conj(Gt_kPQ(ikmq, it, P, Q)) * Gt_kPQ(k, nt - it - 1, P, Q);
          }

      // (b) the same object through the Pi^C pairing primitive (design doc section 2/4):
      //     Pi_notes(inu) = +(2/Nk)(1/beta) sum_{k,w} Gt(k+q, w+nu)_PQ Gt(k, w)_QP
      //                   = -(2/Nk) sum_k [Twt_bb . prod](m),
      //     prod(s)_PQ = Gt(k+q, s)_PQ * Gt(k, beta-s)_QP,
      //     then w_to_tau and the beta-tau mirror.
      nda::array<cplx, 4> Pi_w(nw_b, nk, Np, Np);
      Pi_w() = cplx(0.0);
      nda::array<cplx, 2> prod(nt, Np * Np), bub(nw_b, Np * Np);
      for (long q = 0; q < nk; ++q)
        for (long k = 0; k < nk; ++k) {
          long ikpq = mdl.kpq(q, k);
          for (long s = 0; s < nt; ++s) {
            long sm = tools.t_mirror(s);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q)
                prod(s, P * Np + Q) = Gt_kPQ(ikpq, s, P, Q) * Gt_kPQ(k, sm, Q, P);
          }
          nda::blas::gemm(cplx(-2.0 / double(nk)), tools.Twt_bb, prod, cplx(0.0), bub);
          for (long m = 0; m < nw_b; ++m)
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q)
                Pi_w(m, q, P, Q) += bub(m, P * Np + Q);
        }
      nda::array<cplx, 4> Pi_my(nt_half, nk, Np, Np);
      vertex_pi::pi_w_to_code_tau(ft, tools, Pi_w, Pi_my);

      double num = 0, den = 0;
      for (long it = 0; it < nt_half; ++it)
        for (long q = 0; q < nk; ++q)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              num = std::max(num, std::abs(Pi_my(it, q, P, Q) - Pi_code(it, q, P, Q)));
              den = std::max(den, std::abs(Pi_code(it, q, P, Q)));
            }
      app_log(1, "pin_rpa_bubble: max|Pi_my - Pi_code| = {}, max|Pi_code| = {}", num, den);
      REQUIRE(den > 1e-8);
      REQUIRE(num < 1e-7 * den);
    }

    SECTION("dense_cross_check") {
      // ---- kernel: full rung sum, and the production q->0 policy (skip qx = Gamma) ----
      nda::array<cplx, 4> Pi_kern(nw_b, nk, Np, Np), Pi_kern_g0(nw_b, nk, Np, Np);
      Pi_kern() = cplx(0.0);
      Pi_kern_g0() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wdyn,
                                   mdl.kmq, mdl.kpq, C(), Pi_kern, 0, 1);
      nda::array<double, 1> qx_diag(nk);
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wdyn,
                                   mdl.kmq, mdl.kpq, C(), Pi_kern_g0, 0, 1,
                                   /*skip_rung_gamma=*/true, &qx_diag);
      REQUIRE(qx_diag(0) == 0.0);   // toy Gamma is qx = 0 (kmq(0,k) = k): skipped
      REQUIRE(qx_diag(1) > 0.0);

      // ---- dense reference ---------------------------------------------------------
      const double pref = -2.0 / double(nk * nk);   // ns==1 spin sum
      const long Nw = 6000;                          // fermionic window: n = 2j+1, |j| < Nw

      auto leg_L = [&](long k, long r) {   // X(k) . P_r(:,C)   (Np x Nb)
        nda::array<cplx, 2> out(Np, Nb);
        out() = cplx(0.0);
        for (long P = 0; P < Np; ++P)
          for (long j = 0; j < Nb; ++j)
            for (long a = 0; a < nbnd; ++a)
              out(P, j) += mdl.X_skPa(0, k, P, a) * mdl.Pr(k, r, a, C().first() + j);
        return out;
      };
      auto leg_R = [&](long k, long r) {   // P_r(C,:) . X(k)^dag   (Nb x Np)
        nda::array<cplx, 2> out(Nb, Np);
        out() = cplx(0.0);
        for (long j = 0; j < Nb; ++j)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < nbnd; ++a)
              out(j, P) += mdl.Pr(k, r, C().first() + j, a) * std::conj(mdl.X_skPa(0, k, P, a));
        return out;
      };

      // orbital-projected rung [(p1 p1'),(p3 p3')] of a given aux matrix
      auto rung = [&](nda::array<cplx, 2> const& WPQ, long ikmqx, long k, long ikpq, long ikxq) {
        nda::array<cplx, 4> out(Nb, Nb, Nb, Nb);
        out() = cplx(0.0);
        long c0 = C().first();
        for (long j1 = 0; j1 < Nb; ++j1)
          for (long j1p = 0; j1p < Nb; ++j1p)
            for (long j3 = 0; j3 < Nb; ++j3)
              for (long j3p = 0; j3p < Nb; ++j3p)
                for (long P = 0; P < Np; ++P)
                  for (long Q = 0; Q < Np; ++Q)
                    out(j1, j1p, j3, j3p) += mdl.X_skPa(0, ikmqx, P, c0 + j1) *
                        std::conj(mdl.X_skPa(0, k, P, c0 + j1p)) * WPQ(P, Q) *
                        mdl.X_skPa(0, ikpq, Q, c0 + j3) *
                        std::conj(mdl.X_skPa(0, ikxq, Q, c0 + j3p));
        return out;
      };

      // test points (q, m): avoid the (q=0, nu=0) exact double pole in the *reference*
      std::vector<std::pair<long, long>> points = {
          {0, m0 + 1}, {0, nw_b - 1}, {1, m0}, {1, m0 + 2}, {2, m0}, {2, m0 + 1}};

      double max_rel = 0.0;
      for (auto [iqe, m] : points) {
        cplx inu = I * double(tools.wn_b(m)) * M_PI / beta;
        nda::array<cplx, 2> acc(Np, Np), acc_g0(Np, Np), contrib(Np, Np);
        acc() = cplx(0.0);
        acc_g0() = cplx(0.0);

        for (long k = 0; k < nk; ++k)
          for (long qx = 0; qx < nk; ++qx) {
            contrib() = cplx(0.0);
            long ikmqx = mdl.kmq(qx, k);
            long ikpq = mdl.kpq(iqe, k);
            long ikxq = mdl.kmq(qx, ikpq);

            std::vector<nda::array<cplx, 2>> ZL4, ZR1, R3, L2;
            for (long r = 0; r < nbnd; ++r) {
              ZL4.push_back(leg_L(k, r));
              ZR1.push_back(leg_R(ikpq, r));
              R3.push_back(leg_R(ikmqx, r));
              L2.push_back(leg_L(ikxq, r));
            }
            auto A0 = rung(nda::array<cplx, 2>(mdl.Z_qPQ(qx, t_all, t_all)), ikmqx, k, ikpq, ikxq);
            auto AW = rung(nda::array<cplx, 2>(mdl.M_qPQ(qx, t_all, t_all)), ikmqx, k, ikpq, ikxq);

            // ---- instantaneous (Z) part: product of two analytic fermionic bubbles ---
            nda::array<cplx, 3> B41(Nb, Nb, Np), B23(Nb, Nb, Np);   // (p1',p3,N), (p1,p3',M)
            B41() = cplx(0.0); B23() = cplx(0.0);
            for (long r = 0; r < nbnd; ++r)
              for (long t = 0; t < nbnd; ++t) {
                pole_t a4{cplx(mdl.eps(k, r)), mdl.eps(k, r), false};
                pole_t b1{cplx(mdl.eps(ikpq, t)) - inu, mdl.eps(ikpq, t), false};
                cplx c41 = msum_fermi({a4, b1});
                pole_t a3{cplx(mdl.eps(ikmqx, r)), mdl.eps(ikmqx, r), false};
                pole_t b2{cplx(mdl.eps(ikxq, t)) - inu, mdl.eps(ikxq, t), false};
                cplx c23 = msum_fermi({a3, b2});
                for (long jp = 0; jp < Nb; ++jp)
                  for (long j3 = 0; j3 < Nb; ++j3)
                    for (long N = 0; N < Np; ++N)
                      B41(jp, j3, N) += c41 * ZL4[r](N, jp) * ZR1[t](j3, N);
                for (long j1 = 0; j1 < Nb; ++j1)
                  for (long j3p = 0; j3p < Nb; ++j3p)
                    for (long M = 0; M < Np; ++M)
                      B23(j1, j3p, M) += c23 * R3[r](j1, M) * L2[t](M, j3p);
              }
            for (long j1 = 0; j1 < Nb; ++j1)
              for (long j1p = 0; j1p < Nb; ++j1p)
                for (long j3 = 0; j3 < Nb; ++j3)
                  for (long j3p = 0; j3p < Nb; ++j3p)
                    for (long M = 0; M < Np; ++M)
                      for (long N = 0; N < Np; ++N)
                        contrib(M, N) += B41(j1p, j3, N) * A0(j1, j1p, j3, j3p) * B23(j1, j3p, M);

            // ---- dynamic part: analytic in nu_x, explicit fermionic window in w ------
            nda::array<cplx, 3> SM(Nb, Nb, Np);    // (p1, p3', M)
            nda::array<cplx, 3> TN(Nb, Nb, Np);    // (p1', p3, N)
            for (long j = -Nw; j < Nw; ++j) {
              cplx z = I * double(2 * j + 1) * M_PI / beta;
              // legs at z and z + inu
              nda::array<cplx, 2> zL4(Np, Nb), zR1(Nb, Np);
              zL4() = cplx(0.0); zR1() = cplx(0.0);
              for (long r = 0; r < nbnd; ++r) {
                cplx d4 = 1.0 / (z - mdl.eps(k, r));
                cplx d1 = 1.0 / (z + inu - mdl.eps(ikpq, r));
                for (long P = 0; P < Np; ++P)
                  for (long jj = 0; jj < Nb; ++jj) {
                    zL4(P, jj) += d4 * ZL4[r](P, jj);
                    zR1(jj, P) += d1 * ZR1[r](jj, P);
                  }
              }
              SM() = cplx(0.0);
              for (long r = 0; r < nbnd; ++r)
                for (long s = 0; s < nbnd; ++s) {
                  pole_t Pp{z - mdl.eps(ikmqx, r), -mdl.eps(ikmqx, r), true};
                  pole_t Qp{z + inu - mdl.eps(ikxq, s), -mdl.eps(ikxq, s), true};
                  pole_t Wp{cplx(Omega), Omega, false};
                  pole_t Wm{cplx(-Omega), -Omega, false};
                  cplx coef = -msum_boson({Pp, Qp, Wp}) + msum_boson({Pp, Qp, Wm});
                  for (long j1 = 0; j1 < Nb; ++j1)
                    for (long j3p = 0; j3p < Nb; ++j3p)
                      for (long M = 0; M < Np; ++M)
                        SM(j1, j3p, M) += coef * R3[r](j1, M) * L2[s](M, j3p);
                }
              for (long jp = 0; jp < Nb; ++jp)
                for (long j3 = 0; j3 < Nb; ++j3)
                  for (long N = 0; N < Np; ++N)
                    TN(jp, j3, N) = zL4(N, jp) * zR1(j3, N);
              for (long j1 = 0; j1 < Nb; ++j1)
                for (long j1p = 0; j1p < Nb; ++j1p)
                  for (long j3 = 0; j3 < Nb; ++j3)
                    for (long j3p = 0; j3p < Nb; ++j3p)
                      for (long M = 0; M < Np; ++M)
                        for (long N = 0; N < Np; ++N)
                          contrib(M, N) += (1.0 / beta) * TN(j1p, j3, N) *
                                           AW(j1, j1p, j3, j3p) * SM(j1, j3p, M);
            } // omega window

            acc += contrib;
            if (qx != 0) acc_g0 += contrib;   // Gamma-skipped reference
          } // k, qx

        double num = 0, den = 0, num_g0 = 0, den_g0 = 0;
        for (long M = 0; M < Np; ++M)
          for (long N = 0; N < Np; ++N) {
            cplx ref = pref * acc(M, N);
            num = std::max(num, std::abs(Pi_kern(m, iqe, M, N) - ref));
            den = std::max(den, std::abs(ref));
            cplx ref_g0 = pref * acc_g0(M, N);
            num_g0 = std::max(num_g0, std::abs(Pi_kern_g0(m, iqe, M, N) - ref_g0));
            den_g0 = std::max(den_g0, std::abs(ref_g0));
          }
        app_log(1, "dense_cross_check (q={}, m={}, n_b={}): full max|kern-ref| = {} (max|ref| = {}), "
                   "g0-skipped max|kern-ref| = {} (max|ref| = {})",
                iqe, m, tools.wn_b(m), num, den, num_g0, den_g0);
        REQUIRE(den > 1e-10);
        REQUIRE(den_g0 > 1e-10);
        max_rel = std::max(max_rel, num / den);
        max_rel = std::max(max_rel, num_g0 / den_g0);
      }
      app_log(1, "dense_cross_check: max relative deviation = {}", max_rel);
      REQUIRE(max_rel < 1e-5);
    }

    SECTION("static_rung_W0") {
      // INCREMENT S4 (notes/static_vertex_implementation_plan.md section 2.4).
      //
      // B-S/B-L's polarization object is Pi^{C,0}: the SAME kernel with the rung matrix
      // Z -> W0bar and NO dynamic rung. The kernel already supports this exactly --
      // Wdyn_qwPQ is a pointer and phase 2 returns immediately when it is null -- so the
      // static rung is a call-site change, not a new contraction.
      //
      // Pin (the S3 pattern): the static path (Wdyn = nullptr) must agree with the FULL
      // kernel run at an explicitly ZERO dynamic rung and the same core. That composes
      // with the dense_cross_check arbiter above -- which pins the kernel's algebra
      // against a brute-force orbital/pole-space evaluation of Eq. 13 -- to carry that
      // arbiter's authority onto the static-rung case, without a second dense reference.
      nda::array<cplx, 3> W0_qPQ(nk, Np, Np);
      for (long iq = 0; iq < nk; ++iq) {
        nda::array<cplx, 2> A(Np, Np);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            A(P, Q) = cplx(std::cos(1.3 * double((P + 1) * (Q + 3)) + 0.29 * double(iq)),
                           std::sin(0.7 * double(P + 1) - 0.43 * double(Q + 1)
                                    + 0.17 * double(iq)));
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            W0_qPQ(iq, P, Q) = 0.5 * (A(P, Q) + std::conj(A(Q, P)));   // Hermitian
      }

      nda::array<cplx, 4> Pi_stat(nw_b, nk, Np, Np), Pi_dyn0(nw_b, nk, Np, Np);
      Pi_stat() = cplx(0.0);
      Pi_dyn0() = cplx(0.0);
      // (1) STATIC path: no dynamic rung at all
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, W0_qPQ,
                                   static_cast<nda::array<ComplexType, 4> const*>(nullptr), mdl.kmq, mdl.kpq, C(),
                                   Pi_stat, 0, 1);
      // (2) the full kernel with an explicitly ZERO dynamic rung, same core
      nda::array<cplx, 4> Wzero(nk, nw_b, Np, Np);
      Wzero() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, W0_qPQ, &Wzero,
                                   mdl.kmq, mdl.kpq, C(), Pi_dyn0, 0, 1);

      double num = 0.0, den = 0.0;
      for (long m = 0; m < nw_b; ++m)
        for (long iq = 0; iq < nk; ++iq)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              num = std::max(num, std::abs(Pi_stat(m, iq, P, Q) - Pi_dyn0(m, iq, P, Q)));
              den = std::max(den, std::abs(Pi_dyn0(m, iq, P, Q)));
            }
      app_log(1, "static_rung_W0 (Pi): max|Pi_static - Pi_dyn(W_dyn=0)| = {}, "
                 "max|Pi| = {} (pins the twisted-pair phase identically zero for a "
                 "frequency-independent rung)", num, den);
      REQUIRE(den > 1e-10);
      REQUIRE(num <= 1e-14 * den);

      // the tau = 0 row of the S4 primitive, applied to Pi^{C,0}: finite, and equal to the
      // bosonic w -> tau transform evaluated at tau = 0 (the object Delta w consumes).
      auto Rt0 = solvers::vertex_w0_detail::tau0_transform_row(ft);
      REQUIRE(Rt0.shape(0) == nw_b);
      double pmax = 0.0;
      for (long iq = 0; iq < nk; ++iq)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long m = 0; m < nw_b; ++m) acc += Rt0(m) * Pi_stat(m, iq, P, Q);
            REQUIRE(std::isfinite(acc.real()));
            REQUIRE(std::isfinite(acc.imag()));
            pmax = std::max(pmax, std::abs(acc));
          }
      app_log(1, "static_rung_W0 (Pi): max|Pi^(C,0)(q, tau=0)| = {}", pmax);
      REQUIRE(pmax > 1e-12);
    }

    SECTION("wannier_gauge") {
      // KERNEL-LEVEL gauge oracle (notes/wannier_projector_theory.md section 6.2):
      // the Pi^C output is an aux x aux object (no orbital indices survive). Under a
      // Wannier re-mixing U -> U V (V an M x M unitary, M == Nb here so range(P) is the
      // whole window), the kernel INPUTS transform as Xbar -> Xbar V, Gbar -> V^dag G V.
      // Pi^C depends on range(P) ONLY => Pi_wannier(V) must EQUAL Pi_window bitwise-close
      // for every unitary V. This replicates the production vertex_t Wannier threading
      // (build_Xbar / downfold_G feeding the same kernel with C = [0, M)) but at O(seconds).
      const long M = Nb;                       // full-window rotation (range(P) invariant)
      auto run_gauge = [&](nda::array<cplx, 2> const& V, nda::array<cplx, 4>& Pi_out) {
        // Xbar(is,ik,P,a) = sum_{j in C} X(is,ik,P, C.first()+j) V(j,a)     (Np x M)
        nda::array<cplx, 4> Xbar(ns, nk, Np, M);
        Xbar() = cplx(0.0);
        for (long ik = 0; ik < nk; ++ik)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < M; ++a)
              for (long j = 0; j < Nb; ++j)
                Xbar(0, ik, P, a) += mdl.X_skPa(0, ik, P, C().first() + j) * V(j, a);
        // Gbar(it,is,ik,a,b) = sum_{ij} conj(V(i,a)) G_CC(it,is,ik,i,j) V(j,b)  (M x M)
        // embedded into a full "band space" of size M with C' = [0, M).
        nda::array<cplx, 5> Gbar(nt, ns, nk, M, M);
        Gbar() = cplx(0.0);
        for (long it = 0; it < nt; ++it)
          for (long ik = 0; ik < nk; ++ik)
            for (long a = 0; a < M; ++a)
              for (long b = 0; b < M; ++b)
                for (long i = 0; i < Nb; ++i)
                  for (long jj = 0; jj < Nb; ++jj)
                    Gbar(it, 0, ik, a, b) += std::conj(V(i, a)) *
                        G(it, 0, ik, C().first() + i, C().first() + jj) * V(jj, b);
        Pi_out() = cplx(0.0);
        vertex_pi::pi_c_accumulate_w(ft, tools, Gbar, Xbar, mdl.Z_qPQ, &Wdyn,
                                     mdl.kmq, mdl.kpq, nda::range(0, M), Pi_out, 0, 1);
      };
      // reference: V = identity (== the window path, up to the trivial C-slice offset).
      nda::array<cplx, 4> Pi_ref(nw_b, nk, Np, Np), Pi_V(nw_b, nk, Np, Np);
      {
        nda::array<cplx, 2> Vid(M, M);
        Vid() = cplx(0.0);
        for (long a = 0; a < M; ++a) Vid(a, a) = cplx(1.0);
        run_gauge(Vid, Pi_ref);
      }
      auto max_dev = [&](nda::array<cplx, 4> const& A) {
        double num = 0, den = 0;
        for (long l = 0; l < nw_b; ++l)
          for (long q = 0; q < nk; ++q)
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) {
                num = std::max(num, std::abs(A(l, q, P, Q) - Pi_ref(l, q, P, Q)));
                den = std::max(den, std::abs(Pi_ref(l, q, P, Q)));
              }
        return std::make_pair(num, den);
      };
      REQUIRE(std::isfinite(nda::sum(nda::abs(Pi_ref))));
      // (a) real orthogonal V: real-U sector -- must be exact (degenerate-class).
      {
        rng_t rr(31);
        auto Vc = unitary(M, rr);
        nda::array<cplx, 2> Vre(M, M);
        for (long a = 0; a < M; ++a)
          for (long b = 0; b < M; ++b) Vre(a, b) = cplx(Vc(a, b).real());
        // re-orthonormalize the real part (Gram-Schmidt) so Vre is exactly orthogonal
        for (long b = 0; b < M; ++b) {
          for (long c = 0; c < b; ++c) {
            cplx ip = 0;
            for (long a = 0; a < M; ++a) ip += std::conj(Vre(a, c)) * Vre(a, b);
            for (long a = 0; a < M; ++a) Vre(a, b) -= ip * Vre(a, c);
          }
          double nrm = 0;
          for (long a = 0; a < M; ++a) nrm += std::norm(Vre(a, b));
          nrm = std::sqrt(nrm);
          for (long a = 0; a < M; ++a) Vre(a, b) /= nrm;
        }
        run_gauge(Vre, Pi_V);
        auto [num, den] = max_dev(Pi_V);
        app_log(1, "wannier_gauge: REAL orthogonal V  max|dPi| = {:.3e} (max|Pi| = {:.3e}, rel = {:.3e})",
                num, den, num / den);
        REQUIRE(num / den < 1e-10);
      }
      // (b) 1x1-style diagonal complex phase on each orbital (diagonal unitary).
      {
        nda::array<cplx, 2> Vph(M, M);
        Vph() = cplx(0.0);
        Vph(0, 0) = std::exp(cplx(0.0, 0.7));
        for (long a = 1; a < M; ++a) Vph(a, a) = std::exp(cplx(0.0, 1.3 * double(a)));
        run_gauge(Vph, Pi_V);
        auto [num, den] = max_dev(Pi_V);
        app_log(1, "wannier_gauge: DIAGONAL phase V   max|dPi| = {:.3e} (max|Pi| = {:.3e}, rel = {:.3e})",
                num, den, num / den);
        REQUIRE(num / den < 1e-10);
      }
      // (c) GENUINE off-diagonal COMPLEX unitary V -- the sharp check (the bug).
      {
        rng_t rr(53);
        auto Vc = unitary(M, rr);
        run_gauge(Vc, Pi_V);
        auto [num, den] = max_dev(Pi_V);
        app_log(1, "wannier_gauge: COMPLEX off-diag V  max|dPi| = {:.3e} (max|Pi| = {:.3e}, rel = {:.3e})",
                num, den, num / den);
        REQUIRE(num / den < 1e-10);
      }
    }

    SECTION("static_rung_wannier_gauge") {
      // T-2 of notes/wannier_static_vertex_plan.md (section 3): Pi^{C,0} -- B-S's
      // response ingredient and B-L's Dyson polarization P^{C,L} -- depends on
      // range(P) ONLY, exactly like the dynamic Pi^C: it is aux-emitted with every
      // orbital slot contracted against the projected collocation, and the rung
      // (here the static W0bar) is an orbital-blind aux matrix that the gauge never
      // touches. Same oracle as "wannier_gauge", rung Z -> W0bar, NO dynamic rung
      // (the production static call path: Wdyn = nullptr).
      nda::array<cplx, 3> W0_qPQ(nk, Np, Np);
      for (long iq = 0; iq < nk; ++iq) {
        nda::array<cplx, 2> A(Np, Np);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            A(P, Q) = cplx(std::cos(1.3 * double((P + 1) * (Q + 3)) + 0.29 * double(iq)),
                           std::sin(0.7 * double(P + 1) - 0.43 * double(Q + 1)
                                    + 0.17 * double(iq)));
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            W0_qPQ(iq, P, Q) = 0.5 * (A(P, Q) + std::conj(A(Q, P)));   // Hermitian
      }
      const long M = Nb;                       // full-window rotation (range(P) invariant)
      auto run_gauge0 = [&](nda::array<cplx, 2> const& V, nda::array<cplx, 4>& Pi_out) {
        nda::array<cplx, 4> Xbar(ns, nk, Np, M);
        Xbar() = cplx(0.0);
        for (long ik = 0; ik < nk; ++ik)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < M; ++a)
              for (long j = 0; j < Nb; ++j)
                Xbar(0, ik, P, a) += mdl.X_skPa(0, ik, P, C().first() + j) * V(j, a);
        nda::array<cplx, 5> Gbar(nt, ns, nk, M, M);
        Gbar() = cplx(0.0);
        for (long it = 0; it < nt; ++it)
          for (long ik = 0; ik < nk; ++ik)
            for (long a = 0; a < M; ++a)
              for (long b = 0; b < M; ++b)
                for (long i = 0; i < Nb; ++i)
                  for (long jj = 0; jj < Nb; ++jj)
                    Gbar(it, 0, ik, a, b) += std::conj(V(i, a)) *
                        G(it, 0, ik, C().first() + i, C().first() + jj) * V(jj, b);
        Pi_out() = cplx(0.0);
        vertex_pi::pi_c_accumulate_w(ft, tools, Gbar, Xbar, W0_qPQ,
                                     static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     mdl.kmq, mdl.kpq, nda::range(0, M), Pi_out, 0, 1);
      };
      nda::array<cplx, 4> Pi_ref(nw_b, nk, Np, Np), Pi_V(nw_b, nk, Np, Np);
      {
        nda::array<cplx, 2> Vid(M, M);
        Vid() = cplx(0.0);
        for (long a = 0; a < M; ++a) Vid(a, a) = cplx(1.0);
        run_gauge0(Vid, Pi_ref);
      }
      REQUIRE(std::isfinite(nda::sum(nda::abs(Pi_ref))));
      auto check0 = [&](nda::array<cplx, 2> const& V, const char* tag) {
        run_gauge0(V, Pi_V);
        double num = 0, den = 0;
        for (long l = 0; l < nw_b; ++l)
          for (long q = 0; q < nk; ++q)
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) {
                num = std::max(num, std::abs(Pi_V(l, q, P, Q) - Pi_ref(l, q, P, Q)));
                den = std::max(den, std::abs(Pi_ref(l, q, P, Q)));
              }
        app_log(1, "static_rung_wannier_gauge: {} max|dPi^(C,0)| = {:.3e} "
                   "(max|Pi| = {:.3e}, rel = {:.3e})", tag, num, den, num / den);
        REQUIRE(den > 1e-10);
        REQUIRE(num / den < 1e-10);
      };
      {
        rng_t rr(31);
        auto Vc = unitary(M, rr);
        nda::array<cplx, 2> Vre(M, M);
        for (long a = 0; a < M; ++a)
          for (long b = 0; b < M; ++b) Vre(a, b) = cplx(Vc(a, b).real());
        for (long b = 0; b < M; ++b) {
          for (long c = 0; c < b; ++c) {
            cplx ip = 0;
            for (long a = 0; a < M; ++a) ip += std::conj(Vre(a, c)) * Vre(a, b);
            for (long a = 0; a < M; ++a) Vre(a, b) -= ip * Vre(a, c);
          }
          double nrm = 0;
          for (long a = 0; a < M; ++a) nrm += std::norm(Vre(a, b));
          nrm = std::sqrt(nrm);
          for (long a = 0; a < M; ++a) Vre(a, b) /= nrm;
        }
        check0(Vre, "REAL orthogonal V ");
      }
      {
        nda::array<cplx, 2> Vph(M, M);
        Vph() = cplx(0.0);
        for (long a = 0; a < M; ++a) Vph(a, a) = std::exp(cplx(0.0, 0.7 + 1.3 * double(a)));
        check0(Vph, "DIAGONAL phase V  ");
      }
      {
        rng_t rr(53);
        auto Vc = unitary(M, rr);
        check0(Vc, "COMPLEX off-diag V");
      }
    }

    SECTION("structure") {
      nda::array<cplx, 4> Pi_kern(nw_b, nk, Np, Np);
      Pi_kern() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wdyn,
                                   mdl.kmq, mdl.kpq, C(), Pi_kern, 0, 1);
      nda::array<cplx, 4> Pi_t(nt, nk, Np, Np);
      ft.w_to_tau(Pi_kern, Pi_t, imag_axes_ft::boson);

      // NOTE: the exact structural identities of this single diagram relate
      // Pi_MN(q, beta-tau) to Pi_NM(-q, tau) and involve the physical symmetries of W
      // (W(-q) = W^T(q)); the deliberately asymmetric random toy (independent random
      // X(k), h(k), Z(q), M(q)) does NOT possess them, so same-q mirror/hermiticity
      // deviations below are expected and are reported for the record only. Value
      // correctness is pinned by dense_cross_check/pin_rpa_bubble; the physical-system
      // PH storage convention mirrors rpa_pi (design doc section 2, rule 3).
      double scale = 0, d_mirrorT = 0, d_mirror = 0, d_herm = 0;
      long n_bad = 0;
      for (long it = 0; it < nt; ++it)
        for (long q = 0; q < nk; ++q)
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) {
              auto a = Pi_t(it, q, M, N);
              if (not std::isfinite(std::abs(a))) ++n_bad;
              scale = std::max(scale, std::abs(a));
              d_mirrorT = std::max(d_mirrorT, std::abs(a - Pi_t(nt - it - 1, q, N, M)));
              d_mirror = std::max(d_mirror, std::abs(a - Pi_t(nt - it - 1, q, M, N)));
              d_herm = std::max(d_herm, std::abs(a - std::conj(Pi_t(it, q, N, M))));
            }
      app_log(1, "structure: scale = {}, |Pi(b-t) - Pi^T(t)| = {}, |Pi(b-t) - Pi(t)| = {}, "
                 "|Pi - Pi^dag| = {}", scale, d_mirrorT, d_mirror, d_herm);
      REQUIRE(n_bad == 0);
      REQUIRE(scale > 1e-10);
    }
#endif  // ENABLE_DLR
  }

  TEST_CASE("vertex_pi_lih_smoke", "[methods][vertex][pi_c][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_pi_lih_smoke skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax: the vertex's dynamic-rung intermediates (twisted-pair x wbar~ triple products)
    // carry spectral rates up to ~3x the physical scale (~1.2 for LiH nbnd=16), so the
    // basis needs ~3-4x headroom -- the same [A-comp] requirement as the double
    // convolution (iaft_dconv.hpp). wmax = 1.2 (the plain-GW choice) is NOT sufficient
    // here and produces uncontrolled cancellation loss; the GW parts are insensitive
    // (test_thc_gw passes with wmax = 12 as well).
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_smoke";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // isolated Pi^C on the current MBState (dynamic rung, production scale)
    auto eval_pi_max = [&](solvers::vertex_t& vtx, MBState& mb_state) -> double {
      long nt_f = ft.nt_f();
      long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;
      std::array<long, 4> pgrid = {1, 1, 1, mpi_context->comm.size()};
      std::array<long, 4> bsize = {1, 1, 1, 1};
      std::array<long, 4> gshape = {nt_half, (long)mf->nqpts_ibz(), (long)thc.Np(),
                                    (long)thc.Np()};
      auto dPiC = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);
      double mx = 0;
      long n_bad = 0;
      for (auto const& v : dPiC.local()) {
        double a = std::abs(v);
        if (not std::isfinite(a)) ++n_bad;
        else mx = std::max(mx, a);
      }
      REQUIRE(n_bad == 0);
      // reduce the distributed local maxima so every rank reports the global figure
      mpi_context->comm.all_reduce_in_place_n(&mx, 1, boost::mpi3::max<>{});
      return mx;
    };

    // run one 2-iteration scGW with the vertex under the given q->0 policy ("" = no
    // vertex). Returns (e_hf, e_corr, isolated dynamic-rung max|Pi^C|[policy],
    // max|Pi^C|[gygi] measured on the same state when requested).
    auto run = [&](std::string const& policy, bool also_gygi = false) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");

      const bool with_vertex = not policy.empty();
      // C straddles the LiH gap: HOMO = 1, LUMO = 2 (4 electrons, 16 bands)
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            with_vertex ? policy : "ignore_g0");
      if (vtx.enabled()) {
        // Pi-cut isolation: attach the vertex to the screened-interaction solver only.
        // (Production runs both cuts -- Phi-derivability; MBPT_drivers enforces that.
        // This unit test isolates Pi^C so its outcome does not depend on the Sigma^C
        // kernel, which has its own test target.)
        scr_eri.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();

      double pi_max = -1.0, pi_max_gygi = -1.0;
      if (with_vertex) {
        // With an active vertex the scf driver keeps dW alive across iterations
        // (scf_driver.cpp W-lifetime exception), so iteration 2 above already ran the
        // dynamic rung. Additionally rebuild W from the final G and drive the
        // dynamic-rung Pi^C path directly at production scale (beta = 1000, DLR "low").
        scr_eri.update_w(mb_state, thc, -1);
        REQUIRE(mb_state.dW_qtPQ.has_value());
        pi_max = eval_pi_max(vtx, mb_state);
        app_log(1, "vertex_pi_lih_smoke [{}]: dynamic-rung Pi^C max|.| = {}", policy, pi_max);
        REQUIRE(pi_max > 0.0);
        // bounded magnitude is the q->0 regression guard (the historic uncontrolled
        // run reached ~7e15 -- root-caused to wmax headroom, notes/pi_c_kernel_design 4b;
        // the Gamma cell itself is regular, notes/q0_head_treatment.md section 1)
        REQUIRE(pi_max < 1e6);
        if (also_gygi) {
          REQUIRE(mb_state.eps_inv_head.has_value());
          vtx.set_div_treatment("gygi");
          pi_max_gygi = eval_pi_max(vtx, mb_state);
          app_log(1, "vertex_pi_lih_smoke [gygi, same state]: dynamic-rung Pi^C max|.| = {}",
                  pi_max_gygi);
          REQUIRE(pi_max_gygi > 0.0);
          REQUIRE(pi_max_gygi < 1e6);
        }
      }

      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, pi_max, pi_max_gygi);
    };

    auto [e_hf_0, e_corr_0, pi0, pig0] = run("");
    auto [e_hf_1, e_corr_1, pi_v1, pig1] = run("v1_skip");
    auto [e_hf_2, e_corr_2, pi_v2, pi_gy] = run("ignore_g0", /*also_gygi=*/true);
    (void)pi0; (void)pig0; (void)pig1;

    app_log(1, "vertex_pi_lih_smoke: ---- q->0 policy comparison table (LiH-222 nosym, "
               "2-iteration scGW, Pi-cut only) ----");
    app_log(1, "vertex_pi_lih_smoke: no-vertex          e_hf = {}, e_corr = {}", e_hf_0, e_corr_0);
    app_log(1, "vertex_pi_lih_smoke: vertex v1_skip     e_hf = {}, e_corr = {}, "
               "max|Pi^C| = {}", e_hf_1, e_corr_1, pi_v1);
    app_log(1, "vertex_pi_lih_smoke: vertex ignore_g0   e_hf = {}, e_corr = {}, "
               "max|Pi^C| = {}", e_hf_2, e_corr_2, pi_v2);
    app_log(1, "vertex_pi_lih_smoke: vertex gygi (isolated on the ignore_g0 state) "
               "max|Pi^C| = {}", pi_gy);
    app_log(1, "vertex_pi_lih_smoke: Delta e_corr: v1_skip = {}, ignore_g0 = {}, "
               "(v2 - v1) = {}", e_corr_1 - e_corr_0, e_corr_2 - e_corr_0, e_corr_2 - e_corr_1);

    for (double e : {e_hf_1, e_corr_1, e_hf_2, e_corr_2}) REQUIRE(std::isfinite(e));
    // the vertex must change the screened interaction (and thus e_corr), but not blow up
    REQUIRE(std::abs(e_corr_1 - e_corr_0) > 1e-12);
    REQUIRE(std::abs(e_corr_1 - e_corr_0) < 5e-2);
    REQUIRE(std::abs(e_corr_2 - e_corr_0) > 1e-12);
    REQUIRE(std::abs(e_corr_2 - e_corr_0) < 5e-2);
    // the Gamma cell is one of Nq = 8 rung cells: its effect must be a modest correction
    // on the vertex shift itself, not a new energy scale
    REQUIRE(std::abs(e_corr_2 - e_corr_1) < 0.5 * std::abs(e_corr_1 - e_corr_0) + 1e-4);
#endif  // ENABLE_DLR
  }

} // namespace bdft_tests
