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
 * ISDF-Vertex Phase 3: conservation validation (notes/conservation_validation.md).
 *
 * Sigma^C (G^3 W^2) and Pi^C (G^4 W) are the two cuts of ONE generating functional
 * Phi_2^C. With the discrete pairings (derivation: conservation notes section 1)
 *
 *   S_SigmaG = (1/(Nk beta)) sum_{s,k,w,ab} Sigma^C_ab(k,iw) G~_ba(k,iw)   = +4 Phi^
 *   S_PW     = (1/(Nk beta)) sum_{q!=Gamma,nu} Tr[ Pi^C(q,inu) W(q,inu) ]  = -4 Phi^
 *
 * (G~ = P_C G P_C fed to BOTH kernels; W = Z + W_dyn exactly as the kernels consume
 * it; the external q = Gamma cell excluded to match the kernels' ignore_g0 rung
 * skips), the identity S_SigmaG + S_PW = 0 must hold to kernel accuracy. It pins the
 * RELATIVE sign and normalization of the two kernels independently of the dense
 * references.
 *
 *  - pairing_calibration: the two Parseval primitives (fermionic/bosonic
 *    (1/beta)sum_w A B -> tau convolution rows) and the exact tau=0 DLR row against
 *    fully analytic Matsubara sums. Guards only the NEW numerics of this test.
 *  - toy identity: the sigma test's synthetic THC model, both kernels evaluated for
 *    real, residual |S_SigmaG + S_PW| asserted at the kernels' accuracy class.
 *    Alternate trace orientations (conservation notes section 1.8) are computed and
 *    reported: the identity discriminates them on the asymmetric toy.
 *  - positive controls: (a) flipped relative sign breaks by O(2|S|); (b) the RPA
 *    bubble substituted for Pi^C breaks the identity. Both asserted LARGE.
 *  - lih_conservation: same identity on LiH-222 (nosym), physical G from one scGW
 *    iteration and the real RPA-screened W, looser tolerance (DLR "low", [A-comp]
 *    composite fits at wmax = 6), with the sign-flip and RPA controls at scale.
 */

#undef NDEBUG

#include <array>
#include <complex>
#include <cmath>
#include <random>
#include <utility>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_pi.icc"
#include "methods/vertex/vertex_sigma.icc"

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
  static auto const c_all = nda::range::all;

  namespace cons {

    // ---------------- pairing primitives (conservation notes, section 1.6) ----------
    //
    //   fermionic: (1/beta) sum_w A(iw)B(iw) = - int_0^beta dtau A(tau) B(beta-tau)
    //   bosonic:   (1/beta) sum_nu A(inu)B(inu) = + int_0^beta dtau A(tau) B(beta-tau)
    //
    // int_0^beta dtau e^{i nu_m tau} (.) = row m of Twt_bb (pinned by pin_rpa_bubble);
    // the m0 row is the plain integral. beta-tau = exact index mirror on the
    // PH-symmetric mesh. The tau=0 value of a bosonic-class DLR function is the exact
    // interpolation row at x = -1 (dlr_driver.hpp:134-178).

    // S_SigmaG in ALL FOUR trace orientations (conservation notes, section 1.8):
    //   [0] same-index    sum_ab Sig_ab G_ab        <- the conserving pairing (measured)
    //   [1] matrix-trace  sum_ab Sig_ab G_ba        (the notes' Eq.-4 labeling)
    //   [2] conj same     sum_ab Sig_ab conj(G_ab)
    //   [3] conj matrix   sum_ab Sig_ab conj(G_ba)
    // On hermitian G, [0]==[3] and [1]==[2]; the non-hermitian-G control breaks the
    // degeneracy. The projected G must be passed so the trace restricts to the C block.
    template<typename SArr, typename GArr>
    std::array<cplx, 4> trace_sigma_G(iaft_tools const& tools,
                                      SArr const& Sig_tskab, GArr const& G_tskij) {
      long nt = Sig_tskab.shape(0), ns = Sig_tskab.shape(1), nk = Sig_tskab.shape(2);
      long nb = Sig_tskab.shape(3);
      double spinfac = (ns == 1) ? 2.0 : 1.0;
      std::array<cplx, 4> S = {cplx(0.0), cplx(0.0), cplx(0.0), cplx(0.0)};
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long it = 0; it < nt; ++it) {
            long itm = tools.t_mirror(it);
            std::array<cplx, 4> acc = {cplx(0.0), cplx(0.0), cplx(0.0), cplx(0.0)};
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b) {
                cplx s = Sig_tskab(it, is, ik, a, b);
                acc[0] += s * G_tskij(itm, is, ik, a, b);
                acc[1] += s * G_tskij(itm, is, ik, b, a);
                acc[2] += s * std::conj(G_tskij(itm, is, ik, a, b));
                acc[3] += s * std::conj(G_tskij(itm, is, ik, b, a));
              }
            for (int o = 0; o < 4; ++o) S[o] += tools.Twt_bb(tools.m0, it) * acc[o];
          }
      cplx pref = cplx(-spinfac / double(nk));
      for (int o = 0; o < 4; ++o) S[o] *= pref;
      return S;
    }

    // S_PW in BOTH aux-trace directions {sum_MN Pi_MN W_NM, sum_MN Pi_MN W_MN}.
    // Pi_wqMN: notes-convention Pi(inu) on the full bosonic mesh (kernel output);
    // Z_qPQ + Wt_qtPQ (dynamic W on the FULL tau mesh): the rung exactly as consumed;
    // iq_skip: external q cell excluded (Gamma; ignore_g0 consistency, notes 1.4).
    template<typename ZArr>
    std::pair<cplx, cplx> trace_pi_W(imag_axes_ft::IAFT const& ft, iaft_tools const& tools,
                                     nda::array<cplx, 2> const& T0row,
                                     nda::array<cplx, 4> const& Pi_wqMN,
                                     ZArr const& Z_qPQ,
                                     nda::array<cplx, 4> const& Wt_qtPQ,
                                     long iq_skip, long nk_norm) {
      long nq = Pi_wqMN.shape(1), Np = Pi_wqMN.shape(2);
      long nt = tools.nt_f;
      nda::array<cplx, 4> Pi_t(nt, nq, Np, Np);
      ft.w_to_tau(Pi_wqMN, Pi_t, imag_axes_ft::boson);
      cplx S1(0.0), S2(0.0);
      for (long iq = 0; iq < nq; ++iq) {
        if (iq == iq_skip) continue;
        // instantaneous part: (1/beta) sum_nu Pi(inu) Z = Pi(tau = 0) Z
        for (long M = 0; M < Np; ++M)
          for (long N = 0; N < Np; ++N) {
            cplx p0(0.0);
            for (long it = 0; it < nt; ++it) p0 += T0row(0, it) * Pi_t(it, iq, M, N);
            S1 += p0 * Z_qPQ(iq, N, M);
            S2 += p0 * Z_qPQ(iq, M, N);
          }
        // dynamic part: + int_0^beta dtau Pi(tau) W_dyn(beta - tau)
        for (long it = 0; it < nt; ++it) {
          long itm = tools.t_mirror(it);
          cplx acc1(0.0), acc2(0.0);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) {
              acc1 += Pi_t(it, iq, M, N) * Wt_qtPQ(iq, itm, N, M);
              acc2 += Pi_t(it, iq, M, N) * Wt_qtPQ(iq, itm, M, N);
            }
          S1 += tools.Twt_bb(tools.m0, it) * acc1;
          S2 += tools.Twt_bb(tools.m0, it) * acc2;
        }
      }
      return {S1 / double(nk_norm), S2 / double(nk_norm)};
    }

    // notes-convention RPA aux bubble of the SAME (projected) G, on the full bosonic
    // mesh -- the "wrong Pi" for the positive control. Built with the pin_rpa_bubble
    // primitive (test_vertex_pi.cpp): Pi_w(m,q,M,N) = -(spin/Nk) sum_k Twt_bb(m,:) .
    // [ Gt(k+q, s)_MN Gt(k, beta-s)_NM ](s).
    template<typename GArr, typename XArr>
    nda::array<cplx, 4> rpa_pi_notes_w(iaft_tools const& tools, GArr const& G_tskij,
                                       XArr const& X_skPa, nda::array<long, 2> const& kpq) {
      long nt = G_tskij.shape(0), ns = G_tskij.shape(1), nk = G_tskij.shape(2);
      long nbnd = G_tskij.shape(4);
      long Np = X_skPa.shape(2);
      long nw_b = tools.nw_b;
      double spinfac = (ns == 1) ? 2.0 : 1.0;

      nda::array<cplx, 4> Pi_w(nw_b, nk, Np, Np);
      Pi_w() = cplx(0.0);
      nda::array<cplx, 2> tmp(Np, nbnd);
      nda::array<cplx, 2> prod(nt, Np * Np), bub(nw_b, Np * Np);
      for (long is = 0; is < ns; ++is) {
        nda::array<cplx, 4> Gt(nk, nt, Np, Np);
        for (long k = 0; k < nk; ++k) {
          auto X = X_skPa(is, k, c_all, c_all);
          for (long it = 0; it < nt; ++it) {
            auto Gv = G_tskij(it, is, k, c_all, c_all);
            auto out = Gt(k, it, c_all, c_all);
            nda::blas::gemm(X, Gv, tmp);
            nda::blas::gemm(tmp, nda::dagger(X), out);
          }
        }
        for (long q = 0; q < nk; ++q)
          for (long k = 0; k < nk; ++k) {
            long ikpq = kpq(q, k);
            for (long s = 0; s < nt; ++s) {
              long sm = tools.t_mirror(s);
              for (long M = 0; M < Np; ++M)
                for (long N = 0; N < Np; ++N)
                  prod(s, M * Np + N) = Gt(ikpq, s, M, N) * Gt(k, sm, N, M);
            }
            nda::blas::gemm(cplx(-spinfac / double(nk)), tools.Twt_bb, prod,
                            cplx(0.0), bub);
            for (long m = 0; m < nw_b; ++m)
              for (long M = 0; M < Np; ++M)
                for (long N = 0; N < Np; ++N)
                  Pi_w(m, q, M, N) += bub(m, M * Np + N);
          }
      }
      return Pi_w;
    }

    inline double rel_residual(cplx a, cplx b) {
      double den = std::max(std::abs(a), std::abs(b));
      return std::abs(a + b) / std::max(den, 1e-300);
    }

  } // namespace cons

  // ==================================================================================
  // toy fixture: the sigma test's synthetic THC model (test_vertex_sigma.cpp sig_toy),
  // reproduced verbatim (same seed/spectra) + the kpq map and the bosonic-mesh W_dyn
  // needed by the Pi^C kernel.
  // ==================================================================================
  namespace cons_toy {

    constexpr double beta = 20.0;
    constexpr double wmax = 8.0;
    constexpr double OmX = 0.61;     // toy W_dyn pole
    constexpr long nk = 3, nbnd = 3, Np = 5, ns = 1;
    constexpr long C0 = 0, ncw = 2;  // C = [0, 2)
    inline nda::range C() { return nda::range(C0, C0 + ncw); }

    struct rng_t {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dist{-0.5, 0.5};
      explicit rng_t(unsigned seed) : gen(seed) {}
      cplx operator()() { return cplx(dist(gen), dist(gen)); }
    };

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

    // stable -e^{-eps*tau} (1 - nF(eps)) for tau in [0, beta]
    inline double g_tau(double eps, double tau) {
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-beta * eps));
      return -std::exp(eps * (beta - tau)) / (1.0 + std::exp(beta * eps));
    }

    struct model_t {
      nda::array<double, 2> eps;    // (nk, nbnd) poles of G(k)
      nda::array<cplx, 4> Pr;       // (nk, nbnd[r], nbnd, nbnd) U_r U_r^dag
      nda::array<cplx, 4> X_skPa;   // (ns, nk, Np, nbnd)
      nda::array<cplx, 3> Z_qPQ;    // (nq, Np, Np) hermitian instantaneous part
      nda::array<cplx, 3> M_qPQ;    // (nq, Np, Np) hermitian; W_dyn(q,inu) = M_q s(nu)
      nda::array<long, 2> kmq;      // (nq, nk): k - q
      nda::array<long, 2> kpq;      // (nq, nk): k + q
      nda::array<long, 1> qmin;     // (nq): -q

      // nonherm = true replaces the hermitian pole matrices U_r U_r^dag by U_r V_r^dag
      // with an independent unitary V: G(k,tau) keeps its exact DLR pole structure but
      // loses orbital hermiticity -- this breaks the G_ab = conj(G_ba) degeneracy and
      // pins the trace orientation AND the conjugation convention of the pairing
      // (conservation notes, section 1.8). The rung data (X, Z, M) are identical.
      explicit model_t(bool nonherm = false)
                : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd), X_skPa(ns, nk, Np, nbnd),
                  Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np), kmq(nk, nk), kpq(nk, nk), qmin(nk) {
        rng_t rng(29);
        rng_t rng_nh(101);   // separate stream: the base model data stay identical
        const double base[3] = {-0.67, -0.11, 0.52};
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.06 * double(k);
          auto U = unitary(nbnd, rng);
          auto V = nonherm ? unitary(nbnd, rng_nh) : U;
          for (long r = 0; r < nbnd; ++r)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                Pr(k, r, i, j) = U(i, r) * std::conj(V(j, r));
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
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.25) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        for (long q = 0; q < nk; ++q) {
          qmin(q) = (nk - q) % nk;
          for (long k = 0; k < nk; ++k) {
            kmq(q, k) = (k - q + nk) % nk;
            kpq(q, k) = (k + q) % nk;
          }
        }
      }

      // G(k, tau) on the fermionic IAFT tau mesh: (nt, ns, nk, nbnd, nbnd)
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

      // scalar dynamic profile s(nu) with W_dyn = M * s
      static double s_of(double nu) { return 2.0 * OmX / (OmX * OmX + nu * nu); }

      // W_dyn(q, tau) on the FULL fermionic tau mesh (Sigma^C kernel input + pairing)
      nda::array<cplx, 4> Wdyn_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f(), nw_b = ft.nw_b();
        auto wnb = ft.wn_mesh_b();
        nda::array<cplx, 4> Wt(nk, nt, Np, Np);
        nda::array<cplx, 3> Ww(nw_b, Np, Np), Wtau(nt, Np, Np);
        for (long q = 0; q < nk; ++q) {
          for (long m = 0; m < nw_b; ++m) {
            double nu = double(wnb(m)) * M_PI / beta;
            double s = s_of(nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) Ww(m, P, Q) = M_qPQ(q, P, Q) * s;
          }
          ft.w_to_tau(Ww, Wtau, imag_axes_ft::boson);
          Wt(q, c_all, c_all, c_all) = Wtau;
        }
        return Wt;
      }

      // W_dyn(q, inu) on the full bosonic mesh (Pi^C kernel input)
      nda::array<cplx, 4> Wdyn_w(iaft_tools const& tools) const {
        nda::array<cplx, 4> W(nk, tools.nw_b, Np, Np);
        for (long q = 0; q < nk; ++q)
          for (long l = 0; l < tools.nw_b; ++l) {
            double nu = double(tools.wn_b(l)) * M_PI / beta;
            double s = s_of(nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) W(q, l, P, Q) = M_qPQ(q, P, Q) * s;
          }
        return W;
      }
    };

  } // namespace cons_toy

  TEST_CASE("vertex_conservation_toy", "[methods][vertex][conservation]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_conservation_toy skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace cons_toy;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto& comm = mpi_context->comm;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-10);
    iaft_tools tools(ft);
    model_t mdl;
    const long nt = ft.nt_f();

    // exact tau = 0 evaluation row for bosonic-class DLR functions (x = -1 <-> tau = 0)
    nda::array<double, 1> x0(1);
    x0(0) = -1.0;
    nda::array<cplx, 2> T0row(ft.construct_tau_interpolate_matrix(x0));   // (1, nt)

    SECTION("pairing_calibration") {
      auto xm = ft.tau_mesh();
      auto nF = [&](double e) { return 1.0 / (std::exp(beta * e) + 1.0); };

      // (i) fermionic Parseval: A = 1/(iw-e1), B = 1/(iw-e2)
      //     exact: (1/beta) sum_w A B = (nF(e1) - nF(e2))/(e1 - e2)
      //     route: -sum_it Twt_bb(m0, it) g_e1(tau_it) g_e2(beta - tau_it)
      {
        const double e1 = -0.43, e2 = 0.31;
        cplx route(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * beta;
          double taum = (xm(tools.t_mirror(it)) + 1.0) * 0.5 * beta;
          route += tools.Twt_bb(tools.m0, it) * g_tau(e1, tau) * g_tau(e2, taum);
        }
        route *= -1.0;
        double exact = (nF(e1) - nF(e2)) / (e1 - e2);
        app_log(1, "pairing_calibration fermionic: route = ({}, {}), exact = {}",
                route.real(), route.imag(), exact);
        REQUIRE(std::abs(route - exact) < 1e-8 * std::abs(exact));
      }

      // (ii) bosonic Parseval: a = 2*O1/(O1^2+nu^2), b = 2*O2/(O2^2+nu^2)
      //      a(tau) = cosh(O(tau - beta/2))/sinh(O beta/2)
      //      exact: 2/(O2^2 - O1^2) [O2 coth(O1 b/2) - O1 coth(O2 b/2)]
      {
        const double O1 = 0.61, O2 = 1.10;
        auto a_tau = [&](double O, double tau) {
          return std::cosh(O * (tau - 0.5 * beta)) / std::sinh(0.5 * O * beta);
        };
        cplx route(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * beta;
          double taum = (xm(tools.t_mirror(it)) + 1.0) * 0.5 * beta;
          route += tools.Twt_bb(tools.m0, it) * a_tau(O1, tau) * a_tau(O2, taum);
        }
        double coth1 = 1.0 / std::tanh(0.5 * O1 * beta);
        double coth2 = 1.0 / std::tanh(0.5 * O2 * beta);
        double exact = 2.0 / (O2 * O2 - O1 * O1) * (O2 * coth1 - O1 * coth2);
        app_log(1, "pairing_calibration bosonic: route = ({}, {}), exact = {}",
                route.real(), route.imag(), exact);
        REQUIRE(std::abs(route - exact) < 1e-8 * std::abs(exact));
      }

      // (iii) tau = 0 row on a w_to_tau-fitted Lorentzian: (1/beta) sum_nu a(inu)
      //       = a(tau = 0) = coth(O beta/2)
      {
        const double O1 = 0.61;
        long nw_b = tools.nw_b;
        auto wnb = ft.wn_mesh_b();
        nda::array<cplx, 2> aw(nw_b, 1), at(nt, 1);
        for (long m = 0; m < nw_b; ++m) {
          double nu = double(wnb(m)) * M_PI / beta;
          aw(m, 0) = 2.0 * O1 / (O1 * O1 + nu * nu);
        }
        ft.w_to_tau(aw, at, imag_axes_ft::boson);
        cplx route(0.0);
        for (long it = 0; it < nt; ++it) route += T0row(0, it) * at(it, 0);
        double exact = 1.0 / std::tanh(0.5 * O1 * beta);
        app_log(1, "pairing_calibration tau0-row: route = ({}, {}), exact = {}",
                route.real(), route.imag(), exact);
        REQUIRE(std::abs(route - exact) < 1e-8 * std::abs(exact));
      }
    }

    SECTION("identity_and_controls") {
      auto G = mdl.G_tau(ft);
      auto Wt = mdl.Wdyn_tau(ft);
      auto Ww = mdl.Wdyn_w(tools);

      // G~ = P_C G P_C (conservation notes section 1.2)
      nda::array<cplx, 5> Gproj(nt, ns, nk, nbnd, nbnd);
      Gproj() = cplx(0.0);
      for (long it = 0; it < nt; ++it)
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            Gproj(it, is, ik, C(), C()) = G(it, is, ik, C(), C());

      // ---- Sigma^C via the actual kernel, on G and on G~ (invariance check) --------
      nda::array<cplx, 5> Sig(nt, ns, nk, nbnd, nbnd), Sig_p(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, C(), G, mdl.X_skPa, Wt,
                                                      mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                                      /*iq_gamma*/ 0, Sig);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, C(), Gproj, mdl.X_skPa, Wt,
                                                      mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                                      /*iq_gamma*/ 0, Sig_p);
      double d_inv = 0, s_scale = 0;
      for (long i = 0; i < Sig.size(); ++i) {
        d_inv = std::max(d_inv, std::abs(Sig.data()[i] - Sig_p.data()[i]));
        s_scale = std::max(s_scale, std::abs(Sig.data()[i]));
      }
      app_log(1, "conservation_toy: Sigma^C[G~] vs Sigma^C[G]: max|diff| = {} "
                 "(scale {}) -- internal lines read only the C block", d_inv, s_scale);
      REQUIRE(s_scale > 1e-10);
      REQUIRE(d_inv < 1e-12 * s_scale);

      // ---- Pi^C via the actual kernel on G~, notes convention, rung Gamma skipped --
      nda::array<cplx, 4> Pi_w(tools.nw_b, nk, Np, Np);
      Pi_w() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, Gproj, mdl.X_skPa, mdl.Z_qPQ, &Ww,
                                   mdl.kmq, mdl.kpq, C(), Pi_w, 0, 1,
                                   /*skip_rung_gamma=*/true);

      // ---- the pairings -------------------------------------------------------------
      auto S_SG_o = cons::trace_sigma_G(tools, Sig_p, Gproj);
      auto [S_PW, S_PW_alt] = cons::trace_pi_W(ft, tools, T0row, Pi_w, mdl.Z_qPQ, Wt,
                                               /*iq_skip=*/0, nk);
      cplx S_SG = S_SG_o[0];   // same-index pairing: the conserving one (notes 1.8)

      double scale = std::max(std::abs(S_SG), std::abs(S_PW));
      double rel = cons::rel_residual(S_SG, S_PW);
      app_log(1, "conservation_toy: S_SigmaG    = ({}, {})", S_SG.real(), S_SG.imag());
      app_log(1, "conservation_toy: S_PW        = ({}, {})", S_PW.real(), S_PW.imag());
      app_log(1, "conservation_toy: |S_SG + S_PW| = {}, scale = {}, rel = {}",
              std::abs(S_SG + S_PW), scale, rel);
      // alternate-reading diagnostics (conservation notes section 1.8); on this
      // hermitian-G toy [1](matrix-trace) == [2](conj same-index) and [0] == [3]
      double rel_mat = cons::rel_residual(S_SG_o[1], S_PW);
      double rel_altW = cons::rel_residual(S_SG, S_PW_alt);
      app_log(1, "conservation_toy: alternate readings: rel(matrix-trace G_ba) = {}, "
                 "rel(conj same-index) = {}, rel(conj matrix-trace) = {}, "
                 "rel(W_MN direction) = {}",
              rel_mat, cons::rel_residual(S_SG_o[2], S_PW),
              cons::rel_residual(S_SG_o[3], S_PW), rel_altW);

      REQUIRE(scale > 1e-8);
      REQUIRE(std::isfinite(rel));
      // tolerance: kernels are pinned at ~3e-12 (Sigma) / ~1e-13 abs (Pi) on these
      // grids; the pairing adds only DLR-eps-class transforms of [A-comp]-representable
      // products (>= 2x wmax headroom). 1e-8 keeps the fused_vs_batched safety-margin
      // style (assert 1e-8 vs measured ~1e-11).
      REQUIRE(rel < 1e-8);
      // the asymmetric toy discriminates the readings: the matrix-trace orientation
      // and the transposed aux contraction must NOT satisfy the identity
      REQUIRE(rel_mat > 1e3 * std::max(rel, 1e-14));
      REQUIRE(rel_altW > 1e3 * std::max(rel, 1e-14));

      // ---- positive control (a): flipped relative sign ------------------------------
      double rel_flip = std::abs(S_SG - S_PW) / scale;
      app_log(1, "conservation_toy: control (sign flip): |S_SG - S_PW|/scale = {}", rel_flip);
      REQUIRE(rel_flip > 1.0);   // = 2|S|/scale up to the residual

      // ---- positive control (b): RPA bubble substituted for Pi^C --------------------
      auto Pi_rpa = cons::rpa_pi_notes_w(tools, Gproj, mdl.X_skPa, mdl.kpq);
      auto [S_PW_rpa, S_PW_rpa_alt] = cons::trace_pi_W(ft, tools, T0row, Pi_rpa,
                                                       mdl.Z_qPQ, Wt, /*iq_skip=*/0, nk);
      (void)S_PW_rpa_alt;
      double rel_rpa = cons::rel_residual(S_SG, S_PW_rpa);
      app_log(1, "conservation_toy: control (RPA Pi): S_PW_rpa = ({}, {}), rel = {}",
              S_PW_rpa.real(), S_PW_rpa.imag(), rel_rpa);
      REQUIRE(rel_rpa > 1e3 * std::max(rel, 1e-14));
      REQUIRE(rel_rpa > 1e-2);
    }

    SECTION("identity_nonhermitian_G") {
      // The hermitian-G toy cannot distinguish the same-index pairing sum_ab Sig_ab G_ab
      // from the conjugate matrix-trace sum_ab Sig_ab conj(G_ba) (they coincide when
      // G_ab = conj(G_ba)). A non-hermitian orbital structure (U_r V_r^dag pole
      // matrices, same poles, same rung data) breaks the degeneracy: the identity is
      // algebraic in the plain matrix elements of G, so ONLY the same-index pairing
      // must survive. This pins both the orientation and the conjugation convention.
      model_t mdl_nh(/*nonherm=*/true);
      auto G_nh = mdl_nh.G_tau(ft);
      nda::array<cplx, 5> Gproj(nt, ns, nk, nbnd, nbnd);
      Gproj() = cplx(0.0);
      for (long it = 0; it < nt; ++it)
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            Gproj(it, is, ik, C(), C()) = G_nh(it, is, ik, C(), C());

      auto Wt = mdl_nh.Wdyn_tau(ft);
      auto Ww = mdl_nh.Wdyn_w(tools);
      nda::array<cplx, 5> Sig(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, C(), Gproj, mdl_nh.X_skPa,
                                                      Wt, mdl_nh.Z_qPQ, mdl_nh.kmq,
                                                      mdl_nh.qmin, /*iq_gamma*/ 0, Sig);
      nda::array<cplx, 4> Pi_w(tools.nw_b, nk, Np, Np);
      Pi_w() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, Gproj, mdl_nh.X_skPa, mdl_nh.Z_qPQ, &Ww,
                                   mdl_nh.kmq, mdl_nh.kpq, C(), Pi_w, 0, 1,
                                   /*skip_rung_gamma=*/true);

      auto S_SG_o = cons::trace_sigma_G(tools, Sig, Gproj);
      auto [S_PW, S_PW_alt] = cons::trace_pi_W(ft, tools, T0row, Pi_w, mdl_nh.Z_qPQ, Wt,
                                               /*iq_skip=*/0, nk);
      (void)S_PW_alt;

      double scale = std::max(std::abs(S_SG_o[0]), std::abs(S_PW));
      double rel = cons::rel_residual(S_SG_o[0], S_PW);
      app_log(1, "conservation_toy_nh: S_SigmaG(same-index) = ({}, {}), S_PW = ({}, {})",
              S_SG_o[0].real(), S_SG_o[0].imag(), S_PW.real(), S_PW.imag());
      app_log(1, "conservation_toy_nh: rel(same-index) = {}, rel(matrix-trace) = {}, "
                 "rel(conj same-index) = {}, rel(conj matrix-trace) = {}",
              rel, cons::rel_residual(S_SG_o[1], S_PW),
              cons::rel_residual(S_SG_o[2], S_PW),
              cons::rel_residual(S_SG_o[3], S_PW));
      REQUIRE(scale > 1e-8);
      // the non-hermitian |S| is ~40x smaller than the hermitian toy's while the
      // absolute residual floor (kernel/DLR eps class) is unchanged -- measured
      // rel = 7.7e-9 (abs 6e-13); assert with the same margin style
      REQUIRE(rel < 1e-7);
      for (int o = 1; o < 4; ++o)
        REQUIRE(cons::rel_residual(S_SG_o[o], S_PW) > 1e5 * std::max(rel, 1e-14));
    }
#endif  // ENABLE_DLR
  }

  TEST_CASE("vertex_conservation_lih", "[methods][vertex][conservation][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_conservation_lih skipped: build has ENABLE_DLR=OFF.");
#else
    decltype(nda::range::all) all;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax = 6.0: the vertex kernels' [A-comp] intermediates need ~3x headroom over the
    // LiH spectral range (~1.2) -- the pi-design section 4b requirement; wmax = 1.2
    // (the plain-GW choice) destroys the dynamic-rung cancellations.
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_conservation_smoke";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");

    // one plain scGW iteration -> physical G; then rebuild the real RPA-screened W from
    // it. The identity is algebraic in (G, W): any pair fed CONSISTENTLY to both kernels
    // and both pairings is valid; the pure-RPA W isolates this test from the in-loop
    // vertex state (same isolation choice as the sigma smoke's rebuild).
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                   1, false, 1e-9, true);
    mpi_context->comm.barrier();
    app_log(1, "vertex_conservation_lih: e_hf = {}, e_corr = {}", e_hf, e_corr);
    REQUIRE(std::isfinite(e_hf));
    REQUIRE(std::isfinite(e_corr));
    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());

    // ------------- glue, replicated from vertex_t.cpp (eval_Sigma_C / eval_Pi_C) -----
    auto MF = thc.MF();
    const long nkpts = MF->nkpts(), nqpts = MF->nqpts();
    const long nkpts_ibz = MF->nkpts_ibz(), nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np(), nbnd = MF->nbnd();
    utils::check(nqpts == nqpts_ibz and nkpts == nkpts_ibz and nqpts == nkpts,
                 "vertex_conservation_lih: needs a symmetry-free mesh.");
    const nda::range Crng(1, 3);   // straddles the LiH gap (HOMO = 1, LUMO = 2)

    auto G_loc = mb_state.sG_tskij.value().local();
    const long nt = G_loc.shape(0), ns = G_loc.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    utils::check(nt == ft.nt_f(), "vertex_conservation_lih: G time axis != nt_f.");

    iaft_tools tools(ft);

    // G~ = P_C G P_C, fed to BOTH kernels and both pairings
    nda::array<cplx, 5> Gproj(nt, ns, nkpts, nbnd, nbnd);
    Gproj() = cplx(0.0);
    for (long it = 0; it < nt; ++it)
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          Gproj(it, is, ik, Crng, Crng) = G_loc(it, is, ik, Crng, Crng);

    // collocations and bare core
    nda::array<cplx, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);
    nda::array<cplx, 3> Z_qPQ(nqpts, Np, Np);
    for (long iq = 0; iq < nqpts; ++iq)
      Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // dynamic W: full-tau unfold (Sigma^C kernel input + the S_PW dynamic pairing)
    nda::array<cplx, 4> Wt_qtPQ(nqpts, nt, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_conservation_lih: unexpected dW_qtPQ global shape.");
      nda::array<cplx, 4> W_half(nqpts, nt_half, Np, Np);
      W_half() = cplx(0.0);
      W_half(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(W_half.data(), W_half.size(), std::plus<>{});
      for (long it = 0; it < nt; ++it) {
        long ith = std::min(it, nt - it - 1);
        Wt_qtPQ(all, it, all, all) = W_half(all, ith, all, all);
      }
    }
    // dynamic W on the full bosonic mesh (Pi^C kernel input), vertex_t.cpp:298-322
    nda::array<cplx, 4> Wdyn_qwPQ(nqpts, tools.nw_b, Np, Np);
    {
      long nw_b = tools.nw_b;
      long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
      nda::array<cplx, 3> W_wpos(nw_half, Np, Np);
      for (long iq = 0; iq < nqpts; ++iq) {
        auto W_t = Wt_qtPQ(iq, nda::range(0, nt_half), all, all);
        ft.tau_to_w_PHsym(W_t, W_wpos);
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ(iq, l, all, all) = W_wpos(lpos, all, all);
        }
      }
    }

    // momentum maps and Gamma (vertex_t.cpp:161-189)
    nda::array<long, 2> kmq(nqpts, nkpts), kpq(nqpts, nkpts);
    nda::array<long, 1> qmin(nqpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      qmin(iq) = MF->qminus()(iq);
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
      for (long ik = 0; ik < nkpts; ++ik) kpq(iq, kmq(iq, ik)) = ik;
    }
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) { iq_gamma = iq; break; }
      }
      utils::check(iq_gamma >= 0, "vertex_conservation_lih: no Gamma q-point found.");
    }

    // ------------- the two cuts via the actual kernels --------------------------------
    nda::array<cplx, 5> Sig(nt, ns, nkpts, nbnd, nbnd);
    solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, mpi_context->comm, Crng, Gproj,
                                                    X_skPa, Wt_qtPQ, Z_qPQ, kmq, qmin,
                                                    iq_gamma, Sig);
    nda::array<cplx, 4> Pi_w(tools.nw_b, nqpts, Np, Np);
    Pi_w() = cplx(0.0);
    vertex_pi::pi_c_accumulate_w(ft, tools, Gproj, X_skPa, Z_qPQ, &Wdyn_qwPQ,
                                 kmq, kpq, Crng, Pi_w,
                                 mpi_context->comm.rank(), mpi_context->comm.size(),
                                 /*skip_rung_gamma=*/true);
    mpi_context->comm.all_reduce_in_place_n(Pi_w.data(), Pi_w.size(), std::plus<>{});

    // ------------- the pairings --------------------------------------------------------
    nda::array<double, 1> x0(1);
    x0(0) = -1.0;
    nda::array<cplx, 2> T0row(ft.construct_tau_interpolate_matrix(x0));

    auto S_SG_o = cons::trace_sigma_G(tools, Sig, Gproj);
    auto [S_PW, S_PW_alt] = cons::trace_pi_W(ft, tools, T0row, Pi_w, Z_qPQ, Wt_qtPQ,
                                             iq_gamma, nkpts);
    cplx S_SG = S_SG_o[0];   // same-index pairing (conserving; notes section 1.8)

    double scale = std::max(std::abs(S_SG), std::abs(S_PW));
    double rel = cons::rel_residual(S_SG, S_PW);
    app_log(1, "vertex_conservation_lih: S_SigmaG = ({}, {})", S_SG.real(), S_SG.imag());
    app_log(1, "vertex_conservation_lih: alternate readings: rel(matrix-trace) = {}, "
               "rel(W_MN direction) = {}",
            cons::rel_residual(S_SG_o[1], S_PW), cons::rel_residual(S_SG, S_PW_alt));
    app_log(1, "vertex_conservation_lih: S_PW     = ({}, {})", S_PW.real(), S_PW.imag());
    app_log(1, "vertex_conservation_lih: |S_SG + S_PW| = {}, scale = {}, rel = {}",
            std::abs(S_SG + S_PW), scale, rel);
    app_log(1, "vertex_conservation_lih: realness: Im/Re(S_SG) = {}, Im/Re(S_PW) = {}",
            std::abs(S_SG.imag()) / std::max(std::abs(S_SG.real()), 1e-300),
            std::abs(S_PW.imag()) / std::max(std::abs(S_PW.real()), 1e-300));

    REQUIRE(scale > 1e-12);
    REQUIRE(std::isfinite(rel));
    // tolerance (conservation notes section 3): DLR "low" (eps = 1e-6; measured pole-fit
    // error ~2.4e-5 at this setting), [A-comp] composite fits at wmax = 6 -- the residual
    // is a cancellation of two O(|S|) numbers each carrying those representation errors.
    // Measured: rel = 7.1e-6 (2026-07-16, 2 ranks); assert with ~100x headroom.
    REQUIRE(rel < 1e-3);

    // positive control (a): flipped relative sign
    double rel_flip = std::abs(S_SG - S_PW) / scale;
    app_log(1, "vertex_conservation_lih: control (sign flip): |S_SG - S_PW|/scale = {}",
            rel_flip);
    REQUIRE(rel_flip > 1.0);

    // positive control (b): RPA bubble substituted for Pi^C
    auto Pi_rpa = cons::rpa_pi_notes_w(tools, Gproj, X_skPa, kpq);
    auto [S_PW_rpa, S_PW_rpa_alt] = cons::trace_pi_W(ft, tools, T0row, Pi_rpa, Z_qPQ,
                                                     Wt_qtPQ, iq_gamma, nkpts);
    (void)S_PW_rpa_alt;
    double rel_rpa = cons::rel_residual(S_SG, S_PW_rpa);
    app_log(1, "vertex_conservation_lih: control (RPA Pi): S_PW_rpa = ({}, {}), rel = {}",
            S_PW_rpa.real(), S_PW_rpa.imag(), rel_rpa);
    REQUIRE(rel_rpa > 1e2 * rel);
    REQUIRE(rel_rpa > 5e-2);

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif  // ENABLE_DLR
  }

} // namespace bdft_tests
