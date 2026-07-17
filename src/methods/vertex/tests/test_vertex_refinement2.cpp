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
 * ISDF-Vertex Refinement 2: secondary ISDF basis with the Option-A downfold
 * (notes/refinement2_optionA.md; theoryB.pdf Sec. 11).
 *
 *  - vertex_refinement2_toy: transfer-algebra unit checks on a synthetic THC model
 *    (independently-coded pair matrices B/C, truncated-SVD t, fold/upfold):
 *    s = B^dag B Hermitian PSD with cond reported; B t = C at full secondary rank;
 *    the no-leak identity Tr[(t^dag Pibar t) W] = Tr[Pibar (t W t^dag)] at machine
 *    precision (full AND reduced rank); both kernels in the secondary basis vs the
 *    global path at FULL rank (Sigma^C on the C-C block, upfolded Pi^C vs the global
 *    kernel on G~ = P_C G P_C) at machine class; the conservation identity
 *    S_SigmaG + S_PW = 0 in the secondary path; eta and controlled degradation at
 *    reduced rank.
 *  - vertex_refinement2_lih: LiH-222 (nosym), C = [1,3). One restricted-range ISDF
 *    point selection (max = 32 = the pair rank nc^2 nk); nested rank scan
 *    N_m in {8, 16, 24, 32}: eta(q, nu) table (Eq. 40), monotone decreasing and
 *    small at full rank. Kernel-level secondary-vs-global agreement at N_m = 32 on
 *    the physical (G, W) state; no-leak; conservation identity in the secondary path
 *    under ignore_g0 AND gygi (the head-augmented W^(Gamma) downfolds through t).
 *    End-to-end: 1-iteration scGW (both cuts) global vs secondary (N_m = 32, 8)
 *    through the production vertex_t plumbing.
 */

#undef NDEBUG

#include <array>
#include <complex>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "nda/lapack.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_pi.icc"
#include "methods/vertex/vertex_sigma.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  namespace vertex_pi = methods::solvers::vertex_pi;
  using vertex_pi::iaft_tools;
  using cplx = ComplexType;
  static auto const r_all = nda::range::all;

  namespace r2 {

    // ------------- transfer algebra, INDEPENDENTLY coded (guards vertex_t.cpp) -------
    // Pair rows at transfer q in the kernels' pinned in/out rule (pi design section 2
    // rule 1): I = ((is*nk + ik)*nc + o)*nc + i,
    //   A(I, u) = X(is, k - q, u, orb0 + i) * conj(X(is, k, u, orb0 + o)).
    inline nda::array<cplx, 2> pair_matrix(nda::array<cplx, 4> const& X_skua, long orb0,
                                           long nc, nda::array<long, 2> const& kmq,
                                           long iq) {
      long ns = X_skua.shape(0), nk = X_skua.shape(1), naux = X_skua.shape(2);
      nda::array<cplx, 2> A(ns * nk * nc * nc, naux);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long o = 0; o < nc; ++o)
            for (long i = 0; i < nc; ++i) {
              long I = ((is * nk + ik) * nc + o) * nc + i;
              for (long u = 0; u < naux; ++u)
                A(I, u) = X_skua(is, kmq(iq, ik), u, orb0 + i) *
                          std::conj(X_skua(is, ik, u, orb0 + o));
            }
      return A;
    }

    // t = argmin || B t - C ||_F via truncated-SVD least squares (gelss on B; rcond
    // acts on sv(B), i.e. the metric s = B^dag B is regularized at rcond^2). Returns
    // (t (Nm x Np), sv(B), rank).
    inline std::tuple<nda::array<cplx, 2>, nda::array<double, 1>, int>
    solve_t(nda::array<cplx, 2> const& B, nda::array<cplx, 2> const& C, double rcond) {
      long Npair = B.shape(0), Nm = B.shape(1), Np = C.shape(1);
      REQUIRE(C.shape(0) == Npair);
      REQUIRE(Npair >= Nm);
      // gelss requires F-layout: keep transposes in C-layout, pass transposed views
      nda::array<cplx, 2> BT(Nm, Npair), CT(Np, Npair);
      for (long I = 0; I < Npair; ++I) {
        for (long m = 0; m < Nm; ++m) BT(m, I) = B(I, m);
        for (long P = 0; P < Np; ++P) CT(P, I) = C(I, P);
      }
      nda::array<double, 1> sv(std::min(Npair, Nm));
      int rank = 0;
      int info = nda::lapack::gelss(nda::transpose(BT), nda::transpose(CT), sv, rcond, rank);
      REQUIRE(info == 0);
      nda::array<cplx, 2> t(Nm, Np);
      for (long m = 0; m < Nm; ++m)
        for (long P = 0; P < Np; ++P) t(m, P) = CT(P, m);
      return {std::move(t), std::move(sv), rank};
    }

    // plain-loop fold / upfold (independent of the production gemm implementations)
    template<typename AArr>
    nda::array<cplx, 2> fold(nda::array<cplx, 2> const& t, AArr const& A) {
      long Nm = t.shape(0), Np = t.shape(1);
      nda::array<cplx, 2> tmp(Nm, Np), out(Nm, Nm);
      tmp() = cplx(0.0); out() = cplx(0.0);
      for (long m = 0; m < Nm; ++m)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) tmp(m, Q) += t(m, P) * A(P, Q);
      for (long m = 0; m < Nm; ++m)
        for (long n = 0; n < Nm; ++n)
          for (long Q = 0; Q < Np; ++Q) out(m, n) += tmp(m, Q) * std::conj(t(n, Q));
      return out;
    }
    template<typename PArr>
    nda::array<cplx, 2> upfold(nda::array<cplx, 2> const& t, PArr const& Pi) {
      long Nm = t.shape(0), Np = t.shape(1);
      nda::array<cplx, 2> tmp(Np, Nm), out(Np, Np);
      tmp() = cplx(0.0); out() = cplx(0.0);
      for (long P = 0; P < Np; ++P)
        for (long m = 0; m < Nm; ++m)
          for (long n = 0; n < Nm; ++n) tmp(P, n) += std::conj(t(m, P)) * Pi(m, n);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q)
          for (long n = 0; n < Nm; ++n) out(P, Q) += tmp(P, n) * t(n, Q);
      return out;
    }

    // eta(q, .) of theoryB Eq. 40 for one core A: ||(Bt)A(Bt)^dag - CAC^dag||_F/||.||_F
    template<typename AArr>
    double eta(nda::array<cplx, 2> const& B, nda::array<cplx, 2> const& C,
               nda::array<cplx, 2> const& t, AArr const& A) {
      long Npair = B.shape(0), Np = C.shape(1), Nm = B.shape(1);
      nda::array<cplx, 2> Cf(Npair, Np);
      Cf() = cplx(0.0);
      for (long I = 0; I < Npair; ++I)
        for (long m = 0; m < Nm; ++m)
          for (long P = 0; P < Np; ++P) Cf(I, P) += B(I, m) * t(m, P);
      auto sandwich = [&](nda::array<cplx, 2> const& R) {
        nda::array<cplx, 2> E(Npair, Np), W(Npair, Npair);
        E() = cplx(0.0); W() = cplx(0.0);
        for (long I = 0; I < Npair; ++I)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) E(I, Q) += R(I, P) * A(P, Q);
        for (long I = 0; I < Npair; ++I)
          for (long J = 0; J < Npair; ++J)
            for (long Q = 0; Q < Np; ++Q) W(I, J) += E(I, Q) * std::conj(R(J, Q));
        return W;
      };
      auto WC = sandwich(C), WA = sandwich(Cf);
      double num = 0, den = 0;
      for (long I = 0; I < Npair; ++I)
        for (long J = 0; J < Npair; ++J) {
          num += std::norm(WA(I, J) - WC(I, J));
          den += std::norm(WC(I, J));
        }
      return std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
    }

    // ------------- conservation pairings (conservation notes section 1.6/1.8) --------
    // S_SigmaG: the conserving SAME-INDEX pairing sum_ab Sig_ab G_ab.
    template<typename SArr, typename GArr>
    cplx trace_sigma_G(iaft_tools const& tools, SArr const& Sig, GArr const& G) {
      long nt = Sig.shape(0), ns = Sig.shape(1), nk = Sig.shape(2), nb = Sig.shape(3);
      double spinfac = (ns == 1) ? 2.0 : 1.0;
      cplx S(0.0);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long it = 0; it < nt; ++it) {
            long itm = tools.t_mirror(it);
            cplx acc(0.0);
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b)
                acc += Sig(it, is, ik, a, b) * G(itm, is, ik, a, b);
            S += tools.Twt_bb(tools.m0, it) * acc;
          }
      return S * cplx(-spinfac / double(nk));
    }

    // S_PW = (1/nk_norm) sum_{q} [ Pi(tau=0)_MN Z_NM + int Pi(tau)_MN Wdyn(beta-tau)_NM ]
    template<typename ZArr, typename WArr>
    cplx trace_pi_W(imag_axes_ft::IAFT const& ft, iaft_tools const& tools,
                    nda::array<cplx, 2> const& T0row, nda::array<cplx, 4> const& Pi_w,
                    ZArr const& Z_q, WArr const& Wt_qt, long nk_norm) {
      long nq = Pi_w.shape(1), Na = Pi_w.shape(2), nt = tools.nt_f;
      nda::array<cplx, 4> Pi_t(nt, nq, Na, Na);
      ft.w_to_tau(Pi_w, Pi_t, imag_axes_ft::boson);
      cplx S(0.0);
      for (long iq = 0; iq < nq; ++iq) {
        for (long M = 0; M < Na; ++M)
          for (long N = 0; N < Na; ++N) {
            cplx p0(0.0);
            for (long it = 0; it < nt; ++it) p0 += T0row(0, it) * Pi_t(it, iq, M, N);
            S += p0 * Z_q(iq, N, M);
          }
        for (long it = 0; it < nt; ++it) {
          long itm = tools.t_mirror(it);
          cplx acc(0.0);
          for (long M = 0; M < Na; ++M)
            for (long N = 0; N < Na; ++N)
              acc += Pi_t(it, iq, M, N) * Wt_qt(iq, itm, N, M);
          S += tools.Twt_bb(tools.m0, it) * acc;
        }
      }
      return S / double(nk_norm);
    }

    inline double rel_residual(cplx a, cplx b) {
      double den = std::max(std::abs(a), std::abs(b));
      return std::abs(a + b) / std::max(den, 1e-300);
    }

    // ------------- synthetic THC toy (the sigma/pi test fixture class) ----------------
    constexpr double beta = 20.0, wmax = 8.0, OmX = 0.61;
    constexpr long nk = 3, nbnd = 3, Np = 5, ns = 1;
    constexpr long C0 = 0, ncw = 2;
    inline nda::range Cw() { return nda::range(C0, C0 + ncw); }

    struct rng_t {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dist{-0.5, 0.5};
      explicit rng_t(unsigned seed) : gen(seed) {}
      cplx operator()() { return cplx(dist(gen), dist(gen)); }
    };

    inline double g_tau(double eps, double tau) {
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-beta * eps));
      return -std::exp(eps * (beta - tau)) / (1.0 + std::exp(beta * eps));
    }

    struct model_t {
      nda::array<double, 2> eps;   // (nk, nbnd)
      nda::array<cplx, 4> Pr;      // (nk, r, nbnd, nbnd)
      nda::array<cplx, 4> X_skPa;  // (ns, nk, Np, nbnd)
      nda::array<cplx, 3> Z_qPQ, M_qPQ;   // (nq, Np, Np) hermitian
      nda::array<long, 2> kmq, kpq;
      nda::array<long, 1> qmin;

      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd), X_skPa(ns, nk, Np, nbnd),
                  Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np), kmq(nk, nk), kpq(nk, nk),
                  qmin(nk) {
        rng_t rng(83);
        const double base[3] = {-0.67, -0.11, 0.52};
        auto unitary = [&](long n) {
          auto refl = [&](nda::array<cplx, 1> const& v) {
            nda::array<cplx, 2> H(n, n);
            double nv = 0;
            for (long i = 0; i < n; ++i) nv += std::norm(v(i));
            for (long i = 0; i < n; ++i)
              for (long j = 0; j < n; ++j)
                H(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) -
                          2.0 * v(i) * std::conj(v(j)) / nv;
            return H;
          };
          nda::array<cplx, 1> v1(n), v2(n);
          for (long i = 0; i < n; ++i) { v1(i) = rng(); v2(i) = rng(); }
          nda::array<cplx, 2> U(n, n);
          nda::blas::gemm(refl(v1), refl(v2), U);
          return U;
        };
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.06 * double(k);
          auto U = unitary(nbnd);
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
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.25) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        for (long q = 0; q < nk; ++q) {
          qmin(q) = (nk - q) % nk;
          for (long k = 0; k < nk; ++k) kmq(q, k) = (k - q + nk) % nk;
          for (long k = 0; k < nk; ++k) kpq(q, k) = (k + q) % nk;
        }
      }

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
                for (long j = 0; j < nbnd; ++j) G(it, 0, k, i, j) += g * Pr(k, r, i, j);
            }
        }
        return G;
      }

      static double s_of(double nu) { return 2.0 * OmX / (OmX * OmX + nu * nu); }

      // dynamic W on the full bosonic Matsubara mesh: (nq, nw_b, Np, Np)
      nda::array<cplx, 4> Wdyn_w(imag_axes_ft::IAFT const& ft) const {
        long nw_b = ft.nw_b();
        auto wnb = ft.wn_mesh_b();
        nda::array<cplx, 4> Ww(nk, nw_b, Np, Np);
        for (long q = 0; q < nk; ++q)
          for (long m = 0; m < nw_b; ++m) {
            double nu = double(wnb(m)) * M_PI / beta;
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) Ww(q, m, P, Q) = M_qPQ(q, P, Q) * s_of(nu);
          }
        return Ww;
      }

      // dynamic W(tau) on the FULL fermionic tau mesh: (nq, nt, Np, Np)
      nda::array<cplx, 4> Wdyn_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f(), nw_b = ft.nw_b();
        auto wnb = ft.wn_mesh_b();
        nda::array<cplx, 4> Wt(nk, nt, Np, Np);
        nda::array<cplx, 3> Ww(nw_b, Np, Np), Wtau(nt, Np, Np);
        for (long q = 0; q < nk; ++q) {
          for (long m = 0; m < nw_b; ++m) {
            double nu = double(wnb(m)) * M_PI / beta;
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) Ww(m, P, Q) = M_qPQ(q, P, Q) * s_of(nu);
          }
          ft.w_to_tau(Ww, Wtau, imag_axes_ft::boson);
          Wt(q, r_all, r_all, r_all) = Wtau;
        }
        return Wt;
      }
    };

  } // namespace r2

  // ====================================================================================
  TEST_CASE("vertex_refinement2_toy", "[methods][vertex][refinement2]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_refinement2_toy skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto& comm = mpi_context->comm;   // kernels all-reduce internally; inputs replicated

    imag_axes_ft::IAFT ft(r2::beta, r2::wmax, imag_axes_ft::dlr_basis, 1e-10);
    iaft_tools tools(ft);
    r2::model_t mdl;
    const long nc = r2::ncw, Npair = r2::ns * r2::nk * nc * nc;   // = 12
    const long Nm_full = Npair;
    const long nt = ft.nt_f();

    auto G = mdl.G_tau(ft);
    auto Wt = mdl.Wdyn_tau(ft);
    auto Ww = mdl.Wdyn_w(ft);

    // random full-rank secondary collocation (the toy has no real grid to re-select
    // on; ANY Xb whose pair matrix B(q) spans the pair space is a valid secondary
    // basis at full rank -- the transfer algebra is what is under test)
    r2::rng_t rng(17);
    nda::array<cplx, 4> Xb(r2::ns, r2::nk, Nm_full, nc);
    for (auto& x : Xb) x = rng();

    // ---- per-q transfers at full rank + algebra checks --------------------------------
    std::vector<nda::array<cplx, 2>> t_q;
    for (long iq = 0; iq < r2::nk; ++iq) {
      auto B = r2::pair_matrix(Xb, 0, nc, mdl.kmq, iq);
      auto C = r2::pair_matrix(mdl.X_skPa, r2::C0, nc, mdl.kmq, iq);
      // s = B^dag B: Hermitian PSD
      nda::array<cplx, 2> s(Nm_full, Nm_full);
      s() = cplx(0.0);
      for (long m = 0; m < Nm_full; ++m)
        for (long n = 0; n < Nm_full; ++n)
          for (long I = 0; I < Npair; ++I) s(m, n) += std::conj(B(I, m)) * B(I, n);
      double herm = 0.0;
      for (long m = 0; m < Nm_full; ++m)
        for (long n = 0; n < Nm_full; ++n)
          herm = std::max(herm, std::abs(s(m, n) - std::conj(s(n, m))));
      REQUIRE(herm < 1e-13);
      for (int trial = 0; trial < 4; ++trial) {   // PSD: x^dag s x >= 0
        nda::array<cplx, 1> x(Nm_full);
        for (auto& v : x) v = rng();
        cplx xsx(0.0);
        double xx = 0.0;
        for (long m = 0; m < Nm_full; ++m) {
          xx += std::norm(x(m));
          for (long n = 0; n < Nm_full; ++n) xsx += std::conj(x(m)) * s(m, n) * x(n);
        }
        REQUIRE(xsx.real() > -1e-12 * xx);
        REQUIRE(std::abs(xsx.imag()) < 1e-12 * std::abs(xsx.real()) + 1e-14);
      }
      auto [t, sv, rank] = r2::solve_t(B, C, 1e-12);
      double cond_s = (sv(0) / sv(sv.size() - 1)) * (sv(0) / sv(sv.size() - 1));
      app_log(1, "r2 toy: q = {}: cond(s) = {}, rank = {}/{}", iq, cond_s, rank, Nm_full);
      REQUIRE(rank == Nm_full);
      // full rank: B t = C exactly (range(B) spans the pair space)
      double dmax = 0.0, cmax = 0.0;
      for (long I = 0; I < Npair; ++I)
        for (long P = 0; P < r2::Np; ++P) {
          cplx bt(0.0);
          for (long m = 0; m < Nm_full; ++m) bt += B(I, m) * t(m, P);
          dmax = std::max(dmax, std::abs(bt - C(I, P)));
          cmax = std::max(cmax, std::abs(C(I, P)));
        }
      app_log(1, "r2 toy: q = {}: max|B t - C| = {} (scale {})", iq, dmax, cmax);
      REQUIRE(dmax < 1e-10 * cmax);
      // eta on the bare core must vanish at full rank
      double e = r2::eta(B, C, t, mdl.Z_qPQ(iq, r_all, r_all));
      app_log(1, "r2 toy: q = {}: eta[Z] (full rank) = {}", iq, e);
      REQUIRE(e < 1e-10);
      t_q.push_back(std::move(t));
    }

    // ---- no-leak identity (Eq. 39) at full AND reduced rank: pure algebra -------------
    for (long Nm : {Nm_full, long(6)}) {
      nda::array<cplx, 4> Xb_r(r2::ns, r2::nk, Nm, nc);
      Xb_r = Xb(r_all, r_all, nda::range(0, Nm), r_all);
      for (long iq = 0; iq < r2::nk; ++iq) {
        auto B = r2::pair_matrix(Xb_r, 0, nc, mdl.kmq, iq);
        auto C = r2::pair_matrix(mdl.X_skPa, r2::C0, nc, mdl.kmq, iq);
        auto [t, sv, rank] = r2::solve_t(B, C, 1e-12);
        (void)sv; (void)rank;
        nda::array<cplx, 2> Pib(Nm, Nm), W(r2::Np, r2::Np);
        for (auto& v : Pib) v = rng();
        for (auto& v : W) v = rng();
        auto Pi_up = r2::upfold(t, Pib);
        auto Wb = r2::fold(t, W);
        cplx S1(0.0), S2(0.0);
        for (long M = 0; M < r2::Np; ++M)
          for (long N = 0; N < r2::Np; ++N) S1 += Pi_up(M, N) * W(N, M);
        for (long m = 0; m < Nm; ++m)
          for (long n = 0; n < Nm; ++n) S2 += Pib(m, n) * Wb(n, m);
        double rel = std::abs(S1 - S2) / std::max(std::abs(S2), 1e-300);
        app_log(1, "r2 toy: no-leak (Nm = {}, q = {}): rel = {}", Nm, iq, rel);
        REQUIRE(rel < 1e-12);
      }
    }

    // ---- folded kernel inputs at full rank ---------------------------------------------
    auto fold_inputs = [&](std::vector<nda::array<cplx, 2>> const& ts, long Nm) {
      nda::array<cplx, 3> Zb(r2::nk, Nm, Nm);
      nda::array<cplx, 4> Wtb(r2::nk, nt, Nm, Nm), Wwb(r2::nk, ft.nw_b(), Nm, Nm);
      for (long iq = 0; iq < r2::nk; ++iq) {
        Zb(iq, r_all, r_all) = r2::fold(ts[iq], mdl.Z_qPQ(iq, r_all, r_all));
        for (long it = 0; it < nt; ++it)
          Wtb(iq, it, r_all, r_all) = r2::fold(ts[iq], Wt(iq, it, r_all, r_all));
        for (long l = 0; l < ft.nw_b(); ++l)
          Wwb(iq, l, r_all, r_all) = r2::fold(ts[iq], Ww(iq, l, r_all, r_all));
      }
      return std::make_tuple(std::move(Zb), std::move(Wtb), std::move(Wwb));
    };
    auto [Zb, Wtb, Wwb] = fold_inputs(t_q, Nm_full);
    nda::array<cplx, 4> Xb_kernel(Xb);   // (ns, nk, Nm, nc) is already kernel layout
    nda::array<cplx, 5> G_CC(nt, r2::ns, r2::nk, nc, nc);
    G_CC = G(r_all, r_all, r_all, r2::Cw(), r2::Cw());

    // G~ = P_C G P_C for the GLOBAL Pi reference (the exact all-C cut, conservation
    // notes section 1.2 -- the object the secondary path computes by construction)
    nda::array<cplx, 5> Gproj(nt, r2::ns, r2::nk, r2::nbnd, r2::nbnd);
    Gproj() = cplx(0.0);
    Gproj(r_all, r_all, r_all, r2::Cw(), r2::Cw()) = G_CC;

    // ---- Sigma^C: global vs secondary at full rank (C-C block) ------------------------
    {
      nda::array<cplx, 5> Sig_g(nt, r2::ns, r2::nk, r2::nbnd, r2::nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, r2::Cw(), G, mdl.X_skPa,
                                                      Wt, mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                                      /*iq_gamma*/ 0, /*skip*/ false, Sig_g);
      nda::array<cplx, 5> Sig_s(nt, r2::ns, r2::nk, nc, nc);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, nda::range(0, nc), G_CC,
                                                      Xb_kernel, Wtb, Zb, mdl.kmq, mdl.qmin,
                                                      /*iq_gamma*/ 0, /*skip*/ false, Sig_s);
      double dmax = 0.0, smax = 0.0;
      for (long it = 0; it < nt; ++it)
        for (long ik = 0; ik < r2::nk; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long b = 0; b < nc; ++b) {
              cplx g = Sig_g(it, 0, ik, r2::C0 + a, r2::C0 + b);
              smax = std::max(smax, std::abs(g));
              dmax = std::max(dmax, std::abs(g - Sig_s(it, 0, ik, a, b)));
            }
      app_log(1, "r2 toy: Sigma^C secondary vs global (C-C block, full rank): "
                 "max|diff| = {}, scale = {}, rel = {}", dmax, smax, dmax / smax);
      REQUIRE(smax > 1e-10);
      REQUIRE(dmax < 1e-9 * std::max(smax, 1.0));
    }

    // ---- Pi^C: global (on G~) vs upfolded secondary at full rank + conservation -------
    {
      nda::array<cplx, 4> Pi_g(ft.nw_b(), r2::nk, r2::Np, r2::Np);
      Pi_g() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, Gproj, mdl.X_skPa, mdl.Z_qPQ, &Ww,
                                   mdl.kmq, mdl.kpq, r2::Cw(), Pi_g, 0, 1);
      nda::array<cplx, 4> Pi_s(ft.nw_b(), r2::nk, Nm_full, Nm_full);
      Pi_s() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, Xb_kernel, Zb, &Wwb,
                                   mdl.kmq, mdl.kpq, nda::range(0, nc), Pi_s, 0, 1);
      nda::array<cplx, 4> Pi_up(ft.nw_b(), r2::nk, r2::Np, r2::Np);
      for (long l = 0; l < ft.nw_b(); ++l)
        for (long iq = 0; iq < r2::nk; ++iq)
          Pi_up(l, iq, r_all, r_all) = r2::upfold(t_q[iq], Pi_s(l, iq, r_all, r_all));
      double dmax = 0.0, smax = 0.0;
      for (auto const& v : Pi_g) smax = std::max(smax, std::abs(v));
      for (long l = 0; l < ft.nw_b(); ++l)
        for (long iq = 0; iq < r2::nk; ++iq)
          for (long M = 0; M < r2::Np; ++M)
            for (long N = 0; N < r2::Np; ++N)
              dmax = std::max(dmax, std::abs(Pi_g(l, iq, M, N) - Pi_up(l, iq, M, N)));
      app_log(1, "r2 toy: Pi^C upfolded secondary vs global[G~] (full rank): "
                 "max|diff| = {}, scale = {}, rel = {}", dmax, smax, dmax / smax);
      REQUIRE(smax > 1e-10);
      REQUIRE(dmax < 1e-9 * std::max(smax, 1.0));

      // conservation identity in the SECONDARY path: Sigma_s paired with G_CC,
      // Pibar_s paired with the folded rungs; and (no-leak at the trace level) the
      // upfolded Pi paired with the GLOBAL rungs gives the same S_PW.
      nda::array<cplx, 5> Sig_s(nt, r2::ns, r2::nk, nc, nc);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, nda::range(0, nc), G_CC,
                                                      Xb_kernel, Wtb, Zb, mdl.kmq, mdl.qmin,
                                                      0, false, Sig_s);
      nda::array<double, 1> x0(1);
      x0(0) = -1.0;
      nda::array<cplx, 2> T0row(ft.construct_tau_interpolate_matrix(x0));
      cplx S_SG = r2::trace_sigma_G(tools, Sig_s, G_CC);
      cplx S_PW_bar = r2::trace_pi_W(ft, tools, T0row, Pi_s, Zb, Wtb, r2::nk);
      cplx S_PW_up = r2::trace_pi_W(ft, tools, T0row, Pi_up, mdl.Z_qPQ, Wt, r2::nk);
      double rel = r2::rel_residual(S_SG, S_PW_bar);
      double leak = std::abs(S_PW_bar - S_PW_up) / std::max(std::abs(S_PW_bar), 1e-300);
      app_log(1, "r2 toy: conservation (secondary): S_SG = ({}, {}), S_PW = ({}, {}), "
                 "rel = {}", S_SG.real(), S_SG.imag(), S_PW_bar.real(), S_PW_bar.imag(), rel);
      app_log(1, "r2 toy: trace-level no-leak: |S_PW(bar) - S_PW(upfold vs global)| rel = {}",
              leak);
      REQUIRE(std::abs(S_SG) > 1e-12);
      REQUIRE(rel < 1e-8);
      REQUIRE(leak < 1e-10);
    }

    // ---- reduced rank: eta > 0, kernels finite, controlled deviation (reported) -------
    {
      const long Nm = 6;
      nda::array<cplx, 4> Xb_r(r2::ns, r2::nk, Nm, nc);
      Xb_r = Xb(r_all, r_all, nda::range(0, Nm), r_all);
      std::vector<nda::array<cplx, 2>> ts;
      double eta_max = 0.0;
      for (long iq = 0; iq < r2::nk; ++iq) {
        auto B = r2::pair_matrix(Xb_r, 0, nc, mdl.kmq, iq);
        auto C = r2::pair_matrix(mdl.X_skPa, r2::C0, nc, mdl.kmq, iq);
        auto [t, sv, rank] = r2::solve_t(B, C, 1e-12);
        (void)sv; (void)rank;
        eta_max = std::max(eta_max, r2::eta(B, C, t, mdl.Z_qPQ(iq, r_all, r_all)));
        ts.push_back(std::move(t));
      }
      auto [Zb_r, Wtb_r, Wwb_r] = fold_inputs(ts, Nm);
      (void)Wwb_r;
      nda::array<cplx, 5> Sig_r(nt, r2::ns, r2::nk, nc, nc);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, nda::range(0, nc), G_CC,
                                                      Xb_r, Wtb_r, Zb_r, mdl.kmq, mdl.qmin,
                                                      0, false, Sig_r);
      double smax = 0.0;
      long nbad = 0;
      for (auto const& v : Sig_r) {
        if (not std::isfinite(std::abs(v))) ++nbad;
        smax = std::max(smax, std::abs(v));
      }
      app_log(1, "r2 toy: reduced rank Nm = {}: max_q eta[Z] = {}, max|Sigma^C| = {} "
                 "(full-rank scale for comparison logged above)", Nm, eta_max, smax);
      REQUIRE(nbad == 0);
      REQUIRE(eta_max > 1e-8);   // genuinely truncated
    }
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  TEST_CASE("vertex_refinement2_lih", "[methods][vertex][refinement2][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_refinement2_lih skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax = 6.0: the vertex [A-comp] headroom requirement (pi design section 4b)
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_r2";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    const nda::range Crng(1, 3);   // straddles the LiH gap (HOMO = 1, LUMO = 2)
    const long nc = Crng.size();

    // ---------------- state: one plain scGW iteration + RPA-W rebuild ------------------
    // (the identity/agreement checks are algebraic in (G, W): any consistent pair is
    // valid; same isolation choice as the conservation test)
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    app_log(1, "r2 lih: baseline scGW iteration: e_hf = {}, e_corr = {}", e_hf_0, e_corr_0);
    REQUIRE(std::isfinite(e_hf_0));
    REQUIRE(std::isfinite(e_corr_0));
    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());

    // ---------------- replicated glue (vertex_t.cpp pattern) ---------------------------
    auto MF = thc.MF();
    const long nkpts = MF->nkpts(), nqpts = MF->nqpts();
    const long Np = thc.Np(), nbnd = MF->nbnd();
    utils::check(nqpts == MF->nqpts_ibz() and nkpts == MF->nkpts_ibz() and nqpts == nkpts,
                 "vertex_refinement2_lih: needs a symmetry-free mesh.");
    auto G_loc = mb_state.sG_tskij.value().local();
    const long nt = G_loc.shape(0), ns = G_loc.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    iaft_tools tools(ft);
    const long Npair = ns * nkpts * nc * nc;   // = 32 for LiH-222, C = [1,3)

    nda::array<cplx, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik) X_skPa(is, ik, r_all, r_all) = thc.X(is, 0, ik);
    nda::array<cplx, 3> Z_qPQ(nqpts, Np, Np);
    for (long iq = 0; iq < nqpts; ++iq) Z_qPQ(iq, r_all, r_all) = thc.Z(int(iq));

    nda::array<cplx, 4> Wt_qtPQ(nqpts, nt, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      nda::array<cplx, 4> W_half(nqpts, nt_half, Np, Np);
      W_half() = cplx(0.0);
      W_half(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(W_half.data(), W_half.size(), std::plus<>{});
      for (long it = 0; it < nt; ++it) {
        long ith = std::min(it, nt - it - 1);
        Wt_qtPQ(r_all, it, r_all, r_all) = W_half(r_all, ith, r_all, r_all);
      }
    }
    nda::array<cplx, 4> Wdyn_qwPQ(nqpts, tools.nw_b, Np, Np);
    {
      long nw_b = tools.nw_b;
      long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
      nda::array<cplx, 3> W_wpos(nw_half, Np, Np);
      for (long iq = 0; iq < nqpts; ++iq) {
        auto W_t = Wt_qtPQ(iq, nda::range(0, nt_half), r_all, r_all);
        ft.tau_to_w_PHsym(W_t, W_wpos);
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ(iq, l, r_all, r_all) = W_wpos(lpos, r_all, r_all);
        }
      }
    }
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
        for (long i = 0; i < 3; ++i) d += std::abs(Qpts(iq, i) - std::round(Qpts(iq, i)));
        if (d < 1e-8) { iq_gamma = iq; break; }
      }
      REQUIRE(iq_gamma >= 0);
    }

    // ---------------- STAGE A: restricted point selection + eta(q, nu) table -----------
    // ONE selection at max = 32 (greedy pivoted Cholesky => nested: first N points of
    // the max-32 selection ARE the max-N selection); rank scan on nested subsets.
    nda::array<cplx, 4> Xb_full;   // (ns, nk, Nm_sel, nc) kernel layout
    long Nm_sel = 0;
    {
      ptree pt_b;
      pt_b.put("thresh", 1e-13);
      pt_b.put("chol_block_size", 1);   // blocked pivoting is unstable at tiny thresh
      methods::thc builder(mf.get(), *mpi_context, pt_b, /*print*/ false);
      auto [ipts, dXa, dXb] = builder.interpolating_points<HOST_MEMORY>(
          int(iq_gamma), int(Npair), Crng, Crng);
      (void)dXb;
      Nm_sel = ipts.extent(0);
      app_log(1, "r2 lih: restricted point selection returned N_m = {} (requested {})",
              Nm_sel, Npair);
      REQUIRE(Nm_sel >= 8);
      nda::array<cplx, 4> Xa(ns, nkpts, nc, Nm_sel);
      Xa() = cplx(0.0);
      Xa(dXa.local_range(0), dXa.local_range(1), dXa.local_range(2), dXa.local_range(3)) =
          dXa.local();
      mpi_context->comm.all_reduce_in_place_n(Xa.data(), Xa.size(), std::plus<>{});
      Xb_full = nda::array<cplx, 4>(ns, nkpts, Nm_sel, nc);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long m = 0; m < Nm_sel; ++m) Xb_full(is, ik, m, a) = Xa(is, ik, a, m);
    }

    std::vector<long> ranks;
    for (long n : {8L, 16L, 24L, 32L})
      if (n <= Nm_sel) ranks.push_back(n);
    if (ranks.empty() or ranks.back() != Nm_sel) ranks.push_back(Nm_sel);

    // per-rank transfers + eta table (eta on Z and on dW(tau = 0) slices, all q)
    std::vector<std::vector<nda::array<cplx, 2>>> t_of_rank(ranks.size());
    std::vector<double> etaZ_max(ranks.size(), 0.0), etaW_max(ranks.size(), 0.0);
    app_log(1, "r2 lih: ---- eta(q) table (Eq. 40): rows = N_m, cols = q ----");
    for (size_t ir = 0; ir < ranks.size(); ++ir) {
      long Nm = ranks[ir];
      nda::array<cplx, 4> Xb(ns, nkpts, Nm, nc);
      Xb = Xb_full(r_all, r_all, nda::range(0, Nm), r_all);
      std::string rowZ, rowW;
      double cond_max = 0.0;
      for (long iq = 0; iq < nqpts; ++iq) {
        auto B = r2::pair_matrix(Xb, 0, nc, kmq, iq);
        auto C = r2::pair_matrix(X_skPa, Crng.first(), nc, kmq, iq);
        auto [t, sv, rank] = r2::solve_t(B, C, 1e-8);
        (void)rank;
        cond_max = std::max(cond_max, (sv(0) / std::max(sv(sv.size() - 1), 1e-300)) *
                                      (sv(0) / std::max(sv(sv.size() - 1), 1e-300)));
        double eZ = r2::eta(B, C, t, Z_qPQ(iq, r_all, r_all));
        double eW = r2::eta(B, C, t, Wt_qtPQ(iq, 0, r_all, r_all));
        etaZ_max[ir] = std::max(etaZ_max[ir], eZ);
        etaW_max[ir] = std::max(etaW_max[ir], eW);
        auto sci = [](double v) {
          std::ostringstream os;
          os << " " << std::scientific << std::setprecision(3) << v;
          return os.str();
        };
        rowZ += sci(eZ);
        rowW += sci(eW);
        t_of_rank[ir].push_back(std::move(t));
      }
      app_log(1, "r2 lih: N_m = {:2}: eta[Z](q)  = {}   (max = {:.3e}, max cond(s) = {:.3e})",
              Nm, rowZ, etaZ_max[ir], cond_max);
      app_log(1, "r2 lih: N_m = {:2}: eta[dW](q) = {}   (max = {:.3e})",
              Nm, rowW, etaW_max[ir]);
    }
    // monotone decreasing in N_m (nested bases; small slack for the truncated solve)
    for (size_t ir = 0; ir + 1 < ranks.size(); ++ir) {
      REQUIRE(etaZ_max[ir + 1] <= etaZ_max[ir] * 1.05 + 1e-12);
      REQUIRE(etaW_max[ir + 1] <= etaW_max[ir] * 1.05 + 1e-12);
    }
    // -> 0 once range(B) spans the C pair space (full rank)
    if (ranks.back() == Npair) {
      REQUIRE(etaZ_max.back() < 1e-5);
      REQUIRE(etaW_max.back() < 1e-5);
    }

    // ---------------- STAGE B: kernel-level secondary vs global at full rank -----------
    auto& t_full = t_of_rank.back();
    const long NmB = ranks.back();
    nda::array<cplx, 5> G_CC(nt, ns, nkpts, nc, nc);
    G_CC = G_loc(r_all, r_all, r_all, Crng, Crng);
    nda::array<cplx, 5> Gproj(nt, ns, nkpts, nbnd, nbnd);
    Gproj() = cplx(0.0);
    Gproj(r_all, r_all, r_all, Crng, Crng) = G_CC;
    nda::array<cplx, 4> Xb_B(ns, nkpts, NmB, nc);
    Xb_B = Xb_full(r_all, r_all, nda::range(0, NmB), r_all);

    nda::array<cplx, 3> Zb(nqpts, NmB, NmB);
    nda::array<cplx, 4> Wtb(nqpts, nt, NmB, NmB), Wwb(nqpts, tools.nw_b, NmB, NmB);
    for (long iq = 0; iq < nqpts; ++iq) {
      Zb(iq, r_all, r_all) = r2::fold(t_full[iq], Z_qPQ(iq, r_all, r_all));
      for (long it = 0; it < nt; ++it)
        Wtb(iq, it, r_all, r_all) = r2::fold(t_full[iq], Wt_qtPQ(iq, it, r_all, r_all));
      for (long l = 0; l < tools.nw_b; ++l)
        Wwb(iq, l, r_all, r_all) = r2::fold(t_full[iq], Wdyn_qwPQ(iq, l, r_all, r_all));
    }

    // Sigma^C: global (full externals; Sigma^C[G~] = Sigma^C[G] on the internals) vs
    // secondary -- compare on the C-C block
    nda::array<cplx, 5> Sig_g(nt, ns, nkpts, nbnd, nbnd);
    solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, mpi_context->comm, Crng, G_loc,
                                                    X_skPa, Wt_qtPQ, Z_qPQ, kmq, qmin,
                                                    iq_gamma, /*skip*/ false, Sig_g);
    nda::array<cplx, 5> Sig_s(nt, ns, nkpts, nc, nc);
    solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, mpi_context->comm,
                                                    nda::range(0, nc), G_CC, Xb_B, Wtb, Zb,
                                                    kmq, qmin, iq_gamma, false, Sig_s);
    {
      double dmax = 0.0, scc = 0.0, sfull = 0.0;
      for (auto const& v : Sig_g) sfull = std::max(sfull, std::abs(v));
      for (long it = 0; it < nt; ++it)
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nkpts; ++ik)
            for (long a = 0; a < nc; ++a)
              for (long b = 0; b < nc; ++b) {
                cplx g = Sig_g(it, is, ik, Crng.first() + a, Crng.first() + b);
                scc = std::max(scc, std::abs(g));
                dmax = std::max(dmax, std::abs(g - Sig_s(it, is, ik, a, b)));
              }
      app_log(1, "r2 lih: Sigma^C at N_m = {}: max|Sig_glob| (full) = {}, (C-C) = {}, "
                 "max|Sig_sec - Sig_glob|_CC = {}, rel = {}",
              NmB, sfull, scc, dmax, dmax / std::max(scc, 1e-300));
      REQUIRE(scc > 1e-12);
      // NmB is the selection's returned count = the NUMERICAL rank of the C pair
      // metric, so the downfold is complete to the svd_tol class there regardless of
      // the counting rank 32 (measured: rel = 4.7e-9 at N_m = 24)
      REQUIRE(dmax < 1e-4 * scc);
    }

    // Pi^C: global on G~ (the exact all-C cut) vs upfolded secondary
    nda::array<cplx, 4> Pi_g(tools.nw_b, nqpts, Np, Np);
    Pi_g() = cplx(0.0);
    vertex_pi::pi_c_accumulate_w(ft, tools, Gproj, X_skPa, Z_qPQ, &Wdyn_qwPQ, kmq, kpq,
                                 Crng, Pi_g, mpi_context->comm.rank(),
                                 mpi_context->comm.size());
    mpi_context->comm.all_reduce_in_place_n(Pi_g.data(), Pi_g.size(), std::plus<>{});
    nda::array<cplx, 4> Pi_s(tools.nw_b, nqpts, NmB, NmB);
    Pi_s() = cplx(0.0);
    vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, Xb_B, Zb, &Wwb, kmq, kpq,
                                 nda::range(0, nc), Pi_s, mpi_context->comm.rank(),
                                 mpi_context->comm.size());
    mpi_context->comm.all_reduce_in_place_n(Pi_s.data(), Pi_s.size(), std::plus<>{});
    nda::array<cplx, 4> Pi_up(tools.nw_b, nqpts, Np, Np);
    for (long l = 0; l < tools.nw_b; ++l)
      for (long iq = 0; iq < nqpts; ++iq)
        Pi_up(l, iq, r_all, r_all) = r2::upfold(t_full[iq], Pi_s(l, iq, r_all, r_all));
    {
      double dmax = 0.0, smax = 0.0;
      for (auto const& v : Pi_g) smax = std::max(smax, std::abs(v));
      for (long l = 0; l < tools.nw_b; ++l)
        for (long iq = 0; iq < nqpts; ++iq)
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N)
              dmax = std::max(dmax, std::abs(Pi_g(l, iq, M, N) - Pi_up(l, iq, M, N)));
      app_log(1, "r2 lih: Pi^C at N_m = {}: max|Pi_glob[G~]| = {}, "
                 "max|Pi_upfold - Pi_glob| = {}, rel = {}",
              NmB, smax, dmax, dmax / std::max(smax, 1e-300));
      REQUIRE(smax > 1e-12);
      // see the Sigma^C note above (measured: rel = 8.9e-10 at N_m = 24)
      REQUIRE(dmax < 1e-4 * smax);
    }

    // ---------------- conservation identity in the SECONDARY path ----------------------
    // ignore_g0: Sigma_s paired with G_CC; Pibar_s paired with the folded rungs; the
    // trace-level no-leak equates that to the upfolded Pi paired with the global rungs.
    nda::array<double, 1> x0(1);
    x0(0) = -1.0;
    nda::array<cplx, 2> T0row(ft.construct_tau_interpolate_matrix(x0));
    {
      cplx S_SG = r2::trace_sigma_G(tools, Sig_s, G_CC);
      cplx S_PW_bar = r2::trace_pi_W(ft, tools, T0row, Pi_s, Zb, Wtb, nkpts);
      cplx S_PW_up = r2::trace_pi_W(ft, tools, T0row, Pi_up, Z_qPQ, Wt_qtPQ, nkpts);
      double rel = r2::rel_residual(S_SG, S_PW_bar);
      double leak = std::abs(S_PW_bar - S_PW_up) / std::max(std::abs(S_PW_bar), 1e-300);
      app_log(1, "r2 lih: conservation (secondary, ignore_g0): S_SG = ({}, {}), "
                 "S_PW = ({}, {}), rel = {}",
              S_SG.real(), S_SG.imag(), S_PW_bar.real(), S_PW_bar.imag(), rel);
      app_log(1, "r2 lih: trace-level no-leak: rel = {}", leak);
      REQUIRE(std::abs(S_SG) > 1e-12);
      REQUIRE(rel < 1e-3);
      REQUIRE(leak < 1e-10);
    }
    // gygi: the head-augmented W^(Gamma) downfolds automatically through t (rank-1
    // t H t^dag); both kernels and the trace consume the SAME folded augmented arrays.
    {
      REQUIRE(mb_state.eps_inv_head.has_value());
      auto& eps = mb_state.eps_inv_head.value();
      const double xi = MF->madelung();
      auto chi = thc.basis_head();
      REQUIRE(xi != 0.0);
      nda::array<cplx, 2> H(Np, Np);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q)
          H(P, Q) = double(nkpts) * xi * std::conj(chi(iq_gamma, P)) * chi(iq_gamma, Q);
      nda::array<cplx, 3> Z_aug(Z_qPQ);
      Z_aug(iq_gamma, r_all, r_all) += H;
      nda::array<cplx, 4> Wt_aug(Wt_qtPQ);
      for (long it = 0; it < nt; ++it) {
        long ith = std::min(it, nt - it - 1);
        Wt_aug(iq_gamma, it, r_all, r_all) += cplx(eps(ith).real()) * H;
      }
      nda::array<cplx, 4> Wdyn_aug(Wdyn_qwPQ);
      {
        long nw_b = tools.nw_b;
        long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
        nda::array<cplx, 3> W_wpos(nw_half, Np, Np);
        auto W_t = Wt_aug(iq_gamma, nda::range(0, nt_half), r_all, r_all);
        ft.tau_to_w_PHsym(W_t, W_wpos);
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_aug(iq_gamma, l, r_all, r_all) = W_wpos(lpos, r_all, r_all);
        }
      }
      // fold the AUGMENTED arrays with the same t
      nda::array<cplx, 3> Zb_a(Zb);
      Zb_a(iq_gamma, r_all, r_all) = r2::fold(t_full[iq_gamma], Z_aug(iq_gamma, r_all, r_all));
      nda::array<cplx, 4> Wtb_a(Wtb), Wwb_a(Wwb);
      for (long it = 0; it < nt; ++it)
        Wtb_a(iq_gamma, it, r_all, r_all) =
            r2::fold(t_full[iq_gamma], Wt_aug(iq_gamma, it, r_all, r_all));
      for (long l = 0; l < tools.nw_b; ++l)
        Wwb_a(iq_gamma, l, r_all, r_all) =
            r2::fold(t_full[iq_gamma], Wdyn_aug(iq_gamma, l, r_all, r_all));

      nda::array<cplx, 5> Sig_sg(nt, ns, nkpts, nc, nc);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, mpi_context->comm,
                                                      nda::range(0, nc), G_CC, Xb_B,
                                                      Wtb_a, Zb_a, kmq, qmin, iq_gamma,
                                                      false, Sig_sg);
      nda::array<cplx, 4> Pi_sg(tools.nw_b, nqpts, NmB, NmB);
      Pi_sg() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, Xb_B, Zb_a, &Wwb_a, kmq, kpq,
                                   nda::range(0, nc), Pi_sg, mpi_context->comm.rank(),
                                   mpi_context->comm.size());
      mpi_context->comm.all_reduce_in_place_n(Pi_sg.data(), Pi_sg.size(), std::plus<>{});
      cplx S_SG_g = r2::trace_sigma_G(tools, Sig_sg, G_CC);
      cplx S_PW_g = r2::trace_pi_W(ft, tools, T0row, Pi_sg, Zb_a, Wtb_a, nkpts);
      double rel_g = r2::rel_residual(S_SG_g, S_PW_g);
      app_log(1, "r2 lih: conservation (secondary, gygi head downfolded): "
                 "S_SG = ({}, {}), S_PW = ({}, {}), rel = {}",
              S_SG_g.real(), S_SG_g.imag(), S_PW_g.real(), S_PW_g.imag(), rel_g);
      REQUIRE(std::isfinite(rel_g));
      REQUIRE(rel_g < 1e-3);
    }

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  // End-to-end: 1-iteration scGW with BOTH cuts through the PRODUCTION vertex_t
  // plumbing (lazy secondary-basis build, per-iteration folding, C-C placement,
  // upfold), global vs secondary at N_m = 32 and N_m = 8.
  TEST_CASE("vertex_refinement2_lih_end2end", "[methods][vertex][refinement2][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_refinement2_lih_end2end skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_r2_e2e";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](std::string const& isdf_mode, long isdf_rank) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "ignore_g0", isdf_mode, isdf_rank, 1e-8);
      REQUIRE(vtx.active());
      // BOTH cuts (production wiring, MBPT_drivers pattern)
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     1, false, 1e-9, true);
      mpi_context->comm.barrier();
      long nm = vtx.secondary_rank();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, nm);
    };

    auto [e_hf_g, e_corr_g, nm_g] = run("global", -1);
    auto [e_hf_32, e_corr_32, nm_32] = run("secondary", 32);
    auto [e_hf_8, e_corr_8, nm_8] = run("secondary", 8);
    (void)nm_g;

    app_log(1, "r2 e2e: ---- end-to-end table (LiH-222 nosym, 1-iteration scGW, both "
               "cuts, ignore_g0) ----");
    app_log(1, "r2 e2e: global            e_hf = {}, e_corr = {}", e_hf_g, e_corr_g);
    app_log(1, "r2 e2e: secondary N_m={:2} e_hf = {}, e_corr = {}  (D e_corr vs global = {})",
            nm_32, e_hf_32, e_corr_32, e_corr_32 - e_corr_g);
    app_log(1, "r2 e2e: secondary N_m={:2} e_hf = {}, e_corr = {}  (D e_corr vs N_m=32 = {})",
            nm_8, e_hf_8, e_corr_8, e_corr_8 - e_corr_32);
    app_log(1, "r2 e2e: NOTE: STRICT C-C externals (theory-owner ruling, memo DECISION "
               "2) -- both paths now\n"
               "        evaluate the SAME functional cuts; at full secondary rank the "
               "residual difference is\n"
               "        pure downfold truncation (svd_tol class), asserted below.");

    for (double e : {e_hf_g, e_corr_g, e_hf_32, e_corr_32, e_hf_8, e_corr_8})
      REQUIRE(std::isfinite(e));
    REQUIRE(nm_8 == 8);
    REQUIRE(nm_32 >= 8);
    // strict externals: identical functional; at the TOP secondary rank (the selection
    // returns the NUMERICAL rank of the C pair metric, <= the counting rank 32) global
    // and secondary must agree to the downfold/kernel accuracy class.
    // Measured (N_m = 24 returned): |Delta e_hf| = 2e-15, |Delta e_corr| = 2.2e-15.
    if (nm_32 < 32)
      app_log(1, "r2 e2e: [NOTE] selection returned N_m = {} < 32 (numerical pair-metric "
                 "rank); the downfold is complete there.", nm_32);
    REQUIRE(std::abs(e_hf_32 - e_hf_g) < 1e-5);
    REQUIRE(std::abs(e_corr_32 - e_corr_g) < 1e-5);
    // the N_m truncation is a controlled perturbation (monitored by eta)
    REQUIRE(std::abs(e_corr_8 - e_corr_32) < 5e-3);
#endif  // ENABLE_DLR
  }

} // namespace bdft_tests
