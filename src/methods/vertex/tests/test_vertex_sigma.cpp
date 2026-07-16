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
 * Tests for the ISDF-Vertex Sigma^C kernel (Phase 1c), see
 * notes/sigma_c_kernel_design.md.
 *
 *  - fused_vs_batched: the fused orbital-window channel-algebra kernel vs a
 *    reference that loops double_boson_conv over orbital tuples x (a,b) with
 *    independently-coded orbital W blocks. Same algebra => near machine
 *    precision. Pins the entire family fusion, index routing and both
 *    instantaneous paths.
 *  - dense_matsubara_arbiter: single (k,qx,qy) combo, dynamic-only W, dense
 *    truncated double Matsubara sums of the verified orbital spec (prefactor
 *    +1) with analytic pole evaluations, vs the batched-primitive reference
 *    (which fused_vs_batched ties to the fused kernel). Pins the +1 sign, the
 *    THC dictionary and the 1/beta^2 normalization.
 *  - instantaneous_reduction: both rungs instantaneous (W_dyn = 0) vs an
 *    independently-coded bare second-order-exchange tau-contraction
 *    (index-reflection for beta-tau, no DLR machinery).
 *  - lih_smoke: one THC-scGW iteration on LiH-222 (nosym) with the vertex
 *    active, then an isolated Sigma^C evaluation on the resulting state:
 *    finite, non-zero, hermiticity deviation reported.
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
#include "numerics/imag_axes_ft/iaft_dconv.hpp"
#include "methods/vertex/vertex_sigma.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;
  static auto const s_all = nda::range::all;

  namespace sig_toy {

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
      nda::array<long, 1> qmin;     // (nq): -q

      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd), X_skPa(ns, nk, Np, nbnd),
                  Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np), kmq(nk, nk), qmin(nk) {
        rng_t rng(29);
        const double base[3] = {-0.67, -0.11, 0.52};
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.06 * double(k);
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
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.25) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        for (long q = 0; q < nk; ++q) {
          qmin(q) = (nk - q) % nk;
          for (long k = 0; k < nk; ++k) kmq(q, k) = (k - q + nk) % nk;
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

      // analytic G(k, z)
      nda::array<cplx, 2> G_z(long k, cplx z) const {
        nda::array<cplx, 2> G(nbnd, nbnd);
        G() = cplx(0.0);
        for (long r = 0; r < nbnd; ++r) {
          cplx d = 1.0 / (z - eps(k, r));
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) G(i, j) += d * Pr(k, r, i, j);
        }
        return G;
      }

      // scalar dynamic profile s(nu) with W_dyn = M * s
      static double s_of(double nu) { return 2.0 * OmX / (OmX * OmX + nu * nu); }

      // W_dyn(q, tau) on the FULL fermionic tau mesh (input format of the kernel)
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
          Wt(q, s_all, s_all, s_all) = Wtau;
        }
        return Wt;
      }
    };

    // ---- independently-coded orbital W blocks (the section-1 dictionary) --------------
    // Wx block: (b, p1, p3, p3') from core A_PQ at (k, k-qx | k+qy, k-qx+qy)
    inline nda::array<cplx, 4> orb_Wx(model_t const& m, nda::array<cplx, 2> const& A,
                                      long ik, long ikmqx, long ikpqy, long ikmqxpqy) {
      nda::array<cplx, 4> out(nbnd, ncw, ncw, ncw);
      out() = cplx(0.0);
      for (long b = 0; b < nbnd; ++b)
        for (long j1 = 0; j1 < ncw; ++j1)
          for (long j3 = 0; j3 < ncw; ++j3)
            for (long j3p = 0; j3p < ncw; ++j3p)
              for (long P = 0; P < Np; ++P)
                for (long Q = 0; Q < Np; ++Q)
                  out(b, j1, j3, j3p) += std::conj(m.X_skPa(0, ik, P, b)) *
                      m.X_skPa(0, ikmqx, P, C0 + j1) * A(P, Q) *
                      m.X_skPa(0, ikpqy, Q, C0 + j3) *
                      std::conj(m.X_skPa(0, ikmqxpqy, Q, C0 + j3p));
      return out;
    }
    // Wy block: (a, p2', p4, p4') from core A_PQ at (k+qy, k | k-qx+qy, k-qx)
    inline nda::array<cplx, 4> orb_Wy(model_t const& m, nda::array<cplx, 2> const& A,
                                      long ik, long ikmqx, long ikpqy, long ikmqxpqy) {
      nda::array<cplx, 4> out(nbnd, ncw, ncw, ncw);
      out() = cplx(0.0);
      for (long a = 0; a < nbnd; ++a)
        for (long j2p = 0; j2p < ncw; ++j2p)
          for (long j4 = 0; j4 < ncw; ++j4)
            for (long j4p = 0; j4p < ncw; ++j4p)
              for (long P = 0; P < Np; ++P)
                for (long Q = 0; Q < Np; ++Q)
                  out(a, j2p, j4, j4p) += std::conj(m.X_skPa(0, ikpqy, P, C0 + j2p)) *
                      m.X_skPa(0, ik, P, a) * A(P, Q) *
                      m.X_skPa(0, ikmqxpqy, Q, C0 + j4) *
                      std::conj(m.X_skPa(0, ikmqx, Q, C0 + j4p));
      return out;
    }

    // ---- batched double_boson_conv reference for one (k,qx,qy) combo ------------------
    // Returns S_t(tau, a, b) = sum over orbital tuples of the scalar chains (NO 1/Nk^2).
    inline nda::array<cplx, 3> combo_reference(imag_axes_ft::IAFT const& ft, model_t const& m,
                                               nda::array<cplx, 5> const& G, long ik,
                                               long iqx, long iqy, bool with_dyn, bool with_Z) {
      long nt = ft.nt_f(), nw_b = ft.nw_b();
      auto wnb = ft.wn_mesh_b();
      long ikmqx = m.kmq(iqx, ik);
      long ikpqy = m.kmq(m.qmin(iqy), ik);
      long ikmqxpqy = m.kmq(m.qmin(iqy), ikmqx);

      nda::array<cplx, 2> zero(Np, Np);
      zero() = cplx(0.0);
      auto MX = orb_Wx(m, with_dyn ? nda::array<cplx, 2>(m.M_qPQ(iqx, s_all, s_all)) : zero,
                       ik, ikmqx, ikpqy, ikmqxpqy);
      auto ZX = orb_Wx(m, with_Z ? nda::array<cplx, 2>(m.Z_qPQ(iqx, s_all, s_all)) : zero,
                       ik, ikmqx, ikpqy, ikmqxpqy);
      auto MY = orb_Wy(m, with_dyn ? nda::array<cplx, 2>(m.M_qPQ(iqy, s_all, s_all)) : zero,
                       ik, ikmqx, ikpqy, ikmqxpqy);
      auto ZY = orb_Wy(m, with_Z ? nda::array<cplx, 2>(m.Z_qPQ(iqy, s_all, s_all)) : zero,
                       ik, ikmqx, ikpqy, ikmqxpqy);

      const long nbat = nbnd * nbnd * ncw * ncw * ncw * ncw * ncw * ncw;
      auto J = [](long a, long b, long j1, long j2p, long j3, long j3p, long j4, long j4p) {
        return ((((((a * nbnd + b) * ncw + j1) * ncw + j2p) * ncw + j3) * ncw + j3p) * ncw + j4) * ncw + j4p;
      };
      nda::array<cplx, 2> B_t(nt, nbat), C_t(nt, nbat), D_t(nt, nbat), S_t(nt, nbat);
      nda::array<cplx, 2> Wx_w(nw_b, nbat), Wy_w(nw_b, nbat);
      nda::array<cplx, 1> cx(nbat), cy(nbat);
      for (long a = 0; a < nbnd; ++a)
        for (long b = 0; b < nbnd; ++b)
          for (long j1 = 0; j1 < ncw; ++j1)
            for (long j2p = 0; j2p < ncw; ++j2p)
              for (long j3 = 0; j3 < ncw; ++j3)
                for (long j3p = 0; j3p < ncw; ++j3p)
                  for (long j4 = 0; j4 < ncw; ++j4)
                    for (long j4p = 0; j4p < ncw; ++j4p) {
                      long jj = J(a, b, j1, j2p, j3, j3p, j4, j4p);
                      for (long it = 0; it < nt; ++it) {
                        B_t(it, jj) = G(it, 0, ikpqy,    C0 + j3, C0 + j2p);
                        C_t(it, jj) = G(it, 0, ikmqxpqy, C0 + j4, C0 + j3p);
                        D_t(it, jj) = G(it, 0, ikmqx,    C0 + j1, C0 + j4p);
                      }
                      for (long mm = 0; mm < nw_b; ++mm) {
                        double nu = double(wnb(mm)) * M_PI / beta;
                        Wx_w(mm, jj) = model_t::s_of(nu) * MX(b, j1, j3, j3p);
                        Wy_w(mm, jj) = model_t::s_of(nu) * MY(a, j2p, j4, j4p);
                      }
                      cx(jj) = ZX(b, j1, j3, j3p);
                      cy(jj) = ZY(a, j2p, j4, j4p);
                    }
      imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, cx, cy);

      nda::array<cplx, 3> out(nt, nbnd, nbnd);
      out() = cplx(0.0);
      for (long it = 0; it < nt; ++it)
        for (long a = 0; a < nbnd; ++a)
          for (long b = 0; b < nbnd; ++b)
            for (long j1 = 0; j1 < ncw; ++j1)
              for (long j2p = 0; j2p < ncw; ++j2p)
                for (long j3 = 0; j3 < ncw; ++j3)
                  for (long j3p = 0; j3p < ncw; ++j3p)
                    for (long j4 = 0; j4 < ncw; ++j4)
                      for (long j4p = 0; j4p < ncw; ++j4p)
                        out(it, a, b) += S_t(it, J(a, b, j1, j2p, j3, j3p, j4, j4p));
      return out;
    }

  } // namespace sig_toy

  TEST_CASE("vertex_sigma_toy", "[methods][vertex][sigma_c]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_sigma_toy skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace sig_toy;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto& comm = mpi_context->comm;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-10);
    model_t mdl;
    auto G = mdl.G_tau(ft);
    const long nt = ft.nt_f(), nw_f = ft.nw_f();
    const cplx I(0.0, 1.0);

    SECTION("fused_vs_batched") {
      // both q->0 policies are pinned (notes/q0_head_treatment.md section 3):
      //   skip = true  : v1_skip -- the toy's Gamma (q-index 0) dropped on both rungs
      //   skip = false : v2 -- all q included (references updated consistently)
      auto Wt = mdl.Wdyn_tau(ft);
      for (bool skip : {true, false}) {
        nda::array<cplx, 5> Sig(nt, ns, nk, nbnd, nbnd);
        solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, C(), G, mdl.X_skPa, Wt,
                                                        mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                                        /*iq_gamma*/ 0, skip, Sig);
        // reference: batched primitive over orbital tuples, all (k, qx, qy)
        nda::array<cplx, 5> Sig_ref(nt, ns, nk, nbnd, nbnd);
        Sig_ref() = cplx(0.0);
        for (long ik = 0; ik < nk; ++ik)
          for (long iqx = 0; iqx < nk; ++iqx)
            for (long iqy = 0; iqy < nk; ++iqy) {
              if (skip and (iqx == 0 or iqy == 0)) continue;
              auto S = combo_reference(ft, mdl, G, ik, iqx, iqy, true, true);
              for (long it = 0; it < nt; ++it)
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b)
                    Sig_ref(it, 0, ik, a, b) += S(it, a, b) / double(nk * nk);
            }
        double num = 0, den = 0;
        for (long it = 0; it < nt; ++it)
          for (long ik = 0; ik < nk; ++ik)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                num = std::max(num, std::abs(Sig(it, 0, ik, a, b) - Sig_ref(it, 0, ik, a, b)));
                den = std::max(den, std::abs(Sig_ref(it, 0, ik, a, b)));
              }
        app_log(1, "fused_vs_batched (skip_rung_gamma = {}): max|fused - batched| = {}, "
                   "max|batched| = {}, rel = {}", skip, num, den, num / den);
        REQUIRE(den > 1e-10);
        REQUIRE(num < 1e-8 * den);
      }
    }

    SECTION("dense_matsubara_arbiter") {
      // single combo, dynamic-only W. Reference = batched primitive (tied to the fused
      // kernel at machine precision by fused_vs_batched); arbiter = dense truncated
      // double Matsubara sum of the +1-prefactor orbital spec with analytic pole
      // evaluations of all five factors.
      const long ik = 1, iqx = 1, iqy = 2;
      const long ikmqx = mdl.kmq(iqx, ik);
      const long ikpqy = mdl.kmq(mdl.qmin(iqy), ik);
      const long ikmqxpqy = mdl.kmq(mdl.qmin(iqy), ikmqx);

      auto S_ref_t = combo_reference(ft, mdl, G, ik, iqx, iqy, true, false);
      nda::array<cplx, 3> S_ref_w(nw_f, nbnd, nbnd);
      ft.tau_to_w(S_ref_t, S_ref_w, imag_axes_ft::fermion);

      auto MX = orb_Wx(mdl, nda::array<cplx, 2>(mdl.M_qPQ(iqx, s_all, s_all)),
                       ik, ikmqx, ikpqy, ikmqxpqy);
      auto MY = orb_Wy(mdl, nda::array<cplx, 2>(mdl.M_qPQ(iqy, s_all, s_all)),
                       ik, ikmqx, ikpqy, ikmqxpqy);

      auto wnf = ft.wn_mesh_f();
      std::vector<long> nodes = {0, nw_f / 2, nw_f - 1};

      auto dense_S = [&](long n_node, long M) {
        cplx z = ft.omega(wnf(n_node));
        nda::array<cplx, 2> acc(nbnd, nbnd);
        acc() = cplx(0.0);
        nda::array<cplx, 1> T1(ncw * ncw * ncw), T2(ncw * ncw * ncw), E(ncw * ncw * ncw);
        auto i3l = [](long i, long j, long k) { return (i * ncw + j) * ncw + k; };
        for (long mx = -M; mx <= M; ++mx) {
          cplx ivx = I * (2.0 * double(mx) * M_PI / beta);
          double sx = model_t::s_of(std::abs(2.0 * double(mx) * M_PI / beta));
          auto Dm = mdl.G_z(ikmqx, z - ivx);
          for (long my = -M; my <= M; ++my) {
            cplx ivy = I * (2.0 * double(my) * M_PI / beta);
            double sy = model_t::s_of(std::abs(2.0 * double(my) * M_PI / beta));
            auto Bm = mdl.G_z(ikpqy, z + ivy);
            auto Cm = mdl.G_z(ikmqxpqy, z - ivx + ivy);
            const double sxy = sx * sy;
            for (long a = 0; a < nbnd; ++a) {
              // T1(p3,p4,p4') = sum_{p2'} B(p3,p2') MY(a,p2';p4,p4')
              for (long j3 = 0; j3 < ncw; ++j3)
                for (long j4 = 0; j4 < ncw; ++j4)
                  for (long j4p = 0; j4p < ncw; ++j4p) {
                    cplx t(0.0);
                    for (long j2p = 0; j2p < ncw; ++j2p)
                      t += Bm(C0 + j3, C0 + j2p) * MY(a, j2p, j4, j4p);
                    T1(i3l(j3, j4, j4p)) = t;
                  }
              // T2(p3,p3',p4') = sum_{p4} T1(p3,p4,p4') C(p4,p3')
              for (long j3 = 0; j3 < ncw; ++j3)
                for (long j3p = 0; j3p < ncw; ++j3p)
                  for (long j4p = 0; j4p < ncw; ++j4p) {
                    cplx t(0.0);
                    for (long j4 = 0; j4 < ncw; ++j4)
                      t += T1(i3l(j3, j4, j4p)) * Cm(C0 + j4, C0 + j3p);
                    T2(i3l(j3, j3p, j4p)) = t;
                  }
              for (long b = 0; b < nbnd; ++b) {
                // E(p3,p3',p4') = sum_{p1} MX(b,p1;p3,p3') D(p1,p4')
                cplx t(0.0);
                for (long j3 = 0; j3 < ncw; ++j3)
                  for (long j3p = 0; j3p < ncw; ++j3p)
                    for (long j4p = 0; j4p < ncw; ++j4p) {
                      cplx e(0.0);
                      for (long j1 = 0; j1 < ncw; ++j1)
                        e += MX(b, j1, j3, j3p) * Dm(C0 + j1, C0 + j4p);
                      t += T2(i3l(j3, j3p, j4p)) * e;
                    }
                acc(a, b) += sxy * t;
              }
            }
          }
        }
        acc() *= 1.0 / (beta * beta);
        return acc;
      };

      double max_rel = 0;
      for (long n : nodes) {
        auto d1 = dense_S(n, 256);
        auto d2 = dense_S(n, 512);
        double num = 0, den = 0, conv = 0;
        for (long a = 0; a < nbnd; ++a)
          for (long b = 0; b < nbnd; ++b) {
            num = std::max(num, std::abs(d2(a, b) - S_ref_w(n, a, b)));
            den = std::max(den, std::abs(S_ref_w(n, a, b)));
            conv = std::max(conv, std::abs(d2(a, b) - d1(a, b)));
          }
        app_log(1, "dense_arbiter node {} (n = {}): max|dense - ref| = {}, max|ref| = {}, "
                   "|dense(M=512) - dense(M=256)| = {}", n, wnf(n), num, den, conv);
        REQUIRE(den > 1e-12);
        max_rel = std::max(max_rel, num / den);
      }
      app_log(1, "dense_arbiter: max relative deviation = {}", max_rel);
      REQUIRE(max_rel < 1e-4);
    }

    SECTION("instantaneous_reduction") {
      // W_dyn = 0, Z only: independent bare second-order-exchange tau-contraction
      //   Sigma(tau) = +(1/Nk^2) sum [ -cx_b cy_a B(tau) C(beta-tau) D(tau) ]
      // with C(beta-tau) by INDEX reflection on the symmetric tau mesh.
      nda::array<cplx, 4> Wt0(nk, nt, Np, Np);
      Wt0() = cplx(0.0);
      const bool skip = false;   // v2 policy: all q included (v1_skip is pinned above)
      nda::array<cplx, 5> Sig(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2_nosym(ft, comm, C(), G, mdl.X_skPa, Wt0,
                                                      mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                                      /*iq_gamma*/ 0, skip, Sig);
      nda::array<cplx, 5> Sig_ref(nt, ns, nk, nbnd, nbnd);
      Sig_ref() = cplx(0.0);
      for (long ik = 0; ik < nk; ++ik)
        for (long iqx = 0; iqx < nk; ++iqx)
          for (long iqy = 0; iqy < nk; ++iqy) {
            if (skip and (iqx == 0 or iqy == 0)) continue;
            long ikmqx = mdl.kmq(iqx, ik);
            long ikpqy = mdl.kmq(mdl.qmin(iqy), ik);
            long ikmqxpqy = mdl.kmq(mdl.qmin(iqy), ikmqx);
            auto ZX = orb_Wx(mdl, nda::array<cplx, 2>(mdl.Z_qPQ(iqx, s_all, s_all)),
                             ik, ikmqx, ikpqy, ikmqxpqy);
            auto ZY = orb_Wy(mdl, nda::array<cplx, 2>(mdl.Z_qPQ(iqy, s_all, s_all)),
                             ik, ikmqx, ikpqy, ikmqxpqy);
            for (long it = 0; it < nt; ++it) {
              long itr = nt - it - 1;   // beta - tau by index reflection
              for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b) {
                  cplx acc(0.0);
                  for (long j1 = 0; j1 < ncw; ++j1)
                    for (long j2p = 0; j2p < ncw; ++j2p)
                      for (long j3 = 0; j3 < ncw; ++j3)
                        for (long j3p = 0; j3p < ncw; ++j3p)
                          for (long j4 = 0; j4 < ncw; ++j4)
                            for (long j4p = 0; j4p < ncw; ++j4p)
                              acc += ZX(b, j1, j3, j3p) * ZY(a, j2p, j4, j4p) *
                                     G(it, 0, ikpqy, C0 + j3, C0 + j2p) *
                                     G(itr, 0, ikmqxpqy, C0 + j4, C0 + j3p) *
                                     G(it, 0, ikmqx, C0 + j1, C0 + j4p);
                  Sig_ref(it, 0, ik, a, b) += -acc / double(nk * nk);
                }
            }
          }
      double num = 0, den = 0;
      for (long it = 0; it < nt; ++it)
        for (long ik = 0; ik < nk; ++ik)
          for (long a = 0; a < nbnd; ++a)
            for (long b = 0; b < nbnd; ++b) {
              num = std::max(num, std::abs(Sig(it, 0, ik, a, b) - Sig_ref(it, 0, ik, a, b)));
              den = std::max(den, std::abs(Sig_ref(it, 0, ik, a, b)));
            }
      app_log(1, "instantaneous_reduction: max|kernel - tau-ref| = {}, max|ref| = {}, rel = {}",
              num, den, num / den);
      REQUIRE(den > 1e-10);
      REQUIRE(num < 1e-7 * den);
    }
#endif  // ENABLE_DLR
  }

  TEST_CASE("vertex_sigma_lih_smoke", "[methods][vertex][sigma_c][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_sigma_lih_smoke skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax = 6.0: the vertex kernels' [A-comp] intermediates need ~3x headroom over the
    // LiH spectral range (~1.2) once the Gamma cell of the dynamic rung is included
    // (v2 q->0 policy) -- the pi-design section 4b requirement, applied uniformly.
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_sigma_smoke";

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

    // C straddles the LiH gap: HOMO = 1, LUMO = 2 (4 electrons, 16 bands)
    solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd());
    scr_eri.set_vertex(&vtx);
    gw.set_vertex(&vtx);

    // one full scGW iteration with the vertex on (integration path: Sigma^C is
    // accumulated inside gw.evaluate on top of Sigma^GW)
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                   1, false, 1e-9, true);
    mpi_context->comm.barrier();
    app_log(1, "vertex_sigma_lih_smoke: e_hf = {}, e_corr = {}", e_hf, e_corr);
    REQUIRE(std::isfinite(e_hf));
    REQUIRE(std::isfinite(e_corr));

    // isolate Sigma^C on the resulting state: rebuild W, then for each q->0 policy
    // (notes/q0_head_treatment.md section 3) snapshot Sigma, add Sigma^C, and inspect
    // the difference -- the v1-skip vs v2 comparison table. The vertex is DETACHED from
    // scr_eri for this rebuild so the isolated checks run against a pure-RPA screened W
    // -- they probe MY Sigma^C kernel only, independent of the Pi^C kernel's state
    // (the in-loop path above already exercised the combined both-cuts flow).
    scr_eri.set_vertex(nullptr);
    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    REQUIRE(mb_state.eps_inv_head.has_value());   // needed by the gygi head insertion

    std::vector<std::string> policies = {"v1_skip", "ignore_g0", "gygi"};
    std::vector<nda::array<cplx, 5>> SigC;
    for (auto const& pol : policies) {
      vtx.set_div_treatment(pol);
      nda::array<cplx, 5> Sig_before(mb_state.sSigma_tskij.value().local());
      vtx.eval_Sigma_C(mb_state, thc);
      auto Sig_after = mb_state.sSigma_tskij.value().local();

      auto [nts, nss, nks, nb1, nb2] = Sig_after.shape();
      nda::array<cplx, 5> dSig(nts, nss, nks, nb1, nb2);
      double scale = 0, d_herm = 0;
      long n_bad = 0;
      for (long it = 0; it < nts; ++it)
        for (long is = 0; is < nss; ++is)
          for (long ik = 0; ik < nks; ++ik)
            for (long a = 0; a < nb1; ++a)
              for (long b = 0; b < nb2; ++b) {
                cplx sc = Sig_after(it, is, ik, a, b) - Sig_before(it, is, ik, a, b);
                cplx sct = Sig_after(it, is, ik, b, a) - Sig_before(it, is, ik, b, a);
                dSig(it, is, ik, a, b) = sc;
                if (not std::isfinite(std::abs(sc))) ++n_bad;
                scale = std::max(scale, std::abs(sc));
                d_herm = std::max(d_herm, std::abs(sc - std::conj(sct)));
              }
      app_log(1, "vertex_sigma_lih_smoke [{}]: Sigma^C scale = {}, hermiticity deviation "
                 "|S_ab - conj(S_ba)| = {} (hermitize() is downstream)", pol, scale, d_herm);
      REQUIRE(n_bad == 0);
      REQUIRE(scale > 0.0);
      REQUIRE(scale < 1e3);          // bounded: no q->0 blow-up under any policy
      REQUIRE(d_herm < 0.5 * scale);
      SigC.emplace_back(std::move(dSig));
    }
    // policy deltas for the comparison table (finite-size-correction sized, not O(scale))
    {
      auto max_abs_diff = [](auto const& A, auto const& B) {
        double d = 0;
        for (long i = 0; i < A.size(); ++i) d = std::max(d, std::abs(A.data()[i] - B.data()[i]));
        return d;
      };
      double d_v2_v1 = max_abs_diff(SigC[1], SigC[0]);
      double d_gy_v2 = max_abs_diff(SigC[2], SigC[1]);
      app_log(1, "vertex_sigma_lih_smoke: policy deltas: max|Sigma^C(ignore_g0) - "
                 "Sigma^C(v1_skip)| = {},", d_v2_v1);
      app_log(1, "                        max|Sigma^C(gygi) - Sigma^C(ignore_g0)| = {}", d_gy_v2);
      REQUIRE(d_v2_v1 > 0.0);   // the Gamma body term is really included
      REQUIRE(d_gy_v2 > 0.0);   // the head insertion really acts
    }

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif  // ENABLE_DLR
  }

} // namespace bdft_tests
