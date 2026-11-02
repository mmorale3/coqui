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
 * The LINEARIZED rung (pol_vertex_rung = "linearized", dynbse.hpp lin_*) on an exact rational toy: nk = 2, nc = 2, G from
 * exact poles (toy_poles.hpp), the pair-space rung W(q'; inu') = Z(q') + sum_p M_p(q') [1/(inu' - e_p) - 1/(inu' + e_p)],
 * and the weights c_j of the LQSGW fit's zeroth-order row (n_fit = 6, symmetric window, exact order 11). Checks:
 *   (0) the separable system M = 1 - A from lin_system equals K_s Cb + sum_j c_j [kd(L0 x)](i w_j) built from kd_apply's
 *       output families evaluated at the fit nodes (the weights rho / kappa / Lam against the operator they replace);
 *   (a) static limit: with W(i nu) = W(0) (no dynamic part) the linearized vertex is the static-rung ladder,
 *       Cb (1 - K0 Cb)^-1 D, to round-off; sum_j c_j = 1;
 *   (d) the brute-force reference: the explicit kernel K(i w, i w') = sum_j c_j W(i w_j - i w'), its frequency sums taken in
 *       closed form (the exact one-rung convolution of the dynbse oracle), solved densely -- the DLR class;
 *   (b) Ward: at q = 0 with a quasiparticle G (a resolvent, gapped) and the identity pair leg u, the ladder head
 *       u^dag (Gsum - Cb u) of the linearized vertex vanishes at every nu (round-off), while the dynamic-rung vertex's
 *       (Gamma_1) does not.
 */

#undef NDEBUG

#include <cmath>
#include <complex>
#include <optional>
#include <random>
#include <vector>

#include "catch2/catch.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/ward_legs.hpp"
#include "methods/vertex/dynbse.hpp"
#include "methods/SCF/qp/linearized_qp.hpp"
#include "toy_poles.hpp"

namespace bdft_tests {

  namespace lin_detail {

    namespace wl = methods::solvers::ward_legs;
    namespace db = methods::solvers::dynbse;
    namespace lqp = methods::lqp;

    // closed-form (1/beta) sum_m [1/(iw_n - iw_m - e) - 1/(iw_n - iw_m + e)] / (iw_m - ej)
    inline cplx conv_pole(double beta, cplx iwn, double ej, double e) {
      const double fj = wl::nF(beta, ej), nb = db::nB(beta, e);
      return -(1.0 - fj + nb) / (iwn - ej - e) - (fj + nb) / (iwn - ej + e);
    }

    /** a quasiparticle toy: G(z) = (z - h)^-1 with the eigenvalues e and a random eigenbasis */
    inline toy_k make_qp_toy(std::vector<double> const &e, unsigned seed) {
      const long nb = long(e.size());
      std::mt19937 gen(seed);
      std::normal_distribution<double> nd(0.0, 1.0);
      nda::matrix<cplx> A(nb, nb);
      for (auto &v : A) v = cplx(nd(gen), nd(gen));
      nda::matrix<cplx> H(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) H(i, j) = 0.5 * (A(i, j) + std::conj(A(j, i)));
      auto [ev, U] = nda::linalg::eigenelements(H);
      toy_k t;
      t.nb = nb; t.np = 0; t.ng = nb;
      t.h = nda::array<cplx, 2>(nb, nb);
      t.epsS = nda::array<double, 1>(0);
      t.R = nda::array<cplx, 3>(0, nb, nb);
      t.lam = nda::array<double, 1>(nb);
      t.g = nda::array<cplx, 3>(nb, nb, nb);
      t.h() = cplx(0.0);
      for (long j = 0; j < nb; ++j) {
        t.lam(j) = e[size_t(j)];
        for (long a = 0; a < nb; ++a)
          for (long c = 0; c < nb; ++c) {
            t.g(j, a, c) = U(a, j) * std::conj(U(c, j));
            t.h(a, c) += e[size_t(j)] * U(a, j) * std::conj(U(c, j));
          }
      }
      return t;
    }

    struct toy {
      long nk = 2, nc = 2, nc2 = 4, npole = 2;
      double beta = 10.0;
      std::vector<toy_k> t;
      nda::array<long, 2> kmk;
      std::vector<long> kpq;
      nda::array<double, 1> epsW;
      nda::array<cplx, 3> Z;      // (nq, nc2, nc2)
      nda::array<cplx, 4> M;      // (npole, nq, nc2, nc2)
      long D() const { return nk * nc2; }
    };

    inline toy make_toy_lin(std::vector<toy_k> tk, std::vector<long> kpq, double beta, double zscale, double mscale) {
      toy T;
      T.beta = beta;
      T.t = std::move(tk);
      T.nc = T.t[0].nb; T.nc2 = T.nc * T.nc;
      T.kmk = nda::array<long, 2>(T.nk, T.nk);
      for (long k = 0; k < T.nk; ++k)
        for (long kp = 0; kp < T.nk; ++kp) T.kmk(k, kp) = (k - kp + T.nk) % T.nk;
      T.kpq = std::move(kpq);
      T.epsW = nda::array<double, 1>(T.npole);
      T.epsW(0) = 0.8; T.epsW(1) = 1.6;
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
      return T;
    }

    /** the LQSGW fit weights: n_fit = 6, symmetric window, exact order (lqp::make_fit_operator's design matrix) */
    inline void fit_weights(std::vector<long> &n, std::vector<cplx> &c) {
      auto nw = lqp::fit_window(6, true);
      n.assign(nw.begin(), nw.end());
      lqp::fit_operator_t op;
      const long nn = long(n.size());
      long nmax = 0;
      for (long v : n) nmax = std::max(nmax, std::abs(v));
      op.fit_order = int(nn) - 1;
      op.M = nda::array<cplx, 2>(nn, nn);
      for (long j = 0; j < nn; ++j) {
        const cplx ix(0.0, double(n[size_t(j)]) / double(nmax));
        cplx x = 1.0;
        for (long q = 0; q < nn; ++q) { op.M(j, q) = x; x *= ix; }
      }
      c = lqp::zeroth_order_row(op);
    }

    /** the solver-side objects of the toy on the exact node set (the union basis of the vertex grid and the toy's poles) */
    struct setup {
      db::freq_basis b;
      db::pair_poles P;
      db::pair_rung R;
      nda::array<cplx, 2> Ks, Kd0;    // dense (D x D): rows (k, pair) out, columns (k', pair) in
      nda::array<cplx, 3> Kslot;      // (nt, D, D): the dynamic part per tau node
    };

    inline setup make_setup(toy const &T, imag_axes_ft::IAFT const &ft) {
      setup S;
      S.b = db::build_freq_basis(ft);
      auto set = assemble(T.t);
      const long ng = set.epsG.shape(0), nk = T.nk, nc = T.nc, nc2 = T.nc2, D = T.D();
      nda::array<cplx, 4> gk(ng, nk, nc, nc), gkq(ng, nk, nc, nc);
      for (long k = 0; k < nk; ++k) {
        gk(all_, k, all_, all_) = set.g(0, all_, k, all_, all_);
        gkq(all_, k, all_, all_) = set.g(0, all_, T.kpq[size_t(k)], all_, all_);
      }
      const long np_fit = S.b.np;
      db::extend_freq_basis(S.b, set.epsG);
      S.P = db::make_pair_poles(T.beta, set.epsG, gk, gkq, np_fit);
      const long nt = S.b.nt;
      auto &R = S.R;
      R.nq = nk; R.nt = nt; R.nc2 = nc2; R.kmk = T.kmk;
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
                                          cplx(db::KB(T.beta, S.b.s(i), T.epsW(p)) - db::KB(T.beta, S.b.s(i), -T.epsW(p)));
        }
        R.K0(iq, all_, all_) = T.Z(iq, all_, all_) + R.Wd0(iq, all_, all_);
      }
      S.Ks = nda::array<cplx, 2>(D, D);
      S.Kd0 = nda::array<cplx, 2>(D, D);
      S.Kslot = nda::array<cplx, 3>(nt, D, D);
      for (long k = 0; k < nk; ++k)
        for (long kp = 0; kp < nk; ++kp) {
          const long iq = T.kmk(k, kp);
          for (long p = 0; p < nc2; ++p)
            for (long pp = 0; pp < nc2; ++pp) {
              S.Ks(k * nc2 + p, kp * nc2 + pp) = R.K0(iq, p, pp);
              S.Kd0(k * nc2 + p, kp * nc2 + pp) = R.Wd0(iq, p, pp);
              for (long i = 0; i < nt; ++i) S.Kslot(i, k * nc2 + p, kp * nc2 + pp) = R.Wd_s(iq, i, p, pp);
            }
        }
      return S;
    }

    /** Cb_k (nk, nc2, nc2) = the exact frequency sum of L0 on the frequency-constant unit pairs */
    inline nda::array<cplx, 3> pair_sum(setup const &S, toy const &T, cplx inu, db::shift_tables const *st) {
      const long nk = T.nk, nc = T.nc, nc2 = T.nc2;
      db::tf_vector X(S.b.np, nk, nc, nc2), F(S.b.np, nk, nc, nc2);
      nda::array<cplx, 4> Fsum(nk, nc, nc, nc2);
      for (long k = 0; k < nk; ++k)
        for (long c = 0; c < nc2; ++c) X.cst(k, c / nc, c % nc, c) = cplx(1.0);
      db::l0_apply(S.b, S.P, inu, false, X, F, Fsum, nullptr, st);
      nda::array<cplx, 3> Cb(nk, nc2, nc2);
      for (long k = 0; k < nk; ++k)
        for (long p = 0; p < nc2; ++p)
          for (long c = 0; c < nc2; ++c) Cb(k, p, c) = Fsum(k, p / nc, p % nc, c);
      return Cb;
    }

    /** the linearized vertex Gsum = Cb (1 - M)^-1 Dc through dynbse.hpp lin_* (per-tau rung slots, scale 1) */
    inline nda::array<cplx, 4> lin_gsum(setup const &S, toy const &T, cplx inu, db::shift_tables const *st, nda::array<cplx, 3> const &Cb,
                                        std::vector<long> const &n, std::vector<cplx> const &c, nda::array<cplx, 4> const &Dc,
                                        nda::matrix<cplx> *Mout = nullptr) {
      const long nt = S.b.nt, D = T.D();
      nda::array<cplx, 2> w(nt, nt);
      w() = cplx(0.0);
      for (long i = 0; i < nt; ++i) w(i, i) = cplx(1.0);
      auto kap = db::lin_kappa(S.b, inu, db::lin_rho(S.b, inu, n, c), w);
      auto Lam = db::lin_lambda(S.b, S.P, inu, false, Cb, st, kap);
      auto A = db::lin_system(S.Ks, Cb, S.Kslot, Lam, S.Kd0, cplx(1.0), nullptr);
      if (Mout != nullptr) {
        *Mout = nda::matrix<cplx>(D, D);
        for (long i = 0; i < D; ++i)
          for (long j = 0; j < D; ++j) (*Mout)(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) - A(i, j);
      }
      nda::array<int, 1> ipiv(D);
      REQUIRE(nda::lapack::getrf(A, ipiv) == 0);
      return db::lin_solve(A, ipiv, Cb, Dc);
    }

    /** the exact (closed-form) rung convolution sum_m W(iw_n - iw_m) L0(k', m) V(k') for a frequency-constant V */
    inline nda::array<cplx, 1> rung_full(toy const &T, long k, cplx iwn, cplx inu, nda::array<cplx, 2> const &V, bool dyn) {
      const long nk = T.nk, nc = T.nc, nc2 = T.nc2;
      const double beta = T.beta;
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
            for (long p1p = 0; p1p < nc; ++p1p)
              for (long p3 = 0; p3 < nc; ++p3) {
                cplx s(0.0);
                for (long aa = 0; aa < nc; ++aa)
                  for (long bb = 0; bb < nc; ++bb) s += tk.g(j, aa, p1p) * V(kp, aa * nc + bb) * tq.g(l, p3, bb);
                a(p1p * nc + p3) = s / Djl;
              }
            const double fj = wl::nF(beta, ej), fl = wl::nF(beta, el);
            nda::array<cplx, 2> Keff(nc2, nc2);
            Keff() = T.Z(iq, all_, all_) * cplx(fj - fl);
            for (long p = 0; p < T.npole; ++p) {
              if (dyn) {
                const cplx cu = conv_pole(beta, iwn, ej, T.epsW(p));
                const cplx cs = conv_pole(beta, iwn + inu, el, T.epsW(p));
                Keff += T.M(p, iq, all_, all_) * (cu - cs);
              } else {
                Keff += T.M(p, iq, all_, all_) * cplx(-2.0 / T.epsW(p) * (fj - fl));
              }
            }
            for (long p = 0; p < nc2; ++p) {
              cplx s(0.0);
              for (long pp = 0; pp < nc2; ++pp) s += Keff(p, pp) * a(pp);
              out(p) += s;
            }
          }
      }
      return out;
    }

    inline double rel(nda::MemoryArrayOfRank<4> auto const &A, nda::MemoryArrayOfRank<4> auto const &B) {
      double d = 0.0, s = 0.0;
      for (long i = 0; i < A.size(); ++i) {
        d = std::max(d, std::abs(A.data()[i] - B.data()[i]));
        s = std::max(s, std::abs(B.data()[i]));
      }
      return (s > 0.0) ? d / s : d;
    }

  } // namespace lin_detail

  TEST_CASE("vertex_lin_toy", "[methods][vertex][lin]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_lin_toy skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace lin_detail;
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;
    std::vector<long> nfit;
    std::vector<cplx> cfit;
    fit_weights(nfit, cfit);
    {
      cplx sum(0.0);
      for (auto const &v : cfit) sum += v;
      app_log(1, "vertex_lin (a): {} fit nodes, sum_j c_j - 1 = {:.3e}", nfit.size(), std::abs(sum - 1.0));
      REQUIRE(std::abs(sum - 1.0) < 1e-12);
    }
    const double beta = 10.0;
    imag_axes_ft::IAFT ft(beta, 8.0, imag_axes_ft::dlr_basis, "high");
    auto T = make_toy_lin({make_toy(2, {-1.5, 1.5}, {0.3, 0.3}, 41u), make_toy(2, {-1.5, 1.5}, {0.3, 0.3}, 43u)}, {1, 0}, beta, 0.10, 0.02);
    auto S = make_setup(T, ft);
    const long nk = T.nk, nc = T.nc, nc2 = T.nc2, D = T.D();
    // the external legs: one column per (k, pair)
    nda::array<cplx, 4> Dc(nk, nc, nc, D);
    Dc() = cplx(0.0);
    for (long k = 0; k < nk; ++k)
      for (long p = 0; p < nc2; ++p) Dc(k, p / nc, p % nc, k * nc2 + p) = cplx(1.0);

    for (cplx inu : {cplx(0.0), I_ * cplx(2.0 * M_PI / beta), I_ * cplx(2.0 * M_PI * 3.0 / beta)}) {
      const bool nu0 = (inu == cplx(0.0));
      std::optional<db::shift_tables> st;
      if (not nu0) st.emplace(db::build_shift_tables(S.b, inu, 1e-11, 0));
      db::shift_tables const *stp = st ? std::addressof(st.value()) : nullptr;
      auto Cb = pair_sum(S, T, inu, stp);
      nda::matrix<cplx> Mlin;
      auto Gl = lin_gsum(S, T, inu, stp, Cb, nfit, cfit, Dc, &Mlin);

      // (0) M against the kd operator evaluated at the fit nodes
      {
        db::tf_vector X(S.b.np, nk, nc, D), F(S.b.np, nk, nc, D), y(S.b.np, nk, nc, D);
        nda::array<cplx, 4> Fsum(nk, nc, nc, D);
        X.cst() = Dc;
        db::l0_apply(S.b, S.P, inu, false, X, F, Fsum, &Cb, stp, false, true);
        (void)db::kd_apply(S.b, S.R, F, Fsum, y, inu);
        nda::matrix<cplx> Mref(D, D);
        for (long r = 0; r < D; ++r)
          for (long k = 0; k < nk; ++k)
            for (long p = 0; p < nc2; ++p) {
              cplx v(0.0);
              for (size_t j = 0; j < nfit.size(); ++j) {
                const cplx z(0.0, double(nfit[j]) * M_PI / beta);
                cplx yz = y.cst(k, p / nc, p % nc, r);
                for (long a = 0; a < S.b.np_fit; ++a) {
                  const cplx u = 1.0 / (z - S.b.eps(a));
                  yz += y.fam(0, a, k, p / nc, p % nc, r) * u + (nu0 ? cplx(0.0) : y.fam(1, a, k, p / nc, p % nc, r) * u / (z + inu - S.b.eps(a)));
                }
                v += cfit[j] * yz;
              }
              // + K_s Cb
              for (long kk = 0; kk < nk; ++kk)
                for (long pp = 0; pp < nc2; ++pp) {
                  cplx cb(0.0);
                  if (kk == r / nc2) cb = Cb(kk, pp, r % nc2);
                  v += S.Ks(k * nc2 + p, kk * nc2 + pp) * cb;
                }
              Mref(k * nc2 + p, r) = v;
            }
        double d = 0.0, sc = 0.0;
        for (long i = 0; i < D; ++i)
          for (long j = 0; j < D; ++j) { d = std::max(d, std::abs(Mlin(i, j) - Mref(i, j))); sc = std::max(sc, std::abs(Mref(i, j))); }
        app_log(1, "vertex_lin (0) inu = {:.3f}i: |M_lin - M_kd| = {:.3e} (|M| {:.3e})", inu.imag(), d, sc);
        // the same algebra summed in another order; the alternating fit weights (|c_j| up to O(10)) and the refit map cost a
        // few digits (measured ~1e-10)
        REQUIRE(d < 1e-9 * sc);
      }

      // (d) the brute-force reference: the explicit kernel's closed-form sums at the fit nodes, a dense solve
      {
        nda::matrix<cplx> Mex(D, D);
        nda::array<cplx, 2> V(nk, nc2);
        for (long col = 0; col < D; ++col) {
          V() = cplx(0.0);
          V(col / nc2, col % nc2) = cplx(1.0);
          for (long k = 0; k < nk; ++k) {
            nda::array<cplx, 1> acc(nc2);
            acc() = cplx(0.0);
            for (size_t j = 0; j < nfit.size(); ++j) acc += cfit[j] * rung_full(T, k, cplx(0.0, double(nfit[j]) * M_PI / beta), inu, V, true);
            for (long p = 0; p < nc2; ++p) Mex(k * nc2 + p, col) = acc(p);
          }
        }
        nda::matrix<cplx> A(D, D);
        for (long i = 0; i < D; ++i)
          for (long j = 0; j < D; ++j) A(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) - Mex(i, j);
        nda::inverse_in_place(A);
        // Gsum = Cb_exact (1 - M)^-1 D; Cb from the toy's exact two-pole T-sums
        nda::array<cplx, 4> Gex(nk, nc, nc, D);
        Gex() = cplx(0.0);
        for (long k = 0; k < nk; ++k) {
          auto const &tk = T.t[size_t(k)];
          auto const &tq = T.t[size_t(T.kpq[size_t(k)])];
          nda::array<cplx, 2> Cbe(nc2, nc2);
          Cbe() = cplx(0.0);
          for (long j = 0; j < tk.ng; ++j)
            for (long l = 0; l < tq.ng; ++l) {
              const cplx Tjl = wl::two_pole(wl::fermi_all(beta, tk.lam(j)), tk.lam(j), wl::fermi_all(beta, tq.lam(l)), tq.lam(l), inu,
                                            false).T;
              for (long p1p = 0; p1p < nc; ++p1p)
                for (long p3 = 0; p3 < nc; ++p3)
                  for (long a = 0; a < nc; ++a)
                    for (long bb = 0; bb < nc; ++bb) Cbe(p1p * nc + p3, a * nc + bb) += Tjl * tk.g(j, a, p1p) * tq.g(l, p3, bb);
            }
          for (long p = 0; p < nc2; ++p)
            for (long r = 0; r < D; ++r) {
              cplx v(0.0);
              for (long pp = 0; pp < nc2; ++pp) v += Cbe(p, pp) * A(k * nc2 + pp, r);
              Gex(k, p / nc, p % nc, r) = v;
            }
        }
        const double e = rel(Gl, Gex);
        {   // the anti-Hermitian part of the ladder block D^dag (Gsum - Cb D) (the one-sided kernel breaks the in/out symmetry)
          nda::array<cplx, 4> Gm(Gl);
          for (long k = 0; k < nk; ++k)
            for (long p = 0; p < nc2; ++p)
              for (long r = 0; r < D; ++r)
                if (r / nc2 == k) Gm(k, p / nc, p % nc, r) -= Cb(k, p, r % nc2);
          auto Pl = db::collapse(Dc, Gm);
          double an = 0.0, nn = 0.0;
          for (long i = 0; i < D; ++i)
            for (long j = 0; j < D; ++j) { an += std::norm(Pl(i, j) - std::conj(Pl(j, i))); nn += std::norm(Pl(i, j)); }
          app_log(1, "vertex_lin (d) inu = {:.3f}i: |P - P^dag|_F / |P|_F of the ladder block = {:.3e}", inu.imag(), std::sqrt(an / nn));
        }
        app_log(1, "vertex_lin (d) inu = {:.3f}i: separable solve vs the explicit-kernel reference: max rel {:.3e}", inu.imag(), e);
        REQUIRE(e < 1e-7);
      }

      // (a) the static limit: no dynamic part -> the static-rung ladder Cb (1 - K0 Cb)^-1 D
      {
        auto T0 = T;
        T0.M() = cplx(0.0);
        auto S0 = make_setup(T0, ft);
        auto Cb0 = pair_sum(S0, T0, inu, stp);
        auto G0 = lin_gsum(S0, T0, inu, stp, Cb0, nfit, cfit, Dc);
        nda::matrix<cplx> A(D, D);
        for (long i = 0; i < D; ++i)
          for (long j = 0; j < D; ++j) {
            cplx v((i == j) ? 1.0 : 0.0);
            const long kj = j / nc2;
            for (long pp = 0; pp < nc2; ++pp) v -= S0.Ks(i, kj * nc2 + pp) * Cb0(kj, pp, j % nc2);
            A(i, j) = v;
          }
        nda::inverse_in_place(A);
        nda::array<cplx, 4> Gs(nk, nc, nc, D);
        for (long k = 0; k < nk; ++k)
          for (long p = 0; p < nc2; ++p)
            for (long r = 0; r < D; ++r) {
              cplx v(0.0);
              for (long pp = 0; pp < nc2; ++pp) v += Cb0(k, p, pp) * A(k * nc2 + pp, r);
              Gs(k, p / nc, p % nc, r) = v;
            }
        const double e = rel(G0, Gs);
        app_log(1, "vertex_lin (a) inu = {:.3f}i: static limit vs the static-rung ladder: max rel {:.3e}", inu.imag(), e);
        REQUIRE(e < 1e-12);
      }
    }

    // (b) Ward at q = 0: a quasiparticle G (gapped resolvent), the identity pair leg u
    {
      const double beta_w = 40.0;
      imag_axes_ft::IAFT ftw(beta_w, 8.0, imag_axes_ft::dlr_basis, "high");
      auto Tw = make_toy_lin({make_qp_toy({-1.1, 0.9}, 7u), make_qp_toy({-0.8, 1.2}, 11u)}, {0, 1}, beta_w, 0.10, 0.05);
      auto Sw = make_setup(Tw, ftw);
      nda::array<cplx, 4> u(nk, nc, nc, 1);
      u() = cplx(0.0);
      for (long k = 0; k < nk; ++k)
        for (long a = 0; a < nc; ++a) u(k, a, a, 0) = cplx(1.0);
      for (cplx inu : {cplx(0.0), I_ * cplx(2.0 * M_PI / beta_w), I_ * cplx(2.0 * M_PI * 4.0 / beta_w)}) {
        const bool nu0 = (inu == cplx(0.0));
        std::optional<db::shift_tables> st;
        if (not nu0) st.emplace(db::build_shift_tables(Sw.b, inu, 1e-11, 0));
        db::shift_tables const *stp = st ? std::addressof(st.value()) : nullptr;
        auto Cb = pair_sum(Sw, Tw, inu, stp);
        auto Gl = lin_gsum(Sw, Tw, inu, stp, Cb, nfit, cfit, u);
        // the bubble of the leg (Cb u) and the ladder head u^dag (Gsum - Cb u)
        nda::array<cplx, 4> Cu(nk, nc, nc, 1);
        for (long k = 0; k < nk; ++k)
          for (long p = 0; p < nc2; ++p) {
            cplx v(0.0);
            for (long pp = 0; pp < nc2; ++pp) v += Cb(k, p, pp) * u(k, pp / nc, pp % nc, 0);
            Cu(k, p / nc, p % nc, 0) = v;
          }
        auto head = [&](nda::array<cplx, 4> const &G) {
          nda::array<cplx, 4> Gm(G);
          Gm -= Cu;
          return db::collapse(u, Gm)(0, 0);
        };
        const cplx h_lin = head(Gl);
        // the dynamic-rung vertex (Gamma_1) on the same toy
        auto Sres = db::build_static_resolvent(Sw.b, Sw.P, Sw.R, inu, false);
        auto kd = [&](db::tf_vector const &F, nda::array<cplx, 4> const &Fsum, db::tf_vector &y) {
          return db::kd_apply(Sw.b, Sw.R, F, Fsum, y, inu);
        };
        auto res = db::solve_dyson_gmres(Sw.b, Sw.P, kd, Sres, inu, false, u, 1e-10, 30, 12, nullptr, stp, nullptr, 0.0, true);
        const cplx h_dyn = head(res.Gsum1);
        // the scale: the static-rung ladder's correction at a finite-q leg of the same size
        double scale = 0.0;
        for (auto const &v : Gl) scale = std::max(scale, std::abs(v));
        app_log(1, "vertex_lin (b) q = 0, inu = {:.3f}i: ladder head u^dag (Gsum - Cb u): linearized {:.3e}, dynamic Gamma_1 {:.3e} "
                   "(|Gsum| {:.3e})", inu.imag(), std::abs(h_lin), std::abs(h_dyn), scale);
        REQUIRE(std::abs(h_lin) < 1e-12 * std::max(scale, 1.0));
        REQUIRE(std::abs(h_dyn) > 1e-6 * scale);
      }
    }
#endif
  }

} // bdft_tests
