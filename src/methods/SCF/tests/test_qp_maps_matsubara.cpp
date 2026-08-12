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

#undef NDEBUG

#include <vector>
#include <cmath>
#include <algorithm>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "methods/SCF/qp_maps_matsubara.hpp"
#include "methods/SCF/sigma_real_axis.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"

namespace bdft_tests {

  using methods::qp_matsubara::qp_lin_matrix;
  using methods::qp_matsubara::qp_gmatch_block;
  using methods::qp_matsubara::gmatch_opts;

  namespace qpmats_detail {
    // fermionic Matsubara values w_n = (2n+1) pi T, ascending, positive
    inline nda::array<double, 1> wmesh(long nw, double beta) {
      nda::array<double, 1> w(nw);
      for (long n = 0; n < nw; ++n) w(n) = (2.0 * n + 1.0) * M_PI / beta;
      return w;
    }
    inline nda::array<ComplexType, 2> resolvent(nda::array<ComplexType, 2> const &H,
                                                double wn, double mu) {
      const long nb = H.shape(0);
      nda::matrix<ComplexType> K(nb, nb);
      K() = -H;
      for (long i = 0; i < nb; ++i) K(i, i) += ComplexType(mu, wn);
      nda::inverse_in_place(K);
      return nda::array<ComplexType, 2>(K);
    }
    inline double maxdiff(nda::array<ComplexType, 2> const &A,
                          nda::array<ComplexType, 2> const &B) {
      double m = 0.0;
      for (long i = 0; i < A.shape(0); ++i)
        for (long j = 0; j < A.shape(1); ++j) m = std::max(m, std::abs(A(i, j) - B(i, j)));
      return m;
    }
  }

  TEST_CASE("qp_mats_lin_static", "[methods][qpgw][qp_maps]") {
    // Q1-a: a STATIC Hermitian Sigma_c => Z = 1 exactly (the odd part vanishes) and
    // V = Sigma_c to machine precision -- both maps' trivial limit.
    const long nb = 3;
    const double beta = 100.0;
    auto wn = qpmats_detail::wmesh(4, beta);
    nda::array<ComplexType, 2> C(nb, nb);
    C() = ComplexType(0.0);
    C(0, 0) = 0.3; C(1, 1) = -0.2; C(2, 2) = 0.05;
    C(0, 1) = ComplexType(0.04, 0.02); C(1, 0) = std::conj(C(0, 1));
    C(1, 2) = ComplexType(-0.03, 0.01); C(2, 1) = std::conj(C(1, 2));
    nda::array<ComplexType, 3> S(4, nb, nb);
    for (long n = 0; n < 4; ++n) S(n, nda::range::all, nda::range::all) = C;
    long ncl = 0;
    auto V = qp_lin_matrix(S, wn, ncl);
    REQUIRE(ncl == 0);
    REQUIRE(qpmats_detail::maxdiff(V, C) < 1e-14);
  }

  TEST_CASE("qp_mats_lin_single_pole", "[methods][qpgw][qp_maps]") {
    // Q1-b (map i): Sigma(iw) = c^2 / (iw - p), scalar. Analytic:
    //   dSigma/d(iw)|_0 = -c^2/p^2  =>  Z = 1/(1 + c^2/p^2);  Sigma(0) = -c^2/p.
    // The first-node difference quotient carries an O(w0^2) error (the pi*T
    // resolution floor documented in the header) -- gate at that class.
    const double beta = 2000.0, c2 = 0.09, p = 0.8;
    auto wn = qpmats_detail::wmesh(3, beta);
    nda::array<ComplexType, 3> S(3, 1, 1);
    for (long n = 0; n < 3; ++n) S(n, 0, 0) = c2 / (ComplexType(0.0, wn(n)) - p);
    long ncl = 0;
    auto V = qp_lin_matrix(S, wn, ncl);
    const double Z = 1.0 / (1.0 + c2 / (p * p));
    const double Vref = Z * (-c2 / p);
    REQUIRE(ncl == 0);
    REQUIRE(std::abs(V(0, 0).real() - Vref) < 5e-4);   // O((pi/beta)^2) class
    REQUIRE(std::abs(V(0, 0).imag()) < 1e-14);
  }

  TEST_CASE("qp_mats_gmatch_exact_recovery", "[methods][qpgw][qp_maps]") {
    // Q1-e: the target G IS a static resolvent => map (ii) must recover H_true from
    // a perturbed start, residual driven to the numerical floor.
    const long nb = 3, nw = 12;
    const double beta = 200.0, mu = 0.1;
    auto wn = qpmats_detail::wmesh(nw, beta);
    nda::array<ComplexType, 2> Htrue(nb, nb);
    Htrue() = ComplexType(0.0);
    Htrue(0, 0) = -0.5; Htrue(1, 1) = 0.2; Htrue(2, 2) = 0.9;
    Htrue(0, 1) = ComplexType(0.1, -0.05); Htrue(1, 0) = std::conj(Htrue(0, 1));
    Htrue(0, 2) = ComplexType(-0.02, 0.03); Htrue(2, 0) = std::conj(Htrue(0, 2));
    nda::array<ComplexType, 3> G(nw, nb, nb);
    for (long n = 0; n < nw; ++n)
      G(n, nda::range::all, nda::range::all) = qpmats_detail::resolvent(Htrue, wn(n), mu);
    auto H = Htrue;
    H(0, 0) += 0.15; H(1, 1) -= 0.1;
    H(0, 1) += ComplexType(0.02, 0.02); H(1, 0) = std::conj(H(0, 1));
    auto info = qp_gmatch_block(G, wn, mu, H, {});
    REQUIRE(info.r < 1e-16 * std::max(1.0, info.r0));
    REQUIRE(qpmats_detail::maxdiff(H, Htrue) < 1e-7);
    // hermiticity by construction
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        REQUIRE(std::abs(H(i, j) - std::conj(H(j, i))) < 1e-13);
  }

  TEST_CASE("qp_mats_gmatch_covariance", "[methods][qpgw][qp_maps]") {
    // Q1-c: basis covariance -- rotating (G, H_init) by a fixed unitary rotates the
    // matched H covariantly (map (ii) is basis-invariant by construction).
    const long nb = 2, nw = 10;
    const double beta = 150.0, mu = 0.0;
    auto wn = qpmats_detail::wmesh(nw, beta);
    // a dynamic (non-representable) target: two-pole G per entry
    nda::array<ComplexType, 2> Ha(nb, nb), Hb(nb, nb);
    Ha() = ComplexType(0.0); Hb() = ComplexType(0.0);
    Ha(0, 0) = -0.4; Ha(1, 1) = 0.6; Ha(0, 1) = ComplexType(0.15, 0.0);
    Ha(1, 0) = std::conj(Ha(0, 1));
    Hb(0, 0) = -0.1; Hb(1, 1) = 0.3;
    nda::array<ComplexType, 3> G(nw, nb, nb);
    for (long n = 0; n < nw; ++n) {
      auto Ga = qpmats_detail::resolvent(Ha, wn(n), mu);
      auto Gb = qpmats_detail::resolvent(Hb, wn(n), mu);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          G(n, i, j) = 0.7 * Ga(i, j) + 0.3 * Gb(i, j);
    }
    // the unitary (a rotation by theta)
    const double th = 0.37;
    nda::array<ComplexType, 2> W(nb, nb);
    W(0, 0) = std::cos(th); W(0, 1) = -std::sin(th);
    W(1, 0) = std::sin(th); W(1, 1) = std::cos(th);
    auto rot = [&](nda::array<ComplexType, 2> const &X) {
      // W^dag X W by explicit loops (nb = 2; dodges lazy-overload pitfalls)
      nda::array<ComplexType, 2> T(nb, nb), Y(nb, nb);
      T() = ComplexType(0.0); Y() = ComplexType(0.0);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          for (long l = 0; l < nb; ++l) T(i, j) += std::conj(W(l, i)) * X(l, j);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          for (long l = 0; l < nb; ++l) Y(i, j) += T(i, l) * W(l, j);
      return Y;
    };
    nda::array<ComplexType, 2> H1 = Ha;                // plain frame
    auto info1 = qp_gmatch_block(G, wn, mu, H1, {});
    nda::array<ComplexType, 3> Gr(nw, nb, nb);
    for (long n = 0; n < nw; ++n)
      Gr(n, nda::range::all, nda::range::all) = rot(nda::array<ComplexType, 2>(
          G(n, nda::range::all, nda::range::all)));
    nda::array<ComplexType, 2> H2 = rot(Ha);           // rotated frame
    auto info2 = qp_gmatch_block(Gr, wn, mu, H2, {});
    REQUIRE(info1.iters > 0);
    REQUIRE(info2.iters > 0);
    REQUIRE(qpmats_detail::maxdiff(H2, rot(H1)) < 1e-8);
    REQUIRE(std::abs(info1.r - info2.r) < 1e-12 * std::max(1.0, info1.r));
  }

  TEST_CASE("qp_mats_gmatch_vs_lin_surrogate_class", "[methods][qpgw][qp_maps]") {
    // Q1-b, MEASURED AND REFRAMED (2026-08-12): near iw -> 0 the true G has residue
    // Z != 1 (G ~ Z/(iw - E*), E* = Z(e0 + Sigma(0)) at mu = 0), which a residue-1
    // resolvent CANNOT reproduce; the G-match therefore lands a compromise H that
    // differs from eq 13's sandwich form (e0 + Z Sigma(0)) at O((1-Z)|E - mu|) --
    // a BETA-INDEPENDENT surrogate difference (measured: 1.13e-3 at beta 200 AND
    // 1.16e-3 at beta 800 on this toy), NOT a discretization artifact. The two maps
    // coincide only at leading order near mu; off mu their spread IS the spec
    // section-4 "genuine ~0.1-0.2 eV scale" to be QUANTIFIED, not converged away.
    // Gate: the spread obeys the analytic class bound |1-Z| * |E* - mu|.
    const double beta = 400.0, mu = 0.0, c2 = 0.04, p = 1.1, e0 = -0.3;
    auto wn = qpmats_detail::wmesh(2, beta);
    nda::array<ComplexType, 3> G(2, 1, 1), S(2, 1, 1);
    for (long n = 0; n < 2; ++n) {
      S(n, 0, 0) = c2 / (ComplexType(0.0, wn(n)) - p);
      G(n, 0, 0) = 1.0 / (ComplexType(mu, wn(n)) - e0 - S(n, 0, 0));
    }
    long ncl = 0;
    auto V = qp_lin_matrix(S, wn, ncl);
    const double H_lin = e0 + V(0, 0).real();
    gmatch_opts o;
    o.wpow = 0.0;
    nda::array<ComplexType, 2> H(1, 1);
    H(0, 0) = e0;
    qp_gmatch_block(G, wn, mu, H, o);
    const double d = std::abs(H(0, 0).real() - H_lin);
    const double Z = 1.0 / (1.0 + c2 / (p * p));
    const double Estar = Z * (e0 - c2 / p);
    const double class_bound = std::abs(1.0 - Z) * std::abs(Estar - mu);
    app_log(1, "qp_mats_gmatch_vs_lin: |H_gmatch - H_lin| = {:.3e}; class bound "
               "|1-Z||E*-mu| = {:.3e} (the reported surrogate spread)", d, class_bound);
    REQUIRE(d < class_bound);      // measured 1.13e-3 vs 1.1e-2 on this toy
    REQUIRE(d > 1e-5);             // and it is GENUINE -- not a numerical zero
  }

  // ==========================================================================================
  // Increment QM1 -- ROUTE A real-axis Sigma evaluator (methods/SCF/sigma_real_axis.hpp).
  // Spec: notes/qm1_route_a_spec.md; parent notes/qsgw_matsubara_plan.pdf section 2 / 6.1-6.3.
  // Model units == eV throughout; beta = 1000 (w_0 = pi/1000 ~ 3.1e-3). 1 meV = 1e-3.
  // ==========================================================================================

  namespace sra = methods::sigma_real_axis;

  namespace route_a_detail {

    // The spec section 2.3 four-pole fixture: poles |p_k| >= 6 straddling zero, residues ~2
    // tuned so Z = (1 - Sigma'(eps*))^{-1} lands in 0.75-0.9 (descriptive, not gated).
    inline const std::vector<double> pk{-9.0, -6.0, 6.0, 8.0};
    inline const std::vector<double> rk{2.0, 1.5, 2.5, 1.8};

    // --- generic four-pole model over an arbitrary (poles, residues) pair -------------------
    inline ComplexType sigma_p(std::vector<double> const &pp, std::vector<double> const &rr,
                               ComplexType z) {
      ComplexType s(0.0);
      for (size_t k = 0; k < pp.size(); ++k) s += rr[k] / (z - pp[k]);
      return s;
    }
    inline double sigma_p_re(std::vector<double> const &pp, std::vector<double> const &rr,
                             double e) {
      double s = 0.0;
      for (size_t k = 0; k < pp.size(); ++k) s += rr[k] / (e - pp[k]);
      return s;
    }
    inline double sigma_p_dre(std::vector<double> const &pp, std::vector<double> const &rr,
                              double e) {
      double s = 0.0;
      for (size_t k = 0; k < pp.size(); ++k) s -= rr[k] / ((e - pp[k]) * (e - pp[k]));
      return s;
    }
    /**
     * The EXACT real-axis root of eps = e0 + Sigma(eps). Between the two INNER poles the
     * residual g(e) = e - e0 - Sigma(e) runs from -inf to +inf and is strictly increasing
     * (g' = 1 - Sigma' > 0 for positive residues), so bisection is unconditional; Newton then
     * polishes to machine precision. Real arithmetic, exact model -- no fit anywhere.
     */
    inline double root_p(std::vector<double> const &pp, std::vector<double> const &rr, double e0) {
      double lo = -1e300, hi = 1e300;
      for (double q : pp) {
        if (q < 0.0) lo = std::max(lo, q);
        else hi = std::min(hi, q);
      }
      auto g = [&](double e) { return e - e0 - sigma_p_re(pp, rr, e); };
      lo += 1e-9; hi -= 1e-9;
      for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (g(mid) < 0.0) lo = mid; else hi = mid;
      }
      double e = 0.5 * (lo + hi);
      for (int it = 0; it < 40; ++it) {
        double de = -g(e) / (1.0 - sigma_p_dre(pp, rr, e));
        e += de;
        if (std::abs(de) < 1e-16) break;
      }
      return e;
    }

    // --- the QM1-a/b/c fixture -------------------------------------------------------------
    inline ComplexType sigma4(ComplexType z) { return sigma_p(pk, rk, z); }
    inline double sigma4_re(double e) { return sigma_p_re(pk, rk, e); }
    inline double sigma4_dre(double e) { return sigma_p_dre(pk, rk, e); }
    inline double exact_root(double e0) { return root_p(pk, rk, e0); }
    /** exact Taylor coefficient about a real z0: c_n = (-1)^n sum_k r_k / (z0 - p_k)^{n+1}. */
    inline double sigma4_cn(double z0, long n) {
      double s = 0.0;
      for (size_t k = 0; k < pk.size(); ++k) s += rk[k] / std::pow(z0 - pk[k], double(n + 1));
      return ((n % 2 == 0) ? 1.0 : -1.0) * s;
    }

    /** Toy matrix model residues A^k = a_k a_k^dag (Hermitian, rank one), nb = 3. */
    inline nda::array<ComplexType, 3> residue_mats() {
      const long nb = 3;
      nda::array<ComplexType, 2> a(4, nb);
      a(0, 0) = ComplexType( 0.80,  0.00); a(0, 1) = ComplexType(-0.30,  0.20);
      a(0, 2) = ComplexType( 0.15, -0.40);
      a(1, 0) = ComplexType( 0.20, -0.50); a(1, 1) = ComplexType( 0.60,  0.10);
      a(1, 2) = ComplexType(-0.25,  0.30);
      a(2, 0) = ComplexType(-0.45,  0.35); a(2, 1) = ComplexType( 0.10,  0.55);
      a(2, 2) = ComplexType( 0.70,  0.00);
      a(3, 0) = ComplexType( 0.33,  0.22); a(3, 1) = ComplexType(-0.50, -0.15);
      a(3, 2) = ComplexType( 0.40,  0.28);
      nda::array<ComplexType, 3> A(4, nb, nb);
      for (long k = 0; k < 4; ++k)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) A(k, i, j) = a(k, i) * std::conj(a(k, j));
      return A;
    }
    /** Sigma_ij(z) = sum_k A^k_ij / (z - p_k) -- Hermitian-Lehmann by construction. */
    inline nda::array<ComplexType, 2> sigma_mat(nda::array<ComplexType, 3> const &A,
                                                ComplexType z) {
      const long nb = A.shape(1);
      nda::array<ComplexType, 2> S(nb, nb);
      S() = ComplexType(0.0);
      for (long k = 0; k < 4; ++k)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) S(i, j) += A(k, i, j) / (z - pk[k]);
      return S;
    }
    /** the polynomial sum_n c_n (i t)^n at the sample point t (both signs handled by t < 0). */
    inline ComplexType poly_it(nda::array<ComplexType, 1> const &c, double t) {
      ComplexType s(0.0), w(1.0, 0.0);
      const ComplexType it(0.0, t);
      for (long n = 0; n < c.shape(0); ++n) { s += c(n) * w; w *= it; }
      return s;
    }

  } // route_a_detail

  TEST_CASE("route_a_qp_root_four_pole", "[methods][qpgw][qp_maps][route_a]") {
    // QM1-a: Route A with re-expansion (p = 2, n_reexp = 4, analytic sampler) against the
    // EXACT real-axis roots of the four-pole model, for five static parts. Gate: < 1e-3
    // (1 meV in model units) -- the increment's acceptance criterion, NOT tunable.
    using namespace route_a_detail;
    const double beta = 1000.0;
    sra::fit_opts opt;                                  // p = 2, m = 3p = 6, n_reexp = 4
    auto S = [](ComplexType z) { return sigma4(z); };
    const std::vector<double> e0s{0.5, 1.5, 3.0, -0.8, -2.5};
    double worst = 0.0;
    for (double e0 : e0s) {
      const double ex = exact_root(e0);
      auto res = sra::qp_root(S, e0, beta, 0.0, opt);
      const double err = std::abs(res.eps - ex);
      const double Z = 1.0 / (1.0 - sigma4_dre(ex));
      app_log(1, "QM1-a e0 = {:+6.3f}: eps_exact = {:+.10f}  eps_routeA = {:+.10f}  "
                 "err = {:.3e} ({:.4f} meV)  Z = {:.4f}  n_reexp = {}  conv = {}  "
                 "rel_resid = {:.2e}  imag_c_rel = {:.2e}  R_conv = {:.4f}  |eps-z0|/R = {:.2e}",
              e0, ex, res.eps, err, err * 1e3, Z, res.n_reexp_used, res.converged,
              res.diag.rel_resid, res.diag.imag_c_rel, res.diag.R_conv, res.diag.dist_over_R);
      REQUIRE(std::isfinite(res.eps));
      REQUIRE(err < 1e-3);
      worst = std::max(worst, err);
    }
    app_log(1, "QM1-a worst error over the five states = {:.3e} ({:.4f} meV)", worst, worst * 1e3);
    REQUIRE(worst < 1e-3);
  }

  TEST_CASE("route_a_fit_internal_consistency", "[methods][qpgw][qp_maps][route_a]") {
    // QM1-b: the fitted expansion reproduces its input samples.
    using namespace route_a_detail;
    sra::fit_opts opt;
    const long p = opt.p, m = sra::m_default(p);

    // (b1) data that IS in the fitted class -- an exact degree-p polynomial in (i t) with the
    // reflection structure. This pins the LS solve, the u = t/t_max scaling round trip and the
    // +/- design at machine precision; the residual has nowhere to hide.
    {
      REQUIRE(p == 2);                      // cex below is written for the default order
      auto tp = sra::window_nodes(1000.0, m);
      nda::array<ComplexType, 1> cex(p + 1);
      cex(0) = ComplexType(-0.1700,  0.0450);
      cex(1) = ComplexType( 0.0310, -0.0120);
      cex(2) = ComplexType(-0.0054,  0.0021);
      nda::array<ComplexType, 1> Fp(m), Fm(m);
      for (long k = 0; k < m; ++k) {
        Fp(k) = poly_it(cex,  tp(k));
        Fm(k) = poly_it(cex, -tp(k));
      }
      sra::fit_diag d;
      auto c = sra::fit_taylor(tp, Fp, Fm, p, d);
      double cmax = 0.0, dev = 0.0;
      for (long n = 0; n <= p; ++n) {
        cmax = std::max(cmax, std::abs(cex(n)));
        dev = std::max(dev, std::abs(c(n) - cex(n)));
      }
      app_log(1, "QM1-b (representable data): rel resid = {:.3e}, max|c - c_exact|/max|c| = {:.3e}",
              d.rel_resid, dev / cmax);
      REQUIRE(d.rel_resid < 1e-13);
      REQUIRE(dev < 1e-10 * cmax);
    }

    // (b2) the four-pole model, production p = 2 and 3p nodes, at a beta cold enough that the
    // Taylor TRUNCATION (~ |c_3| t_max^3 / |c_0|) sits below the 1e-10 gate: t_max = w_5 ~ 3.5e-4.
    {
      const double beta_b = 1.0e5;
      auto tp = sra::window_nodes(beta_b, m);
      nda::array<ComplexType, 1> Fp(m), Fm(m);
      for (long k = 0; k < m; ++k) {
        Fp(k) = sigma4(ComplexType(0.0,  tp(k)));
        Fm(k) = sigma4(ComplexType(0.0, -tp(k)));
      }
      sra::fit_diag d;
      auto c = sra::fit_taylor(tp, Fp, Fm, p, d);
      // Sigmahat re-evaluated at the IN-WINDOW Matsubara points vs the input data
      double num = 0.0, den = 0.0;
      for (long k = 0; k < m; ++k) {
        num = std::max(num, std::abs(poly_it(c,  tp(k)) - Fp(k)));
        num = std::max(num, std::abs(poly_it(c, -tp(k)) - Fm(k)));
        den = std::max(den, std::max(std::abs(Fp(k)), std::abs(Fm(k))));
      }
      app_log(1, "QM1-b (four-pole, beta = {:.1e}): rel resid = {:.3e}, in-window max-norm "
                 "sample reproduction = {:.3e}", beta_b, d.rel_resid, num / den);
      REQUIRE(d.rel_resid < 1e-10);
      REQUIRE(num < 1e-10 * den);
    }

    // (b3) the SAME fit at the production beta = 1000. Here the residual CANNOT reach 1e-10:
    // it is the p = 2 truncation of a four-pole function over a window of half-width w_5, i.e.
    // an analytic floor of order |c_3| t_max^3 / |c_0| ~ 1e-7 -- a property of the model and the
    // window, not a solver defect. Gated at that analytic class from BOTH sides so a silent
    // collapse to machine zero (which would mean the higher terms were being absorbed) fails too.
    {
      auto tp = sra::window_nodes(1000.0, m);
      nda::array<ComplexType, 1> Fp(m), Fm(m);
      for (long k = 0; k < m; ++k) {
        Fp(k) = sigma4(ComplexType(0.0,  tp(k)));
        Fm(k) = sigma4(ComplexType(0.0, -tp(k)));
      }
      sra::fit_diag d;
      auto c = sra::fit_taylor(tp, Fp, Fm, p, d);
      const double tmax = tp(m - 1);
      const double cls = (std::abs(sigma4_cn(0.0, p + 1)) * std::pow(tmax, double(p + 1)) +
                          std::abs(sigma4_cn(0.0, p + 2)) * std::pow(tmax, double(p + 2))) /
                         std::abs(sigma4_cn(0.0, 0));
      app_log(1, "QM1-b (four-pole, production beta = 1000): rel resid = {:.3e}, analytic "
                 "truncation class = {:.3e} (ratio = {:.4f})", d.rel_resid, cls, d.rel_resid / cls);
      REQUIRE(d.rel_resid < cls);
      REQUIRE(d.rel_resid > 1e-3 * cls);
    }
  }

  TEST_CASE("route_a_static_anchor", "[methods][qpgw][qp_maps][route_a]") {
    // QM1-c (spec section 6.3): the z0 = 0 expansion evaluated at eps = 0 against the model's
    // exact Sigma(0) -- an assumption-free anchor; on exact data the error is fit truncation only.
    using namespace route_a_detail;
    const double beta = 1000.0;
    sra::fit_opts opt;
    const long p = opt.p, m = sra::m_default(p);

    // ---- scalar ----------------------------------------------------------------------------
    {
      auto tp = sra::window_nodes(beta, m);
      nda::array<ComplexType, 1> Fp(m), Fm(m);
      for (long k = 0; k < m; ++k) {
        Fp(k) = sigma4(ComplexType(0.0,  tp(k)));
        Fm(k) = sigma4(ComplexType(0.0, -tp(k)));
      }
      sra::fit_diag d;
      auto c = sra::fit_taylor(tp, Fp, Fm, p, d);
      const ComplexType s0 = sra::eval_taylor(c, 0.0);
      const double exact = sigma4_re(0.0);
      app_log(1, "QM1-c scalar: Sigmahat(0) = {:+.12e}, exact Sigma(0) = {:+.12e}, |diff| = {:.3e}, "
                 "|Im| = {:.3e}", s0.real(), exact, std::abs(s0.real() - exact), std::abs(s0.imag()));
      REQUIRE(std::abs(s0.real() - exact) < 1e-6);
    }

    // ---- matrix: Hermitian at machine, and the Q1 even-quadratic cross-check ----------------
    const long nb = 3;
    auto A = residue_mats();
    auto wn = qpmats_detail::wmesh(m, beta);
    nda::array<ComplexType, 3> Sw(m, nb, nb);
    for (long k = 0; k < m; ++k) {
      auto Sk = sigma_mat(A, ComplexType(0.0, wn(k)));
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Sw(k, i, j) = Sk(i, j);
    }
    auto X = sra::fit_matsubara(Sw, wn, opt);          // z0 = 0, dagger identity for the -t half
    nda::array<ComplexType, 2> Sh0(nb, nb);
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) Sh0(i, j) = X.eval(i, j, 0.0);
    auto S0exact = sigma_mat(A, ComplexType(0.0, 0.0));

    double herr = 0.0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        herr = std::max(herr, std::abs(Sh0(i, j) - std::conj(Sh0(j, i))));
    const double aerr = qpmats_detail::maxdiff(Sh0, S0exact);

    // The Q1 even-quadratic extrapolation S0 = (w1^2 H(w0) - w0^2 H(w1)) / (w1^2 - w0^2) with
    // H the Hermitian part -- the qp_lin_matrix internals' convention (qp_maps_matsubara.hpp
    // header lines 39-41), reproduced here because S0 is not exposed. CONSISTENCY, not precision:
    // it carries its own O(w^2)-removed truncation and is a different estimator of the same Sigma(0).
    nda::array<ComplexType, 2> S0q(nb, nb);
    const double w0 = wn(0), w1 = wn(1), dd = w1 * w1 - w0 * w0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) {
        const ComplexType H0 = 0.5 * (Sw(0, i, j) + std::conj(Sw(0, j, i)));
        const ComplexType H1 = 0.5 * (Sw(1, i, j) + std::conj(Sw(1, j, i)));
        S0q(i, j) = (w1 * w1 * H0 - w0 * w0 * H1) / dd;
      }
    const double qerr = qpmats_detail::maxdiff(S0q, Sh0);
    app_log(1, "QM1-c matrix: max|Sigmahat(0) - Sigma(0)| = {:.3e}, hermiticity = {:.3e}, "
               "max|S0_Q1 - Sigmahat(0)| = {:.3e}", aerr, herr, qerr);
    REQUIRE(aerr < 1e-6);
    REQUIRE(herr < 1e-15);            // by construction: c(n,j,i) = conj(c(n,i,j))
    REQUIRE(qerr < 1e-6);
  }

  TEST_CASE("route_a_vxc_hermiticity_covariance", "[methods][qpgw][qp_maps][route_a]") {
    // QM1-d: the mode-A V^xc assembly. (1) hermiticity by construction at distinct eps_i;
    // (2) unitary covariance in the SHARED-functional setting only.
    using namespace route_a_detail;
    const long nb = 3;
    const double beta = 1000.0;
    sra::fit_opts opt;
    const long m = sra::m_default(opt.p);
    auto wn = qpmats_detail::wmesh(m, beta);
    auto A = residue_mats();

    auto build_samples = [&](nda::array<ComplexType, 3> const &Ak) {
      nda::array<ComplexType, 3> S(m, nb, nb);
      for (long k = 0; k < m; ++k) {
        auto Sk = sigma_mat(Ak, ComplexType(0.0, wn(k)));
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) S(k, i, j) = Sk(i, j);
      }
      return S;
    };
    auto Sw = build_samples(A);
    auto X = sra::fit_matsubara(Sw, wn, opt);

    // ---- (1) hermiticity at DISTINCT eps_i --------------------------------------------------
    nda::array<double, 1> eps(nb);
    eps(0) = 0.285; eps(1) = -0.836; eps(2) = 1.132;
    auto V = sra::assemble_vxc(X, eps);
    double herr = 0.0, dimag = 0.0;
    for (long i = 0; i < nb; ++i) {
      dimag = std::max(dimag, std::abs(V(i, i).imag()));
      for (long j = 0; j < nb; ++j)
        herr = std::max(herr, std::abs(V(i, j) - std::conj(V(j, i))));
    }
    app_log(1, "QM1-d hermiticity (distinct eps): max|V - V^dag| = {:.3e}, max|Im V_ii| = {:.3e}",
            herr, dimag);
    REQUIRE(herr < 1e-14);
    REQUIRE(dimag < 1e-16);

    // ---- (2) unitary covariance, SHARED evaluation functional -------------------------------
    // With every eps_i equal, the fit + eval collapses to ONE fixed linear map applied entrywise
    // to the pair (Sigma(z0 + i t_k), Sigma^dag(z0 + i t_k)); both transform covariantly under
    // Sigma -> U Sigma U^dag, so V^xc must rotate exactly.
    //
    // PINNED: with DISTINCT eps_i exact covariance does NOT hold and is deliberately NOT gated.
    // Mode A is DEFINED in the eigenbasis -- eps_i labels state i of THAT frame, so rotating the
    // model while holding the same list of eps_i compares two different prescriptions, not two
    // representations of one. The deviation is measured and logged below to keep that honest.
    nda::array<ComplexType, 2> Hr(nb, nb);
    Hr() = ComplexType(0.0);
    Hr(0, 0) = 0.3; Hr(1, 1) = -0.7; Hr(2, 2) = 1.1;
    Hr(0, 1) = ComplexType(0.40,  0.25); Hr(1, 0) = std::conj(Hr(0, 1));
    Hr(0, 2) = ComplexType(-0.20, 0.50); Hr(2, 0) = std::conj(Hr(0, 2));
    Hr(1, 2) = ComplexType(0.35, -0.15); Hr(2, 1) = std::conj(Hr(1, 2));
    auto [evr, U] = nda::linalg::eigenelements(nda::matrix<ComplexType>(Hr));
    double uerr = 0.0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) {
        ComplexType s(0.0);
        for (long l = 0; l < nb; ++l) s += std::conj(U(l, i)) * U(l, j);
        uerr = std::max(uerr, std::abs(s - ComplexType(i == j ? 1.0 : 0.0)));
      }
    REQUIRE(uerr < 1e-13);

    auto rot = [&](nda::array<ComplexType, 2> const &M) {     // U M U^dag, explicit loops
      nda::array<ComplexType, 2> T(nb, nb), Y(nb, nb);
      T() = ComplexType(0.0); Y() = ComplexType(0.0);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          for (long l = 0; l < nb; ++l) T(i, j) += U(i, l) * M(l, j);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          for (long l = 0; l < nb; ++l) Y(i, j) += T(i, l) * std::conj(U(j, l));
      return Y;
    };

    nda::array<ComplexType, 3> AU(4, nb, nb);                 // rotate the model RESIDUES
    for (long k = 0; k < 4; ++k) {
      nda::array<ComplexType, 2> Ak(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Ak(i, j) = A(k, i, j);
      auto Rk = rot(Ak);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) AU(k, i, j) = Rk(i, j);
    }
    auto XU = sra::fit_matsubara(build_samples(AU), wn, opt);

    nda::array<double, 1> epss(nb);
    epss() = 0.285;                                           // the SHARED functional
    auto Vs = sra::assemble_vxc(X, epss);
    auto VU = sra::assemble_vxc(XU, epss);
    const double dev = qpmats_detail::maxdiff(VU, rot(Vs));
    // descriptive only: the same comparison with DISTINCT eps, which is NOT a covariance defect
    const double dev_distinct = qpmats_detail::maxdiff(sra::assemble_vxc(XU, eps), rot(V));
    app_log(1, "QM1-d covariance: shared-functional max|V[U S U^dag] - U V[S] U^dag| = {:.3e} "
               "(gated); distinct-eps deviation = {:.3e} (NOT gated -- mode A is eigenbasis-defined)",
            dev, dev_distinct);
    REQUIRE(dev < 1e-10);
  }

#ifdef ENABLE_DLR
  TEST_CASE("route_a_dlr_pole_chain", "[methods][qpgw][qp_maps][route_a]") {
    // QM1-e: the PRODUCTION chain rehearsal. Four-pole scalar model -> Sigma(tau) on the real
    // IAFT fermionic tau mesh, built with the SAME kernel convention as imag_axes_ft::dlr_kF
    // -> dlr_pole_fit::build + coeffs() -> pole-rep sampler -> full re-expanded QP roots for
    // the five states against the exact real-axis roots. Gate: < 1e-3 (1 meV), NOT tunable.
    // The sampler's poles lie ON the real axis, hence the mandatory |t| >= w_0 window floor.
    //
    // DEVIATION FROM THE SPEC'S SUGGESTED FIXTURE, and the reason for it (measured, see the
    // logged caveat block at the end of this case). The spec asks for Sigma(tau) built from
    // poles {-9,-6,6,8} "so the pole fit is exact-to-eps". It is NOT: a least-squares DLR pole
    // fit of tau data whose poles are NOT auxiliary-grid nodes is exact-to-eps ON THE IMAGINARY
    // AXIS only. Off it -- which is exactly where re-expansion about z0 != 0 evaluates the
    // sampler -- the fitted measure differs from the true one by residues of order 1e-2 spread
    // over the aux grid, and at z = z0 + i w_0 those get divided by the LOCAL GRID SPACING near
    // z0 (~3e-2), giving an O(1) error. Measured with {-9,-6,6,8}: tau-space fit_error 6.8e-9,
    // |S - Sigma| = 1.8e-7 at z0 = 0 but 1.77 at z0 = 0.3, and QP root errors of 60-450 meV.
    // The |t| >= w_0 floor does not help, because the ill-conditioning is controlled by the
    // ANGLE t/|z0|, not by t alone (two-constants/Hadamard three-lines: the error interpolates
    // between the imaginary-axis accuracy and the global bound as the ray tilts toward the real
    // axis). To make the spec's own premise TRUE the model poles are therefore taken to BE
    // auxiliary-grid nodes -- the two nodes nearest zero with |eps| >= 6 on each side -- so the
    // tau data is genuinely in the span and the fit recovers the exact sparse residues. What
    // QM1-e then measures is the EVALUATOR chain, which is what it is for.
    using namespace route_a_detail;
    const double beta = 1000.0, wmax = 12.0;      // wmax must bracket the model poles
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "low");
    imag_axes_ft::dlr_pole_fit pf(ft);
    app_log(1, "QM1-e grid: np = {}, nt = {}, n_kept = {}, min|hw_l| = {:.4g}, min gap = {:.4g}",
            pf.np, pf.nt, pf.n_kept, pf.min_abs_node, pf.min_node_gap);
    // Exact recovery of a sparse residue vector needs the pole basis at FULL rank; a truncation
    // that drops directions re-spreads the residues and the near-real-axis fidelity goes with it.
    REQUIRE(pf.n_kept == pf.np);

    // the model poles: nodes nearest zero with |eps| >= 6, two per side (straddling, |p_k| >= 6)
    std::vector<double> pe;
    for (int sgn : {-1, 1})
      for (int pick = 0; pick < 2; ++pick) {
        double best = 1e300, bv = 0.0;
        for (long p = 0; p < pf.np; ++p) {
          const double q = pf.epsl(p);
          if (double(sgn) * q < 6.0) continue;
          if (std::find(pe.begin(), pe.end(), q) != pe.end()) continue;
          if (std::abs(q) < best) { best = std::abs(q); bv = q; }
        }
        REQUIRE(best < 1e299);
        pe.push_back(bv);
      }
    const std::vector<double> re{2.4, 3.0, 3.6, 2.6};
    app_log(1, "QM1-e model poles (aux nodes): {:+.8f} {:+.8f} {:+.8f} {:+.8f}, "
               "Z(0) = {:.4f}", pe[0], pe[1], pe[2], pe[3],
            1.0 / (1.0 - sigma_p_dre(pe, re, 0.0)));

    nda::array<ComplexType, 2> F(pf.nt, 1);
    for (long i = 0; i < pf.nt; ++i) {
      double v = 0.0;
      for (size_t k = 0; k < pe.size(); ++k)
        v += re[k] * imag_axes_ft::dlr_kF(pf.beta, pf.s_phys(i), pe[k]);
      F(i, 0) = ComplexType(v);
    }
    auto cp = pf.coeffs(F);
    const double ferr = pf.fit_error(F, cp), fratio = pf.residue_ratio(F, cp);
    app_log(1, "QM1-e pole fit: fit_error = {:.3e}, residue_ratio = {:.4f}", ferr, fratio);
    imag_axes_ft::dlr_pole_fit_gate(ferr, "QM1-e");

    sra::pole_sampler S{nda::array<ComplexType, 1>(pf.np), pf.epsl};
    for (long p = 0; p < pf.np; ++p) S.c(p) = cp(p, 0);

    // precondition: the pole rep must agree with the model where the evaluator samples it
    double smax = 0.0;
    for (double z0 : {0.0, 0.3, 1.2, 2.4, -0.9, -2.3}) {
      const ComplexType z(z0, M_PI / beta);
      smax = std::max(smax, std::abs(S(z) - sigma_p(pe, re, z)));
    }
    app_log(1, "QM1-e sampler vs model at |t| = w_0: max|S(z) - Sigma(z)| = {:.3e}", smax);
    REQUIRE(smax < 1e-6);

    sra::fit_opts opt;                                        // p = 2, m = 3p = 6, n_reexp = 4
    const std::vector<double> e0s{0.5, 1.5, 3.0, -0.8, -2.5};
    double worst = 0.0;
    for (double e0 : e0s) {
      const double ex = root_p(pe, re, e0);
      auto res = sra::qp_root(S, e0, beta, 0.0, opt);
      const double err = std::abs(res.eps - ex);
      app_log(1, "QM1-e e0 = {:+6.3f}: eps_exact = {:+.10f}  eps_poleRep = {:+.10f}  "
                 "err = {:.3e} ({:.4f} meV)  n_reexp = {}  conv = {}  rel_resid = {:.2e}  "
                 "imag_c_rel = {:.2e}  R_conv = {:.4f}  |eps-z0|/R = {:.2e}",
              e0, ex, res.eps, err, err * 1e3, res.n_reexp_used, res.converged,
              res.diag.rel_resid, res.diag.imag_c_rel, res.diag.R_conv, res.diag.dist_over_R);
      REQUIRE(std::isfinite(res.eps));
      REQUIRE(err < 1e-3);
      worst = std::max(worst, err);
    }
    app_log(1, "QM1-e worst error over the five states = {:.3e} ({:.4f} meV)", worst, worst * 1e3);
    REQUIRE(worst < 1e-3);

    // ---- the plan's flagged caveat, MEASURED (logged, not gated) ---------------------------
    // Two separate things were flagged, and only one of them is real.
    //  (1) "the imaginary residual grows near |t| = w_0". It does NOT: imag_c_rel stays at
    //      machine level for every window below, because a real-residue/real-pole rep satisfies
    //      S(z*) = S(z)* identically, so the +/- samples are exact conjugates by construction.
    //  (2) The REAL failure mode is off-imaginary-axis continuation of a pole fit whose data is
    //      not in the span. The block below reruns the whole chain with the spec's suggested
    //      poles {-9,-6,6,8} (NOT aux nodes) and reports what happens.
    for (long n0 : {0L, 2L, 5L, 10L, 40L}) {
      sra::fit_opts o;
      o.n0 = n0;
      auto res = sra::qp_root(S, 0.5, beta, 0.0, o);
      app_log(1, "QM1-e window probe n0 = {:2d} (|t| >= {:.4e}): eps = {:+.10f}, err = {:.3e}, "
                 "imag_c_rel = {:.3e}, rel_resid = {:.3e}",
              n0, (2.0 * double(n0) + 1.0) * M_PI / beta, res.eps,
              std::abs(res.eps - root_p(pe, re, 0.5)), res.diag.imag_c_rel, res.diag.rel_resid);
    }
    {
      nda::array<ComplexType, 2> Fo(pf.nt, 1);
      for (long i = 0; i < pf.nt; ++i) {
        double v = 0.0;
        for (size_t k = 0; k < pk.size(); ++k)
          v += rk[k] * imag_axes_ft::dlr_kF(pf.beta, pf.s_phys(i), pk[k]);
        Fo(i, 0) = ComplexType(v);
      }
      auto co = pf.coeffs(Fo);
      sra::pole_sampler So{nda::array<ComplexType, 1>(pf.np), pf.epsl};
      for (long p = 0; p < pf.np; ++p) So.c(p) = co(p, 0);
      app_log(1, "QM1-e OFF-GRID caveat (poles {{-9,-6,6,8}}, NOT aux nodes): tau fit_error = "
                 "{:.3e} yet |S - Sigma| at |t| = w_0 is {:.3e} at z0 = 0 but {:.3e} at z0 = 0.3 "
                 "and {:.3e} at z0 = 2.4 -- the fit is exact-to-eps on the IMAGINARY AXIS only.",
              pf.fit_error(Fo, co),
              std::abs(So(ComplexType(0.0, M_PI / beta)) - sigma4(ComplexType(0.0, M_PI / beta))),
              std::abs(So(ComplexType(0.3, M_PI / beta)) - sigma4(ComplexType(0.3, M_PI / beta))),
              std::abs(So(ComplexType(2.4, M_PI / beta)) - sigma4(ComplexType(2.4, M_PI / beta))));
      for (double e0 : e0s) {
        auto res = sra::qp_root(So, e0, beta, 0.0, opt);
        app_log(1, "QM1-e OFF-GRID e0 = {:+6.3f}: err = {:.3e} ({:.2f} meV), imag_c_rel = {:.3e}",
                e0, std::abs(res.eps - exact_root(e0)), std::abs(res.eps - exact_root(e0)) * 1e3,
                res.diag.imag_c_rel);
      }
    }
  }
#endif

} // bdft_tests
