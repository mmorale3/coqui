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

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "methods/SCF/qp_maps_matsubara.hpp"

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

} // bdft_tests
