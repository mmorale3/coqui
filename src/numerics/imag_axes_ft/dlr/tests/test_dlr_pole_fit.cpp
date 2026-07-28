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

// Unit tests for imag_axes_ft::dlr_pole_fit.
//
// The last case is the REGRESSION: it is the failure that produced the 2026-07-27
// scGW+vertex divergence, and it is the one no pre-existing check could see. The old
// residue route (interpolate onto the auxiliary tau nodes, then run a square
// interpolatory solve there) is a fixed linear map of 2-norm 9.4e6 at the production
// grid. Feeding it Lehmann data perturbed by a RELATIVE 1e-4 along its worst-conditioned
// direction returned residues 379x the data and a fit error of 2.4e-3 -- reproducing the
// observed break (383x, 3.19e-3) to a few percent -- and the downstream algebras are
// bilinear in those residues, so it entered squared.
//
// The old self-check validated the pole CONVENTION against a single synthetic pole, which
// is trivially in the span; it passed through every divergence. Hence case 1 below uses a
// MULTI-pole object, and case 4 perturbs deliberately.

#undef NDEBUG

#include <vector>
#include <cmath>
#include <complex>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"

namespace bdft_tests {

  using cplx = ComplexType;

  // A genuine Lehmann object sampled on the backend tau grid:  F(tau) = sum_k w_k K_F(tau, e_k)
  static nda::array<cplx, 2> lehmann_tau(imag_axes_ft::dlr_pole_fit const &pf,
                                         std::vector<double> const &e,
                                         std::vector<double> const &w, long ncol = 1) {
    nda::array<cplx, 2> F(pf.nt, ncol);
    for (long i = 0; i < pf.nt; ++i)
      for (long j = 0; j < ncol; ++j) {
        double v = 0;
        for (size_t k = 0; k < e.size(); ++k)
          v += (1.0 + 0.1 * double(j)) * w[k] * imag_axes_ft::dlr_kF(pf.beta, pf.s_phys(i), e[k]);
        F(i, j) = v;
      }
    return F;
  }

  static double maxabs(auto const &A) {
    double m = 0;
    for (auto const &v : A) m = std::max(m, std::abs(v));
    return m;
  }

  TEST_CASE("dlr_pole_fit", "[iaft]") {
    double beta = 1000.0, wmax = 4.0;   // lambda = 4000, close to the production grid
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "low");
    imag_axes_ft::dlr_pole_fit pf(ft);

    REQUIRE(pf.np > 0);
    REQUIRE(pf.nt == ft.nt_f());
    REQUIRE(pf.n_kept > 0);
    REQUIRE(pf.n_kept <= std::min(pf.nt, pf.np));

    // The auxiliary grid must stay well separated: Sigma^C's families (II)/(III) deflate the
    // shared node BY INDEX (family (V) restores it analytically), which is only equivalent to
    // deflating by value on a separated grid.
    REQUIRE(pf.min_node_gap > 1e-6);
    REQUIRE(pf.min_abs_node > 1e-12);

    // ---- 1. multi-pole Lehmann data round-trips to the DLR accuracy ---------------------
    std::vector<double> e{-3.1, -0.9, -0.05, 0.4, 1.7, 3.6};
    std::vector<double> w{0.7, -1.3, 0.45, 1.0, -0.6, 0.25};
    {
      auto F = lehmann_tau(pf, e, w, 3);
      auto c = pf.coeffs(F);
      REQUIRE(c.shape(0) == pf.np);
      REQUIRE(c.shape(1) == 3);
      double err = pf.fit_error(F, c);
      // exact-to-eps for data that IS in the represented class
      REQUIRE(err < 1e-4);
      // and the residues stay the size of the data -- this is what the old route lost
      REQUIRE(maxabs(c) < 50.0 * maxabs(F));
    }

    // ---- 2. the pole CONVENTION: a tau fit continues to sum_p c_p/(z - eps_p) -----------
    {
      auto F = lehmann_tau(pf, e, w, 1);
      auto c = pf.coeffs(F);
      auto wf = ft.wn_mesh_f();
      double err = 0, scale = 0;
      for (long n = 0; n < std::min(6L, (long)wf.size()); ++n) {
        cplx z(0.0, double(wf(n)) * M_PI / beta);
        cplx got = 0, ref = 0;
        for (long p = 0; p < pf.np; ++p) got += c(p, 0) / (z - pf.epsl(p));
        for (size_t k = 0; k < e.size(); ++k) ref += w[k] / (z - e[k]);
        err = std::max(err, std::abs(got - ref));
        scale = std::max(scale, std::abs(ref));
      }
      // a wrong convention is an O(1) RELATIVE error, not a small one
      REQUIRE(err / scale < 1e-4);
    }

    // ---- 3. a looser accuracy target really does use fewer directions -------------------
    // n_kept is per-call state (the rank is chosen from the data), so compare AFTER fitting
    // the same object with two different targets.
    {
      imag_axes_ft::dlr_pole_fit coarse(ft, 1e-2);
      auto F = lehmann_tau(pf, e, w, 1);
      auto c_fine = pf.coeffs(F);
      long k_fine = pf.n_kept;
      auto c_coarse = coarse.coeffs(F);
      REQUIRE(coarse.n_kept < k_fine);
      // and the loose fit is correspondingly less accurate, but still meets its own target
      REQUIRE(coarse.fit_error(F, c_coarse) > pf.fit_error(F, c_fine));
      REQUIRE(coarse.fit_error(F, c_coarse) < 1e-1);
    }

    // ---- 3b. BATCH SEMANTICS: batched must equal per-element, bit for bit ---------------
    // The rank is data-dependent, so it must be chosen per COLUMN; a rank derived from
    // batch-summed norms would couple independent columns.
    {
      auto Fb = lehmann_tau(pf, e, w, 4);
      auto cb = pf.coeffs(Fb);
      for (long j = 0; j < 4; ++j) {
        nda::array<cplx, 2> Fj(pf.nt, 1);
        for (long i = 0; i < pf.nt; ++i) Fj(i, 0) = Fb(i, j);
        auto cj = pf.coeffs(Fj);
        double d = 0;
        for (long p = 0; p < pf.np; ++p) d = std::max(d, std::abs(cb(p, j) - cj(p, 0)));
        REQUIRE(d == 0.0);
      }
    }

    // ---- 4. REGRESSION: unrepresentable content must NOT explode the residues -----------
    // Perturb Lehmann data along the direction the pole kernel resolves LEAST (its smallest
    // retained right-singular direction is inside the span, so use the tau-space direction
    // orthogonal to what the kernel can reach: the residual of a random vector after
    // projection onto range(K)). Relative size 1e-4 -- the level that broke production.
    {
      auto F = lehmann_tau(pf, e, w, 1);
      double fmax = maxabs(F);

      // build a tau-space vector with no component the poles can represent
      nda::array<double, 1> r(pf.nt);
      unsigned seed = 12345;
      for (long i = 0; i < pf.nt; ++i) {
        seed = seed * 1103515245u + 12345u;
        r(i) = 2.0 * (double((seed >> 16) & 0x7fff) / 32767.0) - 1.0;
      }
      // Gram-Schmidt r against every column of Kmat
      for (int pass = 0; pass < 2; ++pass)
        for (long p = 0; p < pf.np; ++p) {
          double kk = 0, kr = 0;
          for (long i = 0; i < pf.nt; ++i) { kk += pf.Kmat(i, p) * pf.Kmat(i, p); kr += pf.Kmat(i, p) * r(i); }
          if (kk <= 0) continue;
          double a = kr / kk;
          for (long i = 0; i < pf.nt; ++i) r(i) -= a * pf.Kmat(i, p);
        }
      double rn = 0;
      for (long i = 0; i < pf.nt; ++i) rn = std::max(rn, std::abs(r(i)));

      if (rn > 1e-12) {          // a rank-deficient kernel always leaves such a direction
        for (long i = 0; i < pf.nt; ++i) F(i, 0) += cplx(1e-4 * fmax * r(i) / rn);
        auto c = pf.coeffs(F);
        // The residues must stay O(data). The unregularized route returned 379x here.
        REQUIRE(maxabs(c) < 50.0 * fmax);
        // The fit error must REPORT the unrepresentable content rather than absorb it into
        // huge residues -- but stay far below the hard gate, since the content is tiny.
        double err = pf.fit_error(F, c);
        REQUIRE(err < 1e-2);
        // and the gate must not fire on a perturbation this small
        imag_axes_ft::dlr_pole_fit_gate(err, "test_dlr_pole_fit");
      }
    }
  }

} // namespace bdft_tests
