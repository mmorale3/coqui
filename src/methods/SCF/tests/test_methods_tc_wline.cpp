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
 * GATE TC-3-a of notes/tc_coqui_impl_spec.md, and the two pins Fable's review
 * asked for ahead of any fixture run.
 *
 *   tc_sigma_cd_single_pole   THE SIGN PIN. The eq-1 CD assembly with the
 *                             DERIVED sigma_m, against the EXACT finite-T
 *                             Lehmann self-energy, for one G pole and one
 *                             plasmon pair. Both the integral term and the
 *                             reference are closed form, so this is a machine-
 *                             precision identity and NOT quadrature-limited.
 *   tc_sigma_cd_sign_bites    The same pin with the spec-as-written (opposite)
 *                             signs must FAIL, and the beta -> infinity limit
 *                             of sigma_m = theta(w-eps) - f(eps) must be the
 *                             +1/-1/0 table of results section 2.3.
 *   tc_wc_line_dyson          W^c = ([I - Z.Pi]^{-1} - I).Z -- CoQuI's OWN
 *                             operation order -- against the campaign's
 *                             independently generated RPA W^c, by both of its
 *                             routes (dense solve and the closed-form plasmon
 *                             pole sum).
 *   tc_wc_line_krylov         dense == warm-started GMRES on the contracted
 *                             elements (the campaign measured 9e-13).
 *   tc_sigma_cd_multipole     the eq-1 assembly on the campaign's full
 *                             SigmaModel, against its exact Lehmann pole sum.
 *
 * The reference file is written by tc_validation/tests/export_tc3_reference.py.
 */

#undef NDEBUG

#include <cmath>
#include <complex>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/test_common.hpp"
#include "utilities/minimal_json.hpp"
#include "methods/SCF/wc_line.hpp"
#include "methods/SCF/sigma_cd_line.hpp"

namespace bdft_tests {

  namespace wl = methods::wc_line;
  namespace scd = methods::sigma_cd_line;
  using dcomplex = std::complex<double>;

  // ---- helpers for the exported complex tensors --------------------------
  static std::vector<long> jshape(mjson::Value const &v) {
    std::vector<long> s;
    auto const &a = v["shape"];
    for (std::size_t i = 0; i < a.size(); ++i) s.push_back(a[i].i());
    return s;
  }
  static std::vector<dcomplex> jdata(mjson::Value const &v) {
    auto re = v["re"].vd(), im = v["im"].vd();
    REQUIRE(re.size() == im.size());
    std::vector<dcomplex> out(re.size());
    for (std::size_t i = 0; i < re.size(); ++i) out[i] = dcomplex(re[i], im[i]);
    return out;
  }

  static std::string ref_path() {
    return std::string(PROJECT_SOURCE_DIR)
         + "/tests/unit_test_files/tilted_contour/tc3_reference.json";
  }

  // =====================================================================
  //  GATE TC-3-a -- THE SIGN PIN (closed form, machine precision)
  // =====================================================================
  //
  // One G pole at eps_m (weight 1), one PH-symmetric W^c pair:
  //     W^c(z) = r/(z - wp) - r/(z + wp)      (EVEN in z)
  // Exact finite-T self-energy, the sigma_route_b closed form:
  //     Sigma^c(z) = sum_j w_j [n_B(om_j) + f(eps_m)] / (z - eps_m + om_j)
  // with (w, om) = {(r, +wp), (-r, -wp)}.
  //
  static dcomplex exact_lehmann(dcomplex z, double eps_m, double mu,
                                double r, double wp, double beta) {
    const double f = scd::stable_nF(beta, eps_m - mu);
    dcomplex s(0.0, 0.0);
    const double w[2] = {r, -r};
    const double om[2] = {wp, -wp};
    for (int j = 0; j < 2; ++j)
      s += w[j] * (scd::stable_nB(beta, om[j]) + f) / (z - eps_m + om[j]);
    return s;
  }
  static dcomplex wc_pair(dcomplex z, double r, double wp) {
    return r / (z - wp) - r / (z + wp);
  }

  TEST_CASE("tc_sigma_cd_single_pole", "[methods][tc_wline]") {
    double worst = 0.0, worst_leftover = 0.0;
    long n = 0;
    // a deterministic sweep over (r, wp, eps_m, omega, beta)
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b)
        for (int c = 0; c < 5; ++c)
          for (int d = 0; d < 5; ++d)
            for (int e = 0; e < 2; ++e) {
              const double r = 0.2 + 2.8 * double(a) / 4.0;
              const double wp = 0.5 + 4.5 * double(b) / 4.0;
              const double eps = -8.0 + 16.0 * double(c) / 4.0;
              const double om = -8.0 + 16.0 * double(d) / 4.0;
              const double beta = (e == 0) ? 1000.0 : 10000.0;
              const double mu = 0.0;
              const double A = om - eps;
              if (std::abs(A) < 1e-2 or std::abs(eps - mu) < 1e-2) continue;

              nda::array<double, 1> wv(2), ov(2);
              wv(0) = r;  wv(1) = -r;
              ov(0) = wp; ov(1) = -wp;
              const dcomplex I = scd::imag_axis_term_poles(A, wv, ov);
              const double sg = scd::sigma_m_weight(om, eps, mu, beta);
              const dcomplex got = I + sg * wc_pair(dcomplex(A, 0.0), r, wp);
              const dcomplex ex = exact_lehmann(dcomplex(om, 0.0), eps, mu, r, wp, beta);
              worst = std::max(worst, std::abs(got - ex) / std::abs(ex));
              worst_leftover = std::max(worst_leftover,
                                        scd::thermal_leftover_bound(beta, ov));
              ++n;
            }
    app_log(2, "[TC-3-a sign pin] eq-1 with sigma_m = theta(w-eps) - f(eps) vs the "
               "EXACT finite-T Lehmann over {} (r, wp, eps, omega, beta) points: "
               "max rel = {:.3e}; bosonic leftover bound max |n_B(om)+theta(-om)| = "
               "{:.3e} (the ONE approximation eq 1 makes)", n, worst, worst_leftover);
    REQUIRE(n > 400);
    REQUIRE(worst < 1e-13);          // the spec's class; measured ~1e-14
  }

  TEST_CASE("tc_sigma_cd_sign_bites", "[methods][tc_wline]") {
    // (1) the beta -> infinity limit of sigma_m IS the results-2.3 table
    const double mu = 0.0, big = 1e8;
    REQUIRE(std::abs(scd::sigma_m_weight(+5.0, +2.0, mu, big) - 1.0) < 1e-12); // empty, below w
    REQUIRE(std::abs(scd::sigma_m_weight(-5.0, -2.0, mu, big) + 1.0) < 1e-12); // occ, above w
    REQUIRE(std::abs(scd::sigma_m_weight(-5.0, +2.0, mu, big)) < 1e-12);       // empty, above w
    REQUIRE(std::abs(scd::sigma_m_weight(+5.0, -2.0, mu, big)) < 1e-12);       // occ, below w
    app_log(2, "[TC-3-a sign pin] beta -> inf limit of sigma_m = theta(w-eps) - f(eps): "
               "+1 (mu<eps<w), -1 (w<eps<mu), 0 otherwise -- the DERIVED table of "
               "results section 2.3");

    // (2) THE OPPOSITE (spec-as-written) SIGNS MUST FAIL -- otherwise the pin
    //     above is not testing the signs at all.
    double worst_flipped = 0.0;
    for (int c = 0; c < 5; ++c)
      for (int d = 0; d < 5; ++d) {
        const double r = 1.3, wp = 2.2, beta = 1000.0;
        const double eps = -8.0 + 16.0 * double(c) / 4.0;
        const double om = -8.0 + 16.0 * double(d) / 4.0;
        const double A = om - eps;
        if (std::abs(A) < 1e-2 or std::abs(eps) < 1e-2) continue;
        nda::array<double, 1> wv(2), ov(2);
        wv(0) = r;  wv(1) = -r;
        ov(0) = wp; ov(1) = -wp;
        const double sg = -scd::sigma_m_weight(om, eps, 0.0, beta);   // FLIPPED
        const dcomplex got = scd::imag_axis_term_poles(A, wv, ov)
                           + sg * wc_pair(dcomplex(A, 0.0), r, wp);
        const dcomplex ex = exact_lehmann(dcomplex(om, 0.0), eps, 0.0, r, wp, beta);
        worst_flipped = std::max(worst_flipped, std::abs(got - ex) / std::abs(ex));
      }
    app_log(2, "[TC-3-a sign pin] with the OPPOSITE (spec-as-written) signs the same "
               "identity is off by {:.3e} -- the pin bites", worst_flipped);
    REQUIRE(worst_flipped > 1e-2);
  }

  // =====================================================================
  //  DOES THE FINITE-T sigma_m MATTER?  (Fable review point, section 6 item 5)
  //
  //  sigma_m = theta(w - eps_m) - f(eps_m) reduces to the campaign's +1/-1/0
  //  table as beta -> infinity. The question the SVO metal leg asks is whether
  //  the FRACTIONAL value matters for states within k_B T of mu. That is a
  //  property of the weight, not of the fixture, so it is answered here: score
  //  the exact finite-T Lehmann against eq 1 with (a) the fractional sigma_m
  //  and (b) the T = 0 step, as beta*|eps_m - mu| is swept through 1.
  // =====================================================================
  TEST_CASE("tc_sigma_cd_fractional_occupation", "[methods][tc_wline]") {
    const double r = 1.3, wp = 2.2, mu = 0.0, beta = 100.0;
    app_log(2, "[TC-3-a fractional] beta = {:.4g}; one G pole at eps_m, one plasmon pair "
               "(r = {:.3g}, omega_p = {:.3g}); the exact finite-T Lehmann vs eq 1 with "
               "(a) sigma_m = theta - f  and  (b) sigma_m = theta - step(mu - eps_m)",
            beta, r, wp);
    app_log(2, "[TC-3-a fractional] {:>12} {:>10} {:>12} {:>12} {:>12}",
            "beta*(eps-mu)", "f(eps)", "sigma_m", "(a) frac", "(b) T=0 step");
    double worst_frac = 0.0, worst_step = 0.0, worst_step_at_small = 0.0;
    for (double be : {-8.0, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 8.0}) {
      const double eps = be / beta;
      const double om = eps + 3.7;            // a fixed A = 3.7 > 0, well off any pole
      const double A = om - eps;
      nda::array<double, 1> wv(2), ov(2);
      wv(0) = r;  wv(1) = -r;
      ov(0) = wp; ov(1) = -wp;
      const dcomplex I = scd::imag_axis_term_poles(A, wv, ov);
      const double f = scd::stable_nF(beta, eps - mu);
      const double sg_frac = scd::sigma_m_weight(om, eps, mu, beta);
      const double sg_step = ((om > eps) ? 1.0 : 0.0) - ((eps < mu) ? 1.0 : 0.0);
      const dcomplex Wl = wc_pair(dcomplex(-A, 0.0), r, wp);      // eq (EXACT)
      const dcomplex ex = exact_lehmann(dcomplex(om, 0.0), eps, mu, r, wp, beta);
      const double ea = std::abs(I + sg_frac * Wl - ex) / std::abs(ex);
      const double eb = std::abs(I + sg_step * Wl - ex) / std::abs(ex);
      app_log(2, "[TC-3-a fractional] {:>12.2f} {:>10.4f} {:>12.4f} {:>12.3e} {:>12.3e}",
              be, f, sg_frac, ea, eb);
      worst_frac = std::max(worst_frac, ea);
      worst_step = std::max(worst_step, eb);
      if (std::abs(be) <= 3.0) worst_step_at_small = std::max(worst_step_at_small, eb);
    }
    app_log(2, "[TC-3-a fractional] VERDICT: the fractional sigma_m is exact to {:.3e} "
               "everywhere; the T = 0 step is wrong by up to {:.3e} overall and {:.3e} "
               "for |beta(eps-mu)| <= 3 -- i.e. the difference is entirely carried by the "
               "states within a few k_B T of mu, which is precisely the population a "
               "METAL has and an insulator does not (the qe_lih222 census reports ZERO "
               "strictly fractional sigma_J).",
            worst_frac, worst_step, worst_step_at_small);
    REQUIRE(worst_frac < 1e-13);
    REQUIRE(worst_step_at_small > 1e-2);   // the step form MUST be visibly wrong near mu
  }

  // =====================================================================
  //  THE RESIDUE ARGUMENT -- eq (EXACT) on a NON-SYMMETRIC pole set
  // =====================================================================
  //
  // The single-pole pin above uses a PH pair, which is exactly EVEN, so it
  // cannot distinguish W^c(eps_m - w) from W^c(w - eps_m). A fitted W^c is even
  // only on the imaginary axis (masked_pole_fit runs on a deliberately NONSYM
  // auxiliary node set -- wc_band_elements.hpp), so at real argument the two
  // differ by O(1). This pins the one eq (EXACT) requires.
  //
  TEST_CASE("tc_sigma_cd_nonsym_poles", "[methods][tc_wline]") {
    double worst_minus = 0.0, worst_plus = 0.0, worst_left = 0.0;
    long n = 0;
    // deterministic non-symmetric pole sets
    for (int c = 0; c < 6; ++c)
      for (int d = 0; d < 6; ++d)
        for (int e = 0; e < 3; ++e) {
          const long npk = 5;
          nda::array<double, 1> om(npk), wr(npk);
          nda::array<dcomplex, 1> w(npk);
          for (long j = 0; j < npk; ++j) {
            om(j) = -5.3 + 2.17 * double(j) + 0.31 * double(c);   // no +- pairing
            wr(j) = 0.4 + 0.9 * std::cos(1.7 * double(j) + 0.5 * double(d));
            w(j) = dcomplex(wr(j), 0.23 * std::sin(2.1 * double(j)));
          }
          const double beta = (e == 0) ? 1000.0 : (e == 1 ? 3000.0 : 10000.0);
          const double eps = -7.0 + 14.0 * double(c) / 5.0;
          const double omg = -7.0 + 14.0 * double(d) / 5.0;
          const double mu = 0.0, A = omg - eps;
          if (std::abs(A) < 0.3 or std::abs(eps - mu) < 0.3) continue;
          bool near = false;
          for (long j = 0; j < npk; ++j)
            if (std::abs(A + om(j)) < 0.3 or std::abs(A - om(j)) < 0.3) near = true;
          if (near) continue;

          // the EXACT finite-T closed form (sigma_route_b), any pole set
          const double f = scd::stable_nF(beta, eps - mu);
          dcomplex routeB(0.0, 0.0);
          double scale = 0.0;
          for (long j = 0; j < npk; ++j) {
            routeB += w(j) * (scd::stable_nB(beta, om(j)) + f) / (A + om(j));
            scale += std::abs(w(j) / (A + om(j)));
          }
          auto Wc = [&](dcomplex z) {
            dcomplex t(0.0, 0.0);
            for (long j = 0; j < npk; ++j) t += w(j) / (z - om(j));
            return t;
          };
          const dcomplex I = scd::imag_axis_term_poles(A, w, om);
          const double sg = scd::sigma_m_weight(omg, eps, mu, beta);
          worst_minus = std::max(worst_minus,
                                 std::abs(I + sg * Wc(dcomplex(-A, 0.0)) - routeB) / scale);
          worst_plus = std::max(worst_plus,
                                std::abs(I + sg * Wc(dcomplex(A, 0.0)) - routeB) / scale);
          worst_left = std::max(worst_left, scd::thermal_leftover_bound(beta, om));
          ++n;
        }
    app_log(2, "[TC-3-a nonsym] eq (EXACT) on {} NON-SYMMETRIC pole sets, deviation from "
               "the exact finite-T closed form normalized by sum |w/(A+om)| (which cannot "
               "cancel): W^c(eps_m - w) gives {:.3e}; W^c(w - eps_m) gives {:.3e}; "
               "bosonic leftover {:.3e}",
            n, worst_minus, worst_plus, worst_left);
    REQUIRE(n > 20);
    REQUIRE(worst_minus < 1e-13);      // the exact argument
    REQUIRE(worst_plus > 1e-2);        // the even-only argument MUST fail here
  }

  // =====================================================================
  //  W ON THE LINE -- CoQuI's Dyson chain vs the campaign's RPA W^c
  // =====================================================================
  TEST_CASE("tc_wc_line_dyson", "[methods][tc_wline]") {
    auto doc = mjson::load(ref_path());
    auto const &rpa = doc["rpa"];
    const long n = rpa["n_aux"].i();
    auto vsh = jshape(rpa["v"]);
    auto vdat = jdata(rpa["v"]);
    REQUIRE(vsh[0] == n);
    REQUIRE(vsh[1] == n);
    nda::array<dcomplex, 2> Z(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) Z(i, j) = vdat[std::size_t(i * n + j)];

    auto zsh = jshape(rpa["z"]);
    auto zdat = jdata(rpa["z"]);
    const long nz = zsh[0];
    auto Pdat = jdata(rpa["P"]);
    auto Wd = jdata(rpa["Wc_dense"]);
    auto Wp = jdata(rpa["Wc_poles"]);

    double worst_d = 0.0, worst_p = 0.0, wmax = 0.0;
    long worst_iz = -1;
    nda::array<dcomplex, 2> Pi(n, n);
    nda::matrix<dcomplex> W(n, n);
    wl::solve_stats_t st;
    for (long iz = 0; iz < nz; ++iz) {
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
          Pi(i, j) = Pdat[std::size_t((iz * n + i) * n + j)];
      wl::dyson_wc_line(Z, Pi, W, &st);
      double dd = 0.0, dp = 0.0, den = 0.0;
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) {
          const dcomplex rd = Wd[std::size_t((iz * n + i) * n + j)];
          const dcomplex rp = Wp[std::size_t((iz * n + i) * n + j)];
          dd = std::max(dd, std::abs(W(i, j) - rd));
          dp = std::max(dp, std::abs(W(i, j) - rp));
          den = std::max(den, std::abs(rd));
        }
      if (dd / den > worst_d) { worst_d = dd / den; worst_iz = iz; }
      worst_p = std::max(worst_p, dp / den);
      wmax = std::max(wmax, den);
    }
    app_log(2, "[TC-3-a W pin] CoQuI's Dyson chain ([I-Z.Pi]^-1 - I).Z on {} targets "
               "({} on the line Im z = {:.3g}, {} on the imaginary axis), n_aux = {}: "
               "vs the campaign's DENSE solve {:.3e} (worst target {}), vs its "
               "independent PLASMON POLE SUM {:.3e}; max|W^c| = {:.4g}; "
               "max |[I-Z.Pi]^-1| = {:.3e}",
            nz, rpa["n_line"].i(), rpa["delta"].d(), nz - rpa["n_line"].i(), n,
            worst_d, worst_iz, worst_p, wmax, st.cond_hint);
    app_log(2, "[TC-3-a W pin] the two REFERENCE routes agree to {:.3e} among "
               "themselves (exported); the plasmon structure sigma_min[I-vP] at w_p "
               "is {:.2e} of its value one spacing away",
            rpa["two_routes_max_rel"].d(),
            rpa["smin_at_wp"].vd()[0] / rpa["smin_off_wp"].vd()[0]);
    REQUIRE(worst_d < 1e-11);
    REQUIRE(worst_p < 1e-11);
  }

  // =====================================================================
  //  dense == warm-started GMRES on the contracted elements
  // =====================================================================
  TEST_CASE("tc_wc_line_krylov", "[methods][tc_wline]") {
    auto doc = mjson::load(ref_path());
    auto const &rpa = doc["rpa"];
    const long n = rpa["n_aux"].i();
    auto vdat = jdata(rpa["v"]);
    nda::array<dcomplex, 2> Z(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) Z(i, j) = vdat[std::size_t(i * n + j)];
    auto zsh = jshape(rpa["z"]);
    const long nz = zsh[0];
    auto Pdat = jdata(rpa["P"]);

    // a handful of band-pair vectors
    const long nv = 4;
    nda::array<dcomplex, 2> gl(n, nv), gr(n, nv);
    for (long v = 0; v < nv; ++v)
      for (long i = 0; i < n; ++i) {
        const double a = std::cos(0.7 * double(i) + 1.3 * double(v));
        const double b = std::sin(0.4 * double(i) - 0.9 * double(v));
        gl(i, v) = dcomplex(a, 0.2 * b);
        gr(i, v) = dcomplex(b, -0.1 * a);
      }

    nda::array<dcomplex, 2> Pi(n, n);
    nda::array<dcomplex, 1> od(nv), ok(nv);
    nda::array<dcomplex, 2> xw(n, nv);
    xw() = dcomplex(0.0, 0.0);
    wl::solve_opts_t dense_o, kry_o;
    kry_o.krylov = true;
    kry_o.tol = 1e-13;
    wl::solve_stats_t sd, sk;
    double worst = 0.0, den = 0.0;
    for (long iz = 0; iz < nz; ++iz) {
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
          Pi(i, j) = Pdat[std::size_t((iz * n + i) * n + j)];
      wl::wc_sandwich(Z, Pi, gl, gr, od, dense_o, sd, nullptr);
      wl::wc_sandwich(Z, Pi, gl, gr, ok, kry_o, sk, &xw);   // warm-started across iz
      for (long v = 0; v < nv; ++v) {
        worst = std::max(worst, std::abs(od(v) - ok(v)));
        den = std::max(den, std::abs(od(v)));
      }
    }
    app_log(2, "[TC-3-a Krylov] dense vs warm-started GMRES on <nm|W^c|mn>, {} targets "
               "x {} vectors: max abs dev {:.3e} over max|element| {:.4g} = {:.3e} rel; "
               "{} solves, {} total iterations (max {} per solve), worst achieved "
               "relative residual {:.2e}, {} warm starts",
            nz, nv, worst, den, worst / den, sk.n_solve, sk.n_iter, sk.n_iter_max,
            sk.resid_max, sk.n_warm);
    REQUIRE(worst / den < 1e-11);
  }

  // =====================================================================
  //  the eq-1 assembly on the campaign's full SigmaModel
  // =====================================================================
  TEST_CASE("tc_sigma_cd_multipole", "[methods][tc_wline]") {
    auto doc = mjson::load(ref_path());
    auto const &sg = doc["sigma"];
    const long na = sg["n_aux"].i();
    const long nm = sg["n_orb"].i();
    const double mu = sg["mu"].d();
    auto eps_v = sg["eps"].vd();
    auto om_v = sg["omegas"].vd();
    auto wp_v = sg["wp"].vd();
    const long np = long(wp_v.size());
    auto Rp = jdata(sg["Rp"]);        // (np, na, na)
    auto Cd = jdata(sg["C"]);         // (na, nm)

    nda::array<double, 1> eps(nm);
    for (long m = 0; m < nm; ++m) eps(m) = eps_v[std::size_t(m)];

    // <nm|W^c(z)|mn> from the model's plasmon pole sum:
    //   W^c(z) = sum_p R_p [ 1/(z-w_p) - 1/(z+w_p) ],   g_i = C_in C_im
    auto wc_band = [&](long nb, long m, dcomplex z) {
      dcomplex acc(0.0, 0.0);
      for (long p = 0; p < np; ++p) {
        const dcomplex t = 1.0 / (z - wp_v[std::size_t(p)])
                         - 1.0 / (z + wp_v[std::size_t(p)]);
        dcomplex q(0.0, 0.0);
        for (long i = 0; i < na; ++i) {
          const dcomplex gi = Cd[std::size_t(i * nm + nb)] * Cd[std::size_t(i * nm + m)];
          for (long j = 0; j < na; ++j) {
            const dcomplex gj = Cd[std::size_t(j * nm + nb)] * Cd[std::size_t(j * nm + m)];
            q += gi * Rp[std::size_t((p * na + i) * na + j)] * gj;
          }
        }
        acc += t * q;
      }
      return acc;
    };

    // the beta of the reference is T = 0 (models.SigmaModel splits at mu with a
    // step); reproduce that limit with a large beta so sigma_m -> the step.
    const double beta = 1e7;
    auto q = scd::tan_quadrature(2000, 50.0);
    const long nnu = q.nu.size();

    double worst = 0.0;
    auto bands = sg["bands"].vi();
    for (auto nb : bands) {
      auto exd = jdata(sg["exact_band"][std::to_string(nb)]);
      nda::array<dcomplex, 2> W_iv(nm, nnu);
      for (long m = 0; m < nm; ++m)
        for (long i = 0; i < nnu; ++i)
          W_iv(m, i) = wc_band(nb, m, dcomplex(0.0, q.nu(i)));
      for (std::size_t io = 0; io < om_v.size(); ++io) {
        const double w = om_v[io];
        nda::array<dcomplex, 1> W_line(nm);
        for (long m = 0; m < nm; ++m)
          W_line(m) = wc_band(nb, m, dcomplex(eps(m) - w, 0.0));  // eq (EXACT)
        const dcomplex got = scd::assemble(w, mu, beta, eps, q.nu, q.w, W_iv, W_line);
        const dcomplex ex = exd[io];
        worst = std::max(worst, std::abs(got - ex) / std::abs(ex));
      }
    }
    app_log(2, "[TC-3-a multipole] eq-1 assembly vs the EXACT Lehmann pole sum on the "
               "campaign's SigmaModel ({} plasmons x {} states, bands {}): max rel = "
               "{:.3e} at n_nu = {}; the campaign's own python value is {:.3e} at "
               "n_nu = 6000 (both quadrature-limited, not sign-limited)",
            np, nm, bands.size(), worst, nnu, sg["cd_vs_exact_max_rel"].d());
    REQUIRE(worst < 1e-6);
  }

} // namespace bdft_tests
