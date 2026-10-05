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
 * S9b: the bosonic response closure (numerics/line_dlr/response_closure.hpp) vs the python design study
 * (scripts/closure_design/; reference tests/unit_test_files/gw_line/response_closure_ref.h5, written by
 * coqui/cayley/scripts/gen_response_closure_ref.py). Cases: W_c projections (si211 SC q0 const, lih222 SC q1 rand), the
 * Si-like loss model (sieps|h) and the synthetic continuum, noise 0 and 1e-8 (seed 1000), theta 20 deg (sieps|h also 10 deg).
 *
 * [nnls]    Lawson-Hanson on small problems: the scipy doc examples (exact), random problems: KKT (x >= 0, dual <= 1e-10,
 *           complementarity), = unconstrained LS when it is positive.
 * [rules]   K rule and K cap at 20 / 10 / 5 deg (the study's numbers 48 / 48 / 45 / 32; 91 at 10 deg, 1e-8), tol_gram rule,
 *           partition of unity.
 * [python]  per case: (a) the odd fit as a function at the nodes vs python's <= 1e-9 (relative), its residual;
 *           (b) MB from PYTHON's fit residues (deterministic part): per-scale poles / weights (<= 2e-7; measured 2e-9..7e-8)
 *           and the blended function at eta = 0.01 Ha and 0.05 omega (<= 2e-6; measured 7e-8..6e-7) vs python, relative to max
 *           (the K 45-48 closures amplify the LAPACK roundoff of the Gram / Schur steps ~1e7 x);
 *           (c) NNLS: residual norm (<= 1e-6 rel; identical to 7 digits), number of positive poles (equal) and the function
 *           (<= 1e-5; measured 3e-8..3e-6) vs python when the data carry noise; noise-free the NNLS residual is at roundoff and
 *           its positive solution is not unique numerically (residual only);
 *           (d) the full C++ pipeline (own fit) vs the exact measure: window errors (0-5 / 5-10 / 10-20 / 20-40 eV) within
 *           20% of python's (or below 2x python's + 1e-6 when python's is tiny).
 * [optics]  Si-like model: routes R1 (close h) / R2 (close m = h/(1+h)) with MB and G: eps1, eps2, n, kappa, alpha, R, loss
 *           vs python (<= 2e-6 from python's fits; measured 1e-7), errors vs exact within 20%; static / f-sum from the fit.
 */

#undef NDEBUG

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"
#include "numerics/line_dlr/response_closure.hpp"

namespace bdft_tests {

namespace rc = numerics::line_dlr::response;
using dcomplex = std::complex<double>;

namespace {

const std::string ref_file = std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/response_closure_ref.h5";
constexpr double deg = std::numbers::pi / 180.0;
constexpr double HA  = rc::HA_EV;

template <typename T> T read_attr(h5::group &g, std::string const &name) {
  T v{};
  h5::h5_read_attribute(g, name, v);
  return v;
}
nda::array<double, 1> rd1(h5::group &g, std::string const &n) {
  nda::array<double, 1> a;
  nda::h5_read(g, n, a);
  return a;
}
nda::array<dcomplex, 1> rdc(h5::group &g, std::string const &n) {
  auto re = rd1(g, n + "_re"), im = rd1(g, n + "_im");
  nda::array<dcomplex, 1> c(re.size());
  for (long i = 0; i < re.size(); ++i) c(i) = dcomplex(re(i), im(i));
  return c;
}
double rel_maxdiff(auto const &a, auto const &b) {
  double d = 0.0, s = 0.0;
  for (long i = 0; i < a.size(); ++i) {
    d = std::max(d, std::abs(a(i) - b(i)));
    s = std::max(s, std::abs(b(i)));
  }
  return s > 0.0 ? d / s : d;
}

struct basis_ref_t {
  nda::array<double, 1> nu;
  nda::array<dcomplex, 1> zeta;
  double theta = 0.0;
};
basis_ref_t read_basis(h5::group &f, std::string const &n) {
  auto g = f.open_group(n);
  basis_ref_t b;
  b.nu    = rd1(g, "nu");
  b.zeta  = rdc(g, "zeta");
  b.theta = read_attr<double>(g, "theta");
  return b;
}

/// window errors (cd_lib.window_errors): per window max|fv - fe| / max|fe| and the same for Im (loss)
std::array<std::array<double, 2>, 4> window_errors(nda::array<double, 1> const &om, nda::array<dcomplex, 1> const &fv,
                                                    nda::array<dcomplex, 1> const &fe) {
  const double W[4][2] = {{0, 5}, {5, 10}, {10, 20}, {20, 40}};
  std::array<std::array<double, 2>, 4> e{};
  for (int w = 0; w < 4; ++w) {
    double d0 = 0, s0 = 0, d1 = 0, s1 = 0;
    for (long i = 0; i < om.size(); ++i) {
      const double x = om(i) * HA;
      if (x <= W[w][0] or x > W[w][1]) continue;
      d0 = std::max(d0, std::abs(fv(i) - fe(i)));
      s0 = std::max(s0, std::abs(fe(i)));
      d1 = std::max(d1, std::abs(fv(i).imag() - fe(i).imag()));
      s1 = std::max(s1, std::abs(fe(i).imag()));
    }
    e[w] = {d0 / s0, d1 / s1};
  }
  return e;
}

/// "within 20%" of a reference error with a floor for tiny ones
bool close_err(double c, double p) { return std::abs(c - p) <= 0.2 * p or (p < 1e-4 and c < 2.0 * p + 1e-6); }

} // namespace

TEST_CASE("response_closure_nnls", "[response_closure][nnls]") {
  // scipy doc examples
  {
    nda::matrix<double, nda::F_layout> A(3, 2);
    A() = 0.0;
    A(0, 0) = 1; A(1, 0) = 1; A(2, 1) = 1;
    nda::array<double, 1> b = {2, 1, 1};
    auto r = rc::nnls(A, b);
    REQUIRE(std::abs(r.x(0) - 1.5) < 1e-14);
    REQUIRE(std::abs(r.x(1) - 1.0) < 1e-14);
    REQUIRE(std::abs(r.rnorm - std::sqrt(0.5)) < 1e-14);
    nda::array<double, 1> b2 = {-1, -1, -1};
    auto r2 = rc::nnls(A, b2);
    REQUIRE(r2.x(0) == 0.0);
    REQUIRE(r2.x(1) == 0.0);
    REQUIRE(std::abs(r2.rnorm - std::sqrt(3.0)) < 1e-14);
  }
  // random problems: KKT conditions
  std::mt19937 gen(4321);
  auto U = [&]() { return 2.0 * (double(gen()) / 4294967296.0) - 1.0; };
  for (int t = 0; t < 20; ++t) {
    const long m = 15 + 7 * t, n = (t % 2 == 0) ? m / 2 : 3 * m;
    nda::matrix<double, nda::F_layout> A(m, n);
    nda::array<double, 1> b(m);
    for (long j = 0; j < n; ++j)
      for (long i = 0; i < m; ++i) A(i, j) = U();
    for (long i = 0; i < m; ++i) b(i) = U();
    auto r = rc::nnls(A, b);
    REQUIRE(r.mode == 1);
    nda::array<double, 1> res(m);
    for (long i = 0; i < m; ++i) {
      double s = b(i);
      for (long j = 0; j < n; ++j) s -= A(i, j) * r.x(j);
      res(i) = s;
    }
    double rn = 0.0, dmax = 0.0, comp = 0.0, xmin = 0.0;
    for (long i = 0; i < m; ++i) rn += res(i) * res(i);
    for (long j = 0; j < n; ++j) {
      double w = 0.0;
      for (long i = 0; i < m; ++i) w += A(i, j) * res(i);
      xmin = std::min(xmin, r.x(j));
      if (r.x(j) > 0.0) comp = std::max(comp, std::abs(w));
      else dmax = std::max(dmax, w);
    }
    INFO("t " << t << " m " << m << " n " << n);
    REQUIRE(xmin >= 0.0);
    REQUIRE(std::abs(std::sqrt(rn) - r.rnorm) < 1e-10);
    REQUIRE(dmax < 1e-10);
    REQUIRE(comp < 1e-10);
  }
}

TEST_CASE("response_closure_rules", "[response_closure][rules]") {
  REQUIRE(rc::k_rule(0.0, 20 * deg) == 48);
  REQUIRE(rc::k_rule(1e-10, 20 * deg) == 48);
  REQUIRE(rc::k_rule(1e-8, 20 * deg) == 45);
  REQUIRE(rc::k_rule(1e-6, 20 * deg) == 32);
  REQUIRE(rc::k_rule(1e-8, 10 * deg) == 91);
  REQUIRE(rc::k_cap(20 * deg) == 48);
  REQUIRE(rc::k_cap(10 * deg) == 97);
  app_log(2, "  [rules] K cap 20 / 10 / 5 deg: {} / {} / {}; K(1e-8) {} / {} / {}", rc::k_cap(20 * deg), rc::k_cap(10 * deg),
          rc::k_cap(5 * deg), rc::k_rule(1e-8, 20 * deg), rc::k_rule(1e-8, 10 * deg), rc::k_rule(1e-8, 5 * deg));
  REQUIRE(rc::tol_gram_rule(0.0) == 1e-10);
  REQUIRE(rc::tol_gram_rule(1e-8) == 1e-7);
  auto sc = rc::log_scales(3.3 / HA, 27.2 / HA, 4);
  auto x  = numerics::line_dlr::detail::logspace(1e-3, 5.0, 400);
  auto P  = rc::partition_weights(x, sc);
  double e = 0.0;
  for (long k = 0; k < x.size(); ++k) {
    double s = 0.0;
    for (long i = 0; i < 4; ++i) {
      REQUIRE(P(i, k) >= 0.0);
      s += P(i, k);
    }
    e = std::max(e, std::abs(s - 1.0));
  }
  REQUIRE(e < 1e-14);
}

TEST_CASE("response_closure_python", "[response_closure][python]") {
  h5::file f(ref_file, 'r');
  h5::group g0(f);
  auto b20 = read_basis(g0, "basis20"), b10 = read_basis(g0, "basis10");
  auto om = rd1(g0, "omega"), grid = rd1(g0, "grid_G"), scv = rd1(g0, "scales");
  std::vector<double> scales(scv.begin(), scv.end());
  const long nc = read_attr<long>(g0, "ncases");
  nda::array<double, 1> omc(om.size() / 5);
  for (long i = 0; i < omc.size(); ++i) omc(i) = om(5 * i);
  rc::closure_opts_t parity;
  parity.drop_nonpositive = false;
  int n_err = 0, n_err_ok = 0;
  for (long ic = 0; ic < nc; ++ic) {
    auto g = g0.open_group("case" + std::to_string(ic));
    const auto name  = read_attr<std::string>(g, "name");
    const double dl  = read_attr<double>(g, "delta"), th = read_attr<double>(g, "theta");
    const long K     = read_attr<long>(g, "K");
    const double tg  = read_attr<double>(g, "tol_gram");
    auto const &b    = (th > 15.0) ? b20 : b10;
    auto data = rdc(g, "data");
    auto fj_py = rd1(g, "fj");
    rc::odd_measure_t ex(rd1(g, "W"), rd1(g, "r"));
    INFO("case " << ic << " " << name << " delta " << dl << " theta " << th);
    REQUIRE(K == rc::k_rule(dl, th * deg));
    REQUIRE(tg == rc::tol_gram_rule(dl));
    // (a) odd fit
    auto fj = rc::odd_fit(b.zeta, data, b.nu);
    rc::odd_measure_t mf(b.nu, fj), mpy(b.nu, fj_py);
    const double dfit = rel_maxdiff(mf(b.zeta), mpy(b.zeta));
    const double res = rc::fit_residual(b.zeta, data, b.nu, fj), res_py = read_attr<double>(g, "fit_resid");
    app_log(2, "  [{}] {:<20s} delta {:.0e} theta {:.0f}: odd fit vs python {:.1e} (resid {:.2e} / py {:.2e}), K {}", ic, name, dl,
            th, dfit, res, res_py, K);
    REQUIRE(dfit < 1e-9);
    REQUIRE(std::abs(res - res_py) <= 0.5 * res_py + 1e-13);
    // (b) MB from python's residues
    auto mb_py = rc::mb_close(b.nu, fj_py, scales, K, tg, parity);
    double dpole = 0.0;
    for (long i = 0; i < long(scales.size()); ++i) {
      auto d = rd1(g, "mb_d" + std::to_string(i)), a = rd1(g, "mb_a" + std::to_string(i));
      REQUIRE(d.size() == mb_py.parts[i].d.size());
      double sa = 0.0;
      for (long l = 0; l < a.size(); ++l) sa += a(l);
      for (long l = 0; l < d.size(); ++l)
        dpole = std::max(dpole, std::abs(mb_py.parts[i].a(l) - a(l)) / sa +
                                    std::abs(mb_py.parts[i].a(l)) / sa * std::abs(mb_py.parts[i].d(l) - d(l)) / std::abs(d(l)));
    }
    double dmb = 0.0;
    for (auto [lab, rel] : {std::pair{"e01", false}, std::pair{"r05", true}}) {
      nda::array<dcomplex, 1> z(omc.size());
      for (long i = 0; i < omc.size(); ++i) z(i) = dcomplex(omc(i), rel ? 0.05 * omc(i) : 0.01);
      dmb = std::max(dmb, rel_maxdiff(mb_py(z), rdc(g, std::string("mb_") + lab)));
    }
    // (c) NNLS
    auto G = rc::nnls_odd_fit(b.zeta, data, grid);
    const double rn_py = read_attr<double>(g, "G_rnorm");
    const long np_py   = read_attr<long>(g, "G_npos");
    double dG = 0.0;
    for (auto [lab, rel] : {std::pair{"e01", false}, std::pair{"r05", true}}) {
      nda::array<dcomplex, 1> z(omc.size());
      for (long i = 0; i < omc.size(); ++i) z(i) = dcomplex(omc(i), rel ? 0.05 * omc(i) : 0.01);
      dG = std::max(dG, rel_maxdiff(G.measure(z), rdc(g, std::string("G_") + lab)));
    }
    app_log(2, "       MB(python fit) vs python: poles/weights {:.1e}, function {:.1e}; NNLS: rnorm {:.6e} / py {:.6e}, {} / {} "
               "poles, function {:.1e} ({} iterations)",
            dpole, dmb, G.rnorm, rn_py, G.measure.size(), np_py, dG, G.iter);
    // gates (measured on the Mac vs numpy/scipy: poles 2e-9..7e-8, function 7e-8..6e-7; the K 45-48 closures amplify the
    // LAPACK roundoff of the Gram / Schur steps ~1e7 x; NNLS identical to 7+ digits when the data carry noise; at delta = 0
    // the NNLS residual is at roundoff (1e-14) and the positive solution is not unique numerically: residual only)
    double bnorm = 0.0;
    for (long i = 0; i < data.size(); ++i) bnorm += std::norm(data(i));
    bnorm = std::sqrt(bnorm);
    REQUIRE(dpole < 2e-7);
    REQUIRE(dmb < 2e-6);
    REQUIRE(std::abs(G.rnorm - rn_py) <= 1e-6 * rn_py + 1e-13 * bnorm);
    if (dl > 0.0) {
      REQUIRE(G.measure.size() == np_py);
      REQUIRE(dG < 1e-5);
    }
    // (d) the C++ pipeline (own fit) vs exact: window errors vs python's
    auto mb = rc::mb_close(b.nu, fj, scales, K, tg, parity);
    for (auto [lab, rel] : {std::pair{"e01", false}, std::pair{"r05", true}}) {
      nda::array<dcomplex, 1> z(om.size());
      for (long i = 0; i < om.size(); ++i) z(i) = dcomplex(om(i), rel ? 0.05 * om(i) : 0.01);
      auto fe = ex(z);
      auto em = window_errors(om, mb(z), fe), eg = window_errors(om, G.measure(z), fe);
      nda::array<double, 2> pm, pg;
      nda::h5_read(g, std::string("err_mb_") + lab, pm);
      nda::h5_read(g, std::string("err_G_") + lab, pg);
      std::string s;
      for (int w = 0; w < 4; ++w) {
        s += std::format(" {:.2e}/{:.2e}", em[w][0], pm(w, 0));
        for (int c = 0; c < 2; ++c) {
          INFO("eta " << lab << " window " << w << " comp " << c << " MB " << em[w][c] << " py " << pm(w, c) << " G " << eg[w][c]
                      << " py " << pg(w, c));
          n_err += 1;
          n_err_ok += close_err(em[w][c], pm(w, c));
          CHECK(close_err(em[w][c], pm(w, c)));
          if (dl > 0.0) {   // noise-free: the NNLS solution is not unique at roundoff (see above)
            n_err += 1;
            n_err_ok += close_err(eg[w][c], pg(w, c));
            CHECK(close_err(eg[w][c], pg(w, c)));
          }
        }
      }
      app_log(2, "       C++ MB window errors (C++/py) eta {}:{}", lab, s);
    }
    // statics from the fit
    REQUIRE(std::abs(mf.fsum() - read_attr<double>(g, "fsum_fit")) < 1e-6 * std::abs(read_attr<double>(g, "fsum_fit")));
    REQUIRE(std::abs(mf.static_value() - read_attr<double>(g, "f0_fit")) < 1e-8 * std::abs(read_attr<double>(g, "f0_fit")));
  }
  app_log(2, "  [python] window errors within 20% of python: {} / {}", n_err_ok, n_err);
}

TEST_CASE("response_closure_optics", "[response_closure][optics]") {
  h5::file f(ref_file, 'r');
  h5::group g0(f);
  auto b   = read_basis(g0, "basis20");
  auto om  = rd1(g0, "omega"), grid = rd1(g0, "grid_G"), scv = rd1(g0, "scales");
  std::vector<double> scales(scv.begin(), scv.end());
  auto g   = g0.open_group("optics");
  const long K   = read_attr<long>(g, "K");
  const double tg = read_attr<double>(g, "tol_gram");
  auto hd = rdc(g, "h"), md = rdc(g, "m");
  auto fh_py = rd1(g, "fh"), fm_py = rd1(g, "fm");
  nda::array<double, 1> omc(om.size() / 5);
  for (long i = 0; i < omc.size(); ++i) omc(i) = om(5 * i);
  rc::closure_opts_t parity;
  parity.drop_nonpositive = false;
  const std::vector<std::string> qn = {"eps1", "eps2", "n", "kappa", "alpha", "R", "loss"};
  auto zgrid = [](nda::array<double, 1> const &w) {
    nda::array<dcomplex, 1> z(w.size());
    for (long i = 0; i < w.size(); ++i) z(i) = dcomplex(w(i), 0.05 * w(i));
    return z;
  };
  auto quantities = [&](nda::array<dcomplex, 1> const &hcl, bool route1, nda::array<double, 1> const &w) {
    nda::array<dcomplex, 1> eps(hcl.size());
    for (long i = 0; i < hcl.size(); ++i) eps(i) = route1 ? 1.0 / (1.0 + hcl(i)) : 1.0 - hcl(i);
    return rc::optical_quantities(eps, w);
  };
  // (1) deterministic: from python's fits (MB) and the data (G), coarse grid
  {
    auto zc  = zgrid(omc);
    auto mh  = rc::mb_close(b.nu, fh_py, scales, K, tg, parity), mm = rc::mb_close(b.nu, fm_py, scales, K, tg, parity);
    auto Gh  = rc::nnls_odd_fit(b.zeta, hd, grid), Gm = rc::nnls_odd_fit(b.zeta, md, grid);
    double dmax = 0.0;
    for (auto [route, meth] : {std::pair{"R1", "mb"}, std::pair{"R2", "mb"}, std::pair{"R1", "G"}, std::pair{"R2", "G"}}) {
      const bool r1 = std::string(route) == "R1", isG = std::string(meth) == "G";
      auto fv = r1 ? (isG ? Gh.measure(zc) : mh(zc)) : (isG ? Gm.measure(zc) : mm(zc));
      auto Q  = quantities(fv, r1, omc);
      for (auto const &q : qn) {
        const double d = rel_maxdiff(Q.get(q), rd1(g, std::string(route) + "_" + meth + "_" + q));
        INFO(route << " " << meth << " " << q << " " << d);
        CHECK(d < 2e-6);   // measured <= 1.0e-7: the derived quantities amplify the function difference (1/|eps|^2 ~ 24)
        dmax = std::max(dmax, d);
      }
    }
    app_log(2, "  [optics] Si-like model, eta 0.05 omega: quantities from python's fits / the data vs python: max {:.1e}", dmax);
  }
  // (2) the C++ pipeline vs exact: errors within 20% of python's
  {
    auto z  = zgrid(om);
    auto fh = rc::odd_fit(b.zeta, hd, b.nu), fm = rc::odd_fit(b.zeta, md, b.nu);
    auto mh = rc::mb_close(b.nu, fh, scales, K, tg, parity), mm = rc::mb_close(b.nu, fm, scales, K, tg, parity);
    auto Gh = rc::nnls_odd_fit(b.zeta, hd, grid), Gm = rc::nnls_odd_fit(b.zeta, md, grid);
    nda::array<dcomplex, 1> hex(om.size());
    rc::odd_measure_t ex;
    {
      // exact sieps|h: the case of the reference with that name
      const long nc = read_attr<long>(g0, "ncases");
      for (long ic = 0; ic < nc; ++ic) {
        auto gc = g0.open_group("case" + std::to_string(ic));
        if (read_attr<std::string>(gc, "name") == "sieps|h") {
          ex = rc::odd_measure_t(rd1(gc, "W"), rd1(gc, "r"));
          break;
        }
      }
    }
    const double cfac = read_attr<double>(g, "cfac");
    auto hx = ex(z);
    for (long i = 0; i < hx.size(); ++i) hx(i) *= cfac;
    auto Qx = quantities(hx, true, om);
    const double W[4][2] = {{0, 5}, {5, 10}, {10, 20}, {20, 40}};
    int ok = 0, tot = 0;
    for (auto [route, meth] : {std::pair{"R1", "mb"}, std::pair{"R2", "mb"}, std::pair{"R1", "G"}, std::pair{"R2", "G"}}) {
      const bool r1 = std::string(route) == "R1", isG = std::string(meth) == "G";
      auto fv = r1 ? (isG ? Gh.measure(z) : mh(z)) : (isG ? Gm.measure(z) : mm(z));
      auto Q  = quantities(fv, r1, om);
      nda::array<double, 2> E;
      nda::h5_read(g, std::string("err_") + route + "_" + meth, E);
      std::string s;
      for (long iq = 0; iq < long(qn.size()); ++iq) {
        auto const &a = Q.get(qn[iq]);
        auto const &x = Qx.get(qn[iq]);
        for (int w = 0; w < 4; ++w) {
          double d = 0, sx = 0;
          for (long i = 0; i < om.size(); ++i) {
            const double e = om(i) * HA;
            if (e <= W[w][0] or e > W[w][1]) continue;
            d  = std::max(d, std::abs(a(i) - x(i)));
            sx = std::max(sx, std::abs(x(i)));
          }
          const double e = d / sx;
          ++tot;
          ok += close_err(e, E(iq, w));
          INFO(route << " " << meth << " " << qn[iq] << " window " << w << ": " << e << " py " << E(iq, w));
          CHECK(close_err(e, E(iq, w)));
          if (qn[iq] == "eps2" or qn[iq] == "loss") s += std::format(" {}:{:.2e}/{:.2e}", qn[iq], e, E(iq, w));
        }
      }
      app_log(2, "  [optics] {} {} vs exact (C++/py):{}", route, meth, s);
    }
    app_log(2, "  [optics] errors within 20% of python: {} / {}", ok, tot);
    // statics and f-sum from the fits: eps_inf both routes, f-sum of h and m (both omega_p^2)
    rc::odd_measure_t mfh(b.nu, fh), mfm(b.nu, fm);
    const double einf1 = 1.0 / (1.0 + mfh.static_value()), einf2 = 1.0 - mfm.static_value();
    app_log(2, "  [optics] eps_inf from h {:.8f}, from m {:.8f} (exact 12); f-sum h {:.6e}, m {:.6e}, exact {:.6e}", einf1, einf2,
            mfh.fsum(), mfm.fsum(), cfac * ex.fsum());
    REQUIRE(std::abs(einf1 - 12.0) < 1e-5 * 12.0);
    REQUIRE(std::abs(einf2 - 12.0) < 1e-5 * 12.0);
    REQUIRE(std::abs(mfh.fsum() - cfac * ex.fsum()) < 1e-4 * cfac * ex.fsum());
  }
}

} // namespace bdft_tests
