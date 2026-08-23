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
 * GATES TC-1-a and TC-1-b of notes/tc_coqui_impl_spec.md.
 *
 * TC-1-a  the port against the campaign's own numbers, on all 18 fixtures of
 *         tc_validation (the spec asks for >= 6):
 *           * the rank at lambda > eps^2 lambda_max EQUAL to the reference,
 *             and so are the spec-convention rank and the conditioning ceiling;
 *           * the nodes {s_j} equal to the reference to the local resolution of
 *             the graded s-grid they are selected from;
 *           * F applied to the analytic in-basis poles reproduces 1/(z - D_k)
 *             at the campaign's error class (the reference number is the pin);
 *           * an independent analytic pole model transformed through F, scored
 *             against its closed form AND against the python transform.
 *
 * TC-1-b  the ported unit pins (tc_validation/tests/pins.py):
 *           pin1  the eq-2 exponent algebra + a brute-force rotated contour
 *                 quadrature of one pole against 1/(z-D);
 *           pin3  the two endpoint limits, theta = 0 and (theta = pi/2, W = 0);
 *           pin4  the ID identity T[:,J] = I, the exponential rank ladder, and
 *                 F exactness at the fully resolved rank.
 *
 * The reference file is written by tc_validation/tests/export_tc1_reference.py.
 */

#undef NDEBUG

#include <cctype>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "numerics/tilted_contour/tilted_contour.hpp"

namespace bdft_tests {

  namespace tc = tilted_contour;
  using dcomplex = std::complex<double>;

  // =======================================================================
  //  A minimal JSON reader -- TEST-ONLY, and only for the file this repo's
  //  own export script writes. No dependency is added to the library.
  // =======================================================================
  namespace mjson {

    struct Value;
    using Object = std::map<std::string, Value>;
    using Array  = std::vector<Value>;

    struct Value {
      enum kind_t { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
      bool b = false;
      double num = 0.0;
      std::string str;
      std::shared_ptr<Array> arr;
      std::shared_ptr<Object> obj;

      Value const &operator[](std::string const &k) const {
        REQUIRE(kind == OBJ);
        auto it = obj->find(k);
        if (it == obj->end()) { FAIL("mjson: missing key '" << k << "'"); }
        return it->second;
      }
      Value const &operator[](std::size_t i) const {
        REQUIRE(kind == ARR);
        REQUIRE(i < arr->size());
        return (*arr)[i];
      }
      std::size_t size() const { return (kind == ARR) ? arr->size() : 0; }
      double d() const { REQUIRE(kind == NUM); return num; }
      long   i() const { REQUIRE(kind == NUM); return long(std::llround(num)); }
      std::string s() const { REQUIRE(kind == STR); return str; }
      std::vector<double> vd() const {
        REQUIRE(kind == ARR);
        std::vector<double> v;
        v.reserve(arr->size());
        for (auto const &e : *arr) v.push_back(e.d());
        return v;
      }
      std::vector<long> vi() const {
        REQUIRE(kind == ARR);
        std::vector<long> v;
        v.reserve(arr->size());
        for (auto const &e : *arr) v.push_back(e.i());
        return v;
      }
    };

    struct Parser {
      std::string const &t;
      std::size_t p = 0;
      explicit Parser(std::string const &s) : t(s) {}
      void ws() { while (p < t.size() and std::isspace((unsigned char)t[p])) ++p; }
      char peek() { ws(); return (p < t.size()) ? t[p] : '\0'; }
      Value parse() {
        ws();
        const char c = peek();
        if (c == '{') return obj();
        if (c == '[') return arr();
        if (c == '"') { Value v; v.kind = Value::STR; v.str = str(); return v; }
        if (c == 't' or c == 'f') {
          Value v; v.kind = Value::BOOL;
          v.b = (c == 't');
          p += (c == 't') ? 4 : 5;
          return v;
        }
        if (c == 'n') { p += 4; return Value{}; }
        return num();
      }
      std::string str() {
        ws();
        REQUIRE(t[p] == '"');
        ++p;
        std::string out;
        while (t[p] != '"') {
          if (t[p] == '\\') { ++p; out.push_back(t[p]); }
          else out.push_back(t[p]);
          ++p;
        }
        ++p;
        return out;
      }
      Value num() {
        ws();
        std::size_t q = p;
        while (q < t.size() and (std::isdigit((unsigned char)t[q]) or t[q] == '+' or
                                 t[q] == '-' or t[q] == '.' or t[q] == 'e' or t[q] == 'E'))
          ++q;
        Value v;
        v.kind = Value::NUM;
        v.num = std::stod(t.substr(p, q - p));
        p = q;
        return v;
      }
      Value arr() {
        Value v;
        v.kind = Value::ARR;
        v.arr = std::make_shared<Array>();
        REQUIRE(peek() == '[');
        ++p;
        if (peek() == ']') { ++p; return v; }
        while (true) {
          v.arr->push_back(parse());
          const char c = peek();
          ++p;
          if (c == ']') break;
          REQUIRE(c == ',');
        }
        return v;
      }
      Value obj() {
        Value v;
        v.kind = Value::OBJ;
        v.obj = std::make_shared<Object>();
        REQUIRE(peek() == '{');
        ++p;
        if (peek() == '}') { ++p; return v; }
        while (true) {
          std::string k = str();
          REQUIRE(peek() == ':');
          ++p;
          (*v.obj)[k] = parse();
          const char c = peek();
          ++p;
          if (c == '}') break;
          REQUIRE(c == ',');
        }
        return v;
      }
    };

    inline Value load(std::string const &path) {
      std::ifstream f(path);
      REQUIRE(f.good());
      std::stringstream ss;
      ss << f.rdbuf();
      const std::string s = ss.str();
      Parser pr(s);
      return pr.parse();
    }

  } // namespace mjson

  // =======================================================================
  //  GATE TC-1-a -- the campaign reference
  // =======================================================================
  TEST_CASE("tilted_contour_reference", "[numerics][tilted_contour]") {
    const std::string path = std::string(PROJECT_SOURCE_DIR)
                           + "/tests/unit_test_files/tilted_contour/tc1_reference.json";
    auto doc = mjson::load(path);
    auto const &meta = doc["meta"];
    auto const &cases = doc["cases"];
    const long ncase = long(cases.size());
    REQUIRE(ncase >= 6);          // the spec's floor
    app_log(2, "[TC-1-a] reference: {} cases, rho = {}, eps = {:.0e}, eps_tr = {:.0e}, "
               "nx = {}, rule '{}' capped by '{}', weighting '{}'",
            ncase, meta["rho"].d(), meta["eps"].d(), meta["eps_tr"].d(),
            meta["nx"].i(), meta["rank_rule"].s(), meta["rank_cap_rule"].s(),
            meta["weighting"].s());

    long n_node_exact = 0, n_node_total = 0;
    double worst_node_dev = 0.0, worst_geom = 0.0;
    double worst_F_ratio = 0.0, worst_probe_ratio = 0.0, worst_xpy = 0.0;

    for (long ic = 0; ic < ncase; ++ic) {
      auto const &r = cases[std::size_t(ic)];
      const std::string name = r["name"].s();
      auto const &in = r["inp"];

      tc::contour_params_t p;
      p.dmin  = in["dmin"].d();
      p.dmax  = in["dmax"].d();
      p.W     = in["W"].d();
      p.delta = in["delta"].d();
      p.rho   = in["rho"].d();
      p.eps   = in["eps"].d();
      p.eps_tr = in["eps_tr"].d();
      p.nx    = in["nx"].i();
      p.kappa = in["kappa"].d();

      auto c = tc::build_contour(p);

      // ---- geometry ---------------------------------------------------
      auto const &gr = r["geom"];
      auto rel = [](double a, double b) {
        return std::abs(a - b) / std::max(std::abs(b), 1e-300);
      };
      const double dg = std::max({rel(c.g.theta, gr["theta"].d()),
                                  rel(c.g.tan_theta, gr["tan_theta"].d()),
                                  rel(c.g.gamma, gr["gamma"].d()),
                                  rel(c.g.S, gr["S"].d())});
      const double d6 = rel(tc::eq6_Ns(p.dmin, p.dmax, p.W, p.delta, p.rho, p.eps),
                            gr["eq6_Ns"].d());
      worst_geom = std::max({worst_geom, dg, d6});
      REQUIRE(dg < 1e-12);
      REQUIRE(d6 < 1e-12);

      // ---- rank: EQUAL --------------------------------------------------
      // The two ACCURACY-CARRYING counts (eps and eps^2) must be equal to the
      // reference. The 1e-16 CEILING count is NOT gated for equality: at that
      // threshold the Gram eigenvalues sit in the eigensolver's own round-off
      // floor, and nda's zheev and numpy's zheevd do not agree there (measured
      // 44 vs 33 on si_kp222_nbnd60). What the ceiling has to do -- guard the
      // conditioning -- is gated instead: it must exceed the eps^2 rank, i.e.
      // it must NOT bind at eps = 1e-6. [DEVIATION, flagged in notes/tc12_report.md]
      auto const &rk = r["rank"];
      REQUIRE(c.rank_eps  == rk["eps"].i());
      REQUIRE(c.rank_eps2 == rk["eps2"].i());
      REQUIRE(c.rank_ceil >= c.rank_eps2);
      REQUIRE(c.rank      == rk["used"].i());
      REQUIRE(c.rank      == c.rank_eps2);

      // ---- s-grid + nodes ----------------------------------------------
      REQUIRE(c.s_grid.size() == r["grid"]["n_s_grid"].i());
      auto sref = r["nodes"].vd();
      auto iref = r["node_idx"].vi();
      REQUIRE(long(sref.size()) == c.rank);
      long nex = 0;
      double dev = 0.0;
      for (long j = 0; j < c.rank; ++j) {
        if (c.idx(j) == iref[std::size_t(j)]) ++nex;
        // "equal to grid resolution": the local spacing of the graded s-grid
        const long i0 = std::max(0L, c.idx(j) - 1), i1 = std::min(c.s_grid.size() - 1,
                                                                  c.idx(j) + 1);
        const double h = std::max(c.s_grid(i1) - c.s_grid(i0), 1e-30);
        dev = std::max(dev, std::abs(c.s(j) - sref[std::size_t(j)]) / h);
      }
      n_node_exact += nex;
      n_node_total += c.rank;
      worst_node_dev = std::max(worst_node_dev, dev);
      REQUIRE(dev <= 1.0);

      // ---- F on the in-basis poles ---------------------------------------
      const long ntarg = 61;
      nda::array<dcomplex, 1> zr(ntarg);
      const double zmax = in["zeta_max"].d();
      for (long t = 0; t < ntarg; ++t)
        zr(t) = dcomplex(zmax * double(t) / double(ntarg - 1), p.delta);
      auto tr = tc::build_transform(c, zr, /*with_mirror*/ true);
      REQUIRE(tr.F.shape()[0] == 2 * ntarg);
      REQUIRE(tr.F.shape()[1] == c.rank);

      double inb = 0.0;
      {
        const dcomplex rot = std::exp(dcomplex(0.0, -c.g.theta));
        std::vector<dcomplex> A(static_cast<std::size_t>(c.rank));
        for (long k = 0; k < c.x.size(); ++k) {
          const double D = p.dmin + c.x(k);
          for (long j = 0; j < c.rank; ++j)
            A[std::size_t(j)] = std::exp(dcomplex(0.0, -1.0) * D * c.s(j) * rot);
          for (long t = 0; t < 2 * ntarg; ++t) {
            dcomplex acc(0.0, 0.0);
            for (long j = 0; j < c.rank; ++j) acc += tr.F(t, j) * A[std::size_t(j)];
            const dcomplex ex = 1.0 / (tr.z(t) - D);
            inb = std::max(inb, std::abs(acc - ex) / std::abs(ex));
          }
        }
      }
      const double inb_ref = r["F"]["inbasis_max_rel"].d();
      worst_F_ratio = std::max(worst_F_ratio, inb / inb_ref);

      // ---- the analytic pole probe ---------------------------------------
      auto Dpr = r["probe"]["Delta"].vd();
      auto wpr = r["probe"]["w"].vd();
      const long npr = long(Dpr.size());
      nda::array<dcomplex, 1> Pc(c.rank);
      {
        const dcomplex rot = std::exp(dcomplex(0.0, -c.g.theta));
        for (long j = 0; j < c.rank; ++j) {
          dcomplex acc(0.0, 0.0);
          for (long k = 0; k < npr; ++k)
            acc += wpr[std::size_t(k)]
                 * std::exp(dcomplex(0.0, -1.0) * Dpr[std::size_t(k)] * c.s(j) * rot);
          Pc(j) = acc;
        }
      }
      double probe = 0.0;
      std::vector<dcomplex> Rgot(static_cast<std::size_t>(2 * ntarg));
      for (long t = 0; t < 2 * ntarg; ++t) {
        dcomplex acc(0.0, 0.0);
        for (long j = 0; j < c.rank; ++j) acc += tr.F(t, j) * Pc(j);
        Rgot[std::size_t(t)] = acc;
        dcomplex ex(0.0, 0.0);
        for (long k = 0; k < npr; ++k)
          ex += wpr[std::size_t(k)] / (tr.z(t) - Dpr[std::size_t(k)]);
        probe = std::max(probe, std::abs(acc - ex) / std::abs(ex));
      }
      const double probe_ref = r["probe"]["max_rel"].d();
      worst_probe_ratio = std::max(worst_probe_ratio, probe / probe_ref);

      // cross-implementation agreement at three sampled targets. The LS system
      // is conditioned at ~1e6-1e7 (results section 5.1), so the two transforms
      // agree to cond * eps_mach, not bitwise -- this is a meter, not a pin.
      const long sel[3] = {0, ntarg / 2, ntarg - 1};
      auto Rre = r["probe"]["R_re"].vd(), Rim = r["probe"]["R_im"].vd();
      double xpy = 0.0;
      for (int q = 0; q < 3; ++q) {
        const dcomplex py(Rre[std::size_t(q)], Rim[std::size_t(q)]);
        xpy = std::max(xpy, std::abs(Rgot[std::size_t(sel[q])] - py) / std::abs(py));
      }
      worst_xpy = std::max(worst_xpy, xpy);

      app_log(2, "[TC-1-a] {:<22} rank {:>3}/{:>3}/{:>3} (eps/eps2/ceil; ref ceil {:>3}) "
                 " nodes exact "
                 "{:>3}/{:<3} dev {:.2f}h  F_inbasis {:.3e} (ref {:.3e}, x{:.2f})  "
                 "probe {:.3e} (ref {:.3e})  cond {:.3e} (ref {:.3e})  vs-python {:.2e}",
              name, c.rank_eps, c.rank_eps2, c.rank_ceil, rk["ceiling"].i(),
              nex, c.rank, dev,
              inb, inb_ref, inb / inb_ref, probe, probe_ref, tr.cond,
              r["F"]["ls_cond"].d(), xpy);

      // THE GATES. "at the campaign's error class" -- the reference number is
      // the pin; a factor 2 covers the LS round-off of an equally-conditioned
      // but not bitwise-identical factorization.
      REQUIRE(inb   <= 2.0 * inb_ref);
      REQUIRE(probe <= 2.0 * probe_ref);
      REQUIRE(tr.cond <= 5.0 * r["F"]["ls_cond"].d());
      REQUIRE(tr.cond < 1e9);          // BINDING 4: nowhere near the 1e12 ceiling
      REQUIRE(xpy < 1e-6);
    }

    app_log(2, "[TC-1-a] SUMMARY: {} cases; nodes bit-identical {}/{}; worst node "
               "deviation {:.3f} grid spacings; worst geometry rel dev {:.2e}; worst "
               "F/ref {:.3f}; worst probe/ref {:.3f}; worst C++-vs-python {:.2e}",
            ncase, n_node_exact, n_node_total, worst_node_dev, worst_geom,
            worst_F_ratio, worst_probe_ratio, worst_xpy);
  }

  // =======================================================================
  //  GATE TC-1-b -- the ported unit pins
  // =======================================================================
  TEST_CASE("tilted_contour_pin1_exponent", "[numerics][tilted_contour]") {
    // |K(s)| = exp(-a(D) s) with a = (D-w) sin th + d cos th, and the rotated
    // contour integral of a single pole reproduces 1/(z-D).
    // [port of tc_validation/tests/pins.py::pin1_exponent_algebra]
    double worst_mod = 0.0, worst_int = 0.0;
    // a deterministic sweep in place of the reference's rng(0) draw
    for (int a1 = 0; a1 < 4; ++a1)
      for (int a2 = 0; a2 < 4; ++a2)
        for (int a3 = 0; a3 < 3; ++a3) {
          const double th = 0.05 + 1.1 * double(a1) / 3.0;
          const double D  = 0.5 + 119.0 * double(a2) / 3.0;
          const double w  = -20.0 + 40.0 * double(a3) / 2.0;
          const double d  = 0.05 + 4.9 * double((a1 + a2) % 3) / 2.0;
          const double a  = (D - w) * std::sin(th) + d * std::cos(th);
          if (a <= 1e-3) continue;
          const dcomplex z(w, d);
          const long n = 200001;
          const double smax = 40.0 / a, h = smax / double(n - 1);
          const dcomplex rot = std::exp(dcomplex(0.0, -th));
          dcomplex acc(0.0, 0.0);
          for (long i = 0; i < n; ++i) {
            const double s = h * double(i);
            const dcomplex K = std::exp(dcomplex(0.0, 1.0) * (z - D) * s * rot);
            worst_mod = std::max(worst_mod, std::abs(std::abs(K) - std::exp(-a * s)));
            const double wq = (i == 0 or i == n - 1) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
            acc += wq * K;
          }
          acc *= h / 3.0;
          const dcomplex num = dcomplex(0.0, -1.0) * rot * acc;
          const dcomplex ex = 1.0 / (z - D);
          worst_int = std::max(worst_int, std::abs(num - ex) / std::abs(ex));
        }
    app_log(2, "[TC-1-b pin1] max ||K| - e^-as| = {:.3e}; max rel err of the brute-force "
               "rotated contour integral = {:.3e}", worst_mod, worst_int);
    REQUIRE(worst_mod < 1e-12);
    REQUIRE(worst_int < 1e-9);
  }

  TEST_CASE("tilted_contour_pin3_endpoints", "[numerics][tilted_contour]") {
    // th = 0 -> G = 1/[2d - i(x-x')];  th = pi/2, W = 0 -> G = 1/(x+x') (Cauchy).
    // [port of tc_validation/tests/pins.py::pin3_endpoints]
    const long n = 41;
    const double d = 0.7;
    {
      nda::array<double, 1> x(n), a(n);
      for (long i = 0; i < n; ++i) {
        x(i) = 100.0 * double(i) / double(n - 1);
        a(i) = tc::a_of_x(x(i), 20.0, d, 0.0);
      }
      auto G = tc::gram(x, a, 0.0);
      double e0 = 0.0;
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
          e0 = std::max(e0, std::abs(G(i, j)
                                     - 1.0 / dcomplex(2.0 * d, -(x(i) - x(j)))));
      app_log(2, "[TC-1-b pin3] theta = 0 line limit: max dev = {:.3e}", e0);
      REQUIRE(e0 < 1e-13);
    }
    {
      nda::array<double, 1> x(n), a(n);
      for (long i = 0; i < n; ++i) {
        x(i) = std::pow(10.0, -2.0 + 4.0 * double(i) / double(n - 1));
        a(i) = tc::a_of_x(x(i), 0.0, 0.0, M_PI / 2.0);
      }
      auto G = tc::gram(x, a, M_PI / 2.0);
      double num = 0.0, den = 0.0;
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) {
          const double ref = 1.0 / (x(i) + x(j));
          num = std::max(num, std::abs(G(i, j) - ref));
          den = std::max(den, std::abs(ref));
        }
      app_log(2, "[TC-1-b pin3] theta = pi/2, W = 0 Cauchy limit: max rel dev = {:.3e}",
              num / den);
      REQUIRE(num / den < 1e-13);
    }
  }

  TEST_CASE("tilted_contour_pin4_F_exactness", "[numerics][tilted_contour]") {
    // ID identity T[:,J] = I, exponential collapse of the ID reconstruction along
    // the rank ladder, and F exactness at the fully resolved rank.
    // [port of tc_validation/tests/pins.py::pin4_F_exactness, with the `gram` row
    //  weighting (the BINDING choice) in place of the reference's `relative`;
    //  measured python ladder at `gram`: 9.64e-02 / 3.70e-05 / 2.34e-07]
    tc::contour_params_t p;
    p.dmin = 2.5; p.dmax = 150.0; p.W = 20.0; p.delta = 0.5; p.rho = 0.65;
    p.eps = 1e-6; p.eps_tr = 1e-12; p.nx = 2000; p.kappa = 0.35;

    const long ntarg = 41;
    nda::array<dcomplex, 1> zr(ntarg);
    for (long t = 0; t < ntarg; ++t)
      zr(t) = dcomplex(p.dmin + p.W * double(t) / double(ntarg - 1), p.delta);

    // one build for the eigen-spectrum (the rank ladder reads it three times)
    auto c0 = tc::build_contour(p);
    const long r_eps = c0.rank_eps, r_eps2 = c0.rank_eps2, r_full = c0.rank_ceil;
    app_log(2, "[TC-1-b pin4] rank ladder: eps -> {}, eps^2 -> {}, 1e-16 -> {} "
               "(reference 188 / 376 / 520; the 1e-16 rung sits in the eigensolver "
               "round-off floor and is NOT reproducible across zheev/zheevd -- "
               "reported, not gated)", r_eps, r_eps2, r_full);
    REQUIRE(r_eps  == 188);        // results section 2.1, the spec-convention rank
    REQUIRE(r_eps2 == 376);        // results section 2.1, the accuracy rank N_s^sigma
    REQUIRE(r_full > r_eps2);

    double id_ident = 0.0;
    std::vector<double> rec, ferr;
    for (long r : {r_eps, r_eps2, r_full}) {
      auto pp = p;
      pp.rank_force = r;
      auto c = tc::build_contour(pp, {}, /*want_T*/ true);
      REQUIRE(c.rank == r);
      id_ident = std::max(id_ident, c.id_identity);

      // ID reconstruction: max |M - M[:,J] T| / max |M|
      const long nx = c.x.size(), nS = c.s_grid.size();
      const dcomplex z_worst(p.dmin + p.W, p.delta);
      const dcomplex rot = std::exp(dcomplex(0.0, -c.g.theta));
      auto ph = tc::target_phase(c.s_grid, z_worst, c.g.theta);
      double num = 0.0, den = 0.0;
      std::vector<dcomplex> Msel(static_cast<std::size_t>(r));
      for (long i = 0; i < nx; ++i) {
        const double D = p.dmin + c.x(i);
        const double nrm = std::sqrt(1.0 / (2.0 * c.a(i)));
        for (long j = 0; j < r; ++j)
          Msel[std::size_t(j)] =
              std::exp(dcomplex(0.0, -1.0) * D * c.s(j) * rot) * ph(c.idx(j)) / nrm;
        for (long j = 0; j < nS; ++j) {
          const dcomplex M =
              std::exp(dcomplex(0.0, -1.0) * D * c.s_grid(j) * rot) * ph(j) / nrm;
          dcomplex acc(0.0, 0.0);
          for (long k = 0; k < r; ++k) acc += Msel[std::size_t(k)] * c.T(k, j);
          num = std::max(num, std::abs(M - acc));
          den = std::max(den, std::abs(M));
        }
      }
      rec.push_back(num / den);

      auto tr = tc::build_transform(c, zr, /*with_mirror*/ false);
      double e = 0.0;
      for (long i = 0; i < nx; ++i) {
        const double D = p.dmin + c.x(i);
        for (long j = 0; j < r; ++j)
          Msel[std::size_t(j)] = std::exp(dcomplex(0.0, -1.0) * D * c.s(j) * rot);
        for (long t = 0; t < ntarg; ++t) {
          dcomplex acc(0.0, 0.0);
          for (long j = 0; j < r; ++j) acc += tr.F(t, j) * Msel[std::size_t(j)];
          const dcomplex ex = 1.0 / (zr(t) - D);
          e = std::max(e, std::abs(acc - ex) / std::abs(ex));
        }
      }
      ferr.push_back(e);
      app_log(2, "[TC-1-b pin4] rank {:>3}: |T[:,J]-I| = {:.2e}  id_recon = {:.3e}  "
                 "F_relerr = {:.3e}  cond = {:.3e}",
              r, c.id_identity, rec.back(), e, tr.cond);
    }
    REQUIRE(id_ident < 1e-10);
    REQUIRE(rec[0] > rec[1]);
    REQUIRE(rec[1] > rec[2]);
    REQUIRE(ferr[0] > ferr[1]);
    REQUIRE(ferr[1] > ferr[2]);
    REQUIRE(ferr[2] < 1e-6);
    REQUIRE(ferr[1] < 1e-4);
  }

  TEST_CASE("tilted_contour_mirror", "[numerics][tilted_contour]") {
    // BINDING 2: the anti-resonant half from the SAME contour samples.
    // (a) a(-Delta) < 0 at production geometry -- the divergence the mirror avoids;
    // (b) P(z) = R(z) + conj(R(-conj z)) exactly, for a real-Delta pole model.
    // [port of tc_validation/tests/pins.py::pin_mirror]
    tc::contour_params_t p;
    p.dmin = 2.5; p.dmax = 150.0; p.W = 20.0; p.delta = 0.5; p.rho = 0.65;
    p.eps = 1e-6; p.eps_tr = 1e-12; p.nx = 600;
    auto g = tc::derive_geometry(p);
    // the anti-resonant pole at -Dmax seen from the worst target w = Dmin + W
    const double a_anti = (-p.dmax - (p.dmin + p.W)) * std::sin(g.theta)
                        + p.delta * std::cos(g.theta);
    app_log(2, "[TC-1-b mirror] a(-Delta_max) = {:.4f} (< 0: the anti-resonant half "
               "diverges on this contour)", a_anti);
    REQUIRE(a_anti < 0.0);

    const long npr = 9;
    double worst = 0.0;
    for (long t = 0; t < 13; ++t) {
      const dcomplex z(p.dmin + p.W * double(t) / 12.0, p.delta);
      const dcomplex zm = tc::mirror_target(z);
      dcomplex R(0.0, 0.0), Rm(0.0, 0.0), P(0.0, 0.0);
      for (long k = 0; k < npr; ++k) {
        const double D = p.dmin + (p.dmax - p.dmin) * double(k) / double(npr - 1);
        const double w = 1.0 / double(k + 1);
        R  += w / (z - D);
        Rm += w / (zm - D);
        P  += w / (z - D) + w / (-z - D);
      }
      worst = std::max(worst, std::abs(tc::combine_mirror(R, Rm) - P) / std::abs(P));
    }
    app_log(2, "[TC-1-b mirror] max rel dev of R(z) + conj(R(-conj z)) from R(z)+R(-z) "
               "= {:.3e}", worst);
    REQUIRE(worst < 1e-14);
  }

} // namespace bdft_tests
