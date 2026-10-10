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
 * test_line_dlr (plan notes/line_gw_cpp_plan.md, session S1, task 4): the line-DLR numerics on pole models.
 *   1. bases and ray against the python oracle (tests/unit_test_files/gw_line/line_dlr_ref.h5, written by
 *      coqui/cayley/scripts/gen_line_dlr_ref.py): ranks, poles, nodes, GL panels;
 *   2. fermionic LS fit on dense nodes: residual, imaginary axis, sector split;
 *   3. Cayley moments from the fitted poles vs exact (the key accuracy test);
 *   4. bosonic odd-symmetric fit: full W and both sectors on the imaginary axis;
 *   5. ray transform of pole-pair products on both rays;
 *   6. time-node ID (session S7a, numerics/line_dlr/time_id.hpp, tag [time_id]): transform error on pole models,
 *      products (Sigma-like and Pi-like with a conjugate-time factor), agreement with the GL ray, robustness
 *      (out-of-range energies, noise amplification) and construction time; hidden tag [.time_id_scan]: scaling of
 *      r_t with log(Emax/Emin) log(1/eps), GL node counts for the same accuracy, accuracy vs number of nodes.
 * Every measured number is printed.
 */

#undef NDEBUG

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"

namespace bdft_tests {

namespace ldlr = numerics::line_dlr;
using ldlr::sector_t;
using dcomplex = std::complex<double>;

namespace {

const std::string ref_file = std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/line_dlr_ref.h5";
constexpr double deg = 3.14159265358979323846 / 180.0;

template <typename T> T read_attr(h5::group &g, std::string const &name) {
  T v{};
  h5::h5_read_attribute(g, name, v);
  return v;
}

nda::array<dcomplex, 1> read_nodes(h5::group &g) {
  nda::array<double, 1> re, im;
  nda::h5_read(g, "nodes_re", re);
  nda::h5_read(g, "nodes_im", im);
  nda::array<dcomplex, 1> z(re.size());
  for (long i = 0; i < re.size(); ++i) z(i) = dcomplex(re(i), im(i));
  return z;
}

/// number of entries of a not within rel*|a_i| of some entry of b, and max |a_i - nearest b| over the matched ones
template <typename A, typename B> std::pair<long, double> match_sets(A const &a, B const &b, double rel) {
  long nmiss = 0;
  double dmax = 0.0;
  for (long i = 0; i < a.size(); ++i) {
    double best = 1e300;
    for (long j = 0; j < b.size(); ++j) best = std::min(best, std::abs(a(i) - b(j)));
    if (best > rel * std::abs(a(i))) ++nmiss;
    else dmax = std::max(dmax, best);
  }
  return {nmiss, dmax};
}

/// max over the poles a of the 2-norm LS residual of the normalized kernel column k(a) projected on span{k(b)}:
/// basis equivalence independent of which near-degenerate candidate the pivoting picked
template <typename KCol>
double span_residual(nda::array<double, 1> const &pa, nda::array<double, 1> const &pb, KCol &&kcol) {
  auto build = [&](nda::array<double, 1> const &p) {
    const long nr = kcol(p(0)).size();
    nda::array<dcomplex, 2> K(nr, p.size());
    for (long j = 0; j < p.size(); ++j) {
      auto c     = kcol(p(j));
      double nrm = 0.0;
      for (auto const &x : c) nrm += std::norm(x);
      nrm = std::sqrt(nrm);
      for (long i = 0; i < nr; ++i) K(i, j) = c(i) / nrm;
    }
    return K;
  };
  auto Ka = build(pa), Kb = build(pb);
  auto X  = ldlr::detail::lstsq(Kb, Ka);
  nda::array<dcomplex, 2> R = Ka;
  R -= ldlr::detail::matmul(Kb, X);
  double m = 0.0;
  for (long j = 0; j < R.extent(1); ++j) {
    double n2 = 0.0;
    for (long i = 0; i < R.extent(0); ++i) n2 += std::norm(R(i, j));
    m = std::max(m, std::sqrt(n2));
  }
  return m;
}

/// max over a of min_b |a - b|/|a| (0 for a = 0 matched exactly)
double max_nearest_rel(nda::array<double, 1> const &a, nda::array<double, 1> const &b) {
  double m = 0.0;
  for (long i = 0; i < a.size(); ++i) {
    double best = 1e300;
    for (long j = 0; j < b.size(); ++j) best = std::min(best, std::abs(a(i) - b(j)));
    m = std::max(m, a(i) == 0.0 ? best : best / std::abs(a(i)));
  }
  return m;
}

nda::array<dcomplex, 1> imag_axis(double nmin, double nmax, long n) {
  auto nu = ldlr::detail::logspace(nmin, nmax, n);
  nda::array<dcomplex, 1> z(n);
  for (long i = 0; i < n; ++i) z(i) = dcomplex(0.0, nu(i));
  return z;
}

double max_abs(nda::array<dcomplex, 2> const &X) {
  double m = 0.0;
  for (auto const &x : X) m = std::max(m, std::abs(x));
  return m;
}
double max_abs_diff(nda::array<dcomplex, 2> const &X, nda::array<dcomplex, 2> const &Y) {
  double m = 0.0;
  for (long i = 0; i < X.extent(0); ++i)
    for (long j = 0; j < X.extent(1); ++j) m = std::max(m, std::abs(X(i, j) - Y(i, j)));
  return m;
}

/// random Hermitian positive n x n matrix v v^dag, flattened row-major
std::vector<dcomplex> random_psd(std::mt19937_64 &gen, long n) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<dcomplex> v(n * n), a(n * n, 0.0);
  for (auto &x : v) x = dcomplex(nd(gen), nd(gen));
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j)
      for (long k = 0; k < n; ++k) a[i * n + j] += v[i * n + k] * std::conj(v[j * n + k]);
  return a;
}

/// fermionic pole model: X(zeta) = sum_p a_p / (zeta - E_p), a_p 2x2 Hermitian positive, sum_p tr a_p = 1
struct fermionic_model {
  std::vector<double> E;
  std::vector<std::vector<dcomplex>> a;          // (np) x (4)
  fermionic_model(long nper, double emin, double emax, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u(std::log(emin), std::log(emax));
    for (long p = 0; p < nper; ++p) E.push_back(std::exp(u(gen)));
    for (long p = 0; p < nper; ++p) E.push_back(-std::exp(u(gen)));
    double tr = 0.0;
    for (size_t p = 0; p < E.size(); ++p) {
      a.push_back(random_psd(gen, 2));
      tr += std::real(a.back()[0] + a.back()[3]);
    }
    for (auto &ap : a)
      for (auto &x : ap) x /= tr;
  }
  /// [nz, 4]; sector both / particle (E > 0) / hole (E < 0)
  nda::array<dcomplex, 2> operator()(nda::array<dcomplex, 1> const &z, sector_t s = sector_t::both) const {
    nda::array<dcomplex, 2> X(z.size(), 4);
    X() = 0.0;
    for (size_t p = 0; p < E.size(); ++p) {
      if (s == sector_t::particle and E[p] < 0.0) continue;
      if (s == sector_t::hole and E[p] > 0.0) continue;
      for (long i = 0; i < z.size(); ++i)
        for (long k = 0; k < 4; ++k) X(i, k) += a[p][k] / (z(i) - E[p]);
    }
    return X;
  }
  /// exact Cayley moments sum_p a_p u(E_p)^n, [nmax+1, 4] (sector as above)
  nda::array<dcomplex, 2> moments(double wp, long nmax, sector_t s = sector_t::both) const {
    nda::array<dcomplex, 2> C(nmax + 1, 4);
    C() = 0.0;
    for (size_t p = 0; p < E.size(); ++p) {
      if (s == sector_t::particle and E[p] < 0.0) continue;
      if (s == sector_t::hole and E[p] > 0.0) continue;
      const dcomplex u = cayley(E[p], wp);
      dcomplex un      = 1.0;
      for (long n = 0; n <= nmax; ++n, un *= u)
        for (long k = 0; k < 4; ++k) C(n, k) += a[p][k] * un;
    }
    return C;
  }
  /// u(w) = (w + i wp)/(w - i wp)  (coqui/cayley/cayley/maps.py::cayley with mu = 0)
  static dcomplex cayley(double w, double wp) { return dcomplex(w, wp) / dcomplex(w, -wp); }
};

/// Cayley moments of a pole representation sum_l c_l/(z - w_l): [nmax+1, ncol]
nda::array<dcomplex, 2> moments_from_poles(nda::array<double, 1> const &w, nda::array<dcomplex, 2> const &c, double wp,
                                           long nmax) {
  nda::array<dcomplex, 2> C(nmax + 1, c.extent(1));
  C() = 0.0;
  for (long l = 0; l < w.size(); ++l) {
    const dcomplex u = fermionic_model::cayley(w(l), wp);
    dcomplex un      = 1.0;
    for (long n = 0; n <= nmax; ++n, un *= u)
      for (long k = 0; k < c.extent(1); ++k) C(n, k) += c(l, k) * un;
  }
  return C;
}

/// sum_l c_l / (z - w_l) for an explicit pole list
nda::array<dcomplex, 2> eval_poles(nda::array<double, 1> const &w, nda::array<dcomplex, 2> const &c,
                                   nda::array<dcomplex, 1> const &z) {
  nda::array<dcomplex, 2> X(z.size(), c.extent(1));
  X() = 0.0;
  for (long i = 0; i < z.size(); ++i)
    for (long l = 0; l < w.size(); ++l)
      for (long k = 0; k < c.extent(1); ++k) X(i, k) += c(l, k) / (z(i) - w(l));
  return X;
}

} // namespace

// =======================================================================
//  1. bases and ray vs the python oracle
// =======================================================================
TEST_CASE("line_dlr_reference", "[numerics][line_dlr]") {
  h5::file file(ref_file, 'r');
  h5::group root(file);
  std::cout << std::scientific << std::setprecision(3);
  // Pole IDENTITY is not reproducible: neighbouring candidates (~1% apart) are near-degenerate columns, and a 1e-16
  // relative perturbation of the kernel already changes 107/121 (f0) or 158/186 (f2) of the python poles. What is
  // reproducible is the rank and the SPAN: every python kernel column lies in the span of the C++ columns (and vice
  // versa) to ~eps; python-vs-python under such perturbations gives 0.8-1.0 eps. Exact matches are reported.
  std::cout << "\n[line_dlr] case  lam  eps  gap  rank(py) rank(C++) | poles: #not-identical  max|d| of identical  "
               "max nearest-rel  span py->C++  span C++->py | nodes #not-identical\n";

  for (std::string name : {"f0", "f1", "f2"}) {
    auto g = root.open_group(name);
    const auto theta = read_attr<double>(g, "theta"), lam = read_attr<double>(g, "lam"), eps = read_attr<double>(g, "eps");
    const auto gm = read_attr<double>(g, "gap_minus"), gp = read_attr<double>(g, "gap_plus");
    const auto tmin = read_attr<double>(g, "tmin"), tmax = read_attr<double>(g, "tmax");
    const auto nline = read_attr<long>(g, "nline"), npole = read_attr<long>(g, "npole"), rpy = read_attr<long>(g, "rank");
    nda::array<double, 1> wpy;
    nda::h5_read(g, "poles", wpy);
    auto zpy = read_nodes(g);

    ldlr::line_basis_t b(theta, lam, eps, gm, gp, tmin, tmax, nline, npole);
    auto [pmiss, pd] = match_sets(b.w, wpy, 1e-6);
    auto [zmiss, zd] = match_sets(b.zeta_nodes, zpy, 1e-6);
    auto kcol = [&](double e) {
      nda::array<dcomplex, 1> c(b.zeta_dense.size());
      for (long i = 0; i < c.size(); ++i) c(i) = 1.0 / (b.zeta_dense(i) - e);
      return c;
    };
    const double s1 = span_residual(wpy, b.w, kcol), s2 = span_residual(b.w, wpy, kcol);
    const double nr = std::max(max_nearest_rel(b.w, wpy), max_nearest_rel(wpy, b.w));
    std::cout << "[line_dlr] " << name << "  " << lam << " " << eps << " (" << gm << "," << gp << ")  " << rpy << " "
              << b.rank << " | " << pmiss << "  " << pd << "  " << nr << "  " << s1 << "  " << s2 << " | " << zmiss
              << "\n";
    CHECK(std::abs(b.rank - rpy) <= 5);
    CHECK(s1 <= 10 * eps);
    CHECK(s2 <= 10 * eps);
    CHECK(b.zeta_nodes.size() == b.rank);
    CHECK(b.w.size() == b.rank);
    for (long l = 1; l < b.rank; ++l) REQUIRE(b.w(l) > b.w(l - 1));
  }

  for (std::string name : {"b0", "b1"}) {
    auto g = root.open_group(name);
    const auto theta = read_attr<double>(g, "theta"), lam = read_attr<double>(g, "lam"), eps = read_attr<double>(g, "eps");
    const auto gap = read_attr<double>(g, "gap"), tmin = read_attr<double>(g, "tmin"), tmax = read_attr<double>(g, "tmax");
    const auto nline = read_attr<long>(g, "nline"), npole = read_attr<long>(g, "npole"), rpy = read_attr<long>(g, "rank");
    nda::array<double, 1> npy;
    nda::h5_read(g, "poles", npy);
    auto zpy = read_nodes(g);

    ldlr::bosonic_basis_t b(theta, lam, eps, gap, tmin, tmax, nline, npole, 0.0);   // python node selection
    auto [pmiss, pd] = match_sets(b.nu, npy, 1e-6);
    auto [zmiss, zd] = match_sets(b.zeta_nodes, zpy, 1e-6);
    const long nd = b.zeta_dense.size();
    auto kcol = [&](double e) {               // stacked kernel [1/(z - nu); -1/(z + nu)]
      nda::array<dcomplex, 1> c(2 * nd);
      for (long i = 0; i < nd; ++i) {
        c(i)      = 1.0 / (b.zeta_dense(i) - e);
        c(nd + i) = -1.0 / (b.zeta_dense(i) + e);
      }
      return c;
    };
    const double s1 = span_residual(npy, b.nu, kcol), s2 = span_residual(b.nu, npy, kcol);
    const double nr = std::max(max_nearest_rel(b.nu, npy), max_nearest_rel(npy, b.nu));
    std::cout << "[line_dlr] " << name << "(bos) " << lam << " " << eps << " " << gap << "  " << rpy << " " << b.rank
              << " | " << pmiss << "  " << pd << "  " << nr << "  " << s1 << "  " << s2 << " | " << zmiss << " (nodes "
              << b.zeta_nodes.size() << " vs " << zpy.size() << ")\n";
    CHECK(std::abs(b.rank - rpy) <= 5);
    CHECK(s1 <= 10 * eps);
    CHECK(s2 <= 10 * eps);
    CHECK(b.zeta_nodes.size() == std::min(2 * b.rank, 2 * nline));
    for (long l = 0; l < b.rank; ++l) REQUIRE(b.nu(l) > 0.0);
  }

  {
    auto g = root.open_group("ray");
    const auto th_t = read_attr<double>(g, "theta_t"), emin = read_attr<double>(g, "emin");
    const auto dec = read_attr<double>(g, "decades"), smin = read_attr<double>(g, "smin");
    const auto pe = read_attr<double>(g, "per_efold");
    const auto nn = read_attr<long>(g, "nn");
    nda::array<double, 1> spy, wpy;
    nda::h5_read(g, "s", spy);
    nda::h5_read(g, "ws", wpy);
    auto r = ldlr::time_ray_t::for_spectrum(th_t, emin, dec, smin, pe, nn, sector_t::particle);
    REQUIRE(r.size() == spy.size());
    double es = 0.0, ew = 0.0;
    for (long m = 0; m < r.size(); ++m) {
      es = std::max(es, std::abs(r.s(m) - spy(m)) / std::abs(spy(m)));
      ew = std::max(ew, std::abs(r.ws(m) - wpy(m)) / std::abs(wpy(m)));
    }
    std::cout << "[line_dlr] ray: " << r.size() << " nodes, max rel |ds| = " << es << ", |dws| = " << ew << "\n";
    CHECK(es <= 1e-13);
    CHECK(ew <= 1e-13);
  }
}

// =======================================================================
//  2-3. fermionic fit on the dense nodes and Cayley moments
// =======================================================================
TEST_CASE("line_dlr_fermionic_fit_moments", "[numerics][line_dlr]") {
  std::cout << std::scientific << std::setprecision(3);
  const double theta = 20.0 * deg, eps = 1e-10, lam = 6.0, wp = 0.11;
  const long nmax = 24;
  fermionic_model model(20, 0.03, 0.8 * lam, 12345u);
  auto zd = ldlr::dense_nodes(theta, 1e-3, 60.0, 120);
  auto zi = imag_axis(1e-2, 100.0, 200);
  auto Xd = model(zd), Xi = model(zi);
  const double sd = max_abs(Xd), si = max_abs(Xi);

  ldlr::line_basis_t b(theta, lam, eps, 0.02, 0.02);
  auto c = b.fit(zd, Xd);
  const double res  = max_abs_diff(b.eval(c, zd), Xd) / sd;
  const double eimg = max_abs_diff(b.eval(c, zi), Xi) / si;
  std::cout << "\n[line_dlr] fermionic fit: rank " << b.rank << ", " << zd.size() << " dense nodes; residual " << res
            << ", imag-axis " << eimg << " (tol " << 10 * eps << ")\n";
  CHECK(res <= 10 * eps);
  CHECK(eimg <= 10 * eps);

  // 3x3 convenience overload equals the flattened fit
  {
    nda::array<dcomplex, 3> X3(zd.size(), 2, 2);
    for (long i = 0; i < zd.size(); ++i)
      for (long k = 0; k < 4; ++k) X3(i, k / 2, k % 2) = Xd(i, k);
    auto c3 = b.fit(zd, X3);
    double d = 0.0;
    for (long l = 0; l < b.rank; ++l)
      for (long k = 0; k < 4; ++k) d = std::max(d, std::abs(c3(l, k / 2, k % 2) - c(l, k)));
    CHECK(d == 0.0);
  }

  // sector split of a plain signed fit. The two-sided basis can trade weight across the gap: the python oracle on an
  // analogous random model gives 1.2e-2 (relative to max|X|), uniformly in nu; here 2.7e-2. The target (1e-6 / 1e-3) is
  // NOT reachable with a signed fit; production splits sectors with one-sided bases (driver.py: bp/bh, gap=(lam, .)),
  // checked right below at 10 eps. Assert the achieved level here.
  {
    auto [ch, cp] = b.split(c);
    REQUIRE(ch.extent(0) + cp.extent(0) == b.rank);
    auto Xh = eval_poles(b.hole_poles(), ch, zi), Xp = eval_poles(b.particle_poles(), cp, zi);
    const double eh = max_abs_diff(Xh, model(zi, sector_t::hole)) / si;
    const double ep = max_abs_diff(Xp, model(zi, sector_t::particle)) / si;
    nda::array<dcomplex, 2> Xs = Xh + Xp;
    const double erec = max_abs_diff(Xs, b.eval(c, zi)) / si;
    std::cout << "[line_dlr] signed-fit split on the imag axis: hole " << eh << ", particle " << ep
              << " (not separable at this level; asserted <= 5e-2); recombination " << erec << "\n";
    CHECK(eh <= 5e-2);
    CHECK(ep <= 5e-2);
    CHECK(erec <= 1e-14);
  }

  // production sector split: one-sided bases (gap >= lam removes that side), each fitted to its sector's samples
  {
    ldlr::line_basis_t bp(theta, lam, eps, lam, 0.02), bh(theta, lam, eps, 0.02, lam);
    REQUIRE(bp.w(0) > 0.0);
    REQUIRE(bh.w(bh.rank - 1) < 0.0);
    auto Xpd = model(zd, sector_t::particle), Xhd = model(zd, sector_t::hole);
    auto cp = bp.fit(zd, Xpd), ch = bh.fit(zd, Xhd);
    const double ep = max_abs_diff(bp.eval(cp, zi), model(zi, sector_t::particle)) / si;
    const double eh = max_abs_diff(bh.eval(ch, zi), model(zi, sector_t::hole)) / si;
    auto Cp = moments_from_poles(bp.w, cp, wp, nmax), Ch = moments_from_poles(bh.w, ch, wp, nmax);
    const double mp = max_abs_diff(Cp, model.moments(wp, nmax, sector_t::particle));
    const double mh = max_abs_diff(Ch, model.moments(wp, nmax, sector_t::hole));
    std::cout << "[line_dlr] one-sided sector fits: ranks " << bp.rank << "/" << bh.rank << "; imag-axis particle " << ep
              << ", hole " << eh << "; sector moments n<=24: particle " << mp << ", hole " << mh << "\n";
    CHECK(ep <= 10 * eps);
    CHECK(eh <= 10 * eps);
    CHECK(mp <= 1e-9);
    CHECK(mh <= 1e-9);
  }

  // 3. Cayley moments from the fitted poles vs exact, n <= 24 (u = (w + i wp)/(w - i wp), wp = 0.11)
  {
    auto Cf = moments_from_poles(b.w, c, wp, nmax), Ce = model.moments(wp, nmax);
    const double em = max_abs_diff(Cf, Ce);
    std::cout << "[line_dlr] Cayley moments n<=" << nmax << " from fitted poles: max abs err " << em << " (tol 1e-9; |C0| ~ "
              << max_abs(Ce) << ")\n";
    CHECK(em <= 1e-9);
  }
}

// =======================================================================
//  4. bosonic odd-symmetric fit and sectors
// =======================================================================
TEST_CASE("line_dlr_bosonic_fit", "[numerics][line_dlr]") {
  std::cout << std::scientific << std::setprecision(3);
  const double theta = 20.0 * deg, eps = 1e-10;
  const long N = 3, nj = 15;
  std::mt19937_64 gen(777u);
  std::uniform_real_distribution<double> u(std::log(0.05), std::log(3.0));
  std::vector<double> nuj(nj);
  std::vector<std::vector<dcomplex>> wj(nj);
  double tr = 0.0;
  for (long j = 0; j < nj; ++j) nuj[j] = std::exp(u(gen));
  for (long j = 0; j < nj; ++j) {
    wj[j] = random_psd(gen, N);
    for (long P = 0; P < N; ++P) tr += std::real(wj[j][P * N + P]);
  }
  for (auto &w : wj)
    for (auto &x : w) x /= tr;
  // W_PQ = sum_j [ w_j,PQ/(z - nu_j) - w_j,QP/(z + nu_j) ]
  auto model = [&](nda::array<dcomplex, 1> const &z, sector_t s) {
    nda::array<dcomplex, 3> W(z.size(), N, N);
    W() = 0.0;
    for (long i = 0; i < z.size(); ++i)
      for (long j = 0; j < nj; ++j)
        for (long P = 0; P < N; ++P)
          for (long Q = 0; Q < N; ++Q) {
            if (s != sector_t::hole) W(i, P, Q) += wj[j][P * N + Q] / (z(i) - nuj[j]);
            if (s != sector_t::particle) W(i, P, Q) -= wj[j][Q * N + P] / (z(i) + nuj[j]);
          }
    return W;
  };
  auto diff = [](nda::array<dcomplex, 3> const &A, nda::array<dcomplex, 3> const &B) {
    double m = 0.0;
    for (long i = 0; i < A.size(); ++i) m = std::max(m, std::abs(A.data()[i] - B.data()[i]));
    return m;
  };

  auto zd = ldlr::dense_nodes(theta, 1e-3, 60.0, 120);
  auto zi = imag_axis(1e-2, 100.0, 200);
  ldlr::bosonic_basis_t b(theta, 4.0, eps, 0.02);
  auto w   = b.fit(zd, model(zd, sector_t::both));
  auto Wi  = model(zi, sector_t::both);
  double sc = 0.0;
  for (auto const &x : Wi) sc = std::max(sc, std::abs(x));
  const double eb = diff(b.eval(w, zi, sector_t::both), Wi) / sc;
  const double ep = diff(b.eval(w, zi, sector_t::particle), model(zi, sector_t::particle)) / sc;
  const double eh = diff(b.eval(w, zi, sector_t::hole), model(zi, sector_t::hole)) / sc;
  const double ed = diff(b.eval(w, zd, sector_t::both), model(zd, sector_t::both)) / sc;
  std::cout << "\n[line_dlr] bosonic fit: rank " << b.rank << ", nodes " << b.zeta_nodes.size() << "; residual " << ed
            << "; imag axis: full " << eb << ", '>' " << ep << ", '<' " << eh << " (tol " << 10 * eps << ")\n";
  CHECK(eb <= 10 * eps);
  CHECK(ep <= 10 * eps);
  CHECK(eh <= 10 * eps);

  // the same fit on the basis' own line nodes (the production W fit) instead of the dense set: the default
  // mirror-symmetric nodes (perf 7.1) and the python QR nodes
  for (double nf : {ldlr::bosonic_basis_t::default_node_factor, 1.0, 1.5, 0.0}) {
    ldlr::bosonic_basis_t bn(theta, 4.0, eps, 0.02, -1.0, -1.0, 1200, 800, nf);
    auto wn       = bn.fit(bn.zeta_nodes, model(bn.zeta_nodes, sector_t::both));
    const double en  = diff(bn.eval(wn, zi, sector_t::both), Wi) / sc;
    const double enp = diff(bn.eval(wn, zi, sector_t::particle), model(zi, sector_t::particle)) / sc;
    std::cout << "[line_dlr] bosonic fit on the " << bn.zeta_nodes.size() << " " << (nf > 0 ? "mirror-symmetric" : "QR")
              << " nodes (node factor " << nf << "): imag axis full " << en << ", '>' " << enp << "\n";
    CHECK(en <= 10 * eps);
    CHECK(enp <= 10 * eps);
    CHECK(ldlr::mirror_half(bn.zeta_nodes) == (nf > 0 ? bn.zeta_nodes.size() / 2 : 0));
    CHECK(bn.n_mirror == ldlr::mirror_half(bn.zeta_nodes));
  }

  // time exponentials: W^>(t) = sum_j w_j e^{-i nu_j t}, W^<(t) = -sum_j w_j^T e^{+i nu_j t}
  nda::array<dcomplex, 1> t(2);
  t(0) = dcomplex(0.7, -0.1);
  t(1) = dcomplex(2.0, 0.3);
  auto Ep = b.time_exponentials(t, sector_t::particle), Eh = b.time_exponentials(t, sector_t::hole);
  CHECK(std::abs(Ep(1, 3) - std::exp(dcomplex(0, -1) * b.nu(3) * t(1))) < 1e-15);
  CHECK(std::abs(Eh(0, 2) + std::exp(dcomplex(0, 1) * b.nu(2) * t(0))) < 1e-15);
}

// =======================================================================
//  5. ray transform of pole-pair products
// =======================================================================
TEST_CASE("line_dlr_ray_transform", "[numerics][line_dlr]") {
  std::cout << std::scientific << std::setprecision(3);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = ldlr::dense_nodes(theta, 1e-2, 1e3, 30);            // 60 targets on both upper rays
  auto run = [&](sector_t s, long nn, std::vector<std::pair<double, double>> const &pairs) {
    auto r  = ldlr::time_ray_t::for_spectrum(theta_t, 0.05, 40.0, 1e-5, 3.0, nn, s);
    auto F  = r.transform_matrix(z);
    double err = 0.0;
    for (auto [E1, E2] : pairs) {
      nda::array<double, 1> e1(1), e2(1);
      e1(0) = E1;
      e2(0) = E2;
      auto X1 = r.exponentials(e1), X2 = r.exponentials(e2);
      for (long i = 0; i < z.size(); ++i) {
        dcomplex acc = 0.0;
        for (long m = 0; m < r.size(); ++m) acc += F(i, m) * X1(m, 0) * X2(m, 0);
        err = std::max(err, std::abs(acc * (z(i) - E1 - E2) - 1.0));
      }
    }
    return std::make_pair(r.size(), err);
  };
  const std::vector<std::pair<double, double>> pp = {{0.03, 0.04}, {0.5, 2.0}, {3.0, 4.0}};
  const std::vector<std::pair<double, double>> hh = {{-0.03, -0.04}, {-0.5, -2.0}, {-3.0, -4.0}};
  auto [np16, ep16] = run(sector_t::particle, 16, pp);
  auto [nh16, eh16] = run(sector_t::hole, 16, hh);
  auto [np20, ep20] = run(sector_t::particle, 20, pp);
  auto [nh20, eh20] = run(sector_t::hole, 20, hh);
  std::cout << "\n[line_dlr] ray transform, |zeta| in [1e-2, 1e3] on both rays, max rel err:\n"
            << "[line_dlr]   nn=16 (" << np16 << " nodes): particle " << ep16 << ", hole " << eh16
            << "   (python oracle 1.7e-12)\n"
            << "[line_dlr]   nn=20 (" << np20 << " nodes): particle " << ep20 << ", hole " << eh20 << "\n";
  CHECK(np16 >= 850);
  CHECK(np16 <= 1040);
  // Default panels (3/e-fold, 16 GL nodes) reach 1.7e-12, not 1e-12, on the theta-ray targets with |zeta| >~ 3 Ha:
  // there the integrand oscillates cot(theta - theta_t) ~ 5.7 times faster than it decays and 16 nodes per panel
  // under-resolve it. Same number in the python oracle (quadrature truncation, not a transcription error); nn = 20
  // (or 4 panels per e-fold) reaches 1e-15. Assert the achieved level for the default and 1e-12 for nn = 20.
  CHECK(ep16 <= 2.5e-12);
  CHECK(eh16 <= 2.5e-12);
  CHECK(ep20 <= 1e-12);
  CHECK(eh20 <= 1e-12);
}

// =======================================================================
//  6. time-node ID (S7a)
// =======================================================================
namespace {

using clk = std::chrono::steady_clock;
double seconds_since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

/// production-like targets: 240 dense line nodes (both upper rays, |zeta| in [1e-3, 60]) + 40 imaginary-axis points
nda::array<dcomplex, 1> id_targets(double theta) {
  auto zl = ldlr::dense_nodes(theta, 1e-3, 60.0, 120);
  auto zi = imag_axis(1e-3, 60.0, 40);
  nda::array<dcomplex, 1> z(zl.size() + zi.size());
  for (long i = 0; i < zl.size(); ++i) z(i) = zl(i);
  for (long i = 0; i < zi.size(); ++i) z(zl.size() + i) = zi(i);
  return z;
}

/// scalar pole model sum_p c_p e^{-i E_p t} <-> sum_p c_p / (zeta - E_p)
struct exp_model {
  std::vector<double> E, c;
  /// log-uniform random magnitudes in [emin, emax] + both ends + nclust poles clustered within 5% above emin;
  /// positive weights summing to 1; sign = +1 (particle) / -1 (hole)
  exp_model(double emin, double emax, long nrand, long nclust, double sign, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    E.push_back(emin);
    E.push_back(emax);
    for (long p = 0; p < nrand; ++p) E.push_back(std::exp(std::log(emin) + u(gen) * std::log(emax / emin)));
    for (long p = 0; p < nclust; ++p) E.push_back(emin * (1.0 + 0.05 * u(gen)));
    double tot = 0.0;
    for (size_t p = 0; p < E.size(); ++p) {
      E[p] *= sign;
      c.push_back(0.1 + u(gen));
      tot += c.back();
    }
    for (auto &x : c) x /= tot;
  }
  dcomplex time(dcomplex t) const {
    dcomplex acc = 0.0;
    for (size_t p = 0; p < E.size(); ++p) acc += c[p] * std::exp(dcomplex(0.0, -E[p]) * t);
    return acc;
  }
  dcomplex freq(dcomplex z) const {
    dcomplex acc = 0.0;
    for (size_t p = 0; p < E.size(); ++p) acc += c[p] / (z - E[p]);
    return acc;
  }
};

/// max_i |sum_j F(i,j) x_j - X(z_i)| / |X(z_i)|
double transform_rel_err(nda::array<dcomplex, 2> const &F, nda::array<dcomplex, 1> const &x,
                         nda::array<dcomplex, 1> const &Xex) {
  double m = 0.0;
  for (long i = 0; i < F.extent(0); ++i) {
    dcomplex acc = 0.0;
    for (long j = 0; j < F.extent(1); ++j) acc += F(i, j) * x(j);
    m = std::max(m, std::abs(acc - Xex(i)) / std::abs(Xex(i)));
  }
  return m;
}

template <typename Nodes> double model_err(Nodes const &nd, nda::array<dcomplex, 2> const &F, exp_model const &M,
                                           nda::array<dcomplex, 1> const &z) {
  nda::array<dcomplex, 1> x(nd.size()), Xex(z.size());
  for (long j = 0; j < nd.size(); ++j) x(j) = M.time(nd.t(j));
  for (long i = 0; i < z.size(); ++i) Xex(i) = M.freq(z(i));
  return transform_rel_err(F, x, Xex);
}

/// strictest metric: single poles on a log grid of ne energies in [emin, emax] (signed by the sector),
/// max over targets and poles of |F e^{-iEt} - 1/(z-E)| |z-E|
template <typename Nodes> double per_pole_err(Nodes const &nd, nda::array<dcomplex, 2> const &F, double emin,
                                              double emax, long ne, nda::array<dcomplex, 1> const &z) {
  const double sgn = (nd.sector == sector_t::particle) ? 1.0 : -1.0;
  auto e           = ldlr::detail::logspace(emin, emax, ne);
  nda::array<double, 1> Ep(ne);
  for (long p = 0; p < ne; ++p) Ep(p) = sgn * e(p);
  nda::array<dcomplex, 2> X(nd.size(), ne);
  for (long j = 0; j < nd.size(); ++j)
    for (long p = 0; p < ne; ++p) X(j, p) = std::exp(dcomplex(0.0, -Ep(p)) * nd.t(j));
  auto Y   = ldlr::detail::matmul(F, X);
  double m = 0.0;
  for (long i = 0; i < z.size(); ++i)
    for (long p = 0; p < ne; ++p) m = std::max(m, std::abs(Y(i, p) * (z(i) - Ep(p)) - 1.0));
  return m;
}

struct id_case {
  const char *name;
  double emin, emax;
};
const std::vector<id_case> id_cases = {{"Pi    [0.04, 6]", 0.04, 6.0}, {"Sigma [0.06,10]", 0.06, 10.0},
                                       {"smallgap[.005,6]", 0.005, 6.0}};
const std::vector<double> id_eps = {1e-6, 1e-8, 1e-10};

} // namespace

TEST_CASE("line_dlr_time_id_transform", "[numerics][line_dlr][time_id]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = id_targets(theta);
  std::cout << "\n[time_id] 1. transform error on pole models, 240 line nodes + 40 imaginary-axis points, theta=20 "
               "theta_t=10 deg\n"
            << "[time_id]   case             sector  eps    over rank r_t  LSres    max|F|   cond     mix-err  "
               "per-pole  build(s)\n";
  for (auto const &c : id_cases)
    for (double eps : id_eps)
      for (sector_t sec : {sector_t::particle, sector_t::hole})
        for (double over : {1.0, 1.25}) {
          const double sgn = (sec == sector_t::particle) ? 1.0 : -1.0;
          ldlr::time_id_opts_t o;
          o.oversample = over;
          auto t0      = clk::now();
          ldlr::time_id_t id(theta_t, sec, c.emin, c.emax, eps, o);
          double res = 0.0;
          auto F     = id.transform_matrix(z, &res);
          const double tb = seconds_since(t0);
          exp_model M(c.emin, c.emax, 40, 8, sgn, 11u);
          const double em = model_err(id, F, M, z);
          const double ep = per_pole_err(id, F, c.emin, c.emax, 400, z);
          std::cout << "[time_id]   " << c.name << "  " << (sec == sector_t::particle ? "part" : "hole") << "  "
                    << std::setprecision(0) << eps << std::setprecision(2) << "  " << over << "  " << id.rank << "  "
                    << id.size() << "  " << res << " " << max_abs(F) << " " << id.ls_cond() << " " << em << " " << ep
                    << " " << tb << "\n";
          CHECK(em <= 10.0 * eps);
          CHECK(ep <= 10.0 * eps);
          CHECK(res <= 10.0 * eps);
        }
}

TEST_CASE("line_dlr_time_id_products", "[numerics][line_dlr][time_id]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = id_targets(theta);
  std::cout << "\n[time_id] 2. products at the ID nodes vs exact pole sums (rel. error)\n";
  for (double eps : id_eps)
    for (sector_t sec : {sector_t::particle, sector_t::hole}) {
      const double sgn = (sec == sector_t::particle) ? 1.0 : -1.0;
      // Sigma-like: G energies in [0.03, 4], W energies in [0.03, 6] -> summed range [0.06, 10]
      ldlr::time_id_t ids(theta_t, sec, 0.06, 10.0, eps);
      exp_model f(0.03, 4.0, 15, 3, sgn, 21u), g(0.03, 6.0, 15, 3, sgn, 22u);
      auto Fs = ids.transform_matrix(z);
      nda::array<dcomplex, 1> x(ids.size()), Xex(z.size());
      for (long j = 0; j < ids.size(); ++j) x(j) = f.time(ids.t(j)) * g.time(ids.t(j));
      for (long i = 0; i < z.size(); ++i) {
        dcomplex acc = 0.0;
        for (size_t p = 0; p < f.E.size(); ++p)
          for (size_t q = 0; q < g.E.size(); ++q) acc += f.c[p] * g.c[q] / (z(i) - f.E[p] - g.E[q]);
        Xex(i) = acc;
      }
      const double es = transform_rel_err(Fs, x, Xex);
      // Pi-like: conj(f(conj t)) g(t), f with energies of the OTHER sign (G^< for Pi^>), |e_i|, e_a in [0.02, 3]
      // -> energies e_a - e_i in [0.04, 6]
      ldlr::time_id_t idp(theta_t, sec, 0.04, 6.0, eps);
      exp_model fo(0.02, 3.0, 15, 3, -sgn, 23u), ga(0.02, 3.0, 15, 3, sgn, 24u);
      auto Fp = idp.transform_matrix(z);
      nda::array<dcomplex, 1> xp(idp.size()), Xp(z.size());
      for (long j = 0; j < idp.size(); ++j) xp(j) = std::conj(fo.time(std::conj(idp.t(j)))) * ga.time(idp.t(j));
      for (long i = 0; i < z.size(); ++i) {
        dcomplex acc = 0.0;
        for (size_t p = 0; p < fo.E.size(); ++p)
          for (size_t q = 0; q < ga.E.size(); ++q) acc += fo.c[p] * ga.c[q] / (z(i) - (ga.E[q] - fo.E[p]));
        Xp(i) = acc;
      }
      const double ep = transform_rel_err(Fp, xp, Xp);
      std::cout << "[time_id]   eps " << std::setprecision(0) << eps << std::setprecision(2) << " "
                << (sec == sector_t::particle ? "part" : "hole") << ": Sigma-like G o W (r_t=" << ids.size()
                << ") " << es << ",  Pi-like conj(G(conj t)) o G (r_t=" << idp.size() << ") " << ep << "\n";
      CHECK(es <= 10.0 * eps);
      CHECK(ep <= 10.0 * eps);
    }
}

TEST_CASE("line_dlr_time_id_vs_gl", "[numerics][line_dlr][time_id]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = id_targets(theta);
  std::cout << "\n[time_id] 3. ID vs GL ray (for_spectrum(Emin), 40 decades, 3 panels/e-fold, 16 GL nodes)\n";
  for (auto const &c : id_cases)
    for (double eps : id_eps)
      for (sector_t sec : {sector_t::particle, sector_t::hole}) {
        const double sgn = (sec == sector_t::particle) ? 1.0 : -1.0;
        ldlr::time_id_t id(theta_t, sec, c.emin, c.emax, eps);
        auto ray = ldlr::time_ray_t::for_spectrum(theta_t, c.emin, 40.0, 1e-5, 3.0, 16, sec);
        exp_model M(c.emin, c.emax, 40, 8, sgn, 31u);
        auto Fi = id.transform_matrix(z);
        auto Fg = ray.transform_matrix(z);
        double d = 0.0, eg = 0.0;
        for (long i = 0; i < z.size(); ++i) {
          dcomplex xi = 0.0, xg = 0.0;
          for (long j = 0; j < id.size(); ++j) xi += Fi(i, j) * M.time(id.t(j));
          for (long m = 0; m < ray.size(); ++m) xg += Fg(i, m) * M.time(ray.t(m));
          const dcomplex ex = M.freq(z(i));
          d  = std::max(d, std::abs(xi - xg) / std::abs(ex));
          eg = std::max(eg, std::abs(xg - ex) / std::abs(ex));
        }
        std::cout << "[time_id]   " << c.name << " " << (sec == sector_t::particle ? "part" : "hole") << " eps "
                  << std::setprecision(0) << eps << std::setprecision(2) << ": ID " << id.size() << " nodes, GL "
                  << ray.size() << " nodes (ratio " << std::setprecision(1) << std::fixed
                  << double(ray.size()) / double(id.size()) << std::scientific << std::setprecision(2)
                  << "), |ID - GL| " << d << ", GL err " << eg << "\n";
        CHECK(d <= 10.0 * eps);
      }
}

TEST_CASE("line_dlr_time_id_robustness", "[numerics][line_dlr][time_id]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = id_targets(theta);
  std::cout << "\n[time_id] 4. robustness: single poles outside the design range, 1e-12 relative noise on X(t_j)\n";
  std::mt19937_64 gen(41u);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (auto const &c : id_cases)
    for (double eps : id_eps) {
      ldlr::time_id_t id(theta_t, sector_t::particle, c.emin, c.emax, eps);
      auto F            = id.transform_matrix(z);
      const double ein  = per_pole_err(id, F, c.emin, c.emax, 200, z);
      const double elo  = per_pole_err(id, F, 0.8 * c.emin, 0.8 * c.emin * 1.0000001, 2, z);
      const double ehi  = per_pole_err(id, F, 1.2 * c.emax, 1.2 * c.emax * 1.0000001, 2, z);
      const double elo2 = per_pole_err(id, F, 0.5 * c.emin, 0.5 * c.emin * 1.0000001, 2, z);
      const double ehi2 = per_pole_err(id, F, 2.0 * c.emax, 2.0 * c.emax * 1.0000001, 2, z);
      // noise amplification on the mixture model
      exp_model M(c.emin, c.emax, 40, 8, 1.0, 43u);
      nda::array<dcomplex, 1> x(id.size()), xn(id.size()), Xex(z.size());
      for (long j = 0; j < id.size(); ++j) {
        x(j)  = M.time(id.t(j));
        xn(j) = x(j) * (1.0 + 1e-12 * dcomplex(nd(gen), nd(gen)) / std::sqrt(2.0));
      }
      for (long i = 0; i < z.size(); ++i) Xex(i) = M.freq(z(i));
      const double e0 = transform_rel_err(F, x, Xex), en = transform_rel_err(F, xn, Xex);
      double amp = 0.0;   // relative output change caused by the noise alone: max_i |F (xn - x)|_i / |X_i|
      for (long i = 0; i < z.size(); ++i) {
        dcomplex acc = 0.0;
        for (long j = 0; j < id.size(); ++j) acc += F(i, j) * (xn(j) - x(j));
        amp = std::max(amp, std::abs(acc) / std::abs(Xex(i)));
      }
      std::cout << "[time_id]   " << c.name << " eps " << std::setprecision(0) << eps << std::setprecision(2)
                << ": in-range " << ein << ", E=0.8Emin " << elo << ", E=1.2Emax " << ehi << ", E=0.5Emin " << elo2
                << ", E=2Emax " << ehi2 << " | noise 1e-12: err " << e0 << " -> " << en << ", noise-only output " << amp
                << " (max|F| " << max_abs(F) << ")\n";
      // same with the design range padded by 1.25 on both ends
      ldlr::time_id_opts_t o;
      o.pad = 1.25;
      ldlr::time_id_t idq(theta_t, sector_t::particle, c.emin, c.emax, eps, o);
      auto Fq            = idq.transform_matrix(z);
      const double qlo   = per_pole_err(idq, Fq, 0.8 * c.emin, 0.8 * c.emin * 1.0000001, 2, z);
      const double qhi   = per_pole_err(idq, Fq, 1.2 * c.emax, 1.2 * c.emax * 1.0000001, 2, z);
      std::cout << "[time_id]       pad 1.25: r_t " << id.size() << " -> " << idq.size() << ", E=0.8Emin " << qlo
                << ", E=1.2Emax " << qhi << "\n";
      CHECK(elo <= 1.0);   // information only: unconstrained outside the design range
      CHECK(ehi <= 1.0);
      CHECK(qlo <= 10.0 * eps);
      CHECK(qhi <= 10.0 * eps);
      CHECK(en <= std::max(10.0 * eps, 1e-9));
      CHECK(amp <= 1e-10);
    }
}

TEST_CASE("line_dlr_time_id_build_time", "[numerics][line_dlr][time_id]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z = id_targets(theta);
  std::cout << "\n[time_id] 5. construction time (both sectors, nodes + factorization + F for 280 targets)\n";
  for (auto const &c : id_cases)
    for (double eps : id_eps) {
      auto t0 = clk::now();
      ldlr::time_id_t idp(theta_t, sector_t::particle, c.emin, c.emax, eps);
      ldlr::time_id_t idh(theta_t, sector_t::hole, c.emin, c.emax, eps);
      const double tb = seconds_since(t0);
      auto t1         = clk::now();
      auto Fp = idp.transform_matrix(z);
      auto Fh = idh.transform_matrix(z);
      const double tf = seconds_since(t1);
      std::cout << "[time_id]   " << c.name << " eps " << std::setprecision(0) << eps << std::setprecision(2)
                << ": nE " << idp.nE() << ", candidates " << idp.n_cand << ", build " << tb << " s, F " << tf
                << " s\n";
      CHECK(tb + tf <= 5.0);
      CHECK(Fp.extent(1) == idp.size());
      CHECK(Fh.extent(1) == idh.size());
    }
  // the type-erased view accepts both node sets
  ldlr::time_id_t id(theta_t, sector_t::hole, 0.04, 6.0, 1e-8);
  auto ray = ldlr::time_ray_t::for_spectrum(theta_t, 0.04, 40.0, 1e-5, 3.0, 16, sector_t::hole);
  ldlr::time_nodes_t a(id), b(ray);
  CHECK(a.size() == id.size());
  CHECK(b.size() == ray.size());
  CHECK(a.sector == sector_t::hole);
  CHECK(max_abs_diff(a.transform_matrix(z), id.transform_matrix(z)) == 0.0);
  {   // the same arithmetic in two inlining contexts: gcc's fp-contract (FMA) may differ by an ulp (rusty, S8b.2)
    auto Fr = ray.transform_matrix(z);
    double fm = 0.0;
    for (auto const &x : Fr) fm = std::max(fm, std::abs(x));
    CHECK(max_abs_diff(b.transform_matrix(z), Fr) <= 1e-15 * fm);
  }
  // Gram eps-rank (Eq. gram, closed form) vs the QR rank
  // (only where eps^2 is above the double-precision floor of the Gram eigenvalues, ~1e-16 lambda_max)
  for (double eps : {1e-5, 1e-6, 1e-7}) {
    const long rg = ldlr::gram_rank(theta_t, 0.04, 6.0, eps, 600);
    const long rq = ldlr::time_id_t(theta_t, sector_t::particle, 0.04, 6.0, eps).rank;
    std::cout << "[time_id]   eps-rank [0.04, 6] eps " << std::setprecision(0) << eps << ": Gram " << rg << ", QR "
              << rq << "\n";
    CHECK(std::abs(rg - rq) <= 5 + rq / 5);
  }
}

// hidden: scaling of r_t, GL node counts for the same accuracy, accuracy vs number of nodes
TEST_CASE("line_dlr_time_id_scan", "[.time_id_scan]") {
  std::cout << std::scientific << std::setprecision(2);
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto zl = ldlr::dense_nodes(theta, 1e-3, 60.0, 120);
  struct row {
    double emin, emax, eps;
    long rank, r1, r125, ngl, ngl_def;
    double e1, e125, f1, f125, egl, egl_def;
  };
  std::vector<row> rows;
  // minimal GL quadrature (for_spectrum family) reaching 10 eps on the same metric
  auto gl_search = [&](double emin, double emax, double eps, double &err_out) {
    long best = -1;
    double best_err = 0.0;
    for (double smin : {1e-5, 1e-2 / emax})
      for (double pe : {1.0, 1.5, 2.0, 3.0})
        for (long nn : {6L, 8L, 10L, 12L, 16L, 20L}) {
          auto ray = ldlr::time_ray_t::for_spectrum(theta_t, emin, std::log(1.0 / eps) + 3.0, smin, pe, nn,
                                                    sector_t::particle);
          if (best > 0 and ray.size() >= best) continue;
          const double e = per_pole_err(ray, ray.transform_matrix(zl), emin, emax, 300, zl);
          if (e <= 10.0 * eps) {
            best     = ray.size();
            best_err = e;
          }
        }
    err_out = best_err;
    return best;
  };
  auto run = [&](double emin, double emax, double eps) {
    row r{emin, emax, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ldlr::time_id_opts_t o;
    ldlr::time_id_t a(theta_t, sector_t::particle, emin, emax, eps, o);
    o.oversample = 1.25;
    ldlr::time_id_t b(theta_t, sector_t::particle, emin, emax, eps, o);
    auto Fa = a.transform_matrix(zl), Fb = b.transform_matrix(zl);
    r.rank = a.rank;
    r.r1   = a.size();
    r.r125 = b.size();
    r.e1   = per_pole_err(a, Fa, emin, emax, 300, zl);
    r.e125 = per_pole_err(b, Fb, emin, emax, 300, zl);
    r.f1   = max_abs(Fa);
    r.f125 = max_abs(Fb);
    r.ngl  = gl_search(emin, emax, eps, r.egl);
    auto rd   = ldlr::time_ray_t::for_spectrum(theta_t, emin, 40.0, 1e-5, 3.0, 16, sector_t::particle);
    r.ngl_def = rd.size();
    r.egl_def = per_pole_err(rd, rd.transform_matrix(zl), emin, emax, 300, zl);
    rows.push_back(r);
    std::cout << "| " << std::setprecision(3) << std::defaultfloat << emin << " | " << emax << " | "
              << std::scientific << std::setprecision(0) << eps << std::setprecision(1) << " | " << r.rank << " | "
              << r.r1 << " | " << r.e1 << " | " << r.f1 << " | " << r.r125 << " | " << r.e125 << " | " << r.f125
              << " | " << r.ngl << " | " << r.egl << " | " << std::fixed << double(r.ngl) / double(r.r125)
              << std::scientific << " | " << r.ngl_def << " | " << r.egl_def << " |" << std::endl;
  };
  std::cout << "\n[time_id_scan] particle ray, theta=20 theta_t=10 deg, per-pole error on 240 line nodes\n"
            << "| Emin | Emax | eps | rank | r_t(1.0) | err | max|F| | r_t(1.25) | err | max|F| | GL min (10eps) | "
               "GL err | GL/ID(1.25) | GL default | GL def err |\n";
  for (double eps : id_eps) {
    for (double emax : {1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0, 100.0}) run(0.04, emax, eps);
    for (double emin : {0.2, 0.1, 0.01, 0.005, 0.001}) run(emin, 6.0, eps);
  }
  // fit rank = c L log(1/eps) (through the origin) and rank = a + c L log(1/eps), L = log(Emax/Emin)
  double sxx = 0, sxy = 0, sx = 0, sy = 0;
  const double n = double(rows.size());
  for (auto const &r : rows) {
    const double x = std::log(r.emax / r.emin) * std::log(1.0 / r.eps);
    sxx += x * x;
    sxy += x * double(r.rank);
    sx += x;
    sy += double(r.rank);
  }
  const double c0 = sxy / sxx;
  const double c1 = (n * sxy - sx * sy) / (n * sxx - sx * sx), a1 = (sy - c1 * sx) / n;
  double m0 = 0, m1 = 0;
  for (auto const &r : rows) {
    const double x = std::log(r.emax / r.emin) * std::log(1.0 / r.eps);
    m0 = std::max(m0, std::abs(double(r.rank) - c0 * x) / double(r.rank));
    m1 = std::max(m1, std::abs(double(r.rank) - a1 - c1 * x) / double(r.rank));
  }
  std::cout << std::fixed << std::setprecision(3) << "[time_id_scan] fit rank = c L ln(1/eps): c = " << c0
            << " (max rel dev " << m0 << ");  rank = a + c L ln(1/eps): a = " << a1 << ", c = " << c1
            << " (max rel dev " << m1 << ")\n";
  // two-parameter-in-logs fit: rank = a + b ln(1/eps) + c L + d L ln(1/eps)
  {
    nda::array<dcomplex, 2> A(rows.size(), 4), y(rows.size(), 1);
    for (size_t i = 0; i < rows.size(); ++i) {
      const double L = std::log(rows[i].emax / rows[i].emin), le = std::log(1.0 / rows[i].eps);
      A(i, 0) = 1.0;
      A(i, 1) = le;
      A(i, 2) = L;
      A(i, 3) = L * le;
      y(i, 0) = double(rows[i].rank);
    }
    auto p = ldlr::detail::lstsq(A, y);
    std::cout << "[time_id_scan] fit rank = a + b ln(1/eps) + c L + d L ln(1/eps): a=" << p(0, 0).real()
              << " b=" << p(1, 0).real() << " c=" << p(2, 0).real() << " d=" << p(3, 0).real() << "\n";
  }
  // accuracy vs number of nodes (forced pivot counts), Pi and Sigma ranges at eps = 1e-10 selection
  std::cout << std::scientific << std::setprecision(2)
            << "[time_id_scan] accuracy vs nodes (selection eps 1e-10, first r pivots), per-pole err / max|F|\n";
  for (auto const &c : id_cases) {
    ldlr::time_id_t base(theta_t, sector_t::particle, c.emin, c.emax, 1e-10);
    std::cout << "[time_id_scan]   " << c.name << " (rank " << base.rank << "):";
    for (double f : {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}) {
      ldlr::time_id_opts_t o;
      o.rank_force = long(std::ceil(f * double(base.rank)));
      ldlr::time_id_t id(theta_t, sector_t::particle, c.emin, c.emax, 1e-10, o);
      auto F = id.transform_matrix(zl);
      std::cout << "  r=" << id.size() << ": " << per_pole_err(id, F, c.emin, c.emax, 300, zl) << "/"
                << std::setprecision(0) << max_abs(F) << std::setprecision(2);
    }
    std::cout << "\n";
  }
}

} // namespace bdft_tests
