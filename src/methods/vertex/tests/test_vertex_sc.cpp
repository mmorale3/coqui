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
 * The mesh bookkeeping of the coarse-sublattice vertex self-consistency (vertex_view.hpp, vertex_sc.hpp) on a synthetic
 * Gamma-centred mesh: sublattices, restricted k views (internal points, partner points, rung transfers, W rows),
 * transfer classes and the real-space interpolation kernel. Algebra only -- no mean field; seconds to run.
 */

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "catch2/catch.hpp"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "methods/vertex/vertex_view.hpp"
#include "methods/vertex/vertex_sc.hpp"

namespace bdft_tests {

  namespace vv = methods::solvers::vertex_view;
  namespace vs = methods::solvers::vertex_sc;

  namespace {
    // an N x N x N Gamma-centred mesh: index = (i0 N + i1) N + i2, crystal coordinates i / N folded to [-1/2, 1/2);
    // the transfer mesh is the same list (q index = k index), k - q and k + q by index arithmetic
    struct mesh {
      long N;
      long n() const { return N * N * N; }
      std::array<long, 3> idx(long k) const { return {k / (N * N), (k / N) % N, k % N}; }
      long of(std::array<long, 3> i) const {
        for (auto &x : i) x = ((x % N) + N) % N;
        return (i[0] * N + i[1]) * N + i[2];
      }
      long kmq(long q, long k) const { auto a = idx(k), b = idx(q); return of({a[0] - b[0], a[1] - b[1], a[2] - b[2]}); }
      long kpq(long q, long k) const { auto a = idx(k), b = idx(q); return of({a[0] + b[0], a[1] + b[1], a[2] + b[2]}); }
      nda::array<double, 2> crys() const {
        nda::array<double, 2> c(n(), 3);
        for (long k = 0; k < n(); ++k) { auto i = idx(k); for (int a = 0; a < 3; ++a) c(k, a) = vs::fold(double(i[size_t(a)]) / double(N)); }
        return c;
      }
    };
    vv::k_view view_of(mesh const &m, std::vector<long> const &k_int, std::vector<long> const &q_ext) {
      return vv::make_view(m.n(), m.n(), m.n(), k_int, q_ext, [&](long q, long k) { return m.kmq(q, k); },
                           [&](long q, long k) { return m.kpq(q, k); }, [](long q) { return q; });
    }
  }

  TEST_CASE("vertex_sc_views", "[methods][vertex][vertex_sc]") {
    const mesh m{4};
    const auto kc = m.crys();
    std::vector<long> all_q(static_cast<size_t>(m.n()));
    for (long q = 0; q < m.n(); ++q) all_q[size_t(q)] = q;

    SECTION("identity view") {
      const auto k_int = vv::sublattice(kc, {4, 4, 4});
      REQUIRE(long(k_int.size()) == m.n());
      const auto v = view_of(m, k_int, all_q);
      REQUIRE(v.nk_int == m.n());
      REQUIRE(v.nk_arr() == m.n());                 // every partner is an internal point
      REQUIRE(v.n_wrows() == m.n());                // every transfer is a rung transfer
      for (long e = 0; e < v.n_ext(); ++e)
        for (long i = 0; i < v.nk_int; ++i) REQUIRE(v.kpts(v.kpq(e, i)) == m.kpq(all_q[size_t(e)], k_int[size_t(i)]));
      for (long i = 0; i < v.nk_int; ++i)
        for (long j = 0; j < v.nk_int; ++j) REQUIRE(m.kmq(v.qx_full(i, j), k_int[size_t(i)]) == k_int[size_t(j)]);
    }

    SECTION("sublattice view with external transfers off the sublattice") {
      const auto k_int = vv::sublattice(kc, {2, 2, 2});
      REQUIRE(k_int.size() == 8);
      // a transfer of the sublattice: closed view
      const auto vc = view_of(m, k_int, {k_int[3]});
      REQUIRE(vc.nk_arr() == 8);
      REQUIRE(vc.n_wrows() == 8);                   // the rung transfers are the 8 sublattice transfers
      // a transfer off the sublattice: the partners are 8 new points
      const long q_off = m.of({1, 0, 0});
      const auto vo = view_of(m, k_int, {q_off});
      REQUIRE(vo.nk_int == 8);
      REQUIRE(vo.nk_arr() == 16);
      REQUIRE(vo.n_wrows() == 8);                   // the rungs still read sublattice transfers only
      for (long i = 0; i < 8; ++i) {
        REQUIRE(vo.kpq(0, i) >= 8);                 // partner points follow the internal ones
        REQUIRE(vo.kpts(vo.kpq(0, i)) == m.kpq(q_off, k_int[size_t(i)]));
      }
      for (long r : vo.wrows) {                     // W rows = transfers with even indices
        auto i = m.idx(r);
        REQUIRE((i[0] % 2 == 0 and i[1] % 2 == 0 and i[2] % 2 == 0));
      }
      for (long i = 0; i < 8; ++i)
        for (long j = 0; j < 8; ++j) REQUIRE(m.kmq(vo.qx_full(i, j), k_int[size_t(i)]) == k_int[size_t(j)]);
    }

    SECTION("transfer classes") {
      const auto cls = vs::transfer_classes(kc, all_q, {2, 2, 2});
      REQUIRE(cls.size() == 8);                     // (4/2)^3 classes of 8 transfers
      long tot = 0;
      for (auto const &c : cls) {
        REQUIRE(c.size() == 8);
        tot += long(c.size());
        for (long q : c) {                          // members differ by sublattice transfers
          auto a = m.idx(q), b = m.idx(c[0]);
          for (int x = 0; x < 3; ++x) REQUIRE(((a[size_t(x)] - b[size_t(x)]) % 2 + 2) % 2 == 0);
        }
      }
      REQUIRE(tot == m.n());
      REQUIRE(vs::transfer_classes(kc, all_q, {4, 4, 4}).size() == 1);
    }
  }

  TEST_CASE("vertex_sc_interpolation", "[methods][vertex][vertex_sc]") {
    const mesh mf{4};
    const auto qf = mf.crys();
    // a non-orthogonal lattice (fcc primitive vectors, rows)
    nda::array<double, 2> lattv = {{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    const std::array<long, 3> sub = {2, 2, 2};
    const auto q_sub = vs::sublattice_transfers(qf, sub);
    REQUIRE(q_sub.size() == 8);
    nda::array<double, 2> qc(8, 3);
    for (long c = 0; c < 8; ++c)
      for (int a = 0; a < 3; ++a) qc(c, a) = qf(q_sub[size_t(c)], a);
    const auto K = vs::interpolation_kernel(lattv, sub, qc, qf);
    REQUIRE(K.shape(0) == mf.n());
    REQUIRE(K.shape(1) == 8);

    SECTION("exact on the sublattice, partition of unity, time reversal") {
      for (long c = 0; c < 8; ++c)
        for (long d = 0; d < 8; ++d) {
          const double ref = (c == d) ? 1.0 : 0.0;
          REQUIRE(std::abs(K(q_sub[size_t(c)], d) - ref) < 1e-12);
        }
      for (long f = 0; f < mf.n(); ++f) {
        std::complex<double> s(0.0);
        for (long c = 0; c < 8; ++c) s += K(f, c);
        REQUIRE(std::abs(s - 1.0) < 1e-12);         // a constant is reproduced
      }
      // K(-q_f, -q_c) = conj(K(q_f, q_c))
      for (long f = 0; f < mf.n(); ++f) {
        auto i = mf.idx(f);
        const long fm = mf.of({-i[0], -i[1], -i[2]});
        for (long c = 0; c < 8; ++c) {
          auto j = mf.idx(q_sub[size_t(c)]);
          const long qm = mf.of({-j[0], -j[1], -j[2]});
          long cm = -1;
          for (long d = 0; d < 8; ++d) if (q_sub[size_t(d)] == qm) cm = d;
          REQUIRE(cm >= 0);
          REQUIRE(std::abs(K(fm, cm) - std::conj(K(f, c))) < 1e-12);
        }
      }
    }

    SECTION("hybrid: the directly solved set (radius, shells, count; minimum image)") {
      // the fine mesh itself plays the IBZ list; |q| from the reciprocal vectors b = 2 pi (a^-1)^T
      const auto b = vs::reciprocal_vectors(lattv);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          double s = 0.0;
          for (int x = 0; x < 3; ++x) s += lattv(i, x) * b[size_t(j)][size_t(x)];
          REQUIRE(std::abs(s - (i == j ? 2.0 * M_PI : 0.0)) < 1e-12);   // a_i . b_j = 2 pi delta_ij
        }
      std::vector<double> qn(static_cast<size_t>(mf.n()));
      for (long q = 0; q < mf.n(); ++q) qn[size_t(q)] = vs::min_image_norm(b, qf(q, 0), qf(q, 1), qf(q, 2));
      double dq_c = 1e300, q1 = 1e300;
      for (int i = 0; i < 3; ++i) {
        double n2 = 0.0;
        for (int j = 0; j < 3; ++j) n2 += b[size_t(i)][size_t(j)] * b[size_t(i)][size_t(j)];
        dq_c = std::min(dq_c, std::sqrt(n2) / double(sub[size_t(i)]));
      }
      for (long q = 0; q < mf.n(); ++q) if (qn[size_t(q)] > 1e-12) q1 = std::min(q1, qn[size_t(q)]);
      vs::config cfg;
      cfg.mode = "hybrid"; cfg.mesh = sub;
      // radius: brute force with the same norm
      cfg.direct_select = "radius"; cfg.direct_radius = 1.0;
      auto fr = vs::direct_set(cfg, lattv, qf, mf.n());
      long nd = 0;
      for (long q = 0; q < mf.n(); ++q) { REQUIRE(bool(fr[size_t(q)]) == (qn[size_t(q)] <= dq_c * (1.0 + 1e-9))); nd += fr[size_t(q)]; }
      REQUIRE(nd > 1);                                // Gamma and at least one shell (the fine spacing is half the coarse one)
      REQUIRE(nd < mf.n());
      // shells = 1: Gamma and the nearest shell exactly
      cfg.direct_select = "shells"; cfg.direct_shells = 1;
      auto fs = vs::direct_set(cfg, lattv, qf, mf.n());
      for (long q = 0; q < mf.n(); ++q) REQUIRE(bool(fs[size_t(q)]) == (qn[size_t(q)] <= q1 * (1.0 + 1e-9)));
      cfg.direct_shells = 0;                          // Gamma alone
      auto f0 = vs::direct_set(cfg, lattv, qf, mf.n());
      for (long q = 0; q < mf.n(); ++q) REQUIRE(bool(f0[size_t(q)]) == (qn[size_t(q)] < 1e-12));
      // count: completed to whole shells (count 2 = Gamma + the whole first shell)
      cfg.direct_select = "count"; cfg.direct_count = 2;
      auto fc = vs::direct_set(cfg, lattv, qf, mf.n());
      for (long q = 0; q < mf.n(); ++q) REQUIRE(fc[size_t(q)] == fs[size_t(q)]);
      cfg.direct_count = 0;
      for (char f : vs::direct_set(cfg, lattv, qf, mf.n())) REQUIRE(f == 0);
      // the minimum image: the same points given in [0, 1) select the same set
      nda::array<double, 2> qu(mf.n(), 3);
      for (long q = 0; q < mf.n(); ++q) for (int a = 0; a < 3; ++a) qu(q, a) = qf(q, a) - std::floor(qf(q, a));
      cfg.direct_select = "radius"; cfg.direct_radius = 1.0;
      auto fu = vs::direct_set(cfg, lattv, qu, mf.n());
      for (long q = 0; q < mf.n(); ++q) REQUIRE(fu[size_t(q)] == fr[size_t(q)]);
    }

    SECTION("a function of the first shell is reproduced") {
      // f(q) = sum over the nearest lattice vectors R of cos(2 pi q.R): its real-space support lies inside the
      // Wigner-Seitz cell of the 2 x 2 x 2 supercell only for |R| = one primitive vector with weight shared across the
      // zone boundary images, so the interpolant equals f on the sublattice and stays bounded by its range elsewhere
      auto fq = [&](long q) {
        double s = 0.0;
        for (int a = 0; a < 3; ++a) s += std::cos(2.0 * M_PI * qf(q, a));
        return s;
      };
      for (long f = 0; f < mf.n(); ++f) {
        std::complex<double> s(0.0);
        for (long c = 0; c < 8; ++c) s += K(f, c) * fq(q_sub[size_t(c)]);
        REQUIRE(std::abs(s.imag()) < 1e-12);
        REQUIRE(std::abs(s.real()) <= 3.0 + 1e-12);
      }
    }
  }

} // bdft_tests
