/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Unit tests for `detail_qp::align_mo_to_prev`. Synthesizes small
 * (s, k, nbnd, nbnd) MO arrays and arbitrary unitary rotations within
 * degenerate ε-clusters, then verifies the alignment recovers MO_prev
 * up to numerical noise. Also exercises the singleton (1×1) phase-fix
 * branch and the all-non-degenerate (no-op-modulo-phase) case.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "methods/GW_real_axis/real_axis_qp_scf_driver.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests {

using cval_t = std::complex<double>;
using methods::real_axis::detail_qp::align_mo_to_prev;

namespace {

  // Build a Haar-ish random unitary as exp(i H) with a simple Hermitian H
  // seeded from `seed`. Small dimensions only.
  nda::matrix<cval_t> small_unitary(long m, unsigned seed)
  {
    nda::matrix<cval_t> H(m, m);
    H = cval_t(0.0, 0.0);
    auto rnd = [&](unsigned& s) {
      s = s * 1664525u + 1013904223u;
      return (static_cast<double>(s) / 4294967295.0) - 0.5;
    };
    unsigned s = seed;
    for (long i = 0; i < m; ++i) {
      H(i, i) = cval_t(rnd(s), 0.0);
      for (long j = i + 1; j < m; ++j) {
        const double re = rnd(s);
        const double im = rnd(s);
        H(i, j) = cval_t(re,  im);
        H(j, i) = cval_t(re, -im);
      }
    }
    // U = sum_n exp(i lambda_n) |n><n|; since H is small + Hermitian, do
    // a direct power-series exp(i H) truncated at high order. With ||H||
    // bounded by ~m the series converges in ~30 terms.
    nda::matrix<cval_t> iH(m, m);
    for (long i = 0; i < m; ++i)
      for (long j = 0; j < m; ++j)
        iH(i, j) = cval_t(0.0, 1.0) * H(i, j);
    nda::matrix<cval_t> U(m, m), term(m, m);
    U = cval_t(0.0, 0.0);
    for (long i = 0; i < m; ++i) U(i, i) = cval_t(1.0, 0.0);
    term = U;
    for (int k = 1; k < 40; ++k) {
      nda::matrix<cval_t> next(m, m);
      next = cval_t(0.0, 0.0);
      for (long i = 0; i < m; ++i)
        for (long j = 0; j < m; ++j) {
          cval_t acc(0.0, 0.0);
          for (long l = 0; l < m; ++l) acc += term(i, l) * iH(l, j);
          next(i, j) = acc / static_cast<double>(k);
        }
      term = next;
      U = U + term;
    }
    return U;
  }

  // Frobenius norm of A - B for rank-2 arrays.
  double frob_diff(nda::matrix<cval_t> const& A, nda::matrix<cval_t> const& B)
  {
    double s = 0.0;
    const long n0 = A.shape()[0], n1 = A.shape()[1];
    for (long i = 0; i < n0; ++i)
      for (long j = 0; j < n1; ++j)
        s += std::norm(A(i, j) - B(i, j));
    return std::sqrt(s);
  }

} // anonymous namespace

TEST_CASE("align_mo_to_prev_degenerate_pair_recovers_basis",
          "[real_axis][mo_align]")
{
  // 4-band system; bands 1-2 are exactly degenerate at ε=0.5,
  // bands 0 and 3 are well-separated. MO_new = MO_prev * R_block where
  // R_block = block_diag(I_1, R_2, I_1) for some 2×2 unitary R_2.
  // Alignment should recover MO_new == MO_prev (up to noise).
  const long ns = 1, Nk = 1, nbnd = 4;
  const double dE_cluster = 1e-3;

  nda::matrix<cval_t> MO_prev_mat(nbnd, nbnd);
  MO_prev_mat = small_unitary(nbnd, /*seed*/ 7);

  // Embed into rank-4 (s, k, i, n).
  nda::array<cval_t, 4> MO_prev(ns, Nk, nbnd, nbnd);
  nda::array<cval_t, 4> MO_new (ns, Nk, nbnd, nbnd);
  for (long i = 0; i < nbnd; ++i)
    for (long n = 0; n < nbnd; ++n) {
      MO_prev(0, 0, i, n) = MO_prev_mat(i, n);
      MO_new (0, 0, i, n) = MO_prev_mat(i, n);
    }

  // R_block: identity on (0) and (3); 2×2 unitary on cluster (1,2).
  auto R2 = small_unitary(2, /*seed*/ 13);
  nda::matrix<cval_t> R_block(nbnd, nbnd);
  R_block = cval_t(0.0, 0.0);
  R_block(0, 0) = cval_t(1.0, 0.0);
  R_block(3, 3) = cval_t(1.0, 0.0);
  R_block(1, 1) = R2(0, 0);
  R_block(1, 2) = R2(0, 1);
  R_block(2, 1) = R2(1, 0);
  R_block(2, 2) = R2(1, 1);

  // MO_new = MO_prev · R_block.
  for (long i = 0; i < nbnd; ++i)
    for (long n = 0; n < nbnd; ++n) {
      cval_t acc(0.0, 0.0);
      for (long m = 0; m < nbnd; ++m)
        acc += MO_prev_mat(i, m) * R_block(m, n);
      MO_new(0, 0, i, n) = acc;
    }

  // Eigenvalues: 0.0, 0.5, 0.5, 1.0  (cluster of 2 at 0.5).
  nda::array<cval_t, 3> E(ns, Nk, nbnd);
  E(0, 0, 0) = cval_t(0.0, 0.0);
  E(0, 0, 1) = cval_t(0.5, 0.0);
  E(0, 0, 2) = cval_t(0.5, 0.0);
  E(0, 0, 3) = cval_t(1.0, 0.0);

  align_mo_to_prev(MO_new, MO_prev, E, dE_cluster, ns, Nk, nbnd);

  // Each column should match MO_prev up to a global phase (exact for
  // the degenerate cluster, modulo numerical noise; pure phase fix for
  // the singletons).
  for (long n = 0; n < nbnd; ++n) {
    cval_t alpha(0.0, 0.0);
    for (long i = 0; i < nbnd; ++i)
      alpha += std::conj(MO_prev(0, 0, i, n)) * MO_new(0, 0, i, n);
    // |alpha| should be 1; alignment should leave alpha real-positive.
    REQUIRE(std::abs(alpha) == Approx(1.0).margin(1e-10));
    REQUIRE(alpha.imag()    == Approx(0.0).margin(1e-10));
    REQUIRE(alpha.real()    == Approx(1.0).margin(1e-10));
  }
}

TEST_CASE("align_mo_to_prev_singleton_phase_fix",
          "[real_axis][mo_align]")
{
  // No degeneracies: align should reduce to per-column phase fix.
  const long ns = 1, Nk = 1, nbnd = 3;
  const double dE_cluster = 1e-3;

  auto U_prev = small_unitary(nbnd, /*seed*/ 21);

  nda::array<cval_t, 4> MO_prev(ns, Nk, nbnd, nbnd);
  nda::array<cval_t, 4> MO_new (ns, Nk, nbnd, nbnd);
  for (long i = 0; i < nbnd; ++i)
    for (long n = 0; n < nbnd; ++n) {
      MO_prev(0, 0, i, n) = U_prev(i, n);
      // Apply per-column random phases.
      const double theta = 0.7 * static_cast<double>(n + 1);
      const cval_t phase = std::polar(1.0, theta);
      MO_new (0, 0, i, n) = U_prev(i, n) * phase;
    }

  // Distinct eigenvalues: 0.0, 0.5, 1.0.
  nda::array<cval_t, 3> E(ns, Nk, nbnd);
  E(0, 0, 0) = cval_t(0.0, 0.0);
  E(0, 0, 1) = cval_t(0.5, 0.0);
  E(0, 0, 2) = cval_t(1.0, 0.0);

  align_mo_to_prev(MO_new, MO_prev, E, dE_cluster, ns, Nk, nbnd);

  // After alignment, MO_new should equal MO_prev within tight tolerance.
  for (long n = 0; n < nbnd; ++n) {
    cval_t alpha(0.0, 0.0);
    for (long i = 0; i < nbnd; ++i)
      alpha += std::conj(MO_prev(0, 0, i, n)) * MO_new(0, 0, i, n);
    REQUIRE(std::abs(alpha)  == Approx(1.0).margin(1e-12));
    REQUIRE(alpha.real()     == Approx(1.0).margin(1e-12));
    REQUIRE(alpha.imag()     == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("align_mo_to_prev_triplet_recovery",
          "[real_axis][mo_align]")
{
  // 3-fold degenerate cluster (Si Γ_25′ pattern): all three eigenvalues
  // equal. MO_new is MO_prev rotated by an arbitrary 3×3 unitary on
  // the cluster. Alignment must recover MO_prev exactly.
  const long ns = 1, Nk = 1, nbnd = 5;
  const double dE_cluster = 1e-3;

  auto U_prev = small_unitary(nbnd, /*seed*/ 31);

  nda::array<cval_t, 4> MO_prev(ns, Nk, nbnd, nbnd);
  nda::array<cval_t, 4> MO_new (ns, Nk, nbnd, nbnd);

  // R3 acts on bands {1, 2, 3}.
  auto R3 = small_unitary(3, /*seed*/ 41);
  nda::matrix<cval_t> R_block(nbnd, nbnd);
  R_block = cval_t(0.0, 0.0);
  R_block(0, 0) = cval_t(1.0, 0.0);
  R_block(4, 4) = cval_t(1.0, 0.0);
  for (long a = 0; a < 3; ++a)
    for (long b = 0; b < 3; ++b)
      R_block(1 + a, 1 + b) = R3(a, b);

  for (long i = 0; i < nbnd; ++i)
    for (long n = 0; n < nbnd; ++n) {
      MO_prev(0, 0, i, n) = U_prev(i, n);
      cval_t acc(0.0, 0.0);
      for (long m = 0; m < nbnd; ++m)
        acc += U_prev(i, m) * R_block(m, n);
      MO_new (0, 0, i, n) = acc;
    }

  // Eigenvalues: 0.0, 0.5, 0.5, 0.5, 1.0.
  nda::array<cval_t, 3> E(ns, Nk, nbnd);
  E(0, 0, 0) = cval_t(0.0, 0.0);
  E(0, 0, 1) = cval_t(0.5, 0.0);
  E(0, 0, 2) = cval_t(0.5, 0.0);
  E(0, 0, 3) = cval_t(0.5, 0.0);
  E(0, 0, 4) = cval_t(1.0, 0.0);

  align_mo_to_prev(MO_new, MO_prev, E, dE_cluster, ns, Nk, nbnd);

  // Frobenius diff over the whole MO array should be ~0.
  double diff_F = 0.0;
  for (long i = 0; i < nbnd; ++i)
    for (long n = 0; n < nbnd; ++n) {
      const cval_t d = MO_new(0, 0, i, n) - MO_prev(0, 0, i, n);
      diff_F += std::norm(d);
    }
  diff_F = std::sqrt(diff_F);
  REQUIRE(diff_F == Approx(0.0).margin(1e-10));
}

} // namespace gw_real_axis_tests
