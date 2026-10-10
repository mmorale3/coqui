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

#ifndef COQUI_NUMERICS_LINE_DLR_UTILS_HPP
#define COQUI_NUMERICS_LINE_DLR_UTILS_HPP

/**
 * Small shared pieces of the line-DLR numerics (notes/line_gw/line_gw_notes.tex, sections 3-4):
 * the sector tag, numpy-compatible log grids, column-pivoted QR (geqp3) and the least-squares solve (gelss)
 * with numpy's default cutoff, and Gauss-Legendre nodes. Python oracle: coqui/cayley/cayley/line/.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "utilities/check.hpp"

namespace numerics::line_dlr {

/// Sector of a function on the line: both, particle ('>', poles above mu) or hole ('<', poles below mu).
enum class sector_t { both, particle, hole };

namespace detail {

/// exp(np.linspace(log a, log b, n)), reproducing numpy's linspace arithmetic (y_i = start + i*step, y_{n-1} = stop).
inline nda::array<double, 1> logspace(double a, double b, long n) {
  utils::check(n >= 1 and a > 0.0 and b > 0.0, "line_dlr::logspace: invalid arguments a={} b={} n={}", a, b, n);
  nda::array<double, 1> y(n);
  const double start = std::log(a), stop = std::log(b);
  if (n == 1) { y(0) = std::exp(start); return y; }
  const double step = (stop - start) / double(n - 1);
  for (long i = 0; i < n; ++i) y(i) = std::exp(double(i) * step + start);
  y(n - 1) = std::exp(stop);
  return y;
}

/// np.linspace(a, b, n) (y_i = a + i step, y_{n-1} = b), S8b (guarded-ray panels, wedge band)
inline nda::array<double, 1> linspace(double a, double b, long n) {
  utils::check(n >= 1, "line_dlr::linspace: n = {}", n);
  nda::array<double, 1> y(n);
  if (n == 1) { y(0) = a; return y; }
  const double step = (b - a) / double(n - 1);
  for (long i = 0; i < n; ++i) y(i) = double(i) * step + a;
  y(n - 1) = b;
  return y;
}

/// Result of a column-pivoted QR: pivots (0-based, "column l of A P was column piv[l] of A") and |R_ll|.
struct pivoted_qr_t {
  std::vector<long> piv;
  std::vector<double> rdiag;
};

/// Column-pivoted QR (LAPACK geqp3) of A; A is overwritten.
inline pivoted_qr_t pivoted_qr(nda::matrix<ComplexType, nda::F_layout> &A) {
  const long m = A.extent(0), n = A.extent(1), k = std::min(m, n);
  nda::array<int, 1> jpvt(n);
  jpvt() = 0;                                   // all columns free
  nda::array<ComplexType, 1> tau(k);
  nda::array<ComplexType, 1> work(1);           // 4-arg overload: work is resized to the optimal size
  nda::lapack::geqp3(A, jpvt, tau, work);       // jpvt is returned 0-based
  pivoted_qr_t out;
  out.piv.resize(n);
  for (long j = 0; j < n; ++j) out.piv[j] = long(jpvt(j));
  out.rdiag.resize(k);
  for (long l = 0; l < k; ++l) out.rdiag[l] = std::abs(A(l, l));
  return out;
}

/// numerical rank r = #{ |R_ll| > eps |R_00| } (python: (d > eps * d[0]).sum())
inline long qr_rank(pivoted_qr_t const &q, double eps) {
  long r = 0;
  for (double d : q.rdiag)
    if (d > eps * q.rdiag[0]) ++r;
  return r;
}

/// Least squares min ||A X - B|| by complex gelss (SVD), rcond < 0 -> DBL_EPSILON * max(m, n) (numpy's lstsq default).
/// A: [m, n], B: [m, nrhs] (any layout); returns X [n, nrhs].
template <typename MA, typename MB>
nda::array<ComplexType, 2> lstsq(MA const &A, MB const &B, double rcond = -1.0, int *rank_out = nullptr) {
  const long m = A.extent(0), n = A.extent(1), nrhs = B.extent(1);
  utils::check(B.extent(0) == m, "line_dlr::lstsq: row mismatch {} vs {}", B.extent(0), m);
  if (rcond < 0.0) rcond = std::numeric_limits<double>::epsilon() * double(std::max(m, n));
  nda::matrix<ComplexType, nda::F_layout> Af(m, n);
  for (long j = 0; j < n; ++j)
    for (long i = 0; i < m; ++i) Af(i, j) = A(i, j);
  const long mb = std::max(m, n);
  nda::matrix<ComplexType, nda::F_layout> Bf(mb, nrhs);
  Bf() = ComplexType(0.0);
  for (long j = 0; j < nrhs; ++j)
    for (long i = 0; i < m; ++i) Bf(i, j) = B(i, j);
  nda::array<double, 1> sv(std::min(m, n));
  int rank = 0;
  nda::lapack::gelss(Af, Bf, sv, rcond, rank);
  if (rank_out) *rank_out = rank;
  nda::array<ComplexType, 2> X(n, nrhs);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < nrhs; ++j) X(i, j) = Bf(i, j);
  return X;
}

/// C = A B for C-layout rank-2 arrays (BLAS gemm).
inline nda::array<ComplexType, 2> matmul(nda::array<ComplexType, 2> const &A, nda::array<ComplexType, 2> const &B) {
  utils::check(A.extent(1) == B.extent(0), "line_dlr::matmul: inner dimension mismatch {} vs {}", A.extent(1), B.extent(0));
  nda::array<ComplexType, 2> C(A.extent(0), B.extent(1));
  if (A.extent(1) == 0) { C() = ComplexType(0.0); return C; }
  nda::blas::gemm(ComplexType(1.0), A, B, ComplexType(0.0), C);
  return C;
}

/// Gauss-Legendre nodes (ascending) and weights on [-1, 1] by Newton iteration on P_n (double precision, n <= 64).
inline void gauss_legendre(long n, nda::array<double, 1> &x, nda::array<double, 1> &w) {
  utils::check(n >= 1 and n <= 64, "line_dlr::gauss_legendre: n={} out of range [1, 64]", n);
  x.resize(n);
  w.resize(n);
  auto legendre = [n](double z, double &dp) {        // returns P_n(z), dp = P_n'(z)
    double p0 = 1.0, p1 = z;
    if (n == 1) { dp = 1.0; return z; }
    for (long k = 2; k <= n; ++k) {
      const double p2 = (double(2 * k - 1) * z * p1 - double(k - 1) * p0) / double(k);
      p0 = p1;
      p1 = p2;
    }
    dp = double(n) * (z * p1 - p0) / (z * z - 1.0);
    return p1;
  };
  const long nh = (n + 1) / 2;
  for (long i = 0; i < nh; ++i) {
    double z = std::cos(std::numbers::pi * (double(i) + 0.75) / (double(n) + 0.5)), dp = 0.0;
    if (n % 2 == 1 and i == nh - 1) z = 0.0;           // middle root of odd n is exactly 0
    else {
      for (int it = 0; it < 100; ++it) {
        const double p  = legendre(z, dp);
        const double dz = p / dp;
        z -= dz;
        if (std::abs(dz) < 1e-16) break;
      }
    }
    if (n == 1) dp = 1.0;
    else legendre(z, dp);
    const double wi = 2.0 / ((1.0 - z * z) * dp * dp);
    x(n - 1 - i) = z;
    x(i)         = -z;
    w(i) = w(n - 1 - i) = wi;
  }
}

} // namespace detail
} // namespace numerics::line_dlr

#endif
