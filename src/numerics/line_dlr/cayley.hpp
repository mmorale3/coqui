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

#ifndef COQUI_NUMERICS_LINE_DLR_CAYLEY_HPP
#define COQUI_NUMERICS_LINE_DLR_CAYLEY_HPP

/**
 * Closure of the line scGW (notes/line_gw/line_gw_notes.tex section 6, Eqs. moments, upfold, Htilde; section 7, Eq. A):
 * Cayley moments of a real-pole Sigma_c, block-Toeplitz unitary realization ("upfolding"), Sigma_c / G / A(omega) from the
 * upfolded poles, the Lehmann G from the upfolded Hamiltonian and the T=0 chemical potential (widest admissible QP gap).
 *
 * Transcription of the python oracle coqui/cayley/cayley/: maps.py (cayley, inv_cayley), moments.py (moments_from_poles,
 * bound_check), upfold.py (block_toeplitz, normalize_c0, upfold_block), spectral.py (sigma_from_poles, greens_function,
 * spectral_function, upfolded_hamiltonian, lehmann_from_upfolded), line/closure.py (chemical_potential).
 *
 * Energy convention: EVERYTHING here is mu-relative (python called with mu = 0, as line/closure.py does). The moments are
 * taken about the centre omega = 0; upfold_block returns poles d_l = wp cot(theta_l / 2) relative to that centre; H must be
 * passed as H_stat - mu and frequencies as omega - mu. The caller adds mu back where absolute energies are needed.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "nda/linalg/det_and_inverse.hpp"

/*
 * LAPACK routines without an nda wrapper: Hermitian divide-and-conquer eigensolver (zheevd), the complex Schur form
 * (zgees), the divide-and-conquer SVD (zgesdd) and the LU solve (zgetrf / zgetrs). Standard Fortran-77 signatures (LP64 integers, trailing hidden string lengths omitted as nda's own interface
 * does); provided by OpenBLAS/Accelerate (Mac) and MKL (rusty).
 */
namespace numerics::line_dlr::detail::f77 {
extern "C" {
void zheevd_(const char *jobz, const char *uplo, const int *n, std::complex<double> *a, const int *lda, double *w,
             std::complex<double> *work, const int *lwork, double *rwork, const int *lrwork, int *iwork, const int *liwork,
             int *info);
void zgees_(const char *jobvs, const char *sort, int (*select)(const std::complex<double> *), const int *n,
            std::complex<double> *a, const int *lda, int *sdim, std::complex<double> *w, std::complex<double> *vs,
            const int *ldvs, std::complex<double> *work, const int *lwork, double *rwork, int *bwork, int *info);
void zgesdd_(const char *jobz, const int *m, const int *n, std::complex<double> *a, const int *lda, double *s,
             std::complex<double> *u, const int *ldu, std::complex<double> *vt, const int *ldvt, std::complex<double> *work,
             const int *lwork, double *rwork, int *iwork, int *info);
void zgetrf_(const int *m, const int *n, std::complex<double> *a, const int *lda, int *ipiv, int *info);
void zgetrs_(const char *trans, const int *n, const int *nrhs, const std::complex<double> *a, const int *lda, const int *ipiv,
             std::complex<double> *b, const int *ldb, int *info);
}
} // namespace numerics::line_dlr::detail::f77

namespace numerics::line_dlr {

using cmatrix_F = nda::matrix<ComplexType, nda::F_layout>;

// ------------------------------------------------------------------------------------------------------------------
// Cayley map (maps.py)
// ------------------------------------------------------------------------------------------------------------------

/// u(omega) = (omega + i wp)/(omega - i wp), omega mu-relative: real axis -> unit circle, 0 -> -1, +-inf -> +1.
template <typename T> inline ComplexType cayley_map(T omega, double wp) {
  const ComplexType w(omega);
  return (w + ComplexType(0.0, wp)) / (w - ComplexType(0.0, wp));
}

/// Real (mu-relative) pole position of a unit-circle point u = e^{i theta}: wp cot(theta/2), theta = arg u in (-pi, pi].
inline double cayley_inverse(ComplexType u, double wp) { return wp / std::tan(std::arg(u) / 2.0); }

// ------------------------------------------------------------------------------------------------------------------
// Moments (moments.py)
// ------------------------------------------------------------------------------------------------------------------

/// C^(m) = sum_l g_l u(w_l)^m, m = 0..nmax (Eq. moments). w: [r] mu-relative real poles, g: [r, n, n]. Returns [nmax+1, n, n].
inline nda::array<ComplexType, 3> moments_from_poles(nda::array<double, 1> const &w, nda::array<ComplexType, 3> const &g,
                                                     double wp, long nmax) {
  const long r = w.size(), n = g.extent(1);
  utils::check(g.extent(0) == r and g.extent(2) == n, "cayley::moments_from_poles: shape mismatch");
  nda::array<ComplexType, 2> V(nmax + 1, r);
  for (long l = 0; l < r; ++l) {
    const ComplexType u = cayley_map(w(l), wp);
    ComplexType un(1.0);
    for (long m = 0; m <= nmax; ++m) { V(m, l) = un; un *= u; }
  }
  nda::array<ComplexType, 2> G(r, n * n);
  for (long l = 0; l < r; ++l)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) G(l, i * n + j) = g(l, i, j);
  auto VG = detail::matmul(V, G);
  nda::array<ComplexType, 3> C(nmax + 1, n, n);
  for (long m = 0; m <= nmax; ++m)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) C(m, i, j) = VG(m, i * n + j);
  return C;
}

namespace detail {
template <typename M> double frob(M const &A) {
  double s = 0.0;
  for (long i = 0; i < A.extent(0); ++i)
    for (long j = 0; j < A.extent(1); ++j) s += std::norm(A(i, j));
  return std::sqrt(s);
}
} // namespace detail

/// bound_check: max_n ||C^(n)||_F / ||C^(0)||_F (<= 1 for a positive measure; = 1 for exact moments, attained at n = 0).
inline double bound_check(nda::array<ComplexType, 3> const &C) {
  const double n0 = detail::frob(C(0, nda::range::all, nda::range::all));
  double mx = 0.0;
  for (long m = 0; m < C.extent(0); ++m) mx = std::max(mx, detail::frob(C(m, nda::range::all, nda::range::all)));
  return mx / n0;
}

/// bound_check - 1: how far the moment sequence is from the positivity bound (0 for exact moments).
inline double moment_bound_violation(nda::array<ComplexType, 3> const &C) { return bound_check(C) - 1.0; }

// ------------------------------------------------------------------------------------------------------------------
// dense linear algebra helpers (F layout)
// ------------------------------------------------------------------------------------------------------------------

/**
 * S7g: optional external drivers for the large dense problems of the closure (GW_line installs the cuSOLVER ones of
 * methods/GW_line/cuda in device builds). Each returns false -- leaving its arguments untouched -- to fall back to the
 * host LAPACK path; matrices with fewer than min_dim rows always stay on the host.
 */
struct lapack_hooks_t {
  std::function<bool(cmatrix_F &, nda::array<double, 1> &)> heevd;   ///< A <- eigenvectors (ascending eigenvalues w)
  std::function<bool(cmatrix_F &, nda::array<double, 1> &, cmatrix_F &, cmatrix_F &)> gesvd;   ///< A = P diag(s) Qh, square
  std::function<bool(cmatrix_F &, cmatrix_F &)> lu_solve;            ///< B <- A^{-1} B (A may be overwritten on success)
  long min_dim = 256;
  bool in_ueig = true;   ///< also for the LU solve + Hermitian eigen of the "cayley" U-eigen path
};

namespace detail {

/// Hermitian eigen-decomposition by zheevd: A is overwritten by the eigenvectors (columns), eigenvalues ascending.
inline nda::array<double, 1> herm_eig(cmatrix_F &A) {
  const int n = int(A.extent(0));
  utils::check(A.extent(1) == n, "cayley::herm_eig: matrix not square");
  nda::array<double, 1> lam(n);
  if (n == 0) return lam;
  int info = 0, lwork = -1, lrwork = -1, liwork = -1, iwq = 0;
  ComplexType wq;
  double rwq = 0.0;
  const char jobz = 'V', uplo = 'L';
  f77::zheevd_(&jobz, &uplo, &n, A.data(), &n, lam.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info);
  lwork  = std::max(1, int(std::real(wq)));
  lrwork = std::max(1, int(rwq));
  liwork = std::max(1, iwq);
  nda::array<ComplexType, 1> work(lwork);
  nda::array<double, 1> rwork(lrwork);
  nda::array<int, 1> iwork(liwork);
  f77::zheevd_(&jobz, &uplo, &n, A.data(), &n, lam.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
               &info);
  utils::check(info == 0, "cayley::herm_eig: zheevd info = {}", info);
  return lam;
}

/// herm_eig through the external driver when one is installed (and the matrix is large enough), else the host zheevd.
inline nda::array<double, 1> herm_eig(cmatrix_F &A, lapack_hooks_t const *h) {
  if (h and h->heevd and A.extent(0) >= h->min_dim) {
    nda::array<double, 1> w(A.extent(0));
    if (h->heevd(A, w)) return w;
  }
  return herm_eig(A);
}

/// Complex Schur form A = Z T Z^dagger (zgees, no sorting): returns diag(T); A is overwritten by T, Z is returned in Z.
inline nda::array<ComplexType, 1> schur(cmatrix_F &A, cmatrix_F &Z) {
  const int n = int(A.extent(0));
  nda::array<ComplexType, 1> w(n);
  Z.resize(n, n);
  if (n == 0) return w;
  int info = 0, sdim = 0, lwork = -1;
  ComplexType wq;
  nda::array<double, 1> rwork(n);
  const char jobvs = 'V', sort = 'N';
  f77::zgees_(&jobvs, &sort, nullptr, &n, A.data(), &n, &sdim, w.data(), Z.data(), &n, &wq, &lwork, rwork.data(), nullptr,
              &info);
  lwork = std::max(1, int(std::real(wq)));
  nda::array<ComplexType, 1> work(lwork);
  f77::zgees_(&jobvs, &sort, nullptr, &n, A.data(), &n, &sdim, w.data(), Z.data(), &n, work.data(), &lwork, rwork.data(),
              nullptr, &info);
  utils::check(info == 0, "cayley::schur: zgees info = {}", info);
  return w;
}

/// SVD A = P diag(s) Qh with all singular vectors by divide and conquer (zgesdd, jobz 'A'); A is destroyed.
inline void svd_dc(cmatrix_F &A, nda::array<double, 1> &s, cmatrix_F &P, cmatrix_F &Qh) {
  const int m = int(A.extent(0)), n = int(A.extent(1)), mn = std::min(m, n), mx = std::max(m, n);
  s.resize(mn);
  P.resize(m, m);
  Qh.resize(n, n);
  if (mn == 0) return;
  int info = 0, lwork = -1;
  ComplexType wq;
  const long lrwork = std::max(1L, long(mn) * std::max(5L * mn + 7, 2L * mx + 2L * mn + 1));
  nda::array<double, 1> rwork(lrwork);
  nda::array<int, 1> iwork(8 * long(mn));
  const char jobz = 'A';
  f77::zgesdd_(&jobz, &m, &n, A.data(), &m, s.data(), P.data(), &m, Qh.data(), &n, &wq, &lwork, rwork.data(), iwork.data(),
               &info);
  lwork = std::max(1, int(std::real(wq)));
  nda::array<ComplexType, 1> work(lwork);
  f77::zgesdd_(&jobz, &m, &n, A.data(), &m, s.data(), P.data(), &m, Qh.data(), &n, work.data(), &lwork, rwork.data(),
               iwork.data(), &info);
  utils::check(info == 0, "cayley::svd_dc: zgesdd info = {}", info);
}

/// C = op(A) op(B) for F-layout matrices; op = 'N' or 'C' (conjugate transpose).
inline cmatrix_F mm(cmatrix_F const &A, cmatrix_F const &B, char opA = 'N', char opB = 'N') {
  const long m = (opA == 'N') ? A.extent(0) : A.extent(1);
  const long k = (opA == 'N') ? A.extent(1) : A.extent(0);
  const long n = (opB == 'N') ? B.extent(1) : B.extent(0);
  utils::check(k == ((opB == 'N') ? B.extent(0) : B.extent(1)), "cayley::mm: inner dimension mismatch");
  cmatrix_F C(m, n);
  C() = ComplexType(0.0);
  if (m == 0 or n == 0 or k == 0) return C;
  const ComplexType one(1.0), zero(0.0);
  if (opA == 'N' and opB == 'N') nda::blas::gemm(one, A, B, zero, C);
  else if (opA == 'C' and opB == 'N') nda::blas::gemm(one, nda::dagger(A), B, zero, C);
  else if (opA == 'N' and opB == 'C') nda::blas::gemm(one, A, nda::dagger(B), zero, C);
  else nda::blas::gemm(one, nda::dagger(A), nda::dagger(B), zero, C);
  return C;
}

/// statistics of unitary_eig_cayley (S7g)
struct ueig_stats_t {
  long nflag      = 0;     ///< columns refined by Rayleigh-Ritz
  double res_max  = 0.0;   ///< max_l |U z_l - u_l z_l| after the refinement
  bool fallback   = false; ///< the caller must use the Schur form instead
  int reason      = 0;     ///< fallback: 1 (U - 1) singular / non-finite solve, 2 too many flagged columns, 3 residual after RR
  double t_rr     = 0.0;   ///< perf 7.1c: seconds in the Rayleigh-Ritz refinement
};

/**
 * Eigen-decomposition U = Z diag(u) Z^dag of a unitary (normal) matrix through a HERMITIAN eigenproblem (S7g; replaces the
 * complex Schur form, ~5x cheaper and well threaded): Hc = i (U - u0)^{-1} (U + u0) (|u0| = 1, default 1) is Hermitian
 * with eigenvalues cot((theta_l - beta) / 2) (u_l = e^{i theta_l}, u0 = e^{i beta}) and the eigenvectors of U. The map theta -> cot(theta/2) is INJECTIVE on the circle
 * minus u0, so (near-)degenerate eigenvalues of Hc are (near-)degenerate u's: eigenvector mixing only happens between
 * poles that are close (harmless for Sigma, exactly as in the Schur form). u_l = z_l^dag U z_l (Rayleigh quotients);
 * columns with a residual |U z_l - u_l z_l| > tol (mixing of close poles, or loss of accuracy of the solve when an eigenvalue
 * is close to u0) are refined by Rayleigh-Ritz in their span (Schur form of the small projected matrix). Returns false
 * (stats.fallback) if (U - 1) is singular, more than max(32, n/4) columns are flagged or the refinement does not reach tol:
 * the caller then uses the Schur form.
 */
inline bool unitary_eig_cayley(cmatrix_F const &U, nda::array<ComplexType, 1> &u, cmatrix_F &Z, double tol,
                               ueig_stats_t &st, lapack_hooks_t const *h = nullptr, ComplexType u0 = ComplexType(1.0),
                               double accept = 1.0) {
  const int n = int(U.extent(0));
  st = ueig_stats_t{};
  u.resize(n);
  Z.resize(n, n);
  if (n == 0) return true;
  cmatrix_F A(U), X(U);
  for (long i = 0; i < n; ++i) {
    A(i, i) -= u0;
    X(i, i) += u0;
  }
  if (not(h and h->lu_solve and n >= h->min_dim and h->lu_solve(A, X))) {
    nda::array<int, 1> ipiv(n);
    int info = 0;
    f77::zgetrf_(&n, &n, A.data(), &n, ipiv.data(), &info);
    if (info != 0) { st.fallback = true; st.reason = 1; return false; }
    const char tr = 'N';
    f77::zgetrs_(&tr, &n, &n, A.data(), &n, ipiv.data(), X.data(), &n, &info);
    utils::check(info == 0, "cayley::unitary_eig_cayley: zgetrs info = {}", info);
  }
  for (auto const &x : X)
    if (not(std::isfinite(x.real()) and std::isfinite(x.imag()))) { st.fallback = true; st.reason = 1; return false; }
  for (long j = 0; j < n; ++j)   // Z = Hermitian part of i X (lower triangle is enough for zheevd 'L')
    for (long i = j; i < n; ++i) Z(i, j) = 0.5 * (ComplexType(0.0, 1.0) * X(i, j) + std::conj(ComplexType(0.0, 1.0) * X(j, i)));
  herm_eig(Z, h);
  auto UZ = mm(U, Z);
  std::vector<double> res(n);
  std::vector<long> F;
  for (long l = 0; l < n; ++l) {
    ComplexType q(0.0);
    for (long i = 0; i < n; ++i) q += std::conj(Z(i, l)) * UZ(i, l);
    u(l)     = q;
    double r = 0.0;
    for (long i = 0; i < n; ++i) r += std::norm(UZ(i, l) - q * Z(i, l));
    res[l] = std::sqrt(r);
    if (res[l] > tol) F.push_back(l);
  }
  st.nflag   = long(F.size());
  st.res_max = *std::max_element(res.begin(), res.end());
  if (st.nflag > std::max(32L, long(n) / 4)) { st.fallback = true; st.reason = 2; return false; }
  if (not F.empty()) {   // mixing partners are neighbours in the (ascending) eigenvalues of Hc: add +-2 around each flagged
    std::vector<char> in(n, 0);
    for (long l : F)
      for (long m = std::max(0L, l - 2); m <= std::min(long(n) - 1, l + 2); ++m) in[m] = 1;
    F.clear();
    for (long l = 0; l < n; ++l)
      if (in[l]) F.push_back(l);
  }
  if (not F.empty()) {   // Rayleigh-Ritz in span(Z[:, F])
    const auto trr = std::chrono::steady_clock::now();
    const long f = long(F.size());
    cmatrix_F VF(n, f), UVF(n, f);
    for (long c = 0; c < f; ++c)
      for (long i = 0; i < n; ++i) {
        VF(i, c)  = Z(i, F[c]);
        UVF(i, c) = UZ(i, F[c]);
      }
    auto Ut = mm(VF, UVF, 'C', 'N');
    cmatrix_F Zs;
    auto w  = schur(Ut, Zs);
    auto ZF = mm(VF, Zs);
    auto UF = mm(UVF, Zs);
    for (long c = 0; c < f; ++c) {
      double r = 0.0;
      for (long i = 0; i < n; ++i) {
        Z(i, F[c]) = ZF(i, c);
        r += std::norm(UF(i, c) - w(c) * ZF(i, c));
      }
      u(F[c])    = w(c);
      res[F[c]]  = std::sqrt(r);
    }
    st.t_rr = std::chrono::duration<double>(std::chrono::steady_clock::now() - trr).count();
  }
  st.res_max = *std::max_element(res.begin(), res.end());
  if (st.res_max > tol * accept) { st.fallback = true; st.reason = 3; return false; }
  return true;
}

} // namespace detail

// ------------------------------------------------------------------------------------------------------------------
// Block Toeplitz unitary realization (upfold.py)
// ------------------------------------------------------------------------------------------------------------------

/// Hermitian block Toeplitz T with [T]_ij = C^(j-i) (j >= i), C^(i-j)^dagger (j < i), i, j = 0..K; C: [>= K+1, n, n].
inline cmatrix_F block_toeplitz(nda::array<ComplexType, 3> const &C, long K) {
  const long n = C.extent(1);
  cmatrix_F T((K + 1) * n, (K + 1) * n);
  for (long i = 0; i <= K; ++i)
    for (long j = 0; j <= K; ++j)
      for (long a = 0; a < n; ++a)
        for (long b = 0; b < n; ++b)
          T(i * n + a, j * n + b) = (j >= i) ? C(j - i, a, b) : std::conj(C(i - j, b, a));
  return T;
}

/// Options of upfold_block (the python defaults reproduce upfold.py; S7f: alternative cuts, phase continuity, diagnostics).
struct upfold_opts_t {
  double tol_c0       = 1e-12;   ///< relative eigenvalue cutoff of C^(0)
  double tol_gram     = 1e-10;   ///< relative eigenvalue cutoff of the block Toeplitz Gram matrix
  double tol_svd      = 1e-12;   ///< relative singular-value cutoff of D+ D-^dagger (rank r1)
  long nphi           = 8;       ///< coarse terminal-phase scan
  double reject_unity = 1e-6;    ///< coarse phases with an eigenvalue |u - 1| < reject_unity are inadmissible
  /**
   * Gram cut (S7f): "hard" = keep lambda > tol_gram lambda_max (python); "gap" = cut at the largest ratio
   * lambda_i / lambda_{i+1} among the boundaries with lambda_i >= tol_gram lambda_max / cut_window and
   * lambda_{i+1} <= tol_gram lambda_max cut_window (the hard boundary is always a candidate); "smooth" = keep
   * lambda > tol_gram lambda_max / cut_window and weight the Gram rows by a smooth step s(log lambda) that goes from
   * 0 at tol_gram / cut_window to 1 at tol_gram cut_window (X = diag(sqrt(lambda s)) V^dagger): a direction crossing
   * the cut enters with zero weight.
   */
  std::string gram_cut = "hard";
  std::string svd_cut  = "hard";   ///< "hard" (python) | "gap" (largest singular-value ratio within the window)
  double cut_window    = 10.0;     ///< window factor of the "gap" / "smooth" cuts
  /// Phase continuity (S7f): if finite, the coarse-scan basin of phi_prev is kept when its held-out error is within
  /// phase_keep x the best coarse error (the golden-section bracket is then centred on phi_prev).
  double phi_prev   = std::numeric_limits<double>::quiet_NaN();
  double phase_keep = 10.0;
  // attribution diagnostics (tests only): force the retained Gram rank, r1, or the terminal phase
  long force_rgram = -1, force_r1 = -1;
  double force_phi = std::numeric_limits<double>::quiet_NaN();
  // S7g linear-algebra drivers (performance; results agree to roundoff-level effects, see notes S7g)
  std::string svd_driver = "gesvd";   ///< SVD of D+ D-^dagger: "gesvd" (python / numpy path) | "gesdd" (divide and conquer)
  std::string ueig       = "schur";   ///< eigenvectors of U: "schur" (zgees) | "cayley" (Hermitian Cayley image, zgees fallback)
  /// residual tolerance |U z - u z| of the "cayley" path (Rayleigh-Ritz above it). si222c (job 7168848): accepted residuals
  /// 1.7e-13..8.4e-13, retried ones 1.1e-12..2.6e-12 at 1e-12 (Schur's backward error ~ n eps ~ 1e-13); a residual of
  /// 1e-11 moves a pole by <= 1e-11 (d^2 + wp^2) / (2 wp) (1.6e-9 Ha at |d| = 6 Ha), far below the closure's roundoff floor
  double ueig_tol = 1e-11;
  /**
   * perf 7.1c: angle beta of the first cut u0 = e^{i beta} of the "cayley" path. 0 (u0 = 1, omega = +-infinity; S7g):
   * ill-conditioned when the realization has a pole at very large |d| (moment-truncation artefacts, e.g. a Gram
   * eigenvalue at the cut), which costs a failed attempt + the gap-cut retry. pi (u0 = -1, omega = mu): inside the gap of
   * an insulating Sigma, never close to an eigenvalue there; the retry rule is unchanged. Changes the eigenvectors at
   * the roundoff level only.
   */
  double ueig_cut = 0.0;
  /**
   * perf 7.1c: the "cayley" path accepts the eigenvectors when the residual after Rayleigh-Ritz is <= ueig_accept x
   * ueig_tol (columns above ueig_tol are still refined); above it: the gap-cut retry. 1 = S7g. si444 (perf 7.1c): the
   * retries (1-2 k per iteration, +3.3-4.0 s each) had post-RR residuals 2.1e-11..2.9e-11 at ueig_tol 1e-11.
   */
  double ueig_accept = 1.0;
  /// perf 7.1c: held-out error of the golden-section refinement: "eigen" (python: the error of each realization) |
  /// "poly" (heldout_poly_t: ||R U(z)^{K+1} R^dag - C^(K+1)|| as an exact recursion in z = e^{i phi}, no eigensolve)
  std::string scan_err = "eigen";
  lapack_hooks_t const *hooks = nullptr;   ///< external (device) drivers of the Gram eigen, SVD and Cayley path; null: host
};

/// Result of upfold_block: Sigma_c(z) = sum_l W[:, l] W[:, l]^dagger / (z - d_l), d mu-RELATIVE (sorted ascending).
struct upfold_result_t {
  nda::array<double, 1> d;           ///< [np] real poles, mu-relative: d_l = wp cot(theta_l / 2)
  nda::array<ComplexType, 2> W;      ///< [n, np] couplings W = R Z
  nda::array<ComplexType, 1> u;      ///< [np] unit-circle eigenvalues of U(phi*), same order as d
  double phi      = 0.0;             ///< terminal phase phi*
  double residual = 0.0;             ///< held-out error ||R U^{K+1} R^dag - C^(K+1)||_F / (1 + ||C^(K+1)||_F)
  long r0         = 0;               ///< rank of C^(0) (tol_c0)
  long r_gram     = 0;               ///< retained rank of the block Toeplitz Gram matrix (tol_gram) = number of poles
  long r1         = 0;               ///< numerical rank of D+ D-^dagger (tol_svd)
  long n_free     = 0;               ///< r_gram - r1: dimension of the free terminal-phase block
  double gram_lam_min = 0.0;         ///< smallest / largest eigenvalue of T
  // S7f diagnostics of the hard decisions
  long gram_near      = 0;           ///< Gram eigenvalues within a factor 10 of the threshold tol_gram lambda_max
  double gram_margin  = 0.0;         ///< min |log10(lambda / (tol_gram lambda_max))| over all eigenvalues (decades)
  double gram_ratio   = 0.0;         ///< lambda_last_kept / lambda_first_dropped at the chosen cut (inf: nothing dropped)
  long svd_near       = 0;           ///< singular values within a factor 10 of tol_svd s_max
  double svd_margin   = 0.0;         ///< min |log10(s / (tol_svd s_max))| (decades)
  long phi_index      = -1;          ///< coarse-scan index of the chosen basin (-1: no scan)
  double phi_tie      = 0.0;         ///< best coarse error of the OTHER local minima / the chosen one (>= 1; ~1 = near tie)
  long n_rejected     = 0;           ///< coarse phases rejected by reject_unity
  bool phi_kept       = false;       ///< phase continuity kept the basin of phi_prev over the best coarse phase
  // S7g profile (wall seconds of this call) and statistics of the "cayley" eigen path
  double t_c0 = 0.0, t_gram = 0.0, t_svd = 0.0, t_ueig = 0.0;   ///< C0 + Chat | Toeplitz + Gram eigen + X | SVD + R | realizations
  long n_realize      = 0;           ///< realizations U(phi) -> (u, W) (1 if n_free = 0)
  long ueig_nflag     = 0;           ///< "cayley": columns refined by Rayleigh-Ritz (max over realizations)
  long ueig_fallback  = 0;           ///< "cayley": realizations that fell back to the Schur form
  double ueig_res     = 0.0;         ///< "cayley": max residual |U z - u z| (max over realizations, incl. failed attempts)
  int ueig_reason     = 0;           ///< "cayley": reason of the last failed attempt (ueig_stats_t::reason; + 10: the retry)
  long ueig_retry     = 0;           ///< "cayley": realizations retried with the cut u0 in the largest gap of the spectrum
  bool svd_reference  = false;       ///< a fast SVD driver found n_free > 0 and the SVD was redone with zgesvd
  // perf 7.1c profile of the realizations and of the terminal-phase scan (wall seconds)
  double t_svd_ref = 0.0;            ///< the reference zgesvd redo (part of t_svd)
  double t_coarse = 0.0, t_refine = 0.0, t_final = 0.0;   ///< scan: coarse phases | golden section | final realization
  double t_eig    = 0.0;             ///< all eigen realizations (incl. retries, Rayleigh-Ritz, Schur fallbacks)
  double t_rr     = 0.0;             ///< Rayleigh-Ritz refinements + gap-cut retries + Schur fallbacks (part of t_eig)
  double t_poly   = 0.0;             ///< building the eigensolve-free held-out error (heldout_poly_t)
  double t_bcast  = 0.0;             ///< distributed scan: broadcast of the problem
  long n_eig = 0, n_mfree = 0;       ///< eigen realizations | eigensolve-free held-out evaluations
  long scan_ranks = 0, scan_threads = 0;   ///< distributed scan: ranks of the batch, BLAS threads of the owner
};

namespace detail {
/// boundary b (keep x[0..b)) of a descending sequence x (normalized by x[0]) for the hard / gap cut at threshold t
inline long cut_boundary(std::vector<double> const &x, double t, std::string const &mode, double window) {
  const long n = long(x.size());
  long bh = 0;
  while (bh < n and x[bh] > t) ++bh;
  if (mode == "hard" or mode == "smooth") return bh;
  utils::check(mode == "gap", "cayley::upfold_block: cut must be \"hard\", \"gap\" or \"smooth\" (got \"{}\")", mode);
  long best = bh;
  double rbest = -1.0;
  for (long b = 1; b <= n; ++b) {   // keep at least one
    const double xk = x[b - 1], xd = (b < n) ? x[b] : 0.0;
    if (xk < t / window or xd > t * window) continue;
    const double r = (xd > 0.0) ? xk / xd : std::numeric_limits<double>::infinity();
    if (r > rbest) { rbest = r; best = b; }
  }
  return best;
}
} // namespace detail

/**
 * perf 7.1c: the realization problem of upfold_block after the SVD: U(phi) = A1 + e^{i phi} A0 (A1 = P1 Q1^dag,
 * A0 = P0 Q0^dag, A0 unused if n_free = 0), couplings W = R Z (R = B X[:, :r]^dag, n x Nr) and the held-out moment
 * Cheld = C^(K+1). upfold_prepare builds it, realize / heldout_poly evaluate it, upfold_finish turns the chosen
 * realization into poles. upfold_block chains them (serial); GW_line/closure_scan.hpp distributes the terminal-phase scan
 * of the same problem over MPI ranks.
 */
struct upfold_problem_t {
  long n = 0, Nr = 0, K = 0, n_free = 0;
  double wp = 0.0, nheld = 0.0;   ///< nheld = ||C^(K+1)||_F
  cmatrix_F A1, A0, R, Cheld;
  cmatrix_F P0, Q0h;              ///< the free block: A0 = P0 Q0h (Nr x n_free, n_free x Nr)
  bool need_ref = false;          ///< upfold_prepare(defer_ref): the reference SVD of Mref is still to be done
  cmatrix_F Mref;                 ///< D+ D-^dag for the deferred reference SVD
};

/// One realization U(phi) = Z diag(u) Z^dag: held-out error ||W diag(u^{K+1}) W^dag - C^(K+1)|| / (1 + ||C^(K+1)||), u, W = R Z.
struct realization_t {
  double err = 0.0;
  nda::array<ComplexType, 1> u;
  cmatrix_F W;
};

namespace detail {
/// numerical rank r1 of the singular values sv (descending) with the cut of o; sets the diagnostics svd_near / svd_margin
inline long svd_rank(nda::array<double, 1> const &sv, upfold_opts_t const &o, upfold_result_t &res) {
  const long Nr = sv.size();
  const double tol_svd = o.tol_svd;
  std::vector<double> xs(Nr);
  for (long i = 0; i < Nr; ++i) xs[i] = (sv(0) > 0.0) ? sv(i) / sv(0) : 0.0;
  res.svd_near   = 0;
  res.svd_margin = std::numeric_limits<double>::infinity();
  for (long i = 0; i < Nr; ++i) {
    if (xs[i] > 0.1 * tol_svd and xs[i] < 10.0 * tol_svd) ++res.svd_near;
    if (xs[i] > 0.0) res.svd_margin = std::min(res.svd_margin, std::abs(std::log10(xs[i] / tol_svd)));
  }
  long r = cut_boundary(xs, tol_svd, o.svd_cut == "smooth" ? "hard" : o.svd_cut, o.cut_window);
  if (o.svd_cut == "hard") {   // python: count of sv > tol_svd s0 (identical to the boundary for a descending sequence)
    r = 0;
    for (long i = 0; i < Nr; ++i)
      if (sv(i) > tol_svd * sv(0)) ++r;
  }
  if (o.force_r1 >= 0) r = std::min(o.force_r1, Nr);
  return r;
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// U(phi) = A1 + e^{i phi} A0 (A1 alone if n_free = 0); the same arithmetic in every realization and held-out evaluation
inline cmatrix_F form_u(upfold_problem_t const &pr, double phi) {
  const long Nr = pr.Nr;
  cmatrix_F U(Nr, Nr);
  const ComplexType eph = std::exp(ComplexType(0.0, phi));
  for (long j = 0; j < Nr; ++j)
    for (long i = 0; i < Nr; ++i) U(i, j) = pr.A1(i, j) + (pr.n_free > 0 ? eph * pr.A0(i, j) : ComplexType(0.0));
  return U;
}
/// A1 = P1 Q1^dag, A0 = P0 Q0^dag (+ P0, Q0h) of the SVD P diag(s) Qh with rank r1; sets r1 / n_free
inline void set_free_block(upfold_problem_t &pr, cmatrix_F const &P, cmatrix_F const &Qh, long r1, upfold_result_t &res) {
  using nda::range;
  const long Nr = pr.Nr;
  res.r1     = r1;
  res.n_free = Nr - r1;
  pr.n_free  = Nr - r1;
  cmatrix_F P1 = P(range::all, range(0, r1)), Q1h = Qh(range(0, r1), range::all);
  pr.P0  = P(range::all, range(r1, Nr));
  pr.Q0h = Qh(range(r1, Nr), range::all);
  pr.A1  = mm(P1, Q1h);                                             // P1 Q1^dag
  pr.A0  = mm(pr.P0, pr.Q0h);                                       // P0 Q0^dag
}
} // namespace detail

/**
 * Steps 1-2 of upfold_block (C^(0) normalization, Gram factorization, SVD of D+ D-^dag): fills the decisions / diagnostics /
 * profile (t_c0, t_gram, t_svd, t_svd_ref) of res and returns the realization problem.
 */
inline upfold_problem_t upfold_prepare(nda::array<ComplexType, 3> const &C, long K, double wp, upfold_opts_t const &o,
                                       upfold_result_t &res, bool defer_ref = false) {
  using nda::range;
  const long n = C.extent(1);
  const double tol_c0 = o.tol_c0, tol_gram = o.tol_gram;
  const long nphi = o.nphi;
  utils::check(C.extent(0) >= K + 2, "cayley::upfold_block: need moments 0..K+1 (K = {}), got {}", K, C.extent(0));
  utils::check(K >= 1 and nphi >= 1, "cayley::upfold_block: invalid K = {} or nphi = {}", K, nphi);
  utils::check(o.cut_window >= 1.0, "cayley::upfold_block: cut_window must be >= 1");
  utils::check(o.svd_driver == "gesvd" or o.svd_driver == "gesdd", "cayley::upfold_block: svd_driver must be \"gesvd\" or "
               "\"gesdd\" (got \"{}\")", o.svd_driver);
  utils::check(o.ueig == "schur" or o.ueig == "cayley", "cayley::upfold_block: ueig must be \"schur\" or \"cayley\" (got \"{}\")",
               o.ueig);
  utils::check(o.scan_err == "eigen" or o.scan_err == "poly", "cayley::upfold_block: scan_err must be \"eigen\" or \"poly\" "
               "(got \"{}\")", o.scan_err);
  using clock_t_ = std::chrono::steady_clock;
  auto tlap      = clock_t_::now();
  auto lap       = [&tlap]() {
    const auto t = clock_t_::now();
    const double x = std::chrono::duration<double>(t - tlap).count();
    tlap = t;
    return x;
  };

  // 1. normalize C^(0) = B B^dag
  cmatrix_F V0(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) V0(i, j) = 0.5 * (C(0, i, j) + std::conj(C(0, j, i)));
  auto lam0 = detail::herm_eig(V0);
  const double lam0_max = *std::max_element(lam0.begin(), lam0.end());
  std::vector<long> keep0;
  for (long i = 0; i < n; ++i)
    if (lam0(i) > tol_c0 * lam0_max) keep0.push_back(i);
  const long r = long(keep0.size());
  res.r0 = r;
  cmatrix_F B(n, r), Bp(r, n);
  for (long c = 0; c < r; ++c) {
    const double s = std::sqrt(lam0(keep0[c]));
    for (long i = 0; i < n; ++i) {
      B(i, c)  = V0(i, keep0[c]) * s;
      Bp(c, i) = std::conj(V0(i, keep0[c])) / s;
    }
  }
  nda::array<ComplexType, 3> Chat(K + 1, r, r);
  for (long m = 0; m <= K; ++m) {
    cmatrix_F Cm(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) Cm(i, j) = C(m, i, j);
    auto Ch = detail::mm(detail::mm(Bp, Cm), Bp, 'N', 'C');
    for (long i = 0; i < r; ++i)
      for (long j = 0; j < r; ++j) Chat(m, i, j) = Ch(i, j);
  }

  res.t_c0 = lap();

  // 2. Gram factorization of the block Toeplitz matrix
  auto T = block_toeplitz(Chat, K);
  utils::check(T.size() > 0, "cayley::upfold_block: moments are empty (C0 rank {})", r);
  for (auto const &x : T) utils::check(std::isfinite(x.real()) and std::isfinite(x.imag()), "cayley::upfold_block: non-finite moments");
  auto lamT = detail::herm_eig(T, o.hooks);                        // T now holds the eigenvectors (ascending eigenvalues)
  const long nT = lamT.size();
  const double lamT_max = *std::max_element(lamT.begin(), lamT.end());
  res.gram_lam_min = *std::min_element(lamT.begin(), lamT.end()) / lamT_max;
  std::vector<double> xg(nT);                                       // normalized eigenvalues, descending
  for (long i = 0; i < nT; ++i) xg[i] = lamT(nT - 1 - i) / lamT_max;
  res.gram_margin = std::numeric_limits<double>::infinity();
  for (long i = 0; i < nT; ++i) {
    if (xg[i] > 0.1 * tol_gram and xg[i] < 10.0 * tol_gram) ++res.gram_near;
    if (xg[i] > 0.0) res.gram_margin = std::min(res.gram_margin, std::abs(std::log10(xg[i] / tol_gram)));
  }
  long bT = detail::cut_boundary(xg, tol_gram, o.gram_cut, o.cut_window);
  const bool smooth = (o.gram_cut == "smooth");
  if (smooth) {
    bT = 0;
    while (bT < nT and xg[bT] > tol_gram / o.cut_window) ++bT;
  }
  if (o.force_rgram >= 0) bT = std::min(o.force_rgram, nT);
  utils::check(bT >= 1, "cayley::upfold_block: empty Gram cut");
  res.gram_ratio = (bT < nT and xg[bT] > 0.0) ? xg[bT - 1] / xg[bT] : std::numeric_limits<double>::infinity();
  const long Nr = bT;
  res.r_gram = Nr;
  // rows of X = the Nr largest eigenvalues in ASCENDING order (zheevd order; python's keep mask, bitwise the S6 path)
  cmatrix_F X(Nr, nT);                                              // T = X^dag X (smooth: filtered)
  for (long a = 0; a < Nr; ++a) {
    const long ie = nT - Nr + a;
    double lam    = lamT(ie);
    if (smooth) {
      const double x  = std::log10(lam / lamT_max * o.cut_window / tol_gram) / (2.0 * std::log10(o.cut_window));   // 0..1
      const double xc = std::clamp(x, 0.0, 1.0);
      lam *= xc * xc * (3.0 - 2.0 * xc);
    }
    const double s = std::sqrt(std::max(lam, 0.0));
    for (long j = 0; j < nT; ++j) X(a, j) = s * std::conj(T(j, ie));
  }
  res.t_gram = lap();
  cmatrix_F Dm = X(range::all, range(0, K * r)), Dp = X(range::all, range(r, (K + 1) * r));
  auto M = detail::mm(Dp, Dm, 'N', 'C');                            // Nr x Nr
  cmatrix_F P(Nr, Nr), Qh(Nr, Nr);
  nda::array<double, 1> sv(Nr);
  // S7g: fast drivers (zgesdd, device hooks). If the SVD has a free block (r1 < Nr), the pairing of its (near-)null left
  // and right singular vectors -- hence P0 Q0^dagger and the terminal-phase scan -- is driver dependent: the SVD is then
  // redone with the reference driver (zgesvd, the pre-S7g path), so the fast drivers only act when n_free = 0.
  const bool fast = (o.svd_driver == "gesdd") or (o.hooks and o.hooks->gesvd and Nr >= o.hooks->min_dim);
  cmatrix_F Mref;
  if (fast) Mref = M;
  if (not(o.hooks and o.hooks->gesvd and Nr >= o.hooks->min_dim and o.hooks->gesvd(M, sv, P, Qh))) {
    if (o.svd_driver == "gesdd") detail::svd_dc(M, sv, P, Qh);
    else nda::lapack::gesvd(M, sv, P, Qh);
  }
  long r1 = detail::svd_rank(sv, o, res);
  upfold_problem_t pr;
  pr.n = n; pr.Nr = Nr; pr.K = K; pr.wp = wp;
  cmatrix_F X0 = X(range::all, range(0, r));
  pr.R = detail::mm(B, X0, 'N', 'C');                               // n x Nr
  if (fast and r1 < Nr and defer_ref) {   // perf 7.1c: the reference SVD later (upfold_prepare_ref, borrowed cores)
    pr.need_ref = true;
    pr.Mref     = std::move(Mref);
    res.r1      = r1;
    res.n_free  = Nr - r1;
    pr.n_free   = Nr - r1;
  } else {
    if (fast and r1 < Nr) {
      const auto tref = clock_t_::now();
      nda::lapack::gesvd(Mref, sv, P, Qh);
      r1                = detail::svd_rank(sv, o, res);
      res.svd_reference = true;
      res.t_svd_ref     = detail::seconds_since(tref);
    }
    detail::set_free_block(pr, P, Qh, r1, res);
  }
  pr.Cheld = cmatrix_F(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) pr.Cheld(i, j) = C(K + 1, i, j);
  pr.nheld  = detail::frob(pr.Cheld);
  res.t_svd = lap();
  return pr;
}

/// perf 7.1c: the reference SVD deferred by upfold_prepare(defer_ref = true) (zgesvd of Mref; may change r1 / n_free)
inline void upfold_prepare_ref(upfold_problem_t &pr, upfold_opts_t const &o, upfold_result_t &res) {
  if (not pr.need_ref) return;
  const auto t0 = std::chrono::steady_clock::now();
  const long Nr = pr.Nr;
  cmatrix_F P(Nr, Nr), Qh(Nr, Nr);
  nda::array<double, 1> sv(Nr);
  nda::lapack::gesvd(pr.Mref, sv, P, Qh);
  const long r1     = detail::svd_rank(sv, o, res);
  res.svd_reference = true;
  detail::set_free_block(pr, P, Qh, r1, res);
  pr.need_ref = false;
  pr.Mref     = cmatrix_F();
  res.t_svd_ref = detail::seconds_since(t0);
  res.t_svd += res.t_svd_ref;
}

/**
 * One realization U(phi) -> (held-out error, u, W) with the eigenvector driver of o.ueig ("cayley": Hermitian Cayley image
 * with Rayleigh-Ritz, a retry with the cut in the largest spectral gap and the Schur form as the last resort). Statistics
 * and the profile (t_eig, t_rr, n_eig) are accumulated into res.
 */
inline realization_t realize(upfold_problem_t const &pr, double phi, upfold_opts_t const &o, upfold_result_t &res) {
  const long Nr = pr.Nr, n = pr.n, K = pr.K;
  const auto te0 = std::chrono::steady_clock::now();
  cmatrix_F U = detail::form_u(pr, phi);
  cmatrix_F Z;
  nda::array<ComplexType, 1> u;
  ++res.n_realize;
  ++res.n_eig;
  bool done = false;
  if (o.ueig == "cayley") {
    detail::ueig_stats_t st;
    auto const *hk = (o.hooks and o.hooks->in_ueig) ? o.hooks : nullptr;
    done           = detail::unitary_eig_cayley(U, u, Z, o.ueig_tol, st, hk,
                                                    o.ueig_cut == 0.0 ? ComplexType(1.0) : std::exp(ComplexType(0.0, o.ueig_cut)),
                                                    o.ueig_accept);
    res.ueig_nflag = std::max(res.ueig_nflag, st.nflag);
    res.ueig_res   = std::max(res.ueig_res, st.res_max);
    res.t_rr += st.t_rr;
    if (not done) res.ueig_reason = st.reason;
    if (not done and st.reason >= 2) {   // retry with the cut u0 in the largest angular gap of the (approximate) spectrum
      const auto tq = std::chrono::steady_clock::now();
      std::vector<double> th(Nr);
      for (long l = 0; l < Nr; ++l) th[l] = std::arg(u(l));
      std::sort(th.begin(), th.end());
      double gap = th[0] + 2.0 * std::numbers::pi - th[Nr - 1], beta = th[Nr - 1] + 0.5 * gap;
      for (long l = 0; l + 1 < Nr; ++l)
        if (th[l + 1] - th[l] > gap) {
          gap  = th[l + 1] - th[l];
          beta = 0.5 * (th[l] + th[l + 1]);
        }
      ++res.ueig_retry;
      detail::ueig_stats_t st2;
      done           = detail::unitary_eig_cayley(U, u, Z, o.ueig_tol, st2, hk, std::exp(ComplexType(0.0, beta)), o.ueig_accept);
      res.ueig_nflag = std::max(res.ueig_nflag, st2.nflag);
      res.ueig_res   = std::max(res.ueig_res, st2.res_max);
      if (not done) res.ueig_reason = 10 + st2.reason;
      res.t_rr += detail::seconds_since(tq);
    }
    if (not done) ++res.ueig_fallback;
  }
  if (not done) {
    const auto ts = std::chrono::steady_clock::now();
    u = detail::schur(U, Z);
    if (o.ueig == "cayley") res.t_rr += detail::seconds_since(ts);   // the Schur fallback is part of the retry cost
  }
  auto W = detail::mm(pr.R, Z);
  cmatrix_F Wu(n, Nr);
  for (long l = 0; l < Nr; ++l) {
    ComplexType uk(1.0);
    for (long p = 0; p < K + 1; ++p) uk *= u(l);
    for (long i = 0; i < n; ++i) Wu(i, l) = W(i, l) * uk;
  }
  auto Ck = detail::mm(Wu, W, 'N', 'C');
  Ck -= pr.Cheld;
  res.t_eig += detail::seconds_since(te0);
  return realization_t{detail::frob(Ck) / (1.0 + pr.nheld), std::move(u), std::move(W)};
}

/**
 * perf 7.1c: the held-out error of the terminal-phase scan without eigen-decompositions. For the unitary
 * U(z) = A1 + z P0 Q0^dag (z = e^{i phi}) with U = Z diag(u) Z^dag, W diag(u^{K+1}) W^dag = R U^{K+1} R^dag (W = R Z), so
 * the error of realize() is ||R U(z)^{K+1} R^dag - C^(K+1)|| / (1 + ||C^(K+1)||) exactly. Unrolling v_m = R U^m
 * (v_{m+1} = v_m A1 + z (v_m P0) Q0^dag):
 *   R U^{K+1} R^dag = G + z sum_{l=0..K} W_l Y_{K-l},   W_m = v_m P0 = X_m + z sum_{l<m} W_l C_{m-1-l},
 *   G = R A1^{K+1} R^dag, X_m = R A1^m P0 (n x f), C_m = Q0^dag A1^m P0 (f x f), Y_m = Q0^dag A1^m R^dag (f x n).
 * The z-independent parts cost K+1 products of the n x Nr block R with A1 plus two chains of f columns / rows (built once
 * per k); an evaluation is then O(K^2 n f^2 + K n^2 f) -- the 32 golden-section errors become free. The recursion is the
 * unitary evolution v_m -> v_m U written in another order (no expansion in powers of z): roundoff ~ K eps, as realize().
 */
struct heldout_poly_t {
  long K1 = 0, n = 0, f = 0;
  cmatrix_F D;                      ///< G - C^(K+1)
  std::vector<cmatrix_F> X, C, Y;   ///< X_m (m = 0..K), C_m (m = 0..K-1), Y_m (m = 0..K)
  double scale = 1.0;               ///< 1 + ||C^(K+1)||

  double operator()(double phi) const {
    const ComplexType z = std::exp(ComplexType(0.0, phi)), one(1.0), zero(0.0);
    std::vector<cmatrix_F> W(K1);
    cmatrix_F acc(n, f);
    for (long m = 0; m < K1; ++m) {
      acc() = zero;
      for (long l = 0; l < m; ++l) nda::blas::gemm(one, W[l], C[m - 1 - l], one, acc);
      W[m] = X[m];
      for (long j = 0; j < f; ++j)
        for (long i = 0; i < n; ++i) W[m](i, j) += z * acc(i, j);
    }
    cmatrix_F E(n, n);
    E() = zero;
    for (long l = 0; l < K1; ++l) nda::blas::gemm(one, W[l], Y[K1 - 1 - l], one, E);
    double s2 = 0.0;
    for (long j = 0; j < n; ++j)
      for (long i = 0; i < n; ++i) s2 += std::norm(D(i, j) + z * E(i, j));
    return std::sqrt(s2) / scale;
  }
};

/// build heldout_poly_t of a problem with a free block (n_free > 0)
inline heldout_poly_t heldout_poly(upfold_problem_t const &pr) {
  const long n = pr.n, Nr = pr.Nr, f = pr.n_free, K1 = pr.K + 1;
  const ComplexType one(1.0), zero(0.0);
  utils::check(f > 0 and pr.P0.extent(1) == f and pr.Q0h.extent(0) == f, "cayley::heldout_poly: no free block");
  heldout_poly_t hp;
  hp.K1 = K1; hp.n = n; hp.f = f; hp.scale = 1.0 + pr.nheld;
  hp.X.resize(K1); hp.C.resize(K1 > 1 ? K1 - 1 : 0); hp.Y.resize(K1);
  {   // G = R A1^{K+1} R^dag
    cmatrix_F V(pr.R), V2(n, Nr);
    for (long p = 0; p < K1; ++p) {
      nda::blas::gemm(one, V, pr.A1, zero, V2);
      std::swap(V, V2);
    }
    hp.D = detail::mm(V, pr.R, 'N', 'C');
    hp.D -= pr.Cheld;
  }
  {   // S_m = A1^m P0: X_m = R S_m, C_m = Q0^dag S_m
    cmatrix_F S(pr.P0), S2(Nr, f);
    for (long m = 0; m < K1; ++m) {
      hp.X[m] = detail::mm(pr.R, S);
      if (m + 1 < K1) {
        hp.C[m] = detail::mm(pr.Q0h, S);
        nda::blas::gemm(one, pr.A1, S, zero, S2);
        std::swap(S, S2);
      }
    }
  }
  {   // T_m = Q0^dag A1^m: Y_m = T_m R^dag
    cmatrix_F T(pr.Q0h), T2(f, Nr);
    for (long m = 0; m < K1; ++m) {
      hp.Y[m] = detail::mm(T, pr.R, 'N', 'C');
      if (m + 1 < K1) {
        nda::blas::gemm(one, T, pr.A1, zero, T2);
        std::swap(T, T2);
      }
    }
  }
  return hp;
}

/// the coarse phases of the terminal-phase scan (np.linspace(0, 2 pi, nphi, endpoint=False))
inline double coarse_phase(long ip, long nphi) { return 2.0 * std::numbers::pi * double(ip) / double(nphi); }

/// smallest |u_l - 1| of a realization (the reject_unity screen of the coarse scan)
inline double unity_distance(nda::array<ComplexType, 1> const &u) {
  double umin = std::numeric_limits<double>::infinity();
  for (auto const &ul : u) umin = std::min(umin, std::abs(ul - 1.0));
  return umin;
}

/**
 * Decision of the coarse scan from the held-out errors and the unity distances of the nphi coarse realizations: the
 * admissible minimum (reject_unity), phase continuity (o.phi_prev / o.phase_keep), the near-tie diagnostic. Returns the
 * centre phi0 of the golden-section bracket; sets res.phi_index, n_rejected, phi_kept, phi_tie.
 */
inline double coarse_decide(std::vector<double> errs, std::vector<double> const &umin, upfold_opts_t const &o,
                            upfold_result_t &res) {
  const long nphi    = long(errs.size());
  const double twopi = 2.0 * std::numbers::pi;
  for (long ip = 0; ip < nphi; ++ip)
    if (umin[ip] < o.reject_unity) {
      errs[ip] = std::numeric_limits<double>::infinity();
      ++res.n_rejected;
    }
  long i0 = long(std::min_element(errs.begin(), errs.end()) - errs.begin());
  utils::check(std::isfinite(errs[i0]), "cayley::upfold_block: no admissible terminal phase");
  double phi0 = coarse_phase(i0, nphi);
  if (std::isfinite(o.phi_prev)) {   // phase continuity: keep the basin of the previous phase if it is competitive
    double pp = std::fmod(o.phi_prev, twopi);
    if (pp < 0.0) pp += twopi;
    const long ipv = long(std::llround(pp / (twopi / double(nphi)))) % nphi;
    if (std::isfinite(errs[ipv]) and errs[ipv] <= o.phase_keep * errs[i0]) {
      res.phi_kept = (ipv != i0);
      i0           = ipv;
      phi0         = pp;
    }
  }
  res.phi_index = i0;
  // near-tie diagnostic: best error among the other local minima of the circular coarse scan
  double other = std::numeric_limits<double>::infinity();
  for (long ip = 0; ip < nphi; ++ip) {
    if (ip == i0 or not std::isfinite(errs[ip])) continue;
    const double em = errs[(ip + nphi - 1) % nphi], ep = errs[(ip + 1) % nphi];
    if (errs[ip] <= em and errs[ip] <= ep) other = std::min(other, errs[ip]);
  }
  res.phi_tie = other / errs[i0];
  return phi0;
}

/**
 * Golden-section refinement on [phi0 - 2 pi / nphi, phi0 + 2 pi / nphi] (30 steps, as upfold.py), as a state machine so
 * that the serial scan and the distributed lockstep scan run the same arithmetic: evaluate c and d (fc, fd), then 30 times
 * x = propose(); accept(f(x)); the result is 0.5 (a + b).
 */
struct golden_t {
  static constexpr int nsteps = 30;
  double a = 0.0, b = 0.0, c = 0.0, d = 0.0, fc = 0.0, fd = 0.0;
  bool left = false;   ///< the pending point is c (true) or d (false)
  golden_t() = default;
  golden_t(double phi0, long nphi) {
    const double twopi = 2.0 * std::numbers::pi;
    a = phi0 - twopi / double(nphi);
    b = phi0 + twopi / double(nphi);
    c = b - gr() * (b - a);
    d = a + gr() * (b - a);
  }
  static double gr() { return (std::sqrt(5.0) - 1.0) / 2.0; }
  double propose() {
    if (fc < fd) {
      b = d; d = c; fd = fc; c = b - gr() * (b - a);
      left = true;
      return c;
    }
    a = c; c = d; fc = fd; d = a + gr() * (b - a);
    left = false;
    return d;
  }
  void accept(double f) { (left ? fc : fd) = f; }
  double result() const { return 0.5 * (a + b); }
};

/// the golden-section refinement of g with the error function f (python order of evaluations)
template <typename F> inline void golden_refine(golden_t &g, F &&f) {
  g.fc = f(g.c);
  g.fd = f(g.d);
  for (int it = 0; it < golden_t::nsteps; ++it) {
    const double x = g.propose();
    g.accept(f(x));
  }
}

/// step 3 of upfold_block: poles (mu-relative) of the chosen realization, sorted ascending
inline void upfold_finish(upfold_problem_t const &pr, realization_t const &best, double phi, upfold_result_t &res) {
  const long Nr = pr.Nr, n = pr.n;
  res.phi      = phi;
  res.residual = best.err;
  std::vector<double> dl(Nr);
  for (long l = 0; l < Nr; ++l) dl[l] = cayley_inverse(best.u(l), pr.wp);
  std::vector<long> order(Nr);
  std::iota(order.begin(), order.end(), 0L);
  std::stable_sort(order.begin(), order.end(), [&](long x, long y) { return dl[x] < dl[y]; });
  res.d.resize(Nr);
  res.u.resize(Nr);
  res.W.resize(n, Nr);
  for (long l = 0; l < Nr; ++l) {
    res.d(l) = dl[order[l]];
    res.u(l) = best.u(order[l]);
    for (long i = 0; i < n; ++i) res.W(i, l) = best.W(i, order[l]);
  }
}

/// true if the terminal phase of this problem is chosen by a scan (n_free > 0 and no forced phase)
inline bool needs_phase_scan(upfold_problem_t const &pr, upfold_opts_t const &o) {
  return pr.n_free > 0 and not std::isfinite(o.force_phi);
}

/**
 * Unitary realization of the block moments C[0..K]; C[K+1] is the held-out moment that fixes the terminal phase (Eq. upfold).
 * Step by step as upfold.py::upfold_block (mu = 0, phase_refine = True):
 *   C^(0) = B B^dag (Hermitian eigen-decomposition, eigenvalues > tol_c0 lam_max), Chat = B^+ C B^+dag;
 *   T = block_toeplitz(Chat, K) = X^dag X with the eigenvalues > tol_gram lam_max; D- = X[:, :K r], D+ = X[:, r:(K+1) r];
 *   SVD D+ D-^dag = P S Q^dag, rank r1 (tol_svd); U(phi) = P1 Q1^dag + e^{i phi} P0 Q0^dag; R = B X[:, :r]^dag;
 *   phi: scan of nphi phases on [0, 2 pi) (a phase is inadmissible if an eigenvalue of U has |u - 1| < reject_unity),
 *   then 30 golden-section steps on [phi0 - 2 pi/nphi, phi0 + 2 pi/nphi]; complex Schur U = Z diag(u) Z^dag, W = R Z,
 *   d = wp cot(arg(u)/2). As in python, reject_unity only screens the coarse-scan phases (the refined phase is not re-screened).
 * Python's default nphi is 72 (kept here for parity); the cost is one Schur form of an r_gram x r_gram matrix per phase.
 * upfold_opts_t selects the S7f alternatives (gap / smooth cuts, phase continuity); its defaults are the python algorithm.
 * perf 7.1c: o.scan_err = "poly" evaluates the golden-section errors without eigen-decompositions (heldout_poly_t).
 */
inline void upfold_complete(upfold_problem_t const &pr, upfold_opts_t const &o, upfold_result_t &res) {
  const auto t0 = std::chrono::steady_clock::now();
  realization_t best;
  double phi_best = 0.0;
  if (res.n_free == 0) {
    best = realize(pr, 0.0, o, res);
  } else if (std::isfinite(o.force_phi)) {
    phi_best = o.force_phi;
    best     = realize(pr, phi_best, o, res);
  } else {
    const long nphi = o.nphi;
    std::vector<double> errs(nphi), umin(nphi);
    for (long ip = 0; ip < nphi; ++ip) {
      auto rz  = realize(pr, coarse_phase(ip, nphi), o, res);
      errs[ip] = rz.err;
      umin[ip] = unity_distance(rz.u);
    }
    golden_t g(coarse_decide(errs, umin, o, res), nphi);
    res.t_coarse = detail::seconds_since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    heldout_poly_t hp;
    if (o.scan_err == "poly") {
      hp         = heldout_poly(pr);
      res.t_poly = detail::seconds_since(t1);
    }
    auto f = [&](double p) {
      if (o.scan_err != "poly") return realize(pr, p, o, res).err;
      ++res.n_mfree;
      return hp(p);
    };
    golden_refine(g, f);
    phi_best      = g.result();
    res.t_refine  = detail::seconds_since(t1);
    const auto t2 = std::chrono::steady_clock::now();
    best          = realize(pr, phi_best, o, res);
    res.t_final   = detail::seconds_since(t2);
  }
  res.t_ueig = detail::seconds_since(t0);
  upfold_finish(pr, best, phi_best, res);
}

/// perf 7.1c: the final realization at a phase found elsewhere (the distributed scan) and the poles; sets t_final
inline void upfold_final(upfold_problem_t const &pr, double phi, upfold_opts_t const &o, upfold_result_t &res) {
  const auto t0 = std::chrono::steady_clock::now();
  auto best     = realize(pr, phi, o, res);
  res.t_final   = detail::seconds_since(t0);
  upfold_finish(pr, best, phi, res);
}

inline upfold_result_t upfold_block(nda::array<ComplexType, 3> const &C, long K, double wp, upfold_opts_t const &o) {
  upfold_result_t res;
  auto pr = upfold_prepare(C, K, wp, o, res);
  upfold_complete(pr, o, res);
  return res;
}

/// python signature (upfold.py defaults except nphi, see above)
inline upfold_result_t upfold_block(nda::array<ComplexType, 3> const &C, long K, double wp, double tol_c0 = 1e-12,
                                    double tol_gram = 1e-10, double tol_svd = 1e-12, long nphi = 8,
                                    double reject_unity = 1e-6) {
  upfold_opts_t o;
  o.tol_c0       = tol_c0;
  o.tol_gram     = tol_gram;
  o.tol_svd      = tol_svd;
  o.nphi         = nphi;
  o.reject_unity = reject_unity;
  return upfold_block(C, K, wp, o);
}

// ------------------------------------------------------------------------------------------------------------------
// Sigma_c, G, A from the poles (spectral.py)
// ------------------------------------------------------------------------------------------------------------------

/// Sigma_c(z) = sum_l W_l W_l^dag / (z - d_l) (Eq. Htilde, left), [n, n]; d, z mu-relative.
inline nda::array<ComplexType, 2> sigma_from_poles(nda::array<double, 1> const &d, nda::array<ComplexType, 2> const &W,
                                                   ComplexType z) {
  const long n = W.extent(0), np = W.extent(1);
  nda::array<ComplexType, 2> Wz(n, np), S(n, n);
  for (long i = 0; i < n; ++i)
    for (long l = 0; l < np; ++l) Wz(i, l) = W(i, l) / (z - d(l));
  S() = ComplexType(0.0);
  if (np > 0) nda::blas::gemm(ComplexType(1.0), Wz, nda::dagger(W), ComplexType(0.0), S);
  return S;
}

/// G(z) = [z - H - Sigma_c(z)]^{-1}, [n, n]; H = H_stat - mu.
inline nda::array<ComplexType, 2> greens_function(nda::array<ComplexType, 2> const &Hrel, nda::array<double, 1> const &d,
                                                  nda::array<ComplexType, 2> const &W, ComplexType z) {
  const long n = Hrel.extent(0);
  auto S       = sigma_from_poles(d, W, z);
  nda::matrix<ComplexType> Mz(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) Mz(i, j) = (i == j ? z : ComplexType(0.0)) - Hrel(i, j) - S(i, j);
  nda::inverse_in_place(Mz);
  return nda::array<ComplexType, 2>(Mz);
}

/// A(omega) = (i / 2 pi) [G(omega + i eta) - G(omega + i eta)^dag] (Eq. A), [nw, n, n]; omega mu-relative.
inline nda::array<ComplexType, 3> spectral_function(nda::array<ComplexType, 2> const &Hrel, nda::array<double, 1> const &d,
                                                    nda::array<ComplexType, 2> const &W, nda::array<double, 1> const &omega,
                                                    double eta) {
  const long n = Hrel.extent(0), nw = omega.size();
  nda::array<ComplexType, 3> A(nw, n, n);
  const ComplexType pref(0.0, 0.5 / std::numbers::pi);
  for (long iw = 0; iw < nw; ++iw) {
    auto G = greens_function(Hrel, d, W, ComplexType(omega(iw), eta));
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) A(iw, i, j) = pref * (G(i, j) - std::conj(G(j, i)));
  }
  return A;
}

/// Tr A(omega) = -Im Tr G(omega + i eta) / pi, [nw].
inline nda::array<double, 1> spectral_trace(nda::array<ComplexType, 2> const &Hrel, nda::array<double, 1> const &d,
                                            nda::array<ComplexType, 2> const &W, nda::array<double, 1> const &omega,
                                            double eta) {
  const long n = Hrel.extent(0), nw = omega.size();
  nda::array<double, 1> A(nw);
  for (long iw = 0; iw < nw; ++iw) {
    auto G = greens_function(Hrel, d, W, ComplexType(omega(iw), eta));
    ComplexType tr(0.0);
    for (long i = 0; i < n; ++i) tr += G(i, i);
    A(iw) = -tr.imag() / std::numbers::pi;
  }
  return A;
}

// ------------------------------------------------------------------------------------------------------------------
// Upfolded Hamiltonian and Lehmann G (spectral.py, line/closure.py)
// ------------------------------------------------------------------------------------------------------------------

/// Htilde = [[H, W], [W^dag, diag d]] (Eq. Htilde, mu-relative), [(n + np), (n + np)].
inline nda::array<ComplexType, 2> upfolded_hamiltonian(nda::array<ComplexType, 2> const &Hrel, nda::array<double, 1> const &d,
                                                       nda::array<ComplexType, 2> const &W) {
  const long n = W.extent(0), np = W.extent(1);
  nda::array<ComplexType, 2> Ht(n + np, n + np);
  Ht() = ComplexType(0.0);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) Ht(i, j) = Hrel(i, j);
  for (long i = 0; i < n; ++i)
    for (long l = 0; l < np; ++l) {
      Ht(i, n + l) = W(i, l);
      Ht(n + l, i) = std::conj(W(i, l));
    }
  for (long l = 0; l < np; ++l) Ht(n + l, n + l) = d(l);
  return Ht;
}

/// Lehmann G(z) = sum_m v_m v_m^dag / (z - e_m) from the eigenpairs of Htilde: e [M] ascending (mu-relative), v [n, M].
struct lehmann_t {
  nda::array<double, 1> e;
  nda::array<ComplexType, 2> v;
};

inline lehmann_t lehmann(nda::array<ComplexType, 2> const &Hrel, nda::array<double, 1> const &d,
                         nda::array<ComplexType, 2> const &W, lapack_hooks_t const *hooks = nullptr) {
  const long n = Hrel.extent(0);
  cmatrix_F Ht(upfolded_hamiltonian(Hrel, d, W));
  auto e     = detail::herm_eig(Ht, hooks);
  const long M = e.size();
  lehmann_t out{e, nda::array<ComplexType, 2>(n, M)};
  for (long i = 0; i < n; ++i)
    for (long m = 0; m < M; ++m) out.v(i, m) = Ht(i, m);
  return out;
}

// ------------------------------------------------------------------------------------------------------------------
// Chemical potential (line/closure.py::chemical_potential)
// ------------------------------------------------------------------------------------------------------------------

struct chemical_potential_t {
  double mu     = 0.0;   ///< midpoint of the chosen gap (same reference as e, i.e. a shift if e is mu-relative)
  double e_homo = 0.0;
  double e_lumo = 0.0;
  double gap    = 0.0;   ///< e_lumo - e_homo
  double N      = 0.0;   ///< electron count 2 sum_k w_k sum_{e_m < mu} |v_m|^2
};

/**
 * T=0 chemical potential of a moment-truncated Lehmann G: candidate gaps are the intervals between consecutive
 * quasiparticle-like poles (sum over orbitals of |v_m|^2 > qp_weight, all k merged) whose midpoint gives |N - nelec| <= ntol;
 * the WIDEST is chosen (fallback: the candidate with the best count). k weights are normalized to sum 1 (empty = uniform).
 */
inline chemical_potential_t chemical_potential(std::vector<nda::array<double, 1>> const &e_k,
                                               std::vector<nda::array<ComplexType, 2>> const &v_k, double nelec,
                                               std::vector<double> k_weight = {}, double qp_weight = 0.1,
                                               double ntol = 0.5) {
  const long nk = long(e_k.size());
  utils::check(nk > 0 and long(v_k.size()) == nk, "cayley::chemical_potential: e_k / v_k size mismatch");
  if (k_weight.empty()) k_weight.assign(nk, 1.0);
  utils::check(long(k_weight.size()) == nk, "cayley::chemical_potential: k_weight size mismatch");
  const double wsum = std::accumulate(k_weight.begin(), k_weight.end(), 0.0);
  std::vector<double> E, Wt, Wtot;
  for (long k = 0; k < nk; ++k) {
    const long M = e_k[k].size(), n = v_k[k].extent(0);
    utils::check(v_k[k].extent(1) == M, "cayley::chemical_potential: v_k[{}] has {} columns, e_k {}", k, v_k[k].extent(1), M);
    for (long m = 0; m < M; ++m) {
      double w = 0.0;
      for (long i = 0; i < n; ++i) w += std::norm(v_k[k](i, m));
      E.push_back(e_k[k](m));
      Wt.push_back(2.0 * (k_weight[k] / wsum) * w);
      Wtot.push_back(w);
    }
  }
  const long Mt = long(E.size());
  std::vector<long> order(Mt);
  std::iota(order.begin(), order.end(), 0L);
  std::stable_sort(order.begin(), order.end(), [&](long a, long b) { return E[a] < E[b]; });
  std::vector<double> Es(Mt), cum(Mt);
  std::vector<long> qp;
  double acc = 0.0;
  for (long i = 0; i < Mt; ++i) {
    Es[i] = E[order[i]];
    acc += Wt[order[i]];
    cum[i] = acc;
    if (Wtot[order[i]] > qp_weight) qp.push_back(i);
  }
  struct cand_t { double width, dn, mid, eh, el, N; };
  std::vector<cand_t> cands;
  for (size_t q = 0; q + 1 < qp.size(); ++q) {
    const long a = qp[q], b = qp[q + 1];
    const double mid = 0.5 * (Es[a] + Es[b]);
    long idx = long(std::lower_bound(Es.begin(), Es.end(), mid) - Es.begin()) - 1;   // np.searchsorted(Es, mid) - 1
    if (idx < 0) idx += Mt;                                                          // python's negative index
    const double N = cum[idx];
    cands.push_back({Es[b] - Es[a], std::abs(N - nelec), mid, Es[a], Es[b], N});
  }
  utils::check(not cands.empty(), "cayley::chemical_potential: fewer than two quasiparticle poles");
  std::vector<cand_t> ok;
  for (auto const &c : cands)
    if (c.dn <= ntol) ok.push_back(c);
  if (ok.empty())
    ok.push_back(*std::min_element(cands.begin(), cands.end(), [](auto const &x, auto const &y) { return x.dn < y.dn; }));
  auto best = *std::max_element(ok.begin(), ok.end(), [](auto const &x, auto const &y) { return x.width < y.width; });
  return chemical_potential_t{best.mid, best.eh, best.el, best.el - best.eh, best.N};
}


// ------------------------------------------------------------------------------------------------------------------
// Finite temperature (S8b, notes section 11.6; python line/closure.py electron_count_T, thermal_carriers,
// chemical_potential_T). Energies relative to the same reference as e (mu-relative: a shift).
// ------------------------------------------------------------------------------------------------------------------

/// f(e) = 1 / (e^{beta e} + 1), overflow-free (python fermi: 0.5 (1 - tanh(beta e / 2)))
inline double fermi(double e, double beta) { return 0.5 * (1.0 - std::tanh(0.5 * beta * e)); }

namespace detail {
inline std::vector<double> norm_k_weights(long nk, std::vector<double> const &k_weight) {
  std::vector<double> w(nk, 1.0 / double(nk));
  if (not k_weight.empty()) {
    utils::check(long(k_weight.size()) == nk, "cayley: k_weight size mismatch");
    const double s = std::accumulate(k_weight.begin(), k_weight.end(), 0.0);
    for (long k = 0; k < nk; ++k) w[k] = k_weight[k] / s;
  }
  return w;
}
} // namespace detail

/// N(mu) = 2 sum_k w_k sum_m f(e_m - mu) |v_m|^2 + dropped (Eq. fT_mu)
inline double electron_count_T(std::vector<nda::array<double, 1>> const &e_k, std::vector<nda::array<ComplexType, 2>> const &v_k,
                               double beta, double mu = 0.0, std::vector<double> const &k_weight = {}, double dropped = 0.0) {
  const long nk = long(e_k.size());
  auto wk       = detail::norm_k_weights(nk, k_weight);
  double N      = dropped;
  for (long k = 0; k < nk; ++k) {
    double acc = 0.0;
    for (long m = 0; m < e_k[k].size(); ++m) {
      double w = 0.0;
      for (long i = 0; i < v_k[k].extent(0); ++i) w += std::norm(v_k[k](i, m));
      acc += fermi(e_k[k](m) - mu, beta) * w;
    }
    N += 2.0 * wk[k] * acc;
  }
  return N;
}

/// n_th(mu) of Eq. fT_nth: 2 sum_k w_k [sum_{e > mu} f |v|^2 + sum_{e < mu} (1 - f) |v|^2]
inline double thermal_carriers(std::vector<nda::array<double, 1>> const &e_k, std::vector<nda::array<ComplexType, 2>> const &v_k,
                               double beta, double mu = 0.0, std::vector<double> const &k_weight = {}) {
  const long nk = long(e_k.size());
  auto wk       = detail::norm_k_weights(nk, k_weight);
  double n      = 0.0;
  for (long k = 0; k < nk; ++k) {
    double acc = 0.0;
    for (long m = 0; m < e_k[k].size(); ++m) {
      double w = 0.0;
      for (long i = 0; i < v_k[k].extent(0); ++i) w += std::norm(v_k[k](i, m));
      const double f = fermi(e_k[k](m) - mu, beta);
      acc += (e_k[k](m) > mu ? f : 1.0 - f) * w;
    }
    n += 2.0 * wk[k] * acc;
  }
  return n;
}

/// N(mu) = N_el by bisection to the root in mu (python chemical_potential_T; poles with |e| >= 1e5 are not bracketing
/// candidates). Returns {mu, N(mu)}.
inline std::pair<double, double> chemical_potential_T(std::vector<nda::array<double, 1>> const &e_k,
                                                      std::vector<nda::array<ComplexType, 2>> const &v_k, double nelec, double beta,
                                                      std::vector<double> const &k_weight = {}, double dropped = 0.0,
                                                      long maxit = 200) {
  double emin = 1e300, emax = -1e300;
  for (auto const &e : e_k)
    for (long m = 0; m < e.size(); ++m)
      if (std::abs(e(m)) < 1e5) {
        emin = std::min(emin, e(m));
        emax = std::max(emax, e(m));
      }
  utils::check(emin <= emax, "cayley::chemical_potential_T: no poles");
  double lo = emin - 50.0 / beta - 1.0, hi = emax + 50.0 / beta + 1.0;
  double Nlo = electron_count_T(e_k, v_k, beta, lo, k_weight, dropped) - nelec;
  double mid = lo, Nm = Nlo;
  for (long it = 0; it < maxit; ++it) {
    mid = 0.5 * (lo + hi);
    Nm  = electron_count_T(e_k, v_k, beta, mid, k_weight, dropped) - nelec;
    if (Nm == 0.0 or hi - lo < 4e-16 * std::max(1.0, std::abs(mid))) break;
    if ((Nm < 0) == (Nlo < 0)) {
      lo  = mid;
      Nlo = Nm;
    } else
      hi = mid;
  }
  return {mid, Nm + nelec};
}

} // namespace numerics::line_dlr

#endif
