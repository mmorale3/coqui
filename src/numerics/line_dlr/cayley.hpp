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
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "nda/linalg/det_and_inverse.hpp"

/*
 * LAPACK routines without an nda wrapper: Hermitian divide-and-conquer eigensolver (zheevd) and the complex Schur form
 * (zgees). Standard Fortran-77 signatures (LP64 integers, trailing hidden string lengths omitted as nda's own interface
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
 */
inline upfold_result_t upfold_block(nda::array<ComplexType, 3> const &C, long K, double wp, upfold_opts_t const &o) {
  using nda::range;
  const long n = C.extent(1);
  const double tol_c0 = o.tol_c0, tol_gram = o.tol_gram, tol_svd = o.tol_svd, reject_unity = o.reject_unity;
  const long nphi = o.nphi;
  utils::check(C.extent(0) >= K + 2, "cayley::upfold_block: need moments 0..K+1 (K = {}), got {}", K, C.extent(0));
  utils::check(K >= 1 and nphi >= 1, "cayley::upfold_block: invalid K = {} or nphi = {}", K, nphi);
  utils::check(o.cut_window >= 1.0, "cayley::upfold_block: cut_window must be >= 1");
  upfold_result_t res;

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

  // 2. Gram factorization of the block Toeplitz matrix
  auto T = block_toeplitz(Chat, K);
  utils::check(T.size() > 0, "cayley::upfold_block: moments are empty (C0 rank {})", r);
  for (auto const &x : T) utils::check(std::isfinite(x.real()) and std::isfinite(x.imag()), "cayley::upfold_block: non-finite moments");
  auto lamT = detail::herm_eig(T);                                 // T now holds the eigenvectors (ascending eigenvalues)
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
  cmatrix_F Dm = X(range::all, range(0, K * r)), Dp = X(range::all, range(r, (K + 1) * r));
  auto M = detail::mm(Dp, Dm, 'N', 'C');                            // Nr x Nr
  cmatrix_F P(Nr, Nr), Qh(Nr, Nr);
  nda::array<double, 1> sv(Nr);
  nda::lapack::gesvd(M, sv, P, Qh);
  std::vector<double> xs(Nr);
  for (long i = 0; i < Nr; ++i) xs[i] = (sv(0) > 0.0) ? sv(i) / sv(0) : 0.0;
  res.svd_margin = std::numeric_limits<double>::infinity();
  for (long i = 0; i < Nr; ++i) {
    if (xs[i] > 0.1 * tol_svd and xs[i] < 10.0 * tol_svd) ++res.svd_near;
    if (xs[i] > 0.0) res.svd_margin = std::min(res.svd_margin, std::abs(std::log10(xs[i] / tol_svd)));
  }
  long r1 = detail::cut_boundary(xs, tol_svd, o.svd_cut == "smooth" ? "hard" : o.svd_cut, o.cut_window);
  if (o.svd_cut == "hard") {   // python: count of sv > tol_svd s0 (identical to the boundary for a descending sequence)
    r1 = 0;
    for (long i = 0; i < Nr; ++i)
      if (sv(i) > tol_svd * sv(0)) ++r1;
  }
  if (o.force_r1 >= 0) r1 = std::min(o.force_r1, Nr);
  res.r1     = r1;
  res.n_free = Nr - r1;
  cmatrix_F P1 = P(range::all, range(0, r1)), Q1h = Qh(range(0, r1), range::all);
  cmatrix_F P0 = P(range::all, range(r1, Nr)), Q0h = Qh(range(r1, Nr), range::all);
  auto A1 = detail::mm(P1, Q1h);                                    // P1 Q1^dag
  auto A0 = detail::mm(P0, Q0h);                                    // P0 Q0^dag
  cmatrix_F X0 = X(range::all, range(0, r));
  auto R = detail::mm(B, X0, 'N', 'C');                             // n x Nr

  cmatrix_F Cheld(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) Cheld(i, j) = C(K + 1, i, j);
  const double nheld = detail::frob(Cheld);

  struct realization_t {
    double err;
    nda::array<ComplexType, 1> u;
    cmatrix_F W;
  };
  auto realize = [&](double phi) {
    cmatrix_F U(Nr, Nr);
    const ComplexType eph = std::exp(ComplexType(0.0, phi));
    for (long j = 0; j < Nr; ++j)
      for (long i = 0; i < Nr; ++i) U(i, j) = A1(i, j) + (res.n_free > 0 ? eph * A0(i, j) : ComplexType(0.0));
    cmatrix_F Z;
    auto u = detail::schur(U, Z);
    auto W = detail::mm(R, Z);
    cmatrix_F Wu(n, Nr);
    for (long l = 0; l < Nr; ++l) {
      ComplexType uk(1.0);
      for (long p = 0; p < K + 1; ++p) uk *= u(l);
      for (long i = 0; i < n; ++i) Wu(i, l) = W(i, l) * uk;
    }
    auto Ck = detail::mm(Wu, W, 'N', 'C');
    Ck -= Cheld;
    return realization_t{detail::frob(Ck) / (1.0 + nheld), std::move(u), std::move(W)};
  };

  realization_t best;
  double phi_best = 0.0;
  if (res.n_free == 0) {
    best = realize(0.0);
  } else if (std::isfinite(o.force_phi)) {
    phi_best = o.force_phi;
    best     = realize(phi_best);
  } else {
    const double twopi = 2.0 * std::numbers::pi;
    std::vector<double> errs(nphi);
    for (long ip = 0; ip < nphi; ++ip) {
      const double phi = twopi * double(ip) / double(nphi);       // np.linspace(0, 2 pi, nphi, endpoint=False)
      auto rz          = realize(phi);
      double umin      = std::numeric_limits<double>::infinity();
      for (auto const &ul : rz.u) umin = std::min(umin, std::abs(ul - 1.0));
      errs[ip] = (umin < reject_unity) ? std::numeric_limits<double>::infinity() : rz.err;
      if (umin < reject_unity) ++res.n_rejected;
    }
    long i0 = long(std::min_element(errs.begin(), errs.end()) - errs.begin());
    utils::check(std::isfinite(errs[i0]), "cayley::upfold_block: no admissible terminal phase");
    double phi0 = twopi * double(i0) / double(nphi);
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
    // golden-section refinement around the grid minimum (30 steps, as upfold.py)
    double a = phi0 - twopi / double(nphi), b = phi0 + twopi / double(nphi);
    auto f          = [&](double p) { return realize(p).err; };
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = f(c), fd = f(d);
    for (int it = 0; it < 30; ++it) {
      if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
      else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    phi_best = 0.5 * (a + b);
    best     = realize(phi_best);
  }
  res.phi      = phi_best;
  res.residual = best.err;

  // 3. poles (mu-relative), sorted ascending
  std::vector<double> dl(Nr);
  for (long l = 0; l < Nr; ++l) dl[l] = cayley_inverse(best.u(l), wp);
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
                         nda::array<ComplexType, 2> const &W) {
  const long n = Hrel.extent(0);
  cmatrix_F Ht(upfolded_hamiltonian(Hrel, d, W));
  auto e     = detail::herm_eig(Ht);
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

} // namespace numerics::line_dlr

#endif
