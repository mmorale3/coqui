/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 *
 * Pulay-style DIIS (Direct Inversion of the Iterative Subspace) mixer for
 * the real-axis spectral function during scGW.
 *
 * At iteration k, given the current iterate A^(k) and the un-mixed Dyson
 * output A_full^(k+1) = Dyson(Sigma[A^(k)]), we form the residual
 *   R^(k) = A_full^(k+1) - A^(k).
 * DIIS keeps a sliding window of the last m pairs (A^(i), R^(i)) and
 * computes the next iterate as
 *   A^(k+1) = sum_i c_i (A^(i) + alpha * R^(i))
 * where the coefficients (c_0, ..., c_{n-1}) minimise || sum_i c_i R^(i) ||^2
 * under the constraint sum_i c_i = 1. This is the standard Pulay DIIS;
 * setting alpha=1 and n=2 recovers the original linear-only Pulay scheme,
 * and falling back to a single history entry recovers plain linear mixing
 * A^(k+1) = (1 - alpha) A^(k) + alpha A_full^(k+1).
 *
 * The DIIS linear system is small ((n+1) x (n+1)) and is solved by
 * `nda::inverse_in_place`. The Frobenius inner product on residuals is
 * taken as a real symmetric positive-semidefinite Gram matrix
 *   B_ij = Re < R^(i), R^(j) >.
 * If the augmented matrix is detected to be too ill-conditioned, the
 * history is restarted (cleared and the call falls back to linear mixing).
 *
 * MPI: the iterate A and residual R are fully replicated on every rank
 * (after the allreduces in evaluate_serial / Sigma_x / Dyson). The inner
 * product is computed locally — no MPI traffic required during DIIS.
 */

#ifndef COQUI_REAL_AXIS_DIIS_HPP
#define COQUI_REAL_AXIS_DIIS_HPP

#include <complex>
#include <deque>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis {

/**
 * Small, self-contained DIIS history + mix routine for rank-5 complex
 * iterates. The class is templated on the array rank only because it's
 * useful for both A_wskij (rank 5) and other potential rank-5 quantities;
 * here we only instantiate the rank=5 case via `diis_mixer_t` below.
 */
class diis_mixer_t
{
 public:
  using array_t = nda::array<ComplexType, 5>;

  /**
   * @param m_max         maximum history length; common choice is 6-10.
   * @param cond_threshold relative-condition threshold for the (n+1)x(n+1)
   *                       solve. If the smallest |B_ii| / largest |B_ii|
   *                       falls below this, the history is restarted.
   */
  explicit diis_mixer_t(long m_max = 8, double cond_threshold = 1e-14)
    : _m_max(m_max), _cond_threshold(cond_threshold)
  {
    utils::check(m_max >= 2, "diis_mixer_t: m_max must be >= 2");
  }

  /// Return current history depth.
  long size() const { return static_cast<long>(_A_hist.size()); }

  /// Drop all history (e.g. after a restart).
  void reset()
  {
    _A_hist.clear();
    _R_hist.clear();
  }

  /**
   * One DIIS mixing step.
   *
   * @param A_old  current iterate A^(k); pushed to history.
   * @param A_full un-mixed Dyson output A_full^(k+1).
   * @param alpha  damping parameter applied to R^(i) inside the sum.
   *               alpha=1 is pure Pulay; smaller alpha damps the
   *               extrapolation (useful very early in SCF).
   * @param A_new  OUTPUT: the DIIS-extrapolated next iterate.
   *
   * @return residual norm sqrt(<R, R>) used as the SCF diagnostic.
   */
  double mix(array_t const& A_old, array_t const& A_full, double alpha,
             array_t & A_new)
  {
    utils::check(A_old.shape() == A_full.shape(),
                 "diis_mixer_t::mix: A_old and A_full shapes differ");
    utils::check(A_new.shape() == A_old.shape(),
                 "diis_mixer_t::mix: A_new shape mismatch");

    // Build R = A_full - A_old (full size; cheap).
    array_t R(A_old.shape());
    {
      const long N = R.size();
      auto * dR = R.data();
      auto const* da = A_old.data();
      auto const* df = A_full.data();
      for (long i = 0; i < N; ++i) dR[i] = df[i] - da[i];
    }

    // Push current (A_old, R) to history, evict the oldest if window full.
    _A_hist.push_back(A_old);
    _R_hist.push_back(std::move(R));
    while (static_cast<long>(_A_hist.size()) > _m_max) {
      _A_hist.pop_front();
      _R_hist.pop_front();
    }

    const long n = static_cast<long>(_A_hist.size());

    // Residual diagnostic: ||R^(latest)||_F.
    double res_norm = 0.0;
    {
      auto const* dR = _R_hist.back().data();
      const long N = _R_hist.back().size();
      for (long i = 0; i < N; ++i)
        res_norm += std::norm(dR[i]);  // |R_i|^2
      res_norm = std::sqrt(res_norm);
    }

    // n=1: nothing to extrapolate from. Fall back to linear mix.
    if (n < 2) {
      const long N = A_old.size();
      auto * da_new = A_new.data();
      auto const* da_old = A_old.data();
      auto const* da_full = A_full.data();
      for (long i = 0; i < N; ++i)
        da_new[i] = (1.0 - alpha) * da_old[i] + alpha * da_full[i];
      return res_norm;
    }

    // Build the (n+1) x (n+1) augmented Pulay matrix.
    // [ B   1 ] [ c     ]   [ 0 ]
    // [ 1^T 0 ] [ lambda] = [ 1 ]
    // with B_ij = Re < R^(i), R^(j) >.
    nda::matrix<double> M(n + 1, n + 1);
    M = 0.0;
    for (long i = 0; i < n; ++i) {
      for (long j = i; j < n; ++j) {
        const long N = _R_hist[i].size();
        auto const* di = _R_hist[i].data();
        auto const* dj = _R_hist[j].data();
        double acc = 0.0;
        for (long k = 0; k < N; ++k) {
          // Re(conj(R_i) * R_j) = Re(R_i)*Re(R_j) + Im(R_i)*Im(R_j).
          acc += di[k].real() * dj[k].real()
               + di[k].imag() * dj[k].imag();
        }
        M(i, j) = acc;
        M(j, i) = acc;
      }
    }
    // Augmenting Lagrange multiplier row/column.
    for (long i = 0; i < n; ++i) { M(i, n) = 1.0; M(n, i) = 1.0; }
    M(n, n) = 0.0;

    // Detect ill-conditioning by checking that the diagonal of B is not
    // collapsing (i.e. one of the residuals has become much smaller than
    // the others -- indicates linear dependence). If so, restart.
    {
      double diag_max = 0.0, diag_min = std::numeric_limits<double>::infinity();
      for (long i = 0; i < n; ++i) {
        diag_max = std::max(diag_max, M(i, i));
        diag_min = std::min(diag_min, M(i, i));
      }
      if (diag_max > 0.0 and (diag_min / diag_max) < _cond_threshold) {
        // Drop all but the latest entry; fall back to linear mixing.
        _A_hist.erase(_A_hist.begin(), _A_hist.end() - 1);
        _R_hist.erase(_R_hist.begin(), _R_hist.end() - 1);
        const long N = A_old.size();
        auto * da_new = A_new.data();
        auto const* da_old = A_old.data();
        auto const* da_full = A_full.data();
        for (long i = 0; i < N; ++i)
          da_new[i] = (1.0 - alpha) * da_old[i] + alpha * da_full[i];
        return res_norm;
      }
    }

    // Solve M * x = b where b = (0, ..., 0, 1). Use inverse_in_place since M
    // is small ((n+1) x (n+1), n<=10), so the cost is trivial.
    nda::matrix<double> Minv = M;
    nda::inverse_in_place(Minv);

    // c_i = (M^{-1})_{i, n}  (last column).
    nda::vector<double> c(n);
    for (long i = 0; i < n; ++i) c(i) = Minv(i, n);

    // A_new = sum_i c_i * (A^(i) + alpha * R^(i)).
    A_new = ComplexType(0.0, 0.0);
    {
      const long N = A_new.size();
      auto * da_new = A_new.data();
      for (long h = 0; h < n; ++h) {
        const double ci = c(h);
        auto const* da = _A_hist[h].data();
        auto const* dr = _R_hist[h].data();
        for (long k = 0; k < N; ++k)
          da_new[k] += ci * (da[k] + alpha * dr[k]);
      }
    }
    return res_norm;
  }

 private:
  long _m_max;
  double _cond_threshold;
  std::deque<array_t> _A_hist;
  std::deque<array_t> _R_hist;
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DIIS_HPP
