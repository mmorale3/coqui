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
 * Memory: history A^(i) and R^(i) are stored as math::shm::shared_array
 * (one copy per node) -- with a window of 8, this is 16 * sizeof(A) per
 * node instead of per rank. Frobenius inner products read shared memory
 * directly on each rank (same data, redundant compute is cheap relative
 * to the SCF iteration cost) so no node-level reduction is required for
 * the dot products themselves; an allreduce across ranks would be needed
 * only if A were distributed (it isn't).
 */

#ifndef COQUI_REAL_AXIS_DIIS_HPP
#define COQUI_REAL_AXIS_DIIS_HPP

#include <complex>
#include <deque>
#include <memory>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/shared_array/nda.hpp"

namespace methods {
namespace real_axis {

/**
 * Small, self-contained DIIS history + mix routine for rank-5 complex
 * iterates with shared-memory storage of the history.
 */
class diis_mixer_t
{
 public:
  using mpi_context_t = utils::mpi_context_t<boost::mpi3::communicator>;
  using sArray5       = math::shm::shared_array<nda::array_view<ComplexType, 5>>;

  /**
   * @param mpi             MPI context (for shared-memory allocation).
   * @param m_max           maximum history length; common choice is 6-10.
   * @param cond_threshold  relative-condition threshold for the (n+1)x(n+1)
   *                        solve. If the smallest |B_ii| / largest |B_ii|
   *                        falls below this, the history is restarted.
   */
  explicit diis_mixer_t(std::shared_ptr<mpi_context_t> mpi,
                        long m_max = 8,
                        double cond_threshold = 1e-14)
    : _mpi(std::move(mpi)),
      _m_max(m_max),
      _cond_threshold(cond_threshold)
  {
    utils::check(_mpi != nullptr, "diis_mixer_t: mpi context must not be null");
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
   * One DIIS mixing step. A_old, A_full, and A_new are sArray5 (one copy
   * per node). On entry, A_new may alias A_full (mix in place); on exit
   * A_new holds the DIIS-extrapolated next iterate.
   *
   * @param A_old  current iterate A^(k); pushed to history.
   * @param A_full un-mixed Dyson output A_full^(k+1).
   * @param alpha  damping parameter applied to R^(i) inside the sum.
   * @param A_new  OUTPUT: the DIIS-extrapolated next iterate.
   *
   * @return residual norm sqrt(<R, R>) used as the SCF diagnostic.
   */
  double mix(sArray5 const& A_old, sArray5 const& A_full, double alpha,
             sArray5 & A_new)
  {
    utils::check(A_old.shape() == A_full.shape(),
                 "diis_mixer_t::mix: A_old and A_full shapes differ");
    utils::check(A_new.shape() == A_old.shape(),
                 "diis_mixer_t::mix: A_new shape mismatch");

    auto shape = A_old.shape();

    // Allocate sArrays for the snapshot of A_old and the new residual R.
    auto A_snap = std::make_unique<sArray5>(*_mpi, shape);
    auto R_snap = std::make_unique<sArray5>(*_mpi, shape);

    // Fill on node-root.
    {
      auto Ao = A_old.local();
      auto Af = A_full.local();
      if (A_snap->node_comm()->root()) {
        A_snap->local() = Ao;
        auto R_loc = R_snap->local();
        const long N = R_loc.size();
        auto * dR = R_loc.data();
        auto const* da = Ao.data();
        auto const* df = Af.data();
        for (long i = 0; i < N; ++i) dR[i] = df[i] - da[i];
      }
      A_snap->node_sync();
      R_snap->node_sync();
    }

    // Push to history; evict oldest if window full.
    _A_hist.push_back(std::move(A_snap));
    _R_hist.push_back(std::move(R_snap));
    while (static_cast<long>(_A_hist.size()) > _m_max) {
      _A_hist.pop_front();
      _R_hist.pop_front();
    }

    const long n = static_cast<long>(_A_hist.size());

    // Residual diagnostic: ||R^(latest)||_F. Read shared memory directly.
    double res_norm = 0.0;
    {
      auto R_loc = _R_hist.back()->local();
      auto const* dR = R_loc.data();
      const long N = R_loc.size();
      for (long i = 0; i < N; ++i) res_norm += std::norm(dR[i]);
      res_norm = std::sqrt(res_norm);
    }

    // n=1: nothing to extrapolate. Linear mix into A_new.
    if (n < 2) {
      linear_mix(A_old, A_full, alpha, A_new);
      return res_norm;
    }

    // Build the (n+1) x (n+1) augmented Pulay matrix.
    nda::matrix<double> M(n + 1, n + 1);
    M = 0.0;
    for (long i = 0; i < n; ++i) {
      auto Ri = _R_hist[i]->local();
      auto const* di = Ri.data();
      const long N = Ri.size();
      for (long j = i; j < n; ++j) {
        auto Rj = _R_hist[j]->local();
        auto const* dj = Rj.data();
        double acc = 0.0;
        for (long k = 0; k < N; ++k) {
          acc += di[k].real() * dj[k].real()
               + di[k].imag() * dj[k].imag();
        }
        M(i, j) = acc;
        M(j, i) = acc;
      }
    }
    for (long i = 0; i < n; ++i) { M(i, n) = 1.0; M(n, i) = 1.0; }
    M(n, n) = 0.0;

    // Detect ill-conditioning -> restart on linear dependence.
    {
      double diag_max = 0.0, diag_min = std::numeric_limits<double>::infinity();
      for (long i = 0; i < n; ++i) {
        diag_max = std::max(diag_max, M(i, i));
        diag_min = std::min(diag_min, M(i, i));
      }
      if (diag_max > 0.0 and (diag_min / diag_max) < _cond_threshold) {
        // Drop all but the latest entry; fall back to linear mixing.
        while (_A_hist.size() > 1) _A_hist.pop_front();
        while (_R_hist.size() > 1) _R_hist.pop_front();
        linear_mix(A_old, A_full, alpha, A_new);
        return res_norm;
      }
    }

    // Solve M * x = b where b = (0, ..., 0, 1).
    nda::matrix<double> Minv = M;
    nda::inverse_in_place(Minv);
    nda::vector<double> c(n);
    for (long i = 0; i < n; ++i) c(i) = Minv(i, n);

    // A_new = sum_i c_i * (A^(i) + alpha * R^(i)). Write on node root.
    if (A_new.node_comm()->root()) {
      auto An_loc = A_new.local();
      An_loc = ComplexType(0.0, 0.0);
      const long N = An_loc.size();
      auto * dn = An_loc.data();
      for (long h = 0; h < n; ++h) {
        const double ci = c(h);
        auto Ah = _A_hist[h]->local();
        auto Rh = _R_hist[h]->local();
        auto const* da = Ah.data();
        auto const* dr = Rh.data();
        for (long k = 0; k < N; ++k)
          dn[k] += ci * (da[k] + alpha * dr[k]);
      }
    }
    A_new.node_sync();
    return res_norm;
  }

 private:
  /// Helper: A_new = (1 - alpha) * A_old + alpha * A_full, on node root.
  void linear_mix(sArray5 const& A_old, sArray5 const& A_full, double alpha,
                  sArray5 & A_new)
  {
    if (A_new.node_comm()->root()) {
      auto Ao = A_old.local();
      auto Af = A_full.local();
      auto An = A_new.local();
      const long N = An.size();
      auto * dn = An.data();
      auto const* da = Ao.data();
      auto const* df = Af.data();
      for (long i = 0; i < N; ++i)
        dn[i] = (1.0 - alpha) * da[i] + alpha * df[i];
    }
    A_new.node_sync();
  }

  std::shared_ptr<mpi_context_t> _mpi;
  long _m_max;
  double _cond_threshold;
  std::deque<std::unique_ptr<sArray5>> _A_hist;
  std::deque<std::unique_ptr<sArray5>> _R_hist;
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DIIS_HPP
