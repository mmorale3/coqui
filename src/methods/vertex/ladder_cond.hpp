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

#ifndef COQUI_VERTEX_LADDER_COND_HPP
#define COQUI_VERTEX_LADDER_COND_HPP

/**
 * The conditioning meter of the static ladder resolvent of the dynamic BSE vertex.
 *
 * The dynamic solver (vertex_dynbse.icc, pair_space_ladder_dyn) applies T_s = K_s M^-1 with M = 1 - Cb K_s on every
 * (spin, q, nu) unit. At nu = 0 M approaches singularity at the excitonic instability of the static ladder: the resolvent
 * then multiplies every error of its inputs (the tau refits, the pole fits of G, the THC rung) by ~1/sigma_min(M), so a
 * pass with a small sigma_min(M) produces noise-amplified vertex corrections even though every solve "converges".
 *
 * THE ESTIMATE. sigma_min(M)^-2 is the largest eigenvalue of the Hermitian positive matrix A = M^-H M^-1. Block Lanczos
 * on A (block size nb, full reorthogonalization, Rayleigh-Ritz on V^H A V), from a fixed deterministic start block (the
 * same sequence on every rank and on host and device), gives its largest Ritz value theta_max <= lambda_max(A), i.e. an
 * estimate sigma_min ~ theta_max^-1/2 that can only err HIGH, with the Ritz residual r = |A u - theta_max u| as the
 * stopping test (r <= tol theta_max: then an eigenvalue of A lies within r of theta_max). Each block step applies A to nb
 * vectors: two triangular-solve pairs (getrs with nb right-hand sides) on the LU factors the solver keeps anyway --
 * O(nb D^2), no SVD, no extra O(D^3) work. The block form exists for the host: MKL's getrs with a few right-hand sides
 * is a memory-bound sweep that does not thread (e.g. at D = 9216: 0.23 s per getrs pair with one right-hand
 * side, 0.45 s with eight, independent of the thread count), so blocks of 8 cost ~4x less per vector and converge in
 * fewer sweeps. In the dense-inverse mode of the host
 * resolvent the same A is applied by two gemm with the explicit M^-1 before it is discarded (the same O(nb D^2) per step).
 * The smallest Ritz value gives theta_min >= lambda_min(A), i.e. sigma_max_lb = theta_min^-1/2 <= sigma_max(M): only a
 * LOOSE lower bound of sigma_max (Lanczos on A converges slowly at that end once a small sigma_min spreads the spectrum --
 * a factor ~25 low at sigma_min ~ 1e-4 in the unit test), so it is kept in the estimate but no condition number is logged:
 * sigma_min alone is the meter (sigma_max(1 - Cb K_s) is O(1) in practice).
 *
 * Every LU stores the column-major factors of B = M^T (the C-layout host matrix as LAPACK sees it; the device unit builds
 * M^T explicitly). sigma(B) = sigma(M), so the meter runs on B: x -> B^-1 x (op N) -> B^-H (op C).
 *
 * BOOKKEEPING. A ladder pass (the P refresh in update_w, the Sigma pass of eval_sigma_pair_dyn) opens a record set with
 * begin(label); every driver call inside it adds one record per metered unit on the rank that solved it; report(comm)
 * gathers them over the ranks and writes ONE log line (minimum sigma_min with its (s, q, nu, driver call), median and
 * maximum over the units, the meter's cost against the pass wall time), warns below vertex_debug ladder_cond_warn
 * (default 1e-2) and aborts below vertex_debug ladder_cond_abort (default 0 = off). The summaries of the passes wait in
 * pending() for the caller that writes them next to the iteration's other diagnostics (scf/iter<N>/ladder_cond).
 *
 * vertex_debug keys: ladder_cond_warn (1e-2), ladder_cond_abort (0 = off), ladder_cond_iters (block-Lanczos steps, 12; 0
 * turns the meter off, logged), ladder_cond_block (the block size, 8), ladder_cond_all_nu (0: nu = 0 units only; 1: every
 * unit of the pass).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "mpi3/communicator.hpp"
#include "methods/vertex/vertex_debug.hpp"

namespace methods {
namespace solvers {
namespace ladder_cond {

  using cplx = ComplexType;

  inline double wall_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

  /** the result of one meter run */
  struct estimate {
    double smin = -1.0;       // sigma_min(M) estimate (theta_max^-1/2; can only err high)
    double smax_lb = -1.0;    // a lower bound of sigma_max(M) (theta_min^-1/2)
    double resid = -1.0;      // the Ritz residual of theta_max relative to theta_max
    long steps = 0;           // block-Lanczos steps
    long napp = 0;            // vectors A was applied to
    bool converged = false;   // resid <= tol (or an invariant subspace)
    double seconds = 0.0;
  };

  /** the block-Lanczos steps per meter run (vertex_debug ladder_cond_iters, default 12; 0 = the meter is off) */
  inline long kmax() { return long(vertex_debug::number("ladder_cond_iters", 12.0)); }   // vertex_debug: ladder_cond_iters
  /** the block size (vertex_debug ladder_cond_block, default 8): the right-hand sides of every getrs / gemm of the meter */
  inline long block() { return std::max(1l, long(vertex_debug::number("ladder_cond_block", 8.0))); }   // vertex_debug: ladder_cond_block
  /** the stopping test on the relative Ritz residual of theta_max (sigma_min to ~tol / 2) */
  inline constexpr double tol = 1e-3;
  /** meter this unit? nu = 0 units always (unless the meter is off); the others with vertex_debug ladder_cond_all_nu = 1 */
  inline bool wanted(bool nu0) {
    if (kmax() <= 0) return false;
    return nu0 or vertex_debug::flag("ladder_cond_all_nu");   // vertex_debug: ladder_cond_all_nu
  }

  /**
   * Block Lanczos on A = M^-H M^-1 (D x D, Hermitian positive definite): the block Krylov basis V (orthonormal, full
   * reorthogonalization), W = A V kept, the Rayleigh-Ritz pair of H = V^H W, the residual |W y - theta V y| of its top pair.
   * apply(X, Y, n): Y = A X for n column-major D-vectors (host pointers; X is not modified). kmax block steps of nb vectors.
   */
  template <class ApplyA>
  inline estimate lanczos_inverse_gram(long D, ApplyA &&apply, long kmax_in, long nb_in, double tol_in = tol) {
    estimate est;
    const double t0 = wall_now();
    const long nb = std::max(1l, std::min(nb_in, D));
    const long k = std::max(1l, std::min(kmax_in, D / nb));
    const long mmax = k * nb;
    std::vector<cplx> V(static_cast<size_t>((mmax + nb) * D)), W(static_cast<size_t>(mmax * D)),
        H(static_cast<size_t>(mmax * mmax)), C(static_cast<size_t>(mmax * nb)), r(static_cast<size_t>(D));
    // the fixed start block: quasi-random magnitudes and phases (Weyl sequences, one pair of irrational steps per column) --
    // the same on every rank and on host and device; no library RNG, so every build produces the same sequence
    for (long c = 0; c < nb; ++c) {
      const double fa = std::fmod(0.6180339887498949 + 0.7071067811865476 * double(c), 1.0);
      const double fb = std::fmod(0.4142135623730951 + 0.3247179572447460 * double(c), 1.0);
      for (long i = 0; i < D; ++i) {
        const double a = std::fmod(fa * double(i + 1), 1.0), b = std::fmod(fb * double(i + 1), 1.0);
        V[size_t(c * D + i)] = (0.5 + a) * std::exp(cplx(0.0, 2.0 * M_PI * b));
      }
    }
    // orthonormalize columns [m0, m1) of V against [0, m0) (block classical Gram-Schmidt, two passes, gemm) and among
    // themselves (modified Gram-Schmidt, two passes); linearly dependent columns are dropped; returns the columns kept
    auto ortho = [&](long m0, long m1) {
      const long nn = m1 - m0;
      cplx *Z = V.data() + m0 * D;
      for (int pass = 0; pass < 2 and m0 > 0; ++pass) {
        nda::blas::f77::gemm('C', 'N', int(m0), int(nn), int(D), cplx(1.0), V.data(), int(D), Z, int(D), cplx(0.0), C.data(), int(m0));
        nda::blas::f77::gemm('N', 'N', int(D), int(nn), int(m0), cplx(-1.0), V.data(), int(D), C.data(), int(m0), cplx(1.0), Z, int(D));
      }
      long kept = m0;
      for (long c = m0; c < m1; ++c) {
        cplx *v = V.data() + c * D;
        double n0 = 0.0;
        for (long i = 0; i < D; ++i) n0 += std::norm(v[i]);
        n0 = std::sqrt(n0);
        for (int pass = 0; pass < 2; ++pass)
          for (long cc = m0; cc < kept; ++cc) {
            cplx const *u = V.data() + cc * D;
            cplx dot(0.0);
            for (long i = 0; i < D; ++i) dot += std::conj(u[i]) * v[i];
            for (long i = 0; i < D; ++i) v[i] -= dot * u[i];
          }
        double n1 = 0.0;
        for (long i = 0; i < D; ++i) n1 += std::norm(v[i]);
        n1 = std::sqrt(n1);
        if (n1 <= 1e-10 * std::max(n0, 1e-300)) continue;          // linearly dependent: dropped
        cplx *dst = V.data() + kept * D;
        for (long i = 0; i < D; ++i) dst[i] = v[i] / n1;
        ++kept;
      }
      return kept;
    };
    long nv = ortho(0, nb), m = 0;
    for (long j = 0; j < k and nv > m; ++j) {
      const long nn = nv - m, m0 = m;
      apply(V.data() + m0 * D, W.data() + m0 * D, nn);
      m = nv;
      // H = V^H W (m x m, column-major in an mmax x mmax array): the new columns by one gemm, the new rows by Hermitian symmetry
      nda::blas::f77::gemm('C', 'N', int(m), int(nn), int(D), cplx(1.0), V.data(), int(D), W.data() + m0 * D, int(D), cplx(0.0),
                           H.data() + m0 * mmax, int(mmax));
      for (long a = m0; a < m; ++a)
        for (long b = 0; b < m0; ++b) H[size_t(b * mmax + a)] = std::conj(H[size_t(a * mmax + b)]);
      nda::matrix<cplx> Hm(m, m);
      for (long a = 0; a < m; ++a)
        for (long b = 0; b < m; ++b) Hm(a, b) = 0.5 * (H[size_t(b * mmax + a)] + std::conj(H[size_t(a * mmax + b)]));
      auto [th, Y] = nda::linalg::eigenelements(Hm);
      const double th_max = th(m - 1), th_min = th(0);
      utils::check(th_max > 0.0 and std::isfinite(th_max), "ladder_cond: non-positive / non-finite Ritz value {} of (M^H M)^-1 "
                   "(the LU factors hold NaN / inf, or M is singular).", th_max);
      // the top Ritz pair's residual: r = W y - theta V y
      std::vector<cplx> y(static_cast<size_t>(m));
      for (long a = 0; a < m; ++a) y[size_t(a)] = Y(a, m - 1);   // (the eigenvector of the Hermitized H)
      nda::blas::f77::gemv('N', int(D), int(m), cplx(1.0), W.data(), int(D), y.data(), 1, cplx(0.0), r.data(), 1);
      nda::blas::f77::gemv('N', int(D), int(m), cplx(-th_max), V.data(), int(D), y.data(), 1, cplx(1.0), r.data(), 1);
      double rn = 0.0;
      for (long i = 0; i < D; ++i) rn += std::norm(r[size_t(i)]);
      est.smin = 1.0 / std::sqrt(th_max);
      est.smax_lb = (th_min > 0.0) ? 1.0 / std::sqrt(th_min) : -1.0;
      est.resid = std::sqrt(rn) / th_max;
      est.steps = j + 1;
      est.napp = m;
      if (est.resid <= tol_in) { est.converged = true; break; }
      if (j + 1 == k) break;
      // the next block: the new columns of W, orthonormalized against the basis
      std::copy(W.data() + (m - nn) * D, W.data() + m * D, V.data() + m * D);
      nv = ortho(m, m + nn);
      if (nv == m) { est.converged = true; break; }                // an invariant subspace: the Ritz values are exact
    }
    est.seconds = wall_now() - t0;
    return est;
  }

  /** A = B^-H B^-1 on n columns from the column-major LU factors of B (LAPACK getrf; B = M^T for the C-layout host M) */
  inline void apply_lu(long D, cplx const *LU, int const *ipiv, cplx const *X, cplx *Y, long n) {
    std::copy(X, X + D * n, Y);
    int info = 0;
    nda::lapack::f77::getrs('N', int(D), int(n), LU, int(D), ipiv, Y, int(D), info);
    utils::check(info == 0, "ladder_cond: getrs (op N) failed (info {}).", info);
    nda::lapack::f77::getrs('C', int(D), int(n), LU, int(D), ipiv, Y, int(D), info);
    utils::check(info == 0, "ladder_cond: getrs (op C) failed (info {}).", info);
  }
  /** A = B^-H B^-1 on n columns from the explicit column-major inverse Bi = B^-1 (two gemm); Z: D x n scratch */
  inline void apply_inverse(long D, cplx const *Bi, cplx const *X, cplx *Y, long n, cplx *Z) {
    nda::blas::f77::gemm('N', 'N', int(D), int(n), int(D), cplx(1.0), Bi, int(D), X, int(D), cplx(0.0), Z, int(D));
    nda::blas::f77::gemm('C', 'N', int(D), int(n), int(D), cplx(1.0), Bi, int(D), Z, int(D), cplx(0.0), Y, int(D));
  }

  // ---------------------------------------------------------------------------------------------------------------------
  // per-pass bookkeeping
  // ---------------------------------------------------------------------------------------------------------------------

  /** one metered unit */
  struct record {
    long is = 0, iq = 0, m = 0, call = 0;
    bool nu0 = true;
    estimate est;
  };

  /** the reduced summary of one pass */
  struct summary {
    bool valid = false;            // at least one unit was metered
    std::string label;
    long calls = 0, n = 0, n_unconv = 0, steps_max = 0;
    double smin = -1.0, smed = -1.0, smax = -1.0;
    long is = -1, iq = -1, m = -1, call = -1;   // the location of the minimum
    double t_meter = 0.0, t_pass = 0.0;         // the slowest rank's meter time; the pass wall time
    bool all_nu = false;
  };

  struct pass_state {
    bool open = false;
    std::string label;
    long calls = 0;
    double t0 = 0.0;
    std::vector<record> rec;
  };
  inline pass_state &state() { static pass_state s; return s; }
  inline std::vector<summary> &pending() { static std::vector<summary> p; return p; }

  /** open a pass (every rank; drops whatever an unreported pass left) */
  inline void begin(std::string const &label) {
    auto &s = state();
    s = pass_state{};
    s.open = true;
    s.label = label;
    s.t0 = wall_now();
  }
  /** a driver call inside the open pass: returns its 1-based index (0 when no pass is open) */
  inline long next_call() {
    auto &s = state();
    if (not s.open) return 0;
    return ++s.calls;
  }
  inline void add(record const &r) {
    auto &s = state();
    if (s.open) s.rec.push_back(r);
  }

  /** collective over comm: gather the pass's records, log the summary line, warn / abort, queue the summary in pending() */
  inline summary report(boost::mpi3::communicator &comm) {
    auto &s = state();
    summary out;
    out.label = s.label;
    out.all_nu = vertex_debug::flag("ladder_cond_all_nu");
    if (not s.open) return out;
    out.calls = s.calls;
    const long nproc = comm.size(), rank = comm.rank();
    constexpr long NF = 11;
    std::vector<double> cnt(static_cast<size_t>(nproc), 0.0);
    cnt[size_t(rank)] = double(s.rec.size());
    comm.all_reduce_in_place_n(cnt.data(), cnt.size(), std::plus<>{});
    long off = 0, tot = 0;
    for (long r = 0; r < nproc; ++r) { if (r < rank) off += long(cnt[size_t(r)]); tot += long(cnt[size_t(r)]); }
    std::vector<double> buf(static_cast<size_t>(std::max(1l, tot * NF)), 0.0);
    double t_meter = 0.0;
    for (size_t i = 0; i < s.rec.size(); ++i) {
      auto const &r = s.rec[i];
      double *row = buf.data() + (off + long(i)) * NF;
      row[0] = double(r.is); row[1] = double(r.iq); row[2] = double(r.m); row[3] = double(r.call); row[4] = r.nu0 ? 1.0 : 0.0;
      row[5] = r.est.smin; row[6] = r.est.smax_lb; row[7] = r.est.resid; row[8] = double(r.est.steps);
      row[9] = r.est.converged ? 1.0 : 0.0; row[10] = r.est.seconds;
      t_meter += r.est.seconds;
    }
    if (tot > 0) comm.all_reduce_in_place_n(buf.data(), size_t(tot * NF), std::plus<>{});
    out.t_meter = comm.all_reduce_value(t_meter, boost::mpi3::max<>{});
    out.t_pass = comm.all_reduce_value(wall_now() - s.t0, boost::mpi3::max<>{});
    out.n = tot;
    s = pass_state{};                                   // the pass is closed
    const double warn = vertex_debug::number("ladder_cond_warn", 1e-2);    // vertex_debug: ladder_cond_warn
    const double abort = vertex_debug::number("ladder_cond_abort", 0.0);   // vertex_debug: ladder_cond_abort
    if (out.calls == 0) return out;                     // no driver call in this pass: nothing to say
    if (tot == 0) {
      app_log(1, "  [ladder cond] {}: no unit was metered in this pass ({} driver calls; the meter covers the nu = 0 units{}){}",
              out.label, out.calls, out.all_nu ? " and every other unit" : "",
              kmax() <= 0 ? " -- the meter is OFF (vertex_debug ladder_cond_iters = 0)" : "");
      return out;
    }
    std::vector<double> sm(static_cast<size_t>(tot));
    long imin = 0;
    for (long u = 0; u < tot; ++u) {
      double const *row = buf.data() + u * NF;
      sm[size_t(u)] = row[5];
      if (row[5] < buf[size_t(imin * NF + 5)]) imin = u;
      if (row[9] == 0.0) ++out.n_unconv;
      out.steps_max = std::max(out.steps_max, long(row[8]));
    }
    std::vector<double> srt = sm;
    std::sort(srt.begin(), srt.end());
    out.valid = true;
    out.smin = srt.front(); out.smax = srt.back();
    out.smed = (tot % 2 == 1) ? srt[size_t(tot / 2)] : 0.5 * (srt[size_t(tot / 2 - 1)] + srt[size_t(tot / 2)]);
    double const *rm = buf.data() + imin * NF;
    out.is = long(rm[0]); out.iq = long(rm[1]); out.m = long(rm[2]); out.call = long(rm[3]);
    const bool loc_nu0 = (rm[4] != 0.0);
    std::string loc = "(s " + std::to_string(out.is) + ", q " + std::to_string(out.iq) +
                      (loc_nu0 ? ", nu = 0" : ", nu node " + std::to_string(out.m)) +
                      (out.calls > 1 ? ", driver call " + std::to_string(out.call) + " of " + std::to_string(out.calls) : std::string()) + ")";
    app_log(1, "  [ladder cond] {}: sigma_min(1 - Cb K_s) over {} {} units: min {:.3e} at {}, median {:.3e}, max {:.3e}; "
               "block-Lanczos steps max {} (blocks of {}){}; meter {:.2f} s on the slowest rank ({:.2f} % of the pass wall {:.1f} s)",
            out.label, tot, out.all_nu ? "metered" : "nu = 0", out.smin, loc, out.smed, out.smax, out.steps_max, block(),
            out.n_unconv > 0 ? ", " + std::to_string(out.n_unconv) + " units NOT converged to the residual tolerance (their sigma_min "
                                                                   "is an upper estimate)" : std::string(", all converged"),
            out.t_meter, (out.t_pass > 0.0) ? 100.0 * out.t_meter / out.t_pass : 0.0, out.t_pass);
    {   // the metered units, smallest sigma_min first (log level 2; the lowest 96)
      std::vector<long> ord(static_cast<size_t>(tot));
      for (long u = 0; u < tot; ++u) ord[size_t(u)] = u;
      std::sort(ord.begin(), ord.end(), [&](long a, long b) { return sm[size_t(a)] < sm[size_t(b)]; });
      if (tot > 96) ord.resize(96);
      app_log(2, "  [ladder cond units] {} ({} of {} units):   s     q   nu-node  call   sigma_min   Lanczos-steps  residual", out.label,
              long(ord.size()), tot);
      for (long u : ord) {
        double const *row = buf.data() + u * NF;
        app_log(2, "  [ladder cond units] {}: {:3d} {:5d} {:9d} {:5d}   {:.4e}   {:13d}  {:.1e}", out.label, long(row[0]), long(row[1]),
                long(row[2]), long(row[3]), row[5], long(row[8]), row[7]);
      }
    }
    if (out.smin < warn)
      app_log(1, "  [ladder cond] WARNING ({}): sigma_min(1 - Cb K_s) = {:.3e} at {} is below ladder_cond_warn = {:.1e}. The static "
                 "ladder of this unit is close to the excitonic instability: its resolvent T_s = K_s (1 - Cb K_s)^-1 amplifies every "
                 "error of its inputs (tau refits, pole fits of G, the THC rung) by up to ~1/sigma_min = {:.1e}, so the vertex "
                 "corrections of this pass are NOISE-AMPLIFIED and must not be trusted at that level.",
              out.label, out.smin, loc, warn, 1.0 / out.smin);
    utils::check(not (abort > 0.0 and out.smin < abort),
                 "[ladder cond] {}: sigma_min(1 - Cb K_s) = {:.3e} at {} is below vertex_debug ladder_cond_abort = {:.1e}. ABORTING "
                 "(the static ladder is at the excitonic instability; see the ladder_cond_warn message above).",
                 out.label, out.smin, loc, abort);
    pending().push_back(out);
    return out;
  }

  /** the summaries reported since the last call (the caller writes them to the iteration's h5 group) */
  inline std::vector<summary> take_pending() {
    std::vector<summary> p;
    p.swap(pending());
    return p;
  }

} // namespace ladder_cond
} // namespace solvers
} // namespace methods

#endif // COQUI_VERTEX_LADDER_COND_HPP
