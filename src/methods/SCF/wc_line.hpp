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

#ifndef COQUI_WC_LINE_HPP
#define COQUI_WC_LINE_HPP

/**
 * ===========================================================================
 * W^c ON THE LINE  (increment TC-3 / spec M4, notes/tc_coqui_impl_spec.md)
 * ===========================================================================
 *
 * Given the polarization Pi(q, z) at an arbitrary complex z -- which
 * methods/SCF/p_contour.hpp supplies on the tilted contour's target list --
 * this file produces the screened interaction there, and the contracted band
 * elements <nm|W^c(q,z)|mn> that the eq-1 CD assembly consumes.
 *
 * ---------------------------------------------------------------------------
 * 1. ONE CONVENTION, AND IT IS THE CODE'S OWN
 * ---------------------------------------------------------------------------
 * The imaginary-axis solver assembles W^c as (scr_coulomb_t::dyson_W_in_place,
 * scr_coulomb_t.cpp, the "W(w) = [ I - Z * Pi(w)]^{-1} * Z - Z" block):
 *
 *      A = Z . Pi                                                        (i)
 *      A = I - A                                                        (ii)
 *      A = A^{-1}                                                      (iii)
 *      A = A - I                                                        (iv)
 *      W^c = A . Z                                                       (v)
 *
 * i.e. W^c = ([I - Z.Pi]^{-1} - I).Z, with Z = thc.Z(q) the bare Coulomb
 * kernel in the THC auxiliary basis. That is ALGEBRAICALLY the spec's
 * v[1 - v P]^{-1} v P v:
 *
 *      ([1-ZP]^{-1} - 1) Z = [1-ZP]^{-1} (1 - (1-ZP)) Z = [1-ZP]^{-1} Z P Z
 *                          = Z [1-PZ]^{-1} P Z,
 *
 * and `dyson_wc_line` below reproduces the FIVE STEPS IN THAT ORDER rather
 * than an equivalent rearrangement, so that a line evaluation and an
 * imaginary-axis evaluation of the same Pi cannot drift apart.
 *
 * ⚠ THE SIGN OF Pi IS NOT FREE. The object this file expects is the CODE's
 * Pi -- the one `scr_coulomb_t::eval_Pi_qdep` stores, negative semi-definite
 * on the imaginary axis. From the tilted contour that is
 *
 *      Pi(z) = -[ R(z) + R(-conj z)^dag ]                          (TC-2 eq SIGN)
 *
 * which is exactly what `p_contour::polarization_from_contour` returns. Feeding
 * the raw resonant half R, or the campaign's P(z) with the opposite overall
 * sign, silently produces a W^c with the wrong screening sign; gate TC-3-a(W)
 * scores this whole chain against an independently generated reference.
 *
 * ---------------------------------------------------------------------------
 * 2. THE CONTRACTED ELEMENTS, AND WHY THE KRYLOV PATH IS THE PRODUCTION ONE
 * ---------------------------------------------------------------------------
 * The CD assembly never needs W^c as a matrix -- it needs the scalars
 * <nm|W^c(q,z)|mn> = gl^dag W^c gr for a handful of band-pair vectors g. With
 * y := Z gr (one matvec) and x the solution of
 *
 *      [I - Z.Pi] x = y,                                                (K)
 *
 * step (iv)-(v) give gl^dag W^c gr = gl^dag (x - y): ONE linear solve per
 * (q, z, vector), never an inverse and never an (Np x Np) intermediate. At test
 * scale `wc_sandwich_dense` forms the inverse once per (q,z) and applies it to
 * every vector, which is cheaper when the vector count approaches Np and is the
 * reference the Krylov path is gated against (the campaign measured the two
 * agreeing to 9e-13). Neighbouring targets z on the line differ little, so the
 * previous z's solution is the natural initial guess -- that is the warm start,
 * and it is what makes (K) cost a few iterations instead of a factorization.
 *
 * `gmres_solve` below is a compact restarted GMRES with modified Gram-Schmidt
 * and Givens rotations; the repository had no Krylov solver to reuse.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace wc_line {

  using dcomplex = std::complex<double>;

  /** Knobs of the line solve. */
  struct solve_opts_t {
    bool   krylov = false;     ///< qp_tc_krylov: warm-started GMRES instead of dense
    double tol = 1e-12;        ///< qp_tc_krylov_tol: relative residual target
    long   restart = 60;       ///< Krylov subspace size before restart
    long   maxit = 600;        ///< hard iteration cap
  };

  /** What one solve reported, for the per-iteration log line. */
  struct solve_stats_t {
    long   n_solve = 0;        ///< linear systems solved
    long   n_iter = 0;         ///< total Krylov iterations (0 on the dense path)
    long   n_iter_max = 0;
    double resid_max = 0.0;    ///< worst achieved relative residual
    double cond_hint = 0.0;    ///< max |[I - Z.Pi]^{-1}| seen (dense path only)
    long   n_warm = 0;         ///< solves that started from a previous solution
  };

  // =========================================================================
  //  the dielectric matrix
  // =========================================================================

  /**
   * E = I - Z.Pi, in the code's operation order (steps (i)-(ii) of section 1).
   * Z and Pi are (n, n); E is overwritten.
   */
  inline void build_epsilon(nda::MemoryArrayOfRank<2> auto const &Z,
                            nda::MemoryArrayOfRank<2> auto const &Pi,
                            nda::matrix<dcomplex> &E) {
    const long n = Z.shape()[0];
    utils::check(Z.shape()[1] == n and Pi.shape()[0] == n and Pi.shape()[1] == n,
                 "wc_line::build_epsilon: Z is {}x{}, Pi is {}x{}; both must be square "
                 "and equal.", Z.shape()[0], Z.shape()[1], Pi.shape()[0], Pi.shape()[1]);
    nda::matrix<dcomplex> Zm(n, n), Pm(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) { Zm(i, j) = Z(i, j); Pm(i, j) = Pi(i, j); }
    E.resize(n, n);
    nda::blas::gemm(dcomplex(1.0), Zm, Pm, dcomplex(0.0), E);   // (i)  A = Z.Pi
    E *= -1.0;                                                   // (ii) A = I - Z.Pi
    for (long i = 0; i < n; ++i) E(i, i) += dcomplex(1.0);
  }

  /**
   * The full W^c at one (q, z): W^c = ([I - Z.Pi]^{-1} - I).Z, steps (i)-(v).
   * Test-scale / reference path.
   */
  inline void dyson_wc_line(nda::MemoryArrayOfRank<2> auto const &Z,
                            nda::MemoryArrayOfRank<2> auto const &Pi,
                            nda::matrix<dcomplex> &W,
                            solve_stats_t *st = nullptr) {
    const long n = Z.shape()[0];
    nda::matrix<dcomplex> E(n, n);
    build_epsilon(Z, Pi, E);
    E = nda::inverse(E);                                         // (iii)
    if (st != nullptr) {
      double m = 0.0;
      for (auto const &v : E) m = std::max(m, std::abs(v));
      st->cond_hint = std::max(st->cond_hint, m);
    }
    for (long i = 0; i < n; ++i) E(i, i) -= dcomplex(1.0);       // (iv)
    nda::matrix<dcomplex> Zm(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) Zm(i, j) = Z(i, j);
    W.resize(n, n);
    nda::blas::gemm(dcomplex(1.0), E, Zm, dcomplex(0.0), W);     // (v)
  }

  // =========================================================================
  //  restarted GMRES  (the repository had none to reuse)
  // =========================================================================

  /**
   * Solve A x = b to `tol` relative residual, restarting every `restart`
   * iterations. `x` is BOTH the initial guess (warm start) and the output.
   * Returns the achieved relative residual; `iters` accumulates the count.
   *
   * Modified Gram-Schmidt Arnoldi + Givens rotations on the Hessenberg least
   * squares -- the textbook construction, written out because it must be
   * auditable next to the dense reference it is gated against.
   */
  inline double gmres_solve(nda::matrix<dcomplex> const &A,
                            nda::array<dcomplex, 1> const &b,
                            nda::array<dcomplex, 1> &x,
                            double tol, long restart, long maxit, long &iters) {
    const long n = b.size();
    const long m = std::min(restart, n);
    auto nrm = [](nda::array<dcomplex, 1> const &v) {
      double s = 0.0;
      for (long i = 0; i < v.size(); ++i) s += std::norm(v(i));
      return std::sqrt(s);
    };
    const double bnorm = std::max(nrm(b), 1e-300);

    nda::array<dcomplex, 1> r(n), w(n);
    nda::array<dcomplex, 2> V(m + 1, n);
    nda::array<dcomplex, 2> H(m + 1, m);
    const std::size_t msz = static_cast<std::size_t>(m);
    std::vector<dcomplex> cs(msz), sn(msz), g(msz + 1);

    double rel = 1.0;
    long done = 0;
    while (done < maxit) {
      // r = b - A x
      for (long i = 0; i < n; ++i) {
        dcomplex acc(0.0, 0.0);
        for (long j = 0; j < n; ++j) acc += A(i, j) * x(j);
        r(i) = b(i) - acc;
      }
      double beta = nrm(r);
      rel = beta / bnorm;
      if (rel <= tol) break;
      for (long i = 0; i < n; ++i) V(0, i) = r(i) / beta;
      std::fill(g.begin(), g.end(), dcomplex(0.0, 0.0));
      g[0] = dcomplex(beta, 0.0);

      long k = 0;
      for (; k < m and done < maxit; ++k, ++done) {
        for (long i = 0; i < n; ++i) {                    // w = A v_k
          dcomplex acc(0.0, 0.0);
          for (long j = 0; j < n; ++j) acc += A(i, j) * V(k, j);
          w(i) = acc;
        }
        for (long i = 0; i <= k; ++i) {                   // modified Gram-Schmidt
          dcomplex h(0.0, 0.0);
          for (long j = 0; j < n; ++j) h += std::conj(V(i, j)) * w(j);
          H(i, k) = h;
          for (long j = 0; j < n; ++j) w(j) -= h * V(i, j);
        }
        const double hk = nrm(w);
        H(k + 1, k) = dcomplex(hk, 0.0);
        if (hk > 1e-300)
          for (long j = 0; j < n; ++j) V(k + 1, j) = w(j) / hk;

        for (long i = 0; i < k; ++i) {                    // apply past rotations
          const dcomplex t = cs[std::size_t(i)] * H(i, k)
                           + sn[std::size_t(i)] * H(i + 1, k);
          H(i + 1, k) = -std::conj(sn[std::size_t(i)]) * H(i, k)
                      + std::conj(cs[std::size_t(i)]) * H(i + 1, k);
          H(i, k) = t;
        }
        const double d = std::sqrt(std::norm(H(k, k)) + std::norm(H(k + 1, k)));
        if (d < 1e-300) { ++k; break; }
        cs[std::size_t(k)] = std::conj(H(k, k)) / d;
        sn[std::size_t(k)] = std::conj(H(k + 1, k)) / d;
        H(k, k) = cs[std::size_t(k)] * H(k, k) + sn[std::size_t(k)] * H(k + 1, k);
        H(k + 1, k) = dcomplex(0.0, 0.0);
        g[std::size_t(k + 1)] = -std::conj(sn[std::size_t(k)]) * g[std::size_t(k)];
        g[std::size_t(k)] = cs[std::size_t(k)] * g[std::size_t(k)];
        rel = std::abs(g[std::size_t(k + 1)]) / bnorm;
        if (rel <= tol) { ++k; break; }
      }
      if (k == 0) break;
      for (long i = k - 1; i >= 0; --i) {                 // back substitution
        dcomplex acc = g[std::size_t(i)];
        for (long j = i + 1; j < k; ++j) acc -= H(i, j) * g[std::size_t(j)];
        g[std::size_t(i)] = acc / H(i, i);
      }
      for (long j = 0; j < n; ++j) {                      // x += V_k y
        dcomplex acc(0.0, 0.0);
        for (long i = 0; i < k; ++i) acc += g[std::size_t(i)] * V(i, j);
        x(j) += acc;
      }
    }
    iters += done;
    return rel;
  }

  // =========================================================================
  //  the contracted band elements
  // =========================================================================

  /**
   * out(v) = gl(:,v)^dag W^c gr(:,v) at one (q, z), for `nvec` band-pair vectors.
   *
   * @param Z, Pi   (n, n) the Coulomb kernel and the polarization in the CODE
   *                convention (section 1).
   * @param gl, gr  (n, nvec) the left and right band-pair vectors.
   * @param out     (nvec) OVERWRITTEN.
   * @param xw      (n, nvec) warm-start / carry-out workspace for the Krylov
   *                path: on entry the previous target's solutions (or zeros),
   *                on exit this target's. Ignored on the dense path.
   */
  inline void wc_sandwich(nda::MemoryArrayOfRank<2> auto const &Z,
                          nda::MemoryArrayOfRank<2> auto const &Pi,
                          nda::MemoryArrayOfRank<2> auto const &gl,
                          nda::MemoryArrayOfRank<2> auto const &gr,
                          nda::array<dcomplex, 1> &out,
                          solve_opts_t const &opt,
                          solve_stats_t &st,
                          nda::array<dcomplex, 2> *xw = nullptr) {
    const long n = Z.shape()[0];
    const long nv = gl.shape()[1];
    utils::check(gr.shape()[1] == nv and gl.shape()[0] == n and gr.shape()[0] == n,
                 "wc_line::wc_sandwich: gl is {}x{}, gr is {}x{}, n = {}.",
                 gl.shape()[0], gl.shape()[1], gr.shape()[0], gr.shape()[1], n);
    out.resize(nv);

    if (not opt.krylov) {
      nda::matrix<dcomplex> W(n, n);
      dyson_wc_line(Z, Pi, W, &st);
      st.n_solve += 1;
      for (long v = 0; v < nv; ++v) {
        dcomplex acc(0.0, 0.0);
        for (long i = 0; i < n; ++i) {
          dcomplex row(0.0, 0.0);
          for (long j = 0; j < n; ++j) row += W(i, j) * gr(j, v);
          acc += std::conj(gl(i, v)) * row;
        }
        out(v) = acc;
      }
      return;
    }

    // ---- Krylov: one solve per vector, warm-started from the previous target
    nda::matrix<dcomplex> E(n, n);
    build_epsilon(Z, Pi, E);
    nda::array<dcomplex, 1> y(n), x(n);
    for (long v = 0; v < nv; ++v) {
      for (long i = 0; i < n; ++i) {                      // y = Z gr
        dcomplex acc(0.0, 0.0);
        for (long j = 0; j < n; ++j) acc += Z(i, j) * gr(j, v);
        y(i) = acc;
      }
      if (xw != nullptr) { for (long i = 0; i < n; ++i) x(i) = (*xw)(i, v); ++st.n_warm; }
      else               { x() = dcomplex(0.0, 0.0); }
      long it = 0;
      const double rel = gmres_solve(E, y, x, opt.tol, opt.restart, opt.maxit, it);
      st.n_solve += 1;
      st.n_iter += it;
      st.n_iter_max = std::max(st.n_iter_max, it);
      st.resid_max = std::max(st.resid_max, rel);
      if (xw != nullptr) for (long i = 0; i < n; ++i) (*xw)(i, v) = x(i);
      dcomplex acc(0.0, 0.0);                             // gl^dag (x - y)
      for (long i = 0; i < n; ++i) acc += std::conj(gl(i, v)) * (x(i) - y(i));
      out(v) = acc;
    }
  }

} // namespace wc_line
} // namespace methods

#endif // COQUI_WC_LINE_HPP
