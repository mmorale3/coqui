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

#ifndef COQUI_SIGMA_REAL_AXIS_HPP
#define COQUI_SIGMA_REAL_AXIS_HPP

/**
 * Project 2 (qpGW+BSE+EDMFT) increment QM1 -- ROUTE A: the real-axis self-energy
 * evaluator (spec notes/qm1_route_a_spec.md; parent notes/qsgw_matsubara_plan.pdf
 * section 2, gates section 6.1-6.3). Sibling of qp_maps_matsubara.hpp: that file
 * carries the AC-free Matsubara-native maps, this one carries the van Schilfgaarde
 * mode-A real-axis evaluation WITHOUT analytic continuation -- Sigma(eps) is obtained
 * from a low-order Taylor expansion about a REAL centre z0, whose coefficients are
 * extracted from samples taken along the IMAGINARY direction, where the data lives.
 *
 *     Sigma(z0 + i t) = sum_{n=0..p} c_n (i t)^n,      Sigma(eps) = sum_n c_n (eps - z0)^n.
 *
 * Sigma is analytic in a disk of radius R_conv about a real z0 in the gap region, so
 * the expansion is legitimate; R_conv is estimated per fit and reported.
 *
 * THE REFLECTION CONSTRAINT (the correctness core, spec section 1)
 * ---------------------------------------------------------------
 * Schwarz reflection for the self-energy matrix reads Sigma_ij(z*) = [Sigma_ji(z)]*,
 * the same identity Q1's even/odd extrapolation uses on the Matsubara axis in the form
 * Sigma(-i w) = Sigma(i w)^dag (qp_maps_matsubara.hpp header). Taylor coefficients
 * about a REAL centre therefore obey
 *
 *     c_n^(ij) = (c_n^(ji))*      =>      diagonal c_n^(ii) is REAL.
 *
 * The pinned implementation rule: for each UNORDERED pair {i,j} the sample function on
 * both signs of t is built from the +t half-window ONLY,
 *
 *     F_ij(+t) = Sigma_ij(z0 + i t),
 *     F_ij(-t) = [Sigma_ji(z0 + i t)]*        (never sample Sigma_ij(z0 - i t) on its own),
 *
 * ONE complex least-squares polynomial is fitted per unordered pair, and the partner is
 * set by c^(ji) = (c^(ij))*. Then Sigmahat_ji(eps) = [Sigmahat_ij(eps)]* holds exactly
 * for real eps, so the mode-A assembly
 *
 *     V^xc_ij = 1/2 [ Sigmahat_ij(eps_i) + Sigmahat_ij(eps_j) ]
 *
 * is HERMITIAN BY CONSTRUCTION at machine precision -- the prescription's "Re" is
 * implemented by the pairing constraint, not by a post-hoc Hermitization. There is
 * deliberately no herm() call in assemble_vxc. On the diagonal the +/- sample symmetry
 * F(-t) = F(t)* already forces a real least-squares solution; Re(c_n) is taken and
 * imag_c_rel = max_n |Im c_n| / max_n |c_n| is retained as the logged fit-quality
 * diagnostic (spec section 2.1).
 *
 * FIT (spec section 2)
 * --------------------
 * Complex LS of the polynomial in (i t), order p (default 2), over 2m samples (both
 * signs). The variable is SCALED before the Vandermonde is formed, u = t/t_max, the fit
 * is done in (i u)^n and unscaled by c_n <- c_n / t_max^n -- pure conditioning, and it
 * matters because t_max ~ 1e-2 in model units while c_2 ~ 1e-2, so the raw design matrix
 * spans many orders of magnitude. The +/- symmetric sample set makes the normal-equations
 * matrix real and parity-block-diagonal, which is why a small explicit solve is enough.
 *
 * Sampling window: t = +/- w_n over m fermionic nodes starting at node n0 (default 0),
 * m default 3p, i.e. the window [w_0, w_{3p-1}] and 6p samples. |t| >= w_0 ALWAYS --
 * mandatory when the sampler is a DLR pole representation, whose fit poles lie ON the
 * real axis (plan section 2 caveat), and kept uniformly for every sampler so that one
 * window convention covers all of them.
 *
 * Diagnostics returned with every fit: relative residual ||F - fit|| / ||F||, imag_c_rel
 * (meaningful for diagonal pairs), the convergence-radius estimate R_conv = |c_{p-1}/c_p|,
 * and |eps - z0| / R_conv at evaluation.
 *
 * SAMPLERS (spec section 3)
 * -------------------------
 * The fit/eval core is sampler-agnostic: it takes arrays (t_k, F_k). Three sources:
 *   - Matsubara data (z0 = 0 first pass): the samples ARE the stored MO-basis
 *     Sigma(i w_n); the negative half comes from the dagger identity, i.e. exactly the
 *     reflection rule above. See fit_matsubara.
 *   - Pole representation: closed-form F(z) = sum_p c_p / (z - eps_p) over an
 *     imag_axes_ft::dlr_pole_fit output (residues coeffs(), poles epsl). This is the
 *     OFF-GRID source needed for re-expansion about z0 != 0. See pole_sampler.
 *   - Analytic models (tests).
 *
 * QP ROOT WITH RE-EXPANSION (spec sections 2.2-2.3, the production algorithm)
 * --------------------------------------------------------------------------
 * Given a sampler S and a static part e0, solve eps = e0 + Sigmahat(eps):
 *   1. z0 <- 0 (or a caller-provided guess);
 *   2. fit the expansion about z0 -- for a general sampler the two signs are sampled
 *      EXPLICITLY, S(z0 + i t) and S(z0 - i t), because they are conjugates only when z0
 *      is real AND S is Hermitian-Lehmann; both are closed form, so nothing is gained by
 *      assuming it;
 *   3. Newton from eps = z0 on eps = e0 + Re sum_n c_n (eps - z0)^n;
 *   4. z0 <- eps and repeat, up to n_reexp times (default 4), early-stopping at
 *      |delta eps| < 1e-9 (model units).
 * At the fixed point z0 = eps the quadratic model reproduces Sigma(eps) through its own
 * c_0, so the re-expanded order-p = 2 map is far more accurate than a high-order
 * expansion about zero (spec section 2.3) -- that is the reason re-expansion exists.
 *
 * Defaults (plan section 5 item 5): p = 2, n_reexp = 4, m = 3p. They are struct options
 * here; toml exposure and the loop dispatch are increment QM2, NOT this file.
 */

#include <cmath>
#include <complex>
#include <limits>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace sigma_real_axis {

  /** Fit + re-expansion options. Defaults are the production settings. */
  struct fit_opts {
    long p = 2;               // Taylor order about z0
    long m = -1;              // fermionic nodes in the half-window; < 0 selects 3p
    long n0 = 0;              // index of the FIRST fermionic node used; 0 => |t| >= w_0
    long n_reexp = 4;         // number of re-expansions of the centre z0
    double reexp_tol = 1e-9;  // early stop on |delta eps| (model units)
    long newton_max = 64;
    double newton_tol = 1e-14;
  };

  /** Per-fit diagnostics (spec section 2, "returned with every fit"). */
  struct fit_diag {
    double rel_resid = 0.0;   // ||F - fit||_2 / ||F||_2 over the sample set
    double imag_c_rel = 0.0;  // max_n |Im c_n| / max_n |c_n|  (diagonal fit quality)
    double R_conv = 0.0;      // |c_{p-1} / c_p|, the convergence-radius estimate
    double dist_over_R = 0.0; // |eps - z0| / R_conv at the evaluation point
  };

  struct qp_root_result {
    double eps = 0.0;         // the quasiparticle energy
    fit_diag diag{};          // diagnostics of the LAST fit
    long n_reexp_used = 0;    // re-expansions actually performed
    long newton_iters = 0;    // Newton steps of the last fit
    bool converged = false;   // the |delta eps| < reexp_tol early stop fired
  };

  /** the default half-window width in fermionic nodes. */
  inline long m_default(long p) { return 3 * p; }

  /**
   * The sampling half-window: fermionic Matsubara values w_n = (2n+1) pi / beta for
   * n = n0 .. n0+m-1, ascending. n0 = 0 pins the inner edge at w_0 -- the mandatory
   * floor when the sampler carries real-axis poles.
   */
  inline nda::array<double, 1> window_nodes(double beta, long m, long n0 = 0) {
    utils::check(beta > 0.0 and m > 0 and n0 >= 0,
                 "sigma_real_axis::window_nodes: need beta > 0 (got {}), m > 0 (got {}), "
                 "n0 >= 0 (got {}).", beta, m, n0);
    nda::array<double, 1> t(m);
    for (long k = 0; k < m; ++k) t(k) = (2.0 * double(n0 + k) + 1.0) * M_PI / beta;
    return t;
  }

  /**
   * Complex least-squares Taylor fit of order p in (i t), from the two half-windows
   * Fp(k) = F(+t_k) and Fm(k) = F(-t_k). Returns c(0..p) in PHYSICAL units (the u = t/t_max
   * scaling is undone internally) and fills the residual / imag_c_rel / R_conv fields of d.
   */
  inline nda::array<ComplexType, 1> fit_taylor(nda::MemoryArrayOfRank<1> auto const &tp,
                                               nda::MemoryArrayOfRank<1> auto const &Fp,
                                               nda::MemoryArrayOfRank<1> auto const &Fm,
                                               long p, fit_diag &d) {
    const long m = tp.shape(0), nc = p + 1, ns = 2 * m;
    utils::check(p >= 0, "sigma_real_axis::fit_taylor: order p = {} must be >= 0.", p);
    utils::check(m > 0 and Fp.shape(0) == m and Fm.shape(0) == m,
                 "sigma_real_axis::fit_taylor: sample arrays must all have length m "
                 "(got m = {}, |Fp| = {}, |Fm| = {}).", m, Fp.shape(0), Fm.shape(0));
    utils::check(ns >= nc,
                 "sigma_real_axis::fit_taylor: {} samples cannot determine {} coefficients.",
                 ns, nc);

    double tmax = 0.0;
    for (long k = 0; k < m; ++k) tmax = std::max(tmax, std::abs(tp(k)));
    utils::check(tmax > 0.0, "sigma_real_axis::fit_taylor: degenerate window (t_max = 0).");

    // design matrix in the SCALED variable u = t / t_max
    nda::array<ComplexType, 2> A(ns, nc);
    nda::array<ComplexType, 1> b(ns);
    for (long k = 0; k < m; ++k) {
      const double u = tp(k) / tmax;
      ComplexType wp(1.0, 0.0), wm(1.0, 0.0);
      const ComplexType iup(0.0, u), ium(0.0, -u);
      for (long n = 0; n < nc; ++n) {
        A(k, n) = wp;      wp *= iup;
        A(m + k, n) = wm;  wm *= ium;
      }
      b(k) = Fp(k);
      b(m + k) = Fm(k);
    }

    // normal equations. The +/- symmetric sample set makes A^dag A real and
    // parity-block-diagonal, so this small solve is well conditioned by construction.
    nda::matrix<ComplexType> M(nc, nc);
    nda::array<ComplexType, 1> r(nc), c(nc);
    M() = ComplexType(0.0);
    r() = ComplexType(0.0);
    for (long n = 0; n < nc; ++n) {
      for (long k = 0; k < ns; ++k) r(n) += std::conj(A(k, n)) * b(k);
      for (long n2 = 0; n2 < nc; ++n2) {
        ComplexType s(0.0);
        for (long k = 0; k < ns; ++k) s += std::conj(A(k, n)) * A(k, n2);
        M(n, n2) = s;
      }
    }
    nda::inverse_in_place(M);
    nda::blas::gemv(ComplexType(1.0), M, r, ComplexType(0.0), c);

    // relative residual on the SCALED problem (identical to the physical one: the
    // scaling acts on the coefficients, not on the samples)
    double rn2 = 0.0, bn2 = 0.0;
    for (long k = 0; k < ns; ++k) {
      ComplexType fit(0.0);
      for (long n = 0; n < nc; ++n) fit += A(k, n) * c(n);
      rn2 += std::norm(b(k) - fit);
      bn2 += std::norm(b(k));
    }
    d.rel_resid = (bn2 > 0.0) ? std::sqrt(rn2 / bn2) : std::sqrt(rn2);

    // unscale: c_n <- c_n / t_max^n
    double s = 1.0;
    for (long n = 0; n < nc; ++n) {
      c(n) /= s;
      s *= tmax;
    }

    double amax = 0.0, imax = 0.0;
    for (long n = 0; n < nc; ++n) {
      amax = std::max(amax, std::abs(c(n)));
      imax = std::max(imax, std::abs(c(n).imag()));
    }
    d.imag_c_rel = (amax > 0.0) ? imax / amax : 0.0;
    d.R_conv = (p >= 1)
                   ? ((std::abs(c(p)) > 0.0) ? std::abs(c(p - 1)) / std::abs(c(p))
                                             : std::numeric_limits<double>::infinity())
                   : std::numeric_limits<double>::infinity();
    d.dist_over_R = 0.0;
    return c;
  }

  /** Sigmahat(eps) = sum_n c_n (eps - z0)^n for real eps. */
  inline ComplexType eval_taylor(nda::MemoryArrayOfRank<1> auto const &c, double dz) {
    ComplexType s(0.0);
    double w = 1.0;
    for (long n = 0; n < c.shape(0); ++n) {
      s += c(n) * w;
      w *= dz;
    }
    return s;
  }

  /**
   * Pole-representation sampler: F(z) = sum_p c_p / (z - eps_p), the closed-form off-grid
   * source built from an imag_axes_ft::dlr_pole_fit output (residues from coeffs(), poles
   * from epsl). Its poles sit ON the real axis, which is precisely why the sampling window
   * is floored at |t| >= w_0.
   */
  struct pole_sampler {
    nda::array<ComplexType, 1> c;   // residues
    nda::array<double, 1> epsl;     // pole energies

    ComplexType operator()(ComplexType z) const {
      ComplexType s(0.0);
      for (long p = 0; p < c.shape(0); ++p) s += c(p) / (z - epsl(p));
      return s;
    }
  };

  /**
   * Diagonal quasiparticle root with re-expansion (spec sections 2.2-2.3).
   * S must be callable as S(ComplexType) -> ComplexType. e0 is the static part.
   */
  template <class Sampler>
  inline qp_root_result qp_root(Sampler const &S, double e0, double beta,
                                double z0_init = 0.0, fit_opts const &opt = {}) {
    const long p = opt.p;
    const long m = (opt.m > 0) ? opt.m : m_default(p);
    auto tp = window_nodes(beta, m, opt.n0);

    qp_root_result out;
    nda::array<ComplexType, 1> Fp(m), Fm(m);
    double z0 = z0_init, e = z0_init;

    for (long it = 0; it <= opt.n_reexp; ++it) {
      // BOTH signs sampled explicitly: they are conjugates only for a Hermitian-Lehmann
      // sampler at real z0, and both are closed form.
      for (long k = 0; k < m; ++k) {
        Fp(k) = S(ComplexType(z0, tp(k)));
        Fm(k) = S(ComplexType(z0, -tp(k)));
      }
      fit_diag d;
      auto c = fit_taylor(tp, Fp, Fm, p, d);

      // Newton from eps = z0 on g(eps) = eps - e0 - Re sum_n c_n (eps - z0)^n
      e = z0;
      long nit = 0;
      for (; nit < opt.newton_max; ++nit) {
        const double x = e - z0;
        double sig = 0.0, dsig = 0.0, w = 1.0;
        for (long n = 0; n <= p; ++n) {
          sig += c(n).real() * w;
          w *= x;
        }
        w = 1.0;
        for (long n = 1; n <= p; ++n) {
          dsig += double(n) * c(n).real() * w;
          w *= x;
        }
        const double g = e - e0 - sig, gp = 1.0 - dsig;
        if (std::abs(gp) < 1e-14) break;    // degenerate slope: keep the current iterate
        const double de = -g / gp;
        e += de;
        if (std::abs(de) < opt.newton_tol) { ++nit; break; }
      }

      out.newton_iters = nit;
      out.diag = d;
      out.diag.dist_over_R = (std::isfinite(d.R_conv) and d.R_conv > 0.0)
                                 ? std::abs(e - z0) / d.R_conv : 0.0;
      out.n_reexp_used = it;

      const double dz = e - z0;
      z0 = e;                              // re-expand about the new estimate
      if (std::abs(dz) < opt.reexp_tol) { out.converged = true; break; }
    }
    out.eps = e;
    return out;
  }

  /**
   * One Taylor expansion per UNORDERED pair of a matrix-valued Sigma, with the reflection
   * constraint imposed exactly: c(n,j,i) = conj(c(n,i,j)).
   */
  struct matrix_expansion {
    long nb = 0, p = 0;
    double z0 = 0.0;
    nda::array<ComplexType, 3> c;                          // (p+1, nb, nb)
    nda::array<double, 2> rel_resid, imag_c_rel, R_conv;   // (nb, nb), symmetric in (i,j)

    ComplexType eval(long i, long j, double e) const {
      ComplexType s(0.0);
      double w = 1.0;
      for (long n = 0; n <= p; ++n) {
        s += c(n, i, j) * w;
        w *= (e - z0);
      }
      return s;
    }
  };

  /**
   * Fit the matrix expansion from the POSITIVE half-window only.
   * Sigma_tab(k,i,j) = Sigma_ij(z0 + i tp(k)); the negative half is generated internally by
   * the reflection rule F_ij(-t) = [Sigma_ji(z0 + i t)]*, which is what makes the assembled
   * V^xc Hermitian at machine precision.
   */
  inline matrix_expansion fit_matrix(nda::MemoryArrayOfRank<3> auto const &Sigma_tab,
                                     nda::MemoryArrayOfRank<1> auto const &tp,
                                     double z0, fit_opts const &opt = {}) {
    const long m = Sigma_tab.shape(0), nb = Sigma_tab.shape(1), p = opt.p;
    utils::check(Sigma_tab.shape(2) == nb,
                 "sigma_real_axis::fit_matrix: Sigma block is {} x {}, must be square.",
                 nb, Sigma_tab.shape(2));
    utils::check(tp.shape(0) == m,
                 "sigma_real_axis::fit_matrix: {} nodes for {} sample slices.",
                 tp.shape(0), m);

    matrix_expansion X;
    X.nb = nb;
    X.p = p;
    X.z0 = z0;
    X.c = nda::array<ComplexType, 3>(p + 1, nb, nb);
    X.rel_resid = nda::array<double, 2>(nb, nb);
    X.imag_c_rel = nda::array<double, 2>(nb, nb);
    X.R_conv = nda::array<double, 2>(nb, nb);

    nda::array<ComplexType, 1> Fp(m), Fm(m);
    for (long i = 0; i < nb; ++i)
      for (long j = i; j < nb; ++j) {
        for (long k = 0; k < m; ++k) {
          Fp(k) = Sigma_tab(k, i, j);
          Fm(k) = std::conj(Sigma_tab(k, j, i));
        }
        fit_diag d;
        auto c = fit_taylor(tp, Fp, Fm, p, d);
        // diagonal: the +/- symmetry already forces a real solution; Re() makes it exact
        // and imag_c_rel is the logged fit-quality diagnostic.
        if (i == j)
          for (long n = 0; n <= p; ++n) c(n) = ComplexType(c(n).real(), 0.0);
        for (long n = 0; n <= p; ++n) {
          X.c(n, i, j) = c(n);
          X.c(n, j, i) = std::conj(c(n));
        }
        X.rel_resid(i, j) = X.rel_resid(j, i) = d.rel_resid;
        X.imag_c_rel(i, j) = X.imag_c_rel(j, i) = d.imag_c_rel;
        X.R_conv(i, j) = X.R_conv(j, i) = d.R_conv;
      }
    return X;
  }

  /**
   * Matsubara-data path (the z0 = 0 first pass): the samples ARE the stored MO-basis
   * Sigma(i w_n) on the fermionic nodes, and the negative-t half follows from the dagger
   * identity Sigma(-i w) = Sigma(i w)^dag (the Q1 identity, qp_maps_matsubara.hpp header)
   * -- i.e. exactly the reflection rule. Uses nodes opt.n0 .. opt.n0+m-1 of the supplied
   * mesh, m defaulting to 3p.
   */
  inline matrix_expansion fit_matsubara(nda::MemoryArrayOfRank<3> auto const &Sigma_wab,
                                        nda::MemoryArrayOfRank<1> auto const &wn,
                                        fit_opts const &opt = {}) {
    const long nw = Sigma_wab.shape(0), nb = Sigma_wab.shape(1);
    const long m = (opt.m > 0) ? opt.m : m_default(opt.p);
    utils::check(wn.shape(0) >= opt.n0 + m and nw >= opt.n0 + m,
                 "sigma_real_axis::fit_matsubara: need {} fermionic nodes from index {} "
                 "(have {} values / {} slices).", m, opt.n0, wn.shape(0), nw);
    utils::check(wn(opt.n0) > 0.0,
                 "sigma_real_axis::fit_matsubara: the mesh must hold POSITIVE ascending "
                 "fermionic nodes (wn({}) = {}).", opt.n0, wn(opt.n0));
    nda::array<double, 1> tp(m);
    nda::array<ComplexType, 3> S(m, nb, nb);
    for (long k = 0; k < m; ++k) {
      tp(k) = wn(opt.n0 + k);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) S(k, i, j) = Sigma_wab(opt.n0 + k, i, j);
    }
    return fit_matrix(S, tp, 0.0, opt);
  }

  /**
   * Mode-A assembly: V^xc_ij = 1/2 [ Sigmahat_ij(eps_i) + Sigmahat_ij(eps_j) ], using the
   * SAME expansion for (i,j) and (j,i). Hermitian by construction -- deliberately NO
   * post-hoc Hermitization here (see the header).
   */
  inline nda::array<ComplexType, 2> assemble_vxc(matrix_expansion const &X,
                                                 nda::MemoryArrayOfRank<1> auto const &eps) {
    const long nb = X.nb;
    utils::check(eps.shape(0) == nb,
                 "sigma_real_axis::assemble_vxc: {} energies for an {} x {} block.",
                 eps.shape(0), nb, nb);
    nda::array<ComplexType, 2> V(nb, nb);
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        V(i, j) = 0.5 * (X.eval(i, j, eps(i)) + X.eval(i, j, eps(j)));
    return V;
  }

  /** one-line provenance for the run log. */
  inline void log_fit(int level, std::string_view who, fit_diag const &d,
                      long n_reexp_used, bool converged) {
    app_log(level, "  {}: rel resid = {:.3e}, imag_c_rel = {:.3e}, R_conv = {:.4g}, "
                   "|eps - z0|/R_conv = {:.3e}, re-expansions = {}, converged = {}",
            who, d.rel_resid, d.imag_c_rel, d.R_conv, d.dist_over_R, n_reexp_used,
            converged);
  }

} // sigma_real_axis
} // methods

#endif // COQUI_SIGMA_REAL_AXIS_HPP
