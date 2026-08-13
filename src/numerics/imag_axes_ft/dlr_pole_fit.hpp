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

#ifndef COQUI_DLR_POLE_FIT_HPP
#define COQUI_DLR_POLE_FIT_HPP

/**
 * REGULARIZED auxiliary DLR pole fit.
 *
 * The residue algebras that close Matsubara sums analytically (iaft_dconv's double bosonic
 * convolution, Pi^C's twisted pairs, Sigma^C's families I-V) all need the same primitive:
 * given a propagator-like object sampled on the backend's tau nodes, produce residues c_p
 * with
 *          F(z) = sum_p c_p / (z - eps_p).
 *
 * WHY THIS FILE EXISTS (measured 2026-07-27/28, notes/vertex_divergence_diagnosis.md section 0)
 * -------------------------------------------------------------------------------------------
 * Three call sites previously built those residues as
 *
 *      c = vals2coefs_NONSYM( Tmap . F ),   Tmap = interpolate(backend tau nodes -> aux nodes)
 *
 * i.e. a square interpolatory solve on an auxiliary grid, fed by an interpolation from the
 * backend grid. As one fixed linear map that composite has 2-norm 9.4e6 at the production
 * (lambda = 3959.84, eps = 1e-6) -- singular values spanning 9.4e6 down to 0.098. It is well
 * behaved on data that is exactly Lehmann-class, which is why it worked for years, but the
 * z objects are Lehmann only to the backend's OWN representation accuracy: they carry O(eps)
 * content in directions no pole basis resolves, and the square solve reproduces that content
 * by emitting enormous compensating residues. The downstream algebras are BILINEAR in the
 * residues, so the damage is squared.
 *
 * Measured, injecting a perturbation of relative size delta along the worst-conditioned
 * direction of the old composite (probe over the production grid):
 *
 *      delta     old max|c|/max|F|   old fit err     this map   fit err
 *      0                   0.83         2.2e-05         0.86    5.1e-06
 *      1e-6                3.81         2.4e-05         0.86    5.1e-06
 *      1e-5               37.8          2.4e-04         0.86    7.1e-06
 *      1e-4              379            2.4e-03         0.86    7.1e-05
 *
 * The delta = 1e-4 row reproduces the observed scGW+vertex blow-up (max|c|/max|z| = 383,
 * fit error 3.19e-03) to a few percent. Every physical input was flat across that break:
 * max|G_CC| 0.98212 -> 0.98246 -> 0.98259, max|Wbar| 1.417e-2 -> 1.406e-2 -> 1.401e-2,
 * max|z| 3.6834 -> 3.6904 -> 3.6774, and the pole-free part of Pi^C 3.2e-2 -> 2.9e-2 -> 1.2e-1,
 * while the residues went 3.0 -> 9.4 -> 1411 and Pibar went 0.33 -> 11 -> 1.2e10.
 *
 * WHAT THIS DOES INSTEAD
 * ----------------------
 * Fit the auxiliary poles DIRECTLY to the backend tau data by regularized least squares:
 *
 *      minimize ||K c - F||,   K(i,p) = K_F(tau_i, eps_p),   solved by truncated SVD
 *
 * WHAT ACTUALLY DOES THE WORK is LEAST SQUARES, not the SVD cut. A least-squares fit is free
 * to leave unrepresentable content in the residual; the old SQUARE interpolatory solve had to
 * reproduce it exactly at its nodes, and could only do that by emitting huge residues. Measured
 * at the amplitude that reproduces the production break, with the truncation switched off
 * entirely (all 38 directions kept), the residue ratio is 1.95 against the old route's 379.
 * Dropping Tmap also removes a gratuitous factor ||Tmap|| = 45 and is strictly more accurate
 * (it interpolated, then interpolated again).
 *
 * The rank is FIXED at build() from the singular spectrum -- see dlr_pole_fit_rel_tol for the
 * three requirements (gauge covariance, MPI invariance, elementwise batch semantics) that
 * rule out every data-dependent rule.
 *
 * Two REJECTED alternatives, recorded so they are not retried (both measured on the same grid):
 *   - "match the aux pole rank to the backend DLR rank" (i.e. use the backend's own SYM basis).
 *     cond(cf2it_SYM) = 6.2e10 against cond(cf2it_NONSYM) = 6.7e6, and on realistic Lehmann
 *     data the SYM basis emits residues 30-200x LARGER. The rank mismatch 38 < 40 is real but
 *     is not the defect; the SYM grid's near-degenerate +/- node pairs make it strictly worse.
 *     The NONSYM grid is retained here for exactly the reason it was chosen originally
 *     (min node gap 2.17 vs 0.0041 dimensionless -- the residue algebra divides by node gaps).
 *   - dropping the pole route entirely. The dynamic rung genuinely needs it; only the
 *     instantaneous rung is representable without poles.
 *
 * The fit is exact-to-eps on Lehmann-class data (that is what the DLR is), and its ACTUAL
 * error is measurable per call via fit_error() -- callers gate on it.
 */

#include <cmath>
#include <complex>
#include <algorithm>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"

#ifdef ENABLE_DLR
// same subset dlr_driver.hpp pulls in: cppdlr.hpp drags dlr_dyson.hpp -> nda/linalg/eigh.hpp,
// which this nda does not ship.
#include "numerics/imag_axes_ft/dlr/nda_linalg_compat.hpp"
#include "cppdlr/dlr_build.hpp"
#endif

namespace imag_axes_ft {

  /**
   * SVD truncation threshold, relative to the largest singular value of the pole kernel.
   *
   * THE RANK IS FIXED AT build() FROM THE SPECTRUM ALONE -- never from the data. Three
   * independent requirements force this, and any data-dependent rule violates at least one:
   *
   *  1. GAUGE COVARIANCE. Physical results depend on range(P) only: under U(k) -> U(k)V(k)
   *     both vertex cuts are exactly invariant (CLAUDE.md section 8), pinned by
   *     test_methods_vertex_wannier. The fit is applied to a batch whose columns carry the
   *     C-orbital index, so a gauge rotation MIXES those columns. Only a fixed linear map
   *     commutes with that mixing. Measured: a per-column data-adaptive rank gives a gauge
   *     deviation of 0.246 (test_dlr_pole_fit case 3c) and breaks the vertex gauge test at
   *     |D e_hf| = 4.8e-08 against its 1e-10 threshold.
   *  2. MPI INVARIANCE. In production the batch axis is distributed over ranks. A rank chosen
   *     from batch-summed quantities would make the physics depend on the processor grid.
   *  3. ELEMENTWISE BATCH SEMANTICS. double_boson_conv is documented elementwise in the
   *     trailing axis and test_iaft_dconv pins batched == per-element exactly.
   *
   * Requirements 2 and 3 together kill batch-wide rules; requirement 1 kills per-column rules.
   * What is left is a fixed threshold, and the value below was chosen by sweeping it against
   * both accuracy gates (see the table in notes/vertex_divergence_diagnosis.md section 8).
   *
   * Note this is NOT tied to the instance's eps: an eps-multiple would give a 0.1 cut at the
   * production setting (eps = 1e-6), which is absurd. It is an absolute conditioning cut.
   */
  inline constexpr double dlr_pole_fit_rel_tol = 1e-8;

  /** floor on the singular-value cut, relative to s_max: below this a direction is roundoff. */
  inline constexpr double dlr_pole_fit_smin_floor = 1e-14;

  /** stable analytic-continuation kernel K_F(s,e) = -e^{-e s}/(1 + e^{-beta e}), s in [0,beta]. */
  inline double dlr_kF(double beta, double s, double e) {
    if (e >= 0.0) return -std::exp(-e * s) / (1.0 + std::exp(-beta * e));
    return -std::exp(e * (beta - s)) / (1.0 + std::exp(beta * e));
  }

  /**
   * Auxiliary pole basis + regularized tau -> residue map. Build once per (IAFT, rel_tol).
   */
  struct dlr_pole_fit {
    long np = 0;                         // number of auxiliary poles
    long nt = 0;                         // backend fermionic tau nodes
    long n_kept = 0;                     // singular values retained by the truncation
    double beta = 0.0;
    double rel_tol = 0.0;
    double s_max = 0.0, s_min_kept = 0.0;
    double amplification = 0.0;          // ||Pinv||_2 = worst-case ||c|| / ||F||
    double min_abs_node = 0.0;           // min |hw_l|, dimensionless
    double min_node_gap = 0.0;           // min |hw_l - hw_l'|, dimensionless

    nda::array<double, 1> rf;            // dimensionless nodes hw_l = beta * eps_l
    nda::array<double, 1> epsl;          // physical pole energies
    nda::array<double, 1> s_phys;        // physical tau values of the backend mesh
    nda::array<double, 2> Kmat;          // (nt, np) reconstruction kernel
    nda::array<ComplexType, 2> Kc;       // (nt, np) same, complex, for the fit_error gemm
    // thin SVD of Kmat, kept so the rank can be chosen PER CALL from the data
    nda::array<ComplexType, 2> Ut;       // (ns, nt)  U^T   (ns = min(nt,np) usable directions)
    nda::array<ComplexType, 2> Vs;       // (np, ns)  V * diag(1/s), columns pre-scaled
    nda::array<double, 1> sval;          // (ns) singular values, descending
    long ns_max = 0;                     // usable directions after the roundoff floor

    dlr_pole_fit() = default;

    dlr_pole_fit(IAFT const& ft, double rtol = -1.0) { build(ft, rtol); }

    /** rtol < 0 selects the default, `dlr_pole_fit_rel_tol`. */
    void build(IAFT const& ft, double rtol = -1.0) {
#ifndef ENABLE_DLR
      (void)ft; (void)rtol;
      utils::check(false, "imag_axes_ft::dlr_pole_fit: requires the DLR backend "
                          "(build with ENABLE_DLR=ON).");
#else
      utils::check(ft.basis() == imag_axes_ft::dlr_basis,
                   "imag_axes_ft::dlr_pole_fit: requires the DLR imaginary-axis backend; "
                   "the IR backend does not expose the needed off-grid evaluations. "
                   "Rerun with iaft basis = \"dlr\".");
      // The accuracy TARGET for the per-call rank choice: a small multiple of the accuracy
      // the caller asked the DLR for. Not a singular-value threshold -- see the header.
      if (rtol < 0.0) rtol = dlr_pole_fit_rel_tol;
      utils::check(rtol > 0.0 and rtol < 1.0,
                   "imag_axes_ft::dlr_pole_fit: rel_tol = {} must be in (0,1).", rtol);
      beta = ft.beta();
      nt = ft.nt_f();
      rel_tol = rtol;

      // --- auxiliary NONSYM pole grid ------------------------------------------------
      // Kept deliberately: the residue algebras divide by node gaps, and the NONSYM grid's
      // min gap is ~500x larger than the backend SYM grid's (2.17 vs 0.0041 dimensionless).
      auto rf_v = cppdlr::build_dlr_rf(ft.lambda(), ft.eps());
      np = rf_v.size();
      rf = nda::array<double, 1>(np);
      epsl = nda::array<double, 1>(np);
      min_abs_node = 1e300;
      min_node_gap = 1e300;
      for (long l = 0; l < np; ++l) {
        rf(l) = rf_v(l);
        epsl(l) = rf_v(l) / beta;
        min_abs_node = std::min(min_abs_node, std::abs(rf_v(l)));
        for (long l2 = 0; l2 < l; ++l2)
          min_node_gap = std::min(min_node_gap, std::abs(rf_v(l) - rf_v(l2)));
      }
      utils::check(min_abs_node > 1e-12,
                   "imag_axes_ft::dlr_pole_fit: auxiliary pole grid contains a (near-)zero "
                   "node (min |hw_l| = {}); the bosonic residue map is singular there.",
                   min_abs_node);
      // The residue algebras deflate the SHARED-NODE diagonal by INDEX (the deflated term is
      // restored analytically as a true double pole). That is only equivalent to deflating by
      // VALUE while distinct nodes stay well separated: two nearly-equal-but-distinct nodes
      // would keep a 1/(eps_p - eps_q) term that no analytic piece compensates.
      utils::check(min_node_gap > 1e-6,
                   "imag_axes_ft::dlr_pole_fit: auxiliary pole grid has near-degenerate "
                   "distinct nodes (min gap = {} dimensionless). The residue algebras deflate "
                   "by node INDEX, which is only valid for a well-separated grid.",
                   min_node_gap);

      // --- physical tau values of the backend mesh -------------------------------------
      auto tau = ft.tau_mesh();
      s_phys = nda::array<double, 1>(nt);
      for (long i = 0; i < nt; ++i) s_phys(i) = (tau(i) + 1.0) * 0.5 * beta;

      // --- pole kernel on the backend tau grid ------------------------------------------
      Kmat = nda::array<double, 2>(nt, np);
      Kc = nda::array<ComplexType, 2>(nt, np);
      for (long i = 0; i < nt; ++i)
        for (long p = 0; p < np; ++p) {
          Kmat(i, p) = dlr_kF(beta, s_phys(i), epsl(p));
          Kc(i, p) = ComplexType(Kmat(i, p));
        }

      // --- thin SVD, kept whole: the rank is a per-call decision -------------------------
      nda::matrix<double, nda::F_layout> A(nt, np);
      A() = Kmat;
      long ms = std::min(nt, np);
      nda::vector<double> sig(ms);
      nda::matrix<double, nda::F_layout> U(nt, nt), VT(np, np);
      int info = nda::lapack::gesvd(A, sig, U, VT);
      utils::check(info == 0, "imag_axes_ft::dlr_pole_fit: gesvd failed (info = {}).", info);

      s_max = sig(0);
      ns_max = 0;
      while (ns_max < ms and sig(ns_max) > dlr_pole_fit_smin_floor * s_max) ++ns_max;
      utils::check(ns_max > 0, "imag_axes_ft::dlr_pole_fit: the pole kernel has no usable "
                               "singular directions.");
      // The rank is fixed HERE, from the spectrum and the instance's target only -- never
      // from the data. See the class comment.
      n_kept = 0;
      while (n_kept < ns_max and sig(n_kept) > rel_tol * s_max) ++n_kept;
      utils::check(n_kept > 0, "imag_axes_ft::dlr_pole_fit: truncation kept no singular "
                               "values (rel_tol = {}).", rel_tol);
      s_min_kept = sig(n_kept - 1);
      amplification = 1.0 / s_min_kept;

      sval = nda::array<double, 1>(ns_max);
      Ut = nda::array<ComplexType, 2>(ns_max, nt);
      Vs = nda::array<ComplexType, 2>(np, ns_max);
      for (long k = 0; k < ns_max; ++k) {
        sval(k) = sig(k);
        for (long i = 0; i < nt; ++i) Ut(k, i) = ComplexType(U(i, k));
        double inv = 1.0 / sig(k);
        for (long p = 0; p < np; ++p) Vs(p, k) = ComplexType(VT(k, p) * inv);
      }
#endif
    }

    /**
     * Residues of tau-grid data (leading axis nt, trailing axis a flat batch).
     * F(z) = sum_p c_p / (z - eps_p).
     *
     * DISCREPANCY PRINCIPLE. In the SVD basis the least-squares residual after keeping k
     * directions is  ||F||^2 - sum_{j<k} |g_j|^2  with g = U^T F, so the whole residual-vs-rank
     * curve is available from one gemm. Keep the smallest k whose residual is already at the
     * requested accuracy; adding directions past that point cannot improve a fit that has
     * plateaued and only inflates the residues, which the bilinear residue algebra squares.
     * `n_kept` records the rank actually used, for the log.
     */
    nda::array<ComplexType, 2> coeffs(nda::MemoryArrayOfRank<2> auto const& F_td) const {
      utils::check(F_td.shape(0) == nt,
                   "imag_axes_ft::dlr_pole_fit::coeffs: leading axis {} != nt = {}.",
                   F_td.shape(0), nt);
      const long d = F_td.shape(1);
      nda::array<ComplexType, 2> c(np, d);
      if (d == 0) return c;

      // FIXED RANK, decided at build() from the singular spectrum alone. The map must not
      // depend on the data -- see the class comment for the three requirements that forces.
      nda::array<ComplexType, 2> g(n_kept, d);
      auto Utk = Ut(nda::range(0, n_kept), nda::range::all);
      nda::array<ComplexType, 2> Utk_c(n_kept, nt);
      Utk_c() = Utk;
      nda::blas::gemm(Utk_c, F_td, g);

      auto Vk = Vs(nda::range::all, nda::range(0, n_kept));
      nda::array<ComplexType, 2> Vk_c(np, n_kept);
      Vk_c() = Vk;
      nda::blas::gemm(Vk_c, g, c);
      return c;
    }

    /**
     * max|c| / max|F| -- the residue amplification actually realised on THIS data.
     *
     * WATCH THIS, NOT ONLY fit_error. Measured on Si kp222/M12 head-on (2026-07-28), the one
     * window that still diverges with the regularized fit in place:
     *
     *     iter   max|z|   max|residue|   res/z    fit err     Pibar
     *      1     3.7519      3.1885       0.85    9.7e-06    3.2e+03
     *      2     3.4301     54.30        15.8     4.5e-05    2.5e+05     <- fit still "healthy"
     *      3     3.4487   6306.8       1829       4.6e-03    2.0e+13
     *
     * The ratio moves a FULL ITERATION before the fit error does. That is the signature of the
     * post-fix failure mode and it is qualitatively different from the pre-fix one: least
     * squares reports an accurate fit (4.5e-05) while the data genuinely REQUIRES residues 16x
     * its own size, i.e. it is a near-cancelling combination of poles. The fit is not lying --
     * the downstream algebra is bilinear in these residues and squares them.
     *
     * Healthy values are 0.70-0.91 across every converged window and both meshes, so a warning
     * at 50 has ~50x headroom. Deliberately NOT a hard gate: a legitimately near-cancelling
     * object is possible, the threshold has not been calibrated across systems, and a false
     * abort costs a multi-hour run. fit_error remains the hard gate.
     */
    double residue_ratio(nda::MemoryArrayOfRank<2> auto const& F_td,
                         nda::array<ComplexType, 2> const& c) const {
      double cm = 0.0, fm = 0.0;
      for (auto const& v : c) cm = std::max(cm, std::abs(v));
      for (auto const& v : F_td) fm = std::max(fm, std::abs(v));
      return (fm > 0.0) ? cm / fm : 0.0;
    }

    /**
     * Relative max-norm reconstruction error on the SAME tau grid the data came from:
     *   err = max|F - sum_p c_p K_F(s, eps_p)| / max|F|.
     * This is the honest accuracy of the pole representation for THIS data. It is the
     * quantity callers gate on: the old single-synthetic-pole self-check passed through
     * every divergence because a single pole is trivially in the span.
     */
    double fit_error(nda::MemoryArrayOfRank<2> auto const& F_td,
                     nda::array<ComplexType, 2> const& c) const {
      // BLOCKED GEMM, not a scalar triple loop. This runs on every object the fit touches,
      // including inside Sigma^C's per-tuple loop, so it is written for the PRODUCTION batch
      // width (dXF ~ 3e4 at nbnd = 60, nc = 8): the naive form (rec += c(p,j)*Kmat(s,p),
      // strided in p over c) has the same flop count as the reconstruction gemm but a far
      // worse access pattern. Blocking keeps the temp in cache instead of allocating an
      // nt x d array, which would be tens of MB per call at that width.
      //   Measured honestly: at the SMALL batch widths of the unit tests this costs nothing
      //   either way -- ablating the fit_error calls entirely moves test_methods_vertex_sigma
      //   from 234.5 s to 235.0 s. (An earlier note here claimed the scalar form caused a 16x
      //   slowdown; that was a misreading of a stale CTestCostData average from when the test
      //   was smaller, and is retracted.)
      long d = F_td.shape(1);
      if (d == 0) return 0.0;
      constexpr long BJ = 256;
      const long nb = std::min(d, BJ);
      // CONTIGUOUS buffers. A column slice of `c` is strided, and handing that straight to
      // gemm segfaults; the tail block is zero-padded so every gemm sees full-width
      // contiguous operands.
      nda::array<ComplexType, 2> cb(np, nb), rb(nt, nb);
      double num = 0.0, den = 0.0;
      for (long j0 = 0; j0 < d; j0 += nb) {
        const long nj = std::min(nb, d - j0);
        for (long p = 0; p < np; ++p) {
          for (long j = 0; j < nj; ++j) cb(p, j) = c(p, j0 + j);
          for (long j = nj; j < nb; ++j) cb(p, j) = ComplexType(0.0);
        }
        nda::blas::gemm(ComplexType(1.0), Kc, cb, ComplexType(0.0), rb);
        for (long s = 0; s < nt; ++s)
          for (long j = 0; j < nj; ++j) {
            num = std::max(num, std::abs(F_td(s, j0 + j) - rb(s, j)));
            den = std::max(den, std::abs(F_td(s, j0 + j)));
          }
      }
      return (den > 0.0) ? num / den : 0.0;
    }

    /** one-line provenance for the run log. */
    void log(int level, std::string_view who) const {
      app_log(level, "  {}: DLR rank = {} (aux pole rank = {}), least-squares pole fit, rank "
                     "chosen per call to reach rel accuracy {:.2g}\n"
                     "    -> {} of {} usable directions on the last call, worst-case residue "
                     "amplification {:.4g}, min|hw_l| = {:.4g}, min node gap = {:.4g}",
              who, nt, np, rel_tol, n_kept, ns_max, amplification,
              min_abs_node, min_node_gap);
    }
  };

  /**
   * SUPPORT-CONSTRAINED (masked-columns) auxiliary pole fit -- Project 2 increment QM3,
   * promoted verbatim from the QM2-b measurement (test_qp_maps_matsubara.cpp, gate
   * "route_b_fitted_W_chain") so that production and its unit gate share ONE code path.
   *
   * WHY IT EXISTS (measured, QM2-b; notes/qm2_route_b_finite_t_spec.md)
   * ------------------------------------------------------------------
   * `dlr_pole_fit` is exact-to-eps ON THE IMAGINARY AXIS, which is all the residue algebras
   * of project 1 ever ask of it. Route B (sigma_route_b.hpp) instead evaluates the fitted
   * measure at REAL arguments eps_l - z. Auxiliary nodes that carry spurious weight where
   * W^c has no spectral support are then divided by a vanishing gap: on the QM2-b fixture
   * the plain fit misses Sigma^c(real z) by 1.5e+04 eV while its tau fit_error is 1e-5.
   *
   * The cure is PRIOR PHYSICAL INFORMATION, not regularization: a bosonic W^c has no
   * spectral weight inside the particle-hole gap, so the kernel columns of auxiliary nodes
   * with |eps_p| < gap_edge are REMOVED from the least squares. Measured on the same
   * fixture: 20.8 / 2.6 / 0.14 meV at DLR prec low / medium / high, i.e. six orders better
   * than the plain fit, with the residual set by how well the retained nodes cover the
   * support (NOT by the DLR accuracy -- more precision alone does not fix the plain fit).
   *
   * DOCTRINE, inherited unchanged from dlr_pole_fit
   * ----------------------------------------------
   *  - the map is a FIXED linear map: the retained column set comes from gap_edge and the
   *    rank from the singular spectrum at build(), never from the data. That is what makes
   *    "fit in the auxiliary basis, then contract" identical to "contract, then fit"
   *    (QM3 spec section 3), and it is required for gauge covariance / MPI invariance /
   *    elementwise batch semantics exactly as in the parent struct.
   *  - the same `dlr_pole_fit_rel_tol` truncated-SVD cut, and `fit_error` is the same honest
   *    max-norm reconstruction error on the grid the data came from.
   *
   * TWO KERNELS (QM3 knob qp_modea_wfit)
   *  - from_tau: the tested convention. Columns are dlr_kF(beta, tau_i, eps_p) on the
   *    backend FERMIONIC tau grid; the frequency-domain BOSONIC residues are
   *    w_p = tanh(hw_p/2) * c_p (`residue_scale`), the iaft_dconv convention.
   *  - from_matsubara: columns are 1/(i nu_m - eps_p) on the supplied node list; the
   *    coefficients ARE the residues (residue_scale == 1).
   */
  struct masked_pole_fit {
    long nrow = 0;        // sample rows (tau nodes or Matsubara nodes)
    long np_all = 0;      // full auxiliary pole grid
    long nkeep = 0;       // retained columns (the support)
    long n_kept = 0;      // singular directions retained
    double rel_tol = 0.0;
    double s_max = 0.0, s_min_kept = 0.0, amplification = 0.0;
    double gap_edge = 0.0;
    bool nu_route = false;

    nda::array<long, 1> keep;            // (nkeep) indices into the full auxiliary grid
    nda::array<double, 1> om;            // (nkeep) retained pole energies, physical units
    nda::array<double, 1> hw;            // (nkeep) dimensionless nodes beta * om
    nda::array<double, 1> residue_scale; // (nkeep) tanh(hw/2) [tau] or 1 [nu]
    nda::array<ComplexType, 2> Kc;       // (nrow, nkeep) reconstruction kernel
    nda::array<ComplexType, 2> Uc;       // (n_kept, nrow)  conj(U)^T
    nda::array<ComplexType, 2> Vc;       // (nkeep, n_kept) V (NOT pre-scaled by 1/s)
    nda::array<double, 1> sval;          // (n_kept) singular values, descending

    masked_pole_fit() = default;

    /** columns retained by the support constraint; gap_edge <= 0 keeps every column. */
    static std::vector<long> support_columns(nda::array<double, 1> const& epsl, double gap_edge) {
      std::vector<long> keep;
      for (long p = 0; p < epsl.shape(0); ++p)
        if (gap_edge <= 0.0 or std::abs(epsl(p)) >= gap_edge) keep.push_back(p);
      return keep;
    }

    /**
     * Generic build from an explicit kernel. `K` is (nrow, np_all) in the FULL auxiliary
     * basis; only the columns in `keep_idx` enter the least squares.
     *
     * The real-kernel path runs a REAL gesvd, and the apply below accumulates in exactly the
     * order the QM2-b measurement used, so the promoted utility reproduces that gate's
     * numbers bit for bit rather than merely to its printed precision.
     */
    template<typename S>
    void build(nda::MemoryArrayOfRank<2> auto const& K,
               std::vector<long> const& keep_idx,
               nda::array<double, 1> const& epsl_all,
               nda::array<double, 1> const& hw_all,
               bool tanh_residues, double beta_unused = 0.0, double rtol = -1.0) {
      (void)beta_unused;
      nrow = K.shape(0);
      np_all = K.shape(1);
      nkeep = long(keep_idx.size());
      nu_route = not tanh_residues;
      if (rtol < 0.0) rtol = dlr_pole_fit_rel_tol;
      rel_tol = rtol;
      utils::check(nkeep > 0, "imag_axes_ft::masked_pole_fit: the support constraint retained "
                              "0 of {} auxiliary nodes.", np_all);
      utils::check(nrow >= 1, "imag_axes_ft::masked_pole_fit: empty sample grid.");

      keep = nda::array<long, 1>(nkeep);
      om = nda::array<double, 1>(nkeep);
      hw = nda::array<double, 1>(nkeep);
      residue_scale = nda::array<double, 1>(nkeep);
      for (long q = 0; q < nkeep; ++q) {
        keep(q) = keep_idx[q];
        om(q) = epsl_all(keep_idx[q]);
        hw(q) = hw_all(keep_idx[q]);
        residue_scale(q) = tanh_residues ? std::tanh(0.5 * hw(q)) : 1.0;
      }

      Kc = nda::array<ComplexType, 2>(nrow, nkeep);
      for (long i = 0; i < nrow; ++i)
        for (long q = 0; q < nkeep; ++q) Kc(i, q) = ComplexType(K(i, keep(q)));

      const long ms = std::min(nrow, nkeep);
      nda::vector<double> sig(ms);
      nda::matrix<S, nda::F_layout> A(nrow, nkeep), U(nrow, nrow), VT(nkeep, nkeep);
      for (long i = 0; i < nrow; ++i)
        for (long q = 0; q < nkeep; ++q) A(i, q) = S(K(i, keep(q)));
      const int info = nda::lapack::gesvd(A, sig, U, VT);
      utils::check(info == 0, "imag_axes_ft::masked_pole_fit: gesvd failed on the reduced "
                              "kernel (info = {}).", info);
      s_max = sig(0);
      n_kept = 0;
      while (n_kept < ms and sig(n_kept) > rel_tol * s_max) ++n_kept;
      utils::check(n_kept > 0, "imag_axes_ft::masked_pole_fit: truncation kept no singular "
                               "values (rel_tol = {}).", rel_tol);
      s_min_kept = sig(n_kept - 1);
      amplification = 1.0 / s_min_kept;

      sval = nda::array<double, 1>(n_kept);
      Uc = nda::array<ComplexType, 2>(n_kept, nrow);
      Vc = nda::array<ComplexType, 2>(nkeep, n_kept);
      for (long k = 0; k < n_kept; ++k) {
        sval(k) = sig(k);
        for (long i = 0; i < nrow; ++i) Uc(k, i) = std::conj(ComplexType(U(i, k)));
        for (long q = 0; q < nkeep; ++q) Vc(q, k) = std::conj(ComplexType(VT(k, q)));
      }
    }

    /** the QM2-b tau chain: backend fermionic tau nodes, bosonic residues via tanh(hw/2). */
    static masked_pole_fit from_tau(dlr_pole_fit const& pf, double gap_edge, double rtol = -1.0) {
      masked_pole_fit f;
      f.gap_edge = gap_edge;
      f.build<double>(pf.Kmat, support_columns(pf.epsl, gap_edge), pf.epsl, pf.rf, true,
                      pf.beta, rtol);
      return f;
    }

    /** the nu route: LS directly on Matsubara-node data, kernel 1/(i nu_m - eps_p). */
    static masked_pole_fit from_matsubara(dlr_pole_fit const& pf,
                                          nda::array<ComplexType, 1> const& z,
                                          double gap_edge, double rtol = -1.0) {
      masked_pole_fit f;
      f.gap_edge = gap_edge;
      nda::array<ComplexType, 2> K(z.shape(0), pf.np);
      for (long m = 0; m < z.shape(0); ++m)
        for (long p = 0; p < pf.np; ++p) K(m, p) = 1.0 / (z(m) - pf.epsl(p));
      f.build<ComplexType>(K, support_columns(pf.epsl, gap_edge), pf.epsl, pf.rf, false,
                           pf.beta, rtol);
      return f;
    }

    /**
     * Residues of grid data (leading axis nrow, trailing axis a flat batch) -> (nkeep, d).
     * FIXED rank and FIXED column set; the accumulation order is the QM2-b one.
     */
    nda::array<ComplexType, 2> coeffs(nda::MemoryArrayOfRank<2> auto const& F) const {
      utils::check(F.shape(0) == nrow,
                   "imag_axes_ft::masked_pole_fit::coeffs: leading axis {} != nrow = {}.",
                   F.shape(0), nrow);
      const long d = F.shape(1);
      nda::array<ComplexType, 2> c(nkeep, d);
      c() = ComplexType(0.0);
      for (long j = 0; j < d; ++j)
        for (long k = 0; k < n_kept; ++k) {
          ComplexType g(0.0);
          for (long i = 0; i < nrow; ++i) g += Uc(k, i) * F(i, j);
          g /= sval(k);
          for (long q = 0; q < nkeep; ++q) c(q, j) += Vc(q, k) * g;
        }
      return c;
    }

    /** relative max-norm reconstruction error on the SAME grid the data came from. */
    double fit_error(nda::MemoryArrayOfRank<2> auto const& F,
                     nda::array<ComplexType, 2> const& c) const {
      const long d = F.shape(1);
      double num = 0.0, den = 0.0;
      for (long i = 0; i < nrow; ++i)
        for (long j = 0; j < d; ++j) {
          ComplexType rec(0.0);
          for (long q = 0; q < nkeep; ++q) rec += Kc(i, q) * c(q, j);
          num = std::max(num, std::abs(F(i, j) - rec));
          den = std::max(den, std::abs(F(i, j)));
        }
      return (den > 0.0) ? num / den : 0.0;
    }

    /** max|c| / max|F| -- see dlr_pole_fit::residue_ratio for why this is watched. */
    double residue_ratio(nda::MemoryArrayOfRank<2> auto const& F,
                         nda::array<ComplexType, 2> const& c) const {
      double cm = 0.0, fm = 0.0;
      for (auto const& v : c) cm = std::max(cm, std::abs(v));
      for (auto const& v : F) fm = std::max(fm, std::abs(v));
      return (fm > 0.0) ? cm / fm : 0.0;
    }
  };

  /**
   * Hard gate on the measured reconstruction error.  `warn_at` only logs; past `abort_at` the
   * pole representation has demonstrably failed and every downstream residue product is
   * meaningless, so continuing would silently poison the self-consistency loop.
   *
   * NOT collective, and it does not need to be: utils::check -> APP_ABORT -> MPI_Abort on
   * MPI_COMM_WORLD, so a gate that fires on any subset of ranks tears the whole job down
   * rather than deadlocking the ranks that passed. Callers may still reduce first when they
   * want the REPORTED number to be the global max (eval_sigma_C does; eval_Pi_C does not need
   * to, because it measures replicated z data and every rank computes the same value).
   */
  inline void dlr_pole_fit_gate(double err, std::string_view who,
                                double warn_at = 1e-3, double abort_at = 1e-2,
                                double res_ratio = -1.0, double ratio_warn_at = 50.0) {
    // EARLY WARNING: the residue ratio leads the fit error by about one scf iteration in the
    // post-fix failure mode (see dlr_pole_fit::residue_ratio). Report it first, because by the
    // time `err` trips, the bilinear algebra has already squared these residues.
    if (res_ratio > ratio_warn_at)
      app_log(1, "  [WARNING] {}: auxiliary pole residues are {:.4g}x the data they represent "
                 "(healthy 0.7-0.9).\n"
                 "            The fit may still look accurate -- the data genuinely needs these "
                 "residues, and the\n"
                 "            downstream algebra is BILINEAR in them. This usually leads the "
                 "reconstruction error by\n"
                 "            one iteration; if it is climbing, the run is on its way out.",
              who, res_ratio);
    if (err <= warn_at) return;
    if (err <= abort_at) {
      app_log(1, "  [WARNING] {}: auxiliary DLR pole fit reconstruction error = {:.4g} "
                 "(warn threshold {:.1g}).\n"
                 "            The tau data is drifting out of the representable class; the "
                 "residue algebra is bilinear\n"
                 "            in the residues, so this error enters downstream squared.",
              who, err, warn_at);
      return;
    }
    utils::check(false,
                 "{}: auxiliary DLR pole fit FAILED -- reconstruction error = {:.4g} exceeds "
                 "the hard gate {:.1g}.\nThe tau data is not representable by the auxiliary "
                 "pole basis at this (lambda, eps); every downstream residue product is "
                 "meaningless.\nRaise the IAFT precision (prec/eps) or the frequency cutoff "
                 "wmax, or lower the pole-fit rel_tol.", who, err, abort_at);
  }

  /**
   * scGW-tilde increment C2: the SAME regularized auxiliary pole fit, ingesting
   * FERMIONIC MATSUBARA-NODE data instead of tau data.
   *
   * WHY IT EXISTS. The CVV head builds M_alpha = v~_alpha G as a PRODUCT at the
   * backend's fermionic iw nodes (that is where both factors live); its tau
   * representation is exactly what the fit is asked to produce, so a tau-side fit is
   * circular, and pushing the product through the square interpolatory w -> tau map is
   * the alias trap this file exists to forbid (rule 7 of
   * notes/scgwt_implementation_plan.md). Kernel
   *
   *      K(n,p) = 1 / (iw_n - eps_p)     (backend fermionic nodes iw_n, aux poles eps_p)
   *
   * complex truncated SVD, and the identical FIXED-RANK doctrine: the rank is chosen at
   * build() from the singular spectrum alone, never from the data (same three
   * requirements as the tau-side struct). fit_error is the same honest max-norm
   * reconstruction error, measured on the SAME iw grid the data came from; callers gate
   * on it with dlr_pole_fit_gate exactly as for the tau fit. Residues share the aux
   * NONSYM pole grid, so downstream kernel evaluations (dlr_kF at arbitrary tau) and
   * any residue algebra see the same well-separated nodes.
   */
  struct dlr_pole_fit_w {
    long np = 0;                         // number of auxiliary poles
    long nw = 0;                         // backend fermionic Matsubara nodes
    long n_kept = 0;
    double beta = 0.0;
    double rel_tol = 0.0;
    double s_max = 0.0, s_min_kept = 0.0;
    double amplification = 0.0;

    nda::array<double, 1> epsl;          // physical pole energies (aux NONSYM grid)
    nda::array<ComplexType, 1> iwn;      // physical iw_n of the backend fermionic mesh
    nda::array<ComplexType, 2> Kc;       // (nw, np) reconstruction kernel
    nda::array<ComplexType, 2> Ut;       // (ns, nw)  U^H (conjugate transpose)
    nda::array<ComplexType, 2> Vs;       // (np, ns)  V * diag(1/s)
    nda::array<double, 1> sval;
    long ns_max = 0;

    dlr_pole_fit_w() = default;
    dlr_pole_fit_w(IAFT const& ft, double rtol = -1.0) { build(ft, rtol); }

    void build(IAFT const& ft, double rtol = -1.0) {
#ifndef ENABLE_DLR
      (void)ft; (void)rtol;
      utils::check(false, "imag_axes_ft::dlr_pole_fit_w: requires the DLR backend "
                          "(build with ENABLE_DLR=ON).");
#else
      utils::check(ft.basis() == imag_axes_ft::dlr_basis,
                   "imag_axes_ft::dlr_pole_fit_w: requires the DLR imaginary-axis backend.");
      if (rtol < 0.0) rtol = dlr_pole_fit_rel_tol;
      utils::check(rtol > 0.0 and rtol < 1.0,
                   "imag_axes_ft::dlr_pole_fit_w: rel_tol = {} must be in (0,1).", rtol);
      beta = ft.beta();
      nw = ft.nw_f();
      rel_tol = rtol;

      // the SAME auxiliary NONSYM pole grid as the tau-side fit (see its build())
      auto rf_v = cppdlr::build_dlr_rf(ft.lambda(), ft.eps());
      np = rf_v.size();
      epsl = nda::array<double, 1>(np);
      for (long l = 0; l < np; ++l) epsl(l) = rf_v(l) / beta;

      auto wn = ft.wn_mesh_f();
      iwn = nda::array<ComplexType, 1>(nw);
      for (long n = 0; n < nw; ++n) iwn(n) = ft.omega(wn(n));

      Kc = nda::array<ComplexType, 2>(nw, np);
      for (long n = 0; n < nw; ++n)
        for (long p = 0; p < np; ++p) Kc(n, p) = 1.0 / (iwn(n) - epsl(p));

      nda::matrix<ComplexType, nda::F_layout> A(nw, np);
      A() = Kc;
      long ms = std::min(nw, np);
      nda::vector<double> sig(ms);
      nda::matrix<ComplexType, nda::F_layout> U(nw, nw), VH(np, np);
      int info = nda::lapack::gesvd(A, sig, U, VH);
      utils::check(info == 0, "imag_axes_ft::dlr_pole_fit_w: gesvd failed (info = {}).", info);

      s_max = sig(0);
      ns_max = 0;
      while (ns_max < ms and sig(ns_max) > dlr_pole_fit_smin_floor * s_max) ++ns_max;
      utils::check(ns_max > 0, "imag_axes_ft::dlr_pole_fit_w: no usable singular directions.");
      // the rank is fixed HERE, from the spectrum and the instance's target only
      n_kept = 0;
      while (n_kept < ns_max and sig(n_kept) > rel_tol * s_max) ++n_kept;
      utils::check(n_kept > 0, "imag_axes_ft::dlr_pole_fit_w: truncation kept no singular "
                               "values (rel_tol = {}).", rel_tol);
      s_min_kept = sig(n_kept - 1);
      amplification = 1.0 / s_min_kept;

      sval = nda::array<double, 1>(ns_max);
      Ut = nda::array<ComplexType, 2>(ns_max, nw);
      Vs = nda::array<ComplexType, 2>(np, ns_max);
      for (long k = 0; k < ns_max; ++k) {
        sval(k) = sig(k);
        // complex LS: c = V diag(1/s) U^H F -- conjugate transposes, unlike the real case
        for (long n = 0; n < nw; ++n) Ut(k, n) = std::conj(U(n, k));
        double inv = 1.0 / sig(k);
        for (long p = 0; p < np; ++p) Vs(p, k) = std::conj(VH(k, p)) * inv;
      }
#endif
    }

    /** Residues of Matsubara-node data (leading axis nw, trailing axis a flat batch). */
    nda::array<ComplexType, 2> coeffs(nda::MemoryArrayOfRank<2> auto const& F_wd) const {
      utils::check(F_wd.shape(0) == nw,
                   "imag_axes_ft::dlr_pole_fit_w::coeffs: leading axis {} != nw = {}.",
                   F_wd.shape(0), nw);
      const long d = F_wd.shape(1);
      nda::array<ComplexType, 2> c(np, d);
      if (d == 0) return c;
      nda::array<ComplexType, 2> g(n_kept, d);
      nda::array<ComplexType, 2> Utk_c(n_kept, nw);
      Utk_c() = Ut(nda::range(0, n_kept), nda::range::all);
      nda::blas::gemm(Utk_c, F_wd, g);
      nda::array<ComplexType, 2> Vk_c(np, n_kept);
      Vk_c() = Vs(nda::range::all, nda::range(0, n_kept));
      nda::blas::gemm(Vk_c, g, c);
      return c;
    }

    /** Relative max-norm reconstruction error on the SAME iw grid the data came from. */
    double fit_error(nda::MemoryArrayOfRank<2> auto const& F_wd,
                     nda::array<ComplexType, 2> const& c) const {
      long d = F_wd.shape(1);
      if (d == 0) return 0.0;
      constexpr long BJ = 256;
      const long nb = std::min(d, BJ);
      nda::array<ComplexType, 2> cb(np, nb), rb(nw, nb);
      double num = 0.0, den = 0.0;
      for (long j0 = 0; j0 < d; j0 += nb) {
        const long nj = std::min(nb, d - j0);
        for (long p = 0; p < np; ++p) {
          for (long j = 0; j < nj; ++j) cb(p, j) = c(p, j0 + j);
          for (long j = nj; j < nb; ++j) cb(p, j) = ComplexType(0.0);
        }
        nda::blas::gemm(ComplexType(1.0), Kc, cb, ComplexType(0.0), rb);
        for (long s = 0; s < nw; ++s)
          for (long j = 0; j < nj; ++j) {
            num = std::max(num, std::abs(F_wd(s, j0 + j) - rb(s, j)));
            den = std::max(den, std::abs(F_wd(s, j0 + j)));
          }
      }
      return (den > 0.0) ? num / den : 0.0;
    }

    /** max|c| / max|F| -- see dlr_pole_fit::residue_ratio for why this is watched. */
    double residue_ratio(nda::MemoryArrayOfRank<2> auto const& F_wd,
                         nda::array<ComplexType, 2> const& c) const {
      double cm = 0.0, fm = 0.0;
      for (auto const& v : c) cm = std::max(cm, std::abs(v));
      for (auto const& v : F_wd) fm = std::max(fm, std::abs(v));
      return (fm > 0.0) ? cm / fm : 0.0;
    }
  };

} // namespace imag_axes_ft

#endif // COQUI_DLR_POLE_FIT_HPP
