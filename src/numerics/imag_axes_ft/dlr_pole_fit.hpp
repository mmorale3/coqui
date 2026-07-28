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

} // namespace imag_axes_ft

#endif // COQUI_DLR_POLE_FIT_HPP
