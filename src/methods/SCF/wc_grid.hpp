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

#ifndef COQUI_WC_GRID_HPP
#define COQUI_WC_GRID_HPP

/**
 * ===========================================================================
 * TC-5 -- THE AMORTIZED W^c TILE CACHE
 * ===========================================================================
 * notes/tc_coqui_impl_spec.md "TC-5"; validation in
 * notes/tilted_contour_validation_results.md sections 8 (Axis D) and 8.7.
 *
 * ---------------------------------------------------------------------------
 * THE FORMULATION GAP
 * ---------------------------------------------------------------------------
 * The eq-1 residue term needs <aJ|W^c(q_J, eps_J - omega + i delta)|Jb> at
 * targets SCATTERED along the line Im z = delta. CoQuI answered each with its
 * own dense Np^3 Dyson solve -- ~2.5 s x 10^4..10^5 per map at Np ~ 1269, which
 * is what made the route "too slow to be useful".
 *
 * W^c(omega + i delta) is delta-SMOOTH by construction: its nearest
 * singularities are the plasmon poles, which sit at distance delta BELOW the
 * line. So it can be built ONCE on a shared grid of spacing h and read off by
 * local interpolation. MEASURED (Axis D, 9 fixtures, exact closed-form
 * reference):
 *
 *      dSigma [meV]  =  K (h/delta)^p / delta[eV]        p = 2.81
 *
 * 29-170 grid points replace 10^4-10^5 solves: 272x-668x on the residue tier.
 *
 * ⚠ THIS IS NOT THE UNIFORM GRID THE CAMPAIGN REJECTED. TCV section 5.2
 * rejected a uniform grid as the REPRESENTATION of P (it loses the 123-1584x
 * the ID buys). This is the CONSUMPTION of an already-built W^c. P still comes
 * from the tilted contour on its ID nodes and the Dyson chain is unchanged;
 * only the number of TARGETS at which that chain runs changes.
 *
 * ---------------------------------------------------------------------------
 * ⚠ THE LAW'S CONSTANT IS NOT UNIVERSAL -- WHICH IS WHY THERE IS AN AUDIT
 * ---------------------------------------------------------------------------
 * Axis D section 8.7 measured K/|Sigma| spanning 1.3 to 417 across plasmon
 * densities. The outlier is a spectrum whose weight is CONCENTRATED on a single
 * in-range pole -- and NOTHING in the sizing inputs (W_band, N_k, delta) reveals
 * it. A silently under-resolved grid then produces a plausible-looking Sigma,
 * which is the failure mode this project has been burned by twice.
 *
 * So the law SIZES the grid and a SAMPLE PROVES it: `audit_wc_grid` evaluates
 * W^c EXACTLY at a handful of real residue targets and compares. Under 3 % of
 * the fill, ~0.005 % of the scheme it replaces. See `wgrid_audit_t` for the
 * defined failure behaviour (warn <= 10x, hard abort above).
 *
 * ---------------------------------------------------------------------------
 * COLLECTIVE DISCIPLINE (the 46f9dfc invariant, one level up)
 * ---------------------------------------------------------------------------
 * The FILL is collective and lives at a lockstep point next to
 * `p_contour::gather_Z_tiles`: the flat (iq, j) index over nq_ibz x N is
 * partitioned across ranks, every rank runs only local Dyson solves, and ONE
 * reduction assembles the shared array. There is no collective inside the loop,
 * so a rank's own trip count is irrelevant.
 *
 * The READ is pure-local and that is enforced by SIGNATURE: `wc_grid_t::at` is
 * a const member over a node-shared array and the evaluator receives the CACHE,
 * never a solver. Reintroducing a per-target solve would require adding a
 * parameter, not merely adding a call.
 */

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "itertools/itertools.hpp"
#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/tilted_contour/tilted_contour.hpp"
#include "methods/SCF/wc_line.hpp"

namespace methods {
namespace wc_grid {

  namespace tc = tilted_contour;
  inline constexpr double ha_to_eV = 27.211386245988;

  // =========================================================================
  //  (a) THE SIZING LAW  -- notes/tilted_contour_validation_results.md 8.2/8.3
  // =========================================================================
  /**
   * The measured constants. K is SVO's -- the largest of the three materials --
   * used for EVERY system by ruling: do NOT auto-detect metallicity, be
   * conservative and let the audit catch what conservatism misses.
   */
  inline constexpr double wgrid_K = 32.4;        ///< meV.eV  (Axis D 8.2, SVO)
  inline constexpr double wgrid_p = 2.81;        ///< the exponent (SVO)
  inline constexpr double wgrid_safety = 3.0;    ///< the measured fit spread is x1.4-2.5
  /** past h = delta/2 the 3-point stencil overshoots and can be worse than
   *  linear (Axis D 8.1) -- the validity clamp, not a tuning choice. */
  inline constexpr double wgrid_hmax_over_delta = 0.5;
  /**
   * ⚠⚠ THE delta-CREDIT CEILING (Axis D3, 2026-08-26) -- the constant that stops
   * the law promising accuracy it does not deliver at large delta.
   *
   * The law carries a 1/delta prefactor: at fixed h/delta it claims the Sigma error
   * falls as 1/delta. MEASURED (tests/run_D3_absh.py, si444 + si222, delta swept
   * 2.4 -> 16 eV at matched h/delta) that is FALSE above ~2 eV -- the error is FLAT
   * or RISING where the law predicts a 0.15-0.23x fall:
   *
   *     h/delta ~ 0.49 :  delta = 2.4 -> 16 eV,  measured 0.987 -> 3.230 meV
   *                       (the law predicts a factor 0.15; measured 3.27)
   *
   * MECHANISM: delta buys smoothness only while it is the SMALLEST scale in the
   * problem. Once delta exceeds the spectral feature spacing, the curvature of W
   * along the line is set by the SPECTRUM's own structure, not by delta, and
   * further delta buys nothing. So the credit must stop -- at the scale where it
   * stopped being earned.
   *
   * THE COST OF NOT HAVING THIS: tc4_444nb60_d35_w, at delta = 0.476 a.u. =
   * 12.95 eV, was sized to h = 6.32 eV (N = 16 over 78 eV) because the law divided
   * by 12.95. The run-time audit measured 14.93 meV against a 0.333 meV prediction
   * and hard-aborted -- the audit's first legitimate kill. Across the whole D3
   * envelope this ceiling cuts the worst under-prediction from 11.45x to 2.09x,
   * i.e. back inside the 3x safety factor the law was always meant to carry.
   *
   * ⚠ Legs with delta <= this value are BIT-IDENTICAL: min() is the identity there,
   * and every result validated at small delta (section 8.2's 1/delta growth, which
   * this does NOT touch) stands unchanged.
   */
  inline constexpr double wgrid_delta_sat_eV = 2.4;
  /** the RELATIVE-error constant of the same fit, for the audit's conversion
   *  from a measured |dW|/|W| back to a Sigma-equivalent meV (Axis D 8.1). */
  inline constexpr double wgrid_Crel = 1.1e-3;

  struct wc_grid_geom_t {
    double delta = 0.0;        ///< a.u., Im z of the target line
    double h = 0.0;            ///< a.u., grid spacing
    double zmax = 0.0;         ///< a.u., largest |Re z| the target set reaches
    long   N = 0;              ///< grid points, omega >= 0 ONLY (bosonic half)
    double target_mev = 0.0;
    double pred_mev = 0.0;     ///< the law's prediction at the resolved h
    double h_over_delta = 0.0;
    bool   clamped = false;    ///< h/delta hit the 1/2 validity bound
    bool   expert = false;     ///< h came from qp_tc_wgrid_h, not the law
    bool   delta_sat = false;  ///< delta exceeded the credit ceiling (Axis D3)
    double omega(long j) const { return double(j) * h; }
  };

  /**
   * Resolve h from the ABSOLUTE accuracy target -- the knob is the target, not
   * the spacing.
   *
   *   h/delta = min( 1/2 , ( T[meV] delta[eV] / (SAFETY K) )^(1/p) )
   *
   * @param zmax  the largest |Re z| the residue targets reach. ⚠ DERIVE THIS
   *              FROM THE STRIP WINDOW, not from E_PH: a CLAMPED state sits at
   *              z = mu where sigma_J vanishes identically and contributes no
   *              target at all, so only IN-STRIP evaluation energies count.
   */
  inline wc_grid_geom_t size_wc_grid(double target_mev, double delta_au,
                                     double zmax_au, double h_override_au = 0.0) {
    utils::check(delta_au > 0.0, "wc_grid: delta = {} must be > 0.", delta_au);
    utils::check(zmax_au >= 0.0, "wc_grid: zmax = {} must be >= 0.", zmax_au);
    wc_grid_geom_t g;
    g.delta = delta_au;
    g.zmax = zmax_au;
    g.target_mev = target_mev;
    if (h_override_au > 0.0) {
      g.h = h_override_au;
      g.expert = true;
    } else {
      utils::check(target_mev > 0.0,
                   "wc_grid: the accuracy target must be > 0 meV (got {}).", target_mev);
      // ⚠ the delta CREDIT is capped (wgrid_delta_sat_eV); the h/delta RESOLUTION
      // is not. The two roles of delta are different: h/delta is the stencil's
      // resolution of the analytic distance and is real, while 1/delta is the
      // conversion from a relative W error to an absolute Sigma error and is the
      // part that stops being earned once delta exceeds the feature spacing.
      const double d_eff = std::min(delta_au * ha_to_eV, wgrid_delta_sat_eV);
      double r = std::pow(target_mev * d_eff / (wgrid_safety * wgrid_K), 1.0 / wgrid_p);
      g.clamped = (r > wgrid_hmax_over_delta);
      g.h = std::min(r, wgrid_hmax_over_delta) * delta_au;
    }
    g.h_over_delta = g.h / delta_au;
    g.delta_sat = (delta_au * ha_to_eV > wgrid_delta_sat_eV);
    g.pred_mev = wgrid_K * std::pow(g.h_over_delta, wgrid_p)
               / std::min(delta_au * ha_to_eV, wgrid_delta_sat_eV);
    // omega in [0, (N-1) h] must cover zmax, plus ONE pad cell, plus the
    // 3-point stencil's centre-node clamp needing an interior neighbour.
    g.N = long(std::ceil(zmax_au / g.h)) + 3;
    return g;
  }

  // =========================================================================
  //  (b) THE TILE CACHE
  // =========================================================================
  struct wc_grid_t {
    wc_grid_geom_t g;
    long nq = 0, NP = 0;
    long N_last() const { return g.N - 1; }
    std::shared_ptr<sArray_t<Array_view_4D_t>> W;   ///< (nq, N, NP, NP), omega >= 0
    double refl_dev = 0.0;      ///< the bosonic reflection assert, measured
    double refl_scale = 0.0;
    /**
     * max |W^c| over the WHOLE filled grid. ⚠ THE AUDIT NORMALIZES BY THIS, not by
     * the sample's own max|W|. |W^c| varies by orders of magnitude across (q, omega)
     * -- MEASURED 40x on qe_lih222 alone -- so a per-sample ratio can read enormous
     * with no wrong value anywhere, purely because that sample sits where screening
     * is weak. What reaches Sigma is the absolute error against the scale that
     * actually contributes, which is this one.
     */
    double w_max = 0.0;
    double t_fill = 0.0;
    long   n_solve = 0;

    /**
     * W^c(q_s, z) by local 3-point Lagrange interpolation. PURE LOCAL -- no
     * communicator is reachable from here.
     *
     * BOSONIC HALVING: only omega >= 0 is stored, and Re z < 0 is served by
     *
     *      W^c(z) = W^c(-conj z)^dagger          (CONJUGATE TRANSPOSE)
     *
     * ⚠ IT IS THE DAGGER, NOT ELEMENTWISE CONJUGATION -- and the offline model
     * could not tell the difference. Derivation: the contour supplies
     * Pi(z) = -[R(z) + R(-conj z)^dag], so elementwise
     *      Pi(-conj z)_PQ = -[R(-conj z)_PQ + conj(R(z)_QP)] = Pi(z)^dag_PQ,
     * i.e. Pi(-conj z) = Pi(z)^dag -- NOT Pi(z)^*, because R is not symmetric in
     * (P,Q) at complex t (p_contour.hpp section 1: "Pi is NOT Hermitian at
     * complex t"). Pushing that through the Dyson chain, with Z Hermitian and
     * the push-through identity Z[I - Pi^dag Z]^-1 = [I - Z Pi^dag]^-1 Z,
     *      W^c(-conj z) = ([I - Z Pi^dag]^-1 - I) Z = W^c(z)^dag.
     * The offline model (Axis D 8.4a) measured plain conjugation as exact to
     * 7.9e-16 only because ITS residues g_p g_p^T are real SYMMETRIC, so the
     * transpose is a no-op there. CoQuI's are not.
     * [caught by the fill's assert on live data, 2026-08-25 -- which is why the
     *  assert lands before the read path is trusted.]
     */
    void at(long qs, ComplexType z, nda::matrix<ComplexType> &out) const {
      const double w = z.real();
      const bool refl = (w < 0.0);
      const double a = std::abs(w);
      // ⚠ OUT-OF-GRID IS AN ERROR, NOT A CLAMP. The stencil below pins the centre
      // node into [1, N-2], so a target beyond the grid would be silently
      // EXTRAPOLATED and return garbage -- which is exactly how a mis-derived span
      // would hide. The span comes from the strip window (wc_band_elements.hpp);
      // if this fires, that derivation missed part of the target set.
      utils::check(a <= g.omega(N_last()) * (1.0 + 1e-12),
                   "wc_grid::at: |Re z| = {:.6g} a.u. is OUTSIDE the grid, which spans "
                   "[0, {:.6g}] a.u. (N = {}, h = {:.6g}). The grid span is derived from "
                   "the strip window and the internal-state range; a target beyond it "
                   "means that derivation is incomplete. Refusing to extrapolate.",
                   a, g.omega(N_last()), g.N, g.h);
      const double t = a / g.h;
      long i = long(std::lrint(t));
      if (i < 1) i = 1;
      if (i > g.N - 2) i = g.N - 2;
      const double d = t - double(i);
      const double lm = 0.5 * d * (d - 1.0);
      const double l0 = 1.0 - d * d;
      const double lp = 0.5 * d * (d + 1.0);
      auto Wl = W->local();
      out.resize(NP, NP);
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q) {
          // reflected targets take the DAGGER: read (Q,P) and conjugate
          const long p = refl ? Q : P, q = refl ? P : Q;
          const ComplexType v = lm * Wl(qs, i - 1, p, q) + l0 * Wl(qs, i, p, q)
                              + lp * Wl(qs, i + 1, p, q);
          out(P, Q) = refl ? std::conj(v) : v;
        }
    }
  };

  namespace detail {
    /** Pi(z) from the contour samples: eq (SIGN) of p_contour.hpp. */
    inline void pi_at(tc::transform_factor_t const &tf,
                      sArray_t<Array_view_4D_t> const &sPi, long qs, long r, long NP,
                      ComplexType z, nda::matrix<ComplexType> &Pi,
                      nda::array<ComplexType, 1> &zz, nda::array<ComplexType, 2> &F) {
      decltype(nda::range::all) all;
      zz.resize(2);
      zz(0) = z;
      zz(1) = tc::mirror_target(z);
      tf.apply_many(zz, F);
      auto Pislab = nda::reshape(sPi.local()(qs, all, all, all),
                                 std::array<long, 2>{r, NP * NP});
      nda::array<ComplexType, 2> R(2, NP * NP);
      nda::blas::gemm(ComplexType(1.0), F(nda::range(0, 2), all), Pislab,
                      ComplexType(0.0), R);
      Pi.resize(NP, NP);
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q)
          Pi(P, Q) = -(R(0, P * NP + Q) + std::conj(R(1, Q * NP + P)));
    }
    /**
     * ⚠ THE ONE EXACT REFERENCE. Both the audit and the read-path gate call THIS --
     * divergence between two "exact" references is how a defect hides: whichever one
     * the gate uses passes, and the other ships. Anything comparing against "the
     * exact W^c" must come through here.
     */
    inline void exact_W_at(tc::transform_factor_t const &tf,
                    sArray_t<Array_view_4D_t> const &sPi,
                    sArray_t<Array_view_3D_t> const &sZ,
                    long qs, long r, long NP, ComplexType z,
                    nda::matrix<ComplexType> &Wex) {
      decltype(nda::range::all) all;
      nda::matrix<ComplexType> Pi(NP, NP);
      nda::array<ComplexType, 1> zz(2);
      nda::array<ComplexType, 2> F;
      pi_at(tf, sPi, qs, r, NP, z, Pi, zz, F);
      auto Zq = sZ.local()(qs, all, all);
      methods::wc_line::dyson_wc_line(Zq, Pi, Wex, nullptr);
    }

  } // namespace detail

  /**
   * Fill the cache. COLLECTIVE, and deliberately so: the (iq, j) flat index is
   * partitioned, every rank does only local solves, ONE reduction assembles it.
   * Call at the lockstep point next to `p_contour::gather_Z_tiles`.
   */
  template<typename thc_t>
  std::shared_ptr<wc_grid_t> fill_wc_grid(thc_t &thc,
                                          sArray_t<Array_view_3D_t> const &sZ,
                                          sArray_t<Array_view_4D_t> const &sPi,
                                          tc::transform_factor_t const &tf,
                                          wc_grid_geom_t const &geom,
                                          long rank_contour, int lvl) {
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    const long NP = thc.Np();
    const long nq = thc.nqpts_ibz();
    auto G = std::make_shared<wc_grid_t>();
    G->g = geom;
    G->nq = nq;
    G->NP = NP;
    G->W = std::make_shared<sArray_t<Array_view_4D_t>>(
        math::shm::make_shared_array<Array_view_4D_t>(*mpi, {nq, geom.N, NP, NP}));
    G->W->set_zero();
    G->W->win().fence();
    mpi->comm.barrier();

    const auto t0 = std::chrono::steady_clock::now();
    const long ntot = nq * geom.N;
    auto [k0, k1] = itertools::chunk_range(0, ntot, mpi->comm.size(), mpi->comm.rank());
    nda::matrix<ComplexType> Pi(NP, NP), Wm(NP, NP);
    nda::array<ComplexType, 1> zz(2);
    nda::array<ComplexType, 2> F;
    for (long k = k0; k < k1; ++k) {
      const long iq = k / geom.N, j = k % geom.N;
      const ComplexType z(geom.omega(j), geom.delta);
      detail::pi_at(tf, sPi, iq, rank_contour, NP, z, Pi, zz, F);
      auto Zq = sZ.local()(iq, all, all);
      methods::wc_line::dyson_wc_line(Zq, Pi, Wm, nullptr);
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q) G->W->local()(iq, j, P, Q) = Wm(P, Q);
    }
    G->n_solve = k1 - k0;
    G->W->win().fence();
    G->W->all_reduce();
    G->W->win().fence();
    mpi->comm.barrier();
    G->t_fill = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    {   // the GLOBAL |W| scale the audit normalizes by
      double m = 0.0;
      auto Wl = G->W->local();
      for (long iq = 0; iq < nq; ++iq)
        for (long j = 0; j < geom.N; ++j)
          for (long P = 0; P < NP; ++P)
            for (long Q = 0; Q < NP; ++Q) m = std::max(m, std::abs(Wl(iq, j, P, Q)));
      G->w_max = m;
    }

    // ---- ⚠ THE BOSONIC REFLECTION ASSERT ---------------------------------
    // W^c(z) = conj(W^c(-conj z)) is ELEMENTWISE only for real-symmetric
    // residues. Check it on the LIVE data: one extra solve at a NEGATIVE
    // omega, compared against the conjugate of the stored positive one. A
    // wrong reflection is a silent conjugate error on half the targets, so
    // this lands BEFORE the read path is trusted.
    {
      const long jt = std::min<long>(2, geom.N - 1);
      const ComplexType zneg(-geom.omega(jt), geom.delta);
      detail::pi_at(tf, sPi, 0, rank_contour, NP, zneg, Pi, zz, F);
      auto Z0 = sZ.local()(0, all, all);
      methods::wc_line::dyson_wc_line(Z0, Pi, Wm, nullptr);
      double dev = 0.0, sc = 0.0;
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q) {
          const ComplexType ref = std::conj(G->W->local()(0, jt, Q, P));   // DAGGER
          dev = std::max(dev, std::abs(Wm(P, Q) - ref));
          sc = std::max(sc, std::abs(Wm(P, Q)));
        }
      G->refl_dev = dev;
      G->refl_scale = sc;
      utils::check(sc <= 0.0 or dev / sc < 1e-8,
                   "wc_grid: THE BOSONIC REFLECTION IDENTITY FAILS. "
                   "W^c(z) vs W^c(-conj z)^dagger deviates by {:.3e} over a scale of "
                   "{:.3e} ({:.3e} relative) at q = 0, omega = {:.6g} a.u. The cache "
                   "stores only omega >= 0 and serves the other half by the DAGGER "
                   "(derived in wc_grid_t::at from Pi(-conj z) = Pi(z)^dag). If this "
                   "fires, that derivation does not hold for this W^c -- half the "
                   "targets would silently carry a transpose/conjugate error, so the "
                   "read path must not be trusted until it is understood.",
                   dev, sc, (sc > 0.0 ? dev / sc : 0.0), geom.omega(jt));
    }

    app_log(lvl, "  - TC-5 W^c GRID:               target {:.3g} meV -> h/delta = {:.4f}{}, "
                 "h = {:.6g} a.u. ({:.4g} eV), N = {} points (omega >= 0; bosonic half), "
                 "nq = {}; PREDICTED error {:.4g} meV; interpolant = local quadratic, "
                 "pad 1; fill {} solves in {:.2f} s; reflection identity {:.2e} rel",
            geom.target_mev, geom.h_over_delta,
            geom.expert ? " (EXPERT h override)" : (geom.clamped ? " (CLAMPED at 1/2)" : ""),
            geom.h, geom.h * ha_to_eV, geom.N, nq, geom.pred_mev, ntot, G->t_fill,
            (G->refl_scale > 0.0 ? G->refl_dev / G->refl_scale : 0.0));
    return G;
  }

  // =========================================================================
  //  (d) THE RUN-TIME AUDIT
  // =========================================================================
  /**
   * ⚠ WHY THIS EXISTS. Axis D 8.7 measured K/|Sigma| spanning 1.3 to 417: on a
   * spectrum whose spectral weight is CONCENTRATED on one in-range pole a fixed
   * constant is wrong by 75x, and nothing in the sizing inputs reveals it. The
   * law sizes the grid; this proves it.
   *
   * FAILURE BEHAVIOUR (defined, not discretionary):
   *   * ALWAYS log predicted vs measured and the worst (q, Re z);
   *   * measured <= 10 x target : WARN and continue;
   *   * measured >  10 x target : HARD ABORT unless `hard` is false -- an
   *     order-of-magnitude breach is the concentrated-spectrum case and every
   *     downstream number is untrustworthy.
   */
  struct wgrid_audit_t {
    long   n_sample = 0;          ///< GLOBAL count (the caller reduces it)
    double dW_abs = 0.0;          ///< measured max |dW| -- the raw quantity
    double w_local = 0.0;         ///< max|W| AT the worst sample (diagnostic only)
    double dW_rel = 0.0;          ///< dW_abs / grid-global max|W|  <- THE measure
    double meas_mev = 0.0;        ///< Sigma-equivalent, via the Axis-D relation
    double pred_mev = 0.0;
    long   worst_q = -1;
    double worst_z = 0.0;
    bool   breached = false;
  };

  /**
   * @param zs  sampled residue-target Re z values (a.u.) -- draw them from the
   *            ACTUAL target set, not from the grid span.
   * @param qs  the transfer of each sample.
   */
  template<typename thc_t>
  wgrid_audit_t audit_wc_grid(wc_grid_t const &G, thc_t &thc,
                              sArray_t<Array_view_3D_t> const &sZ,
                              sArray_t<Array_view_4D_t> const &sPi,
                              tc::transform_factor_t const &tf, long rank_contour,
                              std::vector<double> const &zs,
                              std::vector<long> const &qs) {
    decltype(nda::range::all) all;
    wgrid_audit_t A;
    A.pred_mev = G.g.pred_mev;
    A.n_sample = long(zs.size());
    if (zs.empty()) return A;
    const long NP = G.NP;
    nda::matrix<ComplexType> Wex(NP, NP), Wgot(NP, NP);
    for (std::size_t s = 0; s < zs.size(); ++s) {
      const long q = qs[s];
      const ComplexType z(zs[s], G.g.delta);
      // EXACT: THE shared reference (detail::exact_W_at) -- the same call the
      // read-path gate makes, so the two cannot disagree about what "exact" means
      detail::exact_W_at(tf, sPi, sZ, q, rank_contour, NP, z, Wex);
      G.at(q, z, Wgot);
      double dev = 0.0, sc = 0.0;
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q) {
          dev = std::max(dev, std::abs(Wgot(P, Q) - Wex(P, Q)));
          sc = std::max(sc, std::abs(Wex(P, Q)));
        }
      // ⚠ normalize by the GRID-GLOBAL scale, not this sample's own max|W|:
      // |W| varies by orders of magnitude across (q, omega), so a per-sample ratio
      // reports a huge number wherever screening happens to be weak, with nothing
      // wrong. `sc` is kept for the log so the two are distinguishable next time.
      if (dev > A.dW_abs) {
        A.dW_abs = dev;
        A.w_local = sc;
        A.worst_q = q;
        A.worst_z = zs[s];
      }
    }
    A.dW_rel = (G.w_max > 0.0) ? A.dW_abs / G.w_max : 0.0;
    return A;
  }

  /**
   * The VERDICT, applied AFTER the caller has reduced `dW_rel` across ranks.
   * ⚠ The split is not cosmetic: `utils::check` aborts, and an abort on ONE rank
   * while the others proceed is a hang. Every rank must reach the same verdict
   * from the same reduced number, so the measurement is local and collective-free
   * and the decision is taken on an already-agreed value.
   */
  inline void report_wc_grid_audit(wgrid_audit_t &A, wc_grid_geom_t const &geom,
                                   double w_max_global, bool hard, int lvl) {
    A.pred_mev = geom.pred_mev;
    // recomputed from the REDUCED absolute error against the global scale
    A.dW_rel = (w_max_global > 0.0) ? A.dW_abs / w_max_global : 0.0;
    // Convert the measured RELATIVE W error to a Sigma-equivalent meV by the
    // same fit family that sized the grid: the predicted relative error at this
    // h/delta is Crel (h/delta)^p, so the measured/predicted RATIO carries over.
    const double pred_rel = wgrid_Crel * std::pow(geom.h_over_delta, wgrid_p);
    A.meas_mev = (pred_rel > 0.0) ? A.pred_mev * (A.dW_rel / pred_rel) : 0.0;
    A.breached = (A.meas_mev > geom.target_mev);
    auto const &G = geom;

    app_log(lvl, "  - TC-5 GRID AUDIT:             {} samples: max |dW| = {:.3e} ABS; "
                 "/ grid-global max|W| = {:.3e} -> {:.3e} rel (predicted {:.3e}) -> "
                 "Sigma-equivalent {:.4g} meV against a target of {:.4g} meV and a law "
                 "prediction of {:.4g} meV. Worst at q = {}, Re z = {:+.6g} a.u., where "
                 "the LOCAL max|W| = {:.3e} (local/global = {:.2e}; a small ratio there "
                 "means a weakly-screened sample, NOT a wrong value).",
            A.n_sample, A.dW_abs, w_max_global, A.dW_rel, pred_rel, A.meas_mev,
            G.target_mev, A.pred_mev, A.worst_q, A.worst_z, A.w_local,
            (w_max_global > 0.0 ? A.w_local / w_max_global : 0.0));

    if (A.meas_mev > 10.0 * G.target_mev) {
      utils::check(not hard,
                   "wc_grid AUDIT: the measured residue-tier error is {:.4g} meV, MORE "
                   "THAN 10x the requested qp_tc_wgrid_mev = {:.4g} meV (law predicted "
                   "{:.4g} meV; worst at q = {}, Re z = {:+.6g} a.u., |dW|/|W| = {:.3e}). "
                   "An order-of-magnitude breach is the CONCENTRATED-SPECTRUM case of "
                   "notes/tilted_contour_validation_results.md 8.7 -- the sizing law's "
                   "constant does not describe this system and every downstream number is "
                   "untrustworthy. Lower qp_tc_wgrid_mev (h shrinks as target^(1/{:.2f})), "
                   "or set qp_tc_wgrid_h directly, or set qp_tc_wgrid_audit_hard = false "
                   "to push through DELIBERATELY on a diagnostic run.",
                   A.meas_mev, G.target_mev, A.pred_mev, A.worst_q, A.worst_z,
                   A.dW_rel, wgrid_p);
      app_warning("wc_grid AUDIT: measured {:.4g} meV vs target {:.4g} meV (>10x) -- "
                  "CONTINUING ONLY because qp_tc_wgrid_audit_hard = false.",
                  A.meas_mev, G.target_mev);
    } else if (A.breached) {
      app_warning("wc_grid AUDIT: the measured residue-tier error is {:.4g} meV against "
                  "the requested qp_tc_wgrid_mev = {:.4g} meV (law predicted {:.4g}). "
                  "Within 10x, so the run continues, but the sizing law is optimistic for "
                  "this spectrum -- see notes/tilted_contour_validation_results.md 8.7. "
                  "Lower qp_tc_wgrid_mev if the residue tier carries the physics.",
                  A.meas_mev, G.target_mev, A.pred_mev);
    }
  }

  /**
   * ⚠⚠ THE COLLECTIVE-CARRYING AUDIT DRIVER. EVERY RANK MUST CALL THIS whenever
   * the audit is enabled -- the guard at the call site may test ONLY quantities
   * that are uniform across ranks (`wgrid_audit > 0`), never rank-local ownership.
   *
   * WHY IT IS A SEPARATE FUNCTION. The first version of this code inlined the
   * reduces under `ctx.bstore.size() > 0`, i.e. under "does THIS rank own a
   * block". On the si444 _w leg (nblk = 13, 60 ranks) 47 ranks never entered,
   * the reduces did not pair up, and the run hard-aborted on a garbage sample
   * count (-4.4e18) and a garbage |dW| (9.0e+02 against a grid whose global
   * max|W| is 2.5e-03 -- arithmetically impossible for bounded Lagrange weights,
   * which is what identified the reduce rather than the physics). Owning no
   * block is NORMAL and must cost nothing but an empty sample list: `sample`
   * simply appends nothing, and this function still reduces.
   *
   * @param sample  called once, collective-free, to append this rank's local
   *                samples; it may append none.
   */
  template<typename comm_t, typename thc_t, typename sampler_t>
  wgrid_audit_t run_wc_grid_audit(comm_t &comm, wc_grid_t const &G, thc_t &thc,
                                  sArray_t<Array_view_3D_t> const &sZ,
                                  sArray_t<Array_view_4D_t> const &sPi,
                                  tc::transform_factor_t const &tf, long rank_contour,
                                  long nsamp_per_q, long nq_ibz,
                                  wc_grid_geom_t const &geom, bool hard, int lvl,
                                  sampler_t &&sample) {
    std::vector<double> zs;
    std::vector<long> qs;
    const long ntot = nsamp_per_q * nq_ibz;
    auto [b0, b1] = itertools::chunk_range(0, ntot, comm.size(), comm.rank());
    sample(b0, b1, zs, qs);                       // collective-free, may add nothing
    utils::check(zs.size() == qs.size(),
                 "run_wc_grid_audit: sampler returned {} z but {} q", zs.size(),
                 qs.size());
    utils::check(long(zs.size()) <= b1 - b0,
                 "run_wc_grid_audit: sampler returned {} samples for a slice of {} -- "
                 "the global count would exceed its own bound", zs.size(), b1 - b0);

    wgrid_audit_t A;
    A.pred_mev = geom.pred_mev;
    if (not zs.empty())
      A = audit_wc_grid(G, thc, sZ, sPi, tf, rank_contour, zs, qs);

    // ---- from here to the verdict: COLLECTIVE, entered by every rank
    A.n_sample = comm.all_reduce_value(long(zs.size()), std::plus<>{});
    const double dloc = A.dW_abs;
    A.dW_abs = comm.all_reduce_value(A.dW_abs, boost::mpi3::max<>{});
    // ⚠ THE LOCATION MUST COME FROM THE RANK THAT HOLDS THE MAX. Reducing w_local by
    // an independent max, and printing worst_q/worst_z from whichever rank happens to
    // log, attributes the worst error to a sample that is not the worst one -- the
    // leg-1 banner did exactly that ("worst at q = 3 ... LOCAL max|W| = 4.184e+01")
    // and the plausible-looking location sent the investigation after the physics.
    // Elect the owner, then broadcast ITS triple.
    long owner = (dloc == A.dW_abs) ? long(comm.rank()) : long(comm.size());
    owner = comm.all_reduce_value(owner, boost::mpi3::min<>{});
    if (owner >= long(comm.size())) owner = 0;          // no samples anywhere
    double loc[3] = {double(A.worst_q), A.worst_z, A.w_local};
    comm.broadcast_n(loc, 3, int(owner));
    A.worst_q = long(std::lrint(loc[0]));
    A.worst_z = loc[1];
    A.w_local = loc[2];
    // ⚠ THE BOUNDS CHECK LANDS BEFORE THE VERDICT. A count outside [0, ntot] proves
    // the reduce itself is invalid, and then the |dW| beside it is invalid too --
    // reporting a physics breach from it (as the leg-1 abort did) points the
    // investigation at the wrong subsystem entirely.
    utils::check(A.n_sample >= 0 and A.n_sample <= ntot,
                 "wc_grid AUDIT: reduced sample count {} is outside its own bound "
                 "[0, {}] (= qp_tc_wgrid_audit x nq_ibz). The reduce did not pair up "
                 "across ranks -- some rank did not enter run_wc_grid_audit. Every "
                 "number on the audit line is invalid; do not read it as physics.",
                 A.n_sample, ntot);
    report_wc_grid_audit(A, geom, G.w_max, hard, lvl);
    return A;
  }

} // namespace wc_grid
} // namespace methods

#endif // COQUI_WC_GRID_HPP
