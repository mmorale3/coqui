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

#ifndef COQUI_WC_SPECTRAL_HPP
#define COQUI_WC_SPECTRAL_HPP

/**
 * Project 2 (qpGW+BSE+EDMFT) increment RW-2 -- the SPECTRAL-QUADRATURE pole representation
 * of W^c for the mode-A quasiparticle map (spec notes/rw_real_axis_w_spec.md, RW-2 rev 3).
 *
 * WHY THIS EXISTS. The production mode-A W^c pole representation is a support-constrained
 * least-squares fit on the DLR auxiliary node grid (wc_band_elements.hpp stage 1, knob
 * qp_modea_wfit = tau|nu). Measured on the SVO metal (notes/qpgw_metal_mode_m0.md section 8b),
 * that representation makes the contracted Sigma^c pole sum a CANCELLATION CATASTROPHE: the
 * residue-weighted absolute sum is Sabs = 1.1-1.6e3 a.u. while |Sigma^c| = 5e-3 - 2e-1 a.u.,
 * i.e. 4 to 5.5 decimal digits of cancellation are required and the fitted residues -- which
 * are MIXED SIGN because least squares has no sign constraint -- do not deliver them. It is
 * not a low-omega Bose pile-up (refuted, section 8a) and not an evaluation-point artefact
 * (378x off with no pole within 855 meV, section 8b point 3).
 *
 * This file implements the representation change: the pole set becomes a QUADRATURE of the
 * COMPUTED Im W^c(Omega, q) from the RW-1 real-axis module. Node positions come from the
 * computed spectral support, weights from its values, so the residue matrices are Hermitian
 * and SIGN-DEFINITE per sign(Omega) by construction, and the cancellation disappears
 * structurally rather than by tuning.
 *
 * =====================================================================================
 * THE SIGN CONVENTION, RE-DERIVED MECHANICALLY
 * [The RW-2 spec's residue line is explicitly marked "do not trust"; this is the derivation
 *  that replaces it, and gate RW-2-a(i) pins the result on an analytic two-pole Drude model
 *  where every quantity has a closed form. test_wc_spectral.cpp "wc_spectral_drude_sign".]
 * =====================================================================================
 *
 * (1) THE RETARDED CONVENTION. W^c,R(z) is analytic in the upper half plane and decays at
 *     infinity, so Cauchy on the real axis gives the standard Hilbert representation
 *
 *         W^c(z) = (1/pi) int_{-inf}^{+inf} dOmega'  Im W^c(Omega') / (Omega' - z).       (A)
 *
 *     This is the convention of the ported real-axis module and of the RW-1 gate harness
 *     (test_real_axis_w_lehmann.cpp, "SIGN OF THE FORWARD MAP"), where it was VERIFIED
 *     numerically against the production Matsubara W^c: with (A) the Pi-level least-squares
 *     ratio comes out +1.0000, and the Matsubara W^c(i nu = 0) comes out NEGATIVE, which is
 *     the physical statement that screening lowers W below the bare v. The opposite sign
 *     would give +1/2 -> the RW-1 report's deviation D-6 documents that trap.
 *
 * (2) THE ODD EXTENSION -- AND ITS TRANSPOSE. The module stores Omega >= 0 only
 *     (real_axis_mb_state_t: "bosonic Omega -> -Omega symmetry is exact at any
 *     temperature"). What that symmetry actually says for a MATRIX response is set by the
 *     Hermiticity of W^c(t) in real time, W^c_PQ(t) = conj(W^c_QP(t)):
 *
 *         W^c_PQ(-Omega) = conj( W^c_QP(Omega) )   =>   Im W^c_PQ(-Omega) = -Im W^c_QP(Omega)
 *
 *     -- a TRANSPOSE, not a plain sign flip. The two coincide only when Im W^c is symmetric
 *     in (P,Q), which is NOT generic: measured max|ImW_PQ - ImW_QP| / max|ImW| is 1.1e-13 on
 *     qe_lih222 (a fixture whose collocation makes it symmetric) and 5.3e-01 on the SVO
 *     production case. The RW-1 gate harness flagged exactly this caveat for Im Pi.
 *     Folding the negative half of (A) onto the positive one therefore gives
 *
 *         W^c_PQ(z) = (1/pi) int_0^inf dOmega [ Im W^c_PQ(Omega)/(Omega - z)
 *                                             + Im W^c_QP(Omega)/(Omega + z) ].
 *
 * (3) THE QUADRATURE. Replace the integral by the grid's OWN trapezoid weights w_j at its
 *     nodes Omega_j > 0 (the grids are non-uniform; never resample -- RW-1 hand-off item).
 *     Rewriting each factor as a pole in z,
 *
 *         1/(Omega_j - z) = -1 / (z - Omega_j),      1/(Omega_j + z) = +1 / (z - (-Omega_j)),
 *
 *     so
 *                          [ -(1/pi) w_j Im W^c(Omega_j) ]     [ +(1/pi) w_j Im W^c(Omega_j)^T ]
 *         W^c(z) ~= sum_j  ------------------------------  +  ---------------------------------
 *                                 (z - Omega_j)                        (z - (-Omega_j))
 *
 *     THE RESULT: the pole pair is (+Omega_j, -Omega_j) with residue matrices
 *
 *         A_j := -(1/pi) * w_j * Im W^c(Omega_j)      at  omega_p = +Omega_j,
 *        -A_j^T  (the TRANSPOSE, from (2))            at  omega_p = -Omega_j.               (B)
 *
 *     [The transpose is what gate RW-2-a(i)'s "asymmetric" case pins. Dropping it -- using
 *      -A_j -- is a 100 %-class error the moment Im W^c stops being (P,Q)-symmetric, and it
 *      also breaks Hermiticity of the reconstructed W^c(i nu), which the tau anchor and the
 *      trev rule both rely on.]
 *
 * (4) DEFINITENESS. The matrix-valued spectral function of the bosonic response is
 *     S(Omega) = -(1/pi) (W^R - W^R dagger)/(2i), Hermitian and positive semi-definite for
 *     Omega > 0. Writing W = X + iY elementwise (X, Y real),
 *
 *         (W - W^dag)_PQ/(2i) = (Y_PQ + Y_QP)/2 - i (X_PQ - X_QP)/2,
 *
 *     i.e. the SYMMETRIC part of Im W is the REAL part of the spectral matrix, and the
 *     antisymmetric part of Re W supplies its imaginary part. (When Im W happens to be
 *     symmetric and Re W symmetric -- the qe_lih222 case -- this collapses to
 *     "spectral matrix = Im W".)
 *
 *     What survives in general is exactly what the contraction needs. Write
 *     A_j = S_j + K_j with S_j = (A_j + A_j^T)/2 real symmetric and K_j real antisymmetric.
 *     For a PSD Hermitian H = R + iI (R real symmetric, I real antisymmetric) and any real x,
 *     x^T H x = x^T R x >= 0, so R is PSD; hence
 *
 *         Im W^c(Omega_j) has a negative semi-definite SYMMETRIC part for Omega_j > 0
 *         =>  S_j = -(1/pi) w_j [sym part of Im W^c(Omega_j)] is POSITIVE semi-definite,
 *
 *     and for ANY complex v, v^dag K_j v is purely imaginary (v = x + iy gives
 *     x^T K x + y^T K y + 2i x^T K y = 2i x^T K y), so
 *
 *         Re[ v^dag A_j v ] = v^dag S_j v >= 0.                                            (C)
 *
 *     That is the statement section (5) actually uses.
 *
 *     Consistency check on the static value: W^c(i nu = 0) = sum_j [ A_j/(-Omega_j)
 *     + (-A_j^T)/(+Omega_j) ] = -sum_j (A_j + A_j^T)/Omega_j = -2 sum_j S_j/Omega_j,
 *     negative semi-definite AND Hermitian. Screening lowers W.
 *     Consistency check against the RW-1 forward map: at z = i nu, (B) sums to
 *     -sum_j [ Omega_j (A_j + A_j^T) + i nu (A_j - A_j^T) ] / (Omega_j^2 + nu^2), which is
 *     Hermitian, and reduces term by term to the RW-1 gate's
 *     (2/pi) sum_j w_j Omega_j Im W^c(Omega_j)/(Omega_j^2 + nu^2) exactly when Im W^c is
 *     symmetric -- which is the regime RW-1 measured. The production meter (the Lehmann
 *     forward map of the quadrature rep at the bosonic Matsubara nodes) is the readout that
 *     covers the general case.
 *
 * (5) WHY THE CANCELLATION GOES AWAY. The mode-A closure (qp_modea.hpp) is
 *
 *         Sigma^c_ab(z) = sum_{J,p} M^{(J,p)}_ab [ n_B(om_p) + f(eps_J) ] / (z - (eps_J - om_p))
 *
 *     with M^{(J,p)}_ab = [B^T W^{(p)} conj(B)]_ab, so the DIAGONAL is M_ii = v^dag W^{(p)} v
 *     with v_P = conj(B(P,i)). With (B) and (C) (note Re[v^dag A^T v] = Re[v^dag A v] because
 *     A^T and A share their symmetric part):
 *         om_p > 0 : W^{(p)} = A_j    => Re M_ii >= 0 ; n_B >= 0, f in [0,1] => factor >= 0
 *         om_p < 0 : W^{(p)} = -A_j^T => Re M_ii <= 0 ; n_B(om<0) in (-1,0), so
 *                                                        n_B + f ~ f - 1 <= 0 => factor <= 0
 *     In BOTH cases the numerator N_{Jp} = Re M_ii (n_B + f) is >= 0. The diagonal Sigma^c is
 *     therefore a POSITIVE-WEIGHT Lehmann sum, whose only sign structure is the sign of the
 *     denominator -- an O(1) principal-value balance, not a 4-5.5 digit cancellation between
 *     residues of opposite sign. This is the entire content of RW-2. Gate RW-2-a(ii) measures
 *     it on a metallic toy, RW-2-c(i) on the SVO metal.
 *
 * =====================================================================================
 * COARSENING (the "optional node coarsening knob" of the spec)
 * =====================================================================================
 * The RW-1 accuracy rules put N_Omega ~ Omega_max/eta nodes on the grid (hundreds), and the
 * downstream cost of the map is LINEAR in the pole count npk in two places that dominate
 * memory: the residue slabs (nq, npk, Np, Np) and the per-(s,k) sandwich M(a,b,J*npk+p). On
 * the SVO production case those are 0.21 GB and 0.22 GB per unit of npk respectively, so a
 * 1600-pole raw quadrature is not runnable while the LS path's npk = 62 is. The nodes are
 * therefore MERGED into bins:
 *
 *     R_B = sum_{j in B} A_j          (its symmetric part is exactly psd: a sum of psd terms)
 *     Omega_B = ( sum_{j in B} t_j ) / ( sum_{j in B} t_j / Omega_j ),   t_j = tr A_j >= 0
 *
 * i.e. the zeroth moment sum_j A_j is preserved EXACTLY and the pole position is the
 * trace-weighted HARMONIC mean, which is the choice that makes the bin's contribution to the
 * static value tr W^c(0) exact (W^c(0) weights each pole by 1/Omega). Bin edges are placed at
 * equal cumulative u_j = t_j/Omega_j, i.e. equal shares of that same static weight -- on a
 * metal this automatically concentrates the nodes at low Omega, where the Drude-like weight
 * is, without any prior information. Sign-definiteness is untouched by the merge, and the
 * accuracy cost is MEASURED by the production Lehmann meter, never assumed.
 *
 * =====================================================================================
 * WHAT IS *NOT* IN THIS FILE
 * =====================================================================================
 * The q -> 0 head. Per the spec, eps_inv_head handling stays the production Matsubara path;
 * the spectral quadrature replaces the PQ BODY representation only. The head is carried on
 * an APPENDED pole set produced by the existing masked_pole_fit machinery -- see
 * wc_band_elements.hpp, "SPECTRAL PATH: THE HEAD SECTOR".
 */

#include <cmath>
#include <algorithm>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace wc_spectral {

  /** 1/pi, the quadrature prefactor of eq. (B). */
  inline constexpr double inv_pi = 0.31830988618379067154;

  /**
   * The quadrature node set actually used, built from the real-axis grid.
   *
   * WHY IT IS NOT JUST THE GRID. real_freq_grid_t excludes Omega = 0 by construction (the
   * Bose function diverges there) and its trapezoid weights therefore cover [Omega_0,
   * Omega_{N-1}] only: the segment [0, Omega_0] is simply absent. On an insulator that is a
   * sub-gap region and contributes nothing; on a METAL there is Drude-like weight all the way
   * down, dOmega ~ eta ~ 0.34 eV, and the segment carries real screening -- exactly the
   * region RW-2 exists to represent.
   *
   * It cannot be repaired by re-weighting node 0. The pole representation evaluates
   * sum_j A_j [1/(z - Omega_j) + ...] with A_j fixed, so the weight would have to depend on
   * z: at z = i nu with nu >> Omega_0 the kernel is flat over the segment and the trapezoid
   * wants an extra 0.5 Omega_0, while at nu = 0 the kernel is 1/Omega and it wants Omega_0.
   * Using either leaves an O(dOmega) error at the STATIC node -- measured on the RW-2-a toy:
   * 2.2e-2 and 1.1e-2 relative respectively, converging only linearly.
   *
   * The fix is a VIRTUAL NODE. Im W^c is odd and analytic near 0, so Im W^c(Omega) ~ c Omega
   * on [0, Omega_0] with c = Im W^c(Omega_0)/Omega_0. That segment's contribution to W^c(z)
   * is reproduced by ONE pole pair, and the pole position is fixed twice over:
   *   - matching the segment's zeroth moment  int_0^{Omega_0} Im W^c = c Omega_0^2 / 2
   *     gives residue A* = -(1/pi) (Omega_0/2) Im W^c(Omega_0);
   *   - matching its exact static value (1/pi) 2 int_0^{Omega_0} Im W^c/Omega
   *     = (2/pi) Im W^c(Omega_0) gives A(star) / Omega(star) = -(1/pi) Im W^c(Omega_0);
   * and both are satisfied by Omega* = Omega_0/2 with that A*. So the node list gains one
   * entry at Omega_0/2 whose weight is Omega_0/2 and whose Im W^c value is READ FROM node 0
   * (hence the `src` index map: two quadrature nodes share one data node). Measured on the
   * same toy: 6.3e-4 at the coarsest grid, converging as dOmega^2.
   *
   * [DEVIATION from the RW-1 gate, which used grid.Omega_weights() verbatim on a gapped
   *  fixture where the segment is empty. That gate is not routed through this helper.]
   */
  struct quad_grid_t {
    nda::array<double, 1> Om;    // (N+1) quadrature nodes, ascending, all > 0
    nda::array<double, 1> Ow;    // (N+1) weights
    nda::array<long, 1>   src;   // (N+1) index into the ORIGINAL Im W^c Omega axis
  };

  inline quad_grid_t make_quad_grid(nda::array<double, 1> const &Om_g,
                                    nda::array<double, 1> const &Ow_g) {
    const long N = Om_g.shape(0);
    utils::check(N >= 2, "wc_spectral::make_quad_grid: need at least 2 Omega nodes, got {}.", N);
    utils::check(Ow_g.shape(0) == N,
                 "wc_spectral::make_quad_grid: {} nodes, {} weights.", N, Ow_g.shape(0));
    utils::check(Om_g(0) > 0.0,
                 "wc_spectral::make_quad_grid: the bosonic grid must exclude Omega = 0 "
                 "(first node = {}).", Om_g(0));
    quad_grid_t g;
    g.Om  = nda::array<double, 1>(N + 1);
    g.Ow  = nda::array<double, 1>(N + 1);
    g.src = nda::array<long, 1>(N + 1);
    g.Om(0) = 0.5 * Om_g(0);
    g.Ow(0) = 0.5 * Om_g(0);
    g.src(0) = 0;
    for (long j = 0; j < N; ++j) {
      g.Om(j + 1) = Om_g(j);
      g.Ow(j + 1) = Ow_g(j);
      g.src(j + 1) = j;
    }
    return g;
  }

  /**
   * A coarsening plan: which contiguous runs of Omega nodes are merged, and the pole energy
   * each run collapses onto. `lo(b) .. hi(b)-1` index the ORIGINAL Omega grid.
   */
  struct bin_plan_t {
    nda::array<long, 1>   lo, hi;      // (nbin) node ranges into the Omega grid
    nda::array<double, 1> om_c;        // (nbin) merged pole energies, > 0
    long   nbin = 0;
    double width_worst = 0.0;          // worst (Omega_hi - Omega_lo)/Omega_c over the bins
    long   n_neg = 0;                  // nodes whose trace weight came out negative
    double neg_frac = 0.0;             // |sum of those| / sum |t_j|  -- a noise meter
  };

  /**
   * Build the coarsening plan.
   *
   * @param Om      (N) quadrature nodes, strictly increasing, all > 0.
   * @param t       (N) trace weights t_j = tr A_j = -(1/pi) w_j sum_P Im W_PP(Omega_j),
   *                summed over q. Expected >= 0; negatives are clipped and counted.
   * @param ntarget requested number of bins. <= 0 or >= N means "no coarsening": one bin per
   *                node, om_c = Om (bit-exact identity).
   *
   * Bins carry equal cumulative u_j = t_j / Omega_j (the static-W^c weight). A bin that ends
   * up with zero total weight falls back to the arithmetic centre of its node range so the
   * pole energy is always well defined.
   */
  /**
   * Relative share of the bin budget given to the STATIC-WEIGHT criterion; the rest goes to
   * the LOG-WIDTH criterion. Both are needed: weight alone puts every bin inside the two or
   * three sharp features of Im W^c and leaves single bins spanning a factor 50 in Omega
   * (measured on the RW-2-a toy: worst relative bin width 47.6, forward-map deviation 2e-2),
   * while log-width alone ignores where the screening actually lives. [agent-chosen, FLAGGED;
   * the production Lehmann meter measures the consequence directly.]
   */
  inline constexpr double bin_weight_share = 0.5;

  inline bin_plan_t build_bins(nda::array<double, 1> const &Om,
                               nda::array<double, 1> const &t,
                               long ntarget) {
    const long N = Om.shape(0);
    utils::check(N > 0, "wc_spectral::build_bins: empty Omega grid.");
    utils::check(t.shape(0) == N, "wc_spectral::build_bins: {} weights for {} nodes.",
                 t.shape(0), N);
    for (long j = 0; j < N; ++j)
      utils::check(Om(j) > 0.0, "wc_spectral::build_bins: node {} has Omega = {} <= 0; the "
                                "bosonic grid must exclude Omega = 0.", j, Om(j));

    bin_plan_t P;
    // clip the trace weights and measure how much was clipped (a pure diagnostic: a healthy
    // Im W^c is negative semi-definite for Omega > 0, so t_j >= 0 up to quadrature noise).
    nda::array<double, 1> tc(N), u(N);
    double neg = 0.0, tot = 0.0;
    for (long j = 0; j < N; ++j) {
      if (t(j) < 0.0) { ++P.n_neg; neg += -t(j); }
      tc(j) = std::max(0.0, t(j));
      tot += std::abs(t(j));
      u(j) = tc(j) / Om(j);
    }
    P.neg_frac = (tot > 0.0) ? neg / tot : 0.0;

    std::vector<long> edge;
    if (ntarget <= 0 or ntarget >= N) {
      P.nbin = N;
      P.lo = nda::array<long, 1>(N);
      P.hi = nda::array<long, 1>(N);
      P.om_c = nda::array<double, 1>(N);
      for (long j = 0; j < N; ++j) { P.lo(j) = j; P.hi(j) = j + 1; P.om_c(j) = Om(j); }
      P.width_worst = 0.0;
      return P;
    }

    // ---- the bin-edge density -------------------------------------------------------
    nda::array<double, 1> c(N);
    {
      double su = 0.0, sv = 0.0;
      nda::array<double, 1> v(N);
      for (long j = 0; j < N; ++j) {
        v(j) = (j + 1 < N) ? std::log(Om(j + 1) / Om(j)) : 0.0;
        su += u(j);
      }
      if (N > 1) v(N - 1) = v(N - 2);
      for (long j = 0; j < N; ++j) sv += v(j);
      for (long j = 0; j < N; ++j)
        c(j) = (su > 0.0 ? bin_weight_share * u(j) / su : 0.0)
             + (sv > 0.0 ? (1.0 - bin_weight_share) * v(j) / sv : 0.0);
    }
    nda::array<double, 1> cc(N);
    { double a = 0.0; for (long j = 0; j < N; ++j) { a += c(j); cc(j) = a; } }

    // ---- exactly ntarget bins: clamped inverse CDF ------------------------------------
    // Every bin holds at least one node and the last bin cannot be starved, so the count is
    // EXACT -- a plain quantile split degenerates to far fewer bins whenever the density is
    // concentrated, which is the generic case for Im W^c.
    edge.push_back(0);
    const double step = (cc(N - 1) > 0.0) ? cc(N - 1) / double(ntarget) : 0.0;
    for (long b = 1; b < ntarget; ++b) {
      long idx = 0;
      if (step > 0.0) {
        const double target = double(b) * step;
        while (idx < N and cc(idx) < target) ++idx;
        ++idx;                                    // first node of the next bin
      } else {
        idx = (b * N) / ntarget;
      }
      idx = std::max(idx, edge.back() + 1);
      idx = std::min(idx, N - (ntarget - b));
      edge.push_back(idx);
    }
    edge.push_back(N);

    P.nbin = long(edge.size()) - 1;
    P.lo = nda::array<long, 1>(P.nbin);
    P.hi = nda::array<long, 1>(P.nbin);
    P.om_c = nda::array<double, 1>(P.nbin);
    for (long b = 0; b < P.nbin; ++b) {
      P.lo(b) = edge[size_t(b)];
      P.hi(b) = edge[size_t(b) + 1];
      double num = 0.0, den = 0.0;
      for (long j = P.lo(b); j < P.hi(b); ++j) { num += tc(j); den += u(j); }
      if (den > 0.0) {
        P.om_c(b) = num / den;                     // trace-weighted harmonic mean
      } else {
        P.om_c(b) = 0.5 * (Om(P.lo(b)) + Om(P.hi(b) - 1));
      }
      // guard: the harmonic mean is inside the node range by construction, but clamp against
      // round-off so the pole never leaves the support it came from.
      P.om_c(b) = std::min(std::max(P.om_c(b), Om(P.lo(b))), Om(P.hi(b) - 1));
      const double w = (Om(P.hi(b) - 1) - Om(P.lo(b))) / P.om_c(b);
      P.width_worst = std::max(P.width_worst, w);
    }
    return P;
  }

  /**
   * The pole list of a plan: [ +om_c(0..nbin-1), -om_c(0..nbin-1) ], in that order. The
   * residue of pole b is +A_b and of pole nbin+b is -A_b (eq. B).
   */
  inline nda::array<double, 1> pole_list(bin_plan_t const &P) {
    nda::array<double, 1> om(2 * P.nbin);
    for (long b = 0; b < P.nbin; ++b) { om(b) = P.om_c(b); om(P.nbin + b) = -P.om_c(b); }
    return om;
  }

  /**
   * Accumulate one merged residue matrix element from a run of Omega nodes:
   *
   *     A_b(P,Q) = -(1/pi) sum_{j in b} w_j Im W_PQ(Omega_j)      at +Omega_b,
   *    -A_b(P,Q)  written at the TRANSPOSED position (Q,P)        at -Omega_b   [eq. (B)].
   *
   * `ImW_O` is the Omega-slice of one (q, P, Q) element and `Ow` the grid's trapezoid weights.
   */
  template<typename ImW_slice_t, typename W_t>
  inline double merged_residue(ImW_slice_t const &ImW_O, W_t const &Ow, long lo, long hi) {
    double acc = 0.0;
    for (long j = lo; j < hi; ++j) acc += Ow(j) * ImW_O(j);
    return -inv_pi * acc;
  }

  /**
   * The Lehmann forward map of a pole representation, evaluated at one complex node:
   *     W(z) = sum_p R_p / (z - om_p).
   * Used by the production meter and by the toy gate.
   */
  inline ComplexType eval_pole_rep(nda::array<double, 1> const &om,
                                   nda::array<ComplexType, 1> const &R, ComplexType z) {
    ComplexType acc(0.0, 0.0);
    for (long p = 0; p < om.shape(0); ++p) acc += R(p) / (z - om(p));
    return acc;
  }

} // namespace wc_spectral
} // namespace methods

#endif // COQUI_WC_SPECTRAL_HPP
