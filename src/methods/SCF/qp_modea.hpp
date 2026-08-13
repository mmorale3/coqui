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

#ifndef COQUI_QP_MODEA_HPP
#define COQUI_QP_MODEA_HPP

/**
 * Project 2 (qpGW+BSE+EDMFT) increment QM3 -- the MODE-A quasiparticle map, live in the
 * qp/ev scf loops (spec notes/qm3_mode_a_loop_spec.md).
 *
 *     V^xc_ab(s,k) = 1/2 [ Sigma^c_ab(s,k; eps_a) + Sigma^c_ab(s,k; eps_b) ]
 *
 * assembled in the MO basis of the current outer iteration, with Sigma^c evaluated at REAL
 * quasiparticle energies by the QM2 contour-deformation kernel (sigma_route_b::sigma_cd) --
 * no analytic continuation anywhere. The existing Hermitize + MO -> primary tail of
 * qp_approx is reused unchanged.
 *
 * =====================================================================================
 * DERIVATION 1 -- MOMENTUM / SPIN / PREFACTOR ROUTING
 * [verified: re-derived mechanically FROM the production GW assembly, thc_gw.icc:341-410
 *  together with thc_solver_comm's primary_to_aux (:384-450) and aux_to_primary (:458-520);
 *  every factor is pinned by the QM3-b anchor gate]
 * =====================================================================================
 * The production GW self-energy is, in the THC auxiliary basis,
 *
 *     Sigma_PQ(s,k) += -(1/nkpts) * G_PQ(s, k-q) * W_PQ(q)          (Hadamard in P,Q)
 *
 * with the two basis transforms (thc_solver_comm, verbatim)
 *
 *     G_PQ(k')   = sum_ij X_Pi(k') G_ij(kp_to_ibz(k')) conj(X_Qj(k'))
 *     Sigma_ef   = sum_PQ conj(X_Pe(ks)) Sigma_PQ(ks) X_Qf(ks)
 *
 * and, for a symmetry-mapped external point ks = ks_to_k(isym, k), the D-matrix rotation
 * back to the IBZ orbital basis (thc_gw.icc:310-317)
 *
 *     Sigma_ij(k) = sum_ef conj(D_ei) Sigma_ef(ks) D_fj.
 *
 * Composing the three, and inserting the MO factorizations G_ij(k') = sum_n C_in g_n conj(C_jn)
 * (update_G, qp_scf_common.cpp:130-166 -- unit residues, poles at the CURRENT sE_ska) and
 * Sigma^MO_ab = sum_ij conj(C_ia) Sigma_ij C_jb, the ENTIRE chain collapses onto two
 * (Np x nbnd) MO collocation matrices,
 *
 *     XCe(P,a) = [ X(ks) . D(isym,k) . C(k) ](P,a)          (external leg, D = 1 at isym = 0)
 *     XCi(P,n) = [ X(k') . C(kp_to_ibz(k')) ](P,n)          (internal leg)
 *
 * and the pair vector of the spec,   A_P(k,k'; a,n) = conj(XCe(P,a)) * XCi(P,n),   giving
 *
 *     Sigma^c_ab(s,k; tau) = -(1/nk) sum_q sum_n g_n(k-q,tau)
 *                              sum_PQ A_P(a,n) W_PQ(q,tau) conj(A_Q(b,n)).
 *
 * SPIN: the GW loop carries a single spin label through G, W is spin-summed already (it is
 * built from the RPA polarization), and the (s,k) external index is never mixed -- the
 * internal G leg carries the SAME spin as the external one, with no extra factor. Verified
 * by inspection of the isk loop at thc_gw.icc:366-397 (`G_skPQ(is, ...)` at the external
 * `is`) and of the k-space allocation (:255, spin is a plain outer index).
 *
 * PREFACTOR AND THE KERNEL'S OWN SIGN. In tau space Sigma^{(q)}(tau) = -(1/nk) G(tau) W(tau);
 * Fourier of the product gives Sigma^{(q)}(i w_n) = -(1/(nk beta)) sum_m G(i w_n - i nu_m)
 * W(i nu_m), and W^c(i nu) is EVEN in nu (it is the transform of PH-symmetric-half tau data),
 * so the -i nu_m may be written +i nu_m. Comparing with the QM2 definition
 * Sigma^c(z) = -(1/beta) sum_m G(z + i nu_m) W^c(i nu_m) -- which is what sigma_cd returns,
 * minus sign INCLUDED -- the per-q contribution is
 *
 *     Sigma^{c,(q)}_ab(z) = (1/nk) * sigma_cd[ residues of W^c_{an,nb}(q, i nu) ].
 *
 * i.e. the band-element residues carry +1/nk, NOT -1/nk: the minus lives inside the kernel.
 * [assumed -- gate: QM3-b anchor] that W^c(i nu) is even to fit accuracy; the fitted rep is
 * built from, and measured against, the bosonic mesh, so any violation shows up in the
 * logged reconstruction error.
 *
 * TIME REVERSAL (the spec's section 3 HAZARD, resolved). The two in-tree conventions for a
 * trev transfer disagree: embed_eri_t.cpp:2129-2132 uses conj(W_PQ), vertex_sym.hpp:43-46
 * uses a PQ-TRANSPOSE. thc_gw -- the assembly this map must reproduce -- uses conj(W_PQ)
 * (had_prod2_conj / nda::conj at :381-393), so THE CONJ RULE IS THE ONE IMPLEMENTED HERE.
 * The two coincide iff W_PQ is Hermitian, which is asserted nowhere; the context build
 * MEASURES max|W - W^dag| / max|W| on the stored data every outer iteration and logs it, and
 * the anchor gate would catch a wrong choice at O(1). Unified branch table, read off
 * thc_gw.icc:377-395:
 *
 *     wconj = qp_trev(q');   kk = wconj ? qk_to_k2(qminus(qs), ks) : qk_to_k2(qs, ks);
 *     gconj = kp_trev(kk);   kg = gconj ? kp_trev_pair(kk) : kk;
 *     u(P,n) = gconj ? conj(XCi(kg)(P,n)) : XCi(kg)(P,n)
 *
 * (conj(G_PQ) = sum_n conj(u_P) g_n u_Q because the g_n are real), and the sandwich is
 *
 *     B(P,a) = conj(XCe(P,a)) * u(P,n),   M^(n,p)_ab = [ B^T W^(p) conj(B) ]_ab.
 *
 * =====================================================================================
 * DERIVATION 2 -- FIT LINEARITY (fit-then-contract == contract-then-fit)
 * [verified: exchange of two finite sums; the three hypotheses are checked below]
 * =====================================================================================
 * The support-constrained least squares is a FIXED matrix Mfit (imag_axes_ft::masked_pole_fit:
 * the retained column set comes from gap_edge and the rank from the singular spectrum at
 * build, never from the data), so residues are c_p(x) = sum_i Mfit(p,i) F_i(x) elementwise in
 * the batch label x. With A independent of the frequency/time axis,
 *
 *   contract-then-fit:  sum_i Mfit(p,i) [ sum_PQ A_P W_PQ(t_i) conj(A_Q) ]
 *   fit-then-contract:  sum_PQ A_P [ sum_i Mfit(p,i) W_PQ(t_i) ] conj(A_Q)
 *
 * and these are the same double sum reordered. The per-pole residue rescaling
 * w_p = tanh(hw_p/2) c_p is diagonal in p and commutes with the (a,b) contraction as well, and
 * the Gamma-head augmentation is added to W_PQ(tau) BEFORE the transform, hence is inside the
 * linear part. Hypotheses, all satisfied here: (i) Mfit data-independent; (ii) A independent
 * of tau/nu; (iii) the SAME column mask for every matrix element (gap_edge is a single global
 * number). The cheaper order is therefore used: fit ONCE per q on W_PQ (Np^2 right-hand
 * sides), then contract the resulting residue slabs.
 *
 * =====================================================================================
 * THE POLE CLOSURE AND WHAT IS CACHED
 * =====================================================================================
 * With W^c_{an,nb}(q, i nu) = sum_p w^(anb)_p / (i nu - om_p) and the unit-residue QP-pole G,
 * the bosonic sum closes (QM2) into
 *
 *     Sigma^c_ab(z) = sum_{J,p} M^(J,p)_ab [ n_B(om_p) + f(eps_J) ] / ( z - (eps_J - om_p) )
 *
 * where J = (q, n) is the flat INTERNAL state label (energy eps_J = eps_n(k-q), absolute) and
 * p runs over the retained auxiliary nodes. What is cached per external (s,k) is exactly the
 * residue slab M(a, b, J*npk + p) -- NOT the (a,n,b,nu) tensor. Evaluation at any z is then
 * pole algebra on that slab, so the inner consistency loop costs nbnd^2 * nJ * npk complex
 * FMAs per sweep and never re-touches W.
 */

#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <algorithm>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
#include "methods/SCF/sigma_route_b.hpp"
#include "methods/SCF/sigma_real_axis.hpp"

namespace methods {
namespace qp_modea {

  /** knobs (qp_params_t / toml; spec section 6). */
  struct modea_opts {
    std::string route = "cd";          // {cd, expansion}
    long nconsist = 5;                 // inner-consistency cap
    double consist_tol = 1e-8;         // a.u.
    double eta = 0.0;                  // evaluation offset i*eta (stress only)
    std::string wsupp = "auto";        // {"auto","off",<value in a.u.>}
    std::string wfit = "tau";          // {tau, nu}
    double wrtol = -1.0;               // masked-fit SVD cut; < 0 = the shared doctrine value
    // W^c residue-slab compression (stage 1b of wc_band_elements.hpp). wrank is a RELATIVE
    // eigenvalue cut on each Hermitian slab W^(p)_PQ: <= 0 disables the factorization and
    // takes the dense Np^2 sandwich (the reference path). wsketch selects the factorization
    // backend: 0 = automatic, > 0 = force the randomized sketch with that initial size,
    // < 0 = force LAPACK heev.
    double wrank = 1e-10;
    long wsketch = 0;
    long iter = 1;                     // outer iteration (1 => Route-A root refinement)
    int level = 2;                     // logging level for the per-iteration banner
  };

  /**
   * Last-run diagnostics, for gates and post-mortems. This is a REPORTING hook only -- no
   * code path branches on it. Gate QM3-b reads it to tabulate the anchor / delta_i / gap
   * triples of spec section 7(v) without having to parse the run log.
   */
  struct last_run_t {
    double anchor = -1.0;          // max rel dev of route-B Sigma^c vs the solver Sigma(i w)
    double anchor_expect = -1.0;   // = the W-fit bosonic-mesh reconstruction class
    double ratio_worst = -1.0;     // worst delta_i / class_i over gap-window states
    double anti_herm = -1.0;
    double dmax = -1.0, min_den = -1.0;
    double gap_edge = 0.0, rec_rel = -1.0, wall_s = 0.0, mem_mb = 0.0;
    double res_ratio = -1.0;       // max|c| / max|F| of the production W^c fit (worst q)
    double wrtol = -1.0;           // SVD cut actually used by that fit
    // the gap-window A/B harness read at the INCOMING energies (before the inner loop):
    double delta_in = -1.0;        // max delta_i    over HOMO-1..LUMO+1
    double class_in = -1.0;        // max class_i    over the same states
    double ratio_in = -1.0;        // max delta_i/class_i over the same states
    double tau_dev = -1.0;         // THE GATE quantity (spec rev 2)
    long n_fallback = 0;           // mode_b diagonal states demoted to z = mu
    // mode_a STRIP CLAMP census (rev 3 addendum item 2), last outer iteration, last sweep:
    long n_clamp = 0;              // evaluation energies clamped to mu (out-of-strip states)
    long n_clamp_win = 0;          // ... of which are gap-window states
    long n_homo_clamp = 0;         // (s,k) blocks whose per-k HOMO was clamped -- THE JUDGE
    long n_lumo_clamp = 0;         // (s,k) blocks whose per-k LUMO was clamped -- THE JUDGE
    long n_eval = 0, n_blocks = 0;
    bool converged_inner = false;  // every block's inner-consistency loop met consist_tol
    long iters = 0, n_support = 0, np_total = 0, nJ = 0, npk = 0;
    std::string wfit;
    // stage-1b (low-rank W^c slabs), last context build
    double wrank = 0.0;            // the knob value in force
    long wrank_max = 0;            // worst retained rank over (q,p)
    double wrank_mean = 0.0;       // mean retained rank over (q,p)
    double wtrunc = 0.0;           // worst 2-norm truncation residual over (q,p)
    double t_fit = 0.0, t_fac = 0.0, t_sand = 0.0;   // context-build stage wall times
    long Np = 0;                   // THC auxiliary basis size (the compression denominator)
    // the slab rank ladder: max / mean retained rank over (q,p) at the FIXED tolerances
    // detail::wrank_ladder = {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}. THE low-rank measurement --
    // whether r saturates with Np decides whether this compression reaches production.
    std::array<long, 5> lad_max{};
    std::array<double, 5> lad_mean{};
  };
  inline last_run_t &last_run() { static last_run_t x; return x; }

  /** rev-1 i w anchor threshold. RETAINED for the logged diagnostic only -- NOT a gate;
   *  see the tau anchor below and notes/qm3_mode_a_loop_spec.md rev 2. */
  inline constexpr double modea_anchor_gate = 1e-2;

  /** THE GATE (spec rev 2): the tau-domain anchor, in units of the W-fit reconstruction
   *  class. Measured headroom on lih222 is three orders, so 10x is generous. NOT a tunable. */
  inline constexpr double modea_tau_anchor_mult = 10.0;

  /** per-external-(s,k) cached residue slab. */
  struct sk_block {
    long is = -1, ik = -1;
    nda::array<ComplexType, 3> M;   // (nbnd, nbnd, nJ*npk) residues, 1/nk folded in
  };

  /** diagnostics collected once per context build (all logged, none of them a gate). */
  struct modea_diag {
    double gap_edge = 0.0;             // a.u.
    long n_support = 0, np_total = 0;
    double w_herm_rel = 0.0;           // max|W - W^dag| / max|W| on the stored tau data
    double ttw_imag = 0.0;             // max|Im Ttw_bb| (reality of the nu <-> tau kernel)
    double rec_rel_worst = 0.0;        // worst-q bosonic-mesh reconstruction rel error
    double fit_err_worst = 0.0;        // worst-q fit residual ON ITS OWN grid (NOT a quality
                                       // number -- see binding requirement 3)
    double res_ratio_worst = 0.0;
    double wall_s = 0.0;               // context build wall time
    double mem_mb = 0.0;               // peak extra memory (per rank, residue slabs + buffers)
    long nJ = 0, npk = 0;
    // stage-1b (low-rank W^c slabs) census + the per-stage wall clock
    long wrank_max = 0;                // worst retained rank over (q,p)
    double wrank_mean = 0.0;           // mean retained rank over (q,p)
    double wtrunc_worst = 0.0;         // worst |discarded lambda| / max|lambda| over (q,p)
    double wtrunc_frob_worst = 0.0;    // worst Frobenius-relative discarded weight
    double wanti_worst = 0.0;          // worst max|W - W^dag| / max|W| of a residue slab
    double t_fit = 0.0, t_fac = 0.0, t_sand = 0.0;
  };

  /**
   * The mode-A evaluator context: everything Sigma^c needs at ARBITRARY z, frozen for the
   * whole inner-consistency loop (spec section 4).
   */
  struct modea_ctx {
    bool active = false;
    bool have_cd = false;              // route == cd (residue slabs present)
    double beta = 0.0, mu = 0.0, eta = 0.0;
    long ns = 0, nk = 0, nbnd = 0, nkpts_full = 0;
    long nJ = 0, npk = 0;              // internal states, retained W poles
    modea_opts opts;
    modea_diag diag;

    double vbm = 0.0, cbm = 0.0;       // global QP band edges of the CURRENT spectrum
    nda::array<double, 1> om;          // (npk) retained W^c pole energies
    nda::array<double, 1> nB;          // (npk) n_B(om_p)
    nda::array<double, 1> epsJ;        // (ns*nk*nJ) internal pole energies, ABSOLUTE
    nda::array<double, 1> fJ;          // (ns*nk*nJ) f(eps_J)
    std::vector<sk_block> blocks;      // rank-local owned (s,k)
    nda::array<long, 1> owner;         // (ns*nk) -> rank that owns the block
    // diagonal residues for the evGW leg, replicated: (ns, nk, nbnd, nJ*npk)
    nda::array<ComplexType, 4> Mdiag;
    bool have_diag = false;

    long block_index(long is, long ik) const {
      for (size_t b = 0; b < blocks.size(); ++b)
        if (blocks[b].is == is and blocks[b].ik == ik) return long(b);
      return -1;
    }

    /**
     * The pole weights (n_B(om_p) + f(eps_J)) / (z - (eps_J - om_p)) flattened as
     * P = J*npk + p, for one (s,k) and one ABSOLUTE evaluation point z. Also returns the
     * smallest denominator met (the "min_den" tripwire of spec section 1).
     */
    double pole_weights(long is, long ik, ComplexType z,
                        nda::array<ComplexType, 1> &w) const {
      const long off = (is * nk + ik) * nJ;
      double min_den = 1e300;
      for (long J = 0; J < nJ; ++J) {
        const double e = epsJ(off + J), f = fJ(off + J);
        for (long p = 0; p < npk; ++p) {
          const ComplexType den = z - (e - om(p));
          min_den = std::min(min_den, std::abs(den));
          w(J * npk + p) = (nB(p) + f) / den;
        }
      }
      return min_den;
    }
  };

  // ---------------------------------------------------------------------------------------
  //  gap edge (support constraint)
  // ---------------------------------------------------------------------------------------

  /**
   * E_PH = min_{f<1/2} eps - max_{f>=1/2} eps over the CURRENT QP spectrum: the indirect
   * particle-hole excitation minimum, below which W^c has no spectral weight. Prior physical
   * information, not a tuned regularization (spec section 2).
   */
  inline double ph_gap_edge(nda::MemoryArrayOfRank<3> auto const &E_ska, double mu) {
    double lo = -1e300, hi = 1e300;   // top of occupied, bottom of empty
    auto [ns, nk, nbnd] = E_ska.shape();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k)
        for (long a = 0; a < nbnd; ++a) {
          const double e = E_ska(s, k, a).real();
          if (e < mu) lo = std::max(lo, e);
          else hi = std::min(hi, e);
        }
    if (lo <= -1e299 or hi >= 1e299) return 0.0;
    return hi - lo;
  }

  /** resolve the qp_modea_wsupp knob into an energy in a.u. (0 disables the constraint). */
  inline double resolve_gap_edge(std::string const &wsupp,
                                 nda::MemoryArrayOfRank<3> auto const &E_ska, double mu,
                                 int level) {
    constexpr double floor_au = 1e-4;   // ~2.7 meV: metallic / degenerate
    if (wsupp == "off") {
      app_log(level, "  - W^c support constraint:      OFF (qp_modea_wsupp = off)");
      return 0.0;
    }
    double ge;
    if (wsupp == "auto") {
      ge = ph_gap_edge(E_ska, mu);
      if (ge <= floor_au) {
        app_warning("qp_modea: the current QP spectrum has a particle-hole gap of {:.4g} a.u. "
                    "(<= {:.1g}); the W^c support constraint is DISABLED for this iteration "
                    "(metallic or degenerate). Route B at real z is then exposed to auxiliary "
                    "nodes inside the gap -- see notes/qm2_route_b_finite_t_spec.md.", ge,
                    floor_au);
        return 0.0;
      }
    } else {
      ge = std::stod(wsupp);
      utils::check(ge >= 0.0, "qp_modea: qp_modea_wsupp = {} must be >= 0.", wsupp);
    }
    return ge;
  }

  /**
   * Same, but clamped so that the constraint can never empty the auxiliary basis. A gap edge
   * larger than the auxiliary grid's outermost node retains ZERO columns, which used to abort
   * deep inside the least squares with a confusing message; it happens when a diverged QP
   * spectrum feeds "auto". Falls back to the unconstrained fit with a warning that names the
   * real cause.
   */
  inline double resolve_gap_edge_clamped(std::string const &wsupp,
                                         nda::MemoryArrayOfRank<3> auto const &E_ska,
                                         double mu, double max_abs_node, int level) {
    double ge = resolve_gap_edge(wsupp, E_ska, mu, level);
    if (ge > 0.0 and ge >= max_abs_node) {
      app_warning("qp_modea: the particle-hole gap of the current QP spectrum is {:.4g} a.u., "
                  "at or beyond the outermost auxiliary pole node ({:.4g} a.u.). The support "
                  "constraint would retain NO nodes, so it is disabled for this iteration. "
                  "A gap this large normally means the QP spectrum itself has diverged -- "
                  "check the inner-consistency and min_den diagnostics above.",
                  ge, max_abs_node);
      return 0.0;
    }
    return ge;
  }

  // ---------------------------------------------------------------------------------------
  //  Sigma^c evaluation and the inner consistency loop
  // ---------------------------------------------------------------------------------------

  /** result of one inner-consistency solve on one (s,k) block. */
  struct consist_result {
    long iters = 0;
    double dmax = 0.0;           // max_a |eps^(t) - eps^(t-1)| at exit
    double min_den = 1e300;      // smallest |z - (eps_J - om_p)| met
    double anti_herm = 0.0;      // max|V - V^dag| / max|V| of the RAW map
    bool converged = false;
    // WHERE the closest real-axis pole was met (the spec's min_den tripwire, resolved):
    long min_den_a = -1;         // external state
    double min_den_ea = 0.0;     // its eps_a - mu  (a.u.)
    double vmax = 0.0;           // max|V^xc| of the last raw map
  };

  // ---------------------------------------------------------------------------------------
  //  THE STRIP CLAMP  (spec rev 3 ADDENDUM item 2, 2026-08-12)
  // ---------------------------------------------------------------------------------------
  /**
   * Mode A needs Sigma^c at the quasiparticle energy of EVERY state, including states far
   * outside the analyticity strip (VBM - E_PH, CBM + E_PH), where the exact Sigma^c has
   * genuine spectral weight and the finite fitted pole set is dense: that is the measured
   * cause of the rev-1 inner-loop divergence (max|d eps| ~ 1e4-1e5 a.u.; no fit or eta knob
   * cell cured it). The rev-3.1 convention (MEASURED, see the reversal note below):
   *
   *     z_i = eps_i                                  if eps_i is inside the strip
   *     z_i = mu                                     otherwise
   *
   * applied to BOTH indices of 1/2 [Sigma(eps_a) + Sigma(eps_b)]. In-strip states are exact
   * mode A (eta -> 0); out-of-strip states are evaluated at the Fermi level -- the SAME
   * fallback mode_b uses for its out-of-strip diagonals, which converges on both fixtures.
   * The strip itself is the same prior information (the particle-hole edge E_PH of the
   * current QP spectrum) that the W^c support constraint and the mode_b strip test use.
   *
   * ⚠ REVERSAL OF THE FIRST READING (rev 3 addendum item 2 -> rev 3.1, both measured here
   * on 2026-08-12). The first implementation clamped to the strip BOUNDARY. On qe_lih222
   * that is fatal: with margin 0.05 E_PH = 0.14 eV the boundary sits INSIDE the fitted-pole
   * pile-up (min_den 3.9e-05 a.u. against a nominal clearance of 5.2e-03, fit residue ratio
   * 6.0e+03), the 125 of 128 states evaluated there returned |V^xc| of 10-19 eV against a
   * Sigma scale of 0.85 eV at mu, the gap collapsed 2.83 -> 0.18 eV at the second outer
   * iteration, E_PH -> 0 then disabled BOTH the support constraint and the strip, and the
   * rev-1 divergence followed (final gap -9.6e+03 eV). si222, whose judge states are never
   * clamped, was unaffected (9.248928 eV, 0.0006 eV from ac_pade). mu is the only evaluation
   * point whose analytic quality the fit guarantees.
   *
   * [verified: bounded by construction -- mu keeps a clearance of at least gap_edge from the
   *  nearest pole of the support-constrained rep, so the inner-consistency loop of section 4
   *  is retained.]
   * [assumed: Sigma^c(mu) is an acceptable stand-in for the true Sigma^c of a far state --
   *  gate: the clamp census (judge states must show 0 clamps) + the gap table + the dmax /
   *  gap / converged REQUIREs of the fixture gate. NOTE the tau anchor is VACUOUS once a
   *  spectrum has collapsed: the reconstruction class it is normalized against blows up in
   *  step with it. Those three REQUIREs are what catch this failure mode.]
   *
   * The clamp is INACTIVE when the support constraint itself is off (gap_edge = 0, i.e.
   * metallic / degenerate / qp_modea_wsupp = "off"): there is then no strip to speak of, and
   * collapsing every evaluation onto mu would be a silent Fermi-static map.
   */
  struct strip_t {
    double lo = 0.0, hi = 0.0, mu = 0.0;
    bool active = false;
    /** the mode-A evaluation point of a state at energy e (rev 3.1: out-of-strip -> mu). */
    double clamp(double e, bool *hit = nullptr) const {
      if (hit != nullptr) *hit = false;
      if (not active) return e;
      if (e < lo or e > hi) { if (hit != nullptr) *hit = true; return mu; }
      return e;
    }
  };

  inline strip_t strip_of(modea_ctx const &ctx) {
    strip_t s;
    const double d = 0.95 * ctx.diag.gap_edge;   // the same margin as modeb_in_strip
    s.active = (ctx.diag.gap_edge > 0.0 and ctx.cbm > ctx.vbm);
    s.lo = ctx.vbm - d;
    s.hi = ctx.cbm + d;
    s.mu = ctx.mu;
    return s;
  }

  /** clamp bookkeeping of ONE map assembly (one (s,k) block, one inner sweep). */
  struct clamp_census {
    long n_eval = 0;             // evaluation energies examined (= nbnd)
    long n_clamp = 0;            // ... of which were outside the strip and were moved to mu
    long n_clamp_win = 0;        // ... of which are gap-window states
    bool homo_clamp = false;     // this block's per-k HOMO was clamped (the judge reads it)
    bool lumo_clamp = false;
    double exc_lo = 0.0;         // worst excursion below the lower bound (a.u., >= 0)
    double exc_hi = 0.0;         // worst excursion above the upper bound (a.u., >= 0)
  };

  /**
   * V^xc_ab = 1/2 [ Sigma_ab(z_a) + Sigma_ab(z_b) ] at the STRIP-CLAMPED energies
   * z = clamp(eps) (rev 3.1: eps in-strip, mu out-of-strip).
   * D  (a,b) = Sigma_ab(z_a)   -- one gemv per a over the flat pole axis
   * D' (a,b) = Sigma_ab(z_b)   -- one dot per (a,b)
   * Both are formed EXPLICITLY (no Hermiticity identity) so that max|V - V^dag| is a genuine
   * routing tripwire rather than zero by construction.
   */
  inline double modea_vxc_cd(modea_ctx const &ctx, sk_block const &blk,
                             nda::MemoryArrayOfRank<1> auto const &eps,
                             nda::array<ComplexType, 2> &V, long *argmin = nullptr,
                             clamp_census *cc = nullptr,
                             std::vector<long> const *win = nullptr) {
    const long nbnd = ctx.nbnd, nP = ctx.nJ * ctx.npk;
    nda::array<ComplexType, 1> w(nP);
    nda::array<ComplexType, 2> D(nbnd, nbnd), Dp(nbnd, nbnd);
    double min_den = 1e300;

    // ---- the strip clamp + its census (spec rev 3 addendum item 2) ----
    const strip_t strip = strip_of(ctx);
    nda::array<double, 1> z_a(nbnd);
    if (cc != nullptr) *cc = clamp_census{};
    long homo = -1, lumo = -1;
    if (cc != nullptr)
      for (long a = 0; a < nbnd; ++a) {
        if (eps(a) < ctx.mu and (homo < 0 or eps(a) > eps(homo))) homo = a;
        if (eps(a) >= ctx.mu and (lumo < 0 or eps(a) < eps(lumo))) lumo = a;
      }
    for (long a = 0; a < nbnd; ++a) {
      bool hit = false;
      z_a(a) = strip.clamp(eps(a), &hit);
      if (cc == nullptr) continue;
      ++cc->n_eval;
      if (not hit) continue;
      ++cc->n_clamp;
      if (win != nullptr and std::find(win->begin(), win->end(), a) != win->end())
        ++cc->n_clamp_win;
      if (a == homo) cc->homo_clamp = true;
      if (a == lumo) cc->lumo_clamp = true;
      if (eps(a) < strip.lo) cc->exc_lo = std::max(cc->exc_lo, strip.lo - eps(a));
      else cc->exc_hi = std::max(cc->exc_hi, eps(a) - strip.hi);
    }

    for (long a = 0; a < nbnd; ++a) {
      const ComplexType z(z_a(a), ctx.eta);
      const double md = ctx.pole_weights(blk.is, blk.ik, z, w);
      if (md < min_den and argmin != nullptr) *argmin = a;
      min_den = std::min(min_den, md);
      for (long b = 0; b < nbnd; ++b) {
        ComplexType s(0.0);
        auto Mab = blk.M(a, b, nda::range::all);
        for (long P = 0; P < nP; ++P) s += Mab(P) * w(P);
        D(a, b) = s;
      }
    }
    for (long b = 0; b < nbnd; ++b) {
      const ComplexType z(z_a(b), ctx.eta);
      ctx.pole_weights(blk.is, blk.ik, z, w);
      for (long a = 0; a < nbnd; ++a) {
        ComplexType s(0.0);
        auto Mab = blk.M(a, b, nda::range::all);
        for (long P = 0; P < nP; ++P) s += Mab(P) * w(P);
        Dp(a, b) = s;
      }
    }
    V = 0.5 * (D + Dp);
    return min_den;
  }

  /** Sigma^c_ab(z) for a single ABSOLUTE z (used by the anchor gate and the delta_i table). */
  inline void modea_sigma_at(modea_ctx const &ctx, sk_block const &blk, ComplexType z,
                             nda::array<ComplexType, 2> &S) {
    const long nbnd = ctx.nbnd, nP = ctx.nJ * ctx.npk;
    nda::array<ComplexType, 1> w(nP);
    ctx.pole_weights(blk.is, blk.ik, z, w);
    for (long a = 0; a < nbnd; ++a)
      for (long b = 0; b < nbnd; ++b) {
        ComplexType s(0.0);
        auto Mab = blk.M(a, b, nda::range::all);
        for (long P = 0; P < nP; ++P) s += Mab(P) * w(P);
        S(a, b) = s;
      }
  }

  /** the anti-Hermitian residual of a raw map, max|V - V^dag| / max|V|. */
  inline double anti_herm_rel(nda::array<ComplexType, 2> const &V) {
    double num = 0.0, den = 0.0;
    const long n = V.shape(0);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) {
        num = std::max(num, std::abs(V(i, j) - std::conj(V(j, i))));
        den = std::max(den, std::abs(V(i, j)));
      }
    return (den > 0.0) ? num / den : 0.0;
  }

  /** Sigma^c_ii(z) from the cached slab -- the cheap diagonal used by the delta_i harness. */
  inline ComplexType modea_sigma_diag(modea_ctx const &ctx, sk_block const &blk, long i,
                                      ComplexType z, double *min_den = nullptr) {
    const long nP = ctx.nJ * ctx.npk;
    nda::array<ComplexType, 1> w(nP);
    const double md = ctx.pole_weights(blk.is, blk.ik, z, w);
    if (min_den != nullptr) *min_den = md;
    ComplexType s(0.0);
    auto Mii = blk.M(i, i, nda::range::all);
    for (long P = 0; P < nP; ++P) s += Mii(P) * w(P);
    return s;
  }

  /**
   * MODE B (Faleev - van Schilfgaarde - Kotani, PRL 93, 126406), the map the user ruling of
   * 2026-08-12 pivoted to. Spec rev 2:
   *
   *     V^xc_ab = Re Sigma^c_ab(mu)     a != b    (the strip centre -- always inside the
   *                                                analyticity strip, hence always safe)
   *     V^xc_aa = Re Sigma^c_aa(eps_a)  diagonal
   *
   * i.e. the existing off_diag_mode = "fermi" idiom (qp_scf_common.cpp:679-681) with the CD
   * closed form in place of AC.evaluate. There is NO inner-consistency loop: the off-diagonals
   * do not depend on eps at all, and the outer loop supplies the self-consistency, exactly as
   * on the AC path.
   *
   * WHY THIS AVOIDS THE MODE-A DIVERGENCE. Mode A needed Sigma^c at EVERY eps_a, including
   * states with |eps - mu| >~ wmax, which land on the fitted poles at eps_J - om_p. Mode B
   * evaluates the whole off-diagonal block at z = mu, where the support constraint guarantees
   * a clearance of at least gap_edge, and only the DIAGONAL is evaluated off-centre.
   *
   * READING OF "Re". Taken literally on the DIAGONAL, where it is unambiguous and is what the
   * subsequent Hermitization would produce anyway. The off-diagonals are left as evaluated and
   * the EXISTING :713 Hermitize tail takes their Hermitian part -- elementwise Re on an
   * off-diagonal block is not basis-covariant, whereas (V + V^dag)/2 is, and for a Hermitian
   * W the two agree. FLAGGED: this is an interpretation of the spec's one-line formula.
   *
   * SAFEGUARD (bounded, logged, never silent). Per diagonal state: if the evaluation sits
   * near a pole (min_den below the floor) or the result exceeds a data-derived sanity bound,
   * that state falls back to z = mu -- mode B where the representation resolves the state,
   * Fermi-static where it cannot. Counted and warned, never silent.
   */
  struct modeb_result {
    long n_fallback = 0;         // diagonal states demoted to z = mu
    long n_fallback_win = 0;     // ... of which are in the gap window
    long n_sanity_trip = 0;      // STRIP-INTERIOR states tripping the |ReSigma| bound
    bool homo_fallback = false;  // this block's per-k HOMO fell back (the judge reads it)
    bool lumo_fallback = false;  // this block's per-k LUMO fell back
    double min_den = 1e300;      // smallest |z - (eps_J - om_p)| met on the diagonal
    double anti_herm = 0.0;      // max|V - V^dag| / max|V| of the RAW map
    double vmax = 0.0;
  };

  /**
   * THE STRIP TEST -- the diagonal fallback criterion (SPEC-AUTHOR DIRECTED, 2026-08-12,
   * replacing the agent-chosen min_den floor + |ReSigma| bound that the first mode-B
   * measurement showed were uncalibrated: at floor 1e-3 they demoted 86 of 96 states).
   *
   * Sigma^c has no genuine spectral weight between VBM - E_PH and CBM + E_PH: the nearest
   * pole of the exact object sits at an occupied energy minus a W^c excitation, or an empty
   * energy plus one, and W^c's support starts at the particle-hole edge E_PH. That is the
   * SAME prior information the W support constraint uses, so the criterion is deterministic
   * and calibration-free: inside the strip route B is trustworthy, outside it the finite pole
   * set is dense and the evaluation is meaningless.
   *
   *     fall back to z = mu  iff  eps_a <= VBM - (E_PH - m)  or  eps_a >= CBM + (E_PH - m),
   *     m = 0.05 * E_PH.
   */
  inline bool modeb_in_strip(double eps_a, double vbm, double cbm, double E_PH) {
    const double d = 0.95 * E_PH;          // E_PH - m, m = 0.05 E_PH
    return (eps_a > vbm - d) and (eps_a < cbm + d);
  }

  /**
   * Floor on |z - (eps_J - om_p)|, below which a single fitted pole dominates the sum.
   *
   * AGENT-CHOSEN AND NOT YET CALIBRATED -- FLAGGED. The spec rev 2 asks for "min_den < floor"
   * without fixing the value. A first attempt at 1e-3 a.u. (27 meV) was measured to be far
   * too aggressive: the pole set has nJ x npk ~ 3.6e3 members spread over ~4 a.u., so the mean
   * spacing is ~1e-3 and that floor demoted 86 of 96 diagonal states on qe_lih223_sym,
   * degenerating mode B into the Fermi-static map. It is lowered here so that the DATA-DERIVED
   * |Re Sigma| sanity bound does the real work and this trigger only catches a genuine
   * on-pole evaluation. The fallback counts are reported; the value wants a ruling.
   */
  inline constexpr double modeb_min_den_floor = 1e-6;   // a.u.
  /** |Re Sigma_aa(eps_a)| beyond this multiple of the block's own scale at mu is insane. */
  inline constexpr double modeb_sanity_mult = 10.0;

  inline modeb_result modeb_vxc(modea_ctx const &ctx, sk_block const &blk,
                                nda::MemoryArrayOfRank<1> auto const &eps,
                                std::vector<long> const &win,
                                nda::array<ComplexType, 2> &V) {
    modeb_result out;
    const long nbnd = ctx.nbnd;
    modea_sigma_at(ctx, blk, ComplexType(ctx.mu, ctx.eta), V);
    double scale = 0.0;
    for (long a = 0; a < nbnd; ++a) scale = std::max(scale, std::abs(V(a, a).real()));
    const double bound = modeb_sanity_mult * std::max(scale, 1e-6);
    // this block's own band edges -- the states the judge reads per k
    long homo = -1, lumo = -1;
    for (long a = 0; a < nbnd; ++a) {
      if (eps(a) < ctx.mu and (homo < 0 or eps(a) > eps(homo))) homo = a;
      if (eps(a) >= ctx.mu and (lumo < 0 or eps(a) < eps(lumo))) lumo = a;
    }
    for (long a = 0; a < nbnd; ++a) {
      const bool in_strip = modeb_in_strip(eps(a), ctx.vbm, ctx.cbm, ctx.diag.gap_edge);
      double md = 1e300;
      const ComplexType Sd =
          modea_sigma_diag(ctx, blk, a, ComplexType(eps(a), ctx.eta), &md);
      out.min_den = std::min(out.min_den, md);
      // the |ReSigma| bound is now a FALSIFICATION COUNTER, not a trigger: a strip-interior
      // state tripping it would mean the analyticity argument is wrong.
      if (in_strip and std::abs(Sd.real()) > bound) ++out.n_sanity_trip;
      if (in_strip) {
        V(a, a) = ComplexType(Sd.real(), 0.0);
      } else {
        ++out.n_fallback;
        if (std::find(win.begin(), win.end(), a) != win.end()) ++out.n_fallback_win;
        if (a == homo) out.homo_fallback = true;
        if (a == lumo) out.lumo_fallback = true;
        V(a, a) = ComplexType(V(a, a).real(), 0.0);
      }
    }
    for (long i = 0; i < nbnd; ++i)
      for (long j = 0; j < nbnd; ++j) out.vmax = std::max(out.vmax, std::abs(V(i, j)));
    out.anti_herm = anti_herm_rel(V);
    return out;
  }

  /**
   * The inner quasiparticle consistency loop of spec section 4, at FIXED Sigma data:
   *
   *   repeat: V^xc = 1/2[Sigma(eps_a) + Sigma(eps_b)]; Hermitize;
   *           eps <- eigvals(Hstat + V^xc);  until max|d eps| < tol or the cap is hit.
   *
   * `Hstat_ab` is the static (H0 + HF) part already in the MO basis; the MO basis itself is
   * FROZEN inside this loop -- basis updates belong to the outer loop. Returns the LAST V^xc
   * (raw, un-Hermitized: the caller's :713 tail does that, and the raw residual is logged).
   */
  template<class SigmaBlock>
  inline consist_result inner_consistency(SigmaBlock &&sigma_of_eps,
                                          nda::array<ComplexType, 2> const &Hstat_ab,
                                          nda::array<double, 1> &eps,
                                          nda::array<ComplexType, 2> &V,
                                          long nconsist, double tol) {
    consist_result out;
    const long nbnd = Hstat_ab.shape(0);
    nda::matrix<ComplexType> H(nbnd, nbnd);
    nda::array<double, 1> eps_new(nbnd);
    for (long t = 1; t <= std::max(1L, nconsist); ++t) {
      long amin = -1;
      const double md = sigma_of_eps(eps, V, &amin);
      if (md < out.min_den) { out.min_den = md; out.min_den_a = amin; out.min_den_ea = (amin >= 0) ? eps(amin) : 0.0; }
      out.iters = t;
      out.vmax = 0.0;
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j) out.vmax = std::max(out.vmax, std::abs(V(i, j)));
      app_log(3, "    mode_a inner t = {}: max|V| = {:.4e}, min_den = {:.4e} at state {} "
                 "(eps = {:+.6f})", t, out.vmax, md, amin, (amin >= 0) ? eps(amin) : 0.0);
      // Hermitize (the same 1/2 (V + V^dag) as the qp_approx tail) before diagonalizing
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j)
          H(i, j) = Hstat_ab(i, j) + 0.5 * (V(i, j) + std::conj(V(j, i)));
      auto [ev, evec] = nda::linalg::eigenelements(H);
      double d = 0.0;
      for (long a = 0; a < nbnd; ++a) {
        eps_new(a) = ev(a);
        d = std::max(d, std::abs(eps_new(a) - eps(a)));
      }
      eps = eps_new;
      out.dmax = d;
      if (d < tol) { out.converged = true; break; }
    }
    out.anti_herm = anti_herm_rel(V);
    return out;
  }

  /**
   * The EXACT imaginary-time image of the route-B pole representation, on the backend's
   * fermionic tau grid:
   *
   *     Sigma_B(tau_i) = sum_{J,p} R_{Jp} g(E_{Jp}, tau_i),
   *     R_{Jp} = M_ab^{(J,p)} [n_B(om_p) + f(eps_J)],   E_{Jp} = eps_J - om_p,
   *
   * with g the SAME kernel update_G uses (qp_scf_common.cpp:134-141), i.e. the tau transform
   * of 1/(i w_n - (E - mu)) in the absolute-energy convention of sigma_route_b.
   *
   * WHY THIS EXISTS (the decisive oracle for the anchor discrepancy). The anchor compares
   * route B against the solver AT FERMIONIC NODES, so it sees the sum of (a) any error in the
   * W^c band elements and (b) any error in the solver's OWN tau -> i w transform. The solver
   * forms Sigma^c(tau) = G(tau) W(tau) pointwise on the DLR tau nodes; a product of two
   * rank-N DLR functions is not rank-N representable, so its transform can alias. Comparing
   * in TAU bypasses that transform entirely and separates the two: tau agreement at the W-fit
   * class together with i w disagreement isolates the discrepancy in the reference's
   * transform; equal disagreement in both domains puts it in the contraction.
   */
  inline void modea_sigma_tau(modea_ctx const &ctx, sk_block const &blk, long a, long b,
                              nda::MemoryArrayOfRank<1> auto const &tau,
                              nda::array<ComplexType, 1> &out) {
    const long nt = tau.shape(0), npk = ctx.npk, nJ = ctx.nJ;
    const long off = (blk.is * ctx.nk + blk.ik) * nJ;
    const double beta = ctx.beta;
    auto g = [&](double E, double t) {
      const double x = E - ctx.mu;
      if (x > 0.0) return -std::exp(-x * t) / (1.0 + std::exp(-x * beta));
      return -std::exp(x * (beta - t)) / (1.0 + std::exp(x * beta));
    };
    out() = ComplexType(0.0);
    for (long J = 0; J < nJ; ++J) {
      const double e = ctx.epsJ(off + J), f = ctx.fJ(off + J);
      for (long p = 0; p < npk; ++p) {
        const ComplexType R = blk.M(a, b, J * npk + p) * (ctx.nB(p) + f);
        const double E = e - ctx.om(p);
        for (long i = 0; i < nt; ++i) out(i) += R * g(E, tau(i));
      }
    }
  }

  /**
   * REVERSAL OF A SPEC DEFAULT (measured, 2026-08-12; coordinator ruling the same day).
   *
   * notes/qm3_mode_a_loop_spec.md section 4 adopted, as a flagged default, a refinement of the
   * inner loop's STARTING energies at the first outer iteration: "refine the diagonal by
   * Route-A z0=0 roots from the gathered Sigma(iw) -- cheap, adopted default, log it". That
   * path is DELETED, not switch-guarded.
   *
   * Why: the z0 = 0 expansion is a Taylor series about the Fermi level of radius
   * R_conv = |c_{p-1}/c_p|, fitted from |t| <= 0.053 a.u. (9 fermionic nodes at beta = 1000).
   * Newton on it is a runaway for every state outside that radius. Unguarded it moved lih222
   * states to |eps - mu| ~ 1e4 eV on the first outer iteration. GUARDED -- accepting a root
   * only inside R_conv -- it still accepted 9 of 16 roots at k = 0 with a max shift of
   * 0.21 a.u. and turned the fixture's 2.83 eV particle-hole gap into 30.65 eV, i.e. it
   * destroyed the band structure it was supposed to improve.
   *
   * The incoming sE_ska is the natural continuation of the outer loop and is already a
   * converged-QP-loop spectrum, so the inner loop now starts there unconditionally.
   */

} // qp_modea
} // methods

#endif // COQUI_QP_MODEA_HPP
