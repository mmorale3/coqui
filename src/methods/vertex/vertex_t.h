/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#ifndef COQUI_VERTEX_T_H
#define COQUI_VERTEX_T_H

// Refinement 2 W-bar iteration cache API is available (notes/wbar_cache.md);
// consumed by tests that must also compile against pre-cache checkouts.
#define VERTEX_WCACHE_API 1

// INCREMENT S2: the static-vertex W0[G] infrastructure API is available
// (notes/static_vertex_implementation_plan.md section 2.2); consumed by tests that
// must also compile against pre-S2 checkouts.
#define VERTEX_W0_API 1

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/vertex/vertex_sym.hpp"
#include "methods/embedding/projector_t.h"

namespace methods {
namespace solvers {

  /**
   * Rung mode of the vertex correction (notes/static_vertex_implementation_plan.md
   * section 2.1). ONE vertex_t drives ALL cuts of the selected mode, so half-theories
   * (a static-rung Sigma^C combined with a dynamic-rung Pi^C injection, or either cut
   * alone) have no representable configuration -- Phi-derivability is structural here,
   * not a convention the caller has to respect.
   *   - dynamic_rung: Formulation B, the parent theory: dynamic W rungs, both cuts
   *                   (Sigma^C = G^3W^2 double convolution, Pi^C = G^4W). DEFAULT,
   *                   bit-identical to the pre-vertex_rung code path.
   *   - static_rung : B-S, the iv = 0 statically screened truncation: P = RPA (no Pi^C
   *                   injection at all), Sigma = Sigma^{C,x} + Sigma^{C,r} (never one
   *                   alone).
   *   - linear_rung : B-L, the tangent completion of B-S, first order in
   *                   dW = W - W^0[G]: P^{C,L} injected, four Sigma pieces together.
   * The static_rung/linear_rung KERNELS are not implemented yet: selecting either one
   * with a non-empty subspace C aborts in the constructor (increment S2+ of the plan).
   * C = empty set stays an exact no-op in every mode.
   */
  enum vertex_rung_e {
    dynamic_rung, linear_rung, static_rung
  };

  inline std::string vertex_rung_enum_to_string(int rung) {
    switch (rung) {
      case vertex_rung_e::dynamic_rung:
        return "dynamic";
      case vertex_rung_e::linear_rung:
        return "linear";
      case vertex_rung_e::static_rung:
        return "static";
      default:
        return "not recognized...";
    }
  }

  /**
   * INCREMENT S2 -- the two distribution-level primitives of the W0[G] build
   * (notes/static_vertex_implementation_plan.md section 2.2). They live here, at
   * namespace scope, for the same reason vertex_secondary_fold.hpp's folds do: the
   * (P,Q)-split behavior has to be drivable by a unit test with deterministic stand-in
   * data (test_vertex_dfold), independently of any MF/THC/scGW state.
   */
  namespace vertex_w0_detail {

    /**
     * The i.nu = 0 ROW of the PH-symmetric tau -> nu bosonic transform, i.e. row n = 0
     * of the Twt_pos matrix IAFT::tau_to_w_PHsym builds internally (IAFT.icc:62-70):
     *
     *   R(it) = Twt(nw_b/2, it) + Twt(nw_b/2, nt_b-1-it)     (it != nt_b-1-it)
     *   R(it) = Twt(nw_b/2, it)                              (self-mirrored tau node)
     *
     * so that  A(i.nu = 0, ...) = sum_it R(it) A(it, ...)  reproduces EXACTLY the
     * index-0 slice of tau_to_w_PHsym's output -- the verified static-slice convention
     * of gf2_t::get_static_W / dW0 (thc_gf2.icc:239-247) and of scr_coulomb_t's
     * epsilon_inf report (scr_coulomb_t.cpp:170-178). Nothing else in this file assumes
     * anything about the backend; the row is a pure linear functional of the tau data,
     * so it is available on DLR and IR alike (decision D3 concerns only the LATER
     * Pi^{C,0}(tau = 0) row, section 2.4).
     */
    inline nda::array<ComplexType, 1> nu0_transform_row(imag_axes_ft::IAFT const &ft) {
      const long nt_b = ft.nt_b(), nw_b = ft.nw_b();
      const long nt_half = (nt_b % 2 == 0) ? nt_b / 2 : nt_b / 2 + 1;
      auto Twt = ft.Twt_bb();                       // (nw_b, nt_b)
      const long iw0 = nw_b / 2;                    // the i.nu = 0 node of the full mesh
      nda::array<ComplexType, 1> R(nt_half);
      for (long it = 0; it < nt_half; ++it) {
        const long imt = nt_b - it - 1;
        R(it) = (it == imt) ? Twt(iw0, it) : Twt(iw0, it) + Twt(iw0, imt);
      }
      return R;
    }

    /**
     * INCREMENT S4 -- the tau = 0 ROW (notes/static_vertex_implementation_plan.md
     * section 2.4). Returns the length-nw_b vector R_nu such that, for any BOSONIC
     * object A represented on the imaginary-axis basis,
     *
     *     A(tau = 0) = sum_nu R_nu A(i.nu)      (exact, no truncated Matsubara sum)
     *
     * This is the legal evaluation of (1/beta) sum_nu A(i.nu): the sparse/DLR nodes are
     * FITTING nodes, not Fourier points, so a plain sum over the sampled frequencies is
     * NOT the Matsubara sum (the standing rule of notes/ir-dlr-convolution-rules).
     * Instead the row is the composition
     *
     *     R = [tau-interpolation row at tau = 0] . Ttw_bb
     *
     * i.e. "fit on the bosonic grid, then evaluate the basis at tau = 0".
     *
     * WHY A FERMIONIC-LOOKING TAU BASIS IS CORRECT HERE (checked 2026-07-28): in CoQui's
     * DLR backend there is exactly ONE imaginary-time basis. dlr_driver.hpp:320-330 sets
     * nt_b = nt_f = _it_ops.rank() and tau_mesh_b() = tau_mesh_f(); the statistics enter
     * only on the MATSUBARA side (_if_ops_f vs _if_ops_b => Ttw_ff vs Ttw_bb). So
     * construct_tau_interpolate_matrix, which is built purely from _it_ops, is
     * statistics-agnostic and applies to this bosonic object unchanged.
     *
     * DECISION D3 (resolved here): this row is DLR-ONLY. construct_tau_interpolate_matrix
     * is implemented for DLR (cppdlr coefs2eval, dlr_driver.hpp:134) and hard-aborts on
     * IR ("not implemented yet", ir_driver.hpp:128). Per the ruling, the static modes
     * therefore stay DLR-required and no speculative IR plumbing is added.
     *
     * Built once (a single nw_b vector) and applied per (q, aux block).
     */
    inline nda::array<ComplexType, 1> tau0_transform_row(imag_axes_ft::IAFT const &ft) {
      const long nt_b = ft.nt_b(), nw_b = ft.nw_b();
      // tau = 0 in the [-1, 1] convention of construct_tau_interpolate_matrix
      nda::array<double, 1> tau_out(1);
      tau_out(0) = -1.0;
      auto R0 = ft.construct_tau_interpolate_matrix(tau_out);   // (1, nt_b)
      auto Ttw = ft.Ttw_bb();                                   // (nt_b, nw_b)
      utils::check(R0.shape(1) == nt_b,
                   "vertex_w0_detail::tau0_transform_row: interpolation row length {} != "
                   "nt_b = {}.", R0.shape(1), nt_b);
      nda::array<ComplexType, 1> R(nw_b);
      for (long n = 0; n < nw_b; ++n) {
        ComplexType acc(0.0);
        for (long it = 0; it < nt_b; ++it) acc += R0(0, it) * Ttw(it, n);
        R(n) = acc;
      }
      return R;
    }

    /**
     * Contract a tau-domain (t, q, P, Q)-distributed array against the single
     * transform row R_t, producing the i.nu = 0 row on a DIFFERENT (P,Q)-block
     * layout -- without ever materializing a full (Np x Np) plane on any rank and
     * without a full-size copy of the tau array.
     *
     * The RPA polarizability lives on the grid {nt_procs, 1, np_P, np_Q}
     * (rpa_pi.icc:66-67), whose (P,Q) partition covers only np_P*np_Q of the ranks;
     * the W0 layout wants (P,Q) spread over ALL ranks with q unsplit (the
     * thc.dZ({1,nP,nQ}) layout the production Z fold already uses). Bridging the two
     * needs one reduction over the t-processor direction AND one (P,Q) reshuffle. Both
     * are done by a SINGLE math::nda::redistribute of an array whose first axis is
     * indexed by the t-PROCESSOR (length nt_procs, not nt_half):
     *
     *   (a) each rank contracts its OWN tau chunk into its own slot -> a partial
     *       (1, nq_loc, P_bs, Q_bs) block written with a pure local copy;
     *   (b) redistribute {nt_procs, nq, Np, Np} from the source grid onto the output's
     *       {1, 1, nP, nQ} grid (t-processor axis then unsplit) -- transient size
     *       nt_procs/nt_half of the tau array, i.e. strictly smaller than the input;
     *   (c) sum the nt_procs partials locally in index order (deterministic; the
     *       disjoint tau partition makes the sum EXACT -- no reassociation beyond the
     *       fixed order of the gemm row it replaces).
     *
     * @param dA_tqPQ   - [INPUT]  tau-domain array, global (nt_half, nq, Np, Np)
     * @param R_t       - [INPUT]  the nt_half transform row (nu0_transform_row)
     * @param dA0_1qPQ  - [OUTPUT] global (1, nq, Np, Np) on a grid {1, 1, nP, nQ};
     *                    zeroed and filled here. The leading length-1 axis keeps the
     *                    object a rank-4 "one-frequency" array, so the pinned
     *                    single-frequency machinery (dyson_W_in_place's algebra,
     *                    div_utils::eps_inv_head_w) applies verbatim.
     */
    template<typename dArray_in_t, typename RArr_t, typename dArray_out_t>
    void extract_nu0_row(dArray_in_t const &dA_tqPQ, RArr_t const &R_t,
                         dArray_out_t &dA0_1qPQ) {
      using Arr4 = nda::array<ComplexType, 4>;
      auto comm = dA_tqPQ.communicator();
      auto gs = dA_tqPQ.global_shape();      // (nt_half, nq, Np, Np)
      auto grd = dA_tqPQ.grid();             // (nt_procs, nq_procs, np_P, np_Q)
      auto bsz = dA_tqPQ.block_size();
      auto og = dA0_1qPQ.global_shape();     // (1, nq, Np, Np)
      auto ogr = dA0_1qPQ.grid();            // (1, 1, nP, nQ)
      auto obs = dA0_1qPQ.block_size();
      utils::check(R_t.shape(0) == gs[0],
                   "vertex_w0_detail::extract_nu0_row: transform row length {} != nt_half "
                   "= {}.", R_t.shape(0), gs[0]);
      utils::check(og[0] == 1 and og[1] == gs[1] and og[2] == gs[2] and og[3] == gs[3],
                   "vertex_w0_detail::extract_nu0_row: output global shape ({}, {}, {}, {}) "
                   "!= (1, {}, {}, {}).", og[0], og[1], og[2], og[3], gs[1], gs[2], gs[3]);
      utils::check(ogr[0] == 1 and ogr[1] == 1,
                   "vertex_w0_detail::extract_nu0_row: the output frequency and q axes must "
                   "NOT be split (grid = {{{}, {}, {}, {}}}).",
                   ogr[0], ogr[1], ogr[2], ogr[3]);

      const long ntp = grd[0];               // number of tau-PROCESSOR groups
      // (a) my partial, on the SOURCE (q,P,Q) partition verbatim (same grid, same
      //     shapes, same block sizes on axes 1..3) => pure local write, no comm.
      auto dpart = math::nda::make_distributed_array<Arr4>(
          *comm, grd, {ntp, gs[1], gs[2], gs[3]}, {1, bsz[1], bsz[2], bsz[3]});
      {
        auto A = dA_tqPQ.local();
        auto p = dpart.local();
        auto ls = dA_tqPQ.local_shape();
        auto pls = dpart.local_shape();
        utils::check(pls[0] == 1 and pls[1] == ls[1] and pls[2] == ls[2] and pls[3] == ls[3],
                     "vertex_w0_detail::extract_nu0_row: partial local shape ({}, {}, {}, "
                     "{}) does not mirror the source block ({}, {}, {}, {}).",
                     pls[0], pls[1], pls[2], pls[3], ls[0], ls[1], ls[2], ls[3]);
        const long t0 = dA_tqPQ.origin()[0];
        p() = ComplexType(0.0);
        for (long it = 0; it < ls[0]; ++it) {
          const ComplexType c = R_t(t0 + it);
          for (long iq = 0; iq < ls[1]; ++iq)
            for (long ip = 0; ip < ls[2]; ++ip)
              for (long jq = 0; jq < ls[3]; ++jq)
                p(0, iq, ip, jq) += c * A(it, iq, ip, jq);
        }
      }
      // (b) one redistribute: t-processor axis -> unsplit, (P,Q) -> the output partition.
      auto dgath = math::nda::make_distributed_array<Arr4>(
          *comm, ogr, {ntp, og[1], og[2], og[3]}, {1, obs[1], obs[2], obs[3]});
      if (comm->size() == 1) dgath.local() = dpart.local();
      else math::nda::redistribute(dpart, dgath);
      dpart.reset();
      // (c) sum the nt_procs partials in index order (exact: disjoint tau partition).
      auto g = dgath.local();
      auto out = dA0_1qPQ.local();
      auto ols = dA0_1qPQ.local_shape();
      utils::check(g.shape(1) == ols[1] and g.shape(2) == ols[2] and g.shape(3) == ols[3],
                   "vertex_w0_detail::extract_nu0_row: gathered block ({}, {}, {}) does not "
                   "match the output block ({}, {}, {}).",
                   g.shape(1), g.shape(2), g.shape(3), ols[1], ols[2], ols[3]);
      out() = ComplexType(0.0);
      for (long j = 0; j < ntp; ++j)
        for (long iq = 0; iq < ols[1]; ++iq)
          for (long ip = 0; ip < ols[2]; ++ip)
            for (long jq = 0; jq < ols[3]; ++jq)
              out(0, iq, ip, jq) += g(j, iq, ip, jq);
    }

  } // vertex_w0_detail

  inline vertex_rung_e string_to_vertex_rung_enum(std::string const &rung) {
    if (rung == "dynamic") {
      return vertex_rung_e::dynamic_rung;
    } else if (rung == "linear") {
      return vertex_rung_e::linear_rung;
    } else if (rung == "static") {
      return vertex_rung_e::static_rung;
    } else {
      utils::check(false, "vertex_t: unknown vertex_rung: {}. Valid options are \"dynamic\" "
                          "(default, the parent Formulation B), \"linear\" (B-L) and "
                          "\"static\" (B-S).", rung);
      return vertex_rung_e::dynamic_rung;
    }
  }

  /**
   * @brief vertex_t class
   *
   * Phi-derivable second-order-exchange vertex correction on top of scGW,
   * with all internal lines restricted to a near-E_F orbital subspace C
   * defined by a FIXED projector P(k) = U(k) U(k)^dag onto M correlated
   * orbitals per (spin, k) (notes/wannier_projector_theory.md).
   *
   * TWO subspace modes, one code path (the kernels are projector-general --
   * memo section 2.1, zero kernel edits):
   *   - WINDOW MODE (default): C = the contiguous band window
   *     [band_window.first(), band_window.last()); U(k) is the trivial 0/1
   *     column-selection isometry (identity on the window, zero outside).
   *     The input slices X(:,C) and G_CC and the C-C block injection are used
   *     directly -- the historic path, BIT-IDENTICAL.
   *   - WANNIER MODE (set_wannier_projector): C = span of M Wannier orbitals
   *     |w_a(k)> = sum_i U_ia(k) |psi_i(k)>, U an Norb x M isometry per (s,k)
   *     read from a TRIQS-compatible wan.h5 via projector_t (memo section 0:
   *     U = dagger(proj_mat) on rows W_rng, zero elsewhere; nImps == 1). The
   *     four input-slice sites become X_bar = X.U, G_bar = U^dag G U, the
   *     secondary C(q) is built from the rotated collocation, and the Sigma^C
   *     injection is the operator sandwich U Sigma_bar U^dag into the W_rng
   *     block (memo C2/C3/C4). U is Loewdin-orthonormalized at load (owner
   *     ruling Q1) and is FIXED for the whole SCF loop (demand D1; changing U
   *     = a restart, memo section 1.4). Window mode is the U = 1_window limit.
   *
   * One generating functional Phi_2^C, two cuts, evaluated TOGETHER
   * (never one alone -- Phi-derivability / conservation):
   *   - Sigma^C = dPhi_2^C/dG   (G^3 W^2)  -> eval_Sigma_C()
   *   - Pi^C    = -2 dPhi_2^C/dW (G^4 W)   -> eval_Pi_C()
   *
   * Both entry points are IBZ-resident by construction:
   *   - Sigma^C is accumulated into sSigma_tskij: (nt_f, ns, nkpts_ibz, nbnd, nbnd)
   *   - Pi^C matches the RPA polarizability grid: (nt_half, nqpts_ibz, Np, Np)
   *
   * Semantics of the configuration:
   *   - vertex_type == "none"        : vertex disabled; callers must not invoke
   *                                    the entry points (guard with active()).
   *   - vertex_type == "2nd_exchange": vertex enabled. An empty band window
   *                                    (C = empty set) must reproduce plain scGW
   *                                    exactly -- active() is false and the
   *                                    entry points are never invoked.
   *   - rung (vertex_rung_e above)   : WHICH theory this single vertex_t drives --
   *                                    "dynamic" (default, Formulation B; the path
   *                                    documented below) or the "static"/"linear"
   *                                    (B-S/B-L) truncations, whose kernels land at
   *                                    increment S2+.
   *
   * STATUS: both kernels support symmetry-free AND symmetry-reduced (IBZ)
   * k-meshes (notes/vertex_ibz_symmetry.md): external axes are IBZ-resident,
   * internal sums cover the full BZ, and the rung transfers are sourced from
   * IBZ-stored W/Z through the vertex_sym context (effective collocations +
   * PQ-transpose for time-reversal-mapped transfers). The C-window D-matrix
   * leakage of the symmetry rotations is measured and logged (sym_leakage_max).
   *  - Sigma^C: fused G^3 W^2 double-bosonic-convolution kernel, DLR backend
   *    only (vertex_sigma.icc; notes/sigma_c_kernel_design.md)
   *  - Pi^C: G^4 W single-rung kernel (vertex_pi.icc; see its design notes)
   *
   * Usage (see MBPT_drivers.cpp, "gw" solver branch):
   *   vertex_t vertex(&ft, vertex_type, band_window, mf->nbnd());
   *   if (vertex.enabled()) { scr_eri.set_vertex(&vertex); gw.set_vertex(&vertex); }
   */
  class vertex_t {
  public:
    template<nda::MemoryArray Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;

  public:
    /**
     * @param ft            - [INPUT] imaginary-axis Fourier transform (IAFT) grids
     * @param vertex_type   - [INPUT] type of the vertex correction.
     *                        {choices: "none", "2nd_exchange"}
     * @param band_window   - [INPUT] contiguous orbital range [first, last) defining
     *                        the subspace C. An empty range means C = empty set.
     * @param nbnd          - [INPUT] number of bands in the primary basis
     *                        (used to validate band_window)
     * @param div_treatment - [INPUT] q->0 policy on the rung transfers (both kernels;
     *                        notes/q0_head_treatment.md section 3):
     *                        "ignore_g0" (default): include the q = Gamma cell of the
     *                          rung sums with the STORED regularized W(Gamma)
     *                          (v(G=0) is zeroed at ERI build time), no analytic
     *                          head -- the exact analogue of GW's "ignore_g0".
     *                        "gygi" (or any string containing "gygi"): additionally
     *                          add the analytic rank-1 head insertion at Gamma,
     *                          dW_PQ(Gamma,tau) += Nk*madelung*Re[eps_inv_head(tau)]
     *                          *conj(chi_P)chi_Q (+ the bare piece with factor 1 into
     *                          Z(Gamma)) -- the GW Sigma_div_correction / HF
     *                          K-correction analogue on the vertex rungs.
     *                        "v1_skip": the v1 blanket skip of the whole Gamma cell
     *                          on every rung transfer (kept selectable for
     *                          comparability; NOT equivalent to GW's ignore_g0 --
     *                          it also drops the finite body term).
     * @param isdf_mode     - [INPUT] auxiliary basis of the vertex kernels
     *                        (Refinement 2, notes/refinement2_optionA.md):
     *                        "global" (default): the kernels run in the global THC
     *                          basis (dimension Np) -- the original path, untouched.
     *                        "secondary": a dedicated secondary ISDF basis on the
     *                          correlated subspace C replaces the global auxiliary
     *                          dimension (Np -> N_m) in both kernels, via the
     *                          frequency-independent Option-A transfer t(q) =
     *                          s(q)^+ B(q)^dag C(q) (theoryB Eq. 36): the rung cores
     *                          are DOWNFOLDED, Wbar = t W t^dag; Sigma^C is produced
     *                          directly in the C-C block (no upfold); Pi^C is
     *                          UPFOLDED with the adjoint of the same t (no-leak,
     *                          theoryB Eq. 39). Requires re-running the ISDF
     *                          point-selection on the restricted range (done lazily,
     *                          once per geometry).
     * @param isdf_rank     - [INPUT] secondary basis size N_m ("secondary" mode only).
     *                        -1 (default): the full subspace pair rank nc^2 * nkpts.
     *                        The point selection may return fewer points if the
     *                        pair-density metric is numerically rank-deficient; the
     *                        returned count is used and logged.
     * @param isdf_svd_tol  - [INPUT] relative SVD cutoff on the secondary pair
     *                        collocation B(q) in the truncated pseudo-inverse solve
     *                        for t(q) (the metric s = B^dag B is regularized at the
     *                        SQUARE of this value). Default 1e-8.
     * @param isdf_cond_max - [INPUT] per-q conditioning cap on the secondary downfold
     *                        ("secondary" mode only). <= 0 (default): disabled, the solve
     *                        uses isdf_svd_tol only (legacy behavior). > 0: the per-q
     *                        least-squares solve for t(q) truncates B(q)'s near-null
     *                        directions (gelss rcond = 1/sqrt(cond_max), floored by
     *                        isdf_svd_tol) so each transfer q's downfold is conditioned to
     *                        <= cond_max. NOTE: the ill-conditioning is q-specific (NOT at
     *                        Gamma) and the interpolating points are shared across q, so it
     *                        is bounded per q in the SOLVE, not by pruning the shared point
     *                        set -- pruning can only drop globally-redundant vectors, which
     *                        does not touch the worst q. eta(q,nu) certifies the accuracy.
     * @param rung          - [INPUT] rung mode of the active theory (vertex_rung_e above;
     *                        notes/static_vertex_implementation_plan.md section 2.1):
     *                        "dynamic" (default; Formulation B, bit-identical to the
     *                        historic path), "static" (B-S) or "linear" (B-L). The static
     *                        modes' kernels arrive at increment S2+, so requesting one with
     *                        a non-empty C aborts here; C = empty set is a no-op in every
     *                        mode.
     */
    vertex_t(const imag_axes_ft::IAFT *ft,
             std::string vertex_type,
             nda::range band_window,
             long nbnd,
             std::string div_treatment = "ignore_g0",
             std::string isdf_mode = "global",
             long isdf_rank = -1,
             double isdf_svd_tol = 1e-8,
             double isdf_thresh = -1.0,
             double isdf_cond_max = -1.0,
             std::string rung = "dynamic");

    vertex_t(vertex_t const&) = default;
    vertex_t(vertex_t &&) = default;
    vertex_t& operator=(const vertex_t &) = default;
    vertex_t& operator=(vertex_t &&) = default;

    ~vertex_t() {}

    /**
     * Evaluate the self-energy cut Sigma^C (G^3 W^2) and accumulate it into
     * the dynamic self-energy of the MBState, on top of the GW self-energy:
     *   Sigma_tskij <- Sigma_tskij + Sigma^C_tskij
     *
     * Shapes are IBZ-resident: (nt_f, ns, nkpts_ibz, nbnd, nbnd).
     *
     * Precondition: active() == true. Callers must guard the call so the
     * disabled path performs no allocation and no arithmetic.
     *
     * @param mb_state - [INPUT/OUTPUT] MBState holding sG_tskij, dW_qtPQ and
     *                   the target sSigma_tskij
     * @param thc      - [INPUT] THC-ERI instance
     */
    void eval_Sigma_C(MBState &mb_state, THC_ERI auto const &thc);

    /**
     * Evaluate the polarizability cut Pi^C (G^4 W) as an ADDITIVE contribution
     * to the RPA polarizability, on the same distributed grid:
     *   Pi_tqPQ <- Pi_tqPQ + Pi^C_tqPQ   (the "+=" is done by the caller,
     *                                     following the EDMFT precedent in
     *                                     scr_coulomb_t::eval_Pi_qdep)
     *
     * Shapes are IBZ-resident: (nt_half, nqpts_ibz, Np, Np), distributed with
     * the same pgrid/bsize as the RPA Pi so it flows into dyson_W_in_place
     * untouched.
     *
     * Precondition: active() == true. Callers must guard the call so the
     * disabled path performs no allocation and no arithmetic.
     *
     * @param mb_state  - [INPUT] MBState holding sG_tskij
     * @param thc       - [INPUT] THC-ERI instance
     * @param pi_pgrid  - [INPUT] processor grid of the RPA Pi_tqPQ
     * @param pi_bsize  - [INPUT] block size of the RPA Pi_tqPQ
     * @param pi_gshape - [INPUT] global shape of the RPA Pi_tqPQ:
     *                    (nt_half, nqpts_ibz, Np, Np)
     * @return - Pi^C in the THC product basis: (nt_half, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_C(MBState &mb_state, THC_ERI auto const &thc,
                   shape_t<4> pi_pgrid, shape_t<4> pi_bsize, shape_t<4> pi_gshape)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    /**
     * Refinement 2 W-bar iteration cache (secondary path only; notes/wbar_cache.md).
     *
     * Folds the CURRENT dynamic screened interaction mb_state.dW_qtPQ into the
     * N_m x N_m secondary basis and stores it:
     *   Wbar(q, nu) = t(q) [dW(q, tau) -> PH-sym Matsubara] t(q)^dag,
     * with the gygi head augmentation of dW(Gamma, tau) applied BEFORE the
     * transform/fold, using mb_state.eps_inv_head -- i.e. the eps_inv_head of the
     * SAME iteration as W (both are written by the same scr_coulomb_t::update_w
     * call, whose tail invokes this).
     *
     * The cache is consumed by the NEXT iteration's eval_Pi_C in place of the
     * retained mb_state.dW_qtPQ (identical one-iteration lag; the scf driver then
     * frees dW unconditionally -- plain-GW memory profile). The arithmetic is
     * IDENTICAL to the legacy fold-at-consumption path: same data, same transform,
     * same fold order -- results are machine-identical (memo section 2).
     *
     * Collective on thc.mpi()->comm. Precondition: active() and secondary() and
     * mb_state.dW_qtPQ present.
     */
    void cache_w(MBState &mb_state, THC_ERI auto const &thc);

    /**
     * INCREMENT S2 -- build the statically screened rung W0[G] of the B-S/B-L theories
     * (notes/static_vertex_implementation_plan.md section 2.2, decision D2):
     *
     *   W0(q) = [1 - v P^0_RPA[G]]^{-1} v  at  i.nu = 0   =   Z(q) + dW(q, i.nu = 0)
     *
     * evaluated on the SAME-ITERATION RPA polarizability, with NO iteration lag: the
     * caller (scr_coulomb_t::eval_Pi_qdep) hands over Pi_RPA(q, tau) at the point where
     * it has been assembled and BEFORE any Pi^C / P^{C,L} is added, so this is exactly
     * W0 of the current G (in B-S it coincides with the i.nu = 0 slice of the run's own
     * W -- the self-slice identity the S2 gate pins to machine precision; in B-L it
     * deliberately does NOT, since the run's W carries P^{C,L}).
     *
     * Steps (each one the pinned machinery restricted to a single frequency):
     *   1. i.nu = 0 row of Pi_RPA (vertex_w0_detail::extract_nu0_row + the verified
     *      static-slice convention nu0_transform_row) onto the {1, 1, nP, nQ} layout;
     *   2. one-frequency THC Dyson dW0 = ([I - Z.Pi0]^{-1} - I) Z per q -- the
     *      scr_coulomb_t::dyson_W_in_place algebra with the frequency loop removed;
     *   3. the q->0 HEAD policy at i.nu = 0 (notes/q0_head_treatment.md section 3, one
     *      policy for one W0): "v1_skip" and "ignore_g0" store the regularized body,
     *      gygi-class additionally inserts the analytic rank-1 head
     *      Nk*xi_M*[1 + Re eps_inv_head(i.nu = 0)]*chi chi^dag at Gamma, with the
     *      i.nu = 0 head factor extracted from the FRESH RPA dW0 itself
     *      (div_utils::eps_inv_head_w) -- so the rung and its head factor carry the
     *      same iteration tag by construction (memo section 1.6);
     *   4. W0bar(q) = t(q) W0(q) t(q)^dag through the existing DISTRIBUTED fold
     *      (vertex_secondary_detail::fold_Z_distributed -- the strictly cheaper one-row
     *      sibling of the dW fold: no tau axis, no tau->nu, no PH-unfold).
     *
     * Storage (section 3 table): W0 stays (P,Q)-block-distributed (nq*Np^2 is a
     * 320 GB-class object at production and is NEVER replicated or gathered); W0bar is
     * the replicated MEDIUM (nq, N_m, N_m) array the kernels consume. In the GLOBAL-aux
     * reference path (isdf_mode = "global", small scale only) N_m == Np and W0bar is the
     * gathered W0 -- same class as that path's existing replicated Z_qPQ.
     *
     * ITERATION-LOCAL: nothing crosses the iteration boundary. Both objects are dropped
     * (reset_w0) at the top of the next build, and reset_w0 is public so the driver /
     * the S3+ consumers can release them earlier.
     *
     * Collective on thc.mpi()->comm. Precondition: active(). Independent of rung() --
     * the MODE gate lives at the update_w seam (needs_w0), so this builder is directly
     * unit-testable and is what the S3+ static kernels will consume.
     *
     * @param mb_state      - [INPUT/OUTPUT] MBState (G, and the head data)
     * @param thc           - [INPUT] THC-ERI (Z, basis_head, basis_bar_head)
     * @param dPi_rpa_tqPQ  - [INPUT] RPA-ONLY polarizability, global (nt_half, nq, Np, Np)
     */
    template<typename dArray_t>
    void build_w0(MBState &mb_state, THC_ERI auto const &thc,
                  dArray_t const &dPi_rpa_tqPQ);

    /**
     * Install the general Wannier projector U(s,k) from a projector_t (WANNIER
     * MODE; notes/wannier_projector_theory.md section 0, P1). The subspace C
     * becomes span{ |w_a(k)> = sum_i U_ia(k)|psi_i(k)> }, U an Norb x M isometry
     * built as U = dagger(proj_mat) on the rows W_rng (zero elsewhere), M =
     * nImpOrbs. The window-mode _band_window is replaced by W_rng (the injection
     * support). U is FIXED for the whole SCF loop (demand D1) -- call once at
     * construction time, before the scf loop.
     *
     * Owner ruling Q1: U is Loewdin-orthonormalized per (s,k) so U^dag U = 1_M
     * exactly; the correction norm ||U^dag U - 1|| is measured and logged BEFORE
     * orthonormalization. loewdin = false skips it (warn + proceed with raw U;
     * P then only approximately idempotent, memo section 1.3).
     *
     * @param proj - [INPUT] projector_t carrying proj_mat + band_window from wan.h5
     * @param loewdin - [INPUT] Loewdin-orthonormalize U at load (default true)
     */
    void set_wannier_projector(methods::projector_t const &proj, bool loewdin = true);

    // WANNIER MODE predicate: a general U has been installed (window mode = false)
    bool wannier() const { return _wannier; }
    // subspace rank M (= _band_window.size() in window mode, = _U.shape(3) in
    // Wannier mode); the auxiliary orbital dimension both kernels run on
    long subspace_rank() const {
      return _wannier ? _M : _band_window.size();
    }
    // measured isometry defect max_sk ||U^dag U - 1_M||_F before orthonormalization
    // (0 in window mode; owner ruling Q1 diagnostic)
    double isometry_defect() const { return _iso_defect; }
    // path to the wan.h5 the projector was built from ("" in window mode); used to
    // enforce the shared-object demand D2 against a coexisting embedding projector
    std::string wannier_file() const { return _wannier_file; }

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;

    // type of the vertex correction: "none" or "2nd_exchange"
    std::string _vertex_type = "none";

    // rung mode of the active theory (vertex_rung_e): one mode drives ALL cuts
    // (notes/static_vertex_implementation_plan.md section 2.1)
    vertex_rung_e _rung = dynamic_rung;

    // contiguous orbital range [first, last) defining the subspace C. In
    // WINDOW MODE this is C itself; in WANNIER MODE it is the injection support
    // W_rng (the band range spanned by the M Wannier orbitals), while the
    // subspace rank is M <= _band_window.size().
    nda::range _band_window = nda::range(0, 0);

    // ---- WANNIER MODE: general fixed projector P(k) = U(k) U(k)^dag ------------------
    // (notes/wannier_projector_theory.md). Empty (_wannier = false) => WINDOW MODE:
    // the trivial 0/1 column-selection isometry, dispatched to the existing slice
    // code so window-mode results stay BIT-IDENTICAL.
    bool _wannier = false;
    long _M = 0;                            // subspace rank (columns of U)
    // U(s, k) as an Norb(=W_rng.size()) x M isometry on the W_rng rows, restricted to
    // the injection support (rows outside W_rng are structurally zero and dropped):
    // _U_skia(is, ik, i, a) = U_{(W_rng.first()+i), a}(s, k), U^dag U = 1_M (Loewdin).
    // k axis is FULL BZ (per-k projector; demand D4).
    nda::array<ComplexType, 4> _U_skia;
    // measured max_sk ||U^dag U - 1_M||_F before Loewdin (owner ruling Q1 diagnostic)
    double _iso_defect = 0.0;
    // wan.h5 the projector was built from (shared-object demand D2)
    std::string _wannier_file;

    // q->0 policy on the rung transfers: "ignore_g0" (v2 default), "gygi"-class,
    // or "v1_skip" (the v1 blanket Gamma-skip fallback). See the constructor doc
    // and notes/q0_head_treatment.md.
    std::string _div_treatment = "ignore_g0";

    // ---- Refinement 2: secondary ISDF basis (notes/refinement2_optionA.md) ----------
    // "global" (default) or "secondary"
    std::string _isdf_mode = "global";
    // requested secondary rank N_m (-1 = full subspace pair rank nc^2 * nkpts)
    long _isdf_rank = -1;
    // relative SVD cutoff on B(q) in the truncated pseudo-inverse for t(q)
    double _isdf_svd_tol = 1e-8;
    // secondary-ISDF point-selection threshold override (-1 = default to the global
    // THC thresh, thc.thresh()). A tighter value than the global thresh pushes the
    // selected interpolating vectors outside the global-basis span -> ill-conditioned.
    double _isdf_thresh = -1.0;
    // per-q conditioning cap on the secondary downfold; <= 0 disables it (the solve uses
    // isdf_svd_tol only). > 0 sets the per-q gelss rcond = 1/sqrt(cond_max). See ctor doc.
    double _isdf_cond_max = -1.0;
    // geometry-fixed cache (built lazily on the first kernel evaluation)
    bool _secondary_ready = false;
    long _Nm = 0;                          // ACTUAL secondary rank (selection may
                                           // return fewer points than requested)
    double _cond_s_max = 0.0;              // max_q REGULARIZED downfold conditioning
                                           // (bounded by the cond cap; diagnostic)
    nda::array<ComplexType, 4> _Xb_skma;   // secondary collocation (ns, nk, N_m, nc)
    nda::array<ComplexType, 3> _t_qmP;     // Option-A transfer t(q): (nq, N_m, Np)

    // ---- W-bar iteration cache (secondary path; notes/wbar_cache.md) ----------------
    // Downfolded dynamic rung Wbar(q, nu >= 0): (nq_ibz, nw_half, N_m, N_m), filled by
    // cache_w at the scr_coulomb_t::update_w tail, consumed by the NEXT iteration's
    // eval_Pi_C (nu < 0 reconstructed there via the PH mirror W(-nu) = W(nu)).
    // SYMMETRY EXTENSION HOOK: the cache is KEYED BY q on the first axis. Under the
    // current nosym restriction nq == nq_ibz; when IBZ symmetry lands, this axis
    // becomes IBZ-q and reads at symmetry-related q go through the auxiliary-basis
    // rotation (the same unfolding point as the kernels' planned IBZ support) -- no
    // layout change needed, only the accessor.
    std::optional<nda::array<ComplexType, 4>> _Wb_qwmm;
    // internal/test switch: when false, cache_w is never invoked by scr_coulomb and
    // the scf driver retains dW (needs_dw_retention), so eval_Pi_C takes the legacy
    // fold-at-consumption branch -- the pre-cache behavior, kept as the permanent
    // machine-identity A/B reference (not exposed as an input key).
    bool _w_cache_enabled = true;

    // ---- INCREMENT S2: the static-vertex W0[G] rung (plan section 2.2/3) -------------
    // W0(q) = Z(q) + dW(q, i.nu = 0) of the SAME-ITERATION RPA polarizability, with the
    // q->0 head policy applied. ITERATION-LOCAL: built inside update_w (before any Pi^C
    // is added), consumed within the same iteration, dropped at the next build.
    //   _W0_qPQ  : global aux, (nq_ibz, Np, Np) (P,Q)-BLOCK-DISTRIBUTED on the
    //              {1, nP, nQ} grid (q unsplit, nP*nQ == comm.size() -- the
    //              thc.dZ({1,nP,nQ}) layout the production Z fold already uses).
    //              320 GB-class at production: never replicated, never gathered.
    //   _W0b_qmm : the downfolded rung W0bar = t W0 t^dag, (nq_ibz, N_m, N_m),
    //              replicated MEDIUM (~0.2 GB at production). In the GLOBAL-aux
    //              reference path N_m == Np and this is the gathered W0 (small scale
    //              only, same class as that path's replicated Z_qPQ).
    std::optional<memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator> > _W0_qPQ;
    std::optional<nda::array<ComplexType, 3> > _W0b_qmm;
    // i.nu = 0 head factor Re[eps^{-1}_head - 1] of the RPA-only W0 (0 unless a
    // gygi-class policy actually inserted a head) and the rank-1 weight that was
    // applied at Gamma, c = N_k * madelung. Diagnostics + the S2 head-policy gate.
    double _w0_eps_head = 0.0;
    ComplexType _w0_head_c = ComplexType(0.0);
    bool _w0_head_applied = false;

    // ---- IBZ k-point symmetry (notes/vertex_ibz_symmetry.md) -------------------------
    // Geometry-fixed symmetry contexts, built lazily on the first symmetric
    // evaluation: q'-access tables + effective C-window collocation columns Xhat
    // for the global (Np) and secondary (N_m) bases. Trivial (unused) on
    // symmetry-free meshes -- the kernels then take their historic paths.
    std::optional<vertex_sym::sym_ctx> _sym_global;
    std::optional<vertex_sym::sym_ctx> _sym_secondary;
    // measured C-window D-matrix leakage (diagnostic, no gate; memo section 6)
    double _sym_leak_max = 0.0;
    double _sym_leak_mean = 0.0;
    // measured unitarity defect max ||Dc^dag Dc - 1||_F of the C-sector symmetry
    // rotation Xhat is built from -- the accuracy floor of the symmetry path.
    double _sym_d_unitarity_max = 0.0;
    // VERTEX RAMP: Phi_2^C -> lambda Phi_2^C. _scale is the target strength, _ramp_iters
    // the number of scf iterations over which lambda is walked from 1/_ramp_iters up to
    // _scale (0 = no ramp, lambda = _scale immediately). _vertex_iter counts eval_Pi_C
    // calls = scf iterations with an active vertex.
    // number of times eval_Pi_C had to fall back to the BARE rung. Must stay 0 in any
    // scf loop that goes through scr_coulomb_t::update_w (which bootstraps an RPA W).
    long _bare_rung_uses = 0;
    double _scale = 1.0;
    long _ramp_iters = 0;
    long _vertex_iter = 0;
    // measured G_CC G-rotation consistency residual, running max over this vertex's
    // eval_Pi_C / eval_Sigma_C calls (diagnostic, no gate; memo section 6). Distinct
    // from the iteration-independent D-matrix leakage above: this one tracks whether
    // the self-consistent G_CC itself stays symmetry-consistent across iterations.
    double _g_rot_max = 0.0;

    /**
     * Build (lazily) the symmetry context for the given window collocation
     * X_w (ns, nk_full, naux, nc): q'-access tables, krot = ks_to_k, effective
     * columns Xhat per (spin, qsymms position, k), and the C-window leakage
     * diagnostic. Collective-safe (pure local reads of MF tables + X_w).
     *
     * WANNIER MODE (U_skia != nullptr; notes/wannier_projector_theory.md section 2.8):
     * the C-sector rotation becomes the M x M Wannier rotation
     * d(k;S) = U(Sk)^dag D_win(k;S) U(k) (D_win = the W_rng band block of the MF
     * rotation), so sym + Wannier compose through the SAME Xhat path; the leakage
     * diagnostic is then the projector-level ||(1 - P(Sk)) D U(k)|| and goes to 0
     * by construction for a symmetry-closed Wannier set. C0_global is W_rng.first()
     * and nc = M; X_w = X_bar (the rotated collocation). Window mode = nullptr U.
     */
    void build_sym_ctx(THC_ERI auto const &thc,
                       nda::MemoryArrayOfRank<4> auto const &X_w,
                       long C0_global,
                       std::optional<vertex_sym::sym_ctx> &slot,
                       nda::array<ComplexType, 4> const *U_skia = nullptr);

    /**
     * Build the secondary ISDF basis and the per-q Option-A transfer maps
     * (lazily; no-op once built). Collective on thc.mpi()->comm.
     *   - restricted point selection: thc::interpolating_points(iq_gamma, N_m, C, C)
     *     on a private methods::thc builder (pivoted Cholesky on the C pair-density
     *     metric; greedy importance order).
     *   - per q: B(q)/C(q) pair-collocation matrices (pair rows I = (is, ik, o, i),
     *     k_in = k - q; the kernels' in/out collocation rule), t(q) from the
     *     truncated-SVD least-squares solve min || B t - C ||_F.
     * cond(s), effective rank and discarded singular values are logged.
     *
     * @param X_glob - [INPUT] the GLOBAL collocation the secondary basis fits against:
     *                 WINDOW mode = the full-band replicated collocation (ns, nk, Np,
     *                 nbnd) with orb0 = C.first(); WANNIER mode = the rotated collocation
     *                 X_bar = X.U (ns, nk, Np, M) with orb0 = 0.
     * @param orb0   - [INPUT] first subspace column of X_glob (C.first() / 0)
     * @param kmq    - [INPUT] (nq, nk) index map of k - q
     * @param iq_gamma - [INPUT] index of q = Gamma
     */
    void build_secondary_basis(THC_ERI auto const &thc,
                               nda::MemoryArrayOfRank<4> auto const &X_glob, long orb0,
                               nda::array<long, 2> const &kmq, long iq_gamma);

    /**
     * INCREMENT S2 helper: the collocation / momentum-map / Gamma-index preamble
     * build_secondary_basis needs, packaged so build_w0 can call it from inside
     * update_w -- where (unlike eval_Pi_C / cache_w) no kernel has run yet and the
     * lazy basis therefore does not exist. Idempotent: build_secondary_basis returns
     * immediately once _secondary_ready. Deliberately NOT refactored out of eval_Pi_C /
     * cache_w: those are pinned bit-identity paths of the dynamic theory.
     *
     * @param mb_state - [INPUT] MBState (ns comes from G)
     * @param thc      - [INPUT] THC-ERI
     * @return - the q = Gamma index (also needed by the head insertion / the fold)
     */
    long ensure_secondary_basis(MBState &mb_state, THC_ERI auto const &thc);

    /**
     * Abort unless the ACTIVE rung mode has KERNELS (the S1 guard, relocated at S2).
     *
     * S1 placed this in the constructor, because nothing of the static theories existed.
     * S2 lands the shared W0[G] rung infrastructure, which is built inside update_w and
     * is mode-agnostic; the pieces that are still missing are the KERNELS. The guard
     * therefore moved to the two kernel entry points (eval_Sigma_C, eval_Pi_C), which is
     * where the absence actually bites: an end-to-end static/linear run still aborts
     * inside its FIRST iteration (static: at the Sigma^C evaluation, since the plan's
     * section 2.1 turns the Pi^C injection off for B-S; linear: already at the P^{C,L}
     * injection inside update_w), so no static run can produce numbers -- while the W0
     * build seam is directly exercisable. Removed mode by mode as S3+ lands the kernels.
     *
     * @param where - [INPUT] call site, used verbatim in the abort message
     */
    void check_rung_implemented(std::string_view where) const;

    /**
     * Imaginary-axis backend requirement of the ACTIVE rung mode. Every mode is
     * DLR-only today, but for different reasons, so the abort message is routed
     * through the mode switch: dynamic_rung needs the exact DLR pole algebra of the
     * G^3W^2 double convolution; the static modes need no pole algebra at all, only
     * the Pi^{C,0}(tau = 0) interpolation row, whose IR availability is decision D3
     * (open until increment S4).
     *
     * @param where - [INPUT] call site, used verbatim in the abort message
     */
    void check_iaft_backend(std::string_view where) const;

  public:
    std::string vertex_type() const { return _vertex_type; }
    // rung mode of the active theory (section 2.1 of the static-vertex plan)
    vertex_rung_e rung() const { return _rung; }
    std::string rung_str() const { return vertex_rung_enum_to_string(_rung); }
    nda::range band_window() const { return _band_window; }
    std::string div_treatment() const { return _div_treatment; }
    // runtime-selectable q->0 policy (validated; see constructor doc)
    void set_div_treatment(std::string div);

    // Refinement 2 accessors
    std::string isdf_mode() const { return _isdf_mode; }
    bool secondary() const { return _isdf_mode == "secondary"; }
    // ACTUAL secondary rank N_m (0 until the basis has been built)
    long secondary_rank() const { return _Nm; }
    // Option-A transfer maps t(q): (nq_ibz, N_m, Np), empty until the basis is built.
    // Read-only; the S2 gate needs it to form the replicated t W0 t^dag reference.
    nda::array<ComplexType, 3> const& secondary_transfer() const { return _t_qmP; }

    // ---- INCREMENT S2: W0[G] accessors (plan section 2.2) ---------------------------
    // Does the ACTIVE theory need the static rung? dynamic (Formulation B) does not;
    // B-S and B-L both do. This is the ONLY mode gate on the W0 build -- build_w0
    // itself is mode-agnostic infrastructure.
    bool needs_w0() const { return active() and _rung != dynamic_rung; }
    bool has_w0() const { return _W0_qPQ.has_value(); }
    // ITERATION-LOCAL lifetime: drop both objects. Called at the top of every build and
    // exposed so the driver / the S3+ consumers can release them as soon as they are done.
    void reset_w0() {
      if (_W0_qPQ.has_value()) _W0_qPQ.value().reset();
      _W0_qPQ.reset();
      _W0b_qmm.reset();
      _w0_eps_head = 0.0;
      _w0_head_c = ComplexType(0.0);
      _w0_head_applied = false;
    }
    // global-aux W0, (P,Q)-block-distributed (nq_ibz, Np, Np)
    memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator> const& W0_qPQ() const {
      utils::check(_W0_qPQ.has_value(),
                   "vertex_t::W0_qPQ: the static rung W0 has not been built this "
                   "iteration (build_w0 runs inside scr_coulomb_t::update_w).");
      return _W0_qPQ.value();
    }
    // downfolded rung W0bar = t W0 t^dag, replicated (nq_ibz, N_m, N_m)
    nda::array<ComplexType, 3> const& W0bar_qmm() const {
      utils::check(_W0b_qmm.has_value(),
                   "vertex_t::W0bar_qmm: the downfolded static rung W0bar has not been "
                   "built this iteration (build_w0 runs inside scr_coulomb_t::update_w).");
      return _W0b_qmm.value();
    }
    // i.nu = 0 head factor Re[eps^{-1}_head - 1] of the RPA-only W0 (0 unless a
    // gygi-class policy inserted a head) and the applied rank-1 prefactor N_k*madelung.
    double w0_eps_inv_head() const { return _w0_eps_head; }
    ComplexType w0_head_c() const { return _w0_head_c; }
    bool w0_head_applied() const { return _w0_head_applied; }
    // "v1_skip" fallback: the Gamma cell of the rung transfer is dropped BY THE KERNEL
    // (as for Z / dW -- the stored W0(Gamma) is the regularized body either way). Kept
    // as a flag on the handle so the S3+ kernels inherit the policy from the ONE W0.
    bool w0_skip_gamma() const { return _div_treatment == "v1_skip"; }

    // W-bar iteration cache accessors (notes/wbar_cache.md)
    bool has_cached_w() const { return _Wb_qwmm.has_value(); }
    void reset_w_cache() { _Wb_qwmm.reset(); }
    // legacy/compat switch (see the _w_cache_enabled comment); disabling also drops
    // any cached data so the next eval_Pi_C takes the retained-dW branch
    void set_w_cache_enabled(bool on) {
      _w_cache_enabled = on;
      if (not on) reset_w_cache();
    }
    bool w_cache_enabled() const { return _w_cache_enabled; }

    // vertex requested in the input
    bool enabled() const { return _vertex_type != "none"; }
    // vertex requested AND C is non-empty; C = empty set must be an exact no-op.
    // In WANNIER MODE C is non-empty iff M > 0 (the projector has columns).
    bool active() const {
      return enabled() and _band_window.size() > 0 and (not _wannier or _M > 0);
    }

    // IBZ symmetry diagnostics (notes/vertex_ibz_symmetry.md section 6):
    // measured C-window D-matrix leakage of the symmetry rotations (0 until the
    // first symmetric evaluation; 0 on symmetry-free meshes).
    double sym_leakage_max() const { return _sym_leak_max; }
    double sym_d_unitarity_max() const { return _sym_d_unitarity_max; }

    // Phi-scaling controls. Setting BOTH cuts by the same factor is Phi_2^C ->
    // lambda Phi_2^C, so conservation (Tr[Sigma^C G] + Tr[P^C W] = 0) is exact at
    // every lambda -- the approximation acts on Phi, never on the cuts.
    void set_vertex_scale(double s, long ramp_iters = 0) {
      utils::check(s >= 0.0, "vertex_t::set_vertex_scale: lambda must be >= 0 (got {}).", s);
      utils::check(ramp_iters >= 0,
                   "vertex_t::set_vertex_scale: ramp_iters must be >= 0 (got {}).",
                   ramp_iters);
      _scale = s;
      _ramp_iters = ramp_iters;
    }
    // lambda in force for the CURRENT scf iteration
    double vertex_scale() const {
      if (_ramp_iters <= 0) return _scale;
      const long n = std::max(1l, _vertex_iter);
      return _scale * std::min(1.0, double(n) / double(_ramp_iters));
    }
    long vertex_iter() const { return _vertex_iter; }
    long bare_rung_uses() const { return _bare_rung_uses; }
    double sym_leakage_mean() const { return _sym_leak_mean; }
    // running max of the G_CC G-rotation consistency residual across this vertex's
    // eval calls (0 until the first symmetric evaluation; 0 on symmetry-free meshes).
    double g_rotation_max() const { return _g_rot_max; }
    // secondary-basis diagnostics (0 until build_secondary_basis has run):
    long secondary_Nm() const { return _Nm; }                    // actual secondary rank N_m
    double secondary_cond_s_max() const { return _cond_s_max; }  // max_q regularized downfold cond

  }; // vertex_t

} // solvers
} // methods

#endif //COQUI_VERTEX_T_H
