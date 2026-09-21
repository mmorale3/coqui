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

#include <map>
#include <tuple>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
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
   * All three modes are fully implemented (plan increments S0-S10 complete, plus IBZ
   * symmetry); check_rung_implemented is retained as a no-op seam.
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
     *                        historic path), "static" (B-S) or "linear" (B-L). All three
     *                        modes are fully implemented; C = empty set is a no-op in
     *                        every mode.
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

    /** W-int-0: copy the installed MLWF state (U, M, W_rng) from another already-Wannierized vertex.
     *  scr_coulomb's private readout instance (which actually runs the pol-vertex/dynbse) uses this to
     *  inherit the projector set on the user's vertex, since set_wannier_projector needs the projector_t
     *  object (not just U). Same-class access to src's privates. No-op semantics: only call when src is
     *  Wannier; the readout instance is created "2nd_exchange"-enabled so active() holds after adoption. */
    void adopt_wannier(vertex_t const &src) {
      utils::check(src._wannier and src._M > 0, "vertex_t::adopt_wannier: source vertex is not in Wannier mode.");
      _wannier = src._wannier; _M = src._M; _U_skia = src._U_skia; _band_window = src._band_window;
      _wannier_file = src._wannier_file; _iso_defect = src._iso_defect;
    }

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
    // W-int-1b (notes/wannier_coarse_vertex_plan.md): the coarse->fine interpolation knobs
    std::string _isdf_points_file;      // pol_vertex_isdf_points_file: FREEZE the secondary ISDF points from this file
    bool _isdf_points_dump = false;     // pol_vertex_isdf_points_dump: write <prefix>.secpts.h5 after the selection
    nda::array<long, 1> _sec_ipts;      // the secondary points in use (density-FFT-grid indices)
    std::string _wannier_frame = "aux"; // pol_vertex_wannier_frame: the dynbse output frame in Wannier mode ("aux" | "pair")
    std::string _pol_interp_file;       // pol_vertex_interp_file: the eps readout consumes this Pi(q)_{MN} (frozen points)
    std::string _pol_interp_col = "gam1";
    std::string _run_prefix;            // mb_state.coqui_prefix, captured at ensure_secondary_basis for the dumps

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
    // P19 (vertex_perf_plan.md, 2026-09-21): pol_vertex_wcache = "shared" keeps ONE copy of the cache per NUMA node
    // (math::shm::shared_array, the Xhat_shm pattern of vertex_sym) instead of one per rank; every consumer reads the
    // cache through wb_cache(), a view onto whichever storage is in use. "replicated" (default) = the historic per-rank array.
    std::string _wcache = "replicated";
    std::shared_ptr<math::shm::shared_array<nda::array_view<ComplexType, 4>>> _Wb_shm;
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
    // DA D-7 / H1b: the PRE- vs POST-FOLD head meter of the LAST build_w0
    // (notes/qsgwhat_discrepancy_spec.md Phase 2; theory notes section 7 hazard). The head
    // is rank-1, H = c_eff chi chi^dag with c_eff = c (1 + Re eps^-1_head), so
    // ||H||_F = |c_eff| ||chi||^2 in the GLOBAL basis and |c_eff| ||t chi||^2 after the
    // Option-A fold. What is reported is each one's SHARE of its own rung's Frobenius norm,
    // and the ratio of the two shares = the attenuation the secondary basis applies to the
    // head relative to the body. -1 = never measured (meter off / no head).
    double _w0_head_share_pre = -1.0;
    double _w0_head_share_post = -1.0;
    double _w0_head_atten = -1.0;

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
    // B-L's pi^dyn route (eq:pibardynfact, notes/pibardynfact_increment.md). 0 =
    // FACTORIZED (default): the single-bosonic-pairing primitive, no pole algebra.
    // 1 = KERNEL: the historic route -- the full dynamic-rung Pi^C over all nw_b
    // frequencies, of which only the tau = 0 row is kept (98.9 % of B-L's vertex time,
    // and B-L's only contact with the aux pole basis). 2 = CHECK: run BOTH and log the
    // deviation, aborting past _pidyn_check_tol. Kept reachable because a production-scale
    // disagreement is the one thing the toy gate (test_vertex_pibardynfact) cannot see.
    int _pidyn_mode = 0;
    // CHECK-mode ABORT bar; <= 0 uses 0.25. Deliberately an O(1) ROUTING bar and NOT tied to
    // the DLR tolerance: the two routes are exact Matsubara sums of different integrands read
    // through the same tau = 0 row, so their agreement floor is a representability floor whose
    // prefactor grows with beta*wmax (MEASURED ~30*eps at 160, ~2000*eps at 6000) AND is data
    // dependent (LiH-222 at prec = "low": 3.6e-03 in scf iteration 1, 2.1e-02 in iteration 2).
    // Any eps-derived abort is flaky by construction. Exceeding max(1e-8, 100*eps) warns
    // instead -- that IS the actionable statement, since the floor bounds pi^dyn by EITHER
    // route. The check's real discriminating power is against a routing/plumbing break, and
    // the closest mis-routing the routing pin rejects sits at 1.24.
    double _pidyn_check_tol = -1.0;
    double _pidyn_check_max = 0.0;   // running max relative deviation seen in CHECK mode
    // Project the rank-1 head channel chi chi^dag out of the RESPONSE middle factor at
    // q = Gamma before the W0 . Pi . W0 sandwich.
    //
    // 🚨 OFF BY DEFAULT SINCE 2026-07-31: IT BREAKS PHI-DERIVABILITY. The B-L G-side
    // oracle (test_vertex_fdoracle.cpp, "HEAD-PROJECTION control") measures the identity
    //     dPhi/dl = T[Sigma^(C,L), P_C dG P_C] + T[Sigma^(L,r), dG]      (eq:eulerBL1)
    // and removing just 20.35 % of max|Pi^L(Gamma)| in the head channel takes its residual
    // from 3.348e-11 to 1.559e-01 -- WORSE than the untransposed-sandwich control (1.093e-01)
    // that the same test exists to reject, and 1560x its own 1e-4 gate. On Si the projection
    // removes 66 %, so the real violation is larger still.
    //
    // The reason is structural, not numerical. The head INSERTION is legal because it
    // modifies Phi (via What) and both cuts then follow by differentiation
    // (notes/head_corrections.pdf section 6). This projection does the opposite: it deletes a
    // channel from an ALREADY-CUT object, and only on the Sigma side -- eval_Pi_C's
    // P^{C,L}, which feeds the Dyson equation, keeps its head channel. That is exactly the
    // pattern CLAUDE.md section 2.1 / section 12 forbid ("apply approximations to Phi, then
    // differentiate; approximating the already-cut Sigma/P breaks conservation").
    //
    // What it empirically DOES do (notes/bl_head_channel_diagnosis.md): it controls the
    // COLD-START basin -- cold + projection off diverges, cold + on stays bounded. It is NOT
    // what makes B-L converge (a restart from a converged checkpoint converges either way),
    // and it moves the converged e_corr by 1.17e-02, i.e. 1.6x the whole B-S vertex
    // correction, creating a SECOND fixed point. Use it only as a diagnostic; the cold-start
    // divergence needs a remedy that does not modify a cut (damping, or a head treatment
    // derived from Phi).
    bool _bl_head_projection = false;
    // DIAGNOSTIC (default OFF, not physical): freeze the Gamma rung head's frequency
    // dependence at its i.nu = 0 value inside pi^dyn's rung only. That nu-dependence is
    // the one structural difference between pi^dyn's rung (W(i.nu), head weight running
    // from eps^-1(0) to 1) and Pi^{C,0}'s (W0, head frozen at eps^-1(0)) -- and Pi^{C,0}
    // satisfies the q->0 head suppression while pi^dyn does not. If <H, pi^dyn> collapses
    // under this knob, the head insertion's retardation is what breaks conservation.
    bool _bl_static_head = false;
    // DIAGNOSTIC (default OFF, changes the kernel DEFINITION): take W0's Gamma head
    // weight from the SAME eps^-1 that W's head uses (vertex-corrected) instead of from
    // W0's own RPA-only Dyson. B-L expands in dW = W - W0 with W0 RPA-static BY
    // DEFINITION, and the resulting head mismatch is 7.4 % of the head weight -- which is
    // 94 % of the whole measured |W(q,0) - W0(q)|, because the head is rank-1 and the
    // W0 . Pi . W0 sandwich amplifies that channel by c^2 ||chi||^4. Setting this makes
    // the head part of dW(Gamma, i.nu = 0) vanish while leaving W0's body RPA-static.
    // See the block at build_w0's head insertion for the measurements and the
    // one-iteration-lag caveat.
    bool _bl_w0_head_from_w = false;
    // H1, THE BALANCED FIRST-ORDER HEAD (default OFF pending Gate 0 + a defaults ruling;
    // notes/bl_head_balance_theory_and_plan.md section 4). In linear_rung (B-L) ONLY:
    // give EVERY W input of the vertex functional the SAME STATIC-weight Gamma head as
    // W0 -- the full static-screened head c*(1 + eps_inv_head(i.nu=0)) goes into the
    // INSTANTANEOUS (Z) slot, using build_w0's own _w0_eps_head so the weights match,
    // and NO dynamic-slot head is added. The fluctuation dW = W - W0 then carries no
    // analytic head at all (delta W_head == 0 up to FP association).
    // WHY: B-L is FIRST ORDER in dW, and the retarded head makes dW(Gamma) an O(1)
    // coherent rank-1 fluctuation in the chi channel (its weight sweeps
    // eps^-1(i.nu) - eps^-1(0), i.e. 0 -> 1 - eps^-1(0)); the tangent expansion is
    // outside its radius in exactly that channel, and the N_p^2-coherent kernel sum
    // turns it into the measured |S1+S2|/|S3| = 3.23 sign flip. The retarded head
    // content belongs to the theory that keeps the head-channel SECOND-order terms
    // (H2); at first order the static weight is the consistent choice.
    // CONSERVING: a modified interaction in Phi, both cuts differentiate (same
    // fixed-augmentation class as the insertion itself). B-S consumes only W0 and is
    // bit-identical; the parent (dynamic_rung) keeps its full retarded head.
    bool _bl_head_static_all = false;
    // distr_tol for build_secondary_basis' private thc builder; <= 0 = builder default
    double _isdf_distr_tol = -1.0;
    // ---- scGW-tilde ladder polarization (pol_vertex; notes/scgwt_implementation_plan.md
    // increments L1-L3, notes/scgw_screening_fix_proposal.pdf section 4.2) -------------
    // INDEPENDENT of the Phi-derivable vertex modes above: the ladder resums
    // density-channel static-W-bar_0 rungs in P ONLY (Sigma stays GW-form), so
    // Phi-derivability of the production loop is surrendered by construction (user
    // ruling 2026-08-10). "none" (default) = inert, bit-identical to the pre-scgwt tree.
    std::string _pol_vertex = "none";
    // Q3 (notes/q3_bse_tier_spec.md, ruling R-Q3-3): in-loop INJECTION of the ladder into
    // P. "none" (default) = the L2 report-only readout, bit-identical to the pre-Q3 tree;
    // "ladder_n2" = P_latt = P^RPA + P^lad with P^lad the resummed ladder (rungs >= 1 =
    // eq 6's [.]_{n>=2}, no subtraction). Auto-enables pol_vertex = "ladder".
    std::string _pol_vertex_inject = "none";
    // ladder kernel source (ruling R4): "w0_prev" = W-bar_0 from the previous iteration's
    // W (matches the static-rung convention); "w0_frozen" = RPA@KS W_0 (scGW_0-flavored).
    std::string _pol_kernel = "w0_prev";
    // ladder C-window + secondary-basis knobs. These store RESOLVED values: the driver
    // implements the "pol_vertex_* inherits vertex_*" default rule at parse time.
    nda::range _pol_band_window = nda::range(0, 0);
    long _pol_isdf_rank = -1;
    double _pol_isdf_svd_tol = 1e-8;
    double _pol_isdf_thresh = -1.0;
    double _pol_isdf_cond_max = -1.0;
    double _pol_isdf_distr_tol = -1.0;
    // ---- ladder solve geometry (notes/ladder_b_integration_design.md, increment B) -----
    // g = ranks per SOLVE GRID for the resummed ladder's dense resolvent.
    //   1 (default) = today's per-rank LAPACK path, bit-identical to the pre-B tree;
    //   > 1         = g ranks cooperate on one (s,q,nu) resolvent through SLATE;
    //   0           = AUTO, decided by the per-rank memory fit test against
    //                 _ladder_solve_budget_gb (fits => 1, else the smallest g that fits).
    long _ladder_solve_grid = 1;
    double _ladder_solve_budget_gb = 8.0;
    // ---- DA Phase 2 knobs (notes/qsgwhat_discrepancy_spec.md D-1 / D-4 / D-7) ----------
    // ALL THREE ARE DIAGNOSTICS, ALL DEFAULT-INERT (knob-absent = bitwise fallthrough).
    //
    // D-1 _ladder_tda: Tamm-Dancoff truncation of the ladder KERNEL. In the (hole at k,
    //   particle at k+q) orbital-pair space of vertex_ladder.icc the pairs split by
    //   OCCUPATION CHARACTER -- +1 "resonant" (a occupied at k, b empty at k+q), -1
    //   "anti-resonant" (a empty, b occupied), 0 otherwise (same-occupation pairs, whose
    //   chi0 weight vanishes for a QP-like G). The TDA is realized by ZEROING the kernel
    //   blocks that CONNECT characters of opposite sign, i.e. the resonant/anti-resonant
    //   COUPLING block B of H = [[A, B], [-B*, -A*]]. That is exactly the paper's
    //   "Hermitian matrix with half as many rows and columns" (Cunningham et al., PRB 108,
    //   165104, section III A): with B = 0 the two blocks decouple and each is the mirror
    //   of the other, so P^lad becomes the TDA polarization. DELETING the anti-resonant
    //   pairs instead would destroy the nu -> -nu symmetry of P and is NOT the TDA.
    bool _ladder_tda = false;
    // D-4 _ladder_head_scale: multiplies the ANALYTIC rank-1 q->0 head that build_w0
    //   inserts into the static rung W0(Gamma) -- i.e. the head INSIDE the ladder kernel
    //   W-bar_0, and nothing else (not the loop's own RPA W, not the Sigma^C/Pi^C head
    //   insertions, which are vertex_bl_head_scale). Finding F-DA-1
    //   (notes/qsgwhat_discrepancy_results.md section 4c) showed vertex_div_treatment never
    //   reaches this head, so its contribution to Delta_lad had never been measurable.
    //   1.0 = the committed policy (bitwise), 0.0 = head-free kernel.
    double _ladder_head_scale = 1.0;
    // D-7 _ladder_qnu_meter: the (q, nu) decomposition meters of the injected P^lad and of
    //   the Dyson-W change it drives, plus the pre/post-fold head meter (H1b). PURE
    //   OBSERVERS -- they add print-only arithmetic, never touch a physics array.
    bool _ladder_qnu_meter = false;
    // eps(q_i, i nu) cuts (2026-09-11): the number of transfers q_i (q_min plus evenly spaced
    // |q| ranks) at which the readout reports eps_M(q_i, i nu_j) on EVERY PH-sym bosonic half
    // node for every column it evaluates (RPA, ladder, Lambda legs, the loop's own head).
    // 0 = off (default, bitwise). Report-only.
    long _eps_cut_nq = 0;
    long _eps_cut_dyn_nnu = 0;   // rung = dynamic: evaluate the dynamic columns of the cut on the lowest n half
                                 // nodes only (0 = all); the cost driver of the cut (one dynamic solve per node x q)
    // scGW-tilde TIER 1.5 (notes/tier15_ward_legs_plan.md): the LEG VERTEX of the ladder's
    // pair propagators, "bare" (historic, bitwise) | "ward" (the discrete-Ward Lambda0).
    std::string _ladder_legs = "bare";
    // scGW-tilde Tier 2 full frequency (notes/dynbse_plan.md): the ladder's rung, "static"
    // (historic, bitwise) | "dynamic" (the resummed full-frequency W rung), + its solve knobs.
    std::string _ladder_rung = "static";
    double _dyn_tol = 1e-8, _dyn_sign = -1.0;
    long _dyn_maxit = 30, _dyn_gmres = 12;
    long _dyn_rhs_block = 32;    // pol_vertex_dyn_rhs_block: RHS column block width of the dynamic solves
    bool _dyn_dump = false;      // pol_vertex_dyn_dump: per-unit dump + restart files of the dynamic solves
    bool _dyn_all_nu = false;    // pol_vertex_dyn_all_nu: the all-q x all-node dynamic-rung dump (W-int-4f coarse side)
    bool _dyn_cut_r1 = true;     // pol_vertex_dyn_cut_r1: false skips the one-bare-rung (Pi_dyn1) pass of the cut / all-nu dump
    bool _dyn_bubble_only = false;        // pol_vertex_dyn_bubble_only: the all-nu dump writes ONLY the window bubble column
    std::vector<long> _dyn_all_nu_nodes;  // pol_vertex_dyn_all_nu_nodes: the sampled half nodes of the all-nu dump (empty = all)
    std::string _dyn_fit_file;            // pol_vertex_dyn_fit_file: a previous full all-nu dump = the learned nu-basis (LFF L-3)
    long _dyn_fit_rank = 0;               // pol_vertex_dyn_fit_rank: number of nu-modes (0 = the number of sampled nodes)
    std::string _dyn_fit_mode = "modes";  // pol_vertex_dyn_fit_mode: modes (the L-3 least squares on the top-K modes) | regression (P14)
    long _dyn_fit_auto_nodes = 0;         // pol_vertex_dyn_fit_auto_nodes: K > 0 = choose the K sampled nodes from the fit file's modes
                                          // (nu = 0 and the highest node forced, the rest by column-pivoted QR of the mode matrix)
    // LFF-Sigma (Route 1, notes/lff_aux_plan.md 2026-09-19): the local-field-factor vertex in the SELF-ENERGY,
    // Sigma = G W~ with W~ = W Gamma_eff, Gamma_eff = Pi_0^-1 (Pi_0 + dPi) in the frozen secondary frame -- controlled
    // SEPARATELY from the P-side injection (pol_vertex_inject). "none" (default) | "lff".
    std::string _sigma_lff = "none";        // pol_vertex_sigma
    std::string _sigma_lff_bub = "window";  // pol_vertex_sigma_bub: Pi_0 = the dump's window bubble ("window") | the loop's RPA Pi folded ("full")
    double _sigma_lff_scale = 1.0;          // pol_vertex_sigma_scale: multiplies the correction (0 = bit-identical off)
    // LFF-aux L-6, Route 2 (notes/lff_aux_plan.md): the PAIR-RESOLVED static-ladder vertex in Sigma (vertex_sigma_pair.icc),
    // pol_vertex_sigma = "pair"; evaluated on the readout instance at the update_w tail, added to Sigma by gw_t::evaluate
    bool _sigma_pair = false;
    std::string _sigma_pair_col = "static";      // pol_vertex_sigma_pair_col: "static" (T_s, resummed) | "static1" (one rung K_s)
    std::string _sigma_pair_outer = "dynamic";   // pol_vertex_sigma_pair_outer: the outer W-bar: "dynamic" (W-bar(q, i nu)) | "static" (W-bar_0)
    double _sigma_pair_scale = 1.0;              // pol_vertex_sigma_scale (shared with "lff"): multiplies the correction
    bool _sigma_pair_herm = true;                // pol_vertex_sigma_pair_herm: Hermitize dSigma in (i, j)
    bool _sigma_pair_diag = false;               // pol_vertex_sigma_pair_diag: the nu-rank meter of the amplitude + the Pi-check dump
    std::string _sigma_pair_side = "right";      // pol_vertex_sigma_pair_side: the dressed junction, "right" | "left" | "both" (static path)
    bool _sigma_dyn_dump = false;                // pol_vertex_sigma_dyn_dump: the all-node run writes its per-node objects (L-8 reference)
    std::vector<long> _sigma_dyn_nodes;          // pol_vertex_sigma_dyn_nodes: the sampled FULL-mesh nodes (empty = all)
    std::string _sigma_dyn_fit_file;             // pol_vertex_sigma_dyn_fit_file: the reference dump for the sampled mode
    long _sigma_dyn_fit_rank = 0;                // pol_vertex_sigma_dyn_fit_rank: K (0 = the number of sampled nodes)
    double _sigma_lff_pinv_tol = 1e-3;      // pol_vertex_sigma_pinv_tol: relative eigenvalue cutoff of Pi_0^-1 -- the vertex lives on the
                                            // bubble's strong modes (Si kp444: 49 of 156 carry 99.9 % of |tr B|; |Gamma_eff - 1| is bounded and
                                            // stable for 1e-1..1e-3, blows up below 1e-4 where dPi / Pi_0 is unbounded)
    double _sigma_lff_head_scale = 1.0;     // pol_vertex_sigma_head_scale: the q -> 0 head of the correction (0 = body only)
    std::string _sigma_lff_col;             // pol_vertex_sigma_col: the dPi column ("" = pol_vertex_interp_col)
    bool _sigma_lff_static = true;          // pol_vertex_sigma_static: include the instantaneous part (nu -> inf limit) via the static self-energy
    std::string _dyn_resum_mu_file;       // pol_vertex_dyn_resum_mu_file: mu(nu_j) per half node -> Pi_dyn = mu Pi_gam1 (LFF)
    bool _dyn_dense = true;      // pol_vertex_dyn_dense: the dense per-tau rung K_d(s) (nt/2 x D x D per unit)
    std::string _dyn_resolvent = "inverse";   // pol_vertex_dyn_resolvent: inverse (T_s dense) | lu (P7: factor once, solve per application)
    long _dyn_union_stride = 1;  // pol_vertex_dyn_union_stride: keep every n-th shifted G node of the union grid
    int _dyn_table_mode = 0;     // pol_vertex_dyn_table_mode: 0 fitted twisted-pair tables, 1 exact partial fractions
    std::string _dyn_schedule = "longest";                       // pol_vertex_dyn_schedule: longest (heuristic) | measured (P5)
    std::map<std::tuple<long, long, long>, double> _dyn_unit_cost;   // (is, iq, m) -> wall seconds of the dynamic units solved so far
    double _dyn_tfold = 0.0;     // pol_vertex_dyn_tfold: the small-nu fold ratio (0 = off)
    double _dyn_vmask_lo = 0.0, _dyn_vmask_hi = 0.0;   // pol_vertex_dyn_vmask_lo/_hi (Ha, about mu): in-gap vertex nodes dropped
    bool _dyn_gamma1_only = false;   // pol_vertex_dyn_gamma1_only: stop at Gamma_1 (skip the resummation GMRES)
    std::string _dyn_iaft_prec;  // pol_vertex_dyn_iaft_prec: "" = the loop's grid; "medium"/"high" = a vertex-local finer DLR
    // DIAGNOSTIC (default OFF, not physical): THE CONSTANT-RUNG ABSOLUTE PIN.
    //
    // X^L = pi^dyn - Pi^{C,0}(tau=0) must VANISH when the screening is genuinely static.
    // But the two objects do not merely differ in the rung's frequency dependence -- they
    // are built by different code with DIFFERENT INSTANTANEOUS RUNGS:
    //     Pi^{C,0} = pi_c_accumulate_w(rung = W0bar,      Wdyn = nullptr)   [phase 1 only]
    //     pi^dyn   = pi_dyn_factorized(rung = Z (bare!),  Wdyn = Wdyn_w(i.nu))
    // so simply ZEROING Wdyn_w -- the obvious reading of "feed pi^dyn a constant rung" --
    // leaves pi^dyn with the BARE rung Z, not W0, and X^L stays O(1) for a trivial reason.
    // That test would be vacuous.
    //
    // This knob instead sets  Wdyn_w(i.nu) := W0bar - Z  for EVERY i.nu, so pi^dyn's total
    // rung is exactly Z + (W0 - Z) = W0, constant in frequency -- bit-for-bit the rung
    // Pi^{C,0} uses. The two objects are then the SAME integral by two routes, and
    //     X^L -> 0   and   <H, pi^dyn> -> <H, Pi^{C,0}>
    // must hold to the DLR representability floor (~1e-10 on the toy, test_vertex_
    // pibardynfact/static_rung), not to machine epsilon -- both sides are exact Matsubara
    // sums of different integrands read through the same tau = 0 row.
    //
    // WHAT IT DISCRIMINATES. pi^dyn's Gamma-head violation is far larger relative to its
    // own scale than Pi^{C,0}'s (LiH gygi iteration 1: |<H,Pi^L>| = 2.18e-01 against
    // |<H,Pi^{C,0}>| = 2.85e-06, a factor 7.7e+04; 16x by iteration 2). Either that is
    // RETARDED-RUNG PHYSICS (real, and the pin passes) or it is a DEFECT IN THE EQUAL-TIME
    // PATH that Pi^{C,0} does not share (the pin fails). Nothing else separates those two.
    //
    // ✅ ANSWERED 2026-07-31 -- RETARDED-RUNG PHYSICS; THE EQUAL-TIME PATH IS CLEAN.
    // A single number could not decide it (pi^dyn is grid-limited at prec = "low"), so the
    // test sweeps the DLR tolerance. LiH-222, gygi, 2 cold iterations:
    //
    //                       control eps=1e-6   PINNED eps=1e-6   control 1e-10  PINNED 1e-10
    //   X^L / Pi^0             3.374627e-01      4.066572e-04     3.355807e-01   1.759169e-06
    //   |<H, Pi^L(Gamma)>|     4.354277e-02      9.616238e-06     4.374817e-02   2.525278e-08
    //
    // The PINNED residue falls 231x for a 1e4 tightening of eps => it is representability,
    // converging to zero. The CONTROL moves 1.006x => it is completely grid-INDEPENDENT,
    // i.e. genuine physics of the retarded rung. So "a defect in the tau = 0 path that
    // Pi^{C,0} does not share" is REFUTED, and what remains is that B-L's tangent expansion
    // really does produce an unsuppressed Gamma head because the rung is retarded.
    //
    // ⚠ IT DOES NOT MATTER WHICH SLOT CARRIES THE STATIC CONTENT -- MEASURED, not assumed.
    // A first version offered two modes: put W0bar - Z in the DYNAMIC slot (keeping the
    // bare Z instantaneous), or zero the dynamic slot and hand W0bar to the INSTANTANEOUS
    // one. The reasoning was that a rung constant in i.nu is a delta(tau), which the
    // Z-vs-W_dyn splitting exists to keep out of the dynamic basis, so the two would have
    // different floors. THAT REASONING WAS WRONG: pi_dyn_factorized forms the total rung
    // Zc + Wd(i.nu) additively and never expands Wd in a basis, so both readings hand it
    // the SAME rung. The two modes came out BIT-IDENTICAL on LiH (X^L/Pi^0 = 4.066572e-04,
    // e_corr -0.102015281 both ways) -- which, per this project's own rule, is the tell for
    // a no-op, and here it is a real one. One mode is kept.
    bool _bl_pidyn_const_rung = false;
    // DIAGNOSTIC (default 0 = keep everything, not physical): DROP ONE CUT-PIECE from the
    // accumulated self-energy, so its EXACT energy contribution can be read off.
    //   1 = drop Sigma^(L,r) (response)   2 = drop Sigma^(C,x) (kernel)   3 = drop BOTH
    //
    // ⚠ MODE 3 EXISTS BECAUSE THERE ARE THREE PIECES, NOT TWO. This knob touches only the
    // SIGMA cut; B-L also injects P^{C,L} into the Dyson equation, so EVERY arm -- including
    // 1 and 2 -- still runs with a vertex-corrected W, and that shows up through Sigma_GW.
    // With only modes 1 and 2 that common piece is counted twice and the two-way "sum of
    // shares" overshoots by exactly its size (measured on LiH: -2.062e-03 against a total of
    // -6.676e-03, i.e. 31 %). Mode 3 measures it on its own:
    //     d_b = e[drop both] - e[scGW]          <- P^{C,L} acting through Sigma_GW
    //     x   = e[drop response] - e[drop both] <- Sigma^(C,x)
    //     r   = e[drop kernel]   - e[drop both] <- Sigma^(L,r)
    //     identity:  d_b + x + r == e[full] - e[scGW]
    //
    // ⚠ THE SHARES ARE ABLATIONS WITH FEEDBACK -- THEY DO NOT ADD UP. eval_corr_energy is
    // linear in Sigma at FIXED G, but scf_driver runs update_G BEFORE evaluating it
    // (scf_driver.cpp:188 vs :200), so every arm is measured at its OWN post-Dyson G. The
    // arms never share a G, not even at iteration 1, and the leftover in
    // (d_b + x + r) - d_full is the nonlinearity of G's response to Sigma -- measured at
    // 1.116e-03 against d_full = -6.676e-03 on LiH, i.e. ~17 %. Do not present these as an
    // exact decomposition; the conclusions drawn from them must be sign-level statements
    // that survive that nonlinearity.
    //
    // ⚠ Dropping a cut breaks Phi-derivability exactly as the head projection does
    // (CLAUDE.md section 2.1). DIAGNOSTIC ONLY -- these energies are not conserving.
    int _bl_drop = 0;
    bool _skip_pi_c = false;   // TEST-LEVEL: omit Pi^C of an active vertex (set_skip_pi_c)
    // ---- P0.3: THE Gamma-HEAD STRENGTH lambda -----------------------------------------
    // Multiplies the madelung constant xi at EVERY point where the vertex builds its
    // analytic rank-1 Gamma head -- vertex_head_detail::build_head_rank1 (which serves
    // eval_Sigma_C, eval_Pi_C's global path and cache_w), eval_Pi_C's secondary head_c, and
    // build_w0's _w0_head_c. Consistency across all four is REQUIRED: scaling only some of
    // them re-creates the W0-vs-W head-weight MISMATCH that was already measured and refuted
    // as a cause (notes/bl_head_channel_diagnosis.md section 4.2), and would make the scan
    // measure that instead of what it is for.
    //
    // WHY IT EXISTS -- IT SEPARATES ONE-RUNG FROM TWO-RUNG Gamma CONTRIBUTIONS. Sigma^{C,x}
    // is a TWO-rung kernel, so the Gamma cell enters it in two ways: with ONE rung transfer
    // at Gamma (scales as lambda) and with BOTH at Gamma (scales as lambda^2). Fitting
    //     d(e_corr)(lambda) = a lambda + b lambda^2 + c
    // therefore SEPARATES them, at fixed mesh, fixed G and fixed cost -- which an N_k ladder
    // cannot do (its leverage is only N_k^1/3, and 8 -> 12 could not distinguish a
    // legitimate N_k^-1/3 residual from a flat one). A significant b is the signature of the
    // coincident-Gamma cell, whose int d^3q / q^4 is the non-integrable one
    // (notes/head_corrections.pdf sections 2-3); B-S is the control.
    //
    // lambda = 0 is STRUCTURALLY, not just numerically, the "ignore_g0" path: xi * 0 == 0
    // trips the same `xi == 0` guard that already exists at all four sites, so every head
    // branch bails exactly as it does without a head. Pinned both ways by
    // test_vertex_static_e2e "vertex_bl_head_lambda_scan": lambda = 1 reproduces today and
    // lambda = 0 reproduces ignore_g0, both to machine precision.
    //
    // ⚠ lambda != 0, 1 is a DIAGNOSTIC: it is a deliberately wrong q -> 0 treatment, so the
    // resulting energies are finite-size-incorrect by construction. It does NOT break
    // Phi-derivability, though -- unlike _bl_drop and _bl_head_projection, it rescales one
    // input consistently everywhere rather than deleting a piece of one cut, so both cuts
    // still come from one Phi. Default 1.0 = untouched.
    double _bl_head_scale = 1.0;
    // measured G_CC G-rotation consistency residual, running max over this vertex's
    // eval_Pi_C / eval_Sigma_C calls (diagnostic, no gate; memo section 6). Distinct
    // from the iteration-independent D-matrix leakage above: this one tracks whether
    // the self-consistent G_CC itself stays symmetry-consistent across iterations.
    double _g_rot_max = 0.0;

    // ---- B-L Gamma-HEAD A/B METERS (diagnostic; from the LAST eval_Sigma_C call) ------
    // The numbers the head-channel A/B is read from, recorded so a test can ASSERT them
    // instead of scraping [HEADPROJ] out of the log. They are set only by the LINEAR
    // response cut and stay at their init value for B-S (no pi^dyn => no Pi^L).
    // RULE (notes/bl_head_channel_diagnosis.md): a head-carrying object must be metered on
    // its chi-channel projection <H,.>/||H||^2, NEVER on a max-norm -- three independent
    // max-norm gates have already been passed by objects differing ~10x in the only
    // channel that matters.
    double _diag_head_hl = -1.0;       // |<H, Pi^L(Gamma)>|, AFTER any projection
    double _diag_head_hs = -1.0;       // |<H, Pi^{C,0}(Gamma)>| (the suppressed reference)
    double _diag_head_removed = 0.0;   // max|removed| / max|Pi^L(Gamma)|; 0 if projection off
    double _diag_resp_share = -1.0;    // ||Sigma^(C,r)|| / ||Sigma^(C,x)||, theory meter O3
    double _diag_xl_rel = -1.0;        // X^L/Pi^0 = max|pi^dyn - Pi^{C,0}| / max|Pi^{C,0}|
    // B-L's EXPANSION PARAMETER: max over ALL i.nu of |dW| / |W0|, dW = W - W0. This is the
    // meter that says whether the tangent expansion is controlled. It is deliberately NOT
    // the i.nu = 0 value the log has always carried: W0 IS the nu = 0 slice, so dW is small
    // there by construction, and that meter read 0.02-0.06 while the FIRST-order mixed terms
    // came out 3.23x the ZEROTH-order static one (test_vertex_static_e2e, blmixed).
    double _diag_dw_rel = -1.0;        // -1 = never measured (B-S: no dW exists)
    // ---- P0.1: THE SAME EXPANSION PARAMETER, IN THE CHANNEL THE HEAD LIVES IN --------
    // _diag_dw_rel above is a MAX-NORM, and trap 2 says a max-norm cannot see a rank-1
    // head. These two are the chi-channel version, chi = thc.basis_head()(q, :):
    //     h_A(q) := chi^dag A(q) chi / ||chi||^2 ,   ratio(q) := max_nu |h_dW| / |h_W0|
    // _diag_dw_head_rel is ratio(Gamma) -- the ONLY cell that carries the analytic head
    // insertion -- and _diag_dw_head_bg is max over q != Gamma of the same ratio, i.e. a
    // WITHIN-RUN head-free control at the same G, same iteration, same everything.
    // Reading: dW = W - W0 with W(i.nu -> inf) -> v, so in the G = 0 channel the ratio
    // tends to the screening factor eps_M - 1 of that channel. A value >> 1 means B-L's
    // FIRST-order expansion is not merely uncontrolled but divergent there, while the
    // max-norm meter reads a comfortable 0.28.
    double _diag_dw_head_rel = -1.0;   // -1 = never measured (B-S, or no basis_head)
    double _diag_dw_head_bg = -1.0;    // -1 = never measured / no q != Gamma on the mesh
    // ⚠ MEASURED 2026-07-31 AND IT IS NOT THE RATIO. On LiH the Gamma ratio is 0.408 with
    // the head and 0.016 without -- but the head-free q != Gamma control is 0.39-0.41 at
    // BOTH policies, i.e. 0.4 is simply what the G = 0 channel of dW/W0 looks like anywhere.
    // The head does NOT make the RATIO anomalous; it makes the ABSOLUTE content of that one
    // coherent rank-1 direction enormous (madelung ~ 1/q^2), and the kernel sums it over all
    // N_p^2 terms in phase. So these two are the operative meters:
    //   _abs = max_nu |chi^dag dW(Gamma) chi| / ||chi||^2 in a.u. -- comparable ACROSS
    //          policies (same system, same basis), which no ratio is;
    //   _coh = the COHERENCE, normalized: (_abs / max|dW(Gamma)|) divided by the ceiling
    //          ||chi||^2 / max_P|chi_P|^2 that a PERFECTLY chi-aligned rank-1 matrix
    //          c chi chi^dag attains. So _coh = 1 means dW(Gamma) IS that rank-1 matrix and
    //          _coh ~ 1/N_p means no alignment at all. Dimensionless and system-independent,
    //          and it is precisely what a max-norm gate is blind to (trap 2).
    // MEASURED on LiH-222 (2 cold iterations, last one): _abs 8.39e-03 vs 4.93e-06 and
    // _coh 0.980 vs 0.009 between gygi and ignore_g0 -- a 1702x / 111x split, against 1.81x
    // for the max-norm meter above and 93x for |S1+S2|/|S3|.
    double _diag_dw_head_abs = -1.0;   // -1 = never measured
    double _diag_dw_head_coh = -1.0;   // -1 = never measured
    // the SAME channel at i.nu = 0, which is where the run log's long-standing
    // |W(q,0) - W0(q)| meter lives. Kept as an assertable meter because the RATIO
    // _abs / _nu0 is the size of that meter's blind spot: 13.6x at gygi and 428x at
    // ignore_g0, since dW vanishes at nu = 0 by construction (W0 IS that slice) and the
    // head channel grows monotonically to the mesh cutoff, where W -> v (bare, unscreened).
    double _diag_dw_head_nu0 = -1.0;   // -1 = never measured

    // ---- PERFORMANCE INSTRUMENTATION -------------------------------------------------
    // Flat timer over the vertex's high-level operations. Names are grouped by entry
    // point so print_vertex_timers() can show a partition:
    //   top level (disjoint, called from the scf loop): SIGMA_C, PI_C, CACHE_W, BUILD_W0
    //   sub-stages: SIG_*, PI_* -- these PARTITION their parent, and the printer reports
    //               the unattributed remainder explicitly so the breakdown is honest.
    //   lazy sub-builds: SEC_BASIS, SYM_CTX -- INCLUSIVE in whichever parent first
    //               triggered them (they are built once per geometry, not per iteration),
    //               so they are reported separately and NOT added into the total.
    // Timings are rank-local wall time; the printer runs on the root logger only. Vertex
    // work is round-robin over tuples, so rank 0 is representative only to the extent the
    // tuple split is balanced -- load imbalance shows up as a large SIG_KERNEL spread,
    // which is why the reduce/all_reduce stages are timed separately (they absorb skew).
    mutable utils::TimerManager _Timer;

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

    /** Q3 I1 (vertex_ladder.icc): the pair-space ladder's shared inputs -- full-BZ
     *  C-window G, the transfer maps on the full mesh, and the secondary symmetry
     *  context (nullptr on nosym meshes). Ensures the secondary basis. */
    void ladder_inputs(MBState &mb_state, THC_ERI auto &thc,
                       nda::array<ComplexType, 5> &G_CC,
                       nda::array<long, 2> &kmq, nda::array<long, 2> &kpq,
                       vertex_sym::sym_ctx const *&symc);

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
     * Historic guard seam (the S1 "kernels not implemented" abort, relocated at S2,
     * emptied as S3-S10 landed the kernels). All three rung modes are implemented;
     * this is now a NO-OP retained so future mode additions have a ready guard point
     * at the two kernel entries (eval_Sigma_C, eval_Pi_C).
     *
     * @param where - [INPUT] call site, used verbatim in a (currently unreachable) abort
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
    /**
     * Print the vertex performance breakdown (utilities/Timer.hpp, the hf_t idiom).
     *
     * Layout: the four TOP-LEVEL entry points the scf loop calls (eval_Sigma_C, eval_Pi_C,
     * cache_w, build_w0) are DISJOINT, so their sum is the total time this rank spent
     * inside vertex routines. Each is then broken into its high-level stages, and the
     * printer reports the UNATTRIBUTED remainder per entry point rather than silently
     * letting the parts fail to add up -- an unattributed row that is large means a stage
     * boundary is missing, which is exactly what we want to see.
     *
     * The two lazy geometry-fixed builds (secondary ISDF basis, IBZ symmetry context) are
     * INCLUSIVE in whichever entry point first triggered them, so they are printed in a
     * separate block and NOT re-added to the total.
     *
     * Wall time is rank-local. Call collectively if you want it on every rank; the
     * app_log filter means only the root actually emits.
     */
    void print_vertex_timers() const;

    /** Reset every vertex timer (e.g. to profile a single scf iteration in isolation). */
    void reset_vertex_timers() { _Timer.reset_all(); }

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
      _w0_head_share_pre = -1.0;
      _w0_head_share_post = -1.0;
      _w0_head_atten = -1.0;
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
    // DA D-7 / H1b pre-post fold head meter of the last build_w0 (-1 = not measured)
    double w0_head_share_pre() const { return _w0_head_share_pre; }
    double w0_head_share_post() const { return _w0_head_share_post; }
    double w0_head_attenuation() const { return _w0_head_atten; }
    // "v1_skip" fallback: the Gamma cell of the rung transfer is dropped BY THE KERNEL
    // (as for Z / dW -- the stored W0(Gamma) is the regularized body either way). Kept
    // as a flag on the handle so the S3+ kernels inherit the policy from the ONE W0.
    bool w0_skip_gamma() const { return _div_treatment == "v1_skip"; }

    // W-bar iteration cache accessors (notes/wbar_cache.md)
    bool has_cached_w() const { return _Wb_qwmm.has_value() or _Wb_shm != nullptr; }
    void reset_w_cache() { _Wb_qwmm.reset(); _Wb_shm.reset(); }
    /** pol_vertex_wcache (default "replicated"): the storage of the W-bar cache -- one array per rank, or (P19, "shared")
     *  one node-shared window per NUMA node (16 nq_ibz nw_half N_m^2 bytes once per node instead of once per rank). */
    void set_wcache_mode(std::string const &m) {
      utils::check(m == "replicated" or m == "shared", "vertex_t::set_wcache_mode: pol_vertex_wcache must be replicated | shared (got \"{}\").", m);
      _wcache = m;
    }
    std::string const &wcache_mode() const { return _wcache; }
    /** the W-bar cache (nq_ibz, nw_half, N_m, N_m) as a view onto its storage (replicated array or node-shared window) */
    nda::array_view<ComplexType, 4> wb_cache() {
      utils::check(has_cached_w(), "vertex_t::wb_cache: the W-bar cache is absent -- cache_w must run first.");
      if (_Wb_shm) return _Wb_shm->local();
      return nda::array_view<ComplexType, 4>(_Wb_qwmm.value());
    }
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
    /**
     * B-L's pi^dyn route (eq:pibardynfact). "factorized" (default) evaluates the
     * equal-time dynamic-rung polarization directly as ONE bosonic pairing of two bubbles
     * against W -- pole-free, and the item that was 98.9 % of B-L's vertex time. "kernel"
     * restores the historic route (full dynamic-rung Pi^C over all nw_b frequencies, tau=0
     * row kept). "check" runs both and gates their agreement, for confirming the refactor
     * at production scale rather than only on the toy (test_vertex_pibardynfact).
     *
     * check_tol <= 0 uses an O(1) ROUTING abort bar (0.25) with a separate grid-derived
     * WARNING at max(1e-8, 100*eps) -- see _pidyn_check_tol for why an eps-derived abort
     * would be flaky by construction.
     */
    void set_pidyn_mode(std::string m, double check_tol = -1.0) {
      if (m == "factorized") _pidyn_mode = 0;
      else if (m == "kernel") _pidyn_mode = 1;
      else if (m == "check") _pidyn_mode = 2;
      else utils::check(false, "vertex_t::set_pidyn_mode: unknown vertex_pidyn \"{}\". "
                               "Valid options are \"factorized\", \"kernel\", \"check\".", m);
      if (check_tol > 0.0) _pidyn_check_tol = check_tol;
    }
    int pidyn_mode() const { return _pidyn_mode; }

    /**
     * Enable/disable the q -> 0 head-channel projection of the response middle factor.
     * See _bl_head_projection. DEFAULT OFF since 2026-07-31: it BREAKS Phi-derivability
     * (proven by the fdoracle HEAD-PROJECTION control, 3.3e-11 -> 1.6e-01) and is applied
     * to the Sigma cut only. Turning it ON is a DIAGNOSTIC and yields non-conserving
     * energies; the only thing it is known to buy is the cold-start basin.
     */
    void set_bl_head_projection(bool on) { _bl_head_projection = on; }
    bool bl_head_projection() const { return _bl_head_projection; }

    /** DIAGNOSTIC, default OFF. See _bl_static_head. */
    void set_bl_static_head(bool on) { _bl_static_head = on; }
    bool bl_static_head() const { return _bl_static_head; }

    /** DIAGNOSTIC, default OFF. See _bl_w0_head_from_w. */
    void set_bl_w0_head_from_w(bool on) { _bl_w0_head_from_w = on; }
    bool bl_w0_head_from_w() const { return _bl_w0_head_from_w; }

    /** H1, the balanced first-order head (default OFF): in B-L, every W input of the
     *  vertex functional carries the SAME STATIC-weight Gamma head as W0 (instantaneous
     *  slot, weight 1 + eps_inv_head(i.nu=0); no dynamic-slot head), so dW = W - W0
     *  carries no analytic head. Conserving (modified interaction in Phi). B-S is
     *  bit-identical; the parent keeps its retarded head. See _bl_head_static_all and
     *  notes/bl_head_balance_theory_and_plan.md. */
    void set_bl_head_static_all(bool on) {
      _bl_head_static_all = on;
      if (on)
        app_log(1, "  [ISDF-Vertex] H1 STATIC-HEAD vertex enabled "
                   "(vertex_bl_head_static_all): in B-L, every W input of the vertex "
                   "functional carries W0's STATIC Gamma-head weight\n"
                   "  (instantaneous slot, 1 + eps_inv_head(i.nu=0)); no dynamic-slot "
                   "head; dW = W - W0 is analytic-head-free. Conserving (Phi-level "
                   "modified interaction). No effect on B-S or the parent theory.");
    }
    bool bl_head_static_all() const { return _bl_head_static_all; }

    /** RANK-CAP LIFT for the secondary path: distr_tol handed to the PRIVATE thc builder
     *  of build_secondary_basis, which never saw the toml's value and used the class
     *  default 0.2 (capping nproc at nc-class counts; measured kp444/M8 aborts at
     *  52/104). <= 0 (default) keeps today's behavior exactly; 1.0 lifts the kp444
     *  maxima to 208 (M4) / 260 (M8). Distribution-only: results are unchanged at rank
     *  counts that already ran. */
    void set_isdf_distr_tol(double tol) { _isdf_distr_tol = tol; }
    double isdf_distr_tol() const { return _isdf_distr_tol; }

    /**
     * scGW-tilde ladder polarization (pol_vertex = "ladder"; notes/
     * scgwt_implementation_plan.md increments L1-L3). Validates and stores the knob
     * surface; enforces the double-count guard (ruling R5) and the DLR requirement for
     * an ACTIVE ladder. C = empty (window size 0) is an exact no-op, mirroring the
     * vertex convention -- the inert path is reached BEFORE any not-implemented abort,
     * exactly like the S1 rung-mode plumbing. The pol_* basis knobs arrive RESOLVED
     * (the driver applies the "inherit vertex_*" default rule).
     */
    void set_pol_vertex(std::string mode, std::string kernel, nda::range band_window,
                        long isdf_rank, double isdf_svd_tol, double isdf_thresh,
                        double isdf_cond_max, double isdf_distr_tol,
                        std::string inject = "none") {
      utils::check(mode == "none" or mode == "ladder",
                   "vertex_t::set_pol_vertex: unknown pol_vertex \"{}\". Valid options "
                   "are \"none\", \"ladder\".", mode);
      utils::check(kernel == "w0_prev" or kernel == "w0_frozen",
                   "vertex_t::set_pol_vertex: unknown pol_vertex_kernel \"{}\". Valid "
                   "options are \"w0_prev\" (default; ruling R4), \"w0_frozen\".", kernel);
      utils::check(inject == "none" or inject == "ladder_n2",
                   "vertex_t::set_pol_vertex: unknown pol_vertex_inject \"{}\". Valid "
                   "options are \"none\" (default), \"ladder_n2\".", inject);
      _pol_vertex_inject = inject;
      // R-Q3-3: injection IMPLIES the ladder machinery, so it auto-enables it rather than
      // failing on a half-specified input. Logged -- a knob that changes the theory must
      // never turn itself on silently.
      if (inject != "none" and mode == "none") {
        mode = "ladder";
        app_log(1, "  [qpGW Q3] pol_vertex_inject = \"{}\" auto-enables pol_vertex = "
                   "\"ladder\".", inject);
      }
      _pol_vertex = mode;
      _pol_kernel = kernel;
      _pol_band_window = band_window;
      _pol_isdf_rank = isdf_rank;
      _pol_isdf_svd_tol = isdf_svd_tol;
      _pol_isdf_thresh = isdf_thresh;
      _pol_isdf_cond_max = isdf_cond_max;
      _pol_isdf_distr_tol = isdf_distr_tol;
      if (not pol_vertex_active()) {
        if (pol_vertex_enabled())
          app_log(1, "  [scGW-tilde] pol_vertex = \"ladder\" with an EMPTY C-window: "
                     "the ladder is inert (exact no-op).");
        return;
      }
      // DOUBLE-COUNT GUARD (ruling R5; scgw_screening_fix_proposal.pdf section 5.2): the
      // ladder's first-order term IS the implemented static-rung Pi^C, so an ACTIVE
      // vertex_type is excluded -- "linear"/"dynamic" inject a Pi^C into P (double
      // counting), and "static" would run Sigma^C rungs beside the ladder, which the
      // adopted scGW-tilde scheme defers (Sigma stays GW-form; Sigma^C re-enable is a
      // separate ruling, plan X1 note).
      utils::check(not active(),
                   "pol_vertex = \"ladder\" cannot be combined with an ACTIVE vertex_type "
                   "(= \"{}\", vertex_rung = \"{}\"): the ladder resums the static-rung "
                   "Pi^C (double counting on the P side), and scGW-tilde keeps Sigma "
                   "GW-form. Disable one of the two.", _vertex_type, rung_str());
      // frequency-diagonal solves + the W-bar_0 kernel live on the DLR nodes
      utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                   "pol_vertex = \"ladder\" requires the DLR IAFT backend "
                   "(iaft basis = \"dlr\").");
      // LIVE since increment L2 as a READOUT (stance i): scr_coulomb_t::update_w runs
      // the pair-space ladder on its private readout vertex and reports the
      // ladder-corrected eps_M each iteration. Without pol_vertex_inject the loop is
      // untouched (report-only); with it, the Q3 line below states the actual regime.
      app_log(1, "  [scGW-tilde] pol_vertex = \"ladder\" READOUT active: C window = "
                 "[{}, {}), kernel = {}{}", _pol_band_window.first(),
              _pol_band_window.last(), _pol_kernel,
              pol_vertex_inject_enabled()
                  ? " (L2 readout + the Q3 injection below)."
                  : " (L2, stance i -- report-only; in-loop injection is off,"
                    " knob pol_vertex_inject).");
      if (pol_vertex_inject_enabled())
        app_log(1, "  [qpGW Q3] pol_vertex_inject = \"{}\": the resummed ladder IS "
                   "injected into P (P_latt = P^RPA + P^lad, eq 6 of "
                   "notes/qpgw_bse_edmft_option2.pdf; rung = W-bar_0[RPA] at inu = 0, "
                   "ruling R-Q3-1). The loop is no longer plain RPA-screened.",
                _pol_vertex_inject);
    }
    // scGW-tilde ladder requested in the input ([gw] pol_vertex)
    bool pol_vertex_enabled() const { return _pol_vertex != "none"; }
    // Q3: in-loop ladder injection requested (R-Q3-3). Injection additionally requires
    // pol_vertex_active() (non-empty C window) -- empty window = structural no-op.
    bool pol_vertex_inject_enabled() const { return _pol_vertex_inject != "none"; }
    std::string pol_vertex_inject() const { return _pol_vertex_inject; }
    // requested AND the ladder C-window is non-empty (C = empty = exact no-op)
    bool pol_vertex_active() const {
      return pol_vertex_enabled() and _pol_band_window.size() > 0;
    }
    std::string pol_vertex() const { return _pol_vertex; }
    std::string pol_vertex_kernel() const { return _pol_kernel; }
    // resolved ladder-basis knobs (scr_coulomb_t builds its private readout vertex
    // from these -- increment L2)
    nda::range pol_band_window() const { return _pol_band_window; }
    long pol_isdf_rank() const { return _pol_isdf_rank; }
    double pol_isdf_svd_tol() const { return _pol_isdf_svd_tol; }
    double pol_isdf_thresh() const { return _pol_isdf_thresh; }
    double pol_isdf_cond_max() const { return _pol_isdf_cond_max; }
    double pol_isdf_distr_tol() const { return _pol_isdf_distr_tol; }

    /**
     * INCREMENT B (notes/ladder_b_integration_design.md sections 1-2): the ladder's dense
     * resolvent gets a SOLVE-GRID dimension g = ranks cooperating on one (s,q,nu) solve.
     *   g = 1 (default) : today's per-rank LAPACK path -- bit-identical to the pre-B tree,
     *                     threading comes from the BLAS library.
     *   g > 1           : the SLATE distributed path (no rank holds a full (D,D)); requires
     *                     nproc % g == 0 and, with OMP_NUM_THREADS > 1, MPI_THREAD_MULTIPLE
     *                     (env knob COQUI_MPI_THREAD_MULTIPLE=1, main.cpp).
     *   g = 0           : AUTO -- the per-rank memory fit test against budget_gb picks the
     *                     smallest g whose per-rank footprint fits.
     * budget_gb <= 0 keeps the 8 GB default. Numerics: g = 1 is bitwise the historic path;
     * g > 1 is the SAME exact dense solve reassociated, gated at <= 1e-12 relative.
     */
    void set_ladder_solve(long grid, double budget_gb) {
      utils::check(grid >= 0, "vertex_t::set_ladder_solve: ladder_solve_grid must be >= 0 "
                              "(0 = auto, 1 = per-rank LAPACK, > 1 = SLATE grid); got {}.",
                   grid);
      _ladder_solve_grid = grid;
      if (budget_gb > 0.0) _ladder_solve_budget_gb = budget_gb;
    }
    long ladder_solve_grid() const { return _ladder_solve_grid; }
    double ladder_solve_budget_gb() const { return _ladder_solve_budget_gb; }

    /**
     * DA Phase 2 knob surface (notes/qsgwhat_discrepancy_spec.md, Phase 2 D-1/D-4/D-7).
     * See the member declarations for what each one does and why it exists. All three are
     * default-inert: tda = false leaves the kernel untouched, head_scale = 1.0 multiplies
     * the head coefficient by exactly 1.0 (IEEE-exact), qnu_meter = false emits nothing.
     * scr_coulomb_t::ensure_pol_vertex copies them from the knob carrier onto the READOUT
     * instance, which is the vertex that actually builds W0 and runs the ladder.
     */
    void set_ladder_da(bool tda, double head_scale, bool qnu_meter) {
      utils::check(std::isfinite(head_scale) and head_scale >= 0.0,
                   "vertex_t::set_ladder_da: ladder_head_scale must be finite and >= 0 "
                   "(got {}).", head_scale);
      _ladder_tda = tda;
      _ladder_head_scale = head_scale;
      _ladder_qnu_meter = qnu_meter;
    }
    bool ladder_tda() const { return _ladder_tda; }
    double ladder_head_scale() const { return _ladder_head_scale; }
    bool ladder_qnu_meter() const { return _ladder_qnu_meter; }
    /** eps(q_i, i nu) cut instrumentation (2026-09-11, report-only): see _eps_cut_nq. */
    void set_eps_cut(long nq, long dyn_nnu = 0) {
      utils::check(nq >= 0 and dyn_nnu >= 0, "vertex_t::set_eps_cut: pol_eps_cut / pol_eps_cut_dyn_nnu must be >= 0 (got {}, {}).",
                   nq, dyn_nnu);
      _eps_cut_nq = nq;
      _eps_cut_dyn_nnu = dyn_nnu;
    }
    long eps_cut_nq() const { return _eps_cut_nq; }
    long eps_cut_dyn_nnu() const { return _eps_cut_dyn_nnu; }
    /** W-int-4f coarse side: pol_vertex_dyn_all_nu (default false) -- after the inu = 0 readout, run the dynamic-rung
     *  ladder on EVERY transfer x EVERY PH-sym half node (eval_pol_dynbse_cut over all q / all nodes; needs
     *  pol_vertex_dyn_dump) and write <prefix>.pol_wh_dyn.g<n>.h5 with the four columns (static, dyn1, gam1, dyn),
     *  the full-frequency vertex object in the frozen-able point frame for the fine W-Dyson feed (col "gam1"). */
    void set_ladder_dyn_all_nu(bool on) { _dyn_all_nu = on; }
    bool ladder_dyn_all_nu() const { return _dyn_all_nu; }
    /** pol_vertex_dyn_cut_r1 (default true): the cut / all-nu dump also runs the one-BARE-dynamic-rung pass (the Pi_dyn1
     *  column, D^dag L0 K_d L0 D: a diagnostic, ~35 % of the cut wall at prec high). false skips it and writes
     *  Pi_dyn1 = Pi_static -- the production setting for the Gamma_1 (col "gam1") coarse dump. */
    void set_ladder_dyn_cut_r1(bool on) { _dyn_cut_r1 = on; }
    bool ladder_dyn_cut_r1() const { return _dyn_cut_r1; }
    /** LFF-aux L-0 (notes/lff_aux_plan.md). pol_vertex_dyn_bubble_only (default false): the all-nu dump evaluates ONLY the
     *  C-window bubble Pi_bub(q, i nu) = (spin/nk) D^dag Cb D in the secondary frame at every q x every half node (no rung,
     *  no solve; ~1 s per unit) and writes it as the "Pi_bub" column of <prefix>.pol_wh_dyn.g<n>.h5 -- the reference the
     *  dumped corrections are measured against. Every all-nu dump also carries Pi_bub (cheap).
     *  pol_vertex_dyn_all_nu_nodes (default empty = all): the dynamic columns are evaluated on THIS subset of half nodes
     *  only (the sparse sampling of the fit); the other nodes are written as zeros, "nu_sampled" lists the subset. */
    void set_ladder_dyn_bubble_only(bool on) { _dyn_bubble_only = on; }
    bool ladder_dyn_bubble_only() const { return _dyn_bubble_only; }
    void set_ladder_dyn_all_nu_nodes(std::vector<long> const &nodes) { _dyn_all_nu_nodes = nodes; }
    std::vector<long> const &ladder_dyn_all_nu_nodes() const { return _dyn_all_nu_nodes; }
    /** LFF-aux L-3 (notes/lff_aux_plan.md): the ON-DEMAND FIT of the sampled-node dump. pol_vertex_dyn_fit_file = a previous
     *  FULL all-nu dump (the previous scGW iteration's <prefix>.pol_wh_dyn.g<n>.h5, or any full evaluation on the same
     *  grid): its columns supply the nu-basis (the top-K left singular vectors of the (n_nu, nq Nm^2) unfolding, K =
     *  pol_vertex_dyn_fit_rank or the number of sampled nodes); each column of the sampled-node dump is refit by least
     *  squares in that basis and written at ALL nodes, so the consumer reads the file unchanged. Measured on Si kp444 (plan
     *  §5, round 2): 5 nodes (nu = 0 + 3 pivots + the highest node) reproduce the 40-node Gamma_1 object in the W-Dyson to
     *  0.01 % of the vertex effect on e_corr. */
    void set_ladder_dyn_fit(std::string const &file, long rank, std::string const &mode = "modes", long auto_nodes = 0) {
      utils::check(mode == "modes" or mode == "regression", "vertex_t::set_ladder_dyn_fit: pol_vertex_dyn_fit_mode must be modes | regression (got \"{}\").", mode);
      _dyn_fit_file = file; _dyn_fit_rank = rank; _dyn_fit_mode = mode; _dyn_fit_auto_nodes = auto_nodes;
    }
    std::string const &ladder_dyn_fit_mode() const { return _dyn_fit_mode; }
    long ladder_dyn_fit_auto_nodes() const { return _dyn_fit_auto_nodes; }
    /** LFF (notes/lff_aux_plan.md, "Gamma_1 -> resummed"): pol_vertex_dyn_resum_mu_file = a text table of mu(nu_j), one value
     *  per PH-sym half node (n_nu lines: "j nu mu" or "mu"; '#' comments). The all-nu dump then writes Pi_dyn = mu(nu_j) x
     *  Pi_gam1 (the fitted / evaluated Gamma_1 column) -- the fully resummed vertex to 2-3 % of the correction on Si
     *  (matrix-level mu_F is q-independent to 0.2 %); the consumer selects it with pol_vertex_interp_col = "dyn". */
    void set_ladder_dyn_resum_mu_file(std::string const &f) { _dyn_resum_mu_file = f; }
    std::string const &ladder_dyn_resum_mu_file() const { return _dyn_resum_mu_file; }
    std::string const &ladder_dyn_fit_file() const { return _dyn_fit_file; }
    long ladder_dyn_fit_rank() const { return _dyn_fit_rank; }
    /** LFF-Sigma (Route 1): Sigma = G W~, W~ = W Gamma_eff with the aux-frame (local-field-factor) vertex of the injected
     *  dPi -- Del Sole / Reining / Godby's collapse of Hedin's G W Gamma for a vertex acting on the density index only.
     *  Independent of pol_vertex_inject: the P side and the Sigma side of the vertex are separate input knobs. */
    void set_sigma_lff(std::string const &mode, std::string const &bub, double scale, double pinv_tol, double head_scale,
                       std::string const &col, bool with_static = true) {
      utils::check(mode == "none" or mode == "lff",
                   "vertex_t::set_sigma_lff: unknown pol_vertex_sigma \"{}\". Valid options are \"none\" (default), \"lff\".", mode);
      utils::check(bub == "window" or bub == "full",
                   "vertex_t::set_sigma_lff: unknown pol_vertex_sigma_bub \"{}\". Valid options are \"window\" (default), \"full\".", bub);
      utils::check(pinv_tol >= 0.0 and pinv_tol < 1.0, "vertex_t::set_sigma_lff: pol_vertex_sigma_pinv_tol = {} must be in [0, 1).", pinv_tol);
      _sigma_lff = mode; _sigma_lff_bub = bub; _sigma_lff_scale = scale; _sigma_lff_pinv_tol = pinv_tol;
      _sigma_lff_head_scale = head_scale; _sigma_lff_col = col; _sigma_lff_static = with_static;
      if (mode != "none")
        app_log(1, "  [LFF-Sigma] pol_vertex_sigma = \"{}\": Sigma = G W~, W~ = W Gamma_eff (Gamma_eff = Pi_0^-1 (Pi_0 + dPi) in the frozen "
                   "secondary frame; Pi_0 = \"{}\"; scale {}; pinv tol {:.1e}; head scale {}; column \"{}\"; instantaneous part {}).",
                mode, bub, scale, pinv_tol, head_scale, col.empty() ? std::string("<pol_vertex_interp_col>") : col,
                with_static ? "-> the static self-energy" : "DROPPED");
    }
    bool sigma_lff_enabled() const { return _sigma_lff != "none"; }
    /** LFF-aux L-6 (Route 2): pol_vertex_sigma = "pair" -- the pair-resolved STATIC-LADDER vertex in Sigma, from the same
     *  pair-space machinery as the polarization ladder (the left vertex D^dag (1 + Cb T_s) on the right GW leg, contracted
     *  with W-bar(q, i nu) and G on the C window; vertex_sigma_pair.icc). Independent of pol_vertex_inject (the P side). */
    void set_sigma_pair(bool on, std::string const &col, std::string const &outer, double scale, bool hermitize, bool diag,
                        std::string const &side = "right") {
      utils::check(col == "static" or col == "static1" or col == "static_dyn" or col == "dyn1_bare" or col == "dyn1" or col == "dyn",
                   "vertex_t::set_sigma_pair: unknown pol_vertex_sigma_pair_col \"{}\". Valid options are \"static\" (default), \"static1\" "
                   "(the static-ladder path) and \"static_dyn\", \"dyn1_bare\", \"dyn1\", \"dyn\" (the dynamic-rung path, L-7).", col);
      utils::check(outer == "dynamic" or outer == "static",
                   "vertex_t::set_sigma_pair: unknown pol_vertex_sigma_pair_outer \"{}\". Valid options are \"dynamic\" (default), \"static\".", outer);
      utils::check(side == "right" or side == "left" or side == "both",
                   "vertex_t::set_sigma_pair: unknown pol_vertex_sigma_pair_side \"{}\". Valid options are \"right\" (default), \"left\", \"both\".", side);
      utils::check(side == "right" or col == "static" or col == "static1",
                   "vertex_t::set_sigma_pair: pol_vertex_sigma_pair_side = \"{}\" is implemented on the static-ladder path only (col static | static1).", side);
      _sigma_pair = on; _sigma_pair_col = col; _sigma_pair_outer = outer; _sigma_pair_scale = scale; _sigma_pair_herm = hermitize;
      _sigma_pair_diag = diag; _sigma_pair_side = side;
      if (on)
        app_log(1, "  [LFF-Sigma pair] pol_vertex_sigma = \"pair\": Sigma = G W-bar Lambda with the pair-resolved {} vertex "
                   "(column \"{}\", outer W-bar \"{}\", junction \"{}\", scale {}, Hermitized {}, diagnostics {}).",
                sigma_pair_dynamic() ? "DYNAMIC-rung ladder" : "static-ladder", col, outer, side, scale, hermitize, diag);
    }
    /** L-7: the dynamic-rung columns run through eval_sigma_pair_dyn (vertex_sigma_dyn.icc) */
    bool sigma_pair_dynamic() const {
      return _sigma_pair_col == "static_dyn" or _sigma_pair_col == "dyn1_bare" or _sigma_pair_col == "dyn1" or _sigma_pair_col == "dyn";
    }
    std::string const &sigma_pair_side() const { return _sigma_pair_side; }
    /** L-8: the nu-sampled dynamic Sigma vertex (vertex_sigma_dyn.icc, "THE SAMPLED MODE") and its reference dump. */
    void set_sigma_dyn(bool dump, std::vector<long> const &nodes, std::string const &fit_file, long fit_rank) {
      utils::check(nodes.empty() or not fit_file.empty(),
                   "vertex_t::set_sigma_dyn: pol_vertex_sigma_dyn_nodes needs pol_vertex_sigma_dyn_fit_file (the all-node reference dump).");
      _sigma_dyn_dump = dump; _sigma_dyn_nodes = nodes; _sigma_dyn_fit_file = fit_file; _sigma_dyn_fit_rank = fit_rank;
    }
    bool sigma_dyn_dump() const { return _sigma_dyn_dump; }
    std::vector<long> const &sigma_dyn_nodes() const { return _sigma_dyn_nodes; }
    std::string const &sigma_dyn_fit_file() const { return _sigma_dyn_fit_file; }
    long sigma_dyn_fit_rank() const { return _sigma_dyn_fit_rank; }
    bool sigma_pair_enabled() const { return _sigma_pair; }
    std::string const &sigma_pair_col() const { return _sigma_pair_col; }
    std::string const &sigma_pair_outer() const { return _sigma_pair_outer; }
    double sigma_pair_scale() const { return _sigma_pair_scale; }
    bool sigma_pair_herm() const { return _sigma_pair_herm; }
    bool sigma_pair_diag() const { return _sigma_pair_diag; }
    struct sigma_pair_opts {
      std::string col = "static";     // static | static1 (vertex_sigma_pair.icc) | static_dyn | dyn1_bare | dyn1 | dyn (vertex_sigma_dyn.icc)
      std::string outer = "dynamic";  // "dynamic" = W-bar_0 + [W_dyn(nu) - W_dyn(0)]  |  "static" = W-bar_0
      std::string side = "right";     // the dressed junction: "right" (A = T_s Cb D on the right leg) | "left" (A~ = [Cb T_s]^T D^* on
                                      // the left leg) | "both" (the average of the two one-sided insertions) -- static path only
      double sign_ks = -1.0;          // the ladder's rung sign (pol_vertex_dyn_sign)
      double scale = 1.0;
      bool hermitize = true;
      bool nu_diag = false;           // the Gram spectrum over nu of the amplitude (meter nu_spec)
      nda::array<ComplexType, 4> *Pi_check = nullptr;   // (nw_b, nq, Nm, Nm): (spin/nk) A~^T (Cb D), the P side's own object (gate G1)
      // L-8 (the dynamic path only): the nu-SAMPLED evaluation with a learned rank-K nu-basis, and the reference dump
      bool dyn_dump = false;            // write the per-node objects to <dump_prefix>.sigdyn.h5 (the all-node reference)
      std::vector<long> dyn_nodes;      // FULL-mesh node indices to evaluate (empty = all nodes, the exact nu-sum)
      std::string dyn_fit_file;         // the reference dump the nu-bases are learned from (needed with dyn_nodes)
      long dyn_fit_rank = 0;            // K modes per object type (0 = |dyn_nodes|: interpolation)
      std::string dump_prefix;          // the run prefix for the dump
    };
    struct sigma_pair_meter {
      double dsig_max = 0.0, dsig_herm = 0.0, ks_herm = 0.0;
      double t_total = 0.0, t_setup = 0.0, t_ks = 0.0, t_cb = 0.0, t_lu = 0.0, t_amp = 0.0, t_con = 0.0, rss_gb = 0.0;
      long nunits = 0;
      nda::array<double, 1> nu_spec;   // normalized Gram eigenvalues over nu, descending, max over units (nu_diag)
      // L-7 (the dynamic path): the T-family refit error, the solver's G pole fit / tau refit / watchdog, solve wall, convergence
      double e_fit_err = 0.0, fit_err_G = 0.0, refit_err = 0.0, ritz_max = 0.0, t_solve = 0.0;
      bool all_converged = true;
    };
    /** The pair-resolved static-ladder vertex self-energy on the C window: dSig (nt, ns, nk, nc, nc), replicated.
     *  Runs on the READOUT instance (secondary frame, W-bar_0 of this update; cache_w called on demand for outer = dynamic). */
    void eval_sigma_pair(MBState &mb_state, THC_ERI auto &thc, sigma_pair_opts const &opt,
                         nda::array<ComplexType, 5> &dSig, sigma_pair_meter *met = nullptr);
    /** L-7: the DYNAMIC-rung vertex in Sigma (vertex_sigma_dyn.icc): the dynbse solver on every bosonic node with the Sigma
     *  hook armed; opt.col = static_dyn | dyn1_bare | dyn1 | dyn. Same output contract as eval_sigma_pair. */
    void eval_sigma_pair_dyn(MBState &mb_state, THC_ERI auto &thc, sigma_pair_opts const &opt,
                             nda::array<ComplexType, 5> &dSig, sigma_pair_meter *met = nullptr);
    std::string const &sigma_lff_bub() const { return _sigma_lff_bub; }
    double sigma_lff_scale() const { return _sigma_lff_scale; }
    double sigma_lff_pinv_tol() const { return _sigma_lff_pinv_tol; }
    double sigma_lff_head_scale() const { return _sigma_lff_head_scale; }
    std::string const &sigma_lff_col() const { return _sigma_lff_col; }
    bool sigma_lff_static() const { return _sigma_lff_static; }

    /**
     * scGW-tilde TIER 1.5 (notes/tier15_ward_legs_plan.md; proposal section 4.6): the LEG
     * VERTEX of the ladder's pair propagators.
     *   "bare" (default) : the historic pair propagator -- bitwise the pre-Tier-1.5 tree.
     *   "ward"           : Lambda0 = 1 - [Sigma(iw+inu) - Sigma(iw)]/inu inserted at the
     *                      vertex of every pair propagator (the telescoping discrete-Ward
     *                      vertex, eq 21), evaluated through the DLR pole products of
     *                      ward_legs.hpp from the loop's OWN stored Sigma. The kernel then
     *                      returns Delta P^Lambda (the zero-rung term) + rungs >= 1 on the
     *                      Lambda legs (eq 27, the Tier-1.5 composite), (M,N)-Hermitized.
     * Travels to the READOUT instance with the DA knobs (scr_coulomb_t::ensure_pol_vertex).
     * The Sigma-side double-count guard is the pol_vertex one (an ACTIVE vertex_type is
     * already excluded whenever the ladder is active).
     */
    void set_ladder_legs(std::string legs) {
      utils::check(legs == "bare" or legs == "ward",
                   "vertex_t::set_ladder_legs: unknown pol_vertex_legs \"{}\". Valid options "
                   "are \"bare\" (default), \"ward\".", legs);
      if (legs == "ward")
        utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                     "pol_vertex_legs = \"ward\" requires the DLR IAFT backend "
                     "(iaft basis = \"dlr\").");
      _ladder_legs = legs;
      if (legs == "ward")
        app_log(1, "  [scGW-tilde T1.5] pol_vertex_legs = \"ward\": the ladder's pair "
                   "propagators carry the discrete-Ward leg vertex Lambda0 (proposal "
                   "section 4.6, eq 21; notes/tier15_ward_legs_plan.md) built from the "
                   "loop's own stored Sigma;\n  the kernel returns Delta P^Lambda + rungs "
                   ">= 1 on Lambda-legs (eq 27), (M,N)-Hermitized. Static rungs on Lambda "
                   "legs XOR a dynamical-rung BSE -- never both (eq 24 cor. ii).");
    }
    std::string ladder_legs() const { return _ladder_legs; }
    bool ladder_ward_legs() const { return _ladder_legs == "ward"; }

    /**
     * scGW-tilde Tier 2 FULL FREQUENCY (notes/dynbse_plan.md, increment D3): the ladder's RUNG.
     *   "static"  : the historic W0bar rung (bitwise);
     *   "dynamic" : the full-frequency screened rung W(inu'), resummed to all orders
     *               (vertex_dynbse.icc, pair_space_ladder_dyn) -- the eps_M readout gains the
     *               columns +static (sign-corrected), +static+Pi^C_dyn (one dynamic rung),
     *               +Gamma_1 and +resummed; inu = 0 only at D3.
     * tol / maxit / gmres_m : the dynamic-remainder solve (GMRES(m), 0 = Neumann);
     * sign_ks : the static-rung sign convention in the dynamic driver (-1 = the derived sign,
     *           memory l2-resolvent-sign-finding; +1 = the as-implemented L2 convention).
     * Dynamic rungs XOR Tier-1.5 legs (never both). Travels to the READOUT instance.
     */
    void set_ladder_rung(std::string rung, double tol, long maxit, long gmres_m, double sign_ks) {
      utils::check(rung == "static" or rung == "dynamic",
                   "vertex_t::set_ladder_rung: unknown pol_vertex_rung \"{}\". Valid options are "
                   "\"static\" (default), \"dynamic\".", rung);
      if (rung == "dynamic") {
        utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                     "pol_vertex_rung = \"dynamic\" requires the DLR IAFT backend (iaft basis = \"dlr\").");
        utils::check(_ladder_legs != "ward",
                     "pol_vertex_rung = \"dynamic\" and pol_vertex_legs = \"ward\" double count "
                     "(proposal eq 24 cor. ii): choose one.");
        utils::check(sign_ks == -1.0 or sign_ks == 1.0, "pol_vertex_dyn_sign must be -1 or +1 (got {}).", sign_ks);
      }
      _ladder_rung = rung;
      _dyn_tol = tol; _dyn_maxit = maxit; _dyn_gmres = gmres_m; _dyn_sign = sign_ks;
      if (rung == "dynamic")
        app_log(1, "  [scGW-tilde T2] pol_vertex_rung = \"dynamic\": the ladder is resummed with the "
                   "FULL-FREQUENCY screened rung W(inu') (notes/dynbse_plan.md; two-family DLR "
                   "representation, static part exact, dynamic remainder by GMRES({}) to {:.1e}, "
                   "maxit {}); rung sign convention {:+.0f}.", gmres_m, tol, maxit, sign_ks);
    }
    std::string ladder_rung() const { return _ladder_rung; }
    bool ladder_dynamic_rung() const { return _ladder_rung == "dynamic"; }
    double ladder_dyn_tol() const { return _dyn_tol; }
    long ladder_dyn_maxit() const { return _dyn_maxit; }
    long ladder_dyn_gmres() const { return _dyn_gmres; }
    /** pol_vertex_dyn_rhs_block (default 32): the dynamic-rung solves run the Nm right-hand-side
     *  columns in blocks of this width (0 = all at once); the Krylov basis (gmres_m + 1 vectors of
     *  the block width) is the memory driver of the solve. Results agree to the solve tolerance. */
    void set_ladder_dyn_rhs_block(long nb) {
      utils::check(nb >= 0, "vertex_t::set_ladder_dyn_rhs_block: pol_vertex_dyn_rhs_block must be >= 0 (got {}).", nb);
      _dyn_rhs_block = nb;
    }
    long ladder_dyn_rhs_block() const { return _dyn_rhs_block; }
    /** pol_vertex_dyn_dump (default false): every finished (s, q, nu) unit of the dynamic solves is appended
     *  to "<prefix>.dynunits.<tag>.g<call>.r<rank>.bin"; a rerun of the same leg loads them instead of
     *  re-solving (a walltime kill loses only the units in flight). */
    void set_ladder_dyn_dump(bool on) { _dyn_dump = on; }
    bool ladder_dyn_dump() const { return _dyn_dump; }
    /** W-int-1b: pol_vertex_isdf_points_file (freeze the secondary ISDF points from a coarse run's
     *  <prefix>.secpts.h5) / pol_vertex_isdf_points_dump (write this run's points). */
    void set_isdf_points(std::string const &file, bool dump) { _isdf_points_file = file; _isdf_points_dump = dump; }
    std::string const &isdf_points_file() const { return _isdf_points_file; }
    bool isdf_points_dump() const { return _isdf_points_dump; }
    nda::array<long, 1> const &secondary_points() const { return _sec_ipts; }
    /** pol_vertex_wannier_frame ("aux" | "pair"): in Wannier mode the dynbse outputs are the aux/point-frame
     *  Pi(q)_{MN} on the (frozen-able) secondary points -- the interpolable frame -- or the same-cell MLWF-pair
     *  Pi_loc (the W-int-0 frame, = the Delta = 0 block of the pair-separation-resolved response). */
    void set_wannier_frame(std::string f) { _wannier_frame = std::move(f); }
    std::string const &wannier_frame() const { return _wannier_frame; }
    /** pol_vertex_interp_file / _col: the eps readout takes Pi(q)_{MN} (this mesh's q, the frozen points) from
     *  the file (a coarse run's <prefix>.pol_nu0.g<n>.h5, or the offline Route-B interpolant) instead of solving. */
    void set_pol_interp(std::string const &file, std::string const &col) { _pol_interp_file = file; _pol_interp_col = col; }
    std::string const &pol_interp_file() const { return _pol_interp_file; }
    std::string const &pol_interp_col() const { return _pol_interp_col; }
    /** pol_vertex_dyn_dense (default true): the dynamic rung as dense per-tau blocks K_d(s) = Kbig[W_d(s)] on the
     *  PH-symmetric half of the tau nodes (compute-bound gemms; nt/2 x D^2 complex per rank: 5.4 GB at Si 4^3/8,
     *  27 GB at C = [0,12)); false = the THC pair-space streaming route (memory-bandwidth-bound). */
    void set_ladder_dyn_dense(bool on) { _dyn_dense = on; }
    bool ladder_dyn_dense() const { return _dyn_dense; }
    /** pol_vertex_dyn_resolvent (default "inverse"): the static resolvent T_s = K_s (1 - Cb K_s)^-1 of the dynamic solver --
     *  "inverse" forms the explicit inverse and stores T_s dense (D x D); "lu" (P7) factorizes 1 - Cb K_s once (getrf, built
     *  blockwise) and applies T_s as a solve + one K_s gemm: no D^3 inverse, D^2 fewer words, identical to rounding. */
    void set_ladder_dyn_resolvent(std::string const &m) {
      utils::check(m == "inverse" or m == "lu", "vertex_t::set_ladder_dyn_resolvent: pol_vertex_dyn_resolvent must be inverse | lu (got \"{}\").", m);
      _dyn_resolvent = m;
    }
    std::string const &ladder_dyn_resolvent() const { return _dyn_resolvent; }
    /** pol_vertex_dyn_union_stride (default 1): the inu != 0 union grid keeps every n-th shifted G node (plus the
     *  last); the pair-pole cost falls ~n^2 while the G pole fit must stay clean (reported; gate it). */
    void set_ladder_dyn_union_stride(long n) {
      utils::check(n >= 1, "vertex_t::set_ladder_dyn_union_stride: pol_vertex_dyn_union_stride must be >= 1 (got {}).", n);
      _dyn_union_stride = n;
    }
    long ladder_dyn_union_stride() const { return _dyn_union_stride; }
    /** pol_vertex_dyn_table_mode (default 0): the inu != 0 twisted-pair tables R1/R3 -- 0 = fitted on the dense tau grid,
     *  1 = the exact partial fractions through the Dsq re-expansion (1/nu-amplified; the diagnostic route). */
    void set_ladder_dyn_table_mode(int m) {
      utils::check(m == 0 or m == 1, "vertex_t::set_ladder_dyn_table_mode: pol_vertex_dyn_table_mode must be 0 or 1 (got {}).", m);
      _dyn_table_mode = m;
    }
    int ladder_dyn_table_mode() const { return _dyn_table_mode; }
    /** pol_vertex_dyn_schedule (default "longest"): the order in which the dynamic scheduler hands out the (s, q, nu) units --
     *  "longest" = the heuristic (nu != 0 units, highest node first, then nu = 0); "measured" (P5) = longest-first by the wall
     *  time of the same unit in a previous call of this object (an SCF iteration reuses the previous one's), unseen first. */
    void set_ladder_dyn_schedule(std::string const &m) {
      utils::check(m == "longest" or m == "measured", "vertex_t::set_ladder_dyn_schedule: pol_vertex_dyn_schedule must be longest | measured (got \"{}\").", m);
      _dyn_schedule = m;
    }
    std::string const &ladder_dyn_schedule() const { return _dyn_schedule; }
    /** pol_vertex_dyn_iaft_prec (default "" = the loop's imaginary-axis grid): the dynamic pair algebra runs on its own
     *  DLR of this precision ("low" 1e-6, "medium" 1e-10, "high" 1e-13) with G and W interpolated from the loop's grid;
     *  the small-nu twisted algebra needs a finer class than the loop's "low" (dynbse_small_nu_1000). */
    void set_ladder_dyn_iaft_prec(std::string const &p) {
      utils::check(p.empty() or p == "low" or p == "medium" or p == "high",
                   "vertex_t::set_ladder_dyn_iaft_prec: pol_vertex_dyn_iaft_prec must be \"\", low, medium or high (got {}).", p);
      _dyn_iaft_prec = p;
    }
    std::string const &ladder_dyn_iaft_prec() const { return _dyn_iaft_prec; }
    /** pol_vertex_dyn_tfold (default 0 = off): at inu != 0 the twisted pair components with |eps_a| >= tfold |nu| are
     *  folded into the unshifted family (T_a = U_a^2 - i nu U_a^3 + (i nu)^2 U_a^4, error (nu/eps_a)^3): removes the
     *  tau-metric near-null directions behind the small-nu spurious modes of the resummation (Si: ratio 30). */
    void set_ladder_dyn_tfold(double r) {
      utils::check(r >= 0.0, "vertex_t::set_ladder_dyn_tfold: pol_vertex_dyn_tfold must be >= 0 (got {}).", r);
      _dyn_tfold = r;
    }
    double ladder_dyn_tfold() const { return _dyn_tfold; }
    /** pol_vertex_dyn_vmask_lo / _hi (default off): the vertex DLR nodes with lo < eps < hi (Ha, measured from mu) and the
     *  union's shifted G nodes in that interval are dropped -- the in-gap nodes of a gapped system carry no pair poles and
     *  host the small-nu spurious mode of the resummation (Si q_min nu_1: |Ritz| 427 on six nodes inside (-0.023, +0.01) Ha).
     *  Choose the interval strictly inside the quasiparticle gap; the reported table refit / G fit errors gate it. */
    void set_ladder_dyn_vmask(double lo, double hi) {
      utils::check(hi >= lo, "vertex_t::set_ladder_dyn_vmask: pol_vertex_dyn_vmask_hi must be >= _lo (got {}, {}).", lo, hi);
      _dyn_vmask_lo = lo; _dyn_vmask_hi = hi;
    }
    double ladder_dyn_vmask_lo() const { return _dyn_vmask_lo; }
    double ladder_dyn_vmask_hi() const { return _dyn_vmask_hi; }
    /** pol_vertex_dyn_gamma1_only (default false): evaluate only Gamma_1 (static + one dynamic rung on static-ladder
     *  legs), the first iterate, and skip the resummation GMRES -- 5-8x cheaper; the resummed column then repeats
     *  Gamma_1. Use when only the one-rung (quadratic) dynamic vertex is wanted. */
    void set_ladder_dyn_gamma1_only(bool on) { _dyn_gamma1_only = on; }
    bool ladder_dyn_gamma1_only() const { return _dyn_gamma1_only; }
    double ladder_dyn_sign() const { return _dyn_sign; }

    /**
     * scGW-tilde increment L2 (vertex_ladder.icc): the resummed pair-space ladder
     * polarization at the inu = 0 bosonic node, (nq, N_m, N_m) in THIS vertex's
     * secondary aux basis (all rungs >= 1; the n = 1 term is the static-rung Pi^C,
     * pinned by gate L1-b at machine precision). Requires an ACTIVE static-rung
     * secondary vertex with W0bar built (build_w0 this iteration). Replicated.
     */
    nda::array<ComplexType, 3> eval_pol_ladder_nu0(MBState &mb_state, THC_ERI auto &thc,
                                                   nda::array<ComplexType, 3> *Pi_dlam = nullptr);

    /**
     * Q3 increment I1 (notes/q3_bse_tier_spec.md section 4): the same resummed ladder at
     * ALL PH-sym POSITIVE bosonic half-grid nodes, (n_nu_half, nq_ibz, N_m, N_m). Half
     * index j is the full-mesh node nw_b/2 + j (verified against IAFT.icc's PH-sym
     * transforms; j = 0 is the inu = 0 node of eval_pol_ladder_nu0). lam_max, when given,
     * is RESIZED to n_nu_half and filled with the per-node rho(Xh Kt) watchdog (I3).
     * Replicated; same guards as eval_pol_ladder_nu0.
     */
    nda::array<ComplexType, 4> eval_pol_ladder_whalf(MBState &mb_state, THC_ERI auto &thc,
                                                     nda::array<double, 1> *lam_max = nullptr,
                                                     nda::array<ComplexType, 4> *Pi_dlam = nullptr);

    /**
     * Q4-C3b (notes/q4_c3b_orbital_ladder_dc_spec.md): the ORBITAL / chi-convention local
     * part of the SAME ladder -- the eq-7 bosonic DC's ladder half proper (the C3
     * THC-adjoint object is a diagnostic, R-Q4-2 AMENDMENT). Returns
     * (n_nu_half, nq_ibz, nab, nab), nab = norb^2, in the eval_Pi_rpa_dc pair pack
     * abcd = (m, n, m', n'); the q-average is the caller's (star/trev rule).
     * U_skia (ns, nk_FULL, norb, nc_ladder_window) are the MLWF legs; the derivation of
     * the leg conjugation is in vertex_ladder.icc's header and is pinned by
     * ladder_loc_gate. Pi_onerung_loc, when given, is RESIZED and filled with the
     * one-rung analog. Same guards/scheduling as eval_pol_ladder_whalf; replicated.
     */
    nda::array<ComplexType, 4>
    eval_pol_ladder_loc_whalf(MBState &mb_state, THC_ERI auto &thc,
                              nda::array<ComplexType, 4> const &U_skia,
                              nda::array<ComplexType, 4> *Pi_onerung_loc = nullptr);

    /** Q4-C3b gates G2/G3 (vertex_ladder.icc): the leg pin and the chi-convention pin,
     *  both against brute-force references written from the definitions. NOSYM only. */
    struct ladder_loc_diag {
      bool sym_active = false;
      long norb = 0, nnu_checked = 0;
      double onerung_resid = -1.0, onerung_scale = 0.0;  // G2 (machine class)
      double bub_resid_w = -1.0, bub_scale = 0.0;        // G3 chi0 vs G-space (machine)
      double bub_resid_phsym = -1.0;                     // G3 vs the PH-sym tau route
      // ... and its exact characterization: the PH-sym half-grid route is the SYMMETRIC
      // part of the tau object (tau_asym is the bubble's PH asymmetry; after symmetrizing,
      // the routes agree at machine class)
      double bub_tau_asym = -1.0, bub_resid_phsym_sym = -1.0;
      double loc_ph_sym = -1.0;                          // |P^lad_loc(-nu) - P^lad_loc(nu)|
      double lad_loc_max = 0.0;                          // scale material for G4
    };
    ladder_loc_diag ladder_loc_gate(MBState &mb_state, THC_ERI auto &thc,
                                    nda::array<ComplexType, 4> const &U_skia);

    /** Q3 gates on the multi-nu evaluator (vertex_ladder.icc; spec section 5 Q3-c). */
    struct ladder_whalf_diag {
      double node_map_resid = -1.0;   // half-grid output vs the full mesh at nw_b/2 + j
      double ph_sym_resid = -1.0;     // |Pi_ladder(-nu) - Pi_ladder(+nu)| (transform licence)
      double ladder_max = 0.0;        // max |Pi_ladder| over the full mesh
      double lam_nu0 = -1.0, lam_max = -1.0;   // the I3 watchdog
      double lam_nu0_scaled = -1.0;   // ... with the rung scaled by `scale`
      double scale = 1.0;
    };
    ladder_whalf_diag ladder_whalf_gate(MBState &mb_state, THC_ERI auto &thc,
                                        double scale = 2.0);

    /** P4 gate diagnostics (vertex_ladder.icc / the parallel-memory design note). */
    struct ladder_p4_diag {
      double j1_resid = -1.0;          // rs (tol_L = 0) j=1 vs the direct one-rung
      double neumann_resid = -1.0;     // converged Neumann vs the direct resolvent
      long rungs_used = 0;
      double dropped_frac_test = -1.0; // tol_L = 0.5 kernel: dropped ||w(L)||
      double j1_resid_trunc = -1.0;    // ...and its j=1 error (monotone meter)
      // the sampled kept-(P,Q) apply (design 4b.1 step (ii)):
      double pq_all_j1_resid = -1.0;       // sampled, ALL pairs kept: j=1 vs direct
      double pq_all_neumann_resid = -1.0;  // ...converged Neumann vs direct resolvent
      double pq_all_max_reldiff = -1.0;    // ...max|sampled - dense-rs|/max|dense-rs|, j=1
      double pq_kept_frac_test = -1.0;     // tau_PQ meter kernel: kept pair fraction
      double pq_dropped_wfrac_test = -1.0; // ...dropped pair-channel ||w||_F fraction
      double pq_j1_resid_trunc = -1.0;     // ...j=1 error (the monotone pair meter)
    };
    ladder_p4_diag ladder_p4_gates(MBState &mb_state, THC_ERI auto &thc);

    /**
     * C.2 IBZ-symmetry gate (vertex_ladder.icc): the pair-space one-rung rebuild vs
     * the pi_c_accumulate_w anchor with the SAME symmetry context threaded through
     * both -- the L1-b machine-precision identity with the Xhat rotations live.
     */
    struct ladder_sym_diag {
      bool sym_active = false;    // the mesh is IBZ-reduced (rotations exercised)
      double l1b_resid = -1.0;    // one-rung rebuild vs the Pi^C anchor (symc threaded)
      double ladder_frac = -1.0;  // >= 2-rung content
      double onerung_max = 0.0, ladder_max = 0.0;
    };
    ladder_sym_diag ladder_sym_gate(MBState &mb_state, THC_ERI auto &thc);

    /**
     * P3 gate (C.3, vertex_ladder.icc): scheduling invariance of the pair-space
     * ladder -- P2 round-robin, groups-of-1, and one-group-of-all-ranks against the
     * replicated reference. Disjoint-write group assembly => BITWISE (0.0) expected.
     */
    struct ladder_p3_diag {
      double p2_max_diff = -1.0;    // rank round-robin vs replicated
      double grp1_max_diff = -1.0;  // groups-of-1 grid vs replicated
      double grpN_max_diff = -1.0;  // one group of all ranks vs replicated
    };
    ladder_p3_diag ladder_p3_gate(MBState &mb_state, THC_ERI auto &thc);

    /**
     * scGW-tilde TIER 1.5 increment T15-b gates (notes/tier15_ward_legs_plan.md section
     * 6), on the very state the loop used (nosym window mode; requires W0bar):
     *   bub_pin        : the pair kernel's BARE zero-rung bubble -spin/nk sum_k conj(D)^T Cb D
     *                    at inu = 0 vs eval_pol_pi0's Hadamard bubble (the normalization of
     *                    Delta P^Lambda is that of the RPA bubble) -- machine class;
     *   fit_err, rr    : the aux-grid DLR pole-fit reconstruction errors / residue ratios
     *                    of G and Sigma_c (the fit-class floor every fixture G-g inherits);
     *   gamma_*        : the G-g meter at the Gamma transfer -- |vertex-traced pair
     *                    propagator|, bare vs Lambda-corrected, at the first three PH-sym
     *                    nodes (ratio ~ fit class at a full band window; the window
     *                    truncation's C1 violation otherwise);
     *   pole_vs_tau    : max |Cb_pole - Cb_tau| / |Cb_tau| (the pole route's representation
     *                    error on real data);
     *   asym_*         : (M,N) asymmetry of the ladder / Delta P^Lambda outputs before the
     *                    Hermitization;  dlam_max / ladder_max: the scales.
     */
    struct ward_legs_diag {
      double bub_pin = -1.0;
      double fit_err_G = -1.0, fit_err_S = -1.0, rr_G = -1.0, rr_S = -1.0;
      double gamma_bare[3] = {-1.0, -1.0, -1.0}, gamma_lam[3] = {-1.0, -1.0, -1.0};
      double pole_vs_tau = -1.0;
      double asym_ladder = -1.0, asym_dlam = -1.0;
      double dlam_max = -1.0, ladder_max = -1.0;
    };
    ward_legs_diag ward_legs_gate(MBState &mb_state, THC_ERI auto &thc);

    /**
     * scGW-tilde Tier 2 full frequency, increment D2 (notes/dynbse_plan.md; vertex_dynbse.icc):
     * the resummed DYNAMIC-rung BSE driver's gates on the readout instance (nosym window mode;
     * requires W0bar AND the W-bar cache -- call update_w + cache_w first):
     *   a0_resid       : THC rung operator (nu'-constant W) vs the explicit Kbig -- machine class;
     *   a_resid        : static limit vs the SIGN-CORRECTED L2 resolvent -ladder(-W0) -- 1e-12 class;
     *   a_l2_diff      : static limit vs the as-implemented L2 (the even-order rung sign; reported);
     *   b_resid        : one bare dynamic rung vs pi_c_accumulate_w(Z = 0, W_dyn - W_dyn(0)) -- fit class;
     *   gmres_vs_neumann / gam1_consistency : the two solvers agree; ritz_max / contraction_max /
     *   it_max : the watchdog and iteration counts; herm : (M,N) asymmetry of the resummed vertex;
     *   dyn_vs_static / gam1_vs_static : the dynamic correction at inu = 0 relative to the static ladder.
     */
    struct dynbse_diag {
      double a0_resid = -1.0, a_resid = -1.0, a_l2_diff = -1.0, b_resid = -1.0, b_continuity = -1.0;
      double lam_xhkt_re = 0.0, lam_xhkt_im = 0.0;   // the signed dominant eigenvalue of Xh Kt (the sign pin)
      double fit_err_G = -1.0, rr_G = -1.0, dsq_err = -1.0, wtau_sym = -1.0, refit_err_1 = -1.0, refit_err = -1.0;
      double gmres_vs_neumann = -1.0, gam1_consistency = -1.0, ritz_max = -1.0, contraction_max = -1.0;
      double block_resid = -1.0;      // (C'') RHS-blocked vs unblocked solve
      double a0_ft_resid = -1.0, a0_ft_check = -1.0;   // (A0-FT) mesh-Fourier k-sum vs direct; the mesh identity check
      double nu1_resid = -1.0, nu1_gfit = -1.0;   // (C') union no-mask at the first positive node: |resummed - static|/|static|, G fit
      bool nu1_done = false;
      long it_max = 0, it_max_neumann = 0;
      bool all_converged = false;
      double herm = -1.0, static_max = 0.0, onerung_max = 0.0, dyn_max = 0.0, gam1_max = 0.0;
      double dyn_vs_static = -1.0, gam1_vs_static = -1.0;
    };
    dynbse_diag dynbse_gate(MBState &mb_state, THC_ERI auto &thc, bool quick = false);

    /**
     * scGW-tilde Tier 2 full frequency, increment D3: the inu = 0 columns of the dynamic-rung
     * ladder on the readout instance (nosym window mode; requires W0bar and the W-bar cache):
     * (nq, N_m, N_m) blocks of rungs >= 1, (M,N)-Hermitized --
     *   Pi_static : the static ladder (K_s = sign_ks Kbig/nk; the sign-corrected L2 at -1)
     *   Pi_dyn1   : the one bare dynamic rung (the Pi^C anchor's dynamic part)
     *   Pi_gam1   : the first iterate (one dynamic rung dressed by static ladders)
     *   Pi_dyn    : the resummed dynamic-rung vertex
     * with the solve meters.
     */
    struct dynbse_nu0_result {
      nda::array<ComplexType, 3> Pi_static, Pi_dyn1, Pi_gam1, Pi_dyn;
      double ritz_max = -1.0, refit_err = -1.0, fit_err_G = -1.0, rr_G = -1.0, herm_dyn = -1.0;
      long it_max = 0, it_sum = 0, nunits = 0;
      bool all_converged = false;
      double t_total = 0.0, t_solve = 0.0, rss_gb = 0.0;
    };
    dynbse_nu0_result eval_pol_dynbse_nu0(MBState &mb_state, THC_ERI auto &thc, long gen = 0);
    /** eps(q_i, i nu) cuts (2026-09-11): the four dynamic-rung columns {static, static + one dynamic
     *  rung, Gamma_1, resummed} on a list of PH-sym bosonic HALF nodes at a subset of transfers,
     *  (4, n_nodes, nq, Nm, Nm) replicated (zeros at transfers outside the subset). The inu != 0
     *  framework: union {U, T} basis, in-gap mask, tau metric, GMRES(>= 12), readout stop. */
    struct dynbse_cut_result {
      std::vector<long> half_nodes, qsel;
      nda::array<ComplexType, 5> Pi;
      nda::array<ComplexType, 4> Pi_bub;   // LFF-aux L-0: the window bubble (spin/nk) D^dag Cb D per (node, q), same frame
      double ritz_max = -1.0, refit_err = -1.0, fit_err_G = -1.0;
      long it_max = 0, it_sum = 0, nunits = 0;
      bool all_converged = false;
      double t_total = 0.0, t_solve = 0.0, rss_gb = 0.0;
    };
    dynbse_cut_result eval_pol_dynbse_cut(MBState &mb_state, THC_ERI auto &thc, std::vector<long> const &half_nodes,
                                          std::vector<long> const &qsel, long gen = 0, bool bubble_only = false);

    /**
     * scGW-tilde increment L1 (vertex_ladder.icc): the C-window pair bubble
     * Pi-bar^0_MN(q, tau_pos) in the SECONDARY aux basis, (nt_half, nq, N_m, N_m),
     * replicated -- house RPA conventions (rpa_pi.icc Hadamard pairing, -spin/Nk,
     * PH-sym tau half grid). NOSYM window mode only at L1.
     */
    nda::array<ComplexType, 4> eval_pol_pi0(MBState &mb_state, THC_ERI auto &thc);

    /** L1 gate diagnostics -- see vertex_ladder.icc for the derivation. */
    struct ladder_l1_diag {
      double l1a_eta = -1.0;      // upfold(Pi-bar^0) vs the C-masked global bubble
      double l1b_resid = -1.0;    // pair-space one-rung rebuild vs the Pi^C anchor:
                                  // an algebraic identity -- machine precision, no scale
      double ladder_frac = -1.0;  // ||Pi_ladder - Pi_onerung|| / ||Pi^C||: >= 2-rung content
      double onerung_max = 0.0, ladder_max = 0.0;
    };
    ladder_l1_diag ladder_l1_gates(MBState &mb_state, THC_ERI auto &thc);

    /** DIAGNOSTIC, default 0 -- drop one cut-piece so its exact energy contribution can be
     *  read off. 1 = drop Sigma^(L,r), 2 = drop Sigma^(C,x), 3 = drop both (which isolates
     *  P^{C,L}'s effect through Sigma_GW). See _bl_drop. */
    void set_bl_drop(int which) {
      utils::check(which >= 0 and which <= 3,
                   "vertex_t::set_bl_drop: must be 0, 1, 2 or 3 (got {}).", which);
      _bl_drop = which;
    }
    int bl_drop() const { return _bl_drop; }
    /** TEST-LEVEL (L-7 gate N2): skip the polarization cut Pi^C of an ACTIVE dynamic-rung vertex, so that (Sigma - Sigma_R0)
     *  of a one-iteration run is the G^3 W^2 self-energy cut alone on the RPA W (the pair-resolved reference). Not a physics
     *  knob: the two cuts belong together (CLAUDE.md invariant 1). */
    void set_skip_pi_c(bool on) { _skip_pi_c = on; }
    bool skip_pi_c() const { return _skip_pi_c; }

    /** DIAGNOSTIC, default OFF -- THE CONSTANT-RUNG ABSOLUTE PIN. Forces pi^dyn's total
     *  rung to be W0bar at every frequency, i.e. exactly Pi^{C,0}'s rung, so X^L must
     *  collapse to the grid's representability floor. See _bl_pidyn_const_rung. */
    void set_bl_pidyn_const_rung(bool on) { _bl_pidyn_const_rung = on; }
    bool bl_pidyn_const_rung() const { return _bl_pidyn_const_rung; }

    /** P0.3: scale the analytic Gamma head by lambda EVERYWHERE the vertex inserts it.
     *  lambda = 1 is untouched; lambda = 0 takes the same branches as "ignore_g0". The scan
     *  over lambda separates one-rung-Gamma (linear) from coincident-Gamma (quadratic)
     *  contributions to Sigma^{C,x}. See _bl_head_scale. */
    void set_bl_head_scale(double lambda) {
      utils::check(std::isfinite(lambda) and lambda >= 0.0,
                   "vertex_t::set_bl_head_scale: lambda must be finite and >= 0 (got {}).",
                   lambda);
      _bl_head_scale = lambda;
      if (lambda != 1.0)
        app_log(1, "  [WARNING] vertex Gamma-head STRENGTH set to lambda = {} (default 1). "
                   "The analytic q -> 0\n"
                   "            head is rescaled at EVERY site the vertex inserts it "
                   "(Sigma^C, Pi^C, cache_w, W0).\n"
                   "            This is a deliberately WRONG finite-size treatment: the "
                   "energies are diagnostic\n"
                   "            only. lambda = 0 is equivalent to div_treatment = "
                   "\"ignore_g0\".", lambda);
    }
    double bl_head_scale() const { return _bl_head_scale; }
    // running max relative deviation |factorized - kernel| / |kernel| over this vertex's
    // CHECK-mode evaluations (0 unless vertex_pidyn = "check" has run).
    double pidyn_check_max() const { return _pidyn_check_max; }
    long vertex_iter() const { return _vertex_iter; }
    long bare_rung_uses() const { return _bare_rung_uses; }
    double sym_leakage_mean() const { return _sym_leak_mean; }
    // running max of the G_CC G-rotation consistency residual across this vertex's
    // eval calls (0 until the first symmetric evaluation; 0 on symmetry-free meshes).
    double g_rotation_max() const { return _g_rot_max; }
    // B-L Gamma-head A/B meters from the LAST eval_Sigma_C response evaluation (see the
    // members). -1 means NEVER MEASURED -- B-S, or head_ok false -- which is distinct from
    // a measured zero and must not be read as "the head is clean".
    double diag_head_hl() const { return _diag_head_hl; }
    double diag_head_hs() const { return _diag_head_hs; }
    double diag_head_removed_frac() const { return _diag_head_removed; }
    double diag_resp_share() const { return _diag_resp_share; }
    double diag_xl_rel() const { return _diag_xl_rel; }
    // B-L validity meter, LAST eval_Sigma_C: max_nu |W - W0| / max |W0|. -1 = never
    // measured (B-S has no dW). Near or above 1 => B-L's first-order expansion is not
    // controlled. Do NOT substitute the i.nu = 0 value for this -- see the member.
    double diag_dw_rel() const { return _diag_dw_rel; }
    // the SAME meter in the chi (G = 0) channel at q = Gamma, and its within-run head-free
    // control (the worst q != Gamma, where no head is ever inserted). -1 = never measured.
    // These are the P0.1 numbers: they say whether the Gamma head CANCELS in W - W0.
    double diag_dw_head_rel() const { return _diag_dw_head_rel; }
    double diag_dw_head_bg() const { return _diag_dw_head_bg; }
    // ... and the ones that actually carry the P0.1 result: the ABSOLUTE head-channel
    // content of dW(Gamma) in a.u. (comparable across q -> 0 policies, which no ratio is),
    // its COHERENCE as a fraction of the chi-aligned rank-1 ceiling (1 = dW(Gamma) IS
    // c chi chi^dag), and the same channel at i.nu = 0 -- whose distance from the max is
    // the blind spot of the log's long-standing |W(q,0) - W0(q)| line.
    double diag_dw_head_abs() const { return _diag_dw_head_abs; }
    double diag_dw_head_coh() const { return _diag_dw_head_coh; }
    double diag_dw_head_nu0() const { return _diag_dw_head_nu0; }
    // secondary-basis diagnostics (0 until build_secondary_basis has run):
    long secondary_Nm() const { return _Nm; }                    // actual secondary rank N_m
    double secondary_cond_s_max() const { return _cond_s_max; }  // max_q regularized downfold cond

  }; // vertex_t

} // solvers
} // methods

#endif //COQUI_VERTEX_T_H
