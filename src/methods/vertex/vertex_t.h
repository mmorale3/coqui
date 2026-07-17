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
     */
    vertex_t(const imag_axes_ft::IAFT *ft,
             std::string vertex_type,
             nda::range band_window,
             long nbnd,
             std::string div_treatment = "ignore_g0",
             std::string isdf_mode = "global",
             long isdf_rank = -1,
             double isdf_svd_tol = 1e-8);

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
    // geometry-fixed cache (built lazily on the first kernel evaluation)
    bool _secondary_ready = false;
    long _Nm = 0;                          // ACTUAL secondary rank (selection may
                                           // return fewer points than requested)
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
                       nda::array<ComplexType, 4> const &X_w,
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
                               nda::array<ComplexType, 4> const &X_glob, long orb0,
                               nda::array<long, 2> const &kmq, long iq_gamma);

  public:
    std::string vertex_type() const { return _vertex_type; }
    nda::range band_window() const { return _band_window; }
    std::string div_treatment() const { return _div_treatment; }
    // runtime-selectable q->0 policy (validated; see constructor doc)
    void set_div_treatment(std::string div);

    // Refinement 2 accessors
    std::string isdf_mode() const { return _isdf_mode; }
    bool secondary() const { return _isdf_mode == "secondary"; }
    // ACTUAL secondary rank N_m (0 until the basis has been built)
    long secondary_rank() const { return _Nm; }

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
    double sym_leakage_mean() const { return _sym_leak_mean; }

  }; // vertex_t

} // solvers
} // methods

#endif //COQUI_VERTEX_T_H
