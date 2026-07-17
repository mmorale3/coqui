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

namespace methods {
namespace solvers {

  /**
   * @brief vertex_t class
   *
   * Phi-derivable second-order-exchange vertex correction on top of scGW,
   * with all internal lines restricted to a contiguous near-E_F orbital
   * subspace C = [band_window.first(), band_window.last()).
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
   * STATUS: both kernels are implemented for symmetry-free meshes
   * (nkpts == nkpts_ibz == nqpts; anything else aborts loudly):
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

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;

    // type of the vertex correction: "none" or "2nd_exchange"
    std::string _vertex_type = "none";

    // contiguous orbital range [first, last) defining the subspace C
    nda::range _band_window = nda::range(0, 0);

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
     * @param X_skPa - [INPUT] replicated GLOBAL collocation (ns, nk, Np, nbnd)
     * @param kmq    - [INPUT] (nq, nk) index map of k - q
     * @param iq_gamma - [INPUT] index of q = Gamma
     */
    void build_secondary_basis(THC_ERI auto const &thc,
                               nda::array<ComplexType, 4> const &X_skPa,
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
    // vertex requested AND C is non-empty; C = empty set must be an exact no-op
    bool active() const { return enabled() and _band_window.size() > 0; }

  }; // vertex_t

} // solvers
} // methods

#endif //COQUI_VERTEX_T_H
