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


#ifndef COQUI_SCR_COULOMB_T_H
#define COQUI_SCR_COULOMB_T_H

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "IO/app_loggers.h"

#include "mean_field/MF.hpp"
#include "methods/embedding/projector_boson_t.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {
namespace solvers {
  class vertex_t;
  // TODO
  //    1. timer
  /**
   * @brief scr_coulomb_t class
   *
   * This class handles the computation of screened Coulomb interactions for RPA and beyond theories.
   *
   * The RPA-beyond corrections are controlled by the `screen_type` based on
   * whether certain keywords appear in the `screen_type` string.
   *
   * Meaningful keywords in `screen_type` include:
   *   - "crpa"  - Subtracting q-dependent RPA polarization function within an active subspace
   *   - "edmft" - Adding EDMFT correction to the RPA polarization function
   *
   * For examples:
   *   1) screen_type = "ABC" will return RPA polarization function
   *   2) screen_type = "crpa" will return cRPA polarization function
   *   3) screen_type = "edmft" or "gw_edmft" or "edmft_ABC" will all return EDMFT-corrected RPA polarization function
   *   4) screen_type = "crpa_edmft" will return cRPA polarization function with EDMFT correction
   *
   * Usage:
   *   scr_coulomb_t scr_eri(..., screen_type, ...);
   *
   *   // compute the screened interaction
   *   scr_eri.compute(...);
   *
   *   // compute polarzability
   *   auto Pi = scr_eri.eval_Pi_qdep(...);
   */
  class scr_coulomb_t {
  public:
    using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
    template<nda::MemoryArray Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<nda::MemoryArray Array_base_t>
    using dsArray_t = math::shm::distributed_shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;

  public:
    scr_coulomb_t(
        const imag_axes_ft::IAFT *ft,
        std::string screen_type,
      std::string div = "gygi");

    scr_coulomb_t(scr_coulomb_t const&) = default;
    scr_coulomb_t(scr_coulomb_t &&) = default;
    scr_coulomb_t& operator=(const scr_coulomb_t &) = default;
    scr_coulomb_t& operator=(scr_coulomb_t &&) = default;

    ~scr_coulomb_t() {}

    /**
     * Compute the dynamic screened interaction and store it in the MBState.
     * @param mb_state  - [INPUT/OUTPUT] MBState object to store the results
     * @param thc       - [INPUT] THC bare Coulomb interaction
     */
    void update_w(MBState &mb_state, THC_ERI auto &thc, long h5_iter=-1);
    void update_w(MBState &mb_state, Cholesky_ERI auto &chol, long h5_iter=-1);

    /**
     * Evaluate the screened interaction from a imaginary-time polarizability
     * @tparam w_out - true: output W(iw); false: output W(t)
     * @param dPi_tqPQ_pos - [INPUT] Polarization function with particle-hole symmetry
     * @param thc          - [INPUT] THC ERI
     * @param reset_input  - [INPUT] free the memory of dPi_tqPQ_pos
     * @param w_pgrid      - [OPTIONAL] processor grid for W(iw)
     * @param w_bsize      - [OPTIONAL] block size for W(iw)
     * @return - Screened interaction W in the THC product basis
     */
    template<bool w_out, nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto dyson_W_from_Pi_tau(memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
                             THC_ERI auto &thc, bool reset_input,
                             std::array<long, 4> w_pgrid = {0, 0, 0, 0},
                             std::array<long, 4> w_bsize = {0, 0, 0, 0})
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    static auto W_omega_proc_grid(long nproc, long nqpts_ibz, long nw_b, long Np)
    -> std::tuple<std::array<long, 4>, std::array<long, 4>> {
      // setup pgrid for Pi_wPQ: maximize nqpools: a) nqpts = i * b; b) np = i * bb
      // not sure if this is the optimal configuration
      long np = nproc;
      long nw_half = (nw_b%2==0)? nw_b/2 : nw_b/2 + 1;

      long nqpools = utils::find_proc_grid_max_npools(np, nqpts_ibz, 0.2);
      np /= nqpools;
      long nwpools = utils::find_proc_grid_max_npools(np, nw_half, 0.2);
      np /= nwpools;
      long np_P = utils::find_proc_grid_min_diff(np, 1, 1);
      long np_Q = np / np_P;
      utils::check(nqpools > 0 and nqpools <= nqpts_ibz,
                   "scr_coulomb_t:: nqpools <= 0 or nqpools > nqpts_ibz. nqpools = {}", nqpools);
      utils::check(nqpools > 0 and nwpools <= nw_half,
                   "scr_coulomb_t:: nwpools <= 0 or nwpools > nw_half. nwpools = {}", nwpools);
      utils::check(nproc % nqpools == 0, "gw_t:: gcomm.size() % nqpools != 0");
      utils::check(nproc % (nqpools * nwpools) == 0, "gw_t:: gcomm.size() & (nqpools*nwpools) != 0");


      std::array<long, 4> w_pgrid = {nwpools, nqpools, np_P, np_Q}; // (w, q, P, Q)

      // Setup square block size: make sure block sizes produce at least one full block per task
      std::array<long, 4> w_bsize;
      w_bsize.fill(1);
      w_bsize[2] = std::min( {(long)1024, Np/w_pgrid[2], Np/w_pgrid[3]});
      w_bsize[3] = w_bsize[2];

      return std::make_tuple(w_pgrid, w_bsize);
    }

    /**
     * Evaluate the screened interaction W in place from a Matsubara polarizability
     * @param dPi_wqPQ - [INPUT/OUTPUT] polarizability / screened interaction: (nw, nqpts_ibz, Np, Np)
     * @param thc      - [INTPUT] THC ERI object
     */
    template<nda::MemoryArray Array_4D_t, typename communicator_t>
    void dyson_W_in_place(memory::darray_t<Array_4D_t, communicator_t> &dPi_wqPQ,
                          THC_ERI auto &thc);

    /**
     * Evaluate q-dependent polarization function
     *
     * @param G_tskij         - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
     * @param thc             - [INPUT] THC-ERI instance
     * @param screen_type     - [INPUT] method for polarizability
     * @param proj            - [INPUT] bosonic projector (optional)
     * @param coqui_h5_prefix - [INPUT] prefix for CoQui h5 where pi_imp/pi_dc are stored (optional)
     * @param pi_imp          - [INPUT] impurity polarizability (optional)
     * @param pi_dc           - [INPUT] double-counting polarizability (optional)
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_qdep(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc,
                      const projector_boson_t* proj=nullptr,
                      const nda::array_view<ComplexType, 5> *pi_imp=nullptr,
                      const nda::array_view<ComplexType, 5> *pi_dc=nullptr)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    auto eval_Pi_qdep(MBState &mb_state, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    // TODO function to return Pi(w) directly

    /**
     * compute the Green's function on the THC interpolating points on a specific time point
     * @param it       - [INPUT] time index
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory
     * @param dG_skPQ  - [OUTPUT] Green's function on THC interpolating points at it in a distributed array
     * @param thc      - [INPUT] THC-ERI instance
     * @param minus_t  - [INPUT] false: compute G at tau=(0,beta/2); true: compute G at tau=(0,-beta/2)
     */

    /**
     * Specialized version of FT function for distributed array along (tau, w)-axes
     */
    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto tau_to_w(memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
                  std::array<long, 4> w_pgrid_out,
                  std::array<long, 4> w_bsize_out = {},
                  bool reset_input = false)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
    auto w_to_tau(memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
                  std::array<long, 4> t_pgrid_out,
                  std::array<long, 4> t_bsize_out = {},
                  bool reset_input = false)
    -> memory::darray_t<local_Array_t, mpi3::communicator>;

    template<typename comm_t>
    void dump_eps_inv_head(const nda::ArrayOfRank<2> auto &eps_inv_head_tq,
                           const nda::ArrayOfRank<1> auto &eps_inv_head_t,
                           std::string coqui_h5_prefix, long iter,
                           comm_t &comm, mf::MF &mf);

  private:
    /**
     * Evaluate polarization function by computing the convolution on the R space
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory:
     *                   (nts, ns , nkpts_ibz, nbnd, nbnd)
     * @param thc      - [INPUT] THC-ERI instance
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_rpa_Rspace(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    /**
     * Evaluate polarization function by computing the convolution on the k space
     * @param G_tskij - [INPUT] Green's function in a primary basis on shared memory:
     *                   (nts, ns , nkpts_ibz, nbnd, nbnd)
     * @param thc      - [INPUT] THC-ERI instance
     * @return - Polarization function in k space: (nts, nqpts_ibz, Np, Np)
     */
    auto eval_Pi_rpa_kspace(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    // details of eval_Pi_all_kspace
    template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
    void eval_Pi_rpa_k_impl(const memory::darray_t<Array_5D_t, communicator_t> &dGp_sktPQ_c,
                            const memory::darray_t<Array_5D_t, communicator_t> &dGn_sktPQ,
                            memory::darray_t<Array_4D_t, communicator_t> &dPi_qtPQ,
                            THC_ERI auto &thc);

    auto eval_Pi_rpa_active(const nda::ArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc,
                            const projector_t &proj, int scheme)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

    auto upfold_pi_local(
        const nda::MemoryArrayOfRank<5> auto &Pi_loc_tabcd,
        THC_ERI auto &thc,
        const projector_boson_t &proj,
        std::array<long, 4> pgrid, std::array<long, 4> bsize)
    -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;

    std::string _screen_type = "";

    std::string _div_treatment;
    // scGW-tilde CVV head (div_treatment = "cvv"; notes/scgwt_implementation_plan.md
    // increments C1-C4): R-shell truncation tolerance handed to cvv_head_t
    // ([gw] cvv_rspace_tol). Stored at C0; consumed when the update_w head fill
    // lands (C4).
    double _cvv_rspace_tol = 1e-6;
    utils::TimerManager _Timer;

    // optional second-order-exchange vertex correction (ISDF-Vertex, not owned).
    // nullptr or an inactive vertex leaves the RPA polarizability untouched.
    vertex_t* _vertex = nullptr;

    // scGW-tilde L2 (pol_vertex = "ladder", READOUT-ONLY -- stance i of the proposal):
    // a PRIVATE static-rung secondary vertex built from the pol_* knobs of _vertex,
    // used only to build W0bar + the pair-space ladder for the eps_M report. It is
    // never attached to the Sigma side and nothing is injected into P (that is
    // increment L3, gated on ruling R1). shared_ptr: vertex_t is fwd-declared here.
    std::shared_ptr<vertex_t> _pol_vtx;
    // lazily construct _pol_vtx from _vertex's pol_* knobs
    void ensure_pol_vertex(THC_ERI auto &thc);
    // replicated (nq, Np, Np) copy of the inu = 0 row of the RPA Pi (PH-sym tau axis
    // contracted with vertex_w0_detail::nu0_transform_row) -- the eps-Dyson input.
    // Np^2-per-q replicated: readout-scale only (documented memory note).
    template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
    nda::array<ComplexType, 3> gather_nu0_row(memory::darray_t<Array_t, communicator_t> &dPi_tqPQ);
    // the L2 readout: ladder at nu = 0, upfold via the readout vertex's t(q), redo the
    // single-frequency Dyson with P0 and P0 + dP_ladder, report eps_M(q) both ways.
    // eps_inv_head_q (the loop's OWN q-resolved eps^-1 head on the PH-sym tau half grid,
    // div-treatment independent) is optional: when given, the SAME q_min is converted to
    // eps_M at inu = 0 and stored as the loop-side leg of gate Q3-b(i).
    void pol_ladder_eps_readout(MBState &mb_state, THC_ERI auto &thc,
                                nda::array<ComplexType, 3> const &Pi0_qPQ,
                                nda::array<ComplexType, 2> const *eps_inv_head_q = nullptr);
    // last readout values at q_min (RPA, +ladder); -1 before the first readout
    double _pol_eps_rpa = -1.0, _pol_eps_ladder = -1.0;
    // Q3: the LOOP's own eps_M(q_min, inu = 0) from eps_inv_head_q -- the other route of
    // the Q3-b(i) identity (tau-Dyson of the injected Pi vs the readout's single-frequency
    // Dyson). -1 before the first readout.
    double _pol_eps_loop = -1.0;

    // Q3 I2 (notes/q3_bse_tier_spec.md section 4): inject the resummed ladder into the
    // RPA polarizability IN PLACE -- P_latt = P^RPA + P^lad. The upfold and the nu -> tau
    // transform run on the LOCAL (P, Q) block only; no replicated (nu|t, q, Np, Np) array
    // is ever formed. Called from update_w between the readout's build_w0 hook and the
    // Dyson (ordering pinned by R-Q3-3).
    template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
    void inject_pol_ladder(MBState &mb_state, THC_ERI auto &thc,
                           memory::darray_t<Array_t, communicator_t> &dPi_tqPQ);
    // Q3 injection meters of the last update_w: the watchdog rho(Xh Kt) at inu = 0 and its
    // max over nu, the nu -> tau -> nu round-trip residual, and ||P^lad||/||P^RPA||.
    double _pol_lam_nu0 = -1.0, _pol_lam_max = -1.0;
    double _pol_r_rt = -1.0, _pol_lad_ratio = -1.0;

  public:
    // scGW-tilde L2: the last ladder eps_M readout (RPA, +ladder) at q_min
    std::pair<double, double> pol_eps_readout() const {
      return {_pol_eps_rpa, _pol_eps_ladder};
    }
    // Q3: the loop-side eps_M(q_min, inu = 0) of the same iteration (gate Q3-b(i))
    double pol_eps_loop() const { return _pol_eps_loop; }
    // Q3: the last injection's watchdog / meter values (gate Q3-c); -1 if never injected
    double pol_lambda_nu0() const { return _pol_lam_nu0; }
    double pol_lambda_max() const { return _pol_lam_max; }
    double pol_round_trip() const { return _pol_r_rt; }
    double pol_ladder_ratio() const { return _pol_lad_ratio; }
    // Q3 gates (spec section 5 Q3-c/Q3-d): the PRIVATE ladder instance, so a test can
    // re-run the anchor identity on the very state the loop used. nullptr before the
    // first update_w with an active pol vertex.
    vertex_t* pol_vertex_instance() { return _pol_vtx.get(); }

  private:

    // optional container for screened interaction
    // TODO Remove these
    std::optional<memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator> > _dW_qtPQ_opt;
    std::optional<nda::array<ComplexType, 1> > _eps_inv_head;

    // scGW-tilde C4 (div_treatment = "cvv"): lazily built H0 store for the covariant
    // velocity (hamilt::set_H0, once at the first cvv update_w; the transient
    // pseudopot is discarded).
    std::optional<sArray_t<nda::array_view<ComplexType, 4>>> _sH0_cvv;

    /**
     * scGW-tilde C4: (eps^{-1}_head - 1) on the PH-sym tau half grid from the
     * covariant-velocity SUBTRACTED head (cvv_head_t / cvv_detail::head_subtract):
     * scalar head Dyson eps(q^, inu) = 1 - 4 pi q^q^ : Phead(inu), wings dropped (as
     * the gygi treatment does), angular average of eps^{-1} over the three cartesian
     * directions (exact for cubic symmetry). Also logs the T-d meter v(q).P00 (PDF
     * G-c; warns as it approaches 1). Called by update_w under div_treatment = "cvv"
     * INSTEAD of the stored/gygi q -> 0 extrapolation; every downstream consumer
     * (Sigma_div_correction, vertex rungs, build_w0, cache_w) reads the SAME
     * mb_state.eps_inv_head array -- single-sourcing, none of them is edited.
     */
    nda::array<ComplexType, 1> eval_cvv_eps_inv_head(MBState &mb_state, THC_ERI auto &thc);

  public:
    std::string div_treatment() const { return _div_treatment; }
    // scGW-tilde (C0): cvv_head_t validates > 0 at construction (C4)
    void set_cvv_rspace_tol(double tol) { _cvv_rspace_tol = tol; }
    double cvv_rspace_tol() const { return _cvv_rspace_tol; }
    std::string& screen_type() { return _screen_type; };
    std::string screen_type() const { return _screen_type; };

    void set_vertex(vertex_t* vertex) { _vertex = vertex; }
    const vertex_t* vertex() const { return _vertex; }
    // true iff a vertex correction is attached and active (non-empty subspace C)
    // (defined in the .cpp: vertex_t is only forward-declared here).
    bool has_active_vertex() const;
    // true iff the scf driver must keep mb_state.dW_qtPQ alive across the iteration
    // boundary so Pi^C can use the previous iteration's screened W: an active vertex
    // on the GLOBAL auxiliary basis (or with the W-bar cache disabled). With the
    // SECONDARY basis the downfolded rung is cached at the update_w tail instead
    // (vertex_t::cache_w, notes/wbar_cache.md) and dW is freed unconditionally --
    // plain-GW memory profile. Defined in the .cpp.
    bool needs_dw_retention() const;
    // true iff an attached active vertex already has a SCREENED rung available for
    // Pi^C -- a retained dW (global path) or a folded W-bar cache (secondary path).
    // False only on the very first update_w of a run, which is what the bootstrap in
    // update_w keys on. Defined in the .cpp.
    bool vertex_has_rung(MBState const &mb_state) const;

    // TODO Remove everything below
    const nda::array<ComplexType, 1>& eps_inv_head() const {
      if (!_eps_inv_head.has_value()) {
        utils::check(false, "scr_coulomb_t: uninitialized _eps_inv_head.");
      }
      return _eps_inv_head.value();
    }

    // access to screened interaction
    const memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>& get() const {
      if (!_dW_qtPQ_opt.has_value()) {
        utils::check(false, "scr_coulomb_t: uninitialized _dW_qtPQ_opt.");
      }
      return _dW_qtPQ_opt.value();
    }
    memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator>& get_mutable() {
      if (!_dW_qtPQ_opt.has_value()) {
        utils::check(false, "scr_coulomb_t: uninitialized _dW_qtPQ_opt.");
      }
      return _dW_qtPQ_opt.value();
    }
    bool has_result() const noexcept { return _dW_qtPQ_opt.has_value(); }
    void reset() noexcept {
      if (_dW_qtPQ_opt.has_value()) _dW_qtPQ_opt.reset();
      if (_eps_inv_head.has_value()) _eps_inv_head.reset();
    }

  }; // scr_coulomb_t
} // solvers
}// methods

#endif //COQUI_SCR_COULOMB_T_H
