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
     * @param ft          - [INPUT] imaginary-axis Fourier transform (IAFT) grids
     * @param vertex_type - [INPUT] type of the vertex correction.
     *                      {choices: "none", "2nd_exchange"}
     * @param band_window - [INPUT] contiguous orbital range [first, last) defining
     *                      the subspace C. An empty range means C = empty set.
     * @param nbnd        - [INPUT] number of bands in the primary basis
     *                      (used to validate band_window)
     */
    vertex_t(const imag_axes_ft::IAFT *ft,
             std::string vertex_type,
             nda::range band_window,
             long nbnd);

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

  private:
    const imag_axes_ft::IAFT* _ft = nullptr;

    // type of the vertex correction: "none" or "2nd_exchange"
    std::string _vertex_type = "none";

    // contiguous orbital range [first, last) defining the subspace C
    nda::range _band_window = nda::range(0, 0);

  public:
    std::string vertex_type() const { return _vertex_type; }
    nda::range band_window() const { return _band_window; }

    // vertex requested in the input
    bool enabled() const { return _vertex_type != "none"; }
    // vertex requested AND C is non-empty; C = empty set must be an exact no-op
    bool active() const { return enabled() and _band_window.size() > 0; }

  }; // vertex_t

} // solvers
} // methods

#endif //COQUI_VERTEX_T_H
