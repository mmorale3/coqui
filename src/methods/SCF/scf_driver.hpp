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


#ifndef COQUI_SCF_DRIVER_HPP
#define COQUI_SCF_DRIVER_HPP

#include "mpi3/communicator.hpp"
#include "mpi3/core.hpp"
#include "utilities/Timer.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"

#include "utilities/mpi_context.h"
#include "methods/SCF/qp_params_t.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/mb_state/mb_state.hpp"

namespace methods {
/**
 * Generic self-consistent loop for full-frequency Green's function methods. 
 * The template parameters allow for different types of self-energy solvers (e.g. HF, GW, GF2) 
 * and different types of Coulomb Hamiltonian representations (e.g. THC, Cholesky).
 */
template<typename dyson_type, typename eri_t, typename corr_solver_t>
auto scf_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
              solvers::mb_solver_t<corr_solver_t> mb_solver, iter_scf::iter_scf_t *iter_solver = nullptr,
              int niter = 1, bool restart = false, double conv_tol = 1e-9, bool const_mu = false,
              std::string input_grp = "scf", int input_iter = -1)
              -> std::tuple<double, double>;

/**
 * Generic self-consistent loop for quasiparticle equations.
 * The template parameters allow for different types of self-energy solvers (e.g. HF, GW, GF2)
 * and different types of Coulomb Hamiltonian representations (e.g. THC, Cholesky).
 *
 * Project 2 increment Q5 (notes/q5_option2_outer_loop_spec.md §1): the last two arguments are
 * the Option-2 re-QP-ization knobs, following the scf_loop naming convention.
 * @param gf_grp  - checkpoint group of an EXTERNAL Green's function ("scf"/"embed"), which
 *                  ITERATION 1 consumes in place of the restart-H_eff's analytic QP G (its
 *                  density matrix feeds the HF stage and it feeds the Sigma^GW/W build;
 *                  iterations >= 2 revert to the loop's own QP G). EMPTY = INERT (default):
 *                  the loop is bit-identical to the pre-Q5 one.
 * @param gf_iter - iteration inside gf_grp; -1 = that group's "final_iter".
 */
template<typename eri_t, typename corr_solver_t>
double qp_scf_loop(MBState &mb_state, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                   qp_params_t &qp_params, solvers::mb_solver_t<corr_solver_t> mb_solver,
                   iter_scf::iter_scf_t *iter_solver = nullptr, int niter = 1,
                   bool restart = false, double conv_tol = 1e-8,
                   std::string gf_grp = "", long gf_iter = -1);

/**
 * RPA energy functional loop.
 */
template<typename eri_t, typename dyson_type>
double rpa_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                solvers::mb_solver_t<solvers::gw_t> mb_solver);
} // namespace methods

#endif // COQUI_SCF_DRIVER_HPP
