/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_GW_THC_HPP
#define COQUI_REAL_AXIS_GW_THC_HPP

#include <complex>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"

namespace methods {
namespace real_axis {

/**
 * High-level driver: real-axis G0W0 in the THC auxiliary basis. Thin shim
 * that delegates to the two solver classes that own the math:
 *
 *   real_axis_scr_coulomb_t::update_w   -- A -> Pi -> W (state)
 *   methods::solvers::real_axis_gw_t::evaluate -- A, W -> Sigma_c (state)
 *
 * Kept for backwards compatibility with existing test fixtures that drive
 * the full pipeline through one entry point. New code should call the
 * solver classes directly via the SCF loop.
 *
 * Reads `state.A_wskij`. Writes `state.{Im,Re}{Pi,W}_qPQO` and
 * `state.{Im,Re}Sigma_wskij`.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY,
         methods::THC_ERI THC_t>
void evaluate_thc_serial(real_axis_mb_state_t & state,
                         THC_t const& thc,
                         double eps_nufft = 1e-10,
                         std::string div_treatment = "ignore_g0",
                         bool verbose = false,
                         bool use_rspace = false)
{
  // Both real_axis_scr_coulomb_t and real_axis_gw_t classes run on host
  // (state arrays are sArrays / dArrays<HOST_MEMORY>); MEM is a template
  // marker for callers that template their pipeline uniformly.
  utils::check(state.grid != nullptr,
               "evaluate_thc_serial: state.grid not bound");
  utils::check(state.mpi != nullptr,
               "evaluate_thc_serial: state.mpi not bound");

  auto const& grid = *state.grid;

  real_axis_scr_coulomb_t scr_eri(&grid, "rpa", div_treatment, eps_nufft);
  scr_eri.update_w(state, thc, verbose, use_rspace);

  methods::solvers::real_axis_gw_t gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                                      eps_nufft, /*ntrans*/ 1);
  gw.evaluate(state, thc, eps_nufft, div_treatment, verbose, use_rspace);
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_GW_THC_HPP
