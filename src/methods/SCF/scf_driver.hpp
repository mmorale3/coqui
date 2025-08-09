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
#include "methods/SCF/qp_context.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/mb_state/mb_state.hpp"

namespace methods {
template<typename dyson_type, typename eri_t, typename corr_solver_t>
auto scf_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
              solvers::mb_solver_t<corr_solver_t> mb_solver, iter_scf::iter_scf_t *iter_solver = nullptr,
              int niter = 1, bool restart = false, double conv_tol = 1e-9, bool const_mu = false,
              std::string input_grp = "scf", int input_iter = -1)
              -> std::tuple<double, double>;
template<bool evscf_only, typename eri_t, typename corr_solver_t>
double qp_scf_loop(MBState &mb_state, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                   qp_context_t &qp_context, solvers::mb_solver_t<corr_solver_t> mb_solver,
                   iter_scf::iter_scf_t *iter_solver = nullptr, int niter = 1,
                   bool restart = false, double conv_tol = 1e-8);
template<typename eri_t, typename dyson_type>
double rpa_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                solvers::mb_solver_t<solvers::gw_t> mb_solver);
} // namespace methods

#endif // COQUI_SCF_DRIVER_HPP
