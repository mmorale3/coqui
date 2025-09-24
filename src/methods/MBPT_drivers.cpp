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



#include <string>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "cxxopts.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "hamiltonian/pseudo/pseudopot.h"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/SCF/dca_dyson.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/embed_t.h"
#include "methods/embedding/embed_eri_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

#include "SCF/scf_driver.hpp"
#include "MBPT_drivers.h"

namespace mpi3 = boost::mpi3;
namespace methods
{
/**
 * Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 * Optional arguments (with default values):
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grids (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for Coulomb kernel in HF. {choices: "ignore_g0", "gygi"}
 *  - niter: "1" Number of iterations in the self-consistent loop.
 *  - conv_thr: "1e-9" Convergence threshold for the self-consistent loop.
 *  - const_mu: "false" Fix the chemical potential during the self-consistent loop.
 *  - output: "bdft.mbpt" Prefix of the output h5 file.
 *  - restart: "false" Restart from a previous bdft.scf calculation.
 *  - t_prescreen_thresh: "0.0" Threshold for prescreening in time (GF2 only for now)
 */
template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto output = io::get_value_with_default<std::string>(pt,"output","bdft.mbpt");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto input_grp = io::get_value_with_default<std::string>(pt,"input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  imag_axes_ft::IAFT ft = (!restart)?
      imag_axes_ft::IAFT(beta, wmax, imag_axes_ft::ir_source, iaft_prec) :
      imag_axes_ft::read_iaft(output+".mbpt.h5");

  using namespace solvers;
  hf_t hf(string_to_div_enum(hf_div_treatment));
  if(solver_type == "rpa") {
    simple_dyson dyson(mf.get(), &ft);
    //solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
    gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    rpa_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf,&gw));
  } else if(solver_type == "hf") {

    simple_dyson dyson(mf.get(), &ft);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    MBState mb_state(mpi, ft, output);
    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf),
             std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
             input_grp, input_iter);

  } else if(solver_type == "gw") {
    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson(mf.get(), &ft);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    if (screen_type.substr(0,8)=="gw_edmft") {

      auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      MBState mb_state(ft, output, mf, wannier_file, trans_home_cell);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
               input_grp, input_iter);

    } else {

      MBState mb_state(mpi, ft, output);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
               input_grp, input_iter);

    }

  } else if(solver_type == "gf2") {

    auto gf2_direct_type = io::get_value_with_default<std::string>(pt,"gf2_direct_type","gf2");
    auto gf2_exchange_alg = io::get_value_with_default<std::string>(pt,"gf2_exchange_alg","orb");
    auto gf2_exchange_type = io::get_value_with_default<std::string>(pt,"gf2_exchange_type","gf2");
    auto gf2_save_C = io::get_value_with_default<bool>(pt,"gf2_save_C",true);
    auto gf2_sosex_save_memory = io::get_value_with_default<bool>(pt,"gf2_sosex_save_memory",true);
    auto t_prescreen_thresh = io::get_value_with_default<double>(pt,"t_prescreen_thresh",0.0);

    simple_dyson dyson(mf.get(), &ft);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    solvers::gf2_t gf2(mf.get(), &ft, string_to_div_enum(div_treatment),
                       gf2_direct_type, gf2_exchange_alg, gf2_exchange_type, output,
                       gf2_save_C, gf2_sosex_save_memory);
    gf2.t_thresh() = t_prescreen_thresh;

    MBState mb_state(mpi, ft, output);

    if (gf2_direct_type == "gf2") {
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2),
               std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
               input_grp, input_iter);
    } else {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2, &scr_eri),
               std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
               input_grp, input_iter);
    }

  } else if(solver_type == "gw_dca") {

    utils::check(false, "mbpt: gw_dca is not implemented!");
    /*ptree dca_pt = io::find_child(pt, "gw_mean_field");
    std::string mf_type = (mf.mf_type()==mf::qe_source)?
        "qe" : (mf.mf_type()==mf::pyscf_source)? "pyscf" : "bdft";
    mf::MF dca_mf(mf::make_MF(mpi, dca_pt, mf_type));
    dca_dyson dyson(mpi, &mf, &ft, dca_mf);
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    scf_loop(dyson, eri, ft, mb_solver_t(&hf,&gw), nullptr,
             output, niter, restart, conv_thr, const_mu);*/

  } else if (solver_type == "qphf") {

    auto iter_solver = iter_scf::make_iter_scf(pt);
    MBState mb_state(mpi, ft, output);
    qp_context_t qp_context;
    qp_scf_loop<false>(mb_state, eri, ft, qp_context, mb_solver_t(&hf), &iter_solver,
                       niter, restart, conv_thr);

  } else if (solver_type == "evgw0") {

    auto qp_type = io::get_value_with_default<std::string>(pt,"qp_type","sc");
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",0.0001);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    io::tolower(ac_alg);
    io::tolower(qp_type);
    qp_context_t qp_context(qp_type, ac_alg, Nfit, eta, conv_thr);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop<true>(mb_state, eri, ft, qp_context, mb_solver_t(&hf,&gw,&scr_eri), &iter_solver,
                      niter, restart, conv_thr);

  } else if (solver_type == "qpgw") {

    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",0.0001);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","fermi");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_context_t qp_context("sc", ac_alg, Nfit, eta, 1e-8, off_diag_mode);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop<false>(mb_state, eri, ft, qp_context, mb_solver_t(&hf,&gw,&scr_eri), &iter_solver,
                       niter, restart, conv_thr);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}


template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt,
          nda::array<ComplexType, 5> const& C_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto output = io::get_value_with_default<std::string>(pt,"output","bdft.mbpt");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto input_grp = io::get_value_with_default<std::string>(pt,"input_type", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");
  imag_axes_ft::IAFT ft = (!restart)?
                          imag_axes_ft::IAFT(beta, wmax, imag_axes_ft::ir_source, iaft_prec) :
                          imag_axes_ft::read_iaft(output+".mbpt.h5");

  using namespace solvers;
  hf_t hf(string_to_div_enum(hf_div_treatment));
  if (solver_type == "gw") {

    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    simple_dyson dyson(mf.get(), &ft);
    auto iter_solver = iter_scf::make_iter_scf(pt);
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, string_to_div_enum(div_treatment));
    solvers::gw_t gw(&ft, string_to_div_enum(div_treatment), output);
    MBState mb_state(ft, output, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false);
    if (local_polarizabilities) {
      mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
      local_polarizabilities.reset();
    }

    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
             std::addressof(iter_solver), niter, restart, conv_thr, const_mu,
             input_grp, input_iter);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}


void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    nda::array<ComplexType, 5> const& C_ksIai,
                    nda::array<long, 3> const& band_window,
                    nda::array<RealType, 2> const& kpts_crys,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials) {
  std::string err = std::string("downfolding_1e - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto qp_selfenergy = io::get_value_with_default<bool>(pt,"qp_selfenergy",false);
  auto update_dc = io::get_value_with_default<bool>(pt,"update_dc",true);
  auto dc_type = io::get_value<std::string>(pt, "dc_type",
                                            err+"dc_type. Valid types are \"hartree\", \"hf\", \"gw\", \"gw_dynamic_u\" and \"gw_mix_u\".");
  io::tolower(dc_type);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);

  auto imp_sigma_mixing = io::get_value_with_default<double>(pt,"imp_sigma_mixing",1.0);
  auto dc_sigma_mixing = io::get_value_with_default<double>(pt,"dc_sigma_mixing",1.0);
  std::array<double, 2> sigma_mixing{dc_sigma_mixing, imp_sigma_mixing};

  embed_t embed(*mf, C_ksIai, band_window, kpts_crys, trans_home_cell);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));

  MBState mb_state(ft, outdir+"/"+prefix, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false);
  if (local_selfenergies and local_hf_potentials) {
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    local_selfenergies.reset();
    local_selfenergies.reset();
  }

  if (qp_selfenergy) {

    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",1e-6);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",30);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_context_t qp_context("sc", ac_alg, Nfit, eta, 1e-8, off_diag_mode);
    embed.downfolding(mb_state, qp_selfenergy, update_dc, dc_type, force_real, &qp_context);

  } else {

    embed.downfolding(mb_state, qp_selfenergy, update_dc, dc_type, force_real, nullptr, "default", sigma_mixing);
  }
}

void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials) {
  std::string err = std::string("downfolding_1e - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto qp_selfenergy = io::get_value_with_default<bool>(pt,"qp_selfenergy",false);
  auto update_dc = io::get_value_with_default<bool>(pt,"update_dc",true);
  auto dc_type = io::get_value<std::string>(pt, "dc_type",
      err+"dc_type. Valid types are \"hartree\", \"hf\", \"gw\", \"gw_dynamic_u\" and \"gw_mix_u\".");
  io::tolower(dc_type);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);

  auto imp_sigma_mixing = io::get_value_with_default<double>(pt,"imp_sigma_mixing",1.0);
  auto dc_sigma_mixing = io::get_value_with_default<double>(pt,"dc_sigma_mixing",1.0);
  std::array<double, 2> sigma_mixing{dc_sigma_mixing, imp_sigma_mixing};

  embed_t embed(*mf, wannier_file, trans_home_cell);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));

  MBState mb_state(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false);
  if (local_selfenergies and local_hf_potentials) {
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    local_selfenergies.reset();
    local_selfenergies.reset();
  }

  if (qp_selfenergy) {
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta",1e-6);
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",30);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_context_t qp_context("sc", ac_alg, Nfit, eta, 1e-8, off_diag_mode);
    embed.downfolding(mb_state, qp_selfenergy, update_dc, dc_type, force_real, &qp_context);
  } else {
    embed.downfolding(mb_state, qp_selfenergy, update_dc, dc_type, force_real,
                      nullptr, "default", sigma_mixing);
  }
}

auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state,
                        ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc_impl - Incorrect input - ");
  auto g_grp = io::get_value<std::string>(pt, "input_type", err+"input_type");
  auto g_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);
  embed_t embed(*mf);
  return embed.downfold_gloc(mb_state, force_real, g_grp, g_iter);
}

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                  nda::array<ComplexType, 5> const& C_ksIai,
                  nda::array<long, 3> const& band_window,
                  nda::array<RealType, 2> const& kpts_crys)
  -> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false), pt);
}

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false), pt);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc_impl(eri_t &eri, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_wloc_impl - Incorrect input - ");
  auto g_grp = io::get_value<std::string>(pt, "input_type");
  io::tolower(g_grp);
  g_grp = (g_grp == "mf")? "scf" : g_grp;
  auto g_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto screen_type = io::get_value<std::string>(
      pt, "screen_type", err+"screen_type. This parameter determines the type of screened interactions for the downfolded Hamiltonian. "
                             "Valid types are \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
                             "\"gw_edmft\", \"gw_edmft_rpa\", and \"gw_edmft_density\"");
  io::tolower(screen_type);

  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", true);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);

  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto bare_div_treatment = io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(bare_div_treatment);

  auto output_in_tau = io::get_value_with_default<bool>(pt, "output_in_tau", true);

  auto mf = eri.MF();

  // set local polarizabilities if provided
  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }
  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(bare_div_treatment), "default");
  return (output_in_tau)?
    embed_eri.downfold_wloc<true>(eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, g_grp, g_iter) :
    embed_eri.downfold_wloc<false>(eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, g_grp, g_iter);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_wloc - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto input_type = io::get_value<std::string>(pt, "input_type",
      err+"input_type. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\".");
  io::tolower(input_type);

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  if (input_type == "mf") {
    if (std::filesystem::exists(output+".mbpt.h5")) {
      app_log(1, "");
      app_log(1, "╔══════════════════════════════════════════════════════════╗");
      app_log(1, "║ [ WARNING ]                                              ║");
      app_log(1, "║ Input type is set to \"mf\", while a CoQuí checkpoint      ║");
      app_log(1, "║ HDF5 with the same prefix has been detected. CoQuí will  ║");
      app_log(1, "║ overwrite the old checkpoint. Considering moving the old ║");
      app_log(1, "║ HDF5 or changing CoQuí prefix next time.                 ║");
      app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
    }
    auto beta = io::get_value_with_default<double>(pt,"beta", 1000.0);
    auto wmax = io::get_value_with_default<double>(pt,"wmax", 12.0);
    auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, false);
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);
  } else if (input_type == "scf" or input_type == "embed") {
    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "downfold_2e: input_type == \"{}\" while the coqui h5, {}.mbpt.h5, does not exist!", input_type, output);
  } else
    utils::check(false, "downfold_2e: invalid input_type = {}. Valid options are \"mf\" and \"coqui\".", input_type);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_wloc_impl(
      eri, MBState(ft, output, mf, wannier_file, trans_home_cell, false),
      pt, local_polarizabilities);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              nda::array<ComplexType, 5> const& C_ksIai,
              nda::array<long, 3> const& band_window,
              nda::array<RealType, 2> const& kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_wloc - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto input_type = io::get_value<std::string>(pt,"input_type",
      err+"input_type. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\". ");
  io::tolower(input_type);

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  if (input_type == "mf") {
    if (std::filesystem::exists(output+".mbpt.h5")) {
      app_log(1, "");
      app_log(1, "╔══════════════════════════════════════════════════════════╗");
      app_log(1, "║ [ WARNING ]                                              ║");
      app_log(1, "║ Input type is set to \"mf\", while a CoQuí checkpoint      ║");
      app_log(1, "║ HDF5 with the same prefix has been detected. CoQuí will  ║");
      app_log(1, "║ overwrite the old checkpoint. Considering moving the old ║");
      app_log(1, "║ HDF5 or changing CoQuí prefix next time.                 ║");
      app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
    }
    auto beta = io::get_value_with_default<double>(pt,"beta", 1000.0);
    auto wmax = io::get_value_with_default<double>(pt,"wmax", 12.0);
    auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, false);
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);
  } else if (input_type == "scf" or input_type == "embed") {
    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "downfold_2e: input_type == \"{}\" while the coqui h5, {}.mbpt.h5, does not exist!", input_type, output);
  } else
    utils::check(false, "downfold_2e: invalid input_type = {}. Valid options are \"mf\" and \"coqui\".", input_type);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_wloc_impl(
      eri, MBState(ft, output, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false),
      pt, local_polarizabilities);
}

/**
 * Downfolds the two-electron Hamiltonian with arguments in property tree.
 * Required arguments:
 *  - prefix: Prefix of the output and input files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 *  - screen_type: Screening types for the partially screened interaction u(iw). {choices: "bare", "crpa", "edmft"}
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the source and output files are.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - bare_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grid (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<bool return_vw, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfolding_2e - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto screen_type = io::get_value<std::string>(pt, "screen_type",
         err+"screen_type. This parameter determines the type of screened interactions for the downfolded Hamiltonian. "
             "Valid types are \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
             "\"gw_edmft\", \"gw_edmft_rpa\", and \"gw_edmft_density\"");
  io::tolower(screen_type);
  auto q_dependent = io::get_value_with_default<bool>(pt,"q_dependent", false);

  auto dc_pi_mixing = io::get_value_with_default<double>(pt,"dc_pi_mixing",1.0);

  auto input_type = io::get_value<std::string>(pt,"input_type",
         err+"input_type. This parameter defines the source of input Green's function. Valid types are \"mf\" and \"coqui\". ");
  io::tolower(input_type);
  // coqui has default logic to determine the input G for a given input_type.
  // these two parameters are used only if the users decided not to follow the default logic.
  auto g_grp = io::get_value_with_default<std::string>(pt,"input_grp", "");
  auto g_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", true);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);

  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto bare_div_treatment = io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(bare_div_treatment);

  // only use when outdir/prefix.mbpt.h5 does not exist.
  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  auto mf = eri.MF();

  std::string output = outdir + "/" + prefix;
  if (input_type == "mf") {
    utils::check(not std::filesystem::exists(output+".mbpt.h5"), "downfold_2e: input_type = \"mf\" can not be launched if {}.mbpt.h5 already exists. "
                          "Please set input_type = \"coqui\" and input_iter = 0 if you want to use MF Green's function as the input.", output);
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, true);
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);
  } else if (input_type == "coqui") {
    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "downfold_2e: input_type == \"coqui\" while the coqui h5, {}.mbpt.h5, does not exist!", output);
  } else
    utils::check(false, "downfold_2e: invalid input_type = {}. Valid options are \"mf\" and \"coqui\".", input_type);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  // create MBstate object to store the state of the downfolding
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);
  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }

  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(bare_div_treatment), "default");
  if (screen_type.substr(0,8)=="gw_edmft") {
    embed_eri.downfolding_edmft(eri, mb_state, screen_type, permut_symm, force_real,
                                &ft, g_grp, g_iter, dc_pi_mixing);
  } else {
    embed_eri.downfolding_crpa(eri, mb_state, screen_type, "none", permut_symm, force_real,
                               &ft, g_grp, g_iter, q_dependent);
  }

  if constexpr (return_vw) {
    auto& mpi = eri.mpi();
    nda::array<ComplexType, 4> V_abcd;
    nda::array<ComplexType, 5> W_wabcd;
    long nw_half; long nImpOrbs;
    if (mpi->node_comm.root()) {

      h5::file file(output+".mbpt.h5", 'r');
      auto grp = h5::group(file).open_group("downfold_2e");
      long iter;
      h5::h5_read(grp, "final_iter", iter);
      auto iter_grp = grp.open_group("iter" + std::to_string(iter));
      nda::h5_read(iter_grp, "Vloc_abcd", V_abcd);
      nda::h5_read(iter_grp, "Wloc_wabcd", W_wabcd);

      nw_half = W_wabcd.shape(0);
      nImpOrbs = W_wabcd.shape(1);
      mpi->node_comm.broadcast_n(&nw_half, 1, 0);
      mpi->node_comm.broadcast_n(&nImpOrbs, 1, 0);
      mpi->node_comm.broadcast_n(V_abcd.data(), V_abcd.size(), 0);
      mpi->node_comm.broadcast_n(W_wabcd.data(), W_wabcd.size(), 0);

    } else {

      mpi->node_comm.broadcast_n(&nw_half, 1, 0);
      mpi->node_comm.broadcast_n(&nImpOrbs, 1, 0);
      V_abcd = nda::array<ComplexType, 4>({nw_half, nImpOrbs, nImpOrbs, nImpOrbs});
      W_wabcd = nda::array<ComplexType, 5>({nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
      mpi->node_comm.broadcast_n(V_abcd.data(), V_abcd.size(), 0);
      mpi->node_comm.broadcast_n(W_wabcd.data(), W_wabcd.size(), 0);

    }
    return std::make_tuple(std::move(V_abcd), std::move(W_wabcd));
  }
}

template<bool return_vw, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               nda::array<ComplexType, 5> const& C_ksIai,
               nda::array<long, 3> const& band_window,
               nda::array<RealType, 2> const& kpts_crys,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfolding_2e - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto screen_type = io::get_value<std::string>(pt, "screen_type",
                                                err+"screen_type. This parameter determines the type of screened interactions for the downfolded Hamiltonian. "
                                                    "Valid types are \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
                                                    "\"gw_edmft\", \"gw_edmft_rpa\", and \"gw_edmft_density\"");
  io::tolower(screen_type);
  auto q_dependent = io::get_value_with_default<bool>(pt,"q_dependent", false);

  auto dc_pi_mixing = io::get_value_with_default<double>(pt,"dc_pi_mixing",1.0);

  auto input_type = io::get_value<std::string>(pt,"input_type",
                                               err+"input_type. This parameter defines the source of input Green's function. Valid types are \"mf\" and \"coqui\". ");
  io::tolower(input_type);
  // coqui has default logic to determine the input G for a given input_type.
  // these two parameters are used only if the users decided not to follow the default logic.
  auto g_grp = io::get_value_with_default<std::string>(pt,"input_grp", "");
  auto g_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", true);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);

  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto bare_div_treatment = io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(bare_div_treatment);

  // only use when outdir/prefix.mbpt.h5 does not exist.
  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  auto mf = eri.MF();

  std::string output = outdir + "/" + prefix;
  if (input_type == "mf") {
    utils::check(not std::filesystem::exists(output+".mbpt.h5"), "downfold_2e: input_type = \"mf\" can not be launched if {}.mbpt.h5 already exists. "
                                                                 "Please set input_type = \"coqui\" and input_iter = 0 if you want to use MF Green's function as the input.", output);
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, true);
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);
  } else if (input_type == "coqui") {
    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "downfold_2e: input_type == \"coqui\" while the coqui h5, {}.mbpt.h5, does not exist!", output);
  } else
    utils::check(false, "downfold_2e: invalid input_type = {}. Valid options are \"mf\" and \"coqui\".", input_type);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  // create MBstate object to store the state of the downfolding
  MBState mb_state(ft, output, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false);
  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }

  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(bare_div_treatment), "default");
  if (screen_type.substr(0,8)=="gw_edmft") {
    embed_eri.downfolding_edmft(eri, mb_state, screen_type, permut_symm, force_real,
                                &ft, g_grp, g_iter, dc_pi_mixing);
  } else {
    embed_eri.downfolding_crpa(eri, mb_state, screen_type, "none", permut_symm, force_real,
                               &ft, g_grp, g_iter, q_dependent);
  }

  if constexpr (return_vw) {
    auto& mpi = eri.mpi();
    nda::array<ComplexType, 4> V_abcd;
    nda::array<ComplexType, 5> W_wabcd;
    long nw_half; long nImpOrbs;
    if (mpi->node_comm.root()) {

      h5::file file(output+".mbpt.h5", 'r');
      auto grp = h5::group(file).open_group("downfold_2e");
      long iter;
      h5::h5_read(grp, "final_iter", iter);
      auto iter_grp = grp.open_group("iter" + std::to_string(iter));
      nda::h5_read(iter_grp, "Vloc_abcd", V_abcd);
      nda::h5_read(iter_grp, "Wloc_wabcd", W_wabcd);

      nw_half = W_wabcd.shape(0);
      nImpOrbs = W_wabcd.shape(1);
      mpi->node_comm.broadcast_n(&nw_half, 1, 0);
      mpi->node_comm.broadcast_n(&nImpOrbs, 1, 0);
      mpi->node_comm.broadcast_n(V_abcd.data(), V_abcd.size(), 0);
      mpi->node_comm.broadcast_n(W_wabcd.data(), W_wabcd.size(), 0);

    } else {

      mpi->node_comm.broadcast_n(&nw_half, 1, 0);
      mpi->node_comm.broadcast_n(&nImpOrbs, 1, 0);
      V_abcd = nda::array<ComplexType, 4>({nw_half, nImpOrbs, nImpOrbs, nImpOrbs});
      W_wabcd = nda::array<ComplexType, 5>({nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
      mpi->node_comm.broadcast_n(V_abcd.data(), V_abcd.size(), 0);
      mpi->node_comm.broadcast_n(W_wabcd.data(), W_wabcd.size(), 0);

    }
    return std::make_tuple(std::move(V_abcd), std::move(W_wabcd));
  }
}

/**
 * Generates a downfolded Hamiltonian at the Hartree-Fock (HF) level. Bare 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * HF frozen core contributions are added to the bare 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=bare and dc_type=hf in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules, e.g. can be used in the 
 * mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.mbpt.h5 and prefix.model.h5 files will be placed.
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only
 applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real.
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grids (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt) {
  std::string err = std::string("hf_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
  auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
  auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

  // two-body downfolding options
  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", false);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", false);
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  auto thresh = io::get_value_with_default<double>(pt, "thresh", 1e-6);
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model outputs
  std::string output = outdir + "/" + prefix;

  // initialize
  imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, true);
  hamilt::pseudopot psp(*mf);
  write_mf_data(*mf, ft, psp, output);
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, ignore_g0,
                        string_to_div_enum(hf_div_treatment), "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, "bare", factorization_type, permut_symm, force_real,
                             &ft, "", -1, false, thresh);

  // One-body Hamiltonian
  embed_t embed(*mf, wannier_file, trans_home_cell);
  embed.hf_downfolding(outdir, prefix, eri, ft, force_real, string_to_div_enum(hf_div_treatment));

}

/**
 * Generates a downfolded Hamiltonian at the GW level. cRPA Screened 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * A quasi-particle approximation to the GW self-energy is applied to generate a downfolded
 * 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=crpa and dc_type=gw in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules,
 * e.g. model Hamiltonian type mean-field chkpt file and ERI-compatible h5 chkpt file,
 * which can be used in the mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.model.h5 files will be placed.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real. 
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 *  Parameters used by quasiparticle algorithm:
 *  - ac_alg: Algorithm for analytic continuation, default:pade {choices: pade}
 *  - eta: Smearing parameter: default:1e-6
 *  - Nfit: Number of terms in AC fit, default: 30
 *  - off_diag_mode: Off diagonal treatment, default: qp_energy. {choices: fermi, qp_energy} 
 */
template<typename eri_t>
void gw_downfold(eri_t &eri, ptree const& pt) {
  std::string err = std::string("gw_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  // coqui has default logic to determine the input G for a given input_type.
  // these two parameters are used only if the users decided not to follow the default logic.
  auto g_grp = io::get_value_with_default<std::string>(pt,"input_grp", "");
  auto g_iter = io::get_value_with_default<long>(pt, "input_iter", -1);

  // two-body downfolding options
  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", false);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", false);
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  auto thresh = io::get_value_with_default<double>(pt, "thresh", 1e-6);
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  //  QP/AC parameters
  auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
  auto eta     = io::get_value_with_default<double>(pt,"eta",1e-6);
  auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",30);
  auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
  io::tolower(ac_alg);
  io::tolower(off_diag_mode);
  utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
               "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model output
  std::string output = outdir + "/" + prefix;

  utils::check(std::filesystem::exists(output+".mbpt.h5"),
               "gw_downfolding: {}.mbpt.h5, does not exist!", output);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  // create MBstate object to store the state of the downfolding
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, string_to_div_enum(div_treatment),
                        string_to_div_enum(hf_div_treatment), "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, "crpa", factorization_type, permut_symm,
                             force_real, &ft, g_grp, g_iter, false, thresh);

  // one body hamiltonian
  qp_context_t qp_context("sc", ac_alg, Nfit, eta, 1e-8, off_diag_mode);
  embed_t embed(*mf, wannier_file, trans_home_cell);
  embed.downfolding(mb_state, true, true, "gw", force_real, &qp_context, "model_static");
}

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& C_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto qp_approx_mbpt = io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false);
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto corr_only = io::get_value_with_default<bool>(pt,"corr_only",false);

  auto iter_solver = iter_scf::make_iter_scf(pt, 1.0);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf, C_ksIai, band_window, kpts_crys, trans_home_cell, false);
  if (local_hf_potentials and local_selfenergies) {
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    local_hf_potentials.reset();
    local_selfenergies.reset();
  }
  embed_t embed(*mf);
  embed.dmft_embed(mb_state, &iter_solver, qp_approx_mbpt, corr_only);
}

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto qp_approx_mbpt = io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false);
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto corr_only = io::get_value_with_default<bool>(pt,"corr_only",false);

  auto iter_solver = iter_scf::make_iter_scf(pt, 1.0);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false);
  embed_t embed(*mf);
  embed.dmft_embed(mb_state, &iter_solver, qp_approx_mbpt, corr_only);
}

// instantiations
using mpi3::communicator;

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc_impl(thc_reader_t &, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(thc_reader_t &, ptree const&,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(thc_reader_t &, ptree const&,
              nda::array<ComplexType, 5> const&,
              nda::array<long, 3> const&,
              nda::array<RealType, 2> const&,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void downfolding_2e<false>(
    thc_reader_t&, ptree const&,
    nda::array<ComplexType, 5> const&, nda::array<long, 3> const&, nda::array<RealType, 2> const&,
    std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
downfolding_2e<true>(
    thc_reader_t&, ptree const&,
    nda::array<ComplexType, 5> const&, nda::array<long, 3> const&, nda::array<RealType, 2> const&,
    std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void downfolding_2e<false>(
    thc_reader_t&, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
downfolding_2e<true>(
    thc_reader_t&, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void hf_downfold(thc_reader_t&, ptree const&);
template void gw_downfold(thc_reader_t&, ptree const&);

#define MBPT_INST(HF, HARTREE, EXCHANGE, CORR) \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
     ptree const&);                             \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
     ptree const&,                             \
     nda::array<ComplexType, 5> const&,  \
     nda::array<long, 3> const&,   \
     nda::array<RealType, 2> const&,           \
     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

// All combinations of thc/chol for 4 eri slots
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef MBPT_INST

}
