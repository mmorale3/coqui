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


#ifndef COQUI_PPROC_DRIVERS_HPP
#define COQUI_PPROC_DRIVERS_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_common.hpp"
#include "numerics/ac/ac_context.h"
#include "methods/pproc/pproc_t.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "hamiltonian/one_body_hamiltonian.hpp"

namespace methods {
  /**
   * Post-processing routines with arguments in property tree.
   * @param [INPUT] pp_type - type of post-processing, allowed options: ac, unfold
   * @param [INPUT] mf - mean-field instance
   * Requires:
   *   - prefix: prefix to the scf output h5 file.
   * Optional arguments:
   *   - outdir: location of directory with files
   * Optional arguments for pp_type = ac (analytical continuation):
   *   - ac_alg: algorithm for analytical continuation, allowed options: pade
   *   - dataset: input dataset
   *   - stats: statistics, allowed options: fermi, boson
   *   - w_min: minimum real frequency
   *   - w_max: maximum real frequency
   *   - Nw: number of real frequency points
   *   - eta: shift above the real-frequency axis
   * Optional arguments for pp_type = unfold_bz (unfold BZ from irreducible to the 1st BZ)
   */
  void post_processing(std::string pp_type, std::shared_ptr<mf::MF> mf, ptree const& pt) {
    auto mpi = mf->mpi();
    if (mpi->comm.size()%mpi->node_comm.size()!=0) {
      APP_ABORT("pproc: number of processors on each node should be the same.");
    }

    std::string err = std::string("pproc - Incorrect input - ");
    auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
    auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

    if (pp_type == "ac") {

      pproc_t pp(*mpi, prefix, outdir);
      auto dataset = io::get_value_with_default<std::string>(pt,"dataset","G_tskij");

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto stats   = io::get_value_with_default<std::string>(pt,"stats","fermi");
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-10.0);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",10.0);
      auto Nw      = io::get_value_with_default<int>(pt,"Nw",5000);
      auto eta     = io::get_value_with_default<double>(pt,"eta",0.01);
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", -1);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::string_to_stats_enum(stats), Nfit, eta, w_min, w_max, Nw);
      pp.analyt_cont(*mf, ac_context, dataset);

    } else if (pp_type == "unfold_bz") {

      std::string scf_output = outdir+"/"+prefix;
      unfold_bz(*mpi, *mf, scf_output);

    } else if (pp_type == "band_interpolation") {

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      std::string scf_output = outdir+"/"+prefix;

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(iteration==0 or iteration==-1,
                     "band_interpolation: iteration = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     iteration, scf_output);

        auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
        auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
        auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

        // dump mf data to "scf_output".mbpt.h5
        auto psp = hamilt::make_pseudopot(*mf);
        imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, true);
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }
      pp.wannier_interpolation(*mf, pt, wannier_file, "quasiparticle", "scf", iteration, trans_home_cell);

    } else if (pp_type == "spectral_interpolation") {

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta",0.001);
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", -1);
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-1.0);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",1.0);
      auto Nw      = io::get_value_with_default<int>(pt,"Nw",1000);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermi, Nfit, eta, w_min, w_max, Nw);

      pp.spectral_interpolation(*mf, pt, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "local_dos") {

      pproc_t pp(*mpi, prefix, outdir);
      // if wannier_file is not provided, calculate local DOS in the Bloch basis
      auto wannier_file = io::get_value_with_default<std::string>(pt, "wannier_file", "");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta",0.01);
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", -1);
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-1.0);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",1.0);
      auto Nw      = io::get_value_with_default<int>(pt,"Nw",1000);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermi, Nfit, eta, w_min, w_max, Nw);

      pp.local_density_of_state(*mf, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "dump_vxc") {

      std::string scf_output = outdir+"/"+prefix;
      hamilt::dump_vxc(*mpi, *mf, scf_output);

    } else if (pp_type == "dump_hartree") {

      auto scf_iter = io::get_value_with_default<int>(pt, "scf_iter", -1);

      std::string scf_output = outdir+"/"+prefix;
      auto psp = hamilt::make_pseudopot(*mf);

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(scf_iter==0 or scf_iter==-1,
                     "dump_hartree: scf_iter = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     scf_iter, scf_output);

        auto beta = io::get_value_with_default<double>(pt,"beta",1000.0);
        auto wmax = io::get_value_with_default<double>(pt,"wmax",12.0);
        auto iaft_prec = io::get_value_with_default<std::string>(pt, "iaft_prec", "high");

        // dump mf data to "scf_output".mbpt.h5
        imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source, iaft_prec, true);
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }

      hamilt::dump_hartree(*mpi, *mf, psp.get(), scf_output, scf_iter);

    } else {
      APP_ABORT("pproc: Unkonw post-processing type: {}", pp_type);
    }
  }

} // methods

#endif //COQUI_PPROC_DRIVERS_HPP
