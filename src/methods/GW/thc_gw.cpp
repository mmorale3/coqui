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


#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/sparse/csr_blas.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"
#include "utilities/kpoint_utils.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW/gw_t.h"
#include "methods/GW/thc_gw.icc"

namespace methods {
  namespace solvers {
    template<MEMORY_SPACE MEM>
    void gw_t::evaluate(MBState &mb_state, THC_ERI auto const& thc, bool verbose) {
      if (verbose) {
        //http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20thc-gw
        app_log(1, "\n"
                   "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐ ┌─┐┬ ┬\n"
                   "║  ║ ║║═╬╗║ ║║   │ ├─┤│───│ ┬│││\n"
                   "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ ┴ ┴└─┘ └─┘└┴┘\n");
        //utils::check(scr_eri!=nullptr, "gw_t::evaluate: scr_eri is missing in thc_gw solver.");
        app_log(1, "  Screening type                = {}\n"
                   "  Number of bands               = {}\n"
                   "  Number of THC auxiliary basis = {}\n"
                   "  K-points                      = {} total, {} in the IBZ\n"
                   "  Divergent treatment at q->0   = {}\n",
                mb_state.screen_type,
                thc.MF()->nbnd(), thc.Np(), thc.MF()->nkpts(), thc.MF()->nkpts_ibz(),
                _div_treatment);
        _ft->metadata_log();
      }
      utils::check(mb_state.mpi == thc.mpi(),
                   "gw_t::evaluate: THC_ERI and MBState should have the same MPI context.");
      utils::check(mb_state.sG_tskij.has_value(),
                   "gw_t::evaluate: sG_tskij is not initialized in MBState.");
      utils::check(mb_state.sSigma_tskij.has_value(),
                     "gw_t::evaluate: sSigma_tskij is not initialized in MBState.");
      // Either copy of W will do: the device path keeps only dW_qtPQ_dev and
      // skips the host mirror (which nothing here reads), so requiring the host
      // one aborted every device run.
#if defined(ENABLE_DEVICE)
      utils::check(mb_state.dW_qtPQ.has_value() or mb_state.dW_qtPQ_dev.has_value(),
                   "gw_t::evaluate: neither dW_qtPQ nor dW_qtPQ_dev is initialized in MBState.");
#else
      utils::check(mb_state.dW_qtPQ.has_value(),
                   "gw_t::evaluate: dW_qtPQ is not initialized in MBState.");
#endif
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "thc-gw: We assume nt_f == nt_b at least for now. \n"
                   "        And we assume tau sampling for fermions and bosons are the same.");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "thc-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL",
                     "PI_PRIM_TO_AUX", "SIGMA_PRIM_TO_AUX", "SIGMA_AUX_TO_PRIM",
                     "EVALUATE_PI_K", "PI_ALLOC_K", "PI_HADPROD_K",
                     "EVALUATE_PI_R", "PI_ALLOC_R", "PI_FT_R", "PI_HADPROD_R",
                     "EVALUATE_W",
                     "EVALUATE_SIGMA_K", "SIGMA_ALLOC_K", "SIGMA_HADPROD_K", "SIGMA_MULTIPLY_DMAT_K",
                     "EVALUATE_SIGMA_R", "SIGMA_ALLOC_R", "SIGMA_FT_R", "SIGMA_HADPROD_R",
                     "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE",
                     "SIGMA_DIV_CORR"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
#if defined(ENABLE_DEVICE)
      if constexpr (MEM == DEVICE_MEMORY) {
        // Prefer the device-resident dW set by scr_coulomb_t::update_w<DEVICE>;
        // this skips a full host->device mirror of dW per SCF iter.
        if (mb_state.dW_qtPQ_dev.has_value()) {
          thc_gw_Xqindep<MEM>(mb_state.sG_tskij.value().local(),
                              mb_state.sSigma_tskij.value(), thc,
                              mb_state.dW_qtPQ_dev.value(),
                              mb_state.eps_inv_head.value());
        } else {
          thc_gw_Xqindep<MEM>(mb_state.sG_tskij.value().local(),
                              mb_state.sSigma_tskij.value(), thc,
                              mb_state.dW_qtPQ.value(),
                              mb_state.eps_inv_head.value());
        }
      } else
#endif
      {
        thc_gw_Xqindep<MEM>(mb_state.sG_tskij.value().local(),
                            mb_state.sSigma_tskij.value(), thc,
                            mb_state.dW_qtPQ.value(),
                            mb_state.eps_inv_head.value());
      }
      _Timer.stop("TOTAL");

      print_thc_gw_timers();
      thc.print_timers();
      mb_state.mpi->comm.barrier();
    }


    template<nda::MemoryArray Array_view_5D_t>
    void gw_t::evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        sArray_t<Array_view_5D_t> &sSigma_tskij,
                        THC_ERI auto const& thc, scr_coulomb_t* scr_eri, bool verbose) {
      if (verbose) {
        //http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20thc-gw
        app_log(1, "\n"
                   "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐ ┌─┐┬ ┬\n"
                   "║  ║ ║║═╬╗║ ║║   │ ├─┤│───│ ┬│││\n"
                   "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ ┴ ┴└─┘ └─┘└┴┘\n");
        utils::check(scr_eri!=nullptr, "gw_t::evaluate: scr_eri is missing in thc_gw solver.");
        app_log(1, "  polarizability = {}\n"
                   "  nbnd  = {}\n"
                   "  THC auxiliary basis  = {}\n"
                   "  nkpts = {}\n"
                   "  nkptz_ibz = {}\n"
                   "  divergent treatment at q->0 = {}\n",
                scr_eri->screen_type(),
                thc.MF()->nbnd(), thc.Np(), thc.MF()->nkpts(), thc.MF()->nkpts_ibz(),
                _div_treatment);
        _ft->metadata_log();
      }
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "thc-gw: We assume nt_f == nt_b at least for now. \n"
                   "        And we assume tau sampling for fermions and bosons are the same.");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "thc-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL",
                     "PI_PRIM_TO_AUX", "SIGMA_PRIM_TO_AUX", "SIGMA_AUX_TO_PRIM",
                     "EVALUATE_PI_K", "PI_ALLOC_K", "PI_HADPROD_K",
                     "EVALUATE_PI_R", "PI_ALLOC_R", "PI_FT_R", "PI_HADPROD_R",
                     "EVALUATE_W",
                     "EVALUATE_SIGMA_K", "SIGMA_ALLOC_K", "SIGMA_HADPROD_K", "SIGMA_MULTIPLY_DMAT_K",
                     "EVALUATE_SIGMA_R", "SIGMA_ALLOC_R", "SIGMA_FT_R", "SIGMA_HADPROD_R",
                     "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE",
                     "SIGMA_DIV_CORR"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
      sSigma_tskij.set_zero();
      if (thc.thc_X_type() == "q_dep") {
        APP_ABORT("gw_t::thc_gw_Xqdep: not implemented yet");
      } else if (thc.thc_X_type() == "q_indep") {
        thc_gw_Xqindep(G_tskij, sSigma_tskij, thc,  scr_eri->get_mutable(), scr_eri->eps_inv_head());
      } else {
        APP_ABORT("gw_t::evaluate: Invalid thc_X_type.\n");
      }
      _Timer.stop("TOTAL");

      print_thc_gw_timers();
      thc.print_timers();
      sSigma_tskij.communicator()->barrier();
    }

    template<MEMORY_SPACE MEM,
             nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
    void gw_t::eval_Sigma_all(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        memory::darray_t<Array_4D_t, communicator_t> &dW_qtPQ,
                        sArray_t<Array_5D_t> &sSigma_tskij,
                        THC_ERI auto &thc,
                        std::string alg) {
      sSigma_tskij.set_zero();
      if (alg == "R") {
        auto [qpools, tpools, np_P, np_Q] = dW_qtPQ.grid();
        app_log(2, "  Evaluation of GW self-energy:");
        app_log(2, "    - processor grid for G: (t, k, P, Q) = ({}, {}, {}, {})", tpools, qpools, np_P, np_Q);
        app_log(2, "    - processor grid for W: (t, q, P, Q) = ({}, {}, {}, {})\n", tpools, qpools, np_P, np_Q);

        // If the input dW is already on the target MEM (the caller passed
        // mb_state.dW_qtPQ_dev), use it directly. Otherwise — backwards-
        // compat path — mirror host dW to MEM once for the two FT
        // round-trips and discard on return.
        constexpr bool dW_already_on_target =
            (MEM == HOST_MEMORY && nda::mem::on_host<Array_4D_t>) ||
            (MEM == DEVICE_MEMORY && nda::mem::on_device<Array_4D_t>) ||
            (MEM == UNIFIED_MEMORY && nda::mem::on_unified<Array_4D_t>);
        if constexpr (dW_already_on_target) {
          eval_Sigma_all_Rspace<MEM, false, true>(G_tskij, dW_qtPQ, sSigma_tskij, thc, false);
          eval_Sigma_all_Rspace<MEM, true, false>(G_tskij, dW_qtPQ, sSigma_tskij, thc, true);
        } else {
          using local_Array_4D_dev = memory::array<MEM, ComplexType, 4>;
          using math::nda::make_distributed_array;
          auto pgrid_h = dW_qtPQ.grid();
          auto bsize_h = dW_qtPQ.block_size();
          auto gshape_h = dW_qtPQ.global_shape();
          auto dW_dev = make_distributed_array<local_Array_4D_dev>(
              *dW_qtPQ.communicator(), pgrid_h, gshape_h, bsize_h);
          dW_dev.local() = dW_qtPQ.local();
          eval_Sigma_all_Rspace<MEM, false, true>(G_tskij, dW_dev, sSigma_tskij, thc, false);
          eval_Sigma_all_Rspace<MEM, true, false>(G_tskij, dW_dev, sSigma_tskij, thc, true);
        }
      } else if (alg == "k") {
        auto [qpools, tpools, np_P, np_Q] = dW_qtPQ.grid();
        app_log(2, "  Evaluation of GW self-energy:");
        app_log(2, "    - processor grid for W: (t, q, P, Q) = ({}, {}, {}, {})\n", tpools, qpools, np_P, np_Q);

        if constexpr (MEM == HOST_MEMORY) {
          eval_Sigma_all_kspace<MEM>(G_tskij, dW_qtPQ, sSigma_tskij, thc, false);
          eval_Sigma_all_kspace<MEM>(G_tskij, dW_qtPQ, sSigma_tskij, thc, true);
          // collect terms from all processors
          sSigma_tskij.all_reduce();
        } else {
          utils::check(false, "gw_t::eval_Sigma_all: device path supports only "
                              "the R-space algorithm (alg=\"R\"). Set "
                              "kpts==qpts to dispatch the R-space variant.");
        }
      } else {
        utils::check(false, "Unkown algorithm for GW self-energy: {}. either \"R\" or \"k\"", alg);
      }
    }

    // instantiations
    using Arr4D = nda::array<ComplexType, 4>;
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;
    using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;

    template void gw_t::evaluate(const Arr &, sArray_t<Arrv> &, const thc_reader_t &, scr_coulomb_t*, bool);
    template void gw_t::evaluate(const Arrv &, sArray_t<Arrv> &, const thc_reader_t &, scr_coulomb_t*, bool);
    template void gw_t::evaluate(const Arrv2 &, sArray_t<Arrv> &, const thc_reader_t &, scr_coulomb_t*, bool);

    template void gw_t::evaluate<HOST_MEMORY>(MBState&, const thc_reader_t&, bool);
#if defined(ENABLE_DEVICE)
    template void gw_t::evaluate<DEVICE_MEMORY>(MBState&, const thc_reader_t&, bool);
#endif

    template void gw_t::eval_Sigma_all<HOST_MEMORY>(const Arr &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string);
    template void gw_t::eval_Sigma_all<HOST_MEMORY>(const Arrv &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string);
    template void gw_t::eval_Sigma_all<HOST_MEMORY>(const Arrv2 &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string);

  }
}
