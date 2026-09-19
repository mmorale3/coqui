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
#include "methods/vertex/vertex_t.h"
#include "methods/GW/gw_t.h"
#include "methods/GW/thc_gw.icc"
#include "methods/HF/thc_exchange_kernel.hpp"

namespace methods {
  namespace solvers {
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
      utils::check(mb_state.dW_qtPQ.has_value(),
                   "gw_t::evaluate: dW_qtPQ is not initialized in MBState.");
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
                     "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
      thc_gw_Xqindep(mb_state.sG_tskij.value().local(), mb_state.sSigma_tskij.value(), thc,
                     mb_state.dW_qtPQ.value(), mb_state.eps_inv_head.value());

      // ISDF-Vertex: second-order-exchange self-energy cut Sigma^C, accumulated
      // into sSigma_tskij on top of the GW self-energy. When no active vertex is
      // attached this is a strict no-op -- no allocation, no arithmetic -- so the
      // disabled path is bit-identical to plain scGW.
      if (_vertex != nullptr and _vertex->active())
        _vertex->eval_Sigma_C(mb_state, thc);

      // LFF-Sigma (Route 1, notes/lff_aux_plan.md 2026-09-19): the local-field-factor vertex in Sigma. The SAME GW
      // contraction, driven with the vertex correction of W that scr_coulomb_t::build_sigma_lff published
      // (dW~ = scale x Herm[W (Gamma_eff - 1)], body + its extrapolated head), accumulated on top of Sigma^GW (+ Sigma^C).
      // Computed into its own buffer so Sigma^GW's contraction is untouched; the correction is released here.
      if (mb_state.dWsig_qtPQ.has_value()) {
        utils::check(mb_state.eps_inv_head_sig.has_value(), "gw_t::evaluate: dWsig_qtPQ without eps_inv_head_sig.");
        auto &sSigma = mb_state.sSigma_tskij.value();
        auto sDS = math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(*mb_state.mpi, sSigma.shape());
        sDS.set_zero();
        thc_gw_Xqindep(mb_state.sG_tskij.value().local(), sDS, thc, mb_state.dWsig_qtPQ.value(), mb_state.eps_inv_head_sig.value());
        double dmax = 0.0, smax = 0.0;
        {
          auto S_loc = sSigma.local();
          auto D_loc = sDS.local();
          const int node_rank = sSigma.node_comm()->rank(), node_size = sSigma.node_comm()->size();
          const long nts = S_loc.shape(0);
          const long chunk = long(S_loc.size() / std::max(nts, 1l));
          sSigma.win().fence();
          for (long it = node_rank; it < nts; it += node_size) {
            auto s_t = S_loc(it, nda::ellipsis{});
            auto d_t = D_loc(it, nda::ellipsis{});
            for (long n = 0; n < chunk; ++n) {
              const double ad = std::abs(d_t.data()[n]), as = std::abs(s_t.data()[n]);
              dmax = std::max(dmax, ad); smax = std::max(smax, as);
              s_t.data()[n] += d_t.data()[n];
            }
          }
          sSigma.win().fence();
        }
        dmax = mb_state.mpi->comm.all_reduce_value(dmax, boost::mpi3::max<>{});
        smax = mb_state.mpi->comm.all_reduce_value(smax, boost::mpi3::max<>{});
        _sigma_lff_dmax = dmax; _sigma_lff_smax = smax;
        app_log(1, "  [LFF-Sigma] vertex self-energy ADDED: max |dSigma(tau)| = {:.4e} vs max |Sigma^GW(tau)| = {:.4e} (ratio {:.3e}); "
                   "eps_inv_head_sig(tau_0) = {:.4e}", dmax, smax, dmax / std::max(smax, 1e-300), mb_state.eps_inv_head_sig.value()(0).real());
        mb_state.dWsig_qtPQ.reset();
        mb_state.eps_inv_head_sig.reset();
      }
      // LFF-Sigma: the INSTANTANEOUS part of the vertex correction (the vertex's nu -> inf limit times the bare Coulomb)
      // is a static exchange-like self-energy: the THC exchange contraction with that kernel, added to F (the Dyson reads
      // F + Sigma(tau) after this call; e_hf then carries it). Body only: the q -> 0 head of the static piece is NOT
      // included (the exchange head correction is madelung x the vertex's tail, a rigid shift of the window; flagged).
      _sigma_lff_dfmax = 0.0;
      if (mb_state.dWsig_inf_qPQ.has_value()) {
        utils::check(mb_state.sDm_skij.has_value() and mb_state.sF_skij.has_value(), "gw_t::evaluate: the static LFF-Sigma piece needs Dm and F.");
        auto &sF = mb_state.sF_skij.value();
        auto sDF = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mb_state.mpi, sF.shape());
        lff_sigma_detail::exchange_with_kernel(mb_state.sDm_skij.value().local(), mb_state.dWsig_inf_qPQ.value(), sDF, thc);
        double dfmax = 0.0, fmax = 0.0;
        {
          auto F_loc = sF.local();
          auto D_loc = sDF.local();
          const int node_rank = sF.node_comm()->rank(), node_size = sF.node_comm()->size();
          const long nsk = F_loc.shape(0) * F_loc.shape(1);
          const long chunk = long(F_loc.size() / std::max(nsk, 1l));
          sF.win().fence();
          for (long isk = node_rank; isk < nsk; isk += node_size) {
            auto f = F_loc(isk / F_loc.shape(1), isk % F_loc.shape(1), nda::ellipsis{});
            auto d = D_loc(isk / F_loc.shape(1), isk % F_loc.shape(1), nda::ellipsis{});
            for (long n = 0; n < chunk; ++n) {
              dfmax = std::max(dfmax, std::abs(d.data()[n])); fmax = std::max(fmax, std::abs(f.data()[n]));
              f.data()[n] += d.data()[n];
            }
          }
          sF.win().fence();
        }
        dfmax = mb_state.mpi->comm.all_reduce_value(dfmax, boost::mpi3::max<>{});
        fmax = mb_state.mpi->comm.all_reduce_value(fmax, boost::mpi3::max<>{});
        _sigma_lff_dfmax = dfmax;
        app_log(1, "  [LFF-Sigma] instantaneous part ADDED to the static self-energy: max |dF| = {:.4e} vs max |F| = {:.4e} (ratio {:.3e})",
                dfmax, fmax, dfmax / std::max(fmax, 1e-300));
        mb_state.dWsig_inf_qPQ.reset();
      }
      _Timer.stop("TOTAL");

      print_thc_gw_timers();
      thc.print_timers();
      // ISDF-Vertex breakdown. Printed here so it sits next to the GW/THC tables it must
      // be compared against. Timers ACCUMULATE across scf iterations (never reset), so
      // "elapsed" is the running total and "avg" is the per-iteration cost -- which is the
      // number to watch, since the vertex cost per iteration is what sets the wall time.
      // Only eval_Sigma_C is driven from here; eval_Pi_C / cache_w / build_w0 are called
      // from the scr_coulomb update_w seam, and their slots on the SAME vertex object are
      // filled by the time this prints on the next iteration.
      if (_vertex != nullptr and _vertex->active())
        _vertex->print_vertex_timers();
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
                     "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE"} ) {
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

    template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
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

        eval_Sigma_all_Rspace<false, true>(G_tskij, dW_qtPQ, sSigma_tskij, thc, false);
        eval_Sigma_all_Rspace<true, false>(G_tskij, dW_qtPQ, sSigma_tskij, thc, true);
      } else if (alg == "k") {
        auto [qpools, tpools, np_P, np_Q] = dW_qtPQ.grid();
        app_log(2, "  Evaluation of GW self-energy:");
        app_log(2, "    - processor grid for W: (t, q, P, Q) = ({}, {}, {}, {})\n", tpools, qpools, np_P, np_Q);

        eval_Sigma_all_kspace(G_tskij, dW_qtPQ, sSigma_tskij, thc, false);
        eval_Sigma_all_kspace(G_tskij, dW_qtPQ, sSigma_tskij, thc, true);
        // collect terms from all processors
        sSigma_tskij.all_reduce();
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

    template void gw_t::evaluate(MBState&, const thc_reader_t&, bool);

    template void gw_t::eval_Sigma_all(const Arr &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string); 
    template void gw_t::eval_Sigma_all(const Arrv &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string); 
    template void gw_t::eval_Sigma_all(const Arrv2 &, memory::darray_t<Arr4D, mpi3::communicator> &, sArray_t<Arrv> &,
          thc_reader_t&, std::string); 

  }
}
