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
                     "SIGMA_DIV_CORR", "SIGMA_DIV_T", "SIGMA_DIV_ALLOC", "SIGMA_DIV_GEMM",
                     "SIGMA_DIV_REDUCE", "SIGMA_DIV_ADD"} ) {
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

      // ISDF-Vertex / LFF-Sigma consumers: HOST-only (the vertex routines are not ported to the
      // device yet, notes/vertex_perf_plan.md "gpu port"). On the device path they must be inactive.
      if constexpr (MEM == HOST_MEMORY) {
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
      // LFF-Sigma Route 2 (L-6): the pair-resolved static-ladder vertex self-energy (scr_coulomb_t::build_sigma_pair,
      // the C-window block in band labels, replicated) accumulated on top of Sigma^GW; released here.
      _sigma_pair_dmax = 0.0; _sigma_pair_smax = 0.0;
      if (mb_state.dSigma_pair_tskab.has_value()) {
        auto &sSigma = mb_state.sSigma_tskij.value();
        auto const &dS = mb_state.dSigma_pair_tskab.value();
        const long b0 = mb_state.sigma_pair_window[0], nb = mb_state.sigma_pair_window[1];
        auto S_loc = sSigma.local();
        utils::check(S_loc.shape(0) == dS.shape(0) and S_loc.shape(1) == dS.shape(1) and S_loc.shape(2) == dS.shape(2) and
                     dS.shape(3) == nb and dS.shape(4) == nb and b0 + nb <= S_loc.shape(3),
                     "gw_t::evaluate: dSigma_pair_tskab ({}, {}, {}, {}, {}) does not fit Sigma ({}, {}, {}, {}, {}) at window [{}, {}).",
                     dS.shape(0), dS.shape(1), dS.shape(2), dS.shape(3), dS.shape(4), S_loc.shape(0), S_loc.shape(1),
                     S_loc.shape(2), S_loc.shape(3), S_loc.shape(4), b0, b0 + nb);
        double dmax = 0.0, smax = 0.0;
        {
          const int node_rank = sSigma.node_comm()->rank(), node_size = sSigma.node_comm()->size();
          const long nts = S_loc.shape(0), ns = S_loc.shape(1), nk = S_loc.shape(2);
          sSigma.win().fence();
          for (long it = node_rank; it < nts; it += node_size)
            for (long is = 0; is < ns; ++is)
              for (long ik = 0; ik < nk; ++ik)
                for (long i = 0; i < nb; ++i)
                  for (long j = 0; j < nb; ++j) {
                    auto &sv = S_loc(it, is, ik, b0 + i, b0 + j);
                    const auto dv = dS(it, is, ik, i, j);
                    dmax = std::max(dmax, std::abs(dv)); smax = std::max(smax, std::abs(sv));
                    sv += dv;
                  }
          sSigma.win().fence();
        }
        dmax = mb_state.mpi->comm.all_reduce_value(dmax, boost::mpi3::max<>{});
        smax = mb_state.mpi->comm.all_reduce_value(smax, boost::mpi3::max<>{});
        _sigma_pair_dmax = dmax; _sigma_pair_smax = smax;
        app_log(1, "  [LFF-Sigma pair] vertex self-energy ADDED on the C block [{}, {}): max |dSigma(tau)| = {:.4e} vs max |Sigma^GW(tau)| = {:.4e} (ratio {:.3e})",
                b0, b0 + nb, dmax, smax, dmax / std::max(smax, 1e-300));
        mb_state.dSigma_pair_tskab.reset();
      }
      } else {
        utils::check(_vertex == nullptr or not _vertex->active(),
                     "gw_t::evaluate<DEVICE_MEMORY>: the ISDF-Vertex Sigma^C cut is host-only.");
        utils::check(not mb_state.dWsig_qtPQ.has_value() and not mb_state.dWsig_inf_qPQ.has_value() and
                     not mb_state.dSigma_pair_tskab.has_value(),
                     "gw_t::evaluate<DEVICE_MEMORY>: the LFF-Sigma corrections are host-only.");
        _sigma_lff_dmax = _sigma_lff_smax = _sigma_lff_dfmax = 0.0;
        _sigma_pair_dmax = _sigma_pair_smax = 0.0;
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
                     "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE",
                     "SIGMA_DIV_CORR", "SIGMA_DIV_T", "SIGMA_DIV_ALLOC", "SIGMA_DIV_GEMM",
                     "SIGMA_DIV_REDUCE", "SIGMA_DIV_ADD"} ) {
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
