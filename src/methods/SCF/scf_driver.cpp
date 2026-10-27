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



#include "methods/vertex/vertex_debug.hpp"
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <map>
#include <format>

#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/device_pool.h"
#include "utilities/h5_background_writer.hpp"
#include "utilities/freemem.h"

#include "methods/ERI/mb_eri_context.h"
#include "methods/tools/chkpt_utils.h"
#include "methods/SCF/qp_modea.hpp"   // qp_modea::last_run(), read-only
#include "methods/SCF/sigma_causal_projection.hpp"
#include "methods/SCF/sigma_causal_stage.hpp"
#include "simple_dyson.h"
#include "dca_dyson.h"
#include "scf_driver.hpp"

namespace methods {

template<MEMORY_SPACE MEM,
         typename dyson_type, typename eri_t, typename corr_solver_t>
auto scf_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
              solvers::mb_solver_t<corr_solver_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
              int niter, bool restart, double conv_tol, bool const_mu,
              std::string input_grp, int input_iter, bool eval_thermodynamics)
              -> std::tuple<double, double> {
  utils::TimerManager Timer;
  auto mpi = mb_eri.corr_eri->get().mpi();
  auto mf = mb_eri.corr_eri->get().MF();
  utils::check(mpi == mb_state.mpi,
               "SCF loop: mpi context of mb_state and mb_eri should be the same!");
  utils::check(&FT == mb_state.ft,
               "SCF loop: imag_axes_ft of mb_state and scf_loop should be the same!");
  // The grand potential takes the dynamical Luttinger-Ward functional of the correlated solver
  // (the RPA ring sum for gw_t). A vertex in P or Sigma makes the self-energy depart from that
  // functional, so Omega would not belong to the computed solution.
  utils::check(not eval_thermodynamics or mb_solver.scr_eri == nullptr or
               mb_solver.scr_eri->vertex() == nullptr,
               "scf_loop: eval_thermodynamics = true is not supported with a vertex correction "
               "(the Luttinger-Ward functional of the vertex-corrected self-energy is not "
               "implemented). Set eval_thermodynamics = false or remove the vertex input.");
  // HERMITIZE and ENERGY exist so the children sum to SCF_TOTAL: without them
  // part of the loop falls in the gap between the four phase timers.
  for( auto& v: {"SCF_TOTAL", "DYSON", "MBPT_SOLVERS", "ITERATIVE", "WRITE",
                 "HERMITIZE", "ENERGY", "CAUSAL_PROJ"} ) {
    Timer.add(v);
  }
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20dyson-scf
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐┌─┐┌┐┌   ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║   ││└┬┘└─┐│ ││││───└─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  ─┴┘ ┴ └─┘└─┘┘└┘   └─┘└─┘└  \n");
  app_log(1, "  Maximum iteration number = {}", niter);
  app_log(1, "  Convergence tolerance    = {}", conv_tol);
  app_log(1, "  Checkpoint HDF5          = {}", mb_state.coqui_prefix+".mbpt.h5");
  app_log(1, "  Restart                  = {}", (restart)? "yes" : "no");
  if (restart) {
    app_log(1, "    - H5 group             = {}", input_grp);
    app_log(1, "    - Iteration            = {}", input_iter);
  }
  app_log(1, "  Number of processors     = {} cores per node, {} nodes\n",
          mpi->node_comm.size(), mpi->internode_comm.size());
  FT.metadata_log();

  Timer.start("SCF_TOTAL");
  // Initialize MBState
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  auto& sF_skij = mb_state.sF_skij.value();
  auto& sDm_skij = mb_state.sDm_skij.value();
  auto& sG_tskij = mb_state.sG_tskij.value();
  auto& sSigma_tskij = mb_state.sSigma_tskij.value();
  double mu = 0.0;
  if (!restart) {
    hamilt::set_fock(*mf, dyson.PSP(), sF_skij, true);
  } else {
    input_iter = chkpt::read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu,
                                 mb_state.coqui_prefix, input_grp, input_iter);
  }

  Timer.start("DYSON");
  // init Green's function. By default, we update mu as well.
  update_G(dyson, *mf, FT, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, false);
  Timer.stop("DYSON");


  Timer.start("WRITE");
  if (!restart) { // write metadata and the MF solution
    chkpt::write_metadata(mpi->comm, *mf, FT, dyson.sH0_skij(), dyson.sS_skij(), mb_state.coqui_prefix);
    // force_sync: read_input_iterations quiesces this write a few lines below, with no compute
    // in between, so the async path would copy the whole checkpoint and overlap none of it.
    chkpt::dump_scf(mpi->comm, 0, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu,
                    mb_state.coqui_prefix, "scf", -1, true);
  }
  Timer.stop("WRITE");

  double F_conv = std::numeric_limits<double>::infinity(), Sigma_conv = F_conv;
  std::vector<double> energies(4, 0.0);
  std::vector<double> energies_diff(3, 0.0);
  auto converged = [&]() {
    bool energy_converged = std::all_of(energies_diff.begin(), energies_diff.end(),
                                        [conv_tol](double x) { return std::abs(x) < conv_tol; });
    if (iter_solver!=nullptr) {
      return (energy_converged and std::abs(F_conv) <= std::abs(conv_tol) and std::abs(Sigma_conv) <= std::abs(conv_tol));
    } else {
      return energy_converged;
    }
  };
  // Determine output iteration
  // 1) The output h5 group is always "scf"
  // 2) The output iteration is scf/final_iter + 1
  // Every rank reads the file here, but the iteration-0 checkpoint above may
  // still be in flight on the writing rank, so the join has to be collective:
  // with only a local join the other ranks could read before the write lands and
  // fail with 'h5 group "scf" does not exist'.
  utils::h5_quiesce_collective(mpi->comm);
  std::tie(mb_state.mbpt_iter, mb_state.df_1e_iter, mb_state.df_2e_iter, mb_state.embed_iter) =
      chkpt::read_input_iterations(mb_state.coqui_prefix+".mbpt.h5");
  long output_iter_init = mb_state.mbpt_iter+1;
  long output_iter = output_iter_init;

  // Reserve the shared device pool for the duration of the SCF loop, so the
  // scratch that recurs every iteration (the redistribute staging buffers
  // above all) is served without a cudaMalloc/cudaFree per use. It is taken
  // here rather than at startup because ERI/THC construction sizes its own
  // blocking from the free device memory it observes and allocates through the
  // raw allocator; by now that is done. Inert unless COQUI_DEVICE_POOL_GB is
  // set, and released when the loop ends.
  std::optional<utils::device_pool_guard> pool_guard;
  if constexpr (MEM != HOST_MEMORY) {
    if (std::size_t pool_bytes = utils::device_pool_size_from_env(); pool_bytes > 0)
      pool_guard.emplace(pool_bytes, "scf");
  }

  // Snapshot of the previous iteration's F and Sigma, so simple mixing does not
  // read them back from the checkpoint it just wrote: that serial HDF5 read can
  // dominate the ITERATIVE phase for large Sigma_tskij. One shm copy per iteration
  // replaces it. Skipped when node memory is tight, in which case mixing reads
  // the previous iterate from the checkpoint. DIIS is unaffected: it needs a
  // history, not just the previous iterate, and keeps reading the checkpoint.
  std::optional<sArray_t<Array_view_4D_t>> sF_prev;
  std::optional<sArray_t<Array_view_5D_t>> sSigma_prev;
  // COQUI_NO_PREV_SNAPSHOT=1 disables the snapshot (the previous iterate is read
  // back from the checkpoint), so the two paths can be compared directly.
  const bool snapshot_disabled = [] {
    const char* v = std::getenv("COQUI_NO_PREV_SNAPSHOT");
    return v != nullptr and std::strtol(v, nullptr, 10) != 0;
  }();
  if (iter_solver != nullptr and iter_solver->iter_alg() == iter_scf::damping
      and not snapshot_disabled) {
    double need_mb = double(sSigma_tskij.local().size() + sF_skij.local().size())
                     * sizeof(ComplexType) / (1024.0*1024.0);
    if (double(utils::freemem()) > 3.0*need_mb) {
      sF_prev.emplace(math::shm::make_shared_array<Array_view_4D_t>(*mpi, sF_skij.shape()));
      sSigma_prev.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, sSigma_tskij.shape()));
      app_log(2, "  keeping the previous F/Sigma in memory for mixing ({:.1f} GB/node); "
                 "the checkpoint read is skipped.", need_mb/1024.0);
    } else {
      app_log(2, "  not keeping the previous F/Sigma in memory for mixing: would need "
                 "{:.1f} GB/node with {:.1f} GB free.", need_mb/1024.0, utils::freemem()/1024.0);
    }
  }

  // CAUSAL PROJECTION of the fresh Sigma_c (sigma_causal_stage.hpp); one instance for the loop, which keeps the
  // projector and the per-(s, k) warm starts across iterations.
  sigma_causal_stage causal_stage;
  causality_meter causality;

  // pol_vertex_sigma_dyn_final_full (the in-run nu-sampled dynamic Sigma vertex): the loop ends on an all-node Sigma pass. The
  // iteration niter allows last asks for one; when the convergence test passes earlier on a sampled pass, ONE more iteration
  // runs with an all-node pass and the loop stops after it whatever its own convergence numbers.
  const bool sigma_final_full = mb_solver.scr_eri != nullptr and mb_solver.scr_eri->sigma_dyn_final_full();
  bool final_after_conv = false;   // this iteration is the all-node pass after a convergence on a sampled pass
  bool stop = false;

  // start SCF iteration
  do {
    app_log(1, "\n** Iteration # {} **", output_iter);
    if (sigma_final_full and (final_after_conv or output_iter == output_iter_init + niter - 1)) {
      mb_solver.scr_eri->request_sigma_dyn_full_pass();
      app_log(1, "  [scf] pol_vertex_sigma_dyn_final_full: the dynamic Sigma vertex of this iteration solves every node ({}).",
              final_after_conv ? "the convergence test passed on the previous, sampled pass; the loop stops after this iteration"
                               : "the last iteration niter allows");
    }
    // Take the snapshot before the solvers overwrite F and Sigma in place.
    if (sSigma_prev.has_value()) {
      sF_prev.value().win().fence();
      sSigma_prev.value().win().fence();
      if (mpi->node_comm.root()) {
        sF_prev.value().local()     = sF_skij.local();
        sSigma_prev.value().local() = sSigma_tskij.local();
      }
      sF_prev.value().win().fence();
      sSigma_prev.value().win().fence();
    }
    Timer.start("MBPT_SOLVERS");
    // HF
    if (mb_solver.hf != nullptr) {
      if (mb_eri.hf_eri) {
        mb_solver.hf->evaluate(sF_skij, sDm_skij.local(),
                               mb_eri.hf_eri->get(), dyson.sS_skij().local(), true, true);
      } else if (mb_eri.hartree_eri and mb_eri.exchange_eri) {
        mb_solver.hf->evaluate(sF_skij, sDm_skij.local(),
                               mb_eri.hartree_eri->get(), dyson.sS_skij().local(), true, false);
        // create temporary buffer for K since hf_solver.evaluate(F) performs in-place evaluation for F.
        sArray_t<Array_view_4D_t> sK_skij(
            math::shm::make_shared_array<Array_view_4D_t>(*mpi, sF_skij.shape()));
        mb_solver.hf->evaluate(sK_skij, sDm_skij.local(),
                               mb_eri.exchange_eri->get(), dyson.sS_skij().local(), false, true);
        if (mpi->node_comm.root()) {
          sF_skij.local() += sK_skij.local();
        }
      } else {
        mb_solver.hf->evaluate(sF_skij, sDm_skij.local(), mb_eri.corr_eri->get(),
                               dyson.sS_skij().local(), true, true);
      }
      mpi->comm.barrier();
    }
    // correlated solver for dynamic self-energy, e.g. gw, gf2
    if (mb_solver.corr != nullptr) {

      if (mb_solver.scr_eri != nullptr)
        mb_solver.scr_eri->template update_w<MEM>(
            mb_state, mb_eri.corr_eri->get(), output_iter);

      mb_solver.corr->iter() = output_iter;
      // gw_t::evaluate is MEM-templated; gf2_t::evaluate is not (its
      // explicit instantiations below are HOST-only). Use a constexpr
      // branch on the corr solver type so HOST-only solvers still build.
      if constexpr (std::is_same_v<corr_solver_t, solvers::gw_t>) {
        mb_solver.corr->template evaluate<MEM>(mb_state, mb_eri.corr_eri->get());
      } else {
        static_assert(MEM == HOST_MEMORY,
                      "scf_loop: only gw_t supports DEVICE_MEMORY");
        mb_solver.corr->evaluate(mb_state, mb_eri.corr_eri->get());
      }
      // deallocate mb_state.dW_qtPQ after this since it's only used in the corr solver and can be very large for GW.
      // Exception (vertex): with an active dynamic-rung vertex on the GLOBAL auxiliary basis, keep
      // W alive across the iteration boundary so eval_Pi_qdep (which runs BEFORE this
      // iteration's update_w) can evaluate Pi^C with the previous iteration's screened rung
      // (one-iteration lag; converges to the same self-consistent fixed point). With the
      // SECONDARY basis the vertex caches the DOWNFOLDED rung
      // Wbar = t W t^dag at update_w time instead (vertex_t::cache_w),
      // so dW is freed here -- the plain-GW memory profile (see needs_dw_retention).
      if (mb_solver.scr_eri == nullptr or not mb_solver.scr_eri->needs_dw_retention())
        mb_state.dW_qtPQ.reset();
      mpi->comm.barrier();
    }

    Timer.start("HERMITIZE");
    if (mpi->node_comm.root()) {
      hermitize_in_tau(sF_skij.local(), "Fock matrix");
      hermitize_in_tau(sSigma_tskij.local(), "dynamic self-energy");
    }
    mpi->comm.barrier();
    Timer.stop("HERMITIZE");
    Timer.stop("MBPT_SOLVERS");

    // Remove the non-causal part of the new Sigma_c BEFORE mixing (sigma_causal_stage.hpp), so Dyson's G is causal and
    // the damped mix of two causal Sigma_c stays causal.
    if (sigma_causal_stage::enabled(mb_solver)) {
      Timer.start("CAUSAL_PROJ");
      causal_stage.apply<MEM>(sSigma_tskij, mpi, FT, output_iter, "scf_loop");
      Timer.stop("CAUSAL_PROJ");
    }


    Timer.start("ITERATIVE");
    if (iter_solver != nullptr) {
      // Hand the in-memory previous iterate to mixing when we kept one; it
      // falls back to the checkpoint read when we did not.
      std::tie(F_conv, Sigma_conv) = solve_iterative(*mpi, *iter_solver, output_iter,
                                                     mb_state.coqui_prefix,
                                                     sF_skij, sSigma_tskij, &FT,
                                                     {"scf", "F_skij", "Sigma_tskij"},
                                                     sF_prev.has_value() ? &sF_prev.value() : nullptr,
                                                     sSigma_prev.has_value() ? &sSigma_prev.value() : nullptr);
    }
    Timer.stop("ITERATIVE");

    Timer.start("DYSON");
    // whether to update mu depends on const_mu
    update_G(dyson, *mf, FT, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, const_mu);
    Timer.start("HERMITIZE");
    if (mpi->node_comm.root()) {
      hermitize_in_tau(sDm_skij.local(), "density matrix");
      hermitize_in_tau(sG_tskij.local(), "Green's function");
    }
    mpi->comm.barrier();
    Timer.stop("HERMITIZE");
    Timer.stop("DYSON");
    // CAUSALITY METER and guard on G and Sigma_c (causality_meter, sigma_causal_stage.hpp); the abort itself waits until
    // this iteration's checkpoint is written (below)
    const std::string causality_abort_msg =
        causality.measure(sG_tskij, sSigma_tskij, mpi, FT, output_iter, "scf_loop");


    Timer.start("ENERGY");
    auto k_weight = mf->k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
    double e_corr = (mb_solver.corr != nullptr)? eval_corr_energy(mpi->comm, FT, sG_tskij, sSigma_tskij, k_weight) : 0.0;
    Timer.stop("ENERGY");
    energies_diff = {e_1e - energies[0], e_hf - energies[1], e_corr - energies[2]};
    energies = {e_1e, e_hf, e_corr, e_1e+e_hf+e_corr};

    // print energies and scf convergence
    app_log(1, "\nEnergy contributions");
    app_log(1, "--------------------");
    app_log(1, "  non-interacting (H0):           {} a.u.", e_1e);
    app_log(1, "  Hartree-Fock:                   {} a.u.", e_hf);
    app_log(1, "  correlation:                    {} a.u.", e_corr);
    app_log(1, "  total energy:                   {} a.u.", e_1e+e_hf+e_corr);
    app_log(1, " ");
    app_log(1, "energy difference");
    app_log(1, "  - non-interacting (H0):         {} a.u.", energies_diff[0]);
    app_log(1, "  - Hartree-Fock:                 {} a.u.", energies_diff[1]);
    app_log(1, "  - correlation:                  {} a.u.", energies_diff[2]);
    if (iter_solver != nullptr) {
      app_log(1, "abs max diff of Fock matrix:   {}", F_conv);
      if (mb_solver.corr != nullptr)
        app_log(1, "abs max diff of self-energy:   {}\n", Sigma_conv);
    }
    Timer.start("WRITE");
    chkpt::dump_scf(mpi->comm, output_iter, sDm_skij, sG_tskij, sF_skij,
                    sSigma_tskij, mu, mb_state.coqui_prefix,
                    input_grp, input_iter);
    Timer.stop("WRITE");
    utils::check(causality_abort_msg.empty(), "{}", causality_abort_msg);
    output_iter++;
    if (final_after_conv) {
      stop = true;
    } else {
      stop = (output_iter >= output_iter_init+niter) or converged();
      if (stop and sigma_final_full and output_iter < output_iter_init+niter and not mb_solver.scr_eri->sigma_dyn_last_pass_full()) {
        app_log(1, "  [scf] converged on a nu-SAMPLED dynamic Sigma pass: one more iteration with an all-node pass "
                   "(pol_vertex_sigma_dyn_final_full); the loop stops after it.");
        final_after_conv = true;
        stop = false;
      }
    }
  } while (not stop);
  // The last checkpoint may still be in flight. Everything downstream --
  // post-processing, embedding, the next driver -- reads this file, and with a
  // non-threadsafe HDF5 none of it may run concurrently with the writer.
  Timer.start("WRITE");
  utils::h5_quiesce_collective(mpi->comm);
  Timer.stop("WRITE");
  pool_guard.reset();
  Timer.stop("SCF_TOTAL");

  app_log(2, "\n  Dyson-SCF timers");
  app_log(2, "  ----------------");
  app_log(2, "    Total:                {0:.3f} sec", Timer.elapsed("SCF_TOTAL"));
  app_log(2, "    Dyson:                {0:.3f} sec", Timer.elapsed("DYSON"));
  app_log(2, "    MBPT solvers:         {0:.3f} sec", Timer.elapsed("MBPT_SOLVERS"));
  app_log(2, "    Iterative alg:        {0:.3f} sec", Timer.elapsed("ITERATIVE"));
  app_log(2, "    Write:                {0:.3f} sec", Timer.elapsed("WRITE"));
  app_log(2, "    Energies:             {0:.3f} sec", Timer.elapsed("ENERGY"));
  // Nested inside DYSON / MBPT_SOLVERS, so not part of the residual below.
  app_log(2, "    (of which hermitize:  {0:.3f} sec)", Timer.elapsed("HERMITIZE"));
  app_log(2, "    (unaccounted):        {0:.3f} sec\n",
          Timer.elapsed("SCF_TOTAL") - Timer.elapsed("DYSON") - Timer.elapsed("MBPT_SOLVERS")
          - Timer.elapsed("ITERATIVE") - Timer.elapsed("WRITE") - Timer.elapsed("ENERGY"));

  if (eval_thermodynamics) {
    for (auto &v: {"THERMODYNAMICS", "PHI_DYNAMICAL", "OTHERS"}) {
      Timer.add(v);
    }
    Timer.start("THERMODYNAMICS");

    // Obtain the dynamical part of the Luttinger-Ward functional Phi. 
    // std::nullopt is returned when the phi functional is not implemented for 
    // a given solver. In such case the grand potential is incomplete and the
    // evaluation below is skipped. 
    // A null corr solver (HF-only) has no dynamic self-energy, so Phi_dynamical = 0.
    Timer.start("PHI_DYNAMICAL");
    std::optional<double> Phi_dynamical = 0.0;
    if (mb_solver.corr != nullptr) {
      // gw_t evaluates its polarization in MEM, as its self-energy; gf2_t is host-only
      if constexpr (std::is_same_v<corr_solver_t, solvers::gw_t>)
        Phi_dynamical = mb_solver.corr->template eval_phi_dynamical<MEM>(sG_tskij.local(),
                                                                         mb_eri.corr_eri->get());
      else
        Phi_dynamical = mb_solver.corr->eval_phi_dynamical(sG_tskij.local(), mb_eri.corr_eri->get());
    }
    Timer.stop("PHI_DYNAMICAL");

    // Evaluate the thermodynamic properties only when Phi_dynamical is available.
    if (Phi_dynamical) {
      Timer.start("OTHERS");
      mb_state.thermodynamics = eval_thermodynamic_properties(mpi->comm, dyson, sF_skij, sSigma_tskij,
                                                              energies, *Phi_dynamical, mu, false);
      Timer.stop("OTHERS");
      Timer.stop("THERMODYNAMICS");

      app_log(2, "\n  Thermodynamic property evaluation timers");
      app_log(2, "  ----------------");
      app_log(2, "    Total:                 {0:.3f} sec", Timer.elapsed("THERMODYNAMICS"));
      app_log(2, "    Dynamical part of Phi: {0:.3f} sec", Timer.elapsed("PHI_DYNAMICAL"));
      app_log(2, "    Other properties:      {0:.3f} sec", Timer.elapsed("OTHERS"));
      app_log(2, "\n");
    } else {
      Timer.stop("THERMODYNAMICS");
      app_log(1, "\n[WARNING] scf_loop: thermodynamic-property (grand potential) evaluation "
                 "is not yet supported for this correlated solver; skipping it. "
                 "Set eval_thermodynamics = false to silence this message.\n");
    }
    
  }

  app_log(1, "####### SCF routines end #######\n");
  return std::make_tuple(energies[0]+energies[1], energies[2]);
}


template<MEMORY_SPACE MEM, typename eri_t, typename corr_solver_t>
double qp_scf_loop(
  MBState &mb_state, 
  eri_t &mb_eri, 
  const imag_axes_ft::IAFT& FT,
  qp_context_t& qp_params, 
  solvers::mb_solver_t<corr_solver_t> mb_solver,
  iter_scf::iter_scf_t *iter_solver,
  int niter,
  bool restart,
  double conv_tol,
  std::string gf_grp,
  long gf_iter) {

  using math::shm::make_shared_array;
  utils::TimerManager Timer;
  auto mpi = mb_eri.corr_eri->get().mpi();
  auto mf = mb_eri.corr_eri->get().MF();
  for( auto& v: {"SCF_TOTAL", "CANONICALIZATION", "MBPT_SOLVERS", "ITERATIVE", "WRITE"} ) {
    Timer.add(v);
  }
  // Two independent axes: qp_approx says how Sigma(iw) is reduced to a static QP Hamiltonian,
  // qp_scf_mode says what the loop updates with it.
  utils::check(qp_params.qp_approx == "qp_eqn" or qp_params.qp_approx == "lqp",
               "scf_driver.cpp::qp_scf_loop: unknown qp_approx {}: qp_eqn or lqp.",
               qp_params.qp_approx);
  utils::check(qp_params.qp_scf_mode == "qpscf" or qp_params.qp_scf_mode == "evscf",
               "scf_driver.cpp::qp_scf_loop: unknown qp_scf_mode {}: qpscf or evscf.",
               qp_params.qp_scf_mode);
  utils::check(not (qp_params.qp_scf_mode == "evscf" and qp_params.qp_approx == "lqp"),
               "scf_driver.cpp::qp_scf_loop: qp_scf_mode = evscf with qp_approx = lqp is not "
               "implemented. Use qp_scf_mode = qpscf (solver_type = lqsgw) for the linearized "
               "scheme.");
  if (qp_params.qp_approx == "qp_eqn") {
    auto const& sol = qp_params.qp_eqn.solver;
    utils::check(sol=="sc" or sol=="sc_newton" or sol=="sc_bisection" or
                 sol=="linearized" or sol=="spectral",
                 "scf_driver.cpp::qp_scf_loop: unknown qp_eqn.solver {}: sc, sc_bisection, "
                 "sc_newton, spectral or linearized.", sol);
  }
  // Re-quasiparticle-ization of an external Green's function. gf_grp empty (the default):
  // iteration 1 builds its own analytic QP G. When set ("scf"/"embed"), iteration 1 consumes
  // the EXTERNAL G of that checkpoint group for the HF density matrix (the Hartree term of
  // the lattice density) and for the Sigma^GW/W build; iterations >= 2 revert to the loop's
  // own QP G.
  const bool ext_gf = not gf_grp.empty();
  utils::check(not ext_gf or gf_grp == "scf" or gf_grp == "embed",
               "qp_scf_loop: greens_func_source = \"{}\" is not supported. Valid options: "
               "\"\" (inert, the loop's own analytic QP G), \"scf\", \"embed\".", gf_grp);
  utils::check(not ext_gf or qp_params.qp_scf_mode != "evscf",
               "qp_scf_loop: the external Green's function injection (greens_func_source = "
               "\"{}\") is not implemented for qp_scf_mode = \"evscf\" -- use qp_scf_mode = "
               "\"qpscf\" or leave greens_func_source empty.", gf_grp);
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20qp-scf
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐ ┌─┐   ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║  │─┼┐├─┘───└─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘└┴     └─┘└─┘└  \n");
  app_log(1, "  Maximum iteration number    = {}", niter);
  utils::check(mb_solver.scr_eri == nullptr or not mb_solver.scr_eri->sigma_dyn_final_full(),
               "qp_scf_loop: pol_vertex_sigma_dyn_final_full is implemented for the Dyson scf loop only.");
  app_log(1, "  QP-SCF mode                 = {}", qp_params.qp_scf_mode);
  app_log(1, "  QP approximation            = {}", qp_params.qp_approx);
  app_log(1, "  Keep screened Coulomb fixed = {}", qp_params.keep_scr_coulomb_fixed);
  app_log(1, "  Convergence tolerance       = {}", conv_tol);
  app_log(1, "  Checkpoint HDF5             = {}", mb_state.coqui_prefix+".mbpt.h5");
  app_log(1, "  Restart                     = {}", (restart)? "yes" : "no");
  if (ext_gf) {
    app_log(1, "  External G (iteration 1)    = {}/iter{}", gf_grp, gf_iter);
  }
  app_log(1, "  Number of processors        = {} cores per node, {} nodes\n",
          mpi->node_comm.size(), mpi->internode_comm.size());
  FT.metadata_log();

  // the shared device pool for the duration of the loop, as in scf_loop (inert unless
  // COQUI_DEVICE_POOL_GB is set)
  std::optional<utils::device_pool_guard> pool_guard;
  if constexpr (MEM != HOST_MEMORY) {
    if (std::size_t pool_bytes = utils::device_pool_size_from_env(); pool_bytes > 0)
      pool_guard.emplace(pool_bytes, "qp-scf");
  }

  Timer.start("SCF_TOTAL");
  mb_state.sHeff_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sMO_skia.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sE_ska.emplace(make_shared_array<Array_view_3D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()}));
  // pole weights are set only by add_lqp_vcorr: clear any left from an earlier run on this
  // MBState, so dump_qpscf never writes another scheme's weights
  mb_state.sZqp_ska.reset();
  auto& sHeff_skij = mb_state.sHeff_skij.value();
  auto& sDm_skij = mb_state.sDm_skij.value();
  auto& sMO_skia = mb_state.sMO_skia.value();
  auto& sE_ska = mb_state.sE_ska.value();
  double mu = 0.0;

  // pseudopotential handler 
  auto psp = hamilt::make_pseudopot(*mf);
  long init_it = 0;
  if (!restart) {
    hamilt::set_fock(*mf, psp.get(), sHeff_skij, false);
  } else {
    init_it = chkpt::read_qpscf(mpi->node_comm, sHeff_skij, mu, mb_state.coqui_prefix);
  }
  
  auto sH0_skij = make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sS_skij = make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  hamilt::set_H0(*mf, psp.get(), sH0_skij);
  hamilt::set_ovlp(*mf, sS_skij);

  // Obtains MO coefficients and energies from the given mean-field object
  Timer.start("CANONICALIZATION");
  update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
  mu = update_mu(mu, *mf, sE_ska, FT.beta(), qp_params.mu_tolerance, qp_params.mu_update_alg);
  update_Dm(sDm_skij, sMO_skia, sE_ska, mu, FT.beta());
  Timer.stop("CANONICALIZATION");

  // ITERATION 1 consumes an EXTERNAL G instead of the restart-H_eff's analytic QP G. Two
  // consumers, one object:
  //   (a) sDm_skij <- Dm[G_ext] here, for the HF stage (Hartree term of the lattice density);
  //   (b) sG_ext handed to add_qpscf_vcorr below, so update_w AND the Sigma^GW build screen
  //       with the SAME G (W_corr = W[P^RPA[G_ext] + P^lad + P_C(P_imp-P_dc)P_C^dag]).
  // Dm-from-G follows the Dyson convention verbatim (simple_dyson.cpp, dca_dyson.cpp):
  // Dm = -G(tau -> beta). NOTE: mu is NOT taken from the external checkpoint -- the qp/map
  // stage keeps the loop's OWN spectrum and chemical potential (the CD kernel evaluates
  // Sigma at the MAP stage from the loop's spectrum).
  std::optional<sArray_t<Array_view_5D_t> > sG_ext;
  if (ext_gf) {
    const std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    utils::check(std::filesystem::exists(filename),
                 "qp_scf_loop: greens_func_source = \"{}\" while the CoQui checkpoint {} does "
                 "not exist.", gf_grp, filename);
    if (gf_iter < 0) {
      auto [scf_it, df1e_it, df2e_it, embed_it] = chkpt::read_input_iterations(filename);
      (void) df1e_it; (void) df2e_it;
      gf_iter = (gf_grp == "embed")? embed_it : scf_it;
      utils::check(gf_iter >= 0,
                   "qp_scf_loop: greens_func_iteration = -1 with greens_func_source = \"{}\", "
                   "but {} carries no \"{}/final_iter\".", gf_grp, filename, gf_grp);
    }
    app_log(1, "\n  External-G re-QP-ization: iteration {} consumes the external Green's "
               "function {}/iter{} of {}.\n", init_it+1, gf_grp, gf_iter, filename);
    sG_ext.emplace(read_greens_function(*mpi, mf.get(), filename, gf_iter, gf_grp));
    FT.check_leakage(sG_ext.value(), imag_axes_ft::fermion, "external Green's function");
    sDm_skij.win().fence();
    if (mpi->node_comm.root()) {
      auto Dm = sDm_skij.local();
      FT.tau_to_beta(sG_ext.value().local(), Dm);
      Dm *= -1;
    }
    sDm_skij.win().fence();
    mpi->comm.barrier();
  }

  Timer.start("WRITE");
  if (!restart) {
    chkpt::write_metadata(mpi->comm, *mf, FT, sH0_skij, sS_skij, mb_state.coqui_prefix);
    mb_state.dump_qpscf(0, mu);
  }
  Timer.stop("WRITE");

  double Heff_conv = std::numeric_limits<double>::infinity();
  double e_tot  = 0.0;
  double e_diff = 0.0;
  // the energy contributions {non-interacting (H0), beyond H0} and their change per iteration
  std::vector<double> energies(2, 0.0), energies_diff(2, 0.0);
  // as in scf_loop: converged when every energy contribution has stopped changing and, 
  // Heff as well (if iter_solver is present).  
  auto converged = [&]() {
    bool energy_converged = std::all_of(energies_diff.begin(), energies_diff.end(),
                                        [conv_tol](double x) { return std::abs(x) < conv_tol; });
    if (iter_solver != nullptr)
      return energy_converged and std::abs(Heff_conv) <= std::abs(conv_tol);
    return energy_converged;
  };
  // the causality stages of scf_loop before the quasiparticle reduction (sigma_causal_stage.hpp): the
  // projection of Sigma_c (on when a vertex adds to Sigma) and the causality meter / guard; one
  // instance for the warm starts and the meter's history
  qp_causality_t causal_stage;
  long it = init_it + 1;
  do {
    app_log(1, "\n** Iteration # {} **", it);

    Timer.start("MBPT_SOLVERS");
    if (mb_solver.hf != nullptr) { // HF
      if (mb_eri.hf_eri) {
        mb_solver.hf->evaluate(sHeff_skij, sDm_skij.local(), mb_eri.hf_eri->get(),
                               sS_skij.local(), true, true);
      } else if (mb_eri.hartree_eri and mb_eri.exchange_eri) {
        mb_solver.hf->evaluate(sHeff_skij, sDm_skij.local(), mb_eri.hartree_eri->get(),
                               sS_skij.local(), true, false);
        // create temporary buffer for K since hf_solver.evaluate(F) performs in-place evaluation for F.
        sArray_t<Array_view_4D_t> sK_skij(math::shm::make_shared_array<Array_view_4D_t>(*mpi, sHeff_skij.shape()));
        mb_solver.hf->evaluate(sK_skij, sDm_skij.local(), mb_eri.exchange_eri->get(),
                               sS_skij.local(), false, true);
        if (mpi->node_comm.root()) {
          sHeff_skij.local() += sK_skij.local();
        }
      } else {
        mb_solver.hf->evaluate(sHeff_skij, sDm_skij.local(), mb_eri.corr_eri->get(),
                               sS_skij.local(), true, true);
      }
      mpi->comm.barrier();
      sHeff_skij.win().fence();
      for (size_t sk = mpi->node_comm.rank(); sk < mf->nspin()*mf->nkpts_ibz(); sk += mpi->node_comm.size()) {
        size_t is = sk / mf->nkpts_ibz();
        size_t ik = sk % mf->nkpts_ibz();
        sHeff_skij.local()(is, ik, nda::ellipsis{}) += sH0_skij.local()(is, ik, nda::ellipsis{});
      }
      sHeff_skij.win().fence();
      mpi->comm.barrier();
    }
    // The lineshape meter is populated by qp_approx, i.e. by the qpscf MAP stage. Reset it
    // here so an iteration that never reaches that stage (evscf mode, or no corr solver at
    // all) reports the MISSING sentinel in the summary line below instead of a stale value
    // left by an earlier loop in the same process.
    q6_lineshape() = q6_lineshape_t{};
    if (mb_solver.corr != nullptr) { // GW
      mb_solver.corr->iter() = it;
      if (qp_params.qp_scf_mode == "evscf") {
        // add_evscf_vcorr update two things in mb_state: 
        // 1. QP energies
        // 2. sHeff_skij with the updated QP energies while keeping sMO_skia the same.
        add_evscf_vcorr<MEM>(mb_state, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_params,
                             qp_params.keep_scr_coulomb_fixed, &causal_stage);
      } else {
        // add_qpscf_vcorr / add_lqp_vcorr only update sHeff_skij (add_lqp_vcorr also stores the
        // pole weights). MO_skia and E_ska are updated later.
        // sG_ext is non-null in ITERATION 1 ONLY -- it is released right after, so
        // iterations >= 2 fall back to the loop's own analytic QP G.
        if (qp_params.qp_approx == "lqp") {
          add_lqp_vcorr<MEM>(mb_state, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_params,
                             sG_ext? std::addressof(sG_ext.value()) : nullptr, &causal_stage);
        } else {
          add_qpscf_vcorr<MEM>(mb_state, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_params,
                               sG_ext? std::addressof(sG_ext.value()) : nullptr, &causal_stage);
        }
        sG_ext.reset();
      }
    }
    Timer.stop("MBPT_SOLVERS");

    Timer.start("ITERATIVE");
    if (iter_solver != nullptr)
      Heff_conv = solve_iterative(*mpi, *iter_solver, it, mb_state.coqui_prefix, sHeff_skij, sS_skij);
    Timer.stop("ITERATIVE");

    Timer.start("CANONICALIZATION");
    if (qp_params.qp_scf_mode != "evscf") {
      // update MO_skia and E_ska
      update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
    }
    mu = update_mu(mu, *mf, sE_ska, FT.beta(), qp_params.mu_tolerance, qp_params.mu_update_alg);
    update_Dm(sDm_skij, sMO_skia, sE_ska, mu, FT.beta());
    Timer.stop("CANONICALIZATION");

    auto k_weight = mf->k_weight();
    // e_beyond_h0 = 1/2 Tr[Dm (Heff - H0)]: the interaction energy of the static QP Hamiltonian
    // (Hartree-Fock plus its static correlation potential), not a Hartree-Fock energy
    auto [e_1e, e_beyond_h0] = eval_hf_energy(sDm_skij, sHeff_skij, sH0_skij, k_weight, true);
    e_diff = e_tot - (e_1e + e_beyond_h0);
    e_tot  = e_1e + e_beyond_h0;
    energies_diff = {e_1e - energies[0], e_beyond_h0 - energies[1]};
    energies = {e_1e, e_beyond_h0};

    // print energies and scf convergence
    app_log(1, "\nEnergy contributions");
    app_log(1, "--------------------");
    app_log(1, "  non-interacting (H0):           {} a.u.", e_1e);
    app_log(1, "  beyond H0:                      {} a.u.", e_beyond_h0);
    app_log(1, "  total energy:                   {} a.u.", e_tot);
    app_log(1, " ");
    app_log(1, "energy difference:                {} a.u.", e_diff);
    app_log(1, "abs max diff of QP Hamiltonian:    {} a.u.\n", Heff_conv);

    // ---- qpGW iteration summary --------
    // ONE consolidated summary line per qp iteration. Strictly a READ of meters that already
    // exist: Heff_conv (this loop), qp_modea::last_run() (the mode-A inner loop + the strip
    // census; -1/0 for every other map), q6_lineshape() (the map-stage meter, which
    // qp_approx populates for EVERY map), and the scr_coulomb_t ladder-injection meters.
    // Nothing here is computed.
    // NOT AVAILABLE IN C++: the band-reordering count is a PYTHON-side meter
    // (dmft/outer_loop.py::count_band_reorderings); it is reported as the -1 MISSING
    // sentinel rather than duplicated here.
    {
      auto const &LR = qp_modea::last_run();
      auto const &LS = q6_lineshape();
      const bool has_scr = (mb_solver.scr_eri != nullptr);
      const double lam_max   = has_scr ? mb_solver.scr_eri->pol_lambda_max()   : -1.0;
      const double lad_ratio = has_scr ? mb_solver.scr_eri->pol_ladder_ratio() : -1.0;
      const double r_rt      = has_scr ? mb_solver.scr_eri->pol_round_trip()   : -1.0;
      app_log(1, "[qpGW summary] iteration {}: dmax(H_eff) = {:.3e} a.u., "
                 "dmax(map inner) = {:.3e}, inner-consist iters = {}, band-reorder = {} "
                 "(python-side meter), strip census in-strip/eta-far/clamped = {}/{}/{}, "
                 "wgrid_aud = {:.4g}/{:.4g} meV (worst q = {}, Re z = {:+.6g}), "
                 "lambda_max = {:.6f}, ||P^lad||/||P^RPA|| = {:.6e}, r_rt = {:.3e}, "
                 "lineshape max |Sigma^c - V^xc|/|Sigma^c| iw_0/iw_top = {:.6e}/{:.6e} "
                 "(mean {:.6e}/{:.6e}, {} states; ABS discard max iw_0/iw_top = "
                 "{:.6e}/{:.6e} a.u.)",
              it, Heff_conv, LR.dmax, LR.iters, -1,
              LR.n_eval - LR.n_clamp, LR.n_eta, LR.n_clamp - LR.n_eta,
              LR.wgrid_meas_mev, LR.wgrid_pred_mev, LR.wgrid_worst_q, LR.wgrid_worst_z,
              lam_max, lad_ratio, r_rt,
              LS.frac_w0_max, LS.frac_top_max, LS.frac_w0_mean, LS.frac_top_mean,
              LS.n_states, LS.abs_w0_max, LS.abs_top_max);
    }

    Timer.start("WRITE");
    mb_state.dump_qpscf(it, mu);
    Timer.stop("WRITE");
    // the causality guard of this iteration's update (vertex_debug scf_causality_abort), after its checkpoint
    utils::check(causal_stage.abort_msg.empty(), "{}", causal_stage.abort_msg);

    it++;
  } while(it<init_it+niter+1 and not converged());
  Timer.stop("SCF_TOTAL");

  app_log(2, "\n  QP-SCF timers");
  app_log(2, "  -------------");
  app_log(2, "    Total:                  {0:.3f} sec", Timer.elapsed("SCF_TOTAL"));
  app_log(2, "    Canonicalization:       {0:.3f} sec", Timer.elapsed("CANONICALIZATION"));
  app_log(2, "    MBPT solvers:           {0:.3f} sec", Timer.elapsed("MBPT_SOLVERS"));
  app_log(2, "    Iterative alg:          {0:.3f} sec", Timer.elapsed("ITERATIVE"));
  app_log(2, "    Write:                  {0:.3f} sec\n", Timer.elapsed("WRITE"));

  app_log(1, "####### quasi-particle SCF routines end #######\n");
  return e_tot;
}

/** Instantiation of public templates **/
// standard dyson for gw/hf
#define GW_SCF_LOOP_INST_MEM(MEM, HF, HARTREE, EXCHANGE, CORR) \
template std::tuple<double, double> \
scf_loop<MEM>(MBState&, simple_dyson&, \
         mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
         const imag_axes_ft::IAFT&, \
         solvers::mb_solver_t<solvers::gw_t>, \
         iter_scf::iter_scf_t*, \
         int, bool, double, bool, std::string, int, bool);

#define GW_SCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
   GW_SCF_LOOP_INST_MEM(HOST_MEMORY, HF, HARTREE, EXCHANGE, CORR)

// DEVICE_MEMORY instantiation only when the corr ERI is THC (the only
// ERI type whose gw_t::evaluate / scr_coulomb_t::update_w currently has
// a real device path; the Cholesky overload static_asserts HOST).
#if defined(ENABLE_DEVICE)
#  define GW_SCF_LOOP_INST_DEV_THC(HF, HARTREE, EXCHANGE) \
   GW_SCF_LOOP_INST_MEM(DEVICE_MEMORY, HF, HARTREE, EXCHANGE, thc_reader_t)
#else
#  define GW_SCF_LOOP_INST_DEV_THC(HF, HARTREE, EXCHANGE)
#endif

// All combinations of thc/chol for 4 eri slots (HOST)
GW_SCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

// DEVICE instantiations only for CORR=thc_reader_t
GW_SCF_LOOP_INST_DEV_THC(thc_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST_DEV_THC(thc_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST_DEV_THC(thc_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST_DEV_THC(thc_reader_t, chol_reader_t, chol_reader_t)
GW_SCF_LOOP_INST_DEV_THC(chol_reader_t, thc_reader_t, thc_reader_t)
GW_SCF_LOOP_INST_DEV_THC(chol_reader_t, thc_reader_t, chol_reader_t)
GW_SCF_LOOP_INST_DEV_THC(chol_reader_t, chol_reader_t, thc_reader_t)
GW_SCF_LOOP_INST_DEV_THC(chol_reader_t, chol_reader_t, chol_reader_t)

#undef GW_SCF_LOOP_INST
#undef GW_SCF_LOOP_INST_DEV_THC
#undef GW_SCF_LOOP_INST_MEM


// standard dyson for gf2
#define GF2_SCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template std::tuple<double, double> \
scf_loop(MBState&, simple_dyson&, \
         mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
         const imag_axes_ft::IAFT&, \
         solvers::mb_solver_t<solvers::gf2_t>, \
         iter_scf::iter_scf_t*, \
         int, bool, double, bool, std::string, int, bool);

// All combinations of thc/chol for 4 eri slots
GF2_SCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
GF2_SCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef GF2_SCF_LOOP_INST


#define QPSCF_LOOP_INST_MEM(MEM, HF, HARTREE, EXCHANGE, CORR) \
template double                                      \
qp_scf_loop<MEM>(MBState&,                         \
            mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
            const imag_axes_ft::IAFT&,         \
            qp_context_t&, \
            solvers::mb_solver_t<solvers::gw_t>,       \
            iter_scf::iter_scf_t*, \
            int, bool, double, std::string, long);

#define QPSCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
   QPSCF_LOOP_INST_MEM(HOST_MEMORY, HF, HARTREE, EXCHANGE, CORR)

// DEVICE_MEMORY instantiation only when the corr ERI is THC, as for scf_loop
#if defined(ENABLE_DEVICE)
#  define QPSCF_LOOP_INST_DEV_THC(HF, HARTREE, EXCHANGE) \
   QPSCF_LOOP_INST_MEM(DEVICE_MEMORY, HF, HARTREE, EXCHANGE, thc_reader_t)
#else
#  define QPSCF_LOOP_INST_DEV_THC(HF, HARTREE, EXCHANGE)
#endif

// All combinations of thc/chol for 4 eri slots (HOST)
QPSCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
QPSCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
QPSCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
QPSCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

// DEVICE instantiations only for CORR=thc_reader_t
QPSCF_LOOP_INST_DEV_THC(thc_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST_DEV_THC(thc_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST_DEV_THC(thc_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST_DEV_THC(thc_reader_t, chol_reader_t, chol_reader_t)
QPSCF_LOOP_INST_DEV_THC(chol_reader_t, thc_reader_t, thc_reader_t)
QPSCF_LOOP_INST_DEV_THC(chol_reader_t, thc_reader_t, chol_reader_t)
QPSCF_LOOP_INST_DEV_THC(chol_reader_t, chol_reader_t, thc_reader_t)
QPSCF_LOOP_INST_DEV_THC(chol_reader_t, chol_reader_t, chol_reader_t)

#undef QPSCF_LOOP_INST
#undef QPSCF_LOOP_INST_DEV_THC
#undef QPSCF_LOOP_INST_MEM

}
