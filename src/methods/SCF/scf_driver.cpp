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



#include <filesystem>
#include <optional>

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"

#include "methods/ERI/mb_eri_context.h"
#include "methods/tools/chkpt_utils.h"
#include "methods/SCF/qp_modea.hpp"   // Q6 §1.4(b): qp_modea::last_run(), read-only
#include "simple_dyson.h"
#include "dca_dyson.h"
#include "scf_driver.hpp"

namespace methods {

template<typename dyson_type, typename eri_t, typename corr_solver_t>
auto scf_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
              solvers::mb_solver_t<corr_solver_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
              int niter, bool restart, double conv_tol, bool const_mu,
              std::string input_grp, int input_iter)
              -> std::tuple<double, double> {
  utils::TimerManager Timer;
  auto mpi = mb_eri.corr_eri->get().mpi();
  auto mf = mb_eri.corr_eri->get().MF();
  utils::check(mpi == mb_state.mpi,
               "SCF loop: mpi context of mb_state and mb_eri should be the same!");
  utils::check(&FT == mb_state.ft,
               "SCF loop: imag_axes_ft of mb_state and scf_loop should be the same!");
  for( auto& v: {"SCF_TOTAL", "DYSON", "MBPT_SOLVERS", "ITERATIVE", "WRITE"} ) {
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
    chkpt::dump_scf(mpi->comm, 0, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, mb_state.coqui_prefix);
  }
  Timer.stop("WRITE");

  double F_conv, Sigma_conv;
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
  std::tie(mb_state.mbpt_iter, mb_state.df_1e_iter, mb_state.df_2e_iter, mb_state.embed_iter) =
      chkpt::read_input_iterations(mb_state.coqui_prefix+".mbpt.h5");
  long output_iter_init = mb_state.mbpt_iter+1;
  long output_iter = output_iter_init;
  // start SCF iteration
  do {
    app_log(1, "\n** Iteration # {} **", output_iter);
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
        mb_solver.scr_eri->update_w(mb_state, mb_eri.corr_eri->get(), output_iter);

      mb_solver.corr->iter() = output_iter;
      mb_solver.corr->evaluate(mb_state, mb_eri.corr_eri->get());
      // deallocate mb_state.dW_qtPQ after this since it's only used in the corr solver and can be very large for GW.
      // Exception (ISDF-Vertex): with an active vertex on the GLOBAL auxiliary basis, keep
      // W alive across the iteration boundary so eval_Pi_qdep (which runs BEFORE this
      // iteration's update_w) can evaluate Pi^C with the previous iteration's screened rung
      // (one-iteration lag; converges to the same self-consistent fixed point). With the
      // SECONDARY basis (Refinement 2) the vertex caches the DOWNFOLDED rung
      // Wbar = t W t^dag at update_w time instead (vertex_t::cache_w, notes/wbar_cache.md),
      // so dW is freed unconditionally here -- restoring the plain-GW memory profile.
      if (mb_solver.scr_eri == nullptr or not mb_solver.scr_eri->needs_dw_retention())
        mb_state.dW_qtPQ.reset();
      mpi->comm.barrier();
    }

    if (mpi->node_comm.root()) {
      hermitize_in_tau(sF_skij.local(), "Fock matrix");
      hermitize_in_tau(sSigma_tskij.local(), "dynamic self-energy");
    }
    mpi->comm.barrier();
    Timer.stop("MBPT_SOLVERS");


    Timer.start("ITERATIVE");
    if (iter_solver != nullptr) {
      std::tie(F_conv, Sigma_conv) = solve_iterative(*mpi, *iter_solver, output_iter,
                                                     mb_state.coqui_prefix,
                                                     sF_skij, sSigma_tskij, &FT);
    }
    Timer.stop("ITERATIVE");

    Timer.start("DYSON");
    // whether to update mu depends on const_mu
    update_G(dyson, *mf, FT, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, const_mu);
    if (mpi->node_comm.root()) {
      hermitize_in_tau(sDm_skij.local(), "density matrix");
      hermitize_in_tau(sG_tskij.local(), "Green's function");
    }
    mpi->comm.barrier();
    Timer.stop("DYSON");


    auto k_weight = mf->k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
    double e_corr = (mb_solver.corr != nullptr)? eval_corr_energy(mpi->comm, FT, sG_tskij, sSigma_tskij, k_weight) : 0.0;
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
    output_iter++;
  } while (output_iter<output_iter_init+niter and not converged());
  Timer.stop("SCF_TOTAL");

  app_log(2, "\n  Dyson-SCF timers");
  app_log(2, "  ----------------");
  app_log(2, "    Total:                {0:.3f} sec", Timer.elapsed("SCF_TOTAL"));
  app_log(2, "    Dyson:                {0:.3f} sec", Timer.elapsed("DYSON"));
  app_log(2, "    MBPT solvers:         {0:.3f} sec", Timer.elapsed("MBPT_SOLVERS"));
  app_log(2, "    Iterative alg:        {0:.3f} sec", Timer.elapsed("ITERATIVE"));
  app_log(2, "    Write:                {0:.3f} sec\n", Timer.elapsed("WRITE"));

  app_log(1, "####### SCF routines end #######\n");
  return std::make_tuple(energies[0]+energies[1], energies[2]);
}


template<typename eri_t, typename corr_solver_t>
double qp_scf_loop(
  MBState &mb_state, 
  eri_t &mb_eri, 
  const imag_axes_ft::IAFT& FT,
  qp_params_t& qp_params, 
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
  utils::check(qp_params.qp_type=="sc" or qp_params.qp_type=="sc_newton" or
               qp_params.qp_type=="sc_bisection" or qp_params.qp_type=="linearized" or qp_params.qp_type=="spectral",
               "qp_scf_loop: unknown qp_type {}: sc or linearized.", qp_params.qp_type);
  // Project 2 increment Q5 (notes/q5_option2_outer_loop_spec.md §1): the Option-2
  // re-QP-ization knobs. gf_grp EMPTY (the default) = INERT -- iteration 1 builds its own
  // analytic QP G exactly as before. When set ("scf"/"embed"), iteration 1 consumes the
  // EXTERNAL G of that checkpoint group for the HF density matrix (eq 3's Sigma^H[rho_latt])
  // and for the Sigma^GW/W build; iterations >= 2 revert to the loop's own QP G.
  const bool ext_gf = not gf_grp.empty();
  utils::check(not ext_gf or gf_grp == "scf" or gf_grp == "embed",
               "qp_scf_loop: greens_func_source = \"{}\" is not supported. Valid options: "
               "\"\" (inert, the loop's own analytic QP G), \"scf\", \"embed\".", gf_grp);
  utils::check(not ext_gf or qp_params.qp_scf_mode != "evscf",
               "qp_scf_loop: the external Green's function injection (greens_func_source = "
               "\"{}\") is not implemented for qp_scf_mode = \"evscf\" -- evGW keeps its own "
               "convention (notes/q5_option2_outer_loop_spec.md §4).", gf_grp);
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20qp-scf
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐ ┌─┐   ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║  │─┼┐├─┘───└─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘└┴     └─┘└─┘└  \n");
  app_log(1, "  Maximum iteration number    = {}", niter);
  app_log(1, "  Eigenvalue scf only         = {}", qp_params.qp_scf_mode == "evscf");
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

  Timer.start("SCF_TOTAL");
  mb_state.sHeff_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sMO_skia.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sE_ska.emplace(make_shared_array<Array_view_3D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()}));
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

  // Project 2 increment Q5 (spec §1 piece 1): ITERATION 1 consumes an EXTERNAL G instead of
  // the restart-H_eff's analytic QP G. Two consumers, one object:
  //   (a) sDm_skij <- Dm[G_ext] here, for the HF stage (eq 3's Sigma^H[rho_latt]);
  //   (b) sG_ext handed to add_qpscf_vcorr below, so update_w AND the Sigma^GW build screen
  //       with the SAME G (W_corr = W[P^RPA[G_ext] + P^lad + P_C(P_imp-P_dc)P_C^dag]).
  // Dm-from-G follows the Dyson convention verbatim (simple_dyson.cpp:143-145,
  // dca_dyson.cpp:227): Dm = -G(tau -> beta). NOTE: mu is NOT taken from the external
  // checkpoint -- the qp/map stage keeps the loop's OWN spectrum and chemical potential
  // (spec §1: the CD kernel evaluates Sigma at the MAP stage from the loop's spectrum).
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
    app_log(1, "\n  Q5 re-QP-ization (Option 2): iteration {} consumes the external Green's "
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
    chkpt::dump_scf(mpi->comm, 0, sDm_skij, sHeff_skij, sMO_skia, sE_ska, mu, mb_state.coqui_prefix);
  }
  Timer.stop("WRITE");

  double Heff_conv;
  double e_tot  = 0.0;
  double e_diff = 0.0;
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
    // Q6 §1.3: the lineshape meter is populated by qp_approx, i.e. by the qpscf MAP stage.
    // Reset it here so an iteration that never reaches that stage (evscf mode, or no corr
    // solver at all) reports the MISSING sentinel in the Q6 summary line below instead of a
    // stale value left by an earlier loop in the same process.
    q6_lineshape() = q6_lineshape_t{};
    if (mb_solver.corr != nullptr) { // GW
      mb_solver.corr->iter() = it;
      if (qp_params.qp_scf_mode == "evscf") {
        // add_evscf_vcorr update two things in mb_state: 
        // 1. QP energies
        // 2. sHeff_skij with the updated QP energies while keeping sMO_skia the same.
        add_evscf_vcorr(mb_state, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_params, qp_params.keep_scr_coulomb_fixed);
      } else {
        // add_qpscf_vcorr only updates sHeff_skij. MO_skia and E_ska are updated later.
        // Q5: sG_ext is non-null in ITERATION 1 ONLY -- it is released right after, so
        // iterations >= 2 fall back to the loop's own analytic QP G (spec §1).
        add_qpscf_vcorr(mb_state, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_params,
                        sG_ext? std::addressof(sG_ext.value()) : nullptr);
        sG_ext.reset();
      }
    }
    Timer.stop("MBPT_SOLVERS");

    Timer.start("ITERATIVE");
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
    auto [e_1e, e_hf] = eval_hf_energy(sDm_skij, sHeff_skij, sH0_skij, k_weight, true);
    e_diff = e_tot - (e_1e + e_hf);
    e_tot  = e_1e + e_hf;

    // print energies and scf convergence
    app_log(1, "\nEnergy contributions");
    app_log(1, "--------------------");
    app_log(1, "  non-interacting (H0):           {} a.u.", e_1e);
    app_log(1, "  beyond H0:                      {} a.u.", e_hf);
    app_log(1, "  total energy:                   {} a.u.", e_tot);
    app_log(1, " ");
    app_log(1, "energy difference:                {} a.u.", e_diff);
    app_log(1, "abs max diff of QP Hamiltonian:    {} a.u.\n", Heff_conv);

    // ---- Project 2 increment Q6 (notes/q6_diagnostics_closeout_spec.md §1.4(b)) --------
    // ONE consolidated summary line per qp iteration. Strictly a READ of meters that already
    // exist: Heff_conv (this loop), qp_modea::last_run() (the mode-A inner loop + the rev-3
    // strip census; -1/0 for every other map), q6_lineshape() (the Q6 map-stage meter, which
    // qp_approx populates for EVERY map), and the scr_coulomb_t Q3 injection meters. Nothing
    // here is computed.
    // NOT AVAILABLE IN C++: the band-reordering count is a PYTHON-side meter
    // (dmft/outer_loop.py::count_band_reorderings, Q5/R-Q5-2) and §1.4(b) forbids computing
    // anything new, so it is reported as the -1 MISSING sentinel rather than duplicated.
    {
      auto const &LR = qp_modea::last_run();
      auto const &LS = q6_lineshape();
      const bool has_scr = (mb_solver.scr_eri != nullptr);
      const double lam_max   = has_scr ? mb_solver.scr_eri->pol_lambda_max()   : -1.0;
      const double lad_ratio = has_scr ? mb_solver.scr_eri->pol_ladder_ratio() : -1.0;
      const double r_rt      = has_scr ? mb_solver.scr_eri->pol_round_trip()   : -1.0;
      app_log(1, "[Q6] qpgw iteration summary  it = {}: dmax(H_eff) = {:.3e} a.u., "
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
    chkpt::dump_scf(mpi->comm, it, sDm_skij, sHeff_skij, sMO_skia, sE_ska, mu, mb_state.coqui_prefix);
    Timer.stop("WRITE");

    it++;
  } while(it<init_it+niter+1 and std::abs(Heff_conv) > std::abs(conv_tol));
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
#define GW_SCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template std::tuple<double, double> \
scf_loop(MBState&, simple_dyson&, \
         mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
         const imag_axes_ft::IAFT&, \
         solvers::mb_solver_t<solvers::gw_t>, \
         iter_scf::iter_scf_t*, \
         int, bool, double, bool, std::string, int);

// All combinations of thc/chol for 4 eri slots
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

#undef GW_SCF_LOOP_INST


// standard dyson for gf2
#define GF2_SCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template std::tuple<double, double> \
scf_loop(MBState&, simple_dyson&, \
         mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
         const imag_axes_ft::IAFT&, \
         solvers::mb_solver_t<solvers::gf2_t>, \
         iter_scf::iter_scf_t*, \
         int, bool, double, bool, std::string, int);

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


#define QPSCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template double                                      \
qp_scf_loop(MBState&,                         \
            mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
            const imag_axes_ft::IAFT&,         \
            qp_params_t&, \
            solvers::mb_solver_t<solvers::gw_t>,       \
            iter_scf::iter_scf_t*, \
            int, bool, double, std::string, long);

// All combinations of thc/chol for 4 eri slots
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

#undef QPSCF_LOOP_INST

}
