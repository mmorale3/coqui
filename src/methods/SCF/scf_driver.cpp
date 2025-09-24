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



#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"

#include "methods/ERI/mb_eri_context.h"
#include "methods/tools/chkpt_utils.h"
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
          mpi->comm.size(), mpi->internode_comm.size());
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
  long init_it = 0;
  if (!restart) {
    hamilt::set_fock(*mf, dyson.PSP(), sF_skij, true);
  } else {
    init_it = chkpt::read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu,
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
  double e_1e, e_hf, e_corr;
  double e_tot = 0.0;
  double e_tot_diff = 0.0;
  long it = 1 + init_it;
  do {
    app_log(1, "\n** Iteration # {} **", it);
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
        mb_solver.scr_eri->update_w(mb_state, mb_eri.corr_eri->get(), it);

      mb_solver.corr->iter() = it;
      mb_solver.corr->evaluate(mb_state, mb_eri.corr_eri->get());
      mpi->comm.barrier();
    }

    if (mpi->node_comm.root()) {
      hermitize(sF_skij.local(), "Fock matrix");
      hermitize(sSigma_tskij.local(), "dynamic self-energy");
    }
    mpi->comm.barrier();
    Timer.stop("MBPT_SOLVERS");


    Timer.start("ITERATIVE");
    std::tie(F_conv, Sigma_conv) = solve_iterative(*mpi, *iter_solver, it, mb_state.coqui_prefix,
                                                   sF_skij, sSigma_tskij, &FT, restart);
    Timer.stop("ITERATIVE");

    Timer.start("DYSON");
    // whether to update mu depends on const_mu
    update_G(dyson, *mf, FT, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, const_mu);
    if (mpi->node_comm.root()) {
      hermitize(sDm_skij.local(), "density matrix");
      hermitize(sG_tskij.local(), "Green's function");
    }
    mpi->comm.barrier();
    Timer.stop("DYSON");


    auto k_weight = mf->k_weight();
    std::tie(e_1e, e_hf) = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
    e_corr = (mb_solver.corr != nullptr)? eval_corr_energy(mpi->comm, FT, sG_tskij, sSigma_tskij, k_weight) : 0.0;
    double e_tot_new = e_1e + e_hf + e_corr;
    e_tot_diff = e_tot_new - e_tot;

    // print energies and scf convergence
    app_log(1, "\nEnergy contributions");
    app_log(1, "--------------------");
    app_log(1, "  non-interacting (H0):           {} a.u.", e_1e);
    app_log(1, "  Hartree-Fock:                   {} a.u.", e_hf);
    app_log(1, "  correlation:                    {} a.u.", e_corr);
    app_log(1, "  total energy:                   {} a.u.", e_tot_new);
    app_log(1, " ");
    app_log(1, "energy difference:                {} a.u.", e_tot_diff);
    app_log(1, "abs max diff of Fock matrix:   {}", F_conv);
    if (mb_solver.corr!=nullptr)
      app_log(1, "abs max diff of self-energy:   {}\n", Sigma_conv);
    e_tot  = e_tot_new;
    Timer.start("WRITE");
    chkpt::dump_scf(mpi->comm, it, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, mb_state.coqui_prefix);
    Timer.stop("WRITE");
    it++;
  } while(it<init_it+niter+1 and (std::abs(F_conv) > std::abs(conv_tol) or std::abs(Sigma_conv) > std::abs(conv_tol)));
  Timer.stop("SCF_TOTAL");

  app_log(2, "\n  Dyson-SCF timers");
  app_log(2, "  ----------------");
  app_log(2, "    Total:                {0:.3f} sec", Timer.elapsed("SCF_TOTAL"));
  app_log(2, "    Dyson:                {0:.3f} sec", Timer.elapsed("DYSON"));
  app_log(2, "    MBPT solvers:         {0:.3f} sec", Timer.elapsed("MBPT_SOLVERS"));
  app_log(2, "    Iterative alg:        {0:.3f} sec", Timer.elapsed("ITERATIVE"));
  app_log(2, "    Write:                {0:.3f} sec\n", Timer.elapsed("WRITE"));

  app_log(1, "####### SCF routines end #######\n");
  return std::make_tuple(e_1e+e_hf, e_corr);
}


template<bool evscf_only, typename eri_t, typename corr_solver_t>
double qp_scf_loop(MBState &mb_state, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                   qp_context_t& qp_context, solvers::mb_solver_t<corr_solver_t> mb_solver,
                   iter_scf::iter_scf_t *iter_solver,
                   int niter, bool restart, double conv_tol) {
  using math::shm::make_shared_array;
  utils::TimerManager Timer;
  auto mpi = mb_eri.corr_eri->get().mpi();
  auto mf = mb_eri.corr_eri->get().MF();
  for( auto& v: {"SCF_TOTAL", "CANONICALIZATION", "MBPT_SOLVERS", "ITERATIVE", "WRITE"} ) {
    Timer.add(v);
  }
  utils::check(qp_context.qp_type=="sc" or qp_context.qp_type=="sc_newton" or
               qp_context.qp_type=="sc_bisection" or qp_context.qp_type=="linearized" or qp_context.qp_type=="spectral",
               "qp_scf_loop: unknown qp_type {}: sc or linearized.", qp_context.qp_type);
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20qp-scf
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐ ┌─┐   ┌─┐┌─┐┌─┐\n"
             "║  ║ ║║═╬╗║ ║║  │─┼┐├─┘───└─┐│  ├┤ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘└┴     └─┘└─┘└  \n");
  app_log(1, "  - maximum iteration number:        {}", niter);
  app_log(1, "  - eigenvalue scf only:             {}", (evscf_only)? "true" : "false");
  app_log(1, "  - convergence tolerance:           {}", conv_tol);
  app_log(1, "  - output:                          {}", mb_state.coqui_prefix+".mbpt.h5");
  app_log(1, "  - Restart mode:                    {}", (restart)? "yes" : "no");
  app_log(1, "  - total number processors:         {}", mpi->comm.size());
  app_log(1, "  - number of nodes:                 {}\n", mpi->internode_comm.size());
  Timer.start("SCF_TOTAL");

  mb_state.sF_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  auto& sHeff_skij = mb_state.sF_skij.value();
  auto& sDm_skij = mb_state.sDm_skij.value();
  auto sH0_skij = make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sS_skij = make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  double mu = 0.0;
  // generates a new pseudopot object if not found in mf. Stores shared_ptr in mf and returns it.
  auto psp = hamilt::make_pseudopot(*mf);
  hamilt::set_H0(*mf, psp.get(), sH0_skij);
  hamilt::set_ovlp(*mf, sS_skij);
  long init_it = 0;
  if (!restart) {
    hamilt::set_fock(*mf, psp.get(), sHeff_skij, false);
  } else {
    init_it = chkpt::read_qpscf(mpi->node_comm, sHeff_skij, mu, mb_state.coqui_prefix);
  }

  auto sMO_skia = make_shared_array<Array_view_4D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
  auto sE_ska = make_shared_array<Array_view_3D_t>(*mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()});

  // Obtains MO coefficients and energies from the given mean-field object
  Timer.start("CANONICALIZATION");
  update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
  mu = update_mu(mu, *mf, sE_ska, FT.beta());
  update_Dm(sDm_skij, sMO_skia, sE_ska, mu, FT.beta());
  Timer.stop("CANONICALIZATION");

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
    if (mb_solver.corr != nullptr) { // GW
      mb_solver.corr->iter() = it;
      if constexpr (evscf_only) {
        // add_evscf_vcorr() does the following two things:
        // 1. return GW quasiparticle energies
        // 2. update sHeff_skij as a diagonal matrix whose diagonals correspond to GW qp energies.
        if (niter>1)
          add_evscf_vcorr<false>(mb_state, sE_ska, sMO_skia, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_context);
        else
          add_evscf_vcorr<true>(mb_state, sE_ska, sMO_skia, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_context);
        mpi->comm.barrier();
      } else {
        // add_qpscf_vcorr only updates sHeff_skij. sE_ska and sMO_skia are fixed.
        add_qpscf_vcorr(mb_state, sE_ska, sMO_skia, mu, mb_solver, mb_eri.corr_eri->get(), FT, qp_context);
      }
    }
    Timer.stop("MBPT_SOLVERS");

    Timer.start("ITERATIVE");
    Heff_conv = solve_iterative(*mpi, *iter_solver, it, mb_state.coqui_prefix, sHeff_skij);
    Timer.stop("ITERATIVE");

    Timer.start("CANONICALIZATION");
    if constexpr (!evscf_only) {
      // update MO_skia and E_ska
      update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
    }
    mu = update_mu(mu, *mf, sE_ska, FT.beta());
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

// dca-gw/dca-hf
/*template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<thc_reader_t, thc_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gw_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<thc_reader_t, chol_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gw_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<chol_reader_t, thc_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gw_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<chol_reader_t, chol_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gw_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
// dca-gf2
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<thc_reader_t, thc_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gf2_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<thc_reader_t, chol_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gf2_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<chol_reader_t, thc_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gf2_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);
template std::tuple<double, double>
scf_loop(utils::mpi_context_t<mpi3::communicator> &comm, dca_dyson & scf, mf::MF &mf,
         mb_eri_t<chol_reader_t, chol_reader_t> &eri, const imag_axes_ft::IAFT& FT,
         solvers::mb_solver_t<solvers::gf2_t> mb_solver, iter_scf::iter_scf_t *iter_solver,
         std::string output, int niter, bool restart, double conv_tol, bool const_mu,
         std::string input_grp, int input_iter);*/

#define EVSCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template double                                      \
qp_scf_loop<true>(MBState&,                          \
                  mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
                  const imag_axes_ft::IAFT&,         \
                  qp_context_t&, \
                  solvers::mb_solver_t<solvers::gw_t>,       \
                  iter_scf::iter_scf_t*, \
                  int, bool, double);

// All combinations of thc/chol for 4 eri slots
EVSCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
EVSCF_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
EVSCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
EVSCF_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
EVSCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
EVSCF_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
EVSCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
EVSCF_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
EVSCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
EVSCF_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
EVSCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
EVSCF_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
EVSCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
EVSCF_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
EVSCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
EVSCF_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef EVSCF_LOOP_INST


#define QPSCF_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template double                                      \
qp_scf_loop<false>(MBState&,                         \
                   mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
                   const imag_axes_ft::IAFT&,         \
                   qp_context_t&, \
                   solvers::mb_solver_t<solvers::gw_t>,       \
                   iter_scf::iter_scf_t*, \
                   int, bool, double);

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
