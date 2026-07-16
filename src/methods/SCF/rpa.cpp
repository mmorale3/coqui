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
#include "scf_driver.hpp"


namespace methods {
template<typename eri_t, typename dyson_type>
double rpa_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                solvers::mb_solver_t<solvers::gw_t> mb_solver) {
  utils::TimerManager Timer;
  auto mpi = mb_eri.corr_eri->get().mpi();
  auto mf = mb_eri.corr_eri->get().MF();
  for( auto& v: {"RPA_TOTAL", "DYSON", "MBPT_SOLVERS", "WRITE"} ) {
    Timer.add(v);
  }

  Timer.start("RPA_TOTAL");
  app_log(1, "###### RPA energy routines start ######\n");
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
  hamilt::set_fock(*mf, dyson.PSP(), sF_skij, true);
  double mu = 0.0;

  Timer.start("WRITE");
  chkpt::write_metadata(mpi->comm, *mf, FT, dyson.sH0_skij(), dyson.sS_skij(), mb_state.coqui_prefix);
  Timer.stop("WRITE");

  Timer.start("DYSON");
  // init Green's function. By default, we update mu as well.
  update_G(dyson, *mf, FT, sDm_skij, sG_tskij, sF_skij, sSigma_tskij, mu, false);
  Timer.stop("DYSON");

  Timer.start("MBPT_SOLVERS");
  if (mb_solver.hf != nullptr) {
    // Fock matrix evaluated using the KS Green's function
    mb_solver.hf->evaluate(sF_skij, sDm_skij.local(), mb_eri.hf_eri->get(), dyson.sS_skij().local());
    mpi->comm.barrier();
  }
  // RPA energy evaluated using the KS Green's function
  double e_rpa = (mb_solver.corr != nullptr)? mb_solver.corr->rpa_energy(sG_tskij.local(), mb_eri.corr_eri->get()) : 0.0;
  Timer.stop("MBPT_SOLVERS");

  // HF energy: E_HF[G_KS, F[G_KS]]
  auto k_weight = mf->k_weight();
  auto [e_1e_new, e_hf_new] = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
  app_log(2, "\nOne-electron energy:       {} a.u.", e_1e_new);
  app_log(2, "Hartree-Fock energy:       {} a.u.", e_hf_new);
  app_log(2, "RPA energy:                {} a.u.", e_rpa);
  app_log(2, "Total energy:              {} a.u.\n", e_1e_new + e_hf_new + e_rpa);

  // Diagnostic: split e_1e = Tr[Dm*H0] into the smooth kinetic energy Tr[Dm*T]
  // and the remaining external/pseudopotential + one-center part, for
  // component-by-component comparison against ABINIT (kinetic / local_psp / ...).
  {
    auto sT_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
    hamilt::set_kinetic(*mf, dyson.PSP(), sT_skij);
    auto [e_kinetic, e_dummy] = eval_hf_energy(sDm_skij, sF_skij, sT_skij, k_weight, false);
    app_log(2, "Kinetic energy:            {} a.u.", e_kinetic);
    app_log(2, "External+1center energy:   {} a.u.\n", e_1e_new - e_kinetic);
  }

  // Diagnostic Hartree/Exchange split of the Fock energy (for code-vs-code
  // comparison, e.g. ABINIT "New Exchange energy"). Note: this is the smooth +
  // valence-valence one-center (K_a) exchange only; PAW core-valence exchange
  // (ex_cvij) is frozen into H0 and therefore counted in the one-electron term.
  if (mb_solver.hf != nullptr) {
    mb_solver.hf->evaluate(sF_skij, sDm_skij.local(), mb_eri.hf_eri->get(),
                           dyson.sS_skij().local(), true, false);
    mpi->comm.barrier();
    auto [e1_h, e_hartree] = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
    mb_solver.hf->evaluate(sF_skij, sDm_skij.local(), mb_eri.hf_eri->get(),
                           dyson.sS_skij().local(), false, true);
    mpi->comm.barrier();
    auto [e1_x, e_exchange] = eval_hf_energy(sDm_skij, sF_skij, dyson.sH0_skij(), k_weight, false);
    app_log(2, "Hartree energy:            {} a.u.", e_hartree);
    app_log(2, "Exchange energy (vv):      {} a.u.\n", e_exchange);
    // Per-state exchange self-energy diag at k=0 (Gamma) in eV, for direct
    // comparison with ABINIT's GW "SigX" column (vv-only; cv is in H0).
    if (mpi->comm.root()) {
      auto F = sF_skij.local();
      int nb = std::min(10, (int)mf->nbnd());
      app_log(2, "Per-state Sigma_x^vv at k=0 (eV):");
      for (int n = 0; n < nb; ++n)
        app_log(2, "   band {}: {:.4f}", n, F(0, 0, n, n).real() * 27.211386245988);
      app_log(2, "");
    }
  }

  // Diagnostic (env COQUI_VX_SHAPE_DIAG): direct-path exchange energy with the
  // compensated augmentation vs. the shape-restored "Option A" augmentation
  // (full AE−PS pair density, deltaC dropped). Compensated should approximately
  // reproduce the THC "Exchange energy (vv)" above; shape-restored is the
  // ABINIT-correct on-site exchange (Si a=10.20 bare: -1.7382 -> -1.6863 Ha).
  // Occupations are thresholded to the KS-occupied bands (|occ|>0.5): the direct
  // FFT exchange over all virtual bands is impractically slow and virtuals
  // contribute negligibly to the exchange energy. E_x = 0.5*spin*Σ_k w_k Σ_n
  // occ_n K(n,n) (diagonal density matrix). The full density-matrix (nij) path
  // is validated separately by the hamiltonian unit tests (nij==nii to 1e-16).
  if (std::getenv("COQUI_VX_SHAPE_DIAG") != nullptr) {
    long ns = mf->nspin(), nk = mf->nkpts_ibz(), nb = mf->nbnd();
    nda::array<ComplexType, 3> nii(ns, nk, nb);
    nii() = ComplexType(0.0);
    if (mpi->comm.root()) {
      auto Dm = sDm_skij.local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k)
          for (long n = 0; n < nb; ++n) {
            ComplexType occ = Dm(s, k, n, n);
            if (std::abs(occ.real()) > 0.5) nii(s, k, n) = occ;  // KS-occupied only
          }
    }
    mpi->comm.broadcast_n(nii.data(), nii.size(), 0);
    double spin_factor = (ns == 2) ? 1.0 : 2.0;
    auto sK = math::shm::make_shared_array<Array_view_4D_t>(*mpi, {ns, nk, nb, nb});
    hamilt::pseudopot V(*mf);
    for (int pass = 0; pass < 2; ++pass) {
      V.set_paw_exx_shape_restored(pass == 1);
      hamilt::set_Vexchange(*mf, &V, nii, sK);
      double e_x = 0.0;
      if (mpi->comm.root()) {
        auto Kloc = sK.local();
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long n = 0; n < nb; ++n)
              e_x += k_weight(k) * (nii(s, k, n) * Kloc(s, k, n, n)).real();
        e_x *= 0.5 * spin_factor;
      }
      mpi->comm.broadcast_n(&e_x, 1, 0);
      app_log(2, "Direct exchange, {}: {} a.u.",
              (pass == 0) ? "compensated   " : "shape-restored", e_x);
    }
    app_log(2, "");
  }

  Timer.start("WRITE");
  if (mpi->comm.root()) {
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    h5::file file(filename, 'a');
    h5::group grp(file);

    auto rpa_grp = grp.create_group("RPA");
    h5::h5_write(rpa_grp, "1e_energy", e_1e_new);
    h5::h5_write(rpa_grp, "hf_energy", e_hf_new);
    h5::h5_write(rpa_grp, "rpa_energy", e_rpa);
  }
  Timer.stop("WRITE");
  Timer.stop("RPA_TOTAL");

  app_log(2, "\n***************************************************");
  app_log(2, "                 RPA timers ");
  app_log(2, "***************************************************");
  app_log(2, "    Total:                {0:.3f} sec", Timer.elapsed("RPA_TOTAL"));
  app_log(2, "    Dyson:                {0:.3f} sec", Timer.elapsed("DYSON"));
  app_log(2, "    RPA solvers:          {0:.3f} sec", Timer.elapsed("MBPT_SOLVERS"));
  app_log(2, "    Write:                {0:.3f} sec", Timer.elapsed("WRITE"));
  app_log(2, "***************************************************\n");

  app_log(1, "####### RPA energy routines end #######\n");
  return e_rpa;
}

#define RPA_LOOP_INST(HF, HARTREE, EXCHANGE, CORR) \
template double \
rpa_loop(MBState&, simple_dyson&, \
         mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
         const imag_axes_ft::IAFT&, \
         solvers::mb_solver_t<solvers::gw_t>);

// All combinations of thc/chol for 4 eri slots
RPA_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
RPA_LOOP_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
RPA_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
RPA_LOOP_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
RPA_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
RPA_LOOP_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
RPA_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
RPA_LOOP_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
RPA_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
RPA_LOOP_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
RPA_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
RPA_LOOP_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
RPA_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
RPA_LOOP_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
RPA_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
RPA_LOOP_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef RPA_LOOP_INST

}
