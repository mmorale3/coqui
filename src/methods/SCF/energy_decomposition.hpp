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

#ifndef METHODS_SCF_ENERGY_DECOMPOSITION_HPP
#define METHODS_SCF_ENERGY_DECOMPOSITION_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "mean_field/MF.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/paw/paw_onecenter.hpp"
#include "methods/SCF/scf_common.hpp"

namespace methods
{

/**
 * Split the one-electron energy e_1e = Tr[Dm*H0] into the individual terms that
 * a PAW code accounts separately, so that a volume series can be compared
 * against an external reference term by term.
 *
 * Printed terms (all Hartree, all with the same spin factor and k-weights as
 * e_1e itself — they are obtained by feeding a single-term H matrix through the
 * SAME eval_hf_energy contraction, so no prefactor is re-derived here):
 *
 *   kinetic      Tr[Dm*T]                       <-> ABINIT `kinetic`
 *   dion         Tr[Dm*V_NL(dion)]              <-> ABINIT `e1t10`(PAW) or
 *                                                   `non_local_psp`(NCPP)
 *   ex_cvij      Tr[Dm*V_NL(ex_cvij)]           <-> ABINIT `dijfock_cv`, i.e.
 *                                                   this belongs with EXCHANGE
 *   int_VQ       Tr[Dm*V_NL(∫V_loc·Q̂)]          <-> descreening; ABINIT folds
 *                                                   the equivalent into
 *                                                   `local_psp` + `psp_core`
 *   local        e_1e - (the above)             <-> ABINIT `local_psp` +
 *                                                   `psp_core`, jointly with
 *                                                   int_VQ
 *
 * Why this exists: the Si PAW EXX+RPA equation of state has no minimum while
 * the norm-conserving series through the identical pipeline gives
 * a0 = 10.2259 Bohr / B0 = 101.1 GPa. The PAW-minus-NC total energy drifts
 * -22.3 mHa across a = 10.05..10.55 Bohr and 78% of that drift sits in e_1e,
 * so e_1e has to be resolved into terms that map onto an external reference.
 * Full protocol: notes/paw_article_results/eos_exchange_ledger.md.
 *
 * `sScratch` is used as workspace and is left holding the last piece built;
 * pass a matrix the caller no longer needs.
 */
template<typename X_t, nda::ArrayOfRank<1> Array1D>
void print_e1_decomposition(mf::MF &mf, hamilt::pseudopot *psp,
                            X_t const &sDm_skij, X_t &sScratch,
                            Array1D &k_weight, double e_1e)
{
  utils::check(psp != nullptr, "print_e1_decomposition: null pseudopot.");

  // Tr[Dm*M] with e_1e's exact spin factor / k-weights: eval_hf_energy's first
  // return value IS that contraction, so reuse it rather than re-deriving the
  // prefactors here (the -1/(4N_k) class of bug).
  auto trace_with_dm = [&](X_t const &sM) {
    auto [tr, dummy] = eval_hf_energy(sDm_skij, sM, sM, k_weight, false);
    (void) dummy;
    return tr;
  };

  hamilt::set_kinetic(mf, psp, sScratch);
  double e_kin = trace_with_dm(sScratch);

  double e_dion = 0.0, e_cv = 0.0, e_vq = 0.0;
  if (psp->pp_type() == hamilt::pp_ncpp_t) {
    // NCPP: the whole non-local pseudopotential is the species-resolved dion.
    hamilt::set_vnl_only(mf, psp, psp->Dnn_view(), sScratch);
    e_dion = trace_with_dm(sScratch);
  } else {
    auto D_cv = psp->static_D_cv_only();                 // (nat, nhm, nhm)
    hamilt::set_vnl_only(mf, psp, D_cv, sScratch);
    e_cv = trace_with_dm(sScratch);

    // dion replica = Dnn_atom_static - ex_cvij part.
    auto D_st = psp->Dnn_atom_view();
    nda::array<ComplexType, 3> D_dion(D_st.shape());
    for (long a = 0; a < D_st.extent(0); ++a)
      for (long i = 0; i < D_st.extent(1); ++i)
        for (long j = 0; j < D_st.extent(2); ++j)
          D_dion(a, i, j) = D_st(a, i, j) - D_cv(a, i, j);
    hamilt::set_vnl_only(mf, psp, D_dion, sScratch);
    e_dion = trace_with_dm(sScratch);

    // ∫V_loc·Q̂ = static_h0_D - Dnn_atom_static (Eq. (h0), settled 2026-07-24).
    auto const &D_h0 = psp->static_h0_D();
    nda::array<ComplexType, 3> D_vq(D_st.shape());
    for (long a = 0; a < D_st.extent(0); ++a)
      for (long i = 0; i < D_st.extent(1); ++i)
        for (long j = 0; j < D_st.extent(2); ++j)
          D_vq(a, i, j) = D_h0(a, i, j) - D_st(a, i, j);
    hamilt::set_vnl_only(mf, psp, D_vq, sScratch);
    e_vq = trace_with_dm(sScratch);

    // Genuine check: the three pieces were built by differencing D tensors, so
    // verify against the SAME quantity assembled in one shot from static_h0_D
    // (the D that add_Vpp itself uses). Catches an index/shape mismatch in the
    // differencing, which the printed sum below cannot see -- e_loc is taken by
    // difference and so makes that sum agree with e_1e unconditionally.
    hamilt::set_vnl_only(mf, psp, D_h0, sScratch);
    double e_vnl_direct = trace_with_dm(sScratch);
    double d_vnl = std::abs(e_vnl_direct - (e_dion + e_cv + e_vq));
    utils::check(d_vnl < 1e-9 * std::max(1.0, std::abs(e_vnl_direct)),
        "print_e1_decomposition: the split one-center pieces do not add up to "
        "Tr[Dm V_NL(static_h0_D)]: {} vs {} (diff {}). The dion/ex_cvij/int_VQ "
        "differencing is inconsistent with the D that add_Vpp uses.",
        e_dion + e_cv + e_vq, e_vnl_direct, d_vnl);
  }

  // Everything in H0 that is neither kinetic nor non-local is the smooth local
  // potential. Taken by difference (there is no local-only H0 flag), so the
  // printed sum matching e_1e is bookkeeping, NOT evidence -- the check above
  // is what tests the pieces.
  double e_loc = e_1e - e_kin - e_dion - e_cv - e_vq;

  app_log(2, "\n  One-electron energy decomposition (Ha)");
  app_log(2, "  --------------------------------------");
  app_log(2, "    kinetic   Tr[Dm T]              = {}", e_kin);
  app_log(2, "    local     Tr[Dm V_loc]          = {}", e_loc);
  app_log(2, "    dion      Tr[Dm V_NL(dion)]     = {}", e_dion);
  app_log(2, "    ex_cvij   Tr[Dm V_NL(ex_cvij)]  = {}", e_cv);
  app_log(2, "    int_VQ    Tr[Dm V_NL(intV Q)]   = {}", e_vq);
  app_log(2, "    ---------------------------------");
  app_log(2, "    sum                             = {}", e_kin+e_loc+e_dion+e_cv+e_vq);
  app_log(2, "    e_1e                            = {}", e_1e);
  app_log(2, "  NOTE: ex_cvij is frozen core-valence EXACT EXCHANGE. CoQui keeps it");
  app_log(2, "  in the static D (so it lands in e_1e); ABINIT accounts it with the");
  app_log(2, "  Fock energy (dijfock_cv). Move it across before comparing.\n");
}

} // namespace methods

#endif // METHODS_SCF_ENERGY_DECOMPOSITION_HPP
