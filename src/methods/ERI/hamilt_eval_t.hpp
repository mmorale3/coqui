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

#ifndef METHODS_ERI_HAMILT_EVAL_T_HPP
#define METHODS_ERI_HAMILT_EVAL_T_HPP

#include <memory>
#include <optional>
#include <string>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "methods/ERI/paw_exx_options_parse.hpp"

namespace methods {

/**
 * Direct-route ("hamilt") static-term evaluator — the `[interaction.hamilt]`
 * toml block. Fills the STATIC slots
 * of mb_eri_t (interaction_hf / interaction_hartree / interaction_exchange)
 * so that hf_t builds V_H and/or the exact-exchange K through the
 * hamiltonian routines (hamilt::Vhartree / hamilt::Vexchange) instead of
 * factorized ERIs. Never valid as the dynamic
 * (`interaction`) slot — dynamic self-energies always need factorized ERIs.
 *
 * The heavy state is (a) the shared pseudopot, acquired lazily through
 * hamilt::make_pseudopot (same instance as simple_dyson / thc_reader_t),
 * and (b) the distributed orbital set, read once and cached across SCF
 * iterations.
 */
class hamilt_eval_t {
public:
  using darray4_t =
      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>,
                       boost::mpi3::communicator>;

  hamilt_eval_t(std::shared_ptr<mf::MF> MF, ptree const& pt)
      : _MF(std::move(MF)),
        _exx_opts(parse_paw_exx_options(pt, /*in_thc=*/false,
                                        "hamilt_eval_t [interaction.hamilt]")) {}

  /// concept tag (methods::Hamilt_ERI)
  std::string hamilt_eval_type() const { return "direct"; }

  auto mpi() const { return _MF->mpi(); }
  std::shared_ptr<mf::MF> MF() const { return _MF; }

  hamilt::paw_exx_options const& exx_options() const { return _exx_opts; }

  /**
   * Lazily acquire the SHARED pseudopot (hamilt::make_pseudopot through the
   * MF) and propagate the exx options. Options already set by another
   * consumer (e.g. an [interaction.thc] block) must be either the defaults
   * or identical to ours — conflicting physics options are a hard error,
   * never last-writer-wins (plan item 0.2).
   */
  hamilt::pseudopot* psp() {
    if (!_psp) {
      _psp = hamilt::make_pseudopot(*_MF);
      utils::check(_psp != nullptr,
          "hamilt_eval_t: make_pseudopot returned null — the mean field has "
          "no pseudopotential support (mf_type without /Hamiltonian data?).");
      auto const& cur = _psp->exx_options();
      hamilt::paw_exx_options defs{};
      auto same = [](hamilt::paw_exx_options const& a,
                     hamilt::paw_exx_options const& b) {
        return a.vv_compensation == b.vv_compensation &&
               a.aug_lmax == b.aug_lmax && a.qfac_cache_mb == b.qfac_cache_mb;
      };
      utils::check(same(cur, defs) or same(cur, _exx_opts),
          "hamilt_eval_t: paw_exx_options conflict on the shared pseudopot — "
          "another interaction block already set different exx options "
          "(vv_compensation/aug_lmax/qfac_cache_mb). Make the blocks agree; "
          "conflicting physics options are not resolved silently.");
      _psp->set_exx_options(_exx_opts);
    }
    return _psp.get();
  }

  /**
   * Distributed orbital set over `comm`, read once and cached (the SCF loop
   * calls evaluate() every iteration with the same communicator — the eval
   * object must not re-read the orbitals each time). Full IBZ, all bands.
   */
  darray4_t& psi(boost::mpi3::communicator& comm) {
    if (!_psi) {
      using larray = memory::array<HOST_MEMORY, ComplexType, 4>;
      _psi.emplace(mf::read_distributed_orbital_set_ibz<larray>(
          *_MF, comm, 'w', std::array<long, 4>{0}, nda::range(-1, -1),
          nda::range(_MF->nkpts_ibz()), nda::range(_MF->nbnd()),
          std::array<long, 4>{1, 1, 2048, 2048}));
    }
    return *_psi;
  }

  /// drop the cached orbital set (e.g. to free memory after SCF)
  void release_psi() { _psi.reset(); }

private:
  std::shared_ptr<mf::MF> _MF;
  hamilt::paw_exx_options _exx_opts;
  std::shared_ptr<hamilt::pseudopot> _psp;
  std::optional<darray4_t> _psi;
};

} // methods

#endif // METHODS_ERI_HAMILT_EVAL_T_HPP
