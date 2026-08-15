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

#ifndef HAMILTONIAN_PSEUDOPOT_QUERY_H
#define HAMILTONIAN_PSEUDOPOT_QUERY_H

#include "hamiltonian/pseudo/pseudopot_type.hpp"

namespace hamilt
{

// Cheap query: returns true if the MF (or backend) carries USPP/PAW data.
//
// Routines that compute orbital-orbital overlaps from only the smooth
// (pseudo) wavefunction silently lose the augmentation term
// Σ_a Q_ab^I <p̃_a|ψ̃><ψ̃|p̃_b> for USPP/PAW. This helper is the predicate
// they consult to decide whether to abort.
//
// Two-tier:
//   1) If a pseudopot is already attached (mf.get_pseudopot() non-null),
//      use psp->pp_type() — definitive, free.
//   2) Otherwise use the backend-cached mf.pp_type(); for qe-source this is
//      read once from /Hamiltonian/pp_type by qe_interface::read_h5, so no
//      pseudopot construction is needed.
//
// Templated so it works both on mf::MF (for callers that have the wrapper)
// and on the concrete backends seen by templated callers like
// utils::generate_dmatrix (which is parameterised on MF_t).
template<typename MF_t>
inline bool mf_requires_augmentation(MF_t &mf)
{
  auto t = mf.pp_type();
  return t == pp_uspp_t || t == pp_paw_t;
}

} // namespace hamilt

#endif
