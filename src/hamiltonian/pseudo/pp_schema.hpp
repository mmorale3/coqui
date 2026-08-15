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

#ifndef HAMILTONIAN_PSEUDO_PP_SCHEMA_HPP
#define HAMILTONIAN_PSEUDO_PP_SCHEMA_HPP

#include "h5/h5.hpp"

namespace hamilt
{

/**
 * On-disk unit convention of the /Hamiltonian pseudopotential datasets
 *
 *
 *   schema_version (int attribute on /Hamiltonian):
 *     absent : legacy export (QE deeq era) — energy datasets in Ry
 *     1      : deeq-free — still Ry
 *     >= 2   : deeq-free + HARTREE on disk — no scaling on read
 *     >= 3   : additionally /Orbitals@ecutrho is HARTREE (older pw2coqui
 *              wrote raw Ry — qe_interface::read_h5 scales x0.5 below v3)
 *              and datasets with no CoQui consumer are dropped by the
 *              converters.
 *              /Hamiltonian dataset units are unchanged from v2.
 *
 * Returns the version (0 when the attribute is absent).
 */
inline int h5_pp_schema_version(h5::group& hamil_grp)
{
  // Probe with H5Aexists instead of try/catch: the catch path made every
  // legacy-file open dump an HDF5-DIAG error stack to stderr.
  int sv = 0;
  if (H5Aexists(h5::hid_t(hamil_grp), "schema_version") > 0)
    h5::h5_read_attribute(hamil_grp, "schema_version", sv);
  return sv;
}

/** Read-time scale for energy-valued datasets: 0.5 (Ry→Ha) for legacy
 *  files, 1.0 for schema_version >= 2 (already Hartree on disk). */
inline double h5_pp_ry2ha(h5::group& hamil_grp)
{
  return (h5_pp_schema_version(hamil_grp) >= 2) ? 1.0 : 0.5;
}

} // namespace hamilt

#endif // HAMILTONIAN_PSEUDO_PP_SCHEMA_HPP
