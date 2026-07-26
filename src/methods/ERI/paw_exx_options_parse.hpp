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

#ifndef METHODS_ERI_PAW_EXX_OPTIONS_PARSE_HPP
#define METHODS_ERI_PAW_EXX_OPTIONS_PARSE_HPP

#include <string>
#include "IO/ptree/ptree_utilities.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace methods {

/**
 * Shared toml -> hamilt::paw_exx_options parser, used by both
 * [interaction.thc] (thc_reader_t) and [interaction.hamilt] (hamilt_eval_t)
 * so the option surface cannot drift between the two routes.
 * NB: core-valence exchange is NOT a toml option — it is auto-detected from
 * the h5 content in the pseudopot constructor (see cv_exchange_e).
 */
inline hamilt::paw_exx_options
parse_paw_exx_options(ptree const& pt, bool in_thc, std::string const& where) {
  hamilt::paw_exx_options o;
  std::string vv = io::tolower_copy(
      io::get_value_with_default<std::string>(pt, "vv_compensation", "moment"));
  utils::check(vv == "moment" || vv == "shape",
      "{}: unknown vv_compensation = '{}' (choices: 'moment', 'shape').", where, vv);
  o.vv_compensation = (vv == "shape") ? hamilt::vv_compensation_e::shape
                                      : hamilt::vv_compensation_e::moment;
  // Max angular momentum L of the compensation-charge multipoles.
  // Default -1 = full 2*lmax per species (complete augmentation, cf. VASP
  // LMAXPAW); a value >= 0 caps it (cf. VASP LMAXFOCK).
  o.aug_lmax = io::get_value_with_default<int>(pt, "aug_lmax", -1);
  // Δk-keyed Qfac cache budget for the direct v_x path (plan C3).
  o.qfac_cache_mb = io::get_value_with_default<int>(pt, "qfac_cache_mb", 256);
  o.validate(in_thc, where);
  return o;
}

} // methods

#endif // METHODS_ERI_PAW_EXX_OPTIONS_PARSE_HPP
