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


#ifndef COQUI_AC_TYPE_E_HPP
#define COQUI_AC_TYPE_E_HPP

namespace analyt_cont {
  enum ac_type_e {
    pade,
    // Thiele's recursion divides by the pivot g(i-1,i-1); when that underflows,
    // "pade" keeps going and the coefficients blow up. "pade_updated" stops the
    // recursion at the last well-conditioned order and evaluates the continued
    // fraction with a normalised (Lentz-style) recurrence instead of the raw
    // backward form. Same interpolant when the fit is well conditioned.
    pade_updated
  };

  inline std::string ac_enum_to_string(int ac_enum) {
    switch(ac_enum) {
      case ac_type_e::pade:
        return "pade";
      case ac_type_e::pade_updated:
        return "pade_updated";
      default:
        return "not recognized...";
    }
  }

  inline ac_type_e string_to_ac_enum(std::string ac_type) {
    if (ac_type == "pade") {
      return ac_type_e::pade;
    } else if (ac_type == "pade_updated") {
      return ac_type_e::pade_updated;
    } else {
      utils::check(false, "Unrecognized ac_type: {} (choices: pade, pade_updated)", ac_type);
      return ac_type_e::pade;
    }
  }

}

#endif //COQUI_AC_TYPE_E_HPP
