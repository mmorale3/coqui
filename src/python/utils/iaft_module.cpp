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


#include <c2py/c2py.hpp>

#include "numerics/imag_axes_ft/iaft_utils.hpp"


// ==========  Module declaration ==========
namespace c2py_module {
  // Name of the package if any. Default is ""
  auto package_name = ""; // the module will be Package.MyModule

  // The documentation string of the module. Default = ""
  auto documentation = "IAFT module for CoQui ";

  // -------- Automatic selection of function, classes, enums -----------
  auto match_names = "imag_axes_ft::(ir::IR|IAFT|read_iaft)";
  // FIXME CNY: How do we explicit exclude one of the constructor in IAFT?
  //       This allows us to hide ir::IR at the Python level
  //auto reject_names = ".*";

} // namespace c2py_module