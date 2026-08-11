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

#include "utilities/check.hpp"
#include "cvv_head.hpp"

namespace methods {
namespace solvers {

  cvv_head_t::cvv_head_t(const imag_axes_ft::IAFT *ft, double rspace_tol)
      : _ft(ft), _rspace_tol(rspace_tol) {
    utils::check(ft != nullptr, "cvv_head_t: null IAFT pointer.");
    utils::check(rspace_tol > 0.0,
                 "cvv_head_t: cvv_rspace_tol must be > 0 (got {}).", rspace_tol);
  }

  void cvv_head_t::not_implemented(std::string_view where) const {
    utils::check(false,
                 "{}: the CVV head evaluator is scaffolding only (increment C0); the "
                 "R-space engine and head tensor land in increments C1/C2 of "
                 "notes/scgwt_implementation_plan.md.", where);
    std::abort();  // unreachable: utils::check(false, ...) aborts
  }

  void cvv_head_t::build_rspace_sigma() { not_implemented("cvv_head_t::build_rspace_sigma"); }
  void cvv_head_t::eval_head_tensor()   { not_implemented("cvv_head_t::eval_head_tensor"); }

} // solvers
} // methods
