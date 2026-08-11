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

#ifndef COQUI_CVV_HEAD_HPP
#define COQUI_CVV_HEAD_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace methods {
namespace solvers {

  /**
   * scGW-tilde CVV head ("conserving velocity vertex"; notes/scgwt_implementation_plan.md
   * increments C1-C4, theory notes/scgw_screening_fix_proposal.pdf section 4.1).
   *
   * Evaluates the q -> 0 head of the polarization as the covariant-velocity O(q^2)
   * coefficient
   *
   *   Pi_ab(inu) = -(2 / (beta N_k V)) sum_{k,iw} tr[ v~_a G v~_b G ],
   *   v~ = d_k (H0 + F + Sigma(k,iw)),
   *
   * replacing the gygi/stored q -> 0 EXTRAPOLATION of eps_inv_head under the
   * div_treatment = "cvv" policy. The Sigma term of v~ is built by the R-space engine
   * (increment C1): unfold Sigma from the IBZ, k -> R by one gemm against
   * utils::k_to_R_coefficients, truncate R-shells at rspace_tol, and differentiate
   * analytically as sum_R iR_a e^{ikR} Sigma(R, iw).
   *
   * INCREMENT C0: scaffolding only -- knob storage and the class surface. Every
   * evaluator entry aborts until C1 (R-space engine + v~) and C2 (head tensor +
   * telescoping identity tests) land.
   */
  class cvv_head_t {
  public:
    cvv_head_t(const imag_axes_ft::IAFT *ft, double rspace_tol = 1e-6);

    cvv_head_t(cvv_head_t const&) = default;
    cvv_head_t(cvv_head_t &&) = default;
    cvv_head_t& operator=(cvv_head_t const&) = default;
    cvv_head_t& operator=(cvv_head_t &&) = default;
    ~cvv_head_t() {}

    // R-shell truncation tolerance of the Sigma(R, iw) store ([gw] cvv_rspace_tol)
    double rspace_tol() const { return _rspace_tol; }

    // ---- increment C1: R-space engine + covariant velocity (aborts until it lands) ----
    void build_rspace_sigma();
    // ---- increment C2: head tensor Pi_ab(inu) (aborts until it lands) ----
    void eval_head_tensor();

  private:
    [[noreturn]] void not_implemented(std::string_view where) const;

    // consumed from increment C1 (tau -> DLR-omega transforms of the R-space Sigma)
    [[maybe_unused]] const imag_axes_ft::IAFT* _ft = nullptr;
    double _rspace_tol = 1e-6;
  };

} // solvers
} // methods

#endif // COQUI_CVV_HEAD_HPP
