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


#ifndef COQUI_QP_CONTEXT_H
#define COQUI_QP_CONTEXT_H

namespace methods {

struct qp_context_t {
  std::string qp_type = "sc";
  std::string ac_alg = "pade";
  int Nfit = 18;
  double eta = 0.0001;
  double tol = 1e-8;

  // off-diagonal mode defined in T. Kotani et. al., Phys. Rev. B 76, 165106 (2007)
  std::string off_diag_mode = "fermi";
};

} // methods

#endif //COQUI_QP_CONTEXT_H
