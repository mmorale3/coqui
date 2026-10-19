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


#ifndef GW_TOOLS_CUDA_KERNELS_HPP
#define GW_TOOLS_CUDA_KERNELS_HPP

#include <complex>

namespace kernels::device
{

/**
 * The k-space GW self-energy Hadamard stage in ONE launch per tau slice (gw_t::eval_Sigma_all_kspace):
 *   Sigma(isk, e) += sum over the terms t of isk of  f_mode(G(ikmq_t, e), W(iqs_t, e)) * factor
 * with e over the Np x Nq tile and the four symmetry modes of the host path:
 *   0: conj(factor W G), 1: factor G conj(W), 2: factor W conj(G), 3: factor G W.
 * Every output element accumulates its terms in the list order on one thread (no atomics: deterministic).
 * Sigma: ns nk_ibz tiles at stride tile; G: tiles at stride tile, indexed by (is nk_full + ikmq) -> g_of[t] holds that
 * flat tile index; W: the tau slice's tiles at stride wq, indexed by iqs. term_first[isk] .. term_first[isk + 1] delimit the
 * terms of output tile isk (nsk + 1 entries). All pointers are DEVICE pointers; the term tables are host arrays (uploaded here).
 */
void sigma_hadamard_terms(std::complex<double> *Sigma, std::complex<double> const *G, std::complex<double> const *W,
                          long nsk, long tile, long wq, long nterm, long const *term_first, long const *g_of, long const *w_of,
                          int const *mode, std::complex<double> factor);

} // kernels::device

#endif
