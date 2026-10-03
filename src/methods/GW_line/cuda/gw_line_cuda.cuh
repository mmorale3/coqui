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


#ifndef COQUI_METHODS_GW_LINE_CUDA_GW_LINE_CUDA_CUH
#define COQUI_METHODS_GW_LINE_CUDA_GW_LINE_CUDA_CUH

/**
 * Fused Hadamard ("slab convolution") kernel of the GW_line ray products (S7d, notes/line_gw_device_bringup.md):
 *
 *   O[o][e] (+)= alpha * sum_{j < nterm} X[ix(o, j)][e] * Y[iy(o, j)][e]        for all o < nO, e < E,
 *
 * i.e. many elementwise products of (P,Q) blocks summed in registers in ONE launch, with the index pairs (ix, iy) of an
 * output given by a small table. The two uses (polarization.hpp, self_energy.hpp):
 *   Pi    : X = A(k, chunk), Y = B(k, chunk), O = acc(q, chunk), pairs(q, j = k) = (k, k-q):   acc(q) = alpha sum_k A(k) o B(k-q)
 *   Sigma : X = G~(k, chunk), Y = W(q, chunk), O = acc(k, chunk), pairs(k, j = q) = (k-q, q): acc(k) = sum_q G~(k-q) o W(q)
 * A "slab" is the [t_chunk x nP x nQ] array of one k (or q); the first E = n_t * nP * nQ elements are used (strides sX, sY,
 * sO between slabs, in elements). Every input slab is read once from global memory and every output written once
 * (staged variant), against 4 block passes per product of the cuTENSOR trinary (read A, B, acc, write acc).
 *
 * Plain C++ header (POD of raw DEVICE pointers, no CUDA types), the pattern of methods/vertex/cuda/l0_cuda.cuh; the .cu is
 * compiled by nvcc into gw_line_cuda (ENABLE_CUDA builds only). Launches on the default stream (ordered with nda's
 * cuBLAS / cuTENSOR calls); asynchronous (the caller synchronizes for its timers). Aborts (APP_ABORT) on a CUDA error.
 */

#include <complex>

namespace methods::gw_line::cuda {

using cplx = std::complex<double>;

struct slab_conv_t {
  long E = 0;                              ///< elements per slab used
  int nX = 0, nY = 0, nO = 0, nterm = 0;   ///< slab counts of X, Y, O; products summed per output
  cplx const *X = nullptr;                 ///< DEVICE, slab i at X + i * sX
  long sX = 0;
  cplx const *Y = nullptr;                 ///< DEVICE, slab i at Y + i * sY
  long sY = 0;
  cplx *O = nullptr;                       ///< DEVICE, slab o at O + o * sO
  long sO = 0;
  int const *pairs = nullptr;              ///< DEVICE (nO, nterm, 2): (ix, iy) of term j of output o
  cplx alpha = cplx(1.0, 0.0);
  bool accumulate = false;                 ///< false: O = alpha sum, true: O += alpha sum
};

/**
 * variant: 0 automatic, 1 direct (one thread per element, all outputs, operands read from global memory per term),
 *          2 staged (the nX + nY slabs of a tile of elements staged in shared memory, outputs spread over threadIdx.y).
 * Automatic: staged when nO > 1 and nterm > 1 and the tile fits the shared memory, else direct. Returns the variant used.
 */
int slab_conv(slab_conv_t const &a, int variant = 0);

} // namespace methods::gw_line::cuda

#endif
