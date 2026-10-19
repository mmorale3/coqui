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


#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include <vector>
#include <cuComplex.h>
#include "numerics/device_kernels/cuda/gw_tools.cuh"

namespace kernels::device
{
namespace {
  __global__ void sigma_hadamard_kernel(cuDoubleComplex *__restrict__ S, cuDoubleComplex const *__restrict__ G, cuDoubleComplex const *__restrict__ W,
                                        long nsk, long tile, long wq, long const *__restrict__ first, long const *__restrict__ g_of,
                                        long const *__restrict__ w_of, int const *__restrict__ mode, cuDoubleComplex factor) {
    const long tot = nsk * tile;
    for (long x = blockIdx.x * long(blockDim.x) + threadIdx.x; x < tot; x += long(gridDim.x) * blockDim.x) {
      const long isk = x / tile, e = x - isk * tile;
      cuDoubleComplex acc = S[x];
      for (long t = first[isk]; t < first[isk + 1]; ++t) {
        cuDoubleComplex g = G[g_of[t] * tile + e], w = W[w_of[t] * wq + e], p;
        switch (mode[t]) {
          case 0: p = cuCmul(cuCmul(w, g), factor); p = cuConj(p); break;               // conj(factor W G)
          case 1: p = cuCmul(cuCmul(g, cuConj(w)), factor); break;                      // factor G conj(W)
          case 2: p = cuCmul(cuCmul(w, cuConj(g)), factor); break;                      // factor W conj(G)
          default: p = cuCmul(cuCmul(g, w), factor); break;                             // factor G W
        }
        acc = cuCadd(acc, p);
      }
      S[x] = acc;
    }
  }
}

void sigma_hadamard_terms(std::complex<double> *Sigma, std::complex<double> const *G, std::complex<double> const *W,
                          long nsk, long tile, long wq, long nterm, long const *term_first, long const *g_of, long const *w_of,
                          int const *mode, std::complex<double> factor) {
  if (nsk <= 0 or tile <= 0) return;
  long *d_first = nullptr, *d_g = nullptr, *d_w = nullptr;
  int *d_mode = nullptr;
  check_launch(cudaMalloc(&d_first, size_t(nsk + 1) * sizeof(long)), "sigma_hadamard alloc");
  check_launch(cudaMalloc(&d_g, size_t(std::max(nterm, 1l)) * sizeof(long)), "sigma_hadamard alloc");
  check_launch(cudaMalloc(&d_w, size_t(std::max(nterm, 1l)) * sizeof(long)), "sigma_hadamard alloc");
  check_launch(cudaMalloc(&d_mode, size_t(std::max(nterm, 1l)) * sizeof(int)), "sigma_hadamard alloc");
  check_launch(cudaMemcpy(d_first, term_first, size_t(nsk + 1) * sizeof(long), cudaMemcpyHostToDevice), "sigma_hadamard h2d");
  if (nterm > 0) {
    check_launch(cudaMemcpy(d_g, g_of, size_t(nterm) * sizeof(long), cudaMemcpyHostToDevice), "sigma_hadamard h2d");
    check_launch(cudaMemcpy(d_w, w_of, size_t(nterm) * sizeof(long), cudaMemcpyHostToDevice), "sigma_hadamard h2d");
    check_launch(cudaMemcpy(d_mode, mode, size_t(nterm) * sizeof(int), cudaMemcpyHostToDevice), "sigma_hadamard h2d");
  }
  const long tot = nsk * tile;
  const unsigned grid = unsigned(std::min<long>((tot + 255) / 256, 65535));
  sigma_hadamard_kernel<<<grid, 256>>>(reinterpret_cast<cuDoubleComplex *>(Sigma), reinterpret_cast<cuDoubleComplex const *>(G),
                                       reinterpret_cast<cuDoubleComplex const *>(W), nsk, tile, wq, d_first, d_g, d_w, d_mode,
                                       make_cuDoubleComplex(factor.real(), factor.imag()));
  check_launch(cudaGetLastError(), "sigma_hadamard launch");
  check_launch(cudaDeviceSynchronize(), "sigma_hadamard sync");
  cudaFree(d_first); cudaFree(d_g); cudaFree(d_w); cudaFree(d_mode);
}

} // kernels::device
