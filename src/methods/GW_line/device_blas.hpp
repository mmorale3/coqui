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

#ifndef COQUI_METHODS_GW_LINE_DEVICE_BLAS_HPP
#define COQUI_METHODS_GW_LINE_DEVICE_BLAS_HPP

/**
 * Device-only helpers of the GW_line kernels (notes/line_gw_device_bringup.md).
 *
 *  - gemm_strided_cm: strided-batched gemm on COLUMN-MAJOR raw device pointers (cuBLAS ZgemmStridedBatched through
 *    nda::blas::device). A stride of 0 broadcasts that operand over the batch, which is what the per-time-node products
 *    X C(t) X^dagger and X^dagger acc(t) X need (the X slice is the same for every t). nda's gemm_batch_strided takes 3D
 *    arrays and cannot express a zero stride, hence the raw call. Row-major (C-layout) operands enter as their transposes:
 *    a C-layout r x c matrix IS the column-major c x r matrix with ld = c.
 *  - scratch_t: a grow-only MEM buffer handing out views of any shape (preallocated scratch instead of per-call
 *    allocations in the hot loops).
 *  - device_t_chunk: the time-chunk length on the device from the free device memory.
 */

#include <algorithm>
#include <array>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "utilities/device_pool.h"

namespace methods::gw_line::detail {

inline void gemm_strided_cm([[maybe_unused]] char opa, [[maybe_unused]] char opb, [[maybe_unused]] long m,
                            [[maybe_unused]] long n, [[maybe_unused]] long k, [[maybe_unused]] ComplexType alpha,
                            [[maybe_unused]] ComplexType const *A, [[maybe_unused]] long lda, [[maybe_unused]] long sA,
                            [[maybe_unused]] ComplexType const *B, [[maybe_unused]] long ldb, [[maybe_unused]] long sB,
                            [[maybe_unused]] ComplexType beta, [[maybe_unused]] ComplexType *C, [[maybe_unused]] long ldc,
                            [[maybe_unused]] long sC, [[maybe_unused]] long batch) {
#if defined(NDA_HAVE_DEVICE)
  if (batch <= 0) return;
  nda::blas::device::gemm_batch_strided(opa, opb, int(m), int(n), int(k), alpha, A, int(lda), int(sA), B, int(ldb), int(sB),
                                        beta, C, int(ldc), int(sC), int(batch));
#else
  utils::check(false, "gw_line::gemm_strided_cm: device build required");
#endif
}

/// Grow-only scratch buffer in MEM; view<R>(shape) is a contiguous C-layout view on its first prod(shape) elements.
template <MEMORY_SPACE MEM>
struct scratch_t {
  memory::array<MEM, ComplexType, 1> buf;
  template <int R>
  memory::array_view<MEM, ComplexType, R> view(std::array<long, R> const &shape) {
    long n = 1;
    for (auto s : shape) n *= s;
    if (buf.size() < n) buf = memory::array<MEM, ComplexType, 1>(n);
    return memory::array_view<MEM, ComplexType, R>(shape, buf.data());
  }
  double bytes() const { return double(buf.size()) * 16.0; }
};

/**
 * Time-chunk length on the device: the largest chunk whose per-chunk arrays (bytes_per_t each) fit in `frac` of the
 * effective free device memory, clamped to [8, min(nt, tmax)]. Host: 8 (the measured CPU optimum of S3-S5).
 */
template <MEMORY_SPACE MEM>
long auto_t_chunk(long nt, double bytes_per_t, double frac = 0.4, long tmax = 256) {
  long tc = 8;
  if constexpr (MEM != HOST_MEMORY) {
    const double freeb = double(utils::freemem_device_effective()) * 1048576.0;   // MB -> bytes
    if (bytes_per_t > 0.0) tc = std::max(8L, std::min(tmax, long(frac * freeb / bytes_per_t)));
  }
  return std::max(1L, std::min(tc, nt));
}

} // namespace methods::gw_line::detail

#endif
