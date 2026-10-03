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
 *  - auto_t_chunk: the time-chunk length (device: from the free device memory; host: a fixed default).
 *  - slab_conv: the fused Hadamard kernel of cuda/gw_line_cuda.cuh (S7d) on MEM arrays, and the runtime switches
 *    COQUI_GWLINE_FUSED (default 1: fused kernel on the device; 0: the cuTENSOR trinary of the bring-up).
 */

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "utilities/device_pool.h"
#if defined(ENABLE_CUDA)
#include "methods/GW_line/cuda/gw_line_cuda.cuh"
#endif

namespace methods::gw_line::detail {

inline void gemm_strided_cm([[maybe_unused]] char opa, [[maybe_unused]] char opb, [[maybe_unused]] long m,
                            [[maybe_unused]] long n, [[maybe_unused]] long k, [[maybe_unused]] ComplexType alpha,
                            [[maybe_unused]] ComplexType const *A, [[maybe_unused]] long lda, [[maybe_unused]] long sA,
                            [[maybe_unused]] ComplexType const *B, [[maybe_unused]] long ldb, [[maybe_unused]] long sB,
                            [[maybe_unused]] ComplexType beta, [[maybe_unused]] ComplexType *C, [[maybe_unused]] long ldc,
                            [[maybe_unused]] long sC, [[maybe_unused]] long batch) {
#if defined(NDA_HAVE_DEVICE)
  if (batch <= 0) return;
  // nda's device interface takes int dimensions and strides
  utils::check(std::max({m, n, k, lda, ldb, ldc, sA, sB, sC, batch}) <= long(INT_MAX),
               "gw_line::gemm_strided_cm: dimension or stride beyond the int range (sA {} sB {} sC {})", sA, sB, sC);
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

/// integer environment switch, read at every call (tests may change it between kernel calls); def when unset or empty
inline long env_long(char const *nm, long def) {
  char const *v = std::getenv(nm);
  return (v != nullptr and *v != '\0') ? std::strtol(v, nullptr, 10) : def;
}

/// device fused Hadamard kernels (COQUI_GWLINE_FUSED, default 1); always false in builds without CUDA
inline bool fused_hadamard() {
#if defined(ENABLE_CUDA)
  return env_long("COQUI_GWLINE_FUSED", 1) != 0;
#else
  return false;
#endif
}

/**
 * The fused Hadamard ("slab convolution", cuda/gw_line_cuda.cuh) on MEM arrays (DEVICE builds with CUDA only):
 *   O[o][e] (+)= alpha sum_{j < nterm} X[ix][e] Y[iy][e],  (ix, iy) = pairs[off + 2 (o nterm + j) + {0, 1}],  e < E.
 * X, Y, O: raw MEM pointers of slab arrays (slab strides sX, sY, sO in elements); pairs: MEM int table.
 */
template <MEMORY_SPACE MEM>
void slab_conv([[maybe_unused]] long E, [[maybe_unused]] long nX, [[maybe_unused]] ComplexType const *X,
               [[maybe_unused]] long sX, [[maybe_unused]] long nY, [[maybe_unused]] ComplexType const *Y,
               [[maybe_unused]] long sY, [[maybe_unused]] long nO, [[maybe_unused]] ComplexType *O, [[maybe_unused]] long sO,
               [[maybe_unused]] long nterm, [[maybe_unused]] memory::array<MEM, int, 1> const &pairs,
               [[maybe_unused]] long off, [[maybe_unused]] ComplexType alpha, [[maybe_unused]] bool accumulate) {
#if defined(ENABLE_CUDA)
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(off >= 0 and off + 2 * nO * nterm <= pairs.size(), "gw_line::slab_conv: pair table too short");
    cuda::slab_conv_t a;
    a.E = E;
    a.nX = int(nX); a.nY = int(nY); a.nO = int(nO); a.nterm = int(nterm);
    a.X = X; a.sX = sX; a.Y = Y; a.sY = sY; a.O = O; a.sO = sO;
    a.pairs = pairs.data() + off;
    a.alpha = alpha;
    a.accumulate = accumulate;
    cuda::slab_conv(a, int(env_long("COQUI_GWLINE_FUSED_VARIANT", 0)));
    return;
  }
#endif
  utils::check(false, "gw_line::slab_conv: device build with CUDA required");
}

/**
 * Time-chunk length: device: the largest chunk whose per-chunk arrays (bytes_per_t each) fit in `frac` of the effective
 * free device memory, clamped to [8, min(nt, tmax)]; host: host_t_chunk (S7d: 32, see notes/line_gw_device_bringup.md
 * S7d; the S3-S5 setting was 8; env COQUI_GWLINE_HOST_TCHUNK overrides).
 */
inline constexpr long host_t_chunk_default = 32;
template <MEMORY_SPACE MEM>
long auto_t_chunk(long nt, double bytes_per_t, double frac = 0.4, long tmax = 256) {
  long tc = env_long("COQUI_GWLINE_HOST_TCHUNK", host_t_chunk_default);
  if constexpr (MEM != HOST_MEMORY) {
    const double freeb = double(utils::freemem_device_effective()) * 1048576.0;   // MB -> bytes
    tc = 8;
    if (bytes_per_t > 0.0) tc = std::max(8L, std::min(tmax, long(frac * freeb / bytes_per_t)));
  }
  return std::max(1L, std::min(tc, nt));
}

} // namespace methods::gw_line::detail

#endif
