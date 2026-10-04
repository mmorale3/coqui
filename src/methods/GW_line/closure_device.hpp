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

#ifndef COQUI_METHODS_GW_LINE_CLOSURE_DEVICE_HPP
#define COQUI_METHODS_GW_LINE_CLOSURE_DEVICE_HPP

/**
 * S7g: the cuSOLVER drivers of cuda/gw_line_lapack.cuh as numerics::line_dlr::lapack_hooks_t (Gram eigen, SVD of
 * D+ D-^dagger, LU solve and Hermitian eigen of the "cayley" U-eigen path, Lehmann eigen). CUDA builds only; elsewhere
 * device_lapack_hooks() returns null (host LAPACK). Each hook falls back to the host on any device failure.
 * svd_variant: 0 = zgesvd, 1 = Xgesvdp (polar-decomposition SVD).
 */

#include "configuration.hpp"
#include "numerics/line_dlr/cayley.hpp"
#if defined(ENABLE_CUDA)
#include "methods/GW_line/cuda/gw_line_lapack.cuh"
#endif

namespace methods::gw_line {

inline numerics::line_dlr::lapack_hooks_t const *device_lapack_hooks([[maybe_unused]] int svd_variant = 0) {
#if defined(ENABLE_CUDA)
  using numerics::line_dlr::cmatrix_F;
  auto make = [](int var) {
    numerics::line_dlr::lapack_hooks_t h;
    h.heevd = [](cmatrix_F &A, nda::array<double, 1> &w) {
      w.resize(A.extent(0));
      return cuda::dev_heevd(int(A.extent(0)), A.data(), w.data());
    };
    h.gesvd = [var](cmatrix_F &A, nda::array<double, 1> &s, cmatrix_F &P, cmatrix_F &Qh) {
      const long n = A.extent(0);
      if (A.extent(1) != n) return false;
      s.resize(n);
      P.resize(n, n);
      Qh.resize(n, n);
      return cuda::dev_gesvd(int(n), A.data(), s.data(), P.data(), Qh.data(), var);
    };
    h.lu_solve = [](cmatrix_F &A, cmatrix_F &B) {
      return cuda::dev_lu_solve(int(A.extent(0)), int(B.extent(1)), A.data(), B.data());
    };
    return h;
  };
  static const numerics::line_dlr::lapack_hooks_t h0 = make(0), h1 = make(1);
  return svd_variant == 1 ? &h1 : &h0;
#else
  return nullptr;
#endif
}

/// device calls that fell back to the host so far (0 without CUDA)
inline long device_lapack_failures() {
#if defined(ENABLE_CUDA)
  return cuda::dev_lapack_failures();
#else
  return 0;
#endif
}

} // namespace methods::gw_line

#endif
