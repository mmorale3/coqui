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

#ifndef COQUI_METHODS_GW_LINE_CUDA_GW_LINE_LAPACK_CUH
#define COQUI_METHODS_GW_LINE_CUDA_GW_LINE_LAPACK_CUH

/**
 * S7g: cuSOLVER drivers for the dense problems of the closure (numerics/line_dlr/cayley.hpp lapack_hooks_t): Hermitian
 * eigen-decomposition (zheevd), SVD of a square matrix (zgesvd or the polar-decomposition SVD Xgesvdp) and the LU solve
 * (zgetrf / zgetrs). HOST column-major arrays in and out (one H2D / D2H copy per call, ~30 MB at the si222c size); a
 * per-thread cuSOLVER handle and a growing device workspace are kept between calls. Every function returns false -- with
 * its outputs untouched -- on any CUDA / cuSOLVER error, allocation failure or nonzero info, so the caller falls back to
 * the host LAPACK. Plain C++ header (no CUDA types), as gw_line_cuda.cuh.
 */

#include <complex>

namespace methods::gw_line::cuda {

using cplx = std::complex<double>;

/// A (n x n, lda = n) <- eigenvectors, w <- eigenvalues ascending (lower triangle of A referenced).
bool dev_heevd(int n, cplx *A, double *w);

/// A (n x n) = P diag(s) Qh, all vectors; variant 0 = zgesvd (QR iteration), 1 = Xgesvdp (polar decomposition). A untouched.
bool dev_gesvd(int n, cplx const *A, double *s, cplx *P, cplx *Qh, int variant = 0);

/// B (n x nrhs) <- A^{-1} B (A n x n untouched).
bool dev_lu_solve(int n, int nrhs, cplx const *A, cplx *B);

/// Number of calls of this process that failed on the device and returned false (host fallback).
long dev_lapack_failures();

/// Release the workspace and handle of the calling thread (optional; also released at exit).
void dev_lapack_release();

} // namespace methods::gw_line::cuda

#endif
