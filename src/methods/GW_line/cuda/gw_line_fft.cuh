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

#ifndef COQUI_METHODS_GW_LINE_CUDA_GW_LINE_FFT_CUH
#define COQUI_METHODS_GW_LINE_CUDA_GW_LINE_FFT_CUH

/**
 * perf 7.5c: batched 3-D FFTs of the k mesh on the device (cuFFT), for the real-space convolutions (kmesh_fft.hpp).
 * The array is row-major (n1 n2 n3 rows, ld columns) in DEVICE memory; the columns [0, ncols) are transformed:
 *   out[m][c] = sum_j in[j][c] exp(sign 2 pi i sum_d j_d m_d / n_d)      (sign +1: CUFFT_INVERSE, -1: CUFFT_FORWARD)
 * with j, m the row-major mesh indices. In place when in == out. Plain C++ header (POD arguments), compiled into
 * gw_line_cuda (ENABLE_CUDA builds only). Z2Z plans with the advanced layout (istride = ld, idist = 1, batch = columns),
 * executed in column blocks (env COQUI_GWLINE_FFT_DEV_CB, default 2^24 / N columns) with one shared work area; plans are
 * cached per (mesh, block width, ld). Default stream (ordered with nda's cuBLAS / cuTENSOR calls); asynchronous.
 */

#include <complex>

namespace methods::gw_line::cuda {

void fft_mesh(int n1, int n2, int n3, std::complex<double> const *in, std::complex<double> *out, long ncols, long ld, int sign);

/// release the cached plans and the work area (tests)
void fft_mesh_release();

} // namespace methods::gw_line::cuda

#endif
