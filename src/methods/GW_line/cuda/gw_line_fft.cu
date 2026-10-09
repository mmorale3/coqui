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

// perf 7.5c: cuFFT plans of the k mesh (gw_line_fft.cuh). Errors abort through APP_ABORT (gw_line_cuda.cu pattern).

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include "IO/AppAbort.hpp"
#include "methods/GW_line/cuda/gw_line_fft.cuh"

namespace methods::gw_line::cuda {

namespace {

void check_cufft(cufftResult r, char const *what) {
  if (r != CUFFT_SUCCESS) APP_ABORT(std::string(" gw_line_fft: cuFFT error ") + std::to_string(int(r)) + " in " + what);
}
void check_cuda(cudaError_t e, char const *what) {
  if (e != cudaSuccess)
    APP_ABORT(std::string(" gw_line_fft: CUDA error in ") + what + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
}

struct plan_cache_t {
  std::map<std::array<long, 5>, cufftHandle> plans;   // (n1, n2, n3, batch, ld)
  void *work      = nullptr;
  size_t work_cap = 0;
  ~plan_cache_t() { release(); }
  void release() {
    for (auto &p : plans) cufftDestroy(p.second);
    plans.clear();
    if (work != nullptr) cudaFree(work);
    work     = nullptr;
    work_cap = 0;
  }
  cufftHandle get(int n1, int n2, int n3, long batch, long ld) {
    std::array<long, 5> key = {n1, n2, n3, batch, ld};
    auto it                 = plans.find(key);
    if (it != plans.end()) return it->second;
    cufftHandle h;
    check_cufft(cufftCreate(&h), "cufftCreate");
    check_cufft(cufftSetAutoAllocation(h, 0), "cufftSetAutoAllocation");
    long long n[3] = {n1, n2, n3};
    size_t ws      = 0;   // 64-bit planner: the element offsets N ld exceed the int range for the residue transforms
    check_cufft(cufftMakePlanMany64(h, 3, n, n, (long long)ld, 1, n, (long long)ld, 1, CUFFT_Z2Z, (long long)batch, &ws),
                "cufftMakePlanMany64");
    if (ws > work_cap) {
      if (work != nullptr) check_cuda(cudaFree(work), "cudaFree(work)");
      check_cuda(cudaMalloc(&work, ws), "cudaMalloc(work)");
      work_cap = ws;
    }
    plans[key] = h;
    return h;
  }
};

plan_cache_t &cache() {
  static plan_cache_t c;
  return c;
}

} // namespace

void fft_mesh(int n1, int n2, int n3, std::complex<double> const *in, std::complex<double> *out, long ncols, long ld, int sign) {
  if (ncols <= 0) return;
  const long N = long(n1) * n2 * n3;
  long cb = (1L << 24) / std::max(1L, N);
  if (char const *v = std::getenv("COQUI_GWLINE_FFT_DEV_CB"); v != nullptr and *v != '\0') cb = std::strtol(v, nullptr, 10);
  cb = std::max(1L, std::min(cb, ncols));
  auto &C = cache();
  for (long c0 = 0; c0 < ncols; c0 += cb) {
    const long w  = std::min(cb, ncols - c0);
    cufftHandle h = C.get(n1, n2, n3, w, ld);
    check_cufft(cufftSetWorkArea(h, C.work), "cufftSetWorkArea");
    auto *i = reinterpret_cast<cufftDoubleComplex *>(const_cast<std::complex<double> *>(in + c0));
    auto *o = reinterpret_cast<cufftDoubleComplex *>(out + c0);
    check_cufft(cufftExecZ2Z(h, i, o, sign > 0 ? CUFFT_INVERSE : CUFFT_FORWARD), "cufftExecZ2Z");
  }
}

void fft_mesh_release() { cache().release(); }

} // namespace methods::gw_line::cuda
