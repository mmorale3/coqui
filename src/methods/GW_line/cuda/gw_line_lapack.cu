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

// S7g: cuSOLVER drivers of the closure (see gw_line_lapack.cuh). Errors never abort: they return false (host fallback).

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "methods/GW_line/cuda/gw_line_lapack.cuh"

namespace methods::gw_line::cuda {

namespace {

using cd = cuDoubleComplex;

/// per-thread handle + one growing device buffer
struct ws_t {
  cusolverDnHandle_t h = nullptr;
  void *buf            = nullptr;
  size_t bytes         = 0;
  int dev              = -1;
  ~ws_t() { release(); }
  void release() {
    if (buf) cudaFree(buf);
    if (h) cusolverDnDestroy(h);
    buf   = nullptr;
    h     = nullptr;
    bytes = 0;
    dev   = -1;
  }
  bool init() {
    int cur = 0;
    if (cudaGetDevice(&cur) != cudaSuccess) return false;
    if (h and cur == dev) return true;
    release();
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) { h = nullptr; return false; }
    dev = cur;
    return true;
  }
  /// device buffer of at least nb bytes (contents not preserved)
  void *get(size_t nb) {
    if (nb <= bytes) return buf;
    if (buf) cudaFree(buf);
    buf   = nullptr;
    bytes = 0;
    if (cudaMalloc(&buf, nb) != cudaSuccess) {
      cudaGetLastError();   // clear the sticky-free error state of a failed allocation
      buf = nullptr;
      return nullptr;
    }
    bytes = nb;
    return buf;
  }
};

long g_failures = 0;   // calls that returned false (host fallback), all threads of the process

ws_t &ws() {
  static thread_local ws_t w;
  return w;
}

size_t align(size_t x) { return (x + 255) & ~size_t(255); }

bool ok(cudaError_t e) { return e == cudaSuccess; }
bool ok(cusolverStatus_t s) { return s == CUSOLVER_STATUS_SUCCESS; }

} // namespace

static bool impl_heevd(int n, cplx *A, double *w) {
  if (n <= 0) return true;
  auto &W = ws();
  if (not W.init()) return false;
  int lwork = 0;
  const size_t nA = size_t(n) * n;
  if (not ok(cusolverDnZheevd_bufferSize(W.h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, nullptr, n, nullptr, &lwork)))
    return false;
  const size_t oA = 0, oW = align(oA + nA * sizeof(cd)), oWk = align(oW + size_t(n) * sizeof(double)),
               oI = align(oWk + size_t(lwork) * sizeof(cd)), tot = oI + sizeof(int);
  auto *base = static_cast<char *>(W.get(tot));
  if (not base) return false;
  auto *dA = reinterpret_cast<cd *>(base + oA);
  auto *dW = reinterpret_cast<double *>(base + oW);
  auto *dK = reinterpret_cast<cd *>(base + oWk);
  auto *dI = reinterpret_cast<int *>(base + oI);
  if (not ok(cudaMemcpy(dA, A, nA * sizeof(cd), cudaMemcpyHostToDevice))) return false;
  if (not ok(cusolverDnZheevd(W.h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, dK, lwork, dI))) return false;
  int info = -1;
  if (not ok(cudaMemcpy(&info, dI, sizeof(int), cudaMemcpyDeviceToHost)) or info != 0) return false;
  std::vector<double> wh(n);
  if (not ok(cudaMemcpy(wh.data(), dW, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost))) return false;
  if (not ok(cudaMemcpy(A, dA, nA * sizeof(cd), cudaMemcpyDeviceToHost))) return false;   // last: A changes only on success
  std::copy(wh.begin(), wh.end(), w);
  return true;
}

static bool impl_gesvd(int n, cplx const *A, double *s, cplx *P, cplx *Qh, int variant) {
  if (n <= 0) return true;
  auto &W = ws();
  if (not W.init()) return false;
  const size_t nA = size_t(n) * n;
  if (variant == 1) {   // polar-decomposition SVD (returns V, not V^dagger)
    cusolverDnParams_t prm = nullptr;
    if (not ok(cusolverDnCreateParams(&prm))) return false;
    size_t wdev = 0, whost = 0;
    bool good = ok(cusolverDnXgesvdp_bufferSize(W.h, prm, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, CUDA_C_64F, nullptr, n, CUDA_R_64F,
                                                nullptr, CUDA_C_64F, nullptr, n, CUDA_C_64F, nullptr, n, CUDA_C_64F, &wdev, &whost));
    const size_t oA = 0, oS = align(nA * sizeof(cd)), oU = align(oS + size_t(n) * sizeof(double)), oV = align(oU + nA * sizeof(cd)),
                 oWk = align(oV + nA * sizeof(cd)), oI = align(oWk + wdev), tot = oI + sizeof(int);
    char *base = good ? static_cast<char *>(W.get(tot)) : nullptr;
    if (not base) { cusolverDnDestroyParams(prm); return false; }
    auto *dA = reinterpret_cast<cd *>(base + oA);
    auto *dS = reinterpret_cast<double *>(base + oS);
    auto *dU = reinterpret_cast<cd *>(base + oU);
    auto *dV = reinterpret_cast<cd *>(base + oV);
    auto *dI = reinterpret_cast<int *>(base + oI);
    std::vector<char> hw(whost > 0 ? whost : 1);
    double err_sigma = 0.0;
    good = ok(cudaMemcpy(dA, A, nA * sizeof(cd), cudaMemcpyHostToDevice)) and
           ok(cusolverDnXgesvdp(W.h, prm, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, CUDA_C_64F, dA, n, CUDA_R_64F, dS, CUDA_C_64F, dU, n,
                                CUDA_C_64F, dV, n, CUDA_C_64F, base + oWk, wdev, hw.data(), whost, dI, &err_sigma));
    cusolverDnDestroyParams(prm);
    int info = -1;
    if (not good or not ok(cudaMemcpy(&info, dI, sizeof(int), cudaMemcpyDeviceToHost)) or info != 0) return false;
    std::vector<cplx> V(nA);
    if (not ok(cudaMemcpy(P, dU, nA * sizeof(cd), cudaMemcpyDeviceToHost)) or
        not ok(cudaMemcpy(V.data(), dV, nA * sizeof(cd), cudaMemcpyDeviceToHost)) or
        not ok(cudaMemcpy(s, dS, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost)))
      return false;
    for (int j = 0; j < n; ++j)   // Qh = V^dagger (column-major)
      for (int i = 0; i < n; ++i) Qh[i + size_t(j) * n] = std::conj(V[j + size_t(i) * n]);
    return true;
  }
  int lwork = 0;
  if (not ok(cusolverDnZgesvd_bufferSize(W.h, n, n, &lwork))) return false;
  const size_t oA = 0, oS = align(nA * sizeof(cd)), oU = align(oS + size_t(n) * sizeof(double)), oV = align(oU + nA * sizeof(cd)),
               oWk = align(oV + nA * sizeof(cd)), oR = align(oWk + size_t(lwork) * sizeof(cd)),
               oI = align(oR + size_t(5 * n) * sizeof(double)), tot = oI + sizeof(int);
  auto *base = static_cast<char *>(W.get(tot));
  if (not base) return false;
  auto *dA = reinterpret_cast<cd *>(base + oA);
  auto *dS = reinterpret_cast<double *>(base + oS);
  auto *dU = reinterpret_cast<cd *>(base + oU);
  auto *dV = reinterpret_cast<cd *>(base + oV);
  auto *dK = reinterpret_cast<cd *>(base + oWk);
  auto *dR = reinterpret_cast<double *>(base + oR);
  auto *dI = reinterpret_cast<int *>(base + oI);
  if (not ok(cudaMemcpy(dA, A, nA * sizeof(cd), cudaMemcpyHostToDevice))) return false;
  if (not ok(cusolverDnZgesvd(W.h, 'A', 'A', n, n, dA, n, dS, dU, n, dV, n, dK, lwork, dR, dI))) return false;
  int info = -1;
  if (not ok(cudaMemcpy(&info, dI, sizeof(int), cudaMemcpyDeviceToHost)) or info != 0) return false;
  return ok(cudaMemcpy(P, dU, nA * sizeof(cd), cudaMemcpyDeviceToHost)) and
         ok(cudaMemcpy(Qh, dV, nA * sizeof(cd), cudaMemcpyDeviceToHost)) and
         ok(cudaMemcpy(s, dS, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
}

static bool impl_lu_solve(int n, int nrhs, cplx const *A, cplx *B) {
  if (n <= 0 or nrhs <= 0) return true;
  auto &W = ws();
  if (not W.init()) return false;
  int lwork = 0;
  if (not ok(cusolverDnZgetrf_bufferSize(W.h, n, n, nullptr, n, &lwork))) return false;
  const size_t nA = size_t(n) * n, nB = size_t(n) * nrhs;
  const size_t oA = 0, oB = align(nA * sizeof(cd)), oWk = align(oB + nB * sizeof(cd)), oP = align(oWk + size_t(lwork) * sizeof(cd)),
               oI = align(oP + size_t(n) * sizeof(int)), tot = oI + sizeof(int);
  auto *base = static_cast<char *>(W.get(tot));
  if (not base) return false;
  auto *dA = reinterpret_cast<cd *>(base + oA);
  auto *dB = reinterpret_cast<cd *>(base + oB);
  auto *dK = reinterpret_cast<cd *>(base + oWk);
  auto *dP = reinterpret_cast<int *>(base + oP);
  auto *dI = reinterpret_cast<int *>(base + oI);
  if (not ok(cudaMemcpy(dA, A, nA * sizeof(cd), cudaMemcpyHostToDevice)) or
      not ok(cudaMemcpy(dB, B, nB * sizeof(cd), cudaMemcpyHostToDevice)))
    return false;
  if (not ok(cusolverDnZgetrf(W.h, n, n, dA, n, dK, dP, dI))) return false;
  int info = -1;
  if (not ok(cudaMemcpy(&info, dI, sizeof(int), cudaMemcpyDeviceToHost)) or info != 0) return false;
  if (not ok(cusolverDnZgetrs(W.h, CUBLAS_OP_N, n, nrhs, dA, n, dP, dB, n, dI))) return false;
  if (not ok(cudaMemcpy(&info, dI, sizeof(int), cudaMemcpyDeviceToHost)) or info != 0) return false;
  return ok(cudaMemcpy(B, dB, nB * sizeof(cd), cudaMemcpyDeviceToHost));
}

bool dev_heevd(int n, cplx *A, double *w) {
  const bool r = impl_heevd(n, A, w);
  if (not r) { cudaGetLastError(); ++g_failures; }
  return r;
}
bool dev_gesvd(int n, cplx const *A, double *s, cplx *P, cplx *Qh, int variant) {
  const bool r = impl_gesvd(n, A, s, P, Qh, variant);
  if (not r) { cudaGetLastError(); ++g_failures; }
  return r;
}
bool dev_lu_solve(int n, int nrhs, cplx const *A, cplx *B) {
  const bool r = impl_lu_solve(n, nrhs, A, B);
  if (not r) { cudaGetLastError(); ++g_failures; }
  return r;
}

long dev_lapack_failures() { return g_failures; }

void dev_lapack_release() { ws().release(); }

} // namespace methods::gw_line::cuda
