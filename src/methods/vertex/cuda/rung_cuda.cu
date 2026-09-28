/*
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

// The streaming THC rung on the device -- see rung_cuda.cuh. The host reference is
// vertex_dynbse.icc::thc_rung_apply_ft (FFTW route); the arithmetic follows it step for step.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "IO/AppAbort.hpp"
#include "methods/vertex/cuda/rung_cuda.cuh"

namespace methods::solvers::dynbse_cuda {

  namespace {

    using cd = cuDoubleComplex;

    void rs_cu(cudaError_t e, char const *what) {
      if (e == cudaSuccess) return;
      APP_ABORT(std::string(" rung_cuda: CUDA error in ") + what + ": " + cudaGetErrorName(e) + " (" +
                cudaGetErrorString(e) + ")");
    }
    void rs_cub(cublasStatus_t e, char const *what) {
      if (e == CUBLAS_STATUS_SUCCESS) return;
      APP_ABORT(std::string(" rung_cuda: cuBLAS error in ") + what + ": status " + std::to_string(int(e)));
    }
    void rs_fft(cufftResult e, char const *what) {
      if (e == CUFFT_SUCCESS) return;
      APP_ABORT(std::string(" rung_cuda: cuFFT error in ") + what + ": status " + std::to_string(int(e)));
    }

    constexpr int TPB = 256;
    inline int nblocks(long n) { return int(std::min<long>((n + TPB - 1) / TPB, 65535l * 8)); }

    // ---- the leg tables of one (s, q), by mesh row R (k = kinv[R], kq = kpq[k]) -------------------------------------
    //   XcL(R; P, a) = conj X(k, P, a)     XkqT(R; a, P) = X(kq, P, a)
    //   XkT(R; a, P) = X(k, P, a)          XcqL(R; P, a) = conj X(kq, P, a)
    __global__ void legs_kernel(long nk, long Nm, long nc, cd const *__restrict__ X, long const *__restrict__ kinv,
                                long const *__restrict__ kpq, cd *__restrict__ XcL, cd *__restrict__ XkqT,
                                cd *__restrict__ XkT, cd *__restrict__ XcqL) {
      const long tot = nk * Nm * nc;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long R = e / (Nm * nc), P = (e / nc) % Nm, a = e % nc;
        const long k = kinv[R], kq = kpq[k];
        const cd xk = X[(k * Nm + P) * nc + a], xq = X[(kq * Nm + P) * nc + a];
        XcL[(R * Nm + P) * nc + a] = cuConj(xk);
        XkT[(R * nc + a) * Nm + P] = xk;
        XkqT[(R * nc + a) * Nm + P] = xq;
        XcqL[(R * Nm + P) * nc + a] = cuConj(xq);
      }
    }

    // ---- the rung table into mesh rows: Ah(lex_q(q), PQ) = Ws(q, PQ) (Ws = the contiguous staging copy) --------------
    __global__ void wgather_kernel(long nq, long NN, cd const *__restrict__ Ws, long const *__restrict__ lex_q,
                                   cd *__restrict__ Ah) {
      const long tot = nq * NN;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long q = e / NN, pq = e % NN;
        Ah[lex_q[q] * NN + pq] = Ws[e];
      }
    }
    __global__ void scale_kernel(long n, double s, cd *__restrict__ x) {
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x)
        x[e] = make_cuDoubleComplex(s * x[e].x, s * x[e].y);
    }

    // ---- the input block by mesh row: Fk(R; p1', n nc + p3) = F(k, p1' nc + p3, n0 + n), zero for n >= nbl -----------
    __global__ void packF_kernel(long nk, long nc, long nb, long nR, long n0, long nbl, cd const *__restrict__ F,
                                 long const *__restrict__ kinv, cd *__restrict__ Fk) {
      const long nc2 = nc * nc, tot = nk * nc * nb * nc;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long R = e / (nc * nb * nc), p1p = (e / (nb * nc)) % nc, n = (e / nc) % nb, p3 = e % nc;
        const long k = kinv[R];
        Fk[e] = (n < nbl) ? F[(k * nc2 + p1p * nc + p3) * nR + n0 + n] : make_cuDoubleComplex(0.0, 0.0);
      }
    }

    // ---- the k-sum's product in the mesh-transform domain: Y(R, (P n Q)) *= Ah(R, P Q) (Ah carries the 1/nk) ---------
    __global__ void wmul_kernel(long nk, long Nm, long nb, cd *__restrict__ Y, cd const *__restrict__ Ah) {
      const long M = Nm * nb * Nm, tot = nk * M;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long R = e / M, rem = e % M, P = rem / (nb * Nm), Q = rem % Nm;
        Y[e] = cuCmul(Y[e], Ah[(R * Nm + P) * Nm + Q]);
      }
    }

    // ---- out(k, p1 nc + p3', n0 + n) = scale O2(R; p1 nb + n, p3') for n < nbl (one writer per element) ---------------
    __global__ void scatter_kernel(long nk, long nc, long nb, long nR, long n0, long nbl, cd scale,
                                   cd const *__restrict__ O2, long const *__restrict__ kinv, cd *__restrict__ out) {
      const long nc2 = nc * nc, tot = nk * nc * nbl * nc;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long R = e / (nc * nbl * nc), p1 = (e / (nbl * nc)) % nc, n = (e / nc) % nbl, p3p = e % nc;
        const long k = kinv[R];
        out[(k * nc2 + p1 * nc + p3p) * nR + n0 + n] = cuCmul(scale, O2[(R * nc * nb + p1 * nb + n) * nc + p3p]);
      }
    }

    // row-major C (m x n) = A (m x k) B (k x n), batched with uniform strides (the column-major C^T = B^T A^T)
    void gemm_rm(cublasHandle_t h, long m, long n, long k, cd const *A, long sA, cd const *B, long sB, cd *C, long sC,
                 long batch, char const *what) {
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      rs_cub(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(n), int(m), int(k), &one, B, int(n), sB, A, int(k),
                                       sA, &zero, C, int(n), sC, int(batch)),
             what);
    }

    double now() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // the element counts of the buffers at block width nb (F / out staging at nR_max columns)
    struct rs_sizes {
      long X, legs, Y, Ah, A, Fk, B, O2, Fio;
      rs_sizes(rs_config const &c, long nb, long nR_max) {
        const long nk = c.nk, Nm = c.Nm, nc = c.nc;
        X = c.ns * nk * Nm * nc;
        legs = nk * Nm * nc;
        Y = std::max(nk * Nm * Nm * nb, c.nq * Nm * Nm);      // also the W staging buffer
        Ah = nk * Nm * Nm;
        A = nk * Nm * nb * nc;
        Fk = nk * nc * nb * nc;
        B = nk * nc * nb * Nm;
        O2 = nk * nc * nb * nc;
        Fio = nk * nc * nc * nR_max;
      }
      double elems() const { return double(X + 4 * legs + Y + Ah + A + Fk + B + O2 + 2 * Fio); }
    };
    // the cuFFT work areas are not known before the plans exist: budget one data-sized area for the block transform
    double rs_bytes_nb(rs_config const &c, long nb, long nR_max) {
      rs_sizes s(c, nb, nR_max);
      return 16.0 * (s.elems() + double(s.Y)) + 8.0 * double(3 * c.nk + c.nq) + 64.0e6;
    }

  } // namespace

  struct rung_stream {
    rs_config c;
    long nb = 1, nR_max = 0, is_cur = -1;
    cublasHandle_t cb = nullptr;
    cufftHandle plan_y = 0, plan_w = 0;
    void *fft_work = nullptr;
    long *kinv = nullptr, *lex_q = nullptr, *kpq = nullptr;
    cd *X = nullptr, *XcL = nullptr, *XkqT = nullptr, *XkT = nullptr, *XcqL = nullptr;
    cd *Y = nullptr, *Ah = nullptr, *A = nullptr, *Fk = nullptr, *B = nullptr, *O2 = nullptr, *F = nullptr, *out = nullptr;
  };

  double rs_bytes(rs_config const &c, long nR_max) { return rs_bytes_nb(c, std::max(1l, std::min(c.nb, nR_max)), nR_max); }

  long rs_nb(rung_stream const *e) { return e ? e->nb : 0; }

  rung_stream *rs_create(rs_config const &c, long nR_max, long const *lex_k, long const *lex_q, cplx const *Xb,
                         double free_bytes, char *why, long why_len) {
    auto fail = [&](std::string const &m) -> rung_stream * {
      if (why && why_len > 0) std::snprintf(why, size_t(why_len), "%s", m.c_str());
      return nullptr;
    };
    if (c.nk <= 0 || c.Nm <= 0 || c.nc <= 0 || c.ns <= 0 || nR_max <= 0) return fail("empty configuration");
    if (long(c.ndim[0]) * c.ndim[1] * c.ndim[2] != c.nk || c.nq != c.nk) return fail("not a full nosym mesh");
    long nb = std::max(1l, std::min(c.nb, nR_max));
    while (nb > 1 && rs_bytes_nb(c, nb, nR_max) > 0.9 * free_bytes) nb = std::max(1l, nb / 2);
    if (rs_bytes_nb(c, nb, nR_max) > 0.9 * free_bytes) {
      char b[160];
      std::snprintf(b, sizeof(b), "needs %.2f GB at nb = 1, %.2f GB free", rs_bytes_nb(c, 1, nR_max) / 1e9, free_bytes / 1e9);
      return fail(b);
    }
    auto *e = new rung_stream;
    e->c = c;
    e->nb = nb;
    e->nR_max = nR_max;
    rs_sizes s(c, nb, nR_max);
    auto dalloc = [&](auto *&p, long n, char const *what) {
      rs_cu(cudaMalloc(reinterpret_cast<void **>(&p), size_t(std::max(1l, n)) * sizeof(*p)), what);
    };
    dalloc(e->kinv, c.nk, "kinv");
    dalloc(e->lex_q, c.nq, "lex_q");
    dalloc(e->kpq, c.nk, "kpq");
    dalloc(e->X, s.X, "X");
    dalloc(e->XcL, s.legs, "XcL");
    dalloc(e->XkqT, s.legs, "XkqT");
    dalloc(e->XkT, s.legs, "XkT");
    dalloc(e->XcqL, s.legs, "XcqL");
    dalloc(e->Y, s.Y, "Y");
    dalloc(e->Ah, s.Ah, "Ah");
    dalloc(e->A, s.A, "A");
    dalloc(e->Fk, s.Fk, "Fk");
    dalloc(e->B, s.B, "B");
    dalloc(e->O2, s.O2, "O2");
    dalloc(e->F, s.Fio, "F");
    dalloc(e->out, s.Fio, "out");
    {
      std::vector<long> kinv(size_t(c.nk), -1);
      for (long k = 0; k < c.nk; ++k) kinv[size_t(lex_k[k])] = k;
      for (long r = 0; r < c.nk; ++r)
        if (kinv[size_t(r)] < 0) { rs_destroy(e); return fail("lex_k is not a permutation of the mesh rows"); }
      rs_cu(cudaMemcpy(e->kinv, kinv.data(), size_t(c.nk) * sizeof(long), cudaMemcpyHostToDevice), "kinv H2D");
      rs_cu(cudaMemcpy(e->lex_q, lex_q, size_t(c.nq) * sizeof(long), cudaMemcpyHostToDevice), "lex_q H2D");
      rs_cu(cudaMemcpy(e->X, Xb, size_t(s.X) * sizeof(cd), cudaMemcpyHostToDevice), "X H2D");
    }
    rs_cub(cublasCreate(&e->cb), "rs cublasCreate");
    // the plans: 3-D transforms over the mesh rows, the transform index fastest (stride = batch count, distance 1) -- the
    // FFTW plan_many layout of thc_rung_apply_ft; one shared work area
    {
      long long n[3] = {c.ndim[0], c.ndim[1], c.ndim[2]};
      const long long My = c.Nm * nb * c.Nm, Mw = c.Nm * c.Nm;
      size_t wy = 0, ww = 0;
      rs_fft(cufftCreate(&e->plan_y), "cufftCreate y");
      rs_fft(cufftCreate(&e->plan_w), "cufftCreate w");
      rs_fft(cufftSetAutoAllocation(e->plan_y, 0), "autoalloc y");
      rs_fft(cufftSetAutoAllocation(e->plan_w, 0), "autoalloc w");
      rs_fft(cufftMakePlanMany64(e->plan_y, 3, n, n, My, 1, n, My, 1, CUFFT_Z2Z, My, &wy), "plan y");
      rs_fft(cufftMakePlanMany64(e->plan_w, 3, n, n, Mw, 1, n, Mw, 1, CUFFT_Z2Z, Mw, &ww), "plan w");
      const size_t wmax = std::max(wy, ww);
      if (wmax > 0) {
        const cudaError_t err = cudaMalloc(&e->fft_work, wmax);
        if (err != cudaSuccess) {
          cudaGetLastError();
          rs_destroy(e);
          char b[160];
          std::snprintf(b, sizeof(b), "cuFFT work area %.2f GB does not fit", double(wmax) / 1e9);
          return fail(b);
        }
      }
      rs_fft(cufftSetWorkArea(e->plan_y, e->fft_work), "work y");
      rs_fft(cufftSetWorkArea(e->plan_w, e->fft_work), "work w");
    }
    return e;
  }

  void rs_destroy(rung_stream *e) {
    if (!e) return;
    if (e->plan_y) cufftDestroy(e->plan_y);
    if (e->plan_w) cufftDestroy(e->plan_w);
    if (e->cb) cublasDestroy(e->cb);
    for (void *p : {(void *)e->kinv, (void *)e->lex_q, (void *)e->kpq, (void *)e->X, (void *)e->XcL, (void *)e->XkqT,
                    (void *)e->XkT, (void *)e->XcqL, (void *)e->Y, (void *)e->Ah, (void *)e->A, (void *)e->Fk, (void *)e->B,
                    (void *)e->O2, (void *)e->F, (void *)e->out, e->fft_work})
      if (p) cudaFree(p);
    delete e;
  }

  void rs_set_sq(rung_stream *e, long is, long const *kpq_row) {
    auto const &c = e->c;
    rs_cu(cudaMemcpy(e->kpq, kpq_row, size_t(c.nk) * sizeof(long), cudaMemcpyHostToDevice), "kpq H2D");
    legs_kernel<<<nblocks(c.nk * c.Nm * c.nc), TPB>>>(c.nk, c.Nm, c.nc, e->X + is * c.nk * c.Nm * c.nc, e->kinv, e->kpq,
                                                      e->XcL, e->XkqT, e->XkT, e->XcqL);
    rs_cu(cudaGetLastError(), "legs_kernel");
    e->is_cur = is;
  }

  void rs_set_w(rung_stream *e, cplx const *W, long ldq, double *timing) {
    auto const &c = e->c;
    const double t0 = now();
    const long NN = c.Nm * c.Nm;
    // staging: q rows of NN elements at host pitch ldq into Y (contiguous), then into the mesh rows of Ah
    rs_cu(cudaMemcpy2D(e->Y, size_t(NN) * sizeof(cd), W, size_t(ldq) * sizeof(cd), size_t(NN) * sizeof(cd), size_t(c.nq),
                       cudaMemcpyHostToDevice),
          "W H2D");
    wgather_kernel<<<nblocks(c.nq * NN), TPB>>>(c.nq, NN, e->Y, e->lex_q, e->Ah);
    rs_cu(cudaGetLastError(), "wgather_kernel");
    // A(R) = sum_q e^{+i q.R} W(q): the unnormalized backward transform (FFTW_BACKWARD == CUFFT_INVERSE), then the 1/nk of
    // the forward-backward pair folded in (the host multiplies by a * inv_nk)
    rs_fft(cufftExecZ2Z(e->plan_w, e->Ah, e->Ah, CUFFT_INVERSE), "exec w");
    scale_kernel<<<nblocks(c.nk * NN), TPB>>>(c.nk * NN, 1.0 / double(c.nk), e->Ah);
    rs_cu(cudaGetLastError(), "scale_kernel");
    if (timing) {
      rs_cu(cudaDeviceSynchronize(), "set_w sync");
      timing[0] += now() - t0;
    }
  }

  void rs_apply(rung_stream *e, cplx scale, cplx const *F, cplx *out, long nR, double *timing) {
    auto const &c = e->c;
    if (nR > e->nR_max) APP_ABORT(" rung_cuda: rs_apply block wider than the engine's nR_max.");
    if (e->is_cur < 0) APP_ABORT(" rung_cuda: rs_apply before rs_set_sq.");
    const long nk = c.nk, Nm = c.Nm, nc = c.nc, nb = e->nb, M = Nm * nb * Nm;
    const long nio = nk * nc * nc * nR;
    double t = now(), tt[4] = {0.0, 0.0, 0.0, 0.0};
    auto lap = [&](int slot) {
      if (!timing) return;
      rs_cu(cudaDeviceSynchronize(), "apply sync");
      const double t1 = now();
      tt[slot] += t1 - t;
      t = t1;
    };
    rs_cu(cudaMemcpy(e->F, F, size_t(nio) * sizeof(cd), cudaMemcpyHostToDevice), "F H2D");
    lap(3);
    const cd sc = make_cuDoubleComplex(scale.real(), scale.imag());
    for (long n0 = 0; n0 < nR; n0 += nb) {
      const long nbl = std::min(nb, nR - n0);
      // ---- legs in: Y(R, (P n Q)) = sum_p3 [sum_p1' conj X(k, P, p1') F(k, p1' p3, n)] X(k+q, Q, p3) ---------------
      packF_kernel<<<nblocks(nk * nc * nb * nc), TPB>>>(nk, nc, nb, nR, n0, nbl, e->F, e->kinv, e->Fk);
      rs_cu(cudaGetLastError(), "packF_kernel");
      gemm_rm(e->cb, Nm, nb * nc, nc, e->XcL, Nm * nc, e->Fk, nc * nb * nc, e->A, Nm * nb * nc, nk, "legs in A");
      gemm_rm(e->cb, Nm * nb, Nm, nc, e->A, Nm * nb * nc, e->XkqT, nc * Nm, e->Y, M, nk, "legs in Y");
      lap(0);
      // ---- the k-sum: forward transform, the product with A, backward transform ------------------------------------
      rs_fft(cufftExecZ2Z(e->plan_y, e->Y, e->Y, CUFFT_FORWARD), "exec y fwd");
      wmul_kernel<<<nblocks(nk * M), TPB>>>(nk, Nm, nb, e->Y, e->Ah);
      rs_cu(cudaGetLastError(), "wmul_kernel");
      rs_fft(cufftExecZ2Z(e->plan_y, e->Y, e->Y, CUFFT_INVERSE), "exec y inv");
      lap(1);
      // ---- legs out: B(p1, (n Q)) = X(k', ., p1)^T Z; O2((p1 n), p3') = B2 conj X(k'+q, ., p3') ----------------------
      gemm_rm(e->cb, nc, nb * Nm, Nm, e->XkT, nc * Nm, e->Y, M, e->B, nc * nb * Nm, nk, "legs out B");
      gemm_rm(e->cb, nc * nb, nc, Nm, e->B, nc * nb * Nm, e->XcqL, Nm * nc, e->O2, nc * nb * nc, nk, "legs out O2");
      scatter_kernel<<<nblocks(nk * nc * nbl * nc), TPB>>>(nk, nc, nb, nR, n0, nbl, sc, e->O2, e->kinv, e->out);
      rs_cu(cudaGetLastError(), "scatter_kernel");
      lap(2);
    }
    rs_cu(cudaMemcpy(out, e->out, size_t(nio) * sizeof(cd), cudaMemcpyDeviceToHost), "out D2H");
    lap(3);
    if (timing)
      for (int i = 0; i < 4; ++i) timing[i] += tt[i];
  }

} // namespace methods::solvers::dynbse_cuda
