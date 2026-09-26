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

// The device L0 kernel (see l0_cuda.cuh). Kept free of fmt-based logging: this TU is compiled by nvcc
// and, like numerics/device_kernels/cuda, reports through APP_ABORT with a std::string.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "IO/AppAbort.hpp"
#include "methods/vertex/cuda/l0_cuda.cuh"

namespace methods::solvers::dynbse_cuda {

  namespace {

    using cd = cuDoubleComplex;

    void cu_check(cudaError_t e, char const *what) {
      if (e == cudaSuccess) return;
      APP_ABORT(std::string(" l0_cuda: CUDA error in ") + what + ": " + cudaGetErrorName(e) + " (" +
                cudaGetErrorString(e) + ")");
    }
    void cub_check(cublasStatus_t e, char const *what) {
      if (e == CUBLAS_STATUS_SUCCESS) return;
      APP_ABORT(std::string(" l0_cuda: cuBLAS error in ") + what + ": status " + std::to_string(int(e)));
    }
    void launch_check(char const *what) {
      cu_check(cudaGetLastError(), what);
    }

    __device__ __forceinline__ cd operator+(cd a, cd b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
    __device__ __forceinline__ cd operator-(cd a, cd b) { return make_cuDoubleComplex(a.x - b.x, a.y - b.y); }
    __device__ __forceinline__ cd operator*(cd a, cd b) { return cuCmul(a, b); }
    __device__ __forceinline__ cd operator/(cd a, cd b) { return cuCdiv(a, b); }
    __device__ __forceinline__ cd neg(cd a) { return make_cuDoubleComplex(-a.x, -a.y); }
    __device__ __forceinline__ cd real(double s) { return make_cuDoubleComplex(s, 0.0); }
    __device__ __forceinline__ cd scal(double s, cd a) { return make_cuDoubleComplex(s * a.x, s * a.y); }
    __device__ __forceinline__ bool nonzero(cd a) { return a.x != 0.0 || a.y != 0.0; }
    __device__ __forceinline__ void atomic_add(cd *p, cd v) {
      atomicAdd(reinterpret_cast<double *>(p), v.x);
      atomicAdd(reinterpret_cast<double *>(p) + 1, v.y);
    }

    // nca = the number of ACTIVE input components packed into Vt (P-3a); act[il] = their global component index
    // c (0 = the constant, 1 + f np + a = node a of family f). A frequency-constant input packs one component.
    struct kdims { long nc, nR, np, ng, nca, blk, nk, np_fit; long const *act; };

    // ---- pack the ACTIVE components of X of K k-points into Vt(k; x, il, r, y) ----------------------
    __global__ void pack_kernel(kdims d, long ik0, long K, cd const *__restrict__ Xfam,
                                cd const *__restrict__ Xcst, cd *__restrict__ Vt) {
      const long kb = blockIdx.y, ik = ik0 + kb;
      const long nc = d.nc, nR = d.nR, np = d.np, nk = d.nk, nca = d.nca;
      const long W = nc * nca * nR * nc, tot = nc * nR * nc;
      if (kb >= K) return;
      cd *V = Vt + kb * W;
      for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < tot; e += gridDim.x * blockDim.x) {
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        for (long il = 0; il < nca; ++il) {
          const long c = d.act[il];
          cd v;
          if (c == 0) v = Xcst[((ik * nc + x) * nc + y) * nR + r];
          else {
            const long f = (c - 1) / np, a = (c - 1) % np;
            v = Xfam[(((f * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r];
          }
          V[((x * nca + il) * nR + r) * nc + y] = v;
        }
      }
    }

    // ---- the per-pole nc x nc operands of the gemms for K k-points ---------------------------------
    // transpose: dst(j; x, y) = src(j, ik; y, x) for gk / gkq (layout (ng, nk, nc, nc));
    // per_k_major: src is (nk, ng, nc, nc) (Ghat / Gtil), copied as is.
    __global__ void poleT_kernel(kdims d, long ik0, long K, cd const *__restrict__ src, bool transpose,
                                 bool per_k_major, cd *__restrict__ dst) {
      const long kb = blockIdx.y, ik = ik0 + kb;
      const long nc = d.nc, ng = d.ng, nk = d.nk;
      if (kb >= K) return;
      cd *D = dst + kb * ng * nc * nc;
      for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < ng * nc * nc; e += gridDim.x * blockDim.x) {
        const long j = e / (nc * nc), x = (e / nc) % nc, y = e % nc;
        const long s = per_k_major ? ((ik * ng + j) * nc + (transpose ? y : x)) * nc + (transpose ? x : y)
                                   : (((j * nk + ik) * nc + (transpose ? y : x)) * nc + (transpose ? x : y));
        D[(j * nc + x) * nc + y] = src[s];
      }
    }

    // ---- the cuBLAS batched-pointer arrays (batch index i = kb * ng + j) ----------------------------
    __global__ void fill_ptrs(long K, long ng, long W, long nc2, cd *Vt, cd *gjT, cd *glT, cd *Gh,
                              cd *Pj, cd *Qj, cd *Bj, cd **pVt, cd **pGjT, cd **pGlT, cd **pGh,
                              cd **pPj, cd **pQj, cd **pBj) {
      const long i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= K * ng) return;
      const long kb = i / ng;
      pVt[i] = Vt + kb * W;          // the ng poles of one k share Vt: only a pointer array can say this
      pGjT[i] = gjT + i * nc2;
      pGlT[i] = glT + i * nc2;
      pGh[i] = Gh + i * nc2;
      pPj[i] = Pj + i * W;
      pQj[i] = Qj + i * W;
      pBj[i] = Bj + i * W;
    }

    // ---- the scatter: dynbse.hpp's mulU / mulT on the (x, r, y) block c of every pole's output -------
    // grid (ncomp, K), one thread per element of the block, the ng poles looped inside; the
    // nj-indexed targets are atomics, the a-indexed targets (AU[a], AT[a]) are reduced in registers
    // over the poles and added once. which_pass 0: the j loop (mulU on Q_j, mulT on B_j);
    // 1: the l loop (mulU on -R_l, mulT on i nu R_l), with Qall = Ball = R.
    __global__ void scatter_kernel(kdims d, long K, int which_pass, cd inu, bool skip_cst,
                                   cd const *__restrict__ Qall, cd const *__restrict__ Ball,
                                   double const *__restrict__ eps, double const *__restrict__ epsG,
                                   long const *__restrict__ gnode,
                                   cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2,
                                   cd *__restrict__ A1, cd *__restrict__ A3) {
      const long il = blockIdx.x, kb = blockIdx.y;   // il: the packed position; c: the global component
      if (kb >= K) return;
      const long c = d.act[il];
      if (c == 0 && skip_cst) return;
      const long np = d.np, ng = d.ng, blk = d.blk, nca = d.nca;
      const long W = d.nc * nca * d.nR * d.nc, asz = 2 * np * blk;
      const int part = (c == 0) ? 1 : 0;
      const long abase = kb * asz + (long(part) * np) * blk;
      const bool isU = (c >= 1 && c <= np), isT = (c > np);
      const long a = isU ? c - 1 : (isT ? c - 1 - np : -1);
      const double ea = (a >= 0) ? eps[a] : 0.0;
      cd const *Qk = Qall + kb * ng * W;
      cd const *Bk = Ball + kb * ng * W;
      for (long e = threadIdx.x; e < blk; e += blockDim.x) {
        const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
        const long src = (x * nca + il) * d.nR * d.nc + rem;
        cd accU = make_cuDoubleComplex(0.0, 0.0), accT = accU;
        for (long pole = 0; pole < ng; ++pole) {
          const long nj = gnode[pole];
          const double ej = epsG[pole];
          const cd q = Qk[pole * W + src];
          // the vectors the production multiplications act on
          const cd vU = (which_pass == 0) ? q : neg(q);                   // U_j . Q_j   |  - U_l . R_l
          const cd vT = (which_pass == 0) ? Bk[pole * W + src] : inu * q;  // T_j . B_j   |  i nu T_l . R_l
          cd *AUnj = &AU[abase + nj * blk + e], *ATnj = &AT[abase + nj * blk + e];
          // ---- mulU(pole, c, vU) ----
          if (c == 0) {                                   // U_j . C
            atomic_add(AUnj, vU);
          } else if (isU) {                               // U_j . U_a
            if (a == nj) {
              atomic_add(&M2[abase + nj * blk + e], vU);
            } else {
              const double w = 1.0 / (ej - ea);
              atomic_add(AUnj, scal(w, vU));
              accU = accU - scal(w, vU);                  // AU[a] -= w v
            }
          } else {                                        // U_j . T_a
            if (a == nj) {                                // U_j T_j = R1_j
              atomic_add(&A1[abase + nj * blk + e], vU);
            } else {
              const cd w = real(1.0 / (ea - ej));
              const cd dd = real(ej - ea) + inu;
              const cd wd = w / dd;
              const cd wt = w - inu * wd;
              accT = accT + wt * vU;                      // AT[a] += wt v
              atomic_add(AUnj, neg(wd * vU));             // AU[nj] -= wd v
              accU = accU + wd * vU;                      // AU[a] += wd v
            }
          }
          // ---- mulT(pole, c, vT) ----
          if (c == 0) {                                   // T_l . C
            atomic_add(ATnj, vT);
          } else if (isU) {                               // T_l . U_a
            if (a == nj) {                                // U_l T_l = R1_l
              atomic_add(&A1[abase + nj * blk + e], vT);
            } else {
              const cd w = real(1.0 / (ej - ea));
              const cd dd = real(ea - ej) + inu;
              const cd wd = w / dd;
              const cd wt = w - inu * wd;
              atomic_add(ATnj, wt * vT);                  // AT[nl] += wt v
              accU = accU - wd * vT;                      // AU[a] -= wd v
              atomic_add(AUnj, wd * vT);                  // AU[nl] += wd v
            }
          } else {                                        // T_l . T_a
            if (a == nj) {                                // T_l^2 = R3_l
              atomic_add(&A3[abase + nj * blk + e], vT);
            } else {
              const double g = ej - ea;
              const cd w2 = real(1.0 / (g * g));
              const cd dla = real(ej - ea) + inu, dal = real(ea - ej) + inu;
              const cd wla = w2 / dla, wal = w2 / dal;
              const cd ctl = w2 - inu * wal, cta = w2 - inu * wla, cul = wal - wla, cua = wla - wal;
              atomic_add(ATnj, ctl * vT);                 // AT[nl] += ctl v
              accT = accT + cta * vT;                     // AT[a]  += cta v
              atomic_add(AUnj, cul * vT);                 // AU[nl] += cul v
              accU = accU + cua * vT;                     // AU[a]  += cua v
            }
          }
        }
        if (a >= 0) {
          if (nonzero(accU)) atomic_add(&AU[abase + a * blk + e], accU);
          if (nonzero(accT)) atomic_add(&AT[abase + a * blk + e], accT);
        }
      }
    }

    // ---- the assembly of dynbse.hpp: AU / AT / M2 / A1 / A3 -> F, Fsum --------------------------------
    // grid (np, 2, K): one block per (node n, part, k); one thread per (x, r, y).
    __global__ void assemble_kernel(kdims d, long ik0, long K, bool sum_part1,
                                    double const *__restrict__ fhalf, double const *__restrict__ fd1,
                                    cd const *__restrict__ Dsq,
                                    cd const *__restrict__ s1, cd const *__restrict__ s3,
                                    cd const *__restrict__ r1u, cd const *__restrict__ r3u, cd const *__restrict__ r3t,
                                    cd const *__restrict__ R1U, cd const *__restrict__ R1T,
                                    cd const *__restrict__ R3U, cd const *__restrict__ R3T,
                                    cd const *__restrict__ AU, cd const *__restrict__ AT, cd const *__restrict__ M2,
                                    cd const *__restrict__ A1, cd const *__restrict__ A3,
                                    cd *__restrict__ Ffam, cd *__restrict__ Fsum, int *__restrict__ err) {
      const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk, asz = 2 * np * blk;
      const long n = blockIdx.x, part = blockIdx.y, kb = blockIdx.z;
      if (kb >= K) return;
      const long ik = ik0 + kb;
      const bool sum_this = (part == 0) || sum_part1;
      for (long e = threadIdx.x; e < blk; e += blockDim.x) {
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        const long ia = kb * asz + (part * np + n) * blk + e;
        const cd u = AU[ia], t = AT[ia], m = M2[ia], v1 = A1[ia], v3 = A3[ia];
        const long f0n = (((0 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        const long f1n = (((1 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        const long fs = ((ik * nc + x) * nc + y) * nR + r;
        atomic_add(&Ffam[f0n], u);
        atomic_add(&Ffam[f1n], t);
        if (sum_this) atomic_add(&Fsum[fs], scal(fhalf[n], u));           // T sums to 0 exactly
        const bool any2 = nonzero(m), any1 = nonzero(v1), any3 = nonzero(v3);
        if ((any2 || any1 || any3) && n >= d.np_fit) { atomicOr(err, 1); continue; }
        if (any2) {                                                        // U_n^2 through Dsq
          if (sum_this) atomic_add(&Fsum[fs], scal(fd1[n], m));
          for (long c = 0; c < np; ++c) {
            const cd dc = Dsq[n * np + c];
            if (!nonzero(dc)) continue;
            atomic_add(&Ffam[(((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r], dc * m);
          }
        }
        if (any1) {                                                        // R1_n
          if (sum_this) atomic_add(&Fsum[fs], s1[n] * v1);
          atomic_add(&Ffam[f0n], r1u[n] * v1);
          for (long c = 0; c < np; ++c) {
            const long f0c = (((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
            const long f1c = (((1 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
            atomic_add(&Ffam[f0c], R1U[n * np + c] * v1);
            atomic_add(&Ffam[f1c], R1T[n * np + c] * v1);
          }
        }
        if (any3) {                                                        // R3_n
          if (sum_this) atomic_add(&Fsum[fs], s3[n] * v3);
          atomic_add(&Ffam[f0n], r3u[n] * v3);
          atomic_add(&Ffam[f1n], r3t[n] * v3);
          for (long c = 0; c < np; ++c) {
            const long f0c = (((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
            const long f1c = (((1 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
            atomic_add(&Ffam[f0c], R3U[n * np + c] * v3);
            atomic_add(&Ffam[f1c], R3T[n * np + c] * v3);
          }
        }
      }
    }

    // ---- the small-nu fold (D2f): T_a with |eps_a| >= ratio |nu| folded into the U family ------------
    // grid (np, K): after the assembly of this batch (a separate launch, so every contribution to
    // F.fam(1, a) is in). Reads family 1, writes family 0 (atomics) and zeroes its own family-1 slot.
    __global__ void tfold_kernel(kdims d, long ik0, long K, cd inu, double tfold,
                                 double const *__restrict__ eps,
                                 cd const *__restrict__ Dsq, cd const *__restrict__ Dcb, cd const *__restrict__ Dqt,
                                 cd *__restrict__ Ffam) {
      const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk;
      const long a = blockIdx.x, kb = blockIdx.y;
      if (kb >= K) return;
      const double anu = cuCabs(inu);
      if (fabs(eps[a]) < tfold * anu) return;
      const long ik = ik0 + kb;
      const cd inu2 = inu * inu;
      for (long e = threadIdx.x; e < blk; e += blockDim.x) {
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        const long f1a = (((1 * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        const cd t = Ffam[f1a];
        if (!nonzero(t)) continue;
        for (long c = 0; c < np; ++c) {
          const cd dc = Dsq[a * np + c] - inu * Dcb[a * np + c] + inu2 * Dqt[a * np + c];
          if (!nonzero(dc)) continue;
          atomic_add(&Ffam[(((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r], dc * t);
        }
        Ffam[f1a] = make_cuDoubleComplex(0.0, 0.0);
      }
    }

    template <typename T>
    T *upload(T const *h, size_t n, char const *what) {
      T *p = nullptr;
      cu_check(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), what);
      if (n > 0) cu_check(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice), what);
      return p;
    }
    cd *upload_c(cplx const *h, size_t n, char const *what) {
      return upload<cd>(reinterpret_cast<cd const *>(h), n, what);
    }
    template <typename T>
    T *alloc(size_t n, char const *what) {
      T *p = nullptr;
      cu_check(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), what);
      return p;
    }

  } // namespace

  long l0_apply_shift_cols(l0_dims const &d, l0_tables const &t, cplx *Ffam_h, cplx *Fsum_h,
                           double free_bytes) {
    const long np = d.np, nk = d.nk, nc = d.nc, ng = d.ng, nR = d.nR, nca = d.nca;
    const long blk = nc * nR * nc, W = nc * nca * nR * nc, nc2 = nc * nc;
    const size_t asz = size_t(2 * np) * size_t(blk);
    const size_t nF = size_t(2 * np) * nk * nc * nc * nR, nFs = size_t(nk) * nc * nc * nR;
    if (nk == 0 || np == 0 || nca == 0) return 0;
    if (nca < 1 || nca > 1 + 2 * np || t.act == nullptr)
      APP_ABORT(std::string(" l0_cuda: the active-component list is missing or out of range (nca = ") + std::to_string(nca) + ").");

    // wall clock for the optional timing split (t.timing: alloc, H2D, kernel, D2H)
    auto dnow = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double tt0 = dnow();

    // ---- fixed device data ------------------------------------------------------------------------
    long *dact = upload(t.act, size_t(nca), "act");
    kdims kd{nc, nR, np, ng, nca, blk, nk, d.np_fit, dact};
    cd *dX = upload_c(t.Xfam, size_t(2 * np) * nk * nc * nc * nR, "Xfam");
    cd *dXc = upload_c(t.Xcst, nFs, "Xcst");
    cd *dgk = upload_c(t.gk, size_t(ng) * nk * nc2, "gk");
    cd *dgkq = upload_c(t.gkq, size_t(ng) * nk * nc2, "gkq");
    cd *dGhat = upload_c(t.Ghat, size_t(nk) * ng * nc2, "Ghat");
    cd *dGtil = upload_c(t.Gtil, size_t(nk) * ng * nc2, "Gtil");
    double *deps = upload(t.eps, size_t(np), "eps"), *depsG = upload(t.epsG, size_t(ng), "epsG");
    long *dgn = upload(t.gnode, size_t(ng), "gnode");
    double *dfh = upload(t.fhalf, size_t(np), "fhalf"), *dfd1 = upload(t.fd1, size_t(np), "fd1");
    cd *dDsq = upload_c(t.Dsq, size_t(np) * np, "Dsq");
    cd *dDcb = upload_c(t.Dcb, size_t(np) * np, "Dcb");
    cd *dDqt = upload_c(t.Dqt, size_t(np) * np, "Dqt");
    cd *ds1 = upload_c(t.s1, size_t(np), "s1"), *ds3 = upload_c(t.s3, size_t(np), "s3");
    cd *dr1u = upload_c(t.r1u, size_t(np), "r1u"), *dr3u = upload_c(t.r3u, size_t(np), "r3u");
    cd *dr3t = upload_c(t.r3t, size_t(np), "r3t");
    cd *dR1U = upload_c(t.R1U, size_t(np) * np, "R1U"), *dR1T = upload_c(t.R1T, size_t(np) * np, "R1T");
    cd *dR3U = upload_c(t.R3U, size_t(np) * np, "R3U"), *dR3T = upload_c(t.R3T, size_t(np) * np, "R3T");
    cd *dF = alloc<cd>(nF, "F"), *dFs = alloc<cd>(nFs, "Fsum");
    cu_check(cudaMemset(dF, 0, nF * sizeof(cd)), "memset F");
    cu_check(cudaMemset(dFs, 0, nFs * sizeof(cd)), "memset Fsum");
    int *derr = alloc<int>(1, "err");
    cu_check(cudaMemset(derr, 0, sizeof(int)), "memset err");
    cu_check(cudaDeviceSynchronize(), "uploads");
    const double tt1 = dnow();                       // end of the fixed uploads (H2D)

    // ---- the k-batch: the largest K whose working set fits the memory we were given --------------
    // per k: Vt + 3 ng W (P/Q/B) + 5 accumulators + 3 ng nc^2 pole operands, 16 B each
    const double per_k = double(W + 3 * ng * W + 5 * long(asz) + 3 * ng * nc2) * 16.0;
    const double fixed = double(nF * 2 + nFs * 2 + size_t(2 * ng) * nk * nc2 * 2 + 3 * size_t(np) * np) * 16.0;
    const double budget = 0.85 * free_bytes - fixed;     // 15 % held back for the cuBLAS workspace
    long K = std::max(1L, std::min(nk, long(budget / per_k)));
    if (budget < per_k) K = 1;                           // one k must fit; the allocation will say if not

    // The per-batch working set, allocated with a RETRY: the budget above assumes an exclusive device, but
    // several ranks may share one GPU (P-1 of the gpu port, 2026-09-25: two ranks sized their batches from
    // the same cudaMemGetInfo at the same instant and the second cudaMalloc of Pj failed with
    // cudaErrorMemoryAllocation). On a failed allocation everything of the attempt is freed and K is halved,
    // down to K = 1, which must fit (abort otherwise).
    cd *dVt = nullptr, *dPj = nullptr, *dQj = nullptr, *dBj = nullptr, *dgjT = nullptr, *dglT = nullptr, *dGh = nullptr;
    cd *dAU = nullptr, *dAT = nullptr, *dM2 = nullptr, *dA1 = nullptr, *dA3 = nullptr;
    cd **pVt = nullptr, **pGjT = nullptr, **pGlT = nullptr, **pGh = nullptr, **pPj = nullptr, **pQj = nullptr, **pBj = nullptr;
    size_t nptr = 0;
    {
      auto try_alloc = [&](void **p, size_t bytes) -> bool {
        const cudaError_t e = cudaMalloc(p, std::max<size_t>(bytes, 1));
        if (e != cudaSuccess) { (void)cudaGetLastError(); *p = nullptr; return false; }
        return true;
      };
      auto free_all = [&]() {
        for (void **p : {(void **)&dVt, (void **)&dPj, (void **)&dQj, (void **)&dBj, (void **)&dgjT, (void **)&dglT,
                         (void **)&dGh, (void **)&dAU, (void **)&dAT, (void **)&dM2, (void **)&dA1, (void **)&dA3,
                         (void **)&pVt, (void **)&pGjT, (void **)&pGlT, (void **)&pGh, (void **)&pPj, (void **)&pQj,
                         (void **)&pBj})
          if (*p != nullptr) { (void)cudaFree(*p); *p = nullptr; }
      };
      const long K_first = K;
      bool ok = false;
      while (true) {
        nptr = size_t(K) * size_t(ng);
        const size_t bW = size_t(K) * W * sizeof(cd), bP = size_t(K) * ng * W * sizeof(cd);
        const size_t bG = size_t(K) * ng * nc2 * sizeof(cd), bA = size_t(K) * asz * sizeof(cd), bp = nptr * sizeof(cd *);
        ok = try_alloc((void **)&dVt, bW) and try_alloc((void **)&dPj, bP) and try_alloc((void **)&dQj, bP) and
             try_alloc((void **)&dBj, bP) and try_alloc((void **)&dgjT, bG) and try_alloc((void **)&dglT, bG) and
             try_alloc((void **)&dGh, bG) and try_alloc((void **)&dAU, bA) and try_alloc((void **)&dAT, bA) and
             try_alloc((void **)&dM2, bA) and try_alloc((void **)&dA1, bA) and try_alloc((void **)&dA3, bA) and
             try_alloc((void **)&pVt, bp) and try_alloc((void **)&pGjT, bp) and try_alloc((void **)&pGlT, bp) and
             try_alloc((void **)&pGh, bp) and try_alloc((void **)&pPj, bp) and try_alloc((void **)&pQj, bp) and
             try_alloc((void **)&pBj, bp);
        if (ok) break;
        free_all();
        if (K == 1) break;
        K = std::max(1L, K / 2);
      }
      if (not ok)
        APP_ABORT(std::string(" l0_cuda: the per-k working set does not fit the device even at K = 1 (per_k = ") +
                  std::to_string(per_k / 1e9) + " GB, fixed = " + std::to_string(fixed / 1e9) + " GB, free at entry = " +
                  std::to_string(free_bytes / 1e9) + " GB); the device is shared or too small for this unit.");
      if (K != K_first)
        std::fprintf(stderr, "  [l0_cuda] k-batch reduced from %ld to %ld after a device allocation failure (shared GPU?)\n",
                     K_first, K);
    }
    const double tt2 = dnow();                       // end of the per-k allocations (alloc)
    fill_ptrs<<<unsigned((nptr + 255) / 256), 256>>>(K, ng, W, nc2, dVt, dgjT, dglT, dGh, dPj, dQj, dBj,
                                                     pVt, pGjT, pGlT, pGh, pPj, pQj, pBj);
    launch_check("fill_ptrs");

    cublasHandle_t h;
    cub_check(cublasCreate(&h), "cublasCreate");
    const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const cd inu = make_cuDoubleComplex(t.inu.real(), t.inu.imag());
    const int Ncols = int(nca * nR * nc), Mrows = int(nc * nca * nR);

    for (long ik0 = 0; ik0 < nk; ik0 += K) {
      const long Kb = std::min(K, nk - ik0);
      const int nb = int(Kb * ng);
      for (cd *p : {dAU, dAT, dM2, dA1, dA3}) cu_check(cudaMemset(p, 0, size_t(Kb) * asz * sizeof(cd)), "memset acc");
      pack_kernel<<<dim3(64u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dX, dXc, dVt);
      launch_check("pack_kernel");
      poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dgk, true, false, dgjT);
      poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dgkq, true, false, dglT);
      poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dGhat, false, true, dGh);
      launch_check("poleT_kernel");
      // pass 0: P_j = g_j^T Vt (row-major: Vt^T g_j in cuBLAS's column-major), Q_j = P_j Ghat_j, B_j = P_j gkq_j^T
      cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                                   (const cd **)pVt, Ncols, (const cd **)pGjT, int(nc), &zero, pPj, Ncols, nb), "gemm Pj");
      cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                   (const cd **)pGh, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb), "gemm Qj");
      cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                   (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pBj, int(nc), nb), "gemm Bj");
      scatter_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 0, inu, t.skip_cst, dQj, dBj,
                                                                       deps, depsG, dgn, dAU, dAT, dM2, dA1, dA3);
      launch_check("scatter_kernel pass 0");
      // pass 1: P_l = Gtil_l Vt, R_l = P_l gkq_l^T
      poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dGtil, false, true, dGh);
      launch_check("poleT_kernel Gtil");
      cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                                   (const cd **)pVt, Ncols, (const cd **)pGh, int(nc), &zero, pPj, Ncols, nb), "gemm Pl");
      cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                   (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb), "gemm Rl");
      scatter_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 1, inu, t.skip_cst, dQj, dQj,
                                                                       deps, depsG, dgn, dAU, dAT, dM2, dA1, dA3);
      launch_check("scatter_kernel pass 1");
      assemble_kernel<<<dim3(unsigned(np), 2u, unsigned(Kb)), 256>>>(kd, ik0, Kb, t.sum_part1, dfh, dfd1, dDsq,
                                                                     ds1, ds3, dr1u, dr3u, dr3t, dR1U, dR1T, dR3U, dR3T,
                                                                     dAU, dAT, dM2, dA1, dA3, dF, dFs, derr);
      launch_check("assemble_kernel");
      if (t.tfold > 0.0) {
        tfold_kernel<<<dim3(unsigned(np), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, inu, t.tfold, deps, dDsq, dDcb, dDqt, dF);
        launch_check("tfold_kernel");
      }
    }
    cu_check(cudaDeviceSynchronize(), "l0 batches");
    const double tt3 = dnow();                       // end of the batches (kernel)
    int err = 0;
    cu_check(cudaMemcpy(&err, derr, sizeof(int), cudaMemcpyDeviceToHost), "err");
    if (err != 0)
      APP_ABORT(std::string(" dynbse::l0_apply_shift_cols (device): a confluent product at a node outside the DLR set."));
    cu_check(cudaMemcpy(Ffam_h, dF, nF * sizeof(cd), cudaMemcpyDeviceToHost), "F d2h");
    cu_check(cudaMemcpy(Fsum_h, dFs, nFs * sizeof(cd), cudaMemcpyDeviceToHost), "Fsum d2h");
    const double tt4 = dnow();                       // end of the downloads (D2H)
    if (t.timing != nullptr) {
      t.timing[0] += tt2 - tt1;   // alloc (incl. retries)
      t.timing[1] += tt1 - tt0;   // H2D
      t.timing[2] += tt3 - tt2;   // kernel (pack, gemms, scatter, assemble, fold; to the sync)
      t.timing[3] += tt4 - tt3;   // D2H
    }

    cub_check(cublasDestroy(h), "cublasDestroy");
    for (void *p : {(void *)dact, (void *)dX, (void *)dXc, (void *)dgk, (void *)dgkq, (void *)dGhat, (void *)dGtil, (void *)deps,
                    (void *)depsG, (void *)dgn, (void *)dfh, (void *)dfd1, (void *)dDsq, (void *)dDcb, (void *)dDqt,
                    (void *)ds1, (void *)ds3, (void *)dr1u, (void *)dr3u, (void *)dr3t, (void *)dR1U, (void *)dR1T,
                    (void *)dR3U, (void *)dR3T, (void *)dF, (void *)dFs, (void *)derr, (void *)dVt, (void *)dPj,
                    (void *)dQj, (void *)dBj, (void *)dgjT, (void *)dglT, (void *)dGh, (void *)dAU, (void *)dAT,
                    (void *)dM2, (void *)dA1, (void *)dA3, (void *)pVt, (void *)pGjT, (void *)pGlT, (void *)pGh,
                    (void *)pPj, (void *)pQj, (void *)pBj})
      cu_check(cudaFree(p), "cudaFree");
    return K;
  }

} // namespace methods::solvers::dynbse_cuda
