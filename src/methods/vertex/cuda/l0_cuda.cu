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
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>
#include <cuda_runtime.h>
#include "IO/AppAbort.hpp"
#include "methods/vertex/cuda/l0_cuda.cuh"
#include "methods/vertex/cuda/rung_cuda.cuh"

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

    // nca = the number of ACTIVE input components packed into Vt; act[il] = their global component index
    // c (0 = the constant, 1 + f np + a = node a of family f). A frequency-constant input packs one component.
    struct kdims { long nc, nR, np, ng, nca, blk, nk, np_fit; long const *act; };

    // ---- pack the ACTIVE components of X of K k-points into Vt(k; x, il, r, y) ----------------------
    // fold (the nu = 0 kernel): component c = 1 + a is the FOLDED family X.fam(0, a) + X.fam(1, a).
    __global__ void pack_kernel(kdims d, long ik0, long K, bool fold, cd const *__restrict__ Xfam,
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
          else if (fold) {                              // nu = 0: V_a = fam(0, a) + fam(1, a), the host kernel's order
            const long a = c - 1;
            const long i0 = ((a * nk + ik) * nc + x) * nc * nR + y * nR + r;
            v = Xfam[i0] + Xfam[i0 + np * nk * nc * nc * nR];
          } else {
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
          // the vectors dynbse.hpp's mulU / mulT act on
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

    // ---- the nu = 0 scatter: dynbse.hpp's l0_apply_cols on the (x, r, y) block c of every pole's output ----
    // grid (nca, K), one thread per element, the ng poles looped inside. which_pass 0: the j loop -- Q_j(a) (the
    // single poles: F1(nj) += Q/(ej - ea), F1(a) -= ..., or M2(nj) += Q at a == nj) and B_j(a) (the confluent
    // U_j^2 U_a: M2(nj) += B/(ej - ea), F1(nj) -= B/(ea - ej)^2, F1(a) += ..., or M3(nj) += B at a == nj);
    // 1: the l loop on R_l(a) (F1(nl) -= R/(el - ea), F1(a) += ..., or M2(nl) -= R at a == nl), Ball unused.
    // The constant component (c == 0) lands in the part-1 slots of F1 / M2 (= the host kernel's F1c / M2c):
    // F1c(nj) += Qc_j, M2c(nj) += Bc_j, F1c(nl) -= Rc_l. The a-indexed target F1(a) is reduced in registers over
    // the poles and added once (one atomic), the nj-indexed targets are atomics as in the shift kernel.
    __global__ void scatter_nu0_kernel(kdims d, long K, int which_pass, bool skip_cst,
                                       cd const *__restrict__ Qall, cd const *__restrict__ Ball,
                                       double const *__restrict__ eps, double const *__restrict__ epsG,
                                       long const *__restrict__ gnode,
                                       cd *__restrict__ F1, cd *__restrict__ M2, cd *__restrict__ M3) {
      const long il = blockIdx.x, kb = blockIdx.y;   // il: the packed position; c: the global folded component
      if (kb >= K) return;
      const long c = d.act[il];
      if (c == 0 && skip_cst) return;
      const long np = d.np, ng = d.ng, blk = d.blk, nca = d.nca;
      const long W = d.nc * nca * d.nR * d.nc, asz = 2 * np * blk;
      const int part = (c == 0) ? 1 : 0;
      const long abase = kb * asz + (long(part) * np) * blk;
      const long a = (c == 0) ? -1 : c - 1;
      const double ea = (a >= 0) ? eps[a] : 0.0;
      cd const *Qk = Qall + kb * ng * W;
      cd const *Bk = Ball + kb * ng * W;
      for (long e = threadIdx.x; e < blk; e += blockDim.x) {
        const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
        const long src = (x * nca + il) * d.nR * d.nc + rem;
        cd acc1 = make_cuDoubleComplex(0.0, 0.0);                     // F1(a) over the poles
        for (long pole = 0; pole < ng; ++pole) {
          const long nj = gnode[pole];
          const double ej = epsG[pole];
          const cd q = Qk[pole * W + src];
          cd *F1nj = &F1[abase + nj * blk + e], *M2nj = &M2[abase + nj * blk + e];
          if (which_pass == 0) {
            // ---- the j side: Q_j(a) = g_j^T V_a Ghat_j ----
            if (c == 0) {                                             // Qc_j -> F1c(nj)
              atomic_add(F1nj, q);
            } else if (a == nj) {
              atomic_add(M2nj, q);
            } else {
              const double w = 1.0 / (ej - ea);
              atomic_add(F1nj, scal(w, q));
              acc1 = acc1 - scal(w, q);
            }
            // ---- the confluent U_j^2 x U_a: B_j(a) = g_j^T V_a gkq_j^T ----
            const cd bq = Bk[pole * W + src];
            if (c == 0) {                                             // Bc_j -> M2c(nj)
              atomic_add(M2nj, bq);
            } else if (a == nj) {
              atomic_add(&M3[abase + nj * blk + e], bq);
            } else {
              const double dd = ea - ej;
              const double c2 = 1.0 / (ej - ea), c1 = 1.0 / (dd * dd);
              atomic_add(M2nj, scal(c2, bq));
              atomic_add(F1nj, neg(scal(c1, bq)));
              acc1 = acc1 + scal(c1, bq);
            }
          } else {
            // ---- the l side: R_l(a) = Gtil_l V_a gkq_l^T ----
            if (c == 0) {                                             // Rc_l -> -F1c(nl)
              atomic_add(F1nj, neg(q));
            } else if (a == nj) {
              atomic_add(M2nj, neg(q));
            } else {
              const double w = 1.0 / (ej - ea);
              atomic_add(F1nj, neg(scal(w, q)));
              acc1 = acc1 + scal(w, q);
            }
          }
        }
        if (a >= 0 && nonzero(acc1)) atomic_add(&F1[abase + a * blk + e], acc1);
      }
    }

    // ---- the nu = 0 assembly of dynbse.hpp's l0_apply_cols: F1 / F1c / M2 / M2c / M3 -> F, Fsum ---------
    // grid (np, K): one block per (node n, k); one thread per (x, r, y). Part 0 of F1 / M2 is the family's
    // (F1, M2), part 1 the constant's (F1c, M2c); M3 has part 0 only.
    __global__ void assemble_nu0_kernel(kdims d, long ik0, long K, bool sum_part1,
                                        double const *__restrict__ fhalf, double const *__restrict__ fd1,
                                        double const *__restrict__ fd2,
                                        cd const *__restrict__ Dsq, cd const *__restrict__ Dcb,
                                        cd const *__restrict__ F1, cd const *__restrict__ M2, cd const *__restrict__ M3,
                                        cd *__restrict__ Ffam, cd *__restrict__ Fsum, int *__restrict__ err) {
      const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk, asz = 2 * np * blk;
      const long n = blockIdx.x, kb = blockIdx.y;
      if (kb >= K) return;
      const long ik = ik0 + kb;
      for (long e = threadIdx.x; e < blk; e += blockDim.x) {
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        const long i0 = kb * asz + n * blk + e, i1 = kb * asz + (np + n) * blk + e;
        const cd u = F1[i0], uc = F1[i1], m = M2[i0], mc = M2[i1], m3 = M3[i0];
        const long f0n = (((0 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        const long f1n = (((1 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        const long fs = ((ik * nc + x) * nc + y) * nR + r;
        // the single poles: F.fam(0, n) += F1 + F1c; Fsum += fhalf F1 (+ fhalf F1c unless Cb_cst supplies it)
        atomic_add(&Ffam[f0n], u + uc);
        atomic_add(&Fsum[fs], scal(fhalf[n], u));
        if (sum_part1) atomic_add(&Fsum[fs], scal(fhalf[n], uc));
        // the double poles U_n^2 (the family's and the constant's): f' into the sum, Dsq into family 0 on the DLR
        // set, the S family (family 1) at a node outside it
        if (nonzero(m)) {
          atomic_add(&Fsum[fs], scal(fd1[n], m));
          if (n < d.np_fit) {
            for (long c = 0; c < np; ++c) {
              const cd dc = Dsq[n * np + c];
              if (!nonzero(dc)) continue;
              atomic_add(&Ffam[(((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r], dc * m);
            }
          } else {
            atomic_add(&Ffam[f1n], m);
          }
        }
        if (nonzero(mc)) {
          if (sum_part1) atomic_add(&Fsum[fs], scal(fd1[n], mc));
          if (n < d.np_fit) {
            for (long c = 0; c < np; ++c) {
              const cd dc = Dsq[n * np + c];
              if (!nonzero(dc)) continue;
              atomic_add(&Ffam[(((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r], dc * mc);
            }
          } else {
            atomic_add(&Ffam[f1n], mc);
          }
        }
        // the triple poles U_n^3: f''/2 into the sum, Dcb into family 0 (a node outside the DLR set is an error)
        if (nonzero(m3)) {
          if (n >= d.np_fit) { atomicOr(err, 1); continue; }
          atomic_add(&Fsum[fs], scal(0.5 * fd2[n], m3));
          for (long c = 0; c < np; ++c) {
            const cd dc = Dcb[n * np + c];
            if (!nonzero(dc)) continue;
            atomic_add(&Ffam[(((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r], dc * m3);
          }
        }
      }
    }

    // ---- the small-nu fold: T_a with |eps_a| >= ratio |nu| folded into the U family ------------
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

    // =================================================================================================================
    // THE FUSED OUTPUT-STATIONARY L0 PASSES. In the batched-gemm + scatter passes the per-pole nc x nc gemms are
    // memory-bound on the materialized Pj / Qj / Bj (ng K W elements each) and the scatter kernels are L2-atomic-bound
    // (every component adds into the same pole-indexed targets). Here one block per (r, k) owns every
    // accumulator element e = (x', r, y') of its slice -- no atomics -- and forms the per-pole products in registers:
    //   pass 0: P(y, x') = sum_x V(x, y) gk_j(x, x'),  Q(x', y') = sum_y Ghat_j(y, y') P(y, x'),
    //           B(x', y') = sum_y gkq_j(y', y) P(y, x')                      (the gemms Pj, Qj, Bj of the batched pass)
    //   pass 1: P(y, x') = sum_x V(x, y) Gtil_j(x', x),  R(x', y') = sum_y gkq_j(y', y) P(y, x')   (Pl, Rl)
    // Threads: G groups of S x nc slots (S = the power of two >= nc, the shuffle segment); thread (x', y') of a group
    // computes P(y = y', x') and gets P(y, x') for every y from its segment by shuffles. The groups split the
    // components of a chunk (CHG each). Per (chunk, pole j): the pole's tables and the chunk's partial-fraction
    // coefficients (coef_kernel: the scatter kernels' own expressions) in shared memory; the pole-indexed targets
    // (at nj = gnode[j]) accumulate over the group's components in registers, are summed over the groups in shared
    // memory and added once; the component-indexed targets (at the component's node a) accumulate over the poles in
    // registers and are added once per chunk, the groups in turn. Every term is the scatter's product with the
    // scatter's coefficient; only the summation order differs from the batched-gemm passes (rounding level).
    // =================================================================================================================
    constexpr int FZ_NCOEF = 6;

    // the partial-fraction coefficients per (row, pole j), rows: nu != 0 -- [0, np) the U_a components
    // {w (mulU, real), wt, wd (mulT on U_a)}, [np, 2np) the T_a components {wt, wd (mulU on T_a), ctl, cta, cul, cua};
    // nu = 0 -- [0, np) the folded components {w, c2, c1} (real). a == gnode(j) (confluent) entries are never read.
    __global__ void coef_kernel(bool nu0, long np, long ng, cd inu, double const *__restrict__ eps, double const *__restrict__ epsG,
                                cd *__restrict__ coef) {
      const long nrow = nu0 ? np : 2 * np;
      const long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
      if (i >= nrow * ng) return;
      const long row = i / ng, j = i % ng, a = row % np;
      const bool isT = (not nu0) and row >= np;
      const double ea = eps[a], ej = epsG[j];
      cd *o = coef + i * FZ_NCOEF;
      for (int k = 0; k < FZ_NCOEF; ++k) o[k] = make_cuDoubleComplex(0.0, 0.0);
      if (ea == ej) return;                               // the confluent case (a == nj): handled without coefficients
      if (nu0) {
        const double dd = ea - ej;
        o[0] = real(1.0 / (ej - ea));                     // w  (Q_j and R_l)
        o[1] = real(1.0 / (ej - ea));                     // c2 (B_j -> M2)
        o[2] = real(1.0 / (dd * dd));                     // c1 (B_j -> F1)
      } else if (not isT) {
        o[0] = real(1.0 / (ej - ea));                     // mulU on U_a: w
        const cd w = real(1.0 / (ej - ea));               // mulT on U_a
        const cd dd = real(ea - ej) + inu;
        const cd wd = w / dd;
        const cd wt = w - inu * wd;
        o[1] = wt; o[2] = wd;
      } else {
        const cd w = real(1.0 / (ea - ej));               // mulU on T_a
        const cd dd = real(ej - ea) + inu;
        const cd wd = w / dd;
        const cd wt = w - inu * wd;
        o[0] = wt; o[1] = wd;
        const double g = ej - ea;                         // mulT on T_a
        const cd w2 = real(1.0 / (g * g));
        const cd dla = real(ej - ea) + inu, dal = real(ea - ej) + inu;
        const cd wla = w2 / dla, wal = w2 / dal;
        o[2] = w2 - inu * wal;                            // ctl
        o[3] = w2 - inu * wla;                            // cta
        o[4] = wal - wla;                                 // cul
        o[5] = wla - wal;                                 // cua
      }
    }

    // the accumulator targets of a pass (nu != 0: AU AT M2 A1 A3 of part 0 at nj, AU AT of part 1 at nj;
    // nu = 0: F1 M2 M3 of part 0 at nj, F1 M2 of part 1 at nj -- F1 lives in AU, M3 in A3)
    template <bool NU0>
    struct fz_acc { static constexpr int n = NU0 ? 5 : 7; };

    // MINB blocks of 256 threads per SM: caps the register budget. The kernel is latency-bound, not FP64-bound; without the
    // cap a high register count leaves one resident block per SM and most cycles without an eligible warp.
    template <bool NU0, int CHG, int MINB>
    __global__ void __launch_bounds__(256, MINB)
    fused_pass_kernel(kdims d, long ik0, long K, int pass, cd inu, bool skip_cst, int S, int G,
                                      cd const *__restrict__ Vt, cd const *__restrict__ gk, cd const *__restrict__ gkq,
                                      cd const *__restrict__ Ghat, cd const *__restrict__ Gtil, cd const *__restrict__ coef,
                                      long const *__restrict__ gnode,
                                      cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2, cd *__restrict__ A1,
                                      cd *__restrict__ A3) {
      extern __shared__ double2 fz_smem[];
      constexpr int NACC = fz_acc<NU0>::n;
      const long nc = d.nc, nR = d.nR, np = d.np, ng = d.ng, nca = d.nca, nk = d.nk, blk = d.blk, nc2 = nc * nc;
      const long r = blockIdx.x, kb = blockIdx.y;
      if (kb >= K) return;
      const long ik = ik0 + kb;
      const int slots = S * int(nc);
      const int t = int(threadIdx.x);
      const bool inb = (t < G * slots);                   // the padding threads (to a whole warp) only take part in the shuffles
      const int g = inb ? t / slots : 0, q = inb ? t % slots : 0, xp = q / S, yp = q % S;
      const bool act = inb and (yp < nc);
      const long e = act ? (long(xp) * nR + r) * nc + yp : 0;
      const long W = nc * nca * nR * nc, asz = 2 * np * blk;
      const int CH = G * CHG;
      cd *Vs = fz_smem;                                   // (CH, nc, nc): the chunk's V(x, y) at this r
      cd *T1 = Vs + CH * nc2;                             // gk_j (pass 0) | Gtil_j (pass 1)
      cd *T2 = T1 + nc2;                                  // Ghat_j (pass 0)
      cd *T3 = T2 + nc2;                                  // gkq_j, TRANSPOSED (bank-conflict-free reads)
      cd *T4 = T3 + nc2;                                  // Gtil_j (the merged pass 2)
      cd *Cf = T4 + nc2;                                  // (CH, FZ_NCOEF): the chunk's coefficients at pole j
      cd *Red = Cf + CH * FZ_NCOEF;                       // (NACC, G, slots): the pole-indexed partial sums
      cd const *Vk = Vt + kb * W;
      const cd zero = make_cuDoubleComplex(0.0, 0.0);
      const cd *Ghk = Ghat + (ik * ng) * nc2, *Gtk = Gtil + (ik * ng) * nc2;
      for (long il0 = 0; il0 < nca; il0 += CH) {
        const int nch = int(min(long(CH), nca - il0));
        __syncthreads();                                  // the previous chunk's reads of Vs are done
        for (long i = t; i < long(nch) * nc2; i += blockDim.x) {
          const long cl = i / nc2, xy = i % nc2, x = xy / nc, y = xy % nc;
          Vs[i] = Vk[((x * nca + il0 + cl) * nR + r) * nc + y];
        }
        // this thread's components: cl = g CHG + m
        int cm[CHG], am[CHG];                              // component index (<= 2 np) and its node: 32-bit (registers)
        bool vm[CHG];
#pragma unroll
        for (int m = 0; m < CHG; ++m) {
          const int cl = g * CHG + m;
          vm[m] = inb and (cl < nch);
          const int c = vm[m] ? int(d.act[il0 + cl]) : 0;
          if (vm[m] and c == 0 and skip_cst) vm[m] = false;
          cm[m] = c;
          am[m] = (c == 0) ? -1 : (NU0 ? c - 1 : ((c <= int(np)) ? c - 1 : c - 1 - int(np)));
        }
        cd aU[CHG], aT[CHG];
#pragma unroll
        for (int m = 0; m < CHG; ++m) { aU[m] = zero; aT[m] = zero; }
        for (long j = 0; j < ng; ++j) {
          __syncthreads();                                // Vs loaded; the previous pole's reads of T / Cf / Red done
          for (long i = t; i < nc2; i += blockDim.x) {
            if (pass != 1) { T1[i] = gk[(j * nk + ik) * nc2 + i]; T2[i] = Ghk[j * nc2 + i]; }
            else T1[i] = Gtk[j * nc2 + i];
            if (pass == 2) T4[i] = Gtk[j * nc2 + i];
            T3[(i % nc) * nc + i / nc] = gkq[(j * nk + ik) * nc2 + i];   // transposed: T3[y nc + y'] = gkq_j(y', y)
          }
          for (long i = t; i < long(nch) * FZ_NCOEF; i += blockDim.x) {
            const long cl = i / FZ_NCOEF, kk = i % FZ_NCOEF, c = d.act[il0 + cl];
            const long row = (c == 0) ? -1 : (NU0 ? c - 1 : c - 1);   // nu != 0: U_a rows [0, np), T_a rows [np, 2np) = c - 1
            Cf[i] = (row >= 0) ? coef[(row * ng + j) * FZ_NCOEF + kk] : zero;
          }
          __syncthreads();
          const long nj = gnode[j];
          cd p[NACC];
#pragma unroll
          for (int k = 0; k < NACC; ++k) p[k] = zero;
#pragma unroll
          for (int m = 0; m < CHG; ++m) {
            const int cl = g * CHG + m;
            const int clr = (cl < nch) ? cl : 0;          // a valid smem row for the (discarded) padding components
            // P(y = y', x'): lane y' of the segment (pass 1: the G-tilde P; the merged pass 2 forms both, P and P1)
            cd P = zero, P1 = zero;
            if (act) {
              cd const *Vc = Vs + clr * nc2;
              if (pass != 1) for (long x = 0; x < nc; ++x) P = P + Vc[x * nc + yp] * T1[x * nc + xp];
              else           for (long x = 0; x < nc; ++x) P = P + Vc[x * nc + yp] * T1[xp * nc + x];
              if (pass == 2) for (long x = 0; x < nc; ++x) P1 = P1 + Vc[x * nc + yp] * T4[xp * nc + x];
            }
            cd Qv = zero, Bv = zero, Xv = zero;
            for (long y = 0; y < nc; ++y) {               // every lane of the warp shuffles (no divergence)
              const double px = __shfl_sync(0xffffffffu, P.x, int(y), S);
              const double py = __shfl_sync(0xffffffffu, P.y, int(y), S);
              const cd Py = make_cuDoubleComplex(px, py);
              cd P1y = zero;
              if (pass == 2) {
                const double qx = __shfl_sync(0xffffffffu, P1.x, int(y), S);
                const double qy = __shfl_sync(0xffffffffu, P1.y, int(y), S);
                P1y = make_cuDoubleComplex(qx, qy);
              }
              if (act) {
                if (pass != 1) Qv = Qv + T2[y * nc + yp] * Py;
                Bv = Bv + T3[y * nc + yp] * Py;          // B (pass 0, 2) | R (pass 1): gkq_j(y', y) P(y, x')
                if (pass == 2) Xv = Xv + T3[y * nc + yp] * P1y;   // R of the merged pass
              }
            }
            if (not (act and vm[m])) continue;
            // the merged pass: the scatter is linear in the pass-0 pair (Q, B) and in pass 1's R with the same coefficients
            // and targets (pass 1 enters as vU = -R, vT = i nu R; at nu = 0 as Q -> -R), so one scatter of the sums
            if (NU0 and pass == 2) Qv = Qv - Xv;
            const long c = cm[m], a = am[m];
            cd const *cf = Cf + clr * FZ_NCOEF;
            if (NU0) {
              if (pass != 1) {
                if (c == 0) { p[3] = p[3] + Qv; p[4] = p[4] + Bv; }                    // F1c(nj), M2c(nj)
                else if (a == nj) { p[1] = p[1] + Qv; p[2] = p[2] + Bv; }             // M2(nj), M3(nj)
                else {
                  const double w = cf[0].x, c2 = cf[1].x, c1 = cf[2].x;
                  p[0] = p[0] + scal(w, Qv); aU[m] = aU[m] - scal(w, Qv);
                  p[1] = p[1] + scal(c2, Bv); p[0] = p[0] + neg(scal(c1, Bv)); aU[m] = aU[m] + scal(c1, Bv);
                }
              } else {
                if (c == 0) p[3] = p[3] + neg(Bv);                                    // F1c(nl) -= R
                else if (a == nj) p[1] = p[1] + neg(Bv);                              // M2(nl) -= R
                else { const double w = cf[0].x; p[0] = p[0] + neg(scal(w, Bv)); aU[m] = aU[m] + scal(w, Bv); }
              }
            } else {
              const cd vU = (pass == 0) ? Qv : (pass == 1) ? neg(Bv) : Qv - Xv;       // U_j . Q_j | -U_l . R_l | both
              const cd vT = (pass == 0) ? Bv : (pass == 1) ? inu * Bv : Bv + inu * Xv; // T_j . B_j | i nu T_l . R_l | both
              if (c == 0) { p[5] = p[5] + vU; p[6] = p[6] + vT; }                     // AU(nj), AT(nj) of part 1
              else if (c <= np) {                                                     // U_a
                if (a == nj) { p[2] = p[2] + vU; p[3] = p[3] + vT; }                  // M2(nj); A1(nj) (U_l T_l)
                else {
                  const double w = cf[0].x;
                  p[0] = p[0] + scal(w, vU); aU[m] = aU[m] - scal(w, vU);
                  const cd wt = cf[1], wd = cf[2];
                  p[1] = p[1] + wt * vT; aU[m] = aU[m] - wd * vT; p[0] = p[0] + wd * vT;
                }
              } else {                                                                 // T_a
                if (a == nj) { p[3] = p[3] + vU; p[4] = p[4] + vT; }                  // A1(nj) (U_j T_j); A3(nj)
                else {
                  const cd wt = cf[0], wd = cf[1];
                  aT[m] = aT[m] + wt * vU; p[0] = p[0] + neg(wd * vU); aU[m] = aU[m] + wd * vU;
                  p[1] = p[1] + cf[2] * vT; aT[m] = aT[m] + cf[3] * vT;
                  p[0] = p[0] + cf[4] * vT; aU[m] = aU[m] + cf[5] * vT;
                }
              }
            }
          }
          // the pole-indexed targets: sum over the groups, one add per element
          if (inb) {
#pragma unroll
            for (int k = 0; k < NACC; ++k) Red[(k * G + g) * slots + q] = p[k];
          }
          __syncthreads();
          if (g == 0 and act) {
            const long i0 = kb * asz + nj * blk + e, i1 = kb * asz + (np + nj) * blk + e;
#pragma unroll
            for (int k = 0; k < NACC; ++k) {
              cd s2 = zero;
              for (int gg = 0; gg < G; ++gg) s2 = s2 + Red[(k * G + gg) * slots + q];
              if (not nonzero(s2)) continue;
              cd *dst;
              if (NU0) dst = (k == 0) ? &AU[i0] : (k == 1) ? &M2[i0] : (k == 2) ? &A3[i0] : (k == 3) ? &AU[i1] : &M2[i1];
              else     dst = (k == 0) ? &AU[i0] : (k == 1) ? &AT[i0] : (k == 2) ? &M2[i0] : (k == 3) ? &A1[i0]
                           : (k == 4) ? &A3[i0] : (k == 5) ? &AU[i1] : &AT[i1];
              *dst = *dst + s2;
            }
          }
        }
        // the component-indexed targets (at the component's node a, part 0): the groups in turn
        for (int gg = 0; gg < G; ++gg) {
          __syncthreads();
          if (inb and g == gg and act) {
#pragma unroll
            for (int m = 0; m < CHG; ++m) {
              if (not vm[m] or am[m] < 0) continue;
              const long ia = kb * asz + am[m] * blk + e;
              if (nonzero(aU[m])) AU[ia] = AU[ia] + aU[m];
              if (not NU0 and nonzero(aT[m])) AT[ia] = AT[ia] + aT[m];
            }
          }
        }
      }
    }

    // ---- the nu = 0 assembly without its Dsq / Dcb re-expansions (those are one gemm per k-batch, below): one thread
    // per (k, e) looping the nodes n, so Fsum is summed in a register and Ffam(0 | 1, n) is written by one thread each
    __global__ void assemble_nu0_elem_kernel(kdims d, long ik0, long K, bool sum_part1,
                                             double const *__restrict__ fhalf, double const *__restrict__ fd1,
                                             double const *__restrict__ fd2,
                                             cd const *__restrict__ F1, cd const *__restrict__ M2, cd const *__restrict__ M3,
                                             cd *__restrict__ Ffam, cd *__restrict__ Fsum, int *__restrict__ err) {
      const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk, asz = 2 * np * blk;
      const long kb = blockIdx.y;
      if (kb >= K) return;
      const long ik = ik0 + kb;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < blk; e += long(gridDim.x) * blockDim.x) {
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        const long fs = ((ik * nc + x) * nc + y) * nR + r;
        cd sacc = make_cuDoubleComplex(0.0, 0.0);
        for (long n = 0; n < np; ++n) {
          const long i0 = kb * asz + n * blk + e, i1 = kb * asz + (np + n) * blk + e;
          const cd u = F1[i0], uc = F1[i1], m = M2[i0], mc = M2[i1], m3 = M3[i0];
          const long f0n = (((0 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
          const long f1n = (((1 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
          if (nonzero(u) or nonzero(uc)) Ffam[f0n] = Ffam[f0n] + (u + uc);
          sacc = sacc + scal(fhalf[n], u);
          if (sum_part1) sacc = sacc + scal(fhalf[n], uc);
          if (nonzero(m)) {
            sacc = sacc + scal(fd1[n], m);
            if (n >= d.np_fit) Ffam[f1n] = Ffam[f1n] + m;
          }
          if (nonzero(mc)) {
            if (sum_part1) sacc = sacc + scal(fd1[n], mc);
            if (n >= d.np_fit) Ffam[f1n] = Ffam[f1n] + mc;
          }
          if (nonzero(m3)) {
            if (n >= d.np_fit) { atomicOr(err, 1); continue; }
            sacc = sacc + scal(0.5 * fd2[n], m3);
          }
        }
        if (nonzero(sacc)) Fsum[fs] = Fsum[fs] + sacc;
      }
    }
    // T(kb; c, e) (the gemm's output, (np, blk) per k) added into Ffam(0, c, ik, x, y, r)
    __global__ void nu0_reexp_add_kernel(kdims d, long ik0, long K, cd const *__restrict__ T, cd *__restrict__ Ffam) {
      const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk;
      const long kb = blockIdx.y;
      if (kb >= K) return;
      const long ik = ik0 + kb, tot = np * blk;
      for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < tot; i += long(gridDim.x) * blockDim.x) {
        const long c = i / blk, e = i % blk;
        const cd v = T[kb * tot + i];
        if (not nonzero(v)) continue;
        const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
        const long f = (((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        Ffam[f] = Ffam[f] + v;
      }
    }

    // host helpers shared by run_l0 and the resident plan
    inline int fz_segment(long nc) { int S = 1; while (S < nc) S <<= 1; return S; }
    inline size_t fz_smem_bytes(bool nu0, long nc, int S, int G, int CHG) {
      const long nc2 = nc * nc, CH = long(G) * CHG, nacc = nu0 ? 5 : 7;
      return size_t(CH * nc2 + 4 * nc2 + CH * FZ_NCOEF + nacc * long(G) * S * nc) * sizeof(cd);
    }
    // the fused kernel's variants: cfg = 10 CHG + MINB (components per thread x minimum 256-thread blocks per SM). The
    // default 44 (4 blocks / SM, at the price of register spills) is the fastest variant on H100-class devices; the
    // fz_bench switch times every variant on the current device.
    template <bool NU0, int CHG, int MINB>
    void fz_launch(dim3 grid, unsigned threads, kdims kd, long ik0, long Kb, int pass, cd inu, bool skip_cst, int S, int G,
                   cd const *Vt, cd const *gk, cd const *gkq, cd const *Ghat, cd const *Gtil, cd const *coef, long const *gnode,
                   cd *AU, cd *AT, cd *M2, cd *A1, cd *A3) {
      const size_t sm = fz_smem_bytes(NU0, kd.nc, S, G, CHG);
      if (sm > 48 * 1024)
        cu_check(cudaFuncSetAttribute(fused_pass_kernel<NU0, CHG, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sm)), "fz attr");
      fused_pass_kernel<NU0, CHG, MINB><<<grid, threads, sm>>>(kd, ik0, Kb, pass, inu, skip_cst, S, G, Vt, gk, gkq, Ghat, Gtil, coef,
                                                              gnode, AU, AT, M2, A1, A3);
    }
    constexpr int FZ_CFGS[] = {81, 82, 42, 43, 44};
    template <bool NU0>
    void fz_dispatch(int cfg, dim3 grid, unsigned threads, kdims kd, long ik0, long Kb, int pass, cd inu, bool skip_cst, int S, int G,
                     cd const *Vt, cd const *gk, cd const *gkq, cd const *Ghat, cd const *Gtil, cd const *coef, long const *gnode,
                     cd *AU, cd *AT, cd *M2, cd *A1, cd *A3) {
#define FZ_CASE(C, M) case 10 * C + M: \
      fz_launch<NU0, C, M>(grid, threads, kd, ik0, Kb, pass, inu, skip_cst, S, G, Vt, gk, gkq, Ghat, Gtil, coef, gnode, AU, AT, M2, A1, A3); break;
      switch (cfg) {
        FZ_CASE(8, 1) FZ_CASE(8, 2) FZ_CASE(4, 2) FZ_CASE(4, 3) FZ_CASE(4, 4)
        default: APP_ABORT(std::string(" l0_cuda: unknown fused-L0 variant (dynbse_l0_fz_cfg: 81 | 82 | 42 | 43 | 44)."));
      }
#undef FZ_CASE
    }
    // both fused passes of one k-batch: merge = true runs them as ONE pass (pass 2: the pass-1 products formed alongside
    // the pass-0 ones, one scatter of the sums -- half the scatter, the syncs and the target writes). bench: time every
    // variant once on this k-batch (the first call per nu class; the accumulators are re-zeroed after) and print the table.
    void fused_passes(kdims kd, long ik0, long Kb, bool nu0, cd inu, bool skip_cst, cd const *Vt, cd const *gk, cd const *gkq,
                      cd const *Ghat, cd const *Gtil, cd const *coef, long const *gnode, cd *AU, cd *AT, cd *M2, cd *A1, cd *A3,
                      bool merge = true, int cfg = 44, bool bench = false) {
      const int S = fz_segment(kd.nc);
      if (S > 32) APP_ABORT(std::string(" l0_cuda: the fused L0 passes need nc <= 32."));
      const int slots = S * int(kd.nc);
      const int G = std::max(1, 256 / slots);
      const unsigned threads = unsigned(((G * slots + 31) / 32) * 32);   // whole warps: the shuffles take the full mask
      if (threads > 256) APP_ABORT(std::string(" l0_cuda: the fused L0 passes are built for <= 256 threads (nc <= 16); set vertex_debug dynbse_l0_fused = 0."));
      const dim3 grid(unsigned(kd.nR), unsigned(Kb), 1u);
      auto run = [&](int c) {
        for (int pass = (merge ? 2 : 0); pass < (merge ? 3 : 2); ++pass) {
          if (nu0) fz_dispatch<true>(c, grid, threads, kd, ik0, Kb, pass, inu, skip_cst, S, G, Vt, gk, gkq, Ghat, Gtil, coef, gnode, AU, AT, M2, A1, A3);
          else     fz_dispatch<false>(c, grid, threads, kd, ik0, Kb, pass, inu, skip_cst, S, G, Vt, gk, gkq, Ghat, Gtil, coef, gnode, AU, AT, M2, A1, A3);
          launch_check(pass == 0 ? "fused pass 0" : pass == 1 ? "fused pass 1" : "fused merged pass");
        }
      };
      static bool benched[2] = {false, false};
      // a representative batch: >= 16 active components (the first batch of a run can be constant-only)
      if (bench and not benched[nu0 ? 1 : 0] and kd.nca >= std::min<long>(16, 1 + kd.np)) {
        benched[nu0 ? 1 : 0] = true;
        const size_t accb = size_t(Kb) * size_t(2 * kd.np * kd.blk) * sizeof(cd);
        auto zero = [&] { for (cd *p : {AU, AT, M2, A1, A3}) cu_check(cudaMemsetAsync(p, 0, accb, 0), "fz bench zero"); };
        cudaEvent_t e0, e1;
        cu_check(cudaEventCreate(&e0), "fz bench ev");
        cu_check(cudaEventCreate(&e1), "fz bench ev");
        std::string line;
        for (int c : FZ_CFGS) {
          float best = 1e30f;
          for (int rep = 0; rep < 2; ++rep) {
            zero();
            cu_check(cudaEventRecord(e0, 0), "fz bench rec");
            run(c);
            cu_check(cudaEventRecord(e1, 0), "fz bench rec");
            cu_check(cudaEventSynchronize(e1), "fz bench sync");
            float ms = 0.0f;
            cu_check(cudaEventElapsedTime(&ms, e0, e1), "fz bench time");
            best = std::min(best, ms);
          }
          char b[64];
          std::snprintf(b, sizeof(b), " %d: %.2f ms", c, best);
          line += b;
        }
        std::fprintf(stderr, "  [l0 fused bench] nu0 %d, nc %ld nR %ld np %ld ng %ld nca %ld K %ld, merge %d -- cfg (10 CHG + MINB):%s\n",
                     int(nu0), kd.nc, kd.nR, kd.np, kd.ng, kd.nca, Kb, int(merge), line.c_str());
        cu_check(cudaEventDestroy(e0), "fz bench ev");
        cu_check(cudaEventDestroy(e1), "fz bench ev");
        zero();
      }
      run(cfg);
    }
    void build_coef(bool nu0, long np, long ng, cd inu, double const *eps, double const *epsG, cd *coef) {
      const long n = (nu0 ? np : 2 * np) * ng;
      coef_kernel<<<unsigned((n + 255) / 256), 256>>>(nu0, np, ng, inu, eps, epsG, coef);
      launch_check("coef_kernel");
    }
    // the nu = 0 assembly with the Dsq / Dcb re-expansions as gemms: T(kb; c, e) = sum_{n < np_fit} [Dsq(n, c) (M2 + M2c)(n, e)
    // + Dcb(n, c) M3(n, e)] (column-major (blk x np) per k = the row-major (np, blk) block), then added into Ffam(0, c)
    void assemble_nu0_gemm(cublasHandle_t h, kdims kd, long ik0, long Kb, bool sum_part1, double const *fh, double const *fd1,
                           double const *fd2, cd const *Dsq, cd const *Dcb, cd const *F1, cd const *M2, cd const *M3, cd *T,
                           cd *Ffam, cd *Fsum, int *err) {
      const long np = kd.np, blk = kd.blk, npf = kd.np_fit, asz = 2 * np * blk;
      assemble_nu0_elem_kernel<<<dim3(unsigned((blk + 255) / 256), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, sum_part1, fh, fd1, fd2,
                                                                                            F1, M2, M3, Ffam, Fsum, err);
      launch_check("assemble_nu0_elem");
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      // part 0 (the family's M2), part 1 (the constant's M2c), M3 (part 0) with Dcb
      cub_check(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, int(blk), int(np), int(npf), &one, M2, int(blk), (long long)asz,
                                          Dsq, int(np), 0LL, &zero, T, int(blk), (long long)(np * blk), int(Kb)), "reexp M2");
      cub_check(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, int(blk), int(np), int(npf), &one, M2 + np * blk, int(blk),
                                          (long long)asz, Dsq, int(np), 0LL, &one, T, int(blk), (long long)(np * blk), int(Kb)), "reexp M2c");
      cub_check(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, int(blk), int(np), int(npf), &one, M3, int(blk), (long long)asz,
                                          Dcb, int(np), 0LL, &one, T, int(blk), (long long)(np * blk), int(Kb)), "reexp M3");
      nu0_reexp_add_kernel<<<dim3(unsigned(std::min<long>((np * blk + 255) / 256, 4096)), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, T, Ffam);
      launch_check("nu0_reexp_add");
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

    // ---- the driver shared by the two kernels: nu0 = false the shift kernel (inu != 0), true the nu = 0 kernel ----
    // The uploads, the k-batch sizing with its allocation retry, the pack / pole / gemm passes and the timing split
    // are common; the scatter, the assembly and the fold differ. In nu0 mode the shift tables, Dqt and the fold are
    // not read (uploaded as empty), fd2 is; the components are the folded ones (nca <= 1 + np).
    long run_l0(l0_dims const &d, l0_tables const &t, cplx *Ffam_h, cplx *Fsum_h, double free_bytes, bool nu0) {
    const long np = d.np, nk = d.nk, nc = d.nc, ng = d.ng, nR = d.nR, nca = d.nca;
    const long blk = nc * nR * nc, W = nc * nca * nR * nc, nc2 = nc * nc;
    const size_t asz = size_t(2 * np) * size_t(blk);
    const size_t nF = size_t(2 * np) * nk * nc * nc * nR, nFs = size_t(nk) * nc * nc * nR;
    const long nca_max = nu0 ? 1 + np : 1 + 2 * np;
    if (nk == 0 || np == 0 || nca == 0) return 0;
    if (nca < 1 || nca > nca_max || t.act == nullptr)
      APP_ABORT(std::string(" l0_cuda: the active-component list is missing or out of range (nca = ") + std::to_string(nca) + ").");
    if (nu0 && t.fd2 == nullptr) APP_ABORT(std::string(" l0_cuda: the nu = 0 kernel needs the fd2 table."));
    const size_t nsh1 = nu0 ? 0 : size_t(np), nsh2 = nu0 ? 0 : size_t(np) * np;   // the shift tables' upload sizes

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
    double *dfd2 = upload(t.fd2, nu0 ? size_t(np) : 0, "fd2");
    cd *dDsq = upload_c(t.Dsq, size_t(np) * np, "Dsq");
    cd *dDcb = upload_c(t.Dcb, size_t(np) * np, "Dcb");
    cd *dDqt = upload_c(t.Dqt, nsh2, "Dqt");
    cd *ds1 = upload_c(t.s1, nsh1, "s1"), *ds3 = upload_c(t.s3, nsh1, "s3");
    cd *dr1u = upload_c(t.r1u, nsh1, "r1u"), *dr3u = upload_c(t.r3u, nsh1, "r3u");
    cd *dr3t = upload_c(t.r3t, nsh1, "r3t");
    cd *dR1U = upload_c(t.R1U, nsh2, "R1U"), *dR1T = upload_c(t.R1T, nsh2, "R1T");
    cd *dR3U = upload_c(t.R3U, nsh2, "R3U"), *dR3T = upload_c(t.R3T, nsh2, "R3T");
    cd *dF = alloc<cd>(nF, "F"), *dFs = alloc<cd>(nFs, "Fsum");
    cu_check(cudaMemsetAsync(dF, 0, nF * sizeof(cd), 0), "memset F");
    cu_check(cudaMemsetAsync(dFs, 0, nFs * sizeof(cd), 0), "memset Fsum");
    int *derr = alloc<int>(1, "err");
    cu_check(cudaMemsetAsync(derr, 0, sizeof(int), 0), "memset err");
    cu_check(cudaDeviceSynchronize(), "uploads");
    const double tt1 = dnow();                       // end of the fixed uploads (H2D)

    // ---- the k-batch: the largest K whose working set fits the memory we were given --------------
    // per k: Vt + 3 ng W (P/Q/B) + 5 accumulators + 3 ng nc^2 pole operands, 16 B each; the fused passes keep no P/Q/B
    // and no pole operands, the nu = 0 gemm assembly adds its (np, blk) re-expansion block
    const bool fused = (t.fused != 0), merge = (t.fused == 2), asmg = nu0 and (t.asm_gemm != 0);
    const double per_k = double(W + (fused ? 0 : 3 * ng * W) + 5 * long(asz) + (fused ? 0 : 3 * ng * nc2) +
                                (asmg ? np * blk : 0)) * 16.0;
    const double fixed = double(nF * 2 + nFs * 2 + size_t(2 * ng) * nk * nc2 * 2 + 3 * size_t(np) * np) * 16.0;
    const double budget = 0.85 * free_bytes - fixed;     // 15 % held back for the cuBLAS workspace
    long K = std::max(1L, std::min(nk, long(budget / per_k)));
    if (budget < per_k) K = 1;                           // one k must fit; the allocation will say if not

    // The per-batch working set, allocated with a RETRY: the budget above assumes an exclusive device, but
    // several ranks may share one GPU (ranks that size their batches from the same cudaMemGetInfo at the same
    // instant can over-commit it). On a failed allocation everything of the attempt is freed and K is halved,
    // down to K = 1, which must fit (abort otherwise).
    cd *dVt = nullptr, *dPj = nullptr, *dQj = nullptr, *dBj = nullptr, *dgjT = nullptr, *dglT = nullptr, *dGh = nullptr;
    cd *dAU = nullptr, *dAT = nullptr, *dM2 = nullptr, *dA1 = nullptr, *dA3 = nullptr, *dT = nullptr;
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
                         (void **)&dGh, (void **)&dAU, (void **)&dAT, (void **)&dM2, (void **)&dA1, (void **)&dA3, (void **)&dT,
                         (void **)&pVt, (void **)&pGjT, (void **)&pGlT, (void **)&pGh, (void **)&pPj, (void **)&pQj,
                         (void **)&pBj})
          if (*p != nullptr) { (void)cudaFree(*p); *p = nullptr; }
      };
      const long K_first = K;
      bool ok = false;
      while (true) {
        nptr = size_t(K) * size_t(ng);
        const size_t bW = size_t(K) * W * sizeof(cd), bP = fused ? 16 : size_t(K) * ng * W * sizeof(cd);
        const size_t bG = fused ? 16 : size_t(K) * ng * nc2 * sizeof(cd), bA = size_t(K) * asz * sizeof(cd);
        const size_t bp = fused ? 16 : nptr * sizeof(cd *), bT = asmg ? size_t(K) * np * blk * sizeof(cd) : 16;
        ok = try_alloc((void **)&dVt, bW) and try_alloc((void **)&dPj, bP) and try_alloc((void **)&dQj, bP) and
             try_alloc((void **)&dBj, bP) and try_alloc((void **)&dgjT, bG) and try_alloc((void **)&dglT, bG) and
             try_alloc((void **)&dGh, bG) and try_alloc((void **)&dAU, bA) and try_alloc((void **)&dAT, bA) and
             try_alloc((void **)&dM2, bA) and try_alloc((void **)&dA1, bA) and try_alloc((void **)&dA3, bA) and
             try_alloc((void **)&dT, bT) and
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
    if (not fused) {
      fill_ptrs<<<unsigned((nptr + 255) / 256), 256>>>(K, ng, W, nc2, dVt, dgjT, dglT, dGh, dPj, dQj, dBj,
                                                       pVt, pGjT, pGlT, pGh, pPj, pQj, pBj);
      launch_check("fill_ptrs");
    }

    cublasHandle_t h;
    cub_check(cublasCreate(&h), "cublasCreate");
    const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const cd inu = make_cuDoubleComplex(t.inu.real(), t.inu.imag());
    const int Ncols = int(nca * nR * nc), Mrows = int(nc * nca * nR);
    cd *dcoef = nullptr;
    if (fused) {
      dcoef = alloc<cd>(size_t(nu0 ? np : 2 * np) * ng * FZ_NCOEF, "coef");
      build_coef(nu0, np, ng, inu, deps, depsG, dcoef);
    }

    for (long ik0 = 0; ik0 < nk; ik0 += K) {
      const long Kb = std::min(K, nk - ik0);
      const int nb = int(Kb * ng);
      for (cd *p : {dAU, dAT, dM2, dA1, dA3})              // at nu = 0 only AU, M2, A3 are written and read
        if (not nu0 or (p != dAT and p != dA1)) cu_check(cudaMemsetAsync(p, 0, size_t(Kb) * asz * sizeof(cd), 0), "memset acc");
      pack_kernel<<<dim3(64u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, nu0, dX, dXc, dVt);
      launch_check("pack_kernel");
      if (fused) {
        fused_passes(kd, ik0, Kb, nu0, inu, t.skip_cst, dVt, dgk, dgkq, dGhat, dGtil, dcoef, dgn, dAU, dAT, dM2, dA1, dA3, merge,
                     t.fz_cfg, t.fz_bench != 0);
      } else {
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
      if (nu0)
        scatter_nu0_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 0, t.skip_cst, dQj, dBj,
                                                                           deps, depsG, dgn, dAU, dM2, dA3);
      else
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
      if (nu0)
        scatter_nu0_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 1, t.skip_cst, dQj, dQj,
                                                                           deps, depsG, dgn, dAU, dM2, dA3);
      else
        scatter_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 1, inu, t.skip_cst, dQj, dQj,
                                                                       deps, depsG, dgn, dAU, dAT, dM2, dA1, dA3);
      launch_check("scatter_kernel pass 1");
      }
      if (nu0 and asmg)
        assemble_nu0_gemm(h, kd, ik0, Kb, t.sum_part1, dfh, dfd1, dfd2, dDsq, dDcb, dAU, dM2, dA3, dT, dF, dFs, derr);
      else if (nu0)
        assemble_nu0_kernel<<<dim3(unsigned(np), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, t.sum_part1, dfh, dfd1, dfd2,
                                                                           dDsq, dDcb, dAU, dM2, dA3, dF, dFs, derr);
      else
        assemble_kernel<<<dim3(unsigned(np), 2u, unsigned(Kb)), 256>>>(kd, ik0, Kb, t.sum_part1, dfh, dfd1, dDsq,
                                                                       ds1, ds3, dr1u, dr3u, dr3t, dR1U, dR1T, dR3U, dR3T,
                                                                       dAU, dAT, dM2, dA1, dA3, dF, dFs, derr);
      launch_check("assemble_kernel");
      if (t.tfold > 0.0 && !nu0) {
        tfold_kernel<<<dim3(unsigned(np), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, inu, t.tfold, deps, dDsq, dDcb, dDqt, dF);
        launch_check("tfold_kernel");
      }
    }
    cu_check(cudaDeviceSynchronize(), "l0 batches");
    const double tt3 = dnow();                       // end of the batches (kernel)
    int err = 0;
    cu_check(cudaMemcpy(&err, derr, sizeof(int), cudaMemcpyDeviceToHost), "err");
    if (err != 0)
      APP_ABORT(std::string(nu0 ? " dynbse::l0_apply_cols (device): a confluent triple pole at a node outside the DLR set."
                                : " dynbse::l0_apply_shift_cols (device): a confluent product at a node outside the DLR set."));
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
                    (void *)depsG, (void *)dgn, (void *)dfh, (void *)dfd1, (void *)dfd2, (void *)dDsq, (void *)dDcb, (void *)dDqt,
                    (void *)ds1, (void *)ds3, (void *)dr1u, (void *)dr3u, (void *)dr3t, (void *)dR1U, (void *)dR1T,
                    (void *)dR3U, (void *)dR3T, (void *)dF, (void *)dFs, (void *)derr, (void *)dVt, (void *)dPj,
                    (void *)dQj, (void *)dBj, (void *)dgjT, (void *)dglT, (void *)dGh, (void *)dAU, (void *)dAT,
                    (void *)dM2, (void *)dA1, (void *)dA3, (void *)pVt, (void *)pGjT, (void *)pGlT, (void *)pGh,
                    (void *)pPj, (void *)pQj, (void *)pBj, (void *)dT, (void *)dcoef})
      if (p) cu_check(cudaFree(p), "cudaFree");
    return K;
    }

  } // namespace

  long l0_apply_shift_cols(l0_dims const &d, l0_tables const &t, cplx *Ffam_h, cplx *Fsum_h, double free_bytes) {
    return run_l0(d, t, Ffam_h, Fsum_h, free_bytes, false);
  }

  long l0_apply_cols(l0_dims const &d, l0_tables const &t, cplx *Ffam_h, cplx *Fsum_h, double free_bytes) {
    return run_l0(d, t, Ffam_h, Fsum_h, free_bytes, true);
  }


  // =====================================================================================================================
  // THE DEVICE-RESIDENT UNIT (l0_cuda.cuh). Part 1: the resident L0 plan -- the kernels above,
  // driven on DEVICE buffers with the tables uploaded once per unit and the working set allocated once per engine.
  // =====================================================================================================================
  namespace {

    // any-nonzero flag per input component: c = 0 the constant block (cst, n elements), c = 1 + f np + a the family block
    // (f, a) of fam (each n = D nR elements). grid = ncomp blocks.
    __global__ void comp_flags_kernel(long np, long n, cd const *__restrict__ cst, cd const *__restrict__ fam, int *__restrict__ flags) {
      const long c = blockIdx.x;
      cd const *src = (c == 0) ? cst : (fam == nullptr ? nullptr : fam + (c - 1) * n);
      __shared__ int any;
      if (threadIdx.x == 0) any = 0;
      __syncthreads();
      if (src != nullptr)
        for (long e = threadIdx.x; e < n && !any; e += blockDim.x)
          if (nonzero(src[e])) any = 1;          // benign race: every writer writes 1
      __syncthreads();
      if (threadIdx.x == 0) flags[c] = any;
      (void)np;
    }
    __global__ void conj_kernel(long n, cd *__restrict__ x) {
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) x[e] = cuConj(x[e]);
    }
    // tau-slice permutation Fs (nt, D, nR) <-> P (D, nt, nR) with slot order slot_of[i]
    __global__ void rung_perm_kernel(long nt, long D, long nR, long const *__restrict__ slot_of, cd const *__restrict__ Fs,
                                     cd *__restrict__ P, bool inverse) {
      const long tot = nt * D * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long N = e % nR;
        long t = e / nR;
        const long y = t % D, i = t / D;                 // e indexes Fs (i, y, N)
        const long pe = (y * nt + slot_of[i]) * nR + N;
        if (inverse) P[e] = Fs[pe]; else P[pe] = Fs[e];    // inverse: P := unpermuted(Fs)
      }
    }
    // Pr[(y nt + slot) nR + N] = c(node_of[slot], r) P[...]   (c: (nt, R) row-major)
    __global__ void rung_cscale_kernel(long nt, long D, long nR, long R, long r, long const *__restrict__ node_of,
                                       cd const *__restrict__ c, cd const *__restrict__ P, cd *__restrict__ Pr) {
      const long tot = nt * D * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long slot = (e / nR) % nt;
        Pr[e] = cuCmul(c[node_of[slot] * R + r], P[e]);
      }
    }
    // the unified rung pass's packing: slot-ordered (mirror pairs adjacent) and (input, family) columns side by side,
    // Fbig[(y nt + slot_of[i]) ncol + off + N] = Fs[(i D + y) nR + N]; unpack is the inverse into Ys
    __global__ void pack_slots_kernel(long nt, long D, long nR, long ncol, long off, long const *__restrict__ slot_of,
                                      cd const *__restrict__ Fs, cd *__restrict__ Fbig) {
      const long tot = nt * D * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long N = e % nR, t = e / nR, y = t % D, i = t / D;
        Fbig[(y * nt + slot_of[i]) * ncol + off + N] = Fs[e];
      }
    }
    __global__ void unpack_slots_kernel(long nt, long D, long nR, long ncol, long off, long const *__restrict__ slot_of,
                                        cd const *__restrict__ Ybig, cd *__restrict__ Ys) {
      const long tot = nt * D * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long N = e % nR, t = e / nR, y = t % D, i = t / D;
        Ys[e] = Ybig[(y * nt + slot_of[i]) * ncol + off + N];
      }
    }
    __global__ void add2_kernel(long n, cd const *__restrict__ a, cd const *__restrict__ b, cd *__restrict__ out) {
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) out[e] = a[e] + b[e];
    }
    __global__ void add_identity_kernel(long D, cd *__restrict__ M) {
      for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < D; i += long(gridDim.x) * blockDim.x)
        M[i * D + i] = M[i * D + i] + real(1.0);
    }
    // the max over a block (blockDim.x a multiple of 32, <= 1024); valid in thread 0. Non-negative doubles order like their
    // bit patterns, so the grid-wide max is an atomicMax on the bits (one per block).
    __device__ inline double block_max(double v) {
      __shared__ double sh[32];
      __syncthreads();                                   // sh may still be read by a previous call
      for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
      const int lane = int(threadIdx.x) & 31, wid = int(threadIdx.x) >> 5;
      if (lane == 0) sh[wid] = v;
      __syncthreads();
      v = (int(threadIdx.x) < int((blockDim.x + 31) / 32)) ? sh[lane] : 0.0;
      if (wid == 0)
        for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
      return v;
    }
    // fmax drops NaN -- the meters map a non-finite entry to +inf (kept by fmax / atomicMax on the bit pattern), and
    // the host readback (meter_value) aborts on a non-finite meter
    __device__ inline double nan_inf(double x) { return isfinite(x) ? x : __longlong_as_double(0x7ff0000000000000LL); }
    // max |a - b| and max |a| over n elements into red[0], red[1] (non-negative doubles: the bit patterns order like the values)
    __global__ void maxdiff_kernel(long n, cd const *__restrict__ a, cd const *__restrict__ b, unsigned long long *__restrict__ red) {
      double num = 0.0, den = 0.0;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) {
        num = fmax(num, nan_inf(cuCabs(a[e] - b[e])));
        den = fmax(den, nan_inf(cuCabs(a[e])));
      }
      num = block_max(num);
      den = block_max(den);
      if (threadIdx.x == 0) {
        atomicMax(&red[0], (unsigned long long)__double_as_longlong(num));
        atomicMax(&red[1], (unsigned long long)__double_as_longlong(den));
      }
    }
    __global__ void maxabs_kernel(long n, cd const *__restrict__ x, unsigned long long *__restrict__ slot) {
      double m = 0.0;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) m = fmax(m, nan_inf(cuCabs(x[e])));
      m = block_max(m);
      if (threadIdx.x == 0) atomicMax(slot, (unsigned long long)__double_as_longlong(m));
    }
    inline unsigned grid_for(long n) { return unsigned(std::min<long>((n + 255) / 256, 65535)); }
    inline unsigned grid_red(long n) { return unsigned(std::max<long>(1, std::min<long>((n + 255) / 256, 1024))); }

    /** a device meter's value; a non-finite one (NaN / inf in the reduced data) aborts with what was measured */
    inline double meter_value(unsigned long long u, char const *what) {
      double d = 0.0;
      std::memcpy(&d, &u, sizeof(d));
      if (not std::isfinite(d)) APP_ABORT(std::string(" dynbse device: non-finite values (NaN / inf) in ") + what + " -- ABORTING.");
      return d;
    }
    template <typename T>
    T *dalloc(size_t n, char const *what) {
      T *p = nullptr;
      cu_check(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), what);
      return p;
    }
    template <typename T>
    void h2d(T *d, T const *h, size_t n, char const *what) {
      if (n > 0 && h != nullptr) cu_check(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice), what);
    }
    inline void h2d_c(cd *d, cplx const *h, size_t n, char const *what) { h2d(d, reinterpret_cast<cd const *>(h), n, what); }
    inline double wnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

  } // namespace

  /** the resident L0 plan: every table buffer at its maximal size (both kernels), the working set for nca_max = 1 + 2 np
   *  components and nR_max columns, the k-batch K fixed at creation */
  struct l0_plan {
    long np = 0, np_fit = 0, nk = 0, nc = 0, ng = 0, nR_max = 0, nca_max = 0, K = 1;
    bool nu0 = false;
    bool fused = true, asmg = true;                  // the fused passes; the nu = 0 gemm assembly
    bool merge = true;                               // the fused passes as one merged pass
    int fz_cfg = 44;                                 // the fused kernel's variant (10 CHG + MINB)
    bool fz_bench = false;                           // time every variant on the first k-batch per nu class
    cd *dcoef = nullptr, *dT = nullptr;
    cd inu = make_cuDoubleComplex(0.0, 0.0);
    double tfold = 0.0;
    long *dact = nullptr;
    cd *dgk = nullptr, *dgkq = nullptr, *dGhat = nullptr, *dGtil = nullptr;
    double *deps = nullptr, *depsG = nullptr, *dfh = nullptr, *dfd1 = nullptr, *dfd2 = nullptr;
    long *dgn = nullptr;
    cd *dDsq = nullptr, *dDcb = nullptr, *dDqt = nullptr, *ds1 = nullptr, *ds3 = nullptr, *dr1u = nullptr, *dr3u = nullptr,
       *dr3t = nullptr, *dR1U = nullptr, *dR1T = nullptr, *dR3U = nullptr, *dR3T = nullptr;
    int *derr = nullptr;
    cd *dVt = nullptr, *dPj = nullptr, *dQj = nullptr, *dBj = nullptr, *dgjT = nullptr, *dglT = nullptr, *dGh = nullptr;
    cd *dAU = nullptr, *dAT = nullptr, *dM2 = nullptr, *dA1 = nullptr, *dA3 = nullptr;
    cd **pVt = nullptr, **pGjT = nullptr, **pGlT = nullptr, **pGh = nullptr, **pPj = nullptr, **pQj = nullptr, **pBj = nullptr;
    cublasHandle_t h = nullptr;
    // per-k bytes of the working set at (nca_max, nR_max) (the fused passes keep no Pj / Qj / Bj, no pole operands)
    static double per_k_bytes(long np, long nc, long ng, long nR, bool fused = false, bool asmg = false) {
      const long nca = 1 + 2 * np, W = nc * nca * nR * nc, blk = nc * nR * nc, asz = 2 * np * blk;
      return double(W + (fused ? 0 : 3 * ng * W) + 5 * asz + (fused ? 0 : 3 * ng * nc * nc) + (asmg ? np * blk : 0)) * 16.0 +
             (fused ? 0.0 : double(7 * ng) * 8.0);
    }
    static double fixed_bytes(long np, long nk, long nc, long ng) {
      const double nc2 = double(nc * nc);
      return (2.0 * ng * nk * nc2 * 2.0 + 10.0 * np * np + 6.0 * np + 12.0 * np * ng) * 16.0 + 1.0 + 2.0 * np * 8.0;
    }
    l0_plan(long np_, long np_fit_, long nk_, long nc_, long ng_, long nR_max_, double free_bytes, bool fused_ = true, bool asmg_ = true,
            bool merge_ = true)
        : np(np_), np_fit(np_fit_), nk(nk_), nc(nc_), ng(ng_), nR_max(nR_max_), nca_max(1 + 2 * np_), fused(fused_), asmg(asmg_),
          merge(merge_) {
      const size_t nc2 = size_t(nc * nc);
      dact = dalloc<long>(size_t(nca_max), "plan act");
      dgk = dalloc<cd>(size_t(ng) * nk * nc2, "plan gk"); dgkq = dalloc<cd>(size_t(ng) * nk * nc2, "plan gkq");
      dGhat = dalloc<cd>(size_t(nk) * ng * nc2, "plan Ghat"); dGtil = dalloc<cd>(size_t(nk) * ng * nc2, "plan Gtil");
      deps = dalloc<double>(size_t(np), "plan eps"); depsG = dalloc<double>(size_t(ng), "plan epsG");
      dfh = dalloc<double>(size_t(np), "plan fhalf"); dfd1 = dalloc<double>(size_t(np), "plan fd1"); dfd2 = dalloc<double>(size_t(np), "plan fd2");
      dgn = dalloc<long>(size_t(ng), "plan gnode");
      for (cd **pp : {&dDsq, &dDcb, &dDqt, &dR1U, &dR1T, &dR3U, &dR3T}) *pp = dalloc<cd>(size_t(np) * np, "plan np x np");
      for (cd **pp : {&ds1, &ds3, &dr1u, &dr3u, &dr3t}) *pp = dalloc<cd>(size_t(np), "plan np");
      derr = dalloc<int>(1, "plan err");
      const long W = nc * nca_max * nR_max * nc, blk = nc * nR_max * nc;
      const size_t asz = size_t(2 * np) * size_t(blk);
      const double pk = per_k_bytes(np, nc, ng, nR_max, fused, asmg);
      dcoef = dalloc<cd>(size_t(2 * np) * ng * FZ_NCOEF, "plan coef");
      K = std::max(1L, std::min(nk, long((0.85 * free_bytes - fixed_bytes(np, nk, nc, ng)) / pk)));
      // allocate with the same halving retry as run_l0 (a shared device)
      auto try_alloc = [&](void **p, size_t bytes) -> bool {
        const cudaError_t e = cudaMalloc(p, std::max<size_t>(bytes, 1));
        if (e != cudaSuccess) { (void)cudaGetLastError(); *p = nullptr; return false; }
        return true;
      };
      auto free_ws = [&]() {
        for (void **p : {(void **)&dVt, (void **)&dPj, (void **)&dQj, (void **)&dBj, (void **)&dgjT, (void **)&dglT, (void **)&dGh,
                         (void **)&dAU, (void **)&dAT, (void **)&dM2, (void **)&dA1, (void **)&dA3, (void **)&dT, (void **)&pVt,
                         (void **)&pGjT, (void **)&pGlT, (void **)&pGh, (void **)&pPj, (void **)&pQj, (void **)&pBj})
          if (*p != nullptr) { (void)cudaFree(*p); *p = nullptr; }
      };
      bool ok = false;
      while (true) {
        const size_t nptr = size_t(K) * size_t(ng);
        const size_t bW = size_t(K) * W * sizeof(cd), bP = fused ? 16 : size_t(K) * ng * W * sizeof(cd);
        const size_t bG = fused ? 16 : size_t(K) * ng * nc2 * sizeof(cd), bA = size_t(K) * asz * sizeof(cd);
        const size_t bp = fused ? 16 : nptr * sizeof(cd *), bT = asmg ? size_t(K) * np * blk * sizeof(cd) : 16;
        ok = try_alloc((void **)&dVt, bW) and try_alloc((void **)&dPj, bP) and try_alloc((void **)&dQj, bP) and
             try_alloc((void **)&dBj, bP) and try_alloc((void **)&dgjT, bG) and try_alloc((void **)&dglT, bG) and
             try_alloc((void **)&dGh, bG) and try_alloc((void **)&dAU, bA) and try_alloc((void **)&dAT, bA) and
             try_alloc((void **)&dM2, bA) and try_alloc((void **)&dA1, bA) and try_alloc((void **)&dA3, bA) and
             try_alloc((void **)&dT, bT) and
             try_alloc((void **)&pVt, bp) and try_alloc((void **)&pGjT, bp) and try_alloc((void **)&pGlT, bp) and
             try_alloc((void **)&pGh, bp) and try_alloc((void **)&pPj, bp) and try_alloc((void **)&pQj, bp) and
             try_alloc((void **)&pBj, bp);
        if (ok) break;
        free_ws();
        if (K == 1) break;
        K = std::max(1L, K / 2);
      }
      if (not ok) APP_ABORT(std::string(" l0_plan: the L0 working set does not fit the device even at K = 1."));
      cub_check(cublasCreate(&h), "plan cublasCreate");
    }
    ~l0_plan() {
      if (h) (void)cublasDestroy(h);
      for (void *p : {(void *)dact, (void *)dgk, (void *)dgkq, (void *)dGhat, (void *)dGtil, (void *)deps, (void *)depsG, (void *)dfh,
                      (void *)dfd1, (void *)dfd2, (void *)dgn, (void *)dDsq, (void *)dDcb, (void *)dDqt, (void *)dR1U, (void *)dR1T,
                      (void *)dR3U, (void *)dR3T, (void *)ds1, (void *)ds3, (void *)dr1u, (void *)dr3u, (void *)dr3t, (void *)derr,
                      (void *)dVt, (void *)dPj, (void *)dQj, (void *)dBj, (void *)dgjT, (void *)dglT, (void *)dGh, (void *)dAU,
                      (void *)dAT, (void *)dM2, (void *)dA1, (void *)dA3, (void *)pVt, (void *)pGjT, (void *)pGlT, (void *)pGh,
                      (void *)pPj, (void *)pQj, (void *)pBj, (void *)dT, (void *)dcoef})
        if (p) (void)cudaFree(p);
    }
    /** the unit's tables (host pointers, the layouts of l0_tables); nu0 selects the nu = 0 kernel */
    void load(l0_tables const &t, bool nu0_) {
      nu0 = nu0_;
      const size_t nc2 = size_t(nc * nc);
      h2d_c(dgk, t.gk, size_t(ng) * nk * nc2, "plan gk"); h2d_c(dgkq, t.gkq, size_t(ng) * nk * nc2, "plan gkq");
      h2d_c(dGhat, t.Ghat, size_t(nk) * ng * nc2, "plan Ghat"); h2d_c(dGtil, t.Gtil, size_t(nk) * ng * nc2, "plan Gtil");
      h2d(deps, t.eps, size_t(np), "plan eps"); h2d(depsG, t.epsG, size_t(ng), "plan epsG"); h2d(dgn, t.gnode, size_t(ng), "plan gnode");
      h2d(dfh, t.fhalf, size_t(np), "plan fhalf"); h2d(dfd1, t.fd1, size_t(np), "plan fd1");
      if (nu0) h2d(dfd2, t.fd2, size_t(np), "plan fd2");
      h2d_c(dDsq, t.Dsq, size_t(np) * np, "plan Dsq"); h2d_c(dDcb, t.Dcb, size_t(np) * np, "plan Dcb");
      if (not nu0) {
        h2d_c(dDqt, t.Dqt, size_t(np) * np, "plan Dqt");
        h2d_c(ds1, t.s1, size_t(np), "plan s1"); h2d_c(ds3, t.s3, size_t(np), "plan s3");
        h2d_c(dr1u, t.r1u, size_t(np), "plan r1u"); h2d_c(dr3u, t.r3u, size_t(np), "plan r3u"); h2d_c(dr3t, t.r3t, size_t(np), "plan r3t");
        h2d_c(dR1U, t.R1U, size_t(np) * np, "plan R1U"); h2d_c(dR1T, t.R1T, size_t(np) * np, "plan R1T");
        h2d_c(dR3U, t.R3U, size_t(np) * np, "plan R3U"); h2d_c(dR3T, t.R3T, size_t(np) * np, "plan R3T");
      }
      inu = make_cuDoubleComplex(t.inu.real(), t.inu.imag());
      tfold = t.tfold;
      if (fused) build_coef(nu0, np, ng, inu, deps, depsG, dcoef);   // the unit's partial fractions (inu is the unit's)
    }
    /** F = L0 X on DEVICE buffers (F, Fs overwritten); act_h = the active components (host), sum_part1 as l0_tables */
    void apply(long nR, long nca, long const *act_h, bool skip_cst, bool sum_part1, cd const *Xfam, cd const *Xcst, cd *F, cd *Fs) {
      const long nc2 = nc * nc, D = nk * nc2;
      const size_t nF = size_t(2 * np) * D * nR, nFs = size_t(D) * nR;
      cu_check(cudaMemsetAsync(F, 0, nF * sizeof(cd), 0), "plan memset F");
      cu_check(cudaMemsetAsync(Fs, 0, nFs * sizeof(cd), 0), "plan memset Fs");
      if (nca == 0) return;                                        // nothing to apply: F and Fs are zero
      if (nca > (nu0 ? 1 + np : 1 + 2 * np)) APP_ABORT(std::string(" l0_plan::apply: too many active components."));
      h2d(dact, act_h, size_t(nca), "plan act");
      kdims kd{nc, nR, np, ng, nca, nc * nR * nc, nk, np_fit, dact};
      const long blk = nc * nR * nc, W = nc * nca * nR * nc;
      const size_t asz = size_t(2 * np) * size_t(blk);
      const size_t nptr = size_t(K) * size_t(ng);
      if (not fused) {
        fill_ptrs<<<unsigned((nptr + 255) / 256), 256>>>(K, ng, W, nc2, dVt, dgjT, dglT, dGh, dPj, dQj, dBj,
                                                         pVt, pGjT, pGlT, pGh, pPj, pQj, pBj);
        launch_check("plan fill_ptrs");
      }
      cu_check(cudaMemsetAsync(derr, 0, sizeof(int), 0), "plan memset err");
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      const int Ncols = int(nca * nR * nc), Mrows = int(nc * nca * nR);
      for (long ik0 = 0; ik0 < nk; ik0 += K) {
        const long Kb = std::min(K, nk - ik0);
        const int nb = int(Kb * ng);
        for (cd *p : {dAU, dAT, dM2, dA1, dA3})            // at nu = 0 only AU, M2, A3 are written and read
          if (not nu0 or (p != dAT and p != dA1)) cu_check(cudaMemsetAsync(p, 0, size_t(Kb) * asz * sizeof(cd), 0), "plan memset acc");
        pack_kernel<<<dim3(64u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, nu0, Xfam, Xcst, dVt);
        launch_check("plan pack_kernel");
        if (fused) {
          fused_passes(kd, ik0, Kb, nu0, inu, skip_cst, dVt, dgk, dgkq, dGhat, dGtil, dcoef, dgn, dAU, dAT, dM2, dA1, dA3, merge,
                       fz_cfg, fz_bench);
        } else {
        poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dgk, true, false, dgjT);
        poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dgkq, true, false, dglT);
        poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dGhat, false, true, dGh);
        launch_check("plan poleT_kernel");
        cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                                     (const cd **)pVt, Ncols, (const cd **)pGjT, int(nc), &zero, pPj, Ncols, nb), "plan gemm Pj");
        cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                     (const cd **)pGh, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb), "plan gemm Qj");
        cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                     (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pBj, int(nc), nb), "plan gemm Bj");
        if (nu0)
          scatter_nu0_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 0, skip_cst, dQj, dBj, deps, depsG, dgn, dAU, dM2, dA3);
        else
          scatter_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 0, inu, skip_cst, dQj, dBj, deps, depsG, dgn,
                                                                         dAU, dAT, dM2, dA1, dA3);
        launch_check("plan scatter pass 0");
        poleT_kernel<<<dim3(32u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, dGtil, false, true, dGh);
        launch_check("plan poleT_kernel Gtil");
        cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                                     (const cd **)pVt, Ncols, (const cd **)pGh, int(nc), &zero, pPj, Ncols, nb), "plan gemm Pl");
        cub_check(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                                     (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb), "plan gemm Rl");
        if (nu0)
          scatter_nu0_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 1, skip_cst, dQj, dQj, deps, depsG, dgn, dAU, dM2, dA3);
        else
          scatter_kernel<<<dim3(unsigned(nca), unsigned(Kb), 1u), 256>>>(kd, Kb, 1, inu, skip_cst, dQj, dQj, deps, depsG, dgn,
                                                                         dAU, dAT, dM2, dA1, dA3);
        launch_check("plan scatter pass 1");
        }
        if (nu0 and asmg)
          assemble_nu0_gemm(h, kd, ik0, Kb, sum_part1, dfh, dfd1, dfd2, dDsq, dDcb, dAU, dM2, dA3, dT, F, Fs, derr);
        else if (nu0)
          assemble_nu0_kernel<<<dim3(unsigned(np), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, sum_part1, dfh, dfd1, dfd2, dDsq, dDcb,
                                                                             dAU, dM2, dA3, F, Fs, derr);
        else
          assemble_kernel<<<dim3(unsigned(np), 2u, unsigned(Kb)), 256>>>(kd, ik0, Kb, sum_part1, dfh, dfd1, dDsq, ds1, ds3, dr1u,
                                                                         dr3u, dr3t, dR1U, dR1T, dR3U, dR3T, dAU, dAT, dM2, dA1, dA3,
                                                                         F, Fs, derr);
        launch_check("plan assemble_kernel");
        if (tfold > 0.0 && !nu0) {
          tfold_kernel<<<dim3(unsigned(np), unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, inu, tfold, deps, dDsq, dDcb, dDqt, F);
          launch_check("plan tfold_kernel");
        }
      }
      int err = 0;
      cu_check(cudaMemcpy(&err, derr, sizeof(int), cudaMemcpyDeviceToHost), "plan err");
      if (err != 0) APP_ABORT(std::string(" l0_plan::apply: a confluent product at a node outside the DLR set."));
    }
  };

  // ---- Part 2: the engine (l0_cuda.cuh). Column-major cuBLAS on the host's ROW-MAJOR memory: a row-major (r x c) block is the
  // column-major (c x r) transpose, so the host's C = A B is issued as C^T = B^T A^T on the same buffers.
  // the Sigma deposits' device state (ue_sd_*); every buffer is in `allocs` (freed by ue_destroy)
  struct sd_state {
    bool on = false, has_T = false;
    sd_config c;
    long AC = 0, qcap = 0;                                   // the a-chunk of the T family; the pointer-array capacity
    cd *Ttw = nullptr, *TtwW = nullptr, *Uw = nullptr, *Uw2 = nullptr, *Gw = nullptr, *gk = nullptr;
    cd *WkT = nullptr, *WkU = nullptr, *Wblk = nullptr, *Acst = nullptr, *DW = nullptr, *DWp = nullptr;
    cd *Aw = nullptr, *Fw = nullptr, *Ft = nullptr, *Y = nullptr, *Ym = nullptr, *Mb = nullptr;
    cd *S = nullptr, *RT = nullptr, *RU = nullptr;           // the accumulators (host layouts)
    cd **qA = nullptr, **qB = nullptr, **qC = nullptr;
    unsigned long long *red = nullptr;
    std::vector<cplx> Ttw_h;
    std::vector<long> kpq, gnode;
    std::vector<double> eps, epsG;
    std::vector<cd *> hA, hB, hC;
    std::vector<void *> allocs;
  };

  // the rung builds' device state (ue_kb_*)
  struct kb_state {
    bool on = false;
    long ns = 0, nq = 0, Nm = 0, nrep = 0, KC = 0;
    cd *X = nullptr, *W0 = nullptr, *Wd0 = nullptr, *Wds = nullptr;
    long *kq = nullptr;
    cd *U1 = nullptr, *U2 = nullptr, *WU2 = nullptr, *wb = nullptr;
    cd **pA = nullptr, **pB = nullptr, **pC = nullptr;
    std::vector<cd *> hA, hB, hC;
    std::vector<long> qx;
    std::vector<void *> allocs;
    long is = 0;                                             // the transfer whose legs / kq are set (ue_kb_build)
    long legs_ik0 = -1;                                      // the k chunk whose legs U1 / U2 hold (-1: none valid)
  };

  // the dressed-leg Gamma_1 readout on the device (ue_dressed_prepare / ue_gamma1_dressed)
  struct dressed_state {
    bool on = false;
    long ng = 0, nR_max = 0;
    cd *H = nullptr, *Gh = nullptr, *Gt = nullptr, *gk = nullptr, *gkq = nullptr;   // (2ng, 2np); (ng, nk, nc, nc) x 4
    cd *Etj = nullptr;                                     // conj(e~) (D, nout) row-major (the Dcj layout)
    cd *Z = nullptr, *Zc = nullptr, *Sa = nullptr, *Sbc = nullptr, *Zt = nullptr;   // (2ng, W); (ng, W) x 3; (W)
    std::vector<void *> allocs;
  };

  struct unit_engine {
    ue_config c;
    sd_state sd;
    kb_state kb;
    dressed_state dr;
    // the mirror-pair rung (rung_pair) and the frequency-factorized rung (rr = R > 0: Kds holds K_r, ctr (nt, R))
    bool rung_pair = true;
    long rr = 0;
    cd *ctr = nullptr;
    cd *Gs0 = nullptr;                                       // Gsum0 of the last block (the Sigma columns static_dyn / dyn1_bare)
    cd *Gr1 = nullptr;                                       // the one-bare-rung Gsum of the last block (when requested)
    cd *Dcj = nullptr, *CbD = nullptr, *Vr = nullptr, *Pr = nullptr;   // conj(legs) (D, nout); the readout scratch
    bool legs_ok = false, cbd_ok = false, r1_ok = false;
    long D = 0, nc2 = 0;
    cublasHandle_t cb = nullptr;
    cusolverDnHandle_t cs = nullptr;
    cd *KF = nullptr, *KF2 = nullptr, *Ut = nullptr, *Vs = nullptr, *Kmat = nullptr;                // basis (complex)
    cd *Ks = nullptr, *Kds = nullptr, *Kd0 = nullptr;                                               // (s, q)
    // non-resident dense rungs (partial residency): K_d(s_r) for r >= nres reach the device per rung application through two
    // staging buffers, each either REBUILT on the device from the W tables (src_host[r - nres] = 0) or COPIED from pinned host
    // memory (= 1, Kds_host slot r - nres) on the copy stream cps, overlapping the compute stream; ev_ready / ev_free order the
    // staging buffers. The split between the two sources is re-planned at every transfer from the measured costs.
    cd *Kds_host = nullptr, *stage[2] = {nullptr, nullptr};
    long nhost_cap = 0;                                      // pinned slots allocated (non-resident reps that may be copied)
    std::vector<int> src_host;                               // per non-resident rep: 1 = copy, 0 = rebuild
    cudaStream_t cps = nullptr;
    cudaEvent_t ev_ready[2] = {nullptr, nullptr}, ev_free[2] = {nullptr, nullptr};
    double t_copy_est = 0.0, t_rb_est = 0.0;                 // seconds per rep: H2D copy (measured bandwidth), device rebuild
    double t_gemm_est = 0.0;                                 // seconds per rep: the rung gemm (timed on the resident reps)
    long napp_cur = 0, napp_prev = 0;                        // rung passes of the current / previous transfer
    cudaEvent_t ev_t0 = nullptr, ev_t1 = nullptr;            // timing events around the resident gemms
    long ncopy = 0;
    double t_copy_wait = 0.0;
    std::vector<long> trep;
    cd scale = make_cuDoubleComplex(1.0, 0.0);
    cd *Cbk = nullptr, *M = nullptr, *work = nullptr;                                               // (s, q, nu)
    int *ipiv = nullptr, *dinfo = nullptr, lwork = 0;
    bool nu0 = false, unit_ok = false;
    cd *Xcst = nullptr, *Ffam = nullptr, *Fsum = nullptr, *F2fam = nullptr, *F2sum = nullptr, *Gfam = nullptr, *Gsum = nullptr;
    cd *yfam = nullptr, *ycst = nullptr, *Dblk = nullptr, *Y = nullptr, *csb = nullptr, *cbb = nullptr;
    cd *Fs = nullptr, *Ys = nullptr, *g = nullptr, *coef = nullptr, *rec = nullptr;
    cd **pA = nullptr, **pB = nullptr, **pC = nullptr;
    int *flags = nullptr;
    unsigned long long *red = nullptr;
    l0_plan *l0 = nullptr;
    std::vector<int> hflags;
    // partial residency: K_d(s_r) resident for r < nres (the rest: see Kds_host / stage above)
    long nres = 0;
    long nrebuild = 0;
    double t_rebuild = 0.0;
    // the rung pass: the packed (D, slot, inputs x families x nR) buffers; fuse = two inputs (the one-bare-rung and the Gamma_1
    // input of a block) in one pass, with the one-bare-rung y in yfam1 / ycst1
    bool fuse = false;
    long ncol_max = 0;
    cd *yfam1 = nullptr, *ycst1 = nullptr, *Fbig = nullptr, *Ybig = nullptr;
    long *dslot = nullptr;                                   // slot_of (nt) on the device
    // stream mode: the rung through the caller's streaming engine; slot i = tau node i, slot nt = W_d0
    bool stream = false;
    rung_stream *rs = nullptr;
  };

  double ue_bytes(ue_config const &c) {
    const double D = double(c.nk * c.nc * c.nc), W = D * double(c.nR_max), np = double(c.np);
    const double fam = 2.0 * np * W, cst = W;
    const long nres = c.stream ? 0 : ((c.nres < 0 or c.nres >= c.ndist) ? c.ndist : c.nres);
    const bool partial = (not c.stream and nres < c.ndist);
    double b = 0.0;
    if (c.stream) b += 2.0 * D * D;                                        // K_s, M (the rung streams: no K_d on the device)
    else b += (3.0 + double(std::max(nres, 1l)) + (partial ? 2.0 : 0.0)) * D * D;   // K_s, K_d0, M, K_d(s_r) (>= 1 slot) [+ 2 staging]
    if (not c.stream) b += 2.0 * (c.rung_fuse ? 4.0 : 2.0) * double(c.nt) * W;   // Fbig, Ybig (inputs x families x nR columns)
    if (c.rung_fuse) b += fam + cst;                                       // yfam1, ycst1
    b += 3.0 * fam + 13.0 * cst;                                            // Ffam, F2fam, Gfam + the 13 (D, nR) column buffers
    b += D * double(c.nout) + double(c.nout * c.nR_max);                    // the legs, the readout block
    b += fam;                                                               // yfam
    b += 2.0 * double(c.nt) * W + double(c.nt) * W + double(c.n_kept) * W + np * W;   // Fs, Ys, rec, g, coef
    b += double(c.nk * c.nc * c.nc * c.nc * c.nc);                          // Cb_k
    b *= 16.0;
    b += l0_plan::fixed_bytes(c.np, c.nk, c.nc, c.ng) +
         l0_plan::per_k_bytes(c.np, c.nc, c.ng, c.nR_max, c.l0_fused != 0, c.l0_asm_gemm != 0);   // L0 at K = 1
    return b * 1.05 + 512.0e6;                                              // cuSOLVER / cuBLAS workspaces
  }


  // the device bytes of the stages created after the unit (one source of truth for their own checks and for
  // the unit's memory partition, which keeps exactly this much free for them)
  double ue_kb_bytes(long ns, long nq, long Nm, long nk, long nc, long nrep) {
    const long nc2 = nc * nc;
    const double tables = double(ns * nk * Nm * nc) + double(nq * Nm * Nm) * (2.0 + double(nrep));
    const double per_k = double(nk) * (3.0 * double(Nm * nc2) + double(nc2 * nc2));
    return 16.0 * (tables + per_k) + 64.0e6;
  }
  double ue_sd_bytes(sd_config const &c, long R) {
    const long nt = c.nt, nw = c.nw_f, ns = c.ns, nk = c.nk, nc2 = c.nc * c.nc, np = c.np, ng = c.ng, Nm = c.Nm, D = nk * nc2;
    const double W = double(D) * double(R);
    const double acc = double(nt * ns * nk * nc2) * (1.0 + 2.0 * double(np));
    const double blk = W * (3.0 + 2.0 * double(nw) + double(nt)) + double(R * Nm) + double(ng * nk * nc2) +
                       double(ns * nw * nk * nc2) + 2.0 * double(nt * nw) + 2.0 * double(nw * np) + 2.0 * double(np * ng * nt);
    const double per_a = double(nk) * (2.0 * double(nc2 * nc2) + double(ng * nc2));
    return 16.0 * (acc + blk + per_a) + 64.0e6;
  }
  double ue_dressed_bytes(long ng, long np, long nk, long nc, long nout, long R) {
    const long nc2 = nc * nc, D = nk * nc2;
    const double W = double(D) * double(R);
    return 16.0 * (double(4 * ng * np) + 4.0 * double(ng * nk * nc2) + double(D * nout) * 3.0 + W * (2.0 * ng + 3.0 * ng + 1.0));
  }

  unit_engine *ue_create(ue_config const &c_in, double free_bytes, char *why, long why_len) {
    // the budget: 90 % of the free memory minus the bytes the caller keeps for the later device stages (rung builds, Sigma
    // deposits, dressed legs). Candidates in order: every rung resident (two-input pass if asked, then one input), then the
    // partial residencies (the most resident first) when a source for the non-resident rungs exists (device rebuild with the
    // device rung builds, or a pinned-host copy); an explicit nres only tries that residency.
    ue_config c = c_in;
    const double avail = 0.9 * free_bytes - std::max(0.0, c_in.reserve_bytes);
    const bool can_rebuild = (c_in.partial_ok != 0) and c_in.nonres_src != 2 and c_in.nonres_src != 3;
    const bool can_copy = (c_in.nonres_src == 0 or c_in.nonres_src == 2);
    const bool partial_allowed = (c_in.nonres_src != 3) and (can_rebuild or can_copy);
    std::vector<std::pair<long, int>> cand;
    auto add = [&](long n) {
      if (c_in.rung_fuse) cand.push_back({n, 1});
      cand.push_back({n, 0});
    };
    if (c_in.stream) cand.push_back({0l, 0});           // stream mode: no rung residency, no packed pass
    else if (c_in.nres >= 0) add(std::min(c_in.nres, c_in.ndist));
    else {
      add(c_in.ndist);
      if (partial_allowed) for (long n = c_in.ndist - 1; n >= 0; --n) add(n);
    }
    bool found = false;
    for (auto const &[n, f] : cand) {
      if (not c_in.stream and n < c_in.ndist and not partial_allowed) continue;
      c.nres = n; c.rung_fuse = f;
      if (ue_bytes(c) <= avail) { found = true; break; }
    }
    if (not found) {
      c.nres = (c_in.nres < 0) ? c_in.ndist : std::min(c_in.nres, c_in.ndist); c.rung_fuse = 0;
      std::snprintf(why, size_t(why_len), "needs %.1f GB of device memory, %.1f GB available (%.1f GB free, %.1f GB reserved)%s",
                    ue_bytes(c) / 1e9, avail / 1e9, free_bytes / 1e9, c_in.reserve_bytes / 1e9,
                    partial_allowed ? " even with the least rung residency"
                                    : "; a partial rung residency is not allowed by pol_vertex_dyn_device_memory = resident");
      return nullptr;
    }
    auto *e = new unit_engine;
    e->c = c;
    e->stream = (c.stream != 0);
    if (e->stream) { c.ndist = 0; c.nres = 0; c.rung_fuse = 0; e->c = c; }
    e->nres = c.nres;
    e->fuse = (c.rung_fuse != 0);
    e->nc2 = c.nc * c.nc;
    e->D = c.nk * e->nc2;
    const size_t D = size_t(e->D), W = D * size_t(c.nR_max), np = size_t(c.np), nt = size_t(c.nt);
    cub_check(cublasCreate(&e->cb), "ue cublasCreate");
    if (cusolverDnCreate(&e->cs) != CUSOLVER_STATUS_SUCCESS) APP_ABORT(std::string(" ue_create: cusolverDnCreate failed."));
    e->KF = dalloc<cd>(nt * np, "ue KF"); e->KF2 = dalloc<cd>(nt * np, "ue KF2");
    e->Ut = dalloc<cd>(size_t(c.n_kept) * nt, "ue Ut"); e->Vs = dalloc<cd>(size_t(c.np_fit) * size_t(c.n_kept), "ue Vs");
    e->Kmat = dalloc<cd>(nt * size_t(c.np_fit), "ue Kc");
    e->Ks = dalloc<cd>(D * D, "ue Ks"); e->Kd0 = dalloc<cd>(e->stream ? 1 : D * D, "ue Kd0");
    e->Kds = dalloc<cd>(e->stream ? 1 : size_t(std::max(e->nres, 1l)) * D * D, "ue Kds");
    const long nnr = e->stream ? 0 : c.ndist - e->nres;
    if (nnr > 0) {
      for (int bb = 0; bb < 2; ++bb) {
        e->stage[bb] = dalloc<cd>(D * D, "ue K staging");
        cu_check(cudaEventCreateWithFlags(&e->ev_ready[bb], cudaEventDisableTiming), "ue ev");
        cu_check(cudaEventCreateWithFlags(&e->ev_free[bb], cudaEventDisableTiming), "ue ev");
        cu_check(cudaEventRecord(e->ev_free[bb], 0), "ue ev free");
      }
      cu_check(cudaStreamCreateWithFlags(&e->cps, cudaStreamNonBlocking), "ue copy stream");
      cu_check(cudaEventCreate(&e->ev_t0), "ue ev t0"); cu_check(cudaEventCreate(&e->ev_t1), "ue ev t1");
      const bool can_copy_e = (c.nonres_src == 0 or c.nonres_src == 2 or (c.partial_ok == 0));
      if (can_copy_e) {
        e->nhost_cap = nnr;
        cu_check(cudaMallocHost(&e->Kds_host, size_t(nnr) * D * D * sizeof(cd)), "ue K (pinned host slots)");
        // the H2D bandwidth for the source planner: one staging-buffer copy, timed
        const double tb = wnow();
        cu_check(cudaMemcpy(e->stage[0], e->Kds_host, D * D * sizeof(cd), cudaMemcpyHostToDevice), "ue bw probe");
        e->t_copy_est = std::max(1.0e-6, wnow() - tb);
      }
      e->src_host.assign(size_t(nnr), (c.partial_ok == 0 or c.nonres_src == 2) ? 1 : 0);
    }
    if (not e->stream) {
      e->ncol_max = (e->fuse ? 4 : 2) * c.nR_max;
      e->Fbig = dalloc<cd>(size_t(e->ncol_max) * nt * D, "ue Fbig"); e->Ybig = dalloc<cd>(size_t(e->ncol_max) * nt * D, "ue Ybig");
      e->dslot = dalloc<long>(nt, "ue slot_of");
    }
    if (e->fuse) { e->yfam1 = dalloc<cd>(2 * np * W, "ue yfam1"); e->ycst1 = dalloc<cd>(W, "ue ycst1"); }
    e->Cbk = dalloc<cd>(size_t(c.nk) * e->nc2 * e->nc2, "ue Cbk"); e->M = dalloc<cd>(D * D, "ue M");
    e->ipiv = dalloc<int>(D, "ue ipiv"); e->dinfo = dalloc<int>(1, "ue info");
    if (cusolverDnZgetrf_bufferSize(e->cs, int(D), int(D), e->M, int(D), &e->lwork) != CUSOLVER_STATUS_SUCCESS)
      APP_ABORT(std::string(" ue_create: getrf_bufferSize failed."));
    e->work = dalloc<cd>(size_t(std::max(e->lwork, 1)), "ue getrf work");
    for (cd **pp : {&e->Ffam, &e->F2fam, &e->Gfam, &e->yfam}) *pp = dalloc<cd>(2 * np * W, "ue fam");
    for (cd **pp : {&e->Xcst, &e->Fsum, &e->F2sum, &e->Gsum, &e->ycst, &e->Dblk, &e->Y, &e->csb, &e->cbb, &e->Gs0, &e->Gr1, &e->CbD, &e->Vr})
      *pp = dalloc<cd>(W, "ue cst");
    e->Dcj = dalloc<cd>(D * size_t(std::max(c.nout, 1l)), "ue legs");
    e->Pr = dalloc<cd>(size_t(std::max(c.nout, 1l)) * size_t(c.nR_max), "ue readout");
    e->Fs = dalloc<cd>(nt * W, "ue Fs"); e->Ys = dalloc<cd>(nt * W, "ue Ys"); e->rec = dalloc<cd>(nt * W, "ue rec");
    e->g = dalloc<cd>(size_t(c.n_kept) * W, "ue g"); e->coef = dalloc<cd>(np * W, "ue coef");
    e->pA = dalloc<cd *>(nt, "ue pA"); e->pB = dalloc<cd *>(nt, "ue pB"); e->pC = dalloc<cd *>(nt, "ue pC");
    e->flags = dalloc<int>(1 + 2 * np, "ue flags"); e->hflags.assign(1 + 2 * np, 0);
    e->red = dalloc<unsigned long long>(2, "ue red");
    size_t fr = 0, tot = 0;
    cu_check(cudaMemGetInfo(&fr, &tot), "ue memgetinfo");
    // size the resident L0 k-batch from what is left AFTER the bytes kept for the later device stages (rung builds,
    // Sigma deposits, dressed legs: c.reserve_bytes, computed by the caller)
    const double l0_room = std::max(0.0, double(fr) - c.reserve_bytes);
    e->l0 = new l0_plan(c.np, c.np_fit, c.nk, c.nc, c.ng, c.nR_max, l0_room, c.l0_fused != 0, c.l0_asm_gemm != 0, c.l0_fused == 2);
    {
      const size_t used = std::strlen(why);
      std::snprintf(why + used, size_t(why_len) - used, "%sresident L0 k-batch K = %ld of %ld", used ? "; " : "", e->l0->K, long(c.nk));
    }
    e->l0->fz_cfg = c.l0_fz_cfg;
    e->l0->fz_bench = (c.l0_fz_bench != 0);
    return e;
  }

  void ue_destroy(unit_engine *e) {
    if (e == nullptr) return;
    delete e->l0;
    if (e->cb) (void)cublasDestroy(e->cb);
    if (e->cs) (void)cusolverDnDestroy(e->cs);
    for (void *p : {(void *)e->KF, (void *)e->KF2, (void *)e->Ut, (void *)e->Vs, (void *)e->Kmat, (void *)e->Ks, (void *)e->Kds,
                    (void *)e->Kd0, (void *)e->Cbk, (void *)e->M, (void *)e->work, (void *)e->ipiv, (void *)e->dinfo, (void *)e->Xcst,
                    (void *)e->Ffam, (void *)e->Fsum, (void *)e->F2fam, (void *)e->F2sum, (void *)e->Gfam, (void *)e->Gsum,
                    (void *)e->yfam, (void *)e->ycst, (void *)e->Dblk, (void *)e->Y, (void *)e->csb, (void *)e->cbb, (void *)e->Fs,
                    (void *)e->Ys, (void *)e->g, (void *)e->coef, (void *)e->rec, (void *)e->pA, (void *)e->pB, (void *)e->pC,
                    (void *)e->flags, (void *)e->red, (void *)e->Gs0, (void *)e->Gr1, (void *)e->Dcj, (void *)e->CbD, (void *)e->Vr,
                    (void *)e->Pr, (void *)e->yfam1, (void *)e->ycst1, (void *)e->Fbig, (void *)e->Ybig, (void *)e->dslot})
      if (p) (void)cudaFree(p);
    for (void *p : e->sd.allocs)
      if (p) (void)cudaFree(p);
    for (void *p : e->kb.allocs)
      if (p) (void)cudaFree(p);
    for (void *p : e->dr.allocs)
      if (p) (void)cudaFree(p);
    if (e->ctr) (void)cudaFree(e->ctr);
    if (e->Kds_host) (void)cudaFreeHost(e->Kds_host);
    for (int b = 0; b < 2; ++b) {
      if (e->stage[b]) (void)cudaFree(e->stage[b]);
      if (e->ev_ready[b]) (void)cudaEventDestroy(e->ev_ready[b]);
      if (e->ev_free[b]) (void)cudaEventDestroy(e->ev_free[b]);
    }
    if (e->cps) (void)cudaStreamDestroy(e->cps);
    if (e->ev_t0) (void)cudaEventDestroy(e->ev_t0);
    if (e->ev_t1) (void)cudaEventDestroy(e->ev_t1);
    delete e;
  }

  void ue_set_basis(unit_engine *e, double const *KFh, double const *KF2h, cplx const *Uth, cplx const *Vsh, cplx const *Kch) {
    auto up = [](cd *d, double const *h, size_t n, char const *what) {
      std::vector<cd> v(n);
      for (size_t i = 0; i < n; ++i) v[i] = make_cuDoubleComplex(h[i], 0.0);
      h2d(d, v.data(), n, what);
    };
    const size_t nt = size_t(e->c.nt), np = size_t(e->c.np), npf = size_t(e->c.np_fit), nk_ = size_t(e->c.n_kept);
    up(e->KF, KFh, nt * np, "ue KF"); up(e->KF2, KF2h, nt * np, "ue KF2");
    h2d_c(e->Ut, Uth, nk_ * nt, "ue Ut"); h2d_c(e->Vs, Vsh, npf * nk_, "ue Vs"); h2d_c(e->Kmat, Kch, nt * npf, "ue Kc");
  }

  long ue_nres(unit_engine const *e) { return e->nres; }
  bool ue_rung_fused(unit_engine const *e) { return e->fuse; }
  void ue_rebuild_stats(unit_engine const *e, long *n, double *seconds) { *n = e->nrebuild; *seconds = e->t_rebuild; }
  void ue_source_stats(unit_engine const *e, long *nhost_reps, long *ncopies, double *t_copy_est, double *t_rb_est) {
    long nh = 0;
    for (int v : e->src_host) nh += v;
    *nhost_reps = nh; *ncopies = e->ncopy; *t_copy_est = e->t_copy_est; *t_rb_est = e->t_rb_est;
  }
  void ue_set_stream(unit_engine *e, rung_stream *rs) {
    if (not e->stream) APP_ABORT(std::string(" ue_set_stream: the engine was not created in stream mode."));
    e->rs = rs;
  }

  void ue_set_rung(unit_engine *e, cplx const *Ksh, cplx const *Kdsh, cplx const *Kd0h, long const *treph, cplx scale_k) {
    const size_t D = size_t(e->D);
    h2d_c(e->Ks, Ksh, D * D, "ue Ks");
    if (not e->stream) {                                     // stream mode: Kds / Kd0 are the caller's empty arrays
      h2d_c(e->Kd0, Kd0h, D * D, "ue Kd0");
      h2d_c(e->Kds, Kdsh, size_t(e->nres) * D * D, "ue Kds (resident)");
      // host-built rungs: the non-resident ones are copied per application (no device rebuild without the W tables)
      const long nnr = e->c.ndist - e->nres;
      if (nnr > 0) {
        if (e->Kds_host == nullptr)
          APP_ABORT(std::string(" ue_set_rung: host-built rungs with a partial residency need the pinned-host source (no device "
                                "rung builds to rebuild them) -- ABORTING."));
        std::memcpy(e->Kds_host, Kdsh + size_t(e->nres) * D * D, size_t(nnr) * D * D * sizeof(cd));
        e->src_host.assign(size_t(nnr), 1);
      }
    }
    e->trep.assign(treph, treph + e->c.nt);
    e->scale = make_cuDoubleComplex(scale_k.real(), scale_k.imag());
    e->unit_ok = false;
  }

  int ue_set_unit(unit_engine *e, cplx const *Cbkh, l0_tables const &t, bool nu0, double free_bytes) {
    (void)free_bytes;
    const long D = e->D, nc2 = e->nc2, nk = e->c.nk;
    h2d_c(e->Cbk, Cbkh, size_t(nk) * nc2 * nc2, "ue Cbk");
    e->l0->load(t, nu0);
    e->nu0 = nu0;
    // M = 1 - Cb K_s, one block row per k: (M_k)^T (D x nc2) = -(K_s rows of k)^T (D x nc2) . Cb_k^T (nc2 x nc2)
    const cd mone = make_cuDoubleComplex(-1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(D), int(nc2), int(nc2), &mone,
                                        e->Ks, int(D), (long long)(nc2 * D), e->Cbk, int(nc2), (long long)(nc2 * nc2), &zero,
                                        e->M, int(D), (long long)(nc2 * D), int(nk)), "ue M = -Cb Ks");
    add_identity_kernel<<<grid_for(D), 256>>>(D, e->M);
    launch_check("ue add_identity");
    // the LU of the memory (column-major M^T): LAPACK's getrf on the host buffer computes the same factorization
    if (cusolverDnZgetrf(e->cs, int(D), int(D), e->M, int(D), e->work, e->ipiv, e->dinfo) != CUSOLVER_STATUS_SUCCESS)
      APP_ABORT(std::string(" ue_set_unit: cusolverDnZgetrf failed."));
    int info = 0;
    cu_check(cudaMemcpy(&info, e->dinfo, sizeof(int), cudaMemcpyDeviceToHost), "ue getrf info");
    e->unit_ok = (info == 0);
    return info;
  }

  namespace {
    // the active component list of an L0 input (the host kernels' active-component scan): flags over (cst, fam) on the device, the
    // list on the host. nu0: component 1 + a is the FOLDED family (fam0 or fam1 of node a non-zero).
    long ue_active(unit_engine *e, long nR, cd const *cst, cd const *fam, std::vector<long> &act) {
      const long np = e->c.np, n = e->D * nR, ncomp = (fam == nullptr) ? 1 : 1 + 2 * np;
      comp_flags_kernel<<<unsigned(ncomp), 256>>>(np, n, cst, fam, e->flags);
      launch_check("ue comp_flags");
      cu_check(cudaMemcpy(e->hflags.data(), e->flags, size_t(ncomp) * sizeof(int), cudaMemcpyDeviceToHost), "ue flags");
      act.clear();
      if (e->hflags[0]) act.push_back(0);
      if (fam != nullptr) {
        if (e->nu0) {
          for (long a = 0; a < np; ++a) if (e->hflags[size_t(1 + a)] or e->hflags[size_t(1 + np + a)]) act.push_back(1 + a);
        } else {
          for (long c = 1; c < 1 + 2 * np; ++c) if (e->hflags[size_t(c)]) act.push_back(c);
        }
      }
      return long(act.size());
    }
    // Fsum_k (nc2 x nR) (+)= Cb_k (nc2 x nc2) . X_k (nc2 x nR) for every k (row-major blocks of D x nR)
    void ue_cb_times(unit_engine *e, long nR, cd const *X, cd *out, bool accumulate) {
      const long nc2 = e->nc2, nk = e->c.nk;
      const cd one = make_cuDoubleComplex(1.0, 0.0), beta = make_cuDoubleComplex(accumulate ? 1.0 : 0.0, 0.0);
      cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(nc2), int(nc2), &one,
                                          X, int(nR), (long long)(nc2 * nR), e->Cbk, int(nc2), (long long)(nc2 * nc2), &beta,
                                          out, int(nR), (long long)(nc2 * nR), int(nk)), "ue Cb . X");
    }
    // L0 of (fam, cst) into (F, Fs), then the constant part's frequency sum through Cb (the host's Cb_cst route)
    void ue_l0(unit_engine *e, long nR, cd const *fam, cd const *cst, cd *F, cd *Fs, double *tl0) {
      std::vector<long> act;
      const double t0 = wnow();
      const long nca = ue_active(e, nR, cst, fam, act);
      const bool anyc = (not act.empty() and act[0] == 0);
      e->l0->apply(nR, nca, act.data(), not anyc, false, fam, cst, F, Fs);
      if (anyc) ue_cb_times(e, nR, cst, Fs, true);
      cu_check(cudaDeviceSynchronize(), "ue l0");
      *tl0 += wnow() - t0;
    }
    // c = T_s Fs = K_s M^-1 Fs (into csb), cb = Cb c (into cbb)
    void ue_ts(unit_engine *e, long nR, cd const *Fs, double *tts) {
      const double t0 = wnow();
      const long D = e->D;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      cub_check(cublasZgeam(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(D), int(nR), &one, Fs, int(nR), &zero, e->Y, int(D), e->Y, int(D)),
                "ue transpose Fs");
      if (cusolverDnZgetrs(e->cs, CUBLAS_OP_T, int(D), int(nR), e->M, int(D), e->ipiv, e->Y, int(D), e->dinfo) != CUSOLVER_STATUS_SUCCESS)
        APP_ABORT(std::string(" ue_ts: cusolverDnZgetrs failed."));
      {   // getrs' devInfo (an illegal argument is reported there, not in the status)
        int info = 0;
        cu_check(cudaMemcpy(&info, e->dinfo, sizeof(int), cudaMemcpyDeviceToHost), "ue_ts getrs info");
        if (info != 0) APP_ABORT(" ue_ts: cusolverDnZgetrs devInfo = " + std::to_string(info) + " -- ABORTING.");
      }
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nR), int(D), int(D), &one, e->Y, int(D), e->Ks, int(D), &zero,
                            e->csb, int(nR)), "ue K_s z");
      ue_cb_times(e, nR, e->csb, e->cbb, false);
      cu_check(cudaDeviceSynchronize(), "ue ts");
      *tts += wnow() - t0;
    }
    // y = K_d(Gamma, Gsum): the families to tau (DLR), the dense rung per tau node, the refit, the constant part. Returns the
    // worst refit error (the host's max over families of max|Ys - rec| / max|Ys|).
    void kb_build_tables(unit_engine *e, long const *ts, cd *const *Kts, long nts, cd sk);   // below, with the rung builds

    // the slot order of the dense pass: the tau nodes grouped by representative (mirror pairs adjacent), first / cnt per rep
    void rung_slots(unit_engine *e, std::vector<long> &slot_of, std::vector<long> &first, std::vector<long> &cnt) {
      const long nt = e->c.nt, ndist = e->c.ndist;
      slot_of.assign(size_t(nt), -1); first.assign(size_t(ndist), -1); cnt.assign(size_t(ndist), 0);
      long ns = 0;
      for (long r = 0; r < ndist; ++r)
        for (long i = 0; i < nt; ++i)
          if (e->trep[size_t(i)] == r) {
            if (first[size_t(r)] < 0) first[size_t(r)] = ns;
            ++cnt[size_t(r)];
            slot_of[size_t(i)] = ns++;
          }
      if (ns != nt) APP_ABORT(std::string(" rung_slots: a tau node has no representative."));
    }

    // the dense rung over the packed columns: Ybig rows of rep r (its slots x ncol) = scale K_d(r) Fbig rows. Resident reps
    // first (they overlap the first copies); then the non-resident ones in order, each through staging buffer j % 2, either
    // copied from pinned host memory on the copy stream (prefetched two ahead) or rebuilt on the device from the W tables.
    void ue_rung_dense(unit_engine *e, long ncol, std::vector<long> const &first, std::vector<long> const &cnt) {
      const long D = e->D, nt = e->c.nt, ndist = e->c.ndist, ld = nt * ncol;
      const size_t DD = size_t(D) * size_t(D);
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      auto gemm_rep = [&](long r, cd const *K) {
        const long s0 = first[size_t(r)], ns = cnt[size_t(r)];
        if (e->rung_pair) {                                 // one gemm per representative (both mirror nodes)
          cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(ns * ncol), int(D), int(D), &e->scale, e->Fbig + s0 * ncol,
                                int(ld), K, int(D), &zero, e->Ybig + s0 * ncol, int(ld)), "rung (rep)");
        } else {
          for (long sl = s0; sl < s0 + ns; ++sl)
            cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(ncol), int(D), int(D), &e->scale, e->Fbig + sl * ncol,
                                  int(ld), K, int(D), &zero, e->Ybig + sl * ncol, int(ld)), "rung (node)");
        }
      };
      const long nnr = ndist - e->nres;
      if (nnr <= 0) {
        for (long r = 0; r < e->nres; ++r) gemm_rep(r, e->Kds + size_t(r) * DD);
        return;
      }
      ++e->napp_cur;
      auto host = [&](long j) { return e->src_host[size_t(j)] != 0; };
      auto issue_copy = [&](long j) {
        const int b = int(j % 2);
        cu_check(cudaStreamWaitEvent(e->cps, e->ev_free[b], 0), "copy wait free");
        cu_check(cudaMemcpyAsync(e->stage[b], e->Kds_host + size_t(j) * DD, DD * sizeof(cd), cudaMemcpyHostToDevice, e->cps),
                 "copy H2D");
        cu_check(cudaEventRecord(e->ev_ready[b], e->cps), "copy ready");
        ++e->ncopy;
      };
      for (long j = 0; j < std::min(nnr, 2l); ++j)
        if (host(j)) issue_copy(j);
      if (e->nres > 0) {                                    // the resident gemms (timed: the planner's per-rep gemm cost)
        cu_check(cudaEventRecord(e->ev_t0, 0), "rung t0");
        for (long r = 0; r < e->nres; ++r) gemm_rep(r, e->Kds + size_t(r) * DD);
        cu_check(cudaEventRecord(e->ev_t1, 0), "rung t1");
      }
      for (long j = 0; j < nnr; ++j) {
        const int b = int(j % 2);
        const long r = e->nres + j;
        if (host(j)) {
          cu_check(cudaStreamWaitEvent(0, e->ev_ready[b], 0), "wait copy");
        } else {
          if (not e->kb.on) APP_ABORT(std::string(" ue_rung_dense: a rebuilt rung needs the device rung builds."));
          cu_check(cudaStreamSynchronize(0), "rebuild sync");
          const double tb = wnow();
          cd *K = e->stage[b];
          kb_build_tables(e, &r, &K, 1, one);
          cu_check(cudaStreamSynchronize(0), "rebuild");
          e->t_rebuild += wnow() - tb;
          ++e->nrebuild;
        }
        gemm_rep(r, e->stage[b]);
        cu_check(cudaEventRecord(e->ev_free[b], 0), "staging free");
        if (j + 2 < nnr and host(j + 2)) issue_copy(j + 2);
      }
      if (e->nrebuild > 0) e->t_rb_est = e->t_rebuild / double(e->nrebuild);
      if (e->nres > 0) {
        float ms = 0.0f;
        cu_check(cudaEventSynchronize(e->ev_t1), "rung t1 sync");
        cu_check(cudaEventElapsedTime(&ms, e->ev_t0, e->ev_t1), "rung gemm time");
        e->t_gemm_est = 1.0e-3 * double(ms) / double(e->nres);
      }
    }

    // THE RUNG PASS: nin inputs (1, or 2 = the one-bare-rung and the Gamma_1 input of a block), every frequency family, in one
    // pass over the dense rungs -- each K_d(s_r) is read (or brought to the device) once for nin x nfam x nR packed columns and
    // both mirror nodes; then the refit per (input, family) and the constant part per input. fe_out: the refit error per input.
    void ue_kd_multi(unit_engine *e, long nR, long nin, cd const *const *Gfam, cd const *const *Gs, cd *const *yfam,
                     cd *const *ycst, double *fe_out, double *tim);
    double ue_kd_single(unit_engine *e, long nR, cd const *Gfam, cd const *Gs, cd *yfam, cd *ycst, double *tim);
    void ue_kd_multi(unit_engine *e, long nR, long nin, cd const *const *Gfam, cd const *const *Gs, cd *const *yfam,
                     cd *const *ycst, double *fe_out, double *tim) {
      if (e->stream or e->rr > 0) {
        for (long x = 0; x < nin; ++x) fe_out[x] = ue_kd_single(e, nR, Gfam[x], Gs[x], yfam[x], ycst[x], tim);
        return;
      }
      const long D = e->D, np = e->c.np, nt = e->c.nt, nk_ = e->c.n_kept, W = D * nR;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      const cd mscale = make_cuDoubleComplex(-e->scale.x, -e->scale.y);
      const long nfam = e->nu0 ? 1 : 2, ncol = nin * nfam * nR;
      if (ncol > e->ncol_max) APP_ABORT(" ue_kd_multi: " + std::to_string(ncol) + " packed columns exceed the engine's " +
                                        std::to_string(e->ncol_max) + ".");
      std::vector<long> slot_of, first, cnt;
      rung_slots(e, slot_of, first, cnt);
      h2d(e->dslot, slot_of.data(), size_t(nt), "rung slot_of");
      double t0 = wnow();
      for (long x = 0; x < nin; ++x) {
        cu_check(cudaMemsetAsync(yfam[x], 0, size_t(2 * np) * W * sizeof(cd), 0), "ue memset y");
        for (long fam = 0; fam < nfam; ++fam) {
          for (long ff = fam; ff < (e->nu0 ? 2 : fam + 1); ++ff) {
            const cd beta = (ff == fam) ? zero : one;
            cd const *KFx = (ff == 1 and e->nu0) ? e->KF2 : e->KF;
            cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nt), int(np), &one, Gfam[x] + size_t(ff) * np * W, int(W),
                                  KFx, int(np), &beta, e->Fs, int(W)), "ue Fs = KF fam");
          }
          pack_slots_kernel<<<grid_for(nt * W), 256>>>(nt, D, nR, ncol, (x * nfam + fam) * nR, e->dslot, e->Fs, e->Fbig);
          launch_check("ue pack");
        }
      }
      cu_check(cudaDeviceSynchronize(), "ue dlr");
      tim[7] += wnow() - t0; t0 = wnow();
      ue_rung_dense(e, ncol, first, cnt);
      cu_check(cudaDeviceSynchronize(), "ue rung");
      tim[2] += wnow() - t0;
      for (long x = 0; x < nin; ++x) {
        double fe = 0.0;
        for (long fam = 0; fam < nfam; ++fam) {
          t0 = wnow();
          unpack_slots_kernel<<<grid_for(nt * W), 256>>>(nt, D, nR, ncol, (x * nfam + fam) * nR, e->dslot, e->Ybig, e->Ys);
          launch_check("ue unpack");
          // the refit: g^T = Ys^T Ut^T, c^T = g^T Vs^T, rec^T = c^T Kmat^T; err = max|Ys - rec| / max|Ys|
          cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nk_), int(nt), &one, e->Ys, int(W), e->Ut, int(nt), &zero,
                                e->g, int(W)), "ue g");
          const long npf = e->c.np_fit;
          cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(npf), int(nk_), &one, e->g, int(W), e->Vs, int(nk_), &zero,
                                e->coef, int(W)), "ue coef");
          cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nt), int(npf), &one, e->coef, int(W), e->Kmat, int(npf), &zero,
                                e->rec, int(W)), "ue rec");
          cu_check(cudaMemsetAsync(e->red, 0, 2 * sizeof(unsigned long long), 0), "ue red");
          maxdiff_kernel<<<grid_red(nt * W), 256>>>(nt * W, e->Ys, e->rec, e->red);
          launch_check("ue maxdiff");
          unsigned long long rr[2] = {0, 0};
          cu_check(cudaMemcpy(rr, e->red, sizeof(rr), cudaMemcpyDeviceToHost), "ue red d2h");
          const double num = meter_value(rr[0], "the rung output / its DLR refit"), den = meter_value(rr[1], "the rung output");
          fe = std::max(fe, (den > 0.0) ? num / den : num);
          cu_check(cudaMemcpy(yfam[x] + size_t(fam) * np * W, e->coef, size_t(npf) * W * sizeof(cd), cudaMemcpyDeviceToDevice),
                   "ue scatter");
          cu_check(cudaDeviceSynchronize(), "ue refit");
          tim[3] += wnow() - t0;
        }
        fe_out[x] = fe;
        t0 = wnow();
        cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), int(D), &mscale, Gs[x], int(nR), e->Kd0, int(D), &zero,
                              ycst[x], int(nR)), "ue y.cst");
        cu_check(cudaDeviceSynchronize(), "ue kd0");
        tim[2] += wnow() - t0;
      }
    }

    // the rung of ONE input for the paths outside the packed dense pass: the frequency-factorized rung (rr > 0) and the
    // streaming THC rung (stream mode); per family: the tau expansion, the rung, the refit; then the constant part
    double ue_kd_single(unit_engine *e, long nR, cd const *Gfam, cd const *Gs, cd *yfam, cd *ycst, double *tim) {
      const long D = e->D, np = e->c.np, nt = e->c.nt, nk_ = e->c.n_kept, W = D * nR;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      const cd mscale = make_cuDoubleComplex(-e->scale.x, -e->scale.y);
      double fe = 0.0;
      cu_check(cudaMemsetAsync(yfam, 0, size_t(2 * np) * W * sizeof(cd), 0), "ue memset y");
      if (not e->stream and e->rr <= 0) APP_ABORT(std::string(" ue_kd_single: only the factorized and the streaming rungs run here."));
      const long nfam = e->nu0 ? 1 : 2;
      for (long fam = 0; fam < nfam; ++fam) {
        double t0 = wnow();
        // Fs^T (W x nt) = fam^T (W x np) . KF^T (np x nt)   [+ fam1 . KF2 at nu = 0]
        for (long ff = fam; ff < (e->nu0 ? 2 : fam + 1); ++ff) {
          const cd beta = (ff == fam) ? zero : one;
          cd const *KFx = (ff == 1 and e->nu0) ? e->KF2 : e->KF;
          cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nt), int(np), &one, Gfam + size_t(ff) * np * W, int(W),
                                KFx, int(np), &beta, e->Fs, int(W)), "ue Fs = KF fam");
        }
        cu_check(cudaDeviceSynchronize(), "ue dlr");
        tim[7] += wnow() - t0; t0 = wnow();
        if (e->stream) {
          // Ys_i (D, nR) = scale K[W_d(s_i)] Fs_i through the streaming rung (slot i = tau node i)
          if (e->rs == nullptr) APP_ABORT(std::string(" ue_kd: stream mode without a rung_stream (ue_set_stream)."));
          for (long i = 0; i < nt; ++i)
            rs_apply_dev(e->rs, i, cplx(e->scale.x, e->scale.y), e->Fs + size_t(i) * W, e->Ys + size_t(i) * W, nR, nullptr);
        } else {
          // the frequency-factorized rung: P (D, nt, nR) = the tau slices in node order (P in rec), Y = sum_r K_r (c_r . P) into
          // Fs, unpermuted into Ys
          std::vector<long> slot_of(static_cast<size_t>(nt)), node_of(static_cast<size_t>(nt));
          for (long i = 0; i < nt; ++i) { slot_of[size_t(i)] = i; node_of[size_t(i)] = i; }
          long *dslot = reinterpret_cast<long *>(e->g), *dnode = dslot + nt;   // (the refit scratch g is free here)
          h2d(dslot, slot_of.data(), size_t(nt), "rung slot_of"); h2d(dnode, node_of.data(), size_t(nt), "rung node_of");
          rung_perm_kernel<<<grid_for(nt * W), 256>>>(nt, D, nR, dslot, e->Fs, e->rec, false);
          launch_check("rung perm");
          const long ld = nt * nR;
          cu_check(cudaMemsetAsync(e->Fs, 0, size_t(nt) * W * sizeof(cd), 0), "rung zero");
          for (long r = 0; r < e->rr; ++r) {
            rung_cscale_kernel<<<grid_for(nt * W), 256>>>(nt, D, nR, e->rr, r, dnode, e->ctr, e->rec, e->Ys);
            launch_check("rung cscale");
            cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(ld), int(D), int(D), &e->scale, e->Ys, int(ld),
                                  e->Kds + size_t(r) * size_t(D) * size_t(D), int(D), &one, e->Fs, int(ld)), "rung K_r");
          }
          rung_perm_kernel<<<grid_for(nt * W), 256>>>(nt, D, nR, dslot, e->Fs, e->Ys, true);
          launch_check("rung unperm");
        }
        cu_check(cudaDeviceSynchronize(), "ue rung");
        tim[2] += wnow() - t0; t0 = wnow();
        // the refit: g^T = Ys^T Ut^T, c^T = g^T Vs^T, rec^T = c^T Kmat^T; err = max|Ys - rec| / max|Ys|
        cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nk_), int(nt), &one, e->Ys, int(W), e->Ut, int(nt), &zero,
                              e->g, int(W)), "ue g");
        const long npf = e->c.np_fit;                     // the DLR fit's nodes (the refit never reaches the union extension)
        cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(npf), int(nk_), &one, e->g, int(W), e->Vs, int(nk_), &zero,
                              e->coef, int(W)), "ue coef");
        cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nt), int(npf), &one, e->coef, int(W), e->Kmat, int(npf), &zero,
                              e->rec, int(W)), "ue rec");
        cu_check(cudaMemsetAsync(e->red, 0, 2 * sizeof(unsigned long long), 0), "ue red");
        maxdiff_kernel<<<grid_red(nt * W), 256>>>(nt * W, e->Ys, e->rec, e->red);
        launch_check("ue maxdiff");
        unsigned long long r[2] = {0, 0};
        cu_check(cudaMemcpy(r, e->red, sizeof(r), cudaMemcpyDeviceToHost), "ue red d2h");
        const double num = meter_value(r[0], "the rung output / its DLR refit"), den = meter_value(r[1], "the rung output");
        fe = std::max(fe, (den > 0.0) ? num / den : num);
        // the scatter: y.fam(fam, p < np_fit) = c(p): the same row-major layout, one copy
        cu_check(cudaMemcpy(yfam + size_t(fam) * np * W, e->coef, size_t(e->c.np_fit) * W * sizeof(cd), cudaMemcpyDeviceToDevice), "ue scatter");
        cu_check(cudaDeviceSynchronize(), "ue refit");
        tim[3] += wnow() - t0;
      }
      // the constant part: y.cst = -scale K_d(0) Gsum
      const double t0 = wnow();
      if (e->stream)
        rs_apply_dev(e->rs, nt, cplx(mscale.x, mscale.y), Gs, ycst, nR, nullptr);   // slot nt = W_d0
      else
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), int(D), &mscale, Gs, int(nR), e->Kd0, int(D), &zero,
                            ycst, int(nR)), "ue y.cst");
      cu_check(cudaDeviceSynchronize(), "ue kd0");
      tim[2] += wnow() - t0;
      return fe;
    }

    double ue_kd(unit_engine *e, long nR, cd const *Gfam, cd const *Gs, cd *yfam, cd *ycst, double *tim) {
      double fe = 0.0;
      ue_kd_multi(e, nR, 1, &Gfam, &Gs, &yfam, &ycst, &fe, tim);
      return fe;
    }
  } // namespace

  void ue_set_legs(unit_engine *e, cplx const *Dc) {
    const long n = e->D * e->c.nout;
    h2d_c(e->Dcj, Dc, size_t(n), "ue legs");
    conj_kernel<<<grid_for(n), 256>>>(n, e->Dcj);
    launch_check("ue legs conj");
    e->legs_ok = true;
  }

  void ue_readout(unit_engine *e, long nR, int which, cplx *P, double *timing) {
    if (not e->legs_ok) APP_ABORT(std::string(" ue_readout: the legs of this transfer were not set."));
    if (which == 2 and not e->r1_ok) APP_ABORT(std::string(" ue_readout: the one-bare-rung column was not computed for this block."));
    const double t0 = wnow();
    const long D = e->D, nout = e->c.nout;
    const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0), mone = make_cuDoubleComplex(-1.0, 0.0);
    if (not e->cbd_ok) { ue_cb_times(e, nR, e->Dblk, e->CbD, false); e->cbd_ok = true; }
    cd const *G = (which == 0) ? e->Gs0 : ((which == 1) ? e->Gsum : e->Gr1);
    // V = G - Cb D (D x nR row-major = column-major nR x D); P^T (nR x nout) = V^T . conj(Dc)
    cub_check(cublasZgeam(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), &one, G, int(nR), &mone, e->CbD, int(nR), e->Vr, int(nR)),
              "ue readout V");
    cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nR), int(nout), int(D), &one, e->Vr, int(nR), e->Dcj, int(nout), &zero,
                          e->Pr, int(nR)), "ue readout D^dag V");
    cu_check(cudaMemcpy(P, e->Pr, size_t(nout * nR) * sizeof(cd), cudaMemcpyDeviceToHost), "ue readout d2h");
    timing[0] += wnow() - t0;
  }

  double ue_gamma1(unit_engine *e, long nR, cplx const *Dblkh, cplx *Gsum0h, cplx *Gsum1h, cplx *y1famh, cplx *y1csth, double *tim,
                   bool want_r1) {
    if (not e->unit_ok) APP_ABORT(std::string(" ue_gamma1: the unit's LU failed or was not set."));
    if (nR > e->c.nR_max) APP_ABORT(std::string(" ue_gamma1: block wider than the engine's nR_max."));
    const long D = e->D, np = e->c.np, W = D * nR;
    double t0 = wnow();
    h2d_c(e->Dblk, Dblkh, size_t(W), "ue Dblk");
    e->cbd_ok = false;
    tim[5] += wnow() - t0;
    if (e->fuse) {
      // THE FUSED FLOW (larger spaces): ls_apply(D) first, then ONE rung pass for the one-bare-rung input (F, Fsum of L0 D)
      // and the Gamma_1 input (Gamma, Gsum) -- the same operations as below, in another order (the one-rung input and its
      // second L0 only read / write scratch the Gamma_1 path does not use until after the rung)
      ue_l0(e, nR, nullptr, e->Dblk, e->Ffam, e->Fsum, &tim[0]);
      e->r1_ok = want_r1;
      ue_ts(e, nR, e->Fsum, &tim[1]);
      ue_l0(e, nR, nullptr, e->csb, e->F2fam, e->F2sum, &tim[0]);
      t0 = wnow();
      add2_kernel<<<grid_for(2 * np * W), 256>>>(2 * np * W, e->Ffam, e->F2fam, e->Gfam);
      add2_kernel<<<grid_for(W), 256>>>(W, e->Fsum, e->cbb, e->Gsum);
      launch_check("ue gamma");
      cu_check(cudaDeviceSynchronize(), "ue gamma");
      cu_check(cudaMemcpy(e->Gs0, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "ue Gs0");
      tim[4] += wnow() - t0; t0 = wnow();
      if (Gsum0h != nullptr) cu_check(cudaMemcpy(Gsum0h, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue Gsum0");
      tim[6] += wnow() - t0;
      cd const *gin[2]; cd const *sin[2]; cd *yo[2]; cd *co[2];
      long nin = 0;
      if (want_r1) { gin[nin] = e->Ffam; sin[nin] = e->Fsum; yo[nin] = e->yfam1; co[nin] = e->ycst1; ++nin; }
      const long ig = nin;
      gin[nin] = e->Gfam; sin[nin] = e->Gsum; yo[nin] = e->yfam; co[nin] = e->ycst; ++nin;
      double fes[2] = {0.0, 0.0};
      ue_kd_multi(e, nR, nin, gin, sin, yo, co, fes, tim);
      if (want_r1) {                                      // Gsum_r1 = Fsum[L0(y_r1; D + y_r1.cst)]
        t0 = wnow();
        add2_kernel<<<grid_for(W), 256>>>(W, e->Dblk, e->ycst1, e->Xcst);
        launch_check("ue r1 X.cst");
        cu_check(cudaDeviceSynchronize(), "ue r1 xcst");
        tim[4] += wnow() - t0;
        ue_l0(e, nR, e->yfam1, e->Xcst, e->F2fam, e->F2sum, &tim[0]);
        cu_check(cudaMemcpy(e->Gr1, e->F2sum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "ue Gr1");
      }
      t0 = wnow();
      add2_kernel<<<grid_for(W), 256>>>(W, e->Dblk, e->ycst, e->Xcst);
      launch_check("ue X.cst");
      cu_check(cudaDeviceSynchronize(), "ue xcst");
      tim[4] += wnow() - t0;
      ue_l0(e, nR, e->yfam, e->Xcst, e->Ffam, e->Fsum, &tim[0]);
      ue_ts(e, nR, e->Fsum, &tim[1]);
      t0 = wnow();
      add2_kernel<<<grid_for(W), 256>>>(W, e->Fsum, e->cbb, e->Gsum);
      launch_check("ue gsum1");
      cu_check(cudaDeviceSynchronize(), "ue gsum1");
      tim[4] += wnow() - t0; t0 = wnow();
      if (Gsum1h != nullptr) cu_check(cudaMemcpy(Gsum1h, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue Gsum1");
      if (y1famh != nullptr) cu_check(cudaMemcpy(y1famh, e->yfam, size_t(2 * np) * W * sizeof(cd), cudaMemcpyDeviceToHost), "ue y1 fam");
      if (y1csth != nullptr) cu_check(cudaMemcpy(y1csth, e->ycst, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue y1 cst");
      tim[6] += wnow() - t0;
      return fes[ig];
    }
    // ---- ls_apply(D, y = 0): F = L0 D, c = T_s Fsum, Gamma = F + L0 c, Gsum = Fsum + Cb c
    ue_l0(e, nR, nullptr, e->Dblk, e->Ffam, e->Fsum, &tim[0]);
    e->r1_ok = want_r1;
    if (want_r1) {
      // the ONE BARE dynamic rung (T_s = 0, one application): y_r1 = K_d(F, Fsum) of L0 D, Gsum_r1 = Fsum[L0(y_r1; D + y_r1.cst)].
      // F / Fsum are only read; y and F2 are scratch here (both are rewritten below). Its refit error is not reported (the
      // host pass discards that meter as well).
      (void)ue_kd(e, nR, e->Ffam, e->Fsum, e->yfam, e->ycst, tim);
      t0 = wnow();
      add2_kernel<<<grid_for(W), 256>>>(W, e->Dblk, e->ycst, e->Xcst);
      launch_check("ue r1 X.cst");
      cu_check(cudaDeviceSynchronize(), "ue r1 xcst");
      tim[4] += wnow() - t0;
      ue_l0(e, nR, e->yfam, e->Xcst, e->F2fam, e->F2sum, &tim[0]);
      cu_check(cudaMemcpy(e->Gr1, e->F2sum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "ue Gr1");
    }
    ue_ts(e, nR, e->Fsum, &tim[1]);
    ue_l0(e, nR, nullptr, e->csb, e->F2fam, e->F2sum, &tim[0]);
    t0 = wnow();
    add2_kernel<<<grid_for(2 * np * W), 256>>>(2 * np * W, e->Ffam, e->F2fam, e->Gfam);
    add2_kernel<<<grid_for(W), 256>>>(W, e->Fsum, e->cbb, e->Gsum);
    launch_check("ue gamma");
    cu_check(cudaDeviceSynchronize(), "ue gamma");
    cu_check(cudaMemcpy(e->Gs0, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "ue Gs0");
    tim[4] += wnow() - t0; t0 = wnow();
    if (Gsum0h != nullptr) cu_check(cudaMemcpy(Gsum0h, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue Gsum0");
    tim[6] += wnow() - t0;
    // ---- y1 = K_d(Gamma, Gsum)
    const double fe = ue_kd(e, nR, e->Gfam, e->Gsum, e->yfam, e->ycst, tim);
    // ---- ls_apply(D, y1): only Gsum1 = Fsum + Cb T_s Fsum is read by the Gamma_1 output (its Gamma is not), so the
    // second L0 (on the T_s image) is skipped
    t0 = wnow();
    add2_kernel<<<grid_for(W), 256>>>(W, e->Dblk, e->ycst, e->Xcst);
    launch_check("ue X.cst");
    cu_check(cudaDeviceSynchronize(), "ue xcst");
    tim[4] += wnow() - t0;
    ue_l0(e, nR, e->yfam, e->Xcst, e->Ffam, e->Fsum, &tim[0]);
    ue_ts(e, nR, e->Fsum, &tim[1]);
    t0 = wnow();
    add2_kernel<<<grid_for(W), 256>>>(W, e->Fsum, e->cbb, e->Gsum);
    launch_check("ue gsum1");
    cu_check(cudaDeviceSynchronize(), "ue gsum1");
    tim[4] += wnow() - t0; t0 = wnow();
    if (Gsum1h != nullptr) cu_check(cudaMemcpy(Gsum1h, e->Gsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue Gsum1");
    if (y1famh != nullptr) cu_check(cudaMemcpy(y1famh, e->yfam, size_t(2 * np) * W * sizeof(cd), cudaMemcpyDeviceToHost), "ue y1 fam");
    if (y1csth != nullptr) cu_check(cudaMemcpy(y1csth, e->ycst, size_t(W) * sizeof(cd), cudaMemcpyDeviceToHost), "ue y1 cst");
    tim[6] += wnow() - t0;
    return fe;
  }

  // =====================================================================================================================
  // The Sigma deposits on the device (l0_cuda.cuh). Row-major host layouts throughout; cuBLAS sees their column-major
  // transposes. Index conventions (D = nk nc2, W = D nR, row = k nc2 + x nc + y):
  //   Acst / DW (D, nR); DWp (nk, i, c, nR) = DW(k; c nc + i, r); Aw (nw_f, D, nR); Fw / Ft (nw_f | nt, nk, y, c, nR)
  //   Y (a, k) col-major (nc2 x nc2): Y(c i, x j); Ym (a, k) col-major: Ym(i j, c x); Mb (a, k, j, q = i nc + j)
  //   S (nt, ns, nk, nc, nc), RT / RU (nt, ns, nk, np, nc, nc), WkT / WkU (np, ng, nt), Gw (ns, nw_f, nk, nc, nc)
  // =====================================================================================================================
  namespace {
    __global__ void sd_bcast_kernel(long nw, long n, cd const *__restrict__ src, cd *__restrict__ dst) {
      const long tot = nw * n;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) dst[e] = src[e % n];
    }
    // DWp[((k nc + i) nc + c) nR + r] = DW[(k nc2 + c nc + i) nR + r]
    __global__ void sd_perm_dw_kernel(long nk, long nc, long nR, cd const *__restrict__ DW, cd *__restrict__ DWp) {
      const long tot = nk * nc * nc * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long r = e % nR;
        long t = e / nR;
        const long c = t % nc;
        t /= nc;
        const long i = t % nc, k = t / nc;
        DWp[e] = DW[((k * nc + c) * nc + i) * nR + r];
      }
    }
    // Ym[b nc4 + (i nc + j) + (c nc + x) nc2] = Y[b nc4 + (c nc + i) + (x nc + j) nc2]
    __global__ void sd_perm_y_kernel(long nb, long nc, cd const *__restrict__ Y, cd *__restrict__ Ym) {
      const long nc2 = nc * nc, nc4 = nc2 * nc2, tot = nb * nc4;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long b = e / nc4, rem = e % nc4, q = rem % nc2, cx = rem / nc2;
        const long i = q / nc, j = q % nc, c = cx / nc, x = cx % nc;
        Ym[e] = Y[b * nc4 + (c * nc + i) + (x * nc + j) * nc2];
      }
    }
    double bits_to_double(unsigned long long u) { return meter_value(u, "the device Sigma deposit meters"); }
  } // namespace

  bool ue_sd_init(unit_engine *e, sd_config const &c, cplx const *Ttwh, cplx const *Uwh, cplx const *Gwh, double const *eps,
                  double const *epsG, double free_bytes, char *why, long why_len) {
    auto &s = e->sd;
    if (c.nk != e->c.nk or c.nc != e->c.nc or c.np != e->c.np or c.nt != e->c.nt or c.ng != e->c.ng) {
      APP_ABORT(std::string(" ue_sd_init: the Sigma deposit sizes differ from the engine's (internal inconsistency) -- ABORTING."));
      std::snprintf(why, size_t(why_len), "the deposit sizes differ from the engine's");
      return false;
    }
    const long nt = c.nt, nw = c.nw_f, ns = c.ns, nk = c.nk, nc2 = e->nc2, np = c.np, ng = c.ng, Nm = c.Nm, D = e->D, R = e->c.nR_max;
    (void)nt; (void)nw; (void)ns; (void)Nm; (void)D; (void)R;
    const double per_a = double(nk) * (2.0 * double(nc2 * nc2) + double(ng * nc2));
    const double need = ue_sd_bytes(c, e->c.nR_max);
    if (need > 0.9 * free_bytes) {
      std::snprintf(why, size_t(why_len), "needs %.1f GB of device memory, %.1f GB free", need / 1e9, free_bytes / 1e9);
      return false;
    }
    s.c = c;
    s.AC = long(std::min<double>(double(np), std::max(1.0, std::floor(std::min(0.5 * (0.9 * free_bytes - need), 4.0e9) / (16.0 * per_a)))));
    auto al = [&](size_t n, char const *what) { cd *p = dalloc<cd>(n, what); s.allocs.push_back(p); return p; };
    const size_t Wz = size_t(D) * size_t(R);
    s.Ttw = al(size_t(nt * nw), "sd Ttw"); s.TtwW = al(size_t(nt * nw), "sd TtwW");
    s.Uw = al(size_t(nw * np), "sd Uw"); s.Uw2 = al(size_t(nw * np), "sd Uw2");
    s.Gw = al(size_t(ns * nw * nk * nc2), "sd Gw");
    s.gk = al(size_t(ng * nk * nc2), "sd gk");
    s.WkT = al(size_t(np * ng * nt), "sd WkT"); s.WkU = al(size_t(np * ng * nt), "sd WkU");
    s.Wblk = al(size_t(R * Nm), "sd Wblk");
    s.Acst = al(Wz, "sd Acst"); s.DW = al(Wz, "sd DW"); s.DWp = al(Wz, "sd DWp");
    s.Aw = al(size_t(nw) * Wz, "sd Aw"); s.Fw = al(size_t(nw) * Wz, "sd Fw"); s.Ft = al(size_t(nt) * Wz, "sd Ft");
    s.Y = al(size_t(s.AC * nk * nc2 * nc2), "sd Y"); s.Ym = al(size_t(s.AC * nk * nc2 * nc2), "sd Ym");
    s.Mb = al(size_t(s.AC * nk * ng * nc2), "sd M");
    const size_t nS = size_t(nt * ns * nk * nc2);
    s.S = al(nS, "sd S"); s.RT = al(nS * size_t(np), "sd RT"); s.RU = al(nS * size_t(np), "sd RU");
    cu_check(cudaMemsetAsync(s.S, 0, nS * sizeof(cd), 0), "sd zero S");
    cu_check(cudaMemsetAsync(s.RT, 0, nS * size_t(np) * sizeof(cd), 0), "sd zero RT");
    cu_check(cudaMemsetAsync(s.RU, 0, nS * size_t(np) * sizeof(cd), 0), "sd zero RU");
    s.qcap = std::max({nt * nk, s.AC * nk, nk * ng});
    s.qA = dalloc<cd *>(size_t(s.qcap), "sd qA"); s.qB = dalloc<cd *>(size_t(s.qcap), "sd qB"); s.qC = dalloc<cd *>(size_t(s.qcap), "sd qC");
    for (void *p : {(void *)s.qA, (void *)s.qB, (void *)s.qC}) s.allocs.push_back(p);
    s.hA.assign(size_t(s.qcap), nullptr); s.hB.assign(size_t(s.qcap), nullptr); s.hC.assign(size_t(s.qcap), nullptr);
    s.red = dalloc<unsigned long long>(4, "sd red");
    s.allocs.push_back(s.red);
    // the run-wide tables
    s.Ttw_h.assign(Ttwh, Ttwh + nt * nw);
    h2d_c(s.Ttw, Ttwh, size_t(nt * nw), "sd Ttw");
    h2d_c(s.Uw, Uwh, size_t(nw * np), "sd Uw");
    {
      std::vector<cplx> u2(size_t(nw * np));
      for (size_t i = 0; i < u2.size(); ++i) u2[i] = Uwh[i] * Uwh[i];
      h2d_c(s.Uw2, u2.data(), u2.size(), "sd Uw2");
      std::vector<cplx> g(size_t(ns * nw * nk * nc2));
      const long blkg = nk * nc2;
      for (long is = 0; is < ns; ++is)
        for (long n = 0; n < nw; ++n)
          for (long t = 0; t < blkg; ++t) g[size_t((is * nw + n) * blkg + t)] = Gwh[(n * ns + is) * blkg + t];
      h2d_c(s.Gw, g.data(), g.size(), "sd Gw");
    }
    s.eps.assign(eps, eps + np);
    s.epsG.assign(epsG, epsG + ng);
    s.on = true;
    return true;
  }

  void ue_sd_set_sq(unit_engine *e, long const *kpq_row, cplx const *gkh, long const *gnode) {
    auto &s = e->sd;
    if (e->c.nout != s.c.Nm) APP_ABORT(std::string(" ue_sd_set_sq: the deposits need the aux legs (nout == N_m)."));
    h2d_c(s.gk, gkh, size_t(s.c.ng * s.c.nk * e->nc2), "sd gk");
    s.kpq.assign(kpq_row, kpq_row + s.c.nk);
    s.gnode.assign(gnode, gnode + s.c.ng);
    std::vector<long> gs(s.gnode);
    std::sort(gs.begin(), gs.end());
    if (std::adjacent_find(gs.begin(), gs.end()) != gs.end())
      APP_ABORT(std::string(" ue_sd_set_sq: two G poles share a vertex node (the batched RU_{n_j} deposit needs distinct nodes)."));
    std::vector<long> ks(s.kpq);
    std::sort(ks.begin(), ks.end());
    if (std::adjacent_find(ks.begin(), ks.end()) != ks.end())
      APP_ABORT(std::string(" ue_sd_set_sq: kpq(iq, .) is not a bijection."));
  }

  void ue_sd_set_unit(unit_engine *e, cplx const *w, cplx const *WkTh, cplx const *WkUh) {
    auto &s = e->sd;
    const long nt = s.c.nt, nw = s.c.nw_f;
    std::vector<cplx> tw(size_t(nt * nw));
    for (long it = 0; it < nt; ++it)
      for (long n = 0; n < nw; ++n) tw[size_t(it * nw + n)] = w[it] * s.Ttw_h[size_t(it * nw + n)];
    h2d_c(s.TtwW, tw.data(), tw.size(), "sd TtwW");
    s.has_T = (WkTh != nullptr and WkUh != nullptr);
    if (s.has_T) {
      const size_t n = size_t(s.c.np * s.c.ng * nt);
      h2d_c(s.WkT, WkTh, n, "sd WkT");
      h2d_c(s.WkU, WkUh, n, "sd WkU");
    }
  }

  void ue_sd_block(unit_engine *e, long is, long nR, int col, cplx const *Wblkh, bool use_U, bool use_T, double *meters,
                   double *timing) {
    auto &s = e->sd;
    if (not s.on) APP_ABORT(std::string(" ue_sd_block: the deposits were not initialized."));
    if (nR > e->c.nR_max) APP_ABORT(std::string(" ue_sd_block: block wider than the engine's nR_max."));
    const double t0 = wnow();
    const long D = e->D, nc = s.c.nc, nc2 = e->nc2, nk = s.c.nk, np = s.c.np, npf = s.c.np_fit, ng = s.c.ng, nt = s.c.nt;
    const long nw = s.c.nw_f, ns = s.c.ns, Nm = s.c.Nm, W = D * nR, ldacc = ns * nk * np * nc2;
    const bool with_y = (col != 0);
    cd const *Gs = (col == 2) ? e->Gsum : e->Gs0;
    const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0), mone = make_cuDoubleComplex(-1.0, 0.0);
    h2d_c(s.Wblk, Wblkh, size_t(nR * Nm), "sd Wblk");
    // (1) A_cst^T (nR x D) = Gs^T . K_s^T (+ y.cst^T)
    if (with_y) cu_check(cudaMemcpy(s.Acst, e->ycst, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "sd Acst <- y.cst");
    cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), int(D), &one, Gs, int(nR), e->Ks, int(D),
                          with_y ? &one : &zero, s.Acst, int(nR)), "sd Acst");
    // the meters: max |A_cst|, max |y fam|, max |fam1 on an extension node|
    cu_check(cudaMemsetAsync(s.red, 0, 4 * sizeof(unsigned long long), 0), "sd red");
    maxabs_kernel<<<grid_red(W), 256>>>(W, s.Acst, s.red);
    if (with_y) {
      maxabs_kernel<<<grid_red(2 * np * W), 256>>>(2 * np * W, e->yfam, s.red + 1);
      if (np > npf) maxabs_kernel<<<grid_red((np - npf) * W), 256>>>((np - npf) * W, e->yfam + size_t(np + npf) * W, s.red + 2);
    }
    launch_check("sd meters");
    // (2) DW^T (nR x nk nc2) = W_blk (nR x Nm) . conj(Dc)^T (Nm x nk nc2); DWp = its (k, i, c, r) order
    if (not e->legs_ok) APP_ABORT(std::string(" ue_sd_block: the legs of this transfer were not set."));
    cub_check(cublasZgemm(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nR), int(nk * nc2), int(Nm), &one, s.Wblk, int(Nm), e->Dcj, int(Nm),
                          &zero, s.DW, int(nR)), "sd DW");
    sd_perm_dw_kernel<<<grid_for(W), 256>>>(nk, nc, nR, s.DW, s.DWp);
    launch_check("sd perm DW");
    // (3) the product route: A_w = A_cst + U fam0 (+ U^2 fam1 at nu = 0); F_w = G A_w per (n, k) in the (y, c, r) order;
    //     F_t = [w Ttw] F_w; S(tau, s, kpq(k)) += F_t(tau, k)^T-contracted with DWp(k)
    sd_bcast_kernel<<<grid_for(nw * W), 256>>>(nw, W, s.Acst, s.Aw);
    launch_check("sd bcast");
    if (with_y and use_U)
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nw), int(np), &one, e->yfam, int(W), s.Uw, int(np), &one,
                            s.Aw, int(W)), "sd Aw fam0");
    if (with_y and use_T and e->nu0)
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nw), int(np), &one, e->yfam + size_t(np) * W, int(W), s.Uw2,
                            int(np), &one, s.Aw, int(W)), "sd Aw fam1 (nu = 0)");
    cd const *Gws = s.Gw + size_t(is) * nw * nk * nc2;
    for (long y = 0; y < nc; ++y)
      cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(nc), int(nc), &one, s.Aw + y * nR, int(nc * nR),
                                          (long long)(nc2 * nR), Gws, int(nc), (long long)nc2, &zero, s.Fw + y * nc * nR, int(nR),
                                          (long long)(nc2 * nR), int(nw * nk)), "sd Fw = G Aw");
    cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(nt), int(nw), &one, s.Fw, int(W), s.TtwW, int(nw), &zero,
                          s.Ft, int(W)), "sd Ft = Ttw Fw");
    for (long it = 0; it < nt; ++it)
      for (long ik = 0; ik < nk; ++ik) {
        const size_t b = size_t(it * nk + ik);
        s.hA[b] = s.Ft + size_t(it) * W + size_t(ik) * nc2 * nR;
        s.hB[b] = s.DWp + size_t(ik) * nc2 * nR;
        s.hC[b] = s.S + size_t(((it * ns + is) * nk + s.kpq[size_t(ik)]) * nc2);
      }
    h2d(s.qA, s.hA.data(), size_t(nt * nk), "sd qA"); h2d(s.qB, s.hB.data(), size_t(nt * nk), "sd qB");
    h2d(s.qC, s.hC.data(), size_t(nt * nk), "sd qC");
    cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nc), int(nc), int(nc * nR), &one, (const cd **)s.qA, int(nc * nR),
                                 (const cd **)s.qB, int(nc * nR), &one, s.qC, int(nc), int(nt * nk)), "sd legs -> S");
    // (4) the T family (nu != 0)
    if (with_y and use_T and not e->nu0 and s.has_T) {
      comp_flags_kernel<<<unsigned(1 + 2 * np), 256>>>(np, W, nullptr, e->yfam, e->flags);
      launch_check("sd hasT");
      cu_check(cudaMemcpy(e->hflags.data(), e->flags, size_t(1 + 2 * np) * sizeof(int), cudaMemcpyDeviceToHost), "sd hasT d2h");
      long a_hi = 0;
      for (long a = 0; a < np; ++a)
        if (e->hflags[size_t(1 + np + a)]) {
          a_hi = a + 1;
          for (long j = 0; j < ng; ++j)
            if (std::abs(s.eps[size_t(a)] - s.epsG[size_t(j)]) <= 1e-10)
              APP_ABORT(std::string(" sigma_dyn_accumulate (device): confluent U_j T_a (e_a = e_j = ") + std::to_string(s.eps[size_t(a)]) +
                        ") is not supported.");
        }
      for (long a0 = 0; a0 < a_hi; a0 += s.AC) {
        const long na = std::min(s.AC, a_hi - a0), nb = na * nk;
        // Y(c i, x j) = sum_r DW(k; c i, r) fam1_a(k; x j, r)
        for (long lb = 0; lb < na; ++lb)
          for (long ik = 0; ik < nk; ++ik) {
            const size_t b = size_t(lb * nk + ik);
            s.hA[b] = s.DW + size_t(ik) * nc2 * nR;
            s.hB[b] = e->yfam + size_t(np + a0 + lb) * W + size_t(ik) * nc2 * nR;
            s.hC[b] = s.Y + b * size_t(nc2 * nc2);
          }
        h2d(s.qA, s.hA.data(), size_t(nb), "sd qA"); h2d(s.qB, s.hB.data(), size_t(nb), "sd qB"); h2d(s.qC, s.hC.data(), size_t(nb), "sd qC");
        cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nc2), int(nc2), int(nR), &one, (const cd **)s.qA, int(nR),
                                     (const cd **)s.qB, int(nR), &zero, s.qC, int(nc2), int(nb)), "sd Y");
        sd_perm_y_kernel<<<grid_for(nb * nc2 * nc2), 256>>>(nb, nc, s.Y, s.Ym);
        launch_check("sd perm Y");
        // M_aj(i j) = sum_{c x} g_j(k; c, x) Y(c i, x j): column-major (q x j) = Ym (q x cx) . gk_k (cx x j)
        for (long lb = 0; lb < na; ++lb)
          for (long ik = 0; ik < nk; ++ik) {
            const size_t b = size_t(lb * nk + ik);
            s.hA[b] = s.Ym + b * size_t(nc2 * nc2);
            s.hB[b] = s.gk + size_t(ik) * nc2;
            s.hC[b] = s.Mb + b * size_t(ng * nc2);
          }
        h2d(s.qA, s.hA.data(), size_t(nb), "sd qA"); h2d(s.qB, s.hB.data(), size_t(nb), "sd qB"); h2d(s.qC, s.hC.data(), size_t(nb), "sd qC");
        cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nc2), int(ng), int(nc2), &one, (const cd **)s.qA, int(nc2),
                                     (const cd **)s.qB, int(nk * nc2), &zero, s.qC, int(nc2), int(nb)), "sd M");
        // RT_a(tau) -= sum_j WkT(a, j, tau) M_aj;  RU_a(tau) += sum_j WkU(a, j, tau) M_aj   (C = (q x tau) at ld ns nk np nc2)
        for (int pass = 0; pass < 2; ++pass) {
          cd *acc = (pass == 0) ? s.RT : s.RU;
          cd const *tab = (pass == 0) ? s.WkT : s.WkU;
          for (long lb = 0; lb < na; ++lb)
            for (long ik = 0; ik < nk; ++ik) {
              const size_t b = size_t(lb * nk + ik);
              s.hA[b] = s.Mb + b * size_t(ng * nc2);
              s.hB[b] = const_cast<cd *>(tab) + size_t((a0 + lb) * ng * nt);
              s.hC[b] = acc + size_t(((is * nk + s.kpq[size_t(ik)]) * np + a0 + lb) * nc2);
            }
          h2d(s.qA, s.hA.data(), size_t(nb), "sd qA"); h2d(s.qB, s.hB.data(), size_t(nb), "sd qB"); h2d(s.qC, s.hC.data(), size_t(nb), "sd qC");
          cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nc2), int(nt), int(ng), pass == 0 ? &mone : &one,
                                       (const cd **)s.qA, int(nc2), (const cd **)s.qB, int(nt), &one, s.qC, int(ldacc), int(nb)),
                    pass == 0 ? "sd RT_a" : "sd RU_a");
        }
        // RU_{n_j}(tau) -= sum_a WkU(a, j, tau) M_aj   (batch over (k, j); gnode is injective, kpq a bijection: no two
        // entries share a target)
        for (long ik = 0; ik < nk; ++ik)
          for (long j = 0; j < ng; ++j) {
            const size_t b = size_t(ik * ng + j);
            s.hA[b] = s.Mb + size_t((ik * ng + j) * nc2);
            s.hB[b] = s.WkU + size_t((a0 * ng + j) * nt);
            s.hC[b] = s.RU + size_t(((is * nk + s.kpq[size_t(ik)]) * np + s.gnode[size_t(j)]) * nc2);
          }
        h2d(s.qA, s.hA.data(), size_t(nk * ng), "sd qA"); h2d(s.qB, s.hB.data(), size_t(nk * ng), "sd qB");
        h2d(s.qC, s.hC.data(), size_t(nk * ng), "sd qC");
        cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nc2), int(nt), int(na), &mone, (const cd **)s.qA,
                                     int(nk * ng * nc2), (const cd **)s.qB, int(ng * nt), &one, s.qC, int(ldacc), int(nk * ng)),
                  "sd RU_nj");
      }
    }
    unsigned long long r[4] = {0, 0, 0, 0};
    cu_check(cudaMemcpy(r, s.red, sizeof(r), cudaMemcpyDeviceToHost), "sd meters d2h");
    for (int i = 0; i < 3; ++i) meters[i] = std::max(meters[i], bits_to_double(r[i]));
    cu_check(cudaDeviceSynchronize(), "sd block");
    timing[0] += wnow() - t0;
  }

  void ue_sd_flush(unit_engine *e, cplx *S_cst, cplx *RT, cplx *RU) {
    auto &s = e->sd;
    if (not s.on) return;
    const size_t nS = size_t(s.c.nt * s.c.ns * s.c.nk) * size_t(e->nc2), nA = nS * size_t(s.c.np);
    const size_t chunk = size_t(1) << 22;
    std::vector<cplx> tmp(std::min(chunk, std::max(nS, nA)));
    auto pull = [&](cd *d, cplx *h, size_t n) {
      for (size_t o = 0; o < n; o += chunk) {
        const size_t m = std::min(chunk, n - o);
        cu_check(cudaMemcpy(tmp.data(), d + o, m * sizeof(cd), cudaMemcpyDeviceToHost), "sd flush d2h");
        for (size_t i = 0; i < m; ++i) h[o + i] += tmp[i];
      }
      cu_check(cudaMemsetAsync(d, 0, n * sizeof(cd), 0), "sd flush zero");
    };
    pull(s.S, S_cst, nS);
    if (RT != nullptr) pull(s.RT, RT, nA);
    if (RU != nullptr) pull(s.RU, RU, nA);
  }


  // =====================================================================================================================
  // The rung builds (l0_cuda.cuh). b = (k - k0) nk + k' over a chunk of k; U1 / U2 / WU2 (b, Nm, nc2) and wb (b, nc2, nc2)
  // row-major; X is the spin slice (nk, Nm, nc).
  // =====================================================================================================================
  namespace {
    __global__ void kb_legs_kernel(long ik0, long nki, long nk, long Nm, long nc, cd const *__restrict__ X, long const *__restrict__ kq,
                                   cd *__restrict__ U1, cd *__restrict__ U2) {
      const long nc2 = nc * nc, tot = nki * nk * Nm * nc2;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long pq = e % nc2;
        long t = e / nc2;
        const long P = t % Nm;
        t /= Nm;
        const long ikp = t % nk, ik = ik0 + t / nk;
        const long a = pq / nc, bb = pq % nc;
        U1[e] = X[(ikp * Nm + P) * nc + a] * cuConj(X[(ik * Nm + P) * nc + bb]);
        U2[e] = X[(kq[ik] * Nm + P) * nc + a] * cuConj(X[(kq[ikp] * Nm + P) * nc + bb]);
      }
    }
    // K(k' nc2 + (p1 nc + p3'), k nc2 + (p1' nc + p3)) = alpha wb[b](p1 nc + p1', p3 nc + p3') -- a bijection onto K's elements
    __global__ void kb_scatter_kernel(long ik0, long nki, long nk, long nc, cd alpha, cd const *__restrict__ wb, cd *__restrict__ K) {
      const long nc2 = nc * nc, D = nk * nc2, tot = nki * nk * nc2 * nc2;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long col = e % nc2, row = (e / nc2) % nc2, b = e / (nc2 * nc2);
        const long ik = ik0 + b / nk, ikp = b % nk;
        const long p1 = row / nc, p1p = row % nc, p3 = col / nc, p3p = col % nc;
        K[(ikp * nc2 + p1 * nc + p3p) * D + ik * nc2 + p1p * nc + p3] = alpha * wb[e];
      }
    }
    __global__ void herm_kernel(long D, cd const *__restrict__ K, unsigned long long *__restrict__ red) {
      double num = 0.0, den = 0.0;
      const long tot = D * D;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long i = e / D, j = e % D;
        num = fmax(num, nan_inf(cuCabs(K[e] - cuConj(K[j * D + i]))));
        den = fmax(den, nan_inf(cuCabs(K[e])));
      }
      num = block_max(num);
      den = block_max(den);
      if (threadIdx.x == 0) {
        atomicMax(&red[0], (unsigned long long)__double_as_longlong(num));
        atomicMax(&red[1], (unsigned long long)__double_as_longlong(den));
      }
    }
  } // namespace

  bool ue_kb_init(unit_engine *e, long ns, long nq, long Nm, cplx const *Xb, long const *qx_of, cplx const *W0h, cplx const *Wd0h,
                  cplx const *Wdsh, long nrep, double free_bytes, char *why, long why_len) {
    auto &k = e->kb;
    const long nk = e->c.nk, nc2 = e->nc2;
    if (nrep != e->c.ndist) {
      APP_ABORT(" ue_kb_init: rep count " + std::to_string(nrep) + " differs from the engine's ndist " + std::to_string(e->c.ndist) +
                " (internal inconsistency) -- ABORTING.");
      std::snprintf(why, size_t(why_len), "rep count %ld differs from the engine's ndist %ld", nrep, e->c.ndist);
      return false;
    }
    const double per_k = double(nk) * (3.0 * double(Nm * nc2) + double(nc2 * nc2));
    const double need = ue_kb_bytes(ns, nq, Nm, nk, e->c.nc, nrep);
    if (need > 0.9 * free_bytes) {
      std::snprintf(why, size_t(why_len), "needs %.1f GB of device memory, %.1f GB free", need / 1e9, free_bytes / 1e9);
      return false;
    }
    for (long v = 0; v < nk * nk; ++v)   // an invariant, checked before any allocation
      if (qx_of[v] < 0 or qx_of[v] >= nq)
        APP_ABORT(" ue_kb_init: transfer index " + std::to_string(qx_of[v]) + " out of range [0, " + std::to_string(nq) + ") -- ABORTING.");
    k.ns = ns; k.nq = nq; k.Nm = Nm; k.nrep = nrep;
    k.KC = long(std::min<double>(double(nk), std::max(1.0, std::floor(std::min(0.5 * (0.9 * free_bytes - need), 4.0e9) / (16.0 * per_k)))));
    auto al = [&](size_t n, char const *what) { cd *p = dalloc<cd>(n, what); k.allocs.push_back(p); return p; };
    k.X = al(size_t(ns * nk * Nm * e->c.nc), "kb X");
    k.W0 = al(size_t(nq * Nm * Nm), "kb W0"); k.Wd0 = al(size_t(nq * Nm * Nm), "kb Wd0");
    k.Wds = al(size_t(nrep * nq * Nm * Nm), "kb Wds");
    const size_t chunk = size_t(k.KC * nk);
    k.U1 = al(chunk * size_t(Nm * nc2), "kb U1"); k.U2 = al(chunk * size_t(Nm * nc2), "kb U2");
    k.WU2 = al(chunk * size_t(Nm * nc2), "kb WU2"); k.wb = al(chunk * size_t(nc2 * nc2), "kb wb");
    k.kq = dalloc<long>(size_t(nk), "kb kq"); k.allocs.push_back(k.kq);
    k.pA = dalloc<cd *>(chunk, "kb pA"); k.pB = dalloc<cd *>(chunk, "kb pB"); k.pC = dalloc<cd *>(chunk, "kb pC");
    for (void *p : {(void *)k.pA, (void *)k.pB, (void *)k.pC}) k.allocs.push_back(p);
    k.hA.assign(chunk, nullptr); k.hB.assign(chunk, nullptr); k.hC.assign(chunk, nullptr);
    h2d_c(k.X, Xb, size_t(ns * nk * Nm * e->c.nc), "kb X");
    h2d_c(k.W0, W0h, size_t(nq * Nm * Nm), "kb W0"); h2d_c(k.Wd0, Wd0h, size_t(nq * Nm * Nm), "kb Wd0");
    h2d_c(k.Wds, Wdsh, size_t(nrep * nq * Nm * Nm), "kb Wds");
    k.qx.assign(qx_of, qx_of + nk * nk);
    k.on = true;
    return true;
  }

  namespace {
    // the rung tables ts (-2: K_s (x sk), -1: K_d0, r >= 0: K_d(s_r)) of the transfer set by ue_kb_build, into Kts
    void kb_build_tables(unit_engine *e, long const *ts, cd *const *Kts, long nts, cd sk) {
      auto &k = e->kb;
      const long nk = e->c.nk, nc = e->c.nc, nc2 = e->nc2, Nm = k.Nm;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      cd const *Xs = k.X + size_t(k.is) * nk * Nm * nc;
      for (long ik0 = 0; ik0 < nk; ik0 += k.KC) {
        const long nki = std::min(k.KC, nk - ik0), nb = nki * nk;
        if (k.legs_ik0 != ik0) {                           // the legs of this k chunk (kept while one chunk covers the mesh)
          kb_legs_kernel<<<grid_for(nb * Nm * nc2), 256>>>(ik0, nki, nk, Nm, nc, Xs, k.kq, k.U1, k.U2);
          launch_check("kb legs");
          k.legs_ik0 = ik0;
        }
        // the W tables of this transfer: W0 -> K_s (x scale_k), Wd0 -> K_d0, Wd(rep r) -> K_d(r)
        for (long it = 0; it < nts; ++it) {
          const long t = ts[it];
          cd const *Wt = (t == -2) ? k.W0 : ((t == -1) ? k.Wd0 : k.Wds + size_t(t) * k.nq * Nm * Nm);
          cd *Kt = Kts[it];
          for (long b = 0; b < nb; ++b) {
            const long ik = ik0 + b / nk, ikp = b % nk;
            k.hA[size_t(b)] = k.U2 + size_t(b) * Nm * nc2;
            k.hB[size_t(b)] = const_cast<cd *>(Wt) + size_t(k.qx[size_t(ik * nk + ikp)]) * Nm * Nm;
            k.hC[size_t(b)] = k.WU2 + size_t(b) * Nm * nc2;
          }
          h2d(k.pA, k.hA.data(), size_t(nb), "kb pA"); h2d(k.pB, k.hB.data(), size_t(nb), "kb pB"); h2d(k.pC, k.hC.data(), size_t(nb), "kb pC");
          // WU2 = W(qx) U2: column-major (nc2 x Nm) = U2^T-view (nc2 x Nm) . W^T-view (Nm x Nm)
          cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nc2), int(Nm), int(Nm), &one, (const cd **)k.pA, int(nc2),
                                       (const cd **)k.pB, int(Nm), &zero, k.pC, int(nc2), int(nb)), "kb W U2");
          // wb = U1^T WU2: column-major (nc2 x nc2) = WU2-view (nc2 x Nm) . U1-view^T (Nm x nc2)
          cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nc2), int(nc2), int(Nm), &one, k.WU2, int(nc2),
                                              (long long)(Nm * nc2), k.U1, int(nc2), (long long)(Nm * nc2), &zero, k.wb, int(nc2),
                                              (long long)(nc2 * nc2), int(nb)), "kb U1^T W U2");
          kb_scatter_kernel<<<grid_for(nb * nc2 * nc2), 256>>>(ik0, nki, nk, nc, (t == -2) ? sk : one, k.wb, Kt);
          launch_check("kb scatter");
        }
      }
    }
  } // namespace

  namespace {
    // the copy / rebuild split of the non-resident rungs for the next transfer, chosen by simulating the rung pass of
    // ue_rung_dense: the resident gemms run first while the copy stream prefetches the first two copied rungs; then rung j
    // (staging buffer j % 2) either waits for its copy or is rebuilt on the compute stream (serialized with the gemms), and
    // the copy of rung j + 2 starts once rung j's gemm has freed the buffer. Costs per rung: the gemm (timed on the resident
    // rungs), the H2D copy (timed at creation) and the rebuild (timed). For every copied count nh two orders are simulated
    // (copied rungs spread evenly, copied rungs first). The cost of a split is napp simulated passes (napp = the previous
    // transfer's pass count) plus, once per transfer, the device build and the D2H copy of every copied rung.
    // pol_vertex_dyn_device_memory picks the sources (auto: both; rebuild; host); host-built rungs (no device builds) are copied.
    double ue_sim_pass(long nres, std::vector<int> const &src, double tg, double tc, double trb) {
      const long nnr = long(src.size());
      double T = double(nres) * tg, ce = 0.0;               // compute-stream time, copy-engine free time
      std::vector<double> done(size_t(nnr), 0.0);
      for (long j = 0; j < std::min(nnr, 2l); ++j)
        if (src[size_t(j)]) { ce += tc; done[size_t(j)] = ce; }
      for (long j = 0; j < nnr; ++j) {
        T = src[size_t(j)] ? std::max(T, done[size_t(j)]) + tg : T + trb + tg;
        if (j + 2 < nnr and src[size_t(j + 2)]) { ce = std::max(ce, T) + tc; done[size_t(j + 2)] = ce; }
      }
      return T;
    }
    void ue_plan_sources(unit_engine *e) {
      const long nnr = e->c.ndist - e->nres;
      if (nnr <= 0) return;
      const bool can_rb = e->kb.on and e->c.nonres_src != 2;
      const bool can_cp = (e->Kds_host != nullptr) and e->c.nonres_src != 1;
      if (not can_rb and not can_cp) APP_ABORT(std::string(" ue_plan_sources: no source for the non-resident rungs -- ABORTING."));
      std::vector<int> best_src(size_t(nnr), can_cp ? 1 : 0);
      if (can_rb and can_cp) {
        const double nk = double(e->c.nk), nc2 = double(e->nc2), Nm = double(e->kb.Nm), D = double(e->D);
        // before the first timings: flop counts at an assumed 10 TF/s (replaced by the timed costs afterwards)
        if (e->t_rb_est <= 0.0) e->t_rb_est = nk * nk * 8.0 * (nc2 * Nm * Nm + nc2 * nc2 * Nm) / 1.0e13;
        const double tg = (e->t_gemm_est > 0.0)
            ? e->t_gemm_est : 8.0 * D * D * double(e->c.nt * e->c.nR_max) / double(std::max(e->c.ndist, 1l)) / 1.0e13;
        const double napp = double(e->napp_prev > 0 ? e->napp_prev : 4);
        double best = 1e300;
        std::vector<int> src(static_cast<size_t>(nnr));
        for (long h = 0; h <= std::min(nnr, e->nhost_cap); ++h)
          for (int order = 0; order < 2; ++order) {
            for (long jj = 0; jj < nnr; ++jj)
              src[size_t(jj)] = (order == 0) ? (((jj + 1) * h / nnr > jj * h / nnr) ? 1 : 0) : (jj < h ? 1 : 0);
            const double t = napp * ue_sim_pass(e->nres, src, tg, e->t_copy_est, e->t_rb_est) +
                             double(h) * (e->t_rb_est + e->t_copy_est);
            if (t < best * (1.0 - 1e-9)) { best = t; best_src = src; }
          }
      }
      e->src_host = best_src;
    }
  } // namespace

  void ue_kb_build(unit_engine *e, long is, long const *kpq_row, long const *trep, cplx scale_k, double *herm, double *timing) {
    auto &k = e->kb;
    if (not k.on) APP_ABORT(std::string(" ue_kb_build: the rung builds were not initialized."));
    const double t0 = wnow();
    const long nk = e->c.nk, D = e->D;
    h2d(k.kq, kpq_row, size_t(nk), "kb kq");
    k.is = is;
    k.legs_ik0 = -1;
    const cd sk = make_cuDoubleComplex(scale_k.real(), scale_k.imag());
    // K_s, K_d0 and the RESIDENT K_d(s_r); the non-resident ones: the copy / rebuild split is planned for this transfer and the
    // copied ones are built now (timed: the rebuild estimate) and stored in their pinned host slots
    const cd one = make_cuDoubleComplex(1.0, 0.0);
    std::vector<long> ts = {-2, -1};
    std::vector<cd *> Kts = {e->Ks, e->Kd0};
    if (e->stream) { ts.pop_back(); Kts.pop_back(); }        // stream mode: K_s only (the rung streams)
    for (long r = 0; r < e->nres; ++r) { ts.push_back(r); Kts.push_back(e->Kds + size_t(r) * size_t(D) * size_t(D)); }
    kb_build_tables(e, ts.data(), Kts.data(), long(ts.size()), sk);
    const long nnr = e->stream ? 0 : e->c.ndist - e->nres;
    if (nnr > 0) {
      if (e->napp_cur > 0) e->napp_prev = e->napp_cur;
      e->napp_cur = 0;
      ue_plan_sources(e);
      const size_t DD = size_t(D) * size_t(D);
      double tb_sum = 0.0;
      long nb = 0;
      for (long jj = 0; jj < nnr; ++jj) {
        if (not e->src_host[size_t(jj)]) continue;
        const long r = e->nres + jj;
        cd *K = e->stage[0];
        cu_check(cudaDeviceSynchronize(), "kb host-rep sync");
        const double tb = wnow();
        kb_build_tables(e, &r, &K, 1, one);
        cu_check(cudaDeviceSynchronize(), "kb host-rep build");
        tb_sum += wnow() - tb; ++nb;
        cu_check(cudaMemcpy(e->Kds_host + size_t(jj) * DD, e->stage[0], DD * sizeof(cd), cudaMemcpyDeviceToHost), "kb host rep D2H");
      }
      if (nb > 0) e->t_rb_est = tb_sum / double(nb);
    }
    // the Sigma hook's |K_s - K_s^dag| meter
    cu_check(cudaMemsetAsync(e->red, 0, 2 * sizeof(unsigned long long), 0), "kb red");
    herm_kernel<<<grid_red(D * D), 256>>>(D, e->Ks, e->red);
    launch_check("kb herm");
    unsigned long long r[2] = {0, 0};
    cu_check(cudaMemcpy(r, e->red, sizeof(r), cudaMemcpyDeviceToHost), "kb herm d2h");
    herm[0] = meter_value(r[0], "K_s (rung build)"); herm[1] = meter_value(r[1], "K_s (rung build)");
    e->trep.assign(trep, trep + e->c.nt);
    e->scale = sk;
    e->unit_ok = false;
    cu_check(cudaDeviceSynchronize(), "kb build");
    timing[0] += wnow() - t0;
  }

  void ue_get_ks(unit_engine *e, cplx *Ks) {
    cu_check(cudaMemcpy(Ks, e->Ks, size_t(e->D) * size_t(e->D) * sizeof(cd), cudaMemcpyDeviceToHost), "ue Ks d2h");
  }

  // ---- small device utilities for the host translation units (scr_coulomb_t's Pi hooks, vertex_t::build_w0) -----------
  namespace {
    __global__ void add_rows_kernel(long rows, long n, cd *__restrict__ y, long ldy, cd const *__restrict__ x, long ldx) {
      const long tot = rows * n;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long r = e / n, i = e - r * n;
        y[r * ldy + i] = y[r * ldy + i] + x[r * ldx + i];
      }
    }
    __global__ void add_diag_kernel(long n, long ld, cd *__restrict__ a, double s) {
      for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x)
        a[i * ld + i] = a[i * ld + i] + real(s);
    }
  } // namespace

  double dev_maxabs(cplx const *x, long n) {
    if (n <= 0) return 0.0;
    unsigned long long *slot = nullptr;
    cu_check(cudaMalloc(&slot, sizeof(unsigned long long)), "dev_maxabs slot");
    cu_check(cudaMemsetAsync(slot, 0, sizeof(unsigned long long), 0), "dev_maxabs zero");
    maxabs_kernel<<<grid_red(n), 256>>>(n, reinterpret_cast<cd const *>(x), slot);
    launch_check("dev_maxabs");
    unsigned long long r = 0;
    cu_check(cudaMemcpy(&r, slot, sizeof(r), cudaMemcpyDeviceToHost), "dev_maxabs d2h");
    cu_check(cudaFree(slot), "dev_maxabs free");
    return meter_value(r, "dev_maxabs");
  }

  void dev_add_rows(cplx *y, long ldy, cplx const *x, long ldx, long rows, long n) {
    if (rows <= 0 or n <= 0) return;
    add_rows_kernel<<<grid_for(rows * n), 256>>>(rows, n, reinterpret_cast<cd *>(y), ldy, reinterpret_cast<cd const *>(x), ldx);
    launch_check("dev_add_rows");
    cu_check(cudaDeviceSynchronize(), "dev_add_rows");
  }

  void dev_add_diag(cplx *a, long n, long ld, double s) {
    if (n <= 0) return;
    add_diag_kernel<<<grid_for(n), 256>>>(n, ld, reinterpret_cast<cd *>(a), s);
    launch_check("dev_add_diag");
    cu_check(cudaDeviceSynchronize(), "dev_add_diag");
  }

  // ---- the Sigma hook's finish (vertex_sigma_dyn.icc::sigma_dyn_finish) on the device -----------------------------------
  namespace {
    // E(it; isk, a, ij) = KF(it, a) RT(it; isk, a, ij), in place
    __global__ void fin_scale_kf_kernel(long nt, long nsk, long np, long nc2, double const *__restrict__ KF, cd *__restrict__ E) {
      const long tot = nt * nsk * np * nc2;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long it = e / (nsk * np * nc2), a = (e / nc2) % np;
        E[e] = scal(KF[it * np + a], E[e]);
      }
    }
    // ecP(isk; a, pp, ij) = ec(pp; isk, a, ij)
    __global__ void fin_permute_kernel(long npf, long nsk, long np, long nc2, cd const *__restrict__ ec, cd *__restrict__ ecP) {
      const long tot = npf * nsk * np * nc2, nb = nsk * np * nc2;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long ij = e % nc2;
        long r = e / nc2;
        const long pp = r % npf;
        r /= npf;
        const long a = r % np, isk = r / np;
        ecP[e] = ec[pp * nb + (isk * np + a) * nc2 + ij];
      }
    }
    // out(it; isk, ij) = S(it; isk, ij) + sum_p KF(it, p) RU(it; isk, p, ij)
    __global__ void fin_u_kernel(long nt, long nsk, long np, long nc2, double const *__restrict__ KF, cd const *__restrict__ S,
                                 cd const *__restrict__ RU, cd *__restrict__ out) {
      const long tot = nt * nsk * nc2;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long ij = e % nc2, r = e / nc2, isk = r % nsk, it = r / nsk;
        cd acc = S[e];
        cd const *ru = RU + ((it * nsk + isk) * np) * nc2 + ij;
        for (long p = 0; p < np; ++p) acc = acc + scal(KF[it * np + p], ru[p * nc2]);
        out[e] = acc;
      }
    }
  } // namespace

  void sd_finish(long nt, long nsk, long np, long npf, long nc2, double const *KF, double const *Tpp, long nkept, cplx const *Ut,
                 cplx const *Vs, double const *Kmap, cplx const *S_cst, cplx const *RT, cplx const *RU, bool anyT, cplx pref,
                 cplx *dSig, double *fit_err) {
    const long nb = nsk * np * nc2, nS = nt * nsk * nc2;
    cublasHandle_t h = nullptr;
    cub_check(cublasCreate(&h), "fin handle");
    double *dKF = upload<double>(KF, size_t(nt * np), "fin KF");
    cd *dS = upload_c(S_cst, size_t(nS), "fin S");
    cd *dRU = upload_c(RU, size_t(nt * nb), "fin RU");
    cd *dOut = dalloc<cd>(size_t(nS), "fin out");
    fin_u_kernel<<<grid_for(nS), 256>>>(nt, nsk, np, nc2, dKF, dS, dRU, dOut);
    launch_check("fin U");
    cu_check(cudaFree(dRU), "fin free RU");
    cu_check(cudaFree(dS), "fin free S");
    *fit_err = 0.0;
    if (anyT) {
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      cd *dE = upload_c(RT, size_t(nt * nb), "fin RT");
      fin_scale_kf_kernel<<<grid_for(nt * nb), 256>>>(nt, nsk, np, nc2, dKF, dE);
      launch_check("fin E");
      // ec (npf x nb) = Vs (npf x nkept) . [Ut (nkept x nt) . E (nt x nb)]: the fit's two factors applied in turn, as on the
      // host -- the explicit product Vs Ut carries rounding ~eps / s_min whatever the data
      cd *dU = upload_c(Ut, size_t(nkept * nt), "fin Ut");
      cd *dV = upload_c(Vs, size_t(npf * nkept), "fin Vs");
      cd *dg = dalloc<cd>(size_t(nkept * nb), "fin g");
      cd *dec = dalloc<cd>(size_t(npf * nb), "fin ec");
      cub_check(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nb), int(nkept), int(nt), &one, dE, int(nb), dU, int(nt), &zero, dg,
                            int(nb)), "fin g");
      cub_check(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nb), int(npf), int(nkept), &one, dg, int(nb), dV, int(nkept), &zero, dec,
                            int(nb)), "fin ec");
      cu_check(cudaFree(dg), "fin free g");
      // the refit error: max |E - Kmap ec| / max |E|
      std::vector<cplx> Kc(size_t(nt * npf));
      for (long i = 0; i < nt * npf; ++i) Kc[size_t(i)] = cplx(Kmap[i], 0.0);
      cd *dK = upload_c(Kc.data(), Kc.size(), "fin Kmap");
      cd *drec = dalloc<cd>(size_t(nt * nb), "fin rec");
      cub_check(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nb), int(nt), int(npf), &one, dec, int(nb), dK, int(npf), &zero, drec,
                            int(nb)), "fin rec");
      unsigned long long *red = dalloc<unsigned long long>(2, "fin red");
      cu_check(cudaMemsetAsync(red, 0, 2 * sizeof(unsigned long long), 0), "fin red zero");
      maxdiff_kernel<<<grid_red(nt * nb), 256>>>(nt * nb, dE, drec, red);
      launch_check("fin fit error");
      unsigned long long r[2] = {0, 0};
      cu_check(cudaMemcpy(r, red, sizeof(r), cudaMemcpyDeviceToHost), "fin red d2h");
      const double num = meter_value(r[0], "the Sigma finish fit"), den = meter_value(r[1], "the Sigma finish fit");
      *fit_err = (den > 0.0) ? num / den : num;
      cu_check(cudaFree(red), "fin free red");
      cu_check(cudaFree(drec), "fin free rec");
      cu_check(cudaFree(dK), "fin free K");
      cu_check(cudaFree(dU), "fin free Ut");
      cu_check(cudaFree(dV), "fin free Vs");
      cu_check(cudaFree(dE), "fin free E");
      // ecP(isk; a, pp, ij), then out(it; isk, ij) += sum_{(a, pp)} Tpp(it; a, pp) ecP(isk; (a, pp), ij): one strided batched
      // gemm over isk -- column-major out_isk^T (nc2 x nt, ld nsk nc2) += ecP_isk^T (nc2 x R) . Tpp^T (R x nt), R = np npf
      cd *decP = dalloc<cd>(size_t(npf * nb), "fin ecP");
      fin_permute_kernel<<<grid_for(npf * nb), 256>>>(npf, nsk, np, nc2, dec, decP);
      launch_check("fin permute");
      cu_check(cudaFree(dec), "fin free ec");
      const long R = np * npf;
      std::vector<cplx> Tc(size_t(nt * R));
      for (long i = 0; i < nt * R; ++i) Tc[size_t(i)] = cplx(Tpp[i], 0.0);
      cd *dT = upload_c(Tc.data(), Tc.size(), "fin Tpp");
      cub_check(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc2), int(nt), int(R), &one, decP, int(nc2),
                                          (long long)(R * nc2), dT, int(R), 0LL, &one, dOut, int(nsk * nc2), (long long)nc2,
                                          int(nsk)), "fin T");
      cu_check(cudaFree(dT), "fin free T");
      cu_check(cudaFree(decP), "fin free ecP");
    }
    const cd pr = make_cuDoubleComplex(pref.real(), pref.imag());
    cub_check(cublasZscal(h, int(nS), &pr, dOut, 1), "fin scale");
    cu_check(cudaMemcpy(dSig, dOut, size_t(nS) * sizeof(cd), cudaMemcpyDeviceToHost), "fin d2h");
    cu_check(cudaFree(dOut), "fin free out");
    cu_check(cudaFree(dKF), "fin free KF");
    cub_check(cublasDestroy(h), "fin handle destroy");
  }


  // =====================================================================================================================
  // THE DRESSED-LEG GAMMA_1 READOUT ON THE DEVICE (the host twin: vertex_dynbse.icc dyn_dressed branch,
  // dynbse.hpp build_dressed_grams / dressed_zt). Per block: d~ = D + T_s Cb D, r = L0 d~, y1 = K_d(r), then
  //   Zt = sum_n [ g_n^T Z^U_n Gh_n^T + (Gt_n^T (inu Z^T_n - Z^U_n) + g_n^T Z^T_n) g'_n^T ] + Cb y1.cst,  Z = H y1.fam,
  //   Pd (nout, nR) = e~^dag Zt.   e~ is built once per unit in conjugate space: conj(e~) = conj(D) + M^-T K_s^T Cb^T conj(D)
  //   (the device LU is of the column-major M^T, so the solve is getrs OP_N). No L0 on the frequency-dependent y1.
  // =====================================================================================================================
  namespace {
    __global__ void dr_zc_kernel(long n, cd iv, cd const *__restrict__ ZT, cd const *__restrict__ ZU, cd *__restrict__ Zc) {
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x)
        Zc[e] = cuCsub(cuCmul(iv, ZT[e]), ZU[e]);
    }
    // Zt[k][a'][b'][N] = sum_n sum_b Sa[n][k][a'][b][N] Gh[n][k][b'][b] + Sbc[n][k][a'][b][N] gkq[n][k][b'][b]
    __global__ void dr_rsand_kernel(long ng, long nk, long nc, long nR, cd const *__restrict__ Sa, cd const *__restrict__ Sbc,
                                    cd const *__restrict__ Gh, cd const *__restrict__ gkq, cd *__restrict__ Zt) {
      const long tot = nk * nc * nc * nR;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < tot; e += long(gridDim.x) * blockDim.x) {
        const long N = e % nR;
        long t = e / nR;
        const long bp = t % nc;
        t /= nc;
        const long ap = t % nc, k = t / nc;
        cd acc = make_cuDoubleComplex(0.0, 0.0);
        for (long n = 0; n < ng; ++n) {
          const long bs = (((n * nk + k) * nc + ap) * nc) * nR + N, bb = ((n * nk + k) * nc + bp) * nc;
          for (long b = 0; b < nc; ++b) {
            acc = cuCadd(acc, cuCmul(Sa[bs + b * nR], Gh[bb + b]));
            acc = cuCadd(acc, cuCmul(Sbc[bs + b * nR], gkq[bb + b]));
          }
        }
        Zt[e] = acc;
      }
    }
    // Zt (D, nR) from y1 = (fam, cst) of width nR
    void dr_zt(unit_engine *e, long nR, cplx inu, cd const *yfam, cd const *ycst) {
      auto &d = e->dr;
      const long ng = d.ng, np = e->c.np, nk = e->c.nk, nc = e->c.nc, nc2 = e->nc2, W = e->D * nR;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      // Z (2ng, W) = H (2ng, 2np) . yfam (2np, W)   [row-major; column-major Z^T = yfam^T H^T]
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(W), int(2 * ng), int(2 * np), &one, yfam, int(W), d.H, int(2 * np),
                            &zero, d.Z, int(W)), "dr Z = H y");
      cd const *ZU = d.Z, *ZT = d.Z + size_t(ng) * W;
      dr_zc_kernel<<<grid_for(ng * W), 256>>>(ng * W, make_cuDoubleComplex(inu.real(), inu.imag()), ZT, ZU, d.Zc);
      launch_check("dr Zc");
      // S[n][k] (nc x nc nR) = A[n][k]^T Z[n][k] over the (n, k) batch: column-major S^T = Z^T A (A's storage = A^T -> OP_T)
      const long bnk = ng * nk, sz = nc2 * nR;
      auto sbatch = [&](cd const *Zx, cd const *A, cd beta, cd *S, char const *what) {
        cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nc * nR), int(nc), int(nc), &one, Zx, int(nc * nR),
                                            (long long)sz, A, int(nc), (long long)nc2, &beta, S, int(nc * nR), (long long)sz,
                                            int(bnk)), what);
      };
      sbatch(ZU, d.gk, zero, d.Sa, "dr Sa = g^T ZU");
      sbatch(ZT, d.gk, zero, d.Sbc, "dr Sbc = g^T ZT");
      sbatch(d.Zc, d.Gt, one, d.Sbc, "dr Sbc += Gt^T (inu ZT - ZU)");
      dr_rsand_kernel<<<grid_for(W), 256>>>(ng, nk, nc, nR, d.Sa, d.Sbc, d.Gh, d.gkq, d.Zt);
      launch_check("dr rsand");
      ue_cb_times(e, nR, ycst, d.Zt, true);                  // + Cb y1.cst
    }
    // Ph (nout, nR) row-major = Lj^dag-collapse: P^T (nR x nout) = Zt^T-view . Lj (the ue_readout arithmetic with the legs Lj)
    void dr_collapse(unit_engine *e, long nR, cd const *Zt, cd const *Lj, cplx *Ph) {
      const long D = e->D, nout = e->c.nout;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_T, int(nR), int(nout), int(D), &one, Zt, int(nR), Lj, int(nout), &zero,
                            e->Pr, int(nR)), "dr collapse");
      cu_check(cudaMemcpy(Ph, e->Pr, size_t(nout * nR) * sizeof(cd), cudaMemcpyDeviceToHost), "dr collapse d2h");
    }
  } // namespace

  bool ue_dressed_prepare(unit_engine *e, long ng, cplx const *Hh, cplx const *Ghh, cplx const *Gth, cplx const *gkh,
                          cplx const *gkqh, bool ts_zero, char *why, long why_len) {
    auto &d = e->dr;
    const long np = e->c.np, nk = e->c.nk, nc2 = e->nc2, D = e->D, nout = e->c.nout, R = e->c.nR_max;
    const size_t W = size_t(D) * size_t(R);
    if (not e->legs_ok) { std::snprintf(why, size_t(why_len), "the legs are not set"); return false; }
    if (not d.on or d.ng != ng) {
      for (void *p : d.allocs) if (p) (void)cudaFree(p);
      d.allocs.clear();
      const double need = ue_dressed_bytes(ng, np, nk, e->c.nc, nout, R);
      (void)W;
      size_t fr = 0, tot = 0;
      cu_check(cudaMemGetInfo(&fr, &tot), "dr memgetinfo");
      if (need > 0.9 * double(fr)) {
        std::snprintf(why, size_t(why_len), "needs %.1f GB of device memory, %.1f GB free", need / 1e9, double(fr) / 1e9);
        return false;
      }
      auto al = [&](size_t n, char const *w) { cd *p = dalloc<cd>(n, w); d.allocs.push_back(p); return p; };
      d.H = al(size_t(4 * ng * np), "dr H");
      d.Gh = al(size_t(ng * nk * nc2), "dr Gh"); d.Gt = al(size_t(ng * nk * nc2), "dr Gt");
      d.gk = al(size_t(ng * nk * nc2), "dr gk"); d.gkq = al(size_t(ng * nk * nc2), "dr gkq");
      d.Etj = al(size_t(D * nout), "dr Etj");
      d.Z = al(2 * size_t(ng) * W, "dr Z"); d.Zc = al(size_t(ng) * W, "dr Zc");
      d.Sa = al(size_t(ng) * W, "dr Sa"); d.Sbc = al(size_t(ng) * W, "dr Sbc"); d.Zt = al(W, "dr Zt");
      d.ng = ng; d.nR_max = R; d.on = true;
    }
    h2d_c(d.H, Hh, size_t(4 * ng * np), "dr H"); h2d_c(d.Gh, Ghh, size_t(ng * nk * nc2), "dr Gh");
    h2d_c(d.Gt, Gth, size_t(ng * nk * nc2), "dr Gt"); h2d_c(d.gk, gkh, size_t(ng * nk * nc2), "dr gk");
    h2d_c(d.gkq, gkqh, size_t(ng * nk * nc2), "dr gkq");
    if (ts_zero) {
      cu_check(cudaMemcpy(d.Etj, e->Dcj, size_t(D * nout) * sizeof(cd), cudaMemcpyDeviceToDevice), "dr Etj = Dcj");
      return true;
    }
    if (not e->unit_ok) { std::snprintf(why, size_t(why_len), "the unit's LU is not set"); return false; }
    // conjugate space, column-major (D x nout): X0 = Dcj^T; w = blockdiag(Cb^T) X0; u = K_s^T w; u <- M^-T u; X0 += u; Etj = X0^T
    const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    cd *X0 = dalloc<cd>(size_t(D * nout), "dr X0"), *w = dalloc<cd>(size_t(D * nout), "dr w"), *u = dalloc<cd>(size_t(D * nout), "dr u");
    cub_check(cublasZgeam(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(D), int(nout), &one, e->Dcj, int(nout), &zero, X0, int(D), X0, int(D)),
              "dr X0");
    cub_check(cublasZgemmStridedBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nc2), int(nout), int(nc2), &one, e->Cbk, int(nc2),
                                        (long long)(nc2 * nc2), X0, int(D), (long long)nc2, &zero, w, int(D), (long long)nc2, int(nk)),
              "dr w = Cb^T X0");
    cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(D), int(nout), int(D), &one, e->Ks, int(D), w, int(D), &zero, u, int(D)),
              "dr u = Ks^T w");
    if (cusolverDnZgetrs(e->cs, CUBLAS_OP_N, int(D), int(nout), e->M, int(D), e->ipiv, u, int(D), e->dinfo) != CUSOLVER_STATUS_SUCCESS)
      APP_ABORT(std::string(" ue_dressed_prepare: cusolverDnZgetrs failed."));
    {   // getrs' devInfo (an illegal argument is reported there, not in the status)
      int info = 0;
      cu_check(cudaMemcpy(&info, e->dinfo, sizeof(int), cudaMemcpyDeviceToHost), "dr getrs info");
      if (info != 0) APP_ABORT(" ue_dressed_prepare: cusolverDnZgetrs devInfo = " + std::to_string(info) + " -- ABORTING.");
    }
    cub_check(cublasZgeam(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(D), int(nout), &one, X0, int(D), &one, u, int(D), X0, int(D)), "dr X0 += u");
    cub_check(cublasZgeam(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nout), int(D), &one, X0, int(D), &zero, d.Etj, int(nout), d.Etj, int(nout)),
              "dr Etj");
    cu_check(cudaDeviceSynchronize(), "dr prepare");
    for (cd *p : {X0, w, u}) (void)cudaFree(p);
    return true;
  }

  double ue_gamma1_dressed(unit_engine *e, long nR, cplx const *Dblkh, cplx inu, bool ts_zero, bool want_r1, cplx *Pdh, cplx *Pr1h,
                           double *tim, bool want_gsum1) {
    if (not e->dr.on) APP_ABORT(std::string(" ue_gamma1_dressed: ue_dressed_prepare was not called."));
    if (nR > e->c.nR_max) APP_ABORT(std::string(" ue_gamma1_dressed: block wider than the engine's nR_max."));
    const long W = e->D * nR;
    double t0 = wnow();
    h2d_c(e->Dblk, Dblkh, size_t(W), "dr Dblk");
    e->cbd_ok = false;
    tim[5] += wnow() - t0;
    // d~ = D + T_s Cb D  (into Xcst)
    if (ts_zero) {
      cu_check(cudaMemcpy(e->Xcst, e->Dblk, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "dr d~ = D");
    } else {
      ue_cb_times(e, nR, e->Dblk, e->Xcst, false);
      ue_ts(e, nR, e->Xcst, &tim[1]);
      add2_kernel<<<grid_for(W), 256>>>(W, e->Dblk, e->csb, e->Xcst);
      launch_check("dr d~");
    }
    // r = L0 d~ (Fsum = Cb d~: the static column's Gsum), y1 = K_d(r); with the one bare dynamic rung (T_s = 0: d~ = e~ = D) as
    // a second input of the same rung pass when the engine packs two (its L0 into F2fam / F2sum, its y into yfam1 / ycst1)
    ue_l0(e, nR, nullptr, e->Xcst, e->Ffam, e->Fsum, &tim[0]);
    cu_check(cudaMemcpy(e->Gs0, e->Fsum, size_t(W) * sizeof(cd), cudaMemcpyDeviceToDevice), "dr Gs0");
    const bool r1_here = want_r1 and not ts_zero and not want_gsum1;
    const bool r1_fused = r1_here and e->fuse;
    double fe = 0.0;
    if (r1_fused) {
      ue_l0(e, nR, nullptr, e->Dblk, e->F2fam, e->F2sum, &tim[0]);
      cd const *gin[2] = {e->Ffam, e->F2fam};
      cd const *sin[2] = {e->Fsum, e->F2sum};
      cd *yo[2] = {e->yfam, e->yfam1};
      cd *co[2] = {e->ycst, e->ycst1};
      double fes[2] = {0.0, 0.0};
      ue_kd_multi(e, nR, 2, gin, sin, yo, co, fes, tim);
      fe = fes[0];                                        // the one-bare-rung input's refit error is not reported (as unfused)
    } else {
      fe = ue_kd(e, nR, e->Ffam, e->Fsum, e->yfam, e->ycst, tim);
    }
    t0 = wnow();
    dr_zt(e, nR, inu, e->yfam, e->ycst);
    dr_collapse(e, nR, e->dr.Zt, e->dr.Etj, Pdh);
    tim[4] += wnow() - t0;
    if (want_gsum1) {
      // the Sigma deposits (ue_sd_block) read Gsum1 = Cb d~ + L + Cb T_s L, L = (L0 y1)^sum = Zt, and y1 = (yfam, ycst) in place
      if (ts_zero) {
        add2_kernel<<<grid_for(W), 256>>>(W, e->Gs0, e->dr.Zt, e->Gsum);
      } else {
        ue_ts(e, nR, e->dr.Zt, &tim[1]);
        add2_kernel<<<grid_for(W), 256>>>(W, e->Gs0, e->dr.Zt, e->Gsum);
        add2_kernel<<<grid_for(W), 256>>>(W, e->Gsum, e->cbb, e->Gsum);
      }
      launch_check("dr Gsum1");
      cu_check(cudaDeviceSynchronize(), "dr gsum1");
    }
    e->r1_ok = false;
    if (r1_here) {
      // the one bare dynamic rung (T_s = 0): d~ = e~ = D
      if (not r1_fused) {
        ue_l0(e, nR, nullptr, e->Dblk, e->Ffam, e->Fsum, &tim[0]);
        (void)ue_kd(e, nR, e->Ffam, e->Fsum, e->yfam, e->ycst, tim);
      }
      t0 = wnow();
      if (r1_fused) dr_zt(e, nR, inu, e->yfam1, e->ycst1);
      else dr_zt(e, nR, inu, e->yfam, e->ycst);
      dr_collapse(e, nR, e->dr.Zt, e->Dcj, Pr1h);
      tim[4] += wnow() - t0;
    }
    return fe;
  }


  void ue_set_rung_mode(unit_engine *e, bool rung_pair, long rr, cplx const *ctr_h) {
    // the frequency-factorized rung keeps its R matrices resident (its pass mixes every K_r into every tau column)
    if (rr > 0 and e->nres < e->c.ndist)
      APP_ABORT(std::string(" ue_set_rung_mode: the frequency-factorized rung needs all ") + std::to_string(e->c.ndist) +
                " K_r resident on the device (" + std::to_string(e->nres) + " fit) -- ABORTING.");
    e->rung_pair = rung_pair;
    e->rr = rr;
    if (rr > 0) {
      if (e->ctr) (void)cudaFree(e->ctr);
      e->ctr = dalloc<cd>(size_t(e->c.nt * rr), "rung ctr");
      h2d_c(e->ctr, ctr_h, size_t(e->c.nt * rr), "rung ctr");
    }
  }

} // namespace methods::solvers::dynbse_cuda
