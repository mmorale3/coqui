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
#include <cusolverDn.h>
#include <cstring>
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
      pack_kernel<<<dim3(64u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, nu0, dX, dXc, dVt);
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
      if (nu0)
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
                    (void *)pPj, (void *)pQj, (void *)pBj})
      cu_check(cudaFree(p), "cudaFree");
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
  // THE DEVICE-RESIDENT UNIT (l0_cuda.cuh, gpu port plan section 4d). Part 1: the resident L0 plan -- the kernels above,
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
    __global__ void add2_kernel(long n, cd const *__restrict__ a, cd const *__restrict__ b, cd *__restrict__ out) {
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) out[e] = a[e] + b[e];
    }
    __global__ void add_identity_kernel(long D, cd *__restrict__ M) {
      for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < D; i += long(gridDim.x) * blockDim.x)
        M[i * D + i] = M[i * D + i] + real(1.0);
    }
    // max |a - b| and max |a| over n elements into red[0], red[1] (non-negative doubles: the bit patterns order like the values)
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
    __global__ void maxdiff_kernel(long n, cd const *__restrict__ a, cd const *__restrict__ b, unsigned long long *__restrict__ red) {
      double num = 0.0, den = 0.0;
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) {
        num = fmax(num, cuCabs(a[e] - b[e]));
        den = fmax(den, cuCabs(a[e]));
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
      for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < n; e += long(gridDim.x) * blockDim.x) m = fmax(m, cuCabs(x[e]));
      m = block_max(m);
      if (threadIdx.x == 0) atomicMax(slot, (unsigned long long)__double_as_longlong(m));
    }
    inline unsigned grid_for(long n) { return unsigned(std::min<long>((n + 255) / 256, 65535)); }
    inline unsigned grid_red(long n) { return unsigned(std::max<long>(1, std::min<long>((n + 255) / 256, 1024))); }

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
    // per-k bytes of the working set at (nca_max, nR_max)
    static double per_k_bytes(long np, long nc, long ng, long nR) {
      const long nca = 1 + 2 * np, W = nc * nca * nR * nc, blk = nc * nR * nc, asz = 2 * np * blk;
      return double(W + 3 * ng * W + 5 * asz + 3 * ng * nc * nc) * 16.0 + double(7 * ng) * 8.0;
    }
    static double fixed_bytes(long np, long nk, long nc, long ng) {
      const double nc2 = double(nc * nc);
      return (2.0 * ng * nk * nc2 * 2.0 + 10.0 * np * np + 6.0 * np) * 16.0 + 1.0 + 2.0 * np * 8.0;
    }
    l0_plan(long np_, long np_fit_, long nk_, long nc_, long ng_, long nR_max_, double free_bytes)
        : np(np_), np_fit(np_fit_), nk(nk_), nc(nc_), ng(ng_), nR_max(nR_max_), nca_max(1 + 2 * np_) {
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
      const double pk = per_k_bytes(np, nc, ng, nR_max);
      K = std::max(1L, std::min(nk, long((0.85 * free_bytes - fixed_bytes(np, nk, nc, ng)) / pk)));
      // allocate with the same halving retry as run_l0 (a shared device)
      auto try_alloc = [&](void **p, size_t bytes) -> bool {
        const cudaError_t e = cudaMalloc(p, std::max<size_t>(bytes, 1));
        if (e != cudaSuccess) { (void)cudaGetLastError(); *p = nullptr; return false; }
        return true;
      };
      auto free_ws = [&]() {
        for (void **p : {(void **)&dVt, (void **)&dPj, (void **)&dQj, (void **)&dBj, (void **)&dgjT, (void **)&dglT, (void **)&dGh,
                         (void **)&dAU, (void **)&dAT, (void **)&dM2, (void **)&dA1, (void **)&dA3, (void **)&pVt, (void **)&pGjT,
                         (void **)&pGlT, (void **)&pGh, (void **)&pPj, (void **)&pQj, (void **)&pBj})
          if (*p != nullptr) { (void)cudaFree(*p); *p = nullptr; }
      };
      bool ok = false;
      while (true) {
        const size_t nptr = size_t(K) * size_t(ng);
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
                      (void *)pPj, (void *)pQj, (void *)pBj})
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
    }
    /** F = L0 X on DEVICE buffers (F, Fs overwritten); act_h = the active components (host), sum_part1 as l0_tables */
    void apply(long nR, long nca, long const *act_h, bool skip_cst, bool sum_part1, cd const *Xfam, cd const *Xcst, cd *F, cd *Fs) {
      const long nc2 = nc * nc, D = nk * nc2;
      const size_t nF = size_t(2 * np) * D * nR, nFs = size_t(D) * nR;
      cu_check(cudaMemset(F, 0, nF * sizeof(cd)), "plan memset F");
      cu_check(cudaMemset(Fs, 0, nFs * sizeof(cd)), "plan memset Fs");
      if (nca == 0) return;                                        // nothing to apply: F and Fs are zero
      if (nca > (nu0 ? 1 + np : 1 + 2 * np)) APP_ABORT(std::string(" l0_plan::apply: too many active components."));
      h2d(dact, act_h, size_t(nca), "plan act");
      kdims kd{nc, nR, np, ng, nca, nc * nR * nc, nk, np_fit, dact};
      const long blk = nc * nR * nc, W = nc * nca * nR * nc;
      const size_t asz = size_t(2 * np) * size_t(blk);
      const size_t nptr = size_t(K) * size_t(ng);
      fill_ptrs<<<unsigned((nptr + 255) / 256), 256>>>(K, ng, W, nc2, dVt, dgjT, dglT, dGh, dPj, dQj, dBj,
                                                       pVt, pGjT, pGlT, pGh, pPj, pQj, pBj);
      launch_check("plan fill_ptrs");
      cu_check(cudaMemset(derr, 0, sizeof(int)), "plan memset err");
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      const int Ncols = int(nca * nR * nc), Mrows = int(nc * nca * nR);
      for (long ik0 = 0; ik0 < nk; ik0 += K) {
        const long Kb = std::min(K, nk - ik0);
        const int nb = int(Kb * ng);
        for (cd *p : {dAU, dAT, dM2, dA1, dA3}) cu_check(cudaMemset(p, 0, size_t(Kb) * asz * sizeof(cd)), "plan memset acc");
        pack_kernel<<<dim3(64u, unsigned(Kb), 1u), 256>>>(kd, ik0, Kb, nu0, Xfam, Xcst, dVt);
        launch_check("plan pack_kernel");
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
        if (nu0)
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
  // D-3: the Sigma deposits' device state (ue_sd_*); every buffer is in `allocs` (freed by ue_destroy)
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

  struct unit_engine {
    ue_config c;
    sd_state sd;
    cd *Gs0 = nullptr;                                       // Gsum0 of the last block (the Sigma columns static_dyn / dyn1_bare)
    cd *Gr1 = nullptr;                                       // the one-bare-rung Gsum of the last block (when requested)
    cd *Dcj = nullptr, *CbD = nullptr, *Vr = nullptr, *Pr = nullptr;   // conj(legs) (D, nout); the readout scratch
    bool legs_ok = false, cbd_ok = false, r1_ok = false;
    long D = 0, nc2 = 0;
    cublasHandle_t cb = nullptr;
    cusolverDnHandle_t cs = nullptr;
    cd *KF = nullptr, *KF2 = nullptr, *Ut = nullptr, *Vs = nullptr, *Kmat = nullptr;                // basis (complex)
    cd *Ks = nullptr, *Kds = nullptr, *Kd0 = nullptr;                                               // (s, q)
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
  };

  double ue_bytes(ue_config const &c) {
    const double D = double(c.nk * c.nc * c.nc), W = D * double(c.nR_max), np = double(c.np);
    const double fam = 2.0 * np * W, cst = W;
    double b = 0.0;
    b += (3.0 + double(c.ndist)) * D * D;                                   // K_s, K_d0, M, K_d(s_r)
    b += 3.0 * fam + 10.0 * cst + D * double(c.nR_max);                     // Ffam, F2fam, Gfam, yfam + csts + Y + Gs0 / Gr1 / CbD / Vr
    b += D * double(c.nout) + double(c.nout * c.nR_max);                    // the legs, the readout block
    b += fam;                                                               // yfam
    b += 2.0 * double(c.nt) * W + double(c.nt) * W + double(c.n_kept) * W + np * W;   // Fs, Ys, rec, g, coef
    b += double(c.nk * c.nc * c.nc * c.nc * c.nc);                          // Cb_k
    b *= 16.0;
    b += l0_plan::fixed_bytes(c.np, c.nk, c.nc, c.ng) + l0_plan::per_k_bytes(c.np, c.nc, c.ng, c.nR_max);   // L0 at K = 1
    return b * 1.05 + 512.0e6;                                              // cuSOLVER / cuBLAS workspaces
  }

  unit_engine *ue_create(ue_config const &c, double free_bytes, char *why, long why_len) {
    const double need = ue_bytes(c);
    if (need > 0.9 * free_bytes) {
      std::snprintf(why, size_t(why_len), "needs %.1f GB of device memory, %.1f GB free", need / 1e9, free_bytes / 1e9);
      return nullptr;
    }
    auto *e = new unit_engine;
    e->c = c;
    e->nc2 = c.nc * c.nc;
    e->D = c.nk * e->nc2;
    const size_t D = size_t(e->D), W = D * size_t(c.nR_max), np = size_t(c.np), nt = size_t(c.nt);
    cub_check(cublasCreate(&e->cb), "ue cublasCreate");
    if (cusolverDnCreate(&e->cs) != CUSOLVER_STATUS_SUCCESS) APP_ABORT(std::string(" ue_create: cusolverDnCreate failed."));
    e->KF = dalloc<cd>(nt * np, "ue KF"); e->KF2 = dalloc<cd>(nt * np, "ue KF2");
    e->Ut = dalloc<cd>(size_t(c.n_kept) * nt, "ue Ut"); e->Vs = dalloc<cd>(size_t(c.np_fit) * size_t(c.n_kept), "ue Vs");
    e->Kmat = dalloc<cd>(nt * size_t(c.np_fit), "ue Kc");
    e->Ks = dalloc<cd>(D * D, "ue Ks"); e->Kd0 = dalloc<cd>(D * D, "ue Kd0");
    e->Kds = dalloc<cd>(size_t(c.ndist) * D * D, "ue Kds");
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
    e->l0 = new l0_plan(c.np, c.np_fit, c.nk, c.nc, c.ng, c.nR_max, double(fr));
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
                    (void *)e->Pr})
      if (p) (void)cudaFree(p);
    for (void *p : e->sd.allocs)
      if (p) (void)cudaFree(p);
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

  void ue_set_rung(unit_engine *e, cplx const *Ksh, cplx const *Kdsh, cplx const *Kd0h, long const *treph, cplx scale_k) {
    const size_t D = size_t(e->D);
    h2d_c(e->Ks, Ksh, D * D, "ue Ks"); h2d_c(e->Kd0, Kd0h, D * D, "ue Kd0");
    h2d_c(e->Kds, Kdsh, size_t(e->c.ndist) * D * D, "ue Kds");
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
    // the active component list of an L0 input (the host kernels' P-3a scan): flags over (cst, fam) on the device, the
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
    // L0 of (fam, cst) into (F, Fs), then the constant part's frequency sum through Cb (the production Cb_cst route)
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
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, int(nR), int(D), int(D), &one, e->Y, int(D), e->Ks, int(D), &zero,
                            e->csb, int(nR)), "ue K_s z");
      ue_cb_times(e, nR, e->csb, e->cbb, false);
      cu_check(cudaDeviceSynchronize(), "ue ts");
      *tts += wnow() - t0;
    }
    // y = K_d(Gamma, Gsum): the families to tau (DLR), the dense rung per tau node, the refit, the constant part. Returns the
    // worst refit error (the host's max over families of max|Ys - rec| / max|Ys|).
    double ue_kd(unit_engine *e, long nR, cd const *Gfam, cd const *Gs, cd *yfam, cd *ycst, double *tim) {
      const long D = e->D, np = e->c.np, nt = e->c.nt, nk_ = e->c.n_kept, W = D * nR;
      const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
      const cd mscale = make_cuDoubleComplex(-e->scale.x, -e->scale.y);
      double fe = 0.0;
      cu_check(cudaMemset(yfam, 0, size_t(2 * np) * W * sizeof(cd)), "ue memset y");
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
        // the dense rung: Ys_i^T (nR x D) = scale Fs_i^T (nR x D) . K_d(rep_i)^T (D x D)
        std::vector<cd *> hA(static_cast<size_t>(nt)), hB(static_cast<size_t>(nt)), hC(static_cast<size_t>(nt));
        for (long i = 0; i < nt; ++i) {
          hA[size_t(i)] = e->Fs + size_t(i) * W;
          hB[size_t(i)] = e->Kds + size_t(e->trep[size_t(i)]) * size_t(D) * size_t(D);
          hC[size_t(i)] = e->Ys + size_t(i) * W;
        }
        h2d(e->pA, hA.data(), size_t(nt), "ue pA"); h2d(e->pB, hB.data(), size_t(nt), "ue pB"); h2d(e->pC, hC.data(), size_t(nt), "ue pC");
        cub_check(cublasZgemmBatched(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), int(D), &e->scale, (const cd **)e->pA, int(nR),
                                     (const cd **)e->pB, int(D), &zero, e->pC, int(nR), int(nt)), "ue rung");
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
        cu_check(cudaMemset(e->red, 0, 2 * sizeof(unsigned long long)), "ue red");
        maxdiff_kernel<<<grid_red(nt * W), 256>>>(nt * W, e->Ys, e->rec, e->red);
        launch_check("ue maxdiff");
        unsigned long long r[2] = {0, 0};
        cu_check(cudaMemcpy(r, e->red, sizeof(r), cudaMemcpyDeviceToHost), "ue red d2h");
        double num = 0.0, den = 0.0;
        std::memcpy(&num, &r[0], sizeof(double)); std::memcpy(&den, &r[1], sizeof(double));
        fe = std::max(fe, (den > 0.0) ? num / den : num);
        // the scatter: y.fam(fam, p < np_fit) = c(p): the same row-major layout, one copy
        cu_check(cudaMemcpy(yfam + size_t(fam) * np * W, e->coef, size_t(e->c.np_fit) * W * sizeof(cd), cudaMemcpyDeviceToDevice), "ue scatter");
        cu_check(cudaDeviceSynchronize(), "ue refit");
        tim[3] += wnow() - t0;
      }
      // the constant part: y.cst = -scale K_d(0) Gsum
      const double t0 = wnow();
      cub_check(cublasZgemm(e->cb, CUBLAS_OP_N, CUBLAS_OP_N, int(nR), int(D), int(D), &mscale, Gs, int(nR), e->Kd0, int(D), &zero,
                            ycst, int(nR)), "ue y.cst");
      cu_check(cudaDeviceSynchronize(), "ue kd0");
      tim[2] += wnow() - t0;
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
    // ---- ls_apply(D, y = 0): F = L0 D, c = T_s Fsum, Gamma = F + L0 c, Gsum = Fsum + Cb c
    ue_l0(e, nR, nullptr, e->Dblk, e->Ffam, e->Fsum, &tim[0]);
    e->r1_ok = want_r1;
    if (want_r1) {
      // the ONE BARE dynamic rung (T_s = 0, one application): y_r1 = K_d(F, Fsum) of L0 D, Gsum_r1 = Fsum[L0(y_r1; D + y_r1.cst)].
      // F / Fsum are only read; y and F2 are scratch here (both are rewritten below). Its refit error is not reported (the
      // host pass's meter was discarded too).
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
  // D-3: the Sigma deposits on the device (l0_cuda.cuh). Row-major host layouts throughout; cuBLAS sees their column-major
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
    double bits_to_double(unsigned long long u) { double d = 0.0; std::memcpy(&d, &u, sizeof(d)); return d; }
  } // namespace

  bool ue_sd_init(unit_engine *e, sd_config const &c, cplx const *Ttwh, cplx const *Uwh, cplx const *Gwh, double const *eps,
                  double const *epsG, double free_bytes, char *why, long why_len) {
    auto &s = e->sd;
    if (c.nk != e->c.nk or c.nc != e->c.nc or c.np != e->c.np or c.nt != e->c.nt or c.ng != e->c.ng) {
      std::snprintf(why, size_t(why_len), "the deposit sizes differ from the engine's");
      return false;
    }
    const long nt = c.nt, nw = c.nw_f, ns = c.ns, nk = c.nk, nc2 = e->nc2, np = c.np, ng = c.ng, Nm = c.Nm, D = e->D, R = e->c.nR_max;
    const double W = double(D) * double(R);
    const double acc = double(nt * ns * nk * nc2) * (1.0 + 2.0 * double(np));
    const double blk = W * (3.0 + 2.0 * double(nw) + double(nt)) + double(R * Nm) + double(ng * nk * nc2) +
                       double(ns * nw * nk * nc2) + 2.0 * double(nt * nw) + 2.0 * double(nw * np) + 2.0 * double(np * ng * nt);
    const double per_a = double(nk) * (2.0 * double(nc2 * nc2) + double(ng * nc2));
    const double need = 16.0 * (acc + blk + per_a) + 64.0e6;
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
    cu_check(cudaMemset(s.S, 0, nS * sizeof(cd)), "sd zero S");
    cu_check(cudaMemset(s.RT, 0, nS * size_t(np) * sizeof(cd)), "sd zero RT");
    cu_check(cudaMemset(s.RU, 0, nS * size_t(np) * sizeof(cd)), "sd zero RU");
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
    cu_check(cudaMemset(s.red, 0, 4 * sizeof(unsigned long long)), "sd red");
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
                        ") is not supported (R1).");
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
      cu_check(cudaMemset(d, 0, n * sizeof(cd)), "sd flush zero");
    };
    pull(s.S, S_cst, nS);
    if (RT != nullptr) pull(s.RT, RT, nA);
    if (RU != nullptr) pull(s.RU, RU, nA);
  }


} // namespace methods::solvers::dynbse_cuda
