// CUDA port of the L0 kernel (see l0_common.hpp for what it reproduces and why).
//
// The two phases map very differently onto a GPU:
//   B (gemms): per k there are 5 skinny products per G pole, all sharing the SAME Vt and shapes.
//              They become three cublasZgemmStridedBatched calls over the (k, pole) batch --
//              the CPU issues 5*ng*nk separate gemms with K = N = 8, which is where BLAS dies.
//   C (scatter): the (pole, component, x, r, y) space is ~13 G updates per application with no
//              reuse. One thread per (x, r, y) element of a (pole, component) block, atomics on
//              the node-resolved accumulators. The accumulators are the working set (5 * 2 * np *
//              nc * nR * nc complex = 52 MB per k), so k is processed in a stream-pipelined loop
//              rather than all at once.
#include "l0_common.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>

#define CU(x)   do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
                     std::printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define CUB(x)  do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { \
                     std::printf("cuBLAS %s @%d: %d\n", #x, __LINE__, int(e_)); std::exit(1); } } while (0)

using namespace l0mini;
using cd = cuDoubleComplex;

__device__ __forceinline__ cd operator+(cd a, cd b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ cd operator*(cd a, cd b) { return cuCmul(a, b); }
__device__ __forceinline__ cd scal(double s, cd a) { return make_cuDoubleComplex(s * a.x, s * a.y); }
__device__ __forceinline__ void atomic_add(cd *p, cd v) {
  atomicAdd(reinterpret_cast<double *>(p), v.x);
  atomicAdd(reinterpret_cast<double *>(p) + 1, v.y);
}

struct kdims { long nc, nR, np, ng, ncomp, blk, nk; };

// ---- phase C: one thread per (x, r, y) of one (pole, component) block ------------------------
// grid.x = pole, grid.y = component, block covers blk = nc*nR*nc elements.
// which_pass = 0: the j loop (mulU on Q, mulT on B); 1: the l loop (mulU on -R, mulT on inu R).
__global__ void scatter_kernel(kdims d, int which_pass, cd inu,
                               cd const *__restrict__ Qblk, cd const *__restrict__ Bblk,
                               double const *__restrict__ eps, double const *__restrict__ epsG,
                               long const *__restrict__ gnode,
                               cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2,
                               cd *__restrict__ A1, cd *__restrict__ A3) {
  const long pole = blockIdx.x, c = blockIdx.y;
  const long np = d.np, blk = d.blk, ncomp = d.ncomp;
  const long nj = gnode[pole];
  const double ej = epsG[pole];
  const int part = (c == 0) ? 1 : 0;
  const long abase = (long(part) * np) * blk;
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    // the (pole, component) block is contiguous in the gemm output as (x, c, r, y)
    const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
    const long src = (x * ncomp + c) * d.nR * d.nc + rem;
    cd q = Qblk[pole * (d.nc * ncomp * d.nR * d.nc) + src];
    cd bb = Bblk[pole * (d.nc * ncomp * d.nR * d.nc) + src];
    cd vU = q, vT = bb;
    if (which_pass == 1) { vU = make_cuDoubleComplex(-q.x, -q.y); vT = inu * q; }
    // ---- mulU(pole, c, vU) ----
    if (c == 0) {
      atomic_add(&AU[abase + nj * blk + e], vU);
    } else if (c <= np) {
      const long a = c - 1;
      if (a == nj) atomic_add(&M2[abase + nj * blk + e], vU);
      else {
        const double w = 1.0 / (ej - eps[a]);
        atomic_add(&AU[abase + nj * blk + e], scal(w, vU));
        atomic_add(&AU[abase + a * blk + e], scal(-w, vU));
      }
    } else {
      const long a = c - 1 - np;
      if (a == nj) atomic_add(&A1[abase + nj * blk + e], vU);
      else {
        const double ea = eps[a];
        const cd w = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
        const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
        const cd wd = cuCdiv(w, dd);
        const cd wt = make_cuDoubleComplex(w.x - (inu * wd).x, w.y - (inu * wd).y);
        atomic_add(&AT[abase + a * blk + e], wt * vU);
        atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vU).x, -(wd * vU).y));
        atomic_add(&AU[abase + a * blk + e], wd * vU);
      }
    }
    // ---- mulT(pole, c, vT) ----
    if (c == 0) {
      atomic_add(&AT[abase + nj * blk + e], vT);
    } else if (c <= np) {
      const long a = c - 1;
      const double ea = eps[a];
      const cd w = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
      const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
      const cd wd = cuCdiv(w, dd);
      const cd wt = make_cuDoubleComplex(w.x - (inu * wd).x, w.y - (inu * wd).y);
      atomic_add(&AT[abase + nj * blk + e], wt * vT);
      atomic_add(&AU[abase + a * blk + e], wd * vT);
      atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vT).x, -(wd * vT).y));
    } else {
      const long a = c - 1 - np;
      if (a == nj) atomic_add(&A3[abase + nj * blk + e], vT);
      else {
        const double w = 1.0 / (eps[a] - ej);
        atomic_add(&AT[abase + a * blk + e], scal(w, vT));
        atomic_add(&AT[abase + nj * blk + e], scal(-w, vT));
      }
    }
  }
}


// ---- phase C, v2: one block per COMPONENT, the poles looped inside ---------------------------
// v1 launches (pole, component) blocks and atomically accumulates every term. For a fixed
// component c the a-indexed target A[c-1] is the SAME block for all ng poles, so v1 pays ng
// atomics per address there. v2 makes c the grid and walks the poles inside the block, holding
// the a-indexed contribution in registers and issuing ONE atomic at the end -- ng times fewer
// atomics on that target. The nj-indexed terms keep their atomics (nj varies with the pole).
// Requires all poles' gemm outputs resident (ng * W complex each for Q and B).
__global__ void scatter_kernel_v2(kdims d, int which_pass, cd inu,
                                  cd const *__restrict__ Qall, cd const *__restrict__ Ball,
                                  double const *__restrict__ eps, double const *__restrict__ epsG,
                                  long const *__restrict__ gnode,
                                  cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2,
                                  cd *__restrict__ A1, cd *__restrict__ A3) {
  const long c = blockIdx.x;
  const long np = d.np, ng = d.ng, blk = d.blk, ncomp = d.ncomp;
  const long W = d.nc * ncomp * d.nR * d.nc;
  const int part = (c == 0) ? 1 : 0;
  const long abase = (long(part) * np) * blk;
  const long aU = (c >= 1 && c <= np) ? c - 1 : ((c > np) ? c - 1 - np : -1);
  const bool U_is_a = (c >= 1 && c <= np);
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
    const long src = (x * ncomp + c) * d.nR * d.nc + rem;
    cd accU = make_cuDoubleComplex(0.0, 0.0), accT = accU;    // the a-indexed reductions, in registers
    for (long pole = 0; pole < ng; ++pole) {
      const long nj = gnode[pole];
      const double ej = epsG[pole];
      cd q = Qall[pole * W + src];
      cd b = (which_pass == 0) ? Ball[pole * W + src] : q;
      cd vU = q, vT = b;
      if (which_pass == 1) { vU = make_cuDoubleComplex(-q.x, -q.y); vT = inu * q; }
      if (c == 0) { atomic_add(&AU[abase + nj * blk + e], vU); atomic_add(&AT[abase + nj * blk + e], vT); continue; }
      if (U_is_a) {
        const long a = aU;
        if (a == nj) atomic_add(&M2[abase + nj * blk + e], vU);
        else {
          const double w = 1.0 / (ej - eps[a]);
          atomic_add(&AU[abase + nj * blk + e], scal(w, vU));
          accU = accU + scal(-w, vU);
        }
        const double ea = eps[a];
        const cd w2 = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
        const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
        const cd wd = cuCdiv(w2, dd);
        const cd wt = make_cuDoubleComplex(w2.x - (inu * wd).x, w2.y - (inu * wd).y);
        atomic_add(&AT[abase + nj * blk + e], wt * vT);
        accU = accU + wd * vT;
        atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vT).x, -(wd * vT).y));
      } else {
        const long a = aU;
        if (a == nj) atomic_add(&A1[abase + nj * blk + e], vU);
        else {
          const double ea = eps[a];
          const cd w = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
          const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
          const cd wd = cuCdiv(w, dd);
          const cd wt = make_cuDoubleComplex(w.x - (inu * wd).x, w.y - (inu * wd).y);
          accT = accT + wt * vU;
          atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vU).x, -(wd * vU).y));
          accU = accU + wd * vU;
        }
        if (a == nj) atomic_add(&A3[abase + nj * blk + e], vT);
        else {
          const double w = 1.0 / (eps[a] - ej);
          accT = accT + scal(w, vT);
          atomic_add(&AT[abase + nj * blk + e], scal(-w, vT));
        }
      }
    }
    if (aU >= 0) {
      if (accU.x != 0.0 || accU.y != 0.0) atomic_add(&AU[abase + aU * blk + e], accU);
      if (accT.x != 0.0 || accT.y != 0.0) atomic_add(&AT[abase + aU * blk + e], accT);
    }
  }
}

// ---- phase C, v3: v2 plus the (pole, component) weights precomputed in SHARED memory ---------
// v2 recomputes w, wd, wt -- including a complex division -- inside the element loop, so every
// thread redoes them for each of the blk/blockDim elements it owns. They depend only on
// (pole, component), so one warp can build the table for all ng poles at block entry and the
// element loop then only multiplies. ng * 4 complex = 5 KB at ng = 80, well inside a block's
// shared memory even at nc = 16.
extern __shared__ cd smem[];
__global__ void scatter_kernel_v3(kdims d, int which_pass, cd inu,
                                  cd const *__restrict__ Qall, cd const *__restrict__ Ball,
                                  double const *__restrict__ eps, double const *__restrict__ epsG,
                                  long const *__restrict__ gnode,
                                  cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2,
                                  cd *__restrict__ A1, cd *__restrict__ A3) {
  const long c = blockIdx.x;
  const long np = d.np, ng = d.ng, blk = d.blk, ncomp = d.ncomp;
  const long W = d.nc * ncomp * d.nR * d.nc;
  const int part = (c == 0) ? 1 : 0;
  const long abase = (long(part) * np) * blk;
  const long aU = (c >= 1 && c <= np) ? c - 1 : ((c > np) ? c - 1 - np : -1);
  const bool U_is_a = (c >= 1 && c <= np);
  // shared tables: [0] w_a (the U . U weight), [1] wt, [2] wd, [3] w_T (the T . T weight)
  cd *sw = smem;
  for (long pole = threadIdx.x; pole < ng; pole += blockDim.x) {
    const double ej = epsG[pole];
    if (aU >= 0) {
      const double ea = eps[aU];
      const cd w2 = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
      const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
      const cd wd = cuCdiv(w2, dd);
      sw[pole * 4 + 0] = make_cuDoubleComplex(1.0 / (ej - ea), 0.0);
      sw[pole * 4 + 1] = make_cuDoubleComplex(w2.x - (inu * wd).x, w2.y - (inu * wd).y);
      sw[pole * 4 + 2] = wd;
      sw[pole * 4 + 3] = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
    }
  }
  __syncthreads();
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
    const long src = (x * ncomp + c) * d.nR * d.nc + rem;
    cd accU = make_cuDoubleComplex(0.0, 0.0), accT = accU;
    for (long pole = 0; pole < ng; ++pole) {
      const long nj = gnode[pole];
      cd q = Qall[pole * W + src];
      cd b = (which_pass == 0) ? Ball[pole * W + src] : q;
      cd vU = q, vT = b;
      if (which_pass == 1) { vU = make_cuDoubleComplex(-q.x, -q.y); vT = inu * q; }
      if (c == 0) { atomic_add(&AU[abase + nj * blk + e], vU); atomic_add(&AT[abase + nj * blk + e], vT); continue; }
      const cd wUU = sw[pole * 4 + 0], wt = sw[pole * 4 + 1], wd = sw[pole * 4 + 2], wTT = sw[pole * 4 + 3];
      if (U_is_a) {
        if (aU == nj) atomic_add(&M2[abase + nj * blk + e], vU);
        else { atomic_add(&AU[abase + nj * blk + e], wUU * vU); accU = accU + make_cuDoubleComplex(-(wUU * vU).x, -(wUU * vU).y); }
        atomic_add(&AT[abase + nj * blk + e], wt * vT);
        accU = accU + wd * vT;
        atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vT).x, -(wd * vT).y));
      } else {
        if (aU == nj) { atomic_add(&A1[abase + nj * blk + e], vU); atomic_add(&A3[abase + nj * blk + e], vT); }
        else {
          accT = accT + wt * vU;
          atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vU).x, -(wd * vU).y));
          accU = accU + wd * vU;
          accT = accT + wTT * vT;
          atomic_add(&AT[abase + nj * blk + e], make_cuDoubleComplex(-(wTT * vT).x, -(wTT * vT).y));
        }
      }
    }
    if (aU >= 0) {
      if (accU.x != 0.0 || accU.y != 0.0) atomic_add(&AU[abase + aU * blk + e], accU);
      if (accT.x != 0.0 || accT.y != 0.0) atomic_add(&AT[abase + aU * blk + e], accT);
    }
  }
}

// ============================================================================================
// K-BATCHED PATH. v2's scatter grid is just ncomp blocks (319 x 256 = 82 k threads against an
// H100's ~233 k resident), so a single k-point underfills the device by ~3x, and the gemm batch
// is only ng deep. Processing K k-points together multiplies both: grid (ncomp, K) and a
// cuBLAS batch of ng*K. The cost is memory -- P/Q/B scale with K -- so K is chosen at run time
// from cudaMemGetInfo (see pick_kbatch). The per-k arrays get a leading K index; K = 1 is the
// previous behaviour.
// cublasZgemmStridedBatched cannot express "A varies with (k, pole), B varies with k only", so
// the batched-pointer form is used.
// ============================================================================================
__global__ void pack_kernel_kb(kdims d, long ik0, long K, cd const *__restrict__ Xfam,
                               cd const *__restrict__ Xcst, cd *__restrict__ Vt) {
  const long kb = blockIdx.y, ik = ik0 + kb;
  const long nc = d.nc, nR = d.nR, np = d.np, nk = d.nk, ncomp = d.ncomp;
  const long W = nc * ncomp * nR * nc, tot = nc * nR * nc;
  if (kb >= K) return;
  cd *V = Vt + kb * W;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < tot; e += gridDim.x * blockDim.x) {
    const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
    V[((x * ncomp + 0) * nR + r) * nc + y] = Xcst[((ik * nc + x) * nc + y) * nR + r];
    for (long a = 0; a < np; ++a) {
      const long xu = (((0 * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r;
      const long xt = (((1 * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r;
      V[((x * ncomp + 1 + a) * nR + r) * nc + y] = Xfam[xu];
      V[((x * ncomp + 1 + np + a) * nR + r) * nc + y] = Xfam[xt];
    }
  }
}

__global__ void poleT_kernel_kb(kdims d, long ik0, long K, cd const *__restrict__ src, bool transpose,
                                long nk, cd *__restrict__ dst, bool per_k_major) {
  const long kb = blockIdx.y, ik = ik0 + kb;
  const long nc = d.nc, ng = d.ng;
  if (kb >= K) return;
  cd *D = dst + kb * ng * nc * nc;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < ng * nc * nc; e += gridDim.x * blockDim.x) {
    const long j = e / (nc * nc), x = (e / nc) % nc, y = e % nc;
    const long s = per_k_major ? ((ik * ng + j) * nc + (transpose ? y : x)) * nc + (transpose ? x : y)
                               : (((j * nk + ik) * nc + (transpose ? y : x)) * nc + (transpose ? x : y));
    D[(j * nc + x) * nc + y] = src[s];
  }
}

__global__ void scatter_kernel_kb(kdims d, long K, int which_pass, cd inu,
                                  cd const *__restrict__ Qall, cd const *__restrict__ Ball,
                                  double const *__restrict__ eps, double const *__restrict__ epsG,
                                  long const *__restrict__ gnode,
                                  cd *__restrict__ AU, cd *__restrict__ AT, cd *__restrict__ M2,
                                  cd *__restrict__ A1, cd *__restrict__ A3) {
  const long c = blockIdx.x, kb = blockIdx.y;
  if (kb >= K) return;
  const long np = d.np, ng = d.ng, blk = d.blk, ncomp = d.ncomp;
  const long W = d.nc * ncomp * d.nR * d.nc, asz = 2 * np * blk;
  const int part = (c == 0) ? 1 : 0;
  const long abase = kb * asz + (long(part) * np) * blk;
  const long aU = (c >= 1 && c <= np) ? c - 1 : ((c > np) ? c - 1 - np : -1);
  const bool U_is_a = (c >= 1 && c <= np);
  cd const *Qk = Qall + kb * ng * W;
  cd const *Bk = Ball + kb * ng * W;
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    const long x = e / (d.nR * d.nc), rem = e % (d.nR * d.nc);
    const long src = (x * ncomp + c) * d.nR * d.nc + rem;
    cd accU = make_cuDoubleComplex(0.0, 0.0), accT = accU;
    for (long pole = 0; pole < ng; ++pole) {
      const long nj = gnode[pole];
      const double ej = epsG[pole];
      cd q = Qk[pole * W + src];
      cd b = (which_pass == 0) ? Bk[pole * W + src] : q;
      cd vU = q, vT = b;
      if (which_pass == 1) { vU = make_cuDoubleComplex(-q.x, -q.y); vT = inu * q; }
      if (c == 0) { atomic_add(&AU[abase + nj * blk + e], vU); atomic_add(&AT[abase + nj * blk + e], vT); continue; }
      if (U_is_a) {
        const long a = aU;
        if (a == nj) atomic_add(&M2[abase + nj * blk + e], vU);
        else {
          const double w = 1.0 / (ej - eps[a]);
          atomic_add(&AU[abase + nj * blk + e], scal(w, vU));
          accU = accU + scal(-w, vU);
        }
        const double ea = eps[a];
        const cd w2 = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
        const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
        const cd wd = cuCdiv(w2, dd);
        const cd wt = make_cuDoubleComplex(w2.x - (inu * wd).x, w2.y - (inu * wd).y);
        atomic_add(&AT[abase + nj * blk + e], wt * vT);
        accU = accU + wd * vT;
        atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vT).x, -(wd * vT).y));
      } else {
        const long a = aU;
        if (a == nj) { atomic_add(&A1[abase + nj * blk + e], vU); atomic_add(&A3[abase + nj * blk + e], vT); }
        else {
          const double ea = eps[a];
          const cd w = make_cuDoubleComplex(1.0 / (ea - ej), 0.0);
          const cd dd = make_cuDoubleComplex(ej - ea + inu.x, inu.y);
          const cd wd = cuCdiv(w, dd);
          const cd wt = make_cuDoubleComplex(w.x - (inu * wd).x, w.y - (inu * wd).y);
          accT = accT + wt * vU;
          atomic_add(&AU[abase + nj * blk + e], make_cuDoubleComplex(-(wd * vU).x, -(wd * vU).y));
          accU = accU + wd * vU;
          const double w3 = 1.0 / (eps[a] - ej);
          accT = accT + scal(w3, vT);
          atomic_add(&AT[abase + nj * blk + e], scal(-w3, vT));
        }
      }
    }
    if (aU >= 0) {
      if (accU.x != 0.0 || accU.y != 0.0) atomic_add(&AU[abase + aU * blk + e], accU);
      if (accT.x != 0.0 || accT.y != 0.0) atomic_add(&AT[abase + aU * blk + e], accT);
    }
  }
}

__global__ void assemble_kernel_kb(kdims d, long ik0, long K, double const *__restrict__ fhalf,
                                   double const *__restrict__ fd1, cd const *__restrict__ Dsq,
                                   cd const *__restrict__ AU, cd const *__restrict__ AT, cd const *__restrict__ M2,
                                   cd *__restrict__ Ffam, cd *__restrict__ Fsum) {
  const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk, asz = 2 * np * blk;
  const long n = blockIdx.x, part = blockIdx.y, kb = blockIdx.z;
  if (kb >= K) return;
  const long ik = ik0 + kb;
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
    const long ia = kb * asz + (part * np + n) * blk + e;
    const cd u = AU[ia], t = AT[ia], m = M2[ia];
    const long f0 = (((0 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
    const long f1 = (((1 * np + n) * nk + ik) * nc + x) * nc * nR + y * nR + r;
    atomic_add(&Ffam[f0], u);
    atomic_add(&Ffam[f1], t);
    if (part == 0) {
      const long fs = ((ik * nc + x) * nc + y) * nR + r;
      atomic_add(&Fsum[fs], scal(fhalf[n], u) + scal(fd1[n], m));
    }
    if (m.x != 0.0 || m.y != 0.0)
      for (long c = 0; c < np; ++c) {
        const cd dc = Dsq[n * np + c];
        if (dc.x == 0.0 && dc.y == 0.0) continue;
        const long fc = (((0 * np + c) * nk + ik) * nc + x) * nc * nR + y * nR + r;
        atomic_add(&Ffam[fc], dc * m);
      }
  }
}

// fill the cuBLAS batched-pointer arrays (index i = kb * ng + j)
__global__ void fill_ptrs(long K, long ng, long W, long nc2, cd *Vt, cd *gjT, cd *glT, cd *Gh,
                          cd *Pj, cd *Qj, cd *Bj, cd **pVt, cd **pGjT, cd **pGlT, cd **pGh,
                          cd **pPj, cd **pQj, cd **pBj) {
  const long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= K * ng) return;
  const long kb = i / ng;
  pVt[i] = Vt + kb * W;          // shared by the ng poles of one k: only a pointer array can say this
  pGjT[i] = gjT + i * nc2;
  pGlT[i] = glT + i * nc2;
  pGh[i] = Gh + i * nc2;
  pPj[i] = Pj + i * W;
  pQj[i] = Qj + i * W;
  pBj[i] = Bj + i * W;
}

__global__ void pack_kernel(kdims d, long ik, cd const *__restrict__ Xfam, cd const *__restrict__ Xcst,
                            cd *__restrict__ Vt) {
  const long nc = d.nc, nR = d.nR, np = d.np, nk = d.nk, ncomp = d.ncomp;
  const long tot = nc * nR * nc;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < tot; e += gridDim.x * blockDim.x) {
    const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
    Vt[((x * ncomp + 0) * nR + r) * nc + y] = Xcst[((ik * nc + x) * nc + y) * nR + r];
    for (long a = 0; a < np; ++a) {
      const long xu = (((0 * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r;
      const long xt = (((1 * np + a) * nk + ik) * nc + x) * nc * nR + y * nR + r;
      Vt[((x * ncomp + 1 + a) * nR + r) * nc + y] = Xfam[xu];
      Vt[((x * ncomp + 1 + np + a) * nR + r) * nc + y] = Xfam[xt];
    }
  }
}

// transposed pole matrices for the batched gemms: gjT(pole) = gk(pole, ik)^T etc.
__global__ void poleT_kernel(kdims d, long ik, cd const *__restrict__ src, bool transpose, long nk,
                             cd *__restrict__ dst, bool per_k_major) {
  const long nc = d.nc, ng = d.ng;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < ng * nc * nc; e += gridDim.x * blockDim.x) {
    const long j = e / (nc * nc), x = (e / nc) % nc, y = e % nc;
    const long s = per_k_major ? ((ik * ng + j) * nc + (transpose ? y : x)) * nc + (transpose ? x : y)
                               : (((j * nk + ik) * nc + (transpose ? y : x)) * nc + (transpose ? x : y));
    dst[(j * nc + x) * nc + y] = src[s];
  }
}

int main(int argc, char **argv) {
  dims d; d.parse(argc, argv);
  d.print("l0-miniapp gpu");
  double gf, gb; cost_model(d, gf, gb);
  std::printf("[cost model] %.1f GFLOP of skinny gemm, %.1f GB of scatter traffic per application\n", gf, gb);
  int dev = 0; cudaDeviceProp prop{};
  CU(cudaGetDevice(&dev)); CU(cudaGetDeviceProperties(&prop, dev));
  std::printf("[device] %s, %.0f GB HBM, %d SMs, ECC %d\n", prop.name,
              double(prop.totalGlobalMem) / 1e9, prop.multiProcessorCount, prop.ECCEnabled);

  inputs in; in.build(d);
  const long nc = d.nc, nR = d.nR, np = d.np, ng = d.ng, nk = d.nk, ncomp = d.ncomp();
  const long blk = nc * nR * nc, W = nc * ncomp * nR * nc;
  kdims kd{nc, nR, np, ng, ncomp, blk, nk};

  auto up = [&](std::vector<cplx> const &h) { cd *p = nullptr; CU(cudaMalloc(&p, h.size() * sizeof(cd)));
                                              CU(cudaMemcpy(p, h.data(), h.size() * sizeof(cd), cudaMemcpyHostToDevice)); return p; };
  auto upd = [&](std::vector<double> const &h) { double *p = nullptr; CU(cudaMalloc(&p, h.size() * sizeof(double)));
                                                 CU(cudaMemcpy(p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice)); return p; };
  cd *dgk = up(in.gk), *dgkq = up(in.gkq), *dGhat = up(in.Ghat), *dGtil = up(in.Gtil);
  cd *dX = up(in.Xfam), *dXc = up(in.Xcst), *dDsq = up(in.Dsq);
  double *deps = upd(in.eps), *depsG = upd(in.epsG), *dfh = upd(in.fhalf), *dfd1 = upd(in.fd1);
  long *dgn = nullptr; CU(cudaMalloc(&dgn, in.gnode.size() * sizeof(long)));
  CU(cudaMemcpy(dgn, in.gnode.data(), in.gnode.size() * sizeof(long), cudaMemcpyHostToDevice));

  cd *dVt, *dPj, *dQj, *dBj, *dgjT, *dglT, *dGh, *dAU, *dAT, *dM2, *dA1, *dA3, *dF, *dFs;
  const size_t asz = size_t(2 * np) * size_t(blk);
  CU(cudaMalloc(&dF, size_t(2 * np * nk * nc * nc * nR) * sizeof(cd)));
  CU(cudaMalloc(&dFs, size_t(nk * nc * nc * nR) * sizeof(cd)));
  // (the K-dependent allocations happen after K is chosen, below)

  {
    const double gbX = double(2 * np * nk * nc * nc * nR) * 16e-9, gbP = double(3 * ng * W) * 16e-9;
    const double gbA = double(5 * 2 * np * blk) * 16e-9;
    size_t freeb = 0, totb = 0; CU(cudaMemGetInfo(&freeb, &totb));
    std::printf("[memory] X %.2f GB + F %.2f GB + P/Q/B %.2f GB + accumulators %.2f GB + misc = %.2f GB; "
                "device has %.1f GB free of %.1f GB\n", gbX, gbX, gbP, gbA, 2 * gbX + gbP + gbA + 0.05,
                double(freeb) * 1e-9, double(totb) * 1e-9);
  }

  // ---- dynamic batch size: the largest K whose working set fits the device ------------------
  // fixed (independent of K): X, F, the pole tables, Dsq, Fsum.  per k: Vt + 3 * ng * W (P/Q/B)
  // + 5 accumulators + ng pole matrices. The ultimate implementation has to do exactly this, and
  // at kp666 / nc 16 it is what keeps the problem on the device at all.
  long Kbatch = 0;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--kbatch=", 0) == 0) Kbatch = std::atol(argv[i] + 9);
  {
    size_t freeb = 0, totb = 0; CU(cudaMemGetInfo(&freeb, &totb));
    const double per_k = double(W + 3 * ng * W + 5 * 2 * np * blk + ng * nc * nc) * 16.0;
    const double fixed = double(2 * (2 * np * nk * nc * nc * nR) + nk * nc * nc * nR + np * np
                                + 4 * ng * nk * nc * nc) * 16.0;
    const double budget = 0.85 * double(freeb) - fixed;      // leave 15 % for cuBLAS workspace etc.
    const long Kfit = std::max(1L, long(budget / per_k));
    if (Kbatch <= 0) Kbatch = std::min(nk, Kfit);
    std::printf("[kbatch] per-k working set %.2f GB, fixed %.2f GB, free %.1f GB -> K fits %ld, using K = %ld\n",
                per_k * 1e-9, fixed * 1e-9, double(freeb) * 1e-9, Kfit, Kbatch);
    if (Kbatch > Kfit) std::printf("[kbatch] WARNING: requested K exceeds what fits; expect an allocation failure\n");
  }
  cublasHandle_t h; CUB(cublasCreate(&h));
  const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
  const cd inu = make_cuDoubleComplex(in.inu.real(), in.inu.imag());

  int variant = 1;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--scatter=", 0) == 0) variant = std::atoi(argv[i] + 10);
  cudaEvent_t e0, e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
  double t_pack = 0, t_gemm = 0, t_scat = 0, t_asm = 0, t_zero = 0;
  auto tic = [&]() { CU(cudaEventRecord(e0)); };
  auto toc = [&](double &acc) { CU(cudaEventRecord(e1)); CU(cudaEventSynchronize(e1));
                                float ms = 0; CU(cudaEventElapsedTime(&ms, e0, e1)); acc += ms * 1e-3; };

  // ---- K-dependent allocations ------------------------------------------------------------
  CU(cudaMalloc(&dVt, size_t(Kbatch) * W * sizeof(cd)));
  CU(cudaMalloc(&dPj, size_t(Kbatch) * ng * W * sizeof(cd)));
  CU(cudaMalloc(&dQj, size_t(Kbatch) * ng * W * sizeof(cd)));
  CU(cudaMalloc(&dBj, size_t(Kbatch) * ng * W * sizeof(cd)));
  CU(cudaMalloc(&dgjT, size_t(Kbatch) * ng * nc * nc * sizeof(cd)));
  CU(cudaMalloc(&dglT, size_t(Kbatch) * ng * nc * nc * sizeof(cd)));
  CU(cudaMalloc(&dGh, size_t(Kbatch) * ng * nc * nc * sizeof(cd)));
  for (cd **p : {&dAU, &dAT, &dM2, &dA1, &dA3}) CU(cudaMalloc(p, size_t(Kbatch) * asz * sizeof(cd)));
  cd **pVt, **pGjT, **pGlT, **pGh, **pPj, **pQj, **pBj;
  const size_t nptr = size_t(Kbatch) * size_t(ng);
  for (cd ***p : {&pVt, &pGjT, &pGlT, &pGh, &pPj, &pQj, &pBj}) CU(cudaMalloc(p, nptr * sizeof(cd *)));
  fill_ptrs<<<unsigned((nptr + 255) / 256), 256>>>(Kbatch, ng, W, nc * nc, dVt, dgjT, dglT, dGh,
                                                   dPj, dQj, dBj, pVt, pGjT, pGlT, pGh, pPj, pQj, pBj);
  CU(cudaDeviceSynchronize());

  auto run_once = [&]() {
    t_pack = t_gemm = t_scat = t_asm = t_zero = 0.0;
    CU(cudaMemset(dF, 0, size_t(2 * np * nk * nc * nc * nR) * sizeof(cd)));
    CU(cudaMemset(dFs, 0, size_t(nk * nc * nc * nR) * sizeof(cd)));
    const int Ncols = int(ncomp * nR * nc), Mrows = int(nc * ncomp * nR);
    for (long ik0 = 0; ik0 < nk; ik0 += Kbatch) {
      const long K = std::min(Kbatch, nk - ik0);
      const int nb = int(K * ng);
      tic();
      for (cd *p : {dAU, dAT, dM2, dA1, dA3}) CU(cudaMemset(p, 0, size_t(K) * asz * sizeof(cd)));
      toc(t_zero);
      tic();
      pack_kernel_kb<<<dim3(64u, unsigned(K), 1u), 256>>>(kd, ik0, K, dX, dXc, dVt);
      poleT_kernel_kb<<<dim3(32u, unsigned(K), 1u), 256>>>(kd, ik0, K, dgk, true, nk, dgjT, false);
      poleT_kernel_kb<<<dim3(32u, unsigned(K), 1u), 256>>>(kd, ik0, K, dgkq, true, nk, dglT, false);
      poleT_kernel_kb<<<dim3(32u, unsigned(K), 1u), 256>>>(kd, ik0, K, dGhat, false, nk, dGh, true);
      toc(t_pack);
      tic();
      CUB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                             (const cd **)pVt, Ncols, (const cd **)pGjT, int(nc), &zero, pPj, Ncols, nb));
      CUB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                             (const cd **)pGh, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb));
      CUB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                             (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pBj, int(nc), nb));
      toc(t_gemm);
      tic();
      scatter_kernel_kb<<<dim3(unsigned(ncomp), unsigned(K), 1u), 256>>>(kd, K, 0, inu, dQj, dBj, deps, depsG,
                                                                        dgn, dAU, dAT, dM2, dA1, dA3);
      toc(t_scat);
      tic();
      poleT_kernel_kb<<<dim3(32u, unsigned(K), 1u), 256>>>(kd, ik0, K, dGtil, false, nk, dGh, true);
      toc(t_pack);
      tic();
      CUB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc), &one,
                             (const cd **)pVt, Ncols, (const cd **)pGh, int(nc), &zero, pPj, Ncols, nb));
      CUB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc), &one,
                             (const cd **)pGlT, int(nc), (const cd **)pPj, int(nc), &zero, pQj, int(nc), nb));
      toc(t_gemm);
      tic();
      scatter_kernel_kb<<<dim3(unsigned(ncomp), unsigned(K), 1u), 256>>>(kd, K, 1, inu, dQj, dQj, deps, depsG,
                                                                        dgn, dAU, dAT, dM2, dA1, dA3);
      toc(t_scat);
      tic();
      assemble_kernel_kb<<<dim3(unsigned(np), 2u, unsigned(K)), 256>>>(kd, ik0, K, dfh, dfd1, dDsq,
                                                                       dAU, dAT, dM2, dF, dFs);
      toc(t_asm);
    }
    CU(cudaDeviceSynchronize());
  };

  run_once();                                     // warm-up (plans, JIT, first-touch)
  double best = 1e30;
  for (int it = 0; it < 3; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    run_once();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, s);
    std::printf("[gpu] application %d: %.3f s\n", it, s);
  }
  std::printf("[gpu] BEST %.3f s -> %.1f GFLOP/s, %.1f GB/s effective (k-batch %ld)\n", best, gf / best, gb / best, Kbatch);
  const double tt = t_zero + t_pack + t_gemm + t_scat + t_asm;
  std::printf("[gpu profile] zero %.3f s (%.0f%%), pack+poleT %.3f s (%.0f%%), cuBLAS gemms %.3f s (%.0f%%), "
              "scatter %.3f s (%.0f%%), assemble %.3f s (%.0f%%); sum %.3f s\n",
              t_zero, 100*t_zero/tt, t_pack, 100*t_pack/tt, t_gemm, 100*t_gemm/tt,
              t_scat, 100*t_scat/tt, t_asm, 100*t_asm/tt, tt);
  std::vector<cplx> Fh(size_t(2 * np * nk * nc * nc * nR));
  CU(cudaMemcpy(Fh.data(), dF, Fh.size() * sizeof(cd), cudaMemcpyDeviceToHost));
  double chk = 0.0; for (auto const &z : Fh) chk += std::norm(z);
  std::printf("[gpu] |F|^2 = %.10e\n", chk);
  return 0;
}
