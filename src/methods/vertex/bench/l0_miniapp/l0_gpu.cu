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

// ---- phase D: assemble AU/AT (+ the sparse confluent tables) into F and Fsum ------------------
__global__ void assemble_kernel(kdims d, long ik, double const *__restrict__ fhalf,
                                double const *__restrict__ fd1, cd const *__restrict__ Dsq,
                                cd const *__restrict__ AU, cd const *__restrict__ AT, cd const *__restrict__ M2,
                                cd *__restrict__ Ffam, cd *__restrict__ Fsum) {
  const long np = d.np, blk = d.blk, nc = d.nc, nR = d.nR, nk = d.nk;
  const long n = blockIdx.x, part = blockIdx.y;
  for (long e = threadIdx.x; e < blk; e += blockDim.x) {
    const long x = e / (nR * nc), r = (e / nc) % nR, y = e % nc;
    const long ia = (part * np + n) * blk + e;
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

// ---- pack Vt for one k: (x, comp, r, y) ------------------------------------------------------
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
  CU(cudaMalloc(&dVt, W * sizeof(cd)));
  CU(cudaMalloc(&dPj, ng * W * sizeof(cd)));      // one Pj per pole: the batch
  CU(cudaMalloc(&dQj, ng * W * sizeof(cd)));
  CU(cudaMalloc(&dBj, ng * W * sizeof(cd)));
  CU(cudaMalloc(&dgjT, ng * nc * nc * sizeof(cd)));
  CU(cudaMalloc(&dglT, ng * nc * nc * sizeof(cd)));
  CU(cudaMalloc(&dGh, ng * nc * nc * sizeof(cd)));
  const size_t asz = size_t(2 * np) * size_t(blk);
  for (cd **p : {&dAU, &dAT, &dM2, &dA1, &dA3}) CU(cudaMalloc(p, asz * sizeof(cd)));
  CU(cudaMalloc(&dF, size_t(2 * np * nk * nc * nc * nR) * sizeof(cd)));
  CU(cudaMalloc(&dFs, size_t(nk * nc * nc * nR) * sizeof(cd)));

  cublasHandle_t h; CUB(cublasCreate(&h));
  const cd one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
  const cd inu = make_cuDoubleComplex(in.inu.real(), in.inu.imag());

  auto run_once = [&]() {
    CU(cudaMemset(dF, 0, size_t(2 * np * nk * nc * nc * nR) * sizeof(cd)));
    CU(cudaMemset(dFs, 0, size_t(nk * nc * nc * nR) * sizeof(cd)));
    for (long ik = 0; ik < nk; ++ik) {
      for (cd *p : {dAU, dAT, dM2, dA1, dA3}) CU(cudaMemset(p, 0, asz * sizeof(cd)));
      pack_kernel<<<64, 256>>>(kd, ik, dX, dXc, dVt);
      // cuBLAS is column-major; our row-major C = A B is computed as C^T = B^T A^T.
      // Pj(nc, W/nc) = gjT(nc,nc) Vt(nc, W/nc)  -> batched over the ng poles (Vt shared, stride 0)
      poleT_kernel<<<32, 256>>>(kd, ik, dgk, true, nk, dgjT, false);
      poleT_kernel<<<32, 256>>>(kd, ik, dgkq, true, nk, dglT, false);
      poleT_kernel<<<32, 256>>>(kd, ik, dGhat, false, nk, dGh, true);
      const int Ncols = int(ncomp * nR * nc);
      CUB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc),
                                    &one, dVt, Ncols, 0, dgjT, int(nc), nc * nc,
                                    &zero, dPj, Ncols, W, int(ng)));
      // Qj((nc ncomp nR), nc) = Pj2 Ghat ; Bj = Pj2 gkq^T   (row-major M x 8 by 8 x 8)
      const int Mrows = int(nc * ncomp * nR);
      CUB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc),
                                    &one, dGh, int(nc), nc * nc, dPj, int(nc), W,
                                    &zero, dQj, int(nc), W, int(ng)));
      CUB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc),
                                    &one, dglT, int(nc), nc * nc, dPj, int(nc), W,
                                    &zero, dBj, int(nc), W, int(ng)));
      dim3 grid{unsigned(ng), unsigned(ncomp), 1u};   // braces: dim3 g(unsigned(x), ...) parses as a declaration
      scatter_kernel<<<grid, 256>>>(kd, 0, inu, dQj, dBj, deps, depsG, dgn, dAU, dAT, dM2, dA1, dA3);
      // the l pass: Pl = Gtil Vt, Rl = Pl gkq^T
      poleT_kernel<<<32, 256>>>(kd, ik, dGtil, false, nk, dGh, true);
      CUB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, int(nc), int(nc),
                                    &one, dVt, Ncols, 0, dGh, int(nc), nc * nc,
                                    &zero, dPj, Ncols, W, int(ng)));
      CUB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(nc), Mrows, int(nc),
                                    &one, dglT, int(nc), nc * nc, dPj, int(nc), W,
                                    &zero, dQj, int(nc), W, int(ng)));
      scatter_kernel<<<grid, 256>>>(kd, 1, inu, dQj, dQj, deps, depsG, dgn, dAU, dAT, dM2, dA1, dA3);
      dim3 ag{unsigned(np), 2u, 1u};
      assemble_kernel<<<ag, 256>>>(kd, ik, dfh, dfd1, dDsq, dAU, dAT, dM2, dF, dFs);
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
  std::printf("[gpu] BEST %.3f s -> %.1f GFLOP/s, %.1f GB/s effective\n", best, gf / best, gb / best);
  std::vector<cplx> Fh(size_t(2 * np * nk * nc * nc * nR));
  CU(cudaMemcpy(Fh.data(), dF, Fh.size() * sizeof(cd), cudaMemcpyDeviceToHost));
  double chk = 0.0; for (auto const &z : Fh) chk += std::norm(z);
  std::printf("[gpu] |F|^2 = %.10e\n", chk);
  return 0;
}
