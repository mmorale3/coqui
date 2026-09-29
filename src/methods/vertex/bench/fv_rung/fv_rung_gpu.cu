// ============================================================================================
// fv_rung_gpu -- the GPU side of the factorize_vertex rung miniapp (see fv_rung.cpp for the math and variants).
// Rungs / tables are built on the host (fv_common.hpp, the same code as the CPU miniapp) and uploaded; only the APPLICATION
// is timed on the device (cuBLAS, row-major data used through the column-major transpose identity C^T = B^T A^T).
//   dense   : nt zgemm (D x D)(D x nR), K_d(tau_r) resident.
//   dense2  : the mirror pair of each representative as one (D x D)(D x 2nR) gemm (inputs pre-gathered).
//   freqR   : ONE zgemm (D x R D)(R D x nt nR) with the c_r(tau)-scaled stacked input (a scaling kernel builds it).
//   spatial : per tau rep: Lam fold (pointer-batched Ns x Ns x nc2 gemms), stage 1 (pointer-batched nc x nc x nc nR gemms
//             over (k, t)), stage 2 (strided-batched over j1 with K = nk Ns nc), per k'.
// ============================================================================================
#include "fv_common.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::printf("cuBLAS %d at %s:%d\n", int(s_), __FILE__, __LINE__); std::exit(1); } } while (0)

using dc = cuDoubleComplex;
static const dc d1 = {1.0, 0.0}, d0 = {0.0, 0.0};
template <class T> static T *dalloc(size_t n) { T *p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
static dc *up(std::vector<cplx> const &v) { dc *p = dalloc<dc>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(dc), cudaMemcpyHostToDevice)); return p; }

// row-major C(MxN) = A(MxK) B(KxN) (all no-trans, leading dims = row strides) via column-major C^T = B^T A^T
static void rgemm(cublasHandle_t h, long M, long N, long K, dc alpha, const dc *A, long lda, const dc *B, long ldb, dc beta, dc *C, long ldc,
                  cublasOperation_t opA = CUBLAS_OP_N) {
  // opA = CUBLAS_OP_T means row-major A is given as its transpose (K x M, lda = row stride of that storage)
  CB(cublasZgemm(h, CUBLAS_OP_N, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

__global__ void scale_stack(const dc *F, const dc *ct, dc *Fs, long R, long D, long nt, long nR) {
  // Fs[((r D + y) nt + i) nR + N] = c_r(tau_i) F[(i D + y) nR + N]
  const long total = R * D * nt * nR;
  for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < total; e += long(gridDim.x) * blockDim.x) {
    const long N = e % nR; long t = e / nR; const long i = t % nt; t /= nt; const long y = t % D; const long r = t / D;
    Fs[e] = cuCmul(ct[i * R + r], F[(i * D + y) * nR + N]);
  }
}
__global__ void unstack(const dc *Ys, dc *Y, long D, long nt, long nR) {
  const long total = D * nt * nR;
  for (long e = blockIdx.x * long(blockDim.x) + threadIdx.x; e < total; e += long(gridDim.x) * blockDim.x) {
    const long N = e % nR; long t = e / nR; const long x = t % D; const long i = t / D;
    Y[e] = Ys[(x * nt + i) * nR + N];
  }
}

static double relerr(std::vector<cplx> const &a, std::vector<cplx> const &b) {
  double e = 0, n = 0;
  for (size_t i = 0; i < a.size(); ++i) { e += std::norm(a[i] - b[i]); n += std::norm(b[i]); }
  return std::sqrt(e / n);
}

int main(int argc, char **argv) {
  if (argc < 2) { std::printf("usage: fv_rung_gpu <bench.h5> [nR=32] [napply=3] [variants=dense,dense2,freqR,spatial]\n"); return 1; }
  const long nR = argc > 2 ? std::atol(argv[2]) : 32;
  const int napply = argc > 3 ? std::atoi(argv[3]) : 3;
  const std::string variants = argc > 4 ? argv[4] : "dense,dense2,freqR,spatial";
  auto want = [&](const char *v) { return variants.find(v) != std::string::npos; };
  data d;
  double t0 = now();
  load_data(argv[1], d);
  const long nk = d.nk, nc = d.nc, nc2 = d.nc2, D = d.D, nt = d.nt, ndist = d.ndist, R = d.R, Ns = d.Ns, nsc = Ns * nc;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("[fv_rung_gpu] %s (%.0f GB); nk %ld Nm %ld nc %ld D %ld nt %ld ndist %ld R %ld Ns %ld nR %ld (load %.1f s)\n", prop.name,
              prop.totalGlobalMem / 1e9, nk, d.Nm, nc, D, nt, ndist, R, Ns, nR, now() - t0);
  cublasHandle_t h; CB(cublasCreate(&h));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto tic = [&] { CK(cudaEventRecord(e0)); };
  auto toc = [&] { CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / 1e3; };

  std::vector<cplx> F(size_t(nt) * D * nR); random_fill(F);
  dc *Fd = up(F), *Yd = dalloc<dc>(F.size());
  std::vector<cplx> Yref(F.size()), Y(F.size());
  const double gb = 16.0 / 1e9;

  if (want("dense")) {
    std::vector<cplx> Kh(size_t(D) * D);
    const size_t need = size_t(ndist) * D * D * 16;
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    if (need > fr * 0.95) std::printf("  dense   : SKIPPED -- K_d needs %.1f GB, %.1f GB free\n", need / 1e9, fr / 1e9);
    else {
      dc *Kd = dalloc<dc>(size_t(ndist) * D * D);
      double tb = now();
      for (long r = 0; r < ndist; ++r) {
        build_kbig(d, [&](long q) { return &d.Wt[((q * ndist + r) * d.Nm) * d.Nm]; }, Kh.data(), D);
        CK(cudaMemcpy(Kd + size_t(r) * D * D, Kh.data(), Kh.size() * 16, cudaMemcpyHostToDevice));
      }
      tb = now() - tb;
      double ta = 1e30;
      for (int it = 0; it < napply; ++it) {
        tic();
        for (long i = 0; i < nt; ++i) rgemm(h, D, nR, D, d1, Kd + size_t(d.trep[i]) * D * D, D, Fd + size_t(i) * D * nR, nR, d0, Yd + size_t(i) * D * nR, nR);
        ta = std::min(ta, double(toc()));
      }
      CK(cudaMemcpy(Yref.data(), Yd, Yref.size() * 16, cudaMemcpyDeviceToHost));
      const double fl = 8.0 * nt * double(D) * D * nR;
      std::printf("  dense   : host build + upload %.1f s, apply %.4f s/app (%.0f GF/s), K %.1f GB\n", tb, ta, fl / ta / 1e9, ndist * double(D) * D * gb);
      if (want("dense2")) {
        // pre-gathered pair inputs (the gather is a trivial copy kernel in a real code; excluded here)
        std::vector<cplx> F2(size_t(ndist) * D * 2 * nR, 0.0);
        for (long r = 0; r < ndist; ++r) {
          const long i0 = d.reps[r], i1 = d.tmirror[i0];
          for (long x = 0; x < D; ++x)
            for (long b = 0; b < 2; ++b)
              if (b == 0 or i1 != i0)
                std::memcpy(&F2[((size_t(r) * D + x) * 2 + b) * nR], &F[(size_t(b ? i1 : i0) * D + x) * nR], nR * 16);
        }
        dc *F2d = up(F2), *Y2d = dalloc<dc>(F2.size());
        double t2 = 1e30;
        for (int it = 0; it < napply; ++it) {
          tic();
          for (long r = 0; r < ndist; ++r)
            rgemm(h, D, 2 * nR, D, d1, Kd + size_t(r) * D * D, D, F2d + size_t(r) * D * 2 * nR, 2 * nR, d0, Y2d + size_t(r) * D * 2 * nR, 2 * nR);
          t2 = std::min(t2, double(toc()));
        }
        std::printf("  dense2  : apply %.4f s/app (mirror pairs in one gemm)\n", t2);
        CK(cudaFree(F2d)); CK(cudaFree(Y2d));
      }
      CK(cudaFree(Kd));
    }
  }

  if (want("freqR")) {
    std::vector<cplx> Kc(size_t(D) * R * D);
    double tb = now();
    for (long r = 0; r < R; ++r) build_kbig(d, [&](long q) { return &d.A[((r * d.nq + q) * d.Nm) * d.Nm]; }, &Kc[size_t(r) * D], R * D);
    tb = now() - tb;
    dc *Kcd = up(Kc); std::vector<cplx>().swap(Kc);
    dc *ctd = up(d.ctau), *Fs = dalloc<dc>(size_t(R) * D * nt * nR), *Ys = dalloc<dc>(size_t(D) * nt * nR);
    double ta = 1e30, tg = 0;
    for (int it = 0; it < napply; ++it) {
      tic();
      scale_stack<<<4096, 256>>>(Fd, ctd, Fs, R, D, nt, nR);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      cudaEvent_t g0; CK(cudaEventCreate(&g0)); CK(cudaEventRecord(g0));
      rgemm(h, D, nt * nR, R * D, d1, Kcd, R * D, Fs, nt * nR, d0, Ys, nt * nR);
      cudaEvent_t g1; CK(cudaEventCreate(&g1)); CK(cudaEventRecord(g1)); CK(cudaEventSynchronize(g1));
      float gms; CK(cudaEventElapsedTime(&gms, g0, g1));
      unstack<<<4096, 256>>>(Ys, Yd, D, nt, nR);
      const double t = toc();
      if (t < ta) { ta = t; tg = gms / 1e3; }
    }
    CK(cudaMemcpy(Y.data(), Yd, Y.size() * 16, cudaMemcpyDeviceToHost));
    const double fl = 8.0 * R * double(D) * D * nt * nR;
    std::printf("  freqR   : R %ld, host build %.1f s, apply %.4f s/app (gemm %.4f s, %.0f GF/s), K %.1f GB, err vs dense %.2e\n", R, tb, ta, tg,
                fl / tg / 1e9, R * double(D) * D * gb, want("dense") ? relerr(Y, Yref) : -1.0);
    CK(cudaFree(Kcd)); CK(cudaFree(ctd)); CK(cudaFree(Fs)); CK(cudaFree(Ys));
  }

  if (want("spatial")) {
    std::vector<cplx> m, NT;
    double tb = now();
    build_spatial_tables(d, m, NT);
    tb = now() - tb;
    dc *md = up(m), *NTd = up(NT), *Lamd = up(d.Lam);
    dc *mtd = dalloc<dc>(m.size());
    const long cbsz = nc * nk * nsc * nR;
    dc *Cd = dalloc<dc>(size_t(cbsz));
    // pointer arrays: Lam fold (per (kp,k), per tau rep: A = Lam(q(k,kp), r)), stage 1 (per kp: (k, t) batch)
    std::vector<const dc *> hA(nk * nk), hB(nk * nk); std::vector<dc *> hC(nk * nk);
    const dc **pA = dalloc<const dc *>(nk * nk), **pB = dalloc<const dc *>(nk * nk); dc **pC = dalloc<dc *>(nk * nk);
    std::vector<const dc *> sA(size_t(nk) * nk * Ns), sB(size_t(nk) * nk * Ns); std::vector<dc *> sC(size_t(nk) * nk * Ns);
    for (long kp = 0; kp < nk; ++kp)
      for (long k = 0; k < nk; ++k)
        for (long t = 0; t < Ns; ++t) {
          const size_t b = (size_t(kp) * nk + k) * Ns + t;
          sA[b] = mtd + ((size_t(kp) * nk + k) * Ns + t) * nc2;   // mt_t (nc x nc) row-major
          sB[b] = nullptr;                                         // f(k, tau): set per tau below (offset)
          sC[b] = Cd + (size_t(k) * Ns + t) * nc * nR;             // C[j1][k][t][j3][N], ldc = nk nsc nR
        }
    const dc **psA = dalloc<const dc *>(sA.size()), **psB = dalloc<const dc *>(sB.size()); dc **psC = dalloc<dc *>(sC.size());
    CK(cudaMemcpy(psA, sA.data(), sA.size() * sizeof(void *), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(psC, sC.data(), sC.size() * sizeof(void *), cudaMemcpyHostToDevice));
    std::vector<std::vector<const dc *>> sBtau(nt, std::vector<const dc *>(sB.size()));
    for (long i = 0; i < nt; ++i)
      for (long kp = 0; kp < nk; ++kp)
        for (long k = 0; k < nk; ++k)
          for (long t = 0; t < Ns; ++t) sBtau[i][(size_t(kp) * nk + k) * Ns + t] = Fd + (size_t(i) * D + k * nc2) * nR;
    const dc **psBt = dalloc<const dc *>(size_t(nt) * sB.size());
    for (long i = 0; i < nt; ++i) CK(cudaMemcpy(psBt + size_t(i) * sB.size(), sBtau[i].data(), sB.size() * sizeof(void *), cudaMemcpyHostToDevice));
    double ta = 1e30, tfold = 0, tst = 0;
    for (int it = 0; it < napply; ++it) {
      double tf = 0, ts = 0;
      tic();
      for (long r = 0; r < ndist; ++r) {
        // Lam fold: mt[t][x] = sum_s Lam[s][t] m[s][x]  (row-major A = Lam^T given as Lam storage with OP_T)
        for (long kp = 0; kp < nk; ++kp)
          for (long k = 0; k < nk; ++k) {
            hA[kp * nk + k] = Lamd + ((size_t(d.qx_of[k * nk + kp]) * ndist + r) * Ns) * Ns;
            hB[kp * nk + k] = md + (size_t(kp) * nk + k) * Ns * nc2;
            hC[kp * nk + k] = mtd + (size_t(kp) * nk + k) * Ns * nc2;
          }
        CK(cudaMemcpy(pA, hA.data(), hA.size() * sizeof(void *), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pB, hB.data(), hB.size() * sizeof(void *), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pC, hC.data(), hC.size() * sizeof(void *), cudaMemcpyHostToDevice));
        // column-major: mt^T (nc2 x Ns) = m^T (nc2 x Ns) Lam; the col-major view of the row-major Lam storage is Lam^T -> OP_T
        CB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, nc2, Ns, Ns, &d1, (const dc **)pB, nc2, (const dc **)pA, Ns, &d0, pC, nc2, nk * nk));
        const long i0 = d.reps[r], i1 = d.tmirror[i0];
        for (long ii = 0; ii < ((i1 == i0) ? 1 : 2); ++ii) {
          const long i = ii ? i1 : i0;
          for (long kp = 0; kp < nk; ++kp) {
            // stage 1: C(k,t) (nc x nc nR, ldc nk nsc nR) = mt_t(kp,k) (nc x nc) f(k) (nc x nc nR)
            const size_t off = size_t(kp) * nk * Ns;
            CB(cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, nc * nR, nc, nc, &d1, psBt + size_t(i) * sB.size() + off, nc * nR,
                                  psA + off, nc, &d0, psC + off, nk * nsc * nR, nk * Ns));
            // stage 2: out(kp)[j1] (nc x nR) = NT(kp) (nc x nk nsc) C[j1] (nk nsc x nR), strided over j1
            CB(cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, nR, nc, nk * nsc, &d1, Cd, nR, nk * nsc * nR,
                                         NTd + size_t(kp) * nc * nk * nsc, nk * nsc, 0, &d0, Yd + (size_t(i) * D + kp * nc2) * nR, nR,
                                         nc * nR, nc));
          }
        }
      }
      const double t = toc();
      if (t < ta) { ta = t; tfold = tf; tst = ts; }
    }
    (void)tfold; (void)tst;
    CK(cudaMemcpy(Y.data(), Yd, Y.size() * 16, cudaMemcpyDeviceToHost));
    const double fl = 8.0 * nt * double(nk) * nk * (2.0 * Ns * nc2 * nc * nR) + 8.0 * ndist * double(nk) * nk * Ns * Ns * nc2;
    std::printf("  spatial : Ns %ld, host tables %.1f s (%.2f GB), apply %.4f s/app (%.0f GF/s), err vs dense %.2e\n", Ns, tb,
                (m.size() + NT.size()) * gb, ta, fl / ta / 1e9, want("dense") ? relerr(Y, Yref) : -1.0);
  }
  std::printf("[fv_rung_gpu] done %.1f s\n", now() - t0);
  return 0;
}
