// CPU kernels of the L0 miniapp: a faithful transcription of dynbse.hpp::l0_apply_shift_cols
// ("ref") and the re-blocked variant ("opt"). See l0_common.hpp for what and why.
#include "l0_common.hpp"
#include <chrono>
#include <omp.h>

namespace l0mini {

// ---- a tiny column-major-free gemm: C(M,N) = A(M,K) B(K,N), all row-major -------------------
// The production code calls nda::blas::gemm (MKL). The miniapp keeps its own so it builds
// anywhere; -DUSE_BLAS swaps in zgemm, which is what the timings below use on rusty.
#ifdef USE_BLAS
extern "C" void zgemm_(const char *, const char *, const int *, const int *, const int *, const void *,
                       const void *, const int *, const void *, const int *, const void *, void *, const int *);
inline void gemm_rm(long M, long N, long K, cplx const *A, cplx const *B, cplx *C) {
  // row-major C = A B  <=>  column-major C^T = B^T A^T
  const int m = int(N), n = int(M), k = int(K);
  const cplx one(1.0, 0.0), zero(0.0, 0.0);
  zgemm_("N", "N", &m, &n, &k, &one, B, &m, A, &k, &zero, C, &m);
}
#else
inline void gemm_rm(long M, long N, long K, cplx const *A, cplx const *B, cplx *C) {
  for (long i = 0; i < M; ++i) {
    for (long j = 0; j < N; ++j) C[i * N + j] = cplx(0.0);
    for (long p = 0; p < K; ++p) {
      const cplx a = A[i * K + p];
      if (a == cplx(0.0)) continue;
      cplx const *bp = B + p * N;
      cplx *cp = C + i * N;
      for (long j = 0; j < N; ++j) cp[j] += a * bp[j];
    }
  }
}
#endif

// ============================================================================================
// REFERENCE: one L0 application, the production loop structure, one k per thread.
// ============================================================================================
void l0_ref(inputs const &in, outputs &out, int nthreads) {
  dims const d = in.d;
  const long nc = d.nc, nR = d.nR, np = d.np, ng = d.ng, nk = d.nk, ncomp = d.ncomp();
  const long blk = nc * nR * nc;             // one (x, r, y) block
  const size_t blkz = size_t(blk);
  const cplx inu = in.inu;
  std::fill(out.Ffam.begin(), out.Ffam.end(), cplx(0.0));
  std::fill(out.Fsum.begin(), out.Fsum.end(), cplx(0.0));

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (long ik = 0; ik < nk; ++ik) {
    std::vector<cplx> Vt(size_t(nc * ncomp * nR * nc));         // (x, c, r, y)
    std::vector<cplx> Pj(size_t(nc * ncomp * nR * nc));
    std::vector<cplx> Qj(size_t(nc * ncomp * nR * nc));
    std::vector<cplx> Bj(size_t(nc * ncomp * nR * nc));
    std::vector<cplx> gjT(size_t(nc * nc)), glT(size_t(nc * nc)), Gh(size_t(nc * nc));
    std::vector<cplx> Vq(blkz), Vb(blkz), mV(blkz), tV(blkz);   // per-(pole, component) blocks
    // the five node-resolved accumulators, (part, node, x, r, y)
    const size_t asz = size_t(2 * np) * size_t(blk);
    std::vector<cplx> AU(asz, cplx(0.0)), AT(asz, cplx(0.0)), M2(asz, cplx(0.0)),
                      A1(asz, cplx(0.0)), A3(asz, cplx(0.0));

    // ---- pack the components: 0 = constant, 1..np = U_a, np+1..2np = T_a --------------------
    for (long x = 0; x < nc; ++x)
      for (long r = 0; r < nR; ++r)
        for (long y = 0; y < nc; ++y) {
          Vt[size_t(((x * ncomp + 0) * nR + r) * nc + y)] = in.Xcst[size_t(((ik * nc + x) * nc + y) * nR + r)];
          for (long a = 0; a < np; ++a) {
            const size_t xu = size_t((((0 * np + a) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
            const size_t xt = size_t((((1 * np + a) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
            Vt[size_t(((x * ncomp + 1 + a) * nR + r) * nc + y)] = in.Xfam[xu];
            Vt[size_t(((x * ncomp + 1 + np + a) * nR + r) * nc + y)] = in.Xfam[xt];
          }
        }

    auto acc = [&](std::vector<cplx> &A, int part, long node, long x, long r, long y, cplx v) {
      A[size_t((part * np + node) * blk + (x * nR + r) * nc + y)] += v;
    };
    // U_j . (component c) applied to the block V, and T_l . V -- the two scatter lambdas
    auto mulU = [&](long j, long c, cplx const *V, int part) {
      const long nj = in.gnode[size_t(j)];
      const double ej = in.epsG[size_t(j)];
      for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
        const cplx v = V[(x * nR + r) * nc + y];
        if (c == 0) { acc(AU, part, nj, x, r, y, v); continue; }
        if (c <= np) {
          const long a = c - 1;
          if (a == nj) { acc(M2, part, nj, x, r, y, v); continue; }
          const cplx w = cplx(1.0 / (ej - in.eps[size_t(a)]));
          acc(AU, part, nj, x, r, y, w * v);
          acc(AU, part, a, x, r, y, -w * v);
        } else {
          const long a = c - 1 - np;
          if (a == nj) { acc(A1, part, nj, x, r, y, v); continue; }
          const double ea = in.eps[size_t(a)];
          const cplx w = cplx(1.0 / (ea - ej));
          const cplx dd = cplx(ej - ea) + inu;
          const cplx wd = w / dd, wt = w - inu * wd;
          acc(AT, part, a, x, r, y, wt * v);
          acc(AU, part, nj, x, r, y, -wd * v);
          acc(AU, part, a, x, r, y, wd * v);
        }
      }
    };
    auto mulT = [&](long l, long c, cplx const *V, int part) {
      const long nl = in.gnode[size_t(l)];
      const double el = in.epsG[size_t(l)];
      for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
        const cplx v = V[(x * nR + r) * nc + y];
        if (c == 0) { acc(AT, part, nl, x, r, y, v); continue; }
        if (c <= np) {
          const long a = c - 1;
          const double ea = in.eps[size_t(a)];
          const cplx w = cplx(1.0 / (ea - el));
          const cplx dd = cplx(el - ea) + inu;
          const cplx wd = w / dd, wt = w - inu * wd;
          acc(AT, part, nl, x, r, y, wt * v);
          acc(AU, part, a, x, r, y, wd * v);
          acc(AU, part, nl, x, r, y, -wd * v);
        } else {
          const long a = c - 1 - np;
          if (a == nl) { acc(A3, part, nl, x, r, y, v); continue; }
          const cplx w = cplx(1.0 / (in.eps[size_t(a)] - el));
          acc(AT, part, a, x, r, y, w * v);
          acc(AT, part, nl, x, r, y, -w * v);
        }
      }
    };

    // ---- phase B: the skinny gemms, phase C: the scatter ------------------------------------
    for (long j = 0; j < ng; ++j) {
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y) {
          gjT[size_t(x * nc + y)] = in.gk[size_t(((j * nk + ik) * nc + y) * nc + x)];
          glT[size_t(x * nc + y)] = in.gkq[size_t(((j * nk + ik) * nc + y) * nc + x)];
          Gh[size_t(x * nc + y)] = in.Ghat[size_t(((ik * ng + j) * nc + x) * nc + y)];
        }
      gemm_rm(nc, ncomp * nR * nc, nc, gjT.data(), Vt.data(), Pj.data());          // Pj = gj^T Vt
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), Gh.data(), Qj.data());           // Qj = Pj Ghat
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Bj.data());          // Bj = Pj gkq^T
      for (long c = 0; c < ncomp; ++c) {
        const int part = (c == 0) ? 1 : 0;
        for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
          Vq[size_t((x * nR + r) * nc + y)] = Qj[size_t(((x * ncomp + c) * nR + r) * nc + y)];
          Vb[size_t((x * nR + r) * nc + y)] = Bj[size_t(((x * ncomp + c) * nR + r) * nc + y)];
        }
        mulU(j, c, Vq.data(), part);
        mulT(j, c, Vb.data(), part);
      }
    }
    for (long l = 0; l < ng; ++l) {
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y) {
          glT[size_t(x * nc + y)] = in.gkq[size_t(((l * nk + ik) * nc + y) * nc + x)];
          Gh[size_t(x * nc + y)] = in.Gtil[size_t(((ik * ng + l) * nc + x) * nc + y)];
        }
      gemm_rm(nc, ncomp * nR * nc, nc, Gh.data(), Vt.data(), Pj.data());           // Pl = Gtil Vt
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Qj.data());          // Rl = Pl gkq^T
      for (long c = 0; c < ncomp; ++c) {
        const int part = (c == 0) ? 1 : 0;
        for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
          const cplx v = Qj[size_t(((x * ncomp + c) * nR + r) * nc + y)];
          mV[size_t((x * nR + r) * nc + y)] = -v;
          tV[size_t((x * nR + r) * nc + y)] = inu * v;
        }
        mulU(l, c, mV.data(), part);
        mulT(l, c, tV.data(), part);
      }
    }

    // ---- phase D: assemble -----------------------------------------------------------------
    for (int part = 0; part < 2; ++part)
      for (long n = 0; n < np; ++n) {
        const cplx wh(in.fhalf[size_t(n)]), w1(in.fd1[size_t(n)]);
        for (long x = 0; x < nc; ++x)
          for (long r = 0; r < nR; ++r)
            for (long y = 0; y < nc; ++y) {
              const size_t ia = size_t((part * np + n) * blk + (x * nR + r) * nc + y);
              const size_t f0 = size_t((((0 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              const size_t f1 = size_t((((1 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              const size_t fs = size_t(((ik * nc + x) * nc + y) * nR + r);
              const cplx u = AU[ia], t = AT[ia], m = M2[ia];
              out.Ffam[f0] += u;
              out.Ffam[f1] += t;
              if (part == 0) {
                const cplx s = wh * u + w1 * m;
                out.Fsum[fs] += s;
              }
              // the confluent pieces go through the sparse D^2 / D^3 tables
              if (m != cplx(0.0))
                for (long c = 0; c < np; ++c) {
                  const cplx dc = in.Dsq[size_t(n * np + c)];
                  if (dc == cplx(0.0)) continue;
                  const size_t fc = size_t((((0 * np + c) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
                  const cplx v = dc * m;
                  out.Ffam[fc] += v;
                }
            }
      }
  }
}

} // namespace l0mini

int main(int argc, char **argv) {
  using namespace l0mini;
  dims d; d.parse(argc, argv);
  int nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--threads=", 0) == 0) nthreads = std::atoi(argv[i] + 10);
  d.print("l0-miniapp cpu");
  double gf, gb; cost_model(d, gf, gb);
  std::printf("[cost model] %.1f GFLOP of skinny gemm, %.1f GB of scatter traffic per application\n", gf, gb);

  inputs in; in.build(d);
  outputs ref; ref.alloc(d);
  const int reps = 3;
  double best = 1e30;
  for (int it = 0; it < reps; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    l0_ref(in, ref, nthreads);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, s);
    std::printf("[ref] threads %d, application %d: %.3f s\n", nthreads, it, s);
  }
  std::printf("[ref] BEST %.3f s -> %.1f GFLOP/s, %.1f GB/s effective\n", best, gf / best, gb / best);
  double chk = 0.0;
  for (auto const &z : ref.Ffam) chk += std::norm(z);
  std::printf("[ref] |F|^2 = %.10e\n", chk);
  return 0;
}
