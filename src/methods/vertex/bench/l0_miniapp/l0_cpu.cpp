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


// ============================================================================================
// OPTIMIZED: same arithmetic, three changes the miniapp exists to test.
//   O1  the per-(pole, component) blocks are READ IN PLACE from the gemm output (the reference
//       copies them out first: ng * ncomp * blk complex per k of pure memory traffic). The
//       gemm output is (x, c, r, y), so the block is a strided view, not contiguous -- the
//       scatter loops x outermost and walks (r, y) contiguously inside.
//   O2  the scalar weights of mulU / mulT depend only on (pole, component), not on (x, r, y):
//       hoist them out of the element loop, so the inner loop is a fused multiply-add into at
//       most three contiguous accumulator blocks.
//   O3  the three accumulator writes of a component are done in one pass over the block, so
//       each V element is read once instead of once per target.
// ============================================================================================
void l0_kernel(inputs const &in, outputs &out, int nthreads, bool materialize_l);
void l0_opt(inputs const &in, outputs &out, int nthreads) { l0_kernel(in, out, nthreads, false); }
void l0_prod(inputs const &in, outputs &out, int nthreads) { l0_kernel(in, out, nthreads, true); }
void l0_kernel(inputs const &in, outputs &out, int nthreads, bool materialize_l) {
  dims const d = in.d;
  const long nc = d.nc, nR = d.nR, np = d.np, ng = d.ng, nk = d.nk, ncomp = d.ncomp();
  const long blk = nc * nR * nc, ry = nR * nc, W = nc * ncomp * nR * nc;
  const size_t Wz = size_t(W);            // a bare cast of a single identifier parses as a declaration
  const cplx inu = in.inu;
  std::fill(out.Ffam.begin(), out.Ffam.end(), cplx(0.0));
  std::fill(out.Fsum.begin(), out.Fsum.end(), cplx(0.0));

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (long ik = 0; ik < nk; ++ik) {
    std::vector<cplx> Vt(Wz), Pj(Wz), Qj(Wz), Bj(Wz);
    std::vector<cplx> gjT(size_t(nc * nc)), glT(size_t(nc * nc)), Gh(size_t(nc * nc));
    std::vector<cplx> mVb(size_t(nc * ncomp * ry)), tVb(size_t(nc * ncomp * ry));   // production's l-pass blocks
    const size_t asz = size_t(2 * np) * size_t(blk);
    std::vector<cplx> AU(asz, cplx(0.0)), AT(asz, cplx(0.0)), M2(asz, cplx(0.0)),
                      A1(asz, cplx(0.0)), A3(asz, cplx(0.0));
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

    // O2/O3: one pass over the strided block, up to three contiguous targets, weights hoisted
    auto scatter3 = [&](cplx const *src, long part, cplx w0, cplx *A0, cplx w1, cplx *A1p, cplx w2, cplx *A2p) {
      for (long x = 0; x < nc; ++x) {
        cplx const *v = src + x * ncomp * ry;          // (x, c, ., .) -- c fixed by the caller
        const long o = x * ry;
        if (A2p) {
          for (long e = 0; e < ry; ++e) { const cplx z = v[e]; A0[o + e] += w0 * z; A1p[o + e] += w1 * z; A2p[o + e] += w2 * z; }
        } else if (A1p) {
          for (long e = 0; e < ry; ++e) { const cplx z = v[e]; A0[o + e] += w0 * z; A1p[o + e] += w1 * z; }
        } else {
          for (long e = 0; e < ry; ++e) A0[o + e] += w0 * v[e];
        }
      }
    };
    auto base = [&](std::vector<cplx> &A, long part, long node) { return A.data() + size_t((part * np + node) * blk); };

    auto do_pole = [&](long pole, cplx const *Qsrc, cplx const *Bsrc, cplx sU, cplx sT) {
      const long nj = in.gnode[size_t(pole)];
      const double ej = in.epsG[size_t(pole)];
      for (long c = 0; c < ncomp; ++c) {
        const long part = (c == 0) ? 1 : 0;
        cplx const *q = Qsrc + c * ry;
        cplx const *bb = Bsrc + c * ry;
        // ---- mulU on sU * q ----
        if (c == 0) scatter3(q, part, sU, base(AU, part, nj), cplx(0.0), nullptr, cplx(0.0), nullptr);
        else if (c <= np) {
          const long a = c - 1;
          if (a == nj) scatter3(q, part, sU, base(M2, part, nj), cplx(0.0), nullptr, cplx(0.0), nullptr);
          else {
            const cplx w = sU * cplx(1.0 / (ej - in.eps[size_t(a)]));
            scatter3(q, part, w, base(AU, part, nj), -w, base(AU, part, a), cplx(0.0), nullptr);
          }
        } else {
          const long a = c - 1 - np;
          if (a == nj) scatter3(q, part, sU, base(A1, part, nj), cplx(0.0), nullptr, cplx(0.0), nullptr);
          else {
            const double ea = in.eps[size_t(a)];
            const cplx w = cplx(1.0 / (ea - ej)), dd = cplx(ej - ea) + inu;
            const cplx wd = w / dd, wt = w - inu * wd;
            scatter3(q, part, sU * wt, base(AT, part, a), -sU * wd, base(AU, part, nj), sU * wd, base(AU, part, a));
          }
        }
        // ---- mulT on sT * b ----
        if (c == 0) scatter3(bb, part, sT, base(AT, part, nj), cplx(0.0), nullptr, cplx(0.0), nullptr);
        else if (c <= np) {
          const long a = c - 1;
          const double ea = in.eps[size_t(a)];
          const cplx w = cplx(1.0 / (ea - ej)), dd = cplx(ej - ea) + inu;
          const cplx wd = w / dd, wt = w - inu * wd;
          scatter3(bb, part, sT * wt, base(AT, part, nj), sT * wd, base(AU, part, a), -sT * wd, base(AU, part, nj));
        } else {
          const long a = c - 1 - np;
          if (a == nj) scatter3(bb, part, sT, base(A3, part, nj), cplx(0.0), nullptr, cplx(0.0), nullptr);
          else {
            const cplx w = sT * cplx(1.0 / (in.eps[size_t(a)] - ej));
            scatter3(bb, part, w, base(AT, part, a), -w, base(AT, part, nj), cplx(0.0), nullptr);
          }
        }
      }
    };

    for (long j = 0; j < ng; ++j) {
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y) {
          gjT[size_t(x * nc + y)] = in.gk[size_t(((j * nk + ik) * nc + y) * nc + x)];
          glT[size_t(x * nc + y)] = in.gkq[size_t(((j * nk + ik) * nc + y) * nc + x)];
          Gh[size_t(x * nc + y)] = in.Ghat[size_t(((ik * ng + j) * nc + x) * nc + y)];
        }
      gemm_rm(nc, ncomp * nR * nc, nc, gjT.data(), Vt.data(), Pj.data());
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), Gh.data(), Qj.data());
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Bj.data());
      do_pole(j, Qj.data(), Bj.data(), cplx(1.0), cplx(1.0));       // O1: read in place
    }
    for (long l = 0; l < ng; ++l) {
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y) {
          glT[size_t(x * nc + y)] = in.gkq[size_t(((l * nk + ik) * nc + y) * nc + x)];
          Gh[size_t(x * nc + y)] = in.Gtil[size_t(((ik * ng + l) * nc + x) * nc + y)];
        }
      gemm_rm(nc, ncomp * nR * nc, nc, Gh.data(), Vt.data(), Pj.data());
      gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Qj.data());
      if (not materialize_l) {
        do_pole(l, Qj.data(), Qj.data(), cplx(-1.0), inu);          // -U_l R_l and +i nu T_l R_l, scalars folded
      } else {
        // the production path: build mV = -R and tV = i nu R into their own blocks first
        for (long c = 0; c < ncomp; ++c)
          for (long x = 0; x < nc; ++x)
            for (long e = 0; e < ry; ++e) {
              const cplx v = Qj[size_t((x * ncomp + c) * ry + e)];
              mVb[size_t((x * ncomp + c) * ry + e)] = -v;      // same (x, c, ry) layout do_pole reads
              tVb[size_t((x * ncomp + c) * ry + e)] = inu * v;
            }
        do_pole(l, mVb.data(), tVb.data(), cplx(1.0), cplx(1.0));
      }
    }

    for (int part = 0; part < 2; ++part)
      for (long n = 0; n < np; ++n) {
        const cplx wh(in.fhalf[size_t(n)]), w1(in.fd1[size_t(n)]);
        cplx const *au = AU.data() + size_t((part * np + n) * blk);
        cplx const *at = AT.data() + size_t((part * np + n) * blk);
        cplx const *m2 = M2.data() + size_t((part * np + n) * blk);
        for (long x = 0; x < nc; ++x)
          for (long r = 0; r < nR; ++r)
            for (long y = 0; y < nc; ++y) {
              const long e = (x * nR + r) * nc + y;
              const size_t f0 = size_t((((0 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              const size_t f1 = size_t((((1 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              out.Ffam[f0] += au[e];
              out.Ffam[f1] += at[e];
              if (part == 0) out.Fsum[size_t(((ik * nc + x) * nc + y) * nR + r)] += wh * au[e] + w1 * m2[e];
              if (m2[e] != cplx(0.0))
                for (long c = 0; c < np; ++c) {
                  const cplx dc = in.Dsq[size_t(n * np + c)];
                  if (dc == cplx(0.0)) continue;
                  out.Ffam[size_t((((0 * np + c) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r)] += dc * m2[e];
                }
            }
      }
  }
}


// ============================================================================================
// TILED: the traffic restructuring.
//
// In mulU / mulT the component index c selects the node a = c - 1 (or c - 1 - np), so each
// (pole, component) pair does a read-modify-write of TWO 32 KB node blocks: A[nj], whose index
// depends only on the pole, and A[a], whose index sweeps the whole node axis as c runs. With the
// pole loop outside, A[nj] stays in L1 across the 319 components -- but A[a] is a cold RMW
// 2 * ng * ncomp times per k (~0.8 GB per k), and that is the kernel's DRAM traffic.
//
// Swap the nesting inside a TILE of PB poles: gemm the tile first, then loop components outside
// and the tile's poles inside. The a-indexed term becomes a REDUCTION over the tile accumulated
// in one 32 KB stack buffer and written once, so its traffic falls by PB; the nj-indexed terms
// stay read-modify-write but touch only the tile's PB blocks, which sit in L2. Same arithmetic,
// same order of accumulation within a pole -- only the loop nesting and the buffering change.
// ============================================================================================
void l0_tiled(inputs const &in, outputs &out, int nthreads, long PB) {
  dims const d = in.d;
  const long nc = d.nc, nR = d.nR, np = d.np, ng = d.ng, nk = d.nk, ncomp = d.ncomp();
  const long blk = nc * nR * nc, ry = nR * nc, W = nc * ncomp * nR * nc;
  const size_t Wz = size_t(W), blkz = size_t(blk);
  const cplx inu = in.inu;
  if (PB < 1) PB = 1;
  std::fill(out.Ffam.begin(), out.Ffam.end(), cplx(0.0));
  std::fill(out.Fsum.begin(), out.Fsum.end(), cplx(0.0));

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (long ik = 0; ik < nk; ++ik) {
    std::vector<cplx> Vt(Wz), Pj(Wz);
    std::vector<cplx> Qt(size_t(PB) * Wz), Bt(size_t(PB) * Wz);     // the tile's gemm outputs
    std::vector<cplx> gjT(size_t(nc * nc)), glT(size_t(nc * nc)), Gh(size_t(nc * nc));
    std::vector<cplx> accU(blkz), accT(blkz);                        // the a-indexed reductions
    const size_t asz = size_t(2 * np) * blkz;
    std::vector<cplx> AU(asz, cplx(0.0)), AT(asz, cplx(0.0)), M2(asz, cplx(0.0)),
                      A1(asz, cplx(0.0)), A3(asz, cplx(0.0));
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
    auto base = [&](std::vector<cplx> &A, long part, long node) { return A.data() + size_t((part * np + node) * blk); };
    // add w * (the c-th block of pole p's output) into dst
    auto axpy_blk = [&](cplx *dst, cplx const *src, long c, cplx w) {
      for (long x = 0; x < nc; ++x) {
        cplx const *v = src + (x * ncomp + c) * ry;
        cplx *o = dst + x * ry;
        for (long e = 0; e < ry; ++e) o[e] += w * v[e];
      }
    };
    auto zero_blk = [&](cplx *b) { std::fill(b, b + blk, cplx(0.0)); };

    // one tile of poles; pass = 0 the j loop (U on Q, T on B), pass = 1 the l loop (-U, +i nu T on R)
    auto do_tile = [&](long p0, long np_tile, int pass) {
      const cplx sU = (pass == 0) ? cplx(1.0) : cplx(-1.0);
      const cplx sT = (pass == 0) ? cplx(1.0) : inu;
      for (long c = 0; c < ncomp; ++c) {
        const long part = (c == 0) ? 1 : 0;
        const long aU = (c >= 1 && c <= np) ? c - 1 : ((c > np) ? c - 1 - np : -1);
        const bool U_is_a = (c >= 1 && c <= np);          // mulU's a-indexed target is AU[a]
        const bool U_is_T = (c > np);                      // ... or AT[a]
        bool used_U = false, used_T = false;
        if (aU >= 0) { zero_blk(accU.data()); zero_blk(accT.data()); }
        for (long t = 0; t < np_tile; ++t) {
          const long pole = p0 + t;
          const long nj = in.gnode[size_t(pole)];
          const double ej = in.epsG[size_t(pole)];
          cplx const *Q = Qt.data() + size_t(t) * Wz;
          cplx const *B = (pass == 0) ? Bt.data() + size_t(t) * Wz : Q;
          // ---- mulU on sU * Q ----
          if (c == 0) axpy_blk(base(AU, part, nj), Q, c, sU);
          else if (U_is_a) {
            const long a = aU;
            if (a == nj) axpy_blk(base(M2, part, nj), Q, c, sU);
            else {
              const cplx w = sU * cplx(1.0 / (ej - in.eps[size_t(a)]));
              axpy_blk(base(AU, part, nj), Q, c, w);
              axpy_blk(accU.data(), Q, c, -w); used_U = true;         // -> AU[a], reduced over the tile
            }
          } else {
            const long a = aU;
            if (a == nj) axpy_blk(base(A1, part, nj), Q, c, sU);
            else {
              const double ea = in.eps[size_t(a)];
              const cplx w = cplx(1.0 / (ea - ej)), dd = cplx(ej - ea) + inu;
              const cplx wd = w / dd, wt = w - inu * wd;
              axpy_blk(accT.data(), Q, c, sU * wt); used_T = true;    // -> AT[a]
              axpy_blk(base(AU, part, nj), Q, c, -sU * wd);
              axpy_blk(accU.data(), Q, c, sU * wd); used_U = true;    // -> AU[a]
            }
          }
          // ---- mulT on sT * B ----
          if (c == 0) axpy_blk(base(AT, part, nj), B, c, sT);
          else if (U_is_a) {
            const long a = aU;
            const double ea = in.eps[size_t(a)];
            const cplx w = cplx(1.0 / (ea - ej)), dd = cplx(ej - ea) + inu;
            const cplx wd = w / dd, wt = w - inu * wd;
            axpy_blk(base(AT, part, nj), B, c, sT * wt);
            axpy_blk(accU.data(), B, c, sT * wd); used_U = true;      // -> AU[a]
            axpy_blk(base(AU, part, nj), B, c, -sT * wd);
          } else {
            const long a = aU;
            if (a == nj) axpy_blk(base(A3, part, nj), B, c, sT);
            else {
              const cplx w = sT * cplx(1.0 / (in.eps[size_t(a)] - ej));
              axpy_blk(accT.data(), B, c, w); used_T = true;          // -> AT[a]
              axpy_blk(base(AT, part, nj), B, c, -w);
            }
          }
        }
        if (aU >= 0 && used_U) { cplx *dst = base(AU, part, aU); for (long e = 0; e < blk; ++e) dst[e] += accU[size_t(e)]; }
        if (aU >= 0 && used_T) { cplx *dst = base(AT, part, aU); for (long e = 0; e < blk; ++e) dst[e] += accT[size_t(e)]; }
      }
    };

    for (int pass = 0; pass < 2; ++pass)
      for (long p0 = 0; p0 < ng; p0 += PB) {
        const long nt = std::min(PB, ng - p0);
        for (long t = 0; t < nt; ++t) {
          const long pole = p0 + t;
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              gjT[size_t(x * nc + y)] = in.gk[size_t(((pole * nk + ik) * nc + y) * nc + x)];
              glT[size_t(x * nc + y)] = in.gkq[size_t(((pole * nk + ik) * nc + y) * nc + x)];
              Gh[size_t(x * nc + y)] = (pass == 0) ? in.Ghat[size_t(((ik * ng + pole) * nc + x) * nc + y)]
                                                   : in.Gtil[size_t(((ik * ng + pole) * nc + x) * nc + y)];
            }
          if (pass == 0) {
            gemm_rm(nc, ncomp * nR * nc, nc, gjT.data(), Vt.data(), Pj.data());
            gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), Gh.data(), Qt.data() + size_t(t) * Wz);
            gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Bt.data() + size_t(t) * Wz);
          } else {
            gemm_rm(nc, ncomp * nR * nc, nc, Gh.data(), Vt.data(), Pj.data());
            gemm_rm(nc * ncomp * nR, nc, nc, Pj.data(), glT.data(), Qt.data() + size_t(t) * Wz);
          }
        }
        do_tile(p0, nt, pass);
      }

    for (int part = 0; part < 2; ++part)
      for (long n = 0; n < np; ++n) {
        const cplx wh(in.fhalf[size_t(n)]), w1(in.fd1[size_t(n)]);
        cplx const *au = AU.data() + size_t((part * np + n) * blk);
        cplx const *at = AT.data() + size_t((part * np + n) * blk);
        cplx const *m2 = M2.data() + size_t((part * np + n) * blk);
        for (long x = 0; x < nc; ++x)
          for (long r = 0; r < nR; ++r)
            for (long y = 0; y < nc; ++y) {
              const long e = (x * nR + r) * nc + y;
              const size_t f0 = size_t((((0 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              const size_t f1 = size_t((((1 * np + n) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r);
              out.Ffam[f0] += au[e];
              out.Ffam[f1] += at[e];
              if (part == 0) out.Fsum[size_t(((ik * nc + x) * nc + y) * nR + r)] += wh * au[e] + w1 * m2[e];
              if (m2[e] != cplx(0.0))
                for (long c = 0; c < np; ++c) {
                  const cplx dc = in.Dsq[size_t(n * np + c)];
                  if (dc == cplx(0.0)) continue;
                  out.Ffam[size_t((((0 * np + c) * nk + ik) * nc + x) * nc + y) * size_t(nR) + size_t(r)] += dc * m2[e];
                }
            }
      }
  }
}
long &tile_pb() { static long v = 8; return v; }
void l0_tiled_w(inputs const &in, outputs &out, int nthreads) { l0_tiled(in, out, nthreads, tile_pb()); }

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

  std::string which = "both";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--kernel=", 0) == 0) which = argv[i] + 9;

  inputs in; in.build(d);
  outputs ref, opt; ref.alloc(d); opt.alloc(d);
  const int reps = 3;
  auto bench = [&](const char *tag, void (*fn)(inputs const &, outputs &, int), outputs &o) {
    double best = 1e30;
    for (int it = 0; it < reps; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      fn(in, o, nthreads);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      best = std::min(best, s);
      std::printf("[%s] threads %d, application %d: %.3f s\n", tag, nthreads, it, s);
    }
    double chk = 0.0;
    for (auto const &z : o.Ffam) chk += std::norm(z);
    std::printf("[%s] BEST %.3f s -> %.1f GFLOP/s, %.1f GB/s effective; |F|^2 = %.10e\n",
                tag, best, gf / best, gb / best, chk);
    return best;
  };
  double tref = 0.0, topt = 0.0;
  // "prod" is the production kernel's structure (views in the j pass, materialised blocks in the l pass);
  // "opt" folds the l-pass scalars into the weights and drops that materialisation; "ref" is the naive form.
  outputs prod; prod.alloc(d);
  double tprod = 0.0;
  if (which == "ref" or which == "both") tref = bench("ref", l0_ref, ref);
  if (which == "prod" or which == "both") tprod = bench("prod", l0_prod, prod);
  if (which == "opt" or which == "both") topt = bench("opt", l0_opt, opt);
  outputs tl; tl.alloc(d);
  double ttl = 0.0;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--pb=", 0) == 0) tile_pb() = std::atol(argv[i] + 5);
  if (which == "tiled" or which == "both") {
    ttl = bench("tiled", l0_tiled_w, tl);
    std::printf("[tiled] pole tile PB = %ld\n", tile_pb());
  }
  if (which == "both") {
    std::printf("[check] opt vs ref  : F %.3e, Fsum %.3e\n", rel_diff(opt.Ffam, ref.Ffam), rel_diff(opt.Fsum, ref.Fsum));
    std::printf("[check] prod vs ref : F %.3e, Fsum %.3e\n", rel_diff(prod.Ffam, ref.Ffam), rel_diff(prod.Fsum, ref.Fsum));
    std::printf("[speedup] opt / prod = %.2fx   (THE portable number: prod == what dynbse.hpp does today)\n", tprod / topt);
    std::printf("[check] tiled vs ref: F %.3e, Fsum %.3e\n", rel_diff(tl.Ffam, ref.Ffam), rel_diff(tl.Fsum, ref.Fsum));
    std::printf("[speedup] opt / ref  = %.2fx\n", tref / topt);
    std::printf("[speedup] tiled / prod = %.2fx   (the traffic restructuring against what dynbse.hpp does today)\n", tprod / ttl);
  }
  return 0;
}
