// ============================================================================================
// fv_rung -- factorized-vertex miniapp: the DYNAMIC RUNG APPLICATION of the Gamma_1 vertex, extracted.
//
//   y(k', tau_i) = sum_k K_d(tau_i)[(k', p1 p3'), (k, p1' p3)] F(k, tau_i)[(p1' p3)],   i = 0 .. nt-1,
//   K_d(tau)[(k',p1 p3'),(k,p1' p3)] = sum_PQ X_P,p1(k') X*_P,p1'(k) W_PQ(k - k', tau) X_Q,p3(k+q) X*_Q,p3'(k'+q)
// (vertex_dynbse.icc::build_kbig / kd, the scale -1/nk and the constant part omitted: they are identical in every variant).
// Input: real data in an HDF5 file (load_data in fv_common.hpp: the collocation X, the rung W per transfer and tau
// representative, the k maps, and the freqR / spatial factors ctau, A, U, Lam), e.g. from a W-bar cache dump. Variants:
//   dense   : the production path. K_d(tau_r) built per PH-representative tau node (ndist D x D matrices); nt gemms (D x D)(D x nR).
//   dense2  : same matrices, the two mirror nodes of a representative in ONE gemm (D x D)(D x 2 nR): half the K traffic.
//   freqR   : W(q', tau) ~ sum_r c_r(tau) A_r(q') (universal time functions): R matrices K_r = Kbig[A_r];
//             Y = [K_1 .. K_R] [c_1 . F ; .. ; c_R . F]  -- ONE gemm (D x R D)(R D x nt nR).
//   spatial : W(q', tau) ~ U(q') Lam(q', tau) U(q')^dag per transfer; tables m_s(k',k) = U^T U1, n_t(k,k') = U^dag U2 (tau-free);
//             per tau: mt_t = sum_s Lam_st m_s, then out(k') = sum_(k,t) mt_t F(k) n_t in two gemm stages (K = nk Ns nc).
// Every variant is checked against dense (relative Frobenius error of y) and timed (build, apply per application).
// ============================================================================================
#include "fv_common.hpp"

int main(int argc, char **argv) {
  if (argc < 2) { std::printf("usage: fv_rung <bench.h5> [nR=32] [napply=2] [variants=dense,dense2,freqR,spatial]\n"); return 1; }
  const long nR = argc > 2 ? std::atol(argv[2]) : 32;
  const int napply = argc > 3 ? std::atoi(argv[3]) : 2;
  const std::string variants = argc > 4 ? argv[4] : "dense,dense2,freqR,spatial";
  auto want = [&](const char *v) { return variants.find(v) != std::string::npos; };
  data d;
  double t0 = now();
  load_data(argv[1], d);
  const long nk = d.nk, Nm = d.Nm, nc = d.nc, nc2 = nc * nc, D = nc2 * nk, nt = d.nt, ndist = d.ndist, R = d.R, Ns = d.Ns;
  const double gb = 16.0 / 1e9;
  std::printf("[fv_rung] nk %ld Nm %ld nc %ld D %ld nq %ld nt %ld ndist %ld R %ld Ns %ld nR %ld threads %d (load %.1f s)\n", nk, Nm,
              nc, D, d.nq, nt, ndist, R, Ns, nR, omp_get_max_threads(), now() - t0);
  // random input F(tau_i): (nt, D, nR)
  std::vector<cplx> F(size_t(nt) * D * nR);
  random_fill(F);
  std::vector<cplx> Yref(F.size()), Y(F.size());

  // ---------------- dense (and dense2 on the same matrices) ----------------
  if (want("dense")) {
    std::vector<cplx> Kd(size_t(ndist) * D * D);
    double tb = now();
    for (long r = 0; r < ndist; ++r)
      build_kbig(d, [&](long q) { return &d.Wt[((q * ndist + r) * Nm) * Nm]; }, &Kd[size_t(r) * D * D], D);
    tb = now() - tb;
    double ta = 1e30;
    for (int it = 0; it < napply; ++it) {
      double t = now();
      for (long i = 0; i < nt; ++i)
        gemm(CblasNoTrans, CblasNoTrans, D, nR, D, one, &Kd[size_t(d.trep[i]) * D * D], D, &F[size_t(i) * D * nR], nR, zero,
             &Yref[size_t(i) * D * nR], nR);
      ta = std::min(ta, now() - t);
    }
    const double fl = 8.0 * nt * double(D) * D * nR;
    std::printf("  dense   : build %.2f s (%ld tau reps), apply %.3f s/app (%.1f GF/s), K memory %.2f GB\n", tb, ndist, ta, fl / ta / 1e9,
                ndist * double(D) * D * gb);
    if (want("dense2")) {
      // the two mirror nodes of each representative in one gemm (a gather into a (D x 2nR) block)
      std::vector<cplx> F2(size_t(D) * 2 * nR), Y2(size_t(D) * 2 * nR);
      double t2 = 1e30;
      for (int it = 0; it < napply; ++it) {
        double t = now();
        for (long r = 0; r < ndist; ++r) {
          const long i0 = d.reps[r], i1 = d.tmirror[i0];
          const long nb = (i1 == i0) ? 1 : 2;
#pragma omp parallel for
          for (long x = 0; x < D; ++x)
            for (long b = 0; b < nb; ++b)
              std::memcpy(&F2[(x * 2 + b) * nR], &F[(size_t(b ? i1 : i0) * D + x) * nR], nR * sizeof(cplx));
          gemm(CblasNoTrans, CblasNoTrans, D, 2 * nR, D, one, &Kd[size_t(r) * D * D], D, F2.data(), 2 * nR, zero, Y2.data(), 2 * nR);
#pragma omp parallel for
          for (long x = 0; x < D; ++x)
            for (long b = 0; b < nb; ++b)
              std::memcpy(&Y[(size_t(b ? i1 : i0) * D + x) * nR], &Y2[(x * 2 + b) * nR], nR * sizeof(cplx));
        }
        t2 = std::min(t2, now() - t);
      }
      std::printf("  dense2  : apply %.3f s/app (mirror pairs in one gemm), err vs dense %.1e\n", t2, relerr(Y, Yref));
    }
  }

  // ---------------- freqR: R matrices, one long gemm ----------------
  if (want("freqR")) {
    std::vector<cplx> Kc(size_t(D) * R * D);          // [K_1 | K_2 | ...]: row x, column r D + y
    double tb = now();
    for (long r = 0; r < R; ++r)
      build_kbig(d, [&](long q) { return &d.A[((r * d.nq + q) * Nm) * Nm]; }, &Kc[size_t(r) * D], R * D);
    tb = now() - tb;
    // stacked right side: rows (r, y), columns (i, N):  c_r(tau_i) F(tau_i)(y, N)
    std::vector<cplx> Fs(size_t(R) * D * nt * nR), Ys(size_t(D) * nt * nR);
    double ta = 1e30, tp = 0;
    for (int it = 0; it < napply; ++it) {
      double t = now();
#pragma omp parallel for collapse(2)
      for (long r = 0; r < R; ++r)
        for (long y = 0; y < D; ++y)
          for (long i = 0; i < nt; ++i) {
            const cplx c = d.ctau[i * R + r];
            const cplx *src = &F[(size_t(i) * D + y) * nR];
            cplx *dst = &Fs[((size_t(r) * D + y) * nt + i) * nR];
            for (long N = 0; N < nR; ++N) dst[N] = c * src[N];
          }
      const double t1 = now();
      gemm(CblasNoTrans, CblasNoTrans, D, nt * nR, R * D, one, Kc.data(), R * D, Fs.data(), nt * nR, zero, Ys.data(), nt * nR);
      const double t2 = now();
#pragma omp parallel for collapse(2)
      for (long i = 0; i < nt; ++i)
        for (long x = 0; x < D; ++x) std::memcpy(&Y[(size_t(i) * D + x) * nR], &Ys[(size_t(x) * nt + i) * nR], nR * sizeof(cplx));
      ta = std::min(ta, now() - t); tp = t2 - t1;
    }
    const double fl = 8.0 * R * double(D) * D * nt * nR;
    std::printf("  freqR   : R %ld, build %.2f s, apply %.3f s/app (gemm %.3f s, %.1f GF/s), K memory %.2f GB, err vs dense %.2e\n", R, tb,
                ta, tp, fl / tp / 1e9, R * double(D) * D * gb, want("dense") ? relerr(Y, Yref) : -1.0);
  }

  // ---------------- spatial: per-transfer modes, tau-free tables ----------------
  if (want("spatial")) {
    const long nsc = Ns * nc;
    // m(k', k): (Ns, nc, nc) [s][j1][j1'];  NT(k'): (nc_j3', nk Ns nc_j3)  (= n_t(k,k')[j3][j3'] transposed and stacked over (k,t))
    std::vector<cplx> m, NT;
    double tb = now();
    build_spatial_tables(d, m, NT);
    tb = now() - tb;
    // per tau representative: mt(k', k) = sum_s Lam(q', r)[s][t] m_s  -> [t][j1][j1'];  stage 1: C(k)[j1][t][j3][N] = mt_t F(k) (for each k');
    // stage 2: out(k')[j1] (nc_j3' x nR) = NT(k') (nc x nk Ns nc) * Cbig[j1] (nk Ns nc x nR)
    const int nth = omp_get_max_threads();
    std::vector<cplx> mt(size_t(nk) * nk * Ns * nc2);
    std::vector<std::vector<cplx>> Cb(nth, std::vector<cplx>(size_t(nc) * nk * nsc * nR));
    double ta = 1e30, tm_ = 0, ts_ = 0;
    for (int it = 0; it < napply; ++it) {
      double t = now(), tm = 0, ts = 0;
      for (long r = 0; r < ndist; ++r) {
        double t1 = now();
#pragma omp parallel for collapse(2) schedule(static)
        for (long kp = 0; kp < nk; ++kp)
          for (long k = 0; k < nk; ++k) {
            const cplx *L = &d.Lam[((size_t(d.qx_of[k * nk + kp]) * ndist + r) * Ns) * Ns];   // Lam[s][t]
            // mt[t][x] = sum_s L[s][t] m[s][x]
            gemm(CblasTrans, CblasNoTrans, Ns, nc2, Ns, one, L, Ns, &m[(size_t(kp) * nk + k) * Ns * nc2], nc2, zero,
                 &mt[(size_t(kp) * nk + k) * Ns * nc2], nc2);
          }
        tm += now() - t1; t1 = now();
        const long i0 = d.reps[r], i1 = d.tmirror[i0];
        for (long ii = 0; ii < ((i1 == i0) ? 1 : 2); ++ii) {
          const long i = ii ? i1 : i0;
#pragma omp parallel for schedule(dynamic, 1)
          for (long kp = 0; kp < nk; ++kp) {
            cplx *C = Cb[omp_get_thread_num()].data();                      // [j1][k][t][j3][N]
            for (long k = 0; k < nk; ++k) {
              const cplx *f = &F[(size_t(i) * D + k * nc2) * nR];           // [j1'][j3][N]
              for (long tt = 0; tt < Ns; ++tt)
                gemm(CblasNoTrans, CblasNoTrans, nc, nc * nR, nc, one, &mt[((size_t(kp) * nk + k) * Ns + tt) * nc2], nc, f, nc * nR,
                     zero, C + (k * Ns + tt) * nc * nR, nk * nsc * nR);
            }
            for (long j1 = 0; j1 < nc; ++j1)
              gemm(CblasNoTrans, CblasNoTrans, nc, nR, nk * nsc, one, &NT[size_t(kp) * nc * nk * nsc], nk * nsc,
                   C + size_t(j1) * nk * nsc * nR, nR, zero, &Y[(size_t(i) * D + kp * nc2 + j1 * nc) * nR], nR);
          }
        }
        ts += now() - t1;
      }
      if (now() - t < ta) { ta = now() - t; tm_ = tm; ts_ = ts; }
    }
    const double fl = 8.0 * nt * double(nk) * nk * (2.0 * Ns * nc2 * nc * nR) + 8.0 * ndist * double(nk) * nk * Ns * Ns * nc2;
    std::printf("  spatial : Ns %ld, tables %.2f s (%.2f GB), apply %.3f s/app (Lam fold %.3f, gemm stages %.3f; %.1f GF/s), err vs dense %.2e\n", Ns, tb,
                (m.size() + NT.size()) * gb, ta, tm_, ts_, fl / ta / 1e9, want("dense") ? relerr(Y, Yref) : -1.0);
  }
  std::printf("[fv_rung] done %.1f s\n", now() - t0);
  return 0;
}
