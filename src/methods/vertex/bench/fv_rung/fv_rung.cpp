// ============================================================================================
// fv_rung -- factorize_vertex miniapp: the DYNAMIC RUNG APPLICATION of the Gamma_1 vertex, extracted.
//
//   y(k', tau_i) = sum_k K_d(tau_i)[(k', p1 p3'), (k, p1' p3)] F(k, tau_i)[(p1' p3)],   i = 0 .. nt-1,
//   K_d(tau)[(k',p1 p3'),(k,p1' p3)] = sum_PQ X_P,p1(k') X*_P,p1'(k) W_PQ(k - k', tau) X_Q,p3(k+q) X*_Q,p3'(k'+q)
// (vertex_dynbse.icc::build_kbig / kd, the scale -1/nk and the constant part omitted: they are identical in every variant).
// Real Si 4^3 data (the W-bar cache dump, analysis/prep_bench.py). Variants:
//   dense   : the production path. K_d(tau_r) built per PH-representative tau node (ndist D x D matrices); nt gemms (D x D)(D x nR).
//   dense2  : same matrices, the two mirror nodes of a representative in ONE gemm (D x D)(D x 2 nR): half the K traffic.
//   freqR   : W(q', tau) ~ sum_r c_r(tau) A_r(q') (universal time functions): R matrices K_r = Kbig[A_r];
//             Y = [K_1 .. K_R] [c_1 . F ; .. ; c_R . F]  -- ONE gemm (D x R D)(R D x nt nR).
//   spatial : W(q', tau) ~ U(q') Lam(q', tau) U(q')^dag per transfer; tables m_s(k',k) = U^T U1, n_t(k,k') = U^dag U2 (tau-free);
//             per tau: mt_t = sum_s Lam_st m_s, then out(k') = sum_(k,t) mt_t F(k) n_t in two gemm stages (K = nk Ns nc).
// Every variant is checked against dense (relative Frobenius error of y) and timed (build, apply per application).
// ============================================================================================
#include <complex>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <hdf5.h>
#include <mkl.h>

using cplx = std::complex<double>;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- h5 helpers: complex arrays stored as float64 (..., 2); integer arrays as int64 --------------------------------
static std::vector<hsize_t> h5dims(hid_t f, const char *name) {
  hid_t d = H5Dopen2(f, name, H5P_DEFAULT); hid_t s = H5Dget_space(d);
  int nd = H5Sget_simple_extent_ndims(s); std::vector<hsize_t> dims(nd); H5Sget_simple_extent_dims(s, dims.data(), nullptr);
  H5Sclose(s); H5Dclose(d); return dims;
}
static std::vector<cplx> h5c(hid_t f, const char *name) {
  auto dims = h5dims(f, name); size_t n = 1; for (auto x : dims) n *= x;
  std::vector<cplx> v(n / 2); hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
  H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()); H5Dclose(d); return v;
}
static std::vector<long> h5l(hid_t f, const char *name) {
  auto dims = h5dims(f, name); size_t n = 1; for (auto x : dims) n *= x;
  std::vector<long> v(n); hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
  H5Dread(d, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()); H5Dclose(d); return v;
}

static const cplx one(1.0), zero(0.0);
// row-major gemm C(MxN) = alpha op(A) op(B) + beta C
static void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, long M, long N, long K, cplx alpha, const cplx *A, long lda,
                 const cplx *B, long ldb, cplx beta, cplx *C, long ldc) {
  cblas_zgemm(CblasRowMajor, ta, tb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

struct data {
  long nk, Nm, nc, nc2, D, nq, nt, ndist, R, Ns;
  std::vector<cplx> X, Wt, ctau, A, U, Lam;
  std::vector<long> kmq, kpq, trep, reps, tmirror, qx_of;   // qx_of(k, k')
  cplx Xa(long k, long P, long a) const { return X[(k * Nm + P) * nc + a]; }
};

// U1(P, j1 nc + j1') = X(k', P, j1) conj X(k, P, j1');  U2(Q, j3 nc + j3') = X(k+q, Q, j3) conj X(k'+q, Q, j3')
static void legs(data const &d, long k, long kp, cplx *U1, cplx *U2) {
  const long nc = d.nc, Nm = d.Nm, nc2 = d.nc2;
  const long kq = d.kpq[k], kpq_ = d.kpq[kp];
  for (long P = 0; P < Nm; ++P)
    for (long a = 0; a < nc; ++a) {
      const cplx x1 = d.Xa(kp, P, a), x3 = d.Xa(kq, P, a);
      for (long b = 0; b < nc; ++b) {
        U1[P * nc2 + a * nc + b] = x1 * std::conj(d.Xa(k, P, b));
        U2[P * nc2 + a * nc + b] = x3 * std::conj(d.Xa(kpq_, P, b));
      }
    }
}

// Kbig[W] for a per-transfer W accessor (Nm x Nm row-major), the build_kbig layout; K row-major (D x D), ldk = row stride
template <class WF>
static void build_kbig(data const &d, WF Wof, cplx *K, long ldk) {
  const long nk = d.nk, Nm = d.Nm, nc = d.nc, nc2 = d.nc2;
#pragma omp parallel
  {
    std::vector<cplx> U1(Nm * nc2), U2(Nm * nc2), WU2(Nm * nc2), wb(nc2 * nc2);
#pragma omp for collapse(2) schedule(dynamic, 4)
    for (long k = 0; k < nk; ++k)
      for (long kp = 0; kp < nk; ++kp) {
        legs(d, k, kp, U1.data(), U2.data());
        const cplx *W = Wof(d.qx_of[k * nk + kp]);
        gemm(CblasNoTrans, CblasNoTrans, Nm, nc2, Nm, one, W, Nm, U2.data(), nc2, zero, WU2.data(), nc2);
        gemm(CblasTrans, CblasNoTrans, nc2, nc2, Nm, one, U1.data(), nc2, WU2.data(), nc2, zero, wb.data(), nc2);
        for (long p1 = 0; p1 < nc; ++p1)
          for (long p1p = 0; p1p < nc; ++p1p)
            for (long p3 = 0; p3 < nc; ++p3)
              for (long p3p = 0; p3p < nc; ++p3p)
                K[(kp * nc2 + p1 * nc + p3p) * ldk + k * nc2 + p1p * nc + p3] = wb[(p1 * nc + p1p) * nc2 + p3 * nc + p3p];
      }
  }
}

static double relerr(std::vector<cplx> const &a, std::vector<cplx> const &b) {
  double e = 0, n = 0;
  for (size_t i = 0; i < a.size(); ++i) { e += std::norm(a[i] - b[i]); n += std::norm(b[i]); }
  return std::sqrt(e / n);
}

int main(int argc, char **argv) {
  if (argc < 2) { std::printf("usage: fv_rung <bench.h5> [nR=32] [napply=2] [variants=dense,dense2,freqR,spatial]\n"); return 1; }
  const long nR = argc > 2 ? std::atol(argv[2]) : 32;
  const int napply = argc > 3 ? std::atoi(argv[3]) : 2;
  const std::string variants = argc > 4 ? argv[4] : "dense,dense2,freqR,spatial";
  auto want = [&](const char *v) { return variants.find(v) != std::string::npos; };
  data d;
  double t0 = now();
  {
    hid_t f = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    auto xd = h5dims(f, "X"); d.nk = xd[0]; d.Nm = xd[1]; d.nc = xd[2];
    d.X = h5c(f, "X"); d.kmq = h5l(f, "kmq"); d.kpq = h5l(f, "kpq"); d.trep = h5l(f, "trep"); d.reps = h5l(f, "reps");
    d.tmirror = h5l(f, "tmirror");
    auto wd = h5dims(f, "Wt"); d.nq = wd[0]; d.ndist = wd[1]; d.Wt = h5c(f, "Wt");
    auto cd = h5dims(f, "ctau"); d.nt = cd[0]; d.R = cd[1]; d.ctau = h5c(f, "ctau"); d.A = h5c(f, "A");
    auto ud = h5dims(f, "U"); d.Ns = ud[2]; d.U = h5c(f, "U"); d.Lam = h5c(f, "Lam");
    H5Fclose(f);
  }
  const long nk = d.nk, Nm = d.Nm, nc = d.nc, nc2 = nc * nc, D = nc2 * nk, nt = d.nt, ndist = d.ndist, R = d.R, Ns = d.Ns;
  d.nc2 = nc2; d.D = D;
  d.qx_of.assign(nk * nk, -1);
  for (long q = 0; q < d.nq; ++q)
    for (long k = 0; k < nk; ++k) d.qx_of[k * nk + d.kmq[q * nk + k]] = q;
  const double gb = 16.0 / 1e9;
  std::printf("[fv_rung] nk %ld Nm %ld nc %ld D %ld nq %ld nt %ld ndist %ld R %ld Ns %ld nR %ld threads %d (load %.1f s)\n", nk, Nm,
              nc, D, d.nq, nt, ndist, R, Ns, nR, omp_get_max_threads(), now() - t0);
  // random input F(tau_i): (nt, D, nR)
  std::vector<cplx> F(size_t(nt) * D * nR);
  {
    unsigned s = 12345;
    for (auto &v : F) { s = s * 1664525u + 1013904223u; double a = (s >> 8) * (1.0 / 16777216.0) - 0.5; s = s * 1664525u + 1013904223u;
                        double b = (s >> 8) * (1.0 / 16777216.0) - 0.5; v = cplx(a, b); }
  }
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
    std::vector<cplx> m(size_t(nk) * nk * Ns * nc2), NT(size_t(nk) * nc * nk * nsc);
    double tb = now();
#pragma omp parallel
    {
      std::vector<cplx> U1(Nm * nc2), U2(Nm * nc2), nn(Ns * nc2);
#pragma omp for collapse(2) schedule(dynamic, 4)
      for (long kp = 0; kp < nk; ++kp)
        for (long k = 0; k < nk; ++k) {
          legs(d, k, kp, U1.data(), U2.data());
          const cplx *Uq = &d.U[size_t(d.qx_of[k * nk + kp]) * Nm * Ns];     // (Nm, Ns)
          // m = U^T U1 : (Ns x nc2);  n = U^dag U2 : (Ns x nc2)
          gemm(CblasTrans, CblasNoTrans, Ns, nc2, Nm, one, Uq, Ns, U1.data(), nc2, zero, &m[(size_t(kp) * nk + k) * Ns * nc2], nc2);
          gemm(CblasConjTrans, CblasNoTrans, Ns, nc2, Nm, one, Uq, Ns, U2.data(), nc2, zero, nn.data(), nc2);
          for (long t = 0; t < Ns; ++t)
            for (long j3 = 0; j3 < nc; ++j3)
              for (long j3p = 0; j3p < nc; ++j3p)
                NT[(size_t(kp) * nc + j3p) * (nk * nsc) + (k * Ns + t) * nc + j3] = nn[t * nc2 + j3 * nc + j3p];
        }
    }
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
