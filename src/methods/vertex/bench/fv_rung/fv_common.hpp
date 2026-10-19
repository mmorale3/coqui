// factorized-vertex rung miniapp: shared host code (inputs, dense build_kbig, factorized tables)
#pragma once
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


// spatial tables: m(k',k) (Ns, nc2) = U^T U1 and NT(k') (nc_j3', nk Ns nc_j3) = n_t(k,k')[j3][j3'] stacked over (k, t)
static void build_spatial_tables(data const &d, std::vector<cplx> &m, std::vector<cplx> &NT) {
  const long nk = d.nk, Nm = d.Nm, nc = d.nc, nc2 = d.nc2, Ns = d.Ns, nsc = Ns * nc;
  m.assign(size_t(nk) * nk * Ns * nc2, 0.0); NT.assign(size_t(nk) * nc * nk * nsc, 0.0);
#pragma omp parallel
  {
    std::vector<cplx> U1(Nm * nc2), U2(Nm * nc2), nn(Ns * nc2);
#pragma omp for collapse(2) schedule(dynamic, 4)
    for (long kp = 0; kp < nk; ++kp)
      for (long k = 0; k < nk; ++k) {
        legs(d, k, kp, U1.data(), U2.data());
        const cplx *Uq = &d.U[size_t(d.qx_of[k * nk + kp]) * Nm * Ns];
        gemm(CblasTrans, CblasNoTrans, Ns, nc2, Nm, one, Uq, Ns, U1.data(), nc2, zero, &m[(size_t(kp) * nk + k) * Ns * nc2], nc2);
        gemm(CblasConjTrans, CblasNoTrans, Ns, nc2, Nm, one, Uq, Ns, U2.data(), nc2, zero, nn.data(), nc2);
        for (long t = 0; t < Ns; ++t)
          for (long j3 = 0; j3 < nc; ++j3)
            for (long j3p = 0; j3p < nc; ++j3p)
              NT[(size_t(kp) * nc + j3p) * (nk * nsc) + (k * Ns + t) * nc + j3] = nn[t * nc2 + j3 * nc + j3p];
      }
  }
}

static void load_data(const char *fn, data &d) {
  hid_t f = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
  auto xd = h5dims(f, "X"); d.nk = xd[0]; d.Nm = xd[1]; d.nc = xd[2];
  d.X = h5c(f, "X"); d.kmq = h5l(f, "kmq"); d.kpq = h5l(f, "kpq"); d.trep = h5l(f, "trep"); d.reps = h5l(f, "reps");
  d.tmirror = h5l(f, "tmirror");
  auto wd = h5dims(f, "Wt"); d.nq = wd[0]; d.ndist = wd[1]; d.Wt = h5c(f, "Wt");
  auto cd = h5dims(f, "ctau"); d.nt = cd[0]; d.R = cd[1]; d.ctau = h5c(f, "ctau"); d.A = h5c(f, "A");
  auto ud = h5dims(f, "U"); d.Ns = ud[2]; d.U = h5c(f, "U"); d.Lam = h5c(f, "Lam");
  H5Fclose(f);
  d.nc2 = d.nc * d.nc; d.D = d.nc2 * d.nk;
  d.qx_of.assign(d.nk * d.nk, -1);
  for (long q = 0; q < d.nq; ++q)
    for (long k = 0; k < d.nk; ++k) d.qx_of[k * d.nk + d.kmq[q * d.nk + k]] = q;
}

static void random_fill(std::vector<cplx> &F) {
  unsigned s = 12345;
  for (auto &v : F) { s = s * 1664525u + 1013904223u; double a = (s >> 8) * (1.0 / 16777216.0) - 0.5; s = s * 1664525u + 1013904223u;
                      double b = (s >> 8) * (1.0 / 16777216.0) - 0.5; v = cplx(a, b); }
}
