// The device streaming THC rung (src/methods/vertex/cuda/rung_cuda.cu) on random data at a production shape, timed per phase.
//   nvcc -O3 -std=c++17 -arch=sm_80 -Ishim -I../../src rung_bench.cu -lcublas -lcufft -o rung_bench   (RS_LAYOUT=0/1/2, COQUI_RS_FZ_G)
//   ./rung_bench [n1 n2 n3 Nm nc nR nb reps]      (default: Si kp444, C = 12, Nm 291, one RHS block of 32)
// The engine source is compiled in directly (the shim replaces CoQui's AppAbort); checks the result against a direct
// k-sum on a few output entries.
#include "methods/vertex/cuda/rung_cuda.cu"   // -I<src tree or a private copy>
#include <random>
#include <cmath>
using namespace methods::solvers::dynbse_cuda;
int main(int argc, char **argv) {
  long n[3] = {4, 4, 4}, Nm = 291, nc = 12, nR = 32, nb = 16, reps = 10;
  if (argc > 8) { n[0] = atol(argv[1]); n[1] = atol(argv[2]); n[2] = atol(argv[3]); Nm = atol(argv[4]); nc = atol(argv[5]);
                  nR = atol(argv[6]); nb = atol(argv[7]); reps = atol(argv[8]); }
  const long nk = n[0] * n[1] * n[2], nc2 = nc * nc;
  // mesh: k index = lex row; q = the shift (1, 0, 0) for the leg partner, transfers q' = every mesh vector
  std::vector<long> lex(nk), kpq(nk);
  for (long k = 0; k < nk; ++k) lex[k] = k;
  for (long k = 0; k < nk; ++k) { long m0 = k / (n[1] * n[2]), r = k % (n[1] * n[2]); kpq[k] = (((m0 + 1) % n[0]) * n[1] * n[2]) + r; }
  std::mt19937_64 g(7); std::uniform_real_distribution<double> u(-1, 1);
  std::vector<cplx> X(nk * Nm * nc), W(nk * Nm * Nm), F(nk * nc2 * nR), O(nk * nc2 * nR);
  for (auto &v : X) v = cplx(u(g), u(g));
  for (auto &v : W) v = cplx(u(g), u(g));
  for (auto &v : F) v = cplx(u(g), u(g));
  rs_config c; c.ns = 1; c.nk = nk; c.nq = nk; c.Nm = Nm; c.nc = nc; c.nb = nb; c.ntab = 1;
  if (const char *l = getenv("RS_LAYOUT")) c.layout = atoi(l);
  for (int a = 0; a < 3; ++a) c.ndim[a] = int(n[a]);
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  char why[256] = {0};
  rung_stream *e = rs_create(c, nR, lex.data(), lex.data(), X.data(), double(fr), why, 256);
  if (!e) { printf("rs_create failed: %s\n", why); return 1; }
  printf("layout %d; mesh %ld x %ld x %ld (nk %ld), Nm %ld, nc %ld, nR %ld, nb %ld; engine %.2f GB\n", c.layout, n[0], n[1], n[2], nk, Nm, nc, nR, rs_nb(e),
         rs_bytes(c, nR) / 1e9);
  rs_set_sq(e, 0, kpq.data());
  double tw = 0; rs_load_w(e, 0, W.data(), Nm * Nm, &tw);
  void *Fd, *Od; cudaMalloc(&Fd, F.size() * 16); cudaMalloc(&Od, O.size() * 16);
  cudaMemcpy(Fd, F.data(), F.size() * 16, cudaMemcpyHostToDevice);
  double t4[4] = {0, 0, 0, 0};
  rs_apply_dev(e, 0, cplx(1.0), Fd, Od, nR, nullptr);   // warm-up
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (long r = 0; r < reps; ++r) rs_apply_dev(e, 0, cplx(1.0), Fd, Od, nR, nullptr);
  cudaDeviceSynchronize();
  const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
  for (long r = 0; r < reps; ++r) rs_apply_dev(e, 0, cplx(1.0), Fd, Od, nR, t4);
  const double Ybytes = 16.0 * nk * Nm * Nm * rs_nb(e);
  printf("per application (nR %ld): %.2f ms unsynced; phases (synced) legs-in %.2f, FFT+product %.2f, legs-out %.2f ms; W load %.1f ms\n",
         nR, 1e3 * tt, 1e3 * t4[0] / reps, 1e3 * t4[1] / reps, 1e3 * t4[2] / reps, 1e3 * tw);
  const double nblk = std::ceil(double(nR) / rs_nb(e));
  printf("Y block %.2f GB; per block %.2f ms -> %.0f GB/s per Y pass-equivalent\n", Ybytes / 1e9, 1e3 * tt / nblk, Ybytes / (tt / nblk) / 1e9);
  cudaMemcpy(O.data(), Od, O.size() * 16, cudaMemcpyDeviceToHost);
  // spot check: out(k', p1 nc + p3', N) = sum_k sum_PQ X(k',P,p1) conj X(k'+q,Q,p3') W_PQ(k - k') sum_{p1' p3} conj X(k,P,p1') F X(k+q,Q,p3)
  auto Xa = [&](long k, long P, long a) { return X[(k * Nm + P) * nc + a]; };
  auto qidx = [&](long k, long kp) { long v[3], a = k, b = kp; for (int d = 2; d >= 0; --d) { v[d] = ((a % n[d]) - (b % n[d]) + n[d]) % n[d]; a /= n[d]; b /= n[d]; }
    return (v[0] * n[1] + v[1]) * n[2] + v[2]; };
  double dmax = 0, smax = 0;
  for (long t = 0; t < 3; ++t) {
    const long kp = (t * 7) % nk, p1 = t % nc, p3p = (t * 5) % nc, N = (t * 11) % nR;
    cplx acc(0);
    for (long k = 0; k < nk; ++k) {
      const long iq = qidx(k, kp);
      for (long P = 0; P < Nm; ++P) {
        // Y(k, P, Q) = sum_{p1' p3} conj X(k,P,p1') F(k, p1' p3, N) X(k+q, Q, p3)
        for (long Q = 0; Q < Nm; ++Q) {
          cplx y(0);
          for (long a = 0; a < nc; ++a) for (long b = 0; b < nc; ++b) y += std::conj(Xa(k, P, a)) * F[(k * nc2 + a * nc + b) * nR + N] * Xa(kpq[k], Q, b);
          acc += Xa(kp, P, p1) * std::conj(Xa(kpq[kp], Q, p3p)) * W[(iq * Nm + P) * Nm + Q] * y;
        }
      }
    }
    const cplx got = O[(kp * nc2 + p1 * nc + p3p) * nR + N];
    dmax = std::max(dmax, std::abs(got - acc)); smax = std::max(smax, std::abs(acc));
  }
  printf("spot check vs the direct k-sum (3 entries): max|d| %.3e, max|ref| %.3e, rel %.2e\n", dmax, smax, dmax / smax);
  rs_destroy(e);
  return 0;
}
