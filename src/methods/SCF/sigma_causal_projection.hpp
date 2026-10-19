/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Causal projection of the dynamic self-energy (e.g. a vertex-corrected Sigma_c with a small non-causal part).
 *
 * A causal Sigma_c has a positive semidefinite spectral function:
 *   -Sigma_c(tau) = int dw K(tau, w) A(w),   A(w) >= 0 (matrix),   K(tau, w) = e^{-w tau} / (1 + e^{-beta w}) > 0.
 * Per (s, k) the Hermitian part Y(tau) = -Sigma_c^H(tau) on the fermionic tau mesh is fitted as
 *   Y(tau) ~ sum_l K(tau, w_l) A_l,   A_l PSD,
 * on real-frequency nodes w_l chosen by pivoted Gram-Schmidt from a fine geometric grid (a DLR-like set: the fine grid
 * alone makes the kernel near-singular), by ADMM on
 *   min 1/2 |Y - Kt B|^2 + I_PSD(C),  B = C
 * (Kt = K with unit columns, which keeps positivity; PSD = clip the negative eigenvalues of each node's matrix). The
 * result
 *   Sigma_c(tau) <- -sum_l K(tau, w_l) A_l
 * is causal by construction: Sigma(i w) = sum_l A_l / (i w - w_l) with A_l >= 0 is a Herglotz matrix function, so
 * Dyson's G is causal for any Hermitian static part, and a convex mix of two such Sigma_c (damping) stays causal.
 * The removed part Y - Kt C is what no causal Sigma_c on the nodes can produce plus the solver's distance from the
 * exact projection (see Convergence below).
 *
 * Solver. The ADMM runs in its Douglas-Rachford form on the argument v of the PSD clip:
 *   C = PSD(v),   B = (Kt^T Kt + rho)^{-1} (Kt^T Y + rho (2 C - v)),   v <- v + relax (B - C),
 * which is the over-relaxed scaled ADMM in (C, U = v - C) step for step (the warm-start state C, U keeps its meaning).
 * v is Anderson-accelerated (type II, memory `aa_mem`, Tikhonov `aa_reg`; the history restarts when |B - C| exceeds
 * `aa_restart` times its best value). The output weights are always C = PSD(v), so the result is causal at any
 * iteration count. Converged when the scaled-ADMM residuals pass: |B - C| <= tol |C| and rho |C - C_prev| <= tol
 * |Kt^T Y|. Otherwise stats::converged is false (stats::iters = maxit) and the causal, unconverged iterate is returned;
 * stats::fit_res = |Kt (B - C)| / |Y| is the last primal residual seen in tau.
 *
 * Convergence. The problem is ill-conditioned (79 nodes at beta 1000, wmax 15: singular values of Kt from 5.6 down to
 * ~3e-11) and degenerate on nearly causal input (optimal residual ~0, most nodes carry rank-deficient weights), so the
 * ADMM converges sublinearly and tol = 1e-10 is not reached in maxit on such input. What matters is the distance of the
 * output from the exact projection. On a synthetic production-like block (92 tau, 79 nodes, nb 12, a causal part
 * represented exactly on the nodes with mixed ranks plus a non-causal part of 1e-7 relative, so the exact answer is
 * known; unit test "sigma_causal_projection_convergence" and the hidden harness "[.conv]") at equal cost:
 *   plain ADMM (rho 1e-6 s_max^2, relax 1.6, 2000 iterations):           |result - exact| = 4.9e-6 |Y|
 *   Anderson   (memory 10, rho 3e-7 s_max^2, relax 1, 1800 iterations): |result - exact| = 1.3e-6 |Y|
 * (causal continuum of 400 off-node poles: removed 1.5e-6 -> 6.5e-7 |Y|; non-causal part of 1e-3: error 1.8e-4 ->
 * 1.3e-4). Each warm-started call continues from there. Tried and rejected: residual balancing / cycling / continuation
 * of rho (any change of rho mid-run loses the progress; the node-space residual ratio points the wrong way), a
 * semismooth-Newton augmented Lagrangian (stalls at small KKT residuals far from the optimum: the node-space KKT residual
 * does not see the ill-conditioned directions), CGLS polishing on the identified faces (3x at best), fewer nodes.
 *
 * Threads: every OpenMP region takes an explicit num_threads(...) from the `omp_threads` knob (utilities/omp_threads.hpp);
 * OMP_NUM_THREADS stays 1 in production runs, so a region without the clause would run serially. The BLAS / LAPACK calls
 * inside the regions (column-block gemms, per-row eigendecompositions) are the per-thread work items (MKL runs them
 * sequentially inside a parallel region).
 *
 * Device path (CUDA builds): project_device() runs the same solver on the GPU for several (s, k) blocks at once
 * (methods/SCF/cuda/causal_proj_cuda.cu: the whole ADMM state device-resident, batched Jacobi PSD clips warm-started from
 * the previous iteration's eigenvectors, real x complex gemms, the scalar control in device kernels). It agrees with
 * project() to roundoff per iteration; over many iterations the ill-conditioned Anderson solve amplifies any roundoff
 * difference, the host path's own included.
 * ==========================================================================
 */

#ifndef COQUI_SIGMA_CAUSAL_PROJECTION_HPP
#define COQUI_SIGMA_CAUSAL_PROJECTION_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/blas.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utilities/check.hpp"
#include "utilities/omp_threads.hpp"
#if defined(ENABLE_CUDA)
#include "methods/SCF/cuda/causal_proj_cuda.cuh"
#endif

namespace methods {
namespace causal_proj {

  struct options {
    long maxit = 6000;        // ADMM iterations per (s, k) and SCF iteration; the warm start carries the convergence
                              // over to the next SCF iteration (the result is causal at any iteration count). At 8^3
                              // the gap bias of the unconverged projection is ~2 meV at 1800 and <= 0.1 meV at 18000.
    double tol = 1e-10;       // relative primal |B - C| <= tol |C| and dual rho |C - C_prev| <= tol |Kt^T Y| residuals
    double rho = 3e-7;        // ADMM penalty relative to s_max(Kt)^2
    double relax = 1.0;       // relaxation of the Douglas-Rachford step (1 = plain ADMM)
    long aa_mem = 10;         // Anderson memory on v (0 = none)
    double aa_restart = 10.0; // restart the Anderson history when |B - C| exceeds this factor times its best value
    double aa_reg = 1e-12;    // Tikhonov regularisation of the Anderson least squares (relative to the trace of its Gram matrix)
    double wmin = 1e-4;       // fine grid: geometric |w| in [wmin, wmax] Ha, nw points per sign
    double wmax = 100.0;      // scf_loop sets the imaginary-axis window's wmax (nodes beyond it are not representable)
    long nw = 400;
    double select = 1e-10;    // pivoted Gram-Schmidt tolerance (relative to the largest column)
    double jacobi_tol = 1e-14; // device path: a Jacobi pair of the PSD clip rotates when |A_pq| > jacobi_tol |A|_F
    bool device_profile = false; // device path: time the kernel classes and the PSD clip's phases (call_info)
  };

  struct stats {
    double y_norm = 0.0;      // |Y| (Frobenius over tau and the band matrix)
    double removed = 0.0;     // |Y - Kt C|
    double lam_before = 0.0;  // min over tau of lambda_min(-Sigma_c^H)
    double lam_after = 0.0;
    long iters = 0;
    bool converged = false;   // both tests passed before maxit
    double fit_res = 0.0;     // |Kt (B - C)| / |Y| at the last iteration (the primal residual seen in tau)
    long aa_restarts = 0;
  };

  // the real-frequency nodes and the ADMM operators of one tau mesh
  class projector {
   public:
    // x: the fermionic tau mesh scaled to [-1, 1] (IAFT::tau_mesh), tau = beta (x + 1) / 2
    projector(double beta, nda::ArrayOfRank<1> auto const &x, options const &o) : _o(o) {
      const long nt = x.shape(0);
      nda::array<double, 1> tau(nt);
      for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);
      // fine geometric grid, both signs
      std::vector<double> wf;
      const long nw = _o.nw;
      for (long i = 0; i < nw; ++i)
        wf.push_back(-_o.wmax * std::pow(_o.wmin / _o.wmax, double(i) / double(nw - 1)));
      for (long i = 0; i < nw; ++i)
        wf.push_back(_o.wmin * std::pow(_o.wmax / _o.wmin, double(i) / double(nw - 1)));
      const long nf = long(wf.size());
      auto kern = [&](long t, double w) {   // e^{-w tau} / (1 + e^{-beta w}), evaluated without overflow
        return (w >= 0.0) ? std::exp(-w * tau(t)) / (1.0 + std::exp(-beta * w))
                          : std::exp(w * (beta - tau(t))) / (1.0 + std::exp(beta * w));
      };
      nda::matrix<double> Kf(nt, nf);
      for (long t = 0; t < nt; ++t)
        for (long j = 0; j < nf; ++j) Kf(t, j) = kern(t, wf[size_t(j)]);
      // pivoted Gram-Schmidt on the unit columns
      nda::matrix<double> R(nt, nf);
      for (long j = 0; j < nf; ++j) {
        double n = 0.0;
        for (long t = 0; t < nt; ++t) n += Kf(t, j) * Kf(t, j);
        n = std::sqrt(n);
        for (long t = 0; t < nt; ++t) R(t, j) = Kf(t, j) / n;
      }
      std::vector<long> piv;
      double n0 = -1.0;
      for (long s = 0; s < std::min(nt, nf); ++s) {
        long jb = -1;
        double nb = -1.0;
        for (long j = 0; j < nf; ++j) {
          double n = 0.0;
          for (long t = 0; t < nt; ++t) n += R(t, j) * R(t, j);
          if (n > nb) { nb = n; jb = j; }
        }
        nb = std::sqrt(nb);
        if (n0 < 0.0) n0 = nb;
        if (nb < _o.select * n0) break;
        piv.push_back(jb);
        nda::array<double, 1> q(nt);
        for (long t = 0; t < nt; ++t) q(t) = R(t, jb) / nb;
        for (long j = 0; j < nf; ++j) {
          double c = 0.0;
          for (long t = 0; t < nt; ++t) c += q(t) * R(t, j);
          for (long t = 0; t < nt; ++t) R(t, j) -= c * q(t);
        }
      }
      std::sort(piv.begin(), piv.end());
      const long L = long(piv.size());
      _w = nda::array<double, 1>(L);
      _Kt = nda::matrix<ComplexType>(nt, L);
      for (long l = 0; l < L; ++l) {
        _w(l) = wf[size_t(piv[size_t(l)])];
        double n = 0.0;
        for (long t = 0; t < nt; ++t) n += Kf(t, piv[size_t(l)]) * Kf(t, piv[size_t(l)]);
        n = std::sqrt(n);
        for (long t = 0; t < nt; ++t) _Kt(t, l) = Kf(t, piv[size_t(l)]) / n;
      }
      // Kt^T Kt, its largest eigenvalue, and Minv = (Kt^T Kt + rho)^{-1}
      _KtT = nda::matrix<ComplexType>(nda::transpose(_Kt));
      nda::matrix<ComplexType> G(L, L);
      nda::blas::gemm(ComplexType(1.0), _KtT, _Kt, ComplexType(0.0), G);
      auto ev = nda::linalg::eigenvalues(G);
      double smax2 = 0.0;
      for (auto const &v : ev) smax2 = std::max(smax2, double(std::real(v)));
      _rho = _o.rho * smax2;
      for (long l = 0; l < L; ++l) G(l, l) += _rho;
      _Minv = nda::inverse(G);
    }

    long n_nodes() const { return _w.shape(0); }
    nda::array<double, 1> const &nodes() const { return _w; }

    // Project one (s, k) block S (nt, nb, nb) in place. C, U (L, nb * nb): the ADMM state (v = C + U, C = PSD(v)), warm start
    // in and out (empty on the first call).
    stats project(nda::ArrayOfRank<3> auto &&S, nda::matrix<ComplexType> &C2, nda::matrix<ComplexType> &U2) const {
      const long nt = S.shape(0), nb = S.shape(1), L = n_nodes(), nb2 = nb * nb;
      utils::check(_Kt.shape(0) == nt and S.shape(2) == nb, "causal_proj::project: S is ({} x {} x {}), the mesh has {} tau nodes.",
                   S.shape(0), S.shape(1), S.shape(2), _Kt.shape(0));
      if (C2.shape(0) != L or C2.shape(1) != nb2) {
        C2 = nda::matrix<ComplexType>(L, nb2); C2() = 0.0;
        U2 = nda::matrix<ComplexType>(L, nb2); U2() = 0.0;
      }
      stats st;
      nda::matrix<ComplexType> Y(nt, nb2);
      for (long t = 0; t < nt; ++t)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) Y(t, i * nb + j) = -0.5 * (S(t, i, j) + std::conj(S(t, j, i)));
      st.y_norm = frob(Y);
      st.lam_before = lam_min(Y, nb);
      const long n_el = L * nb2;
      const long m = std::max(0l, _o.aa_mem);
      nda::matrix<ComplexType> KtY(L, nb2), V(L, nb2), C(L, nb2), Cp(L, nb2), B(L, nb2), Rhs(L, nb2), G(L, nb2), F(L, nb2),
          Gp(L, nb2), Fp(L, nb2), M(nt, nb2);
      // Anderson history: differences of the residual g = relax (B - C) and of the map value f = v + g (ring buffers)
      nda::array<ComplexType, 2> dG(std::max(m, 1l), n_el), dF(std::max(m, 1l), n_el);
      nda::matrix<double> H(std::max(m, 1l), std::max(m, 1l));
      gemm_cols(_KtT, Y, KtY);
      const double nkty = std::max(frob(KtY), 1e-300), ny = std::max(st.y_norm, 1e-300);
      const double rho = _rho, relax = _o.relax;
      ComplexType *pV = V.data(), *pC = C.data(), *pCp = Cp.data(), *pK = KtY.data(), *pR = Rhs.data(), *pB = B.data(),
                  *pG = G.data(), *pF = F.data(), *pGp = Gp.data(), *pFp = Fp.data();
      {
        const ComplexType *c2 = C2.data(), *u2 = U2.data();
#pragma omp parallel for schedule(static) num_threads(utils::omp_threads_for(n_el))
        for (long e = 0; e < n_el; ++e) pV[e] = c2[e] + u2[e];
      }
      [[maybe_unused]] const long nth = utils::omp_threads_for(n_el);
      copy(pCp, C2.data(), n_el, nth);
      long n_hist = 0, slot = 0;                      // stored differences (ring slots 0 .. n_hist - 1), next slot
      bool have_prev = false;
      double g_best = 1e300;
      long it = 0;
      for (; it < _o.maxit; ++it) {
        // C = PSD(v); B = (Kt^T Kt + rho)^{-1} (Kt^T Y + rho (2 C - v)); g = relax (B - C), f = v + g
        copy(pC, pV, n_el, nth);
        psd_rows(C, nb);
#pragma omp parallel for schedule(static) num_threads(nth)
        for (long e = 0; e < n_el; ++e) pR[e] = pK[e] + rho * (2.0 * pC[e] - pV[e]);
        gemm_cols(_Minv, Rhs, B);
        // g, f, the residual norms and (Anderson) the new differences dG, dF in the next ring slot, in one pass
        const bool store = m > 0 and have_prev;
        ComplexType *dg = dG.data() + slot * n_el, *df = dF.data() + slot * n_el;
        std::vector<double> sums;                        // |g|^2, |C - C_prev|^2, |C|^2
        blocked_sums(n_el, 3, nth, sums, [&](long e, double *acc) {
          const ComplexType g = relax * (pB[e] - pC[e]), f = pV[e] + g;
          if (store) { dg[e] = g - pGp[e]; df[e] = f - pFp[e]; }
          pG[e] = g;
          pF[e] = f;
          acc[0] += std::norm(g);
          acc[1] += std::norm(pC[e] - pCp[e]);
          acc[2] += std::norm(pC[e]);
          pCp[e] = pC[e];
        });
        const double g2 = sums[0], dc2 = sums[1], nc2 = sums[2];
        const double gn = std::sqrt(g2);
        // primal |B - C| <= tol |C| and dual rho |C - C_prev| <= tol |Kt^T Y| (the scaled-ADMM residuals)
        const bool conv = gn / relax <= _o.tol * std::max(std::sqrt(nc2), 1e-300) and rho * std::sqrt(dc2) <= _o.tol * nkty;
        if (conv or it + 1 == _o.maxit) {             // diagnostic: the same primal residual seen in tau, |Kt (B - C)| / |Y|
#pragma omp parallel for schedule(static) num_threads(nth)
          for (long e = 0; e < n_el; ++e) pR[e] = pB[e] - pC[e];
          gemm_cols(_Kt, Rhs, M);
          st.fit_res = frob(M) / ny;
        }
        if (conv) { st.converged = true; ++it; break; }
        if (m == 0) { copy(pV, pF, n_el, nth); continue; }
        // Anderson (type II) on v: v <- f - sum_i gamma_i dF_i, gamma = argmin |g - sum_i gamma_i dG_i|
        std::swap(pG, pGp); std::swap(pF, pFp);          // g, f -> the previous ones of the next iteration
        have_prev = true;
        if (gn > _o.aa_restart * g_best) { n_hist = 0; slot = 0; ++st.aa_restarts; }
        else if (store) { n_hist = std::min(n_hist + 1, m); slot = (slot + 1) % m; }
        g_best = std::min(g_best, gn);
        if (n_hist == 0) { copy(pV, pFp, n_el, nth); continue; }
        // the Gram row of the newest difference and the right-hand side, one pass over the history
        const long newest = (slot + m - 1) % m;
        const ComplexType *pdG = dG.data(), *pdF = dF.data(), *dgn = dG.data() + newest * n_el, *gk = pGp, *fk = pFp;
        std::vector<double> dots;                        // [0, n_hist): Re <dG_i, dG_newest>; [n_hist, 2 n_hist): Re <dG_i, g>
        blocked_sums(n_el, 2 * n_hist, nth, dots, [&](long e, double *acc) {
          for (long i = 0; i < n_hist; ++i) {
            const ComplexType ci = std::conj(pdG[i * n_el + e]);
            acc[i] += std::real(ci * dgn[e]);
            acc[n_hist + i] += std::real(ci * gk[e]);
          }
        });
        for (long i = 0; i < n_hist; ++i) H(newest, i) = H(i, newest) = dots[size_t(i)];
        nda::matrix<double> A(n_hist, n_hist);
        double tr = 0.0;
        for (long i = 0; i < n_hist; ++i) tr += H(i, i);
        for (long i = 0; i < n_hist; ++i) {
          for (long j = 0; j < n_hist; ++j) A(i, j) = H(i, j);
          A(i, i) += _o.aa_reg * tr;
        }
        nda::matrix<double> Ainv = nda::inverse(A);
        std::vector<double> gam(size_t(n_hist), 0.0);
        for (long i = 0; i < n_hist; ++i)
          for (long j = 0; j < n_hist; ++j) gam[size_t(i)] += Ainv(i, j) * dots[size_t(n_hist + j)];
#pragma omp parallel for schedule(static) num_threads(nth)
        for (long e = 0; e < n_el; ++e) {
          ComplexType v = fk[e];
          for (long i = 0; i < n_hist; ++i) v -= gam[size_t(i)] * pdF[i * n_el + e];
          pV[e] = v;
        }
      }
      st.iters = it;
      if (not st.converged) { copy(pC, pV, n_el, nth); psd_rows(C, nb); }    // the weights of the last iterate
      C2 = C;
      U2 = V - C;
      gemm_cols(_Kt, C2, M);
      nda::matrix<ComplexType> D = Y - M;
      st.removed = frob(D);
      st.lam_after = lam_min(M, nb);
      for (long t = 0; t < nt; ++t)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) S(t, i, j) = -M(t, i * nb + j);
      return st;
    }

    /** whether this build has the device path (project_device) */
    static constexpr bool device_built() {
#if defined(ENABLE_CUDA)
      return true;
#else
      return false;
#endif
    }

#if defined(ENABLE_CUDA)
    // Project several (s, k) blocks on the device: S[i] (nt, nb, nb) in place, C2[i] / U2[i] the warm-start state of block i
    // (as in project(); empty on a first call). The same solver, batched over the blocks (causal_proj_cuda.cuh). Aborts when
    // the device path cannot run (no device, Anderson memory beyond its limit, no memory): there is no host fallback here.
    std::vector<stats> project_device(std::vector<nda::array<ComplexType, 3> *> const &S, std::vector<nda::matrix<ComplexType> *> const &C2,
                                      std::vector<nda::matrix<ComplexType> *> const &U2, causal_proj_cuda::call_info *info = nullptr) const {
      const long nblk = long(S.size()), L = n_nodes(), nt = _Kt.shape(0);
      utils::check(long(C2.size()) == nblk and long(U2.size()) == nblk, "causal_proj::project_device: {} blocks, {} / {} states.",
                   nblk, C2.size(), U2.size());
      std::vector<stats> out(static_cast<size_t>(nblk));
      if (nblk == 0) return out;
      const long nb = S[0]->shape(1), nb2 = nb * nb;
      for (long i = 0; i < nblk; ++i) {
        auto &s = *S[size_t(i)];
        utils::check(s.shape(0) == nt and s.shape(1) == nb and s.shape(2) == nb,
                     "causal_proj::project_device: block {} is ({} x {} x {}), expected ({} x {} x {}).", i, s.shape(0), s.shape(1),
                     s.shape(2), nt, nb, nb);
        auto &c = *C2[size_t(i)];
        auto &u = *U2[size_t(i)];
        if (c.shape(0) != L or c.shape(1) != nb2) {
          c = nda::matrix<ComplexType>(L, nb2); c() = 0.0;
          u = nda::matrix<ComplexType>(L, nb2); u() = 0.0;
        }
      }
      if (not _dev or _dev_nb != nb) {
        // Kt and Minv are real (Kt from the kernel, Minv the inverse of the real Kt^T Kt + rho): the device runs real x complex
        nda::array<double, 2> Kt(nt, L), KtT(L, nt), Minv(L, L);
        double imax = 0.0;
        for (long t = 0; t < nt; ++t)
          for (long l = 0; l < L; ++l) { Kt(t, l) = std::real(_Kt(t, l)); KtT(l, t) = Kt(t, l); imax = std::max(imax, std::abs(std::imag(_Kt(t, l)))); }
        for (long i = 0; i < L; ++i)
          for (long j = 0; j < L; ++j) { Minv(i, j) = std::real(_Minv(i, j)); imax = std::max(imax, std::abs(std::imag(_Minv(i, j)))); }
        utils::check(imax == 0.0, "causal_proj::project_device: Kt / Minv have imaginary parts ({:.1e}); the device path assumes them real.", imax);
        causal_proj_cuda::config c;
        c.nt = nt; c.L = L; c.nb = nb; c.maxit = _o.maxit; c.tol = _o.tol; c.rho = _rho; c.relax = _o.relax;
        c.aa_restart = _o.aa_restart; c.aa_reg = _o.aa_reg; c.aa_mem = std::max(0l, _o.aa_mem); c.jacobi_tol = _o.jacobi_tol; c.profile = _o.device_profile;
        char why[512] = {0};
        auto *e = causal_proj_cuda::create(c, Kt.data(), KtT.data(), Minv.data(), why, 512);
        utils::check(e != nullptr, "causal_proj::project_device: the device causal projection cannot run: {}. Select the host path "
                     "with vertex_debug scf_causal_proj_device = 0.", std::string(why));
        _dev = std::shared_ptr<causal_proj_cuda::engine>(e, [](causal_proj_cuda::engine *p) { causal_proj_cuda::destroy(p); });
        _dev_nb = nb;
      }
      std::vector<ComplexType *> pS(static_cast<size_t>(nblk)), pC(static_cast<size_t>(nblk)), pU(static_cast<size_t>(nblk));
      for (long i = 0; i < nblk; ++i) {
        pS[size_t(i)] = S[size_t(i)]->data();
        pC[size_t(i)] = C2[size_t(i)]->data();
        pU[size_t(i)] = U2[size_t(i)]->data();
      }
      std::vector<causal_proj_cuda::block_stats> bs(static_cast<size_t>(nblk));
      causal_proj_cuda::project(_dev.get(), nblk, pS.data(), pC.data(), pU.data(), bs.data(), info);
      for (long i = 0; i < nblk; ++i) {
        auto const &b = bs[size_t(i)];
        auto &s = out[size_t(i)];
        s.y_norm = b.y_norm; s.removed = b.removed; s.lam_before = b.lam_before; s.lam_after = b.lam_after;
        s.iters = b.iters; s.converged = b.converged != 0; s.fit_res = b.fit_res; s.aa_restarts = b.aa_restarts;
      }
      return out;
    }
#endif

   private:
    options _o;
    nda::array<double, 1> _w;
    nda::matrix<ComplexType> _Kt, _KtT, _Minv;
    double _rho = 0.0;
#if defined(ENABLE_CUDA)
    mutable std::shared_ptr<causal_proj_cuda::engine> _dev;   // the device operators and handles (built on first use)
    mutable long _dev_nb = -1;
#endif

    // Y = A X with the columns of X split over the OpenMP threads (one single-threaded gemm per column block)
    static void gemm_cols(nda::matrix<ComplexType> const &A, nda::matrix<ComplexType> const &X, nda::matrix<ComplexType> &Y) {
      const long n = X.shape(1);
#pragma omp parallel num_threads(utils::omp_threads_for(n))
      {
#ifdef _OPENMP
        const long nth = omp_get_num_threads(), th = omp_get_thread_num();
#else
        const long nth = 1, th = 0;
#endif
        const long c0 = n * th / nth, c1 = n * (th + 1) / nth;
        if (c1 > c0) {
          auto Yv = Y(nda::range::all, nda::range(c0, c1));
          nda::blas::gemm(ComplexType(1.0), A, X(nda::range::all, nda::range(c0, c1)), ComplexType(0.0), Yv);
        }
      }
    }
    // k sums over e in [0, n): term(e, acc) adds element e's contributions to acc[0 .. k). The range is cut into a FIXED
    // number of blocks, summed block by block and then in block order, so the result does not depend on the thread count
    // (the Anderson coefficients are built from these sums; a thread-dependent summation order would make the projection
    // differ between omp_threads settings beyond roundoff after many iterations).
    template <typename Term>
    static void blocked_sums(long n, long k, [[maybe_unused]] long nth, std::vector<double> &out, Term &&term) {
      constexpr long n_blk = 64;
      const long nblk = std::max(1l, std::min(n_blk, n));
      std::vector<double> part(size_t(nblk * k), 0.0);
#pragma omp parallel for schedule(static) num_threads(std::min(nth, nblk))
      for (long b = 0; b < nblk; ++b) {
        double *acc = part.data() + b * k;
        for (long e = n * b / nblk; e < n * (b + 1) / nblk; ++e) term(e, acc);
      }
      out.assign(size_t(k), 0.0);
      for (long b = 0; b < nblk; ++b)
        for (long i = 0; i < k; ++i) out[size_t(i)] += part[size_t(b * k + i)];
    }
    // the kernels below work on raw pointers taken once per call: no nda assignment may reallocate those buffers
    static void copy(ComplexType *dst, ComplexType const *src, long n, [[maybe_unused]] long nth) {
#pragma omp parallel for schedule(static) num_threads(nth)
      for (long e = 0; e < n; ++e) dst[e] = src[e];
    }
    static double frob(nda::matrix<ComplexType> const &A) {
      double s = 0.0;
      for (auto const &v : A) s += std::norm(v);
      return std::sqrt(s);
    }
    // min over the rows (tau) of lambda_min of the Hermitian nb x nb matrix stored in the row
    static double lam_min(nda::matrix<ComplexType> const &Y, long nb) {
      double lm = 1e300;
      nda::matrix<ComplexType> A(nb, nb);
      for (long t = 0; t < Y.shape(0); ++t) {
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) A(i, j) = 0.5 * (Y(t, i * nb + j) + std::conj(Y(t, j * nb + i)));
        auto ev = nda::linalg::eigenvalues(A);
        for (auto const &v : ev) lm = std::min(lm, double(std::real(v)));
      }
      return lm;
    }
   public:
    // every row of X, read as a Hermitian nb x nb matrix, replaced by its PSD part (negative eigenvalues clipped). A row whose
    // matrix passes a semidefinite Cholesky test (no eigenvalue below -1e-13 of its largest diagonal) is kept as it is, without
    // the eigendecomposition; the clipped ones are rebuilt as W W^dag, W = V sqrt(max(lambda, 0)), by one gemm.
    static void psd_rows(nda::matrix<ComplexType> &X, long nb) {
      const long L = X.shape(0);
#pragma omp parallel for schedule(dynamic) num_threads(utils::omp_threads_for(L))
      for (long l = 0; l < L; ++l) {
        nda::matrix<ComplexType> A(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) A(i, j) = 0.5 * (X(l, i * nb + j) + std::conj(X(l, j * nb + i)));
        if (semidefinite(A)) {
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) X(l, i * nb + j) = A(i, j);
          continue;
        }
        auto [lam, V] = nda::linalg::eigenelements(A);
        nda::matrix<ComplexType> W(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long a = 0; a < nb; ++a) W(i, a) = V(i, a) * std::sqrt(std::max(double(lam(a)), 0.0));
        nda::blas::gemm(ComplexType(1.0), W, nda::dagger(W), ComplexType(0.0), A);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) X(l, i * nb + j) = A(i, j);
      }
    }

   private:
    // semidefinite Cholesky test of a Hermitian matrix with tolerance tau = 1e-13 max_i A_ii (no LAPACK call: n^3 / 6 complex
    // operations on a copy): a pivot below -tau, or a pivot within tau whose column below does not vanish, means indefinite
    static bool semidefinite(nda::matrix<ComplexType> const &A) {
      const long n = A.shape(0);
      double dmax = 0.0;
      for (long j = 0; j < n; ++j) dmax = std::max(dmax, std::real(A(j, j)));
      if (not(dmax > 0.0)) return false;
      const double tau = 1e-13 * dmax, vtol = std::sqrt(tau * dmax);
      nda::matrix<ComplexType> Lc(n, n);
      Lc() = 0.0;
      for (long j = 0; j < n; ++j) {
        double d = std::real(A(j, j));
        for (long k = 0; k < j; ++k) d -= std::norm(Lc(j, k));
        if (d < -tau) return false;
        if (d <= tau) {                                   // (numerically) zero pivot: the rest of the column must vanish
          for (long i = j + 1; i < n; ++i) {
            ComplexType v = A(i, j);
            for (long k = 0; k < j; ++k) v -= Lc(i, k) * std::conj(Lc(j, k));
            if (std::abs(v) > vtol) return false;
          }
          continue;                                       // L(:, j) = 0
        }
        d = std::sqrt(d);
        Lc(j, j) = d;
        for (long i = j + 1; i < n; ++i) {
          ComplexType v = A(i, j);
          for (long k = 0; k < j; ++k) v -= Lc(i, k) * std::conj(Lc(j, k));
          Lc(i, j) = v / d;
        }
      }
      return true;
    }
  };

} // causal_proj
} // methods

#endif // COQUI_SIGMA_CAUSAL_PROJECTION_HPP
