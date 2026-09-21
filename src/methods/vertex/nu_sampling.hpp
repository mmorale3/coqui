#ifndef COQUI_VERTEX_NU_SAMPLING_HPP
#define COQUI_VERTEX_NU_SAMPLING_HPP

/**
 * The sampled-nu service (vertex_perf_plan.md P14, 2026-09-21): ONE set of routines for the reconstruction of a
 * bosonic-node-resolved object H(m, col) (m over the nw nodes, col over everything else) from its values at a SUBSET S
 * of the nodes, used by the P side (the L-3 on-demand fit of the all-nu dump, scr_coulomb_t.cpp) and by the Sigma side
 * (the nu-sampled dynamic Sigma vertex, vertex_dynbse.icc). Everything is expressed through the Gram matrix over the
 * nodes of a REFERENCE object, G = H_ref H_ref^dag (nw x nw), whose eigenvectors are the nu-modes of that object.
 *
 *   reconstruction(G, S, K, mode) -> R (nw x |S|) with H(m) ~ sum_s R(m, s) H(S_s):
 *     "modes":      R = Phi (Phi_S^dag Phi_S)^-1 Phi_S^dag, Phi = the top-K eigenvectors of G (the L-3 form: exact at
 *                   the sampled nodes when K = |S|; least squares on the K modes otherwise);
 *     "regression": R = G(:, S) [G(S, S)]^-1_K, the least-squares predictor of all nodes from the sampled ones (Nystrom
 *                   form; G(S, S)^-1 through its top-K eigenpairs). The two agree for K well below |S|; the regression
 *                   form keeps improving up to K ~ |S| - 2 where the modes form degrades (measured on the LiH Sigma
 *                   objects: K 6 5.0e-2, K 9 3.5e-3, K 10 3.3e-3 regression vs 9.2e-3 modes).
 *   modes_to(G, tol)         -> the number of leading modes that carry 1 - tol of the trace (a smoothness meter and
 *                               the automatic rank: K_auto = max(1, min(modes_to(G, tol), |S| - 2))).
 *   pivot_nodes(G, K, forced, nw) -> a sample: the row pivots of the top-K mode matrix (QR with column pivoting on
 *                               Phi^T, i.e. the nodes that best condition the K x K system), with the forced nodes
 *                               (nu = 0, the tail) kept in front.
 */

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace nusamp {

  using cplx = ComplexType;

  /** the Hermitized Gram matrix over the first index: G = H2 H2^dag */
  inline nda::array<cplx, 2> gram(nda::array<cplx, 2> const &H2) {
    const long nw = H2.shape(0);
    nda::array<cplx, 2> G(nw, nw);
    nda::blas::gemm(H2, nda::dagger(H2), G);
    for (long a = 0; a < nw; ++a)
      for (long b = 0; b < a; ++b) { G(a, b) = 0.5 * (G(a, b) + std::conj(G(b, a))); G(b, a) = std::conj(G(a, b)); }
    return G;
  }

  /** the eigenpairs of a Hermitian G in DESCENDING order: lam (nw), V columns */
  inline std::pair<nda::array<double, 1>, nda::array<cplx, 2>> modes(nda::array<cplx, 2> const &G) {
    const long nw = G.shape(0);
    nda::matrix<cplx> Gm(nw, nw);
    Gm() = G;
    for (long a = 0; a < nw; ++a)
      for (long b = 0; b < a; ++b) { Gm(a, b) = 0.5 * (Gm(a, b) + std::conj(Gm(b, a))); Gm(b, a) = std::conj(Gm(a, b)); }
    auto [lam, V] = nda::linalg::eigenelements(Gm);     // ascending
    nda::array<double, 1> l(nw);
    nda::array<cplx, 2> Vd(nw, nw);
    for (long a = 0; a < nw; ++a) {
      l(a) = std::max(double(lam(nw - 1 - a)), 0.0);
      for (long m = 0; m < nw; ++m) Vd(m, a) = V(m, nw - 1 - a);
    }
    return {std::move(l), std::move(Vd)};
  }

  /** the number of leading modes carrying 1 - tol of the trace (0 if the trace is 0); kept: the weight of the first K */
  inline long modes_to(nda::array<cplx, 2> const &G, double tol, long K = 0, double *kept = nullptr) {
    auto [lam, V] = modes(G);
    const long nw = G.shape(0);
    double tr = 0.0;
    for (long a = 0; a < nw; ++a) tr += lam(a);
    if (kept) *kept = 0.0;
    if (tr <= 0.0) return 0;
    double cum = 0.0;
    long n = 0;
    for (long a = 0; a < nw; ++a) {
      cum += lam(a) / tr;
      if (kept and a < K) *kept += lam(a) / tr;
      if (n == 0 and 1.0 - cum < tol) n = a + 1;
    }
    return (n == 0) ? nw : n;
  }

  /** R (nw x |S|): the reconstruction of all nodes from the sampled ones, see the header */
  inline nda::array<cplx, 2> reconstruction(nda::array<cplx, 2> const &G, std::vector<long> const &S, long K, std::string const &mode) {
    const long nw = G.shape(0), nS = long(S.size());
    utils::check(nS > 0 and nS <= nw, "nusamp::reconstruction: {} sampled nodes of {}.", nS, nw);
    for (long s : S) utils::check(s >= 0 and s < nw, "nusamp::reconstruction: sampled node {} outside [0, {}).", s, nw);
    K = std::max(1L, std::min(K, nS));
    nda::array<cplx, 2> R(nw, nS);
    R() = cplx(0.0);
    if (mode == "modes") {
      auto [lam, V] = modes(G);
      nda::array<cplx, 2> Phi(nw, K), PhiS(nS, K), tmp(nw, K);
      nda::matrix<cplx> N(K, K);
      for (long k = 0; k < K; ++k) {
        for (long m = 0; m < nw; ++m) Phi(m, k) = V(m, k);
        for (long s = 0; s < nS; ++s) PhiS(s, k) = V(S[size_t(s)], k);
      }
      nda::blas::gemm(nda::dagger(PhiS), PhiS, N);
      nda::inverse_in_place(N);
      nda::blas::gemm(Phi, N, tmp);
      nda::blas::gemm(tmp, nda::dagger(PhiS), R);                        // Phi (Phi_S^dag Phi_S)^-1 Phi_S^dag
    } else if (mode == "regression") {
      nda::array<cplx, 2> Gss(nS, nS), GcS(nw, nS), Ginv(nS, nS);
      for (long s = 0; s < nS; ++s)
        for (long s2 = 0; s2 < nS; ++s2) Gss(s, s2) = G(S[size_t(s)], S[size_t(s2)]);
      auto [lam, V] = modes(Gss);                                        // descending
      Ginv() = cplx(0.0);
      for (long a = 0; a < K; ++a) {
        if (lam(a) <= 1e-14 * lam(0)) break;
        for (long s = 0; s < nS; ++s)
          for (long s2 = 0; s2 < nS; ++s2) Ginv(s, s2) += V(s, a) * std::conj(V(s2, a)) / lam(a);
      }
      for (long m = 0; m < nw; ++m)
        for (long s = 0; s < nS; ++s) GcS(m, s) = G(m, S[size_t(s)]);
      nda::blas::gemm(GcS, Ginv, R);                                      // G(:, S) G(S, S)^-1_K
    } else {
      utils::check(false, "nusamp::reconstruction: unknown mode \"{}\" (modes | regression).", mode);
    }
    return R;
  }

  /** the sampled nodes: the forced ones first, then the row pivots of the top-K mode matrix (column-pivoted QR of Phi^T
   *  restricted to the free rows) until K nodes in total */
  inline std::vector<long> pivot_nodes(nda::array<cplx, 2> const &G, long K, std::vector<long> const &forced) {
    const long nw = G.shape(0);
    std::vector<long> S;
    for (long f : forced) if (std::find(S.begin(), S.end(), f) == S.end()) S.push_back(f);
    K = std::max(K, long(S.size()));
    K = std::min(K, nw);
    auto [lam, V] = modes(G);
    const long nmode = K;                                                 // pivots of the first K modes
    // greedy column-pivoted QR on the rows of Phi (nw x nmode): pick the row with the largest residual norm, orthogonalize the rest
    nda::array<cplx, 2> Phi(nw, nmode);
    for (long m = 0; m < nw; ++m)
      for (long k = 0; k < nmode; ++k) Phi(m, k) = V(m, k);
    std::vector<char> taken(size_t(nw), 0);
    for (long s : S) taken[size_t(s)] = 1;
    auto orth = [&](long piv) {   // remove the component along row piv from every other row
      double n2 = 0.0;
      for (long k = 0; k < nmode; ++k) n2 += std::norm(Phi(piv, k));
      if (n2 <= 0.0) return;
      for (long m = 0; m < nw; ++m) {
        if (m == piv) continue;
        cplx d(0.0);
        for (long k = 0; k < nmode; ++k) d += std::conj(Phi(piv, k)) * Phi(m, k);
        d /= n2;
        for (long k = 0; k < nmode; ++k) Phi(m, k) -= d * Phi(piv, k);
      }
      for (long k = 0; k < nmode; ++k) Phi(piv, k) = cplx(0.0);
    };
    for (long s : S) orth(s);
    while (long(S.size()) < K) {
      long best = -1;
      double bn = -1.0;
      for (long m = 0; m < nw; ++m) {
        if (taken[size_t(m)]) continue;
        double n2 = 0.0;
        for (long k = 0; k < nmode; ++k) n2 += std::norm(Phi(m, k));
        if (n2 > bn) { bn = n2; best = m; }
      }
      if (best < 0 or bn <= 0.0) break;
      S.push_back(best);
      taken[size_t(best)] = 1;
      orth(best);
    }
    std::sort(S.begin(), S.end());
    return S;
  }

} // namespace nusamp
} // namespace methods

#endif // COQUI_VERTEX_NU_SAMPLING_HPP
