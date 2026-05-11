/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Pade analytic continuation: fit a complex function W_c(iω') on a set of
 * Matsubara-axis nodes z_i = i*ω'_i and evaluate at arbitrary complex
 * z = Ω + i*η (typically Ω real, η small).
 *
 * Standard Thiele recursive Pade: given N nodes (z_i, f_i), constructs an
 * (N-1)/2-order continued fraction
 *
 *     P(z) = a_0 / (1 + a_1 (z - z_0) / (1 + a_2 (z - z_1) / (1 + ...)))
 *
 * that interpolates exactly through all (z_i, f_i). The a_i are obtained
 * by the inverse-difference recursion (Vidberg & Serene 1977).
 *
 * Use case: W_c_aux(iω') is computed by the imag-axis machinery on a
 * Matsubara mesh; for the QP-form QSGW residue sum we need W_c(Ω real)
 * evaluated at Ω = ε_n^QP - ω for many (n, ω) pairs. Pade extrapolation
 * from the imag axis is the cheapest route, accurate well below the
 * Matsubara cutoff for smooth W_c.
 *
 * Caveats (per Vidberg & Serene + later literature):
 *  - Pade is numerically unstable when W_c has sharp poles near the real
 *    axis (e.g., plasmons). Strategy: fall back to a real-axis Dyson
 *    solve at the specific Ω when Pade evaluation produces |P(Ω)| > some
 *    threshold or NaN.
 *  - Recommend using N_iω ≥ 16-32 Matsubara nodes and pruning very-small
 *    |ω'| ones to avoid numerical instability at the inverse difference
 *    near z = 0.
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_QP_PADE_HPP
#define COQUI_REAL_AXIS_QP_PADE_HPP

#include <complex>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis_qp {

using ComplexType = std::complex<double>;

/**
 * Compute Thiele Pade coefficients g_i from inverse differences of the
 * input (z_i, f_i). The coefficients can be passed to pade_eval below.
 *
 * @param z         Nodes (complex, typically iω' on the Matsubara mesh).
 * @param f         Values at z (complex).
 * @return          (g_i) of length z.size(); g_0 = f_0, others from
 *                  recursive inverse differences.
 */
inline std::vector<ComplexType>
pade_coefficients(std::vector<ComplexType> const& z,
                  std::vector<ComplexType> const& f)
{
  const std::size_t N = z.size();
  utils::check(z.size() == f.size() and N >= 1,
               "pade_coefficients: empty or mismatched (z, f).");

  // a[i][j] = i-th inverse difference at the j-th level.
  // We need a tabular Neville-like recursion in place of one column.
  std::vector<std::vector<ComplexType>> tab(N, std::vector<ComplexType>(N));
  for (std::size_t i = 0; i < N; ++i) tab[i][0] = f[i];
  for (std::size_t j = 1; j < N; ++j) {
    for (std::size_t i = j; i < N; ++i) {
      ComplexType num   = tab[j-1][j-1] - tab[i][j-1];
      ComplexType denom = (z[i] - z[j-1]) * tab[i][j-1];
      // If denom is exactly zero we cannot continue; defensive guard.
      if (std::abs(denom) < 1e-300) {
        utils::check(false, "pade_coefficients: zero denominator at i={}, j={}; "
                     "duplicate nodes or pathological data?", i, j);
      }
      tab[i][j] = num / denom;
    }
  }
  std::vector<ComplexType> g(N);
  for (std::size_t i = 0; i < N; ++i) g[i] = tab[i][i];
  return g;
}

/**
 * Evaluate the Pade continued fraction at a target z (complex).
 *
 * @param z_nodes   Nodes z_i used to compute coefficients.
 * @param g         Coefficients from pade_coefficients(z_nodes, f).
 * @param z         Evaluation point (complex).
 * @return          P(z) (complex).
 */
inline ComplexType
pade_eval(std::vector<ComplexType> const& z_nodes,
          std::vector<ComplexType> const& g,
          ComplexType z)
{
  const std::size_t N = z_nodes.size();
  utils::check(g.size() == N and N >= 1, "pade_eval: bad sizes.");
  // Backwards recursion of the continued fraction:
  //   A_{N-1} = 1
  //   A_{k}   = 1 + g_{k+1} (z - z_k) / A_{k+1}     for k = N-2..0
  //   P(z)    = g_0 / A_0
  ComplexType A = ComplexType(1.0, 0.0);
  for (std::size_t k = N - 1; k > 0; --k) {
    A = ComplexType(1.0, 0.0) + g[k] * (z - z_nodes[k-1]) / A;
  }
  return g[0] / A;
}

/**
 * Batched Pade for a tensor of W_c values. Each (P, Q) channel has its
 * own coefficient set; evaluation at a single target z returns a (NP, NQ)
 * complex matrix.
 *
 * Input layout: W_c_iw has shape (NP, NQ, N_iω) complex, treating the
 * leading 2 indices as channels.
 *
 * @param z_nodes        common Matsubara nodes (size N_iω).
 * @param W_c_iw         (NP, NQ, N_iω) complex; per-channel input.
 * @param z              evaluation point.
 * @param out            (NP, NQ) complex; written with W_c(z) per channel.
 */
inline void pade_eval_batched(
    std::vector<ComplexType> const& z_nodes,
    nda::array<ComplexType, 3> const& W_c_iw,
    ComplexType z,
    nda::array<ComplexType, 2>& out)
{
  const long NP = W_c_iw.shape()[0];
  const long NQ = W_c_iw.shape()[1];
  const long N_iw = W_c_iw.shape()[2];
  utils::check(static_cast<std::size_t>(N_iw) == z_nodes.size(),
               "pade_eval_batched: N_iw mismatch.");
  utils::check(out.shape()[0] == NP and out.shape()[1] == NQ,
               "pade_eval_batched: out shape mismatch.");

  // Per-channel coefficient extraction is the expensive piece;
  // amortize it over many evaluations by caching outside this routine.
  std::vector<ComplexType> f(N_iw);
  for (long P = 0; P < NP; ++P) {
    for (long Q = 0; Q < NQ; ++Q) {
      for (long i = 0; i < N_iw; ++i) f[i] = W_c_iw(P, Q, i);
      auto g = pade_coefficients(z_nodes, f);
      out(P, Q) = pade_eval(z_nodes, g, z);
    }
  }
}

} // namespace real_axis_qp
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_PADE_HPP
