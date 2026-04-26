/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_AXIS_SIGMA_HPP
#define COQUI_REAL_AXIS_REAL_AXIS_SIGMA_HPP

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"

namespace methods {
namespace real_axis {

/**
 * Compute Im Sigma^{c,R}(k, w) -- imaginary part of the retarded correlation
 * self-energy in the THC auxiliary basis -- from the projected fermionic
 * spectral function A_{P}(k-q, w) on the FERMIONIC w-grid and the bosonic
 * spectral function B_{PQ}(q, Omega) of the screened interaction on the
 * BOSONIC Omega-grid.
 *
 * Implements the spectral form (notes Eq. ImSigma_spectral_isdf), specialized
 * to the auxiliary basis kernel:
 *
 *   Im Sigma^{c,R}_{P}(k-q, w) = -pi * sum_Q  Z_factor *
 *       int de int dO  A_Q(k-q, e) B_{PQ}(q, O) [f(e) + n_B(O)] delta(w-e-O)
 *
 * Restricted to the bosonic half-grid Omega>=0 we use B(-Omega) = -B(Omega)
 * (diagonal) to fold the integral; equivalently, separate into emission and
 * absorption channels:
 *
 *   Im Sigma^{c,R}(w) = -pi * sum_Q  [
 *           ( A_Q(k-q, .) star B_{PQ}^+(q, .) )(w; weight = f+n_B)
 *           - ( A_Q(k-q, .) star B_{PQ}^+(q, .) )(w; ...)  swapped
 *       ]
 *
 * For the standalone/test interface we expose a single-(k, q) accumulator
 * that takes the projected spectra at (k-q) and the bosonic spectral function
 * at (q), and accumulates the contribution into Im Sigma at this k.
 *
 * NOTE: This is the auxiliary-basis Im Sigma^c contribution; the final
 * ORBITAL-BASIS self-energy comes from the contraction with Z_{(mu alpha)P}
 * Z*_{(beta nu)Q}. That contraction lives in the higher-level driver code.
 *
 * @param conv         NUFFT engine
 * @param A_Q_kq       (Naux, N_w) projected spectral function at k-q
 * @param B_PQ_q       (Naux, Naux, N_Omega) bosonic auxiliary spectral function
 *                     of W^c at this q. Stored on Omega>=0 half-grid.
 * @param ImSigma_P_w  OUTPUT (Naux, N_w) Im Sigma^c contribution, accumulated.
 * @param q_weight     weight of this q point (1/Nq, IBZ stars, etc.)
 *
 * Internally this packs Naux*Naux batched cross-correlations.
 */
inline void accumulate_ImSigma_one_kq(real_axis_conv_t & conv,
                                      nda::array<ComplexType, 2> const& A_Q_kq,
                                      nda::array<ComplexType, 3> const& B_PQ_q,
                                      nda::array<ComplexType, 2>      & ImSigma_P_w,
                                      double q_weight)
{
  const long Naux = A_Q_kq.shape()[0];
  const long N_w = A_Q_kq.shape()[1];
  const long N_O = B_PQ_q.shape()[2];

  utils::check(B_PQ_q.shape()[0] == Naux and B_PQ_q.shape()[1] == Naux,
               "accumulate_ImSigma_one_kq: B shape mismatch");
  utils::check(ImSigma_P_w.shape()[0] == Naux and ImSigma_P_w.shape()[1] == N_w,
               "accumulate_ImSigma_one_kq: ImSigma shape mismatch");
  utils::check(N_w == conv.N_w() and N_O == conv.N_Omega(),
               "accumulate_ImSigma_one_kq: grid mismatch");

  auto const& grid = conv.grid();

  // Build kernel-weighted spectra.
  // Channel 1 (emission, Omega > 0):  weight = f(e) + n_B(O)
  //   delta(w - e - O) means w = e + O, so e = w - O ranges over fermionic w.
  // The cross-correlation engine performs
  //   H(w) = int de F^*(e) G(e + w)
  // We use src=fermionic, dst=fermionic.
  //
  // Identification with the self-energy formula (notes ImSigma_spectral_isdf):
  //   Im Sigma_P(w) = -pi sum_Q int de int dO A_Q(e) B_{PQ}(O) [f(e)+n_B(O)] delta(w-e-O)
  // Setting O = w - e:
  //   = -pi sum_Q int de A_Q(e) [f(e) + n_B(w-e)] B_{PQ}(w-e)  on Omega>=0 (w>=e)
  //   + (channel for w < e via Omega -> -Omega, with B odd: B(-O) = -B(O))
  //   = -pi sum_Q int de A_Q(e) [f(e) + n_B(w-e)] B_{PQ}(w-e) Theta(w-e)
  //   - pi sum_Q int de A_Q(e) [f(e) - n_B(e-w)-1] B_{PQ}(e-w) Theta(e-w)
  //
  // To express both as cross-correlations of fermionic-grid spectra, define
  //   F1(e) = A_Q(e) f(e),    F2(e) = A_Q(e) [1 - f(e)]
  // and the bosonic kernel B(Omega) extended to Omega<0 via odd extension.
  //
  // For a pragmatic single-pass implementation we INTERPOLATE B from the
  // bosonic grid onto the fermionic grid (since Omega = w - e ranges over
  // [-2 w_max, 2 w_max], wider than the fermionic window). Linear interp.
  // Then the contribution becomes a fermionic-fermionic cross-correlation.
  //
  // This linear interpolation is acceptable because B is smooth where it is
  // appreciable, the auxiliary spectral function has compact support away
  // from |Omega| > Omega_max where it tends to zero by tail subtraction.

  // Helper: linear interpolation of B on the bosonic grid (Omega>=0) with odd
  // extension for Omega<0. Returns 0 outside the grid.
  auto B_at = [&](long P, long Q, double O) -> ComplexType {
    auto const& Og = grid.Omega();
    const long N = Og.shape()[0];
    const double sign = (O >= 0.0 ? 1.0 : -1.0);
    const double Oa = std::abs(O);
    if (Oa <= Og(0)) {
      // linear extrapolation toward 0; B is odd, B(0)=0.
      return sign * B_PQ_q(P, Q, 0) * (Oa / Og(0));
    }
    if (Oa >= Og(N-1)) return ComplexType(0.0, 0.0);
    // bisection
    long lo = 0, hi = N - 1;
    while (hi - lo > 1) {
      long mid = (lo + hi) / 2;
      if (Og(mid) <= Oa) lo = mid; else hi = mid;
    }
    const double t = (Oa - Og(lo)) / (Og(hi) - Og(lo));
    const ComplexType v = (1.0 - t) * B_PQ_q(P, Q, lo) + t * B_PQ_q(P, Q, hi);
    return sign * v;
  };

  // Build, per (P, Q), the fermionic-grid quantity
  //   K_{PQ}(w, e) = [f(e) + n_B(w - e)]  but evaluated as a function of e
  //                 with parameter w, multiplied by B_{PQ}(w - e).
  // Express through a cross-correlation:
  //   Im Sigma_P(w) ≈ -pi sum_Q int de A_Q(e) [f(e) + n_B(w-e)] B_{PQ}(w-e)
  // Let G_{PQ}(e) = A_Q(e) f(e), H1_{PQ}(O) = B_{PQ}(O) (for Omega>=0)
  //     extended to Omega<0 via B(-O)=-B(O). Then
  //   ∫ de G(e) [n_B(w-e) + f(e)] B(w-e) =
  //     ∫ de G(e) f(e) B(w-e)  +  ∫ de G(e) n_B(w-e) B(w-e)
  // The first integrand is a function of e with parameter w; substituting
  // O = w-e makes it ∫ dO G(w-O) f(w-O) B(O).
  //
  // We implement directly by sampling on the fermionic grid (since the
  // bosonic kernel can be interpolated onto it).
  //
  // For each (P, Q), evaluate:
  //   For each w_l on fermionic grid:
  //     I1(w_l) += sum_j w_j A_Q(e_j) [f(e_j) + n_B(w_l - e_j)] B_{PQ}(w_l - e_j)
  // This is O(N_w * N_w * Naux^2 * N_kq) which is acceptable for moderate
  // sizes; downstream the cross-correlation kernel can be substituted in.

  auto const& w_grid = grid.w();
  auto const& w_wts  = grid.w_weights();

  // Loop over (P, Q) and w_l.
  for (long P = 0; P < Naux; ++P) {
    for (long l = 0; l < N_w; ++l) {
      const double w_l = w_grid(l);
      ComplexType acc(0.0, 0.0);
      for (long Q = 0; Q < Naux; ++Q) {
        for (long j = 0; j < N_w; ++j) {
          const double e_j = w_grid(j);
          const double O   = w_l - e_j;
          const ComplexType Bv = B_at(P, Q, O);
          if (Bv == ComplexType(0.0, 0.0)) continue;
          const double f_e  = grid.fermi(e_j);
          // n_B at O = w_l - e_j; if O is small in magnitude, the singularity
          // is regularized by the small-|O| linear extrapolation of B which
          // already kills the divergence (B ~ O so B*n_B is finite). For
          // safety, when O is exactly zero we skip the contribution (set to
          // limit).
          double nB_O;
          if (std::abs(O) < 1e-12) {
            // B(O) ~ b1 * O, n_B(O) ~ 1/(beta O), so B*n_B -> b1/beta finite.
            // The product Bv * n_B(O) for B already multiplied by O via the
            // sign*Oa/Og(0) extrapolation: this product is ~ b1*O * 1/(beta O)
            // = b1/beta. We compute it as the limit using one-sided values.
            nB_O = 0.0;  // contribute zero from the n_B branch
          } else {
            nB_O = grid.bose(O);
          }
          const double weight = f_e + nB_O;
          acc += w_wts(j) * A_Q_kq(Q, j) * weight * Bv;
        }
      }
      ImSigma_P_w(P, l) += -M_PI * q_weight * acc;
    }
  }
}

/**
 * Recover Re Sigma^{c,R} from Im Sigma^{c,R} by Hilbert transform on the
 * fermionic grid. Must subtract the m^{(1)}/w high-frequency tail before the
 * transform for accuracy; the basic interface here applies the raw transform
 * and is suitable for testing or for use with sufficiently wide windows.
 */
inline void ReSigma_from_ImSigma(real_axis_conv_t & conv,
                                 nda::array<double, 3> const& ImSigma_skP_w,
                                 nda::array<double, 3>      & ReSigma_skP_w)
{
  const long ns_kP = ImSigma_skP_w.shape()[0];
  const long N_w = ImSigma_skP_w.shape()[1];
  // We treat the leading dimension as a flat batch.
  utils::check(ImSigma_skP_w.shape() == ReSigma_skP_w.shape(),
               "ReSigma_from_ImSigma: shape mismatch");
  utils::check(N_w == conv.N_w(),
               "ReSigma_from_ImSigma: grid mismatch");

  // Flatten to (B, N_w).
  const long B = ns_kP;
  const long N_extra = ImSigma_skP_w.shape()[2];

  nda::array<double, 2> ImBuf(B * N_extra, N_w), ReBuf(B * N_extra, N_w);
  for (long b = 0; b < B; ++b)
    for (long n = 0; n < N_extra; ++n)
      for (long l = 0; l < N_w; ++l)
        ImBuf(b * N_extra + n, l) = ImSigma_skP_w(b, l, n);

  conv.hilbert(ImBuf, ReBuf, real_axis_conv_t::grid_kind::fermionic);

  for (long b = 0; b < B; ++b)
    for (long n = 0; n < N_extra; ++n)
      for (long l = 0; l < N_w; ++l)
        ReSigma_skP_w(b, l, n) = ReBuf(b * N_extra + n, l);
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_SIGMA_HPP
