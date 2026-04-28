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
 * Resample a bosonic auxiliary spectral function B_{PQ}(Omega) (defined on
 * Omega >= 0) onto the fermionic w-grid via the diagonal-odd extension
 *   B_{PQ}(-Omega) = -B_{PQ}(Omega).
 * Linear interpolation between bosonic grid points; linear extrapolation to
 * 0 inside [0, Omega_min); zero beyond Omega_max.
 *
 * @param grid     real-frequency grid
 * @param B_PQ_O   (Naux, Naux, N_Omega)  bosonic spectral function on Omega>=0
 * @param B_PQ_w   OUTPUT (Naux, Naux, N_w) bosonic function resampled on w
 */
template <typename BIn, typename BOut>
inline void resample_bosonic_to_fermionic(real_freq_grid_t const& grid,
                                          BIn  const& B_PQ_O,
                                          BOut      & B_PQ_w)
{
  const long Naux = B_PQ_O.shape()[0];
  const long N_w  = grid.N_w();
  utils::check(B_PQ_O.shape()[1] == Naux,
               "resample_bosonic_to_fermionic: B not square in (P,Q)");
  utils::check(B_PQ_O.shape()[2] == grid.N_Omega(),
               "resample_bosonic_to_fermionic: bosonic length mismatch");
  utils::check(B_PQ_w.shape()[0] == Naux and B_PQ_w.shape()[1] == Naux and
               B_PQ_w.shape()[2] == N_w,
               "resample_bosonic_to_fermionic: output shape mismatch");

  auto const& Og = grid.Omega();
  const long N_O = Og.shape()[0];

  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iw = 0; iw < N_w; ++iw) {
        const double w_l = grid.w()(iw);
        const double sign = (w_l >= 0.0 ? 1.0 : -1.0);
        const double a    = std::abs(w_l);
        ComplexType v(0.0, 0.0);
        if (a <= Og(0)) {
          v = B_PQ_O(P, Q, 0) * (a / Og(0));
        } else if (a < Og(N_O - 1)) {
          long lo = 0, hi = N_O - 1;
          while (hi - lo > 1) {
            long mid = (lo + hi) / 2;
            if (Og(mid) <= a) lo = mid; else hi = mid;
          }
          const double t = (a - Og(lo)) / (Og(hi) - Og(lo));
          v = (1.0 - t) * B_PQ_O(P, Q, lo) + t * B_PQ_O(P, Q, hi);
        }
        B_PQ_w(P, Q, iw) = sign * v;
      }
}

/**
 * NUFFT-accelerated single-(k, q) Im Sigma^{c,R} accumulator.
 * Mathematically equivalent to accumulate_ImSigma_one_kq below; replaces the
 * O(N_w^2) direct quadrature with two O(N_w * log N_t) NUFFT-based
 * convolutions, one per (f, n_B) channel.
 *
 * @param conv          NUFFT engine (ntrans must be >= Naux^2)
 * @param A_PQ_kmq      (Naux, Naux, N_w) projected spectral function at k-q
 * @param B_PQ_q        (Naux, Naux, N_Omega) bosonic spectral function at q
 * @param ImSigma_PQ_w  (Naux, Naux, N_w) accumulated output
 * @param q_weight      weight of this q point
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename AKMQ, typename BQ, typename SOut>
inline void accumulate_ImSigma_one_kq_nufft(
    detail::real_axis_conv_base_t<MEM> & conv,
    AKMQ const& A_PQ_kmq,
    BQ   const& B_PQ_q,
    SOut      & ImSigma_PQ_w,
    double q_weight)
{
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(false,
                 "accumulate_ImSigma_one_kq_nufft<DEVICE>: device kernels "
                 "for the inner f / n_B weighting and Sigma accumulation "
                 "are not yet implemented.");
    return;
  }
  const long Naux = A_PQ_kmq.shape()[0];
  const long N_w  = A_PQ_kmq.shape()[2];
  const long B    = Naux * Naux;

  utils::check(B <= conv.ntrans(),
               "accumulate_ImSigma_one_kq_nufft: ntrans={} too small (need {})",
               conv.ntrans(), B);

  auto const& grid = conv.grid();
  auto const& wq   = grid.w_weights();
  const long N_t   = conv.N_t();

  // Resample B onto the fermionic grid (with diagonal odd extension).
  nda::array<ComplexType, 3> B_PQ_w(Naux, Naux, N_w);
  resample_bosonic_to_fermionic(grid, B_PQ_q, B_PQ_w);

  // Pre-multiply A by f(eps) -> F1, resampled B by n_B(w) -> G2 (the
  // n_B factor handles the singular Omega=0 limit using
  //   lim_{O -> 0} n_B(O) * B(O) = b_1 / beta
  // via the linear extrapolation already applied to B in the resampler:
  // for |w| < Omega_min, B is linear in w. We evaluate n_B(w) for w != 0
  // and set the product to 0 at w=0 (the surrounding integration weights
  // make that point negligible).
  // The trapezoidal quadrature weights are absorbed into F1, F2, G1, G2
  // here so the lower-level NUFFT primitives can run unweighted.
  nda::array<ComplexType, 2> F1(B, N_w), G1(B, N_w);
  nda::array<ComplexType, 2> F2(B, N_w), G2(B, N_w);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long iw = 0; iw < N_w; ++iw) {
        const double w_l = grid.w()(iw);
        const double f_w = grid.fermi(w_l);
        const double q_j = wq(iw);
        F1(b, iw) = q_j * f_w * A_PQ_kmq(P, Q, iw);
        G1(b, iw) = q_j * B_PQ_w(P, Q, iw);
        F2(b, iw) = q_j * A_PQ_kmq(P, Q, iw);
        double nB_w = 0.0;
        if (std::abs(w_l) > 1e-12) nB_w = grid.bose(w_l);
        G2(b, iw) = q_j * nB_w * B_PQ_w(P, Q, iw);
      }
    }

  // Im Sigma ~ convolve(F1, G1) + convolve(F2, G2). Compute each pair's
  // Hhat in time space, sum (no conjugates for convolve), then a single
  // type-2 NUFFT. Saves one type-2 per call vs two convolve calls.
  using gk = grid_kind;
  nda::array<ComplexType, 2> F1hat(B, N_t), G1hat(B, N_t);
  nda::array<ComplexType, 2> F2hat(B, N_t), G2hat(B, N_t);
  conv.forward(F1, F1hat, gk::fermionic);
  conv.forward(G1, G1hat, gk::fermionic);
  conv.forward(F2, F2hat, gk::fermionic);
  conv.forward(G2, G2hat, gk::fermionic);

  // Sigma Hadamard kernel: 4-ary elementwise map (host/device-agnostic).
  memory::array<MEM, ComplexType, 2> Hhat(B, N_t);
  Hhat = nda::map([](ComplexType f1, ComplexType g1,
                     ComplexType f2, ComplexType g2) {
    return f1 * g1 + f2 * g2;
  })(F1hat, G1hat, F2hat, G2hat);

  nda::array<ComplexType, 2> Hraw(B, N_w);
  conv.backward(Hhat, Hraw, gk::fermionic);
  const double s_nufft = conv.nufft_scale();
  const double s_sig   = -M_PI * q_weight * s_nufft;
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long iw = 0; iw < N_w; ++iw)
        ImSigma_PQ_w(P, Q, iw) += s_sig * Hraw(b, iw);
    }
}

/**
 * Compute Im Sigma^{c,R}_{PQ}(k, w) -- imaginary part of the retarded
 * correlation self-energy in the THC auxiliary basis -- for a single (k, q)
 * pair, with HADAMARD product structure in the auxiliary indices (P, Q):
 *
 *   Im Sigma^{c,R}_{PQ}(k, w) -=  pi * q_weight *
 *       int de  A_{PQ}(k-q, e) * B_{PQ}(q, w-e) * [f(e) + n_B(w-e)]
 *
 * This is the auxiliary-basis form of the GW self-energy under the CoQui
 * THC ansatz where the auxiliary index pair (P, Q) is Hadamard-coupled
 * between the projected G and the auxiliary W (Eq. ImSigma_spectral_isdf
 * in the v2 notes, in the basis where the THC factor X is k-diagonal).
 *
 * Direct quadrature implementation. The bosonic spectral function B is
 * defined on Omega >= 0 only; for the convolution argument w-e we use
 * the diagonal-odd extension B_{PQ}(-Omega) = -B_{PQ}(Omega) on the
 * diagonal (P=Q) and the corresponding hermitian relation for off-diagonal.
 * For simplicity we apply the diagonal-odd extension uniformly and rely on
 * downstream cross-validation; the precise off-diagonal symmetry can be
 * refined when fully needed.
 *
 * @param conv          NUFFT engine (carries the grid via conv.grid())
 * @param A_PQ_kmq      (Naux, Naux, N_w) projected spectral function at k-q
 * @param B_PQ_q        (Naux, Naux, N_Omega) bosonic auxiliary spectral
 *                      function of W^c at q (Omega>=0)
 * @param ImSigma_PQ_w  OUTPUT (Naux, Naux, N_w) accumulated Im Sigma at this k
 * @param q_weight      weight of this q point
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename AKMQ, typename BQ, typename SOut>
inline void accumulate_ImSigma_one_kq(detail::real_axis_conv_base_t<MEM> & conv,
                                      AKMQ const& A_PQ_kmq,
                                      BQ   const& B_PQ_q,
                                      SOut      & ImSigma_PQ_w,
                                      double q_weight)
{
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(false,
                 "accumulate_ImSigma_one_kq<DEVICE>: device kernel for the "
                 "direct-quadrature implementation not yet implemented "
                 "(prefer the NUFFT variant on device).");
    return;
  }
  const long Naux = A_PQ_kmq.shape()[0];
  const long N_w  = A_PQ_kmq.shape()[2];
  const long N_O  = B_PQ_q.shape()[2];

  utils::check(A_PQ_kmq.shape()[1] == Naux,
               "accumulate_ImSigma_one_kq: A not square in (P,Q)");
  utils::check(B_PQ_q.shape()[0] == Naux and B_PQ_q.shape()[1] == Naux,
               "accumulate_ImSigma_one_kq: B shape mismatch");
  utils::check(ImSigma_PQ_w.shape()[0] == Naux and
               ImSigma_PQ_w.shape()[1] == Naux and
               ImSigma_PQ_w.shape()[2] == N_w,
               "accumulate_ImSigma_one_kq: ImSigma shape mismatch");
  utils::check(N_w == conv.N_w() and N_O == conv.N_Omega(),
               "accumulate_ImSigma_one_kq: grid mismatch");

  auto const& grid = conv.grid();
  auto const& w_grid = grid.w();
  auto const& w_wts  = grid.w_weights();

  // Helper: linear interpolation of B_{P,Q}(Omega) on the bosonic grid
  // (Omega>=0) extended to Omega<0 via odd-extension on the diagonal and
  // (for off-diagonal) the same odd extension as a working approximation.
  // Returns 0 outside the grid.
  auto B_at = [&](long P, long Q, double O) -> ComplexType {
    auto const& Og = grid.Omega();
    const long N = Og.shape()[0];
    const double sign = (O >= 0.0 ? 1.0 : -1.0);
    const double Oa = std::abs(O);
    if (Oa <= Og(0)) {
      // B(0) = 0 by oddness; linear extrapolation toward the smallest grid pt.
      return sign * B_PQ_q(P, Q, 0) * (Oa / Og(0));
    }
    if (Oa >= Og(N-1)) return ComplexType(0.0, 0.0);
    long lo = 0, hi = N - 1;
    while (hi - lo > 1) {
      long mid = (lo + hi) / 2;
      if (Og(mid) <= Oa) lo = mid; else hi = mid;
    }
    const double t = (Oa - Og(lo)) / (Og(hi) - Og(lo));
    const ComplexType v = (1.0 - t) * B_PQ_q(P, Q, lo) + t * B_PQ_q(P, Q, hi);
    return sign * v;
  };

  // Per (P, Q) scalar convolution. O(Naux^2 * N_w^2) per (k, q) — direct
  // quadrature; the NUFFT-accelerated version becomes available once a
  // generic real-axis convolve primitive (no conjugate) is added.
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long l = 0; l < N_w; ++l) {
        const double w_l = w_grid(l);
        ComplexType acc(0.0, 0.0);
        for (long j = 0; j < N_w; ++j) {
          const double e_j = w_grid(j);
          const double O   = w_l - e_j;
          const ComplexType Bv = B_at(P, Q, O);
          if (Bv == ComplexType(0.0, 0.0)) continue;
          const double f_e = grid.fermi(e_j);
          double nB_O;
          if (std::abs(O) < 1e-12) {
            // n_B * B is finite as Omega->0 (n_B ~ 1/(beta Omega), B ~ b1*Omega).
            // Set n_B contribution to zero here; the f(e) branch carries the rest.
            nB_O = 0.0;
          } else {
            nB_O = grid.bose(O);
          }
          acc += w_wts(j) * A_PQ_kmq(P, Q, j) * Bv * (f_e + nB_O);
        }
        ImSigma_PQ_w(P, Q, l) += -M_PI * q_weight * acc;
      }
}

/**
 * Recover Re Sigma^{c,R} from Im Sigma^{c,R} on the fermionic grid via
 * batched Hilbert transform over the (P, Q) auxiliary indices.
 *
 * @param conv           NUFFT engine
 * @param ImSigma_PQ_w   (Naux, Naux, N_w) input, real part is Im Sigma
 * @param ReSigma_PQ_w   (Naux, Naux, N_w) output, real part is Re Sigma
 *
 * NOTE: The complex-valued I/O is purely a storage convention; physically,
 * Im/Re Sigma are real-valued matrix elements at each (P,Q,w). We carry the
 * Im part through .real() of the input ComplexType and write back into the
 * .real() of the output, leaving the imaginary slot unused.
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename AIn, typename AOut>
inline void ReSigma_from_ImSigma_aux(detail::real_axis_conv_base_t<MEM> & conv,
                                     AIn  const& ImSigma_PQ_w,
                                     AOut      & ReSigma_PQ_w)
{
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(false,
                 "ReSigma_from_ImSigma_aux<DEVICE>: device kernel for the "
                 "(P,Q) <-> batch gather/scatter not yet implemented.");
    return;
  }
  const long Naux = ImSigma_PQ_w.shape()[0];
  const long N_w  = ImSigma_PQ_w.shape()[2];
  const long B = Naux * Naux;

  memory::array<MEM, double, 2> ImBuf(B, N_w), ReBuf(B, N_w);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long l = 0; l < N_w; ++l)
        ImBuf(b, l) = ImSigma_PQ_w(P, Q, l).real();
    }

  conv.hilbert(ImBuf, ReBuf, grid_kind::fermionic);

  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long l = 0; l < N_w; ++l)
        ReSigma_PQ_w(P, Q, l) = ComplexType(ReBuf(b, l), 0.0);
    }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_SIGMA_HPP
