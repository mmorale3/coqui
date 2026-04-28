/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_AXIS_PI_HPP
#define COQUI_REAL_AXIS_REAL_AXIS_PI_HPP

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
 * Compute the imaginary part of the retarded polarization in the THC
 * auxiliary basis for one (q, k) pair, given the projected fermionic spectral
 * function at k and at k+q.
 *
 * Implements the spectral form (notes Eq. ImPi_correlation_form):
 *
 *   Im Pi^R_{PQ}(q, Omega)
 *     = -pi  ( (A_<  star A_>)_{PQ}(k, q; Omega)
 *            - (A_>  star A_<)_{PQ}(k, q; Omega) )
 *
 * where the cross-correlation star is defined by Eq. cross_correlation_def
 * with weighted spectra
 *
 *   A^<_{PQ}(k, w; q) = f(w)        * A_{PQ}(k,   w; q)
 *   A^>_{PQ}(k, w; q) = (1 - f(w))  * A_{PQ}(k+q, w; q)
 *
 * (note: the second leg of the bubble lives at k+q; the caller supplies
 * A_PQ_kq for that leg).
 *
 * The "P,Q" indices here are NOT batched together: the contribution to a
 * single Im Pi(Omega; P, Q) from one (k,q) pair comes from (A_PQ_k * A_QP_kq).
 * The batched call therefore packs B = Naux*Naux pairs along the leading
 * dimension of the cross-correlation, with internal mapping
 *      b = P * Naux + Q  ->  cross-correlation of leg(P,Q) at k vs leg(Q,P) at k+q.
 *
 * @param conv      NUFFT engine (must have ntrans >= Naux*Naux)
 * @param A_PQ_k    projected spectral function at k:    (Naux, Naux, N_w)
 * @param A_PQ_kq   projected spectral function at k+q:  (Naux, Naux, N_w)
 * @param ImPi_PQ_O OUTPUT: Im Pi at this (k,q):         (Naux, Naux, N_Omega)
 *                  ACCUMULATED into (caller zeros first).
 * @param k_weight  weight of this k point in the BZ sum (1/Nk, IBZ stars, etc.)
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename AK, typename AKQ, typename POut>
inline void accumulate_ImPi_one_kq(detail::real_axis_conv_base_t<MEM> & conv,
                                   AK   const& A_PQ_k,
                                   AKQ  const& A_PQ_kq,
                                   POut      & ImPi_PQ_O,
                                   double k_weight,
                                   long P_origin = 0,
                                   long Q_origin = 0)
{
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(false,
                 "accumulate_ImPi_one_kq<DEVICE>: device kernels for the "
                 "weighted A_PQ projection and the auxiliary-index gather "
                 "are not yet implemented.");
    return;
  }
  // A_PQ_k / A_PQ_kq are full (Naux, Naux, N_w) replicated views.
  // ImPi_PQ_O is a local slice of shape (Naux_loc_P, Naux_loc_Q, N_Omega) into
  // which this rank's contribution is accumulated. (P_origin, Q_origin)
  // give the global offsets for the local slice; default 0 reproduces the
  // legacy fully-replicated behavior.
  const long Naux = A_PQ_k.shape()[0];
  const long N_w  = A_PQ_k.shape()[2];
  const long Naux_loc_P = ImPi_PQ_O.shape()[0];
  const long Naux_loc_Q = ImPi_PQ_O.shape()[1];
  const long N_O  = ImPi_PQ_O.shape()[2];

  utils::check(A_PQ_k.shape() == A_PQ_kq.shape(),
               "accumulate_ImPi_one_kq: A_k and A_kq must have same shape");
  utils::check(A_PQ_k.shape()[1] == Naux,
               "accumulate_ImPi_one_kq: A must be square in (P,Q)");
  utils::check(P_origin >= 0 and P_origin + Naux_loc_P <= Naux,
               "accumulate_ImPi_one_kq: P range out of bounds");
  utils::check(Q_origin >= 0 and Q_origin + Naux_loc_Q <= Naux,
               "accumulate_ImPi_one_kq: Q range out of bounds");
  utils::check(N_w == conv.N_w() and N_O == conv.N_Omega(),
               "accumulate_ImPi_one_kq: grid mismatch");

  auto const& grid = conv.grid();
  auto const& w    = grid.w();
  auto const& wq   = grid.w_weights();

  // Build weighted spectra at k and k+q for the bubble.
  // Layout: (B, N_w) with B = Naux_loc_P * Naux_loc_Q. For index b = iP*Naux_loc_Q + iQ:
  //   - F^<_{b}(w) = f(w) A_{P,Q}(k, w)            (left leg, at k, global (P,Q))
  //   - G^>_{b}(w) = (1-f(w)) A_{Q,P}(k+q, w)      (right leg, at k+q, transposed indices)
  // The weighted forms also absorb the trapezoidal quadrature weights, so
  // the lower-level NUFFT primitives don't need to apply them again.
  const long B   = Naux_loc_P * Naux_loc_Q;
  const long N_t = conv.N_t();

  nda::array<ComplexType, 2> Aless_k(B, N_w);
  nda::array<ComplexType, 2> Agtr_k(B, N_w);
  nda::array<ComplexType, 2> Aless_kq(B, N_w);
  nda::array<ComplexType, 2> Agtr_kq(B, N_w);

  for (long iP = 0; iP < Naux_loc_P; ++iP) {
    const long P = P_origin + iP;
    for (long iQ = 0; iQ < Naux_loc_Q; ++iQ) {
      const long Q = Q_origin + iQ;
      const long b = iP * Naux_loc_Q + iQ;
      for (long j = 0; j < N_w; ++j) {
        const double f_w  = grid.fermi(w(j));
        const double fb_w = 1.0 - f_w;
        const double q_j  = wq(j);
        Aless_k (b, j) = f_w  * q_j * A_PQ_k (P, Q, j);
        Agtr_k  (b, j) = fb_w * q_j * A_PQ_k (P, Q, j);
        // For the "QP" leg at k+q, swap (P,Q) index access.
        Aless_kq(b, j) = f_w  * q_j * A_PQ_kq(Q, P, j);
        Agtr_kq (b, j) = fb_w * q_j * A_PQ_kq(Q, P, j);
      }
    }
  }

  // Im Pi ~ cross-correlate(Aless_k, Agtr_kq) - cross-correlate(Agtr_k, Aless_kq).
  // Compute each pair's Hhat in time space, sum (with sign) before a single
  // type-2 NUFFT. Saves one type-2 per call vs two cross_correlate calls.
  using gk = grid_kind;
  nda::array<ComplexType, 2> Fless_hat(B, N_t), Fgtr_hat(B, N_t);
  nda::array<ComplexType, 2> Gless_hat(B, N_t), Ggtr_hat(B, N_t);
  conv.forward(Aless_k,  Fless_hat, gk::fermionic);
  conv.forward(Agtr_k,   Fgtr_hat,  gk::fermionic);
  conv.forward(Aless_kq, Gless_hat, gk::fermionic);
  conv.forward(Agtr_kq,  Ggtr_hat,  gk::fermionic);

  // Pi Hadamard kernel: a 4-ary elementwise map, MEM-agnostic via nda::map
  // (lazy expression evaluated on host or device per the array memory space).
  memory::array<MEM, ComplexType, 2> Hhat(B, N_t);
  Hhat = nda::map([](ComplexType fl, ComplexType gg,
                     ComplexType fg, ComplexType gl) {
    return std::conj(fl) * gg - std::conj(fg) * gl;
  })(Fless_hat, Ggtr_hat, Fgtr_hat, Gless_hat);

  nda::array<ComplexType, 2> Hraw(B, N_O);
  conv.backward(Hhat, Hraw, gk::bosonic);
  const double s_nufft = conv.nufft_scale();
  const double s_pi    = -M_PI * k_weight * s_nufft;
  for (long iP = 0; iP < Naux_loc_P; ++iP)
    for (long iQ = 0; iQ < Naux_loc_Q; ++iQ) {
      const long b = iP * Naux_loc_Q + iQ;
      for (long iO = 0; iO < N_O; ++iO)
        ImPi_PQ_O(iP, iQ, iO) += s_pi * Hraw(b, iO);
    }
}

/**
 * Recover Re Pi^R from Im Pi^R via Hilbert transform on the bosonic grid.
 *
 * @param conv      NUFFT engine (ntrans >= Naux_P*Naux_Q)
 * @param ImPi_PQ_O Im Pi for one q (Naux_P, Naux_Q, N_Omega), real-valued.
 *                  Either the full (Naux, Naux, ...) array or a (P_loc, Q_loc, ...)
 *                  local slice from a distributed array.
 * @param RePi_PQ_O OUTPUT Re Pi, same shape as ImPi_PQ_O.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
inline void RePi_from_ImPi(detail::real_axis_conv_base_t<MEM> & conv,
                           memory::array<MEM, double, 3> const& ImPi_PQ_O,
                           memory::array<MEM, double, 3>      & RePi_PQ_O)
{
  if constexpr (MEM != HOST_MEMORY) {
    utils::check(false,
                 "RePi_from_ImPi<DEVICE>: device kernel for the (P, Q) "
                 "<-> batch gather/scatter not yet implemented.");
    return;
  }
  const long Naux_P = ImPi_PQ_O.shape()[0];
  const long Naux_Q = ImPi_PQ_O.shape()[1];
  const long N_O = ImPi_PQ_O.shape()[2];
  const long B = Naux_P * Naux_Q;

  memory::array<MEM, double, 2> ImBuf(B, N_O), ReBuf(B, N_O);
  for (long P = 0; P < Naux_P; ++P)
    for (long Q = 0; Q < Naux_Q; ++Q) {
      const long b = P * Naux_Q + Q;
      for (long iO = 0; iO < N_O; ++iO)
        ImBuf(b, iO) = ImPi_PQ_O(P, Q, iO);
    }

  conv.hilbert(ImBuf, ReBuf, grid_kind::bosonic);

  for (long P = 0; P < Naux_P; ++P)
    for (long Q = 0; Q < Naux_Q; ++Q) {
      const long b = P * Naux_Q + Q;
      for (long iO = 0; iO < N_O; ++iO)
        RePi_PQ_O(P, Q, iO) = ReBuf(b, iO);
    }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_PI_HPP
