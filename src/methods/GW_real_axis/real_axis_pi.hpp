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
inline void accumulate_ImPi_one_kq(real_axis_conv_t & conv,
                                   nda::array<ComplexType, 3> const& A_PQ_k,
                                   nda::array<ComplexType, 3> const& A_PQ_kq,
                                   nda::array<ComplexType, 3>      & ImPi_PQ_O,
                                   double k_weight)
{
  const long Naux = A_PQ_k.shape()[0];
  const long N_w = A_PQ_k.shape()[2];
  const long N_O = ImPi_PQ_O.shape()[2];

  utils::check(A_PQ_k.shape() == A_PQ_kq.shape(),
               "accumulate_ImPi_one_kq: A_k and A_kq must have same shape");
  utils::check(A_PQ_k.shape()[1] == Naux,
               "accumulate_ImPi_one_kq: A must be square in (P,Q)");
  utils::check(ImPi_PQ_O.shape()[0] == Naux and ImPi_PQ_O.shape()[1] == Naux,
               "accumulate_ImPi_one_kq: ImPi shape mismatch");
  utils::check(N_w == conv.N_w() and N_O == conv.N_Omega(),
               "accumulate_ImPi_one_kq: grid mismatch");

  auto const& grid = conv.grid();
  auto const& w   = grid.w();

  // Build weighted spectra at k and k+q for the bubble.
  // Layout: (B, N_w) with B = Naux*Naux, and for index b = P*Naux + Q:
  //   - F^<_{b}(w) = f(w) A_{PQ}(k, w)        (left leg, at k)
  //   - G^>_{b}(w) = (1-f(w)) A_{QP}(k+q, w)  (right leg, at k+q, transposed indices)
  // and similarly for the swapped sign of the kernel.
  const long B = Naux * Naux;

  nda::array<ComplexType, 2> Aless_k(B, N_w);
  nda::array<ComplexType, 2> Agtr_k(B, N_w);
  nda::array<ComplexType, 2> Aless_kq(B, N_w);
  nda::array<ComplexType, 2> Agtr_kq(B, N_w);

  for (long P = 0; P < Naux; ++P) {
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long j = 0; j < N_w; ++j) {
        const double f_w   = grid.fermi(w(j));
        const double fb_w  = 1.0 - f_w;
        Aless_k (b, j) = f_w  * A_PQ_k (P, Q, j);
        Agtr_k  (b, j) = fb_w * A_PQ_k (P, Q, j);
        // For the "QP" leg at k+q, swap (P,Q) index access.
        Aless_kq(b, j) = f_w  * A_PQ_kq(Q, P, j);
        Agtr_kq (b, j) = fb_w * A_PQ_kq(Q, P, j);
      }
    }
  }

  nda::array<ComplexType, 2> term1(B, N_O);
  nda::array<ComplexType, 2> term2(B, N_O);
  // Im Pi  ~  (A^< * A^>)(k,q; Omega) - (A^> * A^<)(k,q; Omega)
  conv.cross_correlate(Aless_k,  Agtr_kq, term1,
                       real_axis_conv_t::grid_kind::fermionic,
                       real_axis_conv_t::grid_kind::bosonic);
  conv.cross_correlate(Agtr_k,   Aless_kq, term2,
                       real_axis_conv_t::grid_kind::fermionic,
                       real_axis_conv_t::grid_kind::bosonic);

  const double scale = -M_PI * k_weight;
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long iO = 0; iO < N_O; ++iO)
        ImPi_PQ_O(P, Q, iO) += scale * (term1(b, iO) - term2(b, iO));
    }
}

/**
 * Recover Re Pi^R from Im Pi^R via Hilbert transform on the bosonic grid.
 *
 * @param conv      NUFFT engine (ntrans >= Naux*Naux)
 * @param ImPi_PQ_O Im Pi for one q (Naux, Naux, N_Omega), real-valued
 * @param RePi_PQ_O OUTPUT Re Pi for one q (Naux, Naux, N_Omega), real-valued
 */
inline void RePi_from_ImPi(real_axis_conv_t & conv,
                           nda::array<double, 3> const& ImPi_PQ_O,
                           nda::array<double, 3>      & RePi_PQ_O)
{
  const long Naux = ImPi_PQ_O.shape()[0];
  const long N_O = ImPi_PQ_O.shape()[2];
  const long B = Naux * Naux;

  nda::array<double, 2> ImBuf(B, N_O), ReBuf(B, N_O);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long iO = 0; iO < N_O; ++iO)
        ImBuf(b, iO) = ImPi_PQ_O(P, Q, iO);
    }

  conv.hilbert(ImBuf, ReBuf, real_axis_conv_t::grid_kind::bosonic);

  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q) {
      const long b = P * Naux + Q;
      for (long iO = 0; iO < N_O; ++iO)
        RePi_PQ_O(P, Q, iO) = ReBuf(b, iO);
    }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_PI_HPP
