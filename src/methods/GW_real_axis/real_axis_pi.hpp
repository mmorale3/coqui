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
#include "nda/tensor.hpp"
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
                                   double k_weight)
{
  // A_PQ_k, A_PQ_kq, and ImPi_PQ_O are all (Naux_P, Naux_Q, N_*) views
  // with matching (P, Q) ranges. When (P, Q) is distributed, all three are
  // local blocks of the same shape and rank-ordering, so the kernel runs
  // entirely on local data with no cross-rank reads.
  //
  // The bare bubble formally calls for A_QP on the second leg (transposed
  // aux indices). For the matrix-hermitian symmetrized spectral function
  // -- which is what the class API feeds into the kernel -- aux A_aux is
  // hermitian: A_aux(Q, P) = conj(A_aux(P, Q)). We therefore replace the
  // "global (Q, P)" read on the second leg with conj of the LOCAL (P, Q)
  // entry on the same rank. See test_real_axis_hermiticity (commit
  // 40d91a2) for the round-off-level validation of this identity.
  const long Naux_P = ImPi_PQ_O.shape()[0];
  const long Naux_Q = ImPi_PQ_O.shape()[1];
  const long N_w    = A_PQ_k.shape()[2];
  const long N_O    = ImPi_PQ_O.shape()[2];

  utils::check(A_PQ_k.shape() == A_PQ_kq.shape(),
               "accumulate_ImPi_one_kq: A_k and A_kq must have same shape");
  utils::check(A_PQ_k.shape()[0] == Naux_P and A_PQ_k.shape()[1] == Naux_Q,
               "accumulate_ImPi_one_kq: A and ImPi (P, Q) shapes must match");
  utils::check(N_w == conv.N_w() and N_O == conv.N_Omega(),
               "accumulate_ImPi_one_kq: grid mismatch");

  auto const& grid = conv.grid();
  auto const& w    = grid.w();
  auto const& wq_h = grid.w_weights();

  const long B   = Naux_P * Naux_Q;
  const long N_t = conv.N_t();

  // Build weighted spectra at k and k+q for the bubble.
  // Layout: (B, N_w) with B = Naux_P * Naux_Q. For index b = iP*Naux_Q + iQ:
  //   - F^<_{b}(w) = f(w) A_{P,Q}(k, w)                 (left leg, at k)
  //   - G^>_{b}(w) = (1-f(w)) conj(A_{P,Q}(k+q, w))     (right leg, at k+q,
  //                                                      conj of LOCAL block via
  //                                                      aux hermiticity)
  // The weighted forms also absorb the trapezoidal quadrature weights, so
  // the lower-level NUFFT primitives don't need to apply them again.
  memory::array<MEM, ComplexType, 2> Aless_k(B, N_w);
  memory::array<MEM, ComplexType, 2> Agtr_k(B, N_w);
  memory::array<MEM, ComplexType, 2> Aless_kq(B, N_w);
  memory::array<MEM, ComplexType, 2> Agtr_kq(B, N_w);

  if constexpr (MEM == HOST_MEMORY) {
    for (long iP = 0; iP < Naux_P; ++iP) {
      for (long iQ = 0; iQ < Naux_Q; ++iQ) {
        const long b = iP * Naux_Q + iQ;
        for (long j = 0; j < N_w; ++j) {
          const double f_w  = grid.fermi(w(j));
          const double fb_w = 1.0 - f_w;
          const double q_j  = wq_h(j);
          Aless_k (b, j) = f_w  * q_j * A_PQ_k (iP, iQ, j);
          Agtr_k  (b, j) = fb_w * q_j * A_PQ_k (iP, iQ, j);
          // Second leg "Q, P" at k+q -> conj of LOCAL (iP, iQ) block via
          // matrix-hermiticity of the symmetrized A_aux (commit 40d91a2).
          Aless_kq(b, j) = f_w  * q_j * std::conj(A_PQ_kq(iP, iQ, j));
          Agtr_kq (b, j) = fb_w * q_j * std::conj(A_PQ_kq(iP, iQ, j));
        }
      }
    }
  } else {
    // Device path: build f(w)*wq and (1-f(w))*wq vectors on host (cheap,
    // N_w is hundreds at most) and push to MEM-side complex 1D arrays so
    // we can do the broadcast multiply via cuTENSOR.
    nda::array<ComplexType, 1> f_wq_h(N_w), fb_wq_h(N_w);
    for (long j = 0; j < N_w; ++j) {
      const double f_w  = grid.fermi(w(j));
      const double fb_w = 1.0 - f_w;
      const double q_j  = wq_h(j);
      f_wq_h(j)  = ComplexType(f_w  * q_j, 0.0);
      fb_wq_h(j) = ComplexType(fb_w * q_j, 0.0);
    }
    auto f_wq  = memory::to_memory_space<MEM>(f_wq_h);
    auto fb_wq = memory::to_memory_space<MEM>(fb_wq_h);

    // C-layout (Naux_P, Naux_Q, N_w) is contiguous so reshape to (B, N_w)
    // is a no-copy view; copy that view into the scratch buffers.
    auto A_k_2D  = nda::reshape(A_PQ_k,
                                std::array<long,2>{B, N_w});
    auto A_kq_2D = nda::reshape(A_PQ_kq,
                                std::array<long,2>{B, N_w});

    Aless_k  = A_k_2D;
    Agtr_k   = A_k_2D;
    Aless_kq = A_kq_2D;
    nda::tensor::scale(ComplexType(1.0), Aless_kq, nda::tensor::op::CONJ);
    Agtr_kq  = Aless_kq;  // reuse the conj'd k+q block

    // Multiply each by the appropriate (f or 1-f)*wq broadcast vector.
    nda::tensor::elementwise(ComplexType(1.0), f_wq,  std::string_view("j"),
                             ComplexType(1.0), Aless_k,  std::string_view("bj"),
                             nda::tensor::op::MUL);
    nda::tensor::elementwise(ComplexType(1.0), fb_wq, std::string_view("j"),
                             ComplexType(1.0), Agtr_k,   std::string_view("bj"),
                             nda::tensor::op::MUL);
    nda::tensor::elementwise(ComplexType(1.0), f_wq,  std::string_view("j"),
                             ComplexType(1.0), Aless_kq, std::string_view("bj"),
                             nda::tensor::op::MUL);
    nda::tensor::elementwise(ComplexType(1.0), fb_wq, std::string_view("j"),
                             ComplexType(1.0), Agtr_kq,  std::string_view("bj"),
                             nda::tensor::op::MUL);
  }

  // Im Pi ~ cross-correlate(Aless_k, Agtr_kq) - cross-correlate(Agtr_k, Aless_kq).
  // Compute each pair's Hhat in time space, sum (with sign) before a single
  // type-2 NUFFT. Saves one type-2 per call vs two cross_correlate calls.
  using gk = grid_kind;
  memory::array<MEM, ComplexType, 2> Fless_hat(B, N_t), Fgtr_hat(B, N_t);
  memory::array<MEM, ComplexType, 2> Gless_hat(B, N_t), Ggtr_hat(B, N_t);
  conv.forward(Aless_k,  Fless_hat, gk::fermionic);
  conv.forward(Agtr_k,   Fgtr_hat,  gk::fermionic);
  conv.forward(Aless_kq, Gless_hat, gk::fermionic);
  conv.forward(Agtr_kq,  Ggtr_hat,  gk::fermionic);

  // Pi Hadamard kernel:  Hhat = conj(Fless)*Ggtr - conj(Fgtr)*Gless.
  // On host we use a 4-arg nda::map; on device, lazy expressions segfault
  // so we build it explicitly via tensor::scale(op::CONJ) + elementwise.
  memory::array<MEM, ComplexType, 2> Hhat(B, N_t);
  if constexpr (MEM == HOST_MEMORY) {
    Hhat = nda::map([](ComplexType fl, ComplexType gg,
                       ComplexType fg, ComplexType gl) {
      return std::conj(fl) * gg - std::conj(fg) * gl;
    })(Fless_hat, Ggtr_hat, Fgtr_hat, Gless_hat);
  } else {
    memory::array<MEM, ComplexType, 2> tmp(B, N_t);

    // tmp = conj(Fless) * Ggtr
    tmp = Fless_hat;
    nda::tensor::scale(ComplexType(1.0), tmp, nda::tensor::op::CONJ);
    nda::tensor::elementwise(ComplexType(1.0), Ggtr_hat, std::string_view("bk"),
                             ComplexType(1.0), tmp,      std::string_view("bk"),
                             nda::tensor::op::MUL);

    // Hhat = conj(Fgtr) * Gless
    Hhat = Fgtr_hat;
    nda::tensor::scale(ComplexType(1.0), Hhat, nda::tensor::op::CONJ);
    nda::tensor::elementwise(ComplexType(1.0), Gless_hat, std::string_view("bk"),
                             ComplexType(1.0), Hhat,      std::string_view("bk"),
                             nda::tensor::op::MUL);

    // Hhat = tmp - Hhat = conj(Fless)*Ggtr - conj(Fgtr)*Gless
    nda::tensor::scale(ComplexType(-1.0), Hhat);
    nda::tensor::elementwise(ComplexType(1.0), tmp,  std::string_view("bk"),
                             ComplexType(1.0), Hhat, std::string_view("bk"),
                             nda::tensor::op::SUM);
  }

  memory::array<MEM, ComplexType, 2> Hraw(B, N_O);
  conv.backward(Hhat, Hraw, gk::bosonic);
  const double s_nufft = conv.nufft_scale();
  const double s_pi    = -M_PI * k_weight * s_nufft;

  if constexpr (MEM == HOST_MEMORY) {
    for (long iP = 0; iP < Naux_P; ++iP)
      for (long iQ = 0; iQ < Naux_Q; ++iQ) {
        const long b = iP * Naux_Q + iQ;
        for (long iO = 0; iO < N_O; ++iO)
          ImPi_PQ_O(iP, iQ, iO) += s_pi * Hraw(b, iO);
      }
  } else {
    // Device: reshape Hraw (B, N_O) -> (Naux_P, Naux_Q, N_O), then accumulate.
    auto Hraw_3D = nda::reshape(Hraw,
                                std::array<long,3>{Naux_P, Naux_Q, N_O});
    nda::tensor::elementwise(ComplexType(s_pi), Hraw_3D,   std::string_view("PQO"),
                             ComplexType(1.0),  ImPi_PQ_O, std::string_view("PQO"),
                             nda::tensor::op::SUM);
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
template<MEMORY_SPACE MEM = HOST_MEMORY,
         typename ImIn, typename ReOut>
inline void RePi_from_ImPi(detail::real_axis_conv_base_t<MEM> & conv,
                           ImIn  const& ImPi_PQ_O,
                           ReOut      & RePi_PQ_O)
{
  const long Naux_P = ImPi_PQ_O.shape()[0];
  const long Naux_Q = ImPi_PQ_O.shape()[1];
  const long N_O = ImPi_PQ_O.shape()[2];
  const long B = Naux_P * Naux_Q;

  // Both gather and scatter are pure C-layout reshapes (no data motion); no
  // host/device branch is needed -- nda::reshape returns a memory-view of
  // the same storage.
  auto ImPi_2D = nda::reshape(ImPi_PQ_O,
                              std::array<long,2>{B, N_O});
  auto RePi_2D = nda::reshape(RePi_PQ_O,
                              std::array<long,2>{B, N_O});

  // hilbert takes mutable rarray<2> by reference. Stage through scratch.
  memory::array<MEM, double, 2> ImBuf(B, N_O), ReBuf(B, N_O);
  ImBuf = ImPi_2D;
  conv.hilbert(ImBuf, ReBuf, grid_kind::bosonic);
  RePi_2D = ReBuf;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_PI_HPP
