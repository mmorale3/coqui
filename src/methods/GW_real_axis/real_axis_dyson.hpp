/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_AXIS_DYSON_HPP
#define COQUI_REAL_AXIS_REAL_AXIS_DYSON_HPP

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis {

/**
 * Solve the THC-auxiliary-basis Dyson equation for the screened interaction
 * at a single (q, Omega):
 *
 *     W^R(q, Omega) = [V^{-1}(q) - Pi^R(q, Omega)]^{-1}
 *                   = V(q) [I - Pi^R(q, Omega) V(q)]^{-1}
 *                   = [I - V(q) Pi^R(q, Omega)]^{-1} V(q)
 *
 * The implementation uses the form  W = (I - V*Pi)^{-1} V  to keep the
 * inversion well-conditioned even when V has a strong q->0 singularity
 * (the small-q regularisation should be applied to V before calling this
 * routine; see notes section on Coulomb singularities).
 *
 * @param V_PQ      [INPUT]  bare Coulomb in auxiliary basis: (Naux, Naux)
 * @param Pi_PQ     [INPUT]  retarded polarization at (q, Omega):
 *                           Pi^R = Re Pi + i Im Pi  (Naux, Naux), complex
 * @param W_PQ      [OUTPUT] retarded screened interaction at (q, Omega):
 *                           W^R = Re W + i Im W (Naux, Naux), complex
 *
 * Naux is small (typically O(100-500)) so a dense inverse is the right
 * choice. Each call costs O(Naux^3).
 */
inline void solve_dyson_W_aux(nda::array<ComplexType, 2> const& V_PQ,
                              nda::array<ComplexType, 2> const& Pi_PQ,
                              nda::array<ComplexType, 2> & W_PQ)
{
  const long N = V_PQ.shape()[0];
  utils::check(V_PQ.shape()[0] == V_PQ.shape()[1],
               "solve_dyson_W_aux: V must be square");
  utils::check(Pi_PQ.shape()[0] == N and Pi_PQ.shape()[1] == N,
               "solve_dyson_W_aux: Pi shape mismatch");
  utils::check(W_PQ.shape()[0] == N and W_PQ.shape()[1] == N,
               "solve_dyson_W_aux: W shape mismatch");

  // M = (I - V * Pi) as a dense Naux x Naux matrix.
  nda::matrix<ComplexType> M(N, N);
  for (long P = 0; P < N; ++P)
    for (long R = 0; R < N; ++R) {
      ComplexType acc(0.0, 0.0);
      for (long S = 0; S < N; ++S)
        acc += V_PQ(P, S) * Pi_PQ(S, R);
      M(P, R) = -acc;
    }
  for (long P = 0; P < N; ++P) M(P, P) += ComplexType(1.0, 0.0);

  // Invert in place; nda routes through LAPACK getrf/getri.
  nda::inverse_in_place(M);

  // W = Minv * V
  for (long P = 0; P < N; ++P)
    for (long R = 0; R < N; ++R) {
      ComplexType acc(0.0, 0.0);
      for (long S = 0; S < N; ++S)
        acc += M(P, S) * V_PQ(S, R);
      W_PQ(P, R) = acc;
    }
}

/**
 * Apply the Dyson equation across all (q, Omega) entries of a polarization
 * tensor. The corresponding W tensor is written into W_qOmegaPQ. Re/Im are
 * combined into a single complex array for the duration of the inversion;
 * the caller can split back into Re/Im after.
 *
 * @param V_qPQ        bare Coulomb, (Nq, Naux, Naux)
 * @param ImPi_qOmPQ   imaginary part of Pi, (Nq, NOmega, Naux, Naux), real-valued
 * @param RePi_qOmPQ   real part of Pi, (Nq, NOmega, Naux, Naux), real-valued
 * @param ImW_qOmPQ    output Im W, (Nq, NOmega, Naux, Naux)
 * @param ReW_qOmPQ    output Re W, (Nq, NOmega, Naux, Naux)
 */
inline void apply_dyson_W_all(nda::array<ComplexType, 3> const& V_qPQ,
                              nda::array<double, 4>      const& ImPi_qOmPQ,
                              nda::array<double, 4>      const& RePi_qOmPQ,
                              nda::array<double, 4>      & ImW_qOmPQ,
                              nda::array<double, 4>      & ReW_qOmPQ)
{
  const long Nq    = V_qPQ.shape()[0];
  const long N     = V_qPQ.shape()[1];
  const long NOm   = ImPi_qOmPQ.shape()[1];
  utils::check(V_qPQ.shape()[2] == N, "apply_dyson_W_all: V not square in (P,Q)");
  utils::check(ImPi_qOmPQ.shape()[0] == Nq and RePi_qOmPQ.shape()[0] == Nq,
               "apply_dyson_W_all: q-dim mismatch");
  utils::check(ImPi_qOmPQ.shape()[2] == N and ImPi_qOmPQ.shape()[3] == N,
               "apply_dyson_W_all: ImPi shape mismatch");
  utils::check(RePi_qOmPQ.shape()[2] == N and RePi_qOmPQ.shape()[3] == N,
               "apply_dyson_W_all: RePi shape mismatch");
  utils::check(ImW_qOmPQ.shape() == ImPi_qOmPQ.shape() and
               ReW_qOmPQ.shape() == ImPi_qOmPQ.shape(),
               "apply_dyson_W_all: ImW/ReW shape mismatch");

  nda::array<ComplexType, 2> V(N, N), Pi(N, N), W(N, N);
  for (long iq = 0; iq < Nq; ++iq) {
    for (long P = 0; P < N; ++P)
      for (long Q = 0; Q < N; ++Q)
        V(P, Q) = V_qPQ(iq, P, Q);
    for (long iO = 0; iO < NOm; ++iO) {
      for (long P = 0; P < N; ++P)
        for (long Q = 0; Q < N; ++Q)
          Pi(P, Q) = ComplexType(RePi_qOmPQ(iq, iO, P, Q),
                                 ImPi_qOmPQ(iq, iO, P, Q));
      solve_dyson_W_aux(V, Pi, W);
      for (long P = 0; P < N; ++P)
        for (long Q = 0; Q < N; ++Q) {
          ReW_qOmPQ(iq, iO, P, Q) = W(P, Q).real();
          ImW_qOmPQ(iq, iO, P, Q) = W(P, Q).imag();
        }
    }
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_DYSON_HPP
