/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 *
 * Real-axis q->0 divergence-treatment utilities.
 *
 * Provides the building blocks needed for Gygi-Baldereschi-style head/wings
 * treatment of the q=0 Coulomb singularity in periodic systems. Mirrors the
 * imag-axis machinery in `methods/GW/g0_div_utils.hpp` but on the real-
 * frequency bosonic grid.
 *
 * What's here:
 *   - `compute_eps_inv_head_O`: head of the inverse symmetric dielectric
 *     function `eps^-1_{G=G'=0}(q, Omega)` from the screened interaction W
 *     on the bosonic real-frequency grid, plus the q->0 estimate using the
 *     smallest-|q| in the IBZ ("gygi_smallest_q" treatment).
 *
 * What's NOT here yet:
 *   - The Sigma^c divergence correction itself. The imag-axis correction in
 *     `methods/GW/thc_gw.icc::Sigma_div_correction` is a tau-pointwise
 *     formula, `Delta(t) = -madelung * eps_inv_head(t) * T G(t) T^dag`,
 *     because in imag-time the Sigma^c convolution collapses to a tau-by-tau
 *     product. On the real axis, the analog is a frequency-domain
 *     convolution between A_T(s,k,i,j,eps) (the T-rotated spectral function)
 *     and B_head(Omega) (a head-channel bosonic kernel built from
 *     eps_inv_head). The exact formula (sign conventions, factor of pi,
 *     interaction with the existing f / n_B kernel structure) needs to be
 *     derived against `notes/isdf_gw_prb_draft_v2.tex` Sec. VII before being
 *     implemented. Until then `evaluate_serial` just zeros iq_gamma.
 */

#ifndef COQUI_REAL_AXIS_DIV_UTILS_HPP
#define COQUI_REAL_AXIS_DIV_UTILS_HPP

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis {

/**
 * Compute the head of the inverse symmetric dielectric function
 * `eps^-1_{G=G'=0}(q, Omega)` from the THC-auxiliary-basis screened
 * interaction `W_qPQO`, evaluated on the bosonic real-frequency grid.
 *
 * Formula (matches the imag-axis convention, see
 * `methods/GW/g0_div_utils.hpp::eval_eps_inv_q`):
 *
 *   eps^-1(q, Omega) = (|q|^2 / 4 pi) * Volume *
 *                       sum_{P,Q} chi_bar(q, P) W(q, P, Q, Omega)
 *                                   conj(chi_bar(q, Q))
 *
 * For q at the Gamma point (|q|=0) the prefactor is zero; that entry is
 * left as 0 in the output and the q->0 estimate uses the smallest-non-zero
 * |q| point ("gygi_smallest_q" treatment), matching what the imag-axis
 * `extrapolate_eps_inv_q0(div_treatment="gygi_smallest_q")` does.
 *
 * @param W_qPQO      [INPUT]  shape (Nq, Naux, Naux, N_Omega), screened W
 *                             on bosonic grid (complex W=Re W + i Im W).
 * @param Qpts        [INPUT]  shape (Nq, 3), q-points (Cartesian).
 * @param chi_bar_qu  [INPUT]  shape (Nq, Naux), auxiliary basis "bar" head
 *                             from `thc.basis_bar_head()`.
 * @param volume      [INPUT]  unit-cell volume.
 * @param eps_inv_qO  [OUTPUT] shape (Nq, N_Omega). Allocated by caller.
 * @param eps_inv_O   [OUTPUT] shape (N_Omega,). Estimate of eps^-1(q->0, O)
 *                             via the smallest-|q| treatment.
 */
inline void compute_eps_inv_head_O(
    nda::array<ComplexType, 4> const& W_qPQO,
    nda::array<double, 2>      const& Qpts,
    nda::array<ComplexType, 2> const& chi_bar_qu,
    double                            volume,
    nda::array<ComplexType, 2>      & eps_inv_qO,
    nda::array<ComplexType, 1>      & eps_inv_O)
{
  const long Nq   = W_qPQO.shape()[0];
  const long Naux = W_qPQO.shape()[1];
  const long N_O  = W_qPQO.shape()[3];

  utils::check(W_qPQO.shape()[2] == Naux,
               "compute_eps_inv_head_O: W not square in (P,Q)");
  utils::check(Qpts.shape()[0] == Nq and Qpts.shape()[1] == 3,
               "compute_eps_inv_head_O: Qpts shape ({}, {}) does not match (Nq, 3)=({}, 3)",
               Qpts.shape()[0], Qpts.shape()[1], Nq);
  utils::check(chi_bar_qu.shape()[0] == Nq and chi_bar_qu.shape()[1] == Naux,
               "compute_eps_inv_head_O: chi_bar_qu shape mismatch");
  utils::check(eps_inv_qO.shape()[0] == Nq and eps_inv_qO.shape()[1] == N_O,
               "compute_eps_inv_head_O: eps_inv_qO output shape mismatch");
  utils::check(eps_inv_O.shape()[0] == N_O,
               "compute_eps_inv_head_O: eps_inv_O output shape mismatch");

  const double fpi = 4.0 * M_PI;
  eps_inv_qO = ComplexType(0.0, 0.0);

  // Per-q head: eps^-1(q, O) = (|q|^2 / 4pi) * Vol * sum_PQ chi_bar(q,P) W(q,P,Q,O) conj(chi_bar(q,Q))
  // W has Omega innermost, so the (P, Q) slice does not have unit inner stride;
  // BLAS gemv would error. The triple sum is O(Nq * N_O * Naux^2) -- trivial.
  for (long iq = 0; iq < Nq; ++iq) {
    const double qx = Qpts(iq, 0), qy = Qpts(iq, 1), qz = Qpts(iq, 2);
    const double q_abs2 = qx*qx + qy*qy + qz*qz;
    if (q_abs2 < 1e-20) continue;          // Gamma: leave 0 (no contribution by formula).
    const double prefactor = (q_abs2 / fpi) * volume;

    for (long iO = 0; iO < N_O; ++iO) {
      ComplexType acc(0.0, 0.0);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          acc += chi_bar_qu(iq, P) * W_qPQO(iq, P, Q, iO)
                                   * std::conj(chi_bar_qu(iq, Q));
      eps_inv_qO(iq, iO) = prefactor * acc;
    }
  }

  // q->0 estimate via the smallest-|q| (non-zero) point in the IBZ.
  // Mirrors `g0_div_utils::find_smallest_qabs` + `extrapolate_eps_inv_q0`'s
  // "gygi_smallest_q" branch.
  long smallest_iq = -1;
  double smallest_q2 = std::numeric_limits<double>::infinity();
  for (long iq = 0; iq < Nq; ++iq) {
    const double qx = Qpts(iq, 0), qy = Qpts(iq, 1), qz = Qpts(iq, 2);
    const double q_abs2 = qx*qx + qy*qy + qz*qz;
    if (q_abs2 < 1e-20) continue;          // skip Gamma
    if (q_abs2 < smallest_q2) {
      smallest_q2 = q_abs2;
      smallest_iq = iq;
    }
  }

  if (smallest_iq >= 0) {
    for (long iO = 0; iO < N_O; ++iO)
      eps_inv_O(iO) = eps_inv_qO(smallest_iq, iO);
  } else {
    // Single-q case (Nq==1, Gamma only): no extrapolation possible.
    eps_inv_O = ComplexType(0.0, 0.0);
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DIV_UTILS_HPP
