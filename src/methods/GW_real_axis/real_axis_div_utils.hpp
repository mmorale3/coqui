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

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"

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

/**
 * Resample a bosonic spectral function defined on Omega>=0 onto the
 * fermionic w grid in [-w_max, w_max], using the diagonal-odd extension
 *   B(-Omega) = -B(Omega).
 * Linear interpolation between bosonic grid points; linear extrapolation
 * to 0 inside [0, Omega_min); zero beyond Omega_max. Mirrors the helper
 * in real_axis_sigma.hpp::resample_bosonic_to_fermionic but for a
 * scalar (1D) bosonic input rather than a (Naux, Naux, N_Omega) tensor.
 */
inline void resample_scalar_bosonic_to_fermionic(
    real_freq_grid_t       const& grid,
    nda::array<double, 1>  const& B_O,
    nda::array<double, 1>      & B_w)
{
  auto const& Og = grid.Omega();
  auto const& wg = grid.w();
  const long N_O = Og.shape()[0];
  const long N_w = wg.shape()[0];
  utils::check(B_O.shape()[0] == N_O,
               "resample_scalar_bosonic_to_fermionic: B_O length mismatch");
  utils::check(B_w.shape()[0] == N_w,
               "resample_scalar_bosonic_to_fermionic: B_w length mismatch");

  for (long iw = 0; iw < N_w; ++iw) {
    const double w_l = wg(iw);
    const double sign = (w_l >= 0.0 ? 1.0 : -1.0);
    const double a    = std::abs(w_l);
    double v = 0.0;
    if (a <= Og(0)) {
      // Linear ramp to 0 at Omega=0 (B(0+) = 0 by physics).
      v = (a / Og(0)) * B_O(0);
    } else if (a >= Og(N_O - 1)) {
      v = 0.0;
    } else {
      // Locate bin.
      long lo = 0, hi = N_O - 1;
      while (hi - lo > 1) {
        long mid = (lo + hi) / 2;
        if (Og(mid) <= a) lo = mid;
        else              hi = mid;
      }
      const double t = (a - Og(lo)) / (Og(hi) - Og(lo));
      v = (1.0 - t) * B_O(lo) + t * B_O(hi);
    }
    B_w(iw) = sign * v;
  }
}

/**
 * Apply the q=0 GW self-energy divergence correction (head channel) on
 * the real-frequency axis.
 *
 * Mirrors the imag-axis `gw_t::Sigma_div_correction` (in `methods/GW/
 * thc_gw.icc`), which on the imag-axis is a tau-pointwise product
 * Delta(t) = -madelung * eps_inv_head(t) * T G(t) T^dag. On the real
 * axis the GW formula in spectral form is a frequency convolution,
 * so the correction becomes a (fermionic/bosonic) convolution rather
 * than a w-pointwise product:
 *
 *   Im DSigma_ij(s, k, w) = -pi * madelung *
 *       int dw' [f(w' - mu) + n_B(w - w' + mu)]
 *               (T_skia A_phys_skab(w' - mu) T*_skjb)
 *               B_head(w - w')
 *
 * where:
 *   T_skia       = sum_P conj(X(s, k, P, i)) X(s, k, P, a) conj(chi_head(0, P))
 *   A_phys_ij(w) = matrix-hermitian symmetrized A_wskij (the physical
 *                  spectral function; .real() of A_wskij is its componentwise
 *                  storage and is NOT the matrix-hermitian object)
 *   B_head(Om)   = -(1/pi) Im eps_inv_head(0, Om), Om>=0; odd-extended for Om<0
 *
 * The Re part follows by Hilbert transform (KK) on the real-w grid.
 *
 * Inputs:
 *   conv             NUFFT engine for the per-rank Hilbert transform.
 *   grid             real-frequency grid.
 *   madelung         lattice Madelung constant (analytic q=0 BZ integral).
 *   eps_inv_head_O   shape (N_Omega,): head of eps^-1 at q->0 (complex).
 *   chi_head_qP      shape (Nq, Naux): thc.basis_head() at q-grid; we use q=0 row.
 *   X_skPmu          shape (ns, Nk, Naux, nbnd): THC orbital factor.
 *   A_wskij          shape (N_w, ns, Nk, nbnd, nbnd): stored A_wskij (we
 *                    compute the matrix-hermitian symmetrization internally).
 * Outputs:
 *   ImSigma_wskij    shape (N_w, ns, Nk, nbnd, nbnd): updated in place
 *                    (DSigma added).
 *   ReSigma_wskij    shape (N_w, ns, Nk, nbnd, nbnd): updated in place.
 */
template<typename ChiHead_t, typename X_t, typename Awskij_t,
         typename ImSout_t, typename ReSout_t>
inline void apply_sigma_head_correction_real_axis(
    detail::real_axis_conv_base_t<HOST_MEMORY> & conv,
    real_freq_grid_t            const& grid,
    double                             madelung,
    nda::array<ComplexType, 1>  const& eps_inv_head_O,
    ChiHead_t                   const& chi_head_qP,
    X_t                         const& X_skPmu,
    Awskij_t                    const& A_wskij,
    ImSout_t                         & ImSigma_wskij,
    ReSout_t                         & ReSigma_wskij)
{
  const long N_w  = A_wskij.shape()[0];
  const long ns   = A_wskij.shape()[1];
  const long Nk   = A_wskij.shape()[2];
  const long nbnd = A_wskij.shape()[3];
  const long Naux = X_skPmu.shape()[2];
  const long N_O  = eps_inv_head_O.shape()[0];

  utils::check(grid.N_w() == N_w,
               "apply_sigma_head_correction_real_axis: N_w mismatch");
  utils::check(grid.N_Omega() == N_O,
               "apply_sigma_head_correction_real_axis: N_Omega mismatch");
  utils::check(chi_head_qP.shape()[1] == Naux,
               "apply_sigma_head_correction_real_axis: chi_head Naux mismatch");

  // 1. B_head(Om) = -(1/pi) Im eps_inv_head_O(Om) on bosonic grid.
  nda::array<double, 1> B_head_O(N_O);
  for (long iO = 0; iO < N_O; ++iO)
    B_head_O(iO) = -eps_inv_head_O(iO).imag() / M_PI;

  // 2. Resample to fermionic grid with diagonal-odd extension.
  nda::array<double, 1> B_head_w(N_w);
  resample_scalar_bosonic_to_fermionic(grid, B_head_O, B_head_w);

  // 3. Per (s, k): T_skia = sum_P conj(X(s,k,P,i)) X(s,k,P,a) conj(chi_head(0,P)).
  //    chi_head_qP(0, _) is the q=0 (Gamma) row -- guaranteed to exist
  //    in the q-mesh by convention.
  nda::array<ComplexType, 4> T_skia(ns, Nk, nbnd, nbnd);
  T_skia() = ComplexType(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long i = 0; i < nbnd; ++i)
        for (long a = 0; a < nbnd; ++a) {
          ComplexType acc(0.0, 0.0);
          for (long P = 0; P < Naux; ++P)
            acc += std::conj(X_skPmu(s, k, P, i))
                 * X_skPmu(s, k, P, a)
                 * std::conj(chi_head_qP(0, P));
          T_skia(s, k, i, a) = acc;
        }

  // 4. Per (s, k, w'): build the rank-one projection
  //       AT(s, k, w', i, j) = sum_ab T_skia A_phys_skab(w') T*_skjb
  //    where A_phys is the matrix-hermitian symmetrization of A_wskij.
  //    Then ImDSigma_ij(s, k, w) = -pi * madelung *
  //       sum_w' dw_w' [f(w'-mu) + n_B(w-w'+mu)] AT(s,k,w',i,j) B_head_w(w-w')
  //    Direct quadrature (cost O(ns * Nk * N_w^2 * nbnd^2)).
  auto const& w_arr = grid.w();
  auto const& wq    = grid.w_weights();
  const double mu   = grid.mu_chem();

  // A_phys has the same (ns, Nk, N_w, nbnd, nbnd) layout used by the kernels.
  nda::array<ComplexType, 5> A_phys(ns, Nk, N_w, nbnd, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long mu_i = 0; mu_i < nbnd; ++mu_i)
          for (long nu = 0; nu < nbnd; ++nu) {
            // Note: state.A_wskij has layout (N_w, ns, Nk, nbnd, nbnd).
            A_phys(s, k, iw, mu_i, nu) =
                ComplexType(0.5, 0.0) *
                (A_wskij(iw, s, k, mu_i, nu)
                 + std::conj(A_wskij(iw, s, k, nu, mu_i)));
          }

  // Precompute AT(s, k, w', i, j) = T A_phys T^†.
  nda::array<ComplexType, 5> AT(ns, Nk, N_w, nbnd, nbnd);
  AT() = ComplexType(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw) {
        // Tmp = T A_phys
        nda::array<ComplexType, 2> tmp(nbnd, nbnd);
        nda::array<ComplexType, 2> T_view(nbnd, nbnd);
        nda::array<ComplexType, 2> A_view(nbnd, nbnd);
        nda::array<ComplexType, 2> AT_view(nbnd, nbnd);
        for (long i = 0; i < nbnd; ++i)
          for (long a = 0; a < nbnd; ++a)
            T_view(i, a) = T_skia(s, k, i, a);
        for (long a = 0; a < nbnd; ++a)
          for (long b = 0; b < nbnd; ++b)
            A_view(a, b) = A_phys(s, k, iw, a, b);
        const ComplexType c_one(1.0, 0.0), c_zero(0.0, 0.0);
        nda::blas::gemm(c_one, T_view, A_view, c_zero, tmp);
        nda::blas::gemm(c_one, tmp, nda::dagger(T_view), c_zero, AT_view);
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j)
            AT(s, k, iw, i, j) = AT_view(i, j);
      }

  // Direct quadrature: Im DSigma(s, k, w, i, j) =
  //   -pi * madelung * sum_w' dw_w' [f(w'-mu)+n_B(w-w'+mu)] AT(w') B_head_w(w-w')
  // The B_head_w grid is already odd-extended to negative w; for the
  // arg `w - w'` we evaluate B_head at the closest grid point via linear
  // interpolation of B_head_w (real-axis grid is uniform around 0).
  //
  // For uniformity / simplicity we interpolate B_head_w(w - w') by linear
  // search on the (potentially nonuniform) w-grid.
  auto interp_B = [&](double x) -> double {
    if (x <= w_arr(0))     return 0.0;
    if (x >= w_arr(N_w-1)) return 0.0;
    long lo = 0, hi = N_w - 1;
    while (hi - lo > 1) {
      long mid = (lo + hi) / 2;
      if (w_arr(mid) <= x) lo = mid;
      else                  hi = mid;
    }
    const double t = (x - w_arr(lo)) / (w_arr(hi) - w_arr(lo));
    return (1.0 - t) * B_head_w(lo) + t * B_head_w(hi);
  };

  // Accumulate Im DSigma into a scratch buffer; we add to ImSigma at the end.
  nda::array<double, 5> ImD(N_w, ns, Nk, nbnd, nbnd);
  ImD() = 0.0;
  const double pi_mad = M_PI * madelung;
  for (long iw = 0; iw < N_w; ++iw) {
    const double w = w_arr(iw);
    for (long iwp = 0; iwp < N_w; ++iwp) {
      const double wp = w_arr(iwp);
      const double dwp = wq(iwp);
      const double f_wp = grid.fermi(wp - mu);
      const double Om   = w - wp + mu;
      double nB = 0.0;
      if (std::abs(Om) > 1e-12) nB = grid.bose(Om);
      const double weight = -pi_mad * dwp * (f_wp + nB) * interp_B(w - wp);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              ImD(iw, s, k, i, j) += weight * AT(s, k, iwp, i, j).real();
    }
  }

  // 5. Add to ImSigma_wskij. The kernel-stored ImSigma uses .real() as the
  //    physical Im part (with .imag() carrying noise from the convolution).
  for (long iw = 0; iw < N_w; ++iw)
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j) {
            ImSigma_wskij(iw, s, k, i, j) +=
                ComplexType(ImD(iw, s, k, i, j), 0.0);
          }

  // 6. Re DSigma via Hilbert transform on the fermionic grid: build the
  //    Re part of each (s, k, i, j) from the corresponding Im part using
  //    the conv engine's batched Hilbert transform.
  using gk = grid_kind;
  const long B = ns * Nk * nbnd * nbnd;
  memory::array<HOST_MEMORY, double, 2> ImBuf(B, N_w), ReBuf(B, N_w);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j) {
          const long b = ((s * Nk + k) * nbnd + i) * nbnd + j;
          for (long iw = 0; iw < N_w; ++iw)
            ImBuf(b, iw) = ImD(iw, s, k, i, j);
        }
  conv.hilbert(ImBuf, ReBuf, gk::fermionic);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j) {
          const long b = ((s * Nk + k) * nbnd + i) * nbnd + j;
          for (long iw = 0; iw < N_w; ++iw)
            ReSigma_wskij(iw, s, k, i, j) +=
                ComplexType(ReBuf(b, iw), 0.0);
        }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DIV_UTILS_HPP
