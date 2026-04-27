/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_DYSON_G_HPP
#define COQUI_REAL_AXIS_DYSON_G_HPP

#include <complex>
#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * Solve the retarded Dyson equation for the one-particle Green's function
 * on the real axis at one (s, k, w):
 *
 *   G^R(k, w) = [(w + mu + i eta) I - H_MF(k) - Sigma^R(k, w)]^{-1}
 *
 * with Sigma^R = Sigma_x + Sigma^c (real plus imaginary parts), and update
 * the spectral function A(k, w) = -(1/pi) Im G^R(k, w).
 *
 * @param H_MF      (nbnd, nbnd) one-body mean-field Hamiltonian at k
 * @param Sigma_x   (nbnd, nbnd) static exchange self-energy at (s, k)
 * @param ReSigma_c (nbnd, nbnd) Re of Sigma^c at (s, k, w)
 * @param ImSigma_c (nbnd, nbnd) Im of Sigma^c at (s, k, w)
 * @param w         real frequency
 * @param mu        chemical potential
 * @param eta       infinitesimal positive broadening (default 1e-3)
 * @param G_out     OUTPUT (nbnd, nbnd) retarded G
 * @param A_out     OUTPUT (nbnd, nbnd) spectral function = -(1/pi) Im G^R
 */
inline void dyson_G_one_kw(nda::array<ComplexType, 2> const& H_MF,
                           nda::array<ComplexType, 2> const& Sigma_x,
                           nda::array<ComplexType, 2> const& ReSigma_c,
                           nda::array<ComplexType, 2> const& ImSigma_c,
                           double w, double mu, double eta,
                           nda::array<ComplexType, 2> & G_out,
                           nda::array<ComplexType, 2> & A_out)
{
  const long nbnd = H_MF.shape()[0];
  utils::check(H_MF.shape()[1] == nbnd, "dyson_G_one_kw: H_MF not square");
  utils::check(Sigma_x.shape() == H_MF.shape(),  "dyson_G_one_kw: Sigma_x shape");
  utils::check(ReSigma_c.shape() == H_MF.shape(), "dyson_G_one_kw: ReSigma_c shape");
  utils::check(ImSigma_c.shape() == H_MF.shape(), "dyson_G_one_kw: ImSigma_c shape");
  utils::check(G_out.shape() == H_MF.shape(),    "dyson_G_one_kw: G_out shape");
  utils::check(A_out.shape() == H_MF.shape(),    "dyson_G_out: A_out shape");

  nda::matrix<ComplexType> M(nbnd, nbnd);
  const ComplexType wmuieta(w + mu, eta);
  for (long mu_i = 0; mu_i < nbnd; ++mu_i)
    for (long nu = 0; nu < nbnd; ++nu) {
      ComplexType S = Sigma_x(mu_i, nu)
                    + ComplexType(ReSigma_c(mu_i, nu).real(),
                                  ImSigma_c(mu_i, nu).real());
      M(mu_i, nu) = -H_MF(mu_i, nu) - S;
    }
  for (long n = 0; n < nbnd; ++n) M(n, n) += wmuieta;

  nda::inverse_in_place(M);

  for (long mu_i = 0; mu_i < nbnd; ++mu_i)
    for (long nu = 0; nu < nbnd; ++nu) {
      G_out(mu_i, nu) = M(mu_i, nu);
      A_out(mu_i, nu) = ComplexType(-M(mu_i, nu).imag() / M_PI,
                                    +M(mu_i, nu).real() / M_PI);
      // We store -(1/pi) G^R as a complex number whose REAL part is the
      // spectral function and whose IMAG part is the Hilbert transform of A
      // (i.e. -(1/pi) Re G^R). The downstream Pi/Sigma kernels use the
      // .real() part of A_skwij as the actual spectral function.
    }
}

/**
 * Apply the Dyson update across all (s, k, w) and return the updated
 * spectral function.
 *
 * @param grid          finite-T real-frequency grid (carries mu_chem)
 * @param H_MF_skij     (ns, Nk, nbnd, nbnd) one-body mean-field Hamiltonian
 * @param Sigma_x_skij  (ns, Nk, nbnd, nbnd) static exchange self-energy
 * @param ReSigma_c_skwij (ns, Nk, N_w, nbnd, nbnd) Re of correlation Sigma
 * @param ImSigma_c_skwij (ns, Nk, N_w, nbnd, nbnd) Im of correlation Sigma
 * @param eta           Lorentzian broadening
 * @param A_out_wskij   OUTPUT (N_w, ns, Nk, nbnd, nbnd) spectral function
 *                      (real part of A_out is -(1/pi) Im G^R; imag part is
 *                       -(1/pi) Re G^R, retained for diagnostic use)
 */
inline void dyson_update_A(real_freq_grid_t           const& grid,
                           nda::array<ComplexType, 4> const& H_MF_skij,
                           nda::array<ComplexType, 4> const& Sigma_x_skij,
                           nda::array<ComplexType, 5> const& ReSigma_c_skwij,
                           nda::array<ComplexType, 5> const& ImSigma_c_skwij,
                           double                            eta,
                           nda::array<ComplexType, 5>      & A_out_wskij)
{
  const long ns   = H_MF_skij.shape()[0];
  const long Nk   = H_MF_skij.shape()[1];
  const long nbnd = H_MF_skij.shape()[2];
  const long N_w  = grid.N_w();
  utils::check(H_MF_skij.shape()[3] == nbnd, "dyson_update_A: H_MF not square");
  utils::check(Sigma_x_skij.shape() == H_MF_skij.shape(),
               "dyson_update_A: Sigma_x shape");
  utils::check(A_out_wskij.shape()[0] == N_w and
               A_out_wskij.shape()[1] == ns and
               A_out_wskij.shape()[2] == Nk and
               A_out_wskij.shape()[3] == nbnd and
               A_out_wskij.shape()[4] == nbnd,
               "dyson_update_A: A_out shape mismatch");

  nda::array<ComplexType, 2> H(nbnd, nbnd), Sx(nbnd, nbnd);
  nda::array<ComplexType, 2> Re(nbnd, nbnd), Im(nbnd, nbnd);
  nda::array<ComplexType, 2> G(nbnd, nbnd), A(nbnd, nbnd);

  const double mu_chem = grid.mu_chem();
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k) {
      for (long mu_i = 0; mu_i < nbnd; ++mu_i)
        for (long nu = 0; nu < nbnd; ++nu) {
          H (mu_i, nu) = H_MF_skij(s, k, mu_i, nu);
          Sx(mu_i, nu) = Sigma_x_skij(s, k, mu_i, nu);
        }
      for (long iw = 0; iw < N_w; ++iw) {
        const double w = grid.w()(iw);
        for (long mu_i = 0; mu_i < nbnd; ++mu_i)
          for (long nu = 0; nu < nbnd; ++nu) {
            Re(mu_i, nu) = ReSigma_c_skwij(s, k, iw, mu_i, nu);
            Im(mu_i, nu) = ImSigma_c_skwij(s, k, iw, mu_i, nu);
          }
        dyson_G_one_kw(H, Sx, Re, Im, w, mu_chem, eta, G, A);
        for (long mu_i = 0; mu_i < nbnd; ++mu_i)
          for (long nu = 0; nu < nbnd; ++nu)
            A_out_wskij(iw, s, k, mu_i, nu) = A(mu_i, nu);
      }
    }
}

/**
 * Bisection root-find of mu_chem so that
 *
 *   sum_{s, k, mu} k_weight(k) * int dw f(w) A_{mu,mu}(s, k, w) = N_elec
 *
 * given a fixed spectral function A. Returns the new mu and updates
 * grid.mu_chem? — no, grid is const. Caller may use the returned value to
 * construct a new grid. Implementation is a bracketed bisection over a
 * generous initial interval [-w_max, +w_max].
 *
 * @param grid       grid (used for w-grid and weights, but NOT mu_chem)
 * @param A_wskij    spectral function (real part used as A)
 * @param k_weights  (Nk,) BZ weights
 * @param N_elec     target electron count
 * @param tol        tolerance on |N(mu) - N_elec|
 * @param max_iter   max bisection iterations
 * @return           mu_chem
 */
inline double find_mu_chem(real_freq_grid_t           const& grid,
                           nda::array<ComplexType, 5> const& A_wskij,
                           nda::array<double, 1>      const& k_weights,
                           double                            N_elec,
                           double                            tol = 1e-7,
                           long                              max_iter = 100)
{
  const long N_w  = A_wskij.shape()[0];
  const long ns   = A_wskij.shape()[1];
  const long Nk   = A_wskij.shape()[2];
  const long nbnd = A_wskij.shape()[3];

  auto N_of_mu = [&](double mu) -> double {
    double n = 0.0;
    for (long iw = 0; iw < N_w; ++iw) {
      const double f = real_freq_grid_t::fermi(grid.w()(iw), mu, grid.beta());
      const double dw = grid.w_weights()(iw);
      const double cf = f * dw;
      double trA = 0.0;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          double tr_sk = 0.0;
          for (long m = 0; m < nbnd; ++m)
            tr_sk += A_wskij(iw, s, k, m, m).real();
          trA += k_weights(k) * tr_sk;
        }
      n += cf * trA;
    }
    return n;
  };

  // Bracket: [-w_max, +w_max].
  const double mu_lo0 = grid.w()(0);
  const double mu_hi0 = grid.w()(N_w - 1);
  double mu_lo = mu_lo0, mu_hi = mu_hi0;
  double f_lo = N_of_mu(mu_lo) - N_elec;
  double f_hi = N_of_mu(mu_hi) - N_elec;
  utils::check(f_lo * f_hi <= 0.0,
               "find_mu_chem: target N_elec={} not bracketed in [N({})={}, N({})={}]",
               N_elec, mu_lo, f_lo + N_elec, mu_hi, f_hi + N_elec);

  for (long it = 0; it < max_iter; ++it) {
    const double mu_mid = 0.5 * (mu_lo + mu_hi);
    const double f_mid  = N_of_mu(mu_mid) - N_elec;
    if (std::abs(f_mid) < tol) return mu_mid;
    if (f_lo * f_mid <= 0.0) { mu_hi = mu_mid; f_hi = f_mid; }
    else                     { mu_lo = mu_mid; f_lo = f_mid; }
  }
  return 0.5 * (mu_lo + mu_hi);
}

/**
 * Apply causality projection in place on Im Sigma^c: clip positive .real()
 * components to zero on the diagonal (mu = nu).
 */
inline void project_causality_ImSigma(nda::array<ComplexType, 5> & ImSigma_skwij)
{
  const long ns   = ImSigma_skwij.shape()[0];
  const long Nk   = ImSigma_skwij.shape()[1];
  const long N_w  = ImSigma_skwij.shape()[2];
  const long nbnd = ImSigma_skwij.shape()[3];
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long m = 0; m < nbnd; ++m) {
          auto & v = ImSigma_skwij(s, k, iw, m, m);
          if (v.real() > 0.0) v = ComplexType(0.0, v.imag());
        }
}

/**
 * Linear mixing of two Sigma arrays in place:
 *   S_new = alpha * S_in + (1 - alpha) * S_old
 *   S_old <- S_new
 */
inline void linear_mix(nda::array<ComplexType, 5> & S_old,
                       nda::array<ComplexType, 5> const& S_in,
                       double alpha)
{
  utils::check(S_old.shape() == S_in.shape(),
               "linear_mix: shape mismatch");
  const long total = S_old.size();
  auto * d_old = S_old.data();
  auto const* d_in  = S_in.data();
  for (long i = 0; i < total; ++i)
    d_old[i] = alpha * d_in[i] + (1.0 - alpha) * d_old[i];
}

/**
 * Frobenius-norm difference between two arrays of the same shape.
 */
inline double frobenius_diff(nda::array<ComplexType, 5> const& A,
                             nda::array<ComplexType, 5> const& B)
{
  utils::check(A.shape() == B.shape(), "frobenius_diff: shape mismatch");
  const long total = A.size();
  auto const* da = A.data();
  auto const* db = B.data();
  double s = 0.0;
  for (long i = 0; i < total; ++i) {
    const auto d = da[i] - db[i];
    s += std::norm(d);
  }
  return std::sqrt(s);
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DYSON_G_HPP
