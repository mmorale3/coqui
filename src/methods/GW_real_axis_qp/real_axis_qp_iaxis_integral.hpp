/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Imaginary-axis convolution kernel for the QP-form QSGW / CD framework.
 *
 *   J_n(ω) = (1/2π) ∫_{-∞}^{+∞} dω'  W_c(iω')  /  (ω - ε_n^QP + i ω')
 *
 * The integrand is smooth on the imaginary axis (no poles since
 * ε_n^QP is real). Standard Gauss-Legendre or trapezoidal quadrature
 * suffices for a few-percent accuracy with N_iω ~ 16-32 nodes.
 *
 * The full Σ_c^{iaxis}_{ij}(s, k, ω) contribution is:
 *
 *   Σ_c^{iaxis}_{ij}(s, k, ω) = Σ_n MO_{i,n} MO*_{j,n} J_n(s, k, ω)
 *
 * where MO are QP orbitals at (s, k_ibz) and the inner J_n is itself the
 * convolution above in aux basis:
 *
 *   J_n^{aux}_{P,Q}(s, k, q, ω) = (1/2π) Σ_l w_l W_c_aux^{PQ}(q, iω'_l)
 *                                  / (ω - ε_n^QP(s, k+q) + i ω'_l)
 *
 * where w_l are quadrature weights for the iω' integration.
 *
 * In this header we provide the SCALAR variant (single (n, ω) pair, one
 * (P, Q) channel) and a BATCHED variant that loops over (P, Q).
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_QP_IAXIS_INTEGRAL_HPP
#define COQUI_REAL_AXIS_QP_IAXIS_INTEGRAL_HPP

#include <complex>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis_qp {

using ComplexType = std::complex<double>;

/**
 * Scalar iaxis-integral evaluation.
 *
 * @param iw_nodes    quadrature nodes ω'_l (real) on the imaginary axis.
 *                    Both positive and negative ω' should be included
 *                    (the integral spans [-∞, +∞] symmetrically).
 * @param iw_weights  quadrature weights w_l (real) — typically
 *                    Gauss-Legendre on a finite [-Ω_iω_max, +Ω_iω_max] segment
 *                    with the (1/2π) factor folded in already.
 * @param W_c_iw      complex values W_c(iω'_l) at the nodes; shape (N_iω,).
 * @param omega       target real frequency Ω (the (ε_n^QP, ω) appears as
 *                    arg_pole = ω - ε_n^QP).
 * @param eps_pole    ε_n^QP value (real).
 * @return            J_n(ω) (complex).
 */
inline ComplexType
iaxis_integral_scalar(std::vector<double> const& iw_nodes,
                      std::vector<double> const& iw_weights,
                      std::vector<ComplexType> const& W_c_iw,
                      double omega,
                      double eps_pole)
{
  const std::size_t Nq = iw_nodes.size();
  utils::check(iw_weights.size() == Nq and W_c_iw.size() == Nq and Nq >= 1,
               "iaxis_integral_scalar: bad sizes.");
  const double arg = omega - eps_pole;
  ComplexType J(0.0, 0.0);
  for (std::size_t l = 0; l < Nq; ++l) {
    // 1 / (arg + i ω'_l)
    const ComplexType denom = ComplexType(arg, iw_nodes[l]);
    J += W_c_iw[l] / denom * iw_weights[l];
  }
  return J;
}

/**
 * Batched iaxis-integral for a full (P, Q) channel block.
 *
 * @param iw_nodes      shape (N_iω,) real
 * @param iw_weights    shape (N_iω,) real
 * @param W_c_iw_PQ    shape (NP, NQ, N_iω) complex (per-channel W_c on iω axis)
 * @param omega         scalar real
 * @param eps_pole      scalar real
 * @param out           shape (NP, NQ) complex; written with J_n(omega) per channel.
 */
inline void iaxis_integral_batched(
    nda::array<double, 1> const& iw_nodes,
    nda::array<double, 1> const& iw_weights,
    nda::array<ComplexType, 3> const& W_c_iw_PQ,
    double omega,
    double eps_pole,
    nda::array<ComplexType, 2>& out)
{
  const long Nq = iw_nodes.shape()[0];
  const long NP = W_c_iw_PQ.shape()[0];
  const long NQ = W_c_iw_PQ.shape()[1];
  utils::check(W_c_iw_PQ.shape()[2] == Nq and iw_weights.shape()[0] == Nq,
               "iaxis_integral_batched: N_iw mismatch.");
  utils::check(out.shape()[0] == NP and out.shape()[1] == NQ,
               "iaxis_integral_batched: out shape mismatch.");

  const double arg = omega - eps_pole;
  out() = ComplexType(0.0, 0.0);
  for (long l = 0; l < Nq; ++l) {
    const ComplexType denom = ComplexType(arg, iw_nodes(l));
    const ComplexType inv = ComplexType(1.0, 0.0) / denom;
    const ComplexType w_inv = ComplexType(iw_weights(l), 0.0) * inv;
    // out += w_inv * W_c_iw(:, :, l)
    for (long P = 0; P < NP; ++P)
      for (long Q = 0; Q < NQ; ++Q)
        out(P, Q) += w_inv * W_c_iw_PQ(P, Q, l);
  }
}

/**
 * Build a Gauss-Legendre mesh on a finite interval [-Ω_iω_max, +Ω_iω_max]
 * with the (1/2π) prefactor folded into the weights. Useful for testing.
 *
 * For production, the caller will typically provide the IAFT mesh from
 * numerics/imag_axes_ft/IAFT.hpp; this helper is for unit tests of the
 * kernels in isolation.
 */
inline std::pair<std::vector<double>, std::vector<double>>
make_gauss_legendre_iw_mesh(double iw_max, long N_iw)
{
  utils::check(N_iw >= 2 and (N_iw % 2 == 0),
               "make_gauss_legendre_iw_mesh: N_iw must be even and >= 2.");
  utils::check(iw_max > 0.0, "make_gauss_legendre_iw_mesh: iw_max > 0.");
  // Tiny GL implementation: Newton on Legendre polynomials in [-1, 1],
  // then rescale to [-iw_max, +iw_max].
  std::vector<double> x(N_iw), w(N_iw);
  const long N = N_iw;
  for (long i = 0; i < (N + 1) / 2; ++i) {
    double xi = std::cos(M_PI * (i + 0.75) / (N + 0.5));
    double dx = 1.0;
    for (long it = 0; it < 50 and std::abs(dx) > 1e-15; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (long k = 1; k <= N; ++k) {
        double p2 = p1; p1 = p0;
        p0 = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p2) / k;
      }
      double dp = N * (xi * p0 - p1) / (xi * xi - 1.0);
      dx = p0 / dp;
      xi -= dx;
    }
    double p0 = 1.0, p1 = 0.0;
    for (long k = 1; k <= N; ++k) {
      double p2 = p1; p1 = p0;
      p0 = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p2) / k;
    }
    double dp = N * (xi * p0 - p1) / (xi * xi - 1.0);
    x[i] = -xi;
    x[N - 1 - i] = xi;
    double wi = 2.0 / ((1.0 - xi * xi) * dp * dp);
    w[i] = wi;
    w[N - 1 - i] = wi;
  }
  // Rescale to [-iw_max, +iw_max] and fold in 1/(2π).
  for (long i = 0; i < N; ++i) {
    x[i] *= iw_max;
    w[i] *= iw_max / (2.0 * M_PI);
  }
  return {std::move(x), std::move(w)};
}

} // namespace real_axis_qp
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_IAXIS_INTEGRAL_HPP
