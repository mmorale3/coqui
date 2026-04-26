/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_AXIS_CONV_HPP
#define COQUI_REAL_AXIS_REAL_AXIS_CONV_HPP

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "numerics/fft/finufft_define.hpp"
#include "numerics/fft/finufft_nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * NUFFT-based convolution and Hilbert-transform engine for real-axis GW.
 *
 * The grid pair consists of a non-uniform frequency grid {w_j} and a uniform
 * conjugate "time" grid {t_k = (k - N_t/2) * dt, k=0..N_t-1, dt=T_window/N_t}.
 * For each non-empty frequency grid we hold a pair of pre-built FINUFFT plans
 * (one type-1, one type-2). Plans are bound to "scaled" coordinates
 *   x_j = w_j * dt
 * which lie in [-pi, pi] by the Nyquist condition enforced in real_freq_grid_t.
 *
 * Two primitive operations are exposed:
 *
 *   cross_correlate(F, G, H, src_grid, dst_grid):
 *       H(Omega) = (dt / (2*pi)) * NUFFT2( conj(F_hat) * G_hat )
 *     where F_hat(t) and G_hat(t) are the type-1 NUFFTs of the (weighted)
 *     spectra F(w) and G(w) on src_grid, multiplied pointwise in time, and
 *     transformed back via type-2 NUFFT to dst_grid.
 *
 *   hilbert(ImX, ReX, grid):
 *       ReX(w) = -(dt / (2*pi)) * i * NUFFT2( sgn(t) * NUFFT1(ImX) )
 *     i.e. the Kramers-Kronig relation
 *       Re X(w) = (1/pi) PV int dw' Im X(w')/(w'-w)
 *     evaluated as multiplication by -i*sgn(t) in the conjugate time variable.
 *
 * Both primitives accept additional pre-multipliers on F, G, ImX (e.g. fermi
 * factors and quadrature weights). The caller is responsible for absorbing
 * those into the inputs before calling.
 *
 * The engine is single-threaded at the FINUFFT level (the underlying nuplan_t
 * uses the FINUFFT internal threading). Higher-level parallelism over
 * (k, q) batches is the responsibility of the caller.
 */
class real_axis_conv_t {
public:

  using cval_t = std::complex<double>;

  /// Frequency-grid identifier: which set of FINUFFT plans to use.
  enum class grid_kind { fermionic, bosonic };

  /// Construct from a real_freq_grid_t. Stores a non-owning reference to it.
  /// Builds two type-1 plans (fermionic, bosonic) and two type-2 plans
  /// targeting both grids each. The total memory cost is dominated by the
  /// FINUFFT internal interpolation tables.
  ///
  /// The "many" template form is used so that users can batch transforms
  /// over auxiliary indices (P, Q) or band indices (mu, nu) at runtime via
  /// the ntrans argument supplied to the per-call helpers below.
  ///
  /// @param grid    real-frequency grid container
  /// @param ntrans  maximum simultaneous transforms; must be >= the largest
  ///                "leading" dimension passed to cross_correlate / hilbert.
  /// @param eps     FINUFFT accuracy tolerance
  real_axis_conv_t(real_freq_grid_t const& grid, long ntrans = 1,
                   double eps = 1e-10)
    : _grid(&grid)
    , _ntrans(ntrans)
    , _eps(eps)
  {
    utils::check(_ntrans >= 1, "real_axis_conv_t: ntrans must be >= 1.");

    const long N_t = grid.N_t();
    const long N_w = grid.N_w();
    const long N_Omega = grid.N_Omega();
    const double dt = grid.dt();

    // Scaled coordinates x = w * dt for FINUFFT (must lie in [-pi, pi]).
    _x_w = nda::array<double,1>(N_w);
    for (long j = 0; j < N_w; ++j)
      _x_w(j) = grid.w()(j) * dt;
    _x_Omega = nda::array<double,1>(N_Omega);
    for (long l = 0; l < N_Omega; ++l)
      _x_Omega(l) = grid.Omega()(l) * dt;

    // Plans use iflag=+1: type-1 sums  exp(+i * x_j * k_int) per finufft conv
    // (where k_int is the integer mode index that we re-interpret as t_k).
    // The mathematical interpretation is f_hat(t_k) = sum_j c_j exp(+i*w_j*t_k)
    // under the identification t_k = k_int * dt (after the symmetric shift).
    const std::array<int64_t,1> nm = { N_t };

    _plan_w = std::make_unique<math::nda::nufft>(
        nm, N_w, _ntrans, _eps, math::nufft::NUFFT_FORWARD);
    _plan_w->setpts(_x_w);

    _plan_Omega = std::make_unique<math::nda::nufft>(
        nm, N_Omega, _ntrans, _eps, math::nufft::NUFFT_FORWARD);
    _plan_Omega->setpts(_x_Omega);
  }

  long N_t()      const { return _grid->N_t(); }
  long N_w()      const { return _grid->N_w(); }
  long N_Omega()  const { return _grid->N_Omega(); }
  long ntrans()   const { return _ntrans; }
  real_freq_grid_t const& grid() const { return *_grid; }

  /**
   * Compute, for ntrans batches independently:
   *   H(Omega_l) = (dt / (2*pi)) * sum_k F_hat^*(t_k) G_hat(t_k) exp(-i Omega_l t_k)
   *
   * Mathematically:  H(Omega) = int dw F^*(w) G(w + Omega) (the cross-correlation).
   *
   * Frequency grids: F and G live on the SAME source grid (`src_grid`); H
   * lives on `dst_grid` (typically `bosonic` for polarization and
   * `fermionic` for self-energy).
   *
   * Layout convention: all batched arrays are C_layout with leading batch
   * dimension. F has shape (B, src_grid.size()), H has shape (B, dst.size()).
   *
   * Caller is responsible for absorbing weights into F (and into G if
   * needed). The factor (dt / (2 pi)) and the conjugate are applied here.
   *
   * @param F    [INPUT]  shape (B, N_src), already includes quadrature weights
   * @param G    [INPUT]  shape (B, N_src), bare values
   * @param H    [OUTPUT] shape (B, N_dst)
   * @param src  source frequency grid kind
   * @param dst  destination frequency grid kind
   */
  void cross_correlate(nda::array<cval_t,2> const& F_in,
                       nda::array<cval_t,2> const& G_in,
                       nda::array<cval_t,2> & H,
                       grid_kind src,
                       grid_kind dst)
  {
    const long B    = F_in.shape()[0];
    const long N_src = (src == grid_kind::fermionic ? N_w() : N_Omega());
    const long N_dst = (dst == grid_kind::fermionic ? N_w() : N_Omega());
    const long N_t_  = N_t();

    utils::check(F_in.shape()[0] == B and F_in.shape()[1] == N_src,
                 "cross_correlate: F shape mismatch");
    utils::check(G_in.shape()[0] == B and G_in.shape()[1] == N_src,
                 "cross_correlate: G shape mismatch");
    utils::check(H.shape()[0] == B and H.shape()[1] == N_dst,
                 "cross_correlate: H shape mismatch");
    utils::check(B <= _ntrans,
                 "cross_correlate: B={} > ntrans={}; rebuild engine with larger ntrans",
                 B, _ntrans);

    // Make weighted local copies. Both F and G must carry quadrature weights
    // because each is independently FT'd (the cross-correlation is a single
    // integral over w but the Fourier-space implementation transforms each
    // factor separately and multiplies in conjugate-time space).
    auto const& wq = (src == grid_kind::fermionic
                      ? _grid->w_weights() : _grid->Omega_weights());
    nda::array<cval_t,2> F(B, N_src), G(B, N_src);
    for (long b = 0; b < B; ++b)
      for (long j = 0; j < N_src; ++j) {
        F(b, j) = F_in(b, j) * wq(j);
        G(b, j) = G_in(b, j) * wq(j);
      }

    // Fhat, Ghat live on the uniform t-grid. Allocate per-call (small).
    nda::array<cval_t,2> Fhat(B, N_t_);
    nda::array<cval_t,2> Ghat(B, N_t_);

    // For ntrans larger than B, we still need to provide a B-row plan.
    // FINUFFT supports passing data with the "many" plan if the trailing
    // batch dimension matches; for simplicity we rebuild a sized plan when
    // B != _ntrans. (Most call sites should size _ntrans to B once.)
    auto run_type1 = [&](nda::array<cval_t,2>& C, nda::array<cval_t,2>& F_out,
                         grid_kind kind) {
      if (B == _ntrans) {
        if (kind == grid_kind::fermionic) _plan_w->forward(C, F_out);
        else                              _plan_Omega->forward(C, F_out);
      } else {
        const std::array<int64_t,1> nm = { N_t_ };
        const long N_pts = (kind == grid_kind::fermionic ? N_w() : N_Omega());
        math::nda::nufft tmp(nm, N_pts, B, _eps, math::nufft::NUFFT_FORWARD);
        if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
        else                              tmp.setpts(_x_Omega);
        tmp.forward(C, F_out);
      }
    };

    auto run_type2 = [&](nda::array<cval_t,2>& F_in, nda::array<cval_t,2>& C_out,
                         grid_kind kind) {
      const std::array<int64_t,1> nm = { N_t_ };
      const long N_pts = (kind == grid_kind::fermionic ? N_w() : N_Omega());
      math::nda::nufft tmp(nm, N_pts, B, _eps, math::nufft::NUFFT_FORWARD);
      if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
      else                              tmp.setpts(_x_Omega);
      tmp.backward(F_in, C_out);
    };

    run_type1(F, Fhat, src);
    run_type1(G, Ghat, src);

    // H_hat(t_k) = conj(Fhat(t_k)) * Ghat(t_k), with the time-shift phase
    // factor that compensates the symmetric shift of the t-grid.
    // Internal NUFFT mode index k_int = 0..N_t-1 corresponds to physical
    // t_k = (k_int - N_t/2) * dt. The forward NUFFT yields
    //   sum_j c_j exp(+i * x_j * k_int) = sum_j c_j exp(+i*w_j*(k_int*dt))
    // i.e., F_hat at virtual time k_int*dt, NOT at the shifted t_k.
    // We absorb the shift e^{+i w_max_shift t} after the type-2 to recover
    // the physical-time interpretation. Equivalently, multiply Fhat and Ghat
    // by exp(+i * w_offset * t) factors in time -- but since we only need
    // their product, the time-shift phase factors cancel for cross_correlate
    // when src==same. We therefore apply NO shift here for cross_correlate.
    nda::array<cval_t,2> Hhat(B, N_t_);
    for (long b = 0; b < B; ++b)
      for (long k = 0; k < N_t_; ++k)
        Hhat(b, k) = std::conj(Fhat(b, k)) * Ghat(b, k);

    nda::array<cval_t,2> Hraw(B, N_dst);
    run_type2(Hhat, Hraw, dst);

    // Final scaling: dt/(2*pi).  This converts the discrete sum
    //   (1/N_t) sum_k Hhat_k exp(-i Omega t_k)  ~  (1/(2pi)) int dt
    // when N_t * dt = T_window, so dt = 2pi/(N_t * (2pi/T)) and the
    // appropriate trapezoidal weight per t-sample is dt. The 1/(2pi) is
    // the inverse-FT normalization. We do NOT divide by N_t.
    const double s = _grid->dt() / (2.0 * M_PI);
    H = s * Hraw;
  }

  /**
   * Hilbert transform via NUFFT.
   *
   *   Re X(w) = (1/pi) PV int dw' Im X(w') / (w' - w)
   *           = -(dt / (2*pi)) i sum_k sgn(t_k) NUFFT1(weighted ImX)(t_k)
   *                                    exp(-i w t_k)
   *
   * Caller must already have multiplied ImX by the trapezoidal quadrature
   * weights for the source grid.
   *
   * @param ImX  [INPUT] real-valued imaginary part, shape (B, N_grid),
   *              ALREADY multiplied by quadrature weights.
   * @param ReX  [OUTPUT] real-valued real part, shape (B, N_grid).
   * @param kind frequency grid (same source and destination).
   */
  void hilbert(nda::array<double,2> const& ImX_in,
               nda::array<double,2> & ReX_w,
               grid_kind kind)
  {
    const long B = ImX_in.shape()[0];
    const long N_grid = (kind == grid_kind::fermionic ? N_w() : N_Omega());
    const long N_t_ = N_t();

    utils::check(ImX_in.shape()[1] == N_grid,
                 "hilbert: ImX shape mismatch (got {}, expected {})",
                 ImX_in.shape()[1], N_grid);
    utils::check(ReX_w.shape()[0] == B and ReX_w.shape()[1] == N_grid,
                 "hilbert: ReX shape mismatch");

    // Promote real input to complex for the NUFFT, applying quadrature
    // weights as we go (the Hilbert transform is an integral over w').
    auto const& wq = (kind == grid_kind::fermionic
                      ? _grid->w_weights() : _grid->Omega_weights());
    nda::array<cval_t,2> C(B, N_grid);
    for (long b = 0; b < B; ++b)
      for (long j = 0; j < N_grid; ++j)
        C(b, j) = cval_t(ImX_in(b, j) * wq(j), 0.0);

    nda::array<cval_t,2> Chat(B, N_t_);
    {
      const std::array<int64_t,1> nm = { N_t_ };
      math::nda::nufft tmp(nm, N_grid, B, _eps, math::nufft::NUFFT_FORWARD);
      if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
      else                              tmp.setpts(_x_Omega);
      tmp.forward(C, Chat);
    }

    // Multiply by -i * sgn(t_k). The internal mode index k_int = 0..N_t-1
    // corresponds to physical t_k = (k_int - N_t/2) * dt. Therefore
    // sgn(t_k) is positive for k_int > N_t/2, negative for k_int < N_t/2,
    // and the boundary value at k_int == N_t/2 (t=0) is set to 0. But
    // FINUFFT internally indexes from -N_t/2 to N_t/2-1, NOT 0..N_t-1.
    // We need to be careful here: the output Chat(b, k_int) corresponds to
    // mode index k = k_int - N_t/2 in finufft's convention, which IS the
    // physical t_k = k * dt. So sgn(t_k) is positive for k_int >= N_t/2 + 1,
    // zero at k_int == N_t/2, and negative for k_int < N_t/2.
    // Wait: with a symmetric grid t_k = (k_int - N_t/2) * dt, the value at
    // k_int = N_t/2 is t=0 with sgn=0; for k_int < N_t/2, t<0 (negative),
    // sgn = -1; for k_int > N_t/2, t>0, sgn = +1.
    // Hilbert kernel multiplier: with our convention F̂(t) = ∫f(w)exp(+iwt)dw,
    // (Hf)^(t) = +i sgn(t) F̂(t), where t_k = (k - N_t/2) * dt.
    nda::array<cval_t,2> Hhat(B, N_t_);
    for (long b = 0; b < B; ++b) {
      for (long k = 0; k < N_t_; ++k) {
        double s_k;
        if (k > N_t_ / 2)      s_k = +1.0;
        else if (k < N_t_ / 2) s_k = -1.0;
        else                   s_k =  0.0;
        Hhat(b, k) = cval_t(0.0, +s_k) * Chat(b, k);
      }
    }

    nda::array<cval_t,2> Rraw(B, N_grid);
    {
      const std::array<int64_t,1> nm = { N_t_ };
      math::nda::nufft tmp(nm, N_grid, B, _eps, math::nufft::NUFFT_FORWARD);
      if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
      else                              tmp.setpts(_x_Omega);
      tmp.backward(Hhat, Rraw);
    }

    const double s = _grid->dt() / (2.0 * M_PI);
    for (long b = 0; b < B; ++b)
      for (long j = 0; j < N_grid; ++j)
        ReX_w(b, j) = s * Rraw(b, j).real();
  }

  /**
   * Helper: multiply each row of arr by the source-grid quadrature weights
   * in-place. Convenience for callers preparing inputs to cross_correlate
   * or hilbert.
   */
  void apply_weights(nda::array<cval_t,2>& arr, grid_kind kind) const {
    auto const& w = (kind == grid_kind::fermionic
                     ? _grid->w_weights() : _grid->Omega_weights());
    const long N = arr.shape()[1];
    utils::check(w.shape()[0] == N, "apply_weights: shape mismatch");
    for (long b = 0; b < arr.shape()[0]; ++b)
      for (long j = 0; j < N; ++j)
        arr(b, j) *= w(j);
  }

  void apply_weights(nda::array<double,2>& arr, grid_kind kind) const {
    auto const& w = (kind == grid_kind::fermionic
                     ? _grid->w_weights() : _grid->Omega_weights());
    const long N = arr.shape()[1];
    utils::check(w.shape()[0] == N, "apply_weights: shape mismatch");
    for (long b = 0; b < arr.shape()[0]; ++b)
      for (long j = 0; j < N; ++j)
        arr(b, j) *= w(j);
  }

private:
  real_freq_grid_t const* _grid;
  long                    _ntrans;
  double                  _eps;
  nda::array<double,1>    _x_w;       // scaled fermionic coords w_j * dt
  nda::array<double,1>    _x_Omega;   // scaled bosonic coords Omega_l * dt
  std::unique_ptr<math::nda::nufft> _plan_w;
  std::unique_ptr<math::nda::nufft> _plan_Omega;
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_CONV_HPP
