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

/// Frequency-grid identifier: which set of (cu)FINUFFT plans to use. Lifted
/// to namespace level since it's independent of the conv-engine memory space.
enum class grid_kind { fermionic, bosonic };

namespace detail {

/**
 * NUFFT-based convolution and Hilbert-transform engine for real-axis GW.
 *
 * Templated on MEMORY_SPACE MEM. The internal storage and the underlying
 * (cu)FINUFFT plan live in MEM. For MEM == HOST_MEMORY (the default via the
 * `real_axis_conv_t` alias) the engine uses FINUFFT and host arrays; for
 * MEM == DEVICE_MEMORY it uses cuFINUFFT and device (cuarray) storage.
 *
 * Existing callers that name `real_axis_conv_t` directly continue to bind
 * to the host instantiation. Future device callers should use
 * `real_axis_conv_mem_t<DEVICE_MEMORY>`.
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
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_conv_base_t {
public:

  static constexpr MEMORY_SPACE memory_space = MEM;

  using cval_t = std::complex<double>;
  template<int N> using array_t      = memory::array<MEM, cval_t, N>;
  template<int N> using rarray_t     = memory::array<MEM, double, N>;
  using plan_t                       = math::nda::nufft_t<MEM>;
  // grid_kind is the namespace-level enum, re-exposed here for backward
  // compatibility with the previous nested-enum API
  // `real_axis_conv_t::grid_kind::fermionic`.
  using grid_kind                    = methods::real_axis::grid_kind;

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
  real_axis_conv_base_t(real_freq_grid_t const& grid, long ntrans = 1,
                        double eps = 1e-10)
    : _grid(&grid)
    , _ntrans(ntrans)
    , _eps(eps)
  {
    utils::check(_ntrans >= 1, "real_axis_conv_base_t: ntrans must be >= 1.");

    const long N_t = grid.N_t();
    const long N_w = grid.N_w();
    const long N_Omega = grid.N_Omega();
    const double dt = grid.dt();

    // Scaled coordinates x = w * dt for (cu)FINUFFT (must lie in [-pi, pi]).
    // Built on host first, then promoted to MEM (a no-op for HOST_MEMORY).
    nda::array<double, 1> x_w_host(N_w);
    for (long j = 0; j < N_w; ++j) x_w_host(j) = grid.w()(j) * dt;
    nda::array<double, 1> x_O_host(N_Omega);
    for (long l = 0; l < N_Omega; ++l) x_O_host(l) = grid.Omega()(l) * dt;

    if constexpr (MEM == HOST_MEMORY) {
      _x_w = std::move(x_w_host);
      _x_Omega = std::move(x_O_host);
    } else {
      _x_w     = memory::to_memory_space<MEM>(x_w_host);
      _x_Omega = memory::to_memory_space<MEM>(x_O_host);
    }

    // Plans use iflag=+1: type-1 sums  exp(+i * x_j * k_int) per finufft conv
    // (where k_int is the integer mode index that we re-interpret as t_k).
    const std::array<int64_t,1> nm = { N_t };

    _plan_w = std::make_unique<plan_t>(
        nm, N_w, _ntrans, _eps, math::nufft::NUFFT_FORWARD);
    _plan_w->setpts(_x_w);

    _plan_Omega = std::make_unique<plan_t>(
        nm, N_Omega, _ntrans, _eps, math::nufft::NUFFT_FORWARD);
    _plan_Omega->setpts(_x_Omega);

    // Pre-build the Hilbert-kernel sgn(t_k) array as i*sgn(t_k) in MEM.
    // The internal mode index k_int = 0..N_t-1 corresponds to physical
    // t_k = (k_int - N_t/2) * dt, so sgn(t_k) = +1 for k_int > N_t/2,
    // -1 for k_int < N_t/2, and 0 at k_int = N_t/2.
    nda::array<cval_t, 1> sgn_host(N_t);
    for (long k = 0; k < N_t; ++k) {
      double s = (k > N_t/2 ? +1.0 : (k < N_t/2 ? -1.0 : 0.0));
      sgn_host(k) = cval_t(0.0, s);   // i*sgn(t_k)
    }
    if constexpr (MEM == HOST_MEMORY)
      _sgn_t = std::move(sgn_host);
    else
      _sgn_t = memory::to_memory_space<MEM>(sgn_host);
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
  void cross_correlate(array_t<2> const& F_in,
                       array_t<2> const& G_in,
                       array_t<2> & H,
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
                 "cross_correlate: B={} > ntrans={}", B, _ntrans);

    if constexpr (MEM != HOST_MEMORY) {
      // TODO: device kernels for weight application + Hadamard product.
      // The host body below uses straight 2D loops; on device this becomes
      // either a per-row gemv-style scaling + per-element conj()*mul, or a
      // single fused kernel. The (cu)FINUFFT plan calls already work in
      // both spaces.
      utils::check(false,
                   "real_axis_conv_base_t<DEVICE>::cross_correlate: device "
                   "kernels for the inner Hadamard loops are not yet implemented.");
    } else {
      auto const& wq = (src == grid_kind::fermionic
                        ? _grid->w_weights() : _grid->Omega_weights());
      array_t<2> F(B, N_src), G(B, N_src);
      for (long b = 0; b < B; ++b)
        for (long j = 0; j < N_src; ++j) {
          F(b, j) = F_in(b, j) * wq(j);
          G(b, j) = G_in(b, j) * wq(j);
        }

      array_t<2> Fhat(B, N_t_);
      array_t<2> Ghat(B, N_t_);
      run_forward(F, Fhat, src, B);
      run_forward(G, Ghat, src, B);

      // 2-arg Hadamard, MEM-agnostic via nda::map.
      array_t<2> Hhat(B, N_t_);
      Hhat = nda::map([](cval_t f, cval_t g) { return std::conj(f) * g; })(Fhat, Ghat);

      array_t<2> Hraw(B, N_dst);
      run_backward(Hhat, Hraw, dst, B);

      const double s = _grid->dt() / (2.0 * M_PI);
      H = s * Hraw;
    }
  }

  /**
   * Convolution H(w) = int de F(e) G(w - e), with F, G, H all on the SAME
   * frequency grid (`kind`). Identical NUFFT structure to `cross_correlate`
   * but WITHOUT the conjugate on F_hat: Hhat(t) = F_hat(t) * G_hat(t).
   *
   * Quadrature weights on the source grid are applied internally to BOTH
   * inputs (each is independently FT'd).
   *
   * @param F_in  (B, N_grid) input F (no weights pre-applied)
   * @param G_in  (B, N_grid) input G (no weights pre-applied)
   * @param H     (B, N_grid) output H, OVERWRITTEN
   * @param kind  frequency grid (same source and destination)
   */
  void convolve(array_t<2> const& F_in,
                array_t<2> const& G_in,
                array_t<2> & H,
                grid_kind kind)
  {
    const long B    = F_in.shape()[0];
    const long N    = (kind == grid_kind::fermionic ? N_w() : N_Omega());
    const long N_t_ = N_t();

    utils::check(F_in.shape()[0] == B and F_in.shape()[1] == N,
                 "convolve: F shape mismatch");
    utils::check(G_in.shape()[0] == B and G_in.shape()[1] == N,
                 "convolve: G shape mismatch");
    utils::check(H.shape()[0] == B and H.shape()[1] == N,
                 "convolve: H shape mismatch");
    utils::check(B <= _ntrans,
                 "convolve: B={} > ntrans={}", B, _ntrans);

    if constexpr (MEM != HOST_MEMORY) {
      utils::check(false,
                   "real_axis_conv_base_t<DEVICE>::convolve: device kernels "
                   "for the inner Hadamard loops are not yet implemented.");
    } else {
      auto const& wq = (kind == grid_kind::fermionic
                        ? _grid->w_weights() : _grid->Omega_weights());
      array_t<2> F(B, N), G(B, N);
      for (long b = 0; b < B; ++b)
        for (long j = 0; j < N; ++j) {
          F(b, j) = F_in(b, j) * wq(j);
          G(b, j) = G_in(b, j) * wq(j);
        }

      array_t<2> Fhat(B, N_t_), Ghat(B, N_t_);
      run_forward(F, Fhat, kind, B);
      run_forward(G, Ghat, kind, B);

      // Convolution Hadamard: NO conjugate. MEM-agnostic via nda::map.
      array_t<2> Hhat(B, N_t_);
      Hhat = nda::map([](cval_t f, cval_t g) { return f * g; })(Fhat, Ghat);

      array_t<2> Hraw(B, N);
      run_backward(Hhat, Hraw, kind, B);
      const double s = _grid->dt() / (2.0 * M_PI);
      H = s * Hraw;
    }
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
  void hilbert(rarray_t<2> const& ImX_in,
               rarray_t<2> & ReX_w,
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

    if constexpr (MEM != HOST_MEMORY) {
      utils::check(false,
                   "real_axis_conv_base_t<DEVICE>::hilbert: device kernels "
                   "for the inner sgn(t)-multiply and weight-application "
                   "loops are not yet implemented. _sgn_t is already in "
                   "MEM-space, so the device port is mostly a single fused "
                   "kernel: Hhat(b, k) = _sgn_t(k) * Chat(b, k).");
    } else {
      auto const& wq = (kind == grid_kind::fermionic
                        ? _grid->w_weights() : _grid->Omega_weights());
      array_t<2> C(B, N_grid);
      for (long b = 0; b < B; ++b)
        for (long j = 0; j < N_grid; ++j)
          C(b, j) = cval_t(ImX_in(b, j) * wq(j), 0.0);

      array_t<2> Chat(B, N_t_);
      run_forward(C, Chat, kind, B);

      array_t<2> Hhat(B, N_t_);
      for (long b = 0; b < B; ++b)
        for (long k = 0; k < N_t_; ++k)
          Hhat(b, k) = _sgn_t(k) * Chat(b, k);

      array_t<2> Rraw(B, N_grid);
      run_backward(Hhat, Rraw, kind, B);

      const double s = _grid->dt() / (2.0 * M_PI);
      for (long b = 0; b < B; ++b)
        for (long j = 0; j < N_grid; ++j)
          ReX_w(b, j) = s * Rraw(b, j).real();
    }
  }

  /**
   * Type-1 NUFFT: nonuniform-frequency strengths C → uniform-time modes F.
   * Public wrapper around the cached plan; falls back to a fresh per-call
   * plan when B != _ntrans (set ntrans = B at construction to avoid).
   *
   * @param C   [INPUT, weights NOT applied internally]  (B, N_grid)
   * @param F   [OUTPUT]  (B, N_t)
   * @param kind  source grid (fermionic or bosonic)
   */
  void forward(array_t<2>& C, array_t<2>& F, grid_kind kind) {
    run_forward(C, F, kind, C.shape()[0]);
  }

  /// Type-2 NUFFT: uniform-time modes F → nonuniform-frequency values C.
  void backward(array_t<2>& F, array_t<2>& C, grid_kind kind) {
    run_backward(F, C, kind, F.shape()[0]);
  }

  /// dt / (2*pi) — the trapezoidal-quadrature × inverse-FT normalization
  /// applied to NUFFT2 outputs in cross_correlate / convolve / hilbert.
  double nufft_scale() const { return _grid->dt() / (2.0 * M_PI); }

  /**
   * Helper: multiply each row of arr by the source-grid quadrature weights
   * in-place. Convenience for callers preparing inputs to cross_correlate
   * or hilbert.
   */
  void apply_weights(array_t<2>& arr, grid_kind kind) const {
    if constexpr (MEM != HOST_MEMORY) {
      utils::check(false,
                   "real_axis_conv_base_t<DEVICE>::apply_weights: device "
                   "kernel for 1D weights * 2D array broadcast not yet implemented.");
    } else {
      auto const& w = (kind == grid_kind::fermionic
                       ? _grid->w_weights() : _grid->Omega_weights());
      const long N = arr.shape()[1];
      utils::check(w.shape()[0] == N, "apply_weights: shape mismatch");
      for (long b = 0; b < arr.shape()[0]; ++b)
        for (long j = 0; j < N; ++j)
          arr(b, j) *= w(j);
    }
  }

  void apply_weights(rarray_t<2>& arr, grid_kind kind) const {
    if constexpr (MEM != HOST_MEMORY) {
      utils::check(false,
                   "real_axis_conv_base_t<DEVICE>::apply_weights(double): "
                   "device kernel for 1D weights * 2D array broadcast not yet implemented.");
    } else {
      auto const& w = (kind == grid_kind::fermionic
                       ? _grid->w_weights() : _grid->Omega_weights());
      const long N = arr.shape()[1];
      utils::check(w.shape()[0] == N, "apply_weights: shape mismatch");
      for (long b = 0; b < arr.shape()[0]; ++b)
        for (long j = 0; j < N; ++j)
          arr(b, j) *= w(j);
    }
  }

private:
  // Type-1 (nonuniform -> uniform). Uses cached plans when B == _ntrans;
  // builds a fresh plan otherwise. Both type-1 and type-2 plans share the
  // same nuplan_t and setpts, so when we build a fresh "many" plan we get
  // both directions for free; the helper is parameterised on the call
  // pattern to keep cross_correlate / convolve / hilbert call sites flat.
  void run_forward(array_t<2>& C, array_t<2>& F_out, grid_kind kind, long B) {
    if (B == _ntrans) {
      if (kind == grid_kind::fermionic) _plan_w->forward(C, F_out);
      else                              _plan_Omega->forward(C, F_out);
      return;
    }
    const std::array<int64_t,1> nm = { _grid->N_t() };
    const long N_pts = (kind == grid_kind::fermionic ? N_w() : N_Omega());
    plan_t tmp(nm, N_pts, B, _eps, math::nufft::NUFFT_FORWARD);
    if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
    else                              tmp.setpts(_x_Omega);
    tmp.forward(C, F_out);
  }

  void run_backward(array_t<2>& F_in, array_t<2>& C_out, grid_kind kind, long B) {
    if (B == _ntrans) {
      if (kind == grid_kind::fermionic) _plan_w->backward(F_in, C_out);
      else                              _plan_Omega->backward(F_in, C_out);
      return;
    }
    const std::array<int64_t,1> nm = { _grid->N_t() };
    const long N_pts = (kind == grid_kind::fermionic ? N_w() : N_Omega());
    plan_t tmp(nm, N_pts, B, _eps, math::nufft::NUFFT_FORWARD);
    if (kind == grid_kind::fermionic) tmp.setpts(_x_w);
    else                              tmp.setpts(_x_Omega);
    tmp.backward(F_in, C_out);
  }

  real_freq_grid_t const* _grid;
  long                    _ntrans;
  double                  _eps;
  rarray_t<1>             _x_w;       // scaled fermionic coords w_j * dt, in MEM
  rarray_t<1>             _x_Omega;   // scaled bosonic coords Omega_l * dt, in MEM
  array_t<1>              _sgn_t;     // i*sgn(t_k) for the Hilbert kernel, in MEM
  std::unique_ptr<plan_t> _plan_w;
  std::unique_ptr<plan_t> _plan_Omega;
};

} // namespace detail

/// Backwards-compat alias: existing host-only call sites can continue to
/// reference `real_axis_conv_t`. Future device callers use
/// `real_axis_conv_mem_t<DEVICE_MEMORY>`.
using real_axis_conv_t = detail::real_axis_conv_base_t<HOST_MEMORY>;

template<MEMORY_SPACE MEM>
using real_axis_conv_mem_t = detail::real_axis_conv_base_t<MEM>;

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_CONV_HPP
