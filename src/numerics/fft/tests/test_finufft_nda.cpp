/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

/*
 * Unit tests for the finufft / nda interface (finufft_nda.hpp).
 *
 * -----------------------------------------------------------------------
 * CRITICAL: finufft array layout vs nda array layout
 * -----------------------------------------------------------------------
 * finufft stores multi-dimensional mode arrays in Fortran (column-major)
 *
 * -----------------------------------------------------------------------
 * Mathematical checks used
 * -----------------------------------------------------------------------
 * (A) Single-mode injection (primary correctness check):
 *     c[j] = exp(-i * iflag * k0 * x[j])  →  F[k0] = M, all others ≈ 0.
 *     Uses the index mapping above.
 *
 * (B) Type-2 from a known single-mode F:
 *     Set the single mode F[k0]=1, all others 0.
 *     type-2 with the type-2 plan (which uses -iflag internally) gives:
 *       c[j] = exp(+i * iflag * k0 * x[j])
 *
 * (C) Uniform-grid round-trip:  type2(type1(c)) = N * c
 *     Holds exactly when M=N and x is the uniform grid on [-π, π).
 *
 * (D) Parseval:  ||F||^2 = N * ||c||^2   (uniform grid, M=N)
 *
 * Tolerance: eps=1e-6 (double) / 1e-4 (float); checks use ~10x that.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "numerics/fft/finufft_nda.hpp"
#include "utilities/test_common.hpp"

#include <cmath>
#include <complex>

namespace nufft_tests
{

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------
template<typename T> struct tol_traits;
template<> struct tol_traits<std::complex<double>>      { static constexpr double val = 1e-5; };
template<> struct tol_traits<std::complex<float>> { static constexpr double val = 1e-3; };
template<typename T> constexpr double tol() { return tol_traits<T>::val; }

template<typename T>
constexpr auto finufft_eps()
{
  if constexpr (std::is_same_v<T, std::complex<float>>) return 1e-4f;
  else return 1e-6;
}

// ---------------------------------------------------------------------------
// Real scalar type matching the complex type T
// ---------------------------------------------------------------------------
template<typename T> struct real_of                   { using type = double; };
template<>           struct real_of<std::complex<float>> { using type = float;  };
template<typename T> using real_of_t = typename real_of<T>::type;

// ---------------------------------------------------------------------------
// Uniform nonuniform points on [-π, π) with spacing 2π/N.
// ---------------------------------------------------------------------------
template<typename RealT>
nda::array<RealT,1> uniform_pts(int N)
{
  nda::array<RealT,1> x(N);
  const RealT h = RealT(2.0 * M_PI) / RealT(N);
  for (int j = 0; j < N; ++j)
    x(j) = RealT(-M_PI) + RealT(j) * h;
  return x;
}

// ---------------------------------------------------------------------------
// Element-wise approximate equality, rank-1
// ---------------------------------------------------------------------------
template<typename T>
bool approx_equal(nda::array<T,1> const &a, nda::array<T,1> const &b, double rtol)
{
  if (a.shape() != b.shape()) return false;
  for (int i = 0; i < a.shape()[0]; ++i)
    if (std::abs(a(i) - b(i)) > rtol * (1.0 + std::abs(b(i)))) return false;
  return true;
}

// L2 norm squared, rank-1
template<typename T>
double norm2(nda::array<T,1> const &v)
{
  double s = 0.0;
  for (int i = 0; i < v.shape()[0]; ++i) s += std::norm(v(i));
  return s;
}

// ===========================================================================
// Test A — 1-D, all interface layers
//
// In 1-D: wave-number k lives at C-order index  k + N/2.
// Valid range for even N: k in [-N/2, N/2 - 1].
// ===========================================================================
template<typename T>
void test_1d()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;

  const int N  = 64;   // mode count; also M (uniform grid)
  const int NT = 3;    // batch size
  const auto eps = finufft_eps<T>();

  // uniform nonuniform points on [-π, π)
  auto x = uniform_pts<RealT>(N);

  // A1 — single-mode injection -------------------------------------------
  // c[j] = exp(-i * k0 * x[j]),  iflag=+1
  // Expected: F[k0 + N/2] = N,  all others ≈ 0.
  {
    const int k0  = 7;
    const int idx = k0 + N/2;   // CMCL: index 0 = mode -N/2

    nda::array<T,1> c(N), F(N);
    for (int j = 0; j < N; ++j)
      c(j) = std::exp(T(0, -RealT(k0) * x(j)));

    auto p = create_plan({N}, N, 1, eps, NUFFT_FORWARD);
    setpts(p, x);
    fwdnufft(p, c, F);
    destroy_plan(p);

    REQUIRE(std::abs(F(idx) - T(RealT(N))) < tol<T>() * double(N));
    for (int k = 0; k < N; ++k)
      if (k != idx)
        REQUIRE(std::abs(F(k)) < tol<T>() * double(N));
  }

  // A2 — type-2 from a known single-mode F --------------------------------
  // F[idx0] = 1, all others 0.
  // type-2 (plan created with iflag=+1 uses -iflag=-1 internally):
  //   c[j] = exp(-i * k0 * x[j])
  {
    const int k0  = -3;
    const int idx = k0 + N/2;

    nda::array<T,1> F(N), c(N);
    F() = T(0);
    F(idx) = T(1);

    auto p = create_plan({N}, N, 1, eps, NUFFT_FORWARD);
    setpts(p, x);
    invnufft(p, F, c);
    destroy_plan(p);

    for (int j = 0; j < N; ++j) {
      // type-2 with plan iflag=+1 (internally -1): exp(-i*k0*x[j])
      T expected = std::exp(T(0, RealT(-k0) * x(j)));
      REQUIRE(std::abs(c(j) - expected) < tol<T>());
    }
  }

  // A3 — uniform-grid round-trip: type2(type1(c)) = N*c -----------------
  {
    nda::array<T,1> c_ref(N), c(N), F(N), c_rt(N);
    utils::fillRandomArray(c_ref);
    c = c_ref;

    auto p = create_plan({N}, N, 1, eps, NUFFT_FORWARD);
    setpts(p, x);
    fwdnufft(p, c, F);
    invnufft(p, F, c_rt);
    destroy_plan(p);

    nda::array<T,1> c_scaled(N);
    for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
    REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
  }

  // A4 — Parseval: ||F||^2 = N * ||c||^2 --------------------------------
  {
    nda::array<T,1> c_ref(N), c(N), F(N);
    utils::fillRandomArray(c_ref);
    c = c_ref;

    auto p = create_plan({N}, N, 1, eps, NUFFT_FORWARD);
    setpts(p, x);
    fwdnufft(p, c, F);
    destroy_plan(p);

    double nc2 = norm2(c_ref);
    double nF2 = norm2(F);
    REQUIRE(std::abs(nF2 - double(N) * nc2) < tol<T>() * double(N) * nc2);
  }

  // A5 — batched plan ----------------------------------------------------
  {
    nda::array<T,1> c_ref(N);
    utils::fillRandomArray(c_ref);
    nda::array<T,2> C(NT,N), F(NT,N), C_rt(NT,N);
    for (int t = 0; t < NT; ++t) C(t, nda::range::all) = c_ref;

    auto p = create_plan({N}, N, NT, eps, NUFFT_FORWARD);
    setpts(p, x);
    fwdnufft(p, C, F);
    invnufft(p, F, C_rt);
    destroy_plan(p);

    nda::array<T,1> c_scaled(N);
    for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
    for (int t = 0; t < NT; ++t)
      REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                           c_scaled, tol<T>()));
  }

  // A6 — plan-less convenience wrappers ----------------------------------
  {
    nda::array<T,1> c_ref(N), c(N), F(N), c_rt(N);
    utils::fillRandomArray(c_ref);
    c = c_ref;
    fwdnufft(c, F, x, eps, NUFFT_FORWARD);
    invnufft(F, c_rt, x, eps, NUFFT_FORWARD);

    nda::array<T,1> c_scaled(N);
    for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
    REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
  }

  // A7 — RAII nufft<false> -----------------------------------------------
  {
    nda::array<T,1> c_ref(N), c(N), F(N), c_rt(N);
    utils::fillRandomArray(c_ref);
    c = c_ref;

    math::nda::nufft nft({N}, N, 1, eps, NUFFT_FORWARD);
    nft.setpts(x);
    nft.forward(c, F);
    nft.backward(F, c_rt);

    nda::array<T,1> c_scaled(N);
    for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
    REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
  }

  // A8 — RAII nufft<true>  (batched) -------------------------------------
  {
    nda::array<T,1> c_ref(N);
    utils::fillRandomArray(c_ref);
    nda::array<T,2> C(NT,N), F(NT,N), C_rt(NT,N);
    for (int t = 0; t < NT; ++t) C(t, nda::range::all) = c_ref;

    math::nda::nufft nft({N}, N, NT, eps, NUFFT_FORWARD);
    nft.setpts(x);
    nft.forward(C, F);
    nft.backward(F, C_rt);

    nda::array<T,1> c_scaled(N);
    for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
    for (int t = 0; t < NT; ++t)
      REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                           c_scaled, tol<T>()));
  }
}

// ===========================================================================
// Test B — 2-D
//
//   wave-number (k1, k2) is at F(k2 + N2/2, k1 + N1/2)
// ===========================================================================
template<typename T>
void test_2d()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;

  const int N1 = 9, N2 = 8;     // mode grid sizes
  const int M  = N1 * N2;       // nonuniform points: full 2-D uniform grid
  const int NT = 2;
  const auto eps = finufft_eps<T>();

  // Build a 2-D uniform grid on [-π,π) x [-π,π) with N1*N2 points.
  nda::array<RealT,1> x(M), y(M);
  {
    const RealT hx = RealT(2.0*M_PI) / RealT(N1);
    const RealT hy = RealT(2.0*M_PI) / RealT(N2);
    int idx = 0;
    for (int j2 = 0; j2 < N2; ++j2)
      for (int j1 = 0; j1 < N1; ++j1, ++idx) {
        x(idx) = RealT(-M_PI) + RealT(j1) * hx;
        y(idx) = RealT(-M_PI) + RealT(j2) * hy;
      }
  }

  // B1 — single-mode injection ------------------------------------------
  // c[j] = exp(-i*(k1*x[j] + k2*y[j])),  iflag=+1
  // Expected: F(k2+N2/2, k1+N1/2) = M,  all others ≈ 0.
  {
    const int k1 = 3, k2 = -2;
    const int r = k1 + N1/2;  
    const int c_ = k2 + N2/2; 
    nda::array<T,1> c(M);
    for (int j = 0; j < M; ++j)
      c(j) = std::exp(T(0, RealT(-k1)*x(j) + RealT(-k2)*y(j)));

    // c-ordering
    {
      nda::array<T,2> F(N1, N2);
      // C-ordered array. Must transpose the definition of the modes.
      auto p = create_plan(std::array<int,2>{N2,N1}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, y, x);
      fwdnufft(p, c, F);
      destroy_plan(p);

      // Each of the M=N1*N2 points contributes 1, so the peak value is M.
      REQUIRE(std::abs(F(r, c_) - T(RealT(M))) < tol<T>() * double(M));
      for (int a = 0; a < N1; ++a)
        for (int b = 0; b < N2; ++b)
          if (a != r || b != c_)
            REQUIRE(std::abs(F(a,b)) < tol<T>() * double(M));
    }

    // f-ordering
    { 
      nda::array<T,2,nda::F_layout> F(N1, N2);
      // C-ordered array. Must transpose the definition of the modes.
      auto p = create_plan(std::array<int,2>{N1,N2}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, x, y);
      fwdnufft(p, c, F);
      destroy_plan(p);
      
      // Each of the M=N1*N2 points contributes 1, so the peak value is M.
      REQUIRE(std::abs(F(r, c_) - T(RealT(M))) < tol<T>() * double(M));
      for (int a = 0; a < N1; ++a)
        for (int b = 0; b < N2; ++b)
          if (a != r || b != c_)
            REQUIRE(std::abs(F(a,b)) < tol<T>() * double(M));
    }

  }

  // B2 — type-2 from a known single-mode F ------------------------------
  // Set F(r, c_) = 1 at (k1,k2) = (2,-1), all others 0.
  // type-2 (internal -iflag): c[j] = exp(-i*(k1*x[j] + k2*y[j]))
  {
    const int k1 = 2, k2 = -1;
    const int r  = k1 + N1/2;
    const int c_ = k2 + N2/2;
    nda::array<T,1> c(M), c_ref(M);
    for (int j = 0; j < M; ++j)
      c_ref(j) = std::exp(T(0, RealT(-k1)*x(j) + RealT(-k2)*y(j)));

    // c-ordering
    {
      nda::array<T,2> F(N1, N2);
      F() = T(0);
      F(r, c_) = T(1);

      auto p = create_plan(std::array<int,2>{N2,N1}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, y, x);
      invnufft(p, F, c);
      destroy_plan(p);

      REQUIRE(approx_equal(c, c_ref, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,2,nda::F_layout> F(N1, N2);
      F() = T(0);
      F(r, c_) = T(1);

      auto p = create_plan(std::array<int,2>{N1,N2}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, x, y);
      invnufft(p, F, c);
      destroy_plan(p);

      REQUIRE(approx_equal(c, c_ref, tol<T>()));
    }

  }

  // B3 — round-trip (uniform grid) --------------------------------------
  {
    nda::array<T,1> c_ref(M), c(M), c_rt(M), c_scaled(M);
    utils::fillRandomArray(c_ref);
    c = c_ref;
    // type2(type1(c)) = M*c for a complete uniform grid (M = N1*N2)
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(M)) * c_ref(j);

    // c-ordering
    { 
      nda::array<T,2> F(N1, N2);
      auto p = create_plan(std::array<int,2>{N2,N1}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, y, x);
      fwdnufft(p, c, F);
      invnufft(p, F, c_rt);
      destroy_plan(p);

      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,2,nda::F_layout> F(N1, N2);
      auto p = create_plan(std::array<int,2>{N1,N2}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, x, y);
      fwdnufft(p, c, F);
      invnufft(p, F, c_rt);
      destroy_plan(p);

      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

  }

  // B4 — batched plan ----------------------------------------------------
  {
    const double scale = double(N1) * double(N2);
    nda::array<T,1> c_ref(M);
    utils::fillRandomArray(c_ref);
    nda::array<T,2> C(NT,M), C_rt(NT,M);
    for (int t = 0; t < NT; ++t) C(t, nda::range::all) = c_ref;
    nda::array<T,1> c_scaled(M);
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(scale)) * c_ref(j);

    // c-ordering
    {
      nda::array<T,3> F(NT,N1,N2);
      auto p = create_plan(std::array<int,2>{N2,N1}, M, NT, eps, NUFFT_FORWARD);
      setpts(p, y, x);
      fwdnufft(p, C, F);
      invnufft(p, F, C_rt);
      destroy_plan(p);

      for (int t = 0; t < NT; ++t)
        REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                             c_scaled, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,3,nda::F_layout> F(N1,N2,NT);
      auto p = create_plan(std::array<int,2>{N1,N2}, M, NT, eps, NUFFT_FORWARD);
      setpts(p, x, y);
      fwdnufft(p, C, F);
      invnufft(p, F, C_rt);
      destroy_plan(p);
      
      for (int t = 0; t < NT; ++t)
        REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                             c_scaled, tol<T>()));
    }

  }

  // B5 — plan-less wrappers ---------------------------------------------
  {
    nda::array<T,1> c_ref(M), c(M), c_rt(M);
    utils::fillRandomArray(c_ref);
    c = c_ref;
    const double scale = double(N1) * double(N2);
    nda::array<T,1> c_scaled(M);
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(scale)) * c_ref(j);

    // c-ordering
    {
      nda::array<T,2> F(N1, N2);
      fwdnufft(c, F, y, x, eps, NUFFT_FORWARD);
      invnufft(F, c_rt, y, x, eps, NUFFT_FORWARD);
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,2,nda::F_layout> F(N1, N2);
      fwdnufft(c, F, x, y, eps, NUFFT_FORWARD);
      invnufft(F, c_rt, x, y, eps, NUFFT_FORWARD);
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

  }
}

// ===========================================================================
// Test C — 3-D
// ===========================================================================
template<typename T>
void test_3d()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;

  const int N1 = 6, N2 = 5, N3 = 4;  // smaller to keep manageable
  const int M  = N1 * N2 * N3;
  const int NT = 2;
  const auto eps = finufft_eps<T>();

  // Build a 3-D uniform grid on [-π,π)^3 with N1*N2*N3 points.
  nda::array<RealT,1> x(M), y(M), z(M);
  {
    const RealT hx = RealT(2.0*M_PI) / RealT(N1);
    const RealT hy = RealT(2.0*M_PI) / RealT(N2);
    const RealT hz = RealT(2.0*M_PI) / RealT(N3);
    int idx = 0;
    for (int j3 = 0; j3 < N3; ++j3)
      for (int j2 = 0; j2 < N2; ++j2)
        for (int j1 = 0; j1 < N1; ++j1, ++idx) {
          x(idx) = RealT(-M_PI) + RealT(j1) * hx;
          y(idx) = RealT(-M_PI) + RealT(j2) * hy;
          z(idx) = RealT(-M_PI) + RealT(j3) * hz;
        }
  }

  // C1 — single-mode injection ------------------------------------------
  {
    const int k1 = -1, k2 = -1, k3 = 1;
    const int i0 = k1 + N1/2;
    const int i1 = k2 + N2/2;
    const int i2 = k3 + N3/2;
    nda::array<T,1> c(M);
    for (int j = 0; j < M; ++j)
      c(j) = std::exp(T(0, RealT(-k1)*x(j) + RealT(-k2)*y(j) + RealT(-k3)*z(j)));

    // c-ordering
    {
      nda::array<T,3> F(N1, N2, N3);
      auto p = create_plan(std::array<int,3>{N3,N2,N1}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, z, y, x);
      fwdnufft(p, c, F);
      destroy_plan(p);

      // Transposed mapping: nda[0]=k3+N3/2, nda[1]=k2+N2/2, nda[2]=k1+N1/2
      REQUIRE(std::abs(F(i0,i1,i2) - T(RealT(M))) < tol<T>() * double(M));
    }

    // f-ordering
    {
      nda::array<T,3,nda::F_layout> F(N1, N2, N3);
      auto p = create_plan(std::array<int,3>{N1,N2,N3}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, x, y, z);
      fwdnufft(p, c, F);
      destroy_plan(p);

      // Transposed mapping: nda[0]=k3+N3/2, nda[1]=k2+N2/2, nda[2]=k1+N1/2
      REQUIRE(std::abs(F(i0,i1,i2) - T(RealT(M))) < tol<T>() * double(M));
    }

  }

  // C2 — round-trip -----------------------------------------------------
  {
    nda::array<T,1> c_ref(M), c(M), c_rt(M);
    utils::fillRandomArray(c_ref);
    nda::array<T,1> c_scaled(M);
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(M)) * c_ref(j);
    c = c_ref;

    // c-ordering
    {
      nda::array<T,3> F(N1, N2, N3);
      auto p = create_plan(std::array<int,3>{N3,N2,N1}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, z, y, x);
      fwdnufft(p, c, F);
      invnufft(p, F, c_rt);
      destroy_plan(p);

      // type2(type1(c)) = M*c for a complete uniform grid (M = N1*N2*N3)
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,3,nda::F_layout> F(N1, N2, N3);
      auto p = create_plan(std::array<int,3>{N1,N2,N3}, M, 1, eps, NUFFT_FORWARD);
      setpts(p, x, y, z);
      fwdnufft(p, c, F);
      invnufft(p, F, c_rt);
      destroy_plan(p);

      // type2(type1(c)) = M*c for a complete uniform grid (M = N1*N2*N3)
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

  }

  // C3 — batched plan ---------------------------------------------------
  {
    nda::array<T,1> c_ref(M);
    utils::fillRandomArray(c_ref);
    nda::array<T,2> C(NT,M), C_rt(NT,M);
    for (int t = 0; t < NT; ++t) C(t, nda::range::all) = c_ref;
    nda::array<T,1> c_scaled(M);
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(M)) * c_ref(j);

    // c-ordering
    {
      nda::array<T,4> F(NT,N1,N2,N3);
      auto p = create_plan(std::array<int,3>{N3,N2,N1}, M, NT, eps, NUFFT_FORWARD);
      setpts(p, z, y, x);
      fwdnufft(p, C, F);
      invnufft(p, F, C_rt);
      destroy_plan(p);

      for (int t = 0; t < NT; ++t)
        REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                             c_scaled, tol<T>()));
    }

    // c-ordering
    {
      nda::array<T,4,nda::F_layout> F(N1,N2,N3,NT);
      auto p = create_plan(std::array<int,3>{N1,N2,N3}, M, NT, eps, NUFFT_FORWARD);
      setpts(p, x, y, z);
      fwdnufft(p, C, F);
      invnufft(p, F, C_rt);
      destroy_plan(p);

      for (int t = 0; t < NT; ++t)
        REQUIRE(approx_equal(nda::array<T,1>(C_rt(t, nda::range::all)),
                             c_scaled, tol<T>()));
    }

  }

  // C4 — plan-less wrappers ---------------------------------------------
  {
    nda::array<T,1> c_ref(M), c(M), c_rt(M);
    utils::fillRandomArray(c_ref);
    c = c_ref;
    nda::array<T,1> c_scaled(M);
    for (int j = 0; j < M; ++j) c_scaled(j) = T(RealT(M)) * c_ref(j);

    // c-ordering
    {
      nda::array<T,3> F(N1, N2, N3);
      fwdnufft(c, F, z, y, x, eps, NUFFT_FORWARD);
      invnufft(F, c_rt, z, y, x, eps, NUFFT_FORWARD);
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

    // f-ordering
    {
      nda::array<T,3,nda::F_layout> F(N1, N2, N3);
      fwdnufft(c, F, x, y, z, eps, NUFFT_FORWARD);
      invnufft(F, c_rt, x, y, z, eps, NUFFT_FORWARD);
      REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));
    }

  }

}

// ===========================================================================
// Test D — sign convention (iflag)
//
// c[j] = exp(+i * k0 * x[j])
//   iflag=+1: F[k0 + N/2] = N  (type-1 sum exp(+i*k0*x)*exp(+i*k*x) peaks at k=-k0... wait)
//
// Careful: type-1 definition with iflag=+1:
//   F[k] = sum_j c[j] exp(+i * k * x[j])
// So for c[j] = exp(+i*k0*x[j]):
//   F[k] = sum_j exp(+i*(k0+k)*x[j])
// This peaks at k = -k0 for a uniform grid (orthogonality).
//
// Wait — that contradicts test A1.  Let's be precise:
//   F[k] = sum_j exp(+i*k0*x[j]) * exp(+i*k*x[j])  [iflag=+1 means exp(+ikx)]
//         = sum_j exp(+i*(k0+k)*x[j])
// For a uniform grid sum_j exp(+i*m*x[j]) = N * delta(m, 0).
// So F[k] = N * delta(k0 + k, 0) = N * delta(k, -k0).
//
// But A1 uses c[j]=exp(+i*k0*x[j]) with NUFFT_FORWARD (+1) and checks F[k0+N/2].
// That would be F[k0] = N*delta(k0,-k0) which is only true for k0=0.
//
// The correct check is: F[-k0 + N/2] = N.
// HOWEVER, finufft's definition of type-1 with iflag=+1 is:
//   f[k] = sum_j c[j] exp(+i * iflag * k * x[j])
// So for c[j]=exp(+i*k0*x[j]), iflag=+1:
//   f[k] = sum_j exp(+i*(k0 + k)*x[j]) = N * delta(k, -k0)
// → F[-k0 + N/2] = N.
//
// But wait, in A1 we set c[j]=exp(+i*k0*x[j]) and expect F[k0+N/2]=N.
// That would require f[k] = sum_j exp(+i*k0*x[j])*exp(-i*k*x[j]) (negative sign).
// That's iflag=-1 definition: f[k]=sum_j c[j]*exp(-i*k*x[j]).
//
// So A1 and A2 needed iflag = NUFFT_BACKWARD (-1) to make c[j]=exp(+i*k0*x[j])
// produce F[k0+N/2]=N, OR we set c[j]=exp(-i*k0*x[j]) with iflag=+1.
//
// We fix all single-mode injection tests to use the correct definition:
//   With NUFFT_FORWARD (iflag=+1):
//     f[k] = sum_j c[j] exp(+i*k*x[j])
//     Set c[j] = exp(-i*k0*x[j])  →  f[k0+N/2] = N
// ===========================================================================
template<typename T>
void test_iflag()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;
  const int N = 32, k0 = 5;
  const auto eps = finufft_eps<T>();

  auto x = uniform_pts<RealT>(N);

  // With NUFFT_FORWARD (+1) and c[j]=exp(-i*k0*x[j]):
  //   f[k] = sum_j exp(-i*k0*x[j]) * exp(+i*k*x[j]) = N*delta(k,k0)
  //   → F[k0 + N/2] = N
  {
    nda::array<T,1> c(N), F(N);
    for (int j = 0; j < N; ++j)
      c(j) = std::exp(T(0, RealT(-k0) * x(j)));

    auto p = create_plan(std::array<int,1>{N}, N, 1, eps, NUFFT_FORWARD);
    setpts(p, x);
    fwdnufft(p, c, F);
    destroy_plan(p);
    REQUIRE(std::abs(F(k0 + N/2) - T(RealT(N))) < tol<T>() * double(N));
  }

  // With NUFFT_BACKWARD (-1) and c[j]=exp(+i*k0*x[j]):
  //   f[k] = sum_j exp(+i*k0*x[j]) * exp(-i*k*x[j]) = N*delta(k,k0)
  //   → F[k0 + N/2] = N  (same result, different sign convention)
  {
    nda::array<T,1> c(N), F(N);
    for (int j = 0; j < N; ++j)
      c(j) = std::exp(T(0, RealT(k0) * x(j)));

    auto p = create_plan(std::array<int,1>{N}, N, 1, eps, NUFFT_BACKWARD);
    setpts(p, x);
    fwdnufft(p, c, F);
    destroy_plan(p);
    REQUIRE(std::abs(F(k0 + N/2) - T(RealT(N))) < tol<T>() * double(N));
  }
}

// ===========================================================================
// Test E — move semantics on math::nda::nufft
// ===========================================================================
template<typename T>
void test_move_semantics()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;
  const int N = 16;
  const auto eps = finufft_eps<T>();

  auto x = uniform_pts<RealT>(N);
  nda::array<T,1> c_ref(N), c(N), F(N), c_rt(N);
  utils::fillRandomArray(c_ref);
  c = c_ref;

  math::nda::nufft nft1(std::array<int,1>{N}, N, 1, eps, NUFFT_FORWARD);
  math::nda::nufft nft2(std::move(nft1));
  nft2.setpts(x);
  nft2.forward(c, F);
  nft2.backward(F, c_rt);

  nda::array<T,1> c_scaled(N);
  for (int j = 0; j < N; ++j) c_scaled(j) = T(RealT(N)) * c_ref(j);
  REQUIRE(approx_equal(c_rt, c_scaled, tol<T>()));

  nda::array<T,1> c2(c_ref), F2(N), c_rt2(N);
  math::nda::nufft nft3(std::array<int,1>{N}, N, 1, eps, NUFFT_FORWARD);
  nft3 = std::move(nft2);
  nft3.setpts(x);
  nft3.forward(c2, F2);
  nft3.backward(F2, c_rt2);
  REQUIRE(approx_equal(c_rt2, c_scaled, tol<T>()));
}

// ===========================================================================
// Test F — setpts reuse
// ===========================================================================
template<typename T>
void test_setpts_reuse()
{
  using namespace math::nufft;
  using RealT = real_of_t<T>;
  const int N = 32, k0 = 4;
  const auto eps = finufft_eps<T>();

  auto x1 = uniform_pts<RealT>(N);
  // x2: denser grid on [-π/2, π/2)
  nda::array<RealT,1> x2(N);
  for (int j = 0; j < N; ++j)
    x2(j) = RealT(-M_PI/2.0) + RealT(j) * RealT(M_PI) / RealT(N);

  // c chosen to excite mode k0 on grid x1 with NUFFT_FORWARD:
  //   c[j] = exp(-i*k0*x[j])  →  F[k0+N/2] = N on x1
  nda::array<T,1> c(N);
  for (int j = 0; j < N; ++j) c(j) = std::exp(T(0, RealT(-k0)*x1(j)));

  nda::array<T,1> F1(N), F2(N), ctmp(c);
  auto p = create_plan(std::array<int,1>{N}, N, 1, eps, NUFFT_FORWARD);

  setpts(p, x1);
  ctmp = c;
  fwdnufft(p, ctmp, F1);
  REQUIRE(std::abs(F1(k0 + N/2) - T(RealT(N))) < tol<T>() * double(N));

  // Re-use plan on x2 — same c, different points → different output
  setpts(p, x2);
  ctmp = c;
  fwdnufft(p, ctmp, F2);
  REQUIRE(!approx_equal(F1, F2, 1e-2));

  destroy_plan(p);
}

// ===========================================================================
// TEST_CASEs
// ===========================================================================

TEST_CASE("finufft_nda_1d_double",    "[nufft]") { test_1d<std::complex<double>>();      }
TEST_CASE("finufft_nda_2d_double",    "[nufft]") { test_2d<std::complex<double>>();      }
TEST_CASE("finufft_nda_3d_double",    "[nufft]") { test_3d<std::complex<double>>();      }
TEST_CASE("finufft_nda_iflag",        "[nufft]") { test_iflag<std::complex<double>>(); }  
TEST_CASE("finufft_nda_move",         "[nufft]") { test_move_semantics<std::complex<double>>(); }
TEST_CASE("finufft_nda_setpts_reuse", "[nufft]") { test_setpts_reuse<std::complex<double>>(); } 

} // namespace nufft_tests
