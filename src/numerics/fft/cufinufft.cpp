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
 * ==========================================================================
 */

#include <algorithm>
#include "utilities/check.hpp"
#include "numerics/fft/finufft_define.hpp"
#include "numerics/fft/cufinufft.h"

#if defined(COQUI_HAVE_CUFINUFFT)
  #include <cufinufft.h>
  #define CUNUFFT_CHECK(ier, msg) \
    utils::check((ier) == 0, "cufinufft error {}: " msg, (ier))
#endif

// Single source of truth for the "compiled without cuFINUFFT" error path.
// We make it a function so individual call sites stay readable.
namespace {
[[noreturn]] inline void cufinufft_unavailable(const char* fn)
{
  utils::check(false,
               "cufinufft.cpp::{}: device path requires CoQui to be built "
               "with COQUI_HAVE_CUFINUFFT (cuFINUFFT not available in this build).",
               fn);
  // utils::check throws / aborts; this is unreachable but silences -Wreturn.
  std::abort();
}
}

namespace math::nufft::impl::dev
{

#if defined(COQUI_HAVE_CUFINUFFT)

// =========================================================================
// Internal helpers — make a single cufinufft plan and wrap it in nuplan_t.
// type=1: NU→U,  type=2: U→NU.
// The type-2 plan uses -iflag so it is the mathematical adjoint of type-1.
// =========================================================================

static cufinufft_plan* make_plan_d(int type, int rank, const int64_t *nmodes,
                                    int ntrans, double eps, int iflag)
{
  auto *plan = new cufinufft_plan{};
  // cufinufft_makeplan signature mirrors finufft_makeplan; passing
  // nullptr for opts uses cuFINUFFT's defaults.
  int ier = cufinufft_makeplan(type, rank, const_cast<int64_t*>(nmodes),
                                iflag, ntrans, eps, plan, /*opts=*/nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_makeplan (double) failed");
  return plan;
}

static cufinufftf_plan* make_plan_f(int type, int rank, const int64_t *nmodes,
                                     int ntrans, float eps, int iflag)
{
  auto *plan = new cufinufftf_plan{};
  int ier = cufinufftf_makeplan(type, rank, const_cast<int64_t*>(nmodes),
                                 iflag, ntrans, eps, plan, /*opts=*/nullptr);
  CUNUFFT_CHECK(ier, "cufinufftf_makeplan (float) failed");
  return plan;
}

#endif // COQUI_HAVE_CUFINUFFT

// =========================================================================
// Plan creation — double precision
// =========================================================================
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, double eps, int iflag)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  nuplan_t p;
  p.bend        = NUFFT_BACKEND_CUFINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts;
  std::copy_n(nmodes, rank, p.nmodes.begin());
  p.iflag       = iflag;
  p.single_prec = false;
  p.fwd = static_cast<void*>(make_plan_d(1, rank, nmodes, ntrans,  eps,  iflag));
  p.inv = static_cast<void*>(make_plan_d(2, rank, nmodes, ntrans,  eps, -iflag));
  return p;
#else
  (void)rank; (void)nmodes; (void)npts; (void)ntrans; (void)eps; (void)iflag;
  cufinufft_unavailable("create_plan_impl_(double)");
#endif
}

// =========================================================================
// Plan creation — single precision
// =========================================================================
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, float eps, int iflag)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  nuplan_t p;
  p.bend        = NUFFT_BACKEND_CUFINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts;
  std::copy_n(nmodes, rank, p.nmodes.begin());
  p.iflag       = iflag;
  p.single_prec = true;
  p.fwd = static_cast<void*>(make_plan_f(1, rank, nmodes, ntrans,  eps,  iflag));
  p.inv = static_cast<void*>(make_plan_f(2, rank, nmodes, ntrans,  eps, -iflag));
  return p;
#else
  (void)rank; (void)nmodes; (void)npts; (void)ntrans; (void)eps; (void)iflag;
  cufinufft_unavailable("create_plan_impl_(float)");
#endif
}

// =========================================================================
// setpts — double / single
// Both type-1 and type-2 plans share the same nonuniform points.
// =========================================================================
void setpts(nuplan_t &p, double *x, double *y, double *z)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::setpts: incorrect NUFFT backend.");
  utils::check(!p.single_prec,
               "dev::setpts: double pointers passed to single-precision plan.");
  auto *pfwd = static_cast<cufinufft_plan*>(p.fwd);
  auto *pinv = static_cast<cufinufft_plan*>(p.inv);
  utils::check(pfwd && pinv, "dev::setpts: uninitialised cufinufft plan.");
  int ier;
  ier = cufinufft_setpts(*pfwd, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_setpts (double, type-1) failed");
  ier = cufinufft_setpts(*pinv, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_setpts (double, type-2) failed");
#else
  (void)p; (void)x; (void)y; (void)z;
  cufinufft_unavailable("setpts(double)");
#endif
}

void setpts(nuplan_t &p, float *x, float *y, float *z)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::setpts: incorrect NUFFT backend.");
  utils::check(p.single_prec,
               "dev::setpts: float pointers passed to double-precision plan.");
  auto *pfwd = static_cast<cufinufftf_plan*>(p.fwd);
  auto *pinv = static_cast<cufinufftf_plan*>(p.inv);
  utils::check(pfwd && pinv, "dev::setpts: uninitialised cufinufftf plan.");
  int ier;
  ier = cufinufftf_setpts(*pfwd, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_setpts (float, type-1) failed");
  ier = cufinufftf_setpts(*pinv, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_setpts (float, type-2) failed");
#else
  (void)p; (void)x; (void)y; (void)z;
  cufinufft_unavailable("setpts(float)");
#endif
}

// =========================================================================
// fwdnufft — type-1 (NU→U)  /  invnufft — type-2 (U→NU)
// =========================================================================
void fwdnufft(nuplan_t const &p, std::complex<double> *c, std::complex<double> *f)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::fwdnufft: incorrect NUFFT backend.");
  utils::check(!p.single_prec,
               "dev::fwdnufft: double pointers passed to single-precision plan.");
  auto *pfwd = static_cast<cufinufft_plan*>(p.fwd);
  utils::check(pfwd, "dev::fwdnufft: uninitialised cufinufft plan.");
  int ier = cufinufft_execute(*pfwd,
                               reinterpret_cast<cuDoubleComplex*>(c),
                               reinterpret_cast<cuDoubleComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufft_execute type-1 (double) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("fwdnufft(double)");
#endif
}

void invnufft(nuplan_t const &p, std::complex<double> *f, std::complex<double> *c)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::invnufft: incorrect NUFFT backend.");
  auto *pinv = static_cast<cufinufft_plan*>(p.inv);
  utils::check(pinv, "dev::invnufft: uninitialised cufinufft plan.");
  // type-2: cufinufft_execute(plan, c_out, f_in)
  int ier = cufinufft_execute(*pinv,
                               reinterpret_cast<cuDoubleComplex*>(c),
                               reinterpret_cast<cuDoubleComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufft_execute type-2 (double) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("invnufft(double)");
#endif
}

void fwdnufft(nuplan_t const &p, std::complex<float> *c, std::complex<float> *f)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::fwdnufft: incorrect NUFFT backend.");
  utils::check(p.single_prec,
               "dev::fwdnufft: float pointers passed to double-precision plan.");
  auto *pfwd = static_cast<cufinufftf_plan*>(p.fwd);
  utils::check(pfwd, "dev::fwdnufft: uninitialised cufinufftf plan.");
  int ier = cufinufftf_execute(*pfwd,
                                reinterpret_cast<cuFloatComplex*>(c),
                                reinterpret_cast<cuFloatComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufftf_execute type-1 (float) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("fwdnufft(float)");
#endif
}

void invnufft(nuplan_t const &p, std::complex<float> *f, std::complex<float> *c)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::invnufft: incorrect NUFFT backend.");
  utils::check(p.single_prec,
               "dev::invnufft: float pointers passed to double-precision plan.");
  auto *pinv = static_cast<cufinufftf_plan*>(p.inv);
  utils::check(pinv, "dev::invnufft: uninitialised cufinufftf plan.");
  int ier = cufinufftf_execute(*pinv,
                                reinterpret_cast<cuFloatComplex*>(c),
                                reinterpret_cast<cuFloatComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufftf_execute type-2 (float) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("invnufft(float)");
#endif
}

// =========================================================================
// Type-3 (NU → NU) plan creation. Stored in nuplan_t::t3.
// =========================================================================
nuplan_t create_plan_t3_impl_(int rank, int64_t npts_in, int64_t npts_out,
                              int ntrans, double eps, int iflag)
{
  nuplan_t p;
#if defined(COQUI_HAVE_CUFINUFFT)
  p.bend        = NUFFT_BACKEND_CUFINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts_in;
  p.npts_out    = npts_out;
  p.nmodes      = {1, 1, 1};
  p.iflag       = iflag;
  p.single_prec = false;
  auto *plan = new cufinufft_plan{};
  int ier = cufinufft_makeplan(/*type*/ 3, rank, p.nmodes.data(),
                                iflag, ntrans, eps, plan, /*opts*/ nullptr);
  CUNUFFT_CHECK(ier, "cufinufft_makeplan (double, type-3) failed");
  p.t3 = static_cast<void*>(plan);
#else
  (void)rank; (void)npts_in; (void)npts_out;
  (void)ntrans; (void)eps; (void)iflag;
  cufinufft_unavailable("create_plan_t3_impl_(double)");
#endif
  return p;
}

nuplan_t create_plan_t3_impl_(int rank, int64_t npts_in, int64_t npts_out,
                              int ntrans, float eps, int iflag)
{
  nuplan_t p;
#if defined(COQUI_HAVE_CUFINUFFT)
  p.bend        = NUFFT_BACKEND_CUFINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts_in;
  p.npts_out    = npts_out;
  p.nmodes      = {1, 1, 1};
  p.iflag       = iflag;
  p.single_prec = true;
  auto *plan = new cufinufftf_plan{};
  int ier = cufinufftf_makeplan(/*type*/ 3, rank, p.nmodes.data(),
                                 iflag, ntrans, eps, plan, /*opts*/ nullptr);
  CUNUFFT_CHECK(ier, "cufinufftf_makeplan (float, type-3) failed");
  p.t3 = static_cast<void*>(plan);
#else
  (void)rank; (void)npts_in; (void)npts_out;
  (void)ntrans; (void)eps; (void)iflag;
  cufinufft_unavailable("create_plan_t3_impl_(float)");
#endif
  return p;
}

// =========================================================================
// setpts_t3 — bind source + target NU coordinate arrays (device pointers).
// =========================================================================
void setpts_t3(nuplan_t &p,
               double *x, double *y, double *z,
               double *s, double *t, double *u)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::setpts_t3: incorrect NUFFT backend.");
  utils::check(!p.single_prec,
               "dev::setpts_t3: double pointers passed to single-precision plan.");
  auto *pt3 = static_cast<cufinufft_plan*>(p.t3);
  utils::check(pt3, "dev::setpts_t3: uninitialised cufinufft type-3 plan.");
  int ier = cufinufft_setpts(*pt3, p.npts, x, y, z,
                              static_cast<int>(p.npts_out), s, t, u);
  CUNUFFT_CHECK(ier, "cufinufft_setpts (double, type-3) failed");
#else
  (void)p; (void)x; (void)y; (void)z; (void)s; (void)t; (void)u;
  cufinufft_unavailable("setpts_t3(double)");
#endif
}

void setpts_t3(nuplan_t &p,
               float *x, float *y, float *z,
               float *s, float *t, float *u)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::setpts_t3: incorrect NUFFT backend.");
  utils::check(p.single_prec,
               "dev::setpts_t3: float pointers passed to double-precision plan.");
  auto *pt3 = static_cast<cufinufftf_plan*>(p.t3);
  utils::check(pt3, "dev::setpts_t3: uninitialised cufinufftf type-3 plan.");
  int ier = cufinufftf_setpts(*pt3, p.npts, x, y, z,
                               static_cast<int>(p.npts_out), s, t, u);
  CUNUFFT_CHECK(ier, "cufinufftf_setpts (float, type-3) failed");
#else
  (void)p; (void)x; (void)y; (void)z; (void)s; (void)t; (void)u;
  cufinufft_unavailable("setpts_t3(float)");
#endif
}

// =========================================================================
// execnufft_t3 — type-3 execution (NU → NU).
// =========================================================================
void execnufft_t3(nuplan_t const &p, std::complex<double> *c, std::complex<double> *f)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::execnufft_t3: incorrect NUFFT backend.");
  utils::check(!p.single_prec,
               "dev::execnufft_t3: double pointers passed to single-precision plan.");
  auto *pt3 = static_cast<cufinufft_plan*>(p.t3);
  utils::check(pt3, "dev::execnufft_t3: uninitialised cufinufft type-3 plan.");
  int ier = cufinufft_execute(*pt3,
                               reinterpret_cast<cuDoubleComplex*>(c),
                               reinterpret_cast<cuDoubleComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufft_execute type-3 (double) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("execnufft_t3(double)");
#endif
}

void execnufft_t3(nuplan_t const &p, std::complex<float> *c, std::complex<float> *f)
{
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::execnufft_t3: incorrect NUFFT backend.");
  utils::check(p.single_prec,
               "dev::execnufft_t3: float pointers passed to double-precision plan.");
  auto *pt3 = static_cast<cufinufftf_plan*>(p.t3);
  utils::check(pt3, "dev::execnufft_t3: uninitialised cufinufftf type-3 plan.");
  int ier = cufinufftf_execute(*pt3,
                                reinterpret_cast<cuFloatComplex*>(c),
                                reinterpret_cast<cuFloatComplex*>(f));
  CUNUFFT_CHECK(ier, "cufinufftf_execute type-3 (float) failed");
#else
  (void)p; (void)c; (void)f;
  cufinufft_unavailable("execnufft_t3(float)");
#endif
}

// =========================================================================
// destroy_plan — free type-1 / type-2 / type-3 plans (whichever non-null).
// =========================================================================
void destroy_plan(nuplan_t &p)
{
  if (p.fwd == nullptr && p.inv == nullptr && p.t3 == nullptr) return;
#if defined(COQUI_HAVE_CUFINUFFT)
  utils::check(p.bend == NUFFT_BACKEND_CUFINUFFT,
               "dev::destroy_plan: incorrect NUFFT backend.");
  if (!p.single_prec) {
    if (auto *pfwd = static_cast<cufinufft_plan*>(p.fwd)) {
      cufinufft_destroy(*pfwd); delete pfwd; p.fwd = nullptr;
    }
    if (auto *pinv = static_cast<cufinufft_plan*>(p.inv)) {
      cufinufft_destroy(*pinv); delete pinv; p.inv = nullptr;
    }
    if (auto *pt3 = static_cast<cufinufft_plan*>(p.t3)) {
      cufinufft_destroy(*pt3); delete pt3; p.t3 = nullptr;
    }
  } else {
    if (auto *pfwd = static_cast<cufinufftf_plan*>(p.fwd)) {
      cufinufftf_destroy(*pfwd); delete pfwd; p.fwd = nullptr;
    }
    if (auto *pinv = static_cast<cufinufftf_plan*>(p.inv)) {
      cufinufftf_destroy(*pinv); delete pinv; p.inv = nullptr;
    }
    if (auto *pt3 = static_cast<cufinufftf_plan*>(p.t3)) {
      cufinufftf_destroy(*pt3); delete pt3; p.t3 = nullptr;
    }
  }
#else
  utils::check(p.fwd == nullptr && p.inv == nullptr && p.t3 == nullptr,
               "dev::destroy_plan: device plan present but COQUI_HAVE_CUFINUFFT undefined");
#endif
}

} // namespace math::nufft::impl::dev
