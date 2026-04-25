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

#include <finufft.h>
#include <algorithm>
#include "utilities/check.hpp"
#include "numerics/fft/finufft_define.hpp"
#include "numerics/fft/finufft.h"

#define NUFFT_CHECK(ier, msg) \
  utils::check((ier) == 0, "finufft error {}: " msg, (ier))

namespace math::nufft::impl::host
{

// =========================================================================
// Internal helpers
// =========================================================================

// Make a single finufft_plan (double) for the given type.
// type=1: NU→U,  type=2: U→NU.
// The type-2 plan uses -iflag so that type2 is the exact adjoint of type1.
static finufft_plan* make_plan_d(int type, int rank, const int64_t *nmodes,
                                  int ntrans, double eps, int iflag)
{
  auto *plan = new finufft_plan{};
  int ier = finufft_makeplan(type, rank, const_cast<int64_t*>(nmodes),
                              iflag, ntrans, eps, plan, /*opts=*/nullptr);
  NUFFT_CHECK(ier, "finufft_makeplan (double) failed");
  return plan;
}

static finufftf_plan* make_plan_f(int type, int rank, const int64_t *nmodes,
                                   int ntrans, float eps, int iflag)
{
  auto *plan = new finufftf_plan{};
  int ier = finufftf_makeplan(type, rank, const_cast<int64_t*>(nmodes),
                               iflag, ntrans, eps, plan, /*opts=*/nullptr);
  NUFFT_CHECK(ier, "finufft_makeplan (float) failed");
  return plan;
}

// =========================================================================
// Plan creation — double precision
//
// Creates both a type-1 (fwd) and type-2 (inv) plan, stored in nuplan_t
// as void*, mirroring fftw.cpp's fftw_plan* fwd / fftw_plan* inv pattern.
// The type-2 plan uses -iflag so it is the mathematical adjoint of type-1.
// =========================================================================
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts, 
                            int ntrans, double eps, int iflag)
{
  nuplan_t p;
  p.bend        = NUFFT_BACKEND_FINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts;
  std::copy_n(nmodes,rank,p.nmodes.begin());
  p.iflag       = iflag;
  p.single_prec = false;
  p.fwd = static_cast<void*>(make_plan_d(1, rank, nmodes, ntrans,  eps,  iflag));
  p.inv = static_cast<void*>(make_plan_d(2, rank, nmodes, ntrans,  eps, -iflag));
  return p;
}

// =========================================================================
// Plan creation — single precision
// =========================================================================
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, float eps, int iflag)
{
  nuplan_t p;
  p.bend        = NUFFT_BACKEND_FINUFFT;
  p.rank        = rank;
  p.ntrans      = ntrans;
  p.npts        = npts;
  std::copy_n(nmodes,rank,p.nmodes.begin());
  p.iflag       = iflag;
  p.single_prec = true;
  p.fwd = static_cast<void*>(make_plan_f(1, rank, nmodes, ntrans,  eps,  iflag));
  p.inv = static_cast<void*>(make_plan_f(2, rank, nmodes, ntrans,  eps, -iflag));
  return p;
}

// =========================================================================
// setpts — double
// Both type-1 and type-2 plans share the same nonuniform points.
// =========================================================================
void setpts(nuplan_t &p, double *x, double *y, double *z)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "setpts: incorrect NUFFT backend.");
  utils::check(!p.single_prec, "setpts: double pointers passed to single-precision plan.");

  auto *pfwd = static_cast<finufft_plan*>(p.fwd);
  auto *pinv = static_cast<finufft_plan*>(p.inv);
  utils::check(pfwd != nullptr && pinv != nullptr, "setpts: uninitialised finufft plan.");

  int ier;
  ier = finufft_setpts(*pfwd, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  NUFFT_CHECK(ier, "finufft_setpts (double, type-1) failed");
  ier = finufft_setpts(*pinv, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  NUFFT_CHECK(ier, "finufft_setpts (double, type-2) failed");
}

// =========================================================================
// setpts — single
// =========================================================================
void setpts(nuplan_t &p, float *x, float *y, float *z)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "setpts: incorrect NUFFT backend.");
  utils::check(p.single_prec, "setpts: float pointers passed to double-precision plan.");

  auto *pfwd = static_cast<finufftf_plan*>(p.fwd);
  auto *pinv = static_cast<finufftf_plan*>(p.inv);
  utils::check(pfwd != nullptr && pinv != nullptr, "setpts: uninitialised finufftf plan.");

  int ier;
  ier = finufftf_setpts(*pfwd, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  NUFFT_CHECK(ier, "finufft_setpts (float, type-1) failed");
  ier = finufftf_setpts(*pinv, p.npts, x, y, z, 0, nullptr, nullptr, nullptr);
  NUFFT_CHECK(ier, "finufft_setpts (float, type-2) failed");
}

// =========================================================================
// fwdnufft — type-1, double  (NU→U)
// =========================================================================
void fwdnufft(nuplan_t const &p, std::complex<double> *c, std::complex<double> *f)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "fwdnufft: incorrect NUFFT backend.");
  utils::check(!p.single_prec, "fwdnufft: double pointers passed to single-precision plan.");
  auto *pfwd = static_cast<finufft_plan*>(p.fwd);
  utils::check(pfwd != nullptr, "fwdnufft: uninitialised finufft plan.");
  int ier = finufft_execute(*pfwd,
                             reinterpret_cast<std::complex<double>*>(c),
                             reinterpret_cast<std::complex<double>*>(f));
  NUFFT_CHECK(ier, "finufft_execute type-1 (double) failed");
}

// =========================================================================
// invnufft — type-2, double  (U→NU)
// =========================================================================
void invnufft(nuplan_t const &p, std::complex<double> *f, std::complex<double> *c)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "invnufft: incorrect NUFFT backend.");
  auto *pinv = static_cast<finufft_plan*>(p.inv);
  utils::check(pinv != nullptr, "invnufft: uninitialised finufft plan.");
  // type-2 signature: finufft_execute(plan, c_out, f_in)
  // i.e. first array arg is always the NU array (c), second is the U array (f).
  int ier = finufft_execute(*pinv,
                             reinterpret_cast<std::complex<double>*>(c),
                             reinterpret_cast<std::complex<double>*>(f));
  NUFFT_CHECK(ier, "finufft_execute type-2 (double) failed");
}

// =========================================================================
// fwdnufft — type-1, single  (NU→U)
// =========================================================================
void fwdnufft(nuplan_t const &p, std::complex<float> *c, std::complex<float> *f)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "fwdnufft: incorrect NUFFT backend.");
  utils::check(p.single_prec, "fwdnufft: float pointers passed to double-precision plan.");
  auto *pfwd = static_cast<finufftf_plan*>(p.fwd);
  utils::check(pfwd != nullptr, "fwdnufft: uninitialised finufftf plan.");
  int ier = finufftf_execute(*pfwd,
                              reinterpret_cast<std::complex<float>*>(c),
                              reinterpret_cast<std::complex<float>*>(f));
  NUFFT_CHECK(ier, "finufft_execute type-1 (float) failed");
}

// =========================================================================
// invnufft — type-2, single  (U→NU)
// =========================================================================
void invnufft(nuplan_t const &p, std::complex<float> *f, std::complex<float> *c)
{
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "invnufft: incorrect NUFFT backend.");
  utils::check(p.single_prec, "invnufft: float pointers passed to double-precision plan.");
  auto *pinv = static_cast<finufftf_plan*>(p.inv);
  utils::check(pinv != nullptr, "invnufft: uninitialised finufftf plan.");
  int ier = finufftf_execute(*pinv,
                              reinterpret_cast<std::complex<float>*>(c),
                              reinterpret_cast<std::complex<float>*>(f));
  NUFFT_CHECK(ier, "finufft_execute type-2 (float) failed");
}

// =========================================================================
// destroy_plan — free both type-1 and type-2 plans.
// Mirrors fftw.cpp: check backend, cast void*, call destroy, delete.
// =========================================================================
void destroy_plan(nuplan_t &p)
{
  if (p.fwd == nullptr && p.inv == nullptr) return;
  utils::check(p.bend == NUFFT_BACKEND_FINUFFT, "destroy_plan: incorrect NUFFT backend.");

  if (!p.single_prec) {
    if (auto *pfwd = static_cast<finufft_plan*>(p.fwd)) {
      finufft_destroy(*pfwd); delete pfwd; p.fwd = nullptr;
    }
    if (auto *pinv = static_cast<finufft_plan*>(p.inv)) {
      finufft_destroy(*pinv); delete pinv; p.inv = nullptr;
    }
  } else {
    if (auto *pfwd = static_cast<finufftf_plan*>(p.fwd)) {
      finufftf_destroy(*pfwd); delete pfwd; p.fwd = nullptr;
    }
    if (auto *pinv = static_cast<finufftf_plan*>(p.inv)) {
      finufftf_destroy(*pinv); delete pinv; p.inv = nullptr;
    }
  }

}

} // namespace math::nufft::impl::host
