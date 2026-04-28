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

#ifndef NUMERICS_FFT_CUFINUFFT_H
#define NUMERICS_FFT_CUFINUFFT_H

#include <cstdint>
#include <complex>
#include <array>
#include <algorithm>
#include "numerics/fft/finufft_define.hpp"

/*
 * Device-side counterpart to finufft.h. Mirrors the host-side
 * `math::nufft::impl::host` namespace one-for-one with the same function
 * signatures, but the pointers passed to setpts / fwdnufft / invnufft are
 * device pointers (cudaMalloc'd / `nda::cuarray` data()). The actual
 * cuFINUFFT calls live in cufinufft.cpp and are gated on the build flag
 * `COQUI_HAVE_CUFINUFFT`. Without that flag the symbols are still
 * declared (so the templated dispatch in finufft_nda.hpp compiles), but
 * any runtime call to them from a device path aborts with a clear
 * "compiled without cuFINUFFT" message.
 *
 * This header intentionally does NOT include <cufinufft.h> upstream; the
 * device-side `nuplan_t::fwd` / `inv` slots stay as `void*` exactly as in
 * the host case, and the upstream cuFINUFFT plan types are an
 * implementation detail of cufinufft.cpp.
 */

namespace math::nufft::impl::dev
{

// -------------------------------------------------------------------------
// Low-level plan creation. Same semantics as the host counterpart in
// finufft.h. The returned `nuplan_t::bend` is `NUFFT_BACKEND_CUFINUFFT`.
// -------------------------------------------------------------------------

nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, double eps, int iflag);
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, float  eps, int iflag);

template<typename EpsType>
nuplan_t create_plan(int Rank, const int64_t *nmodes, int64_t npts,
                     int ntrans, EpsType eps, int iflag)
{
  return create_plan_impl_(Rank, nmodes, npts, ntrans, eps, iflag);
}

// -------------------------------------------------------------------------
// setpts -- coordinate arrays must be DEVICE pointers (e.g. data() of an
// nda::cuarray). Pass nullptr for unused dimensions.
// -------------------------------------------------------------------------

void setpts(nuplan_t &p, double *x, double *y, double *z);
void setpts(nuplan_t &p, float  *x, float  *y, float  *z);

// -------------------------------------------------------------------------
// Execute transforms.
//
// fwdnufft : type-1  NU → U   c[j] (device) → f[k] (device)
// invnufft : type-2  U  → NU  f[k] (device) → c[j] (device)
// -------------------------------------------------------------------------
void fwdnufft(nuplan_t const &p, std::complex<double> *c, std::complex<double> *f);
void invnufft(nuplan_t const &p, std::complex<double> *f, std::complex<double> *c);

void fwdnufft(nuplan_t const &p, std::complex<float>  *c, std::complex<float>  *f);
void invnufft(nuplan_t const &p, std::complex<float>  *f, std::complex<float>  *c);

// -------------------------------------------------------------------------
// Destroy both plans and release cufinufft-internal memory.
// -------------------------------------------------------------------------
void destroy_plan(nuplan_t &p);

} // namespace math::nufft::impl::dev

#endif // NUMERICS_FFT_CUFINUFFT_H
