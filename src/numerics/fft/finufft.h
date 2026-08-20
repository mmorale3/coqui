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

#ifndef NUMERICS_FFT_FINUFFT_H
#define NUMERICS_FFT_FINUFFT_H

#include <cstdint>
#include <complex>
#include <array>
#include <algorithm>
#include "numerics/fft/finufft_define.hpp"
#include <finufft.h>

namespace math::nufft::impl::host
{

// -------------------------------------------------------------------------
// Low-level plan creation.
//
// Creates both a type-1 (NU→U) and type-2 (U→NU) finufft plan and stores
// them in nuplan_t::fwd and nuplan_t::inv respectively — mirroring the way
// fftw.cpp stores fftw_plan* fwd / fftw_plan* inv.
//
// Parameters:
//   rank   : spatial dimension (1, 2, or 3)
//   nmodes : array of length rank — number of Fourier modes per dimension
//   npts   : number of nonuniform points M
//   ntrans : number of simultaneous transforms (1 for single)
//   eps    : requested precision
//   iflag  : sign convention for the type-1 transform (+1 or -1);
//            the type-2 plan uses -iflag so that type2 is the adjoint.
// -------------------------------------------------------------------------

nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts, 
                            int ntrans, double eps, int iflag);
nuplan_t create_plan_impl_(int rank, const int64_t *nmodes, int64_t npts,
                            int ntrans, float  eps, int iflag);

// -------------------------------------------------------------------------
// Template helpers — convert long int arrays and forward to the overloads.
// (Same pattern as fftw.h.)
// -------------------------------------------------------------------------

/// Single transform.
template<typename EpsType>
nuplan_t create_plan(int Rank, const int64_t *nmodes, int64_t npts,
                     int ntrans, EpsType eps, int iflag)
{
  return create_plan_impl_(Rank, nmodes, npts, ntrans, eps, iflag);
}

// -------------------------------------------------------------------------
// Type-3 (NU → NU) plan creation.
//
// Creates a single finufft type-3 plan and stores it in nuplan_t::t3.
// Use this when you want to go directly from a nonuniform source grid to
// a nonuniform target grid (no uniform-time intermediary). The fwd/inv
// slots are left null; setpts_t3 / execute_t3 must be used.
//
// Parameters:
//   rank     : spatial dimension (1, 2, or 3)
//   npts_in  : number of source nonuniform points M
//   npts_out : number of target nonuniform points N
//   ntrans   : number of simultaneous transforms
//   eps      : requested precision
//   iflag    : sign convention (+1 or -1) for exp(±i s.x)
// -------------------------------------------------------------------------
nuplan_t create_plan_t3_impl_(int rank, int64_t npts_in, int64_t npts_out,
                              int ntrans, double eps, int iflag);
nuplan_t create_plan_t3_impl_(int rank, int64_t npts_in, int64_t npts_out,
                              int ntrans, float  eps, int iflag);

template<typename EpsType>
nuplan_t create_plan_t3(int Rank, int64_t npts_in, int64_t npts_out,
                        int ntrans, EpsType eps, int iflag)
{
  return create_plan_t3_impl_(Rank, npts_in, npts_out, ntrans, eps, iflag);
}

// -------------------------------------------------------------------------
// Set nonuniform points.  Must be called after create_plan and before
// fwdnufft / invnufft.  The arrays must remain valid until after execute.
// Pass nullptr for unused dimensions.
// -------------------------------------------------------------------------
void setpts(nuplan_t &p, double *x, double *y, double *z);
void setpts(nuplan_t &p, float  *x, float  *y, float  *z);

// -------------------------------------------------------------------------
// setpts_t3 — bind source AND target nonuniform coordinates to a type-3
// plan. Both arrays must remain valid until execute_t3.
//
// In 1D: pass `s` for source coords and `s_out` for target coords (others nullptr).
// In 2D / 3D pass the corresponding y / z arrays.
// -------------------------------------------------------------------------
void setpts_t3(nuplan_t &p,
               double *x, double *y, double *z,
               double *s, double *t, double *u);
void setpts_t3(nuplan_t &p,
               float  *x, float  *y, float  *z,
               float  *s, float  *t, float  *u);

// -------------------------------------------------------------------------
// Execute transforms.
//
// fwdnufft : type-1  NU→U   c[j] → f[k]
// invnufft : type-2  U→NU   f[k] → c[j]
// execnufft_t3 : type-3 NU→NU   c[j] → f[k]
// -------------------------------------------------------------------------
void fwdnufft(nuplan_t const &p, std::complex<double>      *c, std::complex<double>      *f);
void invnufft(nuplan_t const &p, std::complex<double>      *f, std::complex<double>      *c);

void fwdnufft(nuplan_t const &p, std::complex<float>      *c, std::complex<float>      *f);
void invnufft(nuplan_t const &p, std::complex<float>      *f, std::complex<float>      *c);

void execnufft_t3(nuplan_t const &p, std::complex<double> *c, std::complex<double> *f);
void execnufft_t3(nuplan_t const &p, std::complex<float>  *c, std::complex<float>  *f);

// -------------------------------------------------------------------------
// Destroy plans and release finufft-internal memory.
// Handles whichever of fwd/inv/t3 are non-null.
// -------------------------------------------------------------------------
void destroy_plan(nuplan_t &p);

} // namespace math::nufft::impl::host

#endif // NUMERICS_FFT_FINUFFT_H
