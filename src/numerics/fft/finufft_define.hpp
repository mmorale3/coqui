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

#pragma once
#include <cstdint>
#include <complex>

namespace math::nufft
{

enum NUFFT_BACKEND { NUFFT_BACKEND_UNDEFINED, NUFFT_BACKEND_FINUFFT };

// Sign-convention flag passed to finufft_makeplan.
// +1 : f(k) = sum_j c_j exp(+i k.x_j)   (type-1 forward)
// -1 : f(k) = sum_j c_j exp(-i k.x_j)   (type-1 backward)
enum NUFFT_IFLAG : int { NUFFT_FORWARD = +1, NUFFT_BACKWARD = -1 };

// -------------------------------------------------------------------------
// nuplan_t  —  mirrors fftplan_t from fft_define.hpp.
//
// fwd points to a heap-allocated finufft_plan  (type-1: NU → U).
// inv points to a heap-allocated finufft_plan  (type-2: U  → NU).
//
// Both are void* to avoid exposing finufft internal types in this header,
// exactly as fftplan_t uses void* for fftw_plan*.
//
// single_prec == false : fwd/inv are finufft_plan*  (double precision)
// single_prec == true  : fwd/inv are finufftf_plan* (single precision)
// -------------------------------------------------------------------------
struct nuplan_t
{
  NUFFT_BACKEND bend           = NUFFT_BACKEND_UNDEFINED;
  int           rank           = 0;     ///< spatial dimension (1, 2, or 3)
  int           ntrans         = 0;     ///< number of simultaneous transforms
  int64_t       npts           = 0;     ///< number of nonuniform points M 
  std::array<int64_t,3> nmodes = {0,0,0}; ///< number of modes in uniform grid 
  int           iflag          = +1;    ///< sign convention
  bool          single_prec    = false; ///< true for float plans

  void *fwd = nullptr;  ///< type-1 plan (NU → U)
  void *inv = nullptr;  ///< type-2 plan (U  → NU)
};

} // namespace math::nufft

