/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
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


// The fused Hadamard kernel of GW_line (see gw_line_cuda.cuh). Kept free of fmt-based logging: this TU is compiled by nvcc
// and reports through APP_ABORT with a std::string (the pattern of methods/vertex/cuda/l0_cuda.cu).

#include <algorithm>
#include <string>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "IO/AppAbort.hpp"
#include "methods/GW_line/cuda/gw_line_cuda.cuh"

namespace methods::gw_line::cuda {

namespace {

using cd = cuDoubleComplex;

void cu_check(cudaError_t e, char const *what) {
  if (e == cudaSuccess) return;
  APP_ABORT(std::string(" gw_line_cuda: CUDA error in ") + what + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) +
            ")");
}

struct conv_dev {
  long E;
  int nX, nY, nO, nterm;
  cd const *X;
  long sX;
  cd const *Y;
  long sY;
  cd *O;
  long sO;
  int const *pairs;
  cd alpha;
  int accumulate;
};

__device__ __forceinline__ cd ld(cd const *p) {
  double2 v = __ldg(reinterpret_cast<double2 const *>(p));
  return make_cuDoubleComplex(v.x, v.y);
}
/// s += a b, real and imaginary parts written out
__device__ __forceinline__ void cfma(cd &s, cd a, cd b) {
  s.x = fma(a.x, b.x, fma(-a.y, b.y, s.x));
  s.y = fma(a.x, b.y, fma(a.y, b.x, s.y));
}
__device__ __forceinline__ void store(conv_dev const &d, cd *p, cd s) {
  cd r = make_cuDoubleComplex(d.alpha.x * s.x - d.alpha.y * s.y, d.alpha.x * s.y + d.alpha.y * s.x);
  if (d.accumulate) {
    cd o = *p;
    r.x += o.x;
    r.y += o.y;
  }
  *p = r;
}

// one thread per element (grid-stride), all outputs and terms; operands from global memory
__global__ void conv_direct(conv_dev d) {
  const long stride = long(gridDim.x) * blockDim.x;
  for (long e = long(blockIdx.x) * blockDim.x + threadIdx.x; e < d.E; e += stride) {
    for (int o = 0; o < d.nO; ++o) {
      int const *pr = d.pairs + 2L * o * d.nterm;
      cd s          = make_cuDoubleComplex(0.0, 0.0);
      for (int j = 0; j < d.nterm; ++j) {
        const int ix = __ldg(pr + 2 * j), iy = __ldg(pr + 2 * j + 1);
        cfma(s, ld(d.X + ix * d.sX + e), ld(d.Y + iy * d.sY + e));
      }
      store(d, d.O + o * d.sO + e, s);
    }
  }
}

// tiles of TE consecutive elements: the nX + nY slabs of the tile are staged in shared memory (each read ONCE from global
// memory, coalesced: threadIdx.x runs over the elements), then threadIdx.y spreads the outputs. Shared layout
// [slab][TE]: a warp (32 consecutive x at fixed y) reads 32 consecutive 16-byte words -- conflict free.
template <int TE>
__global__ void conv_staged(conv_dev d) {
  extern __shared__ cd sm[];
  const int tx = threadIdx.x, ty = threadIdx.y, TY = blockDim.y, ns = d.nX + d.nY;
  for (long e0 = long(blockIdx.x) * TE; e0 < d.E; e0 += long(gridDim.x) * TE) {
    const long e  = e0 + tx;
    const bool in = (e < d.E);
    for (int s = ty; s < ns; s += TY) {
      cd v = make_cuDoubleComplex(0.0, 0.0);
      if (in) v = (s < d.nX) ? ld(d.X + s * d.sX + e) : ld(d.Y + (s - d.nX) * d.sY + e);
      sm[s * TE + tx] = v;
    }
    __syncthreads();
    if (in) {
      for (int o = ty; o < d.nO; o += TY) {
        int const *pr = d.pairs + 2L * o * d.nterm;
        cd s          = make_cuDoubleComplex(0.0, 0.0);
        for (int j = 0; j < d.nterm; ++j) {
          const int ix = __ldg(pr + 2 * j), iy = __ldg(pr + 2 * j + 1);
          cfma(s, sm[ix * TE + tx], sm[(d.nX + iy) * TE + tx]);
        }
        store(d, d.O + o * d.sO + e, s);
      }
    }
    __syncthreads();
  }
}

struct dev_props {
  int sms = 0, smem_optin = 0;
};
dev_props const &props() {
  static thread_local int dev = -1;
  static thread_local dev_props p;
  int cur = 0;
  cu_check(cudaGetDevice(&cur), "cudaGetDevice");
  if (cur != dev) {
    cu_check(cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, cur), "attr SM count");
    cu_check(cudaDeviceGetAttribute(&p.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cur), "attr smem optin");
    dev = cur;
  }
  return p;
}

} // namespace

int slab_conv(slab_conv_t const &a, int variant) {
  if (a.E <= 0 or a.nO <= 0) return 0;
  if (a.nterm <= 0) APP_ABORT(std::string(" gw_line_cuda::slab_conv: nterm must be > 0"));
  conv_dev d{a.E,
             a.nX,
             a.nY,
             a.nO,
             a.nterm,
             reinterpret_cast<cd const *>(a.X),
             a.sX,
             reinterpret_cast<cd const *>(a.Y),
             a.sY,
             reinterpret_cast<cd *>(a.O),
             a.sO,
             a.pairs,
             make_cuDoubleComplex(a.alpha.real(), a.alpha.imag()),
             a.accumulate ? 1 : 0};
  auto const &p        = props();
  constexpr int TE     = 32;
  const long smem      = long(a.nX + a.nY) * TE * long(sizeof(cd));
  const bool fits      = smem <= long(p.smem_optin);
  if (variant == 0) variant = (a.nO > 1 and a.nterm > 1 and fits) ? 2 : 1;
  if (variant == 2 and not fits) variant = 1;
  if (variant == 2) {
    const int TY = std::max(1, std::min(8, a.nO));
    if (smem > 48 * 1024)
      cu_check(cudaFuncSetAttribute(conv_staged<TE>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)), "smem attr");
    const long tiles = (a.E + TE - 1) / TE;
    const unsigned nb = unsigned(std::min<long>(tiles, long(p.sms) * 64));
    conv_staged<TE><<<nb, dim3(TE, TY), size_t(smem)>>>(d);
    cu_check(cudaGetLastError(), "conv_staged launch");
  } else {
    constexpr int NT = 256;
    const long blocks = (a.E + NT - 1) / NT;
    const unsigned nb = unsigned(std::min<long>(blocks, long(p.sms) * 32));
    conv_direct<<<nb, NT>>>(d);
    cu_check(cudaGetLastError(), "conv_direct launch");
  }
  return variant;
}

} // namespace methods::gw_line::cuda
