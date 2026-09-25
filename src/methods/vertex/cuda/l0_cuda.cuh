/*
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

#ifndef COQUI_VERTEX_L0_CUDA_CUH
#define COQUI_VERTEX_L0_CUDA_CUH

/**
 * The device implementation of dynbse::l0_apply_shift_cols (the inu != 0 twisted {U, T} pair-pole
 * application, ~96 % of the Sigma-side dynamic-vertex solve; notes/gpu_port_plan.md section 5).
 *
 * Plain pointers in, plain pointers out: the caller (dynbse.hpp) owns the nda arrays and hands over
 * their C-contiguous data; this header carries no CoQuí array types so that the .cu stays a small,
 * self-contained CUDA translation unit (built only with ENABLE_CUDA, target vertex_cuda).
 *
 * Structure (from bench/l0_miniapp/l0_gpu.cu, measured 0.64 s per application at the Si 4^3 shape on
 * an H100, 1.6x over the first port): per batch of K k-points, pack X into Vt(k; x, c, r, y), form the
 * per-pole nc x nc operands, three cuBLAS batched gemms per pass (batch = ng K), the scatter with the
 * component index on the grid and the ng poles looped inside the block (the a-indexed targets reduced
 * in registers, one atomic each at the end), then the assembly. The scatter and the assembly follow the
 * PRODUCTION mulU / mulT / assemble of dynbse.hpp (confluent a == nj branches, the R1/R3 tables, the
 * small-nu fold), not the miniapp's subset.
 */

#include <complex>

namespace methods::solvers::dynbse_cuda {

  using cplx = std::complex<double>;

  struct l0_dims {
    long np = 0, np_fit = 0, nk = 0, nc = 0, ng = 0, nR = 0;
  };

  /** host pointers to C-contiguous data, shapes as in dynbse.hpp */
  struct l0_tables {
    cplx const *Xfam = nullptr;     // (2, np, nk, nc, nc, nR)
    cplx const *Xcst = nullptr;     // (nk, nc, nc, nR)
    cplx const *gk = nullptr;       // (ng, nk, nc, nc)
    cplx const *gkq = nullptr;      // (ng, nk, nc, nc)
    cplx const *Ghat = nullptr;     // (nk, ng, nc, nc)  sum_{l != j} gkq(l)^T / (epsG_j - epsG_l + inu)
    cplx const *Gtil = nullptr;     // (nk, ng, nc, nc)  sum_{j != l} gk(j)^T / (epsG_j - epsG_l + inu)
    double const *eps = nullptr;    // (np)
    double const *epsG = nullptr;   // (ng)
    long const *gnode = nullptr;    // (ng)
    double const *fhalf = nullptr;  // (np)
    double const *fd1 = nullptr;    // (np)   f'(eps_a)
    cplx const *Dsq = nullptr;      // (np, np)
    cplx const *Dcb = nullptr;      // (np, np)
    cplx const *Dqt = nullptr;      // (np, np)
    cplx const *s1 = nullptr;       // (np)  shift tables
    cplx const *s3 = nullptr;       // (np)
    cplx const *r1u = nullptr;      // (np)
    cplx const *r3u = nullptr;      // (np)
    cplx const *r3t = nullptr;      // (np)
    cplx const *R1U = nullptr;      // (np, np)
    cplx const *R1T = nullptr;      // (np, np)
    cplx const *R3U = nullptr;      // (np, np)
    cplx const *R3T = nullptr;      // (np, np)
    cplx inu = cplx(0.0, 0.0);
    double tfold = 0.0;             // the small-nu fold ratio (<= 0: off)
    bool sum_part1 = true;          // assemble: Fsum receives part 1 (the constant component) too
    bool skip_cst = false;          // the constant component is exactly zero: skip c = 0
  };

  /**
   * F(2, np, nk, nc, nc, nR) and Fsum(nk, nc, nc, nR) are overwritten (zeroed, then filled) on the host.
   * free_bytes: the device memory the k-batch may use (the caller decides how much of the device it
   * gets, e.g. utils::freemem_device_effective()); the batch size K is chosen from it. Returns K.
   * Aborts (APP_ABORT) on a CUDA / cuBLAS error and when a confluent product lands outside the DLR set.
   */
  long l0_apply_shift_cols(l0_dims const &d, l0_tables const &t, cplx *Ffam, cplx *Fsum,
                           double free_bytes);

} // namespace methods::solvers::dynbse_cuda

#endif
