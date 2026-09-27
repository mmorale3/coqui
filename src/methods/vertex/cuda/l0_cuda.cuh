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
    long nca = 0;                 // the number of ACTIVE input components packed (P-3a); 1 + 2 np = all
  };

  /** host pointers to C-contiguous data, shapes as in dynbse.hpp */
  struct l0_tables {
    cplx const *Xfam = nullptr;     // (2, np, nk, nc, nc, nR)
    cplx const *Xcst = nullptr;     // (nk, nc, nc, nR)
    long const *act = nullptr;      // (nca) the ACTIVE input components, global c = 0 (constant) | 1 + f np + a (P-3a)
    double *timing = nullptr;       // optional (4): the driver ADDS its wall times -- [0] alloc, [1] H2D, [2] kernel, [3] D2H
    cplx const *gk = nullptr;       // (ng, nk, nc, nc)
    cplx const *gkq = nullptr;      // (ng, nk, nc, nc)
    cplx const *Ghat = nullptr;     // (nk, ng, nc, nc)  sum_{l != j} gkq(l)^T / (epsG_j - epsG_l + inu)
    cplx const *Gtil = nullptr;     // (nk, ng, nc, nc)  sum_{j != l} gk(j)^T / (epsG_j - epsG_l + inu)
    double const *eps = nullptr;    // (np)
    double const *epsG = nullptr;   // (ng)
    long const *gnode = nullptr;    // (ng)
    double const *fhalf = nullptr;  // (np)
    double const *fd1 = nullptr;    // (np)   f'(eps_a)
    double const *fd2 = nullptr;    // (np)   f''(eps_a) -- the nu = 0 kernel's triple poles (unused by the shift kernel)
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

  /**
   * The inu = 0 twin: dynbse::l0_apply_cols on the device (gpu port R3, notes/gpu_port_plan.md section 4c). The two
   * input families are folded into one on the device (V_a = X.fam(0, a) + X.fam(1, a), the host kernel's order), the
   * passes are the shift kernel's (P_j = g_j^T V, Q_j = P_j Ghat_j, B_j = P_j gkq_j^T; P_l = Gtil_l V, R_l = P_l gkq_l^T)
   * with the nu = 0 partial fractions in the scatter (single poles into F1, the confluent U_j^2 U_a into M2 / M3, the
   * constant part into F1c / M2c) and the nu = 0 assembly (fhalf, f', f''/2, Dsq, Dcb). `act` lists the ACTIVE
   * folded components: c = 0 the constant, c = 1 + a the node a (nca <= 1 + np). Reads Ghat / Gtil formed at inu = 0
   * (l == j excluded), eps, epsG, gnode, fhalf, fd1, fd2, Dsq, Dcb; the shift tables, Dqt, inu and tfold are not read.
   * Same contract, timing split and return as l0_apply_shift_cols.
   */
  long l0_apply_cols(l0_dims const &d, l0_tables const &t, cplx *Ffam, cplx *Fsum, double free_bytes);

  // =====================================================================================================================
  // THE DEVICE-RESIDENT UNIT (gpu port plan section 4d, 2026-09-27; user directive: "the gpu execution should only use
  // the host for trivially fast things"). One engine per rank holds, ON THE DEVICE:
  //   run-wide   the frequency basis (KF, KF2 on the tau grid, the refit maps Ut / Vs / Kmat of the active pole fit)
  //   (s, q)     the static rung K_s, the dense dynamic rungs K_d(s_r) (PH-sym representatives) and K_d(0)
  //   (s, q, nu) Cb_k, the LU of M = 1 - Cb K_s (cuSOLVER), the L0 tables (a resident L0 plan)
  //   block      every tf_vector of the Gamma_1 path (X, F, F2, Gamma, y) and the tau buffers of K_d
  // and runs the Gamma_1 block  ls_apply(D, 0) -> K_d -> ls_apply(D, y1)  without host arithmetic; the host uploads D
  // (D x nR) and downloads Gsum0 / Gsum1 (D x nR) and, when the Sigma deposits still run on the host, y1.
  // ALL host-facing arrays keep the host's ROW-MAJOR layouts (D = nk nc^2):
  //   tf fam (2, np, D, nR), tf cst / Fsum / Gsum / D-block (D, nR), K_s / K_d0 (D, D), K_d(s) (ndist, D, D),
  //   Cb_k (nk, nc^2, nc^2), KF / KF2 (nt, np), Ut (n_kept, nt), Vs (np, n_kept), Kmat (nt, np)
  // so a device buffer can be compared bytewise with its host twin. The LU is of the row-major M's memory (= M^T in
  // column-major terms) and every solve uses op T -- the same factorization LAPACK's getrf computes on that buffer.
  // =====================================================================================================================
  struct unit_engine;

  struct ue_config {
    long np = 0, np_fit = 0, nk = 0, nc = 0, ng = 0, nt = 0, nR_max = 0, ndist = 0, n_kept = 0;
  };

  /** nullptr when the device cannot hold the working set (the caller keeps the host path); `why` then says what failed */
  unit_engine *ue_create(ue_config const &c, double free_bytes, char *why, long why_len);
  void ue_destroy(unit_engine *e);
  /** bytes the engine needs for a configuration (the caller's feasibility check) */
  double ue_bytes(ue_config const &c);

  /** run-wide: KF, KF2 (nt, np) real; the active refit: Ut (n_kept, nt), Vs (np, n_kept), Kmat (nt, np) real */
  void ue_set_basis(unit_engine *e, double const *KF, double const *KF2, double const *Ut, double const *Vs, double const *Kmat);
  /** per (s, q): K_s (D, D), K_d(s_r) (ndist, D, D), K_d(0) (D, D); trep (nt) maps a tau node to its representative */
  void ue_set_rung(unit_engine *e, cplx const *Ks, cplx const *Kds, cplx const *Kd0, long const *trep, cplx scale_k);
  /** per (s, q, nu): Cb_k (nk, nc^2, nc^2); the L0 tables (host pointers; Xfam / Xcst / act of t are ignored); returns the
   *  getrf info of M = 1 - Cb K_s (0 = success). tfold and inu come in t; nu0 selects the nu = 0 kernel. */
  int ue_set_unit(unit_engine *e, cplx const *Cbk, l0_tables const &t, bool nu0, double free_bytes);
  /** the Gamma_1 block of width nR: Dblk (D, nR) in; Gsum0, Gsum1 (D, nR) out; y1fam (2, np, D, nR) and y1cst (D, nR)
   *  out when non-null (the host deposits). Returns the worst tau-refit error of K_d. timing (8): [0] L0, [1] T_s,
   *  [2] rung, [3] refit, [4] vector ops, [5] H2D, [6] D2H, [7] the DLR expansion gemm -- all ADDED, seconds. */
  double ue_gamma1(unit_engine *e, long nR, cplx const *Dblk, cplx *Gsum0, cplx *Gsum1, cplx *y1fam, cplx *y1cst, double *timing);

} // namespace methods::solvers::dynbse_cuda

#endif
