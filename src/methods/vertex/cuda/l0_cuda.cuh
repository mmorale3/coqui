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
    // gpu port 2026-09-27 (the nsys / ncu profile of the device L0): 1 = the FUSED output-stationary passes (the per-pole
    // products formed in registers / shared memory, no materialized Pj / Qj / Bj, no atomics), 0 = the batched-gemm +
    // scatter passes; asm_gemm 1 = the nu = 0 assembly's Dsq / Dcb re-expansions as one gemm per k-batch (0 = atomics)
    int fused = 2;                     // 0 = batched gemms + scatter, 1 = the two fused passes, 2 = one merged pass
    int fz_cfg = 44;                   // the fused kernel's variant: 10 x components per thread + minimum blocks per SM
    int fz_bench = 0;                  // 1: time every variant on the first k-batch per nu class (stderr table)
    int asm_gemm = 1;
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
    long nout = 0;                  // the external-leg / readout dimension (aux N_m, or the Wannier pair count)
    int l0_fused = 2, l0_asm_gemm = 1;   // the resident L0 plan's kernels (l0_tables::fused / asm_gemm)
    int l0_fz_cfg = 44, l0_fz_bench = 0; // l0_tables::fz_cfg / fz_bench
    // larger spaces (2026-09-27): the dense tau rungs K_d(s_r) resident on the device, nres of ndist (-1 = the most that fit
    // the budget); the others are rebuilt from the W tables for every rung application (needs the device rung builds,
    // partial_ok = 1). reserve_bytes = device memory the engine must leave free (the rung-build tables, the Sigma accumulators).
    long nres = -1;
    int partial_ok = 0;
    double reserve_bytes = 0.0;
  };

  /** nullptr when the device cannot hold the working set (the caller keeps the host path); `why` then says what failed */
  unit_engine *ue_create(ue_config const &c, double free_bytes, char *why, long why_len);
  void ue_destroy(unit_engine *e);
  /** bytes the engine needs for a configuration (the caller's feasibility check; nres < 0 counts every rung resident) */
  double ue_bytes(ue_config const &c);
  /** the resident tau rungs of an engine (== ndist: all), and the non-resident rebuilds so far (count, wall seconds) */
  long ue_nres(unit_engine const *e);
  void ue_rebuild_stats(unit_engine const *e, long *n, double *seconds);

  /** run-wide: KF, KF2 (nt, np) real; the basis' DLR refit (imag_axes_ft::dlr_pole_fit at its fixed rank, the np_fit DLR
   *  nodes): Ut (n_kept, nt) = its first n_kept rows, Vs (np_fit, n_kept) = its first n_kept columns (compacted), Kc (nt, np_fit) */
  void ue_set_basis(unit_engine *e, double const *KF, double const *KF2, cplx const *Ut, cplx const *Vs, cplx const *Kc);
  /** per (s, q): K_s (D, D), K_d(s_r) (ndist, D, D), K_d(0) (D, D); trep (nt) maps a tau node to its representative */
  void ue_set_rung(unit_engine *e, cplx const *Ks, cplx const *Kds, cplx const *Kd0, long const *trep, cplx scale_k);
  /** per (s, q, nu): Cb_k (nk, nc^2, nc^2); the L0 tables (host pointers; Xfam / Xcst / act of t are ignored); returns the
   *  getrf info of M = 1 - Cb K_s (0 = success). tfold and inu come in t; nu0 selects the nu = 0 kernel. */
  int ue_set_unit(unit_engine *e, cplx const *Cbk, l0_tables const &t, bool nu0, double free_bytes);
  /** per (s, q): the external legs Dc (nk, nc, nc, nout) -- the readout's left leg and the Sigma deposits' legs */
  void ue_set_legs(unit_engine *e, cplx const *Dc);
  /** a readout of the last ue_gamma1 block: which = 0 Gsum0 | 1 Gsum1 | 2 the one-bare-rung Gsum (when requested there);
   *  P (nout, nR) row-major = D^dag (G - Cb D) (the host's collapse(Dc, G - CbD)); timing (1) ADDED */
  void ue_readout(unit_engine *e, long nR, int which, cplx *P, double *timing);
  /** the Gamma_1 block of width nR: Dblk (D, nR) in; Gsum0, Gsum1 (D, nR) out (each may be null: no D2H); y1fam (2, np, D, nR) and y1cst (D, nR)
   *  out when non-null (the host deposits). Returns the worst tau-refit error of K_d. timing (8): [0] L0, [1] T_s,
   *  [2] rung, [3] refit, [4] vector ops, [5] H2D, [6] D2H, [7] the DLR expansion gemm -- all ADDED, seconds.
   *  want_r1: also the ONE BARE dynamic rung column (the host's one_rung_only pass: T_s = 0, one application) from the same
   *  resident L0 D -- one extra K_d and L0 instead of a second driver pass; kept on the device for ue_readout(which = 2). */
  double ue_gamma1(unit_engine *e, long nR, cplx const *Dblk, cplx *Gsum0, cplx *Gsum1, cplx *y1fam, cplx *y1cst, double *timing,
                   bool want_r1 = false);

  // ---- D-1b: THE RUNG BUILDS ON THE DEVICE (vertex_dynbse.icc::build_kbig, nosym meshes). Per (s, q) every dense rung
  // K[W](k' nc2 + (p1 nc + p3'), k nc2 + (p1' nc + p3)) = [U1^T W(qx(k, k')) U2](p1 nc + p1', p3 nc + p3') with the pair legs
  // U1(P, p1 nc + p1') = X(k', P, p1) conj(X(k, P, p1')), U2(P, p3 nc + p3') = X(k+q, P, p3) conj(X(k'+q, P, p3')) is built
  // straight into the engine: K_s = scale_k K[W0], K_d(r) = K[Wd(rep r)], K_d0 = K[Wd0] -- two batched gemms per W table and
  // (k-chunk) and a scatter kernel; no host rung arrays, no per-transfer upload of the ~10 GB K_d stack.
  /** run-wide: Xb (ns, nk, Nm, nc), qx_of (nk, nk), W0 (nq, Nm, Nm), Wd0 (nq, Nm, Nm), Wds (nrep, nq, Nm, Nm) (the tau
   *  representatives' W in the driver's rep order). false (why) = no room: the host builds and ue_set_rung uploads. */
  bool ue_kb_init(unit_engine *e, long ns, long nq, long Nm, cplx const *Xb, long const *qx_of, cplx const *W0, cplx const *Wd0,
                  cplx const *Wds, long nrep, double free_bytes, char *why, long why_len);
  /** per (s, q): K_s, K_d(r), K_d0 built on the device (replaces ue_set_rung); trep (nt) as in ue_set_rung. timing (1) ADDED.
   *  herm (2) out: max |K_s - K_s^dag|, max |K_s| (the Sigma hook's meter). */
  void ue_kb_build(unit_engine *e, long is, long const *kpq_row, long const *trep, cplx scale_k, double *herm, double *timing);
  /** D2H of the resident K_s (D, D) for host consumers (the host Sigma deposits) */
  void ue_get_ks(unit_engine *e, cplx *Ks);

  // ---- D-3: THE SIGMA DEPOSITS ON THE DEVICE (vertex_sigma_dyn.icc::sigma_dyn_accumulate, the production path: the product
  // route, split accumulators, no IBZ fold, no dump, the DW legs). They read the engine's resident output of the last
  // ue_gamma1 (Gsum0 / Gsum1, y1 = (fam, cst)) and K_s, deposit into device-resident S_cst / RT / RU, and are flushed (ADDED)
  // into the host accumulators before a checkpoint and at the end of the run. Per block (width nR, D = nk nc^2):
  //   A_cst = K_s Gs + y.cst;  DW(k; c i, r) = sum_M conj(Dc(k, c, i, M)) W(r, M);
  //   A_w(n) = A_cst + sum_a U_a(i w_n) fam0_a (+ U_a^2 fam1_a at nu = 0);  F_w(n, k; c y r) = sum_x G(n, k)_{c x} A_w(n, k; x y r);
  //   F_t = [w(tau) Ttw_ff] F_w;  S_cst(tau, s, kpq(k)) += sum_{c r} DW(k; c i, r) F_t(tau, k; c j, r)
  //   T family (nu != 0): Y_a(k; c i, x j) = sum_r DW(k; c i, r) fam1_a(k; x j, r);  M_aj(k; i j) = sum_{c x} g_j(k; c, x) Y_a(c i, x j);
  //   RT_a -= WkT_a M_a,  RU_a += WkU_a M_a,  RU_{gnode(j)} -= sum_a WkU(a, j) M_aj   (batched cuBLAS over (a, k) / (k, j))
  // =====================================================================================================================
  struct sd_config {
    long nt = 0, nw_f = 0, ns = 0, nk = 0, nc = 0, np = 0, np_fit = 0, ng = 0, Nm = 0;
  };
  /** run-wide: Ttw_ff (nt, nw_f) the fermionic w -> tau matrix; Uw (nw_f, np) = 1 / (i w_n - e_a); Gw (nw_f, ns, nk, nc, nc)
   *  G on the fermionic nodes; eps (np), epsG (ng) for the confluence check. Allocates and zeroes the accumulators. false
   *  (with `why`) when the device cannot hold them -- the caller then keeps the host deposits. */
  bool ue_sd_init(unit_engine *e, sd_config const &c, cplx const *Ttw, cplx const *Uw, cplx const *Gw, double const *eps,
                  double const *epsG, double free_bytes, char *why, long why_len);
  /** per (s, q): the row kpq(iq, :) (nk), the G residues gk (ng, nk, nc, nc), gnode (ng); the legs are the engine's (ue_set_legs,
   *  nout == Nm) */
  void ue_sd_set_sq(unit_engine *e, long const *kpq_row, cplx const *gk, long const *gnode);
  /** per unit (node m): the cst + U deposit weights w (nt); the T-family tables WkT (np, ng, nt) (RT_a -= sum_j WkT M_aj) and
   *  WkU (np, ng, nt) (RU_a += sum_j WkU M_aj, RU_{n_j} -= sum_a WkU M_aj); both nullptr = no T family at this unit */
  void ue_sd_set_unit(unit_engine *e, cplx const *w, cplx const *WkT, cplx const *WkU);
  /** per block, right after ue_gamma1 of the same block: col 0 = static_dyn (Gsum0, no y), 1 = dyn1_bare (Gsum0 + y1),
   *  2 = dyn1 (Gsum1 + y1); Wblk (nR, Nm) the block's rows of the outer W; meters (3) max-accumulated: max |A_cst|,
   *  max |y fam|, max |fam1| on an extension node a >= np_fit; timing (1) ADDED. Aborts on a confluent U_j T_a. */
  void ue_sd_block(unit_engine *e, long is, long nR, int col, cplx const *Wblk, bool use_U, bool use_T, double *meters,
                   double *timing);
  /** ADD the device accumulators into the host ones -- S_cst (nt, ns, nk, nc, nc), RT / RU (nt, ns, nk, np, nc, nc) -- and zero them */
  void ue_sd_flush(unit_engine *e, cplx *S_cst, cplx *RT, cplx *RU);

  // ---- small device utilities for the host translation units (all pointers are DEVICE pointers; synchronous) ----------
  /** max_i |x_i| over n elements (0 for n <= 0) */
  double dev_maxabs(cplx const *x, long n);
  /** y[r * ldy + i] += x[r * ldx + i] for r < rows, i < n */
  void dev_add_rows(cplx *y, long ldy, cplx const *x, long ldx, long rows, long n);
  /** a[i * ld + i] += s for i < n */
  void dev_add_diag(cplx *a, long n, long ld, double s);

  /** the Sigma hook's finish (vertex_sigma_dyn.icc::sigma_dyn_finish, product route) on the device; HOST pointers, row-major:
   *  dSig(it; isk, ij) = pref [S_cst + sum_p KF(it, p) RU(it; isk, p, ij) + sum_{a, pp} Tpp(it; a, pp) ec(pp; isk, a, ij)],
   *  ec = Vs . (Ut . E) with E(it; isk, a, ij) = KF(it, a) RT(it; isk, a, ij) (only when anyT; the fit's factors applied in
   *  turn); *fit_err = max|E - Kmap ec| / max|E| (0 without T). KF (nt, np), Tpp (nt, np, npf), Ut (nkept, nt), Vs (npf, nkept),
   *  Kmap (nt, npf); S_cst / dSig (nt, nsk, nc2), RT / RU (nt, nsk, np, nc2). */
  void sd_finish(long nt, long nsk, long np, long npf, long nc2, double const *KF, double const *Tpp, long nkept, cplx const *Ut,
                 cplx const *Vs, double const *Kmap, cplx const *S_cst, cplx const *RT, cplx const *RU, bool anyT, cplx pref,
                 cplx *dSig, double *fit_err);

} // namespace methods::solvers::dynbse_cuda

#endif
