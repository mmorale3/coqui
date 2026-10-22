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
 * application, the dominant cost of the Sigma-side dynamic-vertex solve).
 *
 * Plain pointers in, plain pointers out: the caller (dynbse.hpp) owns the nda arrays and hands over
 * their C-contiguous data; this header carries no CoQuí array types so that the .cu stays a small,
 * self-contained CUDA translation unit (built only with ENABLE_CUDA, target vertex_cuda).
 *
 * Structure: per batch of K k-points, pack X into Vt(k; x, c, r, y), form the
 * per-pole nc x nc operands, three cuBLAS batched gemms per pass (batch = ng K), the scatter with the
 * component index on the grid and the ng poles looped inside the block (the a-indexed targets reduced
 * in registers, one atomic each at the end), then the assembly. The scatter and the assembly follow
 * mulU / mulT / assemble of dynbse.hpp in full (confluent a == nj branches, the R1/R3 tables, the
 * small-nu fold).
 */

#include <complex>

namespace methods::solvers::dynbse_cuda {

  using cplx = std::complex<double>;

  struct l0_dims {
    long np = 0, np_fit = 0, nk = 0, nc = 0, ng = 0, nR = 0;
    long nca = 0;                 // the number of ACTIVE input components packed; 1 + 2 np = all
  };

  /** host pointers to C-contiguous data, shapes as in dynbse.hpp */
  struct l0_tables {
    cplx const *Xfam = nullptr;     // (2, np, nk, nc, nc, nR)
    cplx const *Xcst = nullptr;     // (nk, nc, nc, nR)
    long const *act = nullptr;      // (nca) the ACTIVE input components, global c = 0 (constant) | 1 + f np + a
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
    // fused: 1 = the FUSED output-stationary passes (the per-pole products formed in registers / shared memory, no
    // materialized Pj / Qj / Bj, no atomics), 0 = the batched-gemm + scatter passes; asm_gemm 1 = the nu = 0 assembly's
    // Dsq / Dcb re-expansions as one gemm per k-batch (0 = atomics)
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
   * The inu = 0 twin: dynbse::l0_apply_cols on the device. The two
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
  // THE DEVICE-RESIDENT UNIT: the device path uses the host only for trivially cheap work. One engine per rank holds,
  // ON THE DEVICE:
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
    // the dense tau rungs K_d(s_r): nres of ndist resident on the device (-1 = the most that fit the budget, 90 % of the free
    // memory minus reserve_bytes = what the later device stages need: rung builds, Sigma deposits, dressed legs). The others
    // reach the device per rung application through two staging buffers, rebuilt from the W tables (needs the device rung
    // builds: partial_ok = 1) or copied from pinned host memory. nonres_src: 0 = both (split by timed cost), 1 = rebuild
    // only, 2 = copy only, 3 = none (every rung must be resident).
    long nres = -1;
    int partial_ok = 0;
    int nonres_src = 0;
    double reserve_bytes = 0.0;
    // 1: two inputs per rung pass (the one-bare-rung and the Gamma_1 input of a block, both frequency families packed side by
    // side: every K_d(s_r) reached once per block); dropped when its buffers would cost residency
    int rung_fuse = 1;
    // the STREAMING THC rung (rung_cuda.cu) instead of the dense tau rungs -- no K_d(s) / K_d0 on the device (no partial
    // residency, no packed pass); the rung and the constant part run through the rung_stream handed over by ue_set_stream.
    // K_s and the LU of M stay dense (T_s).
    int stream = 0;
    // BLOCK-BATCHED rung pass (partial residency only): the RHS blocks of a unit are split in two halves around the rung, and
    // the rung inputs of up to nblk blocks are packed side by side so that every non-resident K_d(s_r) reaches the device once
    // per group of blocks instead of once per block. nblk_want = the blocks of a unit (ceil(nout / nR_max)); batch_nin = the
    // rung inputs per block (2 when a pass may add the one-bare-rung input). The engine picks nblk (<= nblk_want) together with
    // nres; batch = 0 never batches, > 0 forces that many blocks per group (vertex_debug dynbse_dev_batch).
    long nblk_want = 1;
    long batch_nin = 1;
    long batch = -1;
    long nblk = 1;                  // the blocks per group of the configuration being sized (set by ue_create's planner)
    // the pinned host memory per rank (GB) for copies of non-resident rungs when the device rebuild is also available ("auto");
    // a copy-only source ("host", host-built rungs) pins every non-resident rung (vertex_debug dyn_host_pin_gb)
    double host_pin_gb = 8.0;
    // the frequency-factorized rung: 1 = the PROJECTED pass after the first pass of every transfer (the rung output formed in the
    // n_kept directions of the DLR refit only, R n_kept instead of R nt columns per K_r; the refit meter then comes from the first
    // pass); 0 = every pass unprojected (vertex_debug dyn_rung_proj)
    int fact_proj = 1;
    // the frequency-factorized rung APPLIED DENSE (pol_vertex_dyn_rung_apply = "reconstruct"): nfac = R > 0 resident factor slots
    // K_r = K[A_r]; the ndist rung slots are then the per-representative rungs K_d(rep) = sum_r c_r(rep) K_r, applied by the dense
    // packed pass with the mirror-pair gemms (the time functions are mirror-even, rung_basis.hpp). nres of the ndist are formed from
    // the factors (one gemm) at the transfer setup; the others are reconstructed from the factors per rung pass, recon_stage of them
    // per gemm into a staging area of recon_stage slots (the factors are read once per such gemm); ue_create lowers recon_stage,
    // down to 1, when the configured count does not fit.
    // nonres_src 3 = every node rung resident; 0 / 1 = the others reconstructed (2, the pinned-host copy, is not a source here)
    long nfac = 0;
    long recon_stage = 4;
  };
  struct rung_stream;

  struct sd_config;
  /** the device bytes of the later stages (the unit's partition keeps them free; their own checks use them) */
  double ue_kb_bytes(long ns, long nq, long Nm, long nk, long nc, long nrep, long nk_arr = 0, long nsym = 1);
  double ue_sd_bytes(sd_config const &c, long nR_max);
  double ue_dressed_bytes(long ng, long np, long nk, long nc, long nout, long nR_max);
  /** nullptr when the device cannot hold the working set (the caller keeps the host path); `why` then says what failed */
  unit_engine *ue_create(ue_config const &c, double free_bytes, char *why, long why_len);
  void ue_destroy(unit_engine *e);
  /** bytes the engine needs for a configuration (the caller's feasibility check; nres < 0 counts every rung resident) */
  double ue_bytes(ue_config const &c);
  /** the smallest configuration ue_create would try for `c` (the least rung residency, staging and block grouping it allows):
   *  the unit's floor for the device memory plan of a pass */
  double ue_min_bytes(ue_config const &c);
  /** the resident tau rungs of an engine (== ndist: all), and the non-resident rebuilds so far (count, wall seconds) */
  long ue_nres(unit_engine const *e);
  /** the reconstructed rung: the staging slots the engine allocated (ue_create lowers ue_config::recon_stage when that many do
   *  not fit; 0 when every node rung is resident or the rung is not reconstructed) */
  long ue_recon_staging(unit_engine const *e);
  /** whether the engine runs the fused rung pass (ue_config::rung_fuse, kept when it fit the budget) */
  bool ue_rung_fused(unit_engine const *e);
  void ue_rebuild_stats(unit_engine const *e, long *n, double *seconds);
  /** the reconstructed rung (ue_config::nfac > 0): node rungs reconstructed from the factors in the rung passes so far, the
   *  reconstruct gemms, their wall seconds; the resident node rungs formed at the transfer setups (count, wall seconds) */
  void ue_recon_stats(unit_engine const *e, long *nnode, long *ngemm, double *seconds, long *nform, double *t_form);
  /** the reconstructed rung's check (vertex_debug dyn_rung_recon_check = 1): every rung pass also runs the factorized pass on
   *  the same packed input and records max |Y_reconstruct - Y_factorized| and max |Y_factorized| since the last call (reset) */
  void ue_recon_check(unit_engine *e, double *num, double *den);
  /** the non-resident rungs: how many are copied from pinned host memory (the rest rebuilt), copies so far, the planner's
   *  per-rep cost estimates (seconds) */
  void ue_source_stats(unit_engine const *e, long *nhost_reps, long *ncopies, double *t_copy_est, double *t_rb_est);
  /** the planner's other inputs: the timed rung gemm per representative (seconds) and the rung passes of the last
   *  completed transfer */
  void ue_plan_inputs(unit_engine const *e, double *t_gemm_est, long *napp);
  /** the timed copy / rebuild splits (copied count, cost per pass, transfers timed; the one in use marked) as text */
  void ue_plan_report(unit_engine const *e, char *buf, long len);
  /** stream mode: the streaming rung the engine applies (owned by the caller; its (s, q) legs set by the caller, rs_set_sq) */
  void ue_set_stream(unit_engine *e, rung_stream *rs);

  /** run-wide: KF, KF2 (nt, np) real; the basis' DLR refit (imag_axes_ft::dlr_pole_fit at its fixed rank, the np_fit DLR
   *  nodes): Ut (n_kept, nt) = its first n_kept rows, Vs (np_fit, n_kept) = its first n_kept columns (compacted), Kc (nt, np_fit) */
  void ue_set_basis(unit_engine *e, double const *KF, double const *KF2, cplx const *Ut, cplx const *Vs, cplx const *Kc);
  /** per (s, q): K_s (D, D), K_d(s_r) (ndist, D, D), K_d(0) (D, D); trep (nt) maps a tau node to its representative */
  void ue_set_rung(unit_engine *e, cplx const *Ks, cplx const *Kds, cplx const *Kd0, long const *trep, cplx scale_k);
  /** per (s, q, nu): Cb_k (nk, nc^2, nc^2); the L0 tables (host pointers; Xfam / Xcst / act of t are ignored); returns the
   *  getrf info of M = 1 - Cb K_s (0 = success). tfold and inu come in t; nu0 selects the nu = 0 kernel. */
  int ue_set_unit(unit_engine *e, cplx const *Cbk, l0_tables const &t, bool nu0, double free_bytes);
  /** the conditioning meter's operator on the unit's LU (ladder_cond.hpp): Y = B^-H B^-1 X with B = M^T the factored matrix,
   *  i.e. sigma(B) = sigma(1 - Cb K_s); X, Y host column-major (D x n) blocks (two cuSOLVER getrs with n right-hand sides) */
  void ue_cond_apply(unit_engine *e, cplx const *X, cplx *Y, long n);
  /** the same operator on a standalone device LU of a column-major (D x D) host matrix A (the meter's device check) */
  struct dense_lu;
  dense_lu *dlu_create(long D, cplx const *A, int *info);
  void dlu_apply(dense_lu *f, cplx const *X, cplx *Y, long n);
  void dlu_destroy(dense_lu *f);
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
  /** the blocks per rung pass the engine chose (1 = the per-block flow above) */
  long ue_nblk(unit_engine const *e);
  /** THE BLOCK-BATCHED FLOW (ue_nblk > 1): ue_batch_begin with the widths of the group's blocks (nb <= ue_nblk) and the rung
   *  inputs per block (1, or 2 with the dressed path's one-bare-rung input); then ue_gamma1_pre / ue_dressed_pre for every
   *  block in order, ONE ue_batch_rung, and ue_gamma1_post / ue_dressed_post for every block in order. Each post leaves the
   *  engine's per-block state (Dblk, Gs0, Gsum, y1) exactly where ue_gamma1 / ue_gamma1_dressed leave it, so ue_readout and
   *  the device Sigma deposits run after it unchanged. Same arithmetic as the per-block flow; only the rung gemms see wider
   *  column sets (rounding-level differences). */
  void ue_batch_begin(unit_engine *e, long nb, long const *widths, long nin);
  void ue_gamma1_pre(unit_engine *e, cplx const *Dblk, cplx *Gsum0, double *timing);
  void ue_batch_rung(unit_engine *e, double *timing);
  double ue_gamma1_post(unit_engine *e, cplx *Gsum1, cplx *y1fam, cplx *y1cst, double *timing);

  // ---- THE RUNG BUILDS ON THE DEVICE (vertex_dynbse.icc::build_kbig). Per (s, q) every dense rung
  // K[W](k' nc2 + (p1 nc + p3'), k nc2 + (p1' nc + p3)) = [U1^T W(qx(k, k')) U2](p1 nc + p1', p3 nc + p3') with the pair legs
  // U1(P, p1 nc + p1') = X(k', P, p1) conj(X(k, P, p1')), U2(P, p3 nc + p3') = X(k+q, P, p3) conj(X(k'+q, P, p3')) is built
  // straight into the engine: K_s = scale_k K[W0], K_d(r) = K[Wd(rep r)], K_d0 = K[Wd0] -- two batched gemms per W table and
  // (k-chunk) and a scatter kernel; no host rung arrays, no per-transfer upload of the (large) K_d stack.
  /** run-wide: Xb (ns, nk, Nm, nc), qx_of (nk, nk), W0 (nq, Nm, Nm), Wd0 (nq, Nm, Nm), Wds (nrep, nq, Nm, Nm) (the tau
   *  representatives' W in the driver's rep order). false (why) = no room: the host builds and ue_set_rung uploads. */
  /** The layout of the rung builds beyond the plain full mesh (null = the plain layout: nk_arr = nk, one leg set, no
   *  transposed reads). nk_arr: the k extent of the leg table, whose first nk entries are the internal points (the partner
   *  indices of ue_kb_build point anywhere in it). nsym > 1: the legs are (ns, nsym, nk_arr, Nm, nc) -- one leg set per
   *  symmetry class -- and js_of (nk, nk) names the class of every internal pair (k, k'), whose W row is qx_of(k, k') as
   *  before; tr_of (nk, nk) != 0 reads that W row transposed (a time-reversed transfer). */
  struct kb_layout {
    long nk_arr = 0, nsym = 1;
    long const *js_of = nullptr;
    unsigned char const *tr_of = nullptr;
  };
  bool ue_kb_init(unit_engine *e, long ns, long nq, long Nm, cplx const *Xb, long const *qx_of, cplx const *W0, cplx const *Wd0,
                  cplx const *Wds, long nrep, double free_bytes, char *why, long why_len,
                  kb_layout const *lay = nullptr);
  /** per (s, q): K_s, K_d(r), K_d0 built on the device (replaces ue_set_rung); trep (nt) as in ue_set_rung. timing (1) ADDED.
   *  herm (2) out: max |K_s - K_s^dag|, max |K_s| (the Sigma hook's meter). */
  void ue_kb_build(unit_engine *e, long is, long const *kpq_row, long const *trep, cplx scale_k, double *herm, double *timing);
  /** D2H of the resident K_s (D, D) for host consumers (the host Sigma deposits) */
  void ue_get_ks(unit_engine *e, cplx *Ks);

  // ---- THE SIGMA DEPOSITS ON THE DEVICE (vertex_sigma_dyn.icc::sigma_dyn_accumulate: the product
  // route, split accumulators, the IBZ fold, the reference dump, the DW legs). They read the engine's resident output of the last
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
    // the k extent of the ACCUMULATORS (the IBZ externals under the IBZ solve, else the deposit mesh; 0 = nk) and of the
    // per-unit SCRATCH of the IBZ solve (the full mesh the deposits land on before the star fold; 0 = no scratch: the deposits
    // go to the accumulators). The deposit targets (ue_sd_set_sq's row) index the scratch when there is one, else the accumulators.
    long nk_acc = 0, nk_scr = 0;
    // the REFERENCE DUMP of the all-node run (the per-node objects a later nu-sampled run learns its nu-bases from): dump_nw =
    // the number of bosonic nodes (0 = no dump), dump_nsk = the number of random sketch contractions of the T family (0 = none).
    // Device-resident Hc (nw, ns, nt, nk_dep, nc2), HT (nw, ns, np, nk_dep, nc2), HS (nw, ns, np, nk_dep, nsk), the host's layouts.
    long dump_nw = 0, dump_nsk = 0;
    // THE DEVICE FOOTPRINT (0 = everything resident, the layouts above). acc_rows < nk_acc (the IBZ solve only): the device
    // holds a WRITE-BACK cache of that many accumulator k rows over the host accumulators (ue_sd_set_host; pinned staging): the
    // star fold loads a target's host row on a miss and, when the cache is full, writes every cached row back. dump_slots <
    // dump_nw: the reference dump keeps that many bosonic nodes on the device the same way (a unit of another node loads its
    // host block, evicting the slots when all are taken). The device adds every deposit onto the value the resident path would
    // hold: bitwise identical when the host objects start at zero (a loaded checkpoint / a carried hook: rounding class).
    long acc_rows = 0, dump_slots = 0;
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
  /** the IBZ solve: fold the unit scratch of spin `is` onto the accumulators through a star-fold term table (vertex_sym::
   *  fold_star_terms) -- acc(:, k[t]) += (cflag[t] ? conj : id)[N_t^dag X(:, src[t]) N_t] (N_t = I when ident[t]); N (nterm, nc, nc)
   *  row-major -- and zero the scratch */
  void ue_sd_fold(unit_engine *e, long is, long nterm, long const *k, long const *src, unsigned char const *ident,
                  unsigned char const *cflag, cplx const *N);
  /** ADD the device accumulators into the host ones -- S_cst (nt, ns, nk_acc, nc, nc), RT / RU (nt, ns, nk_acc, np, nc, nc) -- and
   *  zero them (the pending unit scratch, if any, must have been folded: ue_sd_fold) */
  void ue_sd_flush(unit_engine *e, cplx *S_cst, cplx *RT, cplx *RU);
  /** the host targets of the partial flushes (sd_config::acc_rows / dump_slots below full residency): the accumulators (the
   *  ue_sd_flush layouts) and the dump (the ue_sd_flush_dump layouts; HS may be nullptr). Required before the first fold /
   *  node change when either cache is active. */
  void ue_sd_set_host(unit_engine *e, cplx *S_cst, cplx *RT, cplx *RU, cplx *Hc, cplx *HT, cplx *HS);
  /** the write-back caches so far: accumulator cache evictions (count, rows written back) and the GB moved (loads + write-backs),
   *  dump slot evictions (count) and the GB moved, their wall seconds */
  void ue_sd_flush_stats(unit_engine const *e, long *n_acc, long *rows_acc, double *gb_acc, long *n_dump, double *gb_dump, double *t);
  /** the reference dump: the sketch contractions Om (nsk, nc2, nc2) row-major (once, after ue_sd_init with dump_nsk > 0) */
  void ue_sd_set_dump(unit_engine *e, cplx const *Om);
  /** the reference dump: the bosonic node m of the following blocks (the dump's first index) */
  void ue_sd_set_node(unit_engine *e, long m);
  /** the reference dump: ADD the device's Hc / HT / HS into the host arrays (the layouts of sd_config) and zero them; HS may be
   *  nullptr when dump_nsk = 0 */
  void ue_sd_flush_dump(unit_engine *e, cplx *Hc, cplx *HT, cplx *HS);

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

  // ---- the DRESSED-LEG Gamma_1 readout on the device (l0_cuda.cu block comment). Per unit, after ue_set_unit and
  // ue_set_legs: H (2 ng, 2 np) the pole-family Grams, Gh / Gt / gk / gkq (ng, nk, nc, nc); ts_zero = the one-bare-rung pass (e~ = D).
  bool ue_dressed_prepare(unit_engine *e, long ng, cplx const *H, cplx const *Gh, cplx const *Gt, cplx const *gk, cplx const *gkq,
                          bool ts_zero, char *why, long why_len);
  /** per block: Pd (nout, nR) = e~^dag (L0 K_d L0 d~)^sum (host pointer, row-major); the static column's Gsum stays on the device
   *  (ue_readout(which = 0)); want_r1: also the one-bare-rung column Pr1 (nout, nR) = D^dag (L0 K_d L0 D)^sum. Returns the refit error. */
  double ue_gamma1_dressed(unit_engine *e, long nR, cplx const *Dblk, cplx inu, bool ts_zero, bool want_r1, cplx *Pd, cplx *Pr1,
                           double *timing, bool want_gsum1 = false);
  /** the block-batched halves of ue_gamma1_dressed (see ue_batch_begin); want_r1 as there (the batch's nin must be 2 when the
   *  one-bare-rung column is computed: want_r1 and not ts_zero and not want_gsum1) */
  void ue_dressed_pre(unit_engine *e, cplx const *Dblk, bool ts_zero, bool want_r1, double *timing, bool want_gsum1 = false);
  double ue_dressed_post(unit_engine *e, cplx inu, bool ts_zero, bool want_r1, cplx *Pd, cplx *Pr1, double *timing,
                         bool want_gsum1 = false);

  /** the rung form of ue_kd: rung_pair = the two PH-mirror nodes of a representative in one gemm (default); rr = R > 0:
   *  the frequency-factorized rung (the engine's rung slots hold K_r = K[A_r], ndist = R), ctr (nt, R) row-major the time functions.
   *  The factorized rung runs through the same packed pass as the per-tau rungs (every K_r on all tau slots of the packed columns:
   *  two-input fusion, both frequency families, the block-batched flow) with the same residency: the non-resident K_r are rebuilt
   *  from A_r on the device or copied from pinned host memory, once per pass. */
  /** recon = true (an engine created with ue_config::nfac = rr): the factorized rung applied dense -- the rung slots are the ndist
   *  per-representative rungs K_d(rep) = sum_r c_r(rep) K_r reconstructed from the resident factors (ue_config::nfac), and ctr is
   *  (ndist, R) by representative slot; check = the factorized pass is run next to every reconstructed pass and compared
   *  (ue_recon_check) */
  void ue_set_rung_mode(unit_engine *e, bool rung_pair, long rr, cplx const *ctr, bool recon = false, bool check = false);

} // namespace methods::solvers::dynbse_cuda

#endif
