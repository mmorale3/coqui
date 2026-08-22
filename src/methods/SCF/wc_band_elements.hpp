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

#ifndef COQUI_WC_BAND_ELEMENTS_HPP
#define COQUI_WC_BAND_ELEMENTS_HPP

/**
 * Project 2 increment QM3: the state-resolved W^c band elements and the mode-A evaluator
 * context. Sibling of qp_modea.hpp -- read ITS header first: the momentum/spin/prefactor
 * routing, the fit-linearity argument and the trev rule are derived there, and this file is
 * their implementation.
 *
 * The build runs ONCE per outer iteration, inside the only window where W is live
 * (qp_scf_common.cpp: between update_w and the dW reset), and produces
 *
 *     M(a, b, J*npk + p)     per external (s,k) held by this rank,   J = (q', n)
 *
 * the residues of Sigma^c_ab in the pole representation of spec section 1, with the +1/nk
 * prefactor folded in. Two stages:
 *
 *  STAGE 1 (collective, per IBZ q): gather the distributed dW(q, tau) into a NODE-SHARED
 *    buffer, augment the Gamma head if the divergence treatment asks for it, transform to
 *    the bosonic Matsubara mesh, and run the SUPPORT-CONSTRAINED pole fit
 *    (imag_axes_ft::masked_pole_fit -- the promoted QM2-b chain) with the auxiliary-node
 *    columns of |eps_p| < gap_edge removed. Result: node-shared residue slabs
 *    W^(p)_PQ(q), p over the retained nodes. The Np^2 right-hand sides are partitioned over
 *    the node's ranks; the fit is elementwise in that batch axis.
 *
 *  STAGE 1b (collective, per (q,p)): the LOW-RANK FACTORIZATION of those slabs,
 *    W^(p)_PQ = sum_r V(P,r) s_r conj(V(Q,r)) with |s_r| >= wrank * max|s|. See
 *    "WHY STAGE 1b EXISTS" below -- this is what makes the production sizes reachable.
 *
 *  STAGE 1c (collective, per q): the UNION SUBSPACE -- ONE orthonormal basis Q_q for all npk
 *    slab ranges of that q, and the (r_p, R_q) coefficient blocks of every slab in it. This
 *    takes the Np axis out of the p loop of stage 2; see "THE UNION SUBSPACE" below for what
 *    it is worth, which is a MEASURED function of the truncation and not a free win. The
 *    dense slabs (and, here, the per-slab factors) are released as soon as they are dead.
 *
 *  STAGE 2 (distributed): for each external (s,k), loop the qsymms/star structure of thc_gw
 *    verbatim and accumulate the sandwich
 *        B(P,a) = conj(XCe(P,a)) u(P,n),    M^(n,p) += (1/nk) B^T W^(p) conj(B),
 *    densely (wrank <= 0: the reference path), through the stage-1b factors, or through the
 *    stage-1c union basis. The block STORAGE is owned by rank sk % size as before, but the
 *    flops are split over the whole congruence class of that rank on the (isym, q) pair axis
 *    -- with 8 blocks on 32 ranks the single-owner loop left 24 ranks idle.
 *
 * Nothing here re-materializes W at (a,n,b,nu); the working set is one q's residue slab
 * (node-shared) plus this rank's (a,b,J,p) slabs.
 *
 * =====================================================================================
 * THE GATE'S NORMALIZATION -- A DOCUMENTED GATE-SEMANTICS CORRECTION (2026-08-13)
 * [spec-author ruling, recorded; notes/qm3_mode_a_loop_spec.md rev 4. NOT a silent edit:
 *  the numbers every fixture reports for the tau anchor CHANGE with it, and the tables in
 *  this header and in test_qp_map_ab.cpp were re-measured on the same day.]
 * =====================================================================================
 * The tau anchor replaced the i w anchor as THE gate (spec rev 2) but, until this date,
 * inherited its threshold without its NORMALIZATION. The i w anchor divides by the largest
 * |Sigma^GW| of the whole probed set (one number per (s,k) block, qp_scf_common.cpp:270-283);
 * the tau oracle divided each probed element by ITS OWN magnitude. On a symmetry-reduced
 * production mesh those two differ by orders, and the gate fired on the difference:
 *
 *   MEASURED [Si kp444, 13 IBZ k of 64, first symmetry-reduced production run, block (0,0)]:
 *     gap-window diagonals (2,2) (3,3) (4,4) (5,5):  tau rel dev 3.0-3.6e-08
 *     largest gap-window OFF-diagonal (2,5):         tau rel dev 6.6e-05      <- fired
 *     absolute deviation of the whole probed set:    5.5556e-09 a.u. (0.15 ueV), UNIFORM
 *     max|Sigma^GW| of the block:                    1.5648e-01 a.u.
 *     => max|Sigma^GW_(2,5)| = 5.5556e-09 / 6.6e-05 = 8.4e-05 a.u., i.e. the largest
 *        off-diagonal of that window is 1860x BELOW the diagonal (symmetry-suppressed in the
 *        MO basis), and its "relative deviation" measures the smallness of the element, not
 *        an error. The same element read 7.8e-04 in the oracle's per-element i w column while
 *        the BLOCK-normalized i w anchor on the same data read 3.5e-08 -- same numbers, two
 *        denominators, four orders apart. That is the whole effect.
 *   The run's own evidence agreed: per-isym breakdown <= 1.4e-06 for every class, W error
 *   budget 1.586e-07, and 5.5556e-09/1.5648e-01 = 3.55e-08 = exactly the diagonal readings.
 *
 * THE GATE QUANTITY IS THEREFORE, since 2026-08-13,
 *     tau_dev(s,k) = max_{probed elements, tau} |Sigma_B - Sigma^GW|
 *                    / max_{SAME set, tau} |Sigma^GW|,
 * one denominator per block, and the gate is its max over blocks -- reduced together with the
 * (s,k,a,b) where it is attained, because only the ROOT rank's oracle rows reach the log while
 * the gate maxes over every rank's blocks (that is why the kp444 fire was unnameable). The
 * per-element ratios remain as log lines: they are diagnostics, and they are what identified
 * this. The per-isym census keys on the largest ABSOLUTE deviation, which under this
 * normalization IS the element that sets the gate.
 *
 * =====================================================================================
 * THE SYMMETRY PATH -- WHAT THE PRODUCTION DOES TO THE ORBITAL INDICES, AND WHY THIS MAP
 * IS EXACT AGAINST IT FOR ANY D (INCLUDING A NON-SYMMORPHIC GROUP)
 * [verified: re-derived mechanically from thc_gw.icc:287-322 (the isym loop and the D
 *  application), thc_solver_comm.hpp:537-590 (aux_to_primary) and :459-517 / the
 *  primary_to_aux impl (:384-450), mf::MF::symmetry_rotation -> bdft_readonly.hpp:571-585,
 *  and utils::generate_dmatrix, symmetry.hpp:764-1092.
 *  MEASURED: the lih223 symmetry ladder, test_qp_map_ab.cpp "qp_map_modea_sym_ladder",
 *  2026-08-13 -- the table below.]
 * =====================================================================================
 * For an external IBZ point k and symmetry class isym, the production assembles Sigma in the
 * auxiliary basis AT THE ROTATED POINT ks = ks_to_k(isym, k), sums that class's transfers
 * there, and only then rotates the ORBITAL indices back to k:
 *
 *   Sigma^(isym)_ab(ks) = sum_PQ conj(X_ks(P,a)) [ sum_{q in star} -(1/nk) G_PQ(ks-qs)
 *                                                  W_PQ(qs) ] X_ks(Q,b)
 *                                                  [aux_to_primary, kp_map = ks_to_k(isym)]
 *   Sigma_ij(k)        += sum_ab conj(D(a,i)) Sigma^(isym)_ab(ks) D(b,j)   [thc_gw.icc:310-317]
 *
 * with D = MF->symmetry_rotation(isym, k): ROWS are bands at ks, COLUMNS bands at k, and the
 * conjugation flag cjg is false for every k of the IBZ (it is kp_trev(k), bdft_readonly.hpp:
 * 579-584 -- it marks a stored D that itself composes with time reversal, which only happens
 * at a trev image k, never at an IBZ k). The internal leg carries NO D at all: primary_to_aux
 * pairs X(k') with G_ab(kp_to_ibz(k')) directly.
 *
 * NO FRACTIONAL-TRANSLATION PHASE APPEARS ANYWHERE IN THIS CHAIN, symmorphic or not:
 *   * the D matrices are overlaps of the ROTATED IBZ orbital set against the stored one, and
 *     generate_dmatrix builds them with Xft = nullptr (symmetry.hpp:839 and the
 *     transform_k2g call at :1016), i.e. the e^{-i sg Rinv (G+k) T} factor that transform_k2g
 *     CAN return is deliberately not applied;
 *   * the orbitals at an image k-point -- hence the collocation columns X(ks) -- are DEFINED
 *     by the same index rotation with the canonical operation kp_symm(ks). That is why
 *     generate_dmatrix stores the IDENTITY at (kp_symm(ks), k) ("by convention,
 *     d(kp_symm(k), kp_to_ibz(k)) = delta", symmetry.hpp:906-921) and why primary_to_aux may
 *     use the IBZ G matrix at an image k with no rotation at all.
 * Whatever phase convention the stored orbitals carry is therefore COMMON to both sides, and
 * this map never re-derives a rotation: it calls the same MF->symmetry_rotation and reads the
 * same thc.X(is, 0, ks) columns the GW assembly does.
 *
 * Composing with the MO factorizations gives the two collocation matrices of DERIVATION 1
 * (qp_modea.hpp): XCe = X(ks) . D . C(k) and XCi = X(k') . C(kp_to_ibz(k')). Note the ORDER:
 * D multiplies C from the LEFT -- csrmm(1, D, C) -- contracting D's COLUMN index against C's
 * primary index, which is the order both in-tree consumers use
 * (projector_boson_t.cpp:108-121; vertex_sym.hpp:36-42, Xhat = X(krot) . Dc).
 *
 * EXACTNESS, AND WHY D-MATRIX LEAKAGE CANNOT REACH THE ANCHOR. Since
 * conj(XCe(P,n)) = sum_a conj(X_ks(P,a)) conj([D C](a,n)), this map's per-(isym, q) term is
 * the production's per-(isym, q) term with C^dag (.) C applied -- and C^dag (.) C is exactly
 * the MO transform the reference receives before the comparison (qp_scf_common.cpp:1283-1295).
 * The two sides are identical TERM BY TERM for ANY D, unitary or not. The nbnd-truncation
 * leakage of the stored D (row normalization, symmetry.hpp:1067-1092; the "accuracy floor of
 * the symmetry path" of vertex_sym.hpp:52-56) is a property of the shared data, cancels in
 * the difference, and is REPORTED by the symmetry census below, never gated. The only object
 * the two sides do NOT share is the W REPRESENTATION (the support-constrained pole fit and
 * the stage-1b/1c truncations) -- so any tau-anchor deviation is either that or the head.
 *
 * MEASURED [qp_map_modea_sym_ladder, mode_b, one outer iteration; three reductions of the
 * SAME cell and the SAME 2x2x3 mesh, so the W-fit class is common and the deviation is
 * attributable]:
 *
 *     fixture          nk  nk_ibz  nqsym  ntrev     tau dev    W-fit class   ratio
 *     qe_lih223        12    12      1      0     4.1824e-04   3.8581e-03    0.108
 *     qe_lih223_inv    12     8      1      4     4.1824e-04   3.8581e-03    0.108
 *     qe_lih223_sym    12     6      2      4     4.1827e-04   3.8581e-03    0.108
 *
 * [re-measured 2026-08-13 under the block normalization above; the same three rows read
 *  6.3697 / 6.3697 / 6.3703e-04 (ratio 0.165) under the retired per-element one. The fixture
 *  moves by only 1.5x because its largest gap-window off-diagonal is of the same order as the
 *  diagonal -- which is exactly why no fixture ever caught the normalization, and kp444, where
 *  that off-diagonal is 1860x smaller, did.]
 *
 * Turning on the star loop, the trev branches and the D rotation moves the tau anchor by
 * 3e-09 ABSOLUTE on a deviation of 4.2e-04. The sym row is not vacuous: its census reads
 * "D-rotation exercised on 4 of 6 (isym > 0, k) pairs, worst max|D - 1| = 2.0e+00, worst
 * max|D^dag D - 1| = 5.3e-16", i.e. the rotation really is non-trivial there (and exactly
 * unitary on this fixture, whose 16 bands close every multiplet).
 *
 * WHAT THE LADDER DOES NOT COVER, and it is worth knowing before blaming symmetry again:
 * (i) a NON-SYMMORPHIC group -- every symmetry fixture in tests/unit_test_files has ft = 0
 * (checked 2026-08-13: lih222_sym, lih223_sym, svo222_sym, GaAs, all Si; the only
 * non-symmorphic cells in the tree are unreduced meshes, nsym = 1), so a fractional
 * translation is untested LOCALLY -- though by the paragraph above it cannot enter;
 * (ii) a LEAKY D (max|D^dag D - 1| >> 0), which needs a fixture whose band window cuts a
 * degenerate multiplet; (iii) more than two symmetry classes.
 *
 * =====================================================================================
 * WHY STAGE 1b EXISTS -- THE FLOP MODEL OF THE SANDWICH
 * [measured: rusty, Si kp222, nbnd = 60, Np = 2918, nq = 8, ~60 retained poles, 32 ranks
 *  -> ~45 min per outer iteration, i.e. two orders over budget]
 * =====================================================================================
 * The dense sandwich is two gemms per (n,p): (Np,Np)x(Np,nbnd) then (nbnd,Np)x(Np,nbnd),
 * and the (isym, q-in-star) loops cover the FULL q mesh exactly once, so per owned (s,k)
 *
 *     F_dense = nqpts * nbnd * npk * 8 * Np * nbnd * (Np + nbnd)     [real flops]
 *
 * = 1.2e14 for the numbers above -- at ~40 GFLOP/s of complex-gemm throughput on one core
 * that is ~50 min, and only ns*nk_ibz of the ranks carry a block at all (owner = sk % size),
 * so the wall time is ONE block's cost. That is the measurement, explained.
 *
 * With W^(p) = V S V^dag (V: Np x r) the SAME sandwich becomes, for g_r(a) = sum_P B(P,a) V(P,r),
 *
 *     M^(n,p)_ab = sum_r s_r g_r(a) conj(g_r(b)),
 *
 * and g is obtained by r gemms of shape (nbnd,Np)x(Np,nbnd) -- the P axis is contracted ONCE
 * per (q,p,r), never Np times:
 *
 *     F_lowrank = nqpts * npk * r * 8 * Np * nbnd^2   +   nqpts * npk * nbnd * 8 * nbnd^2 * r
 *               = F_dense * (r / Np) * (1 + O(nbnd/Np)).
 *
 * The speedup is r/Np -- exactly the compression ratio of the slab, with NO crossover to
 * worry about below r = Np (the low-rank path is never slower in flops; at r ~ Np it merely
 * stops helping).
 *
 * IS r SMALL? [measured, qe_lih222, test_qp_map_ab "qp_map_modea_np_scan": mean retained rank
 *  over the (q,p) slabs at a FIXED tolerance, with only the THC basis size swept]
 *
 *      Np        96     192     288     384          <- auxiliary basis
 *      1e-4    35.4    36.3    37.5    37.3
 *      1e-6    65.9    72.3    72.7    72.2
 *      1e-8    90.0   105.5   107.0   106.9
 *      1e-10   95.9   135.0   140.0   139.4
 *
 * r SATURATES: past Np ~ 200 the retained rank is flat to a few percent while Np quadruples,
 * i.e. it counts screening modes of the CELL, not basis functions, and r/Np falls as 1/Np.
 * That is the property the whole increment rests on -- the compression is not a fixture
 * artifact, it improves as the basis grows toward the production Np = 2918. (What r is for a
 * given SYSTEM is a different question, and cannot be measured on a fixture; every context
 * build logs the ladder, so the first production run reports it directly.)
 *
 * Setup: one dense heev is 9*Np^3 per (q,p), which at Np = 2918 is 1.1e14 over the whole
 * (q,p) set and would REPLACE the bottleneck it removes, so above detail::wslab_dense_max the
 * factorization uses a randomized Nystrom sketch instead: 3 passes of (Np,Np)x(Np,l) =
 * 24*Np^2*l per (q,p), and the (q,p) axis is partitioned over ALL of the node's ranks (unlike
 * stage 2, which only parallelizes over owned (s,k) blocks). At l = 512 that is 5e13 over the
 * whole set, ~40 s on 32 ranks against the several minutes stage 2 still costs. Subdominant,
 * as required.
 *
 * =====================================================================================
 * THE UNION SUBSPACE (stage 1c) -- AND WHY IT IS A TRUNCATION TRADE, NOT A FREE WIN
 * =====================================================================================
 * Stage 1b contracts the Np axis once per (q,p,r), i.e. sum_p r_p times per (k,q): the npk
 * slabs of one q are npk DIFFERENT bases. Let ONE orthonormal basis Q_q (R_q vectors) span
 * all of them, V^(p) = Q_q A^(p) with A^(p) the (R_q, r_p) coefficient block. Then, with
 *
 *     H(R,a) = sum_P q_R(P) B(P,a)          <- the ONLY contraction of P, once per (k,q,n)
 *     g^(p)(a,r) = sum_R A^(p)(R,r) H(R,a),   M^(n,p)_ab = sum_r s_r g(a,r) conj(g(b,r)),
 *
 *     F_union = 8 * nqpts * nbnd^2 * [ Np * R + npk * r * (R + nbnd) ]
 *     F_1b    = 8 * nqpts * nbnd^2 * [ npk * r * (Np + nbnd) ]
 *
 *     F_union / F_1b  =  R/(npk*r)  +  (R + nbnd)/(Np + nbnd)   ~   R/Np.
 *
 * Np is replaced by R in the dominant term and the one-time projection costs 1/(npk*r) of it.
 * The whole increment is therefore worth exactly what R is -- and R is set by the TRUNCATION,
 * not by the cell:
 *
 * MEASURED [qe_lih222, "qp_map_modea_np_scan"; rank of the npk = 28 slab stack of q = 0, i.e.
 *  of sum_p W^(p) W^(p)dag, at |sigma| >= tol * max|sigma|; per-slab mean r for scale]
 *
 *      Np           96     192     288     384        mean r at Np = 384
 *      1e-4         41      43      49      49              37.3
 *      1e-6         76      86      89      89              72.2
 *      1e-8         95     133     139     143             106.9
 *      1e-10        96     192     288     384             139.4     <- FULL RANK
 *
 * Two things are measured there, and only one of them is good news.
 *
 *  (a) AT A FIXED CUT OF 1e-8 OR LOOSER, R SATURATES IN Np (143 at Np = 384 against 133 at
 *      192) and sits at ~1.2 x max_p r_p: the slabs of one q really do share a subspace, so
 *      R/Np falls as 1/Np and the restructure improves as the basis grows.
 *  (b) AT THE DEFAULT CUT 1e-10 THE SHARING IS GONE: R = Np at every Np. The directions the
 *      slab cut retains down there are mutually orthogonal noise tails, they fill the space,
 *      and the restructure is then break-even (Np*R + npk*r*R against npk*r*Np) at the price
 *      of one extra (nq, Np, R) window and the stage-1c build.
 *
 * DO NOT read the absolute numbers above as production numbers: R >= max_p r_p ALWAYS (the
 * basis must at least span one slab), so the ceiling of the restructure is R/Np >= r/Np, and
 * the production kp222 cell measures r_mean = 520 of Np = 2918 at 1e-10. Even in the best case
 * -- a cut where R falls to ~1.2 max_p r_p -- that is R/Np ~ 0.2, i.e. a 5x stage-2 speedup,
 * not the 20-30x the fixture's r/Np would suggest. lih222 has 16 bands and ~140 screening
 * modes; the production cell has ~4x that, and r is a property of the CELL.
 *
 * So qp_modea_wunion is a knob: a column of V^(p) is dropped from the basis only when
 * |s_r| * ||(1-P) v_r|| < wunion * max_p max|s^(p)|, i.e. only when it moves the RESIDUE SUM
 * of that q by less than wunion of its largest slab (the normalization is discussed at
 * detail::union_build -- it is the absolute one, and it is what the ladder above measures).
 * R, R/Np and the worst projection residual in both norms are logged on every build, the cut
 * is interlocked by the SAME tau anchor that aborts an over-aggressive wrank, and its default
 * is chosen by the scan in test_qp_map_ab.cpp ("qp_map_modea_wunion_scan"). It is never a
 * silent accuracy change.
 *
 * THE DEFAULT IS OFF, and this is why [measured, qe_lih222 / mode_a / qpscf, 20 outer
 * iterations, Np = 192; "R" is what the incremental basis of union_build actually achieves,
 * "gap" is the converged fundamental gap against the per-slab reference]:
 *
 *      wunion       R/Np    gap (eV)     d vs OFF   proj resid   anchor     1c / stage 2 wall
 *      OFF          --      11.856870     --         --          4.26e-3    0.00 s / 1.70 s
 *      1e-10        1.000   11.856870    -1.7e-11    1.8e-12     4.26e-3    0.47 s / 1.41 s
 *      1e-8         0.859   11.856872    +1.7e-06    9.8e-09     4.28e-3    0.36 s / 1.23 s
 *      1e-6         0.818   11.856873    +2.6e-06    6.7e-08     4.28e-3    0.34 s / 1.17 s
 *
 * At the cut the QM3 gates are pinned to, the basis IS the whole space; loosening it by FOUR
 * orders still leaves R = 0.82 Np and has already moved the sixth decimal of the gap. Stage 2
 * does get faster even at R = Np (1.70 -> 1.41 s) but that is BLAS SHAPE, not compression --
 * the union path builds B once per (k,q,n) where the per-slab path rebuilds an (Np,nbnd)
 * Hadamard product once per (k,q,p,r) -- and stage 1c eats it.
 *
 * OPEN, and where to look if this is ever revisited: the ACHIEVED R (0.82 Np at 1e-6) is far
 * above the SVD-optimal stack rank the union probe reports on comparable data (0.45 Np). The
 * probe diagonalizes sum_p W^(p) W^(p)dag, which costs 8*Np^2*r per (q,p) -- more than the
 * stage 2 it would save -- so the incremental basis is what is affordable, and it is the
 * incremental basis that does not pay. Closing that factor of two is the only route left to a
 * compression-based speedup; what scales without it is stage 2's parallelism, below.
 *
 * =====================================================================================
 * WHAT ACTUALLY REACHES kp444: THE PARALLEL CEILING, NOT THE COMPRESSION
 * =====================================================================================
 * Np is set by the CELL (nbnd x the ISDF prefactor) and not by the k mesh, so kp222 -> kp444
 * multiplies the stage-2 work of ONE block by nqpts (8 -> 64) and leaves Np, npk and r where
 * they were. The old loop gave a block to exactly one rank, so it could use ns*nk_ibz ranks
 * (8 at kp222, of 32) and the iteration cost was ONE block's serial time. The pair split
 * raises that ceiling to ns*nk_ibz*nqpts ranks -- 64 at kp222, 512 at kp444 -- and a kp444
 * iteration costs (8 x the kp222 block) / (ranks per block). At the 1e-10 accuracy class,
 * where R = Np and the union restructure is break-even, THAT is the whole increment.
 */

#include <chrono>
#include <format>
#include <random>

#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/qp_modea.hpp"
#include "methods/SCF/wc_spectral.hpp"
#include "methods/mb_state/mb_state.hpp"

#ifdef ENABLE_FINUFFT
// RW-2: the spectral-quadrature W^c path. Only reachable with qp_modea_wfit = "spectral";
// with the flag OFF the knob value is rejected with a clear message at parse time and this
// translation unit is byte-identical in behavior to pre-RW-2 (gate RW-2-b / the RW-1-c
// flag-off inertness class).
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_qp_A.hpp"
#endif

namespace methods {
namespace qp_modea {

  namespace detail {

    /** max|A - A^dag| / max|A| over a (nt, Np, Np) tau slab -- the trev-rule tripwire. */
    inline void herm_probe(nda::MemoryArrayOfRank<3> auto const &W_tPQ,
                           double &num, double &den) {
      auto [nt, NP, NQ] = W_tPQ.shape();
      if (NP != NQ) return;
      for (long t = 0; t < nt; ++t)
        for (long P = 0; P < NP; ++P)
          for (long Q = 0; Q < NP; ++Q) {
            num = std::max(num, std::abs(W_tPQ(t, P, Q) - std::conj(W_tPQ(t, Q, P))));
            den = std::max(den, std::abs(W_tPQ(t, P, Q)));
          }
    }

    // -------------------------------------------------------------------------------------
    //  stage 1b: low-rank factorization of one W^c residue slab
    // -------------------------------------------------------------------------------------

    /** the fixed tolerance ladder reported for every slab -- THE low-rank measurement. */
    inline constexpr std::array<double, 5> wrank_ladder{1e-2, 1e-4, 1e-6, 1e-8, 1e-10};

    /** above this Np the dense LAPACK backend (9*Np^3 per (q,p)) is not affordable. */
    inline constexpr long wslab_dense_max = 600;

    /**
     * One residue slab, factored:  W^(p)_PQ ~= sum_r V(P,r) s_r conj(V(Q,r)),  s real.
     *
     * HERMITICITY. The stored W(tau) is Hermitian in (P,Q) to ~1e-9 relative (measured every
     * build, logged as w_herm_rel), and the tau -> nu -> fit chain is a REAL linear map in the
     * frequency index once the +nu/-nu mirror pairs are summed (Ttw_bb entries are complex but
     * T(t,-nu) = conj(T(t,+nu)), and the mirrored mesh always contains both), so Hermiticity
     * transfers to the residues. `anti` measures it per slab anyway; nothing branches on it.
     */
    struct wslab_factor {
      nda::array<ComplexType, 2> V;   // (Np, r), orthonormal columns
      nda::array<double, 1> s;        // (r) signed eigenvalues, |s| descending
      long r = 0;                     // retained rank
      long nprobe = 0;                // eigenvalues actually computed (Np, or the sketch size)
      double amax = 0.0;              // max|eigenvalue| of the slab
      double tail = 0.0;              // max|discarded lambda| / amax  == ||dW||_2 / ||W||_2
      double frob = 0.0;              // ||discarded||_F / ||all||_F
      double anti = 0.0;              // max|W - W^dag| / max|W| (projected, in the sketch path)
      bool exact = false;             // the LAPACK heev backend was used
      std::array<long, 5> ladder{};   // rank at each wrank_ladder tolerance
    };

    /** keep the eigenpairs with |lambda| >= tol * max|lambda|, |lambda| descending. */
    inline void wslab_truncate(nda::array<double, 1> const &ev,
                               nda::MemoryArrayOfRank<2> auto const &U,
                               double tol, wslab_factor &f) {
      const long m = ev.size(), NP = U.shape(0);
      std::vector<long> idx(m);
      for (long i = 0; i < m; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(),
                [&](long a, long b) { return std::abs(ev(a)) > std::abs(ev(b)); });
      f.nprobe = m;
      f.amax = (m > 0) ? std::abs(ev(idx[0])) : 0.0;
      auto rank_at = [&](double t) {
        long rr = 0;
        if (f.amax > 0.0)
          while (rr < m and std::abs(ev(idx[rr])) >= t * f.amax) ++rr;
        return rr;
      };
      const long r = rank_at(tol);
      f.r = r;
      f.tail = (r < m and f.amax > 0.0) ? std::abs(ev(idx[r])) / f.amax : 0.0;
      double f2all = 0.0, f2d = 0.0;
      for (long i = 0; i < m; ++i) f2all += ev(i) * ev(i);
      for (long i = r; i < m; ++i) f2d += ev(idx[i]) * ev(idx[i]);
      f.frob = (f2all > 0.0) ? std::sqrt(f2d / f2all) : 0.0;
      for (size_t t = 0; t < wrank_ladder.size(); ++t) f.ladder[t] = rank_at(wrank_ladder[t]);
      f.V = nda::array<ComplexType, 2>(NP, r);
      f.s = nda::array<double, 1>(r);
      for (long j = 0; j < r; ++j) {
        f.s(j) = ev(idx[j]);
        for (long P = 0; P < NP; ++P) f.V(P, j) = U(P, idx[j]);
      }
    }

    /** exact backend: LAPACK heev on (W + W^dag)/2. Cost 9*Np^3 -- small Np only. */
    inline void wslab_factor_exact(nda::MemoryArrayOfRank<2> auto const &W, double tol,
                                   wslab_factor &f) {
      const long NP = W.shape(0);
      nda::matrix<ComplexType> Wh(NP, NP);
      double num = 0.0, den = 0.0;
      for (long P = 0; P < NP; ++P)
        for (long Q = 0; Q < NP; ++Q) {
          const ComplexType d = W(P, Q) - std::conj(W(Q, P));
          num = std::max(num, std::abs(d));
          den = std::max(den, std::abs(W(P, Q)));
          Wh(P, Q) = W(P, Q) - 0.5 * d;
        }
      f.anti = (den > 0.0) ? num / den : 0.0;
      auto [ev, U] = nda::linalg::eigenelements(Wh);
      wslab_truncate(ev, U, tol, f);
      f.exact = true;
    }

    /**
     * randomized backend: Nystrom range finding with ONE power iteration (the sketch is
     * W^2 Omega, so the retained space is the |lambda|-dominant one for an INDEFINITE
     * Hermitian slab), then an exact eigendecomposition of the l x l projection.
     *
     * Returns TRUE only if the sketch resolved the tail, i.e. if at least one computed mode
     * fell BELOW tol * max|lambda| -- the standard indicator that the sketch space already
     * contains everything above the threshold. On false the caller must grow l (and ends at
     * the exact backend), so a slab that is NOT low-rank costs speed, never accuracy.
     */
    inline bool wslab_factor_rand(nda::MemoryArrayOfRank<2> auto const &W, double tol, long l,
                                  unsigned long long seed, wslab_factor &f) {
      const long NP = W.shape(0);
      if (l >= NP) return false;
      nda::array<ComplexType, 2> Om(NP, l), Y(NP, l), Q(NP, l), WQ(NP, l), Qc(NP, l), C(l, l);
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double> ud(-1.0, 1.0);
      for (long P = 0; P < NP; ++P)
        for (long j = 0; j < l; ++j) Om(P, j) = ComplexType(ud(rng), ud(rng));
      nda::blas::gemm(W, Om, Y);                     // Y = W Omega
      nda::blas::gemm(W, Y, Q);                      // Q = W^2 Omega
      {   // orthonormalize the sketch -- LAPACK QR wants Fortran order
        nda::matrix<ComplexType, nda::F_layout> Yf(NP, l);
        Yf = Q;
        nda::array<ComplexType, 1> tau(l);
        int info = nda::lapack::geqrf(Yf, tau);
        utils::check(info == 0, "qp_modea: geqrf failed on the W^c sketch (info = {}).", info);
        info = nda::lapack::ungqr(Yf, tau);
        utils::check(info == 0, "qp_modea: ungqr failed on the W^c sketch (info = {}).", info);
        Q = Yf;
      }
      nda::blas::gemm(W, Q, WQ);
      Qc = nda::conj(Q);
      nda::blas::gemm(nda::transpose(Qc), WQ, C);    // C = Q^dag W Q
      double num = 0.0, den = 0.0;
      for (long i = 0; i < l; ++i)
        for (long j = 0; j < l; ++j) {
          num = std::max(num, std::abs(C(i, j) - std::conj(C(j, i))));
          den = std::max(den, std::abs(C(i, j)));
        }
      f.anti = (den > 0.0) ? num / den : 0.0;
      for (long i = 0; i < l; ++i) {
        for (long j = 0; j < i; ++j) {
          const ComplexType h = 0.5 * (C(i, j) + std::conj(C(j, i)));
          C(i, j) = h;
          C(j, i) = std::conj(h);
        }
        C(i, i) = ComplexType(C(i, i).real(), 0.0);
      }
      auto [ev, Z] = nda::linalg::eigenelements(nda::matrix<ComplexType>(C));
      nda::array<ComplexType, 2> V(NP, l);
      nda::blas::gemm(Q, Z, V);                      // V = Q Z
      wslab_truncate(ev, V, tol, f);
      f.exact = false;
      return f.r < l;
    }

    /**
     * Factor one slab. `wsketch`: 0 = automatic (exact below wslab_dense_max, otherwise a
     * sketch of 64 doubling as needed), > 0 = force the sketch with that initial size,
     * < 0 = force the exact backend. The seed depends ONLY on (q,p), so the factors -- and
     * therefore the residues -- are independent of the rank/node layout.
     */
    inline void wslab_factorize(nda::MemoryArrayOfRank<2> auto const &W, double tol,
                                long wsketch, unsigned long long seed, wslab_factor &f) {
      const long NP = W.shape(0);
      long l0 = 0;
      if (wsketch > 0) l0 = wsketch;
      else if (wsketch == 0 and NP > wslab_dense_max) l0 = 64;
      for (long l = l0; l > 0 and 2 * l <= NP; l *= 2)
        if (wslab_factor_rand(W, tol, l, seed, f)) return;
      wslab_factor_exact(W, tol, f);
    }

    // -------------------------------------------------------------------------------------
    //  stage 1c: ONE basis for all npk slabs of a q  (the union-subspace restructure)
    // -------------------------------------------------------------------------------------

    /** grow a (cap, NP) ROW-major basis buffer, preserving its first R rows. */
    inline void union_grow(nda::array<ComplexType, 2> &Q, long newcap, long R) {
      decltype(nda::range::all) all;
      nda::array<ComplexType, 2> Q2(newcap, Q.shape(1));
      Q2() = ComplexType(0.0);
      if (R > 0) Q2(nda::range(0, R), all) = Q(nda::range(0, R), all);
      Q = std::move(Q2);
    }

    /**
     * Build the union basis of one q's retained slab ranges. The basis is stored ROW-major,
     * Q(R, P) = q_R(P), because every consumer wants a leading-dimension-clean row block:
     *
     *   coefficients   A^(p)(R,r) = <q_R, v_r>   ->  At = transpose(V) * dagger(Q)   (r, R)
     *   sandwich       H(R,a) = sum_P q_R(P) B(P,a)  ->  H  = Q * B                  (R, nbnd)
     *
     * -- neither needs a conjugation of the stored basis or of the coefficient blocks (the
     * trev-q rule is carried on the (Np, nbnd) side instead; see stage 2).
     *
     * Incremental modified Gram-Schmidt with one re-orthogonalization, p in INDEX order, so
     * the result depends on nothing but (q, the slabs) -- every node builds the same basis.
     * A column v_j of V^(p) is appended only if the part of it that the current basis does
     * NOT span carries significant WEIGHT:
     *
     *     |s_j| * ||(1 - P) v_j||  >=  tol * max_p max|s^(p)|,
     *
     * i.e. the scale is the LARGEST residue slab of this q, not the slab the column belongs
     * to. That is deliberate and it is the difference between a restructure that pays and one
     * that does not:
     *
     *   - what the sandwich sums is sum_p B^T W^(p) conj(B), so the error that reaches
     *     Sigma^c is sum_p ||(1-P) W^(p)||, an ABSOLUTE quantity. A slab whose norm is 1e-6
     *     of the largest one may be projected badly in its OWN relative terms and still not
     *     move the answer;
     *   - measured (qe_lih222, wunion = 1e-8, Np = 192): the per-slab-relative criterion
     *     needs R = 186 of 192 -- the restructure buys nothing -- while this one needs 143,
     *     the same number the independent stack-rank probe reports at that tolerance;
     *   - it is also the criterion the "W^c union subspace" probe measures, so the ladder in
     *     this file's header IS the prediction for this basis.
     *
     * The price is that the tolerance is NOT a per-slab relative accuracy (qp_modea_wrank
     * is); the per-slab residual is measured and logged separately so the difference is never
     * hidden. With tol below the slab cut the basis spans every retained direction and the
     * restructure is EXACT -- but then R is the rank of the whole stack, which is Np at the
     * production cut (again, the ladder). That is the trade.
     */
    template<typename Vfn, typename Sfn>
    inline long union_build(long NP, long npk, double tol, nda::array<long, 1> const &rk,
                            nda::array<double, 1> const &amax, Vfn &&Vof, Sfn &&Sof,
                            nda::array<ComplexType, 2> &Q, long &nappend, long &ndrop) {
      decltype(nda::range::all) all;
      long cap = 64;
      for (long p = 0; p < npk; ++p) cap = std::max(cap, rk(p));
      cap = std::min(NP, cap + 64);
      Q = nda::array<ComplexType, 2>(cap, NP);
      Q() = ComplexType(0.0);
      long R = 0;
      nappend = 0;
      ndrop = 0;
      double am = 0.0;                         // the scale: the largest slab of this q
      for (long p = 0; p < npk; ++p) am = std::max(am, amax(p));
      if (am <= 0.0) am = 1.0;
      nda::array<ComplexType, 2> E(1, NP);
      std::vector<char> add;
      for (long p = 0; p < npk; ++p) {
        const long r = rk(p);
        if (r == 0) continue;
        auto V = Vof(p);                       // (NP, r), orthonormal columns
        auto s = Sof(p);                       // (r), signed eigenvalues
        add.assign(size_t(r), 1);
        if (R > 0) {
          // At(j,R) = <q_R, v_j> and the residual of the whole slab in TWO gemms. The norm is
          // taken on the residual VECTOR, never as 1 - ||At(j,:)||^2: that difference of two
          // numbers of size 1 cannot resolve below sqrt(eps) ~ 1.5e-8, which is above the cut
          // this criterion has to make.
          auto QR = Q(nda::range(0, R), all);
          nda::array<ComplexType, 2> At(r, R), Rt(r, NP);
          nda::blas::gemm(nda::transpose(V), nda::dagger(QR), At);
          nda::blas::gemm(At, QR, Rt);                 // Rt(j,P) = (P v_j)(P)
          for (long j = 0; j < r; ++j) {
            double d2 = 0.0;
            for (long P = 0; P < NP; ++P) d2 += std::norm(V(P, j) - Rt(j, P));
            add[size_t(j)] = ((std::abs(s(j)) / am) * std::sqrt(d2) >= tol) ? 1 : 0;
          }
        }
        for (long j = 0; j < r; ++j) {
          if (not add[size_t(j)]) { ++ndrop; continue; }
          if (R >= NP) { ++ndrop; continue; }
          for (long P = 0; P < NP; ++P) E(0, P) = V(P, j);
          double nrm = 1.0;
          for (int pass = 0; pass < 2; ++pass) {
            if (R > 0) {
              nda::array<ComplexType, 2> C(1, R);
              auto QR = Q(nda::range(0, R), all);
              nda::blas::gemm(E, nda::dagger(QR), C);                  // C = <q_R, e>
              nda::blas::gemm(ComplexType(-1.0), C, QR, ComplexType(1.0), E);
            }
            nrm = 0.0;
            for (long P = 0; P < NP; ++P) nrm += std::norm(E(0, P));
            nrm = std::sqrt(nrm);
            if (nrm > 0.5) break;   // no cancellation happened; the second pass is a no-op
          }
          // a residual this small is the numerical zero of an already-spanned direction
          if (nrm <= 1e-8) { ++ndrop; continue; }
          if (R == cap) {
            const long nc = std::min(NP, 2 * cap);
            if (nc == cap) { ++ndrop; continue; }
            union_grow(Q, nc, R);
            cap = nc;
          }
          for (long P = 0; P < NP; ++P) Q(R, P) = E(0, P) / nrm;
          ++R;
          ++nappend;
        }
      }
      return R;
    }

    // -------------------------------------------------------------------------------------
    //  RW-2: the SPECTRAL-QUADRATURE W^c representation
    //  (derivation + sign convention: methods/SCF/wc_spectral.hpp)
    // -------------------------------------------------------------------------------------

    /** Everything the spectral path produces before the residue slabs are filled. */
    struct spectral_ctx_t {
      bool on = false;
      // the real-axis grid actually used
      double eta = 0.0, w_max = 0.0, Omega_max = 0.0;
      long   N_w = 0, N_t = 0, N_O = 0;
      double dw = 0.0, dOmega = 0.0, T_window = 0.0, dt = 0.0;
      // the quadrature
      wc_spectral::quad_grid_t qg;          // (N_O + 1) quadrature nodes / weights / src
      wc_spectral::bin_plan_t plan;         // node -> pole coarsening
      // this rank's tile of Im W^c(q, P, Q, Omega)
      nda::array<double, 4> ImW;            // (nq_ibz, nPloc, nQloc, N_O)
      long P0 = 0, Q0 = 0, nPl = 0, nQl = 0;
      // the q = Gamma slot: appended LS poles carrying the WHOLE Gamma column (body + head).
      bool  gamma_ls = true;                // qp_modea_spectral_gamma == "ls"
      long npole_gamma = 0;
      nda::array<double, 1> om_gamma;       // (npole_gamma)
      nda::array<ComplexType, 1> r_head;    // ("spectral" mode) scalar eps_inv_head residues
      double gamma_rec = 0.0;               // Gamma-slot reconstruction rel err on the nu mesh
      long iq_gamma = 0;                    // IBZ index of the Gamma transfer
      // diagnostics
      double t_wall = 0.0;
      double imw_sym = 0.0;                 // max|ImW_PQ - ImW_QP| / max|ImW|
      double psd_neg = 0.0;                 // most negative eigenvalue share of a residue slab
      long   n_neg_trace = 0;
      double neg_frac = 0.0, width_worst = 0.0;
      long   nbin = 0;
    };

#ifdef ENABLE_FINUFFT
    /**
     * Run the RW-1 real-axis chain on the CURRENT QP spectrum and leave Im W^c(q,P,Q,Omega)
     * plus the quadrature grid in `sp`. Called once per mode-A context build.
     *
     * Convention pins, all inherited from the RW-1 gate and re-asserted here:
     *   - the SAME THC eri as the imaginary-axis side (`thc`), so the factorization is common
     *     mode and cancels out of every comparison;
     *   - the SAME beta and the SAME mu (grid.mu_chem() is the loop's mu);
     *   - the SAME div_treatment string, so the q = Gamma BODY is treated identically on both
     *     axes: under ignore_g0 both zero it, otherwise both solve the Dyson equation there
     *     with the regularized auxiliary V. (The q -> 0 HEAD is NOT taken from this chain --
     *     see the head sector in build_modea_context.)
     *   - eta enters ONLY through A; the grids are sized from eta by the RW-1 rules
     *     (real_axis_qp_A.hpp), and real_freq_grid_t enforces Nyquist as a hard error.
     */
    template<typename thc_t>
    void build_spectral_W(spectral_ctx_t &sp, thc_t &thc,
                          const sArray_t<Array_view_4D_t> &sMO_skia,
                          const sArray_t<Array_view_3D_t> &sE_ska,
                          double mu, double beta, modea_opts const &opts,
                          std::string const &div_treatment, int lvl) {
      using methods::real_axis::real_freq_grid_t;
      using methods::real_axis::real_axis_mb_state_t;
      using methods::real_axis::real_axis_scr_coulomb_t;
      const auto t0 = std::chrono::steady_clock::now();

      auto mpi = thc.mpi();
      auto MF  = thc.MF();
      const long ns      = sE_ska.shape()[0];
      const long nk_ibz  = sE_ska.shape()[1];
      const long nbnd    = sE_ska.shape()[2];
      const long Naux    = thc.Np();
      const long Nq_ibz  = MF->nqpts_ibz();

      utils::check(nk_ibz == MF->nkpts_ibz(),
                   "qp_modea (spectral): the QP block carries {} k-points but the mean field "
                   "has {} in the IBZ.", nk_ibz, MF->nkpts_ibz());

      // ---- the frequency window, from the CURRENT QP spectrum (absolute energies) --------
      double e_min = 1e300, e_max = -1e300;
      {
        auto E = sE_ska.local();
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk_ibz; ++k)
            for (long a = 0; a < nbnd; ++a) {
              const double e = E(s, k, a).real();
              e_min = std::min(e_min, e); e_max = std::max(e_max, e);
            }
      }
      sp.eta       = opts.spectral_eta;
      sp.w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
      sp.Omega_max = 2.0 * sp.w_max;
      auto gs = methods::real_axis::size_grids(sp.eta, sp.w_max, sp.Omega_max);
      sp.N_w = gs.N_w; sp.N_t = gs.N_t; sp.N_O = gs.N_Omega;
      sp.dw = gs.dw; sp.dOmega = gs.dOmega; sp.dt = gs.dt; sp.T_window = gs.T_window;

      app_log(lvl, "  - SPECTRAL W^c (RW-2):         eta = {:.4e} a.u. ({:.4g} eV); eps range "
                   "[{:+.4f}, {:+.4f}] Ha -> w_max = {:.4f}, Omega_max = {:.4f}",
              sp.eta, sp.eta * 27.211386245988, e_min, e_max, sp.w_max, sp.Omega_max);
      // THE CHEMICAL POTENTIAL. real_freq_grid_t::mu_chem is set once at construction and has
      // no setter, and it feeds BOTH the absolute-energy Fermi factors inside the fixed Pi
      // kernel AND the w_abs = w + mu convention of the A builder. It must therefore be the
      // LIVE loop mu at every rebuild -- on a metal it moves iteration to iteration. (The
      // branch's own dispatcher instead sets it once to the KS direct-gap midpoint; that is
      // defect #7 of notes/real_axis_refs_audit.md section 8.7a and it cannot occur here
      // because the context build is handed the loop's mu, but it is asserted and logged.)
      utils::check(std::isfinite(mu), "qp_modea (spectral): non-finite mu.");
      app_log(lvl, "  - SPECTRAL mu (LIVE):          grid.mu_chem() = {:.12f} a.u. -- the "
                   "CURRENT loop mu, rebuilt every outer iteration", mu);
      app_log(lvl, "  - SPECTRAL grids (derived):    N_w = {} (dw = {:.5f} = eta/{:.2f}), "
                   "N_t = {} (T = {:.1f} = {:.1f}/eta, dt = {:.4f}), N_Omega = {} "
                   "(dOmega = {:.5f} = eta/{:.2f})",
              sp.N_w, sp.dw, sp.eta / sp.dw, sp.N_t, sp.T_window, sp.T_window * sp.eta,
              sp.dt, sp.N_O, sp.dOmega, sp.eta / sp.dOmega);

      // GUARD. Every grid size is DERIVED from the QP spectrum through
      // w_max = max|eps| + 2, so a spectrum that has run away makes them run away with it:
      // measured on the RW-2-d leg, after a first map that left dmax(H_eff) at 1.7e+04 a.u.
      // the second map's sizing came out N_w = 1366129, N_t = 4194304, N_Omega = 683064 and
      // the job died of resources instead of saying why. A physics divergence must not
      // present as an OOM, so it is turned into a diagnosable abort here. The cap is far
      // above any converged case (SVO at eta = 0.05 needs N_w = 381, N_t = 2048,
      // N_Omega = 190; lih222 at eta = 0.0125 needs 1187 / 4096 / 657).
      {
        constexpr long grid_cap = 1L << 17;      // 131072
        utils::check(sp.N_w <= grid_cap and sp.N_t <= grid_cap and sp.N_O <= grid_cap,
                     "qp_modea (spectral): the derived real-axis grids are unreasonable "
                     "(N_w = {}, N_t = {}, N_Omega = {}, cap {}). They are set by the CURRENT "
                     "QP spectrum through w_max = max|eps| + 2 = {:.4g} a.u. (eps range "
                     "[{:+.4g}, {:+.4g}] Ha) and by qp_modea_spectral_eta = {:.4g}. A window "
                     "this wide means the quasiparticle spectrum has run away -- look at "
                     "dmax(H_eff) of the previous iteration -- not that the grid rule is "
                     "wrong. Fix the divergence, or raise eta.",
                     sp.N_w, sp.N_t, sp.N_O, grid_cap, sp.w_max, e_min, e_max, sp.eta);
      }

      real_freq_grid_t grid = real_freq_grid_t::make_uniform(
          beta, mu, sp.w_max, sp.N_w, sp.Omega_max, sp.N_O, sp.N_t, sp.T_window);

      real_axis_mb_state_t state(grid);
      state.mpi = mpi;
      state.A_wskij.emplace(*state.mpi,
          std::array<long, 5>{sp.N_w, ns, nk_ibz, nbnd, nbnd});
      if (state.A_wskij->node_comm()->root()) {
        auto MO_loc = sMO_skia.local();
        methods::real_axis::build_A_from_QP_poles(state.A_wskij->local(), grid,
                                                  sE_ska.local(), &MO_loc, sp.eta);
      }
      state.A_wskij->node_sync();

      // The k-space Pi branch hard-requires IBZ == FBZ; R-space is the production default
      // for Nk > 1 and is the only legal choice on a symmetry-reduced mesh.
      const bool use_rspace = (MF->nkpts() > 1);

      // THE q = Gamma BODY. The real-axis chain zeroes Pi AND W at Gamma when and only when
      // it is constructed with div_treatment == "ignore_g0" (real_axis_scr_coulomb_t.h:341,
      // :634, :875) -- the RW-1 gate's known structural difference D1. The IMAGINARY-axis
      // chain never does that: it solves the Dyson equation at Gamma with the regularized
      // auxiliary V = thc.Z(0) whatever div_treatment says, and the stored dW_qtPQ that the
      // LS routes fit therefore HAS a Gamma body. Zeroing it here would drop one q of nq
      // from Sigma^c and is measurable: on qe_lih222 it left the Lehmann meter at exactly
      // 1.0 (100 % relative) at q = 0. So the solver is always constructed with a
      // non-ignore_g0 string and the two axes treat the Gamma BODY identically.
      // This does NOT touch the q -> 0 HEAD, which stays the production Matsubara
      // eps_inv_head (spec item 3); the module's own eps_inv_head_O is computed as a side
      // effect of this choice and is never read.
      real_axis_scr_coulomb_t scr_re(&grid, "rpa", "gygi", 1e-9);
      (void)div_treatment;
      scr_re.update_w(state, thc, /*verbose*/ false, use_rspace);
      utils::check(state.ImW_qPQO.has_value(),
                   "qp_modea (spectral): the real-axis chain produced no Im W^c.");

      // ---- keep this rank's (P,Q) tile, plus the grid's own nodes/weights ---------------
      auto Pr = state.ImW_qPQO->local_range(1);
      auto Qr = state.ImW_qPQO->local_range(2);
      sp.P0 = Pr.first(); sp.Q0 = Qr.first();
      sp.nPl = Pr.size(); sp.nQl = Qr.size();
      sp.ImW = nda::array<double, 4>(state.ImW_qPQO->local());
      {
        nda::array<double, 1> Og(grid.Omega()), Ogw(grid.Omega_weights());
        sp.qg = wc_spectral::make_quad_grid(Og, Ogw);
      }
      utils::check(sp.ImW.shape()[0] == Nq_ibz and sp.ImW.shape()[3] == sp.N_O,
                   "qp_modea (spectral): Im W^c tile has shape ({}, {}, {}, {}), expected "
                   "({}, {}, {}, {}).", sp.ImW.shape()[0], sp.ImW.shape()[1],
                   sp.ImW.shape()[2], sp.ImW.shape()[3], Nq_ibz, sp.nPl, sp.nQl, sp.N_O);
      (void)Naux;
      sp.t_wall = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
      sp.on = true;
    }
#endif // ENABLE_FINUFFT

  } // detail

  /**
   * Build the mode-A evaluator context. `thc` must be a THC-ERI (the pair vectors are the
   * collocation columns); `sE_ska` / `sMO_skia` are the CURRENT outer-iteration QP spectrum
   * and MO coefficients, and both are frozen for the whole inner-consistency loop.
   *
   * @param need_diag  also build the replicated diagonal residues (the evGW leg needs to
   *                   evaluate Sigma_ii on a DIFFERENT processor grid than the block owner).
   */
  template<typename thc_t>
  void build_modea_context(modea_ctx &ctx, MBState &mb_state, thc_t &thc,
                           const sArray_t<Array_view_4D_t> &sMO_skia,
                           const sArray_t<Array_view_3D_t> &sE_ska,
                           double mu, const imag_axes_ft::IAFT &FT,
                           modea_opts const &opts, std::string const &div_treatment,
                           bool need_diag) {
    using math::shm::make_shared_array;
    decltype(nda::range::all) all;
    const auto t_start = std::chrono::steady_clock::now();

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long NP = thc.Np();
    const long nbnd = sE_ska.shape()[2];
    const long ns = sE_ska.shape()[0];
    const long nk_ibz = sE_ska.shape()[1];
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nsym = MF->qsymms().size();
    const int lvl = opts.level;

    ctx = modea_ctx{};
    ctx.opts = opts;
    ctx.beta = FT.beta();
    ctx.mu = mu;
    ctx.eta = opts.eta;
    ctx.eta_far = opts.eta_far;
    ctx.ns = ns;
    ctx.nk = nk_ibz;
    ctx.nbnd = nbnd;
    ctx.nkpts_full = nkpts;

    utils::check(FT.basis() == imag_axes_ft::dlr_basis,
                 "qp_modea: qp_map = \"mode_a\" requires the DLR imaginary-axis backend "
                 "(the support-constrained auxiliary pole fit lives there). Rerun with "
                 "iaft basis = \"dlr\".");
    utils::check(mb_state.dW_qtPQ.has_value(),
                 "qp_modea: mb_state.dW_qtPQ is empty -- the mode-A context must be built "
                 "inside the live W window (between update_w and the dW reset).");
    utils::check(thc.X(0, 0, 0).shape(1) == nbnd,
                 "qp_modea: the THC collocation carries {} orbitals but the QP block is {} "
                 "wide; mode_a needs the full band window.", thc.X(0, 0, 0).shape(1), nbnd);

    // ---------------- support constraint + the promoted fit ---------------------------
    imag_axes_ft::dlr_pole_fit pf(FT);
    double max_abs_node = 0.0;
    for (long p = 0; p < pf.np; ++p) max_abs_node = std::max(max_abs_node, std::abs(pf.epsl(p)));
    const double gap_edge =
        resolve_gap_edge_clamped(opts.wsupp, sE_ska.local(), mu, max_abs_node, lvl);

    // The stored dW lives on the PH-symmetric HALF tau grid, so tau_to_w_PHsym returns the
    // POSITIVE half of the bosonic Matsubara mesh (its own index convention:
    // full index nw_b/2 + n). W^c(i nu) is EVEN in nu -- that is exactly what the PH-sym
    // transform pair encodes -- so the full mesh is recovered by mirroring. Both fit routes
    // then see the FULL bosonic mesh, which is what the QM2-b chain measured and what lets
    // the fit reproduce an even function on a NONSYM auxiliary node set.
    // [verified: W(tau) = W(beta - tau) and e^{i nu beta} = 1 give W(-i nu) = W(i nu);
    //  IAFT.icc:48-72 is built on the same identity]
    const long nwb_full = FT.nw_b();
    const long nw_half = (nwb_full % 2 == 0) ? nwb_full / 2 : nwb_full / 2 + 1;
    nda::array<long, 1> half_of(nwb_full);
    nda::array<ComplexType, 1> zb(nwb_full);
    {
      auto wnb = FT.wn_mesh_b();
      for (long iw = 0; iw < nwb_full; ++iw) {
        zb(iw) = FT.omega(wnb(iw));
        const long target = std::abs(wnb(iw));
        long found = -1;
        for (long n = 0; n < nw_half; ++n)
          if (wnb(nwb_full / 2 + n) == target) { found = n; break; }
        utils::check(found >= 0, "qp_modea: bosonic node {} (n = {}) has no partner on the "
                                 "PH-symmetric positive half mesh.", iw, wnb(iw));
        half_of(iw) = found;
      }
    }

    // RW-2: the pole set itself is built AFTER the stage-1 preamble below, because the
    // spectral route needs the Gamma-head decision (and the head's own Matsubara data) to
    // size its appended head sector. Nothing between here and there depends on npk.
    const bool spectral = (opts.wfit == "spectral");
    ctx.diag.gap_edge = gap_edge;
    {   // global QP band edges of the CURRENT spectrum -- the strip test needs them
      double lo = -1e300, hi = 1e300;
      auto E = sE_ska.local();
      for (long is2 = 0; is2 < ns; ++is2)
        for (long ik2 = 0; ik2 < nk_ibz; ++ik2)
          for (long a = 0; a < nbnd; ++a) {
            const double e = E(is2, ik2, a).real();
            if (e < mu) lo = std::max(lo, e); else hi = std::min(hi, e);
          }
      ctx.vbm = lo; ctx.cbm = hi;
    }
    ctx.diag.np_total = pf.np;

    // reality of the tau <-> nu kernels: if Ttw_bb is real then Hermiticity of the STORED
    // W(tau) in (P,Q) transfers exactly to W(i nu), which is what the trev rule needs.
    {
      auto T = FT.Ttw_bb();
      double im = 0.0;
      for (auto const &v : T) im = std::max(im, std::abs(std::imag(ComplexType(v))));
      ctx.diag.ttw_imag = im;
    }

    // ---------------- stage 1: per-q residue slabs ------------------------------------
    auto &dW = mb_state.dW_qtPQ.value();
    auto [nq_ibz, nt_half, NPg, NQg] = dW.global_shape();
    utils::check(NPg == NP and NQg == NP,
                 "qp_modea: dW has (P,Q) = ({},{}) but thc.Np() = {}.", NPg, NQg, NP);
    auto q_rng = dW.local_range(0);
    auto t_rng = dW.local_range(1);
    auto P_rng = dW.local_range(2);
    auto Q_rng = dW.local_range(3);
    auto W_loc = dW.local();

    const long nwb = nwb_full, ntf = FT.nt_f();
    const long ncols = NP * NP;
    auto [c0, c1] = itertools::chunk_range(0, ncols, mpi->node_comm.size(),
                                           mpi->node_comm.rank());
    const long nc = c1 - c0;

    // the Gamma head (spec section 3). ignore_g0 -> absent on BOTH sides by construction,
    // which is the convention of every QM3 gate and of the QM3-c judge protocol.
    bool head_on = (div_treatment.find("gygi") != std::string::npos or div_treatment == "cvv");
    if (head_on and MF->nqpts_ibz() == 1) {
      app_log(lvl, "  - Gamma head:                  nqpts_ibz == 1 with div_treatment = {} "
                   "-> taking ignore_g0 (same downgrade as gw_t::Sigma_div_correction).",
              div_treatment);
      head_on = false;
    }
    if (head_on and not mb_state.eps_inv_head.has_value()) {
      app_warning("qp_modea: div_treatment = {} but mb_state.eps_inv_head is absent; the "
                  "mode-A Sigma^c is built WITHOUT the long-wavelength head while the "
                  "reference GW Sigma has it. The anchor gate will show the difference.",
                  div_treatment);
      head_on = false;
    }
    nda::array<ComplexType, 1> Hcol(head_on ? nc : 0);
    if (head_on) {
      // W^head_PQ(tau) = nk * madelung * eps_inv_head(tau) * conj(chi_P) * chi_Q.
      // [verified: algebraically identical to gw_t::Sigma_div_correction (thc_gw.icc:461-527),
      //  whose Delta_ij = -madelung * eps_inv_head(tau) * sum_PQ conj(X_Pi) X_Qj G_PQ(k)
      //  conj(chi_P) chi_Q is exactly the q = Gamma term of the main sum with this W added --
      //  the -1/nk prefactor cancels the nk here.]
      // [verified -- gate: test_qp_map_ab "qp_map_modeb_head_anchor" (2026-08-13), the tau
      //  anchor on qe_lih223 with div_treatment = gygi. MEASURED (block-normalized, the
      //  2026-08-13 semantics): head ON gives anchor 3.4578e-04, ratio 0.158 of its gate,
      //  from an ABSOLUTE deviation of 7.11e-05 a.u. over a block scale of 2.057e-01;
      //  the same fixture with the head OFF gives 4.1824e-04 from 6.77e-05 a.u. over
      //  1.620e-01. Turning the head on adds ~27% to |Sigma| and leaves the map-vs-reference
      //  ABSOLUTE deviation within 5% -- i.e. this augmentation and gw_t::Sigma_div_correction
      //  are the same physics to well inside the W-fit scale they share, and the head carries
      //  no error of its own that the anchor can resolve. Before that gate this was the ONE
      //  unexercised branch of the map (both QM3-b fixtures and the QM3-c judge run
      //  div_treatment = ignore_g0, where the head is absent on both sides).]
      auto chi = thc.basis_head()(0, all);
      const double mad = MF->madelung();
      for (long j = 0; j < nc; ++j) {
        const long P = (c0 + j) / NP, Q = (c0 + j) % NP;
        Hcol(j) = double(nkpts) * mad * std::conj(chi(P)) * chi(Q);
      }
      app_log(lvl, "  - Gamma head:                  ON (div_treatment = {}, madelung = "
                   "{:.6g}) -- gated by test_qp_map_ab \"qp_map_modeb_head_anchor\"",
              div_treatment, mad);
    } else {
      app_log(lvl, "  - Gamma head:                  OFF (div_treatment = {})", div_treatment);
    }

    // ================= THE POLE SET ===================================================
    // Two representation classes share everything downstream of this block: a list of pole
    // energies ctx.om(p) and, per q, a residue slab sWres(q, p, P, Q) such that
    //     W^c_PQ(q, z) = sum_p sWres(q, p, P, Q) / (z - om_p).
    std::optional<imag_axes_ft::masked_pole_fit> mpf_opt;
    detail::spectral_ctx_t sp;
    long npk = 0;
    if (not spectral) {
      mpf_opt = (opts.wfit == "nu")
                    ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, opts.wrtol)
                    : imag_axes_ft::masked_pole_fit::from_tau(pf, gap_edge, opts.wrtol);
      npk = mpf_opt->nkeep;
      ctx.om = nda::array<double, 1>(npk);
      ctx.nB = nda::array<double, 1>(npk);
      for (long p = 0; p < npk; ++p) {
        ctx.om(p) = mpf_opt->om(p);
        ctx.nB(p) = sigma_route_b::stable_nB(ctx.beta, ctx.om(p));
      }
      app_log(lvl, "  - W^c support constraint:      |eps_p| >= {:.6g} a.u. ({:.4g} eV) -- "
                   "{} of {} auxiliary nodes retained, {} singular directions",
              gap_edge, gap_edge * 27.211386245988, npk, pf.np, mpf_opt->n_kept);
      app_log(lvl, "  - W^c pole-fit route:          {} ({} rows), SVD cut rel_tol = {:.2g}",
              opts.wfit, mpf_opt->nrow, mpf_opt->rel_tol);
      // RW-2 spec item 5: on a metal E_PH is the k-mesh level spacing, not a physical gap,
      // so the support constraint strips REAL low-omega weight from W^c. Measured on SVO
      // (notes/qpgw_metal_mode_m0.md section 8a): 12 of 74 nodes, i.e. everything below
      // 0.52 eV, gone at the first map and 1 of 74 at the second -- set by the k mesh.
      if (gap_edge > 0.0 and gap_edge < 10.0 * M_PI / ctx.beta)
        app_warning("qp_modea: the W^c support constraint is active at gap_edge = {:.4g} a.u. "
                    "({:.4g} eV), which is INSIDE the Matsubara mesh-spacing class 10 pi/beta "
                    "= {:.4g} a.u. On a metal E_PH is the k-mesh level spacing rather than a "
                    "physical particle-hole gap, so this is stripping physical low-frequency "
                    "weight from W^c and how much is stripped depends on the k mesh. Use "
                    "qp_modea_wfit = \"spectral\" (the computed-support representation), or "
                    "qp_modea_wsupp = off.",
                    gap_edge, gap_edge * 27.211386245988, 10.0 * M_PI / ctx.beta);
    } else {
#ifndef ENABLE_FINUFFT
      utils::check(false,
                   "qp_modea: qp_modea_wfit = \"spectral\" needs the real-axis W chain, which "
                   "is only compiled with -DENABLE_FINUFFT=ON. Rebuild with that flag (see "
                   "notes/rw1_port_report.md section 8) or use qp_modea_wfit = tau|nu.");
#else
      // ---- (a) the computed Im W^c(q, P, Q, Omega) ----------------------------------
      detail::build_spectral_W(sp, thc, sMO_skia, sE_ska, mu, ctx.beta, opts,
                               div_treatment, lvl);

      // ---- (b) the trace weights t_j = tr A_j, summed over q -------------------------
      // A_j = -(1/pi) w_j Im W^c(Omega_j) is the residue matrix of the pole at +Omega_j
      // (wc_spectral.hpp eq. B) and is positive semi-definite; its trace steers the
      // coarsening. The diagonal lives on whichever ranks own (P == Q), so reduce.
      const long NQD = sp.qg.Om.shape(0);
      nda::array<double, 1> tj(NQD);
      tj() = 0.0;
      double imw_num = 0.0, imw_den = 0.0;
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long iP = 0; iP < sp.nPl; ++iP) {
          const long Pg = sp.P0 + iP;
          for (long iQ = 0; iQ < sp.nQl; ++iQ) {
            const long Qg = sp.Q0 + iQ;
            for (long j = 0; j < sp.N_O; ++j)
              imw_den = std::max(imw_den, std::abs(sp.ImW(iq, iP, iQ, j)));
            if (Pg != Qg) continue;
            for (long j = 0; j < NQD; ++j)
              tj(j) += -wc_spectral::inv_pi * sp.qg.Ow(j)
                     * sp.ImW(iq, iP, iQ, sp.qg.src(j));
          }
        }
      mpi->comm.all_reduce_in_place_n(tj.data(), tj.size(), std::plus<>{});
      // (P,Q) symmetry of the stored Im W^c -- the premise that makes the elementwise
      // imaginary part the matrix spectral function. Only measurable where a rank owns
      // both (P,Q) and (Q,P); a square-ish proc grid does for most of them.
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long iP = 0; iP < sp.nPl; ++iP)
          for (long iQ = 0; iQ < sp.nQl; ++iQ) {
            const long Pg = sp.P0 + iP, Qg = sp.Q0 + iQ;
            if (Pg < sp.Q0 or Pg >= sp.Q0 + sp.nQl) continue;
            if (Qg < sp.P0 or Qg >= sp.P0 + sp.nPl) continue;
            for (long j = 0; j < sp.N_O; ++j)
              imw_num = std::max(imw_num, std::abs(sp.ImW(iq, iP, iQ, j)
                                                 - sp.ImW(iq, Qg - sp.P0, Pg - sp.Q0, j)));
          }
      imw_num = mpi->comm.all_reduce_value(imw_num, boost::mpi3::max<>{});
      imw_den = mpi->comm.all_reduce_value(imw_den, boost::mpi3::max<>{});
      sp.imw_sym = (imw_den > 0.0) ? imw_num / imw_den : 0.0;

      // ---- (c) coarsening ------------------------------------------------------------
      sp.plan = wc_spectral::build_bins(sp.qg.Om, tj, opts.spectral_npole);
      sp.nbin = sp.plan.nbin;
      sp.n_neg_trace = sp.plan.n_neg;
      sp.neg_frac = sp.plan.neg_frac;
      sp.width_worst = sp.plan.width_worst;

      // ---- (d) THE q = GAMMA SLOT (per-q HYBRID; Fable ruling, 2026-08-20) -------------
      // The ported real-axis chain is asked here to Dyson Gamma like the Matsubara side
      // does (see the div_treatment note in build_spectral_W), but the Gamma column is
      // special on three counts -- it carries the 1/q^2 head, it is the largest |W| of the
      // mesh, and it is the one q whose real-axis treatment on the branch is a hard-coded
      // zero rather than a computed object. The RULING is therefore a PER-Q HYBRID: the
      // spectral quadrature represents q != Gamma, and Gamma keeps the EXISTING LS
      // representation of its whole Matsubara column (body + eps_inv_head augmentation),
      // on an APPENDED pole set. Every q's representation is logged.
      //
      // [DEVIATIONS, flagged: (i) this sector is NOT sign-definite -- it is exactly the LS
      //  object the body used to be, on 1 of nq transfers, and the Sabs probe reports its
      //  share of Sabs separately so any residual cancellation is measured; (ii) the fit is
      //  the "nu" route rather than the production "tau" default, which the QM3-b survey
      //  measured as the better-conditioned of the two at REAL z by an order in the residue
      //  ratio (qp_params_t.h: nu 1e-8 -> 4.3e2 vs tau 1e-8 -> 6.0e3); (iii) it carries the
      //  SAME support constraint gap_edge the production LS path uses.]
      sp.iq_gamma = 0;                 // the Gamma transfer, div_utils / embed_eri_t convention
      sp.gamma_ls = (opts.spectral_gamma != "spectral");
      if (sp.gamma_ls) {
        auto gfit = imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, opts.wrtol);
        sp.npole_gamma = gfit.nkeep;
        sp.om_gamma = nda::array<double, 1>(sp.npole_gamma);
        for (long p = 0; p < sp.npole_gamma; ++p) sp.om_gamma(p) = gfit.om(p);
      } else if (head_on) {
        // "spectral" Gamma: the BODY comes from the quadrature like every other q, and only
        // the SCALAR eps_inv_head rides appended LS poles.
        //
        // RW-2 FOLLOW-UP (report section 7 item 1, second half; the first implementation
        // fit this UNCONSTRAINED with a comment claiming the support constraint "does not
        // apply" to the head -- WRONG on both counts). eps_inv_head - 1 is built from the
        // particle-hole polarization, so its spectral support obeys the same E_PH bound as
        // the body; and on a metal under gygi_metal the head DATA is the enforced metallic
        // zero plus noise, so an unconstrained fit resolves NOISE into sub-mesh-spacing
        // poles whose n_B ~ 1/(beta*omega_p) weights are exactly the M-0b amplification
        // class. Measured on SVO (rw2 leg 6917684): 6 poles below 3 pi/beta, down to
        // |om_p| = 1.57e-06 a.u., carrying 86.7% of Sum|n_B| -- contaminating the eta = 0
        // probe rows. The constrained fit removes them; the Gamma-head reconstruction
        // meter (sp.gamma_rec, logged every build) reports what the constraint costs.
        auto const &eih = mb_state.eps_inv_head.value();
        utils::check(eih.shape(0) == nt_half,
                     "qp_modea (spectral): eps_inv_head has {} nodes, dW(tau) has {}.",
                     eih.shape(0), nt_half);
        nda::array<ComplexType, 2> ht(nt_half, 1), hw_half(nw_half, 1), hw(nwb, 1);
        for (long t = 0; t < nt_half; ++t) ht(t, 0) = ComplexType(eih(t).real(), 0.0);
        FT.tau_to_w_PHsym(ht, hw_half);
        for (long iw = 0; iw < nwb; ++iw) hw(iw, 0) = hw_half(half_of(iw), 0);
        auto hfit = imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge,
                                                                  opts.wrtol);
        auto hc = hfit.coeffs(hw);
        sp.npole_gamma = hfit.nkeep;
        sp.om_gamma = nda::array<double, 1>(sp.npole_gamma);
        sp.r_head   = nda::array<ComplexType, 1>(sp.npole_gamma);
        double hn = 0.0, hd = 0.0;
        for (long p = 0; p < sp.npole_gamma; ++p) {
          sp.om_gamma(p) = hfit.om(p);
          sp.r_head(p)   = hfit.residue_scale(p) * hc(p, 0);
        }
        for (long m = 0; m < nwb; ++m) {
          ComplexType rec(0.0, 0.0);
          for (long p = 0; p < sp.npole_gamma; ++p) rec += sp.r_head(p) / (zb(m) - sp.om_gamma(p));
          hn = std::max(hn, std::abs(rec - hw(m, 0)));
          hd = std::max(hd, std::abs(hw(m, 0)));
        }
        sp.gamma_rec = (hd > 0.0) ? hn / hd : 0.0;
      }

      // ---- (e) the pole list ---------------------------------------------------------
      npk = 2 * sp.nbin + sp.npole_gamma;
      ctx.om = nda::array<double, 1>(npk);
      ctx.nB = nda::array<double, 1>(npk);
      {
        auto om_s = wc_spectral::pole_list(sp.plan);
        for (long p = 0; p < 2 * sp.nbin; ++p) ctx.om(p) = om_s(p);
        for (long p = 0; p < sp.npole_gamma; ++p) ctx.om(2 * sp.nbin + p) = sp.om_gamma(p);
        for (long p = 0; p < npk; ++p)
          ctx.nB(p) = sigma_route_b::stable_nB(ctx.beta, ctx.om(p));
      }
      app_log(lvl, "  - W^c pole-fit route:          {} -- SPECTRAL QUADRATURE at "
                   "the {} transfers ({} Omega nodes (+1 virtual at Omega_0/2) -> "
                   "{} bins -> {} poles at +-Omega), appended LS poles at q = Gamma (IBZ "
                   "index {}, {} poles, support |eps_p| >= {:.6g} a.u.) => npk {}",
              sp.gamma_ls ? "PER-Q HYBRID (spectral_gamma = ls)"
                          : "SPECTRAL AT EVERY q (spectral_gamma = spectral; only the "
                            "eps_inv_head rides the appended poles)",
              sp.gamma_ls ? nq_ibz - 1 : nq_ibz, sp.N_O, sp.nbin, 2 * sp.nbin, sp.iq_gamma,
              sp.npole_gamma, sp.gamma_ls ? gap_edge : 0.0, npk);
      app_log(lvl, "  - SPECTRAL coarsening:         worst relative bin width = {:.3e}; "
                   "|Omega| in [{:.4e}, {:.4e}] a.u.; trace weight negative at {} of {} "
                   "nodes ({:.2e} of the total)",
              sp.width_worst, sp.plan.om_c(0), sp.plan.om_c(sp.nbin - 1),
              sp.n_neg_trace, sp.qg.Om.shape(0), sp.neg_frac);
      app_log(lvl, "  - SPECTRAL Im W^c symmetry:    max|ImW_PQ - ImW_QP| / max|ImW| = {:.3e} "
                   "(REPORTED, not required: the -Omega residue is the TRANSPOSE -A^T, so a "
                   "non-symmetric Im W^c is handled exactly; a value near 0 only means the "
                   "transpose happens to be a no-op on this fixture); real-axis chain wall "
                   "{:.2f} s", sp.imw_sym, sp.t_wall);
      app_log(lvl, "  - SPECTRAL Gamma slot:         q = {} carries {} appended LS poles "
                   "representing {}; the eps_inv_head augmentation is unchanged from the "
                   "tau/nu routes",
              sp.iq_gamma, sp.npole_gamma,
              sp.gamma_ls ? "the WHOLE Matsubara column (body + head)"
                          : "the eps_inv_head SCALAR only (the body is spectral)");
      // memory forecast: the two objects that are LINEAR in npk.
      app_log(lvl, "  - SPECTRAL cost forecast:      residue slabs {:.1f} GB (node-shared), "
                   "sandwich {:.1f} GB per owned (s,k) block",
              double(nq_ibz) * double(npk) * double(NP) * double(NP) * 16.0 / 1.073741824e9,
              double(nbnd) * double(nbnd) * double(nkpts) * double(nbnd) * double(npk)
                  * 16.0 / 1.073741824e9);
#endif
    }
    ctx.npk = npk;
    ctx.diag.n_support = npk;

    auto sWres = make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, {nq_ibz, npk, NP, NP});
    auto sWt = make_shared_array<nda::array_view<ComplexType, 3>>(*mpi, {nt_half, NP, NP});
    sWres.set_zero();
    sWres.win().fence();

    nda::array<ComplexType, 2> Wt(nt_half, nc), Whalf(nw_half, nc), Ww(nwb, nc);
    nda::array<ComplexType, 2> Wf((opts.wfit == "tau") ? ntf : 0, (opts.wfit == "tau") ? nc : 0);
    // RW-2: the spectral fill needs an internode all_reduce of sWres before the rep can be
    // read back, so its Lehmann meter runs in a second pass and the per-q reference is kept.
    nda::array<ComplexType, 3> Ww_all(spectral ? nq_ibz : 0, spectral ? nwb : 0,
                                      spectral ? nc : 0);
    double herm_num = 0.0, herm_den = 0.0, rec_worst = 0.0, fit_worst = 0.0, ratio_worst = 0.0;
    double rec_abs_worst = 0.0, rec_abs_den = 0.0, rec_abs_sum = 0.0, w_max_sum = 0.0;
    long rec_q = -1, rec_abs_q = -1;

    for (long iq = 0; iq < nq_ibz; ++iq) {
      sWt.set_zero();
      sWt.win().fence();
      if (iq >= q_rng.first() and iq < q_rng.last())
        sWt.local()(t_rng, P_rng, Q_rng) = W_loc(iq - q_rng.first(), all, all, all);
      sWt.win().fence();
      sWt.all_reduce();
      detail::herm_probe(sWt.local(), herm_num, herm_den);

      auto Wall = nda::reshape(sWt.local(), std::array<long, 2>{nt_half, ncols});
      for (long t = 0; t < nt_half; ++t)
        for (long j = 0; j < nc; ++j) Wt(t, j) = Wall(t, c0 + j);
      // q = 0 is the Gamma transfer (the convention of div_utils / embed_eri_t's head).
      if (head_on and iq == 0) {
        auto const &eih = mb_state.eps_inv_head.value();
        utils::check(eih.shape(0) == nt_half,
                     "qp_modea: eps_inv_head has {} nodes, dW(tau) has {}.",
                     eih.shape(0), nt_half);
        for (long t = 0; t < nt_half; ++t)
          for (long j = 0; j < nc; ++j) Wt(t, j) += eih(t).real() * Hcol(j);
      }
      if (nc > 0) {
        FT.tau_to_w_PHsym(Wt, Whalf);
        for (long iw = 0; iw < nwb; ++iw)
          for (long j = 0; j < nc; ++j) Ww(iw, j) = Whalf(half_of(iw), j);
      }

      if (spectral and sp.gamma_ls and iq == sp.iq_gamma) {
        // ---- THE GAMMA SLOT: the EXISTING LS representation, appended pole set ---------
        // Ww here is the FULL Matsubara Gamma column -- body plus the eps_inv_head
        // augmentation applied to Wt above -- so this is bit-for-bit the object the tau/nu
        // routes fit, on the same support-constrained node set, written into the appended
        // pole range. Node-comm column chunking, so it is node-complete and must NOT be
        // included in the internode all_reduce below (it is written after it).
        for (long m = 0; m < nwb; ++m)
          for (long j = 0; j < nc; ++j) Ww_all(iq, m, j) = Ww(m, j);
        mpi->node_comm.barrier();
        continue;
      }
      if (spectral) {
        // ---- SPECTRAL FILL (RW-2, wc_spectral.hpp eq. B) -----------------------------
        // Poles 0..nbin-1 sit at +Omega_b with residue A_b = -(1/pi) sum_{j in b} w_j
        // Im W^c(Omega_j); poles nbin..2nbin-1 sit at -Omega_b with residue -A_b^T -- the
        // TRANSPOSE, because the bosonic reflection of a MATRIX response is
        // Im W_PQ(-Omega) = -Im W_QP(Omega) and Im W^c is NOT (P,Q)-symmetric in general
        // (measured 5.3e-01 on SVO against 1.1e-13 on qe_lih222). The transpose is applied
        // by writing the -Omega element at the SWAPPED global position, which keeps every
        // access inside this rank's own tile: the union of the tiles covers all (P,Q), so
        // the union of the swapped writes covers all (Q,P).
        // This rank writes its OWN global (P,Q) tile of the real-axis distribution -- a
        // different partition from the (c0, c1) column chunk used above -- so the slabs
        // are completed by an internode all_reduce after the loop.
        for (long p = 0; p < sp.nbin; ++p) {
          const long lo = sp.plan.lo(p), hi = sp.plan.hi(p);
          for (long iP = 0; iP < sp.nPl; ++iP)
            for (long iQ = 0; iQ < sp.nQl; ++iQ) {
              double acc = 0.0;
              for (long j = lo; j < hi; ++j)
                acc += sp.qg.Ow(j) * sp.ImW(iq, iP, iQ, sp.qg.src(j));
              const ComplexType A(-wc_spectral::inv_pi * acc, 0.0);
              sWres.local()(iq, p, sp.P0 + iP, sp.Q0 + iQ) = A;
              sWres.local()(iq, sp.nbin + p, sp.Q0 + iQ, sp.P0 + iP) = -A;
            }
        }
        for (long m = 0; m < nwb; ++m)
          for (long j = 0; j < nc; ++j) Ww_all(iq, m, j) = Ww(m, j);
        mpi->node_comm.barrier();
        continue;
      }

      auto const &mpf = *mpf_opt;
      nda::array<ComplexType, 2> cfit;
      if (opts.wfit == "nu") {
        cfit = mpf.coeffs(Ww);
        if (nc > 0) {
          fit_worst = std::max(fit_worst, mpf.fit_error(Ww, cfit));
          ratio_worst = std::max(ratio_worst, mpf.residue_ratio(Ww, cfit));
        }
      } else {
        if (nc > 0) nda::blas::gemm(FT.Ttw_bb(), Ww, Wf);
        cfit = mpf.coeffs(Wf);
        if (nc > 0) {
          fit_worst = std::max(fit_worst, mpf.fit_error(Wf, cfit));
          ratio_worst = std::max(ratio_worst, mpf.residue_ratio(Wf, cfit));
        }
      }
      for (long p = 0; p < npk; ++p)
        for (long j = 0; j < nc; ++j) cfit(p, j) *= mpf.residue_scale(p);

      // QUALITY METRIC (binding requirement 3): the bosonic-mesh reconstruction of the
      // FITTED representation. NEVER the tau-space fit residual.
      {
        double num = 0.0, den = 0.0;
        for (long m = 0; m < nwb; ++m) {
          const ComplexType z = zb(m);
          for (long j = 0; j < nc; ++j) {
            ComplexType rec(0.0);
            for (long p = 0; p < npk; ++p) rec += cfit(p, j) / (z - ctx.om(p));
            num = std::max(num, std::abs(rec - Ww(m, j)));
            den = std::max(den, std::abs(Ww(m, j)));
          }
        }
        if (den > 0.0 and num / den > rec_worst) { rec_worst = num / den; rec_q = iq; }
        // ... and the ABSOLUTE error of the same reconstruction. The gate's class is the
        // RELATIVE number above, normalized per q by max|W_q| -- but Sigma sums the q mesh
        // with band factors of a common scale, so what reaches the tau anchor is the
        // ABSOLUTE error, and the two orderings differ by orders once the mesh resolves the
        // 1/q^2 growth of W near Gamma (max|W_q| itself is printed so the ratio is readable).
        if (num > rec_abs_worst) { rec_abs_worst = num; rec_abs_q = iq; rec_abs_den = den; }
        // THE ERROR BUDGET. Sigma^c_ab = -(1/nk) sum_q sum_PQ A_P W_PQ(q) conj(A_Q) with
        // band factors A of a q-independent scale, so the RELATIVE error the tau anchor
        // measures is bounded by (sum_q |dW_q|) / (sum_q max|W_q|) up to the correlation
        // between dW and the band factors. That ratio is the anchor scale the W
        // representation ALONE predicts -- if the measured anchor sits at it, there is no
        // routing error left to look for.
        rec_abs_sum += num;
        w_max_sum += den;
      }

      for (long p = 0; p < npk; ++p)
        for (long j = 0; j < nc; ++j)
          sWres.local()(iq, p, (c0 + j) / NP, (c0 + j) % NP) = cfit(p, j);

      // ---- ONE-OFF fit-conditioning survey on the first q (cheap; diagnostics only) -----
      // Route B at REAL z is destroyed by NEAR-CANCELLING residues, not by the imaginary-axis
      // fit error (dlr_pole_fit::residue_ratio documents exactly this failure mode). Report
      // the (reconstruction, residue-ratio) pair for the unconstrained fit and for the
      // support-constrained fit at a few FIXED SVD cuts, so the choice is measured.
      if (iq == 0 and nc > 0) {
        auto probe = [&](std::string const &name, imag_axes_ft::masked_pole_fit const &f,
                         nda::array<ComplexType, 2> const &data) {
          auto cc = f.coeffs(data);
          const double own = f.fit_error(data, cc);
          const double rr = f.residue_ratio(data, cc);
          double num = 0.0, den = 0.0;
          for (long m = 0; m < nwb; ++m)
            for (long j = 0; j < nc; ++j) {
              ComplexType rec(0.0);
              for (long q2 = 0; q2 < f.nkeep; ++q2)
                rec += f.residue_scale(q2) * cc(q2, j) / (zb(m) - f.om(q2));
              num = std::max(num, std::abs(rec - Ww(m, j)));
              den = std::max(den, std::abs(Ww(m, j)));
            }
          app_log(lvl + 1, "    [fit survey q=0] {:<34} nodes {:>3}/{:<3} rank {:>3}  bosonic-mesh "
                       "rec = {:.3e}  own-grid = {:.3e}  residue ratio = {:.3e}",
                  name, f.nkeep, f.np_all, f.n_kept, (den > 0.0 ? num / den : 0.0), own, rr);
        };
        auto const &data = (opts.wfit == "nu") ? Ww : Wf;
        for (double rt : {1e-8, 1e-6, 1e-4, 1e-2}) {
          auto f0 = (opts.wfit == "nu")
                        ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, 0.0, rt)
                        : imag_axes_ft::masked_pole_fit::from_tau(pf, 0.0, rt);
          probe(std::format("plain,      rel_tol = {:.0e}", rt), f0, data);
          auto f1 = (opts.wfit == "nu")
                        ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, rt)
                        : imag_axes_ft::masked_pole_fit::from_tau(pf, gap_edge, rt);
          probe(std::format("support-constrained, rel_tol = {:.0e}", rt), f1, data);
        }
      }
      mpi->node_comm.barrier();
    }
    sWres.win().fence();

    if (spectral) {
      // ---- complete the body slabs across nodes -------------------------------------
      // Each rank wrote its own global (P,Q) tile of the real-axis distribution and zeros
      // elsewhere, so the internode sum is the completion (within a node the tiles are
      // already disjoint). This is the ONE collective the spectral path adds.
      sWres.all_reduce();
      sWres.win().fence();

      // ---- the q = Gamma slot, on the appended LS pole range ------------------------
      // Written with the (c0, c1) column chunking, which is node-complete, so it must come
      // AFTER the all_reduce above or it would be double counted.
      if (not sp.gamma_ls) {
        // "spectral" Gamma: only the rank-one head sector rides the appended poles.
        for (long p = 0; p < sp.npole_gamma; ++p)
          for (long j = 0; j < nc; ++j)
            sWres.local()(sp.iq_gamma, 2 * sp.nbin + p, (c0 + j) / NP, (c0 + j) % NP)
                = sp.r_head(p) * Hcol(j);
        sWres.win().fence();
        app_log(lvl, "  - SPECTRAL Gamma head fit:     bosonic-mesh reconstruction of the "
                     "scalar eps_inv_head on its {} appended poles = {:.4e}",
                sp.npole_gamma, sp.gamma_rec);
      } else {
        auto gfit = imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, opts.wrtol);
        nda::array<ComplexType, 2> Wg(nwb, nc);
        for (long m = 0; m < nwb; ++m)
          for (long j = 0; j < nc; ++j) Wg(m, j) = Ww_all(sp.iq_gamma, m, j);
        auto cg = gfit.coeffs(Wg);
        for (long p = 0; p < sp.npole_gamma; ++p)
          for (long j = 0; j < nc; ++j) cg(p, j) *= gfit.residue_scale(p);
        double gn = 0.0, gd = 0.0;
        for (long m = 0; m < nwb; ++m)
          for (long j = 0; j < nc; ++j) {
            ComplexType rec(0.0);
            for (long p = 0; p < sp.npole_gamma; ++p) rec += cg(p, j) / (zb(m) - sp.om_gamma(p));
            gn = std::max(gn, std::abs(rec - Wg(m, j)));
            gd = std::max(gd, std::abs(Wg(m, j)));
          }
        sp.gamma_rec = mpi->comm.all_reduce_value((gd > 0.0) ? gn / gd : 0.0,
                                                  boost::mpi3::max<>{});
        for (long p = 0; p < sp.npole_gamma; ++p)
          for (long j = 0; j < nc; ++j)
            sWres.local()(sp.iq_gamma, 2 * sp.nbin + p, (c0 + j) / NP, (c0 + j) % NP) = cg(p, j);
        sWres.win().fence();
        app_log(lvl, "  - SPECTRAL Gamma LS fit:       bosonic-mesh reconstruction of the "
                     "q = {} column on its {} appended poles = {:.4e}",
                sp.iq_gamma, sp.npole_gamma, sp.gamma_rec);
      }

      // ---- THE PRODUCTION METER (spec item 4) --------------------------------------
      // The Lehmann forward map of the quadrature representation, evaluated at the bosonic
      // Matsubara nodes, against the STORED W^c(i nu) of the imaginary-axis solver. This is
      // the two-sided anchor: it is the same number the LS routes report as
      // rec_rel_worst, so the tau-anchor gate downstream scales with it unchanged.
      // The STATIC node nu = 0 sits ON the real axis, where the dispersion relation carries
      // an extra boundary term i Im W^c(Omega -> 0). That term is ANTISYMMETRIC in (P,Q)
      // (section 2 of wc_spectral.hpp) and is a logarithm, not a pole, so no real-residue
      // pole set can carry it: if the stored W^c(i nu = 0) has an imaginary part, the
      // quadrature misses exactly that much there and nowhere else. The meter is therefore
      // split -- nu = 0 vs nu != 0 -- and the imaginary content of the reference at nu = 0
      // is reported, so a large meter can be attributed instead of guessed.
      double nu0_dev = 0.0, nu0_den = 0.0, nu0_imag = 0.0;
      for (long iq = 0; iq < nq_ibz; ++iq) {
        double num = 0.0, den = 0.0, rmax = 0.0;
        for (long j = 0; j < nc; ++j) {
          const long Pg = (c0 + j) / NP, Qg = (c0 + j) % NP;
          for (long p = 0; p < npk; ++p)
            rmax = std::max(rmax, std::abs(sWres.local()(iq, p, Pg, Qg)));
          for (long m = 0; m < nwb; ++m) {
            const ComplexType z = zb(m);
            ComplexType rec(0.0);
            for (long p = 0; p < npk; ++p)
              rec += sWres.local()(iq, p, Pg, Qg) / (z - ctx.om(p));
            num = std::max(num, std::abs(rec - Ww_all(iq, m, j)));
            den = std::max(den, std::abs(Ww_all(iq, m, j)));
            if (std::abs(z) < 1e-12) {
              nu0_dev = std::max(nu0_dev, std::abs(rec - Ww_all(iq, m, j)));
              nu0_den = std::max(nu0_den, std::abs(Ww_all(iq, m, j)));
              nu0_imag = std::max(nu0_imag, std::abs(Ww_all(iq, m, j).imag()));
            }
          }
        }
        if (den > 0.0 and num / den > rec_worst) { rec_worst = num / den; rec_q = iq; }
        if (num > rec_abs_worst) { rec_abs_worst = num; rec_abs_q = iq; rec_abs_den = den; }
        if (den > 0.0) ratio_worst = std::max(ratio_worst, rmax / den);
        rec_abs_sum += num;
        w_max_sum += den;
      }
      Ww_all = nda::array<ComplexType, 3>();
      nu0_dev  = mpi->comm.all_reduce_value(nu0_dev,  boost::mpi3::max<>{});
      nu0_den  = mpi->comm.all_reduce_value(nu0_den,  boost::mpi3::max<>{});
      nu0_imag = mpi->comm.all_reduce_value(nu0_imag, boost::mpi3::max<>{});
      app_log(lvl, "  - SPECTRAL static node:        rel dev at nu = 0 is {:.4e}; the stored "
                   "reference there carries max|Im W^c| / max|W^c| = {:.3e} (the part the "
                   "real-residue pole class cannot represent -- see the nu = 0 boundary term "
                   "in wc_spectral.hpp section 2)",
              (nu0_den > 0.0 ? nu0_dev / nu0_den : 0.0),
              (nu0_den > 0.0 ? nu0_imag / nu0_den : 0.0));

      // ---- definiteness audit of the merged residue slabs --------------------------
      // Eq. (C) says the SYMMETRIC part of A_b is positive semi-definite; its diagonal is
      // the diagonal of A_b itself, which is the cheap necessary condition probed here
      // (diagonalizing a (Np x Np) slab per (q, b) is what stage 1b does anyway).
      // Reported, never gated: a violation means the computed Im W^c does not have a
      // negative semi-definite symmetric part, which is a statement about the real-axis
      // chain, not about this representation.
      {
        double worst = 0.0;
        for (long iq = 0; iq < nq_ibz; ++iq) {
          if (sp.gamma_ls and iq == sp.iq_gamma) continue;   // LS slabs there, not quadrature
          for (long b = 0; b < sp.nbin; ++b) {
            double dmin = 1e300, dmax = -1e300;
            for (long j = 0; j < nc; ++j) {
              const long Pg = (c0 + j) / NP, Qg = (c0 + j) % NP;
              if (Pg != Qg) continue;
              const double d = sWres.local()(iq, b, Pg, Qg).real();
              dmin = std::min(dmin, d); dmax = std::max(dmax, d);
            }
            if (dmax > 0.0 and dmin < 0.0) worst = std::max(worst, -dmin / dmax);
          }
        }
        sp.psd_neg = mpi->comm.all_reduce_value(worst, boost::mpi3::max<>{});
      }
      app_log(lvl, "  - SPECTRAL definiteness:       worst |min diag| / max diag of a merged "
                   "residue slab = {:.3e} (eq. B predicts 0: A_b is positive semi-definite)",
              sp.psd_neg);
      auto &SPD = ctx.diag;
      SPD.sp_eta = sp.eta;
      SPD.sp_NO = sp.N_O; SPD.sp_Nw = sp.N_w; SPD.sp_Nt = sp.N_t;
      SPD.sp_nbin = sp.nbin; SPD.sp_nhead = sp.npole_gamma;
      SPD.sp_width = sp.width_worst; SPD.sp_negfrac = sp.neg_frac;
      SPD.sp_sym = sp.imw_sym; SPD.sp_psdneg = sp.psd_neg;
      SPD.sp_headrec = sp.gamma_rec; SPD.sp_wall = sp.t_wall;
    }

    ctx.diag.w_herm_rel = (herm_den > 0.0) ? herm_num / herm_den : 0.0;
    ctx.diag.w_herm_rel = mpi->comm.all_reduce_value(ctx.diag.w_herm_rel, boost::mpi3::max<>{});
    ctx.diag.rec_rel_worst = mpi->comm.all_reduce_value(rec_worst, boost::mpi3::max<>{});
    ctx.diag.fit_err_worst = mpi->comm.all_reduce_value(fit_worst, boost::mpi3::max<>{});
    ctx.diag.res_ratio_worst = mpi->comm.all_reduce_value(ratio_worst, boost::mpi3::max<>{});
    {   // the ABSOLUTE reconstruction error and where each worst case sits (see above)
      const double a_max = mpi->comm.all_reduce_value(rec_abs_worst, boost::mpi3::max<>{});
      if (rec_abs_worst < a_max) { rec_abs_q = -1; rec_abs_den = 0.0; }
      if (rec_worst < ctx.diag.rec_rel_worst) rec_q = -1;
      ctx.diag.rec_abs_worst = a_max;
      ctx.diag.rec_abs_q = mpi->comm.all_reduce_value(rec_abs_q, boost::mpi3::max<>{});
      ctx.diag.rec_abs_wmax = mpi->comm.all_reduce_value(rec_abs_den, boost::mpi3::max<>{});
      ctx.diag.rec_rel_q = mpi->comm.all_reduce_value(rec_q, boost::mpi3::max<>{});
      // the budget is the WORST column slice: each rank holds a slice of every q, so its
      // (sum_q |dW|) / (sum_q max|W|) is one sample of the same ratio -- reduce the ratio,
      // not the two sums separately (they would come from different ranks).
      const double br = (w_max_sum > 0.0) ? rec_abs_sum / w_max_sum : 0.0;
      ctx.diag.rec_budget = mpi->comm.all_reduce_value(br, boost::mpi3::max<>{});
    }
    Wt = nda::array<ComplexType, 2>();
    Whalf = nda::array<ComplexType, 2>();
    Ww = nda::array<ComplexType, 2>();
    Wf = nda::array<ComplexType, 2>();
    const double t_fit = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    // ---------------- stage 1b: low-rank factorization of the residue slabs -------------
    // W^(p) = V S V^dag per (q,p), truncated at |s| >= wrank * max|s|. The (q,p) axis is
    // partitioned over the NODE's ranks (the factors, like the slabs, are node-shared), the
    // seed of the randomized backend depends only on (q,p), and the truncation residual of
    // every slab is reported -- there is no silent accuracy change here.
    const auto t_fac0 = std::chrono::steady_clock::now();
    const bool lowrank = (opts.wrank > 0.0);
    long rcap = 0;
    nda::array<long, 2> rk_qp(nq_ibz, npk);
    nda::array<double, 2> tail_qp(nq_ibz, npk), frob_qp(nq_ibz, npk), anti_qp(nq_ibz, npk);
    nda::array<double, 2> amax_qp(nq_ibz, npk);
    nda::array<long, 3> lad_qp(nq_ibz, npk, long(detail::wrank_ladder.size()));
    nda::array<long, 2> exact_qp(nq_ibz, npk), probe_qp(nq_ibz, npk);
    rk_qp() = 0; tail_qp() = 0.0; frob_qp() = 0.0; anti_qp() = 0.0; amax_qp() = 0.0;
    lad_qp() = 0; exact_qp() = 0; probe_qp() = 0;
    std::vector<detail::wslab_factor> myf;
    long f0 = 0, f1 = 0;
    if (lowrank) {
      std::tie(f0, f1) = itertools::chunk_range(0, nq_ibz * npk, mpi->node_comm.size(),
                                               mpi->node_comm.rank());
      for (long j = f0; j < f1; ++j) {
        const long iq = j / npk, p = j % npk;
        detail::wslab_factor f;
        detail::wslab_factorize(sWres.local()(iq, p, all, all), opts.wrank, opts.wsketch,
                                0x9E3779B97F4A7C15ull + 1000003ull * (unsigned long long)(j), f);
        rk_qp(iq, p) = f.r;
        tail_qp(iq, p) = f.tail;
        frob_qp(iq, p) = f.frob;
        anti_qp(iq, p) = f.anti;
        amax_qp(iq, p) = f.amax;
        exact_qp(iq, p) = f.exact ? 1 : 0;
        probe_qp(iq, p) = f.nprobe;
        for (size_t t = 0; t < detail::wrank_ladder.size(); ++t) lad_qp(iq, p, long(t)) = f.ladder[t];
        myf.push_back(std::move(f));
      }
      // every node covers the whole (q,p) set among its own ranks, so a plus-reduce over the
      // NODE communicator (never over comm -- that would multiply by the node count) makes
      // the census complete and identical on all of them.
      mpi->node_comm.all_reduce_in_place_n(rk_qp.data(), rk_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(tail_qp.data(), tail_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(frob_qp.data(), frob_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(anti_qp.data(), anti_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(amax_qp.data(), amax_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(lad_qp.data(), lad_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(exact_qp.data(), exact_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(probe_qp.data(), probe_qp.size(), std::plus<>{});
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long p = 0; p < npk; ++p) rcap = std::max(rcap, rk_qp(iq, p));
      // the shared window must have the same shape on every node
      rcap = mpi->comm.all_reduce_value(rcap, boost::mpi3::max<>{});
    }
    auto sWv = make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, {lowrank ? nq_ibz : 1L, lowrank ? npk : 1L, lowrank ? NP : 1L,
               std::max(rcap, 1L)});
    auto sWs = make_shared_array<nda::array_view<double, 3>>(
        *mpi, {lowrank ? nq_ibz : 1L, lowrank ? npk : 1L, std::max(rcap, 1L)});
    if (lowrank) {
      sWv.set_zero();
      sWs.set_zero();
      sWv.win().fence();
      sWs.win().fence();
      for (long j = f0; j < f1; ++j) {
        auto const &f = myf[size_t(j - f0)];
        const long iq = j / npk, p = j % npk;
        if (f.r > 0) {
          sWv.local()(iq, p, all, nda::range(0, f.r)) = f.V;
          sWs.local()(iq, p, nda::range(0, f.r)) = f.s;
        }
      }
      sWv.win().fence();
      sWs.win().fence();
      mpi->node_comm.barrier();
      // the factors are in the shared window now; drop the rank-local copies. Without this
      // every rank keeps its own (nq*npk/node_size) slabs alive for the rest of the build --
      // a second copy of the whole factor set per node, invisible to the mem_mb below.
      myf.clear();
      myf.shrink_to_fit();
    }
    const double t_fac = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_fac0).count();

    // ---------------- stage 1c: the union subspace of one q's slab stack ---------------
    // ONE orthonormal basis Q_q for all npk retained slab ranges of q, and the (r_p, R_q)
    // coefficient blocks that express every slab in it. Stage 2 then contracts the Np axis
    // R_q times per (k,q,n) instead of sum_p r_p times -- see the flop model in the header.
    // The q axis of the CONSTRUCTION is partitioned over the node's ranks (it is sequential
    // in p, so it does not parallelize further), the PROJECTION over the (q,p) axis like
    // stage 1b, and both are deterministic: every node builds the same basis for the same q.
    const auto t_un0 = std::chrono::steady_clock::now();
    const double wu = (opts.wunion > 0.0) ? opts.wunion : opts.wrank;
    const bool union_on = lowrank and (opts.wunion >= 0.0);
    nda::array<long, 1> Rq(nq_ibz);
    nda::array<double, 2> utail_qp(nq_ibz, npk), ufrob_qp(nq_ibz, npk);
    Rq() = 0; utail_qp() = 0.0; ufrob_qp() = 0.0;
    long Rcap = 0, nappend = 0, ndrop = 0, uq0 = 0, uq1 = 0;
    std::vector<nda::array<ComplexType, 2>> myQ;
    if (union_on) {
      std::tie(uq0, uq1) = itertools::chunk_range(0, nq_ibz, mpi->node_comm.size(),
                                                  mpi->node_comm.rank());
      for (long iq = uq0; iq < uq1; ++iq) {
        nda::array<ComplexType, 2> Q;
        long nap = 0, ndr = 0;
        Rq(iq) = detail::union_build(
            NP, npk, wu, rk_qp(iq, all), amax_qp(iq, all),
            [&](long p) { return sWv.local()(iq, p, all, nda::range(0, rk_qp(iq, p))); },
            [&](long p) { return sWs.local()(iq, p, nda::range(0, rk_qp(iq, p))); },
            Q, nap, ndr);
        nappend += nap;
        ndrop += ndr;
        myQ.push_back(std::move(Q));
      }
      mpi->node_comm.all_reduce_in_place_n(Rq.data(), Rq.size(), std::plus<>{});
      for (long iq = 0; iq < nq_ibz; ++iq) Rcap = std::max(Rcap, Rq(iq));
      // the shared window must have the same shape on every node
      Rcap = mpi->comm.all_reduce_value(Rcap, boost::mpi3::max<>{});
      // node_comm, never comm: every node builds the whole q set among its own ranks, so a
      // global reduce would multiply the census by the node count (same rule as stage 1b).
      nappend = mpi->node_comm.all_reduce_value(nappend, std::plus<>{});
      ndrop = mpi->node_comm.all_reduce_value(ndrop, std::plus<>{});
    }
    // Q is stored ROW-major (R, Np) and the coefficients TRANSPOSED (r, R): both are then
    // leading-dimension-clean row blocks at the exact rank in use, and neither needs a
    // conjugation (the trev-q rule rides on the (Np, nbnd) side of stage 2).
    auto sQ = make_shared_array<nda::array_view<ComplexType, 3>>(
        *mpi, {union_on ? nq_ibz : 1L, std::max(Rcap, 1L), union_on ? NP : 1L});
    auto sWa = make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, {union_on ? nq_ibz : 1L, union_on ? npk : 1L,
               union_on ? std::max(rcap, 1L) : 1L, std::max(Rcap, 1L)});
    if (union_on) {
      sQ.win().fence();
      for (long iq = uq0; iq < uq1; ++iq)
        if (Rq(iq) > 0)
          sQ.local()(iq, nda::range(0, Rq(iq)), all) =
              myQ[size_t(iq - uq0)](nda::range(0, Rq(iq)), all);
      sQ.win().fence();
      myQ.clear();
      myQ.shrink_to_fit();
      sWa.win().fence();
      auto [g0, g1] = itertools::chunk_range(0, nq_ibz * npk, mpi->node_comm.size(),
                                             mpi->node_comm.rank());
      for (long j = g0; j < g1; ++j) {
        const long iq = j / npk, p = j % npk;
        const long r = rk_qp(iq, p);
        if (r == 0 or Rq(iq) == 0) continue;
        auto At = sWa.local()(iq, p, nda::range(0, r), all);          // (r, Rcap)
        nda::blas::gemm(nda::transpose(sWv.local()(iq, p, all, nda::range(0, r))),
                        nda::dagger(sQ.local()(iq, all, all)), At);
        // WHAT THE PROJECTION COSTS THIS SLAB. Two numbers, because the criterion and the
        // stage-1b truncation are normalized differently (see union_build): the 2-norm one is
        // scaled by the LARGEST slab of this q -- that is what wunion bounds -- and the
        // Frobenius one is relative to THIS slab, i.e. directly comparable to the stage-1b
        // residual. Both are measured on the residual VECTOR (the 1 - ||At||^2 shortcut has a
        // sqrt(eps) floor and would report 1e-8 for an EXACT projection).
        double am = 0.0;
        for (long p2 = 0; p2 < npk; ++p2) am = std::max(am, amax_qp(iq, p2));
        if (am <= 0.0) am = 1.0;
        double tmax = 0.0, num = 0.0, den = 0.0;
        nda::array<ComplexType, 2> Rt(r, NP);
        nda::blas::gemm(At, sQ.local()(iq, all, all), Rt);
        auto Vp = sWv.local()(iq, p, all, nda::range(0, r));
        for (long jj = 0; jj < r; ++jj) {
          double d2 = 0.0;
          for (long P = 0; P < NP; ++P) d2 += std::norm(Vp(P, jj) - Rt(jj, P));
          const double s2 = sWs.local()(iq, p, jj) * sWs.local()(iq, p, jj);
          tmax = std::max(tmax, std::sqrt(s2 * d2) / am);
          num += s2 * d2;
          den += s2;
        }
        utail_qp(iq, p) = tmax;
        ufrob_qp(iq, p) = (den > 0.0) ? std::sqrt(num / den) : 0.0;
      }
      sWa.win().fence();
      mpi->node_comm.all_reduce_in_place_n(utail_qp.data(), utail_qp.size(), std::plus<>{});
      mpi->node_comm.all_reduce_in_place_n(ufrob_qp.data(), ufrob_qp.size(), std::plus<>{});
      mpi->node_comm.barrier();
    }
    const double t_union = mpi->comm.all_reduce_value(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_un0).count(),
        boost::mpi3::max<>{});

    if (lowrank) {
      double tw = 0.0, fw = 0.0, aw = 0.0, rmean = 0.0;
      long rmin = NP, rmax = 0, nex = 0, qw = -1, pw = -1;
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long p = 0; p < npk; ++p) {
          if (tail_qp(iq, p) > tw) { tw = tail_qp(iq, p); qw = iq; pw = p; }
          fw = std::max(fw, frob_qp(iq, p));
          aw = std::max(aw, anti_qp(iq, p));
          rmin = std::min(rmin, rk_qp(iq, p));
          rmax = std::max(rmax, rk_qp(iq, p));
          rmean += double(rk_qp(iq, p));
          nex += exact_qp(iq, p);
        }
      const long nqp = nq_ibz * npk;
      rmean /= double(std::max(nqp, 1L));
      ctx.diag.wrank_max = rmax;
      ctx.diag.wrank_mean = rmean;
      ctx.diag.wtrunc_worst = tw;
      ctx.diag.wtrunc_frob_worst = fw;
      ctx.diag.wanti_worst = aw;
      app_log(lvl, "  - W^c slab low-rank:           ON (qp_modea_wrank = {:.2g}, backend = {}"
                   ", probe {} of Np = {}) -- rank r: min {}, mean {:.1f}, max {} "
                   "(compression r/Np = {:.4f})",
              opts.wrank, (nex == nqp ? "heev" : (nex == 0 ? "randomized" : "mixed")),
              probe_qp(0, 0), NP, rmin, rmean, rmax, rmean / double(NP));
      app_log(lvl, "  - W^c slab truncation:         worst 2-norm residual {:.3e} at (q,p) = "
                   "({},{}), worst Frobenius {:.3e}, worst |W-W^dag|/|W| {:.3e} "
                   "[flop model: stage 2 scales as r/Np = {:.4f} of the dense sandwich]",
              tw, qw, pw, fw, aw, rmean / double(NP));
      {
        std::string lad;
        for (size_t t = 0; t < detail::wrank_ladder.size(); ++t) {
          long mx = 0;
          double mn = 0.0;
          for (long iq = 0; iq < nq_ibz; ++iq)
            for (long p = 0; p < npk; ++p) {
              mx = std::max(mx, lad_qp(iq, p, long(t)));
              mn += double(lad_qp(iq, p, long(t)));
            }
          mn /= double(std::max(nqp, 1L));
          last_run().lad_max[t] = mx;
          last_run().lad_mean[t] = mn;
          lad += std::format("{}{:.0e} -> {}/{:.1f}", (t ? ", " : ""), detail::wrank_ladder[t],
                             mx, mn);
        }
        app_log(lvl, "  - W^c slab rank ladder:        (max/mean over the {} (q,p) slabs, of "
                     "Np = {})  {}", nqp, NP, lad);
      }
      // ---- UNION-SUBSPACE probe on q = 0 (small Np only; diagnostics, root-only) ----------
      // The per-slab rank r bounds the compression of the CURRENT restructure (stage 2 costs
      // r/Np of dense). The next one available -- a basis shared by all npk slabs of one q,
      // so that the Np axis is contracted once per q instead of once per (q,p) -- is bounded
      // instead by R = rank of [W^(0) | ... | W^(npk-1)], i.e. of sum_p W^(p) W^(p)^dag.
      // R << npk*r is what would make that restructure worth writing; measure it here.
      if (NP <= detail::wslab_dense_max and mpi->node_comm.root()) {
        nda::matrix<ComplexType> G(NP, NP), Wc(NP, NP);
        G() = ComplexType(0.0);
        for (long p = 0; p < npk; ++p) {
          auto Wp = sWres.local()(0, p, all, all);
          Wc = nda::conj(Wp);
          nda::blas::gemm(ComplexType(1.0), Wp, nda::transpose(Wc), ComplexType(1.0), G);
        }
        auto ev = nda::linalg::eigenvalues(G);
        double emax = 0.0;
        for (long i = 0; i < ev.size(); ++i) emax = std::max(emax, std::abs(ev(i)));
        std::string lad;
        for (size_t t = 0; t < detail::wrank_ladder.size(); ++t) {
          long rr = 0;   // singular values of the stack are sqrt(eigenvalues of G)
          for (long i = 0; i < ev.size(); ++i)
            if (std::abs(ev(i)) >= detail::wrank_ladder[t] * detail::wrank_ladder[t] * emax) ++rr;
          lad += std::format("{}{:.0e} -> {}", (t ? ", " : ""), detail::wrank_ladder[t], rr);
        }
        app_log(lvl, "  - W^c union subspace (q = 0):  rank of the {}-slab stack, of Np = {}:  "
                     "{}   [sum_p r_p = {}]", npk, NP, lad, [&] {
                  long s = 0;
                  for (long p = 0; p < npk; ++p) s += rk_qp(0, p);
                  return s; }());
      }
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long p = 0; p < npk; ++p)
          app_log(lvl + 1, "    [wslab] q = {:>3} p = {:>3}: r = {:>4} of {:>4} probed, "
                           "2-norm residual = {:.3e}, Frobenius = {:.3e}, "
                           "|W-W^dag|/|W| = {:.3e}, backend = {}",
                  iq, p, rk_qp(iq, p), probe_qp(iq, p), tail_qp(iq, p), frob_qp(iq, p),
                  anti_qp(iq, p), (exact_qp(iq, p) ? "heev" : "randomized"));
    } else {
      app_log(lvl, "  - W^c slab low-rank:           OFF (qp_modea_wrank = {:.2g}) -- the dense "
                   "Np^2 sandwich is used", opts.wrank);
    }

    if (union_on) {
      double Rmean = 0.0, ut = 0.0, uf = 0.0;
      long Rmax = 0, Rmin = NP;
      for (long iq = 0; iq < nq_ibz; ++iq) {
        Rmax = std::max(Rmax, Rq(iq));
        Rmin = std::min(Rmin, Rq(iq));
        Rmean += double(Rq(iq));
      }
      Rmean /= double(std::max(nq_ibz, 1L));
      for (long iq = 0; iq < nq_ibz; ++iq)
        for (long p = 0; p < npk; ++p) {
          ut = std::max(ut, utail_qp(iq, p));
          uf = std::max(uf, ufrob_qp(iq, p));
        }
      ctx.diag.union_R_max = Rmax;
      ctx.diag.union_R_mean = Rmean;
      ctx.diag.union_tail_worst = ut;
      ctx.diag.union_frob_worst = uf;
      ctx.diag.t_union = t_union;
      double rsum = 0.0;
      for (long p = 0; p < npk; ++p) rsum += double(rk_qp(0, p));
      app_log(lvl, "  - W^c union subspace:          ON (qp_modea_wunion = {:.2g}{}) -- rank R: "
                   "min {}, mean {:.1f}, max {} of Np = {}  [sum_p r_p = {:.0f} at q = 0, so the "
                   "Np contraction of stage 2 is done {:.1f}x less often; R/Np = {:.4f}]",
              wu, (opts.wunion > 0.0 ? "" : " = qp_modea_wrank"), Rmin, Rmean, Rmax, NP,
              rsum, (Rmax > 0 ? rsum / double(Rmax) : 0.0), Rmean / double(NP));
      app_log(lvl, "  - W^c union projection:        worst residual: 2-norm {:.3e} (scaled by "
                   "the largest slab of the q -- THE quantity wunion = {:.2g} bounds), "
                   "Frobenius {:.3e} (relative to the slab itself; stage-1b truncation is "
                   "{:.3e}) -- {} basis vectors kept, {} slab directions absorbed",
              ut, wu, uf, ctx.diag.wtrunc_worst, nappend, ndrop);
      if (Rmax >= NP)
        app_log(lvl, "  - W^c union subspace:          NOTE: R = Np, i.e. the retained slab "
                     "ranges of this q span everything at wunion = {:.2g}. The restructure is "
                     "then break-even (Np*R + npk*r*R against npk*r*Np) and costs one extra "
                     "(nq, Np, R) window -- loosen qp_modea_wunion to buy anything.", wu);
    } else if (lowrank) {
      app_log(lvl, "  - W^c union subspace:          OFF (qp_modea_wunion = {:.2g}) -- the "
                   "per-slab stage-1b sandwich is used", opts.wunion);
    }

    // ---- the dense slabs are DEAD once the factors exist (only the wrank <= 0 reference
    // path reads them in stage 2). At the production sizes this window is the single biggest
    // allocation of the whole build -- (nq_ibz, npk, Np, Np) complex, node-shared -- and
    // holding it through stage 2 was measured at +9 GB per node for nothing. Likewise the
    // per-slab factors, once every slab has been expressed in its q's union basis.
    if (lowrank) {
      mpi->node_comm.barrier();
      sWres = make_shared_array<nda::array_view<ComplexType, 4>>(*mpi, {1L, 1L, 1L, 1L});
    }
    if (union_on) {
      mpi->node_comm.barrier();
      sWv = make_shared_array<nda::array_view<ComplexType, 4>>(*mpi, {1L, 1L, 1L, 1L});
    }

    // ---------------- MO collocation columns ------------------------------------------
    // internal leg: XCi(s, k_full) = X(k_full) . C(kp_to_ibz(k_full))   [derivation 1]
    nda::array<ComplexType, 3> XCi(ns * nkpts, NP, nbnd);
    {
      auto kp_to_ibz = MF->kp_to_ibz();
      for (long sk = 0; sk < ns * nkpts; ++sk) {
        const long is = sk / nkpts, ik = sk % nkpts;
        nda::blas::gemm(thc.X(is, 0, ik),
                        sMO_skia.local()(is, kp_to_ibz(ik), all, all),
                        XCi(sk, all, all));
      }
    }

    // ---------------- stage 2: the sandwiches -----------------------------------------
    const long nJ = nqpts * nbnd;
    ctx.nJ = nJ;
    ctx.diag.nJ = nJ;
    ctx.diag.npk = npk;
    ctx.epsJ = nda::array<double, 1>(ns * nk_ibz * nJ);
    ctx.fJ = nda::array<double, 1>(ns * nk_ibz * nJ);
    ctx.epsJ() = 0.0;
    ctx.fJ() = 0.0;
    ctx.owner = nda::array<long, 1>(ns * nk_ibz);
    ctx.owner() = -1;

    const long nP_flat = nJ * npk;
    if (need_diag) {
      ctx.Mdiag = nda::array<ComplexType, 4>(ns, nk_ibz, nbnd, nP_flat);
      ctx.Mdiag() = ComplexType(0.0);
      ctx.have_diag = true;
    }

    auto kp_to_ibz = MF->kp_to_ibz();
    auto kp_trev = MF->kp_trev();
    auto kp_trev_pair = MF->kp_trev_pair();
    auto qp_trev = MF->qp_trev();
    auto qminus = MF->qminus();
    const double pref = 1.0 / double(nkpts);

    // ---------------- the SYMMETRY CENSUS (permanent, level 2) -------------------------
    // Two things the kp444 post-mortem needed and no log carried:
    //  (1) q_isym: which symmetry class handles each full-BZ transfer. The (isym, q-in-star)
    //      pairs partition the full mesh (the coverage tripwire below re-checks it), so the
    //      flat internal label J = q'*nbnd + n inherits a class and the tau oracle can split
    //      its route-B side per class -- the per-isym anchor breakdown.
    //  (2) whether the D-matrix external rotation is EXERCISED AT ALL, and its unitarity
    //      defect. generate_dmatrix stores the IDENTITY for the symmetry that defines the
    //      image k-point's orbitals ("by convention, d(kp_symm(k), kp_to_ibz(k)) = delta",
    //      symmetry.hpp:906-921), so a symmetry-reduced mesh can run the whole isym loop with
    //      D = 1 everywhere: the star structure is then tested and the ROTATION is not. The
    //      defect max|D^dag D - 1| is the accuracy floor of the symmetry path in the sense of
    //      vertex_sym.hpp:52-56 (nbnd-truncated degenerate multiplets, row-normalized rows).
    //      It cancels between this map and the reference -- both apply the SAME D to the same
    //      object (see DERIVATION 1) -- so it is reported, never gated.
    ctx.nsym = nsym;
    ctx.q_isym = nda::array<long, 1>(nqpts);
    ctx.q_isym() = 0;
    {
      nda::array<long, 1> seen(nqpts);
      seen() = 0;
      for (long isym = 0; isym < nsym; ++isym)
        for (long iq = 0; iq < MF->nq_per_s(isym); ++iq) {
          const long qp = MF->Qs(isym, iq);
          utils::check(qp >= 0 and qp < nqpts, "qp_modea: Qs({},{}) = {} out of range.",
                       isym, iq, qp);
          ctx.q_isym(qp) = isym;
          ++seen(qp);
        }
      for (long qp = 0; qp < nqpts; ++qp)
        utils::check(seen(qp) == 1, "qp_modea: transfer q' = {} is handled by {} (isym, q) "
                     "pairs of the star loop, not 1.", qp, seen(qp));
    }
    if (nsym > 1) {
      nda::array<long, 1> ncls(nsym), nnid(nsym);
      nda::array<double, 1> did(nsym), dun(nsym);
      ncls() = 0; nnid() = 0; did() = 0.0; dun() = 0.0;
      nda::array<ComplexType, 2> Dd(nbnd, nbnd), DhD(nbnd, nbnd), Id(nbnd, nbnd);
      Id() = ComplexType(0.0);
      for (long i = 0; i < nbnd; ++i) Id(i, i) = ComplexType(1.0);
      for (long isym = 0; isym < nsym; ++isym) {
        ncls(isym) = MF->nq_per_s(isym);
        if (isym == 0) continue;
        for (long ik = 0; ik < nk_ibz; ++ik) {
          auto [cjg, D] = MF->symmetry_rotation(isym, ik);
          Dd() = ComplexType(0.0);
          math::sparse::csrmm(ComplexType(1.0), *D, Id, ComplexType(0.0), Dd);
          double di = 0.0;
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              di = std::max(di, std::abs(Dd(i, j) - (i == j ? ComplexType(1.0)
                                                            : ComplexType(0.0))));
          nda::blas::gemm(nda::dagger(Dd), Dd, DhD);
          double du = 0.0;
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              du = std::max(du, std::abs(DhD(i, j) - (i == j ? ComplexType(1.0)
                                                             : ComplexType(0.0))));
          if (di > 1e-12) ++nnid(isym);
          did(isym) = std::max(did(isym), di);
          dun(isym) = std::max(dun(isym), du);
        }
        ctx.diag.n_D_nonid += nnid(isym);
        ctx.diag.d_ident_worst = std::max(ctx.diag.d_ident_worst, did(isym));
        ctx.diag.d_unit_worst = std::max(ctx.diag.d_unit_worst, dun(isym));
      }
      app_log(lvl, "  - symmetry census:             {} q-symmetry classes over {} full-BZ "
                   "transfers, {} IBZ k; D-rotation exercised on {} of {} (isym > 0, k) pairs, "
                   "worst max|D - 1| = {:.3e}, worst max|D^dag D - 1| = {:.3e} (REPORTED -- it "
                   "cancels against the reference, which applies the same D)",
              nsym, nqpts, nk_ibz, ctx.diag.n_D_nonid, (nsym - 1) * nk_ibz,
              ctx.diag.d_ident_worst, ctx.diag.d_unit_worst);
      for (long isym = 0; isym < nsym; ++isym)
        app_log(lvl + 1, "  - symmetry class {:>3}:          {} transfers, D non-identity at "
                         "{} of {} k, max|D - 1| = {:.3e}, max|D^dag D - 1| = {:.3e}",
                isym, ncls(isym), nnid(isym), (isym == 0 ? 0 : nk_ibz), did(isym), dun(isym));
    }

    nda::array<ComplexType, 2> XCe(NP, nbnd), DC(nbnd, nbnd);
    nda::array<ComplexType, 2> B(NP, nbnd), Bc(NP, nbnd), T(NP, nbnd), Msand(nbnd, nbnd);
    // low-rank path scratch: the r-th column of the slab basis contracts BOTH legs at once,
    // Z(P,a) = conj(XCe(P,a)) V(P,r)  ->  Gr = Z^T Uc = g_r(a, n)  (one gemm per (q,p,r)).
    const long rc = std::max(rcap, 1L);
    const bool perslab = lowrank and not union_on;
    nda::array<ComplexType, 2> Uc(lowrank ? NP : 1, lowrank ? nbnd : 1);
    nda::array<ComplexType, 2> Z(perslab ? NP : 1, perslab ? nbnd : 1);
    nda::array<ComplexType, 2> Vw(perslab ? NP : 1, perslab ? rc : 1);
    nda::array<ComplexType, 2> Gr(perslab ? nbnd : 1, perslab ? nbnd : 1);
    nda::array<ComplexType, 3> Gall(perslab ? rc : 1, perslab ? nbnd : 1, perslab ? nbnd : 1);
    nda::array<ComplexType, 2> Gs(lowrank ? rc : 1, lowrank ? nbnd : 1);
    nda::array<ComplexType, 2> Gc(lowrank ? rc : 1, lowrank ? nbnd : 1);
    // union path scratch: H is the ONE contraction of the Np axis, done per (k,q,n).
    nda::array<ComplexType, 2> Hn(union_on ? std::max(Rcap, 1L) : 1, union_on ? nbnd : 1);
    nda::array<ComplexType, 2> Gt(union_on ? rc : 1, union_on ? nbnd : 1);

    // ---- WHO DOES THE WORK (and who stores it) ----------------------------------------
    // The ctx layout is unchanged: block sk lives on rank sk % size, and every consumer of
    // ctx.blocks still finds it there. What changes is that the FLOPS of a block are spread
    // over the whole congruence class {r : r % nblk == sk}: with ns*nk_ibz = 8 blocks on 32
    // ranks the old loop left 24 ranks idle for the entire sandwich stage. The split axis is
    // the (isym, q-in-star) PAIR, which the star loop visits exactly once per full-mesh q, so
    // every member gets a disjoint slice of both the M accumulation and the (eps_J, f_J)
    // writes -- nothing is computed twice and the plus-reduce of eps_J/f_J stays exact.
    // [multi-node caveat: the class is congruence-based, so on more than one node a group can
    //  straddle nodes and the per-pair reduction crosses the network. Its volume is one
    //  block's worth per outer iteration -- seconds at the production sizes.]
    const long nblk = ns * nk_ibz;
    const int csize = mpi->comm.size(), crank = mpi->comm.rank();
    const bool helpers_on = (csize > nblk);
    auto wcomm = mpi->comm.split(helpers_on ? int(crank % nblk) : crank, crank);
    const int gsize = helpers_on ? wcomm.size() : 1;
    const int grank = helpers_on ? wcomm.rank() : 0;
    // the staging buffer of ONE (isym, q) pair, (n, a, b, p) so that a pair is contiguous and
    // the group reduction is a single call. The owner adds it into blk.M pair by pair.
    bool any_work = false;
    for (long sk = 0; sk < nblk; ++sk)
      if (sk % csize == crank or (helpers_on and crank % nblk == sk)) { any_work = true; break; }
    nda::array<ComplexType, 4> Mq(any_work ? nbnd : 1, any_work ? nbnd : 1,
                                  any_work ? nbnd : 1, any_work ? npk : 1);
    if (helpers_on)
      app_log(lvl, "  - stage 2 work sharing:        {} (s,k) blocks over {} ranks -> {} ranks "
                   "per block, split over the {} (isym, q) pairs; block storage unchanged "
                   "(owner = sk % size)", nblk, csize, gsize, nqpts);

    const auto t_s2 = std::chrono::steady_clock::now();

    for (long sk = 0; sk < nblk; ++sk) {
      ctx.owner(sk) = sk % csize;
      const bool own = (ctx.owner(sk) == crank);
      const bool help = helpers_on and (crank % nblk == sk);
      if (not own and not help) continue;
      const long is = sk / nk_ibz, ik = sk % nk_ibz;
      sk_block blk;
      blk.is = is;
      blk.ik = ik;
      if (own) {
        blk.M = nda::array<ComplexType, 3>(nbnd, nbnd, nP_flat);
        blk.M() = ComplexType(0.0);
      }

      long pair = 0, npair_mine = 0;
      for (long isym = 0; isym < nsym; ++isym) {
        const long nqs = MF->nq_per_s(isym);
        bool mine_any = false;
        for (long iq = 0; iq < nqs; ++iq)
          if (not helpers_on or ((pair + iq) % gsize == grank)) { mine_any = true; break; }
        long ks = -1;
        if (mine_any) {
          ks = MF->ks_to_k(isym, ik);
          if (isym == 0) {
            XCe = XCi(is * nkpts + ks, all, all);
          } else {
            auto [cjg, D] = MF->symmetry_rotation(isym, ik);
            utils::check(not cjg, "qp_modea: symmetry_rotation(isym = {}, k = {}) reports the "
                                  "conjugation flag, which the GW assembly this map reproduces "
                                  "does not handle either (thc_gw.icc:311).", isym, ik);
            math::sparse::csrmm(ComplexType(1.0), *D,
                                nda::make_regular(sMO_skia.local()(is, ik, all, all)),
                                ComplexType(0.0), DC);
            nda::blas::gemm(thc.X(is, 0, ks), DC, XCe);
          }
        }

        for (long iq = 0; iq < nqs; ++iq, ++pair) {
          const bool mine = (not helpers_on) or (pair % gsize == grank);
          const bool reduce = (gsize > 1) and ((pair % gsize) != 0);
          if (mine or reduce) Mq() = ComplexType(0.0);

          if (mine) {
          ++npair_mine;
          const long qp = MF->Qs(isym, iq);
          const long qs = MF->qp_to_ibz(qp);
          const bool wconj = qp_trev(qp);
          const long kk = wconj ? MF->qk_to_k2(qminus(qs), ks) : MF->qk_to_k2(qs, ks);
          const bool gconj = kp_trev(kk);
          const long kg = gconj ? kp_trev_pair(kk) : kk;
          const long kg_ibz = kp_to_ibz(kg);
          auto U = XCi(is * nkpts + kg, all, all);

          for (long n = 0; n < nbnd; ++n) {
            const long J = qp * nbnd + n;
            const double e = sE_ska.local()(is, kg_ibz, n).real();
            ctx.epsJ((is * nk_ibz + ik) * nJ + J) = e;
            ctx.fJ((is * nk_ibz + ik) * nJ + J) = sigma_route_b::stable_nF(ctx.beta, e - mu);
          }

          if (union_on) {
            // W^(p) = Q_q A^(p) S A^(p)dag Q_q^dag, so with H(R,a) = sum_P q_R(P) B(P,a)
            //     g^(p)(a,r) = sum_R A^(p)(R,r) H(R,a),   M = sum_r s_r g conj(g),
            // and the Np axis is contracted ONCE per (k,q,n) -- for the WHOLE slab stack.
            // The trev-q rule rides on the (Np,nbnd) side here, through
            //     B^T conj(W) conj(B) = conj( conj(B)^T W B ),
            // so Q and the coefficient blocks are used exactly as stored (no (Np,r) copy per
            // (q,p), which is what the per-slab path below has to pay).
            if (gconj != wconj) Uc = nda::conj(U);
            else                Uc = U;
            if (wconj) Bc = XCe;
            else       Bc = nda::conj(XCe);
            for (long n = 0; n < nbnd; ++n) {
              for (long P = 0; P < NP; ++P) {
                const ComplexType u = Uc(P, n);
                for (long a = 0; a < nbnd; ++a) B(P, a) = Bc(P, a) * u;
              }
              nda::blas::gemm(sQ.local()(qs, all, all), B, Hn);   // (Rcap, NP) x (NP, nbnd)
              for (long p = 0; p < npk; ++p) {
                const long r = rk_qp(qs, p);
                if (r == 0) continue;
                auto Gtr = Gt(nda::range(0, r), all);
                nda::blas::gemm(sWa.local()(qs, p, nda::range(0, r), all), Hn, Gtr);
                for (long rr = 0; rr < r; ++rr) {
                  const double sr = sWs.local()(qs, p, rr);
                  for (long a = 0; a < nbnd; ++a) {
                    Gs(rr, a) = sr * Gt(rr, a);
                    Gc(rr, a) = std::conj(Gt(rr, a));
                  }
                }
                nda::blas::gemm(nda::transpose(Gs(nda::range(0, r), all)),
                                Gc(nda::range(0, r), all), Msand);
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b)
                    Mq(n, a, b, p) += pref * (wconj ? std::conj(Msand(a, b)) : Msand(a, b));
              }
            }
          } else if (lowrank) {
            // M^(n,p)_ab = sum_r s_r g_r(a) conj(g_r(b)),  g_r(a) = sum_P B(P,a) V(P,r),
            // which is the SAME double sum as the dense sandwich with W^(p) = V S V^dag
            // substituted -- see the flop model in the file header.
            if (gconj) Uc = nda::conj(U);
            else       Uc = U;
            for (long p = 0; p < npk; ++p) {
              const long r = rk_qp(qs, p);
              if (r == 0) continue;
              // the trev-q rule is conj(W_PQ) elementwise, i.e. V -> conj(V) with S real
              // (an (Np,r) copy, not the (Np,Np) one the dense path used to make per (n,p))
              if (wconj) Vw(all, nda::range(0, r)) = nda::conj(sWv.local()(qs, p, all, nda::range(0, r)));
              else       Vw(all, nda::range(0, r)) = sWv.local()(qs, p, all, nda::range(0, r));
              for (long rr = 0; rr < r; ++rr) {
                for (long P = 0; P < NP; ++P) {
                  const ComplexType v = Vw(P, rr);
                  for (long a = 0; a < nbnd; ++a) Z(P, a) = std::conj(XCe(P, a)) * v;
                }
                nda::blas::gemm(nda::transpose(Z), Uc, Gr);        // Gr(a,n) = g_r(a) at n
                Gall(rr, all, all) = Gr;
              }
              // the r-major buffers keep the USED block (r, nbnd) contiguous, so the two gemm
              // operands are honest C-layout matrices at the exact rank of this slab -- a
              // (nbnd, r) column sub-block of an (nbnd, rc) buffer would be strided, and BLAS
              // would be handed the wrong leading dimension.
              auto Gsr = Gs(nda::range(0, r), all);
              auto Gcr = Gc(nda::range(0, r), all);
              for (long n = 0; n < nbnd; ++n) {
                for (long rr = 0; rr < r; ++rr) {
                  const double sr = sWs.local()(qs, p, rr);
                  for (long a = 0; a < nbnd; ++a) {
                    Gs(rr, a) = sr * Gall(rr, a, n);
                    Gc(rr, a) = std::conj(Gall(rr, a, n));
                  }
                }
                nda::blas::gemm(nda::transpose(Gsr), Gcr, Msand);
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b) Mq(n, a, b, p) += pref * Msand(a, b);
              }
            }
          } else {
            // reference path: the dense Np^2 sandwich. p is the OUTER loop so that the slab
            // is used through a view -- the (Np,Np) copy this loop used to make per (n,p) was
            // pure memory traffic. The trev-q conj is moved off the big matrix through
            //     B^T conj(W) conj(B) = conj( conj(B)^T W B ),
            // an (nbnd,nbnd) conjugation instead of an (Np,Np) one.
            for (long p = 0; p < npk; ++p) {
              auto Wp = sWres.local()(qs, p, all, all);
              for (long n = 0; n < nbnd; ++n) {
                for (long P = 0; P < NP; ++P) {
                  const ComplexType u = gconj ? std::conj(U(P, n)) : U(P, n);
                  for (long a = 0; a < nbnd; ++a) {
                    B(P, a) = std::conj(XCe(P, a)) * u;
                    Bc(P, a) = std::conj(B(P, a));
                  }
                }
                if (wconj) {
                  nda::blas::gemm(Wp, B, T);                        // (NP, nbnd)
                  nda::blas::gemm(nda::transpose(Bc), T, Msand);    // (nbnd, nbnd)
                } else {
                  nda::blas::gemm(Wp, Bc, T);
                  nda::blas::gemm(nda::transpose(B), T, Msand);
                }
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b)
                    Mq(n, a, b, p) += pref * (wconj ? std::conj(Msand(a, b)) : Msand(a, b));
              }
            }
          }
          }   // mine

          // the block accumulation is linear in the pair, so the group's partial results just
          // add: exactly one member computed this pair, the rest hold zeros.
          if (reduce) wcomm.reduce_in_place_n(Mq.data(), Mq.size(), std::plus<>{}, 0);
          if (own) {
            const long qp = MF->Qs(isym, iq);
            for (long n = 0; n < nbnd; ++n)
              for (long p = 0; p < npk; ++p) {
                const long Pf = (qp * nbnd + n) * npk + p;
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b) blk.M(a, b, Pf) += Mq(n, a, b, p);
              }
          }
        }
      }
      // COVERAGE TRIPWIRE for the split above. Every (isym, q) pair of the star must have been
      // computed by exactly ONE member of the group, and the pairs must cover the full q mesh
      // -- the two assumptions the whole distribution rests on. Both are cheap to verify and
      // neither is reachable by the single-rank gates (a helper group needs more ranks than
      // there are (s,k) blocks), so they are checked at run time instead.
      {
        const long tot = (gsize > 1) ? wcomm.all_reduce_value(npair_mine, std::plus<>{})
                                     : npair_mine;
        utils::check(tot == nqpts,
                     "qp_modea: stage-2 work sharing lost or duplicated work on block "
                     "(s,k) = ({},{}): the group computed {} (isym,q) pairs, the star of the "
                     "full mesh has {}. [group size {}, rank {} of {}]",
                     is, ik, tot, nqpts, gsize, crank, csize);
      }
      if (own) {
        if (need_diag)
          for (long a = 0; a < nbnd; ++a)
            for (long Pf = 0; Pf < nP_flat; ++Pf) ctx.Mdiag(is, ik, a, Pf) = blk.M(a, a, Pf);
        ctx.blocks.push_back(std::move(blk));
      }
    }

    mpi->comm.all_reduce_in_place_n(ctx.epsJ.data(), ctx.epsJ.size(), std::plus<>{});
    mpi->comm.all_reduce_in_place_n(ctx.fJ.data(), ctx.fJ.size(), std::plus<>{});
    if (need_diag)
      mpi->comm.all_reduce_in_place_n(ctx.Mdiag.data(), ctx.Mdiag.size(), std::plus<>{});

    ctx.active = true;
    ctx.have_cd = true;
    const double mb = 1.0 / (1024.0 * 1024.0);
    const double sz = double(sizeof(ComplexType)) * mb;
    const double m_res = double(nq_ibz) * double(npk) * double(NP) * double(NP) * sz;
    const double m_v = lowrank ? double(nq_ibz) * double(npk) * double(NP) * double(rc) * sz : 0.0;
    const double m_qb = union_on ? double(nq_ibz) * double(std::max(Rcap, 1L)) * double(NP) * sz
                                 : 0.0;
    const double m_a = union_on ? double(nq_ibz) * double(npk) * double(rc)
                                      * double(std::max(Rcap, 1L)) * sz : 0.0;
    const double m_blk = double(ctx.blocks.size()) * double(nbnd) * double(nbnd)
                         * double(nP_flat) * sz;
    const double m_mq = any_work ? double(nbnd) * double(nbnd) * double(nbnd) * double(npk) * sz
                                 : 0.0;
    const double m_dia = need_diag ? double(ns) * nk_ibz * nbnd * nP_flat * sz : 0.0;
    // the two footprints the build actually passes through: stage 1c (the dense slabs, the
    // factors and the union data all live) and stage 2 (whatever was freed above is gone,
    // the blocks and the pair buffer are up).
    const double m_s1 = m_res + m_v + m_qb + m_a + m_dia;
    const double m_s2 = (lowrank ? 0.0 : m_res) + (union_on ? 0.0 : m_v) + m_qb + m_a
                        + m_blk + m_mq + m_dia;
    double mem = std::max(m_s1, m_s2);
    ctx.diag.mem_mb = mpi->comm.all_reduce_value(mem, boost::mpi3::max<>{});
    ctx.diag.wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    ctx.diag.wall_s = mpi->comm.all_reduce_value(ctx.diag.wall_s, boost::mpi3::max<>{});
    ctx.diag.t_fit = mpi->comm.all_reduce_value(t_fit, boost::mpi3::max<>{});
    ctx.diag.t_fac = mpi->comm.all_reduce_value(t_fac, boost::mpi3::max<>{});
    ctx.diag.t_sand = mpi->comm.all_reduce_value(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s2).count(),
        boost::mpi3::max<>{});

    app_log(lvl, "  - internal pole structure:     nJ = {} (nq_full {} x nbnd {}) x npk = {} "
                 "-> {} poles per (a,b)", nJ, nqpts, nbnd, npk, nP_flat);
    app_log(lvl, "  - stored W Hermiticity:        max|W - W^dag|/max|W| = {:.3e} "
                 "(max|Im Ttw_bb| = {:.3e}); the trev-q rule implemented is conj(W_PQ), "
                 "matching thc_gw.icc:381-393", ctx.diag.w_herm_rel, ctx.diag.ttw_imag);
    app_log(lvl, "  - W^c fit quality:             bosonic-mesh reconstruction rel err = "
                 "{:.3e} (worst q = {})   [own-grid residual {:.3e}, residue ratio {:.3g} -- "
                 "reported, NOT the quality number]",
            ctx.diag.rec_rel_worst, ctx.diag.rec_rel_q, ctx.diag.fit_err_worst,
            ctx.diag.res_ratio_worst);
    app_log(lvl, "  - W^c fit, ABSOLUTE scale:     worst absolute reconstruction error = "
                 "{:.3e} at IBZ q = {} (max|W_q| there = {:.3e}, i.e. {:.3e} relative; the "
                 "relative class above is worst at q = {}). The two orderings differ once the "
                 "mesh resolves the 1/q^2 growth of W near Gamma, and Sigma sums the mesh "
                 "against band factors of a common scale, so it is the ABSOLUTE column that "
                 "reaches the anchor.",
            ctx.diag.rec_abs_worst, ctx.diag.rec_abs_q, ctx.diag.rec_abs_wmax,
            (ctx.diag.rec_abs_wmax > 0.0 ? ctx.diag.rec_abs_worst / ctx.diag.rec_abs_wmax
                                         : 0.0), ctx.diag.rec_rel_q);
    app_log(lvl, "  - W^c error BUDGET:            sum_q |dW_q| / sum_q max|W_q| = {:.3e} -- "
                 "the tau-anchor scale the W REPRESENTATION alone predicts. A measured anchor "
                 "at this scale leaves nothing for a routing error to explain; an anchor far "
                 "ABOVE it is the only reading that points back at the contraction.",
            ctx.diag.rec_budget);
    app_log(lvl, "  - context build:               {:.2f} s wall, {:.1f} MB extra per rank "
                 "(peak)", ctx.diag.wall_s, ctx.diag.mem_mb);
    app_log(lvl, "  - context build memory:        stage 1c {:.0f} MB (dense slabs {:.0f} + "
                 "slab factors {:.0f} + union basis {:.0f} + coefficients {:.0f}), stage 2 "
                 "{:.0f} MB (blocks {:.0f} + pair buffer {:.0f}); freed after stage 1c: {}",
            m_s1, m_res, m_v, m_qb, m_a, m_s2, m_blk, m_mq,
            (lowrank ? (union_on ? "dense slabs + slab factors" : "dense slabs") : "nothing"));
    app_log(lvl, "  - context build breakdown:     stage 1 (gather + fit) {:.2f} s, stage 1b "
                 "(slab factorization) {:.2f} s, stage 1c (union subspace) {:.2f} s, stage 2 "
                 "(sandwiches) {:.2f} s  [worst rank]",
            ctx.diag.t_fit, ctx.diag.t_fac, t_union, ctx.diag.t_sand);

    auto &LR = last_run();
    LR.gap_edge = ctx.diag.gap_edge;
    LR.rec_rel = ctx.diag.rec_rel_worst;
    LR.wall_s = ctx.diag.wall_s;
    LR.mem_mb = ctx.diag.mem_mb;
    LR.n_support = ctx.diag.n_support;
    LR.np_total = ctx.diag.np_total;
    LR.nJ = nJ;
    LR.npk = npk;
    LR.wfit = opts.wfit;
    LR.res_ratio = ctx.diag.res_ratio_worst;
    LR.wrtol = mpf_opt.has_value() ? mpf_opt->rel_tol : opts.wrtol;
    // RW-2 spectral census (all zero on the tau/nu routes)
    LR.sp_eta = ctx.diag.sp_eta;
    LR.sp_NO = ctx.diag.sp_NO;
    LR.sp_nbin = ctx.diag.sp_nbin;
    LR.sp_nhead = ctx.diag.sp_nhead;
    LR.sp_width = ctx.diag.sp_width;
    LR.sp_sym = ctx.diag.sp_sym;
    LR.sp_psdneg = ctx.diag.sp_psdneg;
    LR.sp_headrec = ctx.diag.sp_headrec;
    LR.sp_wall = ctx.diag.sp_wall;
    LR.wrank = opts.wrank;
    LR.wrank_max = ctx.diag.wrank_max;
    LR.wrank_mean = ctx.diag.wrank_mean;
    LR.wtrunc = ctx.diag.wtrunc_worst;
    LR.t_fit = ctx.diag.t_fit;
    LR.t_fac = ctx.diag.t_fac;
    LR.t_sand = ctx.diag.t_sand;
    LR.Np = NP;
    LR.wunion = union_on ? wu : -1.0;
    LR.union_R_max = ctx.diag.union_R_max;
    LR.union_R_mean = ctx.diag.union_R_mean;
    LR.union_tail = ctx.diag.union_tail_worst;
    LR.union_frob = ctx.diag.union_frob_worst;
    LR.t_union = ctx.diag.t_union;
  }

} // qp_modea
} // methods

#endif // COQUI_WC_BAND_ELEMENTS_HPP
