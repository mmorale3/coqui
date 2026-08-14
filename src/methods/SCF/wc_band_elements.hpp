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
#include "methods/mb_state/mb_state.hpp"

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

    auto mpf = (opts.wfit == "nu")
                   ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, opts.wrtol)
                   : imag_axes_ft::masked_pole_fit::from_tau(pf, gap_edge, opts.wrtol);
    const long npk = mpf.nkeep;
    ctx.npk = npk;
    ctx.om = nda::array<double, 1>(npk);
    ctx.nB = nda::array<double, 1>(npk);
    for (long p = 0; p < npk; ++p) {
      ctx.om(p) = mpf.om(p);
      ctx.nB(p) = sigma_route_b::stable_nB(ctx.beta, ctx.om(p));
    }
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
    ctx.diag.n_support = npk;
    ctx.diag.np_total = pf.np;

    // reality of the tau <-> nu kernels: if Ttw_bb is real then Hermiticity of the STORED
    // W(tau) in (P,Q) transfers exactly to W(i nu), which is what the trev rule needs.
    {
      auto T = FT.Ttw_bb();
      double im = 0.0;
      for (auto const &v : T) im = std::max(im, std::abs(std::imag(ComplexType(v))));
      ctx.diag.ttw_imag = im;
    }

    app_log(lvl, "  - W^c support constraint:      |eps_p| >= {:.6g} a.u. ({:.4g} eV) -- "
                 "{} of {} auxiliary nodes retained, {} singular directions",
            gap_edge, gap_edge * 27.211386245988, npk, pf.np, mpf.n_kept);
    app_log(lvl, "  - W^c pole-fit route:          {} ({} rows), SVD cut rel_tol = {:.2g}",
            opts.wfit, mpf.nrow, mpf.rel_tol);

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
      // [assumed -- gate: NONE in QM3; both QM3-b fixtures and the QM3-c judge run
      //  div_treatment = ignore_g0, so this branch is UNEXERCISED by any gate.]
      auto chi = thc.basis_head()(0, all);
      const double mad = MF->madelung();
      for (long j = 0; j < nc; ++j) {
        const long P = (c0 + j) / NP, Q = (c0 + j) % NP;
        Hcol(j) = double(nkpts) * mad * std::conj(chi(P)) * chi(Q);
      }
      app_log(lvl, "  - Gamma head:                  ON (div_treatment = {}, madelung = "
                   "{:.6g}) -- UNGATED code path, see wc_band_elements.hpp", div_treatment, mad);
    } else {
      app_log(lvl, "  - Gamma head:                  OFF (div_treatment = {})", div_treatment);
    }

    auto sWres = make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, {nq_ibz, npk, NP, NP});
    auto sWt = make_shared_array<nda::array_view<ComplexType, 3>>(*mpi, {nt_half, NP, NP});
    sWres.set_zero();
    sWres.win().fence();

    nda::array<ComplexType, 2> Wt(nt_half, nc), Whalf(nw_half, nc), Ww(nwb, nc);
    nda::array<ComplexType, 2> Wf((opts.wfit == "nu") ? 0 : ntf, (opts.wfit == "nu") ? 0 : nc);
    double herm_num = 0.0, herm_den = 0.0, rec_worst = 0.0, fit_worst = 0.0, ratio_worst = 0.0;

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
        if (den > 0.0) rec_worst = std::max(rec_worst, num / den);
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
    ctx.diag.w_herm_rel = (herm_den > 0.0) ? herm_num / herm_den : 0.0;
    ctx.diag.w_herm_rel = mpi->comm.all_reduce_value(ctx.diag.w_herm_rel, boost::mpi3::max<>{});
    ctx.diag.rec_rel_worst = mpi->comm.all_reduce_value(rec_worst, boost::mpi3::max<>{});
    ctx.diag.fit_err_worst = mpi->comm.all_reduce_value(fit_worst, boost::mpi3::max<>{});
    ctx.diag.res_ratio_worst = mpi->comm.all_reduce_value(ratio_worst, boost::mpi3::max<>{});
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
                 "{:.3e} (worst q)   [own-grid residual {:.3e}, residue ratio {:.3g} -- "
                 "reported, NOT the quality number]",
            ctx.diag.rec_rel_worst, ctx.diag.fit_err_worst, ctx.diag.res_ratio_worst);
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
    LR.wrtol = mpf.rel_tol;
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
