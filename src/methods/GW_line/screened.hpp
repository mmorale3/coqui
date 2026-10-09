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

#ifndef COQUI_METHODS_GW_LINE_SCREENED_HPP
#define COQUI_METHODS_GW_LINE_SCREENED_HPP

/**
 * Screened interaction on the line (notes section 5.3, Eq. dyson_w; section 3.3, Eqs. brep, bfit; plan 6.3(c);
 * python LineGW.dyson_w / screened_interaction):
 *
 *   W(q, zeta_i)  = ([1 - Z(q) Pi(q, zeta_i)]^{-1} - 1) Z(q)                          at the bosonic line nodes zeta_i,
 *   W_PQ(q, zeta) = sum_j [ w_j(q)_PQ / (zeta - nu_j) - w_j(-q)_QP / (zeta + nu_j) ]  (paired real-pole fit, all pairs).
 *
 * q <-> -q pairing (fix of 2026-10-04, notes section 3.3). The bosonic propagator of rho_q obeys W(q, -zeta) = W(-q, zeta)^T
 * (the same for Pi; B(q) = A(-q)^T for the negative / positive frequency residues), NOT W(q, -zeta) = W(q, zeta)^T: the
 * hole residues of q are the transposed particle residues of -q = qminus(q) (mf::MF::qminus, coulomb_blocks_t::qminus).
 * The per-q form holds only for self-inverse q (q = -q mod G: every q of the 2x2x2 / 2x1x1 meshes); on Si 4x4x4 it put
 * Sigma_c 50-62% off. The fit of q therefore needs W(-q)^T at the nodes: a q group must contain -q with every q (q_groups_t
 * builds pair-closed groups), and the mirror data come from the partner's W (see below).
 *
 * Distribution (plan 6.2 / 6.3(c)). Every Np x Np object lives as the (P_rng, Q_rng) block of the ONE 2D aux grid
 * (aux_grid_t). The Dyson step needs whole matrices: Pi {g, N_zeta, Np, Np} is wrapped as a darray with grid
 * {1, 1, np_P, np_Q} (the block layout; its chunks are exactly aux_grid_t's) and redistributed to the "whole-matrix"
 * layout {np_q, np_z, 1, 1} (dyson_layout_t: contiguous (q, zeta) slabs per rank, full Np x Np). Per local (q, zeta) the
 * five steps of the imaginary-axis code are done on full matrices; besides W the transposed copy W^T is formed locally
 * (a plain transpose), and BOTH are redistributed back to the block layout. Block (I,J) of the redistributed W^T is
 * (W_JI)^T, i.e. exactly the mirror block the coupled (PQ, QP) fit needs: there is no inter-rank transpose logic anywhere.
 * S7d: the stage runs in sub-steps (w_plan_t, proc_grid.hpp): one q per q pool and a zeta sub-slab at a time, Dyson IN
 * PLACE in the whole-matrix buffer (W^T), the in-place transpose giving W, the fit per q sub-step. Besides the resident
 * Pi group (reused for W) only one q row per pool of W^T and the sub-slab buffers are held (before: two extra slabs).
 *
 * The fit (Eq. bfit) for all pairs at once, from the truncated SVD of the stacked 2 N_zeta x 2 r kernel
 * [[K^-, -K^+], [-K^+, K^-]] = U S V^dagger (same rcond = DBL_EPSILON max(2 N_zeta, 2 r) as bosonic_basis_t::fit /
 * numpy lstsq; bosonic_fit_t, built once on the host), unknowns [w(q); w(-q)^T], data [W(q); W(-q)^T]:
 *   w(q)_IJ = VS (U1H W(q)_IJ + U2H (W(-q)^T)_IJ)   (three gemms per q: [k x N_zeta] . [N_zeta x block] twice, [r x k] . [k x block]).
 * Mirror data. Self-inverse q: (W(q)^T)_IJ is the block of the Dyson's own W^T (step 3 below), fitted in the Dyson sub-step
 * as before the fix (bitwise the old path). q != -q: after all Dyson sub-steps of the group (W of every q of the group in the
 * Pi buffer), a PAIR PASS per sub-step redistributes the block-layout W(-q) rows to the whole-matrix buffer, transposes
 * them in place and redistributes back: the block of W(-q)^T, then fits q. Same buffers (WT, Tb, D) as the Dyson pass, one
 * extra pair of redistributions per non-self-inverse q.
 * The explicit pinv blocks A11 = VS U1H, A12 = VS U2H are mathematically the same but numerically unstable (cond ~1e13).
 * The residues themselves are determined only up to the near-threshold singular directions (gesvd here vs gelss in
 * bosonic_basis_t::fit differ by ~1e-5 in w); the pole functions they define agree to ~1e-13 (test [V2](b)).
 *
 * Only w is stored (not w^T). The hole-sector interaction W^<(q,t)_PQ = -sum_j w_j(-q)_QP e^{+i nu_j t} needs the mirror
 * block, so it is consumed (S5) in TRANSPOSED orientation together with the transposed propagator
 * (gtilde_form_t::transposed), exactly as the polarization already does:
 *   Sigma~^<(k,t)^T = -(1/N_k) sum_q G~^<(k-q,t)^T o W^<(q,t)^T,    W^<(q,t)^T = -sum_j w_j(-q) e^{+i nu_j t},
 * and the orbital contraction of a transposed aux matrix is the transpose of an nb x nb result: with B = Sigma~^T,
 *   Sigma_ab = sum_PQ conj(X_Pa) Sigma~_PQ X_Qb = [X^T B conj(X)]_ba,
 * i.e. S5 contracts B with the plain / conjugated X slices (both mirrored already) and transposes the small nb x nb.
 * The particle sector is used in plain orientation: W^>(q,t) = sum_j w_j(q) e^{-i nu_j t}. A non-transposed hole
 * W^<(q,t) is therefore NOT provided (w_time / eval_poles abort on it).
 *
 * Collectives: coulomb_blocks_t's constructor (lockstep thc.Z over all q), screened_interaction (redistributes).
 * Device rules (plan 6.4): MEM arrays touched only by copies, nda::blas::gemm, nda::lapack::{getrf, getrs},
 * nda::tensor::{add, set}; the fit pseudo-inverse, the exponentials and the kernels are built on the host and copied.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <complex>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/tensor.hpp"
#include "itertools/itertools.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/device_pool.h"
#include "utilities/mpi_context.h"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/Timer.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/device_blas.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::sector_t;

// dyson_layout_t (the whole-matrix layout of the Dyson step) and w_plan_t (its S7d sub-steps): proc_grid.hpp

/// -q of every q (mf::MF::qminus: Q_q = G - Q_{qminus[q]}), checked to be an involution (the W pairing, q_groups_t)
inline std::vector<long> qminus_list(mf::MF const &mf) {
  auto qm = mf.qminus();
  const long n = mf.nqpts();
  utils::check(long(qm.size()) == n, "gw_line: qminus has {} entries for {} q", long(qm.size()), n);
  std::vector<long> v(n);
  for (long q = 0; q < n; ++q) v[q] = qm(q);
  for (long q = 0; q < n; ++q)
    utils::check(v[q] >= 0 and v[q] < n and v[v[q]] == q, "gw_line: qminus is not an involution at q = {}", q);
  return v;
}

/**
 * Coulomb matrices of all q in the block layout (resident in MEM), Z[iq] = Z(q)[P_rng, Q_rng], plus the FULL Z(q) (host)
 * for the q's this rank Dyson-solves (a q range, or, with q groups (S7e), the union of this rank's Dyson q slabs over the
 * groups: q_list). Acquired by ONE lockstep loop over all q: thc.Z(iq) is collective (every rank must call it the same
 * number of times, in the same order). Z(Gamma) carries no G=0 term (ignore_g0).
 */
template <MEMORY_SPACE MEM>
struct coulomb_blocks_t {
  aux_grid_t grid;
  long nq = 0, Np = 0;
  memory::array<MEM, ComplexType, 3> Z;     ///< (nq, nP, nQ)
  std::vector<long> q_pos;                   ///< (nq) row of Z_full holding q, or -1
  std::vector<long> qminus;                  ///< (nq) index of -q (mf::MF::qminus: Q_q = G - Q_qminus[q]); the W pairing
  nda::array<ComplexType, 3> Z_full;         ///< (nq_full, Np, Np) host

  coulomb_blocks_t() = default;

  /// q_full: absolute q range whose full matrices this rank keeps (dyson_layout_t::q_rng() + q0 of the group).
  coulomb_blocks_t(methods::thc_reader_t const &thc, aux_grid_t const &grid_, nda::range q_full,
                   utils::TimerManager &Timer)
     : coulomb_blocks_t(thc, grid_, range_list(q_full), Timer) {}

  static std::vector<long> range_list(nda::range r) {
    std::vector<long> v;
    for (long q = r.first(); q < r.last(); ++q) v.push_back(q);
    return v;
  }

  /// q_list: absolute q's whose full matrices this rank keeps (any subset of [0, N_q), e.g. the union of the Dyson q slabs
  /// of all q groups, see q_groups_t).
  coulomb_blocks_t(methods::thc_reader_t const &thc, aux_grid_t const &grid_, std::vector<long> const &q_list,
                   utils::TimerManager &Timer)
     : grid(grid_) {
    // perf 7.3: on a symmetric mesh the reader holds Z(q) of the IBZ q only (the first nq_ibz); the rows -q of IBZ q that are
    // not IBZ q themselves (ibz_t's virtual rows) get Z(-q) := conj Z(q); all other rows stay zero (never used)
    const long nqI = thc.nqpts_ibz();
    utils::check(thc.ns() == 1 and thc.npol() == 1, "gw_line::coulomb_blocks_t: spin-restricted collinear only (ns={}, npol={})",
                 thc.ns(), thc.npol());
    utils::check(thc.Np() == grid.Np, "gw_line::coulomb_blocks_t: grid Np {} != thc Np {}", grid.Np, thc.Np());
    nq = thc.nqpts();
    Np = thc.Np();
    qminus = qminus_list(*thc.MF());
    q_pos.assign(nq, -1);
    long nfull = 0;
    for (long q : q_list) {
      utils::check(q >= 0 and q < nq, "gw_line::coulomb_blocks_t: q = {} out of [0, {})", q, nq);
      if (q_pos[q] < 0) q_pos[q] = nfull++;
    }
    Timer.add("Z_gather");
    Timer.start("Z_gather");
    nda::array<ComplexType, 3> zb(nq, grid.nP, grid.nQ);
    Z_full = nda::array<ComplexType, 3>(nfull, Np, Np);
    if (nqI < nq) zb() = ComplexType(0.0);
    for (long iq = 0; iq < nqI; ++iq) {   // LOCKSTEP: identical call sequence on every rank
      auto Zq = thc.Z(int(iq));
      zb(iq, nda::range::all, nda::range::all) = Zq(grid.P_rng(), grid.Q_rng());
      if (q_pos[iq] >= 0) Z_full(q_pos[iq], nda::range::all, nda::range::all) = Zq;
      const long qm = qminus[iq];
      if (qm >= nqI) {   // virtual row -q
        zb(qm, nda::range::all, nda::range::all) = nda::conj(Zq(grid.P_rng(), grid.Q_rng()));
        if (q_pos[qm] >= 0) Z_full(q_pos[qm], nda::range::all, nda::range::all) = nda::conj(Zq);
      }
    }
    Z = memory::to_memory_space<MEM>(zb);
    Timer.stop("Z_gather");
    app_log(2, "  gw_line Coulomb blocks: Z blocks {} x {} x {} in {} ({:.4f} GB per rank), full Z(q) for {} q on the host "
               "({:.4f} GB per rank)",
            nq, grid.nP, grid.nQ, MEM == HOST_MEMORY ? "host" : "device", double(nq * grid.max_block_size()) * 16.0 / 1073741824.0,
            nfull, double(nfull * Np * Np) * 16.0 / 1073741824.0);
  }

  bool self_inverse(long iq) const { return qminus[iq] == iq; }
  bool has_full(long iq) const { return iq >= 0 and iq < long(q_pos.size()) and q_pos[iq] >= 0; }
  auto full(long iq) const {
    utils::check(has_full(iq), "gw_line::coulomb_blocks_t: full Z(q={}) not kept on this rank", iq);
    return Z_full(q_pos[iq], nda::range::all, nda::range::all);
  }
  double resident_bytes() const { return double(Z.size() + Z_full.size()) * 16.0; }
};

/**
 * Factorized pseudo-inverse of the stacked kernel of Eq. bfit at the nodes, A = [[K^-, -K^+], [-K^+, K^-]] (2nz x 2r):
 * A = U S V^dagger (gesvd), k = #{ s_i > rcond s_0 }, rcond = DBL_EPSILON max(2nz, 2r) (numpy lstsq / bosonic_basis_t::fit):
 *   [w_PQ; w_QP] = V_k S_k^{-1} U_k^dagger [W_PQ; W_QP]   ->   w = VS (U1H W + U2H W^T),
 *   U1H = U_k[0:nz, :]^dagger (k x nz), U2H = U_k[nz:2nz, :]^dagger (k x nz), VS = (V_k S_k^{-1})[0:r, :] (r x k).
 * The explicit pinv (VS [U1H | U2H]) is NOT formed: the stacked kernel has cond ~ 1e13-1e14 and the product of the
 * explicit pinv with the data loses ~eps cond (measured: refit residual 1e-5 on lih222 vs 4e-13 here); applying the
 * orthonormal U^dagger first and S^{-1} afterwards is the order gelss uses.
 */
struct bosonic_fit_t {
  long nz = 0, r = 0, k = 0;
  nda::array<ComplexType, 2> U1H, U2H, VS;

  bosonic_fit_t(bosonic_basis_t const &basis, nda::array<ComplexType, 1> const &zeta) : nz(zeta.size()), r(basis.rank) {
    auto [Km, Kp] = basis.kernels(zeta);
    const long m = 2 * nz, n = 2 * r, dm = std::min(m, n);
    nda::matrix<ComplexType, nda::F_layout> A(m, n), U(m, m), VT(n, n);
    for (long i = 0; i < nz; ++i)
      for (long j = 0; j < r; ++j) {
        A(i, j)          = Km(i, j);
        A(i, r + j)      = -Kp(i, j);
        A(nz + i, j)     = -Kp(i, j);
        A(nz + i, r + j) = Km(i, j);
      }
    nda::array<double, 1> sv(dm);
    nda::lapack::gesvd(A, sv, U, VT);
    const double rcond = std::numeric_limits<double>::epsilon() * double(std::max(m, n));
    k = 0;
    for (long i = 0; i < dm; ++i)
      if (sv(i) > rcond * sv(0)) ++k;
    U1H = nda::array<ComplexType, 2>(k, nz);
    U2H = nda::array<ComplexType, 2>(k, nz);
    VS  = nda::array<ComplexType, 2>(r, k);
    for (long i = 0; i < k; ++i)
      for (long z = 0; z < nz; ++z) {
        U1H(i, z) = std::conj(U(z, i));
        U2H(i, z) = std::conj(U(nz + z, i));
      }
    for (long j = 0; j < r; ++j)
      for (long i = 0; i < k; ++i) VS(j, i) = std::conj(VT(i, j)) / sv(i);
  }
};

namespace detail {
/**
 * Device Dyson IN PLACE for the nzl matrices D(z) = Pi(q, zeta_z) of one q (whole matrices, C layout), in sub-batches of
 * nbat nodes: M_z = I - Z Pi_z into the scratch (one strided-batched gemm), LU of all M_z (nda 3D getrf ->
 * cublasZgetrfBatched), then the right-hand sides Z are copied INTO the Pi_z slots (F layout: the F-layout solution X_z
 * IS W'_z^T in C order), solved in place (nda 3D getrs) and W^T_z = W'^T_z - Z^T. On return D(z) = W(q, zeta_z)^T.
 * Same factorization as the per-matrix path (LU of the memory = M^T, solve with op 'T').
 */
template <typename Dv_t, typename ZF_t, typename Id_t, MEMORY_SPACE MEM>
void dyson_batched_device(Dv_t &Dv, ZF_t const &ZF, Id_t const &Id, long nbat, scratch_t<MEM> &sM,
                          memory::array<MEM, int, 2> &ipiv_b, long iq, long z_first) {
  auto all       = nda::range::all;
  const long nzl = Dv.extent(0), Np = Dv.extent(1), N2 = Np * Np;
  auto ZT        = nda::transpose(ZF);
  for (long z0 = 0; z0 < nzl; z0 += nbat) {
    const long nzb = std::min(nbat, nzl - z0);
    auto Mb        = sM.template view<3>({nzb, Np, Np});
    for (long z = 0; z < nzb; ++z) Mb(z, all, all) = Id;
    // column-major: M_z^T = I - Pi_z^T Z^T (Pi_z C-layout = Pi_z^T col-major; ZF F-layout = Z col-major, op 'T')
    gemm_strided_cm('N', 'T', Np, Np, Np, ComplexType(-1.0), Dv(z0, all, all).data(), Np, N2, ZF.data(), Np, 0,
                    ComplexType(1.0), Mb.data(), Np, N2, nzb);
    auto info = nda::lapack::getrf(Mb, ipiv_b);
    for (long z = 0; z < nzb; ++z)
      utils::check(info(z) == 0, "gw_line::screened_interaction: batched getrf of I - Z Pi failed (q={}, node={}, info={})", iq,
                   z_first + z0 + z, info(z));
    memory::array_view<MEM, ComplexType, 3, nda::F_layout> Xb(std::array<long, 3>{Np, Np, nzb}, Dv(z0, all, all).data());
    for (long z = 0; z < nzb; ++z) Xb(all, all, z) = ZF;
    auto info2 = nda::lapack::getrs(Mb, Xb, ipiv_b);
    for (long z = 0; z < nzb; ++z)
      utils::check(info2(z) == 0, "gw_line::screened_interaction: batched getrs failed (q={}, node={}, info={})", iq,
                   z_first + z0 + z, info2(z));
    for (long z = 0; z < nzb; ++z) nda::tensor::add(ComplexType(-1.0), ZT, ComplexType(1.0), Dv(z0 + z, all, all));
  }
}

/// D(z) <- D(z)^T for the nzl matrices of D (whole matrices, C layout), through the scratch in sub-batches of nbat
/// (device: one cuTENSOR permutation per sub-batch; host: per matrix). Exact copies.
template <MEMORY_SPACE MEM, typename Dv_t>
void transpose_in_place(Dv_t &Dv, long nbat, scratch_t<MEM> &sM) {
  auto all       = nda::range::all;
  const long nzl = Dv.extent(0), Np = Dv.extent(1);
  for (long z0 = 0; z0 < nzl; z0 += nbat) {
    const long nzb = std::min(nbat, nzl - z0);
    const auto zr  = nda::range(z0, z0 + nzb);
    auto Tb        = sM.template view<3>({nzb, Np, Np});
    Tb             = Dv(zr, all, all);
    if constexpr (MEM == HOST_MEMORY) {
      for (long z = 0; z < nzb; ++z) Dv(z0 + z, all, all) = nda::transpose(Tb(z, all, all));
    } else {
      nda::tensor::add(ComplexType(1.0), Tb, "zab", ComplexType(0.0), Dv(zr, all, all), "zba");
    }
  }
}

/// dst <- conj(src) (equal sizes, MEM; device: cuTENSOR add with the conjugation op)
template <MEMORY_SPACE MEM>
void conj_copy(memory::array_view<MEM, ComplexType, 1> dst, memory::array_view<MEM, ComplexType, 1> src) {
  utils::check(dst.size() == src.size(), "gw_line::conj_copy: size mismatch");
  if constexpr (MEM == HOST_MEMORY) {
    for (long i = 0; i < dst.size(); ++i) dst(i) = std::conj(src(i));
  } else {
    nda::tensor::set(ComplexType(0.0), dst);   // beta = 0 below: never read uninitialized scratch
    nda::tensor::add(ComplexType(1.0), nda::conj(src), "a", ComplexType(0.0), dst, "a");
  }
}

/// S(z) <- X(z)^T for the n matrices of X (n, a, b) -> S (n, b, a), MEM (host loops; device one cuTENSOR permutation)
template <MEMORY_SPACE MEM, typename X_t, typename S_t>
void transpose_blocks(X_t const &X, S_t &&S) {
  if constexpr (MEM == HOST_MEMORY) {
    for (long z = 0; z < X.extent(0); ++z) S(z, nda::range::all, nda::range::all) = nda::transpose(X(z, nda::range::all, nda::range::all));
  } else {
    nda::tensor::add(ComplexType(1.0), X, "zab", ComplexType(0.0), S, "zba");
  }
}

/**
 * Mirror W stage (perf 7.1 (b)/(c), notes section 5), used by screened_interaction when the nodes are mirror-symmetric
 * (numerics::line_dlr::mirror_half(zeta) = n1 > 0: zeta = [z_1..z_n1 on ray 1, -conj z_1..-conj z_n1]):
 *   (b) Dyson on the ray-1 nodes only (redistribute Pi rows -> whole matrices -> Dyson in place -> transpose -> back), the
 *       ray-2 values by W(q, -conj z_i) = conj W(-q, z_i) (exact: Pi(q, -conj z) = conj Pi(-q, z), Z(-q) = conj Z(q) to the
 *       THC asymmetry of Z; [.perf71_rel] (b)): half the Dyson solves, half the volume of the two remaining redistributes,
 *       no W^T redistribute and no pair pass;
 *   (c) the transposed data of the pair fit, w(q) = VS (U1H W(q) + U2H W(-q)^T), from the ray-1 blocks only:
 *         W(-q, z_i)^T            at the ray-1 nodes: the transposed ray-1 block of -q,
 *         W(-q, -conj z_i)^T      at the ray-2 nodes: conj of the transposed ray-1 block of q (= W(q, z_i)^dagger),
 *       and the transposed ray-1 blocks come from ONE redistribute per unit {q, -q} of the transposed local blocks
 *       (a darray of W^T with origin (Q0, P0) and grid {np_Q, np_P} -> the (P, Q) block layout). On a square grid this is
 *       the pairwise exchange with the transposed rank; on any other grid the same volume with a few partners.
 * Volume per q (Pi-group units, nz = 2 n1 nodes): 1/2 + 1/2 + 1/2 instead of 3 (q = -q) or 4 (q != -q).
 */
template <MEMORY_SPACE MEM>
void screened_mirror(memory::array<MEM, ComplexType, 4> &Pi, coulomb_blocks_t<MEM> const &Zb, bosonic_basis_t const &basis,
                     aux_grid_t const &grid, utils::mpi_context_t<boost::mpi3::communicator> &mpi,
                     memory::array<MEM, ComplexType, 4> &w, utils::TimerManager &Timer,
                     memory::array<MEM, ComplexType, 4> *W_nodes, std::vector<long> const &qs, bool w_group,
                     std::vector<long> const &mrow, long n1) {
  using arr4_t  = memory::array<MEM, ComplexType, 4>;
  using arr2_t  = memory::array<MEM, ComplexType, 2>;
  using arrF_t  = memory::array<MEM, ComplexType, 2, nda::F_layout>;
  using dview_t = math::nda::distributed_array_view<arr4_t, boost::mpi3::communicator>;
  using v1_t    = memory::array_view<MEM, ComplexType, 1>;
  auto all      = nda::range::all;
  auto &comm    = mpi.comm;

  const long g = Pi.extent(0), nz = Pi.extent(1), Np = grid.Np, nP = grid.nP, nQ = grid.nQ, r = basis.rank;
  const long blk = nP * nQ;
  utils::check(nz == 2 * n1, "gw_line::screened_mirror: {} nodes, n1 = {}", nz, n1);
  dyson_layout_t lay(comm.size(), comm.rank(), g, n1, Np);
  for (auto nm : {"W_redistribute", "W_dyson", "W_fit"}) Timer.add(nm);
  for (long s = 0; s < lay.nq_loc; ++s)
    utils::check(Zb.has_full(qs[lay.q_first + s]),
                 "gw_line::screened_interaction: the Coulomb blocks do not hold the full Z of this rank's Dyson q slab");

  const long w_rows = w_group ? g : Zb.nq;
  auto w_row        = [&](long ql) { return w_group ? ql : qs[ql]; };
  if (w.extent(0) != w_rows or w.extent(1) != r or w.extent(2) != nP or w.extent(3) != nQ) {
    w = arr4_t(w_rows, r, nP, nQ);
    nda::tensor::set(ComplexType(0.0), w);
  }
  bosonic_fit_t fit(basis, basis.zeta_nodes);
  const long k  = fit.k;
  const long bc = std::clamp(long(double(g) * nz * blk / (64.0 * double(std::max(k, 1L)))), std::min(blk, 4096L), blk);
  arr2_t U1H = memory::to_memory_space<MEM>(fit.U1H), U2H = memory::to_memory_space<MEM>(fit.U2H),
         VS = memory::to_memory_space<MEM>(fit.VS), Y(k, bc);

  long nzs_cap = -1;
  if constexpr (MEM != HOST_MEMORY) {
    const double freeb = double(utils::freemem_device_effective()) * 1048576.0;
    const double per_z = 16.0 * (double(lay.np_q) * grid.max_block_size() + double(Np) * Np / double(lay.np_z));
    nzs_cap            = std::max(1L, long(0.4 * freeb / per_z));
    nzs_cap            = comm.all_reduce_value(nzs_cap, boost::mpi3::min<>{});
  }
  const w_plan_t plan(lay, grid.max_block_size(), nzs_cap);
  plan.log();
  app_log(3, "  gw_line W (mirror, perf 7.1 b/c): Dyson on {} ray-1 nodes of {}, ray 2 by conjugation, transposed fit data by "
             "one exchange per {{q, -q}}",
          n1, nz);
  [[maybe_unused]] std::optional<utils::device_pool_guard> stage_pool;
  if constexpr (MEM != HOST_MEMORY) {
    if (comm.size() > 1 and utils::device_pool_capacity() == 0) {
      const double stg = std::min(double(math::nda::detail::redistribute_chunk_bytes()),
                                  16.0 * double(lay.np_q) * plan.nzs * grid.max_block_size());
      stage_pool.emplace(std::size_t(2.0 * stg) + (std::size_t(64) << 20), "gw_line W stage");
    }
  }
  const long nzl_max = (plan.nzs + lay.np_z - 1) / lay.np_z;

  // Dyson workspace (as screened_interaction)
  arr2_t Id(Np, Np), M(Np, Np);
  {
    nda::array<ComplexType, 2> Ih(Np, Np);
    Ih() = ComplexType(0.0);
    for (long P = 0; P < Np; ++P) Ih(P, P) = ComplexType(1.0);
    Id = memory::to_memory_space<MEM>(Ih);
  }
  arrF_t ZF(Np, Np), XF(Np, Np);
  memory::array<MEM, int, 1> ipiv(Np);
  memory::array<MEM, ComplexType, 1> lwork;
  [[maybe_unused]] bool batched = false;
  long nbat                     = std::max(1L, std::min(nzl_max, 16L));
  scratch_t<MEM> sM, sTb, sD, sS, sT, sC;
  [[maybe_unused]] memory::array<MEM, int, 2> ipiv_b;
  if constexpr (MEM != HOST_MEMORY) {
    batched = env_long("COQUI_GWLINE_DYSON_BATCHED", Np <= 1024 ? 1 : 0) != 0;
    const double mat = double(Np) * Np * 16.0, freeb = double(utils::freemem_device_effective()) * 1048576.0;
    nbat = std::max(1L, std::min({nzl_max, dyson_nbat_max(), long(0.25 * freeb / mat)}));
    if (batched) ipiv_b = memory::array<MEM, int, 2>(nbat, Np);
  }

  // perf 7.5b profile of this path (env COQUI_GWLINE_W_PROFILE = 1): a barrier before every redistribute (timer W_wait:
  // the imbalance of the preceding step, otherwise hidden in W_redistribute) and the message pattern of the calls
  const bool wprof = env_long("COQUI_GWLINE_W_PROFILE", 0) != 0;
  long n_calls = 0;
  double t_wait = 0.0, t_red = 0.0;
  auto pbar = [&]() {
    if (not wprof) return;
    Timer.add("W_wait");
    Timer.start("W_wait");
    const auto t0 = std::chrono::steady_clock::now();
    comm.barrier();
    t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Timer.stop("W_wait");
  };
  auto redist = [&](auto &A, auto &B) {
    const auto t0 = std::chrono::steady_clock::now();
    math::nda::redistribute(A, B);
    t_red += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n_calls;
  };
  if (wprof) {
    // forward call of a sub-step: every rank sends its blocks of the n_act x nzs matrices; the receivers are the ranks of the
    // active q pools with a non-empty zeta chunk (one message each of nzl blocks)
    const long nzs = plan.nzs, nzl = (nzs + lay.np_z - 1) / lay.np_z;
    app_log(2, "  gw_line W profile (redistribute path): grid {} x {}, block {} x {} ({:.1f} KB), Dyson pools (q, zeta) = ({} x {}), "
               "{} q sub-steps x {} zeta sub-slabs of {} nodes; forward / backward call: <= {} messages per rank of <= {} blocks "
               "({:.1f} KB), {:.1f} MB per rank; transposed exchange: {} calls (one per {{q, -q}}) of {} x {} blocks to <= {} partners",
            grid.np_P, grid.np_Q, nP, nQ, 16.0 * nP * nQ / 1024.0, lay.np_q, lay.np_z, plan.nsub_q, plan.n_zsub(), nzs,
            lay.np_q * std::min(lay.np_z, nzs), nzl, 16.0 * nzl * nP * nQ / 1024.0, 16.0 * lay.np_q * nzs * nP * nQ / 1048576.0,
            g, 2, n1, std::max(grid.np_P, grid.np_Q) / std::min(grid.np_P, grid.np_Q) + 1);
  }

  // ---- (b) Dyson on the ray-1 nodes
  const std::array<long, 4> bgrid = {1, 1, grid.np_P, grid.np_Q}, ones = {1, 1, 1, 1};
  for (long s = 0; s < plan.nsub_q; ++s) {
    const long na   = plan.n_act(s);
    const bool act  = s < lay.nq_loc;
    const long iq_a = qs[lay.q_first + (act ? s : 0)];
    if (act) {
      Timer.start("W_dyson");
      nda::matrix<ComplexType, nda::F_layout> zf_h(Zb.full(iq_a));
      ZF = zf_h;
      Timer.stop("W_dyson");
    }
    for (long za = 0; za < n1; za += plan.nzs) {
      const long nzs                = std::min(plan.nzs, n1 - za);
      const auto zrng               = nda::range(za, za + nzs);
      const auto [zf, nzl]          = plan.z_chunk(nzs);
      const std::array<long, 4> gsh = {na, nzs, Np, Np};
      const bool solve              = act and nzl > 0;

      Timer.start("W_redistribute");
      auto Tb = sTb.template view<4>({na, nzs, nP, nQ});
      for (long p = 0; p < na; ++p) Tb(p, all, all, all) = Pi(plan.q_row(p, s), zrng, all, all);
      dview_t dTb(std::addressof(comm), bgrid, gsh, {0, 0, grid.P0, grid.Q0}, ones, Tb);
      auto D4 = sD.template view<4>({act ? 1L : 0L, nzl, Np, Np});
      dview_t dD(std::addressof(comm), lay.pgrid(), gsh, {act ? lay.ip_q : na, zf, 0, 0}, ones, D4);
      Timer.stop("W_redistribute");
      pbar();
      Timer.start("W_redistribute");
      redist(dTb, dD);
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        device_mem_probe();
      }
      Timer.stop("W_redistribute");

      Timer.start("W_dyson");
      if (solve) {
        auto Dv   = D4(0, all, all, all);
        bool done = false;
        if constexpr (MEM != HOST_MEMORY) {
          if (batched) {
            dyson_batched_device(Dv, ZF, Id, nbat, sM, ipiv_b, iq_a, za + zf);
            done = true;
          }
        }
        if (not done) {
          auto ZT = nda::transpose(ZF);
          for (long izl = 0; izl < nzl; ++izl) {
            auto Pv = Dv(izl, all, all);
            M       = Id;
            nda::blas::gemm(ComplexType(-1.0), ZF, Pv, ComplexType(1.0), M);   // M = I - Z Pi
            int info = nda::lapack::getrf(M, ipiv, lwork);
            utils::check(info == 0, "gw_line::screened_interaction: getrf of I - Z Pi failed (q={}, node={}, info={})", iq_a,
                         za + zf + izl, info);
            XF   = ZF;
            info = nda::lapack::getrs(M, XF, ipiv);   // XF = (I - Z Pi)^{-1} Z
            utils::check(info == 0, "gw_line::screened_interaction: getrs failed (q={}, node={}, info={})", iq_a, za + zf + izl,
                         info);
            Pv = nda::transpose(XF);
            if constexpr (MEM == HOST_MEMORY) Pv -= ZT;
            else nda::tensor::add(ComplexType(-1.0), ZT, ComplexType(1.0), Pv);
          }
        }
        transpose_in_place<MEM>(Dv, nbat, sM);   // W^T -> W
      }
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        device_mem_probe();
      }
      Timer.stop("W_dyson");

      pbar();
      Timer.start("W_redistribute");
      redist(dD, dTb);
      for (long p = 0; p < na; ++p) Pi(plan.q_row(p, s), zrng, all, all) = Tb(p, all, all, all);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_redistribute");
    }
  }

  // ---- (b) ray 2: W(q, -conj z_i) = conj W(-q, z_i)   (ray-1 rows are read only)
  Timer.start("W_dyson");   // the ray-2 values (counted with the Dyson)
  auto half = [&](long row, long h) {   // the n1 nodes of ray h (0, 1) of group row `row`, flat
    return v1_t(std::array<long, 1>{n1 * blk}, Pi.data() + (row * nz + h * n1) * blk);
  };
  for (long i = 0; i < g; ++i) conj_copy<MEM>(half(i, 1), half(mrow[i], 0));
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("W_dyson");

  // ---- (c) per unit {q, -q}: transposed ray-1 blocks by one exchange, then the fits
  auto fit_row = [&](long ql, auto const &WT) {
    auto W2  = nda::reshape(Pi(ql, all, all, all), std::array<long, 2>{nz, blk});
    auto WT2 = nda::reshape(WT, std::array<long, 2>{nz, blk});
    auto w2  = nda::reshape(w(w_row(ql), all, all, all), std::array<long, 2>{r, blk});
    for (long c0 = 0; c0 < blk; c0 += bc) {
      const auto cr = nda::range(c0, std::min(blk, c0 + bc));
      auto Yc       = Y(all, nda::range(cr.size()));
      nda::blas::gemm(ComplexType(1.0), U1H, W2(all, cr), ComplexType(0.0), Yc);
      nda::blas::gemm(ComplexType(1.0), U2H, WT2(all, cr), ComplexType(1.0), Yc);
      nda::blas::gemm(ComplexType(1.0), VS, Yc, ComplexType(0.0), w2(all, cr));
    }
  };
  const std::array<long, 4> tgrid = {1, 1, grid.np_Q, grid.np_P};
  for (long i = 0; i < g; ++i) {
    const long j = mrow[i];
    if (j < i) continue;
    const long nu         = (j == i) ? 1 : 2;
    const long rows[2]    = {i, j};
    pbar();
    Timer.start("W_redistribute");
    auto S = sS.template view<4>({nu, n1, nQ, nP});   // local transposed blocks: the piece (Q_rng, P_rng) of W^T
    for (long u = 0; u < nu; ++u) transpose_blocks<MEM>(Pi(rows[u], nda::range(n1), all, all), S(u, all, all, all));
    auto T = sT.template view<4>({nu, n1, nP, nQ});
    const std::array<long, 4> gsh = {nu, n1, Np, Np};
    dview_t dS(std::addressof(comm), tgrid, gsh, {0, 0, grid.Q0, grid.P0}, ones, S);
    dview_t dT(std::addressof(comm), bgrid, gsh, {0, 0, grid.P0, grid.Q0}, ones, T);
    redist(dS, dT);   // T(u) = block (P_rng, Q_rng) of W(q_u, z_ray1)^T
    if constexpr (MEM != HOST_MEMORY) utils::device_sync();
    Timer.stop("W_redistribute");
    Timer.start("W_fit");
    auto WT = sC.template view<3>({nz, nP, nQ});
    for (long u = 0; u < nu; ++u) {
      const long pu = (nu == 1) ? 0 : 1 - u;   // the partner -q
      WT(nda::range(n1), all, all) = T(pu, all, all, all);
      conj_copy<MEM>(v1_t(std::array<long, 1>{n1 * blk}, WT.data() + n1 * blk),
                     v1_t(std::array<long, 1>{n1 * blk}, T.data() + u * n1 * blk));
      fit_row(rows[u], WT);
    }
    if constexpr (MEM != HOST_MEMORY) {
      utils::device_sync();
      device_mem_probe();
    }
    Timer.stop("W_fit");
  }

  if (wprof) {
    double x[2] = {t_wait, t_red};
    double xm[2] = {0, 0};
    comm.all_reduce_n(x, 2, xm, boost::mpi3::max<>{});
    double xa[2] = {0, 0};
    comm.all_reduce_n(x, 2, xa, std::plus<>{});
    app_log(2, "  gw_line W profile (redistribute path): {} redistribute calls, inside the calls avg {:.3f} / max {:.3f} s, "
               "waiting at the barriers before them avg {:.3f} / max {:.3f} s",
            n_calls, xa[1] / comm.size(), xm[1], xa[0] / comm.size(), xm[0]);
  }
  if (W_nodes != nullptr) *W_nodes = std::move(Pi);
  Pi = arr4_t{};
}

/**
 * perf 7.5b: node and cross-node communicators of the node-shared W stage. node: the ranks of one shared-memory node
 * (MPI_COMM_TYPE_SHARED; env COQUI_GWLINE_W_NODE_SIZE > 0 splits it further into virtual nodes of that many ranks, for
 * tests of the cross-node exchange on one machine); cross: the ranks with the same local index l on every node, ordered
 * by the nodes' leader rank (cross rank = node index a). ok: every node has the same number of ranks.
 */
struct w_nodes_t {
  MPI_Comm node = MPI_COMM_NULL, cross = MPI_COMM_NULL;
  int L = 1, l = 0, NN = 1, a = 0;
  bool ok = false;
  explicit w_nodes_t(MPI_Comm comm) {
    int rank = 0, np = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    MPI_Comm shm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm);
    const long vs = env_long("COQUI_GWLINE_W_NODE_SIZE", 0);
    if (vs > 0) {
      int sr = 0;
      MPI_Comm_rank(shm, &sr);
      MPI_Comm_split(shm, int(sr / vs), sr, &node);
      MPI_Comm_free(&shm);
    } else {
      node = shm;
    }
    MPI_Comm_size(node, &L);
    MPI_Comm_rank(node, &l);
    int lr[2] = {L, -L};
    MPI_Allreduce(MPI_IN_PLACE, lr, 2, MPI_INT, MPI_MIN, comm);
    int leader = rank;
    MPI_Bcast(&leader, 1, MPI_INT, 0, node);
    MPI_Comm_split(comm, l, leader, &cross);
    MPI_Comm_size(cross, &NN);
    MPI_Comm_rank(cross, &a);
    ok = (lr[0] == -lr[1]) and long(NN) * long(L) == long(np);
  }
  ~w_nodes_t() {
    if (cross != MPI_COMM_NULL) MPI_Comm_free(&cross);
    if (node != MPI_COMM_NULL) MPI_Comm_free(&node);
  }
  w_nodes_t(w_nodes_t const &)            = delete;
  w_nodes_t &operator=(w_nodes_t const &) = delete;
};

/**
 * perf 7.5b: the node-shared buffer of screened_mirror_node. Backend (env COQUI_GWLINE_W_SHM):
 *   "posix" (default): shm_open in /dev/shm (tmpfs) by the node's local rank 0, mapped by every rank of the node; kept
 *           between calls (env COQUI_GWLINE_W_KEEP, default 1) and reused while large enough, so that the page faults of the
 *           first touch are paid once per run (freed at exit);
 *   "mpi"  : MPI_Win_allocate_shared per call (OpenMPI 4.1 puts the backing file in its session directory, on rusty the
 *           NVMe /tmp: 16.8 s of page faults + write-back for the 21 GB of si444 IBZ, measured in job 7203962).
 * acquire() is collective over comm (all ranks of all nodes): false on every rank if any node fails (no space in /dev/shm,
 * shm_open / mmap errors); the caller then falls back.
 */
class node_shm_t {
 public:
  ComplexType *data = nullptr;
  bool kept = false;   ///< the buffer of a previous call was reused
  std::string backend;
  node_shm_t()                              = default;
  node_shm_t(node_shm_t const &)            = delete;
  node_shm_t &operator=(node_shm_t const &) = delete;
  ~node_shm_t() { release(); }

  bool acquire(MPI_Comm comm, MPI_Comm node, size_t bytes) {
    backend = env_string_w("COQUI_GWLINE_W_SHM", "posix");
    bytes   = std::max<size_t>(bytes, 4096);
    int lrank = 0, gnp = 0, grank = 0;
    MPI_Comm_rank(node, &lrank);
    MPI_Comm_size(comm, &gnp);
    MPI_Comm_rank(comm, &grank);
    int ok = 1;
    if (backend == "mpi") {
      ComplexType *base  = nullptr;
      const MPI_Aint wb  = (lrank == 0) ? MPI_Aint(bytes) : MPI_Aint(0);
      if (MPI_Win_allocate_shared(wb, int(sizeof(ComplexType)), MPI_INFO_NULL, node, &base, &win_) != MPI_SUCCESS) ok = 0;
      if (ok) {
        MPI_Aint sz = 0;
        int du      = 0;
        MPI_Win_shared_query(win_, 0, &sz, &du, &data);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
      }
      MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, comm);
      return ok != 0;
    }
    // posix: reuse the kept mapping when it belongs to the same node group and is large enough (collective decision)
    auto &c   = cache();
    int leader = grank;
    MPI_Bcast(&leader, 1, MPI_INT, 0, node);
    int nl = 0;
    MPI_Comm_size(node, &nl);
    int reuse = (c.p != nullptr and c.bytes >= bytes and c.leader == leader and c.nl == nl and c.gnp == gnp) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &reuse, 1, MPI_INT, MPI_MIN, comm);
    if (reuse) {
      data = static_cast<ComplexType *>(c.p);
      kept = true;
      own_ = false;
      return true;
    }
    drop_cache();   // every rank (the decision is collective)
    char name[128] = {0};
    if (lrank == 0) {
      static long counter = 0;
      std::snprintf(name, sizeof(name), "/coqui_gwline_w_%d_%ld", int(getpid()), counter++);
      struct statvfs sv;
      if (statvfs("/dev/shm", &sv) == 0 and double(sv.f_bavail) * double(sv.f_bsize) < 1.05 * double(bytes)) ok = 0;
      int fd = ok ? shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600) : -1;
      if (fd < 0) ok = 0;
      if (ok and ftruncate(fd, off_t(bytes)) != 0) ok = 0;
      if (fd >= 0) close(fd);
      if (not ok and fd >= 0) shm_unlink(name);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, node);
    MPI_Bcast(name, int(sizeof(name)), MPI_CHAR, 0, node);
    void *p = nullptr;
    if (ok) {
      int fd = shm_open(name, O_RDWR, 0600);
      if (fd >= 0) {
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) p = nullptr;
      }
      if (p == nullptr) ok = 0;
    }
    MPI_Barrier(node);
    if (lrank == 0 and name[0] != '\0') shm_unlink(name);   // the memory lives until the last munmap
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (not ok) {
      if (p != nullptr) munmap(p, bytes);
      return false;
    }
    data = static_cast<ComplexType *>(p);
    // keep only a buffer that is small against the node's available memory (it then also sits through the Sigma stage, which
    // the q plan did not budget): bytes <= COQUI_GWLINE_W_KEEP_FRAC (default 0.1) x MemAvailable, the same on every node
    int keep = env_long("COQUI_GWLINE_W_KEEP", 1) != 0 ? 1 : 0;
    if (keep and lrank == 0) {
      double avail = -1.0;
      if (FILE *fp = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), fp))
          if (std::strncmp(line, "MemAvailable:", 13) == 0) avail = 1024.0 * std::strtod(line + 13, nullptr);
        std::fclose(fp);
      }
      const char *fv   = std::getenv("COQUI_GWLINE_W_KEEP_FRAC");
      const double frac = (fv != nullptr and *fv != '\0') ? std::strtod(fv, nullptr) : 0.1;
      if (avail > 0.0 and double(bytes) > frac * avail) keep = 0;
    }
    MPI_Bcast(&keep, 1, MPI_INT, 0, node);
    MPI_Allreduce(MPI_IN_PLACE, &keep, 1, MPI_INT, MPI_MIN, comm);
    if (keep) {
      c.p      = p;   // (no temporary cache_t: its destructor unmaps)
      c.bytes  = bytes;
      c.leader = leader;
      c.nl     = nl;
      c.gnp    = gnp;
      own_     = false;
    } else {
      own_   = true;
      bytes_ = bytes;
    }
    return true;
  }
  /// collective over the node (barrier) for the mpi backend; posix: unmaps a non-kept buffer
  void release() {
    if (win_ != MPI_WIN_NULL) {
      int fin = 0;
      MPI_Finalized(&fin);
      if (not fin) {
        MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
      }
      win_ = MPI_WIN_NULL;
    }
    if (own_ and data != nullptr) munmap(static_cast<void *>(data), bytes_);
    own_ = false;
    data = nullptr;
  }
  /// fences + node barrier (the copies of the other ranks are visible after it)
  void sync(MPI_Comm node) const {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (win_ != MPI_WIN_NULL) MPI_Win_sync(win_);
    MPI_Barrier(node);
    if (win_ != MPI_WIN_NULL) MPI_Win_sync(win_);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
  static double kept_bytes() { return double(cache().bytes); }

 private:
  struct cache_t {
    void *p      = nullptr;
    size_t bytes = 0;
    int leader = -1, nl = 0, gnp = 0;
    cache_t() = default;
    cache_t(cache_t const &) = delete;
    cache_t &operator=(cache_t const &) = delete;
    ~cache_t() {
      if (p != nullptr) munmap(p, bytes);
    }
  };
  static cache_t &cache() {
    static cache_t c;
    return c;
  }
  static void drop_cache() {
    auto &c = cache();
    if (c.p != nullptr) munmap(c.p, c.bytes);
    c.p     = nullptr;
    c.bytes = 0;
  }
  static std::string env_string_w(char const *nm, char const *def) {
    char const *v = std::getenv(nm);
    return (v == nullptr or *v == '\0') ? std::string(def) : std::string(v);
  }
  MPI_Win win_ = MPI_WIN_NULL;
  bool own_    = false;
  size_t bytes_ = 0;
};

/// [first, last) of chunk i of [b, e) in n chunks (itertools::chunk_range)
inline std::array<long, 2> w_chunk(long b, long e, long n, long i) {
  auto [x0, x1] = itertools::chunk_range(b, e, n, i);
  return {long(x0), long(x1)};
}

/**
 * perf 7.5b: the mirror W stage (screened_mirror, same arithmetic) with the (P, Q)-block -> whole-matrix transposition
 * through NODE-SHARED memory instead of three all-to-all redistributes. Host only.
 *   matrices    : the M = g n1 ray-1 Dyson problems m = i n1 + z (group row i, node z), node a solves the contiguous chunk
 *                 [Ma0, Ma1) of m (chunk_range over the NN nodes), its rank l the chunk [m0, m1) of it (over the L ranks):
 *                 balanced to one matrix (the q-pool layout leaves pools idle when np_q does not divide g);
 *   forward     : every rank writes its block (P_rng, Q_rng) of the matrices of its node straight into the node buffer
 *                 (node_shm_t: /dev/shm, kept between calls; C layout Np x Np per matrix); blocks of the matrices of other nodes go to the
 *                 rank with the same local index there (pairwise MPI_Sendrecv rounds over the cross communicator), which
 *                 writes them at the sender's (P, Q) coordinates;
 *   Dyson       : in place in the buffer, per matrix the operations of screened_mirror (gemm I - Z Pi, getrf, getrs with
 *                 Z, W = X - Z: the same values as its transpose / subtract / transpose sequence);
 *   backward    : every rank reads its block of W(m) (-> Pi rows) and, for the pair fit, the transposed block
 *                 W(m)[Q_rng, P_rng]^T (= its block of W(m)^T) from the buffer; for the matrices of other nodes the
 *                 counterpart there packs both and sends them back (pairwise rounds);
 *   ray 2, fit  : as screened_mirror (conj of the mirror row; w = VS (U1H W + U2H W^T) per row, the same fit_row).
 * All moves are exact copies: W at the nodes and the residues equal those of the redistribute path up to the BLAS
 * arithmetic of the Dyson on another buffer ([gw_line][w] gates). Returns false (collectively, before any work) when the
 * layout does not apply: unequal ranks per node, a missing full Z(q) of a rank's rows (the Coulomb blocks keep the
 * union of both layouts' rows, q_groups_t::dyson_q_list), MPI counts beyond int, or a transient above the budget
 * (max(1.25 x the redistribute path's transient, 64 MB)).
 */
inline bool screened_mirror_node(memory::array<HOST_MEMORY, ComplexType, 4> &Pi, coulomb_blocks_t<HOST_MEMORY> const &Zb,
                                 bosonic_basis_t const &basis, aux_grid_t const &grid, boost::mpi3::communicator &comm,
                                 memory::array<HOST_MEMORY, ComplexType, 4> &w, utils::TimerManager &Timer,
                                 memory::array<HOST_MEMORY, ComplexType, 4> *W_nodes, std::vector<long> const &qs, bool w_group,
                                 std::vector<long> const &mrow, long n1, double old_transient) {
  using Arr4_t  = memory::array<HOST_MEMORY, ComplexType, 4>;
  using clk     = std::chrono::steady_clock;
  auto secs     = [](clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); };
  auto all      = nda::range::all;
  const long g = Pi.extent(0), nz = Pi.extent(1), Np = grid.Np, nP = grid.nP, nQ = grid.nQ;
  const long blk = nP * nQ, N2 = Np * Np, M = g * n1;
  const long r   = basis.rank;
  const auto t_all = clk::now();

  w_nodes_t nd(comm.get());
  const long NN = nd.NN, L = nd.L, a = nd.a, l = nd.l;
  (void)L;
  // geometry of the counterparts (b, l), b = 0..NN-1
  std::vector<long> geo(4 * NN);
  {
    long me[4] = {grid.P0, grid.nP, grid.Q0, grid.nQ};
    MPI_Allgather(me, 4, MPI_LONG, geo.data(), 4, MPI_LONG, nd.cross);
  }
  // this rank's matrices: chunk of [0, M) over the global ranks (q_groups_t::dyson_q_list keeps their full Z(q)); the node's
  // matrices: the union over its ranks, contiguous when the node's ranks are consecutive (else: the redistribute path)
  const auto [m0, m1] = w_chunk(0, M, comm.size(), comm.rank());
  long mr[3] = {m0, -m1, m1 - m0};
  MPI_Allreduce(MPI_IN_PLACE, mr, 2, MPI_LONG, MPI_MIN, nd.node);
  MPI_Allreduce(MPI_IN_PLACE, mr + 2, 1, MPI_LONG, MPI_SUM, nd.node);
  const long Ma0 = mr[0], Ma1 = -mr[1];
  std::vector<long> nr(2 * NN);
  {
    long me2[2] = {Ma0, Ma1};
    MPI_Allgather(me2, 2, MPI_LONG, nr.data(), 2, MPI_LONG, nd.cross);
  }
  auto nrange = [&](long b) { return std::array<long, 2>{nr[2 * b], nr[2 * b + 1]}; };
  // feasibility (collective)
  long bad = nd.ok ? 0 : 1;
  if (not bad and mr[2] != Ma1 - Ma0) bad = 5;
  for (long m = m0; m < m1 and not bad; ++m)
    if (not Zb.has_full(qs[m / n1])) bad = 2;
  double peak = 16.0 * double(Ma1 - Ma0) * double(N2) / double(L) + 16.0 * double(nz) * double(blk);
  // cross-node messages: chunks of <= xcap elements per side (COQUI_GWLINE_W_XCHUNK_MB, default 16)
  long bmax = 0;
  for (long b = 0; b < NN; ++b) bmax = std::max(bmax, geo[4 * b + 1] * geo[4 * b + 3]);
  long mbmax = 0;
  for (long b = 0; b < NN; ++b) mbmax = std::max(mbmax, nrange(b)[1] - nrange(b)[0]);
  const long xcap = std::max(2 * bmax, std::min(2 * mbmax * bmax, long(double(env_long("COQUI_GWLINE_W_XCHUNK_MB", 16)) * 1048576.0 / 16.0)));
  if (NN > 1) {
    peak += 16.0 * double(M - (Ma1 - Ma0)) * double(blk) + 2.0 * 16.0 * double(xcap);
    if (xcap > long(INT_MAX)) bad = 3;
  }
  const double budget = std::max(1.25 * old_transient, 64.0 * 1048576.0);
  if (not bad and peak > budget) bad = 4;
  bad = comm.all_reduce_value(bad, boost::mpi3::max<>{});
  const double peak_max = comm.all_reduce_value(peak, boost::mpi3::max<>{});
  if (bad) {
    app_log(2, "  gw_line W (perf 7.5b node path) not used ({}): the redistribute path follows",
            bad == 1 ? "unequal ranks per node" : bad == 2 ? "full Z(q) of a row not kept" : bad == 3 ? "MPI count beyond int"
            : bad == 4 ? "transient above the budget" : "the ranks of a node are not consecutive");
    return false;
  }
  // node-shared buffer of the node's matrices (C layout, Np x Np each)
  auto tw = clk::now();
  node_shm_t shm;
  if (not shm.acquire(comm.get(), nd.node, size_t(Ma1 - Ma0) * size_t(N2) * sizeof(ComplexType))) {
    app_log(2, "  gw_line W (perf 7.5b node path) not used (node-shared buffer \"{}\" not available): the redistribute path follows",
            shm.backend);
    return false;
  }
  const double t_alloc = secs(tw);
  app_log(2, "  gw_line W (perf 7.5b node path): {} ray-1 matrices on {} node(s) x {} ranks, <= {} per rank, node buffer "
             "{:.2f} GB ({}{}), transient <= {:.3f} GB per rank (redistribute path {:.3f} GB)",
          M, NN, L, (M + comm.size() - 1) / comm.size(), 16.0 * double(Ma1 - Ma0) * double(N2) / 1073741824.0, shm.backend,
          shm.kept ? ", kept from the previous call" : "", peak_max / 1073741824.0, old_transient / 1073741824.0);

  for (auto nm : {"W_redistribute", "W_dyson", "W_fit", "W_wait"}) Timer.add(nm);
  const long w_rows = w_group ? g : Zb.nq;
  if (w.extent(0) != w_rows or w.extent(1) != r or w.extent(2) != nP or w.extent(3) != nQ) {
    w = Arr4_t(w_rows, r, nP, nQ);
    w() = ComplexType(0.0);
  }
  bosonic_fit_t fit(basis, basis.zeta_nodes);
  double t_fwd = 0.0, t_fx = 0.0, t_dys = 0.0, t_bwd = 0.0, t_bx = 0.0, t_fit = 0.0, t_wait = 0.0;
  double by_intra = 0.0, by_inter = 0.0;

  Timer.start("W_redistribute");
  tw = clk::now();
  ComplexType *W0 = shm.data;
  auto mat = [&](long m) { return W0 + (m - Ma0) * N2; };
  auto put = [&](ComplexType *Wm, long P0, long nP_, long Q0, long nQ_, ComplexType const *src) {
    for (long p = 0; p < nP_; ++p) std::copy_n(src + p * nQ_, nQ_, Wm + (P0 + p) * Np + Q0);
  };
  auto get = [&](ComplexType const *Wm, long P0, long nP_, long Q0, long nQ_, ComplexType *dst) {
    for (long p = 0; p < nP_; ++p) std::copy_n(Wm + (P0 + p) * Np + Q0, nQ_, dst + p * nQ_);
  };
  // transposed block: dst(p, q) = Wm(Q0 + q, P0 + p), (nP_ x nQ_), tiled
  auto get_t = [&](ComplexType const *Wm, long P0, long nP_, long Q0, long nQ_, ComplexType *dst) {
    constexpr long TB = 32;
    for (long q0 = 0; q0 < nQ_; q0 += TB)
      for (long p0 = 0; p0 < nP_; p0 += TB)
        for (long q = q0; q < std::min(nQ_, q0 + TB); ++q) {
          ComplexType const *s = Wm + (Q0 + q) * Np + P0;
          for (long p = p0; p < std::min(nP_, p0 + TB); ++p) dst[p * nQ_ + q] = s[p];
        }
  };
  auto pi_blk = [&](long m) { return Pi.data() + ((m / n1) * nz + (m % n1)) * blk; };
  auto sync_node = [&]() {
    const auto t0 = clk::now();
    shm.sync(nd.node);
    t_wait += secs(t0);
  };
  const MPI_Datatype ct = MPI_CXX_DOUBLE_COMPLEX;
  // one round of the cross-node exchange: ns matrices of es elements to `to` (pack(j, dst)), nr matrices of er elements from
  // `from` (unpack(j, src)), in chunks of <= xcap elements (one tag per direction; the chunks of a pair are matched in order,
  // both partners derive the same chunking)
  std::vector<ComplexType> xs, xr;
  auto xround = [&](long to, long ns, long es, auto &&pack, long from, long nr, long er, auto &&unpack, int tag0) {
    const long cs = std::max(1L, xcap / std::max(1L, es)), cr = std::max(1L, xcap / std::max(1L, er));
    const long ks = (ns + cs - 1) / cs, kr = (nr + cr - 1) / cr;
    xs.resize(size_t(std::min(ns, cs) * es));
    xr.resize(size_t(std::min(nr, cr) * er));
    for (long c = 0; c < std::max(ks, kr); ++c) {
      MPI_Request rq[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
      const long j0r = c * cr, j1r = std::min(nr, j0r + cr);
      if (c < kr) MPI_Irecv(xr.data(), int((j1r - j0r) * er), ct, int(from), tag0, nd.cross, &rq[0]);
      if (c < ks) {
        const long j0 = c * cs, j1 = std::min(ns, j0 + cs);
        for (long j = j0; j < j1; ++j) pack(j, xs.data() + (j - j0) * es);
        MPI_Isend(xs.data(), int((j1 - j0) * es), ct, int(to), tag0, nd.cross, &rq[1]);
        by_inter += 16.0 * double((j1 - j0) * es);
      }
      MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
      if (c < kr)
        for (long j = j0r; j < j1r; ++j) unpack(j, xr.data() + (j - j0r) * er);
    }
  };

  // ---- forward: own blocks of the node's matrices into the window
  for (long m = Ma0; m < Ma1; ++m) put(mat(m), grid.P0, nP, grid.Q0, nQ, pi_blk(m));
  by_intra += 16.0 * double(Ma1 - Ma0) * double(blk);
  // blocks of the other nodes' matrices: pairwise rounds with the counterparts
  if (NN > 1) {
    const auto tx = clk::now();
    for (long s = 1; s < NN; ++s) {
      const long to = (a + s) % NN, from = (a - s + NN) % NN;
      const auto [t0, t1] = nrange(to);
      const long fb       = geo[4 * from + 1] * geo[4 * from + 3];
      xround(
          to, t1 - t0, blk, [&](long j, ComplexType *d) { std::copy_n(pi_blk(t0 + j), blk, d); }, from, Ma1 - Ma0, fb,
          [&](long j, ComplexType const *x) { put(mat(Ma0 + j), geo[4 * from], geo[4 * from + 1], geo[4 * from + 2], geo[4 * from + 3], x); },
          751);
    }
    t_fx = secs(tx);
  }
  sync_node();
  t_fwd = secs(tw);
  Timer.stop("W_redistribute");

  // ---- Dyson on this rank's matrices, in place (screened_mirror's arithmetic)
  Timer.start("W_dyson");
  const auto td = clk::now();
  {
    nda::matrix<ComplexType> Id(Np, Np), Mm(Np, Np);
    Id() = ComplexType(0.0);
    for (long P = 0; P < Np; ++P) Id(P, P) = ComplexType(1.0);
    nda::matrix<ComplexType, nda::F_layout> ZF(Np, Np), XF(Np, Np);
    nda::array<int, 1> ipiv(Np);
    nda::array<ComplexType, 1> lwork;
    long izf = -1;
    for (long m = m0; m < m1; ++m) {
      const long i = m / n1, z = m % n1;
      if (i != izf) {
        nda::matrix<ComplexType, nda::F_layout> zf_h(Zb.full(qs[i]));
        ZF  = zf_h;
        izf = i;
      }
      nda::array_view<ComplexType, 2> Pv(std::array<long, 2>{Np, Np}, mat(m));
      Mm = Id;
      nda::blas::gemm(ComplexType(-1.0), ZF, Pv, ComplexType(1.0), Mm);   // M = I - Z Pi
      int info = nda::lapack::getrf(Mm, ipiv, lwork);
      utils::check(info == 0, "gw_line::screened_interaction: getrf of I - Z Pi failed (q={}, node={}, info={})", qs[i], z, info);
      XF   = ZF;
      info = nda::lapack::getrs(Mm, XF, ipiv);   // XF = (I - Z Pi)^{-1} Z
      utils::check(info == 0, "gw_line::screened_interaction: getrs failed (q={}, node={}, info={})", qs[i], z, info);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) Pv(P, Q) = XF(P, Q) - ZF(P, Q);   // W = X - Z
    }
  }
  t_dys = secs(td);
  Timer.stop("W_dyson");
  Timer.start("W_wait");
  sync_node();
  Timer.stop("W_wait");

  // ---- backward: W blocks -> Pi rows (ray 1), transposed blocks for the fit (own node: read later from the window)
  Timer.start("W_redistribute");
  const auto tb = clk::now();
  for (long m = Ma0; m < Ma1; ++m) get(mat(m), grid.P0, nP, grid.Q0, nQ, pi_blk(m));
  by_intra += 16.0 * double(Ma1 - Ma0) * double(blk);
  std::vector<ComplexType> tr;            // transposed blocks of the other nodes' matrices, m order (m outside [Ma0, Ma1))
  std::vector<long> tr_off(NN > 1 ? M : 0, -1);
  if (NN > 1) {
    const auto tx = clk::now();
    tr.resize(size_t((M - (Ma1 - Ma0)) * blk));
    long off = 0;
    for (long s = 1; s < NN; ++s) {
      const long to = (a + s) % NN, from = (a - s + NN) % NN;
      const long P0t = geo[4 * to], nPt = geo[4 * to + 1], Q0t = geo[4 * to + 2], nQt = geo[4 * to + 3], bt = nPt * nQt;
      const auto [f0, f1] = nrange(from);
      xround(
          to, Ma1 - Ma0, 2 * bt,
          [&](long j, ComplexType *d) {
            get(mat(Ma0 + j), P0t, nPt, Q0t, nQt, d);
            get_t(mat(Ma0 + j), P0t, nPt, Q0t, nQt, d + bt);
          },
          from, f1 - f0, 2 * blk,
          [&](long j, ComplexType const *x) {
            std::copy_n(x, blk, pi_blk(f0 + j));
            std::copy_n(x + blk, blk, tr.data() + off);
            tr_off[f0 + j] = off;
            off += blk;
          },
          752);
    }
    t_bx = secs(tx);
  }
  t_bwd = secs(tb);
  Timer.stop("W_redistribute");

  // ---- ray 2: W(q, -conj z_i) = conj W(-q, z_i)
  Timer.start("W_dyson");
  for (long i = 0; i < g; ++i) {
    ComplexType *dst = Pi.data() + (i * nz + n1) * blk;
    ComplexType const *src = Pi.data() + (mrow[i] * nz) * blk;
    for (long x = 0; x < n1 * blk; ++x) dst[x] = std::conj(src[x]);
  }
  Timer.stop("W_dyson");

  // ---- fits: w(q) = VS (U1H W(q) + U2H W(-q)^T)
  Timer.start("W_fit");
  const auto tf = clk::now();
  {
    const long k  = fit.k;
    const long bc = std::clamp(long(double(g) * nz * blk / (64.0 * double(std::max(k, 1L)))), std::min(blk, 4096L), blk);
    nda::matrix<ComplexType> U1H = fit.U1H, U2H = fit.U2H, VS = fit.VS, Y(k, bc);
    nda::array<ComplexType, 3> WT(nz, nP, nQ);
    std::vector<ComplexType> tmp(blk);
    auto T_of = [&](long m, ComplexType *dst) {   // block of W(m)^T: W(m)[Q_rng, P_rng]^T
      if (m >= Ma0 and m < Ma1) get_t(mat(m), grid.P0, nP, grid.Q0, nQ, dst);
      else std::copy_n(tr.data() + tr_off[m], blk, dst);
    };
    for (long i = 0; i < g; ++i) {
      for (long z = 0; z < n1; ++z) {
        T_of(mrow[i] * n1 + z, WT.data() + z * blk);   // ray 1: W(-q, z_i)^T
        T_of(i * n1 + z, tmp.data());                  // ray 2: conj W(q, z_i)^T
        ComplexType *d2 = WT.data() + (n1 + z) * blk;
        for (long x = 0; x < blk; ++x) d2[x] = std::conj(tmp[x]);
      }
      auto W2  = nda::reshape(Pi(i, all, all, all), std::array<long, 2>{nz, blk});
      auto WT2 = nda::reshape(WT, std::array<long, 2>{nz, blk});
      auto w2  = nda::reshape(w(w_group ? i : qs[i], all, all, all), std::array<long, 2>{r, blk});
      for (long c0 = 0; c0 < blk; c0 += bc) {
        const auto cr = nda::range(c0, std::min(blk, c0 + bc));
        auto Yc       = Y(all, nda::range(cr.size()));
        nda::blas::gemm(ComplexType(1.0), U1H, W2(all, cr), ComplexType(0.0), Yc);
        nda::blas::gemm(ComplexType(1.0), U2H, WT2(all, cr), ComplexType(1.0), Yc);
        nda::blas::gemm(ComplexType(1.0), VS, Yc, ComplexType(0.0), w2(all, cr));
      }
    }
  }
  t_fit = secs(tf);
  Timer.stop("W_fit");

  Timer.start("W_wait");
  sync_node();   // nobody reads the buffer any more (it may be kept for the next call)
  shm.release();
  Timer.stop("W_wait");

  {   // profile (max over ranks)
    double x[10] = {t_fwd, t_fx, t_dys, t_bwd, t_bx, t_fit, t_wait, by_intra, by_inter, t_alloc};
    comm.all_reduce_in_place_n(x, 10, boost::mpi3::max<>{});
    double tot = secs(t_all);
    tot        = comm.all_reduce_value(tot, boost::mpi3::max<>{});
    app_log(2, "  gw_line W node path (s, max over ranks): buffer {:.3f} | forward {:.3f} (inter-node {:.3f}) | Dyson {:.3f} | "
               "backward {:.3f} (inter-node {:.3f}) | fit {:.3f} | node barriers {:.3f} | total {:.3f}; per rank {:.3f} GB "
               "intra-node copies, {:.3f} GB sent to other nodes in {} pairwise rounds",
            x[9], x[0], x[1], x[2], x[3], x[4], x[5], x[6], tot, x[7] / 1073741824.0, x[8] / 1073741824.0, 2 * (NN - 1));
  }
  if (W_nodes != nullptr) *W_nodes = std::move(Pi);
  Pi = Arr4_t{};
  return true;
}

/// perf 7.5b: env COQUI_GWLINE_W_REDIST: "node" (default, screened_mirror_node on the host) | "old" (redistribute path)
inline std::string w_redist_mode() {
  char const *v = std::getenv("COQUI_GWLINE_W_REDIST");
  return (v == nullptr or *v == '\0') ? std::string("node") : std::string(v);
}

} // namespace detail

/**
 * W at the bosonic nodes and its residues, for the q group qs (absolute q of the group rows; closed under q -> -q).
 *   Pi      : (g, N_zeta, nP, nQ) block layout in MEM, row i = Pi(qs[i], zeta) at zeta = basis.zeta_nodes (mu-relative).
 *             CONSUMED: its buffer is reused for the block-layout W(q, zeta_i); on return Pi is empty.
 *   Zb      : Coulomb blocks; must hold the full Z(q) of this rank's Dyson slab (dyson_layout_t(np, rank, g, N_zeta, Np)
 *             over the rows of qs) and qminus.
 *   w       : (N_q, r, nP, nQ) residues in MEM; rows qs[i] are written (allocated and zeroed if the shape differs).
 *             w_group (S7e, host-resident residues): w holds only the rows of this group, (g, r, nP, nQ), row i.
 *   W_nodes : optional, receives W(q, zeta_i) blocks (g, N_zeta, nP, nQ) (tests).
 * Collective over mpi.comm.
 */
template <MEMORY_SPACE MEM>
void screened_interaction(memory::array<MEM, ComplexType, 4> &Pi, coulomb_blocks_t<MEM> const &Zb,
                          bosonic_basis_t const &basis, aux_grid_t const &grid,
                          utils::mpi_context_t<boost::mpi3::communicator> &mpi, memory::array<MEM, ComplexType, 4> &w,
                          utils::TimerManager &Timer, memory::array<MEM, ComplexType, 4> *W_nodes,
                          std::vector<long> const &qs, bool w_group) {
  using arr4_t  = memory::array<MEM, ComplexType, 4>;
  using arr2_t  = memory::array<MEM, ComplexType, 2>;
  using arrF_t  = memory::array<MEM, ComplexType, 2, nda::F_layout>;
  using dview_t = math::nda::distributed_array_view<arr4_t, boost::mpi3::communicator>;
  auto all      = nda::range::all;
  auto &comm    = mpi.comm;

  const long g = Pi.extent(0), nz = Pi.extent(1), Np = grid.Np, nP = grid.nP, nQ = grid.nQ, r = basis.rank;
  const long blk = nP * nQ;
  utils::check(nz == basis.zeta_nodes.size(), "gw_line::screened_interaction: Pi has {} nodes, basis has {}", nz,
               basis.zeta_nodes.size());
  utils::check(Pi.extent(2) == nP and Pi.extent(3) == nQ, "gw_line::screened_interaction: Pi block ({}, {}) != grid ({}, {})",
               Pi.extent(2), Pi.extent(3), nP, nQ);
  utils::check(grid.np == comm.size() and grid.rank == comm.rank(), "gw_line::screened_interaction: grid/communicator mismatch");
  utils::check(Zb.grid.P0 == grid.P0 and Zb.grid.nP == nP and Zb.grid.Q0 == grid.Q0 and Zb.grid.nQ == nQ,
               "gw_line::screened_interaction: Coulomb blocks and grid differ");
  utils::check(long(qs.size()) == g, "gw_line::screened_interaction: {} q in the group, Pi has {} rows", long(qs.size()), g);
  utils::check(long(Zb.qminus.size()) == Zb.nq, "gw_line::screened_interaction: the Coulomb blocks carry no qminus map");
  // mirror row of every group row: the group row of -q (the group must be closed under q -> -q)
  std::vector<long> mrow(g, -1);
  for (long i = 0; i < g; ++i) {
    utils::check(qs[i] >= 0 and qs[i] < Zb.nq, "gw_line::screened_interaction: q = {} out of [0, {})", qs[i], Zb.nq);
    for (long j = 0; j < g; ++j)
      if (qs[j] == Zb.qminus[qs[i]]) mrow[i] = j;
    utils::check(mrow[i] >= 0, "gw_line::screened_interaction: the q group does not contain -q = {} of q = {} (the residues of q "
                               "need W(-q), notes section 3.3: use pair-closed q groups, q_groups_t)",
                 Zb.qminus[qs[i]], qs[i]);
  }
  // perf 7.1 (b)/(c): mirror-symmetric nodes -> Dyson on ray 1, ray 2 by conjugation, transposed fit data by exchange
  // (detail::screened_mirror); env COQUI_GWLINE_W_MIRROR = 0 keeps the path below
  if (const long n1 = numerics::line_dlr::mirror_half(basis.zeta_nodes); n1 > 0 and detail::env_long("COQUI_GWLINE_W_MIRROR", 1) != 0) {
    if constexpr (MEM == HOST_MEMORY) {
      if (detail::w_redist_mode() != "old") {
        // the budget: the W-stage transient the q plan reserved for the redistribute path (aux_grid_t::model)
        double old_t = 16.0 * double(g) * double(nz) * double(grid.max_block_size());
        if (dyson_layout_t::valid(comm.size(), g, nz)) {
          const long mb = grid.max_block_size();
          dyson_layout_t lay0(comm.size(), comm.rank(), g, nz, Np);
          w_plan_t plan0(lay0, mb);
          const double kfit = std::min(double(nz), std::max(double(g) * nz / 64.0, double(nz) * std::min(mb, 4096L) / double(mb)));
          const double stg  = std::min(2.0 * 1073741824.0, 2.0 * 16.0 * double(lay0.np_q) * plan0.nzs * mb);
          old_t = std::max(old_t, plan0.transient_bytes(kfit, 1, comm.size() > 1 ? stg : 0.0));
        }
        old_t = comm.all_reduce_value(old_t, boost::mpi3::min<>{});
        if (detail::screened_mirror_node(Pi, Zb, basis, grid, comm, w, Timer, W_nodes, qs, w_group, mrow, n1, old_t)) return;
      }
    }
    const long np_q = utils::find_proc_grid_max_npools(comm.size(), g, 0.2);
    if (comm.size() / np_q <= n1) {
      detail::screened_mirror<MEM>(Pi, Zb, basis, grid, mpi, w, Timer, W_nodes, qs, w_group, mrow, n1);
      return;
    }
    app_log(2, "  gw_line W: {} zeta pools > {} ray-1 nodes: the full-node path is used", comm.size() / np_q, n1);
  }
  dyson_layout_t lay(comm.size(), comm.rank(), g, nz, Np);
  for (auto nm : {"W_redistribute", "W_dyson", "W_fit"}) Timer.add(nm);
  for (long s = 0; s < lay.nq_loc; ++s)
    utils::check(Zb.has_full(qs[lay.q_first + s]),
                 "gw_line::screened_interaction: the Coulomb blocks do not hold the full Z of this rank's Dyson q slab");

  // residues: resident (allocated first, so that the stage's high-water includes them as the plan 6.7 model does)
  // w_group (S7e, host-resident residues): w holds only the rows of this group, (g, r, nP, nQ), row = group row
  const long w_rows = w_group ? g : Zb.nq;
  auto w_row        = [&](long ql) { return w_group ? ql : qs[ql]; };
  if (w.extent(0) != w_rows or w.extent(1) != r or w.extent(2) != nP or w.extent(3) != nQ) {
    w = arr4_t(w_rows, r, nP, nQ);
    nda::tensor::set(ComplexType(0.0), w);
  }
  bosonic_fit_t fit(basis, basis.zeta_nodes);
  const long k = fit.k;
  // the fit buffer Y holds k x bc: the block columns are processed in chunks of bc so that Y stays <= 1/64 of the Pi group
  const long bc = std::clamp(long(double(g) * nz * blk / (64.0 * double(std::max(k, 1L)))), std::min(blk, 4096L), blk);
  arr2_t U1H = memory::to_memory_space<MEM>(fit.U1H), U2H = memory::to_memory_space<MEM>(fit.U2H),
         VS = memory::to_memory_space<MEM>(fit.VS), Y(k, bc);

  // sub-step plan (w_plan_t); device: the zeta sub-slab is capped so that its block + whole-matrix buffers take <= 40% of
  // the free device memory left after the W^T rows (min over ranks: the plan must be identical everywhere)
  long nzs_cap = -1;
  if constexpr (MEM != HOST_MEMORY) {
    const double freeb = double(utils::freemem_device_effective()) * 1048576.0;
    const double wt    = 16.0 * double(lay.np_q) * nz * grid.max_block_size();
    const double per_z = 16.0 * (double(lay.np_q) * grid.max_block_size() + double(Np) * Np / double(lay.np_z));
    nzs_cap            = std::max(1L, long(0.4 * std::max(0.0, freeb - wt) / per_z));
    nzs_cap            = comm.all_reduce_value(nzs_cap, boost::mpi3::min<>{});
  }
  const w_plan_t plan(lay, grid.max_block_size(), nzs_cap);
  plan.log();
  // q pairing per sub-step (identical on every rank): the self-inverse q of a sub-step are fitted in the Dyson pass from
  // the Dyson's own W^T (step 3), the others in the pair pass from the transposed W(-q)
  auto step_has = [&](long s, bool self_inv) {
    for (long p = 0; p < plan.n_act(s); ++p)
      if ((mrow[plan.q_row(p, s)] == plan.q_row(p, s)) == self_inv) return true;
    return false;
  };
  long n_pair = 0;
  for (long i = 0; i < g; ++i) n_pair += (mrow[i] != i) ? 1 : 0;
  app_log(3, "  gw_line W: {} of {} q of the group are not self-inverse (fitted with the transposed W(-q), pair pass)", n_pair, g);
  // device, several ranks: the redistribute staging buffers (memory::pooled_array, two per call) come from a device pool
  // reserved for this stage, so every call reuses the SAME buffers. Without it each call cudaMallocs fresh ones, and with
  // UCX's cuda_ipc transport (NVLink) the peer's IPC-handle cache keeps the freed buffers mapped: measured +2.8 GB per
  // rank at Np 640 and +10 GB at Np 1024 on 2 A100 (S7d). Skipped when a pool is already active (e.g. an SCF guard).
  [[maybe_unused]] std::optional<utils::device_pool_guard> stage_pool;
  if constexpr (MEM != HOST_MEMORY) {
    if (comm.size() > 1 and utils::device_pool_capacity() == 0) {
      const double stg = std::min(double(math::nda::detail::redistribute_chunk_bytes()),
                                  16.0 * double(lay.np_q) * plan.nzs * grid.max_block_size());
      stage_pool.emplace(std::size_t(2.0 * stg) + (std::size_t(64) << 20), "gw_line W stage");
    }
  }
  const long nzl_max = (plan.nzs + lay.np_z - 1) / lay.np_z;

  // Dyson workspace
  arr2_t Id(Np, Np), M(Np, Np);
  {
    nda::array<ComplexType, 2> Ih(Np, Np);
    Ih() = ComplexType(0.0);
    for (long P = 0; P < Np; ++P) Ih(P, P) = ComplexType(1.0);
    Id = memory::to_memory_space<MEM>(Ih);
  }
  arrF_t ZF(Np, Np), XF(Np, Np);
  memory::array<MEM, int, 1> ipiv(Np);
  memory::array<MEM, ComplexType, 1> lwork;   // getrf workspace (device: sized once by cusolver's bufferSize, reused)
  // device: batched LU (cuBLAS getrf/getrsBatched through nda's 3D getrf/getrs) in sub-batches of nbat nodes, or the
  // per-(q, zeta) cuSOLVER loop. COQUI_GWLINE_DYSON_BATCHED = 1 / 0 forces either; default: batched for Np <= 1024
  // (measured on A100: 8.5x faster than the loop at Np = 128, 4.4x at Np = 640; not measured beyond).
  [[maybe_unused]] bool batched = false;
  long nbat                     = std::max(1L, std::min(nzl_max, 16L));   // host: matrices per in-place transpose batch
  detail::scratch_t<MEM> sM, sTb, sD, sWT;
  [[maybe_unused]] memory::array<MEM, int, 2> ipiv_b;
  if constexpr (MEM != HOST_MEMORY) {
    batched = detail::env_long("COQUI_GWLINE_DYSON_BATCHED", Np <= 1024 ? 1 : 0) != 0;
    const double mat = double(Np) * Np * 16.0, freeb = double(utils::freemem_device_effective()) * 1048576.0;
    nbat = std::max(1L, std::min({nzl_max, dyson_nbat_max(), long(0.25 * freeb / mat)}));
    if (batched) ipiv_b = memory::array<MEM, int, 2>(nbat, Np);
    app_log(3, "  gw_line::screened_interaction: Dyson on the device {} ({} matrices per batch)",
            batched ? "batched (cuBLAS getrf/getrsBatched)" : "per matrix (cuSOLVER)", nbat);
  }

  // fit of group row ql (sub-step slot p): w(q) = VS (U1H W(q) + U2H W(-q)^T), W(q) = the Pi row, W(-q)^T = WT(p)
  auto fit_row = [&](long ql, long p, auto const &WT) {
    auto W2  = nda::reshape(Pi(ql, all, all, all), std::array<long, 2>{nz, blk});
    auto WT2 = nda::reshape(WT(p, all, all, all), std::array<long, 2>{nz, blk});
    auto w2  = nda::reshape(w(w_row(ql), all, all, all), std::array<long, 2>{r, blk});
    for (long c0 = 0; c0 < blk; c0 += bc) {
      const auto cr = nda::range(c0, std::min(blk, c0 + bc));
      auto Yc       = Y(all, nda::range(cr.size()));
      nda::blas::gemm(ComplexType(1.0), U1H, W2(all, cr), ComplexType(0.0), Yc);
      nda::blas::gemm(ComplexType(1.0), U2H, WT2(all, cr), ComplexType(1.0), Yc);
      nda::blas::gemm(ComplexType(1.0), VS, Yc, ComplexType(0.0), w2(all, cr));
    }
  };

  const std::array<long, 4> bgrid = {1, 1, grid.np_P, grid.np_Q}, ones = {1, 1, 1, 1};
  for (long s = 0; s < plan.nsub_q; ++s) {
    const long na    = plan.n_act(s);
    const bool act   = s < lay.nq_loc;                // this rank's q pool solves a q in this sub-step
    const long iq_a  = qs[lay.q_first + (act ? s : 0)];   // ... namely this one (absolute)
    const bool wt_now = step_has(s, true);            // the sub-step has self-inverse q: their W^T rows now (step 3)
    auto WT          = sWT.template view<4>({na, nz, nP, nQ});
    if (act) {
      Timer.start("W_dyson");
      nda::matrix<ComplexType, nda::F_layout> zf_h(Zb.full(iq_a));   // layout change on the host
      ZF = zf_h;
      Timer.stop("W_dyson");
    }
    for (long za = 0; za < nz; za += plan.nzs) {
      const long nzs           = std::min(plan.nzs, nz - za);
      const auto zrng          = nda::range(za, za + nzs);
      const auto [zf, nzl]     = plan.z_chunk(nzs);
      const std::array<long, 4> gsh = {na, nzs, Np, Np};
      const bool solve         = act and nzl > 0;

      // 1. Pi rows of the sub-step -> block buffer Tb -> whole-matrix buffer D
      Timer.start("W_redistribute");
      auto Tb = sTb.template view<4>({na, nzs, nP, nQ});
      for (long p = 0; p < na; ++p) Tb(p, all, all, all) = Pi(plan.q_row(p, s), zrng, all, all);
      dview_t dTb(std::addressof(comm), bgrid, gsh, {0, 0, grid.P0, grid.Q0}, ones, Tb);
      auto D4 = sD.template view<4>({act ? 1L : 0L, nzl, Np, Np});
      dview_t dD(std::addressof(comm), lay.pgrid(), gsh, {act ? lay.ip_q : na, zf, 0, 0}, ones, D4);
      math::nda::redistribute(dTb, dD);
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        device_mem_probe();
      }
      Timer.stop("W_redistribute");

      // 2. Dyson IN PLACE, five-step order of the imaginary-axis code: A = Z Pi; A <- I - A (one gemm onto the identity);
      //    LU(A); solve A W' = Z; W = W' - Z. W' comes out of getrs in Fortran layout, i.e. its buffer IS W'^T in C order:
      //    D(z) <- W'^T - Z^T = W^T.
      Timer.start("W_dyson");
      if (solve) {
        auto Dv = D4(0, all, all, all);
        bool done = false;
        if constexpr (MEM != HOST_MEMORY) {
          if (batched) {
            detail::dyson_batched_device(Dv, ZF, Id, nbat, sM, ipiv_b, iq_a, za + zf);
            done = true;
          }
        }
        if (not done) {
          auto ZT = nda::transpose(ZF);   // C-ordered view of Z^T
          for (long izl = 0; izl < nzl; ++izl) {
            auto Pv = Dv(izl, all, all);
            M = Id;
            nda::blas::gemm(ComplexType(-1.0), ZF, Pv, ComplexType(1.0), M);   // M = I - Z Pi
            int info = nda::lapack::getrf(M, ipiv, lwork);
            utils::check(info == 0, "gw_line::screened_interaction: getrf of I - Z Pi failed (q={}, node={}, info={})", iq_a,
                         za + zf + izl, info);
            XF   = ZF;
            info = nda::lapack::getrs(M, XF, ipiv);   // XF = (I - Z Pi)^{-1} Z
            utils::check(info == 0, "gw_line::screened_interaction: getrs failed (q={}, node={}, info={})", iq_a,
                         za + zf + izl, info);
            Pv = nda::transpose(XF);   // W'^T (contiguous copy; Pi(q, zeta) is no longer needed)
            if constexpr (MEM == HOST_MEMORY) Pv -= ZT;   // W^T = W'^T - Z^T
            else nda::tensor::add(ComplexType(-1.0), ZT, ComplexType(1.0), Pv);
          }
        }
      }
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        device_mem_probe();
      }
      Timer.stop("W_dyson");

      // 3. (sub-steps with self-inverse q) W^T -> block layout -> the WT rows of the sub-step (block (I,J) of W^T =
      //    (W_JI)^T, the fit's mirror block for q = -q)
      if (wt_now) {
        Timer.start("W_redistribute");
        math::nda::redistribute(dD, dTb);
        for (long p = 0; p < na; ++p) WT(p, zrng, all, all) = Tb(p, all, all, all);
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("W_redistribute");
      }

      // 4. W = (W^T)^T in place, -> block layout -> the Pi rows (Pi's buffer becomes the block-layout W)
      Timer.start("W_dyson");
      if (solve) {
        auto Dv = D4(0, all, all, all);
        detail::transpose_in_place<MEM>(Dv, nbat, sM);
      }
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_dyson");
      Timer.start("W_redistribute");
      math::nda::redistribute(dD, dTb);
      for (long p = 0; p < na; ++p) Pi(plan.q_row(p, s), zrng, all, all) = Tb(p, all, all, all);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_redistribute");
    }

    // 5. fit of the sub-step's self-inverse q, all pairs at once: w(q) = VS (U1H W(q) + U2H W^T(q))
    if (wt_now) {
      Timer.start("W_fit");
      for (long p = 0; p < na; ++p) {
        const long ql = plan.q_row(p, s);
        if (mrow[ql] == ql) fit_row(ql, p, WT);
      }
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        device_mem_probe();
      }
      Timer.stop("W_fit");
    }
  }

  // PAIR PASS (q != -q): the W of every q of the group is now in the Pi rows. Per sub-step: the rows of the partners -q
  // -> whole matrices -> transposed in place -> block layout: the block of W(-q)^T, then the fit of q
  for (long s = 0; s < plan.nsub_q; ++s) {
    if (not step_has(s, false)) continue;
    const long na  = plan.n_act(s);
    const bool act = s < lay.nq_loc;
    auto WT        = sWT.template view<4>({na, nz, nP, nQ});
    for (long za = 0; za < nz; za += plan.nzs) {
      const long nzs           = std::min(plan.nzs, nz - za);
      const auto zrng          = nda::range(za, za + nzs);
      const auto [zf, nzl]     = plan.z_chunk(nzs);
      const std::array<long, 4> gsh = {na, nzs, Np, Np};
      Timer.start("W_redistribute");
      auto Tb = sTb.template view<4>({na, nzs, nP, nQ});
      for (long p = 0; p < na; ++p) Tb(p, all, all, all) = Pi(mrow[plan.q_row(p, s)], zrng, all, all);
      dview_t dTb(std::addressof(comm), bgrid, gsh, {0, 0, grid.P0, grid.Q0}, ones, Tb);
      auto D4 = sD.template view<4>({act ? 1L : 0L, nzl, Np, Np});
      dview_t dD(std::addressof(comm), lay.pgrid(), gsh, {act ? lay.ip_q : na, zf, 0, 0}, ones, D4);
      math::nda::redistribute(dTb, dD);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_redistribute");
      Timer.start("W_dyson");
      if (act and nzl > 0) {
        auto Dv = D4(0, all, all, all);
        detail::transpose_in_place<MEM>(Dv, nbat, sM);
      }
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_dyson");
      Timer.start("W_redistribute");
      math::nda::redistribute(dD, dTb);
      for (long p = 0; p < na; ++p) WT(p, zrng, all, all) = Tb(p, all, all, all);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_redistribute");
    }
    Timer.start("W_fit");
    for (long p = 0; p < na; ++p) {
      const long ql = plan.q_row(p, s);
      if (mrow[ql] != ql) fit_row(ql, p, WT);
    }
    if constexpr (MEM != HOST_MEMORY) {
      utils::device_sync();
      device_mem_probe();
    }
    Timer.stop("W_fit");
  }

  if (W_nodes != nullptr) *W_nodes = std::move(Pi);   // Pi's buffer holds W(q, zeta_i) in the block layout
  Pi = arr4_t{};                                       // consumed
}

/// The q group [q0, q0 + g) with g = Pi.extent(0) (must be closed under q -> -q; q0 = 0, g = N_q: all q).
template <MEMORY_SPACE MEM>
void screened_interaction(memory::array<MEM, ComplexType, 4> &Pi, coulomb_blocks_t<MEM> const &Zb,
                          bosonic_basis_t const &basis, aux_grid_t const &grid,
                          utils::mpi_context_t<boost::mpi3::communicator> &mpi, memory::array<MEM, ComplexType, 4> &w,
                          utils::TimerManager &Timer, memory::array<MEM, ComplexType, 4> *W_nodes = nullptr, long q0 = 0,
                          bool w_group = false) {
  std::vector<long> qs(Pi.extent(0));
  for (long i = 0; i < long(qs.size()); ++i) qs[i] = q0 + i;
  screened_interaction<MEM>(Pi, Zb, basis, grid, mpi, w, Timer, W_nodes, qs, w_group);
}

namespace detail {
/// out[n, nP, nQ] = C[n, r] . w(iq)[r, nP, nQ]   (C built on the host)
template <MEMORY_SPACE MEM>
void pole_contract(memory::array<MEM, ComplexType, 4> const &w, long iq, nda::array<ComplexType, 2> const &C,
                   memory::array_view<MEM, ComplexType, 3> out) {
  const long n = C.extent(0), r = w.extent(1), blk = w.extent(2) * w.extent(3);
  utils::check(C.extent(1) == r, "gw_line::pole_contract: {} coefficients vs {} poles", C.extent(1), r);
  utils::check(iq >= 0 and iq < w.extent(0), "gw_line::pole_contract: q={} out of range", iq);
  utils::check(out.extent(0) == n and out.extent(1) == w.extent(2) and out.extent(2) == w.extent(3),
               "gw_line::pole_contract: out shape mismatch");
  memory::array<MEM, ComplexType, 2> Cm = memory::to_memory_space<MEM>(C);
  auto w2 = nda::reshape(w(iq, nda::range::all, nda::range::all, nda::range::all), std::array<long, 2>{r, blk});
  auto o2 = nda::reshape(out, std::array<long, 2>{n, blk});
  nda::blas::gemm(ComplexType(1.0), Cm, w2, ComplexType(0.0), o2);
}
/// out[n, nP, nQ] = Cm[n, r] . w(iq)[r, nP, nQ]   (Cm already in MEM: precomputed once per time chunk by self_energy)
template <MEMORY_SPACE MEM>
void pole_contract_m(memory::array<MEM, ComplexType, 4> const &w, long iq, memory::array_view<MEM, ComplexType, 2> Cm,
                     memory::array_view<MEM, ComplexType, 3> out) {
  const long n = Cm.extent(0), r = w.extent(1), blk = w.extent(2) * w.extent(3);
  utils::check(Cm.extent(1) == r, "gw_line::pole_contract: {} coefficients vs {} poles", Cm.extent(1), r);
  utils::check(iq >= 0 and iq < w.extent(0), "gw_line::pole_contract: q={} out of range", iq);
  utils::check(out.extent(0) == n and out.extent(1) == w.extent(2) and out.extent(2) == w.extent(3),
               "gw_line::pole_contract: out shape mismatch");
  auto w2 = nda::reshape(w(iq, nda::range::all, nda::range::all, nda::range::all), std::array<long, 2>{r, blk});
  auto o2 = nda::reshape(out, std::array<long, 2>{n, blk});
  nda::blas::gemm(ComplexType(1.0), Cm, w2, ComplexType(0.0), o2);
}
inline void check_orientation(sector_t s, bool transposed, char const *who) {
  utils::check((s == sector_t::particle and not transposed) or (s == sector_t::hole and transposed),
               "gw_line::{}: only W^> (particle, plain orientation) and W^<^T (hole, transposed orientation) are provided "
               "from the stored residues w (see screened.hpp)",
               who);
}
} // namespace detail

/**
 * Residue exponentials on complex times t (block layout, MEM), out: (nt, nP, nQ), for q = iq with -q = iq_minus
 * (mf::MF::qminus()(iq); rows of w are absolute q):
 *   particle, plain       : W^>(q,t)   =  sum_j w_j(q) e^{-i nu_j t}
 *   hole,     transposed  : W^<(q,t)^T = -sum_j w_j(-q) e^{+i nu_j t}    (bosonic_basis_t::time_exponentials(t, hole))
 */
template <MEMORY_SPACE MEM>
void w_time(memory::array<MEM, ComplexType, 4> const &w, bosonic_basis_t const &basis, long iq, long iq_minus,
            nda::array<ComplexType, 1> const &t, sector_t s, bool transposed, memory::array_view<MEM, ComplexType, 3> out) {
  detail::check_orientation(s, transposed, "w_time");
  detail::pole_contract<MEM>(w, s == sector_t::particle ? iq : iq_minus, basis.time_exponentials(t, s), out);
}

/**
 * Pole sums at mu-relative zeta (block layout, MEM), out: (nz, nP, nQ), for q = iq with -q = iq_minus:
 *   particle, plain       : W^>(q,zeta)   =  sum_j w_j(q) / (zeta - nu_j)
 *   hole,     transposed  : W^<(q,zeta)^T = -sum_j w_j(-q) / (zeta + nu_j)
 */
template <MEMORY_SPACE MEM>
void eval_poles(memory::array<MEM, ComplexType, 4> const &w, bosonic_basis_t const &basis, long iq, long iq_minus,
                nda::array<ComplexType, 1> const &zeta, sector_t s, bool transposed,
                memory::array_view<MEM, ComplexType, 3> out) {
  detail::check_orientation(s, transposed, "eval_poles");
  auto [Km, Kp] = basis.kernels(zeta);
  if (s == sector_t::particle) detail::pole_contract<MEM>(w, iq, Km, out);
  else {
    Kp *= ComplexType(-1.0);
    detail::pole_contract<MEM>(w, iq_minus, Kp, out);
  }
}

#define GW_LINE_SCREENED_EXTERN(MEM)                                                                                     \
  extern template struct coulomb_blocks_t<MEM>;                                                                          \
  extern template void screened_interaction<MEM>(memory::array<MEM, ComplexType, 4> &, coulomb_blocks_t<MEM> const &,     \
                                                 bosonic_basis_t const &, aux_grid_t const &,                             \
                                                 utils::mpi_context_t<boost::mpi3::communicator> &,                       \
                                                 memory::array<MEM, ComplexType, 4> &, utils::TimerManager &,             \
                                                 memory::array<MEM, ComplexType, 4> *, std::vector<long> const &, bool);  \
  extern template void w_time<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long, long,       \
                                   nda::array<ComplexType, 1> const &, sector_t, bool,                                    \
                                   memory::array_view<MEM, ComplexType, 3>);                                              \
  extern template void eval_poles<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long, long,   \
                                       nda::array<ComplexType, 1> const &, sector_t, bool,                                \
                                       memory::array_view<MEM, ComplexType, 3>);

GW_LINE_SCREENED_EXTERN(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SCREENED_EXTERN(DEVICE_MEMORY)
#endif
#undef GW_LINE_SCREENED_EXTERN

} // namespace methods::gw_line

#endif
