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
 *   W(q, zeta_i) = ([1 - Z(q) Pi(q, zeta_i)]^{-1} - 1) Z(q)                     at the bosonic line nodes zeta_i,
 *   W_PQ(zeta)   = sum_j [ w_j,PQ / (zeta - nu_j) - w_j,QP / (zeta + nu_j) ]     (symmetric real-pole fit, all pairs).
 *
 * Distribution (plan 6.2 / 6.3(c)). Every Np x Np object lives as the (P_rng, Q_rng) block of the ONE 2D aux grid
 * (aux_grid_t). The Dyson step needs whole matrices: Pi {g, N_zeta, Np, Np} is wrapped as a darray with grid
 * {1, 1, np_P, np_Q} (the block layout; its chunks are exactly aux_grid_t's) and redistributed to the "whole-matrix"
 * layout {np_q, np_z, 1, 1} (dyson_layout_t: contiguous (q, zeta) slabs per rank, full Np x Np). Per local (q, zeta) the
 * five steps of the imaginary-axis code are done on full matrices; besides W the transposed copy W^T is formed locally
 * (a plain transpose), and BOTH are redistributed back to the block layout. Block (I,J) of the redistributed W^T is
 * (W_JI)^T, i.e. exactly the mirror block the coupled (PQ, QP) fit needs: there is no inter-rank transpose logic anywhere.
 *
 * The fit (Eq. bfit) for all pairs at once, from the truncated SVD of the stacked 2 N_zeta x 2 r kernel
 * [[K^-, -K^+], [-K^+, K^-]] = U S V^dagger (same rcond = DBL_EPSILON max(2 N_zeta, 2 r) as bosonic_basis_t::fit /
 * numpy lstsq; bosonic_fit_t, built once on the host):
 *   w_IJ = VS (U1H W_IJ + U2H (W^T)_IJ)          (three gemms per q: [k x N_zeta] . [N_zeta x block] twice, [r x k] . [k x block]).
 * The explicit pinv blocks A11 = VS U1H, A12 = VS U2H are mathematically the same but numerically unstable (cond ~1e13).
 * The residues themselves are determined only up to the near-threshold singular directions (gesvd here vs gelss in
 * bosonic_basis_t::fit differ by ~1e-5 in w); the pole functions they define agree to ~1e-13 (test [V2](b)).
 *
 * Only w is stored (not w^T). The hole-sector interaction W^<(q,t)_PQ = -sum_j w_j,QP e^{+i nu_j t} needs the mirror
 * block, so it is consumed (S5) in TRANSPOSED orientation together with the transposed propagator
 * (gtilde_form_t::transposed), exactly as the polarization already does:
 *   Sigma~^<(k,t)^T = -(1/N_k) sum_q G~^<(k-q,t)^T o W^<(q,t)^T,    W^<(q,t)^T = -sum_j w_j(q) e^{+i nu_j t},
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

#include <array>
#include <limits>
#include <optional>
#include <string>

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
#include "utilities/mpi_context.h"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/Timer.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/proc_grid.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::sector_t;

/**
 * Whole-matrix layout of the Dyson step: 4D darray {g, N_zeta, Np, Np} with grid {np_q, np_z, 1, 1}. np_q is the
 * largest number of q pools (<= g, dividing np, load imbalance <= 20%: find_proc_grid_max_npools), so that each rank's
 * full-Z footprint (its q subset x Np^2) is minimal; np_z = np / np_q. Rank -> coordinates and the chunks follow
 * math::nda::make_distributed_array for a C-layout darray with block size 1 (ip_z = rank % np_z, ip_q = rank / np_z).
 * q indices are relative to the group (q0 of screened_interaction).
 */
struct dyson_layout_t {
  long np = 1, rank = 0, g = 0, nz = 0, Np = 0;
  long np_q = 1, np_z = 1, ip_q = 0, ip_z = 0;
  long q_first = 0, nq_loc = 0;   ///< local q slab [q_first, q_first + nq_loc) (group-relative)
  long z_first = 0, nz_loc = 0;   ///< local zeta slab

  dyson_layout_t() = default;
  dyson_layout_t(long np_, long rank_, long g_, long nz_, long Np_) : np(np_), rank(rank_), g(g_), nz(nz_), Np(Np_) {
    utils::check(np > 0 and rank >= 0 and rank < np and g > 0 and nz > 0 and Np > 0,
                 "dyson_layout_t: invalid np={} rank={} g={} nz={} Np={}", np, rank, g, nz, Np);
    np_q = utils::find_proc_grid_max_npools(np, g, 0.2);
    np_z = np / np_q;
    utils::check(np_q * np_z == np, "dyson_layout_t: np_q*np_z != np");
    utils::check(np_q <= g and np_z <= nz, "dyson_layout_t: too many ranks ({} x {}) for g={} nz={}", np_q, np_z, g, nz);
    ip_z = rank % np_z;
    ip_q = rank / np_z;
    auto [q0, q1] = itertools::chunk_range(0, g, np_q, ip_q);
    auto [z0, z1] = itertools::chunk_range(0, nz, np_z, ip_z);
    q_first = q0; nq_loc = q1 - q0;
    z_first = z0; nz_loc = z1 - z0;
  }
  template <typename comm_t>
  dyson_layout_t(utils::mpi_context_t<comm_t> const &mpi, long g_, long nz_, long Np_)
     : dyson_layout_t(long(mpi.comm.size()), long(mpi.comm.rank()), g_, nz_, Np_) {}

  std::array<long, 4> pgrid() const { return {np_q, np_z, 1, 1}; }
  nda::range q_rng() const { return nda::range(q_first, q_first + nq_loc); }
  nda::range z_rng() const { return nda::range(z_first, z_first + nz_loc); }
  /// largest slab over the grid (elements)
  long max_slab() const { return ((g + np_q - 1) / np_q) * ((nz + np_z - 1) / np_z) * Np * Np; }

  void log() const {
    app_log(2, "  gw_line Dyson layout: {} ranks -> (q, zeta) pools = ({} x {}) over (g, N_zeta) = ({}, {}); slab <= {} "
               "matrices ({:.3f} GB), full Z for <= {} q per rank",
            np, np_q, np_z, g, nz, max_slab() / (Np * Np), double(max_slab()) * 16.0 / 1024.0 / 1024.0 / 1024.0,
            (g + np_q - 1) / np_q);
  }
};

/**
 * Coulomb matrices of all q in the block layout (resident in MEM), Z[iq] = Z(q)[P_rng, Q_rng], plus the FULL Z(q) (host)
 * for the q's this rank Dyson-solves (q_full range). Acquired by ONE lockstep loop over all q: thc.Z(iq) is collective
 * (every rank must call it the same number of times, in the same order). Z(Gamma) carries no G=0 term (ignore_g0).
 */
template <MEMORY_SPACE MEM>
struct coulomb_blocks_t {
  aux_grid_t grid;
  long nq = 0, Np = 0;
  memory::array<MEM, ComplexType, 3> Z;     ///< (nq, nP, nQ)
  long q_full0 = 0;                          ///< first (absolute) q of Z_full
  nda::array<ComplexType, 3> Z_full;         ///< (nq_full, Np, Np) host

  coulomb_blocks_t() = default;

  /// q_full: absolute q range whose full matrices this rank keeps (dyson_layout_t::q_rng() + q0 of the group).
  coulomb_blocks_t(methods::thc_reader_t const &thc, aux_grid_t const &grid_, nda::range q_full,
                   utils::TimerManager &Timer)
     : grid(grid_) {
    utils::check(thc.nkpts() == thc.nkpts_ibz() and thc.nqpts() == thc.nqpts_ibz(),
                 "gw_line::coulomb_blocks_t: requires a mesh without symmetry reduction (nk {} nk_ibz {} nq {} nq_ibz {})",
                 thc.nkpts(), thc.nkpts_ibz(), thc.nqpts(), thc.nqpts_ibz());
    utils::check(thc.ns() == 1 and thc.npol() == 1, "gw_line::coulomb_blocks_t: spin-restricted collinear only (ns={}, npol={})",
                 thc.ns(), thc.npol());
    utils::check(thc.Np() == grid.Np, "gw_line::coulomb_blocks_t: grid Np {} != thc Np {}", grid.Np, thc.Np());
    nq = thc.nqpts();
    Np = thc.Np();
    utils::check(q_full.first() >= 0 and q_full.last() <= nq, "gw_line::coulomb_blocks_t: q_full range out of [0, {})", nq);
    q_full0 = q_full.first();
    Timer.add("Z_gather");
    Timer.start("Z_gather");
    nda::array<ComplexType, 3> zb(nq, grid.nP, grid.nQ);
    Z_full = nda::array<ComplexType, 3>(q_full.size(), Np, Np);
    for (long iq = 0; iq < nq; ++iq) {   // LOCKSTEP: identical call sequence on every rank
      auto Zq = thc.Z(int(iq));
      zb(iq, nda::range::all, nda::range::all) = Zq(grid.P_rng(), grid.Q_rng());
      if (iq >= q_full.first() and iq < q_full.last()) Z_full(iq - q_full0, nda::range::all, nda::range::all) = Zq;
    }
    Z = memory::to_memory_space<MEM>(zb);
    Timer.stop("Z_gather");
    app_log(2, "  gw_line Coulomb blocks: Z blocks {} x {} x {} in {} ({:.4f} GB per rank), full Z(q) for {} q on the host "
               "({:.4f} GB per rank)",
            nq, grid.nP, grid.nQ, MEM == HOST_MEMORY ? "host" : "device", double(nq * grid.max_block_size()) * 16.0 / 1073741824.0,
            q_full.size(), double(q_full.size() * Np * Np) * 16.0 / 1073741824.0);
  }

  bool has_full(long iq) const { return iq >= q_full0 and iq < q_full0 + Z_full.extent(0); }
  auto full(long iq) const {
    utils::check(has_full(iq), "gw_line::coulomb_blocks_t: full Z(q={}) not kept on this rank (range [{}, {}))", iq, q_full0,
                 q_full0 + Z_full.extent(0));
    return Z_full(iq - q_full0, nda::range::all, nda::range::all);
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

/**
 * W at the bosonic nodes and its residues, for the q group [q0, q0 + g).
 *   Pi      : (g, N_zeta, nP, nQ) block layout in MEM, Pi(q, zeta_i) at zeta_i = basis.zeta_nodes (mu-relative).
 *             CONSUMED: its buffer is reused for the block-layout W(q, zeta_i); on return Pi is empty.
 *   Zb      : Coulomb blocks; must hold the full Z(q) of this rank's Dyson slab (dyson_layout_t(np, rank, g, N_zeta, Np)).
 *   w       : (N_q, r, nP, nQ) residues in MEM; rows q0..q0+g-1 are written (allocated and zeroed if the shape differs).
 *   W_nodes : optional, receives W(q, zeta_i) blocks (g, N_zeta, nP, nQ) (tests).
 * Collective over mpi.comm.
 */
template <MEMORY_SPACE MEM>
void screened_interaction(memory::array<MEM, ComplexType, 4> &Pi, coulomb_blocks_t<MEM> const &Zb,
                          bosonic_basis_t const &basis, aux_grid_t const &grid,
                          utils::mpi_context_t<boost::mpi3::communicator> &mpi, memory::array<MEM, ComplexType, 4> &w,
                          utils::TimerManager &Timer, memory::array<MEM, ComplexType, 4> *W_nodes = nullptr, long q0 = 0) {
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  using arrF_t = memory::array<MEM, ComplexType, 2, nda::F_layout>;
  auto all     = nda::range::all;
  auto &comm   = mpi.comm;

  const long g = Pi.extent(0), nz = Pi.extent(1), Np = grid.Np, nP = grid.nP, nQ = grid.nQ, r = basis.rank;
  const long blk = nP * nQ;
  utils::check(nz == basis.zeta_nodes.size(), "gw_line::screened_interaction: Pi has {} nodes, basis has {}", nz,
               basis.zeta_nodes.size());
  utils::check(Pi.extent(2) == nP and Pi.extent(3) == nQ, "gw_line::screened_interaction: Pi block ({}, {}) != grid ({}, {})",
               Pi.extent(2), Pi.extent(3), nP, nQ);
  utils::check(grid.np == comm.size() and grid.rank == comm.rank(), "gw_line::screened_interaction: grid/communicator mismatch");
  utils::check(Zb.grid.P0 == grid.P0 and Zb.grid.nP == nP and Zb.grid.Q0 == grid.Q0 and Zb.grid.nQ == nQ,
               "gw_line::screened_interaction: Coulomb blocks and grid differ");
  utils::check(q0 >= 0 and q0 + g <= Zb.nq, "gw_line::screened_interaction: q group [{}, {}) out of [0, {})", q0, q0 + g, Zb.nq);
  dyson_layout_t lay(comm.size(), comm.rank(), g, nz, Np);
  for (auto nm : {"W_redistribute", "W_dyson", "W_fit"}) Timer.add(nm);

  const std::array<long, 4> gshape = {g, nz, Np, Np}, bgrid = {1, 1, grid.np_P, grid.np_Q}, ones = {1, 1, 1, 1};

  // 1. block layout -> whole-matrix layout
  Timer.start("W_redistribute");
  auto dPi = math::nda::make_distributed_array<arr4_t>(comm, bgrid, gshape, ones, std::move(Pi));
  utils::check(dPi.local_range(2) == grid.P_rng() and dPi.local_range(3) == grid.Q_rng(),
               "gw_line::screened_interaction: darray block layout does not match aux_grid_t");
  auto dD = math::nda::make_distributed_array<arr4_t>(comm, lay.pgrid(), gshape);
  utils::check(dD.local_range(0) == lay.q_rng() and dD.local_range(1) == lay.z_rng() and dD.local_shape()[2] == Np and
                   dD.local_shape()[3] == Np,
               "gw_line::screened_interaction: darray Dyson layout does not match dyson_layout_t");
  math::nda::redistribute(dPi, dD);
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("W_redistribute");

  // 2. Dyson per local (q, zeta), five-step order of the imaginary-axis code:
  //    A = Z Pi; A <- I - A (one gemm onto the identity); LU(A); solve A W' = Z; W = W' - Z.
  //    W' comes out of getrs in Fortran layout, i.e. its buffer IS W'^T in C order: W^T is a plain copy, W a transpose.
  Timer.start("W_dyson");
  auto dDT = math::nda::make_distributed_array<arr4_t>(comm, lay.pgrid(), gshape);
  {
    auto Dl  = dD.local();
    auto DTl = dDT.local();
    arr2_t Id(Np, Np), M(Np, Np);
    {
      nda::array<ComplexType, 2> Ih(Np, Np);
      Ih() = ComplexType(0.0);
      for (long P = 0; P < Np; ++P) Ih(P, P) = ComplexType(1.0);
      Id = memory::to_memory_space<MEM>(Ih);
    }
    arrF_t ZF(Np, Np), XF(Np, Np);
    memory::array<MEM, int, 1> ipiv(Np);
    for (long iql = 0; iql < lay.nq_loc; ++iql) {
      const long iq = q0 + lay.q_first + iql;
      {
        nda::matrix<ComplexType, nda::F_layout> zf_h(Zb.full(iq));   // layout change on the host
        ZF = zf_h;
      }
      auto ZT = nda::transpose(ZF);   // C-ordered view of Z^T
      for (long izl = 0; izl < lay.nz_loc; ++izl) {
        auto Pv  = Dl(iql, izl, all, all);
        auto WTv = DTl(iql, izl, all, all);
        M = Id;
        nda::blas::gemm(ComplexType(-1.0), ZF, Pv, ComplexType(1.0), M);   // M = I - Z Pi
        int info = nda::lapack::getrf(M, ipiv);
        utils::check(info == 0, "gw_line::screened_interaction: getrf of I - Z Pi failed (q={}, node={}, info={})", iq,
                     lay.z_first + izl, info);
        XF   = ZF;
        info = nda::lapack::getrs(M, XF, ipiv);   // XF = (I - Z Pi)^{-1} Z
        utils::check(info == 0, "gw_line::screened_interaction: getrs failed (q={}, node={}, info={})", iq, lay.z_first + izl,
                     info);
        WTv = nda::transpose(XF);   // W'^T (contiguous copy)
        if constexpr (MEM == HOST_MEMORY) {
          WTv -= ZT;                    // W^T = W'^T - Z^T
          Pv = nda::transpose(WTv);     // W  (Pi(q, zeta) is no longer needed: overwritten)
        } else {
          nda::tensor::add(ComplexType(-1.0), ZT, ComplexType(1.0), WTv);
          nda::tensor::add(ComplexType(1.0), WTv, "ab", ComplexType(0.0), Pv, "ba");
        }
      }
    }
  }
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("W_dyson");

  // 3. back to the block layout: W into Pi's old buffer, W^T into a new block array (block (I,J) of W^T = (W_JI)^T)
  Timer.start("W_redistribute");
  auto dW = math::nda::make_distributed_array<arr4_t>(comm, bgrid, gshape, ones, std::move(dPi.local_()));
  math::nda::redistribute(dD, dW);
  dD.reset();
  auto dWT = math::nda::make_distributed_array<arr4_t>(comm, bgrid, gshape);
  utils::check(dWT.local_range(2) == grid.P_rng() and dWT.local_range(3) == grid.Q_rng(),
               "gw_line::screened_interaction: W^T block layout does not match aux_grid_t");
  math::nda::redistribute(dDT, dWT);
  dDT.reset();
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("W_redistribute");

  // 4. symmetric fit, all pairs at once: w(q) = VS (U1H W(q) + U2H W^T(q))
  Timer.start("W_fit");
  bosonic_fit_t fit(basis, basis.zeta_nodes);
  const long k = fit.k;
  arr2_t U1H = memory::to_memory_space<MEM>(fit.U1H), U2H = memory::to_memory_space<MEM>(fit.U2H),
         VS = memory::to_memory_space<MEM>(fit.VS), Y(k, blk);
  if (w.extent(0) != Zb.nq or w.extent(1) != r or w.extent(2) != nP or w.extent(3) != nQ) {
    w = arr4_t(Zb.nq, r, nP, nQ);
    nda::tensor::set(ComplexType(0.0), w);
  }
  {
    auto Wl  = dW.local();
    auto WTl = dWT.local();
    for (long iq = 0; iq < g; ++iq) {
      auto W2  = nda::reshape(Wl(iq, all, all, all), std::array<long, 2>{nz, blk});
      auto WT2 = nda::reshape(WTl(iq, all, all, all), std::array<long, 2>{nz, blk});
      auto w2  = nda::reshape(w(q0 + iq, all, all, all), std::array<long, 2>{r, blk});
      nda::blas::gemm(ComplexType(1.0), U1H, W2, ComplexType(0.0), Y);
      nda::blas::gemm(ComplexType(1.0), U2H, WT2, ComplexType(1.0), Y);
      nda::blas::gemm(ComplexType(1.0), VS, Y, ComplexType(0.0), w2);
    }
  }
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("W_fit");

  Pi = arr4_t{};   // consumed
  if (W_nodes != nullptr) *W_nodes = std::move(dW.local_());
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
inline void check_orientation(sector_t s, bool transposed, char const *who) {
  utils::check((s == sector_t::particle and not transposed) or (s == sector_t::hole and transposed),
               "gw_line::{}: only W^> (particle, plain orientation) and W^<^T (hole, transposed orientation) are provided "
               "from the stored residues w (see screened.hpp)",
               who);
}
} // namespace detail

/**
 * Residue exponentials on complex times t (block layout, MEM), out: (nt, nP, nQ):
 *   particle, plain       : W^>(q,t)   =  sum_j w_j(q) e^{-i nu_j t}
 *   hole,     transposed  : W^<(q,t)^T = -sum_j w_j(q) e^{+i nu_j t}     (bosonic_basis_t::time_exponentials(t, hole))
 */
template <MEMORY_SPACE MEM>
void w_time(memory::array<MEM, ComplexType, 4> const &w, bosonic_basis_t const &basis, long iq,
            nda::array<ComplexType, 1> const &t, sector_t s, bool transposed, memory::array_view<MEM, ComplexType, 3> out) {
  detail::check_orientation(s, transposed, "w_time");
  detail::pole_contract<MEM>(w, iq, basis.time_exponentials(t, s), out);
}

/**
 * Pole sums at mu-relative zeta (block layout, MEM), out: (nz, nP, nQ):
 *   particle, plain       : W^>(q,zeta)   =  sum_j w_j(q) / (zeta - nu_j)
 *   hole,     transposed  : W^<(q,zeta)^T = -sum_j w_j(q) / (zeta + nu_j)
 */
template <MEMORY_SPACE MEM>
void eval_poles(memory::array<MEM, ComplexType, 4> const &w, bosonic_basis_t const &basis, long iq,
                nda::array<ComplexType, 1> const &zeta, sector_t s, bool transposed,
                memory::array_view<MEM, ComplexType, 3> out) {
  detail::check_orientation(s, transposed, "eval_poles");
  auto [Km, Kp] = basis.kernels(zeta);
  if (s == sector_t::particle) detail::pole_contract<MEM>(w, iq, Km, out);
  else {
    Kp *= ComplexType(-1.0);
    detail::pole_contract<MEM>(w, iq, Kp, out);
  }
}

#define GW_LINE_SCREENED_EXTERN(MEM)                                                                                     \
  extern template struct coulomb_blocks_t<MEM>;                                                                          \
  extern template void screened_interaction<MEM>(memory::array<MEM, ComplexType, 4> &, coulomb_blocks_t<MEM> const &,     \
                                                 bosonic_basis_t const &, aux_grid_t const &,                             \
                                                 utils::mpi_context_t<boost::mpi3::communicator> &,                       \
                                                 memory::array<MEM, ComplexType, 4> &, utils::TimerManager &,             \
                                                 memory::array<MEM, ComplexType, 4> *, long);                             \
  extern template void w_time<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long,             \
                                   nda::array<ComplexType, 1> const &, sector_t, bool,                                    \
                                   memory::array_view<MEM, ComplexType, 3>);                                              \
  extern template void eval_poles<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long,         \
                                       nda::array<ComplexType, 1> const &, sector_t, bool,                                \
                                       memory::array_view<MEM, ComplexType, 3>);

GW_LINE_SCREENED_EXTERN(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SCREENED_EXTERN(DEVICE_MEMORY)
#endif
#undef GW_LINE_SCREENED_EXTERN

} // namespace methods::gw_line

#endif
