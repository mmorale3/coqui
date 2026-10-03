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
 * S7d: the stage runs in sub-steps (w_plan_t, proc_grid.hpp): one q per q pool and a zeta sub-slab at a time, Dyson IN
 * PLACE in the whole-matrix buffer (W^T), the in-place transpose giving W, the fit per q sub-step. Besides the resident
 * Pi group (reused for W) only one q row per pool of W^T and the sub-slab buffers are held (before: two extra slabs).
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

#include <algorithm>
#include <array>
#include <cstdlib>
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
#include "methods/GW_line/device_blas.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::sector_t;

// dyson_layout_t (the whole-matrix layout of the Dyson step) and w_plan_t (its S7d sub-steps): proc_grid.hpp

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

} // namespace detail

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
  utils::check(q0 >= 0 and q0 + g <= Zb.nq, "gw_line::screened_interaction: q group [{}, {}) out of [0, {})", q0, q0 + g, Zb.nq);
  dyson_layout_t lay(comm.size(), comm.rank(), g, nz, Np);
  for (auto nm : {"W_redistribute", "W_dyson", "W_fit"}) Timer.add(nm);
  if (lay.nq_loc > 0) {
    utils::check(Zb.has_full(q0 + lay.q_first) and Zb.has_full(q0 + lay.q_first + lay.nq_loc - 1),
                 "gw_line::screened_interaction: the Coulomb blocks do not hold the full Z of this rank's Dyson q slab");
  }

  // residues: resident (allocated first, so that the stage's high-water includes them as the plan 6.7 model does)
  if (w.extent(0) != Zb.nq or w.extent(1) != r or w.extent(2) != nP or w.extent(3) != nQ) {
    w = arr4_t(Zb.nq, r, nP, nQ);
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

  const std::array<long, 4> bgrid = {1, 1, grid.np_P, grid.np_Q}, ones = {1, 1, 1, 1};
  for (long s = 0; s < plan.nsub_q; ++s) {
    const long na   = plan.n_act(s);
    const bool act  = s < lay.nq_loc;                // this rank's q pool solves a q in this sub-step
    const long iq_a = q0 + lay.q_first + s;          // ... namely this one (absolute)
    auto WT         = sWT.template view<4>({na, nz, nP, nQ});
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

      // 3. W^T -> block layout -> the WT rows of the sub-step (block (I,J) of W^T = (W_JI)^T, the fit's mirror block)
      Timer.start("W_redistribute");
      math::nda::redistribute(dD, dTb);
      for (long p = 0; p < na; ++p) WT(p, zrng, all, all) = Tb(p, all, all, all);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("W_redistribute");

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

    // 5. symmetric fit of the sub-step's q, all pairs at once: w(q) = VS (U1H W(q) + U2H W^T(q))
    Timer.start("W_fit");
    for (long p = 0; p < na; ++p) {
      const long ql = plan.q_row(p, s);
      auto W2       = nda::reshape(Pi(ql, all, all, all), std::array<long, 2>{nz, blk});
      auto WT2      = nda::reshape(WT(p, all, all, all), std::array<long, 2>{nz, blk});
      auto w2       = nda::reshape(w(q0 + ql, all, all, all), std::array<long, 2>{r, blk});
      for (long c0 = 0; c0 < blk; c0 += bc) {
        const auto cr = nda::range(c0, std::min(blk, c0 + bc));
        auto Yc       = Y(all, nda::range(cr.size()));
        nda::blas::gemm(ComplexType(1.0), U1H, W2(all, cr), ComplexType(0.0), Yc);
        nda::blas::gemm(ComplexType(1.0), U2H, WT2(all, cr), ComplexType(1.0), Yc);
        nda::blas::gemm(ComplexType(1.0), VS, Yc, ComplexType(0.0), w2(all, cr));
      }
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
