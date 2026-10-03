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

#ifndef COQUI_METHODS_GW_LINE_STATIC_PART_HPP
#define COQUI_METHODS_GW_LINE_STATIC_PART_HPP

/**
 * Static part of the self-energy (notes section 5.5, Eq. hf; plan 6.3(e); python LineGW.density_matrix /
 * hartree_exchange). Spin-restricted: D(k) per spin, total density 2 D.
 *
 *   D(k)     = sum_{m in <} coef_m                       (T = 0 density matrix from the hole-sector poles;
 *                                                         V_h V_h^dagger in the factorized form)
 *   D~(k)    = X(k) D(k) X(k)^dagger,   rho_Q = (2/N_k) sum_k D~_QQ(k)
 *   V_H,ab(k)  = sum_P conj(X_Pa(k)) [Z(0) rho]_P X_Pb(k)
 *   Sigma_x,ab(k) = -(1/N_k) sum_q sum_PQ conj(X_Pa(k)) D~_PQ(k-q) Z_PQ(q) X_Qb(k)
 *   F = V_H + Sigma_x
 *
 * G = 0 treatment: Z(Gamma) of the THC reader carries no G = 0 term, and NO Madelung / head correction is added, i.e. the
 * "ignore_g0" convention of the imaginary-axis code (hf_t("ignore_g0")). The gygi-type correction is S6.
 *
 * Block layout (plan 6.2): every rank works on its (P_rng, Q_rng) block for all k and q, and the result is the sum of the
 * per-block contributions, so ONE all_reduce of [N_k, nb, nb] over the grid finishes both terms:
 *   - rho on this rank's Q_rng is computed locally for all k (rho_Q = sum_b [D^T Xq^T]_bQ conj(X_Qb): one gemm, one
 *     elementwise product and a column sum by a gemm with a ones row), no communication;
 *   - v_P = sum_{Q in Q_rng} Z(0)_PQ rho_Q is the partial Hartree potential of the block, and
 *     sum_{P in P_rng} conj(X_Pa) v_P X_Pb summed over ALL blocks is exactly V_H;
 *   - Sigma_x per block: Xp^dagger [sum_q D~(k-q) o Z(q)] Xq.
 * Everything in MEM is copies, gemm and tensor::elementwise (device-safe); D and F are host arrays. Collective over mpi.comm.
 */

#include <array>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"

namespace methods::gw_line {

/// D(k) = sum over the hole-sector poles of coef_m (per spin, T = 0); (nk, nb, nb), host. Factorized hole sectors:
/// D = V_h V_h^dagger (one gemm, pole_sector_t::density).
inline nda::array<ComplexType, 3> density_matrix(pole_data_t const &poles) {
  nda::array<ComplexType, 3> D(poles.nk, poles.nb, poles.nb);
  D() = ComplexType(0.0);
  for (long ik = 0; ik < poles.nk; ++ik)
    if (poles.hole[ik].size() > 0) D(ik, nda::range::all, nda::range::all) = poles.hole[ik].density();
  return D;
}

/// Index of q = Gamma: the q with k - q = k for every k.
inline long gamma_index(mf::MF const &mf) {
  auto qk = mf.qk_to_k2();
  for (long iq = 0; iq < mf.nqpts(); ++iq) {
    bool g = true;
    for (long ik = 0; ik < mf.nkpts() and g; ++ik) g = (qk(iq, ik) == ik);
    if (g) return iq;
  }
  utils::check(false, "gw_line::gamma_index: no q with k - q = k for all k");
  return -1;
}

/**
 * F = V_H + Sigma_x (Eq. hf) from the density matrix D (nk, nb, nb) (per spin). F (nk, nb, nb) on the host, identical on
 * every rank (overwritten). Collective over comm, which must span exactly the ranks of `grid` (or be of size 1: one block
 * of a virtual grid, tests).
 */
template <MEMORY_SPACE MEM>
void hartree_exchange(propagator_t<MEM> &prop, coulomb_blocks_t<MEM> const &Zb, nda::array<ComplexType, 3> const &D,
                      mf::MF const &mf, aux_grid_t const &grid, boost::mpi3::communicator &comm,
                      nda::array<ComplexType, 3> &F, utils::TimerManager &Timer) {
  using arr3_t = memory::array<MEM, ComplexType, 3>;
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  auto all     = nda::range::all;

  utils::check(mf.nkpts() == mf.nkpts_ibz() and mf.nqpts() == mf.nqpts_ibz(),
               "gw_line::hartree_exchange: requires a mesh without symmetry reduction");
  utils::check(prop.grid.P0 == grid.P0 and prop.grid.nP == grid.nP and prop.grid.Q0 == grid.Q0 and prop.grid.nQ == grid.nQ,
               "gw_line::hartree_exchange: propagator and grid blocks differ");
  utils::check(Zb.grid.P0 == grid.P0 and Zb.grid.nP == grid.nP and Zb.grid.Q0 == grid.Q0 and Zb.grid.nQ == grid.nQ,
               "gw_line::hartree_exchange: Coulomb blocks and grid differ");
  utils::check(grid.np == comm.size() or comm.size() == 1, "gw_line::hartree_exchange: grid of {} ranks on a communicator of {}",
               grid.np, comm.size());
  const long nk = prop.nk, nb = prop.nb, nq = mf.nqpts(), nP = grid.nP, nQ = grid.nQ;
  utils::check(D.extent(0) == nk and D.extent(1) == nb and D.extent(2) == nb, "gw_line::hartree_exchange: D shape mismatch");
  utils::check(Zb.nq == nq, "gw_line::hartree_exchange: Coulomb blocks for {} q, mesh has {}", Zb.nq, nq);
  auto qk        = mf.qk_to_k2();
  const long iq0 = gamma_index(mf);

  for (auto nm : {"HX_density", "HX_hartree", "HX_exchange", "HX_allreduce"}) Timer.add(nm);

  arr3_t Dm = memory::to_memory_space<MEM>(D);
  arr3_t Dt(nk, nP, nQ);              // D~(k) blocks
  arr3_t Fp(nk, nb, nb);              // per-block partial F
  arr2_t tPb(nP, nb), tbQ(nb, nQ), tbP(nb, nP);
  arr2_t rho(1, nQ), vP(nP, 1), Vb(nP, nb), vX(nP, nb), acc(nP, nQ);
  [[maybe_unused]] arr2_t tmp;
  if constexpr (MEM != HOST_MEMORY) tmp = arr2_t(nP, nQ);
  arr2_t ones_b(1, nb);
  {
    nda::array<ComplexType, 2> o(1, nb);
    o() = ComplexType(1.0);
    ones_b = memory::to_memory_space<MEM>(o);
  }

  // D~(k) blocks and rho on Q_rng
  Timer.start("HX_density");
  nda::tensor::set(ComplexType(0.0), rho);
  for (long ik = 0; ik < nk; ++ik) {
    nda::blas::gemm(ComplexType(1.0), prop.Xp(ik, all, all), Dm(ik, all, all), ComplexType(0.0), tPb);
    nda::blas::gemm(ComplexType(1.0), tPb, prop.XqH(ik, all, all), ComplexType(0.0), Dt(ik, all, all));
    // rho_Q += (2/N_k) sum_b [D^T Xq^T]_bQ conj(X_Qb)
    nda::blas::gemm(ComplexType(1.0), nda::transpose(Dm(ik, all, all)), prop.XqT(ik, all, all), ComplexType(0.0), tbQ);
    if constexpr (MEM == HOST_MEMORY) {
      tbQ *= prop.XqH(ik, all, all);
    } else {
      nda::tensor::elementwise(ComplexType(1.0), prop.XqH(ik, all, all), ComplexType(1.0), tbQ, nda::tensor::op::MUL);
    }
    nda::blas::gemm(ComplexType(2.0 / double(nk)), ones_b, tbQ, ComplexType(1.0), rho);
  }
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("HX_density");

  // Hartree: v_P = sum_{Q in Q_rng} Z(0)_PQ rho_Q, then Fp(k) = Xp^dagger diag(v) Xp
  Timer.start("HX_hartree");
  nda::blas::gemm(ComplexType(1.0), Zb.Z(iq0, all, all), nda::transpose(rho), ComplexType(0.0), vP);
  nda::blas::gemm(ComplexType(1.0), vP, ones_b, ComplexType(0.0), Vb);   // v broadcast over the nb columns
  for (long ik = 0; ik < nk; ++ik) {
    vX = prop.Xp(ik, all, all);
    if constexpr (MEM == HOST_MEMORY) {
      vX *= Vb;
    } else {
      nda::tensor::elementwise(ComplexType(1.0), Vb, ComplexType(1.0), vX, nda::tensor::op::MUL);
    }
    nda::blas::gemm(ComplexType(1.0), nda::transpose(prop.Xpc(ik, all, all)), vX, ComplexType(0.0), Fp(ik, all, all));
  }
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("HX_hartree");

  // exchange: Fp(k) -= (1/N_k) Xp^dagger [sum_q D~(k-q) o Z(q)] Xq
  Timer.start("HX_exchange");
  for (long ik = 0; ik < nk; ++ik) {
    for (long iq = 0; iq < nq; ++iq) {
      auto Dv = Dt(qk(iq, ik), all, all);
      auto Zv = Zb.Z(iq, all, all);
      if constexpr (MEM == HOST_MEMORY) {
        if (iq == 0) acc = Dv * Zv;
        else acc += Dv * Zv;
      } else {
        if (iq == 0) {
          acc = Dv;
          nda::tensor::elementwise(ComplexType(1.0), Zv, ComplexType(1.0), acc, nda::tensor::op::MUL);
        } else {
          tmp = Dv;
          nda::tensor::elementwise(ComplexType(1.0), Zv, ComplexType(1.0), tmp, nda::tensor::op::MUL);
          nda::tensor::elementwise(ComplexType(1.0), tmp, ComplexType(1.0), acc, nda::tensor::op::SUM);
        }
      }
    }
    nda::blas::gemm(ComplexType(1.0), nda::transpose(prop.Xpc(ik, all, all)), acc, ComplexType(0.0), tbQ);
    nda::blas::gemm(ComplexType(-1.0 / double(nk)), tbQ, nda::transpose(prop.XqT(ik, all, all)), ComplexType(1.0),
                    Fp(ik, all, all));
  }
  if constexpr (MEM != HOST_MEMORY) utils::device_sync();
  Timer.stop("HX_exchange");

  Timer.start("HX_allreduce");
  F = memory::to_memory_space<HOST_MEMORY>(Fp);
  comm.all_reduce_in_place_n(F.data(), F.size(), std::plus<>{});
  Timer.stop("HX_allreduce");
}

/// F on the grid of mpi.comm (the production entry point).
template <MEMORY_SPACE MEM>
void hartree_exchange(propagator_t<MEM> &prop, coulomb_blocks_t<MEM> const &Zb, nda::array<ComplexType, 3> const &D,
                      mf::MF const &mf, aux_grid_t const &grid, utils::mpi_context_t<boost::mpi3::communicator> &mpi,
                      nda::array<ComplexType, 3> &F, utils::TimerManager &Timer) {
  utils::check(grid.np == mpi.comm.size() and grid.rank == mpi.comm.rank(), "gw_line::hartree_exchange: grid/communicator mismatch");
  hartree_exchange<MEM>(prop, Zb, D, mf, grid, mpi.comm, F, Timer);
}

#define GW_LINE_STATIC_EXTERN(MEM)                                                                                       \
  extern template void hartree_exchange<MEM>(propagator_t<MEM> &, coulomb_blocks_t<MEM> const &,                         \
                                             nda::array<ComplexType, 3> const &, mf::MF const &, aux_grid_t const &,      \
                                             boost::mpi3::communicator &,                                                 \
                                             nda::array<ComplexType, 3> &, utils::TimerManager &);

GW_LINE_STATIC_EXTERN(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_STATIC_EXTERN(DEVICE_MEMORY)
#endif
#undef GW_LINE_STATIC_EXTERN

} // namespace methods::gw_line

#endif
