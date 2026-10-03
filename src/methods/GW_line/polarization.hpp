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

#ifndef COQUI_METHODS_GW_LINE_POLARIZATION_HPP
#define COQUI_METHODS_GW_LINE_POLARIZATION_HPP

/**
 * Polarization on the line by ray products (notes Eqs. pigtr, pilss; python LineGW.polarization):
 *
 *   Pi^>(q,t)_PQ = +(2/N_k) sum_k [ conj(G~^<(k, conj t)) o G~^>(k-q, t) ]^T_PQ   on the particle ray,
 *   Pi^<(q,t)_PQ = -(2/N_k) sum_k [ conj(G~^>(k, conj t)) o G~^<(k-q, t) ]^T_PQ   on the hole ray,
 *   Pi(q, zeta)  = sum_{sectors} sum_t F_ray(zeta, t) Pi^{>/<}(q, t)              (Eq. laplace, ray.transform_matrix).
 *
 * Orientation (no inter-rank transpose). Since (A o B)^T = A^T o B^T, the bracket's transpose is formed by building the
 * TRANSPOSED factors directly as (P_rng, Q_rng) blocks of this rank:
 *   conj(G~(k, conj t))^T = G~(k, conj t)^dagger = Xp C_k(conj t)^dagger Xq^dagger   (gtilde_form_t::adjoint_conj_t),
 *   G~(k-q, t)^T          = conj(Xp) C_{k-q}(t)^T Xq^T                              (gtilde_form_t::transposed),
 * i.e. the transpose lands on the small nb x nb C(t) (a BLAS op flag) and on the X slices prepared once, never on the
 * Np x Np blocks. The product is then elementwise in (P,Q): no communication at all.
 *
 * Loop structure (plan 6.3b): per sector, per time chunk, the A and B factors of ALL k are built once (memory
 * 2 N_k t_chunk block), then per q: acc(t) = sum_k A(k) o B(k-q) and Pi(q, :) += (sign 2/N_k) F[:, chunk] acc (one gemm
 * [N_zeta x t_chunk] . [t_chunk x block]).
 *
 * t_chunk <= 0 selects the chunk automatically: 8 on the host, on the device the largest chunk whose A, B, acc fit in 40%
 * of the free device memory (<= 256; the transform gemm [N_zeta x t_chunk] . [t_chunk x block] has inner dimension t_chunk).
 * Device Hadamard: one cuTENSOR elementwise_trinary per (q, k), acc <- A o B + acc.
 *
 * Output: Pi is the local block in MEM, shape (N_q, N_zeta, nP, nQ) = Pi(q, zeta)[P_rng, Q_rng] (overwritten). It stays
 * in MEM because the next consumer (S4: Dyson for W) redistributes a MEM darray.
 */

#include <array>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/device_blas.hpp"

namespace methods::gw_line {

/**
 * Pi(q, zeta) blocks for all q at the mu-relative points zeta. `sectors` restricts the sum to one sector (tests);
 * ray_p / ray_h must be particle / hole rays. Collective: none (every rank computes its own (P,Q) block).
 */
template <MEMORY_SPACE MEM>
void polarization(propagator_t<MEM> &prop, pole_data_t const &poles, mf::MF const &mf, aux_grid_t const &grid,
                  nda::array<ComplexType, 1> const &zeta, numerics::line_dlr::time_nodes_t const &ray_p,
                  numerics::line_dlr::time_nodes_t const &ray_h, long t_chunk, memory::array<MEM, ComplexType, 4> &Pi,
                  utils::TimerManager &Timer, sector_t sectors = sector_t::both) {
  using time_ray_t = numerics::line_dlr::time_nodes_t;   // GL ray or ID nodes (S7b)
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  using arr3_t = memory::array<MEM, ComplexType, 3>;
  auto all     = nda::range::all;

  utils::check(mf.nkpts() == mf.nkpts_ibz() and mf.nqpts() == mf.nqpts_ibz(),
               "gw_line::polarization: requires a mesh without symmetry reduction (nkpts {} nkpts_ibz {})", mf.nkpts(),
               mf.nkpts_ibz());
  utils::check(ray_p.sector == sector_t::particle and ray_h.sector == sector_t::hole,
               "gw_line::polarization: ray_p must be a particle ray and ray_h a hole ray");
  utils::check(prop.grid.P0 == grid.P0 and prop.grid.nP == grid.nP and prop.grid.Q0 == grid.Q0 and prop.grid.nQ == grid.nQ,
               "gw_line::polarization: propagator and grid blocks differ");
  const long nk = prop.nk, nq = mf.nqpts(), nz = zeta.size(), nP = grid.nP, nQ = grid.nQ, blk = nP * nQ;
  utils::check(mf.nkpts() == nk, "gw_line::polarization: MF nkpts {} != X nkpts {}", mf.nkpts(), nk);
  auto qk = mf.qk_to_k2();   // (nqpts, nkpts): index of k - q

  for (auto nm : {"G_tilde", "Pi_hadamard", "Pi_transform"}) Timer.add(nm);
  prop.set_poles(poles);

  if (Pi.extent(0) != nq or Pi.extent(1) != nz or Pi.extent(2) != nP or Pi.extent(3) != nQ) Pi = arr4_t(nq, nz, nP, nQ);
  nda::tensor::set(ComplexType(0.0), Pi);

  struct leg_t {
    time_ray_t const *ray;
    double sign;
    sector_t s_k, s_kmq;   // sector of the state at k (factor A) and at k-q (factor B)
  };
  const leg_t legs[2] = {{&ray_p, +1.0, sector_t::hole, sector_t::particle},
                         {&ray_h, -1.0, sector_t::particle, sector_t::hole}};

  for (auto const &leg : legs) {
    if (sectors != sector_t::both and sectors != leg.ray->sector) continue;
    time_ray_t const &ray = *leg.ray;
    const long nt         = ray.size();
    // t_chunk <= 0: automatic (host 8; device from the free device memory: A, B of all k and acc per time node)
    const long tc = (t_chunk > 0) ? std::min(t_chunk, nt) : detail::auto_t_chunk<MEM>(nt, double(2 * nk + 1) * blk * 16.0);
    memory::array<MEM, ComplexType, 2> F = memory::to_memory_space<MEM>(ray.transform_matrix(zeta));   // (nz, nt)
    arr4_t A(nk, tc, nP, nQ), B(nk, tc, nP, nQ);
    arr3_t acc(tc, nP, nQ);
    if constexpr (MEM != HOST_MEMORY) device_mem_probe();
    app_log(3, "  gw_line::polarization: {} sector, {} time nodes in chunks of {}", leg.ray->sector == sector_t::particle ? "particle" : "hole", nt, tc);
    const ComplexType alpha(leg.sign * 2.0 / double(nk));

    for (long i0 = 0; i0 < nt; i0 += tc) {
      const long n = std::min(tc, nt - i0);
      nda::array<ComplexType, 1> t(ray.t(nda::range(i0, i0 + n)));
      const auto tr = nda::range(n);

      Timer.start("G_tilde");
      for (long ik = 0; ik < nk; ++ik) {
        prop.build(ik, t, leg.s_k, gtilde_form_t::adjoint_conj_t, A(ik, tr, all, all));
        prop.build(ik, t, leg.s_kmq, gtilde_form_t::transposed, B(ik, tr, all, all));
      }
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("G_tilde");

      for (long iq = 0; iq < nq; ++iq) {
        Timer.start("Pi_hadamard");
        auto acc_v = acc(tr, all, all);
        for (long ik = 0; ik < nk; ++ik) {
          const long ikmq = qk(iq, ik);
          auto Ak = A(ik, tr, all, all);
          auto Bk = B(ikmq, tr, all, all);
          if constexpr (MEM == HOST_MEMORY) {
            if (ik == 0) acc_v = Ak * Bk;
            else acc_v += Ak * Bk;
          } else {
            // device: one cuTENSOR trinary per k, acc <- (A o B) + acc (no lazy nda expressions on device arrays)
            if (ik == 0) nda::tensor::set(ComplexType(0.0), acc_v);
#if defined(ENABLE_DEVICE)   // device-only: older host nda checkouts lack elementwise_trinary
            nda::tensor::elementwise_trinary(ComplexType(1.0), Ak, "abc", ComplexType(1.0), Bk, "abc", ComplexType(1.0), acc_v,
                                             "abc", nda::tensor::op::MUL, nda::tensor::op::SUM);
#endif
          }
        }
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Pi_hadamard");

        Timer.start("Pi_transform");
        auto acc2 = nda::reshape(acc, std::array<long, 2>{tc, blk})(tr, all);
        auto Pi2  = nda::reshape(Pi(iq, all, all, all), std::array<long, 2>{nz, blk});
        nda::blas::gemm(alpha, F(all, nda::range(i0, i0 + n)), acc2, ComplexType(1.0), Pi2);
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Pi_transform");
      }
    }
  }
}

extern template void polarization<HOST_MEMORY>(propagator_t<HOST_MEMORY> &, pole_data_t const &, mf::MF const &,
                                               aux_grid_t const &, nda::array<ComplexType, 1> const &,
                                               numerics::line_dlr::time_nodes_t const &,
                                               numerics::line_dlr::time_nodes_t const &, long,
                                               memory::array<HOST_MEMORY, ComplexType, 4> &, utils::TimerManager &, sector_t);
#if defined(ENABLE_DEVICE)
extern template void polarization<DEVICE_MEMORY>(propagator_t<DEVICE_MEMORY> &, pole_data_t const &, mf::MF const &,
                                                 aux_grid_t const &, nda::array<ComplexType, 1> const &,
                                                 numerics::line_dlr::time_nodes_t const &,
                                                 numerics::line_dlr::time_nodes_t const &, long,
                                                 memory::array<DEVICE_MEMORY, ComplexType, 4> &, utils::TimerManager &,
                                                 sector_t);
#endif

} // namespace methods::gw_line

#endif
