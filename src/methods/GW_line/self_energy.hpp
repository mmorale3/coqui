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

#ifndef COQUI_METHODS_GW_LINE_SELF_ENERGY_HPP
#define COQUI_METHODS_GW_LINE_SELF_ENERGY_HPP

/**
 * Correlation self-energy on the line by ray products (notes section 5.4, Eqs. siggtr, siglss; Eq. thc for the orbital
 * contraction; python LineGW.sigma):
 *
 *   Sigma~^>(k,t) = +(1/N_k) sum_q G~^>(k-q,t) o W^>(q,t),   W^>(q,t)   =  sum_j w_j(q) e^{-i nu_j t}     (particle ray)
 *   Sigma~^<(k,t) = -(1/N_k) sum_q G~^<(k-q,t) o W^<(q,t),   W^<(q,t)   = -sum_j w_j(q)^T e^{+i nu_j t}  (hole ray)
 *   Sigma^{>/<}_ab(k,zeta) = sum_t F_ray(zeta,t) sum_PQ conj(X_Pa(k)) Sigma~^{>/<}_PQ(k,t) X_Qb(k)
 *
 * The minus sign of the hole sector is the T = 0 factor [theta(nu) - theta(-eps)] = -1; with it both sectors have
 * positive residues (Sigma_c is retarded-like, A_Sigma >= 0).
 *
 * Orientation (no inter-rank transpose; see the header of screened.hpp). The particle sector is formed in PLAIN
 * orientation: G~(k-q,t) blocks (gtilde_form_t::plain) times W^>(q,t) blocks (w_time(.., particle, false)). Only w (not
 * w^T) is stored, so the hole sector is formed TRANSPOSED: B(k,t) = Sigma~^<(k,t)^T = -(1/N_k) sum_q G~^<(k-q,t)^T o
 * W^<(q,t)^T with gtilde_form_t::transposed and w_time(.., hole, true). The orbital contraction of a transposed aux
 * matrix is the transpose of an nb x nb result:
 *   Sigma_ab = sum_PQ conj(X_Pa) B_QP X_Qb = [X^T B conj(X)]_ba,
 * so the hole sector contracts the local block with Xp^T (left) and conj(Xq) (right) and the small nb x nb matrix is
 * transposed on the host after the all_reduce. Plain: Xp^dagger . block . Xq. All four X variants are the slices already
 * mirrored by propagator_t (Xp^dagger = Xpc^T, Xq = XqT^T, conj(Xq) = XqH^T: BLAS transpose flags only, no conj flags).
 *
 * Loop order (plan 6.3(d)): per sector, per time chunk:
 *   1. G~ blocks of ALL k for the chunk (memory N_k t_chunk block)                          timer Sigma_G_tilde
 *   2. per q: W(q, chunk) block once from the residues (gemm [t_chunk x r_b] . [r_b x block])  timer Sigma_W_time
 *      then per k: acc(k) += G~(k-q) o W(q)    (Hadamard; device: copies + tensor::elementwise) timer Sigma_hadamard
 *   3. per k, per t: partial(k,t) = (left X) acc(k,t) (right X), nb x nb (two gemms)        timer Sigma_contract
 *   4. ONE all_reduce of the host buffer [N_k, t_chunk, nb, nb] over the grid               timer Sigma_allreduce
 *   5. Sigma(k, :) += (sign/N_k) F[:, chunk] . partial(k)   (host gemm [N_zeta x t_chunk] . [t_chunk x nb^2]) timer Sigma_transform
 * Memory per rank: G~ and acc, 2 N_k t_chunk blocks, plus one W(q, chunk) (and one temp on device).
 *
 * Output: Sigma (N_k, N_zeta, nb, nb) on the host, identical on every rank (overwritten). `sectors` restricts the sum to one
 * sector (tests). Collective over mpi.comm (all ranks must call with the same rays, zeta and t_chunk).
 */

#include <array>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"

namespace methods::gw_line {

/// Implementation on an explicit communicator (comm must span exactly the ranks of `grid`; tests use a size-1
/// communicator with one block of a virtual grid).
template <MEMORY_SPACE MEM>
void self_energy(propagator_t<MEM> &prop, pole_data_t const &poles, memory::array<MEM, ComplexType, 4> const &w,
                 bosonic_basis_t const &basis, mf::MF const &mf, aux_grid_t const &grid, boost::mpi3::communicator &comm,
                 nda::array<ComplexType, 1> const &zeta, numerics::line_dlr::time_ray_t const &ray_p,
                 numerics::line_dlr::time_ray_t const &ray_h, long t_chunk, nda::array<ComplexType, 4> &Sigma,
                 utils::TimerManager &Timer, sector_t sectors = sector_t::both) {
  using numerics::line_dlr::time_ray_t;
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  using arr3_t = memory::array<MEM, ComplexType, 3>;
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  auto all     = nda::range::all;

  utils::check(mf.nkpts() == mf.nkpts_ibz() and mf.nqpts() == mf.nqpts_ibz(),
               "gw_line::self_energy: requires a mesh without symmetry reduction (nkpts {} nkpts_ibz {})", mf.nkpts(),
               mf.nkpts_ibz());
  utils::check(ray_p.sector == sector_t::particle and ray_h.sector == sector_t::hole,
               "gw_line::self_energy: ray_p must be a particle ray and ray_h a hole ray");
  utils::check(t_chunk > 0, "gw_line::self_energy: t_chunk must be > 0");
  utils::check(prop.grid.P0 == grid.P0 and prop.grid.nP == grid.nP and prop.grid.Q0 == grid.Q0 and prop.grid.nQ == grid.nQ,
               "gw_line::self_energy: propagator and grid blocks differ");
  utils::check(grid.np == comm.size() or comm.size() == 1, "gw_line::self_energy: grid of {} ranks on a communicator of {}",
               grid.np, comm.size());
  const long nk = prop.nk, nb = prop.nb, nq = mf.nqpts(), nz = zeta.size(), nP = grid.nP, nQ = grid.nQ;
  utils::check(mf.nkpts() == nk, "gw_line::self_energy: MF nkpts {} != X nkpts {}", mf.nkpts(), nk);
  utils::check(w.extent(0) == nq and w.extent(1) == basis.rank and w.extent(2) == nP and w.extent(3) == nQ,
               "gw_line::self_energy: residues w ({}, {}, {}, {}) vs (N_q, r_b, nP, nQ) = ({}, {}, {}, {})", w.extent(0),
               w.extent(1), w.extent(2), w.extent(3), nq, basis.rank, nP, nQ);
  auto qk = mf.qk_to_k2();   // (nqpts, nkpts): index of k - q

  for (auto nm : {"Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce", "Sigma_transform"})
    Timer.add(nm);
  prop.set_poles(poles);

  if (Sigma.extent(0) != nk or Sigma.extent(1) != nz or Sigma.extent(2) != nb or Sigma.extent(3) != nb)
    Sigma = nda::array<ComplexType, 4>(nk, nz, nb, nb);
  Sigma() = ComplexType(0.0);

  struct leg_t {
    time_ray_t const *ray;
    double sign;
    sector_t s;
    bool transposed;   // hole sector: Sigma~^T is formed (see the file header)
  };
  const leg_t legs[2] = {{&ray_p, +1.0, sector_t::particle, false}, {&ray_h, -1.0, sector_t::hole, true}};

  for (auto const &leg : legs) {
    if (sectors != sector_t::both and sectors != leg.s) continue;
    time_ray_t const &ray = *leg.ray;
    const long nt         = ray.size();
    const long tc         = std::min(t_chunk, nt);
    const auto form       = leg.transposed ? gtilde_form_t::transposed : gtilde_form_t::plain;
    nda::array<ComplexType, 2> F = ray.transform_matrix(zeta);   // (nz, nt), host
    arr4_t G(nk, tc, nP, nQ), acc(nk, tc, nP, nQ);
    arr3_t Wq(tc, nP, nQ);
    [[maybe_unused]] arr3_t tmp;
    if constexpr (MEM != HOST_MEMORY) tmp = arr3_t(tc, nP, nQ);
    arr2_t t1(nb, nQ);
    arr4_t part_m(nk, tc, nb, nb);                // contraction output in MEM (host: IS the reduce buffer)
    nda::array<ComplexType, 4> part(nk, tc, nb, nb), partT;
    if (leg.transposed) partT = nda::array<ComplexType, 4>(nk, tc, nb, nb);
    const ComplexType alpha(leg.sign / double(nk));

    for (long i0 = 0; i0 < nt; i0 += tc) {
      const long n = std::min(tc, nt - i0);
      nda::array<ComplexType, 1> t(ray.t(nda::range(i0, i0 + n)));
      const auto tr = nda::range(n);

      // 1. G~(k, chunk) for all k
      Timer.start("Sigma_G_tilde");
      for (long ik = 0; ik < nk; ++ik) prop.build(ik, t, leg.s, form, G(ik, tr, all, all));
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("Sigma_G_tilde");

      // 2. acc(k, chunk) = sum_q G~(k-q, chunk) o W(q, chunk)
      for (long iq = 0; iq < nq; ++iq) {
        Timer.start("Sigma_W_time");
        auto Wv = Wq(tr, all, all);
        w_time<MEM>(w, basis, iq, t, leg.s, leg.transposed, Wv);
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Sigma_W_time");

        Timer.start("Sigma_hadamard");
        for (long ik = 0; ik < nk; ++ik) {
          auto Gv   = G(qk(iq, ik), tr, all, all);
          auto acc_v = acc(ik, tr, all, all);
          if constexpr (MEM == HOST_MEMORY) {
            if (iq == 0) acc_v = Gv * Wv;
            else acc_v += Gv * Wv;
          } else {
            // device: copies + tensor::elementwise only (no lazy nda expressions into device arrays)
            if (iq == 0) {
              acc_v = Gv;
              nda::tensor::elementwise(ComplexType(1.0), Wv, ComplexType(1.0), acc_v, nda::tensor::op::MUL);
            } else {
              auto tmp_v = tmp(tr, all, all);
              tmp_v      = Gv;
              nda::tensor::elementwise(ComplexType(1.0), Wv, ComplexType(1.0), tmp_v, nda::tensor::op::MUL);
              nda::tensor::elementwise(ComplexType(1.0), tmp_v, ComplexType(1.0), acc_v, nda::tensor::op::SUM);
            }
          }
        }
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Sigma_hadamard");
      }

      // 3. orbital contraction of the local block, per k and t
      Timer.start("Sigma_contract");
      for (long ik = 0; ik < nk; ++ik) {
        for (long it = 0; it < n; ++it) {
          auto a = acc(ik, it, all, all);
          auto o = part_m(ik, it, all, all);
          if (not leg.transposed) {   // Xp^dagger a Xq
            nda::blas::gemm(ComplexType(1.0), nda::transpose(prop.Xpc(ik, all, all)), a, ComplexType(0.0), t1);
            nda::blas::gemm(ComplexType(1.0), t1, nda::transpose(prop.XqT(ik, all, all)), ComplexType(0.0), o);
          } else {                    // Xp^T a conj(Xq) = (contribution to Sigma)^T
            nda::blas::gemm(ComplexType(1.0), nda::transpose(prop.Xp(ik, all, all)), a, ComplexType(0.0), t1);
            nda::blas::gemm(ComplexType(1.0), t1, nda::transpose(prop.XqH(ik, all, all)), ComplexType(0.0), o);
          }
        }
      }
      if constexpr (MEM != HOST_MEMORY) {
        utils::device_sync();
        part = memory::to_memory_space<HOST_MEMORY>(part_m);
      } else {
        part = part_m;
      }
      Timer.stop("Sigma_contract");

      // 4. one all_reduce per chunk (the full buffer: identical size on every rank)
      Timer.start("Sigma_allreduce");
      comm.all_reduce_in_place_n(part.data(), part.size(), std::plus<>{});
      Timer.stop("Sigma_allreduce");

      // 5. transform to the line nodes (host)
      Timer.start("Sigma_transform");
      if (leg.transposed) {
        for (long ik = 0; ik < nk; ++ik)
          for (long it = 0; it < n; ++it) partT(ik, it, all, all) = nda::transpose(part(ik, it, all, all));
      }
      auto &P = leg.transposed ? partT : part;
      for (long ik = 0; ik < nk; ++ik) {
        auto P2 = nda::reshape(P(ik, all, all, all), std::array<long, 2>{tc, nb * nb})(tr, all);
        auto S2 = nda::reshape(Sigma(ik, all, all, all), std::array<long, 2>{nz, nb * nb});
        nda::blas::gemm(alpha, F(all, nda::range(i0, i0 + n)), P2, ComplexType(1.0), S2);
      }
      Timer.stop("Sigma_transform");
    }
  }
}

/// Sigma on the grid of mpi.comm (the production entry point).
template <MEMORY_SPACE MEM>
void self_energy(propagator_t<MEM> &prop, pole_data_t const &poles, memory::array<MEM, ComplexType, 4> const &w,
                 bosonic_basis_t const &basis, mf::MF const &mf, aux_grid_t const &grid,
                 utils::mpi_context_t<boost::mpi3::communicator> &mpi, nda::array<ComplexType, 1> const &zeta,
                 numerics::line_dlr::time_ray_t const &ray_p, numerics::line_dlr::time_ray_t const &ray_h, long t_chunk,
                 nda::array<ComplexType, 4> &Sigma, utils::TimerManager &Timer, sector_t sectors = sector_t::both) {
  utils::check(grid.np == mpi.comm.size() and grid.rank == mpi.comm.rank(), "gw_line::self_energy: grid/communicator mismatch");
  self_energy<MEM>(prop, poles, w, basis, mf, grid, mpi.comm, zeta, ray_p, ray_h, t_chunk, Sigma, Timer, sectors);
}

#define GW_LINE_SIGMA_EXTERN(MEM)                                                                                        \
  extern template void self_energy<MEM>(propagator_t<MEM> &, pole_data_t const &, memory::array<MEM, ComplexType, 4> const &, \
                                        bosonic_basis_t const &, mf::MF const &, aux_grid_t const &,                      \
                                        boost::mpi3::communicator &,                                                      \
                                        nda::array<ComplexType, 1> const &, numerics::line_dlr::time_ray_t const &,       \
                                        numerics::line_dlr::time_ray_t const &, long, nda::array<ComplexType, 4> &,       \
                                        utils::TimerManager &, sector_t);

GW_LINE_SIGMA_EXTERN(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SIGMA_EXTERN(DEVICE_MEMORY)
#endif
#undef GW_LINE_SIGMA_EXTERN

} // namespace methods::gw_line

#endif
