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
 *   Sigma~^<(k,t) = -(1/N_k) sum_q G~^<(k-q,t) o W^<(q,t),   W^<(q,t)   = -sum_j w_j(-q)^T e^{+i nu_j t} (hole ray)
 *   Sigma^{>/<}_ab(k,zeta) = sum_t F_ray(zeta,t) sum_PQ conj(X_Pa(k)) Sigma~^{>/<}_PQ(k,t) X_Qb(k)
 *
 * The minus sign of the hole sector is the T = 0 factor [theta(nu) - theta(-eps)] = -1; with it both sectors have
 * positive residues (Sigma_c is retarded-like, A_Sigma >= 0).
 *
 * q <-> -q pairing (screened.hpp, notes section 3.3): the hole interaction of q carries the residues of -q. The hole leg
 * runs over the residue ROWS q' (contiguous, also for the host-resident residue groups) and pairs each with the propagator
 * of k - q at q = -q' (mf::MF::qminus): sum_q G~^<(k-q) o W^<(q)^T = -sum_q' G~^<(k + q') o sum_j w_j(q') e^{+i nu_j t}.
 * For self-inverse q this is the old per-q sum.
 *
 * Orientation (no inter-rank transpose; see the header of screened.hpp). The particle sector is formed in PLAIN
 * orientation: G~(k-q,t) blocks (gtilde_form_t::plain) times W^>(q,t) blocks (w_time(.., particle, false)). Only w (not
 * w^T) is stored, so the hole sector is formed TRANSPOSED: B(k,t) = Sigma~^<(k,t)^T = -(1/N_k) sum_q G~^<(k-q,t)^T o
 * W^<(q,t)^T with gtilde_form_t::transposed and w_time(.., hole, true) (residue row -q). The orbital contraction of a transposed aux
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
 *   Device: 4-5 are replaced by the transform of the rank-local partials on the device (one strided-batched gemm per
 *   chunk) and ONE copy + all_reduce of [N_k, N_zeta, nb, nb] per sector at the end.
 * Memory per rank: G~ and acc, 2 N_k t_chunk blocks, plus one W(q, chunk) (and one temp on device).
 *
 * t_chunk <= 0 selects the chunk automatically (host host_t_chunk_default; device from the free device memory, <= 128).
 * Device: contraction as two strided-batched gemms per (k, chunk); the residue exponentials are formed once per chunk (q
 * independent) on both paths. Device Hadamard (S7d, the fused kernel of cuda/gw_line_cuda.cuh): default
 * (COQUI_GWLINE_SIGMA_KOUTER = 1) W(q, chunk) of ALL q is formed by one strided-batched gemm (memory N_q t_chunk block)
 * and ONE launch forms acc(k) = sum_q G~(k-q) o W(q) for all k (G~, W read once, acc written once, no zeroing);
 * KOUTER = 0: per q one launch updating acc(k) += G~(k-q) o W(q) for all k (W(q) transient). COQUI_GWLINE_FUSED = 0:
 * one cuTENSOR elementwise_trinary per (q, k) (the bring-up path).
 *
 * Output: Sigma (N_k, N_zeta, nb, nb) on the host, identical on every rank (overwritten). `sectors` restricts the sum to one
 * sector (tests). Collective over mpi.comm (all ranks must call with the same rays, zeta and t_chunk).
 *
 * k_local (S7e, the driver's mode): Sigma is k-DISTRIBUTED, (nloc, N_zeta, nb, nb) with the rows of the k owned by this
 * rank (k_dist_t: owner(k) = k mod np, the closure's ownership). Step 4 becomes ONE MPI_Reduce_scatter per chunk of the
 * owner-ordered partials (half the volume of the all_reduce, no replicated result) and step 5 is done by the owner only
 * (before: every rank transformed all N_k: N_k N_zeta t_chunk nb^2 per chunk on every rank). Device: the final copy is
 * reduce-scattered the same way. Same sums in the same order per element -> identical to the replicated mode up to the
 * reduction order of MPI.
 *
 * Orbital contraction order (S7e): a(t) is nP x nQ with nP <= nQ (aux_grid_t: np_P >= np_Q); right factor first,
 * u = a R (nP nQ nb), then L u (nb^2 nP), instead of L a (nb nP nQ) then (.) R (nb^2 nQ): the nb^2 term then scales with
 * the larger grid factor np_P instead of np_Q.
 */

#include <array>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/device_blas.hpp"
#include "methods/GW_line/k_dist.hpp"

namespace methods::gw_line {

/// Implementation on an explicit communicator (comm must span exactly the ranks of `grid`; tests use a size-1
/// communicator with one block of a virtual grid).
template <MEMORY_SPACE MEM>
void self_energy(propagator_t<MEM> &prop, pole_data_t const &poles, memory::array<MEM, ComplexType, 4> const &w,
                 bosonic_basis_t const &basis, mf::MF const &mf, aux_grid_t const &grid, boost::mpi3::communicator &comm,
                 nda::array<ComplexType, 1> const &zeta, numerics::line_dlr::time_nodes_t const &ray_p,
                 numerics::line_dlr::time_nodes_t const &ray_h, long t_chunk, nda::array<ComplexType, 4> &Sigma,
                 utils::TimerManager &Timer, sector_t sectors = sector_t::both, bool k_local = false,
                 memory::array<HOST_MEMORY, ComplexType, 4> const *w_host = nullptr, long sig_qgroup = 0) {
  using time_ray_t = numerics::line_dlr::time_nodes_t;   // GL ray or ID nodes (S7b)
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  using arr3_t = memory::array<MEM, ComplexType, 3>;
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  auto all     = nda::range::all;

  utils::check(mf.nkpts() == mf.nkpts_ibz() and mf.nqpts() == mf.nqpts_ibz(),
               "gw_line::self_energy: requires a mesh without symmetry reduction (nkpts {} nkpts_ibz {})", mf.nkpts(),
               mf.nkpts_ibz());
  utils::check(ray_p.sector == sector_t::particle and ray_h.sector == sector_t::hole,
               "gw_line::self_energy: ray_p must be a particle ray and ray_h a hole ray");
  utils::check(prop.grid.P0 == grid.P0 and prop.grid.nP == grid.nP and prop.grid.Q0 == grid.Q0 and prop.grid.nQ == grid.nQ,
               "gw_line::self_energy: propagator and grid blocks differ");
  utils::check(grid.np == comm.size() or comm.size() == 1, "gw_line::self_energy: grid of {} ranks on a communicator of {}",
               grid.np, comm.size());
  const long nk = prop.nk, nb = prop.nb, nq = mf.nqpts(), nz = zeta.size(), nP = grid.nP, nQ = grid.nQ, r = basis.rank;
  utils::check(mf.nkpts() == nk, "gw_line::self_energy: MF nkpts {} != X nkpts {}", mf.nkpts(), nk);
  {
    auto const &wc = (w_host != nullptr) ? w_host->shape() : w.shape();
    utils::check(wc[0] == nq and wc[1] == basis.rank and wc[2] == nP and wc[3] == nQ,
                 "gw_line::self_energy: residues w ({}, {}, {}, {}) vs (N_q, r_b, nP, nQ) = ({}, {}, {}, {})", wc[0], wc[1], wc[2],
                 wc[3], nq, basis.rank, nP, nQ);
  }
  // q groups of the Sigma stage (S7e): with host-resident residues (w_host) the q are processed in groups of sig_qgroup
  // (default all) whose residues are copied to MEM once per (group, sector); G~ is rebuilt per group, the contraction and
  // the transform are linear and accumulate over the groups
  const long gsz = (w_host != nullptr and sig_qgroup > 0) ? std::min(sig_qgroup, nq) : nq;
  auto qk = mf.qk_to_k2();   // (nqpts, nkpts): index of k - q
  auto qm = mf.qminus();     // (nqpts): index of -q
  // the q whose interaction the residue row iq enters: q for the particle leg, -q for the hole leg (see the header)
  auto q_of_row = [&](sector_t s, long iq) -> long { return s == sector_t::hole ? long(qm(iq)) : iq; };

  for (auto nm : {"Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce", "Sigma_transform"})
    Timer.add(nm);
  prop.set_poles(poles);

  const k_dist_t kd(nk, comm);
  const long nks = k_local ? kd.nloc() : nk;   // rows of Sigma on this rank
  if (Sigma.extent(0) != nks or Sigma.extent(1) != nz or Sigma.extent(2) != nb or Sigma.extent(3) != nb)
    Sigma = nda::array<ComplexType, 4>(nks, nz, nb, nb);
  Sigma() = ComplexType(0.0);
  // contraction order: right factor first when nP < nQ (see the file header); COQUI_GWLINE_CONTRACT_RIGHT = 0/1 forces
  const bool right_first = detail::env_long("COQUI_GWLINE_CONTRACT_RIGHT", nP < nQ ? 1 : 0) != 0;

  struct leg_t {
    time_ray_t const *ray;
    double sign;
    sector_t s;
    bool transposed;   // hole sector: Sigma~^T is formed (see the file header)
  };
  const leg_t legs[2] = {{&ray_p, +1.0, sector_t::particle, false}, {&ray_h, -1.0, sector_t::hole, true}};

  // ---- S7d Hadamard setup (device fused kernel; see the file header)
  const long blk = nP * nQ;
  [[maybe_unused]] const bool fused  = (MEM != HOST_MEMORY) and detail::fused_hadamard();
  [[maybe_unused]] const bool kouter = fused and detail::env_long("COQUI_GWLINE_SIGMA_KOUTER", 1) != 0;
  // kouter: (N_k, gs, 2) = (k-q, q - qs0); else (gs, N_k, 1, 2) = (k-q, 0), for the q group [qs0, qs0 + gs)
  [[maybe_unused]] memory::array<MEM, int, 1> pairs;
  auto make_pairs = [&](long qs0, long gs, sector_t s) {
    if (not fused) return;
    nda::array<int, 1> ph(2 * gs * nk);
    for (long iqr = 0; iqr < gs; ++iqr)
      for (long ik = 0; ik < nk; ++ik) {
        const long i = kouter ? (ik * gs + iqr) : (iqr * nk + ik);
        ph(2 * i)     = int(qk(q_of_row(s, qs0 + iqr), ik));
        ph(2 * i + 1) = kouter ? int(iqr) : 0;
      }
    pairs = memory::to_memory_space<MEM>(ph);
  };
  const long nwq = kouter ? gsz : 1;   // W(q, chunk) arrays held
  arr4_t wbuf;                         // host-resident residues: the group's rows in MEM

  for (auto const &leg : legs) {
    if (sectors != sector_t::both and sectors != leg.s) continue;
    time_ray_t const &ray = *leg.ray;
    const long nt         = ray.size();
    // t_chunk <= 0: automatic (host default; device from the free device memory: G~, acc of all k, W(q) (all q when
    // fused k-outer), contraction buffers)
    const long tc = (t_chunk > 0) ? std::min(t_chunk, nt)
                                  : detail::auto_t_chunk<MEM>(nt, double(2 * nk + nwq) * nP * nQ * 16.0 +
                                                                      double(nk * nb + nQ) * nb * 16.0);
    const auto form       = leg.transposed ? gtilde_form_t::transposed : gtilde_form_t::plain;
    nda::array<ComplexType, 2> F = ray.transform_matrix(zeta);   // (nz, nt), host
    arr4_t G(nk, tc, nP, nQ), acc(nk, tc, nP, nQ);
    arr4_t Wq(nwq, tc, nP, nQ);                   // W(q, chunk): all q (fused k-outer) or the current q
    arr2_t Em(tc, r);                             // residue exponentials of the chunk (q independent)
    arr2_t t1(nb, nQ), u1(nP, nb);
    [[maybe_unused]] arr3_t t1b;                  // device: the batched first contraction factor, all t of a chunk
    if constexpr (MEM != HOST_MEMORY) t1b = arr3_t(tc, right_first ? nP : nb, right_first ? nb : nQ);
    arr4_t part_m(nk, tc, nb, nb);                // contraction output in MEM (host: IS the reduce buffer)
    nda::array<ComplexType, 4> part, partT;
    nda::array<ComplexType, 1> kd_send, kd_recv;   // k_local: owner-ordered partials / this rank's reduced rows
    [[maybe_unused]] arr2_t Fm;                   // device: the transform matrix (nz, nt)
    [[maybe_unused]] arr4_t Sig_m;                // device: this rank's (unreduced) Sigma of the leg, (nk, nz, nb, nb)
    if constexpr (MEM == HOST_MEMORY) {
      if (k_local) {
        kd_send = nda::array<ComplexType, 1>(nk * tc * nb * nb);
        kd_recv = nda::array<ComplexType, 1>(std::max(1L, kd.nloc()) * tc * nb * nb);
      } else {
        part = nda::array<ComplexType, 4>(nk, tc, nb, nb);
        if (leg.transposed) partT = nda::array<ComplexType, 4>(nk, tc, nb, nb);
      }
    } else {
      Fm    = memory::to_memory_space<MEM>(F);
      Sig_m = arr4_t(nk, nz, nb, nb);
      nda::tensor::set(ComplexType(0.0), Sig_m);
    }
    if constexpr (MEM != HOST_MEMORY) device_mem_probe();
    app_log(3, "  gw_line::self_energy: {} sector, {} time nodes in chunks of {}", leg.s == sector_t::particle ? "particle" : "hole",
            nt, tc);
    const ComplexType alpha(leg.sign / double(nk));

    for (long qs0 = 0; qs0 < nq; qs0 += gsz) {   // q groups (one group = all q unless host-resident residues)
      const long gs = std::min(gsz, nq - qs0);
      make_pairs(qs0, gs, leg.s);
      if (w_host != nullptr) {
        Timer.start("Sigma_W_time");
        wbuf = memory::to_memory_space<MEM>(nda::array<ComplexType, 4>((*w_host)(nda::range(qs0, qs0 + gs), all, all, all)));
        Timer.stop("Sigma_W_time");
      }
      auto const &wsrc = (w_host != nullptr) ? wbuf : w;   // rows: q - qoff
      const long qoff  = (w_host != nullptr) ? qs0 : 0;
      for (long i0 = 0; i0 < nt; i0 += tc) {
        const long n = std::min(tc, nt - i0);
        nda::array<ComplexType, 1> t(ray.t(nda::range(i0, i0 + n)));
        const auto tr = nda::range(n);

        // 1. G~(k, chunk) for all k
        Timer.start("Sigma_G_tilde");
        for (long ik = 0; ik < nk; ++ik) prop.build(ik, t, leg.s, form, G(ik, tr, all, all));
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Sigma_G_tilde");

        // 2. acc(k, chunk) = sum_q G~(k-q, chunk) o W(q, chunk); the exponentials of the chunk once for all q
        Timer.start("Sigma_W_time");
        auto Ev = Em(tr, all);
        detail::check_orientation(leg.s, leg.transposed, "self_energy");
        Ev = basis.time_exponentials(t, leg.s);
        Timer.stop("Sigma_W_time");
        if (kouter) {   // device, fused: W(q, chunk) for all q (one gemm), then acc(k) for all k in one launch
          Timer.start("Sigma_W_time");
          // column-major: W(q)^T (blk x n) = w(q)^T (blk x r) . E^T (r x n), batched over q
          detail::gemm_strided_cm('N', 'N', blk, n, r, ComplexType(1.0), wsrc.data() + (qs0 - qoff) * r * blk, blk, r * blk,
                                  Em.data(), r, 0, ComplexType(0.0), Wq.data(), blk, tc * blk, gs);
          utils::device_sync();
          Timer.stop("Sigma_W_time");
          Timer.start("Sigma_hadamard");
          detail::slab_conv<MEM>(n * blk, nk, G.data(), tc * blk, gs, Wq.data(), tc * blk, nk, acc.data(), tc * blk, gs, pairs,
                                 0, ComplexType(1.0), false);
          utils::device_sync();
          Timer.stop("Sigma_hadamard");
        } else {
          if constexpr (MEM != HOST_MEMORY)
            if (not fused) nda::tensor::set(ComplexType(0.0), acc);
          for (long iq = qs0; iq < qs0 + gs; ++iq) {
            Timer.start("Sigma_W_time");
            auto Wv = Wq(0, tr, all, all);
            detail::pole_contract_m<MEM>(wsrc, iq - qoff, Ev, Wv);
            if constexpr (MEM != HOST_MEMORY) utils::device_sync();
            Timer.stop("Sigma_W_time");

            Timer.start("Sigma_hadamard");
            if (fused) {   // device, fused: acc(k) (+)= G~(k-q) o W(q) for all k in one launch
              detail::slab_conv<MEM>(n * blk, nk, G.data(), tc * blk, 1, Wq.data(), tc * blk, nk, acc.data(), tc * blk, 1, pairs,
                                     2 * (iq - qs0) * nk, ComplexType(1.0), iq > qs0);
            } else {
              for (long ik = 0; ik < nk; ++ik) {
                auto Gv    = G(qk(q_of_row(leg.s, iq), ik), tr, all, all);
                auto acc_v = acc(ik, tr, all, all);
                if constexpr (MEM == HOST_MEMORY) {
                  if (iq == qs0) acc_v = Gv * Wv;
                  else acc_v += Gv * Wv;
                } else {
                  // device: one cuTENSOR trinary per (q, k), acc <- (G~ o W) + acc (acc zeroed before the q loop)
  #if defined(ENABLE_DEVICE)   // device-only: older host nda checkouts lack elementwise_trinary
                  nda::tensor::elementwise_trinary(ComplexType(1.0), Gv, "abc", ComplexType(1.0), Wv, "abc", ComplexType(1.0),
                                                   acc_v, "abc", nda::tensor::op::MUL, nda::tensor::op::SUM);
  #endif
                }
              }
            }
            if constexpr (MEM != HOST_MEMORY) utils::device_sync();
            Timer.stop("Sigma_hadamard");
          }
        }

        // 3. orbital contraction of the local block, per k and t
        Timer.start("Sigma_contract");
        for (long ik = 0; ik < nk; ++ik) {
          if constexpr (MEM == HOST_MEMORY) {
            // L = Xp^dagger = Xpc^T, R = Xq = XqT^T (plain); L = Xp^T, R = conj(Xq) = XqH^T (transposed: Sigma^T)
            auto L = nda::transpose((leg.transposed ? prop.Xp : prop.Xpc)(ik, all, all));
            auto R = nda::transpose((leg.transposed ? prop.XqH : prop.XqT)(ik, all, all));
            for (long it = 0; it < n; ++it) {
              auto a = acc(ik, it, all, all);
              auto o = part_m(ik, it, all, all);
              if (right_first) {   // u = a R (nP x nb), o = L u
                nda::blas::gemm(ComplexType(1.0), a, R, ComplexType(0.0), u1);
                nda::blas::gemm(ComplexType(1.0), L, u1, ComplexType(0.0), o);
              } else {             // t1 = L a (nb x nQ), o = t1 R
                nda::blas::gemm(ComplexType(1.0), L, a, ComplexType(0.0), t1);
                nda::blas::gemm(ComplexType(1.0), t1, R, ComplexType(0.0), o);
              }
            }
          } else {
            // device: two strided-batched gemms over the n times of the chunk (column-major view, device_blas.hpp):
            //   t1(t) = L a(t),  L = Xp^dagger = Xpc^T (plain) or Xp^T (transposed)  ->  t1(t)^T = a(t)^T L^T
            //   o(t)  = t1(t) R, R = Xq = XqT^T (plain) or conj(Xq) = XqH^T (transposed) -> o(t)^T = R^T t1(t)^T
            ComplexType const *xl = (leg.transposed ? prop.Xp : prop.Xpc).data() + ik * nP * nb;
            ComplexType const *xr = (leg.transposed ? prop.XqH : prop.XqT).data() + ik * nb * nQ;
            auto a = acc(ik, tr, all, all);
            auto o = part_m(ik, tr, all, all);
            if (right_first) {
              //   u(t) = a(t) R  ->  u(t)^T (nb x nP) = R^T a(t)^T: R_cm = xr (nQ x nb, ld nQ) with op 'T', a_cm (nQ x nP)
              //   o(t) = L u(t)  ->  o(t)^T = u(t)^T L^T: L_cm = xl (nb x nP, ld nb) with op 'T'
              detail::gemm_strided_cm('T', 'N', nb, nP, nQ, ComplexType(1.0), xr, nQ, 0, a.data(), nQ, nP * nQ, ComplexType(0.0),
                                      t1b.data(), nb, nb * nP, n);
              detail::gemm_strided_cm('N', 'T', nb, nb, nP, ComplexType(1.0), t1b.data(), nb, nb * nP, xl, nb, 0, ComplexType(0.0),
                                      o.data(), nb, nb * nb, n);
            } else {
              detail::gemm_strided_cm('N', 'T', nQ, nb, nP, ComplexType(1.0), a.data(), nQ, nP * nQ, xl, nb, 0, ComplexType(0.0),
                                      t1b.data(), nQ, nb * nQ, n);
              detail::gemm_strided_cm('T', 'N', nb, nb, nQ, ComplexType(1.0), xr, nQ, 0, t1b.data(), nQ, nb * nQ, ComplexType(0.0),
                                      o.data(), nb, nb * nb, n);
            }
          }
        }
        if constexpr (MEM != HOST_MEMORY) {
          utils::device_sync();
          Timer.stop("Sigma_contract");
          // 4'-5'. device: transform the rank-local partials on the device, Sig_m(k) += alpha F[:, chunk] part_m(k, chunk)
          //        for all k in one strided-batched gemm (F broadcast); column-major: Sig(k)^T = part(k)^T F_chunk^T.
          //        No per-chunk device->host copy or all_reduce: one of each at the end of the leg (Sigma is linear in
          //        the partials, so reducing after the transform is the same sum).
          Timer.start("Sigma_transform");
          detail::gemm_strided_cm('N', 'N', nb * nb, nz, n, alpha, part_m.data(), nb * nb, tc * nb * nb, Fm.data() + i0, nt, 0,
                                  ComplexType(1.0), Sig_m.data(), nb * nb, nz * nb * nb, nk);
          utils::device_sync();
          Timer.stop("Sigma_transform");
          continue;
        } else if (k_local) {
          // owner-ordered copy of the n times of every k (row = n nb^2)
          const long row = n * nb * nb;
          long o = 0;
          for (long r = 0; r < kd.np; ++r)
            for (long l = 0; l < kd.nloc(r); ++l, ++o) {
              auto src = part_m(kd.global(l, r), tr, all, all);
              nda::array_view<ComplexType, 3> dst(std::array<long, 3>{n, nb, nb}, kd_send.data() + o * row);
              dst = src;
            }
        } else {
          part = part_m;
        }
        Timer.stop("Sigma_contract");

        if (k_local) {
          // 4''. ONE reduce-scatter per chunk: every owner receives the summed partials of its k
          Timer.start("Sigma_allreduce");
          const long row = n * nb * nb;
          kd_reduce_scatter(comm, kd, kd_send.data(), kd_recv.data(), row);
          Timer.stop("Sigma_allreduce");
          // 5''. transform of the owned k only
          Timer.start("Sigma_transform");
          for (long l = 0; l < kd.nloc(); ++l) {
            nda::array_view<ComplexType, 3> P3(std::array<long, 3>{n, nb, nb}, kd_recv.data() + l * row);
            if (leg.transposed)
              for (long it = 0; it < n; ++it) P3(it, all, all) = nda::make_regular(nda::transpose(P3(it, all, all)));
            auto P2 = nda::reshape(P3, std::array<long, 2>{n, nb * nb});
            auto S2 = nda::reshape(Sigma(l, all, all, all), std::array<long, 2>{nz, nb * nb});
            nda::blas::gemm(alpha, F(all, nda::range(i0, i0 + n)), P2, ComplexType(1.0), S2);
          }
          Timer.stop("Sigma_transform");
          continue;
        }

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

    if constexpr (MEM != HOST_MEMORY) {   // device: one copy + one all_reduce of [N_k, N_zeta, nb, nb] per leg
      Timer.start("Sigma_allreduce");
      nda::array<ComplexType, 4> S_h = memory::to_memory_space<HOST_MEMORY>(Sig_m);
      if (k_local) {   // reduce-scatter of the owner-ordered copy: each owner gets its k
        nda::array<ComplexType, 4> S_o, S_l(std::max(1L, kd.nloc()), nz, nb, nb);
        kd_to_owner_order(kd, S_h, S_o);
        kd_reduce_scatter(comm, kd, S_o.data(), S_l.data(), nz * nb * nb);
        for (long l = 0; l < kd.nloc(); ++l)
          for (long iz = 0; iz < nz; ++iz) {
            if (leg.transposed) Sigma(l, iz, all, all) += nda::transpose(S_l(l, iz, all, all));
            else Sigma(l, iz, all, all) += S_l(l, iz, all, all);
          }
        Timer.stop("Sigma_allreduce");
        continue;
      }
      comm.all_reduce_in_place_n(S_h.data(), S_h.size(), std::plus<>{});
      for (long ik = 0; ik < nk; ++ik)
        for (long iz = 0; iz < nz; ++iz) {
          if (leg.transposed) Sigma(ik, iz, all, all) += nda::transpose(S_h(ik, iz, all, all));
          else Sigma(ik, iz, all, all) += S_h(ik, iz, all, all);
        }
      Timer.stop("Sigma_allreduce");
    }
  }
}

/// Sigma on the grid of mpi.comm (the production entry point).
template <MEMORY_SPACE MEM>
void self_energy(propagator_t<MEM> &prop, pole_data_t const &poles, memory::array<MEM, ComplexType, 4> const &w,
                 bosonic_basis_t const &basis, mf::MF const &mf, aux_grid_t const &grid,
                 utils::mpi_context_t<boost::mpi3::communicator> &mpi, nda::array<ComplexType, 1> const &zeta,
                 numerics::line_dlr::time_nodes_t const &ray_p, numerics::line_dlr::time_nodes_t const &ray_h, long t_chunk,
                 nda::array<ComplexType, 4> &Sigma, utils::TimerManager &Timer, sector_t sectors = sector_t::both,
                 bool k_local = false, memory::array<HOST_MEMORY, ComplexType, 4> const *w_host = nullptr,
                 long sig_qgroup = 0) {
  utils::check(grid.np == mpi.comm.size() and grid.rank == mpi.comm.rank(), "gw_line::self_energy: grid/communicator mismatch");
  self_energy<MEM>(prop, poles, w, basis, mf, grid, mpi.comm, zeta, ray_p, ray_h, t_chunk, Sigma, Timer, sectors, k_local,
                   w_host, sig_qgroup);
}

#define GW_LINE_SIGMA_EXTERN(MEM)                                                                                        \
  extern template void self_energy<MEM>(propagator_t<MEM> &, pole_data_t const &, memory::array<MEM, ComplexType, 4> const &, \
                                        bosonic_basis_t const &, mf::MF const &, aux_grid_t const &,                      \
                                        boost::mpi3::communicator &,                                                      \
                                        nda::array<ComplexType, 1> const &, numerics::line_dlr::time_nodes_t const &,     \
                                        numerics::line_dlr::time_nodes_t const &, long, nda::array<ComplexType, 4> &,     \
                                        utils::TimerManager &, sector_t, bool,                                    \
                                        memory::array<HOST_MEMORY, ComplexType, 4> const *, long);

GW_LINE_SIGMA_EXTERN(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SIGMA_EXTERN(DEVICE_MEMORY)
#endif
#undef GW_LINE_SIGMA_EXTERN

} // namespace methods::gw_line

#endif
