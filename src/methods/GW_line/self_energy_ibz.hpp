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

#ifndef COQUI_METHODS_GW_LINE_SELF_ENERGY_IBZ_HPP
#define COQUI_METHODS_GW_LINE_SELF_ENERGY_IBZ_HPP

/**
 * Correlation self-energy at the IBZ k only (perf 7.3; ibz.hpp item (S); the equations of self_energy.hpp):
 *
 *   Sigma~^>_isym(ks, t) = +(1/N_k) sum_{q' in class isym} G~^>(ks - q_eff, t) o W^>(q_eff, t),   W^> from residue row q_eff
 *   Sigma~^<_isym(ks, t)^T = -(1/N_k) sum_{q' in class}  G~^<(ks - q_eff, t)^T o V(-q_eff, t),   V(r, t) = sum_j w_j(r) e^{+i nu_j t}
 *   Sigma_ij(k, zeta) = sum_t F(zeta, t) sum_isym [ (X(ks) D)^dagger Sigma~_isym(ks, t) (X(ks) D) ]_ij       (hole: transposed)
 * with ks = ks_to_k(isym, k), D = symmetry_rotation(isym, k) (identity class: ks = k, D = 1), q_eff in R (ibz_t).
 *
 * Every class sum is a k-mesh convolution evaluated at the nk_ibz points ks only (kmesh_ft.hpp conventions):
 *   G^(R)       = sum_{k'} e^{-ik'R} G~(k')              (all full-BZ k', once per chunk; hole leg: conj A^ of Pi, cached)
 *   W_c^(R)     = sum_{q' in class} e^{-iQ_{q_eff} R} W(q_eff)       (the class rows gathered from W(r, t) of the rows R)
 *   A_c(ks)     = 1/N sum_R e^{+i ks R} G^(R) o W_c^(R)              (back transform to the nk_ibz rows ks of the class)
 * Cost per (t, block element): N^2 (G^) + |R| r_b (W(t) of the rows) + sum_c [N n_c + N + nk_ibz N] = N^2 + |R| r_b + N N_q +
 * n_cls N (1 + nk_ibz), against N^2 + N r_b + 2 N^2 + N for the full-BZ path: the convolution does NOT shrink with the IBZ
 * (the aux basis cannot be rotated: one class sum per symmetry); what shrinks is W(t) (|R| instead of N_q rows), the
 * contraction / reduction / transform (nk_ibz n_cls instead of N_k contractions, nk_ibz rows reduced) and G~ (C(t) shared
 * within each star, propagators.hpp).
 *
 * Output: Sigma (nk_ibz or the k_local rows of k_dist_t(nk_ibz), N_zeta, nb, nb) on the host; Sigma_h != nullptr: the hole leg
 * goes there (as self_energy). Collective over comm (all ranks must call with the same rays, zeta, t_chunk). w: the residues of
 * the rows R in ibz.rows order, (|R|, r_b, nP, nQ) in MEM.
 */

#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/device_blas.hpp"
#include "methods/GW_line/k_dist.hpp"
#include "methods/GW_line/kmesh_ft.hpp"
#include "methods/GW_line/kmesh_fft.hpp"
#include "methods/GW_line/ibz.hpp"

namespace methods::gw_line {

template <MEMORY_SPACE MEM>
void self_energy_ibz(propagator_t<MEM> &prop, pole_data_t const &poles, memory::array<MEM, ComplexType, 4> const &w,
                     bosonic_basis_t const &basis, mf::MF const &mf, ibz_t const &ibz, aux_grid_t const &grid,
                     boost::mpi3::communicator &comm, nda::array<ComplexType, 1> const &zeta,
                     numerics::line_dlr::time_nodes_t const &ray_p, numerics::line_dlr::time_nodes_t const &ray_h, long t_chunk,
                     nda::array<ComplexType, 4> &Sigma, utils::TimerManager &Timer, sector_t sectors = sector_t::both,
                     bool k_local = false, nda::array<ComplexType, 4> *Sigma_h = nullptr) {
  using time_ray_t = numerics::line_dlr::time_nodes_t;
  using arr4_t     = memory::array<MEM, ComplexType, 4>;
  using arr2_t     = memory::array<MEM, ComplexType, 2>;
  auto all         = nda::range::all;

  const long nk = prop.nk, nb = prop.nb, nz = zeta.size(), nP = grid.nP, nQ = grid.nQ, r = basis.rank, blk = nP * nQ;
  const long nkI = ibz.nkI, nR = ibz.nrows(), ncl = ibz.nclasses();
  utils::check(nk == ibz.nk and mf.nkpts() == nk, "gw_line::self_energy_ibz: X has {} k, IBZ tables {}", nk, ibz.nk);
  utils::check(ray_p.sector == sector_t::particle and ray_h.sector == sector_t::hole, "gw_line::self_energy_ibz: ray sectors");
  utils::check(w.extent(0) == nR and w.extent(1) == r and w.extent(2) == nP and w.extent(3) == nQ,
               "gw_line::self_energy_ibz: residues ({}, {}, {}, {}) vs (|R| {}, r_b {}, {}, {})", w.extent(0), w.extent(1),
               w.extent(2), w.extent(3), nR, r, nP, nQ);
  utils::check(grid.np == comm.size() or comm.size() == 1, "gw_line::self_energy_ibz: grid / communicator mismatch");
  const kmesh_ft_t kft(mf);
  utils::check(kft.ok, "gw_line::self_energy_ibz: the k mesh has no valid Fourier matrices (kmesh_ft_t)");

  for (auto nm : {"Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce", "Sigma_transform",
                  "Sig_ft_G", "Sig_ft_W", "Sig_ft_prod", "Sig_ft_back"})
    Timer.add(nm);
  // perf 7.5c instrumentation: COQUI_GWLINE_FTPROF = 1 logs the class sums' time per class (level 1, once per call)
  const bool ftprof = detail::env_long("COQUI_GWLINE_FTPROF", 0) != 0;
  std::vector<double> t_cls(ibz.nclasses(), 0.0);
  prop.set_poles(poles);   // IBZ poles unfolded by the propagator (set_ibz)

  const k_dist_t kd(nkI, comm);
  const long nks = k_local ? kd.nloc() : nkI;
  for (auto *S : {&Sigma, Sigma_h}) {
    if (S == nullptr) continue;
    if (S->extent(0) != nks or S->extent(1) != nz or S->extent(2) != nb or S->extent(3) != nb)
      *S = nda::array<ComplexType, 4>(nks, nz, nb, nb);
    (*S)() = ComplexType(0.0);
  }
  auto out_of = [&](sector_t s) -> nda::array<ComplexType, 4> & { return (s == sector_t::hole and Sigma_h) ? *Sigma_h : Sigma; };

  // perf 7.5c: the class sums with FFTs of the mesh (kmesh_fft.hpp): G^ = FFT-[G built into the rows kpos],
  // W_c^ = FFT-[W(q_eff) placed at qpos(q_eff)], A_c(ks) = 1/N FFT+[G^ o W_c^](kpos(ks)); host: fused per cache block
  // (one G^ per block shared by the classes); device: whole-chunk cuFFT, 1/N in the contraction
  const kmesh_map_t kmap(kft, mf);
  const bool use_fft = kft_mode<MEM>(kmap) == kft_mode_t::fft;
  auto krow          = [&](long ik) { return use_fft ? kmap.kpos[ik] : ik; };
  std::unique_ptr<kmesh_fft_host_t> eng;
  if constexpr (MEM == HOST_MEMORY)
    if (use_fft) eng = std::make_unique<kmesh_fft_host_t>(kmap, 2);
  std::vector<std::vector<long>> wpos(ncl), kspos(ncl);   // FFT: mesh rows of the class's q_eff and of its ks
  std::vector<std::vector<char>> wdup(ncl);               // a q_eff row already placed in this class (accumulate)
  if (use_fft)
    for (long c = 0; c < ncl; ++c) {
      auto const &cl = ibz.cls[c];
      std::vector<char> seen(nk, 0);
      for (long q : cl.q_eff) {
        wpos[c].push_back(kmap.qpos[q]);
        wdup[c].push_back(seen[kmap.qpos[q]]);
        seen[kmap.qpos[q]] = 1;
      }
      for (long k : cl.ks) kspos[c].push_back(kmap.kpos[k]);
    }

  // ---- fixed per call: Fourier matrices, XD slices of every (class, IBZ k) on the host
  memory::array<MEM, ComplexType, 2> FmM;
  if (not use_fft) FmM = memory::to_memory_space<MEM>(kft.Fm);
  std::vector<arr2_t> Hc(ncl), Bc(ncl);   // (N, n_c) e^{-iQ_{q_eff} R}; (nk_ibz, N) e^{+i ks R} / N
  for (long c = 0; c < ncl and not use_fft; ++c) {
    auto const &cl = ibz.cls[c];
    const long nc  = cl.q_eff.size();
    nda::array<ComplexType, 2> h(nk, nc), b(nkI, nk);
    for (long R = 0; R < nk; ++R)
      for (long j = 0; j < nc; ++j) h(R, j) = kft.Hm(R, cl.q_eff[j]);
    for (long k = 0; k < nkI; ++k)
      for (long R = 0; R < nk; ++R) b(k, R) = kft.Bp(cl.ks[k], R) / double(nk);
    Hc[c] = memory::to_memory_space<MEM>(h);
    Bc[c] = memory::to_memory_space<MEM>(b);
  }
  nda::array<ComplexType, 3> xp_h = memory::to_memory_space<HOST_MEMORY>(prop.Xp);    // (nk, nP, nb)
  nda::array<ComplexType, 3> xq_h(nk, nQ, nb);                                         // Xq = XqT^T
  {
    nda::array<ComplexType, 3> xqt = memory::to_memory_space<HOST_MEMORY>(prop.XqT);   // (nk, nb, nQ)
    for (long k = 0; k < nk; ++k) xq_h(k, all, all) = nda::transpose(xqt(k, all, all));
  }
  // XD slices: XDp(c, k) = Xp(ks) D (nP, nb), XDq(c, k) = Xq(ks) D (nQ, nb)
  nda::array<ComplexType, 4> XDp(ncl, nkI, nP, nb), XDq(ncl, nkI, nQ, nb);
  for (long c = 0; c < ncl; ++c)
    for (long k = 0; k < nkI; ++k) {
      const long ks = ibz.cls[c].ks[k];
      if (ibz.D[c].empty()) {
        XDp(c, k, all, all) = xp_h(ks, all, all);
        XDq(c, k, all, all) = xq_h(ks, all, all);
      } else {
        nda::blas::gemm(ComplexType(1.0), xp_h(ks, all, all), ibz.D[c][k], ComplexType(0.0), XDp(c, k, all, all));
        nda::blas::gemm(ComplexType(1.0), xq_h(ks, all, all), ibz.D[c][k], ComplexType(0.0), XDq(c, k, all, all));
      }
    }
  const bool right_first = detail::env_long("COQUI_GWLINE_CONTRACT_RIGHT", nP < nQ ? 1 : 0) != 0;

  struct leg_t {
    time_ray_t const *ray;
    double sign;
    sector_t s;
    bool transposed;
  };
  const leg_t legs[2] = {{&ray_p, +1.0, sector_t::particle, false}, {&ray_h, -1.0, sector_t::hole, true}};

  for (auto const &leg : legs) {
    if (sectors != sector_t::both and sectors != leg.s) continue;
    time_ray_t const &ray = *leg.ray;
    const long nt         = ray.size();
    long ncmax            = 1;
    for (auto const &cl : ibz.cls) ncmax = std::max(ncmax, long(cl.q_eff.size()));
    const long tc = (t_chunk > 0) ? std::min(t_chunk, nt)
                                  : detail::auto_t_chunk<MEM>(nt, double(3 * nk + nR + ncmax + ncl * nkI) * blk * 16.0);
    auto &Sigma_out = out_of(leg.s);
    bool use_cache  = leg.s == sector_t::hole and prop.ahat_key >= 0.0 and prop.ahat_key == prop.pole_key and
                     prop.ahat_t.size() == nt and prop.ahat.extent(0) == nk and prop.ahat.extent(2) == nP and
                     prop.ahat.extent(3) == nQ and detail::env_long("COQUI_GWLINE_GT_CACHE", -1) != 0;
    if (use_cache)
      for (long i = 0; i < nt; ++i)
        if (ray.t(i) != std::conj(prop.ahat_t(i))) { use_cache = false; break; }
    // residue row of every R row for this leg: the particle leg reads row q, the hole leg row -q
    std::vector<long> wrow(nR);
    for (long i = 0; i < nR; ++i) wrow[i] = leg.s == sector_t::hole ? ibz.rpos[ibz.qminus[ibz.rows[i]]] : i;
    // contraction factors of the leg (MEM): plain L = XD_p^dagger (nb, nP), R = XD_q (nQ, nb); transposed L = XD_p^T,
    // R = conj(XD_q)
    std::vector<arr2_t> Lc(ncl * nkI), Rc(ncl * nkI);
    for (long c = 0; c < ncl; ++c)
      for (long k = 0; k < nkI; ++k) {
        nda::array<ComplexType, 2> L(nb, nP), Rr(nQ, nb);
        if (leg.transposed) {
          L  = nda::transpose(XDp(c, k, all, all));
          Rr = nda::conj(XDq(c, k, all, all));
        } else {
          L  = nda::dagger(XDp(c, k, all, all));
          Rr = XDq(c, k, all, all);
        }
        Lc[c * nkI + k] = memory::to_memory_space<MEM>(L);
        Rc[c * nkI + k] = memory::to_memory_space<MEM>(Rr);
      }
    const auto form              = leg.transposed ? gtilde_form_t::transposed : gtilde_form_t::plain;
    nda::array<ComplexType, 2> F = ray.transform_matrix(zeta);   // (nz, nt), host
    // (host FFT: G^ lives in the cache blocks, no Gh array)
    arr4_t G(nk, tc, nP, nQ), Gh((use_fft and MEM == HOST_MEMORY) ? 0 : nk, tc, nP, nQ), Wt(nR, tc, nP, nQ);
    memory::array<MEM, ComplexType, 5> accC(ncl, nkI, tc, nP, nQ);   // the back-transformed class sums of the chunk
    [[maybe_unused]] arr4_t Wg, Wc;                                   // device: whole-chunk class buffers
    if constexpr (MEM != HOST_MEMORY) {
      Wg = arr4_t(ncmax, tc, nP, nQ);
      Wc = arr4_t(nk, tc, nP, nQ);
    }
    const long cb = std::max(16L, detail::env_long("COQUI_GWLINE_IBZ_CB", 512));   // host column block
    detail::scratch_t<HOST_MEMORY> s_gb, s_wc;
    arr2_t Em(tc, r), u1(nP, nb), t1(nb, nQ);
    arr4_t part_m(nkI, tc, nb, nb);
    nda::array<ComplexType, 4> part, partT;
    nda::array<ComplexType, 1> kd_send, kd_recv;
    if (k_local) {
      kd_send = nda::array<ComplexType, 1>(nkI * tc * nb * nb);
      kd_recv = nda::array<ComplexType, 1>(std::max(1L, kd.nloc()) * tc * nb * nb);
    } else {
      part = nda::array<ComplexType, 4>(nkI, tc, nb, nb);
      if (leg.transposed) partT = nda::array<ComplexType, 4>(nkI, tc, nb, nb);
    }
    app_log(3, "  gw_line::self_energy_ibz: {} sector, {} time nodes in chunks of {}, {} classes, {} rows{}{}",
            leg.s == sector_t::particle ? "particle" : "hole", nt, tc, ncl, nR, use_cache ? " (G^ from the cached A^ of Pi)" : "",
            use_fft ? " (FFT of the mesh, perf 7.5c)" : "");
    const ComplexType alpha(leg.sign / double(nk));

    for (long i0 = 0; i0 < nt; i0 += tc) {
      const long n = std::min(tc, nt - i0);
      nda::array<ComplexType, 1> t(ray.t(nda::range(i0, i0 + n)));
      const auto tr = nda::range(n);
      const auto cr = nda::range(n * blk);

      // 1. G~ of all full-BZ k and G^(R)
      Timer.start("Sigma_G_tilde");
      if (not use_cache)
        for (long ik = 0; ik < nk; ++ik) prop.build(ik, t, leg.s, form, G(krow(ik), tr, all, all));
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("Sigma_G_tilde");
      const long ld = tc * blk, ncols = n * blk;
      if (use_fft) {   // perf 7.5c: W(r, chunk) first, then the class sums with FFTs
        Timer.start("Sigma_W_time");
        {
          nda::array<ComplexType, 2> Eh = basis.time_exponentials(t, leg.s);
          auto Ev = Em(tr, all);
          Ev      = Eh;
          for (long i = 0; i < nR; ++i) {
            auto w2 = nda::reshape(w(wrow[i], all, all, all), std::array<long, 2>{r, blk});
            auto W2 = nda::reshape(Wt(i, all, all, all), std::array<long, 2>{tc, blk})(tr, all);
            nda::blas::gemm(ComplexType(1.0), Ev, w2, ComplexType(0.0), W2);
          }
        }
        if constexpr (MEM != HOST_MEMORY) utils::device_sync();
        Timer.stop("Sigma_W_time");
        Timer.start("Sigma_hadamard");
        const ComplexType sN(1.0 / double(nk));
        if constexpr (MEM == HOST_MEMORY) {
          eng->prepare(ncols, {{0, -1}, {1, -1}, {1, +1}});
          ComplexType *g0 = eng->buf(0), *b = eng->buf(1);
          for (long c0 = 0; c0 < ncols; c0 += eng->cb) {
            const long wd = std::min(eng->cb, ncols - c0);
            Timer.start("Sig_ft_G");
            if (use_cache) detail::block_in_conj(g0, nk, wd, prop.ahat.data() + i0 * blk + c0, nt * blk);
            else {
              detail::block_in(g0, nk, wd, G.data() + c0, ld);
              eng->fft(0, wd, -1);
            }
            Timer.stop("Sig_ft_G");
            for (long c = 0; c < ncl; ++c) {
              auto const &cl = ibz.cls[c];
              const long nc  = cl.q_eff.size();
              const auto tcl = ftprof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
              Timer.start("Sig_ft_W");
              std::memset(static_cast<void *>(b), 0, sizeof(ComplexType) * size_t(nk * wd));
              for (long j = 0; j < nc; ++j)
                detail::block_place(b, wd, Wt.data() + ibz.rpos[cl.q_eff[j]] * ld + c0, ld, 1, &wpos[c][j], wdup[c][j] != 0);
              eng->fft(1, wd, -1);
              Timer.stop("Sig_ft_W");
              Timer.start("Sig_ft_prod");
              detail::block_mul(b, nk, wd, g0, wd);
              Timer.stop("Sig_ft_prod");
              Timer.start("Sig_ft_back");
              eng->fft(1, wd, +1);
              detail::block_out(accC.data() + c * nkI * ld + c0, ld, nkI, wd, b, kspos[c], 0, sN);
              Timer.stop("Sig_ft_back");
              if (ftprof) t_cls[c] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcl).count();
            }
          }
        } else {
          auto Gh2 = nda::reshape(Gh, std::array<long, 2>{nk, tc * blk})(all, cr);
          auto Wc2 = nda::reshape(Wc, std::array<long, 2>{nk, tc * blk})(all, cr);
          Timer.start("Sig_ft_G");
          if (use_cache) {
            for (long R = 0; R < nk; ++R)
              detail::conj_copy<MEM>(memory::array_view<MEM, ComplexType, 1>(std::array<long, 1>{n * blk}, Gh.data() + R * ld),
                                     memory::array_view<MEM, ComplexType, 1>(std::array<long, 1>{n * blk},
                                                                             prop.ahat.data() + (R * nt + i0) * blk));
          } else
            detail::fft_mesh_device(kmap.n, G.data(), Gh.data(), ncols, ld, -1);
          utils::device_sync();
          Timer.stop("Sig_ft_G");
          for (long c = 0; c < ncl; ++c) {
            auto const &cl = ibz.cls[c];
            const long nc  = cl.q_eff.size();
            const auto tcl = ftprof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            Timer.start("Sig_ft_W");
            nda::tensor::set(ComplexType(0.0), Wc);
            for (long j = 0; j < nc; ++j) {
              auto dst = Wc(wpos[c][j], tr, all, all);
              if (wdup[c][j]) nda::tensor::add(ComplexType(1.0), Wt(ibz.rpos[cl.q_eff[j]], tr, all, all), "abc", ComplexType(1.0), dst, "abc");
              else dst = Wt(ibz.rpos[cl.q_eff[j]], tr, all, all);
            }
            detail::fft_mesh_device(kmap.n, Wc.data(), Wc.data(), ncols, ld, -1);
            utils::device_sync();
            Timer.stop("Sig_ft_W");
            Timer.start("Sig_ft_prod");
            nda::tensor::elementwise(ComplexType(1.0), Gh2, ComplexType(1.0), Wc2, nda::tensor::op::MUL);
            utils::device_sync();
            Timer.stop("Sig_ft_prod");
            Timer.start("Sig_ft_back");
            detail::fft_mesh_device(kmap.n, Wc.data(), Wc.data(), ncols, ld, +1);   // N A_c at the rows kpos(ks)
            for (long k = 0; k < nkI; ++k) accC(c, k, tr, all, all) = Wc(kspos[c][k], tr, all, all);
            utils::device_sync();
            Timer.stop("Sig_ft_back");
            if (ftprof) t_cls[c] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcl).count();
          }
        }
        Timer.stop("Sigma_hadamard");
      } else {
      Timer.start("Sigma_hadamard");
      Timer.start("Sig_ft_G");
      auto G2  = nda::reshape(G, std::array<long, 2>{nk, tc * blk})(all, cr);
      auto Gh2 = nda::reshape(Gh, std::array<long, 2>{nk, tc * blk})(all, cr);
      if (use_cache) {
        for (long R = 0; R < nk; ++R)
          detail::conj_copy<MEM>(memory::array_view<MEM, ComplexType, 1>(std::array<long, 1>{n * blk}, Gh.data() + R * tc * blk),
                                 memory::array_view<MEM, ComplexType, 1>(std::array<long, 1>{n * blk},
                                                                         prop.ahat.data() + (R * nt + i0) * blk));
      } else
        nda::blas::gemm(ComplexType(1.0), FmM, G2, ComplexType(0.0), Gh2);
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("Sig_ft_G");
      Timer.stop("Sigma_hadamard");

      // 2. W(r, chunk) of the rows R from the residues (the leg's residue row)
      Timer.start("Sigma_W_time");
      {
        nda::array<ComplexType, 2> Eh = basis.time_exponentials(t, leg.s);
        auto Ev = Em(tr, all);
        Ev      = Eh;
        for (long i = 0; i < nR; ++i) {
          auto w2 = nda::reshape(w(wrow[i], all, all, all), std::array<long, 2>{r, blk});
          auto W2 = nda::reshape(Wt(i, all, all, all), std::array<long, 2>{tc, blk})(tr, all);
          nda::blas::gemm(ComplexType(1.0), Ev, w2, ComplexType(0.0), W2);
        }
      }
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("Sigma_W_time");

      // 3. per class: W_c^(R) = sum_j e^{-iQ_j R} W(q_eff_j), product with G^(R), back to the nk_ibz rows ks -> accC(c)
      //    host: blocked over the (t, P, Q) columns so that the G^ / W_c / product stripes of all classes stay in cache
      //    (the unblocked form makes ~6 passes over N x (t block) arrays per class: memory-bound); env COQUI_GWLINE_IBZ_CB
      Timer.start("Sigma_hadamard");
      if constexpr (MEM == HOST_MEMORY) {
        auto Gfull = nda::reshape(Gh, std::array<long, 2>{nk, tc * blk});
        auto Wfull = nda::reshape(Wt, std::array<long, 2>{nR, tc * blk});
        for (long c0 = 0; c0 < ncols; c0 += cb) {
          const long w  = std::min(cb, ncols - c0);
          const auto br = nda::range(c0, c0 + w);
          auto Gs       = Gfull(all, br);
          for (long c = 0; c < ncl; ++c) {
            auto const &cl = ibz.cls[c];
            const long nc  = cl.q_eff.size();
            const auto tcl = ftprof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            Timer.start("Sig_ft_W");
            auto gb        = s_gb.template view<2>({nc, w});
            for (long j = 0; j < nc; ++j) gb(j, all) = Wfull(ibz.rpos[cl.q_eff[j]], br);
            auto wc = s_wc.template view<2>({nk, w});
            nda::blas::gemm(ComplexType(1.0), Hc[c], gb, ComplexType(0.0), wc);
            Timer.stop("Sig_ft_W");
            Timer.start("Sig_ft_prod");
            for (long R = 0; R < nk; ++R) {
              ComplexType *wp = &wc(R, 0);
              ComplexType const *gp = &Gs(R, 0);
              for (long i = 0; i < w; ++i) wp[i] *= gp[i];
            }
            Timer.stop("Sig_ft_prod");
            Timer.start("Sig_ft_back");
            auto A2 = nda::reshape(accC(c, all, all, all, all), std::array<long, 2>{nkI, tc * blk})(all, br);
            nda::blas::gemm(ComplexType(1.0), Bc[c], wc, ComplexType(0.0), A2);
            Timer.stop("Sig_ft_back");
            if (ftprof) t_cls[c] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcl).count();
          }
        }
      } else {
        for (long c = 0; c < ncl; ++c) {
          auto const &cl = ibz.cls[c];
          const long nc  = cl.q_eff.size();
          const auto tcl = ftprof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
          Timer.start("Sig_ft_W");
          for (long j = 0; j < nc; ++j) Wg(j, tr, all, all) = Wt(ibz.rpos[cl.q_eff[j]], tr, all, all);
          auto Wg2 = nda::reshape(Wg, std::array<long, 2>{ncmax, tc * blk})(nda::range(nc), cr);
          auto Wc2 = nda::reshape(Wc, std::array<long, 2>{nk, tc * blk})(all, cr);
          auto A2  = nda::reshape(accC(c, all, all, all, all), std::array<long, 2>{nkI, tc * blk})(all, cr);
          nda::blas::gemm(ComplexType(1.0), Hc[c], Wg2, ComplexType(0.0), Wc2);
          utils::device_sync();
          Timer.stop("Sig_ft_W");
          Timer.start("Sig_ft_prod");
          nda::tensor::elementwise(ComplexType(1.0), Gh2, ComplexType(1.0), Wc2, nda::tensor::op::MUL);
          utils::device_sync();
          Timer.stop("Sig_ft_prod");
          Timer.start("Sig_ft_back");
          nda::blas::gemm(ComplexType(1.0), Bc[c], Wc2, ComplexType(0.0), A2);
          utils::device_sync();
          Timer.stop("Sig_ft_back");
          if (ftprof) t_cls[c] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcl).count();
        }
        utils::device_sync();
      }
      Timer.stop("Sigma_hadamard");
      }   // gemm path

      // 4. contraction with the XD slices of (class, k), accumulated over the classes
      Timer.start("Sigma_contract");
      // perf 7.5c device FFT: accC = N x the class sums (1/N here)
      const ComplexType cs = (use_fft and MEM != HOST_MEMORY) ? ComplexType(1.0 / double(nk)) : ComplexType(1.0);
      for (long c = 0; c < ncl; ++c) {
        const ComplexType beta = (c == 0) ? ComplexType(0.0) : ComplexType(1.0);
        for (long k = 0; k < nkI; ++k) {
          auto const &L  = Lc[c * nkI + k];
          auto const &Rr = Rc[c * nkI + k];
          for (long it = 0; it < n; ++it) {
            auto a = accC(c, k, it, all, all);
            auto o = part_m(k, it, all, all);
            if (right_first) {
              nda::blas::gemm(cs, a, Rr, ComplexType(0.0), u1);
              nda::blas::gemm(ComplexType(1.0), L, u1, beta, o);
            } else {
              nda::blas::gemm(cs, L, a, ComplexType(0.0), t1);
              nda::blas::gemm(ComplexType(1.0), t1, Rr, beta, o);
            }
          }
        }
      }
      if constexpr (MEM != HOST_MEMORY) utils::device_sync();
      Timer.stop("Sigma_contract");

      // 5. reduction over the grid and transform to the line nodes (as self_energy, nk_ibz rows)
      Timer.start("Sigma_allreduce");
      nda::array<ComplexType, 4> pm = memory::to_memory_space<HOST_MEMORY>(part_m);
      if (k_local) {
        const long row = n * nb * nb;
        long o         = 0;
        for (long rk = 0; rk < kd.np; ++rk)
          for (long l = 0; l < kd.nloc(rk); ++l, ++o) {
            nda::array_view<ComplexType, 3> dst(std::array<long, 3>{n, nb, nb}, kd_send.data() + o * row);
            dst = pm(kd.global(l, rk), tr, all, all);
          }
        kd_reduce_scatter(comm, kd, kd_send.data(), kd_recv.data(), row);
        Timer.stop("Sigma_allreduce");
        Timer.start("Sigma_transform");
        for (long l = 0; l < kd.nloc(); ++l) {
          nda::array_view<ComplexType, 3> P3(std::array<long, 3>{n, nb, nb}, kd_recv.data() + l * row);
          if (leg.transposed)
            for (long it = 0; it < n; ++it) P3(it, all, all) = nda::make_regular(nda::transpose(P3(it, all, all)));
          auto P2 = nda::reshape(P3, std::array<long, 2>{n, nb * nb});
          auto S2 = nda::reshape(Sigma_out(l, all, all, all), std::array<long, 2>{nz, nb * nb});
          nda::blas::gemm(alpha, F(all, nda::range(i0, i0 + n)), P2, ComplexType(1.0), S2);
        }
        Timer.stop("Sigma_transform");
        continue;
      }
      part = pm;
      comm.all_reduce_in_place_n(part.data(), part.size(), std::plus<>{});
      Timer.stop("Sigma_allreduce");
      Timer.start("Sigma_transform");
      if (leg.transposed)
        for (long k = 0; k < nkI; ++k)
          for (long it = 0; it < n; ++it) partT(k, it, all, all) = nda::transpose(part(k, it, all, all));
      auto &P = leg.transposed ? partT : part;
      for (long k = 0; k < nkI; ++k) {
        auto P2 = nda::reshape(P(k, all, all, all), std::array<long, 2>{tc, nb * nb})(tr, all);
        auto S2 = nda::reshape(Sigma_out(k, all, all, all), std::array<long, 2>{nz, nb * nb});
        nda::blas::gemm(alpha, F(all, nda::range(i0, i0 + n)), P2, ComplexType(1.0), S2);
      }
      Timer.stop("Sigma_transform");
    }
  }
  if (ftprof) {
    std::string line;
    for (long c = 0; c < ncl; ++c) line += std::to_string(ibz.cls[c].q_eff.size()) + ":" + std::to_string(t_cls[c]).substr(0, 6) + " ";
    app_log(1, "  self_energy_ibz (COQUI_GWLINE_FTPROF): class sums per class (n_c:s, rank 0, both legs): {}", line);
  }
}

} // namespace methods::gw_line

#endif
