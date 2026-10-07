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

#ifndef COQUI_METHODS_GW_LINE_PROPAGATORS_HPP
#define COQUI_METHODS_GW_LINE_PROPAGATORS_HPP

/**
 * Auxiliary-space propagators on a complex-time ray (notes Eq. gtilde, python LineGW.gtilde):
 *
 *   G~^s(k, t) = X(k) C^s_k(t) X(k)^dagger,   C^s_k(t) = sum_{m in s} coef_m e^{-i e_m t}   (s = particle / hole)
 *
 * computed as ONE gemm [nt x M] . [M x nb^2] for C (phases built on the host, copied to MEM) and two gemms per t for the
 * (P_rng, Q_rng) block of this rank (plan section 6.3a). Three block forms are provided, all as (P_rng, Q_rng) blocks:
 *
 *   plain          : [G~(k,t)]_PQ                  = Xp       C(t)        Xq^dagger
 *   transposed     : [G~(k,t)^T]_PQ  = G~(k,t)_QP  = conj(Xp) C(t)^T      Xq^T
 *   adjoint_conj_t : [G~(k,conj t)^dagger]_PQ      = Xp       C(conj t)^dagger Xq^dagger
 *
 * with Xp = X(k)[P_rng, :], Xq = X(k)[Q_rng, :]. The last two are what the polarization needs (Eq. pigtr/pilss carry a
 * transpose in (P,Q)); forming them directly avoids any inter-rank transpose of the block layout. None of the forms
 * assumes Hermitian residues.
 *
 * C(t) from the two residue forms of pole_sector_t (S7c):
 *   matrix coefficients : C(t) = [nt x M] phases . [M x nb^2] coefficients              (one gemm)
 *   factorized (Lehmann): C(t) = V diag(e^{-i e_m t}) V^dagger = VP(t) V^dagger,  VP(t)[a, m] = V[a, m] e^{-i e_m t}
 *                         formed on the host for the chunk ([nt, nb, M], copied once) + ONE batched gemm with V^dagger
 * (same flop count nt M nb^2; V (nb x M) is mirrored to MEM instead of M nb x nb matrices).
 *
 * Host "XV" form (S7e). For factorized poles the three forms are also
 *   plain          : L diag(ph(t)) R^dagger,   transposed : conj(L) diag(ph(t)) R^T,   adjoint_conj_t : L diag(conj ph(conj t)) R^dagger
 * with L = Xp V (nP x M), R = Xq V (nQ x M), ph(tau)_m = e^{-i e_m tau}, formed once per set_poles. Cost per t: nP nQ M
 * (one gemm, scales with the block), vs nb^2 M (C(t), the SAME on every rank: it does not scale with the number of ranks)
 * + nP nb^2 + nP nb nQ for the C(t) form. The host picks per (k, sector) the cheaper of the two by this flop count
 * (env COQUI_GWLINE_GT_XV = 1 / 0 forces XV / C(t); -1 or unset = automatic). Si 4x4x4 nb 60, M ~ 650: C(t) is cheaper on
 * <= ~200 ranks, XV beyond (3.6x fewer flops at 768 ranks). Results agree to rounding (1e-15 relative).
 *
 * Device rules (plan 6.4): the X slices, the coefficients and all outputs live in MEM; the heavy work is nda::blas::gemm
 * only (op flags for transpose / dagger of the small nb x nb C(t)); host-side setup is O(N_k nb (Np_loc + Nq_loc)).
 * On the device the per-t products are two strided-batched gemms per call (all nt times at once, the X slice broadcast);
 * intermediates live in grow-only scratch (no per-call device allocation once warm).
 */

#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/device_blas.hpp"

namespace methods::gw_line {

enum class gtilde_form_t { plain, transposed, adjoint_conj_t };

template <MEMORY_SPACE MEM>
struct propagator_t {
  template <int R> using arr_t  = memory::array<MEM, ComplexType, R>;
  template <int R> using view_t = memory::array_view<MEM, ComplexType, R>;

  aux_grid_t grid;
  long nk = 0, nb = 0;
  arr_t<3> Xp;    ///< (nk, nP, nb)  X(k)[P_rng, :]
  arr_t<3> Xpc;   ///< (nk, nP, nb)  conj(X(k)[P_rng, :])
  arr_t<3> XqH;   ///< (nk, nb, nQ)  X(k)[Q_rng, :]^dagger
  arr_t<3> XqT;   ///< (nk, nb, nQ)  X(k)[Q_rng, :]^T

  // pole data mirrored to MEM by set_poles: per k and sector, coef as (M, nb*nb) (matrix form) or V as (nb, M)
  // (factorized form; the host copy of V forms VP); energies stay on the host
  std::vector<arr_t<2>> coef_p, coef_h;
  std::vector<nda::array<double, 1>> e_p, e_h;
  std::vector<char> fac_p, fac_h;                     ///< factorized form per k (particle / hole)
  std::vector<nda::array<ComplexType, 2>> vh_p, vh_h; ///< host V (nb, M) of the factorized sectors
  bool poles_set = false;
  double pole_key = 0.0;                              ///< fingerprint of the poles of the last set_poles (A^ cache key)

  /**
   * perf 7.1 (d)+(e): the real-space transform of Pi's hole factor, A^(R, t) = sum_k e^{+ikR} G~^<(k, conj t)^dagger
   * (gtilde_form_t::adjoint_conj_t) on the particle nodes t, filled by polarization (real-space path) for ALL nodes of its
   * ray and consumed by the Sigma^< leg on the conjugated ray: there G~^<(k, conj t)^T = conj of the same block, and
   * sum_k e^{-ikR} conj(.) = conj A^(R, t), so the hole leg skips one G~ build and one transform per (k, t).
   * Valid while pole_key and the node set (ahat_t) match. Memory N_k N_t block (env COQUI_GWLINE_GT_CACHE: 0 off, 1 on,
   * -1 / unset: on when <= 1.5 GB per rank on the host, <= 15% of the free memory on the device).
   */
  memory::array<MEM, ComplexType, 4> ahat;            ///< (N_k as R, N_t, nP, nQ)
  nda::array<ComplexType, 1> ahat_t;                  ///< the particle nodes of ahat
  double ahat_key = -1.0;
  long ahat_filled = 0;                               ///< nodes filled (valid only when == ahat_t.size())
  // host XV form (S7e): L = Xp V (nP, M), R = Xq V (nQ, M) per k and sector (empty where the C(t) form is used)
  std::vector<nda::array<ComplexType, 2>> L_p, L_h, R_p, R_h;
  long n_xv = 0;                                      ///< (k, sector) pairs using the XV form

  // grow-only MEM scratch of build(): phases (or VP), C(t), and the intermediate (host: Xp C; device: C Xq^dagger for all t)
  mutable detail::scratch_t<MEM> s_ph, s_C, s_T;
  mutable detail::scratch_t<HOST_MEMORY> s_VPh;       ///< host staging of VP (device builds only)

  /// X slices of all k for the block of `grid`, from the collinear THC collocation matrices thc.X(0, 0, ik).
  propagator_t(methods::thc_reader_t const &thc, aux_grid_t const &grid_) : grid(grid_) {
    utils::check(thc.ns() == 1 and thc.npol() == 1, "gw_line::propagator_t: spin-restricted collinear only (ns={}, npol={})",
                 thc.ns(), thc.npol());
    utils::check(thc.Np() == grid.Np, "gw_line::propagator_t: grid Np {} != thc Np {}", grid.Np, thc.Np());
    nk = thc.nkpts();
    nb = thc.nbnd();
    const long nP = grid.nP, nQ = grid.nQ, P0 = grid.P0, Q0 = grid.Q0;
    nda::array<ComplexType, 3> xp(nk, nP, nb), xpc(nk, nP, nb), xqh(nk, nb, nQ), xqt(nk, nb, nQ);
    for (long ik = 0; ik < nk; ++ik) {
      auto X = thc.X(0, 0, ik);   // (Np, nb), host view into the node-shared array
      utils::check(X.extent(1) == nb, "gw_line::propagator_t: X(k) has {} columns, nbnd = {}", X.extent(1), nb);
      for (long P = 0; P < nP; ++P)
        for (long i = 0; i < nb; ++i) {
          xp(ik, P, i)  = X(P0 + P, i);
          xpc(ik, P, i) = std::conj(X(P0 + P, i));
        }
      for (long i = 0; i < nb; ++i)
        for (long Q = 0; Q < nQ; ++Q) {
          xqt(ik, i, Q) = X(Q0 + Q, i);
          xqh(ik, i, Q) = std::conj(X(Q0 + Q, i));
        }
    }
    Xp  = memory::to_memory_space<MEM>(xp);
    Xpc = memory::to_memory_space<MEM>(xpc);
    XqH = memory::to_memory_space<MEM>(xqh);
    XqT = memory::to_memory_space<MEM>(xqt);
  }

  /// Mirror the pole residues to MEM (matrix form N_k M nb^2, factorized N_k nb M; cheap). Must be called whenever the
  /// poles change.
  void set_poles(pole_data_t const &poles) {
    utils::check(poles.nk == nk and poles.nb == nb, "gw_line::propagator_t::set_poles: pole data ({} k, {} bands) vs X ({} k, {} bands)",
                 poles.nk, poles.nb, nk, nb);
    coef_p.clear(); coef_h.clear(); e_p.clear(); e_h.clear();
    fac_p.clear(); fac_h.clear(); vh_p.clear(); vh_h.clear();
    for (long ik = 0; ik < nk; ++ik) {
      for (auto s : {sector_t::particle, sector_t::hole}) {
        auto const &ps  = poles(ik, s);
        const bool part = (s == sector_t::particle);
        if (ps.is_factorized()) {
          utils::check(ps.v.extent(0) == nb, "gw_line::propagator_t::set_poles: V has {} rows, nb = {}", ps.v.extent(0), nb);
          (part ? coef_p : coef_h).emplace_back(memory::to_memory_space<MEM>(ps.v));
          (part ? vh_p : vh_h).emplace_back(ps.v);
        } else {
          nda::array<ComplexType, 2> c(ps.size(), nb * nb);
          for (long m = 0; m < ps.size(); ++m)
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j < nb; ++j) c(m, i * nb + j) = ps.coef(m, i, j);
          (part ? coef_p : coef_h).emplace_back(memory::to_memory_space<MEM>(c));
          (part ? vh_p : vh_h).emplace_back();
        }
        (part ? fac_p : fac_h).push_back(ps.is_factorized() ? 1 : 0);
        (part ? e_p : e_h).emplace_back(ps.e);
      }
    }
    // host XV form where it is cheaper (see the file header)
    L_p.assign(nk, {}); L_h.assign(nk, {}); R_p.assign(nk, {}); R_h.assign(nk, {});
    n_xv = 0;
    if constexpr (MEM == HOST_MEMORY) {
      const long mode = detail::env_long("COQUI_GWLINE_GT_XV", -1);
      const double nP = double(grid.nP), nQ = double(grid.nQ), b = double(nb);
      auto all        = nda::range::all;
      for (long ik = 0; ik < nk; ++ik)
        for (auto s : {sector_t::particle, sector_t::hole}) {
          const bool part = (s == sector_t::particle);
          if (not((part ? fac_p : fac_h)[ik])) continue;
          auto const &V  = (part ? vh_p : vh_h)[ik];
          const double M = double(V.extent(1));
          const bool xv  = (mode == 1) or (mode < 0 and nP * nQ * M + nP * M < b * b * M + nP * b * b + nP * b * nQ);
          if (not xv or V.extent(1) == 0) continue;
          nda::array<ComplexType, 2> L(grid.nP, V.extent(1)), R(grid.nQ, V.extent(1));
          nda::blas::gemm(ComplexType(1.0), Xp(ik, all, all), V, ComplexType(0.0), L);
          nda::blas::gemm(ComplexType(1.0), nda::transpose(XqT(ik, all, all)), V, ComplexType(0.0), R);
          (part ? L_p : L_h)[ik] = std::move(L);
          (part ? R_p : R_h)[ik] = std::move(R);
          ++n_xv;
        }
    }
    poles_set = true;
    {   // fingerprint of the pole data (energies and residues), for the A^ cache
      double key = double(nk);
      for (long ik = 0; ik < nk; ++ik) {
        for (auto const *e : {&e_p[ik], &e_h[ik]})
          for (long m = 0; m < e->size(); ++m) key += (*e)(m) * double(m + 1 + 7 * ik);
        auto const &ps = poles(ik, sector_t::particle), &ph = poles(ik, sector_t::hole);
        for (auto const *pp : {&ps, &ph}) {
          if (pp->is_factorized())
            for (auto const &x : pp->v) key += std::abs(x) * 1.000001;
          else
            for (auto const &x : pp->coef) key += std::abs(x) * 0.999999;
        }
      }
      if (key != pole_key) ahat_key = -1.0;   // new poles: the cache is stale
      pole_key = key;
    }
  }

  /// bytes of the host XV factors
  double xv_bytes() const {
    double b = 0.0;
    for (auto const *v : {&L_p, &L_h, &R_p, &R_h})
      for (auto const &c : *v) b += 16.0 * double(c.size());
    return b;
  }

  /// bytes of the pole residues mirrored to MEM
  double pole_bytes() const {
    double b = 0.0;
    for (auto const *v : {&coef_p, &coef_h})
      for (auto const &c : *v) b += 16.0 * double(c.size());
    return b;
  }

  long nP() const { return grid.nP; }
  long nQ() const { return grid.nQ; }

  /// out(it, :, :) = block (P_rng, Q_rng) of the requested form of G~^s(k, t_it); out: (nt, nP, nQ) in MEM.
  void build(long ik, nda::array<ComplexType, 1> const &t, sector_t s, gtilde_form_t form, view_t<3> out) const {
    utils::check(poles_set, "gw_line::propagator_t: set_poles was not called");
    utils::check(s != sector_t::both, "gw_line::propagator_t: sector must be particle or hole");
    const long nt = t.size();
    utils::check(out.extent(0) == nt and out.extent(1) == grid.nP and out.extent(2) == grid.nQ,
                 "gw_line::propagator_t::build: out shape mismatch");
    auto const &e    = (s == sector_t::particle) ? e_p[ik] : e_h[ik];
    auto const &coef = (s == sector_t::particle) ? coef_p[ik] : coef_h[ik];
    const long M     = e.size();
    if (M == 0) {
      nda::tensor::set(ComplexType(0.0), out);
      return;
    }
    if constexpr (MEM == HOST_MEMORY) {
      auto const &Lx = (s == sector_t::particle) ? L_p[ik] : L_h[ik];
      if (Lx.size() > 0) {   // XV form (S7e): out(t) = A(t) B^op, A(t) = (conj) L diag(phases), B = R
        auto const &Rx = (s == sector_t::particle) ? R_p[ik] : R_h[ik];
        auto all       = nda::range::all;
        auto A         = s_T.template view<2>({grid.nP, M});
        for (long it = 0; it < nt; ++it) {
          switch (form) {
            case gtilde_form_t::plain: {   // L diag(e^{-i e t}) R^dagger
              for (long m = 0; m < M; ++m) {
                const ComplexType ph = std::exp(ComplexType(0.0, -e(m)) * t(it));
                for (long P = 0; P < grid.nP; ++P) A(P, m) = Lx(P, m) * ph;
              }
              nda::blas::gemm(ComplexType(1.0), A, nda::dagger(Rx), ComplexType(0.0), out(it, all, all));
              break;
            }
            case gtilde_form_t::transposed: {   // conj(L) diag(e^{-i e t}) R^T
              for (long m = 0; m < M; ++m) {
                const ComplexType ph = std::exp(ComplexType(0.0, -e(m)) * t(it));
                for (long P = 0; P < grid.nP; ++P) A(P, m) = std::conj(Lx(P, m)) * ph;
              }
              nda::blas::gemm(ComplexType(1.0), A, nda::transpose(Rx), ComplexType(0.0), out(it, all, all));
              break;
            }
            case gtilde_form_t::adjoint_conj_t: {   // L diag(conj e^{-i e conj t}) R^dagger
              for (long m = 0; m < M; ++m) {
                const ComplexType ph = std::conj(std::exp(ComplexType(0.0, -e(m)) * std::conj(t(it))));
                for (long P = 0; P < grid.nP; ++P) A(P, m) = Lx(P, m) * ph;
              }
              nda::blas::gemm(ComplexType(1.0), A, nda::dagger(Rx), ComplexType(0.0), out(it, all, all));
              break;
            }
          }
        }
        return;
      }
    }
    // C(tau) for tau = t (plain, transposed) or conj(t) (adjoint_conj_t): phases on the host, one (batched) gemm in MEM.
    // All MEM intermediates are views of grow-only scratch buffers (no allocation per call once warm).
    const bool conj_time = (form == gtilde_form_t::adjoint_conj_t);
    const bool fact      = (s == sector_t::particle) ? fac_p[ik] != 0 : fac_h[ik] != 0;
    auto C               = s_C.template view<3>({nt, nb, nb});
    nda::array<ComplexType, 2> ph_h(nt, M);
    for (long it = 0; it < nt; ++it) {
      const ComplexType tau = conj_time ? std::conj(t(it)) : t(it);
      for (long m = 0; m < M; ++m) ph_h(it, m) = std::exp(ComplexType(0.0, -e(m)) * tau);
    }
    if (fact) {
      // factorized: VP(it)[a, m] = V[a, m] ph(it, m) on the host, copied once; C(it) = VP(it) V^dagger
      auto const &Vh = (s == sector_t::particle) ? vh_p[ik] : vh_h[ik];
      auto fill_vp   = [&](auto &&VPh) {
        for (long it = 0; it < nt; ++it)
          for (long a = 0; a < nb; ++a)
            for (long m = 0; m < M; ++m) VPh(it, a, m) = Vh(a, m) * ph_h(it, m);
      };
      auto VP = s_ph.template view<3>({nt, nb, M});
      if constexpr (MEM == HOST_MEMORY) {
        fill_vp(VP);
        for (long it = 0; it < nt; ++it)
          nda::blas::gemm(ComplexType(1.0), VP(it, nda::range::all, nda::range::all), nda::dagger(coef), ComplexType(0.0),
                          C(it, nda::range::all, nda::range::all));
      } else {
        auto VPh = s_VPh.template view<3>({nt, nb, M});
        fill_vp(VPh);
        VP = VPh;
        // column-major view (device_blas.hpp): C(it)^T = conj(V) VP(it)^T; V (nb x M, C layout) is the column-major
        // M x nb matrix V^T (ld M), op 'C' -> conj(V); VP(it) is the column-major VP(it)^T (M x nb, ld M)
        detail::gemm_strided_cm('C', 'N', nb, nb, M, ComplexType(1.0), coef.data(), M, 0, VP.data(), M, nb * M,
                                ComplexType(0.0), C.data(), nb, nb * nb, nt);
      }
    } else {
      auto ph = s_ph.template view<2>({nt, M});
      ph      = ph_h;
      auto C2 = nda::reshape(C, std::array<long, 2>{nt, nb * nb});
      nda::blas::gemm(ComplexType(1.0), ph, coef, ComplexType(0.0), C2);
    }

    auto all = nda::range::all;
    if constexpr (MEM == HOST_MEMORY) {
      auto tmp = s_T.template view<2>({grid.nP, nb});
      for (long it = 0; it < nt; ++it) {
        auto Ct = C(it, all, all);
        auto ot = out(it, all, all);
        switch (form) {
          case gtilde_form_t::plain:
            nda::blas::gemm(ComplexType(1.0), Xp(ik, all, all), Ct, ComplexType(0.0), tmp);
            nda::blas::gemm(ComplexType(1.0), tmp, XqH(ik, all, all), ComplexType(0.0), ot);
            break;
          case gtilde_form_t::transposed:
            nda::blas::gemm(ComplexType(1.0), Xpc(ik, all, all), nda::transpose(Ct), ComplexType(0.0), tmp);
            nda::blas::gemm(ComplexType(1.0), tmp, XqT(ik, all, all), ComplexType(0.0), ot);
            break;
          case gtilde_form_t::adjoint_conj_t:
            nda::blas::gemm(ComplexType(1.0), Xp(ik, all, all), nda::dagger(Ct), ComplexType(0.0), tmp);
            nda::blas::gemm(ComplexType(1.0), tmp, XqH(ik, all, all), ComplexType(0.0), ot);
            break;
        }
      }
    } else {
      // device: two strided-batched gemms over the nt times (the X slice broadcast with stride 0), column-major view of
      // the C-layout blocks (see device_blas.hpp):
      //   T(t)   = op(C(t)) Xq'     ->  T(t)^T   = Xq'^T op'(C(t)^T):  plain  C XqH   (op 'N'), transposed  C^T XqT ('T'),
      //                                                              adjoint C^dagger XqH ('C')
      //   out(t) = Xp' T(t)         ->  out(t)^T = T(t)^T Xp'^T        (Xp' = Xp, or conj(Xp) for transposed)
      const long nP = grid.nP, nQ = grid.nQ;
      utils::check(out.indexmap().strides()[1] == nQ and out.indexmap().strides()[2] == 1,
                   "gw_line::propagator_t::build: out blocks must be contiguous");
      auto T           = s_T.template view<3>({nt, nb, nQ});
      const char opC   = (form == gtilde_form_t::plain) ? 'N' : (form == gtilde_form_t::transposed ? 'T' : 'C');
      auto const &Xq_  = (form == gtilde_form_t::transposed) ? XqT : XqH;
      auto const &Xp_  = (form == gtilde_form_t::transposed) ? Xpc : Xp;
      ComplexType const *xq = Xq_.data() + ik * nb * nQ;
      ComplexType const *xp = Xp_.data() + ik * nP * nb;
      detail::gemm_strided_cm('N', opC, nQ, nb, nb, ComplexType(1.0), xq, nQ, 0, C.data(), nb, nb * nb, ComplexType(0.0),
                              T.data(), nQ, nb * nQ, nt);
      detail::gemm_strided_cm('N', 'N', nQ, nP, nb, ComplexType(1.0), T.data(), nQ, nb * nQ, xp, nb, 0, ComplexType(0.0),
                              out.data(), nQ, out.indexmap().strides()[0], nt);
      device_mem_probe();
    }
  }

  /// Eq. gtilde: out(it) = block (P_rng, Q_rng) of G~^s(k, t_it) = X(k) C^s_k(t_it) X(k)^dagger.
  void gtilde(long ik, nda::array<ComplexType, 1> const &t, sector_t s, view_t<3> out) const {
    build(ik, t, s, gtilde_form_t::plain, out);
  }
};

extern template struct propagator_t<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
extern template struct propagator_t<DEVICE_MEMORY>;
#endif

} // namespace methods::gw_line

#endif
