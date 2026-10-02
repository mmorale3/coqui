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
 * Device rules (plan 6.4): the X slices, the coefficients and all outputs live in MEM; the heavy work is nda::blas::gemm
 * only (op flags for transpose / dagger of the small nb x nb C(t)); host-side setup is O(N_k nb (Np_loc + Nq_loc)).
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

  // pole data mirrored to MEM by set_poles: per k and sector, coef as (M, nb*nb); energies stay on the host
  std::vector<arr_t<2>> coef_p, coef_h;
  std::vector<nda::array<double, 1>> e_p, e_h;

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

  /// Mirror the pole coefficients to MEM (N_k M nb^2; cheap). Must be called whenever the poles change.
  void set_poles(pole_data_t const &poles) {
    utils::check(poles.nk == nk and poles.nb == nb, "gw_line::propagator_t::set_poles: pole data ({} k, {} bands) vs X ({} k, {} bands)",
                 poles.nk, poles.nb, nk, nb);
    coef_p.clear(); coef_h.clear(); e_p.clear(); e_h.clear();
    for (long ik = 0; ik < nk; ++ik) {
      for (auto s : {sector_t::particle, sector_t::hole}) {
        auto const &ps = poles(ik, s);
        nda::array<ComplexType, 2> c(ps.size(), nb * nb);
        for (long m = 0; m < ps.size(); ++m)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) c(m, i * nb + j) = ps.coef(m, i, j);
        if (s == sector_t::particle) {
          coef_p.emplace_back(memory::to_memory_space<MEM>(c));
          e_p.emplace_back(ps.e);
        } else {
          coef_h.emplace_back(memory::to_memory_space<MEM>(c));
          e_h.emplace_back(ps.e);
        }
      }
    }
  }

  long nP() const { return grid.nP; }
  long nQ() const { return grid.nQ; }

  /// out(it, :, :) = block (P_rng, Q_rng) of the requested form of G~^s(k, t_it); out: (nt, nP, nQ) in MEM.
  void build(long ik, nda::array<ComplexType, 1> const &t, sector_t s, gtilde_form_t form, view_t<3> out) const {
    utils::check(not coef_p.empty(), "gw_line::propagator_t: set_poles was not called");
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
    // C(tau) for tau = t (plain, transposed) or conj(t) (adjoint_conj_t): phases on the host, one gemm in MEM.
    const bool conj_time = (form == gtilde_form_t::adjoint_conj_t);
    nda::array<ComplexType, 2> ph_h(nt, M);
    for (long it = 0; it < nt; ++it) {
      const ComplexType tau = conj_time ? std::conj(t(it)) : t(it);
      for (long m = 0; m < M; ++m) ph_h(it, m) = std::exp(ComplexType(0.0, -e(m)) * tau);
    }
    arr_t<2> ph = memory::to_memory_space<MEM>(ph_h);
    arr_t<3> C(nt, nb, nb);
    auto C2 = nda::reshape(C, std::array<long, 2>{nt, nb * nb});
    nda::blas::gemm(ComplexType(1.0), ph, coef, ComplexType(0.0), C2);

    arr_t<2> tmp(grid.nP, nb);
    auto all = nda::range::all;
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
