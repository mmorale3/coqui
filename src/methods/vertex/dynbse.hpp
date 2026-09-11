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

#ifndef COQUI_VERTEX_DYNBSE_HPP
#define COQUI_VERTEX_DYNBSE_HPP

/**
 * scGW-tilde Tier 2, full frequency: the resummed DYNAMIC-rung Bethe-Salpeter polarization
 * (notes/dynbse_plan.md, increment D1 -- the algebra; D2 puts the THC rung and the
 * parallel driver around it).
 *
 * THE OBJECT (pair space of vertex_ladder.icc: pair r = (a nc + b), a on the k line, b on the
 * k+q line; the loop frequency iw kept explicit; the external (q, inu) fixed per unit):
 *
 *   L0(k, iw)_{(p1'p3),(ab)} = G_{a p1'}(k, iw) G_{p3 b}(k+q, iw+inu)
 *   Gamma(k, iw) = D + sum_{k',iw'} K(k,k'; iw-iw') L0(k',iw') Gamma(k',iw')
 *   Ptilde       = sum_k D(k)^dag (1/beta) sum_iw L0(k,iw) Gamma(k,iw)
 *
 * THE REPRESENTATION. With G in DLR pole form, G(k,z) = sum_j g_j(k)/(z - e_j), every function
 * of the loop frequency in the ladder has poles in exactly two families,
 *
 *   U_a(z) = 1/(z - eps_a)   (unshifted)     S_a(z) = 1/(z + inu - eps_a)   (shifted by -inu),
 *
 * on a fixed node set eps (the DLR aux grid of the vertex's own IAFT instance). In tau the
 * shifted family carries the phase e^{inu s}: S_a <-> e^{inu s} K_F(s, eps_a). The class is
 * closed under the ladder's two operations:
 *   * multiplication by L0: partial fractions of simple poles (closed form);
 *       1/((z-e_j)(z+inu-e_l)) = [U_j - S_l]/(e_j - e_l + inu)
 *       U_j U_a = [U_j - U_a]/(e_j - e_a) (a != j),  U_a^2 -> the double-pole table Dsq
 *       S_l S_a = [S_l - S_a]/(e_l - e_a) (a != l),  S_a^2 -> Dsq (same function of z + inu)
 *       U_j S_a = [U_j - S_a]/(e_j - e_a + inu),     S_l U_a = [U_a - S_l]/(e_a - e_l + inu)
 *     where Dsq re-expands U_a^2 on the node set through the regularized tau fit of its exact
 *     tau function, d/de K_F(s,e) = -K_F(s,e) [s - beta f(e)] (the parent's degenerate twisted
 *     pair, vertex_pi.icc build_kappa). At inu = 0 the families coincide (S = U): one family.
 *   * convolution with the rung W (frequency convolution = tau product): the smooth parts of
 *     both families are evaluated on the DLR tau grid, multiplied by W(s), and refit; the
 *     e^{inu s} factor of the shifted family rides along untouched.
 * The frequency SUMS the readout and the instantaneous rung need are closed form (the T-family
 * of ward_legs.hpp) and are taken from the PRODUCT form inside l0_apply, never from the
 * re-expanded coefficients (whose 1/w tails cancel only approximately).
 *
 * THE SOLVE. K = K_s + K_d with K_s the static (inu'-independent) rung W0bar and
 * K_d(inu') = W_dyn(inu') - W_dyn(0) the remainder (K_d(0) = 0). The static ladder is closed
 * form and separable in frequency, L_s = L0 + L0 T_s L0 with T_s = K_s (1 - Cb K_s)^-1 in the
 * D = nk nc^2 pair space (the existing dense static resolvent), so the vertex solves the Dyson
 * equation in the dynamic remainder,
 *
 *   Gamma = L_s (D + y),   y = K_d Gamma,    y = (two families) + (a frequency constant:
 *                                              -W_dyn(0) sum_iw Gamma).
 * Iterated as a Neumann series on y with Anderson(2) acceleration; the first iterate is the
 * "static-dressed one dynamic rung". Everything here is pure algebra on residue arrays and
 * explicit pair-space rung matrices (the toy / gate form); the THC rung with the k-FFT and the
 * node-shared W(s) are the production driver's business (D2).
 *
 * Conventions pinned by the gates: dense Matsubara BSE oracle (G-C), the L2 static resolvent
 * (G-A) and pi_c_accumulate_w's one rung (G-B).
 */

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
#include "methods/vertex/ward_legs.hpp"

namespace methods {
namespace solvers {
namespace dynbse {

  using cplx = ComplexType;
  using ward_legs::fermi_derivs;
  using ward_legs::fermi_all;

  template<class A>
  inline long nan_count(A const &a) {
    long n = 0;
    for (auto const &v : a) if (std::isnan(v.real()) or std::isnan(v.imag())) ++n;
    return n;
  }

  /** stable n_B(e) = 1/(exp(beta e) - 1), e != 0 */
  inline double nB(double beta, double e) {
    if (e > 0.0) {
      const double x = std::exp(-beta * e);
      return x / (1.0 - x);
    }
    return -1.0 / (1.0 - std::exp(beta * e));
  }

  /** bosonic tau kernel: (1/beta) sum_n e^{-i nu_n s}/(i nu_n - e) = -e^{-e s}/(1 - e^{-beta e}), 0 < s < beta */
  inline double KB(double beta, double s, double e) {
    if (e > 0.0) return -std::exp(-e * s) / (1.0 - std::exp(-beta * e));
    return std::exp(e * (beta - s)) / (1.0 - std::exp(beta * e));   // = -e^{-es}/(1-e^{-beta e}) for e < 0
  }

  // ==================================================================================
  // THE FREQUENCY BASIS
  // ==================================================================================

  struct freq_basis {
    double beta = 0.0;
    long np = 0, nt = 0;
    long np_fit = 0;                      // the first np_fit nodes are the DLR set (the refit target);
                                          // nodes >= np_fit are exact extra poles (union set, tests)
    nda::array<double, 1> eps;            // (np) the node set
    nda::array<double, 1> s;              // (nt) backend tau values
    nda::array<double, 2> KF;             // (nt, np) K_F(s_i, eps_a)
    std::vector<fermi_derivs> fd;         // f, f', f'', f''' at the nodes
    nda::array<double, 1> fhalf;          // (np) f(eps_a) - 1/2 (the symmetric-sum weight)
    nda::array<cplx, 2> Dsq;              // (np, np): U_a^2 ~ sum_c Dsq(a, c) U_c
    nda::array<cplx, 2> Dcb;              // (np, np): U_a^3 ~ sum_c Dcb(a, c) U_c
    double dsq_fit_err = 0.0;             // max refit error of the double/triple-pole tau functions
    imag_axes_ft::dlr_pole_fit pf;        // the regularized tau -> coefficient map
  };

  inline freq_basis build_freq_basis(imag_axes_ft::IAFT const &ft) {
    freq_basis b;
    b.pf.build(ft);
    b.beta = ft.beta();
    b.np = b.pf.np;
    b.np_fit = b.np;
    b.nt = b.pf.nt;
    b.eps = b.pf.epsl;
    b.s = b.pf.s_phys;
    b.KF = b.pf.Kmat;
    b.fd.resize(size_t(b.np));
    b.fhalf = nda::array<double, 1>(b.np);
    for (long a = 0; a < b.np; ++a) {
      b.fd[size_t(a)] = fermi_all(b.beta, b.eps(a));
      b.fhalf(a) = b.fd[size_t(a)].f - 0.5;
    }
    // the double-pole table: tau function of U_a^2 is d/de K_F = -K_F(s,e) [s - beta f(e)];
    // the triple-pole table: tau function of U_a^3 is (1/2) d^2/de^2 K_F
    //                       = (1/2) K_F(s,e) [ (s - beta f(e))^2 + beta f'(e) ]
    nda::array<cplx, 2> F2(b.nt, b.np), F3(b.nt, b.np);
    for (long a = 0; a < b.np; ++a)
      for (long i = 0; i < b.nt; ++i) {
        const double u = b.s(i) - b.beta * b.fd[size_t(a)].f;
        F2(i, a) = cplx(-b.KF(i, a) * u);
        F3(i, a) = cplx(0.5 * b.KF(i, a) * (u * u + b.beta * b.fd[size_t(a)].f1));
      }
    auto c2 = b.pf.coeffs(F2);                          // (np, np): column a = coefficients of U_a^2
    auto c3 = b.pf.coeffs(F3);
    b.dsq_fit_err = std::max(b.pf.fit_error(F2, c2), b.pf.fit_error(F3, c3));
    b.Dsq = nda::array<cplx, 2>(b.np, b.np);
    b.Dcb = nda::array<cplx, 2>(b.np, b.np);
    for (long a = 0; a < b.np; ++a)
      for (long cc = 0; cc < b.np; ++cc) {
        b.Dsq(a, cc) = c2(cc, a);
        b.Dcb(a, cc) = c3(cc, a);
      }
    return b;
  }

  /**
   * Append exact extra poles to the node set (the union set of the exact-pole tests): the refit
   * still targets the first np_fit (DLR) nodes; the extension carries K_F columns and Fermi data
   * but no double/triple-pole tables (a confluent product on an extension node is an error).
   */
  inline void extend_freq_basis(freq_basis &b, nda::array<double, 1> const &extra) {
    const long ne = extra.shape(0), np0 = b.np, np1 = np0 + ne;
    nda::array<double, 1> eps(np1), fhalf(np1);
    nda::array<double, 2> KF(b.nt, np1);
    nda::array<cplx, 2> Dsq(np1, np1), Dcb(np1, np1);
    Dsq() = cplx(0.0);
    Dcb() = cplx(0.0);
    for (long a = 0; a < np0; ++a) {
      eps(a) = b.eps(a);
      fhalf(a) = b.fhalf(a);
      KF(nda::range::all, a) = b.KF(nda::range::all, a);
      Dsq(a, nda::range(np0)) = b.Dsq(a, nda::range::all);
      Dcb(a, nda::range(np0)) = b.Dcb(a, nda::range::all);
    }
    for (long e = 0; e < ne; ++e) {
      const long a = np0 + e;
      eps(a) = extra(e);
      b.fd.push_back(fermi_all(b.beta, extra(e)));
      fhalf(a) = b.fd[size_t(a)].f - 0.5;
      for (long i = 0; i < b.nt; ++i) KF(i, a) = imag_axes_ft::dlr_kF(b.beta, b.s(i), extra(e));
    }
    b.np = np1;
    b.eps = std::move(eps);
    b.fhalf = std::move(fhalf);
    b.KF = std::move(KF);
    b.Dsq = std::move(Dsq);
    b.Dcb = std::move(Dcb);
  }

  // ==================================================================================
  // THE UNIT: one external (q, inu); pair propagator residues and the pair-space rung
  // ==================================================================================

  /**
   * Pole data of the pair propagator for one unit. gk(j, k, :, :) = g_j(k) (nc x nc, the C x C
   * block), gkq(l, k, :, :) = g_l(k+q); epsG the G nodes (shared with the vertex basis or not).
   * The pair-space action of the (j, l) term on a pair matrix V (row a on the k line, column b
   * on the k+q line): V -> g_j(k)^T V g_l(k+q)^T  [(L0 v)_{(p1'p3)} = sum_ab g_{a p1'} v_ab g_{p3 b}].
   */
  struct pair_poles {
    long nk = 0, nc = 0, ng = 0;
    nda::array<double, 1> epsG;                 // (ng)
    nda::array<long, 1> gnode;                  // (ng) the vertex-basis node index of each G pole
    nda::array<cplx, 4> gk, gkq;                // (ng, nk, nc, nc)
    std::vector<fermi_derivs> fdG;
  };

  /**
   * gnode0: the vertex node index of G pole 0 (the G poles occupy gnode0 .. gnode0 + ng - 1 of
   * the vertex node set): 0 for the shared (production) set, np_fit for the union set built by
   * extend_freq_basis.
   */
  inline pair_poles make_pair_poles(double beta, nda::array<double, 1> const &epsG,
                                    nda::array<cplx, 4> const &gk, nda::array<cplx, 4> const &gkq,
                                    long gnode0 = 0) {
    pair_poles p;
    p.ng = epsG.shape(0);
    p.nk = gk.shape(1);
    p.nc = gk.shape(2);
    p.epsG = epsG;
    p.gk = gk;
    p.gkq = gkq;
    p.fdG.resize(size_t(p.ng));
    p.gnode = nda::array<long, 1>(p.ng);
    for (long j = 0; j < p.ng; ++j) {
      p.fdG[size_t(j)] = fermi_all(beta, epsG(j));
      p.gnode(j) = gnode0 + j;
    }
    return p;
  }

  /**
   * A two-family vector with a frequency-constant part, for all k, pairs (matrix form) and
   * right-hand sides: fam(2, np, nk, nc, nc, nR) + cst(nk, nc, nc, nR).
   */
  struct tf_vector {
    long np = 0, nk = 0, nc = 0, nR = 0;
    nda::array<cplx, 6> fam;
    nda::array<cplx, 4> cst;
    tf_vector() = default;
    tf_vector(long np_, long nk_, long nc_, long nR_)
        : np(np_), nk(nk_), nc(nc_), nR(nR_), fam(2, np_, nk_, nc_, nc_, nR_), cst(nk_, nc_, nc_, nR_) {
      fam() = cplx(0.0);
      cst() = cplx(0.0);
    }
    void zero() { fam() = cplx(0.0); cst() = cplx(0.0); }
    double norm2() const {
      double n = 0.0;
      for (auto const &v : fam) n += std::norm(v);
      for (auto const &v : cst) n += std::norm(v);
      return n;
    }
  };

  /**
   * F = L0 X (X = cst + two families). Output F is pure two-family (cst = 0); Fsum(k,:,:,r)
   * = (1/beta) sum_iw F(k, iw) EXACT from the product form (ward_legs T-sums), so the readout
   * and the instantaneous rung never see the re-expanded tails.
   *
   * The G poles are vertex nodes through P.gnode (the shared production set: identity; the
   * exact tests: the union extension). A product confluent on one node (X carrying a component
   * at a G node, or U_j^2 at inu = 0) is re-expanded with the Dsq/Dcb tables, which exist for
   * the DLR part of the node set only. `shared` is informational (checks the gnode map). At
   * inu = 0 the two families coincide: the shifted input is folded into the unshifted one and
   * only family 0 is produced.
   */
  inline void l0_apply(freq_basis const &b, pair_poles const &P, cplx inu, bool shared,
                       tf_vector const &X, tf_vector &F, nda::array<cplx, 4> &Fsum) {
    decltype(nda::range::all) all;
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR;
    utils::check(X.np == np and X.nk == nk and X.nc == nc, "dynbse::l0_apply: shape mismatch.");
    utils::check(not shared or (ng == np and P.gnode(0) == 0), "dynbse::l0_apply: shared node sets differ.");
    utils::check(P.gnode(ng - 1) < np, "dynbse::l0_apply: G node map exceeds the vertex node set.");
    const bool nu0 = (inu == cplx(0.0));
    F.zero();
    Fsum() = cplx(0.0);

    nda::array<cplx, 2> T(nc, nc), A(nc, nc), Vtmp(nc, nc);
    auto pair_map = [&](long j, long l, long ik, nda::MemoryArrayOfRank<2> auto const &V,
                        nda::array<cplx, 2> &out) {
      // out = g_j(k)^T V g_l(k+q)^T
      for (long a = 0; a < nc; ++a)
        for (long p1p = 0; p1p < nc; ++p1p) {
          cplx s(0.0);
          for (long aa = 0; aa < nc; ++aa) s += P.gk(j, ik, aa, p1p) * V(aa, a);
          T(p1p, a) = s;
        }
      bool any = false;
      for (long p1p = 0; p1p < nc; ++p1p)
        for (long p3 = 0; p3 < nc; ++p3) {
          cplx s(0.0);
          for (long bb = 0; bb < nc; ++bb) s += T(p1p, bb) * P.gkq(l, ik, p3, bb);
          out(p1p, p3) = s;
          any = any or (s != cplx(0.0));
        }
      return any;
    };
    // emit one output term: family fam (0 = U, 1 = S; at inu = 0 everything is family 0),
    // node `node` of the set `onG` (true: a G node -- only meaningful when shared, where it
    // is also a vertex node; false: a vertex node), multiplicity m in {1,2,3}, scalar coef,
    // pair matrix M. Also accumulates the exact frequency sum of the term.
    auto emit = [&](long fam, long node, int m, cplx coef, long ik, long r, nda::array<cplx, 2> const &M,
                    double e, fermi_derivs const &fdn) {
      if (coef == cplx(0.0)) return;
      const long f = nu0 ? 0 : fam;
      // frequency sum: (1/beta) sum 1/(z-e) = f - 1/2 (symmetric; the 1/z tails cancel in
      // the total), 1/(z-e)^2 -> f', 1/(z-e)^3 -> f''/2 -- identical for the shifted family
      const cplx wsum = coef * ((m == 1) ? cplx(fdn.f - 0.5) : (m == 2) ? cplx(fdn.f1) : cplx(0.5 * fdn.f2));
      (void)e;
      for (long i = 0; i < nc; ++i)
        for (long jj = 0; jj < nc; ++jj) Fsum(ik, i, jj, r) += wsum * M(i, jj);
      if (m == 1) {
        for (long i = 0; i < nc; ++i)
          for (long jj = 0; jj < nc; ++jj) F.fam(f, node, ik, i, jj, r) += coef * M(i, jj);
        return;
      }
      utils::check(node < b.np_fit, "dynbse::l0_apply: a confluent pole of multiplicity {} at node {} outside the DLR set.", m, node);
      auto const &Dt = (m == 2) ? b.Dsq : b.Dcb;
      for (long c = 0; c < np; ++c) {
        const cplx wc = coef * Dt(node, c);
        if (wc == cplx(0.0)) continue;
        for (long i = 0; i < nc; ++i)
          for (long jj = 0; jj < nc; ++jj) F.fam(f, c, ik, i, jj, r) += wc * M(i, jj);
      }
    };

    for (long ik = 0; ik < nk; ++ik)
      for (long r = 0; r < nR; ++r)
        for (long j = 0; j < ng; ++j)
          for (long l = 0; l < ng; ++l) {
            const double ej = P.epsG(j), el = P.epsG(l);
            auto const &fj = P.fdG[size_t(j)];
            auto const &fl = P.fdG[size_t(l)];
            const long nj = P.gnode(j), nl = P.gnode(l);      // vertex node indices of the G poles
            const bool jl_conf = nu0 and (j == l);           // U_j^2 (same node, same set)
            const cplx Djl = cplx(ej - el) + inu;             // [U_j - S_l]/Djl otherwise
            // ---- the constant part -----------------------------------------------------
            if (pair_map(j, l, ik, X.cst(ik, all, all, r), A)) {
              if (jl_conf) emit(0, nj, 2, cplx(1.0), ik, r, A, ej, fj);
              else {
                emit(0, nj, 1, cplx(1.0) / Djl, ik, r, A, ej, fj);
                emit(1, nl, 1, cplx(-1.0) / Djl, ik, r, A, el, fl);
              }
            }
            // ---- the two families -------------------------------------------------------
            for (long fam = 0; fam < 2; ++fam) {
              if (nu0 and fam == 1) break;      // folded below
              for (long a = 0; a < np; ++a) {
                for (long i = 0; i < nc; ++i)
                  for (long jj = 0; jj < nc; ++jj)
                    Vtmp(i, jj) = X.fam(fam, a, ik, i, jj, r) + ((nu0) ? X.fam(1, a, ik, i, jj, r) : cplx(0.0));
                bool anyv = false;
                for (auto const &v : Vtmp) anyv = anyv or (v != cplx(0.0));
                if (not anyv) continue;
                if (not pair_map(j, l, ik, Vtmp, A)) continue;
                const double ea = b.eps(a);
                auto const &fa = b.fd[size_t(a)];
                const bool aj = (a == nj), al = (a == nl);
                if (jl_conf) {
                  // U_j^2 x U_a  (inu = 0 only)
                  if (aj) emit(0, nj, 3, cplx(1.0), ik, r, A, ej, fj);
                  else {
                    const cplx dd = cplx(ea - ej);
                    emit(0, nj, 2, cplx(1.0) / cplx(ej - ea), ik, r, A, ej, fj);
                    emit(0, nj, 1, cplx(-1.0) / (dd * dd), ik, r, A, ej, fj);
                    emit(0, a, 1, cplx(1.0) / (dd * dd), ik, r, A, ea, fa);
                  }
                  continue;
                }
                const cplx w = cplx(1.0) / Djl;
                if (fam == 0) {
                  // [U_j - S_l]/Djl x U_a
                  if (aj) emit(0, nj, 2, w, ik, r, A, ej, fj);
                  else {
                    emit(0, nj, 1, w / cplx(ej - ea), ik, r, A, ej, fj);
                    emit(0, a, 1, -w / cplx(ej - ea), ik, r, A, ea, fa);
                  }
                  if (nu0) {
                    // - U_l U_a
                    if (al) emit(0, nl, 2, -w, ik, r, A, el, fl);
                    else {
                      emit(0, nl, 1, -w / cplx(el - ea), ik, r, A, el, fl);
                      emit(0, a, 1, w / cplx(el - ea), ik, r, A, ea, fa);
                    }
                  } else {
                    // - S_l U_a = -[U_a - S_l]/(e_a - e_l + inu)
                    const cplx d = cplx(ea - el) + inu;
                    emit(0, a, 1, -w / d, ik, r, A, ea, fa);
                    emit(1, nl, 1, w / d, ik, r, A, el, fl);
                  }
                } else {
                  // [U_j - S_l]/Djl x S_a  (inu != 0):  U_j S_a = [U_j - S_a]/(e_j - e_a + inu)
                  const cplx d = cplx(ej - ea) + inu;
                  emit(0, nj, 1, w / d, ik, r, A, ej, fj);
                  emit(1, a, 1, -w / d, ik, r, A, ea, fa);
                  // - S_l S_a
                  if (al) emit(1, nl, 2, -w, ik, r, A, el, fl);
                  else {
                    emit(1, nl, 1, -w / cplx(el - ea), ik, r, A, el, fl);
                    emit(1, a, 1, w / cplx(el - ea), ik, r, A, ea, fa);
                  }
                }
              }
            }
          }
  }

  /**
   * The explicit pair-space rung of the toy/gate form: K(q'; s) as (nc^2 x nc^2) matrices on the
   * tau grid (rows (p1' nc + p3), columns (a nc + b)), q' = k - k' through the map kmk(k, k'),
   * plus the instantaneous value K0(q') (the static rung W0bar = the full rung at inu' = 0) and
   * the inu' = 0 value of the dynamic part, Wd0(q') = W_dyn(0), used for K_d = W_dyn(s) - Wd0.
   */
  struct pair_rung {
    long nq = 0, nt = 0, nc2 = 0;
    nda::array<long, 2> kmk;                    // (nk, nk): q' index of k - k'
    nda::array<cplx, 4> Wd_s;                   // (nq, nt, nc2, nc2) dynamic part on the tau grid
    nda::array<cplx, 3> Wd0;                    // (nq, nc2, nc2) dynamic part at inu' = 0
    nda::array<cplx, 3> K0;                     // (nq, nc2, nc2) the static rung Z + W_dyn(0)
  };

  /** the pair block of a tf_vector at (k, r) flattened as (a nc + b) */
  inline void pack_pair(nda::MemoryArrayOfRank<2> auto const &V, long nc, nda::array<cplx, 1> &v) {
    for (long a = 0; a < nc; ++a)
      for (long bb = 0; bb < nc; ++bb) v(a * nc + bb) = V(a, bb);
  }

  /**
   * y = K_d F for a pure two-family F (F.cst ignored) and its exact frequency sum Fsum:
   *   two-family part: refit of W_dyn(s) F(s) per family (the e^{inu s} of the shifted family
   *   rides along), summed over k' with the rung at q' = k - k';
   *   constant part: -sum_k' Wd0(k - k') Fsum(k').
   * Returns the maximal refit error of the tau products.
   */
  inline double kd_apply(freq_basis const &b, pair_rung const &R, tf_vector const &F,
                         nda::array<cplx, 4> const &Fsum, tf_vector &y) {
    const long np = b.np, nt = b.nt, nk = F.nk, nc = F.nc, nc2 = nc * nc, nR = F.nR;
    y.zero();
    const long nfam = 2;
    // tau values of each family: (nt, nk, nc2, nR)
    nda::array<cplx, 4> Fs(nt, nk, nc2, nR), Ys(nt, nk, nc2, nR);
    nda::array<cplx, 1> v(nc2), w(nc2);
    double fit_err = 0.0;
    for (long fam = 0; fam < nfam; ++fam) {
      Fs() = cplx(0.0);
      for (long ik = 0; ik < nk; ++ik)
        for (long r = 0; r < nR; ++r)
          for (long a = 0; a < np; ++a) {
            pack_pair(F.fam(fam, a, ik, nda::range::all, nda::range::all, r), nc, v);
            bool anyv = false;
            for (auto const &x : v) anyv = anyv or (x != cplx(0.0));
            if (not anyv) continue;
            for (long i = 0; i < nt; ++i) {
              const double kf = b.KF(i, a);
              for (long p = 0; p < nc2; ++p) Fs(i, ik, p, r) += kf * v(p);
            }
          }
      // the rung product on the tau grid, summed over k'
      Ys() = cplx(0.0);
      for (long ik = 0; ik < nk; ++ik)
        for (long ikp = 0; ikp < nk; ++ikp) {
          const long iq = R.kmk(ik, ikp);
          for (long i = 0; i < nt; ++i)
            for (long r = 0; r < nR; ++r) {
              for (long p = 0; p < nc2; ++p) v(p) = Fs(i, ikp, p, r);
              for (long p = 0; p < nc2; ++p) {
                cplx s(0.0);
                for (long pp = 0; pp < nc2; ++pp) s += R.Wd_s(iq, i, p, pp) * v(pp);
                Ys(i, ik, p, r) += s;
              }
            }
        }
      // refit: (nt, batch) -> (np, batch)
      auto Y2 = nda::reshape(Ys, std::array<long, 2>{nt, nk * nc2 * nR});
      nda::array<cplx, 2> Yc(nt, nk * nc2 * nR);
      Yc() = Y2;
      auto c = b.pf.coeffs(Yc);
      fit_err = std::max(fit_err, b.pf.fit_error(Yc, c));
      for (long a = 0; a < b.np_fit; ++a)
        for (long ik = 0; ik < nk; ++ik)
          for (long p = 0; p < nc2; ++p)
            for (long r = 0; r < nR; ++r)
              y.fam(fam, a, ik, p / nc, p % nc, r) = c(a, (ik * nc2 + p) * nR + r);
    }
    // the constant part: -sum_k' Wd0(k-k') Fsum(k')
    for (long ik = 0; ik < nk; ++ik)
      for (long ikp = 0; ikp < nk; ++ikp) {
        const long iq = R.kmk(ik, ikp);
        for (long r = 0; r < nR; ++r) {
          pack_pair(Fsum(ikp, nda::range::all, nda::range::all, r), nc, v);
          for (long p = 0; p < nc2; ++p) {
            cplx s(0.0);
            for (long pp = 0; pp < nc2; ++pp) s += R.Wd0(iq, p, pp) * v(pp);
            y.cst(ik, p / nc, p % nc, r) -= s;
          }
        }
      }
    return fit_err;
  }

  // ==================================================================================
  // THE STATIC RESOLVENT AND THE DYSON ITERATION
  // ==================================================================================

  /**
   * The static ladder in the D = nk nc^2 pair space (rows/cols packed as k nc^2 + (a nc + b)):
   * Cb = (1/beta) sum_iw L0 (block-diagonal in k, from the T-sums) and
   * T_s = K_s (1 - Cb K_s)^-1 with K_s(k,k') = K0(k - k').
   */
  struct static_resolvent {
    long D = 0, nk = 0, nc = 0;
    nda::array<cplx, 2> Cb;      // (D, D) block diagonal
    nda::array<cplx, 2> Ts;      // (D, D)
  };

  inline static_resolvent build_static_resolvent(freq_basis const &b, pair_poles const &P,
                                                 pair_rung const &R, cplx inu, bool shared) {
    (void)b; (void)shared;
    static_resolvent S;
    const long nk = P.nk, nc = P.nc, nc2 = nc * nc, ng = P.ng, D = nk * nc2;
    S.D = D; S.nk = nk; S.nc = nc;
    S.Cb = nda::array<cplx, 2>(D, D);
    S.Cb() = cplx(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long j = 0; j < ng; ++j)
        for (long l = 0; l < ng; ++l) {
          // confluence WITHIN the G node set is by index (each index is one node) -- independent
          // of whether the vertex basis shares the set
          const cplx T = ward_legs::two_pole(P.fdG[size_t(j)], P.epsG(j), P.fdG[size_t(l)], P.epsG(l), inu,
                                             j == l).T;
          // Cb_{(p1'p3),(ab)} += T g_j(k)_{a p1'} g_l(k+q)_{p3 b}
          for (long p1p = 0; p1p < nc; ++p1p)
            for (long p3 = 0; p3 < nc; ++p3)
              for (long a = 0; a < nc; ++a)
                for (long bb = 0; bb < nc; ++bb)
                  S.Cb(ik * nc2 + p1p * nc + p3, ik * nc2 + a * nc + bb) +=
                      T * P.gk(j, ik, a, p1p) * P.gkq(l, ik, p3, bb);
        }
    // K_s (D x D)
    nda::array<cplx, 2> Ks(D, D);
    for (long ik = 0; ik < nk; ++ik)
      for (long ikp = 0; ikp < nk; ++ikp) {
        const long iq = R.kmk(ik, ikp);
        for (long p = 0; p < nc2; ++p)
          for (long pp = 0; pp < nc2; ++pp) Ks(ik * nc2 + p, ikp * nc2 + pp) = R.K0(iq, p, pp);
      }
    // T_s = K_s (1 - Cb K_s)^-1
    nda::matrix<cplx> M(D, D);
    nda::array<cplx, 2> CbK(D, D);
    nda::blas::gemm(S.Cb, Ks, CbK);
    for (long i = 0; i < D; ++i)
      for (long jj = 0; jj < D; ++jj) M(i, jj) = ((i == jj) ? cplx(1.0) : cplx(0.0)) - CbK(i, jj);
    const long nan_cb = nan_count(S.Cb), nan_ks = nan_count(Ks), nan_cbk = nan_count(CbK), nan_m0 = nan_count(M);
    nda::inverse_in_place(M);
    S.Ts = nda::array<cplx, 2>(D, D);
    nda::array<cplx, 2> Minv(D, D);
    Minv() = M;
    nda::blas::gemm(Ks, Minv, S.Ts);
    utils::check(nan_cb + nan_ks + nan_cbk + nan_m0 + nan_count(M) + nan_count(S.Ts) == 0,
                 "dynbse::build_static_resolvent: NaN (Cb {} Ks {} CbK {} 1-CbK {} inv {} Ts {}).",
                 nan_cb, nan_ks, nan_cbk, nan_m0, nan_count(M), nan_count(S.Ts));
    return S;
  }

  /**
   * Gamma = L_s (D + y): with F = L0 (D + y) and its exact sum Fsum,
   *   Gamma = F + L0 [ T_s Fsum ]   (the second term is L0 applied to a constant),
   *   sum_iw Gamma = Fsum + Cb T_s Fsum.
   * Returns Gamma (two-family) and Gsum. `Dc` is the constant external leg D (nk, nc, nc, nR).
   */
  inline void ls_apply(freq_basis const &b, pair_poles const &P, static_resolvent const &S, cplx inu,
                       bool shared, nda::array<cplx, 4> const &Dc, tf_vector const &y,
                       tf_vector &Gamma, nda::array<cplx, 4> &Gsum) {
    decltype(nda::range::all) all;
    const long nk = P.nk, nc = P.nc, nc2 = nc * nc, nR = y.nR, D = S.D;
    tf_vector X(b.np, nk, nc, nR);
    X.fam() = y.fam;
    X.cst() = Dc + y.cst;
    tf_vector F(b.np, nk, nc, nR);
    nda::array<cplx, 4> Fsum(nk, nc, nc, nR);
    l0_apply(b, P, inu, shared, X, F, Fsum);
    // c = T_s Fsum  (D x nR)
    nda::array<cplx, 2> fs(D, nR), cs(D, nR), cb(D, nR);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r) fs(ik * nc2 + p, r) = Fsum(ik, p / nc, p % nc, r);
    nda::blas::gemm(S.Ts, fs, cs);
    nda::blas::gemm(S.Cb, cs, cb);
    tf_vector Xc(b.np, nk, nc, nR);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r) Xc.cst(ik, p / nc, p % nc, r) = cs(ik * nc2 + p, r);
    tf_vector F2(b.np, nk, nc, nR);
    nda::array<cplx, 4> F2sum(nk, nc, nc, nR);
    l0_apply(b, P, inu, shared, Xc, F2, F2sum);
    Gamma.fam() = F.fam + F2.fam;
    Gamma.cst() = cplx(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r)
          Gsum(ik, p / nc, p % nc, r) = Fsum(ik, p / nc, p % nc, r) + cb(ik * nc2 + p, r);
    (void)all;
  }

  struct dyson_result {
    long iterations = 0;
    bool converged = false;
    double contraction = -1.0;     // last |dy_n|/|dy_{n-1}| (before the tolerance was met)
    double residual = -1.0;        // last |dy|/|y|
    std::vector<double> history;   // |dy|/|y| per iteration
    double fit_err_max = 0.0;      // worst tau refit in kd_apply
    nda::array<cplx, 4> Gsum;      // (nk, nc, nc, nR) sum_iw Gamma of the converged vertex
    nda::array<cplx, 4> Gsum1;     // the first iterate (static-dressed one dynamic rung)
    nda::array<cplx, 4> Gsum0;     // the static ladder (y = 0)
  };

  /**
   * The Dyson iteration on the dynamic remainder: y_{n+1} = K_d L_s (D + y_n), y_0 = 0, with
   * Anderson(2) mixing after the first two plain steps. Converges when |y_{n+1} - y_n| <=
   * tol |y_{n+1}| or when maxit is reached (the result then carries the last iterate; the caller
   * decides). The observable is sum_iw Gamma; the caller applies the D^dag legs.
   */
  inline dyson_result solve_dyson(freq_basis const &b, pair_poles const &P, pair_rung const &R,
                                  static_resolvent const &S, cplx inu, bool shared,
                                  nda::array<cplx, 4> const &Dc, double tol, long maxit,
                                  bool anderson = true) {
    const long nk = P.nk, nc = P.nc, nR = Dc.shape(3), np = b.np;
    dyson_result out;
    out.Gsum = nda::array<cplx, 4>(nk, nc, nc, nR);
    out.Gsum1 = nda::array<cplx, 4>(nk, nc, nc, nR);
    out.Gsum0 = nda::array<cplx, 4>(nk, nc, nc, nR);
    tf_vector y(np, nk, nc, nR), Gamma(np, nk, nc, nR), ynew(np, nk, nc, nR);
    tf_vector yprev(np, nk, nc, nR), ynew_prev(np, nk, nc, nR);
    nda::array<cplx, 4> Gsum(nk, nc, nc, nR);
    double dprev = -1.0;
    for (long it = 0; it <= maxit; ++it) {
      ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum);
      if (it == 0) out.Gsum0 = Gsum;
      if (it == 1) out.Gsum1 = Gsum;
      out.Gsum = Gsum;
      out.iterations = it;
      if (it == maxit) break;
      const double fe = kd_apply(b, R, Gamma, Gsum, ynew);
      out.fit_err_max = std::max(out.fit_err_max, fe);
      // convergence on the update
      double d2 = 0.0, n2 = 0.0;
      for (long i = 0; i < long(ynew.fam.size()); ++i) {
        const cplx dv = ynew.fam.data()[i] - y.fam.data()[i];
        d2 += std::norm(dv); n2 += std::norm(ynew.fam.data()[i]);
      }
      for (long i = 0; i < long(ynew.cst.size()); ++i) {
        const cplx dv = ynew.cst.data()[i] - y.cst.data()[i];
        d2 += std::norm(dv); n2 += std::norm(ynew.cst.data()[i]);
      }
      const double d = std::sqrt(d2), n = std::sqrt(std::max(n2, 1e-300));
      out.residual = d / n;
      out.history.push_back(out.residual);
      if (dprev > 0.0) out.contraction = d / dprev;
      // Anderson(2): x_{n+1} = ynew - theta (ynew - ynew_prev), theta from the secular equation
      if (anderson and it >= 1) {
        // r_n = ynew - y (the fixed-point residual at y), r_{n-1} = ynew_prev - yprev
        double num_re = 0.0, den = 0.0;
        auto acc = [&](cplx const *a, cplx const *ap, cplx const *yy, cplx const *yp, long n_) {
          for (long i = 0; i < n_; ++i) {
            const cplx rn = a[i] - yy[i], rp = ap[i] - yp[i], dr = rn - rp;
            num_re += std::real(std::conj(dr) * rn);
            den += std::norm(dr);
          }
        };
        acc(ynew.fam.data(), ynew_prev.fam.data(), y.fam.data(), yprev.fam.data(), long(y.fam.size()));
        acc(ynew.cst.data(), ynew_prev.cst.data(), y.cst.data(), yprev.cst.data(), long(y.cst.size()));
        const double theta = (den > 0.0) ? num_re / den : 0.0;
        yprev.fam() = y.fam; yprev.cst() = y.cst;
        ynew_prev.fam() = ynew.fam; ynew_prev.cst() = ynew.cst;
        // mixed iterate: (1-theta) ynew + theta ynew_prev  (theta = 0 recovers plain Neumann)
        y.fam() = ynew.fam * cplx(1.0 - theta) + ynew_prev.fam * cplx(theta);
        y.cst() = ynew.cst * cplx(1.0 - theta) + ynew_prev.cst * cplx(theta);
        // careful: ynew_prev must hold the UNMIXED previous evaluation; restore it
        ynew_prev.fam() = ynew.fam; ynew_prev.cst() = ynew.cst;
      } else {
        yprev.fam() = y.fam; yprev.cst() = y.cst;
        ynew_prev.fam() = ynew.fam; ynew_prev.cst() = ynew.cst;
        y.fam() = ynew.fam; y.cst() = ynew.cst;
      }
      dprev = d;
      if (out.residual <= tol) {
        // one more L_s application on the converged y gives the final Gamma
        ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum);
        out.Gsum = Gsum;
        out.iterations = it + 1;
        out.converged = true;
        break;
      }
    }
    return out;
  }

  /** the readout block: P(r', r) = sum_k sum_{ab} conj(Dleft(k, a, b, r')) Gsum(k, a, b, r) */
  inline nda::array<cplx, 2> collapse(nda::array<cplx, 4> const &Dleft, nda::array<cplx, 4> const &Gsum) {
    const long nk = Gsum.shape(0), nc = Gsum.shape(1), nR = Gsum.shape(3), nL = Dleft.shape(3);
    nda::array<cplx, 2> P(nL, nR);
    P() = cplx(0.0);
    for (long rl = 0; rl < nL; ++rl)
      for (long r = 0; r < nR; ++r)
        for (long ik = 0; ik < nk; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long bb = 0; bb < nc; ++bb)
              P(rl, r) += std::conj(Dleft(ik, a, bb, rl)) * Gsum(ik, a, bb, r);
    return P;
  }

} // namespace dynbse
} // namespace solvers
} // namespace methods

#endif // COQUI_VERTEX_DYNBSE_HPP
