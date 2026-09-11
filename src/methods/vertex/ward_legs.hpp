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

#ifndef COQUI_VERTEX_WARD_LEGS_HPP
#define COQUI_VERTEX_WARD_LEGS_HPP

/**
 * scGW-tilde Tier 1.5 (notes/scgw_screening_fix_proposal.pdf section 4.6; plan
 * notes/tier15_ward_legs_plan.md): the telescoping (discrete-Ward) leg vertex
 *
 *     Lambda0(k, iw; inu) = 1 - [Sigma(k, iw + inu) - Sigma(k, iw)] / inu
 *
 * as a correction to the ladder's pair propagator (vertex_ladder.icc header),
 *
 *     chi0_{(p1' nc + p3),(a nc + b)}(k; q, nu) = (1/beta) sum_iw G_{a p1'}(k, iw) G_{p3 b}(k+q, iw+inu),
 *
 * inserted ADJACENT TO THE VERTEX pair (a, b) -- a on the k line, b on the k+q line --
 * and symmetrized over the two legs (eq 21 of the proposal):
 *
 *     Dchi0 = (1/beta) sum_iw 1/2 { [(Lambda0 - 1) G]_{a p1'}(k, iw; inu) G_{p3 b}(k+q, iw+inu)
 *                                 + G_{a p1'}(k, iw) [G (Lambda0 - 1)]_{p3 b}(k+q, iw+inu; inu) }.
 *
 * That placement is the one that telescopes at q = 0 (proposal eq 22): with the vertex
 * traced, sum_a G_{p3 a}(iw+inu) [Lambda0 G]_{a p1'}(iw) = [G(iw+inu) Lambda0 G(iw)]_{p3 p1'}
 * and inu Lambda0 = G^-1(iw+inu) - G^-1(iw). The internal band sum of [Lambda0 G] runs
 * over ALL bands; only the external pair labels are restricted to the C window.
 *
 * EVALUATION -- pole products, no difference quotient anywhere. With the DLR pole
 * representations G(k, z) = sum_l g_l(k)/(z - e_l) and Sigma_c(k, z) = sum_p R_p(k)/(z - e_p)
 * (residue convention of vertex_pi::iaft_tools::pole_coeffs), partial fractions give
 * (Lambda0 - 1)(k, iw; inu) = sum_p R_p(k) / [(iw + inu - e_p)(iw - e_p)] exactly, and
 *
 *     Dchi0 = 1/2 sum_{p,l,m} S_{plm}(nu) { [R_p g_l](k)_{a p1'} g_m(k+q)_{p3 b}
 *                                         + g_l(k)_{a p1'} [g_m R_p](k+q)_{p3 b} },
 *     S_{plm}(nu) = (1/beta) sum_iw 1 / [(iw - e_l)(iw - e_p)(iw + inu - e_p)(iw + inu - e_m)].
 *
 * S closes in the two-pole family T(a, b; nu) = (1/beta) sum_iw 1/[(iw - a)(iw + inu - b)]
 * = [f(a) - f(b)] / (a - b + inu) (fermionic sum theorem; f(e - inu) = f(e) exactly for
 * bosonic nu) and its a-, b-, ab-derivatives (the confluent, same-node cases). Confluence
 * is decided BY INDEX (the dlr_pole_fit doctrine: the auxiliary grid is well separated,
 * min gap 2.17 dimensionless); every denominator is a real node gap or a complex gap with
 * |Im| = |nu| >= 2 pi / beta, so there is no small-nu cancellation and nu = 0 is an
 * ordinary node (the derivative branches). The same tables give the bare pair propagator
 * as sum_{lm} g_l(k)_{a p1'} g_m(k+q)_{p3 b} T(e_l, e_m; nu), which is the pin of the
 * normalization against the tau-product build (gate P0 of the plan).
 *
 * Pure algebra: no IAFT, no MPI, no loop state. Residues in, pair-space blocks out.
 * The residue arrays may come from iaft_tools::pole_coeffs (production, shared node set)
 * or from an exact rational toy (the unit gates, distinct node sets for G and Sigma).
 */

#include <cmath>
#include <complex>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"

namespace methods {
namespace solvers {
namespace ward_legs {

  using cplx = ComplexType;

  /** stable n_F(e) = 1/(exp(beta e) + 1) (same form as iaft_dconv's stable_nF). */
  inline double nF(double beta, double e) {
    if (e >= 0.0) {
      const double x = std::exp(-beta * e);
      return x / (1.0 + x);
    }
    return 1.0 / (1.0 + std::exp(beta * e));
  }

  /** f and its first three derivatives at one node; f(1-f) is formed from n_F(e) n_F(-e). */
  struct fermi_derivs {
    double f = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;
  };

  inline fermi_derivs fermi_all(double beta, double e) {
    fermi_derivs d;
    const double f = nF(beta, e), fm = nF(beta, -e);   // fm = 1 - f, no cancellation
    const double g = f * fm;
    d.f = f;
    d.f1 = -beta * g;                                   // f'   = -beta f (1-f)
    d.f2 = beta * beta * g * (fm - f);                  // f''  =  beta^2 f (1-f) (1-2f)
    d.f3 = -beta * beta * beta * g * (1.0 - 6.0 * g);   // f''' = -beta^3 f (1-f) (1 - 6 f (1-f))
    return d;
  }

  /**
   * The two-pole family at nodes (a, b) and bosonic shift inu:
   *   T   = (1/beta) sum_n 1/[(iw_n - a)(iw_n + inu - b)] = [f(a) - f(b)] / (a - b + inu)
   *   T1  = d/da T,   T2 = d/db T,   T12 = d/da d/db T.
   * `same` = a and b are the SAME node (confluence by index). inu must be EXACTLY zero at
   * the nu = 0 node (the derivative branches are selected on that).
   */
  struct pair_sums {
    cplx T = 0.0, T1 = 0.0, T2 = 0.0, T12 = 0.0;
  };

  inline pair_sums two_pole(fermi_derivs const &fa, double a, fermi_derivs const &fb, double b,
                            cplx inu, bool same) {
    pair_sums r;
    if (same) {
      if (inu == cplx(0.0)) {
        r.T = fa.f1;
        r.T1 = 0.5 * fa.f2;
        r.T2 = 0.5 * fa.f2;
        r.T12 = fa.f3 / 6.0;
      } else {
        r.T = 0.0;                     // f(a) - f(a - inu) = 0 exactly
        r.T1 = fa.f1 / inu;
        r.T2 = -fa.f1 / inu;
        r.T12 = 2.0 * fa.f1 / (inu * inu);
      }
      return r;
    }
    const cplx d = cplx(a - b) + inu;
    const double df = fa.f - fb.f;
    const cplx d2 = d * d, d3 = d2 * d;
    r.T = df / d;
    r.T1 = fa.f1 / d - df / d2;
    r.T2 = -fb.f1 / d + df / d2;
    r.T12 = (fa.f1 + fb.f1) / d2 - 2.0 * df / d3;
    return r;
  }

  /**
   * The k-independent tables at the requested bosonic nodes.
   *   Sm(inu, m, p, l) = S_{plm}(nu)   laid out for the C_m contraction (rows m, cols (p,l))
   *   Sl(inu, l, p, m) = S_{plm}(nu)   laid out for the C'_l contraction (rows l, cols (p,m))
   *   Tb(inu, l, m)    = T(e_l, e_m; nu)   the bare pair propagator kernel
   * epsS: Sigma_c nodes (np); epsG: G nodes (ng). `shared` = the two node sets are the same
   * array (production: both from one dlr_pole_fit), so l == p is confluent by index.
   */
  struct s_tables {
    long np = 0, ng = 0, nnu = 0;
    bool shared = false;
    nda::array<cplx, 1> inu;            // (nnu) the bosonic values i nu (exactly 0 at nu = 0)
    nda::array<cplx, 4> Sm;             // (nnu, ng, np, ng)
    nda::array<cplx, 4> Sl;             // (nnu, ng, np, ng)
    nda::array<cplx, 3> Tb;             // (nnu, ng, ng)
  };

  inline s_tables build_s_tables(double beta, nda::array<double, 1> const &epsS,
                                 nda::array<double, 1> const &epsG, bool shared,
                                 nda::array<cplx, 1> const &inu_list) {
    s_tables t;
    t.np = epsS.shape(0);
    t.ng = epsG.shape(0);
    t.nnu = inu_list.shape(0);
    t.shared = shared;
    utils::check(t.np > 0 and t.ng > 0 and t.nnu > 0, "ward_legs::build_s_tables: empty input.");
    if (shared) {
      utils::check(t.np == t.ng, "ward_legs::build_s_tables: shared node sets differ in size "
                                 "({} vs {}).", t.np, t.ng);
      for (long p = 0; p < t.np; ++p)
        utils::check(epsS(p) == epsG(p), "ward_legs::build_s_tables: shared node sets differ "
                                         "at index {}.", p);
    }
    t.inu = inu_list;
    t.Sm = nda::array<cplx, 4>(t.nnu, t.ng, t.np, t.ng);
    t.Sl = nda::array<cplx, 4>(t.nnu, t.ng, t.np, t.ng);
    t.Tb = nda::array<cplx, 3>(t.nnu, t.ng, t.ng);

    std::vector<fermi_derivs> fS(size_t(t.np)), fG(size_t(t.ng));
    for (long p = 0; p < t.np; ++p) fS[size_t(p)] = fermi_all(beta, epsS(p));
    for (long l = 0; l < t.ng; ++l) fG[size_t(l)] = fermi_all(beta, epsG(l));

    std::vector<pair_sums> Tlp(size_t(t.ng * t.np)), Tpm(size_t(t.np * t.ng));
    for (long jn = 0; jn < t.nnu; ++jn) {
      const cplx inu = inu_list(jn);
      // T(e_l^G, e_m^G): the bare kernel, confluent when l == m
      for (long l = 0; l < t.ng; ++l)
        for (long m = 0; m < t.ng; ++m)
          t.Tb(jn, l, m) = two_pole(fG[size_t(l)], epsG(l), fG[size_t(m)], epsG(m), inu, l == m).T;
      // T(e_l^G, e_p^S) and T(e_p^S, e_m^G)
      for (long l = 0; l < t.ng; ++l)
        for (long p = 0; p < t.np; ++p)
          Tlp[size_t(l * t.np + p)] = two_pole(fG[size_t(l)], epsG(l), fS[size_t(p)], epsS(p), inu,
                                               shared and l == p);
      for (long p = 0; p < t.np; ++p)
        for (long m = 0; m < t.ng; ++m)
          Tpm[size_t(p * t.ng + m)] = two_pole(fS[size_t(p)], epsS(p), fG[size_t(m)], epsG(m), inu,
                                               shared and m == p);
      for (long p = 0; p < t.np; ++p) {
        const pair_sums Tpp = two_pole(fS[size_t(p)], epsS(p), fS[size_t(p)], epsS(p), inu, true);
        for (long l = 0; l < t.ng; ++l) {
          const bool lp = shared and l == p;
          for (long m = 0; m < t.ng; ++m) {
            const bool mp = shared and m == p;
            cplx S;
            if (lp and mp) {
              S = Tpp.T12;
            } else if (lp) {
              S = (Tpp.T1 - Tpm[size_t(p * t.ng + m)].T1) / cplx(epsS(p) - epsG(m));
            } else if (mp) {
              S = (Tpp.T2 - Tlp[size_t(l * t.np + p)].T2) / cplx(epsS(p) - epsG(l));
            } else {
              S = (Tpp.T - Tpm[size_t(p * t.ng + m)].T - Tlp[size_t(l * t.np + p)].T + t.Tb(jn, l, m)) /
                  cplx((epsS(p) - epsG(l)) * (epsS(p) - epsG(m)));
            }
            t.Sm(jn, m, p, l) = S;
            t.Sl(jn, l, p, m) = S;
          }
        }
      }
    }
    return t;
  }

  /**
   * The assembled, k-resolved context. Per spin s, k and bosonic node:
   *   Cm(s, inu, k, m)_{a p1'} = sum_{p,l} S_{plm}(nu) [R_p g_l](k)_{a p1'}
   *   Cl(s, inu, k, l)_{p3 b}  = sum_{p,m} S_{plm}(nu) [g_m R_p](k)_{p3 b}
   * with the internal band index of the products running over ALL nb bands and the
   * external labels in the C window (nc). gC(s, l, k) = g_l(k) restricted to C x C.
   */
  struct ward_ctx {
    long ns = 0, nk = 0, nc = 0, nb = 0, np = 0, ng = 0, nnu = 0;
    s_tables tab;
    nda::array<cplx, 5> gC;   // (ns, ng, nk, nc, nc)
    nda::array<cplx, 6> Cm;   // (ns, nnu, nk, ng, nc, nc)
    nda::array<cplx, 6> Cl;   // (ns, nnu, nk, ng, nc, nc)
  };

  /**
   * Build the context from the C-restricted residue blocks:
   *   gC (ns, ng, nk, nc, nc): g_l(k), C x C
   *   gA (ns, ng, nk, nb, nc): g_l(k), all x C      (the [R_p g_l] product's right factor)
   *   gB (ns, ng, nk, nc, nb): g_m(k), C x all      (the [g_m R_p] product's left factor)
   *   RA (ns, np, nk, nc, nb): R_p(k), C x all
   *   RB (ns, np, nk, nb, nc): R_p(k), all x C
   * `tab` must have been built with the same (np, ng).
   */
  inline ward_ctx build_ward_ctx(s_tables tab,
                                 nda::array<cplx, 5> const &gC, nda::array<cplx, 5> const &gA,
                                 nda::array<cplx, 5> const &gB, nda::array<cplx, 5> const &RA,
                                 nda::array<cplx, 5> const &RB) {
    decltype(nda::range::all) all;
    ward_ctx c;
    c.ns = gC.shape(0); c.ng = gC.shape(1); c.nk = gC.shape(2); c.nc = gC.shape(3);
    c.nb = gA.shape(3); c.np = RA.shape(1);
    c.nnu = tab.nnu;
    utils::check(gC.shape(4) == c.nc and gA.shape(0) == c.ns and gA.shape(1) == c.ng and
                 gA.shape(2) == c.nk and gA.shape(4) == c.nc and gB.shape(0) == c.ns and
                 gB.shape(1) == c.ng and gB.shape(2) == c.nk and gB.shape(3) == c.nc and
                 gB.shape(4) == c.nb and RA.shape(0) == c.ns and RA.shape(2) == c.nk and
                 RA.shape(3) == c.nc and RA.shape(4) == c.nb and RB.shape(0) == c.ns and
                 RB.shape(1) == c.np and RB.shape(2) == c.nk and RB.shape(3) == c.nb and
                 RB.shape(4) == c.nc,
                 "ward_legs::build_ward_ctx: residue block shape mismatch.");
    utils::check(tab.np == c.np and tab.ng == c.ng,
                 "ward_legs::build_ward_ctx: tables built for (np, ng) = ({}, {}), residues "
                 "carry ({}, {}).", tab.np, tab.ng, c.np, c.ng);
    c.tab = std::move(tab);
    c.gC = gC;
    c.Cm = nda::array<cplx, 6>(c.ns, c.nnu, c.nk, c.ng, c.nc, c.nc);
    c.Cl = nda::array<cplx, 6>(c.ns, c.nnu, c.nk, c.ng, c.nc, c.nc);

    const long npg = c.np * c.ng, nc2 = c.nc * c.nc;
    // products for one (s, k): A(p*ng + l) = [R_p g_l]_{C x C}, B(p*ng + m) = [g_m R_p]_{C x C}
    nda::array<cplx, 3> A(npg, c.nc, c.nc), B(npg, c.nc, c.nc);
    nda::array<cplx, 2> Sx(c.ng, npg), Cx(c.ng, nc2);
    for (long is = 0; is < c.ns; ++is)
      for (long ik = 0; ik < c.nk; ++ik) {
        for (long p = 0; p < c.np; ++p)
          for (long l = 0; l < c.ng; ++l) {
            nda::blas::gemm(RA(is, p, ik, all, all), gA(is, l, ik, all, all), A(p * c.ng + l, all, all));
            nda::blas::gemm(gB(is, l, ik, all, all), RB(is, p, ik, all, all), B(p * c.ng + l, all, all));
          }
        auto A2 = nda::reshape(A, std::array<long, 2>{npg, nc2});
        auto B2 = nda::reshape(B, std::array<long, 2>{npg, nc2});
        for (long jn = 0; jn < c.nnu; ++jn) {
          // Cm(m, :) = sum_{(p,l)} Sm(m, p, l) A(p, l, :)
          Sx() = nda::reshape(c.tab.Sm(jn, all, all, all), std::array<long, 2>{c.ng, npg});
          nda::blas::gemm(Sx, A2, Cx);
          c.Cm(is, jn, ik, all, all, all) = nda::reshape(Cx, std::array<long, 3>{c.ng, c.nc, c.nc});
          // Cl(l, :) = sum_{(p,m)} Sl(l, p, m) B(p, m, :)
          Sx() = nda::reshape(c.tab.Sl(jn, all, all, all), std::array<long, 2>{c.ng, npg});
          nda::blas::gemm(Sx, B2, Cx);
          c.Cl(is, jn, ik, all, all, all) = nda::reshape(Cx, std::array<long, 3>{c.ng, c.nc, c.nc});
        }
      }
    return c;
  }

  /**
   * Dchi0(k; k+q, nu_jn) += into the (nc^2, nc^2) pair block, row (p1' nc + p3),
   * column (a nc + b), scaled by `scale` (1 = the correction as defined above).
   */
  inline void add_pair_correction(ward_ctx const &c, long is, long jn, long ik, long ikpq,
                                  nda::MemoryArrayOfRank<2> auto &&Cb, double scale = 1.0) {
    const long nc = c.nc, nc2 = nc * nc;
    utils::check(Cb.shape(0) == nc2 and Cb.shape(1) == nc2,
                 "ward_legs::add_pair_correction: block shape ({}, {}) != ({}, {}).",
                 Cb.shape(0), Cb.shape(1), nc2, nc2);
    const cplx half(0.5 * scale);
    for (long m = 0; m < c.ng; ++m)
      for (long a = 0; a < nc; ++a)
        for (long p1p = 0; p1p < nc; ++p1p) {
          const cplx cm = c.Cm(is, jn, ik, m, a, p1p);      // [sum_{pl} S R_p g_l](k)_{a p1'}
          const cplx gk = c.gC(is, m, ik, a, p1p);          // g_m(k)_{a p1'}
          for (long p3 = 0; p3 < nc; ++p3)
            for (long b = 0; b < nc; ++b)
              Cb(p1p * nc + p3, a * nc + b) +=
                  half * (cm * c.gC(is, m, ikpq, p3, b) + gk * c.Cl(is, jn, ikpq, m, p3, b));
        }
  }

  /** the bare pair propagator through the poles: sum_{lm} T(e_l, e_m; nu) g_l(k)_{a p1'} g_m(k+q)_{p3 b}. */
  inline void pole_bare_bubble(ward_ctx const &c, long is, long jn, long ik, long ikpq,
                               nda::MemoryArrayOfRank<2> auto &&out) {
    const long nc = c.nc, nc2 = nc * nc;
    utils::check(out.shape(0) == nc2 and out.shape(1) == nc2,
                 "ward_legs::pole_bare_bubble: block shape mismatch.");
    out() = cplx(0.0);
    for (long l = 0; l < c.ng; ++l)
      for (long m = 0; m < c.ng; ++m) {
        const cplx T = c.tab.Tb(jn, l, m);
        for (long a = 0; a < nc; ++a)
          for (long p1p = 0; p1p < nc; ++p1p) {
            const cplx gl = T * c.gC(is, l, ik, a, p1p);
            for (long p3 = 0; p3 < nc; ++p3)
              for (long b = 0; b < nc; ++b)
                out(p1p * nc + p3, a * nc + b) += gl * c.gC(is, m, ikpq, p3, b);
          }
      }
  }

  /**
   * Convenience for callers holding FULL nb x nb residues (toys, small fixtures): slice the
   * five C-restricted blocks for the window `bw`. g: (ns, ng, nk, nb, nb), R: (ns, np, nk, nb, nb).
   */
  struct residue_blocks {
    nda::array<cplx, 5> gC, gA, gB, RA, RB;
  };

  inline residue_blocks slice_blocks(nda::array<cplx, 5> const &g, nda::array<cplx, 5> const &R,
                                     nda::range bw) {
    decltype(nda::range::all) all;
    residue_blocks b;
    b.gC = g(all, all, all, bw, bw);
    b.gA = g(all, all, all, all, bw);
    b.gB = g(all, all, all, bw, all);
    b.RA = R(all, all, all, bw, all);
    b.RB = R(all, all, all, all, bw);
    return b;
  }

} // namespace ward_legs
} // namespace solvers
} // namespace methods

#endif // COQUI_VERTEX_WARD_LEGS_HPP
