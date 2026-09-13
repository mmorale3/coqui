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

#include <chrono>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/omp_threads.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
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
    nda::array<double, 2> KF2;            // (nt, np) tau[U_a^2] = -K_F(s_i, eps_a) (s_i - beta f_a): the
                                          // kernel of the inu = 0 second family at the union's G nodes
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
    b.KF2 = nda::array<double, 2>(b.nt, b.np);
    for (long a = 0; a < b.np; ++a)
      for (long i = 0; i < b.nt; ++i) {
        const double u = b.s(i) - b.beta * b.fd[size_t(a)].f;
        F2(i, a) = cplx(-b.KF(i, a) * u);
        b.KF2(i, a) = -b.KF(i, a) * u;
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
    nda::array<double, 2> KF(b.nt, np1), KF2(b.nt, np1);
    nda::array<cplx, 2> Dsq(np1, np1), Dcb(np1, np1);
    Dsq() = cplx(0.0);
    Dcb() = cplx(0.0);
    for (long a = 0; a < np0; ++a) {
      eps(a) = b.eps(a);
      fhalf(a) = b.fhalf(a);
      KF(nda::range::all, a) = b.KF(nda::range::all, a);
      KF2(nda::range::all, a) = b.KF2(nda::range::all, a);
      Dsq(a, nda::range(np0)) = b.Dsq(a, nda::range::all);
      Dcb(a, nda::range(np0)) = b.Dcb(a, nda::range::all);
    }
    for (long e = 0; e < ne; ++e) {
      const long a = np0 + e;
      eps(a) = extra(e);
      b.fd.push_back(fermi_all(b.beta, extra(e)));
      fhalf(a) = b.fd[size_t(a)].f - 0.5;
      for (long i = 0; i < b.nt; ++i) {
        KF(i, a) = imag_axes_ft::dlr_kF(b.beta, b.s(i), extra(e));
        KF2(i, a) = -KF(i, a) * (b.s(i) - b.beta * b.fd[size_t(a)].f);
      }
    }
    // the double / triple-pole re-expansions of the EXTENSION nodes on the DLR set (same closed-form
    // tau functions as build_freq_basis, fitted on the vertex pole fit): used by the small-nu fold of
    // the twisted family (T_a -> U_a^2 - i nu U_a^3 for |eps_a| >> |nu|); a confluent product on an
    // extension node remains an error.
    if (ne > 0) {
      nda::array<cplx, 2> F2e(b.nt, ne), F3e(b.nt, ne);
      for (long e = 0; e < ne; ++e) {
        const long a = np0 + e;
        for (long i = 0; i < b.nt; ++i) {
          const double u = b.s(i) - b.beta * b.fd[size_t(a)].f;
          F2e(i, e) = cplx(-KF(i, a) * u);
          F3e(i, e) = cplx(0.5 * KF(i, a) * (u * u + b.beta * b.fd[size_t(a)].f1));
        }
      }
      auto c2e = b.pf.coeffs(F2e);   // (np0, ne)
      auto c3e = b.pf.coeffs(F3e);
      for (long e = 0; e < ne; ++e)
        for (long cc = 0; cc < np0; ++cc) {
          Dsq(np0 + e, cc) = c2e(cc, e);
          Dcb(np0 + e, cc) = c3e(cc, e);
        }
    }
    b.np = np1;
    b.eps = std::move(eps);
    b.fhalf = std::move(fhalf);
    b.KF = std::move(KF);
    b.KF2 = std::move(KF2);
    b.Dsq = std::move(Dsq);
    b.Dcb = std::move(Dcb);
  }

  /**
   * A regularized pole fit onto an ARBITRARY node set (the dlr_pole_fit machinery: real thin SVD of
   * the kernel on the backend tau grid, rank fixed from the spectrum at rel_tol). Used to put G's
   * residues on the UNION scheme's G grid (a half-spacing-shifted copy of the vertex grid), so that
   * no G node coincides with a vertex node and the pair algebra never meets a confluent product.
   */
  struct node_pole_fit {
    long np = 0, nt = 0, n_kept = 0;
    double beta = 0.0, s_max = 0.0, s_min_kept = 0.0;
    nda::array<double, 1> eps, s;
    nda::array<double, 2> Kmat;                 // (nt, np)
    nda::array<double, 2> Ut, Vs;               // (n_kept, nt), (np, n_kept) with 1/sigma folded in
    void build(double beta_, nda::array<double, 1> const &s_grid, nda::array<double, 1> const &nodes, double rtol = 1e-8) {
      beta = beta_; nt = s_grid.shape(0); np = nodes.shape(0);
      s = s_grid; eps = nodes;
      Kmat = nda::array<double, 2>(nt, np);
      for (long i = 0; i < nt; ++i)
        for (long p = 0; p < np; ++p) Kmat(i, p) = imag_axes_ft::dlr_kF(beta, s(i), eps(p));
      nda::matrix<double, nda::F_layout> A(nt, np);
      A() = Kmat;
      const long ms = std::min(nt, np);
      nda::vector<double> sig(ms);
      nda::matrix<double, nda::F_layout> U(nt, nt), VT(np, np);
      const int info = nda::lapack::gesvd(A, sig, U, VT);
      utils::check(info == 0, "dynbse::node_pole_fit: gesvd failed (info = {}).", info);
      s_max = sig(0);
      n_kept = 0;
      while (n_kept < ms and sig(n_kept) > rtol * s_max) ++n_kept;
      utils::check(n_kept > 0, "dynbse::node_pole_fit: no singular direction kept.");
      s_min_kept = sig(n_kept - 1);
      Ut = nda::array<double, 2>(n_kept, nt);
      Vs = nda::array<double, 2>(np, n_kept);
      for (long k = 0; k < n_kept; ++k) {
        for (long i = 0; i < nt; ++i) Ut(k, i) = U(i, k);
        for (long p = 0; p < np; ++p) Vs(p, k) = VT(k, p) / sig(k);
      }
    }
    /** residues of tau-grid data (nt, d) -> (np, d) */
    nda::array<cplx, 2> coeffs(nda::MemoryArrayOfRank<2> auto const &F) const {
      const long d = F.shape(1);
      nda::array<cplx, 2> c(np, d);
      c() = cplx(0.0);
      for (long jd = 0; jd < d; ++jd)
        for (long k = 0; k < n_kept; ++k) {
          cplx g(0.0);
          for (long i = 0; i < nt; ++i) g += Ut(k, i) * F(i, jd);
          for (long p = 0; p < np; ++p) c(p, jd) += Vs(p, k) * g;
        }
      return c;
    }
    double fit_error(nda::MemoryArrayOfRank<2> auto const &F, nda::array<cplx, 2> const &c) const {
      const long d = F.shape(1);
      double num = 0.0, den = 0.0;
      for (long i = 0; i < nt; ++i)
        for (long jd = 0; jd < d; ++jd) {
          cplx rec(0.0);
          for (long p = 0; p < np; ++p) rec += Kmat(i, p) * c(p, jd);
          num = std::max(num, std::abs(F(i, jd) - rec));
          den = std::max(den, std::abs(F(i, jd)));
        }
      return (den > 0.0) ? num / den : num;
    }
  };

  /** the union scheme's G grid: the midpoints of the (sorted) vertex nodes plus one node beyond
   *  each end at half the end spacing -- np_fit + 1 nodes, none coinciding with a vertex node. */
  inline nda::array<double, 1> shifted_nodes(freq_basis const &b) {
    const long n = b.np_fit;
    std::vector<double> e(static_cast<size_t>(n), 0.0);
    for (long a = 0; a < n; ++a) e[size_t(a)] = b.eps(a);
    std::sort(e.begin(), e.end());
    nda::array<double, 1> g(n + 1);
    g(0) = e[0] - 0.5 * (e[1] - e[0]);
    for (long a = 0; a + 1 < n; ++a) g(a + 1) = 0.5 * (e[size_t(a)] + e[size_t(a + 1)]);
    g(n) = e[size_t(n - 1)] + 0.5 * (e[size_t(n - 1)] - e[size_t(n - 2)]);
    return g;
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
                                    long gnode0 = 0, nda::array<long, 1> const *gmap = nullptr) {
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
      p.gnode(j) = (gmap != nullptr) ? (*gmap)(j) : gnode0 + j;
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
  inline void l0_apply_ref(freq_basis const &b, pair_poles const &P, cplx inu, bool shared,
                           tf_vector const &X, tf_vector &F, nda::array<cplx, 4> &Fsum) {
    decltype(nda::range::all) all;
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR;
    utils::check(X.np == np and X.nk == nk and X.nc == nc, "dynbse::l0_apply: shape mismatch.");
    for (long j = 0; j < ng; ++j)
      utils::check(P.gnode(j) < np and (not shared or P.gnode(j) < b.np_fit), "dynbse::l0_apply: G node map out of range.");
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

  // ==================================================================================
  // THE TWISTED-PAIR BASIS AT inu != 0  (increment D2e)
  // ==================================================================================
  //
  // At inu != 0 the shifted family is carried as the TWISTED PAIRS T_a = U_a S_a instead of S_a:
  //   S_a = U_a - i nu T_a,   T_a = [U_a - S_a]/(i nu),   tau[T_a] = K_F(s, e_a) phi_nu(s),
  //   phi_nu(s) = (1 - e^{i nu s})/(i nu)  (bounded: -> -s),   (1/beta) sum_iw T_a = 0 EXACTLY.
  // Every partial-fraction coefficient is then bounded by node gaps (never 1/nu), which is
  // what the {U, S} representation lacked: its index-confluent pairs [U_j - S_j]/(i nu) put
  // +-1/nu-sized canceling content into both families and the per-family refit noise
  // re-entered the iteration at the physical level (1/nu = beta/2pi = 159 at beta = 1000).
  // The confluent products that remain are re-expanded through two nu-dependent tables,
  // in the Dsq/Dcb pattern (exact large parts + a fitted bounded remainder):
  //   R1_j = U_j^2 S_j     = (beta f_j / i nu) U_j              + fit[K_F(s,e_j) psi_nu(s)],
  //   R3_j = U_j^2 S_j^2   = (2 beta f_j/(i nu)^2) U_j - (beta f_j/i nu) T_j
  //                                                        + fit[K_F(s,e_j)(2 xi_nu(s) - s psi_nu(s))],
  //   psi_nu = (e^{i nu s} - 1 - i nu s)/(i nu)^2 (-> s^2/2),  xi_nu = (e^{i nu s} - 1 - i nu s - (i nu s)^2/2)/(i nu)^3,
  // with the exact frequency sums  sum R1_j = f'_j/(i nu),  sum R3_j = 2 f'_j/(i nu)^2,  sum U_j^2 = f'_j.
  // The fits target {K_F(.,e_c)} U {K_F(.,e_c) phi_nu} on a dense tau grid (real-embedded SVD,
  // columns normalized, relative truncation). U_j^2 keeps the Dsq re-expansion.

  struct shift_tables {
    cplx inu = cplx(0.0);
    long np = 0, np_fit = 0;
    nda::array<cplx, 1> r1u, r3u, r3t;        // exact parts on U_j / U_j / T_j
    nda::array<cplx, 1> s1, s3;               // exact frequency sums of R1_j, R3_j
    nda::array<cplx, 2> R1U, R1T, R3U, R3T;   // (np, np): the fitted remainders onto U_c / T_c
    double fit_err = 0.0;                     // max relative residual of the two fits
    double cond_kept = 0.0;                   // s_max / s_min_kept of the fit system
  };

  namespace shift_detail {
    inline cplx phi_nu(cplx inu, double s) {
      const cplx x = inu * s;
      if (std::abs(x) < 0.5) {
        // -(s) sum_{k>=1} x^{k-1}/k!
        cplx acc(1.0), term(1.0);
        for (int k = 2; k <= 22; ++k) { term *= x / double(k); acc += term; }
        return -s * acc;
      }
      return (cplx(1.0) - std::exp(x)) / inu;
    }
    inline cplx psi_nu(cplx inu, double s) {
      const cplx x = inu * s;
      if (std::abs(x) < 0.5) {
        // s^2 sum_{k>=2} x^{k-2}/k!
        cplx acc(0.5), term(0.5);
        for (int k = 3; k <= 24; ++k) { term *= x / double(k); acc += term; }
        return s * s * acc;
      }
      return (std::exp(x) - cplx(1.0) - x) / (inu * inu);
    }
    inline cplx xi_nu(cplx inu, double s) {
      const cplx x = inu * s;
      if (std::abs(x) < 0.5) {
        // s^3 sum_{k>=3} x^{k-3}/k!
        cplx acc(1.0 / 6.0), term(1.0 / 6.0);
        for (int k = 4; k <= 25; ++k) { term *= x / double(k); acc += term; }
        return s * s * s * acc;
      }
      return (std::exp(x) - cplx(1.0) - x - x * x / 2.0) / (inu * inu * inu);
    }
  } // namespace shift_detail

  /**
   * mode 0: the fitted tables (default). mode 1: the EXACT partial-fraction tables through Dsq --
   *   R1_j = U_j^2 S_j   = [U_j^2 - T_j]/(i nu)            -> R1U(j,c) = Dsq(j,c)/(i nu), R1T(j,j) = -1/(i nu)
   *   R3_j = U_j^2 S_j^2 = [U_j^2 + S_j^2]/(i nu)^2 - 2 T_j/(i nu)^2, S_j^2 = sum_c Dsq(j,c)(U_c - i nu T_c)
   *                                                      -> R3U(j,c) = 2 Dsq(j,c)/(i nu)^2, R3T(j,c) = -Dsq(j,c)/(i nu), R3T(j,j) -= 2/(i nu)^2
   * (1/nu-amplified but with Dsq's accuracy only) -- the diagnostic of the fit route.
   */
  inline shift_tables build_shift_tables(freq_basis const &b, cplx inu, double rtol = 1e-11, int mode = 0) {
    using namespace shift_detail;
    utils::check(inu != cplx(0.0), "dynbse::build_shift_tables: inu = 0 has no shifted family.");
    shift_tables st;
    if (mode == 1) {
      st.inu = inu; st.np = b.np; st.np_fit = b.np_fit;
      const long np = b.np;
      st.r1u = nda::array<cplx, 1>(np); st.r3u = nda::array<cplx, 1>(np); st.r3t = nda::array<cplx, 1>(np);
      st.s1 = nda::array<cplx, 1>(np); st.s3 = nda::array<cplx, 1>(np);
      st.R1U = nda::array<cplx, 2>(np, np); st.R1T = nda::array<cplx, 2>(np, np);
      st.R3U = nda::array<cplx, 2>(np, np); st.R3T = nda::array<cplx, 2>(np, np);
      st.R1U() = cplx(0.0); st.R1T() = cplx(0.0); st.R3U() = cplx(0.0); st.R3T() = cplx(0.0);
      const cplx in2 = inu * inu;
      for (long j = 0; j < np; ++j) {
        const double f1 = b.fd[size_t(j)].f1;
        st.r1u(j) = cplx(0.0); st.r3u(j) = cplx(0.0); st.r3t(j) = cplx(0.0);
        st.s1(j) = cplx(f1) / inu;
        st.s3(j) = cplx(2.0 * f1) / in2;
        if (j >= b.np_fit) continue;
        for (long c = 0; c < np; ++c) {
          const cplx d = b.Dsq(j, c);
          st.R1U(j, c) += d / inu;
          st.R3U(j, c) += cplx(2.0) * d / in2;
          st.R3T(j, c) += -d / inu;
        }
        st.R1T(j, j) += cplx(-1.0) / inu;
        st.R3T(j, j) += cplx(-2.0) / in2;
      }
      st.fit_err = b.dsq_fit_err;
      st.cond_kept = 0.0;
      return st;
    }
    st.inu = inu;
    st.np = b.np;
    st.np_fit = b.np_fit;
    const long np = b.np, npf = b.np_fit, nt = b.nt;
    const double beta = b.beta;
    st.r1u = nda::array<cplx, 1>(np); st.r3u = nda::array<cplx, 1>(np); st.r3t = nda::array<cplx, 1>(np);
    st.s1 = nda::array<cplx, 1>(np); st.s3 = nda::array<cplx, 1>(np);
    // The FULL tau-functions are fitted (no analytic split): tau[U_j^2 S_j] = K_F [beta f_j/(i nu)
    // + psi_nu(s)] and tau[U_j^2 S_j^2] = K_F [2 beta f_j/(i nu)^2 - (beta f_j/(i nu)) phi_nu + 2 xi_nu
    // - s psi_nu] are benign (Dcb-class: for f_j = 1 the bracket vanishes at s = beta where K_F
    // peaks), whereas their "exact part + remainder" pieces are two huge canceling functions --
    // fitting the remainder alone lost the cancellation at large beta (R3 wrong by O(1) at
    // beta = 1000). The exact frequency sums are kept.
    for (long j = 0; j < np; ++j) {
      const double f1 = b.fd[size_t(j)].f1;
      st.r1u(j) = cplx(0.0);
      st.r3u(j) = cplx(0.0);
      st.r3t(j) = cplx(0.0);
      st.s1(j) = cplx(f1) / inu;
      st.s3(j) = cplx(2.0 * f1) / (inu * inu);
    }
    // ---- the dense tau grid: the backend nodes and three interior points per interval -------
    std::vector<double> ss;
    ss.reserve(size_t(4 * nt + 8));
    std::vector<double> s0(static_cast<size_t>(nt), 0.0);
    for (long i = 0; i < nt; ++i) s0[size_t(i)] = b.s(i);
    std::sort(s0.begin(), s0.end());
    for (long i = 0; i < nt; ++i) {
      ss.push_back(s0[size_t(i)]);
      if (i + 1 < nt) {
        const double a = s0[size_t(i)], c = s0[size_t(i + 1)];
        ss.push_back(a + 0.25 * (c - a)); ss.push_back(a + 0.5 * (c - a)); ss.push_back(a + 0.75 * (c - a));
      }
    }
    const long ntau = long(ss.size());
    // ---- the basis: K_F(s, e_c) and K_F(s, e_c) phi_nu(s), c < np_fit; columns normalized --------
    const long ncol = 2 * npf;
    nda::array<cplx, 2> A(ntau, ncol);
    nda::array<double, 1> cn(ncol);
    // EXPERIMENT (small-nu spurious mode): keep the twisted column T_c only when |eps_c| <= ratio |nu|
    // (for |eps_c| >> |nu| the twist is invisible on K_F's support and T_c ~ -s K_F(eps_c) lies in the
    // U span to the DLR class); env COQUI_DYNBSE_TKEEP = ratio, unset / 0 = keep all.
    double tkeep = 0.0;
    if (char const *e = std::getenv("COQUI_DYNBSE_TKEEP")) tkeep = std::atof(e);
    long n_tkept = 0;
    for (long c = 0; c < npf; ++c) {
      const bool keep_t = (tkeep <= 0.0) or (std::abs(b.eps(c)) <= tkeep * std::abs(inu));
      if (keep_t) ++n_tkept;
      double n1 = 0.0, n2 = 0.0;
      for (long i = 0; i < ntau; ++i) {
        const double kf = imag_axes_ft::dlr_kF(beta, ss[size_t(i)], b.eps(c));
        const cplx ph = phi_nu(inu, ss[size_t(i)]);
        A(i, c) = cplx(kf);
        A(i, npf + c) = keep_t ? cplx(kf) * ph : cplx(0.0);
        n1 += kf * kf;
        n2 += std::norm(A(i, npf + c));
      }
      cn(c) = std::sqrt(std::max(n1, 1e-300));
      cn(npf + c) = std::sqrt(std::max(n2, 1e-300));
      for (long i = 0; i < ntau; ++i) { A(i, c) /= cn(c); A(i, npf + c) /= cn(npf + c); }
    }
    // ---- real embedding [Ar -Ai; Ai Ar] (2 ntau x 2 ncol), thin SVD -------------------------
    const long M = 2 * ntau, N = 2 * ncol;
    nda::matrix<double, nda::F_layout> Ar(M, N);
    for (long i = 0; i < ntau; ++i)
      for (long c = 0; c < ncol; ++c) {
        Ar(i, c) = A(i, c).real();          Ar(i, ncol + c) = -A(i, c).imag();
        Ar(ntau + i, c) = A(i, c).imag();   Ar(ntau + i, ncol + c) = A(i, c).real();
      }
    const long ms = std::min(M, N);
    nda::vector<double> sig(ms);
    nda::matrix<double, nda::F_layout> U(M, M), VT(N, N);
    {
      nda::matrix<double, nda::F_layout> Acopy(M, N);
      Acopy() = Ar;
      const int info = nda::lapack::gesvd(Acopy, sig, U, VT);
      utils::check(info == 0, "dynbse::build_shift_tables: gesvd failed (info = {}).", info);
    }
    long nk = 0;
    while (nk < ms and sig(nk) > rtol * sig(0)) ++nk;
    utils::check(nk > 0, "dynbse::build_shift_tables: no singular direction kept.");
    st.cond_kept = sig(0) / sig(nk - 1);
    if (tkeep > 0.0)
      app_log(2, "  [dynbse shift tables] inu = {:.4e}i: T columns kept {} of {} (|eps_c| <= {} |nu|), kept condition {:.3e}",
              inu.imag(), n_tkept, npf, tkeep, st.cond_kept);
    // ---- targets: t1_j = K_F psi, t3_j = K_F (2 xi - s psi), j < np_fit; solve, unscale --------
    st.R1U = nda::array<cplx, 2>(np, np); st.R1T = nda::array<cplx, 2>(np, np);
    st.R3U = nda::array<cplx, 2>(np, np); st.R3T = nda::array<cplx, 2>(np, np);
    st.R1U() = cplx(0.0); st.R1T() = cplx(0.0); st.R3U() = cplx(0.0); st.R3T() = cplx(0.0);
    nda::array<double, 1> Fr(M), g(nk), x(N);
    double err_max = 0.0;
    for (int which = 1; which <= 3; which += 2)
      for (long j = 0; j < npf; ++j) {
        double fmax = 0.0;
        const double fj = b.fd[size_t(j)].f;
        const cplx bfj(beta * fj);
        for (long i = 0; i < ntau; ++i) {
          const double s = ss[size_t(i)];
          const double kf = imag_axes_ft::dlr_kF(beta, s, b.eps(j));
          const cplx t = (which == 1)
              ? cplx(kf) * (bfj / inu + psi_nu(inu, s))
              : cplx(kf) * (cplx(2.0) * bfj / (inu * inu) - (bfj / inu) * phi_nu(inu, s) + cplx(2.0) * xi_nu(inu, s) - s * psi_nu(inu, s));
          Fr(i) = t.real();
          Fr(ntau + i) = t.imag();
          fmax = std::max(fmax, std::abs(t));
        }
        // x = V S^-1 U^T F on the kept directions
        for (long k = 0; k < nk; ++k) {
          double acc = 0.0;
          for (long i = 0; i < M; ++i) acc += U(i, k) * Fr(i);
          g(k) = acc / sig(k);
        }
        for (long c = 0; c < N; ++c) {
          double acc = 0.0;
          for (long k = 0; k < nk; ++k) acc += VT(k, c) * g(k);
          x(c) = acc;
        }
        // residual on the dense grid
        double rmax = 0.0;
        for (long i = 0; i < M; ++i) {
          double acc = 0.0;
          for (long c = 0; c < N; ++c) acc += Ar(i, c) * x(c);
          rmax = std::max(rmax, std::abs(acc - Fr(i)));
        }
        err_max = std::max(err_max, (fmax > 0.0) ? rmax / fmax : rmax);
        for (long c = 0; c < npf; ++c) {
          const cplx xu = cplx(x(c), x(ncol + c)) / cn(c);
          const cplx xt = cplx(x(npf + c), x(ncol + npf + c)) / cn(npf + c);
          if (which == 1) { st.R1U(j, c) = xu; st.R1T(j, c) = xt; }
          else            { st.R3U(j, c) = xu; st.R3T(j, c) = xt; }
        }
      }
    st.fit_err = err_max;
    return st;
  }


  /** l0_apply_shift with the RHS columns batched into the gemms (the inu != 0 twin of l0_apply_cols): the
   *  same terms and tables, per (k, G node) ONE (nc x nc)(nc x ncomp nR nc) gemm and its two follow-ups
   *  instead of nR sets. Gated against l0_apply_ref by the toy tests (L)/(G) at inu != 0. */
  inline void l0_apply_shift_cols(freq_basis const &b, pair_poles const &P, shift_tables const &st,
                                  tf_vector const &X, tf_vector &F, nda::array<cplx, 4> &Fsum,
                                  nda::array<cplx, 3> const *Cb_cst) {
    decltype(nda::range::all) all;
    const cplx inu = st.inu;
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR, nc2 = nc * nc;
    utils::check(st.np == np, "dynbse::l0_apply_shift_cols: table / basis size mismatch.");
    F.zero();
    Fsum() = cplx(0.0);
    const long ncomp = 1 + 2 * np;
    double tfold = 0.0;
    if (char const *e = std::getenv("COQUI_DYNBSE_TFOLD")) tfold = std::atof(e);
#pragma omp parallel for schedule(dynamic, 1) num_threads(utils::omp_threads())
    for (long ik = 0; ik < nk; ++ik) {
      nda::array<cplx, 3> Ghat(ng, nc, nc), Gtil(ng, nc, nc);
      nda::array<cplx, 2> gjT(nc, nc), glT(nc, nc);
      // accumulators over all columns: (part, node, x, r, y)
      nda::array<cplx, 5> AU(2, np, nc, nR, nc), AT(2, np, nc, nR, nc), M2(2, np, nc, nR, nc), A1(2, np, nc, nR, nc),
          A3(2, np, nc, nR, nc);   // (part, node, x, r, y): the order of the V blocks
      nda::array<cplx, 4> Vt(nc, ncomp, nR, nc), Sv(ncomp, nR, nc, nc);
      nda::array<cplx, 2> Pj(nc, ncomp * nR * nc), Qj(nc * ncomp * nR, nc), Bj(nc * ncomp * nR, nc);
      nda::array<cplx, 2> Sl_out(ncomp * nR * nc, nc), Slt(nc, ncomp * nR * nc), Rl(nc, ncomp * nR * nc);
      nda::array<cplx, 3> mV(nc, nR, nc), tV(nc, nR, nc);
      AU() = cplx(0.0); AT() = cplx(0.0); M2() = cplx(0.0); A1() = cplx(0.0); A3() = cplx(0.0);
      // ---- the elementary multiplications on a (x, r, y) block ----------------------------------
      auto mulU = [&](long j, long c, auto const &V, int part) {
        const long nj = P.gnode(j);
        const double ej = P.epsG(j);
        if (c == 0) {                                   // U_j . C
          for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
            AU(part, nj, x, r, y) += V(x, r, y);
        } else if (c <= np) {                           // U_j . U_a
          const long a = c - 1;
          if (a == nj) {
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
              M2(part, nj, x, r, y) += V(x, r, y);
          } else {
            const cplx w = cplx(1.0 / (ej - b.eps(a)));
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
              const cplx v = w * V(x, r, y);
              AU(part, nj, x, r, y) += v;
              AU(part, a, x, r, y) -= v;
            }
          }
        } else {                                        // U_j . T_a
          const long a = c - 1 - np;
          if (a == nj) {                                // U_j T_j = R1_j
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
              A1(part, nj, x, r, y) += V(x, r, y);
          } else {
            const double ea = b.eps(a);
            const cplx w = cplx(1.0 / (ea - ej));
            const cplx d = cplx(ej - ea) + inu;
            const cplx wd = w / d;
            const cplx wt = w - inu * wd;
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
              const cplx v = V(x, r, y);
              AT(part, a, x, r, y) += wt * v;
              AU(part, nj, x, r, y) -= wd * v;
              AU(part, a, x, r, y) += wd * v;
            }
          }
        }
      };
      auto mulT = [&](long l, long c, auto const &V, int part) {
        const long nl = P.gnode(l);
        const double el = P.epsG(l);
        if (c == 0) {                                   // T_l . C
          for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
            AT(part, nl, x, r, y) += V(x, r, y);
        } else if (c <= np) {                           // T_l . U_a
          const long a = c - 1;
          if (a == nl) {                                // U_l T_l = R1_l
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
              A1(part, nl, x, r, y) += V(x, r, y);
          } else {
            const double ea = b.eps(a);
            const cplx w = cplx(1.0 / (el - ea));
            const cplx d = cplx(ea - el) + inu;
            const cplx wd = w / d;
            const cplx wt = w - inu * wd;
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
              const cplx v = V(x, r, y);
              AT(part, nl, x, r, y) += wt * v;
              AU(part, a, x, r, y) -= wd * v;
              AU(part, nl, x, r, y) += wd * v;
            }
          }
        } else {                                        // T_l . T_a
          const long a = c - 1 - np;
          if (a == nl) {                                // T_l^2 = R3_l
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y)
              A3(part, nl, x, r, y) += V(x, r, y);
          } else {
            const double ea = b.eps(a);
            const double g = el - ea;
            const cplx w2 = cplx(1.0 / (g * g));
            const cplx dla = cplx(el - ea) + inu, dal = cplx(ea - el) + inu;
            const cplx wla = w2 / dla, wal = w2 / dal;
            const cplx ctl = w2 - inu * wal, cta = w2 - inu * wla, cul = wal - wla, cua = wla - wal;
            for (long x = 0; x < nc; ++x) for (long r = 0; r < nR; ++r) for (long y = 0; y < nc; ++y) {
              const cplx v = V(x, r, y);
              AT(part, nl, x, r, y) += ctl * v;
              AT(part, a, x, r, y) += cta * v;
              AU(part, nl, x, r, y) += cul * v;
              AU(part, a, x, r, y) += cua * v;
            }
          }
        }
      };
      Ghat() = cplx(0.0);
      Gtil() = cplx(0.0);
      for (long j = 0; j < ng; ++j)
        for (long l = 0; l < ng; ++l) {
          if (j == l) continue;
          const cplx w = cplx(1.0) / (cplx(P.epsG(j) - P.epsG(l)) + inu);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              Ghat(j, x, y) += w * P.gkq(l, ik, y, x);
              Gtil(l, x, y) += w * P.gk(j, ik, y, x);
            }
        }
      // the input components: C (comp 0), U_a (1..np), T_a (np+1..2np), for all columns
      bool anyc = false, anyv = false;
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y)
          for (long r = 0; r < nR; ++r) {
            const cplx cv = X.cst(ik, x, y, r);
            Vt(x, 0, r, y) = cv; Sv(0, r, x, y) = cv;
            anyc = anyc or (cv != cplx(0.0));
            for (long a = 0; a < np; ++a) {
              const cplx u = X.fam(0, a, ik, x, y, r), t = X.fam(1, a, ik, x, y, r);
              Vt(x, 1 + a, r, y) = u; Sv(1 + a, r, x, y) = u;
              Vt(x, 1 + np + a, r, y) = t; Sv(1 + np + a, r, x, y) = t;
              anyv = anyv or (u != cplx(0.0)) or (t != cplx(0.0));
            }
          }
      if (anyc or anyv) {
        auto Vt2 = nda::reshape(Vt, std::array<long, 2>{nc, ncomp * nR * nc});
        auto Sv2 = nda::reshape(Sv, std::array<long, 2>{ncomp * nR * nc, nc});
        auto part_of = [&](long c) { return (c == 0) ? 1 : 0; };
        for (long j = 0; j < ng; ++j) {
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) { gjT(x, y) = P.gk(j, ik, y, x); glT(x, y) = P.gkq(j, ik, y, x); }
          nda::blas::gemm(gjT, Vt2, Pj);                                        // (p1', (c r y))
          auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * ncomp * nR, nc});
          nda::blas::gemm(Pj2, Ghat(j, all, all), Qj);                          // Q_j(c, r): ((p1' c r), p3)
          nda::blas::gemm(Pj2, glT, Bj);                                        // B_j(c, r)
          auto Q4 = nda::reshape(Qj, std::array<long, 4>{nc, ncomp, nR, nc});
          auto B4 = nda::reshape(Bj, std::array<long, 4>{nc, ncomp, nR, nc});
          for (long c = 0; c < ncomp; ++c) {
            if (c == 0 and not anyc) continue;
            mulU(j, c, Q4(all, c, all, all), part_of(c));     // + U_j . Q_j
            mulT(j, c, B4(all, c, all, all), part_of(c));     // + T_j . B_j
          }
        }
        for (long l = 0; l < ng; ++l) {
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);
          nda::blas::gemm(Gtil(l, all, all), Vt2, Pj);                          // (p1', (c r y))
          auto Pl2 = nda::reshape(Pj, std::array<long, 2>{nc * ncomp * nR, nc});
          nda::blas::gemm(Pl2, glT, Qj);                                        // R_l(c, r): ((p1' c r), p3)
          auto R4 = nda::reshape(Qj, std::array<long, 4>{nc, ncomp, nR, nc});
          for (long c = 0; c < ncomp; ++c) {
            if (c == 0 and not anyc) continue;
            for (long x = 0; x < nc; ++x)
              for (long r = 0; r < nR; ++r)
                for (long y = 0; y < nc; ++y) {
                  const cplx v = R4(x, c, r, y);
                  mV(x, r, y) = -v;
                  tV(x, r, y) = inu * v;
                }
            mulU(l, c, mV, part_of(c));                     // - U_l . R_l
            mulT(l, c, tV, part_of(c));                     // + i nu T_l . R_l
          }
        }
      }
      // ---- assemble --------------------------------------------------------------------------
      for (int part = 0; part < 2; ++part) {
        const bool sum_this = (part == 0) or (Cb_cst == nullptr);
        for (long n = 0; n < np; ++n) {
          const cplx wh(b.fhalf(n));
          for (long x = 0; x < nc; ++x)
            for (long r = 0; r < nR; ++r)
              for (long y = 0; y < nc; ++y) {
                const cplx u = AU(part, n, x, r, y), t = AT(part, n, x, r, y);
                F.fam(0, n, ik, x, y, r) += u;
                F.fam(1, n, ik, x, y, r) += t;
                if (sum_this) Fsum(ik, x, y, r) += wh * u;           // T sums to 0 exactly
              }
          bool any2 = false, any1 = false, any3 = false;
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y)
              for (long r = 0; r < nR; ++r) {
                any2 = any2 or (M2(part, n, x, r, y) != cplx(0.0));
                any1 = any1 or (A1(part, n, x, r, y) != cplx(0.0));
                any3 = any3 or (A3(part, n, x, r, y) != cplx(0.0));
              }
          if (any2 or any1 or any3)
            utils::check(n < b.np_fit, "dynbse::l0_apply_shift_cols: a confluent product at node {} outside the DLR set.", n);
          if (any2) {
            const cplx w1(b.fd[size_t(n)].f1);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) {
                  const cplx m = M2(part, n, x, r, y);
                  if (sum_this) Fsum(ik, x, y, r) += w1 * m;
                  for (long c = 0; c < np; ++c) {
                    const cplx dc = b.Dsq(n, c);
                    if (dc != cplx(0.0)) F.fam(0, c, ik, x, y, r) += dc * m;
                  }
                }
          }
          if (any1) {
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) {
                  const cplx v = A1(part, n, x, r, y);
                  if (sum_this) Fsum(ik, x, y, r) += st.s1(n) * v;
                  F.fam(0, n, ik, x, y, r) += st.r1u(n) * v;
                  for (long c = 0; c < np; ++c) {
                    F.fam(0, c, ik, x, y, r) += st.R1U(n, c) * v;
                    F.fam(1, c, ik, x, y, r) += st.R1T(n, c) * v;
                  }
                }
          }
          if (any3) {
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) {
                  const cplx v = A3(part, n, x, r, y);
                  if (sum_this) Fsum(ik, x, y, r) += st.s3(n) * v;
                  F.fam(0, n, ik, x, y, r) += st.r3u(n) * v;
                  F.fam(1, n, ik, x, y, r) += st.r3t(n) * v;
                  for (long c = 0; c < np; ++c) {
                    F.fam(0, c, ik, x, y, r) += st.R3U(n, c) * v;
                    F.fam(1, c, ik, x, y, r) += st.R3T(n, c) * v;
                  }
                }
          }
        }
      }
      // EXPERIMENT (small-nu spurious mode): fold the twisted components T_a of the DLR nodes with
      // |eps_a| >= ratio |nu| back into the U family through T_a = U_a^2 - i nu U_a^3 + O((nu/eps_a)^2)
      // (Dsq / Dcb re-expansions); env COQUI_DYNBSE_TFOLD = ratio, unset / 0 = off. The frequency sums
      // are untouched (T sums to 0 exactly; they were accumulated from the product form).
      if (tfold > 0.0)
        for (long a = 0; a < np; ++a) {                 // DLR and extension nodes (both carry Dsq / Dcb rows)
          if (std::abs(b.eps(a)) < tfold * std::abs(inu)) continue;
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y)
              for (long r = 0; r < nR; ++r) {
                const cplx t = F.fam(1, a, ik, x, y, r);
                if (t == cplx(0.0)) continue;
                for (long c = 0; c < np; ++c) {
                  const cplx dc = b.Dsq(a, c) - inu * b.Dcb(a, c);
                  if (dc != cplx(0.0)) F.fam(0, c, ik, x, y, r) += dc * t;
                }
                F.fam(1, a, ik, x, y, r) = cplx(0.0);
              }
        }
      if (Cb_cst != nullptr and anyc)
        for (long r = 0; r < nR; ++r)
          for (long p = 0; p < nc2; ++p) {
            cplx sacc(0.0);
            for (long pp = 0; pp < nc2; ++pp) sacc += (*Cb_cst)(ik, p, pp) * X.cst(ik, pp / nc, pp % nc, r);
            Fsum(ik, p / nc, p % nc, r) += sacc;
          }
    }
  }

  /**
   * F = L0 X at inu != 0 in the {U, T} basis (family 1 = the twisted pairs T_a). Grouped form:
   * with the diagonal (j == l) separated,
   *   L0 = sum_{j != l} [g_j (x) g_l / D_jl] (U_j - U_l + i nu T_l) + sum_j [g_j (x) g_j] T_j,
   * so with Q_j = g_j^T X Ghat_j, R_l = Gtil_l X g_l^T, B_j = g_j^T X g_j^T (pair maps applied to
   * every component of X: the constant, the U_a and the T_a)
   *   F = sum_j U_j . Q_j  -  sum_l U_l . R_l  +  i nu sum_l T_l . R_l  +  sum_j T_j . B_j ,
   * and "U_j ." / "T_l ." are the elementary single-pole multiplications of a {C, U, T} vector
   * (partial fractions with node-gap denominators; the confluent U_j^2 -> Dsq, U_j T_j -> R1_j,
   * T_j T_j -> R3_j). Frequency sums: U -> f - 1/2, T -> 0, Dsq -> f', R1 -> f'/(i nu), R3 -> 2 f'/(i nu)^2.
   */
  inline void l0_apply_shift(freq_basis const &b, pair_poles const &P, shift_tables const &st,
                             tf_vector const &X, tf_vector &F, nda::array<cplx, 4> &Fsum,
                             nda::array<cplx, 3> const *Cb_cst) {
    decltype(nda::range::all) all;
    const cplx inu = st.inu;
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR, nc2 = nc * nc;
    utils::check(st.np == np, "dynbse::l0_apply_shift: table / basis size mismatch.");
    F.zero();
    Fsum() = cplx(0.0);
    // host threads over k (omp_threads knob): every k writes its own F(.., ik, ..) / Fsum(ik, ..) slices; the
    // workspaces and accumulators are per iteration (no shared mutable state, no MPI in the body)
#pragma omp parallel for schedule(dynamic, 1) num_threads(utils::omp_threads())
    for (long ik = 0; ik < nk; ++ik) {
    // per-k pole-summed legs, diagonal EXCLUDED
    nda::array<cplx, 3> Ghat(ng, nc, nc), Gtil(ng, nc, nc);
    nda::array<cplx, 2> gjT(nc, nc), glT(nc, nc);
    // accumulators (per (k, r)); index 0 = the family/constant-independent part, 1 = the
    // constant part of X (kept apart for the Cb_cst override of its frequency sum)
    nda::array<cplx, 5> AU(2, 2, np, nc, nc), AT(2, 2, np, nc, nc);   // [part][unused=0][node] single poles: AU(part,0,...) U, AT(part,0,...) T
    nda::array<cplx, 4> M2(2, np, nc, nc), A1(2, np, nc, nc), A3(2, np, nc, nc);   // [part][node]
    // workspaces: the input components stacked as columns (x, (comp), y): comp = 0 (C), 1..np (U), np+1..2np (T)
    const long ncomp = 1 + 2 * np;
    nda::array<cplx, 3> Vt(nc, ncomp, nc), Sv(ncomp, nc, nc);
    nda::array<cplx, 2> Pj(nc, ncomp * nc), Qj(nc * ncomp, nc), Bj(nc * ncomp, nc);
    nda::array<cplx, 2> Sl_out(ncomp * nc, nc), Slt(nc, ncomp * nc), Rl(nc, ncomp * nc);

    // ---- the elementary multiplications --------------------------------------------------
    // mulU(j, comp c, matrix V, part): U_j x (component c of the vector)
    auto mulU = [&](long j, long c, auto const &V, int part) {
      const long nj = P.gnode(j);
      const double ej = P.epsG(j);
      if (c == 0) {                                   // U_j . C
        for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) AU(part, 0, nj, x, y) += V(x, y);
      } else if (c <= np) {                           // U_j . U_a
        const long a = c - 1;
        if (a == nj) {
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) M2(part, nj, x, y) += V(x, y);
        } else {
          const cplx w = cplx(1.0 / (ej - b.eps(a)));
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) {
            AU(part, 0, nj, x, y) += w * V(x, y);
            AU(part, 0, a, x, y) -= w * V(x, y);
          }
        }
      } else {                                        // U_j . T_a
        const long a = c - 1 - np;
        if (a == nj) {                                // U_j T_j = R1_j
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) A1(part, nj, x, y) += V(x, y);
        } else {
          // [T_a - U_j S_a]/(e_a - e_j),  U_j S_a = [U_j - U_a + i nu T_a]/(e_j - e_a + i nu)
          const double ea = b.eps(a);
          const cplx w = cplx(1.0 / (ea - ej));
          const cplx d = cplx(ej - ea) + inu;
          const cplx wd = w / d;
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) {
            const cplx v = V(x, y);
            AT(part, 0, a, x, y) += (w - inu * wd) * v;
            AU(part, 0, nj, x, y) -= wd * v;
            AU(part, 0, a, x, y) += wd * v;
          }
        }
      }
    };
    // mulT(l, comp c, V, part): T_l x (component c)
    auto mulT = [&](long l, long c, auto const &V, int part) {
      const long nl = P.gnode(l);
      const double el = P.epsG(l);
      if (c == 0) {                                   // T_l . C
        for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) AT(part, 0, nl, x, y) += V(x, y);
      } else if (c <= np) {                           // T_l . U_a
        const long a = c - 1;
        if (a == nl) {                                // U_l T_l = R1_l
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) A1(part, nl, x, y) += V(x, y);
        } else {
          // [T_l - U_a S_l]/(e_l - e_a),  U_a S_l = [U_a - U_l + i nu T_l]/(e_a - e_l + i nu)
          const double ea = b.eps(a);
          const cplx w = cplx(1.0 / (el - ea));
          const cplx d = cplx(ea - el) + inu;
          const cplx wd = w / d;
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) {
            const cplx v = V(x, y);
            AT(part, 0, nl, x, y) += (w - inu * wd) * v;
            AU(part, 0, a, x, y) -= wd * v;
            AU(part, 0, nl, x, y) += wd * v;
          }
        }
      } else {                                        // T_l . T_a
        const long a = c - 1 - np;
        if (a == nl) {                                // T_l^2 = R3_l
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) A3(part, nl, x, y) += V(x, y);
        } else {
          // [T_l - U_l S_a - U_a S_l + T_a]/g^2, g = e_l - e_a,
          // U_l S_a = [U_l - U_a + i nu T_a]/d_la, U_a S_l = [U_a - U_l + i nu T_l]/d_al
          const double ea = b.eps(a);
          const double g = el - ea;
          const cplx w2 = cplx(1.0 / (g * g));
          const cplx dla = cplx(el - ea) + inu, dal = cplx(ea - el) + inu;
          const cplx wla = w2 / dla, wal = w2 / dal;
          for (long x = 0; x < nc; ++x) for (long y = 0; y < nc; ++y) {
            const cplx v = V(x, y);
            AT(part, 0, nl, x, y) += (w2 - inu * wal) * v;
            AT(part, 0, a, x, y) += (w2 - inu * wla) * v;
            AU(part, 0, nl, x, y) += (wal - wla) * v;
            AU(part, 0, a, x, y) += (wla - wal) * v;
          }
        }
      }
    };

      Ghat() = cplx(0.0);
      Gtil() = cplx(0.0);
      for (long j = 0; j < ng; ++j)
        for (long l = 0; l < ng; ++l) {
          if (j == l) continue;
          const cplx w = cplx(1.0) / (cplx(P.epsG(j) - P.epsG(l)) + inu);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              Ghat(j, x, y) += w * P.gkq(l, ik, y, x);
              Gtil(l, x, y) += w * P.gk(j, ik, y, x);
            }
        }
      for (long r = 0; r < nR; ++r) {
        AU() = cplx(0.0); AT() = cplx(0.0); M2() = cplx(0.0); A1() = cplx(0.0); A3() = cplx(0.0);
        // the input components: C, U_a, T_a  (the constant part is component 0 = "part 1")
        bool anyc = false, anyv = false;
        for (long x = 0; x < nc; ++x)
          for (long y = 0; y < nc; ++y) {
            const cplx cv = X.cst(ik, x, y, r);
            Vt(x, 0, y) = cv; Sv(0, x, y) = cv;
            anyc = anyc or (cv != cplx(0.0));
            for (long a = 0; a < np; ++a) {
              const cplx u = X.fam(0, a, ik, x, y, r), t = X.fam(1, a, ik, x, y, r);
              Vt(x, 1 + a, y) = u; Sv(1 + a, x, y) = u;
              Vt(x, 1 + np + a, y) = t; Sv(1 + np + a, x, y) = t;
              anyv = anyv or (u != cplx(0.0)) or (t != cplx(0.0));
            }
          }
        if (not anyc and not anyv) continue;
        auto Vt2 = nda::reshape(Vt, std::array<long, 2>{nc, ncomp * nc});
        auto Sv2 = nda::reshape(Sv, std::array<long, 2>{ncomp * nc, nc});
        auto part_of = [&](long c) { return (c == 0) ? 1 : 0; };
        for (long j = 0; j < ng; ++j) {
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) { gjT(x, y) = P.gk(j, ik, y, x); glT(x, y) = P.gkq(j, ik, y, x); }
          nda::blas::gemm(gjT, Vt2, Pj);                                    // (p1', (c b))
          auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * ncomp, nc});
          nda::blas::gemm(Pj2, Ghat(j, all, all), Qj);                      // Q_j(c): ((p1' c), p3)
          nda::blas::gemm(Pj2, glT, Bj);                                    // B_j(c): ((p1' c), p3)
          auto Q3 = nda::reshape(Qj, std::array<long, 3>{nc, ncomp, nc});
          auto B3 = nda::reshape(Bj, std::array<long, 3>{nc, ncomp, nc});
          for (long c = 0; c < ncomp; ++c) {
            if (c == 0 and not anyc) continue;
            mulU(j, c, Q3(all, c, all), part_of(c));        // + U_j . Q_j
            mulT(j, c, B3(all, c, all), part_of(c));        // + T_j . B_j
          }
        }
        for (long l = 0; l < ng; ++l) {
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);
          nda::blas::gemm(Sv2, glT, Sl_out);                                // ((c x), p3)
          auto So = nda::reshape(Sl_out, std::array<long, 3>{ncomp, nc, nc});
          for (long c = 0; c < ncomp; ++c)
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) Slt(x, c * nc + y) = So(c, x, y);
          nda::blas::gemm(Gtil(l, all, all), Slt, Rl);                      // R_l(c): (p1', (c p3))
          auto R3v = nda::reshape(Rl, std::array<long, 3>{nc, ncomp, nc});
          nda::array<cplx, 2> mV(nc, nc), tV(nc, nc);
          for (long c = 0; c < ncomp; ++c) {
            if (c == 0 and not anyc) continue;
            mV() = -R3v(all, c, all);
            tV() = inu * R3v(all, c, all);
            mulU(l, c, mV, part_of(c));                     // - U_l . R_l
            mulT(l, c, tV, part_of(c));                     // + i nu T_l . R_l
          }
        }
        // ---- assemble ----------------------------------------------------------------------
        for (int part = 0; part < 2; ++part) {
          const bool sum_this = (part == 0) or (Cb_cst == nullptr);
          for (long n = 0; n < np; ++n) {
            const cplx wh(b.fhalf(n));
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) {
                const cplx u = AU(part, 0, n, x, y), t = AT(part, 0, n, x, y);
                F.fam(0, n, ik, x, y, r) += u;
                F.fam(1, n, ik, x, y, r) += t;
                if (sum_this) Fsum(ik, x, y, r) += wh * u;           // T sums to 0 exactly
              }
            // U_n^2 -> Dsq ; U_n^2 S_n -> R1 ; U_n^2 S_n^2 -> R3
            bool any2 = false, any1 = false, any3 = false;
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) {
                any2 = any2 or (M2(part, n, x, y) != cplx(0.0));
                any1 = any1 or (A1(part, n, x, y) != cplx(0.0));
                any3 = any3 or (A3(part, n, x, y) != cplx(0.0));
              }
            if (any2 or any1 or any3)
              utils::check(n < b.np_fit, "dynbse::l0_apply_shift: a confluent product at node {} outside the DLR set.", n);
            if (any2) {
              const cplx w1(b.fd[size_t(n)].f1);
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) {
                  if (sum_this) Fsum(ik, x, y, r) += w1 * M2(part, n, x, y);
                  for (long c = 0; c < np; ++c) {
                    const cplx dc = b.Dsq(n, c);
                    if (dc != cplx(0.0)) F.fam(0, c, ik, x, y, r) += dc * M2(part, n, x, y);
                  }
                }
            }
            if (any1) {
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) {
                  const cplx v = A1(part, n, x, y);
                  if (sum_this) Fsum(ik, x, y, r) += st.s1(n) * v;
                  F.fam(0, n, ik, x, y, r) += st.r1u(n) * v;
                  for (long c = 0; c < np; ++c) {
                    F.fam(0, c, ik, x, y, r) += st.R1U(n, c) * v;
                    F.fam(1, c, ik, x, y, r) += st.R1T(n, c) * v;
                  }
                }
            }
            if (any3) {
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) {
                  const cplx v = A3(part, n, x, y);
                  if (sum_this) Fsum(ik, x, y, r) += st.s3(n) * v;
                  F.fam(0, n, ik, x, y, r) += st.r3u(n) * v;
                  F.fam(1, n, ik, x, y, r) += st.r3t(n) * v;
                  for (long c = 0; c < np; ++c) {
                    F.fam(0, c, ik, x, y, r) += st.R3U(n, c) * v;
                    F.fam(1, c, ik, x, y, r) += st.R3T(n, c) * v;
                  }
                }
            }
          }
        }
        if (Cb_cst != nullptr and anyc)
          for (long p = 0; p < nc2; ++p) {
            cplx sacc(0.0);
            for (long pp = 0; pp < nc2; ++pp) sacc += (*Cb_cst)(ik, p, pp) * X.cst(ik, pp / nc, pp % nc, r);
            Fsum(ik, p / nc, p % nc, r) += sacc;
          }
      }
    }
  }


  /** the RHS columns batched: the same terms as l0_apply's inu = 0 path (below), with the column index r
   *  folded into the gemms' free dimension -- per (k, family, G node) ONE (nc x nc)(nc x np nc nR) and ONE
   *  ((nc np nR) x nc)(nc x nc) gemm instead of nR pairs of tiny ones. Gated against l0_apply_ref by the
   *  toy test (L). Host threads over k as in l0_apply. */
  inline bool &l0_cols_state() { static bool v = true; return v; }
  inline void l0_apply_cols(freq_basis const &b, pair_poles const &P, tf_vector const &X, tf_vector &F,
                            nda::array<cplx, 4> &Fsum, nda::array<cplx, 3> const *Cb_cst) {
    decltype(nda::range::all) all;
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR, nc2 = nc * nc;
    F.zero();
    Fsum() = cplx(0.0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(utils::omp_threads())
    for (long ik = 0; ik < nk; ++ik) {
      nda::array<cplx, 3> Ghat(ng, nc, nc), Gtil(ng, nc, nc);
      nda::array<cplx, 2> gjT(nc, nc), glT(nc, nc);
      // per-k accumulators over all columns: (fam, node, x, y, r)
      nda::array<cplx, 5> F1(2, np, nc, nR, nc), M2(2, np, nc, nR, nc), F1c(2, np, nc, nR, nc);   // (fam, node, x, r, y)
      nda::array<cplx, 4> M3(np, nc, nR, nc), M2c(np, nc, nR, nc);
      // workspaces: the input stacked as (x, (a r y)) and ((a r x), y)
      nda::array<cplx, 4> Vt(nc, np, nR, nc), Sv(np, nR, nc, nc);
      nda::array<cplx, 2> Pj(nc, np * nR * nc), Qj(nc * np * nR, nc);
      nda::array<cplx, 2> Sl_out(np * nR * nc, nc), Slt(nc, np * nR * nc), Rl(nc, np * nR * nc);
      nda::array<cplx, 3> C(nc, nR, nc);                         // (x, r, y)
      nda::array<cplx, 2> T(nc, nR * nc), Bm(nc * nR, nc);
      F1() = cplx(0.0); M2() = cplx(0.0); M3() = cplx(0.0); F1c() = cplx(0.0); M2c() = cplx(0.0);
      Ghat() = cplx(0.0);
      Gtil() = cplx(0.0);
      for (long j = 0; j < ng; ++j)
        for (long l = 0; l < ng; ++l) {
          if (j == l) continue;
          const cplx w = cplx(1.0) / cplx(P.epsG(j) - P.epsG(l));
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              Ghat(j, x, y) += w * P.gkq(l, ik, y, x);
              Gtil(l, x, y) += w * P.gk(j, ik, y, x);
            }
        }
      bool anyc = false;
      for (long x = 0; x < nc; ++x)
        for (long y = 0; y < nc; ++y)
          for (long r = 0; r < nR; ++r) {
            const cplx v = X.cst(ik, x, y, r);
            C(x, r, y) = v;
            anyc = anyc or (v != cplx(0.0));
          }
      // the single input family at inu = 0 (both families folded)
      bool anyv = false;
      for (long a = 0; a < np; ++a)
        for (long x = 0; x < nc; ++x)
          for (long y = 0; y < nc; ++y)
            for (long r = 0; r < nR; ++r) {
              const cplx v = X.fam(0, a, ik, x, y, r) + X.fam(1, a, ik, x, y, r);
              Vt(x, a, r, y) = v;
              Sv(a, r, x, y) = v;
              anyv = anyv or (v != cplx(0.0));
            }
      if (anyv) {
        auto Vt2 = nda::reshape(Vt, std::array<long, 2>{nc, np * nR * nc});
        auto Sv2 = nda::reshape(Sv, std::array<long, 2>{np * nR * nc, nc});
        // ---- j side: Q_j(a, r) = g_j^T V_{a r} Ghat_j --------------------------------------------
        for (long j = 0; j < ng; ++j) {
          const long nj = P.gnode(j);
          const double ej = P.epsG(j);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) gjT(x, y) = P.gk(j, ik, y, x);
          nda::blas::gemm(gjT, Vt2, Pj);                                        // (p1', (a r y))
          auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * np * nR, nc});   // ((p1' a r), y)
          nda::blas::gemm(Pj2, Ghat(j, all, all), Qj);                          // ((p1' a r), p3)
          auto Q4 = nda::reshape(Qj, std::array<long, 4>{nc, np, nR, nc});
          for (long a = 0; a < np; ++a) {
            const double ea = b.eps(a);
            if (a == nj) {
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) M2(0, nj, x, r, y) += Q4(x, a, r, y);
            } else {
              const cplx c = cplx(1.0 / (ej - ea));
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) {
                    const cplx v = c * Q4(x, a, r, y);
                    F1(0, nj, x, r, y) += v;
                    F1(0, a, x, r, y) -= v;
                  }
            }
          }
        }
        // ---- l side: R_l(a, r) = [Gtil_l V_{a r}] g_l^T (the j-side pattern: no transposes) ------
        for (long l = 0; l < ng; ++l) {
          const long nl = P.gnode(l);
          const double el = P.epsG(l);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);      // (b, p3)
          nda::blas::gemm(Gtil(l, all, all), Vt2, Pj);                          // (p1', (a r y))
          auto Pl2 = nda::reshape(Pj, std::array<long, 2>{nc * np * nR, nc});   // ((p1' a r), y)
          nda::blas::gemm(Pl2, glT, Qj);                                        // ((p1' a r), p3)
          auto R4 = nda::reshape(Qj, std::array<long, 4>{nc, np, nR, nc});
          for (long a = 0; a < np; ++a) {
            const double ea = b.eps(a);
            // - U_l U_a
            if (a == nl) {
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) M2(0, nl, x, r, y) -= R4(x, a, r, y);
            } else {
              const cplx c = cplx(1.0 / (el - ea));
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) {
                    const cplx v = c * R4(x, a, r, y);
                    F1(0, nl, x, r, y) -= v;
                    F1(0, a, x, r, y) += v;
                  }
            }
          }
        }
        // ---- the confluent U_j^2 x U_a terms (j == l) --------------------------------------------
        for (long j = 0; j < ng; ++j) {
          const long nj = P.gnode(j);
          const double ej = P.epsG(j);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              gjT(x, y) = P.gk(j, ik, y, x);
              glT(x, y) = P.gkq(j, ik, y, x);
            }
          nda::blas::gemm(gjT, Vt2, Pj);                                        // (p1', (a r y))
          auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * np * nR, nc});
          nda::blas::gemm(Pj2, glT, Qj);                                        // ((p1' a r), p3) = B_j(a, r)
          auto B4 = nda::reshape(Qj, std::array<long, 4>{nc, np, nR, nc});
          for (long a = 0; a < np; ++a) {
            const double ea = b.eps(a);
            if (a == nj) {
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) M3(nj, x, r, y) += B4(x, a, r, y);
            } else {
              const double dd = ea - ej;
              const cplx c2 = cplx(1.0 / (ej - ea)), c1 = cplx(1.0 / (dd * dd));
              for (long x = 0; x < nc; ++x)
                for (long r = 0; r < nR; ++r)
                  for (long y = 0; y < nc; ++y) {
                    const cplx v = B4(x, a, r, y);
                    M2(0, nj, x, r, y) += c2 * v;
                    F1(0, nj, x, r, y) -= c1 * v;
                    F1(0, a, x, r, y) += c1 * v;
                  }
            }
          }
        }
      }
      // ---- the constant part -------------------------------------------------------------------
      if (anyc) {
        auto C2 = nda::reshape(C, std::array<long, 2>{nc, nR * nc});             // (x, (r y))
        for (long j = 0; j < ng; ++j) {
          const long nj = P.gnode(j);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) gjT(x, y) = P.gk(j, ik, y, x);
          nda::blas::gemm(gjT, C2, T);                                          // (p1', (r y))
          auto T2 = nda::reshape(T, std::array<long, 2>{nc * nR, nc});          // ((p1' r), y)
          nda::blas::gemm(T2, Ghat(j, all, all), Bm);                           // ((p1' r), p3) = Qc_j
          for (long x = 0; x < nc; ++x)
            for (long r = 0; r < nR; ++r)
              for (long y = 0; y < nc; ++y) F1c(0, nj, x, r, y) += Bm(x * nR + r, y);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(j, ik, y, x);
          nda::blas::gemm(T2, glT, Bm);                                         // Bc_j = g_j^T C g_j^T
          for (long x = 0; x < nc; ++x)
            for (long r = 0; r < nR; ++r)
              for (long y = 0; y < nc; ++y) M2c(nj, x, r, y) += Bm(x * nR + r, y);
        }
        for (long l = 0; l < ng; ++l) {
          const long nl = P.gnode(l);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);
          // C g_l^T per column: ((x r), y) . (y, p3)
          auto Cr = nda::reshape(C, std::array<long, 2>{nc * nR, nc});          // ((x r), y)
          nda::blas::gemm(Cr, glT, Bm);                                         // ((x r), p3)
          for (long x = 0; x < nc; ++x)
            for (long r = 0; r < nR; ++r)
              for (long y = 0; y < nc; ++y) T(x, r * nc + y) = Bm(x * nR + r, y);   // (x, (r p3))
          nda::array<cplx, 2> Rc(nc, nR * nc);
          nda::blas::gemm(Gtil(l, all, all), T, Rc);                            // (p1', (r p3)) = Rc_l
          for (long x = 0; x < nc; ++x)
            for (long r = 0; r < nR; ++r)
              for (long y = 0; y < nc; ++y) F1c(0, nl, x, r, y) -= Rc(x, r * nc + y);
        }
      }
      // ---- assemble: single poles, re-expanded double/triple poles, the frequency sums -----------
      for (long f = 0; f < 2; ++f)
        for (long n = 0; n < np; ++n) {
          bool any1 = false, any2 = false, anyc1 = false;
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y)
              for (long r = 0; r < nR; ++r) {
                any1 = any1 or (F1(f, n, x, r, y) != cplx(0.0));
                any2 = any2 or (M2(f, n, x, r, y) != cplx(0.0));
                anyc1 = anyc1 or (F1c(f, n, x, r, y) != cplx(0.0));
              }
          if (any1 or anyc1) {
            const cplx wh(b.fhalf(n));
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) {
                  const cplx v = F1(f, n, x, r, y), vc = F1c(f, n, x, r, y);
                  F.fam(f, n, ik, x, y, r) += v + vc;
                  Fsum(ik, x, y, r) += wh * v;
                  if (Cb_cst == nullptr) Fsum(ik, x, y, r) += wh * vc;
                }
          }
          if (any2) {
            const cplx w1(b.fd[size_t(n)].f1);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) Fsum(ik, x, y, r) += w1 * M2(f, n, x, r, y);
            if (n < b.np_fit) {
              for (long c = 0; c < np; ++c) {
                const cplx dc = b.Dsq(n, c);
                if (dc == cplx(0.0)) continue;
                for (long x = 0; x < nc; ++x)
                  for (long y = 0; y < nc; ++y)
                    for (long r = 0; r < nR; ++r) F.fam(f, c, ik, x, y, r) += dc * M2(f, n, x, r, y);
              }
            } else {
              utils::check(f == 0, "dynbse::l0_apply_cols: a double pole of the shifted family at a G node.");
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y)
                  for (long r = 0; r < nR; ++r) F.fam(1, n, ik, x, y, r) += M2(0, n, x, r, y);
            }
          }
        }
      for (long n = 0; n < np; ++n) {
        bool any2c = false, any3 = false;
        for (long x = 0; x < nc; ++x)
          for (long y = 0; y < nc; ++y)
            for (long r = 0; r < nR; ++r) {
              any2c = any2c or (M2c(n, x, r, y) != cplx(0.0));
              any3 = any3 or (M3(n, x, r, y) != cplx(0.0));
            }
        if (any2c) {
          const cplx w1(b.fd[size_t(n)].f1);
          if (Cb_cst == nullptr)
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) Fsum(ik, x, y, r) += w1 * M2c(n, x, r, y);
          if (n < b.np_fit) {
            for (long c = 0; c < np; ++c) {
              const cplx dc = b.Dsq(n, c);
              if (dc == cplx(0.0)) continue;
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y)
                  for (long r = 0; r < nR; ++r) F.fam(0, c, ik, x, y, r) += dc * M2c(n, x, r, y);
            }
          } else {
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) F.fam(1, n, ik, x, y, r) += M2c(n, x, r, y);
          }
        }
        if (any3) {
          utils::check(n < b.np_fit, "dynbse::l0_apply_cols: a confluent triple pole at node {} outside the DLR set.", n);
          const cplx w2(0.5 * b.fd[size_t(n)].f2);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y)
              for (long r = 0; r < nR; ++r) Fsum(ik, x, y, r) += w2 * M3(n, x, r, y);
          for (long c = 0; c < np; ++c) {
            const cplx dc = b.Dcb(n, c);
            if (dc == cplx(0.0)) continue;
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y)
                for (long r = 0; r < nR; ++r) F.fam(0, c, ik, x, y, r) += dc * M3(n, x, r, y);
          }
        }
      }
      // the constant part's frequency sum from the supplied chi0
      if (Cb_cst != nullptr and anyc)
        for (long r = 0; r < nR; ++r)
          for (long p = 0; p < nc2; ++p) {
            cplx sacc(0.0);
            for (long pp = 0; pp < nc2; ++pp) sacc += (*Cb_cst)(ik, p, pp) * C(pp / nc, r, pp % nc);
            Fsum(ik, p / nc, p % nc, r) += sacc;
          }
    }
  }

  /**
   * F = L0 X, the GROUPED (production) form of l0_apply_ref: identical terms, reassociated so
   * that the G-pole double sum never meets the vertex nodes. Per (k, r) and input family the
   * partial fractions of every term factor through the two (nc x nc) "pole-summed legs"
   *   Ghat_j = sum_l g_l(k+q)^T / D_jl,   Gtil_l = sum_j g_j(k)^T / D_jl,   D_jl = e_j - e_l + inu
   * (l == j excluded at inu = 0, where it is the confluent U_j^2 handled separately), so that
   * every output is built from Q_j(a) = g_j^T V_a Ghat_j and R_l(a) = Gtil_l V_a g_l^T --
   * O(ng np nc^3) small gemms per (k, r, family) instead of O(ng^2 np nc^3). Confluent products
   * (a == the G node of j or l; U_j^2 at inu = 0) collect the double/triple-pole matrices and
   * are re-expanded with the Dsq/Dcb tables; their frequency sums use the exact f', f''/2.
   *
   * `Cb_cst` (nk, nc^2, nc^2) or nullptr: when given, the frequency sum of the CONSTANT part of
   * X is taken as Cb_cst(k) . X.cst(k) instead of the pole route -- the production driver passes
   * the ladder's own tau-route chi0 here so the static limit reproduces the L2 resolvent to
   * working precision (the two-family output itself is unchanged).
   */
  inline void l0_apply(freq_basis const &b, pair_poles const &P, cplx inu, bool shared,
                       tf_vector const &X, tf_vector &F, nda::array<cplx, 4> &Fsum,
                       nda::array<cplx, 3> const *Cb_cst = nullptr, shift_tables const *st = nullptr) {
    decltype(nda::range::all) all;
    // inu != 0: the {U, T} twisted-pair basis (D2e). The {U, S} branches below this dispatch are
    // the inu = 0 (folded, single-family) path only.
    if (inu != cplx(0.0)) {
      utils::check(st != nullptr and st->inu == inu,
                   "dynbse::l0_apply: inu != 0 needs the shift tables built for this inu (build_shift_tables).");
      for (long j = 0; j < P.ng; ++j)
        utils::check(P.gnode(j) < b.np and (not shared or P.gnode(j) < b.np_fit), "dynbse::l0_apply: G node map out of range.");
      if (l0_cols_state()) l0_apply_shift_cols(b, P, *st, X, F, Fsum, Cb_cst);
      else l0_apply_shift(b, P, *st, X, F, Fsum, Cb_cst);
      return;
    }
    if (l0_cols_state()) {
      const long np_ = b.np;
      for (long j = 0; j < P.ng; ++j)
        utils::check(P.gnode(j) < np_ and (not shared or P.gnode(j) < b.np_fit), "dynbse::l0_apply: G node map out of range.");
      utils::check(X.np == np_ and X.nk == P.nk and X.nc == P.nc, "dynbse::l0_apply: shape mismatch.");
      l0_apply_cols(b, P, X, F, Fsum, Cb_cst);
      return;
    }
    const long np = b.np, nk = P.nk, nc = P.nc, ng = P.ng, nR = X.nR, nc2 = nc * nc;
    utils::check(X.np == np and X.nk == nk and X.nc == nc, "dynbse::l0_apply: shape mismatch.");
    for (long j = 0; j < ng; ++j)
      utils::check(P.gnode(j) < np and (not shared or P.gnode(j) < b.np_fit), "dynbse::l0_apply: G node map out of range.");
    if (Cb_cst != nullptr)
      utils::check(Cb_cst->shape(0) == nk and Cb_cst->shape(1) == nc2 and Cb_cst->shape(2) == nc2,
                   "dynbse::l0_apply: Cb_cst shape mismatch.");
    const bool nu0 = (inu == cplx(0.0));
    const long nfam_in = nu0 ? 1 : 2;
    F.zero();
    Fsum() = cplx(0.0);
    // host threads over k (omp_threads knob): every k writes its own F(.., ik, ..) / Fsum(ik, ..) slices; the
    // workspaces and accumulators are per iteration (no shared mutable state, no MPI in the body)
#pragma omp parallel for schedule(dynamic, 1) num_threads(utils::omp_threads())
    for (long ik = 0; ik < nk; ++ik) {

    // per-k pole-summed legs: Ghat_j(b, p3) = sum_l gkq(l, p3, b)/D_jl ; Gtil_l(p1', a) = sum_j gk(j, a, p1')/D_jl
    nda::array<cplx, 3> Ghat(ng, nc, nc), Gtil(ng, nc, nc);
    nda::array<cplx, 2> gjT(nc, nc), glT(nc, nc);
    // per-(k, r) accumulators: single poles per family, double poles per family, triple poles;
    // the constant part's single and double poles kept apart (their sums may be overridden)
    nda::array<cplx, 4> F1(2, np, nc, nc), M2(2, np, nc, nc), F1c(2, np, nc, nc);
    nda::array<cplx, 3> M3(np, nc, nc), M2c(np, nc, nc);
    // workspaces
    nda::array<cplx, 3> Vt(nc, np, nc);            // Vt(x, a, y) = V_a(x, y)
    nda::array<cplx, 3> Sv(np, nc, nc);            // Sv(a, x, y) = V_a(x, y)
    nda::array<cplx, 2> Pj(nc, np * nc), Qj(nc * np, nc);
    nda::array<cplx, 2> Sl_out(np * nc, nc), Slt(nc, np * nc), Rl(nc, np * nc);
    nda::array<cplx, 2> C(nc, nc), T(nc, nc), Bm(nc, nc);

      Ghat() = cplx(0.0);
      Gtil() = cplx(0.0);
      for (long j = 0; j < ng; ++j)
        for (long l = 0; l < ng; ++l) {
          if (nu0 and j == l) continue;
          const cplx w = cplx(1.0) / (cplx(P.epsG(j) - P.epsG(l)) + inu);
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              Ghat(j, x, y) += w * P.gkq(l, ik, y, x);
              Gtil(l, x, y) += w * P.gk(j, ik, y, x);
            }
        }
      for (long r = 0; r < nR; ++r) {
        F1() = cplx(0.0); M2() = cplx(0.0); M3() = cplx(0.0); F1c() = cplx(0.0); M2c() = cplx(0.0);
        C() = X.cst(ik, all, all, r);
        bool anyc = false;
        for (auto const &v : C) anyc = anyc or (v != cplx(0.0));
        for (long fin = 0; fin < nfam_in; ++fin) {
          bool anyv = false;
          for (long a = 0; a < np; ++a)
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) {
                cplx v = X.fam(fin, a, ik, x, y, r);
                if (nu0) v += X.fam(1, a, ik, x, y, r);
                Vt(x, a, y) = v;
                Sv(a, x, y) = v;
                anyv = anyv or (v != cplx(0.0));
              }
          if (not anyv) continue;
          auto Vt2 = nda::reshape(Vt, std::array<long, 2>{nc, np * nc});
          auto Sv2 = nda::reshape(Sv, std::array<long, 2>{np * nc, nc});
          // ---- j side: Q_j(a) = g_j^T V_a Ghat_j ----------------------------------------------
          for (long j = 0; j < ng; ++j) {
            const long nj = P.gnode(j);
            const double ej = P.epsG(j);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) gjT(x, y) = P.gk(j, ik, y, x);
            nda::blas::gemm(gjT, Vt2, Pj);                                   // (p1', (a b))
            auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * np, nc});   // ((p1' a), b)
            nda::blas::gemm(Pj2, Ghat(j, all, all), Qj);                     // ((p1' a), p3)
            auto Q3 = nda::reshape(Qj, std::array<long, 3>{nc, np, nc});
            for (long a = 0; a < np; ++a) {
              const double ea = b.eps(a);
              if (fin == 0) {
                if (a == nj) {
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) M2(0, nj, x, y) += Q3(x, a, y);
                } else {
                  const cplx c = cplx(1.0 / (ej - ea));
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) {
                      F1(0, nj, x, y) += c * Q3(x, a, y);
                      F1(0, a, x, y) -= c * Q3(x, a, y);
                    }
                }
              } else {
                // U_j S_a = [U_j - S_a] / (e_j - e_a + inu)
                const cplx c = cplx(1.0) / (cplx(ej - ea) + inu);
                for (long x = 0; x < nc; ++x)
                  for (long y = 0; y < nc; ++y) {
                    F1(0, nj, x, y) += c * Q3(x, a, y);
                    F1(1, a, x, y) -= c * Q3(x, a, y);
                  }
              }
            }
          }
          // ---- l side: R_l(a) = Gtil_l V_a g_l^T ---------------------------------------------
          for (long l = 0; l < ng; ++l) {
            const long nl = P.gnode(l);
            const double el = P.epsG(l);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);   // (b, p3)
            nda::blas::gemm(Sv2, glT, Sl_out);                                 // ((a x), p3)
            auto So = nda::reshape(Sl_out, std::array<long, 3>{np, nc, nc});
            for (long a = 0; a < np; ++a)
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) Slt(x, a * nc + y) = So(a, x, y);   // (x, (a p3))
            nda::blas::gemm(Gtil(l, all, all), Slt, Rl);                       // (p1', (a p3))
            auto R3 = nda::reshape(Rl, std::array<long, 3>{nc, np, nc});
            for (long a = 0; a < np; ++a) {
              const double ea = b.eps(a);
              if (fin == 0) {
                if (nu0) {
                  // - U_l U_a
                  if (a == nl) {
                    for (long x = 0; x < nc; ++x)
                      for (long y = 0; y < nc; ++y) M2(0, nl, x, y) -= R3(x, a, y);
                  } else {
                    const cplx c = cplx(1.0 / (el - ea));
                    for (long x = 0; x < nc; ++x)
                      for (long y = 0; y < nc; ++y) {
                        F1(0, nl, x, y) -= c * R3(x, a, y);
                        F1(0, a, x, y) += c * R3(x, a, y);
                      }
                  }
                } else {
                  // - S_l U_a = -[U_a - S_l] / (e_a - e_l + inu)
                  const cplx c = cplx(1.0) / (cplx(ea - el) + inu);
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) {
                      F1(0, a, x, y) -= c * R3(x, a, y);
                      F1(1, nl, x, y) += c * R3(x, a, y);
                    }
                }
              } else {
                // - S_l S_a
                if (a == nl) {
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) M2(1, nl, x, y) -= R3(x, a, y);
                } else {
                  const cplx c = cplx(1.0 / (el - ea));
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) {
                      F1(1, nl, x, y) -= c * R3(x, a, y);
                      F1(1, a, x, y) += c * R3(x, a, y);
                    }
                }
              }
            }
          }
          // ---- inu = 0: the confluent U_j^2 x U_a terms (j == l) ----------------------------
          if (nu0) {
            for (long j = 0; j < ng; ++j) {
              const long nj = P.gnode(j);
              const double ej = P.epsG(j);
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) {
                  gjT(x, y) = P.gk(j, ik, y, x);
                  glT(x, y) = P.gkq(j, ik, y, x);
                }
              nda::blas::gemm(gjT, Vt2, Pj);                                 // (p1', (a b))
              auto Pj2 = nda::reshape(Pj, std::array<long, 2>{nc * np, nc});
              nda::blas::gemm(Pj2, glT, Qj);                                 // ((p1' a), p3) = B_j(a)
              auto B3 = nda::reshape(Qj, std::array<long, 3>{nc, np, nc});
              for (long a = 0; a < np; ++a) {
                const double ea = b.eps(a);
                if (a == nj) {
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) M3(nj, x, y) += B3(x, a, y);
                } else {
                  const double dd = ea - ej;
                  const cplx c2 = cplx(1.0 / (ej - ea)), c1 = cplx(1.0 / (dd * dd));
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) {
                      M2(0, nj, x, y) += c2 * B3(x, a, y);
                      F1(0, nj, x, y) -= c1 * B3(x, a, y);
                      F1(0, a, x, y) += c1 * B3(x, a, y);
                    }
                }
              }
            }
          }
        }
        // ---- the constant part ----------------------------------------------------------------
        if (anyc) {
          for (long j = 0; j < ng; ++j) {
            const long nj = P.gnode(j);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) gjT(x, y) = P.gk(j, ik, y, x);
            nda::blas::gemm(gjT, C, T);
            nda::blas::gemm(T, Ghat(j, all, all), Bm);              // Qc_j
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) F1c(0, nj, x, y) += Bm(x, y);
            if (nu0) {
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(j, ik, y, x);
              nda::blas::gemm(T, glT, Bm);                          // Bc_j = g_j^T C g_j^T
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) M2c(nj, x, y) += Bm(x, y);
            }
          }
          const long fo = nu0 ? 0 : 1;
          for (long l = 0; l < ng; ++l) {
            const long nl = P.gnode(l);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) glT(x, y) = P.gkq(l, ik, y, x);
            nda::blas::gemm(C, glT, T);
            nda::blas::gemm(Gtil(l, all, all), T, Bm);              // Rc_l
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) F1c(fo, nl, x, y) -= Bm(x, y);
          }
        }
        // ---- assemble: single poles, re-expanded double/triple poles, the frequency sums -------
        for (long f = 0; f < 2; ++f)
          for (long n = 0; n < np; ++n) {
            bool any1 = false, any2 = false, anyc1 = false;
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) {
                any1 = any1 or (F1(f, n, x, y) != cplx(0.0));
                any2 = any2 or (M2(f, n, x, y) != cplx(0.0));
                anyc1 = anyc1 or (F1c(f, n, x, y) != cplx(0.0));
              }
            if (any1 or anyc1) {
              const cplx wh(b.fhalf(n));
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) {
                  const cplx v = F1(f, n, x, y), vc = F1c(f, n, x, y);
                  F.fam(f, n, ik, x, y, r) += v + vc;
                  Fsum(ik, x, y, r) += wh * v;
                  if (Cb_cst == nullptr) Fsum(ik, x, y, r) += wh * vc;
                }
            }
            if (any2) {
              const cplx w1(b.fd[size_t(n)].f1);
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) Fsum(ik, x, y, r) += w1 * M2(f, n, x, y);
              if (n < b.np_fit) {
                for (long c = 0; c < np; ++c) {
                  const cplx dc = b.Dsq(n, c);
                  if (dc == cplx(0.0)) continue;
                  for (long x = 0; x < nc; ++x)
                    for (long y = 0; y < nc; ++y) F.fam(f, c, ik, x, y, r) += dc * M2(f, n, x, y);
                }
              } else {
                // the union's G node: U_n^2 is kept as the second family's basis function
                // (tau kernel KF2 = -K_F (s - beta f_n)), no re-expansion
                utils::check(f == 0, "dynbse::l0_apply: a double pole of the shifted family at a G node.");
                for (long x = 0; x < nc; ++x)
                  for (long y = 0; y < nc; ++y) F.fam(1, n, ik, x, y, r) += M2(0, n, x, y);
              }
            }
          }
        for (long n = 0; n < np; ++n) {
          bool any2c = false, any3 = false;
          for (long x = 0; x < nc; ++x)
            for (long y = 0; y < nc; ++y) {
              any2c = any2c or (M2c(n, x, y) != cplx(0.0));
              any3 = any3 or (M3(n, x, y) != cplx(0.0));
            }
          if (any2c) {
            const cplx w1(b.fd[size_t(n)].f1);
            if (Cb_cst == nullptr)
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) Fsum(ik, x, y, r) += w1 * M2c(n, x, y);
            if (n < b.np_fit) {
              for (long c = 0; c < np; ++c) {
                const cplx dc = b.Dsq(n, c);
                if (dc == cplx(0.0)) continue;
                for (long x = 0; x < nc; ++x)
                  for (long y = 0; y < nc; ++y) F.fam(0, c, ik, x, y, r) += dc * M2c(n, x, y);
              }
            } else {
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) F.fam(1, n, ik, x, y, r) += M2c(n, x, y);
            }
          }
          if (any3) {
            utils::check(n < b.np_fit, "dynbse::l0_apply: a confluent triple pole at node {} outside the DLR set.", n);
            const cplx w2(0.5 * b.fd[size_t(n)].f2);
            for (long x = 0; x < nc; ++x)
              for (long y = 0; y < nc; ++y) Fsum(ik, x, y, r) += w2 * M3(n, x, y);
            for (long c = 0; c < np; ++c) {
              const cplx dc = b.Dcb(n, c);
              if (dc == cplx(0.0)) continue;
              for (long x = 0; x < nc; ++x)
                for (long y = 0; y < nc; ++y) F.fam(0, c, ik, x, y, r) += dc * M3(n, x, y);
            }
          }
        }
        // the constant part's frequency sum from the supplied chi0
        if (Cb_cst != nullptr and anyc)
          for (long p = 0; p < nc2; ++p) {
            cplx sacc(0.0);
            for (long pp = 0; pp < nc2; ++pp) sacc += (*Cb_cst)(ik, p, pp) * C(pp / nc, pp % nc);
            Fsum(ik, p / nc, p % nc, r) += sacc;
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
                         nda::array<cplx, 4> const &Fsum, tf_vector &y, cplx inu = cplx(0.0)) {
    const long np = b.np, nt = b.nt, nk = F.nk, nc = F.nc, nc2 = nc * nc, nR = F.nR;
    y.zero();
    const bool nu0 = (inu == cplx(0.0));
    // inu = 0: family 1 (U_a^2 at the union's G nodes, kernel KF2) is an ordinary fermionic
    // function -> its tau values join family 0's and ONE refit (into family 0) follows;
    // inu != 0: family 1 = the twisted pairs (kernel K_F, the phi_nu factor rides along) -> its own refit.
    const long nfam = nu0 ? 1 : 2;
    // tau values of each family: (nt, nk, nc2, nR)
    nda::array<cplx, 4> Fs(nt, nk, nc2, nR), Ys(nt, nk, nc2, nR);
    nda::array<cplx, 1> v(nc2), w(nc2);
    double fit_err = 0.0;
    for (long fam = 0; fam < nfam; ++fam) {
      Fs() = cplx(0.0);
      for (long ik = 0; ik < nk; ++ik)
        for (long r = 0; r < nR; ++r)
          for (long ff = fam; ff < (nu0 ? 2 : fam + 1); ++ff)
            for (long a = 0; a < np; ++a) {
              pack_pair(F.fam(ff, a, ik, nda::range::all, nda::range::all, r), nc, v);
              bool anyv = false;
              for (auto const &x : v) anyv = anyv or (x != cplx(0.0));
              if (not anyv) continue;
              for (long i = 0; i < nt; ++i) {
                const double kf = (ff == 1 and nu0) ? b.KF2(i, a) : b.KF(i, a);
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

  /**
   * The PHYSICAL (tau-value) inner product on the vertex-grid functions: <u, v> = sum over
   * (family, k, pair, column) of the tau-grid dot product of the functions the coefficients
   * represent, G_ab = sum_i K_F(s_i, e_a) K_F(s_i, e_b) (a, b < np_fit; the union's G nodes carry no
   * iterate content). The coefficient inner product is blind to the near-confluent pairs whose
   * coefficients grow with every L0 application while their values cancel; in the tau metric the
   * Arnoldi process and the residual see the functions (measured on lih222 at the first bosonic
   * node: coefficient-norm Ritz 2.6 and a stalled GMRES, physical readouts continuous in nu).
   */
  struct tf_metric {
    long np = 0, np_fit = 0;
    nda::array<cplx, 2> G;         // (np, np), zero outside the np_fit block
  };
  inline tf_metric build_tf_metric(freq_basis const &b) {
    tf_metric m;
    m.np = b.np; m.np_fit = b.np_fit;
    m.G = nda::array<cplx, 2>(b.np, b.np);
    m.G() = cplx(0.0);
    double gmax = 0.0;
    for (long a = 0; a < b.np_fit; ++a)
      for (long c = 0; c < b.np_fit; ++c) {
        double acc = 0.0;
        for (long i = 0; i < b.nt; ++i) acc += b.KF(i, a) * b.KF(i, c);
        m.G(a, c) = cplx(acc);
        gmax = std::max(gmax, std::abs(acc));
      }
    // normalized so that a unit coefficient on the largest-norm node has unit norm
    if (gmax > 0.0) m.G() /= cplx(gmax);
    return m;
  }

  /** per-column inner products <a, b>_r = sum conj(a) b over (fam, node, k, pair) -- (nR);
   *  with a metric: the tau-value inner product (family blocks through G, the constant plainly). */
  inline void tf_dots(tf_vector const &a, tf_vector const &b, nda::array<cplx, 1> &out,
                      tf_metric const *met = nullptr) {
    const long nR = a.nR;
    out() = cplx(0.0);
    if (met != nullptr) {
      const long np = a.np, nk = a.nk, nc = a.nc, nrest = nk * nc * nc * nR;
      utils::check(met->np == np, "dynbse::tf_dots: metric size mismatch.");
      // Gb(fam) = G . b(fam) as (np, nrest) gemm (on the contiguous view, no copy), then the per-column
      // sums conj(a) Gb with a contiguous inner loop over the columns; the p loop is threaded (omp_threads)
      // with per-thread accumulators combined in a fixed order (1e-12-class reproducible, not bitwise)
      nda::array<cplx, 2> Gb(np, nrest);
      const long nthr = std::max(1l, std::min(utils::omp_threads(), np));
      nda::array<cplx, 2> acc(nthr, nR);
      for (long fam = 0; fam < 2; ++fam) {
        auto bv = nda::reshape(b.fam(fam, nda::ellipsis{}), std::array<long, 2>{np, nrest});
        nda::blas::gemm(met->G, bv, Gb);
        auto av = nda::reshape(a.fam(fam, nda::ellipsis{}), std::array<long, 2>{np, nrest});
        acc() = cplx(0.0);
#pragma omp parallel for schedule(static) num_threads(nthr)
        for (long p = 0; p < np; ++p) {
#ifdef _OPENMP
          const long t = omp_get_thread_num();
#else
          const long t = 0;
#endif
          const cplx *ap = &av(p, 0);
          const cplx *gp = &Gb(p, 0);
          cplx *o = &acc(t, 0);
          for (long i0 = 0; i0 < nrest; i0 += nR)
            for (long r = 0; r < nR; ++r) o[r] += std::conj(ap[i0 + r]) * gp[i0 + r];
        }
        for (long t = 0; t < nthr; ++t)
          for (long r = 0; r < nR; ++r) out(r) += acc(t, r);
      }
      const cplx *pa = a.cst.data(); const cplx *pb = b.cst.data();
      const long ncst = long(a.cst.size()) / nR;
      for (long i = 0; i < ncst; ++i)
        for (long r = 0; r < nR; ++r) out(r) += std::conj(pa[i * nR + r]) * pb[i * nR + r];
      return;
    }
    const long nfam = a.fam.shape(0) * a.fam.shape(1) * a.fam.shape(2) * a.fam.shape(3) * a.fam.shape(4);
    const long ncst = a.cst.shape(0) * a.cst.shape(1) * a.cst.shape(2);
    const cplx *pa = a.fam.data(), *pb = b.fam.data();
    for (long i = 0; i < nfam; ++i)
      for (long r = 0; r < nR; ++r) out(r) += std::conj(pa[i * nR + r]) * pb[i * nR + r];
    pa = a.cst.data(); pb = b.cst.data();
    for (long i = 0; i < ncst; ++i)
      for (long r = 0; r < nR; ++r) out(r) += std::conj(pa[i * nR + r]) * pb[i * nR + r];
  }
  /** y += alpha_r x (per column) */
  inline void tf_axpy(nda::array<cplx, 1> const &alpha, tf_vector const &x, tf_vector &y) {
    const long nR = x.nR;
    const long nfam = long(x.fam.size()) / nR, ncst = long(x.cst.size()) / nR;
    cplx *py = y.fam.data(); const cplx *px = x.fam.data();
    for (long i = 0; i < nfam; ++i)
      for (long r = 0; r < nR; ++r) py[i * nR + r] += alpha(r) * px[i * nR + r];
    py = y.cst.data(); px = x.cst.data();
    for (long i = 0; i < ncst; ++i)
      for (long r = 0; r < nR; ++r) py[i * nR + r] += alpha(r) * px[i * nR + r];
  }
  /** x *= s_r per column */
  inline void tf_scale(nda::array<cplx, 1> const &sc, tf_vector &x) {
    const long nR = x.nR;
    const long nfam = long(x.fam.size()) / nR, ncst = long(x.cst.size()) / nR;
    cplx *px = x.fam.data();
    for (long i = 0; i < nfam; ++i)
      for (long r = 0; r < nR; ++r) px[i * nR + r] *= sc(r);
    px = x.cst.data();
    for (long i = 0; i < ncst; ++i)
      for (long r = 0; r < nR; ++r) px[i * nR + r] *= sc(r);
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

  /** T_s = K_s (1 - Cb K_s)^-1 from the dense block-diagonal Cb held by S and the static rung Ks. */
  inline void finish_static_resolvent(static_resolvent &S, nda::array<cplx, 2> const &Ks) {
    const long D = S.D;
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
  }

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
    finish_static_resolvent(S, Ks);
    return S;
  }

  /** the production builder: Cb_k (nk, nc^2, nc^2) the per-k chi0 (any route), Ks (D, D) the
   *  static rung mapping the pair vector at k' (column) to k (row), D = nk nc^2. */
  inline static_resolvent build_static_resolvent_from(nda::array<cplx, 3> const &Cb_k,
                                                      nda::array<cplx, 2> const &Ks) {
    static_resolvent S;
    const long nk = Cb_k.shape(0), nc2 = Cb_k.shape(1), D = nk * nc2;
    utils::check(Ks.shape(0) == D and Ks.shape(1) == D, "dynbse::build_static_resolvent_from: Ks shape.");
    S.D = D; S.nk = nk; S.nc = long(std::lround(std::sqrt(double(nc2))));
    S.Cb = nda::array<cplx, 2>(D, D);
    S.Cb() = cplx(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long pp = 0; pp < nc2; ++pp) S.Cb(ik * nc2 + p, ik * nc2 + pp) = Cb_k(ik, p, pp);
    finish_static_resolvent(S, Ks);
    return S;
  }

  /**
   * Gamma = L_s (D + y): with F = L0 (D + y) and its exact sum Fsum,
   *   Gamma = F + L0 [ T_s Fsum ]   (the second term is L0 applied to a constant),
   *   sum_iw Gamma = Fsum + Cb T_s Fsum.
   * Returns Gamma (two-family) and Gsum. `Dc` is the constant external leg D (nk, nc, nc, nR).
   */
  /** wall-time sinks of the solver internals (seconds, cumulative; the driver resets and reads them per
   *  unit): the L0 pair-pole applications, the static-resolvent gemms (T_s, Cb), the Arnoldi
   *  orthogonalization. Not thread-safe by design: the solver runs outside any omp region. */
  struct solve_timers {
    double t_l0 = 0.0, t_ts = 0.0, t_orth = 0.0;
    void reset() { t_l0 = t_ts = t_orth = 0.0; }
  };
  inline solve_timers &solve_timers_state() { static solve_timers t; return t; }
  inline double wall_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

  inline void ls_apply(freq_basis const &b, pair_poles const &P, static_resolvent const &S, cplx inu,
                       bool shared, nda::array<cplx, 4> const &Dc, tf_vector const &y,
                       tf_vector &Gamma, nda::array<cplx, 4> &Gsum,
                       nda::array<cplx, 3> const *Cb_cst = nullptr, shift_tables const *st = nullptr) {
    decltype(nda::range::all) all;
    auto &stt = solve_timers_state();
    const long nk = P.nk, nc = P.nc, nc2 = nc * nc, nR = y.nR, D = S.D;
    tf_vector X(b.np, nk, nc, nR);
    X.fam() = y.fam;
    X.cst() = Dc + y.cst;
    tf_vector F(b.np, nk, nc, nR);
    nda::array<cplx, 4> Fsum(nk, nc, nc, nR);
    double tw = wall_now();
    l0_apply(b, P, inu, shared, X, F, Fsum, Cb_cst, st);
    stt.t_l0 += wall_now() - tw; tw = wall_now();
    // c = T_s Fsum  (D x nR)
    nda::array<cplx, 2> fs(D, nR), cs(D, nR), cb(D, nR);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r) fs(ik * nc2 + p, r) = Fsum(ik, p / nc, p % nc, r);
    nda::blas::gemm(S.Ts, fs, cs);
    nda::blas::gemm(S.Cb, cs, cb);
    stt.t_ts += wall_now() - tw; tw = wall_now();
    tf_vector Xc(b.np, nk, nc, nR);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r) Xc.cst(ik, p / nc, p % nc, r) = cs(ik * nc2 + p, r);
    tf_vector F2(b.np, nk, nc, nR);
    nda::array<cplx, 4> F2sum(nk, nc, nc, nR);
    l0_apply(b, P, inu, shared, Xc, F2, F2sum, Cb_cst, st);
    stt.t_l0 += wall_now() - tw;
    Gamma.fam() = F.fam + F2.fam;
    Gamma.cst() = cplx(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long p = 0; p < nc2; ++p)
        for (long r = 0; r < nR; ++r)
          Gsum(ik, p / nc, p % nc, r) = Fsum(ik, p / nc, p % nc, r) + cb(ik * nc2 + p, r);
    (void)all;
  }

  // the readout block D^dag Gsum (the physical observable of the iteration)
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

  struct dyson_result {
    long iterations = 0;
    bool converged = false;
    bool stagnated = false;        // the residual stopped decreasing (the tau-refit floor): stopped early
    bool readout_converged = false; // GMRES: the physical readout (D^dag Gsum) moved by less than readout_tol
    std::vector<double> readout_history;   // per cycle: |Delta P| / |P| of the readout block
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
  /**
   * KdOp: double(tf_vector const& F, nda::array<cplx,4> const& Fsum, tf_vector& y) -- applies the
   * dynamic remainder K_d to the two-family F (with its exact frequency sums) and returns the tau
   * refit error. The toy form is kd_apply(b, R, ...); the production driver supplies the THC rung.
   */
  template<class KdOp>
  inline dyson_result solve_dyson_op(freq_basis const &b, pair_poles const &P, KdOp &&kd,
                                     static_resolvent const &S, cplx inu, bool shared,
                                     nda::array<cplx, 4> const &Dc, double tol, long maxit,
                                     bool anderson = true, nda::array<cplx, 3> const *Cb_cst = nullptr,
                                     shift_tables const *st = nullptr, tf_metric const *metric = nullptr) {
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
      ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum, Cb_cst, st);
      if (it == 0) out.Gsum0 = Gsum;
      if (it == 1) out.Gsum1 = Gsum;
      out.Gsum = Gsum;
      out.iterations = it;
      if (it == maxit) break;
      const double fe = kd(Gamma, Gsum, ynew);
      out.fit_err_max = std::max(out.fit_err_max, fe);
      // convergence on the update (the tau metric when given)
      double d2 = 0.0, n2 = 0.0;
      if (metric != nullptr) {
        tf_vector dy(np, nk, nc, nR);
        dy.fam() = ynew.fam - y.fam;
        dy.cst() = ynew.cst - y.cst;
        nda::array<cplx, 1> dd(nR), nn(nR);
        tf_dots(dy, dy, dd, metric);
        tf_dots(ynew, ynew, nn, metric);
        for (long r = 0; r < nR; ++r) { d2 += std::real(dd(r)); n2 += std::real(nn(r)); }
      } else {
        for (long i = 0; i < long(ynew.fam.size()); ++i) {
          const cplx dv = ynew.fam.data()[i] - y.fam.data()[i];
          d2 += std::norm(dv); n2 += std::norm(ynew.fam.data()[i]);
        }
        for (long i = 0; i < long(ynew.cst.size()); ++i) {
          const cplx dv = ynew.cst.data()[i] - y.cst.data()[i];
          d2 += std::norm(dv); n2 += std::norm(ynew.cst.data()[i]);
        }
      }
      const double d = std::sqrt(d2), n = std::sqrt(std::max(n2, 1e-300));
      out.residual = d / n;
      out.history.push_back(out.residual);
      if (dprev > 0.0) out.contraction = d / dprev;
      // stagnation: three consecutive iterations without a 30% decrease -> the refit floor
      {
        const auto &h = out.history;
        const size_t m_ = h.size();
        if (m_ >= 4 and h[m_ - 1] > 0.7 * h[m_ - 2] and h[m_ - 2] > 0.7 * h[m_ - 3] and h[m_ - 3] > 0.7 * h[m_ - 4]
            and h[m_ - 1] < 1e-3) {
          out.stagnated = true;
          ls_apply(b, P, S, inu, shared, Dc, ynew, Gamma, Gsum, Cb_cst, st);
          out.Gsum = Gsum;
          out.iterations = it + 1;
          break;
        }
      }
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
        ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum, Cb_cst, st);
        out.Gsum = Gsum;
        out.iterations = it + 1;
        out.converged = true;
        break;
      }
    }
    return out;
  }

  inline dyson_result solve_dyson(freq_basis const &b, pair_poles const &P, pair_rung const &R,
                                  static_resolvent const &S, cplx inu, bool shared,
                                  nda::array<cplx, 4> const &Dc, double tol, long maxit,
                                  bool anderson = true, shift_tables const *st = nullptr) {
    auto kd = [&](tf_vector const &F, nda::array<cplx, 4> const &Fsum, tf_vector &y) {
      return kd_apply(b, R, F, Fsum, y, inu);
    };
    return solve_dyson_op(b, P, kd, S, inu, shared, Dc, tol, maxit, anderson, nullptr, st);
  }


  // ==================================================================================
  // GMRES(m) ON THE DYNAMIC REMAINDER, batched over the right-hand-side columns
  // ==================================================================================

  /**
   * Restarted GMRES(m) on (1 - K_d L_s) y = K_d L_s D, every right-hand-side column solved with its
   * own Arnoldi coefficients (the operator applications are batched over the columns). Same
   * result contract as solve_dyson_op (Gsum, Gsum1 = the first iterate, Gsum0 = the static limit);
   * `contraction` reports the largest |Ritz value| of K_d L_s seen in the last cycle (the
   * watchdog), `residual` the largest relative residual over the columns, `history` the cycle
   * residuals. Converged when every column's residual <= tol |rhs_r|.
   */
  template<class KdOp>
  inline dyson_result solve_dyson_gmres(freq_basis const &b, pair_poles const &P, KdOp &&kd,
                                        static_resolvent const &S, cplx inu, bool shared,
                                        nda::array<cplx, 4> const &Dc, double tol, long maxit, long m,
                                        nda::array<cplx, 3> const *Cb_cst = nullptr, shift_tables const *st = nullptr,
                                        tf_metric const *metric = nullptr, double readout_tol = 0.0) {
    const long nk = P.nk, nc = P.nc, nR = Dc.shape(3), np = b.np;
    utils::check(m >= 1, "dynbse::solve_dyson_gmres: m >= 1.");
    dyson_result out;
    out.Gsum = nda::array<cplx, 4>(nk, nc, nc, nR);
    out.Gsum1 = nda::array<cplx, 4>(nk, nc, nc, nR);
    out.Gsum0 = nda::array<cplx, 4>(nk, nc, nc, nR);
    nda::array<cplx, 4> Dzero(nk, nc, nc, nR), Gsum(nk, nc, nc, nR);
    Dzero() = cplx(0.0);
    tf_vector y(np, nk, nc, nR), Gamma(np, nk, nc, nR), rhs(np, nk, nc, nR), w(np, nk, nc, nR), r(np, nk, nc, nR);
    // the operator: A v = v - K_d L_s v   (L_s v with a zero external leg)
    auto apply_A = [&](tf_vector const &v, tf_vector &Av) {
      ls_apply(b, P, S, inu, shared, Dzero, v, Gamma, Gsum, Cb_cst, st);
      const double fe = kd(Gamma, Gsum, Av);
      out.fit_err_max = std::max(out.fit_err_max, fe);
      Av.fam() = v.fam - Av.fam;
      Av.cst() = v.cst - Av.cst;
    };
    // rhs = K_d L_s D ; the static limit and the first iterate
    ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum, Cb_cst, st);
    out.Gsum0 = Gsum;
    out.fit_err_max = std::max(out.fit_err_max, kd(Gamma, Gsum, rhs));
    ls_apply(b, P, S, inu, shared, Dc, rhs, Gamma, Gsum, Cb_cst, st);
    out.Gsum1 = Gsum;
    nda::array<cplx, 1> rhs_norm2(nR), dots(nR), sc(nR);
    tf_dots(rhs, rhs, rhs_norm2, metric);
    std::vector<tf_vector> V;
    V.reserve(size_t(m + 1));
    for (long j = 0; j <= m; ++j) V.emplace_back(np, nk, nc, nR);
    nda::array<cplx, 3> H(m + 1, m, nR);          // Hessenberg per column
    nda::array<cplx, 1> beta(nR);
    y.zero();
    long it = 0;
    bool done = false;
    double ritz_max = 0.0;
    const bool ritz_prof = (std::getenv("COQUI_DYNBSE_RITZ") != nullptr);
    double ritz_best = -1.0;
    long c_best = -1, n_best = 0;
    nda::array<cplx, 1> v_best;
    // the readout of the current iterate (the physical observable D^dag sum_iw L0 Gamma): a
    // convergence criterion blind to the readout-invisible modes of the iteration
    nda::array<cplx, 2> Pprev(nR, nR);
    bool have_prev = false;
    auto readout_of = [&]() {
      ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum, Cb_cst, st);
      return collapse(Dc, Gsum);
    };
    while (not done and it < maxit) {
      // r = rhs - A y
      apply_A(y, w);
      r.fam() = rhs.fam - w.fam;
      r.cst() = rhs.cst - w.cst;
      tf_dots(r, r, dots, metric);
      double rel = 0.0;
      for (long c = 0; c < nR; ++c) {
        beta(c) = cplx(std::sqrt(std::real(dots(c))));
        const double nr = std::sqrt(std::max(std::real(rhs_norm2(c)), 1e-300));
        rel = std::max(rel, std::real(beta(c)) / nr);
      }
      out.residual = rel;
      out.history.push_back(rel);
      if (rel <= tol) { done = true; break; }
      // stagnation: two consecutive cycles without a 30% decrease (below 1e-3) -> the refit floor
      {
        const auto &h = out.history;
        const size_t m_ = h.size();
        if (m_ >= 3 and h[m_ - 1] > 0.7 * h[m_ - 2] and h[m_ - 2] > 0.7 * h[m_ - 3] and h[m_ - 1] < 1e-3) {
          out.stagnated = true;
          break;
        }
      }
      for (long c = 0; c < nR; ++c) sc(c) = (std::real(beta(c)) > 0.0) ? cplx(1.0) / beta(c) : cplx(0.0);
      V[0].fam() = r.fam; V[0].cst() = r.cst;
      tf_scale(sc, V[0]);
      H() = cplx(0.0);
      ritz_best = -1.0; c_best = -1;
      long jdone = 0;
      for (long j = 0; j < m; ++j) {
        apply_A(V[size_t(j)], w);
        ++it;
        const double tw_o = wall_now();
        for (long i = 0; i <= j; ++i) {
          tf_dots(V[size_t(i)], w, dots, metric);
          for (long c = 0; c < nR; ++c) { H(i, j, c) = dots(c); sc(c) = -dots(c); }
          tf_axpy(sc, V[size_t(i)], w);
        }
        tf_dots(w, w, dots, metric);
        solve_timers_state().t_orth += wall_now() - tw_o;
        for (long c = 0; c < nR; ++c) {
          const double hn = std::sqrt(std::max(std::real(dots(c)), 0.0));
          H(j + 1, j, c) = cplx(hn);
          sc(c) = (hn > 1e-300) ? cplx(1.0 / hn) : cplx(0.0);
        }
        V[size_t(j + 1)].fam() = w.fam; V[size_t(j + 1)].cst() = w.cst;
        tf_scale(sc, V[size_t(j + 1)]);
        jdone = j + 1;
        // least squares per column: min || beta e1 - H_j z ||, via the normal equations (m small)
        double rel_j = 0.0;
        for (long c = 0; c < nR; ++c) {
          const long n = jdone;
          nda::matrix<cplx> N(n, n);
          nda::array<cplx, 1> g(n);
          for (long a = 0; a < n; ++a) {
            g(a) = std::conj(H(0, a, c)) * beta(c);
            for (long bb2 = 0; bb2 < n; ++bb2) {
              cplx acc(0.0);
              for (long i = 0; i <= n; ++i) acc += std::conj(H(i, a, c)) * H(i, bb2, c);
              N(a, bb2) = acc;
            }
          }
          nda::inverse_in_place(N);
          // residual = || beta e1 - H z ||
          nda::array<cplx, 1> z(n);
          for (long a = 0; a < n; ++a) { cplx acc(0.0); for (long bb2 = 0; bb2 < n; ++bb2) acc += N(a, bb2) * g(bb2); z(a) = acc; }
          double res2 = 0.0;
          for (long i = 0; i <= n; ++i) {
            cplx acc = (i == 0) ? beta(c) : cplx(0.0);
            for (long a = 0; a < n; ++a) acc -= H(i, a, c) * z(a);
            res2 += std::norm(acc);
          }
          const double nr = std::sqrt(std::max(std::real(rhs_norm2(c)), 1e-300));
          rel_j = std::max(rel_j, std::sqrt(res2) / nr);
        }
        if (rel_j <= tol or it >= maxit) break;
      }
      // (the update of y and the Ritz estimate follow; the readout check comes after the update)
      for (long c = 0; c < nR; ++c) {
        const long n = jdone;
        nda::matrix<cplx> N(n, n);
        nda::array<cplx, 1> g(n), z(n);
        for (long a = 0; a < n; ++a) {
          g(a) = std::conj(H(0, a, c)) * beta(c);
          for (long bb2 = 0; bb2 < n; ++bb2) {
            cplx acc(0.0);
            for (long i = 0; i <= n; ++i) acc += std::conj(H(i, a, c)) * H(i, bb2, c);
            N(a, bb2) = acc;
          }
        }
        nda::inverse_in_place(N);
        for (long a = 0; a < n; ++a) { cplx acc(0.0); for (long bb2 = 0; bb2 < n; ++bb2) acc += N(a, bb2) * g(bb2); z(a) = acc; }
        for (long a = 0; a < n; ++a) {
          nda::array<cplx, 1> al(nR);
          al() = cplx(0.0);
          al(c) = z(a);
          tf_axpy(al, V[size_t(a)], y);
        }
        // Ritz values of A = 1 - K_d L_s from the square part of H: |1 - lambda_A| = |Ritz of K_d L_s|
        if (n >= 1) {
          nda::matrix<cplx> Hs(n, n);
          for (long a = 0; a < n; ++a)
            for (long bb2 = 0; bb2 < n; ++bb2) Hs(a, bb2) = H(a, bb2, c);
          // power iteration on (1 - Hs) for the dominant |Ritz| of K_d L_s (n <= m, cheap)
          nda::array<cplx, 1> v(n), u(n);
          for (long a = 0; a < n; ++a) v(a) = cplx(1.0 / std::sqrt(double(n)));
          double lam = 0.0;
          for (int itp = 0; itp < 50; ++itp) {
            for (long a = 0; a < n; ++a) {
              cplx acc = v(a);
              for (long bb2 = 0; bb2 < n; ++bb2) acc -= Hs(a, bb2) * v(bb2);
              u(a) = acc;
            }
            double nn = 0.0;
            for (long a = 0; a < n; ++a) nn += std::norm(u(a));
            lam = std::sqrt(nn);
            if (lam <= 0.0) break;
            for (long a = 0; a < n; ++a) v(a) = u(a) / lam;
          }
          if (lam > ritz_best) { ritz_best = lam; c_best = c; v_best = v; n_best = n; }
          ritz_max = std::max(ritz_max, lam);
        }
      }
      out.contraction = ritz_max;
      if (ritz_prof and c_best >= 0) {
        // the Ritz vector of the dominant |Ritz(K_d L_s)| in the pair basis: family and node weights
        nda::array<double, 2> wf(2, np);
        wf() = 0.0;
        double wc = 0.0, wt = 0.0;
        for (long p = 0; p < np; ++p)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < nc; ++a)
              for (long bb2 = 0; bb2 < nc; ++bb2) {
                cplx x0(0.0), x1(0.0);
                for (long i = 0; i < n_best; ++i) {
                  x0 += v_best(i) * V[size_t(i)].fam(0, p, k, a, bb2, c_best);
                  x1 += v_best(i) * V[size_t(i)].fam(1, p, k, a, bb2, c_best);
                }
                wf(0, p) += std::norm(x0); wf(1, p) += std::norm(x1);
              }
        for (long k = 0; k < nk; ++k)
          for (long a = 0; a < nc; ++a)
            for (long bb2 = 0; bb2 < nc; ++bb2) {
              cplx x(0.0);
              for (long i = 0; i < n_best; ++i) x += v_best(i) * V[size_t(i)].cst(k, a, bb2, c_best);
              wc += std::norm(x);
            }
        double wu = 0.0;
        for (long p = 0; p < np; ++p) { wu += wf(0, p); wt += wf(1, p); }
        const double tot = std::max(wu + wt + wc, 1e-300);
        std::string top;
        for (int fam_ = 0; fam_ < 2; ++fam_) {
          std::vector<long> idx(static_cast<size_t>(np), 0l);
          for (long p = 0; p < np; ++p) idx[size_t(p)] = p;
          std::sort(idx.begin(), idx.end(), [&](long x, long y) { return wf(fam_, x) > wf(fam_, y); });
          top += (fam_ == 0) ? "  U:" : "  T:";
          for (long t = 0; t < std::min<long>(6, np); ++t) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), " (%.3g:%.2f)", b.eps(idx[size_t(t)]), wf(fam_, idx[size_t(t)]) / tot);
            top += buf;
          }
        }
        app_log(2, "  [dynbse ritz] inu = {:.4e}i cycle {}: dominant |Ritz| {:.3e} (column {}); weights U {:.3f} T {:.3f} cst {:.3f};"
                   " top nodes (eps:weight){}", inu.imag(), out.history.size(), ritz_best, c_best, wu / tot, wt / tot, wc / tot, top);
      }
      if (readout_tol > 0.0) {
        auto Pnow = readout_of();
        if (have_prev) {
          double dm = 0.0, sm = 0.0;
          for (long i = 0; i < nR; ++i)
            for (long j = 0; j < nR; ++j) {
              dm = std::max(dm, std::abs(Pnow(i, j) - Pprev(i, j)));
              sm = std::max(sm, std::abs(Pnow(i, j)));
            }
          const double rel = (sm > 0.0) ? dm / sm : dm;
          out.readout_history.push_back(rel);
          if (rel <= readout_tol) { out.readout_converged = true; done = true; }
        }
        Pprev = Pnow;
        have_prev = true;
      }
    }
    out.iterations = it;
    out.converged = done;
    ls_apply(b, P, S, inu, shared, Dc, y, Gamma, Gsum, Cb_cst, st);
    out.Gsum = Gsum;
    return out;
  }

  /** the readout block: P(r', r) = sum_k sum_{ab} conj(Dleft(k, a, b, r')) Gsum(k, a, b, r) */

} // namespace dynbse
} // namespace solvers
} // namespace methods

#endif // COQUI_VERTEX_DYNBSE_HPP
