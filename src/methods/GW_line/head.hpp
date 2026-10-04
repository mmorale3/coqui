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

#ifndef COQUI_METHODS_GW_LINE_HEAD_HPP
#define COQUI_METHODS_GW_LINE_HEAD_HPP

/**
 * S9a: the Coulomb HEAD on the line (notes section 5, subsection "Head of W and the q -> 0 divergence").
 *
 * 1. Head of the inverse dielectric function at finite q (copy of methods/GW/g0_div_utils.hpp::eval_eps_inv_q):
 *      h(q, zeta) = eps^{-1}_{00}(q, zeta) - 1 = f(q) sum_PQ cb_P(q) W_PQ(q, zeta) conj(cb_Q(q)),   f(q) = (|q|^2 / 4 pi) Omega,
 *    cb = thc.basis_bar_head() (q, P). h(Gamma) = 0 (f = 0). From the block-layout W at the bosonic nodes (head_nodes_partial)
 *    and, equivalently, from the residues of Eq. brep (head_residues_partial):
 *      h(q, zeta)   = sum_j [ hp_j(q) / (zeta - nu_j) - hh_j(q) / (zeta + nu_j) ],
 *      hp_j(q)      = f(q) sum_PQ      cb_P(q)  w_j(q)_PQ  conj(cb_Q(q)),
 *      hh_j(q)      = f(q) sum_PQ conj(cb_P(q)) w_j(-q)_PQ      cb_Q(q)      (the hole part of W(q) carries w(-q)^T).
 *    Exactly hp_j = hh_j >= 0 (time reversal, eps^{-1}(q) = eps^{-1}(-q); positive loss weights); both are kept as computed.
 *    The raw fit residues are NOT unique (the near-threshold directions of the stacked kernel, screened.hpp): measured on the
 *    fixtures max|hp - hh| / max|hp| 3e-5 - 2e-4, |Im hp| and negative Re hp up to 0.1 - 0.6 of max|hp|, while the pole function
 *    is exact to 1e-11. A positive measure (optics, S9b) needs a constrained refit, not these residues.
 *    Per rank: two gemms per (q, residue row or node set) on the local (P_rng, Q_rng) block, then ONE all_reduce of the
 *    [nq, nz] (+ 2 [nq, r]) scalars (head_reduce). Device-safe: copies and nda::blas::gemm on MEM arrays only.
 * 2. q -> 0 extrapolation (copy of div_utils::extrapolate_eps_inv_q0 and its helpers find_n_closest_per_direction,
 *    find_smallest_qabs(_indices), extrapolate_to_q0): every variant CoQui uses is LINEAR in the per-q heads with real
 *    weights (polynomial least squares in |q|^2 with a real design matrix), so it is stored as the weights c_q,
 *      h0(zeta) = sum_q c_q h(q, zeta),     h0 residues: hp0_j = sum_q c_q hp_j(q), hh0_j = sum_q c_q hh_j(q),
 *    and applies unchanged at complex zeta, at the nodes and to the residues (it commutes with the pole fit and with any
 *    transform; CoQui extrapolates on the bosonic Matsubara axis). Variants: "gygi" (axis-folded fit, default since
 *    2026-09-22; "order_N", "perdir", "2d" suffixes as in CoQui), "gygi_smallest_q", "gygi_average". Not supported: "metal"
 *    (sets eps^{-1}(i nu = 0) = 0, a single-frequency override with no line analogue), "gygi_extrplt" (deprecated), "cvv".
 * 3. Exchange (copy of hf_t::HF_K_correction): Sigma_x(k) += -madelung S D(k) S with S = 1 (the overlap of the KS band basis;
 *    CoQui's call passes dyson.sS_skij, = 1 to 1e-8 on every fixture, checked in the [V2] harness).
 * 4. Correlation (transcription of gw_t::Sigma_div_correction): CoQui adds dSigma(tau) = -madelung Re[h0(tau)] T G(tau) T^dagger,
 *    T_ia(k) = sum_P conj(X_Pi(k)) X_Pa(k) conj(chi(Gamma, P)), chi = thc.basis_head(). With G~ = X G X^dagger this is
 *    [X^dagger (G~ o conj(chi) chi^T) X] = T G T^dagger, i.e. the Sigma_c formula of the code with W(q)/N_k replaced by
 *    madelung h0 conj(chi) chi^T at q = Gamma (k - q = k). The line sector formulas (Eqs. siggtr, siglss) with the head
 *    residues (pole +nu_j: hp0_j, pole -nu_j: -hh0_j) give, per k and sector, in closed form (the ray products of the two pole
 *    sums transform exactly; no time nodes are needed):
 *      dSigma^>(k, zeta) = madelung sum_{m in >} sum_j hp0_j T c_m T^dagger / (zeta - e_m - nu_j),
 *      dSigma^<(k, zeta) = madelung sum_{m in <} sum_j hh0_j T c_m T^dagger / (zeta - e_m + nu_j),
 *    (positive weights for the exact h_j >= 0). Difference to CoQui: Re[h0(tau)] is not taken; the analytic h0 of
 *    the line is used (its residues are real and hp0 = hh0 up to the fit / THC noise; measured in [V3][gygi]).
 *    Host only: per k one [nz x M] kernel (M poles of the sector, r bosonic poles) and M rank-1 (or matrix) residues.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "mpi3/communicator.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "utilities/check.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"

namespace methods::gw_line {

/// div_treatment strings of the line code: "ignore_g0" or a gygi variant (see the file header)
inline bool head_div_is_gygi(std::string const &div) { return div.find("gygi") != std::string::npos; }

/// throws for the variants the line code does not support
inline void head_check_variant(std::string const &div) {
  utils::check(div == "ignore_g0" or head_div_is_gygi(div),
               "gw_line: div_treatment must be \"ignore_g0\" or a gygi variant (gygi, gygi_order_N, gygi_perdir, gygi_2d, "
               "gygi_smallest_q, gygi_average); got \"{}\"",
               div);
  utils::check(div.find("metal") == std::string::npos,
               "gw_line: div_treatment \"{}\": the \"metal\" override (eps^-1(i nu = 0) = 0) has no line analogue", div);
  utils::check(div.find("gygi_extrplt") == std::string::npos, "gw_line: div_treatment \"gygi_extrplt\" (deprecated in CoQui) "
                                                               "is not supported");
}

/**
 * Head vectors of this rank's block: cb(q, P_rng), cb(q, Q_rng) (thc.basis_bar_head()), f(q) = (|q|^2 / 4 pi) Omega, qminus.
 * Host arrays (small: nq (nP + nQ)); the contractions copy the vectors of one q to MEM per call.
 */
struct head_basis_t {
  long nq = 0, Np = 0, nP = 0, nQ = 0;
  nda::array<ComplexType, 2> cbP, cbQ;   ///< (nq, nP), (nq, nQ)
  nda::array<double, 1> fac;             ///< (nq)
  std::vector<long> qminus;

  head_basis_t() = default;
  head_basis_t(methods::thc_reader_t const &thc, mf::MF const &mf, aux_grid_t const &grid)
     : nq(mf.nqpts()), Np(thc.Np()), nP(grid.nP), nQ(grid.nQ) {
    utils::check(thc.nqpts() == thc.nqpts_ibz() and mf.nqpts() == mf.nqpts_ibz(), "gw_line::head_basis_t: nosym mesh required");
    auto cb = thc.basis_bar_head();
    utils::check(cb.extent(0) == nq and cb.extent(1) == Np, "gw_line::head_basis_t: basis_bar_head ({}, {}) vs (nq {}, Np {})",
                 cb.extent(0), cb.extent(1), nq, Np);
    cbP = nda::array<ComplexType, 2>(cb(nda::range::all, grid.P_rng()));
    cbQ = nda::array<ComplexType, 2>(cb(nda::range::all, grid.Q_rng()));
    fac = nda::array<double, 1>(nq);
    constexpr double fpi = 4.0 * 3.14159265358979323846;   // as eval_eps_inv_q
    for (long iq = 0; iq < nq; ++iq) {
      auto qp  = mf.Qpts_ibz(iq);
      fac(iq)  = (qp(0) * qp(0) + qp(1) * qp(1) + qp(2) * qp(2)) / fpi * mf.volume();
    }
    auto qm = mf.qminus();
    qminus.resize(nq);
    for (long iq = 0; iq < nq; ++iq) qminus[iq] = qm(iq);
  }
};

namespace detail {
/// printf-style formatting into a std::string (log lines; independent of the fmt configuration)
template <typename... A> std::string head_sfmt(char const *f, A... a) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a...);
  return std::string(buf);
}
/// out(n) = l^T [M (n*nP x nQ)] r for M = reshape(A, {n*nP, nQ}) in MEM, l (nP), r (nQ) host vectors; result on the host
template <MEMORY_SPACE MEM, typename A_t>
nda::array<ComplexType, 1> head_contract(A_t const &A, long n, long nP, long nQ, nda::array<ComplexType, 1> const &l,
                                         nda::array<ComplexType, 1> const &r) {
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  nda::array<ComplexType, 2> lh(nP, 1), rh(nQ, 1);
  lh(nda::range::all, 0) = l;
  rh(nda::range::all, 0) = r;
  arr2_t lm = memory::to_memory_space<MEM>(lh), rm = memory::to_memory_space<MEM>(rh);
  arr2_t buf(n * nP, 1), out(n, 1);
  auto A2 = nda::reshape(A, std::array<long, 2>{n * nP, nQ});
  nda::blas::gemm(ComplexType(1.0), A2, rm, ComplexType(0.0), buf);
  auto B2 = nda::reshape(buf, std::array<long, 2>{n, nP});
  nda::blas::gemm(ComplexType(1.0), B2, lm, ComplexType(0.0), out);
  nda::array<ComplexType, 2> oh = memory::to_memory_space<HOST_MEMORY>(out);
  return nda::array<ComplexType, 1>(oh(nda::range::all, 0));
}
} // namespace detail

/**
 * h(q, zeta_i) at the bosonic nodes from the block-layout W (g, nz, nP, nQ) in MEM, row i = q = qs[i]: this rank's partial
 * sums are ADDED to H (nq, nz) (host); head_reduce finishes. Local, no communication.
 */
template <MEMORY_SPACE MEM>
void head_nodes_partial(memory::array<MEM, ComplexType, 4> const &W, std::vector<long> const &qs, head_basis_t const &hb,
                        nda::array<ComplexType, 2> &H) {
  const long g = W.extent(0), nz = W.extent(1);
  utils::check(long(qs.size()) == g and W.extent(2) == hb.nP and W.extent(3) == hb.nQ,
               "gw_line::head_nodes_partial: W (g {}, nP {}, nQ {}) vs {} q, block ({}, {})", g, W.extent(2), W.extent(3),
               long(qs.size()), hb.nP, hb.nQ);
  if (H.extent(0) != hb.nq or H.extent(1) != nz) {
    H = nda::array<ComplexType, 2>(hb.nq, nz);
    H() = ComplexType(0.0);
  }
  auto all = nda::range::all;
  for (long i = 0; i < g; ++i) {
    const long q = qs[i];
    if (hb.fac(q) == 0.0 or hb.nP == 0 or hb.nQ == 0) continue;   // Gamma: f = 0
    nda::array<ComplexType, 1> l(hb.cbP(q, all)), r(nda::conj(hb.cbQ(q, all)));
    auto o = detail::head_contract<MEM>(W(i, all, all, all), nz, hb.nP, hb.nQ, l, r);
    H(q, all) += hb.fac(q) * o;
  }
}

/**
 * Scalar head residues from the W residues w (nrows, r, nP, nQ) in MEM (or HOST), row i = q' = rows[i]: ADDS this rank's
 * partial sums to hp(q', :) (particle, residues of q') and hh(-q', :) (hole part of -q', which carries w(q')^T):
 *   hp_j(q')  += f(q')  cb(q')^T  w_j(q') conj(cb(q')),     hh_j(q) += f(q) conj(cb(q))^T w_j(q') cb(q),  q = -q'.
 */
template <MEMORY_SPACE MEM>
void head_residues_partial(memory::array<MEM, ComplexType, 4> const &w, std::vector<long> const &rows, head_basis_t const &hb,
                           nda::array<ComplexType, 2> &hp, nda::array<ComplexType, 2> &hh) {
  const long nr = w.extent(0), r = w.extent(1);
  utils::check(long(rows.size()) == nr and w.extent(2) == hb.nP and w.extent(3) == hb.nQ,
               "gw_line::head_residues_partial: w ({}, {}, {}, {}) vs {} rows, block ({}, {})", nr, r, w.extent(2), w.extent(3),
               long(rows.size()), hb.nP, hb.nQ);
  for (auto *h : {&hp, &hh})
    if (h->extent(0) != hb.nq or h->extent(1) != r) {
      *h = nda::array<ComplexType, 2>(hb.nq, r);
      (*h)() = ComplexType(0.0);
    }
  auto all = nda::range::all;
  for (long i = 0; i < nr; ++i) {
    const long q = rows[i], qm = hb.qminus[q];
    if (hb.nP == 0 or hb.nQ == 0) continue;
    auto wi = w(i, all, all, all);
    if (hb.fac(q) != 0.0) {
      nda::array<ComplexType, 1> l(hb.cbP(q, all)), rr(nda::conj(hb.cbQ(q, all)));
      hp(q, all) += hb.fac(q) * detail::head_contract<MEM>(wi, r, hb.nP, hb.nQ, l, rr);
    }
    if (hb.fac(qm) != 0.0) {
      nda::array<ComplexType, 1> l(nda::conj(hb.cbP(qm, all))), rr(hb.cbQ(qm, all));
      hh(qm, all) += hb.fac(qm) * detail::head_contract<MEM>(wi, r, hb.nP, hb.nQ, l, rr);
    }
  }
}

/// ONE all_reduce of the given host arrays (concatenated), in place
inline void head_reduce(boost::mpi3::communicator &comm, std::vector<nda::array<ComplexType, 2> *> const &arrs) {
  long n = 0;
  for (auto *a : arrs) n += a->size();
  nda::array<ComplexType, 1> buf(n);
  long o = 0;
  for (auto *a : arrs) {
    std::copy_n(a->data(), a->size(), buf.data() + o);
    o += a->size();
  }
  comm.all_reduce_in_place_n(buf.data(), n, std::plus<>{});
  o = 0;
  for (auto *a : arrs) {
    std::copy_n(buf.data() + o, a->size(), a->data());
    o += a->size();
  }
}

/// h(zeta) = sum_j [hp_j / (zeta - nu_j) - hh_j / (zeta + nu_j)] for one q (or the extrapolated head)
inline nda::array<ComplexType, 1> head_eval(nda::array<ComplexType, 1> const &hp, nda::array<ComplexType, 1> const &hh,
                                            nda::array<double, 1> const &nu, nda::array<ComplexType, 1> const &zeta) {
  nda::array<ComplexType, 1> h(zeta.size());
  for (long i = 0; i < zeta.size(); ++i) {
    ComplexType s(0.0);
    for (long j = 0; j < nu.size(); ++j) s += hp(j) / (zeta(i) - nu(j)) - hh(j) / (zeta(i) + nu(j));
    h(i) = s;
  }
  return h;
}

/// static eps_inf = 1 / (1 + Re h0(0)), h0(0) = -sum_j (hp0_j + hh0_j) / nu_j  (CoQui: 1 / (1 + Re h0(i nu = 0)))
inline double head_eps_inf(nda::array<ComplexType, 1> const &hp0, nda::array<ComplexType, 1> const &hh0,
                           nda::array<double, 1> const &nu) {
  nda::array<ComplexType, 1> z0(1);
  z0(0) = ComplexType(0.0);
  return 1.0 / (1.0 + head_eval(hp0, hh0, nu, z0)(0).real());
}

// ------------------------------------------------------------------------------------------------------------------
// q -> 0 extrapolation (copied from methods/GW/g0_div_utils.hpp, div_utils; weights instead of values)
// ------------------------------------------------------------------------------------------------------------------
namespace detail {
/// copy of div_utils::find_n_closest_per_direction: up to n indices per +b1, -b1, +b2, -b2, +b3, -b3 (crystal coordinates)
template <typename Q_t, typename L_t>
std::array<std::vector<int>, 6> head_find_n_closest_per_direction(Q_t const &Qpts, L_t const &lattv, int n, double tolerance = 1e-10) {
  utils::check(n > 0, "gw_line head: find_n_closest_per_direction: n must be positive");
  if (Qpts.shape(0) == 1) return {{{0}, {0}, {0}, {0}, {0}, {0}}};
  nda::array<double, 2> Qc(Qpts.shape(0), 3);
  for (long iq = 0; iq < Qpts.shape(0); ++iq) {
    const double tpiinv = 1.0 / (2.0 * 3.14159265358979);   // as CoQui
    for (int a = 0; a < 3; ++a) {
      double s = 0.0;
      for (int b = 0; b < 3; ++b) s += lattv(a, b) * Qpts(iq, b);
      Qc(iq, a) = tpiinv * s;
    }
  }
  std::array<std::vector<int>, 6> res;
  for (int dir = 0; dir < 3; ++dir) {
    std::vector<std::pair<double, int>> pos, neg;
    for (int i = 0; i < Qc.shape(0); ++i) {
      double coord = Qc(i, dir);
      bool other_zero = true;
      for (int o = 0; o < 3; ++o)
        if (o != dir and std::abs(Qc(i, o)) > tolerance) { other_zero = false; break; }
      if (not other_zero) continue;
      if (std::abs(coord + 0.5) < 1e-6) coord = std::abs(coord);   // -0.5 == 0.5
      if (coord > tolerance) pos.push_back({coord, i});
      else if (coord < -tolerance) neg.push_back({std::abs(coord), i});
    }
    std::sort(pos.begin(), pos.end());
    std::sort(neg.begin(), neg.end());
    for (int j = 0; j < std::min(n, int(pos.size())); ++j) res[2 * dir].push_back(pos[j].second);
    for (int j = 0; j < std::min(n, int(neg.size())); ++j) res[2 * dir + 1].push_back(neg[j].second);
  }
  return res;
}

/// copy of div_utils::find_smallest_qabs (including its two_dim conditions)
template <typename Q_t> int head_find_smallest_qabs(Q_t const &Qpts, bool two_dim = false) {
  if (Qpts.shape(0) == 1) return 0;
  double mn = -1;
  int idx   = -1;
  for (int i = 0; i < Qpts.shape(0); ++i) {
    const double nrm = std::sqrt(Qpts(i, 0) * Qpts(i, 0) + Qpts(i, 1) * Qpts(i, 1) + Qpts(i, 2) * Qpts(i, 2));
    if (nrm > 0.0 and mn == -1 and (!two_dim or Qpts(i, 2) != 0.0)) { mn = nrm; idx = i; }
    else if (nrm > 0.0 and nrm < mn and (!two_dim or Qpts(i, 2) == 0.0)) { mn = nrm; idx = i; }
  }
  return idx;
}

/// copy of div_utils::find_smallest_qabs_indices (gamma excluded)
template <typename Q_t> std::vector<int> head_find_smallest_qabs_indices(Q_t const &Qpts, bool two_dim = false) {
  if (Qpts.shape(0) == 1) return {0};
  constexpr double tol = 1e-10;
  double mn = -1.0;
  auto nrm  = [&](int i) { return std::sqrt(Qpts(i, 0) * Qpts(i, 0) + Qpts(i, 1) * Qpts(i, 1) + Qpts(i, 2) * Qpts(i, 2)); };
  for (int i = 0; i < Qpts.shape(0); ++i) {
    if (nrm(i) < tol or (two_dim and std::abs(Qpts(i, 2)) > tol)) continue;
    if (mn < 0.0 or nrm(i) < mn) mn = nrm(i);
  }
  std::vector<int> idx;
  for (int i = 0; i < Qpts.shape(0); ++i) {
    if (nrm(i) < tol or (two_dim and std::abs(Qpts(i, 2)) > tol)) continue;
    if (std::abs(nrm(i) - mn) < tol) idx.push_back(i);
  }
  return idx;
}

/// weights of div_utils::extrapolate_to_q0: x(0) of the least-squares polynomial of order p in |q|^2 (normal equations,
/// as CoQui), i.e. row 0 of (A^T A)^{-1} A^T
inline std::vector<double> head_extrapolate_weights(std::vector<double> const &q2, int p) {
  const long n = q2.size();
  if (n == 1) return {1.0};
  nda::matrix<double> A(n, p + 1);
  for (long i = 0; i < n; ++i) {
    A(i, 0) = 1.0;
    for (int j = 1; j <= p; ++j) A(i, j) = std::pow(q2[i], j);
  }
  nda::matrix<double> AT = nda::transpose(A);
  nda::matrix<double> ATA = AT * A;
  nda::matrix<double> Ai = nda::inverse(ATA);
  nda::matrix<double> B = Ai * AT;
  std::vector<double> c(n);
  for (long i = 0; i < n; ++i) c[i] = B(0, i);
  return c;
}
} // namespace detail

/**
 * The q -> 0 extrapolation as weights c_q (nq): h0 = sum_q c_q h(q). Copy of div_utils::extrapolate_eps_inv_q0 (same branch
 * order: "gygi_average", "gygi_smallest_q", "gygi" with the axis fold / "perdir", "order_N", "2d").
 */
struct head_extrapolation_t {
  std::string variant;
  nda::array<double, 1> c;
  std::vector<std::string> info;   ///< log lines (q indices and orders per axis)

  head_extrapolation_t() = default;
  head_extrapolation_t(mf::MF const &mf, std::string div) : variant(std::move(div)) {
    head_check_variant(variant);
    utils::check(head_div_is_gygi(variant), "gw_line::head_extrapolation_t: a gygi variant is required (got \"{}\")", variant);
    auto Q        = mf.Qpts_ibz();
    const long nq = Q.shape(0);
    c             = nda::array<double, 1>(nq);
    c()           = 0.0;
    const bool two_dim = variant.find("2d") != std::string::npos;
    std::vector<double> q2(nq);
    for (long iq = 0; iq < nq; ++iq) q2[iq] = Q(iq, 0) * Q(iq, 0) + Q(iq, 1) * Q(iq, 1) + Q(iq, 2) * Q(iq, 2);
    if (variant.find("gygi_average") != std::string::npos) {
      auto idx = detail::head_find_smallest_qabs_indices(Q, false);
      for (int i : idx) c(i) += 1.0 / double(idx.size());
      info.push_back(detail::head_sfmt("gygi_average: mean over the %ld q of smallest |q|", long(idx.size())));
      return;
    }
    if (variant.find("gygi_smallest_q") != std::string::npos) {
      const int i = detail::head_find_smallest_qabs(Q, false);
      c(i)        = 1.0;
      info.push_back(detail::head_sfmt("gygi_smallest_q: q index %d", i));
      return;
    }
    int fit_order = 10;   // CoQui default; "order_N" overrides
    if (auto pos = variant.find("order_"); pos != std::string::npos) {
      try {
        fit_order = std::stoi(variant.substr(pos + 6));
      } catch (...) {}
    }
    auto closest = detail::head_find_n_closest_per_direction(Q, mf.lattv(), fit_order + 1);
    const bool axis_fold = (variant.find("perdir") == std::string::npos);
    int ndim             = 0;
    if (axis_fold) {
      constexpr std::array<char const *, 3> lab = {"b1", "b2", "b3"};
      for (int ax = 0; ax < 3; ++ax) {
        if (two_dim and ax >= 2) continue;
        std::vector<std::pair<double, int>> merged;   // (|q|^2, index), closest first, one per |q|
        for (int side = 0; side < 2; ++side)
          for (int idx : closest[2 * ax + side]) {
            bool seen = false;
            for (auto const &m : merged)
              if (std::abs(m.first - q2[idx]) < 1e-12) { seen = true; break; }
            if (not seen) merged.push_back({q2[idx], idx});
          }
        std::sort(merged.begin(), merged.end());
        if (merged.size() > size_t(fit_order + 1)) merged.resize(size_t(fit_order + 1));
        if (merged.empty()) continue;
        std::vector<double> qq;
        for (auto const &m : merged) qq.push_back(m.first);
        auto w = detail::head_extrapolate_weights(qq, int(merged.size()) - 1);
        std::string s = detail::head_sfmt("axis %s: %ld distinct |q| (order %ld):", lab[ax], long(merged.size()), long(merged.size()) - 1);
        for (size_t i = 0; i < merged.size(); ++i) {
          c(merged[i].second) += w[i];
          s += detail::head_sfmt(" q%d (|q|^2 %.5f, w %+.4f)", merged[i].second, merged[i].first, w[i]);
        }
        info.push_back(s);
        ++ndim;
      }
    } else {
      constexpr std::array<char const *, 6> lab = {"+b1", "-b1", "+b2", "-b2", "+b3", "-b3"};
      for (int dir = 0; dir < 6; ++dir) {
        if (two_dim and dir >= 4) continue;
        auto const &ci = closest[dir];
        if (ci.empty()) continue;
        std::vector<double> qq;
        for (int idx : ci) qq.push_back(q2[idx]);
        auto w = detail::head_extrapolate_weights(qq, int(ci.size()) - 1);
        std::string s = detail::head_sfmt("direction %s: %ld points:", lab[dir], long(ci.size()));
        for (size_t i = 0; i < ci.size(); ++i) {
          c(ci[i]) += w[i];
          s += detail::head_sfmt(" q%d (w %+.4f)", ci[i], w[i]);
        }
        info.push_back(s);
        ++ndim;
      }
    }
    utils::check(ndim > 0, "gw_line::head_extrapolation_t: no valid q-point found for extrapolation on any axis");
    c /= double(ndim);
  }

  /// h0(n) = sum_q c_q h(q, n) for h (nq, n)
  nda::array<ComplexType, 1> apply(nda::array<ComplexType, 2> const &h) const {
    utils::check(h.extent(0) == c.size(), "gw_line::head_extrapolation_t::apply: {} q vs {} weights", h.extent(0), c.size());
    nda::array<ComplexType, 1> o(h.extent(1));
    o() = ComplexType(0.0);
    for (long q = 0; q < c.size(); ++q)
      if (c(q) != 0.0) o += c(q) * h(q, nda::range::all);
    return o;
  }
  void log(int lvl = 2) const {
    app_log(lvl, "  gw_line head: q -> 0 extrapolation \"{}\" (weights on the per-q heads):", variant);
    for (auto const &s : info) app_log(lvl, "    {}", s);
  }
};

// ------------------------------------------------------------------------------------------------------------------
// Madelung corrections
// ------------------------------------------------------------------------------------------------------------------
/// exchange (hf_t::HF_K_correction with S = 1): F(k) -= madelung D(k)
inline void exchange_head_correction(nda::array<ComplexType, 3> &F, nda::array<ComplexType, 3> const &D, double madelung) {
  utils::check(F.shape() == D.shape(), "gw_line::exchange_head_correction: F / D shape mismatch");
  F -= madelung * D;
}

/// T(k)_ia = sum_P conj(X_Pi(k)) X_Pa(k) conj(chi(Gamma, P)) (gw_t::Sigma_div_correction, its gemm form), (nk, nb, nb), host
inline nda::array<ComplexType, 3> head_overlap_T(methods::thc_reader_t const &thc, long iq_gamma) {
  const long nk = thc.nkpts(), nb = thc.nbnd(), Np = thc.Np();
  auto chi = thc.basis_head();
  nda::array<ComplexType, 3> T(nk, nb, nb);
  nda::matrix<ComplexType> Y(Np, nb);
  for (long ik = 0; ik < nk; ++ik) {
    auto X = thc.X(0, 0, ik);   // (Np, nb)
    for (long P = 0; P < Np; ++P) {
      const ComplexType cP = std::conj(chi(iq_gamma, P));
      for (long i = 0; i < nb; ++i) Y(P, i) = cP * std::conj(X(P, i));
    }
    nda::matrix<ComplexType> Xm(X);
    nda::matrix<ComplexType> Tk = nda::transpose(Y) * Xm;
    T(ik, nda::range::all, nda::range::all) = Tk;
  }
  return T;
}

/**
 * dSigma^{>,<}(k, zeta) of the head (see the file header, item 4), ADDED to Sp, Sh rows l with global k = krows[l]
 * (Sp, Sh: (nrows, nz, nb, nb), the driver's replicated or k-distributed Sigma). Host; no communication.
 */
inline void head_sigma_correction(pole_data_t const &poles, nda::array<ComplexType, 3> const &T, nda::array<double, 1> const &nu,
                                  nda::array<ComplexType, 1> const &hp0, nda::array<ComplexType, 1> const &hh0, double madelung,
                                  nda::array<ComplexType, 1> const &zeta, std::vector<long> const &krows,
                                  nda::array<ComplexType, 4> &Sp, nda::array<ComplexType, 4> &Sh) {
  const long nz = zeta.size(), nb = poles.nb, r = nu.size();
  utils::check(hp0.size() == r and hh0.size() == r, "gw_line::head_sigma_correction: residues vs {} poles", r);
  utils::check(Sp.extent(0) == long(krows.size()) and Sh.extent(0) == long(krows.size()) and Sp.extent(1) == nz and
                   Sp.extent(2) == nb,
               "gw_line::head_sigma_correction: Sigma shape mismatch");
  auto all = nda::range::all;
  for (long l = 0; l < long(krows.size()); ++l) {
    const long ik = krows[l];
    nda::matrix<ComplexType> Tk(T(ik, all, all));
    for (auto s : {sector_t::particle, sector_t::hole}) {
      auto const &ps = poles(ik, s);
      const long M   = ps.size();
      if (M == 0) continue;
      const bool part = (s == sector_t::particle);
      auto const &hres = part ? hp0 : hh0;
      // K(z, m) = madelung sum_j h_j / (z - e_m -+ nu_j)
      nda::matrix<ComplexType> K(nz, M);
      for (long iz = 0; iz < nz; ++iz)
        for (long m = 0; m < M; ++m) {
          ComplexType a(0.0);
          const ComplexType z = zeta(iz) - ps.e(m);
          for (long j = 0; j < r; ++j) a += hres(j) / (part ? (z - nu(j)) : (z + nu(j)));
          K(iz, m) = madelung * a;
        }
      auto &S = part ? Sp : Sh;
      auto S2 = nda::reshape(S(l, all, all, all), std::array<long, 2>{nz, nb * nb});
      if (ps.is_factorized()) {
        // dSigma(z) = U diag(K(z, :)) U^dagger, U = T V
        nda::matrix<ComplexType> V(ps.v);
        nda::matrix<ComplexType> U = Tk * V;
        nda::matrix<ComplexType> UK(nb, M), D(nb, nb);
        for (long iz = 0; iz < nz; ++iz) {
          for (long i = 0; i < nb; ++i)
            for (long m = 0; m < M; ++m) UK(i, m) = U(i, m) * K(iz, m);
          nda::blas::gemm(ComplexType(1.0), UK, nda::dagger(U), ComplexType(0.0), D);
          S(l, iz, all, all) += D;
        }
      } else {
        // B(m, :) = T c_m T^dagger, dSigma = K B
        nda::matrix<ComplexType> B(M, nb * nb);
        for (long m = 0; m < M; ++m) {
          nda::matrix<ComplexType> cm(ps.coef(m, all, all));
          nda::matrix<ComplexType> Bm = Tk * cm * nda::dagger(Tk);
          B(m, all) = nda::reshape(Bm, std::array<long, 1>{nb * nb});
        }
        nda::blas::gemm(ComplexType(1.0), K, B, ComplexType(1.0), S2);
      }
    }
  }
}

} // namespace methods::gw_line

#endif
