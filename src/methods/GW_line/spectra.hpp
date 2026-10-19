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

#ifndef COQUI_METHODS_GW_LINE_SPECTRA_HPP
#define COQUI_METHODS_GW_LINE_SPECTRA_HPP

/**
 * Spectral functions from the final line state (notes section 7, Eq. A; python scripts/si222c_v5_qp.py and
 * si222c_v4_from_state.py):
 *
 *   per k: Sigma_c at the nodes (the last mixed Sigma, sampled about mu_sigma) -> one-sided sector fits -> poles shifted to
 *   the final centre mu (w -> w - (mu - mu_sigma)) -> Cayley moments -> upfolding (d, W);
 *   G(k, w + i eta) = [w + i eta - (H0 + F - mu) - Sigma_c(w + i eta)]^{-1},   A = (i / 2 pi)(G - G^dag),
 *   on a user grid w (relative to mu) for each eta. Stored: the diagonal A_ii(k, w) in the band basis and Tr A.
 * The QP edges (VBM/CBM) are those of the widest admissible gap of the Lehmann G of the same upfolded representation
 * (chemical_potential), relative to mu (absolute: + mu).
 *
 * D5 multi-scale spectra (python closure.spectra_multiscale): the resolution of one closure about mu is
 * (pi / K)[(w - mu)^2 + wp^2] / wp, so the SCF scale (Si: 0.11 Ha) cannot resolve satellites 10-30 eV below mu. With
 * wp_scales = {s_1 < s_2 ...} (only s_i > wp are used) the SAME Sigma_c fit is closed again at every s_i (order K, default the
 * closure's), and A = sum_i chi_i(w) A_i with the cos^2 partition of unity in ln|w - mu| whose node i sits at |w - mu| = s_i
 * (s_0 = wp: |w - mu| <= wp is the base closure unchanged; a convex combination, A >= 0). Empty wp_scales = bitwise the
 * single closure. Sum rules (always): M0 = int Tr A, M1 = int w Tr A on the grid (trapezoid) + the analytic Lorentzian tails of
 * the last scale's Lehmann G outside it; m0 = |M0 - nb| / nb, m1 = |M1 - Tr Hrel| / sum_i |Hrel_ii| per (eta, k).
 *
 * k is distributed round-robin over comm, the [n_eta, nk, nw, nb] result is summed with one all_reduce (one nonzero
 * contribution per element). Collective over comm.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "utilities/check.hpp"
#include "methods/GW_line/closure.hpp"

namespace methods::gw_line {

struct spectra_params_t {
  std::vector<double> eta = {0.004, 0.01};   ///< broadenings (Ha)
  double wmin = -0.45, wmax = 0.45;          ///< grid relative to mu (Ha)
  long nw = 601;
  std::vector<double> wp_scales;   ///< D5: extra Cayley scales (Ha) of the spectra closures (> the closure's wp); empty = off
  long K = 0;                      ///< D5: order of the extra closures (0 = the closure's K)
};

/// D5: chi (ns, nw), cos^2 partition of unity in ln|w| (w mu-relative), node of scale i at |w| = scales[i] (ascending)
inline nda::array<double, 2> multiscale_weights(nda::array<double, 1> const &om, std::vector<double> const &scales) {
  const long ns = long(scales.size()), nw = om.size();
  nda::array<double, 2> chi(ns, nw);
  chi() = 0.0;
  std::vector<double> L(ns);
  for (long i = 0; i < ns; ++i) L[i] = std::log(scales[i]);
  for (long j = 0; j < nw; ++j) {
    const double x = std::log(std::max(std::abs(om(j)), 1e-300));
    if (x <= L[0]) { chi(0, j) = 1.0; continue; }
    if (x >= L[ns - 1]) { chi(ns - 1, j) = 1.0; continue; }
    long i = long(std::upper_bound(L.begin(), L.end(), x) - L.begin()) - 1;
    const double t = (x - L[i]) / (L[i + 1] - L[i]), c = std::cos(0.5 * std::numbers::pi * t);
    chi(i, j)     = c * c;
    chi(i + 1, j) = 1.0 - c * c;
  }
  return chi;
}

/// D5: weight and principal-value first moment of sum_m a_m L_eta(w - e_m) outside [lo, hi] (python lorentz_tails)
inline std::pair<double, double> lorentz_tails(nda::array<double, 1> const &e, nda::array<double, 1> const &a, double eta, double lo,
                                               double hi) {
  double m0 = 0.0, m1 = 0.0;
  for (long m = 0; m < e.size(); ++m) {
    const double fin = (std::atan((hi - e(m)) / eta) - std::atan((lo - e(m)) / eta)) / std::numbers::pi;
    m0 += a(m) * (1.0 - fin);
    m1 += a(m) * (e(m) * (1.0 - fin) - eta / (2.0 * std::numbers::pi) *
                                            std::log(((hi - e(m)) * (hi - e(m)) + eta * eta) / ((lo - e(m)) * (lo - e(m)) + eta * eta)));
  }
  return {m0, m1};
}

struct spectra_out_t {
  nda::array<double, 1> omega;      ///< [nw] relative to mu
  nda::array<double, 1> eta;        ///< [n_eta]
  nda::array<double, 4> A_diag;     ///< [n_eta, nk, nw, nb]
  nda::array<double, 3> A_trace;    ///< [n_eta, nk, nw]
  double e_homo = 0.0, e_lumo = 0.0, N = 0.0;   ///< QP edges relative to mu, electron count at the chosen gap
  std::vector<long> npoles;
  std::vector<double> scales;       ///< D5: Cayley scales of the blend (scales[0] = the closure's wp)
  long K_scales = 0;                ///< D5: order of the extra closures
  nda::array<double, 2> m0_err, m1_err;   ///< D5: [n_eta, nk] sum rules of the blended (or single) A
};

/// D5: spectra of one k: the base closure (p) and, for scales.size() > 1, the closures at scales[1..] (order K_scales) blended
/// with chi (multiscale_weights); A_diag [ne, nw, nb], A_trace [ne, nw], sum rules m0 / m1 [ne]; ck = the base closure.
struct spectra_k_t {
  nda::array<double, 3> A_diag;
  nda::array<double, 2> A_trace;
  nda::array<double, 1> m0, m1;
  closure_k_t ck;
};

inline nda::array<double, 1> lehmann_trace_weights(closure_k_t const &c) {
  const long nb = c.v.extent(0);
  nda::array<double, 1> a(c.e.size());
  for (long m = 0; m < c.e.size(); ++m) {
    double x = 0.0;
    for (long i = 0; i < nb; ++i) x += std::norm(c.v(i, m));
    a(m) = x;
  }
  return a;
}

inline spectra_k_t spectra_k(nda::array<ComplexType, 2> const &H, sigma_poles_t const &spk, closure_params_t const &p,
                             std::vector<double> const &scales, long K_scales, nda::array<double, 2> const &chi,
                             nda::array<double, 1> const &omega, nda::array<double, 1> const &eta) {
  const long nb = H.extent(0), nw = omega.size(), ne = eta.size(), ns = long(scales.size());
  spectra_k_t r;
  r.A_diag  = nda::array<double, 3>(ne, nw, nb);
  r.A_trace = nda::array<double, 2>(ne, nw);
  r.m0      = nda::array<double, 1>(ne);
  r.m1      = nda::array<double, 1>(ne);
  r.A_diag()  = 0.0;
  r.A_trace() = 0.0;
  r.m0()      = 0.0;
  r.m1()      = 0.0;
  r.ck        = closure_k(H, spk, p);
  closure_k_t ck_last;
  for (long is = 0; is < ns; ++is) {
    closure_k_t cks;
    if (is > 0) {
      closure_params_t ps = p;
      ps.wp         = scales[is];
      ps.K          = K_scales;
      ps.phase_keep = 0.0;
      ps.phi_prev.clear();
      ps.force_rgram.clear();
      ps.force_r1.clear();
      ps.force_phi.clear();
      cks = closure_k(H, spk, ps);
    }
    closure_k_t const &cs = (is == 0) ? r.ck : cks;
    std::vector<long> idx;
    for (long iw = 0; iw < nw; ++iw)
      if (ns == 1 or chi(is, iw) > 0.0) idx.push_back(iw);
    if (!idx.empty()) {
      nda::array<double, 1> om_s(long(idx.size()));
      for (long j = 0; j < long(idx.size()); ++j) om_s(j) = omega(idx[j]);
      for (long ie = 0; ie < ne; ++ie) {
        auto A = numerics::line_dlr::spectral_function(H, cs.d, cs.W, om_s, eta(ie));
        for (long j = 0; j < long(idx.size()); ++j) {
          const long iw = idx[j];
          if (ns == 1) {   // single closure: bitwise the pre-D5 arithmetic
            double tr = 0.0;
            for (long i = 0; i < nb; ++i) {
              r.A_diag(ie, iw, i) = std::real(A(j, i, i));
              tr += std::real(A(j, i, i));
            }
            r.A_trace(ie, iw) = tr;
          } else {
            const double c = chi(is, iw);
            for (long i = 0; i < nb; ++i) {
              r.A_diag(ie, iw, i) += c * std::real(A(j, i, i));
              r.A_trace(ie, iw) += c * std::real(A(j, i, i));
            }
          }
        }
      }
    }
    if (is == ns - 1 and is > 0) ck_last = std::move(cks);
  }
  // sum rules of the (blended) trace: trapezoid on the grid + the analytic tails of the last scale outside it
  closure_k_t const &ct = (ns > 1) ? ck_last : r.ck;
  auto ta = lehmann_trace_weights(ct);
  if (nw >= 2) {
    double trH = 0.0, absH = 0.0;
    for (long i = 0; i < nb; ++i) {
      trH += std::real(H(i, i));
      absH += std::abs(std::real(H(i, i)));
    }
    for (long ie = 0; ie < ne; ++ie) {
      double M0 = 0.0, M1 = 0.0;
      for (long iw = 0; iw + 1 < nw; ++iw) {
        const double h = omega(iw + 1) - omega(iw), a0 = r.A_trace(ie, iw), a1 = r.A_trace(ie, iw + 1);
        M0 += 0.5 * h * (a0 + a1);
        M1 += 0.5 * h * (omega(iw) * a0 + omega(iw + 1) * a1);
      }
      auto [t0, t1] = lorentz_tails(ct.e, ta, eta(ie), omega(0), omega(nw - 1));
      r.m0(ie) = std::abs(M0 + t0 - double(nb)) / double(nb);
      r.m1(ie) = std::abs(M1 + t1 - trH) / std::max(absH, 1e-300);
    }
  }
  return r;
}

/**
 * Hrel (nk, nb, nb) = H0 + F - mu (final centre); Sig_p / Sig_h (nk, nz, nb, nb) at the nodes zeta about mu_sigma;
 * shift = mu - mu_sigma. Collective over comm. Sig_p / Sig_h may be k-distributed (nloc rows, see closure()).
 */
inline spectra_out_t line_spectra(boost::mpi3::communicator &comm, nda::array<ComplexType, 3> const &Hrel,
                                  nda::array<ComplexType, 4> const &Sig_p, nda::array<ComplexType, 4> const &Sig_h,
                                  nda::array<ComplexType, 1> const &zeta, double shift, line_basis_t const &bp,
                                  line_basis_t const &bh, closure_params_t const &p, double nelec,
                                  spectra_params_t const &sp, closure_thermal_t const *th = nullptr) {
  auto all      = nda::range::all;
  const long nk = Hrel.extent(0), nb = Hrel.extent(1), ne = long(sp.eta.size()), nw = sp.nw;
  const long np = comm.size(), rank = comm.rank();
  utils::check(nw >= 1 and ne >= 1, "gw_line::line_spectra: empty omega or eta grid");
  spectra_out_t out;
  out.omega = nda::array<double, 1>(nw);
  for (long iw = 0; iw < nw; ++iw) out.omega(iw) = (nw == 1) ? sp.wmin : sp.wmin + (sp.wmax - sp.wmin) * double(iw) / double(nw - 1);
  out.eta = nda::array<double, 1>(ne);
  for (long i = 0; i < ne; ++i) out.eta(i) = sp.eta[i];
  out.A_diag  = nda::array<double, 4>(ne, nk, nw, nb);
  out.A_trace = nda::array<double, 3>(ne, nk, nw);
  out.A_diag() = 0.0;
  out.A_trace() = 0.0;

  // D5: scales of the blend (scales[0] = the closure's wp; extra scales above it, ascending)
  out.scales = {p.wp};
  {
    std::vector<double> xs;
    for (double s : sp.wp_scales)
      if (s > p.wp * (1.0 + 1e-12)) xs.push_back(s);
    std::sort(xs.begin(), xs.end());
    for (double s : xs)
      if (s > out.scales.back() * (1.0 + 1e-12)) out.scales.push_back(s);
  }
  out.K_scales   = sp.K > 0 ? sp.K : p.K;
  auto chi       = multiscale_weights(out.omega, out.scales);
  out.m0_err     = nda::array<double, 2>(ne, nk);
  out.m1_err     = nda::array<double, 2>(ne, nk);
  out.m0_err()   = 0.0;
  out.m1_err()   = 0.0;

  std::vector<nda::array<double, 1>> e_loc(nk);
  std::vector<nda::array<ComplexType, 2>> v_loc(nk);
  nda::array<double, 1> npol(nk);
  npol() = 0.0;
  const bool sig_loc = (Sig_p.extent(0) != nk);   // k-distributed Sigma (S7e)
  for (long ik = rank; ik < nk; ik += np) {
    const long ks = sig_loc ? ik / np : ik;
    nda::array<ComplexType, 3> Sp(Sig_p(ks, all, all, all)), Sh(Sig_h(ks, all, all, all));
    auto spk = fit_sigma_closure(bp, bh, zeta, Sp, Sh, th);   // S8b: the thermal total fit in thermal mode
    spk.w -= shift;
    nda::array<ComplexType, 2> H(Hrel(ik, all, all));
    auto r    = spectra_k(H, spk, p, out.scales, out.K_scales, chi, out.omega, out.eta);
    npol(ik)  = double(r.ck.d.size());
    for (long ie = 0; ie < ne; ++ie) {
      for (long iw = 0; iw < nw; ++iw) {
        for (long i = 0; i < nb; ++i) out.A_diag(ie, ik, iw, i) = r.A_diag(ie, iw, i);
        out.A_trace(ie, ik, iw) = r.A_trace(ie, iw);
      }
      out.m0_err(ie, ik) = r.m0(ie);
      out.m1_err(ie, ik) = r.m1(ie);
    }
    auto &ck = r.ck;
    e_loc[ik] = std::move(ck.e);
    v_loc[ik] = std::move(ck.v);
  }
  detail::exact_allreduce(comm, out.A_diag.data(), out.A_diag.size());
  detail::exact_allreduce(comm, out.A_trace.data(), out.A_trace.size());
  detail::exact_allreduce(comm, npol.data(), npol.size());
  detail::exact_allreduce(comm, out.m0_err.data(), out.m0_err.size());
  detail::exact_allreduce(comm, out.m1_err.data(), out.m1_err.size());
  for (long ik = 0; ik < nk; ++ik) out.npoles.push_back(long(std::llround(npol(ik))));
  auto leh   = gather_lehmann(comm, nk, nb, e_loc, v_loc);
  auto cp    = numerics::line_dlr::chemical_potential(leh.e, leh.v, nelec, p.k_weight);
  out.e_homo = cp.e_homo;
  out.e_lumo = cp.e_lumo;
  out.N      = cp.N;
  return out;
}

} // namespace methods::gw_line

#endif
