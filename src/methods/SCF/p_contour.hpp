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

#ifndef COQUI_P_CONTOUR_HPP
#define COQUI_P_CONTOUR_HPP

/**
 * ===========================================================================
 * P ON THE TILTED CONTOUR  (increment TC-2 / spec M3, notes/tc_coqui_impl_spec.md)
 * ===========================================================================
 *
 * The ISDF space-time polarization kernel evaluated at COMPLEX times
 * t_j = s_j e^{-i theta} on the tilted contour of
 * numerics/tilted_contour/tilted_contour.hpp, from the CURRENT QP spectrum and
 * MO coefficients -- the same G provenance as the RW-2 spectral path's
 * A builder (rw2_report.md section 1.3).
 *
 * ---------------------------------------------------------------------------
 * 1. THE CONTRACTION IS THE tau CODE'S, WITH A COMPLEX ARGUMENT
 * ---------------------------------------------------------------------------
 * scr_coulomb_t::eval_Pi_rpa_Rspace (methods/scr_coulomb/rpa_pi.icc:102-176) is
 *
 *   Gp_PQ(k)  = X_Pa(k) G_ab(tau; k_ibz) conj(X_Qb(k))         [primary_to_aux]
 *   Gn_PQ(k)  = the same at G(beta - tau)
 *   Gp_PQ(R)  = sum_k f_Rk(R,k) Gp_PQ(k),   f_Rk(R,k) = e^{-i k.R}/N_k
 *   GG_PQ(R)  = sp_factor sum_s Gp^s_PQ(R) conj( Gn^s_PQ(R) )
 *   Pi_PQ(q)  = sum_R f_qR(q,R) GG_PQ(R),    sp_factor = -2 (ns=1) / -1 (ns=2)
 *
 * This module reproduces that chain with tau -> i t, i.e. every
 * e^{-xi_n tau} -> e^{-i xi_n t}, and with TWO conjugations replaced by
 * TRANSPOSES:
 *
 *   (a) THE TIME-REVERSAL FILL. rpa_pi.icc:110 writes the trev-paired FBZ point
 *       as conj(G_PQ(k')). The general relation is
 *
 *           G_PQ(-k) = G_QP(k)                                             (T)
 *
 *       -- from X_Pa(-k) = conj(X_Pa(k)) and O_ab(-k) = O_ba(k) -- and the two
 *       coincide only because G(k) is HERMITIAN in (P,Q) at imaginary time,
 *       where the pole weights are real. At complex t the weights are complex,
 *       G(k,t) is not Hermitian, and the conjugation is WRONG.
 *   (b) THE HADAMARD PARTNER. rpa_pi.icc:156 forms Gp_PQ(R) conj(Gn_PQ(R)); the
 *       object it means is Gn_QP(-R), i.e.
 *
 *           Gnt_PQ(R) = sum_k conj(f_Rk(R,k)) Gn_QP(k,t)                   (H)
 *
 *       which equals conj(Gn_PQ(R)) again only through the same Hermiticity.
 *
 * This is the RW-2 transpose finding (rw2_report.md section 3.7) in a second
 * place, and for the same reason. Both reductions are EXACT at t = -i tau, so
 * this module and the tau code are algebraically the same object there -- which
 * is gate TC-2-a(i), a direct numerical check that costs nothing extra.
 * [verified -- gate: TC-2-a(i), test_methods_tc_contour.cpp]
 *
 * ---------------------------------------------------------------------------
 * 2. THE POLE WEIGHTS, AND WHY THE OCCUPATION SITS IN THE EXPONENT
 * ---------------------------------------------------------------------------
 * With xi_n = eps_n - mu the analytic continuations of the tau code's two legs
 * are, per (s, k) and MO n,
 *
 *   fp_n(t) = -exp( -i xi_n t - softplus(-beta xi_n) )      [ = -e^{-xi tau}(1-f) ]
 *   fn_n(t) = -exp( +i xi_n t - softplus(+beta xi_n) )      [ = -e^{+xi tau} f    ]
 *
 * softplus(u) = log(1 + e^u), so exp(-softplus(-b xi)) = 1 - f(xi) and
 * exp(-softplus(+b xi)) = f(xi) EXACTLY -- these are the full finite-temperature
 * Fermi factors, not a T = 0 step. Keeping them inside the exponent is not
 * cosmetic: on the contour |e^{-i xi t}| = e^{-xi s sin th}, which for an
 * OCCUPIED state (xi < 0) grows like e^{|xi| s sin th} and overflows double
 * precision at production S; the occupation factor e^{-beta|xi|} cancels it
 * exactly, and the two must therefore be combined before exponentiating. The
 * combined exponent is bounded by -|xi| (beta - S sin th), so the construction
 * is safe iff
 *
 *      beta > S sin(theta)                                             (GUARD)
 *
 * which is checked and logged (`beta_guard` = beta / (S sin th)).
 *
 * The product then carries only e^{-i (xi_a - xi_i) t} = e^{-i Delta t} with
 * Delta > 0 for the physical particle-hole pairs -- the RESONANT half, which is
 * exactly what the contour transform represents. The thermally-occupied
 * "wrong-way" pairs are present too, with weight e^{-beta|Delta|}; at the beta
 * of any production run that is 1e-40 and below, and it is reported.
 *
 * ---------------------------------------------------------------------------
 * 3. THE ANTI-RESONANT HALF *AND* THE OVERALL SIGN
 * ---------------------------------------------------------------------------
 * What `sample_P_at_times` returns is the analytic continuation of the CODE'S
 * Pi(tau) -- so that the t = -i tau check of gate TC-2-a(i) is a bare identity:
 *
 *      Pi(t) = sum_p c_p e^{-i Delta_p t},
 *
 * summed over EVERY band pair, so Delta_p runs over both signs; the wrong-way
 * pairs carry c_p ~ e^{-beta |Delta_p|} and are precisely what makes Pi(tau)
 * particle-hole symmetric. The contour transform F represents the RESONANT half,
 *
 *      R(z) = sum_{Delta_p > 0} c_p / (z - Delta_p),
 *
 * while the object the imaginary-axis code computes is
 *
 *      Pi(i nu) = Int_0^beta dtau e^{i nu tau} Pi(tau)
 *               = sum_{Delta_p > 0} c_p [ 1/(Delta_p - i nu) + 1/(Delta_p + i nu) ]
 *
 * -- the second term is the wrong-way pairs, whose e^{-beta|Delta|} weight
 * cancels the e^{+beta|Delta|} of the tau integral exactly and relabels the pair
 * a <-> i. Since 1/(Delta -+ i nu) = -1/(+- i nu - Delta),
 *
 *      Pi(z) = -[ R(z) + R(-z) ],      R(-z) = [ R(-conj z) ]^dag.        (SIGN)
 *
 * TWO things are load-bearing there and neither is optional:
 *   * THE MINUS SIGN is the Jacobian of dtau = i dt between the two transform
 *     conventions (the campaign's P(z) = sum w/(z-D) versus the code's
 *     Pi(i nu) = sum w/(D - i nu)). It propagates straight into
 *     W^c = v[1 - v Pi]^{-1} v Pi v at TC-3.
 *     MEASURED: dropping it makes gate TC-2-a(ii) read a relative error of
 *     exactly 2.0000 at every target -- got = -exact.
 *   * THE DAGGER on the mirror row, for the same reason as (a)/(b) above.
 * `polarization_from_contour` below is that combination; on the imaginary axis
 * z = i nu the mirror target -conj(z) IS z, so Pi(i nu) = -[R + R^dag] and the
 * transform needs no extra row there. Off the imaginary axis the conjugate-
 * mirror rows of BINDING 2 in tilted_contour.hpp supply R(-conj z).
 *
 * ---------------------------------------------------------------------------
 * 4. THE GEOMETRY, FROM THE CURRENT QP SPECTRUM
 * ---------------------------------------------------------------------------
 * Ported from tc_validation/spectra.py (the campaign's own conventions, section
 * 1 of notes/tilted_contour_validation_results.md):
 *   Delta support   the q-INTEGRATED particle-hole support,
 *                   Dmin = min(empty) - max(occupied),
 *                   Dmax = max(empty) - min(occupied);
 *   W_band          the frontier valence cluster: pool the occupied eigenvalues,
 *                   walk down from mu until a gap > 2 eV;
 *   target window   W_target = zeta_max - Dmin with zeta_max the QP half-window;
 *   delta           eq 8 with the MEASURED mesh constant,
 *                   delta_mesh = 1.2 W_band / N_k   <-- FLAGGED: the spec writes
 *                   0.7; results section 5.4 measured the 1 %-crossing at
 *                   0.81-3.99 x the 0.7 prediction, median 1.69, i.e. a fitted
 *                   constant of ~1.2. And delta = eta_targ, NOT 3.5 eta_targ
 *                   (results section 7.2 item 5, the adopted sub-meV tier).
 *   N_k             cbrt(nkpts) -- exact for the cubic meshes the campaign and
 *                   the fixtures use; FLAGGED for anisotropic meshes.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/tilted_contour/tilted_contour.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/wc_line.hpp"
#include "methods/SCF/qp_modea.hpp"
#include "methods/mb_state/mb_state.hpp"

namespace methods {
namespace p_contour {

  namespace tc = tilted_contour;

  inline constexpr double ha_to_eV = 27.211386245988;
  /** the campaign's frontier-cluster separator, 2 eV (tc_validation/spectra.py GAP_CUT) */
  inline constexpr double gap_cut_eV = 2.0;
  /** the MEASURED k-mesh constant (results section 5.4); the spec writes 0.7. FLAGGED. */
  inline constexpr double delta_mesh_const = 1.2;

  // =========================================================================
  //  knobs
  // =========================================================================
  struct opts_t {
    double eps = 1e-6;             ///< qp_tc_eps: rank / truncation tolerance
    double delta = 0.0;            ///< qp_tc_delta, a.u.; 0 selects the recipe
    double rho = 0.65;             ///< qp_tc_rho
    std::string profile = "flat";  ///< qp_tc_profile: "flat" | "growing"
    bool trunc = false;            ///< qp_tc_trunc: band truncation along the contour
    double zeta_max = 10.0 / ha_to_eV;  ///< QP half-window (a.u.); the campaign's JUDGE_SPAN
    long   nx = 2500;              ///< adapted Delta-grid resolution
    int    level = 2;              ///< logging level
  };

  // =========================================================================
  //  spectrum geometry
  // =========================================================================
  struct geom_t {
    double dmin = 0.0, dmax = 0.0;        ///< a.u.
    double W_band = 0.0, W_target = 0.0;
    double delta_mesh = 0.0, delta = 0.0;
    double mu = 0.0, zeta_max = 0.0;
    long   nk_lin = 0, n_occ = 0, n_emp = 0;
    long   n_valence_clusters = 0;
    bool   window_floored = false;   ///< Dmin > zeta_max: W_target took its delta floor
  };

  /**
   * Delta support, W_band and the eq-8 delta from the CURRENT QP spectrum.
   * `E` is (ns, nk_ibz, nbnd) ABSOLUTE quasiparticle energies in a.u.
   * [port of tc_validation/spectra.py::Fixture._derive]
   */
  template<typename E_t>
  geom_t analyze_spectrum(E_t const &E, double mu, long nk_lin, opts_t const &o) {
    const long ns = E.shape()[0], nk = E.shape()[1], nb = E.shape()[2];
    geom_t g;
    g.mu = mu;
    g.zeta_max = o.zeta_max;
    // nk_lin is the LINEAR mesh count of delta_mesh = 1.2 W_band / N_k. The caller
    // supplies min(kp_grid) -- the coarsest direction, which is where transition
    // energies move most between neighbouring k -- and that reduces to N_k on the
    // cubic meshes the campaign measured. FLAGGED for strongly anisotropic meshes:
    // the campaign never measured one (results section 5.4 is a cubic-mesh law).
    g.nk_lin = std::max(1L, nk_lin);

    std::vector<double> occ, emp;
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k)
        for (long n = 0; n < nb; ++n) {
          const double e = std::real(ComplexType(E(s, k, n)));
          (e < mu ? occ : emp).push_back(e);
        }
    utils::check(not occ.empty() and not emp.empty(),
                 "p_contour: the QP spectrum has {} occupied and {} empty states at "
                 "mu = {:.6g} -- the particle-hole support is empty.",
                 occ.size(), emp.size(), mu);
    g.n_occ = long(occ.size());
    g.n_emp = long(emp.size());
    std::sort(occ.begin(), occ.end());
    std::sort(emp.begin(), emp.end());
    g.dmin = emp.front() - occ.back();
    g.dmax = emp.back()  - occ.front();

    // W_band: walk down from mu through the pooled occupied eigenvalues until a
    // gap wider than 2 eV. That cluster is the DISPERSING frontier manifold that
    // sets the k-mesh floor; a deep flat semicore band is irrelevant however deep.
    const double cut = gap_cut_eV / ha_to_eV;
    long lo = 0;
    g.n_valence_clusters = 1;
    for (long i = long(occ.size()) - 1; i > 0; --i)
      if (occ[std::size_t(i)] - occ[std::size_t(i - 1)] > cut) { lo = i; break; }
    for (long i = 1; i < long(occ.size()); ++i)
      if (occ[std::size_t(i)] - occ[std::size_t(i - 1)] > cut) ++g.n_valence_clusters;
    g.W_band = mu - occ[std::size_t(lo)];

    g.delta_mesh = delta_mesh_const * g.W_band / double(g.nk_lin);
    g.delta = (o.delta > 0.0) ? o.delta : g.delta_mesh;   // eq 8 at eta_targ, no 3.5x
    // ⚠ W_target's FLOOR. The campaign's convention is W_target = zeta_max - Dmin, with
    // the residue targets running over w in [Dmin, Dmin + W_target] -- which silently
    // assumes Dmin < zeta_max, i.e. that the particle-hole edge lies INSIDE the QP
    // window. On a wide-gap insulator it does not: LiH's converged mode-A gap is
    // 12.9 eV against zeta_max = 10 eV, W_target hit its 1e-6 guard, tan(theta) =
    // rho*delta/W_target exploded and S sin(theta) reached 5.1e+07 -- the beta guard
    // below fired, correctly, at the second outer iteration.
    // The regime is actually the EASY one: when Dmin > zeta_max every residue target
    // sits BELOW the particle-hole edge, where a(D) = (D - w) sin th + d cos th is large
    // for every transition, so eq 3 imposes no real constraint. What it must not do is
    // divide by zero. Flooring W_target at delta bounds tan(theta) <= rho and leaves S
    // finite, without touching the Dmin < zeta_max case the campaign measured.
    // [FLAGGED: the campaign never ran a fixture with Dmin > zeta_max.]
    g.W_target = std::max(g.zeta_max - g.dmin, g.delta);
    g.window_floored = (g.zeta_max - g.dmin) < g.delta;
    utils::check(g.delta > 0.0,
                 "p_contour: the delta recipe produced {} (W_band = {}, N_k = {}); set "
                 "qp_tc_delta explicitly.", g.delta, g.W_band, g.nk_lin);
    return g;
  }

  /**
   * The growing-delta profile (spec eq 8 behind the qp_tc_profile knob).
   *   delta(zeta) = max(delta_mesh, 0.05 |zeta|)
   *   tan th      = rho * min_{zeta > dmin} delta(zeta)/(zeta - dmin)
   *   a(x)        = min_zeta [ (dmin + x - zeta) sin th + delta(zeta) cos th ]
   * [port of tc_validation/spectra.py::{delta_profile, growing_tilt, a_growing},
   *  with delta = eta_targ rather than the campaign's 3.5 eta_targ -- results
   *  section 7.2 item 5 adopts eta_targ. FLAGGED: the campaign measured the
   *  growing profile's 1.0-2.8x gain at 3.5 eta_targ.]
   */
  struct growing_profile_t {
    std::vector<double> zeta, dz;
    double dmin = 0.0, tan_theta = 0.0, theta = 0.0;

    growing_profile_t(geom_t const &g, double rho, long n = 400) : dmin(g.dmin) {
      zeta.resize(std::size_t(n));
      dz.resize(std::size_t(n));
      for (long i = 0; i < n; ++i) {
        zeta[std::size_t(i)] = g.zeta_max * double(i) / double(n - 1);
        dz[std::size_t(i)] = std::max(g.delta_mesh, 0.05 * std::abs(zeta[std::size_t(i)]));
      }
      double m = 1e300;
      bool any = false;
      for (long i = 0; i < n; ++i)
        if (zeta[std::size_t(i)] > g.dmin) {
          any = true;
          m = std::min(m, dz[std::size_t(i)] / (zeta[std::size_t(i)] - g.dmin));
        }
      if (not any) {
        double dmaxv = 0.0;
        for (auto v : dz) dmaxv = std::max(dmaxv, v);
        m = dmaxv / std::max(g.W_target, 1e-6);
      }
      tan_theta = rho * m;
      theta = std::atan(tan_theta);
    }

    double operator()(double x) const {
      const double st = std::sin(theta), ct = std::cos(theta);
      double m = 1e300;
      for (std::size_t i = 0; i < zeta.size(); ++i)
        m = std::min(m, (dmin + x - zeta[i]) * st + dz[i] * ct);
      return m;
    }
  };

  // =========================================================================
  //  the context
  // =========================================================================
  struct ctx_t {
    bool on = false;
    geom_t geom;
    tc::contour_t c;
    nda::array<ComplexType, 1> t_node;   ///< (rank) t_j = s_j e^{-i theta}
    // diagnostics
    double t_wall = 0.0;
    double beta_guard = 0.0;             ///< beta / (S sin theta); must exceed 1
    double thermal_worst = 0.0;          ///< the largest wrong-way pair weight e^{-beta|D|}
    long   nband_p_min = 0, nband_p_max = 0;
    long   nband_n_min = 0, nband_n_max = 0;
    double trunc_ratio = 1.0;            ///< full band work / truncated band work
    double herm_rel = 0.0;               ///< max|Pi_PQ - conj(Pi_QP)| / max|Pi| at the
                                         ///< first node (REPORTED: it is NOT Hermitian
                                         ///< at complex t, only on the imaginary axis)
  };

  /**
   * Eq (SIGN) of the header: from the two transformed halves to the polarization
   * in the CODE'S convention,
   *
   *      Pi(z) = -[ R(z) + R(-conj z)^dag ].
   *
   * On the imaginary axis pass the SAME matrix twice (-conj(i nu) = i nu).
   */
  inline void polarization_from_contour(nda::MemoryArrayOfRank<2> auto const &R_res,
                                        nda::MemoryArrayOfRank<2> auto const &R_mir,
                                        nda::MemoryArrayOfRank<2> auto &&out) {
    const long n = R_res.shape()[0];
    for (long P = 0; P < n; ++P)
      for (long Q = 0; Q < n; ++Q)
        out(P, Q) = -(R_res(P, Q) + std::conj(R_mir(Q, P)));
  }

  namespace detail {

    inline double softplus(double u) {
      return (u > 30.0) ? u : std::log1p(std::exp(u));
    }

    /**
     * The two leg weights at complex time t, occupation folded into the exponent
     * (section 2 of the header). Returns log|w| in `lg` so the caller can apply a
     * RELATIVE truncation without ever forming the overflowing intermediate.
     */
    inline void leg_weights(double xi, double beta, ComplexType t,
                            ComplexType &fp, ComplexType &fn,
                            double &lgp, double &lgn) {
      const ComplexType ep = ComplexType(0.0, -1.0) * xi * t - softplus(-beta * xi);
      const ComplexType en = ComplexType(0.0,  1.0) * xi * t - softplus( beta * xi);
      lgp = ep.real();
      lgn = en.real();
      fp = -std::exp(ep);
      fn = -std::exp(en);
    }

  } // namespace detail

  /**
   * Sample Pi_PQ(q, t) at an arbitrary list of COMPLEX times.
   *
   * @param sPi    [OUT] shared (nq_ibz, nt, Np, Np); OVERWRITTEN.
   * @param tlist  (nt) complex times. t = -i tau reproduces the tau code exactly.
   * @param sMO    (ns, nk_ibz, nbnd, nbnd) MO coefficients, column n = MO n.
   * @param sE     (ns, nk_ibz, nbnd) ABSOLUTE QP energies, a.u.
   * @param trunc_rel if > 0, drop MOs whose leg weight is below this RELATIVE
   *                  threshold at the node (the band truncation along the contour):
   *                  log|w_n| < max_m log|w_m| + log(trunc_rel). <= 0 keeps every MO
   *                  whose weight has not underflowed to zero.
   *
   * The nt axis is distributed over `thc.mpi()->comm` and reduced at the end;
   * every rank does the full (k, P, Q) work of its own nodes, which is the same
   * per-pool footprint the tau code's ntpools split has.
   */
  template<typename thc_t>
  void sample_P_at_times(sArray_t<Array_view_4D_t> &sPi,
                         nda::array<ComplexType, 1> const &tlist,
                         thc_t &thc,
                         const sArray_t<Array_view_4D_t> &sMO_skia,
                         const sArray_t<Array_view_3D_t> &sE_ska,
                         double mu, double beta, double trunc_rel,
                         ctx_t *diag = nullptr) {
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();

    const long ns      = sE_ska.shape()[0];
    const long nk_ibz  = sE_ska.shape()[1];
    const long nbnd    = sE_ska.shape()[2];
    const long NP      = thc.Np();
    const long nkpts   = MF->nkpts();
    const long nq_ibz  = MF->nqpts_ibz();
    const long nk_notr = nkpts - MF->nkpts_trev_pairs();
    const long nt      = tlist.size();
    const double sp_factor = (ns == 2) ? -1.0 : -2.0;

    utils::check(sPi.shape()[0] == nq_ibz and sPi.shape()[1] == nt
                 and sPi.shape()[2] == NP and sPi.shape()[3] == NP,
                 "p_contour::sample_P_at_times: Pi has shape ({}, {}, {}, {}), expected "
                 "({}, {}, {}, {}).", sPi.shape()[0], sPi.shape()[1], sPi.shape()[2],
                 sPi.shape()[3], nq_ibz, nt, NP, NP);
    utils::check(thc.X(0, 0, 0).shape(1) == nbnd,
                 "p_contour: the THC collocation carries {} orbitals but the QP block is "
                 "{} wide.", thc.X(0, 0, 0).shape(1), nbnd);

    auto kp_map  = MF->kp_to_ibz();
    auto kp_trev = MF->kp_trev();
    auto kp_pair = MF->kp_trev_pair();

    // the two Fourier kernels, exactly as rpa_pi.icc builds them
    math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(*mpi, {nkpts, nkpts});
    utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(),
                               MF->kp_grid(), sf_Rk);
    math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_qR(*mpi, {nq_ibz, nkpts});
    utils::R_to_k_coefficients(mpi->comm, nda::range(nkpts), MF->Qpts_ibz(), MF->lattv(),
                               MF->kp_grid(), sf_qR);
    auto f_Rk = sf_Rk.local();
    auto f_qR = sf_qR.local();
    nda::matrix<ComplexType> f_Rk_c(nkpts, nkpts);            // conj(f_Rk) = f_{-R,k}
    for (long i = 0; i < nkpts; ++i)
      for (long j = 0; j < nkpts; ++j) f_Rk_c(i, j) = std::conj(f_Rk(i, j));
    nda::matrix<ComplexType> f_qR_m(nq_ibz, nkpts);
    for (long i = 0; i < nq_ibz; ++i)
      for (long j = 0; j < nkpts; ++j) f_qR_m(i, j) = f_qR(i, j);
    nda::matrix<ComplexType> f_Rk_m(nkpts, nkpts);
    for (long i = 0; i < nkpts; ++i)
      for (long j = 0; j < nkpts; ++j) f_Rk_m(i, j) = f_Rk(i, j);

    sPi.set_zero();
    sPi.win().fence();
    mpi->comm.barrier();

    // work space (per rank)
    nda::array<ComplexType, 4> Gp_skab(ns, nk_ibz, nbnd, nbnd);
    nda::array<ComplexType, 4> Gn_skab(ns, nk_ibz, nbnd, nbnd);
    nda::array<ComplexType, 4> buf_skPQ(ns, nk_notr, NP, NP);
    nda::array<ComplexType, 4> Gp_sRPQ(ns, nkpts, NP, NP);
    nda::array<ComplexType, 4> Gn_sRPQ(ns, nkpts, NP, NP);
    nda::matrix<ComplexType>   tmp_RPQ(nkpts, NP * NP);
    nda::array<ComplexType, 3> GG_RPQ(nkpts, NP, NP);
    nda::matrix<ComplexType>   Piq(nq_ibz, NP * NP);
    nda::matrix<ComplexType>   Cw(nbnd, nbnd), Ck(nbnd, nbnd);

    long np_min = nbnd, np_max = 0, nn_min = nbnd, nn_max = 0, kept_sum = 0, full_sum = 0;
    double therm = 0.0, herm_num = 0.0, herm_den = 0.0;

    auto [j0, j1] = itertools::chunk_range(0, nt, mpi->comm.size(), mpi->comm.rank());

    for (long jt = j0; jt < j1; ++jt) {
      const ComplexType t = tlist(jt);

      // ---- (1) the two band-basis legs at the IBZ ------------------------------
      Gp_skab() = ComplexType(0.0);
      Gn_skab() = ComplexType(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk_ibz; ++k) {
          auto MO = sMO_skia.local()(s, k, all, all);
          auto E  = sE_ska.local()(s, k, all);
          for (int leg = 0; leg < 2; ++leg) {
            // collect the surviving MOs and their weights
            std::vector<long> keep;
            std::vector<ComplexType> wgt;
            double lgmax = -1e300;
            std::vector<double> lg(static_cast<std::size_t>(nbnd));
            std::vector<ComplexType> ww(static_cast<std::size_t>(nbnd));
            for (long n = 0; n < nbnd; ++n) {
              const double xi = std::real(ComplexType(E(n))) - mu;
              ComplexType fp, fn;
              double lgp, lgn;
              detail::leg_weights(xi, beta, t, fp, fn, lgp, lgn);
              lg[std::size_t(n)] = (leg == 0) ? lgp : lgn;
              ww[std::size_t(n)] = (leg == 0) ? fp : fn;
              lgmax = std::max(lgmax, lg[std::size_t(n)]);
              // the "wrong-way" thermal weight of this MO on the OTHER leg
              if (leg == 0 and xi < 0.0) therm = std::max(therm, std::exp(lgp));
              if (leg == 1 and xi > 0.0) therm = std::max(therm, std::exp(lgn));
            }
            const double cut = (trunc_rel > 0.0) ? lgmax + std::log(trunc_rel) : -1e300;
            for (long n = 0; n < nbnd; ++n)
              if (lg[std::size_t(n)] > cut and ww[std::size_t(n)] != ComplexType(0.0)) {
                keep.push_back(n);
                wgt.push_back(ww[std::size_t(n)]);
              }
            const long nk_keep = long(keep.size());
            kept_sum += nk_keep;
            full_sum += nbnd;
            if (leg == 0) { np_min = std::min(np_min, nk_keep); np_max = std::max(np_max, nk_keep); }
            else          { nn_min = std::min(nn_min, nk_keep); nn_max = std::max(nn_max, nk_keep); }
            if (nk_keep == 0) continue;
            // G_ab = sum_{n in keep} C_an w_n conj(C_bn) = (C diag(w)) C^dag
            auto Cwv = Cw(all, nda::range(nk_keep));
            auto Ckv = Ck(all, nda::range(nk_keep));
            for (long c = 0; c < nk_keep; ++c)
              for (long a = 0; a < nbnd; ++a) {
                const ComplexType ca = MO(a, keep[std::size_t(c)]);
                Ckv(a, c) = ca;
                Cwv(a, c) = ca * wgt[std::size_t(c)];
              }
            auto dst = (leg == 0) ? Gp_skab(s, k, all, all) : Gn_skab(s, k, all, all);
            nda::blas::gemm(ComplexType(1.0), Cwv, nda::dagger(Ckv), ComplexType(0.0), dst);
          }
        }

      // ---- (2) to the auxiliary basis, then to the full BZ ---------------------
      // The trev-paired FBZ points take the TRANSPOSE, not the conjugate -- eq (T).
      for (int leg = 0; leg < 2; ++leg) {
        auto const &src = (leg == 0) ? Gp_skab : Gn_skab;
        auto &dst = (leg == 0) ? Gp_sRPQ : Gn_sRPQ;
        solvers::thc_solver_comm::primary_to_aux(0, 0, src, buf_skPQ, thc, kp_map, kp_trev);
        for (long s = 0; s < ns; ++s)
          for (long ik = 0; ik < nkpts; ++ik) {
            if (kp_trev(ik)) {
              const long kp = kp_pair(ik);
              for (long P = 0; P < NP; ++P)
                for (long Q = 0; Q < NP; ++Q) dst(s, ik, P, Q) = buf_skPQ(s, kp, Q, P);
            } else {
              dst(s, ik, all, all) = buf_skPQ(s, ik, all, all);
            }
          }
      }

      // ---- (3) k -> R.  Gp with f_Rk; the Gn PARTNER is Gn_QP at -R -- eq (H). --
      if (nkpts != 1) {
        for (long s = 0; s < ns; ++s) {
          auto Gp2 = nda::reshape(Gp_sRPQ(s, nda::ellipsis{}),
                                  std::array<long, 2>{nkpts, NP * NP});
          nda::blas::gemm(f_Rk_m, Gp2, tmp_RPQ);
          Gp2 = tmp_RPQ;
        }
      }
      for (long s = 0; s < ns; ++s) {         // transpose (P,Q) BEFORE the -R transform
        for (long ik = 0; ik < nkpts; ++ik)
          for (long P = 0; P < NP; ++P)
            for (long Q = P + 1; Q < NP; ++Q)
              std::swap(Gn_sRPQ(s, ik, P, Q), Gn_sRPQ(s, ik, Q, P));
        if (nkpts != 1) {
          auto Gn2 = nda::reshape(Gn_sRPQ(s, nda::ellipsis{}),
                                  std::array<long, 2>{nkpts, NP * NP});
          nda::blas::gemm(f_Rk_c, Gn2, tmp_RPQ);
          Gn2 = tmp_RPQ;
        }
      }

      // ---- (4) the Hadamard product -------------------------------------------
      GG_RPQ() = ComplexType(0.0);
      for (long s = 0; s < ns; ++s)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long P = 0; P < NP; ++P)
            for (long Q = 0; Q < NP; ++Q)
              GG_RPQ(ik, P, Q) += Gp_sRPQ(s, ik, P, Q) * Gn_sRPQ(s, ik, P, Q);
      GG_RPQ *= sp_factor;

      // ---- (5) R -> q ----------------------------------------------------------
      if (nkpts != 1) {
        auto GG2 = nda::reshape(GG_RPQ, std::array<long, 2>{nkpts, NP * NP});
        nda::blas::gemm(f_qR_m, GG2, Piq);
        for (long iq = 0; iq < nq_ibz; ++iq)
          for (long P = 0; P < NP; ++P)
            for (long Q = 0; Q < NP; ++Q)
              sPi.local()(iq, jt, P, Q) = Piq(iq, P * NP + Q);
      } else {
        for (long P = 0; P < NP; ++P)
          for (long Q = 0; Q < NP; ++Q)
            sPi.local()(0, jt, P, Q) = GG_RPQ(0, P, Q);
      }
      if (jt == j0) {
        for (long iq = 0; iq < nq_ibz; ++iq)
          for (long P = 0; P < NP; ++P)
            for (long Q = 0; Q < NP; ++Q) {
              herm_num = std::max(herm_num, std::abs(sPi.local()(iq, jt, P, Q)
                                          - std::conj(sPi.local()(iq, jt, Q, P))));
              herm_den = std::max(herm_den, std::abs(sPi.local()(iq, jt, P, Q)));
            }
      }
    }

    sPi.win().fence();
    sPi.all_reduce();
    sPi.win().fence();

    if (diag != nullptr) {
      diag->nband_p_min = mpi->comm.all_reduce_value(np_min, boost::mpi3::min<>{});
      diag->nband_p_max = mpi->comm.all_reduce_value(np_max, boost::mpi3::max<>{});
      diag->nband_n_min = mpi->comm.all_reduce_value(nn_min, boost::mpi3::min<>{});
      diag->nband_n_max = mpi->comm.all_reduce_value(nn_max, boost::mpi3::max<>{});
      const long ks = mpi->comm.all_reduce_value(kept_sum, std::plus<>{});
      const long fs = mpi->comm.all_reduce_value(full_sum, std::plus<>{});
      diag->trunc_ratio = (ks > 0) ? double(fs) / double(ks) : 1.0;
      diag->thermal_worst = mpi->comm.all_reduce_value(therm, boost::mpi3::max<>{});
      const double hn = mpi->comm.all_reduce_value(herm_num, boost::mpi3::max<>{});
      const double hd = mpi->comm.all_reduce_value(herm_den, boost::mpi3::max<>{});
      diag->herm_rel = (hd > 0.0) ? hn / hd : 0.0;
    }
  }

  /**
   * Build the contour from the current QP spectrum and sample P on it.
   * `sPi` is allocated by the caller at (nq_ibz, rank, Np, Np) AFTER the contour
   * is built, so this is split in two: `build_contour_for_spectrum` then
   * `sample_P_at_times`.
   */
  template<typename E_t>
  ctx_t build_contour_for_spectrum(E_t const &E, double mu, long nk_lin, double beta,
                                   opts_t const &o) {
    ctx_t ctx;
    ctx.geom = analyze_spectrum(E, mu, nk_lin, o);

    tc::contour_params_t p;
    p.dmin = ctx.geom.dmin;
    p.dmax = ctx.geom.dmax;
    p.W    = ctx.geom.W_target;
    p.delta = ctx.geom.delta;
    p.rho  = o.rho;
    p.eps  = o.eps;
    p.eps_tr = -1.0;                    // BINDING 3: eps_tr = eps^2
    p.nx   = o.nx;

    if (o.profile == "growing") {
      growing_profile_t gp(ctx.geom, o.rho);
      // The growing profile fixes theta itself (the binding tilt over the whole
      // target window); feed it back through rho so derive_geometry reproduces the
      // same tilt, and override gamma with the profile's OWN worst case a(0) --
      // eq 4's flat closed form is not it, and gamma sets the contour length.
      // [the campaign does exactly this: tc_validation/tests/common.py::Geom]
      p.rho = std::min(0.999, gp.tan_theta * p.W / p.delta);
      auto a_fun = [gp](double x) { return gp(x); };
      p.gamma_override = a_fun(0.0);
      utils::check(p.gamma_override > 0.0,
                   "p_contour: the growing delta profile gives a(0) = {} <= 0 at "
                   "rho = {}; the tilt violates eq 3.", p.gamma_override, o.rho);
      ctx.c = tc::build_contour(p, a_fun);
    } else {
      utils::check(o.profile == "flat",
                   "p_contour: qp_tc_profile = \"{}\" is not one of flat|growing.",
                   o.profile);
      ctx.c = tc::build_contour(p);
    }

    ctx.t_node = nda::array<ComplexType, 1>(ctx.c.rank);
    const ComplexType rot = std::exp(ComplexType(0.0, -ctx.c.g.theta));
    for (long j = 0; j < ctx.c.rank; ++j) ctx.t_node(j) = ctx.c.s(j) * rot;

    // THE GUARD of section 2: beta must dominate the contour's growth rate.
    const double Ssin = ctx.c.g.S * std::sin(ctx.c.g.theta);
    ctx.beta_guard = (Ssin > 0.0) ? beta / Ssin : 1e300;
    utils::check(ctx.beta_guard > 1.0,
                 "p_contour: beta = {:.4g} does not exceed S sin(theta) = {:.4g} "
                 "(ratio {:.3f}). The occupation factor no longer cancels the contour's "
                 "growth on the deepest states and the complex-time legs overflow. "
                 "THE GEOMETRY THAT PRODUCED IT: Delta in [{:.4g}, {:.4g}] a.u., "
                 "W_band = {:.4g}, N_k = {}, delta = {:.4g}, zeta_max = {:.4g}, "
                 "W_target = {:.4g}{}, theta = {:.4g}, gamma = {:.4g}, S = {:.4g}. "
                 "Raise beta, raise qp_tc_delta (which shortens S) or lower qp_tc_rho "
                 "(which lowers the tilt).",
                 beta, Ssin, ctx.beta_guard, ctx.geom.dmin, ctx.geom.dmax,
                 ctx.geom.W_band, ctx.geom.nk_lin, ctx.geom.delta, ctx.geom.zeta_max,
                 ctx.geom.W_target,
                 ctx.geom.window_floored ? " (FLOORED: Dmin > zeta_max)" : "",
                 ctx.c.g.theta, ctx.c.g.gamma, ctx.c.g.S);
    ctx.on = true;
    return ctx;
  }

  /**
   * ===========================================================================
   * THE eq-1 RESIDUE SOURCE, FED BY THE CONTOUR  (TC-3, batched at TC-4)
   * ===========================================================================
   *
   * Answers, for a LIST of internal states J at their targets z, the sandwiches
   * <aJ|W^c(q_J, z)|Jb>. The chain is
   *
   *   1. the transform rows for every z and for every conjugate mirror -conj(z), from
   *      the REUSABLE factorization (tilted_contour::transform_factor_t) -- ONE gemm
   *      for the whole chunk, never a factorization;
   *   2. R(z) = sum_j F(z,j) Pi_contour(q_J, j, :, :), which for all targets SHARING a
   *      transfer q is the single gemm (n_z x rank) x (rank x Np^2);
   *   3. eq (SIGN):  Pi(z) = -[ R(z) + R(-conj z)^dag ];
   *   4. CoQuI's own Dyson chain, W^c = ([I - Z.Pi]^{-1} - I).Z with Z = thc.Z(q_J);
   *   5. the trev-q rule and the band sandwich, EXACTLY as stage 2 does it:
   *          wconj : T = W.B ,  Ms = conj( Bc^T . T )
   *          else  : T = W.Bc,  Ms = B^T . T
   *      times the 1/nkpts prefactor stage 2 folds in.
   *
   * STEPS 1 AND 2 ARE THE BATCHED ONES and they are the reason the batching exists: at
   * rank r and grid nD both were BLAS-2 passes per target, r*nD and r*Np^2, which at the
   * fixture sizes outweigh the Np^3 Dyson solve they feed. Steps 4-5 stay per target --
   * different z means a different dielectric matrix, and there is nothing to share.
   * Grouping by q is a pure REORDERING of independent targets: each one still gets its
   * own row of F and its own solve, and the results are written back in the caller's
   * order. [gate: the [TC-4 batch] leg of tc3b1_identity_lih222 (batched vs per-target,
   *  0.000e+00) and tc_contour_batch_scaling (the same two primitives at Np = 364)]
   *
   * The band factors B and the (IBZ q, trev) pair come from `modea_ctx::cd_band_store`,
   * which supplies them from its store or recomputes them (TC-4 F5); either way the
   * evaluator re-derives no symmetry bookkeeping.
   */
  // ===========================================================================
  //  THE Z TILES -- and THE COLLECTIVE-FREE INVARIANT
  // ===========================================================================
  /**
   * ⚠⚠ INVARIANT, LOAD-BEARING: `detail::contour_residue_batch` AND EVERYTHING IT
   * CALLS MUST BE FREE OF MPI COLLECTIVES.
   *
   * The evaluator runs inside the per-rank work loop of `modea_vxc_cd`, where every
   * rank carries DIFFERENT (s, k) blocks, a different number of residue targets, and a
   * different set of q-transfers. Any collective reached from there has a call sequence
   * that differs across ranks, and MPI deadlocks -- as a spin, not a crash.
   *
   * MEASURED: the m3d SVO run (60 ranks) hung for 19 h at 100 % CPU on every rank, in
   *   PMPI_Gather <- boost::mpi3 gather <- math::nda::gather_sub_matrix
   *   <- thc_reader_t::Z(int, bool) <- p_contour::detail::contour_residue_batch.
   *
   * `thc_reader_t::Z(iq)` (thc_reader_t.hpp:720) is that collective: in the `incore`
   * path it loops over EVERY rank of the THC array's communicator, broadcasting the
   * requested iq from each in turn and gathering that rank's tile. It supports
   * different iq per rank, but ONLY if every rank calls it the same number of times, in
   * lockstep. The repository already knew this -- scr_coulomb_t.cpp:1341 pads its own
   * call count with the comment "prevent dead block in thc.Z() in case nq_loc is not
   * the same for all processors".
   *
   * THE FIX is to acquire every tile ONCE, here, where all ranks are in lockstep, and
   * hand the evaluator a plain node-shared array. `contour_residue_batch` therefore
   * takes NO reader handle at all -- not `thc`, not an ERI object, nothing that can
   * reach a communicator. That is the invariant made structural rather than remembered:
   * to reintroduce the deadlock one would have to add a parameter, not just a call.
   *
   * Cost: nq_ibz collectives per context build (once per SCF iteration), against the
   * O(n_eval x nJ) it replaces. Memory: nq_ibz x Np^2 complex, ONE COPY PER NODE.
   *
   * [gate: tc_contour_multirank_zseq, which runs deliberately divergent per-rank target
   *  lists under mpiexec -- it hangs on the pre-fix code and completes on this one.]
   */
  template<typename thc_t>
  std::shared_ptr<sArray_t<Array_view_3D_t>> gather_Z_tiles(thc_t &thc) {
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    const long nq = thc.nqpts_ibz();
    const long NP = thc.Np();
    auto sZ = std::make_shared<sArray_t<Array_view_3D_t>>(
        math::shm::make_shared_array<Array_view_3D_t>(*mpi, {nq, NP, NP}));
    sZ->set_zero();
    sZ->win().fence();
    mpi->comm.barrier();
    for (long iq = 0; iq < nq; ++iq) {
      // COLLECTIVE, and deliberately so: the SAME iq on every rank, in a loop whose
      // trip count is nq_ibz on every rank. This is the only place the contour route
      // touches the reader after the context is built.
      auto Zq = thc.template Z<HOST_MEMORY>(int(iq));
      if (mpi->node_comm.root()) sZ->local()(iq, all, all) = Zq;
    }
    sZ->win().fence();
    mpi->comm.barrier();
    return sZ;
  }

  namespace detail {

    /**
     * The evaluator's work space, OWNED BY THE SOURCE and grown once.
     *
     * `Rb` is the big one -- (2 x chunk, Np^2) -- and at the production sizes it IS the
     * batch budget, tens to hundreds of MB. The evaluator is called once per (s,k)
     * evaluation point, tens of thousands of times per SCF iteration, so allocating it
     * per call would be that many malloc/free cycles of a buffer far larger than any
     * cache. Buffers only ever GROW here; a shorter target list reuses the allocation.
     */
    struct residue_scratch_t {
      nda::array<ComplexType, 1> zz;   ///< (>= 2*nt) the target list + its mirrors
      nda::array<ComplexType, 2> F;    ///< (>= 2*nt, r); grown by transform_factor_t
      nda::array<ComplexType, 2> Fc, Rb, Bb;
      nda::matrix<ComplexType> Pi, W, B, Bc, T, Mr;

      /** @param nz    2*nt, the target-plus-mirror count of this call
       *  @param nrow  2*chunk, the depth of the per-transfer contraction buffers */
      void reserve(long nz, long nrow, long r, long NP, long nbnd) {
        if (zz.size() < nz) zz.resize(nz);
        if (Fc.shape()[0] < nrow or Fc.shape()[1] != r) Fc.resize(nrow, r);
        if (Rb.shape()[0] < nrow or Rb.shape()[1] != NP * NP) Rb.resize(nrow, NP * NP);
        if (Pi.shape()[0] != NP) { Pi.resize(NP, NP); W.resize(NP, NP); }
        if (B.shape()[0] != NP or B.shape()[1] != nbnd) {
          B.resize(NP, nbnd); Bc.resize(NP, nbnd); T.resize(NP, nbnd); Bb.resize(NP, nbnd);
        }
        if (Mr.shape()[0] != nbnd) Mr.resize(nbnd, nbnd);
      }
    };

    /**
     * The batched residue evaluation, shared by every maker below. `nt` targets are read
     * from the first entries of Js/zs and written to the first slices of Ms.
     *
     * ⚠ COLLECTIVE-FREE BY CONSTRUCTION -- see the invariant on `gather_Z_tiles` above.
     * It takes the Coulomb tiles as a plain node-shared array and holds NO reader handle,
     * so there is nothing in scope here that can reach an MPI communicator. Keep it that
     * way: this function runs where every rank has different work.
     */
    inline void contour_residue_batch(qp_modea::modea_ctx const &ctx,
                                      ctx_t const &pctx,
                                      tc::transform_factor_t const &tf,
                                      sArray_t<Array_view_4D_t> const &sPi,
                                      sArray_t<Array_view_3D_t> const &sZ,
                                      long NP,
                                      methods::wc_line::solve_stats_t *sstat,
                                      residue_scratch_t &sc,
                                      long nchunk, long is, long ik, long nt,
                                      nda::array<long, 1> const &Js,
                                      nda::array<ComplexType, 1> const &zs,
                                      nda::array<ComplexType, 3> &Ms) {
      decltype(nda::range::all) all;
      const long r = pctx.c.rank;
      const long nbnd = ctx.nbnd;
      if (nt <= 0) return;
      utils::check(Js.size() >= nt and zs.size() >= nt and Ms.shape()[0] >= nt
                   and Ms.shape()[1] == nbnd and Ms.shape()[2] == nbnd,
                   "p_contour::contour_residue_batch: {} targets against J {}, z {}, "
                   "Ms ({}, {}, {}); nbnd = {}.", nt, Js.size(), zs.size(),
                   Ms.shape()[0], Ms.shape()[1], Ms.shape()[2], nbnd);
      const long bi = ctx.bstore_index(is, ik);
      utils::check(bi >= 0, "contour residue source: no band store for ({}, {}).", is, ik);
      auto const &bs = ctx.bstore[std::size_t(bi)];

      const long nc_max = std::max(1L, (nchunk > 0) ? std::min(nchunk, nt) : nt);
      sc.reserve(2 * nt, 2 * nc_max, r, NP, nbnd);
      auto &Fc = sc.Fc;
      auto &Rb = sc.Rb;
      auto &Pi = sc.Pi;
      auto &W = sc.W;
      auto &B = sc.B;
      auto &Bc = sc.Bc;
      auto &T = sc.T;
      auto &Mr = sc.Mr;
      auto &Bb = sc.Bb;

      // (1) THE TRANSFORM ROWS, for the whole call in ONE gemm: rows [0, nt) are the
      //     resonant targets, rows [nt, 2nt) their conjugate mirrors. This step is
      //     q-INDEPENDENT -- it costs r*nD per row and nD is the 2500-point adapted
      //     grid, which is why it is batched across the call and not merely within a
      //     transfer, where the groups can be a single target wide.
      auto zv = sc.zz(nda::range(0, 2 * nt));
      for (long i = 0; i < nt; ++i) {
        zv(i) = zs(i);
        zv(nt + i) = tc::mirror_target(zs(i));
      }
      auto &F = sc.F;                                     // (2*nt, r), sized by apply_many
      tf.apply_many(nda::make_regular(zv), F);

      // group the targets by IBZ transfer -- the axis the Pi slab is shared over
      std::vector<long> ord(static_cast<std::size_t>(nt));
      for (long i = 0; i < nt; ++i) ord[static_cast<std::size_t>(i)] = i;
      std::stable_sort(ord.begin(), ord.end(), [&](long a, long b) {
        return bs.qs_of_J(Js(a)) < bs.qs_of_J(Js(b));
      });

      auto Zloc = sZ.local();
      utils::check(Zloc.shape()[1] == NP and Zloc.shape()[2] == NP,
                   "p_contour::contour_residue_batch: the Z tile table is ({}, {}, {}), "
                   "expected (nq_ibz, {}, {}). It must come from gather_Z_tiles.",
                   Zloc.shape()[0], Zloc.shape()[1], Zloc.shape()[2], NP, NP);

      long g0 = 0;
      while (g0 < nt) {
        const long qs = bs.qs_of_J(Js(ord[std::size_t(g0)]));
        long g1 = g0;
        while (g1 < nt and bs.qs_of_J(Js(ord[std::size_t(g1)])) == qs) ++g1;
        auto Pislab = nda::reshape(sPi.local()(qs, all, all, all),
                                   std::array<long, 2>{r, NP * NP});
        // the PRE-GATHERED tile -- a local read, never thc.Z(qs), which is collective
        auto Zq = Zloc(qs, all, all);

        for (long c0 = g0; c0 < g1; c0 += nc_max) {
          const long nc = std::min(nc_max, g1 - c0);
          // (2) R = F . Pi(q_s) -- ONE gemm for the chunk, (2nc x r) x (r x Np^2)
          for (long i = 0; i < nc; ++i) {
            const long slot = ord[std::size_t(c0 + i)];
            for (long j = 0; j < r; ++j) {
              Fc(i, j) = F(slot, j);
              Fc(nc + i, j) = F(nt + slot, j);
            }
          }
          auto Fv = Fc(nda::range(0, 2 * nc), all);
          auto Rv = Rb(nda::range(0, 2 * nc), all);
          nda::blas::gemm(ComplexType(1.0), Fv, Pislab, ComplexType(0.0), Rv);

          for (long i = 0; i < nc; ++i) {
            const long slot = ord[std::size_t(c0 + i)];
            const long J = Js(slot);
            const bool wconj = (bs.wconj_of_J(J) != 0);
            // (3) eq (SIGN): Pi = -[ R + Rm^dag ]
            for (long P = 0; P < NP; ++P)
              for (long Q = 0; Q < NP; ++Q)
                Pi(P, Q) = -(Rv(i, P * NP + Q) + std::conj(Rv(nc + i, Q * NP + P)));
            // (4) CoQuI's Dyson chain
            methods::wc_line::dyson_wc_line(Zq, Pi, W, sstat);
            // (5) the trev-q rule + the band sandwich, stage 2's exact form
            bs.band(J, Bb);
            for (long P = 0; P < NP; ++P)
              for (long a = 0; a < nbnd; ++a) {
                B(P, a) = Bb(P, a);
                Bc(P, a) = std::conj(Bb(P, a));
              }
            if (wconj) {
              nda::blas::gemm(W, B, T);
              nda::blas::gemm(nda::transpose(Bc), T, Mr);
              for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b)
                  Ms(slot, a, b) = ctx.cd_pref * std::conj(Mr(a, b));
            } else {
              nda::blas::gemm(W, Bc, T);
              nda::blas::gemm(nda::transpose(B), T, Mr);
              for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b) Ms(slot, a, b) = ctx.cd_pref * Mr(a, b);
            }
          }
        }
        g0 = g1;
      }
    }

  } // namespace detail

  /**
   * The OWNING variant: every input is held by shared_ptr inside the closure, so the
   * returned function is valid after the builder's scope ends. This is what
   * build_modea_context installs on modea_ctx.
   */
  inline qp_modea::cd_residue_batch_fn make_contour_residue_batch_owning(
      qp_modea::modea_ctx const &ctx,
      std::shared_ptr<ctx_t> pctx,
      std::shared_ptr<tc::transform_factor_t> tf,
      std::shared_ptr<sArray_t<Array_view_4D_t>> sPi,
      std::shared_ptr<sArray_t<Array_view_3D_t>> sZ,   // from gather_Z_tiles, NOT a reader
      long NP,
      std::shared_ptr<methods::wc_line::solve_opts_t> sopt,
      std::shared_ptr<methods::wc_line::solve_stats_t> sstat,
      long nchunk) {
    utils::check(ctx.have_bstore,
                 "p_contour::make_contour_residue_batch_owning: no band factors.");
    auto sc = std::make_shared<detail::residue_scratch_t>();
    return [&ctx, pctx, tf, sPi, sZ, NP, sopt, sstat, sc, nchunk]
           (long is, long ik, long nt, nda::array<long, 1> const &Js,
            nda::array<ComplexType, 1> const &zs, nda::array<ComplexType, 3> &Ms) {
      detail::contour_residue_batch(ctx, *pctx, *tf, *sPi, *sZ, NP, sstat.get(), *sc,
                                    nchunk, is, ik, nt, Js, zs, Ms);
    };
  }

  /** The non-owning batched variant (the unit gates hold their own inputs). */
  inline qp_modea::cd_residue_batch_fn make_contour_residue_batch(
      qp_modea::modea_ctx const &ctx,
      ctx_t const &pctx,
      tc::transform_factor_t const &tf,
      sArray_t<Array_view_4D_t> const &sPi,      // (nq_ibz, rank, Np, Np)
      sArray_t<Array_view_3D_t> const &sZ,       // (nq_ibz, Np, Np) from gather_Z_tiles
      long NP,
      methods::wc_line::solve_opts_t const &sopt,
      methods::wc_line::solve_stats_t &sstat,
      long nchunk = 0) {
    utils::check(ctx.have_bstore,
                 "p_contour::make_contour_residue_batch: the mode-A context carries no "
                 "band factors; build it with modea_opts::cd_bfactor / cd_bstore_cap_gb.");
    auto sc = std::make_shared<detail::residue_scratch_t>();
    return [&ctx, &pctx, &tf, &sPi, &sZ, NP, &sopt, &sstat, sc, nchunk]
           (long is, long ik, long nt, nda::array<long, 1> const &Js,
            nda::array<ComplexType, 1> const &zs, nda::array<ComplexType, 3> &Ms) {
      detail::contour_residue_batch(ctx, pctx, tf, sPi, sZ, NP, &sstat, *sc, nchunk,
                                    is, ik, nt, Js, zs, Ms);
    };
  }

  /**
   * The SCALAR variant, kept for callers that hold one (J, z) at a time. It is the batch
   * core at nt = 1, so there is exactly ONE implementation of the algebra.
   */
  inline qp_modea::cd_residue_fn make_contour_residue_source(
      qp_modea::modea_ctx const &ctx,
      ctx_t const &pctx,
      tc::transform_factor_t const &tf,
      sArray_t<Array_view_4D_t> const &sPi,      // (nq_ibz, rank, Np, Np)
      sArray_t<Array_view_3D_t> const &sZ,       // (nq_ibz, Np, Np) from gather_Z_tiles
      long NP,
      methods::wc_line::solve_opts_t const &sopt,
      methods::wc_line::solve_stats_t &sstat) {
    utils::check(ctx.have_bstore,
                 "p_contour::make_contour_residue_source: the mode-A context carries no "
                 "band factors; build it with modea_opts::cd_bfactor / cd_bstore_cap_gb.");
    auto sc = std::make_shared<detail::residue_scratch_t>();
    return [&ctx, &pctx, &tf, &sPi, &sZ, NP, &sopt, &sstat, sc]
           (long is, long ik, long J, ComplexType z, nda::array<ComplexType, 2> &Ms) {
      const long nbnd = ctx.nbnd;
      nda::array<long, 1> Js(1);
      nda::array<ComplexType, 1> zs(1);
      nda::array<ComplexType, 3> M1(1, nbnd, nbnd);
      Js(0) = J;
      zs(0) = z;
      detail::contour_residue_batch(ctx, pctx, tf, sPi, sZ, NP, &sstat, *sc, 1,
                                    is, ik, 1L, Js, zs, M1);
      Ms.resize(nbnd, nbnd);
      for (long a = 0; a < nbnd; ++a)
        for (long b = 0; b < nbnd; ++b) Ms(a, b) = M1(0, a, b);
    };
  }

  /** One-line banner + the geometry table. */
  inline void log_contour(ctx_t const &ctx, opts_t const &o, int lvl) {
    auto const &g = ctx.geom;
    auto const &c = ctx.c;
    app_log(lvl, "  - TILTED CONTOUR (TC-2):       Delta in [{:.6g}, {:.6g}] a.u. "
                 "([{:.4g}, {:.4g}] eV) from {} occupied x {} empty QP states; "
                 "W_band = {:.4g} eV ({} valence cluster(s)); N_k = {}",
            g.dmin, g.dmax, g.dmin * ha_to_eV, g.dmax * ha_to_eV, g.n_occ, g.n_emp,
            g.W_band * ha_to_eV, g.n_valence_clusters, g.nk_lin);
    app_log(lvl, "  - TC geometry:                 delta = {:.6g} a.u. ({:.4g} eV; "
                 "recipe floor {:.4g} eV = {} W_band/N_k -- FLAGGED constant), "
                 "W_target = {:.4g} eV{}, rho = {:.3g}, profile = {}",
            g.delta, g.delta * ha_to_eV, g.delta_mesh * ha_to_eV, delta_mesh_const,
            g.W_target * ha_to_eV,
            g.window_floored ? " [FLOORED at delta: Dmin > zeta_max]" : "",
            o.rho, o.profile);
    app_log(lvl, "  - TC contour:                  theta = {:.6f} rad (tan {:.4g}), "
                 "gamma = {:.4g}, S = {:.4g} a.u. ({:.3g} fs); eps = {:.0e}, "
                 "eps_tr = {:.0e}",
            c.g.theta, c.g.tan_theta, c.g.gamma, c.g.S,
            c.g.S / ha_to_eV * tc::hbar_eV_fs, o.eps, c.g.eps_tr);
    app_log(lvl, "  - TC rank:                     {} nodes at lambda > eps^2 lambda_max "
                 "(spec-convention rank {}, conditioning ceiling {}); eq-6 estimate "
                 "{:.1f}; s in [{:.4g}, {:.4g}] of S",
            c.rank, c.rank_eps, c.rank_ceil,
            tc::eq6_Ns(g.dmin, g.dmax, g.W_target, g.delta, o.rho, o.eps),
            c.s(0), c.s(c.rank - 1));
    app_log(lvl, "  - TC beta guard:               beta / (S sin theta) = {:.3f} "
                 "(> 1 required; the occupation factor cancels the contour growth)",
            ctx.beta_guard);
  }

} // namespace p_contour
} // namespace methods

#endif // COQUI_P_CONTOUR_HPP
