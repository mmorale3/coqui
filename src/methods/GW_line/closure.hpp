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

#ifndef COQUI_METHODS_GW_LINE_CLOSURE_HPP
#define COQUI_METHODS_GW_LINE_CLOSURE_HPP

/**
 * Closure of the line scGW loop (notes section 6; plan 6.3(f); python coqui/cayley/cayley/line/closure.py and the closure
 * block of line/driver.py::LineSCGW.iterate):
 *
 *   Sigma^{>/<}(k) at the dense fermionic nodes
 *     -> real-pole fit per sector on ONE-SIDED bases (particle basis: poles in [gap_p, lam]; hole basis: [-lam, -gap_h])
 *     -> Cayley moments C^(n), n = 0..K+1, of the TOTAL measure (both sectors in one sum; python lehmann_from_sigma)
 *     -> block-Toeplitz upfolding (d_l, W) -> Htilde = [[H_stat - mu, W], [W^dag, diag d]] -> Lehmann G (e_m, v_m)
 *     -> chemical potential over all k (widest admissible QP gap), re-centring e_m -> e_m - dmu
 *     -> the next pole data, by one of two representations (g_repr_params_t, S7c):
 *        "lehmann"   : the Lehmann (e_m, v_m) themselves, split by the sign of e_m, after PRUNING only (no refit): poles
 *                      with |e_m| > g_emax (moment-truncation artefacts) and poles of weight |v_m|^2 < g_wtol are dropped
 *                      (optionally also tiny-weight poles very close to mu, see g_repr_params_t), the dropped weights are
 *                      logged; factorized residues v_m v_m^dagger (positive, rank 1), D and N exact for the retained poles;
 *        "compressed": per-sector refit of the Lehmann G on GAPLESS one-sided bases (g_gap = 0) from the dense nodes; poles
 *                      with |e_m| > lam are dropped and their weight logged -> matrix coefficients (signed; the S6 path,
 *                      python parity).
 *
 * Everything is on the host and mu-relative: Sigma is sampled at zeta (relative to the centre mu at which it was
 * computed), Hrel = H0 + F - mu, the returned Lehmann energies and compressed poles are relative to the NEW centre
 * mu + dmu.
 *
 * k-parallel: k is distributed round-robin over the ranks of `comm` (owner(k) = k mod np); every rank does the closure
 * of its k, then the results are packed into flat buffers at global offsets (ragged Lehmann pole counts) and summed
 * with ONE all_reduce per buffer, in which every element has exactly one nonzero contribution (x + 0 = x exactly).
 * Each k is processed by the same code on the same input whatever its owner, so the result is bitwise independent of
 * the number of ranks. Collective over comm.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/blas_scope.hpp"
#include "methods/GW_line/closure_device.hpp"

namespace methods::gw_line {

using numerics::line_dlr::line_basis_t;

/// Real-pole Sigma_c of one k (mu-relative): poles w = [hole basis poles, particle basis poles] (python order), g [r, nb, nb].
struct sigma_poles_t {
  nda::array<double, 1> w;
  nda::array<ComplexType, 3> g;
};

/// python fit_sigma_sectors: LS fits of Sigma^> on the particle basis and Sigma^< on the hole basis at the nodes zeta.
inline sigma_poles_t fit_sigma_sectors(line_basis_t const &bp, line_basis_t const &bh, nda::array<ComplexType, 1> const &zeta,
                                       nda::array<ComplexType, 3> const &Sig_p, nda::array<ComplexType, 3> const &Sig_h) {
  auto gp = bp.fit(zeta, Sig_p);
  auto gh = bh.fit(zeta, Sig_h);
  const long rp = bp.rank, rh = bh.rank, nb = Sig_p.extent(1);
  sigma_poles_t out{nda::array<double, 1>(rh + rp), nda::array<ComplexType, 3>(rh + rp, nb, nb)};
  for (long l = 0; l < rh; ++l) {
    out.w(l)                      = bh.w(l);
    out.g(l, nda::ellipsis{})      = gh(l, nda::ellipsis{});
  }
  for (long l = 0; l < rp; ++l) {
    out.w(rh + l)                 = bp.w(l);
    out.g(rh + l, nda::ellipsis{}) = gp(l, nda::ellipsis{});
  }
  return out;
}

/// Parameters of the moment closure (notes section 6; plan section 5; S7f: cut modes, phase continuity, diagnostics).
struct closure_params_t {
  double wp       = 0.11;    ///< Cayley scale (Ha)
  long K          = 24;      ///< moments 0..K (+ the held-out K+1)
  double tol_gram = 1e-10;   ///< relative eigenvalue cutoff of the block-Toeplitz Gram matrix
  long nphi       = 8;       ///< coarse terminal-phase scan (+ golden section)
  double tol_svd  = 1e-12;   ///< relative singular-value cutoff of D+ D-^dagger (python 1e-12)
  std::string gram_cut = "hard";   ///< "hard" (python) | "gap" | "smooth" (numerics::line_dlr::upfold_opts_t)
  std::string svd_cut  = "hard";   ///< "hard" (python) | "gap"
  double cut_window    = 10.0;     ///< window factor of the gap / smooth cuts
  double phase_keep    = 0.0;      ///< > 0: phase continuity with phi_prev (per k) and this tolerance factor
  std::vector<double> phi_prev;    ///< [nk] terminal phase of the previous closure per k (empty / NaN: none)
  // diagnostics (tests): relative noise added to the moments of every k (seeded per k), forced decisions per k
  double moment_noise = 0.0;
  unsigned noise_seed = 0;
  std::vector<long> force_rgram, force_r1;
  std::vector<double> force_phi;
  // S7g performance: BLAS threads inside the per-k closure (<= 0: untouched) and the linear-algebra drivers
  long blas_threads      = 0;
  std::string svd_driver = "gesvd";   ///< numerics::line_dlr::upfold_opts_t::svd_driver
  std::string ueig       = "schur";   ///< numerics::line_dlr::upfold_opts_t::ueig
  numerics::line_dlr::lapack_hooks_t const *hooks = nullptr;   ///< external (device) eigensolvers; null: host LAPACK

  numerics::line_dlr::upfold_opts_t upfold_opts(long ik) const {
    numerics::line_dlr::upfold_opts_t o;
    o.tol_c0     = 1e-12;
    o.tol_gram   = tol_gram;
    o.tol_svd    = tol_svd;
    o.nphi       = nphi;
    o.gram_cut   = gram_cut;
    o.svd_cut    = svd_cut;
    o.cut_window = cut_window;
    o.svd_driver = svd_driver;
    o.ueig       = ueig;
    o.hooks      = hooks;
    if (ik >= 0) {
      if (phase_keep > 0.0 and ik < long(phi_prev.size())) {
        o.phi_prev   = phi_prev[ik];
        o.phase_keep = phase_keep;
      }
      if (ik < long(force_rgram.size())) o.force_rgram = force_rgram[ik];
      if (ik < long(force_r1.size())) o.force_r1 = force_r1[ik];
      if (ik < long(force_phi.size())) o.force_phi = force_phi[ik];
    }
    return o;
  }
};

/// Scalar diagnostics of the upfolding of one k (numerics::line_dlr::upfold_result_t without the arrays; S7f).
struct upfold_diag_t {
  long r_gram = 0, r1 = 0, gram_near = 0, svd_near = 0, phi_index = -1, n_rejected = 0, phi_kept = 0;
  double phi = 0.0, gram_margin = 0.0, gram_ratio = 0.0, svd_margin = 0.0, phi_tie = 0.0;
  static constexpr long nfields = 13;
  void pack(double *x) const {
    double v[nfields] = {double(r_gram), double(r1), double(gram_near), double(svd_near), double(phi_index), double(n_rejected),
                         double(phi_kept), phi, gram_margin, gram_ratio, svd_margin, phi_tie, 0.0};
    std::copy_n(v, nfields, x);
  }
  void unpack(double const *x) {
    r_gram = std::llround(x[0]); r1 = std::llround(x[1]); gram_near = std::llround(x[2]); svd_near = std::llround(x[3]);
    phi_index = std::llround(x[4]); n_rejected = std::llround(x[5]); phi_kept = std::llround(x[6]);
    phi = x[7]; gram_margin = x[8]; gram_ratio = x[9]; svd_margin = x[10]; phi_tie = x[11];
  }
  static upfold_diag_t from(numerics::line_dlr::upfold_result_t const &u) {
    upfold_diag_t d;
    d.r_gram = u.r_gram; d.r1 = u.r1; d.gram_near = u.gram_near; d.svd_near = u.svd_near; d.phi_index = u.phi_index;
    d.n_rejected = u.n_rejected; d.phi_kept = u.phi_kept ? 1 : 0; d.phi = u.phi; d.gram_margin = u.gram_margin;
    d.gram_ratio = u.gram_ratio; d.svd_margin = u.svd_margin; d.phi_tie = u.phi_tie;
    return d;
  }
};

/**
 * Representation of G handed to the next iteration (S7c; driver keys g_repr, g_emax, g_wtol, g_emin_frac, g_wsmall).
 * "lehmann": a pole m (mu-relative energy e_m, weight w_m = |v_m|^2) is dropped if
 *   (i)   |e_m| > emax (default: lam of the G bases; weight logged as "dropped", as the compressed path), or e_m == 0;
 *   (ii)  w_m < wtol (count and summed weight logged);
 *   (iii) |e_m| < emin_frac * (QP half gap) AND w_m < wsmall (count and summed weight logged; off when emin_frac <= 0).
 */
struct g_repr_params_t {
  std::string repr = "compressed";   ///< "lehmann" | "compressed"
  double emax      = -1.0;           ///< (i); < 0: lam of the particle G basis
  double wtol      = 1e-12;          ///< (ii)
  double emin_frac = 0.0;            ///< (iii) fraction of the QP half gap
  double wsmall    = 0.0;            ///< (iii) weight threshold
};

/// Result of the closure of one k: the Lehmann G (mu-relative to the closure centre) and the upfolded Sigma_c poles.
struct closure_k_t {
  nda::array<double, 1> e;         ///< [M] Lehmann energies, ascending
  nda::array<ComplexType, 2> v;    ///< [nb, M] Lehmann vectors
  nda::array<double, 1> d;         ///< [np] upfolded Sigma_c poles
  nda::array<ComplexType, 2> W;    ///< [nb, np] couplings
  double heldout = 0.0;            ///< held-out moment error of the upfolding
  upfold_diag_t diag;              ///< hard decisions of the upfolding (S7f)
  /// S7g profile (s): moments, C0, Gram, SVD, U eigen, Lehmann eigen; U-eigen fallbacks to the Schur form
  double t_mom = 0.0, t_c0 = 0.0, t_gram = 0.0, t_svd = 0.0, t_ueig = 0.0, t_leh = 0.0;
  long ueig_fallback = 0;
};

/// python lehmann_from_sigma: moments of the total measure -> upfold_block -> eig of Htilde. ik: k index (per-k options,
/// noise seed; -1 = none).
inline closure_k_t closure_k(nda::array<ComplexType, 2> const &Hrel, sigma_poles_t const &sp, closure_params_t const &p,
                             long ik = -1) {
  using namespace numerics::line_dlr;
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  auto C = moments_from_poles(sp.w, sp.g, p.wp, p.K + 1);
  if (p.moment_noise > 0.0) {   // diagnostic: relative complex Gaussian noise on every moment, scale max|C^(0)|
    double c0 = 0.0;
    for (long i = 0; i < C.extent(1); ++i)
      for (long j = 0; j < C.extent(2); ++j) c0 = std::max(c0, std::abs(C(0, i, j)));
    std::mt19937_64 gen(std::uint64_t(p.noise_seed) * 1000003ull + std::uint64_t(ik + 1));
    std::normal_distribution<double> N01;
    for (auto &x : C) x += p.moment_noise * c0 * ComplexType(N01(gen), N01(gen));
  }
  const auto t1 = clk::now();
  auto up = upfold_block(C, p.K, p.wp, p.upfold_opts(ik));
  const auto t2 = clk::now();
  auto L  = lehmann(Hrel, up.d, up.W, p.hooks);
  const auto t3 = clk::now();
  auto dg = upfold_diag_t::from(up);
  closure_k_t out{std::move(L.e), std::move(L.v), std::move(up.d), std::move(up.W), up.residual, dg};
  out.t_mom         = std::chrono::duration<double>(t1 - t0).count();
  out.t_c0          = up.t_c0;
  out.t_gram        = up.t_gram;
  out.t_svd         = up.t_svd;
  out.t_ueig        = up.t_ueig;
  out.t_leh         = std::chrono::duration<double>(t3 - t2).count();
  out.ueig_fallback = up.ueig_fallback;
  return out;
}

/**
 * Lehmann G(zeta) = sum_{m in idx} v_m v_m^dag / (zeta - e_m) at the nodes, [nz, nb, nb] (python compress_sectors.lehmann).
 */
inline nda::array<ComplexType, 3> lehmann_on_nodes(nda::array<ComplexType, 1> const &zeta, nda::array<double, 1> const &e,
                                                   nda::array<ComplexType, 2> const &v, std::vector<long> const &idx) {
  const long nz = zeta.size(), nb = v.extent(0), M = long(idx.size());
  nda::array<ComplexType, 3> G(nz, nb, nb);
  G() = ComplexType(0.0);
  if (M == 0) return G;
  nda::matrix<ComplexType> Vs(nb, M), Vz(nb, M), Gz(nb, nb);
  for (long i = 0; i < nb; ++i)
    for (long m = 0; m < M; ++m) Vs(i, m) = v(i, idx[m]);
  for (long iz = 0; iz < nz; ++iz) {
    for (long m = 0; m < M; ++m) {
      const ComplexType kz = 1.0 / (zeta(iz) - e(idx[m]));
      for (long i = 0; i < nb; ++i) Vz(i, m) = Vs(i, m) * kz;
    }
    nda::blas::gemm(ComplexType(1.0), Vz, nda::dagger(Vs), ComplexType(0.0), Gz);
    G(iz, nda::range::all, nda::range::all) = Gz;
  }
  return G;
}

/// Compressed per-sector poles of one k plus the dropped weight (python compress_sectors).
struct compress_k_t {
  pole_sector_t part, hole;
  double dropped = 0.0;
};

/**
 * python compress_sectors: poles with |e| <= emax (default: the particle basis range lam) are split by sign and their
 * Lehmann G at the nodes is refit on the gapless one-sided bases gp (particle) / gh (hole); the weight sum |v|^2 of the
 * dropped poles is returned. Poles exactly at e = 0 belong to neither sector (as python).
 */
inline compress_k_t compress_sectors(line_basis_t const &gp, line_basis_t const &gh, nda::array<ComplexType, 1> const &zeta,
                                     nda::array<double, 1> const &e, nda::array<ComplexType, 2> const &v, double emax = -1.0) {
  if (emax < 0.0) emax = gp.lam;
  const long M = e.size(), nb = v.extent(0);
  std::vector<long> ip, ih;
  compress_k_t out;
  for (long m = 0; m < M; ++m) {
    if (std::abs(e(m)) <= emax) {
      if (e(m) > 0.0) ip.push_back(m);
      else if (e(m) < 0.0) ih.push_back(m);
    } else {
      for (long i = 0; i < nb; ++i) out.dropped += std::norm(v(i, m));
    }
  }
  auto Gp = lehmann_on_nodes(zeta, e, v, ip);
  auto Gh = lehmann_on_nodes(zeta, e, v, ih);
  out.part = pole_sector_t{nda::array<double, 1>(gp.w), gp.fit(zeta, Gp)};
  out.hole = pole_sector_t{nda::array<double, 1>(gh.w), gh.fit(zeta, Gh)};
  return out;
}

namespace detail {
/// Sum a buffer over comm in place (each element has one nonzero contributor -> exact).
template <typename T> inline void exact_allreduce(boost::mpi3::communicator &comm, T *data, long n) {
  if (comm.size() > 1 and n > 0) comm.all_reduce_in_place_n(data, n, std::plus<>{});
}
} // namespace detail

/// Gathered Lehmann representations of all k (identical on every rank).
struct lehmann_all_t {
  std::vector<nda::array<double, 1>> e;
  std::vector<nda::array<ComplexType, 2>> v;
};

/**
 * Pack the Lehmann (e, v) of the owned k (owner(k) = k mod np; other entries of `loc` are ignored) and all_reduce them
 * into every rank. Ragged pole counts: counts first (one all_reduce), then the flat buffers at global offsets.
 */
inline lehmann_all_t gather_lehmann(boost::mpi3::communicator &comm, long nk, long nb,
                                    std::vector<nda::array<double, 1>> const &e_loc,
                                    std::vector<nda::array<ComplexType, 2>> const &v_loc) {
  const long np = comm.size(), rank = comm.rank();
  nda::array<double, 1> cnt(nk);
  cnt() = 0.0;
  for (long ik = rank; ik < nk; ik += np) cnt(ik) = double(e_loc[ik].size());
  detail::exact_allreduce(comm, cnt.data(), nk);
  std::vector<long> off(nk + 1, 0);
  for (long ik = 0; ik < nk; ++ik) off[ik + 1] = off[ik] + long(std::llround(cnt(ik)));
  nda::array<double, 1> E(off[nk]);
  nda::array<ComplexType, 1> V(off[nk] * nb);
  E() = 0.0;
  V() = ComplexType(0.0);
  for (long ik = rank; ik < nk; ik += np) {
    const long M = e_loc[ik].size();
    for (long m = 0; m < M; ++m) E(off[ik] + m) = e_loc[ik](m);
    for (long i = 0; i < nb; ++i)
      for (long m = 0; m < M; ++m) V(off[ik] * nb + i * M + m) = v_loc[ik](i, m);
  }
  detail::exact_allreduce(comm, E.data(), E.size());
  detail::exact_allreduce(comm, V.data(), V.size());
  lehmann_all_t out;
  out.e.resize(nk);
  out.v.resize(nk);
  for (long ik = 0; ik < nk; ++ik) {
    const long M = off[ik + 1] - off[ik];
    out.e[ik]    = nda::array<double, 1>(M);
    out.v[ik]    = nda::array<ComplexType, 2>(nb, M);
    for (long m = 0; m < M; ++m) out.e[ik](m) = E(off[ik] + m);
    for (long i = 0; i < nb; ++i)
      for (long m = 0; m < M; ++m) out.v[ik](i, m) = V(off[ik] * nb + i * M + m);
  }
  return out;
}

/// Output of the full closure (identical on every rank).
struct closure_out_t {
  lehmann_all_t leh;              ///< Lehmann G per k, mu-relative to the NEW centre (re-centred)
  pole_data_t poles;              ///< compressed pole data (new centre)
  double dmu = 0.0;               ///< new mu - old mu
  double e_homo = 0.0, e_lumo = 0.0;   ///< QP edges relative to the NEW centre
  double N_mu = 0.0;              ///< electron count of the Lehmann G at the chosen mu (mu finder)
  double nel_lehmann = 0.0;       ///< 2 sum_k w_k sum_{e_m < 0} |v_m|^2 after re-centring
  double nel_compressed = 0.0;    ///< 2 sum_k w_k Tr D(k) of the compressed poles
  double dropped = 0.0;           ///< max over k of the dropped weight (python; |e| > emax)
  double dropped_sum = 0.0;       ///< sum over k of the dropped weight
  std::string repr = "compressed";   ///< representation of `poles`
  long pruned_w = 0;              ///< lehmann: poles dropped by the weight rule (ii), all k
  double pruned_w_weight = 0.0;   ///<   their summed weight
  long pruned_near = 0;           ///< lehmann: poles dropped by the near-mu rule (iii), all k
  double pruned_near_weight = 0.0;   ///< their summed weight
  std::vector<long> npoles;       ///< upfolded poles per k
  std::vector<double> heldout;    ///< held-out moment error per k
  std::vector<upfold_diag_t> diag;   ///< upfolding decisions per k (S7f; phi = the terminal phase for phase continuity)
};

/**
 * The closure for all k (python LineSCGW.iterate, step 3). Hrel (nk, nb, nb) = H0 + F - mu; Sig_p / Sig_h (nk, nz, nb, nb)
 * at the mu-relative nodes zeta; bp / bh: one-sided Sigma bases; gp / gh: gapless one-sided G bases; nelec: electrons per
 * cell (both spins). k weights uniform (nosym meshes). Collective over comm.
 * Sig_p / Sig_h may also be k-DISTRIBUTED (S7e, k_dist.hpp): (nloc, nz, nb, nb) with the rows of the k owned by this rank
 * (k mod np == rank, the ownership of the loop below); detected by extent(0) != nk (unambiguous for np > 1).
 */
inline closure_out_t closure(boost::mpi3::communicator &comm, nda::array<ComplexType, 3> const &Hrel,
                             nda::array<ComplexType, 4> const &Sig_p, nda::array<ComplexType, 4> const &Sig_h,
                             nda::array<ComplexType, 1> const &zeta, line_basis_t const &bp, line_basis_t const &bh,
                             line_basis_t const &gp, line_basis_t const &gh, closure_params_t const &p, double nelec,
                             utils::TimerManager &Timer, g_repr_params_t const &gr = {}) {
  auto all       = nda::range::all;
  const long nk  = Hrel.extent(0), nb = Hrel.extent(1), nz = zeta.size();
  const long np  = comm.size(), rank = comm.rank();
  const bool sig_loc = (Sig_p.extent(0) != nk);   // k-distributed Sigma (S7e): row of k = k / np
  const long nloc    = rank < nk ? (nk - 1 - rank) / np + 1 : 0;
  utils::check((Sig_p.extent(0) == nk or Sig_p.extent(0) == nloc) and Sig_p.extent(1) == nz and Sig_p.extent(2) == nb and
                   Sig_h.shape() == Sig_p.shape(),
               "gw_line::closure: Sigma shape mismatch");
  for (auto nm : {"closure_upfold", "closure_gather", "closure_mu", "closure_compress"}) Timer.add(nm);
  closure_out_t out;

  // 1. per owned k: sector fits -> moments -> upfold -> Lehmann (S7g: BLAS threads of the rank, p.blas_threads)
  Timer.start("closure_upfold");
  using clk = std::chrono::steady_clock;
  double prof[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // fit, moments, C0, Gram, SVD, U eigen, Lehmann, U-eigen fallbacks
  blas_threads_scope_t blas_scope(p.blas_threads);
  std::vector<nda::array<double, 1>> e_loc(nk);
  std::vector<nda::array<ComplexType, 2>> v_loc(nk);
  const long ni = 2 + upfold_diag_t::nfields;
  nda::array<double, 2> info(nk, ni);   // (npoles, heldout, diag)
  info() = 0.0;
  for (long ik = rank; ik < nk; ik += np) {
    const long ks = sig_loc ? ik / np : ik;
    const auto tf0 = clk::now();
    nda::array<ComplexType, 3> Sp(Sig_p(ks, all, all, all)), Sh(Sig_h(ks, all, all, all));
    auto sp = fit_sigma_sectors(bp, bh, zeta, Sp, Sh);
    prof[0] += std::chrono::duration<double>(clk::now() - tf0).count();
    nda::array<ComplexType, 2> H(Hrel(ik, all, all));
    auto ck   = closure_k(H, sp, p, ik);
    prof[1] += ck.t_mom; prof[2] += ck.t_c0; prof[3] += ck.t_gram; prof[4] += ck.t_svd; prof[5] += ck.t_ueig;
    prof[6] += ck.t_leh; prof[7] += double(ck.ueig_fallback);
    e_loc[ik] = std::move(ck.e);
    v_loc[ik] = std::move(ck.v);
    info(ik, 0) = double(ck.d.size());
    info(ik, 1) = ck.heldout;
    ck.diag.pack(&info(ik, 2));
  }
  Timer.stop("closure_upfold");
  {   // S7g: per-step profile of the closure (max over ranks of the per-rank sums over the owned k)
    double pmax[8];
    std::copy_n(prof, 8, pmax);
    if (comm.size() > 1) comm.all_reduce_n(prof, 8, pmax, boost::mpi3::max<>{});
    app_log(2, "          closure profile (s, max over ranks): fit {:.2f} moments {:.2f} C0 {:.2f} Gram {:.2f} SVD {:.2f} U-eigen "
               "{:.2f} Lehmann {:.2f} | BLAS threads {} ({}), drivers {} / {}{}, U-eigen Schur fallbacks {}, device fallbacks "
               "(rank 0, cumulative) {}",
            pmax[0], pmax[1], pmax[2], pmax[3], pmax[4], pmax[5], pmax[6], p.blas_threads > 0 ? p.blas_threads : 0,
            blas_scope.backend(), p.svd_driver, p.ueig, p.hooks ? " + GPU" : "", long(pmax[7]), device_lapack_failures());
  }

  Timer.start("closure_gather");
  out.leh = gather_lehmann(comm, nk, nb, e_loc, v_loc);
  detail::exact_allreduce(comm, info.data(), info.size());
  for (long ik = 0; ik < nk; ++ik) {
    out.npoles.push_back(long(std::llround(info(ik, 0))));
    out.heldout.push_back(info(ik, 1));
    out.diag.emplace_back();
    out.diag.back().unpack(&info(ik, 2));
  }
  Timer.stop("closure_gather");

  // 2. chemical potential (every rank, same data) and re-centring
  Timer.start("closure_mu");
  auto cp       = numerics::line_dlr::chemical_potential(out.leh.e, out.leh.v, nelec);
  out.dmu       = cp.mu;
  out.e_homo    = cp.e_homo - cp.mu;
  out.e_lumo    = cp.e_lumo - cp.mu;
  out.N_mu      = cp.N;
  out.nel_lehmann = 0.0;
  for (long ik = 0; ik < nk; ++ik) {
    out.leh.e[ik] -= cp.mu;
    for (long m = 0; m < out.leh.e[ik].size(); ++m)
      if (out.leh.e[ik](m) < 0.0)
        for (long i = 0; i < nb; ++i) out.nel_lehmann += 2.0 / double(nk) * std::norm(out.leh.v[ik](i, m));
  }
  Timer.stop("closure_mu");

  // 3a. Lehmann representation: pruning only (every rank, same data, k in order -> rank-count independent)
  if (gr.repr == "lehmann") {
    Timer.start("closure_compress");
    out.repr            = "lehmann";
    const double emax   = (gr.emax < 0.0) ? gp.lam : gr.emax;
    const double e_near = (gr.emin_frac > 0.0) ? gr.emin_frac * 0.5 * (out.e_lumo - out.e_homo) : 0.0;
    std::vector<nda::array<double, 1>> ek(nk);
    std::vector<nda::array<ComplexType, 2>> vk(nk);
    out.dropped = out.dropped_sum = 0.0;
    out.nel_compressed = 0.0;
    for (long ik = 0; ik < nk; ++ik) {
      auto const &e = out.leh.e[ik];
      auto const &v = out.leh.v[ik];
      std::vector<long> keep;
      double drop = 0.0;
      for (long m = 0; m < e.size(); ++m) {
        double w = 0.0;
        for (long i = 0; i < nb; ++i) w += std::norm(v(i, m));
        const double ae = std::abs(e(m));
        if (ae > emax or e(m) == 0.0) {
          drop += w;
        } else if (w < gr.wtol) {
          out.pruned_w += 1;
          out.pruned_w_weight += w;
        } else if (ae < e_near and w < gr.wsmall) {
          out.pruned_near += 1;
          out.pruned_near_weight += w;
        } else {
          keep.push_back(m);
          if (e(m) < 0.0) out.nel_compressed += 2.0 / double(nk) * w;
        }
      }
      ek[ik] = nda::array<double, 1>(long(keep.size()));
      vk[ik] = nda::array<ComplexType, 2>(nb, long(keep.size()));
      for (long j = 0; j < long(keep.size()); ++j) {
        ek[ik](j) = e(keep[j]);
        for (long i = 0; i < nb; ++i) vk[ik](i, j) = v(i, keep[j]);
      }
      out.dropped = std::max(out.dropped, drop);
      out.dropped_sum += drop;
    }
    out.poles = pole_data_t::from_lehmann(ek, vk);
    Timer.stop("closure_compress");
    return out;
  }
  utils::check(gr.repr == "compressed", "gw_line::closure: g_repr must be \"lehmann\" or \"compressed\" (got \"{}\")", gr.repr);

  // 3b. per owned k: compression; gather (fixed pole counts: the basis ranks)
  Timer.start("closure_compress");
  const long rp = gp.rank, rh = gh.rank;
  nda::array<ComplexType, 4> cpart(nk, rp, nb, nb), chole(nk, rh, nb, nb);
  nda::array<double, 1> drop(nk);
  cpart() = ComplexType(0.0);
  chole() = ComplexType(0.0);
  drop()  = 0.0;
  for (long ik = rank; ik < nk; ik += np) {
    auto c = compress_sectors(gp, gh, zeta, out.leh.e[ik], out.leh.v[ik]);
    cpart(ik, all, all, all) = c.part.coef;
    chole(ik, all, all, all) = c.hole.coef;
    drop(ik)                 = c.dropped;
  }
  detail::exact_allreduce(comm, cpart.data(), cpart.size());
  detail::exact_allreduce(comm, chole.data(), chole.size());
  detail::exact_allreduce(comm, drop.data(), drop.size());
  out.poles.nk = nk;
  out.poles.nb = nb;
  out.poles.part.resize(nk);
  out.poles.hole.resize(nk);
  out.dropped = out.dropped_sum = 0.0;
  for (long ik = 0; ik < nk; ++ik) {
    out.poles.part[ik] = pole_sector_t{nda::array<double, 1>(gp.w), nda::array<ComplexType, 3>(cpart(ik, all, all, all))};
    out.poles.hole[ik] = pole_sector_t{nda::array<double, 1>(gh.w), nda::array<ComplexType, 3>(chole(ik, all, all, all))};
    out.dropped        = std::max(out.dropped, drop(ik));
    out.dropped_sum += drop(ik);
  }
  out.nel_compressed = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long m = 0; m < rh; ++m)
      for (long i = 0; i < nb; ++i) out.nel_compressed += 2.0 / double(nk) * std::real(chole(ik, m, i, i));
  Timer.stop("closure_compress");
  return out;
}

} // namespace methods::gw_line

#endif
