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
 *     -> per-sector refit of the Lehmann G on GAPLESS one-sided bases (g_gap = 0) from the dense nodes; poles with
 *        |e_m| > lam are dropped and their weight logged -> the next pole data (matrix coefficients).
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
#include <cmath>
#include <numeric>
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

/// Parameters of the moment closure (notes section 6; plan section 5).
struct closure_params_t {
  double wp       = 0.11;    ///< Cayley scale (Ha)
  long K          = 24;      ///< moments 0..K (+ the held-out K+1)
  double tol_gram = 1e-10;   ///< relative eigenvalue cutoff of the block-Toeplitz Gram matrix
  long nphi       = 8;       ///< coarse terminal-phase scan (+ golden section)
};

/// Result of the closure of one k: the Lehmann G (mu-relative to the closure centre) and the upfolded Sigma_c poles.
struct closure_k_t {
  nda::array<double, 1> e;         ///< [M] Lehmann energies, ascending
  nda::array<ComplexType, 2> v;    ///< [nb, M] Lehmann vectors
  nda::array<double, 1> d;         ///< [np] upfolded Sigma_c poles
  nda::array<ComplexType, 2> W;    ///< [nb, np] couplings
  double heldout = 0.0;            ///< held-out moment error of the upfolding
};

/// python lehmann_from_sigma: moments of the total measure -> upfold_block -> eig of Htilde.
inline closure_k_t closure_k(nda::array<ComplexType, 2> const &Hrel, sigma_poles_t const &sp, closure_params_t const &p) {
  using namespace numerics::line_dlr;
  auto C  = moments_from_poles(sp.w, sp.g, p.wp, p.K + 1);
  auto up = upfold_block(C, p.K, p.wp, 1e-12, p.tol_gram, 1e-12, p.nphi);
  auto L  = lehmann(Hrel, up.d, up.W);
  return closure_k_t{std::move(L.e), std::move(L.v), std::move(up.d), std::move(up.W), up.residual};
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
  double dropped = 0.0;           ///< max over k of the dropped weight (python)
  double dropped_sum = 0.0;       ///< sum over k of the dropped weight
  std::vector<long> npoles;       ///< upfolded poles per k
  std::vector<double> heldout;    ///< held-out moment error per k
};

/**
 * The closure for all k (python LineSCGW.iterate, step 3). Hrel (nk, nb, nb) = H0 + F - mu; Sig_p / Sig_h (nk, nz, nb, nb)
 * at the mu-relative nodes zeta; bp / bh: one-sided Sigma bases; gp / gh: gapless one-sided G bases; nelec: electrons per
 * cell (both spins). k weights uniform (nosym meshes). Collective over comm.
 */
inline closure_out_t closure(boost::mpi3::communicator &comm, nda::array<ComplexType, 3> const &Hrel,
                             nda::array<ComplexType, 4> const &Sig_p, nda::array<ComplexType, 4> const &Sig_h,
                             nda::array<ComplexType, 1> const &zeta, line_basis_t const &bp, line_basis_t const &bh,
                             line_basis_t const &gp, line_basis_t const &gh, closure_params_t const &p, double nelec,
                             utils::TimerManager &Timer) {
  auto all       = nda::range::all;
  const long nk  = Hrel.extent(0), nb = Hrel.extent(1), nz = zeta.size();
  const long np  = comm.size(), rank = comm.rank();
  utils::check(Sig_p.extent(0) == nk and Sig_p.extent(1) == nz and Sig_p.extent(2) == nb and Sig_h.shape() == Sig_p.shape(),
               "gw_line::closure: Sigma shape mismatch");
  for (auto nm : {"closure_upfold", "closure_gather", "closure_mu", "closure_compress"}) Timer.add(nm);
  closure_out_t out;

  // 1. per owned k: sector fits -> moments -> upfold -> Lehmann
  Timer.start("closure_upfold");
  std::vector<nda::array<double, 1>> e_loc(nk);
  std::vector<nda::array<ComplexType, 2>> v_loc(nk);
  nda::array<double, 2> info(nk, 2);   // (npoles, heldout)
  info() = 0.0;
  for (long ik = rank; ik < nk; ik += np) {
    nda::array<ComplexType, 3> Sp(Sig_p(ik, all, all, all)), Sh(Sig_h(ik, all, all, all));
    auto sp = fit_sigma_sectors(bp, bh, zeta, Sp, Sh);
    nda::array<ComplexType, 2> H(Hrel(ik, all, all));
    auto ck   = closure_k(H, sp, p);
    e_loc[ik] = std::move(ck.e);
    v_loc[ik] = std::move(ck.v);
    info(ik, 0) = double(ck.d.size());
    info(ik, 1) = ck.heldout;
  }
  Timer.stop("closure_upfold");

  Timer.start("closure_gather");
  out.leh = gather_lehmann(comm, nk, nb, e_loc, v_loc);
  detail::exact_allreduce(comm, info.data(), info.size());
  for (long ik = 0; ik < nk; ++ik) {
    out.npoles.push_back(long(std::llround(info(ik, 0))));
    out.heldout.push_back(info(ik, 1));
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

  // 3. per owned k: compression; gather (fixed pole counts: the basis ranks)
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
