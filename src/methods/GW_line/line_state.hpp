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

#ifndef COQUI_METHODS_GW_LINE_LINE_STATE_HPP
#define COQUI_METHODS_GW_LINE_LINE_STATE_HPP

/**
 * Pole data of the Green's function on the line (notes section 5.1, Eq. gtilde):
 *   G(k, zeta) = sum_m coef_m / (zeta - e_m),   e_m mu-relative (real), coef_m in C^{nb x nb},
 * sector of a pole = sign of e_m (particle '>' for e_m > 0, hole '<' for e_m < 0).
 * Python oracle: coqui/cayley/cayley/line/thc_gw.py (LineGW.set_poles, poles_from_hamiltonian).
 *
 * Storage (host, replicated on every rank): RAGGED per k and per sector, so the number of poles may differ per k and per
 * sector (padded nothing). Two forms of the residues (S7c):
 *   - matrix coefficients: coef (M, nb, nb), arbitrary (signed, non-Hermitian allowed): the gapless per-sector compression
 *     (g_repr = "compressed") and the python-parity path;
 *   - FACTORIZED (Lehmann) form: v (nb, M) with coef_m = v_m v_m^dagger (rank-1, positive), never materialized; produced by
 *     from_ks / from_hamiltonian / from_lehmann and by the closure with g_repr = "lehmann". Memory nb M instead of M nb^2.
 */

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "utilities/check.hpp"

namespace methods::gw_line {

using numerics::line_dlr::sector_t;

/// The poles of one sector at one k: either matrix coefficients coef (M, nb, nb) or the factorized form v (nb, M).
struct pole_sector_t {
  nda::array<double, 1> e;             ///< (M) mu-relative energies, all of one sign
  nda::array<ComplexType, 3> coef;     ///< (M, nb, nb) residues (matrix-coefficient form; empty when factorized)
  nda::array<ComplexType, 2> v;        ///< (nb, M) Lehmann vectors (factorized form: coef_m = v_m v_m^dagger)
  bool factorized = false;

  pole_sector_t() = default;
  /// matrix-coefficient form
  pole_sector_t(nda::array<double, 1> e_, nda::array<ComplexType, 3> coef_) : e(std::move(e_)), coef(std::move(coef_)) {
    utils::check(coef.extent(0) == e.size() and coef.extent(1) == coef.extent(2), "pole_sector_t: coef shape mismatch");
  }
  /// factorized form, v (nb, M)
  static pole_sector_t factorized_form(nda::array<double, 1> e_, nda::array<ComplexType, 2> v_) {
    utils::check(v_.extent(1) == e_.size(), "pole_sector_t: v has {} columns for {} poles", v_.extent(1), e_.size());
    pole_sector_t ps;
    ps.e          = std::move(e_);
    ps.v          = std::move(v_);
    ps.factorized = true;
    return ps;
  }

  long size() const { return e.size(); }
  bool is_factorized() const { return factorized; }
  long nb() const { return factorized ? v.extent(0) : coef.extent(1); }
  /// storage of the residues (bytes): 16 nb M (factorized) or 16 M nb^2
  double residue_bytes() const { return 16.0 * double(factorized ? v.size() : coef.size()); }
  /// weight of pole m: Tr coef_m (= |v_m|^2 when factorized)
  double weight(long m) const {
    double w = 0.0;
    if (factorized)
      for (long i = 0; i < v.extent(0); ++i) w += std::norm(v(i, m));
    else
      for (long i = 0; i < coef.extent(1); ++i) w += std::real(coef(m, i, i));
    return w;
  }
  /// sum_m coef_m (nb, nb): one gemm V V^dagger when factorized (the T = 0 density matrix of the hole sector)
  nda::matrix<ComplexType> density() const {
    const long n = nb();
    nda::matrix<ComplexType> D(n, n);
    D() = ComplexType(0.0);
    if (size() == 0) return D;
    if (factorized) {
      nda::matrix<ComplexType> V(v);
      nda::blas::gemm(ComplexType(1.0), V, nda::dagger(V), ComplexType(0.0), D);
    } else {
      for (long m = 0; m < size(); ++m) D += coef(m, nda::range::all, nda::range::all);
    }
    return D;
  }
  /// the matrix coefficients (M, nb, nb), materialized from v when factorized (tests / diagnostics only)
  nda::array<ComplexType, 3> coef_matrices() const {
    if (not factorized) return coef;
    const long M = size(), n = nb();
    nda::array<ComplexType, 3> c(M, n, n);
    for (long m = 0; m < M; ++m)
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) c(m, i, j) = v(i, m) * std::conj(v(j, m));
    return c;
  }
};

struct pole_data_t {
  long nk = 0, nb = 0;
  std::vector<pole_sector_t> part;     ///< [nk] poles with e > 0
  std::vector<pole_sector_t> hole;     ///< [nk] poles with e < 0

  static sector_t sector(double e) { return (e > 0.0) ? sector_t::particle : sector_t::hole; }

  pole_sector_t const &operator()(long ik, sector_t s) const {
    utils::check(s != sector_t::both, "pole_data_t: sector must be particle or hole");
    return (s == sector_t::particle) ? part[ik] : hole[ik];
  }

  /// true if every sector holds the factorized form (empty sectors count as either)
  bool is_factorized() const {
    for (long ik = 0; ik < nk; ++ik)
      for (auto const *ps : {&part[ik], &hole[ik]})
        if (ps->size() > 0 and not ps->is_factorized()) return false;
    return true;
  }
  /// total number of poles (both sectors, all k)
  long total_poles() const {
    long n = 0;
    for (long ik = 0; ik < nk; ++ik) n += part[ik].size() + hole[ik].size();
    return n;
  }
  /// storage of all residues (bytes, host; the propagator mirrors the same amount to its memory space)
  double residue_bytes() const {
    double b = 0.0;
    for (long ik = 0; ik < nk; ++ik) b += part[ik].residue_bytes() + hole[ik].residue_bytes();
    return b;
  }
  /// the same poles in the matrix-coefficient form (tests: factorized vs coefficient kernels)
  pole_data_t to_coefficients() const {
    pole_data_t pd;
    pd.nk = nk;
    pd.nb = nb;
    pd.part.resize(nk);
    pd.hole.resize(nk);
    for (long ik = 0; ik < nk; ++ik) {
      pd.part[ik] = pole_sector_t(part[ik].e, part[ik].coef_matrices());
      pd.hole[ik] = pole_sector_t(hole[ik].e, hole[ik].coef_matrices());
    }
    return pd;
  }

  /// Smallest |e_m| over all k and both sectors (sets s_max of the time rays).
  double emin() const {
    double em = std::numeric_limits<double>::max();
    bool has_p = false, has_h = false;
    for (long ik = 0; ik < nk; ++ik) {
      for (long m = 0; m < part[ik].size(); ++m) em = std::min(em, std::abs(part[ik].e(m)));
      for (long m = 0; m < hole[ik].size(); ++m) em = std::min(em, std::abs(hole[ik].e(m)));
      has_p = has_p or part[ik].size() > 0;
      has_h = has_h or hole[ik].size() > 0;
    }
    utils::check(has_p and has_h, "pole_data_t::emin: a sector is empty at every k");
    return em;
  }

  /// General constructor: e (nk, M), coef (nk, M, nb, nb) (same M for all k); poles are split by sign.
  static pole_data_t from_poles(nda::array<double, 2> const &e, nda::array<ComplexType, 4> const &coef) {
    utils::check(e.extent(0) == coef.extent(0) and e.extent(1) == coef.extent(1) and coef.extent(2) == coef.extent(3),
                 "pole_data_t::from_poles: shape mismatch");
    pole_data_t pd;
    pd.nk = e.extent(0);
    pd.nb = coef.extent(2);
    pd.part.resize(pd.nk);
    pd.hole.resize(pd.nk);
    for (long ik = 0; ik < pd.nk; ++ik) {
      std::vector<long> ip, ih;
      for (long m = 0; m < e.extent(1); ++m) {
        utils::check(e(ik, m) != 0.0, "pole_data_t: pole exactly at mu (k={}, m={})", ik, m);
        (e(ik, m) > 0.0 ? ip : ih).push_back(m);
      }
      auto fill = [&](pole_sector_t &ps, std::vector<long> const &idx) {
        ps.e    = nda::array<double, 1>(long(idx.size()));
        ps.coef = nda::array<ComplexType, 3>(long(idx.size()), pd.nb, pd.nb);
        for (long j = 0; j < long(idx.size()); ++j) {
          ps.e(j)                      = e(ik, idx[j]);
          ps.coef(j, nda::ellipsis{}) = coef(ik, idx[j], nda::ellipsis{});
        }
      };
      fill(pd.part[ik], ip);
      fill(pd.hole[ik], ih);
    }
    return pd;
  }

  /**
   * Factorized constructor from the Lehmann form per k: e[ik] (M_k), v[ik] (nb, M_k); poles are split by sign (a pole
   * exactly at e = 0 is an error), the order within a sector is the input order.
   */
  static pole_data_t from_lehmann(std::vector<nda::array<double, 1>> const &e, std::vector<nda::array<ComplexType, 2>> const &v) {
    utils::check(e.size() == v.size() and not e.empty(), "pole_data_t::from_lehmann: size mismatch");
    pole_data_t pd;
    pd.nk = long(e.size());
    pd.nb = v[0].extent(0);
    pd.part.resize(pd.nk);
    pd.hole.resize(pd.nk);
    for (long ik = 0; ik < pd.nk; ++ik) {
      utils::check(v[ik].extent(0) == pd.nb and v[ik].extent(1) == e[ik].size(), "pole_data_t::from_lehmann: v shape (k={})", ik);
      std::vector<long> ip, ih;
      for (long m = 0; m < e[ik].size(); ++m) {
        utils::check(e[ik](m) != 0.0, "pole_data_t: pole exactly at mu (k={}, m={})", ik, m);
        (e[ik](m) > 0.0 ? ip : ih).push_back(m);
      }
      auto fill = [&](std::vector<long> const &idx) {
        nda::array<double, 1> es(long(idx.size()));
        nda::array<ComplexType, 2> vs(pd.nb, long(idx.size()));
        for (long j = 0; j < long(idx.size()); ++j) {
          es(j) = e[ik](idx[j]);
          for (long i = 0; i < pd.nb; ++i) vs(i, j) = v[ik](i, idx[j]);
        }
        return pole_sector_t::factorized_form(std::move(es), std::move(vs));
      };
      pd.part[ik] = fill(ip);
      pd.hole[ik] = fill(ih);
    }
    return pd;
  }

  /// Lehmann poles of a Hermitian H(k) (nk, nb, nb): e_m - mu and v_m = the eigenvectors (factorized; python
  /// poles_from_hamiltonian, coef_m = v_m v_m^dagger).
  static pole_data_t from_hamiltonian(nda::array<ComplexType, 3> const &H, double mu) {
    const long nk = H.extent(0), nb = H.extent(1);
    std::vector<nda::array<double, 1>> e(nk);
    std::vector<nda::array<ComplexType, 2>> v(nk);
    for (long ik = 0; ik < nk; ++ik) {
      auto [ev, V] = nda::linalg::eigenelements(nda::matrix<ComplexType>(H(ik, nda::range::all, nda::range::all)));
      e[ik] = nda::array<double, 1>(nb);
      for (long m = 0; m < nb; ++m) e[ik](m) = ev(m) - mu;
      v[ik] = nda::array<ComplexType, 2>(V);
    }
    return from_lehmann(e, v);
  }

  /// Kohn-Sham poles in the KS band basis: e_m = eig(k, m) - mu, v_m = unit vector (factorized; coef_m = e_m e_m^T).
  static pole_data_t from_ks(nda::array<double, 2> const &eig, double mu) {
    const long nk = eig.extent(0), nb = eig.extent(1);
    std::vector<nda::array<double, 1>> e(nk);
    std::vector<nda::array<ComplexType, 2>> v(nk);
    for (long ik = 0; ik < nk; ++ik) {
      e[ik] = nda::array<double, 1>(nb);
      v[ik] = nda::array<ComplexType, 2>(nb, nb);
      v[ik]() = ComplexType(0.0);
      for (long m = 0; m < nb; ++m) {
        e[ik](m)    = eig(ik, m) - mu;
        v[ik](m, m) = ComplexType(1.0);
      }
    }
    return from_lehmann(e, v);
  }
};

} // namespace methods::gw_line

#endif
