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
 * Storage (host, replicated on every rank; N_k M nb^2 is small): RAGGED per k and per sector,
 *   part[ik] = {e (M_p(k)), coef (M_p(k), nb, nb)},  hole[ik] = {e (M_h(k)), coef (M_h(k), nb, nb)},
 * so the number of poles may differ per k and per sector (Lehmann form, compressed signed matrices, padded nothing).
 */

#include <cmath>
#include <limits>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "utilities/check.hpp"

namespace methods::gw_line {

using numerics::line_dlr::sector_t;

/// The poles of one sector at one k.
struct pole_sector_t {
  nda::array<double, 1> e;             ///< (M) mu-relative energies, all of one sign
  nda::array<ComplexType, 3> coef;     ///< (M, nb, nb) residues
  long size() const { return e.size(); }
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

  /// Lehmann poles of a Hermitian H(k) (nk, nb, nb): e_m - mu and coef_m = v_m v_m^dagger (python poles_from_hamiltonian).
  static pole_data_t from_hamiltonian(nda::array<ComplexType, 3> const &H, double mu) {
    const long nk = H.extent(0), nb = H.extent(1);
    nda::array<double, 2> e(nk, nb);
    nda::array<ComplexType, 4> coef(nk, nb, nb, nb);
    for (long ik = 0; ik < nk; ++ik) {
      auto [ev, V] = nda::linalg::eigenelements(nda::matrix<ComplexType>(H(ik, nda::range::all, nda::range::all)));
      for (long m = 0; m < nb; ++m) {
        e(ik, m) = ev(m) - mu;
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) coef(ik, m, i, j) = V(i, m) * std::conj(V(j, m));
      }
    }
    return from_poles(e, coef);
  }

  /// Kohn-Sham poles in the KS band basis: e_m = eig(k, m) - mu, coef_m = unit matrix e_m e_m^T.
  static pole_data_t from_ks(nda::array<double, 2> const &eig, double mu) {
    const long nk = eig.extent(0), nb = eig.extent(1);
    nda::array<double, 2> e(nk, nb);
    nda::array<ComplexType, 4> coef(nk, nb, nb, nb);
    coef() = ComplexType(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long m = 0; m < nb; ++m) {
        e(ik, m)          = eig(ik, m) - mu;
        coef(ik, m, m, m) = ComplexType(1.0);
      }
    return from_poles(e, coef);
  }
};

} // namespace methods::gw_line

#endif
