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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "utilities/check.hpp"
#include "cvv_head.hpp"

namespace methods {
namespace solvers {

  cvv_head_t::cvv_head_t(const imag_axes_ft::IAFT *ft, double rspace_tol)
      : _ft(ft), _rspace_tol(rspace_tol) {
    utils::check(ft != nullptr, "cvv_head_t: null IAFT pointer.");
    utils::check(rspace_tol > 0.0,
                 "cvv_head_t: cvv_rspace_tol must be > 0 (got {}).", rspace_tol);
  }

  nda::array<ComplexType, 4> cvv_head_t::velocity(long is, long ik) const {
    decltype(nda::range::all) all;
    utils::check(_built, "cvv_head_t::velocity: build() has not run.");
    utils::check(is >= 0 and is < _ns and ik >= 0 and ik < _kpts.shape(0),
                 "cvv_head_t::velocity: index out of range (is = {}, ik = {}).", is, ik);
    auto P = cvv_detail::phase_rows(_Rcart, _wR, _kpts(ik, all), false);   // (3, nR)
    nda::array<ComplexType, 4> v(3, _nw, _nb, _nb);

    // static part d_k(H0 + F): (3, nR) x (nR, nb^2), then broadcast over iw
    auto hs = _shstat.value().local()(is, all, all, all);                  // (nR, nb, nb)
    auto hs2 = nda::reshape(hs, std::array<long, 2>{_nR, _nb * _nb});
    nda::array<ComplexType, 2> vs(3, _nb * _nb);
    nda::blas::gemm(P, hs2, vs);
    for (long a = 0; a < 3; ++a) {
      auto va = nda::reshape(v(a, all, all, all), std::array<long, 2>{_nw, _nb * _nb});
      for (long iw = 0; iw < _nw; ++iw) va(iw, all) = vs(a, all);
    }

    // dynamic part d_k Sigma(k, iw): (3, nR) x (nR, nw*nb^2)
    if (_has_sigma) {
      auto sg = _ssig.value().local()(is, all, all, all, all);             // (nR, nw, nb, nb)
      auto sg2 = nda::reshape(sg, std::array<long, 2>{_nR, _nw * _nb * _nb});
      nda::array<ComplexType, 2> vd(3, _nw * _nb * _nb);
      nda::blas::gemm(P, sg2, vd);
      auto v2 = nda::reshape(v, std::array<long, 2>{3, _nw * _nb * _nb});
      v2 += vd;
    }
    return v;
  }

  void cvv_head_t::truncate_shells() {
    decltype(nda::range::all) all;
    // per-R squared Frobenius norm over the FULL stored object (rule 6: the truncation
    // is measured on the data, not guessed)
    std::vector<double> n2(_nR, 0.0);
    auto hs = _shstat.value().local();
    for (long is = 0; is < _ns; ++is)
      for (long iR = 0; iR < _nR; ++iR)
        for (long a = 0; a < _nb; ++a)
          for (long b = 0; b < _nb; ++b) n2[iR] += std::norm(hs(is, iR, a, b));
    if (_has_sigma) {
      auto sg = _ssig.value().local();
      for (long is = 0; is < _ns; ++is)
        for (long iR = 0; iR < _nR; ++iR)
          for (long iw = 0; iw < _nw; ++iw)
            for (long a = 0; a < _nb; ++a)
              for (long b = 0; b < _nb; ++b) n2[iR] += std::norm(sg(is, iR, iw, a, b));
    }

    // group R points into |R| shells (1e-8 radius bins), outermost first
    std::vector<double> rad(_nR);
    for (long iR = 0; iR < _nR; ++iR)
      rad[iR] = std::sqrt(_Rcart(iR, 0) * _Rcart(iR, 0) + _Rcart(iR, 1) * _Rcart(iR, 1) +
                          _Rcart(iR, 2) * _Rcart(iR, 2));
    std::vector<long> order(_nR);
    for (long i = 0; i < _nR; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](long a, long b) { return rad[a] > rad[b]; });

    const double tot2 = std::accumulate(n2.begin(), n2.end(), 0.0);
    // walk inward shell by shell; a shell is dropped only if the CUMULATIVE dropped
    // norm (it plus everything outside it) stays within tol * total
    std::vector<bool> drop(_nR, false);
    double drop2 = 0.0;
    long ndrop = 0;
    {
      long i = 0;
      while (i < _nR) {
        long j = i;
        double shell2 = 0.0;
        while (j < _nR and std::abs(rad[order[j]] - rad[order[i]]) < 1e-8) {
          shell2 += n2[order[j]];
          ++j;
        }
        if (rad[order[i]] < 1e-12) break;   // never drop the R = 0 shell
        if (std::sqrt((drop2 + shell2) / std::max(tot2, 1e-300)) > _rspace_tol) break;
        drop2 += shell2;
        for (long m = i; m < j; ++m) { drop[order[m]] = true; ++ndrop; }
        i = j;
      }
    }

    // zero the dropped rows in the store (|R| shells are inversion-symmetric sets, so
    // the truncated interpolant keeps v~(k)^dag = v~(k) exactly)
    for (long iR = 0; iR < _nR; ++iR) {
      if (not drop[iR]) continue;
      for (long is = 0; is < _ns; ++is) hs(is, iR, all, all) = ComplexType(0.0);
      if (_has_sigma) {
        auto sg = _ssig.value().local();
        for (long is = 0; is < _ns; ++is) sg(is, iR, all, all, all) = ComplexType(0.0);
      }
    }
    _nR_kept = _nR - ndrop;
    app_log(2, "  [CVV] R-space store: {} / {} WS points kept (cvv_rspace_tol = {}); "
               "dropped norm fraction = {:.3e}",
            _nR_kept, _nR, _rspace_tol, std::sqrt(drop2 / std::max(tot2, 1e-300)));
  }

} // solvers
} // methods
