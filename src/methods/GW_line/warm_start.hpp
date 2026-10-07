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

#ifndef COQUI_METHODS_GW_LINE_WARM_START_HPP
#define COQUI_METHODS_GW_LINE_WARM_START_HPP

/**
 * Quasiparticle warm starts of the line scGW (plan 7.2; driver key `start`):
 *
 *   qp_diag_energies: the diagonal G0W0 QP equation in the KS band basis, per k and band,
 *       E_n = h_nn + Re Sigma_nn(E_n + i eta),   h = H0 + F - mu (KS basis), energies mu-relative,
 *     with Sigma_c(z) = W (z - d)^{-1} W^dagger the upfolded (Cayley-moment) representation of the closure of this k
 *     (closure_k: sector fits -> moments -> upfolding; positive weights |W_nm|^2, the representation the spectra use; the raw
 *     sector-fit residues are not usable on the real axis: they are determined on the line only). Newton from the KS energy (f' = Re Sigma'_nn - 1 <= -1 for positive residues; step <= 0.05 Ha,
 *     40 steps); if it does not converge (|f| > 1e-9 Ha) or ends > 0.1 Ha from the linearized solution
 *     E_lin = e + Z (h_nn + Re Sigma_nn(e + i eta) - e), Z = 1 / (1 - Re Sigma'_nn(e + i eta)), E_lin is used.
 *   read_qp_energies: QP energies (absolute, Ha, band order) from an h5 file (e.g. a CoQui Matsubara mbpt.h5).
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "methods/GW_line/closure.hpp"

namespace methods::gw_line {

struct qp_diag_out_t {
  nda::array<double, 2> e;   ///< (nk, nb) mu-relative QP energies
  long n_newton = 0, n_lin = 0;
  double max_shift = 0.0;    ///< max |E - e_KS| (Ha)
};

/**
 * eks (nk, nb): KS energies mu-relative; Hrel (nk, nb, nb) = H0 + F - mu; Sig_p / Sig_h: the rows of this rank (k_rows =
 * their global k; replicated rows: every rank computes all k, no reduction needed). Collective over comm.
 */
inline qp_diag_out_t qp_diag_energies(boost::mpi3::communicator &comm, nda::array<double, 2> const &eks,
                                      nda::array<ComplexType, 3> const &Hrel, nda::array<ComplexType, 4> const &Sig_p,
                                      nda::array<ComplexType, 4> const &Sig_h, std::vector<long> const &k_rows, bool distributed,
                                      nda::array<ComplexType, 1> const &zeta, line_basis_t const &bp, line_basis_t const &bh,
                                      closure_params_t const &cp, double eta) {
  auto all      = nda::range::all;
  const long nk = eks.extent(0), nb = eks.extent(1);
  qp_diag_out_t out;
  out.e = nda::array<double, 2>(nk, nb);
  out.e() = 0.0;
  nda::array<double, 1> cnt(2);
  cnt() = 0.0;
  for (long l = 0; l < long(k_rows.size()); ++l) {
    const long k = k_rows[l];
    if (not distributed and k % comm.size() != comm.rank()) continue;   // replicated: round-robin, then all_reduce
    nda::array<ComplexType, 3> Sp(Sig_p(l, all, all, all)), Sh(Sig_h(l, all, all, all));
    auto sp      = fit_sigma_sectors(bp, bh, zeta, Sp, Sh);
    nda::array<ComplexType, 2> Hk(Hrel(k, all, all));
    auto ck      = closure_k(Hk, sp, cp);
    const long r = ck.d.size();
    for (long n = 0; n < nb; ++n) {
      auto sig = [&](double E, double &dre) {   // Re Sigma_nn(E + i eta) and its E-derivative
        const ComplexType z(E, eta);
        ComplexType s = 0.0, ds = 0.0;
        for (long j = 0; j < r; ++j) {
          const ComplexType d = 1.0 / (z - ck.d(j));
          const double g      = std::norm(ck.W(n, j));
          s += g * d;
          ds -= g * d * d;
        }
        dre = std::real(ds);
        return std::real(s);
      };
      const double h = std::real(Hrel(k, n, n)), e0 = eks(k, n);
      double d0      = 0.0;
      const double s0 = sig(e0, d0);
      const double Z  = 1.0 / (1.0 - d0);
      const double El = e0 + Z * (h + s0 - e0);
      double E = e0;
      bool ok  = false;
      for (int it = 0; it < 40; ++it) {
        double d       = 0.0;
        const double f = h + sig(E, d) - E;
        if (std::abs(f) < 1e-9) {
          ok = true;
          break;
        }
        const double fp = d - 1.0;
        double step     = (std::abs(fp) > 1e-12) ? -f / fp : f;
        step            = std::clamp(step, -0.05, 0.05);
        E += step;
      }
      if (ok and std::abs(E - El) <= 0.1) {
        out.e(k, n) = E;
        cnt(0) += 1.0;
      } else {
        out.e(k, n) = El;
        cnt(1) += 1.0;
      }
    }
  }
  if (comm.size() > 1) {
    comm.all_reduce_in_place_n(out.e.data(), out.e.size(), std::plus<>{});
    comm.all_reduce_in_place_n(cnt.data(), cnt.size(), std::plus<>{});
  }
  out.n_newton = long(std::llround(cnt(0)));
  out.n_lin    = long(std::llround(cnt(1)));
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) out.max_shift = std::max(out.max_shift, std::abs(out.e(k, n) - eks(k, n)));
  return out;
}

/// QP energies (nk, nb) absolute (Ha) from `file`: dataset `ds`, or (ds empty) scf/iter<final_iter>/qp_approx/E_ska, else
/// E_ska / qp_energies at the root. Shapes (nk, nb') or (1, nk, nb') with nb' >= nb (the first nb bands are used).
/// Root reads, broadcast.
inline nda::array<double, 2> read_qp_energies(boost::mpi3::communicator &comm, std::string const &file, std::string ds, long nk,
                                              long nb) {
  nda::array<double, 2> E(nk, nb);
  std::string used;
  if (comm.root()) {
    utils::check(not file.empty(), "gw_line: start = \"qp_file\" needs start_file");
    h5::file f(file, 'r');
    h5::group g(f);
    if (ds.empty()) {
      if (g.has_subgroup("scf") and g.open_group("scf").has_dataset("final_iter")) {
        long fi = 0;
        h5::h5_read(g.open_group("scf"), "final_iter", fi);
        ds = "scf/iter" + std::to_string(fi) + "/qp_approx/E_ska";
      } else if (g.has_dataset("E_ska")) {
        ds = "E_ska";
      } else {
        ds = "qp_energies";
      }
    }
    used = ds;
    nda::array<double, 3> e3;
    nda::array<double, 2> e2;
    utils::check(g.has_dataset(ds), "gw_line: {} has no dataset {}", file, ds);
    const int rk = h5::array_interface::get_dataset_info(g, ds).rank();
    utils::check(rk == 2 or rk == 3, "gw_line: {}:{} has rank {}, expected (nk, nb) or (1, nk, nb)", file, ds, rk);
    if (rk == 3) {
      nda::h5_read(g, ds, e3);
      utils::check(e3.extent(0) == 1 and e3.extent(1) == nk and e3.extent(2) >= nb,
                   "gw_line: {}:{} has shape ({}, {}, {}), expected (1, {}, >= {})", file, ds, e3.extent(0), e3.extent(1),
                   e3.extent(2), nk, nb);
      for (long k = 0; k < nk; ++k)
        for (long n = 0; n < nb; ++n) E(k, n) = e3(0, k, n);
    } else {
      nda::h5_read(g, ds, e2);
      utils::check(e2.extent(0) == nk and e2.extent(1) >= nb, "gw_line: {}:{} has shape ({}, {}), expected ({}, >= {})", file, ds,
                   e2.extent(0), e2.extent(1), nk, nb);
      for (long k = 0; k < nk; ++k)
        for (long n = 0; n < nb; ++n) E(k, n) = e2(k, n);
    }
    app_log(1, "  start: QP energies read from {}:{} ({} k x {} bands used)", file, ds, nk, nb);
  }
  comm.broadcast_n(E.data(), E.size(), 0);
  return E;
}

} // namespace methods::gw_line

#endif
