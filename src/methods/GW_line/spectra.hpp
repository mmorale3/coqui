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
 * k is distributed round-robin over comm, the [n_eta, nk, nw, nb] result is summed with one all_reduce (one nonzero
 * contribution per element). Collective over comm.
 */

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
};

struct spectra_out_t {
  nda::array<double, 1> omega;      ///< [nw] relative to mu
  nda::array<double, 1> eta;        ///< [n_eta]
  nda::array<double, 4> A_diag;     ///< [n_eta, nk, nw, nb]
  nda::array<double, 3> A_trace;    ///< [n_eta, nk, nw]
  double e_homo = 0.0, e_lumo = 0.0, N = 0.0;   ///< QP edges relative to mu, electron count at the chosen gap
  std::vector<long> npoles;
};

/**
 * Hrel (nk, nb, nb) = H0 + F - mu (final centre); Sig_p / Sig_h (nk, nz, nb, nb) at the nodes zeta about mu_sigma;
 * shift = mu - mu_sigma. Collective over comm. Sig_p / Sig_h may be k-distributed (nloc rows, see closure()).
 */
inline spectra_out_t line_spectra(boost::mpi3::communicator &comm, nda::array<ComplexType, 3> const &Hrel,
                                  nda::array<ComplexType, 4> const &Sig_p, nda::array<ComplexType, 4> const &Sig_h,
                                  nda::array<ComplexType, 1> const &zeta, double shift, line_basis_t const &bp,
                                  line_basis_t const &bh, closure_params_t const &p, double nelec,
                                  spectra_params_t const &sp) {
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

  std::vector<nda::array<double, 1>> e_loc(nk);
  std::vector<nda::array<ComplexType, 2>> v_loc(nk);
  nda::array<double, 1> npol(nk);
  npol() = 0.0;
  const bool sig_loc = (Sig_p.extent(0) != nk);   // k-distributed Sigma (S7e)
  for (long ik = rank; ik < nk; ik += np) {
    const long ks = sig_loc ? ik / np : ik;
    nda::array<ComplexType, 3> Sp(Sig_p(ks, all, all, all)), Sh(Sig_h(ks, all, all, all));
    auto spk = fit_sigma_sectors(bp, bh, zeta, Sp, Sh);
    spk.w -= shift;
    nda::array<ComplexType, 2> H(Hrel(ik, all, all));
    auto ck   = closure_k(H, spk, p);
    npol(ik)  = double(ck.d.size());
    for (long ie = 0; ie < ne; ++ie) {
      auto A = numerics::line_dlr::spectral_function(H, ck.d, ck.W, out.omega, out.eta(ie));
      for (long iw = 0; iw < nw; ++iw) {
        double tr = 0.0;
        for (long i = 0; i < nb; ++i) {
          out.A_diag(ie, ik, iw, i) = std::real(A(iw, i, i));
          tr += std::real(A(iw, i, i));
        }
        out.A_trace(ie, ik, iw) = tr;
      }
    }
    e_loc[ik] = std::move(ck.e);
    v_loc[ik] = std::move(ck.v);
  }
  detail::exact_allreduce(comm, out.A_diag.data(), out.A_diag.size());
  detail::exact_allreduce(comm, out.A_trace.data(), out.A_trace.size());
  detail::exact_allreduce(comm, npol.data(), npol.size());
  for (long ik = 0; ik < nk; ++ik) out.npoles.push_back(long(std::llround(npol(ik))));
  auto leh   = gather_lehmann(comm, nk, nb, e_loc, v_loc);
  auto cp    = numerics::line_dlr::chemical_potential(leh.e, leh.v, nelec);
  out.e_homo = cp.e_homo;
  out.e_lumo = cp.e_lumo;
  out.N      = cp.N;
  return out;
}

} // namespace methods::gw_line

#endif
