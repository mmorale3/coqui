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

#ifndef COQUI_METHODS_GW_LINE_NU0_HPP
#define COQUI_METHODS_GW_LINE_NU0_HPP

/**
 * D4 option nu0_intraband (notes section 12 "The nu_0 term"; python LineGW.nu0_dW / sigma_nu0): the Matsubara (isothermal)
 * static term of the EXACTLY degenerate pairs (|e_n(k) - e_m(k-q)| < deg_tol: intraband pairs at q = Gamma for every k, and the
 * symmetry-degenerate pairs e_n(k) = e_n(Sk) at k - q = Sk). The line works in the dynamic (Kubo, analytic-continuation) convention,
 * where these pairs carry no weight; the Matsubara code's Pi(q, i nu_0) = int_0^beta Pi(q, tau) carries beta f (1 - f) for them:
 *   dPi(q) = Pi^Mats(q, i nu_0) - Pi^an(q, 0)                                      (pi_tau_dpi, <= 0)
 *   dW(q)  = W[Pi^an(q, 0) + dPi] - W[Pi^an(q, 0)],  W[P] = (1 - Z P)^-1 Z         (= W^Mats dPi W^an, negative semidefinite)
 * It is a Kronecker-delta term at nu_0 only, i.e. W(q, tau) += dW(q) / beta, so in Sigma(k, i w_n) = -(1/beta) sum_m ...:
 *   Sigma_nu0(k, z) = -(1/(beta N_k)) sum_q X(k)^dag [G~(k - q, z) o dW(q)] X(k) = sum_{q, m} R_qm(k) / (z - e_m(k - q)),
 *   R_qm(k) = (1/(beta N_k)) sum_j T_j T_j^dag >= 0,  T_j = (X(k) D)^dag (a_m o l_j),  a_m = X(k - q) v_m,  dW = -L L^dag (L = U sqrt(lam))
 * analytic in z (poles at the poles of G, positive residues): it is carried on the line (Sp / Sh at the nodes, sector weights
 * (1 - f_m) / f_m as the head term) and in the hybrid tau leg (Stp(tau) = sum R (1 - f) e^{-e tau}, Sth(tau) = sum R f e^{e tau}).
 * IBZ: the q sum runs over the classes of ibz_t at the rotated k (ks, D), q_eff a row of R, exactly as self_energy_ibz.
 * The head (G = 0 of Z(Gamma)) is absent: div_treatment = "ignore_g0" only (the Matsubara reference uses the same THC Z).
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "mpi3/communicator.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/ibz.hpp"
#include "methods/GW_line/thermal.hpp"

namespace methods::gw_line {

/// factor L (Np x r) of -dW(q) = L L^dag per absolute q (empty: no degenerate pair at q), replicated on every rank
struct nu0_data_t {
  std::vector<nda::array<ComplexType, 2>> L;
  double max_dW = 0.0, anti_herm = 0.0, neg = 0.0;   ///< max|dW|, max|dW - dW^dag|, largest dropped negative eigenvalue (rel)
  long rank_sum = 0, nrows = 0;
  void reset(long nq) {
    L.assign(nq, nda::array<ComplexType, 2>());
    max_dW = anti_herm = neg = 0.0;
    rank_sum = nrows = 0;
  }
  void log(int lvl = 1) const {
    app_log(lvl, "  nu0_intraband: dW at {} q rows (total rank {}), max|dW| {:.3e}, max|dW - dW^+| {:.1e}, dropped negative part {:.1e} "
                 "(rel)",
            nrows, rank_sum, max_dW, anti_herm, neg);
  }
};

namespace detail {
/// full Np x Np matrix from this rank's block (all_reduce of the zero-padded blocks over comm)
inline nda::matrix<ComplexType> nu0_full(nda::array<ComplexType, 2> const &blk, aux_grid_t const &g, boost::mpi3::communicator &comm) {
  nda::matrix<ComplexType> M(g.Np, g.Np);
  M() = 0.0;
  for (long P = 0; P < g.nP; ++P)
    for (long Q = 0; Q < g.nQ; ++Q) M(g.P0 + P, g.Q0 + Q) = blk(P, Q);
  comm.all_reduce_in_place_n(M.data(), M.size(), std::plus<>{});
  return M;
}
} // namespace detail

/**
 * dW of the rows qs (a q group of the W step): Pi0 (g, nz, nP, nQ) the tau leg's DYNAMIC Pi(q, 0) at z index 0, dPi (g, nP, nQ)
 * its degenerate-pair term (pi_tau_leg dpi_out), Zb the Coulomb blocks. Collective over comm (same qs on every rank).
 */
template <MEMORY_SPACE MEM>
void nu0_dw_rows(memory::array<MEM, ComplexType, 4> const &Pi0, nda::array<ComplexType, 3> const &dPi, coulomb_blocks_t<MEM> const &Zb,
                 std::vector<long> const &qs, aux_grid_t const &grid, boost::mpi3::communicator &comm, nu0_data_t &out,
                 double eig_rel = 1e-14) {
  const long g = qs.size(), Np = grid.Np;
  nda::array<ComplexType, 4> P0h(memory::to_memory_space<HOST_MEMORY>(Pi0));
  nda::array<ComplexType, 3> Zh(memory::to_memory_space<HOST_MEMORY>(Zb.Z));
  auto all = nda::range::all;
  for (long r = 0; r < g; ++r) {
    const long q = qs[r];
    nda::array<ComplexType, 2> bd(dPi(r, all, all));
    auto Pd = detail::nu0_full(bd, grid, comm);
    double md = 0.0;
    for (auto const &x : Pd) md = std::max(md, std::abs(x));
    if (md == 0.0) {
      out.L[q] = nda::array<ComplexType, 2>();
      continue;
    }
    nda::array<ComplexType, 2> b0(P0h(r, 0, all, all)), bz(Zh(q, all, all));
    auto Pa = detail::nu0_full(b0, grid, comm);
    auto Zq = detail::nu0_full(bz, grid, comm);
    long rk = 0;
    nda::array<ComplexType, 2> Lq;
    if (comm.root()) {
      nda::matrix<ComplexType> I(Np, Np);
      I = 0.0;
      for (long P = 0; P < Np; ++P) I(P, P) = 1.0;
      nda::matrix<ComplexType> Pm = Pa + Pd;
      nda::matrix<ComplexType> Wa = nda::inverse(I - Zq * Pa) * Zq;
      nda::matrix<ComplexType> Wm = nda::inverse(I - Zq * Pm) * Zq;
      nda::matrix<ComplexType> dW = Wm - Wa;
      nda::matrix<ComplexType> H(Np, Np);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          H(P, Q)       = -0.5 * (dW(P, Q) + std::conj(dW(Q, P)));   // -dW, Hermitian part (>= 0)
          out.max_dW    = std::max(out.max_dW, std::abs(dW(P, Q)));
          out.anti_herm = std::max(out.anti_herm, std::abs(dW(P, Q) - std::conj(dW(Q, P))));
        }
      auto [lam, U] = nda::linalg::eigenelements(H);   // ascending
      const double lmax = std::max(std::abs(lam(0)), std::abs(lam(Np - 1)));
      if (lam(0) < 0.0) out.neg = std::max(out.neg, -lam(0) / lmax);
      std::vector<long> keep;
      for (long j = 0; j < Np; ++j)
        if (lam(j) > eig_rel * lmax) keep.push_back(j);
      rk = keep.size();
      Lq = nda::array<ComplexType, 2>(Np, rk);
      for (long c = 0; c < rk; ++c) {
        const double s = std::sqrt(lam(keep[c]));
        for (long P = 0; P < Np; ++P) Lq(P, c) = U(P, keep[c]) * s;
      }
    }
    comm.broadcast_n(&rk, 1, 0);
    if (not comm.root()) Lq = nda::array<ComplexType, 2>(Np, rk);
    if (rk > 0) comm.broadcast_n(Lq.data(), Lq.size(), 0);
    out.L[q] = std::move(Lq);
    out.rank_sum += rk;
    out.nrows += 1;
  }
  double st[3] = {out.max_dW, out.anti_herm, out.neg};
  comm.broadcast_n(st, 3, 0);
  out.max_dW = st[0];
  out.anti_herm = st[1];
  out.neg = st[2];
}

/**
 * Sigma_nu0 (file header) added at the line nodes zeta (Sp: (1 - f_m) part, Sh: f_m part) and, if tau != nullptr, at the hybrid
 * tau nodes (Stp(s) += sum R (1 - f) e^{-e s}, Sth(s) += sum R f e^{e s}, s in [0, beta / 2]). Rows: k_rows (IBZ k of the rows of
 * Sp / Sh / Stp / Sth). poles: the IBZ Lehmann poles (mu-relative, factorized). kdist = false: every rank holds all rows; the
 * work items (row, class, q') are split over comm and summed (all_reduce). Collective over comm.
 */
inline void nu0_sigma_add(pole_data_t const &poles, ibz_t const &ibz, mf::MF const &mf, methods::thc_reader_t const &thc,
                          nu0_data_t const &dw, double beta, std::vector<long> const &k_rows, bool kdist, boost::mpi3::communicator &comm,
                          nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 4> &Sp, nda::array<ComplexType, 4> &Sh,
                          nda::array<double, 1> const *tau = nullptr, nda::array<ComplexType, 4> *Stp = nullptr,
                          nda::array<ComplexType, 4> *Sth = nullptr) {
  const long nk = mf.nkpts(), nb = poles.nb, nz = zeta.size(), nrow = k_rows.size(), ns = tau ? tau->size() : 0;
  const long Np = thc.Np();
  auto qk  = mf.qk_to_k2();
  auto pf  = unfold_poles(poles, ibz);
  auto all = nda::range::all;
  nda::array<ComplexType, 3> X(nk, Np, nb);
  for (long k = 0; k < nk; ++k) X(k, all, all) = thc.X(0, 0, k);
  // the (merged) poles of every full-BZ k: energies, Fermi factors, A = X(k) V (Np x M)
  std::vector<std::vector<double>> E(nk);
  std::vector<nda::matrix<ComplexType>> A(nk);
  for (long k = 0; k < nk; ++k) {
    std::vector<long> S, C;
    detail::merged_poles(pf, k, E[k], S, C);
    const long M = E[k].size();
    nda::matrix<ComplexType> V(nb, M);
    for (long m = 0; m < M; ++m) {
      auto const &ps = S[m] == 0 ? pf.hole[k] : pf.part[k];
      V(all, m) = ps.v(all, C[m]);
    }
    A[k] = nda::matrix<ComplexType>(nda::matrix<ComplexType>(X(k, all, all)) * V);
  }
  const double s0 = 1.0 / (beta * double(nk));
  nda::array<ComplexType, 4> eSp(nrow, nz, nb, nb), eSh(nrow, nz, nb, nb), eTp(nrow, ns, nb, nb), eTh(nrow, ns, nb, nb);
  eSp() = 0.0; eSh() = 0.0; eTp() = 0.0; eTh() = 0.0;
  long item = 0;
  for (long l = 0; l < nrow; ++l) {
    const long k = k_rows[l];
    for (long c = 0; c < ibz.nclasses(); ++c) {
      auto const &cl = ibz.cls[c];
      const long ks = cl.ks[k];
      nda::matrix<ComplexType> Y(X(ks, all, all));
      if (not ibz.D[c].empty()) Y = nda::matrix<ComplexType>(Y * nda::matrix<ComplexType>(ibz.D[c][k]));
      for (long j = 0; j < long(cl.q_eff.size()); ++j) {
        const long qe = cl.q_eff[j];
        auto const &L = dw.L[qe];
        if (L.size() == 0) continue;
        if (not kdist and (item++ % comm.size()) != comm.rank()) continue;
        const long k2 = qk(qe, ks), M = E[k2].size(), r = L.extent(1);
        if (M == 0 or r == 0) continue;
        // R_m = s0 sum_j T_j(:, m) T_j(:, m)^dag, T_j = Y^dag (A o l_j)  ->  Rm (M, nb * nb)
        nda::matrix<ComplexType> Rm(M, nb * nb), AL(Np, M), T(nb, M);
        Rm = 0.0;
        for (long jj = 0; jj < r; ++jj) {
          for (long P = 0; P < Np; ++P)
            for (long m = 0; m < M; ++m) AL(P, m) = A[k2](P, m) * L(P, jj);
          T = nda::dagger(Y) * AL;
          for (long m = 0; m < M; ++m)
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b) Rm(m, a * nb + b) += s0 * T(a, m) * std::conj(T(b, m));
        }
        // weights at the nodes and the tau points, then one gemm per leg
        nda::matrix<ComplexType> Wp(nz, M), Wh(nz, M), Tp(std::max(ns, 1L), M), Th(std::max(ns, 1L), M);
        for (long m = 0; m < M; ++m) {
          const double e = E[k2][m], lp = -detail::log1pexp(-beta * e), lh = -detail::log1pexp(beta * e);   // log(1 - f), log f
          const double fp = std::exp(lp), fh = std::exp(lh);
          for (long iz = 0; iz < nz; ++iz) {
            const ComplexType g = 1.0 / (zeta(iz) - e);
            Wp(iz, m) = fp * g;
            Wh(iz, m) = fh * g;
          }
          for (long it = 0; it < ns; ++it) {
            Tp(it, m) = std::exp(lp - e * (*tau)(it));
            Th(it, m) = std::exp(lh + e * (*tau)(it));
          }
        }
        nda::matrix<ComplexType> Op = Wp * Rm, Oh = Wh * Rm;
        for (long iz = 0; iz < nz; ++iz)
          for (long a = 0; a < nb; ++a)
            for (long b = 0; b < nb; ++b) {
              eSp(l, iz, a, b) += Op(iz, a * nb + b);
              eSh(l, iz, a, b) += Oh(iz, a * nb + b);
            }
        if (ns > 0) {
          nda::matrix<ComplexType> Qp = Tp * Rm, Qh = Th * Rm;
          for (long it = 0; it < ns; ++it)
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b) {
                eTp(l, it, a, b) += Qp(it, a * nb + b);
                eTh(l, it, a, b) += Qh(it, a * nb + b);
              }
        }
      }
    }
  }
  if (not kdist) {
    for (auto *a : {&eSp, &eSh, &eTp, &eTh})
      if (a->size() > 0) comm.all_reduce_in_place_n(a->data(), a->size(), std::plus<>{});
  }
  Sp += eSp;
  Sh += eSh;
  if (ns > 0) {
    utils::check(Stp and Sth and Stp->extent(1) == ns, "gw_line::nu0_sigma_add: tau-node arrays");
    *Stp += eTp;
    *Sth += eTh;
  }
}

} // namespace methods::gw_line

#endif
