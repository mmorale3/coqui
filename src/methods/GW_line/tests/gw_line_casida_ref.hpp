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

#ifndef COQUI_METHODS_GW_LINE_TESTS_CASIDA_REF_HPP
#define COQUI_METHODS_GW_LINE_TESTS_CASIDA_REF_HPP

/**
 * Test-only helpers shared by the GW_line kernel tests ([V2] test_gw_line_w.cpp, [V3] test_gw_line_sigma.cpp):
 * the exact Casida (full RPA) solution of the THC problem with KS poles (python oracle si_pipeline.py::casida_q,
 * Hermitian form), its evaluation, block gathering and max-norm helpers.
 *
 * Casida: S_Pt = sqrt(2/N_k) X_Pn(k) conj(X_Pm(k-q)), E_t = e_m(k-q) - e_n(k), s_t = +1 (n occupied) / -1,
 * K = S^dagger Z S, M = diag(|E|) + K > 0, C = M^{1/2} eta M^{1/2} = U diag(lambda) U^dagger, v = Z S M^{-1/2} U:
 *   W_dyn(q, z) = Z Pi_RPA(z) Z = sum_s lambda_s v_s v_s^dagger / (z - lambda_s).
 */

#include <cmath>
#include <vector>

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/proc_grid.hpp"

namespace gw_line_test {

using methods::gw_line::aux_grid_t;
namespace mpi3 = boost::mpi3;
using cmat = nda::matrix<ComplexType>;

inline double max_abs3(auto const &a) {
  double d = 0.0;
  for (auto const &v : a) d = std::max(d, std::abs(v));
  return d;
}
inline double max_diff3(auto const &a, auto const &b) {
  auto a1 = nda::reshape(a, std::array<long, 1>{a.size()});
  auto b1 = nda::reshape(b, std::array<long, 1>{b.size()});
  double d = 0.0;
  for (long i = 0; i < a1.size(); ++i) d = std::max(d, std::abs(a1(i) - b1(i)));
  return d;
}

/// full [n, Np, Np] from every rank's [n, nP, nQ] block (zero-padded all_reduce; small fixtures)
inline nda::array<ComplexType, 3> gather_full(mpi3::communicator &comm, aux_grid_t const &g, auto const &blk) {
  const long n = blk.extent(0);
  nda::array<ComplexType, 3> F(n, g.Np, g.Np);
  F() = 0.0;
  F(nda::range::all, g.P_rng(), g.Q_rng()) = blk;
  comm.all_reduce_in_place_n(F.data(), F.size(), std::plus<>{});
  return F;
}

/// Casida W_dyn(q, z) = sum_s lam_s V_s V_s^dagger / (z - lam_s)  (see the file header)
struct casida_t {
  nda::array<double, 1> lam;
  cmat V;    // (Np, T)
  cmat S;    // (Np, T) transition columns
  std::vector<double> E, sg;
  long T = 0, npos = 0;
  double min_M = 0.0;
};

inline casida_t casida_q(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel, long iq, cmat const &Z) {
  const long nk = mf.nkpts(), nb = e_rel.extent(1), Np = Z.extent(0);
  auto qk = mf.qk_to_k2();
  std::vector<long> tk, tn, tm;
  std::vector<double> E, sg;
  for (long ik = 0; ik < nk; ++ik) {
    const long ikmq = qk(iq, ik);
    for (long n = 0; n < nb; ++n)
      for (long m = 0; m < nb; ++m) {
        const bool occ_n = e_rel(ik, n) < 0.0, occ_m = e_rel(ikmq, m) < 0.0;
        if (occ_n == occ_m) continue;
        tk.push_back(ik); tn.push_back(n); tm.push_back(m);
        E.push_back(e_rel(ikmq, m) - e_rel(ik, n));
        sg.push_back(occ_n ? 1.0 : -1.0);
      }
  }
  casida_t c;
  const long T = E.size();
  c.T = T;
  const double nrm = std::sqrt(2.0 / double(nk));
  cmat S(Np, T);
  for (long t = 0; t < T; ++t) {
    auto Xk = thc.X(0, 0, tk[t]);
    auto Xm = thc.X(0, 0, qk(iq, tk[t]));
    for (long P = 0; P < Np; ++P) S(P, t) = nrm * Xk(P, tn[t]) * std::conj(Xm(P, tm[t]));
  }
  cmat ZS = Z * S;
  cmat M  = nda::dagger(S) * ZS;
  for (long t = 0; t < T; ++t) M(t, t) += std::abs(E[t]);
  cmat Mh = 0.5 * (M + nda::dagger(M));
  auto [mv, Vm] = nda::linalg::eigenelements(Mh);
  c.min_M = mv(0);
  utils::check(mv(0) > 0.0, "casida_q: M = |E| + S^dag Z S is not positive definite (min eig {})", mv(0));
  cmat Vs(T, T), Vi(T, T);
  for (long t = 0; t < T; ++t)
    for (long u = 0; u < T; ++u) {
      Vs(t, u) = Vm(t, u) * std::sqrt(mv(u));
      Vi(t, u) = Vm(t, u) / std::sqrt(mv(u));
    }
  cmat Msq  = Vs * nda::dagger(Vm);   // M^{1/2}
  cmat Mmsq = Vi * nda::dagger(Vm);   // M^{-1/2}
  cmat eMsq(T, T);
  for (long t = 0; t < T; ++t)
    for (long u = 0; u < T; ++u) eMsq(t, u) = sg[t] * Msq(t, u);
  cmat C  = Msq * eMsq;
  cmat Ch = 0.5 * (C + nda::dagger(C));
  auto [lam, U] = nda::linalg::eigenelements(Ch);
  c.lam = lam;
  cmat MU = Mmsq * U;
  c.V     = ZS * MU;
  for (long s = 0; s < T; ++s) c.npos += (lam(s) > 0.0) ? 1 : 0;
  c.S  = S;
  c.E  = E;
  c.sg = sg;
  return c;
}

/// block (P_rng, Q_rng) of the Casida sum at z, restricted to lam > 0 (particle) or all poles
inline nda::array<ComplexType, 3> casida_eval(casida_t const &c, aux_grid_t const &g, nda::array<ComplexType, 1> const &z,
                                       bool particle_only) {
  const long nz = z.size(), T = c.T;
  nda::array<ComplexType, 3> out(nz, g.nP, g.nQ);
  cmat A(g.nP, T);
  cmat B(c.V(g.Q_rng(), nda::range::all));
  for (long iz = 0; iz < nz; ++iz) {
    for (long s = 0; s < T; ++s) {
      const ComplexType f = (particle_only and c.lam(s) <= 0.0) ? ComplexType(0.0) : c.lam(s) / (z(iz) - c.lam(s));
      for (long P = 0; P < g.nP; ++P) A(P, s) = c.V(g.P0 + P, s) * f;
    }
    nda::matrix_view<ComplexType> o(out(iz, nda::range::all, nda::range::all));
    nda::blas::gemm(ComplexType(1.0), A, nda::dagger(B), ComplexType(0.0), o);
  }
  return out;
}

} // namespace gw_line_test

#endif
