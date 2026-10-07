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

#ifndef COQUI_METHODS_GW_LINE_KMESH_FT_HPP
#define COQUI_METHODS_GW_LINE_KMESH_FT_HPP

/**
 * Fourier matrices of the k mesh for the real-space convolutions of the line GW (perf 7.1 (e), plan section 7.1):
 *
 *   Pi^>(q,t)     = (2/N) sum_k A(k) o B(k-q)        = (2/N^2) sum_R e^{-iQ_q R} A^(R) o B^(R),
 *                   A^(R) = sum_k e^{+ikR} A(k),  B^(R) = sum_k e^{-ikR} B(k)
 *   acc^>(k,t)    = sum_q G(k-q) o W(q)              = (1/N) sum_R e^{+ikR} G^(R) o W^(R),
 *                   G^(R) = sum_k e^{-ikR} G(k),  W^(R) = sum_q e^{-iQ_q R} W(q)
 *   acc^<(k,t)    = sum_q' G(k+q') o V(q')           = (1/N) sum_R e^{+ikR} G^(R) o V^(R),
 *                   V^(R) = sum_q e^{+iQ_q R} V(q) = [the e^{-iQR} transform of V](-R)
 * where k - q is the mesh index qk_to_k2(q, k) and k + q' = qk_to_k2(qminus(q'), k) (the index maps of the k-space kernels).
 * R runs over the N = n1 n2 n3 lattice vectors n_R = (a, b, c), 0 <= a < n1, ... (any representatives: e^{iGR} = 1);
 * phases from the fractional k (kpts_crystal): e^{ikR} = e^{2 pi i k_crys . n_R}. The transfer momenta are taken from the
 * index map itself, e^{iQ_q R} = e^{i (k_0 - k_{qk(q,0)}) R}, so no Q coordinates (and no G-shift convention of Qpts) enter:
 * the identity holds exactly whenever qk_to_k2 is a group translation of the mesh, which the constructor verifies,
 *   closure_err = max_{q,k,R} |e^{i (k_{qk(q,k)} - k + Q_q) R} - 1|,   unitarity |F F^dagger / N - 1|,
 * together with the R -> -R map (minusR, |e^{iQ(-R)} - conj e^{iQR}|). ok = false (and the kernels keep the k-space path)
 * when a check fails or the mesh is not n1 x n2 x n3 = N_k.
 *
 * Cost per (t, block element): 3 gemm passes of N x N (k -> R, k -> R, R -> q) against N_q N_k products of the k sum:
 * compute-bound zgemm instead of memory-bound Hadamard sums.
 */

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "mean_field/MF.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods::gw_line {

struct kmesh_ft_t {
  bool ok = false;
  long N = 0;
  std::array<long, 3> mesh = {0, 0, 0};
  nda::array<ComplexType, 2> Fp;   ///< (R, k) e^{+ik.R}
  nda::array<ComplexType, 2> Fm;   ///< (R, k) e^{-ik.R}
  nda::array<ComplexType, 2> Bp;   ///< (k, R) e^{+ik.R}        (back transform R -> k, without 1/N)
  nda::array<ComplexType, 2> Hm;   ///< (R, q) e^{-iQ_q.R}      (q -> R)
  nda::array<ComplexType, 2> Gm;   ///< (q, R) e^{-iQ_q.R}      (back transform R -> q, without 1/N)
  std::vector<long> minusR;        ///< index of -R (mod the mesh)
  double closure_err = 0.0, unitary_err = 0.0, minusR_err = 0.0;

  kmesh_ft_t() = default;

  explicit kmesh_ft_t(mf::MF const &mf) {
    const long nk = mf.nkpts(), nq = mf.nqpts();
    N             = nk;
    auto kg       = mf.kp_grid();
    for (int i = 0; i < 3; ++i) mesh[i] = long(kg(i));
    if (nq != nk or mesh[0] * mesh[1] * mesh[2] != nk or nk < 1) {
      app_log(2, "  kmesh_ft_t: mesh {}x{}x{} vs N_k {} N_q {}: real-space convolutions disabled", mesh[0], mesh[1], mesh[2], nk,
              nq);
      return;
    }
    auto kc = mf.kpts_crystal();   // (nk, 3) fractional coordinates
    auto qk = mf.qk_to_k2();
    auto qm = mf.qminus();
    std::vector<std::array<long, 3>> R(N);
    for (long a = 0, i = 0; a < mesh[0]; ++a)
      for (long b = 0; b < mesh[1]; ++b)
        for (long c = 0; c < mesh[2]; ++c, ++i) R[i] = {a, b, c};
    const double tp = 2.0 * std::numbers::pi;
    auto phase      = [&](long k, long r) {   // e^{+i k.R}
      double x = 0.0;
      for (int d = 0; d < 3; ++d) x += double(kc(k, d)) * double(R[r][d]);
      x -= std::floor(x);   // reduce the argument before the exponential (exactness for large meshes)
      return std::exp(ComplexType(0.0, tp * x));
    };
    Fp = nda::array<ComplexType, 2>(N, N);
    for (long r = 0; r < N; ++r)
      for (long k = 0; k < N; ++k) Fp(r, k) = phase(k, r);
    Fm = nda::conj(Fp);
    Bp = nda::transpose(Fp);
    // e^{iQ_q R} = e^{i (k_0 - k_{qk(q,0)}) R}
    Hm = nda::array<ComplexType, 2>(N, nq);
    for (long r = 0; r < N; ++r)
      for (long q = 0; q < nq; ++q) Hm(r, q) = std::conj(Fp(r, 0) * std::conj(Fp(r, qk(q, 0))));
    Gm = nda::transpose(Hm);
    // checks: the q map is a mesh translation; unitarity; R -> -R
    closure_err = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long k = 0; k < nk; ++k)
        for (long r = 0; r < N; ++r) {
          // e^{i (k_{qk(q,k)} - k + Q_q) R} = Fp(r, qk) conj(Fp(r, k)) conj(Hm(r, q))
          const ComplexType v = Fp(r, qk(q, k)) * std::conj(Fp(r, k)) * std::conj(Hm(r, q));
          closure_err         = std::max(closure_err, std::abs(v - 1.0));
        }
    for (long q = 0; q < nq; ++q)   // Q_{-q} = -Q_q (mod G)
      for (long r = 0; r < N; ++r) closure_err = std::max(closure_err, std::abs(Hm(r, qm(q)) - std::conj(Hm(r, q))));
    unitary_err = 0.0;
    for (long a = 0; a < N; ++a)
      for (long b = 0; b < N; ++b) {
        ComplexType s = 0.0;
        for (long k = 0; k < N; ++k) s += Fp(a, k) * std::conj(Fp(b, k));
        unitary_err = std::max(unitary_err, std::abs(s / double(N) - (a == b ? 1.0 : 0.0)));
      }
    minusR.assign(N, -1);
    for (long r = 0; r < N; ++r) {
      std::array<long, 3> m = {(mesh[0] - R[r][0]) % mesh[0], (mesh[1] - R[r][1]) % mesh[1], (mesh[2] - R[r][2]) % mesh[2]};
      minusR[r]             = (m[0] * mesh[1] + m[1]) * mesh[2] + m[2];
    }
    minusR_err = 0.0;
    for (long r = 0; r < N; ++r)
      for (long q = 0; q < nq; ++q) minusR_err = std::max(minusR_err, std::abs(Hm(minusR[r], q) - std::conj(Hm(r, q))));
    ok = (closure_err < 1e-10 and unitary_err < 1e-12 and minusR_err < 1e-10);
    app_log(ok ? 3 : 1, "  kmesh_ft_t: mesh {}x{}x{}, N {}: closure {:.1e}, unitarity {:.1e}, -R map {:.1e}{}", mesh[0], mesh[1],
            mesh[2], N, closure_err, unitary_err, minusR_err, ok ? "" : " -> real-space convolutions DISABLED");
  }
};

} // namespace methods::gw_line

#endif
