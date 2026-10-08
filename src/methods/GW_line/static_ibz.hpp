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

#ifndef COQUI_METHODS_GW_LINE_STATIC_IBZ_HPP
#define COQUI_METHODS_GW_LINE_STATIC_IBZ_HPP

/**
 * F = V_H + Sigma_x at the IBZ k (perf 7.3; static_part.hpp, ibz.hpp item (S)): D(k_I) (nk_ibz, nb, nb) unfolded to every
 * full-BZ k (conj for time-reversed k'), D~(k') = X(k') D(k') X(k')^dagger, rho from all k', V_H at the IBZ k, and
 *   Sigma_x(k) = -(1/N_k) sum_isym (X(ks) D)^dagger [ sum_{q' in class} D~(ks - q_eff) o Z(q_eff) ] (X(ks) D),
 * the class sum of thc_hf.icc (exact against CoQui's hf_t on symmetric fixtures, [ibz][V0]). Z: the Coulomb blocks of the
 * rows R (coulomb_blocks_t, virtual rows conj Z(q)). F (nk_ibz, nb, nb) on the host, identical on every rank.
 */

#include <array>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "mean_field/MF.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/ibz.hpp"

namespace methods::gw_line {

template <MEMORY_SPACE MEM>
void hartree_exchange_ibz(propagator_t<MEM> &prop, coulomb_blocks_t<MEM> const &Zb, nda::array<ComplexType, 3> const &DI,
                          mf::MF const &mf, ibz_t const &ibz, aux_grid_t const &grid, boost::mpi3::communicator &comm,
                          nda::array<ComplexType, 3> &F, utils::TimerManager &Timer) {
  using arr3_t = memory::array<MEM, ComplexType, 3>;
  using arr2_t = memory::array<MEM, ComplexType, 2>;
  auto all     = nda::range::all;
  const long nk = prop.nk, nb = prop.nb, nP = grid.nP, nQ = grid.nQ, nkI = ibz.nkI;
  utils::check(DI.extent(0) == nkI and DI.extent(1) == nb, "gw_line::hartree_exchange_ibz: D ({}, {}) vs (nk_ibz {}, nb {})",
               DI.extent(0), DI.extent(1), nkI, nb);
  utils::check(grid.np == comm.size() or comm.size() == 1, "gw_line::hartree_exchange_ibz: grid / communicator mismatch");
  auto qk        = mf.qk_to_k2();
  const long iq0 = gamma_index(mf);
  for (auto nm : {"HX_density", "HX_hartree", "HX_exchange", "HX_allreduce"}) Timer.add(nm);

  auto Dfull = unfold_matrices(DI, ibz);
  arr3_t Dm  = memory::to_memory_space<MEM>(Dfull);
  arr3_t Dt(nk, nP, nQ), Fp(nkI, nb, nb);
  arr2_t tPb(nP, nb), tbQ(nb, nQ), rho(1, nQ), vP(nP, 1), Vb(nP, nb), vX(nP, nb), acc(nP, nQ), tmp(nP, nQ), ones_b(1, nb);
  {
    nda::array<ComplexType, 2> o(1, nb);
    o()    = ComplexType(1.0);
    ones_b = memory::to_memory_space<MEM>(o);
  }
  Timer.start("HX_density");
  nda::tensor::set(ComplexType(0.0), rho);
  for (long ik = 0; ik < nk; ++ik) {
    nda::blas::gemm(ComplexType(1.0), prop.Xp(ik, all, all), Dm(ik, all, all), ComplexType(0.0), tPb);
    nda::blas::gemm(ComplexType(1.0), tPb, prop.XqH(ik, all, all), ComplexType(0.0), Dt(ik, all, all));
    nda::blas::gemm(ComplexType(1.0), nda::transpose(Dm(ik, all, all)), prop.XqT(ik, all, all), ComplexType(0.0), tbQ);
    nda::tensor::elementwise(ComplexType(1.0), prop.XqH(ik, all, all), ComplexType(1.0), tbQ, nda::tensor::op::MUL);
    nda::blas::gemm(ComplexType(2.0 / double(nk)), ones_b, tbQ, ComplexType(1.0), rho);
  }
  Timer.stop("HX_density");

  Timer.start("HX_hartree");
  nda::blas::gemm(ComplexType(1.0), Zb.Z(iq0, all, all), nda::transpose(rho), ComplexType(0.0), vP);
  nda::blas::gemm(ComplexType(1.0), vP, ones_b, ComplexType(0.0), Vb);
  for (long ik = 0; ik < nkI; ++ik) {   // the IBZ k are the first nk_ibz of the full list
    vX = prop.Xp(ik, all, all);
    nda::tensor::elementwise(ComplexType(1.0), Vb, ComplexType(1.0), vX, nda::tensor::op::MUL);
    nda::blas::gemm(ComplexType(1.0), nda::transpose(prop.Xpc(ik, all, all)), vX, ComplexType(0.0), Fp(ik, all, all));
  }
  Timer.stop("HX_hartree");

  Timer.start("HX_exchange");
  nda::array<ComplexType, 3> xp_h = memory::to_memory_space<HOST_MEMORY>(prop.Xp), xqt_h = memory::to_memory_space<HOST_MEMORY>(prop.XqT);
  for (long c = 0; c < ibz.nclasses(); ++c) {
    auto const &cl = ibz.cls[c];
    for (long ik = 0; ik < nkI; ++ik) {
      const long ks = cl.ks[ik];
      for (long j = 0; j < long(cl.q_eff.size()); ++j) {
        const long qe = cl.q_eff[j];
        tmp           = Dt(qk(qe, ks), all, all);
        nda::tensor::elementwise(ComplexType(1.0), Zb.Z(qe, all, all), ComplexType(1.0), tmp, nda::tensor::op::MUL);
        if (j == 0) acc = tmp;
        else nda::tensor::elementwise(ComplexType(1.0), tmp, ComplexType(1.0), acc, nda::tensor::op::SUM);
      }
      // L = (Xp(ks) D)^dagger (nb, nP), R = Xq(ks) D (nQ, nb)
      nda::array<ComplexType, 2> XDp(nP, nb), XDq(nQ, nb);
      nda::array<ComplexType, 2> xq = nda::transpose(xqt_h(ks, all, all));
      if (ibz.D[c].empty()) {
        XDp = xp_h(ks, all, all);
        XDq = xq;
      } else {
        nda::blas::gemm(ComplexType(1.0), xp_h(ks, all, all), ibz.D[c][ik], ComplexType(0.0), XDp);
        nda::blas::gemm(ComplexType(1.0), xq, ibz.D[c][ik], ComplexType(0.0), XDq);
      }
      nda::array<ComplexType, 2> Lh = nda::dagger(XDp);
      arr2_t L = memory::to_memory_space<MEM>(Lh), R = memory::to_memory_space<MEM>(XDq);
      nda::blas::gemm(ComplexType(1.0), L, acc, ComplexType(0.0), tbQ);
      nda::blas::gemm(ComplexType(-1.0 / double(nk)), tbQ, R, ComplexType(1.0), Fp(ik, all, all));
    }
  }
  Timer.stop("HX_exchange");

  Timer.start("HX_allreduce");
  F = memory::to_memory_space<HOST_MEMORY>(Fp);
  comm.all_reduce_in_place_n(F.data(), F.size(), std::plus<>{});
  Timer.stop("HX_allreduce");
}

} // namespace methods::gw_line

#endif
