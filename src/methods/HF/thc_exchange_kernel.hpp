#ifndef COQUI_THC_EXCHANGE_KERNEL_HPP
#define COQUI_THC_EXCHANGE_KERNEL_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/HF/thc_solver_comm.hpp"

namespace methods {
namespace solvers {
namespace lff_sigma_detail {

template<int N> using shape_t = std::array<long, N>;

/**
 * LFF-Sigma (Route 1, notes/lff_aux_plan.md 2026-09-19): the THC EXCHANGE contraction of hf_t::thc_hf_Xqindep with a
 * CUSTOM static kernel U(q) in place of the bare Coulomb Z -- the instantaneous part of the vertex-corrected screened
 * interaction, dW~(i nu -> inf) = Herm[Z t^dag G1(inf) t], which is a delta(tau) the bosonic tau machinery cannot carry
 * and therefore enters the STATIC (Fock-like) self-energy:  F_out(k) = - sum_q  X^dag [ Dm(k - q) o U(q) ] X  (the K
 * term only: no Hartree, no q -> 0 head correction -- the head of the static piece is not included, see the caller).
 *   Dm_skij : (ns, nk_ibz, nb, nb) density matrix (mb_state.sDm_skij.local())
 *   dU_qPQ  : distributed (nq, Np, Np), grid {1, np_P, np_Q} with np_P np_Q = comm.size() (default blocks): the kernel;
 *             OVERWRITTEN in place by its R-space transform (a scratch object)
 *   sF_out  : (ns, nk_ibz, nb, nb) node-shared output, zeroed here
 * Restrictions as thc_hf_Xqindep: q-independent interpolating points, no symmetry reduction (nq_ibz == nq).
 */
template<typename Array_primary_t, typename dArray_t, typename sArray_t>
void exchange_with_kernel(const Array_primary_t &Dm_skij, dArray_t &dU_qPQ, sArray_t &sF_skij, THC_ERI auto &thc) {
  using local_Array_4D_t = memory::array<HOST_MEMORY, ComplexType, 4>;
  using math::nda::make_distributed_array;
  auto MF = thc.MF();
  auto mpi = thc.mpi();
  utils::check(thc.thc_X_type() == "q_indep" and MF->nqpts_ibz() == MF->nqpts(),
               "lff_sigma_detail::exchange_with_kernel: only the q-independent, symmetry-free (nq_ibz == nq) THC path is "
               "implemented (the static LFF-Sigma piece on a symmetric mesh is not).");
  const long NP = thc.Np(), ns = Dm_skij.extent(0), npol = MF->npol(), nkpts = MF->nkpts(), nkpts_ibz = MF->nkpts_ibz();
  auto grd = dU_qPQ.grid();
  const int np_P = int(grd[1]), np_Q = int(grd[2]);
  utils::check(grd[0] == 1 and long(np_P) * long(np_Q) == long(mpi->comm.size()),
               "lff_sigma_detail::exchange_with_kernel: the kernel grid must be {{1, np_P, np_Q}} with np_P np_Q = comm size.");
  nda::array<long, 1> R_grid = MF->kp_grid();
  math::shm::shared_array<nda::array_view<ComplexType, 2>> sf_Rk(*mpi, {nkpts, nkpts});
  nda::matrix<ComplexType> buffer;
  auto dDm_skPQ = make_distributed_array<local_Array_4D_t>(mpi->comm, {1, 1, np_P, np_Q}, {ns, nkpts, NP, NP});
  auto dF_skPQ = make_distributed_array<local_Array_4D_t>(mpi->comm, {1, 1, np_P, np_Q}, {ns, nkpts_ibz, NP, NP});
  const long NP_loc = dDm_skPQ.local_shape()[2], NQ_loc = dDm_skPQ.local_shape()[3];
  utils::check(dU_qPQ.local_shape()[1] == NP_loc and dU_qPQ.local_shape()[2] == NQ_loc
               and dU_qPQ.origin()[1] == dDm_skPQ.origin()[2] and dU_qPQ.origin()[2] == dDm_skPQ.origin()[3],
               "lff_sigma_detail::exchange_with_kernel: the kernel's (P, Q) blocks do not match the HF partition.");
  auto dU_qPQ_loc = dU_qPQ.local();
  sF_skij.set_zero();
  if (nkpts != 1) {   // U(q) -> U(R)
    buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
    auto f_Rk = sf_Rk.local();
    utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->Qpts(), MF->lattv(), R_grid, sf_Rk);
    auto U_2D = nda::reshape(dU_qPQ_loc, shape_t<2>{nkpts, NP_loc * NQ_loc});
    nda::blas::gemm(f_Rk, U_2D, buffer);
    U_2D = buffer;
  }
  for (auto ip : nda::range(npol)) {
    for (auto iq : nda::range(ip, npol)) {
      dF_skPQ.local() = ComplexType(0.0);
      thc_solver_comm::primary_to_aux(ip, iq, Dm_skij, dDm_skPQ, thc, MF->kp_to_ibz(), MF->kp_trev());
      if (nkpts != 1) {
        auto f_Rk = sf_Rk.local();
        utils::k_to_R_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(), R_grid, sf_Rk);
        auto DmR_3D = nda::reshape(dDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
        if (buffer.shape() != shape_t<2>{nkpts, NP_loc * NQ_loc}) buffer.resize(shape_t<2>{nkpts, NP_loc * NQ_loc});
        for (int s = 0; s < ns; ++s) {
          nda::blas::gemm(f_Rk, DmR_3D(s, nda::ellipsis{}), buffer);
          DmR_3D(s, nda::ellipsis{}) = buffer;
        }
      }
      auto had_prod2 = nda::map([](ComplexType x, ComplexType y) { return -1.0 * (x * y); });
      for (long s = 0; s < ns; ++s) {
        auto Dm_RPQ = dDm_skPQ.local()(s, nda::ellipsis{});
        Dm_RPQ = had_prod2(Dm_RPQ, dU_qPQ_loc);
      }
      if (nkpts != 1) {
        auto f_kR = sf_Rk.local();
        utils::R_to_k_coefficients(mpi->comm, nda::range(nkpts), MF->kpts(), MF->lattv(), R_grid, sf_Rk);
        auto Dm_3D = nda::reshape(dDm_skPQ.local(), shape_t<3>{ns, nkpts, NP_loc * NQ_loc});
        auto FR_3D = nda::reshape(dF_skPQ.local(), shape_t<3>{ns, nkpts_ibz, NP_loc * NQ_loc});
        for (int s = 0; s < ns; ++s)
          nda::blas::gemm(ComplexType(1.0), f_kR(nda::range(nkpts_ibz), nda::range::all), Dm_3D(s, nda::ellipsis{}),
                          ComplexType(1.0), FR_3D(s, nda::ellipsis{}));
      } else {
        auto F_loc = dF_skPQ.local();
        auto buff_loc = dDm_skPQ.local();
        F_loc += buff_loc;
      }
      thc_solver_comm::aux_to_primary(ip, iq, (ip == iq ? ComplexType(1.0) : ComplexType(2.0)), dF_skPQ, sF_skij, thc, MF->ks_to_k(0));
    }
  }
  if (npol > 1) {   // only ip <= iq was added: symmetrize (as thc_hf_Xqindep)
    auto node_comm = sF_skij.node_comm();
    node_comm->barrier();
    auto F = sF_skij.local();
    for (auto isk : nda::range(F.extent(0) * F.extent(1))) {
      if (isk % node_comm->size() != node_comm->rank()) continue;
      auto Fij = F(isk / F.extent(1), isk % F.extent(1), nda::ellipsis{});
      for (auto i : nda::range(F.extent(2)))
        for (auto j : nda::range(i + 1, F.extent(3))) {
          Fij(i, j) += std::conj(Fij(j, i));
          Fij(i, j) *= ComplexType(0.5);
          Fij(j, i) = std::conj(Fij(i, j));
        }
    }
    node_comm->barrier();
  }
  dU_qPQ.reset();
  dDm_skPQ.reset();
  mpi->comm.barrier();
}

} // lff_sigma_detail
} // solvers
} // methods

#endif // COQUI_THC_EXCHANGE_KERNEL_HPP
