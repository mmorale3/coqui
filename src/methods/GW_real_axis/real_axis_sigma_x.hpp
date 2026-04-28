/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_SIGMA_X_HPP
#define COQUI_REAL_AXIS_SIGMA_X_HPP

#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "methods/GW_real_axis/real_axis_proc_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * Static exchange (Hartree-Fock-X) self-energy in the THC auxiliary basis.
 *
 * Implements (notes Eq. Sigma_exchange) under the CoQui k-diagonal THC
 * factorization:
 *
 *   Sigma^x_{munu}(k) = - (1/N_q) sum_q sum_{P,Q}
 *       X^*_{P,mu}(k) V_{P,Q}(q) n^aux_{P,Q}(k+q) X_{Q,nu}(k)
 *
 * where the auxiliary-basis density matrix is
 *
 *   n^aux_{P,Q}(k) = sum_{alpha,beta} X_{P,alpha}(k) n_{alpha,beta}(k) X^*_{Q,beta}(k)
 *
 * and the orbital density matrix is the integrated occupied spectral weight
 *
 *   n_{alpha,beta}(k) = int dw f(w) A_{alpha,beta}(k, w)
 *
 * The k-shift in the q-sum follows the convention of the screened-exchange
 * channel of the GW self-energy, which uses k-q (matching the Sigma^c
 * channel). Some references define exchange with k+q; both are equivalent
 * after relabeling q -> -q on a uniform mesh.
 *
 * Step-1 of MPI distribution: the comm is plumbed through the API but the
 * body still does the full computation redundantly on every rank.
 *
 * @param comm          MPI communicator (plumbed; not yet used to partition).
 * @param grid          finite-T real-frequency grid
 * @param A_skwij       (ns, Nk, N_w, nbnd, nbnd) input spectral function
 * @param X_skPmu       (ns, Nk, Naux, nbnd) THC factor
 * @param V_qPQ         (Nq, Naux, Naux) auxiliary Coulomb
 * @param kmq_to_kp     (Nk, Nq) BZ index of k-q
 * @param Sigma_x_skij  OUTPUT (ns, Nk, nbnd, nbnd) static exchange self-energy
 */
template<MEMORY_SPACE MEM = HOST_MEMORY,
         nda::ArrayOfRank<2> KMap_t>
inline void evaluate_Sigma_x_serial(
    boost::mpi3::communicator        & comm,
    real_freq_grid_t            const& grid,
    memory::array<MEM, ComplexType, 5> const& A_skwij,
    memory::array<MEM, ComplexType, 4> const& X_skPmu,
    memory::array<MEM, ComplexType, 3> const& V_qPQ,
    KMap_t                       const& kmq_to_kp,
    memory::array<MEM, ComplexType, 4>       & Sigma_x_skij,
    long iq_gamma = -1)
{
  static_assert(MEM == HOST_MEMORY,
                "evaluate_Sigma_x_serial<DEVICE>: device-side allocation is "
                "MEM-aware but the inner element-wise loops over (s, k, q, "
                "P, Q, w) (Sigma_x = -V * n_aux) are still host-only. The "
                "Hadamard product is a single nda::map call once devicified.");
  const long ns    = A_skwij.shape()[0];
  const long Nk    = A_skwij.shape()[1];
  const long N_w   = A_skwij.shape()[2];
  const long nbnd  = A_skwij.shape()[3];
  const long Nq    = V_qPQ.shape()[0];
  const long Naux  = V_qPQ.shape()[1];

  utils::check(X_skPmu.shape()[0] == ns and X_skPmu.shape()[1] == Nk and
               X_skPmu.shape()[2] == Naux and X_skPmu.shape()[3] == nbnd,
               "evaluate_Sigma_x_serial: X shape mismatch");
  utils::check(V_qPQ.shape()[2] == Naux,
               "evaluate_Sigma_x_serial: V not square in (P,Q)");
  utils::check(kmq_to_kp.shape()[0] == Nk and kmq_to_kp.shape()[1] == Nq,
               "evaluate_Sigma_x_serial: kmq_to_kp shape mismatch");
  utils::check(Sigma_x_skij.shape()[0] == ns and Sigma_x_skij.shape()[1] == Nk and
               Sigma_x_skij.shape()[2] == nbnd and Sigma_x_skij.shape()[3] == nbnd,
               "evaluate_Sigma_x_serial: Sigma_x shape mismatch");
  utils::check(N_w == grid.N_w(),
               "evaluate_Sigma_x_serial: A N_w != grid.N_w()");

  // Step 1: orbital-basis density matrix n_{munu}(k) = int dw f(w) A_{munu}(k, w).
  nda::array<ComplexType, 4> n_skij(ns, Nk, nbnd, nbnd);
  n_skij = ComplexType(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k) {
      for (long iw = 0; iw < N_w; ++iw) {
        const double f_w = grid.fermi(grid.w()(iw));
        const double w_w = grid.w_weights()(iw);
        const double coeff = f_w * w_w;
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            n_skij(s, k, mu, nu) += coeff * A_skwij(s, k, iw, mu, nu);
      }
    }

  // Step 2: project n to auxiliary basis using the (P, Q) distribution
  // pattern: each rank fills only its local (P_loc, Q_loc) block of
  // n_aux_skPQ via the two-X-views overload of primary_to_aux_one_k.
  using nda::range;
  const auto _ = range::all;

  // Pick the same (P, Q) proc grid used by the bosonic/fermionic kernels.
  const long nproc = static_cast<long>(comm.size());
  auto pgrid_PQ = square_factor_capped(nproc, Naux);
  const long gridP = pgrid_PQ[0];
  const long gridQ = pgrid_PQ[1];
  // Determine this rank's (P_loc, Q_loc) block. Convention matches
  // make_distributed_array's row-major C-layout assignment.
  const long ip = static_cast<long>(comm.rank());
  const long iP_block = (ip / gridQ) % gridP;
  const long iQ_block = ip % gridQ;
  auto chunk = [](long N, long G, long i) {
    long base = N / G;
    long rem  = N % G;
    long start = i * base + std::min(i, rem);
    long sz    = base + (i < rem ? 1 : 0);
    return std::array<long, 2>{start, sz};
  };
  auto [P0, NP_loc] = chunk(Naux, gridP, iP_block);
  auto [Q0, NQ_loc] = chunk(Naux, gridQ, iQ_block);
  range Pr(P0, P0 + NP_loc);
  range Qr(Q0, Q0 + NQ_loc);

  nda::array<ComplexType, 4> n_aux_skPQ_loc(ns, Nk, NP_loc, NQ_loc);
  {
    nda::array<ComplexType, 3> n_dummy_munu(1, nbnd, nbnd);
    nda::array<ComplexType, 3> n_dummy_PQ(NP_loc, NQ_loc, 1);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        auto X_P_slice = X_skPmu(s, k, Pr, _);
        auto X_Q_slice = X_skPmu(s, k, Qr, _);
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            n_dummy_munu(0, mu, nu) = n_skij(s, k, mu, nu);
        primary_to_aux_one_k(X_P_slice, X_Q_slice,
                             n_dummy_munu, n_dummy_PQ);
        for (long iP = 0; iP < NP_loc; ++iP)
          for (long iQ = 0; iQ < NQ_loc; ++iQ)
            n_aux_skPQ_loc(s, k, iP, iQ) = n_dummy_PQ(iP, iQ, 0);
      }
  }

  // Step 3: each rank accumulates the local (P_loc, Q_loc) block of
  // SxA(k) = -(1/Nq) sum_q V(q)_local Hadamard n_aux(k-q)_local, then
  // back-projects to orbital. Single allreduce on Sigma_x_skij at the end.
  Sigma_x_skij = ComplexType(0.0, 0.0);
  {
    nda::array<ComplexType, 3> SxA_dummy_PQ(NP_loc, NQ_loc, 1);
    nda::array<ComplexType, 3> SxA_dummy_munu(1, nbnd, nbnd);
    const double inv_Nq = 1.0 / static_cast<double>(Nq);
    for (long s = 0; s < ns; ++s) {
      for (long k = 0; k < Nk; ++k) {
        SxA_dummy_PQ = ComplexType(0.0, 0.0);
        for (long iq = 0; iq < Nq; ++iq) {
          if (iq == iq_gamma) continue;
          const long ikmq = kmq_to_kp(k, iq);
          for (long iP = 0; iP < NP_loc; ++iP) {
            const long P = P0 + iP;
            for (long iQ = 0; iQ < NQ_loc; ++iQ) {
              const long Q = Q0 + iQ;
              SxA_dummy_PQ(iP, iQ, 0) -=
                  inv_Nq * V_qPQ(iq, P, Q) * n_aux_skPQ_loc(s, ikmq, iP, iQ);
            }
          }
        }
        auto X_P_slice = X_skPmu(s, k, Pr, _);
        auto X_Q_slice = X_skPmu(s, k, Qr, _);
        aux_to_primary_one_k(X_P_slice, X_Q_slice,
                             SxA_dummy_PQ, SxA_dummy_munu);
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            Sigma_x_skij(s, k, mu, nu) += SxA_dummy_munu(0, mu, nu);
      }
    }
    if (comm.size() > 1)
      comm.all_reduce_in_place_n(Sigma_x_skij.data(), Sigma_x_skij.size(), std::plus<>{});
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SIGMA_X_HPP
