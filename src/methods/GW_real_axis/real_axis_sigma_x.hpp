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
inline void evaluate_Sigma_x_serial(
    boost::mpi3::communicator        & comm,
    real_freq_grid_t            const& grid,
    nda::array<ComplexType, 5>  const& A_skwij,
    nda::array<ComplexType, 4>  const& X_skPmu,
    nda::array<ComplexType, 3>  const& V_qPQ,
    nda::array<long, 2>         const& kmq_to_kp,
    nda::array<ComplexType, 4>       & Sigma_x_skij,
    long iq_gamma = -1)
{
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

  // Step 2: project n to auxiliary basis. Reuse primary_to_aux_one_k by
  // packaging n as a single-frequency array. Layout convention now has
  // iw innermost, so the dummy aux array is (Naux, Naux, 1).
  using nda::range;
  const auto _ = range::all;

  nda::array<ComplexType, 4> n_aux_skPQ(ns, Nk, Naux, Naux);
  {
    nda::array<ComplexType, 3> n_dummy_munu(1, nbnd, nbnd);
    nda::array<ComplexType, 3> n_dummy_PQ(Naux, Naux, 1);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        auto X_view = X_skPmu(s, k, _, _);
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            n_dummy_munu(0, mu, nu) = n_skij(s, k, mu, nu);
        primary_to_aux_one_k(X_view, n_dummy_munu, n_dummy_PQ);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            n_aux_skPQ(s, k, P, Q) = n_dummy_PQ(P, Q, 0);
      }
  }

  // Step 3: for each (s, k), accumulate Sigma_x_aux_PQ(k) =
  //         -(1/Nq) sum_q V_PQ(q) (Hadamard) n_aux_PQ(k-q),
  // then back-project to orbital basis. Distributed over (s, k) by rank;
  // single allreduce on the orbital-basis Sigma_x at the end.
  Sigma_x_skij = ComplexType(0.0, 0.0);
  {
    const int rank = comm.rank();
    const int size = comm.size();
    const long total_sk = ns * Nk;
    nda::array<ComplexType, 3> SxA_dummy_PQ(Naux, Naux, 1);
    nda::array<ComplexType, 3> SxA_dummy_munu(1, nbnd, nbnd);
    const double inv_Nq = 1.0 / static_cast<double>(Nq);
    for (long sk = rank; sk < total_sk; sk += size) {
      const long s = sk / Nk;
      const long k = sk % Nk;
      SxA_dummy_PQ = ComplexType(0.0, 0.0);
      for (long iq = 0; iq < Nq; ++iq) {
        if (iq == iq_gamma) continue;
        const long ikmq = kmq_to_kp(k, iq);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            SxA_dummy_PQ(P, Q, 0) -=
                inv_Nq * V_qPQ(iq, P, Q) * n_aux_skPQ(s, ikmq, P, Q);
      }
      auto X_view = X_skPmu(s, k, _, _);
      aux_to_primary_one_k(X_view, SxA_dummy_PQ, SxA_dummy_munu);
      for (long mu = 0; mu < nbnd; ++mu)
        for (long nu = 0; nu < nbnd; ++nu)
          Sigma_x_skij(s, k, mu, nu) = SxA_dummy_munu(0, mu, nu);
    }
    if (size > 1)
      comm.all_reduce_in_place_n(Sigma_x_skij.data(), Sigma_x_skij.size(), std::plus<>{});
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SIGMA_X_HPP
