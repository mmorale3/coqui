/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_THC_PROJECT_HPP
#define COQUI_REAL_AXIS_THC_PROJECT_HPP

#include <array>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis {

/**
 * Project a fermionic spectral function from the orbital basis to the THC
 * auxiliary basis (k-diagonal CoQui convention):
 *
 *     A_{PQ}(k, w) = sum_{mu,nu} X_{P,mu}(k) A_{mu,nu}(k, w) X^*_{Q,nu}(k)
 *
 * Layout convention: the auxiliary-basis output stores frequency innermost,
 *
 *     A_aux_PQw[P, Q, iw]
 *
 * which is the layout the FINUFFT-batched polarization / self-energy kernels
 * already require. Keeping iw innermost throughout the driver eliminates the
 * per-(k,q) transpose copies that would otherwise dominate at large Naux.
 *
 * Implementation: fold the frequency axis into a single large GEMM rather
 * than issuing N_w small (Naux x nbnd) GEMMs, with one permutation on the way
 * in (A from (w, mu, nu) to (mu, w, nu)) and one on the way out (M from
 * (P, w, Q) to (P, Q, w)).
 *
 * @param X_kPmu     (Naux, nbnd) THC factor at k.
 * @param A_wmunu    (N_w, nbnd, nbnd) orbital-basis spectral function at k.
 * @param A_aux_PQw  OUTPUT (Naux, Naux, N_w) auxiliary-basis spectral function.
 *
 * Cost: 2 BLAS-3 GEMMs (totaling 2 * N_w * Naux * nbnd * (nbnd + Naux) FLOPs)
 * plus two element permutations of total size O(N_w * Naux * (nbnd + Naux)).
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename XKP, typename XKQ, typename AIn, typename AOut>
inline void primary_to_aux_one_k(XKP  const& X_kP_mu,
                                 XKQ  const& X_kQ_nu,
                                 AIn  const& A_wmunu,
                                 AOut      & A_aux_PQw)
{
  const long Naux_P = X_kP_mu.shape()[0];
  const long Naux_Q = X_kQ_nu.shape()[0];
  const long nbnd = X_kP_mu.shape()[1];
  const long N_w  = A_wmunu.shape()[0];

  utils::check(X_kQ_nu.shape()[1] == nbnd,
               "primary_to_aux_one_k: X_P and X_Q nbnd mismatch");
  utils::check(A_wmunu.shape()[1] == nbnd and A_wmunu.shape()[2] == nbnd,
               "primary_to_aux_one_k: A nbnd mismatch");
  utils::check(A_aux_PQw.shape()[0] == Naux_P and
               A_aux_PQw.shape()[1] == Naux_Q and
               A_aux_PQw.shape()[2] == N_w,
               "primary_to_aux_one_k: A_aux shape mismatch");

  if (N_w == 0) return;

  const ComplexType c_one(1.0, 0.0);
  const ComplexType c_zero(0.0, 0.0);

  if constexpr (MEM != HOST_MEMORY) {
    // Device path: two cuTENSOR contractions, no manual permutations.
    //   T(P, w, nu) = sum_mu X_P(P, mu) A(w, mu, nu)
    //   A_aux(P, Q, w) = sum_nu T(P, w, nu) conj(X_Q)(Q, nu)
    memory::array<MEM, ComplexType, 3> T_PWN(Naux_P, N_w, nbnd);
    nda::tensor::contract(c_one, X_kP_mu, std::string_view("Pm"),
                          A_wmunu, std::string_view("wmn"),
                          c_zero, T_PWN, std::string_view("Pwn"));
    nda::tensor::contract(c_one, T_PWN, std::string_view("Pwn"),
                          nda::conj(X_kQ_nu), std::string_view("Qn"),
                          c_zero, A_aux_PQw, std::string_view("PQw"));
    return;
  }

  // Host path: two GEMMs through nda::blas + manual axis permutations
  // (host elementwise has no broadcast/permute path; host tensor::contract
  // would dispatch to TBLIS or fallback, neither validated here).

  // Step 1: T(P, w, nu) = sum_mu X_P(P, mu) A(w, mu, nu).
  // Permute A from (w, mu, nu) -> (mu, w, nu) so (w, nu) is the contiguous
  // inner pair of the 2D matrix view.
  nda::array<ComplexType, 3> A_perm(nbnd, N_w, nbnd);
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu)
        A_perm(mu, iw, nu) = A_wmunu(iw, mu, nu);
  auto A_perm_2d = nda::reshape(A_perm, std::array<long, 2>{nbnd, N_w * nbnd});

  // T_PWN storage (Naux_P, N_w, nbnd) row-major.
  nda::array<ComplexType, 3> T_PWN(Naux_P, N_w, nbnd);
  auto T_PWN_2d = nda::reshape(T_PWN, std::array<long, 2>{Naux_P, N_w * nbnd});
  // The X_kP_mu argument may be a strided view (e.g. a row-range of a
  // larger sArray). Materialize a contiguous copy for gemm.
  nda::array<ComplexType, 2> X_P_local(Naux_P, nbnd);
  for (long P = 0; P < Naux_P; ++P)
    for (long mu = 0; mu < nbnd; ++mu)
      X_P_local(P, mu) = X_kP_mu(P, mu);
  nda::blas::gemm(c_one, X_P_local, A_perm_2d, c_zero, T_PWN_2d);

  // Step 2: A_aux(P, Q, w) = sum_nu T(P, w, nu) conj(X_Q)(Q, nu).
  auto T_PWN_2d_b = nda::reshape(T_PWN, std::array<long, 2>{Naux_P * N_w, nbnd});
  nda::array<ComplexType, 3> M_PWQ(Naux_P, N_w, Naux_Q);
  auto M_PWQ_2d = nda::reshape(M_PWQ, std::array<long, 2>{Naux_P * N_w, Naux_Q});
  nda::array<ComplexType, 2> X_Q_local(Naux_Q, nbnd);
  for (long Q = 0; Q < Naux_Q; ++Q)
    for (long nu = 0; nu < nbnd; ++nu)
      X_Q_local(Q, nu) = X_kQ_nu(Q, nu);
  nda::blas::gemm(c_one, T_PWN_2d_b, nda::dagger(X_Q_local), c_zero, M_PWQ_2d);

  for (long P = 0; P < Naux_P; ++P)
    for (long iw = 0; iw < N_w; ++iw)
      for (long Q = 0; Q < Naux_Q; ++Q)
        A_aux_PQw(P, Q, iw) = M_PWQ(P, iw, Q);
}

// Convenience overload: same X for both P and Q (legacy callers and the
// fully-replicated path).
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename XK, typename AIn, typename AOut>
inline void primary_to_aux_one_k(XK   const& X_kPmu,
                                 AIn  const& A_wmunu,
                                 AOut      & A_aux_PQw)
{
  primary_to_aux_one_k<MEM>(X_kPmu, X_kPmu, A_wmunu, A_aux_PQw);
}

/**
 * Inverse projection from auxiliary basis (frequency-innermost) back to the
 * orbital basis at a single k:
 *
 *     M_{mu,nu}(k, w) = sum_{P,Q} X^*_{P,mu}(k) M_aux_{P,Q}(k, w) X_{Q,nu}(k)
 *
 * @param X_kPmu     (Naux, nbnd) THC factor at k.
 * @param M_aux_PQw  (Naux, Naux, N_w) auxiliary quantity (iw innermost).
 * @param M_wmunu    OUTPUT (N_w, nbnd, nbnd) orbital-basis quantity.
 */
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename XKP, typename XKQ, typename MIn, typename MOut>
inline void aux_to_primary_one_k(XKP  const& X_kP_mu,
                                 XKQ  const& X_kQ_nu,
                                 MIn  const& M_aux_PQw,
                                 MOut      & M_wmunu)
{
  const long Naux_P = X_kP_mu.shape()[0];
  const long Naux_Q = X_kQ_nu.shape()[0];
  const long nbnd = X_kP_mu.shape()[1];
  const long N_w  = M_wmunu.shape()[0];

  utils::check(X_kQ_nu.shape()[1] == nbnd,
               "aux_to_primary_one_k: X_P and X_Q nbnd mismatch");
  utils::check(M_aux_PQw.shape()[0] == Naux_P and
               M_aux_PQw.shape()[1] == Naux_Q and
               M_aux_PQw.shape()[2] == N_w,
               "aux_to_primary_one_k: M_aux shape mismatch");
  utils::check(M_wmunu.shape()[1] == nbnd and M_wmunu.shape()[2] == nbnd,
               "aux_to_primary_one_k: M shape mismatch");

  if (N_w == 0) return;

  const ComplexType c_one(1.0, 0.0);
  const ComplexType c_zero(0.0, 0.0);

  if constexpr (MEM != HOST_MEMORY) {
    // Device path: two cuTENSOR contractions.
    //   T(P, w, nu) = sum_Q M_aux(P, Q, w) X_Q(Q, nu)
    //   M(w, mu, nu) = sum_P conj(X_P(P, mu)) T(P, w, nu)
    memory::array<MEM, ComplexType, 3> T_PWN(Naux_P, N_w, nbnd);
    nda::tensor::contract(c_one, M_aux_PQw, std::string_view("PQw"),
                          X_kQ_nu, std::string_view("Qn"),
                          c_zero, T_PWN, std::string_view("Pwn"));
    nda::tensor::contract(c_one, nda::conj(X_kP_mu), std::string_view("Pm"),
                          T_PWN, std::string_view("Pwn"),
                          c_zero, M_wmunu, std::string_view("wmn"));
    return;
  }

  // Host path: GEMM-based with manual axis permutations.
  // Step 1: T(P, w, nu) = sum_Q M_aux(P, Q, w) X_Q(Q, nu).
  //
  // Permute M_aux from (P, Q, w) -> (P, w, Q) so we can reshape the (P, w)
  // pair into a single matrix row index and dot the Q axis against X_Q.
  nda::array<ComplexType, 3> M_PWQ(Naux_P, N_w, Naux_Q);
  for (long P = 0; P < Naux_P; ++P)
    for (long Q = 0; Q < Naux_Q; ++Q)
      for (long iw = 0; iw < N_w; ++iw)
        M_PWQ(P, iw, Q) = M_aux_PQw(P, Q, iw);
  auto M_PWQ_2d = nda::reshape(M_PWQ, std::array<long, 2>{Naux_P * N_w, Naux_Q});
  nda::array<ComplexType, 3> T_PWN(Naux_P, N_w, nbnd);
  auto T_PWN_2d = nda::reshape(T_PWN, std::array<long, 2>{Naux_P * N_w, nbnd});
  // Materialize a contiguous local copy of X_Q (may be a strided view).
  nda::array<ComplexType, 2> X_Q_local(Naux_Q, nbnd);
  for (long Q = 0; Q < Naux_Q; ++Q)
    for (long nu = 0; nu < nbnd; ++nu)
      X_Q_local(Q, nu) = X_kQ_nu(Q, nu);
  nda::blas::gemm(c_one, M_PWQ_2d, X_Q_local, c_zero, T_PWN_2d);

  // Step 2: M(w, mu, nu) = sum_P conj(X_P)(P, mu) T(P, w, nu).
  auto T_PWN_2d_b = nda::reshape(T_PWN, std::array<long, 2>{Naux_P, N_w * nbnd});
  nda::array<ComplexType, 3> M_perm(nbnd, N_w, nbnd);
  auto M_perm_2d = nda::reshape(M_perm, std::array<long, 2>{nbnd, N_w * nbnd});
  nda::array<ComplexType, 2> X_P_local(Naux_P, nbnd);
  for (long P = 0; P < Naux_P; ++P)
    for (long mu = 0; mu < nbnd; ++mu)
      X_P_local(P, mu) = X_kP_mu(P, mu);
  nda::blas::gemm(c_one, nda::dagger(X_P_local), T_PWN_2d_b, c_zero, M_perm_2d);

  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu)
        M_wmunu(iw, mu, nu) = M_perm(mu, iw, nu);
}

// Convenience overload: same X for both P and Q.
template <MEMORY_SPACE MEM = HOST_MEMORY,
          typename XK, typename MIn, typename MOut>
inline void aux_to_primary_one_k(XK   const& X_kPmu,
                                 MIn  const& M_aux_PQw,
                                 MOut      & M_wmunu)
{
  aux_to_primary_one_k<MEM>(X_kPmu, X_kPmu, M_aux_PQw, M_wmunu);
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_THC_PROJECT_HPP
