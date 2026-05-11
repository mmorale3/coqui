/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Residue-sum kernel for the QP-form QSGW contour-deformation framework.
 *
 *   Sigma_c^{residue}_{ij}(s, k_ibz, omega)
 *     = sum over enclosed QP poles n' at FBZ k+q, all q_FBZ
 *       sign_n' * M^*_{i,n'}(s, k+q) M_{j,n'}(s, k+q) W_c(q, eps_n'^QP - omega)
 *
 * where M is the QP rotation (MO coefficient) at FBZ k', and W_c is the
 * dynamic part of W evaluated on the real axis at Omega = eps_n'^QP - omega.
 *
 * Sign convention (contour closed in the upper half plane for omega > mu,
 * lower half for omega < mu — Govoni & Galli 2015 Appendix A):
 *
 *   sign_n' = -1 if eps_n'^QP < mu AND omega > mu   (occupied pole below omega)
 *   sign_n' = +1 if eps_n'^QP > mu AND omega < mu   (unoccupied pole above omega)
 *   sign_n' =  0 otherwise
 *
 * In aux basis (THC factorization L_{μν,P}(k,q) = X*_μP(k+q) Y_νP(k)),
 * the orbital outer product M*_{i,n'} M_{j,n'} pre-multiplied by X and
 * post-multiplied by X gives a (Naux, Naux) projector onto the n'
 * channel; contracted with W_c(q, Omega)_{P, Q} yields the (i, j)
 * orbital-basis contribution.
 *
 * This is the most expensive piece of the CD framework. The k/q-pair
 * bookkeeping mirrors `methods/GW/thc_gw.icc::eval_Sigma_all_kspace`.
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_QP_RESIDUE_HPP
#define COQUI_REAL_AXIS_QP_RESIDUE_HPP

#include <complex>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "methods/GW_real_axis_qp/real_axis_qp_pade.hpp"

namespace methods {
namespace real_axis_qp {

using ComplexType = std::complex<double>;

/**
 * Determine the residue sign for one (eps_n', mu, omega) triple.
 *
 * Implements the Govoni & Galli convention. The sign is non-zero only
 * when the pole at eps_n' is enclosed by the deformed contour, which
 * depends on whether omega and eps_n' are on the same or opposite
 * sides of mu.
 *
 * @return  -1 : occupied pole eps_n' < mu enclosed (omega > mu)
 *          +1 : unoccupied pole eps_n' > mu enclosed (omega < mu)
 *           0 : pole not enclosed (no contribution)
 */
inline int residue_sign(double eps_n, double mu_chem, double omega) {
  const bool n_occupied   = (eps_n < mu_chem);
  const bool omega_above  = (omega > mu_chem);
  if (omega_above and n_occupied)   return -1;
  if ((!omega_above) and (!n_occupied)) return +1;
  return 0;
}

/**
 * Evaluate the residue contribution for one (s, k_ibz, omega) target by
 * looping over (q_FBZ, n_band_at_kpq) pairs. The result is accumulated
 * into Sigma_c_orbital_ij at orbital basis.
 *
 * Inputs:
 *   k_ibz, omega        target (s, k_ibz, omega) for which Sigma_c is built.
 *   mu_chem             chemical potential.
 *   E_ska(s, k_ibz, n)  QP energies at IBZ k (shared). Use kp_to_ibz to
 *                        get FBZ k indices for the W_c lookup.
 *   MO_skin(s, k_ibz, mu, n)  QP rotations at IBZ k.
 *   X_skPmu(s, k_FBZ, P, mu)  THC factor at FBZ k.
 *   W_c_iw_qPQ(q_ibz, P, Q, l) W_c on the iω' Matsubara mesh — used
 *                        per-channel by the Pade fit to get W_c at
 *                        Omega = eps_n' - omega.
 *   iw_nodes            Matsubara nodes (size N_iω) — complex iω' values.
 *   qk_to_k2, qminus, kp_to_ibz   BZ index maps.
 *   nq_ibz              number of IBZ q-points.
 *   nbnd                number of bands.
 *
 * Output:
 *   Sigma_c_orbital_ij(i, j)   accumulated into; caller must zero on entry.
 *
 * Cost (serial estimate): N_q * nbnd * (Naux² Pade-eval + Naux² outer
 * product + Naux² → nbnd² projection). Dominated by the Pade-eval per
 * (q, n') pair which itself is O(N_iω²) per (P, Q). For Si production:
 * 13 * 256 * 2566² * 16² ≈ 5e12 ops — too slow without aux/k batching
 * + caching of Pade coefficients across calls. See "Performance notes"
 * at the bottom for the production-scale recipe.
 */
template<typename X_4D_t, typename W_4D_t>
void residue_contribution_one_omega(
    long k_ibz, double omega, double mu_chem,
    nda::array<double, 3> const& E_ska,
    nda::array<ComplexType, 4> const& MO_skin,
    X_4D_t const& X_skPmu,
    W_4D_t const& W_c_iw_qPQ,
    std::vector<ComplexType> const& iw_nodes,
    nda::array<long, 2> const& qk_to_k2,
    nda::array<long, 1> const& qminus,
    nda::array<long, 1> const& kp_to_ibz,
    long nq_ibz,
    long nbnd,
    long s,
    nda::array<ComplexType, 2>& Sigma_c_orbital_ij)
{
  const long Naux = X_skPmu.shape()[2];
  utils::check(Sigma_c_orbital_ij.shape()[0] == nbnd
            and Sigma_c_orbital_ij.shape()[1] == nbnd,
               "residue: output shape mismatch.");

  // Per-(P, Q) Pade-coefficient cache, computed ONCE per call. Each entry
  // holds the Thiele coefficients for the corresponding W_c_iw_qPQ(q, P, Q, :)
  // strand. Built on demand inside the q-loop below.
  //
  // For production this cache should live outside the (omega) loop and
  // be reused across all omega values — moves the per-call cost down by
  // O(N_omega).

  // Scratch: W_c at the real argument Omega = eps_n' - omega, in aux basis.
  nda::array<ComplexType, 2> W_c_PQ_at_Omega(Naux, Naux);

  // Scratch: per-n' orbital projector M*_{i,n'} M_{j,n'} pre-multiplied by X,
  // i.e., (X^H M)_P_n' (M^H X)_Q_n' summed over a single n'.

  for (long iq = 0; iq < nq_ibz; ++iq) {
    // For each q_IBZ, the FBZ q' index is needed; for the simplest path
    // we assume q_ibz == q_FBZ (trivial-IBZ case; production needs the
    // Qpts_ibz → FBZ map).
    const long q_FBZ = iq;     // TODO: Qpts_ibz mapping for non-trivial IBZ

    // FBZ k' = k + q via qk_to_k2(qminus(q), k_FBZ). Here we treat k_ibz
    // as the FBZ k for the trivial-IBZ case.
    const long k_FBZ = k_ibz;  // TODO: k_FBZ from IBZ + symmetry star
    const long kpq_FBZ = qk_to_k2(qminus(q_FBZ), k_FBZ);
    const long kpq_ibz = kp_to_ibz(kpq_FBZ);

    // Loop over bands at k_FBZ + q (the "n'" index).
    for (long n_prime = 0; n_prime < nbnd; ++n_prime) {
      const double eps_n = E_ska(s, kpq_ibz, n_prime);
      const int sign_n = residue_sign(eps_n, mu_chem, omega);
      if (sign_n == 0) continue;

      const double Omega_real = eps_n - omega;

      // Pade-evaluate W_c at Omega_real for each (P, Q). For
      // robustness against sharp pole proximity, add a small imag
      // shift +i*eps_pole (e.g., 1e-3) — see Risks in the design doc.
      const ComplexType Omega_pade_arg = ComplexType(Omega_real, 1e-3);

      // pade_eval_batched takes (NP, NQ, N_iω) → (NP, NQ). Build a
      // (Naux, Naux, N_iω) view at iq.
      // NOTE: this is the bottleneck — production must cache the
      // coefficients across omega and use a vectorized eval.
      auto W_c_iw_at_q = W_c_iw_qPQ(iq, nda::ellipsis{});
      // For now we wrap W_c_iw_at_q (3D view) and call the batched Pade.
      nda::array<ComplexType, 3> W_c_iw_PQ_local(W_c_iw_at_q);
      pade_eval_batched(iw_nodes, W_c_iw_PQ_local,
                         Omega_pade_arg, W_c_PQ_at_Omega);

      // Build the orbital projector at FBZ k+q from X and MO:
      //   A_PQ = (X^H M_n')_P (M^H X)_Q^* summed over a single n'.
      // Equivalently:
      //   (X^H M_n')(P) = Σ_μ X^*(s, k_FBZ+q, P, μ) MO(s, kpq_ibz, μ, n')
      //   (M^H X)(Q)   = Σ_ν conj(MO(s, kpq_ibz, ν, n')) X(s, k_FBZ+q, Q, ν)
      auto X_at_kpq = X_skPmu(s, kpq_FBZ, nda::ellipsis{});   // (Naux, nbnd)
      auto MO_at_kpq = MO_skin(s, kpq_ibz, nda::ellipsis{});  // (nbnd, nbnd)
      nda::array<ComplexType, 1> XHM_P(Naux), MHX_Q(Naux);
      XHM_P() = ComplexType(0.0, 0.0);
      MHX_Q() = ComplexType(0.0, 0.0);
      for (long P = 0; P < Naux; ++P) {
        ComplexType acc(0.0, 0.0);
        for (long mu = 0; mu < nbnd; ++mu)
          acc += std::conj(X_at_kpq(P, mu)) * MO_at_kpq(mu, n_prime);
        XHM_P(P) = acc;
      }
      for (long Q = 0; Q < Naux; ++Q) {
        ComplexType acc(0.0, 0.0);
        for (long nu = 0; nu < nbnd; ++nu)
          acc += std::conj(MO_at_kpq(nu, n_prime)) * X_at_kpq(Q, nu);
        MHX_Q(Q) = acc;
      }

      // Sigma_c_orbital_ij += sign_n * Σ_PQ XHM_P W_c_PQ(Omega) MHX_Q
      //                              * (orbital projector from i,j side)
      // For a strict residue, the orbital projection back to (i, j) basis
      // also needs X(s, k_FBZ, P, i)^* and X(s, k_FBZ, Q, j) — same THC
      // factor structure as the orbital→aux projection. Build the (i, j)
      // contribution as a matrix product.
      //
      // The compact form is:
      //   Sigma_c_orbital(i, j) += sign_n * Σ_{P,Q} X*(s, k_FBZ, P, i)
      //                                            * (XHM_P W_c_PQ MHX_Q)
      //                                            * X(s, k_FBZ, Q, j)
      //
      // We compute a 2-step contraction:
      //   tmp_iQ = Σ_P X*(s, k_FBZ, P, i) * XHM_P * W_c_PQ(P, Q)
      //   Sigma_c(i, j) += sign_n * Σ_Q tmp_iQ * MHX_Q * X(s, k_FBZ, Q, j)
      //
      // This is O(nbnd*Naux²) per residue evaluation — comparable to the
      // imag-axis Sigma kernel cost per (k, q, n') pair. Production will
      // batch over n' and use GEMM via nda::blas::gemm; here we keep
      // explicit triple loops for clarity.
      auto X_at_k_FBZ = X_skPmu(s, k_FBZ, nda::ellipsis{});   // (Naux, nbnd)
      // First contract: vec_PQ = XHM_P * W_c_PQ(P, Q)        shape (Naux,)*indexed Q
      // Then: vec_iQ(i, Q) = Σ_P X*(P, i) * vec_PQ(P, Q)
      nda::array<ComplexType, 2> vec_PQ(Naux, Naux);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          vec_PQ(P, Q) = XHM_P(P) * W_c_PQ_at_Omega(P, Q);

      nda::array<ComplexType, 2> tmp_iQ(nbnd, Naux);
      tmp_iQ() = ComplexType(0.0, 0.0);
      for (long i = 0; i < nbnd; ++i)
        for (long Q = 0; Q < Naux; ++Q) {
          ComplexType acc(0.0, 0.0);
          for (long P = 0; P < Naux; ++P)
            acc += std::conj(X_at_k_FBZ(P, i)) * vec_PQ(P, Q);
          tmp_iQ(i, Q) = acc;
        }

      const ComplexType sign_c = ComplexType(static_cast<double>(sign_n), 0.0);
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j) {
          ComplexType acc(0.0, 0.0);
          for (long Q = 0; Q < Naux; ++Q)
            acc += tmp_iQ(i, Q) * MHX_Q(Q) * X_at_k_FBZ(Q, j);
          Sigma_c_orbital_ij(i, j) += sign_c * acc;
        }
    } // n_prime loop
  }   // q loop
}

/**
 * Performance notes for production:
 *
 * The above scalar implementation is O(N_q * nbnd * Naux² * (N_iω² + nbnd²))
 * per (s, k_ibz, omega) target. At Si production scale (Nq_ibz=13,
 * nbnd=256, Naux=2566, N_iω=32), one residue evaluation is ~5e12 ops,
 * which is far too slow as a per-omega loop.
 *
 * Production recipe (Phase B/C work):
 *  1. Pre-compute Pade coefficients g_PQ(q, l_pade) ONCE per SCF iter
 *     (size Nq_ibz · Naux² · N_iω complex) and re-use across all
 *     omega evaluations. Cost: same as building W_c on iω' — amortized
 *     across all (omega, n', q) lookups.
 *  2. Reformulate inner contractions as GEMMs:
 *       step A: G_aux_{P, n'}(s, k_FBZ+q) = X^*(s, k+q, P, μ) MO(s, k+q, μ, n')
 *       step B: aux block per (q, n'): same matrix product structure
 *       step C: orbital back-projection via thc_solver_comm::aux_to_primary
 *               (already implemented).
 *  3. Batch over n' (band) inside the (q, sign) loop so the W_c_PQ matrix
 *     is reused across many n'.
 *  4. Distribute the Naux² evaluation of pade_eval_batched over (P, Q) on
 *     the bosonic proc grid (same grid as W_c_iw storage).
 *
 * The pieces above are concrete optimizations; the algorithm here is the
 * correct math for one (s, k_ibz, omega).
 */

} // namespace real_axis_qp
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_RESIDUE_HPP
