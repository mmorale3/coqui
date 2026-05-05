/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Self-consistency energy checks for PAW-ISDF-THC factorization.
 *
 * Computes Hartree (Coulomb) and exchange energies through the (X, 𝒱)
 * THC factorization and provides independent algebraic-equivalent paths
 * for cross-validation. Mathematically:
 *
 *   V_{ijkl}  ≈ Σ_{ΛΣ} X*_{Λi} X_{Λj} 𝒱_{ΛΣ} X*_{Σk} X_{Σl}
 *
 *   E_H       = (1/2) Σ_{ij} f_i f_j (ii | jj)
 *             = (1/2) Σ_{ij} f_i f_j Σ_{ΛΣ} |X_{Λi}|^2 𝒱_{ΛΣ} |X_{Σj}|^2
 *             ≡ (1/2) Σ_{ΛΣ} D_Λ 𝒱_{ΛΣ} D_Σ          (D_Λ = Σ_i f_i |X_{Λi}|^2)
 *
 *   E_x       = -(1/2) Σ_{ij} f_i f_j (ij | ji)
 *             = -(1/2) Σ_{ΛΣ} G_{ΛΣ} 𝒱_{ΛΣ} G^*_{ΛΣ}
 *             where G_{ΛΣ} = Σ_i f_i X*_{Λi} X_{Σi}
 *
 * The closed-form expressions above are evaluated as `hartree_energy_thc` /
 * `exchange_energy_thc`. As an independent check, we also materialize the
 * full V_{ijkl} from (X, 𝒱) and contract with f_i f_j directly. Any
 * disagreement signals a bug in the THC contraction algebra (the
 * X-concatenation or K_a addition in thc_reader_t::augment_thc_with_paw).
 *
 * Real-data validation against a direct PAW Coulomb routine on smooth
 * orbitals + augmentation is a Phase 5 follow-up; this header provides the
 * machinery for the algebra-equivalence test that doesn't need a fixture.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_ENERGY_CHECK_HPP
#define HAMILTONIAN_PAW_PAW_ENERGY_CHECK_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Compute D_Λ = Σ_i f_i |X_{Λi}|^2.
 * X has shape (nbnd, N_Lambda) for one (s, k); occ has shape (nbnd,).
 */
template<typename X_t, typename occ_t, typename D_t>
inline void compute_diag_density(X_t const& X, occ_t const& occ, D_t & D)
{
    long N_Lambda = X.extent(0);
    long nbnd     = X.extent(1);
    utils::check(occ.extent(0) == nbnd, "occ size mismatch");
    utils::check(D.extent(0)   == N_Lambda, "D size mismatch");
    for (long L=0; L<N_Lambda; ++L) {
        double acc = 0.0;
        for (long i=0; i<nbnd; ++i) {
            ComplexType x = X(L, i);
            acc += occ(i) * (std::real(x)*std::real(x) + std::imag(x)*std::imag(x));
        }
        D(L) = ComplexType(acc, 0.0);
    }
}

/**
 * Compute G_{ΛΣ} = Σ_i f_i X*_{Λi} X_{Σi}.
 */
template<typename X_t, typename occ_t, typename G_t>
inline void compute_offdiag_density(X_t const& X, occ_t const& occ, G_t & G)
{
    long N_Lambda = X.extent(0);
    long nbnd     = X.extent(1);
    utils::check(G.extent(0) == N_Lambda && G.extent(1) == N_Lambda,
                 "G shape mismatch");
    G() = ComplexType(0.0);
    for (long L=0; L<N_Lambda; ++L)
    for (long S=0; S<N_Lambda; ++S) {
        ComplexType acc(0.0);
        for (long i=0; i<nbnd; ++i)
            acc += occ(i) * std::conj(X(L, i)) * X(S, i);
        G(L, S) = acc;
    }
}

/**
 * E_H via the THC closed form:
 *   E_H = (1/2) Σ_{ΛΣ} D_Λ 𝒱_{ΛΣ} D_Σ
 *
 * V is (N_Lambda, N_Lambda) Hermitian. occ has shape (nbnd,). X is
 * (N_Lambda, nbnd). All inputs at one (s, k); spin/k aggregation is the
 * caller's job.
 */
template<typename X_t, typename V_t, typename occ_t>
inline double hartree_energy_thc(X_t const& X, V_t const& V, occ_t const& occ)
{
    long N = X.extent(0);
    nda::array<ComplexType,1> D(N);
    compute_diag_density(X, occ, D);
    ComplexType acc(0.0);
    for (long L=0; L<N; ++L)
    for (long S=0; S<N; ++S)
        acc += D(L) * V(L, S) * D(S);
    return 0.5 * std::real(acc);
}

/**
 * E_x via the THC closed form:
 *   E_x = -(1/2) Σ_{ΛΣ} G_{ΛΣ} 𝒱_{ΛΣ} G*_{ΛΣ}   (= -1/2 trace-like)
 */
template<typename X_t, typename V_t, typename occ_t>
inline double exchange_energy_thc(X_t const& X, V_t const& V, occ_t const& occ)
{
    long N = X.extent(0);
    nda::array<ComplexType,2> G(N, N);
    compute_offdiag_density(X, occ, G);
    ComplexType acc(0.0);
    for (long L=0; L<N; ++L)
    for (long S=0; S<N; ++S)
        acc += G(L, S) * V(L, S) * std::conj(G(L, S));
    return -0.5 * std::real(acc);
}

/**
 * Materialize V_{ijkl} from (X, 𝒱) via Σ_{ΛΣ} X*_{Λi} X_{Λj} 𝒱_{ΛΣ} X*_{Σk} X_{Σl}.
 *
 * O(nbnd^4 * N_Lambda^2) — only intended for unit tests on small systems.
 * Output: V4(i, j, k, l), shape (nbnd, nbnd, nbnd, nbnd).
 */
template<typename X_t, typename V_t, typename V4_t>
inline void materialize_eri(X_t const& X, V_t const& V, V4_t & V4)
{
    long N    = X.extent(0);
    long nbnd = X.extent(1);
    utils::check(V4.extent(0) == nbnd && V4.extent(1) == nbnd &&
                 V4.extent(2) == nbnd && V4.extent(3) == nbnd,
                 "V4 must be (nbnd, nbnd, nbnd, nbnd)");
    V4() = ComplexType(0.0);
    // Form A_{Λij} = X*_{Λi} X_{Λj} (a Λ×nbnd×nbnd tensor)
    nda::array<ComplexType,3> A(N, nbnd, nbnd);
    for (long L=0; L<N; ++L)
    for (long i=0; i<nbnd; ++i)
    for (long j=0; j<nbnd; ++j)
        A(L, i, j) = std::conj(X(L, i)) * X(L, j);

    // V4(ij, kl) += Σ_{ΛΣ} A(Λ,i,j) V(Λ,Σ) A(Σ,k,l)
    for (long i=0; i<nbnd; ++i)
    for (long j=0; j<nbnd; ++j)
    for (long k=0; k<nbnd; ++k)
    for (long l=0; l<nbnd; ++l) {
        ComplexType acc(0.0);
        for (long L=0; L<N; ++L)
        for (long S=0; S<N; ++S)
            acc += A(L, i, j) * V(L, S) * A(S, k, l);
        V4(i, j, k, l) = acc;
    }
}

/**
 * E_H via direct contraction of materialized V_{ijkl}:
 *   E_H = (1/2) Σ_{ij} f_i f_j V_{iijj}
 */
template<typename V4_t, typename occ_t>
inline double hartree_energy_direct(V4_t const& V4, occ_t const& occ)
{
    long nbnd = V4.extent(0);
    utils::check(occ.extent(0) == nbnd, "occ size mismatch");
    ComplexType acc(0.0);
    for (long i=0; i<nbnd; ++i)
    for (long j=0; j<nbnd; ++j)
        acc += occ(i) * occ(j) * V4(i, i, j, j);
    return 0.5 * std::real(acc);
}

/**
 * E_x via direct contraction of materialized V_{ijkl}:
 *   E_x = -(1/2) Σ_{ij} f_i f_j V_{ijji}
 */
template<typename V4_t, typename occ_t>
inline double exchange_energy_direct(V4_t const& V4, occ_t const& occ)
{
    long nbnd = V4.extent(0);
    utils::check(occ.extent(0) == nbnd, "occ size mismatch");
    ComplexType acc(0.0);
    for (long i=0; i<nbnd; ++i)
    for (long j=0; j<nbnd; ++j)
        acc += occ(i) * occ(j) * V4(i, j, j, i);
    return -0.5 * std::real(acc);
}

/**
 * Convenience: cross-check both E_H paths and both E_x paths on a single
 * (X, V) factorization at one (s, k). Throws if violation > tol.
 *
 * This is the algebra-equivalence test: the two evaluation paths must agree
 * to machine epsilon for any (X, V) that came out of thc_reader_t::augment_thc_with_paw
 * (or any other valid THC factorization). A nonzero violation indicates a
 * bug in the contraction wiring (most likely in `materialize_eri` for the
 * direct path or in the closed-form helpers).
 */
template<typename X_t, typename V_t, typename occ_t>
struct EnergyCheckResult {
    double E_H_thc      = 0.0;
    double E_H_direct   = 0.0;
    double E_x_thc      = 0.0;
    double E_x_direct   = 0.0;
    double max_violation = 0.0;
};

template<typename X_t, typename V_t, typename occ_t>
inline EnergyCheckResult<X_t, V_t, occ_t>
cross_check_energies(X_t const& X, V_t const& V, occ_t const& occ,
                     double tol = 1e-10)
{
    EnergyCheckResult<X_t, V_t, occ_t> r;
    r.E_H_thc    = hartree_energy_thc(X, V, occ);
    r.E_x_thc    = exchange_energy_thc(X, V, occ);

    long nbnd = X.extent(1);
    nda::array<ComplexType,4> V4(nbnd, nbnd, nbnd, nbnd);
    materialize_eri(X, V, V4);
    r.E_H_direct = hartree_energy_direct(V4, occ);
    r.E_x_direct = exchange_energy_direct(V4, occ);

    r.max_violation = std::max(std::abs(r.E_H_thc - r.E_H_direct),
                                std::abs(r.E_x_thc - r.E_x_direct));
    utils::check(r.max_violation < tol,
        "cross_check_energies: ΔE_H={}, ΔE_x={}, max_violation={} > tol={}",
        std::abs(r.E_H_thc - r.E_H_direct),
        std::abs(r.E_x_thc - r.E_x_direct),
        r.max_violation, tol);
    return r;
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_ENERGY_CHECK_HPP
