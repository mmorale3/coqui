/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * PAW-ISDF-THC energy-equivalence test (Phase 4 self-consistency).
 *
 * Tests the algebra of the (X_full, V_full) factorization by computing
 * Hartree and exchange energies two independent ways:
 *
 *   (1) Closed-form THC contraction:
 *         E_H = (1/2) Σ_ΛΣ D_Λ V_ΛΣ D_Σ
 *         E_x = -(1/2) Σ_ΛΣ G_ΛΣ V_ΛΣ G*_ΛΣ
 *   (2) Materialized V_{ijkl} → direct contraction with f_i f_j.
 *
 * Both must agree to machine epsilon for any valid (X, V). Disagreement
 * indicates a bug in either path.
 *
 * Synthetic test cases — no fixture dependency. Runs in <1 second.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <random>

#include "configuration.hpp"
#include "nda/nda.hpp"

#include "hamiltonian/paw/paw_energy_check.hpp"

namespace {

// Build a synthetic Hermitian V of shape (N, N) with arbitrary entries.
nda::array<ComplexType,2> random_hermitian(long N, std::mt19937& rng)
{
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    nda::array<ComplexType,2> V(N, N);
    for (long i=0; i<N; ++i)
    for (long j=i; j<N; ++j) {
        if (i == j) V(i, j) = ComplexType(u(rng), 0.0);
        else {
            ComplexType v(u(rng), u(rng));
            V(i, j) = v;
            V(j, i) = std::conj(v);
        }
    }
    return V;
}

// Build a synthetic X of shape (N, nbnd) with arbitrary complex entries.
nda::array<ComplexType,2> random_X(long N, long nbnd, std::mt19937& rng)
{
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    nda::array<ComplexType,2> X(N, nbnd);
    for (long L=0; L<N; ++L)
    for (long i=0; i<nbnd; ++i)
        X(L, i) = ComplexType(u(rng), u(rng));
    return X;
}

} // anonymous

TEST_CASE("PAW THC energy-equivalence (algebra)", "[paw][energy_check]")
{
    using namespace hamilt::paw;

    SECTION("Smooth-only THC: E_H and E_x match closed-form vs materialized")
    {
        std::mt19937 rng(0xC0DE01u);
        long N = 6, nbnd = 4;
        auto X = random_X(N, nbnd, rng);
        auto V = random_hermitian(N, rng);
        nda::array<double,1> occ({1.0, 1.0, 0.5, 0.0});
        auto r = cross_check_energies(X, V, occ, 1e-10);
        // Same numbers, two paths
        CHECK(std::abs(r.E_H_thc - r.E_H_direct) < 1e-10);
        CHECK(std::abs(r.E_x_thc - r.E_x_direct) < 1e-10);
        // Sanity: E_x is non-positive for a non-negative occupation set
        // (assuming V is positive-semidefinite as a Coulomb metric should be;
        //  the random V here is just Hermitian so E_x can be either sign.
        //  We don't assert sign).
    }

    SECTION("Augmented composite Λ = G ∪ A: same algebra holds")
    {
        // Same test with a larger Λ space mimicking the augmented (X_full,
        // V_full). Bugs in the X concatenation or K_a addition would show up
        // here as a mismatch between the two energy paths.
        std::mt19937 rng(0xBAD42u);
        long N_mu = 5, N_A = 3, nbnd = 4;
        long N = N_mu + N_A;
        auto X = random_X(N, nbnd, rng);
        auto V = random_hermitian(N, rng);
        // Add a "K_a-like" same-atom block to the L-L corner: entries on the
        // last N_A × N_A block, additive. Simulates the K_a injection in
        // hamilt::paw::add_K_a_to_LL (paw_aug_thc.hpp).
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        for (long L=N_mu; L<N; ++L)
        for (long S=N_mu; S<N; ++S) {
            ComplexType k(u(rng), 0.0);
            // make it Hermitian
            V(L, S) += k;
            if (L != S) V(S, L) += std::conj(k);
        }
        nda::array<double,1> occ({1.0, 1.0, 0.5, 0.0});
        auto r = cross_check_energies(X, V, occ, 1e-10);
        CHECK(std::abs(r.E_H_thc - r.E_H_direct) < 1e-10);
        CHECK(std::abs(r.E_x_thc - r.E_x_direct) < 1e-10);
    }

    SECTION("Zero augmentation reduces to smooth-only result")
    {
        // If we extend a smooth (X_smooth, V_smooth) with N_A zero rows of X
        // and a zero block in V, the energies must equal the smooth-only
        // computation. This validates the X concatenation in
        // thc_reader_t::augment_thc_with_paw.
        std::mt19937 rng(0x5A001u);
        long N_mu = 5, N_A = 2, nbnd = 3;
        auto X_sm = random_X(N_mu, nbnd, rng);
        auto V_sm = random_hermitian(N_mu, rng);
        nda::array<double,1> occ({1.0, 0.5, 0.0});
        double E_H_sm = hartree_energy_thc(X_sm, V_sm, occ);
        double E_x_sm = exchange_energy_thc(X_sm, V_sm, occ);

        // Build the "augmented" version with the aug rows zero
        long N = N_mu + N_A;
        nda::array<ComplexType,2> X_full(N, nbnd);
        nda::array<ComplexType,2> V_full(N, N);
        X_full() = ComplexType(0.0);
        V_full() = ComplexType(0.0);
        for (long L=0; L<N_mu; ++L) {
            for (long i=0; i<nbnd; ++i) X_full(L, i) = X_sm(L, i);
            for (long S=0; S<N_mu; ++S) V_full(L, S) = V_sm(L, S);
        }
        double E_H_full = hartree_energy_thc(X_full, V_full, occ);
        double E_x_full = exchange_energy_thc(X_full, V_full, occ);
        CHECK(std::abs(E_H_full - E_H_sm) < 1e-12);
        CHECK(std::abs(E_x_full - E_x_sm) < 1e-12);
    }

    SECTION("ERI Hermiticity: V*_{ijkl} = V_{lkji}")
    {
        // For a single (s,k) — no k-point reflection — the available
        // Hermiticity from V Hermitian is:
        //   V*_{ijkl} = Σ_ΛΣ X_Λi X*_Λj V_ΣΛ X_Σk X*_Σl
        // After reindexing the (Λ,Σ) dummies and using V Hermitian:
        //   V*_{ijkl} = Σ_ΛΣ X*_Λl X_Λk V_ΛΣ X*_Σj X_Σi  =  V_{lkji}
        // The .tex Eq. (eri-hermitian) writes V*_{ijkl}=V_{jilk} but with an
        // explicit k-point swap (k_i↔k_j, k_k↔k_l); absent that swap the
        // index permutation is (l,k,j,i) instead of (j,i,l,k).
        std::mt19937 rng(0xED1234u);
        long N = 5, nbnd = 4;
        auto X = random_X(N, nbnd, rng);
        auto V = random_hermitian(N, rng);
        nda::array<ComplexType,4> V4(nbnd, nbnd, nbnd, nbnd);
        materialize_eri(X, V, V4);
        double max_viol = 0.0;
        for (long i=0; i<nbnd; ++i)
        for (long j=0; j<nbnd; ++j)
        for (long k=0; k<nbnd; ++k)
        for (long l=0; l<nbnd; ++l) {
            ComplexType lhs = std::conj(V4(i, j, k, l));
            ComplexType rhs = V4(l, k, j, i);
            max_viol = std::max(max_viol, std::abs(lhs - rhs));
        }
        CHECK(max_viol < 1e-12);
    }
}
