/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * PAW-ISDF-THC kernel assembly (Phase 4, .tex §IV).
 *
 * Combines the existing smooth ISDF factorization (X, V) for pseudo pair
 * densities with atom-local PAW augmentation features (Y, η, K_a) to
 * produce the composite all-electron THC factorization
 *
 *   V_{ijkl}^{k_i k_j k_k k_l, PAW}
 *     = Σ_{ΛΣ} X*_{Λi}^{k_i} X_{Λj}^{k_j}
 *               𝒱_{ΛΣ}^q
 *               X*_{Σk}^{k_k} X_{Σl}^{k_l}
 *
 * (.tex Eq. final-paw-isdf-thc), where
 *
 *   X_{Λ,i}^k = ψ̃_i(r_μ)              for Λ = μ ∈ G   (smooth)
 *             = Σ_α U_{a,λα} P_{i,aα}^k  for Λ = (a,λ) ∈ A (augmentation)
 *
 *   𝒱_{ΛΣ}^q = V_{ΛΣ}^{q,C}
 *             + δ_{Λ,aλ} δ_{Σ,aξ} K_{a,λξ}      (.tex Eq. kernel-final)
 *
 * Block decomposition of the projected Coulomb matrix V^{q,C}:
 *   V_{μν}^{q,C}        = ⟨ζ_μ^{-q} | ζ_ν^q⟩          (G-G; existing thc.cpp)
 *   V_{μ,aλ}^{q,C}      = ⟨ζ_μ^{-q} | η_{aλ}^q⟩       (G-L; q=0 here)
 *   V_{aλ,bξ}^{q,C}     = ⟨η_{aλ}^{-q} | η_{bξ}^q⟩    (L-L; q=0 here)
 *
 * Phase 4 minimum-viable scope:
 *   * X_full / V_full data layout pinned.
 *   * Y rows of X built from pseudopot's Pskna and species_local_isdf::U.
 *   * K_a addition uses the closed-form full-rank expression
 *       K_{λξ} = sign(λ)·sign(ξ)·ΔC[i(λ),j(λ),i(ξ),j(ξ)]
 *     (.tex Eq. local-k-factorization, exact in the symmetric-pair full-rank).
 *   * G-L and L-L Coulomb blocks: implemented at q = 0 only, using the
 *     pre-baked η_{a,λ}(G) = sign(λ)·qgm[ij(λ),G] from species_local_isdf.
 *   * q ≠ 0 path requires η at q+G via radial spherical-Bessel transform —
 *     `eta_to_G.hpp` carries the helper for that follow-up.
 *   * Hermiticity check at q=0 (.tex Eq. eri-hermitian) provided.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_THC_KERNEL_HPP
#define HAMILTONIAN_PAW_PAW_THC_KERNEL_HPP

#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/local_isdf.hpp"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Augmented THC factorization. Wraps the smooth (X_smooth, V_smooth_q)
 * factorization with the augmentation block (Y, V_GL_q, V_LL_q + K_a).
 *
 * Composite Λ index layout:
 *   Λ = 0 .. N_μ-1            : smooth ISDF rows (G)
 *   Λ = N_μ .. N_μ+N_A-1      : atom-local rows (A)
 * For atom a with λ_a ∈ [0, nlam(t(a))),
 *   composite_index = N_μ + Σ_{a' < a} nlam(t(a')) + λ_a
 */
struct PAWAugmentedTHC {
    int N_mu = 0;
    int N_A  = 0;
    int N_Lambda = 0;

    nda::array<ComplexType,4> X_full;   // (s, k, Λ, i)
    nda::array<ComplexType,3> V_full;   // (q, Λ, Σ)

    std::vector<int> atom_aug_offset;   // first aug row offset (from N_mu) per atom
    std::vector<int> atom_aug_count;    // nlam(t(atom)) per atom
};

namespace detail {

inline void compute_aug_layout(pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    std::vector<int>& offsets, std::vector<int>& counts, int& N_A)
{
    long nat = psp.ityp_view().extent(0);
    auto const& ityp = psp.ityp_view();
    offsets.assign(nat, 0);
    counts.assign(nat, 0);
    int total = 0;
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        int n  = (nt < (int)isdf.size()) ? isdf[nt].nlambda : 0;
        offsets[ia] = total;
        counts[ia]  = n;
        total += n;
    }
    N_A = total;
}

} // namespace detail

/**
 * Assemble (X_full, V_full) from a smooth-only THC and pseudopot's
 * augmentation data.
 *
 * Inputs:
 *   X_smooth(s, k, μ, i)   : standard ISDF collocation matrix
 *   V_smooth(q, μ, ν)      : standard projected Coulomb matrix at all q
 *   psp.Pskna_view()       : (s, k, ofs(a)*npol+α*npol+σ, i) projector overlaps
 *   isdf                   : per-species local ISDF U / η
 *
 * Outputs (in `out`):
 *   * X_full: smooth rows = X_smooth; aug rows Y_{aλ,i} = Σ_I U_{a,λI} P_{i,aI}.
 *   * V_full: G-G block = V_smooth.
 *   * Same-atom K_a addition to L-L blocks at every q (atom-local, q-independent).
 *   * G-L and L-L Coulomb blocks remain ZERO here; populate via
 *     `compute_VGL_VLL_blocks_at_q0` for q=0 work.
 */
inline PAWAugmentedTHC assemble_paw_augmented_thc(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    nda::ArrayOfRank<4> auto const& X_smooth,
    nda::ArrayOfRank<3> auto const& V_smooth,
    int npol = 1)
{
    PAWAugmentedTHC out;

    int nspin = X_smooth.extent(0);
    int nk    = X_smooth.extent(1);
    int N_mu  = X_smooth.extent(2);
    int nbnd  = X_smooth.extent(3);
    int Nq    = V_smooth.extent(0);
    utils::check(V_smooth.extent(1) == N_mu && V_smooth.extent(2) == N_mu,
        "assemble_paw_augmented_thc: V_smooth must be (Nq, N_mu, N_mu); got "
        "({}, {}, {}) vs N_mu={}", Nq, V_smooth.extent(1), V_smooth.extent(2), N_mu);

    int N_A = 0;
    detail::compute_aug_layout(psp, isdf, out.atom_aug_offset,
                                out.atom_aug_count, N_A);

    out.N_mu = N_mu;
    out.N_A  = N_A;
    out.N_Lambda = N_mu + N_A;

    // --- X_full assembly ---------------------------------------------------
    out.X_full = nda::array<ComplexType,4>::zeros(
        {nspin, nk, out.N_Lambda, nbnd});
    for (int s=0; s<nspin; ++s)
        for (int k=0; k<nk; ++k)
            for (int mu=0; mu<N_mu; ++mu)
                for (int i=0; i<nbnd; ++i)
                    out.X_full(s, k, mu, i) = X_smooth(s, k, mu, i);

    auto Pskna = psp.Pskna_view();
    auto const& ityp = psp.ityp_view();
    auto const& nh_v = psp.nh_view();
    auto const& ofs  = psp.ofs_view();
    long nat = ityp.extent(0);
    for (int s=0; s<nspin; ++s)
    for (int k=0; k<nk; ++k)
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nh_a = nh_v(nt);
        int nlam = isdf[nt].nlambda;
        if (nlam == 0 || nh_a == 0) continue;
        int row0 = N_mu + out.atom_aug_offset[ia];
        int p0   = ofs(ia) * npol;
        for (int lam=0; lam<nlam; ++lam) {
            for (int i=0; i<nbnd; ++i) {
                ComplexType acc(0.0);
                for (int I=0; I<nh_a; ++I) {
                    double u = isdf[nt].U(lam, I);
                    if (u == 0.0) continue;
                    for (int sigma=0; sigma<npol; ++sigma) {
                        acc += ComplexType(u) *
                               Pskna(s, k, p0 + I*npol + sigma, i);
                    }
                }
                out.X_full(s, k, row0 + lam, i) = acc;
            }
        }
    }

    // --- V_full assembly ---------------------------------------------------
    out.V_full = nda::array<ComplexType,3>::zeros(
        {Nq, out.N_Lambda, out.N_Lambda});
    for (int q=0; q<Nq; ++q)
        for (int mu=0; mu<N_mu; ++mu)
            for (int nu=0; nu<N_mu; ++nu)
                out.V_full(q, mu, nu) = V_smooth(q, mu, nu);

    // --- K_a same-atom L-L addition ---------------------------------------
    // K_{λξ} = sign(λ)·sign(ξ)·ΔC[i(λ),j(λ),i(ξ),j(ξ)]  (closed form)
    auto const& sps = psp.paw_species_view();
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size() || nt >= (int)sps.size())
            continue;
        auto const& sp_isdf = isdf[nt];
        auto const& sp_paw  = sps[nt];
        if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
        if (sp_isdf.nlambda == 0) continue;

        auto K = compute_K_a(sp_isdf, sp_paw.deltaC);
        int row0 = N_mu + out.atom_aug_offset[ia];
        for (int lam=0; lam<sp_isdf.nlambda; ++lam)
        for (int xi =0; xi <sp_isdf.nlambda; ++xi) {
            ComplexType k_val(K(lam, xi), 0.0);
            for (int q=0; q<Nq; ++q)
                out.V_full(q, row0 + lam, row0 + xi) += k_val;
        }
    }
    return out;
}

/**
 * Populate the V_GL and V_LL Coulomb blocks at q = 0 using the pre-baked
 * eta_qg_q0 from species_local_isdf and the dense FFT G-grid in pseudopot.
 *
 * Definitions (q=0 specialization of .tex Eq. v-reciprocal):
 *   V_{μ,aλ}^{q=0,C}    = (4π/Ω) Σ_{G≠0} ζ_μ(G) η_{aλ}^*(G) e^{+iG·τ_a}/|G|²
 *   V_{aλ,bξ}^{q=0,C}   = (4π/Ω) Σ_{G≠0} η_{aλ}^*(G) e^{+iG·τ_a}
 *                                         η_{bξ}(G)  e^{−iG·τ_b}/|G|²
 *
 * The G=0 component is dropped (charge-neutralized by the compensation
 * functions, .tex §VI Discussion); call sites for q≠0 will need a separate
 * regularization (see compute_VGL_blocks_at_q stubs in eta_to_G.hpp).
 *
 * Inputs:
 *   psp        : pseudopot with qgm/miller_g_dense/atom_pos_cart populated
 *   isdf       : per-species local ISDF (eta_qg_q0 must be populated)
 *   thc        : the augmented THC; must already have V_full sized and the
 *                G-G block (and K_a) populated by `assemble_paw_augmented_thc`
 *   zeta_mq0   : (N_mu, ngm_dense) — the smooth ISDF auxiliary functions
 *                ζ_μ(G) at q=0 on the dense G grid; caller must pre-compute
 *                this from the smooth ISDF X_smooth.
 *   recv       : reciprocal lattice vectors (3,3)
 *   omega      : cell volume Ω
 *   q_index    : which V_full slot to write (typically the q=0 slot)
 *
 * Adds to V_full (does not overwrite the K_a contribution).
 */
inline void compute_VGL_VLL_blocks_at_q0(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    PAWAugmentedTHC& thc,
    nda::ArrayOfRank<2> auto const& zeta_mq0,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    int q_index = 0)
{
    long N_mu = thc.N_mu;
    long N_A  = thc.N_A;
    if (N_A == 0) return;
    long ngm  = psp.ngm_dense_get();
    auto const& mill = psp.miller_g_dense_view();
    auto const& tau  = psp.atom_pos_cart_view();
    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    utils::check(zeta_mq0.extent(0) == N_mu, "zeta_mq0 mu-dim mismatch");
    utils::check(zeta_mq0.extent(1) == ngm,  "zeta_mq0 G-dim mismatch");
    utils::check(q_index < (int)thc.V_full.extent(0), "q_index OOR");

    // Coulomb weight 4π/(Ω|G|²) for G≠0, 0 for G=0.
    nda::array<double,1> wG(ngm);
    for (long g=0; g<ngm; ++g) {
        double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
        double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
        double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
        double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
        wG(g) = (G2 > 1e-14) ? (4.0*M_PI/(omega*G2)) : 0.0;
    }

    // Pre-compute G·τ_a phases for every atom (cos, sin) on the dense grid.
    // Memory: 2 × nat × ngm doubles — fine for typical fixtures.
    nda::array<double,2> cos_tau(nat, ngm), sin_tau(nat, ngm);
    for (long ia=0; ia<nat; ++ia) {
        for (long g=0; g<ngm; ++g) {
            double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
            double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
            double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
            double phase = Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2);
            cos_tau(ia, g) = std::cos(phase);
            sin_tau(ia, g) = std::sin(phase);
        }
    }

    auto V = thc.V_full;

    // ----------------------- G-L blocks --------------------------------
    // V_{μ, aλ}^{C} = Σ_G zeta_μ(G) · [η_{aλ}(G) e^{-iG·τ_a}]^* · 4π/(Ω|G|²)
    //              = Σ_G zeta_μ(G) · η_{aλ}^*(G) · e^{+iG·τ_a} · w(G)
    //
    // Hermitian counterpart (V_{aλ, μ}) = conj(V_{μ, aλ}).
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nlam = isdf[nt].nlambda;
        if (nlam == 0) continue;
        long row0 = N_mu + thc.atom_aug_offset[ia];
        auto const& eta = isdf[nt].eta_qg_q0;     // (nlam, ngm)
        for (int lam=0; lam<nlam; ++lam) {
            long Lam = row0 + lam;
            for (long mu=0; mu<N_mu; ++mu) {
                ComplexType acc(0.0);
                for (long g=0; g<ngm; ++g) {
                    ComplexType eta_sf =
                        eta(lam, g) *
                        ComplexType(cos_tau(ia, g), -sin_tau(ia, g));
                    acc += zeta_mq0(mu, g) * std::conj(eta_sf) * wG(g);
                }
                V(q_index, mu, Lam) += acc;
                V(q_index, Lam, mu) += std::conj(acc);
            }
        }
    }

    // ----------------------- L-L blocks --------------------------------
    // V_{aλ, bξ}^{C} = Σ_G [η_aλ(G) e^{-iG·τ_a}]^* · [η_bξ(G) e^{-iG·τ_b}] · w(G)
    //              = Σ_G η_aλ^*(G) e^{+iG·τ_a} · η_bξ(G) e^{-iG·τ_b} · w(G)
    for (long ia=0; ia<nat; ++ia) {
        int nt_a = ityp(ia);
        if (nt_a >= (int)isdf.size()) continue;
        int nlam_a = isdf[nt_a].nlambda;
        if (nlam_a == 0) continue;
        long rowa = N_mu + thc.atom_aug_offset[ia];
        auto const& eta_a = isdf[nt_a].eta_qg_q0;

        for (long ib=0; ib<nat; ++ib) {
            int nt_b = ityp(ib);
            if (nt_b >= (int)isdf.size()) continue;
            int nlam_b = isdf[nt_b].nlambda;
            if (nlam_b == 0) continue;
            long rowb = N_mu + thc.atom_aug_offset[ib];
            auto const& eta_b = isdf[nt_b].eta_qg_q0;

            for (int lam=0; lam<nlam_a; ++lam)
            for (int xi =0; xi <nlam_b; ++xi) {
                ComplexType acc(0.0);
                for (long g=0; g<ngm; ++g) {
                    // e^{+iG·(τ_a − τ_b)}
                    double cP = cos_tau(ia,g)*cos_tau(ib,g) +
                                sin_tau(ia,g)*sin_tau(ib,g);
                    double sP = sin_tau(ia,g)*cos_tau(ib,g) -
                                cos_tau(ia,g)*sin_tau(ib,g);
                    ComplexType ph(cP, sP);
                    acc += std::conj(eta_a(lam, g)) * eta_b(xi, g)
                            * ph * wG(g);
                }
                V(q_index, rowa + lam, rowb + xi) += acc;
            }
        }
    }
}

/**
 * Hermiticity check on V_full:  V^q_{ΛΣ} == [V^q_{ΣΛ}]^*  at every q.
 */
inline double validate_hermiticity(PAWAugmentedTHC const& thc,
                                    double tol = 1e-8)
{
    double max_viol = 0.0;
    int Nq = thc.V_full.extent(0);
    int N  = thc.N_Lambda;
    for (int q=0; q<Nq; ++q) {
        for (int A=0; A<N; ++A)
        for (int B=0; B<N; ++B) {
            ComplexType v  = thc.V_full(q, A, B);
            ComplexType vT = std::conj(thc.V_full(q, B, A));
            max_viol = std::max(max_viol, std::abs(v - vT));
        }
    }
    utils::check(max_viol < tol,
        "PAWAugmentedTHC: Hermiticity violation {} exceeds tol {}",
        max_viol, tol);
    return max_viol;
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_THC_KERNEL_HPP
