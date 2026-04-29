/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Per-species local channel ISDF for PAW (Phase 3, .tex §IV.A) — full-rank
 * symmetric-pair decomposition in the nh-projector basis.
 *
 * For each PAW species, the augmentation pair-density Q̂_{a,IJ}(s) is
 * expanded as
 *
 *   Q̂_{a,IJ}(s) ≈ Σ_λ U_{a,λI} U_{a,λJ} η_{a,λ}(s)             (Eq. local-isdf-qhat)
 *
 * with U_{a,λI} a real (nλ × nh) matrix. In the nh basis, every (I,J)
 * symmetric pair index appears in QE's qgm tensor (qvan2 already does the
 * (αβ,L) → (I,J) Gaunt expansion), so we don't need to re-derive the angular
 * (L,M) selection rules here — the augmentation is just the qgm[nt, ij, G]
 * lookup with ij = ijtoh(nt, I, J) − 1.
 *
 * ── Construction (full-rank, exact) ──────────────────────────────────────
 *
 * For each diagonal pair (I, I)   (nh of them):
 *   one λ row:  U(λ, I) = 1,                           sign(λ) = +1
 *
 * For each off-diagonal pair I < J  (nh(nh-1)/2 of them):
 *   two λ rows (rank-2 spectral split of e_I e_J^T + e_J e_I^T):
 *     λ_+:  U(λ, I) = U(λ, J) = +1/√2,                  sign(λ) = +1
 *     λ_−:  U(λ, I) = +1/√2,  U(λ, J) = −1/√2,           sign(λ) = −1
 *
 * Total nλ = nh + 2·nh(nh−1)/2 = nh².
 *
 * The "sign" is absorbed into η so that the .tex Eq.(local-isdf-qhat) holds
 * with no auxiliary sign vector visible to consumers:
 *   η_{a,λ}(s)  ≡  sign(λ) · q̂_{a, (i(λ),j(λ))}(s)
 *
 * Round-trip identity (verified algebraically):
 *   For any function g symmetric in its two args,
 *     Σ_λ sign(λ) U_{λI'} U_{λJ'} g(i(λ), j(λ))  =  g(I', J')             (★)
 *
 * (★) makes both
 *   (a) Q̂ reconstruction: Σ_λ U_{λI} U_{λJ} η_λ = Σ_λ sign U_{λI} U_{λJ} q̂_{i(λ)j(λ)} = q̂_{IJ}
 *   (b) ΔC factorization: see compute_K_a_full_rank below
 * exact in the full-rank limit (no compression, no fitting).
 *
 * ── K_a in closed form ───────────────────────────────────────────────────
 *
 *   ΔC_{IJKL} = Σ_{λξ} U_{λI} U_{λJ} K_{λξ} U_{ξK} U_{ξL}                (.tex Eq. local-k-factorization)
 *
 * Applying (★) twice gives the closed form:
 *   K_{λξ} = sign(λ) · sign(ξ) · ΔC[i(λ), j(λ), i(ξ), j(ξ)]
 *
 * Compression via pivoted Cholesky / ID over the training set B_{a,αβ} of
 * Eq.(local-training-set), and Coulomb-metric refinement of η_λ, are explicit
 * follow-ups. The data path here lets a consumer transparently swap in those
 * variants without changing the assemble code.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_LOCAL_ISDF_HPP
#define HAMILTONIAN_PAW_LOCAL_ISDF_HPP

#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Per-species local ISDF data (one entry per PAW species).
 *
 * Layout convention:
 *   λ ∈ [0, nλ) enumerates symmetric-pair rows in nh-projector space:
 *     - one row per diagonal (I, I)
 *     - two rows per off-diagonal (I < J) — the "+" and "−" spectral pieces
 *
 *   U(λ, I) is real, sparse-encoded as dense (one or two nonzero entries).
 *
 *   lambda_i(λ) ≤ lambda_j(λ) carry the projector pair attached to λ.
 *
 *   lambda_sign(λ) ∈ {+1, −1} is the sign absorbed into η so that
 *   Σ_λ U_{λI} U_{λJ} η_λ reproduces Q̂_{IJ} without a separate sign field
 *   in consumer code.
 *
 *   eta_qg_q0(λ, g): η_{a,λ}(G) at q=0 on the dense G grid, with the sign
 *   already baked in:
 *     eta_qg_q0(λ, g) = lambda_sign(λ) · qgm(nt, ij_pair(λ), g)
 *
 *   Atom structure factors e^{-iG·τ_a} are NOT included; consumers apply
 *   them at the call site (since η is species-shared, not atom-shared).
 */
struct species_local_isdf {
    int nlambda = 0;
    int nh      = 0;

    nda::array<double, 2>      U;             // (nlambda, nh)
    nda::array<int, 1>         lambda_i;      // (nlambda) — first proj index
    nda::array<int, 1>         lambda_j;      // (nlambda) — second proj index
    nda::array<double, 1>      lambda_sign;   // (nlambda) — +1 or -1
    nda::array<int, 1>         lambda_ij;     // (nlambda) — ij = ijtoh(nt,I,J)-1
                                              //  (cached for fast eta lookup)

    nda::array<ComplexType, 2> eta_qg_q0;     // (nlambda, ngm_dense)
};

/**
 * Build the full-rank, symmetric-pair local ISDF for a single PAW species.
 *
 *   - Sets U, lambda_{i,j,sign,ij}.
 *   - Populates eta_qg_q0(λ, g) = sign(λ) · qgm(nt, ij_pair(λ), g)
 *
 * Returns an empty struct if the species is not PAW or has no projectors.
 *
 * Parameters:
 *   is_paw : species PAW flag (true → fill, false → return empty).
 *   nt     : species index into pseudopot's nh / ijtoh / qgm arrays.
 *   nh_a   : nh for this species (== psp.nh(nt)).
 *   ijtoh  : (nsp, nhm, nhm) — 1-based map (I,J) → ij; subtract 1 before use.
 *   qgm    : (nsp, nij_max, ngm_dense) — Q^{IJ}(G) on the dense grid.
 */
inline species_local_isdf build_local_isdf_full_rank(
    bool is_paw,
    int nt, int nh_a,
    nda::array<int, 3> const& ijtoh,
    nda::ArrayOfRank<3> auto const& qgm)
{
    species_local_isdf out;
    if (!is_paw)        return out;
    if (nh_a <= 0)      return out;
    if (qgm.size() == 0 || ijtoh.size() == 0) return out;

    long ngm = qgm.extent(2);
    long nlam = static_cast<long>(nh_a) * static_cast<long>(nh_a);

    out.nlambda     = (int)nlam;
    out.nh          = nh_a;
    out.U           = nda::array<double,2>::zeros({nlam, (long)nh_a});
    out.lambda_i    = nda::array<int,1>::zeros({nlam});
    out.lambda_j    = nda::array<int,1>::zeros({nlam});
    out.lambda_sign = nda::array<double,1>::zeros({nlam});
    out.lambda_ij   = nda::array<int,1>::zeros({nlam});
    out.eta_qg_q0   = nda::array<ComplexType,2>::zeros({nlam, ngm});

    constexpr double inv_sqrt2 = 0.7071067811865475244; // 1/√2

    long lam = 0;
    // Diagonal rows: one per I.
    for (int I = 0; I < nh_a; ++I) {
        out.U(lam, I)        = 1.0;
        out.lambda_i(lam)    = I;
        out.lambda_j(lam)    = I;
        out.lambda_sign(lam) = +1.0;
        int ij = ijtoh(nt, I, I) - 1;
        utils::check(ij >= 0,
            "build_local_isdf_full_rank: ijtoh(nt={}, I={}, I={}) is invalid",
            nt, I, I);
        out.lambda_ij(lam) = ij;
        for (long g = 0; g < ngm; ++g)
            out.eta_qg_q0(lam, g) = qgm(nt, ij, g);   // sign = +1
        ++lam;
    }
    // Off-diagonal rows: two per I < J (λ_+ then λ_−).
    for (int I = 0; I < nh_a; ++I)
    for (int J = I + 1; J < nh_a; ++J) {
        int ij = ijtoh(nt, I, J) - 1;
        utils::check(ij >= 0,
            "build_local_isdf_full_rank: ijtoh(nt={}, I={}, J={}) is invalid",
            nt, I, J);

        // λ_+ : U(I) = U(J) = 1/√2, sign = +1, η = +qgm
        out.U(lam, I)        = inv_sqrt2;
        out.U(lam, J)        = inv_sqrt2;
        out.lambda_i(lam)    = I;
        out.lambda_j(lam)    = J;
        out.lambda_sign(lam) = +1.0;
        out.lambda_ij(lam)   = ij;
        for (long g = 0; g < ngm; ++g)
            out.eta_qg_q0(lam, g) = qgm(nt, ij, g);
        ++lam;

        // λ_− : U(I) = +1/√2, U(J) = −1/√2, sign = −1, η = −qgm
        out.U(lam, I)        = +inv_sqrt2;
        out.U(lam, J)        = -inv_sqrt2;
        out.lambda_i(lam)    = I;
        out.lambda_j(lam)    = J;
        out.lambda_sign(lam) = -1.0;
        out.lambda_ij(lam)   = ij;
        for (long g = 0; g < ngm; ++g)
            out.eta_qg_q0(lam, g) = -qgm(nt, ij, g);
        ++lam;
    }
    utils::check(lam == nlam,
        "build_local_isdf_full_rank: row count mismatch (got {}, expected {})",
        lam, nlam);
    return out;
}

/**
 * Build local ISDF for every species in a pseudopot. Empty entries for
 * non-PAW species (NCPP / USPP without one-center data).
 */
inline std::vector<species_local_isdf> build_local_isdf(pseudopot const& psp)
{
    auto const& sps   = psp.paw_species_view();
    auto const& ijtoh = psp.ijtoh_view();
    auto qgm          = psp.qgm_view();
    auto const& nh_v  = psp.nh_view();
    std::vector<species_local_isdf> out;
    out.reserve(sps.size());
    long nsp = (long)sps.size();
    for (long nt = 0; nt < nsp; ++nt) {
        // Build for any species with augmentation (PAW or USPP). The ISDF
        // construction itself only needs qgm + ijtoh + nh; PAW-specific data
        // (deltaC, qfuncl) is consumed elsewhere (compute_K_a_full_rank).
        bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
        out.push_back(build_local_isdf_full_rank(
            has_aug, (int)nt, nh_v(nt), ijtoh, qgm));
    }
    return out;
}

/**
 * Closed-form K_a in the symmetric-pair basis (.tex Eq.(local-k-factorization)):
 *
 *   K_{λξ}  =  sign(λ) · sign(ξ) · ΔC[ i(λ), j(λ), i(ξ), j(ξ) ]
 *
 * Verified by applying the round-trip identity (★) twice:
 *   Σ_{λξ} U_{λI} U_{λJ} K_{λξ} U_{ξK} U_{ξL}
 *      = Σ_λ sign(λ) U_{λI} U_{λJ} ΔC[ i(λ),j(λ), K, L ]
 *      = ΔC[I, J, K, L]                                  (full-rank)
 *
 * The formula is keyed only on (lambda_i, lambda_j, lambda_sign) and the
 * raw ΔC tensor — it works UNCHANGED for the compressed isdf produced by
 * `local_isdf_compress.hpp`. In that case the round-trip is exact only on
 * (I,J,K,L) such that both (I,J) and (K,L) lie in the kept-pair set; ΔC
 * entries indexed by dropped pairs are reconstructed as zero.
 *
 * Returns a (nλ, nλ) real matrix.
 */
inline nda::array<double, 2> compute_K_a(
    species_local_isdf const& isdf,
    nda::ArrayOfRank<4> auto const& deltaC)
{
    long nlam = isdf.nlambda;
    nda::array<double,2> K = nda::array<double,2>::zeros({nlam, nlam});
    if (nlam == 0 || deltaC.size() == 0) return K;
    long n = deltaC.extent(0);
    utils::check(deltaC.extent(1) == n &&
                 deltaC.extent(2) == n &&
                 deltaC.extent(3) == n,
                 "compute_K_a: deltaC must be (nh, nh, nh, nh)");
    utils::check(isdf.nh <= (int)n,
                 "compute_K_a: isdf.nh={} > deltaC.nh={}",
                 isdf.nh, (int)n);
    for (long lam = 0; lam < nlam; ++lam) {
        int il = isdf.lambda_i(lam);
        int jl = isdf.lambda_j(lam);
        double sl = isdf.lambda_sign(lam);
        for (long xi = 0; xi < nlam; ++xi) {
            int ix = isdf.lambda_i(xi);
            int jx = isdf.lambda_j(xi);
            double sx = isdf.lambda_sign(xi);
            K(lam, xi) = sl * sx * deltaC(il, jl, ix, jx);
        }
    }
    return K;
}

/**
 * Backward-compat alias for the previous name. The "full_rank" suffix was
 * misleading: the same closed form holds for any compressed isdf produced
 * by the symmetric-pair construction. New code should use `compute_K_a`.
 */
template<class DC>
[[deprecated("use compute_K_a — works for both full-rank and compressed isdf")]]
inline nda::array<double, 2> compute_K_a_full_rank(
    species_local_isdf const& isdf, DC const& deltaC)
{
    return compute_K_a(isdf, deltaC);
}

/**
 * Compute the augmentation Y features for one species, single (s, k):
 *   Y^k_{aλ,n} = Σ_I U_{a,λI} P^k_{n,aI}
 *
 * Inputs:
 *   U      : (nlambda, nh)   — the species's U matrix (dense, mostly sparse)
 *   P_sub  : (nh*npol, nbnd) — Pskna(s, k, ofs(a)*npol : (ofs(a)+nh)*npol, all)
 *   npol   : 1 (collinear) or 2 (SOC/noncollinear; spinor index merged into nbnd axis)
 * Output:
 *   Y_out  : (nlambda, nbnd*npol) — written, not added.
 */
inline void compute_Y_features_single(
    nda::ArrayOfRank<2> auto const& U,
    nda::ArrayOfRank<2> auto const& P_sub,
    int npol,
    nda::ArrayOfRank<2> auto       & Y_out)
{
    long nlam = U.extent(0);
    long nh   = U.extent(1);
    long nbnd = P_sub.extent(1);
    utils::check(P_sub.extent(0) == nh*npol,
        "compute_Y_features_single: P slice shape mismatch ({} vs {})",
        P_sub.extent(0), nh*npol);
    utils::check(Y_out.extent(0) == nlam, "Y_out lambda dim mismatch");
    utils::check(Y_out.extent(1) == nbnd*npol, "Y_out band*spinor mismatch");

    Y_out() = ComplexType(0.0);
    for (long lam = 0; lam < nlam; ++lam)
        for (long I = 0; I < nh; ++I) {
            double u = U(lam, I);
            if (u == 0.0) continue;
            for (long ib = 0; ib < nbnd; ++ib)
                for (int s = 0; s < npol; ++s)
                    Y_out(lam, ib*npol + s) +=
                        ComplexType(u) * P_sub(I*npol + s, ib);
        }
}

/**
 * Roundtrip diagnostic: max |Q̂_{IJ}(g) − Σ_λ U_{λI} U_{λJ} η_λ(g)| over
 * (I, J, g). Should be at machine precision in the full-rank construction.
 *
 * Pulls reference Q̂ from psp.qgm_view() at species `nt`.
 */
inline double qhat_roundtrip_max_error(
    pseudopot const& psp, int nt,
    species_local_isdf const& isdf)
{
    if (isdf.nlambda == 0) return 0.0;
    auto qgm = psp.qgm_view();
    auto const& ijtoh = psp.ijtoh_view();
    int nh_a = isdf.nh;
    long ngm = qgm.extent(2);

    double max_err = 0.0;
    for (int I = 0; I < nh_a; ++I)
    for (int J = 0; J < nh_a; ++J) {
        int ij_ref = ijtoh(nt, I, J) - 1;
        if (ij_ref < 0) continue;
        for (long g = 0; g < ngm; ++g) {
            ComplexType acc(0.0);
            for (long lam = 0; lam < isdf.nlambda; ++lam) {
                double w = isdf.U(lam, I) * isdf.U(lam, J);
                if (w == 0.0) continue;
                acc += ComplexType(w) * isdf.eta_qg_q0(lam, g);
            }
            ComplexType ref = qgm(nt, ij_ref, g);
            max_err = std::max(max_err, std::abs(acc - ref));
        }
    }
    return max_err;
}

/**
 * Roundtrip diagnostic: max |ΔC[I,J,K,L] − Σ_{λξ} U_{λI} U_{λJ} K_{λξ}
 *                                                  U_{ξK} U_{ξL}|.
 * Should be at machine precision in the full-rank construction.
 *
 * `K` is typically the output of compute_K_a_full_rank.
 */
inline double deltaC_roundtrip_max_error(
    species_local_isdf const& isdf,
    nda::ArrayOfRank<4> auto const& deltaC,
    nda::ArrayOfRank<2> auto const& K)
{
    if (isdf.nlambda == 0 || deltaC.size() == 0) return 0.0;
    int nh_a = isdf.nh;
    double max_err = 0.0;
    for (int I = 0; I < nh_a; ++I)
    for (int J = 0; J < nh_a; ++J)
    for (int Kp = 0; Kp < nh_a; ++Kp)
    for (int L = 0; L < nh_a; ++L) {
        double acc = 0.0;
        for (long lam = 0; lam < isdf.nlambda; ++lam) {
            double wl = isdf.U(lam, I) * isdf.U(lam, J);
            if (wl == 0.0) continue;
            for (long xi = 0; xi < isdf.nlambda; ++xi) {
                double wx = isdf.U(xi, Kp) * isdf.U(xi, L);
                if (wx == 0.0) continue;
                acc += wl * K(lam, xi) * wx;
            }
        }
        max_err = std::max(max_err, std::abs(acc - deltaC(I, J, Kp, L)));
    }
    return max_err;
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_LOCAL_ISDF_HPP
