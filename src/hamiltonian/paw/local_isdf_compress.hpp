/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Pivoted-Cholesky compression of the local-ISDF augmentation factorization
 * (.tex §V "Local ISDF in PAW channel space").
 *
 * Starting from the full-rank symmetric-pair construction (`local_isdf.hpp`,
 * nλ = nh²), we identify a low-rank approximation in the (αβ)-pair space:
 *
 *   qgm[ij_{IJ}, G]  ≈  Σ_{(αβ) ∈ pivots} U_{λ(αβ),I} U_{λ(αβ),J} η_{λ(αβ)}(G)
 *
 * Pivoted Cholesky on the pair-space Gram in either L² or Coulomb metric
 * picks pivots in order of residual magnitude. Because the symmetric-pair
 * construction couples ± rows of an off-diagonal pair (their η signs
 * cancel cross-contributions), compression operates at the pair level —
 * dropping a single ± row of an off-diagonal pair would break the
 * cancellation. Each kept pair contributes 1 row (diagonal) or 2 rows
 * (off-diagonal split) to the compressed U.
 *
 * Truncation error in the chosen metric:
 *   ‖Q̂ − Q̂_compressed‖²_M  =  Σ_{(γδ) dropped} ‖qgm[ij_{γδ}, ·]‖²_M
 *
 * which equals the sum of dropped diagonals from the pivoted-Cholesky run.
 * The report below tracks this curve at every pivot step.
 *
 * The "compression rank" we report is the NUMBER OF PAIRS kept (pivots),
 * which translates to nλ ≤ 2·#pivots rows.
 *
 * Future refinements (placeholders only):
 *   - Bring in pfunc/ptfunc/ΔQ training-set rows alongside qgm; the .tex's
 *     B_{a,αβ} also includes those, which would let η pick up information
 *     about the AE/PS pair densities the augmentation alone misses.
 *   - Coulomb-metric refinement of η (LS fit after pivot selection,
 *     .tex §VI Discussion final paragraph).
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_LOCAL_ISDF_COMPRESS_HPP
#define HAMILTONIAN_PAW_LOCAL_ISDF_COMPRESS_HPP

#include <cmath>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/local_isdf.hpp"
#include "nda/nda.hpp"

namespace hamilt::paw {

enum class isdf_metric {
    L2,        ///< ⟨f|g⟩ = Σ_G f*(G) g(G)
    Coulomb    ///< ⟨f|g⟩ = Σ_{G≠0} f*(G) g(G) × 4π/(Ω|G|²)
};

inline char const* metric_name(isdf_metric m)
{
    return (m == isdf_metric::L2) ? "L2" : "Coulomb";
}

/**
 * Pivoted-Cholesky output for one species.
 *
 *   - pair_pivot_order[k] = ij index of the k-th selected pair (0 ≤ ij < nij)
 *   - error_at_step[k]    = √( Σ_{ij not yet picked} d_{ij}^{(k)} )
 *                           i.e. residual norm in the chosen metric AFTER k pivots
 *                           (length n_pivots+1; index 0 = initial total norm)
 *   - ij_to_IJ_i / _j     = forward map ij → (I,J) ordered with I ≤ J;
 *                           non-existent pairs have entry -1.
 *
 *   - pivot_diag[k]       = diagonal of step k = squared L² magnitude added by pivot k
 *                           Useful as an alternative log of significance.
 */
struct isdf_compression_report {
    int nij = 0;
    isdf_metric metric = isdf_metric::L2;
    std::vector<int> pair_pivot_order;
    std::vector<int> ij_to_IJ_i;
    std::vector<int> ij_to_IJ_j;
    std::vector<double> error_at_step;
    std::vector<double> pivot_diag;
};

namespace detail {

/**
 * Build (ij ↔ I,J) map: only fill entries for valid I ≤ J pairs, leave
 * others as -1 sentinel. Asserts that each ij appears exactly once in the
 * (I≤J) iteration — i.e. that ijtoh defines a bijection between symmetric
 * (I,J) pairs and ij ∈ [0, nij). This catches PAW pseudo files where the
 * stored ij range is smaller than nh·(nh+1)/2 (e.g. when an angular
 * selection rule zeroes out a pair); the caller can then prune ij values
 * left at -1 from the pivoted-Cholesky search.
 */
inline void make_pair_map(int nt, int nh_a,
    nda::array<int,3> const& ijtoh, int nij,
    std::vector<int>& ij_to_I, std::vector<int>& ij_to_J)
{
    ij_to_I.assign(nij, -1);
    ij_to_J.assign(nij, -1);
    for (int I = 0; I < nh_a; ++I)
    for (int J = I; J < nh_a; ++J) {
        int ij = ijtoh(nt, I, J) - 1;
        if (ij >= 0 && ij < nij) {
            utils::check(ij_to_I[ij] == -1,
                "make_pair_map: ijtoh(nt={}) is not bijective: pair ({},{}) "
                "and ({},{}) both map to ij={}",
                nt, ij_to_I[ij], ij_to_J[ij], I, J, ij);
            ij_to_I[ij] = I;
            ij_to_J[ij] = J;
        }
    }
}

/**
 * Coulomb (or L²) weights w(G) per dense G grid point.
 *   L²:      w(G) = 1
 *   Coulomb: w(G) = 4π/(Ω |G|²) for G≠0, 0 at G=0
 */
inline nda::array<double,1> make_metric_weights(
    nda::ArrayOfRank<2> auto const& mill,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric)
{
    long ngm = mill.extent(0);
    nda::array<double,1> w(ngm);
    if (metric == isdf_metric::L2) {
        w() = 1.0;
        return w;
    }
    for (long g = 0; g < ngm; ++g) {
        double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
        double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
        double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
        double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
        w(g) = (G2 > 1e-14) ? (4.0*M_PI/(omega*G2)) : 0.0;
    }
    return w;
}

/**
 * Compute Gram column G_{ij,p} = Σ_g w(g) qgm*(nt, ij, g) qgm(nt, p, g)
 * for all ij (returns real part — Gram is Hermitian for our complex qgm).
 */
inline void gram_column(
    nda::ArrayOfRank<3> auto const& qgm, int nt, long pivot,
    nda::ArrayOfRank<1> auto const& w, long nij,
    nda::ArrayOfRank<1> auto       & col)
{
    long ngm = qgm.extent(2);
    col() = 0.0;
    for (long ij = 0; ij < nij; ++ij) {
        double acc_re = 0.0, acc_im = 0.0;
        for (long g = 0; g < ngm; ++g) {
            // ⟨qgm[ij] | w | qgm[p]⟩ = Σ_g w(g) (Re·Re + Im·Im) + i (Re·Im - Im·Re)
            double a_re = std::real(qgm(nt, ij, g));
            double a_im = std::imag(qgm(nt, ij, g));
            double b_re = std::real(qgm(nt, pivot, g));
            double b_im = std::imag(qgm(nt, pivot, g));
            acc_re += w(g) * (a_re*b_re + a_im*b_im);
            acc_im += w(g) * (a_re*b_im - a_im*b_re);
        }
        // Real Gram in the Hermitian sense; the imaginary residue should be
        // negligible for real-density-projecting qgm. We carry only the real
        // part since pivoted Cholesky requires a symmetric positive-(semi)
        // definite matrix.
        col(ij) = acc_re;
    }
}

} // namespace detail

/**
 * Pivoted Cholesky on the pair-space Gram of qgm, in the chosen metric.
 *
 * Stops when:
 *   - residual diagonal max falls below `tol²`   (default tol = 1e-12),
 *   - or all nij pivots have been selected.
 *
 * Implementation is matrix-free in the Gram (each column is recomputed
 * from qgm × qgm^T contractions; cost O(r·nij·ngm)). For our fixtures
 * (nij ≤ 36, ngm ~ 5×10⁴) this is well under a second per species.
 *
 * Returned report's `pair_pivot_order` lists the chosen ij indices in
 * selection order. `error_at_step` tracks the residual norm in the metric.
 */
inline isdf_compression_report pivoted_cholesky_qgm_pairs(
    pseudopot const& psp, int nt,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol = 1e-12)
{
    isdf_compression_report rep;
    rep.metric = metric;

    auto qgm        = psp.qgm_view();
    auto const& mill   = psp.miller_g_dense_view();
    auto const& ijtoh  = psp.ijtoh_view();
    auto const& nh_v   = psp.nh_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    long nij = static_cast<long>(nh_a) * (nh_a + 1) / 2;
    rep.nij = (int)nij;
    if (nh_a <= 0 || qgm.size() == 0) return rep;

    detail::make_pair_map(nt, nh_a, ijtoh, (int)nij, rep.ij_to_IJ_i, rep.ij_to_IJ_j);

    auto w = detail::make_metric_weights(mill, recv, omega, metric);
    long ngm = w.extent(0);

    // Initial diagonals d_ij = Σ_g w(g) |qgm[ij, g]|²
    nda::array<double,1> diag(nij);
    for (long ij = 0; ij < nij; ++ij) {
        double acc = 0.0;
        for (long g = 0; g < ngm; ++g) {
            double a_re = std::real(qgm(nt, ij, g));
            double a_im = std::imag(qgm(nt, ij, g));
            acc += w(g) * (a_re*a_re + a_im*a_im);
        }
        diag(ij) = acc;
    }

    // Initial residual norm = √Σ d
    double total_sq = 0.0;
    for (long ij = 0; ij < nij; ++ij) total_sq += diag(ij);
    rep.error_at_step.push_back(std::sqrt(std::max(0.0, total_sq)));

    // Cholesky factor columns built incrementally; rows = nij.
    nda::array<double,2> Lf = nda::array<double,2>::zeros({nij, nij});
    nda::array<double,1> col(nij);

    // Track picked ij's so a numerically-resurrected residual diagonal can
    // never re-select an exhausted pair. Without this, a near-singular
    // Cholesky column late in the run can perturb already-zeroed diagonals
    // back to small positive values and corrupt the pivot order.
    std::vector<bool> picked(nij, false);
    // Also drop ij's with no (I, J) assigned (selection-rule zero pairs)
    // from the search up-front, since their diag is identically zero.
    std::vector<bool> available(nij, true);
    for (long ij = 0; ij < nij; ++ij)
        if (rep.ij_to_IJ_i[ij] < 0) available[ij] = false;

    double tol2 = tol * tol;
    int n_kept = 0;
    while (n_kept < nij) {
        // Find argmax of residual diagonal among unpicked, available ij's.
        long pivot = -1;
        double max_d = 0.0;
        for (long ij = 0; ij < nij; ++ij) {
            if (picked[ij] || !available[ij]) continue;
            if (diag(ij) > max_d) { max_d = diag(ij); pivot = ij; }
        }
        if (pivot < 0 || max_d <= tol2) break;

        rep.pair_pivot_order.push_back((int)pivot);
        rep.pivot_diag.push_back(max_d);
        picked[pivot] = true;

        // Compute Gram column for pivot
        detail::gram_column(qgm, nt, pivot, w, nij, col);

        // Subtract previous Cholesky columns: G[:,p] - L[:,:k] @ L[p,:k]
        double sqrt_d = std::sqrt(max_d);
        for (long ij = 0; ij < nij; ++ij) {
            double sub = 0.0;
            for (int kk = 0; kk < n_kept; ++kk) sub += Lf(ij, kk) * Lf(pivot, kk);
            double l_new = (col(ij) - sub) / sqrt_d;
            Lf(ij, n_kept) = l_new;
            diag(ij) -= l_new * l_new;
            if (diag(ij) < 0.0) diag(ij) = 0.0;   // clamp tiny negatives
        }
        ++n_kept;

        double remaining = 0.0;
        for (long ij = 0; ij < nij; ++ij)
            if (!picked[ij] && available[ij]) remaining += diag(ij);
        rep.error_at_step.push_back(std::sqrt(std::max(0.0, remaining)));
    }

    return rep;
}

/**
 * Build the compressed local ISDF from the first `n_pairs_kept` pivots of
 * a previously-computed report. n_pairs_kept can be:
 *   - 0           → empty struct
 *   - any value ≤ report.pair_pivot_order.size()
 *   - n_pairs_kept > size → clamped to size (returns full pivot list)
 *
 * Each kept pair contributes:
 *   - 1 row to U if (I,J) diagonal,
 *   - 2 rows to U if (I,J) off-diagonal (the symmetric-pair ± split).
 *
 * eta_qg_q0 is built directly from qgm with the appropriate sign — same as
 * `build_local_isdf_full_rank`, just over the kept pivots.
 */
inline species_local_isdf build_local_isdf_compressed(
    pseudopot const& psp, int nt, int nh_a,
    isdf_compression_report const& report,
    int n_pairs_kept)
{
    species_local_isdf out;
    out.nh = nh_a;
    if (nh_a <= 0 || report.pair_pivot_order.empty()) return out;
    n_pairs_kept = std::max(0, std::min(n_pairs_kept,
                                        (int)report.pair_pivot_order.size()));
    if (n_pairs_kept == 0) return out;

    auto qgm = psp.qgm_view();
    long ngm = qgm.extent(2);

    // Count rows
    int n_rows = 0;
    for (int k = 0; k < n_pairs_kept; ++k) {
        int ij = report.pair_pivot_order[k];
        int I = report.ij_to_IJ_i[ij];
        int J = report.ij_to_IJ_j[ij];
        n_rows += (I == J) ? 1 : 2;
    }

    out.nlambda     = n_rows;
    out.U           = nda::array<double,2>::zeros({(long)n_rows, (long)nh_a});
    out.lambda_i    = nda::array<int,1>::zeros({n_rows});
    out.lambda_j    = nda::array<int,1>::zeros({n_rows});
    out.lambda_sign = nda::array<double,1>::zeros({n_rows});
    out.lambda_ij   = nda::array<int,1>::zeros({n_rows});
    out.eta_qg_q0   = nda::array<ComplexType,2>::zeros({(long)n_rows, ngm});

    constexpr double inv_sqrt2 = 0.7071067811865475244;
    int lam = 0;
    for (int k = 0; k < n_pairs_kept; ++k) {
        int ij = report.pair_pivot_order[k];
        int I = report.ij_to_IJ_i[ij];
        int J = report.ij_to_IJ_j[ij];
        utils::check(I >= 0 && J >= 0,
            "build_local_isdf_compressed: invalid pair entry for ij={}", ij);
        if (I == J) {
            out.U(lam, I)        = 1.0;
            out.lambda_i(lam)    = I;
            out.lambda_j(lam)    = I;
            out.lambda_sign(lam) = +1.0;
            out.lambda_ij(lam)   = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = qgm(nt, ij, g);
            ++lam;
        } else {
            out.U(lam, I)        = inv_sqrt2;
            out.U(lam, J)        = inv_sqrt2;
            out.lambda_i(lam)    = I;
            out.lambda_j(lam)    = J;
            out.lambda_sign(lam) = +1.0;
            out.lambda_ij(lam)   = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = qgm(nt, ij, g);
            ++lam;

            out.U(lam, I)        = +inv_sqrt2;
            out.U(lam, J)        = -inv_sqrt2;
            out.lambda_i(lam)    = I;
            out.lambda_j(lam)    = J;
            out.lambda_sign(lam) = -1.0;
            out.lambda_ij(lam)   = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = -qgm(nt, ij, g);
            ++lam;
        }
    }
    utils::check(lam == n_rows, "build_local_isdf_compressed: row count mismatch");
    return out;
}

/**
 * One-shot helper: pivoted Cholesky + build compressed at the resulting
 * tolerance-induced rank.
 *
 * Returns the (compressed isdf, compression report) pair. The report is
 * useful for downstream rank-vs-error reporting and h5 caching.
 */
inline std::pair<species_local_isdf, isdf_compression_report>
compress_local_isdf_species(
    pseudopot const& psp, int nt,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol = 1e-12)
{
    auto rep = pivoted_cholesky_qgm_pairs(psp, nt, recv, omega, metric, tol);
    auto const& nh_v = psp.nh_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    auto isdf = build_local_isdf_compressed(psp, nt, nh_a, rep,
                                            (int)rep.pair_pivot_order.size());
    return {std::move(isdf), std::move(rep)};
}

/**
 * Build a `species_local_isdf` that includes every (αβ) pair whose initial
 * (un-Cholesky-residual) norm in the chosen metric exceeds `tol`.
 *
 * Use this when you want EXACT compressed reconstruction at the picked
 * rank (modulo numerically-zero pairs). Pivoted Cholesky on the (αβ) Gram
 * picks pivots that span the row space, but our compressed reconstruction
 * does not exploit linear combinations (no LS refit of η) — so a pair that
 * is linearly dependent on the picked set is still NEEDED at its own (I,J)
 * slot, otherwise the reconstruction at (I,J) is identically zero.
 *
 * In practice: tol² = 1e-30 (default) keeps all pairs with non-trivial
 * augmentation. Looser tol drops pairs whose augmentation is below noise
 * in the chosen metric.
 */
inline species_local_isdf build_local_isdf_compressed_by_norm(
    pseudopot const& psp, int nt,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol = 1e-15)
{
    species_local_isdf out;
    auto const& nh_v = psp.nh_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    out.nh = nh_a;
    if (nh_a <= 0) return out;
    auto qgm = psp.qgm_view();
    if (qgm.size() == 0) return out;

    auto const& ijtoh = psp.ijtoh_view();
    auto const& mill  = psp.miller_g_dense_view();
    long nij = static_cast<long>(nh_a)*(nh_a+1)/2;
    long ngm = qgm.extent(2);
    auto w = detail::make_metric_weights(mill, recv, omega, metric);

    std::vector<int> ij_to_I, ij_to_J;
    detail::make_pair_map(nt, nh_a, ijtoh, (int)nij, ij_to_I, ij_to_J);

    double tol2 = tol*tol;
    std::vector<int> kept_ij;
    kept_ij.reserve(nij);
    for (long ij = 0; ij < nij; ++ij) {
        if (ij_to_I[ij] < 0) continue;
        double d = 0.0;
        for (long g = 0; g < ngm; ++g) {
            double a_re = std::real(qgm(nt, ij, g));
            double a_im = std::imag(qgm(nt, ij, g));
            d += w(g) * (a_re*a_re + a_im*a_im);
        }
        if (d > tol2) kept_ij.push_back((int)ij);
    }

    int n_rows = 0;
    for (int ij : kept_ij)
        n_rows += (ij_to_I[ij] == ij_to_J[ij]) ? 1 : 2;
    out.nlambda     = n_rows;
    out.U           = nda::array<double,2>::zeros({(long)n_rows, (long)nh_a});
    out.lambda_i    = nda::array<int,1>::zeros({n_rows});
    out.lambda_j    = nda::array<int,1>::zeros({n_rows});
    out.lambda_sign = nda::array<double,1>::zeros({n_rows});
    out.lambda_ij   = nda::array<int,1>::zeros({n_rows});
    out.eta_qg_q0   = nda::array<ComplexType,2>::zeros({(long)n_rows, ngm});

    constexpr double inv_sqrt2 = 0.7071067811865475244;
    int lam = 0;
    for (int ij : kept_ij) {
        int I = ij_to_I[ij], J = ij_to_J[ij];
        if (I == J) {
            out.U(lam, I)        = 1.0;
            out.lambda_i(lam)    = I;
            out.lambda_j(lam)    = I;
            out.lambda_sign(lam) = +1.0;
            out.lambda_ij(lam)   = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = qgm(nt, ij, g);
            ++lam;
        } else {
            out.U(lam, I) = inv_sqrt2; out.U(lam, J) = inv_sqrt2;
            out.lambda_i(lam) = I; out.lambda_j(lam) = J;
            out.lambda_sign(lam) = +1.0; out.lambda_ij(lam) = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = qgm(nt, ij, g);
            ++lam;
            out.U(lam, I) = +inv_sqrt2; out.U(lam, J) = -inv_sqrt2;
            out.lambda_i(lam) = I; out.lambda_j(lam) = J;
            out.lambda_sign(lam) = -1.0; out.lambda_ij(lam) = ij;
            for (long g = 0; g < ngm; ++g) out.eta_qg_q0(lam, g) = -qgm(nt, ij, g);
            ++lam;
        }
    }
    return out;
}

/**
 * Compute the squared error (in the metric used by the report) of the
 * compressed reconstruction against the full qgm tensor for a single
 * species. Useful for the rank-vs-accuracy report.
 *
 *   err²(M)  =  Σ_{(IJ)} ‖qgm[ij_{IJ}, ·] − Σ_λ U_{λI} U_{λJ} η_λ(·)‖²_M
 *
 * In the symmetric-pair compressed scheme, this equals Σ over the dropped
 * pairs only — matches `report.error_at_step[r]²` to numerical precision.
 */
inline double compressed_qhat_error_in_metric(
    pseudopot const& psp, int nt,
    species_local_isdf const& compressed,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric)
{
    auto qgm = psp.qgm_view();
    auto const& nh_v = psp.nh_view();
    auto const& ijtoh = psp.ijtoh_view();
    auto const& mill  = psp.miller_g_dense_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    long ngm = qgm.extent(2);
    if (nh_a <= 0 || qgm.size() == 0) return 0.0;

    auto w = detail::make_metric_weights(mill, recv, omega, metric);

    double err2 = 0.0;
    for (int I = 0; I < nh_a; ++I)
    for (int J = 0; J < nh_a; ++J) {
        int ij_ref = ijtoh(nt, I, J) - 1;
        if (ij_ref < 0) continue;
        // Reconstruction at this (I, J): Σ_λ U_λI U_λJ η_λ(g)
        for (long g = 0; g < ngm; ++g) {
            ComplexType acc(0.0);
            for (long lam = 0; lam < compressed.nlambda; ++lam) {
                double w_uu = compressed.U(lam, I) * compressed.U(lam, J);
                if (w_uu == 0.0) continue;
                acc += ComplexType(w_uu) * compressed.eta_qg_q0(lam, g);
            }
            ComplexType ref = qgm(nt, ij_ref, g);
            ComplexType diff = ref - acc;
            err2 += w(g) * (std::real(diff)*std::real(diff)
                           + std::imag(diff)*std::imag(diff));
        }
    }
    // Each (I, J) and (J, I) entry is counted; halve the off-diagonal pair
    // contribution to account for the symmetric pair counting.
    // Actually: the loop visits all (I, J) including (J, I), and the
    // reconstruction is symmetric in (I, J) under our U construction. So
    // off-diagonal pairs are counted twice, diagonal once. For a "per
    // unique-pair" total error we'd halve off-diagonals; we keep the
    // doubled convention since `report.error_at_step` is in the same
    // doubled-pair convention (no, it's not — see the next note).
    //
    // Note: report.error_at_step accumulates over ij (unique pair index),
    // each ij counted once. To match, we should integrate only over unique
    // pairs. We do the unique-pair version here for clean comparison.
    // Re-do the loop with I ≤ J only:
    err2 = 0.0;
    for (int I = 0; I < nh_a; ++I)
    for (int J = I; J < nh_a; ++J) {
        int ij_ref = ijtoh(nt, I, J) - 1;
        if (ij_ref < 0) continue;
        for (long g = 0; g < ngm; ++g) {
            ComplexType acc(0.0);
            for (long lam = 0; lam < compressed.nlambda; ++lam) {
                double w_uu = compressed.U(lam, I) * compressed.U(lam, J);
                if (w_uu == 0.0) continue;
                acc += ComplexType(w_uu) * compressed.eta_qg_q0(lam, g);
            }
            ComplexType ref = qgm(nt, ij_ref, g);
            ComplexType diff = ref - acc;
            err2 += w(g) * (std::real(diff)*std::real(diff)
                           + std::imag(diff)*std::imag(diff));
        }
    }
    return std::sqrt(std::max(0.0, err2));
}

/**
 * Compute the error of the compressed K_a reconstruction of ΔC_a, in
 * (nh,nh,nh,nh) tensor space:
 *
 *   err_max  = max_{IJKL} |ΔC[I,J,K,L] − Σ_{λξ} U_{λI} U_{λJ} K_{λξ}
 *                                                  U_{ξK} U_{ξL}|
 *   err_F    = √( Σ_{IJKL} |ΔC − ΣUUKUU|² )
 *
 * For the closed-form K (compute_K_a), entries indexed by KEPT pairs are
 * exact; entries with at least one dropped pair are reconstructed as zero,
 * so the residual equals the dropped-pair contribution.
 */
struct K_a_error_report {
    double err_max = 0.0;
    double err_F   = 0.0;
};

inline K_a_error_report compressed_K_a_error(
    species_local_isdf const& isdf,
    nda::ArrayOfRank<4> auto const& deltaC,
    nda::ArrayOfRank<2> auto const& K)
{
    K_a_error_report r;
    if (isdf.nlambda == 0 || deltaC.size() == 0) return r;
    int nh_a = isdf.nh;
    double sum2 = 0.0;
    double m = 0.0;
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
        double diff = acc - deltaC(I, J, Kp, L);
        m = std::max(m, std::abs(diff));
        sum2 += diff * diff;
    }
    r.err_max = m;
    r.err_F   = std::sqrt(std::max(0.0, sum2));
    return r;
}

/**
 * Rank-vs-K_a-error row, companion to `rank_error_row`.
 */
struct K_a_rank_error_row {
    int n_pairs_kept = 0;
    int n_lambda     = 0;
    double K_err_max = 0.0;   ///< max|ΔC − ΣUUKUU| over (I,J,K,L)
    double K_err_F   = 0.0;   ///< Frobenius norm of the residual tensor
};

/**
 * Build a (rank, K_a-reconstruction-error) curve. For each cumulative
 * kept-pair count n ∈ [0, n_pivots]:
 *   - assemble compressed isdf at rank n (using pivoted-Cholesky order)
 *   - compute K_a from (compressed isdf, raw ΔC tensor)
 *   - measure ΔC reconstruction error
 *
 * Useful as a diagnostic alongside `build_rank_error_curve` (which targets
 * Q̂ reconstruction): the two curves together tell you what the smallest
 * rank is that meets a target Hartree+one-center accuracy.
 */
inline std::vector<K_a_rank_error_row> build_K_a_rank_error_curve(
    pseudopot const& psp, int nt,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol = 1e-14)
{
    std::vector<K_a_rank_error_row> out;
    auto const& sps = psp.paw_species_view();
    if (nt < 0 || nt >= (int)sps.size()) return out;
    auto const& sp_paw = sps[nt];
    if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) return out;

    auto rep = pivoted_cholesky_qgm_pairs(psp, nt, recv, omega, metric, tol);
    auto const& nh_v = psp.nh_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    int n_max = (int)rep.pair_pivot_order.size();
    out.reserve(n_max + 1);

    // n = 0 → no isdf rows → ΣUUKUU = 0 → error = ‖ΔC‖
    {
        K_a_rank_error_row r;
        r.n_pairs_kept = 0;
        r.n_lambda     = 0;
        double s2 = 0.0, mx = 0.0;
        for (int I = 0; I < nh_a; ++I)
        for (int J = 0; J < nh_a; ++J)
        for (int Kp = 0; Kp < nh_a; ++Kp)
        for (int L = 0; L < nh_a; ++L) {
            double v = sp_paw.deltaC(I, J, Kp, L);
            mx = std::max(mx, std::abs(v));
            s2 += v*v;
        }
        r.K_err_max = mx;
        r.K_err_F   = std::sqrt(std::max(0.0, s2));
        out.push_back(r);
    }
    for (int n = 1; n <= n_max; ++n) {
        auto isdf_n = build_local_isdf_compressed(psp, nt, nh_a, rep, n);
        auto K = compute_K_a(isdf_n, sp_paw.deltaC);
        auto er = compressed_K_a_error(isdf_n, sp_paw.deltaC, K);
        K_a_rank_error_row r;
        r.n_pairs_kept = n;
        r.n_lambda     = isdf_n.nlambda;
        r.K_err_max    = er.err_max;
        r.K_err_F      = er.err_F;
        out.push_back(r);
    }
    return out;
}

/**
 * Build a "rank-vs-error" curve over [1, nij] by re-building compressed
 * isdfs at every cumulative number of kept pairs. Returns (rank_kept,
 * error_metric, max_qhat_pointwise_error, nlambda).
 *
 * For diagnostic / report use: compares the predicted residual from the
 * pivoted-Cholesky run with the actual reconstructed qhat error.
 */
struct rank_error_row {
    int n_pairs_kept = 0;     ///< pivots included
    int n_lambda     = 0;     ///< actual U row count = diag-pairs + 2·offdiag-pairs
    double error_metric = 0.0;///< L²/Coulomb residual norm at this rank
    double max_qhat_err = 0.0;///< max over (I,J,G) of |qgm − reconstructed|
};

inline std::vector<rank_error_row> build_rank_error_curve(
    pseudopot const& psp, int nt,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol = 1e-14)
{
    std::vector<rank_error_row> out;
    auto rep = pivoted_cholesky_qgm_pairs(psp, nt, recv, omega, metric, tol);
    auto const& nh_v = psp.nh_view();
    int nh_a = (nt < (int)nh_v.extent(0)) ? (int)nh_v(nt) : 0;
    int n_max = (int)rep.pair_pivot_order.size();
    out.reserve(n_max + 1);

    auto qgm = psp.qgm_view();
    auto const& ijtoh = psp.ijtoh_view();

    // Step 0 (no pivots) — full residual = initial total norm
    {
        rank_error_row r;
        r.n_pairs_kept = 0;
        r.n_lambda     = 0;
        r.error_metric = (rep.error_at_step.size() > 0) ? rep.error_at_step[0] : 0.0;
        // Pointwise max with empty reconstruction = max |qgm|
        double m = 0.0;
        long ngm = qgm.extent(2);
        for (int I = 0; I < nh_a; ++I)
        for (int J = I; J < nh_a; ++J) {
            int ij = ijtoh(nt, I, J) - 1;
            if (ij < 0) continue;
            for (long g = 0; g < ngm; ++g)
                m = std::max(m, std::abs(qgm(nt, ij, g)));
        }
        r.max_qhat_err = m;
        out.push_back(r);
    }
    // Steps 1 .. n_max
    for (int n = 1; n <= n_max; ++n) {
        auto isdf_n = build_local_isdf_compressed(psp, nt, nh_a, rep, n);
        rank_error_row r;
        r.n_pairs_kept = n;
        r.n_lambda     = isdf_n.nlambda;
        r.error_metric = (n < (int)rep.error_at_step.size())
                            ? rep.error_at_step[n] : 0.0;
        // Pointwise max diff
        double m = 0.0;
        long ngm = qgm.extent(2);
        for (int I = 0; I < nh_a; ++I)
        for (int J = I; J < nh_a; ++J) {
            int ij = ijtoh(nt, I, J) - 1;
            if (ij < 0) continue;
            for (long g = 0; g < ngm; ++g) {
                ComplexType acc(0.0);
                for (long lam = 0; lam < isdf_n.nlambda; ++lam) {
                    double w_uu = isdf_n.U(lam, I) * isdf_n.U(lam, J);
                    if (w_uu == 0.0) continue;
                    acc += ComplexType(w_uu) * isdf_n.eta_qg_q0(lam, g);
                }
                m = std::max(m, std::abs(qgm(nt, ij, g) - acc));
            }
        }
        r.max_qhat_err = m;
        out.push_back(r);
    }
    return out;
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_LOCAL_ISDF_COMPRESS_HPP
