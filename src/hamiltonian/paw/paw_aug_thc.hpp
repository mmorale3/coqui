/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Glue layer between the smooth THC factorization (methods::thc) and the
 * PAW local-ISDF augmentation (hamilt::paw). The functions here remap PAW
 * augmentation data onto the thc's `truncated_g_grid` (typically the same
 * as the QE dense grid, but computed independently) and build the cross
 * (G-L) and atom-local (L-L) Coulomb blocks of V_full at a single q.
 *
 *   V_full = | V_smooth_GG    V_GL          |
 *            | V_LG = V_GL†   V_LL + K_a    |
 *
 * Both q=0 and q≠0 paths share the same builder, build_eta_on_rho_g_at_q,
 * which evaluates η_{a,λ}^q(G) directly on rho_g.g_vectors() via a
 * precomputed qrad table (see paw_aug_q_eval.hpp). Callers pass q_cart=0
 * for the q=0 case.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_AUG_THC_HPP
#define HAMILTONIAN_PAW_PAW_AUG_THC_HPP

#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "grids/g_grids.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/local_isdf.hpp"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Layout summary for the augmented composite Λ index:
 *   Λ = 0 .. N_mu-1            : smooth ISDF rows
 *   Λ = N_mu .. N_mu+N_A-1     : atom-local rows; for atom a (in psp.ityp
 *                                ordering), λ_a ∈ [0, nlam(t(a))) and the
 *                                composite row is N_mu + atom_aug_offset[a]
 *                                + λ_a.
 */
struct paw_aug_layout {
    int N_mu = 0;
    int N_A  = 0;
    int N_Lambda = 0;
    std::vector<int> atom_aug_offset;
    std::vector<int> atom_aug_count;
};

inline paw_aug_layout make_paw_aug_layout(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    int N_mu)
{
    paw_aug_layout L;
    L.N_mu = N_mu;
    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    L.atom_aug_offset.assign(nat, 0);
    L.atom_aug_count.assign(nat, 0);
    int total = 0;
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        int n  = (nt < (int)isdf.size()) ? isdf[nt].nlambda : 0;
        L.atom_aug_offset[ia] = total;
        L.atom_aug_count[ia]  = n;
        total += n;
    }
    L.N_A = total;
    L.N_Lambda = N_mu + total;
    return L;
}

/**
 * Build η_{a,λ}^q(G) on the thc rho_g grid for a specified q vector
 * (cartesian). Output shape (nat, nlam_max, ngm_rho), structure factor
 * e^{-i(q+G)·τ_a} applied. Uses CoQui-side qvan2 reimplementation
 * (paw_aug_q_eval) so no precomputed q-shifted qgm dataset is needed.
 *
 * Convention: K = q_cart + G_cart at each G of the rho_g grid; matches the
 * standard PAW η^q definition. Caller chooses q_cart sign to match the
 * Coulomb kernel convention used in V_GG (thc.icc evaluates the kernel at
 * |G - Q_thc|, so q_cart = -Q_thc(iq) here).
 *
 * `qrad_tabs[nt]` is a precomputed radial Bessel table (built once per
 * species via `build_qrad_tab`) — the runtime cost per (q, G) drops from
 * O(L × mesh) Bessel evaluations to O(L) cubic interpolation, the same
 * amortization QE uses (qrad_mod.f90 / qvan2.f90).
 */
inline nda::array<ComplexType, 3> build_eta_on_rho_g_at_q(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    grids::truncated_g_grid const& rho_g,
    std::array<double, 3> const& q_cart,
    double omega,
    aainit_tables const& aatab,
    std::vector<qrad_tab> const& qrad_tabs)
{
    long ngm_rho = rho_g.size();
    auto const& ityp = psp.ityp_view();
    auto const& sps  = psp.paw_species_view();
    long nat = ityp.extent(0);
    int nlam_max = 0;
    for (auto const& s : isdf) nlam_max = std::max(nlam_max, s.nlambda);

    nda::array<ComplexType, 3> eta_aug =
        nda::array<ComplexType, 3>::zeros({nat, (long)nlam_max, ngm_rho});
    if (nlam_max == 0) return eta_aug;

    auto const& tau    = psp.atom_pos_cart_view();
    auto const& g_cart = rho_g.g_vectors();   // (ngm_rho, 3)

    // Precompute |K| and Y_lp(K̂) at all G's once.
    nda::array<double, 1>           Kmod(ngm_rho);
    nda::array<double, 2>           Ylp_q(ngm_rho, aatab.llx);  // Y_lp(K̂) per G
    Ylp_q() = 0.0;
    int Lmax_aux = 2*(aatab.lli - 1);
    for (long g = 0; g < ngm_rho; ++g) {
        double Kx = q_cart[0] + g_cart(g, 0);
        double Ky = q_cart[1] + g_cart(g, 1);
        double Kz = q_cart[2] + g_cart(g, 2);
        double K  = std::sqrt(Kx*Kx + Ky*Ky + Kz*Kz);
        Kmod(g) = K;
        std::array<double,3> dir = (K > 1e-14)
            ? std::array<double,3>{Kx/K, Ky/K, Kz/K}
            : std::array<double,3>{0.0, 0.0, 1.0};
        nda::array<double,1> Yflat(aatab.llx);
        qe_real_ylm_flat(Lmax_aux, dir, Yflat);
        if (K <= 1e-14) {
            for (long lp = 1; lp < aatab.llx; ++lp) Yflat(lp) = 0.0;
        }
        for (long lp = 0; lp < aatab.llx; ++lp) Ylp_q(g, lp) = Yflat(lp);
    }

    // Per-species radial qrad cache + per-atom angular sum.
    long nsp = (long)sps.size();
    for (long nt = 0; nt < nsp; ++nt) {
        if (nt >= (long)isdf.size()) continue;
        auto const& s  = isdf[nt];
        if (s.nlambda == 0) continue;
        auto const& sp = sps[nt];
        if (sp.qfuncl.size() == 0 || sp.nhtolm.size() == 0) continue;

        long Lp1   = sp.qfuncl.extent(0);
        long n_ijv = sp.qfuncl.extent(1);
        // qrad_g(L, ij, g) = qrad_tab interpolated at |q+G|. Cubic 4-point
        // interpolation in iK = K/dq (matches QE's qvan2.f90:143-157).
        nda::array<double, 3> qrad_g(Lp1, n_ijv, ngm_rho);
        qrad_g() = 0.0;
        utils::check((long)qrad_tabs.size() > nt && qrad_tabs[nt].n_K > 0,
                     "build_eta_on_rho_g_at_q: missing qrad_tab for species nt={}",
                     nt);
        auto const& Tt = qrad_tabs[nt];
        for (long g = 0; g < ngm_rho; ++g) {
            for (long ijv = 0; ijv < n_ijv; ++ijv) {
                auto qrL = qrad_interp_at_K(Tt, (int)ijv, Kmod(g));
                for (long L = 0; L < Lp1; ++L) qrad_g(L, ijv, g) = qrL(L);
            }
        }

        // For each λ in this species, look up (ivl, jvl, ijv).
        // Precompute per-λ (ivl, jvl, ijv) plus the lpx/lpl/ap slice that is
        // active. Reuse across all atoms of this species.
        struct lambda_info {
            int ivl, jvl;
            int ijv;
            int n_lp;
        };
        std::vector<lambda_info> linfo(s.nlambda);
        for (int lam = 0; lam < s.nlambda; ++lam) {
            int I = s.lambda_i(lam), J = s.lambda_j(lam);
            int ivl = sp.nhtolm(I) - 1;
            int jvl = sp.nhtolm(J) - 1;
            int nb  = sp.indv(I) - 1;
            int mb  = sp.indv(J) - 1;
            int n1 = std::max(nb, mb), n2 = std::min(nb, mb);
            int ijv = (n1 * (n1 + 1)) / 2 + n2;
            linfo[lam] = {ivl, jvl, ijv, aatab.lpx(ivl, jvl)};
        }

        double pref = 4.0 * M_PI / omega;
        for (long ia = 0; ia < nat; ++ia) {
            if (ityp(ia) != (int)nt) continue;
            for (int lam = 0; lam < s.nlambda; ++lam) {
                auto const& li = linfo[lam];
                double sgn = s.lambda_sign(lam);
                for (long g = 0; g < ngm_rho; ++g) {
                    ComplexType acc(0.0, 0.0);
                    for (int k = 0; k < li.n_lp; ++k) {
                        int lp = aatab.lpl(li.ivl, li.jvl, k);
                        int L  = (int)std::floor(std::sqrt((double)lp + 1e-9));
                        ComplexType iL_factor;
                        switch (L % 4) {
                            case 0: iL_factor = ComplexType( 1.0,  0.0); break;
                            case 1: iL_factor = ComplexType( 0.0, -1.0); break;
                            case 2: iL_factor = ComplexType(-1.0,  0.0); break;
                            case 3: iL_factor = ComplexType( 0.0,  1.0); break;
                        }
                        double coeff = aatab.ap(lp, li.ivl, li.jvl)
                                     * Ylp_q(g, lp) * qrad_g(L, li.ijv, g);
                        acc += iL_factor * ComplexType(coeff, 0.0);
                    }
                    acc *= ComplexType(pref, 0.0);
                    // structure factor e^{-i(q+G)·τ_a}
                    double Kx = q_cart[0] + g_cart(g, 0);
                    double Ky = q_cart[1] + g_cart(g, 1);
                    double Kz = q_cart[2] + g_cart(g, 2);
                    double ph = -(Kx*tau(ia,0) + Ky*tau(ia,1) + Kz*tau(ia,2));
                    ComplexType sf(std::cos(ph), std::sin(ph));
                    eta_aug(ia, lam, g) = sgn * acc * sf;
                }
            }
        }
    }
    return eta_aug;
}

/**
 * Distributed-friendly variant of build_eta_on_rho_g_at_q.
 *
 *   - Computes only rows λ ∈ la_range of the global flattened (Λ, g) layout
 *     (Λ = atom_aug_offset[ia] + lam, see paw_aug_layout) and only g ∈ g_range
 *     of the rho_g grid.
 *   - Output `eta_out` is shaped (la_range.size(), g_range.size()).
 *   - Per-rank memory: la_chunk × g_chunk complex doubles.
 *
 * Used by the distributed PAW augmentation path: each MPI rank in a q-pool
 * builds its own (la, g) tile of η^q without ever materialising the full
 * (N_aug × ngm_rho) tensor on a single rank/node.
 */
inline void build_eta_on_rho_g_at_q_chunk(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    grids::truncated_g_grid const& rho_g,
    std::array<double, 3> const& q_cart,
    double omega,
    aainit_tables const& aatab,
    std::vector<qrad_tab> const& qrad_tabs,
    nda::range la_range,
    nda::range g_range,
    nda::ArrayOfRank<2> auto& eta_out)
{
    long n_la = la_range.size();
    long n_g  = g_range.size();
    if (eta_out.shape(0) != n_la || eta_out.shape(1) != n_g) {
        utils::check(false,
            "build_eta_on_rho_g_at_q_chunk: eta_out shape ({}, {}) does not match"
            " la_range ({}) × g_range ({})",
            eta_out.shape(0), eta_out.shape(1), n_la, n_g);
    }
    eta_out() = ComplexType(0.0);
    if (n_la == 0 || n_g == 0) return;

    auto const& ityp = psp.ityp_view();
    auto const& sps  = psp.paw_species_view();
    long nat = ityp.extent(0);
    auto const& tau    = psp.atom_pos_cart_view();
    auto const& g_cart = rho_g.g_vectors();

    // Map: for each global la in la_range, find (ia, lam_local).
    struct la_loc { int ia; int lam; };
    std::vector<la_loc> la_map(n_la, {-1, -1});
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nlam = isdf[nt].nlambda;
        if (nlam == 0) continue;
        long row0 = layout.atom_aug_offset[ia];
        long row1 = row0 + nlam;
        long o = std::max((long)la_range.first(), row0);
        long e = std::min((long)la_range.last(),  row1);
        for (long lg = o; lg < e; ++lg) la_map[lg - la_range.first()] = {(int)ia, (int)(lg - row0)};
    }

    // Precompute |K| and Y_lp(K̂) at the g_range subset.
    nda::array<double, 1> Kmod(n_g);
    nda::array<double, 2> Ylp_q(n_g, aatab.llx);
    Ylp_q() = 0.0;
    int Lmax_aux = 2*(aatab.lli - 1);
    for (long ig = 0; ig < n_g; ++ig) {
        long g = g_range.first() + ig;
        double Kx = q_cart[0] + g_cart(g, 0);
        double Ky = q_cart[1] + g_cart(g, 1);
        double Kz = q_cart[2] + g_cart(g, 2);
        double K  = std::sqrt(Kx*Kx + Ky*Ky + Kz*Kz);
        Kmod(ig) = K;
        std::array<double,3> dir = (K > 1e-14)
            ? std::array<double,3>{Kx/K, Ky/K, Kz/K}
            : std::array<double,3>{0.0, 0.0, 1.0};
        nda::array<double,1> Yflat(aatab.llx);
        qe_real_ylm_flat(Lmax_aux, dir, Yflat);
        if (K <= 1e-14) {
            for (long lp = 1; lp < aatab.llx; ++lp) Yflat(lp) = 0.0;
        }
        for (long lp = 0; lp < aatab.llx; ++lp) Ylp_q(ig, lp) = Yflat(lp);
    }

    // Per-species qrad cache (only on g_range).
    long nsp = (long)sps.size();
    std::vector<nda::array<double, 3>> qrad_g_cache(nsp);
    for (long nt = 0; nt < nsp; ++nt) {
        if (nt >= (long)isdf.size()) continue;
        auto const& s  = isdf[nt];
        if (s.nlambda == 0) continue;
        auto const& sp = sps[nt];
        if (sp.qfuncl.size() == 0 || sp.nhtolm.size() == 0) continue;

        long Lp1   = sp.qfuncl.extent(0);
        long n_ijv = sp.qfuncl.extent(1);
        utils::check((long)qrad_tabs.size() > nt && qrad_tabs[nt].n_K > 0,
                     "build_eta_on_rho_g_at_q_chunk: missing qrad_tab for species nt={}", nt);
        auto const& Tt = qrad_tabs[nt];

        nda::array<double, 3> qrad_g(Lp1, n_ijv, n_g);
        qrad_g() = 0.0;
        for (long ig = 0; ig < n_g; ++ig) {
            for (long ijv = 0; ijv < n_ijv; ++ijv) {
                auto qrL = qrad_interp_at_K(Tt, (int)ijv, Kmod(ig));
                for (long L = 0; L < Lp1; ++L) qrad_g(L, ijv, ig) = qrL(L);
            }
        }
        qrad_g_cache[nt] = std::move(qrad_g);
    }

    struct lambda_info { int ivl, jvl, ijv, n_lp; };
    std::vector<std::vector<lambda_info>> linfo_cache(nsp);
    for (long nt = 0; nt < nsp; ++nt) {
        if (nt >= (long)isdf.size()) continue;
        auto const& s  = isdf[nt];
        if (s.nlambda == 0) continue;
        auto const& sp = sps[nt];
        if (sp.qfuncl.size() == 0 || sp.nhtolm.size() == 0) continue;
        linfo_cache[nt].resize(s.nlambda);
        for (int lam = 0; lam < s.nlambda; ++lam) {
            int I = s.lambda_i(lam), J = s.lambda_j(lam);
            int ivl = sp.nhtolm(I) - 1;
            int jvl = sp.nhtolm(J) - 1;
            int nb  = sp.indv(I) - 1;
            int mb  = sp.indv(J) - 1;
            int n1 = std::max(nb, mb), n2 = std::min(nb, mb);
            int ijv = (n1 * (n1 + 1)) / 2 + n2;
            linfo_cache[nt][lam] = {ivl, jvl, ijv, aatab.lpx(ivl, jvl)};
        }
    }

    double pref = 4.0 * M_PI / omega;
    for (long row = 0; row < n_la; ++row) {
        auto const& lm = la_map[row];
        if (lm.ia < 0) continue;
        int ia = lm.ia;
        int lam = lm.lam;
        int nt = ityp(ia);
        if (nt >= (int)isdf.size() || nt >= (int)sps.size()) continue;
        auto const& s = isdf[nt];
        if (s.nlambda == 0) continue;
        auto const& sp = sps[nt];
        if (sp.qfuncl.size() == 0 || sp.nhtolm.size() == 0) continue;
        auto const& qrad_g = qrad_g_cache[nt];
        auto const& linfo  = linfo_cache[nt];
        auto const& li = linfo[lam];
        double sgn = s.lambda_sign(lam);
        for (long ig = 0; ig < n_g; ++ig) {
            long g = g_range.first() + ig;
            ComplexType acc(0.0, 0.0);
            for (int k = 0; k < li.n_lp; ++k) {
                int lp = aatab.lpl(li.ivl, li.jvl, k);
                int L  = (int)std::floor(std::sqrt((double)lp + 1e-9));
                ComplexType iL_factor;
                switch (L % 4) {
                    case 0: iL_factor = ComplexType( 1.0,  0.0); break;
                    case 1: iL_factor = ComplexType( 0.0, -1.0); break;
                    case 2: iL_factor = ComplexType(-1.0,  0.0); break;
                    case 3: iL_factor = ComplexType( 0.0,  1.0); break;
                }
                double coeff = aatab.ap(lp, li.ivl, li.jvl)
                             * Ylp_q(ig, lp) * qrad_g(L, li.ijv, ig);
                acc += iL_factor * ComplexType(coeff, 0.0);
            }
            acc *= ComplexType(pref, 0.0);
            double Kx = q_cart[0] + g_cart(g, 0);
            double Ky = q_cart[1] + g_cart(g, 1);
            double Kz = q_cart[2] + g_cart(g, 2);
            double ph = -(Kx*tau(ia,0) + Ky*tau(ia,1) + Kz*tau(ia,2));
            ComplexType sf(std::cos(ph), std::sin(ph));
            eta_out(row, ig) = sgn * acc * sf;
        }
    }
}

/**
 * Add the closed-form same-atom K_a contribution to a *tile* of V_LL,
 * restricted to global (la_rows, la_cols) ranges. Used by the distributed
 * augmentation path: each rank's local LL tile of _dZ has an irregular
 * row/col range (intersection of its (P, Q) chunk with [N_smooth, N_total)),
 * which doesn't match the canonical (np_P, np_Q) chunking on N_aug.
 *
 * Block-diagonal in atoms; only entries within an atom's own λ-range get
 * a non-zero K_a contribution.
 */
inline void add_K_a_to_tile(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    nda::range la_rows,
    nda::range la_cols,
    nda::ArrayOfRank<2> auto& V_LL_tile)
{
    if (la_rows.size() == 0 || la_cols.size() == 0) return;
    auto const& ityp = psp.ityp_view();
    auto const& sps  = psp.paw_species_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size() || nt >= (int)sps.size()) continue;
        auto const& sp_isdf = isdf[nt];
        auto const& sp_paw  = sps[nt];
        if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
        if (sp_isdf.nlambda == 0) continue;

        long row0 = layout.atom_aug_offset[ia];
        long row1 = row0 + sp_isdf.nlambda;
        long r_o = std::max((long)la_rows.first(), row0);
        long r_e = std::min((long)la_rows.last(),  row1);
        long c_o = std::max((long)la_cols.first(), row0);
        long c_e = std::min((long)la_cols.last(),  row1);
        if (r_e <= r_o || c_e <= c_o) continue;

        auto K = compute_K_a(sp_isdf, sp_paw.deltaC);
        for (long la_g = r_o; la_g < r_e; ++la_g) {
            int lam = (int)(la_g - row0);
            long ir = la_g - la_rows.first();
            for (long lb_g = c_o; lb_g < c_e; ++lb_g) {
                int xi = (int)(lb_g - row0);
                long ic = lb_g - la_cols.first();
                V_LL_tile(ir, ic) += ComplexType(K(lam, xi), 0.0);
            }
        }
    }
}

/**
 * Compute Coulomb weights w(G) = 4π / (Ω |G|²) on the rho_g grid; w(0) = 0
 * (Γ singular term left for the caller's divergence treatment if needed).
 */
inline nda::array<double, 1> coulomb_weights_on_rho_g(
    grids::truncated_g_grid const& rho_g, double omega)
{
    long ngm = rho_g.size();
    nda::array<double, 1> w(ngm);
    auto const& g = rho_g.g_vectors();
    for (long ig = 0; ig < ngm; ++ig) {
        double Gx = g(ig, 0), Gy = g(ig, 1), Gz = g(ig, 2);
        double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
        w(ig) = (G2 > 1e-14) ? (4.0*M_PI/(omega*G2)) : 0.0;
    }
    return w;
}

/**
 * Coulomb weights at q ≠ 0: w(G) = 4π / (Ω |q+G|²). Singularity at q+G=0
 * is set to 0 (caller handles divergent q→0 G=0 term separately, matching
 * the smooth-path divergence treatment).
 */
inline nda::array<double, 1> coulomb_weights_on_rho_g_at_q(
    grids::truncated_g_grid const& rho_g,
    std::array<double, 3> const& q_cart,
    double omega)
{
    long ngm = rho_g.size();
    nda::array<double, 1> w(ngm);
    auto const& g = rho_g.g_vectors();
    for (long ig = 0; ig < ngm; ++ig) {
        double Kx = q_cart[0] + g(ig, 0);
        double Ky = q_cart[1] + g(ig, 1);
        double Kz = q_cart[2] + g(ig, 2);
        double K2 = Kx*Kx + Ky*Ky + Kz*Kz;
        w(ig) = (K2 > 1e-14) ? (4.0*M_PI/(omega*K2)) : 0.0;
    }
    return w;
}

/**
 * Compute the V_GL block at q=0:
 *   V_GL(μ, aλ)  =  Σ_G  ζ_μ^{q=0}(G)  η_{aλ}^*(G) e^{+iG·τ_a}  ·  4π/(Ω|G|²)
 *
 * Inputs:
 *   zeta_mu_g  : (N_mu, ngm_rho) — smooth ζ_μ^{q=0}(G) (e.g. from
 *                thc.evaluate_isdf_only at q=0)
 *   eta_aug    : (nat, nlam_max, ngm_rho) — η_{aλ}(G)·e^{-iG·τ_a} as
 *                produced by build_eta_on_rho_g_at_q
 *   wG         : (ngm_rho) — Coulomb weights from coulomb_weights_on_rho_g
 *   layout     : aug layout
 *   psp        : pseudopot (used for ityp + isdf-row counts)
 *   isdf       : per-species local ISDF
 * Output (caller-owned):
 *   V_GL(μ, Λ_aug)  for Λ_aug ∈ [0, N_A); accumulated additively.
 */
/**
 * Flatten (atom, lam, g) → (Λ, g) for the (nat, nlam_max, ngm)-shaped
 * `eta_aug`. Accumulates into the caller-owned `eta_flat`. wG is NOT
 * applied; use `flatten_eta_apply_wG` for the wG-multiplied flavour.
 */
inline void flatten_eta(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    nda::ArrayOfRank<3> auto const& eta_aug,
    nda::ArrayOfRank<2> auto       & eta_flat)
{
    long N_A = layout.N_A;
    long ngm = eta_aug.extent(2);
    eta_flat() = ComplexType(0.0);
    if (N_A == 0) return;
    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nlam = isdf[nt].nlambda;
        if (nlam == 0) continue;
        long row0 = layout.atom_aug_offset[ia];
        for (int lam = 0; lam < nlam; ++lam)
            for (long g = 0; g < ngm; ++g)
                eta_flat(row0 + lam, g) = eta_aug(ia, lam, g);
    }
}

/**
 * V_GL via GEMM, taking caller-provided scratch:
 *   V_GL(μ, Λ) += Σ_g ζ(μ, g) × conj(η(Λ, g)) × wG(g)
 *              = gemm(ζ, dagger(eta_w))
 * where eta_w(Λ, g) = η(Λ, g) × wG(g) (caller fills before this call).
 */
inline void compute_VGL_q0_on_rho_g(
    paw_aug_layout const& layout,
    nda::ArrayOfRank<2> auto const& zeta_mu_g,
    nda::ArrayOfRank<2> auto const& eta_w,
    nda::ArrayOfRank<2> auto       & V_GL_out)
{
    long N_mu = layout.N_mu;
    long N_A  = layout.N_A;
    if (N_A == 0 || N_mu == 0) return;
    nda::blas::gemm(ComplexType(1.0), zeta_mu_g, nda::dagger(eta_w),
                    ComplexType(1.0), V_GL_out);
}

/**
 * Compute the V_LL block at q=0:
 *   V_LL(aλ, bξ)  =  Σ_G  η_{aλ}^*(G)e^{+iG·τ_a}  η_{bξ}(G)e^{-iG·τ_b}
 *                          ·  4π/(Ω|G|²)
 *
 * eta_aug already carries the e^{-iG·τ} factor from
 * `build_eta_on_rho_g_at_q`, so this is a straightforward conj/mult/sum.
 *
 * Output: V_LL_out(N_A, N_A); accumulated additively.
 */
/**
 * V_LL via GEMM, taking caller-provided scratch:
 *   V_LL(la, lb) += Σ_g conj(η(la, g)) × η(lb, g) × wG(g)
 *                = gemm(eta_conj, transpose(eta_w))
 * Caller is responsible for filling eta_conj = conj(η_flat) and
 * eta_w = η_flat × wG before this call.
 */
inline void compute_VLL_q0_on_rho_g(
    paw_aug_layout const& layout,
    nda::ArrayOfRank<2> auto const& eta_conj,
    nda::ArrayOfRank<2> auto const& eta_w,
    nda::ArrayOfRank<2> auto       & V_LL_out)
{
    long N_A = layout.N_A;
    if (N_A == 0) return;
    nda::blas::gemm(ComplexType(1.0), eta_conj, nda::transpose(eta_w),
                    ComplexType(1.0), V_LL_out);
}

/**
 * Add the closed-form same-atom K_a to the L-L block (q-independent).
 * Adds in-place to `V_LL_inout` (shape (N_A, N_A)).
 */
inline void add_K_a_to_LL(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    nda::ArrayOfRank<2> auto & V_LL_inout)
{
    auto const& ityp = psp.ityp_view();
    auto const& sps  = psp.paw_species_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size() || nt >= (int)sps.size()) continue;
        auto const& sp_isdf = isdf[nt];
        auto const& sp_paw  = sps[nt];
        if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
        if (sp_isdf.nlambda == 0) continue;
        auto K = compute_K_a(sp_isdf, sp_paw.deltaC);
        long row0 = layout.atom_aug_offset[ia];
        for (int lam = 0; lam < sp_isdf.nlambda; ++lam)
        for (int xi  = 0; xi  < sp_isdf.nlambda; ++xi)
            V_LL_inout(row0 + lam, row0 + xi) += ComplexType(K(lam, xi), 0.0);
    }
}

/**
 * Build per-(s,k) Y feature rows Y_{aλ,n}^k = Σ_I U_{a,λI} P^k_{n,aI}
 * stacked into a (N_A, nbnd) buffer. Caller writes into the augmentation
 * rows of the smooth X_full at row offset N_mu + atom_aug_offset[a].
 *
 * This is the row-augmentation of X. It is the same code regardless of q
 * (Y depends on k only). Consumed at every k-point for every spin index.
 */
template<class P_view, class isdf_vec>
inline void fill_Y_rows_for_sk(
    pseudopot const& psp,
    isdf_vec const& isdf,
    paw_aug_layout const& layout,
    int npol,
    int s_in_basis_id,           // index already collapsed for nspin/npol_in_basis
    int k_idx,
    P_view const& Pskna,         // pseudopot's Pskna view
    nda::ArrayOfRank<2> auto& Y_out)  // (N_A, nbnd*npol)
{
    long N_A = layout.N_A;
    if (N_A == 0) return;
    auto const& ityp = psp.ityp_view();
    auto const& nh_v = psp.nh_view();
    auto const& ofs  = psp.ofs_view();
    long nat = ityp.extent(0);
    long nbnd = Y_out.extent(1) / std::max(1, npol);
    Y_out() = ComplexType(0.0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nh_a = nh_v(nt);
        int nlam = isdf[nt].nlambda;
        if (nh_a == 0 || nlam == 0) continue;
        long row0 = layout.atom_aug_offset[ia];
        long p0   = ofs(ia) * npol;
        for (int lam = 0; lam < nlam; ++lam) {
            for (int i = 0; i < nbnd; ++i) {
                ComplexType acc(0.0);
                for (int I = 0; I < nh_a; ++I) {
                    double u = isdf[nt].U(lam, I);
                    if (u == 0.0) continue;
                    for (int sigma = 0; sigma < npol; ++sigma)
                        acc += ComplexType(u) *
                               Pskna(s_in_basis_id, k_idx,
                                     p0 + I*npol + sigma, i);
                }
                Y_out(row0 + lam, i) = acc;
            }
        }
    }
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_AUG_THC_HPP
