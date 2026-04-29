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
 * Phase 4.2 covers the q=0 path explicitly (Hartree validation). The
 * q≠0 path uses the radial spherical-Bessel η^q(G) helpers in
 * `eta_to_G.hpp`; a dedicated builder for that case is a follow-up.
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
 * Build a remap table from the psp dense G grid → the thc's rho_g grid.
 *
 * For each ig ∈ [0, ngm_dense):
 *   psp_to_rho[ig]  = index in rho_g.g_vectors() with matching miller indices,
 *                     or -1 if rho_g doesn't include that G (cutoff mismatch).
 *
 * The convention: use FFT linear index N(m1,m2,m3) = ((m1+Nx)%Nx*Ny + ...
 * ... + (m3+Nz)%Nz) on the rho_g.mesh(); the rho_g exposes the inverse
 * via `fft_to_gv` if available, else we build it from gv_to_fft.
 *
 * If ngm_dense_unmapped > 0 after the pass, a warning is emitted (caller
 * decides whether to abort). For Phase 4.2 we typically expect zero
 * unmapped indices (ecut == ecutrho).
 */
inline std::vector<long> make_psp_to_rho_g_map(
    pseudopot const& psp,
    grids::truncated_g_grid const& rho_g,
    long& n_unmapped_out)
{
    long ngm_dense = psp.ngm_dense_get();
    std::vector<long> map(ngm_dense, -1);
    n_unmapped_out = 0;
    if (ngm_dense == 0) return map;

    long NX = rho_g.mesh(0), NY = rho_g.mesh(1), NZ = rho_g.mesh(2);
    long nnr = rho_g.nnr();
    auto const& mill = psp.miller_g_dense_view();
    utils::check(mill.extent(0) == ngm_dense,
                 "make_psp_to_rho_g_map: miller_g_dense rows ({}) != ngm_dense ({})",
                 mill.extent(0), ngm_dense);

    // Build FFT linear → rho_g index (use the existing fft_to_gv if there).
    std::vector<long> fft_to_g(nnr, -1);
    auto const& gv_to_fft = rho_g.gv_to_fft();
    long ngm_rho = rho_g.size();
    for (long g = 0; g < ngm_rho; ++g) {
        long N = gv_to_fft(g);
        if (N >= 0 && N < nnr) fft_to_g[N] = g;
    }

    for (long ig = 0; ig < ngm_dense; ++ig) {
        int m1 = mill(ig, 0), m2 = mill(ig, 1), m3 = mill(ig, 2);
        long n1 = m1; if (n1 < 0) n1 += NX;
        long n2 = m2; if (n2 < 0) n2 += NY;
        long n3 = m3; if (n3 < 0) n3 += NZ;
        if (n1 < 0 || n1 >= NX || n2 < 0 || n2 >= NY ||
            n3 < 0 || n3 >= NZ) { ++n_unmapped_out; continue; }
        long N = (n1*NY + n2)*NZ + n3;
        long ig_rho = fft_to_g[N];
        if (ig_rho < 0) { ++n_unmapped_out; continue; }
        map[ig] = ig_rho;
    }
    return map;
}

/**
 * Project η_{a,λ}^{q=0}(G) onto the thc rho_g grid. Output shape:
 *   eta_aλ_on_rho(a, λ, g_rho)  with g_rho ∈ [0, rho_g.size()).
 *
 * Atom structure factor e^{-iG·τ_a} is APPLIED here so callers can use the
 * result directly in V_GL / V_LL contractions.
 */
inline nda::array<ComplexType, 3> build_eta_on_rho_g_q0(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    grids::truncated_g_grid const& rho_g)
{
    long ngm_rho = rho_g.size();
    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    int nlam_max = 0;
    for (auto const& s : isdf) nlam_max = std::max(nlam_max, s.nlambda);

    nda::array<ComplexType, 3> eta_aug =
        nda::array<ComplexType, 3>::zeros({nat, (long)nlam_max, ngm_rho});
    if (nlam_max == 0) return eta_aug;

    long n_unmapped = 0;
    auto psp_to_rho = make_psp_to_rho_g_map(psp, rho_g, n_unmapped);
    if (n_unmapped > 0)
        app_warning("paw_aug_thc: {} of {} dense-grid G entries fall outside "
                    "the thc rho_g grid (likely cutoff mismatch). Truncating.",
                    n_unmapped, psp.ngm_dense_get());

    // Atom structure factor on rho_g.
    auto const& tau = psp.atom_pos_cart_view();
    auto const& g_cart = rho_g.g_vectors();   // (ngm_rho, 3)
    nda::array<double,2> cos_tau(nat, ngm_rho), sin_tau(nat, ngm_rho);
    for (long ia = 0; ia < nat; ++ia)
    for (long g = 0; g < ngm_rho; ++g) {
        double ph = g_cart(g,0)*tau(ia,0) + g_cart(g,1)*tau(ia,1)
                  + g_cart(g,2)*tau(ia,2);
        cos_tau(ia, g) = std::cos(ph);
        sin_tau(ia, g) = std::sin(ph);
    }

    long ngm_psp = psp.ngm_dense_get();
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        auto const& s = isdf[nt];
        if (s.nlambda == 0) continue;
        utils::check(s.eta_qg_q0.extent(1) == ngm_psp,
            "paw_aug_thc: isdf eta_qg_q0 G dim ({}) != psp ngm_dense ({})",
            s.eta_qg_q0.extent(1), ngm_psp);
        for (int lam = 0; lam < s.nlambda; ++lam) {
            for (long ig = 0; ig < ngm_psp; ++ig) {
                long ig_rho = psp_to_rho[ig];
                if (ig_rho < 0) continue;
                // Apply structure factor e^{-iG·τ_a}
                ComplexType eta_q0 = s.eta_qg_q0(lam, ig);
                ComplexType sf(cos_tau(ia, ig_rho), -sin_tau(ia, ig_rho));
                eta_aug(ia, lam, ig_rho) = eta_q0 * sf;
            }
        }
    }
    return eta_aug;
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
 * Optimization: precompute the radial Bessel transforms qrad(L, ij, G)
 * once per (species, ij) over the entire G grid (atom-independent), then
 * the per-atom angular sum is O(N_λ × n_lp × ngm).
 */
inline nda::array<ComplexType, 3> build_eta_on_rho_g_at_q(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    grids::truncated_g_grid const& rho_g,
    std::array<double, 3> const& q_cart,
    double omega,
    aainit_tables const& aatab)
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
        // qrad_g(L, ij, g) = ∫ qfuncl(L, ij, r) j_L(|q+G|·r) dr  (no extra r²;
        // qfuncl already carries it per UPF convention).
        nda::array<double, 3> qrad_g(Lp1, n_ijv, ngm_rho);
        qrad_g() = 0.0;
        for (long g = 0; g < ngm_rho; ++g) {
            for (long ijv = 0; ijv < n_ijv; ++ijv) {
                auto qrL = qrad_at_K(sp, (int)ijv, Kmod(g));
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
 *                produced by build_eta_on_rho_g_q0
 *   wG         : (ngm_rho) — Coulomb weights from coulomb_weights_on_rho_g
 *   layout     : aug layout
 *   psp        : pseudopot (used for ityp + isdf-row counts)
 *   isdf       : per-species local ISDF
 * Output (caller-owned):
 *   V_GL(μ, Λ_aug)  for Λ_aug ∈ [0, N_A); accumulated additively.
 */
inline void compute_VGL_q0_on_rho_g(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    nda::ArrayOfRank<2> auto const& zeta_mu_g,
    nda::ArrayOfRank<3> auto const& eta_aug,
    nda::ArrayOfRank<1> auto const& wG,
    nda::ArrayOfRank<2> auto       & V_GL_out)
{
    long N_mu = layout.N_mu;
    long N_A  = layout.N_A;
    if (N_A == 0 || N_mu == 0) return;
    long ngm = wG.extent(0);
    utils::check(zeta_mu_g.extent(0) == N_mu, "compute_VGL_q0: zeta_mu_g mu-dim");
    utils::check(zeta_mu_g.extent(1) == ngm,  "compute_VGL_q0: zeta_mu_g G-dim");
    utils::check(eta_aug.extent(2) == ngm,    "compute_VGL_q0: eta_aug G-dim");
    utils::check(V_GL_out.extent(0) == N_mu,  "compute_VGL_q0: V_GL_out shape");
    utils::check(V_GL_out.extent(1) == N_A,   "compute_VGL_q0: V_GL_out shape");

    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if (nt >= (int)isdf.size()) continue;
        int nlam = isdf[nt].nlambda;
        if (nlam == 0) continue;
        long row0 = layout.atom_aug_offset[ia];
        for (long mu = 0; mu < N_mu; ++mu) {
            for (int lam = 0; lam < nlam; ++lam) {
                ComplexType acc(0.0);
                for (long g = 0; g < ngm; ++g) {
                    acc += zeta_mu_g(mu, g) * std::conj(eta_aug(ia, lam, g))
                         * wG(g);
                }
                V_GL_out(mu, row0 + lam) += acc;
            }
        }
    }
}

/**
 * Compute the V_LL block at q=0:
 *   V_LL(aλ, bξ)  =  Σ_G  η_{aλ}^*(G)e^{+iG·τ_a}  η_{bξ}(G)e^{-iG·τ_b}
 *                          ·  4π/(Ω|G|²)
 *
 * eta_aug already carries the e^{-iG·τ} factor from
 * `build_eta_on_rho_g_q0`, so this is a straightforward conj/mult/sum.
 *
 * Output: V_LL_out(N_A, N_A); accumulated additively.
 */
inline void compute_VLL_q0_on_rho_g(
    pseudopot const& psp,
    std::vector<species_local_isdf> const& isdf,
    paw_aug_layout const& layout,
    nda::ArrayOfRank<3> auto const& eta_aug,
    nda::ArrayOfRank<1> auto const& wG,
    nda::ArrayOfRank<2> auto       & V_LL_out)
{
    long N_A = layout.N_A;
    if (N_A == 0) return;
    long ngm = wG.extent(0);
    utils::check(eta_aug.extent(2) == ngm, "compute_VLL_q0: eta_aug G-dim");
    utils::check(V_LL_out.extent(0) == N_A, "compute_VLL_q0: V_LL_out shape");
    utils::check(V_LL_out.extent(1) == N_A, "compute_VLL_q0: V_LL_out shape");

    auto const& ityp = psp.ityp_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
        int nt_a = ityp(ia);
        if (nt_a >= (int)isdf.size()) continue;
        int nlam_a = isdf[nt_a].nlambda;
        if (nlam_a == 0) continue;
        long rowa = layout.atom_aug_offset[ia];
        for (long ib = 0; ib < nat; ++ib) {
            int nt_b = ityp(ib);
            if (nt_b >= (int)isdf.size()) continue;
            int nlam_b = isdf[nt_b].nlambda;
            if (nlam_b == 0) continue;
            long rowb = layout.atom_aug_offset[ib];
            for (int lam = 0; lam < nlam_a; ++lam)
            for (int xi  = 0; xi  < nlam_b; ++xi) {
                ComplexType acc(0.0);
                for (long g = 0; g < ngm; ++g) {
                    acc += std::conj(eta_aug(ia, lam, g))
                         * eta_aug(ib, xi,  g) * wG(g);
                }
                V_LL_out(rowa + lam, rowb + xi) += acc;
            }
        }
    }
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
