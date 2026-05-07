/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * One-center radial charge densities for the PAW deeq SCF self-consistency.
 *
 * The Hartree-only correction to the PAW non-local D matrix needs, per
 * SCF iteration, the LM-decomposed all-electron and pseudo charge
 * densities on each atom's radial mesh:
 *
 *   ρ_AE^a(r, LM) = Σ_IJ  becsum_aIJ · pfunc_IJ(r)  · G_{LM,IJ}
 *                 + ρ_core_AE(r) · δ_{LM,00}
 *
 *   ρ_PS^a(r, LM) = Σ_IJ  becsum_aIJ · ptfunc_IJ(r) · G_{LM,IJ}
 *                 + Σ_IJ  becsum_aIJ · Σ_L qfuncl_L,IJ(r) · G_{LM,IJ}'_L
 *                 + ρ_core_PS(r) · δ_{LM,00}
 *
 * with `pfunc_IJ(r) = ψ_AE_{indv(I)}(r) · ψ_AE_{indv(J)}(r)` and analogous
 * for `ptfunc`, and `G_{LM,IJ}` the real-spherical-harmonic Clebsch-
 * Gordan coupling (= QE's `aatab.ap(lp, nhtolm(I), nhtolm(J))`).
 *
 * Stage 3.2 deliverable: `compute_paw_core_density(sp)` precomputes
 * ρ_core_AE(r), ρ_core_PS(r) once per species (frozen-core, SCF-invariant).
 * These caches feed the LM=00 channel of ρ_AE/ρ_PS in stage 3.3's
 * Hartree-only deeq driver.
 *
 * The valence-density LM builders (per-iter `becsum`-dependent) live
 * with the Hartree integrator in stage 3.3 — putting them here would
 * be premature since the Hartree integral packs the contraction
 * Σ_IJ becsum × pfunc × G into a single radial pass.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_ONECENTER_HPP
#define HAMILTONIAN_PAW_PAW_ONECENTER_HPP

#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/paw_radial.hpp"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"   // for aainit_tables
#include "hamiltonian/paw/v_h_paw.hpp"          // for compute_becsum_full

namespace hamilt::paw {

/**
 * Frozen-core radial charge densities on the species's radial mesh.
 *
 * AE side: ρ_core_AE(r) = (1/4π) Σ_n occ_n · (u_n(r) / r)² , where
 *   u_n(r) = sp.core_aewfc(n, r) is the core all-electron orbital
 *   stored as u(r) = r · R(r). The closed-shell occupation occ_n =
 *   2 · (2 l_n + 1) (filled subshell).
 *
 * PS side: ρ_core_PS(r) = sp.rho_atc_ps(r)  — already in radial form
 *   on the same mesh, used by QE for non-linear-core-correction (NLCC)
 *   on the smooth side. We treat it as the "frozen-core PS density"
 *   here (consistent with QE's PAW formalism).
 *
 * Both are SCF-invariant under the frozen-core approximation; build once
 * at pseudopot init, cache, reuse every SCF iteration. Returns
 * (ρ_core_AE, ρ_core_PS) — both shape (mesh) on the radial mesh.
 *
 * If the GIPAW core data is absent (`sp.ncore_orbitals == 0` or
 * `sp.core_aewfc.size() == 0`), ρ_core_AE is left zeroed (no
 * core-valence Hartree contribution from the AE side). ρ_core_PS uses
 * `rho_atc_ps` if present, else zero.
 */
inline std::pair<nda::array<double,1>, nda::array<double,1>>
compute_paw_core_density(pseudopot::species_paw_t const& sp)
{
    long mesh = (long)sp.mesh;
    nda::array<double,1> rho_core_AE = nda::array<double,1>::zeros({mesh});
    nda::array<double,1> rho_core_PS = nda::array<double,1>::zeros({mesh});

    // PS side: copy directly from rho_atc_ps.
    if (sp.rho_atc_ps.size() > 0) {
        utils::check((long)sp.rho_atc_ps.size() == mesh,
            "compute_paw_core_density: rho_atc_ps.size ({}) != mesh ({})",
            sp.rho_atc_ps.size(), mesh);
        for (long i = 0; i < mesh; ++i)
            rho_core_PS(i) = sp.rho_atc_ps(i);
    }

    // AE side: build from core orbitals. core_aewfc stores u_n(r) = r·R_n(r);
    // ρ_core_AE(r) = (1/4π) Σ_n occ_n · R_n(r)² = (1/4π) Σ_n occ_n · (u_n/r)².
    // For r=0 we fall back to ρ_core_AE(0) = ρ_core_AE(r_1) (the first
    // mesh point's value), since u_n(0) = 0 by regularity but the limit
    // (u_n/r)² is finite — extrapolating from the next point is good
    // enough for a typical log mesh where r(0) ≈ 1e-6.
    if (sp.ncore_orbitals > 0 && sp.core_aewfc.size() > 0 &&
        sp.core_l.size() > 0) {
        long ncore = sp.ncore_orbitals;
        utils::check((long)sp.core_aewfc.shape(0) == ncore &&
                     (long)sp.core_aewfc.shape(1) == mesh,
            "compute_paw_core_density: core_aewfc shape ({},{}) != ({},{})",
            sp.core_aewfc.shape(0), sp.core_aewfc.shape(1), ncore, mesh);
        utils::check((long)sp.core_l.size() == ncore,
            "compute_paw_core_density: core_l.size ({}) != ncore ({})",
            sp.core_l.size(), ncore);

        double inv4pi = 1.0 / (4.0 * M_PI);
        for (long n = 0; n < ncore; ++n) {
            int l_n = (int)std::lround(sp.core_l(n));
            double occ_n = 2.0 * (2.0 * (double)l_n + 1.0);  // closed-shell
            for (long i = 1; i < mesh; ++i) {
                double u = sp.core_aewfc(n, i);
                double R = u / sp.r(i);
                rho_core_AE(i) += inv4pi * occ_n * R * R;
            }
        }
        // r=0 extrapolation. With a log mesh r(0) is tiny; copying
        // r(1)'s value is accurate to leading order.
        if (mesh > 1) rho_core_AE(0) = rho_core_AE(1);
    }

    return {std::move(rho_core_AE), std::move(rho_core_PS)};
}

/**
 * Per-atom Hartree-only contribution to the PAW deeq matrix from a current
 * `becsum`, plus the matching one-center Hartree energy. SCF-driver-facing.
 *
 * Algorithm (Hartree only — XC dropped per CoQui's no-DFT scGW design):
 *
 *   For each LM channel `lp` allowed by `aatab.lpx/lpl/ap`:
 *     1. ρ_AE(r, lp) = Σ_IJ becsum_IJ · pfunc_IJ(r) · ap(lp, ivl, jvl)
 *                    + δ_{lp,0} · √(4π) · ρ_core_AE(r)
 *        ρ_PS(r, lp) = Σ_IJ becsum_IJ · [ ptfunc_IJ(r)
 *                                        + qfuncl(L, ij(I,J), r) · ap(lp, ivl, jvl) ]
 *                    + δ_{lp,0} · √(4π) · ρ_core_PS(r)
 *        with pfunc_IJ(r)  = aewfc(beta_I, r) · aewfc(beta_J, r)
 *             ptfunc_IJ(r) = pswfc(beta_I, r) · pswfc(beta_J, r)
 *             ij           = upper-triangle (beta_I, beta_J) index
 *             L            = floor(√lp)
 *
 *     2. V_H_AE(r, lp) = radial_hartree_multipole(ρ_AE, r, rab, L)
 *        V_H_PS(r, lp) = radial_hartree_multipole(ρ_PS, r, rab, L)
 *
 *     3. ΔD_IJ += ap(lp, ivl, jvl) · ∫ [ V_H_AE(r) · pfunc_IJ(r)
 *                                         − V_H_PS(r) · (ptfunc_IJ + qfuncl_L_ij) ] dr
 *
 *   E_H_oc = ½ Σ_IJ becsum_IJ · ΔD_IJ.
 *
 * Correctness check (validated by the energy-balance test in stage 3.3
 * once fixtures are regenerated): the same total Hartree energy comes
 * out of (smooth-grid v_h on ρ̃_compensated)  +  (this routine summed
 * over atoms) as out of the augmented-ERI J contraction.
 */
struct paw_oc_hartree_result {
    nda::array<double, 2> dDeeq_H; // (nh, nh) — Hartree contribution to deeq
    double E_H_oc = 0.0;           // ½ Σ_IJ becsum_IJ · dDeeq_H_IJ
};

inline paw_oc_hartree_result compute_paw_hartree_atom(
    pseudopot::species_paw_t const& sp,
    nda::ArrayOfRank<2> auto const& becsum,         // (nh, nh)
    nda::ArrayOfRank<1> auto const& rho_core_AE,    // (mesh)
    nda::ArrayOfRank<1> auto const& rho_core_PS,    // (mesh)
    aainit_tables const& aatab)
{
    long mesh = (long)sp.mesh;
    int  nh   = sp.nh;
    paw_oc_hartree_result out;
    out.dDeeq_H = nda::array<double, 2>::zeros({(long)nh, (long)nh});
    if (nh == 0 || mesh == 0) return out;

    long Lp1 = sp.qfuncl.extent(0);  // 2*lmax_aug + 1, channels of qfuncl
    long llx = aatab.llx;            // (2*lli - 1)²

    // QE-canonical truncation index: the one-center pair density (and ΔC
    // contraction) is supported in [0, kkbeta). Outside, the AE and PS
    // partial waves match by PAW construction, so the integrand
    //   V_AE × pf − V_PS × (ptf + qf)  =  pf × (V_AE − V_PS)
    // vanishes analytically. Numerically the cumulative-trap u-form
    // Poisson at L ≥ 3 amplifies the residual O(eps) cancellation noise
    // from the [kkbeta, mesh) tail by r^L (observed: 5×10⁵ Ha on Si d-d
    // L=4 pairs without truncation). Zeroing rho/integrand past kkbeta
    // matches QE's `simpson(kkbeta, ...)` convention used in its own PAW
    // one-center code paths and is the lowest-disruption robust fix.
    long kkbeta = (sp.kkbeta > 0 && sp.kkbeta <= mesh) ? sp.kkbeta : mesh;

    nda::array<double, 1> rho_AE(mesh), rho_PS(mesh);
    nda::array<double, 1> V_AE(mesh),   V_PS(mesh);
    nda::array<double, 1> integrand(mesh);

    double sqrt4pi = std::sqrt(4.0 * M_PI);

    // Precompute for each (I, J): (ivl, jvl, beta_I, beta_J, ij_pair)
    struct pair_info { int ivl, jvl, bI, bJ, ij; };
    std::vector<pair_info> pinfo((size_t)nh * (size_t)nh);
    for (int I = 0; I < nh; ++I)
    for (int J = 0; J < nh; ++J) {
        int bI = sp.indv(I) - 1;
        int bJ = sp.indv(J) - 1;
        int n1 = std::max(bI, bJ), n2 = std::min(bI, bJ);
        pinfo[(size_t)I * nh + J] = {sp.nhtolm(I) - 1, sp.nhtolm(J) - 1,
                                      bI, bJ, (n1 * (n1 + 1)) / 2 + n2};
    }

    for (long lp = 0; lp < llx; ++lp) {
        int L = (int)std::floor(std::sqrt((double)lp + 1e-9));

        // Build ρ_AE(r), ρ_PS(r) for this lp on [0, kkbeta); zero past.
        rho_AE() = 0.0;
        rho_PS() = 0.0;
        for (int I = 0; I < nh; ++I)
        for (int J = 0; J < nh; ++J) {
            auto const& p = pinfo[(size_t)I * nh + J];
            double cg = aatab.ap(lp, p.ivl, p.jvl);
            if (cg == 0.0) continue;
            double w = becsum(I, J) * cg;
            if (w == 0.0) continue;
            for (long ir = 0; ir < kkbeta; ++ir) {
                double pf  = sp.aewfc(p.bI, ir) * sp.aewfc(p.bJ, ir);
                double ptf = sp.pswfc(p.bI, ir) * sp.pswfc(p.bJ, ir);
                double qf  = (L < Lp1) ? sp.qfuncl(L, p.ij, ir) : 0.0;
                rho_AE(ir) += w * pf;
                rho_PS(ir) += w * (ptf + qf);
            }
        }
        if (lp == 0) {
            // Core densities are stored in proper-ρ form (no r² jacobian);
            // pfunc/ptfunc/qfuncl above are in QE's u-form (already r²-
            // weighted: pf = u·u = r²·R²). Multiply core by r² so the
            // u-form Hartree solver below sees a consistent input.
            for (long ir = 0; ir < kkbeta; ++ir) {
                double r2 = sp.r(ir) * sp.r(ir);
                rho_AE(ir) += sqrt4pi * rho_core_AE(ir) * r2;
                rho_PS(ir) += sqrt4pi * rho_core_PS(ir) * r2;
            }
        }

        // Radial Hartree per LM channel — u-form (densities carry the
        // spherical r² jacobian; integrand uses r'^L / r'^{-(L+1)}).
        radial_hartree_multipole_u_form(rho_AE, sp.r, sp.rab, L, V_AE);
        radial_hartree_multipole_u_form(rho_PS, sp.r, sp.rab, L, V_PS);

        // Integrate (V_AE - V_PS) × pair_density × ap (with q-aug on PS side)
        // on [0, kkbeta). Outside, the integrand is analytically zero by PAW
        // boundary matching; we zero-fill explicitly to avoid noise leakage
        // when V_AE − V_PS hasn't fully cancelled at finite mesh resolution.
        for (int I = 0; I < nh; ++I)
        for (int J = 0; J < nh; ++J) {
            auto const& p = pinfo[(size_t)I * nh + J];
            double cg = aatab.ap(lp, p.ivl, p.jvl);
            if (cg == 0.0) continue;
            integrand() = 0.0;
            for (long ir = 0; ir < kkbeta; ++ir) {
                double pf  = sp.aewfc(p.bI, ir) * sp.aewfc(p.bJ, ir);
                double ptf = sp.pswfc(p.bI, ir) * sp.pswfc(p.bJ, ir);
                double qf  = (L < Lp1) ? sp.qfuncl(L, p.ij, ir) : 0.0;
                integrand(ir) = cg * (V_AE(ir) * pf - V_PS(ir) * (ptf + qf));
            }
            out.dDeeq_H(I, J) += radial_simpson(integrand, sp.rab);
        }
    }

    // E_H_oc = ½ Σ_IJ becsum_IJ × dDeeq_H_IJ.
    double E = 0.0;
    for (int I = 0; I < nh; ++I)
    for (int J = 0; J < nh; ++J)
        E += 0.5 * becsum(I, J) * out.dDeeq_H(I, J);
    out.E_H_oc = E;

    return out;
}

/**
 * SCF-invariant ("static") one-center contribution to the PAW deeq matrix
 * for one species — kinetic + frozen-core-screened ionic potential, on
 * the (AE − PS) difference. This is the analogue of QE's `paw_init_keeq`
 * built directly inside CoQui from the canonical UPF inputs (no QE
 * runtime dependency, no frozen-XC residual baked into the baseline).
 *
 *   D_static^IJ = T_AE^IJ − T_PS^IJ
 *               + ⟨ψ_AE_I | ae_vloc | ψ_AE_J⟩
 *               − ⟨ψ_PS_I | vloc_ps  | ψ_PS_J⟩
 *
 * Output is in Hartree. UPF radial potentials (`ae_vloc`, `vloc_ps`) are
 * Ry by QE convention; we multiply the V-matel integral by ½ to land
 * in Ha. The radial kinetic operator (`radial_kinetic_matel`) is
 * already implemented in Hartree.
 *
 * Selection rule: with a spherically symmetric static potential and the
 * radial-only matrix element formulation, only `lm_I == lm_J` couples
 * (same l AND same m). All other (I, J) entries are exactly zero.
 *
 * Returns shape (nh, nh). Caller adds this to QE's `dion`/`dvan` (already
 * stored as `Dnn` per species, in Ha) to assemble the full SCF-invariant
 * baseline; the dynamic Hartree update from `compute_paw_hartree_atom`
 * is then added per SCF iteration.
 */
inline nda::array<double, 2> compute_paw_static_D(
    pseudopot::species_paw_t const& sp)
{
    long mesh = (long)sp.mesh;
    int  nh   = sp.nh;
    nda::array<double, 2> D = nda::array<double, 2>::zeros({(long)nh, (long)nh});
    if (nh == 0 || mesh == 0) return D;

    // Need ae_vloc, vloc_ps, aewfc, pswfc, indv, nhtol, nhtolm, r, rab.
    bool have_aewfc   = sp.aewfc.size()   > 0;
    bool have_pswfc   = sp.pswfc.size()   > 0;
    bool have_aevloc  = sp.ae_vloc.size() > 0;
    bool have_vlocps  = sp.vloc_ps.size() > 0;
    if (!have_aewfc || !have_pswfc || !have_aevloc || !have_vlocps)
        return D;     // missing inputs → caller falls back to QE deeq

    nda::array<double, 1> integrand(mesh);

    for (int I = 0; I < nh; ++I) {
        int bI  = sp.indv(I)   - 1;
        int lmI = sp.nhtolm(I) - 1;     // 0-based composite L·M index
        int lI  = sp.nhtol(I);
        for (int J = 0; J < nh; ++J) {
            int lmJ = sp.nhtolm(J) - 1;
            // Spherical V + radial-only matel ⇒ only lm_I == lm_J couples.
            if (lmI != lmJ) continue;
            int bJ = sp.indv(J) - 1;
            int l  = lI;

            // Kinetic in Hartree (radial_kinetic_matel already does -½∇²).
            // QE stores u(r) = r·R(r) in {ae,ps}wfc, so the radial form
            //   T_IJ = ½ ∫ [u_I' u_J' + l(l+1)/r² u_I u_J] dr
            // applies without an extra r² factor.
            auto u_AE_I = sp.aewfc(bI, nda::range::all);
            auto u_AE_J = sp.aewfc(bJ, nda::range::all);
            auto u_PS_I = sp.pswfc(bI, nda::range::all);
            auto u_PS_J = sp.pswfc(bJ, nda::range::all);
            double T_AE = radial_kinetic_matel(u_AE_I, u_AE_J, sp.r, sp.rab, l);
            double T_PS = radial_kinetic_matel(u_PS_I, u_PS_J, sp.r, sp.rab, l);

            // V matel — integrand in Ry (UPF convention); ½ converts to Ha.
            for (long ir = 0; ir < mesh; ++ir) {
                integrand(ir) = sp.ae_vloc(ir) * u_AE_I(ir) * u_AE_J(ir)
                              - sp.vloc_ps(ir) * u_PS_I(ir) * u_PS_J(ir);
            }
            double V_IJ_Ry = radial_simpson(integrand, sp.rab);
            double V_IJ    = 0.5 * V_IJ_Ry;

            D(I, J) = (T_AE - T_PS) + V_IJ;
        }
    }
    return D;
}

} // namespace hamilt::paw

/* ==========================================================================
 * pseudopot::build_paw_scf_caches  +  pseudopot::compute_deeq_scf
 *
 * Out-of-class definitions for the methods declared in pseudopot.h. Live
 * in the `paw_onecenter.hpp` header so the implementation has access to the
 * `compute_paw_*` helpers above without creating a circular include between
 * pseudopot.h and the PAW one-center machinery.
 *
 * Callers that invoke either method must include this header (transitively
 * is fine; e.g., `methods/HF/thc_hf.hpp` will pull this in once stage 3.6
 * lands).
 * ========================================================================== */

namespace hamilt {

inline void pseudopot::build_paw_scf_caches()
{
    if (paw_scf_caches_built) return;

    // 1. Compute the aainit lli (= 1 + max_l over all PAW betas).
    int lli = 1;
    for (auto const& sp : paw_species)
        for (long b = 0; b < (long)sp.lll.size(); ++b)
            lli = std::max(lli, (int)sp.lll(b) + 1);
    paw_aainit_lli = lli;

    // 2. Per-species frozen-core radial densities. Cheap (O(mesh × ncore));
    //    held in heap memory on every rank (small — < few KB per species).
    paw_core_density.clear();
    paw_core_density.reserve(paw_species.size());
    for (auto const& sp : paw_species)
        paw_core_density.emplace_back(hamilt::paw::compute_paw_core_density(sp));

    // 3. Allocate the SCF-invariant baseline Dnn_atom_static_paw with the
    //    same shape as Dnn_atom (per-node SHM via sarray_t).
    long nat = ityp.extent(0);
    long nD  = (long)Dnn_atom.local().shape()[1];   // nhm*npol from existing Dnn_atom
    Dnn_atom_static_paw = sarray_t<nda::array_view<ComplexType,3>>(
        *mpi, {std::max((long)1, nat), nD, nD});
    mpi->node_comm.barrier();

    // 4. Populate on the node-root: D_static = dvan_replica + paw_init_keeq.
    //    Mirrors the layout of Dnn_atom (ia, n*npol+p, m*npol+p).
    if (mpi->node_comm.root()) {
        auto Dloc_sp = Dnn.local();           // (nsp, nhm*npol, nhm*npol)
        auto Dloc_st = Dnn_atom_static_paw.local();  // (nat, nhm*npol, nhm*npol)
        Dloc_st() = ComplexType(0.0);

        for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);

            // (a) dvan replica from species-resolved Dnn (already in Ha).
            for (long n = 0; n < nD; ++n)
                for (long m = 0; m < nD; ++m)
                    Dloc_st(ia, n, m) = Dloc_sp(nt, n, m);

            // (b) paw_init_keeq additive piece for PAW species with
            //     populated radial inputs (non-PAW or empty-fields species
            //     leave the dvan-only baseline in place).
            if ((size_t)nt >= paw_species.size()) continue;
            auto const& sp = paw_species[nt];
            if (sp.nh == 0) continue;
            auto Dst = hamilt::paw::compute_paw_static_D(sp);
            if (Dst.size() == 0) continue;
            for (int I = 0; I < sp.nh; ++I)
                for (int J = 0; J < sp.nh; ++J)
                    for (int p = 0; p < npol; ++p)
                        Dloc_st(ia, I * npol + p, J * npol + p)
                            += ComplexType(Dst(I, J), 0.0);
        }
    }
    mpi->node_comm.barrier();
    if (mpi->node_comm.root())
        mpi->internode_comm.broadcast_n(Dnn_atom_static_paw.local().data(),
                                        Dnn_atom_static_paw.size(), 0);
    mpi->comm.barrier();

    paw_scf_caches_built = true;
}

template<typename nij_t>
inline void pseudopot::compute_deeq_scf(nij_t const& nij)
{
    utils::check(npol == 1,
        "pseudopot::compute_deeq_scf: npol > 1 (SOC) is not supported yet");

    // 1. Make sure the static caches are in place.
    build_paw_scf_caches();

    // 2. Build per-atom becsum from Pskna and the current density matrix.
    auto becsum = hamilt::paw::compute_becsum_full(
        Pskna_view(), nij, ityp, nh, ofs, npol);
    long nat = ityp.extent(0);

    // 3. Build the angular-momentum coupling tables (cheap: nibz tables of
    //    size O((2 lli − 1)⁴), all PAW species share the same lli).
    auto aatab = hamilt::paw::aainit_tables_build(paw_aainit_lli);

    // 4. On the node-root: copy the static baseline into Dnn_atom and add
    //    the per-atom Hartree contribution from `compute_paw_hartree_atom`.
    // MAM: Parallelize over atoms and reduce at the end!
    if (mpi->node_comm.root()) {
        auto Dloc_at = Dnn_atom.local();
        auto Dloc_st = Dnn_atom_static_paw.local();
        Dloc_at() = Dloc_st();

        for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);
            if ((size_t)nt >= paw_species.size()) continue;
            auto const& sp = paw_species[nt];
            if (sp.nh == 0 || sp.aewfc.size() == 0) continue;

            // Slice atom-resolved becsum (nh, nh).
            auto bs_a = becsum(ia, nda::range(0, sp.nh),
                                   nda::range(0, sp.nh));

            auto const& cores = paw_core_density[nt];
            auto const& rho_core_AE = cores.first;
            auto const& rho_core_PS = cores.second;

            auto res = hamilt::paw::compute_paw_hartree_atom(
                sp, bs_a, rho_core_AE, rho_core_PS, aatab);

            for (int I = 0; I < sp.nh; ++I)
                for (int J = 0; J < sp.nh; ++J)
                    for (int p = 0; p < npol; ++p)
                        Dloc_at(ia, I * npol + p, J * npol + p)
                            += ComplexType(res.dDeeq_H(I, J), 0.0);
        }
    }
    mpi->node_comm.barrier();
    if (mpi->node_comm.root())
        mpi->internode_comm.broadcast_n(Dnn_atom.local().data(),
                                        Dnn_atom.size(), 0);
    mpi->comm.barrier();
}

} // namespace hamilt

#endif // HAMILTONIAN_PAW_PAW_ONECENTER_HPP
