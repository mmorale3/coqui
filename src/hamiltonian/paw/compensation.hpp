/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * PAW compensation-charge utilities (Phase 3).
 *
 * Companion to notes/paw_isdf_thc_prb.tex, Eq. (deltaq) / (moment-condition):
 *   ΔQ_{αβ}(s) = φ_α^*(s) φ_β(s) - φ̃_α^*(s) φ̃_β(s)
 *   Q̂_{αβ}(s) ≈ ΔQ_{αβ}(s) modulo multipoles through PAW lmax_aug
 *
 * QE's qvan2 already builds Q^{IJ}(G) (= the Bloch sum of Q̂_{αβ}) on the
 * dense FFT grid; pw2coqui exports it as augmentation_function_isp{nt} and
 * pseudopot::read_vnl_h5 loads it into pseudopot::qgm. This file provides
 * (a) accessors to Q̂_a^{IJ}(G) including the e^{-iG·τ_a} structure factor,
 * and (b) a radial multipole validator that confirms qfuncl reproduces the
 * augmom moments stored on the species. Both are pure helpers — no member
 * state of pseudopot is mutated.
 *
 * Phase 4 will use the accessors to assemble the local-local block of the
 * THC projected Coulomb matrix V^{q,C}_{aλ,bξ}.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_COMPENSATION_HPP
#define HAMILTONIAN_PAW_COMPENSATION_HPP

#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Multipole-moment validator on the per-species radial grid.
 *
 * For each PAW species nt with `qfuncl(r, ij, L)` and `augmom(α, β, L)`:
 *
 *   M^L_{ij}  = ∫₀^∞ r^{L+2} qfuncl_{ij,L}(r) dr            (radial)
 *
 * The integral uses Simpson quadrature on the radial mesh r(:), rab(:).
 * This M^L_{ij} should equal `augmom(α, β, L)` (with (α,β) ↔ ij the upper-
 * triangular flatten) up to mesh discretization error.
 *
 * Returns the max absolute violation across all (species, ij, L). Throws
 * via utils::check if it exceeds tol.
 *
 * Note: this is the radial check; it does NOT verify that Q̂_a^{IJ}(G)
 * correctly represents the same moments in cartesian space. That second
 * check requires a 3D integration on the dense FFT grid and lives in the
 * Phase 4 kernel-assembly tests.
 */
inline double validate_multipoles_radial(pseudopot const& psp,
                                          double tol = 1e-6) {
    double max_viol = 0.0;
    auto simpson = [](nda::array<double,1> const& rab,
                      std::vector<double> const& integrand) {
        // Composite Simpson on the linear x-mesh (rab = dr/dx). For the
        // integrand f(r) the contribution at mesh point i is f(r_i)*rab(i).
        long n = static_cast<long>(integrand.size());
        if (n < 3) return 0.0;
        if (n % 2 == 0) --n; // need odd count for composite Simpson
        double s = integrand[0]*rab(0) + integrand[n-1]*rab(n-1);
        for (long i=1; i<n-1; i+=2) s += 4.0*integrand[i]*rab(i);
        for (long i=2; i<n-1; i+=2) s += 2.0*integrand[i]*rab(i);
        return s/3.0;
    };

    for (size_t nt_=0; nt_<psp.paw_species.size(); ++nt_) {
        auto const& sp = psp.paw_species[nt_];
        if (!sp.is_paw) continue;
        if (sp.augmom.size() == 0 || sp.qfuncl.size() == 0) continue;

        // Layout discovered from Fortran column-major → C row-major:
        //   pw2coqui writes augmom(nbeta, nbeta, 0:2*lmax)  → C: (2*lmax+1, nbeta, nbeta)
        //   pw2coqui writes qfuncl(mesh, nij, 0:2*lmax)     → C: (2*lmax+1, nij, mesh)
        long Lmax_p1 = sp.augmom.extent(0);
        long nbeta   = sp.augmom.extent(1);
        long nij     = sp.qfuncl.extent(1);
        long mesh    = sp.qfuncl.extent(2);
        utils::check(sp.augmom.extent(2) == nbeta,
                     "augmom shape mismatch (species {})", nt_);
        utils::check(sp.qfuncl.extent(0) == Lmax_p1,
                     "qfuncl L-dim mismatch (species {})", nt_);
        utils::check(sp.r.extent(0)   == mesh, "r/qfuncl mesh mismatch");
        utils::check(sp.rab.extent(0) == mesh, "rab/qfuncl mesh mismatch");
        utils::check(nij == nbeta*(nbeta+1)/2,
                     "nij != nbeta*(nbeta+1)/2 (species {})", nt_);

        // Compare M^L_{ij} = ∫ r^{L+2} qfuncl_{ij,L}(r) dr  vs augmom(α,β,L).
        // ij = upper-triangular flatten of (α,β); QE convention is
        //   ij = β*(β-1)/2 + α  (1-based, β >= α). In 0-based:
        //   ij = β*(β+1)/2 + α  with β >= α.
        std::vector<double> integrand(mesh);
        for (long L=0; L<Lmax_p1; ++L) {
            for (long beta=0; beta<nbeta; ++beta)
            for (long alpha=0; alpha<=beta; ++alpha) {
                long ij = beta*(beta+1)/2 + alpha;
                for (long ir=0; ir<mesh; ++ir) {
                    double rL = std::pow(sp.r(ir), L+2);
                    integrand[ir] = rL * sp.qfuncl(L, ij, ir);
                }
                double M_calc = simpson(sp.rab, integrand);
                double M_ref  = sp.augmom(L, alpha, beta);
                double viol = std::abs(M_calc - M_ref);
                max_viol = std::max(max_viol, viol);
            }
        }
    }
    utils::check(max_viol < tol,
        "validate_multipoles_radial: max violation {} exceeds tol {}",
        max_viol, tol);
    return max_viol;
}

/**
 * Per-atom Bloch-summed augmentation accessor:
 *   Q̂_a^{IJ}(G) = e^{-iG·τ_a} * Q^{IJ}_{nt(a)}(G)
 *
 * Combines pseudopot::qgm with a structure-factor table the caller provides.
 * Returns a function-like object that maps (atom_index, ih, jh, g_index) to
 * a complex number.
 *
 * For Phase 4 THC kernel assembly:
 *   V^{q,C}_{aλ,bξ}(G) = Σ_{α,β,γ,δ} U^*_{aλα} U_{aλβ} U^*_{bξγ} U_{bξδ}
 *                        × Σ_G Q̂_a^{αβ}(-q-G) (4π/|q+G|^2) Q̂_b^{γδ}(q+G)
 *
 * The q-dependence enters via (q+G) instead of plain G when ψ_a has Bloch
 * crystal momentum k_a and ψ_b has k_b with q = k_b - k_a. The qvan2
 * output is q-independent because Q^{IJ}(G+q) ≈ Q^{IJ}(G) interpolation —
 * we'll need to call qvan2 at (q+G) per-q in Phase 4 (separate dataset
 * needed from pw2coqui or recomputation in CoQui).
 *
 * This helper just packages the bookkeeping (ijtoh + structure factor +
 * qgm slice) cleanly so Phase 4 doesn't have to re-derive it.
 */
class AtomAugmentationAccess {
public:
    AtomAugmentationAccess(pseudopot const& psp,
                           nda::ArrayOfRank<2> auto const& gvec_phase)
        : psp_(&psp), phase_(gvec_phase) {
        utils::check(psp.qgm.local().size() > 0,
                     "AtomAugmentationAccess: pseudopot.qgm not populated");
    }

    /** Q^{ij}_nt(G) where ij = ijtoh(nt, ih, jh) − 1. */
    inline ComplexType operator()(long ia, int ih, int jh, long g) const {
        long nt = psp_->ityp(ia);
        // ijtoh stored 1-based per pw2coqui/QE convention
        long ij = static_cast<long>(psp_->ijtoh(nt, ih, jh)) - 1;
        utils::check(ij >= 0,
            "AtomAugmentationAccess: ijtoh(nt={},ih={},jh={}) returned 0",
            nt, ih, jh);
        return psp_->qgm.local()(nt, ij, g) * phase_(ia, g);
    }

private:
    pseudopot const* psp_;
    nda::array<ComplexType,2> phase_; // (nat, ngm_dense), e^{-iG·τ_a}
};

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_COMPENSATION_HPP
