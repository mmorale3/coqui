/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Matrix-element-level validation of the PAW one-center Hartree solver
 * (`hamilt::paw::compute_paw_hartree_atom`) against the existing
 * four-index closed-form Coulomb residual `species_paw_t::deltaC`
 * (= QE's raw `ke%k` from `PAW_init_fock_kernel`, K_AE − K_PS).
 *
 * Why this test exists:
 *   The pre-existing `test_hartree_thc_paw_aug` and `test_hartree_energy`
 *   tests compare TOTAL Hartree energies between paths. Total-energy
 *   agreement can mask off-diagonal cancellations at the matrix-element
 *   level. The PAW deeq-SCF self-consistency machinery operates on
 *   ΔD_H^a_IJ matrix elements, not energies, so a per-(I,J) check
 *   is what closes the validation gap before deeq SCF use.
 *
 * What it checks:
 *   For each PAW atom `a`, with the KS-occupations becsum_a:
 *
 *     ΔD_H_oc^a_IJ_radial   = compute_paw_hartree_atom(sp, becsum_a,
 *                                                       ρ_core_AE = 0,
 *                                                       ρ_core_PS = 0,
 *                                                       aatab).dDeeq_H(I, J)
 *
 *     ΔD_H_oc^a_IJ_deltaC   = Σ_{KL} sp.deltaC(I, J, K, L) × becsum_a(K, L)
 *
 *   These must agree element-wise to ~radial-quadrature noise (≲ 1e-5 Ha).
 *   Disagreement signals either a bug in the radial Hartree/integrand of
 *   `compute_paw_hartree_atom` or a convention drift relative to the
 *   established `deltaC` machinery.
 *
 *   Neither side touches `Dnn_atom` or QE's frozen `deeq` — both are
 *   computed from radial pseudopot data + becsum, free of any DFT-XC
 *   contamination.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <iostream>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/v_h_paw.hpp"          // compute_becsum_diagonal
#include "hamiltonian/paw/paw_aug_q_eval.hpp"   // aainit_tables_build
#include "hamiltonian/paw/paw_onecenter.hpp"    // compute_paw_hartree_atom

// Forward decl of the per-fixture body so we can drive both LiH and Si
// PAW fixtures from the same TEST_CASE structure.
static void run_dDeeq_H_matches_deltaC_test(std::string fixture_name,
                                            double tol);

TEST_CASE("paw_onecenter_dDeeq_H_matches_deltaC_contraction",
          "[hamilt][paw][onecenter]")
{
    SECTION("qe_lih222_paw") {
        run_dDeeq_H_matches_deltaC_test("qe_lih222_paw", 1e-5);
    }
    SECTION("qe_si222_paw") {
        // Si has d-projectors → exercises L up to 4 channels in the
        // u-form Poisson, which is where the numerical instability
        // hunt lives.
        run_dDeeq_H_matches_deltaC_test("qe_si222_paw", 1e-4);
    }
}

static void run_dDeeq_H_matches_deltaC_test(std::string fixture_name,
                                            double tol)
{
    using nda::range;
    auto& mpi = utils::make_unit_test_mpi_context();
    auto qe_h5 = mf::default_MF(mpi, fixture_name, mf::h5_input_type);
    auto& mfobj = qe_h5;

    long nspin   = mfobj.nspin();
    long nk_ibz  = mfobj.nkpts_ibz();
    long nbnd    = mfobj.nbnd();
    int  npol    = mfobj.npol();
    REQUIRE(npol == 1);   // SOC not in this test path

    hamilt::pseudopot V(mfobj);
    auto const& sps  = V.paw_species_view();
    auto const& ityp = V.ityp_view();
    long nat         = ityp.extent(0);

    // becsum at KS occupations (= the density used by all the existing
    // PAW Hartree-energy tests). compute_becsum_diagonal returns the
    // pre-spin-factored sum Σ wk × occ × |β·ψ|²; we apply the same
    // ns_scl factor (= 2 for nspin=1, npol=1 closed-shell) those tests
    // use so that becsum encodes the full electron density.
    nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
    nii() = mfobj.occ()(range::all, range(nk_ibz), range::all);

    auto becsum = hamilt::paw::compute_becsum_diagonal(
        V.Pskna_view(), nii, ityp, V.nh_view(), V.ofs_view(), npol);
    double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
    for (long ia = 0; ia < becsum.extent(0); ++ia)
    for (long I  = 0; I  < becsum.extent(1); ++I)
    for (long J  = 0; J  < becsum.extent(2); ++J)
        becsum(ia, I, J) *= ns_scl;

    // aatab: pick lli from the largest beta l in any PAW species (same
    // recipe as thc_reader_t / the SCF cache builder).
    int lli = 1;
    for (auto const& sp : sps)
        for (long b = 0; b < (long)sp.lll.size(); ++b)
            lli = std::max(lli, (int)sp.lll(b) + 1);
    auto aatab = hamilt::paw::aainit_tables_build(lli);

    // Per-atom matrix-element comparison.
    bool any_paw_atom = false;
    double max_abs_diff = 0.0;
    double max_abs_val  = 0.0;
    long total_pairs = 0;

    for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia);
        if ((size_t)nt >= sps.size()) continue;
        auto const& sp = sps[nt];
        if (!sp.is_paw || sp.nh == 0 || sp.deltaC.size() == 0
            || sp.aewfc.size() == 0)
            continue;
        any_paw_atom = true;

        long nh_a = sp.nh;
        auto bs_a = becsum(ia, range(0, nh_a), range(0, nh_a));

        // -- Path 1: radial Hartree, no core densities. -------------------
        //    On qe_lih222_paw the core fields (rho_atc_ps, core_aewfc) are
        //    not populated yet (stage 3.7 fixture regen pending); pass
        //    explicit zero buffers so the comparison is core-free on both
        //    sides regardless of fixture state.
        nda::array<double,1> rho_core_AE_zero =
            nda::array<double,1>::zeros({(long)sp.mesh});
        nda::array<double,1> rho_core_PS_zero =
            nda::array<double,1>::zeros({(long)sp.mesh});
        auto res = hamilt::paw::compute_paw_hartree_atom(
            sp, bs_a, rho_core_AE_zero, rho_core_PS_zero, aatab);

        // -- Path 2: closed-form ΔC × becsum. -----------------------------
        //    deltaC has shape (nh, nh, nh, nh); contract over (K, L).
        //    On-disk deltaC is in proper Ha (pw2coqui divides by e2²
        //    before writing), so the contraction has no unit factor.
        nda::array<double,2> dD_dC =
            nda::array<double,2>::zeros({nh_a, nh_a});
        for (long I = 0; I < nh_a; ++I)
        for (long J = 0; J < nh_a; ++J) {
            double acc = 0.0;
            for (long K = 0; K < nh_a; ++K)
            for (long L = 0; L < nh_a; ++L)
                acc += sp.deltaC(I, J, K, L) * bs_a(K, L);
            dD_dC(I, J) = acc;
        }

        // Element-wise check + accumulate diagnostics.
        for (long I = 0; I < nh_a; ++I)
        for (long J = 0; J < nh_a; ++J) {
            double v_radial = res.dDeeq_H(I, J);
            double v_dC     = dD_dC(I, J);
            double diff     = std::abs(v_radial - v_dC);
            max_abs_diff    = std::max(max_abs_diff, diff);
            max_abs_val     = std::max({max_abs_val,
                                        std::abs(v_radial),
                                        std::abs(v_dC)});
            ++total_pairs;
            // Per-pair dump for the first PAW atom only (avoid log spam).
            if (ia == 0 && std::abs(v_radial) + std::abs(v_dC) > 1e-6) {
                double r_v = (std::abs(v_dC) > 1e-12) ? (v_radial / v_dC) : 0.0;
                app_log(2,
                    "  [ia=0 I={:2d} J={:2d}] radial = {:+.6e}  ΔC = {:+.6e}  "
                    "ratio = {:+.4f}  diff = {:+.3e}",
                    I, J, v_radial, v_dC, r_v, v_radial - v_dC);
            }
        }

        app_log(2,
            "[onecenter ia={} nt={}] max element-wise diff so far = {:.2e} Ha, "
            "max |element| = {:.2e}", ia, nt, max_abs_diff, max_abs_val);
    }

    REQUIRE(any_paw_atom);

    // Tolerance is fixture-specific (param). PAW_init_fock_kernel and
    // compute_paw_hartree_atom share the radial mesh and LM decomposition;
    // the divergence is the cumulative-trap vs Simpson quadrature, plus
    // any L-dependent numerical sensitivity that grows with lmax_aug.
    app_log(1,
        "[paw onecenter matrix-element check / {}] "
        "max |ΔD_H_radial − ΔD_H_ΔC| = {:.3e} Ha over {} (I,J) pairs "
        "(max |element| = {:.3e}, tol = {:.1e})",
        fixture_name, max_abs_diff, total_pairs, max_abs_val, tol);
    CHECK(max_abs_diff < tol);
}
