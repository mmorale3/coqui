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
 *   Neither side touches the static per-atom D (`Dnn_atom_static`) — both are
 *   computed from radial pseudopot data + becsum, free of any DFT-XC
 *   contamination. (The QE `deeq` read was removed entirely in plan A1.)
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
        //    not populated, so pass explicit zero buffers to keep the
        //    comparison core-free on both sides regardless of fixture state.
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

// ===========================================================================
// MF-independent consistency of the becsum / deeq builders.
//
// The PAW one-center deeq (and V_H / V_X matrix elements) must be a functional
// of CoQui's self-consistent one-body density matrix `n`, carrying the SAME
// (ns_scl/N_k) normalization the smooth-grid Hartree applies to that same `n`.
// It must NOT depend on the QE mean-field occupations — those only seed the
// initial state. There are two becsum builders that must agree:
//
//   * compute_becsum_diagonal(occ)  — used by the validated paths
//     (compute_paw_deeq, smooth-Hartree nii overload, the energy path);
//     carries wk = 1/N_k internally, caller applies ns_scl.
//   * compute_becsum_full(nij)      — used by the SELF-CONSISTENT paths
//     (compute_deeq_scf, smooth-Hartree nij overload).
//
// Mathematical identity: for a DIAGONAL density matrix n_{skab}=δ_ab occ_ska,
//   compute_becsum_full(n) == compute_becsum_diagonal(occ)   element-wise.
// Both carry the same w_k = 1/N_k k-weight and ns_scl spin factor.
// ===========================================================================
static void run_becsum_full_matches_diagonal_test(std::string fixture_name,
                                                  double tol)
{
    using nda::range;
    auto& mpi = utils::make_unit_test_mpi_context();
    auto qe_h5 = mf::default_MF(mpi, fixture_name, mf::h5_input_type);
    auto& mfobj = qe_h5;

    long nspin  = mfobj.nspin();
    long nk_ibz = mfobj.nkpts_ibz();
    long nbnd   = mfobj.nbnd();
    int  npol   = mfobj.npol();
    REQUIRE(npol == 1);

    hamilt::pseudopot V(mfobj);

    // Diagonal occupations nii(s,k,n) and the equivalent diagonal density
    // matrix nij(s,k,a,b) = δ_ab nii(s,k,a).
    nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
    nii() = mfobj.occ()(range::all, range(nk_ibz), range::all);

    nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
    nij() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
    for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

    auto bec_diag = hamilt::paw::compute_becsum_diagonal(
        V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
    auto bec_full = hamilt::paw::compute_becsum_full(
        V.Pskna_view(), nij, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);

    REQUIRE(bec_diag.shape() == bec_full.shape());

    double max_abs_diff = 0.0, max_abs_val = 0.0;
    for (long ia = 0; ia < bec_diag.extent(0); ++ia)
    for (long I  = 0; I  < bec_diag.extent(1); ++I)
    for (long J  = 0; J  < bec_diag.extent(2); ++J) {
        max_abs_diff = std::max(max_abs_diff,
                                std::abs(bec_full(ia, I, J) - bec_diag(ia, I, J)));
        max_abs_val  = std::max({max_abs_val,
                                 std::abs(bec_full(ia, I, J)),
                                 std::abs(bec_diag(ia, I, J))});
    }
    app_log(1,
        "[paw becsum full-vs-diagonal / {}] max |Δbecsum| = {:.3e} "
        "(max |element| = {:.3e}, tol = {:.1e})",
        fixture_name, max_abs_diff, max_abs_val, tol);
    REQUIRE(max_abs_val > 1e-8);          // becsum is non-trivial
    CHECK(max_abs_diff < tol);
}

TEST_CASE("paw_onecenter_becsum_full_matches_diagonal",
          "[hamilt][paw][onecenter]")
{
    SECTION("qe_lih222_paw") {
        run_becsum_full_matches_diagonal_test("qe_lih222_paw", 1e-10);
    }
    SECTION("qe_si222_paw") {
        run_becsum_full_matches_diagonal_test("qe_si222_paw", 1e-10);
    }
}

// ===========================================================================
// End-to-end: the SELF-CONSISTENT deeq builder compute_deeq_scf(nij) must
// reproduce the validated compute_paw_deeq(occ) for a diagonal density matrix.
//
//   compute_paw_deeq(occ, /*Vloc_r=*/empty, include_static=true)
//        = D_static + radial_Hartree(ns_scl × becsum_diagonal(occ))   [no G term]
//   compute_deeq_scf(diag(occ))   [returned; Dnn_atom_static NOT mutated]
//        = D_static + radial_Hartree(ns_scl × becsum_full(diag(occ)))
//
// With the becsum identity above these are element-wise equal. This guards
// BOTH fixes on the self-consistent path: the 1/N_k in compute_becsum_full
// AND the ns_scl applied on the nij becsum path. It validates the deeq as a
// functional of the density matrix `n` — no comparison against mf.occ() as if
// it were ground truth. Additionally guards the plan-A1 non-mutation contract:
// the static tensor must be bit-identical before and after the call.
// ===========================================================================
TEST_CASE("paw_onecenter_deeq_scf_matches_paw_deeq",
          "[hamilt][paw][onecenter]")
{
    using nda::range;
    auto& mpi = utils::make_unit_test_mpi_context();
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    auto& mfobj = qe_h5;

    long nspin  = mfobj.nspin();
    long nk_ibz = mfobj.nkpts_ibz();
    long nbnd   = mfobj.nbnd();
    int  npol   = mfobj.npol();
    REQUIRE(npol == 1);

    hamilt::pseudopot V(mfobj);

    nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
    nii() = mfobj.occ()(range::all, range(nk_ibz), range::all);

    nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
    nij() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
    for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

    // Validated diagonal builder: static + one-center radial Hartree, no
    // G-space term (empty Vloc_r → Vloc_r.size()==0 short-circuits it).
    nda::array<ComplexType,1> empty_vloc;
    auto Dion_diag = V.compute_paw_deeq(nii, empty_vloc, /*include_static=*/true);

    // Self-consistent builder returns D_static + D^H[n] without mutating the
    // static tensor (Dnn_atom_view now exposes Dnn_atom_static, plan A1).
    nda::array<ComplexType,3> Dstatic_before{V.Dnn_atom_view()};
    auto Dscf = V.compute_deeq_scf(nij);
    {
        auto Dstatic_after = V.Dnn_atom_view();
        double max_mut = 0.0;
        for (long ia = 0; ia < Dstatic_before.extent(0); ++ia)
        for (long I  = 0; I  < Dstatic_before.extent(1); ++I)
        for (long J  = 0; J  < Dstatic_before.extent(2); ++J)
            max_mut = std::max(max_mut,
                std::abs(Dstatic_after(ia, I, J) - Dstatic_before(ia, I, J)));
        REQUIRE(max_mut == 0.0);   // non-mutation contract (plan A1)
    }

    REQUIRE(Dion_diag.extent(0) == Dscf.extent(0));
    REQUIRE(Dion_diag.extent(1) == Dscf.extent(1));
    REQUIRE(Dion_diag.extent(2) == Dscf.extent(2));

    double max_abs_diff = 0.0, max_abs_val = 0.0;
    for (long ia = 0; ia < Dion_diag.extent(0); ++ia)
    for (long I  = 0; I  < Dion_diag.extent(1); ++I)
    for (long J  = 0; J  < Dion_diag.extent(2); ++J) {
        max_abs_diff = std::max(max_abs_diff,
                                std::abs(Dscf(ia, I, J) - Dion_diag(ia, I, J)));
        max_abs_val  = std::max({max_abs_val,
                                 std::abs(Dscf(ia, I, J)),
                                 std::abs(Dion_diag(ia, I, J))});
    }
    app_log(1,
        "[paw deeq_scf vs paw_deeq / qe_lih222_paw] max |ΔD| = {:.3e} Ha "
        "(max |element| = {:.3e})", max_abs_diff, max_abs_val);
    REQUIRE(max_abs_val > 1e-8);          // deeq is non-trivial
    CHECK(max_abs_diff < 1e-10);
}
