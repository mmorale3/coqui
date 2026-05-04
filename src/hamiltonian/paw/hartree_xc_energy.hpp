/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Direct, FFT-based evaluation of the smooth-grid Hartree energy and the
 * potential×density XC integral, for verification against QE's `ehart`,
 * `vtxc`, etc.
 *
 *   E_H   = (Ω/2) Σ_{G≠0} |ρ_total(G)|² × 4π/|G|²        (Hartree, in Ha)
 *   vtxc  = ∫ V_xc(r) ρ_total(r) dr                       (in Hartree)
 *
 * For NCPP, ρ_total = ρ_smooth (the standard valence density). For
 * USPP/PAW, we additionally augment with
 *   ρ_aug(G) = Σ_a Σ_{IJ} becsum_{a,IJ} × Q^{IJ}_{nt(a)}(G) × e^{-iG·τ_a}
 * matching QE's `addusdens` / `v_of_rho` pipeline.
 *
 * Implementation: a serial r-space construction on the root rank, then a
 * forward FFT to G-space and the closed-form sum. Verification routine —
 * not perf-critical.
 *
 * Returns are in HARTREE units (CoQui's convention).
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_HARTREE_XC_ENERGY_HPP
#define HAMILTONIAN_PAW_HARTREE_XC_ENERGY_HPP

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/v_h_paw.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/symmetry.hpp"

namespace hamilt::paw {

/**
 * Build ρ_total(r) on the dense FFT mesh (smooth + augmentation, on the
 * proper ρ scale matching QE's rho%of_r):
 *   ρ_total(r) = (ns_scl/N_k) × Σ_{s,k,n} f_{s,k,n} |ψ̃_{s,k,n}(r)|²
 *              + (1/Ω) × iFFT_G→r [ Σ_a Σ_{IJ} becsum × Q^{IJ}(G) × e^{-iG·τ_a} ]
 *
 * Notes on FFT/normalization conventions:
 *   - NDA's fft<true/false> with backward(): un-normalized FFT, i.e.
 *     ψ(r_grid) = Σ_G ψ(G) × e^{iGr_grid}  (no 1/N factor)
 *   - The integrated electron count check ∫ρ dr ≈ N_e × dr × Σ_r ρ(r)
 *     where dr = Ω/N_grid.
 *
 * Returns ρ(r) on the root rank as a 1D buffer of length N_grid; other
 * ranks return empty. All ranks must call (collective barrier inside).
 */
template<typename Psi_t, typename Nii_t>
inline nda::array<double, 1> build_total_density_r(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    int npol,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& recv,
    nda::ArrayOfRank<1> auto const& k2g,
    nda::ArrayOfRank<2> auto const& kpts,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,
    nda::ArrayOfRank<1> auto const& kp_trev,
    nda::ArrayOfRank<1> auto const& kp_symm,
    std::vector<utils::symm_op> const& symm_list,
    Nii_t const& nii,
    Psi_t const& psi,
    double vol,
    bool include_augmentation = true)
{
    long nnr = mesh(0)*mesh(1)*mesh(2);
    long nspin = psi.global_shape()[0];
    long nbnd = psi.global_shape()[2];
    long nk = kpts.shape(0);   // full BZ
    long ngm = k2g.shape(0);

    nda::array<double, 1> rho(nnr);
    rho() = 0.0;

    // Gather distributed psi to root. Collective: all ranks call. Only root
    // allocates the gather buffer at the global shape; other ranks pass nullptr.
    using local_psi_t = typename std::decay_t<Psi_t>::Array_t;
    local_psi_t psi_full;
    if (mpi.comm.root())
        psi_full = local_psi_t(psi.global_shape());
    math::nda::gather(0, psi, &psi_full);

    if (mpi.comm.root()) {
        // ---- Smooth density: ρ_smooth(r) = (ns_scl/N_k) Σ f |ψ|² ----
        nda::array<ComplexType, 1> psi_g(nnr);
        auto psi_g3d = nda::reshape(psi_g,
            std::array<long,3>{mesh(0), mesh(1), mesh(2)});
        // fft<false>: single rank-3 FFT (NOT batched/many over dim 0)
        math::nda::fft<false> F(psi_g3d);

        nda::array<ComplexType, 1> *Xft = nullptr;
        nda::stack_array<double, 3> Gs;
        Gs() = 0.0;
        memory::array<HOST_MEMORY, long, 1> k2g_rot(k2g.shape(0));

        for (long s = 0; s < nspin; ++s)
        for (long k = 0; k < nk; ++k) {
            long k_sym = kp_to_ibz(k);
            // Apply symmetry on the k2g mapping
            k2g_rot() = k2g();
            if (kp_trev(k) || kp_symm(k) > 0)
                utils::transform_k2g(kp_trev(k), symm_list[kp_symm(k)], Gs,
                                     mesh, kpts(k_sym, nda::range::all),
                                     k2g_rot, Xft);

            for (long p = 0; p < npol; ++p)
            for (long n = 0; n < nbnd; ++n) {
                double f = std::real(nii(s, k_sym, n));
                if (std::abs(f) < 1e-14) continue;

                // Place ψ(G) onto the dense FFT mesh
                psi_g() = ComplexType(0.0);
                for (long g = 0; g < ngm; ++g) {
                    long N = k2g_rot(g);
                    if (N >= 0 && N < nnr)
                        psi_g(N) = psi_full(s, k_sym, n, p*ngm + g);
                }
                // G → r (un-normalized)
                F.backward(psi_g3d);
                if (kp_trev(k)) psi_g() = nda::conj(psi_g);

                // Accumulate f × |ψ(r)|². NDA's iFFT is un-normalized,
                // i.e., ψ(r) = Σ_G ψ_G e^{iGr}. The corresponding ρ(r)
                // is f × |ψ(r)|², which is what QE accumulates.
                for (long r_idx = 0; r_idx < nnr; ++r_idx) {
                    double a = std::real(psi_g(r_idx));
                    double b = std::imag(psi_g(r_idx));
                    rho(r_idx) += f * (a*a + b*b);
                }
            }
        }
        // Normalize: rho × ns_scl / (N_k × Ω). The 1/Ω compensates for the
        // un-normalized backward FFT convention (NDA's backward gives
        // ψ_unnorm(r) = √Ω × ψ_proper(r) when ψ_G are properly Bloch-
        // normalized, so |ψ_unnorm|² = Ω × |ψ_proper|²). Result matches
        // QE's rho%of_r in atomic units e^-/Bohr^3 with ∫ρ dr = N_e.
        double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
        double scl = ns_scl / ((double)nk * vol);
        for (long r_idx = 0; r_idx < nnr; ++r_idx) rho(r_idx) *= scl;

        // ---- Augmentation density: ρ_aug(r) = iFFT[Σ becsum × Q × sf] ----
        if (include_augmentation &&
            psp.pp_type() != pp_ncpp_t && psp.qgm_view().size() > 0) {
            auto becsum = compute_becsum_diagonal(psp.Pskna_view(), nii,
                                                   psp.ityp_view(),
                                                   psp.nh_view(),
                                                   psp.ofs_view(), npol);
            // compute_becsum_diagonal returns Σ wk × occ × |β·ψ|² with
            // wk = 1/N_k and no spin factor; QE's becsum (which qvan2 +
            // ddd_paw assume) is wg × |β·ψ|² with wg = ns_scl × wk × occ.
            // Apply the missing ns_scl here so ρ_aug matches QE's addusdens.
            for (long ia = 0; ia < becsum.extent(0); ++ia)
            for (long I  = 0; I  < becsum.extent(1); ++I)
            for (long J  = 0; J  < becsum.extent(2); ++J)
                becsum(ia, I, J) *= ns_scl;
            // Build ρ_aug(G) on the dense G-grid, then iFFT to r-space.
            nda::array<ComplexType, 1> rho_aug_g(nnr);
            rho_aug_g() = ComplexType(0.0);
            long nat = psp.ityp_view().extent(0);
            long ngm_d = psp.ngm_dense_get();
            auto qg = psp.qgm_view();
            auto const& ijtoh = psp.ijtoh_view();
            auto const& ityp  = psp.ityp_view();
            auto const& nh    = psp.nh_view();
            auto const& mill  = psp.miller_g_dense_view();
            auto const& tau   = psp.atom_pos_cart_view();
            long NX = mesh(0), NY = mesh(1), NZ = mesh(2);
            for (long g = 0; g < ngm_d; ++g) {
                int m1 = mill(g, 0), m2 = mill(g, 1), m3 = mill(g, 2);
                long n1 = m1; if (n1 < 0) n1 += NX;
                long n2 = m2; if (n2 < 0) n2 += NY;
                long n3 = m3; if (n3 < 0) n3 += NZ;
                if (n1 < 0 || n1 >= NX || n2 < 0 || n2 >= NY ||
                    n3 < 0 || n3 >= NZ) continue;
                long N = (n1*NY + n2)*NZ + n3;
                double Gx = m1*recv(0,0) + m2*recv(1,0) + m3*recv(2,0);
                double Gy = m1*recv(0,1) + m2*recv(1,1) + m3*recv(2,1);
                double Gz = m1*recv(0,2) + m2*recv(1,2) + m3*recv(2,2);
                ComplexType acc(0.0);
                for (long ia = 0; ia < nat; ++ia) {
                    int nt = ityp(ia);
                    int nh_a = nh(nt);
                    if (nh_a == 0) continue;
                    double phase = -(Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2));
                    ComplexType sf(std::cos(phase), std::sin(phase));
                    ComplexType atom_acc(0.0);
                    for (int I = 0; I < nh_a; ++I)
                    for (int J = 0; J < nh_a; ++J) {
                        long ij = static_cast<long>(ijtoh(nt, I, J)) - 1;
                        if (ij < 0) continue;
                        atom_acc += ComplexType(becsum(ia, I, J)) * qg(nt, ij, g);
                    }
                    acc += sf * atom_acc;
                }
                rho_aug_g(N) += acc;
            }
            // qgm from qvan2 returns the augmentation Fourier coefficients
            // in the proper density convention, i.e.
            //   ρ_aug,proper(G) = Σ_a Σ_IJ becsum_aIJ × qgm(nt(a),IJ,G) × e^{-iG·τ_a}
            // satisfies ρ_aug(r) = Σ_G ρ_aug,proper(G) × e^{+iGr}.
            // NDA's backward FFT is un-normalized (= Σ_G ψ_G e^{iGr}), so the
            // result IS ρ_aug,proper(r) directly — no extra normalization.
            auto rho3d = nda::reshape(rho_aug_g,
                std::array<long,3>{mesh(0), mesh(1), mesh(2)});
            math::nda::fft<false> Faug(rho3d);
            Faug.backward(rho3d);
            for (long r_idx = 0; r_idx < nnr; ++r_idx)
                rho(r_idx) += std::real(rho_aug_g(r_idx));
        }
    }
    mpi.comm.barrier();
    return rho;
}

/**
 * Compute E_H = (Ω/2) Σ_{G≠0} |ρ_total(G)|² × 4π/|G|² on the dense
 * FFT mesh. Returns E_H in HARTREE.
 *
 * Internally builds ρ_total(r), forward FFTs to G, sums the closed form.
 */
template<typename Psi_t, typename Nii_t>
inline double hartree_energy_paw(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    int npol,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& recv,
    nda::ArrayOfRank<1> auto const& k2g,
    nda::ArrayOfRank<2> auto const& kpts,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,
    nda::ArrayOfRank<1> auto const& kp_trev,
    nda::ArrayOfRank<1> auto const& kp_symm,
    std::vector<utils::symm_op> const& symm_list,
    Nii_t const& nii,
    Psi_t const& psi,
    bool include_augmentation = true)
{
    long nnr = mesh(0)*mesh(1)*mesh(2);

    // Cell volume from reciprocal vectors (rows = b1, b2, b3):
    //   Ω = (2π)^3 / |det(B)|
    double det_B = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                 - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                 + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
    double vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

    auto rho_r = build_total_density_r(mpi, psp, npol, mesh, recv,
                                        k2g, kpts, kp_to_ibz, kp_trev,
                                        kp_symm, symm_list, nii, psi, vol,
                                        include_augmentation);

    double E_H = 0.0;
    if (mpi.comm.root()) {
        // Sanity: total electron count via dr × Σ_r ρ; also ∫|ρ|² for
        // diagnosing the localization / variance of the constructed ρ.
        double sum_rho = 0.0;
        double sum_rho2 = 0.0;
        for (long r = 0; r < nnr; ++r) {
            sum_rho += rho_r(r);
            sum_rho2 += rho_r(r) * rho_r(r);
        }
        double dr_d = vol / (double)nnr;
        app_log(3, "hartree_energy_paw: ∫ρ dr = {:.6f} (N_e ref), ∫|ρ|² dr = {:.6f}, Ω = {:.4f}",
                dr_d * sum_rho, dr_d * sum_rho2, vol);

        // Forward FFT ρ_total(r) → ρ_total(G).
        nda::array<ComplexType, 1> rho_g(nnr);
        for (long r = 0; r < nnr; ++r) rho_g(r) = ComplexType(rho_r(r), 0.0);
        auto rho3d = nda::reshape(rho_g,
            std::array<long,3>{mesh(0), mesh(1), mesh(2)});
        math::nda::fft<false> F(rho3d);
        F.forward(rho3d);
        // NDA's `forward` IS normalized by 1/N (see fft/nda.hpp:250 with
        // `__normalize__ = true`), so rho_g(G) already equals
        // ρ_proper(G) = (1/N) Σ_r ρ(r) e^{-iGr}.  Therefore:
        //   E_H = (Ω/2) Σ_{G≠0} |ρ_proper(G)|² × 4π/|G|²
        //       = (Ω/2) Σ_{G≠0} |rho_g(G)|² × 4π/|G|²
        long NX = mesh(0), NY = mesh(1), NZ = mesh(2);
        double prefac = vol / 2.0;
        for (long n1 = 0; n1 < NX; ++n1) {
            int m1 = (n1 <= NX/2) ? (int)n1 : (int)n1 - (int)NX;
            for (long n2 = 0; n2 < NY; ++n2) {
                int m2 = (n2 <= NY/2) ? (int)n2 : (int)n2 - (int)NY;
                for (long n3 = 0; n3 < NZ; ++n3) {
                    int m3 = (n3 <= NZ/2) ? (int)n3 : (int)n3 - (int)NZ;
                    if (m1 == 0 && m2 == 0 && m3 == 0) continue;
                    double Gx = m1*recv(0,0) + m2*recv(1,0) + m3*recv(2,0);
                    double Gy = m1*recv(0,1) + m2*recv(1,1) + m3*recv(2,1);
                    double Gz = m1*recv(0,2) + m2*recv(1,2) + m3*recv(2,2);
                    double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
                    if (G2 < 1e-14) continue;
                    long N = (n1*NY + n2)*NZ + n3;
                    double rho_re = std::real(rho_g(N));
                    double rho_im = std::imag(rho_g(N));
                    E_H += prefac * (rho_re*rho_re + rho_im*rho_im) *
                           (4.0*M_PI / G2);
                }
            }
        }
    }
    mpi.comm.barrier();
    mpi.comm.broadcast_n(&E_H, 1, 0);
    return E_H;
}

/**
 * Cross-check Hartree energy: build V_H(r) using CoQui's existing
 * v_h_paw pipeline (which is validated by the USPP eigenvalue test), and
 * integrate (1/2) ∫ ρ_total(r) V_H(r) dr in r-space.
 *
 * If this matches `hartree_energy_paw` (G-space formula), the discrepancy
 * is consistent across both methods → likely in ρ. If they disagree, the
 * G-space prefactor is wrong.
 */
template<typename Psi_t, typename Nii_t>
inline double hartree_energy_via_vH(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    int npol,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& lattv,
    nda::stack_array<double, 3, 3> const& recv,
    nda::ArrayOfRank<1> auto const& k2g,
    nda::ArrayOfRank<2> auto const& kpts,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,
    nda::ArrayOfRank<1> auto const& kp_trev,
    nda::ArrayOfRank<1> auto const& kp_symm,
    std::vector<utils::symm_op> const& symm_list,
    Nii_t const& nii,
    Psi_t const& psi)
{
    long nnr = mesh(0)*mesh(1)*mesh(2);
    double det_B = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                 - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                 + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
    double vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);
    double dr = vol / (double)nnr;

    // ρ_total(r) — same construction we use for the G-space form
    auto rho_r = build_total_density_r(mpi, psp, npol, mesh, recv,
                                        k2g, kpts, kp_to_ibz, kp_trev,
                                        kp_symm, symm_list, nii, psi, vol);

    // V_H(r) via the existing v_h_paw pipeline (validated for USPP eigenvalues)
    auto svr = math::shm::shared_array<nda::array_view<ComplexType,1>>(
        mpi, {nnr});
    pots::potential_t vG(ptree{});
    v_h_paw(mpi, vG, psp, npol, mesh, lattv, recv, k2g, kpts, kp_to_ibz,
            kp_trev, kp_symm, symm_list, nii, psi, false, svr);

    double E_H = 0.0;
    if (mpi.comm.root()) {
        auto vh = svr.local();
        for (long r = 0; r < nnr; ++r)
            E_H += 0.5 * rho_r(r) * std::real(vh(r)) * dr;
    }
    mpi.comm.barrier();
    mpi.comm.broadcast_n(&E_H, 1, 0);
    return E_H;
}

/**
 * Compute vtxc = ∫ V_xc(r) × ρ_total(r) dr on the dense grid, where
 * V_xc(r) is provided externally (read from QE's saved file via pseudopot).
 *
 * Returns vtxc in HARTREE.
 */
template<typename Psi_t, typename Nii_t, typename Vxc_t>
inline double vxc_rho_integral_paw(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    int npol,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& recv,
    nda::ArrayOfRank<1> auto const& k2g,
    nda::ArrayOfRank<2> auto const& kpts,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,
    nda::ArrayOfRank<1> auto const& kp_trev,
    nda::ArrayOfRank<1> auto const& kp_symm,
    std::vector<utils::symm_op> const& symm_list,
    Nii_t const& nii,
    Psi_t const& psi,
    Vxc_t const& vxc_r)
{
    long nnr = mesh(0)*mesh(1)*mesh(2);
    double det_B = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                 - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                 + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
    double vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);
    double dr = vol / (double)nnr;

    auto rho_r = build_total_density_r(mpi, psp, npol, mesh, recv,
                                        k2g, kpts, kp_to_ibz, kp_trev,
                                        kp_symm, symm_list, nii, psi, vol);

    double vtxc = 0.0;
    if (mpi.comm.root()) {
        // V_xc shape on dense grid: (nspin, npol×npol, nnr) typically.
        // For nspin=1 with non-magnetic: sum over spin yields V_xc(r),
        // and ρ_total above is the spin-summed density. The integral is
        //   vtxc = ∫ V_xc(r) ρ(r) dr ≈ dr × Σ_r V_xc(r) ρ(r)
        long nspin_v = vxc_r.extent(0);
        for (long r = 0; r < nnr; ++r) {
            double v_sum = 0.0;
            for (long s = 0; s < nspin_v; ++s)
                v_sum += std::real(vxc_r(s, 0, r));
            // For nspin=1, vxc_r(0, 0, r) is the single V_xc; for LSDA,
            // the convention will need refinement (separate up/dn ρ).
            vtxc += v_sum * rho_r(r) * dr;
        }
    }
    mpi.comm.barrier();
    mpi.comm.broadcast_n(&vtxc, 1, 0);
    return vtxc;
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_HARTREE_XC_ENERGY_HPP
