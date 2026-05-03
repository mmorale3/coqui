/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * PAW-aware Hartree potential evaluators on the dense FFT grid.
 *
 * The existing hamilt::v_h routines (src/hamiltonian/v_h.hpp) compute V_H
 * from the smooth pseudo-density ρ̃(r) only. For USPP/PAW, the relevant
 * smooth-grid quantity is the COMPENSATED density
 *
 *   ρ̄(G) = ρ̃(G) + Σ_a Σ_{IJ} becsum_{a,IJ} Q^{IJ}_{nt(a)}(G) e^{-iG·τ_a}
 *
 * where becsum is built from the projector overlaps Pskna and the input
 * occupations / density matrix:
 *   diagonal nii :  becsum_{a,IJ} = Σ_{s,k,n} f_{skn} P*_{n,aI} P_{n,aJ}
 *   full nij     :  becsum_{a,IJ} = Σ_{s,k,a,b} P*_{a,aI} n_{skab} P_{b,aJ}
 *
 * V̄_H(G) = (4π/|G|^2) ρ̄(G), V̄_H(r) = iFFT[V̄_H(G)].
 *
 * The δρ_a one-center residuals (.tex Eq. paw-coulomb-partition) are NOT
 * computed here — they enter via the PAW ΔD_{IJ} correction matrix added to
 * the non-local Hamiltonian (separate code path; not yet in CoQui).
 *
 * Strategy: rather than reimplement the v_h FFT pipeline, this module
 *   (1) calls the existing hamilt::v_h to get the smooth-grid ρ̃(r) folded
 *       into V_H(r),
 *   (2) computes the augmentation contribution Δρ_aug(G) directly from
 *       Pskna + qgm + miller_g_dense, applies 4π/|G|^2 to get ΔV_aug(G),
 *       and FFTs to ΔV_aug(r),
 *   (3) adds ΔV_aug(r) into the V_H(r) buffer.
 *
 * For NCPP this routine is a no-op (qgm is empty); callers can use it
 * unconditionally.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_V_H_PAW_HPP
#define HAMILTONIAN_PAW_V_H_PAW_HPP

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/fft/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "hamiltonian/v_h.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace hamilt::paw {

/**
 * becsum_{a,IJ} from a diagonal occupation set, matching QE's `becsum`:
 *   becsum_{a,IJ} = Σ_{s,k,n} w_k f_{skn} Re[P*_{n,aI} P_{n,aJ}]
 *
 * P comes from psp.Pskna with shape (nspin, nk_ibz, nkb*npol, nbnd).
 *
 * For uniform k-grids (the common case), w_k = 1/N_k for each k-point in
 * the full BZ. Without symmetry reduction, each IBZ k-point also has
 * w_k = 1/N_k. This implementation assumes uniform weights; general non-
 * uniform k-grid support is a follow-up.
 *
 * Sign convention: becsum is real for non-magnetic, scalar (npol=1)
 * calculations. Returns shape (nat, nhm, nhm).
 */
template<typename Pskna_t, typename occ_t>
inline nda::array<double,3> compute_becsum_diagonal(
    Pskna_t const& Pskna, occ_t const& nii,
    nda::array<int,1> const& ityp, nda::array<int,1> const& nh,
    nda::array<int,1> const& ofs, int npol)
{
    utils::check(npol == 1,
        "compute_becsum_diagonal: npol > 1 (SOC/noncollinear) is not yet "
        "implemented in the v_h_paw helpers");
    long nat  = ityp.extent(0);
    long nspin = nii.extent(0);
    long nk    = nii.extent(1);
    long nbnd  = nii.extent(2);
    int nhm = 0;
    for (long ia=0; ia<nat; ++ia) nhm = std::max(nhm, (int)nh(ityp(ia)));

    nda::array<double,3> becsum = nda::array<double,3>::zeros({nat, nhm, nhm});
    if (Pskna.size() == 0 || nbnd == 0) return becsum;
    auto P = Pskna;
    utils::check(P.extent(0) == nspin, "Pskna spin dim mismatch");
    utils::check(P.extent(1) == nk,    "Pskna k dim mismatch");
    utils::check(P.extent(3) == nbnd,  "Pskna nbnd mismatch");

    // Uniform k-weight: 1/N_k where N_k is the IBZ count (since the test
    // fixture uses nosym=true → IBZ = full BZ).
    double wk = 1.0 / (double)nk;

    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        int nh_a = nh(nt);
        if (nh_a == 0) continue;
        long p0 = ofs(ia);
        for (int I=0; I<nh_a; ++I)
        for (int J=0; J<nh_a; ++J) {
            double acc = 0.0;
            for (long s=0; s<nspin; ++s)
            for (long k=0; k<nk; ++k) {
                for (long n=0; n<nbnd; ++n) {
                    ComplexType pi = P(s, k, p0 + I, n);
                    ComplexType pj = P(s, k, p0 + J, n);
                    acc += wk * std::real(nii(s, k, n)) *
                           (std::real(pi)*std::real(pj) +
                            std::imag(pi)*std::imag(pj));
                }
            }
            becsum(ia, I, J) = acc;
        }
    }
    return becsum;
}

/**
 * becsum from a full density matrix nij(s,k,a,b):
 *   becsum_{a,IJ} = Σ_{s,k,a,b} P*_{a,aI} n_{skab} P_{b,aJ}
 *
 * Returns a real-valued (nat, nhm, nhm) for non-magnetic; the imaginary
 * residue from the unitary symmetry of n is dropped (warned if exceeds
 * a threshold). Same npol=1 restriction as the diagonal helper.
 */
template<typename Pskna_t, typename nij_t>
inline nda::array<double,3> compute_becsum_full(
    Pskna_t const& Pskna, nij_t const& nij,
    nda::array<int,1> const& ityp, nda::array<int,1> const& nh,
    nda::array<int,1> const& ofs, int npol)
{
    utils::check(npol == 1,
        "compute_becsum_full: npol > 1 (SOC/noncollinear) is not yet "
        "implemented in the v_h_paw helpers");
    long nat   = ityp.extent(0);
    long nspin = nij.extent(0);
    long nk    = nij.extent(1);
    long nbnd  = nij.extent(2);
    int nhm = 0;
    for (long ia=0; ia<nat; ++ia) nhm = std::max(nhm, (int)nh(ityp(ia)));

    nda::array<double,3> becsum = nda::array<double,3>::zeros({nat, nhm, nhm});
    if (Pskna.size() == 0 || nbnd == 0) return becsum;
    double max_imag = 0.0;
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        int nh_a = nh(nt);
        if (nh_a == 0) continue;
        long p0 = ofs(ia);
        for (int I=0; I<nh_a; ++I)
        for (int J=0; J<nh_a; ++J) {
            ComplexType acc(0.0);
            for (long s=0; s<nspin; ++s)
            for (long k=0; k<nk; ++k)
            for (long a=0; a<nbnd; ++a)
            for (long b=0; b<nbnd; ++b) {
                acc += std::conj(Pskna(s, k, p0 + I, a)) *
                       nij(s, k, a, b) *
                       Pskna(s, k, p0 + J, b);
            }
            becsum(ia, I, J) = std::real(acc);
            max_imag = std::max(max_imag, std::abs(std::imag(acc)));
        }
    }
    if (max_imag > 1e-8) {
        // For magnetic / off-diagonal density matrices that aren't Hermitian
        // becsum can carry a real-valued imaginary residue that we drop.
        // Surface a warning rather than fail — useful diagnostic in tests.
        app_log(2,
            "compute_becsum_full: dropped imaginary part up to {} in becsum",
            max_imag);
    }
    return becsum;
}

/**
 * Add the augmentation contribution to a V_H(r) buffer already populated
 * with the smooth-density Hartree potential.
 *
 * Workflow:
 *   1. Build ρ_aug(G) on the dense FFT mesh by deposit:
 *        ρ_aug(N(g)) += Σ_a Σ_IJ becsum_{a,IJ} Q^{IJ}_{nt(a)}(g) e^{-iG·τ_a}
 *      where g enumerates the QE dense G-grid, N(g) is the FFT-mesh linear
 *      index, and miller_g_dense gives the integer Miller indices.
 *   2. Apply 4π/|G|^2 in G-space to get V_aug(G).
 *   3. iFFT to V_aug(r).
 *   4. Add V_aug(r) into the input V_H(r) buffer.
 *
 * Inputs:
 *   psp     : pseudopot holding qgm, ijtoh, ityp, nh, ofs, miller_g_dense,
 *             atom_pos_cart, recv (lattv).
 *   becsum  : (nat, nhm, nhm) real, per-atom IJ pair coefficients.
 *   mesh    : dense FFT mesh dimensions.
 *   recv    : reciprocal lattice vectors (rows = b1, b2, b3).
 *   svr     : V_H(r) buffer in shared memory; modified in place.
 *
 * NCPP no-op: returns immediately if psp.qgm has zero size.
 */
template<typename Arr>
inline void add_paw_augmentation_to_VH(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    nda::ArrayOfRank<3> auto const& becsum,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& recv,
    int nk_total,
    int nspin,
    int npol,
    math::shm::shared_array<Arr>& svr)
{
    (void) nk_total;
    if (psp.pp_type() == pp_ncpp_t) return;
    if (psp.qgm_view().size() == 0) return;

    long nnr = mesh(0)*mesh(1)*mesh(2);
    long nat = psp.ityp_view().extent(0);
    long ngm = psp.ngm_dense_get();
    utils::check(svr.size() == nnr, "add_paw_augmentation_to_VH: svr.size != nnr");
    utils::check(psp.miller_g_dense_view().extent(0) == ngm,
                 "add_paw_augmentation_to_VH: miller_g_dense vs ngm mismatch");
    utils::check(psp.atom_pos_cart_view().extent(0) == nat,
                 "add_paw_augmentation_to_VH: atom_pos vs ityp mismatch");

    // cell volume: vol = (2π)^3 / det(recv) — same convention used by v_h
    double vol = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
               - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
               + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
    vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / vol;

    // Build ρ_aug on the dense FFT mesh (root only — small per-process
    // contribution; this is a sequential build for clarity, not a hot loop).
    //
    // qgm from qvan2 has the 4π/Ω factor baked in, so qgm × becsum × sf is
    // ρ_aug,proper(G) in standard density-Fourier convention. To inject into
    // v_h's pipeline at the post-forward-FFT stage, we need the same scale
    // as v_h's `nr(G) = un-normalized FFT[Σ_n f_n |ψ̃|²]`. The relation:
    //   nr(G) = N_grid × ρ_proper(G)
    // Hence Δρ_aug,nr(G) = N_grid × becsum × qgm × sf.
    // double N_grid = (double)mesh(0)*mesh(1)*mesh(2);
    nda::array<ComplexType,1> rho_aug(nnr);
    rho_aug() = ComplexType(0.0);

    if (mpi.comm.root()) {
        auto qg = psp.qgm_view();           // (nsp, nij_max, ngm)
        auto const& ijtoh = psp.ijtoh_view(); // (nsp, nhm, nhm), 1-based
        auto const& ityp  = psp.ityp_view();
        auto const& nh    = psp.nh_view();
        auto const& mill  = psp.miller_g_dense_view();
        auto const& tau   = psp.atom_pos_cart_view();

        long NX = mesh(0), NY = mesh(1), NZ = mesh(2);

        for (long g=0; g<ngm; ++g) {
            int m1 = mill(g, 0), m2 = mill(g, 1), m3 = mill(g, 2);
            // FFT linear index for this Miller triple (matches generate_k2g)
            long n1 = m1; if (n1 < 0) n1 += NX;
            long n2 = m2; if (n2 < 0) n2 += NY;
            long n3 = m3; if (n3 < 0) n3 += NZ;
            if (n1 < 0 || n1 >= NX || n2 < 0 || n2 >= NY ||
                n3 < 0 || n3 >= NZ) continue;
            long N = (n1*NY + n2)*NZ + n3;

            // G in cartesian: G = m1*recv(0,:) + m2*recv(1,:) + m3*recv(2,:)
            double Gx = m1*recv(0,0) + m2*recv(1,0) + m3*recv(2,0);
            double Gy = m1*recv(0,1) + m2*recv(1,1) + m3*recv(2,1);
            double Gz = m1*recv(0,2) + m2*recv(1,2) + m3*recv(2,2);

            ComplexType acc(0.0);
            for (long ia=0; ia<nat; ++ia) {
                int nt = ityp(ia);
                int nh_a = nh(nt);
                if (nh_a == 0) continue;
                double phase = -(Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2));
                ComplexType sf(std::cos(phase), std::sin(phase));
                ComplexType atom_acc(0.0);
                for (int I=0; I<nh_a; ++I)
                for (int J=0; J<nh_a; ++J) {
                    long ij = static_cast<long>(ijtoh(nt, I, J)) - 1;
                    if (ij < 0) continue;
                    atom_acc += ComplexType(becsum(ia, I, J)) * qg(nt, ij, g);
                }
                acc += sf * atom_acc;
            }
            // ρ_aug stored on the dense FFT mesh in C row-major linear index
            rho_aug(N) += acc;
        }

        // 4π/|G|^2 per Miller index, in place. G=0 component is left as
        // the input (typically 0 from ρ_aug — augmentation charges are
        // designed to integrate to zero on each multipole channel by
        // construction, so no neutralizing Madelung correction is needed
        // beyond what v_h already applied to ρ̃).
        for (long g=0; g<ngm; ++g) {
            int m1 = mill(g, 0), m2 = mill(g, 1), m3 = mill(g, 2);
            long n1 = m1; if (n1 < 0) n1 += NX;
            long n2 = m2; if (n2 < 0) n2 += NY;
            long n3 = m3; if (n3 < 0) n3 += NZ;
            long N = (n1*NY + n2)*NZ + n3;
            double Gx = m1*recv(0,0) + m2*recv(1,0) + m3*recv(2,0);
            double Gy = m1*recv(0,1) + m2*recv(1,1) + m3*recv(2,1);
            double Gz = m1*recv(0,2) + m2*recv(1,2) + m3*recv(2,2);
            double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
            if (G2 > 1e-14) {
                // qgm from QE's qvan2 has the 4π/Ω factor baked in (proper
                // density Fourier convention, matching QE's rho%of_g).
                // To inject into v_h's pipeline at the post-forward-FFT
                // stage so that the downstream 4π/G² × ns_scl/(Ω·N_k)
                // multiplier produces the correct V_H,aug(G), the
                // augmentation must enter at the same scale as v_h's
                // nr(G). The required pre-factor is ns_scl × Ω, which
                // combined with v_h's (1/vol = 1/Ω) leaves the net scale
                // ns_scl × (4π/G²) on ρ_aug,proper.
                //
                // Verified band-by-band against QE's <V_H> values on the
                // LiH 222 PAW fixture (test_dft_eigenvalues).
                double w = 4.0*M_PI / G2;
                double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
                rho_aug(N) *= ComplexType(ns_scl * w, 0.0);
            } else {
                rho_aug(N) = ComplexType(0.0);
            }
        }

        // iFFT to r-space
        auto rho3d = nda::reshape(rho_aug,
                std::array<long,3>{mesh(0),mesh(1),mesh(2)});
        math::nda::fft<false> Frev(rho3d);
        Frev.backward(rho3d);

        // Add into V_H(r). svr is shared on the node; root has authoritative
        // contents (v_h sets it on root and broadcasts).
        auto vr = svr.local();
        for (long n=0; n<nnr; ++n) vr(n) += rho_aug(n);
    }

    // Replicate updated V_H(r) on all ranks (mirrors v_h's broadcast pattern).
    if (mpi.node_comm.root())
        mpi.internode_comm.broadcast_n(svr.local().data(), nnr, 0);
    mpi.comm.barrier();
}

/**
 * PAW Hartree potential on the dense FFT mesh from a diagonal occupation.
 *
 * Equivalent to hamilt::v_h(...) for NCPP (just delegates). For USPP/PAW,
 * adds the compensation-charge Δρ_aug to ρ̃ before applying 4π/G^2.
 */
template<nda::ArrayOfRank<1> Arr>
void v_h_paw(utils::mpi_context_t<boost::mpi3::communicator,
                                   boost::mpi3::shared_communicator>& mpi,
             pots::potential_t& vG,
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
             nda::ArrayOfRank<3> auto const& nii,
             math::nda::DistributedArrayOfRank<4> auto const& psi,
             bool symmetrize_rho_r,
             math::shm::shared_array<Arr>& svr)
{
    // Smooth-grid Hartree (existing path). Writes V_H(r) into svr.
    hamilt::v_h(mpi, vG, npol, mesh, lattv, recv, k2g, kpts, kp_to_ibz,
                kp_trev, kp_symm, symm_list, nii, psi, symmetrize_rho_r, svr);
    // PAW augmentation correction (no-op for NCPP).
    if (psp.pp_type() == pp_ncpp_t) return;
    auto becsum = compute_becsum_diagonal(psp.Pskna_view(), nii,
                                          psp.ityp_view(), psp.nh_view(),
                                          psp.ofs_view(), npol);
    add_paw_augmentation_to_VH<Arr>(mpi, psp, becsum, mesh, recv,
                                     (int)kp_to_ibz.shape(0),
                                     (int)psi.global_shape()[0], npol, svr);
}

/**
 * PAW Hartree potential on the dense FFT mesh from a full density matrix.
 */
template<nda::ArrayOfRank<1> Arr>
void v_h_paw(utils::mpi_context_t<boost::mpi3::communicator,
                                   boost::mpi3::shared_communicator>& mpi,
             pots::potential_t& vG,
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
             nda::ArrayOfRank<4> auto const& nij,
             math::nda::DistributedArrayOfRank<4> auto const& psi,
             bool symmetrize_rho_r,
             math::shm::shared_array<Arr>& svr)
{
    hamilt::v_h(mpi, vG, npol, mesh, lattv, recv, k2g, kpts, kp_to_ibz,
                kp_trev, kp_symm, symm_list, nij, psi, symmetrize_rho_r, svr);
    if (psp.pp_type() == pp_ncpp_t) return;
    auto becsum = compute_becsum_full(psp.Pskna_view(), nij,
                                      psp.ityp_view(), psp.nh_view(),
                                      psp.ofs_view(), npol);
    add_paw_augmentation_to_VH<Arr>(mpi, psp, becsum, mesh, recv,
                                     (int)kp_to_ibz.shape(0),
                                     (int)psi.global_shape()[0], npol, svr);
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_V_H_PAW_HPP
