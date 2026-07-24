/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Pseudopot-aware Hartree potential entry points on the dense FFT grid.
 *
 * This header provides overloads of `hamilt::v_h(...)` that take a
 * `pseudopot const&` and dispatch internally on `psp.pp_type()`:
 *   - NCPP  : delegate to the smooth-only `hamilt::v_h(...)` overload in
 *             `src/hamiltonian/v_h.hpp` (no augmentation needed).
 *   - USPP/PAW : run the smooth pipeline, then add the compensation-charge
 *             contribution Δρ_aug → ΔV_aug to the V_H(r) buffer.
 *
 * Callers should always use the pseudopot-taking `hamilt::v_h(...)` so that
 * NCPP-vs-USPP/PAW dispatch is selected correctly without per-callsite
 * pp_type branches. The smooth-only overload of `hamilt::v_h` is kept for
 * the internal delegate, but should not be invoked directly from outside
 * the dispatch wrapper.
 *
 * The compensated density is
 *   ρ̄(G) = ρ̃(G) + Σ_a Σ_{IJ} becsum_{a,IJ} Q^{IJ}_{nt(a)}(G) e^{-iG·τ_a}
 * with
 *   diagonal nii :  becsum_{a,IJ} = Σ_{s,k,n} f_{skn} P*_{n,aI} P_{n,aJ}
 *   full nij     :  becsum_{a,IJ} = Σ_{s,k,a,b} P*_{a,aI} n_{skab} P_{b,aJ}
 * The δρ_a one-center residuals enter via the PAW ΔD_{IJ} correction matrix
 * (separate code path).
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
#include "hamiltonian/paw/paw_symmetry.hpp"        // full-BZ projector lift
#include "hamiltonian/paw/paw_runtime_caches.hpp"  // cached lift (plan A4)

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
 *   becsum_{a,IJ} = wk · Σ_{s,k,a,b} P*_{a,aI} n_{skab} P_{b,aJ},   wk = 1/N_k
 *
 * The 1/N_k k-weight matches `compute_becsum_diagonal` and CoQui's smooth-grid
 * density convention (`v_h_impl`: ρ = (ns_scl/N_k) Σ n |ψ|², with the 1/N_k
 * applied externally to an unweighted n). It is a property of CoQui's own
 * density matrix, NOT of the QE mean-field — so the resulting deeq / V_H is a
 * functional of the self-consistent n and is independent of the MF occupations
 * that merely seed it. Like the diagonal helper this assumes uniform k-weights
 * (nosym / full-BZ); a symmetry-reduced IBZ would need real per-k weights.
 *
 * Returns a real-valued (nat, nhm, nhm): the Hermitian pair symmetrization
 * becsum_IJ ← ½(becsum_IJ + becsum_JI^*). For an exactly Hermitian nij the
 * anti-Hermitian residual is zero (acc_JI^* == acc_IJ analytically); a large
 * residual means the INPUT density matrix violates its Hermiticity contract,
 * so it is a hard error, not a warning. The antisymmetric imaginary part of
 * the Hermitian becsum (Im ⟨p_I|γ|p_J⟩) is genuine but contracts to zero
 * against the I↔J-symmetric real Q_IJ(G) and radial-Hartree kernels, so the
 * stored real part is the exact effective becsum for every consumer. Same
 * npol=1 restriction as the diagonal helper. The spin factor ns_scl is NOT
 * applied here (callers apply it, mirroring compute_becsum_diagonal).
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
    // Uniform k-weight (matches compute_becsum_diagonal / the smooth density).
    double wk = 1.0 / (double)nk;
    double max_resid = 0.0;
    nda::array<ComplexType,2> acc_c(nhm, nhm);
    for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        int nh_a = nh(nt);
        if (nh_a == 0) continue;
        long p0 = ofs(ia);
        acc_c() = ComplexType(0.0);
        for (int I=0; I<nh_a; ++I)
        for (int J=0; J<nh_a; ++J) {
            // becsum_IJ = ⟨p_I|γ|p_J⟩ = Σ_ab ⟨p_I|ψ_a⟩ nij_ab ⟨ψ_b|p_J⟩
            //           = Σ_ab P_Ia · nij_ab · conj(P_Jb),   P_Ia = ⟨p_I|ψ_a⟩,
            // matching the convention γ = Σ_ab nij_ab |ψ_a⟩⟨ψ_b| used by the
            // SCF density matrix, the smooth-grid Hartree, and the THC Fock
            // build.
            ComplexType acc(0.0);
            for (long s=0; s<nspin; ++s)
            for (long k=0; k<nk; ++k)
            for (long a=0; a<nbnd; ++a)
            for (long b=0; b<nbnd; ++b) {
                acc += Pskna(s, k, p0 + I, a) *
                       nij(s, k, a, b) *
                       std::conj(Pskna(s, k, p0 + J, b));
            }
            acc_c(I, J) = wk * acc;
        }
        // Hermitian pair symmetrization (plan A3): becsum_IJ ← ½(b_IJ + b_JI^*).
        // Re of the Hermitian part is exact for all consumers (the antisym-
        // metric Im part is inert against symmetric real kernels, see doc);
        // the anti-Hermitian residual measures Hermiticity violation of the
        // input nij and is hard-checked below.
        for (int I=0; I<nh_a; ++I)
        for (int J=0; J<nh_a; ++J) {
            max_resid = std::max(max_resid,
                0.5 * std::abs(acc_c(I, J) - std::conj(acc_c(J, I))));
            becsum(ia, I, J) =
                0.5 * (std::real(acc_c(I, J)) + std::real(acc_c(J, I)));
        }
    }
    utils::check(max_resid <= 1e-8,
        "compute_becsum_full: anti-Hermitian becsum residual {} > 1e-8 — the "
        "input density matrix nij violates its Hermiticity contract "
        "(nij(s,k,a,b) must equal conj(nij(s,k,b,a))).", max_resid);
    return becsum;
}

/**
 * Build the USPP/PAW compensation charge ρ_aug(r) at the un-normalized nr(r)
 * scale used inside hamilt::v_h, so it can be added to the smooth density
 * before the single Coulomb solve (folded path; no separate augmentation FFT).
 *
 *   ρ_aug,nr(G) = vol·N_k · Σ_a Σ_IJ becsum_{a,IJ} Q^{IJ}_{nt(a)}(G) e^{-iG·τ_a}
 *
 * The vol·N_k prefactor cancels v_h's nrm = ns_scl/(vol·N_k) so that, after
 * v_h applies nrm·v_C(G), the augmentation lands at the same scale the
 * standalone add_paw_augmentation_to_VH produced (ns_scl·4π/G²·ρ_aug,proper),
 * provided v_C(G) = 4π/G². NO ns_scl / 4π·G⁻² here — v_h supplies both.
 * Returns ρ_aug(r) on the comm root (length nnr); empty on other ranks.
 */
inline nda::array<ComplexType,1> compute_rho_aug_density_r(
    utils::mpi_context_t<boost::mpi3::communicator,
                         boost::mpi3::shared_communicator>& mpi,
    pseudopot const& psp,
    nda::ArrayOfRank<3> auto const& becsum,
    nda::stack_array<long,3> const& mesh,
    nda::stack_array<double,3,3> const& recv,
    long nk)
{
    long nnr = mesh(0)*mesh(1)*mesh(2);
    nda::array<ComplexType,1> rho_aug_r;
    if (psp.qgm_view().size() == 0) return rho_aug_r;   // NCPP: empty
    long ngm = psp.ngm_dense_get();

    double vol = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
               - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
               + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
    vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / vol;

    if (mpi.comm.root()) {
        nda::array<ComplexType,1> rho_g(nnr);
        rho_g() = ComplexType(0.0);
        auto qg = psp.qgm_view();
        auto const& ijtoh = psp.ijtoh_view();
        auto const& ityp  = psp.ityp_view();
        auto const& nh    = psp.nh_view();
        auto const& mill  = psp.miller_g_dense_view();
        auto const& tau   = psp.atom_pos_cart_view();
        long nat = ityp.extent(0);
        long NX = mesh(0), NY = mesh(1), NZ = mesh(2);
        ComplexType scale(vol * (double)nk, 0.0);
        for (long g=0; g<ngm; ++g) {
            int m1 = mill(g,0), m2 = mill(g,1), m3 = mill(g,2);
            long n1 = m1; if (n1<0) n1+=NX;
            long n2 = m2; if (n2<0) n2+=NY;
            long n3 = m3; if (n3<0) n3+=NZ;
            if (n1<0||n1>=NX||n2<0||n2>=NY||n3<0||n3>=NZ) continue;
            long N = (n1*NY + n2)*NZ + n3;
            double Gx = m1*recv(0,0)+m2*recv(1,0)+m3*recv(2,0);
            double Gy = m1*recv(0,1)+m2*recv(1,1)+m3*recv(2,1);
            double Gz = m1*recv(0,2)+m2*recv(1,2)+m3*recv(2,2);
            ComplexType acc(0.0);
            for (long ia=0; ia<nat; ++ia) {
                int nt = ityp(ia); int nh_a = nh(nt);
                if (nh_a == 0) continue;
                double ph = -(Gx*tau(ia,0)+Gy*tau(ia,1)+Gz*tau(ia,2));
                ComplexType sf(std::cos(ph), std::sin(ph));  // e^{-iG·τ}
                ComplexType atom_acc(0.0);
                for (int I=0; I<nh_a; ++I)
                for (int J=0; J<nh_a; ++J) {
                    long ij = static_cast<long>(ijtoh(nt,I,J)) - 1;
                    if (ij < 0) continue;
                    atom_acc += ComplexType(becsum(ia,I,J)) * qg(nt,ij,g);
                }
                acc += sf * atom_acc;
            }
            rho_g(N) += scale * acc;
        }
        // iFFT to r-space (un-normalized backward; matches v_h's nr convention).
        auto rho3d = nda::reshape(rho_g, std::array<long,3>{NX,NY,NZ});
        math::nda::fft<false> Frev(rho3d);
        Frev.backward(rho3d);
        rho_aug_r = rho_g;
    }
    mpi.comm.barrier();
    return rho_aug_r;
}

/**
 * Full-BZ becsum (diagonal occupations) — symmetry-correct.
 *
 * compute_becsum_diagonal sums only the IBZ k-points with wk = 1/N_ibz, which is
 * correct ONLY for nosym/full-BZ. For a symmetry-reduced IBZ the compensation
 * charge must match the smooth density's full-BZ sum (v_h_impl loops the full
 * BZ via transform_k2g). This helper lifts the projectors to the full BZ — the
 * same machinery the exchange path already uses (compute_Pskna_full_bz) — and
 * accumulates over N_full with wk = 1/N_full. Occupations are symmetry-invariant
 * so they expand by kp_to_ibz indexing. For nosym (IBZ == full BZ) this reduces
 * to compute_becsum_diagonal unchanged.
 *
 * Fixes the symmetry-reduced V_H augmentation error in the DIRECT path
 * (Vhartree / deeq reference); the THC path was already correct (2026-06-11).
 */
template<typename occ_t, typename kti_t, typename ktr_t>
inline nda::array<double,3> compute_becsum_diagonal_symm(
    pseudopot const& psp, occ_t const& nii,
    kti_t const& kp_to_ibz, [[maybe_unused]] ktr_t const& kp_trev, int npol)
    // kp_trev kept for signature parity with compute_becsum_full_symm:
    // occupations are trev-invariant so the diagonal expansion needs no conj.
{
    // Cached full-BZ lift (plan A4); MPI-collective on psp's communicator at
    // the first call, cheap view afterwards. The BZ tables (kp_to_ibz,
    // kp_trev) are still taken from the caller for the band-matrix expansion
    // and must describe the same mesh psp was built from.
    auto Pfull = psp.Pskna_full_bz().local();  // (nspin, nk_full, npol*nkb, nbnd)

    long nspin   = nii.extent(0);
    long nk_full = (long)kp_to_ibz.extent(0);
    long nbnd    = nii.extent(2);
    utils::check(Pfull.extent(1) == nk_full,
        "compute_becsum_diagonal_symm: kp_to_ibz length {} != lifted nk {}",
        nk_full, Pfull.extent(1));
    nda::array<ComplexType,3> nii_full(nspin, nk_full, nbnd);
    for (long s = 0; s < nspin; ++s)
        for (long K = 0; K < nk_full; ++K)
            for (long n = 0; n < nbnd; ++n)
                nii_full(s, K, n) = nii(s, (long)kp_to_ibz(K), n);

    return compute_becsum_diagonal(Pfull, nii_full, psp.ityp_view(),
                                   psp.nh_view(), psp.ofs_view(), npol);
}

/**
 * Full-BZ becsum from a full density matrix — symmetry-correct (plan A3).
 *
 * View-2: the full-BZ states are the exactly rotated IBZ states in G-space,
 * ψ_{K,n} = R ψ_{k_ibz,n}, so γ_K = R γ_{k_ibz} R⁻¹ contracts the SAME
 * IBZ band-space matrix nij(s,k_ibz,·,·) against the lifted projectors
 * (compute_Pskna_full_bz; immune to band-truncation since no band-space
 * rotation is involved). At time-reversal points ψ_K = (Rψ)^*, hence
 * γ_K = Σ_ab ψ_{K,a} n^*_ab ψ†_{K,b}: the band matrix is complex-conjugated
 * (a no-op for the real diagonal case, matching compute_becsum_diagonal_symm).
 * The per-atom Bloch phase of the lift cancels between P(I,·) and
 * conj(P(J,·)) on the same atom, so becsum is lift-gauge-invariant. For
 * nosym (IBZ == full BZ, identity symm) this reduces to compute_becsum_full
 * unchanged.
 */
template<typename nij_t, typename kti_t, typename ktr_t>
inline nda::array<double,3> compute_becsum_full_symm(
    pseudopot const& psp, nij_t const& nij,
    kti_t const& kp_to_ibz, ktr_t const& kp_trev, int npol)
{
    // Cached full-BZ lift (plan A4) — see compute_becsum_diagonal_symm.
    auto Pfull = psp.Pskna_full_bz().local();  // (nspin, nk_full, npol*nkb, nbnd)

    long nspin   = nij.extent(0);
    long nk_full = (long)kp_to_ibz.extent(0);
    long nbnd    = nij.extent(2);
    utils::check(Pfull.extent(1) == nk_full,
        "compute_becsum_full_symm: kp_to_ibz length {} != lifted nk {}",
        nk_full, Pfull.extent(1));
    nda::array<ComplexType,4> nij_full(nspin, nk_full, nbnd, nbnd);
    for (long s = 0; s < nspin; ++s)
        for (long K = 0; K < nk_full; ++K) {
            long k_ibz = (long)kp_to_ibz(K);
            bool trev  = (bool)kp_trev(K);
            for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b)
                    nij_full(s, K, a, b) = trev
                        ? std::conj(nij(s, k_ibz, a, b))
                        : nij(s, k_ibz, a, b);
        }

    return compute_becsum_full(Pfull, nij_full, psp.ityp_view(),
                               psp.nh_view(), psp.ofs_view(), npol);
}

} // namespace hamilt::paw

namespace hamilt {

// MAM: This should not be here, it should be in hamiltonian/v_h.hpp.
//  Then the v_h routine there should be called something else, e.g. v_h_impl,
//  the augmentation densit should be added directly to the soft charge inside v_h_impl
//  and steps involving fft, multiplication of v_coulomb, ifft, should be done once.
/**
 * Pseudopot-aware Hartree potential overload — diagonal-occupation form.
 *
 * Always-correct entry point for NCPP/USPP/PAW. NCPP is handled by
 * delegating to the smooth-only `hamilt::v_h(... no pseudopot ...)`
 * overload in v_h.hpp; USPP/PAW additionally adds the compensation-charge
 * contribution to V_H(r).
 */
template<nda::ArrayOfRank<1> Arr>
void v_h(utils::mpi_context_t<boost::mpi3::communicator,
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
    // Build the USPP/PAW compensation charge ρ_aug(r) (empty for NCPP) and
    // fold it into the smooth density inside v_h — single Coulomb pass, no
    // separate post-hoc augmentation FFT.
    nda::array<ComplexType,1> rho_aug_r;
    if (psp.pp_type() != pp_ncpp_t) {
      // Full-BZ becsum (symmetry-correct). The IBZ-only sum was wrong for
      // symmetry-reduced meshes (mismatched 1/N_ibz vs the ×N_full in
      // compute_rho_aug_density_r), giving a ~3-5% V_H error under symmetry.
      auto becsum = ::hamilt::paw::compute_becsum_diagonal_symm(
          psp, nii, kp_to_ibz, kp_trev, npol);
      rho_aug_r = ::hamilt::paw::compute_rho_aug_density_r(
          mpi, psp, becsum, mesh, recv, (long)kp_to_ibz.shape(0));
    }
    hamilt::detail::v_h_impl(mpi, vG, npol, mesh, lattv, recv, k2g, kpts, kp_to_ibz,
                kp_trev, kp_symm, symm_list, nii, psi, symmetrize_rho_r, svr,
                rho_aug_r);
}

/**
 * Pseudopot-aware Hartree potential overload — full density-matrix form.
 */
template<nda::ArrayOfRank<1> Arr>
void v_h(utils::mpi_context_t<boost::mpi3::communicator,
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
    nda::array<ComplexType,1> rho_aug_r;
    if (psp.pp_type() != pp_ncpp_t) {
      // Full-BZ becsum (symmetry-correct, plan A3) — same lift as the
      // diagonal overload; reduces to the plain IBZ sum for nosym.
      auto becsum = ::hamilt::paw::compute_becsum_full_symm(
          psp, nij, kp_to_ibz, kp_trev, npol);
      rho_aug_r = ::hamilt::paw::compute_rho_aug_density_r(
          mpi, psp, becsum, mesh, recv, (long)kp_to_ibz.shape(0));
    }
    hamilt::detail::v_h_impl(mpi, vG, npol, mesh, lattv, recv, k2g, kpts, kp_to_ibz,
                kp_trev, kp_symm, symm_list, nij, psi, symmetrize_rho_r, svr,
                rho_aug_r);
}

} // namespace hamilt

#endif // HAMILTONIAN_PAW_V_H_PAW_HPP
