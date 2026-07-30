/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

/**
 * Symmetry helpers for the PAW augmentation pipeline.
 *
 * Pskna (the projector-vs-pseudo-orbital overlaps in pseudopot) is stored
 * only at IBZ k-points. For augment_thc_with_paw we need the projector
 * overlaps at every full-BZ k. This file provides the View-2 transformation
 * (notes/paw_symmetry_notes.tex Eq. coef-symm) that lifts the IBZ data to
 * the full BZ:
 *
 *   c^{ib}_{n l m'}(n, k_full)
 *       = e^{-i k_full . t_R}
 *         · Σ_m D^(l)_{m', m}(R) · c^{ia}_{n l m}(n, k_ibz)
 *
 * where (k_ibz, R) come from kp_to_ibz/kp_symm and (ia, ib) are connected
 * by the space-group action: R·τ_ia + t_R = τ_ib + T (lattice T).
 *
 * Because CoQui currently requires force_symmorphic=true (ft = 0; checked
 * at qe_interface.cpp:248-249), the fractional-translation phase
 * e^{-i k_full . t_R} is identically 1 here. A diagnostic check below
 * fires loudly if a non-symmorphic ft ever sneaks in.
 *
 * Time-reversal pairs (kp_trev[k_full] = true) get an extra complex
 * conjugation: c(R̄ k_ibz) = conj(c(R k_ibz)) where R̄ folds the rotated
 * point through the origin.
 */

#ifndef HAMILTONIAN_PAW_PAW_SYMMETRY_HPP
#define HAMILTONIAN_PAW_PAW_SYMMETRY_HPP

#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"
#include "utilities/symmetry.hpp"
#include "numerics/shared_array/nda.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"   // for qe_real_ylm_flat

namespace hamilt::paw {

/**
 * Per-symmetry inverse atom-permutation table.
 *
 * Returns `perm_inv(R_idx, ib) = ia` such that R · τ_ia (+ t_R, currently
 * required to be 0) equals τ_ib up to a Bravais-lattice translation, with
 * species(ia) = species(ib) (always true for a space-group operation).
 *
 * Useful for the View-2 lift of projector coefficients: the c-coefficient
 * at the destination atom ib at k_full = R k_ibz is built from the source
 * atom ia at k_ibz.
 */
inline nda::array<int,2> build_atom_permutation_inverse(
    nda::ArrayOfRank<2> auto const& atom_pos_cart,         // (nat, 3) cartesian
    nda::ArrayOfRank<1> auto const& ityp,                  // (nat) species
    [[maybe_unused]] nda::ArrayOfRank<2> auto const& latt,                  // (3, 3) lattice vecs (row = primitive a_i)
    nda::ArrayOfRank<2> auto const& recv,                  // (3, 3) reciprocal vecs (row = primitive b_i)
    std::vector<utils::symm_op> const& symm_list)
{
    long nat  = atom_pos_cart.extent(0);
    long nsym = (long)symm_list.size();
    nda::array<int,2> perm_inv(nsym, nat);
    perm_inv() = -1;

    // Convert atom positions to crystal coordinates once.
    // recv stored row-by-row with recv(i,:) = b_i (cartesian); a_i · b_j = 2π δ_ij,
    // so c_j = (b_j · v_cart) / (2π) gives the j-th crystal coord of v_cart.
    nda::array<double, 2> tau_crys(nat, 3);
    for (long ia = 0; ia < nat; ++ia)
        for (int i = 0; i < 3; ++i) {
            double v = 0;
            for (int j = 0; j < 3; ++j) v += recv(i, j) * atom_pos_cart(ia, j);
            tau_crys(ia, i) = v / (2.0 * M_PI);
        }

    nda::stack_array<double, 3> Rt_crys;  // R · τ_ia in crystal coords (R stored is in lattice basis).

    for (long isym = 0; isym < nsym; ++isym) {
        auto const& s = symm_list[isym];
        utils::check(std::abs(s.ft(0)) + std::abs(s.ft(1)) + std::abs(s.ft(2)) < 1e-12,
            "build_atom_permutation_inverse: non-symmorphic ft on symm {} not supported "
            "(force_symmorphic=true is required by qe_interface).", isym);

        for (long ia = 0; ia < nat; ++ia) {
            // Apply R (in lattice basis) to τ_ia in crystal coords.
            for (int i = 0; i < 3; ++i) {
                double v = 0;
                for (int j = 0; j < 3; ++j) v += s.R(i, j) * tau_crys(ia, j);
                Rt_crys(i) = v;
            }

            // Find ib s.t. R·τ_ia - τ_ib is integer-valued in crystal coords.
            int ib_match = -1;
            for (long ib = 0; ib < nat; ++ib) {
                if (ityp(ib) != ityp(ia)) continue;
                bool is_lattice = true;
                for (int i = 0; i < 3; ++i) {
                    double d = Rt_crys(i) - tau_crys(ib, i);
                    double r = d - std::round(d);
                    if (std::abs(r) > 1e-5) { is_lattice = false; break; }
                }
                if (is_lattice) { ib_match = (int)ib; break; }
            }
            utils::check(ib_match >= 0,
                "build_atom_permutation_inverse: no atom matches R·τ[{}] for symm {}",
                ia, isym);
            utils::check(perm_inv(isym, ib_match) == -1,
                "build_atom_permutation_inverse: collision at (isym={}, ib={})",
                isym, ib_match);
            perm_inv(isym, ib_match) = (int)ia;
        }

        for (long ib = 0; ib < nat; ++ib)
            utils::check(perm_inv(isym, ib) >= 0,
                "build_atom_permutation_inverse: no source atom found for (isym={}, ib={})",
                isym, ib);
    }
    return perm_inv;
}

/**
 * Build Wigner-D matrices in QE's real-Y_lm basis, one per (R, l).
 *
 * Returns a `vector<array<double, 3>>` indexed by l ∈ [0, lmax]; each
 * `D_l[l]` has shape `(2l+1, 2l+1, nsym)` and satisfies
 *     Y_l m(R^{-1} r̂)  =  Σ_m'  D_l[l](m', m, R_idx) · Y_l m'(r̂),
 * where the m-index 0..2l is the position within the l-block as produced
 * by `qe_real_ylm_flat`. (See paw_aug_q_eval.hpp:90-165.)
 *
 * The Wigner-D matrix is extracted by sampling Y_lm at (2l+1) random unit
 * vectors before and after rotation, then solving the resulting linear
 * system per l. This mirrors the "random points + matrix inverse" pattern
 * already used in `aainit_tables_build`.
 */
inline std::vector<nda::array<double,3>> build_wigner_d_real(
    std::vector<utils::symm_op> const& symm_list,
    nda::ArrayOfRank<2> auto const& latt,    // (3, 3) lattice vecs, row = primitive a_i
    int lmax)
{
    long nsym = (long)symm_list.size();
    // Use over-determined sample (npts >> 2l+1) and least-squares via normal
    // equations. The minimal `npts = 2l+1` direct inverse can be singular if
    // the random sample lands on a Y_lm nodal surface (observed empirically
    // on Si l=2 sub-block).
    int npts  = std::max(8 * (lmax + 1), 16);
    int lmax2 = (lmax + 1) * (lmax + 1);

    // Build the basis-change matrix T (columns = lattice vectors a_i) and its
    // inverse T⁻¹ for converting R from lattice basis to cartesian:
    //     R_cart = T · R_lattice · T⁻¹
    // latt(i, :) = a_i (row), so T(i, j) = a_j(i) = latt(j, i) (transpose).
    nda::matrix<double> T(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T(i, j) = latt(j, i);
    auto Tinv = nda::inverse(T);

    // Sample npts unit vectors deterministically.
    nda::array<double, 2> rhat(npts, 3);
    {
        std::mt19937 rng(20260504);
        std::uniform_real_distribution<double> u(-1, 1);
        for (int i = 0; i < npts; ++i) {
            double x, y, z, n2;
            do {
                x = u(rng); y = u(rng); z = u(rng);
                n2 = x*x + y*y + z*z;
            } while (n2 < 0.01);
            double n = std::sqrt(n2);
            rhat(i, 0) = x/n; rhat(i, 1) = y/n; rhat(i, 2) = z/n;
        }
    }

    // Allocate per-l output blocks.
    std::vector<nda::array<double,3>> Dl(lmax + 1);
    for (int l = 0; l <= lmax; ++l) {
        int sz = 2*l + 1;
        Dl[l] = nda::array<double, 3>::zeros({(long)sz, (long)sz, nsym});
    }

    nda::array<double, 1> Yflat(lmax2);  // qe_real_ylm_flat writes 0-based [0, lmax2)
    nda::array<double, 2> Y_orig(npts, lmax2);

    for (long isym = 0; isym < nsym; ++isym) {
        // Convert R_stored (lattice basis) and Rinv to cartesian for unit-vector rotation.
        // R_cart = T · R_lattice · T⁻¹
        nda::matrix<double> Rlat(3, 3), Rinv_lat(3, 3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Rlat(i, j)     = symm_list[isym].R(i, j);
                Rinv_lat(i, j) = symm_list[isym].Rinv(i, j);
            }
        nda::matrix<double> R_cart    = T * Rlat     * Tinv;
        nda::matrix<double> Rinv_cart = T * Rinv_lat * Tinv;

        nda::array<double, 2> Y_rot(npts, lmax2);
        for (int i = 0; i < npts; ++i) {
            // Y at original r̂.
            std::array<double, 3> r = {rhat(i,0), rhat(i,1), rhat(i,2)};
            qe_real_ylm_flat(lmax, r, Yflat);
            for (int j = 0; j < lmax2; ++j)
                Y_orig(i, j) = Yflat(j);
            // Y at R⁻¹_cart · r̂.
            std::array<double, 3> rrot = {0, 0, 0};
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 3; ++j)
                    rrot[k] += Rinv_cart(k, j) * r[j];
            qe_real_ylm_flat(lmax, rrot, Yflat);
            for (int j = 0; j < lmax2; ++j)
                Y_rot(i, j) = Yflat(j);
        }

        // Per-l: Y_rot[:, m] = Σ_m' Y_orig[:, m'] · D[m', m]. Solve over-determined
        // system via normal equations: D = (Yoᵀ Yo)⁻¹ · Yoᵀ · Yr.
        for (int l = 0; l <= lmax; ++l) {
            int sz = 2*l + 1;
            int l2 = l * l;
            nda::matrix<double> Yo(npts, sz);
            nda::matrix<double> Yr(npts, sz);
            for (int i = 0; i < npts; ++i)
                for (int m = 0; m < sz; ++m) {
                    Yo(i, m) = Y_orig(i, l2 + m);
                    Yr(i, m) = Y_rot (i, l2 + m);
                }
            nda::matrix<double> Yot_Yo = nda::transpose(Yo) * Yo;   // (sz, sz)
            nda::matrix<double> Yot_Yr = nda::transpose(Yo) * Yr;   // (sz, sz)
            auto Yot_Yo_inv = nda::inverse(Yot_Yo);
            nda::matrix<double> D = Yot_Yo_inv * Yot_Yr;
            for (int m1 = 0; m1 < sz; ++m1)
                for (int m2 = 0; m2 < sz; ++m2)
                    Dl[l](m1, m2, isym) = D(m1, m2);
        }
    }
    return Dl;
}

/**
 * Lift the IBZ Pskna table to the full BZ by View-2 symmetry transformation.
 *
 * On entry, `psp.Pskna_view()` has shape `(nspin, nkpts_ibz, npol*nkb, nbnd)`.
 * On return, `Pskna_full` is a freshly-allocated SHM array of shape
 * `(nspin, nkpts_full, npol*nkb, nbnd)` with
 *     Pskna_full(s, k_full, p, n) = Σ_J D · Pskna_ibz(s, k_ibz, p_src, n)
 * applied on the node-root rank.
 *
 * For trev k-points (kp_trev[k_full] = true), the result is additionally
 * complex-conjugated, per Bloch's theorem under time reversal.
 *
 * Identity symmetry (kp_symm[k_full] = 0) is a fast path: just copy.
 */
template<class MPI_t>
inline math::shm::shared_array<nda::array_view<ComplexType, 4>>
compute_Pskna_full_bz(
    pseudopot const& psp,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,        // (nkpts)
    nda::ArrayOfRank<1> auto const& kp_symm,          // (nkpts)
    nda::ArrayOfRank<1> auto const& kp_trev,          // (nkpts)
    nda::ArrayOfRank<2> auto const& kpts_cart,        // (nkpts, 3)
    [[maybe_unused]] std::vector<utils::symm_op> const& symm_list,
    nda::ArrayOfRank<2> auto const& atom_perm_inv,    // (nsym, nat)
    std::vector<nda::array<double, 3>> const& wigner_d,
    int npol,
    MPI_t& mpi)
{
    using nda::range;

    auto Pskna_ibz = psp.Pskna_view();
    long nspin     = Pskna_ibz.extent(0);
    long nkb_pol   = Pskna_ibz.extent(2);
    long nbnd      = Pskna_ibz.extent(3);
    long nkpts     = kp_to_ibz.extent(0);

    auto const& ityp = psp.ityp_view();
    auto const& nh_v = psp.nh_view();
    auto const& ofs  = psp.ofs_view();
    auto const& sps  = psp.paw_species_view();
    auto const& tau  = psp.atom_pos_cart_view();   // (nat, 3) cartesian (Bohr)
    long nat = ityp.extent(0);

    auto Pkfull = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        mpi, std::array<long, 4>{nspin, nkpts, nkb_pol, nbnd});

    Pkfull.win().fence();
    if (Pkfull.node_comm()->root()) {
        auto Pdst = Pkfull.local();
        Pdst() = ComplexType(0.0);

        for (long k_full = 0; k_full < nkpts; ++k_full) {
            int k_ibz = (int)kp_to_ibz(k_full);
            int R_idx = (int)kp_symm(k_full);
            bool trev = (bool)kp_trev(k_full);

            // Identity symm + no trev: straight copy.
            if (R_idx == 0 && !trev) {
                for (long s = 0; s < nspin; ++s)
                    Pdst(s, k_full, range::all, range::all) =
                        Pskna_ibz(s, k_ibz, range::all, range::all);
                continue;
            }

            for (long s = 0; s < nspin; ++s) {
                for (long ib = 0; ib < nat; ++ib) {
                    int nt   = ityp(ib);
                    int nh_a = nh_v(nt);
                    if (nh_a == 0) continue;

                    long ia_src = atom_perm_inv(R_idx, ib);
                    long p0_dst = ofs(ib)     * npol;
                    long p0_src = ofs(ia_src) * npol;

                    // Bloch phase from the lattice translation L = R^{-1}·τ_ib − τ_ia
                    // that brings R·τ_ia back into the unit cell:
                    //     phase = (xk_full · τ_ib) ∓ (xk_ibz · τ_ia)   (− no-trev, + trev)
                    // (mirrors QE's becp_rotate_k in PW/src/us_exx.f90:1031,
                    //  with kpts in cartesian inverse-Bohr so no 2π factor).
                    double phase = 0.0;
                    for (int d = 0; d < 3; ++d)
                        phase += kpts_cart(k_full, d) * tau(ib, d);
                    if (trev) {
                        for (int d = 0; d < 3; ++d)
                            phase += kpts_cart(k_ibz, d) * tau(ia_src, d);
                    } else {
                        for (int d = 0; d < 3; ++d)
                            phase -= kpts_cart(k_ibz, d) * tau(ia_src, d);
                    }
                    ComplexType tau_fact = std::exp(ComplexType(0.0, phase));

                    // For each destination projector channel I (with l, m_dst),
                    // sum over source J of the same (n, l) at the source atom.
                    for (int I = 0; I < nh_a; ++I) {
                        int l_I    = sps[nt].nhtol(I);
                        int lm_I   = sps[nt].nhtolm(I);
                        int m_dst  = (lm_I - 1) - l_I * l_I;
                        int beta_I = sps[nt].indv(I);

                        // Sum over J in the same (n, l) channel of source atom.
                        for (int J = 0; J < nh_a; ++J) {
                            int l_J = sps[nt].nhtol(J);
                            if (l_J != l_I) continue;
                            if (sps[nt].indv(J) != beta_I) continue;
                            int lm_J  = sps[nt].nhtolm(J);
                            int m_src = (lm_J - 1) - l_J * l_J;

                            double D = wigner_d[l_I](m_dst, m_src, R_idx);
                            if (D == 0.0) continue;

                            ComplexType D_phase = tau_fact * ComplexType(D);
                            for (int sigma = 0; sigma < npol; ++sigma)
                                for (long n = 0; n < nbnd; ++n) {
                                    ComplexType c_src =
                                        Pskna_ibz(s, k_ibz, p0_src + J*npol + sigma, n);
                                    if (trev) c_src = std::conj(c_src);
                                    Pdst(s, k_full, p0_dst + I*npol + sigma, n)
                                        += D_phase * c_src;
                                }
                        }
                    }
                }
            }
        }
    }
    Pkfull.win().fence();
    return Pkfull;
}

/**
 * Convenience overload: builds the atom permutation and real Wigner-D tables
 * from the pseudopot + lattice data before delegating to the base lift.
 * Callers that need the tables for other contractions (v_x) keep using the
 * explicit-table overload; becsum-style consumers use this one. Table caching
 * on pseudopot is provided by paw_runtime_caches.
 */
template<class MPI_t>
inline math::shm::shared_array<nda::array_view<ComplexType, 4>>
compute_Pskna_full_bz(
    pseudopot const& psp,
    nda::ArrayOfRank<1> auto const& kp_to_ibz,
    nda::ArrayOfRank<1> auto const& kp_symm,
    nda::ArrayOfRank<1> auto const& kp_trev,
    nda::ArrayOfRank<2> auto const& kpts_cart,
    nda::stack_array<double,3,3> const& lattv,
    nda::stack_array<double,3,3> const& recv,
    std::vector<utils::symm_op> const& symm_list,
    int npol,
    MPI_t& mpi)
{
    auto const& sps = psp.paw_species_view();
    auto atom_perm_inv = build_atom_permutation_inverse(
        psp.atom_pos_cart_view(), psp.ityp_view(), lattv, recv, symm_list);
    int lmax_proj = 0;
    for (long nt = 0; nt < (long)sps.size(); ++nt)
        for (long b = 0; b < sps[nt].lll.extent(0); ++b)
            lmax_proj = std::max(lmax_proj, (int)sps[nt].lll(b));
    auto wigner_d = build_wigner_d_real(symm_list, lattv, lmax_proj);
    return compute_Pskna_full_bz(psp, kp_to_ibz, kp_symm, kp_trev, kpts_cart,
                                 symm_list, atom_perm_inv, wigner_d, npol, mpi);
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_SYMMETRY_HPP
