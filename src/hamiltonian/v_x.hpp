/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * ==========================================================================
 */


#ifndef HAMILTONIAN_V_X_HPP
#define HAMILTONIAN_V_X_HPP

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/symmetry.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "potentials/potentials.hpp"

namespace hamilt
{

using utils::mpi_context_t;
using boost::mpi3::communicator;
using boost::mpi3::shared_communicator;

/**
 * Smooth-only exact-exchange matrix elements K_{ij}(s, k_p) on the IBZ via
 * FFT pair densities — NCPP version. The PAW/USPP-augmented variant lives
 * in `paw/v_x_paw.hpp` and adds per-pair augmentation to the pair density
 * before the Coulomb contraction.
 *
 * Math (cell-normalized convention, ∫_Ω |ψ|² = 1):
 *
 *   K_{ij}(s, k_p) = -(1 / (N_k · n_s · Ω))
 *                     Σ_{k_q ∈ BZ} Σ_n f^total_{s, kp_to_ibz(k_q), n}
 *                     × Σ_G v_C(G + k_p − k_q)
 *                          · p_{n,i}^*(G; k_q, k_p)
 *                          · p_{n,j}(G; k_q, k_p)
 *
 * with the cell-FT pair density
 *
 *   p_{n,a}(G; k_q, k_p) = fwdfft_norm[ u*_n(k_q, r) · u_a(k_p, r) ](G)
 *
 * evaluated on the dense FFT mesh. CoQui's fwdfft is normalized (1/N_grid),
 * so p directly equals the standard density Fourier coefficient.
 *
 *   v_C(G+Δk) = 4π/|G+Δk|²,  with the G+Δk = 0 component zeroed (matches
 *   v_h convention; a Gygi-Baldereschi-style singularity correction is a
 *   follow-up).
 *
 *   n_s = 2 for nspin=1 (so f^total / n_s = per-spin occupation), 1 otherwise.
 *
 * Sign convention: K_{ij} is the SIGNED exchange contribution to the Fock
 * matrix (F = H_core + V_H + K with K already negative), matching CoQui's
 * existing set_fock / test_dft_eigenvalues convention.
 *
 * Storage: this version replicates ψ̃_n(k_full, r) on every rank
 *   psi_r_full(s, k_full, n, r) :  nspin × nk × nbnd × nnr complex
 * fine for the small test fixtures (LiH 2×2×2 PAW: ~95 MB). Larger problems
 * will want streaming over k_q.
 *
 * Output: K_{ij}(s, k_ibz, i, j) ACCUMULATED into `Kij.local()`. Caller is
 * responsible for zeroing or summing with other contributions.
 *
 * @param mpi   - [input] MPI handler
 * @param vG    - [input] Coulomb kernel (4π/|G+Δk|² on FFT mesh; |G+Δk|=0 zeroed)
 * @param npol  - [input] number of polarizations (only npol=1 supported)
 * @param mesh  - [input] dense FFT mesh
 * @param recv  - [input] reciprocal lattice vectors (rows)
 * @param k2g_  - [input] wfc-G → dense-FFT linear-index map for IBZ
 * @param kpts  - [input] full-BZ kpts in cartesian coordinates (nk, 3)
 * @param kp_to_ibz - [input] IBZ representative for each k_full
 * @param kp_trev   - [input] time-reversal flag per k_full
 * @param kp_symm   - [input] symmetry-op index per k_full
 * @param symm_list - [input] list of symmetry operations
 * @param nii_  - [input] diagonal occupations (s, k_ibz, n), spin-summed
 * @param psi   - [input] orbitals on wfc-G grid, (s, k_ibz, n, g)
 * @param Kij   - [in/out] (s, k_ibz, i, j) — K_{ij} accumulated into local slice
 */
void v_x(mpi_context_t<communicator,shared_communicator> &mpi,
         pots::potential_t& vG,
         int npol,
         nda::stack_array<long, 3> const& mesh,
         nda::stack_array<double, 3, 3> const& /*lattv*/,
         nda::stack_array<double, 3, 3> const& recv,
         nda::ArrayOfRank<1> auto const& k2g_,
         nda::ArrayOfRank<2> auto const& kpts,
         nda::ArrayOfRank<1> auto const& kp_to_ibz,
         nda::ArrayOfRank<1> auto const& kp_trev,
         nda::ArrayOfRank<1> auto const& kp_symm,
         std::vector<utils::symm_op> const& symm_list,
         nda::ArrayOfRank<3> auto const& nii_,
         math::nda::DistributedArrayOfRank<4> auto const& psi,
         math::nda::DistributedArrayOfRank<4> auto & Kij)
{
  decltype(nda::range::all) all;
  using nda::range;

  utils::check(npol == 1,
      "v_x: only npol=1 (scalar, non-SOC) supported in this implementation");
  utils::check(psi.grid()[3] == 1, "v_x: psi grid[3] must be 1");

  long nspin   = psi.global_shape()[0];
  long nk_ibz  = psi.global_shape()[1];
  long nbnd    = psi.global_shape()[2];
  long ngm     = k2g_.extent(0);
  long nk      = kp_to_ibz.shape(0);
  long nnr     = mesh(0)*mesh(1)*mesh(2);

  utils::check(psi.global_shape()[3] == npol*ngm,
      "v_x: psi g-dim != npol*ngm ({} vs {})", psi.global_shape()[3], npol*ngm);
  utils::check(nii_.extent(0) == nspin && nii_.extent(1) == nk_ibz &&
               nii_.extent(2) == nbnd,
      "v_x: nii shape mismatch");
  utils::check(Kij.global_shape()[0] == nspin &&
               Kij.global_shape()[1] == nk_ibz &&
               Kij.global_shape()[2] == nbnd &&
               Kij.global_shape()[3] == nbnd,
      "v_x: Kij must be (nspin, nk_ibz, nbnd, nbnd)");

  // Cell volume Ω = (2π)³ / det(recv)
  double det_recv = recv(0,0) * (recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                  - recv(1,0) * (recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                  + recv(2,0) * (recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double Omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / det_recv;
  // Sign + scale convention: QE's vexx uses the FULL occupation f_n directly
  // (not f_n / n_spin), so for nspin=1 closed-shell with f=2 the diagonal K
  // matrix element is -2·J_nn (consistent with F = H_core + V_H_full + K in
  // RHF). Keep the same convention here to match QE's HF eigenvalues.
  const ComplexType scl(-1.0 / (double(nk) * Omega), 0.0);

  // -------------------------------------------------------------------------
  // Step 1: build psi_r_full(s, k_full, n, r) = u_n(k_full, r) on the dense
  // FFT mesh, replicated on every rank via all-reduce-sum (slices are
  // disjoint so this acts as an all-gather).
  // -------------------------------------------------------------------------
  nda::array<ComplexType, 4> psi_r_full(nspin, nk, nbnd, nnr);
  psi_r_full() = ComplexType(0.0);
  {
    nda::array<ComplexType, 2> pr({1, nnr});
    auto pr4d = nda::reshape(pr,
                  std::array<long,4>{1, mesh(0), mesh(1), mesh(2)});
    math::nda::fft<true> F(pr4d);
    nda::array<long, 1> k2g_rotate(ngm);
    nda::stack_array<double, 3> Gs;
    Gs() = 0.0;
    nda::array<ComplexType,1> *Xft = nullptr;
    auto ploc = psi.local();

    for (auto [is_l, s] : itertools::enumerate(psi.local_range(0))) {
      for (auto [ik_l, k_ibz] : itertools::enumerate(psi.local_range(1))) {
        for (long kf = 0; kf < nk; ++kf) {
          if (kp_to_ibz(kf) != k_ibz) continue;
          k2g_rotate() = k2g_();
          if (kp_trev(kf) or kp_symm(kf) > 0) {
            utils::transform_k2g(kp_trev(kf), symm_list[kp_symm(kf)], Gs, mesh,
                                 kpts(k_ibz, all), k2g_rotate, Xft);
          }
          for (auto [ia_l, a] : itertools::enumerate(psi.local_range(2))) {
            pr() = ComplexType(0.0);
            nda::copy_select(true, k2g_rotate,
                             ComplexType(1.0),
                             ploc(is_l, ik_l, ia_l, range(0, ngm)),
                             ComplexType(0.0),
                             pr(0, all));
            F.backward(pr4d);  // G → r (unnormalized)
            if (kp_trev(kf)) {
              for (long r = 0; r < nnr; ++r)
                psi_r_full(s, kf, a, r) = std::conj(pr(0, r));
            } else {
              for (long r = 0; r < nnr; ++r)
                psi_r_full(s, kf, a, r) = pr(0, r);
            }
          }
        }
      }
    }
  }
  mpi.comm.all_reduce_in_place_n(psi_r_full.data(),
                                  psi_r_full.size(),
                                  std::plus<>{});

  // -------------------------------------------------------------------------
  // Step 2: build pair densities and contract into K_ij.
  //
  // For each (s, k_p_ibz, k_q ∈ BZ, n with f_n > 0):
  //   pair(a, g) = fwdfft_norm(u*_n(k_q, r) × u_a(k_p, r))(g)   for a ∈ [0, nbnd)
  //   K_ij(s, k_p_ibz) += scl × f_n × Σ_g v_C(g+Δk) × pair*(i, g) × pair(j, g)
  //
  // The (i, j) range over the local-Kij slice; each rank does its own slice.
  // -------------------------------------------------------------------------
  auto nii_host = nda::to_host(nii_);
  auto Kloc = Kij.local();
  // Zero the local Kij slice, since v_x accumulates and the caller may pass
  // an uninitialized buffer. Callers wanting to ADD to existing Kij should
  // save and restore externally; we follow the gen_Vhartree/add_Hartree
  // pattern where the matrix is freshly built each call.
  Kloc() = ComplexType(0.0);

  // pair-density scratch
  nda::array<ComplexType, 2> pair_buf(nbnd, nnr);
  auto pair4d = nda::reshape(pair_buf,
                  std::array<long,4>{nbnd, mesh(0), mesh(1), mesh(2)});
  math::nda::fft<true> Fpair(pair4d);
  nda::array<ComplexType, 1> v_coul(nnr);

  for (auto [is_l, s] : itertools::enumerate(Kij.local_range(0))) {
    for (auto [ik_l, k_p_ibz] : itertools::enumerate(Kij.local_range(1))) {
      auto K_sk = Kloc(is_l, ik_l, all, all);

      for (long kq = 0; kq < nk; ++kq) {
        long kq_ibz = kp_to_ibz(kq);
        // v_C(G + k_p - k_q) on dense FFT mesh (G+Δk = 0 component zeroed)
        v_coul() = ComplexType(0.0);
        vG.evaluate_in_mesh(range(0, nnr), v_coul, mesh,
                            nda::stack_array<double,3,3>{}, recv,
                            kpts(k_p_ibz, all), kpts(kq, all));

        for (long n = 0; n < nbnd; ++n) {
          double f = std::real(nii_host(s, kq_ibz, n));
          if (std::abs(f) < 1e-15) continue;

          // Build pair_buf(a, r) = u*_n(k_q, r) × u_a(k_p_ibz, r)
          for (long a = 0; a < nbnd; ++a) {
            for (long r = 0; r < nnr; ++r)
              pair_buf(a, r) = std::conj(psi_r_full(s, kq, n, r)) *
                               psi_r_full(s, k_p_ibz, a, r);
          }
          // r → G (normalized fwdfft, batched over a)
          Fpair.forward(pair4d);

          // Contraction K_{ij} += scl × f × Σ_g v_C(g) × pair*(i, g) × pair(j, g)
          //
          // Most efficient: form T(j, g) = v_C(g) × pair(j, g), then K += scl·f · pair^H · T.
          // For the first cut we do the explicit O(nbnd² × nnr) triple loop; matches
          // LiH 222 cost (~16K × 4K = 64M ops per (s, k_p, k_q, n)) comfortably.
          for (long i = 0; i < nbnd; ++i) {
            for (long j = 0; j < nbnd; ++j) {
              ComplexType acc(0.0);
              for (long g = 0; g < nnr; ++g)
                acc += v_coul(g) *
                       std::conj(pair_buf(i, g)) *
                       pair_buf(j, g);
              K_sk(i, j) += scl * ComplexType(f, 0.0) * acc;
            }
          }
        }
      }
    }
  }

  mpi.comm.barrier();
}

}  // namespace hamilt

#endif  // HAMILTONIAN_V_X_HPP
