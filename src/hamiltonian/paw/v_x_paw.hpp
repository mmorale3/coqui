/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Pseudopot-aware exchange matrix element builder K_{ij}(s, k_p).
 *
 *   - NCPP : delegate to the smooth-only `hamilt::v_x(...)` overload in
 *            `src/hamiltonian/v_x.hpp` (no augmentation).
 *   - USPP/PAW : add the per-pair Q^IJ_a(G+Δk) augmentation to each pair
 *            density on the dense FFT mesh BEFORE the Coulomb contraction.
 *            This single augmentation enters all four exchange-channel
 *            cross terms (bra-smooth × ket-aug, bra-aug × ket-smooth,
 *            bra-aug × ket-aug, bra-smooth × ket-smooth) because the
 *            full pair density is what contracts with Σ_G v_C × p* × p
 *            on both bra and ket sides.
 *
 * Math (PAW per-pair augmentation):
 *
 *   p_{n,a}^{full}(G; k_q, k_p)
 *     = fwdfft_norm[ u*_n(k_q, r) · u_a(k_p, r) ](G)
 *     + Σ_atm Σ_{IJ} P*_{n, atm I}(k_q) · P_{a, atm J}(k_p)
 *                     · Q^{IJ}_atm(G + k_p − k_q)
 *                     · e^{−i(G + k_p − k_q) · τ_atm}
 *
 * with Q^{IJ}_a(K) carrying the (4π/Ω) prefactor from QE's `tab_qrad`
 * convention (built via `evaluate_Q_IJ_at_K` / `qrad_interp_at_K` —
 * see `paw_aug_q_eval.hpp`).
 *
 *   K_{ij}(s, k_p) = −(1 / (N_k · n_s · Ω))
 *                     Σ_{k_q ∈ BZ} Σ_n f_{s, kp_to_ibz(k_q), n}
 *                     × Σ_G v_C(G + k_p − k_q)
 *                          · p_{n,i}^{full*}(G)
 *                          · p_{n,j}^{full}(G)
 *
 * Pskna at full-BZ k_q is obtained via View-2 symmetry lift
 * (`compute_Pskna_full_bz` in `paw_symmetry.hpp`).
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_V_X_PAW_HPP
#define HAMILTONIAN_PAW_V_X_PAW_HPP

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/fft/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "hamiltonian/v_x.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"
#include "hamiltonian/paw/paw_symmetry.hpp"
#include "hamiltonian/paw/paw_runtime_caches.hpp"  // A4 caches (lift/aatab/qtab/Qfac)

namespace hamilt::paw {

/**
 * Precompute Q^{IJ}_atm(G + Δk) · e^{−i(G+Δk)·τ_atm} on the dense FFT mesh
 * for a single (k_p, k_q) and all (atm, I, J) pairs.
 *
 * Output: `Qfac(atm, ij_packed, g)` where `ij_packed` indexes the
 * upper-triangular (I,J) block per species via `ijtoh` (1-based) → 0-based.
 * For non-PAW/USPP atoms (nh_a == 0) the per-atom slice is left zero.
 *
 * For lookup speed, this version uses precomputed `qrad_tab` per species
 * (built once outside the loop).
 *
 * @param psp        - [input] pseudopot (uses paw_species, ityp, atom_pos, etc.)
 * @param aatab      - [input] angular-coupling tables (aainit_tables_build)
 * @param qtab_sp    - [input] per-species qrad_tab vector, indexed by species id
 * @param mesh       - [input] dense FFT mesh
 * @param recv       - [input] reciprocal lattice vectors
 * @param Omega      - [input] cell volume Ω
 * @param k_p        - [input] cartesian k_p (3-vector)
 * @param k_q        - [input] cartesian k_q (3-vector)
 * @param Qfac       - [output] (nat, nij_max, ngm_FFT) augmentation factor
 */
template<typename Psp_t>
inline void build_paw_aug_pair_factor(
    Psp_t const& psp,
    aainit_tables const& aatab,
    std::vector<qrad_tab> const& qtab_sp,
    nda::stack_array<long, 3> const& mesh,
    nda::stack_array<double, 3, 3> const& recv,
    double Omega,
    nda::ArrayOfRank<1> auto const& k_p,
    nda::ArrayOfRank<1> auto const& k_q,
    double Gcut,
    nda::array<ComplexType, 3>& Qfac)
{
  auto const& ityp = psp.ityp_view();
  auto const& nh_v = psp.nh_view();
  auto const& sps  = psp.paw_species_view();
  auto const& tau  = psp.atom_pos_cart_view();

  long nat   = ityp.extent(0);
  long NX    = mesh(0), NY = mesh(1), NZ = mesh(2);
  long nnr   = NX*NY*NZ;
  long NX2   = NX/2, NY2 = NY/2, NZ2 = NZ/2;
  utils::check(Qfac.extent(0) == nat, "Qfac atom-dim mismatch");
  utils::check(Qfac.extent(2) == nnr, "Qfac G-dim mismatch");
  Qfac() = ComplexType(0.0);

  double dkx = k_p(0) - k_q(0);
  double dky = k_p(1) - k_q(1);
  double dkz = k_p(2) - k_q(2);

  for (long ia = 0; ia < nat; ++ia) {
    int nt = ityp(ia);
    int nh_a = nh_v(nt);
    if (nh_a == 0) continue;
    auto const& sp = sps[nt];
    if (!sp.is_paw && !sp.is_uspp) continue;
    auto const& qtab = qtab_sp[nt];

    double tx = tau(ia, 0), ty = tau(ia, 1), tz = tau(ia, 2);

    // Y_lm(K̂) buffer, computed ONCE per g below (depends only on the direction
    // of K, not on the projector pair I,J). Previously it was recomputed inside
    // evaluate_Q_IJ_at_K_fast for every (g, I, J) — nh^2 redundant trig
    // recursions that dominated the augmentation cost.
    nda::array<double, 1> Yflat_g(aatab.llx);
    int Lmax_ylm = 2 * (aatab.lli - 1);
    auto const& ijtoh = psp.ijtoh_view();

    for (long g = 0; g < nnr; ++g) {
      // Recover Miller index from linear FFT index (matches eval_mesh_3d_impl)
      long iz = g % NZ; if (iz > NZ2) iz -= NZ;
      long n_ = g / NZ;
      long iy = n_ % NY; if (iy > NY2) iy -= NY;
      long ix = n_ / NY; if (ix > NX2) ix -= NX;
      double Gx = ix*recv(0,0) + iy*recv(1,0) + iz*recv(2,0) + dkx;
      double Gy = ix*recv(0,1) + iy*recv(1,1) + iz*recv(2,1) + dky;
      double Gz = ix*recv(0,2) + iy*recv(1,2) + iz*recv(2,2) + dkz;
      double Kmag = std::sqrt(Gx*Gx + Gy*Gy + Gz*Gz);
      // Restrict the augmentation to the |G+Δk| ≤ Gcut sphere. Beyond Gcut the
      // smooth pair density is ~0 and the on-site AE−PS contribution is
      // Coulomb-negligible (ABINIT truncates the same way at ecutsigx). This
      // skips the FFT-box corners (|G| up to ~2× the inscribed-sphere radius),
      // which is what made the full-grid augmentation intractable, and lets the
      // qrad Bessel table be built only up to Gcut.
      if (Kmag > Gcut) continue;   // Qfac stays 0 here (zeroed above)

      // Structure factor e^{-i(G+Δk)·τ_a}
      double phase = -(Gx*tx + Gy*ty + Gz*tz);
      ComplexType sf(std::cos(phase), std::sin(phase));

      // Y_lm(K̂) once per g (direction only).
      std::array<double, 3> dir = (Kmag > 1e-14)
          ? std::array<double, 3>{Gx / Kmag, Gy / Kmag, Gz / Kmag}
          : std::array<double, 3>{0.0, 0.0, 1.0};
      qe_real_ylm_flat(Lmax_ylm, dir, Yflat_g);
      if (Kmag <= 1e-14)
        for (int lp = 1; lp < aatab.llx; ++lp) Yflat_g(lp) = 0.0;

      // For each (I, J) channel of atom ia, store Q^{IJ}_a(K) at ijtoh(nt,I,J).
      for (int I = 0; I < nh_a; ++I) {
        for (int J = 0; J < nh_a; ++J) {
          long ij_packed = static_cast<long>(ijtoh(nt, I, J)) - 1;
          if (ij_packed < 0) continue;
          utils::check(ij_packed < Qfac.extent(1),
              "build_paw_aug: ij_packed {} >= Qfac dim {} (nt={}, I={}, J={})",
              ij_packed, Qfac.extent(1), nt, I, J);
          ComplexType q_val =
              evaluate_Q_IJ_at_K_fast(sp, aatab, qtab, I, J, Kmag, Yflat_g, Omega);
          Qfac(ia, ij_packed, g) = q_val * sf;
        }
      }
    }
  }
}

/**
 * Variant of `evaluate_Q_IJ_at_K` that uses a precomputed `qrad_tab` for
 * the radial Bessel transform, falling back to the slow per-K computation
 * when `qtab.n_K == 0`. Splits responsibility for radial transforms so the
 * caller can amortize the table build over many (k_p, k_q, G) calls.
 */
inline ComplexType evaluate_Q_IJ_at_K_fast(
    pseudopot::species_paw_t const& sp,
    aainit_tables const& aatab,
    qrad_tab const& qtab,
    int ih, int jh,
    double Kmag,
    nda::array<double, 1> const& Yflat_qe,
    double omega_volume)
{
  if (sp.nhtolm.size() == 0 || sp.indv.size() == 0 || sp.lll.size() == 0) {
    return ComplexType(0.0);
  }
  int ivl = sp.nhtolm(ih) - 1;
  int jvl = sp.nhtolm(jh) - 1;
  int nb  = sp.indv(ih) - 1;
  int mb  = sp.indv(jh) - 1;
  int n1 = std::max(nb, mb), n2 = std::min(nb, mb);
  int ijv = (n1 * (n1 + 1)) / 2 + n2;

  // Radial Bessel: prefer the precomputed table. Use a thread-local scratch
  // buffer for the interpolated qrad(L): this function is called ~nnr*nh^2
  // times per augmentation build, so a per-call heap allocation here (plus the
  // Ylm buffer) is what made the direct augmentation take hours. The
  // allocation-free path mirrors ABINIT's precomputed-form-factor hot loop.
  thread_local nda::array<double, 1> qradL;
  if (qtab.n_K > 0) {
    qrad_interp_at_K_into(qtab, ijv, Kmag, qradL);
  } else {
    qradL = qrad_at_K(sp, ijv, Kmag);   // rare fallback (no table)
  }

  // Y_lp(K̂) is precomputed once per g by the caller (build_paw_aug_pair_factor)
  // and passed in as Yflat_qe — it depends only on the direction of K, not on
  // (ih, jh), so hoisting it out of the I,J loop removes nh^2 redundant trig
  // recursions per grid point.
  ComplexType qout(0.0, 0.0);
  int Lp1 = (int)qradL.extent(0);
  int npairs = aatab.lli * aatab.lli;
  utils::check(ivl < npairs && jvl < npairs,
      "evaluate_Q_IJ_at_K_fast: ivl/jvl out of aatab range");
  int n_lp = aatab.lpx(ivl, jvl);
  for (int k = 0; k < n_lp; ++k) {
    int lp = aatab.lpl(ivl, jvl, k);
    int L  = (int)std::floor(std::sqrt((double)lp + 1e-9));
    utils::check(L >= 0 && L < Lp1,
        "evaluate_Q_IJ_at_K_fast: L={} out of qfuncl L-range [{},{})",
        L, 0, Lp1);
    ComplexType iL_factor;
    switch (L % 4) {
      case 0: iL_factor = ComplexType( 1.0,  0.0); break;
      case 1: iL_factor = ComplexType( 0.0, -1.0); break;
      case 2: iL_factor = ComplexType(-1.0,  0.0); break;
      case 3: iL_factor = ComplexType( 0.0,  1.0); break;
    }
    double coeff = aatab.ap(lp, ivl, jvl) * Yflat_qe(lp) * qradL(L);
    qout += iL_factor * ComplexType(coeff, 0.0);
  }
  // (4π/Ω) prefactor matching QE's tab_qrad convention.
  return qout * ComplexType((4.0 * M_PI) / omega_volume, 0.0);
}

/**
 * Δk-keyed Qfac lookup/build. build_paw_aug_pair_factor depends on
 * (k_p, k_q) only through Δk = k_p − k_q (it adds Δk to every mesh G), so
 * entries are keyed by Δk quantized at 1e-8 a.u.⁻¹ — far below any k-mesh
 * spacing, so distinct Δk never collide; the worst quantization outcome is a
 * redundant rebuild, never a wrong entry. First-come-stays under the byte
 * budget in psp.paw_rt(): returns a reference to the cached entry on hit or
 * after an in-budget build, and to `scratch` (freshly built) once the budget
 * is exhausted. The cache persists across v_x calls (spin loop, SCF
 * iterations); a change of (mesh, Gcut, mode) clears it.
 */
template<typename Psp_t>
inline nda::array<ComplexType,3> const& get_or_build_qfac_pair_factor(
    Psp_t const& psp,
    aainit_tables const& aatab,
    std::vector<qrad_tab> const& qtab_sp,
    nda::stack_array<long,3> const& mesh,
    nda::stack_array<double,3,3> const& recv,
    double Omega,
    nda::ArrayOfRank<1> auto const& k_p,
    nda::ArrayOfRank<1> auto const& k_q,
    double Gcut,
    bool shape_restored,
    nda::array<ComplexType,3>& scratch)
{
  auto& rt = psp.paw_rt();
  // Budget knob: read from the live exx options so a toml/setter
  // change takes effect without touching the cache struct directly.
  rt.qfac_budget_bytes = (long)psp.exx_options().qfac_cache_mb << 20;
  std::array<long,3> mkey{mesh(0), mesh(1), mesh(2)};
  if (rt.qfac_mesh != mkey || rt.qfac_Gcut != Gcut ||
      rt.qfac_shape_restored != shape_restored) {
    rt.qfac.clear();
    rt.qfac_bytes = 0;
    rt.qfac_mesh = mkey;
    rt.qfac_Gcut = Gcut;
    rt.qfac_shape_restored = shape_restored;
  }
  auto quant = [](double x) { return std::llround(x * 1e8); };
  std::array<long,3> key{quant(k_p(0) - k_q(0)),
                         quant(k_p(1) - k_q(1)),
                         quant(k_p(2) - k_q(2))};
  if (auto it = rt.qfac.find(key); it != rt.qfac.end()) {
    ++rt.qfac_hits;
    return it->second;
  }
  long bytes = (long)scratch.size() * (long)sizeof(ComplexType);
  if (rt.qfac_bytes + bytes <= rt.qfac_budget_bytes) {
    ++rt.qfac_builds;
    auto it = rt.qfac
                  .emplace(key, nda::array<ComplexType,3>(
                                    scratch.extent(0), scratch.extent(1),
                                    scratch.extent(2)))
                  .first;
    build_paw_aug_pair_factor(psp, aatab, qtab_sp, mesh, recv, Omega,
                              k_p, k_q, Gcut, it->second);
    rt.qfac_bytes += bytes;
    return it->second;
  }
  ++rt.qfac_uncached;
  build_paw_aug_pair_factor(psp, aatab, qtab_sp, mesh, recv, Omega,
                            k_p, k_q, Gcut, scratch);
  return scratch;
}

/**
 * Pseudopot-aware exact-exchange matrix element K_{ij}(s, k_p) builder.
 *
 * For NCPP this is identical to `hamilt::v_x(...)` (smooth-only). For
 * USPP/PAW the inner loop augments each pair density with the per-pair
 * Σ_atm Σ_{IJ} P*_{n,atmI}(k_q) · P_{a,atmJ}(k_p) Q^{IJ}_atm(G+Δk) ·
 * e^{−i(G+Δk)·τ_atm} term before contraction.
 *
 * Output: K_{ij}(s, k_ibz, i, j) ACCUMULATED-with-zero (the routine zeros
 * Kij.local() before filling — matches the convention used by v_x.hpp and
 * by hamilt::Vhartree).
 */
inline void v_x(utils::mpi_context_t<boost::mpi3::communicator,
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
         math::nda::DistributedArrayOfRank<4> auto & Kij,
         bool shape_restored = false,
         bool onsite = true)
{
  if (psp.pp_type() == pp_ncpp_t) {
    ::hamilt::v_x(mpi, vG, npol, mesh, lattv, recv, k2g, kpts,
                  kp_to_ibz, kp_trev, kp_symm, symm_list, nii,
                  psi, Kij);
    return;
  }

  // ----------------- USPP/PAW path -----------------
  decltype(nda::range::all) all;
  using nda::range;

  utils::check(npol == 1,
      "v_x_paw: only npol=1 supported");
  utils::check(psi.grid()[3] == 1, "v_x_paw: psi grid[3] must be 1");

  long nspin   = psi.global_shape()[0];
  long nk_ibz  = psi.global_shape()[1];
  long nbnd    = psi.global_shape()[2];
  long ngm_wfc = k2g.extent(0);
  long nk      = kp_to_ibz.shape(0);
  long nnr     = mesh(0)*mesh(1)*mesh(2);

  utils::check(psi.global_shape()[3] == npol*ngm_wfc,
      "v_x_paw: psi g-dim mismatch");
  utils::check(Kij.global_shape()[0] == nspin &&
               Kij.global_shape()[1] == nk_ibz &&
               Kij.global_shape()[2] == nbnd &&
               Kij.global_shape()[3] == nbnd,
      "v_x_paw: Kij shape must be (nspin, nk_ibz, nbnd, nbnd)");

  double det_recv = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                  - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                  + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double Omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / det_recv;
  // See v_x.hpp comment: full occupation convention to match QE's vexx.
  const ComplexType scl(-1.0 / (double(nk) * Omega), 0.0);

  // ---- Build psi_r_full ----
  nda::array<ComplexType, 4> psi_r_full(nspin, nk, nbnd, nnr);
  psi_r_full() = ComplexType(0.0);
  {
    nda::array<ComplexType, 2> pr({1, nnr});
    auto pr4d = nda::reshape(pr,
                  std::array<long,4>{1, mesh(0), mesh(1), mesh(2)});
    math::nda::fft<true> F(pr4d);
    nda::array<long, 1> k2g_rotate(ngm_wfc);
    nda::stack_array<double, 3> Gs; Gs() = 0.0;
    nda::array<ComplexType,1> *Xft = nullptr;
    auto ploc = psi.local();
    for (auto [is_l, s] : itertools::enumerate(psi.local_range(0))) {
      for (auto [ik_l, k_ibz] : itertools::enumerate(psi.local_range(1))) {
        for (long kf = 0; kf < nk; ++kf) {
          if (kp_to_ibz(kf) != k_ibz) continue;
          k2g_rotate() = k2g();
          if (kp_trev(kf) or kp_symm(kf) > 0)
            utils::transform_k2g(kp_trev(kf), symm_list[kp_symm(kf)], Gs, mesh,
                                 kpts(k_ibz, all), k2g_rotate, Xft);
          for (auto [ia_l, a] : itertools::enumerate(psi.local_range(2))) {
            pr() = ComplexType(0.0);
            nda::copy_select(true, k2g_rotate,
                             ComplexType(1.0),
                             ploc(is_l, ik_l, ia_l, range(0, ngm_wfc)),
                             ComplexType(0.0),
                             pr(0, all));
            F.backward(pr4d);
            if (kp_trev(kf))
              for (long r = 0; r < nnr; ++r)
                psi_r_full(s, kf, a, r) = std::conj(pr(0, r));
            else
              for (long r = 0; r < nnr; ++r)
                psi_r_full(s, kf, a, r) = pr(0, r);
          }
        }
      }
    }
  }
  // Reduce as a flat double buffer: the boost::mpi3 all_reduce for a
  // std::complex<double> value type does not dispatch to native MPI_SUM and
  // falls onto a serialized path that is glacial for the ~50-200 MB dense-grid
  // psi_r_full array (fine for the small unit-test fixtures, catastrophic
  // here). Reinterpreting as 2*N doubles forces the native reduction — this is
  // the same trick used in methods/ERI/cholesky.icc.
  //
  // Chunked: 2*size() overflows the int count MPI_Allreduce takes once the
  // array gets large (Si 3x3x3, 250 bands, ecutwfc 150 Ry -> ~3.5e9 doubles),
  // which MPI rejects with MPI_ERR_COUNT rather than silently truncating.
  all_reduce_doubles_chunked(mpi.comm,
                             reinterpret_cast<double*>(psi_r_full.data()),
                             2 * psi_r_full.size());

  // ---- Build Pskna at full BZ ----
  // Cached View-2 symmetry lift (collective on psp's communicator
  // at the first build — v_x is dispatched from pseudopot with that context).
  auto const& ityp = psp.ityp_view();
  auto const& nh_v = psp.nh_view();
  auto const& ofs  = psp.ofs_view();
  auto const& sps  = psp.paw_species_view();
  long nat = ityp.extent(0);

  auto Pfull = psp.Pskna_full_bz().local();  // (nspin, nk_full, npol*nkb, nbnd)

  // ---- G-sphere cutoff for the augmentation (physical bound) ----
  // Gcut = largest sphere fully inside the dense FFT box = min over reciprocal
  // axes of M_i·|b_i| (distance to the nearest box face-center). Beyond Gcut the
  // grid holds only the anisotropic box corners (|G| up to ~2× Gcut), where the
  // smooth pair density is ~0 and the on-site AE−PS contribution is
  // Coulomb-negligible. We therefore (i) build the qrad Bessel table only up to
  // Gcut (was: abs-sum of the corners, ~2× larger → ~5-8× more K-points), and
  // (ii) skip |G+Δk| > Gcut in build_paw_aug_pair_factor. This mirrors ABINIT's
  // ecutsigx-bounded G-sphere and is what makes the direct exchange tractable.
  double Gcut = 1e30;
  for (int i = 0; i < 3; ++i) {
    long Mi = mesh(i) / 2;
    double brow = std::sqrt(recv(i,0)*recv(i,0) + recv(i,1)*recv(i,1)
                          + recv(i,2)*recv(i,2));
    Gcut = std::min(Gcut, (double)Mi * brow);
  }
  double Kmax_est = Gcut + 0.1;
  // Cached angular tables + per-species qrad tables. The qrad dq
  // is the project-wide 0.01 (was a local 0.05 here — finer, strictly more
  // accurate, and shared with every other qrad consumer). Option A
  // (shape_restored): PAW species use the full AE−PS pair-density form
  // factor (drops the separate deltaC one-center correction below); USPP
  // always uses the compensation charge.
  auto const& aatab = psp.paw_aatab();
  auto const& qtab_sp = psp.paw_qrad_tabs(Kmax_est, shape_restored);

  // ---- nij_max per species (for Qfac packing) ----
  long nij_max = 0;
  for (size_t nt = 0; nt < sps.size(); ++nt) {
    long nij_nt = (long)nh_v(nt) * ((long)nh_v(nt) + 1) / 2;
    nij_max = std::max(nij_max, nij_nt);
  }
  if (nij_max == 0) nij_max = 1;

  // ---- Inner loop ----
  auto nii_host = nda::to_host(nii);
  auto Kloc = Kij.local();
  Kloc() = ComplexType(0.0);

  nda::array<ComplexType, 2> pair_buf(nbnd, nnr);
  auto pair4d = nda::reshape(pair_buf,
                  std::array<long,4>{nbnd, mesh(0), mesh(1), mesh(2)});
  math::nda::fft<true> Fpair(pair4d);

  nda::array<ComplexType, 1> v_coul(nnr);
  nda::array<ComplexType, 3> Qfac(nat, nij_max, nnr);
  nda::array<ComplexType, 1> aug_acc(nnr);   // Σ over atm/IJ for one (n, a)

  for (auto [is_l, s] : itertools::enumerate(Kij.local_range(0))) {
    for (auto [ik_l, k_p_ibz] : itertools::enumerate(Kij.local_range(1))) {
      auto K_sk = Kloc(is_l, ik_l, all, all);

      for (long kq = 0; kq < nk; ++kq) {
        long kq_ibz = kp_to_ibz(kq);
        v_coul() = ComplexType(0.0);
        vG.evaluate_in_mesh(range(0, nnr), v_coul, mesh,
                            nda::stack_array<double,3,3>{}, recv,
                            kpts(k_p_ibz, all), kpts(kq, all));

        // Q^IJ_atm(G+Δk) × e^{-i(G+Δk)·τ_atm} for this (k_p, k_q), via the
        // Δk-keyed cache (hits across the spin loop and calls).
        auto const& Qf = get_or_build_qfac_pair_factor(
            psp, aatab, qtab_sp, mesh, recv, Omega,
            kpts(k_p_ibz, all), kpts(kq, all), Gcut, shape_restored, Qfac);

        for (long n = 0; n < nbnd; ++n) {
          double f = std::real(nii_host(s, kq_ibz, n));
          if (std::abs(f) < 1e-15) continue;

          // ---- Smooth pair density: pair(a, r) = u*_n(kq, r) × u_a(k_p, r) ----
          for (long a = 0; a < nbnd; ++a)
            for (long r = 0; r < nnr; ++r)
              pair_buf(a, r) = std::conj(psi_r_full(s, kq, n, r)) *
                               psi_r_full(s, k_p_ibz, a, r);
          Fpair.forward(pair4d);  // → pair_buf(a, g) in dense FFT G-space

          // ---- Add PAW augmentation:
          //   pair_aug(a, g) = Σ_atm Σ_{IJ} P*_{n, atmI}(kq) · P_{a, atmJ}(k_p)
          //                     × Qfac(atm, ij_packed(I,J), g)
          // Implemented as: for each (atm, I): accumulate aug_acc(g) =
          //   Σ_J P_{a, atmJ}(k_p) × Qfac(atm, ij_packed(I,J), g)
          // then pair_buf(a, g) += P*_{n, atmI}(kq) × aug_acc(g).
          auto const& ijtoh = psp.ijtoh_view();
          // Aug scales as Ω: pair_g_CoQui = M_cellnorm has G=0 = Σ|c|² (≤1 for
          // PAW with augmentation). The cell-FT density-coefficient form Q (with
          // 4π/Ω baked in, matching QE's qgm) needs to be multiplied by Ω to
          // contribute on the same scale as pair_g_smooth. Empirically this
          // brings the LiH-PAW Li-1s pair_g(G=0) from 0.4 (= Σ|c|²) up to ~1
          // (= ∫|ψ_AE|² = 1), as expected.
          const ComplexType omega_factor(Omega, 0.0);
          for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);
            int nh_a = nh_v(nt);
            if (nh_a == 0) continue;
            if (!sps[nt].is_paw && !sps[nt].is_uspp) continue;
            long p0 = ofs(ia);
            for (long a = 0; a < nbnd; ++a) {
              for (int I = 0; I < nh_a; ++I) {
                ComplexType Pn_cI = std::conj(Pfull(s, kq, p0 + I, n));
                // aug_acc(g) = Σ_J P_{a, atmJ}(k_p) × Qfac(ia, ij(I,J), g)
                aug_acc() = ComplexType(0.0);
                for (int J = 0; J < nh_a; ++J) {
                  long ij = static_cast<long>(ijtoh(nt, I, J)) - 1;
                  if (ij < 0) continue;
                  ComplexType PaJ = Pfull(s, k_p_ibz, p0 + J, a);
                  for (long g = 0; g < nnr; ++g)
                    aug_acc(g) += PaJ * Qf(ia, ij, g);
                }
                ComplexType pref = Pn_cI * omega_factor;
                for (long g = 0; g < nnr; ++g)
                  pair_buf(a, g) += pref * aug_acc(g);
              }
            }
          }

          // ---- Contract K_{ij} += scl × f × Σ_g v_C(g) × pair*(i, g) × pair(j, g) ----
          // The Kij band dimensions (2,3) may be distributed over the process
          // grid, so K_sk holds only the LOCAL band block. Iterate the local
          // band ranges for the K_sk indices (i_l, j_l) while using the GLOBAL
          // band index (i, j) into pair_buf, which holds all bands.
          for (auto [i_l, i] : itertools::enumerate(Kij.local_range(2))) {
            for (auto [j_l, j] : itertools::enumerate(Kij.local_range(3))) {
              ComplexType acc(0.0);
              for (long g = 0; g < nnr; ++g)
                acc += v_coul(g) *
                       std::conj(pair_buf(i, g)) *
                       pair_buf(j, g);
              K_sk(i_l, j_l) += scl * ComplexType(f, 0.0) * acc;
            }
          }
        }
      }
    }
  }

  {
    auto const& rt = psp.paw_rt();
    app_log(3, "v_x_paw qfac cache: hits={} builds={} uncached={} ({} MB)",
            rt.qfac_hits, rt.qfac_builds, rt.qfac_uncached,
            rt.qfac_bytes >> 20);
  }

  // Option A (shape_restored): the full AE−PS on-site pair density is already
  // carried by the smooth+aug contraction above (build_qrad_tab_full_aeps), so
  // the exact on-site exchange is complete and the deltaC one-center correction
  // must NOT be added (it would double count). See
  // notes/paw_onsite_exchange_analysis.{md,pdf} (Route A).
  if (shape_restored) { mpi.comm.barrier(); return; }
  // `onsite=false` isolates the one-center term (pseudopot::paw_onsite_diag).
  if (!onsite) { mpi.comm.barrier(); return; }

  // ----------------------------------------------------------------------
  // PAW one-center exchange correction (deltaC-mediated AE-PS Coulomb on
  // the projector-basis pair densities).
  //
  // ΔK^{oc}_{ij}(s, k_p) = scl_oc · Σ_{n, k_q} f^total_n
  //                         × Σ_a Σ_{IJKL} P*_{i,aI}(k_p) × P_{n,aJ}(k_q)
  //                          × deltaC^a(I,J,K,L)
  //                          × P*_{n,aK}(k_q) × P_{j,aL}(k_p)
  //
  // with deltaC^a(I,J,K,L) = ⟨φ_I φ_J | v_C | φ_K φ_L⟩^{AE-PS}_a
  // (Mulliken / chemist convention, same as Hartree's becsum × deltaC ×
  // becsum energy formula in test_paw_onecenter). The prefactor scl_oc is
  // derived below. USPP species and atoms without deltaC contribute nothing.
  // ----------------------------------------------------------------------
  // ----- One-center exchange prefactor (THC-consistent, derived).
  //
  // Blöchl PAW partition of the AE exchange Fock matrix element:
  //   (i n | n j)^AE = (ρ̃_in + n̂_in | ρ̃_nj + n̂_nj)_{FFT, compensated}
  //                  + Σ_a Σ_IJKL P*_{i,I} P_{n,J} ΔC^a(I,J,K,L) P*_{n,K} P_{j,L}
  // with ΔC^a = ⟨φ_Iφ_J|v|φ_Kφ_L⟩^AE − ⟨φ̃_Iφ̃_J+Q̂|v|φ̃_Kφ̃_L+Q̂⟩^PS.
  // The first term is the smooth+aug contraction above, which carries the Fock
  // prefactor  scl = −1/(N_k·Ω)  contracted with f = nii directly (NO explicit
  // spin factor; the Ω is the smooth G-space Coulomb measure). The one-center
  // is the SAME per-(kq,n) Fock contraction (same f = nii) on the deltaC
  // integral, which is already in energy units — so it inherits scl WITHOUT the
  // Ω measure:  scl_oc = −1/N_k.
  //
  // Validated against THC's own K_a exchange by test vx_onecenter_vs_thc_Ka,
  // which isolates THC's one-center Fock via THC_Vx(onsite=true) −
  // THC_Vx(onsite=false) and fits the scalar prefactor of the raw deltaC
  // contraction: α = −1/N_k to machine precision. The same value follows from
  // QE's own paw_exx.f90 kernel (Eq. 32/33, Paier 2005) and VASP's PAW-EXX; see
  // notes/paw_onecenter_exchange_prefactor.md. No explicit n_s ⇒ generalizes to
  // nspin=2 like the smooth scl. USPP (no ΔC) is unaffected.
  const ComplexType scl_oc(-1.0 / double(nk), 0.0);
  for (auto [is_l, s] : itertools::enumerate(Kij.local_range(0))) {
    for (auto [ik_l, k_p_ibz] : itertools::enumerate(Kij.local_range(1))) {
      auto K_sk = Kloc(is_l, ik_l, all, all);

      for (long kq = 0; kq < nk; ++kq) {
        long kq_ibz = kp_to_ibz(kq);
        for (long n = 0; n < nbnd; ++n) {
          double f = std::real(nii_host(s, kq_ibz, n));
          if (std::abs(f) < 1e-15) continue;

          for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);
            int nh_a = nh_v(nt);
            if (nh_a == 0) continue;
            auto const& sp = sps[nt];
            if (!sp.is_paw || sp.deltaC.size() == 0) continue;
            long p0 = ofs(ia);

            // U(I, L) = Σ_{J, K} deltaC(I, J, K, L) × P_{n, aJ}(k_q) × P*_{n, aK}(k_q)
            nda::array<ComplexType, 2> U(nh_a, nh_a);
            U() = ComplexType(0.0);
            for (int I = 0; I < nh_a; ++I)
              for (int L = 0; L < nh_a; ++L) {
                ComplexType acc(0.0);
                for (int J = 0; J < nh_a; ++J) {
                  ComplexType PnJ = Pfull(s, kq, p0 + J, n);
                  for (int K = 0; K < nh_a; ++K) {
                    ComplexType PnK = std::conj(Pfull(s, kq, p0 + K, n));
                    acc += ComplexType(sp.deltaC(I, J, K, L), 0.0) * PnJ * PnK;
                  }
                }
                U(I, L) = acc;
              }

            // Accumulate K_ij += scl_oc · f · Σ_{IL} P*_i,aI(k_p) × U(I,L) × P_j,aL(k_p)
            // Global band index (i, j) for Pfull; local index (i_l, j_l) for the
            // (possibly band-distributed) K_sk block.
            for (auto [i_l, i] : itertools::enumerate(Kij.local_range(2))) {
              for (auto [j_l, j] : itertools::enumerate(Kij.local_range(3))) {
                ComplexType acc(0.0);
                for (int I = 0; I < nh_a; ++I) {
                  ComplexType PiI = std::conj(Pfull(s, k_p_ibz, p0 + I, i));
                  for (int L = 0; L < nh_a; ++L) {
                    ComplexType PjL = Pfull(s, k_p_ibz, p0 + L, j);
                    acc += PiI * U(I, L) * PjL;
                  }
                }
                K_sk(i_l, j_l) += scl_oc * ComplexType(f, 0.0) * acc;
              }
            }
          }
        }
      }
    }
  }

  mpi.comm.barrier();
}

// ===========================================================================
// Full-density-matrix exact exchange (USPP/PAW). Generalizes the diagonal
// v_x above to a Hermitian density matrix nij(s,k,a,b) via its natural-orbital
// decomposition at each (s,kq): γ = Σ_p w_p |χ_p⟩⟨χ_p|, χ_p = Σ_a U_ap ψ_a.
// The inner ("occupied") index runs over natural orbitals — both the
// real-space orbital AND its projectors ⟨β|χ_p⟩ = Σ_a U_ap ⟨β|ψ_a⟩ are
// rotated — while the outer (k_p) orbitals and projectors stay canonical.
// Reuses the exact same smooth + Q-augmentation + deltaC one-center
// contraction as the diagonal kernel, so K[nij] inherits its conventions
// (including the one-center prefactor currently under review).
//
// Symmetry-reduced meshes: the full-BZ inner states are the exactly rotated
// IBZ states (psi_r_full G-space lift + cached Pskna_full_bz), so the band
// matrix at kq = S·k̃ is the UNCHANGED IBZ matrix nij(s,k̃), conjugated at
// time-reversal points — same View-2 rule as compute_becsum_full_symm;
// derivation in notes/static_route_nij_symmetry_note.md.
// ===========================================================================
inline void v_x(utils::mpi_context_t<boost::mpi3::communicator,
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
         math::nda::DistributedArrayOfRank<4> auto & Kij,
         bool shape_restored = false,
         bool onsite = true)
{
  if (psp.pp_type() == pp_ncpp_t) {
    ::hamilt::v_x(mpi, vG, npol, mesh, lattv, recv, k2g, kpts,
                  kp_to_ibz, kp_trev, kp_symm, symm_list, nij, psi, Kij);
    return;
  }

  decltype(nda::range::all) all;
  using nda::range;

  utils::check(npol == 1, "v_x_paw(nij): only npol=1 supported");
  utils::check(psi.grid()[3] == 1, "v_x_paw(nij): psi grid[3] must be 1");

  long nspin   = psi.global_shape()[0];
  long nk_ibz  = psi.global_shape()[1];
  long nbnd    = psi.global_shape()[2];
  long ngm_wfc = k2g.extent(0);
  long nk      = kp_to_ibz.shape(0);
  long nnr     = mesh(0)*mesh(1)*mesh(2);

  utils::check(nij.extent(0) == nspin && nij.extent(1) == nk_ibz &&
               nij.extent(2) == nbnd && nij.extent(3) == nbnd,
      "v_x_paw(nij): nij shape mismatch");

  double det_recv = recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
                  - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
                  + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double Omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / det_recv;
  const ComplexType scl(-1.0 / (double(nk) * Omega), 0.0);

  // ---- Canonical real-space orbitals (outer index source) ----
  nda::array<ComplexType, 4> psi_r_full(nspin, nk, nbnd, nnr);
  psi_r_full() = ComplexType(0.0);
  {
    nda::array<ComplexType, 2> pr({1, nnr});
    auto pr4d = nda::reshape(pr, std::array<long,4>{1, mesh(0), mesh(1), mesh(2)});
    math::nda::fft<true> F(pr4d);
    nda::array<long, 1> k2g_rotate(ngm_wfc);
    nda::stack_array<double, 3> Gs; Gs() = 0.0;
    nda::array<ComplexType,1> *Xft = nullptr;
    auto ploc = psi.local();
    for (auto [is_l, s] : itertools::enumerate(psi.local_range(0))) {
      for (auto [ik_l, k_ibz] : itertools::enumerate(psi.local_range(1))) {
        for (long kf = 0; kf < nk; ++kf) {
          if (kp_to_ibz(kf) != k_ibz) continue;
          k2g_rotate() = k2g();
          if (kp_trev(kf) or kp_symm(kf) > 0)
            utils::transform_k2g(kp_trev(kf), symm_list[kp_symm(kf)], Gs, mesh,
                                 kpts(k_ibz, all), k2g_rotate, Xft);
          for (auto [ia_l, a] : itertools::enumerate(psi.local_range(2))) {
            pr() = ComplexType(0.0);
            nda::copy_select(true, k2g_rotate, ComplexType(1.0),
                             ploc(is_l, ik_l, ia_l, range(0, ngm_wfc)),
                             ComplexType(0.0), pr(0, all));
            F.backward(pr4d);
            if (kp_trev(kf))
              for (long r = 0; r < nnr; ++r) psi_r_full(s, kf, a, r) = std::conj(pr(0, r));
            else
              for (long r = 0; r < nnr; ++r) psi_r_full(s, kf, a, r) = pr(0, r);
          }
        }
      }
    }
  }
  // Flat-double reduction to force native MPI_SUM (see the nii overload above),
  // chunked because 2*size() overflows MPI_Allreduce's int count at production
  // sizes. This is the nij (full density matrix) overload, which is the one the
  // direct static route reaches from GW.
  all_reduce_doubles_chunked(mpi.comm,
                             reinterpret_cast<double*>(psi_r_full.data()),
                             2 * psi_r_full.size());

  // ---- Canonical projectors at full BZ (outer index source) ----
  auto const& ityp = psp.ityp_view();
  auto const& nh_v = psp.nh_view();
  auto const& ofs  = psp.ofs_view();
  auto const& sps  = psp.paw_species_view();
  long nat = ityp.extent(0);

  // Cached View-2 lift (collective on psp's communicator at the
  // first build — see the diagonal kernel).
  auto Pfull = psp.Pskna_full_bz().local();  // (nspin, nk, npol*nkb, nbnd)
  long nkb = Pfull.extent(2);

  // ---- G-sphere cutoff + qrad tables (same bound as the diagonal kernel) ----
  // Gcut = largest sphere inside the dense FFT box (min_i M_i·|b_i|); the qrad
  // table is built only up to Gcut and build_paw_aug_pair_factor skips
  // |G+Δk| > Gcut. See the diagonal (nii) kernel for the rationale.
  double Gcut = 1e30;
  for (int i = 0; i < 3; ++i) {
    long Mi = mesh(i) / 2;
    double brow = std::sqrt(recv(i,0)*recv(i,0) + recv(i,1)*recv(i,1)
                          + recv(i,2)*recv(i,2));
    Gcut = std::min(Gcut, (double)Mi * brow);
  }
  double Kmax_est = Gcut + 0.1;
  // Cached angular + qrad tables at the project-wide dq = 0.01
  // (see the diagonal kernel).
  auto const& aatab = psp.paw_aatab();
  auto const& qtab_sp = psp.paw_qrad_tabs(Kmax_est, shape_restored);

  long nij_max = 0;
  for (size_t nt = 0; nt < sps.size(); ++nt)
    nij_max = std::max(nij_max, (long)nh_v(nt) * ((long)nh_v(nt) + 1) / 2);
  if (nij_max == 0) nij_max = 1;

  // ---- Natural orbitals (inner index): real-space φ and projectors Pφ ----
  // φ_all(s,kq,p,r) = Σ_a U_ap ψ_a(kq,r); Pnat(s,kq,idx,p) = Σ_a U_ap P(kq,idx,a).
  auto nij_host = nda::to_host(nij);
  nda::array<ComplexType, 4> phi_all(nspin, nk, nbnd, nnr);
  nda::array<ComplexType, 4> Pnat_all(nspin, nk, nkb, nbnd);
  nda::array<double, 3> w_all(nspin, nk, nbnd);
  phi_all() = ComplexType(0.0); Pnat_all() = ComplexType(0.0); w_all() = 0.0;
  for (long s = 0; s < nspin; ++s)
    for (long kq = 0; kq < nk; ++kq) {
      // Band matrix at full-BZ kq: IBZ matrix in the lifted-state basis,
      // conjugated at trev points (View-2, as in compute_becsum_full_symm).
      long kq_ibz = (long)kp_to_ibz(kq);
      bool trev_q = (bool)kp_trev(kq);
      nda::array<ComplexType,2> nblk(nbnd, nbnd);
      for (long a = 0; a < nbnd; ++a)
        for (long b = 0; b < nbnd; ++b)
          nblk(a, b) = trev_q ? std::conj(nij_host(s, kq_ibz, a, b))
                              : nij_host(s, kq_ibz, a, b);
      auto [w, U] = ::hamilt::detail::natural_occupations(nblk);
      for (long p = 0; p < nbnd; ++p) w_all(s, kq, p) = w(p);
      for (long p = 0; p < nbnd; ++p)
        for (long a = 0; a < nbnd; ++a) {
          ComplexType c = U(a, p);
          if (std::abs(c) < 1e-300) continue;
          for (long r = 0; r < nnr; ++r) phi_all(s, kq, p, r) += c * psi_r_full(s, kq, a, r);
          for (long idx = 0; idx < nkb; ++idx) Pnat_all(s, kq, idx, p) += c * Pfull(s, kq, idx, a);
        }
    }

  // ---- Inner loop (mirrors the diagonal kernel; inner uses natural orbitals) ----
  auto Kloc = Kij.local();
  Kloc() = ComplexType(0.0);
  auto const& ijtoh = psp.ijtoh_view();

  nda::array<ComplexType, 2> pair_buf(nbnd, nnr);
  auto pair4d = nda::reshape(pair_buf, std::array<long,4>{nbnd, mesh(0), mesh(1), mesh(2)});
  math::nda::fft<true> Fpair(pair4d);
  nda::array<ComplexType, 1> v_coul(nnr);
  nda::array<ComplexType, 3> Qfac(nat, nij_max, nnr);
  nda::array<ComplexType, 1> aug_acc(nnr);

  for (auto [is_l, s] : itertools::enumerate(Kij.local_range(0))) {
    for (auto [ik_l, k_p_ibz] : itertools::enumerate(Kij.local_range(1))) {
      auto K_sk = Kloc(is_l, ik_l, all, all);
      for (long kq = 0; kq < nk; ++kq) {
        v_coul() = ComplexType(0.0);
        vG.evaluate_in_mesh(range(0, nnr), v_coul, mesh,
                            nda::stack_array<double,3,3>{}, recv,
                            kpts(k_p_ibz, all), kpts(kq, all));
        auto const& Qf = get_or_build_qfac_pair_factor(
            psp, aatab, qtab_sp, mesh, recv, Omega,
            kpts(k_p_ibz, all), kpts(kq, all), Gcut, shape_restored, Qfac);

        for (long p = 0; p < nbnd; ++p) {
          double w = w_all(s, kq, p);
          if (std::abs(w) < 1e-15) continue;

          // Smooth pair density with the natural inner orbital χ_p.
          for (long a = 0; a < nbnd; ++a)
            for (long r = 0; r < nnr; ++r)
              pair_buf(a, r) = std::conj(phi_all(s, kq, p, r)) *
                               psi_r_full(s, k_p_ibz, a, r);
          Fpair.forward(pair4d);

          // PAW augmentation (inner projector rotated to the natural orbital).
          const ComplexType omega_factor(Omega, 0.0);
          for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);
            int nh_a = nh_v(nt);
            if (nh_a == 0) continue;
            if (!sps[nt].is_paw && !sps[nt].is_uspp) continue;
            long p0 = ofs(ia);
            for (long a = 0; a < nbnd; ++a) {
              for (int I = 0; I < nh_a; ++I) {
                ComplexType Pn_cI = std::conj(Pnat_all(s, kq, p0 + I, p));
                aug_acc() = ComplexType(0.0);
                for (int J = 0; J < nh_a; ++J) {
                  long ij = static_cast<long>(ijtoh(nt, I, J)) - 1;
                  if (ij < 0) continue;
                  ComplexType PaJ = Pfull(s, k_p_ibz, p0 + J, a);
                  for (long g = 0; g < nnr; ++g) aug_acc(g) += PaJ * Qf(ia, ij, g);
                }
                ComplexType pref = Pn_cI * omega_factor;
                for (long g = 0; g < nnr; ++g) pair_buf(a, g) += pref * aug_acc(g);
              }
            }
          }

          // Global band index (i, j) into pair_buf; local index into the
          // (possibly band-distributed) K_sk block.
          for (auto [i_l, i] : itertools::enumerate(Kij.local_range(2)))
            for (auto [j_l, j] : itertools::enumerate(Kij.local_range(3))) {
              ComplexType acc(0.0);
              for (long g = 0; g < nnr; ++g)
                acc += v_coul(g) * std::conj(pair_buf(i, g)) * pair_buf(j, g);
              K_sk(i_l, j_l) += scl * ComplexType(w, 0.0) * acc;
            }
        }
      }
    }
  }

  // Option A (shape_restored): full AE−PS pair density already in smooth+aug
  // above → skip the deltaC one-center correction (would double count).
  if (shape_restored) { mpi.comm.barrier(); return; }
  // `onsite=false` isolates the one-center term (pseudopot::paw_onsite_diag).
  if (!onsite) { mpi.comm.barrier(); return; }

  // ---- PAW one-center exchange (deltaC), inner projectors → natural orbitals.
  // Same THC-consistent prefactor as the diagonal kernel: scl_oc = −1/N_k,
  // inheriting the smooth+aug Fock scl = −1/(N_k·Ω) without the Ω G-space
  // measure (deltaC is already an energy integral). See the diagonal v_x above
  // and test vx_onecenter_vs_thc_Ka (machine-precision validation vs THC K_a).
  const ComplexType scl_oc(-1.0 / double(nk), 0.0);
  for (auto [is_l, s] : itertools::enumerate(Kij.local_range(0))) {
    for (auto [ik_l, k_p_ibz] : itertools::enumerate(Kij.local_range(1))) {
      auto K_sk = Kloc(is_l, ik_l, all, all);
      for (long kq = 0; kq < nk; ++kq) {
        for (long p = 0; p < nbnd; ++p) {
          double w = w_all(s, kq, p);
          if (std::abs(w) < 1e-15) continue;
          for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia);
            int nh_a = nh_v(nt);
            if (nh_a == 0) continue;
            auto const& sp = sps[nt];
            if (!sp.is_paw || sp.deltaC.size() == 0) continue;
            long p0 = ofs(ia);
            nda::array<ComplexType, 2> U2(nh_a, nh_a);
            U2() = ComplexType(0.0);
            for (int I = 0; I < nh_a; ++I)
              for (int L = 0; L < nh_a; ++L) {
                ComplexType acc(0.0);
                for (int J = 0; J < nh_a; ++J) {
                  ComplexType PnJ = Pnat_all(s, kq, p0 + J, p);
                  for (int K = 0; K < nh_a; ++K) {
                    ComplexType PnK = std::conj(Pnat_all(s, kq, p0 + K, p));
                    acc += ComplexType(sp.deltaC(I, J, K, L), 0.0) * PnJ * PnK;
                  }
                }
                U2(I, L) = acc;
              }
            for (auto [i_l, i] : itertools::enumerate(Kij.local_range(2)))
              for (auto [j_l, j] : itertools::enumerate(Kij.local_range(3))) {
                ComplexType acc(0.0);
                for (int I = 0; I < nh_a; ++I) {
                  ComplexType PiI = std::conj(Pfull(s, k_p_ibz, p0 + I, i));
                  for (int L = 0; L < nh_a; ++L) {
                    ComplexType PjL = Pfull(s, k_p_ibz, p0 + L, j);
                    acc += PiI * U2(I, L) * PjL;
                  }
                }
                K_sk(i_l, j_l) += scl_oc * ComplexType(w, 0.0) * acc;
              }
          }
        }
      }
    }
  }

  mpi.comm.barrier();
}

}  // namespace hamilt::paw

#endif  // HAMILTONIAN_PAW_V_X_PAW_HPP
