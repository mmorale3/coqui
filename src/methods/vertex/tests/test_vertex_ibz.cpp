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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

// ISDF-Vertex: IBZ k-point symmetry support (notes/vertex_ibz_symmetry.md).
//
// Validation ladder (memo section 7):
//   1. vertex_ibz_leakage_diag: the C-window D-matrix leakage diagnostic computed
//      independently from MF->symmetry_rotation for a family of windows, tied to
//      the measured eigenvalue degeneracy structure (a window boundary slicing a
//      degenerate set <=> nonzero leakage). MF-only, fast.
//   2. vertex_ibz_gold: THE GOLD CHECK -- the same physical LiH-222 state driven
//      through the nosym (qe_lih222) and sym (qe_lih222_sym) variants with the
//      vertex on: e_hf/e_corr must agree to (cross-variant class) + O(leakage);
//      both error sources measured and reported separately (theory-owner item 3a).
//      Includes the near-closed-window control (item 3c) and the secondary-basis
//      path on the sym mesh.
//   3. vertex_ibz_conservation_sym: the conservation identity S_SigmaG + S_PW = 0
//      evaluated with star-weighted IBZ pairings (memo section 3.6) on the sym
//      mesh; sign-flip control at O(1).
//   4. vertex_ibz_noop_sym: C = empty set reproduces plain sym-scGW bitwise.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/sparse/csr_blas.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"
#include "utilities/symmetry.hpp"
#include "methods/ERI/thc.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;
  decltype(nda::range::all) constexpr all_r = nda::range::all;

  namespace ibz_test_detail {

    // Independent reimplementation of the C-window leakage (memo (C-leak)):
    // for every qsymms position >= 1 and full-BZ k, the D-mass of the window
    // columns outside the window. Returns (max, mean).
    inline std::pair<double, double> window_leakage(mf::MF &mf, long c0, long c1) {
      const long nk = mf.nkpts();
      const long nbnd = mf.nbnd();
      const long nc = c1 - c0;
      auto qsymms = mf.qsymms();
      const long nsym = qsymms.extent(0);
      nda::array<cplx, 2> E(nbnd, nc), Dcols(nbnd, nc);
      E() = cplx(0.0);
      for (long j = 0; j < nc; ++j) E(c0 + j, j) = cplx(1.0);
      double mx = 0.0, sum = 0.0;
      long cnt = 0;
      using math::sparse::csrmm;
      for (long js = 1; js < nsym; ++js)
        for (long ik = 0; ik < nk; ++ik) {
          auto [cj, Dsp] = mf.symmetry_rotation(js, ik);
          (void)cj;
          csrmm<'N'>(cplx(1.0), *Dsp, E, cplx(0.0), Dcols);
          double m_in = 0.0, m_all = 0.0;
          for (long a = 0; a < nbnd; ++a)
            for (long j = 0; j < nc; ++j) {
              const double w = std::norm(Dcols(a, j));
              m_all += w;
              if (a >= c0 and a < c1) m_in += w;
            }
          if (m_all > 1e-24) {
            const double l = 1.0 - m_in / m_all;
            mx = std::max(mx, l);
            sum += l;
            ++cnt;
          }
        }
      return {mx, (cnt > 0) ? sum / double(cnt) : 0.0};
    }

    // does the window boundary [c0, c1) slice through a degenerate eigenvalue set
    // at any (spin, k)? (the same 1e-4 degeneracy resolution generate_dmatrix uses,
    // symmetry.hpp:1039)
    inline bool window_splits_degeneracy(mf::MF &mf, long c0, long c1) {
      auto eig = mf.eigval();
      const long ns = eig.shape(0), nk = eig.shape(1), nb = eig.shape(2);
      auto split_at = [&](long b) {  // boundary between b-1 and b
        if (b <= 0 or b >= nb) return false;
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            if (std::abs(eig(is, ik, b) - eig(is, ik, b - 1)) < 1e-4) return true;
        return false;
      };
      return split_at(c0) or split_at(c1);
    }

    // A fixture SPEC: either a registered default_MF name, or "qe:<outdir>:<prefix>" for an arbitrary
    // QE mean field on disk (h5). The second form lets the transport / census diagnostics run against a
    // PRODUCTION mean field (the Si 4^3 symmetric mesh: 6 operations + time reversal, 3-fold rotations AND
    // 28 time-reversal pairs -- the combination no unit-test fixture has; 2026-09-22 time-reversal hunt).
    inline mf::MF make_mf(auto &mpi_context, std::string const &spec) {
      if (spec.rfind("qe:", 0) == 0) {
        const auto c = spec.find(':', 3);
        utils::check(c != std::string::npos, "fixture spec \"{}\": expected qe:<outdir>:<prefix>.", spec);
        return mf::default_MF(mpi_context, mf::qe_source, spec.substr(3, c - 3), spec.substr(c + 1), mf::h5_input_type);
      }
      return mf::default_MF(mpi_context, spec);
    }

  } // ibz_test_detail

  // ====================================================================================
  TEST_CASE("vertex_ibz_leakage_diag", "[methods][vertex][ibz]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    REQUIRE(mf->nkpts() != mf->nkpts_ibz());   // this IS a symmetry-reduced mesh

    struct row { long c0, c1; double mx, mean; bool splits; };
    std::vector<row> table;
    for (auto [c0, c1] : std::vector<std::pair<long,long>>{
             {0, 1}, {0, 2}, {1, 3}, {1, 2}, {0, 4}, {1, 4}, {2, 3}}) {
      auto [mx, mean] = ibz_test_detail::window_leakage(*mf, c0, c1);
      bool sp = ibz_test_detail::window_splits_degeneracy(*mf, c0, c1);
      table.push_back({c0, c1, mx, mean, sp});
      app_log(1, "ibz leakage: window [{}, {}): max = {:.3e}, mean = {:.3e}, "
                 "splits degenerate set = {}", c0, c1, mx, mean, sp);
    }
    // the diagnostic must track the degeneracy structure (theory-owner item 3b):
    // a window that does NOT split any degenerate set must be (near-)closed; a
    // window that does must show correspondingly larger leakage.
    double leak_closed_max = 0.0, leak_split_min = 1e300;
    bool have_closed = false, have_split = false;
    for (auto const& r : table) {
      if (r.splits) { have_split = true; leak_split_min = std::min(leak_split_min, r.mx); }
      else          { have_closed = true; leak_closed_max = std::max(leak_closed_max, r.mx); }
    }
    app_log(1, "ibz leakage: max over non-splitting windows = {:.3e}; "
               "min over splitting windows = {:.3e}", leak_closed_max,
            have_split ? leak_split_min : -1.0);
    if (have_closed) REQUIRE(leak_closed_max < 1e-6);
    if (have_closed and have_split) REQUIRE(leak_split_min > 10.0 * leak_closed_max);

    // probe the trev-carrying mesh used by the gold check (223: 2x2x3)
    {
      auto mf3 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_sym"));
      long ntrev = mf3->nkpts_trev_pairs();
      long nqtrev = 0;
      for (long q = 0; q < mf3->nqpts(); ++q)
        if (mf3->qp_trev(q)) ++nqtrev;
      auto [l13, l13m] = ibz_test_detail::window_leakage(*mf3, 1, 3);
      (void)l13m;
      app_log(1, "ibz leakage: qe_lih223_sym: nk {} -> {} IBZ, nq {} -> {} IBZ, "
                 "trev k-pairs = {}, trev-mapped q = {}, leak[1,3) = {:.3e}",
              mf3->nkpts(), mf3->nkpts_ibz(), mf3->nqpts(), mf3->nqpts_ibz(),
              ntrev, nqtrev, l13);
    }
  }

  // ====================================================================================
  /**
   * 2026-09-22: the TRANSPORT identity behind the symmetry path, tested numerically on the exact orbitals.
   * The kernels' effective columns are Xhat(js, k) = X(krot(js, k)) . Dc(js, k) (vertex_t::build_sym_ctx), meant to be the
   * orbital at k evaluated at the rotated points: psi_{k,b}(S^{-1} r_P) (or S r_P). Both sides are computable exactly: the
   * collocation of every full-mesh k at an arbitrary point list (thc::collocation_at_points, which rotates the IBZ orbitals
   * in real space) and at the point list rotated by S (utils::transform_r). The eight candidates {D, D^T, D^*, D^dag} x
   * {S, S^-1} are compared on the closed window [0, 6) of qe_lih222_sym (a degenerate triplet inside: the conventions
   * differ ONLY inside degenerate blocks, so the historic [1, 3) gates could not see it). COQUI_IBZ_TEST_WINDOW overrides.
   */
  // transform_r's arithmetic without the periodic wrap: returns the wrapped grid index (as transform_r) and the lattice
  // translation R_lat (integers) it removed, so that psi_k(S r_P) = e^{2 pi i k . R_lat} psi_k(r_wrapped) (Bloch phase)
  inline void rotate_points_with_wrap(utils::symm_op const &S, nda::array<long, 1> const &mesh, nda::array<long, 1> const &rin,
                                      nda::array<long, 1> &rout, nda::array<long, 2> &Rlat) {
    const long NX = mesh(0), NY = mesh(1), NZ = mesh(2), NX2 = NX / 2, NY2 = NY / 2, NZ2 = NZ / 2;
    rout = nda::array<long, 1>(rin.size()); Rlat = nda::array<long, 2>(rin.size(), 3);
    for (long i = 0; i < rin.size(); ++i) {
      // positions as the phase factors take them (rspace_phase_factor / load_basis_subset_fft_grid: n/N with n in [0, N)),
      // NOT transform_r's symmetric range: the two differ by lattice translations, invisible in the wrapped index but a
      // Bloch phase e^{2 pi i k.R} on the orbital (a sign at half-integer k)
      long n = rin(i);
      long n2 = n % NZ;
      long n_ = n / NZ;
      long n1 = n_ % NY;
      long n0 = n_ / NY;
      (void)NX2; (void)NY2; (void)NZ2;
      const double N10 = double(NY) / NX, N01 = double(NX) / NY, N12 = double(NY) / NZ, N21 = double(NZ) / NY, N02 = double(NX) / NZ, N20 = double(NZ) / NX;
      long ni = long(std::round(S.R(0, 0) * n0 + S.R(0, 1) * N10 * n1 + S.R(0, 2) * N20 * n2));
      long nj = long(std::round(S.R(1, 0) * N01 * n0 + S.R(1, 1) * n1 + S.R(1, 2) * N21 * n2));
      long nk_ = long(std::round(S.R(2, 0) * N02 * n0 + S.R(2, 1) * N12 * n1 + S.R(2, 2) * n2));
      long wi = 0, wj = 0, wk = 0;
      while (ni < 0) { ni += NX; --wi; } while (nj < 0) { nj += NY; --wj; } while (nk_ < 0) { nk_ += NZ; --wk; }
      while (ni >= NX) { ni -= NX; ++wi; } while (nj >= NY) { nj -= NY; ++wj; } while (nk_ >= NZ) { nk_ -= NZ; ++wk; }
      rout(i) = (ni * NY + nj) * NZ + nk_;
      Rlat(i, 0) = wi; Rlat(i, 1) = wj; Rlat(i, 2) = wk;   // r_rotated = r_wrapped + R_lat (in units of the lattice vectors)
    }
  }

  // ====================================================================================
  // TREV CENSUS (2026-09-22): which fixtures exercise the time-reversal branches at all?
  // The Si 4^3 symmetric P-side path loses 1.5 % of the vertex correction and the noinv
  // experiment pinned it on time reversal (notes/vertex_perf_plan.md, 2026-09-22 ~04:50).
  // Two distinct branches carry trev: the trev k IMAGES (conjugated collocation columns,
  // thc::collocation_at_points / chol_metric_impl_ibz) and the trev TRANSFERS (qp_trev:
  // the PQ-transposed read of the IBZ-stored rung, memo (P2)). This case counts both per
  // fixture, so a branch no gate ever runs is visible.  MF-only, seconds.
  // ====================================================================================
  TEST_CASE("vertex_ibz_trev_census", "[methods][vertex][ibz]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    std::vector<std::string> fxs = {"qe_lih222_sym", "qe_lih223_sym", "qe_lih223_inv", "qe_si222_sym"};
    if (const char *f = std::getenv("COQUI_IBZ_TEST_FIXTURE")) fxs = {std::string(f)};
    for (auto const &fx : fxs) {
      auto mf = std::make_shared<mf::MF>(ibz_test_detail::make_mf(mpi_context, fx));
      const long nk = mf->nkpts(), nkibz = mf->nkpts_ibz(), ntrev = mf->nkpts_trev_pairs();
      const long nq = mf->nqpts(), nqibz = mf->nqpts_ibz();
      auto kp_trev = mf->kp_trev();
      auto qp_trev = mf->qp_trev();
      auto qp_symm = mf->qp_symm();
      auto qsymms = mf->qsymms();
      long nk_trev = 0, n_tail_mismatch = 0;
      for (long k = 0; k < nk; ++k) {
        if (kp_trev(k)) ++nk_trev;
        if (bool(kp_trev(k)) != (k >= nk - ntrev)) ++n_tail_mismatch;   // the "trev images are the last ntrev k" assumption
      }
      // transfers: trev with the identity symmetry (pure time reversal) vs trev composed with a rotation
      long nq_trev = 0, nq_trev_id = 0, nq_trev_rot = 0, nq_rot_only = 0;
      for (long q = 0; q < nq; ++q) {
        if (not qp_trev(q)) { if (q >= nqibz) ++nq_rot_only; continue; }
        ++nq_trev;
        long js = -1;
        for (long i = 0; i < qsymms.extent(0); ++i) if (qsymms(i) == qp_symm(q)) js = i;
        if (js == 0) ++nq_trev_id; else ++nq_trev_rot;
      }
      // 2026-09-22: is -k of a time-reversal image a PLAIN negation in the stored crystal coordinates, or does it
      // fold back with a reciprocal-lattice vector? (an even mesh carries zone-boundary components +-1/2, an odd one
      // does not -- the difference between the broken Si 4^3 and the clean Si 3^3 / LiH fixtures)
      auto kc = mf->kpts_crystal();
      auto trev_pair = mf->kp_trev_pair();
      long n_umk_k = 0, n_half = 0;
      std::string ex;
      for (long k = 0; k < nk; ++k) {
        bool half = false;
        for (int i = 0; i < 3; ++i) if (std::abs(std::abs(kc(k, i)) - 0.5) < 1e-6) half = true;
        if (half) ++n_half;
        if (not kp_trev(k)) continue;
        const long p = long(trev_pair(k));
        double d = 0.0;
        for (int i = 0; i < 3; ++i) d += std::abs(kc(k, i) + kc(p, i));
        if (d > 1e-6) {
          ++n_umk_k;
          if (ex.empty()) { char b[160]; std::snprintf(b, sizeof(b), " e.g. k %ld (%.3f,%.3f,%.3f) + pair %ld (%.3f,%.3f,%.3f)",
                                                       k, kc(k,0), kc(k,1), kc(k,2), p, kc(p,0), kc(p,1), kc(p,2)); ex = b; }
        }
      }
      // the same question for the TRANSFERS the symmetric path reads PQ-transposed: q + (-q) = 0 or a lattice vector?
      auto qmin = mf->qminus();
      nda::array<double, 2> qcr(nq, 3);
      for (long iq = 0; iq < nq; ++iq) {
        const long k2 = mf->qk_to_k2(int(iq), 0);
        for (int i = 0; i < 3; ++i) qcr(iq, i) = kc(0, i) - kc(k2, i);
      }
      long n_umk_q = 0, n_umk_q_trev = 0;
      for (long iq = 0; iq < nq; ++iq) {
        double d = 0.0;
        for (int i = 0; i < 3; ++i) d += std::abs(qcr(iq, i) + qcr(long(qmin(iq)), i));
        if (d > 1e-6) { ++n_umk_q; if (qp_trev(iq)) ++n_umk_q_trev; }
      }
      app_log(1, "trev census [{}]: k {} (IBZ {}, trev pairs {}, kp_trev true {}, tail-order mismatches {}); "
                 "q {} (IBZ {}, {} ops): trev transfers {} (identity {} + composed with a rotation {}), rotation-only images {}",
              fx, nk, nkibz, ntrev, nk_trev, n_tail_mismatch, nq, nqibz, qsymms.extent(0), nq_trev, nq_trev_id, nq_trev_rot, nq_rot_only);
      app_log(1, "trev census [{}]: zone-boundary structure -- {} of {} k have a +-1/2 component; time-reversal pairs whose "
                 "partner is NOT the plain negation: {}{}; transfers whose -q folds back with a reciprocal-lattice vector: "
                 "{} of {} ({} of them marked trev)",
              fx, n_half, nk, n_umk_k, ex, n_umk_q, nq, n_umk_q_trev);
    }
    mpi_context->comm.barrier();
  }

  // ====================================================================================
  // CROSS-MESH TREV-IMAGE CHECK (2026-09-22). The transport test cannot see an error in the
  // time-reversal IMAGE columns themselves: on one mesh both of its sides come from
  // thc::collocation_at_points, which builds a trev image as conj(u_{k_ibz}(S^-1 r)) e^{i k r}
  // -- self-consistent by construction. This case compares the SAME physical k between TWO
  // meshes of the same crystal, one where k is a time-reversal image (reconstructed) and one
  // where it is not (stored, or reached by a rotation -- both paths already verified exact).
  // The comparison is on the GAUGE-INVARIANT window Gram
  //     M(P, P') = sum_{a in C} X(k, a, P) conj(X(k, a, P')),
  // invariant under any unitary mixing inside a window closed under degeneracies, so the
  // arbitrary per-band phases of two independent QE runs drop out.
  //   COQUI_IBZ_TEST_FIXTURE  = the mesh under test (default qe_si444_trevonly)
  //   COQUI_IBZ_TEST_FIXTURE2 = the reference mesh (default qe_si444_noinv)
  //   COQUI_IBZ_TEST_WINDOW   = the C window (default 0,8)
  // ====================================================================================
  TEST_CASE("vertex_ibz_trev_image", "[methods][vertex][ibz]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    const std::string fxA = std::getenv("COQUI_IBZ_TEST_FIXTURE") ? std::getenv("COQUI_IBZ_TEST_FIXTURE") : "qe_si444_trevonly";
    const std::string fxB = std::getenv("COQUI_IBZ_TEST_FIXTURE2") ? std::getenv("COQUI_IBZ_TEST_FIXTURE2") : "qe_si444_noinv";
    long w0 = 0, w1 = 8;
    if (const char *w = std::getenv("COQUI_IBZ_TEST_WINDOW")) std::sscanf(w, "%ld,%ld", &w0, &w1);
    nda::range W(w0, w1);
    decltype(nda::range::all) all;
    auto mfA = std::make_shared<mf::MF>(ibz_test_detail::make_mf(mpi_context, fxA));
    auto mfB = std::make_shared<mf::MF>(ibz_test_detail::make_mf(mpi_context, fxB));
    ptree pt; pt.put("thresh", 1e-4); pt.put("chol_block_size", 1);
    methods::thc bA(mfA.get(), *mpi_context, pt, false);
    methods::thc bB(mfB.get(), *mpi_context, pt, false);
    auto mA = bA.rho_mesh(), mB = bB.rho_mesh();
    utils::check(mA(0) == mB(0) and mA(1) == mB(1) and mA(2) == mB(2),
                 "vertex_ibz_trev_image: the two meshes have different density grids ({} {} {} vs {} {} {}).",
                 mA(0), mA(1), mA(2), mB(0), mB(1), mB(2));
    const long nnr = mA(0) * mA(1) * mA(2);
    std::vector<long> pv;
    for (long n = 3; n < nnr and long(pv.size()) < 40; n += 97) pv.push_back(n);
    nda::array<long, 1> ipts(long(pv.size()));
    for (long i = 0; i < ipts.size(); ++i) ipts(i) = pv[size_t(i)];
    const long Nm = ipts.size(), nW = W.size();
    const long nkA = mfA->nkpts(), nkB = mfB->nkpts();
    auto XA = bA.collocation_at_points(ipts, nda::range(0, nkA), W);
    auto XB = bB.collocation_at_points(ipts, nda::range(0, nkB), W);
    auto kcA = mfA->kpts_crystal();
    auto kcB = mfB->kpts_crystal();
    auto trevA = mfA->kp_trev();
    nda::array<ComplexType, 2> MA(Nm, Nm), MB(Nm, Nm);
    double worst_trev = 0.0, worst_plain = 0.0;
    long k_worst = -1, n_trev = 0, n_plain = 0, n_miss = 0;
    for (long k = 0; k < nkA; ++k) {
      long kb = -1;
      for (long j = 0; j < nkB and kb < 0; ++j) {
        double d = 0.0;
        for (int i = 0; i < 3; ++i) { double x = kcA(k, i) - kcB(j, i); x -= std::round(x); d += std::abs(x); }
        if (d < 1e-6) kb = j;
      }
      if (kb < 0) { ++n_miss; continue; }
      double num = 0.0, den = 0.0;
      for (long P = 0; P < Nm; ++P)
        for (long Q = 0; Q < Nm; ++Q) {
          ComplexType a(0.0), b(0.0);
          for (long j = 0; j < nW; ++j) { a += XA(0, k, j, P) * std::conj(XA(0, k, j, Q));
                                          b += XB(0, kb, j, P) * std::conj(XB(0, kb, j, Q)); }
          num += std::norm(a - b); den += std::norm(b);
        }
      const double r = std::sqrt(num / std::max(den, 1e-300));
      if (trevA(k)) { ++n_trev; if (r > worst_trev) { worst_trev = r; k_worst = k; } }
      else          { ++n_plain; worst_plain = std::max(worst_plain, r); }
    }
    app_log(1, "ibz trev image [{} vs {}]: window [{}, {}), {} points -- the gauge-invariant window Gram agrees to "
               "{:.3e} on the {} TIME-REVERSAL IMAGES of {} (worst k {}) and to {:.3e} on its {} other k-points "
               "({} k of {} had no partner in the reference mesh)",
            fxA, fxB, w0, w1, Nm, worst_trev, n_trev, fxA, k_worst, worst_plain, n_plain, n_miss, nkA);
    mpi_context->comm.barrier();
  }

  TEST_CASE("vertex_ibz_transport", "[methods][vertex][ibz]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // COQUI_IBZ_TEST_FIXTURE overrides the symmetric fixture (qe_lih223_sym: time-reversal pairs, non-TRIM k, 4-fold operations)
    const std::string fxs = std::getenv("COQUI_IBZ_TEST_FIXTURE") ? std::getenv("COQUI_IBZ_TEST_FIXTURE") : "qe_lih222_sym";
    auto mf = std::make_shared<mf::MF>(ibz_test_detail::make_mf(mpi_context, fxs));
    REQUIRE(mf->nkpts() != mf->nkpts_ibz());
    decltype(nda::range::all) all;
    long w0 = 0, w1 = 6;
    if (const char *w = std::getenv("COQUI_IBZ_TEST_WINDOW")) std::sscanf(w, "%ld,%ld", &w0, &w1);
    nda::range W(w0, w1);
    const long nW = W.size(), nk = mf->nkpts(), nbnd = mf->nbnd();
    ptree pt; pt.put("thresh", 1e-4); pt.put("chol_block_size", 1);
    methods::thc builder(mf.get(), *mpi_context, pt, false);
    auto mesh = builder.rho_mesh();
    const long nnr = mesh(0) * mesh(1) * mesh(2);
    // a spread of grid points (every 97th): the identity must hold at EVERY point, so any set is a test
    std::vector<long> pv;
    for (long n = 3; n < nnr and long(pv.size()) < 60; n += 97) pv.push_back(n);
    nda::array<long, 1> ipts(long(pv.size()));
    for (long i = 0; i < ipts.size(); ++i) ipts(i) = pv[size_t(i)];
    const long Nm = ipts.size();
    auto X = builder.collocation_at_points(ipts, nda::range(0, nk), W);   // (ns, nk, nW, Nm): psi_{k,b}(r_P)
    auto symms = mf->symm_list();
    auto qsymms = mf->qsymms();
    auto kp_trev = mf->kp_trev();
    const long nsym = qsymms.extent(0);
    app_log(1, "ibz transport: {} ops, {} k (IBZ {}), window [{}, {}), {} points, {} trev pairs", nsym, nk, mf->nkpts_ibz(), w0, w1, Nm, mf->nkpts_trev_pairs());
    nda::array<ComplexType, 2> E(nbnd, nW), Dfull(nbnd, nW), Dw(nW, nW), Xr(Nm, nW), cand(Nm, nW);
    E() = ComplexType(0.0);
    for (long j = 0; j < nW; ++j) E(w0 + j, j) = ComplexType(1.0);
    const char *names[4] = {"D", "D^T", "D^*", "D^dag"};
    double best[8]; for (double &b : best) b = 0.0;
    for (long js = 1; js < nsym; ++js) {
      auto op = symms[size_t(qsymms(js))];
      // the point lists r' = S r_P and r'' = S^-1 r_P (transform_r: r_out = S.R * r_in)
      nda::array<long, 1> ipS, ipSi;
      nda::array<long, 2> RS, RSi;
      utils::symm_op opi = op; opi.R = op.Rinv; opi.Rinv = op.R;
      rotate_points_with_wrap(op, mesh, ipts, ipS, RS);
      rotate_points_with_wrap(opi, mesh, ipts, ipSi, RSi);
      {
        nda::array<long, 1> chk(ipts);
        utils::transform_r(op, nda::array<long, 1>::zeros({3}), mesh, chk);
        long ndiff = 0, nwrap = 0;
        for (long i = 0; i < Nm; ++i) { if (chk(i) != ipS(i)) ++ndiff; if (RS(i, 0) != 0 or RS(i, 1) != 0 or RS(i, 2) != 0) ++nwrap; }
        // the order of the operation (R^n = 1)
        nda::stack_array<double, 3, 3> Rn = op.R, Rp;
        int order = 1;
        auto is_id = [](nda::stack_array<double, 3, 3> const &M) { double d = 0.0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) d += std::abs(M(i, j) - (i == j ? 1.0 : 0.0)); return d < 1e-8; };
        while (not is_id(Rn) and order < 12) { nda::blas::gemm(op.R, Rn, Rp); Rn = Rp; ++order; }
        app_log(1, "ibz transport: op js {} (symm {}, order {}): R = [{:+.0f} {:+.0f} {:+.0f}; {:+.0f} {:+.0f} {:+.0f}; {:+.0f} {:+.0f} {:+.0f}]; rotated indices vs transform_r differ at {} points; {} of {} points wrapped",
                js, qsymms(js), order, op.R(0,0), op.R(0,1), op.R(0,2), op.R(1,0), op.R(1,1), op.R(1,2), op.R(2,0), op.R(2,1), op.R(2,2), ndiff, nwrap, Nm);
      }
      auto XS0 = builder.collocation_at_points(ipS, nda::range(0, nk), W);    // psi_{k,b}(wrapped S r_P)
      auto XSi0 = builder.collocation_at_points(ipSi, nda::range(0, nk), W);  // psi_{k,b}(wrapped S^-1 r_P)
      // the Bloch phase of the removed lattice translation: psi_k(r + R) = e^{2 pi i k.R} psi_k(r)  (k in crystal units)
      auto kc = mf->kpts_crystal();
      auto XS = XS0, XSi = XSi0;
      for (long k = 0; k < nk; ++k)
        for (long P = 0; P < Nm; ++P) {
          const double phS = 2.0 * M_PI * (kc(k, 0) * RS(P, 0) + kc(k, 1) * RS(P, 1) + kc(k, 2) * RS(P, 2));
          const double phSi = 2.0 * M_PI * (kc(k, 0) * RSi(P, 0) + kc(k, 1) * RSi(P, 1) + kc(k, 2) * RSi(P, 2));
          for (long b = 0; b < nW; ++b) { XS(0, k, b, P) *= std::polar(1.0, phS); XSi(0, k, b, P) *= std::polar(1.0, phSi); }
        }
      auto trev_pair = mf->kp_trev_pair();
      for (long k = 0; k < nk; ++k) {
        auto [cj, Dsp] = mf->symmetry_rotation(js, k);
        // time-reversal images (cj): the kernels' column is conj(X(krot(js, pair(k))) . D) (build_sym_ctx) -- tested against the
        // same transported orbital psi_k(S r_P); the plain points otherwise
        math::sparse::csrmm<'N'>(ComplexType(1.0), *Dsp, E, ComplexType(0.0), Dfull);   // D E: (nbnd, nW), rows = all bands
        for (long a = 0; a < nW; ++a) for (long b = 0; b < nW; ++b) Dw(a, b) = Dfull(w0 + a, b);
        const long kr = mf->ks_to_k(int(js), int(cj ? long(trev_pair(k)) : k));
        for (long P = 0; P < Nm; ++P) for (long a = 0; a < nW; ++a) Xr(P, a) = X(0, kr, a, P);   // X(krot(js, k)) or X(krot(js, pair(k)))
        for (int c = 0; c < 4; ++c) {
          nda::array<ComplexType, 2> Dc(nW, nW);
          for (long a = 0; a < nW; ++a)
            for (long b = 0; b < nW; ++b)
              Dc(a, b) = (c == 0) ? Dw(a, b) : (c == 1) ? Dw(b, a) : (c == 2) ? std::conj(Dw(a, b)) : std::conj(Dw(b, a));
          nda::blas::gemm(Xr, Dc, cand);                                     // sum_a X(kr)(P, a) Dc(a, b)
          if (cj) for (long P = 0; P < Nm; ++P) for (long b = 0; b < nW; ++b) cand(P, b) = std::conj(cand(P, b));   // the trev-image rule
          for (int t = 0; t < 2; ++t) {
            double num = 0.0, den = 0.0;
            for (long P = 0; P < Nm; ++P)
              for (long b = 0; b < nW; ++b) {
                const ComplexType ref = t == 0 ? XS(0, k, b, P) : XSi(0, k, b, P);
                num += std::norm(cand(P, b) - ref); den += std::norm(ref);
              }
            const double r = std::sqrt(num / std::max(den, 1e-300));
            best[c * 2 + t] = std::max(best[c * 2 + t], r);
            if (c == 0 and t == 1) {
              // the same comparison up to ONE overall phase per (js, k): 1 - |<cand, ref>| / (|cand| |ref|)
              ComplexType ov(0.0); double nc_ = 0.0, nr_ = 0.0;
              for (long P = 0; P < Nm; ++P)
                for (long b = 0; b < nW; ++b) {
                  const ComplexType ref = t == 0 ? XS(0, k, b, P) : XSi(0, k, b, P);
                  ov += std::conj(cand(P, b)) * ref; nc_ += std::norm(cand(P, b)); nr_ += std::norm(ref);
                }
              const double ph = 1.0 - std::abs(ov) / std::sqrt(std::max(nc_ * nr_, 1e-300));
              app_log(1, "ibz transport:   js {} k {}{} (krot {}, kp_to_ibz {}, kp_symm {}): X(krot) . D vs psi_k({} r_P) = {:.3e}; up to a phase {:.3e} (phase {:+.3f} pi)",
                      js, k, cj ? " [trev image: conj rule]" : "", kr, mf->kp_to_ibz(int(k)), mf->kp_symm(int(k)), t == 0 ? "S" : "S^-1", r, ph, std::arg(ov) / M_PI);
            }
          }
        }
      }
    }
    // setup sanity: band-resolved MODULI (phase- and D-free): |psi_{krot,b}(r_P)| vs |psi_{k,b}(S^{+-1} r_P)| summed over the
    // window, and the plain |X(k)(S r_P)| vs |X(k)(r_P)| control (must be O(1) different), for the first op and first k
    {
      const long js = 1, k = 0, kr = mf->ks_to_k(int(js), int(k));
      auto op = symms[size_t(qsymms(js))];
      nda::array<long, 1> ipS(ipts), ipSi(ipts);
      utils::transform_r(op, nda::array<long, 1>::zeros({3}), mesh, ipS);
      utils::symm_op opi = op; opi.R = op.Rinv; opi.Rinv = op.R;
      utils::transform_r(opi, nda::array<long, 1>::zeros({3}), mesh, ipSi);
      auto XS = builder.collocation_at_points(ipS, nda::range(0, nk), W);
      auto XSi = builder.collocation_at_points(ipSi, nda::range(0, nk), W);
      double m1 = 0.0, m2 = 0.0, m3 = 0.0, den = 0.0;
      for (long P = 0; P < Nm; ++P) {
        double a = 0.0, bS = 0.0, bSi = 0.0, b0 = 0.0;
        for (long b = 0; b < nW; ++b) { a += std::norm(X(0, kr, b, P)); bS += std::norm(XS(0, k, b, P)); bSi += std::norm(XSi(0, k, b, P)); b0 += std::norm(X(0, k, b, P)); }
        m1 += (a - bS) * (a - bS); m2 += (a - bSi) * (a - bSi); m3 += (a - b0) * (a - b0); den += a * a;
      }
      app_log(1, "ibz transport: setup check (js 1, k 0 -> krot {}): window density |X(krot)(r_P)|^2 vs |X(k)(S r_P)|^2: {:.3e}, vs |X(k)(S^-1 r_P)|^2: {:.3e}, vs the unrotated |X(k)(r_P)|^2: {:.3e}; ipts[0..3] = {} {} {} -> S: {} {} {}",
              kr, std::sqrt(m1 / den), std::sqrt(m2 / den), std::sqrt(m3 / den), ipts(0), ipts(1), ipts(2), ipS(0), ipS(1), ipS(2));
      // the EMPIRICAL transport matrix: least squares X(k)(S^-1 r_P) ~ X(krot)(r_P) . M  (M = pinv(Xr) . XSi), and the stored D
      nda::array<ComplexType, 2> A(Nm, nW), Bm(Nm, nW), G(nW, nW), Rhs(nW, nW), M(nW, nW);
      for (long P = 0; P < Nm; ++P) for (long b = 0; b < nW; ++b) { A(P, b) = X(0, kr, b, P); Bm(P, b) = XSi(0, k, b, P); }
      nda::blas::gemm(nda::dagger(A), A, G); nda::blas::gemm(nda::dagger(A), Bm, Rhs);
      nda::matrix<ComplexType> Gm(nW, nW); Gm() = G; nda::inverse_in_place(Gm);
      nda::blas::gemm(Gm, Rhs, M);
      auto [cj0, Dsp0] = mf->symmetry_rotation(js, k);
      math::sparse::csrmm<'N'>(ComplexType(1.0), *Dsp0, E, ComplexType(0.0), Dfull);
      std::string sm, sd;
      for (long a = 0; a < nW; ++a) {
        for (long b = 0; b < nW; ++b) { char buf[40]; std::snprintf(buf, sizeof(buf), "(%6.3f,%6.3f) ", M(a, b).real(), M(a, b).imag()); sm += buf;
                                        std::snprintf(buf, sizeof(buf), "(%6.3f,%6.3f) ", Dfull(w0 + a, b).real(), Dfull(w0 + a, b).imag()); sd += buf; }
        sm += "\n      "; sd += "\n      ";
      }
      double resM = 0.0, denM = 0.0;
      nda::blas::gemm(A, M, cand);
      for (long P = 0; P < Nm; ++P) for (long b = 0; b < nW; ++b) { resM += std::norm(cand(P, b) - Bm(P, b)); denM += std::norm(Bm(P, b)); }
      app_log(1, "ibz transport: empirical M (X(k)(S^-1 r) = X(krot)(r) M, fit residual {:.3e}):\n      {}\n   stored D(js 1, k 0) window block:\n      {}", std::sqrt(resM / denM), sm, sd);
    }
    // point-resolved ratio for (js 1, k 1), band 0 (non-degenerate?): is the mismatch a POINT-dependent phase e^{i G0 . r_P}?
    {
      const long js = 1, k = 1, kr = mf->ks_to_k(int(js), int(k));
      auto op = symms[size_t(qsymms(js))];
      nda::array<long, 1> ipSi(ipts);
      utils::symm_op opi = op; opi.R = op.Rinv; opi.Rinv = op.R;
      utils::transform_r(opi, nda::array<long, 1>::zeros({3}), mesh, ipSi);
      auto XSi = builder.collocation_at_points(ipSi, nda::range(0, nk), W);
      auto [cj, Dsp] = mf->symmetry_rotation(js, k);
      math::sparse::csrmm<'N'>(ComplexType(1.0), *Dsp, E, ComplexType(0.0), Dfull);
      for (long a = 0; a < nW; ++a) for (long b = 0; b < nW; ++b) Dw(a, b) = Dfull(w0 + a, b);
      for (long P = 0; P < Nm; ++P) for (long a = 0; a < nW; ++a) Xr(P, a) = X(0, kr, a, P);
      nda::blas::gemm(Xr, Dw, cand);
      auto kc = mf->kpts_crystal();
      std::string lines;
      for (long P = 0; P < std::min<long>(Nm, 10); ++P) {
        const long n = ipts(P), n2 = n % mesh(2), n1 = (n / mesh(2)) % mesh(1), n0 = n / (mesh(2) * mesh(1));
        const ComplexType ratio = XSi(0, k, 0, P) / cand(P, 0);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "\n      P %2ld r = (%2ld,%2ld,%2ld)/(%ld,%ld,%ld): |ratio| %.4f, phase %+.4f pi", P, n0, n1, n2, mesh(0), mesh(1), mesh(2), std::abs(ratio), std::arg(ratio) / M_PI);
        lines += buf;
      }
      app_log(1, "ibz transport: (js 1, k 1 = ({:.2f},{:.2f},{:.2f}) -> krot {} = ({:.2f},{:.2f},{:.2f})) band 0: psi_k(S^-1 r_P) / [X(krot) D](P): {}",
              kc(k, 0), kc(k, 1), kc(k, 2), kr, kc(kr, 0), kc(kr, 1), kc(kr, 2), lines);
    }
    // the DIRECTION of the transfer map vs the k map: the kernels rotate the legs with js = q_isym(q') and read W at q_star(q');
    // they need q_star(q') = "js applied to q'" in the SAME sense as krot(js, k) = "js applied to k" (see the kernel identity in
    // notes/vertex_perf_plan.md 2026-09-22). Check on the exact vectors: is q_ibz = k-map(js)(q') or q' = k-map(js)(q_ibz)?
    {
      auto Qcart = mf->Qpts();   // Cartesian: to crystal through the lattice vectors (as scr_coulomb's dump does)
      auto lat = mf->lattv();
      nda::array<double, 2> qc(Qcart.shape(0), 3);
      for (long iq = 0; iq < Qcart.shape(0); ++iq)
        for (int i = 0; i < 3; ++i) { double v = 0.0; for (int j = 0; j < 3; ++j) v += lat(i, j) * Qcart(iq, j); qc(iq, i) = v / (2.0 * M_PI); }
      auto kcr = mf->kpts_crystal();
      auto qp_symm = mf->qp_symm();
      auto qp_to_ibz = mf->qp_to_ibz();
      auto qp_trev = mf->qp_trev();
      const long nq = mf->nqpts();
      long n_fwd = 0, n_bwd = 0, n_tot = 0;
      for (long q = mf->nqpts_ibz(); q < nq; ++q) {
        if (qp_trev(q)) continue;
        long js = -1;
        for (long i = 0; i < nsym; ++i) if (qsymms(i) == qp_symm(q)) js = i;
        if (js <= 0) continue;
        const long qs = qp_to_ibz(q);
        // the k-map of js on the exact crystal vectors: k -> k . R_js (row) or R_js k (column)? take it from ks_to_k on a k that
        // shares the vector with q (Gamma-centered meshes: the q list and the k list are the same grid)
        long kq = -1, kqs = -1;
        for (long k = 0; k < nk; ++k) {
          double dq = 0.0, dqs = 0.0;
          for (int i = 0; i < 3; ++i) { double x = kcr(k, i) - qc(q, i); x -= std::round(x); dq += std::abs(x); double y = kcr(k, i) - qc(qs, i); y -= std::round(y); dqs += std::abs(y); }
          if (dq < 1e-6) kq = k;
          if (dqs < 1e-6) kqs = k;
        }
        if (kq < 0 or kqs < 0) continue;
        ++n_tot;
        if (mf->ks_to_k(int(js), int(kq)) == kqs) ++n_fwd;    // q_ibz = kmap(js)(q')   (what the kernel identity needs)
        if (mf->ks_to_k(int(js), int(kqs)) == kq) ++n_bwd;    // q'   = kmap(js)(q_ibz) (the opposite direction)
      }
      app_log(1, "ibz transport: transfer-map direction on {} non-IBZ, non-trev transfers: q_ibz = kmap(js)(q') for {}, q' = kmap(js)(q_ibz) for {} "
                 "(involutions satisfy both; a 3-fold group distinguishes them)", n_tot, n_fwd, n_bwd);
      // the same question through the MF's own q maps: qs_to_q(is, q_ibz) = "index of q_ibz * S_is" (bz_symmetry.hpp) -- is the
      // star member q' = q_ibz * S_js (js = q_isym(q')), i.e. the SAME action as krot (k -> k * S_js)?
      long n_same = 0, n_q = 0;
      for (long q = mf->nqpts_ibz(); q < nq; ++q) {
        if (qp_trev(q)) continue;
        long js = -1;
        for (long i = 0; i < nsym; ++i) if (qsymms(i) == qp_symm(q)) js = i;
        if (js <= 0) continue;
        ++n_q;
        if (mf->qs_to_q(int(js))(qp_to_ibz(q)) == q) ++n_same;
      }
      app_log(1, "ibz transport: through qs_to_q: q' = q_ibz * S_js for {} of {} non-IBZ transfers (the SAME action as krot: k' = k * S_js). "
                 "The kernels rotate the legs of a q' rung by S_js (k -> k * S_js), which maps the rotated transfer to q' * S_js = q_ibz * S_js^2: "
                 "only an involution lands on q_ibz.", n_same, n_q);
    }
    for (int c = 0; c < 4; ++c)
      app_log(1, "ibz transport: X(krot) . {:<6} vs psi_k(S r_P): {:.3e}   vs psi_k(S^-1 r_P): {:.3e}", names[c], best[c * 2], best[c * 2 + 1]);
    double mn = 1.0;
    for (double b : best) mn = std::min(mn, b);
    app_log(1, "ibz transport: the best candidate reproduces the transported orbital to {:.3e} (the kernels use X(krot) . D against S^-1 r_P)", mn);
    REQUIRE(mn < 1e-8);
    mpi_context->comm.barrier();
  }

  TEST_CASE("vertex_ibz_gold", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_gold skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_gold";

    // one (variant, run) driver: n_iter scGW, optional vertex window, returns
    // (e_hf, e_corr, sym_leakage_max). Side channels (most-recent run): last_grot = max
    // G_CC G-rotation residual (REPRO block); last_Nm / last_cond = secondary basis size
    // and achieved max_q cond(s) (COND-CAP block). cond_max<=0 keeps the legacy behavior.
    double last_grot = 0.0;
    long last_Nm = 0;
    double last_cond = 0.0;
    auto run = [&](std::string const& mf_name, nda::range window, long n_iter,
                   std::string const& isdf_mode, double cond_max = -1.0) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, window.size() > 0 ? "2nd_exchange" : "none", window,
                            mf->nbnd(), "ignore_g0", isdf_mode, 32, 1e-8, -1.0, cond_max);
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     n_iter, false, 1e-12, true);
      mpi_context->comm.barrier();
      double leak = vtx.sym_leakage_max();
      last_grot = vtx.g_rotation_max();
      last_Nm = vtx.secondary_Nm();
      last_cond = vtx.secondary_cond_s_max();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, leak);
    };

    // ---- cross-variant baseline: plain scGW (vertex off), 2 iterations ---------------
    auto [ehf_p_ns, ec_p_ns, l0a] = run("qe_lih222", nda::range(0, 0), 2, "global");
    auto [ehf_p_s, ec_p_s, l0b] = run("qe_lih222_sym", nda::range(0, 0), 2, "global");
    (void)l0a; (void)l0b;
    const double d_plain_hf = std::abs(ehf_p_ns - ehf_p_s);
    const double d_plain_ec = std::abs(ec_p_ns - ec_p_s);
    app_log(1, "ibz gold: PLAIN scGW cross-variant baseline: e_hf {:.12f} vs {:.12f} "
               "(|D| = {:.3e}); e_corr {:.12f} vs {:.12f} (|D| = {:.3e})",
            ehf_p_ns, ehf_p_s, d_plain_hf, ec_p_ns, ec_p_s, d_plain_ec);

    // ---- GOLD: vertex on, production window C = [1, 3), both cuts, 2 iterations ------
    // COQUI_IBZ_TEST_WINDOW = "a,b" overrides the window (2026-09-21: the degenerate-block convention test, e.g. "0,6" or "3,6"
    // -- [1, 3) holds no degenerate band pair, so the D matrices are diagonal phases there and a transposition inside a
    // degenerate block would pass unseen; Si 4^3 C = [0, 8) showed a 1.5 % sym-vs-nosym vertex gap, LiH [0, 6) a 1.7e-2 fold gap)
    nda::range gold_w(1, 3);
    if (const char *w = std::getenv("COQUI_IBZ_TEST_WINDOW")) { long a = 1, b = 3; std::sscanf(w, "%ld,%ld", &a, &b); gold_w = nda::range(a, b); }
    app_log(1, "ibz gold: vertex window C = [{}, {})", gold_w.first(), gold_w.last());
    auto [ehf_v_ns, ec_v_ns, lv_ns] = run("qe_lih222", gold_w, 2, "global");
    auto [ehf_v_s, ec_v_s, lv_s] = run("qe_lih222_sym", gold_w, 2, "global");
    (void)lv_ns;
    const double d_vert_hf = std::abs(ehf_v_ns - ehf_v_s);
    const double d_vert_ec = std::abs(ec_v_ns - ec_v_s);
    // the vertex effect (the physical signal the symmetry path must reproduce)
    const double shift_ns = std::abs(ec_v_ns - ec_p_ns);
    const double shift_s = std::abs(ec_v_s - ec_p_s);
    app_log(1, "ibz gold: VERTEX C=[1,3): e_hf {:.12f} vs {:.12f} (|D| = {:.3e}); "
               "e_corr {:.12f} vs {:.12f} (|D| = {:.3e})",
            ehf_v_ns, ehf_v_s, d_vert_hf, ec_v_ns, ec_v_s, d_vert_ec);
    app_log(1, "ibz gold: vertex e_corr shift: nosym {:.3e}, sym {:.3e}; "
               "measured C-window leakage (sym) = {:.3e}", shift_ns, shift_s, lv_s);
    app_log(1, "ibz gold: attribution: |D e_corr(vertex)| = {:.3e} vs baseline "
               "|D e_corr(plain)| = {:.3e} + O(leakage)*shift = {:.3e}",
            d_vert_ec, d_plain_ec, lv_s * shift_s);
    for (double e : {ehf_v_ns, ec_v_ns, ehf_v_s, ec_v_s}) REQUIRE(std::isfinite(e));
    REQUIRE(shift_ns > 1e-6);            // the vertex actually did something
    REQUIRE(shift_s > 1e-6);
    // the two variants must agree on the vertex physics: the sym-vs-nosym deviation
    // is bounded by (cross-variant baseline) + (leakage scale on the vertex shift)
    // + kernel headroom. A conjugation/rotation bug shows up at O(shift) instead.
    //
    // TIGHTENED 2026-07-25. The old margin was 0.25 * shift -- it accepted a sym-vs-nosym
    // disagreement of a QUARTER of the whole vertex effect, i.e. ~350x the value actually
    // observed, so it could not have caught a moderate symmetry defect. Measured on this
    // case: d_vert_ec = 3.93e-6 against d_plain_ec = 3.41e-6 and shift = 7.05e-4, i.e. the
    // symmetry path costs ~0.07% of the vertex signal. 0.05 * shift keeps ~10x headroom
    // over that while being 5x tighter. (For the record of why this matters: the symmetry
    // path was the leading suspect for the Si divergence for most of a session, and the
    // reason it could not be dismissed quickly is that no gate here was sharp enough to
    // say how accurate it actually is -- notes/vertex_divergence_diagnosis.md section 4.2.)
    const double gold_margin = std::max(0.05 * shift_ns, 5.0 * lv_s * shift_s) + 1e-8;
    REQUIRE(d_vert_ec <= d_plain_ec + gold_margin);
    REQUIRE(d_vert_hf <= d_plain_hf + gold_margin);
    // the shifts themselves must agree to the same class
    REQUIRE(std::abs(shift_ns - shift_s) <= d_plain_ec + gold_margin);

    // NOTE (measured, vertex_ibz_leakage_diag): C = [1,3) is EXACTLY symmetry-closed
    // on qe_lih222_sym (leak = 0) -- the gold comparison above is therefore the
    // clean-separation case of theory-owner item 3c: pure kernel/cross-variant
    // class, no leakage contribution.

    // ---- LEAKY-WINDOW control (theory-owner item 3b): C = [1, 4) splits a
    // degenerate conduction set (measured leak ~0.33). The deviation may grow to
    // O(leak * shift) but must remain finite and controlled.
    {
      auto [ehf_l_ns, ec_l_ns, ll_ns] = run("qe_lih222", nda::range(1, 4), 1, "global");
      auto [ehf_l_s, ec_l_s, ll_s] = run("qe_lih222_sym", nda::range(1, 4), 1, "global");
      (void)ll_ns; (void)ehf_l_ns; (void)ehf_l_s;
      const double shift_l = 0.5 * (std::abs(ec_l_ns - ec_p_ns) + std::abs(ec_l_s - ec_p_s));
      const double d_l = std::abs(ec_l_ns - ec_l_s);
      app_log(1, "ibz gold: LEAKY-WINDOW control C=[1,4): measured leakage = {:.3e}; "
                 "e_corr {:.12f} vs {:.12f}: |D| = {:.3e} vs shift = {:.3e} "
                 "(closed-window |D| = {:.3e})", ll_s, ec_l_ns, ec_l_s, d_l, shift_l, d_vert_ec);
      REQUIRE(std::isfinite(ec_l_ns));
      REQUIRE(std::isfinite(ec_l_s));
      REQUIRE(ll_s > 0.1);                     // the diagnostic sees the deep cut
      // controlled: bounded by the leakage scale on the vertex signal (+ baseline)
      REQUIRE(d_l <= d_plain_ec + 3.0 * ll_s * shift_l + 0.25 * shift_l + 1e-6);
    }

    // ---- TIME-REVERSAL gold (qe_lih223: 4 trev k-pairs, 4 trev-mapped q; the
    // trev-leg conj and the PQ-transpose transfer branches are exercised here;
    // C = [1,3) is measured closed on this mesh too) -------------------------------
    {
      auto [ehf3_p_ns, ec3_p_ns, l3a] = run("qe_lih223", nda::range(0, 0), 1, "global");
      auto [ehf3_p_s, ec3_p_s, l3b] = run("qe_lih223_sym", nda::range(0, 0), 1, "global");
      (void)l3a; (void)l3b;
      auto [ehf3_v_ns, ec3_v_ns, l3c] = run("qe_lih223", nda::range(1, 3), 1, "global");
      auto [ehf3_v_s, ec3_v_s, l3d] = run("qe_lih223_sym", nda::range(1, 3), 1, "global");
      (void)l3c;
      const double d3_plain = std::abs(ec3_p_ns - ec3_p_s);
      const double d3_vert = std::abs(ec3_v_ns - ec3_v_s);
      const double shift3 = 0.5 * (std::abs(ec3_v_ns - ec3_p_ns) + std::abs(ec3_v_s - ec3_p_s));
      app_log(1, "ibz gold (223/trev): plain |D e_corr| = {:.3e}; vertex e_corr "
                 "{:.12f} vs {:.12f}: |D| = {:.3e}, shift = {:.3e}, leak = {:.3e}",
              d3_plain, ec3_v_ns, ec3_v_s, d3_vert, shift3, l3d);
      for (double e : {ehf3_v_ns, ec3_v_ns, ehf3_v_s, ec3_v_s}) REQUIRE(std::isfinite(e));
      REQUIRE(shift3 > 1e-7);
      // TIGHTENED with the same reasoning as the gold block above: measured
      // d3_vert = 2.21e-5 against d3_plain = 2.62e-5 and shift3 = 1.34e-3.
      const double t3_margin = std::max(0.05 * shift3, 5.0 * l3d * shift3) + 1e-8;
      REQUIRE(d3_vert <= d3_plain + t3_margin);
      REQUIRE(std::abs(ehf3_v_ns - ehf3_v_s) <=
              std::abs(ehf3_p_ns - ehf3_p_s) + t3_margin);
    }

    // ---- secondary basis on the sym mesh (Refinement 2 under symmetry) ---------------
    {
      auto [ehf_sec, ec_sec, lsec] = run("qe_lih222_sym", nda::range(1, 3), 1, "secondary");
      auto [ehf_glo, ec_glo, lglo] = run("qe_lih222_sym", nda::range(1, 3), 1, "global");
      (void)lsec; (void)lglo;
      app_log(1, "ibz gold: sym-mesh secondary vs global (1 iteration): e_corr "
                 "{:.12f} vs {:.12f} (|D| = {:.3e}); e_hf |D| = {:.3e}",
              ec_sec, ec_glo, std::abs(ec_sec - ec_glo), std::abs(ehf_sec - ehf_glo));
      REQUIRE(std::isfinite(ec_sec));
      // at the numerical full pair rank the secondary path tracks global to the
      // downfold class (refinement2 memo 10.2: machine-level on the nosym mesh;
      // allow the svd_tol/kernel class here)
      REQUIRE(std::abs(ec_sec - ec_glo) <= 1e-5 + 0.05 * std::abs(ec_glo - ec_p_s));
    }

    // ---- SECONDARY vs GLOBAL past iteration 1 (secondary + sym mesh, TWO iterations).
    // C = [1,3) is symmetry-CLOSED on qe_lih222_sym (D-leak = 0, gold block above) and
    // LiH-222 secondary tracks global to ~1e-5 at 1 iteration, so both the window-leakage
    // and basis-crudeness confounds are removed: this isolates the secondary path itself.
    //
    // HISTORICAL NOTE (2026-07-25) -- this block was written to chase the Si production
    // signature "iter-2 G-rotation residual 5e-9 -> 0.49", on the hypothesis that it was a
    // secondary-path symmetry-unfolding defect. That hypothesis is REFUTED, and the
    // residual it keys on is NOT a defect indicator at all:
    //   * the residual has a plain-scGW baseline of the same order (LiH-222 1.6e-3 with the
    //     vertex OFF; Si M8 3.3e-4 on the converged no-vertex G) -- it is a D-matrix
    //     accuracy floor, and 0.49 is a CONSEQUENCE of the blow-up, not its cause;
    //   * the divergence reproduces identically on a symmetry-FREE mesh, and on a
    //     symmetry-reduced twin of the same mesh the two agree to 3.5e-6.
    // See notes/vertex_divergence_diagnosis.md section 4.2. The block is kept because
    // secondary-vs-global agreement past one iteration is worth pinning on its own -- but
    // do not read a large residual here as evidence of a symmetry bug.
    {
      auto [ehf_sec2, ec_sec2, lsec2] = run("qe_lih222_sym", nda::range(1, 3), 2, "secondary");
      const double grot_sec2 = last_grot;
      auto [ehf_glo2, ec_glo2, lglo2] = run("qe_lih222_sym", nda::range(1, 3), 2, "global");
      const double grot_glo2 = last_grot;
      (void)ehf_sec2; (void)ehf_glo2; (void)lsec2; (void)lglo2;
      app_log(1, "ibz REPRO (2-iter, C=[1,3) closed): e_corr secondary {:.12f} vs global "
                 "{:.12f} (|D| = {:.3e}); G-rotation residual secondary = {:.3e}, global = "
                 "{:.3e}; window D-leak secondary = {:.3e}, global = {:.3e}",
              ec_sec2, ec_glo2, std::abs(ec_sec2 - ec_glo2), grot_sec2, grot_glo2, lsec2, lglo2);
      REQUIRE(std::isfinite(ec_sec2));
      REQUIRE(std::isfinite(ec_glo2));
      // EXPECT (if the secondary sym multi-iter path is correct): the secondary
      // G-rotation residual stays the same small class as global's on this closed window,
      // and the self-consistent e_corr tracks global to the downfold class it held at 1 iter.
      REQUIRE(grot_sec2 <= std::max(1e-6, 10.0 * grot_glo2));
      REQUIRE(std::abs(ec_sec2 - ec_glo2) <= 1e-4 + 0.05 * std::abs(ec_glo2 - ec_p_s));
    }

    // ---- CONDITIONING CAP (vertex_isdf_cond_max): bound the per-q downfold conditioning
    // by regularizing the least-squares solve (rcond = 1/sqrt(cond_max)). last_cond is the
    // REGULARIZED conditioning (max_q, smallest RETAINED singular value). The blowup is
    // q-specific with a SHARED point set, so the cap bites in the SOLVE and leaves N_m
    // unchanged (pruning shared points cannot touch the worst q). We cap BELOW the uncapped
    // regularized conditioning to force the truncation to engage; the real payoff is Si
    // production (raw cond(s) ~ 5e7). The disabled path (cond_max <= 0) uses svd_tol only
    // and is covered bit-identically by every other secondary run in the suite. ---------
    {
      auto [ehf0, ec0, l0] = run("qe_lih222_sym", nda::range(1, 3), 1, "secondary");
      const long Nm0 = last_Nm; const double cond0 = last_cond;
      const double cap = std::max(1e3, cond0 / 1e3);   // cap well below the uncapped cond
      auto [ehf1, ec1, l1] = run("qe_lih222_sym", nda::range(1, 3), 1, "secondary", cap);
      const long Nm1 = last_Nm; const double cond1 = last_cond;
      (void)ehf0; (void)ehf1; (void)l0; (void)l1;
      app_log(1, "ibz COND-CAP: uncapped cond(s)_eff = {:.3e} (N_m = {}); cap = {:.3e} -> "
                 "cond(s)_eff = {:.3e} (N_m = {}); e_corr {:.10f} -> {:.10f}",
              cond0, Nm0, cap, cond1, Nm1, ec0, ec1);
      REQUIRE(std::isfinite(ec1));
      REQUIRE(Nm1 == Nm0);                       // the per-q rcond cap does NOT change N_m
      if (cond0 > 2.0 * cap) {                   // cap genuinely below the uncapped cond
        REQUIRE(cond1 <= 2.0 * cap);             // ... so the regularized cond is bounded
        REQUIRE(cond1 < cond0);                  // ... and strictly improved
      }
    }
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  TEST_CASE("vertex_ibz_conservation_sym", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_conservation_sym skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_cons";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // physical state: one plain scGW iteration + RPA-W rebuild (consistent (G, W))
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    REQUIRE(std::isfinite(e_hf_0));
    REQUIRE(std::isfinite(e_corr_0));
    if (not mb_state.dW_qtPQ.has_value()) scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    mpi_context->comm.barrier();

    auto MFp = thc.MF();
    const long nkpts = MFp->nkpts();
    const long nkpts_ibz = MFp->nkpts_ibz();
    const long nqpts_ibz = MFp->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MFp->nbnd();
    nda::range C(1, 3);

    auto G_loc = mb_state.sG_tskij.value().local();
    const long nt = G_loc.shape(0);
    const long ns = G_loc.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    REQUIRE(G_loc.shape(2) == nkpts_ibz);

    // star multiplicities (memo section 3.6)
    nda::array<double, 1> m_k(nkpts_ibz), m_q(nqpts_ibz);
    m_k() = 0.0; m_q() = 0.0;
    for (long k = 0; k < nkpts; ++k) m_k(MFp->kp_to_ibz(k)) += 1.0;
    for (long q = 0; q < MFp->nqpts(); ++q) m_q(MFp->qp_to_ibz(q)) += 1.0;

    // ---- Sigma^C alone: zero the accumulator, eval, read back ------------------------
    solvers::vertex_t vtx(&ft, "2nd_exchange", C, nbnd, "ignore_g0", "global");
    REQUIRE(vtx.active());
    mb_state.sSigma_tskij.value().set_zero();
    mpi_context->comm.barrier();
    vtx.eval_Sigma_C(mb_state, thc);
    nda::array<cplx, 5> Sig(nt, ns, nkpts_ibz, nbnd, nbnd);
    Sig = mb_state.sSigma_tskij.value().local();
    app_log(1, "ibz cons: sym leakage max = {:.3e}", vtx.sym_leakage_max());

    // ---- Pi^C on the IBZ grid (code tau storage) -------------------------------------
    const std::array<long, 4> pgrid = {1, 1, 1, mpi_context->comm.size()};
    const std::array<long, 4> bsize = {1, 1, 1, 1};
    const std::array<long, 4> gshape = {nt_half, nqpts_ibz, Np, Np};
    auto dPi = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);
    nda::array<cplx, 4> Pi_code(nt_half, nqpts_ibz, Np, Np);
    Pi_code() = cplx(0.0);
    Pi_code(dPi.local_range(0), dPi.local_range(1), dPi.local_range(2), dPi.local_range(3)) =
        dPi.local();
    mpi_context->comm.all_reduce_in_place_n(Pi_code.data(), Pi_code.size(), std::plus<>{});

    // ---- Wdyn(tau) replicated + Z(q) -------------------------------------------------
    nda::array<cplx, 4> Wt(nqpts_ibz, nt_half, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      Wt() = cplx(0.0);
      Wt(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(Wt.data(), Wt.size(), std::plus<>{});
    }
    nda::array<cplx, 3> Zq(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq) Zq(iq, all_r, all_r) = thc.Z(int(iq));

    // ---- pairing machinery (conservation_validation.md section 1.6) -------------------
    auto Twt_bb = ft.Twt_bb();
    long m0 = -1;
    {
      auto wnb = ft.wn_mesh_b();
      for (long m = 0; m < ft.nw_b(); ++m)
        if (wnb(m) == 0) { m0 = m; break; }
    }
    REQUIRE(m0 >= 0);
    // tau = 0 interpolation row (x = -1)
    nda::array<double, 1> x0(1);
    x0(0) = -1.0;
    auto Row0_arr = ft.construct_tau_interpolate_matrix(x0);   // (1, nt)
    const double spin = (ns == 1) ? 2.0 : 1.0;

    // S_SigmaG = -(spin/Nk) sum_{s,k in IBZ} m_k Twt_bb(m0,:) . f_k,
    //   f_k(tau) = sum_{ab in C} Sig_ab(k, tau) G_ab(k, beta - tau)
    cplx S_SG(0.0);
    {
      nda::array<cplx, 1> f(nt);
      for (long is = 0; is < ns; ++is)
        for (long k = 0; k < nkpts_ibz; ++k) {
          f() = cplx(0.0);
          for (long it = 0; it < nt; ++it) {
            const long itm = nt - it - 1;
            cplx acc(0.0);
            for (long a = C.first(); a < C.last(); ++a)
              for (long b = C.first(); b < C.last(); ++b)
                acc += Sig(it, is, k, a, b) * G_loc(itm, is, k, a, b);
            f(it) = acc;
          }
          cplx row(0.0);
          for (long it = 0; it < nt; ++it) row += Twt_bb(m0, it) * f(it);
          S_SG += m_k(k) * row;
        }
      S_SG *= cplx(-spin / double(nkpts));
    }

    // S_PW = +(1/Nk) sum_{q in IBZ} m_q [ sum_MN Pi(q, tau=0) Z_NM
    //                                     + Twt_bb(m0,:) . g_q ],
    //   g_q(tau) = sum_MN Pi_notes(q,tau) Wdyn_NM(q, beta-tau); both PH-symmetric,
    //   Pi_notes on the full grid from the code storage via the PH mirror
    //   (pi design section 2 rule 3: code(it) = notes(beta - tau_it), notes PH-sym).
    cplx S_PW(0.0);
    {
      nda::array<cplx, 2> Pi_full_t(nt, Np * Np);   // notes-tau, one q at a time
      for (long q = 0; q < nqpts_ibz; ++q) {
        for (long it = 0; it < nt; ++it) {
          const long ih = std::min(it, nt - it - 1);   // PH-symmetric storage
          auto src = Pi_code(ih, q, all_r, all_r);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) Pi_full_t(it, M * Np + N) = src(M, N);
        }
        // tau = 0 value from the DLR interpolation row
        cplx SZ(0.0);
        {
          nda::array<cplx, 1> Pi0(Np * Np);
          Pi0() = cplx(0.0);
          for (long it = 0; it < nt; ++it)
            for (long MN = 0; MN < Np * Np; ++MN)
              Pi0(MN) += cplx(Row0_arr(0, it)) * Pi_full_t(it, MN);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) SZ += Pi0(M * Np + N) * Zq(q, N, M);
        }
        // dynamic part
        cplx SW(0.0);
        for (long it = 0; it < nt; ++it) {
          const long ihw = std::min(it, nt - it - 1);   // Wdyn(beta-tau) = Wdyn(tau)
          cplx g(0.0);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N)
              g += Pi_full_t(it, M * Np + N) * Wt(q, ihw, N, M);
          SW += Twt_bb(m0, it) * g;
        }
        S_PW += m_q(q) * (SZ + SW);
      }
      S_PW *= cplx(1.0 / double(nkpts));
    }

    const double scale = std::max(std::abs(S_SG), std::abs(S_PW));
    const double rel = std::abs(S_SG + S_PW) / std::max(scale, 1e-300);
    const double ctrl = std::abs(S_SG - S_PW) / std::max(scale, 1e-300);
    app_log(1, "ibz cons: S_SigmaG = ({:.10e}, {:.2e}), S_PW = ({:.10e}, {:.2e})",
            S_SG.real(), S_SG.imag(), S_PW.real(), S_PW.imag());
    app_log(1, "ibz cons: |S_SG + S_PW| / scale = {:.3e} (sign-flip control = {:.3f}; "
               "leakage = {:.3e})", rel, ctrl, vtx.sym_leakage_max());
    REQUIRE(scale > 1e-8);
    // kernel-accuracy + D-matrix-accuracy + O(leakage) class (memo section 7 item 2;
    // measured 5.05e-5 on qe_lih222_sym with leak = 0 -- ~13x the nosym identity's
    // 3.78e-6, the D-overlap accuracy class); the sign-flip control breaks at O(1)
    REQUIRE(rel < 5e-3);
    REQUIRE(ctrl > 1.5);

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  TEST_CASE("vertex_ibz_noop_sym", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_noop_sym skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_noop";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](bool with_empty_vertex) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, with_empty_vertex ? "2nd_exchange" : "none",
                            nda::range(0, 0), mf->nbnd());
      REQUIRE(not vtx.active());
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     1, false, 1e-12, true);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_pair(e_hf, e_corr);
    };

    auto [ehf0, ec0] = run(false);
    auto [ehf1, ec1] = run(true);
    app_log(1, "ibz noop: plain sym-scGW e_hf = {:.17g}, e_corr = {:.17g}", ehf0, ec0);
    app_log(1, "ibz noop: empty-vertex   e_hf = {:.17g}, e_corr = {:.17g}", ehf1, ec1);
    REQUIRE(ehf0 == ehf1);   // bitwise
    REQUIRE(ec0 == ec1);     // bitwise
#endif  // ENABLE_DLR
  }

} // bdft_tests
