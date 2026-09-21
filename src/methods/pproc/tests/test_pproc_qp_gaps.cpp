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

#undef NDEBUG

#include <cmath>
#include <cstdio>
#include <filesystem>

#include "catch2/catch.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"
#include "utilities/interpolation_utils.hpp"

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/embedding/projector_t.h"
#include "methods/pproc/pproc_t.h"
#include "methods/pproc/pproc_drivers.hpp"

/**
 * P24 / G31 (notes/vertex_perf_plan.md): Wannier interpolation of the quasiparticle bands in qp_gaps.
 *
 * The identity gate: Wannier interpolation is EXACT on the coarse mesh -- at a mean-field k the
 * fine-mesh bands of pproc_t::interpolate_qp_bands_on_mesh must reproduce the eigenvalues of the
 * downfolded Heff(k) (the projector's window block in the Wannier basis) to 1e-8 Ha, for the fine
 * mesh equal to the mean-field mesh and for a refinement that contains it. Two probes on the lih222
 * projector (2 Wannier orbitals on the primary bands [0, 2), unitary within the window):
 *   (a) Heff = diag(mean-field eigenvalues): the reference is the mean-field bands 0 and 1 themselves;
 *   (b) a fixed-seed random Hermitian Heff coupling all 16 bands: the reference is the Hermitized
 *       downfold of the same matrix, computed here independently of the k -> R -> k chain.
 * The second case runs the qp_gaps DRIVER (post_processing) with qp_gaps_interp = true on a short scGW
 * checkpoint: the "_interp" datasets must exist, E_ska_interp at the coarse k must equal the downfolded
 * qp_approx/Heff_skij eigenvalues (1e-8), and fundamental_eV_interp must be the scan of E_ska_interp.
 *
 * HOW TO RUN:  <build>/tests/bin/test_methods_pproc   (Catch2 v2: the bare binary, or -c/tag filters)
 */

namespace bdft_tests {

  using namespace methods;

  namespace {

    // fine-mesh point ikf -> the mean-field k index it coincides with modulo a reciprocal lattice vector, -1 if none
    nda::array<long, 1> map_to_mf_kpts(nda::array<double, 2> const &kf, mf::MF const &mf, double tol = 1e-8) {
      auto kc = mf.kpts_crystal();
      const long nkf = kf.shape(0), nk = kc.shape(0);
      nda::array<long, 1> map(nkf);
      map() = -1;
      for (long i = 0; i < nkf; ++i) {
        for (long ik = 0; ik < nk and map(i) < 0; ++ik) {
          bool same = true;
          for (int d = 0; d < 3 and same; ++d) {
            double x = kf(i, d) - kc(ik, d);
            x -= std::round(x);
            same = (std::abs(x) < tol);
          }
          if (same) map(i) = ik;
        }
      }
      return map;
    }

    // reference: eigenvalues of the Hermitized downfolded H(k) on the coarse full-BZ mesh, (ns, nk, nImpOrbs)
    nda::array<double, 3> downfolded_bands(projector_t const &proj, nda::array<ComplexType, 4> const &H,
                                           mpi3::communicator &comm) {
      auto Hw = proj.downfold_k(H, comm);   // (ns, nk, 1, a, b)
      const long ns = Hw.shape(0), nk = Hw.shape(1), na = Hw.shape(3);
      nda::array<double, 3> E(ns, nk, na);
      nda::array<ComplexType, 2> A(na, na);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) {
          for (long a = 0; a < na; ++a)
            for (long b = 0; b < na; ++b)
              A(a, b) = 0.5 * (Hw(is, ik, 0, a, b) + std::conj(Hw(is, ik, 0, b, a)));
          E(is, ik, nda::range::all) = nda::linalg::eigenvalues(A);
        }
      return E;
    }

    // max |E_fine(s, ikf, a) - E_ref(s, map(ikf), a)| over the fine points that sit on the coarse mesh;
    // also counts them (every coarse point must be hit when the fine mesh contains the coarse one)
    std::pair<double, long> coarse_identity_error(nda::array<double, 3> const &E_fine, nda::array<double, 2> const &kf,
                                                  nda::array<double, 3> const &E_ref, mf::MF const &mf) {
      auto map = map_to_mf_kpts(kf, mf);
      const long ns = E_fine.shape(0), nkf = E_fine.shape(1), na = E_fine.shape(2);
      REQUIRE(E_ref.shape(0) == ns);
      REQUIRE(E_ref.shape(2) == na);
      double err = 0.0;
      long nhit = 0;
      for (long i = 0; i < nkf; ++i) {
        if (map(i) < 0) continue;
        ++nhit;
        for (long is = 0; is < ns; ++is)
          for (long a = 0; a < na; ++a)
            err = std::max(err, std::abs(E_fine(is, i, a) - E_ref(is, map(i), a)));
      }
      return {err, nhit};
    }

  } // namespace

  TEST_CASE("qp_gaps_interp_identity_lih222", "[methods][pproc][qp_gaps]") {
    auto &mpi = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    auto [outdir, prefix] = utils::utest_filename("qe_lih222");
    const std::string wannier_file = outdir + "/lih_wan.h5";
    REQUIRE(std::filesystem::exists(wannier_file));

    const long ns = mf->nspin(), nk = mf->nkpts(), nb = mf->nbnd();
    REQUIRE(nk == mf->nkpts_ibz());   // the fixture has no space-group reduction: eigval() is the full BZ

    projector_t proj(*mf, wannier_file, false, false);
    REQUIRE(proj.nImps() == 1);
    const long na = proj.nImpOrbs();
    const auto W = proj.W_rng()[0];
    REQUIRE(na == W.size());          // the shipped lih projector is a unitary within its window
    REQUIRE(W.last() <= nb);

    auto [rw, rp] = utils::WS_rgrid(mf->lattv(), mf->kp_grid());
    nda::array<long, 2> Rpts_idx(rp);
    nda::array<long, 1> Rpts_weights(rw);
    REQUIRE(Rpts_idx.shape(0) == Rpts_weights.shape(0));

    const std::array<long, 3> mesh_c = {long(mf->kp_grid()(0)), long(mf->kp_grid()(1)), long(mf->kp_grid()(2))};
    const std::array<long, 3> mesh_f = {2 * mesh_c[0], 2 * mesh_c[1], 2 * mesh_c[2]};
    const long nkc = mesh_c[0] * mesh_c[1] * mesh_c[2];
    REQUIRE(nkc == nk);

    auto eig = mf->eigval();   // (ns, nk, nb)

    SECTION("diagonal_heff_is_the_mean_field_bands") {
      // Heff = diag(eps): with a unitary projector the window block's eigenvalues are eps(0..na-1) exactly
      nda::array<ComplexType, 4> H(ns, nk, nb, nb);
      H() = ComplexType(0.0);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long i = 0; i < nb; ++i) H(is, ik, i, i) = ComplexType(eig(is, ik, i), 0.0);

      auto [kf, Ef] = pproc_t::interpolate_qp_bands_on_mesh(*mpi, *mf, proj, H(), Rpts_idx, Rpts_weights, mesh_c);
      REQUIRE(kf.shape(0) == nkc);
      REQUIRE(Ef.shape(0) == ns);
      REQUIRE(Ef.shape(1) == nkc);
      REQUIRE(Ef.shape(2) == na);
      // Gamma is the first fine-mesh point
      for (int d = 0; d < 3; ++d) REQUIRE(kf(0, d) == 0.0);

      nda::array<double, 3> E_ref(ns, nk, na);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long a = 0; a < na; ++a) E_ref(is, ik, a) = eig(is, ik, W.first() + a);
      auto [err_mf, nhit] = coarse_identity_error(Ef, kf, E_ref, *mf);
      auto [err_df, nhit2] = coarse_identity_error(Ef, kf, downfolded_bands(proj, H, mpi->comm), *mf);
      app_log(1, "qp_gaps_interp_identity_lih222 (diagonal Heff, mesh = mf mesh): {} / {} fine points on the coarse mesh; "
                 "max |E_interp - eps_mf| = {:.3e} Ha, max |E_interp - eig(downfold)| = {:.3e} Ha", nhit, nkc, err_mf, err_df);
      REQUIRE(nhit == nk);
      REQUIRE(nhit2 == nk);
      REQUIRE(err_mf < 1e-8);
      REQUIRE(err_df < 1e-8);
    }

    SECTION("random_hermitian_heff_mesh_and_refinement") {
      // a fixed-seed Hermitian Heff coupling every band (the projector window and the rest): the reference is the
      // Hermitized downfold of the same matrix; the chain k -> R (WS, complex H(R) kept) -> k must be the identity
      // at the coarse points, on the mf mesh and on the 2x refinement that contains it
      nda::array<ComplexType, 4> H(ns, nk, nb, nb);
      unsigned long seed = 20260921ul;
      auto rnd = [&seed]() {
        seed = seed * 6364136223846793005ul + 1442695040888963407ul;
        return double((seed >> 33) % 100000ul) / 100000.0 - 0.5;
      };
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long i = 0; i < nb; ++i) {
            H(is, ik, i, i) = ComplexType(eig(is, ik, i) + 0.05 * rnd(), 0.0);
            for (long j = i + 1; j < nb; ++j) {
              ComplexType z(0.05 * rnd(), 0.05 * rnd());
              H(is, ik, i, j) = z;
              H(is, ik, j, i) = std::conj(z);
            }
          }
      auto E_ref = downfolded_bands(proj, H, mpi->comm);

      auto [kf_c, Ef_c] = pproc_t::interpolate_qp_bands_on_mesh(*mpi, *mf, proj, H(), Rpts_idx, Rpts_weights, mesh_c);
      auto [err_c, nhit_c] = coarse_identity_error(Ef_c, kf_c, E_ref, *mf);
      app_log(1, "qp_gaps_interp_identity_lih222 (random Heff, mesh = mf mesh): {} / {} on the coarse mesh; max |E_interp - eig(downfold)| = {:.3e} Ha",
              nhit_c, nkc, err_c);
      REQUIRE(nhit_c == nk);
      REQUIRE(err_c < 1e-8);

      auto [kf_f, Ef_f] = pproc_t::interpolate_qp_bands_on_mesh(*mpi, *mf, proj, H(), Rpts_idx, Rpts_weights, mesh_f);
      REQUIRE(kf_f.shape(0) == 8 * nkc);
      auto [err_f, nhit_f] = coarse_identity_error(Ef_f, kf_f, E_ref, *mf);
      double emin = 1e9, emax = -1e9;
      for (long is = 0; is < ns; ++is)
        for (long i = 0; i < kf_f.shape(0); ++i)
          for (long a = 0; a < na; ++a) {
            REQUIRE(std::isfinite(Ef_f(is, i, a)));
            emin = std::min(emin, Ef_f(is, i, a)); emax = std::max(emax, Ef_f(is, i, a));
          }
      app_log(1, "qp_gaps_interp_identity_lih222 (random Heff, 2x refinement): {} / {} on the coarse mesh; max |E_interp - eig(downfold)| = {:.3e} Ha; "
                 "interpolated bands in [{:.4f}, {:.4f}] Ha", nhit_f, kf_f.shape(0), err_f, emin, emax);
      REQUIRE(nhit_f == nk);   // the refinement contains every coarse point exactly once
      REQUIRE(err_f < 1e-8);
    }
    mpi->comm.barrier();
  }

  TEST_CASE("qp_gaps_interp_driver_lih222", "[methods][pproc][qp_gaps][driver]") {
#ifndef ENABLE_DLR
    SUCCEED("qp_gaps_interp_driver_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const std::string output = "coqui_pproc_qp_gaps";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    auto [outdir, prefix] = utils::utest_filename("qe_lih222");
    const std::string wannier_file = outdir + "/lih_wan.h5";
    REQUIRE(mf->nkpts() == mf->nkpts_ibz());

    // one Dyson scGW iteration -> the checkpoint the qp_gaps driver consumes (the cvv_build_lih222 recipe)
    {
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     1, false, 1e-9, true);
      app_log(1, "qp_gaps_interp_driver_lih222: scGW state e_hf = {}, e_corr = {}", e_hf, e_corr);
    }
    mpi->comm.barrier();
    REQUIRE(std::filesystem::exists(output + ".mbpt.h5"));

    // the qp_gaps driver, knob ON, fine mesh = the mean-field mesh (the identity configuration)
    {
      ptree pt;
      pt.put("prefix", output);
      pt.put("outdir", ".");
      pt.put("grp_name", "scf");
      pt.put("iteration", -1);
      pt.put("qp_gaps_interp", true);
      pt.put("wannier_file", wannier_file);
      ptree mesh_node;
      for (int d = 0; d < 3; ++d) {
        ptree child;
        child.put_value<long>(mf->kp_grid()(d));
        mesh_node.push_back(std::make_pair("", child));
      }
      pt.add_child("qp_gaps_interp_mesh", mesh_node);
      post_processing("qp_gaps", mf, pt);
    }
    mpi->comm.barrier();

    // read back: the mesh report, the _interp report, Heff for the independent reference
    long fiter = 0;
    double mu = 0.0, fund_interp = 0.0, fund_mesh = 0.0;
    nda::array<ComplexType, 4> Heff;
    nda::array<double, 3> E_interp;
    nda::array<double, 2> k_interp;
    nda::array<long, 1> mesh_h5;
    {
      h5::file file(output + ".mbpt.h5", 'r');
      auto scf_grp = h5::group(file).open_group("scf");
      h5::h5_read(scf_grp, "final_iter", fiter);
      auto qp_grp = scf_grp.open_group("iter" + std::to_string(fiter) + "/qp_approx");
      REQUIRE(qp_grp.has_dataset("Heff_skij"));
      REQUIRE(qp_grp.has_subgroup("gaps"));
      auto gaps_grp = qp_grp.open_group("gaps");
      for (auto const &name : {"fundamental_eV", "direct_gamma_eV", "direct_min_eV", "vbm_k", "cbm_k",
                               "fundamental_eV_interp", "direct_gamma_eV_interp", "direct_min_eV_interp",
                               "vbm_eV_interp", "cbm_eV_interp", "vbm_k_interp", "cbm_k_interp", "direct_min_k_interp",
                               "vbm_kpt_crys_interp", "cbm_kpt_crys_interp", "direct_min_kpt_crys_interp",
                               "interp_mesh", "kpts_crys_interp", "E_ska_interp"})
        REQUIRE(gaps_grp.has_dataset(name));
      nda::h5_read(qp_grp, "Heff_skij", Heff);
      h5::h5_read(qp_grp, "mu", mu);
      h5::h5_read(gaps_grp, "fundamental_eV", fund_mesh);
      h5::h5_read(gaps_grp, "fundamental_eV_interp", fund_interp);
      nda::h5_read(gaps_grp, "E_ska_interp", E_interp);
      nda::h5_read(gaps_grp, "kpts_crys_interp", k_interp);
      nda::h5_read(gaps_grp, "interp_mesh", mesh_h5);
    }
    for (int d = 0; d < 3; ++d) REQUIRE(mesh_h5(d) == mf->kp_grid()(d));
    REQUIRE(E_interp.shape(1) == mf->nkpts());
    REQUIRE(k_interp.shape(0) == mf->nkpts());

    projector_t proj(*mf, wannier_file, false, false);
    REQUIRE(E_interp.shape(2) == proj.nImpOrbs());
    auto E_ref = downfolded_bands(proj, Heff, mpi->comm);
    auto [err, nhit] = coarse_identity_error(E_interp, k_interp, E_ref, *mf);
    app_log(1, "qp_gaps_interp_driver_lih222: {} / {} fine points on the coarse mesh; max |E_ska_interp - eig(downfold Heff)| = {:.3e} Ha; "
               "fundamental gap mesh = {:.4f} eV, interp (window bands) = {:.4f} eV", nhit, mf->nkpts(), err, fund_mesh, fund_interp);
    REQUIRE(nhit == mf->nkpts());
    REQUIRE(err < 1e-8);

    // the driver's scan of E_ska_interp (occupation by mu) is what it wrote
    {
      const double HA = 27.211386245988;
      double vbm = -1e9, cbm = 1e9;
      for (long is = 0; is < E_interp.shape(0); ++is)
        for (long ik = 0; ik < E_interp.shape(1); ++ik)
          for (long a = 0; a < E_interp.shape(2); ++a) {
            const double e = E_interp(is, ik, a);
            if (e < mu) vbm = std::max(vbm, e); else cbm = std::min(cbm, e);
          }
      // the LiH projector window [0, 2) holds the two OCCUPIED bands only: the driver then reports the gap as n/a (NaN)
      if (vbm > -1e8 and cbm < 1e8) {
        REQUIRE(std::abs((cbm - vbm) * HA - fund_interp) < 1e-10);
        REQUIRE(fund_interp > 0.0);
      } else {
        REQUIRE(std::isnan(fund_interp));
        app_log(1, "qp_gaps_interp_driver_lih222: the window holds no empty band -> the interpolated fundamental gap is reported as n/a (NaN), as designed");
      }
    }

    mpi->comm.barrier();
    if (mpi->comm.root()) std::remove((output + ".mbpt.h5").c_str());
    mpi->comm.barrier();
#endif
  }

} // bdft_tests
