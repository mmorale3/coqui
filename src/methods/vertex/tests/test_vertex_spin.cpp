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

/**
 * Collinear spin (ns = 2) of the vertex drivers: the SPIN-DOUBLING identity.
 *
 * A spin-unpolarized system run as ns = 2 must reproduce the ns = 1 run: every spin block of G and Sigma equals the
 * ns = 1 one and the vertex polarization (a spin sum, the ns = 1 run carrying the factor 2) is the same object. The
 * ns = 2 copy of pyscf_si222 is generated here: the same orbitals (nspin = 2, nspin_in_basis = 1, so the THC basis and
 * the secondary points are those of the ns = 1 run) and every SCF array doubled along its spin axis.
 *
 * The chains are run end to end through scf_loop (both runs on identical inputs), so any spin routing defect of a
 * driver -- a loop over spin 0 only, a missing or doubled spin factor, a spin unit overwriting the other one's output --
 * shows as a difference between the runs:
 *   - "pol":  the production P + Sigma chain: dynamic-rung ladder (Gamma_1) injected into W (ladder_n2) through the
 *             in-run self-consistency store (pol_vertex_sc on the identity sublattice = "direct"), and the pair-resolved
 *             Sigma vertex (col dyn1, dynamic outer); the stored Pi-bar is compared as well;
 *   - "prod": "pol" with the production accelerations: the frequency-factorized rung (R = 4, reconstructed), the
 *             nu-sampled refreshes of the store, and the sampled Sigma passes against the in-run reference
 *             (pol_vertex_sigma_dyn_fit_file = "self", automatic nodes);
 *   - "bs":   the active B-S theory (vertex_type 2nd_exchange, static rung): Sigma^{C,x} + Sigma^{C,r} and Pi^C.
 */

#undef NDEBUG

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <string>
#include <vector>

#include <hdf5.h>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/vertex/vertex_debug.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"

namespace bdft_tests {

  using namespace methods;
  using cplx = std::complex<double>;

  namespace spin_detail {

    inline herr_t collect_name(hid_t, char const *name, H5L_info2_t const *, void *op) {
      static_cast<std::vector<std::string> *>(op)->emplace_back(name);
      return 0;
    }

    /** every attribute of the object `from` onto `to` (e.g. the "__complex__" tag of a complex dataset) */
    inline herr_t copy_attr(hid_t from, char const *name, H5A_info_t const *, void *op) {
      hid_t to = *static_cast<hid_t *>(op);
      hid_t ai = H5Aopen(from, name, H5P_DEFAULT);
      hid_t ty = H5Aget_type(ai), sp = H5Aget_space(ai);
      std::vector<char> buf(std::max<size_t>(H5Aget_storage_size(ai), H5Tget_size(ty)) + 1, 0);
      hid_t ao = H5Acreate2(to, name, ty, sp, H5P_DEFAULT, H5P_DEFAULT);
      herr_t st = H5Aread(ai, ty, buf.data());
      if (st >= 0) st = H5Awrite(ao, ty, buf.data());
      H5Aclose(ao); H5Sclose(sp); H5Tclose(ty); H5Aclose(ai);
      return st;
    }

    /** the ns = 2 copy of a spin-unpolarized pyscf fixture in dst: nspin = 2, nspin_in_basis = 1 (the orbitals of src,
     *  linked), every dataset of SCF doubled along its leading spin axis, everything else copied */
    inline void make_spin_doubled_pyscf(std::string const &src, std::string const &dst) {
      namespace fs = std::filesystem;
      fs::create_directories(dst);
      if (not fs::exists(dst + "/Orb_fft")) fs::create_directory_symlink(fs::absolute(src + "/Orb_fft"), dst + "/Orb_fft");
      hid_t fi = H5Fopen((src + "/pyscf.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      hid_t fo = H5Fcreate((dst + "/pyscf.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      REQUIRE(fi >= 0);
      REQUIRE(fo >= 0);
      std::vector<std::string> names;
      H5Literate2(fi, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, collect_name, &names);
      for (auto const &n : names)
        if (n != "SCF" and n != "nspin") REQUIRE(H5Ocopy(fi, n.c_str(), fo, n.c_str(), H5P_DEFAULT, H5P_DEFAULT) >= 0);
      {
        int v[2] = {2, 1};
        char const *nm[2] = {"nspin", "nspin_in_basis"};
        for (int i = 0; i < 2; ++i) {
          hid_t sp = H5Screate(H5S_SCALAR);
          hid_t ds = H5Dcreate2(fo, nm[i], H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          REQUIRE(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v[i]) >= 0);
          H5Dclose(ds); H5Sclose(sp);
        }
      }
      hid_t gi = H5Gopen2(fi, "SCF", H5P_DEFAULT);
      hid_t go = H5Gcreate2(fo, "SCF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      std::vector<std::string> scf;
      H5Literate2(gi, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, collect_name, &scf);
      for (auto const &n : scf) {
        hid_t di = H5Dopen2(gi, n.c_str(), H5P_DEFAULT);
        hid_t si = H5Dget_space(di);
        const int rk = H5Sget_simple_extent_ndims(si);
        std::vector<hsize_t> dims(static_cast<size_t>(rk));
        H5Sget_simple_extent_dims(si, dims.data(), nullptr);
        REQUIRE(rk >= 1);
        REQUIRE(dims[0] == 1);
        hsize_t blk = 1;
        for (int i = 1; i < rk; ++i) blk *= dims[size_t(i)];
        std::vector<double> buf(size_t(2 * blk));
        REQUIRE(H5Dread(di, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) >= 0);
        std::copy_n(buf.begin(), blk, buf.begin() + long(blk));
        dims[0] = 2;
        hid_t so = H5Screate_simple(rk, dims.data(), nullptr);
        hid_t dout = H5Dcreate2(go, n.c_str(), H5T_IEEE_F64LE, so, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(H5Dwrite(dout, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) >= 0);
        REQUIRE(H5Aiterate2(di, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, copy_attr, &dout) >= 0);
        H5Dclose(dout); H5Sclose(so); H5Sclose(si); H5Dclose(di);
      }
      H5Gclose(go); H5Gclose(gi);
      H5Fclose(fo); H5Fclose(fi);
    }

    struct run_out {
      double e_hf = 0.0, e_corr = 0.0;
      nda::array<cplx, 5> G, Sigma;   // (nt, ns, nk, nbnd, nbnd)
      nda::array<cplx, 4> Pi;         // the stored vertex polarization (chain "pol")
    };

    /** max |A(s) - B(0)| over every spin block s of A (ns = 2) against B (ns = 1), and max |B| */
    inline std::pair<double, double> spin_block_diff(nda::array<cplx, 5> const &A, nda::array<cplx, 5> const &B) {
      REQUIRE(A.shape(1) == 2);
      REQUIRE(B.shape(1) == 1);
      REQUIRE(A.shape(0) == B.shape(0));
      REQUIRE(A.shape(2) == B.shape(2));
      REQUIRE(A.shape(3) == B.shape(3));
      double d = 0.0, s = 0.0;
      for (long it = 0; it < A.shape(0); ++it)
        for (long is = 0; is < 2; ++is)
          for (long ik = 0; ik < A.shape(2); ++ik)
            for (long a = 0; a < A.shape(3); ++a)
              for (long b = 0; b < A.shape(4); ++b) {
                d = std::max(d, std::abs(A(it, is, ik, a, b) - B(it, 0, ik, a, b)));
                s = std::max(s, std::abs(B(it, 0, ik, a, b)));
              }
      return {d, s};
    }

  } // namespace spin_detail

  TEST_CASE("vertex_spin_doubling", "[methods][vertex][spin]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_spin_doubling skipped: build has ENABLE_DLR=OFF.");
#else
    namespace sd = spin_detail;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    // the causal projection of Sigma_c (on by default with a Sigma vertex) is an iterative fit whose result moves with
    // rounding-level differences of its input; off, so the identities below are at round-off
    vertex_debug::set("scf_causal_proj=0");
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");

    auto [src_dir, src_prefix] = utils::utest_filename("pyscf_si222");
    const std::string dst_dir = "coqui_spin2_si222";
    if (mpi_context->comm.root()) sd::make_spin_doubled_pyscf(src_dir, dst_dir);
    mpi_context->comm.barrier();

    auto mf1 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    auto mf2 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source, dst_dir, src_prefix));
    REQUIRE(mf1->nspin() == 1);
    REQUIRE(mf2->nspin() == 2);
    REQUIRE(mf2->nspin_in_basis() == 1);
    REQUIRE(mf2->nkpts() == mf1->nkpts());
    REQUIRE(mf2->nkpts_ibz() == mf2->nkpts());   // pyscf: no symmetry (the vertex symmetry path is ns = 1 only)
    thc_reader_t thc1(mf1, make_thc_reader_ptree(mf1->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf1->ecutrho(), 1, 1024));
    thc_reader_t thc2(mf2, make_thc_reader_ptree(mf2->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf2->ecutrho(), 1, 1024));
    auto eri1 = mb_eri_t(thc1, thc1);
    auto eri2 = mb_eri_t(thc2, thc2);
    const nda::range C(2, 6);   // two valence and two conduction bands of the 8 (4 occupied)

    auto run = [&](std::string const &chain, std::shared_ptr<mf::MF> mf, auto &eri, std::string const &tag) {
      const std::string out = "coqui_spin_" + chain + "_" + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", out);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, out);
      iter_scf::iter_scf_t iter_sol("damping");
      const bool bs = (chain == "bs");
      solvers::vertex_t vtx(&ft, bs ? "2nd_exchange" : "none", bs ? C : nda::range(0, 0), mf->nbnd(), "ignore_g0", "global", -1,
                            1e-8, -1.0, -1.0, bs ? "static" : "dynamic");
      if (bs) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      } else {
        vtx.set_pol_vertex("ladder", "w0_prev", C, -1, 1e-8, -1.0, -1.0, -1.0, "ladder_n2");
        vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true);
        vtx.set_sigma_pair(true, "dyn1", "dynamic", 1.0, true, false, "right");
        solvers::vertex_sc::config sc;
        sc.mode = "hybrid";
        sc.mesh = {2, 2, 2};   // the identity sublattice: runs as "direct"
        if (chain == "prod") {
          vtx.set_ladder_dyn_rung(4, 0.0);
          vtx.set_sigma_dyn(false, std::vector<long>{}, "self", 0);
          vtx.set_sigma_dyn_auto_nodes(8);
          sc.nu_nodes = 6;
        }
        vtx.set_pol_sc(sc);
        scr_eri.set_vertex(&vtx);
      }
      sd::run_out r;
      std::tie(r.e_hf, r.e_corr) = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                            2, false, 1e-9, true);
      mpi_context->comm.barrier();
      r.G = nda::array<cplx, 5>(mb_state.sG_tskij.value().local());
      r.Sigma = nda::array<cplx, 5>(mb_state.sSigma_tskij.value().local());
      if (not bs) {
        h5::file f(out + ".pol_sc.h5", 'r');
        h5::group g(f);
        nda::h5_read(g, "Pi", r.Pi);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return r;
    };

    for (std::string chain : {std::string("pol"), std::string("prod"), std::string("bs")}) {
      auto r2 = run(chain, mf2, eri2, "ns2");
      auto r1 = run(chain, mf1, eri1, "ns1");
      auto [dG, sG] = sd::spin_block_diff(r2.G, r1.G);
      auto [dS, sS] = sd::spin_block_diff(r2.Sigma, r1.Sigma);
      app_log(1, "vertex_spin_doubling [{}]: e_hf {:.12f} / {:.12f}, e_corr {:.12f} / {:.12f} (ns = 1 / ns = 2); max |G_s - G| = {:.3e} "
                 "(|G| {:.3e}), max |Sigma_s - Sigma| = {:.3e} (|Sigma| {:.3e})", chain, r1.e_hf, r2.e_hf, r1.e_corr, r2.e_corr,
              dG, sG, dS, sS);
      // "prod": the nu-mode reconstructions and the rung basis are eigen/pivot fits of rounding-different inputs
      // (measured 8e-10 relative on the stored Pi-bar, 3e-10 on Sigma): one decade looser
      const double tol = (chain == "prod") ? 1e-8 : 1e-9;
      REQUIRE(std::abs(r2.e_hf - r1.e_hf) < tol * std::abs(r1.e_hf));
      REQUIRE(std::abs(r2.e_corr - r1.e_corr) < 10.0 * tol * std::abs(r1.e_corr));
      REQUIRE(dG < tol * sG);
      REQUIRE(dS < tol * sS);
      if (chain != "bs") {
        REQUIRE(r1.Pi.shape() == r2.Pi.shape());
        double dP = 0.0, sP = 0.0;
        for (long i = 0; i < r1.Pi.size(); ++i) {
          dP = std::max(dP, std::abs(r2.Pi.data()[i] - r1.Pi.data()[i]));
          sP = std::max(sP, std::abs(r1.Pi.data()[i]));
        }
        app_log(1, "vertex_spin_doubling [{}]: stored vertex polarization max |Pi(ns = 2) - Pi(ns = 1)| = {:.3e} (|Pi| {:.3e})", chain,
                dP, sP);
        REQUIRE(sP > 1e-10);
        REQUIRE(dP < tol * sP);
      }
    }
    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) std::filesystem::remove_all(dst_dir);
    mpi_context->comm.barrier();
#endif
  }

  /**
   * A genuinely spin-polarized system (pyscf_li_222u, UHF, spin-dependent orbitals: nspin_in_basis = 2) through the
   * production P + Sigma chain of "vertex_spin_doubling": a SMOKE test of the spin-resolved secondary basis, the spin units
   * of the dynamic ladder and the per-spin pair Sigma -- finite energies, a finite stored vertex polarization, and spin
   * blocks of Sigma that differ. (The quantitative ns = 2 pins are the spin-doubling identity above, the per-spin kernel
   * decoupling of test_vertex_conservation and the UHF GF2 exchange check of test_vertex_gf2xchk.)
   */
  TEST_CASE("vertex_spin_uhf_smoke", "[methods][vertex][spin]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_spin_uhf_smoke skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    vertex_debug::set("scf_causal_proj=0");
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_li_222u"));
    REQUIRE(mf->nspin() == 2);
    REQUIRE(mf->nspin_in_basis() == 2);
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 10, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    const std::string out = "coqui_spin_uhf";
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", out);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, out);
    iter_scf::iter_scf_t iter_sol("damping");
    solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
    vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, mf->nbnd()), -1, 1e-8, -1.0, -1.0, -1.0, "ladder_n2");
    vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
    vtx.set_ladder_dyn_gamma1_only(true);
    vtx.set_sigma_pair(true, "dyn1", "dynamic", 1.0, true, false, "right");
    solvers::vertex_sc::config sc;
    sc.mode = "hybrid";
    sc.mesh = {2, 2, 2};
    vtx.set_pol_sc(sc);
    scr_eri.set_vertex(&vtx);
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 2, false,
                                   1e-9, true);
    mpi_context->comm.barrier();
    auto S = mb_state.sSigma_tskij.value().local();
    double d = 0.0, n = 0.0;
    for (long it = 0; it < S.shape(0); ++it)
      for (long ik = 0; ik < S.shape(2); ++ik)
        for (long a = 0; a < S.shape(3); ++a)
          for (long b = 0; b < S.shape(4); ++b) {
            d = std::max(d, std::abs(S(it, 0, ik, a, b) - S(it, 1, ik, a, b)));
            n = std::max(n, std::abs(S(it, 0, ik, a, b)));
          }
    nda::array<cplx, 4> Pi;
    {
      h5::file f(out + ".pol_sc.h5", 'r');
      h5::group g(f);
      nda::h5_read(g, "Pi", Pi);
    }
    double pmax = 0.0;
    bool finite = true;
    for (auto const &v : Pi) { pmax = std::max(pmax, std::abs(v)); finite = finite and std::isfinite(std::abs(v)); }
    app_log(1, "vertex_spin_uhf_smoke: e_hf {:.12f}, e_corr {:.12f}; max |Sigma_up - Sigma_dn| = {:.3e} (|Sigma_up| {:.3e}); "
               "max |stored Pi| = {:.3e}", e_hf, e_corr, d, n, pmax);
    REQUIRE(std::isfinite(e_hf));
    REQUIRE(std::isfinite(e_corr));
    REQUIRE(finite);
    REQUIRE(pmax > 0.0);
    REQUIRE(d > 1e-3 * n);
    mpi_context->comm.barrier();
    if (mpi_context->comm.root())
      for (auto const &e : std::filesystem::directory_iterator("."))
        if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
    mpi_context->comm.barrier();
#endif
  }

} // namespace bdft_tests
