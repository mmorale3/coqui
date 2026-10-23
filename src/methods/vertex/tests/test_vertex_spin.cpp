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
#include "methods/embedding/projector_t.h"
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
    REQUIRE(mf2->nkpts_ibz() == mf2->nkpts());   // pyscf: no symmetry (the symmetric mesh: vertex_spin_lsda_sym)
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

  /**
   * The SPIN-RESOLVED local outputs, on the spin-doubled pyscf_si222 of vertex_spin_doubling (ns = 2 vs ns = 1):
   *   - the orbital (E-leg) local ladder of eval_pol_ladder_loc_whalf (the pi_lad_loc_orb producer), (nw, nq, ns nab, ns nab):
   *     each spin block is the ns = 1 object over 2 (its spin factor), the blocks of different spins vanish;
   *   - the dynbse Wannier-pair output frame (pol_vertex_wannier_frame = "pair", the inu = 0 dump): the spin-orbital pair
   *     labels (s M + a, s M + b); the (s, s) blocks are the ns = 1 columns over 2, the others vanish.
   */
  TEST_CASE("vertex_spin_resolved_outputs", "[methods][vertex][spin]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_spin_resolved_outputs skipped: build has ENABLE_DLR=OFF.");
#else
    namespace sd = spin_detail;
    decltype(nda::range::all) all;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    vertex_debug::set("scf_causal_proj=0");
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    auto [src_dir, src_prefix] = utils::utest_filename("pyscf_si222");
    const std::string dst_dir = "coqui_spin2_si222_res";
    if (mpi_context->comm.root()) sd::make_spin_doubled_pyscf(src_dir, dst_dir);
    mpi_context->comm.barrier();
    auto mf1 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    auto mf2 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source, dst_dir, src_prefix));
    REQUIRE(mf2->nspin() == 2);
    thc_reader_t thc1(mf1, make_thc_reader_ptree(mf1->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf1->ecutrho(), 1, 1024));
    thc_reader_t thc2(mf2, make_thc_reader_ptree(mf2->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf2->ecutrho(), 1, 1024));
    auto eri1 = mb_eri_t(thc1, thc1);
    auto eri2 = mb_eri_t(thc2, thc2);
    const nda::range C(2, 6);
    const long nc = C.size(), norb = 2, nab = norb * norb;

    // pair = false: the static ladder + the orbital local ladder after the run; pair = true: the dynamic ladder in the
    // Wannier-pair frame (one identity projector per spin), its inu = 0 dump. The ns = 1 pair run selects the secondary
    // points and writes them, the ns = 2 one reuses them (the selection with a rotation needs nspin_in_basis = ns)
    const std::string pts = "coqui_spin_res_pts.h5";
    auto run = [&](std::shared_ptr<mf::MF> mf, auto &thc, auto &eri, std::string const &tag, bool pair) {
      const std::string out = std::string("coqui_spin_res_") + (pair ? "pair_" : "loc_") + tag;
      const long ns = mf->nspin(), nk = mf->nkpts();
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", out);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, out);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", C, -1, 1e-8, -1.0, -1.0, -1.0);
      if (pair) {
        vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true);
        vtx.set_ladder_dyn_dump(true);
        vtx.set_wannier_frame("pair");
        vtx.set_isdf_points(ns == 1 ? "" : pts, ns == 1);
        nda::array<cplx, 5> C_ksIai(nk, ns, 1, nc, nc);
        C_ksIai() = cplx(0.0);
        for (long ik = 0; ik < nk; ++ik)
          for (long is = 0; is < ns; ++is)
            for (long a = 0; a < nc; ++a) C_ksIai(ik, is, 0, a, a) = cplx(1.0);
        nda::array<long, 3> bw(1, 1, 2);
        bw(0, 0, 0) = C.first() + 1;
        bw(0, 0, 1) = C.last();
        auto kc = nda::make_regular(mf->kpts_crystal());
        projector_t proj(*mf, C_ksIai, bw, kc, false, false);
        vtx.set_wannier_projector(proj, true);
      } else {
        vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
      }
      scr_eri.set_vertex(&vtx);
      scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true);
      mpi_context->comm.barrier();
      nda::array<cplx, 4> P;   // (nw, nq, n, n)
      if (pair) {
        long gen = -1;
        for (auto const &e : std::filesystem::directory_iterator(".")) {
          const std::string f = e.path().filename().string(), pre = out + ".pol_nu0.g";
          if (f.rfind(pre, 0) == 0) gen = std::max(gen, std::stol(f.substr(pre.size())));
        }
        REQUIRE(gen >= 0);
        h5::file f(out + ".pol_nu0.g" + std::to_string(gen) + ".h5", 'r');
        h5::group g(f);
        nda::array<cplx, 3> Pg;
        nda::h5_read(g, "Pi_gam1", Pg);
        long nspin = 0;
        h5::h5_read(g, "nspin", nspin);
        REQUIRE(nspin == ns);
        P = nda::array<cplx, 4>(1, Pg.shape(0), Pg.shape(1), Pg.shape(2));
        P(0, all, all, all) = Pg;
        if (ns == 1 and mpi_context->comm.root()) std::filesystem::copy_file(out + ".secpts.h5", pts, std::filesystem::copy_options::overwrite_existing);
      } else {
        nda::array<cplx, 4> U(ns, nk, norb, nc);
        U() = cplx(0.0);
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            for (long m = 0; m < norb; ++m) U(is, ik, m, m + 1) = cplx(1.0);   // two orbitals of the window, its bands 1 and 2
        P = scr_eri.pol_vertex_instance()->eval_pol_ladder_loc_whalf(mb_state, thc, U);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return P;
    };

    // ns = 2 (labels L2(s, p)) against ns = 1 (labels p): the (s, s) blocks are P1 / 2, the rest vanish
    auto check = [&](std::string const &what, nda::array<cplx, 4> const &P1, nda::array<cplx, 4> const &P2, long n1,
                     auto const &L2) {
      REQUIRE(P1.shape(2) == n1);
      REQUIRE(P2.shape(2) == (n1 == nab ? 2 * nab : 4 * n1));
      double d = 0.0, off = 0.0, sc = 0.0;
      nda::array<bool, 2> diag(P2.shape(2), P2.shape(3));
      diag() = false;
      for (long s = 0; s < 2; ++s)
        for (long p = 0; p < n1; ++p)
          for (long r = 0; r < n1; ++r) {
            diag(L2(s, p), L2(s, r)) = true;
            for (long j = 0; j < P1.shape(0); ++j)
              for (long iq = 0; iq < P1.shape(1); ++iq) {
                d = std::max(d, std::abs(P2(j, iq, L2(s, p), L2(s, r)) - 0.5 * P1(j, iq, p, r)));
                sc = std::max(sc, std::abs(P1(j, iq, p, r)));
              }
          }
      for (long a = 0; a < P2.shape(2); ++a)
        for (long b = 0; b < P2.shape(3); ++b)
          if (not diag(a, b))
            for (long j = 0; j < P2.shape(0); ++j)
              for (long iq = 0; iq < P2.shape(1); ++iq) off = std::max(off, std::abs(P2(j, iq, a, b)));
      app_log(1, "vertex_spin_resolved_outputs [{}]: max |P(ns = 2, s s) - P(ns = 1) / 2| = {:.3e}, max off-spin-block {:.3e} "
                 "(|P(ns = 1)| {:.3e})", what, d, off, sc);
      REQUIRE(sc > 1e-10);
      REQUIRE(d < 1e-9 * sc);
      REQUIRE(off < 1e-12 * sc);
    };

    auto L1 = run(mf1, thc1, eri1, "ns1", false);
    auto L2 = run(mf2, thc2, eri2, "ns2", false);
    check("orbital local ladder", L1, L2, nab, [&](long s, long p) { return s * nab + p; });
    auto W1 = run(mf1, thc1, eri1, "ns1", true);
    auto W2 = run(mf2, thc2, eri2, "ns2", true);
    // the pair (a, b) of spin s is the spin-orbital pair (s nc + a, s nc + b)
    check("Wannier pair frame", W1, W2, nc * nc,
          [&](long s, long p) { return (s * nc + p / nc) * (2 * nc) + s * nc + p % nc; });
    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) { std::filesystem::remove_all(dst_dir); std::filesystem::remove(pts); }
    mpi_context->comm.barrier();
#endif
  }

  /**
   * Collinear spin on a SYMMETRY-REDUCED mesh: the Si 2x2x2 LSDA fixture (zero moment, 8 bands closing every degenerate
   * set), whose two spins carry different orbital gauges inside the degenerate subspaces, so the IBZ unfolding of each
   * spin needs its own band rotations (the vertex's Dc / Dfold / effective columns and the unfold fit). On the IBZ the
   * two spins must give the same G and Sigma up to that gauge: the spectra of their Hermitian parts at every (tau, k)
   * agree, and each spin's agree with the unreduced mesh of the same fixture. Both only to the accuracy of the symmetry
   * path on this fixture (the ISDF basis is not symmetric; ns = 1 shows the same IBZ-vs-full deviation), which still
   * separates the per-spin rotations from the spin-0 ones by more than a decade. Chains: "bs" (2nd_exchange, static
   * rung: Sigma^C + Pi^C), "pol" (the production P + Sigma chain of vertex_spin_doubling: dynamic-rung ladder through
   * pol_vertex_sc and the pair Sigma, its IBZ solve + star fold) and "wan" ("pol" in Wannier mode with one identity
   * projector per spin, the aux frame).
   */
  TEST_CASE("vertex_spin_lsda_sym", "[methods][vertex][spin][ibz]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_spin_lsda_sym skipped: build has ENABLE_DLR=OFF.");
#else
    decltype(nda::range::all) all;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    vertex_debug::set("scf_causal_proj=0");
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const nda::range C(0, 8);

    // the largest difference of the spectra of the Hermitian parts of the two spin blocks, relative to max |A|
    auto spin_asym = [&](nda::array<cplx, 5> const &A) {
      const long nb = A.shape(3);
      double d = 0.0, n = 0.0;
      nda::array<cplx, 2> A0(nb, nb), A1(nb, nb);
      for (long it = 0; it < A.shape(0); ++it)
        for (long ik = 0; ik < A.shape(2); ++ik) {
          A0 = A(it, 0, ik, all, all);
          A1 = A(it, 1, ik, all, all);
          A0 = 0.5 * (A0 + nda::dagger(A0));
          A1 = 0.5 * (A1 + nda::dagger(A1));
          auto e0 = std::get<0>(nda::linalg::eigenelements(A0));
          auto e1 = std::get<0>(nda::linalg::eigenelements(A1));
          for (long b = 0; b < nb; ++b) { d = std::max(d, std::abs(e0(b) - e1(b))); n = std::max(n, std::abs(e0(b))); }
        }
      return d / n;
    };

    nda::array<double, 2> kc_last;   // the crystal k of the last run
    auto run = [&](std::string const &chain, std::string const &fx) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
      kc_last = nda::array<double, 2>(mf->kpts_crystal());
      REQUIRE(mf->nspin() == 2);
      REQUIRE(mf->nbnd() == C.size());
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      const std::string out = "coqui_spin_lsda_" + chain + "_" + fx;
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
        vtx.set_sigma_pair_ibz(mf->nkpts_ibz() < mf->nkpts());   // the IBZ solve + star fold on the symmetric mesh
        solvers::vertex_sc::config sc;
        sc.mode = "hybrid";
        sc.mesh = {2, 2, 2};
        vtx.set_pol_sc(sc);
        if (chain == "wan") {   // one identity projector per spin on the window
          const long nk = mf->nkpts(), M = C.size();
          nda::array<cplx, 5> C_ksIai(nk, 2, 1, M, M);
          C_ksIai() = cplx(0.0);
          for (long ik = 0; ik < nk; ++ik)
            for (long is = 0; is < 2; ++is)
              for (long a = 0; a < M; ++a) C_ksIai(ik, is, 0, a, a) = cplx(1.0);
          nda::array<long, 3> bw(1, 1, 2);
          bw(0, 0, 0) = C.first() + 1;
          bw(0, 0, 1) = C.last();
          auto kc = nda::make_regular(mf->kpts_crystal());
          projector_t proj(*mf, C_ksIai, bw, kc, false, false);
          vtx.set_wannier_frame("aux");
          vtx.set_wannier_projector(proj, true);
        }
        scr_eri.set_vertex(&vtx);
      }
      spin_detail::run_out r;
      std::tie(r.e_hf, r.e_corr) = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                            2, false, 1e-9, true);
      mpi_context->comm.barrier();
      r.G = nda::array<cplx, 5>(mb_state.sG_tskij.value().local());
      r.Sigma = nda::array<cplx, 5>(mb_state.sSigma_tskij.value().local());
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return r;
    };

    // the largest deviation of the spin-s spectra of A on the IBZ from those of B on the full mesh at the same k, relative
    // to max |B| (per spin: a wrong rotation of one spin shows in that spin)
    auto ibz_dev = [&](nda::array<cplx, 5> const &A, nda::array<double, 2> const &kA, nda::array<cplx, 5> const &B,
                       nda::array<double, 2> const &kB, long is) {
      const long nb = A.shape(3);
      double d = 0.0, n = 0.0;
      nda::array<cplx, 2> A0(nb, nb), B0(nb, nb);
      for (long ik = 0; ik < A.shape(2); ++ik) {
        long jk = -1;
        for (long j = 0; j < kB.shape(0); ++j) {
          double x = 0.0;
          for (int c = 0; c < 3; ++c) { const double y = kA(ik, c) - kB(j, c); x += std::abs(y - std::round(y)); }
          if (x < 1e-6) jk = j;
        }
        REQUIRE(jk >= 0);
        for (long it = 0; it < A.shape(0); ++it) {
          A0 = A(it, is, ik, all, all);
          B0 = B(it, is, jk, all, all);
          A0 = 0.5 * (A0 + nda::dagger(A0));
          B0 = 0.5 * (B0 + nda::dagger(B0));
          auto ea = std::get<0>(nda::linalg::eigenelements(A0));
          auto eb = std::get<0>(nda::linalg::eigenelements(B0));
          for (long b = 0; b < nb; ++b) { d = std::max(d, std::abs(ea(b) - eb(b))); n = std::max(n, std::abs(eb(b))); }
        }
      }
      return d / n;
    };

    for (std::string chain : {std::string("bs"), std::string("pol"), std::string("wan")}) {
      auto rs = run(chain, "qe_si222_lsda_sym");
      auto ks = kc_last;
      auto rn = run(chain, "qe_si222_lsda_nosym");
      auto kn = kc_last;
      const double aG_s = spin_asym(rs.G), aS_s = spin_asym(rs.Sigma);
      const double aG_n = spin_asym(rn.G), aS_n = spin_asym(rn.Sigma);
      double dS = 0.0;
      for (long is = 0; is < 2; ++is) dS = std::max(dS, ibz_dev(rs.Sigma, ks, rn.Sigma, kn, is));
      app_log(1, "vertex_spin_lsda_sym [{}]: IBZ e_hf {:.12f} e_corr {:.12f}, spin asymmetry G {:.3e} Sigma {:.3e}; full mesh "
                 "e_hf {:.12f} e_corr {:.12f}, spin asymmetry G {:.3e} Sigma {:.3e}; max_s IBZ vs full mesh Sigma {:.3e}", chain,
              rs.e_hf, rs.e_corr, aG_s, aS_s, rn.e_hf, rn.e_corr, aG_n, aS_n, dS);
      REQUIRE(std::isfinite(rs.e_corr));
      // the full mesh: the spin symmetry of the LSDA run (zero moment) through each chain -- 8e-7 on Sigma for "bs", 1e-3
      // for "pol" (the dynamic chain's own floor between two orbital gauges)
      REQUIRE(aG_n < 5e-3);
      REQUIRE(aS_n < 5e-3);
      // the IBZ: each spin at the accuracy of the symmetry path (the ISDF basis is not symmetric: Sigma against the full mesh
      // 5e-3 "bs", 1.4e-2 "pol", ns = 1 alike); the spin-0 rotations applied to spin 1 gave 7.5e-2 / 7.1e-2 ("bs")
      REQUIRE(aG_s < 1e-2);
      REQUIRE(aS_s < 1e-2);
      REQUIRE(dS < 3e-2);
      REQUIRE(std::abs(rs.e_corr - rn.e_corr) < 5e-3 * std::abs(rn.e_corr));
    }
#endif
  }

} // namespace bdft_tests
