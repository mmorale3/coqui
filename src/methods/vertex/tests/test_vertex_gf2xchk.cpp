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

// ISDF-Vertex: ABSOLUTE normalization / sign cross-check of Sigma^C against CoQui's
// own bare second-order exchange (GF2 "gf2" exchange channel).
//
// WHY THIS TEST EXISTS. Everything that pins Sigma^C today is INTERNAL:
//   - the dense double-Matsubara arbiter reproduces theoryB Eq. 22 -- but Eq. 22 is
//     the object under test, so a wrong power of 1/N_k or 1/beta in the NOTES is
//     reproduced faithfully by both sides;
//   - the conservation identity Tr[Sigma^C G] + Tr[P^C W] = 0 fixes the RELATIVE
//     normalization of the two cuts and cancels any factor COMMON to both;
//   - C = empty set is trivially satisfied by any prefactor.
// So the absolute scale and overall sign of Phi_2^C have never been checked against
// an independently written kernel. verified-cut-verdicts flagged exactly this cross-
// check ("Sigma^C at C = full, W -> v vs CoQui's pyscf-pinned bare-GF2 exchange") as
// RECOMMENDED BEFORE PRODUCTION; it was deferred and never run.
//
// THE CHECK. Set C = the FULL band range and give the vertex the BARE rung (no
// dynamic W in MBState => the kernel uses W = Z). Then Phi_2^C degenerates to the
// ordinary same-spin second-order exchange functional and
//
//     Sigma^C  ==  Sigma^{GF2 exchange}
//
// must hold identically. CoQui's GF2 total correlation energy is pinned against
// pyscf/CD-GF2 to 1e-8 (methods/GF2/tests/test_thc_gf2.cpp), so the right-hand side
// is an externally validated reference. The GF2 exchange is isolated by running the
// solver twice, with exchange_alg = "orb" and "none", and subtracting.
//
// The test REPORTS the complex ratio Sigma^C / Sigma^exc before asserting, so a
// clean prefactor discrepancy (-1, 2, N_k, ...) is read straight off the log.

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

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/GF2/gf2_t.h"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;

  namespace gf2xchk_detail {

    // best-fit complex scale of A onto B plus the residual after that scale:
    //   lambda = <B, A> / <B, B>,  resid = ||A - lambda B|| / ||A||
    struct fit_t { cplx lambda; double resid, normA, normB; };

    template<typename A5, typename B5>
    fit_t fit_scale(A5 const &A, B5 const &B) {
      cplx num(0.0, 0.0);
      double den = 0.0, nA = 0.0;
      auto sh = A.shape();
      for (long i0 = 0; i0 < sh[0]; ++i0)
        for (long i1 = 0; i1 < sh[1]; ++i1)
          for (long i2 = 0; i2 < sh[2]; ++i2)
            for (long i3 = 0; i3 < sh[3]; ++i3)
              for (long i4 = 0; i4 < sh[4]; ++i4) {
                cplx a = A(i0, i1, i2, i3, i4), b = B(i0, i1, i2, i3, i4);
                num += std::conj(b) * a;
                den += std::norm(b);
                nA += std::norm(a);
              }
      cplx lam = (den > 1e-300) ? num / cplx(den) : cplx(0.0);
      double num2 = 0.0;
      for (long i0 = 0; i0 < sh[0]; ++i0)
        for (long i1 = 0; i1 < sh[1]; ++i1)
          for (long i2 = 0; i2 < sh[2]; ++i2)
            for (long i3 = 0; i3 < sh[3]; ++i3)
              for (long i4 = 0; i4 < sh[4]; ++i4)
                num2 += std::norm(A(i0, i1, i2, i3, i4) - lam * B(i0, i1, i2, i3, i4));
      return {lam, std::sqrt(num2) / std::max(std::sqrt(nA), 1e-300),
              std::sqrt(nA), std::sqrt(den)};
    }

  } // gf2xchk_detail

  // one (system, wmax, Np-multiplier) cross-check run
  template<typename ctx_t>
  static void run_crosscheck(ctx_t &mpi_context, std::string const &mf_name,
                             double wmax, long np_mult) {
    imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_gf2xchk";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * np_mult, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    const long nbnd = mf->nbnd();

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");

    // a converged-ish G to evaluate both kernels on (identical input for both)
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                   1, false, 1e-9, true);
    (void)e_hf; (void)e_corr;
    mpi_context->comm.barrier();
    REQUIRE(mb_state.sG_tskij.has_value());

    // BARE RUNG: ZERO the dynamic part so the rung is W = Z + dW = Z (bare Coulomb).
    // (eval_Sigma_C requires dW to be PRESENT -- unlike Pi^C it has no absent-dW
    // fallback -- so rebuild the container and zero the block, rather than resetting
    // the optional. scf_loop frees dW at its tail when no vertex is attached.)
    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    mb_state.dW_qtPQ.value().local()() = cplx(0.0);

    auto &Sig = mb_state.sSigma_tskij.value();   // REFERENCE: the kernels write
                                                 // into MBState's own window
    const auto sh = Sig.local().shape();

    // ---- (1) Sigma^C with C = full band range, bare rung -----------------------------
    nda::array<cplx, 5> SigC(sh);
    {
      Sig.win().fence();
      if (mpi_context->node_comm.root()) Sig.local()() = cplx(0.0);
      Sig.win().fence();
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(0, nbnd), nbnd, "ignore_g0");
      vtx.eval_Sigma_C(mb_state, thc);
      mpi_context->comm.barrier();
      SigC() = Sig.local();
    }

    // ---- (2) GF2 direct + exchange, and (3) GF2 direct alone -------------------------
    nda::array<cplx, 5> SigDE(sh), SigD(sh);
    {
      solvers::gf2_t gf2(mf.get(), &ft, "ignore_g0", "gf2", "orb", "gf2", output);
      gf2.evaluate(mb_state, thc);
      mpi_context->comm.barrier();
      SigDE() = Sig.local();
    }
    {
      solvers::gf2_t gf2d(mf.get(), &ft, "ignore_g0", "gf2", "none", "gf2", output);
      gf2d.evaluate(mb_state, thc);
      mpi_context->comm.barrier();
      SigD() = Sig.local();
    }
    nda::array<cplx, 5> SigX(sh);
    SigX() = SigDE - SigD;

    auto f = gf2xchk_detail::fit_scale(SigC, SigX);
    app_log(1, "\n=== vertex vs GF2 bare second-order exchange (C = full, W = Z) ===\n"
               "  ||Sigma^C||_F        = {:.6e}\n"
               "  ||Sigma^GF2-exc||_F  = {:.6e}\n"
               "  best-fit Sigma^C = lambda * Sigma^GF2-exc:  lambda = {:.10f} {:+.10f}i\n"
               "  residual after the fit ||Sigma^C - lambda Sigma^exc|| / ||Sigma^C|| = {:.3e}\n"
               "  (nkpts = {}, beta = {})\n",
            f.normA, f.normB, f.lambda.real(), f.lambda.imag(), f.resid,
            mf->nkpts(), ft.beta());

    REQUIRE(f.normA > 1e-12);
    REQUIRE(f.normB > 1e-12);
    // TOPOLOGY: the two kernels must be proportional -- a shape mismatch (wrong
    // crossing / wrong index routing) shows here and cannot be absorbed by lambda.
    REQUIRE(f.resid < 1e-4);
    // ABSOLUTE SCALE AND SIGN: with C = full and W -> v, Phi_2^C IS the bare
    // second-order exchange functional, so lambda must be exactly +1.
    REQUIRE(std::abs(f.lambda.imag()) < 1e-6);
    REQUIRE(std::abs(f.lambda.real() - 1.0) < 1e-4);

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }

  // ====================================================================================
  // FAST case first: pyscf_h2_222 is the very system CoQui's GF2 is pinned against
  // (methods/GF2/tests/test_thc_gf2.cpp, CD-GF2 to 1e-8), and it is small enough that
  // Sigma^C over the FULL band range is cheap.
  TEST_CASE("vertex_gf2_exchange_crosscheck_h2", "[methods][vertex][gf2]") {
#ifndef ENABLE_DLR
    SUCCEED("skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    run_crosscheck(mpi_context, "pyscf_h2_222", 1.2, 10);
#endif
  }

  // ====================================================================================
  TEST_CASE("vertex_gf2_exchange_crosscheck", "[methods][vertex][gf2]") {
#ifndef ENABLE_DLR
    SUCCEED("skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    run_crosscheck(mpi_context, "qe_lih222", 6.0, 8);
#endif
  }

} // bdft_tests
