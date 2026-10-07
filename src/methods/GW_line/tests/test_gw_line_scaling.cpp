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
 * S7e (multi-node scaling) A/B tests on qe_lih222 (THC nIpts = 8 nbnd), [gw_line][s7e]:
 *  (a) host XV form of the propagator blocks (L diag(ph) R^dagger, COQUI_GWLINE_GT_XV = 1) vs the C(t) form (= 0) on random
 *      factorized poles: the three forms, both sectors, 11 times: <= 1e-13 relative;
 *  (b) the chain Pi -> W -> Sigma on KS poles: XV vs C(t) form (Pi, Sigma <= 1e-12), Sigma orbital contraction right-first
 *      vs left-first (COQUI_GWLINE_CONTRACT_RIGHT = 1 / 0, <= 1e-13), k-distributed Sigma (k_local, reduce-scatter) vs the
 *      replicated all_reduce path on the owned rows (<= 1e-13);
 *  (c) k_dist_t gather / scatter round trip (exact) and the time grids built on four ranks vs a local construction (bitwise).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/k_dist.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using namespace gw_line_test;

struct env_guard {
  std::string nm;
  env_guard(char const *n, char const *v) : nm(n) { setenv(n, v, 1); }
  ~env_guard() { unsetenv(nm.c_str()); }
};

void run_s7e(std::string const &fixture) {
  auto mpi   = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto mf    = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
  methods::thc_reader_t thc(mf, methods::make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                                                               1, 1024));
  const long nk = mf->nkpts(), nq = mf->nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  aux_grid_t grid(*mpi, Np);
  auto rel = [&](auto const &A, auto const &B) {
    const double d = comm.all_reduce_value(max_diff3(A, B), mpi3::max<>{});
    const double m = comm.all_reduce_value(max_abs3(B), mpi3::max<>{});
    return d / m;
  };

  // ---------------------------------------------------------------- (a) XV vs C(t) form of the G~ blocks
  {
    std::mt19937 gen(777);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<nda::array<double, 1>> e(nk);
    std::vector<nda::array<ComplexType, 2>> v(nk);
    const long Mp = 41, Mh = 23;
    for (long ik = 0; ik < nk; ++ik) {
      e[ik] = nda::array<double, 1>(Mp + Mh);
      v[ik] = nda::array<ComplexType, 2>(nb, Mp + Mh);
      for (long m = 0; m < Mp + Mh; ++m) {
        const double a = 5e-3 * std::pow(800.0, U(gen));
        e[ik](m)       = (m % 8 < 5) ? a : -a;
        for (long i = 0; i < nb; ++i) v[ik](i, m) = 0.3 * ComplexType(2.0 * U(gen) - 1.0, 2.0 * U(gen) - 1.0);
      }
    }
    auto pf = pole_data_t::from_lehmann(e, v);
    propagator_t<HOST_MEMORY> pxv(thc, grid), pc(thc, grid);
    {
      env_guard g("COQUI_GWLINE_GT_XV", "1");
      pxv.set_poles(pf);
    }
    {
      env_guard g("COQUI_GWLINE_GT_XV", "0");
      pc.set_poles(pf);
    }
    REQUIRE(pxv.n_xv == 2 * nk);
    REQUIRE(pc.n_xv == 0);
    double errG = 0.0;
    for (auto sec : {sector_t::particle, sector_t::hole}) {
      auto ray      = time_ray_t::for_spectrum(theta_t, 5e-3, 36.0, 1e-5, 3.0, 16, sec);
      const long nt = 11;
      nda::array<ComplexType, 1> t(nt), tall(ray.t(nda::range(0, ray.size())));
      for (long it = 0; it < nt; ++it) t(it) = tall(std::min(ray.size() - 1, it * (ray.size() / nt)));
      for (auto form : {gtilde_form_t::plain, gtilde_form_t::transposed, gtilde_form_t::adjoint_conj_t})
        for (long ik = 0; ik < nk; ++ik) {
          memory::array<HOST_MEMORY, ComplexType, 3> Gx(nt, grid.nP, grid.nQ), Gc(nt, grid.nP, grid.nQ);
          pxv.build(ik, t, sec, form, Gx());
          pc.build(ik, t, sec, form, Gc());
          errG = std::max(errG, rel(Gx, Gc));
        }
    }
    app_log(1, "[s7e] (a) {}: G~ blocks XV form vs C(t) form (3 forms, both sectors, 11 times, {} + {} poles per k): {:.2e}; "
               "XV factors {:.3f} MB per rank",
            fixture, Mp, Mh, errG, pxv.xv_bytes() / 1048576.0);
    REQUIRE(errG <= 1e-13);
  }

  // ---------------------------------------------------------------- (b) chain variants on KS poles
  {
    nda::array<double, 2> eig(nk, nb);
    double homo = -1e300, lumo = 1e300;
    const long nocc = long(std::llround(double(mf->nelec()) / 2.0));
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        eig(ik, n) = mf->eigval()(0, ik, n);
        if (n < nocc) homo = std::max(homo, eig(ik, n));
        else lumo = std::min(lumo, eig(ik, n));
      }
    auto pd = pole_data_t::from_ks(eig, 0.5 * (homo + lumo));
    bosonic_basis_t basis(theta, 4.0, 1e-10, 0.5 * (lumo - homo));
    auto const &zb = basis.zeta_nodes;
    auto fz        = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
    dyson_layout_t lay(*mpi, nq, zb.size(), Np);
    utils::TimerManager T;
    coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), T);
    numerics::line_dlr::time_id_opts_t opts;
    line_time_grids_t G(pd, basis.nu, theta_t, 1e-10, opts, zb, fz, comm);
    propagator_t<HOST_MEMORY> prop(thc, grid);
    memory::array<HOST_MEMORY, ComplexType, 4> Pi_c, Pi_x, w;
    {
      env_guard g("COQUI_GWLINE_GT_XV", "0");
      polarization<HOST_MEMORY>(prop, pd, *mf, grid, zb, G.pi_p, G.pi_h, 8, Pi_c, T);
    }
    {
      env_guard g("COQUI_GWLINE_GT_XV", "1");
      polarization<HOST_MEMORY>(prop, pd, *mf, grid, zb, G.pi_p, G.pi_h, 8, Pi_x, T);
    }
    const double dPi = rel(Pi_x, Pi_c);
    // q groups of <= 3 (3, 3, 2 for N_q = 8 on lih222; pair-closed {q, -q} units on lih223): Pi rows and the residues of a
    // grouped Pi -> W stage vs all q at once
    double dPg = 0.0, dwg = 0.0, dwraw = 0.0;
    {
      const q_groups_t qg(nq, 3, qminus_list(*mf));
      coulomb_blocks_t<HOST_MEMORY> Zg(thc, grid, qg.dyson_q_list(comm.size(), comm.rank(), zb.size(), Np), T);
      memory::array<HOST_MEMORY, ComplexType, 4> Pg, wg, Pa(Pi_c), wa;
      screened_interaction<HOST_MEMORY>(Pa, Zb, basis, grid, *mpi, wa, T);
      for (long ig = 0; ig < qg.n; ++ig) {
        {
          env_guard e("COQUI_GWLINE_GT_XV", "0");
          polarization<HOST_MEMORY>(prop, pd, *mf, grid, zb, G.pi_p, G.pi_h, 8, Pg, T, sector_t::both, qg.rows(ig));
        }
        REQUIRE(Pg.extent(0) == qg.size(ig));
        double d = 0.0;
        for (long r = 0; r < qg.size(ig); ++r)
          d = std::max(d, max_diff3(nda::array<ComplexType, 3>(Pg(r, nda::ellipsis{})),
                                    nda::array<ComplexType, 3>(Pi_c(qg.rows(ig)[r], nda::ellipsis{}))));
        dPg = std::max(dPg, d);
        screened_interaction<HOST_MEMORY>(Pg, Zg, basis, grid, *mpi, wg, T, nullptr, qg.rows(ig), false);
      }
      dPg = comm.all_reduce_value(dPg, mpi3::max<>{}) / comm.all_reduce_value(max_abs3(Pi_c), mpi3::max<>{});
      // the residues compared as pole sums at the nodes, both sectors (the raw residues are determined only up to the
      // near-threshold singular directions of the stacked kernel, ~1e-5 with MKL when the fit's column chunking differs:
      // info dwraw; perf 7.1)
      dwraw   = rel(wg, wa);
      auto qm = mf->qminus();
      for (long q = 0; q < nq; ++q)
        for (auto s : {sector_t::particle, sector_t::hole}) {
          memory::array<HOST_MEMORY, ComplexType, 3> A(zb.size(), grid.nP, grid.nQ), B(zb.size(), grid.nP, grid.nQ);
          eval_poles<HOST_MEMORY>(wg, basis, q, qm(q), zb, s, s == sector_t::hole, A());
          eval_poles<HOST_MEMORY>(wa, basis, q, qm(q), zb, s, s == sector_t::hole, B());
          dwg = std::max(dwg, rel(A, B));
        }
    }
    screened_interaction<HOST_MEMORY>(Pi_c, Zb, basis, grid, *mpi, w, T);
    auto sigma = [&](char const *xv, char const *right, bool k_local, sector_t s) {
      env_guard g1("COQUI_GWLINE_GT_XV", xv), g2("COQUI_GWLINE_CONTRACT_RIGHT", right);
      nda::array<ComplexType, 4> S;
      self_energy<HOST_MEMORY>(prop, pd, w, basis, *mf, grid, *mpi, fz, G.sig_p, G.sig_h, 8, S, T, s, k_local);
      return S;
    };
    const k_dist_t kd(nk, comm);
    double dxv = 0.0, dord = 0.0, dloc = 0.0, dgrp = 0.0;
    for (auto s : {sector_t::particle, sector_t::hole}) {
      auto S0 = sigma("0", "0", false, s);   // C(t), left-first, replicated (the pre-S7e path)
      auto S1 = sigma("1", "0", false, s);
      auto S2 = sigma("0", "1", false, s);
      auto S3 = sigma("-1", "-1", true, s);  // the defaults with the k-distributed result
      nda::array<ComplexType, 4> S4;         // host-resident residues streamed in q groups of 3 (S7e device fallback)
      {
        env_guard g1("COQUI_GWLINE_GT_XV", "0"), g2("COQUI_GWLINE_CONTRACT_RIGHT", "0");
        memory::array<HOST_MEMORY, ComplexType, 4> wh(w), wnone;
        self_energy<HOST_MEMORY>(prop, pd, wnone, basis, *mf, grid, *mpi, fz, G.sig_p, G.sig_h, 8, S4, T, s, false, &wh, 3);
      }
      const double m = max_abs3(S0);
      dgrp = std::max(dgrp, max_diff3(S4, S0) / m);
      dxv  = std::max(dxv, max_diff3(S1, S0) / m);
      dord = std::max(dord, max_diff3(S2, S0) / m);
      REQUIRE(S3.extent(0) == kd.nloc());
      double d = 0.0;
      for (long l = 0; l < kd.nloc(); ++l)
        d = std::max(d, max_diff3(nda::array<ComplexType, 3>(S3(l, nda::ellipsis{})),
                                  nda::array<ComplexType, 3>(S0(kd.global(l, kd.rank), nda::ellipsis{}))));
      dloc = std::max(dloc, comm.all_reduce_value(d, mpi3::max<>{}) / m);
    }
    app_log(1, "[s7e] (b) {}: q groups of 3: Pi rows {:.2e}, residues as pole sums at the nodes {:.2e} (raw residues {:.1e}, info)",
            fixture, dPg, dwg, dwraw);
    REQUIRE(dPg <= 1e-14);
    REQUIRE(dwg <= 1e-12);
    app_log(1, "[s7e] (b) {}: Pi XV vs C(t) {:.2e}; Sigma XV vs C(t) {:.2e}, right- vs left-first contraction {:.2e}, k-distributed "
               "(reduce-scatter, owned rows) vs replicated {:.2e}; grid {} x {}, block {} x {}",
            fixture, dPi, dxv, dord, dloc, grid.np_P, grid.np_Q, grid.nP, grid.nQ);
    REQUIRE(dPi <= 1e-12);
    REQUIRE(dxv <= 1e-12);
    REQUIRE(dord <= 1e-13);
    REQUIRE(dloc <= 1e-13);
    app_log(1, "[s7e] (b) {}: Sigma with host-resident residues in q groups of 3 vs resident (all q) {:.2e}", fixture, dgrp);
    REQUIRE(dgrp <= 1e-13);
  }

  // ---------------------------------------------------------------- (c) k_dist round trip, time grids on four ranks
  {
    const k_dist_t kd(nk, comm);
    nda::array<ComplexType, 4> full(comm.root() ? nk : 0, 3, 2, 2);
    for (long k = 0; k < full.extent(0); ++k)
      for (long a = 0; a < 3; ++a)
        for (long i = 0; i < 2; ++i)
          for (long j = 0; j < 2; ++j) full(k, a, i, j) = ComplexType(k + 0.1 * a, i - 0.5 * j);
    auto loc = kd_scatter_full(comm, kd, full);
    REQUIRE(loc.extent(0) == kd.nloc());
    for (long l = 0; l < kd.nloc(); ++l) REQUIRE(loc(l, 2, 1, 0) == ComplexType(kd.global(l, kd.rank) + 0.2, 1.0));
    auto back = kd_gather_full(comm, kd, loc);
    if (comm.root()) REQUIRE(nda::max_element(nda::abs(back - full)) == 0.0);

    nda::array<double, 2> eig(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) eig(ik, n) = mf->eigval()(0, ik, n);
    const long nocc = long(std::llround(double(mf->nelec()) / 2.0));
    double homo = -1e300, lumo = 1e300;
    for (long ik = 0; ik < nk; ++ik) {
      homo = std::max(homo, eig(ik, nocc - 1));
      lumo = std::min(lumo, eig(ik, nocc));
    }
    auto pd = pole_data_t::from_ks(eig, 0.5 * (homo + lumo));
    bosonic_basis_t basis(theta, 4.0, 1e-10, 0.5 * (lumo - homo));
    auto fz = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
    numerics::line_dlr::time_id_opts_t opts;
    line_time_grids_t G(pd, basis.nu, theta_t, 1e-10, opts, basis.zeta_nodes, fz, comm);
    auto self = comm.split(comm.rank(), 0);   // size-1 communicator
    line_time_grids_t L(pd, basis.nu, theta_t, 1e-10, opts, basis.zeta_nodes, fz, self);   // every grid built locally
    double dt = 0.0;
    for (auto [a, b] : {std::pair{&G.pi_p, &L.pi_p}, std::pair{&G.pi_h, &L.pi_h}, std::pair{&G.sig_p, &L.sig_p},
                        std::pair{&G.sig_h, &L.sig_h}}) {
      REQUIRE(a->size() == b->size());
      REQUIRE(a->ls_rank == b->ls_rank);
      dt = std::max({dt, nda::max_element(nda::abs(a->t - b->t)), nda::max_element(nda::abs(a->Vs - b->Vs)),
                     nda::max_element(nda::abs(a->Uc - b->Uc))});
      REQUIRE(a->phase == b->phase);
      REQUIRE(a->Emin == b->Emin);
    }
    app_log(1, "[s7e] (c) {}: k_dist scatter/gather exact; time grids built on {} rank(s) vs locally: max diff {:.1e} (nodes {} {} {} {})",
            fixture, std::min(4L, long(comm.size())), dt, G.pi_p.size(), G.pi_h.size(), G.sig_p.size(), G.sig_h.size());
    REQUIRE(dt == 0.0);
  }
}

} // namespace

TEST_CASE("gw_line_s7e_lih222", "[gw_line][s7e]") { run_s7e("qe_lih222"); }

// q != -q mesh (2x2x3): pair-closed q groups of <= 3, host-resident residue groups with the hole rows of -q
TEST_CASE("gw_line_s7e_lih223", "[gw_line][s7e]") { run_s7e("qe_lih223"); }
