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
 * S7c of notes/line_gw_cpp_plan.md: factorized (Lehmann) pole residues coef_m = v_m v_m^dagger in the propagators.
 *
 * [lehmann] on qe_lih222 (THC nIpts = 8 nbnd):
 *  (a) random factorized poles (37 particle + 29 hole per k, random complex v, |e| in [5e-3, 4] Ha) vs the SAME poles in the
 *      matrix-coefficient form (pole_data_t::to_coefficients): C(t) and the G~ blocks of all three forms (plain, transposed,
 *      adjoint_conj_t) at 11 complex times of both rays: <= 1e-13 relative (max norm); density() <= 1e-14.
 *  (b) KS poles (from_ks, factorized unit vectors) vs their coefficient form: Pi per sector at the bosonic nodes and Sigma
 *      per sector at the dense fermionic nodes (ID grids, eps_t 1e-10, the W of each chain): <= 1e-12 relative. The V1/V3
 *      tests run on the factorized KS poles unchanged (from_ks is factorized since S7c).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
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
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using namespace gw_line_test;

void run_lehmann_kernels(std::string const &fixture) {
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

  // ---------------------------------------------------------------- (a) random factorized poles vs coefficient form
  {
    std::mt19937 gen(4242);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<nda::array<double, 1>> e(nk);
    std::vector<nda::array<ComplexType, 2>> v(nk);
    const long Mp = 37, Mh = 29;
    for (long ik = 0; ik < nk; ++ik) {
      e[ik] = nda::array<double, 1>(Mp + Mh);
      v[ik] = nda::array<ComplexType, 2>(nb, Mp + Mh);
      for (long m = 0; m < Mp + Mh; ++m) {
        const double a = 5e-3 * std::pow(800.0, U(gen));   // log-uniform |e| in [5e-3, 4]
        e[ik](m)       = (m % 7 < 4) ? a : -a;
        for (long i = 0; i < nb; ++i) v[ik](i, m) = 0.3 * ComplexType(2.0 * U(gen) - 1.0, 2.0 * U(gen) - 1.0);
      }
    }
    auto pf = pole_data_t::from_lehmann(e, v);
    auto pc = pf.to_coefficients();
    REQUIRE(pf.is_factorized());
    REQUIRE(not pc.is_factorized());
    double dD = 0.0, Dm = 0.0;
    {
      auto Df = density_matrix(pf), Dc = density_matrix(pc);
      dD = max_diff3(Df, Dc);
      Dm = max_abs3(Dc);
    }
    propagator_t<HOST_MEMORY> prf(thc, grid), prc(thc, grid);
    prf.set_poles(pf);
    prc.set_poles(pc);
    double errC = 0.0, errG = 0.0;
    for (auto sec : {sector_t::particle, sector_t::hole}) {
      auto ray = time_ray_t::for_spectrum(theta_t, 5e-3, 36.0, 1e-5, 3.0, 16, sec);
      const long nt = 11;
      nda::array<ComplexType, 1> t(nt);
      nda::array<ComplexType, 1> tall(ray.t(nda::range(0, ray.size())));
      for (long it = 0; it < nt; ++it) t(it) = tall(std::min(ray.size() - 1, it * (ray.size() / nt)));
      for (auto form : {gtilde_form_t::plain, gtilde_form_t::transposed, gtilde_form_t::adjoint_conj_t}) {
        for (long ik = 0; ik < nk; ++ik) {
          memory::array<HOST_MEMORY, ComplexType, 3> Gf(nt, grid.nP, grid.nQ), Gc(nt, grid.nP, grid.nQ);
          prf.build(ik, t, sec, form, Gf());
          nda::array<ComplexType, 3> Cf(prf.s_C.view<3>({nt, nb, nb}));
          prc.build(ik, t, sec, form, Gc());
          nda::array<ComplexType, 3> Cc(prc.s_C.view<3>({nt, nb, nb}));
          errC = std::max(errC, max_diff3(Cf, Cc) / max_abs3(Cc));
          errG = std::max(errG, rel(Gf, Gc));
        }
      }
    }
    errC = comm.all_reduce_value(errC, mpi3::max<>{});
    app_log(1, "[lehmann] (a) {} random factorized poles ({} + {} per k) vs coefficient form: C(t) {:.2e}, G~ blocks (3 forms, "
               "both sectors, 11 times) {:.2e}, density {:.2e}; residue storage {:.3f} vs {:.3f} MB",
            fixture, Mp, Mh, errC, errG, dD / Dm, pf.residue_bytes() / 1048576.0, pc.residue_bytes() / 1048576.0);
    REQUIRE(errC <= 1e-13);
    REQUIRE(errG <= 1e-13);
    REQUIRE(dD <= 1e-14 * Dm);
  }

  // ---------------------------------------------------------------- (b) Pi and Sigma on factorized KS poles
  {
    nda::array<double, 2> eig(nk, nb);
    double homo = -1e300, lumo = 1e300;
    const long nocc = long(std::llround(double(mf->nelec()) / 2.0));
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        eig(ik, n) = mf->eigval()(0, ik, n);
        (n < nocc ? homo : lumo) = (n < nocc) ? std::max(homo, eig(ik, n)) : std::min(lumo, eig(ik, n));
      }
    const double mu = 0.5 * (homo + lumo);
    auto pf = pole_data_t::from_ks(eig, mu);
    auto pc = pf.to_coefficients();
    REQUIRE(pf.is_factorized());
    bosonic_basis_t basis(theta, 4.0, 1e-10, 0.5 * (lumo - homo));
    auto const &zb = basis.zeta_nodes;
    auto fz        = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
    dyson_layout_t lay(*mpi, nq, zb.size(), Np);
    utils::TimerManager T;
    coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), T);
    numerics::line_dlr::time_id_opts_t opts;
    line_time_grids_t G(pf, basis.nu, theta_t, 1e-10, opts, zb, fz, comm);
    propagator_t<HOST_MEMORY> prop(thc, grid);
    struct chain_t {
      memory::array<HOST_MEMORY, ComplexType, 4> Pp, Ph, w;
      nda::array<ComplexType, 4> Sp, Sh;
    };
    auto chain = [&](pole_data_t const &pd) {
      chain_t c;
      polarization<HOST_MEMORY>(prop, pd, *mf, grid, zb, G.pi_p, G.pi_h, 8, c.Pp, T, sector_t::particle);
      polarization<HOST_MEMORY>(prop, pd, *mf, grid, zb, G.pi_p, G.pi_h, 8, c.Ph, T, sector_t::hole);
      memory::array<HOST_MEMORY, ComplexType, 4> Pi = c.Pp;
      Pi += c.Ph;
      screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, *mpi, c.w, T);
      self_energy<HOST_MEMORY>(prop, pd, c.w, basis, *mf, grid, *mpi, fz, G.sig_p, G.sig_h, 8, c.Sp, T, sector_t::particle);
      self_energy<HOST_MEMORY>(prop, pd, c.w, basis, *mf, grid, *mpi, fz, G.sig_p, G.sig_h, 8, c.Sh, T, sector_t::hole);
      return c;
    };
    auto F = chain(pf);
    auto C = chain(pc);
    const double pp = rel(F.Pp, C.Pp), ph = rel(F.Ph, C.Ph);
    const double sp = max_diff3(F.Sp, C.Sp) / max_abs3(C.Sp), sh = max_diff3(F.Sh, C.Sh) / max_abs3(C.Sh);
    app_log(1, "[lehmann] (b) {} KS poles factorized vs coefficient form (ID 1e-10, nodes Pi {}+{} Sigma {}+{}): Pi^> {:.2e} "
               "Pi^< {:.2e} Sigma^> {:.2e} Sigma^< {:.2e}",
            fixture, G.pi_p.size(), G.pi_h.size(), G.sig_p.size(), G.sig_h.size(), pp, ph, sp, sh);
    REQUIRE(pp <= 1e-12);
    REQUIRE(ph <= 1e-12);
    REQUIRE(sp <= 1e-12);
    REQUIRE(sh <= 1e-12);
  }
}

} // namespace

TEST_CASE("gw_line_lehmann_kernels_lih222", "[gw_line][lehmann]") { run_lehmann_kernels("qe_lih222"); }
