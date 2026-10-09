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
 * Performance campaign 7.3 (notes/line_gw_cpp_plan.md section 7.3): the IBZ (point-group + time-reversal) reduction of the
 * line GW kernels on symmetric mean fields.
 *
 *  [.ibz_tables] (hidden) the symmetry tables of the symmetric fixtures (and of COQUI_GWLINE_IBZ_MF = "outdir|prefix", h5):
 *                IBZ sizes, q classes, trev, the -q closure of the IBZ q.
 *  [ibz][V0]     F = V_H + Sigma_x of the KS density at the IBZ k (static_ibz.hpp: class sums, D matrices, Z(-q) = conj Z(q))
 *                vs CoQui's symmetric hf_t (thc_hf.icc) on lih222_sym / lih223_sym / lih223_inv; the trivial tables on the
 *                nosym lih222 / lih223 vs the full-BZ hartree_exchange.
 *  [ibz][sigma]  (A) trivial tables (nosym lih222 / lih223 / si211): Pi of the row list, W, and self_energy_ibz vs the full-BZ
 *                self_energy (<= 1e-12); (B) symmetric lih222_sym / lih223_sym / lih223_inv with random NON-diagonal poles:
 *                the propagator's C(t) sharing (transposes / conj of the time-reversed k) vs per-k builds of the unfolded
 *                poles: Pi rows and Sigma (<= 1e-13); (C) [.ibz_phys] sym vs nosym fixture of the same system: eigenvalues
 *                of Sigma(k, zeta) at the IBZ k (report: the THC / mean-field difference of the two fixtures).
 *  [ibz][scf]    the driver on lih222_sym / lih223_sym (IBZ) vs lih222 / lih223 (nosym, full BZ), 3 iterations: mu, QP gap per
 *                iteration (gated at 1e-4 Ha on iterations 1-2: the THC difference of the two fixtures, 1e-5 at Np 128; iteration 3
 *                reported only: closure basin noise); gygi on lih223 (eps_inf, iteration 1).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "mean_field/default_MF.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/hf_t.h"
#include "utilities/Timer.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/static_ibz.hpp"
#include "methods/GW_line/ibz.hpp"
#include "methods/GW_line/driver.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/self_energy_ibz.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/line_dlr_utils.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::sector_t;

struct fix_t {
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nkI = 0, nb = 0, Np = 0;
  double mu = 0.0;
  nda::array<double, 2> eigI;   ///< (nk_ibz, nb) KS energies at the IBZ k
};

fix_t make_fix(std::string const &name, long nI_factor = 8) {
  auto &mpi = utils::make_unit_test_mpi_context();
  fix_t f;
  f.mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, name));
  f.thc = std::make_unique<methods::thc_reader_t>(
      f.mf, methods::make_thc_reader_ptree(f.mf->nbnd() * nI_factor, "", "incore", "", "bdft", 1e-10, f.mf->ecutrho(), 1, 1024));
  f.nk  = f.mf->nkpts();
  f.nkI = f.mf->nkpts_ibz();
  f.nb  = f.mf->nbnd();
  f.Np  = f.thc->Np();
  const long nocc = long(std::llround(f.mf->nelec() / 2.0));
  double homo = -1e300, lumo = 1e300;
  f.eigI = nda::array<double, 2>(f.nkI, f.nb);
  for (long k = 0; k < f.nkI; ++k)
    for (long n = 0; n < f.nb; ++n) {
      f.eigI(k, n) = f.mf->eigval()(0, k, n);
      if (n < nocc) homo = std::max(homo, f.eigI(k, n));
      else lumo = std::min(lumo, f.eigI(k, n));
    }
  f.mu = 0.5 * (homo + lumo);
  return f;
}

void print_tables(mf::MF &mf, std::string const &name) {
  const long nk = mf.nkpts(), nkI = mf.nkpts_ibz(), nq = mf.nqpts(), nqI = mf.nqpts_ibz();
  auto qsym = mf.qsymms();
  auto nqs  = mf.nq_per_s();
  auto qm   = mf.qminus();
  auto qtr  = mf.qp_trev();
  auto q2i  = mf.qp_to_ibz();
  auto ktr  = mf.kp_trev();
  long ntr_q = 0;
  for (long q = 0; q < nq; ++q) ntr_q += qtr(q) ? 1 : 0;
  long ntr_k = 0;
  for (long k = 0; k < nk; ++k) ntr_k += ktr(k) ? 1 : 0;
  std::set<long> R;
  long selfp = 0, minus_in = 0;
  for (long q = 0; q < nqI; ++q) {
    R.insert(q);
    R.insert(qm(q));
    if (qm(q) == q) ++selfp;
    else if (qm(q) < nqI) ++minus_in;
  }
  app_log(1, "IBZTAB {}: nk {} nk_ibz {} nq {} nq_ibz {} nsym(list) {} qsymms {} trev k {} trev q {} (pairs {}); IBZ q self-inverse {}, "
             "-q also IBZ {}, |IBZ u -IBZ| = {}",
          name, nk, nkI, nq, nqI, mf.symm_list().size(), qsym.size(), ntr_k, ntr_q, mf.nkpts_trev_pairs(), selfp, minus_in, R.size());
  std::string s;
  for (long i = 0; i < qsym.size(); ++i) s += std::to_string(qsym(i)) + ":" + std::to_string(nqs(i)) + " ";
  app_log(1, "IBZTAB {}: qsymms:nq_per_s  {}", name, s);
  s.clear();
  for (long q = 0; q < nq; ++q) s += std::to_string(q2i(q)) + (qtr(q) ? "t " : " ");
  app_log(1, "IBZTAB {}: qp_to_ibz {}", name, s);
  // classes per IBZ k: distinct ks_to_k(isym, k)
  auto ks = mf.ks_to_k();
  for (long k = 0; k < nkI; ++k) {
    std::set<long> tg;
    for (long is = 0; is < qsym.size(); ++is) tg.insert(ks(is, k));
    app_log(1, "IBZTAB {}: k {} star targets {}", name, k, tg.size());
  }
}

} // namespace

TEST_CASE("gw_line_ibz_tables", "[.ibz_tables][gw_line]") {
  auto &mpi = utils::make_unit_test_mpi_context();
  for (std::string name : {"qe_lih222_sym", "qe_lih223_sym", "qe_lih223_inv", "qe_lih222", "qe_si211"}) {
    auto mf = mf::default_MF(mpi, name);
    print_tables(mf, name);
  }
  if (char const *e = std::getenv("COQUI_GWLINE_IBZ_MF")) {
    std::string v(e);
    const auto p = v.find('|');
    REQUIRE(p != std::string::npos);
    auto mf = mf::default_MF(mpi, mf::qe_source, v.substr(0, p), v.substr(p + 1), mf::h5_input_type);
    print_tables(mf, v.substr(p + 1));
  }
}

struct env_scope_t {
  std::string nm, old;
  bool had = false;
  env_scope_t(char const *n, char const *v) : nm(n) {
    if (char const *o = std::getenv(n)) { had = true; old = o; }
    ::setenv(n, v, 1);
  }
  ~env_scope_t() { if (had) ::setenv(nm.c_str(), old.c_str(), 1); else ::unsetenv(nm.c_str()); }
};

template <typename A, typename B> double mdiff(A const &a, B const &b) {
  double d = 0.0;
  for (long i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
  return d;
}
template <typename A> double mabs(A const &a) {
  double d = 0.0;
  for (long i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a.data()[i]));
  return d;
}

/// one Pi -> W -> Sigma pass on the rows R of `ibz` (KS or given IBZ poles); outputs Sigma^> / Sigma^< at the IBZ k
struct pass_out_t {
  memory::array<HOST_MEMORY, ComplexType, 4> Pi, w, Wn;
  nda::array<ComplexType, 4> Sp, Sh;
};
pass_out_t ibz_pass(fix_t &f, ibz_t const &ibz, pole_data_t const &polesI, double gap, bool use_full_sigma = false) {
  auto &mpi = *utils::make_unit_test_mpi_context();
  auto &mf  = *f.mf;
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  numerics::line_dlr::bosonic_basis_t basis(theta, 4.0, 1e-10, 0.5 * gap);
  const double smax = 30.0 / (polesI.emin() * std::sin(theta_t));
  numerics::line_dlr::time_ray_t ray_p(theta_t, smax, 1e-5, 1.5, 12, sector_t::particle),
      ray_h(theta_t, smax, 1e-5, 1.5, 12, sector_t::hole);
  aux_grid_t grid(mpi, f.Np);
  utils::TimerManager T;
  propagator_t<HOST_MEMORY> prop(*f.thc, grid);
  prop.set_ibz(&ibz);
  pass_out_t o;
  polarization<HOST_MEMORY>(prop, polesI, mf, grid, basis.zeta_nodes, ray_p, ray_h, 8, o.Pi, T, sector_t::both, ibz.rows);
  q_groups_t qg(ibz.rows, ibz.nrows(), ibz.qminus);
  coulomb_blocks_t<HOST_MEMORY> Zb(*f.thc, grid, qg.dyson_q_list(mpi.comm.size(), mpi.comm.rank(), basis.zeta_nodes.size(), f.Np), T);
  auto Pc = o.Pi;
  screened_interaction<HOST_MEMORY>(Pc, Zb, basis, grid, mpi, o.w, T, &o.Wn, ibz.rows, true);
  auto fz = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 30);
  if (use_full_sigma)
    self_energy<HOST_MEMORY>(prop, polesI, o.w, basis, mf, grid, mpi, fz, ray_p, ray_h, 8, o.Sp, T, sector_t::both, false, nullptr,
                             0, &o.Sh);
  else
    self_energy_ibz<HOST_MEMORY>(prop, polesI, o.w, basis, mf, ibz, grid, mpi.comm, fz, ray_p, ray_h, 8, o.Sp, T, sector_t::both,
                                 false, &o.Sh);
  return o;
}

/// IBZ poles from H = diag(eig) + a Hermitian perturbation (non-diagonal Lehmann vectors), same on every rank
pole_data_t random_poles(fix_t const &f, double amp, unsigned seed) {
  nda::array<ComplexType, 3> H(f.nkI, f.nb, f.nb);
  H() = ComplexType(0.0);
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> N01;
  for (long k = 0; k < f.nkI; ++k)
    for (long i = 0; i < f.nb; ++i) {
      H(k, i, i) = f.eigI(k, i);
      for (long j = 0; j < i; ++j) {
        const ComplexType x = amp * ComplexType(N01(gen), N01(gen));
        H(k, i, j) = x;
        H(k, j, i) = std::conj(x);
      }
    }
  return pole_data_t::from_hamiltonian(H, f.mu);
}

TEST_CASE("gw_line_ibz_sigma", "[ibz][sigma][gw_line]") {
  auto &mpi = *utils::make_unit_test_mpi_context();
  // (A) trivial tables: the IBZ kernels on a nosym mesh = the full-BZ kernels
  for (std::string name : {"qe_lih222", "qe_lih223", "qe_si211"}) {
    auto f = make_fix(name);
    ibz_t ibz(*f.mf, f.nb);
    auto poles = pole_data_t::from_ks(f.eigI, f.mu);
    double gap = 1e300;
    for (long k = 0; k < f.nkI; ++k)
      for (long n = 0; n < f.nb; ++n) gap = std::min(gap, std::abs(f.eigI(k, n) - f.mu));
    auto a = ibz_pass(f, ibz, poles, 2.0 * gap, false);
    auto b = ibz_pass(f, ibz, poles, 2.0 * gap, true);
    const double es = std::max(mdiff(a.Sp, b.Sp) / mabs(b.Sp), mdiff(a.Sh, b.Sh) / mabs(b.Sh));
    app_log(1, "  [ibz][sigma](A) {} ({} ranks): trivial tables, self_energy_ibz vs self_energy: Sigma rel {:.2e}", name,
            mpi.comm.size(), es);
    REQUIRE(es <= 1e-12);
  }
  // (B) symmetric meshes, random non-diagonal poles: C(t) sharing vs per-k builds of the unfolded poles
  for (std::string name : {"qe_lih222_sym", "qe_lih223_sym", "qe_lih223_inv"}) {
    auto f = make_fix(name);
    ibz_t ibz(*f.mf, f.nb);
    auto poles = random_poles(f, 0.01, 5);
    double gap = 1e300;
    for (long k = 0; k < f.nkI; ++k)
      for (long n = 0; n < f.nb; ++n) gap = std::min(gap, std::abs(f.eigI(k, n) - f.mu));
    pass_out_t a, b, c;
    {   // C(t) form forced (on few ranks the XV form wins the flop test and nothing would be shared)
      env_scope_t x("COQUI_GWLINE_GT_XV", "0");
      a = ibz_pass(f, ibz, poles, 2.0 * gap, false);
      env_scope_t e("COQUI_GWLINE_IBZ_CSHARE", "0");
      b = ibz_pass(f, ibz, poles, 2.0 * gap, false);
    }
    {
      env_scope_t x("COQUI_GWLINE_GT_XV", "1");
      c = ibz_pass(f, ibz, poles, 2.0 * gap, false);
    }
    const double ep  = mdiff(a.Pi, b.Pi) / std::max(1e-300, mabs(b.Pi));
    const double epx = mdiff(a.Pi, c.Pi) / std::max(1e-300, mabs(c.Pi));
    const double es  = std::max(mdiff(a.Sp, b.Sp) / mabs(b.Sp), mdiff(a.Sh, b.Sh) / mabs(b.Sh));
    const double esx = std::max(mdiff(a.Sp, c.Sp) / mabs(c.Sp), mdiff(a.Sh, c.Sh) / mabs(c.Sh));
    // W(-q, -conj z) = conj W(q, z) between the rows R (exact with Z(-q) = conj Z(q)): the mirror rows of the Dyson output
    app_log(1, "  [ibz][sigma](B) {} ({} ranks): {} rows; C(t) shared vs per-k C(t): Pi rel {:.2e}, Sigma rel {:.2e}; vs XV builds: "
               "Pi {:.2e}, Sigma {:.2e} (Sigma: rounding through the W pair-fit pseudo-inverse); max|Sigma| {:.3e}",
            name, mpi.comm.size(), ibz.nrows(), ep, es, epx, esx, std::max(mabs(a.Sp), mabs(a.Sh)));
    REQUIRE(ep <= 1e-14);
    REQUIRE(epx <= 1e-14);
    REQUIRE(es <= 1e-11);
    REQUIRE(esx <= 1e-11);
  }
}

TEST_CASE("gw_line_ibz_static", "[ibz][V0][gw_line]") {
  auto &mpi = *utils::make_unit_test_mpi_context();
  auto all  = nda::range::all;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  for (std::string name : {"qe_lih222_sym", "qe_lih223_sym", "qe_lih223_inv", "qe_lih222", "qe_lih223"}) {
    auto f = make_fix(name);
    ibz_t ibz(*f.mf, f.nb);
    ibz.log(2);
    aux_grid_t grid(mpi, f.Np);
    utils::TimerManager Timer;
    propagator_t<HOST_MEMORY> prop(*f.thc, grid);
    prop.set_ibz(&ibz);
    coulomb_blocks_t<HOST_MEMORY> Zb(*f.thc, grid, std::vector<long>{}, Timer);
    auto poles = pole_data_t::from_ks(f.eigI, f.mu);
    auto D     = density_matrix(poles);
    {   // + a small Hermitian complex perturbation (same on every rank): exercises the conj of the time-reversed k
      std::mt19937_64 gen(17);
      std::normal_distribution<double> N01;
      for (long k = 0; k < f.nkI; ++k)
        for (long i = 0; i < f.nb; ++i)
          for (long j = 0; j <= i; ++j) {
            const ComplexType x = 0.01 * ComplexType(N01(gen), i == j ? 0.0 : N01(gen));
            D(k, i, j) += x;
            if (j != i) D(k, j, i) += std::conj(x);
          }
    }
    nda::array<ComplexType, 3> F;
    hartree_exchange_ibz<HOST_MEMORY>(prop, Zb, D, *f.mf, ibz, grid, mpi.comm, F, Timer);
    // CoQui hf_t (symmetric when the MF is)
    methods::solvers::hf_t hf("ignore_g0");
    auto sF = math::shm::make_shared_array<Array_view_4D_t>(mpi, {1, f.nkI, f.nb, f.nb});
    nda::array<ComplexType, 4> Dm(1, f.nkI, f.nb, f.nb), S(1, f.nkI, f.nb, f.nb);
    Dm(0, all, all, all) = D;
    S()                  = ComplexType(0.0);
    for (long k = 0; k < f.nkI; ++k)
      for (long i = 0; i < f.nb; ++i) S(0, k, i, i) = 1.0;
    hf.evaluate(sF, Dm, *f.thc, S, true, true);
    nda::array<ComplexType, 3> Fc(sF.local()(0, all, all, all));
    double err = 0.0, mx = 0.0;
    for (long k = 0; k < f.nkI; ++k)
      for (long i = 0; i < f.nb; ++i)
        for (long j = 0; j < f.nb; ++j) {
          err = std::max(err, std::abs(F(k, i, j) - Fc(k, i, j)));
          mx  = std::max(mx, std::abs(Fc(k, i, j)));
        }
    double err_full = -1.0;
    if (not ibz.active) {   // trivial tables: the full-BZ hartree_exchange
      nda::array<ComplexType, 3> F0;
      hartree_exchange<HOST_MEMORY>(prop, Zb, D, *f.mf, grid, mpi.comm, F0, Timer);
      err_full = nda::max_element(nda::abs(F0 - F));
    }
    app_log(1, "  [ibz][V0] {} ({} ranks): nk {} nk_ibz {} Np {}: F_ibz vs hf_t max|diff| {:.2e} Ha (max|F| {:.4f}){}", name,
            mpi.comm.size(), f.nk, f.nkI, f.Np, err, mx,
            err_full >= 0.0 ? ", vs full-BZ hartree_exchange " + std::to_string(err_full) : std::string(""));
    REQUIRE(err <= 1e-10);
    if (err_full >= 0.0) REQUIRE(err_full <= 1e-13);
  }
}

namespace {
ptree ibz_scf_params(std::string const &output, long niter) {
  ptree pt;
  pt.put("time_grid", "id");
  pt.put("g_repr", "lehmann");
  pt.put("theta_deg", 20.0);
  pt.put("eps", 1e-8);
  pt.put("lam", 6.0);
  pt.put("lam_b", 4.0);
  pt.put("sigma_gap", 0.02);
  pt.put("bos_gap", 0.02);
  pt.put("nodes_per_ray", 120);
  pt.put("K", 8);
  pt.put("niter", niter);
  pt.put("mixing", 0.5);
  pt.put("mixing_alg", "linear");
  pt.put("damp_below", 0.0);
  pt.put("conv_thr", 1e-14);
  pt.put("t_chunk", 8);
  pt.put("restart", false);
  pt.put("output", output);
  pt.put("spectra.enable", false);
  return pt;
}
} // namespace

TEST_CASE("gw_line_ibz_scf", "[ibz][scf][gw_line]") {
  auto &mpi = *utils::make_unit_test_mpi_context();
  for (auto [sym, nosym] : std::vector<std::pair<std::string, std::string>>{{"qe_lih222_sym", "qe_lih222"}, {"qe_lih223_sym", "qe_lih223"}}) {
    auto fs = make_fix(sym), fn = make_fix(nosym);
    const long niter = 3;
    auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fs.thc, *fs.mf, ibz_scf_params("ibz_scf_" + sym, niter));
    auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fn.thc, *fn.mf, ibz_scf_params("ibz_scf_" + nosym, niter));
    REQUIRE(long(A.history.size()) == niter);
    REQUIRE(long(B.history.size()) == niter);
    // gated: iterations 1-2; iteration 3 reported only (the K = 8 closure flips basins on 1e-10 Sigma changes, S7f: the
    // nosym reference itself moves by 2.3e-4 Ha in mu between the Mac and rusty at iteration 3)
    double dmu = 0.0, dgap = 0.0;
    for (long i = 0; i < niter; ++i) {
      if (i < 2) dmu = std::max(dmu, std::abs(A.history[i].mu - B.history[i].mu));
      if (i < 2) dgap = std::max(dgap, std::abs(A.history[i].gap - B.history[i].gap));
      app_log(1, "  [ibz][scf] {} vs {} it {}: mu {:.10f} / {:.10f}, gap {:.8f} / {:.8f} Ha", sym, nosym, i + 1, A.history[i].mu,
              B.history[i].mu, A.history[i].gap, B.history[i].gap);
    }
    // the fixtures themselves: KS mid-gap and KS gap
    app_log(1, "  [ibz][scf] {} vs {} ({} ranks): KS mu {:.10f} / {:.10f}; max |dmu| {:.2e} Ha, max |dgap| {:.2e} Ha (iterations 1-2)",
            sym, nosym, mpi.comm.size(), fs.mu, fn.mu, dmu, dgap);
    REQUIRE(dmu <= 1e-4);
    REQUIRE(dgap <= 1e-4);
  }
  {   // gygi (head terms: per-q heads of the rows R unfolded to the full mesh, Madelung exchange, Sigma_c head) on lih223
    auto fs = make_fix("qe_lih223_sym"), fn = make_fix("qe_lih223");
    auto par = [](std::string const &o) {
      auto pt = ibz_scf_params(o, 1);
      pt.put("div_treatment", "gygi");
      pt.put("hf_div_treatment", "gygi");
      return pt;
    };
    auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fs.thc, *fs.mf, par("ibz_scf_gygi_sym"));
    auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fn.thc, *fn.mf, par("ibz_scf_gygi_nosym"));
    const double de = std::abs(A.eps_inf.back() - B.eps_inf.back());
    app_log(1, "  [ibz][scf][gygi] lih223_sym vs lih223 ({} ranks): eps_inf {:.8f} / {:.8f} (d {:.1e}); it 1 mu {:.10f} / {:.10f}, gap "
               "{:.8f} / {:.8f} Ha",
            mpi.comm.size(), A.eps_inf.back(), B.eps_inf.back(), de, A.history[0].mu, B.history[0].mu, A.history[0].gap,
            B.history[0].gap);
    REQUIRE(de <= 1e-3);
    REQUIRE(std::abs(A.history[0].mu - B.history[0].mu) <= 1e-4);
    REQUIRE(std::abs(A.history[0].gap - B.history[0].gap) <= 1e-4);
  }
}

/// [.ibz_phys]: the sym / nosym difference of iteration 1 (mu, QP gap) and of the HF exchange energy vs the THC size: it must
/// shrink with Np if the IBZ path is exact (the two fixtures have independent THC fits).
TEST_CASE("gw_line_ibz_phys", "[.ibz_phys]") {
  auto &mpi = *utils::make_unit_test_mpi_context();
  auto all  = nda::range::all;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  for (auto [sym, nosym] : std::vector<std::pair<std::string, std::string>>{{"qe_lih222_sym", "qe_lih222"}, {"qe_lih223_sym", "qe_lih223"}})
    for (long fac : {8L, 16L, 32L}) {
      auto fs = make_fix(sym, fac), fn = make_fix(nosym, fac);
      // HF energy of the KS density: sum_k w_k Tr[D F] (gauge invariant)
      auto ehf = [&](fix_t &f) {
        ibz_t ibz(*f.mf, f.nb);
        methods::solvers::hf_t hf("ignore_g0");
        auto sF = math::shm::make_shared_array<Array_view_4D_t>(mpi, {1, f.nkI, f.nb, f.nb});
        nda::array<ComplexType, 4> Dm(1, f.nkI, f.nb, f.nb), S(1, f.nkI, f.nb, f.nb);
        Dm(0, all, all, all) = density_matrix(pole_data_t::from_ks(f.eigI, f.mu));
        S() = ComplexType(0.0);
        for (long k = 0; k < f.nkI; ++k)
          for (long i = 0; i < f.nb; ++i) S(0, k, i, i) = 1.0;
        hf.evaluate(sF, Dm, *f.thc, S, true, true);
        double e = 0.0;
        for (long k = 0; k < f.nkI; ++k)
          for (long i = 0; i < f.nb; ++i)
            for (long j = 0; j < f.nb; ++j) e += ibz.kw(k) * std::real(Dm(0, k, i, j) * sF.local()(0, k, j, i));
        return e;
      };
      const double ea = ehf(fs), eb = ehf(fn);
      auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fs.thc, *fs.mf, ibz_scf_params("ibz_phys_a", 1));
      auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*fn.thc, *fn.mf, ibz_scf_params("ibz_phys_b", 1));
      app_log(1, "  [.ibz_phys] {} / {} Np {} / {}: E_HF(KS D) {:.10f} / {:.10f} (d {:.2e}); it 1 mu {:.10f} / {:.10f} (d {:.2e}), "
                 "gap {:.8f} / {:.8f} (d {:.2e}) Ha",
              sym, nosym, fs.Np, fn.Np, ea, eb, ea - eb, A.history[0].mu, B.history[0].mu, A.history[0].mu - B.history[0].mu,
              A.history[0].gap, B.history[0].gap, A.history[0].gap - B.history[0].gap);
    }
}
