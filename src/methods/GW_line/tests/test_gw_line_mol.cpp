/**
 * D1 (notes/demo_plan.md, "D1 - molecules via pyscf"): the line scGW driver on a MOLECULE, pyscf mean field in the MO basis.
 *
 * Fixture tests/unit_test_files/gw_line/h2o_ccpvdz_mo/ (scripts/molecules/pyscf_to_coqui.py, RHF H2O cc-pVDZ all-electron, mo = True):
 *   pyscf.h5 (mean field, S = 1), thc.eri.h5 (LS-THC from the GDF Cholesky vectors, nIpts 288 = 12 nbnd, thresh 0),
 *   bases.h5 (the C++ real-pole bases of mol_params(), "[.gw_line_dump_mol]", read by both sides).
 * Reference tests/unit_test_files/gw_line/h2o_mol_ref.h5 from the python prototype (coqui/cayley/scripts/gen_h2o_mol_ref.py,
 * LineSCGW g_repr "lehmann", same bases, time rays "gl").
 *
 * [gw_line][mol]: one 3-iteration run (iteration 1 = G0W0@HF, 2-3 = scGW) at the molecular settings (lam 55 by the support
 *   rule, node_tmax 110, lam_b 110, wp 0.35, K 24, ignore_g0, g_repr lehmann):
 *   (o)  the MO basis: max|S - 1| <= 1e-12 (the driver's guard would abort otherwise);
 *   (i)  iteration 0: F[D_HF] (V_H + Sigma_x of the HF density from the THC) vs python (<= 3e-8 Ha; measured 2.7e-9), and
 *        max|H0 + F[D_HF] - diag(eps_HF)| <= 5e-5 Ha (the THC exchange error of this fixture, 8.1e-6);
 *   (ii) iteration 1 (G0W0@HF): mu, HOMO = mu - gap / 2, LUMO vs python <= 1e-6 Ha (measured 3e-8), Sigma rel <= 3e-6 (3.2e-7);
 *   (iii) iterations 2-3: mu / gap (meV) and Sigma at the stored nodes (rel.) within MOL_K x the measured floor (mol_floor).
 */
#undef NDEBUG

#include "catch2/catch.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/driver.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

namespace {

using numerics::line_dlr::line_basis_t;
using boost::property_tree::ptree;

std::string mol_dir() { return std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/h2o_ccpvdz_mo/"; }
std::string mol_ref() { return std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/h2o_mol_ref.h5"; }

constexpr double MOL_THETA = 20.0, MOL_EPS = 1e-8, MOL_LAM = 55.0, MOL_TMAX = 110.0, MOL_LAMB = 110.0;
constexpr long MOL_NIT = 3;
constexpr double MOL_K = 5.0;
/// floors of iterations 2-3 (mu meV, gap meV, Sigma rel) = max of (a) |C++ - python| measured on the Mac (1 and 2 ranks) and (b) the
/// spread of the python reference when the iteration-1 Sigma at the nodes is perturbed by 3e-7 relative white noise before its
/// closure (gen_h2o_mol_ref.py --noise 3e-7, seeds 1 and 2; 3e-7 = the measured iteration-1 C++ - python Sigma difference). The closure
/// amplifies ~3e-7 Sigma differences to ~1e-3 in Sigma[G] one iteration later (K 24, lam 55): (a) it 2 {0.376, 0.337, 5.6e-4},
/// it 3 {1.084, 0.430, 1.39e-3} (1 rank 0.600 meV, 2 ranks 1.084 meV); (b) it 2 {0.676, 0.513, 8.5e-4}, it 3 {1.527, 0.270, 2.19e-3}. Iteration 1 (G0W0@HF, strict
/// gates): QP energies 3e-8 Ha, Sigma rel 3.2e-7, F[D_HF] 2.7e-9 Ha.
constexpr std::array<std::array<double, 3>, MOL_NIT> mol_floor = {{{0, 0, 0}, {0.676, 0.513, 8.5e-4}, {1.527, 0.430, 2.19e-3}}};

struct mol_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  mol_t() {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source, mol_dir(), "pyscf"));
    thc = std::make_unique<methods::thc_reader_t>(mf, "incore", mol_dir() + "thc.eri.h5");
  }
};

/// the molecular settings (= gen_h2o_mol_ref.py P)
ptree mol_params(std::string const &output, long niter) {
  ptree pt;
  pt.put("time_grid", "gl");
  pt.put("g_repr", "lehmann");
  pt.put("theta_deg", MOL_THETA);
  pt.put("eps", MOL_EPS);
  pt.put("lam", MOL_LAM);
  pt.put("lam_b", MOL_LAMB);
  pt.put("sigma_gap", 0.02);
  pt.put("bos_gap", 0.02);
  pt.put("g_gap", 0.0);
  pt.put("nodes_per_ray", 120);
  pt.put("node_tmin", 1e-3);
  pt.put("node_tmax", MOL_TMAX);
  pt.put("wp", 0.35);
  pt.put("K", 24);
  pt.put("tol_gram", 1e-10);
  pt.put("tol_gram_eps", 0.0);   // python's closure: the Gram cut is tol_gram only
  pt.put("nphi", 8);
  pt.put("niter", niter);
  pt.put("mixing", 0.5);
  pt.put("mixing_alg", "linear");
  pt.put("damp_below", 0.0);
  pt.put("conv_thr", 1e-14);
  pt.put("t_chunk", 8);
  pt.put("ray_decades", 36.0);
  pt.put("start", "ks");
  pt.put("div_treatment", "ignore_g0");
  pt.put("hf_div_treatment", "ignore_g0");
  pt.put("restart", false);
  pt.put("output", output);
  pt.put("spectra.enable", false);
  pt.put("checkpoint_sigma", "all");
  pt.put("bases_file", mol_dir() + "bases.h5");
  return pt;
}

bool gate(std::string const &what, double value, double g) {
  const bool ok = std::abs(value) <= g;
  app_log(1, "    {:<34s} {:10.3e} <= gate {:10.3e}  {}", what, std::abs(value), g, ok ? "ok" : "FAIL");
  return ok;
}

nda::array<ComplexType, 3> read_c3(h5::group &g, std::string const &name) {
  nda::array<double, 4> a;
  nda::h5_read(g, name, a);   // TRIQS complex layout: trailing dim 2
  nda::array<ComplexType, 3> c(a.extent(0), a.extent(1), a.extent(2));
  for (long i = 0; i < c.extent(0); ++i)
    for (long j = 0; j < c.extent(1); ++j)
      for (long k = 0; k < c.extent(2); ++k) c(i, j, k) = ComplexType(a(i, j, k, 0), a(i, j, k, 1));
  return c;
}

void remove_ckpt(boost::mpi3::communicator &comm, std::string const &f) {
  comm.barrier();
  if (comm.root() and std::getenv("GW_LINE_TEST_KEEP") == nullptr) {
    for (auto const &x : {f, f.substr(0, f.size() - 3) + ".sigma.h5"})
      if (std::filesystem::exists(x)) std::filesystem::remove(x);
  }
  comm.barrier();
}

} // namespace

/// The C++ bases of mol_params() -> h2o_ccpvdz_mo/bases.h5 (read by gen_h2o_mol_ref.py and by [gw_line][mol]). Run once, 1 rank.
TEST_CASE("gw_line_dump_mol", "[.gw_line_dump_mol]") {
  auto mpi = utils::make_unit_test_mpi_context();
  if (mpi->comm.root()) {
    const double th = MOL_THETA * std::numbers::pi / 180.0;
    line_basis_t bp(th, MOL_LAM, MOL_EPS, MOL_LAM, 0.02, -1.0, MOL_TMAX), bh(th, MOL_LAM, MOL_EPS, 0.02, MOL_LAM, -1.0, MOL_TMAX);
    line_basis_t gp(th, MOL_LAM, MOL_EPS, MOL_LAM, 0.0, -1.0, MOL_TMAX), gh(th, MOL_LAM, MOL_EPS, 0.0, MOL_LAM, -1.0, MOL_TMAX);
    numerics::line_dlr::bosonic_basis_t bos(th, MOL_LAMB, MOL_EPS, 0.02);
    const std::string fn = mol_dir() + "bases.h5";
    h5::file f(fn, 'w');
    h5::group g(f);
    nda::h5_write(g, "sigma_particle_w", bp.w, false);
    nda::h5_write(g, "sigma_hole_w", bh.w, false);
    nda::h5_write(g, "g_particle_w", gp.w, false);
    nda::h5_write(g, "g_hole_w", gh.w, false);
    nda::h5_write(g, "bos_nu", bos.nu, false);
    nda::h5_write(g, "bos_zeta_nodes", bos.zeta_nodes, false);
    h5::h5_write(g, "lam_b", MOL_LAMB);
    app_log(1, "wrote {}: Sigma {}+{}, G {}+{}, bosonic {} ({} nodes)", fn, bp.rank, bh.rank, gp.rank, gh.rank, bos.rank,
            bos.zeta_nodes.size());
  }
  mpi->comm.barrier();
}

TEST_CASE("gw_line_mol", "[gw_line][mol]") {
  if (not std::filesystem::exists(mol_ref())) {
    app_log(1, "[mol] {} not found: run coqui/cayley/scripts/gen_h2o_mol_ref.py", mol_ref());
    FAIL("missing python reference");
  }
  mol_t M;
  auto &comm = M.mpi->comm;
  // reference
  nda::array<double, 1> mu_r, gap_r, dS_r, eps_hf;
  nda::array<long, 1> idx;
  nda::array<ComplexType, 4> Sig_r;
  nda::array<ComplexType, 2> F0_r;
  {
    h5::file f(mol_ref(), 'r');
    h5::group g(f);
    nda::h5_read(g, "mu", mu_r);
    nda::h5_read(g, "gap", gap_r);
    nda::h5_read(g, "dSigma", dS_r);
    nda::h5_read(g, "eps_hf", eps_hf);
    nda::h5_read(g, "node_index", idx);
    nda::array<double, 4> re, im;
    nda::h5_read(g, "Sigma_k0_re", re);
    nda::h5_read(g, "Sigma_k0_im", im);
    Sig_r = nda::array<ComplexType, 4>(re.shape());
    for (long a = 0; a < re.size(); ++a) Sig_r.data()[a] = ComplexType(re.data()[a], im.data()[a]);
    nda::array<double, 2> fr, fi;
    nda::h5_read(g, "F0_re", fr);
    nda::h5_read(g, "F0_im", fi);
    F0_r = nda::array<ComplexType, 2>(fr.shape());
    for (long a = 0; a < fr.size(); ++a) F0_r.data()[a] = ComplexType(fr.data()[a], fi.data()[a]);
  }
  REQUIRE(mu_r.size() >= MOL_NIT);
  const long nb = eps_hf.size();
  // (o) MO basis
  {
    auto sS = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*M.mpi, {1, 1, nb, nb});
    hamilt::set_ovlp(*M.mf, sS);
    auto Sl    = sS.local();
    double err = 0.0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) err = std::max(err, std::abs(Sl(0, 0, i, j) - ComplexType(i == j ? 1.0 : 0.0)));
    app_log(1, "[mol] H2O cc-pVDZ MO basis: nb {}, Np {}, max|S - 1| {:.2e}", nb, M.thc->Np(), err);
    REQUIRE(err <= 1e-12);
  }
  const std::string fo = "gw_line_mol";
  auto pt = mol_params(fo, MOL_NIT);
  auto t0 = std::chrono::steady_clock::now();
  auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*M.thc, *M.mf, pt);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(long(R.history.size()) == MOL_NIT);
  app_log(1, "[mol] ranks {}, {} iterations in {:.1f} s", comm.size(), MOL_NIT, dt);
  bool ok = true;
  // (i) iteration 0: F[D_HF]
  {
    nda::array<ComplexType, 3> F0;
    if (comm.root()) {
      h5::file f(fo + ".gw_line.h5", 'r');
      h5::group g(f);
      auto gi = g.open_group("scf_line/iter0");
      F0      = read_c3(gi, "F");
    }
    std::array<long, 3> shp{1, nb, nb};
    if (not comm.root()) F0.resize(shp);
    comm.broadcast_n(F0.data(), F0.size(), 0);
    double dF = 0.0, dH = 0.0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) {
        dF = std::max(dF, std::abs(F0(0, i, j) - F0_r(i, j)));
        dH = std::max(dH, std::abs(R.H0(0, i, j) + F0(0, i, j) - (i == j ? eps_hf(i) : 0.0)));
      }
    app_log(1, "  (i) iteration 0:");
    ok = gate("max|F[D_HF] C++ - python| (Ha)", dF, 3e-8) and ok;
    ok = gate("max|H0 + F[D_HF] - diag(eps_HF)| (Ha)", dH, 5e-5) and ok;
  }
  app_log(1, "  iter |   mu C++ (Ha)    mu py (Ha)   d(meV) |  gap C++ (eV)  gap py (eV)  d(meV) | dSigma C++  dSigma py | Sigma rel");
  for (long it = 0; it < MOL_NIT; ++it) {
    nda::array<ComplexType, 4> Sp, Sh;
    if (comm.root()) {
      h5::file f(fo + ".gw_line.h5", 'r');
      h5::group g(f);
      auto gi = g.open_group("scf_line/iter" + std::to_string(it + 1));
      nda::h5_read(gi, "Sigma_p", Sp);
      nda::h5_read(gi, "Sigma_h", Sh);
      Sp += Sh;
    }
    std::array<long, 4> shp{};
    if (comm.root()) shp = Sp.shape();
    comm.broadcast_n(shp.data(), 4, 0);
    if (not comm.root()) Sp.resize(shp);
    comm.broadcast_n(Sp.data(), Sp.size(), 0);
    double dmax = 0.0, smax = 0.0;
    for (long n = 0; n < idx.size(); ++n)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          dmax = std::max(dmax, std::abs(Sp(0, idx(n), i, j) - Sig_r(it, n, i, j)));
          smax = std::max(smax, std::abs(Sig_r(it, n, i, j)));
        }
    auto const &h     = R.history[it];
    const double dmu  = (h.mu - mu_r(it)) * 27.211386e3, dgap = (h.gap - gap_r(it)) * 27.211386e3;
    app_log(1, "  {:4d} | {:.9f}  {:.9f}  {:+8.4f} | {:.6f}     {:.6f}    {:+8.4f} | {:.3e}  {:.3e} | {:.2e}", it + 1, h.mu,
            mu_r(it), dmu, h.gap * 27.211386, gap_r(it) * 27.211386, dgap, h.dSigma, dS_r(it), dmax / smax);
    if (it == 0) {   // (ii) G0W0@HF: mu, HOMO, LUMO
      const double dho = (h.mu - 0.5 * h.gap) - (mu_r(it) - 0.5 * gap_r(it));
      const double dlu = (h.mu + 0.5 * h.gap) - (mu_r(it) + 0.5 * gap_r(it));
      app_log(1, "  (ii) G0W0@HF: HOMO {:.8f} LUMO {:.8f} Ha (python {:.8f} {:.8f})", h.mu - 0.5 * h.gap, h.mu + 0.5 * h.gap,
              mu_r(it) - 0.5 * gap_r(it), mu_r(it) + 0.5 * gap_r(it));
      ok = gate("|dmu| (Ha)", h.mu - mu_r(it), 1e-6) and ok;
      ok = gate("|dHOMO| (Ha)", dho, 1e-6) and ok;
      ok = gate("|dLUMO| (Ha)", dlu, 1e-6) and ok;
      ok = gate("Sigma rel", dmax / smax, 3e-6) and ok;
    } else {         // (iii) scGW iterations
      auto const &fl = mol_floor[it];
      ok = gate("|dmu| (meV)", dmu, std::max(1e-3, MOL_K * fl[0])) and ok;
      ok = gate("|dgap| (meV)", dgap, std::max(1e-3, MOL_K * fl[1])) and ok;
      ok = gate("Sigma rel", dmax / smax, std::max(1e-8, MOL_K * fl[2])) and ok;
    }
  }
  REQUIRE(ok);
  remove_ckpt(comm, fo + ".gw_line.h5");
}
