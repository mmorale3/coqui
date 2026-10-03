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
 * gw_line_scf: the self-consistency loop of the line GW (see driver.hpp for the options, the initial guess, the loop and
 * the checkpoint layout). Python oracle: coqui/cayley/cayley/line/driver.py::LineSCGW.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "utilities/h5_background_writer.hpp"
#include "numerics/shared_array/nda.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/closure.hpp"
#include "methods/GW_line/spectra.hpp"
#include "methods/GW_line/driver.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::line_basis_t;
using numerics::line_dlr::time_ray_t;

static constexpr double HA_EV = 27.211386;

// ------------------------------------------------------------------------------------------------------------------
// parameters
// ------------------------------------------------------------------------------------------------------------------
gw_line_params_t gw_line_params_t::from_ptree(ptree const &pt) {
  gw_line_params_t p;
  p.theta_deg     = io::get_value_with_default<double>(pt, "theta_deg", p.theta_deg);
  p.eps           = io::get_value_with_default<double>(pt, "eps", p.eps);
  p.lam           = io::get_value_with_default<double>(pt, "lam", p.lam);
  p.lam_b         = io::get_value_with_default<double>(pt, "lam_b", p.lam_b);
  p.sigma_gap     = io::get_value_with_default<double>(pt, "sigma_gap", p.sigma_gap);
  p.bos_gap       = io::get_value_with_default<double>(pt, "bos_gap", p.bos_gap);
  p.g_gap         = io::get_value_with_default<double>(pt, "g_gap", p.g_gap);
  p.nodes_per_ray = io::get_value_with_default<long>(pt, "nodes_per_ray", p.nodes_per_ray);
  p.node_tmin     = io::get_value_with_default<double>(pt, "node_tmin", p.node_tmin);
  p.node_tmax     = io::get_value_with_default<double>(pt, "node_tmax", p.node_tmax);
  p.wp            = io::get_value_with_default<double>(pt, "wp", p.wp);
  p.K             = io::get_value_with_default<long>(pt, "K", p.K);
  p.tol_gram      = io::get_value_with_default<double>(pt, "tol_gram", p.tol_gram);
  p.nphi          = io::get_value_with_default<long>(pt, "nphi", p.nphi);
  p.niter         = io::get_value_with_default<long>(pt, "niter", p.niter);
  p.mixing        = io::get_value_with_default<double>(pt, "mixing", p.mixing);
  p.conv_thr      = io::get_value_with_default<double>(pt, "conv_thr", p.conv_thr);
  p.t_chunk       = io::get_value_with_default<long>(pt, "t_chunk", p.t_chunk);
  p.ray_decades   = io::get_value_with_default<double>(pt, "ray_decades", p.ray_decades);
  p.restart       = io::get_value_with_default<bool>(pt, "restart", p.restart);
  {
    auto o = pt.get_optional<std::string>("output");
    if (o and not o->empty()) p.output = *o;
    else {
      auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
      auto prefix = io::get_value_with_default<std::string>(pt, "prefix", "gw_line");
      p.output    = outdir + "/" + prefix;
    }
  }
  p.do_spectra    = io::get_value_with_default<bool>(pt, "spectra.enable", true);
  p.spectra.eta   = io::get_array_with_default<double>(pt, "spectra.eta", p.spectra.eta);
  p.spectra.wmin  = io::get_value_with_default<double>(pt, "spectra.wmin", p.spectra.wmin);
  p.spectra.wmax  = io::get_value_with_default<double>(pt, "spectra.wmax", p.spectra.wmax);
  p.spectra.nw    = io::get_value_with_default<long>(pt, "spectra.nw", p.spectra.nw);

  auto div = io::get_value_with_default<std::string>(pt, "div_treatment", "ignore_g0");
  io::tolower(div);
  utils::check(div == "ignore_g0", "gw_line: only div_treatment = \"ignore_g0\" is implemented (got \"{}\"): Z(Gamma) without "
                                   "its G = 0 term and no Madelung/head correction",
               div);
  utils::check(p.theta_deg > 0.0 and p.theta_deg < 90.0, "gw_line: theta_deg must be in (0, 90)");
  utils::check(p.eps > 0.0 and p.lam > 0.0 and p.lam_b > 0.0, "gw_line: eps, lam, lam_b must be > 0");
  utils::check(p.g_gap >= 0.0 and p.g_gap < p.lam, "gw_line: g_gap must be in [0, lam)");
  utils::check(p.nodes_per_ray > 1 and p.node_tmin > 0.0 and p.node_tmax > p.node_tmin, "gw_line: invalid node grid");
  utils::check(p.K >= 1 and p.nphi >= 1 and p.wp > 0.0 and p.tol_gram > 0.0, "gw_line: invalid closure parameters");
  utils::check(p.niter >= 0 and p.t_chunk >= 1 and p.ray_decades > 0.0, "gw_line: invalid niter / t_chunk / ray_decades");
  utils::check(p.mixing > 0.0 and p.mixing <= 1.0, "gw_line: mixing must be in (0, 1]");
  return p;
}

void gw_line_params_t::log() const {
  app_log(1, "  gw_line parameters:");
  app_log(1, "    theta = {} deg (theta_t = {} deg), eps = {:.1e}, lam = {} Ha, lam_b = {} Ha", theta_deg, theta_deg / 2.0, eps,
          lam, lam_b);
  app_log(1, "    sigma_gap = {}{}, bos_gap = {}{}, g_gap = {}", sigma_gap, sigma_gap < 0.0 ? " (auto)" : "", bos_gap,
          bos_gap < 0.0 ? " (auto)" : "", g_gap);
  app_log(1, "    fermionic nodes: {} per ray, |t| in [{}, {}] Ha", nodes_per_ray, node_tmin, node_tmax);
  app_log(1, "    closure: wp = {} Ha, K = {}, tol_gram = {:.1e}, nphi = {}", wp, K, tol_gram, nphi);
  app_log(1, "    niter = {} (total), mixing = {}, conv_thr = {:.1e}, t_chunk = {}, ray_decades = {}", niter, mixing, conv_thr,
          t_chunk, ray_decades);
  app_log(1, "    restart = {}, checkpoint = {}.gw_line.h5", restart, output);
  if (do_spectra) {
    std::string e;
    for (auto x : spectra.eta) e += std::to_string(x) + " ";
    app_log(1, "    spectra: eta = [ {}] Ha, w - mu in [{}, {}] Ha, nw = {}", e, spectra.wmin, spectra.wmax, spectra.nw);
  } else {
    app_log(1, "    spectra: off");
  }
}

// ------------------------------------------------------------------------------------------------------------------
// H0 and the system group
// ------------------------------------------------------------------------------------------------------------------
nda::array<ComplexType, 3> one_body_h0(mf::MF &mf) {
  auto mpi       = mf.mpi();
  const long nkI = mf.nkpts_ibz(), nb = mf.nbnd(), ns = mf.nspin();
  auto psp       = hamilt::make_pseudopot(mf);
  auto sH0       = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mpi, {ns, nkI, nb, nb});
  hamilt::set_H0(mf, psp.get(), sH0);
  auto H0l = sH0.local();
  nda::array<ComplexType, 3> H0(nkI, nb, nb);
  for (long ik = 0; ik < nkI; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) H0(ik, i, j) = 0.5 * (H0l(0, ik, i, j) + std::conj(H0l(0, ik, j, i)));   // hermitize
  mpi->comm.barrier();
  return H0;
}

void write_system_h5(boost::mpi3::communicator &comm, std::string const &file, mf::MF &mf, long Np,
                     nda::array<ComplexType, 3> const &H0, double mu0, bool truncate) {
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, truncate ? 'w' : 'a');
    h5::group g(f);
    auto s        = g.create_group("system");
    const long nk = mf.nkpts(), nb = H0.extent(1);
    nda::array<double, 2> eig(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) eig(ik, n) = mf.eigval()(0, ik, n);
    h5::h5_write(s, "nkpts", nk);
    h5::h5_write(s, "nbnd", nb);
    h5::h5_write(s, "Np", Np);
    h5::h5_write(s, "nelec", double(mf.nelec()));
    h5::h5_write(s, "mu0", mu0);
    nda::h5_write(s, "H0", H0, false);
    nda::h5_write(s, "eigval", eig, false);
    nda::h5_write(s, "qk_to_k2", mf.qk_to_k2(), false);
    nda::h5_write(s, "kpoints", mf.kpts(), false);
  }
  comm.barrier();
}

namespace {

template <typename T, int R> void bcast_array(boost::mpi3::communicator &comm, nda::array<T, R> &A) {
  std::array<long, R> shp{};
  if (comm.root()) shp = A.shape();
  comm.broadcast_n(shp.data(), R, 0);
  if (not comm.root()) A.resize(shp);
  if (A.size() > 0) comm.broadcast_n(A.data(), A.size(), 0);
}

/// ragged pole storage of one sector: counts (nk), flat e, flat coef (sum M, nb, nb)
void write_poles(h5::group &g, pole_data_t const &pd) {
  auto pg = g.create_group("poles");
  for (auto s : {sector_t::particle, sector_t::hole}) {
    const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
    nda::array<long, 1> cnt(pd.nk);
    long tot = 0;
    for (long ik = 0; ik < pd.nk; ++ik) tot += (cnt(ik) = pd(ik, s).size());
    nda::array<double, 1> e(tot);
    nda::array<ComplexType, 3> c(tot, pd.nb, pd.nb);
    long o = 0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      auto const &ps = pd(ik, s);
      for (long m = 0; m < ps.size(); ++m, ++o) {
        e(o)                    = ps.e(m);
        c(o, nda::ellipsis{})   = ps.coef(m, nda::ellipsis{});
      }
    }
    nda::h5_write(pg, nm + "_counts", cnt, false);
    nda::h5_write(pg, nm + "_e", e, false);
    nda::h5_write(pg, nm + "_coef", c, false);
  }
}

pole_data_t read_poles(h5::group &g, long nk, long nb) {
  auto pg = g.open_group("poles");
  pole_data_t pd;
  pd.nk = nk;
  pd.nb = nb;
  pd.part.resize(nk);
  pd.hole.resize(nk);
  for (auto s : {sector_t::particle, sector_t::hole}) {
    const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
    nda::array<long, 1> cnt;
    nda::array<double, 1> e;
    nda::array<ComplexType, 3> c;
    nda::h5_read(pg, nm + "_counts", cnt);
    nda::h5_read(pg, nm + "_e", e);
    nda::h5_read(pg, nm + "_coef", c);
    utils::check(cnt.size() == nk and c.extent(1) == nb, "gw_line restart: pole data shape mismatch");
    long o = 0;
    for (long ik = 0; ik < nk; ++ik) {
      auto &ps = (s == sector_t::particle) ? pd.part[ik] : pd.hole[ik];
      ps.e     = nda::array<double, 1>(e(nda::range(o, o + cnt(ik))));
      ps.coef  = nda::array<ComplexType, 3>(c(nda::range(o, o + cnt(ik)), nda::range::all, nda::range::all));
      o += cnt(ik);
    }
  }
  return pd;
}

void bcast_poles(boost::mpi3::communicator &comm, pole_data_t &pd) {
  std::array<long, 2> d = {pd.nk, pd.nb};
  comm.broadcast_n(d.data(), 2, 0);
  if (not comm.root()) {
    pd.nk = d[0];
    pd.nb = d[1];
    pd.part.resize(pd.nk);
    pd.hole.resize(pd.nk);
  }
  for (long ik = 0; ik < pd.nk; ++ik) {
    bcast_array(comm, pd.part[ik].e);
    bcast_array(comm, pd.part[ik].coef);
    bcast_array(comm, pd.hole[ik].e);
    bcast_array(comm, pd.hole[ik].coef);
  }
}

void write_history(h5::group &g, gw_line_iter_t const &r) {
  auto hg = g.create_group("history");
  h5::h5_write(hg, "iter", r.iter);
  h5::h5_write(hg, "dSigma", r.dSigma);
  h5::h5_write(hg, "mu", r.mu);
  h5::h5_write(hg, "dmu", r.dmu);
  h5::h5_write(hg, "gap", r.gap);
  h5::h5_write(hg, "e_homo", r.e_homo);
  h5::h5_write(hg, "e_lumo", r.e_lumo);
  h5::h5_write(hg, "nelec", r.nelec);
  h5::h5_write(hg, "nelec_lehmann", r.nelec_lehmann);
  h5::h5_write(hg, "N_mu", r.N_mu);
  h5::h5_write(hg, "dropped_weight", r.dropped);
  h5::h5_write(hg, "heldout_max", r.heldout_max);
  h5::h5_write(hg, "npoles_min", r.npoles_min);
  h5::h5_write(hg, "npoles_max", r.npoles_max);
  h5::h5_write(hg, "bos_gap", r.bos_gap);
  h5::h5_write(hg, "sigma_gap_p", r.sigma_gap_p);
  h5::h5_write(hg, "sigma_gap_h", r.sigma_gap_h);
  h5::h5_write(hg, "time", r.time);
}

gw_line_iter_t read_history(h5::group &g) {
  auto hg = g.open_group("history");
  gw_line_iter_t r;
  h5::h5_read(hg, "iter", r.iter);
  h5::h5_read(hg, "dSigma", r.dSigma);
  h5::h5_read(hg, "mu", r.mu);
  h5::h5_read(hg, "dmu", r.dmu);
  h5::h5_read(hg, "gap", r.gap);
  h5::h5_read(hg, "e_homo", r.e_homo);
  h5::h5_read(hg, "e_lumo", r.e_lumo);
  h5::h5_read(hg, "nelec", r.nelec);
  h5::h5_read(hg, "nelec_lehmann", r.nelec_lehmann);
  h5::h5_read(hg, "N_mu", r.N_mu);
  h5::h5_read(hg, "dropped_weight", r.dropped);
  h5::h5_read(hg, "heldout_max", r.heldout_max);
  h5::h5_read(hg, "npoles_min", r.npoles_min);
  h5::h5_read(hg, "npoles_max", r.npoles_max);
  h5::h5_read(hg, "bos_gap", r.bos_gap);
  h5::h5_read(hg, "sigma_gap_p", r.sigma_gap_p);
  h5::h5_read(hg, "sigma_gap_h", r.sigma_gap_h);
  h5::h5_read(hg, "time", r.time);
  return r;
}

void write_input(h5::group &g, gw_line_params_t const &p, nda::array<ComplexType, 1> const &zeta) {
  auto ig = g.create_group("input");
  h5::h5_write(ig, "theta_deg", p.theta_deg);
  h5::h5_write(ig, "eps", p.eps);
  h5::h5_write(ig, "lam", p.lam);
  h5::h5_write(ig, "lam_b", p.lam_b);
  h5::h5_write(ig, "sigma_gap", p.sigma_gap);
  h5::h5_write(ig, "bos_gap", p.bos_gap);
  h5::h5_write(ig, "g_gap", p.g_gap);
  h5::h5_write(ig, "nodes_per_ray", p.nodes_per_ray);
  h5::h5_write(ig, "node_tmin", p.node_tmin);
  h5::h5_write(ig, "node_tmax", p.node_tmax);
  h5::h5_write(ig, "wp", p.wp);
  h5::h5_write(ig, "K", p.K);
  h5::h5_write(ig, "tol_gram", p.tol_gram);
  h5::h5_write(ig, "nphi", p.nphi);
  h5::h5_write(ig, "mixing", p.mixing);
  h5::h5_write(ig, "conv_thr", p.conv_thr);
  h5::h5_write(ig, "t_chunk", p.t_chunk);
  h5::h5_write(ig, "ray_decades", p.ray_decades);
  nda::h5_write(ig, "fermionic_nodes", zeta, false);
}

/// The SCF state carried from one iteration to the next (exactly what a restart needs).
struct state_t {
  long iter = 0;
  double mu = 0.0, mu_sigma = 0.0, dmu = 0.0, e_homo = 0.0, e_lumo = 0.0;
  nda::array<ComplexType, 3> F;
  pole_data_t poles;
  bool have_sigma = false;
  nda::array<ComplexType, 4> Sig_p, Sig_h;
};

void write_state(boost::mpi3::communicator &comm, std::string const &file, state_t const &st,
                 gw_line_iter_t const *rec) {
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'a');
    h5::group g(f);
    auto sg = g.has_subgroup("scf_line") ? g.open_group("scf_line") : g.create_group("scf_line");
    auto it = sg.create_group("iter" + std::to_string(st.iter));
    h5::h5_write(it, "mu", st.mu);
    h5::h5_write(it, "mu_sigma", st.mu_sigma);
    h5::h5_write(it, "dmu", st.dmu);
    h5::h5_write(it, "e_homo", st.e_homo);
    h5::h5_write(it, "e_lumo", st.e_lumo);
    nda::h5_write(it, "F", st.F, false);
    if (st.have_sigma) {
      nda::h5_write(it, "Sigma_p", st.Sig_p, false);
      nda::h5_write(it, "Sigma_h", st.Sig_h, false);
    }
    write_poles(it, st.poles);
    if (rec != nullptr) write_history(it, *rec);
    h5::h5_write(sg, "final_iter", st.iter);
  }
  comm.barrier();
}

/// Root reads scf_line/final_iter (+ the history of iterations 1..final_iter), everything is broadcast.
state_t read_state(boost::mpi3::communicator &comm, std::string const &file, long nk, long nb,
                   nda::array<ComplexType, 1> const &zeta, std::vector<gw_line_iter_t> &history) {
  state_t st;
  long nhist = 0;
  nda::array<double, 2> hist;
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'r');
    h5::group g(f);
    {
      auto ig = g.open_group("input");
      nda::array<ComplexType, 1> z;
      nda::h5_read(ig, "fermionic_nodes", z);
      utils::check(z.size() == zeta.size() and nda::max_element(nda::abs(z - zeta)) <= 1e-14 * nda::max_element(nda::abs(zeta)),
                   "gw_line restart: the fermionic nodes of {} differ from the input (node grid / theta changed)", file);
    }
    auto sg = g.open_group("scf_line");
    h5::h5_read(sg, "final_iter", st.iter);
    auto it = sg.open_group("iter" + std::to_string(st.iter));
    h5::h5_read(it, "mu", st.mu);
    h5::h5_read(it, "mu_sigma", st.mu_sigma);
    h5::h5_read(it, "dmu", st.dmu);
    h5::h5_read(it, "e_homo", st.e_homo);
    h5::h5_read(it, "e_lumo", st.e_lumo);
    nda::h5_read(it, "F", st.F);
    st.have_sigma = it.has_dataset("Sigma_p");
    if (st.have_sigma) {
      nda::h5_read(it, "Sigma_p", st.Sig_p);
      nda::h5_read(it, "Sigma_h", st.Sig_h);
    }
    st.poles = read_poles(it, nk, nb);
    utils::check(st.F.extent(0) == nk and st.F.extent(1) == nb, "gw_line restart: F shape mismatch");
    nhist = st.iter;
    hist  = nda::array<double, 2>(nhist, 18);
    for (long i = 1; i <= st.iter; ++i) {
      auto gi = sg.open_group("iter" + std::to_string(i));
      auto r  = read_history(gi);
      double v[18] = {double(r.iter), r.dSigma, r.mu, r.dmu, r.gap, r.e_homo, r.e_lumo, r.nelec, r.nelec_lehmann, r.N_mu,
                      r.dropped, r.heldout_max, double(r.npoles_min), double(r.npoles_max), r.bos_gap, r.sigma_gap_p,
                      r.sigma_gap_h, r.time};
      for (int j = 0; j < 18; ++j) hist(i - 1, j) = v[j];
    }
  }
  std::array<double, 7> sc = {double(st.iter), st.mu, st.mu_sigma, st.dmu, st.e_homo, st.e_lumo, st.have_sigma ? 1.0 : 0.0};
  comm.broadcast_n(sc.data(), sc.size(), 0);
  st.iter       = long(std::llround(sc[0]));
  st.mu         = sc[1];
  st.mu_sigma   = sc[2];
  st.dmu        = sc[3];
  st.e_homo     = sc[4];
  st.e_lumo     = sc[5];
  st.have_sigma = sc[6] > 0.5;
  bcast_array(comm, st.F);
  if (st.have_sigma) {
    bcast_array(comm, st.Sig_p);
    bcast_array(comm, st.Sig_h);
  }
  bcast_poles(comm, st.poles);
  bcast_array(comm, hist);
  history.clear();
  for (long i = 0; i < hist.extent(0); ++i) {
    gw_line_iter_t r;
    r.iter = long(std::llround(hist(i, 0)));
    r.dSigma = hist(i, 1); r.mu = hist(i, 2); r.dmu = hist(i, 3); r.gap = hist(i, 4); r.e_homo = hist(i, 5);
    r.e_lumo = hist(i, 6); r.nelec = hist(i, 7); r.nelec_lehmann = hist(i, 8); r.N_mu = hist(i, 9); r.dropped = hist(i, 10);
    r.heldout_max = hist(i, 11); r.npoles_min = long(std::llround(hist(i, 12))); r.npoles_max = long(std::llround(hist(i, 13)));
    r.bos_gap = hist(i, 14); r.sigma_gap_p = hist(i, 15); r.sigma_gap_h = hist(i, 16); r.time = hist(i, 17);
    history.push_back(r);
  }
  return st;
}

void write_spectra(boost::mpi3::communicator &comm, std::string const &file, spectra_out_t const &s, double mu) {
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'a');
    h5::group g(f);
    auto sg = g.create_group("spectra");
    h5::h5_write(sg, "mu", mu);
    nda::h5_write(sg, "omega", s.omega, false);
    nda::h5_write(sg, "eta", s.eta, false);
    nda::h5_write(sg, "A_k_w_diag", s.A_diag, false);
    nda::h5_write(sg, "A_k_w_trace", s.A_trace, false);
    h5::h5_write(sg, "e_homo", s.e_homo);
    h5::h5_write(sg, "e_lumo", s.e_lumo);
    h5::h5_write(sg, "vbm", mu + s.e_homo);
    h5::h5_write(sg, "cbm", mu + s.e_lumo);
    h5::h5_write(sg, "gap", s.e_lumo - s.e_homo);
  }
  comm.barrier();
}

void print_line(gw_line_iter_t const &r, double tPi, double tW, double tS, double tC, double tF) {
  app_log(1,
          "iter {:3d}: dSigma {:.2e}  mu {:.6f} (dmu {:+.4f} eV)  QP gap {:.4f} eV  nelec {:.6f} (Lehmann {:.6f}, N(mu) {:.6f}, "
          "dropped {:.1e})  npoles {}-{}  held-out {:.1e}  [Pi {:.1f}s W {:.1f}s Sigma {:.1f}s closure {:.1f}s F {:.1f}s total "
          "{:.1f}s]",
          r.iter, r.dSigma, r.mu, r.dmu * HA_EV, r.gap * HA_EV, r.nelec, r.nelec_lehmann, r.N_mu, r.dropped, r.npoles_min,
          r.npoles_max, r.heldout_max, tPi, tW, tS, tC, tF, r.time);
}

} // namespace

// ------------------------------------------------------------------------------------------------------------------
// driver
// ------------------------------------------------------------------------------------------------------------------
template <MEMORY_SPACE MEM> gw_line_result_t gw_line_scf(methods::thc_reader_t &thc, mf::MF &mf, ptree const &pt) {
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  auto prm     = gw_line_params_t::from_ptree(pt);
  auto &mpi    = *thc.mpi();
  auto &comm   = mpi.comm;
  const std::string chk = prm.output + ".gw_line.h5";

  utils::check(mf.nkpts() == mf.nkpts_ibz() and mf.nqpts() == mf.nqpts_ibz(),
               "gw_line: requires a k mesh without symmetry reduction (nkpts {} nkpts_ibz {}); use a nosym mean field",
               mf.nkpts(), mf.nkpts_ibz());
  utils::check(mf.nspin() == 1 and mf.npol() == 1 and thc.ns() == 1 and thc.npol() == 1,
               "gw_line: spin-restricted collinear only (nspin {}, npol {})", mf.nspin(), mf.npol());
  utils::check(thc.nbnd() == mf.nbnd(), "gw_line: THC nbnd {} != MF nbnd {}", thc.nbnd(), mf.nbnd());
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const double nelec = double(mf.nelec());
  const long nocc    = long(std::llround(nelec / 2.0));
  utils::check(std::abs(nelec - 2.0 * nocc) < 1e-8 and nocc > 0 and nocc < nb,
               "gw_line: need an even electron count with 0 < nelec/2 < nbnd (nelec {}, nbnd {})", nelec, nb);

  utils::TimerManager Timer;
  for (auto nm : {"total", "H0", "bases", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F", "checkpoint",
                  "spectra"})
    Timer.add(nm);
  Timer.start("total");

  app_log(1, "\n╔══════════════════════════════════════════════════════════╗");
  app_log(1, "║  CoQuí: self-consistent GW on the tilted frequency line  ║");
  app_log(1, "╚══════════════════════════════════════════════════════════╝");
  app_log(1, "  nkpts = {}, nqpts = {}, nbnd = {}, Np = {}, nelec = {}, ranks = {}, memory space = {}", nk, nq, nb, Np, nelec,
          comm.size(), MEM == HOST_MEMORY ? "host" : "device");
  prm.log();

  // one-body Hamiltonian, KS spectrum, initial centre
  Timer.start("H0");
  auto H0 = one_body_h0(mf);
  Timer.stop("H0");
  nda::array<double, 2> eig(nk, nb);
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      eig(ik, n) = mf.eigval()(0, ik, n);
      if (n < nocc) homo = std::max(homo, eig(ik, n));
      else lumo = std::min(lumo, eig(ik, n));
    }
  utils::check(lumo > homo, "gw_line: the KS spectrum has no gap (homo {} lumo {})", homo, lumo);
  const double mu0 = 0.5 * (homo + lumo);

  // fixed grids and bases
  const double theta = prm.theta_deg * std::numbers::pi / 180.0, theta_t = 0.5 * theta;
  auto zeta          = numerics::line_dlr::dense_nodes(theta, prm.node_tmin, prm.node_tmax, prm.nodes_per_ray);
  const long nz      = zeta.size();
  Timer.start("bases");
  line_basis_t gp(theta, prm.lam, prm.eps, prm.lam, prm.g_gap, -1.0, prm.node_tmax);
  line_basis_t gh(theta, prm.lam, prm.eps, prm.g_gap, prm.lam, -1.0, prm.node_tmax);
  Timer.stop("bases");
  closure_params_t cprm{prm.wp, prm.K, prm.tol_gram, prm.nphi};

  // state: restart or KS start
  state_t st;
  gw_line_result_t res;
  const bool restart = prm.restart and std::filesystem::exists(chk);
  if (prm.restart and not restart) app_log(1, "  restart requested but {} does not exist: starting from the KS poles", chk);
  aux_grid_t grid(mpi, Np);
  propagator_t<MEM> prop(thc, grid);
  if (restart) {
    st = read_state(comm, chk, nk, nb, zeta, res.history);
    app_log(1, "  resumed from {}: {} iterations done, mu {:.6f} Ha", chk, st.iter, st.mu);
  } else {
    st.iter     = 0;
    st.mu       = mu0;
    st.mu_sigma = mu0;
    st.e_homo   = homo - mu0;
    st.e_lumo   = lumo - mu0;
    st.poles    = pole_data_t::from_ks(eig, mu0);
    write_system_h5(comm, chk, mf, Np, H0, mu0, true);
    if (comm.root()) {
      h5::file f(chk, 'a');
      h5::group g(f);
      write_input(g, prm, zeta);
    }
    comm.barrier();
  }

  // bosonic basis (rebuilt when its gap changes), Sigma bases (rebuilt when their gaps change)
  std::optional<bosonic_basis_t> bos;
  std::optional<line_basis_t> bp, bh;
  auto update_bases = [&]() {
    Timer.start("bases");
    const double bgap = prm.bos_gap >= 0.0 ? prm.bos_gap : 0.5 * (st.e_lumo - st.e_homo);
    if (not bos or bos->gap != bgap) bos.emplace(theta, prm.lam_b, prm.eps, bgap);
    double gpp = prm.sigma_gap, ghh = prm.sigma_gap;
    if (prm.sigma_gap < 0.0) {
      gpp = 0.8 * (st.e_lumo + bos->gap);
      ghh = 0.8 * (std::abs(st.e_homo) + bos->gap);
    }
    if (not bp or bp->gap[1] != gpp) bp.emplace(theta, prm.lam, prm.eps, prm.lam, gpp, -1.0, prm.node_tmax);
    if (not bh or bh->gap[0] != ghh) bh.emplace(theta, prm.lam, prm.eps, ghh, prm.lam, -1.0, prm.node_tmax);
    Timer.stop("bases");
  };
  update_bases();
  app_log(1, "  bases: bosonic rank {} ({} nodes, gap {:.4f}), Sigma {}+{}, G {}+{}, fermionic nodes {}", bos->rank,
          bos->zeta_nodes.size(), bos->gap, bp->rank, bh->rank, gp.rank, gh.rank, nz);

  dyson_layout_t lay(comm.size(), comm.rank(), nq, bos->zeta_nodes.size(), Np);
  coulomb_blocks_t<MEM> Zb(thc, grid, lay.q_rng(), Timer);
  grid.log(nk, nq, bos->zeta_nodes.size(), bos->rank, prm.t_chunk, nb);
  lay.log();

  if (not restart) {
    Timer.start("phase_F");
    auto D = density_matrix(st.poles);
    hartree_exchange<MEM>(prop, Zb, D, mf, grid, mpi, st.F, Timer);
    Timer.stop("phase_F");
    Timer.start("checkpoint");
    write_state(comm, chk, st, nullptr);
    Timer.stop("checkpoint");
    app_log(1, "  start: KS poles, mu0 = {:.6f} Ha (KS mid-gap), KS gap {:.4f} eV, F = V_H + Sigma_x[D_KS]", mu0,
            (lumo - homo) * HA_EV);
  }

  // ---------------------------------------------------------------------------------------------- the loop
  arr4_t Pi, w;
  nda::array<ComplexType, 4> Sp_new, Sh_new;
  bool converged = false;
  while (st.iter < prm.niter) {
    const auto t0 = std::chrono::steady_clock::now();
    update_bases();
    const double emin = st.poles.emin();
    auto ray_p = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::particle);
    auto ray_h = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::hole);
    if (st.iter == 0 or res.history.empty())
      app_log(2, "  rays: emin {:.3e} Ha -> {} + {} time nodes", emin, ray_p.size(), ray_h.size());

    auto tic = [&](char const *nm) { Timer.start(nm); return Timer.elapsed(nm); };
    auto toc = [&](char const *nm, double e0) { Timer.stop(nm); return Timer.elapsed(nm) - e0; };

    // 1. Pi at the bosonic nodes, W residues
    double e0 = tic("phase_Pi");
    polarization<MEM>(prop, st.poles, mf, grid, bos->zeta_nodes, ray_p, ray_h, prm.t_chunk, Pi, Timer);
    const double tPi = toc("phase_Pi", e0);
    e0 = tic("phase_W");
    screened_interaction<MEM>(Pi, Zb, *bos, grid, mpi, w, Timer);
    const double tW = toc("phase_W", e0);

    // 2. Sigma per sector at the dense nodes, mixing
    e0 = tic("phase_Sigma");
    self_energy<MEM>(prop, st.poles, w, *bos, mf, grid, mpi, zeta, ray_p, ray_h, prm.t_chunk, Sp_new, Timer, sector_t::particle);
    self_energy<MEM>(prop, st.poles, w, *bos, mf, grid, mpi, zeta, ray_p, ray_h, prm.t_chunk, Sh_new, Timer, sector_t::hole);
    double dS = 0.0;
    if (st.have_sigma) {
      const double a = prm.mixing, b = 1.0 - prm.mixing;
      for (long ik = 0; ik < nk; ++ik)
        for (long iz = 0; iz < nz; ++iz)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) {
              const ComplexType sp = a * Sp_new(ik, iz, i, j) + b * st.Sig_p(ik, iz, i, j);
              const ComplexType sh = a * Sh_new(ik, iz, i, j) + b * st.Sig_h(ik, iz, i, j);
              dS = std::max(dS, std::abs(sp + sh - st.Sig_p(ik, iz, i, j) - st.Sig_h(ik, iz, i, j)));
              Sp_new(ik, iz, i, j) = sp;
              Sh_new(ik, iz, i, j) = sh;
            }
    }
    st.Sig_p      = Sp_new;
    st.Sig_h      = Sh_new;
    st.have_sigma = true;
    const double tS = toc("phase_Sigma", e0);

    // 3. closure with H_stat - mu = H0 + F - mu
    e0 = tic("phase_closure");
    nda::array<ComplexType, 3> Hrel(nk, nb, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + st.F(ik, i, j) - (i == j ? st.mu : 0.0);
    auto co = closure(comm, Hrel, st.Sig_p, st.Sig_h, zeta, *bp, *bh, gp, gh, cprm, nelec, Timer);
    const double tC = toc("phase_closure", e0);
    st.mu_sigma = st.mu;
    st.mu += co.dmu;
    st.dmu    = co.dmu;
    st.e_homo = co.e_homo;
    st.e_lumo = co.e_lumo;
    st.poles  = std::move(co.poles);

    // 4. static part of the new poles
    e0 = tic("phase_F");
    auto D = density_matrix(st.poles);
    hartree_exchange<MEM>(prop, Zb, D, mf, grid, mpi, st.F, Timer);
    const double tF = toc("phase_F", e0);

    st.iter += 1;
    gw_line_iter_t rec;
    rec.iter          = st.iter;
    rec.dSigma        = dS;
    rec.mu            = st.mu;
    rec.dmu           = co.dmu;
    rec.gap           = co.e_lumo - co.e_homo;
    rec.e_homo        = co.e_homo;
    rec.e_lumo        = co.e_lumo;
    rec.nelec         = co.nel_compressed;
    rec.nelec_lehmann = co.nel_lehmann;
    rec.N_mu          = co.N_mu;
    rec.dropped       = co.dropped;
    rec.heldout_max   = *std::max_element(co.heldout.begin(), co.heldout.end());
    rec.npoles_min    = *std::min_element(co.npoles.begin(), co.npoles.end());
    rec.npoles_max    = *std::max_element(co.npoles.begin(), co.npoles.end());
    rec.bos_gap       = bos->gap;
    rec.sigma_gap_p   = bp->gap[1];
    rec.sigma_gap_h   = bh->gap[0];
    rec.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    res.history.push_back(rec);
    print_line(rec, tPi, tW, tS, tC, tF);

    Timer.start("checkpoint");
    write_state(comm, chk, st, &rec);
    Timer.stop("checkpoint");

    if (rec.iter > 1 and dS < prm.conv_thr) {
      converged = true;
      app_log(1, "  converged: max|dSigma| = {:.2e} < conv_thr = {:.1e}", dS, prm.conv_thr);
      break;
    }
  }
  if (not converged and not res.history.empty() and res.history.back().iter > 1 and res.history.back().dSigma < prm.conv_thr)
    converged = true;

  // ---------------------------------------------------------------------------------------------- spectra
  if (prm.do_spectra and st.have_sigma) {
    Timer.start("spectra");
    nda::array<ComplexType, 3> Hrel(nk, nb, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + st.F(ik, i, j) - (i == j ? st.mu : 0.0);
    auto sp = line_spectra(comm, Hrel, st.Sig_p, st.Sig_h, zeta, st.mu - st.mu_sigma, *bp, *bh, cprm, nelec, prm.spectra);
    write_spectra(comm, chk, sp, st.mu);
    Timer.stop("spectra");
    app_log(1, "  spectra: A(k, w) for {} eta x {} k x {} w written to {}:/spectra ({:.1f} s)", sp.eta.size(), nk, sp.omega.size(),
            chk, Timer.elapsed("spectra"));
    app_log(1, "  final QP edges: VBM {:.6f} Ha, CBM {:.6f} Ha (mu {:.6f} Ha) -> QP gap {:.4f} eV", st.mu + sp.e_homo,
            st.mu + sp.e_lumo, st.mu, (sp.e_lumo - sp.e_homo) * HA_EV);
    res.spectra = std::move(sp);
  } else if (prm.do_spectra) {
    app_log(1, "  spectra: no Sigma in the state (no iteration done), skipped");
  }

  Timer.stop("total");
  app_log(1, "\n  gw_line timers (s, all iterations of this run):");
  for (auto nm : {"total", "H0", "bases", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F", "checkpoint",
                  "spectra"})
    app_log(1, "    {:<20s} {:10.3f}", nm, Timer.elapsed(nm));
  app_log(1, "  kernel sub-timers:");
  for (auto const &nm : Timer.timer_names()) {
    if (nm == "total" or nm == "H0" or nm == "bases" or nm == "checkpoint" or nm == "spectra" or nm.rfind("phase_", 0) == 0)
      continue;
    app_log(1, "    {:<20s} {:10.3f}", nm, Timer.elapsed(nm));
  }

  res.converged = converged;
  res.mu        = st.mu;
  res.mu_sigma  = st.mu_sigma;
  res.poles     = std::move(st.poles);
  res.F         = std::move(st.F);
  res.H0        = std::move(H0);
  res.Sig_p     = std::move(st.Sig_p);
  res.Sig_h     = std::move(st.Sig_h);
  res.zeta      = zeta;
  return res;
}

template gw_line_result_t gw_line_scf<HOST_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#if defined(ENABLE_DEVICE)
template gw_line_result_t gw_line_scf<DEVICE_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#endif

} // namespace methods::gw_line
