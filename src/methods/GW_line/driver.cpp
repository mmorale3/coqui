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
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <sys/resource.h>

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
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/closure.hpp"
#include "methods/GW_line/closure_device.hpp"
#include "methods/GW_line/spectra.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "methods/GW_line/k_dist.hpp"
#include "methods/GW_line/driver.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::line_basis_t;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::time_nodes_t;

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
  p.g_repr        = io::get_value_with_default<std::string>(pt, "g_repr", p.g_repr);
  io::tolower(p.g_repr);
  p.g_emax        = io::get_value_with_default<double>(pt, "g_emax", p.g_emax);
  p.g_wtol        = io::get_value_with_default<double>(pt, "g_wtol", p.g_wtol);
  p.g_emin_frac   = io::get_value_with_default<double>(pt, "g_emin_frac", p.g_emin_frac);
  p.g_wsmall      = io::get_value_with_default<double>(pt, "g_wsmall", p.g_wsmall);
  p.nodes_per_ray = io::get_value_with_default<long>(pt, "nodes_per_ray", p.nodes_per_ray);
  p.node_tmin     = io::get_value_with_default<double>(pt, "node_tmin", p.node_tmin);
  p.node_tmax     = io::get_value_with_default<double>(pt, "node_tmax", p.node_tmax);
  p.wp            = io::get_value_with_default<double>(pt, "wp", p.wp);
  p.K             = io::get_value_with_default<long>(pt, "K", p.K);
  p.tol_gram      = io::get_value_with_default<double>(pt, "tol_gram", p.tol_gram);
  p.nphi          = io::get_value_with_default<long>(pt, "nphi", p.nphi);
  p.tol_svd       = io::get_value_with_default<double>(pt, "tol_svd", p.tol_svd);
  p.tol_gram_eps  = io::get_value_with_default<double>(pt, "tol_gram_eps", p.tol_gram_eps);
  p.closure_cut   = io::get_value_with_default<std::string>(pt, "closure_cut", p.closure_cut);
  io::tolower(p.closure_cut);
  p.closure_svd_cut = io::get_value_with_default<std::string>(pt, "closure_svd_cut", p.closure_svd_cut);
  io::tolower(p.closure_svd_cut);
  p.closure_cut_window = io::get_value_with_default<double>(pt, "closure_cut_window", p.closure_cut_window);
  p.phase_keep    = io::get_value_with_default<double>(pt, "phase_keep", p.phase_keep);
  p.closure_threads = io::get_value_with_default<long>(pt, "closure_threads", p.closure_threads);
  p.closure_svd   = io::get_value_with_default<std::string>(pt, "closure_svd", p.closure_svd);
  io::tolower(p.closure_svd);
  p.closure_ueig  = io::get_value_with_default<std::string>(pt, "closure_ueig", p.closure_ueig);
  io::tolower(p.closure_ueig);
  p.closure_device = io::get_value_with_default<std::string>(pt, "closure_device", p.closure_device);
  io::tolower(p.closure_device);
  p.closure_dev_svd = io::get_value_with_default<std::string>(pt, "closure_dev_svd", p.closure_dev_svd);
  io::tolower(p.closure_dev_svd);
  p.debug_noise_h0   = io::get_value_with_default<double>(pt, "debug_noise_h0", p.debug_noise_h0);
  p.debug_noise_seed = io::get_value_with_default<long>(pt, "debug_noise_seed", p.debug_noise_seed);
  p.debug_noise_sigma = io::get_value_with_default<double>(pt, "debug_noise_sigma", p.debug_noise_sigma);
  p.debug_noise_iter  = io::get_value_with_default<long>(pt, "debug_noise_iter", p.debug_noise_iter);
  p.bases_file    = io::get_value_with_default<std::string>(pt, "bases_file", p.bases_file);
  p.niter         = io::get_value_with_default<long>(pt, "niter", p.niter);
  p.mixing        = io::get_value_with_default<double>(pt, "mixing", p.mixing);
  p.conv_thr      = io::get_value_with_default<double>(pt, "conv_thr", p.conv_thr);
  p.t_chunk       = io::get_value_with_default<long>(pt, "t_chunk", p.t_chunk);
  p.ray_decades   = io::get_value_with_default<double>(pt, "ray_decades", p.ray_decades);
  p.restart       = io::get_value_with_default<bool>(pt, "restart", p.restart);
  p.time_grid     = io::get_value_with_default<std::string>(pt, "time_grid", p.time_grid);
  io::tolower(p.time_grid);
  p.time_eps        = io::get_value_with_default<double>(pt, "time_eps", p.eps);
  p.time_pad        = io::get_value_with_default<double>(pt, "time_pad", p.time_pad);
  p.time_oversample = io::get_value_with_default<double>(pt, "time_oversample", p.time_oversample);
  p.time_snap       = io::get_value_with_default<double>(pt, "time_snap", p.time_snap);
  p.sigma_kdist     = io::get_value_with_default<bool>(pt, "sigma_kdist", p.sigma_kdist);
  p.checkpoint_sigma = io::get_value_with_default<std::string>(pt, "checkpoint_sigma", p.checkpoint_sigma);
  io::tolower(p.checkpoint_sigma);
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
  utils::check(p.eps > 0.0 and p.lam > 0.0, "gw_line: eps, lam must be > 0");
  utils::check(p.g_gap >= 0.0 and p.g_gap < p.lam, "gw_line: g_gap must be in [0, lam)");
  utils::check(p.g_repr == "lehmann" or p.g_repr == "compressed", "gw_line: g_repr must be \"lehmann\" or \"compressed\" (got \"{}\")",
               p.g_repr);
  if (p.g_emax < 0.0) p.g_emax = p.lam;
  // S7c: the bosonic basis must cover the spectrum of Pi, i.e. the summed energies e^> + |e^<| of the G poles (up to
  // 2 g_emax / 2 lam); a narrower lam_b makes the W fit residues ill-conditioned (|w_j| ~ 1e5 x W, measured on lih222)
  // and amplifies every time-grid error in Sigma ~1e5 x. lam_b <= 0 (the default): 2 g_emax (lehmann) / 2 lam (compressed)
  p.lam_b_auto = (p.lam_b <= 0.0);
  if (p.lam_b_auto) p.lam_b = 2.0 * (p.g_repr == "lehmann" ? p.g_emax : p.lam);
  utils::check(p.g_wtol >= 0.0 and p.g_emin_frac >= 0.0 and p.g_wsmall >= 0.0, "gw_line: g_wtol, g_emin_frac, g_wsmall must be >= 0");
  utils::check(p.nodes_per_ray > 1 and p.node_tmin > 0.0 and p.node_tmax > p.node_tmin, "gw_line: invalid node grid");
  utils::check(p.K >= 1 and p.nphi >= 1 and p.wp > 0.0 and p.tol_gram > 0.0, "gw_line: invalid closure parameters");
  utils::check(p.tol_gram_eps >= 0.0, "gw_line: tol_gram_eps must be >= 0");
  utils::check(p.tol_svd > 0.0 and p.closure_cut_window >= 1.0 and p.phase_keep >= 0.0 and p.debug_noise_h0 >= 0.0 and
                   p.debug_noise_sigma >= 0.0,
               "gw_line: invalid tol_svd / closure_cut_window / phase_keep / debug_noise_h0");
  utils::check(p.closure_cut == "hard" or p.closure_cut == "gap" or p.closure_cut == "smooth",
               "gw_line: closure_cut must be \"hard\", \"gap\" or \"smooth\" (got \"{}\")", p.closure_cut);
  utils::check(p.closure_svd_cut == "hard" or p.closure_svd_cut == "gap",
               "gw_line: closure_svd_cut must be \"hard\" or \"gap\" (got \"{}\")", p.closure_svd_cut);
  utils::check(p.closure_svd == "gesvd" or p.closure_svd == "gesdd", "gw_line: closure_svd must be \"gesvd\" or \"gesdd\" (got \"{}\")",
               p.closure_svd);
  utils::check(p.closure_ueig == "schur" or p.closure_ueig == "cayley",
               "gw_line: closure_ueig must be \"schur\" or \"cayley\" (got \"{}\")", p.closure_ueig);
  utils::check(p.closure_device == "auto" or p.closure_device == "on" or p.closure_device == "off",
               "gw_line: closure_device must be \"auto\", \"on\" or \"off\" (got \"{}\")", p.closure_device);
  utils::check(p.closure_dev_svd == "gesvd" or p.closure_dev_svd == "gesvdp",
               "gw_line: closure_dev_svd must be \"gesvd\" or \"gesvdp\" (got \"{}\")", p.closure_dev_svd);
  utils::check(p.niter >= 0 and p.t_chunk >= 0 and p.ray_decades > 0.0, "gw_line: invalid niter / t_chunk / ray_decades");
  utils::check(p.mixing > 0.0 and p.mixing <= 1.0, "gw_line: mixing must be in (0, 1]");
  utils::check(p.time_grid == "id" or p.time_grid == "gl", "gw_line: time_grid must be \"id\" or \"gl\" (got \"{}\")",
               p.time_grid);
  utils::check(p.checkpoint_sigma == "last" or p.checkpoint_sigma == "all",
               "gw_line: checkpoint_sigma must be \"last\" or \"all\" (got \"{}\")", p.checkpoint_sigma);
  utils::check(p.time_eps > 0.0 and p.time_eps < 1.0 and p.time_pad >= 1.0 and p.time_oversample >= 1.0 and p.time_snap >= 0.0,
               "gw_line: need 0 < time_eps < 1, time_pad >= 1, time_oversample >= 1");
  return p;
}

void gw_line_params_t::log() const {
  app_log(1, "  gw_line parameters:");
  app_log(1, "    theta = {} deg (theta_t = {} deg), eps = {:.1e}, lam = {} Ha, lam_b = {} Ha{}", theta_deg, theta_deg / 2.0, eps,
          lam, lam_b, lam_b_auto ? " (auto: 2 x the G pole range)" : "");
  app_log(1, "    sigma_gap = {}{}, bos_gap = {}{}, g_gap = {}", sigma_gap, sigma_gap < 0.0 ? " (auto)" : "", bos_gap,
          bos_gap < 0.0 ? " (auto)" : "", g_gap);
  if (g_repr == "lehmann")
    app_log(1, "    G representation: lehmann (factorized v v^dagger; prune |e| > {} Ha, weight < {:.1e}, |e| < {} x half gap "
               "with weight < {:.1e})",
            g_emax, g_wtol, g_emin_frac, g_wsmall);
  else
    app_log(1, "    G representation: compressed (gapless per-sector refit, g_gap = {})", g_gap);
  app_log(1, "    fermionic nodes: {} per ray, |t| in [{}, {}] Ha", nodes_per_ray, node_tmin, node_tmax);
  app_log(1, "    closure: wp = {} Ha, K = {}, tol_gram = {:.1e} (used: {:.1e} = max(tol_gram, {} x eps)), nphi = {}, tol_svd = {:.1e}, "
             "cuts {} / {} (window {}), phase continuity {}",
          wp, K, tol_gram, std::max(tol_gram, tol_gram_eps * eps), tol_gram_eps, nphi, tol_svd, closure_cut, closure_svd_cut, closure_cut_window,
          phase_keep > 0.0 ? "on (x " + std::to_string(phase_keep) + ")" : std::string("off"));
  app_log(1, "    closure linear algebra: SVD {}, U eigenvectors {}, BLAS threads {}, device {} (SVD {})", closure_svd, closure_ueig,
          closure_threads < 0 ? std::string("auto") : std::to_string(closure_threads), closure_device, closure_dev_svd);
  if (debug_noise_h0 > 0.0)
    app_log(1, "    DIAGNOSTIC: relative noise {:.1e} on H0 (seed {})", debug_noise_h0, debug_noise_seed);
  if (debug_noise_sigma > 0.0)
    app_log(1, "    DIAGNOSTIC: relative noise {:.1e} on Sigma at the nodes in iteration {} (seed {})", debug_noise_sigma,
            debug_noise_iter, debug_noise_seed);
  if (not bases_file.empty()) app_log(1, "    DIAGNOSTIC: real-pole bases read from {}", bases_file);
  app_log(1, "    niter = {} (total), mixing = {}, conv_thr = {:.1e}, t_chunk = {}, ray_decades = {}", niter, mixing, conv_thr,
          t_chunk, ray_decades);
  if (time_grid == "id")
    app_log(1, "    time grid: ID (time_eps = {:.1e}, time_pad = {}, time_oversample = {}, time_snap = {}), rebuilt every iteration",
            time_eps, time_pad, time_oversample, time_snap);
  else
    app_log(1, "    time grid: GL rays (ray_decades = {}, 3 panels/e-fold, 16 nodes/panel)", ray_decades);
  app_log(1, "    restart = {}, checkpoint = {}.gw_line.h5 (Sigma at the nodes: {})", restart, output,
          checkpoint_sigma == "last" ? "last iteration only, in " + output + ".gw_line.sigma.h5" : "every iteration");
  app_log(1, "    Sigma at the nodes {} (sigma_kdist = {})", sigma_kdist ? "k-distributed (owner k mod np)" : "replicated on every rank",
          sigma_kdist);
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

/// ragged pole storage per sector: counts (nk), flat e (sum M), and the residues: flat coef (sum M, nb, nb) (matrix form)
/// or flat v (sum M, nb) (factorized form; row = v_m)
void write_poles(h5::group &g, pole_data_t const &pd) {
  auto pg = g.create_group("poles");
  for (auto s : {sector_t::particle, sector_t::hole}) {
    const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
    bool fact = true;
    nda::array<long, 1> cnt(pd.nk);
    long tot = 0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      tot += (cnt(ik) = pd(ik, s).size());
      fact = fact and pd(ik, s).is_factorized();
    }
    nda::array<double, 1> e(tot);
    nda::array<ComplexType, 3> c(fact ? 0 : tot, pd.nb, pd.nb);
    nda::array<ComplexType, 2> v(fact ? tot : 0, pd.nb);
    long o = 0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      auto const &ps = pd(ik, s);
      for (long m = 0; m < ps.size(); ++m, ++o) {
        e(o) = ps.e(m);
        if (fact)
          for (long i = 0; i < pd.nb; ++i) v(o, i) = ps.v(i, m);
        else
          c(o, nda::ellipsis{}) = ps.coef(m, nda::ellipsis{});
      }
    }
    nda::h5_write(pg, nm + "_counts", cnt, false);
    nda::h5_write(pg, nm + "_e", e, false);
    if (fact) nda::h5_write(pg, nm + "_v", v, false);
    else nda::h5_write(pg, nm + "_coef", c, false);
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
    const bool fact = pg.has_dataset(nm + "_v");
    nda::array<long, 1> cnt;
    nda::array<double, 1> e;
    nda::array<ComplexType, 3> c;
    nda::array<ComplexType, 2> v;
    nda::h5_read(pg, nm + "_counts", cnt);
    nda::h5_read(pg, nm + "_e", e);
    if (fact) {
      nda::h5_read(pg, nm + "_v", v);
      utils::check(cnt.size() == nk and v.extent(1) == nb and v.extent(0) == e.size(), "gw_line restart: pole data shape mismatch");
    } else {
      nda::h5_read(pg, nm + "_coef", c);
      utils::check(cnt.size() == nk and c.extent(1) == nb, "gw_line restart: pole data shape mismatch");
    }
    long o = 0;
    for (long ik = 0; ik < nk; ++ik) {
      auto &ps = (s == sector_t::particle) ? pd.part[ik] : pd.hole[ik];
      nda::array<double, 1> es(e(nda::range(o, o + cnt(ik))));
      if (fact) {
        nda::array<ComplexType, 2> vs(nb, cnt(ik));
        for (long m = 0; m < cnt(ik); ++m)
          for (long i = 0; i < nb; ++i) vs(i, m) = v(o + m, i);
        ps = pole_sector_t::factorized_form(std::move(es), std::move(vs));
      } else {
        ps = pole_sector_t(std::move(es), nda::array<ComplexType, 3>(c(nda::range(o, o + cnt(ik)), nda::range::all, nda::range::all)));
      }
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
    for (auto *ps : {&pd.part[ik], &pd.hole[ik]}) {
      int f = ps->factorized ? 1 : 0;
      comm.broadcast_n(&f, 1, 0);
      ps->factorized = (f != 0);
      bcast_array(comm, ps->e);
      if (ps->factorized) bcast_array(comm, ps->v);
      else bcast_array(comm, ps->coef);
    }
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
  h5::h5_write(hg, "time_grid", r.time_grid);
  h5::h5_write(hg, "nt_pi_p", r.nt_pi_p);
  h5::h5_write(hg, "nt_pi_h", r.nt_pi_h);
  h5::h5_write(hg, "nt_sigma_p", r.nt_sig_p);
  h5::h5_write(hg, "nt_sigma_h", r.nt_sig_h);
  h5::h5_write(hg, "g_repr", r.g_repr);
  h5::h5_write(hg, "ng_min", r.ng_min);
  h5::h5_write(hg, "ng_max", r.ng_max);
  h5::h5_write(hg, "g_emin", r.g_emin);
  h5::h5_write(hg, "pruned_w", r.pruned_w);
  h5::h5_write(hg, "pruned_w_weight", r.pruned_w_weight);
  h5::h5_write(hg, "pruned_near", r.pruned_near);
  h5::h5_write(hg, "pruned_near_weight", r.pruned_near_weight);
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
  if (hg.has_dataset("time_grid")) {   // absent in S6 checkpoints (GL rays)
    h5::h5_read(hg, "time_grid", r.time_grid);
    h5::h5_read(hg, "nt_pi_p", r.nt_pi_p);
    h5::h5_read(hg, "nt_pi_h", r.nt_pi_h);
    h5::h5_read(hg, "nt_sigma_p", r.nt_sig_p);
    h5::h5_read(hg, "nt_sigma_h", r.nt_sig_h);
  }
  if (hg.has_dataset("g_repr")) {   // absent before S7c (compressed poles)
    h5::h5_read(hg, "g_repr", r.g_repr);
    h5::h5_read(hg, "ng_min", r.ng_min);
    h5::h5_read(hg, "ng_max", r.ng_max);
    h5::h5_read(hg, "g_emin", r.g_emin);
    h5::h5_read(hg, "pruned_w", r.pruned_w);
    h5::h5_read(hg, "pruned_w_weight", r.pruned_w_weight);
    h5::h5_read(hg, "pruned_near", r.pruned_near);
    h5::h5_read(hg, "pruned_near_weight", r.pruned_near_weight);
  }
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
  h5::h5_write(ig, "g_repr", p.g_repr);
  h5::h5_write(ig, "g_emax", p.g_emax);
  h5::h5_write(ig, "g_wtol", p.g_wtol);
  h5::h5_write(ig, "g_emin_frac", p.g_emin_frac);
  h5::h5_write(ig, "g_wsmall", p.g_wsmall);
  h5::h5_write(ig, "nodes_per_ray", p.nodes_per_ray);
  h5::h5_write(ig, "node_tmin", p.node_tmin);
  h5::h5_write(ig, "node_tmax", p.node_tmax);
  h5::h5_write(ig, "wp", p.wp);
  h5::h5_write(ig, "K", p.K);
  h5::h5_write(ig, "tol_gram", p.tol_gram);
  h5::h5_write(ig, "nphi", p.nphi);
  h5::h5_write(ig, "tol_svd", p.tol_svd);
  h5::h5_write(ig, "tol_gram_eps", p.tol_gram_eps);
  h5::h5_write(ig, "closure_cut", p.closure_cut);
  h5::h5_write(ig, "closure_svd_cut", p.closure_svd_cut);
  h5::h5_write(ig, "closure_cut_window", p.closure_cut_window);
  h5::h5_write(ig, "phase_keep", p.phase_keep);
  h5::h5_write(ig, "closure_svd", p.closure_svd);
  h5::h5_write(ig, "closure_ueig", p.closure_ueig);
  h5::h5_write(ig, "closure_device", p.closure_device);
  h5::h5_write(ig, "closure_dev_svd", p.closure_dev_svd);
  h5::h5_write(ig, "mixing", p.mixing);
  h5::h5_write(ig, "conv_thr", p.conv_thr);
  h5::h5_write(ig, "t_chunk", p.t_chunk);
  h5::h5_write(ig, "ray_decades", p.ray_decades);
  h5::h5_write(ig, "time_grid", p.time_grid);
  h5::h5_write(ig, "time_eps", p.time_eps);
  h5::h5_write(ig, "time_pad", p.time_pad);
  h5::h5_write(ig, "time_oversample", p.time_oversample);
  h5::h5_write(ig, "time_snap", p.time_snap);
  h5::h5_write(ig, "checkpoint_sigma", p.checkpoint_sigma);
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
  nda::array<double, 1> phi;   ///< terminal phase phi* of the last closure per k (S7f phase continuity; empty = none)
};

/// Sigma file of checkpoint_sigma = "last" (S7e): <output>.gw_line.sigma.h5, rewritten every iteration (tmp + rename)
std::string sigma_file(std::string const &chk) { return chk.substr(0, chk.size() - 3) + ".sigma.h5"; }

/**
 * Root writes iter<N>. Sigma (k-distributed when kd != nullptr: gathered to the root first, collective) goes to
 * iter<N>/Sigma_{p,h} (sigma_all) or, checkpoint_sigma = "last" (S7e), to the separate file sigma_file(file) that is
 * rewritten every iteration (constant size; before S7e the checkpoint grew by 2 N_k N_zeta nb^2 x 16 B per iteration,
 * 1.8 GB for Si 4x4x4 nb 60).
 */
void write_state(boost::mpi3::communicator &comm, std::string const &file, state_t const &st,
                 gw_line_iter_t const *rec, k_dist_t const *kd = nullptr, bool sigma_all = true) {
  nda::array<ComplexType, 4> Sp_full, Sh_full;
  if (st.have_sigma and kd != nullptr) {
    Sp_full = kd_gather_full(comm, *kd, st.Sig_p);
    Sh_full = kd_gather_full(comm, *kd, st.Sig_h);
  }
  auto const &Sp = (kd != nullptr) ? Sp_full : st.Sig_p;
  auto const &Sh = (kd != nullptr) ? Sh_full : st.Sig_h;
  if (comm.root() and st.have_sigma and not sigma_all) {
    utils::h5_quiesce();
    const std::string sf = sigma_file(file), tmp = sf + ".tmp";
    {
      h5::file f(tmp, 'w');
      h5::group g(f);
      h5::h5_write(g, "iter", st.iter);
      nda::h5_write(g, "Sigma_p", Sp, false);
      nda::h5_write(g, "Sigma_h", Sh, false);
    }
    std::filesystem::rename(tmp, sf);
  }
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
    if (st.have_sigma and sigma_all) {
      nda::h5_write(it, "Sigma_p", Sp, false);
      nda::h5_write(it, "Sigma_h", Sh, false);
    }
    h5::h5_write(it, "has_sigma", long(st.have_sigma ? 1 : 0));
    if (st.phi.size() > 0) nda::h5_write(it, "closure_phi", st.phi, false);
    write_poles(it, st.poles);
    if (rec != nullptr) write_history(it, *rec);
    h5::h5_write(sg, "final_iter", st.iter);
  }
  comm.barrier();
}

static constexpr int NHIST = 31;   ///< columns of the broadcast history table

/// Root reads scf_line/final_iter (+ the history of iterations 1..final_iter), everything is broadcast.
state_t read_state(boost::mpi3::communicator &comm, std::string const &file, long nk, long nb,
                   nda::array<ComplexType, 1> const &zeta, std::vector<gw_line_iter_t> &history, k_dist_t const *kd = nullptr) {
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
    } else if (it.has_dataset("has_sigma")) {   // S7e checkpoint_sigma = "last": the separate Sigma file
      long hs = 0;
      h5::h5_read(it, "has_sigma", hs);
      if (hs != 0) {
        const std::string sf = sigma_file(file);
        utils::check(std::filesystem::exists(sf), "gw_line restart: {} has no Sigma for iteration {} and {} is missing", file,
                     st.iter, sf);
        h5::file fs(sf, 'r');
        h5::group gs(fs);
        long si = -1;
        h5::h5_read(gs, "iter", si);
        utils::check(si == st.iter, "gw_line restart: {} holds Sigma of iteration {}, the checkpoint ends at {}", sf, si, st.iter);
        nda::h5_read(gs, "Sigma_p", st.Sig_p);
        nda::h5_read(gs, "Sigma_h", st.Sig_h);
        st.have_sigma = true;
      }
    }
    st.poles = read_poles(it, nk, nb);
    if (it.has_dataset("closure_phi")) nda::h5_read(it, "closure_phi", st.phi);
    utils::check(st.F.extent(0) == nk and st.F.extent(1) == nb, "gw_line restart: F shape mismatch");
    nhist = st.iter;
    hist  = nda::array<double, 2>(nhist, NHIST);
    for (long i = 1; i <= st.iter; ++i) {
      auto gi = sg.open_group("iter" + std::to_string(i));
      auto r  = read_history(gi);
      double v[NHIST] = {double(r.iter), r.dSigma, r.mu, r.dmu, r.gap, r.e_homo, r.e_lumo, r.nelec, r.nelec_lehmann, r.N_mu,
                         r.dropped, r.heldout_max, double(r.npoles_min), double(r.npoles_max), r.bos_gap, r.sigma_gap_p,
                         r.sigma_gap_h, r.time, r.time_grid == "id" ? 1.0 : 0.0, double(r.nt_pi_p), double(r.nt_pi_h),
                         double(r.nt_sig_p), double(r.nt_sig_h), r.g_repr == "lehmann" ? 1.0 : 0.0, double(r.ng_min),
                         double(r.ng_max), r.g_emin, double(r.pruned_w), r.pruned_w_weight, double(r.pruned_near),
                         r.pruned_near_weight};
      for (int j = 0; j < NHIST; ++j) hist(i - 1, j) = v[j];
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
  if (st.have_sigma and kd != nullptr) {   // k-distributed: every rank receives the rows of its k only
    st.Sig_p = kd_scatter_full(comm, *kd, st.Sig_p);
    st.Sig_h = kd_scatter_full(comm, *kd, st.Sig_h);
  } else if (st.have_sigma) {
    bcast_array(comm, st.Sig_p);
    bcast_array(comm, st.Sig_h);
  }
  bcast_poles(comm, st.poles);
  bcast_array(comm, st.phi);
  bcast_array(comm, hist);
  history.clear();
  for (long i = 0; i < hist.extent(0); ++i) {
    gw_line_iter_t r;
    r.iter = long(std::llround(hist(i, 0)));
    r.dSigma = hist(i, 1); r.mu = hist(i, 2); r.dmu = hist(i, 3); r.gap = hist(i, 4); r.e_homo = hist(i, 5);
    r.e_lumo = hist(i, 6); r.nelec = hist(i, 7); r.nelec_lehmann = hist(i, 8); r.N_mu = hist(i, 9); r.dropped = hist(i, 10);
    r.heldout_max = hist(i, 11); r.npoles_min = long(std::llround(hist(i, 12))); r.npoles_max = long(std::llround(hist(i, 13)));
    r.bos_gap = hist(i, 14); r.sigma_gap_p = hist(i, 15); r.sigma_gap_h = hist(i, 16); r.time = hist(i, 17);
    r.time_grid = hist(i, 18) > 0.5 ? "id" : "gl";
    r.nt_pi_p = long(std::llround(hist(i, 19))); r.nt_pi_h = long(std::llround(hist(i, 20)));
    r.nt_sig_p = long(std::llround(hist(i, 21))); r.nt_sig_h = long(std::llround(hist(i, 22)));
    r.g_repr = hist(i, 23) > 0.5 ? "lehmann" : "compressed";
    r.ng_min = long(std::llround(hist(i, 24))); r.ng_max = long(std::llround(hist(i, 25))); r.g_emin = hist(i, 26);
    r.pruned_w = long(std::llround(hist(i, 27))); r.pruned_w_weight = hist(i, 28);
    r.pruned_near = long(std::llround(hist(i, 29))); r.pruned_near_weight = hist(i, 30);
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

/// retained G poles per k and sector (min, max) and the smallest retained |e|
void pole_counts(pole_data_t const &pd, long &nmin, long &nmax, double &emin) {
  nmin = std::numeric_limits<long>::max();
  nmax = 0;
  for (long ik = 0; ik < pd.nk; ++ik)
    for (auto const *ps : {&pd.part[ik], &pd.hole[ik]}) {
      nmin = std::min(nmin, ps->size());
      nmax = std::max(nmax, ps->size());
    }
  emin = pd.emin();
}


/**
 * Memory plan of the q loops (S7e):
 *   g        : q-group size of the Pi -> W stage (polarization + screened_interaction per group of g consecutive q);
 *   w_host   : residues w of all q kept on the HOST (device runs whose w does not fit), Sigma then streams them per group;
 *   gs_sigma : q-group size of the Sigma stage with host-resident residues (residues of gs_sigma q copied to the device
 *              per group and sector; G~ rebuilt per group).
 * Env overrides: COQUI_GWLINE_QGROUP, COQUI_GWLINE_W_HOST (0/1), COQUI_GWLINE_SIGMA_QGROUP. Host default: g = N_q,
 * resident w (the Pi group of all q is 2.5 GB per rank for Si 4x4x4 nb 60 at 64 ranks and shrinks with the ranks).
 * Device: w resident if it takes <= 35% of the free device memory (min over ranks); g = the largest group whose Pi group
 * plus ~0.8 of it (W-stage sub-step buffers) fits in 80% of the free memory left by the Z blocks, the resident w, the
 * device Sigma accumulator (N_k N_zeta_f nb^2, N_zeta_f <= 2 nz) and the Pi-stage factors at t_chunk 16; gs_sigma: the
 * residues of gs_sigma q in <= 25% of the free memory.
 */
struct q_plan_t {
  long g = 0, gs_sigma = 0;
  bool w_host = false;
};

template <MEMORY_SPACE MEM>
q_plan_t choose_q_plan(boost::mpi3::communicator &comm, aux_grid_t const &grid, long nk, long nq, long nz, long r_b, long nb) {
  q_plan_t qp;
  const double GB  = 1073741824.0;
  const double blk = 16.0 * double(grid.max_block_size());
  const double w   = double(nq) * r_b * blk;
  qp.g             = nq;
  qp.w_host        = false;
  qp.gs_sigma      = nq;
  if constexpr (MEM != HOST_MEMORY) {
    double freeb   = double(utils::freemem_device_effective()) * 1048576.0;
    freeb          = comm.all_reduce_value(freeb, boost::mpi3::min<>{});
    qp.w_host      = (w > 0.35 * freeb);
    const double fixed = double(nq) * blk + (qp.w_host ? 0.0 : w) + 16.0 * double(nk) * 2.0 * nz * nb * nb + 2.0 * nk * 16.0 * blk;
    const double per_q = 1.8 * double(nz) * blk;
    qp.g               = std::clamp(long((0.8 * freeb - fixed) / per_q), 1L, nq);
    qp.gs_sigma        = std::clamp(long(0.25 * freeb / (double(r_b) * blk)), 1L, nq);
    utils::check(0.8 * freeb - fixed > per_q, "gw_line: device memory: Z blocks + residues ({:.2f} GB, {}) + Sigma accumulator + "
                                              "one q of the Pi group ({:.2f} GB) exceed 80% of the free device memory ({:.2f} GB); "
                                              "use more GPUs",
                 w / GB, qp.w_host ? "on the host" : "resident", per_q / GB, freeb / GB);
    app_log(2, "  q plan (device): free {:.2f} GB (min over ranks), residues {:.2f} GB {}, fixed {:.2f} GB, Pi group per q {:.3f} GB",
            freeb / GB, w / GB, qp.w_host ? "on the HOST" : "resident", fixed / GB, per_q / GB);
  }
  if (long v = detail::env_long("COQUI_GWLINE_QGROUP", 0); v > 0) qp.g = std::min(v, nq);
  if (long v = detail::env_long("COQUI_GWLINE_W_HOST", -1); v >= 0) qp.w_host = (v != 0);
  if (long v = detail::env_long("COQUI_GWLINE_SIGMA_QGROUP", 0); v > 0) qp.gs_sigma = std::min(v, nq);
  if (not qp.w_host) qp.gs_sigma = nq;
  return qp;
}

// ------------------------------------------------------------------------------------------------------------------
// S7e instrumentation: per-iteration phase timers (min / avg / max over ranks) and host / device high-water memory
// ------------------------------------------------------------------------------------------------------------------
/// high-water resident set of this process (bytes): getrusage ru_maxrss (Linux: kB, = VmHWM of /proc/self/status;
/// macOS: bytes)
double host_hwm_bytes() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return double(ru.ru_maxrss);
#else
  return double(ru.ru_maxrss) * 1024.0;
#endif
}
/// current resident set (bytes): VmRSS of /proc/self/status (0 where unavailable)
double host_rss_bytes() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return 1024.0 * std::strtod(line.c_str() + 6, nullptr);
  return 0.0;
}

/// the phases of one iteration, in print order (indented names are sub-timers of the preceding phase)
static const std::vector<std::string> phase_names = {
    "time_grid",     "bases",          "phase_Pi",        "G_tilde",         "Pi_hadamard",    "Pi_transform",
    "phase_W",       "W_redistribute", "W_dyson",         "W_fit",           "phase_Sigma",    "Sigma_G_tilde",
    "Sigma_W_time",  "Sigma_hadamard", "Sigma_contract",  "Sigma_allreduce", "Sigma_transform", "Sigma_mix",
    "phase_closure", "closure_upfold", "closure_gather",  "closure_mu",      "closure_compress", "phase_F",
    "checkpoint",    "iteration"};

std::vector<double> phase_snapshot(utils::TimerManager &T) {
  std::vector<double> v;
  v.reserve(phase_names.size());
  for (auto const &nm : phase_names) v.push_back(T.elapsed(nm));
  return v;
}

/**
 * Per-iteration report (level 1): the time of every phase in this iteration, min / avg / max over the ranks (the
 * max/avg ratio is the load imbalance), and the high-water memory (host VmHWM max / min over ranks; device: the
 * largest drop of the free device memory since the start of the run, max over ranks) next to the memory model.
 */
void report_iteration(boost::mpi3::communicator &comm, long iter, std::vector<double> const &t0, std::vector<double> const &t1,
                      double rss0, double model_host, double model_dev, bool device) {
  const long n = long(phase_names.size()), np = comm.size();
  std::vector<double> d(n), mn(n), mx(n), sm(n);
  for (long i = 0; i < n; ++i) d[i] = t1[i] - t0[i];
  comm.all_reduce_n(d.data(), n, mn.data(), boost::mpi3::min<>{});
  comm.all_reduce_n(d.data(), n, mx.data(), boost::mpi3::max<>{});
  comm.all_reduce_n(d.data(), n, sm.data(), std::plus<>{});
  app_log(1, "  phase timers, iteration {} (s over {} ranks; min / avg / max, max/avg):", iter, np);
  for (long i = 0; i < n; ++i) {
    if (mx[i] <= 0.0) continue;
    const bool phase = phase_names[i].rfind("phase_", 0) == 0 or phase_names[i] == "time_grid" or
                       phase_names[i] == "checkpoint" or phase_names[i] == "iteration" or phase_names[i] == "bases";
    const double avg = sm[i] / double(np);
    app_log(1, "    {}{:<18s} {:9.3f} {:9.3f} {:9.3f}  {:5.2f}", phase ? "" : "  ", phase_names[i], mn[i], avg, mx[i],
            avg > 0.0 ? mx[i] / avg : 1.0);
  }
  const double GB = 1024.0 * 1024.0 * 1024.0;
  double hw = host_hwm_bytes(), hmax = 0.0, hmin = 0.0;
  comm.all_reduce_n(&hw, 1, &hmax, boost::mpi3::max<>{});
  comm.all_reduce_n(&hw, 1, &hmin, boost::mpi3::min<>{});
  app_log(1, "  memory, iteration {}: host VmHWM per rank max {:.3f} GB, min {:.3f} GB (RSS before the GW_line arrays {:.3f} GB "
             "max; model: that + {:.3f} GB = {:.3f} GB)",
          iter, hmax / GB, hmin / GB, rss0 / GB, model_host / GB, (rss0 + model_host) / GB);
  if (device) {
    double dh = device_high_water_bytes(), dmax = 0.0;
    comm.all_reduce_n(&dh, 1, &dmax, boost::mpi3::max<>{});
    app_log(1, "  memory, iteration {}: device high-water per rank max {:.3f} GB (model {:.3f} GB)", iter, dmax / GB, model_dev / GB);
  }
}

void print_line(gw_line_iter_t const &r, double tPi, double tW, double tS, double tC, double tF) {
  app_log(1,
          "iter {:3d}: dSigma {:.2e}  mu {:.6f} (dmu {:+.4f} eV)  QP gap {:.4f} eV  nelec {:.6f} (Lehmann {:.6f}, N(mu) {:.6f}, "
          "dropped {:.1e})  npoles {}-{}  held-out {:.1e}  t-nodes ({}) Pi {}+{} Sigma {}+{}  [Pi {:.1f}s W {:.1f}s Sigma "
          "{:.1f}s closure {:.1f}s F {:.1f}s total {:.1f}s]",
          r.iter, r.dSigma, r.mu, r.dmu * HA_EV, r.gap * HA_EV, r.nelec, r.nelec_lehmann, r.N_mu, r.dropped, r.npoles_min,
          r.npoles_max, r.heldout_max, r.time_grid, r.nt_pi_p, r.nt_pi_h, r.nt_sig_p, r.nt_sig_h, tPi, tW, tS, tC, tF, r.time);
  if (r.g_repr == "lehmann")
    app_log(1, "          G lehmann: {}-{} poles per k and sector, min|e| {:.3e} Ha; pruned: weight rule {} ({:.1e}), near-mu rule {} "
               "({:.1e})",
            r.ng_min, r.ng_max, r.g_emin, r.pruned_w, r.pruned_w_weight, r.pruned_near, r.pruned_near_weight);
  else
    app_log(2, "          G compressed: {}-{} poles per k and sector, min|e| {:.3e} Ha", r.ng_min, r.ng_max, r.g_emin);
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
  const k_dist_t kd(nk, comm);                         // owner of Sigma(k) (S7e: k-distributed Sigma)
  k_dist_t const *kdp = prm.sigma_kdist ? &kd : nullptr;
  const bool sig_all  = (prm.checkpoint_sigma == "all");
  utils::check(std::abs(nelec - 2.0 * nocc) < 1e-8 and nocc > 0 and nocc < nb,
               "gw_line: need an even electron count with 0 < nelec/2 < nbnd (nelec {}, nbnd {})", nelec, nb);

  utils::TimerManager Timer;
  for (auto nm : {"total", "H0", "bases", "time_grid", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F",
                  "checkpoint", "spectra", "iteration", "Sigma_mix"})
    Timer.add(nm);
  Timer.start("total");
  // S7e: memory baseline (MF, THC, node-shared arrays) before any GW_line array; device high-water from here on
  double rss0 = host_rss_bytes();
  rss0        = comm.all_reduce_value(rss0, boost::mpi3::max<>{});
  if constexpr (MEM != HOST_MEMORY) device_mem_reset();

  app_log(1, "\n╔══════════════════════════════════════════════════════════╗");
  app_log(1, "║  CoQuí: self-consistent GW on the tilted frequency line  ║");
  app_log(1, "╚══════════════════════════════════════════════════════════╝");
  app_log(1, "  nkpts = {}, nqpts = {}, nbnd = {}, Np = {}, nelec = {}, ranks = {}, memory space = {}", nk, nq, nb, Np, nelec,
          comm.size(), MEM == HOST_MEMORY ? "host" : "device");
  prm.log();

  // one-body Hamiltonian, KS spectrum, initial centre
  Timer.start("H0");
  auto H0 = one_body_h0(mf);
  if (prm.debug_noise_h0 > 0.0) {   // diagnostic (noise-floor meter): the same Hermitian noise on every rank
    const double h0max = nda::max_element(nda::abs(H0));
    std::mt19937_64 gen(0x5eedull + 7919ull * std::uint64_t(prm.debug_noise_seed));
    std::normal_distribution<double> N01;
    for (long ik = 0; ik < H0.extent(0); ++ik)
      for (long i = 0; i < H0.extent(1); ++i)
        for (long j = 0; j <= i; ++j) {
          const ComplexType x = prm.debug_noise_h0 * h0max * ComplexType(N01(gen), i == j ? 0.0 : N01(gen));
          H0(ik, i, j) += x;
          if (j != i) H0(ik, j, i) += std::conj(x);
        }
  }
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
  // diagnostics / parity: poles (and bosonic nodes) of the bases from a file (every rank reads it)
  auto read_w = [&](std::string const &key) {
    nda::array<double, 1> w;
    h5::file f(prm.bases_file, 'r');
    h5::group g(f);
    nda::h5_read(g, key, w);
    return w;
  };
  auto set_line_basis = [&](line_basis_t &b, std::string const &key) {
    if (prm.bases_file.empty()) return;
    b.w    = read_w(key);
    b.rank = b.w.size();
  };
  set_line_basis(gp, "g_particle_w");
  set_line_basis(gh, "g_hole_w");
  Timer.stop("bases");
  closure_params_t cprm{prm.wp, prm.K, std::max(prm.tol_gram, prm.tol_gram_eps * prm.eps), prm.nphi};
  cprm.tol_svd    = prm.tol_svd;
  cprm.gram_cut   = prm.closure_cut;
  cprm.svd_cut    = prm.closure_svd_cut;
  cprm.cut_window = prm.closure_cut_window;
  cprm.phase_keep = prm.phase_keep;
  cprm.svd_driver = prm.closure_svd;
  cprm.ueig       = prm.closure_ueig;
  // S7g: BLAS threads of the host closure. Device runs have one rank per GPU and idle cores; host runs fill the cores
  cprm.blas_threads = prm.closure_threads >= 0 ? prm.closure_threads
                                               : (MEM != HOST_MEMORY ? cores_per_rank(long(mpi.node_comm.size())) : 0);
  {
    const bool dev = (prm.closure_device == "on") or (prm.closure_device == "auto" and MEM != HOST_MEMORY);
    cprm.hooks     = dev ? device_lapack_hooks(prm.closure_dev_svd == "gesvdp" ? 1 : 0) : nullptr;
    utils::check(not(prm.closure_device == "on" and cprm.hooks == nullptr),
                 "gw_line: closure_device = \"on\" needs a CUDA build");
  }
  app_log(1, "  closure: BLAS threads per rank {} ({}), dense eigensolvers/SVD on the {}", cprm.blas_threads,
          cprm.blas_threads > 0 ? (prm.closure_threads >= 0 ? "closure_threads" : "auto: cores of the rank") : "untouched",
          cprm.hooks ? "GPU (cuSOLVER, host fallback)" : "host");
  g_repr_params_t grepr{prm.g_repr, prm.g_emax, prm.g_wtol, prm.g_emin_frac, prm.g_wsmall};

  // state: restart or KS start
  state_t st;
  gw_line_result_t res;
  const bool restart = prm.restart and std::filesystem::exists(chk);
  if (prm.restart and not restart) app_log(1, "  restart requested but {} does not exist: starting from the KS poles", chk);
  aux_grid_t grid(mpi, Np);
  propagator_t<MEM> prop(thc, grid);
  if (restart) {
    st = read_state(comm, chk, nk, nb, zeta, res.history, kdp);
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
    if (not bos or bos->gap != bgap) {
      bos.emplace(theta, prm.lam_b, prm.eps, bgap);
      if (not prm.bases_file.empty()) {
        bos->nu   = read_w("bos_nu");
        bos->rank = bos->nu.size();
        h5::file f(prm.bases_file, 'r');
        h5::group g(f);
        nda::h5_read(g, "bos_zeta_nodes", bos->zeta_nodes);
      }
    }
    double gpp = prm.sigma_gap, ghh = prm.sigma_gap;
    if (prm.sigma_gap < 0.0) {
      gpp = 0.8 * (st.e_lumo + bos->gap);
      ghh = 0.8 * (std::abs(st.e_homo) + bos->gap);
    }
    if (not bp or bp->gap[1] != gpp) {
      bp.emplace(theta, prm.lam, prm.eps, prm.lam, gpp, -1.0, prm.node_tmax);
      set_line_basis(*bp, "sigma_particle_w");
    }
    if (not bh or bh->gap[0] != ghh) {
      bh.emplace(theta, prm.lam, prm.eps, ghh, prm.lam, -1.0, prm.node_tmax);
      set_line_basis(*bh, "sigma_hole_w");
    }
    Timer.stop("bases");
  };
  update_bases();
  app_log(1, "  bases: bosonic rank {} ({} nodes, gap {:.4f}), Sigma {}+{}, G {}+{}, fermionic nodes {}", bos->rank,
          bos->zeta_nodes.size(), bos->gap, bp->rank, bh->rank, gp.rank, gh.rank, nz);

  dyson_layout_t lay(comm.size(), comm.rank(), nq, bos->zeta_nodes.size(), Np);
  // q groups of the Pi -> W stage (S7e): all q at once unless the Pi group does not fit (device) or COQUI_GWLINE_QGROUP
  const q_plan_t qplan = choose_q_plan<MEM>(comm, grid, nk, nq, long(bos->zeta_nodes.size()), bos->rank, nb);
  const q_groups_t qg(nq, qplan.g);
  coulomb_blocks_t<MEM> Zb(thc, grid, qg.dyson_q_list(comm.size(), comm.rank(), long(bos->zeta_nodes.size()), Np), Timer);
  if (qg.n > 1) app_log(1, "  q groups of the Pi -> W stage: {} groups of <= {} q (Pi group of all q does not fit)", qg.n, qg.g);
  if (qplan.w_host)
    app_log(1, "  residues w of all q on the HOST ({:.3f} GB per rank); Sigma streams them in groups of {} q", 16.0 * double(nq) *
                   bos->rank * grid.max_block_size() / 1073741824.0, qplan.gs_sigma);
  const bool dev_fused   = (MEM != HOST_MEMORY) and detail::fused_hadamard();
  const double model_dev = grid.log(nk, nq, bos->zeta_nodes.size(), bos->rank,
                                    (prm.t_chunk > 0 ? prm.t_chunk : (MEM == HOST_MEMORY ? detail::host_t_chunk_default : 64)),
                                    nb, qg.g, dev_fused);
  lay.log();
  // host model (S7e): the kernel arrays (= model_dev on the host path) + the Sigma arrays of the driver (Sig_p, Sig_h,
  // Sp_new, Sh_new: 4 N_k N_zeta_f nb^2) + the per-chunk Sigma reduce buffers (2 N_k t_chunk nb^2) + the full Z(q) of the
  // Dyson slab
  const double sig_bytes  = 16.0 * double(prm.sigma_kdist ? kd.nloc(0) : nk) * double(nz) * double(nb * nb);
  const double model_host = (MEM == HOST_MEMORY ? model_dev : 0.0) + 4.0 * sig_bytes +
                            2.0 * 16.0 * double(nk) * double(detail::host_t_chunk_default) * double(nb * nb) +
                            16.0 * double(Zb.Z_full.extent(0)) * double(Np) * double(Np);
  app_log(2, "    driver host arrays: Sigma 4 x {:.4f} GB ({}), full Z(q) {} x {:.4f} GB; host model {:.4f} GB above the "
             "baseline RSS {:.4f} GB",
          sig_bytes / 1073741824.0, prm.sigma_kdist ? "k-distributed, <= ceil(N_k / np) rows" : "replicated on every rank",
          Zb.Z_full.extent(0), 16.0 * double(Np) * Np / 1073741824.0, model_host / 1073741824.0, rss0 / 1073741824.0);

  if (not restart) {
    Timer.start("phase_F");
    auto D = density_matrix(st.poles);
    hartree_exchange<MEM>(prop, Zb, D, mf, grid, mpi, st.F, Timer);
    Timer.stop("phase_F");
    Timer.start("checkpoint");
    write_state(comm, chk, st, nullptr, kdp, sig_all);
    Timer.stop("checkpoint");
    app_log(1, "  start: KS poles, mu0 = {:.6f} Ha (KS mid-gap), KS gap {:.4f} eV, F = V_H + Sigma_x[D_KS]", mu0,
            (lumo - homo) * HA_EV);
  }

  // ---------------------------------------------------------------------------------------------- the loop
  arr4_t Pi, w;
  memory::array<HOST_MEMORY, ComplexType, 4> w_h;   // host-resident residues (q_plan_t::w_host)
  nda::array<ComplexType, 4> Sp_new, Sh_new;
  bool converged = false;
  while (st.iter < prm.niter) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto ph0 = phase_snapshot(Timer);
    Timer.start("iteration");
    update_bases();
    // time nodes of the ray products: GL rays (both kernels) or the four ID grids of the current poles
    Timer.start("time_grid");
    std::optional<time_nodes_t> pi_p, pi_h, sig_p, sig_h;
    if (prm.time_grid == "gl") {
      const double emin = st.poles.emin();
      auto ray_p = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::particle);
      auto ray_h = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::hole);
      if (st.iter == 0 or res.history.empty())
        app_log(2, "  rays: emin {:.3e} Ha -> {} + {} time nodes", emin, ray_p.size(), ray_h.size());
      pi_p.emplace(ray_p);
      pi_h.emplace(ray_h);
      sig_p.emplace(ray_p);
      sig_h.emplace(ray_h);
    } else {
      numerics::line_dlr::time_id_opts_t topt;
      topt.pad        = prm.time_pad;
      topt.oversample = prm.time_oversample;
      line_time_grids_t tg(st.poles, bos->nu, theta_t, prm.time_eps, topt, bos->zeta_nodes, zeta, comm, prm.time_snap);
      tg.log(1);
      pi_p.emplace(tg.pi_p);
      pi_h.emplace(tg.pi_h);
      sig_p.emplace(tg.sig_p);
      sig_h.emplace(tg.sig_h);
    }
    Timer.stop("time_grid");
    {
      // S7c: Pi's spectrum (summed pole energies) vs the bosonic range (see from_ptree)
      auto pr          = pole_ranges_t::from(st.poles);
      const double emx = pr.p_max + pr.h_max;
      if (emx > bos->lam * (1.0 + 1e-12))
        app_log(1, "  WARNING gw_line: the G poles give Pi transitions up to {:.3f} Ha > lam_b = {:.3f} Ha: the W residues are "
                   "ill-conditioned and amplify the time-grid error in Sigma (S7c); use lam_b >= {:.1f} (or lam_b <= 0: auto)",
                emx, bos->lam, emx);
    }

    auto tic = [&](char const *nm) { Timer.start(nm); return Timer.elapsed(nm); };
    auto toc = [&](char const *nm, double e0) { Timer.stop(nm); return Timer.elapsed(nm) - e0; };

    // 1. Pi at the bosonic nodes, W residues
    double tPi = 0.0, tW = 0.0, e0 = 0.0;
    for (long G = 0; G < qg.n; ++G) {   // one group (all q) unless the Pi group does not fit
      e0 = tic("phase_Pi");
      polarization<MEM>(prop, st.poles, mf, grid, bos->zeta_nodes, *pi_p, *pi_h, prm.t_chunk, Pi, Timer, sector_t::both,
                        qg.q0(G), qg.size(G));
      tPi += toc("phase_Pi", e0);
      e0 = tic("phase_W");
      screened_interaction<MEM>(Pi, Zb, *bos, grid, mpi, w, Timer, nullptr, qg.q0(G), qplan.w_host);
      if (qplan.w_host) {   // this group's rows -> the host-resident residues of all q
        if (w_h.extent(0) != nq or w_h.extent(1) != bos->rank)
          w_h = memory::array<HOST_MEMORY, ComplexType, 4>(nq, bos->rank, grid.nP, grid.nQ);
        w_h(nda::range(qg.q0(G), qg.q0(G) + qg.size(G)), nda::range::all, nda::range::all, nda::range::all) =
           memory::to_memory_space<HOST_MEMORY>(w);
      }
      tW += toc("phase_W", e0);
    }

    // 2. Sigma per sector at the dense nodes, mixing
    e0 = tic("phase_Sigma");
    if (qplan.w_host) w = arr4_t{};   // only the last group's rows: free them
    auto const *whp = qplan.w_host ? &w_h : nullptr;
    self_energy<MEM>(prop, st.poles, w, *bos, mf, grid, mpi, zeta, *sig_p, *sig_h, prm.t_chunk, Sp_new, Timer, sector_t::particle,
                     prm.sigma_kdist, whp, qplan.gs_sigma);
    self_energy<MEM>(prop, st.poles, w, *bos, mf, grid, mpi, zeta, *sig_p, *sig_h, prm.t_chunk, Sh_new, Timer, sector_t::hole,
                     prm.sigma_kdist, whp, qplan.gs_sigma);
    if (prm.debug_noise_sigma > 0.0 and st.iter + 1 == prm.debug_noise_iter) {
      // diagnostic (noise-floor meter): relative complex Gaussian noise on the new Sigma, seeded per GLOBAL k (rank-count
      // independent), scale max|Sigma^> + Sigma^<| over all k and nodes
      double smax = nda::max_element(nda::abs(Sp_new + Sh_new));
      if (prm.sigma_kdist) smax = comm.all_reduce_value(smax, boost::mpi3::max<>{});
      for (long l = 0; l < Sp_new.extent(0); ++l) {
        const long k = prm.sigma_kdist ? kd.global(l, kd.rank) : l;
        std::mt19937_64 gen(0xabcdull + 1000003ull * std::uint64_t(prm.debug_noise_seed) + std::uint64_t(k));
        std::normal_distribution<double> N01;
        for (auto *S : {&Sp_new, &Sh_new})
          for (long iz = 0; iz < nz; ++iz)
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j < nb; ++j) (*S)(l, iz, i, j) += prm.debug_noise_sigma * smax * ComplexType(N01(gen), N01(gen));
      }
    }
    Timer.start("Sigma_mix");
    double dS = 0.0;
    if (st.have_sigma) {
      const double a = prm.mixing, b = 1.0 - prm.mixing;
      for (long ik = 0; ik < Sp_new.extent(0); ++ik)   // the rows held by this rank (all k, or the owned k)
        for (long iz = 0; iz < nz; ++iz)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) {
              const ComplexType sp = a * Sp_new(ik, iz, i, j) + b * st.Sig_p(ik, iz, i, j);
              const ComplexType sh = a * Sh_new(ik, iz, i, j) + b * st.Sig_h(ik, iz, i, j);
              dS = std::max(dS, std::abs(sp + sh - st.Sig_p(ik, iz, i, j) - st.Sig_h(ik, iz, i, j)));
              Sp_new(ik, iz, i, j) = sp;
              Sh_new(ik, iz, i, j) = sh;
            }
      if (prm.sigma_kdist) dS = comm.all_reduce_value(dS, boost::mpi3::max<>{});
    }
    st.Sig_p      = Sp_new;
    st.Sig_h      = Sh_new;
    st.have_sigma = true;
    Timer.stop("Sigma_mix");
    const double tS = toc("phase_Sigma", e0);

    // 3. closure with H_stat - mu = H0 + F - mu
    e0 = tic("phase_closure");
    nda::array<ComplexType, 3> Hrel(nk, nb, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + st.F(ik, i, j) - (i == j ? st.mu : 0.0);
    cprm.phi_prev.assign(st.phi.begin(), st.phi.end());   // S7f phase continuity (used only if phase_keep > 0)
    auto co = closure(comm, Hrel, st.Sig_p, st.Sig_h, zeta, *bp, *bh, gp, gh, cprm, nelec, Timer, grepr);
    st.phi = nda::array<double, 1>(nk);
    for (long ik = 0; ik < nk; ++ik) st.phi(ik) = co.diag[ik].phi;
    {   // S7f: how close the hard decisions of the upfolding are to flipping (closure noise floor)
      long gn = 0, sn = 0, kept = 0;
      double gm = 1e300, sm = 1e300, tie = 1e300;
      for (auto const &d : co.diag) {
        gn += d.gram_near; sn += d.svd_near; kept += d.phi_kept;
        gm = std::min(gm, d.gram_margin); sm = std::min(sm, d.svd_margin); tie = std::min(tie, d.phi_tie);
      }
      app_log(2, "          closure decisions: Gram eigenvalues within x10 of the cut {} (min margin {:.2e} dec), singular values "
                 "within x10 of tol_svd {} (min margin {:.2e} dec), closest phase-basin tie {:.3f}, phase basins kept {}",
              gn, gm, sn, sm, tie, kept);
    }
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
    {
      const double np_tot = double(st.poles.total_poles()), MB = 1024.0 * 1024.0;
      app_log(2, "          pole residues ({}): {:.0f} poles, {:.3f} MB host + the same mirrored by the propagator (matrix form "
                 "would be {:.3f} MB)",
              st.poles.is_factorized() ? "factorized nb x M" : "matrix M x nb^2", np_tot, st.poles.residue_bytes() / MB,
              np_tot * double(nb * nb) * 16.0 / MB);
    }

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
    rec.time_grid     = prm.time_grid;
    rec.nt_pi_p       = pi_p->size();
    rec.nt_pi_h       = pi_h->size();
    rec.nt_sig_p      = sig_p->size();
    rec.nt_sig_h      = sig_h->size();
    rec.g_repr        = co.repr;
    pole_counts(st.poles, rec.ng_min, rec.ng_max, rec.g_emin);
    rec.pruned_w           = co.pruned_w;
    rec.pruned_w_weight    = co.pruned_w_weight;
    rec.pruned_near        = co.pruned_near;
    rec.pruned_near_weight = co.pruned_near_weight;
    rec.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    res.history.push_back(rec);
    print_line(rec, tPi, tW, tS, tC, tF);

    Timer.start("checkpoint");
    write_state(comm, chk, st, &rec, kdp, sig_all);
    Timer.stop("checkpoint");
    Timer.stop("iteration");
    if (comm.root() and std::filesystem::exists(chk)) {
      const auto ic = std::distance(phase_names.begin(), std::find(phase_names.begin(), phase_names.end(), "checkpoint"));
      const double sf = std::filesystem::exists(sigma_file(chk)) ? double(std::filesystem::file_size(sigma_file(chk))) : 0.0;
      app_log(1, "  checkpoint {}: {:.3f} GB (+ Sigma file {:.3f} GB) after iteration {} (write {:.2f} s)", chk,
              double(std::filesystem::file_size(chk)) / 1073741824.0, sf / 1073741824.0, st.iter, Timer.elapsed("checkpoint") - ph0[ic]);
    }
    // + the pole residues mirrored by the propagator and its host XV factors (S7e; they depend on the poles of the iteration)
    report_iteration(comm, st.iter, ph0, phase_snapshot(Timer), rss0, model_host + prop.pole_bytes() + prop.xv_bytes(), model_dev,
                     MEM != HOST_MEMORY);
    {
      const double xv = comm.all_reduce_value(prop.xv_bytes(), boost::mpi3::max<>{});
      const long nxv  = comm.all_reduce_value(prop.n_xv, boost::mpi3::max<>{});
      if (MEM == HOST_MEMORY)
        app_log(2, "  propagator: G~ blocks in the XV form for {} of {} (k, sector) pairs (XV factors <= {:.3f} GB per rank)", nxv,
                2 * nk, xv / 1073741824.0);
    }

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
  for (auto nm : {"total", "H0", "bases", "time_grid", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F",
                  "checkpoint", "spectra"})
    app_log(1, "    {:<20s} {:10.3f}", nm, Timer.elapsed(nm));
  app_log(1, "  kernel sub-timers:");
  for (auto const &nm : Timer.timer_names()) {
    if (nm == "total" or nm == "H0" or nm == "bases" or nm == "time_grid" or nm == "checkpoint" or nm == "spectra" or
        nm.rfind("phase_", 0) == 0)
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
