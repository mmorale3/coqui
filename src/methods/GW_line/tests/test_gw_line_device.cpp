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
 * GPU bring-up of the GW_line kernels (notes/line_gw_device_bringup.md): HOST_MEMORY vs DEVICE_MEMORY A/B on identical
 * inputs, setup of [V2]/[V3] (THC nIpts = 8 nbnd, KS poles, mu mid-gap, theta = 20 deg, theta_t = 10 deg, bosonic basis
 * lam_b = max(4, 1.2 x largest transition), gap = KS gap / 2, eps = 1e-10, t_chunk 8):
 *   Pi(q, zeta_i)       polarization<HOST> vs <DEVICE> (same poles, rays, nodes)
 *   W at the nodes      screened_interaction<HOST> vs <DEVICE> from the SAME Pi (host Pi copied to the device)
 *   residues            as pole sums (eval_poles, both orientations) at the nodes and 12 ray points; the raw residues are
 *                       ill-determined at ~eps cond (see screened.hpp) and only reported
 *   w_time              on the SAME residues (host w copied to the device), particle plain and hole transposed
 *   Sigma(k, zeta)      self_energy<HOST> vs <DEVICE> per sector, on the SAME residues
 *   F = V_H + Sigma_x   hartree_exchange<HOST> vs <DEVICE>
 * Gate: max|dev - host| / max|host| <= 1e-12 for each. Host and device timers are printed side by side (rank 0).
 * S7d: Pi and Sigma on the device for every Hadamard variant -- fused (default: Pi all q per launch, Sigma k-outer), fused
 * one q per launch, fused with the direct kernel variant, cuTENSOR trinary (COQUI_GWLINE_FUSED=0) -- and W with forced
 * zeta sub-slabs (COQUI_GWLINE_W_ZSUB=37: uneven sub-slabs) on the device and on the host, all against the default host.
 *
 * [.bench] (hidden; deliberately NOT tagged [gw_line], since Catch2 runs hidden tests matched by a tag filter): one
 * Pi -> W -> Sigma -> F pass in the memory space of the build (device if ENABLE_DEVICE, unless COQUI_GWLINE_BENCH_HOST=1)
 * on a larger THC: COQUI_GWLINE_BENCH_DIR / _PREFIX (h5 input, default the si_kp222_nbnd60 data set of the project),
 * COQUI_GWLINE_BENCH_NP (nIpts, default 640), COQUI_GWLINE_BENCH_TCHUNK (default 0: the kernels' automatic choice).
 * Prints per-phase times, the plan 6.7 memory model and the device high-water mark.
 * [device] uses t_chunk 8 on the host and COQUI_GWLINE_DEV_TCHUNK (default 0 = automatic) on the device.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <utility>
#include <numbers>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/freemem.h"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/closure_device.hpp"
#include "numerics/line_dlr/tests/closure_bench.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/device_blas.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
namespace mpi3 = boost::mpi3;

/// fixture + THC + KS spectrum (as the [V2]/[V3] tests)
struct dev_setup_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  double mu = 0, ks_gap = 0, etr_max = 0;
  pole_data_t poles;

  void init(long nIpts) {
    utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(), "nosym mesh required");
    utils::check(mf->nspin() == 1 and mf->npol() == 1, "spin-restricted collinear mean field required");
    thc = std::make_unique<methods::thc_reader_t>(
        mf, methods::make_thc_reader_ptree(nIpts, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
    nk = mf->nkpts(); nq = mf->nqpts(); nb = thc->nbnd(); Np = thc->Np();
    nda::array<double, 2> eig(nk, nb);
    double omax = 0.0;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        eig(ik, n) = mf->eigval()(0, ik, n);
        omax       = std::max(omax, double(mf->occ()(0, ik, n)));
      }
    double homo = -1e300, lumo = 1e300, emin_occ = 1e300, emax_vir = -1e300;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        if (mf->occ()(0, ik, n) > 0.5 * omax) { homo = std::max(homo, eig(ik, n)); emin_occ = std::min(emin_occ, eig(ik, n)); }
        else { lumo = std::min(lumo, eig(ik, n)); emax_vir = std::max(emax_vir, eig(ik, n)); }
      }
    utils::check(lumo > homo, "no KS gap");
    mu = 0.5 * (homo + lumo); ks_gap = lumo - homo; etr_max = emax_vir - emin_occ;
    poles = pole_data_t::from_ks(eig, mu);
  }
  dev_setup_t(std::string const &fixture, long nI_factor) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
    init(mf->nbnd() * nI_factor);
  }
  dev_setup_t(std::string const &outdir, std::string const &prefix, long nIpts) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source, outdir, prefix, mf::h5_input_type));
    init(nIpts);
  }
};

constexpr double deg = std::numbers::pi / 180.0;

/// sets environment variables for its lifetime (restores the previous values)
struct scoped_env_t {
  std::vector<std::pair<std::string, std::optional<std::string>>> saved;
  explicit scoped_env_t(std::vector<std::pair<std::string, std::string>> const &kv) {
    for (auto const &[k, v] : kv) {
      char const *o = std::getenv(k.c_str());
      saved.emplace_back(k, o ? std::optional<std::string>(o) : std::nullopt);
      ::setenv(k.c_str(), v.c_str(), 1);
    }
  }
  ~scoped_env_t() {
    for (auto const &[k, o] : saved) {
      if (o) ::setenv(k.c_str(), o->c_str(), 1);
      else ::unsetenv(k.c_str());
    }
  }
};

std::string env_or(char const *nm, std::string const &def) {
  char const *v = std::getenv(nm);
  return (v != nullptr and *v != '\0') ? std::string(v) : def;
}

template <typename A>
double local_max_abs(A const &a) {
  double d = 0.0;
  auto a1 = nda::reshape(a, std::array<long, 1>{a.size()});
  for (long i = 0; i < a1.size(); ++i) d = std::max(d, std::abs(a1(i)));
  return d;
}
template <typename A, typename B>
double local_max_diff(A const &a, B const &b) {
  utils::check(a.size() == b.size(), "local_max_diff: size mismatch {} vs {}", a.size(), b.size());
  double d = 0.0;
  auto a1 = nda::reshape(a, std::array<long, 1>{a.size()});
  auto b1 = nda::reshape(b, std::array<long, 1>{b.size()});
  for (long i = 0; i < a1.size(); ++i) d = std::max(d, std::abs(a1(i) - b1(i)));
  return d;
}

/// global (all ranks) relative max difference of a block-distributed (or replicated) pair
template <typename A, typename B>
double rel_diff(mpi3::communicator &comm, A const &host, B const &dev_on_host) {
  const double d = comm.all_reduce_value(local_max_diff(dev_on_host, host), mpi3::max<>{});
  const double m = comm.all_reduce_value(local_max_abs(host), mpi3::max<>{});
  return (m > 0.0) ? d / m : d;
}

#if defined(ENABLE_DEVICE)

double el(utils::TimerManager &T, std::string const &nm) { return T.elapsed(nm); }

void device_warmup() {   // cuBLAS / cuTENSOR handles and the first allocations, outside the timers
  memory::array<DEVICE_MEMORY, ComplexType, 2> a(64, 64), b(64, 64), c(64, 64);
  nda::tensor::set(ComplexType(1.0), a);
  nda::tensor::set(ComplexType(1.0), b);
  nda::blas::gemm(ComplexType(1.0), a, b, ComplexType(0.0), c);
  nda::tensor::elementwise(ComplexType(1.0), a, ComplexType(1.0), c, nda::tensor::op::MUL);
  nda::tensor::add(ComplexType(1.0), a, "ab", ComplexType(0.0), c, "ba");
  utils::device_sync();
}

void run_device_ab(std::string const &fixture) {
  dev_setup_t su(fixture, 8);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  const long nk = su.nk, nq = su.nq, nb = su.nb, Np = su.Np;
  device_warmup();

  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  bosonic_basis_t basis(theta, std::max(4.0, 1.2 * su.etr_max), 1e-10, 0.5 * su.ks_gap);
  auto const &zeta = basis.zeta_nodes;
  const long nz = zeta.size(), r = basis.rank;
  auto ray_p = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  auto fz    = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 60);
  aux_grid_t grid(mpi, Np);
  // host: t_chunk 8 (the [V1]-[V3] setting); device: COQUI_GWLINE_DEV_TCHUNK (default 0 = automatic from free memory)
  const long t_chunk = 8, t_chunk_d = std::stol(env_or("COQUI_GWLINE_DEV_TCHUNK", "0"));
  app_log(2, "\n[device A/B] {} ({} ranks, aux grid {}x{}): nk={} nq={} nb={} Np={}, bosonic rank {} ({} nodes), rays {} + {}, "
             "{} fermionic nodes, t_chunk host {} device {}",
          fixture, comm.size(), grid.np_P, grid.np_Q, nk, nq, nb, Np, r, nz, ray_p.size(), ray_h.size(), fz.size(), t_chunk,
          t_chunk_d);
  const double free0 = device_free_bytes();

  utils::TimerManager Th, Td;
  for (auto nm : {"Pi", "W", "Sigma", "HX"}) { Th.add(nm); Td.add(nm); }
  std::map<std::string, double> err;

  // ---- Pi
  propagator_t<HOST_MEMORY> ph(thc, grid);
  propagator_t<DEVICE_MEMORY> pd(thc, grid);
  memory::array<HOST_MEMORY, ComplexType, 4> Pi_h;
  memory::array<DEVICE_MEMORY, ComplexType, 4> Pi_d;
  Th.start("Pi");
  polarization<HOST_MEMORY>(ph, su.poles, mf, grid, zeta, ray_p, ray_h, t_chunk, Pi_h, Th);
  Th.stop("Pi");
  Td.start("Pi");
  polarization<DEVICE_MEMORY>(pd, su.poles, mf, grid, zeta, ray_p, ray_h, t_chunk_d, Pi_d, Td);
  utils::device_sync();
  Td.stop("Pi");
  {
    nda::array<ComplexType, 4> Pi_dh = memory::to_memory_space<HOST_MEMORY>(Pi_d);
    err["Pi(q,zeta)"] = rel_diff(comm, Pi_h, Pi_dh);
  }
  // S7d: the other Hadamard variants (timers not recorded)
  using env_list_t = std::vector<std::pair<std::string, std::string>>;
  const std::vector<std::pair<std::string, env_list_t>> hvariants = {
      {"fused 1q/launch", {{"COQUI_GWLINE_PI_QFOLD", "0"}, {"COQUI_GWLINE_SIGMA_KOUTER", "0"}}},
      {"fused direct", {{"COQUI_GWLINE_FUSED_VARIANT", "1"}}},
      {"cuTENSOR", {{"COQUI_GWLINE_FUSED", "0"}}}};
  for (auto const &[nm, kv] : hvariants) {
    scoped_env_t env(kv);
    utils::TimerManager Tv;
    polarization<DEVICE_MEMORY>(pd, su.poles, mf, grid, zeta, ray_p, ray_h, t_chunk_d, Pi_d, Tv);
    nda::array<ComplexType, 4> Pi_dh = memory::to_memory_space<HOST_MEMORY>(Pi_d);
    err["Pi(q,zeta) [" + nm + "]"] = rel_diff(comm, Pi_h, Pi_dh);
  }
  Pi_d = memory::array<DEVICE_MEMORY, ComplexType, 4>{};

  // ---- W from the SAME Pi
  dyson_layout_t lay(mpi, nq, nz, Np);
  coulomb_blocks_t<HOST_MEMORY> Zb_h(thc, grid, lay.q_rng(), Th);
  coulomb_blocks_t<DEVICE_MEMORY> Zb_d(thc, grid, lay.q_rng(), Td);
  memory::array<HOST_MEMORY, ComplexType, 4> w_h, Wn_h;
  memory::array<DEVICE_MEMORY, ComplexType, 4> w_d, Wn_d;
  {
    memory::array<HOST_MEMORY, ComplexType, 4> Pin_h(Pi_h);
    memory::array<DEVICE_MEMORY, ComplexType, 4> Pin_d = memory::to_memory_space<DEVICE_MEMORY>(Pi_h);
    Th.start("W");
    screened_interaction<HOST_MEMORY>(Pin_h, Zb_h, basis, grid, mpi, w_h, Th, &Wn_h);
    Th.stop("W");
    Td.start("W");
    screened_interaction<DEVICE_MEMORY>(Pin_d, Zb_d, basis, grid, mpi, w_d, Td, &Wn_d);
    utils::device_sync();
    Td.stop("W");
  }
  {
    nda::array<ComplexType, 4> Wn_dh = memory::to_memory_space<HOST_MEMORY>(Wn_d);
    err["W(q,zeta_i) nodes"] = rel_diff(comm, Wn_h, Wn_dh);
  }
  Wn_d = memory::array<DEVICE_MEMORY, ComplexType, 4>{};
  {   // S7d: forced (uneven) zeta sub-slabs of the W stage, device and host, vs the default host W
    scoped_env_t env({{"COQUI_GWLINE_W_ZSUB", "37"}});
    utils::TimerManager Tv;
    memory::array<HOST_MEMORY, ComplexType, 4> Pz_h(Pi_h), wz_h, Wz_h;
    memory::array<DEVICE_MEMORY, ComplexType, 4> Pz_d = memory::to_memory_space<DEVICE_MEMORY>(Pi_h), wz_d, Wz_d;
    screened_interaction<HOST_MEMORY>(Pz_h, Zb_h, basis, grid, mpi, wz_h, Tv, &Wz_h);
    screened_interaction<DEVICE_MEMORY>(Pz_d, Zb_d, basis, grid, mpi, wz_d, Tv, &Wz_d);
    nda::array<ComplexType, 4> Wz_dh = memory::to_memory_space<HOST_MEMORY>(Wz_d);
    err["W(q,zeta_i) nodes [zsub 37, device]"] = rel_diff(comm, Wn_h, Wz_dh);
    err["W(q,zeta_i) nodes [zsub 37, host]"]   = rel_diff(comm, Wn_h, Wz_h);
  }
  // residues as pole sums at the nodes + 12 ray points, both orientations; device residues evaluated by the device kernel
  {
    auto rr = numerics::line_dlr::detail::logspace(0.05, 5.0, 6);
    nda::array<ComplexType, 1> zp(nz + 12);
    zp(nda::range(0, nz)) = zeta;
    for (long i = 0; i < 6; ++i) {
      zp(nz + i)     = rr(i) * std::exp(ComplexType(0.0, theta));
      zp(nz + 6 + i) = rr(i) * std::exp(ComplexType(0.0, std::numbers::pi - theta));
    }
    const long n = zp.size();
    double d = 0.0, m = 0.0;
    memory::array<HOST_MEMORY, ComplexType, 3> Eh(n, grid.nP, grid.nQ);
    memory::array<DEVICE_MEMORY, ComplexType, 3> Ed(n, grid.nP, grid.nQ);
    for (long iq = 0; iq < nq; ++iq)
      for (auto s : {sector_t::particle, sector_t::hole}) {
        const bool tr = (s == sector_t::hole);
        const long iqm = mf.qminus()(iq);   // W^<(q)^T carries the residues of -q (screened.hpp)
        eval_poles<HOST_MEMORY>(w_h, basis, iq, iqm, zp, s, tr, Eh());
        eval_poles<DEVICE_MEMORY>(w_d, basis, iq, iqm, zp, s, tr, Ed());
        nda::array<ComplexType, 3> Edh = memory::to_memory_space<HOST_MEMORY>(Ed);
        d = std::max(d, local_max_diff(Edh, Eh));
        m = std::max(m, local_max_abs(Eh));
      }
    d = comm.all_reduce_value(d, mpi3::max<>{});
    m = comm.all_reduce_value(m, mpi3::max<>{});
    err["W residues as pole sums"] = d / m;
    nda::array<ComplexType, 4> w_dh = memory::to_memory_space<HOST_MEMORY>(w_d);
    const double raw = rel_diff(comm, w_h, w_dh);
    app_log(2, "  raw residues max|w_dev - w_host| / max|w_host| = {:.2e} (info: ill-determined at ~eps cond)", raw);
  }

  // ---- w_time on the SAME residues
  memory::array<DEVICE_MEMORY, ComplexType, 4> w_hd = memory::to_memory_space<DEVICE_MEMORY>(w_h);
  {
    double d = 0.0, m = 0.0;
    for (auto const *ray : {&ray_p, &ray_h}) {
      const auto s  = ray->sector;
      const bool tr = (s == sector_t::hole);
      const long nt = ray->size();
      memory::array<HOST_MEMORY, ComplexType, 3> Wh(nt, grid.nP, grid.nQ);
      memory::array<DEVICE_MEMORY, ComplexType, 3> Wd(nt, grid.nP, grid.nQ);
      for (long iq = 0; iq < nq; ++iq) {
        const long iqm = mf.qminus()(iq);
        w_time<HOST_MEMORY>(w_h, basis, iq, iqm, ray->t, s, tr, Wh());
        w_time<DEVICE_MEMORY>(w_hd, basis, iq, iqm, ray->t, s, tr, Wd());
        nda::array<ComplexType, 3> Wdh = memory::to_memory_space<HOST_MEMORY>(Wd);
        d = std::max(d, local_max_diff(Wdh, Wh));
        m = std::max(m, local_max_abs(Wh));
      }
    }
    d = comm.all_reduce_value(d, mpi3::max<>{});
    m = comm.all_reduce_value(m, mpi3::max<>{});
    err["w_time (both sectors)"] = d / m;
  }

  // ---- Sigma per sector on the SAME residues
  {
    nda::array<ComplexType, 4> Sp_h, Sh_h, Sp_d, Sh_d;
    Th.start("Sigma");
    self_energy<HOST_MEMORY>(ph, su.poles, w_h, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sp_h, Th, sector_t::particle);
    self_energy<HOST_MEMORY>(ph, su.poles, w_h, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sh_h, Th, sector_t::hole);
    Th.stop("Sigma");
    Td.start("Sigma");
    self_energy<DEVICE_MEMORY>(pd, su.poles, w_hd, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk_d, Sp_d, Td, sector_t::particle);
    self_energy<DEVICE_MEMORY>(pd, su.poles, w_hd, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk_d, Sh_d, Td, sector_t::hole);
    utils::device_sync();
    Td.stop("Sigma");
    err["Sigma^> (particle)"] = rel_diff(comm, Sp_h, Sp_d);
    err["Sigma^< (hole)"]     = rel_diff(comm, Sh_h, Sh_d);
    for (auto const &[nm, kv] : hvariants) {   // S7d: the other Hadamard variants
      scoped_env_t env(kv);
      utils::TimerManager Tv;
      self_energy<DEVICE_MEMORY>(pd, su.poles, w_hd, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk_d, Sp_d, Tv, sector_t::particle);
      self_energy<DEVICE_MEMORY>(pd, su.poles, w_hd, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk_d, Sh_d, Tv, sector_t::hole);
      err["Sigma^> (particle) [" + nm + "]"] = rel_diff(comm, Sp_h, Sp_d);
      err["Sigma^< (hole) [" + nm + "]"]     = rel_diff(comm, Sh_h, Sh_d);
    }
  }

  // ---- F = V_H + Sigma_x
  {
    auto D = density_matrix(su.poles);
    nda::array<ComplexType, 3> F_h, F_d;
    Th.start("HX");
    hartree_exchange<HOST_MEMORY>(ph, Zb_h, D, mf, grid, mpi, F_h, Th);
    Th.stop("HX");
    Td.start("HX");
    hartree_exchange<DEVICE_MEMORY>(pd, Zb_d, D, mf, grid, mpi, F_d, Td);
    utils::device_sync();
    Td.stop("HX");
    err["F = V_H + Sigma_x"] = rel_diff(comm, F_h, F_d);
  }
  const double free1 = device_free_bytes();

  app_log(2, "  [device A/B] {} ({} ranks): max|dev - host| / max|host|", fixture, comm.size());
  for (auto const &[k, v] : err) app_log(2, "    {:<28s} {:.2e}", k, v);
  app_log(2, "  [device A/B] {} timers (rank 0), s:      host      device", fixture);
  for (auto nm : {"Pi", "G_tilde", "Pi_hadamard", "Pi_transform", "W", "Z_gather", "W_redistribute", "W_dyson", "W_fit",
                  "Sigma", "Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce",
                  "Sigma_transform", "HX", "HX_density", "HX_hartree", "HX_exchange", "HX_allreduce"})
    app_log(2, "    {:<18s} {:9.4f} {:11.4f}", nm, el(Th, nm), el(Td, nm));
  app_log(2, "  device free memory at start / end: {:.3f} / {:.3f} GB", free0 / 1073741824.0, free1 / 1073741824.0);
  for (auto const &[k, v] : err) {
    INFO(k);
    REQUIRE(v <= 1e-12);
  }
}

TEST_CASE("gw_line_device_lih222", "[gw_line][device]") { run_device_ab("qe_lih222"); }

TEST_CASE("gw_line_device_si211", "[gw_line][device]") { run_device_ab("qe_si211"); }

// q != -q mesh (2x2x3): the paired W fit (pair pass of screened_interaction) and the hole-sector residue rows of -q
TEST_CASE("gw_line_device_lih223", "[gw_line][device]") { run_device_ab("qe_lih223"); }

#endif   // ENABLE_DEVICE

// ------------------------------------------------------------------------------------------------------------- [.bench]
template <MEMORY_SPACE MEM>
void run_bench() {
  const std::string dir    = env_or("COQUI_GWLINE_BENCH_DIR", "/mnt/ceph/users/mmorales/Cayley_real_axis_scGW/data/si_kp222_nbnd60");
  const std::string prefix = env_or("COQUI_GWLINE_BENCH_PREFIX", "si");
  const long nIpts         = std::stol(env_or("COQUI_GWLINE_BENCH_NP", "640"));
  const long tc_in         = std::stol(env_or("COQUI_GWLINE_BENCH_TCHUNK", "0"));   // 0: automatic
  utils::TimerManager T;
  for (auto nm : {"setup", "Pi", "Z", "W", "Sigma", "HX"}) T.add(nm);
  T.start("setup");
  dev_setup_t su(dir, prefix, nIpts);
  T.stop("setup");
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  const long nk = su.nk, nq = su.nq, nb = su.nb, Np = su.Np;
  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  bosonic_basis_t basis(theta, std::max(4.0, 1.2 * su.etr_max), 1e-10, 0.5 * su.ks_gap);
  auto const &zeta = basis.zeta_nodes;
  const long nz = zeta.size(), r = basis.rank;
  auto ray_p = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  auto fz    = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
  aux_grid_t grid(mpi, Np);
  const long t_chunk = tc_in;
  app_log(2, "\n[bench] {} {} ({} ranks, {}): nk={} nq={} nb={} Np={}, bosonic rank {} ({} nodes), rays {} + {}, {} fermionic "
             "nodes, t_chunk {}",
          dir, prefix, comm.size(), MEM == HOST_MEMORY ? "HOST" : "DEVICE", nk, nq, nb, Np, r, nz, ray_p.size(), ray_h.size(),
          fz.size(), t_chunk);
  // the plan 6.7 memory model at the chunk the kernels will pick (automatic: same rule as polarization)
  const bool dev_fused = (MEM != HOST_MEMORY) and detail::fused_hadamard();
  double model_peak    = 0.0;
  {
    const long nacc = (dev_fused and detail::env_long("COQUI_GWLINE_PI_QFOLD", 1) != 0) ? nq : 1;
    const long tcm  = (t_chunk > 0) ? t_chunk
                                    : detail::auto_t_chunk<MEM>(ray_p.size(), double(2 * nk + nacc) * grid.max_block_size() * 16.0);
    model_peak = grid.log(nk, nq, nz, r, tcm, nb, -1, dev_fused);
  }
  [[maybe_unused]] const double free0 = device_free_bytes();
  device_mem_reset();

  propagator_t<MEM> prop(thc, grid);
  memory::array<MEM, ComplexType, 4> Pi, w;
  [[maybe_unused]] double hw_stage[3] = {0.0, 0.0, 0.0};   // device high-water per stage (Pi, Z + W, Sigma + HX)
  T.start("Pi");
  polarization<MEM>(prop, su.poles, mf, grid, zeta, ray_p, ray_h, t_chunk, Pi, T);
  utils::device_sync();
  T.stop("Pi");
  if constexpr (MEM != HOST_MEMORY) { device_mem_probe(); hw_stage[0] = device_high_water_bytes(); device_mem_hw_restart(); }
  dyson_layout_t lay(mpi, nq, nz, Np);
  lay.log();
  T.start("Z");
  coulomb_blocks_t<MEM> Zb(thc, grid, lay.q_rng(), T);
  T.stop("Z");
  T.start("W");
  screened_interaction<MEM>(Pi, Zb, basis, grid, mpi, w, T);
  utils::device_sync();
  T.stop("W");
  if constexpr (MEM != HOST_MEMORY) { device_mem_probe(); hw_stage[1] = device_high_water_bytes(); device_mem_hw_restart(); }
  nda::array<ComplexType, 4> S;
  T.start("Sigma");
  self_energy<MEM>(prop, su.poles, w, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, S, T);
  utils::device_sync();
  T.stop("Sigma");
  nda::array<ComplexType, 3> F;
  T.start("HX");
  hartree_exchange<MEM>(prop, Zb, density_matrix(su.poles), mf, grid, mpi, F, T);
  utils::device_sync();
  T.stop("HX");

  // per-phase times: max over ranks
  app_log(2, "  [bench] per-phase wall times (max over ranks), s:");
  for (auto nm : {"setup", "Pi", "G_tilde", "Pi_hadamard", "Pi_transform", "Z", "W", "W_redistribute", "W_dyson", "W_fit",
                  "Sigma", "Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce",
                  "Sigma_transform", "HX", "HX_density", "HX_hartree", "HX_exchange", "HX_allreduce"}) {
    const double t = comm.all_reduce_value(T.elapsed(nm), mpi3::max<>{});
    app_log(2, "    {:<18s} {:10.3f}", nm, t);
  }
  app_log(2, "  [bench] checksum: max|Sigma| {:.6e}, max|F| {:.6e}", local_max_abs(S), local_max_abs(F));
  if constexpr (MEM != HOST_MEMORY) {
    const double free1 = device_free_bytes();
    app_log(2, "  [bench] device free memory at start / end: {:.3f} / {:.3f} GB; kernel high-water (rank 0): {:.3f} GB", free0 / 1073741824.0,
            free1 / 1073741824.0, device_high_water_bytes() / 1073741824.0);   // (Sigma stage: restarted after Pi and W)
    device_mem_probe();
    hw_stage[2] = device_high_water_bytes();
    for (auto &h : hw_stage) h = comm.all_reduce_value(h, mpi3::max<>{});
    const double hw_max = std::max({hw_stage[0], hw_stage[1], hw_stage[2]});
    app_log(2, "  [bench] device memory per rank: model (plan 6.7, proc_grid.hpp) {:.3f} GB vs measured high-water (max over "
               "ranks) {:.3f} GB: Pi stage {:.3f}, W stage {:.3f}, Sigma stage {:.3f} GB", model_peak / 1073741824.0,
            hw_max / 1073741824.0, hw_stage[0] / 1073741824.0, hw_stage[1] / 1073741824.0, hw_stage[2] / 1073741824.0);
  }
}

// S7g: the closure's dense linear algebra through the cuSOLVER hooks (device builds) vs the host python path
TEST_CASE("gw_line_closure_device_ab", "[gw_line][device]") {
  auto const *h0 = methods::gw_line::device_lapack_hooks(0);
  if (not h0) return;   // host build: nothing to compare
  // backward errors of the cuSOLVER drivers (the closure itself amplifies ANY roundoff difference in the Gram eigenvectors
  // ~1e5-1e6 x on these exact-moment models, so the end-to-end A/B below is a sanity bound only)
  CHECK(closure_bench::hooks_accuracy(h0, 300) <= 1e-12);
  CHECK(closure_bench::hooks_accuracy(methods::gw_line::device_lapack_hooks(1), 300) <= 1e-12);
  CHECK(closure_bench::ab_small(h0, "gesvd") <= 1e-6);
  CHECK(closure_bench::ab_small(methods::gw_line::device_lapack_hooks(1), "gesvd") <= 1e-6);
}

// S7g (hidden): the closure profile at production size with the device hooks (closure_bench.hpp; run with 1 rank)
TEST_CASE("gw_line_closure_bench_dev", "[.closure_bench_dev]") {
  closure_bench::run(methods::gw_line::device_lapack_hooks(0), methods::gw_line::device_lapack_hooks(1));
}

} // namespace

TEST_CASE("gw_line_bench", "[.bench]") {
#if defined(ENABLE_DEVICE)
  if (env_or("COQUI_GWLINE_BENCH_HOST", "0") != "1") {
    run_bench<DEVICE_MEMORY>();
    return;
  }
#endif
  run_bench<HOST_MEMORY>();
}
