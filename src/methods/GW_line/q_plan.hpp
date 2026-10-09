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

#ifndef COQUI_METHODS_GW_LINE_Q_PLAN_HPP
#define COQUI_METHODS_GW_LINE_Q_PLAN_HPP

/**
 * perf 7.4b: automatic q groups of the Pi -> W stage from the plan section 6.7 memory model (notes/line_gw_cpp_plan.md 7.4:
 * "q-groups must engage automatically when the 6.7 model exceeds the budget"; the 8x8x8 Si G0W0 iteration ran a node out
 * of memory with all q at once).
 *
 * Budgets (bytes per rank, ABOVE what the process already holds when the plan is made: MF, THC, X, node-shared arrays):
 *   host   : frac x (available memory of the node) / (ranks on the node), min over the nodes (the root's node minus what
 *            the root alone holds: the checkpoint's gather of Sigma, 3 N_k N_zeta nb^2 x 16 B). The available memory is
 *            MemAvailable of /proc/meminfo, capped by the job's cgroup (memory.max - memory.current, cgroup v2, or
 *            memory.limit_in_bytes - memory.usage_in_bytes, v1) when that is readable; measured by one rank per node at
 *            the moment of the plan (after the THC is loaded). Unknown (no /proc, e.g. macOS): no host constraint.
 *   device : frac x the effective free device memory (utils::freemem_device_effective), min over the ranks.
 *   [gw_line] overrides: mem_budget_gb (host, per rank), dev_mem_budget_gb (device, per rank), mem_frac (default 0.8),
 *   q_group_size (> 0: the largest group size g, as env COQUI_GWLINE_QGROUP, which still wins).
 *
 * Choice: g = the LARGEST group size (q_groups_t: pair-closed groups of at most g rows) whose model fits the budget(s):
 *   host runs  : model_host(g) = the 6.7 model of the kernels (aux_grid_t::model) + the driver's host arrays (Sigma at the
 *                nodes, reduce buffers, the full Z(q) of the Dyson slab of THIS grouping) <= host budget;
 *   device runs: model_dev(g) <= device budget and the pre-7.4b device rule (Pi group + 0.8 of it for the W sub-steps in
 *                80% of the free memory left by Z, w, the device Sigma accumulator) -- the smaller g of the two;
 *                model_host(g) (device: the driver arrays + host-resident residues) <= host budget.
 * Relief levels (the caller's model at level L; the driver: full BZ 1 = Sigma's real-space residues in place of w, then the
 * host t_chunk halved twice, >= 8; IBZ: the t_chunk halvings only): when no grouping fits at level L, level L + 1 is tried
 * (env COQUI_GWLINE_QPLAN_LEVEL forces one). Levels > 0 change the arithmetic at the roundoff level only (BLAS blocking).
 * What shrinks with g: the Pi group (g N_zeta blocks) and the W-stage sub-step buffers; NOT the residues w (N_q r_b blocks,
 * and their real-space copy in the Sigma stage), Z, X, the Pi / Sigma chunk arrays. If even g = 1 (smallest pair-closed
 * group) does not fit, the plan logs the shortfall and the number of ranks that would fit (blocks shrink as 1/np) and runs
 * with the smallest groups; it aborts only when the model exceeds the WHOLE available memory (frac = 1).
 * The result of the q grouping is bitwise that of all q at once (Pi rows and W of each q are independent; tested in [qplan]).
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/freemem.h"
#include "IO/app_loggers.h"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/device_blas.hpp"

namespace methods::gw_line {

struct q_plan_t {
  long g = 0, gs_sigma = 0;
  bool w_host = false;
  long level = 0;                      ///< relief level of the caller's model (0 = none; the driver: in-place w^(R), smaller t_chunk)
  bool wR_inplace = false;             ///< (set by the driver from level) Sigma's real-space residues overwrite w
  long t_chunk = 0;                    ///< (set by the driver from level) host time chunk of the kernels, 0 = default
  long ngroups = 1;                    ///< number of q groups of the Pi -> W stage
  double budget_host = -1.0;           ///< bytes per rank (< 0: unconstrained)
  double budget_dev = -1.0;
  double model_host = 0.0, model_dev = 0.0;   ///< model of the chosen g
  double model_host_all = 0.0, model_dev_all = 0.0;   ///< model of g = N_q (all q at once)
  double model_host_min = 0.0, model_dev_min = 0.0;   ///< model of the least-memory grouping
  std::string reason = "all q";
};

struct q_budget_params_t {
  double mem_budget_gb = 0.0;       ///< host bytes per rank in GB (0 = from the free memory)
  double dev_mem_budget_gb = 0.0;   ///< device, per rank (0 = from the free device memory)
  double mem_frac = 0.8;            ///< fraction of the available memory the GW_line arrays may take
  long q_group_size = 0;            ///< > 0: fixed largest group size
};

namespace detail {

/// first number of a "Key:   value kB" line of /proc/meminfo (bytes); -1 if absent
inline double meminfo_bytes(std::string const &key) {
  std::ifstream f("/proc/meminfo");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind(key + ":", 0) == 0) return 1024.0 * std::strtod(line.c_str() + key.size() + 1, nullptr);
  return -1.0;
}

/// a number from a file (cgroup); "max" or unreadable -> -1
inline double file_number(std::string const &path) {
  std::ifstream f(path);
  std::string s;
  if (not(f >> s) or s == "max") return -1.0;
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  return (end == s.c_str()) ? -1.0 : v;
}

/// the job's cgroup headroom (limit - usage, bytes), -1 if unknown or unlimited
inline double cgroup_headroom_bytes() {
  std::ifstream f("/proc/self/cgroup");
  std::string line;
  while (std::getline(f, line)) {
    const auto c1 = line.find(':'), c2 = line.find(':', c1 + 1);
    if (c1 == std::string::npos or c2 == std::string::npos) continue;
    const std::string ctl = line.substr(c1 + 1, c2 - c1 - 1), path = line.substr(c2 + 1);
    if (ctl.empty()) {   // cgroup v2
      const double lim = file_number("/sys/fs/cgroup" + path + "/memory.max");
      const double use = file_number("/sys/fs/cgroup" + path + "/memory.current");
      if (lim > 0.0 and use >= 0.0 and lim < 1e18) return lim - use;
    } else if (ctl.find("memory") != std::string::npos) {   // cgroup v1
      const double lim = file_number("/sys/fs/cgroup/memory" + path + "/memory.limit_in_bytes");
      const double use = file_number("/sys/fs/cgroup/memory" + path + "/memory.usage_in_bytes");
      if (lim > 0.0 and use >= 0.0 and lim < 1e18) return lim - use;
    }
  }
  return -1.0;
}

/// available memory of this node now (bytes): MemAvailable capped by the cgroup headroom; -1 if unknown
inline double node_available_bytes() {
  double a = meminfo_bytes("MemAvailable");
  const double c = cgroup_headroom_bytes();
  if (c > 0.0) a = (a > 0.0) ? std::min(a, c) : c;
  return a;
}

} // namespace detail

/// host budget per rank (bytes; < 0 unconstrained): frac x available / ranks on the node, min over the nodes; the node of
/// the world root also carries root_extra (bytes held by the root alone, e.g. the checkpoint's gather of Sigma). Collective.
template <typename comm_t, typename ncomm_t>
double host_budget_per_rank(comm_t &comm, ncomm_t &node_comm, double frac, double override_gb, double root_extra = 0.0) {
  if (override_gb > 0.0) return override_gb * 1073741824.0;
  double a = -1.0;
  if (node_comm.rank() == 0) a = detail::node_available_bytes();
  node_comm.broadcast_n(&a, 1, 0);
  const int has_root = node_comm.all_reduce_value(int(comm.rank() == 0), boost::mpi3::max<>{});
  double b = (a > 0.0) ? (frac * a - (has_root ? root_extra : 0.0)) / double(node_comm.size()) : 1e300;
  b        = comm.all_reduce_value(b, boost::mpi3::min<>{});
  return b >= 1e299 ? -1.0 : b;
}

/// device budget per rank (bytes; < 0 unconstrained / host build): frac x free device memory, min over the ranks
template <MEMORY_SPACE MEM, typename comm_t>
double device_budget_per_rank(comm_t &comm, double frac, double override_gb) {
  if constexpr (MEM == HOST_MEMORY) return -1.0;
  else {
    if (override_gb > 0.0) return override_gb * 1073741824.0;
    double b = frac * double(utils::freemem_device_effective()) * 1048576.0;
    return comm.all_reduce_value(b, boost::mpi3::min<>{});
  }
}

/**
 * The largest group size among `cands` (descending) whose model fits: fits(g) true. Returns {g, index}; when none fits,
 * the last (smallest) candidate with index = -1.
 */
inline std::pair<long, long> largest_fitting(std::vector<long> const &cands, std::function<bool(long)> const &fits) {
  for (long i = 0; i < long(cands.size()); ++i)
    if (fits(cands[i])) return {cands[i], i};
  return {cands.empty() ? 1L : cands.back(), -1L};
}

/// distinct effective group sizes for the rows (q_groups_t with max size g, g = ceil(n / m) for m = 1..n), descending
inline std::vector<long> q_group_candidates(std::vector<long> const &rows, std::vector<long> const &qminus) {
  const long n = long(rows.size());
  std::vector<long> c;
  long last = -1;
  for (long m = 1; m <= n; ++m) {
    const long g = (n + m - 1) / m;
    if (g == last) continue;
    q_groups_t qg(rows, g, qminus);
    const long eff = qg.max_size();
    if (eff != last and (c.empty() or eff < c.back())) c.push_back(eff);
    last = g;
  }
  if (c.empty()) c.push_back(std::max(1L, n));
  return c;
}

/**
 * The q plan (see the file header). model_host(g) / model_dev(g): the driver's model for the largest group size g
 * (bytes per rank; model_dev unused on the host). Collective (budgets).
 */
template <MEMORY_SPACE MEM, typename comm_t, typename ncomm_t>
q_plan_t choose_q_plan(comm_t &comm, ncomm_t &node_comm, [[maybe_unused]] aux_grid_t const &grid, std::vector<long> const &rows,
                       std::vector<long> const &qminus, long nk, long nz, long r_b, long nb, q_budget_params_t const &bp,
                       std::function<double(long, long)> const &model_host, std::function<double(long, long)> const &model_dev,
                       bool w_host_ok = true, long nlevels = 1, double root_extra = 0.0) {
  q_plan_t qp;
  const double GB  = 1073741824.0;
  const long nq    = long(rows.size());
  const double blk = 16.0 * double(grid.max_block_size());
  const double w   = double(nq) * r_b * blk;
  qp.g             = nq;
  qp.gs_sigma      = nq;
  qp.budget_host   = host_budget_per_rank(comm, node_comm, bp.mem_frac, bp.mem_budget_gb, root_extra);
  qp.budget_dev    = device_budget_per_rank<MEM>(comm, bp.mem_frac, bp.dev_mem_budget_gb);
  long g_dev_rule  = nq;
  if constexpr (MEM != HOST_MEMORY) {   // the pre-7.4b device rule (S7e), kept as an upper bound
    double freeb       = double(utils::freemem_device_effective()) * 1048576.0;
    freeb              = comm.all_reduce_value(freeb, boost::mpi3::min<>{});
    // host-resident residues when w takes > 35% of the free memory -- unless the caller cannot stream them (the IBZ Sigma):
    // then w stays resident and the q groups / t_chunk absorb the rest (the 80% check below guards the total)
    qp.w_host          = w_host_ok and (w > 0.35 * freeb);
    const double fixed = double(nq) * blk + (qp.w_host ? 0.0 : w) + 16.0 * double(nk) * 2.0 * nz * nb * nb + 2.0 * nk * 16.0 * blk;
    const double per_q = 1.8 * double(nz) * blk;
    g_dev_rule         = std::clamp(long((0.8 * freeb - fixed) / per_q), 1L, nq);
    qp.gs_sigma        = std::clamp(long(0.25 * freeb / (double(r_b) * blk)), 1L, nq);
    utils::check(0.8 * freeb - fixed > per_q, "gw_line: device memory: Z blocks + residues ({:.2f} GB, {}) + Sigma accumulator + "
                                              "one q of the Pi group ({:.2f} GB) exceed 80% of the free device memory ({:.2f} GB); "
                                              "use more GPUs",
                 w / GB, qp.w_host ? "on the host" : "resident", per_q / GB, freeb / GB);
    app_log(2, "  q plan (device): free {:.2f} GB (min over ranks), residues {:.2f} GB {}, fixed {:.2f} GB, Pi group per q {:.3f} GB "
               "-> device rule g <= {}",
            freeb / GB, w / GB, qp.w_host ? "on the HOST" : "resident", fixed / GB, per_q / GB, g_dev_rule);
  }
  if (long v = detail::env_long("COQUI_GWLINE_W_HOST", -1); v >= 0) qp.w_host = (v != 0);
  if constexpr (MEM == HOST_MEMORY) qp.w_host = qp.w_host and detail::env_long("COQUI_GWLINE_W_HOST", -1) == 1;

  const auto cands = q_group_candidates(rows, qminus);
  // the models of every candidate (collective: the model functions reduce over the ranks; 1e300 = infeasible grouping).
  // Memory is NOT monotonic in g: small groups shrink the Pi group and the W sub-steps but every rank then holds the full
  // Z(q) of its q in every group's Dyson slab.
  const long nc = long(cands.size());
  std::vector<double> mh(nc), md(nc, 0.0);
  auto eval = [&](long lev) {
    for (long i = 0; i < nc; ++i) {
      mh[i] = model_host(cands[i], lev);
      if constexpr (MEM != HOST_MEMORY) md[i] = model_dev(cands[i], lev);
    }
  };
  // relief levels (the caller's model at level L): the first level at which some grouping fits; env COQUI_GWLINE_QPLAN_LEVEL
  // forces a level
  const long lev_env = detail::env_long("COQUI_GWLINE_QPLAN_LEVEL", -1);
  qp.level           = (lev_env >= 0) ? std::min(lev_env, std::max(0L, nlevels - 1)) : 0;
  eval(qp.level);
  auto fits = [&](long i) {
    const bool h = (mh[i] < 1e299) and (qp.budget_host < 0.0 or mh[i] <= qp.budget_host);
    if constexpr (MEM == HOST_MEMORY) return h;
    else return h and md[i] < 1e299 and cands[i] <= g_dev_rule and (qp.budget_dev < 0.0 or md[i] <= qp.budget_dev);
  };
  // perf 7.4b: no grouping fits -> the next relief level
  if (lev_env < 0)
    while (qp.level + 1 < nlevels) {
      bool any = false;
      for (long i = 0; i < nc and not any; ++i) any = fits(i);
      if (any) break;
      eval(++qp.level);
    }
  // the candidate of least memory (host; device runs: device first)
  long imin = 0;
  for (long i = 1; i < nc; ++i) {
    const bool better = (MEM == HOST_MEMORY) ? mh[i] < mh[imin] : (md[i] < md[imin] or (md[i] == md[imin] and mh[i] < mh[imin]));
    if (better) imin = i;
  }
  qp.model_host_all = mh[0];
  qp.model_host_min = mh[imin];
  qp.model_dev_all  = md[0];
  qp.model_dev_min  = md[imin];
  long fixed_g = (bp.q_group_size > 0) ? bp.q_group_size : 0;
  if (long v = detail::env_long("COQUI_GWLINE_QGROUP", 0); v > 0) fixed_g = v;
  long ichosen = -1;
  if (fixed_g > 0) {
    qp.g      = std::min(fixed_g, nq);
    qp.reason = "fixed (q_group_size / COQUI_GWLINE_QGROUP)";
  } else {
    for (long i = 0; i < nc and ichosen < 0; ++i)
      if (fits(i)) ichosen = i;
    if (ichosen == 0) qp.reason = "all q fit";
    else if (ichosen > 0) qp.reason = "memory model";
    if (ichosen >= 0) qp.g = cands[ichosen];
    else {
      ichosen   = imin;
      qp.g      = cands[imin];
      qp.reason = "memory model: no grouping fits the budget, least-memory grouping";
      const double over_h = (qp.budget_host > 0.0) ? qp.model_host_min / qp.budget_host : 0.0;
      const double over_d = (qp.budget_dev > 0.0) ? qp.model_dev_min / qp.budget_dev : 0.0;
      const double over   = std::max(over_h, over_d);
      app_log(1, "  WARNING gw_line q plan: the least-memory grouping (<= {} rows) needs {:.2f} GB host / {:.2f} GB device per rank, "
                 "budget {:.2f} / {:.2f} GB (x{:.2f}): run on ~{:.0f}x the ranks (or nodes / GPUs)",
              qp.g, qp.model_host_min / GB, qp.model_dev_min / GB, qp.budget_host / GB, qp.budget_dev / GB, over, std::ceil(over));
      // abort only when the model exceeds the WHOLE available memory (the budget is mem_frac of it)
      if (bp.mem_budget_gb <= 0.0 and qp.budget_host > 0.0)
        utils::check(qp.model_host_min <= qp.budget_host / bp.mem_frac,
                     "gw_line: the memory model of the least-memory q grouping ({:.2f} GB per rank) exceeds the available host "
                     "memory ({:.2f} GB per rank): use more nodes", qp.model_host_min / GB, qp.budget_host / bp.mem_frac / GB);
    }
  }
  qp.ngroups    = q_groups_t(rows, qp.g, qminus).n;
  qp.model_host = (ichosen >= 0) ? mh[ichosen] : model_host(qp.g, qp.level);
  if constexpr (MEM != HOST_MEMORY) qp.model_dev = (ichosen >= 0) ? md[ichosen] : model_dev(qp.g, qp.level);
  if (long v = detail::env_long("COQUI_GWLINE_SIGMA_QGROUP", 0); v > 0) qp.gs_sigma = std::min(v, nq);
  if (not qp.w_host) qp.gs_sigma = nq;
  app_log(1, "  q plan: {} q groups of <= {} of {} rows ({}{}); model per rank: host {:.3f} GB (all q {:.3f}, least-memory grouping {:.3f}) budget {}{}",
          qp.ngroups, qp.g, nq, qp.reason, qp.level > 0 ? fmt::format("; relief level {}", qp.level) : std::string(""), qp.model_host / GB, qp.model_host_all / GB, qp.model_host_min / GB,
          qp.budget_host > 0.0 ? fmt::format("{:.3f} GB", qp.budget_host / GB) : std::string("unconstrained"),
          MEM == HOST_MEMORY ? std::string("")
                             : fmt::format("; device {:.3f} GB (all q {:.3f}) budget {:.3f} GB", qp.model_dev / GB,
                                           qp.model_dev_all / GB, qp.budget_dev / GB));
  return qp;
}

} // namespace methods::gw_line

#endif
