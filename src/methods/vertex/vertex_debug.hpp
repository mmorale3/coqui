#ifndef COQUI_VERTEX_DEBUG_HPP
#define COQUI_VERTEX_DEBUG_HPP

/**
 * P23 (notes/vertex_perf_plan.md, 2026-09-21): ONE debug interface for the vertex code's diagnostic switches.
 * The switches used to be scattered environment variables (COQUI_SIGDYN_ROUTE, COQUI_DYNBSE_UNION, ...). They are now
 * keys of a process-wide registry filled from the TOML string
 *     vertex_debug = "sigdyn_route=poles, dynbse_union=0, scf_causality_meter=0"
 * (comma-separated key=value pairs; a bare key means "1"), with the historic environment variable COQUI_<KEY uppercased>
 * as the fallback when a key is not set in the TOML, so every existing recipe keeps working. Keys are documented at their
 * consumers; `vertex_debug::list()` logs the registry once at startup (MBPT_drivers).
 *
 * audit A17 (notes/AUDIT.md, 2026-10-01) -- no switch may be dropped, misread or sourced silently:
 *  - set() ABORTS on a key that is not in known_keys() below (a typo used to be stored and never read -- the switch the user
 *    asked for silently did not happen). known_keys() is THE list: every key a consumer reads (grep
 *    'vertex_debug::(number|flag|text|get)\("<key>"' and the "// vertex_debug: <key>" tags under src/, tests included) must
 *    be in it. A new consumer key that is missing here is still READ (get / number / flag / text never check the list), but
 *    cannot be set from the TOML until it is added -- the abort names the list, so the omission is loud, not silent.
 *  - keys are case-insensitive on BOTH sides: set() always lowercased, get() now lowercases too (the consumer key
 *    "keep_host_W" was unreachable from the TOML before -- set() stored "keep_host_w").
 *  - number() ABORTS on a value it cannot parse (it returned the default, so "dyn_rung_pair = off" silently kept the switch ON),
 *    and number() / flag() accept "true" / "false", "on" / "off", "yes" / "no" as 1 / 0 (flag() also aborts on garbage).
 *  - a value taken from the environment fallback COQUI_<KEY> is logged ONCE per key at level 1 as a [WARNING]: an inherited
 *    shell variable changes the run without appearing in the input file.
 *  - the registry is process-wide and every MBPT driver section calls set() with ITS OWN vertex_debug string (MBPT_drivers.cpp,
 *    one call per gw / evgw / qpgw section, never two calls meant to accumulate). set() therefore CLEARS the registry first:
 *    a later section of the same process no longer inherits the switches of an earlier one. Sections that never call set()
 *    (hf, gf2, ...) keep whatever the last vertex section set -- they read at most scf_causality_meter.
 */

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <algorithm>
#include <cctype>

#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace vertex_debug {

  /** audit A17: every vertex_debug key a consumer under src/ reads, lowercased. Keep it sorted; add a key HERE when adding
   *  a consumer (set() aborts on anything else). */
  inline std::set<std::string> const &known_keys() {
    static const std::set<std::string> k = {
        "allreduce_chunk", "cache_w_device", "cache_w_redist", "dyn_dressed", "dyn_fam_fuse", "dyn_lu_threads",
        "dyn_r1_fuse", "dyn_rung_build_rank", "dyn_rung_pair", "dyn_rung_rank", "dyn_um", "dyn_um_extra_gb",
        "dynbse_cb_gemm", "dynbse_collapse_loop", "dynbse_dev_kbig", "dynbse_dev_sigma", "dynbse_dev_unit",
        "dynbse_group_sched", "dynbse_l0_active", "dynbse_l0_asm_gemm", "dynbse_l0_device", "dynbse_l0_fused",
        "dynbse_l0_fz_bench", "dynbse_l0_fz_cfg", "dynbse_l0_pb", "dynbse_refit_loop", "dynbse_ritz", "dynbse_sq_cache",
        "dynbse_tfold", "dynbse_tkeep", "dynbse_union", "dynbse_vmask", "eps_readout_device", "eta_device",
        "inject_device", "interp_cache", "kd_blas_threads", "keep_host_w", "l2_blas_threads", "pi_hooks_device",
        "rung_rank", "scf_causality_meter", "sigma_allow_nonconserving", "sigdyn_bases_cache", "sigdyn_blas_threads", "sigdyn_families",
        "sigdyn_finish_device", "sigdyn_gram", "sigdyn_grid_tol", "sigdyn_legs", "sigdyn_route", "sigdyn_sketch",
        "sigdyn_test_nnu", "sigdyn_test_nq", "sigpair_ibz_diag", "sigpair_ibz_dump", "sym_dt", "sym_fold_conv",
        "sym_fold_dt", "sym_trev_notrans", "trev_w_check", "vertex_rdecay", "w0_dyson_device", "wbar_dump",
        "wbar_dump_exit", "wbar_load"};
    return k;
  }
  /** the known keys as one line (for the abort message) */
  inline std::string known_keys_list() {
    std::string s;
    for (auto const &k : known_keys()) s += (s.empty() ? "" : ", ") + k;
    return s;
  }

  inline std::map<std::string, std::string> &registry() { static std::map<std::string, std::string> r; return r; }

  inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  }

  /** parse "k1=v1, k2=v2, k3" into the registry. audit A17: the registry is CLEARED first (one call per driver section, see
   *  the header), and an unknown key aborts with the list of known keys. */
  inline void set(std::string const &spec) {
    auto trim = [](std::string s) {
      const auto b = s.find_first_not_of(" \t"), e = s.find_last_not_of(" \t");
      return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
    };
    registry().clear();
    size_t pos = 0;
    while (pos <= spec.size()) {
      const size_t nxt = spec.find(',', pos);
      const std::string item = trim(spec.substr(pos, (nxt == std::string::npos ? spec.size() : nxt) - pos));
      if (not item.empty()) {
        const size_t eq = item.find('=');
        const std::string key = lower(trim(eq == std::string::npos ? item : item.substr(0, eq)));
        utils::check(known_keys().count(key) > 0,
                     "vertex_debug: unknown key \"{}\" in vertex_debug = \"{}\" (audit A17: an unknown key used to be stored and "
                     "never read, i.e. the requested switch silently did not happen). Fix the spelling; the known keys are: {}.",
                     key, spec, known_keys_list());
        registry()[key] = (eq == std::string::npos) ? std::string("1") : trim(item.substr(eq + 1));
      }
      if (nxt == std::string::npos) break;
      pos = nxt + 1;
    }
  }

  /** the value of a key: the TOML registry first, then the environment variable COQUI_<KEY>. audit A17: the key is matched
   *  case-insensitively, and an environment-sourced value is logged once per key as a [WARNING]. */
  inline std::optional<std::string> get(std::string const &key_in) {
    const std::string key = lower(key_in);
    auto it = registry().find(key);
    if (it != registry().end()) return it->second;
    std::string env = "COQUI_" + key;
    std::transform(env.begin(), env.end(), env.begin(), [](unsigned char c) { return std::toupper(c); });
    if (const char *e = std::getenv(env.c_str())) {
      static std::mutex mtx;
      static std::set<std::string> logged;
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(mtx);
        first = logged.insert(key).second;
      }
      if (first)
        app_log(1, "  [WARNING] vertex_debug {} = {} taken from the environment ({})", key, std::string(e), env);
      return std::string(e);
    }
    return std::nullopt;
  }
  /** audit A17: "true" / "on" / "yes" -> 1, "false" / "off" / "no" -> 0, else a number consumed in full; nullopt = garbage */
  inline std::optional<double> parse_number(std::string const &v_in) {
    const std::string v = lower(v_in);
    if (v == "true" or v == "on" or v == "yes") return 1.0;
    if (v == "false" or v == "off" or v == "no") return 0.0;
    try {
      size_t used = 0;
      const double d = std::stod(v, &used);
      if (v.find_first_not_of(" \t", used) != std::string::npos) return std::nullopt;   // trailing garbage ("1x", "0.5,3")
      return d;
    } catch (...) {
      return std::nullopt;
    }
  }
  /** a boolean switch: set and not "0" / "false" / "off" / "no" (audit A17: an unparseable value aborts) */
  inline bool flag(std::string const &key) {
    auto v = get(key);
    if (not v) return false;
    if (v->find_first_not_of(" \t") == std::string::npos) return false;   // "key=" (empty value): off, as before
    auto d = parse_number(*v);
    utils::check(d.has_value(),
                 "vertex_debug: switch \"{}\" = \"{}\" is not a boolean (use 1 / 0, true / false, on / off, yes / no).",
                 lower(key), *v);
    return *d != 0.0;
  }
  /** a numeric switch with a default (audit A17: an unparseable value aborts instead of returning the default) */
  inline double number(std::string const &key, double dflt) {
    auto v = get(key);
    if (not v) return dflt;
    auto d = parse_number(*v);
    utils::check(d.has_value(),
                 "vertex_debug: \"{}\" = \"{}\" is not a number (true / false, on / off, yes / no are read as 1 / 0). The "
                 "default {} used to be taken silently here; fix the value.", lower(key), *v, dflt);
    return *d;
  }
  /** a string switch with a default */
  inline std::string text(std::string const &key, std::string const &dflt) {
    auto v = get(key);
    return v ? *v : dflt;
  }
  /** the registry as one line ("" when empty) */
  inline std::string list() {
    std::string s;
    for (auto const &[k, v] : registry()) s += (s.empty() ? "" : ", ") + k + "=" + v;
    return s;
  }

} // namespace vertex_debug
} // namespace methods

#endif // COQUI_VERTEX_DEBUG_HPP
