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
 */

#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <algorithm>
#include <cctype>

namespace methods {
namespace vertex_debug {

  inline std::map<std::string, std::string> &registry() { static std::map<std::string, std::string> r; return r; }

  /** parse "k1=v1, k2=v2, k3" into the registry (later calls override earlier ones) */
  inline void set(std::string const &spec) {
    auto trim = [](std::string s) {
      const auto b = s.find_first_not_of(" \t"), e = s.find_last_not_of(" \t");
      return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
    };
    size_t pos = 0;
    while (pos <= spec.size()) {
      const size_t nxt = spec.find(',', pos);
      const std::string item = trim(spec.substr(pos, (nxt == std::string::npos ? spec.size() : nxt) - pos));
      if (not item.empty()) {
        const size_t eq = item.find('=');
        std::string key = trim(eq == std::string::npos ? item : item.substr(0, eq));
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
        registry()[key] = (eq == std::string::npos) ? std::string("1") : trim(item.substr(eq + 1));
      }
      if (nxt == std::string::npos) break;
      pos = nxt + 1;
    }
  }

  /** the value of a key: the TOML registry first, then the environment variable COQUI_<KEY> */
  inline std::optional<std::string> get(std::string const &key) {
    auto it = registry().find(key);
    if (it != registry().end()) return it->second;
    std::string env = "COQUI_" + key;
    std::transform(env.begin(), env.end(), env.begin(), [](unsigned char c) { return std::toupper(c); });
    if (const char *e = std::getenv(env.c_str())) return std::string(e);
    return std::nullopt;
  }
  /** a boolean switch: set and not "0" / "false" / "off" */
  inline bool flag(std::string const &key) {
    auto v = get(key);
    if (not v) return false;
    std::string s = *v;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return not (s == "0" or s == "false" or s == "off" or s.empty());
  }
  /** a numeric switch with a default */
  inline double number(std::string const &key, double dflt) {
    auto v = get(key);
    if (not v) return dflt;
    try { return std::stod(*v); } catch (...) { return dflt; }
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
