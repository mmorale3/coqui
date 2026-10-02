#ifndef COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP
#define COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP

#include <string>

namespace methods::solvers {
  /** pol_vertex_dyn_device_fallback (input, default false). A dynamic-vertex device stage that does not fit
   *  in device memory (the device-resident unit, the device rung builds, the device Sigma deposits, the device dressed-leg path)
   *  may move to the CPU ONLY when this is true, and then with a WARNING line; false aborts with the reason. Nothing moves
   *  silently. Process-wide: every dynamic-rung pass of the run reads it. */
  inline bool &dyn_device_fallback_state() {
    static bool v = false;
    return v;
  }
  /** pol_vertex_dyn_device_memory (input, default "auto"): how the dense dynamic rungs K_d(s) of the device-resident unit
   *  use device memory. The resident count is chosen at run time per GPU from the problem size and the free memory (the
   *  later device stages are sized exactly and kept free); the rungs that do not fit reach the device per rung application:
   *    "auto"    : rebuilt on the device from the W tables and / or copied from pinned host memory, the split chosen from the
   *                measured costs (copies and rebuilds overlap); copies only when the device rung builds are unavailable;
   *    "rebuild" : rebuilt on the device only (needs the device rung builds: full k meshes);
   *    "host"    : copied from pinned host memory only;
   *    "resident": every rung resident; a unit that does not fit is a device failure (pol_vertex_dyn_device_fallback decides).
   *  The answer does not depend on the choice; the resident count and the split are logged. */
  inline std::string &dyn_device_memory_state() {
    static std::string v = "auto";
    return v;
  }
  /** pol_vertex_dyn_dressed (input, default "auto"). The dressed-leg Gamma_1 readout (exact: the static ladder is moved
   *  onto the frequency-independent legs, so the one-dynamic-rung term needs one rung contraction with dressed legs):
   *    "auto": on every dynamic pass that supports it (the Gamma_1-only and one-bare-rung passes, the Sigma columns dyn1 /
   *            dyn1_bare / static_dyn); a pass that does not (the GMRES resummation, the Sigma column dyn) runs the standard
   *            path and logs that it does;
   *    "on"  : required -- a dynamic pass that cannot use it aborts;
   *    "off" : the standard path everywhere.
   *  vertex_debug dyn_dressed = 1 | 0 overrides it with "on" | "off". */
  inline std::string &dyn_dressed_state() {
    static std::string v = "auto";
    return v;
  }
}

#endif
