#ifndef COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP
#define COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP

namespace methods::solvers {
  /** factorize-vertex: pol_vertex_dyn_device_fallback (input, default false). A dynamic-vertex device stage that does not fit
   *  in device memory (the device-resident unit, the device rung builds, the device Sigma deposits, the device dressed-leg path)
   *  may move to the CPU ONLY when this is true, and then with a WARNING line; false aborts with the reason. Nothing moves
   *  silently. Process-wide: every dynamic-rung pass of the run reads it. */
  inline bool &dyn_device_fallback_state() {
    static bool v = false;
    return v;
  }
}

#endif
