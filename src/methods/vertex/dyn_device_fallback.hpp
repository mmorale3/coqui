#ifndef COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP
#define COQUI_VERTEX_DYN_DEVICE_FALLBACK_HPP

#include <string>

namespace methods::solvers {
  /** factorize-vertex: pol_vertex_dyn_device_fallback (input, default false). A dynamic-vertex device stage that does not fit
   *  in device memory (the device-resident unit, the device rung builds, the device Sigma deposits, the device dressed-leg path)
   *  may move to the CPU ONLY when this is true, and then with a WARNING line; false aborts with the reason. Nothing moves
   *  silently. Process-wide: every dynamic-rung pass of the run reads it. */
  inline bool &dyn_device_fallback_state() {
    static bool v = false;
    return v;
  }
  /** factorize-vertex: pol_vertex_dyn_device_memory (input, default "stream"). Where the dense rung slab K_d(s) of the
   *  device-resident unit lives when the unit does not fit in device memory -- the partition is computed at run time per GPU
   *  from the problem size and the free device memory (the later device stages are sized exactly and kept free):
   *    "resident": all device-resident; a unit that does not fit is a device failure (pol_vertex_dyn_device_fallback decides);
   *    "stream"  : the reps that fit device-resident, the rest in pinned host memory streamed through two device staging
   *                buffers overlapping the rung gemms (nothing streams when it all fits);
   *    "managed" : the slab in CUDA managed memory, the part that fits device-preferred, the rest host-resident and read by the
   *                device over the link. The split is logged; the answer does not change. */
  inline std::string &dyn_device_memory_state() {
    static std::string v = "stream";
    return v;
  }
}

#endif
