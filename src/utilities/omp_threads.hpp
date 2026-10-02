#ifndef UTILITIES_OMP_THREADS_HPP
#define UTILITIES_OMP_THREADS_HPP

/**
 * @file omp_threads.hpp
 *
 * The `omp_threads` knob: the thread count for CoQui's OWN OpenMP regions.
 *
 * WHY THIS EXISTS. `blas_threads` threads the BLAS layer underneath CoQui. It does nothing
 * for the phases that are CoQui's own arithmetic, e.g. the AC/Pade quasiparticle map, which
 * is dominated by one per-state loop of software multiprecision arithmetic with no BLAS in
 * it at all. `omp_threads` is the knob that gives those loops threads.
 *
 * WHY IT IS **NOT** `OMP_NUM_THREADS`. Identical reasoning to blas_threads.hpp, and it is a
 * hard requirement rather than a preference: raising the global OpenMP thread count
 * activates SLATE's libgomp task layer, whose tasks issue MPI calls, which crashes (e.g. a
 * UCX SIGSEGV) below MPI_THREAD_MULTIPLE (see slate_ops.hpp's guard). So every region
 * carries an explicit `num_threads(...)` clause fed from THIS knob, and OMP_NUM_THREADS
 * stays 1. The three thread controls are deliberately independent:
 *
 *     OMP_NUM_THREADS = 1       -- SLATE's task layer stays serial (never raised)
 *     blas_threads    = N       -- threads inside MKL/OpenBLAS
 *     omp_threads     = M       -- threads inside CoQui's own regions (this knob)
 *
 * NESTED-BLAS COMPOSITION. None of the regions driven by this knob contains a BLAS call:
 * the Pade fit/evaluate is scalar multiprecision arithmetic, and the Sigma Hadamard region
 * is elementwise nda expressions. So `omp_threads` x `blas_threads` oversubscription cannot
 * arise from them, and no `mkl_set_num_threads_local` guard is needed. Any region that does
 * call BLAS must revisit this.
 *
 * DETERMINISM. `omp_threads` defaults to 1 and an absent key is exactly 1, in which case
 * every region runs its serial path, bitwise identical to a build without these regions.
 *
 * ABOVE 1, no region reassociates anything: the Pade fit and the two per-state evaluate
 * loops are independent per state, and the Sigma Hadamard is blocked over the OUTPUT index
 * with fixed block boundaries, so each element's accumulation order over (isk, iq) is
 * untouched. Caveat: at high thread counts (8 and above) a 1-2 ulp residue has been observed
 * in Heff that does not come from these regions (it persists with MKL_NUM_THREADS = 1 and
 * with the AC replaced); the ac_pade quasiparticle map amplifies such a residue by many
 * orders of magnitude (the Matsubara-native map does not). For bitwise-reproducible ac_pade
 * runs keep omp_threads moderate (<= 6).
 */

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace utils {

/// Requested CoQui-side OpenMP thread count. 1 = serial, which is the default.
inline long &omp_threads_state() { static long n = 1; return n; }

/// Read-only accessor (used by the regions, the startup echo and chkpt::write_metadata).
inline long omp_threads() { return omp_threads_state(); }

/// Whether this binary was built with host OpenMP support at all.
inline bool omp_available() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

/**
 * Thread count for a region of `niter` independent iterations: never more threads than
 * there is work, never more than the knob, never fewer than 1.
 *
 * Every CoQui OpenMP region takes its `num_threads(...)` argument from here, so a single knob
 * value drives all of them and a region with one iteration is automatically serial.
 */
inline long omp_threads_for(long niter) {
  const long n = omp_threads_state();
  if (n <= 1 or niter <= 1) return 1;
  return std::min(n, niter);
}

/**
 * Apply and record the `omp_threads` setting. `n <= 1` (including an absent key, which the
 * caller reads as the default 1) is the serial behaviour.
 *
 * This deliberately does NOT call omp_set_num_threads(): that would change the ambient
 * thread count for EVERY parallel region in the process, SLATE's included, which is the
 * exact failure mode the whole blas_threads/omp_threads split exists to avoid.
 */
inline void set_omp_threads(long n) {
  utils::check(n >= 0, "omp_threads must be >= 0 (0 or 1 = serial, CoQui's own loops are "
                       "not threaded); got {}.", n);
  if (n == 0) n = 1;
  if (n > 1 and not omp_available()) {
    app_warning("omp_threads = {} was requested but this binary was built without host "
                "OpenMP support (COQUI_ENABLE_HOST_OPENMP=OFF, or the compiler has no "
                "OpenMP). CoQui's own loops stay serial; reconfigure with "
                "-DCOQUI_ENABLE_HOST_OPENMP=ON to use this knob.", n);
    n = 1;
  }
  omp_threads_state() = n;
  if (n > 1)
    app_log(2, "  omp_threads = {} applied to CoQui's own parallel regions via explicit "
               "num_threads() clauses (OMP_NUM_THREADS is deliberately NOT touched: "
               "raising it would activate SLATE's OpenMP task layer, which needs "
               "MPI_THREAD_MULTIPLE).", n);
}

} // namespace utils

#endif // UTILITIES_OMP_THREADS_HPP
