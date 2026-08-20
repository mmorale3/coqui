#ifndef UTILITIES_OMP_THREADS_HPP
#define UTILITIES_OMP_THREADS_HPP

/**
 * @file omp_threads.hpp
 *
 * The `omp_threads` knob -- increment T-3b of notes/coqui_threading_spec.md (rev 3
 * section 7.2 item 1), scoped by FABLE RULING R-T3-1 in notes/coqui_threading_t3a.md
 * section 3.6.
 *
 * WHY THIS EXISTS. `blas_threads` (T-2) threads the BLAS layer underneath CoQui. It does
 * nothing for the phases that are CoQui's OWN arithmetic, and T-3a measured those to be
 * where the parity deficit against an all-MPI layout actually lives: the AC/Pade QP map is
 * 3.89x off parity at 24 ranks x 4 threads and 7.50x at 12 x 8 (t3a section 1), and 84% of
 * it is one per-state loop of software multiprecision arithmetic with no BLAS in it at all.
 * `omp_threads` is the knob that gives those loops threads.
 *
 * WHY IT IS **NOT** `OMP_NUM_THREADS`. Identical reasoning to blas_threads.hpp, and it is a
 * hard requirement rather than a preference: raising the global OpenMP thread count
 * activates SLATE's libgomp task layer, whose tasks issue MPI calls, which is an immediate
 * UCX SIGSEGV below MPI_THREAD_MULTIPLE (job 6890365; see slate_ops.hpp's guard). So every
 * T-3b region carries an explicit `num_threads(...)` clause fed from THIS knob, and
 * OMP_NUM_THREADS stays 1 in every launch harness. The three thread controls are
 * deliberately independent:
 *
 *     OMP_NUM_THREADS = 1       -- SLATE's task layer stays serial (never raised)
 *     blas_threads    = N       -- threads inside MKL/OpenBLAS
 *     omp_threads     = M       -- threads inside CoQui's own regions (this knob)
 *
 * NESTED-BLAS COMPOSITION (spec section 7.2 item 3). None of the R-T3-1 regions contains a
 * BLAS call: the Pade fit/evaluate is scalar multiprecision arithmetic, and the Sigma
 * Hadamard region is elementwise nda expressions. So `omp_threads` x `blas_threads`
 * oversubscription cannot arise from the regions this increment adds, and no
 * `mkl_set_num_threads_local` guard is needed. Any FUTURE region that does call BLAS must
 * revisit this.
 *
 * DETERMINISM (spec section 7.2 item 4). `omp_threads` defaults to 1 and an absent key is
 * exactly 1, in which case every region runs its serial path -- the same statements in the
 * same order as before this increment. That is the gated claim, and it is measured bitwise
 * (Si kp444 nb100 A, 96 r x MKL1 / 24 r x MKL4 / 12 r x MKL8: knob absent and knob = 1 both
 * reproduce the pre-T-3b binary's Heff, E_ska, Dm and mu bit for bit).
 *
 * ABOVE 1, no region reassociates anything: the Pade fit and the two per-state evaluate
 * loops are independent per state, and the Sigma Hadamard is blocked over the OUTPUT index
 * with fixed block boundaries, so each element's accumulation order over (isk, iq) is
 * untouched. Measured consequence on that fixture: omp_threads = 2, 4 and 6 are bitwise
 * identical to serial and reproducible run to run. ⚠ At omp_threads = 8 a 1-2 ulp residue
 * appears in Heff, and the ac_pade quasiparticle map amplifies it by ~2e11 to 2.3 meV
 * (the Matsubara-native map keeps it at 1e-15). The residue is not one of these regions'
 * reduction order -- it is present with MKL_NUM_THREADS = 1 and with the AC replaced -- and a
 * further run-to-run bistability appears only when MKL is ALSO threaded at 8, i.e. it belongs
 * to the T-4 MKL study. Until that closes, omp_threads <= 6 is the reproducible regime for
 * ac_pade workloads. See notes/coqui_threading_spec.md section 7.4 and the T-3b report.
 */

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace utils {

/// Requested CoQui-side OpenMP thread count. 1 = serial, which is the default and the
/// behaviour of every input file written before T-3b.
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
 * Every T-3b region takes its `num_threads(...)` argument from here, so a single knob
 * value drives all of them and a region with one iteration is automatically serial.
 */
inline long omp_threads_for(long niter) {
  const long n = omp_threads_state();
  if (n <= 1 or niter <= 1) return 1;
  return std::min(n, niter);
}

/**
 * Apply and record the `omp_threads` setting. `n <= 1` (including an absent key, which the
 * caller reads as the default 1) is the pre-T-3b serial behaviour.
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
