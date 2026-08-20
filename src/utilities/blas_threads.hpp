#ifndef UTILITIES_BLAS_THREADS_HPP
#define UTILITIES_BLAS_THREADS_HPP

/**
 * @file blas_threads.hpp
 *
 * The `blas_threads` knob -- increment T-2 option (c) of
 * notes/coqui_threading_spec.md (rev 2 section 3), ruled from the T-0 measurements in
 * notes/coqui_threading_t0.md section 4.
 *
 * WHY THIS EXISTS. CoQui contains no OpenMP regions of its own (T-0 census); the only
 * threading available to it is the BLAS/LAPACK layer underneath. T-0 measured that layer
 * to be worth 1.35x overall wall on the non-vertex phases at MKL x4, with real BLAS-3
 * calls scaling 1.4-2.7x (Gij->Guv 2.70x, chol 2.40x, ls solve 2.18x) and nothing
 * regressing. Until now the only way to ask for it was an environment variable, so the
 * setting never appeared in the input file and never reached the checkpoint.
 *
 * WHY IT IS **NOT** `OMP_NUM_THREADS`. Raising the global OpenMP thread count activates
 * SLATE's libgomp task layer, whose tasks issue MPI calls; below MPI_THREAD_MULTIPLE that
 * is an immediate UCX SIGSEGV (job 6890365 -- see slate_ops.hpp's guard). The safe recipe
 * is OMP_NUM_THREADS=1 with the BLAS layer threaded on its own knob: inside MKL,
 * MKL_Set_Num_Threads overrides OMP_NUM_THREADS, so SLATE's task layer stays serial while
 * BLAS threads freely. `blas_threads` is that knob, and it deliberately cannot touch
 * OMP_NUM_THREADS.
 *
 * LIBRARY-AGNOSTIC BY RUNTIME LOOKUP. The name is vendor-neutral because the sink is
 * resolved with dlsym(RTLD_DEFAULT, ...) at call time rather than by linking a vendor
 * header: MKL on rusty, OpenBLAS on the Mac dev tree, and neither one is a build
 * dependency. If no known setter is present the knob degrades to a warning and the
 * library's own default (i.e. the environment) stands -- it never aborts a run.
 */

#include <cstdlib>
#include <string>
#include <dlfcn.h>

#include "IO/app_loggers.h"
#include "utilities/check.hpp"

namespace utils {

/// Requested BLAS thread count. 0 = never requested; the library/environment default wins.
inline long &blas_threads_state() { static long n = 0; return n; }

/// The vendor entry point actually used, for the startup echo and the checkpoint record.
/// Empty until a set is attempted; "none" when no known setter could be resolved.
inline std::string &blas_threads_backend() { static std::string s; return s; }

/// Read-only accessors (used by the startup echo and by chkpt::write_metadata).
inline long blas_threads() { return blas_threads_state(); }
inline std::string blas_threads_backend_name() { return blas_threads_backend(); }

/**
 * Push `n` into whichever threaded BLAS is loaded.
 *
 * Ordered by the two libraries CoQui is actually built against. MKL_Set_Num_Threads is
 * the C entry (int by value); the lowercase `mkl_set_num_threads` symbol is the Fortran
 * binding and takes a POINTER, so it must not be called through this signature.
 *
 * @return true when a setter was found and called.
 */
inline bool apply_blas_threads(long n) {
  using set_int_t = void (*)(int);
  static const char *candidates[] = {
      "MKL_Set_Num_Threads",        // Intel MKL, C API (rusty)
      "openblas_set_num_threads",   // OpenBLAS (Mac dev tree)
  };
  for (const char *nm : candidates) {
    // NOLINTNEXTLINE: dlsym returns void*; the cast is the documented idiom.
    if (auto fn = reinterpret_cast<set_int_t>(dlsym(RTLD_DEFAULT, nm))) {
      fn(static_cast<int>(n));
      blas_threads_backend() = nm;
      // MKL_DYNAMIC would let MKL silently use fewer threads than asked; pin it off so the
      // requested count is the count actually used (mirrors MKL_DYNAMIC=FALSE in the
      // measurement harnesses). Absent on OpenBLAS -- optional by construction.
      if (auto dyn = reinterpret_cast<set_int_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Dynamic")))
        dyn(0);
      return true;
    }
  }
  blas_threads_backend() = "none";
  return false;
}

/**
 * Apply and record the `blas_threads` setting. `n <= 0` is "not requested": nothing is
 * touched and the environment/library default stands, which is the pre-T-2 behaviour and
 * the default for every existing input file.
 */
inline void set_blas_threads(long n) {
  utils::check(n >= 0, "blas_threads must be >= 0 (0 = leave the library/environment "
                       "default alone); got {}.", n);
  if (n == 0) return;
  blas_threads_state() = n;
  const bool ok = apply_blas_threads(n);
  if (ok) {
    app_log(2, "  blas_threads = {} applied via {} (OMP_NUM_THREADS is deliberately NOT "
               "touched: raising it would activate SLATE's OpenMP task layer, which needs "
               "MPI_THREAD_MULTIPLE).", n, blas_threads_backend());
  } else {
    app_warning("blas_threads = {} was requested but no known BLAS thread-count setter "
                "(MKL_Set_Num_Threads / openblas_set_num_threads) could be resolved in "
                "this process. The BLAS layer keeps its own default; set MKL_NUM_THREADS "
                "or OPENBLAS_NUM_THREADS in the environment instead.", n);
  }
}

} // namespace utils

#endif // UTILITIES_BLAS_THREADS_HPP
