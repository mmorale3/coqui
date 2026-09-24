/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

#ifndef COQUI_H5_BACKGROUND_WRITER_HPP
#define COQUI_H5_BACKGROUND_WRITER_HPP

#include <exception>
#include <functional>
#include <thread>
#include <utility>

namespace utils {

/**
 * Runs at most one background HDF5 write at a time.
 *
 * Why this exists: the SCF checkpoint is ~8.9 GB per iteration at Si
 * 2x2x2/500b and is written serially by one rank while every other rank waits
 * in a barrier -- 30 s of a 141 s iteration. It is filesystem-bound (ceph
 * delivers ~340 MB/s on a single stream), so the time cannot be recovered by
 * writing faster, only by not waiting for it.
 *
 * THE INVARIANT, and it is not optional: the HDF5 build here reports
 * "Threadsafety: OFF", so its global state is not protected. Any HDF5 call
 * anywhere in the process -- same file or not -- that runs while a background
 * write is in flight is a data race, not merely a slow path. Every other HDF5
 * entry point reachable while a write may be outstanding therefore calls
 * h5_quiesce() first. It is a cheap no-op when nothing is in flight.
 *
 * The known entry points, all of which call h5_quiesce():
 *   - scr_coulomb_t::dump_eps_inv_head   (same file, same group, every iter)
 *   - damping_impl / diis_impl / diis_init   (same file)
 *   - diis_t::get_mu, com_diis_residual::upload_g_mu, FockSigma read/write
 *   - thc_reader_t::Z   (ERI file, outcore only)
 *   - chol_reader_t::read_V / read_Vq   (ERI file, always lazy)
 * If you add an HDF5 call that can run during an SCF iteration, add one there
 * too.
 */
class h5_background_writer {
 public:
  static h5_background_writer& instance() {
    static h5_background_writer w;   // one per process (C++17 guarantees this)
    return w;
  }

  h5_background_writer(const h5_background_writer&) = delete;
  h5_background_writer& operator=(const h5_background_writer&) = delete;

  /**
   * Join any in-flight write, then run `task` on a background thread.
   * The task must own everything it touches -- move the buffers in.
   */
  void submit(std::function<void()> task) {
    wait();
    _err = nullptr;
    _thread = std::thread([this, t = std::move(task)]() mutable {
      // A throwing background thread would call std::terminate, taking the
      // run down without a usable message. Stash it and rethrow in wait().
      try { t(); } catch (...) { _err = std::current_exception(); }
    });
  }

  /** Join any in-flight write and rethrow whatever it threw. */
  void wait() {
    if (_thread.joinable()) _thread.join();
    if (_err) { auto e = _err; _err = nullptr; std::rethrow_exception(e); }
  }

  bool busy() const { return _thread.joinable(); }

  ~h5_background_writer() {
    // Never let the process exit with a half-written checkpoint. Swallow here
    // rather than throw from a destructor; wait() reports errors normally.
    if (_thread.joinable()) { try { _thread.join(); } catch (...) {} }
  }

 private:
  h5_background_writer() = default;
  std::thread _thread;
  std::exception_ptr _err = nullptr;
};

/**
 * Join any in-flight background HDF5 write on THIS process. Call before any
 * other HDF5 access from the same rank; see the invariant above.
 *
 * This is not enough on its own when another rank reads what this rank wrote:
 * the writer thread lives on one rank only, so h5_quiesce() is a no-op
 * everywhere else and gives them no reason to wait. Use h5_quiesce_collective
 * for that -- see the note there.
 */
inline void h5_quiesce() { h5_background_writer::instance().wait(); }

/**
 * Collective join: every rank quiesces its own writer (a no-op on all but the
 * writing rank) and then waits on the barrier, so no rank proceeds until the
 * checkpoint is actually on disk.
 *
 * Needed wherever a rank OTHER than the writer reads the checkpoint. Learned
 * the hard way: with only the local join, `read_input_iterations` on the
 * non-writing ranks raced the iteration-0 checkpoint and aborted with
 * 'h5 group "scf" does not exist'.
 */
template<typename Comm>
inline void h5_quiesce_collective(Comm& comm) {
  h5_quiesce();
  comm.barrier();
}

}  // namespace utils

#endif  // COQUI_H5_BACKGROUND_WRITER_HPP
