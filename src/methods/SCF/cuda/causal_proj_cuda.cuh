/*
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

#ifndef COQUI_SCF_CAUSAL_PROJ_CUDA_CUH
#define COQUI_SCF_CAUSAL_PROJ_CUDA_CUH

/**
 * The device implementation of causal_proj::projector::project (methods/SCF/sigma_causal_projection.hpp): the same
 * Anderson-accelerated Douglas-Rachford ADMM, step for step, batched over the (s, k) blocks of a call.
 *
 * Plain pointers in, plain pointers out (the caller owns the nda arrays and hands over their C-contiguous data); this
 * header carries no CoQuí array types, so the .cu stays a self-contained CUDA translation unit (target scf_cuda, built
 * only with ENABLE_CUDA).
 *
 * Device residency: per call the blocks are uploaded once (Sigma_c block, warm-start C and U), every ADMM array of every
 * block of a batch lives on the device for all iterations (v, C, C_prev, B, the right-hand side, g, f and their previous
 * values, Kt^T Y, the eigenvectors of the PSD clip, the Anderson differences dG / dF), and the scalar control of the
 * solver (convergence test, Anderson ring buffer, restarts, the small least-squares solve) runs in one-thread-per-block
 * kernels. The host only launches; it reads back a few flags every 25 iterations to stop early when every block has
 * converged. The batch is all the call's blocks when they fit in 85 % of the free device memory (otherwise consecutive
 * batches; the blocks are independent).
 *
 * PSD clip: one thread block per node matrix (nb x nb). The semidefinite Cholesky pre-test of the host code is repeated
 * (same tolerance, in place); an indefinite matrix is diagonalised by a cyclic two-sided complex Jacobi method
 * (round-robin ordering, the nb / 2 disjoint rotations of a round applied at once, only the rotating pairs, upper half
 * plus mirror), WARM-STARTED from the previous ADMM iteration's eigenvectors of the same node (A' = V^dag A V), V
 * re-orthonormalised by a Newton-Schulz polar step every 8 ADMM iterations; then C = W W^dag over the positive
 * eigenvalues, W = V sqrt(max(lambda, 0)), as on the host. A pair rotates when |A_pq| > jacobi_tol |A|_F (1e-14);
 * converged when the largest off-diagonal element is below that. A and a few scratch rows live in shared memory
 * ((nb^2 + 256) complex: 64 KB at nb 60, two clips per SM), the eigenvectors in their global warm-start buffer; when A
 * does not fit (nb > ~117 on H100 / H200, > ~98 on A100) the same kernel runs on a global-memory workspace. The linear
 * steps (Minv R, Kt^T Y, Kt C) are real x complex: Kt and Minv are real, so they run as cuBLAS dgemm on the interleaved
 * real / imaginary parts (strided-batched over the blocks).
 *
 * Reductions (residual norms, Anderson Gram row) use a fixed partition and a fixed-order sum: the result does not
 * depend on scheduling, run to run. It is not bitwise equal to the host's (different summation order and eigensolver);
 * the agreement is at roundoff per iteration, amplified over many iterations by the ill-conditioned Anderson solve the
 * same way as any roundoff perturbation of the host path is.
 */

#include <complex>

namespace methods::causal_proj_cuda {

  using cplx = std::complex<double>;

  /** the solver's parameters (causal_proj::options, with rho ABSOLUTE: options::rho times s_max(Kt)^2) */
  struct config {
    long nt = 0, L = 0, nb = 0;
    long maxit = 0;
    double tol = 0.0, rho = 0.0, relax = 1.0, aa_restart = 10.0, aa_reg = 1e-12;
    long aa_mem = 0;
    double jacobi_tol = 1e-14;    // the PSD clip's Jacobi: a pair rotates when |A_pq| > jacobi_tol |A|_F
    bool profile = false;         // time the kernel classes (synchronises every iteration) and the PSD clip's phases
  };

  /** per (s, k) block: the fields of causal_proj::stats */
  struct block_stats {
    double y_norm = 0.0, removed = 0.0, lam_before = 0.0, lam_after = 0.0, fit_res = 0.0;
    long iters = 0, aa_restarts = 0;
    int converged = 0;
  };

  /** per call: how it ran */
  struct call_info {
    long nbatch = 0, batch = 0;          // batches and blocks per batch
    long sweeps_max = 0;                 // most Jacobi sweeps of one PSD clip (warm-started), over the call
    long sweeps_max_cold = 0;            // most Jacobi sweeps of one cold eigenvalue solve (lambda_min)
    double sweeps_mean = 0.0;            // mean Jacobi sweeps per diagonalised PSD clip
    double psd_calls = 0.0, psd_kept = 0.0;  // node matrices clipped / of them passed the semidefinite test (no Jacobi)
    bool shared_mem = true;              // the PSD matrices lived in shared memory
    double t_entry = 0.0, t_loop = 0.0, t_exit = 0.0;   // seconds (upload + setup, ADMM iterations, finish + download)
    double device_mb = 0.0;              // device memory of one batch
    // config::profile only: the share of the diagonalised clips' cycles in [0] load + semidefinite test, [1] warm-start
    // transform, [2] Jacobi, [3] re-orthonormalisation + rebuild; and seconds in [0] the PSD kernel, [1] the Minv gemm,
    // [2] the elementwise / Anderson / control kernels
    double psd_phase[4] = {0.0, 0.0, 0.0, 0.0};
    double t_kernels[3] = {0.0, 0.0, 0.0};
  };

  /** the largest Anderson memory the device path supports */
  long max_aa_mem();

  struct engine;

  /**
   * One engine per projector (per tau mesh): the operators on the device and the cuBLAS handle / stream.
   * Kt (nt, L), KtT (L, nt), Minv (L, L): row-major, real. Returns nullptr when the device path cannot run (no device,
   * an unsupported configuration); `why` then says what failed.
   */
  engine *create(config const &c, double const *Kt, double const *KtT, double const *Minv, char *why, long why_len);
  void destroy(engine *e);

  /**
   * Project nblk (s, k) blocks. S[i] (nt, nb, nb) in place; C2[i], U2[i] (L, nb * nb): the warm-start state in and out
   * (zero on a first call). st[i]: the block's statistics. info: optional. Aborts (APP_ABORT) on a CUDA / cuBLAS error, a
   * Jacobi solve that does not converge in its sweep limit, a singular Anderson system (the host's nda::inverse throws
   * there), and when not even one block fits in the free device memory.
   */
  void project(engine *e, long nblk, cplx *const *S, cplx *const *C2, cplx *const *U2, block_stats *st, call_info *info);

  /**
   * Test hook: the PSD clip of every row of X (rows, nb * nb), read as a Hermitian nb x nb matrix, on the device, cold
   * started (identity eigenvectors). force_global: the global-memory Jacobi workspace even when shared memory would fit.
   * sweeps_max (optional): the most Jacobi sweeps used. kept (optional): rows that passed the semidefinite test.
   */
  void psd_rows(cplx *X, long rows, long nb, bool force_global, long *sweeps_max, long *kept, double jacobi_tol = 1e-14);

} // namespace methods::causal_proj_cuda

#endif // COQUI_SCF_CAUSAL_PROJ_CUDA_CUH
