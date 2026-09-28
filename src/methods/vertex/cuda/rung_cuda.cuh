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

#ifndef COQUI_VERTEX_RUNG_CUDA_CUH
#define COQUI_VERTEX_RUNG_CUDA_CUH

/**
 * The STREAMING THC rung on the device (gpu port R1(b), notes/gpu_port_plan.md section 4h): the device twin of
 * vertex_dynbse.icc::thc_rung_apply_ft on a full nosym Gamma-centred mesh with the FFT k-sum,
 *   out(k', (p1 p3'), N) = scale sum_k sum_PQ X(k',P,p1) conj(X(k'+q,Q,p3')) W_PQ(k-k')
 *                                     sum_{p1' p3} conj(X(k,P,p1')) F(k,(p1'p3),N) X(k+q,Q,p3),
 * with no D x D object (D = nk nc^2): per block of nb columns the legs-in gemms build Y(k, (P n Q)), a batched 3-D
 * cuFFT over the mesh turns the k-sum into a product with A(R, PQ) = sum_q e^{+i q.R} W(q) / nk (built once per W
 * table), the inverse transform returns Z(k', (P n Q)), and the legs-out gemms + a scatter add scale * O into out.
 * Every k-indexed buffer is stored by the mesh's lexicographic row R = lex_k(k) (the FFT's own order), so the
 * batched gemms run with uniform strides. Plain pointers, row-major, as l0_cuda.cuh.
 */

#include <complex>

namespace methods::solvers::dynbse_cuda {

  using cplx = std::complex<double>;

  struct rung_stream;

  struct rs_config {
    long ns = 0, nk = 0, nq = 0, Nm = 0, nc = 0;
    long nb = 16;                   // columns per FFT block (reduced to fit the budget; >= 1)
    long ntab = 1;                  // resident transformed rung tables (slots); 1 when the requested count does not fit
    int ndim[3] = {0, 0, 0};        // the mesh (ndim[0] ndim[1] ndim[2] == nk), lex row = (m0 n1 + m1) n2 + m2
  };

  /** device bytes for a configuration and the widest block nR_max the caller will apply (F and out staging included) */
  double rs_bytes(rs_config const &c, long nR_max);
  /** nullptr (with `why`) when the device cannot hold the working set even at nb = 1. lex_k (nk), lex_q (nq): the mesh rows
   *  of every k and every transfer (kmesh_ft::lex_k / lex_q, both permutations of [0, nk)); Xb (ns, nk, Nm, nc) host. */
  rung_stream *rs_create(rs_config const &c, long nR_max, long const *lex_k, long const *lex_q, cplx const *Xb,
                         double free_bytes, char *why, long why_len);
  void rs_destroy(rung_stream *e);
  /** the nb and the number of resident table slots the engine settled on */
  long rs_nb(rung_stream const *e);
  long rs_ntab(rung_stream const *e);
  /** per (s, q): the four leg tables from the row kpq(iq, :) (nk) */
  void rs_set_sq(rung_stream *e, long is, long const *kpq_row);
  /** a rung table W(q, P, Q) at host W + q ldq (row-major Nm x Nm per q) into slot (< rs_ntab): A = its transform / nk,
   *  resident until the slot is reloaded. timing (1) ADDED: seconds. */
  void rs_load_w(rung_stream *e, long slot, cplx const *W, long ldq, double *timing);
  /** out (nk, nc^2, nR) = scale K[W_slot] F (nk, nc^2, nR), both HOST row-major (out overwritten). timing (4) ADDED:
   *  [0] legs in, [1] FFT + product, [2] legs out + scatter, [3] H2D + D2H. */
  void rs_apply(rung_stream *e, long slot, cplx scale, cplx const *F, cplx *out, long nR, double *timing);
  /** rs_apply on DEVICE pointers (the device-resident unit's tau buffers): same layouts, no transfers; timing slot [3] unused */
  void rs_apply_dev(rung_stream *e, long slot, cplx scale, void const *F_dev, void *out_dev, long nR, double *timing);

} // namespace methods::solvers::dynbse_cuda

#endif
