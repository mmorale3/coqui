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

/**
 * The device causal projection (causal_proj_cuda.cuh). Every step mirrors causal_proj::projector::project in
 * sigma_causal_projection.hpp; the comments name the host statement each kernel replaces.
 *
 * Layout. A batch of nbc blocks keeps each per-block array contiguous, block after block: the ADMM arrays are
 * (nbc, L, nb^2) complex (row l of a block = the node-l matrix, row-major nb x nb), the tau arrays (nbc, nt, nb^2). Read
 * as interleaved doubles, a block's (L, nb^2) complex array is a row-major (L, 2 nb^2) real one, so A X with a real A is
 * one real gemm; with cuBLAS' column-major convention a row-major product C = A X is the column-major X^T A^T, i.e.
 * gemm(N, N, m = 2 nb^2, n = rows(A), k = cols(A), X, A).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "IO/AppAbort.hpp"
#include "methods/SCF/cuda/causal_proj_cuda.cuh"

namespace methods::causal_proj_cuda {

  namespace {

    using cd = cuDoubleComplex;
    constexpr int NT = 256;            // threads per block, every kernel (the reductions assume it); one PSD clip per block
    constexpr int MMAX = 16;           // the largest Anderson memory
    constexpr int MAX_SWEEPS = 60;     // Jacobi sweep limit (a warm-started clip needs 1-3, a cold solve ~8 at nb 60)

    void cu_check(cudaError_t e, char const *what) {
      if (e == cudaSuccess) return;
      APP_ABORT(std::string(" causal_proj_cuda: CUDA error in ") + what + ": " + cudaGetErrorName(e) + " (" +
                cudaGetErrorString(e) + ")");
    }
    void cub_check(cublasStatus_t e, char const *what) {
      if (e == CUBLAS_STATUS_SUCCESS) return;
      APP_ABORT(std::string(" causal_proj_cuda: cuBLAS error in ") + what + ": status " + std::to_string(int(e)));
    }
    void launch_check(char const *what) { cu_check(cudaGetLastError(), what); }

    __device__ __forceinline__ cd cadd(cd a, cd b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
    __device__ __forceinline__ cd csub(cd a, cd b) { return make_cuDoubleComplex(a.x - b.x, a.y - b.y); }
    __device__ __forceinline__ cd cmul(cd a, cd b) { return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
    __device__ __forceinline__ cd cmulc(cd a, cd b) {   // conj(a) b
      return make_cuDoubleComplex(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
    }
    __device__ __forceinline__ cd cconj(cd a) { return make_cuDoubleComplex(a.x, -a.y); }
    __device__ __forceinline__ cd cscal(double s, cd a) { return make_cuDoubleComplex(s * a.x, s * a.y); }
    __device__ __forceinline__ double cnorm(cd a) { return a.x * a.x + a.y * a.y; }
    __device__ __forceinline__ cd czero() { return make_cuDoubleComplex(0.0, 0.0); }

    // ---- block reductions (NT threads, fixed order: warp shuffles, then the warps in order) -------------------------------
    __device__ double block_sum(double v, double *red) {
      for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
      __syncthreads();
      if (lane == 0) red[w] = v;
      __syncthreads();
      if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < NT / 32; ++i) s += red[i];
        red[NT / 32] = s;
      }
      __syncthreads();
      return red[NT / 32];
    }
    __device__ double block_max(double v, double *red) {
      for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
      const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
      __syncthreads();
      if (lane == 0) red[w] = v;
      __syncthreads();
      if (threadIdx.x == 0) {
        double s = red[0];
        for (int i = 1; i < NT / 32; ++i) s = fmax(s, red[i]);
        red[NT / 32] = s;
      }
      __syncthreads();
      return red[NT / 32];
    }
    // K sums at once; thread k < K writes the block's k-th sum to out[k]. red: NT / 32 * K doubles of shared memory.
    template <int K>
    __device__ void block_sum_k(double (&acc)[K], double *out, double *red) {
      const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        double v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[w * K + k] = v;
      }
      __syncthreads();
      for (int k = threadIdx.x; k < K; k += NT) {
        double s = 0.0;
        for (int i = 0; i < NT / 32; ++i) s += red[i * K + k];
        out[k] = s;
      }
    }

    // ---- the dense nb x nb kernels of one thread block (row-major matrices, shared or global memory) ----------------------
    // the semidefinite Cholesky test of sigma_causal_projection.hpp (projector::semidefinite), right-looking IN PLACE on S
    // (overwritten): tau = 1e-13 dmax; a pivot below -tau, or a pivot within tau whose column below exceeds sqrt(tau dmax),
    // is indefinite
    __device__ bool semidefinite(cd *S, int n, double dmax, int *s_flag) {
      const int tid = threadIdx.x;
      const double tau = 1e-13 * dmax, vtol = sqrt(tau * dmax);
      __syncthreads();
      for (int j = 0; j < n; ++j) {
        const double d = S[j * n + j].x;
        if (d < -tau) { __syncthreads(); return false; }
        if (d <= tau) {                                   // zero pivot: the column below must vanish; L(:, j) = 0
          if (tid == 0) *s_flag = 0;
          __syncthreads();
          for (int i = j + 1 + tid; i < n; i += NT)
            if (sqrt(cnorm(S[i * n + j])) > vtol) *s_flag = 1;
          __syncthreads();
          const int f = *s_flag;
          __syncthreads();
          if (f) return false;
          continue;
        }
        const double sd = sqrt(d);
        for (int i = j + 1 + tid; i < n; i += NT) S[i * n + j] = cscal(1.0 / sd, S[i * n + j]);
        __syncthreads();
        const int nn = n - j - 1;
        for (int t = tid; t < nn * nn; t += NT) {
          const int i = j + 1 + t / nn, k = j + 1 + t % nn;
          if (k <= i) S[i * n + k] = csub(S[i * n + k], cmul(S[i * n + j], cconj(S[k * n + j])));
        }
        __syncthreads();
      }
      return true;
    }

    // A <- A M (the rows of A are independent: rc rows at a time through the scratch W, rc n entries)
    __device__ void mul_right(cd *A, cd const *M, cd *W, int n, int rc) {
      const int tid = threadIdx.x;
      for (int r0 = 0; r0 < n; r0 += rc) {
        const int rows = min(rc, n - r0);
        for (int t = tid; t < rows * n; t += NT) {
          const int i = r0 + t / n, j = t % n;
          cd s = czero();
          for (int k = 0; k < n; ++k) s = cadd(s, cmul(A[i * n + k], M[k * n + j]));
          W[t] = s;
        }
        __syncthreads();
        for (int t = tid; t < rows * n; t += NT) A[(r0 + t / n) * n + t % n] = W[t];
        __syncthreads();
      }
    }
    // A <- M^dag A (the columns of A are independent)
    __device__ void mul_left_dag(cd *A, cd const *M, cd *W, int n, int rc) {
      const int tid = threadIdx.x;
      for (int c0 = 0; c0 < n; c0 += rc) {
        const int cols = min(rc, n - c0);
        for (int t = tid; t < cols * n; t += NT) {
          const int j = c0 + t / n, i = t % n;
          cd s = czero();
          for (int k = 0; k < n; ++k) s = cadd(s, cmulc(M[k * n + i], A[k * n + j]));
          W[t] = s;
        }
        __syncthreads();
        for (int t = tid; t < cols * n; t += NT) A[(t % n) * n + c0 + t / n] = W[t];
        __syncthreads();
      }
    }

    // Cyclic two-sided Jacobi on the Hermitian A (in place: A <- J^dag A J, V <- V J). Round-robin ordering: the n / 2
    // disjoint pairs of a round are rotated at once. Rotation of (p, q), b = A_pq = |b| e: J = [[c, s e], [-s e^*, c]] with
    // the real symmetric Schur rotation of [[A_pp, |b|], [|b|, A_qq]] (Golub & Van Loan, sym.schur2). A pair is skipped
    // when |A_pq| <= thr. Only the rotating pairs are applied: the 2 x 2 blocks of A in their rows or columns (each
    // thread owns whole blocks and their mirror blocks, so a round is one pass over the upper half) and their columns of V;
    // near convergence (and warm-started) most pairs of a round are idle. Returns the sweeps used (0: already diagonal to thr; checked on the largest off-diagonal element before each sweep), or -1 after MAX_SWEEPS; *nrot:
    // the rotations applied. Shared scratch: sp, sq, act, inact (h ints each), sc (h), sse (h), s_cnt (1 int).
    template <bool WANT_V>
    __device__ int jacobi(cd *A, cd *V, int n, double thr, int *sp, int *sq, int *act, int *inact, double *sc, cd *sse, int *s_cnt,
                          double *red, long *nrot) {
      const int tid = threadIdx.x;
      const int m = n + (n & 1), h = m / 2;
      *nrot = 0;
      if (n < 2) return 0;
      long tot = 0;
      for (int sweep = 1; sweep <= MAX_SWEEPS + 1; ++sweep) {
        // converged when no pair would rotate: one pass over the upper triangle instead of an idle sweep
        double off = 0.0;
        for (int idx = tid; idx < n * n; idx += NT) {
          const int i = idx / n, j = idx - i * n;
          if (j > i) off = fmax(off, cnorm(A[idx]));
        }
        off = block_max(off, red);
        if (not(sqrt(off) > thr)) { *nrot = tot; return sweep - 1; }
        if (sweep > MAX_SWEEPS) break;
        long swr = 0;
        for (int r = 0; r < m - 1; ++r) {
          if (tid < 32) {                                 // warp 0: the rotations of the round and the rotating / idle lists
            int na = 0, ni = 0;
            for (int base = 0; base < h; base += 32) {
              const int k = base + tid;
              bool rot = false;
              if (k < h) {
                const int a = (k == 0) ? 0 : 1 + (k - 1 + r) % (m - 1);
                const int b = 1 + (m - 2 - k + r) % (m - 1);
                const int p = min(a, b), q = max(a, b);
                double c = 1.0;
                cd se = czero();
                if (q < n) {
                  const cd apq = A[p * n + q];
                  const double ab = sqrt(cnorm(apq));
                  if (ab > thr) {
                    const double theta = (A[q * n + q].x - A[p * n + p].x) / (2.0 * ab);
                    const double t = (fabs(theta) > 1e150) ? 0.5 / theta
                                                           : ((theta >= 0.0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                    c = 1.0 / sqrt(1.0 + t * t);
                    se = cscal(t * c / ab, apq);
                    rot = true;
                  }
                }
                sp[k] = p; sq[k] = q; sc[k] = c; sse[k] = se;
              }
              const unsigned valid = __ballot_sync(0xffffffffu, k < h);
              const unsigned am = __ballot_sync(0xffffffffu, rot);
              const unsigned lt = (1u << tid) - 1u;
              if (rot) act[na + __popc(am & lt)] = k;
              else if (k < h) inact[ni + __popc(~am & valid & lt)] = k;
              na += __popc(am);
              ni += __popc(~am & valid);
            }
            if (tid == 0) *s_cnt = na;
          }
          __syncthreads();
          const int na = *s_cnt, ni = h - na;
          if (na > 0) {
            swr += na;
            // the 2 x 2 blocks (P, Q) with P or Q rotating, by Hermitian symmetry only P <= Q in the order act, inact:
            // both rotating (i <= j in act), then rotating x idle; the mirror block (Q, P) is the conjugate transpose
            const int nt1 = na * na, nt2 = na * ni;
            for (int task = tid; task < nt1 + nt2; task += NT) {
              int P, Q;
              if (task < nt1) {
                const int i = task / na, j = task - i * na;
                if (i > j) continue;
                P = act[i]; Q = act[j];
              } else {
                const int t2 = task - nt1;
                P = act[t2 / ni]; Q = inact[t2 % ni];
              }
              const int p1 = sp[P], p2 = sq[P], q1 = sp[Q], q2 = sq[Q];
              const double cP = sc[P], cQ = sc[Q];
              const cd sP = sse[P], sQ = sse[Q];
              const bool vp2 = p2 < n, vq2 = q2 < n;
              const cd a11 = A[p1 * n + q1];
              const cd a12 = vq2 ? A[p1 * n + q2] : czero();
              const cd a21 = vp2 ? A[p2 * n + q1] : czero();
              const cd a22 = (vp2 and vq2) ? A[p2 * n + q2] : czero();
              // rows: J_P^dag = [[c, -s e], [s e^*, c]]
              const cd t11 = csub(cscal(cP, a11), cmul(sP, a21)), t12 = csub(cscal(cP, a12), cmul(sP, a22));
              const cd t21 = cadd(cmul(cconj(sP), a11), cscal(cP, a21)), t22 = cadd(cmul(cconj(sP), a12), cscal(cP, a22));
              // columns: J_Q = [[c, s e], [-s e^*, c]]
              const cd r11 = csub(cscal(cQ, t11), cmul(t12, cconj(sQ))), r12 = cadd(cmul(t11, sQ), cscal(cQ, t12));
              const cd r21 = csub(cscal(cQ, t21), cmul(t22, cconj(sQ))), r22 = cadd(cmul(t21, sQ), cscal(cQ, t22));
              A[p1 * n + q1] = r11;
              if (vq2) A[p1 * n + q2] = r12;
              if (vp2) A[p2 * n + q1] = r21;
              if (vp2 and vq2) A[p2 * n + q2] = r22;
              if (P != Q) {
                A[q1 * n + p1] = cconj(r11);
                if (vq2) A[q2 * n + p1] = cconj(r12);
                if (vp2) A[q1 * n + p2] = cconj(r21);
                if (vp2 and vq2) A[q2 * n + p2] = cconj(r22);
              }
            }
            if constexpr (WANT_V) {
              for (int task = tid; task < n * na; task += NT) {
                const int i = task / na, Q = act[task % na];
                const int q1 = sp[Q], q2 = sq[Q];      // a rotating pair never has the dummy partner
                const double cQ = sc[Q];
                const cd sQ = sse[Q];
                const cd v1 = V[i * n + q1], v2 = V[i * n + q2];
                V[i * n + q1] = csub(cscal(cQ, v1), cmul(v2, cconj(sQ)));
                V[i * n + q2] = cadd(cmul(v1, sQ), cscal(cQ, v2));
              }
            }
          }
          __syncthreads();
        }
        tot += swr;
      }
      *nrot = tot;
      return -1;
    }

    // the PSD-clip kernel's shared memory (bytes) and where its pieces live
    struct psd_layout {
      int n, h, rc;
      bool shared;
      __host__ __device__ size_t big() const { return size_t(n * n + rc * n) * sizeof(cd); }   // A, W (V: the global Veig row)
      __host__ __device__ size_t small() const {
        return size_t(h) * sizeof(cd) + size_t(n + h) * sizeof(double) + size_t(4 * h + n) * sizeof(int);
      }
      __host__ __device__ size_t smem() const { return (shared ? big() : 0) + small(); }
    };
    psd_layout make_layout(long n, bool shared) {
      psd_layout l;
      l.n = int(n);
      l.h = int((n + (n & 1)) / 2);
      l.rc = int(std::max(1l, long(NT) / std::max(1l, n)));
      l.shared = shared;
      return l;
    }

    struct psd_args {
      psd_layout lay;
      int L;                       // rows (node matrices) per block
      int mode;                    // 0: ADMM step (active blocks; C and R); 1: finish (blocks not converged; C); 2: every row (C)
      cd const *X;                 // the input rows (v)
      cd *C;                       // the clipped rows (may equal X in mode 2)
      cd *R;                       // mode 0: R = K + rho (2 C - X)
      cd const *K;
      double rho;
      cd *Veig;                    // per row: the eigenvectors of the last clip (the warm start)
      cd *gws;                     // the global workspace (lay.big() bytes per row) when !lay.shared
      double jtol;                 // Jacobi: a pair rotates when |A_pq| > jtol |A|_F
      int ns;                      // 1: re-orthonormalise the stored eigenvectors (Newton-Schulz) this call
      int const *active, *converged;
      int *err, *sweeps;
      unsigned long long *calls;   // [0] clips, [1] kept by the semidefinite test, [2] Jacobi sweeps (sum)
      unsigned long long *prof;    // optional: clock cycles of the diagonalised clips: [0] load + test, [1] the warm-start
                                   // transform, [2] Jacobi, [3] re-orthonormalisation + rebuild
    };

    // C = PSD(v) row by row (projector::psd_rows): the symmetrised matrix is kept when it passes the semidefinite test,
    // otherwise its eigendecomposition (Jacobi, warm-started from Veig) rebuilds W W^dag, W = V sqrt(max(lambda, 0)).
    // Mode 0 also forms the ADMM right-hand side of the row, R = Kt^T Y + rho (2 C - v).
    __global__ void __launch_bounds__(NT) psd_kernel(psd_args a) {
      const int l = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
      if (a.mode == 0 and not a.active[b]) return;
      if (a.mode == 1 and a.converged[b]) return;
      const int n = a.lay.n, h = a.lay.h, rc = a.lay.rc, n2 = n * n;
      const long row = long(b) * a.L + l;
      const long long c0 = clock64();
      extern __shared__ double4 smem_raw[];
      cd *smem = reinterpret_cast<cd *>(smem_raw);
      cd *A, *W, *tail;
      if (a.lay.shared) { A = smem; tail = smem + (n2 + rc * n); }
      else { A = a.gws + row * (n2 + rc * n); tail = smem; }
      W = A + n2;
      cd *V = a.Veig + row * n2;                          // the eigenvectors: warm start in, the new ones out (global memory)
      cd *sse = tail;
      double *lam = reinterpret_cast<double *>(sse + h);
      double *sc = lam + n;
      int *sp = reinterpret_cast<int *>(sc + h);
      int *sq = sp + h, *act = sq + h, *inact = act + h, *pos = inact + h;
      __shared__ double red[NT / 32 + 1];
      __shared__ int s_flag, s_cnt, s_npos;

      cd const *x = a.X + row * n2;
      auto load_sym = [&]() {                             // A = (X + X^dag) / 2 of this row
        for (int idx = tid; idx < n2; idx += NT) {
          const int i = idx / n, j = idx - i * n;
          A[idx] = cscal(0.5, cadd(x[i * n + j], cconj(x[j * n + i])));
        }
        __syncthreads();
      };
      load_sym();
      double f2 = 0.0, dmax = -1e300;
      for (int idx = tid; idx < n2; idx += NT) {
        f2 += cnorm(A[idx]);
        if (idx / n == idx % n) dmax = fmax(dmax, A[idx].x);
      }
      f2 = block_sum(f2, red);
      dmax = block_max(dmax, red);
      bool keep = false;
      if (dmax > 0.0) {                                   // the test factorises A in place; A is reloaded after it
        keep = semidefinite(A, n, dmax, &s_flag);
        __syncthreads();
        load_sym();
      }
      if (tid == 0) {
        atomicAdd(a.calls, 1ull);
        if (keep) atomicAdd(a.calls + 1, 1ull);
      }
      cd *out = a.C + row * n2;
      if (keep) {
        for (int idx = tid; idx < n2; idx += NT) {
          const cd c = A[idx];
          if (a.mode == 0) a.R[row * n2 + idx] = cadd(a.K[row * n2 + idx], cscal(a.rho, csub(cscal(2.0, c), x[idx])));
          out[idx] = c;
        }
        return;
      }
      // warm start: A' = Vp^dag A Vp with the previous eigenvectors of this row
      const long long c1 = clock64();
      mul_right(A, V, W, n, rc);
      mul_left_dag(A, V, W, n, rc);
      long nrot = 0;
      const long long c2 = clock64();
      const int sw = jacobi<true>(A, V, n, a.jtol * sqrt(f2), sp, sq, act, inact, sc, sse, &s_cnt, red, &nrot);
      const long long c3 = clock64();
      if (tid == 0) {
        if (sw < 0) atomicExch(a.err, 1);
        atomicMax(a.sweeps, sw < 0 ? MAX_SWEEPS : sw);
        atomicAdd(a.calls + 2, (unsigned long long)(sw < 0 ? MAX_SWEEPS : sw));
      }
      for (int i = tid; i < n; i += NT) lam[i] = A[i * n + i].x;
      __syncthreads();
      if (nrot > 0) {
        // a Newton-Schulz polar step (every few ADMM iterations) keeps V orthonormal over the warm starts:
        // V <- V (3 I - V^dag V) / 2
        if (a.ns) {
          for (int idx = tid; idx < n2; idx += NT) {
            const int i = idx / n, j = idx - i * n;
            cd s = czero();
            for (int k = 0; k < n; ++k) s = cadd(s, cmulc(V[k * n + i], V[k * n + j]));
            A[idx] = make_cuDoubleComplex((i == j ? 1.5 : 0.0) - 0.5 * s.x, -0.5 * s.y);
          }
          __syncthreads();
          mul_right(V, A, W, n, rc);
        }
      }
      // W = V sqrt(max(lambda, 0)) into A's storage (V stays the warm start), then C = W W^dag over the positive
      // eigenvalues (the other columns of W vanish)
      __syncthreads();
      if (tid == 0) {
        int np = 0;
        for (int k = 0; k < n; ++k)
          if (lam[k] > 0.0) pos[np++] = k;
        s_npos = np;
      }
      for (int idx = tid; idx < n2; idx += NT) A[idx] = cscal(sqrt(fmax(lam[idx % n], 0.0)), V[idx]);
      __syncthreads();
      const int npos = s_npos;
      for (int idx = tid; idx < n2; idx += NT) {
        const int i = idx / n, j = idx - i * n;
        cd s = czero();
        for (int kk = 0; kk < npos; ++kk) {
          const int k = pos[kk];
          s = cadd(s, cmul(A[i * n + k], cconj(A[j * n + k])));
        }
        if (a.mode == 0) a.R[row * n2 + idx] = cadd(a.K[row * n2 + idx], cscal(a.rho, csub(cscal(2.0, s), x[idx])));
        out[idx] = s;
      }
      if (a.prof and tid == 0) {
        const long long c4 = clock64();
        atomicAdd(a.prof, (unsigned long long)(c1 - c0));
        atomicAdd(a.prof + 1, (unsigned long long)(c2 - c1));
        atomicAdd(a.prof + 2, (unsigned long long)(c3 - c2));
        atomicAdd(a.prof + 3, (unsigned long long)(c4 - c3));
      }
    }

    // lambda_min of the Hermitian part of every row of Y (rows, nb^2) (projector::lam_min, before the min over rows):
    // a cold Jacobi solve, eigenvalues only
    __global__ void __launch_bounds__(NT) lammin_kernel(psd_layout lay, double jtol, cd const *Y, cd *gws, double *out, int *err,
                                                         int *sweeps) {
      const long row = blockIdx.x;
      const int tid = threadIdx.x, n = lay.n, h = lay.h, n2 = n * n;
      extern __shared__ double4 smem_raw[];
      cd *smem = reinterpret_cast<cd *>(smem_raw);
      cd *A, *tail;
      if (lay.shared) { A = smem; tail = smem + n2; }
      else { A = gws + row * n2; tail = smem; }
      cd *sse = tail;
      double *sc = reinterpret_cast<double *>(sse + h);
      int *sp = reinterpret_cast<int *>(sc + h);
      int *sq = sp + h, *act = sq + h, *inact = act + h;
      __shared__ double red[NT / 32 + 1];
      __shared__ int s_cnt;
      cd const *y = Y + row * n2;
      double f2 = 0.0;
      for (int idx = tid; idx < n2; idx += NT) {
        const int i = idx / n, j = idx - i * n;
        const cd v = cscal(0.5, cadd(y[i * n + j], cconj(y[j * n + i])));
        A[idx] = v;
        f2 += cnorm(v);
      }
      f2 = block_sum(f2, red);
      long nrot = 0;
      const int sw = jacobi<false>(A, nullptr, n, jtol * sqrt(f2), sp, sq, act, inact, sc, sse, &s_cnt, red, &nrot);
      if (tid == 0) {
        if (sw < 0) atomicExch(err, 1);
        atomicMax(sweeps, sw < 0 ? MAX_SWEEPS : sw);
      }
      double lm = 1e300;
      for (int i = tid; i < n; i += NT) lm = fmin(lm, A[i * n + i].x);
      lm = -block_max(-lm, red);
      if (tid == 0) out[row] = lm;
    }

    // ---- the elementwise / control kernels of the ADMM loop ----------------------------------------------------------------
    struct dstate {
      int *active, *have_prev, *n_hist, *slot, *newest, *converged, *err;
      long *iters, *restarts;
      double *g_best, *H, *gam, *nkty;
    };

    __global__ void __launch_bounds__(NT) state_init_kernel(int nbc, long maxit, dstate s) {
      const int b = blockIdx.x * blockDim.x + threadIdx.x;
      if (b >= nbc) return;
      s.active[b] = 1; s.have_prev[b] = 0; s.n_hist[b] = 0; s.slot[b] = 0; s.newest[b] = 0; s.converged[b] = 0;
      s.iters[b] = maxit; s.restarts[b] = 0; s.g_best[b] = 1e300;
    }

    // V = C2 + U2 (C2 uploaded into Cp, U2 into V); the eigenvector warm starts = identity
    __global__ void __launch_bounds__(NT) entry_kernel(long n_all, cd *V, cd const *Cp, cd *Veig, long nmat, int n) {
      const long stride = long(gridDim.x) * NT;
      for (long i = long(blockIdx.x) * NT + threadIdx.x; i < n_all; i += stride) V[i] = cadd(V[i], Cp[i]);
      const long n2 = long(n) * n;
      for (long i = long(blockIdx.x) * NT + threadIdx.x; i < nmat * n2; i += stride) {
        const long e = i % n2;
        Veig[i] = make_cuDoubleComplex((e / n == e % n) ? 1.0 : 0.0, 0.0);
      }
    }
    // Y(t, i nb + j) = -(S(t, i, j) + S(t, j, i)^*) / 2
    __global__ void __launch_bounds__(NT) hermitian_y_kernel(long nrows, int nb, cd const *S, cd *Y) {
      const long n2 = long(nb) * nb, stride = long(gridDim.x) * NT;
      for (long i = long(blockIdx.x) * NT + threadIdx.x; i < nrows * n2; i += stride) {
        const long r = i / n2, e = i - r * n2, a = e / nb, c = e - a * nb;
        Y[i] = cscal(-0.5, cadd(S[i], cconj(S[r * n2 + c * nb + a])));
      }
    }
    __global__ void __launch_bounds__(NT) sub_kernel(long n_all, cd const *x, cd const *y, cd *z) {   // z = x - y
      const long stride = long(gridDim.x) * NT;
      for (long i = long(blockIdx.x) * NT + threadIdx.x; i < n_all; i += stride) z[i] = csub(x[i], y[i]);
    }
    __global__ void __launch_bounds__(NT) neg_kernel(long n_all, cd *x) {
      const long stride = long(gridDim.x) * NT;
      for (long i = long(blockIdx.x) * NT + threadIdx.x; i < n_all; i += stride) x[i] = make_cuDoubleComplex(-x[i].x, -x[i].y);
    }
    // per block and chunk: sum |x - y|^2 (y may be null)
    __global__ void __launch_bounds__(NT) norm2_kernel(long nel, int nchunk, cd const *x, cd const *y, double *part) {
      const int b = blockIdx.y, ch = blockIdx.x;
      const long off = long(b) * nel, e0 = nel * ch / nchunk, e1 = nel * (ch + 1) / nchunk;
      double acc[1] = {0.0};
      for (long e = e0 + threadIdx.x; e < e1; e += NT) acc[0] += cnorm(y ? csub(x[off + e], y[off + e]) : x[off + e]);
      __shared__ double red[NT / 32];
      block_sum_k<1>(acc, part + long(b) * nchunk + ch, red);
    }

    // g = relax (B - C), f = v + g, the Anderson differences into the ring slot, the residual norms, C_prev = C (one pass)
    __global__ void __launch_bounds__(NT) resid_kernel(long nel, int nchunk, double relax, int m, dstate s, cd const *B, cd const *C, cd const *V,
                                 cd *G, cd *F, cd const *Gp, cd const *Fp, cd *dG, cd *dF, cd *Cp, double *part) {
      const int b = blockIdx.y, ch = blockIdx.x;
      if (not s.active[b]) return;
      const bool store = m > 0 and s.have_prev[b];
      const long off = long(b) * nel, e0 = nel * ch / nchunk, e1 = nel * (ch + 1) / nchunk;
      cd *dg = store ? dG + (long(b) * m + s.slot[b]) * nel : nullptr;
      cd *df = store ? dF + (long(b) * m + s.slot[b]) * nel : nullptr;
      double acc[3] = {0.0, 0.0, 0.0};
      for (long e = e0 + threadIdx.x; e < e1; e += NT) {
        const long i = off + e;
        const cd c = C[i];
        const cd g = cscal(relax, csub(B[i], c)), f = cadd(V[i], g);
        if (store) { dg[e] = csub(g, Gp[i]); df[e] = csub(f, Fp[i]); }
        G[i] = g;
        F[i] = f;
        acc[0] += cnorm(g);
        acc[1] += cnorm(csub(c, Cp[i]));
        acc[2] += cnorm(c);
        Cp[i] = c;
      }
      __shared__ double red[NT / 32 * 3];
      block_sum_k<3>(acc, part + (long(b) * nchunk + ch) * 3, red);
    }

    // the scalar control after the residuals (one thread per block): the convergence test, the Anderson ring buffer
    __global__ void __launch_bounds__(NT) ctl1_kernel(int nbc, int nchunk, long it, double tol, double rho, double relax, int m, double aa_restart,
                                dstate s, double const *part) {
      const int b = blockIdx.x * blockDim.x + threadIdx.x;
      if (b >= nbc or not s.active[b]) return;
      double g2 = 0.0, dc2 = 0.0, nc2 = 0.0;
      for (int ch = 0; ch < nchunk; ++ch) {
        double const *p = part + (long(b) * nchunk + ch) * 3;
        g2 += p[0]; dc2 += p[1]; nc2 += p[2];
      }
      const double gn = sqrt(g2);
      const bool conv = gn / relax <= tol * fmax(sqrt(nc2), 1e-300) and rho * sqrt(dc2) <= tol * s.nkty[b];
      if (conv) { s.converged[b] = 1; s.iters[b] = it + 1; s.active[b] = 0; return; }
      if (m == 0) { s.n_hist[b] = 0; return; }
      const bool store = s.have_prev[b];
      s.have_prev[b] = 1;
      int nh = s.n_hist[b], sl = s.slot[b];
      if (gn > aa_restart * s.g_best[b]) { nh = 0; sl = 0; s.restarts[b] += 1; }
      else if (store) { nh = min(nh + 1, m); sl = (sl + 1) % m; }
      s.g_best[b] = fmin(s.g_best[b], gn);
      s.n_hist[b] = nh;
      s.slot[b] = sl;
      s.newest[b] = (sl + m - 1) % m;
    }

    // the Gram row of the newest difference and the right-hand side: Re <dG_i, dG_newest>, Re <dG_i, g>
    __global__ void __launch_bounds__(NT) dots_kernel(long nel, int nchunk, int m, dstate s, cd const *dG, cd const *Gk, double *part) {
      const int b = blockIdx.y, ch = blockIdx.x;
      if (not s.active[b]) return;
      const int nh = s.n_hist[b];
      if (nh == 0) return;
      const long off = long(b) * nel, e0 = nel * ch / nchunk, e1 = nel * (ch + 1) / nchunk;
      cd const *base = dG + long(b) * m * nel;
      cd const *dgn = base + long(s.newest[b]) * nel;
      double acc[2 * MMAX];
#pragma unroll
      for (int i = 0; i < 2 * MMAX; ++i) acc[i] = 0.0;
      for (long e = e0 + threadIdx.x; e < e1; e += NT) {
        const cd a = dgn[e], g = Gk[off + e];
#pragma unroll
        for (int i = 0; i < MMAX; ++i)
          if (i < nh) {
            const cd ci = base[long(i) * nel + e];
            acc[i] += ci.x * a.x + ci.y * a.y;
            acc[MMAX + i] += ci.x * g.x + ci.y * g.y;
          }
      }
      __shared__ double red[NT / 32 * 2 * MMAX];
      block_sum_k<2 * MMAX>(acc, part + (long(b) * nchunk + ch) * 2 * MMAX, red);
    }

    // the Anderson coefficients: H's newest row, gamma = (H + aa_reg tr(H) I)^{-1} rhs (partial-pivoting elimination)
    __global__ void __launch_bounds__(NT) ctl2_kernel(int nbc, int nchunk, int m, double aa_reg, dstate s, double const *part) {
      const int b = blockIdx.x * blockDim.x + threadIdx.x;
      if (b >= nbc or not s.active[b]) return;
      const int nh = s.n_hist[b];
      if (nh == 0) return;
      double dots[2 * MMAX];
      for (int i = 0; i < 2 * MMAX; ++i) dots[i] = 0.0;
      for (int ch = 0; ch < nchunk; ++ch) {
        double const *p = part + (long(b) * nchunk + ch) * 2 * MMAX;
        for (int i = 0; i < 2 * MMAX; ++i) dots[i] += p[i];
      }
      double *H = s.H + long(b) * m * m;
      const int nw = s.newest[b];
      for (int i = 0; i < nh; ++i) { H[nw * m + i] = dots[i]; H[i * m + nw] = dots[i]; }
      double A[MMAX * MMAX], x[MMAX];
      double tr = 0.0;
      for (int i = 0; i < nh; ++i) tr += H[i * m + i];
      for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < nh; ++j) A[i * MMAX + j] = H[i * m + j];
        A[i * MMAX + i] += aa_reg * tr;
        x[i] = dots[MMAX + i];
      }
      bool sing = false;
      for (int k = 0; k < nh; ++k) {
        int pv = k;
        for (int i = k + 1; i < nh; ++i)
          if (fabs(A[i * MMAX + k]) > fabs(A[pv * MMAX + k])) pv = i;
        if (A[pv * MMAX + k] == 0.0) { sing = true; break; }
        if (pv != k) {
          for (int j = 0; j < nh; ++j) { const double t = A[k * MMAX + j]; A[k * MMAX + j] = A[pv * MMAX + j]; A[pv * MMAX + j] = t; }
          const double t = x[k]; x[k] = x[pv]; x[pv] = t;
        }
        for (int i = k + 1; i < nh; ++i) {
          const double f = A[i * MMAX + k] / A[k * MMAX + k];
          for (int j = k; j < nh; ++j) A[i * MMAX + j] -= f * A[k * MMAX + j];
          x[i] -= f * x[k];
        }
      }
      if (sing) { atomicOr(s.err, 2); for (int i = 0; i < nh; ++i) s.gam[long(b) * MMAX + i] = 0.0; return; }
      for (int i = nh - 1; i >= 0; --i) {
        double v = x[i];
        for (int j = i + 1; j < nh; ++j) v -= A[i * MMAX + j] * x[j];
        x[i] = v / A[i * MMAX + i];
      }
      for (int i = 0; i < nh; ++i) s.gam[long(b) * MMAX + i] = x[i];
    }

    // v <- f - sum_i gamma_i dF_i (n_hist = 0: v <- f)
    __global__ void __launch_bounds__(NT) update_kernel(long nel, int nchunk, int m, dstate s, cd const *Fk, cd const *dF, cd *V) {
      const int b = blockIdx.y, ch = blockIdx.x;
      if (not s.active[b]) return;
      const int nh = (m > 0) ? s.n_hist[b] : 0;
      double gam[MMAX];
#pragma unroll
      for (int i = 0; i < MMAX; ++i) gam[i] = (i < nh) ? s.gam[long(b) * MMAX + i] : 0.0;
      const long off = long(b) * nel, e0 = nel * ch / nchunk, e1 = nel * (ch + 1) / nchunk;
      cd const *base = dF + long(b) * m * nel;
      for (long e = e0 + threadIdx.x; e < e1; e += NT) {
        cd v = Fk[off + e];
#pragma unroll
        for (int i = 0; i < MMAX; ++i)
          if (i < nh) v = csub(v, cscal(gam[i], base[long(i) * nel + e]));
        V[off + e] = v;
      }
    }

    int nchunk_for(long nel) { return int(std::clamp((nel + NT * 4 - 1) / (NT * 4), 1l, 256l)); }
    int grid1d(long n) { return int(std::clamp((n + NT - 1) / NT, 1l, 65535l)); }

    template <typename T>
    T *dalloc(size_t count, char const *what) {
      void *p = nullptr;
      if (count == 0) count = 1;
      cu_check(cudaMalloc(&p, count * sizeof(T)), what);
      return static_cast<T *>(p);
    }

    long max_smem_optin() {
      int dev = 0, v = 0;
      cu_check(cudaGetDevice(&dev), "cudaGetDevice");
      cu_check(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "shared memory attribute");
      return long(v);
    }

    // the static shared memory of the kernels (reductions, flags) plus a margin, kept out of the dynamic budget
    constexpr long STATIC_SMEM_MARGIN = 4096;

    template <typename K>
    void set_smem(K *kernel, size_t bytes) {
      if (bytes > 48 * 1024) cu_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)),
                                      "cudaFuncSetAttribute");
    }
  } // namespace

  // ======================================================================================================================
  struct engine {
    config c;
    double *Kt = nullptr, *KtT = nullptr, *Minv = nullptr;
    cublasHandle_t cb = nullptr;
    cudaStream_t st = nullptr;
    psd_layout lay_psd, lay_lam;
  };

  long max_aa_mem() { return MMAX; }

  engine *create(config const &c, double const *Kt, double const *KtT, double const *Minv, char *why, long why_len) {
    auto fail = [&](std::string const &s) -> engine * {
      if (why and why_len > 0) std::snprintf(why, size_t(why_len), "%s", s.c_str());
      return nullptr;
    };
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess or ndev < 1) {
      (void)cudaGetLastError();
      return fail("no CUDA device is visible to this rank");
    }
    if (c.aa_mem < 0 or c.aa_mem > MMAX)
      return fail("Anderson memory " + std::to_string(c.aa_mem) + " is outside the device path's range [0, " + std::to_string(MMAX) + "]");
    if (c.nb < 1 or c.L < 1 or c.nt < 1) return fail("empty problem");
    auto *e = new engine;
    e->c = c;
    const long nt = c.nt, L = c.L;
    e->Kt = dalloc<double>(size_t(nt * L), "Kt");
    e->KtT = dalloc<double>(size_t(nt * L), "KtT");
    e->Minv = dalloc<double>(size_t(L * L), "Minv");
    cu_check(cudaMemcpy(e->Kt, Kt, sizeof(double) * size_t(nt * L), cudaMemcpyHostToDevice), "H2D Kt");
    cu_check(cudaMemcpy(e->KtT, KtT, sizeof(double) * size_t(nt * L), cudaMemcpyHostToDevice), "H2D KtT");
    cu_check(cudaMemcpy(e->Minv, Minv, sizeof(double) * size_t(L * L), cudaMemcpyHostToDevice), "H2D Minv");
    cub_check(cublasCreate(&e->cb), "cublasCreate");
    cu_check(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking), "stream");
    cub_check(cublasSetStream(e->cb, e->st), "cublasSetStream");
    const long smax = max_smem_optin() - STATIC_SMEM_MARGIN;
    e->lay_psd = make_layout(c.nb, true);
    if (long(e->lay_psd.smem()) > smax) e->lay_psd.shared = false;
    e->lay_lam = make_layout(c.nb, true);
    e->lay_lam.rc = 0;                          // the eigenvalue solve keeps A only: n^2 complex
    {
      const size_t lam_smem = size_t(c.nb * c.nb) * sizeof(cd) + e->lay_lam.small();
      if (long(lam_smem) > smax) e->lay_lam.shared = false;
    }
    return e;
  }

  void destroy(engine *e) {
    if (not e) return;
    if (e->cb) (void)cublasDestroy(e->cb);
    if (e->st) (void)cudaStreamDestroy(e->st);
    (void)cudaFree(e->Kt);
    (void)cudaFree(e->KtT);
    (void)cudaFree(e->Minv);
    delete e;
  }

  namespace {
    size_t lam_smem(psd_layout const &l) { return (l.shared ? size_t(l.n * l.n) * sizeof(cd) : 0) + l.small(); }

    // the batch buffers (allocated once per call for the largest batch)
    struct batch_bufs {
      cd *V = nullptr, *C = nullptr, *Cp = nullptr, *B = nullptr, *R = nullptr, *G = nullptr, *F = nullptr, *Gp = nullptr,
         *Fp = nullptr, *KtY = nullptr, *Veig = nullptr, *dG = nullptr, *dF = nullptr, *Y = nullptr, *M = nullptr, *gws = nullptr;
      double *part = nullptr, *lam = nullptr;
      dstate s{};
      int *sweeps = nullptr, *sweeps_cold = nullptr;
      unsigned long long *calls = nullptr, *prof = nullptr;
      void free_all() {
        for (void *p : {(void *)V, (void *)C, (void *)Cp, (void *)B, (void *)R, (void *)G, (void *)F, (void *)Gp, (void *)Fp,
                        (void *)KtY, (void *)Veig, (void *)dG, (void *)dF, (void *)Y, (void *)M, (void *)gws, (void *)part,
                        (void *)lam, (void *)s.active, (void *)s.iters, (void *)s.g_best, (void *)sweeps, (void *)calls,
                        (void *)prof})
          if (p) (void)cudaFree(p);
      }
    };

    void dgemm_sb(cublasHandle_t h, long m, long n, long k, double const *A, long lda, long long sA, double const *B, long ldb,
                  long long sB, double *C, long ldc, long long sC, long batch, char const *what) {
      const double one = 1.0, zero = 0.0;
      cub_check(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, int(m), int(n), int(k), &one, A, int(lda), sA, B,
                                          int(ldb), sB, &zero, C, int(ldc), sC, int(batch)),
                what);
    }
  } // namespace

  void project(engine *e, long nblk, cplx *const *S, cplx *const *C2, cplx *const *U2, block_stats *st, call_info *info) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    if (nblk <= 0) return;
    config const &c = e->c;
    const long nt = c.nt, L = c.L, nb = c.nb, nb2 = nb * nb, nel = L * nb2, nyel = nt * nb2, m = c.aa_mem;
    const int nchunk = nchunk_for(nel), nchunk_y = nchunk_for(nyel);
    psd_layout const &lp = e->lay_psd, &ll = e->lay_lam;
    // device bytes per block of a batch
    const double per_blk = 16.0 * double(11 * nel + 2 * std::max(m, 1l) * nel + 2 * nyel) +
                           (lp.shared ? 0.0 : double(L) * double(lp.big())) +
                           (ll.shared ? 0.0 : double(nt) * 16.0 * double(nb2)) +
                           8.0 * double(std::max(nchunk, nchunk_y) * 2 * MMAX + nt) + 8.0 * double(m * m + MMAX + 16);
    size_t fr = 0, tot = 0;
    cu_check(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
    const double budget = 0.85 * double(fr) - 64.0 * 1024.0 * 1024.0;
    const long nbat = std::min(nblk, long(std::max(0.0, std::floor(budget / per_blk))));
    if (nbat < 1)
      APP_ABORT(std::string(" causal projection on the device: one (s, k) block needs ") + std::to_string(per_blk / 1048576.0) +
                " MB of device memory, " + std::to_string(double(fr) / 1048576.0) +
                " MB are free. Free device memory, or select the host path with vertex_debug scf_causal_proj_device = 0.");
    cudaStream_t sm = e->st;
    batch_bufs d;
    const size_t NE = size_t(nbat * nel), NY = size_t(nbat * nyel);
    d.V = dalloc<cd>(NE, "V"); d.C = dalloc<cd>(NE, "C"); d.Cp = dalloc<cd>(NE, "Cp"); d.B = dalloc<cd>(NE, "B");
    d.R = dalloc<cd>(NE, "R"); d.G = dalloc<cd>(NE, "G"); d.F = dalloc<cd>(NE, "F"); d.Gp = dalloc<cd>(NE, "Gp");
    d.Fp = dalloc<cd>(NE, "Fp"); d.KtY = dalloc<cd>(NE, "KtY"); d.Veig = dalloc<cd>(NE, "Veig");
    d.dG = dalloc<cd>(size_t(std::max(m, 1l)) * NE, "dG"); d.dF = dalloc<cd>(size_t(std::max(m, 1l)) * NE, "dF");
    d.Y = dalloc<cd>(NY, "Y"); d.M = dalloc<cd>(NY, "M");
    {
      const size_t gw_psd = lp.shared ? 0 : size_t(nbat * L) * lp.big() / sizeof(cd);
      const size_t gw_lam = ll.shared ? 0 : size_t(nbat * nt) * size_t(nb2);
      if (gw_psd + gw_lam > 0) d.gws = dalloc<cd>(std::max(gw_psd, gw_lam), "Jacobi workspace");
    }
    d.part = dalloc<double>(size_t(nbat) * size_t(std::max(nchunk, nchunk_y)) * 2 * MMAX, "partials");
    d.lam = dalloc<double>(size_t(nbat * nt), "lambda_min");
    // the control state: one int array (7 fields), one long array (2), one double array (g_best, nkty, gam, H)
    d.s.active = dalloc<int>(size_t(7 * nbat + 1), "state int");
    d.s.have_prev = d.s.active + nbat; d.s.n_hist = d.s.have_prev + nbat; d.s.slot = d.s.n_hist + nbat;
    d.s.newest = d.s.slot + nbat; d.s.converged = d.s.newest + nbat; d.s.err = d.s.converged + nbat;
    d.s.iters = dalloc<long>(size_t(2 * nbat), "state long");
    d.s.restarts = d.s.iters + nbat;
    d.s.g_best = dalloc<double>(size_t(nbat * (2 + MMAX + std::max(m, 1l) * std::max(m, 1l))), "state double");
    d.s.nkty = d.s.g_best + nbat; d.s.gam = d.s.nkty + nbat; d.s.H = d.s.gam + nbat * MMAX;
    d.sweeps = dalloc<int>(2, "sweeps"); d.sweeps_cold = d.sweeps + 1;
    d.calls = dalloc<unsigned long long>(3, "psd calls");
    d.prof = dalloc<unsigned long long>(4, "psd profile");
    cu_check(cudaMemsetAsync(d.prof, 0, 4 * sizeof(unsigned long long), sm), "memset prof");
    cu_check(cudaMemsetAsync(d.s.err, 0, sizeof(int), sm), "memset err");
    cu_check(cudaMemsetAsync(d.sweeps, 0, 2 * sizeof(int), sm), "memset sweeps");
    cu_check(cudaMemsetAsync(d.calls, 0, 3 * sizeof(unsigned long long), sm), "memset calls");
    cu_check(cudaMemsetAsync(d.s.H, 0, sizeof(double) * size_t(nbat * std::max(m, 1l) * std::max(m, 1l)), sm), "memset H");
    set_smem(psd_kernel, lp.smem());
    set_smem(lammin_kernel, lam_smem(ll));

    double t_entry = 0.0, t_loop = 0.0, t_exit = 0.0, t_kern[3] = {0.0, 0.0, 0.0};
    long nbatch = 0;
    std::vector<double> hpart, hlam;
    std::vector<int> hact(static_cast<size_t>(nbat));
    auto block_sums = [&](int nch, long nbc, std::vector<double> &out) {   // the per-block sums of norm2_kernel partials
      hpart.resize(size_t(nbc * nch));
      cu_check(cudaMemcpyAsync(hpart.data(), d.part, sizeof(double) * hpart.size(), cudaMemcpyDeviceToHost, sm), "D2H partials");
      cu_check(cudaStreamSynchronize(sm), "sync partials");
      out.assign(size_t(nbc), 0.0);
      for (long b = 0; b < nbc; ++b)
        for (int ch = 0; ch < nch; ++ch) out[size_t(b)] += hpart[size_t(b * nch + ch)];
    };
    auto lam_min_rows = [&](cd const *Ysrc, long nbc, std::vector<double> &out) {   // min over tau per block
      lammin_kernel<<<dim3(unsigned(nbc * nt)), NT, lam_smem(ll), sm>>>(ll, c.jacobi_tol, Ysrc, d.gws, d.lam, d.s.err, d.sweeps_cold);
      launch_check("lammin_kernel");
      hlam.resize(size_t(nbc * nt));
      cu_check(cudaMemcpyAsync(hlam.data(), d.lam, sizeof(double) * hlam.size(), cudaMemcpyDeviceToHost, sm), "D2H lambda");
      cu_check(cudaStreamSynchronize(sm), "sync lambda");
      out.assign(size_t(nbc), 1e300);
      for (long b = 0; b < nbc; ++b)
        for (long t = 0; t < nt; ++t) out[size_t(b)] = std::min(out[size_t(b)], hlam[size_t(b * nt + t)]);
    };

    for (long b0 = 0; b0 < nblk; b0 += nbat, ++nbatch) {
      const long nbc = std::min(nbat, nblk - b0);
      auto t0 = clk::now();
      // ---- entry: upload, Y, |Y|, lambda_min(Y), Kt^T Y, |Kt^T Y|, v = C + U, the control state -------------------------
      for (long i = 0; i < nbc; ++i) {
        cu_check(cudaMemcpyAsync(d.M + i * nyel, S[b0 + i], sizeof(cd) * size_t(nyel), cudaMemcpyHostToDevice, sm), "H2D S");
        cu_check(cudaMemcpyAsync(d.Cp + i * nel, C2[b0 + i], sizeof(cd) * size_t(nel), cudaMemcpyHostToDevice, sm), "H2D C");
        cu_check(cudaMemcpyAsync(d.V + i * nel, U2[b0 + i], sizeof(cd) * size_t(nel), cudaMemcpyHostToDevice, sm), "H2D U");
      }
      hermitian_y_kernel<<<grid1d(nbc * nyel), NT, 0, sm>>>(nbc * nt, int(nb), d.M, d.Y);
      launch_check("hermitian_y_kernel");
      entry_kernel<<<grid1d(nbc * nel), NT, 0, sm>>>(nbc * nel, d.V, d.Cp, d.Veig, nbc * L, int(nb));
      launch_check("entry_kernel");
      state_init_kernel<<<(unsigned(nbc) + NT - 1) / NT, NT, 0, sm>>>(int(nbc), c.maxit, d.s);
      launch_check("state_init_kernel");
      std::vector<double> y2, kty2, lam_b;
      norm2_kernel<<<dim3(unsigned(nchunk_y), unsigned(nbc)), NT, 0, sm>>>(nyel, nchunk_y, d.Y, nullptr, d.part);
      launch_check("norm2 Y");
      block_sums(nchunk_y, nbc, y2);
      lam_min_rows(d.Y, nbc, lam_b);
      // KtY = Kt^T Y: row-major (L, nt) x (nt, 2 nb^2)
      dgemm_sb(e->cb, 2 * nb2, L, nt, reinterpret_cast<double const *>(d.Y), 2 * nb2, 2ll * nyel, e->KtT, nt, 0,
               reinterpret_cast<double *>(d.KtY), 2 * nb2, 2ll * nel, nbc, "gemm KtY");
      norm2_kernel<<<dim3(unsigned(nchunk), unsigned(nbc)), NT, 0, sm>>>(nel, nchunk, d.KtY, nullptr, d.part);
      launch_check("norm2 KtY");
      block_sums(nchunk, nbc, kty2);
      {
        std::vector<double> nk(static_cast<size_t>(nbc));
        for (long b = 0; b < nbc; ++b) nk[size_t(b)] = std::max(std::sqrt(kty2[size_t(b)]), 1e-300);
        cu_check(cudaMemcpyAsync(d.s.nkty, nk.data(), sizeof(double) * size_t(nbc), cudaMemcpyHostToDevice, sm), "H2D nkty");
        cu_check(cudaStreamSynchronize(sm), "sync entry");
      }
      for (long b = 0; b < nbc; ++b) {
        st[b0 + b] = block_stats{};
        st[b0 + b].y_norm = std::sqrt(y2[size_t(b)]);
        st[b0 + b].lam_before = lam_b[size_t(b)];
      }
      auto t1 = clk::now();

      // ---- the ADMM iterations ---------------------------------------------------------------------------------------------
      psd_args pa{};
      pa.lay = lp; pa.L = int(L); pa.mode = 0; pa.X = d.V; pa.C = d.C; pa.R = d.R; pa.K = d.KtY; pa.rho = c.rho;
      pa.Veig = d.Veig; pa.gws = d.gws; pa.active = d.s.active; pa.converged = d.s.converged; pa.err = d.s.err;
      pa.sweeps = d.sweeps; pa.calls = d.calls; pa.jtol = c.jacobi_tol; pa.ns = 0;
      cd *pG = d.G, *pGp = d.Gp, *pF = d.F, *pFp = d.Fp;
      // c.profile: every kernel class timed with events (the stream is synchronised once per iteration)
      cudaEvent_t ev[4];
      if (c.profile)
        for (auto &x : ev) cu_check(cudaEventCreate(&x), "event");
      pa.prof = c.profile ? d.prof : nullptr;
      const dim3 gE{static_cast<unsigned>(nchunk), static_cast<unsigned>(nbc)};
      const unsigned gC = (unsigned(nbc) + NT - 1) / NT;
      for (long it = 0; it < c.maxit; ++it) {
        // C = PSD(v); R = Kt^T Y + rho (2 C - v)
        pa.ns = ((it + 1) % 8 == 0) ? 1 : 0;
        if (c.profile) cu_check(cudaEventRecord(ev[0], sm), "event");
        psd_kernel<<<dim3(unsigned(L), unsigned(nbc)), NT, lp.smem(), sm>>>(pa);
        launch_check("psd_kernel");
        if (c.profile) cu_check(cudaEventRecord(ev[1], sm), "event");
        // B = (Kt^T Kt + rho)^{-1} R: row-major (L, L) x (L, 2 nb^2)
        dgemm_sb(e->cb, 2 * nb2, L, L, reinterpret_cast<double const *>(d.R), 2 * nb2, 2ll * nel, e->Minv, L, 0,
                 reinterpret_cast<double *>(d.B), 2 * nb2, 2ll * nel, nbc, "gemm Minv R");
        if (c.profile) cu_check(cudaEventRecord(ev[2], sm), "event");
        resid_kernel<<<gE, NT, 0, sm>>>(nel, nchunk, c.relax, int(m), d.s, d.B, d.C, d.V, pG, pF, pGp, pFp, d.dG, d.dF, d.Cp, d.part);
        launch_check("resid_kernel");
        ctl1_kernel<<<gC, NT, 0, sm>>>(int(nbc), nchunk, it, c.tol, c.rho, c.relax, int(m), c.aa_restart, d.s, d.part);
        launch_check("ctl1_kernel");
        if (m > 0) {
          std::swap(pG, pGp);                              // g, f -> the previous ones of the next iteration
          std::swap(pF, pFp);
          dots_kernel<<<gE, NT, 0, sm>>>(nel, nchunk, int(m), d.s, d.dG, pGp, d.part);
          launch_check("dots_kernel");
          ctl2_kernel<<<gC, NT, 0, sm>>>(int(nbc), nchunk, int(m), c.aa_reg, d.s, d.part);
          launch_check("ctl2_kernel");
          update_kernel<<<gE, NT, 0, sm>>>(nel, nchunk, int(m), d.s, pFp, d.dF, d.V);
        } else {
          update_kernel<<<gE, NT, 0, sm>>>(nel, nchunk, 0, d.s, pF, d.dF, d.V);
        }
        launch_check("update_kernel");
        if (c.profile) {
          cu_check(cudaEventRecord(ev[3], sm), "event");
          cu_check(cudaEventSynchronize(ev[3]), "event sync");
          float ms[3];
          cu_check(cudaEventElapsedTime(&ms[0], ev[0], ev[1]), "event time");
          cu_check(cudaEventElapsedTime(&ms[1], ev[1], ev[2]), "event time");
          cu_check(cudaEventElapsedTime(&ms[2], ev[2], ev[3]), "event time");
          for (int i = 0; i < 3; ++i) t_kern[i] += 1e-3 * double(ms[i]);
        }
        if ((it + 1) % 25 == 0 and it + 1 < c.maxit) {   // stop when every block of the batch has converged
          cu_check(cudaMemcpyAsync(hact.data(), d.s.active, sizeof(int) * size_t(nbc), cudaMemcpyDeviceToHost, sm), "D2H active");
          cu_check(cudaStreamSynchronize(sm), "sync active");
          bool any = false;
          for (long b = 0; b < nbc; ++b) any = any or hact[size_t(b)] != 0;
          if (not any) break;
        }
      }
      cu_check(cudaStreamSynchronize(sm), "sync loop");
      if (c.profile)
        for (auto &x : ev) (void)cudaEventDestroy(x);
      auto t2 = clk::now();

      // ---- exit ---------------------------------------------------------------------------------------------------------
      // fit_res = |Kt (B - C)| / |Y| of the last iteration (of the converged one for a converged block: B, C kept since)
      sub_kernel<<<grid1d(nbc * nel), NT, 0, sm>>>(nbc * nel, d.B, d.C, d.R);
      launch_check("sub B - C");
      dgemm_sb(e->cb, 2 * nb2, nt, L, reinterpret_cast<double const *>(d.R), 2 * nb2, 2ll * nel, e->Kt, L, 0,
               reinterpret_cast<double *>(d.M), 2 * nb2, 2ll * nyel, nbc, "gemm Kt (B - C)");
      std::vector<double> fit2, rem2, lam_a;
      norm2_kernel<<<dim3(unsigned(nchunk_y), unsigned(nbc)), NT, 0, sm>>>(nyel, nchunk_y, d.M, nullptr, d.part);
      launch_check("norm2 fit");
      block_sums(nchunk_y, nbc, fit2);
      // the weights of the last iterate for the blocks that did not converge: C = PSD(v)
      pa.mode = 1;
      pa.ns = 0;
      psd_kernel<<<dim3(unsigned(L), unsigned(nbc)), NT, lp.smem(), sm>>>(pa);
      launch_check("psd_kernel (finish)");
      // C2 = C, U2 = v - C
      sub_kernel<<<grid1d(nbc * nel), NT, 0, sm>>>(nbc * nel, d.V, d.C, d.R);
      launch_check("sub V - C");
      for (long i = 0; i < nbc; ++i) {
        cu_check(cudaMemcpyAsync(C2[b0 + i], d.C + i * nel, sizeof(cd) * size_t(nel), cudaMemcpyDeviceToHost, sm), "D2H C");
        cu_check(cudaMemcpyAsync(U2[b0 + i], d.R + i * nel, sizeof(cd) * size_t(nel), cudaMemcpyDeviceToHost, sm), "D2H U");
      }
      // M = Kt C; removed = |Y - M|; lambda_min(M); S = -M
      dgemm_sb(e->cb, 2 * nb2, nt, L, reinterpret_cast<double const *>(d.C), 2 * nb2, 2ll * nel, e->Kt, L, 0,
               reinterpret_cast<double *>(d.M), 2 * nb2, 2ll * nyel, nbc, "gemm Kt C");
      norm2_kernel<<<dim3(unsigned(nchunk_y), unsigned(nbc)), NT, 0, sm>>>(nyel, nchunk_y, d.Y, d.M, d.part);
      launch_check("norm2 removed");
      block_sums(nchunk_y, nbc, rem2);
      lam_min_rows(d.M, nbc, lam_a);
      neg_kernel<<<grid1d(nbc * nyel), NT, 0, sm>>>(nbc * nyel, d.M);
      launch_check("neg_kernel");
      for (long i = 0; i < nbc; ++i)
        cu_check(cudaMemcpyAsync(S[b0 + i], d.M + i * nyel, sizeof(cd) * size_t(nyel), cudaMemcpyDeviceToHost, sm), "D2H S");
      std::vector<long> its(size_t(2 * nbc));
      std::vector<int> ist(size_t(7 * nbat + 1));
      cu_check(cudaMemcpyAsync(its.data(), d.s.iters, sizeof(long) * size_t(nbc), cudaMemcpyDeviceToHost, sm), "D2H iters");
      cu_check(cudaMemcpyAsync(its.data() + nbc, d.s.restarts, sizeof(long) * size_t(nbc), cudaMemcpyDeviceToHost, sm), "D2H restarts");
      cu_check(cudaMemcpyAsync(ist.data(), d.s.active, sizeof(int) * ist.size(), cudaMemcpyDeviceToHost, sm), "D2H state");
      cu_check(cudaStreamSynchronize(sm), "sync exit");
      const int err = ist[size_t(6 * nbat)];
      if (err & 1)
        APP_ABORT(std::string(" causal projection on the device: a Jacobi eigenvalue solve did not converge in ") +
                  std::to_string(MAX_SWEEPS) + " sweeps (non-finite input?).");
      if (err & 2)
        APP_ABORT(std::string(" causal projection on the device: singular Anderson least-squares system (the host path's "
                              "nda::inverse fails there too)."));
      for (long b = 0; b < nbc; ++b) {
        auto &s = st[b0 + b];
        const double ny = std::max(s.y_norm, 1e-300);
        s.iters = its[size_t(b)];
        s.aa_restarts = its[size_t(nbc + b)];
        s.converged = ist[size_t(5 * nbat + b)];
        s.fit_res = std::sqrt(fit2[size_t(b)]) / ny;
        s.removed = std::sqrt(rem2[size_t(b)]);
        s.lam_after = lam_a[size_t(b)];
      }
      auto t3 = clk::now();
      t_entry += secs(t0, t1);
      t_loop += secs(t1, t2);
      t_exit += secs(t2, t3);
    }
    if (info) {
      int sw[2] = {0, 0};
      unsigned long long cl[3] = {0, 0, 0};
      cu_check(cudaMemcpy(sw, d.sweeps, sizeof(sw), cudaMemcpyDeviceToHost), "D2H sweeps");
      cu_check(cudaMemcpy(cl, d.calls, sizeof(cl), cudaMemcpyDeviceToHost), "D2H calls");
      info->nbatch = nbatch;
      info->batch = nbat;
      info->sweeps_max = sw[0];
      info->sweeps_max_cold = sw[1];
      info->psd_calls = double(cl[0]);
      info->psd_kept = double(cl[1]);
      info->sweeps_mean = double(cl[2]) / std::max(1.0, double(cl[0] - cl[1]));
      if (c.profile) {
        unsigned long long pr[4];
        cu_check(cudaMemcpy(pr, d.prof, sizeof(pr), cudaMemcpyDeviceToHost), "D2H prof");
        const double tot = std::max(1.0, double(pr[0] + pr[1] + pr[2] + pr[3]));
        for (int i = 0; i < 4; ++i) info->psd_phase[i] = double(pr[i]) / tot;
        for (int i = 0; i < 3; ++i) info->t_kernels[i] = t_kern[i];
      }
      info->shared_mem = lp.shared;
      info->t_entry = t_entry;
      info->t_loop = t_loop;
      info->t_exit = t_exit;
      info->device_mb = per_blk * double(nbat) / 1048576.0;
    }
    d.free_all();
  }

  void psd_rows(cplx *X, long rows, long nb, bool force_global, long *sweeps_max, long *kept, double jacobi_tol) {
    if (rows <= 0) return;
    const long n2 = nb * nb;
    psd_layout lay = make_layout(nb, true);
    if (force_global or long(lay.smem()) > max_smem_optin() - STATIC_SMEM_MARGIN) lay.shared = false;
    cd *dX = dalloc<cd>(size_t(rows * n2), "X");
    cd *dVe = dalloc<cd>(size_t(rows * n2), "Veig");
    cd *gws = lay.shared ? nullptr : dalloc<cd>(size_t(rows) * lay.big() / sizeof(cd), "Jacobi workspace");
    int *flags = dalloc<int>(4, "flags");
    unsigned long long *cnt = dalloc<unsigned long long>(3, "counts");
    cu_check(cudaMemset(flags, 0, 4 * sizeof(int)), "memset");
    cu_check(cudaMemset(cnt, 0, 3 * sizeof(unsigned long long)), "memset");
    cu_check(cudaMemcpy(dX, X, sizeof(cd) * size_t(rows * n2), cudaMemcpyHostToDevice), "H2D X");
    {
      cd *dummy = dalloc<cd>(1, "dummy");
      entry_kernel<<<grid1d(rows * n2), NT, 0>>>(0, dummy, dummy, dVe, rows, int(nb));
      launch_check("entry_kernel");
      cu_check(cudaDeviceSynchronize(), "sync");
      (void)cudaFree(dummy);
    }
    psd_args pa{};
    pa.lay = lay; pa.L = int(rows); pa.mode = 2; pa.X = dX; pa.C = dX; pa.Veig = dVe; pa.gws = gws;
    pa.err = flags; pa.sweeps = flags + 1; pa.calls = cnt; pa.jtol = jacobi_tol; pa.ns = 0;
    set_smem(psd_kernel, lay.smem());
    psd_kernel<<<dim3(unsigned(rows), 1u), NT, lay.smem()>>>(pa);
    launch_check("psd_kernel (test)");
    int hf[4];
    unsigned long long hc[3];
    cu_check(cudaMemcpy(X, dX, sizeof(cd) * size_t(rows * n2), cudaMemcpyDeviceToHost), "D2H X");
    cu_check(cudaMemcpy(hf, flags, sizeof(hf), cudaMemcpyDeviceToHost), "D2H flags");
    cu_check(cudaMemcpy(hc, cnt, sizeof(hc), cudaMemcpyDeviceToHost), "D2H counts");
    (void)cudaFree(dX); (void)cudaFree(dVe); (void)cudaFree(gws); (void)cudaFree(flags); (void)cudaFree(cnt);
    if (hf[0]) APP_ABORT(std::string(" causal_proj_cuda::psd_rows: a Jacobi solve did not converge."));
    if (sweeps_max) *sweeps_max = hf[1];
    if (kept) *kept = long(hc[1]);
  }

} // namespace methods::causal_proj_cuda
