/**
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

#ifndef COQUI_METHODS_GW_LINE_KMESH_FFT_HPP
#define COQUI_METHODS_GW_LINE_KMESH_FFT_HPP

/**
 * perf 7.5c: the k-mesh Fourier transforms of the real-space convolutions (kmesh_ft.hpp) as 3-D FFTs on the
 * Monkhorst-Pack mesh instead of dense N x N gemms.
 *
 * Index map. kmesh_ft_t orders the lattice vectors R lexicographically, r = (a n2 + b) n3 + c, and takes every phase from
 * the fractional k (kpts_crystal) and from the index map qk_to_k2 (e^{iQ_q R} = e^{i (k_0 - k_{qk(q,0)}) R}). On an
 * unshifted mesh k_crys = j / n (mod 1) with integer j = (j1, j2, j3), so
 *   e^{+ikR}    = exp(+2 pi i sum_d j_d a_d / n_d),   kpos(k) = (j1 n2 + j2) n3 + j3,
 *   e^{+iQ_q R} = exp(+2 pi i sum_d m_d a_d / n_d),   qpos(q) = (m1 n2 + m2) n3 + m3,  m = j(k_0) - j(qk(q, 0)) mod n,
 * i.e. the gemm phases are exact roots of unity of the mesh and every transform of kmesh_ft.hpp is a DFT of the mesh with
 * the rows k (q) placed at kpos (qpos) and R in the FFT's row-major output order:
 *   sum_k e^{+ikR} A(k)        = FFT_{+}[A placed at kpos](R)        (FFTW_BACKWARD sign)
 *   sum_k e^{-ikR} B(k)        = FFT_{-}[B placed at kpos](R)        (FFTW_FORWARD sign)
 *   sum_q e^{-iQ_q R} w(q)     = FFT_{-}[w placed at qpos](R)
 *   sum_R e^{-iQ_q R} P(R)     = FFT_{-}[P](qpos(q)),   sum_R e^{+ikR} P(R) = FFT_{+}[P](kpos(k)).
 * The map is verified at construction against the gemm matrices of kmesh_ft_t (max |phase - Fp|, |phase - conj Hm|, 1e-12);
 * ok = false (the kernels keep the gemm path) on a shifted mesh or when a check fails. The hole leg's w^(-R) and the
 * -R map are unchanged (row swaps of kmesh_ft_t::minusR); the IBZ class sums use the same map for q_eff and ks.
 *
 * Layout. Every kernel array is (N rows, columns) row-major: the mesh index is the slowest, the (t, P, Q) columns are
 * contiguous. Host (FFTW, CoQui's fft_lib link): the columns are processed in cache blocks of width cb (buffers of N x cb,
 * allocated with fftw_malloc, one FFTW plan per (width, sign) with howmany = width, stride = width, distance = 1), so a
 * whole convolution (gather, two or three FFTs, product, scaled scatter) runs in L2 per block: the 3-5 memory passes of
 * the gemm form become one read of the inputs and one write of the outputs. Device (cuFFT, cuda/gw_line_fft.cu):
 * batched Z2Z plans directly on the chunk arrays (istride = row stride, idist = 1, batch = columns, in column blocks).
 *
 * Mode (kft_mode): env COQUI_GWLINE_KFT = "gemm" (the perf 7.1 dense gemms), "fft", or unset / "auto": FFT when the map is
 * valid, the build has the FFT library and N >= COQUI_GWLINE_KFT_NMIN (defaults kft_nmin_default (host, measured) and
 * kft_nmin_default_device (modelled), see below and the progress entry perf 7.5c).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "methods/GW_line/kmesh_ft.hpp"
#if defined(ENABLE_FFTW)
// CoQui's fft_define.hpp includes <fftw3.h> inside namespace math::fft (the C symbols are the same); include it first so the
// FFTW names are always math::fft::fftw_*
#include "numerics/fft/fft_define.hpp"
#endif
#if defined(ENABLE_CUDA)
#include "methods/GW_line/cuda/gw_line_fft.cuh"
#endif

namespace methods::gw_line {

/**
 * Crossover of the automatic mode (perf 7.5c). Host, measured ([.kft_bench], every core of a node busy): the blocked FFT
 * pipeline is 7.8x (rome) / 7.7x (genoa) faster than the gemms at 4^3, 13x at 6^3, 21-24x at 8^3, and the in-run Pi / Sigma
 * convolutions 2.7x (4^3) - 5.3x (6^3): FFT for every mesh. Device, modelled (not yet measured on a GPU): the dense zgemm
 * at ~15 TF/s vs a memory-bound cuFFT pass (2 N 16 bytes per column at ~1.5 TB/s) gives only ~1.5x at 4^3 and the strided
 * cuFFT layout may lose that: gemm below N = 100, FFT from 5^3 up (COQUI_GWLINE_KFT_NMIN overrides both).
 */
inline constexpr long kft_nmin_default        = 8;
inline constexpr long kft_nmin_default_device = 100;

/// mesh positions of k and Q_q (see the file header); built from kmesh_ft_t and verified against its gemm matrices
struct kmesh_map_t {
  bool ok = false;
  long N  = 0;
  std::array<long, 3> n = {0, 0, 0};
  std::vector<long> kpos, qpos;
  bool kid = false, qid = false;   ///< kpos / qpos are the identity (no row placement needed)
  double err_k = 0.0, err_q = 0.0;

  kmesh_map_t() = default;
  kmesh_map_t(kmesh_ft_t const &kft, mf::MF const &mf) {
    if (not kft.ok) return;
    N = kft.N;
    n = kft.mesh;
    auto kc = mf.kpts_crystal();
    auto qk = mf.qk_to_k2();
    const long nq = mf.nqpts();
    std::vector<std::array<long, 3>> j(N);
    double off = 0.0;
    for (long k = 0; k < N; ++k)
      for (int d = 0; d < 3; ++d) {
        const double x = double(kc(k, d)) * double(n[d]);
        const double r = std::round(x);
        off            = std::max(off, std::abs(x - r));
        j[k][d]        = ((long(r) % n[d]) + n[d]) % n[d];
      }
    if (off > 1e-6) {
      app_log(2, "  kmesh_map_t: shifted mesh (max |k n - round| {:.1e}): FFT path disabled", off);
      return;
    }
    auto lin = [&](std::array<long, 3> const &v) { return (v[0] * n[1] + v[1]) * n[2] + v[2]; };
    kpos.resize(N);
    qpos.resize(nq);
    for (long k = 0; k < N; ++k) kpos[k] = lin(j[k]);
    for (long q = 0; q < nq; ++q) {
      std::array<long, 3> m;
      for (int d = 0; d < 3; ++d) m[d] = ((j[0][d] - j[qk(q, 0)][d]) % n[d] + n[d]) % n[d];
      qpos[q] = lin(m);
    }
    // bijections
    std::vector<char> seen(N, 0), seenq(N, 0);
    bool bij = true;
    for (long k = 0; k < N; ++k) {
      if (seen[kpos[k]]) bij = false;
      seen[kpos[k]] = 1;
    }
    for (long q = 0; q < nq; ++q) {
      if (qpos[q] < 0 or qpos[q] >= N or seenq[qpos[q]]) bij = false;
      else seenq[qpos[q]] = 1;
    }
    kid = qid = true;
    for (long k = 0; k < N; ++k) kid = kid and kpos[k] == k;
    for (long q = 0; q < nq; ++q) qid = qid and qpos[q] == q;
    // the phases of the map vs the gemm matrices of kmesh_ft_t
    const double tp = 2.0 * std::numbers::pi;
    auto ph         = [&](long pos, long r) {   // exp(+2 pi i m . a / n), m = mesh vector of pos, a = R of row r
      const long m1 = pos / (n[1] * n[2]), m2 = (pos / n[2]) % n[1], m3 = pos % n[2];
      const long a1 = r / (n[1] * n[2]), a2 = (r / n[2]) % n[1], a3 = r % n[2];
      const long e1 = (m1 * a1) % n[0], e2 = (m2 * a2) % n[1], e3 = (m3 * a3) % n[2];
      double x      = double(e1) / double(n[0]) + double(e2) / double(n[1]) + double(e3) / double(n[2]);
      x -= std::floor(x);
      return std::exp(ComplexType(0.0, tp * x));
    };
    err_k = err_q = 0.0;
    for (long r = 0; r < N; ++r) {
      for (long k = 0; k < N; ++k) err_k = std::max(err_k, std::abs(ph(kpos[k], r) - kft.Fp(r, k)));
      for (long q = 0; q < nq; ++q) err_q = std::max(err_q, std::abs(ph(qpos[q], r) - std::conj(kft.Hm(r, q))));
    }
    ok = bij and nq == N and err_k < 1e-12 and err_q < 1e-12;
    app_log(ok ? 3 : 1, "  kmesh_map_t: mesh {}x{}x{}: phases vs the gemm matrices k {:.1e} Q {:.1e} (k order {}, q order {}){}",
            n[0], n[1], n[2], err_k, err_q, kid ? "= mesh" : "permuted", qid ? "= mesh" : "permuted",
            ok ? "" : " -> FFT path DISABLED");
  }
};

enum class kft_mode_t { gemm, fft };

/// was the FFT library compiled in for this memory space?
template <MEMORY_SPACE MEM>
constexpr bool kft_fft_available() {
#if defined(ENABLE_CUDA)
  if constexpr (MEM != HOST_MEMORY) return true;
#endif
#if defined(ENABLE_FFTW)
  if constexpr (MEM == HOST_MEMORY) return true;
#endif
  return false;
}

/// the transform mode of the real-space convolutions (see the file header); read at every call (tests switch it)
template <MEMORY_SPACE MEM>
kft_mode_t kft_mode(kmesh_map_t const &map) {
  char const *v = std::getenv("COQUI_GWLINE_KFT");
  std::string s = (v != nullptr) ? std::string(v) : std::string();
  if (s == "gemm") return kft_mode_t::gemm;
  if (not map.ok or not kft_fft_available<MEM>()) {
    utils::check(s != "fft" or kft_fft_available<MEM>(), "gw_line: COQUI_GWLINE_KFT = fft in a build without the FFT library");
    return kft_mode_t::gemm;
  }
  if (s == "fft") return kft_mode_t::fft;
  char const *nm = std::getenv("COQUI_GWLINE_KFT_NMIN");
  const long nmin = (nm != nullptr and *nm != '\0') ? std::strtol(nm, nullptr, 10)
                                                    : (MEM == HOST_MEMORY ? kft_nmin_default : kft_nmin_default_device);
  return map.N >= nmin ? kft_mode_t::fft : kft_mode_t::gemm;
}

/**
 * Host engine: work buffers of N x cb (fftw_malloc) and FFTW plans of the mesh on them. buf(i) is buffer i (i < nbuf) as
 * a row-major (N, cb) array; fft(i, w, sign) transforms its first w columns in place (sign +1: sum e^{+i...}, -1:
 * sum e^{-i...}). Plans are made once per (buffer, width, sign) with FFTW_MEASURE (planning overwrites the buffer: call
 * plan(i, w, sign) before filling it, or fft() plans lazily on first use and the caller fills after prepare()).
 */
struct kmesh_fft_host_t {
  long N = 0, cb = 0, nbuf = 0;
  std::array<long, 3> n = {0, 0, 0};
#if defined(ENABLE_FFTW)
  std::vector<math::fft::fftw_complex *> bufs;
  std::map<std::array<long, 3>, math::fft::fftw_plan> plans;   // (buffer, width, sign)
#endif

  kmesh_fft_host_t() = default;
  kmesh_fft_host_t(kmesh_fft_host_t const &)            = delete;
  kmesh_fft_host_t &operator=(kmesh_fft_host_t const &) = delete;

  /// cb <= 0: from env COQUI_GWLINE_FFT_CB, else 16384 / N (256 KB per buffer) clamped to [16, 1024], multiple of 8
  kmesh_fft_host_t(kmesh_map_t const &map, long nbuf_, long cb_ = 0) : N(map.N), nbuf(nbuf_), n(map.n) {
    char const *v = std::getenv("COQUI_GWLINE_FFT_CB");
    if (cb_ <= 0 and v != nullptr and *v != '\0') cb_ = std::strtol(v, nullptr, 10);
    if (cb_ <= 0) cb_ = 16384 / std::max(1L, N);
    cb = std::max(16L, std::min(1024L, (cb_ + 7) / 8 * 8));
#if defined(ENABLE_FFTW)
    bufs.resize(nbuf);
    for (auto &b : bufs) {
      b = math::fft::fftw_alloc_complex(size_t(N * cb));
      utils::check(b != nullptr, "gw_line::kmesh_fft_host_t: fftw_alloc_complex failed");
    }
#else
    utils::check(false, "gw_line::kmesh_fft_host_t: build without FFTW");
#endif
  }
  ~kmesh_fft_host_t() {
#if defined(ENABLE_FFTW)
    for (auto &p : plans) math::fft::fftw_destroy_plan(p.second);
    for (auto b : bufs) math::fft::fftw_free(b);
#endif
  }

  ComplexType *buf(long i) {
#if defined(ENABLE_FFTW)
    return reinterpret_cast<ComplexType *>(bufs[i]);
#else
    (void)i;
    return nullptr;
#endif
  }

  /// make (or find) the plan of buffer i, width w, sign; may overwrite the buffer when newly made
  void plan(long i, long w, int sign) {
#if defined(ENABLE_FFTW)
    std::array<long, 3> key = {i, w, long(sign)};
    if (plans.count(key)) return;
    int dims[3] = {int(n[0]), int(n[1]), int(n[2])};
    char const *v = std::getenv("COQUI_GWLINE_FFT_PLAN");   // "estimate" -> FFTW_ESTIMATE
    const unsigned flags = (v != nullptr and std::string(v) == "estimate") ? FFTW_ESTIMATE : FFTW_MEASURE;
    math::fft::fftw_plan p = math::fft::fftw_plan_many_dft(3, dims, int(w), bufs[i], nullptr, int(w), 1, bufs[i], nullptr, int(w), 1,
                                     sign > 0 ? FFTW_BACKWARD : FFTW_FORWARD, flags);
    utils::check(p != nullptr, "gw_line::kmesh_fft_host_t: fftw_plan_many_dft failed (mesh {}x{}x{}, width {})", n[0], n[1], n[2],
                 w);
    plans[key] = p;
#else
    (void)i, (void)w, (void)sign;
#endif
  }
  /// plans of every width a column range of ncols uses (call before the buffers are filled)
  void prepare(long ncols, std::initializer_list<std::pair<long, int>> bs) {
    for (auto const &[i, s] : bs) {
      plan(i, std::min(cb, ncols), s);
      if (ncols > cb and ncols % cb != 0) plan(i, ncols % cb, s);
    }
  }
  void fft(long i, long w, int sign) {
#if defined(ENABLE_FFTW)
    auto it = plans.find(std::array<long, 3>{i, w, long(sign)});
    utils::check(it != plans.end(), "gw_line::kmesh_fft_host_t: no plan for buffer {} width {} sign {} (prepare first)", i, w,
                 sign);
    math::fft::fftw_execute(it->second);
#else
    (void)i, (void)w, (void)sign;
#endif
  }
};

namespace detail {

/// host: dst(row r, 0..w) = src + r * lds for r < N (a strided block copy into a contiguous N x w buffer)
inline void block_in(ComplexType *dst, long N, long w, ComplexType const *src, long lds) {
  for (long r = 0; r < N; ++r) std::memcpy(dst + r * w, src + r * lds, sizeof(ComplexType) * size_t(w));
}
/// host: dst(row r) = conj(src row r)
inline void block_in_conj(ComplexType *dst, long N, long w, ComplexType const *src, long lds) {
  for (long r = 0; r < N; ++r) {
    ComplexType *d       = dst + r * w;
    ComplexType const *s = src + r * lds;
    for (long i = 0; i < w; ++i) d[i] = std::conj(s[i]);
  }
}
/// host: a[i] *= b[i] for a contiguous N x w block and a strided N-row source (row stride ldb); b == nullptr: contiguous
inline void block_mul(ComplexType *a, long N, long w, ComplexType const *b, long ldb) {
  double *x = reinterpret_cast<double *>(a);
  for (long r = 0; r < N; ++r) {
    double const *y = reinterpret_cast<double const *>(b + r * ldb);
    double *xr      = x + 2 * r * w;
    for (long i = 0; i < w; ++i) {   // explicit complex product (no __muldc3 NaN branch)
      const double ar = xr[2 * i], ai = xr[2 * i + 1], br = y[2 * i], bi = y[2 * i + 1];
      xr[2 * i]     = ar * br - ai * bi;
      xr[2 * i + 1] = ar * bi + ai * br;
    }
  }
}
/// host: dst + r * ldd = s * src(row pos[r]) for r < nr
inline void block_out(ComplexType *dst, long ldd, long nr, long w, ComplexType const *src, std::vector<long> const &pos, long off,
                      ComplexType s) {
  for (long r = 0; r < nr; ++r) {
    ComplexType const *a = src + pos[off + r] * w;
    ComplexType *d       = dst + r * ldd;
    if (s == ComplexType(1.0)) std::memcpy(d, a, sizeof(ComplexType) * size_t(w));
    else {
      double const *x = reinterpret_cast<double const *>(a);
      double *y       = reinterpret_cast<double *>(d);
      const double sr = s.real(), si = s.imag();
      for (long i = 0; i < w; ++i) {
        y[2 * i]     = sr * x[2 * i] - si * x[2 * i + 1];
        y[2 * i + 1] = sr * x[2 * i + 1] + si * x[2 * i];
      }
    }
  }
}

/// host: dst row pos[r] = src row r (r < nr; the other rows of dst untouched); accumulate: dst row pos[r] += src row r
inline void block_place(ComplexType *dst, long w, ComplexType const *src, long lds, long nr, long const *pos, bool accumulate = false) {
  for (long r = 0; r < nr; ++r) {
    ComplexType *d       = dst + pos[r] * w;
    ComplexType const *s = src + r * lds;
    if (accumulate)
      for (long i = 0; i < w; ++i) d[i] += s[i];
    else std::memcpy(d, s, sizeof(ComplexType) * size_t(w));
  }
}

/**
 * Rows of an (nrows, len) MEM array moved in place so that row r goes to row pos[r] (pos a permutation), by cycles with one
 * row of scratch (host memcpy / device copies).
 */
template <MEMORY_SPACE MEM>
void permute_rows(ComplexType *a, long nrows, long len, std::vector<long> const &pos) {
  std::vector<long> inv(nrows);
  for (long r = 0; r < nrows; ++r) inv[pos[r]] = r;
  std::vector<char> done(nrows, 0);
  memory::array<MEM, ComplexType, 1> tmp;
  auto row = [&](long r) { return memory::array_view<MEM, ComplexType, 1>(std::array<long, 1>{len}, a + r * len); };
  for (long s = 0; s < nrows; ++s) {
    if (done[s] or inv[s] == s) { done[s] = 1; continue; }
    if (tmp.size() != len) tmp = memory::array<MEM, ComplexType, 1>(len);
    tmp      = row(s);   // the old row s
    long i   = s;
    while (inv[i] != s) {   // row i receives the old row inv[i]
      row(i)  = row(inv[i]);
      done[i] = 1;
      i       = inv[i];
    }
    row(i)  = tmp;
    done[i] = 1;
  }
}

/**
 * Device: in-place or out-of-place 3-D FFT of the mesh on an (N, ld) row-major MEM array, columns [0, ncols), sign as the
 * host engine. DEVICE builds with CUDA only.
 */
inline void fft_mesh_device([[maybe_unused]] std::array<long, 3> const &n, [[maybe_unused]] ComplexType const *in,
                            [[maybe_unused]] ComplexType *out, [[maybe_unused]] long ncols, [[maybe_unused]] long ld,
                            [[maybe_unused]] int sign) {
#if defined(ENABLE_CUDA)
  cuda::fft_mesh(int(n[0]), int(n[1]), int(n[2]), in, out, ncols, ld, sign);
#else
  utils::check(false, "gw_line::fft_mesh_device: device build with CUDA required");
#endif
}

} // namespace detail

} // namespace methods::gw_line

#endif
