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

#ifndef COQUI_METHODS_GW_LINE_SCF_MIXING_HPP
#define COQUI_METHODS_GW_LINE_SCF_MIXING_HPP

/**
 * Mixing of the line scGW fixed point (plan 7.2). The fixed-point map is x -> Sigma[G(x)], x = (Sigma^>, Sigma^<) at the
 * dense fermionic nodes of all k (the INPUT of the closure that built G), optionally with F (the static part used in that
 * closure). In iteration n the driver has the input x_{n-1} (state) and the output Sigma[G_{n-1}] (the new Sigma), the
 * residual r_{n-1} = out - in, and builds the next input:
 *
 *   "linear":  x_n = a out + (1 - a) x_{n-1}                    (a = mixing; F = F[D] unmixed; python LineSCGW, bitwise
 *                                                                  the pre-7.2 driver)
 *   "diis"  :  Anderson / Pulay with history: c = argmin |sum_i c_i r_i|^2, sum_i c_i = 1 (real c; B_ij = Re <r_i, r_j>,
 *              Tikhonov B + reg max(diag B) 1), x_n = sum_i c_i (x_i + beta r_i). One history entry: x_n = x + beta r.
 *              Before iteration `start` (damping of the first, large steps): the linear step with `mixing` (the pair is
 *              kept in the history). Safeguards: max|c_i| > cmax or a residual larger than `grow` x the smallest one
 *              in the history -> the history is reset to the newest pair and a linear step with `mixing` is taken.
 *              With mix_F the vector includes F (weight wF per element; the next closure uses the extrapolated F).
 *
 * Damped tail (any algorithm, damp_below > 0): once the residual max|out - in| drops below damp_below, every further step
 * is linear with damp_mixing (sticky; the DIIS history is dropped; kind "damped"). Undamped steps (mixing 1, DIIS) pass the
 * closure's discrete-decision noise (Gram cut, phase basins; ~1e-4 in Sigma for Si) straight into the next input, so
 * the iteration keeps hopping at that level; damped steps let the decisions lock (Si 2x2x2: mixing 1 floors at 1e-4,
 * mixing 0.5 / 0.7 reach 1e-5).
 *
 * The history is NOT checkpointed: after a restart it is rebuilt from the restored input (first step x + beta r).
 *
 * Inner products are rank-count independent: per-k partial sums (k-distributed rows: one nonzero contribution per
 * element in an all_reduce), summed over k in a fixed order; F is replicated and added afterwards.
 */

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

namespace methods::gw_line {

struct mixing_params_t {
  std::string alg = "linear";   ///< "linear" | "diis"
  double mixing   = 1.0;        ///< linear mixing (linear algorithm; DIIS warm-up / reset steps)
  long hist       = 6;          ///< DIIS history length
  long start      = 2;          ///< first iteration (1-based) that extrapolates; earlier iterations: linear with `mixing`
  double beta     = 1.0;        ///< DIIS step: x = sum c (x_i + beta r_i)
  double reg      = 1e-10;      ///< Tikhonov regularization of B, relative to max diag B
  double cmax     = 10.0;       ///< reset when max|c_i| exceeds this
  double grow     = 10.0;       ///< reset when |r_new| > grow x min_i |r_i| (divergence guard)
  bool mix_F      = false;      ///< include F in the DIIS vector
  double damp_below  = 3e-4;    ///< > 0: linear steps with damp_mixing once the residual is below this (sticky)
  double damp_mixing = 0.5;
  double wF       = -1.0;       ///< weight of the F elements in the inner product (< 0: the number of fermionic nodes)
};

/// what one mixing step did
struct mix_info_t {
  double dS    = 0.0;   ///< max |change of the input Sigma^> + Sigma^<| (the pre-7.2 dSigma)
  double resid = 0.0;   ///< max |out - in| of Sigma^> + Sigma^< (the fixed-point residual; linear: dS / mixing)
  double rnorm = 0.0;   ///< ||r|| (2-norm over everything in the vector)
  double residF = 0.0;  ///< max |F[D] - F_in| (the F residual; reported for every algorithm)
  std::string kind = "none";   ///< "linear" | "diis" | "reset" | "damped" | "none"
  long m = 0;           ///< history entries used
  double cmax = 0.0;    ///< max |c_i| of the DIIS step
};

class scf_mixer_t {
  using arr4 = nda::array<ComplexType, 4>;
  using arr3 = nda::array<ComplexType, 3>;
  struct entry_t {
    arr4 yp, yh, rp, rh;   ///< y = x + beta r, r = out - in
    arr3 yF, rF;
    double rn2 = 0.0;
  };

public:
  mixing_params_t prm;
  std::deque<entry_t> H;
  std::vector<std::vector<double>> B;   ///< B[i][j] = Re <r_i, r_j>
  bool damped = false;                  ///< the damped tail is active (damp_below)

  scf_mixer_t() = default;
  explicit scf_mixer_t(mixing_params_t p) : prm(std::move(p)) {}

  void reset() {
    H.clear();
    B.clear();
  }
  long size() const { return long(H.size()); }
  /// bytes of the history on this rank
  double bytes() const {
    double b = 0.0;
    for (auto const &e : H) b += 16.0 * double(e.yp.size() + e.yh.size() + e.rp.size() + e.rh.size() + e.yF.size() + e.rF.size());
    return b;
  }

  /**
   * One step. Sp_in / Sh_in: the input x (rows of this rank; k_rows = their global k; distributed = rows differ between
   * ranks). Sp / Sh: on entry the output Sigma[G], on exit the next input. F_in: the F of the closure that built G; F_out:
   * F[D] of G; F_next (out): the F of the next closure (F_out unless a DIIS step with mix_F). nk: number of k.
   */
  mix_info_t step(boost::mpi3::communicator &comm, long iter, long nk, std::vector<long> const &k_rows, bool distributed,
                  arr4 const &Sp_in, arr4 const &Sh_in, arr4 &Sp, arr4 &Sh, arr3 const &F_in, arr3 const &F_out, arr3 &F_next) {
    mix_info_t info;
    const long nloc = Sp.extent(0);
    utils::check(Sp_in.shape() == Sp.shape() and Sh_in.shape() == Sh.shape() and Sp.shape() == Sh.shape() and
                     long(k_rows.size()) == nloc,
                 "gw_line::scf_mixer_t: shape mismatch");
    const long row = Sp.size() / std::max(1L, nloc);
    // residual maxima (always) and F residual
    double rmax = 0.0;
    for (long a = 0; a < Sp.size(); ++a)
      rmax = std::max(rmax, std::abs(Sp.data()[a] + Sh.data()[a] - Sp_in.data()[a] - Sh_in.data()[a]));
    if (distributed) rmax = comm.all_reduce_value(rmax, boost::mpi3::max<>{});
    info.resid = rmax;
    double rF = 0.0;
    for (long a = 0; a < F_out.size(); ++a) rF = std::max(rF, std::abs(F_out.data()[a] - F_in.data()[a]));
    info.residF = rF;
    F_next      = F_out;

    if (prm.damp_below > 0.0 and (damped or rmax < prm.damp_below)) {   // damped tail (sticky)
      if (not damped)
        app_log(1, "  mixing: residual {:.3e} < damp_below {:.1e}: linear steps with mixing {} from now on", rmax, prm.damp_below,
                prm.damp_mixing);
      damped = true;
      reset();
      info.kind = "damped";
      linear_step(comm, distributed, Sp_in, Sh_in, Sp, Sh, info, prm.damp_mixing);
      return info;
    }
    if (prm.alg == "linear") {   // the pre-7.2 arithmetic, bitwise
      info.kind = "linear";
      linear_step(comm, distributed, Sp_in, Sh_in, Sp, Sh, info);
      return info;
    }

    // DIIS: the new pair
    const bool useF = prm.mix_F;
    const double wF = prm.wF >= 0.0 ? prm.wF : double(Sp.extent(1));
    entry_t e;
    e.rp = Sp;
    e.rh = Sh;
    e.rp -= Sp_in;
    e.rh -= Sh_in;
    e.yp = Sp_in;
    e.yh = Sh_in;
    for (long a = 0; a < e.yp.size(); ++a) {
      e.yp.data()[a] += prm.beta * e.rp.data()[a];
      e.yh.data()[a] += prm.beta * e.rh.data()[a];
    }
    if (useF) {
      e.rF = F_out;
      e.rF -= F_in;
      e.yF = F_in;
      for (long a = 0; a < e.yF.size(); ++a) e.yF.data()[a] += prm.beta * e.rF.data()[a];
    }
    // overlaps of the new residual with the history (and itself), rank-count independent
    const long m0 = long(H.size());
    std::vector<double> ov(m0 + 1, 0.0);
    {
      nda::array<double, 2> pk(nk, m0 + 1);
      pk() = 0.0;
      for (long l = 0; l < nloc; ++l) {
        const long k = k_rows[l];
        for (long i = 0; i <= m0; ++i) {
          auto const &rp = (i < m0) ? H[i].rp : e.rp;
          auto const &rh = (i < m0) ? H[i].rh : e.rh;
          double s = 0.0;
          for (long a = l * row; a < (l + 1) * row; ++a)
            s += std::real(std::conj(rp.data()[a]) * e.rp.data()[a]) + std::real(std::conj(rh.data()[a]) * e.rh.data()[a]);
          pk(k, i) = s;
        }
      }
      if (distributed and comm.size() > 1) comm.all_reduce_in_place_n(pk.data(), pk.size(), std::plus<>{});
      for (long i = 0; i <= m0; ++i) {
        for (long k = 0; k < nk; ++k) ov[i] += pk(k, i);
        if (useF) {
          auto const &rf = (i < m0) ? H[i].rF : e.rF;
          double s = 0.0;
          for (long a = 0; a < rf.size(); ++a) s += std::real(std::conj(rf.data()[a]) * e.rF.data()[a]);
          ov[i] += wF * s;
        }
      }
    }
    e.rn2      = ov[m0];
    info.rnorm = std::sqrt(std::max(0.0, e.rn2));
    // divergence guard: the new residual much larger than the best one in the history
    double best = e.rn2;
    for (auto const &h : H) best = std::min(best, h.rn2);
    const bool diverging = (m0 > 0 and e.rn2 > prm.grow * prm.grow * best);
    // grow the history
    for (long i = 0; i < m0; ++i) B[i].push_back(ov[i]);
    B.push_back(ov);
    H.push_back(std::move(e));
    while (long(H.size()) > std::max(1L, prm.hist)) {
      H.pop_front();
      B.erase(B.begin());
      for (auto &b : B) b.erase(b.begin());
    }
    const long m = long(H.size());

    auto newest_only = [&]() {
      H.erase(H.begin(), H.end() - 1);
      B = {{H.back().rn2}};
    };
    if (iter < prm.start) {   // damping of the first steps: linear (the pair stays in the history)
      info.kind = "linear";
      info.m    = m;
      linear_step(comm, distributed, Sp_in, Sh_in, Sp, Sh, info);
      return info;
    }
    if (diverging) {
      app_log(1, "  mixing: DIIS reset (|r| {:.3e} > {} x the best in the history {:.3e}): linear step with mixing {}",
              std::sqrt(H.back().rn2), prm.grow, std::sqrt(best), prm.mixing);
      newest_only();
      info.kind = "reset";
      info.m    = 1;
      linear_step(comm, distributed, Sp_in, Sh_in, Sp, Sh, info);
      return info;
    }
    // coefficients: (B + reg) c = 1, c /= sum c (B scaled by its largest diagonal)
    std::vector<double> c(m, 1.0);
    if (m > 1) {
      double dmax = 0.0;
      for (long i = 0; i < m; ++i) dmax = std::max(dmax, B[i][i]);
      dmax = std::max(dmax, 1e-300);
      std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
      for (long i = 0; i < m; ++i) {
        for (long j = 0; j < m; ++j) A[i][j] = B[i][j] / dmax + (i == j ? prm.reg : 0.0);
        A[i][m] = 1.0;
      }
      // Gaussian elimination with partial pivoting (m <= ~10)
      for (long p = 0; p < m; ++p) {
        long piv = p;
        for (long i = p + 1; i < m; ++i)
          if (std::abs(A[i][p]) > std::abs(A[piv][p])) piv = i;
        std::swap(A[p], A[piv]);
        const double d = A[p][p];
        if (std::abs(d) < 1e-300) continue;
        for (long i = p + 1; i < m; ++i) {
          const double f = A[i][p] / d;
          for (long j = p; j <= m; ++j) A[i][j] -= f * A[p][j];
        }
      }
      for (long i = m - 1; i >= 0; --i) {
        double s = A[i][m];
        for (long j = i + 1; j < m; ++j) s -= A[i][j] * c[j];
        c[i] = std::abs(A[i][i]) > 1e-300 ? s / A[i][i] : 0.0;
      }
      double sc = 0.0;
      for (double x : c) sc += x;
      bool bad = not std::isfinite(sc) or std::abs(sc) < 1e-300;
      if (not bad)
        for (auto &x : c) x /= sc;
      double cm = 0.0;
      for (double x : c) cm = std::max(cm, std::abs(x));
      info.cmax = cm;
      if (bad or not std::isfinite(cm) or cm > prm.cmax) {
        app_log(1, "  mixing: DIIS reset (max|c| {:.2e} > {}): linear step with mixing {}", cm, prm.cmax, prm.mixing);
        newest_only();
        info.kind = "reset";
        info.m    = 1;
        linear_step(comm, distributed, Sp_in, Sh_in, Sp, Sh, info);
        return info;
      }
    }
    info.kind = "diis";
    info.m    = m;
    if (m == 1) info.cmax = 1.0;
    // x_next = sum c_i y_i
    double dS = 0.0;
    for (long a = 0; a < Sp.size(); ++a) {
      ComplexType sp = 0.0, sh = 0.0;
      for (long i = 0; i < m; ++i) {
        sp += c[i] * H[i].yp.data()[a];
        sh += c[i] * H[i].yh.data()[a];
      }
      dS = std::max(dS, std::abs(sp + sh - Sp_in.data()[a] - Sh_in.data()[a]));
      Sp.data()[a] = sp;
      Sh.data()[a] = sh;
    }
    if (distributed) dS = comm.all_reduce_value(dS, boost::mpi3::max<>{});
    info.dS = dS;
    if (useF) {
      F_next() = ComplexType(0.0);
      for (long i = 0; i < m; ++i)
        for (long a = 0; a < F_next.size(); ++a) F_next.data()[a] += c[i] * H[i].yF.data()[a];
    }
    {
      std::string cs;
      for (long i = 0; i < m; ++i) cs += (i ? " " : "") + std::to_string(c[i]).substr(0, 8);
      app_log(2, "  mixing: DIIS {} entries, c = [{}]", m, cs);
    }
    return info;
  }

private:
  void linear_step(boost::mpi3::communicator &comm, bool distributed, arr4 const &Sp_in, arr4 const &Sh_in, arr4 &Sp, arr4 &Sh,
                   mix_info_t &info, double mix = -1.0) const {
    const double a = mix > 0.0 ? mix : prm.mixing, b = 1.0 - a;
    double dS = 0.0;
    for (long a_ = 0; a_ < Sp.size(); ++a_) {
      const ComplexType sp = a * Sp.data()[a_] + b * Sp_in.data()[a_];
      const ComplexType sh = a * Sh.data()[a_] + b * Sh_in.data()[a_];
      dS = std::max(dS, std::abs(sp + sh - Sp_in.data()[a_] - Sh_in.data()[a_]));
      Sp.data()[a_] = sp;
      Sh.data()[a_] = sh;
    }
    if (distributed) dS = comm.all_reduce_value(dS, boost::mpi3::max<>{});
    info.dS = dS;
  }
};

} // namespace methods::gw_line

#endif
