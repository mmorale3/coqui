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

#ifndef COQUI_METHODS_GW_LINE_IBZ_HPP
#define COQUI_METHODS_GW_LINE_IBZ_HPP

/**
 * IBZ reduction of the line GW (perf 7.3, notes section "IBZ", plan section 7.3) on a SYMMETRIC mean field (QE with
 * force_symmorphic: point operations S + time reversal; mf::MF tables kp_to_ibz, kp_trev, qp_to_ibz, qp_trev, qsymms, Qs,
 * ks_to_k and the band D matrices MF::symmetry_rotation). The pattern is the one of the imaginary-axis code
 * (gw_t::eval_Sigma_all_kspace, thc_hf.icc): the THC auxiliary basis is NOT rotated (the ISDF points are no symmetric set
 * and the reader holds Z(q) for the IBZ q only); the EXTERNAL k is rotated instead.
 *
 *  (G)  Lehmann poles at the IBZ k only. The orbitals at a full-BZ k' are the rotated IBZ orbitals (time-reversed ones
 *       conjugated), so G(k') = G(k_I) in the k' orbital basis, with v_m -> conj(v_m) (coef -> coef^T) when kp_trev(k').
 *       The X(k') of all full-BZ k are used as they are (thc.X).
 *  (R)  Rows R = IBZ q  u  {-q : q in IBZ} (closed under q -> -q). For a virtual row -q (not an IBZ q) the Coulomb matrix
 *       is DEFINED as Z(-q) := conj Z(q) (the convention zeta_{-q} = conj zeta_q that CoQui's trev branches use: they take
 *       conj(W(q)) for W(-q)); Pi(-q) is computed directly (X at all k). With it the mirror relations of perf 7.1 (b)/(c)
 *       hold EXACTLY (no THC asymmetry), so the W stage runs screened_mirror on the rows R: |R| Dyson rows instead of N_q.
 *  (S)  Sigma (and Sigma_x) at an IBZ k: the full-BZ q sum split into the classes of qsymms. For q' with class isym and
 *       IBZ partner q_s = qp_to_ibz(q'), the effective transfer is q_eff = q_s (qp_trev = false) or -q_s (true), a row of
 *       R; the class sum is evaluated at the rotated point ks = ks_to_k(isym, k) in the auxiliary basis,
 *         A_isym(ks) = sum_{q' in class} G~(ks - q_eff) o W(q_eff),
 *       contracted with X(ks) and rotated to the k orbitals by D = symmetry_rotation(isym, k):
 *         Sigma_ij(k) = sum_isym [ (X(ks) D)^dagger A_isym(ks) (X(ks) D) ]_ij      (D folded into the X slices: XD).
 *
 * Built on every rank from the MF (replicated, small). For a mesh without symmetry reduction everything is trivial
 * (active = false): R = all q, one class (identity) with q_eff = q, ks = k, D = 1.
 */

#include <algorithm>
#include <set>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"
#include "numerics/sparse/csr_blas.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "methods/GW_line/line_state.hpp"

namespace methods::gw_line {

struct ibz_t {
  bool active = false;
  long nk = 0, nkI = 0, nq = 0, nqI = 0, nb = 0;
  std::vector<long> k2i;      ///< (nk) IBZ k of every full-BZ k (kp_to_ibz)
  std::vector<char> ktrev;    ///< (nk) the orbitals of k are time-reversed IBZ orbitals (kp_trev)
  std::vector<long> qminus;   ///< (nq) index of -q
  std::vector<long> rows;     ///< R: IBZ q (0..nqI-1) first, then the virtual rows -q (ascending)
  std::vector<long> rpos;     ///< (nq) position of q in rows, or -1
  std::vector<char> virt;     ///< (nq) q is a virtual row (Z(q) := conj Z(qminus(q)))

  struct class_t {
    long isym = 0;              ///< index in qsymms
    std::vector<long> q_eff;    ///< effective transfer momentum (absolute q index, a row of R) of every q' of the class
    std::vector<long> ks;       ///< (nkI) rotated point ks_to_k(isym, k) of every IBZ k
  };
  std::vector<class_t> cls;
  /// dense D(isym, k) (nb x nb, rows: bands at ks, columns: bands at k), [class][k]; empty for the identity class
  std::vector<std::vector<nda::array<ComplexType, 2>>> D;
  nda::array<double, 1> kw;   ///< (nkI) star weights (number of full-BZ k of each IBZ k) / nk

  ibz_t() = default;

  /// use_sym = false: the trivial tables even on a symmetric MF (only valid where nothing needs Z at non-IBZ q)
  ibz_t(mf::MF &mf, long nb_, bool use_sym = true) : nb(nb_) {
    nk  = mf.nkpts();
    nq  = mf.nqpts();
    nkI = use_sym ? mf.nkpts_ibz() : nk;
    nqI = use_sym ? mf.nqpts_ibz() : nq;
    active = (nkI < nk) or (nqI < nq);
    auto qm = mf.qminus();
    qminus.resize(nq);
    for (long q = 0; q < nq; ++q) qminus[q] = qm(q);
    k2i.resize(nk);
    ktrev.assign(nk, 0);
    rpos.assign(nq, -1);
    virt.assign(nq, 0);
    kw = nda::array<double, 1>(nkI);
    kw() = 0.0;
    if (not active) {
      for (long k = 0; k < nk; ++k) k2i[k] = k;
      for (long q = 0; q < nq; ++q) {
        rows.push_back(q);
        rpos[q] = q;
      }
      class_t c;
      c.isym = 0;
      for (long q = 0; q < nq; ++q) c.q_eff.push_back(q);
      for (long k = 0; k < nkI; ++k) c.ks.push_back(k);
      cls.push_back(std::move(c));
      D.resize(1);
      kw() = 1.0 / double(nk);
      return;
    }
    utils::check(nqI == nkI, "gw_line::ibz_t: Gamma-centred meshes only (nq_ibz {} != nk_ibz {})", nqI, nkI);
    auto kti = mf.kp_to_ibz();
    auto ktr = mf.kp_trev();
    for (long k = 0; k < nk; ++k) {
      k2i[k]   = kti(k);
      ktrev[k] = ktr(k) ? 1 : 0;
      utils::check(k2i[k] >= 0 and k2i[k] < nkI, "gw_line::ibz_t: kp_to_ibz({}) = {} out of the IBZ", k, k2i[k]);
      kw(k2i[k]) += 1.0 / double(nk);
    }
    for (long k = 0; k < nkI; ++k)
      utils::check(k2i[k] == k and not ktrev[k], "gw_line::ibz_t: the IBZ k must be the first nk_ibz points (k {})", k);
    // rows R
    for (long q = 0; q < nqI; ++q) rows.push_back(q);
    std::set<long> vr;
    for (long q = 0; q < nqI; ++q)
      if (qminus[q] >= nqI) vr.insert(qminus[q]);
    for (long q : vr) {
      rows.push_back(q);
      virt[q] = 1;
    }
    for (long i = 0; i < long(rows.size()); ++i) rpos[rows[i]] = i;
    for (long q : rows) utils::check(rpos[qminus[q]] >= 0, "gw_line::ibz_t: R not closed under -q at q = {}", q);
    // classes
    auto qsym = mf.qsymms();
    auto nqs  = mf.nq_per_s();
    auto Qs   = mf.Qs();
    auto qtr  = mf.qp_trev();
    auto q2i  = mf.qp_to_ibz();
    auto kst  = mf.ks_to_k();
    long ncov = 0;
    for (long is = 0; is < long(qsym.size()); ++is) {
      if (nqs(is) == 0) continue;
      class_t c;
      c.isym = is;
      for (long j = 0; j < nqs(is); ++j) {
        const long qp = Qs(is, j), qs = q2i(qp);
        utils::check(qs >= 0 and qs < nqI, "gw_line::ibz_t: qp_to_ibz({}) = {}", qp, qs);
        const long qe = qtr(qp) ? qminus[qs] : qs;
        utils::check(rpos[qe] >= 0, "gw_line::ibz_t: effective transfer {} (q' {}) not in R", qe, qp);
        c.q_eff.push_back(qe);
      }
      ncov += nqs(is);
      for (long k = 0; k < nkI; ++k) c.ks.push_back(kst(is, k));
      cls.push_back(std::move(c));
    }
    utils::check(ncov == nq, "gw_line::ibz_t: the q classes cover {} of {} q", ncov, nq);
    utils::check(cls[0].isym == 0, "gw_line::ibz_t: the first q class must be the identity (qsymms(0) = {})", long(qsym(0)));
    // D matrices (dense)
    D.resize(cls.size());
    nda::array<ComplexType, 2> I(nb, nb);
    I() = ComplexType(0.0);
    for (long i = 0; i < nb; ++i) I(i, i) = ComplexType(1.0);
    for (long c = 0; c < long(cls.size()); ++c) {
      if (cls[c].isym == 0) {
        for (long k = 0; k < nkI; ++k)
          utils::check(cls[c].ks[k] == k, "gw_line::ibz_t: identity class maps k {} to {}", k, cls[c].ks[k]);
        continue;
      }
      D[c].resize(nkI);
      for (long k = 0; k < nkI; ++k) {
        auto [cjg, Dp] = mf.symmetry_rotation(cls[c].isym, k);
        utils::check(not cjg, "gw_line::ibz_t: symmetry_rotation({}, {}) composes with time reversal at an IBZ k", cls[c].isym, k);
        utils::check(Dp->shape()[0] == nb and Dp->shape()[1] == nb, "gw_line::ibz_t: D({}, {}) is {} x {}, nbnd {}", cls[c].isym,
                     k, Dp->shape()[0], Dp->shape()[1], nb);
        D[c][k] = nda::array<ComplexType, 2>(nb, nb);
        math::sparse::csrmm<'N'>(ComplexType(1.0), *Dp, I, ComplexType(0.0), D[c][k]);
      }
    }
  }

  long nrows() const { return long(rows.size()); }
  long nclasses() const { return long(cls.size()); }
  /// largest deviation of D^dagger D from the identity (nbnd truncation of the stored D; reported, not gated)
  double d_unitarity() const {
    double e = 0.0;
    for (auto const &v : D)
      for (auto const &d : v) {
        nda::matrix<ComplexType> m(d);
        nda::matrix<ComplexType> p = nda::dagger(m) * m;
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) e = std::max(e, std::abs(p(i, j) - (i == j ? 1.0 : 0.0)));
      }
    return e;
  }
  void log(int lvl = 1) const {
    if (not active) {
      app_log(lvl, "  IBZ: none (mesh without symmetry reduction): {} k, {} q", nk, nq);
      return;
    }
    long nv = 0;
    for (auto v : virt) nv += v;
    std::string s;
    for (auto const &c : cls) s += std::to_string(c.isym) + ":" + std::to_string(c.q_eff.size()) + " ";
    long ntr = 0;
    for (auto t : ktrev) ntr += t;
    app_log(lvl, "  IBZ (perf 7.3): {} of {} k ({} time-reversed), {} of {} q; Pi / W rows {} ({} IBZ + {} virtual -q, Z(-q) = conj Z(q)); "
                 "Sigma classes (qsymms index: size) {}; max|D^+D - 1| = {:.1e}",
            nkI, nk, ntr, nqI, nq, nrows(), nqI, nv, s, d_unitarity());
  }
};

/**
 * (G) the poles of every full-BZ k from the IBZ poles: k' -> k_I = k2i[k'], coef -> conj(coef) (v -> conj(v)) when the
 * orbitals of k' are time-reversed (G(k', tau) = conj G(k_I, tau), as primary_to_aux with kp_trev). Identity when not active.
 */
inline pole_data_t unfold_poles(pole_data_t const &pI, ibz_t const &ibz) {
  if (not ibz.active) return pI;
  utils::check(pI.nk == ibz.nkI, "gw_line::unfold_poles: {} k of poles, IBZ has {}", pI.nk, ibz.nkI);
  pole_data_t pd;
  pd.nk = ibz.nk;
  pd.nb = pI.nb;
  pd.part.resize(pd.nk);
  pd.hole.resize(pd.nk);
  auto one = [&](pole_sector_t const &ps, bool tr) {
    if (not tr) return ps;
    if (ps.is_factorized()) return pole_sector_t::factorized_form(ps.e, nda::array<ComplexType, 2>(nda::conj(ps.v)));
    nda::array<ComplexType, 3> c(ps.coef.shape());
    for (long m = 0; m < ps.size(); ++m)
      c(m, nda::range::all, nda::range::all) = nda::conj(ps.coef(m, nda::range::all, nda::range::all));
    return pole_sector_t(ps.e, std::move(c));
  };
  for (long k = 0; k < pd.nk; ++k) {
    pd.part[k] = one(pI.part[ibz.k2i[k]], ibz.ktrev[k]);
    pd.hole[k] = one(pI.hole[ibz.k2i[k]], ibz.ktrev[k]);
  }
  return pd;
}

/// a per-k array (nkI, ...) of IBZ matrices unfolded to the full BZ: A(k') = A(k_I), or conj A(k_I) for time-reversed k'
inline nda::array<ComplexType, 3> unfold_matrices(nda::array<ComplexType, 3> const &AI, ibz_t const &ibz) {
  if (not ibz.active) return AI;
  utils::check(AI.extent(0) == ibz.nkI, "gw_line::unfold_matrices: {} rows, IBZ has {}", AI.extent(0), ibz.nkI);
  nda::array<ComplexType, 3> A(ibz.nk, AI.extent(1), AI.extent(2));
  for (long k = 0; k < ibz.nk; ++k) {
    auto src = AI(ibz.k2i[k], nda::range::all, nda::range::all);
    if (ibz.ktrev[k]) A(k, nda::range::all, nda::range::all) = nda::conj(src);
    else A(k, nda::range::all, nda::range::all) = src;
  }
  return A;
}

} // namespace methods::gw_line

#endif
