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

#ifndef COQUI_VERTEX_SYM_HPP
#define COQUI_VERTEX_SYM_HPP

/**
 * IBZ k-point symmetry context for the ISDF-Vertex kernels.
 *
 * Carries everything the kernels need to source their rung transfers from
 * IBZ-stored W/Z on a symmetry-reduced mesh:
 *   - per full-BZ transfer q': the qsymms position of the mapping symmetry,
 *     the IBZ transfer index, and the time-reversal flag;
 *   - the effective C-window collocation columns Xhat: for a rung
 *     mapped by symmetry position js, leg k-point kL and window orbital j,
 *       js = 0        :  Xhat(s,0,kL,:,j)  = X(kL)(:, C0+j)          [stored X]
 *       non-trev kL   :  Xhat(s,js,kL,:,j) = sum_a X(krot(js,kL))(:, C0+a) Dc(a,j)
 *       trev kL       :  Xhat(s,js,kL,:,j) = conj( sum_a X(krot(js, pair(kL)))(:, C0+a) Dc(a,j) )
 *     with Dc the PLAIN C-block of MF->symmetry_rotation (no extra normalization,
 *     as in projector_boson_t.cpp; the stored D is already row-normalized at the
 *     nbnd truncation, see symmetry.hpp).
 *   - trev TRANSFERS (qp_trev(q')) require NO conjugation anywhere: the rotated
 *     transfer is -qs and the element is read from the qs storage by the exact
 *     dictionary side-swap = a PQ-TRANSPOSE of W/Z.
 *
 * The kernels receive `sym_ctx const*`; nullptr (or !active) selects the
 * no-symmetry code paths.
 *
 * The C-window D-matrix leakage (the weight a rotated window orbital has outside
 * the window) is reported as a diagnostic, not enforced: symmetry-unfolded vertex
 * quantities carry O(leakage) relative error, the C-window analogue of the
 * nbnd-truncation warning in generate_dmatrix.
 */

#include <memory>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "methods/vertex/vertex_debug.hpp"

namespace methods {
namespace solvers {
namespace vertex_sym {

  struct sym_ctx {
    bool active = false;

    long ns = 0;
    long nk_full = 0, nk_ibz = 0;
    long nq_full = 0, nq_ibz = 0;
    long nsym = 0;          // number of qsymms entries (position 0 = identity)
    long nc = 0;            // C-window size
    long naux = 0;          // collocation row dimension (Np global / N_m secondary)

    // per full-BZ transfer q'
    nda::array<long, 1> q_isym;   // qsymms POSITION of qp_symm(q') (0 = identity)
    nda::array<long, 1> q_star;   // IBZ transfer index qp_to_ibz(q')
    nda::array<bool, 1> q_trev;   // qp_trev(q'): read W/Z at q_star PQ-TRANSPOSED

    // momentum rotation map: krot(js, k) = ks_to_k(js, k) (full-BZ, symmetry.hpp:564)
    nda::array<long, 2> krot;
    // time-reversal partner of every full-BZ k (kp_trev_pair) and the -q map of every full-BZ transfer (qminus);
    // used by the star fold of the Sigma-side vertex
    nda::array<long, 1> ktrev_pair;   // bz_symmetry's pairs: defined (>= 0) ONLY for the points reached by time reversal alone
    nda::array<long, 1> qminus;
    // the index of -k (mod G) for EVERY full-BZ k (kminus(k) == k at the TRIM points). This, not ktrev_pair, is the map the
    // star fold needs: on a mesh with time-reversal images ktrev_pair is -1 at every point not reached by time reversal
    // alone.
    nda::array<long, 1> kminus;

    // effective C-window collocation columns: (ns, nsym, nk_full, naux, nc).
    // NODE-SHARED: the storage is a per-NUMA-node shared_array (one copy per node, not
    // per rank -- this is the large sym member, ns*nsym*nk_full*naux*nc); `Xhat` is a
    // VIEW into that window, accessed as `ctx.Xhat(is, jsym, k, all, all)`.
    // The shared_array lives behind a shared_ptr so its MPI window has a STABLE address
    // and the view survives the `slot = std::move(ctx)` into the optional.
    std::shared_ptr<math::shm::shared_array<nda::array_view<ComplexType, 5>>> Xhat_shm;
    nda::array_view<ComplexType, 5> Xhat;

    // C-window D blocks Dc(js, k, a, j) (js >= 1; identity slot unused) and the
    // conjugation flags from symmetry_rotation (true for trev k) -- kept for the
    // G-rotation consistency diagnostic and for tests.
    nda::array<ComplexType, 4> Dc;
    nda::array<bool, 2> cjg;
    // WANNIER mode only (empty otherwise): the rotation the star fold of a Sigma-side contribution uses in place of Dc,
    //   Dfold(js, k) = U(k)^dag D U(ksrc)   (U(k)^T D U(ksrc) for a conjugated rotation),
    // with D the band rotation on the projector's window and ksrc the point Dc refers to. The fold transports an operator
    // from ksrc back to k, so the projector enters with k and ksrc exchanged relative to the leg rotation
    // Dc = U(ksrc)^dag D U(k) (conj(U(k)) when conjugated); for a band window (U = 1) both are D.
    nda::array<ComplexType, 4> Dfold;
    nda::array<ComplexType, 4> const &fold_rotation() const { return (Dfold.size() > 0) ? Dfold : Dc; }
    // WANNIER mode: whether the projector is time-reversal covariant, U(-k) = conj U(k) (vertex_t::build_sym_ctx measures
    // it; a diagnostic -- the star fold's Sigma_{-q} at a TRIM point always goes through the -q partner unit)
    bool trim_conj_ok = false;

    // C-window leakage diagnostic (max / mean over the rotated window orbitals)
    double leak_max = 0.0;
    double leak_mean = 0.0;
    // measured unitarity defect max ||Dc^dag Dc - 1||_F of the C-sector rotation that
    // Xhat is built from -- the accuracy floor of the whole symmetry path.
    double d_unitarity_max = 0.0;
  };

  /**
   * The index maps of a sym context from the mean field's symmetry tables (no basis, no collocation): the transfer
   * access tables (q_isym, q_star, q_trev), the momentum map krot, the time-reversal pairs, the complete -k and -q maps
   * and the conjugation flags of the rotations (symmetry_rotation). vertex_t::build_sym_ctx fills a context from here
   * and adds the basis-dependent members (Xhat, Dc, Dfold); the star-fold tests build an index-only context the same way.
   */
  template<class MFT>
  inline void fill_index_maps(sym_ctx &ctx, MFT const &MF, bool with_cjg = true) {
    ctx.nk_full = MF.nkpts();
    ctx.nk_ibz = MF.nkpts_ibz();
    ctx.nq_full = MF.nqpts();
    ctx.nq_ibz = MF.nqpts_ibz();
    auto qsymms = MF.qsymms();
    ctx.nsym = qsymms.extent(0);
    auto kp_trev_pair = MF.kp_trev_pair();

    // ---- q'-access tables ------------------------------------------------------------
    ctx.q_isym = nda::array<long, 1>(ctx.nq_full);
    ctx.q_star = nda::array<long, 1>(ctx.nq_full);
    ctx.q_trev = nda::array<bool, 1>(ctx.nq_full);
    for (long iq = 0; iq < ctx.nq_full; ++iq) {
      const int sidx = MF.qp_symm(iq);
      long js = -1;
      for (long i = 0; i < ctx.nsym; ++i)
        if (qsymms(i) == sidx) { js = i; break; }
      utils::check(js >= 0, "vertex_sym::fill_index_maps: qp_symm({}) = {} not found in qsymms.", iq, sidx);
      ctx.q_isym(iq) = js;
      ctx.q_star(iq) = MF.qp_to_ibz(iq);
      ctx.q_trev(iq) = MF.qp_trev(iq);
    }
    for (long iq = 0; iq < ctx.nq_ibz; ++iq)
      utils::check(ctx.q_star(iq) == iq and not ctx.q_trev(iq) and ctx.q_isym(iq) == 0,
                   "vertex_sym::fill_index_maps: IBZ q-point {} is not identity-mapped (star = {}, trev = {}, isym = {}).",
                   iq, ctx.q_star(iq), int(ctx.q_trev(iq)), ctx.q_isym(iq));

    // ---- momentum map: krot = ks_to_k (full-BZ rows) ----------------------------------
    ctx.krot = nda::array<long, 2>(ctx.nsym, ctx.nk_full);
    for (long is = 0; is < ctx.nsym; ++is)
      for (long ik = 0; ik < ctx.nk_full; ++ik)
        ctx.krot(is, ik) = MF.ks_to_k(int(is), int(ik));
    ctx.ktrev_pair = nda::array<long, 1>(ctx.nk_full);
    for (long ik = 0; ik < ctx.nk_full; ++ik) ctx.ktrev_pair(ik) = long(kp_trev_pair(ik));
    {
      // the complete -k map (crystal coordinates, mod G): kp_trev_pair is -1 wherever the point is not a time-reversal image
      auto kc = MF.kpts_crystal();
      ctx.kminus = nda::array<long, 1>(ctx.nk_full);
      ctx.kminus() = -1;
      for (long ik = 0; ik < ctx.nk_full; ++ik) {
        for (long jk = 0; jk < ctx.nk_full and ctx.kminus(ik) < 0; ++jk) {
          double d = 0.0;
          for (int a = 0; a < 3; ++a) { const double x = kc(ik, a) + kc(jk, a); d += std::abs(x - std::round(x)); }
          if (d < 1e-8) ctx.kminus(ik) = jk;
        }
        utils::check(ctx.kminus(ik) >= 0, "vertex_sym::fill_index_maps: the k mesh has no -k for k-point {} (not a Gamma-centered lattice).", ik);
        utils::check(kp_trev_pair(ik) < 0 or long(kp_trev_pair(ik)) == ctx.kminus(ik),
                     "vertex_sym::fill_index_maps: kp_trev_pair({}) = {} is not -k = {}.", ik, kp_trev_pair(ik), ctx.kminus(ik));
      }
    }
    ctx.qminus = nda::array<long, 1>(ctx.nq_full);
    {
      auto qm = MF.qminus();
      for (long iq = 0; iq < ctx.nq_full; ++iq) ctx.qminus(iq) = long(qm(iq));
    }
    // the conjugation flags of the rotations (js >= 1): symmetry_rotation's first return (build_sym_ctx fills them in its
    // distributed tile loop instead, with_cjg = false)
    ctx.cjg = nda::array<bool, 2>(ctx.nsym, ctx.nk_full);
    ctx.cjg() = false;
    if (with_cjg)
      for (long js = 1; js < ctx.nsym; ++js)
      for (long ik = 0; ik < ctx.nk_full; ++ik) {
        auto [cj, Dsp] = MF.symmetry_rotation(js, ik, 0);   // the flag is that of the k map, the same for all spins
        ctx.cjg(js, ik) = cj;
      }
  }

  /**
   * The STAR FOLD of a Sigma-side vertex contribution. dSq(t, p, i, j) is the
   * contribution of ONE unit -- the ladder solved at the IBZ transfer iq in the identity frame -- to the C-window
   * self-energy at every p of the full mesh, in the mean field's band basis at p. Every image q' of the star of iq
   * (q_star(q') = iq; S_js q' = +qs, or -qs for a time-reversal image, js = q_isym(q')) contributes to the IBZ externals
   * the covariant object (the single-transfer rule of Sigma^{C,r}, vertex_sigma_r.icc; both external legs sit at k):
   *
   *     Sigma_{q'}(k)_{ij} = < R psi_{k,i} | Sigma_{S q'}(S k) | R psi_{k,j} >,
   *
   * with the transported orbitals R psi_{k,j} exactly as the kernels build them (Xhat, build_sym_ctx):
   *     no conjugation (cjg(js, k) false):  R psi_{k,j} =      sum_a Dc(js, k)(a, j) psi_{S k, a},      S k = krot(js, k),
   *     conjugated     (cjg(js, k) true):   R psi_{k,j} = conj(sum_a Dc(js, k)(a, j) psi_{ksrc, a}),   ksrc = krot(js, -k).
   * Hence (Dc = Dc(js, k), B(p) = the unit's object at p for the transfer the covariance needs)
   *     plain:       Y = Dc^dag B(S k) Dc,            B = Sigma_{qs}   (trev = false)  or  Sigma_{-qs}  (trev = true),
   *     conjugated:  Y = conj(Dc^dag B(ksrc) Dc),     B = Sigma_{-qs}  (trev = false)  or  Sigma_{qs}   (trev = true),
   * because the conjugated matrix element turns the kernel into its complex conjugate, Sigma_{q}(p)(r, r')^* =
   * Sigma_{-q}(-p)(r, r') (time reversal). Sigma_{qs}(p) = dSq(p); Sigma_{-qs}(p) is dSq(p) when -qs = qs (a TRIM transfer),
   * conj(dSq(-p)) when p and -p are a time-reversal pair of the mesh (psi_{-p} = psi_p^* in the code's gauge), and
   * otherwise the (non-conjugated) star image of -qs at p, one level deep. Anything else is refused with a message.
   * n_trev / n_cjg count the time-reversal images and the conjugated rotations applied (the log reports them).
   */
  /** DIAGNOSTIC: vertex_debug sym_trev_notrans = 1 reads the IBZ-stored core PLAIN (not PQ-transposed) for a time-reversed
   *  transfer at every rung read site. The two differ by W vs conj(W) for a Hermitian core: identical on meshes whose
   *  k-points are all TRIM (real collocations), different otherwise.
   *  The switch alters the answer on any mesh with a non-TRIM time-reversed transfer, so the first read logs a level-1
   *  WARNING. EVERY rung / W-bar read site honours it, Sigma included (vertex_sigma.icc q_access, vertex_sigma_pair.icc's
   *  outer W-bar, vertex_dynbse.icc / build_kbig), so Pi and Sigma always run on the same rungs. Callers reach this only
   *  for a transfer whose q_trev is set (they short-circuit on it), so the WARNING appears exactly when the switch changes
   *  something. */
  inline bool trev_read_transposed() {
    static const bool notrans = [] {
      const bool on = vertex_debug::flag("sym_trev_notrans");   // vertex_debug: sym_trev_notrans (DIAGNOSTIC)
      if (on)
        app_log(1, "  [WARNING] vertex_debug sym_trev_notrans is set: the IBZ-stored rung / W-bar core is read PLAIN "
                   "(NOT PQ-transposed) for every\n"
                   "            time-reversed transfer at every read site (Pi ladder, Pi^C, Sigma^C, the Sigma pair "
                   "vertex, the dynamic rung). DIAGNOSTIC\n"
                   "            switch (time-reversal convention check): it CHANGES the answer on any "
                   "mesh with non-TRIM k-points; unset it for production.");
      return on;
    }();
    return not notrans;
  }

  /**
   * The star fold as a TERM TABLE. For the IBZ transfer iq, the fold dSig(k) += ... over the star images of iq and the IBZ
   * externals k (fold_star_into_ibz below) is a sum of terms
   *     dSig(:, k) += (cflag ? conj : id)[ N^dag X(:, src) N ]        (N = I when `identity`)
   * on the unit's full-k scratch X (one (nc x nc) matrix per tau / node row). Every branch of the fold (a plain star image, a
   * time-reversal image through Sigma_{-q}(p) = conj(Sigma_q(-p)), the TRIM-point image through the -q rotation, the conjugated
   * rotations) is one such term: the conjugation of the source is absorbed into N (N^dag conj(X) N = conj(N'^dag X N'), N' =
   * conj(N)) and the conjugation of the result into cflag. The host fold applies the table in place; the device fold (the
   * Sigma deposits on the device engine) applies the same table with batched gemms, so both evaluate the same arithmetic.
   */
  struct fold_term {
    long k = -1, src = -1;                 // the IBZ external, the full-k source row of the scratch
    bool identity = true, cflag = false;   // N = I; conjugate the result
    nda::array<ComplexType, 2> N;          // (nc, nc) unless identity
  };
  struct fold_terms {
    std::vector<fold_term> terms;
    // the terms sourced from the PARTNER UNIT (the ladder solved at -q): Sigma_{-q} at a TRIM point p when -q is only a
    // time-reversal image of q and the frame has no time-reversal gauge there (sym_ctx::trim_conj_ok false). Empty when the
    // conjugation route is available or no term needs it. The partner unit's scratch holds its own full-k deposits; these
    // terms read it at src = p with no rotation of the source.
    std::vector<fold_term> partner;
    long n_trev = 0, n_cjg = 0;            // the time-reversal images and the conjugated rotations of the star (the log's counts)
    bool needs_partner() const { return not partner.empty(); }
  };

  namespace detail {
    /** the -q source of a time-reversal image at p: Sigma_{-qs}(p) from the unit's Sigma_{qs}: (src, conj_in, N1 or empty) */
    inline void minus_transfer_source(sym_ctx const &c, long iq, long p, long nc, long &src, bool &conj_in, nda::array<ComplexType, 2> &N1,
                                      bool &from_partner) {
      src = p; conj_in = false; N1 = nda::array<ComplexType, 2>(); from_partner = false;
      if (c.qminus(iq) == iq) return;                             // -qs = qs: a TRIM transfer
      const long pm = c.kminus(p);
      if (pm != p) { src = pm; conj_in = true; return; }          // psi_{-p} = psi_p^* : Sigma_{-qs}(p)_{ab} = conj(Sigma_{qs}(-p)_{ab})
      // p is a TRIM point and -qs != qs: -qs must be a plain star image of qs, folded at p one level deep -- or, in the
      // MLWF frame of a time-reversal-covariant projector (real Wannier functions: the Bloch sums at p = -p are real),
      // Sigma_{-qs}(p) = conj(Sigma_{qs}(p)) directly
      const long qm = c.qminus(iq);
      if (c.q_star(qm) != iq or c.q_trev(qm) or c.cjg(c.q_isym(qm), p)) {
        // -qs is only a time-reversal image of qs: the -q PARTNER UNIT's own deposit at p. (A conjugation shortcut
        // Sigma_{-qs}(p) = conj Sigma_{qs}(p) would need the frame's Bloch functions to be REAL at p; U(-k) = conj U(k) does
        // not guarantee that -- the band gauge at a TRIM point is the mean field's -- so it is not used.)
        from_partner = true; src = p; return;
      }
      utils::check(c.q_star(qm) == iq and not c.q_trev(qm) and not c.cjg(c.q_isym(qm), p),
                   "fold_star_into_ibz: Sigma_{{-q}} at a TRIM point p = {} for the non-TRIM IBZ transfer {} is not reachable "
                   "(-q = {}: star {}, trev {}, conjugated rotation {}). In the band frame this star fold needs the -q unit "
                   "(or the time-reversal gauge at p); in Wannier mode a time-reversal-covariant projector gives it by conjugation.", p, iq, qm, c.q_star(qm), int(c.q_trev(qm)),
                   int(c.q_star(qm) == iq and c.cjg(c.q_isym(qm), p)));
      const long js = c.q_isym(qm);
      src = c.krot(js, p);
      auto const &DD = c.fold_rotation();
      N1 = nda::array<ComplexType, 2>(nc, nc);
      for (long a = 0; a < nc; ++a)
        for (long j = 0; j < nc; ++j) N1(a, j) = DD(js, p, a, j);  // B = N1^dag X(src) N1
    }
  }

  /** the term table of the IBZ transfer iq (nc = the C-sector size; see fold_term) */
  inline fold_terms fold_star_terms(sym_ctx const &c, long iq, long nc) {
    utils::check(c.qminus.size() == c.nq_full and c.kminus.size() == c.nk_full, "fold_star_terms: the sym context has no qminus / kminus maps.");
    // THE C-SECTOR ROTATION CONVENTION OF THE FOLD.
    // A star member's deposit is Y = Dm^dag B Dm with Dm = Dc^T, i.e. Y = conj(Dc) B Dc^T, NOT Dc^dag B Dc. The two agree
    // wherever Dc is diagonal (a window with no degenerate bands) and differ inside degenerate blocks. The convention is
    // fixed by the anti-Hermitian residual of dSigma BEFORE Hermitization (which the Hermitization would otherwise hide):
    // with Dm = Dc^T the IBZ fold reproduces the full-mesh residual, with Dm = Dc it does not.
    // vertex_debug sym_fold_conv selects Dm for diagnostics: 1 = Dc^T (default), 0 = Dc, 2 = Dc^*, 3 = Dc^dag.
    const bool fold_dt = vertex_debug::flag("sym_fold_dt") or vertex_debug::flag("sym_dt");
    const long fold_conv = fold_dt ? 1 : long(vertex_debug::number("sym_fold_conv", 1.0));
    auto const &DD = c.fold_rotation();   // Dc, or the Wannier-mode fold rotation
    fold_terms out;
    nda::array<ComplexType, 2> Dm(nc, nc), N1, T(nc, nc);
    for (long qp = 0; qp < c.nq_full; ++qp) {
      if (c.q_star(qp) != iq) continue;
      const long js = c.q_isym(qp);
      const bool trev = c.q_trev(qp);
      if (trev) ++out.n_trev;
      for (long k = 0; k < c.nk_ibz; ++k) {
        const bool cj = (js != 0) and c.cjg(js, k);
        if (cj) ++out.n_cjg;
        const long pt = cj ? c.krot(js, c.kminus(k)) : c.krot(js, k);   // the conjugated rotation acts at -k (the full -k map, not ktrev_pair)
        const bool minus = (trev != cj);
        fold_term t;
        t.k = k;
        bool conj_in = false, from_partner = false;
        if (minus) detail::minus_transfer_source(c, iq, pt, nc, t.src, conj_in, N1, from_partner);
        else { t.src = pt; N1 = nda::array<ComplexType, 2>(); }
        if (js != 0) {
          for (long a = 0; a < nc; ++a)
            for (long j = 0; j < nc; ++j)
              Dm(a, j) = (fold_conv == 1) ? DD(js, k, j, a) : (fold_conv == 2) ? std::conj(DD(js, k, a, j))
                       : (fold_conv == 3) ? std::conj(DD(js, k, j, a)) : DD(js, k, a, j);
          if (N1.size() > 0) { nda::blas::gemm(N1, Dm, T); t.N = T; }   // Y = Dm^dag (N1^dag X N1) Dm = (N1 Dm)^dag X (N1 Dm)
          else t.N = Dm;
          t.identity = false;
        } else if (N1.size() > 0) { t.N = N1; t.identity = false; }
        // N^dag conj(X) N = conj(conj(N)^dag X conj(N)): the source conjugation moves into N and the result's flag
        if (conj_in and not t.identity) for (auto &v : t.N) v = std::conj(v);
        t.cflag = (cj != conj_in);
        if (from_partner) out.partner.push_back(std::move(t));
        else out.terms.push_back(std::move(t));
      }
    }
    return out;
  }

  /** apply a term table: dSq (nt, nk_full, nc, nc) the unit's scratch, dSig (nt, nk_ibz, nc, nc) += the fold */
  inline void apply_fold_terms(fold_terms const &ft, nda::ArrayOfRank<4> auto const &dSq, nda::ArrayOfRank<4> auto &&dSig) {
    decltype(nda::range::all) all;
    const long nt = dSq.shape(0), nc = dSq.shape(2);
    nda::array<ComplexType, 2> X(nc, nc), T(nc, nc), Y(nc, nc);
    for (auto const &t : ft.terms) {
      for (long it = 0; it < nt; ++it) {
        if (t.identity) {
          for (long i = 0; i < nc; ++i)
            for (long j = 0; j < nc; ++j) dSig(it, t.k, i, j) += t.cflag ? std::conj(dSq(it, t.src, i, j)) : dSq(it, t.src, i, j);
          continue;
        }
        X() = dSq(it, t.src, all, all);
        nda::blas::gemm(X, t.N, T);                                // T = X N
        nda::blas::gemm(nda::dagger(t.N), T, Y);                   // Y = N^dag X N
        for (long i = 0; i < nc; ++i)
          for (long j = 0; j < nc; ++j) dSig(it, t.k, i, j) += t.cflag ? std::conj(Y(i, j)) : Y(i, j);
      }
    }
  }

  /**
   * UNFOLD a C-sector object from the IBZ to the full mesh with the fold's rotation convention: the fold says
   * Sigma(k) = (cj ? conj : id)[Dm^dag Sigma(p) Dm] for p = S k (p = S(-k) under a conjugated rotation), so
   * Sigma(p) = Dm (cj ? conj(Sigma(k)) : Sigma(k)) Dm^dag. Y (nt, nk_ibz, nc, nc) -> X (nt, nk_full, nc, nc).
   * Every full-mesh point is reached from several (js, k) (the little group of k); X(p) is the AVERAGE over them, which
   * projects Y onto its little-group-symmetric part (the terms agree exactly when Y commutes with the little-group
   * representation). `consistency` (optional) returns max |Y(k) - (cj ? conj : id)[Dm^dag X(p) Dm]| / max |Y| over all
   * (js, k): 0 for a symmetric Y; a finite value measures how far the input is from symmetric (THC / aux-basis noise).
   */
  inline nda::array<ComplexType, 4> unfold_from_ibz(sym_ctx const &c, nda::ArrayOfRank<4> auto const &Y, double *consistency = nullptr) {
    decltype(nda::range::all) all;
    const long nt = Y.shape(0), nc = Y.shape(2);
    utils::check(Y.shape(1) == c.nk_ibz and Y.shape(3) == nc, "unfold_from_ibz: Y is ({} x {} x {} x {}), nk_ibz {}.", Y.shape(0), Y.shape(1),
                 Y.shape(2), Y.shape(3), c.nk_ibz);
    const bool fold_dt = vertex_debug::flag("sym_fold_dt") or vertex_debug::flag("sym_dt");
    const long fold_conv = fold_dt ? 1 : long(vertex_debug::number("sym_fold_conv", 1.0));
    auto const &DD = c.fold_rotation();
    nda::array<ComplexType, 4> X(nt, c.nk_full, nc, nc);
    X() = ComplexType(0.0);
    std::vector<long> count(size_t(c.nk_full), 0l);
    nda::array<ComplexType, 2> Dm(nc, nc), B(nc, nc), T(nc, nc);
    auto rotation = [&](long js, long k) {
      for (long a = 0; a < nc; ++a)
        for (long j = 0; j < nc; ++j)
          Dm(a, j) = (fold_conv == 1) ? DD(js, k, j, a) : (fold_conv == 2) ? std::conj(DD(js, k, a, j))
                   : (fold_conv == 3) ? std::conj(DD(js, k, j, a)) : DD(js, k, a, j);
    };
    for (long js = 0; js < c.nsym; ++js)
      for (long k = 0; k < c.nk_ibz; ++k) {
        const bool cj = (js != 0) and c.cjg(js, k);
        const long p = cj ? c.krot(js, c.kminus(k)) : c.krot(js, k);
        ++count[size_t(p)];
        if (js == 0) { X(all, p, all, all) += Y(all, k, all, all); continue; }
        rotation(js, k);
        for (long it = 0; it < nt; ++it) {
          for (long i = 0; i < nc; ++i)
            for (long j = 0; j < nc; ++j) B(i, j) = cj ? std::conj(Y(it, k, i, j)) : Y(it, k, i, j);
          nda::blas::gemm(Dm, B, T);                                 // T = Dm B
          nda::blas::gemm(T, nda::dagger(Dm), B);                    // B = Dm B Dm^dag
          X(it, p, all, all) += B;
        }
      }
    for (long p = 0; p < c.nk_full; ++p)
      if (count[size_t(p)] > 1) X(all, p, all, all) *= ComplexType(1.0 / double(count[size_t(p)]));
    // the points reached by TIME REVERSAL alone (p = -S k on a mesh reduced with time reversal): the mean field builds them as
    // psi_p = conj(psi_{-p}) (ktrev_pair(p) = -p), and the frame obeys U(p) = conj U(-p) there (a non-TRIM pair), so
    // Sigma(p) = conj Sigma(-p) -- the relation the star fold uses for its time-reversal images (minus_transfer_source)
    long n_tr = 0;
    for (long p = 0; p < c.nk_full; ++p) {
      if (count[size_t(p)] > 0) continue;
      const long pp = (c.ktrev_pair.size() == c.nk_full) ? c.ktrev_pair(p) : -1;
      utils::check(pp >= 0 and pp != p and count[size_t(pp)] > 0,
                   "unfold_from_ibz: the full-mesh point {} is neither the image of an IBZ point nor the time-reversal partner of one "
                   "(ktrev_pair {}).", p, pp);
      for (long it = 0; it < nt; ++it)
        for (long i = 0; i < nc; ++i)
          for (long j = 0; j < nc; ++j) X(it, p, i, j) = std::conj(X(it, pp, i, j));
      ++n_tr;
    }
    if (n_tr > 0 and vertex_debug::flag("sym_unfold_verbose"))
      app_log(1, "    [unfold] {} full-mesh points filled through time reversal (Sigma(p) = conj Sigma(-p)).", n_tr);
    if (consistency != nullptr) {
      double res = 0.0, sc = 0.0;
      for (auto const &v : Y) sc = std::max(sc, std::abs(v));
      for (long js = 0; js < c.nsym; ++js)
        for (long k = 0; k < c.nk_ibz; ++k) {
          const bool cj = (js != 0) and c.cjg(js, k);
          const long p = cj ? c.krot(js, c.kminus(k)) : c.krot(js, k);
          if (js != 0) rotation(js, k);
          double r1 = 0.0;
          for (long it = 0; it < nt; ++it) {
            B() = X(it, p, all, all);
            if (js != 0) { nda::blas::gemm(nda::dagger(Dm), B, T); nda::blas::gemm(T, Dm, B); }
            for (long i = 0; i < nc; ++i)
              for (long j = 0; j < nc; ++j) r1 = std::max(r1, std::abs(Y(it, k, i, j) - (cj ? std::conj(B(i, j)) : B(i, j))));
          }
          res = std::max(res, r1);
          if (vertex_debug::flag("sym_unfold_verbose"))   // vertex_debug: sym_unfold_verbose
            app_log(1, "    [unfold check] js {} k {} -> p {} (cj {}, little group {}): residual {:.3e} (scale {:.3e})", js, k, p, int(cj),
                    int(p == k), r1, sc);
        }
      *consistency = (sc > 0.0) ? res / sc : res;
    }
    return X;
  }

  inline void fold_star_into_ibz(sym_ctx const &c, long iq, nda::ArrayOfRank<4> auto const &dSq, nda::ArrayOfRank<4> auto &&dSig,
                                 long *n_trev = nullptr, long *n_cjg = nullptr) {
    const long nc = dSq.shape(2);
    utils::check(dSq.shape(1) == c.nk_full and dSig.shape(1) == c.nk_ibz and dSq.shape(3) == nc and dSig.shape(2) == nc,
                 "fold_star_into_ibz: shapes (dSq {} x {} x {} x {}, dSig {} x {} x {} x {}, nk_full {}, nk_ibz {}).",
                 dSq.shape(0), dSq.shape(1), dSq.shape(2), dSq.shape(3), dSig.shape(0), dSig.shape(1), dSig.shape(2), dSig.shape(3),
                 c.nk_full, c.nk_ibz);
    auto ft = fold_star_terms(c, iq, nc);
    utils::check(not ft.needs_partner(), "fold_star_into_ibz: the star of transfer {} needs the -q unit (Sigma_{{-q}} at a TRIM point); "
                 "this entry point has none.", iq);
    if (n_trev) *n_trev += ft.n_trev;
    if (n_cjg) *n_cjg += ft.n_cjg;
    apply_fold_terms(ft, dSq, dSig);
  }

} // vertex_sym
} // solvers
} // methods

#endif // COQUI_VERTEX_SYM_HPP
