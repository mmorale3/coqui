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
 * IBZ k-point symmetry context for the ISDF-Vertex kernels
 * (notes/vertex_ibz_symmetry.md).
 *
 * Carries everything the kernels need to source their rung transfers from
 * IBZ-stored W/Z on a symmetry-reduced mesh:
 *   - per full-BZ transfer q': the qsymms position of the mapping symmetry,
 *     the IBZ transfer index, and the time-reversal flag;
 *   - the effective C-window collocation columns Xhat (memo (X-hat)): for a rung
 *     mapped by symmetry position js, leg k-point kL and window orbital j,
 *       js = 0        :  Xhat(s,0,kL,:,j)  = X(kL)(:, C0+j)          [stored X]
 *       non-trev kL   :  Xhat(s,js,kL,:,j) = sum_a X(krot(js,kL))(:, C0+a) Dc(a,j)
 *       trev kL       :  Xhat(s,js,kL,:,j) = conj( sum_a X(krot(js, pair(kL)))(:, C0+a) Dc(a,j) )
 *     with Dc the PLAIN C-block of MF->symmetry_rotation (no extra normalization
 *     -- the consumer precedent, projector_boson_t.cpp:108-121; the stored D is
 *     already row-normalized at the nbnd truncation, symmetry.hpp:1067-1092).
 *   - trev TRANSFERS (qp_trev(q')) require NO conjugation anywhere: the rotated
 *     transfer is -qs and the element is read from the qs storage by the exact
 *     dictionary side-swap = a PQ-TRANSPOSE of W/Z (memo (P2)).
 *
 * The kernels receive `sym_ctx const*`; nullptr (or !active) selects the
 * original no-symmetry code paths bit-identically.
 *
 * The C-window D-matrix leakage (memo (C-leak)) is a MEASURED DIAGNOSTIC, not a
 * gate (theory-owner ruling 2026-07-17): symmetry-unfolded vertex quantities
 * carry O(leakage) relative error, the C-window analogue of the nbnd-truncation
 * warning in generate_dmatrix.
 */

#include <memory>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"

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
    // P1, the star fold of the Sigma-side vertex
    nda::array<long, 1> ktrev_pair;   // bz_symmetry's pairs: defined (>= 0) ONLY for the points reached by time reversal alone
    nda::array<long, 1> qminus;
    // the index of -k (mod G) for EVERY full-BZ k (kminus(k) == k at the TRIM points). This, not ktrev_pair, is the map the
    // star fold needs: on a mesh with time-reversal images ktrev_pair is -1 at every other point (Si 4^3, 6 symmorphic
    // operations: segfault in minus_transfer_at, 2026-09-21); the LiH fixture has no time-reversal pairs and never saw it.
    nda::array<long, 1> kminus;

    // effective C-window collocation columns: (ns, nsym, nk_full, naux, nc).
    // NODE-SHARED (vertex parallelization M3, change-list item #9): the storage is a
    // per-NUMA-node shared_array (one copy per node, not per rank -- this is the large
    // sym member, ns*nsym*nk_full*naux*nc); `Xhat` is a VIEW into that window so all
    // consumers keep the historic `ctx.Xhat(is, jsym, k, all, all)` access unchanged.
    // The shared_array lives behind a shared_ptr so its MPI window has a STABLE address
    // and the view survives the `slot = std::move(ctx)` into the optional. On a single
    // rank per node this is bit-identical to the former replicated nda::array.
    std::shared_ptr<math::shm::shared_array<nda::array_view<ComplexType, 5>>> Xhat_shm;
    nda::array_view<ComplexType, 5> Xhat;

    // C-window D blocks Dc(js, k, a, j) (js >= 1; identity slot unused) and the
    // conjugation flags from symmetry_rotation (true for trev k) -- kept for the
    // G-rotation consistency diagnostic and for tests.
    nda::array<ComplexType, 4> Dc;
    nda::array<bool, 2> cjg;

    // measured C-window leakage diagnostic (memo section 6)
    double leak_max = 0.0;
    double leak_mean = 0.0;
    // measured unitarity defect max ||Dc^dag Dc - 1||_F of the C-sector rotation that
    // Xhat is built from -- the accuracy floor of the whole symmetry path.
    double d_unitarity_max = 0.0;
  };

  /**
   * P1 (vertex_perf_plan.md, 2026-09-21): the STAR FOLD of a Sigma-side vertex contribution. dSq(t, p, i, j) is the
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
  namespace detail {
    struct fold_scratch {
      nda::array<ComplexType, 2> X, Dm, T, Y;
      fold_scratch(long nc) : X(nc, nc), Dm(nc, nc), T(nc, nc), Y(nc, nc) {}
    };
    /** B(p) = Sigma_{-qs}(p) from the unit's dSq = Sigma_{qs}(.) (see fold_star_into_ibz); the result in `out` (nt x nc x nc) */
    template<class DSQ>
    inline void minus_transfer_at(sym_ctx const &c, long iq, DSQ const &dSq, long p, nda::array<ComplexType, 3> &out,
                                  fold_scratch &w, int depth) {
      decltype(nda::range::all) all;
      const long nt = dSq.shape(0), nc = dSq.shape(2);
      if (c.qminus(iq) == iq) {                                   // -qs = qs: a TRIM transfer
        for (long it = 0; it < nt; ++it) out(it, all, all) = dSq(it, p, all, all);
        return;
      }
      const long pm = c.kminus(p);
      if (pm != p) {                                              // psi_{-p} = psi_p^* : Sigma_{-qs}(p)_{ab} = conj(Sigma_{qs}(-p)_{ab})
        for (long it = 0; it < nt; ++it)
          for (long a = 0; a < nc; ++a)
            for (long b = 0; b < nc; ++b) out(it, a, b) = std::conj(dSq(it, pm, a, b));
        return;
      }
      // p is a TRIM point and -qs != qs: -qs must be a plain star image of qs, folded at p one level deep
      const long qm = c.qminus(iq);
      utils::check(depth == 0 and c.q_star(qm) == iq and not c.q_trev(qm) and not c.cjg(c.q_isym(qm), p),
                   "fold_star_into_ibz: Sigma_{{-q}} at a TRIM point p = {} for the non-TRIM IBZ transfer {} is not reachable "
                   "(-q = {}: star {}, trev {}, conjugated rotation {}). This star fold needs the -q unit; run with "
                   "pol_vertex_sigma_pair_ibz = false on this mesh.", p, iq, qm, c.q_star(qm), int(c.q_trev(qm)),
                   int(c.q_star(qm) == iq and c.cjg(c.q_isym(qm), p)));
      const long js = c.q_isym(qm), sp = c.krot(js, p);
      for (long a = 0; a < nc; ++a)
        for (long j = 0; j < nc; ++j) w.Dm(a, j) = c.Dc(js, p, a, j);
      for (long it = 0; it < nt; ++it) {
        for (long a = 0; a < nc; ++a)
          for (long b = 0; b < nc; ++b) w.X(a, b) = dSq(it, sp, a, b);
        nda::blas::gemm(w.X, w.Dm, w.T);
        nda::blas::gemm(nda::dagger(w.Dm), w.T, w.Y);
        out(it, all, all) = w.Y;
      }
    }
  }

  inline void fold_star_into_ibz(sym_ctx const &c, long iq, nda::ArrayOfRank<4> auto const &dSq, nda::ArrayOfRank<4> auto &&dSig,
                                 long *n_trev = nullptr, long *n_cjg = nullptr) {
    decltype(nda::range::all) all;
    const long nt = dSq.shape(0), nc = dSq.shape(2), nk_ibz = dSig.shape(1);
    utils::check(dSq.shape(1) == c.nk_full and dSig.shape(1) == c.nk_ibz and dSq.shape(3) == nc and dSig.shape(2) == nc,
                 "fold_star_into_ibz: shapes (dSq {} x {} x {} x {}, dSig {} x {} x {} x {}, nk_full {}, nk_ibz {}).",
                 dSq.shape(0), dSq.shape(1), dSq.shape(2), dSq.shape(3), dSig.shape(0), dSig.shape(1), dSig.shape(2), dSig.shape(3),
                 c.nk_full, c.nk_ibz);
    utils::check(c.qminus.size() == c.nq_full and c.kminus.size() == c.nk_full, "fold_star_into_ibz: the sym context has no qminus / kminus maps.");
    detail::fold_scratch w(nc);
    nda::array<ComplexType, 3> B(nt, nc, nc);
    for (long qp = 0; qp < c.nq_full; ++qp) {
      if (c.q_star(qp) != iq) continue;
      const long js = c.q_isym(qp);
      const bool trev = c.q_trev(qp);
      if (trev and n_trev) ++(*n_trev);
      for (long k = 0; k < nk_ibz; ++k) {
        const bool cj = (js != 0) and c.cjg(js, k);
        if (cj and n_cjg) ++(*n_cjg);
        const long pt = cj ? c.krot(js, c.kminus(k)) : c.krot(js, k);   // the conjugated rotation acts at -k (the full -k map, not ktrev_pair)
        const bool minus = (trev != cj);
        if (minus) detail::minus_transfer_at(c, iq, dSq, pt, B, w, 0);
        else for (long it = 0; it < nt; ++it) B(it, all, all) = dSq(it, pt, all, all);
        if (js == 0) {                                            // the identity slot (Dc unset): the IBZ transfer itself or its trev image
          for (long it = 0; it < nt; ++it)
            for (long i = 0; i < nc; ++i)
              for (long j = 0; j < nc; ++j) dSig(it, k, i, j) += B(it, i, j);
          continue;
        }
        for (long a = 0; a < nc; ++a)
          for (long j = 0; j < nc; ++j) w.Dm(a, j) = c.Dc(js, k, a, j);
        for (long it = 0; it < nt; ++it) {
          w.X() = B(it, all, all);
          nda::blas::gemm(w.X, w.Dm, w.T);                              // T = B D
          nda::blas::gemm(nda::dagger(w.Dm), w.T, w.Y);                 // Y = D^dag B D
          for (long i = 0; i < nc; ++i)
            for (long j = 0; j < nc; ++j) dSig(it, k, i, j) += cj ? std::conj(w.Y(i, j)) : w.Y(i, j);
        }
      }
    }
  }

} // vertex_sym
} // solvers
} // methods

#endif // COQUI_VERTEX_SYM_HPP
