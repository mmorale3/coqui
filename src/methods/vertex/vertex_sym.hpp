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

} // vertex_sym
} // solvers
} // methods

#endif // COQUI_VERTEX_SYM_HPP
