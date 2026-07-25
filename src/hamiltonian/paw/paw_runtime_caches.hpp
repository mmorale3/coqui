/**
 * @file paw_runtime_caches.hpp
 *
 * Plan A4: lazily built, keyed runtime caches for the PAW/USPP augmentation
 * machinery, hoisting work that used to be rebuilt inside every
 * compute_paw_deeq / v_x / becsum-symm call (audit finding F5):
 *
 *   1. aainit angular-coupling tables            (paw_aatab)
 *   2. per-species qrad interpolation tables     (paw_qrad_tabs; ONE dq=0.01
 *      for all consumers, plan A4)
 *   3. the full-BZ View-2 Pskna lift             (Pskna_full_bz; shm-backed,
 *      MPI-collective on the pseudopot's communicator at first build)
 *   4. the Δk-keyed Qfac pair-factor cache       (get_or_build_qfac_pair_factor)
 *
 * Every cached object is a pure function of the immutable pseudopot state
 * plus the explicit key arguments (Kmax, mode, Δk), so entries are never
 * stale — a mode change re-keys rather than invalidates silently. The cache
 * struct is held by shared_ptr on the pseudopot and therefore shared across
 * copies.
 *
 * This header defines the pseudopot cache accessors declared in pseudopot.h;
 * it lives outside pseudopot.h because the cached types (aainit_tables,
 * qrad_tab, the lift) are defined in paw headers that themselves include
 * pseudopot.h.
 */

#ifndef HAMILTONIAN_PAW_RUNTIME_CACHES_HPP
#define HAMILTONIAN_PAW_RUNTIME_CACHES_HPP

#include <map>
#include <memory>
#include <vector>

#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"
#include "hamiltonian/paw/paw_symmetry.hpp"

namespace hamilt::paw {

struct runtime_caches {
    // (1) angular coupling tables; lli = 1 + max l over all projectors.
    int aatab_lli = 0;                 // 0 = not built yet
    aainit_tables aatab;

    // (2) per-species qrad tables. Key: (Kmax, shape_restored, aug_lmax);
    // dq is pinned project-wide to 0.01 (plan A4). A table built to a larger
    // Kmax is exact for smaller requests (same uniform interpolation nodes).
    double qtab_Kmax = -1.0;
    bool qtab_shape_restored = false;
    int qtab_aug_lmax = -999;
    std::vector<qrad_tab> qtab_sp;

    // (3) full-BZ Pskna lift (built once; no key — Pskna and the symmetry
    // data are immutable after construction).
    std::shared_ptr<math::shm::shared_array<nda::array_view<ComplexType,4>>>
        Pskna_full;

    // (3b) Eq. (h0) static USPP/PAW D: Dnn_atom_static + ∫V_loc·Q̂ (settled
    // 2026-07-24 — the frozen electrostatic compensation term is always in
    // the static assembly; dion holds only the one-center descreening
    // reference of opposite sign, plan Eq. d0). Built once (V_loc, Q̂ frozen).
    bool h0_static_built = false;
    nda::array<ComplexType,3> h0_static_D;

    // (4) Δk-keyed Qfac cache. First-come-stays under a byte budget: cached
    // Δk are reused deterministically (spin loop, repeated v_x calls); once
    // the budget is exhausted, further Δk are built into caller scratch (no
    // eviction thrash). Context (mesh, Gcut, mode) is part of the validity
    // key — the cache is cleared when it changes.
    std::array<long,3> qfac_mesh{{0,0,0}};
    double qfac_Gcut = -1.0;
    bool qfac_shape_restored = false;
    long qfac_bytes = 0;
    long qfac_budget_bytes = 256l << 20;   // 256 MB/rank default (knob: C3)
    long qfac_hits = 0, qfac_builds = 0, qfac_uncached = 0;
    std::map<std::array<long,3>, nda::array<ComplexType,3>> qfac;
};

} // namespace hamilt::paw

namespace hamilt {

inline paw::runtime_caches& pseudopot::paw_rt() const
{
    if (!paw_rt_cache) paw_rt_cache = std::make_shared<paw::runtime_caches>();
    return *paw_rt_cache;
}

inline paw::aainit_tables const& pseudopot::paw_aatab() const
{
    auto& rt = paw_rt();
    if (rt.aatab_lli == 0) {
        // Same formula as build_paw_scf_caches' paw_aainit_lli and v_x's
        // lli_aat: 1 + max l over all projectors, min 1.
        int lli = 1;
        for (auto const& sp : paw_species)
            for (long b = 0; b < (long)sp.lll.size(); ++b)
                lli = std::max(lli, (int)sp.lll(b) + 1);
        rt.aatab = paw::aainit_tables_build(lli);
        rt.aatab_lli = lli;
    }
    return rt.aatab;
}

inline std::vector<paw::qrad_tab> const& pseudopot::paw_qrad_tabs(
    double Kmax, bool shape_restored_paw) const
{
    auto& rt = paw_rt();
    constexpr double dq = 0.01;   // plan A4: one dq for ALL consumers
    if (rt.qtab_Kmax < Kmax ||
        rt.qtab_shape_restored != shape_restored_paw ||
        rt.qtab_aug_lmax != aug_lmax()) {
        rt.qtab_sp.assign(paw_species.size(), paw::qrad_tab{});
        for (size_t nt = 0; nt < paw_species.size(); ++nt)
            if (paw_species[nt].is_paw || paw_species[nt].is_uspp)
                rt.qtab_sp[nt] =
                    (shape_restored_paw && paw_species[nt].is_paw)
                        ? paw::build_qrad_tab_full_aeps(paw_species[nt], Kmax,
                                                        dq, aug_lmax())
                        : paw::build_qrad_tab(paw_species[nt], Kmax,
                                              dq, aug_lmax());
        rt.qtab_Kmax = Kmax;
        rt.qtab_shape_restored = shape_restored_paw;
        rt.qtab_aug_lmax = aug_lmax();
    }
    return rt.qtab_sp;
}

inline math::shm::shared_array<nda::array_view<ComplexType,4>> const&
pseudopot::Pskna_full_bz() const
{
    auto& rt = paw_rt();
    if (!rt.Pskna_full) {
        rt.Pskna_full = std::make_shared<
            math::shm::shared_array<nda::array_view<ComplexType,4>>>(
            paw::compute_Pskna_full_bz(*this, kp_to_ibz, kp_symm, kp_trev,
                                       kpts, lattv, recv, symm_list,
                                       npol, *mpi));
    }
    return *rt.Pskna_full;
}

} // namespace hamilt

#endif // HAMILTONIAN_PAW_RUNTIME_CACHES_HPP
