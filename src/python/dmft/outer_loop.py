"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

"""
Option-2 outer-loop diagnostics (Project 2 increment Q5,
``notes/q5_option2_outer_loop_spec.md`` §1 piece 2, R-Q5-2 and gate Q5-b).

Three things live here, and nothing else:

  * the **DC staleness meters** ``||Sigma_dc^(n) - Sigma_dc^(n-1)||`` and
    ``||P_dc^(n) - P_dc^(n-1)||`` that Q5 adds to the cycle order
    ``gloc/wloc -> solve_gw_dc -> impurity`` (which already re-evaluates both
    DCs after every lattice/W change -- the meter reports how much that
    re-evaluation actually moved);
  * the **R-Q5-2 subspace-tracking diagnostic** ``o_C``: the projectors
    ``P_B``/``P_C`` are FIXED run inputs, so character continuation is automatic
    and "maximal-overlap continuation" reduces to *watching* the C-window
    MO character per outer cycle. Full per-cycle re-wannierization is out of
    scope (revisit only if ``o_C`` degrades in production);
  * the **Q5-b Mott-feedback-chain trail**: one consolidated per-cycle log block
    and a fixed-layout float array for the checkpoint.

This module is deliberately **numpy-only** at import time, following
``retardation.py``: the Q5-b/Q5-g3 unit checks must run on a host where neither
``triqs`` nor the compiled ``coqui`` package is importable. Only ``app_log`` is
imported from CoQuí, and that import is guarded.
"""
import numpy as np

try:  # pragma: no cover - the fallback only fires on a CoQuí-less host
    from coqui import app_log
except ImportError:  # pragma: no cover
    def app_log(level, msg, *args, **kwargs):
        """Minimal stand-in so the numpy-only unit checks can run standalone."""
        pass


Hartree_eV = 27.211386245988

#: Sentinel for "this cycle did not measure that field". It is a FINITE float on
#: purpose -- the Q5-b gate requires every trail entry to be finite, and this is the
#: same "never populated" convention the C++ readouts use (``scr_coulomb_t.h:317``
#: initialises the lambda meters to ``-1.0``).
MISSING = -1.0

#: Fixed layout of the Q5-b checkpoint trail (spec §3, gate Q5-b). Order is part of
#: the on-disk format: append only, never reorder.
MOTT_CHAIN_TRAIL_LABELS = (
    "gap_eV",               # gap(H_eff) of the re-derived quasiparticle Hamiltonian
    "epsilon_inf",          # eps_infinity of the lattice screening
    "lambda_nu0",           # lambda_max(nu = 0), the eq-6 ladder watchdog
    "pi_imp_minus_dc",      # ||P_imp - P_dc||
    "sigma_imp_minus_dc",   # ||Sigma_imp - Sigma_dc||   (tau metric)
    "u_bar_0",              # Ubar(0)   -- impurity mode (a) only
    "z_b",                  # Z_B       -- impurity mode (a) only
    "dc_sigma_staleness",   # ||Sigma_dc^(n) - Sigma_dc^(n-1)||
    "dc_pi_staleness",      # ||P_dc^(n)     - P_dc^(n-1)||
    "band_reorder_count",   # qp-loop band-reordering events this cycle
    "o_c",                  # C-window MO-character overlap (R-Q5-2)
)


# --------------------------------------------------------------------------
# Distances and staleness meters
# --------------------------------------------------------------------------

def field_distance(a, b, transform=None):
    """
    ``max |a - b|``, optionally after mapping the difference to another axis.

    ``transform`` is the tau-metric hook: pass ``lambda d: iaft.w_to_tau(d, stats='f')``
    (or ``w_to_tau_phsym`` for a bosonic field) to reproduce the convergence metric of
    ``dmft_state.py:267-289``, which measures Matsubara differences on the imaginary-time
    axis. ``None`` measures on the axis the arrays are already on.

    Returns ``MISSING`` when either operand is ``None`` (nothing to compare yet).
    """
    if a is None or b is None:
        return MISSING
    d = np.asarray(a) - np.asarray(b)
    if d.size == 0:
        return 0.0
    if transform is not None:
        d = np.asarray(transform(d))
    return float(np.max(np.abs(d)))


def dc_staleness(curr, prev, transform=None):
    """
    DC re-evaluation meter of spec §1: how far the double counting moved between two
    consecutive outer cycles.

    ``curr``/``prev`` are the ``"dc"`` entries of ``dmft_state.local_sigma_w`` /
    ``local_pi_w`` (or any pair of like-shaped arrays). The first cycle has no
    predecessor and reports ``MISSING``.
    """
    return field_distance(curr, prev, transform=transform)


def imp_minus_dc(local_field, transform=None):
    """
    ``||X_imp - X_dc||`` for an embedded local field, i.e. the lattice correction the
    outer cycle is actually feeding back. Takes the ``{"imp": ..., "dc": ...}`` dict
    that ``weiss.embed_impurities`` returns.
    """
    if local_field is None:
        return MISSING
    try:
        return field_distance(local_field["imp"], local_field["dc"], transform=transform)
    except (KeyError, TypeError, IndexError):
        return MISSING


# --------------------------------------------------------------------------
# R-Q5-2: subspace tracking as a diagnostic
# --------------------------------------------------------------------------

def band_window_slice(band_window, imp=0, spin=0):
    """
    0-based half-open slice of the correlated band window.

    ``band_window`` follows the CoQuí/Wannier90 convention: shape ``(nImp, nspin, 2)``,
    **1-based inclusive** bounds (``projector_t.h:87-88`` converts it with
    ``range(bw(I,0,0) - 1, bw(I,0,1))``). A ``(nImp, 2)`` array is accepted too.
    """
    bw = np.asarray(band_window)
    if bw.ndim == 3:
        lo, hi = int(bw[imp, spin, 0]), int(bw[imp, spin, 1])
    elif bw.ndim == 2:
        lo, hi = int(bw[imp, 0]), int(bw[imp, 1])
    elif bw.ndim == 1:
        lo, hi = int(bw[0]), int(bw[1])
    else:
        raise ValueError(f"band_window_slice: cannot interpret shape {bw.shape}")
    return slice(lo - 1, hi)


def project_mo_on_c(mo_skia, proj_mat, band_window, states=None, imp=0):
    """
    C-window character of each MO: ``(P_C |MO_a>)`` for every k, spin and MO index ``a``.

    Shapes (the ``projector_info`` convention built by ``weiss.get_proj_info``, which
    the C++ side receives as ``projector_ksIai``):

    ==============  ==========================================
    ``mo_skia``     ``(nspin, nkpts, nbnd, nbnd)``  -- ``[s, k, i, a]``, columns are MOs
    ``proj_mat``    ``(nkpts, nspin, nImp, nC, nwin)`` -- ``[k, s, I, a_C, i_win]``
    ``band_window`` ``(nImp, nspin, 2)``, 1-based inclusive
    ==============  ==========================================

    ``states`` restricts the MO index to the judge set (default: all MOs). Returns an
    array ``(nspin, nkpts, nC, nstates)``.
    """
    mo = np.asarray(mo_skia)
    P = np.asarray(proj_mat)
    if mo.ndim != 4:
        raise ValueError(f"project_mo_on_c: mo_skia must be 4-d (s,k,i,a), got {mo.shape}")
    if P.ndim != 5:
        raise ValueError(f"project_mo_on_c: proj_mat must be 5-d (k,s,I,a,i), got {P.shape}")
    win = band_window_slice(band_window, imp=imp)
    mo_win = mo[:, :, win, :]                        # (s, k, i_win, a)
    if states is not None:
        mo_win = mo_win[:, :, :, np.asarray(states)]
    Pc = P[:, :, imp]                                # (k, s, a_C, i_win)
    if Pc.shape[-1] != mo_win.shape[2]:
        raise ValueError(
            f"project_mo_on_c: projector window {Pc.shape[-1]} != band_window width "
            f"{mo_win.shape[2]} (band_window={np.asarray(band_window).tolist()})")
    # [k,s,C,i] x [s,k,i,a] -> [s,k,C,a]
    return np.einsum('ksCi,skia->skCa', Pc, mo_win)


def c_window_overlap(proj_new, proj_old, eps=1e-30):
    """
    R-Q5-2 diagnostic ``o_C``: the WORST C-character retention over the judge states,

    ``o_C = min_{s,k,a}  || P_C |MO_new(s,k,a)> ||  /  || P_C |MO_old(s,k,a)> ||``

    computed only over states that carried C character to begin with (states whose
    denominator is below ``eps`` are skipped -- they say nothing about continuation).

    ``o_C ~ 1`` means the correlated subspace kept its character across the outer cycle,
    which under a FIXED projector is the expected behaviour; a drop is the signal that
    would motivate re-wannierization (out of scope per R-Q5-2). Returns ``MISSING`` when
    there is no predecessor or no state carries C character.
    """
    if proj_new is None or proj_old is None:
        return MISSING
    a = np.asarray(proj_new)
    b = np.asarray(proj_old)
    if a.shape != b.shape:
        raise ValueError(f"c_window_overlap: shape mismatch {a.shape} vs {b.shape}")
    # contract the C index: norm per (spin, k, MO)
    n_new = np.sqrt(np.sum(np.abs(a) ** 2, axis=-2))
    n_old = np.sqrt(np.sum(np.abs(b) ** 2, axis=-2))
    live = n_old > eps
    if not np.any(live):
        return MISSING
    return float(np.min(n_new[live] / n_old[live]))


# --------------------------------------------------------------------------
# Gate Q5-b: the Mott feedback chain, one block per outer cycle
# --------------------------------------------------------------------------

def mott_chain_trail(**fields):
    """
    Assemble the Q5-b trail as a fixed-layout ``float`` array ordered by
    :data:`MOTT_CHAIN_TRAIL_LABELS`. Unknown keys are rejected (a typo must not
    silently vanish into a ``MISSING`` slot); absent keys and ``None`` become
    :data:`MISSING`, which is finite by construction.
    """
    unknown = set(fields) - set(MOTT_CHAIN_TRAIL_LABELS)
    if unknown:
        raise ValueError(
            f"mott_chain_trail: unknown field(s) {sorted(unknown)}; "
            f"valid fields are {list(MOTT_CHAIN_TRAIL_LABELS)}")
    out = np.full(len(MOTT_CHAIN_TRAIL_LABELS), MISSING, dtype=float)
    for i, name in enumerate(MOTT_CHAIN_TRAIL_LABELS):
        v = fields.get(name)
        if v is None:
            continue
        v = float(np.real(v))
        out[i] = v if np.isfinite(v) else MISSING
    return out


def log_mott_chain(cycle, niter, trail, verbose=True):
    """
    ONE consolidated log block per outer cycle (gate Q5-b). ``trail`` is the array from
    :func:`mott_chain_trail`. The PHYSICS of the chain -- gap opens => Drude weight lost
    => eps_inf falls => Ubar grows => gap opens further (PDF §5.3) -- is read off this
    block across cycles; it is a production observable, not a local gate.
    """
    if not verbose:
        return
    v = dict(zip(MOTT_CHAIN_TRAIL_LABELS, np.asarray(trail, dtype=float)))

    def _f(name, unit="", fmt="{:.6f}"):
        x = v[name]
        return "not measured" if x == MISSING else (fmt.format(x) + unit)

    app_log(1, f"\n[GW+EDMFT cycle {cycle}/{niter}] Mott feedback chain (Q5-b)")
    app_log(1,   "-------------------------------------------------------")
    app_log(1, f"  gap(H_eff)                     = {_f('gap_eV', ' eV')}")
    app_log(1, f"  epsilon_inf                    = {_f('epsilon_inf')}")
    app_log(1, f"  lambda_max(nu = 0)             = {_f('lambda_nu0')}")
    app_log(1, f"  ||P_imp - P_dc||               = {_f('pi_imp_minus_dc', '', '{:.6e}')}")
    app_log(1, f"  ||Sigma_imp - Sigma_dc|| (tau) = {_f('sigma_imp_minus_dc', '', '{:.6e}')}")
    app_log(1, f"  Ubar(0)                        = {_f('u_bar_0')}")
    app_log(1, f"  Z_B (impurity mode (a))        = {_f('z_b')}")
    app_log(1, f"  DC staleness, Sigma_dc         = {_f('dc_sigma_staleness', '', '{:.6e}')}")
    app_log(1, f"  DC staleness, P_dc             = {_f('dc_pi_staleness', '', '{:.6e}')}")
    app_log(1, f"  band-reorder events            = {_f('band_reorder_count', '', '{:.0f}')}")
    app_log(1, f"  o_C (C-window MO character)    = {_f('o_c')}\n")


# --------------------------------------------------------------------------
# Reading the qpGW lattice stage back out of the CoQuí checkpoint
# --------------------------------------------------------------------------

def heff_gap_eV(e_ska, nelec, spin=0):
    """
    Indirect gap of the re-derived quasiparticle Hamiltonian, in eV, from the
    checkpointed ``E_ska`` of the qpGW stage: ``min_k E_lumo - max_k E_homo`` with
    ``homo = nelec/2 - 1`` (the qpGW suites' convention,
    ``test_methods_qpgw_bse.cpp:150-158``).

    Returns ``MISSING`` for a metal-shaped input (the band indices do not exist) rather
    than raising -- this is a diagnostic, and it must not take a production run down.
    """
    if e_ska is None:
        return MISSING
    E = np.asarray(e_ska)
    homo = int(nelec // 2) - 1
    lumo = int(nelec // 2)
    if E.ndim != 3 or homo < 0 or lumo >= E.shape[2]:
        return MISSING
    e_homo = float(np.max(np.real(E[spin, :, homo])))
    e_lumo = float(np.min(np.real(E[spin, :, lumo])))
    return (e_lumo - e_homo) * Hartree_eV


def count_band_reorderings(mo_new, mo_old, ovlp):
    """
    R-Q5-2 diagnostic (i): how many quasiparticle states swapped index between two outer
    cycles -- maximal-overlap continuation used as a METER, not as a re-ordering.

    ``update_MOs`` (``qp_scf_common.cpp:876-893``) returns eigenvalues in ascending order,
    so a level crossing does NOT show up in the energies: it shows up as the character of
    slot ``a`` moving to slot ``b``. With the generalised eigenvectors satisfying
    ``C^dag S C = 1``, the continuation matrix is

        ``M[b, a] = <MO_new(b)| S |MO_old(a)>``

    and slot ``a`` is preserved iff ``argmax_b |M[b, a]| == a``. Counts the slots where it
    is not, summed over spin and k.

    ``mo_new``/``mo_old``: ``(nspin, nkpts, nbnd, nbnd)`` = ``[s, k, i, a]``;
    ``ovlp``: ``system/S_skij`` of the CoQuí checkpoint, same leading shape.
    Returns ``MISSING`` when there is no predecessor or the shapes disagree.
    """
    if mo_new is None or mo_old is None or ovlp is None:
        return MISSING
    A = np.asarray(mo_new)
    B = np.asarray(mo_old)
    S = np.asarray(ovlp)
    if A.shape != B.shape or S.shape != A.shape or A.ndim != 4:
        return MISSING
    M = np.einsum('skib,skij,skja->skba', np.conjugate(A), S, B, optimize=True)
    best = np.argmax(np.abs(M), axis=-2)                  # [s, k, a] -> b
    idx = np.arange(A.shape[3])[None, None, :]
    return float(np.count_nonzero(best != idx))
