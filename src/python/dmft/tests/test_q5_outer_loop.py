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
Q5 (the Option-2 outer loop) python gates -- notes/q5_option2_outer_loop_spec.md §3.

  * **Q5-g3 (python wiring):** ``outer_loop="option1"`` must leave the Q4 frozen-stage
    wiring alone (param-level assertion, the P1 pattern), and ``outer_loop="option2"``
    must emit the per-cycle qpGW stage with the R-Q5-1 damping and the external-G source.
  * **Q5-b (Mott feedback chain):** every field of the per-cycle trail is present and
    finite, the layout is fixed, and the diagnostics that feed it (DC staleness, o_C,
    band reordering, gap(H_eff)) behave.

Two tiers, deliberately separated (the ``test_q4_edmft_skeleton.py`` pattern):

  * **numpy-only tier** -- imports ``dmft/outer_loop.py`` by file path, so it runs on a
    host where neither ``coqui`` nor ``triqs`` is importable. This is where the whole
    Q5-b field-presence/finiteness gate lives.
  * **coqui tier** -- ``convert_gw_edmft_params`` and ``scf_driver`` need ``coqui`` +
    ``triqs``; guarded with ``importorskip``.

RUN COMMANDS
------------
On a TRIQS host (rusty), from the build/install tree:

    # everything in this file
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q5_outer_loop.py

    # the Q4 suites must stay green alongside it
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q4_edmft_skeleton.py \\
                           src/python/dmft/tests/test_edmft.py

Without pytest, the numpy-only tier runs standalone (this is how it was measured):

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 src/python/dmft/tests/test_q5_outer_loop.py

MEASURED 2026-08-14 on the implementation host (numpy 2.x, no coqui/triqs): **16/16**
numpy-only legs pass; the 5 Q5-g3 parameter/stage legs take a pytest fixture
(``tmp_path``/``monkeypatch``) and need ``coqui`` + ``triqs``, so they are skipped there
and must be run on a TRIQS host with the pytest command above.

ENVIRONMENT-BLOCKED LEG (spec §5, recorded -- NOT gated here)
------------------------------------------------------------
The full C = empty-set Option-2 end-to-end run needs a TRIQS host with a QMC impurity
solver. On rusty, with an existing ``svo.mbpt.h5`` GW checkpoint::

    params = {
        "niter": 8,
        "lattice_solver": "qpgw",
        "outer_loop": "option2",       # <-- the Q5 switch
        "outer_qpgw_niter": 1,         # the pure one-shot re-QP step
        "prefix": "svo", "outdir": "./",
        "screen_type": "gw_edmft",
        "iter_alg": {"alg": "damping", "mixing": 0.3},   # PDF §7, ruling R-Q5-1
        "wannier_file": ".../svo.mlwf.h5",
        "qpgw": {"qp_map": "mode_a", "pol_vertex": "ladder",
                 "pol_vertex_inject": "ladder_n2", "pol_vertex_band_window": [...]},
        "edmft": {"impurity": {...CT-SEG, "retardation": "static_u_zb"...}},
    }
    run_gw_edmft(h_int, embedding, params=params)

The C = empty-set check is the same run with the impurity corrections switched off: its
H_eff trail must reproduce the ``outer_loop="option1"`` trail, which is the python-level
statement of the C++ gate Q5-g2 (``test_methods_qpgw_q5.cpp``).
"""

import importlib.util
import pathlib

import numpy as np


# --------------------------------------------------------------------------
# Load dmft/outer_loop.py standalone (no coqui / triqs import).
# --------------------------------------------------------------------------

def _load_outer_loop():
    path = pathlib.Path(__file__).resolve().parents[1] / "outer_loop.py"
    spec = importlib.util.spec_from_file_location("_q5_outer_loop_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ol = _load_outer_loop()


# ==========================================================================
# Gate Q5-b -- the Mott feedback chain trail (numpy only)
# ==========================================================================

def _full_trail_fields():
    """One physically-shaped value for every Q5-b field of spec §3."""
    return {
        "gap_eV": 1.85,
        "epsilon_inf": 4.21,
        "lambda_nu0": 0.1147,
        "pi_imp_minus_dc": 3.4e-3,
        "sigma_imp_minus_dc": 7.1e-4,
        "u_bar_0": 0.1532,
        "z_b": 0.83,
        "dc_sigma_staleness": 2.2e-5,
        "dc_pi_staleness": 9.8e-6,
        "band_reorder_count": 0.0,
        "o_c": 0.9997,
        # Q6 §1.1 appended the cancellation-load columns to this same trail
        "r_nu0": 4.2e-2,
        "r_mid": 7.8e-2,
        "r_top": 1.3e-1,
        "lad_over_dc": 0.64,
    }


def test_mott_chain_trail_layout_is_the_q5b_field_list():
    """
    The trail layout IS the gate's field list -- spec §3, gate Q5-b.

    The Q5-b prefix is pinned FIRST and separately: the layout is documented "append only,
    never reorder" (outer_loop.py:68-69), so a later increment adding columns must leave the
    Q5-b slots exactly where they were, and this asserts that rather than just the total.
    Increment Q6 §1.1 appended the four cancellation-load columns.
    """
    q5b = (
        "gap_eV", "epsilon_inf", "lambda_nu0",
        "pi_imp_minus_dc", "sigma_imp_minus_dc",
        "u_bar_0", "z_b",
        "dc_sigma_staleness", "dc_pi_staleness",
        "band_reorder_count", "o_c",
    )
    assert ol.MOTT_CHAIN_TRAIL_LABELS[:len(q5b)] == q5b
    assert ol.MOTT_CHAIN_TRAIL_LABELS == q5b + ("r_nu0", "r_mid", "r_top", "lad_over_dc")
    assert len(set(ol.MOTT_CHAIN_TRAIL_LABELS)) == len(ol.MOTT_CHAIN_TRAIL_LABELS)


def test_mott_chain_trail_fields_present_and_finite():
    """Gate Q5-b proper: every field present, in order, and FINITE."""
    fields = _full_trail_fields()
    trail = ol.mott_chain_trail(**fields)

    assert trail.shape == (len(ol.MOTT_CHAIN_TRAIL_LABELS),)
    assert np.all(np.isfinite(trail)), f"non-finite entries: {trail}"
    for i, name in enumerate(ol.MOTT_CHAIN_TRAIL_LABELS):
        assert trail[i] == fields[name], f"{name}: {trail[i]} != {fields[name]}"
    print(f"    [Q5-b] trail = {dict(zip(ol.MOTT_CHAIN_TRAIL_LABELS, trail))}")


def test_mott_chain_trail_missing_fields_stay_finite():
    """A field the cycle could not source becomes the FINITE MISSING sentinel."""
    trail = ol.mott_chain_trail(gap_eV=1.0)
    assert np.all(np.isfinite(trail))
    assert trail[0] == 1.0
    assert np.all(trail[1:] == ol.MISSING)
    assert np.isfinite(ol.MISSING)
    # None and a non-finite input land on the same sentinel
    assert ol.mott_chain_trail(gap_eV=None)[0] == ol.MISSING
    assert ol.mott_chain_trail(gap_eV=np.nan)[0] == ol.MISSING
    assert ol.mott_chain_trail(gap_eV=np.inf)[0] == ol.MISSING


def test_mott_chain_trail_rejects_unknown_fields():
    """A typo must not vanish into a MISSING slot."""
    try:
        ol.mott_chain_trail(gap_ev=1.0)          # lower-case v
    except ValueError as e:
        assert "gap_ev" in str(e)
    else:
        raise AssertionError("an unknown trail field must raise ValueError")


def test_log_mott_chain_runs_on_a_partial_trail():
    """The consolidated block must survive a cycle that measured nothing."""
    ol.log_mott_chain(1, 8, ol.mott_chain_trail(), verbose=True)
    ol.log_mott_chain(2, 8, ol.mott_chain_trail(**_full_trail_fields()), verbose=True)


# ==========================================================================
# DC staleness meters (numpy only)
# ==========================================================================

def test_dc_staleness_is_missing_on_the_first_cycle():
    """No predecessor => nothing to compare; the meter must stay FINITE."""
    v = ol.dc_staleness(np.ones((3, 2)), None)
    assert v == ol.MISSING and np.isfinite(v)


def test_dc_staleness_measures_the_max_abs_move():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    b = a.copy()
    b[1, 0] += 0.25
    b[0, 1] -= 0.10
    d = ol.dc_staleness(a, b)
    print(f"    [staleness] max|dSigma_dc| = {d:.6f}")
    assert d == 0.25
    assert ol.dc_staleness(a, a) == 0.0


def test_dc_staleness_honours_the_tau_transform():
    """The tau-metric hook is the dmft_state.py:267-289 pattern: transform the DIFFERENCE."""
    a = np.arange(12.0).reshape(6, 2)
    b = a + 1.0
    seen = {}

    def transform(d):
        seen['shape'] = d.shape
        return 2.0 * d           # a stand-in for iaft.w_to_tau

    d = ol.dc_staleness(a, b, transform=transform)
    assert seen['shape'] == a.shape          # the transform saw the difference, not a norm
    assert d == 2.0


def test_imp_minus_dc_vanishes_in_the_clean_limit():
    """Q4-b's clean limit read through the Q5-b meter: imp == dc => exactly 0."""
    x = np.arange(24.0).reshape(4, 3, 2) + 1j
    assert ol.imp_minus_dc({"imp": x, "dc": x.copy()}) == 0.0
    assert ol.imp_minus_dc({"imp": x, "dc": x - 0.5}) == 0.5
    assert ol.imp_minus_dc(None) == ol.MISSING
    assert ol.imp_minus_dc({"imp": x}) == ol.MISSING


# ==========================================================================
# R-Q5-2 -- subspace tracking as a diagnostic (numpy only)
# ==========================================================================

def _mo_fixture(ns=1, nk=3, nbnd=4, seed=5):
    """Orthonormal MO columns (S = 1) so the trackers have a well-posed metric."""
    rng = np.random.default_rng(seed)
    mo = np.zeros((ns, nk, nbnd, nbnd), dtype=complex)
    for s in range(ns):
        for k in range(nk):
            a = rng.normal(size=(nbnd, nbnd)) + 1j * rng.normal(size=(nbnd, nbnd))
            q, _r = np.linalg.qr(a)
            mo[s, k] = q
    ovlp = np.zeros_like(mo)
    ovlp[:, :] = np.eye(nbnd)
    return mo, ovlp


def test_band_window_slice_is_the_coqui_one_based_convention():
    """projector_t.h:87-88: range(bw(I,0,0) - 1, bw(I,0,1)) -- 1-based, inclusive."""
    assert ol.band_window_slice(np.array([[[2, 5]]])) == slice(1, 5)
    assert ol.band_window_slice(np.array([[2, 5]])) == slice(1, 5)
    assert ol.band_window_slice(np.array([2, 5])) == slice(1, 5)


def test_band_reorder_count_is_zero_for_an_unchanged_mo_set():
    mo, ovlp = _mo_fixture()
    n = ol.count_band_reorderings(mo, mo.copy(), ovlp)
    print(f"    [reorder] identical MO sets => {n:.0f} events")
    assert n == 0.0
    assert ol.count_band_reorderings(mo, None, ovlp) == ol.MISSING


def test_band_reorder_count_catches_a_swapped_pair():
    """Two states exchanging slots at ONE k = 2 events (both slots moved)."""
    mo, ovlp = _mo_fixture()
    swapped = mo.copy()
    swapped[0, 1][:, [1, 2]] = swapped[0, 1][:, [2, 1]]
    n = ol.count_band_reorderings(swapped, mo, ovlp)
    print(f"    [reorder] one swapped pair at one k => {n:.0f} events")
    assert n == 2.0


def test_c_window_overlap_is_one_for_a_fixed_projector_and_unchanged_mos():
    """
    R-Q5-2: under a FIXED projector, an unchanged MO set retains its C character
    exactly -- o_C == 1 at machine precision. That is the baseline the production
    meter is read against.
    """
    mo, _ovlp = _mo_fixture(nbnd=4)
    rng = np.random.default_rng(17)
    nk, nC, nwin = mo.shape[1], 2, 3
    proj = (rng.normal(size=(nk, 1, 1, nC, nwin))
            + 1j * rng.normal(size=(nk, 1, 1, nC, nwin)))
    band_window = np.array([[[2, 4]]])            # 1-based inclusive -> slice(1, 4)

    p_new = ol.project_mo_on_c(mo, proj, band_window)
    assert p_new.shape == (1, nk, nC, mo.shape[3])
    o_c = ol.c_window_overlap(p_new, p_new.copy())
    print(f"    [o_C] unchanged MO set => o_C = {o_c!r}")
    assert o_c == 1.0
    assert ol.c_window_overlap(p_new, None) == ol.MISSING


def test_c_window_overlap_reports_the_worst_character_loss():
    """o_C is a MIN over judge states: one degraded state sets the number."""
    mo, _ovlp = _mo_fixture(nbnd=4)
    rng = np.random.default_rng(23)
    nk, nC, nwin = mo.shape[1], 2, 3
    proj = (rng.normal(size=(nk, 1, 1, nC, nwin))
            + 1j * rng.normal(size=(nk, 1, 1, nC, nwin)))
    band_window = np.array([[[2, 4]]])

    p_old = ol.project_mo_on_c(mo, proj, band_window)
    p_new = p_old.copy()
    p_new[0, 1, :, 2] *= 0.4                       # one state loses 60 % of its C weight
    o_c = ol.c_window_overlap(p_new, p_old)
    print(f"    [o_C] one state scaled by 0.4 => o_C = {o_c:.10f}")
    assert abs(o_c - 0.4) < 1e-12


def test_project_mo_on_c_rejects_a_window_mismatch():
    mo, _ovlp = _mo_fixture(nbnd=4)
    proj = np.zeros((mo.shape[1], 1, 1, 2, 3), dtype=complex)
    try:
        ol.project_mo_on_c(mo, proj, np.array([[[1, 4]]]))   # width 4 != 3
    except ValueError as e:
        assert "band_window" in str(e)
    else:
        raise AssertionError("a projector/window width mismatch must raise ValueError")


# ==========================================================================
# gap(H_eff) from the checkpointed qpGW spectrum (numpy only)
# ==========================================================================

def test_heff_gap_matches_the_qpgw_suite_convention():
    """
    ``min_k E_lumo - max_k E_homo`` with homo = nelec/2 - 1 -- the convention of
    test_methods_qpgw_bse.cpp:150-158, so the python trail and the C++ gates quote the
    same number.
    """
    e = np.zeros((1, 3, 4), dtype=complex)
    e[0, :, 0] = [-1.0, -0.9, -1.1]
    e[0, :, 1] = [-0.30, -0.25, -0.40]      # homo (nelec = 4)
    e[0, :, 2] = [0.50, 0.42, 0.61]         # lumo
    e[0, :, 3] = [1.0, 1.1, 1.2]

    gap = ol.heff_gap_eV(e, nelec=4)
    expect = (0.42 - (-0.25)) * ol.Hartree_eV
    print(f"    [gap] {gap:.9f} eV (expected {expect:.9f} eV)")
    assert abs(gap - expect) < 1e-12
    assert ol.heff_gap_eV(None, 4) == ol.MISSING
    assert ol.heff_gap_eV(e, nelec=8) == ol.MISSING     # lumo index out of range


# ==========================================================================
# Gate Q5-g3 -- the parameter wiring (needs coqui + triqs)
# ==========================================================================

def _base_params(tmpdir, **extra):
    p = {
        'niter': 4,
        'prefix': 'q5probe',
        'outdir': str(tmpdir),
        'lattice_solver': 'qpgw',
        'screen_type': 'gw_edmft',
        'edmft': {'impurity': {'n_cycles': 10, 'n_warmup_cycles': 1, 'length_cycle': 1}},
    }
    p.update(extra)
    return p


def _touch_checkpoint(tmpdir):
    """convert_gw_edmft_params requires the GW checkpoint to EXIST (io.py:90-94)."""
    path = pathlib.Path(tmpdir) / "q5probe.mbpt.h5"
    path.touch()
    return path


def test_option1_wiring_is_unchanged_by_the_q5_switch(tmp_path):
    """
    Q5-g3, option1 leg: the Q4 frozen-stage parameter wiring must be untouched. Absent
    ``outer_loop`` and an explicit ``"option1"`` must agree on EVERY forwarded block, and
    the qpGW stage must keep its Q4 defaults (niter = 10 = run once to its own qp fixed
    point, restart = True).
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q5-g3 needs the coqui python package")
    pytest.importorskip("triqs", reason="coqui.dmft.io imports triqs")
    from coqui.dmft.io import convert_gw_edmft_params

    _touch_checkpoint(tmp_path)
    absent = convert_gw_edmft_params(_base_params(tmp_path))
    explicit = convert_gw_edmft_params(_base_params(tmp_path, outer_loop='option1'))

    assert absent['outer_loop'] == 'option1'
    assert absent['outer_qpgw_niter'] == 1
    for key in ('qpgw', 'wloc', 'gloc', 'dmft_embed', 'impurity',
                'niter', 'gw_iter_per_loop', 'edmft_iter_per_loop', 'lattice_solver'):
        assert absent[key] == explicit[key], f"option1 wiring moved for {key!r}"

    # the Q4 defaults themselves (q4_edmft_skeleton_spec.md P1)
    assert absent['qpgw']['niter'] == 10
    assert absent['qpgw']['restart'] is True
    assert absent['qpgw']['screen_type'] == 'gw_edmft'
    # ... and NO external-G injection: absent knob == the C++ inert default
    assert 'greens_func_source' not in absent['qpgw']
    print(f"    [Q5-g3 option1] qpgw block = {absent['qpgw']}")


def test_option2_emits_the_per_cycle_stage(tmp_path):
    """
    Q5-g3, option2 leg: the stage becomes per-cycle (``niter = outer_qpgw_niter``, default
    1 = the pure one-shot re-QP step) and carries the R-Q5-1 damping default of 0.3.
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q5-g3 needs the coqui python package")
    pytest.importorskip("triqs", reason="coqui.dmft.io imports triqs")
    from coqui.dmft.io import convert_gw_edmft_params

    _touch_checkpoint(tmp_path)
    out = convert_gw_edmft_params(_base_params(tmp_path, outer_loop='option2'))
    assert out['outer_loop'] == 'option2'
    assert out['outer_qpgw_niter'] == 1
    assert out['qpgw']['niter'] == 1
    assert out['qpgw']['restart'] is True
    assert out['qpgw']['iter_alg']['mixing'] == 0.3          # R-Q5-1 / PDF §7
    print(f"    [Q5-g3 option2] qpgw block = {out['qpgw']}")

    # outer_qpgw_niter propagates ...
    out3 = convert_gw_edmft_params(
        _base_params(tmp_path, outer_loop='option2', outer_qpgw_niter=3))
    assert out3['qpgw']['niter'] == 3
    # ... and the user's own 'qpgw' section still wins over everything derived
    out_u = convert_gw_edmft_params(_base_params(
        tmp_path, outer_loop='option2', outer_qpgw_niter=3,
        qpgw={'niter': 7, 'iter_alg': {'alg': 'damping', 'mixing': 0.1}}))
    assert out_u['qpgw']['niter'] == 7
    assert out_u['qpgw']['iter_alg']['mixing'] == 0.1


def test_option2_requires_the_qpgw_lattice_solver(tmp_path):
    import pytest
    pytest.importorskip("coqui", reason="gate Q5-g3 needs the coqui python package")
    pytest.importorskip("triqs", reason="coqui.dmft.io imports triqs")
    from coqui.dmft.io import convert_gw_edmft_params

    _touch_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="option2"):
        convert_gw_edmft_params(
            _base_params(tmp_path, lattice_solver='gw', outer_loop='option2'))
    with pytest.raises(ValueError, match="outer_loop"):
        convert_gw_edmft_params(_base_params(tmp_path, outer_loop='option3'))
    with pytest.raises(ValueError, match="outer_qpgw_niter"):
        convert_gw_edmft_params(
            _base_params(tmp_path, outer_loop='option2', outer_qpgw_niter=0))


def test_option2_greens_func_source_falls_back_on_the_first_cycle(tmp_path):
    """
    Q5-g3: the per-cycle stage injects the previous cycle's lattice G through the
    ``embed`` group; before one exists it must inject NOTHING (spec §1: "first cycle
    falls back to the frozen-stage behavior"), which is the C = empty-set limit the C++
    gates pin.
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q5-g3 needs the coqui python package")
    pytest.importorskip("triqs", reason="coqui.dmft.scf_driver imports triqs")
    pytest.importorskip("h5", reason="the source picker reads an h5 archive")
    from h5 import HDFArchive
    from coqui.dmft.scf_driver import qpgw_stage_greens_func_source

    path = str(pathlib.Path(tmp_path) / "cycle.mbpt.h5")
    with HDFArchive(path, 'w') as ar:
        ar.create_group("scf")
        ar["scf"]["final_iter"] = 4
    assert qpgw_stage_greens_func_source(path) == (None, -1)

    with HDFArchive(path, 'a') as ar:
        ar.create_group("embed")
        ar["embed"]["final_iter"] = 2
    assert qpgw_stage_greens_func_source(path) == ("embed", 2)

    # a missing file must not take the run down either
    assert qpgw_stage_greens_func_source(str(pathlib.Path(tmp_path) / "nope.h5")) == (None, -1)


class _StubMPI:
    def barrier(self):
        pass


class _StubHInt:
    def mpi(self):
        return _StubMPI()


class _StubState:
    local_pi_w = None


def test_qpgw_lattice_stage_forwards_the_external_g(tmp_path, monkeypatch):
    """
    Q5-g3, the stage itself (stub level): what ``_qpgw_lattice_stage`` actually hands to
    ``coqui.run_qpgw``.

      * option1 (no ``coqui_chkpt_h5``): NO ``greens_func_*`` key ever -- the C++ knob
        stays inert and the loop is bit-identical to Q4;
      * option2, first cycle (no ``embed`` group): still no key -- the documented
        frozen-stage fallback;
      * option2, later cycles: ``greens_func_source = "embed"`` at that group's
        ``final_iter``.
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q5-g3 needs the coqui python package")
    pytest.importorskip("triqs", reason="coqui.dmft.scf_driver imports triqs")
    pytest.importorskip("h5", reason="the stage reads an h5 archive")
    from h5 import HDFArchive
    from coqui.dmft import scf_driver as sd

    captured = {}

    def fake_run_qpgw(params, **kwargs):
        captured['params'] = dict(params)
        captured['kwargs'] = kwargs

    monkeypatch.setattr(sd.coqui, "run_qpgw", fake_run_qpgw)

    qpgw_params = {'outdir': str(tmp_path), 'prefix': 'cycle', 'restart': True,
                   'screen_type': 'gw_edmft', 'niter': 1,
                   'iter_alg': {'alg': 'damping', 'mixing': 0.3}}
    proj_info = {'proj_mat': None, 'band_window': None, 'kpts_w90': None}

    # --- option1: the frozen stage, no checkpoint argument ---------------------------
    sd._qpgw_lattice_stage(_StubHInt(), proj_info, _StubState(), qpgw_params)
    assert 'greens_func_source' not in captured['params']
    assert captured['kwargs']['local_polarizabilities'] is None

    path = str(pathlib.Path(tmp_path) / "cycle.mbpt.h5")
    with HDFArchive(path, 'w') as ar:
        ar.create_group("scf")
        ar["scf"]["final_iter"] = 4

    # --- option2, first cycle: embed group absent => fall back to no injection --------
    sd._qpgw_lattice_stage(_StubHInt(), proj_info, _StubState(), qpgw_params,
                           coqui_chkpt_h5=path, cycle=1, niter=3)
    assert 'greens_func_source' not in captured['params']
    assert captured['params']['restart'] is True
    assert captured['params']['niter'] == 1
    assert captured['params']['iter_alg']['mixing'] == 0.3

    # --- option2, later cycles: the previous cycle's upfolded lattice G ---------------
    with HDFArchive(path, 'a') as ar:
        ar.create_group("embed")
        ar["embed"]["final_iter"] = 2
    sd._qpgw_lattice_stage(_StubHInt(), proj_info, _StubState(), qpgw_params,
                           coqui_chkpt_h5=path, cycle=2, niter=3)
    assert captured['params']['greens_func_source'] == "embed"
    assert captured['params']['greens_func_iteration'] == 2
    print(f"    [Q5-g3 stage] cycle 2 forwarded {captured['params']}")

    # the caller's dict must not have been mutated by the option2 branch
    assert 'greens_func_source' not in qpgw_params


# ==========================================================================
# Standalone runner for the numpy-only tier (no pytest required)
# ==========================================================================

if __name__ == "__main__":
    import inspect
    import sys
    import traceback

    numpy_only = [name for name, obj in sorted(globals().items())
                  if name.startswith("test_") and callable(obj)
                  and not inspect.signature(obj).parameters]
    failures = 0
    for name in numpy_only:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(numpy_only) - failures}/{len(numpy_only)} numpy-only legs passed "
          f"(the Q5-g3 param legs take a tmp_path fixture and need coqui; run them "
          f"with pytest on a coqui host).")
    sys.exit(1 if failures else 0)
