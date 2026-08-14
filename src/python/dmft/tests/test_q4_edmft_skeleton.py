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
Q4 (EDMFT skeleton) python gates -- notes/q4_edmft_skeleton_spec.md §3 P1.

Two tiers, deliberately separated:

  * **numpy-only tier** (gate Q4-p2 + the causality monitor): imports
    ``dmft/retardation.py`` by file path, so it runs on a host where neither
    ``coqui`` nor ``triqs`` is importable.
  * **coqui tier** (gate Q4-b, the clean limit): needs ``coqui`` + ``h5``
    (``coqui.dmft.weiss`` imports ``h5``); it needs **no** QMC and no
    ``triqs_modest`` -- the "GW impurity" is ``solve_gw_dc`` itself and the
    embedding maps are deterministic stand-ins.

RUN COMMANDS
------------
On a TRIQS host (rusty), from the build/install tree:

    # everything in this file
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q4_edmft_skeleton.py

    # gate Q4-b only (clean limit)
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q4_edmft_skeleton.py -k clean_limit

    # gate Q4-p2 only (Z_B unit check); runs anywhere numpy is available
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q4_edmft_skeleton.py -k zb

    # the pre-existing EDMFT python suite must stay green alongside it
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_edmft.py

Without pytest, the numpy-only tier runs standalone:

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 src/python/dmft/tests/test_q4_edmft_skeleton.py
"""

import importlib.util
import pathlib

import numpy as np


# --------------------------------------------------------------------------
# Load dmft/retardation.py standalone (no coqui / triqs import).
# --------------------------------------------------------------------------

def _load_retardation():
    path = pathlib.Path(__file__).resolve().parents[1] / "retardation.py"
    spec = importlib.util.spec_from_file_location("_q4_retardation_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retardation = _load_retardation()


# --------------------------------------------------------------------------
# Synthetic bosonic-bath interactions (numpy only)
# --------------------------------------------------------------------------

def _uniform_bosonic_mesh(nu_max=200.0, nnu=4001):
    """Uniform non-negative bosonic mesh; nu[0] == 0 exactly."""
    return np.linspace(0.0, nu_max, nnu)


def _single_pole_u(nu, u_inf, lam, omega):
    """U(i nu) = U_inf - 2 lam^2 omega / (nu^2 + omega^2)  (one bosonic mode)."""
    return u_inf - 2.0 * lam ** 2 * omega / (nu ** 2 + omega ** 2)


def _product_basis_u(nu, v_abcd, retarded_ab, omega, sign=-1.0):
    """
    Build U_{abcd}(i nu) = V_{abcd} + sign * 2 * R_{ab} * omega/(nu^2+omega^2)
    on the density-density diagonal (a a b b) of the product basis.
    ``sign = -1`` is the bosonic-causal (screening) case.
    """
    norb = v_abcd.shape[0]
    U = np.zeros((nu.shape[0], norb, norb, norb, norb), dtype=complex)
    U[:] = v_abcd[None, ...]
    kern = 2.0 * omega / (nu ** 2 + omega ** 2)
    for a in range(norb):
        for b in range(norb):
            U[:, a, a, b, b] += sign * retarded_ab[a, b] * kern
    return U


# ==========================================================================
# Gate Q4-p2 -- Casula-Werner Z_B unit checks (numpy only)
# ==========================================================================

def test_zb_is_exactly_one_for_flat_u():
    """R-Q4-5 / Q4-p2: U(i nu) nu-independent  =>  Z_B == 1 at machine precision."""
    nu = _uniform_bosonic_mesh(50.0, 201)
    u_flat = np.full(nu.shape, 3.7)

    z_b = retardation.casula_werner_zb(u_flat, nu)
    s = retardation.casula_werner_exponent(u_flat, nu)

    assert s == 0.0, f"flat U must give S == 0 exactly, got {s!r}"
    assert z_b == 1.0, f"flat U must give Z_B == 1 exactly, got {z_b!r}"

    # ... and also in the multi-orbital product-basis form
    norb = 3
    v = np.zeros((norb, norb, norb, norb))
    for a in range(norb):
        for b in range(norb):
            v[a, a, b, b] = 2.0 + 0.1 * a - 0.05 * b
    u_pb = np.zeros((nu.shape[0],) + v.shape, dtype=complex)
    u_pb[:] = v[None, ...]
    assert retardation.casula_werner_zb(u_pb, nu) == 1.0


def test_zb_matches_the_analytic_single_pole_value():
    """S = lambda^2/omega^2 for one bosonic mode -- pins the formula, not just the limit."""
    lam, omega, u_inf = 0.3, 1.5, 4.0
    nu = _uniform_bosonic_mesh(400.0, 8001)
    u = _single_pole_u(nu, u_inf, lam, omega)

    s = retardation.casula_werner_exponent(u, nu)
    s_exact = lam ** 2 / omega ** 2

    rel = abs(s - s_exact) / s_exact
    print(f"    [zb single pole] S = {s:.12e}  S_exact = {s_exact:.12e}  rel = {rel:.3e}")
    assert rel < 1e-4, f"S = {s} vs exact {s_exact} (rel {rel})"

    z_b = retardation.casula_werner_zb(u, nu)
    assert abs(z_b - np.exp(-s_exact)) < 1e-5
    assert 0.0 < z_b <= 1.0


def test_zb_two_pole_superposition_is_additive_in_the_exponent():
    """S is linear in the bath spectral weight: S(mode1 + mode2) = S1 + S2."""
    nu = _uniform_bosonic_mesh(400.0, 8001)
    u1 = _single_pole_u(nu, 0.0, 0.25, 0.8)
    u2 = _single_pole_u(nu, 0.0, 0.40, 2.2)

    s1 = retardation.casula_werner_exponent(u1, nu)
    s2 = retardation.casula_werner_exponent(u2, nu)
    s12 = retardation.casula_werner_exponent(u1 + u2 + 5.0, nu)

    print(f"    [zb additivity] S1 = {s1:.10e}  S2 = {s2:.10e}  "
          f"S12 = {s12:.10e}  |S12-(S1+S2)| = {abs(s12 - s1 - s2):.3e}")
    assert abs(s12 - (s1 + s2)) < 1e-10 * max(1.0, s1 + s2)


def test_zb_is_bounded_for_causal_random_baths():
    """0 < Z_B <= 1 for any non-negative bath spectral weight (R-Q4-5 contract)."""
    rng = np.random.default_rng(20260814)
    nu = _uniform_bosonic_mesh(300.0, 3001)
    worst = 1.0
    for _ in range(25):
        u = np.full(nu.shape, 5.0)
        for _l in range(rng.integers(1, 5)):
            u += _single_pole_u(nu, 0.0, rng.uniform(0.05, 0.9), rng.uniform(0.2, 4.0))
        z_b = retardation.casula_werner_zb(u, nu)
        assert 0.0 < z_b <= 1.0, f"Z_B = {z_b} out of (0, 1]"
        worst = min(worst, z_b)
    print(f"    [zb bounds] smallest Z_B over 25 random causal baths = {worst:.8f}")


def test_zb_clamps_an_anticausal_u_to_one():
    """An anti-causal U (U(inu) < U(0)) is clamped, keeping 0 < Z_B <= 1."""
    nu = _uniform_bosonic_mesh(300.0, 3001)
    u = 5.0 - _single_pole_u(nu, 0.0, 0.5, 1.0)  # rises then falls: retarded part flipped
    z_b = retardation.casula_werner_zb(u, nu)
    print(f"    [zb anticausal] Z_B = {z_b:.8f}")
    assert z_b == 1.0
    assert retardation.casula_werner_exponent(u, nu, clamp=False) < 0.0


def test_retardation_mode_dynamic_is_a_pass_through():
    """Default mode must leave the solver inputs untouched (same objects)."""
    nu = _uniform_bosonic_mesh(100.0, 401)
    norb = 2
    v = np.zeros((norb, norb, norb, norb))
    v[0, 0, 0, 0] = v[1, 1, 1, 1] = 4.0
    u_weiss = _product_basis_u(nu, np.zeros_like(v), np.full((norb, norb), 0.2), 1.0)
    delta = np.ones((7, 2, norb, norb), dtype=complex)

    d_out, v_out, u_out, z_b = retardation.apply_impurity_retardation_mode(
        delta, u_weiss, v, nu, mode="dynamic")
    assert d_out is delta and v_out is v and u_out is u_weiss and z_b is None


def _mode_a_fixture(norb=2, nu_max=300.0, nnu=3001):
    nu = _uniform_bosonic_mesh(nu_max, nnu)
    v = np.zeros((norb, norb, norb, norb))
    for a in range(norb):
        for b in range(norb):
            v[a, a, b, b] = 4.0 if a == b else 2.5
    u_weiss = _product_basis_u(nu, np.zeros_like(v),
                               np.outer([1.0, 0.9], [1.0, 0.9]) * 0.25, 1.2)
    delta = (np.arange(7 * 2 * norb * norb, dtype=float)
             .reshape(7, 2, norb, norb).astype(complex))
    return nu, v, u_weiss, delta


def test_retardation_mode_static_u_zb_default_source_is_screened_u0():
    """
    Impurity mode (a), R-Q4-5 AMENDMENT default: static U = U(inu = 0) (screened,
    the Casula-Werner standard), Delta -> Z_B Delta.
    """
    nu, v, u_weiss, delta = _mode_a_fixture()
    u_weiss_ref, delta_ref, v_ref = u_weiss.copy(), delta.copy(), v.copy()

    d_out, v_out, u_out, z_b = retardation.apply_impurity_retardation_mode(
        delta, u_weiss, v, nu, mode="static_u_zb")     # default static_u_source="u0"

    u_bar = retardation.total_density_channel(
        retardation.combine_static_and_retarded_u(v, u_weiss))
    static_bar = retardation.total_density_channel(v_out[np.newaxis, ...])[0]
    print(f"    [mode (a) u0] Z_B = {z_b:.10f}  Ubar_static = {static_bar:.10f}  "
          f"Ubar(0) = {u_bar[0]:.10f}  Ubar(inu_max) = {u_bar[-1]:.10f}")

    assert 0.0 < z_b < 1.0
    # the U column IS the screened static value
    assert static_bar == u_bar[0]
    assert np.array_equal(v_out, v + u_weiss[0])
    # ... and it is strictly below the bare value (screening is present)
    assert static_bar < retardation.total_density_channel(v[np.newaxis, ...])[0]

    # Delta scaling and input immutability are unchanged
    assert np.array_equal(u_out, np.zeros_like(u_weiss))
    assert np.array_equal(d_out, delta_ref * z_b)
    assert np.array_equal(u_weiss, u_weiss_ref)
    assert np.array_equal(delta, delta_ref)
    assert np.array_equal(v, v_ref)


def test_retardation_mode_static_u_zb_u_inf_source_is_the_bare_interaction():
    """static_u_source="u_inf": the PDF section 3.3 literal, U(inu -> infty) = Vloc."""
    nu, v, u_weiss, delta = _mode_a_fixture()
    delta_ref = delta.copy()

    d_out, v_out, u_out, z_b = retardation.apply_impurity_retardation_mode(
        delta, u_weiss, v, nu, mode="static_u_zb", static_u_source="u_inf")

    u_bar = retardation.total_density_channel(
        retardation.combine_static_and_retarded_u(v, u_weiss))
    static_bar = retardation.total_density_channel(v_out[np.newaxis, ...])[0]
    bare_bar = retardation.total_density_channel(v[np.newaxis, ...])[0]
    residual = static_bar - u_bar[-1]
    print(f"    [mode (a) u_inf] Z_B = {z_b:.10f}  Ubar_static = {static_bar:.10f}  "
          f"Ubar_bare = {bare_bar:.10f}  Ubar(inu_max) = {u_bar[-1]:.10f}  "
          f"endpoint residual = {residual:.3e}")

    # the U column IS the bare/unscreened value
    assert v_out is v
    assert static_bar == bare_bar
    # ... which is the inu -> infty limit of U(inu): the last mesh node approaches it
    # from below, with the O(1/nu_max^2) residual of the bath kernel 2 lam^2 w/(nu^2+w^2).
    assert 0.0 < residual < 1e-4
    nu_far, v_far, u_far, _d = _mode_a_fixture(nu_max=3000.0, nnu=3001)
    u_bar_far = retardation.total_density_channel(
        retardation.combine_static_and_retarded_u(v_far, u_far))
    residual_far = retardation.total_density_channel(v_far[np.newaxis, ...])[0] - u_bar_far[-1]
    print(f"    [mode (a) u_inf] residual at 10x nu_max = {residual_far:.3e} "
          f"(ratio {residual / residual_far:.1f}, expected ~100)")
    assert residual_far < residual / 50.0

    # Delta scaling is identical to the "u0" leg: Z_B depends only on the retarded part
    _d0, _v0, _u0, z_b_u0 = retardation.apply_impurity_retardation_mode(
        delta, u_weiss, v, nu, mode="static_u_zb", static_u_source="u0")
    assert z_b == z_b_u0
    assert np.array_equal(d_out, delta_ref * z_b)
    assert np.array_equal(u_out, np.zeros_like(u_weiss))


def test_retardation_mode_rejects_unknown_static_u_source():
    nu, v, u_weiss, delta = _mode_a_fixture()
    try:
        retardation.apply_impurity_retardation_mode(
            delta, u_weiss, v, nu, mode="static_u_zb", static_u_source="u_half")
    except ValueError as e:
        assert "static_u_source" in str(e)
    else:
        raise AssertionError("unknown static_u_source must raise ValueError")


def test_retardation_mode_rejects_unknown_modes():
    nu = _uniform_bosonic_mesh(50.0, 201)
    v = np.zeros((1, 1, 1, 1))
    u_weiss = np.zeros((nu.shape[0], 1, 1, 1, 1))
    try:
        retardation.apply_impurity_retardation_mode(
            np.zeros((3, 2, 1, 1)), u_weiss, v, nu, mode="lang_firsov")
    except ValueError as e:
        assert "static_u_zb" in str(e)
    else:
        raise AssertionError("unknown retardation mode must raise ValueError")


# ==========================================================================
# Gate Q4-c -- causality monitor (numpy only)
# ==========================================================================

def _causal_u(nu, norb=3, omega=1.1, scale=0.3):
    """
    A bosonic-causal screened U: the retarded weight matrix R_ab must be positive
    semi-definite, otherwise U(0) - U(inu_max) acquires a positive eigenvalue --
    which is exactly what meter (iii) is built to catch.
    """
    v = np.zeros((norb, norb, norb, norb))
    for a in range(norb):
        for b in range(norb):
            v[a, a, b, b] = 4.0 if a == b else 2.0
    w = 1.0 + 0.1 * np.arange(norb)
    ret = scale * np.outer(w, w)          # rank-1, PSD
    return v, _product_basis_u(nu, v, ret, omega, sign=-1.0)


def test_causality_monitor_passes_a_causal_screened_u():
    nu = _uniform_bosonic_mesh(60.0, 241)
    _v, u = _causal_u(nu)
    m = retardation.u_causality_metrics(u, nu)

    print(f"    [causality causal] herm = {m['hermiticity_max']:.3e}  "
          f"flips = {m['dd_monotonicity_flips']}  "
          f"min_eig = {m['u0_minus_umax_min_eig']:.6e}  "
          f"max_eig = {m['u0_minus_umax_max_eig']:.6e}")
    assert m["hermiticity_max"] < 1e-14
    assert m["dd_monotonicity_flips"] == 0
    # the pair matrix has structurally zero rows -> max eig is 0 up to roundoff
    assert m["u0_minus_umax_max_eig"] < 1e-12
    assert m["u0_minus_umax_psd_violation"] == 0.0
    assert m["u0_minus_umax_min_eig"] < 0.0
    assert not m["causality_violated"]


def test_causality_monitor_flags_anticausal_screening():
    """U(0) - U(inu_max) positive semi-definite in some channel => flagged."""
    nu = _uniform_bosonic_mesh(60.0, 241)
    v = np.zeros((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            v[a, a, b, b] = 3.0 if a == b else 1.5
    u = _product_basis_u(nu, v, np.full((2, 2), 0.4), 1.0, sign=+1.0)
    m = retardation.u_causality_metrics(u, nu)

    print(f"    [causality anticausal] flips = {m['dd_monotonicity_flips']}  "
          f"max_eig = {m['u0_minus_umax_max_eig']:.6e}")
    assert m["u0_minus_umax_max_eig"] > 0.0
    assert m["causality_violated"]


def test_causality_monitor_counts_diagonal_slope_reversals():
    """More than one sign change of dU_aa/dnu is the (ii) meter's flag."""
    nu = _uniform_bosonic_mesh(60.0, 241)
    _v, u = _causal_u(nu, norb=2)
    # inject an oscillation on the (0,0) density-density diagonal
    u[:, 0, 0, 0, 0] += 0.05 * np.sin(nu)
    m = retardation.u_causality_metrics(u, nu)

    print(f"    [causality oscillating] flips = {m['dd_monotonicity_flips']}")
    assert m["dd_monotonicity_flips"] > 1
    assert m["dd_monotonicity_flagged"]
    assert m["causality_violated"]


def test_causality_monitor_detects_non_hermitian_u():
    nu = _uniform_bosonic_mesh(60.0, 121)
    _v, u = _causal_u(nu, norb=2)
    u[3, 0, 1, 1, 0] += 1e-3          # break the pair-matrix hermiticity
    m = retardation.u_causality_metrics(u, nu)
    print(f"    [causality hermiticity] herm = {m['hermiticity_max']:.3e}")
    assert m["hermiticity_max"] >= 1e-3


def test_combine_static_and_retarded_u_handles_both_shape_conventions():
    """U(inu) assembly must work for a product-basis AND a density-density u_weiss."""
    nu = _uniform_bosonic_mesh(60.0, 61)
    norb = 2
    v = np.zeros((norb, norb, norb, norb))
    for a in range(norb):
        for b in range(norb):
            v[a, a, b, b] = 3.0 if a == b else 1.5
    u_pb = _product_basis_u(nu, np.zeros_like(v), np.outer([1.0, 0.8], [1.0, 0.8]) * 0.2, 1.0)

    full_pb = retardation.combine_static_and_retarded_u(v, u_pb)
    assert full_pb.shape == u_pb.shape
    assert full_pb[0, 0, 0, 0, 0] == v[0, 0, 0, 0] + u_pb[0, 0, 0, 0, 0]

    u_dd = retardation.to_density_density(u_pb)
    full_dd = retardation.combine_static_and_retarded_u(v, u_dd)
    assert full_dd.shape == u_dd.shape
    assert np.allclose(full_dd, retardation.to_density_density(full_pb))

    # ... and the monitor accepts the density-density form
    m = retardation.u_causality_metrics(full_dd, nu)
    assert m["dd_monotonicity_flips"] == 0

    try:
        retardation.combine_static_and_retarded_u(v, np.zeros((5, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible shapes must raise ValueError")


def test_causality_trail_layout():
    nu = _uniform_bosonic_mesh(60.0, 121)
    _v, u = _causal_u(nu, norb=2)
    m = retardation.u_causality_metrics(u, nu)
    trail = retardation.causality_trail(m)
    assert trail.shape == (4,)
    assert retardation.CAUSALITY_TRAIL_LABELS == (
        "hermiticity_max", "dd_monotonicity_flips",
        "u0_minus_umax_min_eig", "u0_minus_umax_max_eig")
    assert trail[0] == m["hermiticity_max"]
    assert trail[3] == m["u0_minus_umax_max_eig"]


# ==========================================================================
# Gate Q4-b -- the clean limit (needs coqui + h5; no QMC, no modest)
# ==========================================================================

class _LinearEmbedding1e:
    """
    Deterministic stand-in for ``modest.embedding.merge_embed_block_by_imp``:
    a single fixed linear map ``A -> R A R^dag`` per spin block. Any fixed linear
    map is enough for the clean-limit statement -- the gate is that ``imp`` and
    ``dc`` traverse the SAME map, so identical inputs cancel exactly.
    """

    def __init__(self, rot, gf_struct):
        self.rot = rot
        self.imp_block_structure = [gf_struct]
        self.n_impurities = 1

    def embed(self, per_impurity):
        blocks = per_impurity[0]
        return [np.einsum('mi,...ij,nj->...mn', self.rot, np.asarray(b), self.rot.conj())
                for b in blocks]


class _LinearEmbedding2e(_LinearEmbedding1e):
    def embed(self, per_impurity):
        blk = np.asarray(per_impurity[0][0])
        return [np.einsum('mi,...ij,nj->...mn', self.rot, blk, self.rot.conj())]


def _clean_limit_solver_results(norb=2, nw_f=8, nw_b=5):
    """
    The survey's §5 mapping (spec §3 P1 gate Q4-b): the "GW impurity" IS the DC,
    so ``Sigma_iw_data := Sigma_iw_dc_data``, ``Pi_iw_data := Pi_iw_dc_data`` and
    ``Sigma_infty := Sigma_infty_dc``.
    """
    rng = np.random.default_rng(4242)

    def _herm(shape):
        a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return a + np.conjugate(np.swapaxes(a, -1, -2))

    sigma_dc = [_herm((nw_f, norb, norb)), _herm((nw_f, norb, norb))]
    vhf_dc = [_herm((norb, norb)), _herm((norb, norb))]
    pi_dc = [np.real(_herm((nw_b, norb, norb))).astype(complex)]

    return [{
        'gf_struct': [('up', norb), ('down', norb)],
        'Sigma_infty_dc': vhf_dc,
        'Sigma_iw_dc_data': sigma_dc,
        'Pi_iw_dc_data': pi_dc,
        # --- the clean limit: the impurity solution IS the double counting ---
        'Sigma_infty': vhf_dc,
        'Sigma_iw_data': sigma_dc,
        'Pi_iw_data': pi_dc,
    }]


def test_clean_limit_embed_impurities_cancels_exactly():
    """
    Gate Q4-b: with the "GW impurity" solved by ``solve_gw_dc`` itself,
    ``local_sigma_w["imp"] - ["dc"]`` and ``local_pi_w["imp"] - ["dc"]`` must be
    identically zero at machine precision, i.e. the lattice correction vanishes.
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q4-b needs the coqui python package")
    pytest.importorskip("h5", reason="coqui.dmft.weiss imports h5")
    from coqui.dmft.weiss import embed_impurities

    norb, nmlwf = 2, 3
    rng = np.random.default_rng(7)
    rot = rng.normal(size=(nmlwf, norb)) + 1j * rng.normal(size=(nmlwf, norb))
    gf_struct = [('up', norb), ('down', norb)]

    res = _clean_limit_solver_results(norb=norb)
    local_sigma_w, local_hf, local_pi_w = embed_impurities(
        _LinearEmbedding1e(rot, gf_struct),
        _LinearEmbedding2e(rot, gf_struct),
        res, False
    )

    d_sigma = np.max(np.abs(local_sigma_w['imp'] - local_sigma_w['dc']))
    d_hf = np.max(np.abs(local_hf['imp'] - local_hf['dc']))
    d_pi = np.max(np.abs(local_pi_w['imp'] - local_pi_w['dc']))
    print(f"    [Q4-b] |Sigma_imp - Sigma_dc| = {d_sigma:.3e}   "
          f"|Vhf_imp - Vhf_dc| = {d_hf:.3e}   |Pi_imp - Pi_dc| = {d_pi:.3e}")

    assert d_sigma == 0.0
    assert d_hf == 0.0
    assert d_pi == 0.0
    # the embedding must not be trivial, otherwise the gate is vacuous
    assert np.max(np.abs(local_sigma_w['imp'])) > 0.0
    assert np.max(np.abs(local_pi_w['imp'])) > 0.0


def test_clean_limit_from_solve_gw_dc_outputs():
    """
    Gate Q4-b, end to end: build the DC with the production ``solve_gw_dc``
    (weiss.py:490-511) from a synthetic (Gloc, Wloc, u_weiss), feed its outputs
    back as the impurity solution, and require exact cancellation.
    """
    import pytest
    pytest.importorskip("coqui", reason="gate Q4-b needs the coqui python package")
    pytest.importorskip("h5", reason="coqui.dmft.weiss imports h5")
    from coqui.utils.imag_axes_ft import IAFT
    from coqui.dmft.weiss import solve_gw_dc, embed_impurities

    iaft = IAFT(beta=20.0, wmax=10.0, prec='high', verbose=False)

    norb, nmlwf = 2, 3
    gf_struct = [('up', norb), ('down', norb)]
    rng = np.random.default_rng(11)

    # A synthetic non-interacting G(tau) on the IAFT fermionic tau mesh, and a
    # W(tau) on the ph-symmetric half mesh that eval_pi_rpa/eval_gw_dc_t assume
    # (weiss.py:419-438 / :467-487: nts_half = nt_f//2 + nt_f%2).
    nts = iaft.nt_f
    nts_half = nts // 2 + nts % 2
    tau = np.asarray(iaft.tau_mesh('f'), dtype=float)
    eps = np.array([-0.3, 0.4])
    G_t = np.zeros((nts, 2, norb, norb), dtype=complex)
    for a in range(norb):
        g = -np.exp(-eps[a] * tau) / (1.0 + np.exp(-eps[a] * iaft.beta))
        G_t[:, 0, a, a] = g
        G_t[:, 1, a, a] = g

    V = np.zeros((norb, norb, norb, norb))
    for a in range(norb):
        for b in range(norb):
            V[a, a, b, b] = 4.0 if a == b else 2.0
    W_t = np.zeros((nts_half, norb, norb, norb, norb), dtype=complex)
    for t in range(nts_half):
        W_t[t] = -0.5 * V * np.exp(-3.0 * t / max(nts_half - 1, 1))
    # solve_gw_dc reads only u_weiss_iw[0] (the static limit, weiss.py:494).
    u_weiss = np.zeros((1,) + V.shape, dtype=complex)

    dc = solve_gw_dc(G_t, V, W_t, u_weiss, iaft, density_only=True, gf_struct=gf_struct)

    res = [{
        'gf_struct': gf_struct,
        'Sigma_infty_dc': dc['Sigma_infty_dc'],
        'Sigma_iw_dc_data': dc['Sigma_iw_dc_data'],
        'Pi_iw_dc_data': dc['Pi_iw_dc_data'],
        'Sigma_infty': dc['Sigma_infty_dc'],
        'Sigma_iw_data': dc['Sigma_iw_dc_data'],
        'Pi_iw_data': dc['Pi_iw_dc_data'],
    }]

    rot = rng.normal(size=(nmlwf, norb)) + 1j * rng.normal(size=(nmlwf, norb))
    local_sigma_w, local_hf, local_pi_w = embed_impurities(
        _LinearEmbedding1e(rot, gf_struct),
        _LinearEmbedding2e(rot, gf_struct),
        res, False
    )

    d_sigma = np.max(np.abs(local_sigma_w['imp'] - local_sigma_w['dc']))
    d_hf = np.max(np.abs(local_hf['imp'] - local_hf['dc']))
    d_pi = np.max(np.abs(local_pi_w['imp'] - local_pi_w['dc']))
    print(f"    [Q4-b e2e] |dSigma| = {d_sigma:.3e}  |dVhf| = {d_hf:.3e}  |dPi| = {d_pi:.3e}")

    assert d_sigma == 0.0 and d_hf == 0.0 and d_pi == 0.0
    assert np.max(np.abs(local_pi_w['dc'])) > 0.0


# ==========================================================================
# Standalone runner for the numpy-only tier (no pytest required)
# ==========================================================================

if __name__ == "__main__":
    import sys
    import traceback

    numpy_only = [name for name, obj in sorted(globals().items())
                  if name.startswith("test_") and callable(obj)
                  and "clean_limit" not in name]
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
          f"(clean-limit legs need coqui; run them with pytest on a coqui host).")
    sys.exit(1 if failures else 0)
