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
Q6 §1.1 (notes/q6_diagnostics_closeout_spec.md, PDF §8.3) -- the R(inu) CANCELLATION LOAD,
python half. Gate pattern is the Q5 numpy-only tier's: synthetic arrays with EXACT
expectations, ``outer_loop.py`` loaded by file path so nothing here needs ``coqui`` or
``triqs``.

What is measured:

    R(inu) = ||P_imp(inu) - P_dc(inu)||_max / ||P_dc(inu)||_max

per bosonic node, aggregated into the three nu bands of spec §1.1 (the nu = 0 node, the
middle third, the top third) and APPENDED to the Q5-b Mott-chain trail and log block,
alongside the C3b ladder column ||P^lad_loc,orb||_max / ||P_dc||_max.

A SEPARATE FILE on purpose: extending ``test_q5_outer_loop.py`` would move that suite's
measured 16/16 tally, and the Q5 gates are commit-point gates for this increment.

RUN COMMANDS
------------
Standalone (this is how it was measured -- no pytest needed, no coqui, no triqs):

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 src/python/dmft/tests/test_q6_diagnostics.py

Under pytest on a TRIQS host, alongside the tiers that must stay green:

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      python3 -m pytest -v src/python/dmft/tests/test_q6_diagnostics.py \\
                           src/python/dmft/tests/test_q5_outer_loop.py \\
                           src/python/dmft/tests/test_q4_edmft_skeleton.py

MEASURED 2026-08-14 on the implementation host (numpy 2.x, no coqui/triqs): **10/10** legs
pass. Every leg in this file is numpy-only; there is no coqui-gated leg to skip.
"""

import importlib.util
import pathlib

import numpy as np


# --------------------------------------------------------------------------
# Load dmft/outer_loop.py standalone (no coqui / triqs import) -- the Q5 pattern.
# --------------------------------------------------------------------------

def _load_outer_loop():
    path = pathlib.Path(__file__).resolve().parents[1] / "outer_loop.py"
    spec = importlib.util.spec_from_file_location("_q6_outer_loop_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ol = _load_outer_loop()


# --------------------------------------------------------------------------
# Fixtures: a synthetic {imp, dc} pair with per-node ratios chosen BY HAND.
# --------------------------------------------------------------------------

def _pi_fixture(u_per_node, d_per_node=None):
    """
    Build ``{"imp": ..., "dc": ...}`` with shape ``(n_nu, 2, 2)`` such that, exactly,

        ||P_dc(inu)||_max = d[n]        and    ||P_imp(inu) - P_dc(inu)||_max = u[n],

    so ``R(inu) = u[n] / d[n]`` with no floating-point slack: ``dc`` is a constant block
    (its max is the constant) and the difference is a single-entry spike (its max is the
    spike). ``d = 1`` by default, which makes R equal to u term by term. Every fixture value
    below is a DYADIC rational, so ``(d + u) - d == u`` holds bit-exactly and the
    expectations can be plain ``==`` rather than a tolerance.
    """
    u = np.asarray(u_per_node, dtype=float)
    d = np.ones_like(u) if d_per_node is None else np.asarray(d_per_node, dtype=float)
    n = u.shape[0]
    dc = np.zeros((n, 2, 2), dtype=complex)
    imp = np.zeros((n, 2, 2), dtype=complex)
    for i in range(n):
        dc[i, :, :] = d[i]
        imp[i, :, :] = d[i]
        imp[i, 0, 0] += u[i]
    return {"imp": imp, "dc": dc}


# ==========================================================================
# The nu-band partition (spec §1.1)
# ==========================================================================

def test_nu_band_slices_are_the_spec_bands():
    """
    nu = 0 is the NODE alone; mid and top are floor-division thirds. The BOTTOM third above
    nu = 0 is deliberately not aggregated -- its interesting node IS nu = 0, and folding the
    rest of it into an aggregate would let a low-but-nonzero node masquerade as the node.
    """
    nu0, mid, top = ol.nu_band_slices(9)
    assert nu0.tolist() == [0]
    assert mid.tolist() == [3, 4, 5]
    assert top.tolist() == [6, 7, 8]

    # a leftover node lands in the TOP band (floor division on both cuts)
    nu0, mid, top = ol.nu_band_slices(10)
    assert nu0.tolist() == [0]
    assert mid.tolist() == [3, 4, 5]
    assert top.tolist() == [6, 7, 8, 9]

    # degenerate axes must not raise
    nu0, mid, top = ol.nu_band_slices(1)
    assert nu0.tolist() == [0] and mid.tolist() == [] and top.tolist() == [0]
    nu0, mid, top = ol.nu_band_slices(0)
    assert nu0.tolist() == [] and mid.tolist() == [] and top.tolist() == []
    print("    [Q6 §1.1] nu bands (n=9): nu0=[0], mid=[3,4,5], top=[6,7,8]")


# ==========================================================================
# R(inu) itself
# ==========================================================================

def test_r_cancellation_load_is_exact_on_a_synthetic_pi():
    """
    EXACT expectations. Nodes 1 and 2 carry a deliberately huge R (9.0) and sit in the
    BOTTOM third: they must NOT reach any aggregate. If the bands ever silently become
    "low/mid/top thirds", this leg fires.
    """
    u = [0.5, 9.0, 9.0, 0.125, 0.375, 0.25, 0.0625, 0.75, 0.03125]
    r_nu0, r_mid, r_top = ol.r_cancellation_load(_pi_fixture(u))
    print(f"    [Q6 §1.1] R = (nu0 {r_nu0}, mid {r_mid}, top {r_top})")
    assert r_nu0 == 0.5
    assert r_mid == 0.375         # max over nodes 3,4,5 = max(0.125, 0.375, 0.25)
    assert r_top == 0.75          # max over nodes 6,7,8 = max(0.0625, 0.75, 0.03125)


def test_r_cancellation_load_scales_with_the_dc_norm():
    """R is a RATIO: doubling ||P_dc|| at fixed numerator halves it, node by node."""
    u = [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0]
    d = [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 4.0, 1.0, 1.0]
    r_nu0, r_mid, r_top = ol.r_cancellation_load(_pi_fixture(u, d))
    assert r_nu0 == 0.25          # 0.5 / 2.0
    assert r_mid == 0.25          # 0.5 / 2.0, the only nonzero node of the middle third
    assert r_top == 0.125         # 0.5 / 4.0


def test_r_cancellation_load_vanishes_in_the_clean_limit():
    """P_imp == P_dc is Q4-b's clean limit: perfect cancellation, R exactly 0 everywhere."""
    fix = _pi_fixture([0.0] * 9)
    assert ol.r_cancellation_load(fix) == (0.0, 0.0, 0.0)


def test_r_cancellation_load_skips_dead_nodes():
    """
    A node whose ||P_dc|| vanishes is NOT a cancellation failure (0/0), so it contributes
    nothing; a band with no live node reports the FINITE MISSING sentinel.
    """
    u = [0.5, 0.0, 0.0, 0.125, 0.375, 0.25, 0.0625, 0.75, 0.03125]
    d = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    r_nu0, r_mid, r_top = ol.r_cancellation_load(_pi_fixture(u, d))
    assert r_nu0 == 0.5
    assert r_mid == 0.25          # node 4 (the band max) is dead => max(0.125, 0.25)
    assert r_top == ol.MISSING    # the whole top third is dead
    assert np.isfinite(r_top)
    print(f"    [Q6 §1.1] dead-node handling: mid {r_mid}, top {r_top} (MISSING, finite)")


def test_r_cancellation_load_is_missing_on_unusable_input():
    """Nothing to compare yet, a malformed dict, or mismatched shapes: all MISSING, finite."""
    miss = (ol.MISSING,) * 3
    assert ol.r_cancellation_load(None) == miss
    assert ol.r_cancellation_load({"imp": np.zeros((3, 2))}) == miss
    assert ol.r_cancellation_load({"imp": np.zeros((3, 2)), "dc": np.zeros((4, 2))}) == miss
    assert ol.r_cancellation_load({"imp": np.zeros((0, 2)), "dc": np.zeros((0, 2))}) == miss
    assert all(np.isfinite(v) for v in ol.r_cancellation_load(None))


# ==========================================================================
# The C3b ladder column
# ==========================================================================

def test_ladder_over_dc_is_the_cpp_ratio_definition():
    """
    ``||P^lad_loc,orb||_max / ||P_dc||_max`` -- the same max-norm ratio the C++ side reports
    as pol_lad_loc_orb_ratio() (scr_coulomb_t.h:355-360).
    """
    lad = np.array([1.0, -3.0, 2.0])
    dc = np.array([0.5, 2.0])
    assert ol.ladder_over_dc(lad, dc) == 1.5          # 3 / 2
    assert ol.ladder_over_dc(np.zeros(4), dc) == 0.0
    # absent operands / a vanishing DC are MISSING, never a divide-by-zero
    assert ol.ladder_over_dc(None, dc) == ol.MISSING
    assert ol.ladder_over_dc(lad, None) == ol.MISSING
    assert ol.ladder_over_dc(lad, np.zeros(3)) == ol.MISSING
    assert ol.ladder_over_dc(np.zeros(0), dc) == ol.MISSING


# ==========================================================================
# The trail and the log block (the Q5-b block, extended)
# ==========================================================================

def test_trail_carries_the_q6_columns_in_the_appended_slots():
    """
    The Q6 columns are APPENDED (spec §1.1) -- the Q5-b slots must not move, and the four new
    ones must land in the documented order.
    """
    labels = ol.MOTT_CHAIN_TRAIL_LABELS
    assert labels[-4:] == ("r_nu0", "r_mid", "r_top", "lad_over_dc")
    assert labels[0] == "gap_eV" and labels[10] == "o_c"     # the Q5-b prefix, unmoved
    assert ol.R_BAND_LABELS == ("r_nu0", "r_mid", "r_top")

    r = ol.r_cancellation_load(
        _pi_fixture([0.5, 9.0, 9.0, 0.125, 0.375, 0.25, 0.0625, 0.75, 0.03125]))
    trail = ol.mott_chain_trail(gap_eV=1.85,
                                **dict(zip(ol.R_BAND_LABELS, r)),
                                lad_over_dc=0.64)
    v = dict(zip(labels, trail))
    assert np.all(np.isfinite(trail))
    assert (v["r_nu0"], v["r_mid"], v["r_top"]) == (0.5, 0.375, 0.75)
    assert v["lad_over_dc"] == 0.64
    assert v["gap_eV"] == 1.85
    # a cycle that measured no polarization still produces a FINITE trail
    assert ol.mott_chain_trail(gap_eV=1.0)[-4:].tolist() == [ol.MISSING] * 4
    print(f"    [Q6 §1.1] appended trail slots = "
          f"{ {k: v[k] for k in labels[-4:]} }")


def test_r_columns_reject_a_typo_like_every_other_field():
    """The append must not weaken mott_chain_trail's unknown-key guard."""
    try:
        ol.mott_chain_trail(r_nu_0=1.0)          # extra underscore
    except ValueError as e:
        assert "r_nu_0" in str(e)
    else:
        raise AssertionError("an unknown trail field must raise ValueError")


def test_log_block_survives_a_cycle_that_measured_nothing():
    """The consolidated block gained four lines; it must still run on an empty trail."""
    ol.log_mott_chain(1, 8, ol.mott_chain_trail(), verbose=True)
    ol.log_mott_chain(2, 8, ol.mott_chain_trail(r_nu0=4.2e-2, r_mid=7.8e-2,
                                                r_top=1.3e-1, lad_over_dc=0.64),
                      verbose=True)
    ol.log_mott_chain(3, 8, ol.mott_chain_trail(), verbose=False)


# ==========================================================================
# Standalone runner (no pytest required) -- the Q5 pattern
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
          f"(every Q6 §1.1 leg is numpy-only; nothing here is skipped on a coqui-less host).")
    sys.exit(1 if failures else 0)
