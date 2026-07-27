#!/usr/bin/env python3
"""Run atompaw over a grid of dataset parameters and score each result.

Two uses:
  1. Tune reference energies / r_c so a rung is a legitimately good DFT dataset
     (otherwise a bad baseline, not the PAW completeness error, drives the RPA result).
  2. Drive rung D3's norm matching: scan r_c per channel until
     q_aa = int_0^rc (phi^2 - phit^2) r^2 dr -> 0 for the OCCUPIED partial waves,
     which is KKK's option (i) for building norm-conserving partial waves.

Usage:
    sweep.py energies            # p-channel mid reference-energy scan on the D0 baseline
    sweep.py rc <rung>           # r_c scan for norm matching
"""

import copy
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_inputs as G           # noqa: E402
import paw_diag as D             # noqa: E402

ATOMPAW = os.path.expanduser("~/Projects/PAW_GW/build/atompaw-install/bin/atompaw")
SCRATCH = os.path.expanduser("~/Projects/PAW_GW/si_gw_paw/_sweep")


def run_one(spec, tag):
    """Generate + run one dataset; return (ok, dirpath)."""
    d = os.path.join(SCRATCH, tag)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    G.write_input(copy.deepcopy(spec), os.path.join(d, "Si.input"))
    with open(os.path.join(d, "Si.input")) as fin, \
         open(os.path.join(d, "run.log"), "w") as fout:
        # atompaw can hang on a diverging SCF; alarm() bounds it.
        # atompaw writes every output file into the CWD, so it must run inside `d`.
        subprocess.run(["perl", "-e", f'alarm 300; exec "{ATOMPAW}"'],
                       stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d)
    ok = any(f.endswith(".UPF") for f in os.listdir(d))
    return ok, d


def score(d):
    """Return (Q_occ, {l: rms_by_window}) for a completed run."""
    xmls = [f for f in os.listdir(d) if f.endswith(".xml") and "corewf" not in f]
    if not xmls:
        return None, None
    rows = D.norm_violations(os.path.join(d, xmls[0]))
    q_occ = sum(w["f"] * w["q"] for w in rows)
    wins = [(-12.0, 0.0), (0.0, 5.0), (5.0, 15.0), (15.0, 30.0)]
    binned = {}
    for (lo, hi) in wins:
        for l, (_mx, rms) in D.logderiv_error(d, lo, hi).items():
            binned.setdefault(l, []).append(rms)
    return q_occ, binned


def sweep_energies():
    """Scan the second p reference energy; the AE p resonance sits near 3.6 Ry."""
    print("p-channel mid reference-energy scan (D0 baseline, rc=1.90)")
    print(f"{'E_p(Ry)':>8}  {'Q_occ':>10}   p-RMS by window [-12,0] [0,5] [5,15] [15,30]")
    for ep in [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0]:
        spec = G.rung("D0")
        spec["extras"] = {0: [2.5], 1: [ep]}
        ok, d = run_one(spec, f"ep{ep:g}")
        if not ok:
            print(f"{ep:>8.1f}  FAILED (see {d}/run.log)")
            continue
        q, binned = score(d)
        cells = "  ".join(f"{v:6.3f}" for v in binned.get(1, []))
        print(f"{ep:>8.1f}  {q:>10.6f}   {cells}")


def sweep_rc(rung):
    """Scan the matching radius of the OCCUPIED partial waves only.

    q_aa depends solely on that partial wave, its own matching radius and the
    pseudization scheme -- it is independent of the other basis functions (verified:
    changing every reference energy left Q_occ invariant to 6 digits).  So the norm
    can be matched channel-by-channel by shrinking only the bound valence waves, while
    the augmentation sphere r_c and the unbound waves stay at their original radii.
    That keeps the overlap operator well conditioned, which a global rescaling does not.
    """
    base = G.rung(rung)
    print(f"r_c(occupied) scan for {rung}: driving the occupied-wave norm defect to zero")
    print(f"{'rc_occ':>7}  {'Q_occ':>11}  {'q(3s)':>11} {'q(3p)':>11}")
    for rc_occ in [1.90, 1.75, 1.60, 1.45, 1.30, 1.15, 1.00]:
        spec = copy.deepcopy(base)
        for l in spec["rc"]:
            n_bound = sum(1 for (_n, ll) in spec["valence"] if ll == l)
            for i in range(n_bound):                       # bound waves come first
                spec["rc"][l][i] = min(rc_occ, spec["rc"][l][i])
        ok, d = run_one(spec, f"{rung}_occ{rc_occ:.2f}")
        if not ok:
            print(f"{rc_occ:>7.2f}  FAILED")
            continue
        xmls = [f for f in os.listdir(d) if f.endswith(".xml") and "corewf" not in f]
        rows = D.norm_violations(os.path.join(d, xmls[0]))
        q_occ = sum(w["f"] * w["q"] for w in rows)
        occ = [w["q"] for w in rows if w["f"] > 0]
        print(f"{rc_occ:>7.2f}  {q_occ:>11.6f}  "
              + " ".join(f"{v:>11.6f}" for v in occ))


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "energies"
    if mode == "energies":
        sweep_energies()
    elif mode == "rc":
        sweep_rc(sys.argv[2] if len(sys.argv) > 2 else "D2")
    else:
        raise SystemExit(__doc__)
