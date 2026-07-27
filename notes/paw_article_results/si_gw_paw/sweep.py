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


def evaluate(spec, tag):
    """Build one dataset and return every acceptance metric in one dict.

    The four gates a production dataset has to pass, in the order they bite:
      build     -- atompaw's overlap operator must stay positive definite
      q_ij      -- max|q_ij| (inflated partial wave -> negative rho in QE)
      Q_occ     -- occupancy-weighted norm defect (KKK: wrong RPA limit)
      logderiv  -- scattering fidelity, windowed, since the KKK failure is high-energy
    """
    ok, d = run_one(spec, tag)
    res = dict(tag=tag, dir=d, ok=ok)
    if not ok:
        log = open(os.path.join(d, "run.log"), errors="replace").read()
        res["why"] = ("not positive definite" if "positive definite" in log
                      else "no UPF (see run.log)")
        return res
    xml = [f for f in os.listdir(d) if f.endswith(".xml") and "corewf" not in f][0]
    rows = D.norm_violations(os.path.join(d, xml))
    q = D.qij_matrix(d)
    res.update(
        q_occ=sum(w["f"] * w["q"] for w in rows),
        qij=(abs(q).max() if q is not None else float("nan")),
        # worst pseudo-over-AE interior norm: the inflation that q_ij is a proxy for
        infl=max(w["n_ps"] / w["n_ae"] for w in rows),
        occ=[(w["label"], w["q"]) for w in rows if w["f"] > 0],
        ld={l: v[1] for l, v in D.logderiv_error(d, 15.0, 30.0).items()},
        ld_low={l: v[1] for l, v in D.logderiv_error(d, 0.0, 5.0).items()},
        poles=D.ae_poles(d),
    )
    return res


def show(r):
    if not r["ok"]:
        print(f"{r['tag']:>26}  FAILED: {r['why']}")
        return
    occ = " ".join(f"{lab}={v:+.5f}" for lab, v in r["occ"])
    ld = " ".join(f"l{l}={v:.3f}" for l, v in sorted(r["ld"].items()))
    print(f"{r['tag']:>26}  Q_occ={r['q_occ']:+.6f}  max|q_ij|={r['qij']:8.3f}  "
          f"infl={r['infl']:7.2f}  [{occ}]  RMS[15,30]: {ld}")


def sweep_root(scheme):
    """Locate the norm-conservation root r_c(q_aa=0) for one pseudization scheme.

    q_aa depends only on its own partial wave, that wave's r_c and the pseudization
    scheme, so the root can be bracketed with the cheap, always-well-conditioned
    2-waves-per-l lmax=1 structure and then reused in the d-complete structures.
    E_ref = 11 Ry sits in the first off-resonance window at every radius in this range
    (the s/p poles run 3.4->2.6 and 4.4->3.5 Ry below, 19.7->16.6 Ry above).
    """
    print(f"norm-conservation root scan, scheme={scheme} (2 waves/l, lmax=1, E_ref=11 Ry)")
    print(f"{'rc':>6}  {'q(3s)':>11}  {'q(3p)':>11}   (root = sign change)")
    for rc in [1.55, 1.58, 1.60, 1.62, 1.64, 1.66, 1.70, 1.80, 1.90]:
        r = evaluate(G.flex_spec(rc, rc, [11.0], [11.0], scheme=scheme),
                     f"ROOT_{scheme}_{rc:.2f}")
        if not r["ok"]:
            print(f"{rc:>6.2f}  FAILED: {r['why']}")
            continue
        qs = dict(r["occ"])
        print(f"{rc:>6.2f}  {qs.get('3s', float('nan')):>11.6f}  "
              f"{qs.get('3p', float('nan')):>11.6f}")


def sweep_band(scheme):
    """Map the (r_c, E_ref) positive-definite band and the norm root inside it.

    The band is scheme-dependent and the two schemes occupy *complementary* corners: at
    E_ref = 11 Ry, MODRRKJ builds only for r_c >= 1.80 while VANDERBILT builds only for
    r_c <= 1.58.  Since VANDERBILT's q_aa is still positive (norm-deficient) at 1.58 its
    root lies at larger r_c, so the band has to be mapped in 2D to find out whether the
    root is reachable at all.  Reference energies avoid the s/p poles near 3-4 and
    17-20 Ry.
    """
    energies = [2.0, 6.0, 8.0, 11.0, 14.0, 16.0, 24.0]
    print(f"(rc, E_ref) band map, scheme={scheme}; cell = q(3s) or 'x' if the build fails")
    print(f"{'rc':>6} " + " ".join(f"{e:>10.0f}" for e in energies))
    for rc in [1.58, 1.60, 1.65, 1.70, 1.75, 1.80, 1.90]:
        cells = []
        for e in energies:
            r = evaluate(G.flex_spec(rc, rc, [e], [e], scheme=scheme),
                         f"BAND_{scheme[:4]}_{rc:.2f}_{e:g}")
            cells.append("         x" if not r["ok"]
                         else f"{dict(r['occ']).get('3s', float('nan')):>10.5f}")
        print(f"{rc:>6.2f} " + " ".join(cells))


def sweep_dcomplete(scheme, rc_s, rc_p):
    """d-complete datasets at norm-matched radii: 2 waves/l, off-resonance energies.

    Scans the d-channel content, which is the part never tried before: the 5-wave form
    (single d reference) and the 6-wave form (two d references).  s/p mid energies are
    held at the first off-resonance midpoint; only the d channel moves.
    """
    print(f"d-complete scan, scheme={scheme}, rc_s={rc_s} rc_p={rc_p} (2 waves/l + d)")
    for ed in ([1.0], [8.0], [11.0], [14.0], [1.0, 11.0], [1.0, 14.0], [8.0, 22.0],
               [11.0, 24.0]):
        tag = f"DC_{scheme[:4]}_{rc_s:.3f}_d" + "-".join(f"{e:g}" for e in ed)
        show(evaluate(G.flex_spec(rc_s, rc_p, [11.0], [11.0], d_extras=ed,
                                 scheme=scheme), tag))


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "energies"
    if mode == "energies":
        sweep_energies()
    elif mode == "rc":
        sweep_rc(sys.argv[2] if len(sys.argv) > 2 else "D2")
    elif mode == "root":
        sweep_root(sys.argv[2] if len(sys.argv) > 2 else "MODRRKJ")
    elif mode == "band":
        sweep_band(sys.argv[2] if len(sys.argv) > 2 else "VANDERBILT")
    elif mode == "dcomp":
        sweep_dcomplete(sys.argv[2] if len(sys.argv) > 2 else "MODRRKJ",
                        float(sys.argv[3]) if len(sys.argv) > 3 else 1.555,
                        float(sys.argv[4]) if len(sys.argv) > 4 else 1.796)
    else:
        raise SystemExit(__doc__)
