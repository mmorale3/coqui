#!/usr/bin/env python3
"""ISDF-rank sensitivity of G0W0: static terms via ISDF-ERI vs computed directly.

Two routes, identical in every other respect:

  isdf    [gw]/[evgw0] interaction = "eri"            -- Hartree, exchange AND
                                                         the dynamic self-energy
                                                         all through the ISDF ERI
  direct  ... + interaction_hf = "ham"                -- Hartree and exchange from
                                                         [interaction.hamilt] (no
                                                         factorization), ISDF used
                                                         only for the dynamic term

Both converge to the same answer as N_mu -> infinity, so the question is which
approaches it faster. The static terms carry the bare Coulomb interaction and
the full occupied manifold, so they are the more demanding thing to interpolate;
if that is where the ISDF error lives, the direct route should be markedly
flatter in N_mu.

Two observables, per the intended figure:
  * the quasiparticle band gap          (from evgw0: scf/iter<N>/E_ska)
  * the dynamic self-energy on the      (from gw:    scf/iter<N>/Sigma_tskij)
    imaginary axis

Reference for both is the tightest-threshold DIRECT run, which carries the least
total ISDF error. NOTE the routes are compared against a COMMON reference, not
each against its own -- otherwise each curve measures only its own internal
consistency and the plot cannot show that one route is closer to the truth.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

HA = 27.211386245988
# Strictly <route>_<thresh>. Deliberately NOT tolerant of a leading tag: an
# earlier version accepted one so that differently-sized pilot runs could sit in
# the same tree, and "pilot_isdf_1e-4" then collided with "isdf_1e-4" on the
# (route, thresh) key. The dict silently kept whichever was read last, mixing a
# kp3/n250 run into a kp2/n100 table -- visible only because its N_mu and gap
# were wildly out of line with their neighbours. Keep runs of different size in
# separate roots.
TAG = re.compile(r"^(?P<route>isdf|direct)_(?P<thresh>[0-9.eE+-]+)$")


def _decomplex(a):
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 2 and not np.iscomplexobj(a):
        return a[..., 0] + 1j * a[..., 1]
    return a


def _final_iter(f):
    scf = f["scf"]
    if "final_iter" in scf:
        it = int(np.array(scf["final_iter"]).ravel()[0])
        if ("iter%d" % it) in scf:
            return it
    its = sorted(int(k[4:]) for k in scf if k.startswith("iter"))
    return its[-1] if its else None


def read_gap(d, nocc):
    """QP band gap (eV) from the evgw0 output; None if absent or still iter0."""
    for p in glob.glob(os.path.join(d, "*.mbpt.h5")):
        try:
            with h5py.File(p, "r") as f:
                if "scf" not in f:
                    continue
                it = _final_iter(f)
                if it is None or it < 1:
                    continue
                g = f["scf"]["iter%d" % it]
                if "E_ska" not in g:
                    continue
                E = _decomplex(g["E_ska"][...])
        except (OSError, KeyError):
            continue
        e = E[0].real if np.iscomplexobj(E) else E[0]
        if e.shape[1] <= nocc:
            continue
        return float((np.min(e[:, nocc]) - np.max(e[:, nocc - 1])) * HA)
    return None


def read_sigma(d):
    """Dynamic self-energy Sigma(i tau) from the gw output."""
    for p in glob.glob(os.path.join(d, "*.mbpt.h5")):
        try:
            with h5py.File(p, "r") as f:
                if "scf" not in f:
                    continue
                it = _final_iter(f)
                if it is None or it < 1:
                    continue
                g = f["scf"]["iter%d" % it]
                if "Sigma_tskij" not in g:
                    continue
                return _decomplex(g["Sigma_tskij"][...])
        except (OSError, KeyError):
            continue
    return None


def read_nmu(d):
    for name in ("evgw0.out", "gw.out"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            if "interpolating points" in line:
                try:
                    return int(line.split()[-1])
                except ValueError:
                    pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--nocc", type=int, default=4, help="occupied bands (Si: 4)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = {}
    for d in sorted(os.listdir(args.root)):
        m = TAG.match(d)
        if not m:
            continue
        full = os.path.join(args.root, d)
        runs[(m.group("route"), float(m.group("thresh")))] = dict(
            dir=full, nmu=read_nmu(full),
            gap=read_gap(full, args.nocc), sigma=read_sigma(full))

    if not runs:
        print("no runs matching <route>_<thresh> under %s" % args.root,
              file=sys.stderr)
        return 1

    # common reference: tightest-threshold direct run that actually has data
    dref = sorted([k for k in runs if k[0] == "direct" and runs[k]["gap"] is not None],
                  key=lambda k: k[1])
    ref = runs[dref[0]] if dref else None
    if ref is None:
        print("# no completed 'direct' run yet -- differences vs reference omitted",
              file=sys.stderr)

    print("%-7s %-9s %-7s %12s %12s %14s"
          % ("route", "thresh", "Nmu", "gap(eV)", "d_gap(meV)", "rel|dSigma|"))
    print("-" * 68)
    rows = []
    for (route, thr) in sorted(runs, key=lambda k: (k[0], -k[1])):
        r = runs[(route, thr)]
        dg = ds = None
        if ref is not None and r["gap"] is not None and ref["gap"] is not None:
            dg = (r["gap"] - ref["gap"]) * 1000.0
        if ref is not None and r["sigma"] is not None and ref["sigma"] is not None \
                and r["sigma"].shape == ref["sigma"].shape:
            num = np.linalg.norm(r["sigma"] - ref["sigma"])
            den = np.linalg.norm(ref["sigma"])
            ds = float(num / den) if den > 0 else None
        print("%-7s %-9s %-7s %12s %12s %14s"
              % (route, thr, r["nmu"] if r["nmu"] else "-",
                 "%.4f" % r["gap"] if r["gap"] is not None else "(pending)",
                 "%+.1f" % dg if dg is not None else "-",
                 "%.3e" % ds if ds is not None else "-"))
        rows.append(dict(route=route, thresh=thr, nmu=r["nmu"], gap=r["gap"],
                         d_gap_meV=dg, rel_dSigma=ds))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
