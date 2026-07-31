#!/usr/bin/env python3
"""Harvest CoQui G0W0 gaps and compare them against the ABINIT reference.

CoQui writes quasiparticle energies to <output>.mbpt.h5 at
/scf/iter<final_iter>/E_ska with shape (nspin, nk, nbnd), in Hartree, alongside
the chemical potential mu.

E_ska may be stored on the IBZ or on the full BZ; either is fine for a gap,
since every full-BZ k is a symmetry image of an IBZ k and carries the same
eigenvalues. min/max over whichever set is stored gives the same answer.

Gaps are reported ON THE SAMPLED MESH, matching harvest_abinit_gw.py, so the
two are directly comparable run-for-run.
"""

import argparse
import csv
import os
import re
import sys

import h5py
import numpy as np

HA = 27.211386245988

NOCC = {"si": 4, "c": 4, "sic": 4, "alp": 4, "mgo": 8}
TAG = re.compile(r"^(?P<mat>[a-z]+)_k(?P<k>\d+)_n(?P<nb>\d+)_e(?P<ec>[\d.]+)$")


def read_gaps(h5path, nocc):
    with h5py.File(h5path, "r") as f:
        if "scf" not in f:
            return None
        scf = f["scf"]
        it = int(np.array(scf["final_iter"]).ravel()[0]) if "final_iter" in scf else None
        grp = None
        if it is not None and ("iter%d" % it) in scf:
            grp = scf["iter%d" % it]
        else:                                  # fall back to the highest iterN
            iters = sorted(int(k[4:]) for k in scf if k.startswith("iter"))
            if not iters:
                return None
            it = iters[-1]
            grp = scf["iter%d" % it]
        if "E_ska" not in grp:
            return None
        E = np.array(grp["E_ska"])             # (nspin, nk, nbnd), Hartree
        mu = float(np.array(grp["mu"]).ravel()[0]) if "mu" in grp else None
    if E.ndim != 3 or E.shape[0] < 1:
        return None
    e = E[0].real if np.iscomplexobj(E) else E[0]          # (nk, nbnd)
    if e.shape[1] <= nocc:
        return None
    vb = e[:, nocc - 1]
    cb = e[:, nocc]
    ik_v = int(np.argmax(vb))
    ik_c = int(np.argmin(cb))
    direct = float(np.min(cb - vb))                        # smallest vertical gap
    return dict(indirect_qp=float((cb[ik_c] - vb[ik_v]) * HA),
                direct_min_qp=float(direct * HA),
                mu=mu, nk=int(e.shape[0]), nbnd=int(e.shape[1]),
                iter=it, vbm_ik=ik_v, cbm_ik=ik_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--abinit-csv", default=None,
                    help="output of harvest_abinit_gw.py, for a side-by-side")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    ref = {}
    if args.abinit_csv and os.path.exists(args.abinit_csv):
        for r in csv.DictReader(open(args.abinit_csv)):
            ref[r["run"]] = r

    rows = []
    for d in sorted(os.listdir(args.root)):
        m = TAG.match(d)
        if not m or m.group("mat") not in NOCC:
            continue
        rundir = os.path.join(args.root, d)
        cands = [f for f in os.listdir(rundir) if f.endswith(".mbpt.h5")] \
            if os.path.isdir(rundir) else []
        if not cands:
            continue
        r = None
        for c in cands:
            try:
                r = read_gaps(os.path.join(rundir, c), NOCC[m.group("mat")])
            except Exception as exc:                       # noqa: BLE001
                print("# %s: %s" % (d, exc), file=sys.stderr)
                r = None
            if r:
                break
        if not r:
            continue
        r.update(run=d, material=m.group("mat"), ngkpt=int(m.group("k")),
                 nband=int(m.group("nb")), ecut=float(m.group("ec")))
        rows.append(r)

    if not rows:
        print("no CoQui results under %s" % args.root, file=sys.stderr)
        return 1

    print("%-5s %-5s %-6s %-7s %-4s %10s %10s %10s"
          % ("mat", "ngkpt", "nband", "ecut", "nk",
             "CoQui_ind", "ABINIT_ind", "diff(eV)"))
    for r in sorted(rows, key=lambda x: (x["material"], x["ngkpt"], x["nband"], x["ecut"])):
        a = ref.get(r["run"])
        av = float(a["indirect_qp"]) if a and a.get("indirect_qp") else None
        d = (r["indirect_qp"] - av) if av is not None else None
        print("%-5s %-5d %-6d %-7.1f %-4d %10.3f %10s %10s"
              % (r["material"], r["ngkpt"], r["nband"], r["ecut"], r["nk"],
                 r["indirect_qp"],
                 "%.3f" % av if av is not None else "-",
                 "%+.3f" % d if d is not None else "-"))

    if args.csv:
        cols = ["material", "ngkpt", "nband", "ecut", "nk", "nbnd", "iter",
                "indirect_qp", "direct_min_qp", "mu", "run"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print("\nwrote %s (%d rows)" % (args.csv, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
