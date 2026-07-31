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


def _decomplex(a):
    """nda writes complex arrays with a trailing (2,) real/imag axis."""
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 2 and not np.iscomplexobj(a):
        return a[..., 0] + 1j * a[..., 1]
    return a


def _gap_from(e, nocc):
    """(indirect, smallest-vertical) gap in eV from a real (nk, nbnd) array."""
    vb, cb = e[:, nocc - 1], e[:, nocc]
    return float((np.min(cb) - np.max(vb)) * HA), float(np.min(cb - vb) * HA)


def read_gaps(h5path, nocc):
    with h5py.File(h5path, "r") as f:
        if "scf" not in f:
            return None
        # KS eigenvalues of the converted mean field, on the FULL BZ.  Comparing
        # this against ABINIT's own KS gap is the check that the conversion
        # preserved the mean field: any disagreement there means the QP
        # comparison is meaningless.
        ks_ind = ks_dir = None
        if "mean_field/eigvals" in f:
            ks = _decomplex(f["mean_field/eigvals"][...])
            ks = ks.real if np.iscomplexobj(ks) else ks
            if ks.ndim == 3 and ks.shape[2] > nocc:
                ks_ind, ks_dir = _gap_from(ks[0], nocc)
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
        # iter0 is the KS starting point, NOT a quasiparticle result.  A run that
        # is still going has only iter0, and reporting it silently yields the
        # mean-field gap labelled as G0W0 -- numbers that look entirely
        # plausible and are simply the DFT answer.  Refuse it.
        if it is not None and it < 1:
            return None
        if "E_ska" not in grp:
            return None
        E = _decomplex(grp["E_ska"][...])       # (nspin, nk, nbnd), Hartree
        mu = float(np.array(grp["mu"]).ravel()[0]) if "mu" in grp else None
    if E.ndim != 3 or E.shape[0] < 1:
        return None
    e = E[0].real if np.iscomplexobj(E) else E[0]          # (nk, nbnd)
    if e.shape[1] <= nocc:
        return None
    ind, direct = _gap_from(e, nocc)
    return dict(indirect_qp=ind, direct_min_qp=direct,
                ks_indirect=ks_ind, ks_direct=ks_dir,
                mu=mu, nk=int(e.shape[0]), nbnd=int(e.shape[1]), iter=it)


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

    print("%-5s %-5s %-6s %-7s | %9s %9s %8s | %9s %9s %8s"
          % ("mat", "ngkpt", "nband", "ecut",
             "KS CoQui", "KS ABINIT", "dKS",
             "QP CoQui", "QP ABINIT", "dQP"))
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (x["material"], x["ngkpt"], x["nband"], x["ecut"])):
        a = ref.get(r["run"])
        aq = float(a["indirect_qp"]) if a and a.get("indirect_qp") else None
        ak = float(a["indirect_ks"]) if a and a.get("indirect_ks") else None
        ck, cq = r.get("ks_indirect"), r["indirect_qp"]
        dk = (ck - ak) if (ak is not None and ck is not None) else None
        dq = (cq - aq) if aq is not None else None
        fmt = lambda v, p="%.3f": (p % v) if v is not None else "-"      # noqa: E731
        print("%-5s %-5d %-6d %-7.1f | %9s %9s %8s | %9s %9s %8s"
              % (r["material"], r["ngkpt"], r["nband"], r["ecut"],
                 fmt(ck), fmt(ak), fmt(dk, "%+.3f"),
                 fmt(cq), fmt(aq), fmt(dq, "%+.3f")))

    if args.csv:
        cols = ["material", "ngkpt", "nband", "ecut", "nk", "nbnd", "iter",
                "ks_indirect", "ks_direct", "indirect_qp", "direct_min_qp",
                "mu", "run"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print("\nwrote %s (%d rows)" % (args.csv, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
