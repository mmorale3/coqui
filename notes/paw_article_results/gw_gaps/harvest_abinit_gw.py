#!/usr/bin/env python3
"""Harvest ABINIT PAW G0W0 gaps from a tree of gw_gaps run directories.

ABINIT writes one YAML-ish SelfEnergy_ee block per k-point carrying KS_gap,
QP_gap and a band table whose last column is the quasiparticle energy.  The
DIRECT gap at each k is taken from ABINIT's own QP_gap; the INDIRECT gap is
recomputed here as min_k E_QP(nocc+1) - max_k E_QP(nocc), because ABINIT does
not report it.

Both are gaps *on the sampled mesh*: the true CBM of Si (0.85X) and C (0.76X)
does not lie on a Gamma-centered Monkhorst-Pack grid, so what is reported is the
mesh minimum.  That is the number to compare against other codes run on the same
mesh, and it is what the k-point extrapolation acts on.
"""

import argparse
import csv
import os
import re
import sys

# valence electrons per 2-atom cell -> occupied bands (spin-degenerate)
NOCC = {"si": 4, "c": 4, "sic": 4, "alp": 4, "mgo": 8}

BLOCK = re.compile(
    r"kpoint\s*:\s*\[(?P<k>.*?)\].*?"
    r"KS_gap\s*:\s*(?P<ks>[-\d.]+).*?"
    r"QP_gap\s*:\s*(?P<qp>[-\d.]+).*?"
    r"SigmaeeData \|\n(?P<tab>.*?)(?=\n\s*\n)", re.S)

TAG = re.compile(r"^(?P<mat>[a-z]+)_k(?P<k>\d+)_n(?P<nb>\d+)_e(?P<ec>[\d.]+)$")


def parse_abo(path, nocc):
    """Return (direct_at_gamma, indirect, vbm_k, cbm_k, ks_indirect, nk)."""
    txt = open(path, errors="replace").read()
    vbm = ksv = -1e9
    cbm = ksc = 1e9
    kv = kc = None
    gamma_qp = gamma_ks = None
    nk = 0
    for m in BLOCK.finditer(txt):
        rows = {}
        for line in m.group("tab").strip().split("\n"):
            f = line.split()
            if len(f) >= 10 and f[0].isdigit():
                rows[int(f[0])] = (float(f[1]), float(f[9]))   # E0(KS), E(QP)
        if nocc not in rows or nocc + 1 not in rows:
            continue
        nk += 1
        kstr = " ".join(x.strip() for x in m.group("k").split(",") if x.strip())
        e0v, ev = rows[nocc]
        e0c, ec = rows[nocc + 1]
        if ev > vbm:
            vbm, kv = ev, kstr
        if ec < cbm:
            cbm, kc = ec, kstr
        ksv = max(ksv, e0v)
        ksc = min(ksc, e0c)
        if all(abs(float(x)) < 1e-8 for x in kstr.split()):
            gamma_qp, gamma_ks = float(m.group("qp")), float(m.group("ks"))
    if nk == 0:
        return None
    return dict(direct_gamma_qp=gamma_qp, direct_gamma_ks=gamma_ks,
                indirect_qp=cbm - vbm, indirect_ks=ksc - ksv,
                vbm_k=kv, cbm_k=kc, nk=nk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding <mat>_k*_n*_e* run dirs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    out = []
    for d in sorted(os.listdir(args.root)):
        m = TAG.match(d)
        if not m:
            continue
        abo = os.path.join(args.root, d, "gw", "gw.abo")
        if not os.path.exists(abo):
            continue
        mat = m.group("mat")
        if mat not in NOCC:
            print("# skip %s: no nocc for '%s'" % (d, mat), file=sys.stderr)
            continue
        r = parse_abo(abo, NOCC[mat])
        if r is None:
            print("# skip %s: no complete sigma block" % d, file=sys.stderr)
            continue
        r.update(material=mat, ngkpt=int(m.group("k")),
                 nband=int(m.group("nb")), ecut=float(m.group("ec")), run=d)
        out.append(r)

    if not out:
        print("no results found under %s" % args.root, file=sys.stderr)
        return 1

    cols = ["material", "ngkpt", "nband", "ecut", "nk",
            "indirect_ks", "indirect_qp", "direct_gamma_ks", "direct_gamma_qp",
            "vbm_k", "cbm_k", "run"]
    print("%-5s %-5s %-6s %-7s %-4s %10s %10s %10s %10s"
          % ("mat", "ngkpt", "nband", "ecut", "nk",
             "KS_ind", "QP_ind", "KS_dir(G)", "QP_dir(G)"))
    for r in sorted(out, key=lambda x: (x["material"], x["ngkpt"], x["nband"], x["ecut"])):
        print("%-5s %-5d %-6d %-7.1f %-4d %10.3f %10.3f %10s %10s"
              % (r["material"], r["ngkpt"], r["nband"], r["ecut"], r["nk"],
                 r["indirect_ks"], r["indirect_qp"],
                 "%.3f" % r["direct_gamma_ks"] if r["direct_gamma_ks"] is not None else "-",
                 "%.3f" % r["direct_gamma_qp"] if r["direct_gamma_qp"] is not None else "-"))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in out:
                w.writerow({c: r.get(c) for c in cols})
        print("\nwrote %s (%d rows)" % (args.csv, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
