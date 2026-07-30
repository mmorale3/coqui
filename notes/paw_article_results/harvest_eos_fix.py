#!/usr/bin/env python3
"""Harvest + refit the corrected (sqrt(4pi)-fixed converter) Si PAW EXX+RPA EOS.

    python3 harvest_eos_fix.py

Reads `~/ceph/CoQui/abinit/eos_jthd_coqui_fix/a*/rpa.out` and:

 1. checks that E_H, E_x and E_c came back UNCHANGED from the pre-fix series.
    The converter fix touches pp_local / dion / vloc_ps, i.e. H0 only, so ONLY
    e_1e may move. If E_H / E_x / E_c shifted, something other than H0 changed
    and the comparison against the pre-fix run is invalid.
 2. refits the Birch-Murnaghan EOS and compares against the reference and
    against the prediction registered before the runs landed.

PRE-REGISTERED PREDICTION (2026-07-29, from the instrumented-ABINIT ledger, i.e.
independent of these runs): a0 = 10.2501 Bohr, B0 = 101.1 GPa, B' = 4.18.
Derived as ABINIT's one-body + Hartree + bare exchange, plus CoQui's own (correct)
gygi divergence term, plus CoQui's E_c and Ewald -- see eos_exchange_ledger.md.
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eos_exxrpa_fit import EWALD_HA, birch_murnaghan, check_fittable, fit  # noqa: E402

AVALS = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]
ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"

# pre-fix series (eos_exxrpa), for the "only e_1e may move" check
PRE = {  # a : (e_1e, e_hf, e_x, e_c)
    10.05: (2.4626358263610277, -1.5959474265071731, -2.157602014860545, -0.437371323477737),
    10.15: (2.3440212353628334, -1.5684701446311209, -2.1421368277685087, -0.4363534870915905),
    10.25: (2.2282740018203797, -1.5411630647079502, -2.1269018471245915, -0.4354712185779852),
    10.35: (2.115258175567617, -1.5140248850294997, -2.1119012576722938, -0.4347234568158686),
    10.45: (2.0048419773919157, -1.4870497567776728, -2.0971320147200267, -0.4341065979722086),
    10.55: (1.8969043517275541, -1.460223760773173, -2.0825950760153575, -0.43361653862763594),
}
PREDICTED = dict(a0=10.2501, B0=101.1, Bp=4.18)


def fetch(sub):
    body = "\n".join(
        ['for a in %s; do' % " ".join(AVALS),
         '  f=%s/%s/a$a/rpa.out' % (ROOT, sub),
         '  [ -f $f ] || { echo "$a MISSING"; continue; }',
         '  e1=$(grep -m1 "^One-electron energy:" $f | awk "{print \\$3}")',
         '  eh=$(grep -m1 "^Hartree-Fock energy:" $f | awk "{print \\$3}")',
         '  ex=$(grep -m1 "^Exchange energy:" $f | awk "{print \\$3}")',
         '  ec=$(grep -m1 "^RPA energy:" $f | awk "{print \\$3}")',
         '  echo "$a $e1 $eh $ex $ec"',
         'done'])
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input=body, text=True, capture_output=True)
    if p.returncode:
        sys.exit("ssh failed:\n" + p.stderr[-1500:])
    out = {}
    for line in p.stdout.split("\n"):
        f = line.split()
        if len(f) == 5:
            try:
                out[float(f[0])] = tuple(float(x) for x in f[1:])
            except ValueError:
                pass
        elif f:
            print("  ** a=%s: %s" % (f[0], " ".join(f[1:])))
    return out


def main():
    post = fetch("eos_jthd_coqui_fix")
    if not post:
        sys.exit("no completed runs in eos_jthd_coqui_fix yet")
    ks = sorted(post)

    print("%7s %14s %14s %13s %13s %13s" %
          ("a", "e_1e(post)", "e_1e shift", "E_H", "E_x", "E_c"))
    rows, drift_ok = [], True
    for x in ks:
        e1, ehf, ex, ec = post[x]
        eh = ehf - ex
        rows.append((x, e1, eh, ex, ec))
        if x in PRE:
            p1, phf, px, pc = PRE[x]
            peh = phf - px
            d = max(abs(eh - peh), abs(ex - px), abs(ec - pc))
            flag = "" if d < 5e-6 else "  <-- E_H/E_x/E_c MOVED by %.2e" % d
            if d >= 5e-6:
                drift_ok = False
            print("%7.2f %14.7f %14.7f %13.7f %13.7f %13.7f%s"
                  % (x, e1, e1 - p1, eh, ex, ec, flag))
        else:
            print("%7.2f %14.7f %14s %13.7f %13.7f %13.7f" % (x, e1, "-", eh, ex, ec))

    print("\n'only e_1e may move' check: %s" %
          ("PASS -- E_H / E_x / E_c unchanged to <5 uHa" if drift_ok else
           "FAIL -- see flags above; the fix should touch H0 only"))

    aa = np.array([r[0] for r in rows])
    tot = np.array([r[1] + r[2] + r[3] + r[4] for r in rows])   # e_1e + E_H + E_x + E_c
    EW = np.array([EWALD_HA[x] for x in aa])
    E = tot + EW
    print("\n%7s %16s %14s %16s" % ("a", "CoQui total", "E_Ewald", "E_total"))
    for i, x in enumerate(aa):
        print("%7.2f %16.8f %14.8f %16.8f" % (x, tot[i], EW[i], E[i]))

    bad = check_fittable(aa, E)
    if bad:
        print("\nNOT FITTED:")
        for b in bad:
            print("  * %s" % b)
        return 1
    r = fit(aa, E)
    print("\nBirch-Murnaghan (%d points, %d dof):" % (len(aa), r["dof"]))
    if r["extrapolated"]:
        print("  *** a0 = %.4f is OUTSIDE the sampled range -- extrapolation ***"
              % r["a0_bohr"])
    print("  a0 = %.4f Bohr   B0 = %.1f GPa   B' = %.2f   max resid = %.3f mHa"
          % (r["a0_bohr"], r["B0_GPa"], r["Bp"], r["max_resid_Ha"] * 1e3))
    print("\n  pre-registered prediction   a0 = %.4f  B0 = %.1f  B' = %.2f"
          % (PREDICTED["a0"], PREDICTED["B0"], PREDICTED["Bp"]))
    print("  CoQui NC, same pipeline     a0 = 10.2259  B0 = 101.1  B' = 4.08")
    print("  VASP/PAW RPA@PBE (Harl'10)  a0 = 10.244   B0 = 98")
    print("  pre-fix PAW                 no minimum in 10.05..10.55")
    print("\n  delta vs prediction: a0 %+.4f Bohr, B0 %+.1f GPa"
          % (r["a0_bohr"] - PREDICTED["a0"], r["B0_GPa"] - PREDICTED["B0"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
