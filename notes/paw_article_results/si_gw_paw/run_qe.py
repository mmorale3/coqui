#!/usr/bin/env python3
"""QE driver for the Si PAW ladder: cutoff convergence and SCF for each rung.

The ladder rungs differ substantially in hardness (D3's norm-matched r_c(3s)=1.555 is
much harder than D0's 1.90), so a single shared cutoff cannot be assumed -- each dataset
gets its own convergence curve and the RPA comparison is then run at a cutoff converged
for ALL of them, otherwise a basis-set error would masquerade as the PAW completeness
error we are trying to measure.

Usage:
    run_qe.py cutoff [rungs...]     # total energy vs ecutwfc
    run_qe.py scf <ecut> [rungs...] # SCF at a chosen cutoff
"""

import os
import re
import subprocess
import sys

QE_BIN = os.path.expanduser("~/Software/qe-7.4.1/build/cpu/bin")
LADDER = os.path.expanduser("~/Projects/PAW_GW/si_gw_paw")
WORK = os.path.expanduser("~/Projects/PAW_GW/si_gw_paw/_solid")
ALAT = 10.26          # bohr, Si diamond equilibrium
UPF = "Si.GGA-PBE-paw.UPF"

SCF_TEMPLATE = """&CONTROL
  calculation = 'scf'
  prefix = 'si'
  outdir = './out'
  pseudo_dir = '{pseudo_dir}'
  tprnfor = .true.
/
&SYSTEM
  ibrav = 2
  celldm(1) = {alat}
  nat = 2
  ntyp = 1
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'fixed'
  input_dft = 'PBE'
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = {beta}
/
ATOMIC_SPECIES
  Si 28.0855 {upf}
ATOMIC_POSITIONS crystal
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  {nk} {nk} {nk} 0 0 0
"""


def run_scf(rung, ecutwfc, nk=4, tag=None, rho_factor=8, beta=0.4):
    """Run one SCF; return (total_energy_Ry, ok, logpath)."""
    tag = tag or f"{rung}_e{ecutwfc:g}_r{rho_factor}_b{beta}_k{nk}"
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    inp = SCF_TEMPLATE.format(pseudo_dir=os.path.join(LADDER, rung), alat=ALAT,
                              ecutwfc=ecutwfc, ecutrho=rho_factor * ecutwfc,
                              upf=UPF, nk=nk, beta=beta)
    with open(os.path.join(d, "scf.in"), "w") as f:
        f.write(inp)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(os.path.join(d, "scf.out"), "w") as fout:
        subprocess.run([os.path.join(QE_BIN, "pw.x"), "-in", "scf.in"],
                       stdout=fout, stderr=subprocess.STDOUT, cwd=d, env=env)
    log = os.path.join(d, "scf.out")
    txt = open(log, errors="replace").read()
    m = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", txt, re.M)
    return (float(m[-1]) if m else None), bool(m), log


def cutoff_scan(rungs):
    cuts = [40, 60, 80, 100, 120]
    print(f"Total energy (Ry) vs ecutwfc, Si diamond a={ALAT} bohr, 4x4x4 k, ecutrho=8x")
    print(f"{'rung':>5} " + " ".join(f"{c:>13d}" for c in cuts) + "   dE(last-1) mRy/atom")
    for rung in rungs:
        row, energies = [], []
        for c in cuts:
            e, ok, log = run_scf(rung, c)
            row.append(f"{e:13.6f}" if ok else "       FAILED")
            energies.append(e if ok else None)
        conv = ""
        if energies[-1] is not None and energies[-2] is not None:
            conv = f"   {abs(energies[-1] - energies[-2]) * 1000 / 2:8.3f}"
        print(f"{rung:>5} " + " ".join(row) + conv)


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "cutoff"
    rungs = sys.argv[2:] or ["D0", "D1", "D2", "D3"]
    if mode == "cutoff":
        cutoff_scan(rungs)
    elif mode == "scf":
        ecut = float(rungs[0])
        for rung in (rungs[1:] or ["D0", "D1", "D2", "D3"]):
            e, ok, log = run_scf(rung, ecut)
            print(f"{rung}: {'E=%.6f Ry' % e if ok else 'FAILED ' + log}")
    else:
        raise SystemExit(__doc__)
