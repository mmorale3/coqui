#!/usr/bin/env python3
"""Generate the instrumented-ABINIT term-by-term reference series for the Si
EOS exchange defect (2026-07-29).

For each lattice constant, two datasets on the SAME orbitals:

  DS1  ixc 11 (PBE), converged SCF (tolvrs 1e-10).
       This is where E_1e and E_H come from.  It is a normal variational SCF,
       so its energy decomposition is trustworthy: kinetic, local_psp,
       psp_core, hartree, and -- via the ABI_DUMP_PAWENE dump -- the
       individual on-site pieces e1t10 (= sum_ij rho_ij dij0) and eh2
       (one-centre Hartree).

  DS2  ixc 40 (pure Fock, alpha = 1), nstep 1, getwfk/getden from DS1.
       This is where E_x comes from, but NOT via the printed total: the DC
       total goes through e_eigenvalues from one partially converged
       diagonalization and is noisy at the 100 mHa level across a volume
       series.  We read e_fock0 (smooth plane-wave Fock built from the INPUT
       = PBE orbitals) and efock (one-centre Fock, vv + cv) out of the dumps
       instead.  Those are evaluated on the PBE orbitals by construction.

  fock_icutcoul 3 = bare Coulomb with the q+G=0 term simply omitted, which is
  the convention CoQui calls div_treatment = ignore_g0.  The finite-size
  correction is added back analytically on both sides, so it must be OFF here
  (see notes/paw_article_results/eos_exchange_ledger.md).

  nsym 1 / kptopt 3 = no symmetry, full BZ -- matches the CoQui mf.

Usage:  python3 gen_eos_ledger.py [--submit]
"""
import argparse
import os
import subprocess

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
BIN = ROOT + "/bin/abinit_ene"          # instrumented (abinit_ene_instr.py)
AVALS = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]

# (tag, work subdir, pseudo, extra dataset-1 lines)
CASES = [
    ("jthd", "eos_ledger_jthd", '"jth_with_d/Si.xml"', "pawecutdg 50.0"),
    # NC control: CoQui already reproduces the reference EOS with this pseudo
    # (a0 = 10.2259, B0 = 101.1 GPa), so agreement here validates the LEDGER
    # itself before it is used to indict the PAW path.
    ("nc",   "eos_ledger_nc",   '"Si_GGA_noNLCC.psp8"', ""),
]

ABI = """acell 3*{a}
rprim 0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0
ntypat 1 znucl 14 natom 2 typat 1 1
xred 0.0 0.0 0.0  0.25 0.25 0.25
ecut 25.0
{extra}
ngkpt 4 4 4
nshiftk 1
shiftk 0.0 0.0 0.0
kptopt 3
nsym 1
istwfk *1
iomode 3
occopt 1
nband 12
prtvol 3
pawprtvol 3
fock_icutcoul 3
pp_dirpath "{root}/pseudos"
pseudos {pseudo}
ndtset 2

# --- DS1: converged PBE. Source of E_1e and E_H.
ixc1 11
nstep1 100
tolvrs1 1.0d-10
prtden1 1
prtwf1 1

# --- DS2: one-shot pure Fock on the DS1 orbitals. Source of E_x.
ixc2 40
getwfk2 -1
getden2 1
nstep2 1
tolvrs2 1.0d-8
fockdownsampling2 1 1 1
"""

SBATCH = """#!/bin/bash
#SBATCH -J led_{tag}_{a} -p ccq -N 1 -n 32 -c 1 -t 01:00:00 --mem=0
#SBATCH -o {wd}/run.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3 libxc netcdf-c netcdf-fortran
export OMP_NUM_THREADS=1
export ABI_DUMP_ENE=1
export ABI_DUMP_PAWENE=1
cd {wd}
mpirun -np 32 {bin} si.abi > si.log 2>&1
grep -q "Calculation completed" si.log && echo ok > run.done || echo fail > run.done
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    script = ["set -e"]
    for tag, sub, pseudo, extra in CASES:
        for a in AVALS:
            wd = "%s/%s/a%s" % (ROOT, sub, a)
            abi = ABI.format(a=a, extra=extra, root=ROOT, pseudo=pseudo)
            sb = SBATCH.format(tag=tag, a=a, wd=wd, bin=BIN)
            script.append("mkdir -p %s" % wd)
            script.append("cat > %s/si.abi <<'EOF_ABI'\n%s\nEOF_ABI" % (wd, abi))
            script.append("cat > %s/run.sbatch <<'EOF_SB'\n%s\nEOF_SB" % (wd, sb))
            if args.submit:
                script.append("cd %s && sbatch run.sbatch" % wd)
    body = "\n".join(script)
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input=body, text=True, capture_output=True)
    print(p.stdout.strip())
    if p.returncode:
        print("STDERR:", p.stderr.strip())
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
