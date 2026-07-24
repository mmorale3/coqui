#!/usr/bin/env python3
"""Isolation test for the kjpaw PAW PBE-Si large-a0 anomaly (run on rusty).

Goal: determine whether THIS q-e 7.0 build gives the known-good a0~10.33 for kjpaw
under 'vanilla' settings, vs the campaign settings (force_symmorphic + ecut100/600)
which gave ~10.61. Varies ONE thing at a time:
  - 'nosym'  : drop force_symmorphic (the one unusual campaign flag)
  - 'stdcut' : standard textbook cutoffs (ecutwfc=40, ecutrho=320)
Both at bracket volumes 10.20/10.35/10.50/10.65 (distinguishes min~10.33 from min~10.6).

Usage: python3 setup_kjpaw_isolation.py ; sbatch ~/ceph/CoQui/Si_eos_rpa_hf/diag/run_kjpawiso.sbatch
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag"); PSE = os.path.join(ROOT, "pseudos", "paw")
VOLS = ["10.20", "10.35", "10.50", "10.65"]

def scf_in(a, force_sym, ecutwfc, ecutrho):
    fs = "    force_symmorphic = .true.\n" if force_sym else ""
    return f"""\
&control
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './out'
    pseudo_dir  = '{PSE}'
    verbosity   = 'high'
/
&system
    ibrav       = 2
{fs}    celldm(1)   = {a}
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    input_dft   = 'pbe'
    occupations = 'fixed'
/
&electrons
    diagonalization = 'david'
    conv_thr        = 1.0d-10
    mixing_beta     = 0.4
/
ATOMIC_SPECIES
Si  28.085  Si.UPF
ATOMIC_POSITIONS alat
Si  0.000  0.000  0.000
Si  0.250  0.250  0.250
K_POINTS automatic
8 8 8 0 0 0
"""

VARIANTS = {
    "nosym":  dict(force_sym=False, ecutwfc=100, ecutrho=600),
    "stdcut": dict(force_sym=False, ecutwfc=40,  ecutrho=320),
}
cmds = []
for tag, cfg in VARIANTS.items():
    for a in VOLS:
        d = os.path.join(DIAG, "kjpaw_iso", tag, f"a{a.replace('.','p')}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scf.in"), "w") as f:
            f.write(scf_in(a, **cfg))
        cmds.append(d)

body = "\n".join(
    f'cd {d}\nif [ ! -f scf.out.done ]; then mpirun -np 16 $QE_BIN/pw.x -in scf.in > scf.out && touch scf.out.done; fi'
    for d in cmds)
sb = "\n".join([
    "#!/bin/bash", "#SBATCH -J kjpawiso", "#SBATCH -p ccq", "#SBATCH -N 1",
    "#SBATCH --exclusive", "#SBATCH -c 1", "#SBATCH -t 00:30:00",
    f"#SBATCH -o {DIAG}/kjpawiso.%j.out", f"#SBATCH -e {DIAG}/kjpawiso.%j.err",
    "set -e", "source /etc/profile.d/modules.sh",
    "module purge; module load modules",
    "module load gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3",
    "QE_BIN=/mnt/home/mmorales/Devel/QEF/q-e_7.0/build/CPU/bin",
    "export OMP_NUM_THREADS=1",
    'echo "start $(date)"', body, 'echo "done $(date)"', ""])
with open(os.path.join(DIAG, "run_kjpawiso.sbatch"), "w") as f:
    f.write(sb)
print("wrote run_kjpawiso.sbatch;", len(cmds), "runs:", list(VARIANTS), VOLS)
