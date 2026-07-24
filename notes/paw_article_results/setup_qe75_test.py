#!/usr/bin/env python3
"""Retest kjpaw + ONCV Si PBE EOS on the freshly-built QE 7.5 (clone of qe-7.5 tag).
Matched settings to the QE-6.8/7.4.1 comparison: ecut100/600, k888, no force_symmorphic,
vols 10.20/10.35/10.50/10.65. QE 6.8 gave kjpaw a0=10.335; QE 7.4.1 gave min>10.65.
Run on rusty: python3 setup_qe75_test.py  (emits run_qe75test.sbatch)
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag"); PSE = os.path.join(ROOT, "pseudos")
PWX = "/mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/bin/pw.x"
VOLS = ["10.20", "10.35", "10.50", "10.65"]

def scf(a, pp):
    return f"""\
&control
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './out'
    pseudo_dir  = '{PSE}/{pp}'
    verbosity   = 'high'
/
&system
    ibrav       = 2
    celldm(1)   = {a}
    nat         = 2
    ntyp        = 1
    ecutwfc     = 100
    ecutrho     = 600
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
cmds = []
for pp in ("paw", "oncv"):
    for a in VOLS:
        d = os.path.join(DIAG, "qe75_pbe", pp, f"a{a.replace('.','p')}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "scf.in"), "w").write(scf(a, pp))
        cmds.append(d)
body = "\n".join(f'cd {d}\nif [ ! -f scf.out.done ]; then mpirun -np 16 {PWX} -in scf.in > scf.out && touch scf.out.done; fi' for d in cmds)
sb = "\n".join(["#!/bin/bash", "#SBATCH -J qe75test", "#SBATCH -p ccq", "#SBATCH -N 1",
  "#SBATCH --exclusive", "#SBATCH -c 1", "#SBATCH -t 00:40:00",
  f"#SBATCH -o {DIAG}/qe75test.%j.out", f"#SBATCH -e {DIAG}/qe75test.%j.err",
  "set -e", "source /etc/profile.d/modules.sh", "module purge", "module load modules",
  "module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3",
  "export OMP_NUM_THREADS=1",
  'echo "start $(date)"', body, 'echo "done $(date)"', ""])
open(os.path.join(DIAG, "run_qe75test.sbatch"), "w").write(sb)
print("wrote run_qe75test.sbatch;", len(cmds), "runs (paw+oncv x 4 vols, QE 7.5)")
