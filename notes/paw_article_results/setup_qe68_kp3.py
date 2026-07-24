#!/usr/bin/env python3
"""QE 6.8 PBE at kp3 (matching the campaign scf) for USPP+ONCV across the 7 RPA
volumes, to get a CLEAN qe_1e (one-electron) and re-derive formula A. Tests whether
the formula-A USPP residual (10.62) is also a q-e 7.4.1 build artifact via qe_1e.
Run on rusty: python3 setup_qe68_kp3.py ; sbatch ~/.../diag/run_qe68kp3.sbatch
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag"); PSE = os.path.join(ROOT, "pseudos")
QE68 = "/mnt/home/mmorales/Devel/QEF/q-e/build/CPU/bin/pw.x"
VOLS = ["10.04","10.15","10.26","10.37","10.48","10.55","10.65"]

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
    ibrav            = 2
    force_symmorphic = .true.
    celldm(1)   = {a}
    nat         = 2
    ntyp        = 1
    ecutwfc     = 100
    ecutrho     = 400
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
3 3 3 0 0 0
"""
cmds=[]
for pp in ("uspp","oncv","paw"):
    for a in VOLS:
        d=os.path.join(DIAG,"qe68_kp3",pp,f"a{a.replace('.','p')}")
        os.makedirs(d,exist_ok=True)
        open(os.path.join(d,"scf.in"),"w").write(scf(a,pp))
        cmds.append(d)
body="\n".join(f'cd {d}\nif [ ! -f scf.out.done ]; then mpirun -np 16 {QE68} -in scf.in > scf.out && touch scf.out.done; fi' for d in cmds)
sb="\n".join(["#!/bin/bash","#SBATCH -J qe68kp3","#SBATCH -p ccq","#SBATCH -N 1",
  "#SBATCH --exclusive","#SBATCH -c 1","#SBATCH -t 00:40:00",
  f"#SBATCH -o {DIAG}/qe68kp3.%j.out",f"#SBATCH -e {DIAG}/qe68kp3.%j.err",
  "set -e","source /etc/profile.d/modules.sh","module purge","module load modules",
  "module load gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3",
  "export OMP_NUM_THREADS=1",'echo "start $(date)"',body,'echo "done $(date)"',""])
open(os.path.join(DIAG,"run_qe68kp3.sbatch"),"w").write(sb)
print("wrote run_qe68kp3.sbatch;",len(cmds),"runs (uspp+oncv+paw x 7 vols, kp3, QE6.8)")
