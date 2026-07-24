#!/usr/bin/env python3
"""Pure-QE PBE EOS test for candidate replacement PAW Si datasets (run on rusty).
Confirms whether a candidate gives a sane PBE a0~10.33 (kjpaw psl 1.0.0 gives 10.61).
Usage: python3 setup_paw_test.py ; then sbatch ~/ceph/CoQui/Si_eos_rpa_hf/diag/run_pawtest.sbatch
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag")
PSE  = os.path.join(ROOT, "pseudos")
CANDIDATES = ["jth_with_d_v2"]          # l_max=2, d-complete PBE PAW
VOLS = ["10.04", "10.15", "10.26", "10.37", "10.48"]

PBE = """\
&control
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './out'
    pseudo_dir  = '{pseudo}'
    verbosity   = 'high'
    wf_collect  = .true.
/
&system
    ibrav            = 2
    force_symmorphic = .true.
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
for cand in CANDIDATES:
    for a in VOLS:
        d = os.path.join(DIAG, "paw_test", cand, f"a{a.replace('.','p')}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scf.in"), "w") as f:
            f.write(PBE.format(pseudo=os.path.join(PSE, cand), a=a))
        cmds.append(d)

body = "\n".join(
    f'cd {d}\nif [ ! -f scf.out.done ]; then mpirun -np 16 $QE_BIN/pw.x -in scf.in > scf.out && touch scf.out.done; fi'
    for d in cmds)
sb = "\n".join([
    "#!/bin/bash", "#SBATCH -J pawtest", "#SBATCH -p ccq", "#SBATCH -N 1",
    "#SBATCH --exclusive", "#SBATCH -c 1", "#SBATCH -t 00:30:00",
    f"#SBATCH -o {DIAG}/pawtest.%j.out", f"#SBATCH -e {DIAG}/pawtest.%j.err",
    "set -e", "source /etc/profile.d/modules.sh",
    "module purge; module load modules",
    "module load gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3",
    "QE_BIN=/mnt/home/mmorales/Devel/QEF/q-e_7.0/build/CPU/bin",
    "export OMP_NUM_THREADS=1",
    'echo "start $(date)"', body, 'echo "done $(date)"', ""])
with open(os.path.join(DIAG, "run_pawtest.sbatch"), "w") as f:
    f.write(sb)
print("wrote run_pawtest.sbatch;", len(cmds), "PBE runs:", CANDIDATES, VOLS)
