#!/usr/bin/env python3
"""
Set up two diagnostic QE run-sets on rusty (run THIS on rusty):

  1. kjpaw PAW PBE convergence/anomaly check: pure QE-PBE at high ecutrho=600,
     8x8x8 k-mesh, extended volumes (a up to 10.9) to bracket the minimum and
     rule out cutoff/k-mesh. Confirms or refutes the a0=10.61 finding.

  2. EXX@PBE recipe validation on ONCV a10p26: single-shot exact-exchange on the
     converged PBE orbitals (restart), at matched ecut100/400, kp3, nqx=3,
     gygi-baldereschi. Validation target (NCPP, must reproduce CoQui e_1e+e_hf):
          total energy = -15.1950 Ry   (= 0.80298 Ha * 2 + Ewald -16.80093)
     Once validated, the same recipe is applied to USPP (good PBE PP).

Usage on rusty:  python3 setup_eos_diag.py
Then:            sbatch ~/ceph/CoQui/Si_eos_rpa_hf/diag/run_diag.sbatch
"""
import os, shutil, textwrap

ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag")
PSE  = os.path.join(ROOT, "pseudos")

PBE_HEAD = """\
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

EXX_HEAD = """\
&control
    calculation  = 'scf'
    prefix       = 'si'
    outdir       = './out'
    pseudo_dir   = '{pseudo}'
    restart_mode = 'from_scratch'
    verbosity    = 'high'
    wf_collect   = .true.
/
&system
    ibrav            = 2
    force_symmorphic = .true.
    celldm(1)   = {a}
    nat         = 2
    ntyp        = 1
    ecutwfc     = 100
    ecutrho     = 400
    input_dft   = 'hf'
    nqx1 = 3
    nqx2 = 3
    nqx3 = 3
    exxdiv_treatment = 'gygi-baldereschi'
    occupations = 'fixed'
/
&electrons
    diagonalization = 'david'
    conv_thr        = 1.0d-8
    electron_maxstep = 1
    startingwfc     = 'file'
    startingpot     = 'file'
    adaptive_thr    = .false.
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

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

cmds = []  # (workdir, infile, outfile)

# ---- 1. kjpaw PBE convergence/anomaly ----
for a in ("10.30", "10.50", "10.70", "10.90"):
    d = os.path.join(DIAG, "kjpaw_pbe", f"a{a.replace('.','p')}")
    write(os.path.join(d, "scf.in"),
          PBE_HEAD.format(pseudo=os.path.join(PSE, "paw"), a=a, ecutrho=600))
    cmds.append((d, "scf.in", "scf.out"))

# ---- 2. EXX@PBE recipe validation on ONCV a10p26 (restart from PBE scf) ----
src_out = os.path.join(ROOT, "runs", "a10p26", "oncv", "rpa_kp3_n250", "scf", "out")
d = os.path.join(DIAG, "exx_validate", "a10p26_oncv")
write(os.path.join(d, "exx.in"),
      EXX_HEAD.format(pseudo=os.path.join(PSE, "oncv"), a="10.26"))
dst_out = os.path.join(d, "out")
if os.path.isdir(src_out) and not os.path.isdir(dst_out):
    shutil.copytree(src_out, dst_out)
    print(f"copied PBE save -> {dst_out}")
cmds.append((d, "exx.in", "exx.out"))

# ---- sbatch driver ----
body = "\n".join(
    f'cd {wd}\nif [ ! -f {out}.done ]; then mpirun -np 16 $QE_BIN/pw.x -in {inf} > {out} && touch {out}.done; fi'
    for wd, inf, out in cmds)
sb = "\n".join([
    "#!/bin/bash",
    "#SBATCH -J eos_diag",
    "#SBATCH -p ccq",
    "#SBATCH -N 1",
    "#SBATCH --exclusive",
    "#SBATCH -c 1",
    "#SBATCH -t 00:30:00",
    f"#SBATCH -o {DIAG}/run.%j.out",
    f"#SBATCH -e {DIAG}/run.%j.err",
    "set -e",
    "source /etc/profile.d/modules.sh",
    "module purge; module load modules",
    "module load gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3",
    "QE_BIN=/mnt/home/mmorales/Devel/QEF/q-e_7.0/build/CPU/bin",
    "export OMP_NUM_THREADS=1",
    'echo "=== start $(date) on $(hostname) ==="',
    body,
    'echo "=== done $(date) ==="',
    ""])
write(os.path.join(DIAG, "run_diag.sbatch"), sb)
print("wrote", os.path.join(DIAG, "run_diag.sbatch"))
print("run-set:")
for wd, inf, out in cmds:
    print("  ", wd, inf)
