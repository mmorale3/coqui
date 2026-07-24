#!/usr/bin/env python3
"""QE self-consistent HF (input_dft='hf') on the CLEAN qe-7.5 for paw/uspp/oncv across
5 volumes — to get the trustworthy QE-direct-HF lattice constant for Si (USPP & PAW),
and test whether QE's own augmented EXX shows the +0.4 Bohr anomaly (CoQui-independent).
Same settings as the campaign hf_kp8 (ecut65, k8, nqx8, gygi) for apples-to-apples.
Single-node (N=1) to avoid the inter-node IB fabric crashes. Run on rusty.
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag"); PSE = os.path.join(ROOT, "pseudos")
PWX = "/mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/bin/pw.x"
VOLS = ["10.04", "10.15", "10.26", "10.37", "10.48"]

def hf(a, pp):
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
    ecutwfc     = 65
    ecutrho     = 260
    input_dft   = 'hf'
    nqx1        = 8
    nqx2        = 8
    nqx3        = 8
    exxdiv_treatment = 'gygi-baldereschi'
    occupations = 'fixed'
/
&electrons
    diagonalization = 'david'
    conv_thr        = 1.0d-8
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

paths = []
for pp in ("paw", "uspp", "oncv"):
    for a in VOLS:
        d = os.path.join(DIAG, "hf_qe75", pp, f"a{a.replace('.','p')}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "hf.in"), "w").write(hf(a, pp))
        tag = f"hf75_{pp}_a{a.replace('.','p')}"
        sb = "\n".join([
            "#!/bin/bash", f"#SBATCH -J {tag}", "#SBATCH -p ccq", "#SBATCH -N 1",
            "#SBATCH --exclusive", "#SBATCH -c 1", "#SBATCH -t 02:00:00",
            f"#SBATCH -o {d}/hf.%j.out", f"#SBATCH -e {d}/hf.%j.err",
            "set -e", "source /etc/profile.d/modules.sh", "module purge", "module load modules",
            "module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3",
            "export OMP_NUM_THREADS=1", f"cd {d}",
            'echo "node=$(hostname) start=$(date)"',
            f"if [ ! -f hf.out.done ]; then mpirun -np 32 {PWX} -in hf.in > hf.out && touch hf.out.done; fi",
            'echo "end=$(date)"', ""])
        p = os.path.join(d, "run.sbatch")
        open(p, "w").write(sb); paths.append(p)
print(f"wrote {len(paths)} HF run dirs under {DIAG}/hf_qe75")
for p in paths: print(p)
