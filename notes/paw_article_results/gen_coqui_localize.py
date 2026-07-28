#!/usr/bin/env python3
"""Run ON rusty.  Localize the CoQui-side RPA blow-up at a=10.05 (the most unstable
volume), PAW jth_with_d, reading the existing 500-band mf.h5.

Diagnostic = the n=250 -> n=500 increment in E_c.  Baseline (all terms on, default
compression) is -26.6 mHa then -153.5 mHa.  ABINIT on the identical mean field gives
-21.7 then -6.0 mHa.  Whichever variant collapses the second increment to ~-6 mHa
carries the defect.

Variants:
  aug_off      paw_aug=false                 -> augmentation (Q_ij(G)) path
  onsite_off   paw_onsite=false              -> one-center term
  both_off     both false                    -> smooth-only control
  tight        default terms, 10x tighter THC/ISDF tolerances
                                             -> compression artifact at high bands
"""
import os

ROOT  = "/mnt/home/mmorales/ceph/CoQui/abinit"
SRC   = f"{ROOT}/eos_jthd_coqui"
WORK  = f"{ROOT}/rpa_localize_jthd"
COQUI = "/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui/build/CPU"

ALAT   = "10.05"
NBANDS = [250, 500]

#                 paw_aug, paw_onsite, thresh, paw_isdf_tol, walltime
VARIANTS = {
    "aug_off":    ("false", "true",  "1e-4", "5e-5", "08:00:00"),
    "onsite_off": ("true",  "false", "1e-4", "5e-5", "08:00:00"),
    "both_off":   ("false", "false", "1e-4", "5e-5", "08:00:00"),
    "tight":      ("true",  "true",  "1e-5", "5e-6", "16:00:00"),
}


def toml(nb, aug, onsite, thresh, isdf, tag):
    return f"""[mean_field.bdft]
name = "mf"
prefix = "mf"
outdir = "{SRC}/a{ALAT}"
nbnd = {nb}
[interaction.thc]
name = "eri"
mean_field = "mf"
storage = "incore"
thresh = {thresh}
chol_block_size = 8
paw_aug = {aug}
paw_onsite = {onsite}
paw_isdf_tol = {isdf}
paw_isdf_metric = "coulomb"
[rpa]
interaction = "eri"
interaction_hf = "eri"
beta = 1000
wmax = 12.0
iaft_prec = "high"
output = "rpa_{tag}_n{nb}"
div_treatment = "gygi"
hf_div_treatment = "gygi"
"""


def sbatch(d, tag, wall):
    return f"""#!/bin/bash
#SBATCH -J {tag} -p ccq -N 8 -c 1 -n 128 -t {wall} --mem=0
#SBATCH -o {d}/rpa.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3
export root={COQUI}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root/lib64
export OMP_NUM_THREADS=1
export UCX_TLS=ud,sm,self
cd {d}
rm -f rpa.done thc.eri.h5 thc_eri.h5
timeout 56000 mpirun -np 128 $root/bin/coqui --filenames rpa.toml > rpa.out 2>&1
echo "exit=$?" > rpa.done
"""


cmds = []
for name, (aug, onsite, thresh, isdf, wall) in VARIANTS.items():
    for nb in NBANDS:
        d = f"{WORK}/{name}_n{nb}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/rpa.toml", "w").write(toml(nb, aug, onsite, thresh, isdf, name))
        open(f"{d}/run.sbatch", "w").write(sbatch(d, f"lz{name[:6]}{nb}", wall))
        cmds.append(f"sbatch {d}/run.sbatch")

print(f"wrote {len(cmds)} run dirs under {WORK}  (a={ALAT}, PAW jth_with_d)")
for c in cmds:
    print(c)
