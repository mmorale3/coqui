#!/usr/bin/env python3
"""Run ON rusty.  Round 2 of the CoQui RPA blow-up localization, a=10.05, PAW jth_with_d.

Round 1 established: the one-center term (~2 mHa) and THC/ISDF compression (~4 mHa)
do NOT carry the 153.5 mHa n=250->500 increment, and that WITHOUT augmentation the
increment is 5x worse (-821.3 mHa).  So the smooth (PS) pair densities carry a huge
band-runaway and the augmentation supplies ~668 mHa of cancellation against it --
incompletely.  ABINIT, on the identical mean field, cancels it correctly (-6.0 mHa).

Round 2 splits that cancellation.  The THC ERI is smooth + V_GL/V_LG + V_LL:
  shape       vv_compensation='shape'  -> full AE-PS pair density instead of the
                                          moment-restored compensation charge.
                                          Changes the SHARED ERI (thc_reader_t:2129
                                          comment), so it reaches correlation. FIX CANDIDATE.
  vgl_off     paw_vgl=false            -> drop the smooth<->aug cross block
  vll_off     paw_vll=false            -> drop the aug<->aug block
The last two are DIAGNOSTIC, not physical: whichever block's removal collapses the
n=250->500 increment is the block whose band-scaling is broken.
"""
import os

ROOT  = "/mnt/home/mmorales/ceph/CoQui/abinit"
SRC   = f"{ROOT}/eos_jthd_coqui"
WORK  = f"{ROOT}/rpa_localize2_jthd"
COQUI = "/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui/build/CPU"

ALAT   = "10.05"
NBANDS = [250, 500]

# name -> extra [interaction.thc] lines
VARIANTS = {
    "shape":   'vv_compensation = "shape"',
    "vgl_off": "paw_vgl = false",
    "vll_off": "paw_vll = false",
}


def toml(nb, extra, tag):
    return f"""[mean_field.bdft]
name = "mf"
prefix = "mf"
outdir = "{SRC}/a{ALAT}"
nbnd = {nb}
[interaction.thc]
name = "eri"
mean_field = "mf"
storage = "incore"
thresh = 1e-4
chol_block_size = 8
paw_aug = true
paw_onsite = true
paw_isdf_tol = 5e-5
paw_isdf_metric = "coulomb"
{extra}
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


def sbatch(d, tag):
    return f"""#!/bin/bash
#SBATCH -J {tag} -p ccq -N 8 -c 1 -n 128 -t 12:00:00 --mem=0
#SBATCH -o {d}/rpa.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3
export root={COQUI}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root/lib64
export OMP_NUM_THREADS=1
export UCX_TLS=ud,sm,self
cd {d}
rm -f rpa.done thc.eri.h5 thc_eri.h5
timeout 42000 mpirun -np 128 $root/bin/coqui --filenames rpa.toml > rpa.out 2>&1
echo "exit=$?" > rpa.done
"""


cmds = []
for name, extra in VARIANTS.items():
    for nb in NBANDS:
        d = f"{WORK}/{name}_n{nb}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/rpa.toml", "w").write(toml(nb, extra, name))
        open(f"{d}/run.sbatch", "w").write(sbatch(d, f"m2{name[:5]}{nb}"))
        cmds.append(f"sbatch {d}/run.sbatch")

print(f"wrote {len(cmds)} run dirs under {WORK}  (a={ALAT})")
for c in cmds:
    print(c)
