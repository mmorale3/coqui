#!/usr/bin/env python3
"""Run ON rusty.  CoQui THC-RPA band series (nbnd = 100, 250) at each lattice constant,
to sit alongside the existing nbnd = 500 runs in abinit/eos_jthd_coqui and the ABINIT
ACFD-RPA series in abinit/rpa_eos_jthd.

Reads the EXISTING mf.h5 in eos_jthd_coqui/a{a}/ (500 bands) via outdir, and truncates
with the [mean_field.bdft] nbnd key (src/mean_field/mf_utils.hpp:91 ->
bdft_readonly(..., nbnd)).  No reconversion, no extra disk.

All THC/RPA settings are copied verbatim from the n=500 runs so the band series is
internally consistent.
"""
import os

ROOT  = "/mnt/home/mmorales/ceph/CoQui/abinit"
SRC   = f"{ROOT}/eos_jthd_coqui"          # holds mf.h5 per lattice constant
WORK  = f"{ROOT}/rpa_eos_jthd_coqui_nb"
COQUI = "/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui/build/CPU"

ALAT   = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]
NBANDS = [100, 250]
WALL = "08:00:00"
# Rank count must stay below nbnd: CoQui's make_distributed_array aborts when the proc
# grid exceeds the band dimension (seen at N=8 on small problems in the EOS campaign).
RANKS = {100: (4, 64), 250: (8, 128)}   # nbnd -> (nodes, ranks)


def toml(a, nb):
    return f"""[mean_field.bdft]
name = "mf"
prefix = "mf"
outdir = "{SRC}/a{a}"
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
[rpa]
interaction = "eri"
interaction_hf = "eri"
beta = 1000
wmax = 12.0
iaft_prec = "high"
output = "rpa_jthd_{a}_n{nb}"
div_treatment = "gygi"
hf_div_treatment = "gygi"
"""


def sbatch(d, tag, nb):
    NODES, NPROC = RANKS[nb]
    return f"""#!/bin/bash
#SBATCH -J {tag} -p ccq -N {NODES} -c 1 -n {NPROC} -t {WALL} --mem=0
#SBATCH -o {d}/rpa.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3
export root={COQUI}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root/lib64
export OMP_NUM_THREADS=1
export UCX_TLS=ud,sm,self
cd {d}
rm -f rpa.done thc.eri.h5 thc_eri.h5
timeout 28000 mpirun -np {NPROC} $root/bin/coqui --filenames rpa.toml > rpa.out 2>&1
echo "exit=$?" > rpa.done
"""


cmds = []
for a in ALAT:
    for nb in NBANDS:
        d = f"{WORK}/a{a}_n{nb}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/rpa.toml", "w").write(toml(a, nb))
        open(f"{d}/run.sbatch", "w").write(sbatch(d, f"cq{a}n{nb}", nb))
        cmds.append(f"sbatch {d}/run.sbatch")

print(f"wrote {len(cmds)} CoQui run dirs under {WORK}")
for c in cmds:
    print(c)
