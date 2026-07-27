#!/usr/bin/env python3
"""Run ON rusty.  ABINIT ACFD-RPA correlation energy for fcc Si vs lattice constant,
at nband = 100 / 250 / 500, with the SAME PAW dataset (jth_with_d/Si.xml) and the SAME
500-band WFK used by the CoQui EOS runs in abinit/eos_jthd_coqui.

Purpose: compare the RPA instability (spurious, volume-dependent over-binding that grows
with the number of empty bands) between ABINIT's own RPA and CoQui's THC-RPA, on a
bit-identical mean field.

Recipe (verified against abinit-10.6.7 sources, not guessed):
  optdriver 3 + gwrpacorr 1 + gwcalctyp 1   -- enforced pairing, m_chkinp.F90:1631
  Ec printed as 'RPA energy [Ha] :' (running total over q; take the LAST one)
  and to <prefix>_DS<n>_RPA as '#RPA  <value>'.
Notes:
  - PAW forces inclvkb=0 internally (m_screening_driver.F90:1762), so the q->0 [Vnl,r]
    commutator is omitted regardless of input. Same for every run here, so it cancels
    in the comparison.
  - gw_icutcoul 7 = Gygi-Baldereschi, to match CoQui's div_treatment='gygi'.
  - ecut / pawecutdg / ngkpt / kptopt / nsym MUST match the WFK header.
"""
import os

ABI     = "/mnt/home/mmorales/ceph/CoQui/abinit/bin/abinit"
PSPDIR  = "/mnt/home/mmorales/ceph/CoQui/abinit/pseudos"
PSP     = "jth_with_d/Si.xml"          # same PAW as the CoQui EOS (eos_jthd*)
SRC     = "/mnt/home/mmorales/ceph/CoQui/abinit/eos_jthd"        # holds the 500-band WFKs
WORK    = "/mnt/home/mmorales/ceph/CoQui/abinit/rpa_eos_jthd"

ALAT    = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]
NBANDS  = [100, 250, 500]
ECUTEPS = 12.0     # Ha, dielectric-matrix cutoff (fixed across the whole grid)
NFREQIM = 12       # Gauss-Legendre points on the imaginary axis

# -- must match the WFK header (eos_jthd/a*/si.abi) --
ECUT, PAWECUTDG, NGKPT = 25.0, 50.0, 4

NODES, NPROC, WALL = 2, 64, "12:00:00"


def abi(a, nbands, ecuteps=ECUTEPS, pawcross=0):
    ds = "\n".join(f"nband{i+1} {n}" for i, n in enumerate(nbands))
    return f"""# ABINIT ACFD-RPA correlation energy, fcc Si, a={a} Bohr, PAW {PSP}
# One dataset per nband; all read the SAME 500-band WFK used by the CoQui EOS run.
ndtset {len(nbands)}

# ---- structure / basis / k-mesh: must match the WFK header ----
acell 3*{a}
rprim 0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0
ntypat 1 znucl 14 natom 2 typat 1 1
xred 0.0 0.0 0.0  0.25 0.25 0.25
ecut {ECUT}
pawecutdg {PAWECUTDG}
ngkpt {NGKPT} {NGKPT} {NGKPT}
nshiftk 1
shiftk 0.0 0.0 0.0
kptopt 3
nsym 1
istwfk *1
iomode 3
occopt 1
ixc 11

# ---- RPA correlation energy (adiabatic-connection fluctuation-dissipation) ----
optdriver 3          # screening driver
gwrpacorr 1          # RPA Ec, exact integration over the coupling constant
gwcalctyp 1          # Gauss-Legendre mesh on the imaginary axis (required by gwrpacorr)
nfreqim {NFREQIM}
ecuteps {ecuteps}
gw_icutcoul 7        # Gygi-Baldereschi q->0, matches CoQui div_treatment='gygi'
awtr 0               # WFK is kptopt 3 => timrev=0; awtr/=0 is rejected (m_screening_driver:1781)
symchi 0             # nsym 1: no symmetry to exploit anyway
gwpara 1             # k-point parallelism; gwpara 2 needs time-reversal (m_screening_driver:509)
pawcross {pawcross}  # 0 = assume on-site completeness in the oscillators (CoQui-like)
getwfk_filepath "{SRC}/a{a}/sio_DS2_WFK.nc"

# ---- band series ----
{ds}

pp_dirpath "{PSPDIR}"
pseudos "{PSP}"
"""


def sbatch(d, tag):
    return f"""#!/bin/bash
#SBATCH -J {tag} -p ccq -N {NODES} -n {NPROC} -c 1 -t {WALL} --mem=0
#SBATCH -o {d}/rpa.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3 libxc netcdf-c netcdf-fortran
export OMP_NUM_THREADS=1
cd {d}
rm -f __ABI_MPIABORTFILE__ rpa.done
mpirun -np {NPROC} {ABI} rpa.abi > rpa.log 2>&1
grep -q "Calculation completed" rpa.log && echo ok > rpa.done || echo fail > rpa.done
"""


def emit(d, tag, text):
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "rpa.abi"), "w").write(text)
    open(os.path.join(d, "run.sbatch"), "w").write(sbatch(d, tag))
    return d


cmds = []
# main grid: 6 lattice constants x {100,250,500} bands
for a in ALAT:
    d = emit(f"{WORK}/a{a}", f"rpa{a}", abi(a, NBANDS))
    cmds.append(f"sbatch {d}/run.sbatch")

# ecuteps sensitivity probe at the most unstable volume, n=500 only:
# guards against a small ecuteps masking the augmentation-driven runaway.
for ec in (6.0, 18.0):
    a = "10.05"
    d = emit(f"{WORK}/probe_ecuteps{int(ec)}_a{a}", f"rpaec{int(ec)}",
             abi(a, [500], ecuteps=ec))
    cmds.append(f"sbatch {d}/run.sbatch")

# pawcross probe: relax the on-site-completeness assumption in the oscillators.
for a in ("10.05", "10.55"):
    d = emit(f"{WORK}/probe_pawcross_a{a}", f"rpaxc{a}", abi(a, NBANDS, pawcross=1))
    cmds.append(f"sbatch {d}/run.sbatch")

print(f"wrote {len(cmds)} run dirs under {WORK}")
for c in cmds:
    print(c)
