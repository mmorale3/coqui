#!/usr/bin/env python3
"""THC convergence scan for the residual exchange-row discrepancy.

After the one-centre factor-2 was traced to ABINIT (eos_exchange_ledger.md
§3g), 0.83 mHa of the Si exchange row is left, and the on/off split attributes
all of it to the THC route's K_a (0.884 of the reference, against 0.994 for the
exact direct route). Two mechanisms were already excluded by inspecting the
existing run logs:

  * the one-centre ISDF block is NOT compressed at paw_isdf_tol = 1e-8
    ("kept nlambda=324 (full-rank cap=324)"), and
  * the interpolating-point set is identical with K_a on and off
    (N_smooth=4928, N_aug=648 in both).

So if the THC route really is losing ~0.8 mHa it has to be the ERI truncation
(`thresh`) or the band set the ISDF is fitted on (`nbnd`) — both of which this
scan varies against the `base` run of gen_exx_split.py.

EVERYTHING else is held at the exx_split values, including beta/wmax/iaft_prec:
those perturb the KS density matrix and hence E_x by ~11 mHa, so the comparison
is only meaningful against `exx_split/a10.25_base` (-2.1154385745542683), NOT
against the EOS series.

    python3 gen_thc_conv.py           # write + submit
    python3 gen_thc_conv.py --dry     # write only
"""
import argparse
import subprocess
import sys

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
MF = ROOT + "/eos_jthd_coqui_fix"
OUT = ROOT + "/thc_conv"
BIN = "/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui/build/CPU"

A = "10.25"
BASE_EX = -2.1154385745542683   # exx_split/a10.25_base, thresh 1e-5, nbnd 500

# tag -> (nbnd, thresh, isdf_tol)
VARIANTS = {
    "thresh1e-4":  (500, "1e-4", "1e-8"),
    "thresh1e-6":  (500, "1e-6", "1e-8"),
    "nbnd250":     (250, "1e-5", "1e-8"),
    "nbnd100":     (100, "1e-5", "1e-8"),
}

TOML = """[mean_field.bdft]
name = "mf"
prefix = "mf"
outdir = "{mf}/a{a}"
nbnd = {nbnd}
[interaction.thc]
name = "eri"
mean_field = "mf"
storage = "incore"
thresh = {thresh}
chol_block_size = 8
paw_aug = true
paw_onsite = {onsite}
paw_isdf_tol = {tol}
paw_isdf_metric = "coulomb"
[rpa]
interaction = "eri"
interaction_hf = "eri"
beta = 100
wmax = 4.0
iaft_prec = "low"
output = "conv_{tag}"
div_treatment = "gygi"
hf_div_treatment = "gygi"
"""

SBATCH = """#!/bin/bash
#SBATCH -J c_{tag} -p ccq -N 8 -c 1 -n 64 --ntasks-per-node=8 -t 4:00:00 --mem=0
#SBATCH -o {d}/rpa.%j.out
source /etc/profile.d/modules.sh; module purge; module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{bin}/lib64
export OMP_NUM_THREADS=1
cd {d}
rm -f rpa.done thc.eri.h5 thc_eri.h5
mpirun -np 64 {bin}/bin/coqui --filenames rpa.toml > rpa.out 2>&1
touch rpa.done
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    script, jobs = ["set -e"], []
    for tag, (nbnd, thresh, tol) in VARIANTS.items():
        # each variant is run with K_a ON and OFF so the split can be redone
        # at every setting, not just at the production one
        for onsite, suf in (("true", ""), ("false", "_off")):
            d = "%s/a%s_%s%s" % (OUT, A, tag, suf)
            toml = TOML.format(mf=MF, a=A, nbnd=nbnd, thresh=thresh, tol=tol,
                               onsite=onsite, tag=tag + suf)
            sb = SBATCH.format(tag=tag + suf, d=d, bin=BIN)
            script += ["mkdir -p %s" % d,
                       "cat > %s/rpa.toml <<'TOMLEOF'\n%sTOMLEOF" % (d, toml),
                       "cat > %s/run.sbatch <<'SBEOF'\n%sSBEOF" % (d, sb)]
            jobs.append(d)

    if not args.dry:
        script.append('prev=""')
        for d in jobs:
            script.append(
                'if [ -z "$prev" ]; then jid=$(sbatch --parsable %s/run.sbatch); '
                'else jid=$(sbatch --parsable --dependency=afterany:$prev %s/run.sbatch); fi; '
                'echo "%s -> $jid"; prev=$jid' % (d, d, d.rsplit("/", 1)[1]))

    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input="\n".join(script), text=True, capture_output=True)
    print(p.stdout)
    if p.returncode:
        sys.exit(p.stderr[-2000:])
    print("reference: exx_split/a10.25_base E_x = %.13f" % BASE_EX)


if __name__ == "__main__":
    main()
