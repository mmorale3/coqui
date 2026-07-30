#!/usr/bin/env python3
"""Generate + submit the CoQui exchange-term decomposition runs for the
CoQui-vs-ABINIT exchange-row discrepancy (eos_exchange_ledger.md §4 open item).

The ledger localizes the remaining EOS error to the exchange row:

    CoQui[smooth+aug+K_a] + CoQui[cv]  -  ABINIT[fock0 + efock]  =  +8.29 mHa
                                                       (a=10.25, +8.76 at 10.05)

and the core-valence halves of that agree to 6 uHa, so the whole thing sits in
smooth+aug+K_a vs fock0+efockdc.  These runs split CoQui's side the same way
ABINIT's is already split, WITH ALL OTHER SETTINGS HELD FIXED -- the earlier
`rpa_localize_jthd` probes varied `thresh`/`paw_isdf_tol` at the same time as
`paw_onsite`, so their K_a is contaminated.

    base        aug + K_a          (must reproduce the EOS series E_x)
    onsite_off  aug, no K_a        -> K_a      = base - onsite_off
    aug_off     smooth only        -> aug      = onsite_off - aug_off
    shape       Option A           independent route (K_a auto-dropped)

RPA is left in only because `Exchange energy:` is printed by the rpa driver;
its grid is made deliberately coarse (beta/wmax/iaft_prec), which does not
touch E_x -- `base` reproducing the EOS E_x is the check on that.

    python3 gen_exx_split.py            # write + submit
    python3 gen_exx_split.py --dry      # write only
"""
import argparse
import subprocess
import sys

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
MF = ROOT + "/eos_jthd_coqui_fix"          # sqrt(4pi)-corrected mfs
OUT = ROOT + "/exx_split"
BIN = "/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui/build/CPU"

# EOS-series settings, held fixed across every variant.
THRESH = "1e-5"
ISDF_TOL = "1e-8"
NBND = 500

VARIANTS = {
    "base":       {},
    "onsite_off": {"paw_onsite": "false"},
    "aug_off":    {"paw_aug": "false"},
    "shape":      {"vv_compensation": '"shape"'},
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
paw_aug = {paw_aug}
paw_onsite = {paw_onsite}
paw_isdf_tol = {isdf_tol}
paw_isdf_metric = "coulomb"
{extra}[rpa]
interaction = "eri"
interaction_hf = "eri"
beta = 100
wmax = 4.0
iaft_prec = "low"
output = "exx_{tag}_{a}"
div_treatment = "gygi"
hf_div_treatment = "gygi"
"""

SBATCH = """#!/bin/bash
#SBATCH -J x_{tag}_{a} -p ccq -N 8 -c 1 -n 64 --ntasks-per-node=8 -t 2:00:00 --mem=0
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
    ap.add_argument("--avals", nargs="+", default=["10.25", "10.55"])
    a = ap.parse_args()

    script = ["set -e"]
    jobs = []
    for av in a.avals:
        for tag, ov in VARIANTS.items():
            d = "%s/a%s_%s" % (OUT, av, tag)
            extra = ""
            if "vv_compensation" in ov:
                extra = "vv_compensation = %s\n" % ov["vv_compensation"]
            toml = TOML.format(
                mf=MF, a=av, nbnd=NBND, thresh=THRESH, isdf_tol=ISDF_TOL,
                paw_aug=ov.get("paw_aug", "true"),
                paw_onsite=ov.get("paw_onsite", "true"),
                extra=extra, tag=tag)
            sb = SBATCH.format(tag=tag, a=av, d=d, bin=BIN)
            script += [
                "mkdir -p %s" % d,
                "cat > %s/rpa.toml <<'TOMLEOF'\n%sTOMLEOF" % (d, toml),
                "cat > %s/run.sbatch <<'SBEOF'\n%sSBEOF" % (d, sb),
            ]
            jobs.append(d)

    if not a.dry:
        # Serialize: six concurrent 16-node jobs reliably tripped
        # `ib_mlx5_log.c: Transport retry count exceeded` in the EOS campaign;
        # --dependency=afterany on a chain fixed it.
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


if __name__ == "__main__":
    main()
