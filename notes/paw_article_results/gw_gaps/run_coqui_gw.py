#!/usr/bin/env python3
"""Drive the CoQui half of the G0W0 band-gap campaign.

For each completed ABINIT run directory (<mat>_k*_n*_e*/ with mf/mf.done):
  1. convert the NSCF mean field with abinit2coqui (WFK + DEN + POT + PAW-XML
     + corewf), and
  2. write a CoQui G0W0 toml plus an sbatch that runs it.

The ABINIT side of these runs was generated with symmorphi 0 / kptopt 4, so the
WFK is symmetry-reduced and abinit2coqui takes its minimal-BZ path; CoQui
rebuilds the k/q maps itself.

--isdf-thresh is the ISDF rank axis of the convergence study and is the one
knob that has no ABINIT counterpart: it controls the factorization, not the
physics, so the CoQui-vs-ABINIT comparison is only meaningful once the gap has
stopped moving with it.
"""

import argparse
import os
import re
import subprocess
import sys

TAG = re.compile(r"^(?P<mat>[a-z]+)_k(?P<k>\d+)_n(?P<nb>\d+)_e(?P<ec>[\d.]+)$")

# species in znucl order -- MUST match the order gen_inputs.py wrote into
# `znucl` and `pseudos`, since abinit2coqui pairs --pawxml positionally.
SPECIES = {
    "si":  ["Si"],
    "c":   ["C"],
    "sic": ["Si", "C"],
    "alp": ["Al", "P"],
    "mgo": ["Mg", "O"],
}

TOML = """[mean_field.bdft]
name   = "mf"
prefix = "bdft"
outdir = "./"

[interaction.thc]
name        = "eri"
mean_field  = "mf"
storage     = "{storage}"
thresh      = {thresh}
chol_block_size = 8
r_blk       = 20
distr_tol   = 0.4

[evgw0]
interaction   = "eri"
beta          = {beta}
lambda        = 1200.0
iaft_prec     = "high"
niter         = 1
output        = "g0w0"
div_treatment = "gygi"
qp_type       = "sc"
ac_alg        = "pade"
eta           = 1e-6
Nfit          = 26
"""

SBATCH = """#!/bin/bash
# -C icelake|genoa: both CoQui and ABINIT are built -march=native on an
# AVX-512 node. The rome (Zen 2) partition has no AVX-512 and kills the binary
# with SIGILL at startup, which surfaces as a 3-second job failure on a random
# subset of runs.
#SBATCH -J {job} -C icelake|genoa -p ccq -N {nodes} -n {ranks} -c 1 --mem=0 -t {walltime}
#SBATCH -o coqui.%j.out -e coqui.%j.err
set -e
cd $SLURM_SUBMIT_DIR
source {build}/env.bash
export OMP_NUM_THREADS=1
export UCX_TLS=ud,sm,self
export LD_LIBRARY_PATH={build}/lib64:$LD_LIBRARY_PATH
if [ ! -f gw.done ]; then
  rm -f thc.eri.h5 *.thc.eri.h5 2>/dev/null || true
  mpirun -np {ranks} {build}/bin/coqui --filenames gw.toml --verbosity 2 > gw.out 2>&1
  touch gw.done
fi
echo COQUI_GW_DONE
"""


def convert(rundir, mat, pspdir, a2c, verbose=True):
    """Run abinit2coqui on the NSCF (DS2) outputs. Returns True on success."""
    mf = os.path.join(rundir, "mf")
    wfk = os.path.join(mf, "mfo_DS2_WFK.nc")
    den = os.path.join(mf, "mfo_DS2_DEN.nc")
    pot = os.path.join(mf, "mfo_DS2_POT.nc")
    for f in (wfk, den, pot):
        if not os.path.exists(f):
            print("# %s: missing %s" % (rundir, os.path.basename(f)))
            return False
    if os.path.exists(os.path.join(rundir, "bdft.h5")):
        return True
    xml = [os.path.join(pspdir, "%s.GGA_PBE-JTH.xml" % s) for s in SPECIES[mat]]
    cwf = [os.path.join(pspdir, "%s.GGA_PBE-JTH.corewf.xml" % s) for s in SPECIES[mat]]
    cmd = [sys.executable, os.path.join(a2c, "abinit2coqui.py"), wfk,
           "--outdir", rundir, "--prefix", "bdft",
           "--pawxml"] + xml + ["--corewf"] + cwf + [
           "--pot", pot, "--den", den, "--xc", "pbe"]
    env = dict(os.environ, PYTHONPATH=a2c)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if verbose:
        for line in (r.stdout or "").strip().split("\n")[-4:]:
            if line.strip():
                print("   " + line)
    if r.returncode != 0:
        print("# %s: CONVERSION FAILED\n%s" % (rundir, (r.stderr or "")[-600:]))
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pspdir",
                    default="/mnt/home/mmorales/ceph/CoQui/abinit/pseudos/JTH-PBE-v2.0")
    ap.add_argument("--a2c",
                    default="/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui_gwgap/"
                            "src/python/mean_field/abinit_interface")
    ap.add_argument("--build",
                    default="/mnt/home/mmorales/ceph/CoQui/paw_tests/coqui_gwgap/build/CPU")
    ap.add_argument("--isdf-thresh", default="1e-5")
    ap.add_argument("--beta", default="1000")
    ap.add_argument("--storage", default="incore", choices=["incore", "outcore"])
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these run-dir names")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    made = subd = 0
    for d in sorted(os.listdir(args.root)):
        if not TAG.match(d):
            continue
        if args.only and d not in args.only:
            continue
        rundir = os.path.join(args.root, d)
        if not os.path.exists(os.path.join(rundir, "mf", "mf.done")):
            continue
        mat = TAG.match(d).group("mat")
        if mat not in SPECIES:
            continue
        print("== %s" % d)
        if not convert(rundir, mat, args.pspdir, args.a2c):
            continue

        nk = int(TAG.match(d).group("k"))
        nodes, ranks = (1, 16) if nk <= 4 else (2, 32)
        with open(os.path.join(rundir, "gw.toml"), "w") as f:
            f.write(TOML.format(thresh=args.isdf_thresh, beta=args.beta,
                                storage=args.storage))
        with open(os.path.join(rundir, "run_coqui.sbatch"), "w") as f:
            f.write(SBATCH.format(job="cqgw_" + d, nodes=nodes, ranks=ranks,
                                  walltime="12:00:00", build=args.build))
        made += 1
        if args.submit and not os.path.exists(os.path.join(rundir, "gw.done")):
            r = subprocess.run(["sbatch", "run_coqui.sbatch"], cwd=rundir,
                               capture_output=True, text=True)
            if r.returncode == 0:
                subd += 1
            else:
                print("   submit failed: %s" % (r.stderr or "").strip()[:200])
    print("\nprepared %d run(s); submitted %d" % (made, subd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
