#!/usr/bin/env python3
"""Generate TOML inputs + runner for ISDF-metric error-vs-N_mu sweeps.

Usage:
  python3 gen_sweep.py --root RUNDIR [--mode local|rusty] [--fixture-base DIR]
                       [--fixtures ncpp,uspp,paw] [--configs all|baseline,...]

Produces RUNDIR/<fixture>/<config>/c<CC>/rpa.toml and RUNDIR/run_all.sh
(local: sequential; rusty: one sbatch per fixture/config sweep).
Energies are parsed later by collect.py from run.log (+ <prefix>.mbpt.h5).
"""
import argparse, os, sys, stat, textwrap

FIXTURES = {
    # name -> (subdir under fixture base, nbnd, pp_type)
    "ncpp": ("qe/si_kp222_ncpp", 16, "ncpp"),
    "uspp": ("qe/si_kp222_uspp", 16, "uspp"),
    "paw":  ("qe/si_kp222_paw",  16, "paw"),
}

# knob configurations: name -> dict of extra [interaction.thc] keys
CONFIGS = {
    "baseline":   {},
    # knob 2 scan
    "f025":       {"isdf_filter_alpha": 0.25},
    "f050":       {"isdf_filter_alpha": 0.5},
    "f100":       {"isdf_filter_alpha": 1.0},
    "f200":       {"isdf_filter_alpha": 2.0},
    # knob 1
    "wgap":       {"isdf_pair_weight": "gap"},
    # knob 3
    "mbare":      {"isdf_metric": "bare"},
    "matten":     {"isdf_metric": "attenuated"},
    "mbare_s3":   {"isdf_metric": "bare", "isdf_pool_factor": 3.0},
    # composition
    "f050_mbare": {"isdf_filter_alpha": 0.5, "isdf_metric": "bare"},
}

C_LIST = [4, 6, 8, 10, 12]
C_REF = 20  # THC self-convergence reference

TOML = """\
[mean_field.qe]
name     = "mf"
prefix   = "pwscf"
outdir   = "{outdir}"
filetype = "h5"

[interaction.thc]
name       = "eri"
mean_field = "mf"
storage    = "incore"
nIpts      = {nipts}
{extra}
[rpa]
interaction = "eri"
beta        = 1000
iaft_prec   = "medium"
outdir      = "./"
prefix      = "sweep"
"""

CHOL_TOML = """\
[mean_field.qe]
name     = "mf"
prefix   = "pwscf"
outdir   = "{outdir}"
filetype = "h5"

[interaction.cholesky]
name       = "eri"
mean_field = "mf"
storage    = "incore"
thresh     = 1e-10

[rpa]
interaction = "eri"
beta        = 1000
iaft_prec   = "medium"
outdir      = "./"
prefix      = "sweep"
"""

SB_HEADER = """\
#!/bin/bash
#SBATCH -p ccq -A ccq -C genoa --mem=0 -N1 -n {np} -c1 -t 12:00:00
#SBATCH -J isdf_sweep_{tag}
#SBATCH -o sweep.%j.out
#SBATCH -e sweep.%j.err
source /etc/profile.d/modules.sh; module purge; module load modules
module load gcc/13.3.0 openmpi/4.1.8 hdf5/1.14.5 fftw/3.3.10 boost/1.87.0 intel-oneapi-mkl/2024.2.2
export OMP_NUM_THREADS=1
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--mode", default="local", choices=["local", "rusty"])
    p.add_argument("--fixture-base", default=None,
                   help="dir containing qe/si_* fixtures (default: ../../tests/unit_test_files relative to this file)")
    p.add_argument("--coqui", default=None, help="coqui binary path")
    p.add_argument("--np", type=int, default=2, help="MPI ranks (<= nbnd!)")
    p.add_argument("--fixtures", default="ncpp,uspp,paw")
    p.add_argument("--configs", default="all")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    fbase = args.fixture_base or os.path.normpath(os.path.join(here, "..", "..", "tests", "unit_test_files"))
    coqui = args.coqui or os.path.normpath(os.path.join(here, "..", "..", "build", "bin", "coqui"))
    fixtures = args.fixtures.split(",")
    configs = list(CONFIGS) if args.configs == "all" else args.configs.split(",")

    launcher = ("mpiexec -n {np} --oversubscribe" if args.mode == "local" else "mpirun -n {np}").format(np=args.np)
    env = "export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1" if args.mode == "local" else ""

    jobs = []
    for f in fixtures:
        sub, nbnd, _pp = FIXTURES[f]
        outdir = os.path.join(fbase, sub) + "/"
        assert args.np <= nbnd, f"np={args.np} > nbnd={nbnd} (reader rank cap)"
        # references
        refs = [("thc_ref", TOML.format(outdir=outdir, nipts=C_REF * nbnd, extra=""))]
        if f == "ncpp":
            refs.append(("chol_ref", CHOL_TOML.format(outdir=outdir)))
        for name, toml in refs:
            d = os.path.join(args.root, f, name)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "rpa.toml"), "w").write(toml)
            jobs.append(d)
        # sweeps
        for cfg in configs:
            extra = "".join(f'{k} = {v!r}\n' if isinstance(v, str) else f'{k} = {v}\n'
                            for k, v in CONFIGS[cfg].items())
            for c in C_LIST:
                d = os.path.join(args.root, f, cfg, f"c{c:02d}")
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, "rpa.toml"), "w").write(
                    TOML.format(outdir=outdir, nipts=c * nbnd, extra=extra))
                jobs.append(d)

    runner = os.path.join(args.root, "run_all.sh")
    with open(runner, "w") as fh:
        fh.write("#!/bin/bash\nset -u\n%s\n" % env)
        fh.write("BIN=%s\n" % coqui)
        for d in jobs:
            fh.write(textwrap.dedent(f"""\
                cd {os.path.abspath(d)}
                if [ ! -f run.log ] || ! grep -q 'RPA energy routines end' run.log; then
                  echo "RUN {d}"
                  {launcher} $BIN rpa.toml > run.log 2>&1 || echo "  FAILED (exit $?)"
                fi
                """))
    os.chmod(runner, os.stat(runner).st_mode | stat.S_IEXEC)
    if args.mode == "rusty":
        for f in fixtures:
            tag = f
            sb = os.path.join(args.root, f"run_{f}.sbatch")
            with open(sb, "w") as fh:
                fh.write(SB_HEADER.format(np=args.np, tag=tag))
                fh.write(f"bash {os.path.abspath(runner)}\n")
    print(f"{len(jobs)} run dirs under {args.root}; runner: {runner}")


if __name__ == "__main__":
    main()
