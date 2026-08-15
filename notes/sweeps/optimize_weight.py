#!/usr/bin/env python3
"""M4b: optimize the parametrized separable pair weight on Si (NCPP).

Objective (per the project scoring directive): |dE_x| + |dE_RPA| against the
exact Cholesky reference, summed over the c values in --c-list, for a given
weight family + parameters. Coarse grid first, then Nelder-Mead (pure-python)
around the grid optimum. Each evaluation = one selection + RPA-driver run per c.

Usage:
  python3 optimize_weight.py --work WORKDIR [--family exp|sigmoid|power]
      [--c-list 8,10] [--np 2] [--coqui BIN] [--metric l2|bare|attenuated]

Reference values (qe_si222_ncpp, chol tol=1e-10):
  E_x = -2.221346863233354, E_RPA = -0.235430373774913
"""
import argparse, itertools, math, os, re, subprocess, sys

REF = {"e_x": -2.221346863233354, "e_rpa": -0.235430373774913}
NBND = 16
FIXTURE = None  # resolved from repo layout

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
isdf_pair_weight = "{family}"
isdf_weight_params = [{params}]
{metric_line}
[rpa]
interaction = "eri"
beta        = 1000
iaft_prec   = "medium"
outdir      = "./"
prefix      = "sweep"
"""

GRIDS = {
    # family -> list of parameter tuples for the coarse grid
    "exp":     [(t,) for t in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)],
    "sigmoid": [(w, t) for w in (0.1, 0.3, 0.6, 1.0) for t in (0.05, 0.15, 0.4)],
    "power":   [(p,) for p in (0.25, 0.5, 1.0, 1.5, 2.0)],
}

def run_one(args, family, params, c, tag):
    d = os.path.join(args.work, f"{family}_{tag}_c{c:02d}")
    os.makedirs(d, exist_ok=True)
    metric_line = (f'isdf_metric = "{args.metric}"\n' if args.metric != "l2" else "")
    open(os.path.join(d, "rpa.toml"), "w").write(TOML.format(
        outdir=FIXTURE, nipts=c*NBND, family=family,
        params=", ".join(f"{p}" for p in params), metric_line=metric_line))
    log = os.path.join(d, "run.log")
    if not (os.path.isfile(log) and "RPA energy routines end" in open(log, errors="replace").read()):
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
        with open(log, "w") as fh:
            subprocess.run(["mpiexec", "-n", str(args.np), "--oversubscribe",
                            args.coqui, "rpa.toml"], cwd=d, stdout=fh, stderr=fh, env=env)
    txt = open(log, errors="replace").read()
    ex = re.findall(r"Exchange energy:\s+([-\d.eE+]+)", txt)
    er = re.findall(r"^RPA energy:\s+([-\d.eE+]+)", txt, re.M)
    if not ex or not er:
        return None
    return abs(float(ex[-1]) - REF["e_x"]) + abs(float(er[-1]) - REF["e_rpa"])

def objective(args, family, params, cache={}):
    key = (family, tuple(round(p, 6) for p in params))
    if key in cache: return cache[key]
    if any(p <= 0 for p in params[-1:]) or any(p < 0 for p in params):
        return 1e3
    tot = 0.0
    for c in args.c_list:
        v = run_one(args, family, params, c, "_".join(f"{p:g}" for p in params))
        if v is None: return 1e3
        tot += v
    cache[key] = tot
    print(f"  {family} {params} -> {tot:.6e}", flush=True)
    return tot

def nelder_mead(f, x0, step=0.3, it=25):
    n = len(x0)
    if n == 0: return x0
    pts = [list(x0)] + [[x0[i]*(1+step) if j == i else x0[i] for i, _ in enumerate(x0)] for j in range(n)]
    vals = [f(tuple(p)) for p in pts]
    for _ in range(it):
        order = sorted(range(n+1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < 1e-8: break
        cen = [sum(p[i] for p in pts[:-1])/n for i in range(n)]
        xr = tuple(cen[i] + (cen[i] - pts[-1][i]) for i in range(n))
        fr = f(xr)
        if fr < vals[0]:
            xe = tuple(cen[i] + 2*(cen[i] - pts[-1][i]) for i in range(n))
            fe = f(xe)
            pts[-1], vals[-1] = (list(xe), fe) if fe < fr else (list(xr), fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = list(xr), fr
        else:
            xc = tuple(cen[i] + 0.5*(pts[-1][i] - cen[i]) for i in range(n))
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = list(xc), fc
            else:
                for i in range(1, n+1):
                    pts[i] = [(pts[i][j] + pts[0][j])/2 for j in range(n)]
                    vals[i] = f(tuple(pts[i]))
    order = sorted(range(n+1), key=lambda i: vals[i])
    return tuple(pts[order[0]]), vals[order[0]]

def main():
    global FIXTURE
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--family", default="all")
    p.add_argument("--c-list", default="8,10")
    p.add_argument("--np", type=int, default=2)
    p.add_argument("--coqui", default=None)
    p.add_argument("--metric", default="l2", choices=["l2", "bare", "attenuated"])
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    FIXTURE = os.path.normpath(os.path.join(here, "..", "..", "tests", "unit_test_files", "qe", "si_kp222_ncpp")) + "/"
    args.coqui = args.coqui or os.path.normpath(os.path.join(here, "..", "..", "build", "bin", "coqui"))
    args.c_list = [int(c) for c in args.c_list.split(",")]
    os.makedirs(args.work, exist_ok=True)
    fams = list(GRIDS) if args.family == "all" else [args.family]
    results = {}
    for fam in fams:
        print(f"== coarse grid: {fam} ==", flush=True)
        best = min(GRIDS[fam], key=lambda prm: objective(args, fam, prm))
        print(f"== refine ({fam}) from {best} ==", flush=True)
        xopt, fopt = nelder_mead(lambda prm: objective(args, fam, prm), best)
        results[fam] = (tuple(round(x, 4) for x in xopt), fopt)
        print(f"** {fam}: params={results[fam][0]} objective={fopt:.6e}", flush=True)
    print("\n=== summary (objective = sum_c |dEx|+|dErpa|, c in", args.c_list, ") ===")
    for fam, (prm, val) in sorted(results.items(), key=lambda kv: kv[1][1]):
        print(f"{fam:8s} params={prm} objective={val:.6e}")

if __name__ == "__main__":
    main()
