#!/usr/bin/env python3
"""Full chain for the Si norm series: QE scf -> nscf -> pw2coqui -> CoQui RPA vs nbnd.

The KKK prediction being tested: the RPA correlation energy converges as 1/N with the
number of bands, but to a limit displaced by the pseudo-density norm defect Q_occ.  So
across the norm series N0..N9 (Q_occ spanning 137x) the *extrapolated* E_c should move
systematically with Q_occ and approach the norm-conserving reference, while any purely
scattering-related dataset difference should not show that trend.

k-grid 2x2x2 and nbnd in {50,100,150,250} match the existing ONCV/ccECP/kjpaw/JTH rows in
notes/paw_article_results/si_rpa_proj.csv, so the new datasets drop straight into that
comparison.

Usage:
    run_rpa.py <rung> [<rung> ...] [--alat 10.26,10.60] [--nbnd 50,100,150,250]

The KKK limit is the *extrapolated* E_c, so every chain is reported together with a
linear 1/nbnd extrapolation; two volumes are needed because the norm-defect error is
volume-dependent, which is what makes it an EOS error rather than a constant shift.
"""

import os
import re
import shutil
import subprocess
import sys

# Paths and launcher are overridable so the same driver runs on this host (serial) and
# on rusty (srun), rather than maintaining a divergent cluster copy.  SIPAW_MPI is the
# launcher prefix, e.g. "srun -n 64"; empty means run the binary directly.
QE_BIN = os.environ.get("SIPAW_QE_BIN", os.path.expanduser("~/Software/qe-7.4.1/build/cpu/bin"))
COQUI = os.environ.get(
    "SIPAW_COQUI", os.path.expanduser("~/Software/CoQui_Separate_Development/build/cpu/bin/coqui"))
LADDER = os.environ.get("SIPAW_LADDER", os.path.expanduser("~/Projects/PAW_GW/si_gw_paw"))
WORK = os.environ.get("SIPAW_WORK", os.path.join(LADDER, "_rpa"))
MPI = os.environ.get("SIPAW_MPI", "").split()
UPF = "Si.GGA-PBE-paw.UPF"

ALAT = 10.26
ECUTWFC = 80.0        # converged to 0.01-0.03 mRy/atom for both ends of the series
ECUTRHO = 640.0
NK = 2
NBNDS = [50, 100, 150, 250]

PW = """&CONTROL
   calculation = '{calc}'
   verbosity = 'high'
   prefix = 'si'
   outdir = './out'
   pseudo_dir = '{pseudo}'
/
&SYSTEM
   ibrav = 2
   celldm(1) = {alat}
   nat = 2
   ntyp = 1
   ecutwfc = {ecutwfc}
   ecutrho = {ecutrho}
   occupations = 'fixed'
   input_dft = 'pbe'
   force_symmorphic = .true.
   {nbnd}
/
&ELECTRONS
   conv_thr = 1d-10
   mixing_beta = 0.3
   diagonalization = 'david'
   diago_full_acc = .true.
/
ATOMIC_SPECIES
Si 28.085 {upf}
ATOMIC_POSITIONS crystal
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS automatic
{nk} {nk} {nk} 0 0 0
"""

RPA_TOML = """[mean_field.qe]
name     = "mf_qe"
prefix   = "si"
outdir   = "./out/"
filetype = "h5"
nbnd     = {nbnd}

[interaction.thc]
name        = "eri"
mean_field  = "mf_qe"
storage     = "incore"
thresh      = 1e-6
chol_block_size = 8
r_blk       = 20
distr_tol   = 0.4

[rpa]
interaction      = "eri"
interaction_hf   = "eri"
beta             = 1000
wmax             = 12.0
iaft_prec        = "high"
output           = "si_rpa_{tag}_n{nbnd}"
div_treatment    = "gygi"
hf_div_treatment = "gygi"
"""


def env():
    # KMP_DUPLICATE_LIB_OK: homebrew openblas pulls the formula libomp while the binary
    # links llvm's; harmless at OMP_NUM_THREADS=1 (see CLAUDE.md).
    return dict(os.environ, OMP_NUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE")


def sh(cmd, cwd, logname, stdin=None):
    with open(os.path.join(cwd, logname), "w") as fout:
        fin = open(stdin) if stdin else None
        try:
            r = subprocess.run(cmd, cwd=cwd, env=env(), stdout=fout,
                               stderr=subprocess.STDOUT, stdin=fin)
        finally:
            if fin:
                fin.close()
    return r.returncode


def run_chain(rung, nbnds=NBNDS, alat=ALAT):
    # One directory per (dataset, volume): the EOS needs both volumes kept side by side.
    d = os.path.join(WORK, rung if alat == ALAT else f"{rung}_a{alat:g}")
    os.makedirs(d, exist_ok=True)
    pseudo = os.path.join(LADDER, rung)

    # 1. SCF
    with open(os.path.join(d, "scf.in"), "w") as f:
        f.write(PW.format(calc="scf", pseudo=pseudo, alat=alat, ecutwfc=ECUTWFC,
                          ecutrho=ECUTRHO, nbnd="", upf=UPF, nk=NK))
    sh(MPI + [os.path.join(QE_BIN, "pw.x"), "-in", "scf.in"], d, "scf.out")
    if "convergence NOT" in open(os.path.join(d, "scf.out"), errors="replace").read():
        return {"error": "scf not converged"}

    # 2. NSCF at the largest band count; CoQui then truncates via [mean_field.qe] nbnd
    with open(os.path.join(d, "nscf.in"), "w") as f:
        f.write(PW.format(calc="nscf", pseudo=pseudo, alat=alat, ecutwfc=ECUTWFC,
                          ecutrho=ECUTRHO, nbnd=f"nbnd = {max(nbnds)}", upf=UPF, nk=NK))
    sh(MPI + [os.path.join(QE_BIN, "pw.x"), "-in", "nscf.in"], d, "nscf.out")

    # 3. pw2coqui companion file
    with open(os.path.join(d, "p2c.in"), "w") as f:
        f.write("&input_pw2coqui\n  prefix = 'si'\n  outdir = './out'\n/\n")
    sh(MPI + [os.path.join(QE_BIN, "pw2coqui.x"), "-in", "p2c.in"], d, "p2c.out")

    # 4. CoQui RPA at each band count
    out = {}
    for nb in nbnds:
        toml = f"rpa_n{nb}.toml"
        with open(os.path.join(d, toml), "w") as f:
            f.write(RPA_TOML.format(nbnd=nb, tag=os.path.basename(d)))
        sh(MPI + [COQUI, "--filenames", toml], d, f"rpa_n{nb}.out")
        txt = open(os.path.join(d, f"rpa_n{nb}.out"), errors="replace").read()
        m = re.search(r"RPA energy:\s+(-?[\d.eE+]+)", txt)
        out[nb] = float(m.group(1)) if m else None
    return out


def extrapolate(res):
    """Linear fit of E_c against 1/nbnd -> the nbnd -> inf limit.

    KKK's claim is about this limit, not about any single band count: the norm defect
    leaves the 1/N *slope* essentially intact and displaces the intercept.
    """
    pts = [(1.0 / nb, e) for nb, e in sorted(res.items()) if isinstance(e, float)]
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        return None
    slope = (n * sxy - sx * sy) / den
    return (sy - slope * sx) / n          # intercept = value at 1/nbnd -> 0


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    alats = [float(x) for x in _opt("--alat", str(ALAT)).split(",")]
    nbnds = [int(x) for x in _opt("--nbnd", ",".join(str(n) for n in NBNDS)).split(",")]
    skip = {"--alat", "--nbnd"}
    rungs = [a for i, a in enumerate(sys.argv[1:], 1)
             if a not in skip and sys.argv[i - 1] not in skip] or ["N0"]
    for rung in rungs:
        for alat in alats:
            res = run_chain(rung, nbnds=nbnds, alat=alat)
            lim = extrapolate(res) if isinstance(res, dict) and "error" not in res else None
            cells = "  ".join(f"n{nb}={res[nb]:.6f}" if isinstance(res.get(nb), float)
                              else f"n{nb}=FAILED" for nb in nbnds) \
                if "error" not in res else res["error"]
            print(f"{rung} a={alat:g}: {cells}"
                  + (f"   1/nbnd limit = {lim:.6f}" if lim is not None else ""), flush=True)
