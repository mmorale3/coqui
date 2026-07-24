#!/usr/bin/env python3
"""Two diagnostic threads (run on rusty):
  T1 (qe71_pbe): kjpaw + ONCV Si PBE EOS using the INDEPENDENT espresso/7.1 build
       (pyang) -> tests whether the PAW large-a0 is specific to the ~/Devel q-e
       (v7.4.1) build. If QE-7.1 kjpaw a0~10.33, the Devel build's PAW is broken.
  T2 (hf_matched): self-consistent QE-HF for USPP + ONCV at MATCHED ecut100/400,
       nqx=8, k888 (vs the under-converged hf_kp8 ecut65/nqx8) -> tests whether
       QE's augmented USPP HF anomaly (a0=10.72) is convergence or real.
Usage: python3 setup_threads.py ; sbatch run_qe71.sbatch ; sbatch run_hfmatched.sbatch
"""
import os
ROOT = os.path.expanduser("~/ceph/CoQui/Si_eos_rpa_hf")
DIAG = os.path.join(ROOT, "diag"); PSE = os.path.join(ROOT, "pseudos")
VOLS = ["10.20", "10.35", "10.50", "10.65"]

def qe_in(a, pp, dft, ecutwfc, ecutrho, nqx=None):
    sys_extra = ""
    if dft == "hf":
        sys_extra = (f"    input_dft   = 'hf'\n    nqx1 = {nqx}\n    nqx2 = {nqx}\n"
                     f"    nqx3 = {nqx}\n    exxdiv_treatment = 'gygi-baldereschi'\n")
    else:
        sys_extra = "    input_dft   = 'pbe'\n"
    econv = "1.0d-9" if dft == "hf" else "1.0d-10"
    return f"""\
&control
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './out'
    pseudo_dir  = '{PSE}/{pp}'
    verbosity   = 'high'
/
&system
    ibrav       = 2
    celldm(1)   = {a}
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
{sys_extra}    occupations = 'fixed'
/
&electrons
    diagonalization = 'david'
    conv_thr        = {econv}
    mixing_beta     = 0.4
/
ATOMIC_SPECIES
Si  28.085  Si.UPF
ATOMIC_POSITIONS alat
Si  0.000  0.000  0.000
Si  0.250  0.250  0.250
K_POINTS automatic
8 8 8 0 0 0
"""

def gen(setname, pps, dft, ecutwfc, ecutrho, nqx=None):
    dirs = []
    for pp in pps:
        for a in VOLS:
            d = os.path.join(DIAG, setname, pp, f"a{a.replace('.','p')}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "scf.in"), "w") as f:
                f.write(qe_in(a, pp, dft, ecutwfc, ecutrho, nqx))
            dirs.append(d)
    return dirs

CAMPAIGN_MODS = ["module load modules",
                 "module load gcc openmpi hdf5 boost intel-oneapi-mkl python-mpi lib/fftw3"]

def sbatch(name, dirs, module_lines, pwx, walltime="00:40:00", per_run=False, nproc=16):
    """Single job over all dirs (per_run=False) or one job per dir (per_run=True)."""
    def jobtext(jname, run_dirs):
        body = "\n".join(
            f'cd {d}\nif [ ! -f scf.out.done ]; then mpirun -np {nproc} {pwx} -in scf.in > scf.out && touch scf.out.done; fi'
            for d in run_dirs)
        return "\n".join([
            "#!/bin/bash", f"#SBATCH -J {jname}", "#SBATCH -p ccq", "#SBATCH -N 1",
            "#SBATCH --exclusive", "#SBATCH -c 1", f"#SBATCH -t {walltime}",
            f"#SBATCH -o {DIAG}/{jname}.%j.out", f"#SBATCH -e {DIAG}/{jname}.%j.err",
            "set -e", "source /etc/profile.d/modules.sh", "module purge",
            *module_lines, "export OMP_NUM_THREADS=1",
            'echo "start $(date)"', body, 'echo "done $(date)"', ""])
    paths = []
    if per_run:
        for d in dirs:
            tag = name + "_" + "_".join(d.split("/")[-2:])
            p = os.path.join(DIAG, f"run_{tag}.sbatch")
            with open(p, "w") as f: f.write(jobtext(tag, [d]))
            paths.append(p)
    else:
        p = os.path.join(DIAG, f"run_{name}.sbatch")
        with open(p, "w") as f: f.write(jobtext(name, dirs))
        paths.append(p)
    print(f"wrote {len(paths)} sbatch for {name} ({len(dirs)} runs)")
    return paths

# T1: kjpaw + oncv PBE on QE 6.8 (~/Devel/QEF/q-e, my build, campaign-compatible)
d1 = gen("qe68_pbe", ["paw", "oncv"], "pbe", 100, 600)
QE68 = "/mnt/home/mmorales/Devel/QEF/q-e/build/CPU/bin/pw.x"
sbatch("qe68", d1, CAMPAIGN_MODS, QE68, walltime="00:40:00")

# T2: USPP + oncv self-consistent HF at MATCHED ecut100/400, nqx8, on q-e (7.4.1).
#     One job PER (pp,vol) -- HF+EXX is slow; the single-job version timed out.
d2 = gen("hf_matched", ["uspp", "oncv"], "hf", 100, 400, nqx=8)
QE741 = "/mnt/home/mmorales/Devel/QEF/q-e_7.0/build/CPU/bin/pw.x"
sbatch("hfm", d2, CAMPAIGN_MODS, QE741, walltime="02:00:00", per_run=True, nproc=32)
