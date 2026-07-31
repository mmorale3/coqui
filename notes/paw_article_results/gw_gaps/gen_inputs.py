#!/usr/bin/env python3
"""Generate ABINIT inputs for the PAW G0W0 band-gap campaign.

Two inputs per (material, k-mesh, cutoff):

  mf.abi  -- SCF + NSCF mean field, exported for abinit2coqui (WFK+DEN+POT).
             This is the file CoQui consumes.
  gw.abi  -- ABINIT's own PAW G0W0 on the SAME geometry and dataset, the
             head-to-head reference.

Three settings in mf.abi are REQUIRED by the CoQui path and are not free
choices:

  symmorphi 0   CoQui's PAW symmetry (hamilt::paw::build_atom_permutation_
                inverse) rejects non-symmorphic operations; the QE path passes
                force_symmorphic for the same reason.  Diamond is
                non-symmorphic in the 2-atom cell, so Si/C drop 48 -> 24 ops.
  kptopt 4      spatial symmetries only.  abinit2coqui writes use_trev=0, and
                an IBZ reduced WITH time reversal would not tile the grid under
                the stored operations (the converter checks this and refuses).
  istwfk *1     the converter needs unpacked coefficients at every k.

Cutoffs default to the dataset's own recommended "high" pw_ecut, scaled by
--ecut-scale so the basis-set axis can be swept.
"""

import argparse
import os
import xml.etree.ElementTree as ET

ANG = 1.8897261246257702        # Angstrom -> Bohr

FCC = "0.0 0.5 0.5   0.5 0.0 0.5   0.5 0.5 0.0"

# a_exp in Angstrom; gap character is what the harvester should report.
MATERIALS = {
    #        a(Ang)  species        xred(2nd atom)   nelec  gap
    "si":  dict(a=5.431,  species=["Si"],       typat=[1, 1], x2="0.25 0.25 0.25",
                nelec=8,  gap="indirect G->0.85X"),
    "c":   dict(a=3.567,  species=["C"],        typat=[1, 1], x2="0.25 0.25 0.25",
                nelec=8,  gap="indirect G->0.76X"),
    "sic": dict(a=4.358,  species=["Si", "C"],  typat=[1, 2], x2="0.25 0.25 0.25",
                nelec=8,  gap="indirect G->X"),
    "alp": dict(a=5.4635, species=["Al", "P"],  typat=[1, 2], x2="0.25 0.25 0.25",
                nelec=8,  gap="indirect G->X"),
    "mgo": dict(a=4.212,  species=["Mg", "O"],  typat=[1, 2], x2="0.5 0.5 0.5",
                nelec=16, gap="direct G->G"),
}

Z = {"H": 1, "C": 6, "O": 8, "Mg": 12, "Al": 13, "Si": 14, "P": 15}


def dataset_ecut(pspdir, species):
    """Max over species of the dataset's recommended high pw_ecut (Ha)."""
    ec = 0.0
    for s in species:
        f = os.path.join(pspdir, "%s.GGA_PBE-JTH.xml" % s)
        pw = ET.parse(f).getroot().find("pw_ecut")
        ec = max(ec, float(pw.get("high")))
    return ec


def common(mat, m, ecut, pawecutdg, ngkpt, pspdir):
    znucl = " ".join("%d" % Z[s] for s in m["species"])
    # ABINIT wants ONE quoted string with comma-separated names, not a list of
    # quoted strings -- with the latter it reads only the first and aborts with
    # "Not enough pseudopotentials in input `pseudos` string".
    pseudos = '"%s"' % ", ".join("%s.GGA_PBE-JTH.xml" % s for s in m["species"])
    return f"""# {mat}: {m['gap']}, a = {m['a']} Ang = {m['a'] * ANG:.5f} Bohr
acell 3*{m['a'] * ANG:.6f}
rprim  {FCC}
ntypat {len(m['species'])}  znucl {znucl}
natom 2  typat {' '.join(str(t) for t in m['typat'])}
xred  0.0 0.0 0.0   {m['x2']}

ecut {ecut:.1f}
pawecutdg {pawecutdg:.1f}
ixc 11                 # PBE

ngkpt {ngkpt} {ngkpt} {ngkpt}
nshiftk 1
shiftk 0.0 0.0 0.0

pp_dirpath "{pspdir}"
pseudos {pseudos}
"""


def mf_input(mat, m, ecut, pawecutdg, ngkpt, nband, pspdir):
    nocc = m["nelec"] // 2
    return common(mat, m, ecut, pawecutdg, ngkpt, pspdir) + f"""
# --- required by the CoQui path (see module docstring) ---
symmorphi 0            # CoQui PAW symmetry is symmorphic-only
kptopt 4               # spatial symmetries, NO time reversal
istwfk *1              # unpacked coefficients at every k

occopt 1
nstep 100
iomode 3

ndtset 2

# 1: SCF density
nband1  {max(nocc + 4, 12)}
tolvrs1 1.0d-12
prtden1 1
prtpot1 1

# 2: NSCF, the band set CoQui reads
iscf2    -2
getden2  1
tolwfr2  1.0d-14
nband2   {nband}
nbdbuf2  {max(4, nband // 20)}
prtwf2   1
prtden2  1
prtpot2  1
"""


def gw_input(mat, m, ecut, pawecutdg, ngkpt, nband, ecuteps, pspdir,
             gwcalctyp=1, nfreqim=16):
    nocc = m["nelec"] // 2
    # gwcalctyp selects how Sigma_c is evaluated in frequency, and it has to
    # match what CoQui does or the comparison is meaningless:
    #   0 = plasmon-pole (ABINIT default) -- a DIFFERENT approximation from
    #       CoQui's, and worth 0.3-0.5 eV on these materials.
    #   1 = analytic continuation from imaginary frequencies (Pade), which is
    #       what CoQui's evgw0 does. This is the apples-to-apples setting.
    #   2 = contour deformation, the numerically exact reference both should
    #       converge to.
    ppm = "" if gwcalctyp == 0 else "\nnfreqim%d   %d" % (3, nfreqim)
    return common(mat, m, ecut, pawecutdg, ngkpt, pspdir) + f"""
# ABINIT's own PAW G0W0 -- the head-to-head reference.
# Symmetry is left at ABINIT's default here: this run never goes through
# abinit2coqui, so the symmorphi/kptopt constraints do not apply and using
# ABINIT's full symmetry keeps the reference cheap.
kptopt 1
occopt 1
nstep 100
iomode 3
inclvkb 0              # forced for PAW
gwcalctyp {gwcalctyp}{ppm}

ndtset 4

nband1  {max(nocc + 4, 12)}
tolvrs1 1.0d-12
prtden1 1

iscf2    -2
getden2  1
tolwfr2  1.0d-14
nband2   {nband}
nbdbuf2  {max(4, nband // 20)}

# screening
optdriver3 3
getwfk3    2
nband3     {nband}
ecuteps3   {ecuteps:.1f}

# self-energy
optdriver4 4
getwfk4    2
getscr4    3
nband4     {nband}
ecutsigx4  {ecut:.1f}
gw_qprange4 -{max(2, nocc)}    # all occupied + this many empty
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="inputs")
    ap.add_argument("--pspdir",
                    default="/mnt/home/mmorales/ceph/CoQui/abinit/pseudos/JTH-PBE-v2.0")
    ap.add_argument("--materials", nargs="+", default=sorted(MATERIALS))
    ap.add_argument("--ngkpt", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--nband", type=int, nargs="+", default=[100])
    ap.add_argument("--ecut-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--dg-factor", type=float, default=2.0,
                    help="pawecutdg = dg_factor * ecut")
    ap.add_argument("--ecuteps-frac", type=float, default=0.75,
                    help="ABINIT ecuteps as a fraction of ecut")
    ap.add_argument("--gwcalctyp", type=int, default=1,
                    help="0=plasmon-pole, 1=analytic continuation (matches "
                         "CoQui), 2=contour deformation")
    ap.add_argument("--nfreqim", type=int, default=16,
                    help="imaginary frequencies for gwcalctyp 1/2")
    args = ap.parse_args()

    made = 0
    for mat in args.materials:
        m = MATERIALS[mat]
        base = dataset_ecut(args.pspdir if os.path.isdir(args.pspdir) else ".",
                            m["species"]) if os.path.isdir(args.pspdir) else 20.0
        for sc in args.ecut_scale:
            ecut = base * sc
            dg = ecut * args.dg_factor
            for nk in args.ngkpt:
                for nb in args.nband:
                    tag = "%s_k%d_n%d_e%g" % (mat, nk, nb, ecut)
                    d = os.path.join(args.outdir, tag)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, "mf.abi"), "w") as f:
                        f.write(mf_input(mat, m, ecut, dg, nk, nb, args.pspdir))
                    with open(os.path.join(d, "gw.abi"), "w") as f:
                        f.write(gw_input(mat, m, ecut, dg, nk, nb,
                                         ecut * args.ecuteps_frac, args.pspdir,
                                         gwcalctyp=args.gwcalctyp,
                                         nfreqim=args.nfreqim))
                    made += 1
    print("wrote %d input pairs under %s" % (made, args.outdir))


if __name__ == "__main__":
    main()
