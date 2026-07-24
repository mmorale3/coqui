#!/usr/bin/env python3
"""Compare one-center (PAW augmentation / K_a) ISDF compressibility for Si (light,
no semicore) vs ZnO (Zn semicore 3d).  For each system, error in the Hartree
(Coulomb), exchange, and RPA energies vs the kept fraction of the one-center
auxiliary basis (N_aug / N_aug^full), soft thresh fixed, only paw_isdf_tol varied.
Errors are vs the full-rank (largest N_aug) reference.  3x3x3, N_b=100, kjpaw.
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(fn):
    rows = list(csv.DictReader(open(fn)))
    Na = np.array([int(r["N_aug"]) for r in rows], float)
    eh = np.array([float(r["e_hartree"]) for r in rows])
    ex = np.array([float(r["e_exchange"]) for r in rows])
    er = np.array([float(r["e_rpa"]) for r in rows])
    i0 = int(np.argmax(Na))           # full-rank reference
    frac = Na / Na[i0]
    FL = 1e-6
    dH = np.maximum(np.abs(eh - eh[i0]), FL)
    dX = np.maximum(np.abs(ex - ex[i0]), FL)
    dR = np.maximum(np.abs(er - er[i0]), FL)
    o = np.argsort(frac)
    return frac[o], dH[o], dX[o], dR[o], int(Na[i0])

systems = [("si_oc_sweep.csv",  "Si (kjpaw, no semicore)"),
           ("zno_oc_sweep.csv", "ZnO (Zn semicore $3d$)")]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
for ax, (fn, title) in zip(axes, systems):
    frac, dH, dX, dR, Nfull = load(fn)
    ax.plot(frac, dH, "o-", color="tab:blue",  ms=6, lw=1.6, label="Hartree (Coulomb)")
    ax.plot(frac, dX, "s-", color="tab:green", ms=6, lw=1.6, label="exchange")
    ax.plot(frac, dR, "D-", color="tab:red",   ms=6, lw=1.6, label="RPA correlation")
    ax.axhline(1e-3, color="0.6", ls=":", lw=1.0)
    ax.text(0.02, 1.25e-3, "1 mHa", color="0.45", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel(r"kept fraction of one-center basis  $N_{\rm aug}/N_{\rm aug}^{\rm full}$")
    ax.set_title(f"{title}\n" + r"$N_{\rm aug}^{\rm full}=$" + f"{Nfull}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
axes[0].set_ylabel(r"$|\Delta E|$ vs full-rank one-center  (Ha)")
axes[0].set_ylim(7e-7, 60)
axes[0].legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig("si_zno_oc_compare.pdf", bbox_inches="tight")
fig.savefig("si_zno_oc_compare.png", dpi=150, bbox_inches="tight")
print("wrote si_zno_oc_compare.{pdf,png}\n")
for fn, title in systems:
    frac, dH, dX, dR, Nfull = load(fn)
    print(f"== {title} (N_full={Nfull}) ==")
    print(f"  {'frac':>5} {'dE_H':>10} {'dE_x':>10} {'dE_RPA':>10}")
    for k in range(len(frac)-1, -1, -1):
        print(f"  {frac[k]:>5.2f} {dH[k]:>10.2e} {dX[k]:>10.2e} {dR[k]:>10.2e}")
