#!/usr/bin/env python3
"""Error in the Hartree (Coulomb), exchange, and RPA correlation energies of Si
vs truncation of the PAW one-center / augmentation basis, for a kjpaw PAW set,
3x3x3 k-grid, N_b=100.  The soft ISDF threshold is fixed (thresh=1e-5); only the
one-center tolerance paw_isdf_tol is varied, which sets the kept local rank
nlambda (full rank = nh^2 = 324 per atom).  Errors are vs the full-rank (nlambda
=324) reference.  nlambda=0 = augmentation entirely removed (smooth pseudo only).
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("si_oc_sweep.csv")))
nlam = np.array([int(r["nlambda"]) for r in rows])
eh   = np.array([float(r["e_hartree"]) for r in rows])
ex   = np.array([float(r["e_exchange"]) for r in rows])
erpa = np.array([float(r["e_rpa"]) for r in rows])
# full-rank reference = first row (nlambda=324)
i0 = int(np.argmax(nlam))
dH = np.abs(eh - eh[i0]); dX = np.abs(ex - ex[i0]); dR = np.abs(erpa - erpa[i0])
FLOOR = 3e-6
dH = np.maximum(dH, FLOOR); dX = np.maximum(dX, FLOOR); dR = np.maximum(dR, FLOOR)

fig, ax = plt.subplots(figsize=(6.6, 4.7))
ax.plot(nlam, dH, "o-", color="tab:blue",  ms=6, lw=1.6, label="Hartree (Coulomb)")
ax.plot(nlam, dX, "s-", color="tab:green", ms=6, lw=1.6, label="exchange")
ax.plot(nlam, dR, "D-", color="tab:red",   ms=6, lw=1.6, label="RPA correlation")
ax.axhline(1e-3, color="0.6", ls=":", lw=1.0)
ax.text(322, 1.15e-3, "1 mHa", color="0.45", fontsize=8, ha="right")
ax.axvline(324, color="0.75", ls="--", lw=1.0)
ax.text(324, 4e-6, " full rank\n $n_h^2$=324", color="0.45", fontsize=8, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xlim(-12, 340)
ax.set_ylim(FLOOR*0.8, 0.3)
ax.set_xlabel(r"kept one-center rank  $n_\lambda$  (per atom; full $=n_h^2=324$)")
ax.set_ylabel(r"$|\Delta E|$ vs full-rank one-center  (Ha)")
ax.set_title(r"Si kjpaw, $3\times3\times3$, $N_b=100$:  one-center compression error")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="lower left")
# annotate the no-augmentation endpoint
ax.annotate("no augmentation", xy=(0, dR[nlam==0][0]), xytext=(35, 0.13),
            fontsize=8, color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
fig.tight_layout()
fig.savefig("si_oc_compression.pdf", bbox_inches="tight")
fig.savefig("si_oc_compression.png", dpi=150, bbox_inches="tight")
print("wrote si_oc_compression.{pdf,png}\n")
print(f"{'nlambda':>8} {'frac':>6} {'dE_H':>10} {'dE_x':>10} {'dE_RPA':>10}")
for k in range(len(nlam)):
    print(f"{nlam[k]:>8} {nlam[k]/324:>6.2f} {dH[k]:>10.2e} {dX[k]:>10.2e} {dR[k]:>10.2e}")
