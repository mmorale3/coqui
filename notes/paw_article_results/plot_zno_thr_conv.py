#!/usr/bin/env python3
"""ZnO PAW-ISDF-THC rank convergence (analogue of the LiH Fig. 1, but for a
semicore-d system and including the RPA energy): relative error of the Hartree
(Coulomb), exchange, and RPA correlation energies vs the number of ISDF
interpolation points N_mu, as the soft ISDF threshold is tightened
(3e-2 -> 1e-6) at fixed full-rank one-center augmentation.  Zinc-blende ZnO,
3x3x3 k-mesh, N_b=100, kjpaw (Zn semicore 3s3p3d).  Reference = tightest N_mu.
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("zno_thr_conv.csv")))
Nmu = np.array([int(r["N_smooth"]) for r in rows], float)
eh  = np.array([float(r["e_hartree"])  for r in rows])
ex  = np.array([float(r["e_exchange"]) for r in rows])
er  = np.array([float(r["e_rpa"])      for r in rows])
iref = int(np.argmax(Nmu))                 # tightest threshold = reference
FL = 1.2e-6                                # ISDF reproducibility floor
rH = np.maximum(np.abs(eh - eh[iref]) / abs(eh[iref]), FL)
rX = np.maximum(np.abs(ex - ex[iref]) / abs(ex[iref]), FL)
rR = np.maximum(np.abs(er - er[iref]) / abs(er[iref]), FL)
m = np.arange(len(Nmu)) != iref            # drop the reference point itself

fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.plot(Nmu[m], rH[m], "o-", color="tab:blue",  ms=7, lw=1.7, label="Hartree (Coulomb)")
ax.plot(Nmu[m], rX[m], "s-", color="tab:green", ms=7, lw=1.7, label="exchange")
ax.plot(Nmu[m], rR[m], "D-", color="tab:red",   ms=7, lw=1.7, label="RPA correlation")
ax.set_yscale("log")
ax.set_xlabel(r"number of ISDF interpolation points  $N_\mu$")
ax.set_ylabel(r"relative energy error vs converged $N_\mu$")
ax.set_title(r"ZnO (kjpaw, Zn semicore $3d$),  $3\times3\times3$,  $N_b=100$")
ax.set_ylim(9e-7, 7e-4)
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.legend(fontsize=9.5, loc="upper right")
fig.tight_layout()
fig.savefig("zno_convergence.pdf", bbox_inches="tight")
fig.savefig("zno_convergence.png", dpi=150, bbox_inches="tight")
print("wrote zno_convergence.{pdf,png}\n")
print(f"{'N_mu':>6} {'relH':>10} {'relX':>10} {'relRPA':>10}   absRPA(Ha)")
for k in range(len(Nmu)):
    tag = "  <- ref" if k == iref else ""
    print(f"{int(Nmu[k]):>6} {rH[k]:>10.2e} {rX[k]:>10.2e} {rR[k]:>10.2e}   {er[k]:.6f}{tag}")
