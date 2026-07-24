#!/usr/bin/env python3
"""CeO2 PAW-ISDF-THC rank convergence for a heavy semicore+4f lanthanide oxide,
for BOTH augmented pseudopotential families (PAW and USPP): relative error of the
Hartree (Coulomb), exchange, and RPA-correlation energies vs the number of ISDF
interpolation points N_mu (soft ISDF threshold 3e-2 -> 1e-6, full-rank one-center).
Fluorite CeO2 at a=5.411 Ang, 3x3x3, N_b=100, Ce 4f-in-valence (n_h=32).
Reference = tightest N_mu."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = defaultdict(lambda: {"N":[], "H":[], "X":[], "R":[]})
for r in csv.DictReader(open("ceo2_conv.csv")):
    d = rows[r["pseudo"]]
    d["N"].append(int(r["N_smooth"])); d["H"].append(float(r["e_hartree"]))
    d["X"].append(float(r["e_exchange"])); d["R"].append(float(r["e_rpa"]))

panels = [("paw_fv","PAW (kjpaw)"), ("uspp_fv","USPP")]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
FL = 1.5e-6
for ax,(pp,title) in zip(axes, panels):
    d = rows[pp]
    N = np.array(d["N"]); iref = int(np.argmax(N))
    for key,c,mk,lab in [("H","tab:blue","o","Hartree (Coulomb)"),
                         ("X","tab:green","s","exchange"),
                         ("R","tab:red","D","RPA correlation")]:
        E = np.array(d[key]); rel = np.maximum(np.abs(E-E[iref])/abs(E[iref]), FL)
        m = np.arange(len(N)) != iref
        ax.plot(N[m], rel[m], mk+"-", color=c, ms=7, lw=1.6, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 2e-3)
    ax.set_xlabel(r"number of ISDF interpolation points  $N_\mu$")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.45)
axes[0].set_ylabel(r"relative energy error vs converged $N_\mu$")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle(r"CeO$_2$ (fluorite, Ce semicore+$4f$) PAW-ISDF-THC rank convergence,  $3\times3\times3$, $N_b{=}100$",
             y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig("ceo2_convergence.pdf", bbox_inches="tight")
fig.savefig("ceo2_convergence.png", dpi=150, bbox_inches="tight")
print("wrote ceo2_convergence.{pdf,png}\n")
for pp,title in panels:
    d=rows[pp]; N=np.array(d["N"]); iref=int(np.argmax(N))
    print(f"{title}: converged (N_mu={N[iref]})  H={d['H'][iref]:.4f}  X={d['X'][iref]:.4f}  RPA={d['R'][iref]:.5f} Ha")
