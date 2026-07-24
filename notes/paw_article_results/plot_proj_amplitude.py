#!/usr/bin/env python3
"""Cause diagnostic for the no-d JTH failure: PAW projector-overlap amplitude
max|P_{i,a alpha}| per band vs band energy, Si 4x4x4, N_b=100.

In the SHARED s+p channels (both pseudos have them) the no-d JTH overlaps of
high-energy virtuals blow up (~40), because the d character of those states has
no d-projector to absorb it and floods the s,p projectors.  The d-complete kjpaw
keeps its s+p overlaps bounded (~1) and routes that content into the d-projector
(grey triangles) -- legitimate augmentation.  Through Delta V^a ~ P^4 the ~40x
inflation becomes a ~10^6 amplification of the spurious one-center term.
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(fn):
    e, sp, d = [], [], []
    for r in csv.DictReader(open(fn)):
        e.append(float(r["eig_eV_rel_vbm"]))
        sp.append(max(float(r["maxP_s"]), float(r["maxP_p"])))
        d.append(float(r["maxP_d"]))
    return np.array(e), np.array(sp), np.array(d)

ej, spj, dj = load("P_jth.csv")
ep, spp, dp = load("P_paw.csv")

fig, ax = plt.subplots(figsize=(6.4, 4.7))
ax.scatter(ej, spj, s=34, c="tab:red", marker="X",
           label=r"JTH ($l_{\max}=1$): $s,p$ overlap", zorder=5)
ax.scatter(ep, spp, s=30, c="tab:green", marker="o",
           label=r"kjpaw: $s,p$ overlap", zorder=4)
ax.scatter(ep, dp, s=26, facecolors="none", edgecolors="0.5", marker="^",
           label=r"kjpaw: $d$ overlap (correct augmentation)", zorder=3)

ax.axvline(0.0, color="0.7", ls="--", lw=1.0)
ax.text(0.6, 0.42, "VBM", color="0.5", fontsize=8, rotation=90, va="bottom")
ax.set_yscale("log")
ax.set_xlabel(r"band energy $\varepsilon - \varepsilon_{\mathrm{VBM}}$ (eV)")
ax.set_ylabel(r"max projector overlap  $\max_{a\alpha}|P_{i,a\alpha}|$")
ax.set_ylim(0.3, 70)
ax.set_xlim(-14, max(ej.max(), ep.max()) + 2)
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.annotate(r"no $d$-projector $\Rightarrow$" "\n" r"$d$ content floods $s,p$",
            xy=(ej[spj.argmax()], spj.max()), xytext=(ej[spj.argmax()] - 26, 33),
            fontsize=9, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(r"Si $4\times4\times4$, $N_b=100$:  projector amplitude vs band energy")
fig.tight_layout()
fig.savefig("jth_proj_amplitude.pdf", bbox_inches="tight")
fig.savefig("jth_proj_amplitude.png", dpi=150, bbox_inches="tight")
print("wrote jth_proj_amplitude.{pdf,png}")
hi = ej > 10
print(f"JTH  s,p: occ~{spj[ej<-2].max():.2f}  high-virt(>10eV)~{spj[hi].max():.1f}")
print(f"kjpaw s,p: occ~{spp[ep<-2].max():.2f}  high-virt(>10eV)~{spp[ep>10].max():.2f}")
print(f"kjpaw d  : high-virt(>10eV)~{dp[ep>10].max():.1f}")
