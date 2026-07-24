#!/usr/bin/env python3
"""Cause of the no-d JTH failure, 2-panel appendix figure:
 (a) PAW projector-overlap amplitude max|P_{i,a alpha}| vs band energy (Si 4x4x4,
     N_b=100, from existing h5): the no-d JTH s,p overlaps of high virtuals blow up
     (~40) because the d content has no d-projector and floods s,p; kjpaw keeps s,p
     bounded (~1) and routes it into the d-projector (grey).  Via DeltaV~P^4.
 (b) atompaw AE-vs-PAW logarithmic derivatives of the no-d JTH set: in s,p (projectors
     present) PAW reproduces AE; in d (no projector) it departs from AE across the RPA
     virtual window.  (Cure curve omitted -- shown by the kjpaw curve in (a) and the
     RPA convergence figure.)
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- panel (a): projector amplitude ----------
def loadP(fn):
    e, sp, d = [], [], []
    for r in csv.DictReader(open(fn)):
        e.append(float(r["eig_eV_rel_vbm"]))
        sp.append(max(float(r["maxP_s"]), float(r["maxP_p"])))
        d.append(float(r["maxP_d"]))
    return np.array(e), np.array(sp), np.array(d)

ej, spj, _ = loadP("P_jth.csv")
ep, spp, dp = loadP("P_paw.csv")

# ---------- panel (b): log-derivatives ----------
def loadL(fn):
    a = np.loadtxt(fn)
    return a[:, 0], a[:, 1], a[:, 2]      # E(Ry), AE, PAW

def mask_poles(y, jump=4.0):
    y = y.copy(); y[1:][np.abs(np.diff(y)) > jump] = np.nan; return y

E0, ae0, pw0 = loadL("ld_nod_l0.dat")
E1, ae1, pw1 = loadL("ld_nod_l1.dat")
E2, ae2, pw2 = loadL("ld_nod_l2.dat")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.3, 4.2))

# (a)
axA.scatter(ej, spj, s=32, c="tab:red", marker="X",
            label=r"JTH ($l_{\max}\!=\!1$): $s,p$", zorder=5)
axA.scatter(ep, spp, s=28, c="tab:green", marker="o", label=r"kjpaw: $s,p$", zorder=4)
axA.scatter(ep, dp, s=24, facecolors="none", edgecolors="0.5", marker="^",
            label=r"kjpaw: $d$ (correct aug.)", zorder=3)
axA.axvline(0.0, color="0.75", ls="--", lw=1.0)
axA.set_yscale("log"); axA.set_ylim(0.3, 70); axA.set_xlim(-14, max(ej.max(), ep.max()) + 2)
axA.set_xlabel(r"$\varepsilon - \varepsilon_{\mathrm{VBM}}$ (eV)")
axA.set_ylabel(r"max projector overlap $\max_{a\alpha}|P_{i,a\alpha}|$")
axA.set_title(r"(a) projector amplitude")
axA.grid(True, which="both", ls=":", alpha=0.45)
axA.annotate(r"no $d$ $\Rightarrow$ floods $s,p$", xy=(ej[spj.argmax()], spj.max()),
             xytext=(ej[spj.argmax()] - 30, 30), fontsize=8.5, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.1))
axA.legend(fontsize=8, loc="upper left")

# (b)  feature l=2 (failure) vs l=1 (control); l=0 also agrees (omitted for clarity)
axB.axvspan(0.0, 5.0, color="0.93", zorder=0)
axB.text(2.6, 4.1, "RPA virtual\nwindow", ha="center", va="top", fontsize=8, color="0.45")
# l=1 control (projector present -> PAW tracks AE)
axB.plot(E1, mask_poles(ae1), color="0.6", ls="-", lw=1.4, zorder=2)
axB.plot(E1[::3], pw1[::3], color="0.6", ls="none", marker="o", mfc="none", ms=3.8,
         label=r"$l=1$ ($p$): PAW $\equiv$ AE", zorder=3)
# l=2 failure (no d-projector -> PAW departs from AE)
axB.plot(E2, mask_poles(ae2), "k-", lw=2.0, zorder=4, label=r"$l=2$ ($d$): all-electron")
axB.plot(E2, mask_poles(pw2), color="tab:red", ls="--", lw=2.2, zorder=5,
         label=r"$l=2$ ($d$): PAW (no $d$-proj.)")
axB.annotate(r"$d$-channel mismatch", xy=(2.3, -1.5), xytext=(-4.6, -3.2),
             fontsize=8.5, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.1))
axB.set_xlim(-5, 5); axB.set_ylim(-4.5, 4.5)
axB.set_xlabel("scattering energy (Ry)")
axB.set_ylabel(r"log derivative  $r\,\psi'/\psi\,|_{r_c}$")
axB.set_title(r"(b) log derivatives (no-$d$ JTH)")
axB.grid(True, ls=":", alpha=0.5)
axB.legend(fontsize=8, loc="lower left")

fig.tight_layout()
fig.savefig("jth_causes.pdf", bbox_inches="tight")
fig.savefig("jth_causes.png", dpi=150, bbox_inches="tight")
print("wrote jth_causes.{pdf,png}")
