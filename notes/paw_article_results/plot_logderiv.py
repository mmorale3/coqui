#!/usr/bin/env python3
"""Cause diagnostic 2 (atompaw logarithmic derivatives) for the no-d JTH failure.
Columns of atompaw logderiv.l:  E, AE, PAW, ... (official plotlogderiv uses 1:2 vs 1:3).
 (a) l=2 (d): the no-d JTH PAW scattering departs from all-electron above the valence
     region (no d-projector -> only the local potential scatters d waves); adding a
     d-projector (same construction) restores agreement.
 (b) l=0,1 (s,p): where the JTH set HAS projectors, PAW reproduces AE -- control.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(fn):
    d = np.loadtxt(fn)
    return d[:, 0], d[:, 1], d[:, 2]      # E, AE, PAW

def mask_poles(y, jump=4.0):
    y = y.copy()
    dy = np.abs(np.diff(y))
    y[1:][dy > jump] = np.nan
    return y

E, ae2, paw2_nod = load("ld_nod_l2.dat")
_, _,  paw2_wd  = load("ld_withd_l2.dat")
E0, ae0, paw0 = load("ld_nod_l0.dat")
E1, ae1, paw1 = load("ld_nod_l1.dat")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.1))

# (a) l=2
axL.plot(E, mask_poles(ae2), "k-", lw=2.0, label="all-electron", zorder=3)
axL.plot(E, mask_poles(paw2_nod), color="tab:red", ls="--", lw=2.0,
         label=r"JTH ($l_{\max}=1$, no $d$)", zorder=4)
axL.plot(E, mask_poles(paw2_wd), color="tab:green", ls=":", lw=2.2,
         label=r"JTH $+\,d$ projector", zorder=5)
axL.axvspan(0.0, 5.0, color="0.92", zorder=0)
axL.text(2.5, 3.6, "RPA virtual\nwindow", ha="center", va="top", fontsize=8, color="0.4")
axL.set_xlim(-5, 5); axL.set_ylim(-4.5, 4.5)
axL.set_xlabel("scattering energy (Ry)")
axL.set_ylabel(r"log derivative  $r\,\psi'/\psi\;|_{r_c}$")
axL.set_title(r"(a) $l=2$ ($d$) channel")
axL.grid(True, ls=":", alpha=0.5); axL.legend(fontsize=8.5, loc="lower left")

# (b) l=0,1 control (no-d JTH)
axR.plot(E0, mask_poles(ae0), "k-", lw=1.8, label=r"AE, $l=0$ ($s$)")
axR.plot(E0[::3], paw0[::3], "o", color="tab:red", ms=4, label=r"PAW, $l=0$")
axR.plot(E1, mask_poles(ae1), "-", color="0.5", lw=1.8, label=r"AE, $l=1$ ($p$)")
axR.plot(E1[::3], paw1[::3], "^", color="tab:blue", ms=4, label=r"PAW, $l=1$")
axR.set_xlim(-5, 5); axR.set_ylim(-4.5, 4.5)
axR.set_xlabel("scattering energy (Ry)")
axR.set_title(r"(b) $l=0,1$ ($s,p$): projectors present")
axR.grid(True, ls=":", alpha=0.5); axR.legend(fontsize=8, loc="lower left", ncol=2)

fig.tight_layout()
fig.savefig("jth_logderiv.pdf", bbox_inches="tight")
fig.savefig("jth_logderiv.png", dpi=150, bbox_inches="tight")
print("wrote jth_logderiv.{pdf,png}")
# quantify l=2 AE-PAW mismatch over the virtual window E in [0,5]
win = (E >= 0) & (E <= 5)
rms_nod = np.sqrt(np.nanmean((ae2[win] - paw2_nod[win]) ** 2))
rms_wd  = np.sqrt(np.nanmean((ae2[win] - paw2_wd[win]) ** 2))
print(f"l=2 AE-PAW RMS over E in [0,5] Ry:  no-d JTH = {rms_nod:.3f}   JTH+d = {rms_wd:.3f}")
