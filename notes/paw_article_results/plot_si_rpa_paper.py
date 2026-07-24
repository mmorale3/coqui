#!/usr/bin/env python3
"""Two-panel Si RPA convergence figure for the PAW-ISDF paper:
  (a) basis-set (band) convergence at the densest mesh (kp=6): RPA vs 1/N_b,
      linear fit on N_b>=100 extrapolated to 1/N_b->0 (stars).
  (b) k-point convergence: band-extrapolated RPA vs 1/N_k; stars shown for
      kp>=3, linear fit on kp>=4 (asymptotic regime) -> 1/N_k->0 intercept (X).
jth_with_d_v2 excluded (suspect).
"""
import csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = "si_rpa_proj.csv"
FIT_NMIN, KPLOT_MIN, KFIT_MIN, KP_BAND = 100, 3, 4, 6
KPS = [2, 3, 4, 5, 6]
rows = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(src)):
    rows[r["pseudo"]][int(r["kgrid"])][int(r["nbnd"])] = float(r["rpa"])

style = {
    "oncv":  ("k",         "o", "-",  "ONCV (NC)"),
    "ccecp": ("0.45",      "s", "-",  "ccECP (NC)"),
    "uspp":  ("tab:blue",  "^", "--", r"USPP ($l=2$)"),
    "paw":   ("tab:green", "D", "-",  r"PAW kjpaw ($l=2$)"),
}
order = ["oncv", "ccecp", "uspp", "paw"]

def band_fit(pp, kp):
    pts = [(1.0/nb, r) for nb, r in rows[pp].get(kp, {}).items() if nb >= FIT_NMIN]
    if len(pts) < 2: return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return np.polyfit(x, y, 1)            # (slope, intercept)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.8, 3.9))

# --- Panel (a): band convergence at the densest mesh ---
for pp in order:
    c, mk, ls, lab = style[pp]
    d = sorted(rows[pp].get(KP_BAND, {}).items())
    if not d: continue
    xb = np.array([1.0/nb for nb, _ in d]); yb = np.array([r for _, r in d])
    axL.plot(xb, yb, color=c, marker=mk, ls=ls, ms=6, lw=1.5, label=lab)
    fit = band_fit(pp, KP_BAND)
    if fit is not None:
        m, b = fit
        xmax = max(1.0/nb for nb in rows[pp][KP_BAND] if nb >= FIT_NMIN)
        axL.plot([0.0, xmax], [b, m*xmax + b], color=c, ls=":", lw=1.0, alpha=0.8)
        axL.plot([0.0], [b], color=c, marker="*", ms=12, ls="none", zorder=5)
axL.set_xlim(-0.0009, 0.0215); axL.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
axL.set_xticklabels(["0", ".005", ".010", ".015", ".020"])
axL.set_ylim(-0.452, -0.368)
axL.set_xlabel(r"$1/N_b$"); axL.set_ylabel("RPA correlation energy (Ha)")
axL.set_title(r"(a) band convergence,  $6\times6\times6$")
axL.grid(True, ls=":", alpha=0.5)
axL.legend(fontsize=8, loc="lower right")

# --- Panel (b): k-point convergence of the band-extrapolated values ---
KPLOT = [k for k in KPS if k >= KPLOT_MIN]
KFIT  = [k for k in KPS if k >= KFIT_MIN]
conv = {}
for pp in order:
    c, mk, ls, lab = style[pp]
    pf = [(1.0/k**3, band_fit(pp, k)[1]) for k in KPLOT if band_fit(pp, k) is not None]
    ff = [(1.0/k**3, band_fit(pp, k)[1]) for k in KFIT  if band_fit(pp, k) is not None]
    if len(ff) < 2: continue
    px = np.array([p[0] for p in pf]); py = np.array([p[1] for p in pf])
    fx = np.array([p[0] for p in ff]); fy = np.array([p[1] for p in ff])
    m, b = np.polyfit(fx, fy, 1)
    axR.plot([0.0, fx.max()], [b, m*fx.max() + b], color=c, ls="-", lw=1.4, alpha=0.9, zorder=4)
    axR.plot(px, py, color=c, marker="*", ms=12, ls="none", zorder=5)
    axR.plot([0.0], [b], color=c, marker="X", ms=12, mec="k", mew=0.6, ls="none", zorder=6)
    conv[pp] = b
axR.set_xlim(-0.0022, 0.041); axR.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
axR.set_ylim(-0.462, -0.430)
axR.set_xlabel(r"$1/N_k$   ($N_k=n^3$)"); axR.set_ylabel("RPA correlation energy (Ha)")
axR.set_title(r"(b) $k$-point convergence,  $N_b\!\to\!\infty$")
axR.grid(True, ls=":", alpha=0.5)

fig.tight_layout()
fig.savefig("si_rpa_paper.pdf", bbox_inches="tight")
fig.savefig("si_rpa_paper.png", dpi=150, bbox_inches="tight")
print("wrote si_rpa_paper.{pdf,png}")
print("Doubly-extrapolated (N_b,N_k -> inf) RPA correlation (Ha):")
for pp in order:
    if pp in conv: print(f"  {style[pp][3]:>22}: {conv[pp]:.4f}")
