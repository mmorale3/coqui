#!/usr/bin/env python3
"""Si RPA correlation vs 1/N_k with a double extrapolation:
  faint dots  = finite-N_b data
  star  (*)   = N_b->inf band extrapolation (linear in 1/N_b, N_b>=100) at each k-grid
  cross (X)   = N_b,N_k->inf, linear k-fit of the stars vs 1/N_k (using kp>=3)
jth_lib (no d-shell projector) treated separately / excluded.
"""
import csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = "si_rpa_proj.csv"
FIT_NMIN, KPLOT_MIN, KFIT_MIN = 100, 3, 4   # show stars for kp>=3; fit on kp>=4 only
KPS = [2, 3, 4, 5, 6]
NBNDS = [50, 100, 150, 250, 500]
rows = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(src)):
    rows[r["pseudo"]][int(r["kgrid"])][int(r["nbnd"])] = float(r["rpa"])

style = {
    "oncv":          ("k",          "ONCV (NC)"),
    "ccecp":         ("0.45",       "ccECP (NC)"),
    "uspp":          ("tab:blue",   "USPP (l=2)"),
    "paw":           ("tab:green",  "PAW kjpaw (l=2)"),
    "jth_with_d_v2": ("tab:orange", "PAW JTH+d (l=2)"),
}
order = ["oncv","ccecp","uspp","paw"]   # jth_with_d_v2 dropped (suspicious)

def band_extrap(kp, pp):
    pts = [(1.0/nb, r) for nb, r in rows[pp].get(kp, {}).items() if nb >= FIT_NMIN]
    if len(pts) < 2: return None
    fx = np.array([p[0] for p in pts]); fy = np.array([p[1] for p in pts])
    return float(np.polyfit(fx, fy, 1)[1])

fig, ax = plt.subplots(figsize=(7.0, 5.4))
converged = {}
KPLOT = [k for k in KPS if k >= KPLOT_MIN]               # stars shown: kp=3 and larger
KFIT  = [k for k in KPS if k >= KFIT_MIN]                # fit/intercept: kp=4 and larger
for pp in order:
    c, lab = style[pp]
    pf = [(1.0/kp**3, band_extrap(kp, pp)) for kp in KPLOT if band_extrap(kp, pp) is not None]
    ff = [(1.0/kp**3, band_extrap(kp, pp)) for kp in KFIT  if band_extrap(kp, pp) is not None]
    if len(ff) < 2: continue
    px = np.array([p[0] for p in pf]); py = np.array([p[1] for p in pf])
    fx = np.array([p[0] for p in ff]); fy = np.array([p[1] for p in ff])
    m, b = np.polyfit(fx, fy, 1)                                                              # fit on kp>=4
    ax.plot([0.0, fx.max()], [b, m*fx.max()+b], color=c, ls="-", lw=1.4, alpha=0.9, zorder=4) # fit
    ax.plot(px, py, color=c, marker="*", ms=13, ls="none", label=lab, zorder=5)               # N_b->inf (kp>=3)
    ax.plot([0.0], [b], color=c, marker="X", ms=13, mec="k", mew=0.7, ls="none", zorder=6)     # N_b,N_k->inf
    converged[pp] = b

ax.set_xlim(-0.0022, 0.041)
ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
ax.set_ylim(-0.462, -0.430)
ax.set_xlabel(r"$1/N_k$   ($N_k = n^3$;  kp = 3, 4, 5, 6)")
ax.set_ylabel("RPA correlation energy (Ha)")
ax.set_title(r"Si RPA $k$-extrapolation:  $\star\,N_b\!\to\!\infty$ (kp$\geq$3),  line = fit (kp$\geq$4),  $\times\,N_b,N_k\!\to\!\infty$")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig("si_rpa_vs_k.pdf", bbox_inches="tight"); fig.savefig("si_rpa_vs_k.png", dpi=150, bbox_inches="tight")
print("wrote si_rpa_vs_k.{pdf,png}\n")
print("Doubly-extrapolated (N_b,N_k -> inf) RPA correlation (Ha):")
for pp in order:
    if pp in converged: print(f"  {style[pp][1]:>18}: {converged[pp]:.4f}")
