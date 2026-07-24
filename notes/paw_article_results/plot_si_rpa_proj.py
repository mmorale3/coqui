#!/usr/bin/env python3
"""RPA correlation vs nbnd (or 1/nbnd) for Si across NCPP/USPP/PAW.
With `inv`: x = 1/N_b, plus a linear fit on the high-N_b points extrapolated to
1/N_b -> 0 (stars mark the N_b->inf intercepts; converged values printed).
jth_lib (no d-shell projector) is treated separately and excluded here.
"""
import sys, csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "si_rpa_proj.csv"
inv = "inv" in sys.argv[2:]
FIT_NMIN = 100                       # fit/extrapolate using N_b >= this
only_kp = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("kp")), None)  # e.g. kp6 -> single panel

rows = defaultdict(list)
for r in csv.DictReader(open(src)):
    rows[(int(r["kgrid"]), r["pseudo"], r["type"])].append((int(r["nbnd"]), float(r["rpa"])))
for k in rows: rows[k].sort()

kgrids = sorted({k for (k, _, _) in rows})
if only_kp is not None: kgrids = [k for k in kgrids if k == only_kp]
style = {
    "oncv":          ("k",          "-",  "o", "ONCV (NC)"),
    "ccecp":         ("0.45",       "-",  "s", "ccECP (NC)"),
    "uspp":          ("tab:blue",   "--", "^", "USPP (l=2)"),
    "paw":           ("tab:green",  "-",  "D", "PAW kjpaw (l=2)"),
    "jth_with_d_v2": ("tab:orange", "-",  "v", "PAW JTH+d (l=2)"),
}
order = ["oncv","ccecp","uspp","paw"]   # jth_with_d_v2 dropped (suspicious)
YLO, YHI = -0.455, -0.365

fig, axes = plt.subplots(1, len(kgrids), figsize=(4.7*len(kgrids), 4.3), squeeze=False, sharey=True)
intercepts = {}
for col, kg in enumerate(kgrids):
    ax = axes[0][col]
    for pp in order:
        key = next((k for k in rows if k[0]==kg and k[1]==pp), None)
        if key is None: continue
        data = rows[key]
        xs = [(1.0/d[0] if inv else d[0]) for d in data]; ys = [d[1] for d in data]
        c, ls, mk, lab = style[pp]
        ax.plot(xs, ys, color=c, ls=ls, marker=mk, ms=6, lw=1.6, label=lab)
        if inv:
            fp = [(1.0/n, r) for (n, r) in data if n >= FIT_NMIN]
            if len(fp) >= 2:
                fx = np.array([p[0] for p in fp]); fy = np.array([p[1] for p in fp])
                m, b = np.polyfit(fx, fy, 1)
                ax.plot([0.0, fx.max()], [b, m*fx.max()+b], color=c, ls=":", lw=1.0, alpha=0.8)
                ax.plot([0.0], [b], color=c, marker="*", ms=11, ls="none", zorder=5)
                intercepts[(kg, pp)] = b
    ax.set_ylim(YLO, YHI)
    if inv:
        ax.set_xlim(-0.0008, 0.021)
        ax.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
        ax.set_xticklabels(["0", ".005", ".010", ".015", ".020"])
        ax.set_xlabel(r"$1/N_b$")
    else:
        ax.set_xlabel("number of bands $N_b$")
    ax.set_title(f"Si, {kg}×{kg}×{kg} k-grid")
    ax.grid(True, ls=":", alpha=0.5)
    if col == 0:
        ax.set_ylabel("RPA correlation energy (Ha)")
        ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
base = ("si_rpa_band_kp%d" % only_kp) if only_kp is not None else ("si_rpa_proj_inv" if inv else "si_rpa_proj")
fig.savefig(base + ".pdf", bbox_inches="tight"); fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
print(f"wrote {base}.{{pdf,png}}")

if inv and intercepts:
    print(f"\nExtrapolated N_b->inf RPA correlation (Ha), linear fit on N_b>={FIT_NMIN}:")
    print(f"  {'kp':>3} " + " ".join(f"{pp:>14}" for pp in order))
    for kg in kgrids:
        print(f"  {kg:>3} " + " ".join(f"{intercepts.get((kg,pp), float('nan')):>14.4f}" for pp in order))
