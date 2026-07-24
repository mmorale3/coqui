#!/usr/bin/env python3
"""Fig.1 (Result B): ISDF-rank convergence vs compression ratio alpha = N_mu/N_orb.

Parses CONVCSV lines emitted by run_isdf_threshold_convergence with tags of the
form LiH_<PP>_n<NBND>, e.g. LiH_PAW_n250:

  CONVCSV,<tag>,<thresh>,<Np>,eVH_max,eVH_fro,relVH_fro,eVX_max,eVX_fro,relVX_fro

x-axis is alpha = Np / NBND (interpolation points per band); curves are grouped
by pseudization (color) and band count (line style). Produces
paw_convergence_w150_fixed.{pdf,png} and prints per-(PP,nbnd) LaTeX tables.

Usage: python3 plot_convergence_alpha.py B2_conv.log
"""
import sys, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "B2_conv.log"
# rows[(pp,nbnd)] = list of (alpha, Np, relVH, relVX)
rows = defaultdict(list)
tag_re = re.compile(r"LiH_(?P<pp>[A-Za-z]+)_n(?P<nb>\d+)")
with open(src) as f:
    for line in f:
        if not line.startswith("CONVCSV,"):
            continue
        p = line.strip().split(",")
        m = tag_re.search(p[1])
        if not m:
            continue
        pp = m.group("pp"); nb = int(m.group("nb"))
        try:
            Np = int(p[3]); relVH = float(p[6]); relVX = float(p[9])
        except (IndexError, ValueError):
            continue
        rows[(pp, nb)].append((Np / nb, Np, relVH, relVX))

if not rows:
    print("No CONVCSV lines parsed from", src); sys.exit(1)
for k in rows:
    rows[k].sort(key=lambda r: r[0])

pp_color = {"NCPP": "tab:blue", "USPP": "tab:orange", "PAW": "tab:green"}
nb_style = {100: dict(ls="-", marker="o"), 250: dict(ls="--", marker="s")}
pp_order = ["NCPP", "USPP", "PAW"]
nbnds = sorted({nb for (_, nb) in rows})

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for col, key in enumerate(("relVH", "relVX")):
    ax = axes[col]
    for pp in pp_order:
        for nb in nbnds:
            if (pp, nb) not in rows:
                continue
            data = rows[(pp, nb)]
            alpha = [d[0] for d in data]
            y = [d[2] if key == "relVH" else d[3] for d in data]
            st = nb_style.get(nb, dict(ls="-", marker="o"))
            ax.loglog(alpha, y, color=pp_color.get(pp, "gray"),
                      ls=st["ls"], marker=st["marker"], ms=4.5,
                      label=f"{pp}, {nb} bands")
    ax.set_xlabel(r"$\alpha = N_\mu / N_{\rm orb}$")
    ax.set_ylabel("rel. Frobenius error vs direct AE")
    ax.grid(True, which="both", ls=":", alpha=0.5)
axes[0].set_title(r"Hartree $V_H$")
axes[1].set_title(r"exchange $V_x$")
axes[1].legend(fontsize=7, ncol=1)
fig.tight_layout()
fig.savefig("paw_convergence_w150_fixed.pdf"); fig.savefig("paw_convergence_w150_fixed.png", dpi=150)
print("wrote paw_convergence_w150_fixed.pdf")

print("\n% --- per (PP, nbnd) convergence tables ---")
for pp in pp_order:
    for nb in nbnds:
        if (pp, nb) not in rows:
            continue
        print(f"% {pp}, {nb} bands")
        for alpha, Np, relVH, relVX in rows[(pp, nb)]:
            print(f"%   alpha={alpha:6.2f}  Np={Np:5d}  relVH={relVH:.2e}  relVX={relVX:.2e}")
