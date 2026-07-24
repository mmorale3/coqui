#!/usr/bin/env python3
"""Result B: ISDF-rank convergence of the THC V_H / V_x matrix elements.

Parses CONVCSV lines emitted by test_hamilt.cpp::run_isdf_threshold_convergence:

  CONVCSV,<tag>,<thresh>,<Np>,errVH_max,errVH_fro,relVH_fro,errVX_max,errVX_fro,relVX_fro

Produces:
  - paw_convergence_B.pdf : rel-Frobenius error vs Np (=N_mu), per tag, V_H and V_x
  - a LaTeX table (printed to stdout) of (Np, relVH_fro, relVX_fro) per tag

Usage: python3 plot_convergence_B.py B_isdf_threshold_convergence.log
matplotlib not on rusty -> pull the log locally and run here.
"""
import sys, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "B_isdf_threshold_convergence.log"
rows = defaultdict(list)  # tag -> list of (thresh, Np, relVH, relVX)
with open(src) as f:
    for line in f:
        if not line.startswith("CONVCSV,"):
            continue
        p = line.strip().split(",")
        # CONVCSV, tag, thresh, Np, eVH_max, eVH_fro, relVH_fro, eVX_max, eVX_fro, relVX_fro
        try:
            tag = p[1]
            thresh = float(p[2]); Np = int(p[3])
            relVH = float(p[6]); relVX = float(p[9])
        except (IndexError, ValueError):
            continue
        rows[tag].append((thresh, Np, relVH, relVX))

if not rows:
    print("No CONVCSV lines found in", src); sys.exit(1)

for tag in rows:
    rows[tag].sort(key=lambda r: r[1])  # by Np

# ---- figure ----
tags = list(rows.keys())
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=False)
markers = ["o", "s", "^", "D", "v", "P"]
for i, tag in enumerate(tags):
    Np = [r[1] for r in rows[tag]]
    relVH = [r[2] for r in rows[tag]]
    relVX = [r[3] for r in rows[tag]]
    m = markers[i % len(markers)]
    axes[0].loglog(Np, relVH, m + "-", label=tag, ms=5)
    axes[1].loglog(Np, relVX, m + "-", label=tag, ms=5)
axes[0].set_title(r"Hartree $V_H$")
axes[1].set_title(r"exchange $V_x$")
for ax in axes:
    ax.set_xlabel(r"$N_\mu$ (ISDF interpolation points)")
    ax.set_ylabel(r"rel. Frobenius error vs direct AE")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
fig.tight_layout()
out = "paw_convergence_B.pdf"
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)

# ---- LaTeX table ----
print("\n% --- per-tag convergence tables ---")
for tag in tags:
    print(f"% tag = {tag}")
    print(r"\begin{tabular}{rrll}")
    print(r"\toprule")
    print(r"thresh & $N_\mu$ & rel.\ err.\ $V_H$ & rel.\ err.\ $V_x$ \\ \midrule")
    for thresh, Np, relVH, relVX in rows[tag]:
        print(f"{thresh:.0e} & {Np} & {relVH:.2e} & {relVX:.2e} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
