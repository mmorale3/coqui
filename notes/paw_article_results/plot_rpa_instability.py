#!/usr/bin/env python3
"""Two-panel comparison of the PAW RPA instability between ABINIT and CoQui.

(a) Ec vs lattice constant, one colour per band count, ABINIT solid / CoQui dashed,
    with the norm-conserving n=500 curve as the physical reference.
(b) The instability metric: spread of Ec across the volume range vs band count.

Input: rpa_instability.csv written by harvest_abinit_rpa.py.
"""
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = sys.argv[1] if len(sys.argv) > 1 else "rpa_instability.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "rpa_instability.pdf"

# validated categorical slots (dataviz palette, light mode: all checks pass)
COL = {100: "#2a78d6", 250: "#eb6834", 500: "#1baf7a"}
NEUTRAL = "#8a8a85"
INK, INK2 = "#0b0b0b", "#52514e"
NBANDS = [100, 250, 500]

d = defaultdict(dict)          # (code, pseudo, nband) -> {alat: ec}
for r in csv.DictReader(open(CSV)):
    if r["variant"] != "base" or not r["ec"]:
        continue
    d[(r["code"], r["pseudo"], int(r["nband"]))][float(r["alat"])] = float(r["ec"])

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.3))

# ---------------- (a) Ec vs lattice constant ----------------
for nb in NBANDS:
    for code, style, mk in (("abinit", "-", "o"), ("coqui", "--", "s")):
        s = d.get((code, "jth_with_d", nb), {})
        if not s:
            continue
        xs = sorted(s)
        ys = [s[x] for x in xs]
        ax.plot(xs, ys, style, color=COL[nb], marker=mk, ms=4.5, lw=2,
                mfc=COL[nb] if code == "abinit" else "none", mew=1.6, zorder=3)
        if code == "coqui" and nb == 500:      # direct label on the runaway curve
            ax.annotate("CoQui n=500", (xs[0], ys[0]), textcoords="offset points",
                        xytext=(6, -12), color=INK2, fontsize=8.5)

nc = d.get(("coqui", "nc", 500), {})
if nc:
    xs = sorted(nc)
    ax.plot(xs, [nc[x] for x in xs], ":", color=NEUTRAL, lw=2, zorder=2)
    ax.annotate("norm-conserving, n=500", (xs[-1], nc[xs[-1]]), textcoords="offset points",
                xytext=(-4, 8), ha="right", color=INK2, fontsize=8.5)

ax.set_xlabel("lattice constant  $a$  (Bohr)", color=INK2)
ax.set_ylabel(r"RPA correlation energy  $E_c$  (Ha)", color=INK2)
ax.set_title("(a)  $E_c$ vs volume, PAW jth_with_d", color=INK, fontsize=10.5, loc="left")

handles = [Line2D([], [], color=COL[nb], lw=2, label=f"n = {nb}") for nb in NBANDS]
handles += [Line2D([], [], color=INK2, lw=2, ls="-", marker="o", ms=4.5, label="ABINIT"),
            Line2D([], [], color=INK2, lw=2, ls="--", marker="s", ms=4.5, mfc="none",
                   label="CoQui")]
ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)

# ---------------- (b) instability metric ----------------
for code, style, mk, lab in (("abinit", "-", "o", "ABINIT"), ("coqui", "--", "s", "CoQui")):
    xs, ys = [], []
    for nb in NBANDS:
        s = d.get((code, "jth_with_d", nb), {})
        if len(s) >= 2:
            xs.append(nb)
            ys.append(1000 * (max(s.values()) - min(s.values())))
    if xs:
        bx.plot(xs, ys, style, color=COL[500] if code == "abinit" else COL[250],
                marker=mk, ms=5, lw=2, mfc=(COL[500] if code == "abinit" else "none"),
                mew=1.6, zorder=3)
        bx.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points", xytext=(-6, 8),
                    ha="right", color=INK2, fontsize=9)

if nc:
    ref = 1000 * (max(nc.values()) - min(nc.values()))
    bx.axhline(ref, ls=":", color=NEUTRAL, lw=2, zorder=2)
    bx.annotate(f"norm-conserving ({ref:.1f} mHa)", (NBANDS[0], ref),
                textcoords="offset points", xytext=(2, 5), color=INK2, fontsize=8.5)

bx.set_yscale("log")
bx.set_xticks(NBANDS)
bx.set_xlabel("number of bands", color=INK2)
bx.set_ylabel(r"spread of $E_c$ over $a=10.05\!-\!10.55$  (mHa)", color=INK2)
bx.set_title("(b)  volume-dependence of the error", color=INK, fontsize=10.5, loc="left")

for a_ in (ax, bx):
    a_.grid(True, color="#e6e6e2", lw=0.8, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a_.spines[sp].set_color("#c9c9c4")
    a_.tick_params(colors=INK2, labelsize=9)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print(f"wrote {OUT} and {OUT.replace('.pdf', '.png')}")
