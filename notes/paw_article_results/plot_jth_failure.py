#!/usr/bin/env python3
"""Symptom of the JTH (l_max=1, no d-projector) PAW failure in Si RPA.
Two panels, RPA correlation vs N_b:
  (a) 3x3x3 mesh: the no-d JTH set runs away monotonically while every
      d-complete pseudo (NC ONCV/ccECP, USPP, kjpaw PAW) converges.
  (b) 4x4x4 mesh: the no-d set becomes catastrophic/erratic (~-2.4 Ha),
      whereas the d-complete pseudos stay well-behaved.
Cure shown against trusted d-complete pseudos (jth_with_d_v2 excluded).
"""
import csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = "si_rpa_proj.csv"
rows = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(src)):
    rows[r["pseudo"]][int(r["kgrid"])][int(r["nbnd"])] = float(r["rpa"])

# d-complete (trusted) pseudos + the no-d JTH set
good = {
    "oncv":  ("k",         "o", "-",  "ONCV (NC)"),
    "ccecp": ("0.45",      "s", "-",  "ccECP (NC)"),
    "uspp":  ("tab:blue",  "^", "--", r"USPP ($l=2$)"),
    "paw":   ("tab:green", "D", "-",  r"PAW kjpaw ($l=2$)"),
}
JTH = ("tab:red", "X", "--", r"JTH ($l_{\max}=1$, no $d$)")
order = ["oncv", "ccecp", "uspp", "paw"]

def series(pp, kp):
    d = sorted(rows[pp].get(kp, {}).items())
    return [n for n, _ in d], [e for _, e in d]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.8, 4.0))

for ax, kp, ylim, title in [
        (axA, 3, (-0.70, -0.355), r"(a) $3\times3\times3$ mesh"),
        (axB, 4, (-2.55, -0.30),  r"(b) $4\times4\times4$ mesh")]:
    for pp in order:
        c, mk, ls, lab = good[pp]
        xs, ys = series(pp, kp)
        if xs: ax.plot(xs, ys, color=c, marker=mk, ls=ls, ms=6, lw=1.5, label=lab)
    c, mk, ls, lab = JTH
    xs, ys = series("jth_lib", kp)
    if xs: ax.plot(xs, ys, color=c, marker=mk, ls=ls, ms=9, lw=2.0, mew=2.0, label=lab, zorder=6)
    ax.set_xlabel(r"number of bands $N_b$")
    ax.set_ylim(*ylim); ax.set_xlim(30, 270)
    ax.set_xticks([50, 100, 150, 250])
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
axA.set_ylabel("RPA correlation energy (Ha)")
axA.axhspan(-0.445, -0.42, color="0.85", zorder=0)         # well-behaved band
axB.axhspan(-0.445, -0.40, color="0.85", zorder=0)
axB.annotate("unphysical\nover-binding", xy=(150, -2.40), xytext=(95, -1.55),
             fontsize=9, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
axA.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("jth_failure.pdf", bbox_inches="tight")
fig.savefig("jth_failure.png", dpi=150, bbox_inches="tight")
print("wrote jth_failure.{pdf,png}")
# quick summary of the runaway
for kp in (2, 3, 4):
    xs, ys = series("jth_lib", kp)
    ox, oy = series("oncv", kp)
    print(f"kp{kp}: jth_lib " + " ".join(f"n{n}={e:.3f}" for n, e in zip(xs, ys)))
