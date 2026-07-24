#!/usr/bin/env python3
"""ZnO RPA correlation energy vs 1/N_b at kp3, showing it does NOT reach the
asymptotic linear-in-1/N_b regime even at N_b=2000 (the curve keeps steepening
toward 1/N_b->0 instead of flattening to a straight line).  Combines the campaign
(ecutwfc=90, N_b 50-500) and probe (ecutwfc=120, N_b 1000-2000) points -- they
agree at N_b=500 (THC cutoff barely matters for RPA, <0.3%)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# kp3 RPA(Ha): N_b -> E   (n<=500 campaign ecut90; n>=1000 probe ecut120)
data = {
  "oncv": {50:-0.4799,100:-0.6917,150:-0.8171,250:-0.9793,500:-1.1893,1000:-1.3668,2000:-1.4866},
  "uspp": {50:-0.5832,100:-0.8373,150:-0.9681,250:-1.1177,500:-1.2546,1000:-1.3412,1500:-1.3778},
  "paw":  {50:-0.7011,100:-0.9655,150:-1.0900,250:-1.2252,500:-1.3446,1000:-1.4234,1500:-1.4601},
}
style = {"oncv":("k","o","ONCV (NC)"), "uspp":("tab:blue","^","USPP"),
         "paw":("tab:green","D","PAW (kjpaw)")}

fig, ax = plt.subplots(figsize=(7.0,5.0))
for pp,(c,mk,lab) in style.items():
    d = sorted(data[pp].items())
    x = np.array([1.0/nb for nb,_ in d]); y = np.array([e for _,e in d])
    ax.plot(x, y, color=c, marker=mk, ms=7, lw=1.6, label=lab)
    # naive linear extrapolation through the two highest-N_b points -> 1/N_b=0
    m,b = np.polyfit(x[-2:], y[-2:], 1)
    ax.plot([0.0, x[-2]], [b, m*x[-2]+b], color=c, ls=":", lw=1.1, alpha=0.7)
    ax.plot([0.0],[b], color=c, marker="x", ms=9, mew=2)

# top-axis band labels
for nb in [2000,1000,500,250,100,50]:
    ax.axvline(1.0/nb, color="0.9", lw=0.7, zorder=0)
ax.set_xlim(-0.001, 0.021)
ax.set_xlabel(r"$1/N_b$")
ax.set_ylabel("RPA correlation energy (Ha)")
ax.set_title(r"ZnO RPA vs $1/N_b$ (kp3): no asymptotic regime even at $N_b{=}2000$")
ax.grid(True, ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper right")
ax.annotate("curve keeps steepening\n(no linear regime)", xy=(0.0006,-1.47),
            xytext=(0.004,-1.30), fontsize=9, color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5"))
# secondary x ticks as N_b
ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
ticks=[1/50,1/100,1/250,1/500,1/1000,1/2000]
ax2.set_xticks(ticks); ax2.set_xticklabels(["50","100","250","500","1000","2000"], fontsize=8)
ax2.set_xlabel(r"$N_b$", fontsize=9)
fig.tight_layout()
fig.savefig("zno_band_conv.pdf", bbox_inches="tight")
fig.savefig("zno_band_conv.png", dpi=150, bbox_inches="tight")
print("wrote zno_band_conv.{pdf,png}")
for pp in data:
    d=sorted(data[pp].items());
    print(pp, "  ".join(f"n{nb}:{e:.3f}" for nb,e in d))
