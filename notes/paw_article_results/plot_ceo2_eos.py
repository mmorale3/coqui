#!/usr/bin/env python3
"""CeO2 RPA@PBE equation of state -> Birch-Murnaghan fit -> a0, B0, for PAW vs USPP.
E_total(V) = CoQui 'Total energy'(Ha) + Ewald(Ry)/2.  V = a_Bohr^3/4 (FCC primitive,
1 CeO2 f.u.).  Reference RPA(VASP/PAW): a0=5.421 Ang, B0=202 GPa (JPCL 2021);
expt a0=5.411 Ang.  Usage: plot_ceo2_eos.py [csv]
"""
import sys, csv
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANG2BOHR = 1.8897259886
HA_BOHR3_TO_GPA = 29421.02648
src = sys.argv[1] if len(sys.argv) > 1 else "ceo2_eos.csv"

data = defaultdict(dict)   # tag -> {a(Ang): E_total(Ha)}
for r in csv.DictReader(open(src)):
    try: data[r["tag"]][float(r["a"])] = float(r["e_total_Ha"])
    except (ValueError, KeyError): pass

def bm3(V, E0, V0, B0, Bp):
    t = (V0/V)**(2.0/3.0) - 1.0
    return E0 + 9.0*V0*B0/16.0 * (t**3*Bp + t**2*(6.0 - 4.0*(V0/V)**(2.0/3.0)))

def fit_eos(avals, evals):
    a = np.array(avals); E = np.array(evals)
    aB = a*ANG2BOHR; V = aB**3/4.0
    # initial guess from quadratic in V
    c = np.polyfit(V, E, 2)
    V0g = -c[1]/(2*c[0]); E0g = np.polyval(c, V0g); B0g = 2*c[0]*V0g
    try:
        from scipy.optimize import curve_fit
        p,_ = curve_fit(bm3, V, E, p0=[E0g, V0g, max(B0g,1e-4), 4.0], maxfev=20000)
        E0,V0,B0,Bp = p
    except Exception:
        E0,V0,B0,Bp = E0g,V0g,B0g,4.0   # quadratic fallback
    a0 = (4.0*V0)**(1.0/3.0)/ANG2BOHR
    B0_GPa = B0*HA_BOHR3_TO_GPA
    return a0, B0_GPa, (E0,V0,B0,Bp)

style = {"paw":("tab:green","D","PAW (kjpaw, 4f-val), n250"),
         "uspp":("tab:blue","^","USPP (4f-val), n250"),
         "paw100":("0.55","o","PAW, n100")}
fig, ax = plt.subplots(figsize=(7.0,5.2))
print(f"{'config':>8} {'a0(Ang)':>9} {'B0(GPa)':>9}   (lit RPA: 5.421/202; expt a=5.411)")
for tag in ["paw","uspp","paw100"]:
    if tag not in data or len(data[tag])<4: continue
    items = sorted(data[tag].items())
    a = [x for x,_ in items]; E = [y for _,y in items]
    a0,B0,_ = fit_eos(a,E)
    c,mk,lab = style[tag]
    E0 = min(E)
    ax.plot(a, [(e-E0)*1000 for e in E], mk, color=c, ms=7, label=f"{lab}: a0={a0:.3f}, B0={B0:.0f}")
    # smooth BM curve
    aa = np.linspace(min(a)-0.02, max(a)+0.02, 100)
    _,_,(pE0,pV0,pB0,pBp) = fit_eos(a,E)
    VV = (aa*ANG2BOHR)**3/4.0
    ax.plot(aa, (bm3(VV,pE0,pV0,pB0,pBp)-E0)*1000, "-", color=c, lw=1.3, alpha=0.8)
    print(f"{tag:>8} {a0:>9.3f} {B0:>9.1f}")
ax.axvline(5.421, color="k", ls=":", lw=1.2); ax.text(5.421,ax.get_ylim()[1]*0.9," RPA lit",fontsize=8)
ax.axvline(5.411, color="0.5", ls="--", lw=1.0); ax.text(5.411,ax.get_ylim()[1]*0.78," expt",fontsize=8,color="0.5")
ax.set_xlabel(r"lattice constant $a$ (\AA)".replace("\\AA","Å"))
ax.set_ylabel(r"RPA total energy $-$ min (meV / f.u.)")
ax.set_title("CeO$_2$ RPA@PBE equation of state: PAW vs USPP")
ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8.5, loc="upper center")
fig.tight_layout()
fig.savefig("ceo2_eos.pdf", bbox_inches="tight"); fig.savefig("ceo2_eos.png", dpi=150, bbox_inches="tight")
print("wrote ceo2_eos.{pdf,png}")
