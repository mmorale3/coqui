#!/usr/bin/env python3
"""
Si EXX+RPA equation of state from the post-V_LL-fix CoQui runs.

Total energy convention (RPA@PBE, Harl-Kresse: RPA evaluated on PBE orbitals):

    E_total = <CoQui "Total energy">  +  E_Ewald

CoQui's "Total energy" line is E_1e + E_HF(via ERI on PBE orbitals) + E_RPA and
is MISSING Ewald entirely.

On the factor of 1/2. The campaign memory records the recipe as
"CoQui_Total + 0.5 * E_Ewald", which is correct ONLY because that campaign read
Ewald out of a QE SCF, where it is printed in RYDBERG -- the 0.5 is a Ry->Ha
unit conversion, not a physical halving of the Madelung sum. Here the Ewald
comes from ABINIT, which prints HARTREE, so the factor is 1. Getting this wrong
shifts every point by ~4 Ha; it does not change a0 much (it is nearly constant
across the volume range) but it makes the absolute totals meaningless.

Sanity check on the values below: E_Ewald must scale as 1/a. Indeed
-8.575997 * (10.05/10.55) = -8.1695 = the a=10.55 entry.
"""
import sys, json
import numpy as np

# Ewald from ABINIT .abo, Hartree, ~/ceph/CoQui/abinit/eos_jthd/a{a}/*.abo
EWALD_HA = {
    10.05: -8.57599688619593,
    10.15: -8.49150430603639,
    10.25: -8.40866036158725,
    10.35: -8.32741726630622,
    10.45: -8.24772906280087,
    10.55: -8.16955153613926,
}


def birch_murnaghan(V, E0, V0, B0, Bp):
    """3rd-order Birch-Murnaghan. B0 in Hartree/Bohr^3."""
    t = (V0 / V) ** (2.0 / 3.0) - 1.0
    return E0 + (9.0 * V0 * B0 / 16.0) * (t**3 * Bp + t**2 * (6.0 - 4.0 * (V0 / V) ** (2.0 / 3.0)))


def fit(a_bohr, E_ha):
    from scipy.optimize import curve_fit
    a = np.asarray(a_bohr, float)
    E = np.asarray(E_ha, float)
    V = a**3 / 4.0                      # fcc primitive cell volume
    # Parabola in V for the initial guess.
    c = np.polyfit(V, E, 2)
    V0g = -c[1] / (2 * c[0])
    p, _ = curve_fit(birch_murnaghan, V, E,
                     p0=[E.min(), V0g, 2.0 * c[0] * V0g, 4.0])
    E0, V0, B0, Bp = p
    a0 = (4.0 * V0) ** (1.0 / 3.0)
    B0_GPa = B0 * 29421.02648438959      # Ha/Bohr^3 -> GPa
    resid = E - birch_murnaghan(V, *p)
    return dict(a0_bohr=a0, B0_GPa=B0_GPa, Bp=Bp, E0_Ha=E0,
                max_resid_Ha=float(np.abs(resid).max()))


def main(path):
    """`path` is a json {"10.05": {"total": <Ha>, "ec": <Ha>}, ...}"""
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for k, v in sorted(raw.items(), key=lambda kv: float(kv[0])):
        a = float(k)
        if a not in EWALD_HA:
            sys.exit(f"no Ewald recorded for a={a}")
        rows.append((a, v["total"] + EWALD_HA[a], v.get("ec")))

    print(f"{'a (Bohr)':>10} {'CoQui Total':>16} {'E_Ewald':>14} "
          f"{'E_total':>16} {'E_c(RPA)':>14}")
    for a, tot, ec in rows:
        print(f"{a:10.2f} {tot - EWALD_HA[a]:16.8f} {EWALD_HA[a]:14.8f} "
              f"{tot:16.8f} {'' if ec is None else f'{ec:14.8f}'}")

    if len(rows) < 4:
        sys.exit(f"\nonly {len(rows)} points -- need >=4 for a 3rd-order BM fit")

    r = fit([x[0] for x in rows], [x[1] for x in rows])
    print(f"\nBirch-Murnaghan (3rd order), {len(rows)} points:")
    print(f"  a0 = {r['a0_bohr']:.4f} Bohr   ({r['a0_bohr']*0.529177210903:.4f} Ang)")
    print(f"  B0 = {r['B0_GPa']:.1f} GPa      B' = {r['Bp']:.2f}")
    print(f"  max fit residual = {r['max_resid_Ha']*1e3:.3f} mHa")
    print("\nReference: VASP/PAW RPA@PBE (Harl 2010) a0 = 10.244 Bohr, B0 = 98 GPa;")
    print("           CoQui ONCV RPA@PBE kp4/n500      a0 = 10.228 Bohr, B0 = 101 GPa.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eos_exxrpa.json")
