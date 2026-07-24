#!/usr/bin/env python3
"""
Standalone RPA-EOS fitter. Consumes eos_components.csv (from eos_harvest.py) and
assembles RPA@PBE total energies, then fits a 3rd-order Birch-Murnaghan EOS.

Two non-RPA assembly conventions (neither touches CoQui SCF / QE screened deeq):

  A  E = QE_one-electron/2  + e_hf_CoQui + e_rpa_CoQui + Ewald/2     [+ one-center paw/2 for PAW]
       -> replaces only CoQui's buggy e_1e with QE's correct one-body energy.
          Clean for USPP (no one-center); approximate one-center bookkeeping for PAW.

  B  E = QE_EXX@PBE_total/2 + e_rpa_CoQui                            (Harl-Kresse)
       -> QE supplies ALL non-correlation (1e + Hartree + exact-X + Ewald + PAW one-center)
          on the PBE orbitals; CoQui supplies ONLY the RPA correlation. Needs exx column.

  N  E = e_1e_CoQui + e_hf_CoQui + e_rpa_CoQui + Ewald/2  (CoQui-native, the broken one)

  H  E = E_HF_QE_selfconsistent/2 + e_rpa_CoQui  (RPA+HF, orbital-INCONSISTENT)
       -> QE self-consistent HF total (augmentation-exact, Ewald already included) +
          CoQui RPA correlation on PBE orbitals. Free (hf_kp8 exists). NOTE: hf_kp8 is
          ecutwfc=65 / 8x8x8 vs RPA ecutwfc=100 / 3x3x3 -- mismatch ~cancels in curvature.

Usage:  python3 eos_fit.py eos_components.csv [--plot out.pdf]
"""
import sys, csv, math
import numpy as np
from scipy.optimize import curve_fit

RY2HA = 0.5
HA_BOHR3_TO_GPA = 29421.02648

def bm3(V, E0, V0, B0, Bp):
    """3rd-order Birch-Murnaghan E(V). B0 in Ha/Bohr^3."""
    t = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * (
        (t - 1.0) ** 3 * Bp + (t - 1.0) ** 2 * (6.0 - 4.0 * t)
    )

def fit_bm(V, E):
    V = np.asarray(V, float); E = np.asarray(E, float)
    i = np.argmin(E)
    p0 = [E[i], V[i], 0.02, 4.0]   # B0~0.02 Ha/Bohr^3 ~ 590 GPa upper guess
    popt, _ = curve_fit(bm3, V, E, p0=p0, maxfev=200000)
    E0, V0, B0, Bp = popt
    a0 = (4.0 * V0) ** (1.0 / 3.0)         # fcc primitive: V = a^3/4
    return dict(E0=E0, V0=V0, a0=a0, B0_GPa=B0 * HA_BOHR3_TO_GPA, Bp=Bp,
                rms=float(np.sqrt(np.mean((bm3(V, *popt) - E) ** 2))))

def f(row, key):
    s = row.get(key, "")
    return float(s) if s not in ("", None) else None

def assemble(rows, pp):
    out = {"N": [], "A": [], "B": [], "H": [], "V": [], "a": []}
    for r in rows:
        if r["pp"] != pp:
            continue
        V = f(r, "vol_bohr3"); a = f(r, "a_bohr")
        e1e = f(r, "cq_e1e_ha"); ehf = f(r, "cq_ehf_ha"); erpa = f(r, "cq_erpa_ha")
        ew = f(r, "qe_ewald_ry"); qe1e = f(r, "qe_1e_ry"); ocp = f(r, "qe_onecpaw_ry")
        exx = f(r, "qe_exx_total_ry"); hf = f(r, "qe_scf_hf_total_ry")
        if None in (V, e1e, ehf, erpa, ew, qe1e):
            continue
        ewald_ha = ew * RY2HA
        out["V"].append(V); out["a"].append(a)
        out["N"].append(e1e + ehf + erpa + ewald_ha)
        one_center = (ocp * RY2HA) if ocp is not None else 0.0
        out["A"].append(qe1e * RY2HA + one_center + ehf + erpa + ewald_ha)
        out["B"].append((exx * RY2HA + erpa) if exx is not None else None)
        # RPA+HF: QE self-consistent HF total already includes Ewald
        out["H"].append((hf * RY2HA + erpa) if hf is not None else None)
        # diagnostics: bare QE references (no CoQui correlation)
        out.setdefault("HF0", []).append(hf * RY2HA if hf is not None else None)
        out.setdefault("PBE0", []).append(f(r, "qe_total_ry") * RY2HA)
    return out

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    csv_path = sys.argv[1]
    plot = sys.argv[3] if "--plot" in sys.argv else None
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    pps = sorted({r["pp"] for r in rows})
    series = {}
    for pp in pps:
        d = assemble(rows, pp)
        order = np.argsort(d["V"])
        V = np.array(d["V"])[order]; a = np.array(d["a"])[order]
        series[pp] = dict(V=V, a=a)
        print(f"\n===== {pp.upper()}  ({len(V)} volumes, a = {a.tolist()} Bohr) =====")
        for tag in ("PBE0", "HF0", "N", "A", "H", "B"):
            vals = [d[tag][i] for i in order]
            if any(x is None for x in vals):
                miss = {"B": "needs single-shot EXX@PBE", "H": "needs hf_kp8"}.get(tag, "")
                print(f"  [{tag}] (incomplete — {miss})")
                continue
            E = np.array(vals, float)
            series[pp][tag] = E
            try:
                r = fit_bm(V, E)
                label = {"PBE0": "QE PBE only (no corr)", "HF0": "QE HF only (no corr)",
                         "N": "CoQui-native (broken)", "A": "QE-1e substitution",
                         "H": "RPA+HF (orbital-inconsist)", "B": "Harl-Kresse EXX@PBE"}[tag]
                print(f"  [{tag}] {label:24s}  a0 = {r['a0']:.4f} Bohr   "
                      f"B0 = {r['B0_GPa']:7.1f} GPa   B' = {r['Bp']:5.2f}   rms = {r['rms']*1e3:.3f} mHa")
            except Exception as e:
                print(f"  [{tag}] fit failed: {e}")
    print("\n  reference: VASP-PAW RPA@PBE (Harl 2010) a0=10.244 Bohr B0=98 GPa; "
          "CoQui ONCV RPA@PBE a0=10.228 / 101 GPa")
    if plot:
        make_plot(series, plot)

def make_plot(series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    tags = [("N", "CoQui-native (broken)"), ("A", "QE 1e-substitution"),
            ("B", "Harl-Kresse EXX@PBE")]
    for ax, (tag, title) in zip(axes, tags):
        for pp, s in series.items():
            if tag not in s:
                continue
            E = s[tag]; a = s["a"]
            ax.plot(a, (E - E.min()) * 1e3, "o-", label=pp)
        ax.set_title(title); ax.set_xlabel("a (Bohr)")
        ax.set_ylabel("E - Emin (mHa)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"\n  wrote {path}")

if __name__ == "__main__":
    main()
