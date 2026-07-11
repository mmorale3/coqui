"""
ABINIT-driven Si EXX+RPA EOS harvester + fit.

For each lattice constant, assembles the RPA@PBE total energy avoiding CoQui's
pathological one-electron energy (stiff-projector deeq*becsum artifact):

  E_RPA@PBE = E_total(ABINIT PBE) - [xc_smooth + e_pawxc](ABINIT)
                                  + [e_hf - E_H](CoQui exact exchange)
                                  + e_rpa(CoQui RPA correlation)

Replaces the PBE XC (smooth from .abo + one-center e_pawxc from GSR) with exact
exchange and RPA correlation; the Hartree cancels (E_H(PBE) ~ E_H(CoQui)).

Fits E(a) (both PBE and RPA@PBE) to a quadratic near the minimum -> a0 (Bohr).
Usage:  python3 eos_rpa_fit.py <abinit_eos_dir> <coqui_rpa_dir>
"""
import sys, os, re, glob
import numpy as np

AS = ["10p00", "10p20", "10p40", "10p60", "10p80"]


def a_bohr(tag):
    return float(tag.replace("p", "."))


def grab(path, label):
    if not os.path.isfile(path):
        return None
    pat = re.compile(re.escape(label) + r"\s*:?\s*=?\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)")
    hits = [float(m.group(1)) for line in open(path, errors="ignore")
            for m in [pat.search(line)] if m]
    return hits[-1] if hits else None


def gsr_epawxc(gsr):
    try:
        from netCDF4 import Dataset
        ds = Dataset(gsr, "r")
        v = float(np.array(ds.variables["e_pawxc"][:]).ravel()[0]) if "e_pawxc" in ds.variables else 0.0
        ds.close()
        return v
    except Exception:
        return 0.0


def harvest(eos_dir, coqui_dir):
    rows = []
    for tag in AS:
        abo = os.path.join(eos_dir, "a" + tag, "si.abo")
        gsr = glob.glob(os.path.join(eos_dir, "a" + tag, "*GSR.nc"))
        rpa = os.path.join(coqui_dir, tag, "rpa.out")
        E_pbe = grab(abo, "etotal") or grab(abo, "total_energy")
        xc = grab(abo, "xc")                          # smooth XC (Ha)
        epawxc = gsr_epawxc(gsr[0]) if gsr else 0.0   # one-center XC (Ha)
        ehf = grab(rpa, "Hartree-Fock energy")
        erpa = grab(rpa, "RPA energy")
        eH = grab(rpa, "Hartree energy")
        if None in (E_pbe, xc, ehf, erpa, eH):
            print("  [skip] a%s missing: E_pbe=%s xc=%s ehf=%s erpa=%s eH=%s"
                  % (tag, E_pbe, xc, ehf, erpa, eH))
            continue
        Ex_exact = ehf - eH
        E_rpa = E_pbe - (xc + epawxc) + Ex_exact + erpa
        rows.append(dict(a=a_bohr(tag), E_pbe=E_pbe, xc=xc, epawxc=epawxc,
                         ehf=ehf, erpa=erpa, eH=eH, Ex_exact=Ex_exact, E_rpa=E_rpa))
    return rows


def fit_min(a, E):
    a = np.array(a); E = np.array(E)
    c = np.polyfit(a, E, 2)               # quadratic
    a0 = -c[1] / (2 * c[0])
    return a0, c


def main():
    eos_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    coqui_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    rows = harvest(eos_dir, coqui_dir)
    if len(rows) < 3:
        sys.exit("need >=3 points, got %d" % len(rows))
    print("\n a(Bohr)   E_pbe(Ha)    xc      e_pawxc   ehf       erpa     Ex_exact   E_rpa@pbe(Ha)")
    for r in rows:
        print(" %7.2f  %11.5f %8.4f %8.4f %8.4f %8.4f %9.4f  %12.5f"
              % (r["a"], r["E_pbe"], r["xc"], r["epawxc"], r["ehf"], r["erpa"],
                 r["Ex_exact"], r["E_rpa"]))
    a = [r["a"] for r in rows]
    a0_pbe, _ = fit_min(a, [r["E_pbe"] for r in rows])
    a0_rpa, _ = fit_min(a, [r["E_rpa"] for r in rows])
    print("\n  PBE      a0 = %.3f Bohr = %.4f Ang" % (a0_pbe, a0_pbe * 0.529177))
    print("  RPA@PBE  a0 = %.3f Bohr = %.4f Ang" % (a0_rpa, a0_rpa * 0.529177))
    print("\n  Reference: ONCV/VASP RPA@PBE ~10.23-10.26 Bohr; QE-PAW(broken EXX) ~10.62")


if __name__ == "__main__":
    main()
