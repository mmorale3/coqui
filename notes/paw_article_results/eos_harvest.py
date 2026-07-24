#!/usr/bin/env python3
"""
Standalone RPA-EOS harvester (Harl-Kresse protocol).

Reads, per (volume, pseudopotential), the energy components needed to assemble
an RPA@PBE total energy WITHOUT touching CoQui's SCF or importing QE's screened
deeq:

  QE matched-kp3 scf decomposition (one-electron, hartree, xc, ewald, total,
      one-center paw)                        -> rpa_kp3_n250/scf/scf.out  [Ry]
  CoQui e_1e / e_hf / e_rpa                   -> rpa_kp3_n250/rpa_refix.out [Ha]
  CoQui full-deeq e_1e (cross-check)          -> rpa_kp3_n250/rpa.out       [Ha]
  QE single-shot EXX@PBE total (formula B)    -> exx_kp3/exx.out            [Ry]  (optional)
  QE self-consistent HF total (reference)     -> hf_kp8/hf.out              [Ry]  (optional)

Emits a CSV to stdout. Fit/plot happens locally with eos_fit.py.
Run on rusty:  python3 eos_harvest.py ~/ceph/CoQui/Si_eos_rpa_hf/runs > eos_components.csv
"""
import os, re, sys, glob

RY2HA = 0.5

def _grab(path, label, tail=True):
    """Return the float preceding 'Ry'/'a.u.' on the last (or first) line
    containing `label`, or None."""
    if not os.path.isfile(path):
        return None
    hits = []
    pat = re.compile(re.escape(label) + r"\s*=?\s*(-?\d+\.\d+)")
    with open(path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                hits.append(float(m.group(1)))
    if not hits:
        return None
    return hits[-1] if tail else hits[0]

def vol_from_name(name):
    # a10p26 -> celldm(1) = 10.26 Bohr ; fcc (ibrav=2) primitive cell V = a^3/4
    m = re.match(r"a(\d+)p(\d+)", name)
    if not m:
        return None, None
    a = float(f"{int(m.group(1))}.{m.group(2)}")
    return a, a**3 / 4.0

def harvest(root, pps=("oncv", "uspp", "paw"), rpa="rpa_kp3_n250"):
    rows = []
    vols = sorted(d for d in os.listdir(root) if re.match(r"a\d+p\d+$", d))
    for v in vols:
        a, vol = vol_from_name(v)
        for pp in pps:
            base = os.path.join(root, v, pp)
            scf = os.path.join(base, rpa, "scf", "scf.out")
            if not os.path.isfile(scf):
                continue
            r = dict(vol_name=v, a_bohr=a, vol_bohr3=vol, pp=pp)
            # --- QE matched-kp3 scf decomposition [Ry] ---
            r["qe_total_ry"]    = _grab(scf, "total energy")
            r["qe_1e_ry"]       = _grab(scf, "one-electron contribution")
            r["qe_hartree_ry"]  = _grab(scf, "hartree contribution")
            r["qe_xc_ry"]       = _grab(scf, "xc contribution")
            r["qe_ewald_ry"]    = _grab(scf, "ewald contribution")
            r["qe_onecpaw_ry"]  = _grab(scf, "one-center paw contrib.")
            # --- CoQui components [Ha]: prefer rpa_refix.out (bare-dvan binary),
            #     fall back to rpa.out (fresh bracket runs only have rpa.out) ---
            refix = os.path.join(base, rpa, "rpa_refix.out")
            old   = os.path.join(base, rpa, "rpa.out")
            def _cq(label):
                v = _grab(refix, label)
                return v if v is not None else _grab(old, label)
            r["cq_e1e_ha"]  = _cq("One-electron energy:")
            r["cq_ehf_ha"]  = _cq("Hartree-Fock energy:")
            r["cq_erpa_ha"] = _cq("RPA energy:")
            # --- CoQui full-deeq e_1e cross-check [Ha] (old binary) ---
            r["cq_e1e_fulldeeq_ha"] = _grab(old, "One-electron energy:")
            # --- QE single-shot EXX@PBE total [Ry] (formula B, optional) ---
            exx = os.path.join(base, "exx_kp3", "exx.out")
            r["qe_exx_total_ry"] = _grab(exx, "total energy")
            # --- QE self-consistent HF reference [Ry] (optional) ---
            hf = os.path.join(base, "hf_kp8", "hf.out")
            r["qe_scf_hf_total_ry"] = _grab(hf, "total energy")
            rows.append(r)
    return rows

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = harvest(root)
    if not rows:
        sys.exit(f"no data found under {root}")
    cols = ["vol_name","a_bohr","vol_bohr3","pp",
            "qe_total_ry","qe_1e_ry","qe_hartree_ry","qe_xc_ry","qe_ewald_ry","qe_onecpaw_ry",
            "cq_e1e_ha","cq_ehf_ha","cq_erpa_ha","cq_e1e_fulldeeq_ha",
            "qe_exx_total_ry","qe_scf_hf_total_ry"]
    print(",".join(cols))
    for r in rows:
        print(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))

if __name__ == "__main__":
    main()
