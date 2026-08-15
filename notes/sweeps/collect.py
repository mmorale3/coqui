#!/usr/bin/env python3
"""Collect sweep results into results.csv + errors.csv.

Parses run.log of every run dir under ROOT for the energies printed by the
[rpa] driver and the ISDF selection summary. Cross-checks the RPA group of
sweep.mbpt.h5 when h5py is available. Errors are reported against, per fixture:
chol_ref if present, else thc_ref.

Usage: python3 collect.py --root RUNDIR [--csv results.csv]
"""
import argparse, csv, os, re, sys

PATTERNS = {
    "e_1e":      re.compile(r"One-electron energy:\s+([-\d.eE+]+)"),
    "e_hf":      re.compile(r"Hartree-Fock energy:\s+([-\d.eE+]+)"),
    "e_rpa":     re.compile(r"RPA energy:\s+([-\d.eE+]+)"),
    "e_x":       re.compile(r"Exchange energy:\s+([-\d.eE+]+)"),
    "e_hartree": re.compile(r"Hartree energy:\s+([-\d.eE+]+)"),
}
SEL = re.compile(r"ISDF point selection \((\w+)\): nchol = (\d+), final max \|D\| = ([-\d.eE+]+), c_mu = [^=]+= ([\d.eE+-]+)")
DONE = re.compile(r"RPA energy routines end")


def parse_run(d):
    log = os.path.join(d, "run.log")
    if not os.path.isfile(log):
        return None
    txt = open(log, errors="replace").read()
    if not DONE.search(txt):
        return {"status": "INCOMPLETE"}
    row = {"status": "ok"}
    for k, pat in PATTERNS.items():
        m = pat.findall(txt)
        row[k] = float(m[-1]) if m else float("nan")
    m = SEL.findall(txt)
    if m:
        row["sel_impl"], row["nchol"], row["resid"], row["c_mu"] = \
            m[-1][0], int(m[-1][1]), float(m[-1][2]), float(m[-1][3])
    # h5 cross-check (optional)
    h5f = os.path.join(d, "sweep.mbpt.h5")
    try:
        import h5py
        with h5py.File(h5f, "r") as f:
            for k, ds in (("e_1e", "1e_energy"), ("e_hf", "hf_energy"), ("e_rpa", "rpa_energy")):
                v = float(f["RPA"][ds][()])
                if abs(v - row.get(k, float("nan"))) > 1e-10:
                    row["status"] = "H5_MISMATCH"
    except Exception:
        pass
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    root = args.root

    rows = []
    for fixture in sorted(os.listdir(root)):
        fdir = os.path.join(root, fixture)
        if not os.path.isdir(fdir):
            continue
        for cfg in sorted(os.listdir(fdir)):
            cdir = os.path.join(fdir, cfg)
            if not os.path.isdir(cdir):
                continue
            subdirs = [s for s in sorted(os.listdir(cdir)) if s.startswith("c")] \
                if any(s.startswith("c") and os.path.isdir(os.path.join(cdir, s)) for s in os.listdir(cdir)) else [None]
            for sub in subdirs:
                d = cdir if sub is None else os.path.join(cdir, sub)
                r = parse_run(d)
                if r is None:
                    continue
                r.update({"fixture": fixture, "config": cfg,
                          "c": (int(sub[1:]) if sub else None)})
                rows.append(r)

    # errors vs reference
    for fixture in {r["fixture"] for r in rows}:
        frows = [r for r in rows if r["fixture"] == fixture and r["status"] == "ok"]
        ref = next((r for r in frows if r["config"] == "chol_ref"),
                   next((r for r in frows if r["config"] == "thc_ref"), None))
        for r in frows:
            if ref and r is not ref:
                for k in ("e_hf", "e_x", "e_hartree", "e_rpa"):
                    r["d" + k] = r.get(k, float("nan")) - ref.get(k, float("nan"))
                r["ref"] = ref["config"]

    out = args.csv or os.path.join(root, "results.csv")
    keys = ["fixture", "config", "c", "status", "nchol", "c_mu", "resid",
            "e_1e", "e_hf", "e_x", "e_hartree", "e_rpa",
            "de_hf", "de_x", "de_hartree", "de_rpa", "ref", "sel_impl"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["fixture"], r["config"], r["c"] or 0)):
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)")
    # compact table to stdout
    for r in sorted(rows, key=lambda r: (r["fixture"], r["config"], r["c"] or 0)):
        if r["status"] != "ok":
            print(f'{r["fixture"]:5s} {r["config"]:12s} c={r["c"]}: {r["status"]}')
        elif "de_rpa" in r:
            print(f'{r["fixture"]:5s} {r["config"]:12s} c={r["c"] or "-":>2} '
                  f'dEx={r["de_x"]: .3e} dEh={r["de_hartree"]: .3e} dErpa={r["de_rpa"]: .3e}')


if __name__ == "__main__":
    main()
