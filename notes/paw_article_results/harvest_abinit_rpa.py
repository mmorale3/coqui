#!/usr/bin/env python3
"""Run ON rusty. Harvest ABINIT ACFD-RPA correlation energies (and the CoQui THC-RPA
reference at nband=500) into a CSV for the cross-code instability comparison.

ABINIT prints 'RPA energy [Ha] :' once per q-point as a RUNNING TOTAL over the q-mesh
(m_screening_driver.F90: ec_rpa is allocated at the first q, accumulated with Qmesh%wt,
freed at the last).  The value for a dataset is therefore the LAST occurrence within
that dataset's section of the .abo -- equivalently the single value in <prefix>_DS<n>_RPA.
We read the _RPA files, which are unambiguous.
"""
import os, re, glob, csv

ROOT   = "/mnt/home/mmorales/ceph/CoQui/abinit"
WORK   = f"{ROOT}/rpa_eos_jthd"
COQUI  = {"jth_with_d": f"{ROOT}/eos_jthd_coqui", "nc": f"{ROOT}/eos_nc500_coqui"}
COQUI_NB = f"{ROOT}/rpa_eos_jthd_coqui_nb"      # CoQui nbnd=100/250 companion series
ALAT   = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]
NBANDS = [100, 250, 500]


def read_rpa_file(path):
    """<prefix>_DS<n>_RPA holds '#RPA   <Ec in Ha>' (final, q-summed)."""
    try:
        for line in open(path):
            if line.startswith("#RPA"):
                return float(line.split()[1])
    except OSError:
        pass
    return None


def abinit_rows(d, nbands):
    """Return {nband: Ec}. Dataset i (1-based) corresponds to nbands[i-1]."""
    out = {}
    for i, nb in enumerate(nbands, start=1):
        v = read_rpa_file(os.path.join(d, f"rpao_DS{i}_RPA"))
        if v is None:                       # fall back to the .abo running total
            v = last_abo_value(os.path.join(d, "rpa.abo"), i)
        out[nb] = v
    return out


def last_abo_value(abo, ids):
    """Last 'RPA energy [Ha] :' inside the == DATASET ids == section of the .abo."""
    if not os.path.exists(abo):
        return None
    cur, val = None, None
    for line in open(abo):
        m = re.search(r"==\s*DATASET\s+(\d+)\s*=", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"RPA energy \[Ha\]\s*:\s*([-\d.EeDd+]+)", line)
        if m and cur == ids:
            val = float(m.group(1).replace("D", "E"))
    return val


def coqui_value(d):
    """CoQui rpa.out: 'RPA energy:   <Ec> a.u.' (nband = 500, all WFK bands)."""
    p = os.path.join(d, "rpa.out")
    if not os.path.exists(p):
        return None
    val = None
    for line in open(p):
        m = re.match(r"\s*RPA energy:\s*([-\d.Ee+]+)", line)
        if m:
            val = float(m.group(1))
    return val


rows = []
for a in ALAT:
    ab = abinit_rows(f"{WORK}/a{a}", NBANDS)
    for nb in NBANDS:
        rows.append(dict(pseudo="jth_with_d", code="abinit", alat=a, nband=nb,
                         variant="base", ec=ab[nb]))
    for pp, base in COQUI.items():
        v = coqui_value(f"{base}/a{a}")
        if v is not None:
            rows.append(dict(pseudo=pp, code="coqui", alat=a, nband=500,
                             variant="base", ec=v))
    for nb in (100, 250):                       # CoQui companion band series
        v = coqui_value(f"{COQUI_NB}/a{a}_n{nb}")
        if v is not None:
            rows.append(dict(pseudo="jth_with_d", code="coqui", alat=a, nband=nb,
                             variant="base", ec=v))

# probes
for d in sorted(glob.glob(f"{WORK}/probe_*")):
    tag = os.path.basename(d)
    nbs = [500] if "ecuteps" in tag else NBANDS
    for nb, v in abinit_rows(d, nbs).items():
        m = re.search(r"a(10\.\d+)$", tag)
        rows.append(dict(pseudo="jth_with_d", code="abinit",
                         alat=m.group(1) if m else "?", nband=nb,
                         variant=tag.replace(f"_a{m.group(1)}" if m else "", ""), ec=v))

out = f"{WORK}/rpa_instability.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pseudo", "code", "alat", "nband", "variant", "ec"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

fmt = lambda v: f"{v:10.5f}" if v is not None else f"{'--':>10}"


def series(a, code, pseudo="jth_with_d"):
    return {r["nband"]: r["ec"] for r in rows if r["alat"] == a and r["code"] == code
            and r["pseudo"] == pseudo and r["variant"] == "base"}


print(f"# wrote {out}\n")
print("# RPA correlation energy [Ha], PAW jth_with_d, 4x4x4 k, same 500-band WFK")
print(f"{'':>8} {'---------- ABINIT ----------':^32} {'---------- CoQui ----------':^32} {'NC':>10}")
print(f"{'a[Bohr]':>8} {'n=100':>10} {'n=250':>10} {'n=500':>10} "
      f"{'n=100':>10} {'n=250':>10} {'n=500':>10} {'n=500':>10}")
for a in ALAT:
    ab, cq = series(a, "abinit"), series(a, "coqui")
    nc = next((r["ec"] for r in rows if r["alat"] == a and r["code"] == "coqui"
               and r["pseudo"] == "nc"), None)
    print(f"{a:>8} {fmt(ab.get(100))} {fmt(ab.get(250))} {fmt(ab.get(500))} "
          f"{fmt(cq.get(100))} {fmt(cq.get(250))} {fmt(cq.get(500))} {fmt(nc)}")

# Instability metrics: spread across the volume range, and growth with nband.
print("\n# spread over a=10.05..10.55 [mHa]  (NC reference ~4 mHa)")
for code in ("abinit", "coqui"):
    for nb in NBANDS:
        vals = [series(a, code).get(nb) for a in ALAT]
        vals = [v for v in vals if v is not None]
        if len(vals) == len(ALAT):
            print(f"  {code:>6}  n={nb:<4}  {1000*(max(vals)-min(vals)):8.2f}")

print("\n# probes")
for r in rows:
    if r["variant"] != "base" and r["ec"] is not None:
        print(f"  {r['variant']:>24}  a={r['alat']}  n={r['nband']:>3}  {r['ec']:.6f}")
