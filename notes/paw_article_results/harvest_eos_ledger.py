#!/usr/bin/env python3
"""Harvest the instrumented-ABINIT + CoQui term-by-term energy ledger and print
the volume series with per-term slopes.

Protocol and the ABINIT<->CoQui term mapping: eos_exchange_ledger.md.

    python3 harvest_eos_ledger.py                 # both datasets, ABINIT only
    python3 harvest_eos_ledger.py --coqui DIR     # add a CoQui series

ABINIT values are read from the env-gated dumps, NOT from the printed totals:

  DS1 (converged PBE)            -> kinetic, hartree, localpsp, corepsp, ewald
  DS1 pawdenpot (last call)      -> e1t10, eh2
  DS2 (one-shot Fock, nstep 1)   -> fock0                (smooth Fock on PBE orbitals)
  DS2 pawdenpot (its only call)  -> efock                (one-centre Fock, vv+cv)

Reading E_x from DS2's total instead is wrong: that goes through e_eigenvalues
from one partially converged diagonalization and is noisy at the ~100 mHa level
across a volume series.
"""
import argparse
import json
import re
import subprocess
import sys

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
AVALS = ["10.05", "10.15", "10.25", "10.35", "10.45", "10.55"]

# Remote extraction. Kept as one ssh call per series to keep latency sane.
REMOTE = r"""
for a in %(avals)s; do
  d=%(root)s/%(sub)s/a$a
  abo=$d/si.abo ; log=$d/si.log
  [ -f $abo ] || { echo "$a MISSING"; continue; }
  grep -q "Calculation completed" $log 2>/dev/null || { echo "$a INCOMPLETE"; continue; }
  # --- energies container: first block = DS1, last block = DS2
  ds1=$(grep "COQUI_ENE" $abo | head -21)
  ds2=$(grep "COQUI_ENE" $abo | tail -21)
  g1() { echo "$ds1" | awk -v k="$1" '$2==k {print $4}'; }
  g2() { echo "$ds2" | awk -v k="$1" '$2==k {print $4}'; }
  # --- pawdenpot: DS1 uses the LAST call before the DS2 marker, DS2 its own.
  #     Split on the dataset banner so a change in SCF step count cannot
  #     silently make us read the wrong block.
  nds2=$(grep -n "== DATASET  2" $log | head -1 | cut -d: -f1)
  if [ -n "$nds2" ]; then
    p1=$(awk -v n=$nds2 'NR<n' $log | grep "COQUI_PAWENE" | tail -10)
    p2=$(awk -v n=$nds2 'NR>n' $log | grep "COQUI_PAWENE" | tail -10)
  else
    p1=$(grep "COQUI_PAWENE" $log | tail -10) ; p2=""
  fi
  q1() { echo "$p1" | awk -v k="$1" '$2==k {print $4}'; }
  q2() { echo "$p2" | awk -v k="$1" '$2==k {print $4}'; }
  echo "$a kinetic=$(g1 kinetic) hartree=$(g1 hartree) localpsp=$(g1 localpsp)" \
       "corepsp=$(g1 corepsp) ewald=$(g1 ewald) etotal1=$(g1 etotal)" \
       "e1t10=$(q1 e1t10) eh2=$(q1 eh2)" \
       "fock0=$(g2 fock0) fock=$(g2 fock) efock=$(q2 efock)" \
       "efockdc=$(q2 efockdc) e1t10_2=$(q2 e1t10)"
done
"""


def fetch(sub):
    body = REMOTE % dict(avals=" ".join(AVALS), root=ROOT, sub=sub)
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input=body, text=True, capture_output=True)
    if p.returncode:
        sys.exit("ssh failed for %s:\n%s" % (sub, p.stderr[-2000:]))
    out = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split()
        a = f[0]
        if len(f) < 2 or "=" not in line:
            print("  ** %s: %s" % (a, " ".join(f[1:]) or "no data"))
            continue
        d = {}
        for kv in f[1:]:
            k, _, v = kv.partition("=")
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = None
        out[float(a)] = d
    return out


def fetch_coqui(sub):
    """Read CoQui's printed decomposition from <sub>/a*/hf.out (or rpa.out)."""
    body = "\n".join([
        'for a in %s; do' % " ".join(AVALS),
        '  d=%s/%s/a$a' % (ROOT, sub),
        '  f=$d/hf.out ; [ -f $f ] || f=$d/rpa.out',
        '  [ -f $f ] || { echo "$a MISSING"; continue; }',
        '  echo "BEGIN $a"',
        '  grep -E "^(One-electron|Hartree-Fock|Exchange|Hartree) energy:" $f',
        '  grep -E "Tr\\[Dm|    sum |    e_1e " $f',
        '  echo "END"',
        'done'])
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input=body, text=True, capture_output=True)
    if p.returncode:
        sys.exit("ssh failed for %s:\n%s" % (sub, p.stderr[-2000:]))
    out, cur, a = {}, None, None
    for line in p.stdout.splitlines():
        if line.startswith("BEGIN"):
            a = float(line.split()[1]); cur = {}
            continue
        if line.startswith("END"):
            if a is not None and cur:
                out[a] = cur
            a, cur = None, None
            continue
        if cur is None:
            continue
        m = re.search(r"^(One-electron|Hartree-Fock|Exchange|Hartree) energy: *"
                      r"(-?[\d.]+(?:[eE][-+]?\d+)?)", line)
        if m:
            cur[{"One-electron": "e_1e", "Hartree-Fock": "e_hf",
                 "Exchange": "e_x", "Hartree": "e_h"}[m.group(1)]] = float(m.group(2))
            continue
        m = re.search(r"(kinetic|local|dion|ex_cvij|int_VQ)\s+Tr\[Dm[^=]*=\s*"
                      r"(-?[\d.]+(?:[eE][-+]?\d+)?)", line)
        if m:
            cur["cq_" + m.group(1)] = float(m.group(2))
    return out


def slope(av, get):
    """Central difference at the two interior-ish points, Ha/Bohr."""
    ks = sorted(av)
    out = {}
    for i in range(1, len(ks) - 1):
        y0, y1 = get(av[ks[i-1]]), get(av[ks[i+1]])
        if y0 is None or y1 is None:
            continue
        out[ks[i]] = (y1 - y0) / (ks[i+1] - ks[i-1])
    return out


def report(tag, ab, cq=None):
    print("\n" + "=" * 78)
    print("ABINIT ledger — %s" % tag)
    print("=" * 78)
    ks = sorted(ab)
    if not ks:
        print("  no complete runs yet")
        return

    def A(d, k):     # ABINIT rows
        return {
            "kinetic":  d.get("kinetic"),
            "loc+aZ":   None if None in (d.get("localpsp"), d.get("corepsp"))
                        else d["localpsp"] + d["corepsp"],
            "e1t10":    d.get("e1t10"),
            "E_H":      None if None in (d.get("hartree"), d.get("eh2"))
                        else d["hartree"] + 0.5 * d["eh2"],
            # fock0 = smooth Fock INCLUDING nhat; efock = one-centre (vv + cv).
            # HALF of efockdc is removed: ABINIT's nsppol=1 PAW one-centre
            # valence-valence Fock term is double counted -- pawdijfock
            # (m_pawdij.F90:1223) sets nsp = pawrhoij%nsppol, so it contracts
            # the SPIN-SUMMED rhoij twice. Verified by running the same
            # non-magnetic state at nsppol=2: e1t10 / eh2 / fock0 / core-valence
            # identical to 1e-9, efockdc exactly halved. See
            # eos_exchange_ledger.md §3g. Without this the ABINIT exchange row
            # is ~7.5 mHa too negative with a spurious +1.6 mHa/Bohr slope.
            "E_x":      None if None in (d.get("fock0"), d.get("efock"),
                                         d.get("efockdc"))
                        else d["fock0"] + d["efock"] - 0.5 * d["efockdc"],
        }[k]

    rows = ["kinetic", "loc+aZ", "e1t10", "E_H", "E_x"]
    print("%6s" % "a" + "".join("%14s" % r for r in rows))
    for a in ks:
        vals = [A(ab[a], r) for r in rows]
        print("%6.2f" % a + "".join("%14s" % ("---" if v is None else "%14.7f" % v)
                                    for v in vals))
    print("\nslopes dE/da (Ha/Bohr):")
    for r in rows:
        s = slope(ab, lambda d, r=r: A(d, r))
        print("  %-9s " % r + "  ".join("a=%.2f %+9.5f" % (k, v) for k, v in s.items()))

    # DS1-vs-DS2 sanity: the one-shot Fock must not have moved the occupancies.
    bad = [a for a in ks if ab[a].get("e1t10") and ab[a].get("e1t10_2")
           and abs(ab[a]["e1t10"] - ab[a]["e1t10_2"]) > 1e-4]
    if bad:
        print("\n  ** WARNING: DS2 e1t10 differs from DS1 by >0.1 mHa at a =",
              bad, "-- DS2's pawrhoij is NOT the PBE one, so efock is "
              "contaminated. Reduce the DS2 orbital update.")

    if cq:
        print("\n" + "-" * 78)
        print("CoQui vs ABINIT, matched rows (CoQui - ABINIT, Ha)")
        print("  NOTE ex_cvij is moved from CoQui's e_1e into exchange.")
        print("-" * 78)
        hdr = ["kinetic", "loc+aZ", "dion", "E_H", "E_x"]
        print("%6s" % "a" + "".join("%14s" % h for h in hdr))
        diffs = {}
        for a in ks:
            if a not in cq:
                continue
            c = cq[a]
            row = {
                "kinetic": _sub(c.get("cq_kinetic"), A(ab[a], "kinetic")),
                "loc+aZ":  _sub(_add(c.get("cq_local"), c.get("cq_int_VQ")),
                                A(ab[a], "loc+aZ")),
                "dion":    _sub(c.get("cq_dion"), A(ab[a], "e1t10")),
                "E_H":     _sub(c.get("e_h"), A(ab[a], "E_H")),
                "E_x":     _sub(_add(c.get("e_x"), c.get("cq_ex_cvij")),
                                A(ab[a], "E_x")),
            }
            diffs[a] = row
            print("%6.2f" % a + "".join(
                "%14s" % ("---" if row[h] is None else "%14.7f" % row[h])
                for h in hdr))
        if len(diffs) >= 3:
            print("\ndrift of each difference across the sampled range (Ha):")
            kk = sorted(diffs)
            for h in hdr:
                v0, v1 = diffs[kk[0]][h], diffs[kk[-1]][h]
                if v0 is None or v1 is None:
                    continue
                print("  %-9s %+10.6f  (slope %+9.5f Ha/Bohr)"
                      % (h, v1 - v0, (v1 - v0) / (kk[-1] - kk[0])))
            print("\nThe row whose difference DRIFTS is the defect; a flat "
                  "nonzero offset is only a convention split.")


def _sub(x, y):
    return None if x is None or y is None else x - y


def _add(x, y):
    if x is None:
        return None
    return x + (y or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coqui-jthd", default=None,
                    help="subdir under ceph/CoQui/abinit with CoQui hf.out per volume")
    ap.add_argument("--json", default=None, help="also dump raw values here")
    args = ap.parse_args()

    raw = {}
    for tag, sub in [("jth_with_d (PAW)", "eos_ledger_jthd"),
                     ("Si_GGA_noNLCC (NC control)", "eos_ledger_nc")]:
        ab = fetch(sub)
        raw[sub] = ab
        cq = fetch_coqui(args.coqui_jthd) if (args.coqui_jthd and "jthd" in sub) else None
        if cq:
            raw[args.coqui_jthd] = cq
        report(tag, ab, cq)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: {str(a): v for a, v in d.items()} for k, d in raw.items()},
                      f, indent=1)
        print("\nraw -> %s" % args.json)


if __name__ == "__main__":
    main()
