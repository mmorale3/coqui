#!/usr/bin/env python3
"""Harvest the CoQui exchange decomposition (gen_exx_split.py) and put it next
to the instrumented-ABINIT reference, term by term.

CoQui rows (bare = the gygi divergence removed analytically):

    smooth      = E_x(aug_off)
    aug         = E_x(onsite_off) - E_x(aug_off)
    K_a         = E_x(base)       - E_x(onsite_off)
    shape       = E_x(shape)                       (independent route)

ABINIT rows (bare = its `fock_icutcoul 3` CRYSTAL divergence removed):

    fock0       smooth Fock INCLUDING nhat  (m_fock_getghc.F90:663 pawmknhat_psipsi)
    efockdc     = 1/2 sum_ij rho_ij dijfock_vv = the one-centre vv exchange
    efock-efockdc = the core-valence part (agrees with CoQui's Tr[g ex_cvij] to 6 uHa)

so the two comparable pairs are

    CoQui[smooth+aug]  <->  ABINIT[fock0]
    CoQui[K_a]         <->  ABINIT[efockdc] / 2      <-- NOTE THE FACTOR

**The /2 is not a fudge.** ABINIT's nsppol=1 PAW one-centre valence-valence Fock
term is double counted: `pawdijfock` (m_pawdij.F90:1223) sets
`nsp = pawrhoij%nsppol`, so at nsppol=1 it contracts the SPIN-SUMMED `rhoij`
twice. Running the same non-magnetic state at nsppol=2 (+spinmagntarget 0.0)
leaves e1t10 / eh2 / fock0 / core-valence identical to 1e-9 and halves `efockdc`
exactly (-13.394629907 -> -6.697314956). Only the nsppol=1 numbers need the /2;
every other row in this ledger is unaffected. See eos_exchange_ledger.md §3g.

Both divergence constants are exactly proportional to 1/a and were validated by
the NC control (eos_exchange_ledger.md §2b/§2c) -- E_div*a is bit-constant.
"""
import subprocess
import sys

# E_div * a, bit-constant across the series (eos_exchange_ledger.md §2b)
DIV_COQUI_A = -4.584862
DIV_ABINIT_A = -3.759324

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
TAGS = ["base", "onsite_off", "aug_off", "shape"]

# instrumented-ABINIT reference (eos_ledger_jthd), Ha
AB = {
    10.05: dict(fock0=-2.0685950, efock=-0.5457693, efockdc=-1.5631623419764228e-02),
    10.15: dict(fock0=-2.0540683, efock=-0.5394147, efockdc=-1.5253227740393935e-02),
    10.25: dict(fock0=-2.0397491, efock=-0.5334815, efockdc=-1.4903358254785753e-02),
    10.35: dict(fock0=-2.0256365, efock=-0.5279509, efockdc=-1.4580439348523808e-02),
    10.45: dict(fock0=-2.0117304, efock=-0.5228049, efockdc=-1.4282877307715994e-02),
    10.55: dict(fock0=-1.9980314, efock=-0.5180260, efockdc=-1.4009296028586289e-02),
}


def fetch(avals):
    body = "\n".join(
        ['for a in %s; do' % " ".join(avals),
         '  for t in %s; do' % " ".join(TAGS),
         '    f=%s/exx_split/a${a}_${t}/rpa.out' % ROOT,
         '    [ -f $f ] || { echo "$a $t MISSING"; continue; }',
         '    ex=$(grep -m1 "^Exchange energy:" $f | awk "{print \\$3}")',
         '    cv=$(grep -m1 "ex_cvij" $f | sed "s/.*=//" | awk "{print \\$1}")',
         '    echo "$a $t ${ex:-NONE} ${cv:-NONE}"',
         '  done',
         'done'])
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=40", "rusty", "bash -s"],
                       input=body, text=True, capture_output=True)
    if p.returncode:
        sys.exit("ssh failed:\n" + p.stderr[-1500:])
    out = {}
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        a, tag = float(f[0]), f[1]
        if f[2] in ("MISSING", "NONE"):
            print("  ** a=%s %s: %s" % (f[0], tag, f[2]))
            continue
        out.setdefault(a, {})[tag] = float(f[2])
        if len(f) > 3 and f[3] not in ("NONE",):
            try:
                out[a]["cv"] = float(f[3])
            except ValueError:
                pass
    return out


def main():
    avals = sys.argv[1:] or ["10.25", "10.55"]
    cq = fetch(avals)
    if not cq:
        sys.exit("no CoQui runs harvested yet")

    print("\n%6s %12s %12s %12s %12s" % ("a", "smooth", "aug", "K_a", "shape-tot"))
    rows = {}
    for a in sorted(cq):
        d = cq[a]
        if not {"base", "onsite_off", "aug_off"} <= set(d):
            print("%6.2f  incomplete: have %s" % (a, sorted(d)))
            continue
        div = DIV_COQUI_A / a
        smooth = d["aug_off"] - div
        aug = d["onsite_off"] - d["aug_off"]
        ka = d["base"] - d["onsite_off"]
        shp = (d["shape"] - div) if "shape" in d else float("nan")
        rows[a] = dict(smooth=smooth, aug=aug, ka=ka, shape=shp,
                       total=d["base"] - div)
        print("%6.2f %12.7f %12.7f %12.7f %12.7f" % (a, smooth, aug, ka, shp))

    print("\n--- against instrumented ABINIT (Ha; efockdc HALVED, see header) ---")
    print("%6s %14s %14s %10s   %14s %14s %10s"
          % ("a", "CQ smooth+aug", "AB fock0", "diff mHa", "CQ K_a",
             "AB 1c-vv", "ratio"))
    for a in sorted(rows):
        if a not in AB:
            continue
        r, ab = rows[a], AB[a]
        sm = r["smooth"] + r["aug"]
        f0 = ab["fock0"] - DIV_ABINIT_A / a
        ref = ab["efockdc"] / 2.0
        print("%6.2f %14.7f %14.7f %10.4f   %14.7f %14.7f %10.5f"
              % (a, sm, f0, (sm - f0) * 1000, r["ka"], ref, r["ka"] / ref))

    print("\nExpect: diff ~ 0 (the smooth+compensation halves agree), and a K_a")
    print("ratio of 1 on the DIRECT route. The THC route currently reads ~0.89 —")
    print("the open sub-mHa item (add_K_a_to_LL / local-ISDF compression).")
    print("A `base` variant that does not reproduce the EOS series' E_x means the")
    print("density matrix moved (beta / iaft_prec); the on/off DIFFERENCES stay")
    print("valid, the absolute smooth+aug comparison does not.")


if __name__ == "__main__":
    main()
