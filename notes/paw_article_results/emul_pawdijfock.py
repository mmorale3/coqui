#!/usr/bin/env python3
"""Emulate ABINIT's `pawdijfock` exactly, and measure the `eijkl`-triangle loss.

Reads the ABI_DUMP_PAWKERNEL dump written by the instrumented ABINIT (see
abinit_ene_instr.py) and recomputes `efockdc` two ways:

  raw          `eijkl` exactly as `pawinit` leaves it. `m_paw_init.F90:610` runs
               `do klmn1 = k1min, lmn2_size` with `k1min = klmn`, so only the
               klmn <= klmn1 triangle is ever written. `pawdijfock`
               (`m_pawdij.F90:1223`) then reads `eijkl(pack(i,l), pack(k,j))`
               without ordering the two pair indices, so whatever lands in the
               unfilled half comes back as a structural zero.

  symmetrised  the same contraction on `eijkl + eijkl^T - diag`, i.e. what the
               tensor mathematically IS (it is symmetric under the pair swap).

`raw` reproduces ABINIT's printed `efockdc` to <=2e-15 on every run tried, which
is what makes the comparison meaningful; `symmetrised` is the physical value.
Their ratio is the fraction ABINIT drops. It is NOT a property of the dataset
alone -- it depends on which rho_ij are populated:

    jth_with_d (s,p,d) at Gamma      1.00000
    jth_with_d (s,p,d) 2x2x2         0.87934
    jth_with_d (s,p,d) 4x4x4         0.87601 / 0.88538 / 0.89943
                                       at a = 10.05 / 10.25 / 10.55
    Si_paw_pw_12el (s,p) 2x2x2       0.99419

The physical one-center valence-valence exchange is then
`½ x symmetrised` -- the extra ½ being the closed-shell same-spin factor that
`pawdijfock` also gets wrong at nsppol=1 (see eos_exchange_ledger.md §3g).

    python3 emul_pawdijfock.py PAWKERNEL.dat [printed_efockdc]
"""
import sys

import numpy as np


def read_atoms(path):
    """Every ATOM block of the dump: eijkl (packed, as written) and rhoij."""
    atoms, cur = [], None
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "ATOM":
            cur = dict(lmn=int(f[3]), lmn2=int(f[4]),
                       E=np.zeros((int(f[4]), int(f[4]))), rho={})
            atoms.append(cur)
        elif f[0] == "EIJKL":
            cur["E"][int(f[1]) - 1, int(f[2]) - 1] = float(f[3])
        elif f[0] == "RHOIJ":
            cur["rho"][int(f[2])] = float(f[3])
    return atoms


def pack(i, j):
    """1-based packed pair index, klmn = j*(j-1)/2 + i with i <= j."""
    i, j = min(i, j), max(i, j)
    return j * (j - 1) // 2 + i


def efockdc(atom, symmetrise):
    """pawdijfock + pawaccenergy for one atom, nsppol = 1 (nsploop = 1)."""
    nh, lmn2 = atom["lmn"], atom["lmn2"]
    E = atom["E"]
    E = E + E.T - np.diag(np.diag(E)) if symmetrise else np.triu(E)
    il, jl = {}, {}
    for j in range(1, nh + 1):
        for i in range(1, j + 1):
            il[pack(i, j)] = i
            jl[pack(i, j)] = j
    dltij = {k: (1.0 if il[k] == jl[k] else 2.0) for k in il}
    rho = {k: atom["rho"].get(k, 0.0) for k in il}

    dij = np.zeros(lmn2 + 1)
    for kl in range(1, lmn2 + 1):
        ro = rho[kl] * dltij[kl]
        if ro == 0.0:
            continue
        k_, l_ = il[kl], jl[kl]
        dij[kl] -= ro * E[kl - 1, kl - 1]
        for ij in range(1, kl):                       # (i,j) < (k,l)
            i_, j_ = il[ij], jl[ij]
            dij[ij] -= ro * E[pack(i_, l_) - 1, pack(k_, j_) - 1]
        for ij in range(kl + 1, lmn2 + 1):            # (i,j) > (k,l)
            i_, j_ = il[ij], jl[ij]
            dij[ij] -= ro * E[pack(k_, j_) - 1, pack(i_, l_) - 1]
    # pawaccenergy on half*dijfock_vv
    return 0.5 * sum(rho[k] * dltij[k] * dij[k] for k in range(1, lmn2 + 1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "PAWKERNEL.dat"
    printed = float(sys.argv[2]) if len(sys.argv) > 2 else None
    atoms = read_atoms(path)
    raw = sum(efockdc(a, False) for a in atoms)
    sym = sum(efockdc(a, True) for a in atoms)
    print("atoms in dump          : %d" % len(atoms))
    print("emulated efockdc (raw) : %+.9f" % raw)
    if printed is not None:
        print("ABINIT printed efockdc : %+.9f   (|diff| %.1e)"
              % (printed, abs(raw - printed)))
    print("emulated (symmetrised) : %+.9f" % sym)
    print("triangle ratio sym/raw : %.5f   -> ABINIT drops %.2f%%"
          % (sym / raw, 100 * (1 - sym / raw)))
    print("physical E_x^{1c,vv}   : %+.9f   (= sym/2, the same-spin factor)"
          % (sym / 2))


if __name__ == "__main__":
    main()
