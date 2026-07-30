#!/usr/bin/env python3
"""Compare CoQui's one-center exact-exchange kernel `deltaC` against ABINIT's
`pawtab%eijkl` for the SAME PAW dataset.

Both are the same operator,

    K(I,J,K,L) = <phi_I phi_J|v|phi_K phi_L>^AE
               - <phit_I phit_J + Qhat_IJ|v|phit_K phit_L + Qhat_KL>,

CoQui's built by the QE-port in `paw_deltaC.py` (unpacked, (nh,nh,nh,nh)),
ABINIT's in `65_paw/m_paw_init.F90:586-655` (packed on klmn = jlmn*(jlmn-1)/2 +
ilmn with ilmn<=jlmn, and only the klmn<=klmn1 triangle filled).

The two codes do NOT order/sign the real spherical harmonics the same way, so a
direct element-by-element comparison would report a spurious mismatch.  Every
quantity compared here is instead invariant under an orthogonal transformation
acting within each (l, n) shell -- which is exactly what a differing real-Ylm
convention is:

  T_H  = sum_IJ  K(I,I,J,J)          Hartree-like contraction
  T_X  = sum_IJ  K(I,J,J,I)          exchange-like contraction  <-- the one that
                                     matters for E_x^{1c,vv}
  shell-resolved T_X over (shell(I), shell(J)) blocks
  spectrum of the pair matrix M[(IJ),(KL)] = K(I,J,K,L)

Usage:
    python3 cmp_onecenter_kernel.py PAWKERNEL.dat coqui_kernel_nt0.npz
"""
import sys

import numpy as np


def read_abinit_kernel(path):
    """Parse the ABI_DUMP_PAWKERNEL dump. Returns the first atom's block."""
    lmn_size = lmn2 = None
    indlmn, eijkl, excv, rhoij = {}, None, None, []
    seen_atom = False
    for line in open(path):
        f = line.split()
        if not f:
            continue
        tag = f[0]
        if tag == "ATOM":
            if seen_atom:
                break                      # only the first atom
            seen_atom = True
            lmn_size, lmn2 = int(f[3]), int(f[4])
            eijkl = np.zeros((lmn2, lmn2))
            excv = np.zeros(lmn2)
        elif tag == "INDLMN":
            indlmn[int(f[1])] = [int(x) for x in f[2:8]]
        elif tag == "EIJKL":
            eijkl[int(f[1]) - 1, int(f[2]) - 1] = float(f[3])
        elif tag == "EXCVIJ":
            excv[int(f[1]) - 1] = float(f[2])
        elif tag == "RHOIJ":
            rhoij.append((int(f[2]) - 1, float(f[3])))
    # pawinit fills only klmn <= klmn1; the tensor is symmetric on pair indices.
    eijkl = eijkl + eijkl.T - np.diag(np.diag(eijkl))
    ind = np.array([indlmn[i + 1] for i in range(lmn_size)])
    return dict(lmn_size=lmn_size, lmn2=lmn2, eijkl=eijkl, ex_cvij=excv,
                indlmn=ind, rhoij=rhoij)


def unpack(eijkl, nh):
    """Packed (klmn,klmn1) -> full (nh,nh,nh,nh); klmn = j*(j-1)/2 + i, i<=j."""
    pk = np.zeros((nh, nh), dtype=int)
    for j in range(nh):
        for i in range(j + 1):
            pk[i, j] = pk[j, i] = j * (j + 1) // 2 + i
    return eijkl[pk[:, :, None, None], pk[None, None, :, :]]


def invariants(K, shell):
    """Rotation-invariant contractions of a (nh,nh,nh,nh) one-center kernel."""
    nh = K.shape[0]
    T_H = np.einsum("iijj->", K)
    T_X = np.einsum("ijji->", K)
    ns = shell.max() + 1
    blk = np.zeros((ns, ns))
    for a in range(ns):
        ia = np.where(shell == a)[0]
        for b in range(ns):
            ib = np.where(shell == b)[0]
            blk[a, b] = np.einsum("ijji->", K[np.ix_(ia, ib, ib, ia)])
    M = K.reshape(nh * nh, nh * nh)
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    return T_H, T_X, blk, ev


def main():
    ab = read_abinit_kernel(sys.argv[1])
    cq = np.load(sys.argv[2])
    nh = int(cq["nh"])
    if ab["lmn_size"] != nh:
        sys.exit("channel count differs: ABINIT %d vs CoQui %d"
                 % (ab["lmn_size"], nh))

    K_ab = unpack(ab["eijkl"], nh)
    K_cq = cq["deltaC"]

    # shells: channels sharing (l, radial index). ABINIT indlmn = (l,m,n,lm,ln,s);
    # CoQui carries nhtol (l) and indv (beta channel).
    sh_ab = np.unique(ab["indlmn"][:, [0, 4]], axis=0, return_inverse=True)[1]
    sh_cq = np.unique(np.stack([cq["nhtol"], cq["indv"]], 1), axis=0,
                      return_inverse=True)[1]
    print("shells  ABINIT:", np.bincount(sh_ab), " CoQui:", np.bincount(sh_cq))

    # pair symmetry K(I,J,..) == K(J,I,..) is assumed by the unpacking above
    for tag, K in (("ABINIT", K_ab), ("CoQui", K_cq)):
        asym = np.abs(K - K.transpose(1, 0, 2, 3)).max() / np.abs(K).max()
        print("%-7s pair-symmetry |K-K^T|/max = %.2e" % (tag, asym))

    TH_a, TX_a, blk_a, ev_a = invariants(K_ab, sh_ab)
    TH_c, TX_c, blk_c, ev_c = invariants(K_cq, sh_cq)

    print("\n%-28s %14s %14s %10s" % ("invariant", "ABINIT", "CoQui", "CoQui/AB"))
    print("%-28s %14.8f %14.8f %10.5f" % ("T_H = sum K(iijj)", TH_a, TH_c,
                                          TH_c / TH_a if TH_a else np.nan))
    print("%-28s %14.8f %14.8f %10.5f" % ("T_X = sum K(ijji)", TX_a, TX_c,
                                          TX_c / TX_a if TX_a else np.nan))
    print("%-28s %14.8f %14.8f %10.5f" % ("max|K|", np.abs(K_ab).max(),
                                          np.abs(K_cq).max(),
                                          np.abs(K_cq).max() / np.abs(K_ab).max()))

    print("\nshell-resolved T_X (rows=shell(I), cols=shell(J)) -- ratio CoQui/AB")
    with np.errstate(divide="ignore", invalid="ignore"):
        print(np.array2string(blk_c / blk_a, precision=4, suppress_small=True))
    print("\nABINIT block:\n", np.array2string(blk_a, precision=6))
    print("CoQui  block:\n", np.array2string(blk_c, precision=6))

    print("\npair-matrix spectrum, 6 largest |eigenvalue|:")
    ia = np.argsort(-np.abs(ev_a))[:6]
    ic = np.argsort(-np.abs(ev_c))[:6]
    for k in range(6):
        print("   %14.8f   %14.8f   ratio %8.5f"
              % (ev_a[ia[k]], ev_c[ic[k]], ev_c[ic[k]] / ev_a[ia[k]]))


if __name__ == "__main__":
    main()
