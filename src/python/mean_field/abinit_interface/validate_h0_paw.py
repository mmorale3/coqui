"""
Standalone PAW mean-field correctness test (per user suggestion): rebuild the KS
Hamiltonian and PAW overlap from the converter's h5 + ABINIT's total Dij, solve the
GENERALIZED eigenproblem  H c = eps S c,  and compare to ABINIT's eigenvalues.

  H(G,G') = T delta + V_KS(G-G') + sum_a beta_a Dij_a beta_a^dagger
  S(G,G') = delta            + sum_a beta_a qq_a  beta_a^dagger

A match on the OCCUPIED bands validates projectors beta(k+G), the overlap qq_nt, the
grids, and the units together -- with NO CoQui and NO GW.  Isolates whether a GW failure
is mean-field (converter) or self-energy (CoQui) side.

ABINIT's total converged Dij (Ha, atom 1) is passed in `DIJ_HA` below (from `si.abo`,
pawprtvol output).  The stiff unbound-s self element (b2-b2) overflows ABINIT's text
format; set via --d11 to test its (ir)relevance to the occupied bands.

Usage: python validate_h0_paw.py abinit_paw.h5 [ik] [--d11 VALUE]
"""
import sys
import numpy as np
import h5py

# ABINIT total converged Dij (Ha), 8x8, atom 1, from si.abo. b2-b2 (index [1,1]) overflows.
def dij_ha(d11):
    return np.array([
        [0.43373, 39.73534, 0, 0, 0, 0, 0, 0],
        [39.73534, d11, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.12584, 0, 0, -1.08557, 0, 0],
        [0, 0, 0, 0.12584, 0, 0, -1.08557, 0],
        [0, 0, 0, 0, 0.12584, 0, 0, -1.08557],
        [0, 0, -1.08557, 0, 0, 8.08814, 0, 0],
        [0, 0, 0, -1.08557, 0, 0, 8.08814, 0],
        [0, 0, 0, 0, -1.08557, 0, 0, 8.08814]], dtype=float)


def rd_cplx(g, name):
    a = g[name][:]
    return a[..., 0] + 1j * a[..., 1]


def main():
    fn = sys.argv[1]
    ik = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 0
    d11 = 0.0
    if "--d11" in sys.argv:
        d11 = float(sys.argv[sys.argv.index("--d11") + 1])

    f = h5py.File(fn, "r")
    recv = f["System/reciprocal_vectors"][:]
    kcry = f["System/BZ/kpoints_crystal"][:][ik]
    eig_ref = f["Orbitals/eigval"][:][0, ik, :]
    occ = f["Orbitals/occ"][:][0, ik, :]
    g = f["Hamiltonian/paw"]
    miller_g = g["miller_g"][:]
    vks = 0.5 * rd_cplx(g, "scf_local_potential")[0, 0, :]      # Ry -> Ha
    nh_atom = g["proj_per_atom"][:]; off = g["projector_offset"][:]; atid = g["atomic_id"][:]
    qq_nt = g["qq_nt"][:]                                        # (nsp,nhm,nhm) real
    mill_k = g["miller_k" + str(ik)][:]
    beta = rd_cplx(g, "projector_k" + str(ik)).T                # (npw, nkb)
    f.close()

    n = mill_k.shape[0]
    kcart = kcry @ recv
    kpg = kcart[None, :] + mill_k @ recv
    T = 0.5 * np.einsum("ij,ij->i", kpg, kpg)

    ngf = [int(miller_g[:, d].max() - miller_g[:, d].min() + 1) for d in range(3)]
    Vbox = np.zeros(ngf, dtype=complex)
    Vbox[miller_g[:, 0] % ngf[0], miller_g[:, 1] % ngf[1], miller_g[:, 2] % ngf[2]] = vks
    d1 = (mill_k[:, None, 0] - mill_k[None, :, 0]) % ngf[0]
    d2 = (mill_k[:, None, 1] - mill_k[None, :, 1]) % ngf[1]
    d3 = (mill_k[:, None, 2] - mill_k[None, :, 2]) % ngf[2]
    H = Vbox[d1, d2, d3].copy()
    H[np.diag_indices(n)] += T

    # block-diagonal Dij (Ha) and qq over atoms
    nkb = beta.shape[1]
    Dbig = np.zeros((nkb, nkb)); Qbig = np.zeros((nkb, nkb))
    Dij = dij_ha(d11)
    for a in range(len(nh_atom)):
        nh = int(nh_atom[a]); o = int(off[a]); isp = int(atid[a])
        Dbig[o:o + nh, o:o + nh] = Dij[:nh, :nh]
        Qbig[o:o + nh, o:o + nh] = qq_nt[isp, :nh, :nh]
    H += beta @ (Dbig @ beta.conj().T)
    H = 0.5 * (H + H.conj().T)
    S = np.eye(n, dtype=complex) + beta @ (Qbig @ beta.conj().T)
    S = 0.5 * (S + S.conj().T)

    # generalized eig via Cholesky whitening: H c = eps S c
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    Hw = Linv @ H @ Linv.conj().T
    ev = np.linalg.eigvalsh(Hw)[:len(eig_ref)]

    nocc = int(np.sum(occ > 1e-6))
    print("k%d  (d11=%.1f, %d occupied bands)" % (ik, d11, nocc))
    print("  rebuilt H c = eps S c :", np.round(ev[:min(8, len(ev))], 5))
    print("  ABINIT (h5 eig)       :", np.round(eig_ref[:min(8, len(ev))], 5))
    print("  max|dif| all bands    : %.4e Ha" % np.max(np.abs(ev - eig_ref)))
    print("  max|dif| OCCUPIED     : %.4e Ha" % np.max(np.abs(ev[:nocc] - eig_ref[:nocc])))


if __name__ == "__main__":
    main()
