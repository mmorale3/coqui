"""
Standalone validation of the /Hamiltonian physics (no CoQuí / QE needed).

Rebuilds the Kohn-Sham Hamiltonian in the plane-wave basis for one k-point,
  H(G,G') = 1/2|k+G|^2 delta_GG' + V_KS(G-G') + sum_a sum_i beta_ai(k+G) D_i beta_ai*(k+G')
using:
  - the ABINIT WFK (per-k G-vectors + reference eigenvalues),
  - the ABINIT POT file `vtrial` (total local KS potential on the FFT grid),
  - the psp8 pseudopotential (KB projectors + ekb), via abinit_hamiltonian.build_beta_k,
and compares the eigenvalues to ABINIT's.  A match validates beta(k+G), dion, and the
projector conventions (phase / normalization / lm-order / r-power).

Usage: python validate_h0.py WFK POT psp8 [ik]
"""
import sys
import numpy as np
from netCDF4 import Dataset
from abinit_psp8 import parse_psp8
import abinit_hamiltonian as ah


def load(wfk, pot):
    w = Dataset(wfk, "r")
    rprimd = np.array(w["primitive_vectors"][:])
    recv = 2 * np.pi * np.linalg.inv(rprimd).T
    kg = np.array(w["reduced_coordinates_of_plane_waves"][:]).astype(int)
    npw = np.array(w["number_of_coefficients"][:]).astype(int)
    eig = np.array(w["eigenvalues"][:])
    kpts = np.array(w["reduced_coordinates_of_kpoints"][:])
    xred = np.array(w["reduced_atom_positions"][:])
    w.close()
    p = Dataset(pot, "r")
    vtrial = np.array(p["vtrial"][:])
    p.close()
    return dict(rprimd=rprimd, recv=recv, kg=kg, npw=npw, eig=eig, kpts=kpts,
                xred=xred, vtrial=vtrial)


def vks_on_grid(vtrial, axis_order):
    """vtrial netCDF shape (nspin, na, nb, nc, ncomp). Squeeze to a 3D real grid,
    optionally transpose axes, and FFT to V_KS(G): Vg[m1,m2,m3] indexed by Miller."""
    v = np.array(vtrial[0, :, :, :, 0], dtype=float)     # (na,nb,nc)
    if axis_order == "reverse":
        v = np.transpose(v, (2, 1, 0))                    # netCDF C-order undo
    N = v.size
    Vg = np.fft.fftn(v) / N                                # <G|V|G'> = (1/N) sum_r V e^{-i dG.r}
    return Vg, v


def build_H(d, psp, ik, axis_order, nl_scale=1.0):
    recv, kg, npw, kpts, xred, rprimd = (d["recv"], d["kg"], d["npw"],
                                         d["kpts"], d["xred"], d["rprimd"])
    vol = abs(np.linalg.det(rprimd))
    tau = xred @ rprimd                                    # cartesian atom pos (bohr)
    Vg, _ = vks_on_grid(d["vtrial"], axis_order)
    n1, n2, n3 = Vg.shape
    n = int(npw[ik])
    mill = kg[ik][:n]                                       # (n,3)
    kcart = kpts[ik] @ recv
    kpg = kcart[None, :] + mill @ recv
    T = 0.5 * np.einsum("ij,ij->i", kpg, kpg)

    # local: V_KS(G-G')
    d1 = (mill[:, None, 0] - mill[None, :, 0]) % n1
    d2 = (mill[:, None, 1] - mill[None, :, 1]) % n2
    d3 = (mill[:, None, 2] - mill[None, :, 2]) % n3
    Hloc = Vg[d1, d2, d3].astype(complex)
    Hloc[np.diag_indices(n)] += T

    # nonlocal: sum over atoms of beta D beta^dagger
    Hnl = np.zeros((n, n), dtype=complex)
    for a in range(tau.shape[0]):
        beta, meta = ah.build_beta_k(psp, mill, kcart, recv, tau[a], vol)
        D = np.array([m[1] for m in meta])                 # ekb diag (Ha)
        Hnl += beta @ (D[:, None] * beta.conj().T)
    H = Hloc + nl_scale * Hnl
    H = 0.5 * (H + H.conj().T)
    return H, Hloc


def main():
    wfk, pot, psp8 = sys.argv[1], sys.argv[2], sys.argv[3]
    ik = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    d = load(wfk, pot)
    psp = parse_psp8(psp8)
    ref = d["eig"][0, ik, :]
    nr = len(ref)
    print("mean(vtrial) = %.6f Ha ;  ABINIT eig = %s" % (d["vtrial"].mean(), np.round(ref, 5)))

    # local-only baseline (no nonlocal)
    H, Hloc = build_H(d, psp, ik, "natural", nl_scale=0.0)
    evl = np.linalg.eigvalsh(Hloc)[:nr]
    print("\n[T+local only]      :", np.round(evl, 5), " max|dif| %.4e" % np.max(np.abs(evl - ref)))

    # scan projector conventions: r-power and nonlocal scale
    best = None
    for rp in (2, 1):
        ah._R_POWER = rp
        for nls in (1.0, 0.5, 2.0, 4.0, np.pi * 4):
            H, _ = build_H(d, psp, ik, "natural", nl_scale=nls)
            ev = np.linalg.eigvalsh(H)[:nr]
            dif = float(np.max(np.abs(ev - ref)))
            tag = "r_power=%d nl_scale=%.4g" % (rp, nls)
            print("  [%-24s] max|dif| %.4e  eig=%s" % (tag, dif, np.round(ev, 4)))
            if best is None or dif < best[0]:
                best = (dif, tag)
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
