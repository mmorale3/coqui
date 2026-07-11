"""
End-to-end validation of the /Hamiltonian WRITER: read the pp data back from
the converter's output h5 and rebuild H_KS(k) purely from what was written, then compare
to the ABINIT eigenvalues stored in /Orbitals/eigval.  A machine-precision match proves
the writer serialized beta(k+G), dion, scf_local_potential, and the grids correctly.

Usage: python validate_h0_from_h5.py abinit.h5 [ik]
"""
import sys
import numpy as np
import h5py


def rd_cplx(g, name):
    a = g[name][:]
    return a[..., 0] + 1j * a[..., 1]


def main():
    fn = sys.argv[1]
    ik = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    f = h5py.File(fn, "r")
    recv = f["System/reciprocal_vectors"][:]
    kcry = f["System/BZ/kpoints_crystal"][:][ik]
    eig_ref = f["Orbitals/eigval"][:][0, ik, :]
    g = f["Hamiltonian/ncpp"]
    miller_g = g["miller_g"][:]
    # potentials + dion are stored in Rydberg (CoQui applies x0.5 -> Hartree); do the same
    vks = 0.5 * rd_cplx(g, "scf_local_potential")[0, 0, :]   # (ngm,) Rydberg -> Hartree
    dion = 0.5 * g["dion"][:]                                # (nsp,nhm,nhm) Rydberg -> Hartree
    nh_atom = g["proj_per_atom"][:]
    off = g["projector_offset"][:]
    atid = g["atomic_id"][:]
    mill_k = g["miller_k" + str(ik)][:]
    beta = rd_cplx(g, "projector_k" + str(ik)).T            # stored (nkb,npw) -> (npw,nkb)
    f.close()

    n = mill_k.shape[0]
    kcart = kcry @ recv
    kpg = kcart[None, :] + mill_k @ recv
    T = 0.5 * np.einsum("ij,ij->i", kpg, kpg)

    # rebuild V_KS box from miller_g + vks, then V(G-G')
    ngf = [int(miller_g[:, d].max() - miller_g[:, d].min() + 1) for d in range(3)]
    Vbox = np.zeros(ngf, dtype=complex)
    Vbox[miller_g[:, 0] % ngf[0], miller_g[:, 1] % ngf[1], miller_g[:, 2] % ngf[2]] = vks
    d1 = (mill_k[:, None, 0] - mill_k[None, :, 0]) % ngf[0]
    d2 = (mill_k[:, None, 1] - mill_k[None, :, 1]) % ngf[1]
    d3 = (mill_k[:, None, 2] - mill_k[None, :, 2]) % ngf[2]
    H = Vbox[d1, d2, d3].copy()
    H[np.diag_indices(n)] += T

    # nonlocal: block-diagonal Dbig over atoms, H_NL = beta Dbig beta^dagger
    nkb = beta.shape[1]
    Dbig = np.zeros((nkb, nkb), dtype=float)
    for a in range(len(nh_atom)):
        nh = int(nh_atom[a]); o = int(off[a]); isp = int(atid[a])
        Dbig[o:o + nh, o:o + nh] = dion[isp, :nh, :nh]
    H += beta @ (Dbig @ beta.conj().T)
    H = 0.5 * (H + H.conj().T)

    ev = np.linalg.eigvalsh(H)[:len(eig_ref)]
    print("rebuilt-from-h5 :", np.round(ev, 5))
    print("ABINIT (h5 eig) :", np.round(eig_ref, 5))
    print("max |dif|       : %.4e Ha" % np.max(np.abs(ev - eig_ref)))


if __name__ == "__main__":
    main()
