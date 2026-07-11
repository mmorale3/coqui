"""
validate_deltaC.py -- validate paw_deltaC against QE's stored Onecenter/deltaC.

Reads pfunc/ptfunc/qfuncl + radial grid + maps from a QE pw2coqui h5, recomputes
the one-center Fock kernel deltaC(nh,nh,nh,nh), and compares to
/Hamiltonian/Species/nt{nt}/Onecenter/deltaC in the same file.
"""
import sys
import numpy as np
import h5py
from paw_deltaC import compute_deltaC

H5 = sys.argv[1] if len(sys.argv) > 1 else \
    "../../../../tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5"


def main():
    f = h5py.File(H5, "r")
    nsp = int(f["Hamiltonian/paw"].attrs["number_of_species"])
    worst = 0.0
    for nt in range(nsp):
        sp = f["Hamiltonian/Species/nt%d" % nt]
        pfunc = sp["paw/pfunc"][:]        # (nbeta,nbeta,nr)
        ptfunc = sp["paw/ptfunc"][:]
        qfuncl = sp["qfuncl"][:]          # (nqlc, nij_beta, nr)
        r = sp["r"][:]
        rab = sp["rab"][:]
        lll = sp["lll"][:].astype(int)
        indv = sp["indv"][:].astype(int)
        nhtolm = sp["nhtolm"][:].astype(int)
        nh = int(sp.attrs["nh"])
        lmax_rho = int(sp.attrs["lmax_rho"])

        dC = compute_deltaC(pfunc, ptfunc, qfuncl, r, rab, indv, nhtolm, lll, nh, lmax_rho)
        ref = f["Hamiltonian/Species/nt%d/Onecenter/deltaC" % nt][:]  # (nh,nh,nh,nh)

        num = np.abs(ref).max()
        adiff = np.abs(dC - ref)
        rel = adiff.max() / num
        # correlation of the flattened tensors (structure check independent of scale)
        a, b = dC.ravel(), ref.ravel()
        corr = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        scale = np.dot(a, b) / (np.dot(b, b) + 1e-30)   # best-fit dC ~ scale*ref
        print("nt%d nh=%d: max|ref|=%.4e  max|abs diff|=%.4e  rel=%.3e" % (nt, nh, num, adiff.max(), rel))
        print("        corr(dC,ref)=%.8f   best-fit scale dC=scale*ref: %.6f" % (corr, scale))
        # a few representative elements
        idx = [(0, 0, 0, 0), (0, 0, 5, 5), (2, 2, 2, 2), (nh - 1, nh - 1, nh - 1, nh - 1)]
        for I in idx:
            print("        deltaC%s  mine=% .6e  QE=% .6e" % (I, dC[I], ref[I]))
        worst = max(worst, rel)
    f.close()
    print("\nWORST relative error = %.3e" % worst)
    print("PASS" if worst < 1e-3 else ("CLOSE" if worst < 1e-2 else "FAIL"))


if __name__ == "__main__":
    main()
