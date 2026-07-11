"""
validate_qvan.py -- validate the paw_qvan kernel against a real QE-PAW reference.

Reads the Species radial data (qfuncl, r, rab, indv, nhtolm, lll, kkbeta) and the
dense G grid (miller_g) from a QE pw2coqui h5, recomputes Q^IJ(G), and compares to
QE's own `augmentation_function_isp{nt}` stored in the same file.

Usage:
    python validate_qvan.py [path/to/pwscf.coqui.h5]
"""
import sys
import numpy as np
import h5py
import paw_qvan as pq

H5 = sys.argv[1] if len(sys.argv) > 1 else \
    "../../../../tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5"


def selftest():
    # ylmr2: Y_00 = 1/sqrt(4pi); orthonormality over a Lebedev-ish random sample
    g = np.array([[0, 0, 1.0]])
    y = pq.ylmr2(1, g)
    assert abs(y[0, 0] - np.sqrt(1 / (4 * np.pi))) < 1e-12, y[0, 0]
    rng = np.random.default_rng(0)
    N = 200000
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    Y = pq.ylmr2(16, v)                     # up to l=3
    gram = (Y.T @ Y) / N * (4 * np.pi)
    off = np.abs(gram - np.eye(16))
    print("[selftest] ylmr2 orthonormality max|Gram-I| = %.2e" % off.max())
    # sph_jn vs closed forms j0,j1,j2
    x = np.linspace(0.1, 25, 50)
    j = pq.sph_jn_all(2, x)
    j0 = np.sin(x) / x
    j1 = np.sin(x) / x**2 - np.cos(x) / x
    j2 = (3 / x**3 - 1 / x) * np.sin(x) - 3 / x**2 * np.cos(x)
    print("[selftest] sph_jn max|dj0|=%.1e |dj1|=%.1e |dj2|=%.1e"
          % (np.abs(j[0] - j0).max(), np.abs(j[1] - j1).max(), np.abs(j[2] - j2).max()))


def main():
    selftest()
    f = h5py.File(H5, "r")
    S = f["System"]
    lat = S["lattice_vectors"][:]
    recip = S["reciprocal_vectors"][:]
    omega = abs(np.linalg.det(lat))
    nsp = int(f["Hamiltonian/paw"].attrs["number_of_species"])
    mg = f["Hamiltonian/paw/miller_g"][:]
    print("omega=%.6f  ngm=%d  nsp=%d" % (omega, mg.shape[0], nsp))

    worst = 0.0
    for nt in range(nsp):
        sp = f["Hamiltonian/Species/nt%d" % nt]
        qfuncl = sp["qfuncl"][:]                 # (nqlc, nij_beta, nr)
        r = sp["r"][:]
        rab = sp["rab"][:]
        lll = sp["lll"][:].astype(int)
        indv = sp["indv"][:].astype(int)         # 1-based
        nhtolm = sp["nhtolm"][:].astype(int)     # 1-based
        nh = int(sp.attrs["nh"])
        nqlc = int(sp.attrs["nqlc"])
        kkbeta = int(sp.attrs["kkbeta"])

        qgm = pq.augmentation_function(qfuncl, r, rab, kkbeta, nqlc, lll,
                                       indv, nhtolm, nh, omega, mg, recip)

        ref = f["Hamiltonian/paw/augmentation_function_isp%d" % nt][:]
        ref = ref[..., 0] + 1j * ref[..., 1]     # (nij_proj, ngm)
        num = np.abs(ref).max()
        adiff = np.abs(qgm - ref)
        rel = adiff.max() / num
        print("nt%d: nh=%d nij=%d  max|ref|=%.3e  max|abs diff|=%.3e  rel=%.3e"
              % (nt, nh, qgm.shape[0], num, adiff.max(), rel))
        # a couple of spot rows
        for row in (0, qgm.shape[0] // 2, qgm.shape[0] - 1):
            print("   row %3d: max|ref|=%.3e reldiff=%.3e"
                  % (row, np.abs(ref[row]).max(),
                     np.abs(qgm[row] - ref[row]).max() / (np.abs(ref[row]).max() + 1e-30)))
        worst = max(worst, rel)
    f.close()
    print("\nWORST relative error over all species/pairs/G = %.3e" % worst)
    print("PASS" if worst < 1e-5 else "FAIL (investigate)")


if __name__ == "__main__":
    main()
