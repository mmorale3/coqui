"""
validate_paw_frontend.py -- validate the ABINIT-agnostic PAW radial front-end
(paw_radial.py) against the QE reference h5.

Uses QE's own radial partial waves (aewfc/pswfc, already in u=r*R form) as the
stand-in for what the ABINIT PAW-XML adapter will feed in, and checks that the
front-end reproduces QE's channel maps, moments, and augmentation overlaps.  The
compensation-charge builder is checked by moment preservation (the physical
correctness criterion; the exact analytic shape is checked separately against
the PAW-XML dataset).
"""
import sys
import numpy as np
import h5py
import paw_radial as pr

H5 = sys.argv[1] if len(sys.argv) > 1 else \
    "../../../../tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5"


def main():
    f = h5py.File(H5, "r")
    sp = f["Hamiltonian/Species/nt0"]
    lll = sp["lll"][:].astype(int)
    r = sp["r"][:]; rab = sp["rab"][:]
    mesh = int(sp.attrs["mesh"]); kk = int(sp.attrs["kkbeta"])
    nqlc = int(sp.attrs["nqlc"])
    aewfc = sp["aewfc"][:]; pswfc = sp["pswfc"][:]   # u = r*R form (QE)

    # 1) channel enumeration
    ch = pr.enumerate_channels(lll)
    ok = True
    for key in ("indv", "nhtol", "nhtolm"):
        ref = sp[key][:].astype(int)
        match = np.array_equal(ch[key], ref)
        ok &= match
        print("  %-7s match=%s" % (key, match))
    ijtoh_ref = f["Hamiltonian/paw/ijtoh"][0]
    m_ij = np.array_equal(ch["ijtoh"], ijtoh_ref)
    print("  ijtoh   match=%s   nh=%d" % (m_ij, ch["nh"]))
    ok &= m_ij

    # 2) pair products pfunc/ptfunc from partial waves
    pfunc = pr.pair_products(aewfc)
    ptfunc = pr.pair_products(pswfc)
    e_pf = np.abs(pfunc[:, :, :kk] - sp["paw/pfunc"][:, :, :kk]).max()
    e_pt = np.abs(ptfunc[:, :, :kk] - sp["paw/ptfunc"][:, :, :kk]).max()
    print("  pfunc  max|diff inside kkbeta| = %.2e" % e_pf)
    print("  ptfunc max|diff inside kkbeta| = %.2e" % e_pt)

    # 3) multipole moments vs augmom
    mom = pr.multipole_moments(pfunc, ptfunc, r, rab, lll, nqlc, mesh)
    augmom = sp["paw/augmom"][:]                      # (nqlc, nbeta, nbeta)
    emax = 0.0
    for mb in range(1, len(lll) + 1):
        for nb in range(1, mb + 1):
            ijv = pr.beta_pair_index(nb, mb)
            for L in range(nqlc):
                emax = max(emax, abs(mom[L, ijv] - augmom[L, nb - 1, mb - 1]))
    print("  moments max|q_ij^L - augmom| = %.2e" % emax)

    # 4) qqq and qq_nt
    qqq = pr.compute_qqq(mom, len(lll))
    e_qqq = np.abs(qqq - sp["qqq"][:]).max()
    qq_nt = pr.compute_qq_nt(qqq, ch["indv"], ch["nhtolm"])
    e_qq = np.abs(qq_nt - f["Hamiltonian/paw/qq_nt"][0]).max()
    print("  qqq    max|diff| = %.2e" % e_qqq)
    print("  qq_nt  max|diff| = %.2e" % e_qq)

    # 5) compensation builder: moment preservation with an analytic shape
    shape = pr.shape_gauss(r, sp.attrs["mesh"] and 1.2)   # rc ~ 1.2 Bohr Gaussian
    shape = np.where(r < 2.3, shape, 0.0)
    qfuncl = pr.build_qfuncl(mom, shape, r, rab, lll, nqlc, mesh)
    # check INT qfuncl_L r^L dr == moment_L
    emom = 0.0
    for L in range(nqlc):
        for ijv in range(mom.shape[1]):
            if mom[L, ijv] == 0.0:
                continue
            val = pr.qe_simpson_batch((qfuncl[L, ijv] * r ** L)[None, :], rab, mesh)[0]
            emom = max(emom, abs(val - mom[L, ijv]) / (abs(mom[L, ijv]) + 1e-30))
    print("  qfuncl moment-preservation max rel err = %.2e" % emom)

    f.close()
    good = ok and e_pf < 1e-12 and emax < 1e-8 and e_qqq < 1e-8 and e_qq < 1e-6 and emom < 1e-9
    print("\nFRONT-END", "PASS" if good else "CHECK")


if __name__ == "__main__":
    main()
