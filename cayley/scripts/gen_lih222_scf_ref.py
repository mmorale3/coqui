#!/usr/bin/env python3
"""Python reference for the C++ line scGW driver (CoQui src/methods/GW_line, test_gw_line_scf [parity]), lih222 fixture.

Inputs (written once by the C++ hidden test case `test_gw_line_scf "[.gw_line_dump]"`, committed):
  tests/unit_test_files/gw_line/lih222_thc/thc.eri.h5   CoQui THC of qe_lih222 (nIpts = 8 nbnd = 128): X (ns, nk, Np, nb), Z (nq, Np, Np)
  tests/unit_test_files/gw_line/lih222_thc/system.h5    H0 (nk, nb, nb) (kinetic + pseudopotential, KS band basis), KS eigenvalues,
                                                        qk_to_k2, mu0 (KS mid-gap), nelec
  tests/unit_test_files/gw_line/lih222_thc/bases.h5     the C++ driver's real-pole bases for these settings. The column-pivoted QR
                                                        pole selection of the hole Sigma basis and of the bosonic basis differs
                                                        between CoQui's LAPACK and numpy/scipy's (near-tie pivots; both are valid
                                                        eps-bases, but the moments differ at 1e-5 and the upfold ranks by 2-7,
                                                        and with these settings mu drifts by 2-14 meV and the gap by up to
                                                        33 meV within 6 iterations), so by default the python bases are
                                                        REPLACED by the C++ ones (like-for-like numerics; then mu and the gap
                                                        agree to < 0.01 meV over 6 iterations). --native-bases keeps python's.
Run: LineSCGW from the KS poles at mu0 with the small settings of the C++ test (scf_params in test_gw_line_scf.cpp):
  theta 20 deg, eps 1e-8, lam 6, lam_b 4, sigma_gap = bos_gap = 0.02, g_gap 0, 120 nodes per ray in [1e-3, 60], wp 0.11, K 8,
  tol_gram 1e-10, nphi 8 (the C++ default; python's lehmann_from_sigma default is 72, patched here), mixing 0.5, 6 iterations.
Output: tests/unit_test_files/gw_line/lih222_scf_ref.h5 with per-iteration mu, gap (Ha), nelec (compressed), nelec_lehmann,
  dSigma, and the total Sigma at k = 0 at every 4th fermionic node (Sigma_k0_re/_im (niter, nsel, nb, nb), node_index).
Usage: gen_lih222_scf_ref.py [niter=6] [--native-bases] [--out FILE]"""
import sys, os, time, functools, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
import cayley.line.driver as drv
from cayley.line.closure import lehmann_from_sigma
from cayley.coqui_io import THC, _c

args = [a for a in sys.argv[1:] if not a.startswith('--')]
native = '--native-bases' in sys.argv
niter = int(args[0]) if args else 6
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); mu0 = float(s['mu0'][()]); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, nb = X.shape[0], X.shape[2]
assert np.all(qk[0] == np.arange(nk)), "python hartree_exchange assumes q index 0 = Gamma"
drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=8)

sc = drv.LineSCGW(X, Z, qk, nk, nelec, H0, mu0, theta=np.deg2rad(20.0), eps=1e-8, lam=6.0, bos_lam=4.0, bos_gap=0.02,
                  sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=8, tol_gram=1e-10, mixing=0.5,
                  nodes_per_ray=120, node_range=(1e-3, 60.0))
if not native:
    with h5py.File(D + 'lih222_thc/bases.h5', 'r') as f:
        for b, key in ((sc.bp, 'sigma_particle_w'), (sc.bh, 'sigma_hole_w'), (sc.gp, 'g_particle_w'), (sc.gh, 'g_hole_w')):
            w = np.array(f[key]); print(f"  basis {key}: python rank {b.r}, C++ rank {len(w)}, max|dw| "
                                         f"{np.abs(b.w - w).max() if len(w) == b.r else float('nan'):.2e}")
            b.w = w; b.r = len(w); b.pos = w > 0; b.zeta = None; b.K = None
        nu = np.array(f['bos_nu']); print(f"  bosonic: python rank {sc.bos.r}, C++ rank {len(nu)}")
        sc.bos.nu = nu; sc.bos.r = len(nu); sc.bos.zeta = _c(f['bos_zeta_nodes'][()])
sc.start_from_hamiltonian(np.array([np.diag(eig[k]) for k in range(nk)]).astype(complex))
idx = np.arange(0, len(sc.fz), 4)
mu, gap, nel, nell, dS, Sig = [], [], [], [], [], []
t0 = time.time()
for it in range(niter):
    rec = sc.iterate()
    mu.append(rec['mu']); gap.append(rec['gap_eV'] / 27.211386); nel.append(rec['nelec']); nell.append(rec['nelec_exact'])
    dS.append(rec['dSigma']); Sig.append(sc.sigma_total(0)[idx])
Sig = np.array(Sig)
out = D + 'lih222_scf_ref.h5'
if '--out' in sys.argv: out = sys.argv[sys.argv.index('--out') + 1]
with h5py.File(out, 'w') as f:
    for k, v in dict(mu=mu, gap=gap, nelec=nel, nelec_lehmann=nell, dSigma=dS).items():
        f[k] = np.array(v, float)
    f['node_index'] = idx.astype(np.int64)
    f['Sigma_k0_re'] = Sig.real; f['Sigma_k0_im'] = Sig.imag
    f.attrs.update(mu0=mu0, nelec=nelec, theta_deg=20.0, eps=1e-8, lam=6.0, lam_b=4.0, sigma_gap=0.02, bos_gap=0.02, g_gap=0.0,
                   nodes_per_ray=120, node_tmin=1e-3, node_tmax=60.0, wp=0.11, K=8, tol_gram=1e-10, nphi=8, mixing=0.5,
                   cpp_bases=int(not native))
print(f"wrote {out}: {niter} iterations in {time.time() - t0:.0f} s")
