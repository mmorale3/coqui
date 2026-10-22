#!/usr/bin/env python3
"""D2: python references of the line GW schemes (scheme = scgw0 | evgw | evgw0; cayley.line.driver "Schemes") for the C++
test test_gw_line_scf [gw_line][scheme], lih222 fixture. Settings, inputs and the C++-bases replacement exactly as
gen_lih222_scf_ref.py (its docstring); mixing 0.5 (Sigma for scgw0; QP energies for evgw/evgw0 from iteration 2).
Output tests/unit_test_files/gw_line/lih222_<scheme>_ref.h5: per iteration mu, gap (Ha), dSigma, [scgw0: nelec, nelec_lehmann],
[ev: dE, dE_win (ev_conv "window" metric, Z > 0.5, all bands), E_qp (niter, nk, nb) absolute Ha, Z_qp], Sigma_k0_re/_im at every 4th node (Sigma of the iteration, unmixed for ev).
Usage: gen_lih222_scheme_ref.py <scheme> [niter=3] [--lam-b L (12)] [--out FILE]
The committed lih222_evgw{,0}_ref.h5 predate dE_win (2026-10-22): it was added from their stored E_qp / Z_qp with the same rule
(KS energies / Z = 1 before iteration 1); a regeneration reproduces it to 4e-7 Ha (E_qp drift 1e-5 Ha vs the stored file)."""
import sys, os, time, functools, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
import cayley.line.driver as drv
from cayley.line.closure import lehmann_from_sigma
from cayley.coqui_io import THC, _c

def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
vals = {opt(o, None) for o in ('--out', '--lam-b', '--bases')} - {None}
args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in vals]
native = '--native-bases' in sys.argv
lam_b = float(opt('--lam-b', 12.0))
scheme = args[0]
niter = int(args[1]) if len(args) > 1 else 3
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); mu0 = float(s['mu0'][()]); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, nb = X.shape[0], X.shape[2]
assert np.all(qk[0] == np.arange(nk)), "python hartree_exchange assumes q index 0 = Gamma"
drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=8)
from cayley.line.closure import qp_diag
drv.qp_diag = functools.partial(qp_diag, nphi=8)

sc = drv.LineSCGW(X, Z, qk, nk, nelec, H0, mu0, theta=np.deg2rad(20.0), eps=1e-8, lam=6.0, bos_lam=lam_b, bos_gap=0.02,
                  sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=8, tol_gram=1e-10, mixing=0.5,
                  nodes_per_ray=120, node_range=(1e-3, 60.0), scheme=scheme)
bases_file = opt('--bases', D + ('lih222_thc/bases.h5' if lam_b == 4.0 else f'lih222_thc/bases_lamb{int(round(lam_b))}.h5'))
if not native:
    print(f"  bases from {bases_file}")
    with h5py.File(bases_file, 'r') as f:
        for b, key in ((sc.bp, 'sigma_particle_w'), (sc.bh, 'sigma_hole_w'), (sc.gp, 'g_particle_w'), (sc.gh, 'g_hole_w')):
            w = np.array(f[key]); print(f"  basis {key}: python rank {b.r}, C++ rank {len(w)}, max|dw| "
                                         f"{np.abs(b.w - w).max() if len(w) == b.r else float('nan'):.2e}")
            b.w = w; b.r = len(w); b.pos = w > 0; b.zeta = None; b.K = None
        nu = np.array(f['bos_nu']); print(f"  bosonic: python rank {sc.bos.r}, C++ rank {len(nu)}")
        sc.bos.nu = nu; sc.bos.r = len(nu); sc.bos.zeta = _c(f['bos_zeta_nodes'][()])
sc.start_from_hamiltonian(np.array([np.diag(eig[k]) for k in range(nk)]).astype(complex))
idx = np.arange(0, len(sc.fz), 4)
ev = scheme in ('evgw', 'evgw0')
H = {k: [] for k in ('mu', 'gap', 'dSigma', 'nelec', 'nelec_lehmann', 'dE', 'dE_win')}
Sig, Eq, Zq = [], [], []
t0 = time.time()
for it in range(niter):
    rec = sc.iterate()
    H['mu'].append(rec['mu']); H['gap'].append(rec['gap_eV'] / 27.211386); H['dSigma'].append(rec['dSigma'])
    if ev:
        H['dE'].append(rec['dE']); H['dE_win'].append(rec['dE_win']); Eq.append(sc.ev_E.copy()); Zq.append(sc.ev_Z.copy())
    else:
        H['nelec'].append(rec['nelec']); H['nelec_lehmann'].append(rec['nelec_exact'])
    Sig.append(sc.sigma_total(0)[idx])
Sig = np.array(Sig)
out = opt('--out', D + f'lih222_{scheme}_ref.h5')
with h5py.File(out, 'w') as f:
    for k, v in H.items():
        if len(v): f[k] = np.array(v, float)
    if ev:
        f['E_qp'] = np.array(Eq); f['Z_qp'] = np.array(Zq)
    f['node_index'] = idx.astype(np.int64)
    f['Sigma_k0_re'] = Sig.real; f['Sigma_k0_im'] = Sig.imag
    f.attrs.update(scheme=scheme, mu0=mu0, nelec=nelec, theta_deg=20.0, eps=1e-8, lam=6.0, lam_b=lam_b, sigma_gap=0.02, bos_gap=0.02,
                   g_gap=0.0, nodes_per_ray=120, node_tmin=1e-3, node_tmax=60.0, wp=0.11, K=8, tol_gram=1e-10, nphi=8, mixing=0.5,
                   cpp_bases=int(not native), bases=("" if native else os.path.basename(bases_file)))
print(f"wrote {out}: {scheme} {niter} iterations in {time.time() - t0:.0f} s")
