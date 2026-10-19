#!/usr/bin/env python3
"""D4 closure frame of the hybrid (C++ key hyb_frame; notes section 12 "The closure in the loop"): reference for the C++ test
[finiteT][hybrid][frame], lih222 at beta 200, theta_t = theta / 4, scf_density "matsubara", the [hybrid][scf] settings (K 8, nphi 8,
eps 1e-8, mu_rule auto), 2 iterations with hyb_frame "old" (the S8b.3 rule: closure at the previous mu, poles re-centred) and "new"
(the hybrid mu first, closure at H - mu_new).
Prototype gates: iteration-1 hybrid mu independent of the frame [1e-13]; "new" = "old" with H shifted by -dmu_hyb before the
closure: the iteration-1 Lehmann poles of "new" equal an explicit closure of the same Sigma fit at H - mu_new [1e-12].
H5: attrs beta, generator, cayley_head, date, the gates; /<frame>/{mu (2,), N_closure (2,), D_cl_err (2,), dmu_closure (2,)}.
Usage (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_frame_ref.py [--out F]"""
import sys, os, datetime, functools, subprocess, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC, _c
from cayley import finite_t as ft
from cayley.line import driver as drv
from cayley.line.thc_gw import WSTEP_DEFAULTS
from cayley.line.closure import lehmann_from_sigma, fit_sigma_total

out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else ROOT + '/coqui/tests/unit_test_files/gw_line/lih222_frame_ref.h5'
beta = 200.0
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, Np, nb = X.shape
mu0 = ft.mu_number(eig, nelec, beta)
drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=8)
P = dict(theta=np.deg2rad(20.0), eps=1e-8, lam=6.0, bos_lam=12.0, bos_gap=0.02, sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=8,
         tol_gram=1e-10, mixing=0.5, nodes_per_ray=120, node_range=(1e-3, 60.0), beta=beta, thermal_tol=1e-8, thermal_floor=30.0,
         thermal_floor_f=30.0, wp_floor=15.0, mu_rule='auto', bos_eps_T=1e-10, wstep=dict(WSTEP_DEFAULTS, cut_odd=1e-11),
         scf_density='matsubara', theta_t_frac=0.25, g_repr='lehmann', verbose=False)
R, gates = {}, {}
for frame in ('old', 'new'):
    sc = drv.LineSCGW(X, Z, qk, nk, nelec, H0, mu0, hyb_frame=frame, **P)
    sc.start_from_hamiltonian(np.array([np.diag(eig[k]) for k in range(nk)]).astype(complex))
    mu_s = sc.mu; F_s = sc.F.copy()
    rec = [sc.iterate()]
    if frame == 'new':        # explicit closure of the iteration-1 Sigma at H - mu_new vs the driver's poles (re-centred: none)
        Hn = H0 + F_s - rec[0]['mu'] * np.eye(nb)[None]
        ek = [drv.lehmann_from_sigma(Hn[k], *fit_sigma_total(sc.bt, sc.fz, sc.Sig_prev[k], sc.fmask), sc.history[0]['wp'], sc.K,
                                 tol_gram=sc.tol_gram)[0] for k in range(nk)]
        gates['explicit_closure'] = (max(np.abs(np.sort(ek[k]) - np.sort(sc.e_leh[k][sc.e_leh[k] < 1e5])).max() for k in range(nk)), 1e-12)
    rec.append(sc.iterate())
    R[frame] = {k: np.array([r[k] for r in rec]) for k in ('mu', 'N_closure', 'D_cl_err', 'dmu_closure')}
    print(frame, {k: v.tolist() for k, v in R[frame].items()}, flush=True)
gates['it1_mu_frame_independent'] = (abs(R['old']['mu'][0] - R['new']['mu'][0]), 1e-13)
for k, (v, g) in gates.items():
    print(f"  {k:26s} {v:.2e}  (gate {g:.0e})  {'ok' if v <= g else 'FAIL'}")
assert all(v <= g for v, g in gates.values()), "a gate failed: reference not written"
head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
with h5py.File(out, 'w') as f:
    f.attrs.update(dict(fixture='lih222', beta=beta, mu0=mu0, generator='gen_frame_ref.py', cayley_head=head, date=str(datetime.date.today())))
    for k, (v, g) in gates.items(): f.attrs['ver_' + k] = v
    for fr, d in R.items():
        for k, v in d.items(): f[f'{fr}/{k}'] = v
print(f"wrote {out}")
