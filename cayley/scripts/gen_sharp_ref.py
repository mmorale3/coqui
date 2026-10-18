#!/usr/bin/env python3
"""Reference of the sharper final pass test [gw_line][finiteT][sharp] (plan S8c task 3, notes section 12.6): the EXACT finite-T
G0W0@KS correlation self-energy Sigma_c(k, w + i eta) (Eq. fT_sigma with the finite-T Casida W of the same THC, the oracle of the
T1 / T2 references, cayley.metal.SigmaFT direct route, no binning) on the real-axis grid |w - mu0| <= 0.3 Ha (151 points) at
eta = 0.02 and 0.05 Ha, for lih222 at beta = 50 (k = 0, 3) and svo222 at beta = 200 (k = 1). The C++ test builds the exact
A(k, w) = (i / 2 pi) Tr(G - G^dag), G = [w + mu0 + i eta - (H0 + F_closure) - Sigma_c]^{-1}, with the static part of its own run
(H0 + F[D_KS], identical for both: Sigma_c alone is the oracle) and compares the pass's spectra (spectra.origin = mu0).
Output tests/unit_test_files/gw_line/sharp_ref.h5: group <fixture>: attrs beta, mu0 (absolute, the KS number-rule mu), nelec,
sigma_k; datasets om (w - mu0), eta, Sw_re / Sw_im (n_k, n_eta, n_w, nb, nb) float32 (A error budget 1e-7).
Casida solutions from the cache data/casida_cache/<fixture> (as dev/s8c_assess_exact.py).
Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_sharp_ref.py"""
import sys, os, time, datetime, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC
from cayley import finite_t as ft, metal
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = D + 'sharp_ref.h5'
om = np.linspace(-0.3, 0.3, 151); etas = [0.02, 0.05]
cases = [('lih222', 50.0, [0, 3]), ('svo222', 200.0, [1])]
with h5py.File(out, 'w') as F:
    F.attrs.update(generator='coqui/cayley/scripts/gen_sharp_ref.py', date=datetime.date.today().isoformat(),
                   route='cayley.metal.SigmaFT.evaluate(zeta=...) direct, casida_hermitian', deg_tol=1e-8)
    for fx, beta, ks in cases:
        t0 = time.time()
        casida = metal.cached(metal.casida_hermitian, f'{ROOT}/data/casida_cache/{fx}', verbose=False)
        thc = THC(D + f'{fx}_thc/thc.eri.h5')
        with h5py.File(D + f'{fx}_thc/system.h5', 'r') as f:
            s = f['system']; eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
        X, Z = thc.X[0], thc.Z; nk, nb = eig.shape; qminus = qk[:, 0].copy()
        mu0, rule = ft.mu0_auto(eig, nelec, beta, 1e-8); e = eig - mu0
        cas = [casida(ft.transitions(X, e, qk, iq, beta, deg_tol=1e-8), Z[iq])[:3] for iq in range(nk)]
        z = np.concatenate([om + 1j * eta for eta in etas])
        Sw = np.zeros((len(ks), len(etas), len(om), nb, nb), complex)
        for a, ik in enumerate(ks):
            r = metal.SigmaFT(X, e, qk, qminus, cas, beta, ik).evaluate(zeta=z)
            Sw[a] = r['zeta'].reshape(len(etas), len(om), nb, nb)
        g = F.create_group(fx)
        g.attrs.update(beta=beta, mu0=mu0, mu_rule=rule, nelec=nelec, sigma_k=np.array(ks))
        g.create_dataset('om', data=om); g.create_dataset('eta', data=np.array(etas))
        g.create_dataset('Sw_re', data=Sw.real.astype(np.float32)); g.create_dataset('Sw_im', data=Sw.imag.astype(np.float32))
        print(f"{fx} beta {beta}: mu0 {mu0:.12f} ({rule}), k {ks}, max|Sigma_c| {np.abs(Sw).max():.3e} [{time.time() - t0:.0f} s]", flush=True)
print('wrote', out)
