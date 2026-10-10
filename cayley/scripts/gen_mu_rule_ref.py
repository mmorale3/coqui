#!/usr/bin/env python3
"""Reference of the finite-T chemical-potential rule (notes section 11.6, Eq. fT_nth; plan S8b T5(b) unit case) for the C++
test [gw_line][finiteT][mu_rule]: the KS spectra of lih222 (insulator) and svo222 (metal) with the injected Lehmann weight errors
of dev/s8b_mu_rule.py (weights of the poles below the clean number-rule mu scaled by sqrt(1 + err 2 nk / (2 n_occ(k)))),
at beta = 200 and 100, err in {0, -1e-3, +1e-3, -5e-2, +5e-2}; rules "auto" (closure.chemical_potential_auto), "gap", "number".
Output tests/unit_test_files/gw_line/mu_rule_ref.h5: per case group c<i>: attrs fixture, beta, err, rule_auto, mu_auto, dN, n_th,
mu_gap, mu_number, N_number; the C++ test rebuilds the same weights from <fx>_thc/system.h5 (eigval, nelec).
Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_mu_rule_ref.py"""
import sys, os, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.line.closure import chemical_potential, chemical_potential_T, chemical_potential_auto
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = D + 'mu_rule_ref.h5'
with h5py.File(out, 'w') as F:
    F.attrs.update(generator='coqui/cayley/scripts/gen_mu_rule_ref.py', mu_dn_max=0.1, mu_th_factor=10.0)
    i = 0
    for fx in ['lih222', 'svo222']:
        with h5py.File(D + f'{fx}_thc/system.h5', 'r') as f:
            eig = np.array(f['system/eigval']); nelec = float(f['system/nelec'][()])
        nk, nb = eig.shape
        e = [eig[k] for k in range(nk)]
        for beta in [200.0, 100.0]:
            v0 = [np.eye(nb, dtype=complex) for k in range(nk)]
            mu_ref, _ = chemical_potential_T(e, v0, nk, nelec, beta)
            for err in [0.0, -1e-3, 1e-3, -0.05, 0.05]:
                v = []
                for k in range(nk):
                    V = np.eye(nb, dtype=complex); m = eig[k] < mu_ref
                    V[:, m] *= np.sqrt(1.0 + err * 2.0 * nk / (2.0 * m.sum() if m.sum() else 1)); v.append(V)
                mu_a, rule, N_a, info = chemical_potential_auto(e, v, nk, nelec, beta, return_info=True)
                mu_g = chemical_potential(e, v, nk, nelec)[0]
                mu_n, N_n = chemical_potential_T(e, v, nk, nelec, beta)
                g = F.create_group(f'c{i}'); i += 1
                g.attrs.update(fixture=fx, beta=beta, err=err, mu_ref=mu_ref, rule_auto=rule, mu_auto=mu_a, dN=info['dN'], n_th=info['n_th'],
                               mu_gap=mu_g, mu_number=mu_n, N_number=N_n, nelec=nelec)
                print(f"{fx} beta {beta:4.0f} err {err:+.0e}: auto {rule:6s} mu {mu_a:+.12f}  gap {mu_g:+.12f}  number {mu_n:+.12f}  dN {info['dN']:+.2e} n_th {info['n_th']:.2e}")
    F.attrs['ncases'] = i
print('wrote', out)
