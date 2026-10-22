#!/usr/bin/env python3
"""Reference for the T = 0 chemical potential with an isolated conduction valley (D2 SiC, 2026-10-22; closure.chemical_potential,
C++ numerics::line_dlr::chemical_potential, test cayley_chemical_potential_valley).

Writes <coqui>/tests/unit_test_files/gw_line/mu_valley.h5, group mu_valley: a 4-k Lehmann G (2 orbitals, nelec 2, k weights
3:2:2:1) whose k3 has an isolated low conduction QP (Z 0.85; the SiC X valley) at 0.05 Ha. The physical gap (k3 valence ->
k3 conduction, 0.34 Ha) has k-weighted N = 2.1; the wider spurious "gap" above the valley (0.05 -> 0.62 Ha) has N = 2.31, also
inside ntol = 0.5, so the widest-admissible rule (ntol_k = 0, pre-2026-10-22) picks it; the insulator filter (per-k count within
ntol_k = 0.5 of nelec at every k; k3 has 3.8 there) restores the physical gap. k0 keeps the noise-split valence multiplet of
mu_toy (gen_cayley_ref.py). Datasets e_k{k}, v_k{k}_re, k_weight; attrs nelec, qp_weight, ntol and, per ntol_k in (0.5, 0),
{new,old}_{mu, e_homo, e_lumo, N}.
Usage: cd coqui/cayley && python3 scripts/gen_mu_valley_ref.py"""
import os, sys
import numpy as np
import h5py
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from cayley.line.closure import chemical_potential

out = os.path.join(here, '..', '..', 'tests', 'unit_test_files', 'gw_line', 'mu_valley.h5')
s = np.sqrt
wv = s(np.array([[0.05, 0.95, 0.00, 0.00], [0.05, 0.00, 0.85, 0.10]]))
ev = [np.array([-1.00, -0.30, -0.30 + 1.5e-5, 0.62, 1.20]), np.array([-0.95, -0.32, 0.66, 1.10]),
      np.array([-0.90, -0.31, 0.70, 1.00]), np.array([-1.05, -0.29, 0.05, 0.90])]
vv = [s(np.array([[0.05, 0.80, 0.15, 0.00, 0.00], [0.05, 0.00, 0.00, 0.85, 0.10]])), wv, wv, wv]
kw = np.array([3.0, 2.0, 2.0, 1.0]); nelec, qpw, ntol = 2.0, 0.1, 0.5
with h5py.File(out, 'w') as f:
    t = f.create_group('mu_valley')
    for k in range(4):
        t[f'e_k{k}'] = ev[k]; t[f'v_k{k}_re'] = vv[k]
    t['k_weight'] = kw
    t.attrs.update(nelec=nelec, qp_weight=qpw, ntol=ntol)
    for tag, ntk in (('new', 0.5), ('old', 0.0)):
        r = chemical_potential(ev, vv, 4, nelec, kw, qp_weight=qpw, ntol=ntol, ntol_k=ntk)
        t.attrs.update({f'{tag}_mu': r[0], f'{tag}_e_homo': r[1], f'{tag}_e_lumo': r[2], f'{tag}_N': r[3]})
        print(f'{tag} (ntol_k {ntk}): mu {r[0]:.6f} homo {r[1]:.6f} lumo {r[2]:.6f} N {r[3]:.6f}')
print('wrote', os.path.abspath(out))
