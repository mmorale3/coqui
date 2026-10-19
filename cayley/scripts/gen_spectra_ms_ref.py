#!/usr/bin/env python3
"""D5 multi-scale spectra (C++ keys spectra.wp_scales / spectra.K; notes section "Spectra far from mu (D5)"): reference for the C++
test [gw_line][scf][spectra_ms].

Model (2 orbitals, mu-relative Ha, an Al-like plasmon satellite): H = [[-0.42, 0.01], [0.01, 0.30]],
Sigma_c(z) = sum_l b_l b_l^dag / (z - E_l) with E_l on a hole plasmon branch (-1.25 .. -0.55), a particle branch (0.55 .. 2.5) and a
low-energy e-h part (|E| < 0.5); b_l = sqrt(r_l) (cos phi_l, sin phi_l e^{i chi_l}). The EXACT poles are the closure input (no fit):
the test isolates closure + blend. Python oracle: closure.spectra_multiscale (scales [0.11 (K_base 24), 0.3, 0.8] (K 32)).
Printed: max|A - A_exact| / max A_exact in the satellite window w < -0.6 Ha for the base closure alone and for the blend; sum rules.

H5 (complex as <name>_re / _im): attrs K, K_base, tol_gram, nphi, generator, date; H (2,2); pole_w (P,); pole_g (P,2,2); scales (3,);
  omega (nw,); eta (2,); A_trace (2,nw) blend; A_diag (2,nw,2) blend; comp_trace (3,2,nw) per scale; chi (3,nw);
  m0_err, m1_err (2,); A_exact_trace (2,nw).
Usage (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_spectra_ms_ref.py [--out F]
"""
import sys, os, datetime, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.line.closure import spectra_multiscale

out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else ROOT + '/coqui/tests/unit_test_files/gw_line/spectra_ms_ref.h5'


def bump(c, s, lo, hi, n, wt):
    E = np.linspace(lo, hi, n); r = np.exp(-0.5 * ((E - c) / s) ** 2); return E, wt * r / r.sum()


parts = [bump(-0.78, 0.09, -1.25, -0.55, 160, 0.015), bump(0.95, 0.2, 0.55, 2.5, 160, 0.08),
         bump(-0.15, 0.08, -0.4, -0.002, 60, 0.004), bump(0.2, 0.1, 0.002, 0.5, 60, 0.006)]
E = np.concatenate([p[0] for p in parts]); R = np.concatenate([p[1] for p in parts]); P = len(E)
rng = np.random.default_rng(7)
phi = 0.25 * np.pi * (1 + 0.3 * rng.standard_normal(P)); chi_ = rng.uniform(0, 2 * np.pi, P)
B = np.sqrt(R)[:, None] * np.stack([np.cos(phi), np.sin(phi) * np.exp(1j * chi_)], 1)      # (P, 2)
g = np.einsum('li,lj->lij', B, B.conj())
H = np.array([[-0.42, 0.01], [0.01, 0.30]], complex)
scales, K, K_base, tol_gram, nphi = [0.11, 0.3, 0.8], 32, 24, 1e-10, 72
om = np.linspace(-1.5, 1.0, 1251); etas = [0.02, 0.05]
r = spectra_multiscale(H, E, g, scales, K, om, etas, K_base=K_base, tol_gram=tol_gram, nphi=nphi)
Aex = np.zeros((len(etas), len(om)))
for ie, eta in enumerate(etas):
    z = om + 1j * eta
    S = np.einsum('zl,lij->zij', 1.0 / (z[:, None] - E[None, :]), g)
    G = np.linalg.inv(z[:, None, None] * np.eye(2) - H[None] - S)
    Aex[ie] = -np.imag(np.trace(G, axis1=1, axis2=2)) / np.pi
sat = om < -0.6
for ie, eta in enumerate(etas):
    mx = Aex[ie].max()
    eb = np.abs(r['comp_trace'][0, ie] - Aex[ie])[sat].max() / mx
    em = np.abs(r['A_trace'][ie] - Aex[ie])[sat].max() / mx
    ea = np.abs(r['A_trace'][ie] - Aex[ie]).max() / mx
    print(f"eta {eta}: satellite window base {eb:.2e}  blend {em:.2e}; all w blend {ea:.2e}; sum rules m0 {r['m0_err'][ie]:.2e} m1 {r['m1_err'][ie]:.2e}")
with h5py.File(out, 'w') as f:
    f.attrs.update(K=K, K_base=K_base, tol_gram=tol_gram, nphi=nphi, generator='gen_spectra_ms_ref.py',
                   date=datetime.date.today().isoformat())
    for nm, x in [('H', H), ('pole_g', g)]:
        f[nm + '_re'] = x.real; f[nm + '_im'] = x.imag
    f['pole_w'] = E; f['scales'] = np.array(scales); f['omega'] = om; f['eta'] = np.array(etas)
    for k in ('A_trace', 'A_diag', 'comp_trace', 'chi', 'm0_err', 'm1_err'): f[k] = r[k]
    f['A_exact_trace'] = Aex
print('wrote', out, os.path.getsize(out) // 1024, 'kB')
