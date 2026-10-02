#!/usr/bin/env python3
"""Reference data for the C++ Cayley closure numerics (src/numerics/line_dlr/cayley.hpp, test_cayley, plan session S2).

Writes <coqui>/tests/unit_test_files/gw_line/si222_moments_k0.h5 with
  group si_k0   (from results/si222_g0w0_k0_wp0.110_gapmid.npz, written by scripts/si222_g0w0_casida.py)
      C_re, C_im (18, nb, nb)   exact Cayley moments C^(0..17) of the Casida G0W0 Sigma_c (k=0) about the centre mu
      H_re, H_im (nb, nb)       H0 + F_1 (absolute energies, Ha)
      om (nw,)                  absolute frequencies (Ha); Aex_tr_eta0.01 (nw,) exact Tr A of the same Sigma_c
      A_total_K8_eta0.01, A_total_K16_eta0.01   Tr A stored in the npz (total-measure upfolding, tol_gram=1e-13, nphi=72)
      A_py_K{K}_nphi{nphi}_eta0.01             Tr A recomputed here with the current python (K = 8, 16; nphi = 72, 8)
      attrs: mu, wp, tol_gram, eta, win_halfwidth, and per K in (8, 16) and nphi in (72, 8):
             K{K}_nphi{nphi}_{npoles, phi, residual, rank_c0, rank_gram, n_free, win_err}
             (win_err = max |TrA - TrA_ex| over |om - mu| < 0.19 / max TrA_ex over the same window, the metric of
              si222_g0w0_casida.py) and K{K}_win_err_stored (same metric on the stored npz curves).
  group mu_toy  a 2-k Lehmann G for closure.chemical_potential: a wide QP gap plus a noise-split valence multiplet
                (0.4 meV "gap") that is also admissible and even has the better electron count.
      e_k{k} (M,), v_k{k}_re (n, M), k_weight (nk,); attrs nelec, qp_weight, ntol and the python result
      mu, e_homo, e_lumo, N.
Usage: cd coqui/cayley && python3 scripts/gen_cayley_ref.py"""
import os, sys, time
import numpy as np
import h5py
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from cayley import upfold_block
from cayley.spectral import sigma_from_poles, spectral_function
from cayley.line.closure import chemical_potential

ROOT = os.environ.get('CAYLEY_ROOT', os.path.abspath(os.path.join(here, '..', '..', '..')))
src = os.path.join(ROOT, 'results', 'si222_g0w0_k0_wp0.110_gapmid.npz')
out = os.path.join(here, '..', '..', 'tests', 'unit_test_files', 'gw_line', 'si222_moments_k0.h5')
os.makedirs(os.path.dirname(out), exist_ok=True)

ref = np.load(src)
mu, wp = float(ref['mu']), float(ref['wp'])
C, H, om = ref['C'], ref['H'], ref['om']
eta, TOL_GRAM, NMOM = 0.01, 1e-13, 18                  # si222_g0w0_casida.py: total measure, tol_gram=1e-13
win = np.abs(om - mu) < 0.19                           # ~ +-5 eV, as in si222_g0w0_casida.py
X = ref[f'Aex_tr_eta{eta}']
win_err = lambda trA: np.abs(trA - X)[win].max() / X[win].max()

t0 = time.time()
with h5py.File(out, 'w') as f:
    g = f.create_group('si_k0')
    g['C_re'] = C[:NMOM].real; g['C_im'] = C[:NMOM].imag
    g['H_re'] = H.real; g['H_im'] = H.imag
    g['om'] = om
    g[f'Aex_tr_eta{eta}'] = X
    for K in (8, 16):
        g[f'A_total_K{K}_eta{eta}'] = ref[f'A_total_K{K}_eta{eta}']
        g.attrs[f'K{K}_win_err_stored'] = win_err(ref[f'A_total_K{K}_eta{eta}'])
    g.attrs.update(mu=mu, wp=wp, tol_gram=TOL_GRAM, eta=eta, win_halfwidth=0.19)
    for K in (8, 16):
        for nphi in (72, 8):
            d, W, info = upfold_block(C, K, wp, mu, tol_gram=TOL_GRAM, nphi=nphi, return_info=True)
            trA = spectral_function(H, sigma_from_poles(d, W), om, eta, trace=True)
            e = win_err(trA)
            p = f'K{K}_nphi{nphi}_'
            g[f'A_py_K{K}_nphi{nphi}_eta{eta}'] = trA
            g.attrs.update({p + 'npoles': len(d), p + 'phi': info['phi'], p + 'residual': info['heldout_err'],
                            p + 'rank_c0': info['rank_c0'], p + 'rank_gram': info['rank_gram'],
                            p + 'n_free': info['n_free'], p + 'win_err': e})
            if nphi == 72:
                print(f"  K={K} nphi={nphi}: max|trA - stored npz| {np.abs(trA - ref[f'A_total_K{K}_eta{eta}']).max():.1e}")
            print(f"si K={K:2d} nphi={nphi:2d}: poles {len(d)} rank_gram {info['rank_gram']} n_free {info['n_free']} "
                  f"phi {info['phi']:.10f} residual {info['heldout_err']:.3e} win err {e:.3e}  "
                  f"(stored npz curve {g.attrs[f'K{K}_win_err_stored']:.3e})  [{time.time() - t0:.0f}s]", flush=True)

    # mu finder toy: 2 orbitals, 2 k points (weights 1:1), nelec = 2 (one band, spin 2). Per k, columns of v are the
    # orbital components of the Lehmann poles. Valence QP poles carry some orbital-2 admixture so that the hole weight
    # below the physical gap overshoots (N = 2.2 at the wide-gap midpoint, |dN| = 0.2). At k0 the valence QP is split
    # by "noise" into two poles 1.5e-5 Ha apart (weights 0.85 and 0.15, both > qp_weight): the spurious 0.4 meV gap has
    # N = 2.05, i.e. the BETTER count, but the rule picks the widest admissible gap.
    t = f.create_group('mu_toy')
    s = np.sqrt
    # k0: satellite, split valence QP (0.85 + 0.15), conduction QP, satellite; orbital weights sum to 1 per orbital
    e0 = np.array([-1.00, -0.30, -0.30 + 1.5e-5, 0.15, 0.90])
    v0 = s(np.array([[0.05, 0.80, 0.15, 0.00, 0.00],
                     [0.05, 0.05, 0.00, 0.80, 0.10]]))
    # k1: satellite, valence QP, conduction QP, satellite
    e1 = np.array([-0.95, -0.32, 0.20, 0.85])
    v1 = s(np.array([[0.05, 0.95, 0.00, 0.00],
                     [0.05, 0.05, 0.85, 0.05]]))
    ev, vv = [e0, e1], [v0, v1]
    kw = np.array([1.0, 1.0]); nelec, qpw, ntol = 2.0, 0.1, 0.5
    res = chemical_potential(ev, vv, 2, nelec, kw, qp_weight=qpw, ntol=ntol)
    for k in range(2):
        t[f'e_k{k}'] = ev[k]; t[f'v_k{k}_re'] = vv[k]
    t['k_weight'] = kw
    t.attrs.update(nelec=nelec, qp_weight=qpw, ntol=ntol, mu=res[0], e_homo=res[1], e_lumo=res[2], N=res[3])
    print(f"mu toy: mu {res[0]:.6f} e_homo {res[1]:.6f} e_lumo {res[2]:.6f} N {res[3]:.6f}")
print('wrote', os.path.abspath(out), f'({os.path.getsize(out) / 1e6:.2f} MB)')
