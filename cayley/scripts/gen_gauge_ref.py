#!/usr/bin/env python3
"""Free-terminal-block oracle for the C++ [gw_line][finiteT][gauge] test (plan S8b.4 task 4a, notes section 11.10 sec:fT_gauge).

The Si 2x2x2 Gamma moments of tests/unit_test_files/gw_line/si222_moments_k0.h5 (group si_k0) at K = 4 (Gram rank 290 > K r = 232:
n_free = 58) realized by cayley.upfold.upfold_block(..., free_block=rule) for rule = "orbital" / "canonical" (nphi 8, tol_gram of the
file). Output tests/unit_test_files/gw_line/si222_gauge_ref.h5:
  root attrs: K, wp, tol_gram, nphi, generator, cayley_head, date; dataset mixing_F_re/_im (58 x 58, mixing_unitary(58))
  /<rule>/  attrs n_free, rank_gram, phi, heldout_err, sig_min (sigma_min / sigma_max of the polar problem);
            datasets sigma_re/_im (25, n, n) = Sigma_c(z) = sum_l W_l W_l^dag / (z - d_l) at z = -3 + 0.25 p + 0.05 i, p = 0..24
Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_gauge_ref.py"""
import os, sys, datetime, subprocess, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.upfold import upfold_block, mixing_unitary

D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
K, nphi = 4, 8
with h5py.File(D + 'si222_moments_k0.h5', 'r') as f:
    g = f['si_k0']
    C = g['C_re'][()] + 1j * g['C_im'][()]
    wp, tg = float(g.attrs['wp']), float(g.attrs['tol_gram'])
C = C[:K + 2]
z = -3.0 + 0.25 * np.arange(25) + 0.05j
try:
    head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception:
    head = '?'
out = D + 'si222_gauge_ref.h5'
with h5py.File(out, 'w') as f:
    f.attrs.update(K=K, wp=wp, tol_gram=tg, nphi=nphi, generator='coqui/cayley/scripts/gen_gauge_ref.py', cayley_head=head,
                   date=datetime.date.today().isoformat())
    F = mixing_unitary(C.shape[1])
    f['mixing_F_re'] = F.real; f['mixing_F_im'] = F.imag
    for rule in ('orbital', 'canonical'):
        d, W, info = upfold_block(C, K, wp, tol_gram=tg, nphi=nphi, return_info=True, free_block=rule)
        S = np.einsum('il,jl,pl->pij', W, W.conj(), 1.0 / (z[:, None] - d[None, :]))
        h = f.create_group(rule)
        h.attrs.update(n_free=info['n_free'], rank_gram=info['rank_gram'], phi=info['phi'], heldout_err=info['heldout_err'],
                       sig_min=info['sig_min'])
        h['sigma_re'] = S.real; h['sigma_im'] = S.imag
        print(f"{rule}: n_free {info['n_free']} Nr {info['rank_gram']} phi {info['phi']:.10f} held-out {info['heldout_err']:.6e} "
              f"sigma ratio {info['sig_min']:.6e}", flush=True)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
