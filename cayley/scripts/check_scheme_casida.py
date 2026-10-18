#!/usr/bin/env python3
"""D2: independent (continuation-free, exact) check of the line evGW / evGW0 on lih222.

One-step exact check per iteration i, same input G as the line: E_in = KS (i = 1) or the line's E_QP of iteration i - 1
(mu_in = its mid-gap); W = full RPA (Casida, T = 0) of the KS orbitals with E_in (evgw) or the KS energies (evgw0);
Sigma_c,nn(z) = exact pole sum over (q, n', s) (cayley.casida_g0w0); h = H0 + F[D_KS] - mu_in. The exact QP energy is the root
of E = h_nn + Sigma_nn(E) in the pole interval that contains the line's unmixed QP energy of iteration i; when its Z > 0.5 it is
CERTIFIED to be the largest-Z root (sum Z = 1), i.e. the same definition as the line (closure.qp_secular). Reported: max
|E_line - E_exact| over certified (k, band) and at the band edges (VB top, CB bottom bands). No line / fit / closure in the exact
side; the line's error does not accumulate (each iteration starts from the line's own input).
Usage: check_scheme_casida.py <scheme evgw|evgw0> <ref.h5> [--out results.npz]"""
import sys, os, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC, _c
from cayley import finite_t as ft
from cayley.casida_g0w0 import CasidaG0W0
from cayley.line.thc_gw import LineGW
from cayley.line.closure import qp_secular

scheme, ref = sys.argv[1], sys.argv[2]
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); mu0 = float(s['mu0'][()]); nelec = float(s['nelec'][()])
with h5py.File(ref, 'r') as f:
    Eline = np.array(f['E_qp']); mix = float(f.attrs['mixing'])
X, Z = thc.X[0], thc.Z
nk, nb = X.shape[0], X.shape[2]
nocc = int(round(nelec / 2))
# F[D_KS] (H0 is in the KS band basis: D_KS = diag(occupations))
Dm = np.array([np.diag((eig[k] < mu0).astype(float)).astype(complex) for k in range(nk)])
lg = LineGW.__new__(LineGW); lg.X, lg.Z, lg.nk, lg.qk, lg.nb, lg.Np = X, Z, nk, qk, nb, X.shape[1]
F = LineGW.hartree_exchange(lg, Dm)
hdiag = np.real(np.einsum('kii->ki', H0 + F))


def casida_cache(E, mu):
    e = E - mu
    cache = {}
    for iq in range(nk):
        tr = ft.transitions(X, e, qk, iq, None)
        lam, alpha, bet, _ = ft.casida_from_transitions(tr, Z[iq])
        cache[iq] = (np.real(lam), alpha, bet)
    return cache


def root_in_interval(h, d, g, x):
    """Root of h - E + sum g/(E - d) in the pole interval containing x (d sorted), and its Z."""
    j = int(np.searchsorted(d, x))
    G = g.sum(); s = np.sqrt(G) + 1e-3
    lo = d[j - 1] if j > 0 else min(h, d[0]) - s
    hi = d[j] if j < len(d) else max(h, d[-1]) + s
    f = lambda y: h - y + np.sum(g / (y - d))
    for _ in range(200):
        y = 0.5 * (lo + hi)
        if y == lo or y == hi: break
        if f(y) > 0: lo = y
        else: hi = y
    y = 0.5 * (lo + hi)
    return y, 1.0 / (1.0 + np.sum(g / (y - d) ** 2))


def exact_qp(E, mu, cache, Eguess):
    cg = CasidaG0W0(thc, E, mu, 1e9, qk, nk, casida_dir=None)
    cg._cache = cache
    Enew, Zq = np.zeros((nk, nb)), np.zeros((nk, nb))
    for ik in range(nk):
        Ps, Ws = [], []
        for Ep, A, Bm, cT in cg.iter_blocks(ik):
            w = -cT / nk
            Ps.append(Ep - mu); Ws.append(np.real(A * Bm.T) * w[None, :])   # diagonal element (n, s)
        d = np.concatenate(Ps); g = np.concatenate(Ws, axis=1)
        assert g.min() > -1e-10 * g.max(), f"negative diagonal residue {g.min()}"
        o = np.argsort(d); d = d[o]; g = np.clip(g[:, o], 0.0, None)
        for n in range(nb):
            keep = g[n] > 1e-14 * g[n].sum()
            e, z = root_in_interval(hdiag[ik, n] - mu, d[keep], g[n][keep], Eguess[ik, n] - mu)
            Enew[ik, n], Zq[ik, n] = e + mu, z
    return Enew, Zq


cache0 = casida_cache(eig, mu0)
rows = []
for it in range(Eline.shape[0]):
    Ein = eig if it == 0 else Eline[it - 1]
    mu = 0.5 * (Ein[:, :nocc].max() + Ein[:, nocc:].min())
    cache = cache0 if (scheme == 'evgw0' or it == 0) else casida_cache(Ein, mu)
    Eun = Eline[it] if it == 0 else (Eline[it] - (1 - mix) * Ein) / mix          # the line's unmixed QP energies
    Ex, Zx = exact_qp(Ein, mu, cache, Eun)
    cert = Zx > 0.5
    d = np.abs(Eun - Ex)
    kv, kc = np.argmax(Eun[:, nocc - 1]), np.argmin(Eun[:, nocc])                  # the line's VBM / CBM states
    dv, dc = Eun[kv, nocc - 1] - Ex[kv, nocc - 1], Eun[kc, nocc] - Ex[kc, nocc]
    win = cert & (np.abs(Eun - mu) < 10 / 27.211386)                               # certified within 10 eV of mu
    rows.append((it + 1, dv, dc, dc - dv, Zx[kv, nocc - 1], Zx[kc, nocc], d[win].max() if win.any() else np.nan, d[cert].max()))
    print(f"{scheme} it {it+1}: line - exact (one step): VBM {dv*27211.386:+.2f} meV (Z {Zx[kv, nocc-1]:.3f}), CBM {dc*27211.386:+.2f} meV "
          f"(Z {Zx[kc, nocc]:.3f}), gap {(dc-dv)*27211.386:+.2f} meV; certified (Z > 0.5) within 10 eV of mu: {int(win.sum())} (k, band), "
          f"max {d[win].max()*27211.386 if win.any() else float('nan'):.2f} meV; all certified {int(cert.sum())}/{d.size}: max "
          f"{d[cert].max()*27211.386:.1f} meV at E - mu = {((Eun - mu)[cert][np.argmax(d[cert])])*27.211386:.1f} eV", flush=True)
if out:
    np.savez(out, rows=np.array(rows))
