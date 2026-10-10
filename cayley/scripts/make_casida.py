#!/usr/bin/env python3
"""Exact RPA (Casida) solution per q in the THC basis for a CoQui run: writes casida_q{iq}.npz (lam, alpha, bet, fp) compatible
with cayley.casida_g0w0 (same construction as the user's real_axis_GW/scripts/si_pipeline.py::casida_q, T=0 occupations).
Usage: make_casida.py <chkpt.mbpt.h5> <thc.h5> <outdir>
       make_casida.py <chkpt.mbpt.h5 | system.h5> <thc.h5> <outdir> --beta B [--mu0 M] [--thermal-tol 1e-8] [--deg-tol 1e-8]
--beta (S8b, notes section 11): finite-T Casida (cayley.finite_t): every pair (n at k, m at k-q) with E = e_m - e_n != 0
  (|E| >= deg_tol; degenerate pairs are the Matsubara nu_0 term, not part of the analytic W), sg = sign(E), columns scaled by
  sqrt|f(e_n) - f(e_m)|; mu0 from the KS "auto" rule (gap midpoint if beta * half-gap > ln(1/thermal_tol), else N(mu0) = N_el)
  unless --mu0. Checked against the Dyson W of the finite-T transition sum at i nu_n (n = 1, 2, 5) and at generic points.
  A system.h5 (GW_line test dump: system/eigval, qk_to_k2, mu0, nelec) can replace the checkpoint."""
import sys, os, time, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley.coqui_io import Checkpoint, THC
if '--beta' in sys.argv:
    from cayley import finite_t as ft
    opt = lambda nm, d: float(sys.argv[sys.argv.index(nm) + 1]) if nm in sys.argv else d
    beta, ttol, dtol = opt('--beta', None), opt('--thermal-tol', 1e-8), opt('--deg-tol', 1e-8)
    thc = THC(sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
    with h5py.File(sys.argv[1], 'r') as f:
        if 'system' in f:
            s = f['system']; eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
        else:
            ck = Checkpoint(sys.argv[1]); eig = ck.eig[0]; qk = ck.qk_to_k2; nelec = None
    if '--mu0' in sys.argv: mu0, rule = opt('--mu0', None), 'given'
    elif nelec is None: mu0, rule = ck.mu[0], 'checkpoint'
    else: mu0, rule = ft.mu0_auto(eig, nelec, beta, ttol)
    X = thc.X[0]; Z = thc.Z; nk, nb, Np = eig.shape[0], eig.shape[1], thc.Np; e = eig - mu0
    print(f"finite-T Casida: nk={nk} nb={nb} Np={Np} beta={beta} mu0={mu0:.10f} ({rule})")
    for iq in range(nk):
        t0 = time.time()
        tr = ft.transitions(X, e, qk, iq, beta, deg_tol=dtol)
        lam, alpha, bet, info = ft.casida_from_transitions(tr, Z[iq])
        zs = np.concatenate([1j * 2 * np.pi * np.array([1, 2, 5]) / beta, [0.1j, 1j, 10j, 0.05 + 0.3j]])
        Wd = ft.dyson_w(Z[iq], ft.pi_transition(tr, zs)); Wc = ft.casida_w(lam, alpha, bet, zs)
        err = np.max(np.abs(Wd - Wc)) / np.max(np.abs(Wd))
        fp = np.array([Np, nb, nk, np.abs(Z[iq]).sum(), np.abs(Z[iq][0]).sum(), np.abs(X).sum(), np.abs(X[:, -1]).sum(), eig.sum(), mu0])
        np.savez(f"{out}/casida_q{iq}.npz", fp=fp, lam=lam, alpha=alpha, bet=bet, imlam=info['imlam'], condR=info['condR'],
                 iwrel=err, beta=beta, mu0=mu0, ndeg=len(tr['wd']))
        print(f"q={iq}: Nt={info['Nt']} (degenerate {len(tr['wd'])}) |Im lam|max={info['imlam']:.1e} min|lam|={np.min(np.abs(lam)):.2e} "
              f"Casida-vs-Dyson {err:.1e}  [{time.time()-t0:.0f}s]", flush=True)
    sys.exit(0)
ck = Checkpoint(sys.argv[1]); thc = THC(sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
X = thc.X[0]; Z = thc.Z; eig = ck.eig[0]; mu0 = ck.mu[0]; nk, nb, Np = ck.nk, ck.nb, thc.Np
e = eig - mu0; occ = e < 0
print(f"nk={nk} nb={nb} Np={Np} mu0={mu0:.6f}")
for iq in range(nk):
    t0 = time.time()
    cols, E, sg = [], [], []
    for ik in range(nk):
        ikmq = ck.qk_to_k2[iq, ik]
        for n in range(nb):
            for m in range(nb):
                if occ[ik, n] == occ[ikmq, m]: continue
                cols.append(X[ik][:, n] * np.conj(X[ikmq][:, m])); E.append(e[ikmq, m] - e[ik, n]); sg.append(1.0 if occ[ik, n] else -1.0)
    S = np.sqrt(2.0 / nk) * np.array(cols).T; E = np.array(E); sg = np.array(sg)
    ZS = Z[iq] @ S; K = S.conj().T @ ZS
    H = np.diag(E) + sg[:, None] * K
    lam, R = np.linalg.eig(H); Rinv = np.linalg.inv(R)
    alpha = ZS @ R; bet = Rinv @ (sg[:, None] * ZS.conj().T)
    err = 0.0
    for nu in (0.0, 0.1, 1.0, 10.0):
        Pi = (S * (sg / (1j * nu - E))[None, :]) @ S.conj().T
        Wd = np.linalg.solve(np.eye(Np) - Z[iq] @ Pi, Z[iq]) - Z[iq]
        Wc = (alpha / (1j * nu - lam)[None, :]) @ bet
        err = max(err, np.max(np.abs(Wd - Wc)) / np.max(np.abs(Wd)))
    fp = np.array([Np, nb, nk, np.abs(Z[iq]).sum(), np.abs(Z[iq][0]).sum(), np.abs(X).sum(), np.abs(X[:, -1]).sum(), eig.sum(), mu0])
    np.savez(f"{out}/casida_q{iq}.npz", fp=fp, lam=lam.real, alpha=alpha, bet=bet, imlam=np.max(np.abs(lam.imag)), condR=np.linalg.cond(R), iwrel=err)
    print(f"q={iq}: Nt={len(E)} |Im lam|max={np.max(np.abs(lam.imag)):.1e} min|lam|={np.min(np.abs(lam)):.4f} Ha  Casida-vs-Dyson(iw) {err:.1e}  [{time.time()-t0:.0f}s]", flush=True)
