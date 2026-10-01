#!/usr/bin/env python3
"""Exact RPA (Casida) solution per q in the THC basis for a CoQui run: writes casida_q{iq}.npz (lam, alpha, bet, fp) compatible
with cayley.casida_g0w0 (same construction as the user's real_axis_GW/scripts/si_pipeline.py::casida_q, T=0 occupations).
Usage: make_casida.py <chkpt.mbpt.h5> <thc.h5> <outdir>"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley.coqui_io import Checkpoint, THC
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
