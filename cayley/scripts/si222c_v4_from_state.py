#!/usr/bin/env python3
"""V4 from a saved line-scGW state: Sigma_line(z) at CoQui's Matsubara points vs the converged Matsubara scGW on the same THC;
static part and density matrix comparison; spectral-function preview at Gamma from the line Sigma via Cayley moments.
Usage: si222c_v4_from_state.py [state.npz] [K=24]"""
import sys, os, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint
from cayley.line.line_dlr import LineBasis
from cayley.line.closure import fit_sigma_sectors, sigma_moments
from cayley.dlr import DLRg
from cayley import upfold_block
from cayley.spectral import sigma_from_poles, spectral_function
HA = 27.211386
state = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/results/si222c_line_scgw_th20_K24_state.npz'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 24
z = np.load(state, allow_pickle=True)
fz, Sp, Sh, F_line, mu_l = z['fz'], z['Sig_p'], z['Sig_h'], z['F'], float(z['mu'])
cm = Checkpoint(ROOT + '/data/si222c_scgw/si222c.mbpt.h5'); it = cm.final_iter; mu_m = cm.mu[it]
nk, nb = cm.nk, cm.nb
print(f"line state: {int(z['niter'])} iterations, mu {mu_l:.6f}; CoQui Matsubara scGW iter {it}, mu {mu_m:.6f} (diff {(mu_l-mu_m)*HA:+.3f} eV)")
th = np.deg2rad(20); eps = 1e-10; lam = 6.0
bp = LineBasis(th, lam=lam, eps=eps, gap=(lam, 0.02), tmax=60.0); bh = LineBasis(th, lam=lam, eps=eps, gap=(0.02, lam), tmax=60.0)
# CoQui Sigma(i w_n) from its tau data (DLR refit), about mu_m
St = cm.Sigma(it)[:, 0]; Fd = DLRg(cm.beta, cm.wmax, 1e-12, 'fermi')
n = cm.iwn_f; wn = (2 * n + 1) * np.pi / cm.beta if n.min() >= 0 else n * np.pi / cm.beta
sel = wn > 0; wn = wn[sel]
F_m = cm.F(it)[0]; Dm_m = cm.Dm(it)[0]
print(f"static part F: max|F_line - F_CoQui| {np.abs(F_line - F_m).max():.2e} Ha (max|F| {np.abs(F_m).max():.3f});  Dm: CoQui tr {np.einsum('kii->', Dm_m).real/nk*2:.4f} electrons")
worst = 0.0
for ik in range(nk):
    c = Fd.coefs_from_tau(St[:, ik], tau=cm.tau_f); Sm = Fd.eval_iw(c, wn)                     # (nw, nb, nb), frequencies about mu_m
    w, g = fit_sigma_sectors(bp, bh, fz, Sp[ik], Sh[ik])
    zr = (mu_m + 1j * wn) - mu_l                                                                  # same absolute points, about mu_l
    Sl = np.einsum('zl,lij->zij', 1.0 / (zr[:, None] - w[None, :]), g)
    d = np.abs(Sl - Sm); rel = d.max() / np.abs(Sm).max(); worst = max(worst, rel)
    i0 = np.argmax(d.max(axis=(1, 2)))
    print(f"  k={ik}: max|Sigma_line - Sigma_CoQui| over {len(wn)} Matsubara points: {d.max():.2e} Ha (rel {rel:.1e}); worst at w_n = {wn[i0]:.3f} Ha; "
          f"|Sigma| at lowest w_n: line {np.abs(Sl[0]).max():.4f} CoQui {np.abs(Sm[0]).max():.4f}; diag diff at lowest w_n {np.abs(np.diag(Sl[0]-Sm[0])).max():.2e}")
print(f"V4 summary: worst relative deviation {worst:.1e}")
# spectral preview at Gamma from the line Sigma (moments about mu_l)
ik = 0; w, g = fit_sigma_sectors(bp, bh, fz, Sp[ik], Sh[ik]); wp = 0.11
C = sigma_moments(w, g, wp, K + 2, 0.0)
d, W = upfold_block(C, K, wp, 0.0, tol_gram=1e-10)
H = cm.H0[0, ik] + F_line[ik] - mu_l * np.eye(nb)
om = np.linspace(-0.45, 0.45, 601)
for eta in [0.004, 0.01]:
    A = spectral_function(H, sigma_from_poles(d, W), om, eta, trace=True)
    np.save(ROOT + f'/results/si222c_line_A_gamma_eta{eta}.npy', np.vstack([om, A]))
    pk = om[np.r_[False, (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]), False] & (A > 0.05 * A.max())]
    print(f"A(Gamma, w) from the line scGW (K={K}, eta {eta*HA:.2f} eV): peaks (eV rel. mu) " + " ".join(f"{p*HA:+.2f}" for p in pk[:12]))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for eta, c in [(0.01, 'k'), (0.004, 'C3')]:
        o, A = np.load(ROOT + f'/results/si222c_line_A_gamma_eta{eta}.npy'); ax.plot(o * HA, A, c, lw=1.2, label=f'eta={eta*HA:.2f} eV')
    ax.set_xlabel('w - mu (eV)'); ax.set_ylabel('Tr A(Gamma, w)'); ax.set_title(f'Si 2x2x2 line scGW ({int(z["niter"])} it, theta 20, K {K}): spectral function at Gamma'); ax.legend(); ax.axvspan(-5, 5, color='0.92', zorder=0)
    fig.tight_layout(); fig.savefig(ROOT + '/results/si222c_line_A_gamma.png', dpi=130); print('saved results/si222c_line_A_gamma.png')
except Exception as ex: print('plot skipped', ex)
