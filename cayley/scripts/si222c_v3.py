#!/usr/bin/env python3
"""V3: iteration-1 (G0W0) Sigma_c on the line from the line kernels (KS poles -> Pi -> W -> Sigma) vs the EXACT Casida
Sigma_c(zeta) of the same THC (cayley.casida_g0w0), at k = ik; then the moment/upfolding chain through the line
representation vs exact moments and exact A(w). Usage: si222c_v3.py [ik=0] [theta_deg=20] [eps=1e-8] [wp_Ha=0.11]"""
import sys, os, time, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint, THC
from cayley.casida_g0w0 import CasidaG0W0
from cayley.line.line_dlr import LineBasis, BosonicLineBasis
from cayley.line.thc_gw import LineGW
from cayley.line.closure import fit_sigma_sectors, sigma_moments
from cayley import upfold_block
from cayley.spectral import sigma_from_poles, spectral_function
HA = 27.211386
ik = int(sys.argv[1]) if len(sys.argv) > 1 else 0
theta = np.deg2rad(float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
wp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.11
D = ROOT + '/data/si222c_nb58_thc1e-4'; CD = ROOT + '/data/si222c_casida'
ck = Checkpoint(D + '/si222c.mbpt.h5'); thc = THC(D + '/si222c.thc.h5')
X, Z = thc.X[0], thc.Z; nk, nb, Np = ck.nk, ck.nb, thc.Np; mu0 = ck.mu[0]; eks = ck.eig[0]
cas = CasidaG0W0(thc, eks, mu0, ck.beta, ck.qk_to_k2, nk, CD)
t0 = time.time()
ps = cas.pole_summary(ik, mu0); Dh, Dp = mu0 - ps['hole_edge'], ps['particle_edge'] - mu0
print(f"Sigma_c support about mu0 at k={ik}: hole edge {-Dh*HA:.3f} eV, particle edge {Dp*HA:.3f} eV")
bos = BosonicLineBasis(theta, lam=3.0, eps=eps, gap=0.02)
bp = LineBasis(theta, lam=3.0, eps=eps, gap=(3.0, 0.8 * Dp)); bh = LineBasis(theta, lam=3.0, eps=eps, gap=(0.8 * Dh, 3.0))
fz = np.unique(np.concatenate([bp.zeta, bh.zeta]))
print(bos); print("Sigma particle basis", bp); print("Sigma hole basis", bh, " fermionic nodes:", len(fz))
gw = LineGW(X, Z, ck.qk_to_k2, nk, mu0, theta, theta / 2, bos, fz)
e_rel, v = LineGW.poles_from_hamiltonian(np.array([np.diag(eks[k]) for k in range(nk)]).astype(complex), mu0)
gw.set_poles(e_rel, v)
wres = []
for iq in range(nk):
    w, Wl = gw.screened_interaction(iq); wres.append(w)
    print(f"  W(q={iq}) fitted: {w.shape[0]} poles [{time.time()-t0:.0f}s]", flush=True)
# Sigma per sector on the fermionic nodes (particle ray and hole ray separately)
def sigma_sector(ik, sector):
    ray = gw.ray_p if sector == '>' else gw.ray_h
    F = ray.transform_matrix(fz); out = np.zeros((len(fz), nb, nb), complex); Xk = X[ik]
    for i0 in range(0, len(ray), gw.t_chunk):
        t = ray.t[i0:i0 + gw.t_chunk]; acc = np.zeros((len(t), Np, Np), complex); Ew = bos.time_exponentials(t, sector)
        for iq in range(nk):
            wq = wres[iq] if sector == '>' else np.transpose(wres[iq], (0, 2, 1))
            acc += gw.gtilde(ck.qk_to_k2[iq, ik], t, sector) * np.einsum('tj,jpq->tpq', Ew, wq)
        acc *= (1.0 if sector == '>' else -1.0) / nk
        out += np.einsum('zt,tab->zab', F[:, i0:i0 + gw.t_chunk], (Xk.conj().T @ acc) @ Xk)
    return out
Sp = sigma_sector(ik, '>'); Sh = sigma_sector(ik, '<'); Sl = Sp + Sh
Sx = cas.sigma_z(ik, mu0 + fz)
print(f"[V3] Sigma_c(k={ik}) on {len(fz)} line nodes: line kernels vs exact Casida: max|diff| {np.abs(Sl-Sx).max():.2e} Ha (rel {np.abs(Sl-Sx).max()/np.abs(Sx).max():.1e})  [{time.time()-t0:.0f}s]")
# sector check against exact sector sums
Sxp = np.zeros_like(Sx); Sxh = np.zeros_like(Sx)
for E, A, Bm, cT in cas.iter_blocks(ik):
    w = -cT / nk; m = E > mu0
    for iz, zz in enumerate(mu0 + fz):
        Sxp[iz] += (A[:, m] * (w[m] / (zz - E[m]))[None, :]) @ Bm[m]; Sxh[iz] += (A[:, ~m] * (w[~m] / (zz - E[~m]))[None, :]) @ Bm[~m]
print(f"[V3] sectors: particle rel {np.abs(Sp-Sxp).max()/np.abs(Sxp).max():.1e}, hole rel {np.abs(Sh-Sxh).max()/np.abs(Sxh).max():.1e}")
# real-pole fits per sector -> moments -> upfold -> A(w) vs exact
w_all, g_all = fit_sigma_sectors(bp, bh, fz, Sp, Sh)
fitres = np.abs(bp.eval(g_all[len(bh.w):], fz) + bh.eval(g_all[:len(bh.w)], fz) - Sl).max() / np.abs(Sl).max()
mu_c = 0.5 * (ps['hole_edge'] + ps['particle_edge'])
C_line = sigma_moments(w_all, g_all, wp, 41, mu_c - mu0)
C_ex = cas.moments(ik, wp, 41, mu_c)
err = [np.linalg.norm(C_line[n] - C_ex[n]) / np.linalg.norm(C_ex[0]) for n in range(42)]
print(f"[V3] Sigma real-pole fit residual {fitres:.1e}; Cayley moments (line) vs exact, n=4/8/16/24/32/40: " + " ".join(f"{err[n]:.0e}" for n in (4, 8, 16, 24, 32, 40)))
H = ck.H0[0, ik] + ck.F(1)[0, ik]
om = mu_c + np.linspace(-0.45, 0.45, 241); eta = 0.01; win = np.abs(om - mu_c) < 0.19
Aex = spectral_function(H, lambda z, S=cas.sigma_z(ik, om + 1j * eta): S, om, eta, trace=True)
for K in [8, 16, 24, 32]:
    d, Wc = upfold_block(C_line, K, wp, mu_c, tol_gram=10 * max(fitres, 1e-12))
    A = spectral_function(H, sigma_from_poles(d, Wc), om, eta, trace=True)
    print(f"[V3] upfold K={K}: poles {len(d)}, rel. Tr A error |w-mu|<5 eV (eta 0.27 eV) {np.abs(A-Aex)[win].max()/Aex[win].max():.1e}  [{time.time()-t0:.0f}s]")
np.savez(ROOT + f'/results/si222c_v3_k{ik}.npz', fz=fz, Sl=Sl, Sx=Sx, C_line=C_line, C_ex=C_ex, om=om, Aex=Aex)
print("done")
