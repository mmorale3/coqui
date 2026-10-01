#!/usr/bin/env python3
"""Si kp222 nbnd58 G0W0 (CoQui v6_nb58 base): EXACT Cayley moments from the Casida pole structure of Sigma_c
-> block Toeplitz upfolding -> A(k, w) against the exact spectral function of the same Sigma_c. Usage: [ik] [wp_Ha]"""
import sys, os, time, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint, THC
from cayley.casida_g0w0 import CasidaG0W0
from cayley import bound_check, upfold_block
from cayley.upfold import upfold_sectors
from cayley.spectral import sigma_from_poles, spectral_function
HA = 27.211386
ik = int(sys.argv[1]) if len(sys.argv) > 1 else 0
wp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.11
centre = sys.argv[3] if len(sys.argv) > 3 else 'gapmid'          # 'gapmid' (Sigma_c gap midpoint ~ mu0) | 'mu1' (G0W0 Dyson mu)
out = ROOT + f'/results/si222_g0w0_k{ik}_wp{wp:.3f}_{centre}'; os.makedirs(ROOT + '/results', exist_ok=True)
t0 = time.time()
ck = Checkpoint(ROOT + '/data/si222_g0w0_nb58/si222b.mbpt.h5')
thc = THC(ROOT + '/data/si222_g0w0_nb58/si222b.thc.h5')
mu0, mu1 = ck.mu[0], ck.mu[1]
print(f"checkpoint: nk={ck.nk} nb={ck.nb} beta={ck.beta} wmax={ck.wmax} nt={len(ck.tau_f)}  mu0(KS)={mu0:.6f} mu1(G0W0)={mu1:.6f} Ha; THC Np={thc.Np}")
cas = CasidaG0W0(thc, ck.eig[0], mu0, ck.beta, ck.qk_to_k2, ck.nk, ROOT + '/data/si222_casida_nb58')
# fingerprint check against the Casida cache (built from the same THC/KS data?)
z0 = np.load(ROOT + '/data/si222_casida_nb58/casida_q0.npz'); fp = z0['fp']
X = thc.X[0]; Z = thc.Z
myfp = np.array([thc.Np, ck.nb, ck.nk, np.abs(Z[0]).sum(), np.abs(Z[0][0]).sum(), np.abs(X).sum(), np.abs(X[:, -1]).sum(), ck.eig[0].sum(), mu0])
print("fingerprint rel. mismatch:", np.max(np.abs(myfp - fp) / np.maximum(np.abs(fp), 1e-300)))
# 1. validation: Sigma(tau) from the pole structure vs the exact reference and CoQui iter1
St = cas.sigma_tau(ik, ck.tau_f)
Sc = ck.Sigma(1)[:, 0, ik]
print(f"[validate] pole-structure Sigma(tau) vs CoQui iter1 Sigma_tskij(k={ik}): max|diff| {np.abs(St - Sc).max():.2e} (rel {np.abs(St - Sc).max()/np.abs(Sc).max():.1e})  [{time.time()-t0:.0f}s]")
fx = ROOT + '/data/si222_g0w0_nb58/si222b_SigmaExact_k0.npy'
if ik == 0 and os.path.exists(fx):
    Sx = np.load(fx); print(f"[validate] vs si222b_SigmaExact_k0.npy: max|diff| {np.abs(St - Sx).max():.2e}")
# 2. pole summary -> Sigma spectral gap, choose the Cayley centre mu (midpoint of the Sigma gap) and report mu1
ps = cas.pole_summary(ik, mu0)
mu = 0.5 * (ps['hole_edge'] + ps['particle_edge']) if centre == 'gapmid' else mu1
print(f"[poles] Sigma_c gap edges (abs, Ha): hole {ps['hole_edge']:.5f}  particle {ps['particle_edge']:.5f}  -> width {(ps['particle_edge']-ps['hole_edge'])*HA:.3f} eV; "
      f"weights tr: hole {ps['weight_hole']:.4f} particle {ps['weight_particle']:.4f}; centre mu={mu:.5f} (mu1-mu = {(mu1-mu)*HA:+.3f} eV)")
# 3. exact moments (total measure and sectors)
nmax = 41
C = cas.moments(ik, wp, nmax, mu); Cl = cas.moments(ik, wp, nmax, mu, sector='<'); Cg = cas.moments(ik, wp, nmax, mu, sector='>')
print(f"[moments] bound check total {bound_check(C):.6f}  hole {bound_check(Cl):.6f}  particle {bound_check(Cg):.6f}; "
      f"hermiticity of C0 {np.abs(C[0]-C[0].conj().T).max():.1e}; |C_n|/|C_0| at n=8,16,32: " + " ".join(f"{np.linalg.norm(C[n])/np.linalg.norm(C[0]):.3f}" for n in (8, 16, 32)) + f"  [{time.time()-t0:.0f}s]")
# 4. exact spectral function of the same Sigma_c with H = H0 + F_1 (Hartree + exchange of the KS density)
H = ck.H0[0, ik] + ck.F(1)[0, ik]
om = mu + np.linspace(-0.45, 0.45, 361)
etas = [0.01, 0.004]
Aex = {}
for eta in etas:
    Sz = cas.sigma_z(ik, om + 1j * eta)
    Aex[eta] = spectral_function(H, lambda z, Sz=Sz: Sz, om, eta)
print(f"[exact] A(k,w) done [{time.time()-t0:.0f}s]")
win = np.abs(om - mu) < 0.19     # ~ +-5 eV
res = dict(om=om, mu=mu, mu0=mu0, mu1=mu1, wp=wp, C=C, Cl=Cl, Cg=Cg, H=H, pole_summary=np.array([ps['hole_edge'], ps['particle_edge']]))
for eta in etas: res[f'Aex_tr_eta{eta}'] = np.trace(Aex[eta], axis1=1, axis2=2).real; res[f'Aex_diag_eta{eta}'] = np.einsum('wii->wi', Aex[eta]).real
# 5. upfold at several orders, total measure and sector-split
Ks = [4, 8, 12, 16, 24, 32]
for mode in ['total', 'sectors']:
    print(f"== upfolding ({mode}); rel. max error of Tr A in |w-mu|<5 eV / full +-12 eV, per eta")
    for K in Ks:
        try:
            if mode == 'total': d, W, info = upfold_block(C, K, wp, mu, tol_gram=1e-13, return_info=True); extra = f"gram rank {info['rank_gram']}, held-out {info['heldout_err']:.1e}"
            else: d, W = upfold_sectors(Cl, Cg, K, wp, mu, tol_gram=1e-13); extra = ""
            Sf = sigma_from_poles(d, W)
            line = f"  K={K:2d} poles={len(d):4d} " + extra
            for eta in etas:
                A = spectral_function(H, Sf, om, eta)
                trA = np.trace(A, axis1=1, axis2=2).real; trX = res[f'Aex_tr_eta{eta}']
                e_win = np.abs(trA - trX)[win].max() / trX[win].max(); e_all = np.abs(trA - trX).max() / trX.max()
                dg = np.abs(np.einsum('wii->wi', A).real - res[f'Aex_diag_eta{eta}'])[win].max() / res[f'Aex_diag_eta{eta}'][win].max()
                line += f" | eta={eta}: trA {e_win:.1e}/{e_all:.1e} diag {dg:.1e}"
                res[f'A_{mode}_K{K}_eta{eta}'] = trA
            res[f'poles_{mode}_K{K}'] = d; res[f'W_{mode}_K{K}'] = W
            print(line, flush=True)
        except Exception as ex:
            print(f"  K={K}: failed: {ex}")
np.savez(out + '.npz', **res)
print(f"saved {out}.npz  [{time.time()-t0:.0f}s]")
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, eta in zip(axs, etas):
        ax.plot((om - mu) * HA, res[f'Aex_tr_eta{eta}'], 'k-', lw=2, label='exact (Casida poles)')
        for K, c in zip([8, 16, 32], ['C0', 'C1', 'C3']):
            if f'A_total_K{K}_eta{eta}' in res: ax.plot((om - mu) * HA, res[f'A_total_K{K}_eta{eta}'], c + '--', lw=1, label=f'Cayley total K={K}')
            if f'A_sectors_K{K}_eta{eta}' in res: ax.plot((om - mu) * HA, res[f'A_sectors_K{K}_eta{eta}'], c + ':', lw=1, label=f'sectors K={K}')
        ax.set_ylabel(f'Tr A(k,w)  eta={eta*HA:.2f} eV'); ax.legend(fontsize=8); ax.axvspan(-5, 5, color='0.9', zorder=0)
    axs[1].set_xlabel('w - mu (eV)'); axs[0].set_title(f'Si 2x2x2 nbnd58 G0W0, k={ik}, wp={wp*HA:.2f} eV, centre {centre}, exact moments')
    fig.tight_layout(); fig.savefig(out + '.png', dpi=130); print('saved', out + '.png')
except Exception as ex:
    print('plot skipped:', ex)
