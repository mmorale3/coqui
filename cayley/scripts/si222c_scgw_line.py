#!/usr/bin/env python3
"""Self-consistent GW on the tilted line for Si 2x2x2 nbnd58 (si222c THC), starting from the KS Hamiltonian.
Per iteration: W (8 q), Sigma (8 k, both sectors), moments -> upfold -> Lehmann G, mu, F. Saves the history and the final
Sigma on the line; if a converged CoQui Matsubara checkpoint for the same THC exists (data/si222c_scgw/si222c.mbpt.h5), does
V4: Sigma_line(i w_n) vs CoQui Sigma(i w_n). Usage: si222c_scgw_line.py [niter=10] [theta_deg=20] [eps=1e-8] [K=16] [mixing=0.5]"""
import sys, os, time, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint, THC
from cayley.line.driver import LineSCGW
from cayley.line.closure import fit_sigma_sectors
from cayley.dlr import DLRg
HA = 27.211386
niter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
theta = np.deg2rad(float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
K = int(sys.argv[4]) if len(sys.argv) > 4 else 16
mixing = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
D = ROOT + '/data/si222c_nb58_thc1e-4'
ck = Checkpoint(D + '/si222c.mbpt.h5'); thc = THC(D + '/si222c.thc.h5')
X, Z = thc.X[0], thc.Z; nk, nb = ck.nk, ck.nb; mu0 = ck.mu[0]; eks = ck.eig[0]; H0 = ck.H0[0]
nelec = 8.0
sc = LineSCGW(X, Z, ck.qk_to_k2, nk, nelec, H0, mu0, theta=theta, eps=eps, wp=0.11, K=K, mixing=mixing, k_weight=ck.k_weight)
sc.start_from_hamiltonian(np.array([np.diag(eks[k]) for k in range(nk)]).astype(complex))
F1 = ck.F(1)[0]
print(f"start: F[Dm_KS] vs CoQui F_1 max|diff| {np.abs(sc.F - F1).max():.2e}; mu0 {mu0:.6f}", flush=True)
out = ROOT + f'/results/si222c_line_scgw_th{np.degrees(theta):.0f}_K{K}'
for it in range(niter):
    rec = sc.iterate()
    np.savez(out + '.npz', history=np.array(sc.history, dtype=object), fz=sc.fz, mu=sc.mu,
             Sig=np.array([sc.sigma_total(k) for k in range(nk)]), F=sc.F)
# V4 against the converged Matsubara scGW on the same THC, if available
fn = ROOT + '/data/si222c_scgw/si222c.mbpt.h5'
if os.path.exists(fn):
    cm = Checkpoint(fn); it = cm.final_iter
    St = cm.Sigma(it)[:, 0]; mu_m = cm.mu[it]
    Fd = DLRg(cm.beta, cm.wmax, 1e-12, 'fermi')
    wn = (2 * cm.iwn_f + 1) * np.pi / cm.beta if cm.iwn_f.min() >= 0 else cm.iwn_f * np.pi / cm.beta
    wn = wn[wn > 0]
    print(f"V4 oracle: CoQui Matsubara scGW iter {it}, mu {mu_m:.6f} Ha (line mu {sc.mu:.6f}, diff {(sc.mu-mu_m)*HA:+.4f} eV)")
    for ik in [0, 1]:
        c = Fd.coefs_from_tau(St[:, ik], tau=cm.tau_f)
        Sm = Fd.eval_iw(c, wn)                                            # CoQui Sigma(i w_n), energies about mu_m
        w, g = fit_sigma_sectors(sc.bp, sc.bh, sc.fz, sc.Sig_prev[ik][0], sc.Sig_prev[ik][1])
        zi = 1j * wn + (mu_m - sc.mu)                                     # same absolute frequency, expressed about the line centre
        Sl = np.einsum('zl,lij->zij', 1.0 / (zi[:, None] - w[None, :]), g)
        print(f"  k={ik}: Sigma_line(i w_n) vs CoQui: max|diff| {np.abs(Sl-Sm).max():.2e} (rel {np.abs(Sl-Sm).max()/np.abs(Sm).max():.1e}) over {len(wn)} Matsubara points; "
              f"low-frequency |Sigma| {np.abs(Sm[0]).max():.4f}")
    print(f"gap (line poles) {sc.history[-1]['gap_eV']:.4f} eV")
print("done")
