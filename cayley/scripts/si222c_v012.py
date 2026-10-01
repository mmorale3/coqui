#!/usr/bin/env python3
"""V0-V2 for the line GW kernels on Si 2x2x2 nbnd58 (THC thresh 1e-4 = si222c):
 V0 Hartree+exchange from the KS density vs CoQui F_1;  V1 Pi(q, zeta) on the line via the time ray vs the Casida transition
 sum;  V2 W(q, zeta) vs Casida W_dyn and the quality of the symmetric real-pole refit (incl. the particle part on the axis).
Usage: si222c_v012.py [theta_deg=20] [eps=1e-8]"""
import sys, os, time, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint, THC
from cayley.line.line_dlr import BosonicLineBasis
from cayley.line.thc_gw import LineGW
HA = 27.211386
theta = np.deg2rad(float(sys.argv[1]) if len(sys.argv) > 1 else 20.0)
eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
D = ROOT + '/data/si222c_nb58_thc1e-4'; CD = ROOT + '/data/si222c_casida'
ck = Checkpoint(D + '/si222c.mbpt.h5'); thc = THC(D + '/si222c.thc.h5')
X, Z = thc.X[0], thc.Z; nk, nb, Np = ck.nk, ck.nb, thc.Np
mu0 = ck.mu[0]; eks = ck.eig[0]
print(f"si222c: nk={nk} nb={nb} Np={Np} beta={ck.beta} mu0={mu0:.6f}; KS gap {(eks[eks>mu0].min()-eks[eks<mu0].max())*HA:.3f} eV")
t0 = time.time()
# ---- V0: static part from the KS density matrix (diag occupations in the KS basis) vs CoQui F_1 (= V_H + Sigma_x of Dm_0)
Dm0 = np.array([np.diag((eks[k] < mu0).astype(float)) for k in range(nk)]).astype(complex)
bos = BosonicLineBasis(theta, lam=3.0, eps=eps, gap=0.02)          # bosonic poles up to 3 Ha (~80 eV), gap 0.02 Ha (lowest excitation 0.73 eV = 0.027 Ha)
print(bos)
fz = np.exp(np.linspace(np.log(1e-4), np.log(20.0), 60)); fz = np.concatenate([fz * np.exp(1j * theta), fz * np.exp(1j * (np.pi - theta))])
gw = LineGW(X, Z, ck.qk_to_k2, nk, mu0, theta, theta / 2, bos, fz)
F0 = gw.hartree_exchange(Dm0)
F1 = ck.F(1)[0]
print(f"[V0] F[Dm_KS] vs CoQui F_1: max|diff| {np.abs(F0 - F1).max():.2e} Ha (rel {np.abs(F0-F1).max()/np.abs(F1).max():.1e}); max|F1| {np.abs(F1).max():.3f}  [{time.time()-t0:.0f}s]")
# ---- V1: Pi0 on the line (KS poles) vs the Casida transition sum
e_rel, v = LineGW.poles_from_hamiltonian(np.array([np.diag(eks[k]) for k in range(nk)]).astype(complex), mu0)
gw.set_poles(e_rel, v)
print(f"time rays: {len(gw.ray_p)} nodes, theta_t = {np.degrees(gw.theta_t):.1f} deg, s_max = {gw.ray_p.s.max():.1f} Ha^-1")
occ = e_rel < 0
def pi_casida(iq, zeta):
    cols, E, sg = [], [], []
    for ik in range(nk):
        ikmq = ck.qk_to_k2[iq, ik]
        for n in range(nb):
            for m in range(nb):
                if occ[ik, n] == occ[ikmq, m]: continue
                cols.append(X[ik][:, n] * np.conj(X[ikmq][:, m])); E.append(e_rel[ikmq, m] - e_rel[ik, n]); sg.append(1.0 if occ[ik, n] else -1.0)
    S = np.sqrt(2.0 / nk) * np.array(cols).T; E = np.array(E); sg = np.array(sg)
    return np.array([(S * (sg / (z - E))[None, :]) @ S.conj().T for z in zeta])
ztest = bos.zeta[::max(1, len(bos.zeta) // 12)]
for iq in [0, 3]:
    Pl = gw.polarization(iq, ztest); Pc = pi_casida(iq, ztest)
    print(f"[V1] q={iq}: Pi(line, time-ray) vs Casida transition sum at {len(ztest)} nodes: max|diff| {np.abs(Pl-Pc).max():.2e} (rel {np.abs(Pl-Pc).max()/np.abs(Pc).max():.1e}); "
          f"hermiticity Pi(zeta)-Pi(conj zeta)^dag n/a; |Pi| max {np.abs(Pc).max():.3e}  [{time.time()-t0:.0f}s]")
# ---- V2: W on the line vs Casida W_dyn; symmetric refit
for iq in [0, 3]:
    z = np.load(f"{CD}/casida_q{iq}.npz"); lam, alpha, bet = z['lam'], z['alpha'], z['bet']
    Pi = gw.polarization(iq, bos.zeta)
    wres, Wl = gw.screened_interaction(iq, Pi)
    Wc = np.array([(alpha / (zz - lam)[None, :]) @ bet for zz in bos.zeta])
    fit = bos.eval(wres, bos.zeta)
    zi = 1j * np.exp(np.linspace(np.log(1e-3), np.log(10.0), 20))
    Wi = bos.eval(wres, zi); Wci = np.array([(alpha / (zz - lam)[None, :]) @ bet for zz in zi])
    Wp = bos.eval(wres, zi, '>'); Wcp = np.array([(alpha[:, lam > 0] / (zz - lam[lam > 0])[None, :]) @ bet[lam > 0] for zz in zi])
    print(f"[V2] q={iq}: W(line) vs Casida W_dyn at nodes: rel {np.abs(Wl-Wc).max()/np.abs(Wc).max():.1e}; refit residual at nodes {np.abs(fit-Wl).max()/np.abs(Wl).max():.1e}; "
          f"W on imag axis vs Casida {np.abs(Wi-Wci).max()/np.abs(Wci).max():.1e}; W^> (particle part) on imag axis vs Casida {np.abs(Wp-Wcp).max()/np.abs(Wcp).max():.1e}; "
          f"odd-symmetry of W(line): {np.abs(Wl - np.transpose(Wl,(0,2,1)).conj()).max()/np.abs(Wl).max():.1e} (hermiticity, info)  [{time.time()-t0:.0f}s]")
print("done")
