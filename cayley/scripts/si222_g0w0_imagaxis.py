#!/usr/bin/env python3
"""Si kp222 nbnd58 G0W0: Cayley moments from the IMAGINARY-AXIS data only (CoQui Sigma_tskij on its tau nodes), via a DLR
refit and the gap-strip (truncated Laplace) estimator, compared with the EXACT moments from the Casida pole structure
(results/si222_g0w0_k{ik}_wp{wp}.npz from si222_g0w0_casida.py). This quantifies the imaginary-axis route on real data.
Usage: [ik] [wp_Ha]"""
import sys, os, time, numpy as np
ROOT = '/Users/mmorales/Projects/Cayley_real_axis_scGW'
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint
from cayley.dlr import DLRg
from cayley.strip import sigma_strip_dlr, sigma_dlr_direct, circle_moments, max_rho
from cayley import upfold_block
from cayley.spectral import sigma_from_poles, spectral_function
HA = 27.211386
ik = int(sys.argv[1]) if len(sys.argv) > 1 else 0
wp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.11
fn = ROOT + f'/results/si222_g0w0_k{ik}_wp{wp:.3f}_gapmid.npz'
if not os.path.exists(fn): fn = ROOT + f'/results/si222_g0w0_k{ik}_wp{wp:.3f}.npz'
ref = np.load(fn)
mu = float(ref['mu']); C_ex = ref['C']
ck = Checkpoint(ROOT + '/data/si222_g0w0_nb58/si222b.mbpt.h5')
# weight-aware Sigma_c gap edges about the reference centre (the exact pole structure is available here)
from cayley.coqui_io import THC
from cayley.casida_g0w0 import CasidaG0W0
thc = THC(ROOT + '/data/si222_g0w0_nb58/si222b.thc.h5')
cas = CasidaG0W0(thc, ck.eig[0], ck.mu[0], ck.beta, ck.qk_to_k2, ck.nk, ROOT + '/data/si222_casida_nb58')
ps = cas.pole_summary(ik, mu); hole_edge, part_edge = ps['hole_edge'], ps['particle_edge']
print(f"reference {os.path.basename(fn)}: centre mu={mu:.5f}; Sigma_c support edges {hole_edge:.5f} / {part_edge:.5f} Ha")
S_tau = ck.Sigma(1)[:, 0, ik]                                     # (nt, nb, nb) on CoQui tau nodes
t0 = time.time()
# DLR refit of CoQui's tau-grid data (own nodes; poles measured from mu so that the strip formulas apply)
F = DLRg(ck.beta, ck.wmax, 1e-13, 'fermi')
# shift: Sigma(tau) data correspond to energies measured from the checkpoint's KS mu0 (the G0W0 Sigma was built with G0 at mu0);
# the kernel K(tau, w) with w measured from mu0. For the Cayley centre mu (mid-gap of Sigma) we need poles relative to mu:
mu0 = ck.mu[0]
c = F.coefs_from_tau(S_tau, tau=ck.tau_f)                        # (r, nb, nb)
fit = F.eval_tau(c, ck.tau_f)
delta_fit = np.abs(fit - S_tau).max() / np.abs(S_tau).max()
print(f"DLR refit: beta={ck.beta} wmax={ck.wmax} rank {F.r}; relative residual on CoQui nodes {delta_fit:.1e}  [{time.time()-t0:.0f}s]")
w_rel = F.w - (mu - mu0)                                          # DLR poles relative to the Cayley centre
D_minus, D_plus = mu - hole_edge, part_edge - mu
print(f"Sigma gap about the centre: D- = {D_minus*HA:.3f} eV, D+ = {D_plus*HA:.3f} eV;  wp = {wp*HA:.2f} eV")
# the exact C0 (1/z tail) is the only input besides the data: estimate it from the data too (sum of DLR coefficients = -Sigma tail)
C0_data = -c.sum(0)                                               # Sigma(z) -> sum_l c_l/(z - w_l) * (-1)?  check sign against exact
sgn = 1.0 if np.linalg.norm(C0_data - C_ex[0]) < np.linalg.norm(C0_data + C_ex[0]) else -1.0
C0_data *= sgn
print(f"C0 from DLR coefficients vs exact: rel diff {np.linalg.norm(C0_data - C_ex[0])/np.linalg.norm(C_ex[0]):.1e} (sign {sgn:+.0f})")
ns = [2, 4, 6, 8, 12, 16, 24, 32]
nmax = max(ns)
for delta in [1e-9, 1e-11]:
    print(f"== assumed data error delta = {delta:g}: rel. moment error |C_n - C_n^exact|/|C_0| (circle radius rho, strip fraction rfrac)")
    for rfrac in [0.33, 0.5, 0.7]:
        rho = max_rho(wp, min(D_minus, D_plus), rfrac)
        sf = lambda zr: sgn * sigma_strip_dlr(zr, c, w_rel, ck.beta, D_minus, D_plus, delta)
        Cs = circle_moments(sf, C_ex[0], wp, rho, nmax)
        err = [np.linalg.norm(Cs[n] - C_ex[n]) / np.linalg.norm(C_ex[0]) for n in range(nmax + 1)]
        print(f"  strip-Laplace rfrac={rfrac:.2f} rho={rho:.3f}: " + " ".join(f"n={n}:{err[n]:.0e}" for n in ns))
# direct (untruncated) evaluation of the DLR interpolant off the axis, for comparison
for rfrac in [0.5, 0.7]:
    rho = max_rho(wp, min(D_minus, D_plus), rfrac)
    Cd = circle_moments(lambda zr: sgn * sigma_dlr_direct(zr, c, w_rel), C_ex[0], wp, rho, nmax)
    err = [np.linalg.norm(Cd[n] - C_ex[n]) / np.linalg.norm(C_ex[0]) for n in range(nmax + 1)]
    print(f"  direct DLR pole sum  rho={rho:.3f}: " + " ".join(f"n={n}:{err[n]:.0e}" for n in ns))
# spectra from the best imaginary-axis moments at low order, vs exact
H = ref['H']; om = ref['om']; win = np.abs(om - mu) < 0.19
rho = max_rho(wp, min(D_minus, D_plus), 0.5)
Cs = circle_moments(lambda zr: sgn * sigma_strip_dlr(zr, c, w_rel, ck.beta, D_minus, D_plus, 1e-9), C_ex[0], wp, rho, 13)
for K in [2, 4, 6, 8, 12]:
    for tol in [1e-6, 1e-4]:
        try:
            d, W = upfold_block(Cs, K, wp, mu, tol_gram=tol)
            for eta in [0.01]:
                A = spectral_function(H, sigma_from_poles(d, W), om, eta, trace=True); X = ref[f'Aex_tr_eta{eta}']
                print(f"  imag-axis moments K={K:2d} tol_gram={tol:g}: poles {len(d):3d}, rel. Tr A error |w-mu|<5 eV: {np.abs(A-X)[win].max()/X[win].max():.2e}")
        except Exception as ex:
            print(f"  K={K} tol={tol}: {ex}")
print(f"done [{time.time()-t0:.0f}s]")
