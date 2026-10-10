#!/usr/bin/env python3
"""Finite-temperature references for the C++ [gw_line][finiteT] tests (plan S8b T1-T5, notes section 11) and their
verification against the python line prototype. All reference values are ORACLES independent of the line machinery
(cayley/finite_t.py): the finite-T transition sum (Eq. fT_pi), the finite-T Casida W, Eq. fT_sigma from the exact Casida
poles, the Matsubara nu_0 term; the masks / windows / scales are the design rules of notes section 11.

Inputs: tests/unit_test_files/gw_line/<fx>_thc/{thc.eri.h5, system.h5} (C++ dumps: lih222 by test_gw_line_scf
"[.gw_line_dump]", lih223 by "[.gw_line_dump_lih223]"; KS poles, nelec, qk_to_k2; qminus = qk_to_k2[:, 0] (k index 0 = Gamma)).
Output: tests/unit_test_files/gw_line/<fx>_finiteT_ref.h5. Complex arrays are stored as <name>_re / <name>_im (float64).
All energies in Ha; "mu-relative" = measured from the group's mu0. Test points are mu-relative complex numbers.

H5 layout
  attrs (root): fixture, thc_dir, generator, cayley_head, date, theta_deg (20), theta_t_deg (10), rho (= sin(theta-theta_t)/
        sin(theta_t) = 1), thermal_tol (1e-8), thermal_floor (c_zeta = 30), wp_floor (15; S8b.1 E3 on real data), deg_tol (1e-8: |E| below which a pair
        is degenerate), nk, nb, Np, nelec, nprobe, betas (list), sigma_k (list)
  /eig (nk, nb)                    KS eigenvalues (absolute)
  /qk_to_k2 (nk, nk), /qminus (nk) index maps (qminus[q] = -q)
  /probe_re, /probe_im (Np, nprobe) fixed orthonormal probe vectors U: the *_probe datasets are U^H M U (nprobe x nprobe)
  /bos_zeta_re, _im (nzb)           bosonic test points: 48 log points per ray, |zeta| in [1e-3, 40] Ha, rays theta and pi-theta
  /ferm_zeta_re, _im (nzf)          fermionic test points: 20 log points per ray, |zeta| in [1e-3, 60] Ha
  /beta_<int(beta)>/  (beta = 50, 200, 10000)
     attrs: beta, mu0 (absolute), mu_rule ("number" | "gap": KS "auto" rule), N_mu0 (N at mu0), dN_dmu, E_T = ln(1/thermal_tol)/beta,
            zeta_T = c_zeta/(rho beta), S_T = beta/sin(theta_t), thermal_active (1 iff a KS pole lies within E_T of mu0),
            verification numbers (see below), nu_n_max_index
     f (nk, nb)                     Fermi occupations f(e - mu0) (exact)
     window (nk, nb) int8           |e - mu0| <= E_T ; window_counts (nk)
     D_exact_diag (nk, nb)          density matrix per spin (KS band basis, diagonal) = f
     D_lists_diag (nk, nb)          the thermal hole list's density (Eq. fT_sectors: window f, far holes 1, far particles 0)
     bos_mask (nzb) int8            1 = rho beta |zeta| >= c_zeta (unmasked; all 1 when thermal_active = 0)
     Pi_probe (nq, nzb, np, np)     U^H Pi(q, zeta) U, transition sum (dynamic convention; exact at every node, gate on bos_mask)
     Pi_full (nfull, Np, Np), Pi_full_q (nfull), Pi_full_node (nfull)   full Pi at two unmasked nodes (debugging; beta = 50 only)
     ndeg (nq), dPi_nu0_probe (nq, np, np)   degenerate pairs and dPi = Pi^Mats(q, i nu_0) - Pi^an(q, 0)
     W_zeta_probe (nq, nzb, np, np) finite-T Casida W_dyn(q, zeta) (W - Z) at the bosonic test points
     nu_n (nnu) int, W_inu_probe (nq, nnu, np, np)   W_dyn(q, i nu_n), nu_n = 2 pi n/beta, n >= 1 (n list log-spaced to 40 Ha)
     W0_an_probe, W0_mats_probe (nq, np, np)          analytic W(q, 0) and Matsubara W(q, i nu_0) (= Dyson with Pi^an(0) + dPi)
     Sigma_zeta (nks, nzf, nb, nb)  Sigma_c(k, zeta) of Eq. fT_sigma at the fermionic test points (k = sigma_k), ferm_mask (nzf) int8
     w_n (nw) int, iw_mask (nw) int8, Sigma_iw (nks, nw, nb, nb)   Sigma_c(k, i w_n), w_n = (2n+1) pi/beta, mask |w_n| >= zeta_T
     nu0_term (nks, nw, nb, nb)     predicted Sigma^Mats(k, i w_n) - Sigma^an(k, i w_n) (test T3: compare CoQui gw_t minus this)
  Verification attrs per beta group (line prototype vs these oracles, measured before writing; relative max-norm errors):
     ver_tau_route_pi            tau-quadrature Matsubara Pi vs the transition sum at i nu_n, n = 1, 2, 5 (2 q)
     ver_tau_route_nu0           same at n = 0 after adding dPi
     ver_casida_vs_dyson         Casida W vs Dyson[transition-sum Pi] at i nu_n and generic points (all q)
     ver_casida_vs_tau           Casida W vs Dyson[tau-route Pi] at i nu_n, n = 1, 2, 5 (2 q)
     ver_sigma_vs_cT             Eq. fT_sigma (n, f weights) vs CasidaG0W0's c_T thermal factor form (k = sigma_k[0])
     The line checks run with thermal_tol = ver_thermal_tol = 1e-12 (attr): with the default 1e-8 the far poles just outside E_T
     carry weight 1 instead of 1 - f ~ 1e-8 (Eq. fT_sectors), which alone gives 1.2e-9 on lih223 at beta 200 (> 10 eps);
     ver_line_pi_tol_default is that number (thermal_tol 1e-8, report only).
     ver_line_pi                 line Pi (thermal lists, guarded GL rays) vs transition sum, unmasked test nodes, all q    [gate 1e-9]
     ver_line_w_nodes            Dyson[line Pi] vs Casida W at the unmasked test nodes, all q                              [gate 1e-8]
     ver_line_sigma_fullW        line Sigma_c (thermal lists + Bose-weighted W(t)) with W residues fitted on ALL nodes of the exact
                                 Casida W, vs Eq. fT_sigma at unmasked fermionic test points and i w_n >= zeta_T (k = sigma_k[0]) [gate 1e-8]
  REDESIGNED W STEP (S8b.1b; notes section 11.5 sec:fT_W; groups with thermal_active = 1 only, attr wstep_active):
     The data set D, the tau grid and the D-selected basis are the PYTHON prototype's (LineGW.bos_data / tau_grid /
     BosonicLineBasis.from_data with the W-step parameters in the root attrs wstep_*); D's line nodes are the unmasked nodes of the
     gapless bosonic line basis BosonicLineBasis(theta, lam_b_verify, eps 1e-10, gap 0). A C++ test evaluates its OWN Pi / W at
     the points D_zeta (transform columns at any wedge point; nu_0 from its tau leg) and compares with the oracle datasets below;
     to compare residues it must use nu_b (pivoted-QR pole choices differ between LAPACK builds).
     D_zeta (nD), D_kind (nD) int8  points of D, kind 0 = unmasked line node, 1 = wedge band (band_heights rows x band_x points, row-
                                 major, each row symmetric under z -> -conj(z)), 2 = i nu_n (n = 1..n_mats), 3 = nu_0 (z = 0)
       attrs: n_mats, band_d0 (= band_c sin(theta_t)/beta), band_vtop (= band_top zeta_T sin(theta)), nD
     Pi_D_probe (nq, nB, np, np)    oracle Pi (transition sum, DYNAMIC convention) at the D points of kind 1, 2, 3 (in D order;
                                 kind 0 = line nodes: see Pi_probe at bos_zeta); dynamic Pi(q, 0) = Pi^Mats(q, i nu_0) - dPi
     W_D_probe (nq, nD, np, np)     oracle W_dyn (Casida) at every point of D (kind 3: W^an(q, 0))
     tau_nodes, tau_weights (ntau) the tau-leg grid (timeray.tau_grid(beta, E_max of the KS poles, wstep_tau_nn, _per_efold, _x0))
     Pi0_tau_line_probe (nq, np, np)   the PROTOTYPE's tau-leg dynamic Pi(q, 0) (line code, thermal_tol ver_thermal_tol)
     nu_b (r)                       poles of the D-selected basis (eps_b = wstep_eps_b, candidates [1e-4 lam_b_verify, lam_b_verify])
     wres_probe (2, r, np, np), wres_q (2)   U^H w_j(q) U of the line-only path (line Pi on D, Dyson, fit_split) for 2 q
     need_zeta (nneed)              needed points z = zeta_f - e_m: fermionic test points with rho beta |zeta_f| in [c_zeta + c_T, 3 (c_zeta + c_T)]
                                 (inside the wedge, notes section 11.5(b)) minus the distinct window KS energies (|e| <= E_T at thermal_tol)
     W_need_probe (nq, nneed, np, np)   oracle W_dyn at need_zeta
     Sigma_line_zeta (nzf, nb, nb), Sigma_line_iw (nw, nb, nb)   line-only Sigma_c (KS thermal G lists at ver_thermal_tol,
                                 W from the redesigned step) at k = sigma_k[1] (compare with Sigma_zeta / Sigma_iw [1] on the masks)
     Verification attrs (line-only path at ver_thermal_tol vs the oracles; relative to the max over the point set and q):
     ver_tau_pi0                 tau-leg dynamic Pi(q, 0) vs the transition sum, all q                                      [gate 1e-12]
     ver_tau_dpi                 tau-leg degenerate-pair term vs pi_nu0_extra, all q (relative to max|Pi(q, 0)|)          [gate 1e-12]
     ver_line_pi_D               line Pi on D (rays at kinds 0-2, tau leg at kind 3) vs the transition sum, all q            [gate 1e-9]
     ver_w_nu0, ver_w_band, ver_w_mats, ver_w_line, ver_w_need   fitted W (fit_split on nu_b) at nu_0 / band / i nu_n / line
                                 nodes of D / needed points vs Casida W                                                       [gate 1e-10]
     ver_w_need30_wedge, ver_w_need30_all   (report) the same at zeta_f with rho beta |zeta_f| >= c_f = c_zeta (needed points
                                 inside the wedge / all, the latter up to 4/beta below its edge: extrapolated)
     ver_sigma_line_nodes, ver_sigma_line_iw   line-only Sigma_c vs Eq. fT_sigma at the fermionic test points rho beta |zeta| >= c_f
                                 and at i w_n >= zeta_T, max over sigma_k                                                     [gate 1e-8]
     report_sigma_line_tol_default   the line-only Sigma_c at the DEFAULT thermal_tol (far poles within ln(1/tol)/beta of the
                                 window edge carry weight 1 instead of 1 - f; Bose weights cut at E_T): the tau_T floor (k = sigma_k[1])

Usage: gen_finiteT_ref.py <lih222|lih223> [--betas 50,200,10000] [--out FILE] [--no-verify] [--lam-b 4]
(--no-verify skips the line checks and the line-only datasets Pi0_tau_line_probe / wres_probe / Sigma_line_*)
       gen_finiteT_ref.py <fx> --merge F1,F2,... --out FILE   merges single-beta runs (--betas B --out F_B, run in parallel) into
       one file: root datasets/attrs of F1 (checked equal across the files), the beta groups of all (Pi_full* from F1 only),
       attr betas updated.
Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_finiteT_ref.py lih222
"""
import sys, os, time, subprocess, datetime, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC
from cayley import finite_t as ft
from cayley.casida_g0w0 import thermal_factor
from cayley.line.thc_gw import LineGW, node_floor_mask, WSTEP_DEFAULTS
from cayley.line.line_dlr import BosonicLineBasis
from cayley.line.closure import chemical_potential_auto

def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
fx = sys.argv[1]
if '--merge' in sys.argv:
    srcs = opt('--merge', '').split(','); out = opt('--out', None)
    with h5py.File(out, 'w') as F:
        bl = []
        for i, src in enumerate(srcs):
            with h5py.File(src, 'r') as f:
                assert f.attrs['fixture'] == fx
                for k in f.keys():
                    if k.startswith('beta_'):                     # Pi_full* only in the first group, as in a single run
                        g = F.create_group(k); g.attrs.update(dict(f[k].attrs)); bl.extend(list(f.attrs['betas']))
                        for d in f[k].keys():
                            if not (d.startswith('Pi_full') and i > 0): f[k].copy(d, g)
                    elif i == 0:
                        f.copy(k, F)
                    else:
                        assert np.array_equal(f[k][()], F[k][()]), k
                if i == 0:
                    for k, a in f.attrs.items(): F.attrs[k] = a
        F.attrs['betas'] = np.array(bl, float)
    print(f"merged {srcs} -> {out} ({os.path.getsize(out) / 1e6:.2f} MB)"); sys.exit(0)
betas = [float(b) for b in opt('--betas', '50,200,10000').split(',')]
verify = '--no-verify' not in sys.argv
lam_b = float(opt('--lam-b', 4.0))
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = opt('--out', D + f'{fx}_finiteT_ref.h5')
TH, THT = np.deg2rad(20.0), np.deg2rad(10.0)
RHO = np.sin(TH - THT) / np.sin(THT)
TTOL, CZ, WPF, DEG = 1e-8, 30.0, 15.0, 1e-8
VTTOL = 1e-12                                     # thermal_tol of the line verification (strict gates)
GATES = dict(ver_line_pi=1e-9, ver_line_w_nodes=1e-8, ver_line_sigma_fullW=1e-8,
             ver_tau_pi0=1e-12, ver_tau_dpi=1e-12, ver_line_pi_D=1e-9, ver_w_nu0=1e-10, ver_w_band=1e-10, ver_w_mats=1e-10,
             ver_w_line=1e-10, ver_w_need=1e-10, ver_sigma_line_nodes=1e-8, ver_sigma_line_iw=1e-8)
WSTEP = dict(WSTEP_DEFAULTS)                      # the W-step parameters of the prototype (written as root attrs wstep_*)
CT = np.log(1 / TTOL)

thc = THC(D + f'{fx}_thc/thc.eri.h5')
with h5py.File(D + f'{fx}_thc/system.h5', 'r') as f:
    s = f['system']; eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, nb = eig.shape; Np = X.shape[1]; qminus = qk[:, 0].copy()
assert all(qk[qminus[iq], qk[iq, 0]] == 0 for iq in range(nk)) and np.all(qminus[qminus] == np.arange(nk))
nocc = int(round(nelec / 2))
kgap = int(np.argmin(np.sort(eig, 1)[:, nocc] - np.sort(eig, 1)[:, nocc - 1]))
sigma_k = [0, kgap] if kgap != 0 else [0, 1]
rng = np.random.default_rng(2026)
U = np.linalg.qr(rng.standard_normal((Np, 4)) + 1j * rng.standard_normal((Np, 4)))[0]
probe = lambda M: np.einsum('pa,...pq,qb->...ab', U.conj(), M, U)
tb = np.exp(np.linspace(np.log(1e-3), np.log(40.0), 48)); bz = np.concatenate([tb * np.exp(1j * TH), tb * np.exp(1j * (np.pi - TH))])
tf = np.exp(np.linspace(np.log(1e-3), np.log(60.0), 20)); fz = np.concatenate([tf * np.exp(1j * TH), tf * np.exp(1j * (np.pi - TH))])
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
try:
    head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception:
    head = '?'
print(f"{fx}: nk {nk} nb {nb} Np {Np} nelec {nelec} sigma_k {sigma_k} out {out}", flush=True)

def wr(g, name, a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        g[name + '_re'] = a.real; g[name + '_im'] = a.imag
    else:
        g[name] = a

def nu_list(beta, numax=40.0):
    nmax = int(numax * beta / (2 * np.pi))
    return np.unique(np.round(np.exp(np.linspace(0, np.log(max(nmax, 1)), 24))).astype(int))

def w_list(beta, wmax=60.0):
    nmax = int((wmax * beta / np.pi - 1) / 2)
    return np.unique(np.concatenate([np.arange(0, 4), np.round(np.exp(np.linspace(np.log(4), np.log(max(nmax, 5)), 20))).astype(int)]))

groups = {}
for beta in betas:
    T0 = time.time(); res = {}
    mu0, rule = ft.mu0_auto(eig, nelec, beta, TTOL)
    v = np.array([np.eye(nb)] * nk)
    dmu_c, rule_c, _ = chemical_potential_auto(eig - mu0, v, nk, nelec, beta, TTOL)
    e = eig - mu0
    E_T = np.log(1 / TTOL) / beta
    win = np.abs(e) <= E_T; active = bool(win.any())
    zT = CZ / (RHO * beta)
    bmask = node_floor_mask(bz, beta, TH, THT, CZ) if active else np.ones(len(bz), bool)
    fmask = node_floor_mask(fz, beta, TH, THT, CZ) if active else np.ones(len(fz), bool)
    f = ft.fermi(e, beta)
    Dl = np.where(win, f, (e < 0).astype(float))
    h = 1e-7
    res.update(beta=beta, mu0=mu0, mu_rule=rule, N_mu0=ft.electron_number(eig, mu0, beta),
               dN_dmu=(ft.electron_number(eig, mu0 + h, beta) - ft.electron_number(eig, mu0 - h, beta)) / (2 * h),
               E_T=E_T, zeta_T=zT, S_T=beta / np.sin(THT), thermal_active=int(active))
    print(f"\n[beta {beta:g}] mu0 {mu0:.12f} ({rule}; closure.chemical_potential_auto: {rule_c}, dmu {dmu_c:.1e}) E_T {E_T:.4f} zeta_T {zT:.4f} "
          f"active {active} window/k {win.sum(1)}", flush=True)
    assert rule == rule_c and abs(dmu_c) < 1e-9
    # ---------------------------------------------------------------- oracles
    nus = nu_list(beta); wn = w_list(beta)
    iw = 1j * np.pi * (2 * wn + 1) / beta; iwmask = np.abs(iw) >= zT if active else np.ones(len(iw), bool)
    trs, cas = [], []
    Pi_probe = np.zeros((nk, len(bz), 4, 4), complex); W_zeta = np.zeros_like(Pi_probe)
    W_inu = np.zeros((nk, len(nus), 4, 4), complex); W0an = np.zeros((nk, 4, 4), complex); W0m = np.zeros_like(W0an)
    dPi_p = np.zeros_like(W0an); ndeg = np.zeros(nk, int); dW = []
    cvd = 0.0
    for iq in range(nk):
        tr = ft.transitions(X, e, qk, iq, beta, deg_tol=DEG); trs.append(tr)
        lam, al, be, info = ft.casida_from_transitions(tr, Z[iq]); cas.append((lam, al, be))
        Pt = ft.pi_transition(tr, bz); Pi_probe[iq] = probe(Pt)
        Wc = ft.casida_w(lam, al, be, bz); W_zeta[iq] = probe(Wc)
        cvd = max(cvd, rel(Wc, ft.dyson_w(Z[iq], Pt)))
        W_inu[iq] = probe(ft.casida_w(lam, al, be, 2j * np.pi * nus / beta))
        P0 = ft.pi_transition(tr, [0.0])[0]; dP = ft.pi_nu0_extra(tr, beta)
        Wa = ft.dyson_w(Z[iq], [P0])[0]; Wm = ft.dyson_w(Z[iq], [P0 + dP])[0]
        W0an[iq], W0m[iq], dPi_p[iq], ndeg[iq] = probe(Wa), probe(Wm), probe(dP), len(tr['wd'])
        dW.append(Wm - Wa if len(tr['wd']) else None)
        cvd = max(cvd, rel(ft.casida_w(lam, al, be, [1e-300j])[0], Wa))
    res['ver_casida_vs_dyson'] = cvd
    print(f"  oracles: Casida vs Dyson[transition sum] {cvd:.1e}; ndeg per q {ndeg}  [{time.time() - T0:.0f}s]", flush=True)
    # second route (tau quadrature, Matsubara convention) on q = 0 and the first q != -q (or q = 1)
    qs2 = [0, next((iq for iq in range(nk) if qminus[iq] != iq), 1)]
    e1 = e0 = ecv = 0.0
    for iq in qs2:
        Pm = ft.pi_matsubara_tau(X, e, qk, iq, beta, [0, 1, 2, 5])
        Pt = ft.pi_transition(trs[iq], 2j * np.pi * np.array([0, 1, 2, 5]) / beta)
        e1 = max(e1, rel(Pm[1:], Pt[1:])); e0 = max(e0, rel(Pm[0], Pt[0] + ft.pi_nu0_extra(trs[iq], beta)))
        ecv = max(ecv, rel(ft.casida_w(*cas[iq], 2j * np.pi * np.array([1, 2, 5]) / beta), ft.dyson_w(Z[iq], Pm[1:])))
    res.update(ver_tau_route_pi=e1, ver_tau_route_nu0=e0, ver_casida_vs_tau=ecv)
    print(f"  tau route (q {qs2}): Pi(i nu_1,2,5) {e1:.1e}, Pi(i nu_0) with dPi {e0:.1e}, Casida W vs Dyson[tau Pi] {ecv:.1e}", flush=True)
    Sz = np.array([ft.sigma_fT(X, e, qk, qminus, cas, beta, k, fz) for k in sigma_k])
    Siw = np.array([ft.sigma_fT(X, e, qk, qminus, cas, beta, k, iw) for k in sigma_k])
    nu0 = np.array([ft.sigma_nu0_term(X, e, qk, dW, beta, k, iw) for k in sigma_k])
    # c_T form (casida_g0w0): all Casida poles of q, weight -c_T(e_m, lam)/Nk at E = e_m + lam
    k0 = sigma_k[0]; zc = np.concatenate([fz[fmask][:6], iw[:4]]); Sc = np.zeros((len(zc), nb, nb), complex)
    for iq in range(nk):
        lam, al, be = cas[iq]; ikmq = qk[iq, k0]; Xm = X[ikmq]
        for m in range(nb):
            A = (X[k0].conj() * Xm[:, m][:, None]).T @ al; Bm = be @ (Xm[:, m].conj()[:, None] * X[k0])
            w = -thermal_factor(e[ikmq, m], lam, beta) / nk
            for iz, z in enumerate(zc): Sc[iz] += (A * (w / (z - e[ikmq, m] - lam))[None, :]) @ Bm
    res['ver_sigma_vs_cT'] = rel(Sc, ft.sigma_fT(X, e, qk, qminus, cas, beta, k0, zc))
    print(f"  Sigma: Eq. fT_sigma vs c_T form {res['ver_sigma_vs_cT']:.1e}; |nu0 term|/|Sigma_iw| {np.abs(nu0).max() / np.abs(Siw).max():.1e}  [{time.time() - T0:.0f}s]", flush=True)
    # ---------------------------------------------------------------- redesigned W step: data set D and its oracles
    bos = BosonicLineBasis(TH, lam_b, eps=1e-10, gap=0.0)
    gw = LineGW(X, Z, qk, nk, mu0, TH, THT, bos, fz, wstep=WSTEP)
    gw.set_poles(e, v.astype(complex), beta=beta, thermal_tol=VTTOL, thermal_floor=CZ)
    data_w = {}
    res['wstep_active'] = int(active)
    if active:
        zD, kD = gw.zD, gw.kD
        ewin = np.unique(np.round(e[win], 10))
        mneed = (RHO * beta * np.abs(fz) >= CZ + CT) & (RHO * beta * np.abs(fz) <= 3 * (CZ + CT))
        need = (fz[mneed][:, None] - ewin[None, :]).ravel()
        PiD = np.zeros((nk, int(np.sum(kD > 0)), 4, 4), complex); WD = np.zeros((nk, len(zD), 4, 4), complex)
        Wn = np.zeros((nk, len(need), 4, 4), complex)
        for iq in range(nk):
            PiD[iq] = probe(ft.pi_transition(trs[iq], zD[kD > 0]))
            WD[iq] = probe(ft.casida_w(*cas[iq], zD)); Wn[iq] = probe(ft.casida_w(*cas[iq], need))
        res.update(n_mats=int(np.sum(kD == 2)), nD=len(zD), band_d0=float((CZ if WSTEP['band_c'] is None else WSTEP['band_c']) * np.sin(THT) / beta),
                   band_vtop=float(WSTEP['band_top'] * zT * np.sin(TH)), tau_emax=float(np.abs(e).max()))
        data_w.update(D_zeta=zD, D_kind=kD, Pi_D_probe=PiD, W_D_probe=WD, tau_nodes=gw.tau, tau_weights=gw.wtau, nu_b=gw.bos_w.nu,
                      need_zeta=need, W_need_probe=Wn)
        print(f"  W step data set D: {len(zD)} points (line {np.sum(kD == 0)}, band {np.sum(kD == 1)}, i nu_n {np.sum(kD == 2)}, nu_0 1), "
              f"basis rank {gw.bos_w.r}, tau nodes {len(gw.tau)}, needed points {len(need)}  [{time.time() - T0:.0f}s]", flush=True)
    # ---------------------------------------------------------------- line prototype vs the oracles
    if verify:
        gw.set_poles(e, v.astype(complex), beta=beta, thermal_tol=TTOL, thermal_floor=CZ)
        assert gw.thermal == active
        res['ver_line_pi_tol_default'] = max(rel(gw.polarization(iq, bz[bmask]), ft.pi_transition(trs[iq], bz[bmask])) for iq in range(nk))
        print(f"  line Pi at the default thermal_tol {TTOL:g} (far poles at weight 1): {res['ver_line_pi_tol_default']:.1e} (report only)", flush=True)
        if active:
            wtd, _ = gw.w_step(qminus)
            k1 = sigma_k[1]; zs1 = np.concatenate([fz[fmask], iw[iwmask]])
            res['report_sigma_line_tol_default'] = rel(gw.sigma(k1, wtd, zs1, qminus=qminus), np.concatenate([Sz[1][fmask], Siw[1][iwmask]]))
            print(f"  line-only Sigma_c at the default thermal_tol (k {k1}): {res['report_sigma_line_tol_default']:.1e} (report only: tau_T floor)  [{time.time() - T0:.0f}s]", flush=True)
            del wtd
        gw.set_poles(e, v.astype(complex), beta=beta, thermal_tol=VTTOL, thermal_floor=CZ)
        bm_basis = node_floor_mask(bos.zeta, beta, TH, THT, CZ) if active else np.ones(len(bos.zeta), bool)
        epi = ewn = 0.0
        for iq in range(nk):
            Pl = gw.polarization(iq, bz[bmask])
            Pt = ft.pi_transition(trs[iq], bz[bmask])
            epi = max(epi, rel(Pl, Pt))
            ewn = max(ewn, rel(gw.dyson_w(iq, Pl), ft.casida_w(*cas[iq], bz[bmask])))
        res.update(ver_line_pi=epi, ver_line_w_nodes=ewn)
        print(f"  line Pi vs transition sum (unmasked test nodes, all q) {epi:.1e}; Dyson W vs Casida {ewn:.1e}  [{time.time() - T0:.0f}s]", flush=True)
        # Sigma machinery with W residues from the exact W on ALL basis nodes
        Wx = [ft.casida_w(*cas[iq], bos.zeta) for iq in range(nk)]
        wx = [bos.fit(bos.zeta, Wx[iq], W_minus=None if qminus[iq] == iq else Wx[qminus[iq]]) for iq in range(nk)]
        zs = np.concatenate([fz[fmask], iw[iwmask]])
        Sl = gw.sigma(k0, wx, zs, qminus=qminus, nu=[bos.nu] * nk)
        res['ver_line_sigma_fullW'] = rel(Sl, np.concatenate([Sz[0][fmask], Siw[0][iwmask]]))
        print(f"  line Sigma (W fitted on all nodes of the exact W) vs Eq. fT_sigma: {res['ver_line_sigma_fullW']:.1e}  [{time.time() - T0:.0f}s]", flush=True)
        del wx, Wx
        if active:
            # redesigned W step, line only: tau leg, Pi on D, Dyson, D-selected basis, joint split fit, Sigma_c
            et0 = edp = epd = 0.0; P0l = np.zeros((nk, 4, 4), complex); PiDl = []
            for iq in range(nk):
                PiM, dPi = gw.pi_tau_leg(iq, [0])
                P0 = ft.pi_transition(trs[iq], [0.0])[0]
                et0 = max(et0, rel(PiM[0] - dPi, P0)); edp = max(edp, float(np.abs(dPi - ft.pi_nu0_extra(trs[iq], beta)).max() / np.abs(P0).max()))
                P0l[iq] = probe(PiM[0] - dPi)
                Pl = gw.polarization(iq, zD); PiDl.append(Pl)
                epd = max(epd, rel(Pl, np.concatenate([ft.pi_transition(trs[iq], zD[kD < 3]), [P0]])))
            res.update(ver_tau_pi0=et0, ver_tau_dpi=edp, ver_line_pi_D=epd)
            print(f"  tau leg: dynamic Pi(q, 0) {et0:.1e}, degenerate term {edp:.1e}; line Pi on D vs transition sum {epd:.1e}  [{time.time() - T0:.0f}s]", flush=True)
            wres, _ = gw.w_step(qminus, Pi=PiDl); del PiDl
            def werr(zp):
                return max(rel(gw.bos_w.eval(wres[iq], zp, w_minus=wres[qminus[iq]]), ft.casida_w(*cas[iq], zp)) for iq in range(nk))
            m30 = RHO * beta * np.abs(fz) >= CZ
            n30 = (fz[m30][:, None] - ewin[None, :]).ravel()
            in30 = n30.imag * np.cos(THT) - np.abs(n30.real) * np.sin(THT) >= res['band_d0']
            res.update(ver_w_nu0=werr(zD[kD == 3]), ver_w_band=werr(zD[kD == 1]), ver_w_mats=werr(zD[kD == 2]), ver_w_line=werr(zD[kD == 0]),
                       ver_w_need=werr(need), ver_w_need30_wedge=werr(n30[in30]), ver_w_need30_all=werr(n30))
            print("  fitted W vs Casida: " + ", ".join(f"{k[6:]} {res[k]:.1e}" for k in ('ver_w_nu0', 'ver_w_band', 'ver_w_mats', 'ver_w_line', 'ver_w_need',
                  'ver_w_need30_wedge', 'ver_w_need30_all')) + f"  [{time.time() - T0:.0f}s]", flush=True)
            qw = [0, qs2[1]]
            data_w['wres_probe'] = np.array([probe(wres[q]) for q in qw]); data_w['wres_q'] = np.array(qw)
            data_w['Pi0_tau_line_probe'] = P0l
            esn = esi = 0.0
            for j, kk in enumerate(sigma_k):
                Sl_z = gw.sigma(kk, wres, fz, qminus=qminus); Sl_i = gw.sigma(kk, wres, iw, qminus=qminus)
                sc_ = max(np.abs(Sz[j][fmask]).max(), np.abs(Siw[j][iwmask]).max())
                esn = max(esn, float(np.abs(Sl_z - Sz[j])[fmask].max() / sc_)); esi = max(esi, float(np.abs(Sl_i - Siw[j])[iwmask].max() / sc_))
                if j == 1: data_w.update(Sigma_line_zeta=Sl_z, Sigma_line_iw=Sl_i)
            res.update(ver_sigma_line_nodes=esn, ver_sigma_line_iw=esi)
            print(f"  line-only Sigma_c vs Eq. fT_sigma (k {sigma_k}): nodes rho beta |zeta| >= {CZ:g} {esn:.1e}, i w_n >= zeta_T {esi:.1e}  [{time.time() - T0:.0f}s]", flush=True)
            del wres
        for k, g in GATES.items():
            if k in res and res[k] > g:
                print(f"  GATE FAILED: {k} = {res[k]:.1e} > {g:g}", flush=True)
                sys.exit(2)
    data = dict(f=f, window=win.astype(np.int8), window_counts=win.sum(1), D_exact_diag=f, D_lists_diag=Dl,
                bos_mask=bmask.astype(np.int8), Pi_probe=Pi_probe, ndeg=ndeg, dPi_nu0_probe=dPi_p, W_zeta_probe=W_zeta,
                nu_n=nus, W_inu_probe=W_inu, W0_an_probe=W0an, W0_mats_probe=W0m, Sigma_zeta=Sz, ferm_mask=fmask.astype(np.int8),
                w_n=wn, iw_mask=iwmask.astype(np.int8), Sigma_iw=Siw, nu0_term=nu0, **data_w)
    if beta == betas[0]:
        n1 = np.where(bmask)[0][0]; qf = [0, qs2[1]]
        data.update(Pi_full=np.array([ft.pi_transition(trs[q], [bz[n1]])[0] for q in qf]), Pi_full_q=np.array(qf), Pi_full_node=np.array([n1, n1]))
    groups[beta] = (res, data)
    del trs, cas

with h5py.File(out, 'w') as F:
    F.attrs.update(fixture=fx, thc_dir=f'{fx}_thc', generator='coqui/cayley/scripts/gen_finiteT_ref.py', cayley_head=head,
                   date=datetime.date.today().isoformat(), theta_deg=20.0, theta_t_deg=10.0, rho=RHO, thermal_tol=TTOL,
                   thermal_floor=CZ, wp_floor=WPF, deg_tol=DEG, nk=nk, nb=nb, Np=Np, nelec=nelec, nprobe=4,
                   betas=np.array(betas), sigma_k=np.array(sigma_k), lam_b_verify=lam_b, ver_thermal_tol=VTTOL, c_T=CT,
                   **{'wstep_' + k: (np.nan if v is None else v) for k, v in WSTEP.items()})
    wr(F, 'eig', eig); wr(F, 'qk_to_k2', qk); wr(F, 'qminus', qminus); wr(F, 'probe', U); wr(F, 'bos_zeta', bz); wr(F, 'ferm_zeta', fz)
    for beta, (res, data) in groups.items():
        g = F.create_group(f'beta_{int(beta)}')
        g.attrs.update(res)
        for k, a in data.items(): wr(g, k, a)
print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
