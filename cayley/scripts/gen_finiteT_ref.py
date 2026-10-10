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
     design_w_inu_masked         DESIGN PATH (report only): W residues fitted on the unmasked bosonic basis nodes from line Pi,
                                 W(i nu_n >= zeta_T) vs Casida
     design_sigma_masked         DESIGN PATH (report only): line Sigma_c with those residues vs Eq. fT_sigma (unmasked nodes / i w_n)

Usage: gen_finiteT_ref.py <lih222|lih223> [--betas 50,200,10000] [--out FILE] [--no-verify] [--lam-b 4]
Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_finiteT_ref.py lih222
"""
import sys, os, time, subprocess, datetime, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC
from cayley import finite_t as ft
from cayley.casida_g0w0 import thermal_factor
from cayley.line.thc_gw import LineGW, node_floor_mask
from cayley.line.line_dlr import BosonicLineBasis
from cayley.line.closure import chemical_potential_auto

def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
fx = sys.argv[1]
betas = [float(b) for b in opt('--betas', '50,200,10000').split(',')]
verify = '--no-verify' not in sys.argv
lam_b = float(opt('--lam-b', 4.0))
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = opt('--out', D + f'{fx}_finiteT_ref.h5')
TH, THT = np.deg2rad(20.0), np.deg2rad(10.0)
RHO = np.sin(TH - THT) / np.sin(THT)
TTOL, CZ, WPF, DEG = 1e-8, 30.0, 15.0, 1e-8
VTTOL = 1e-12                                     # thermal_tol of the line verification (strict gates)
GATES = dict(ver_line_pi=1e-9, ver_line_w_nodes=1e-8, ver_line_sigma_fullW=1e-8)

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
    # ---------------------------------------------------------------- line prototype vs the oracles
    if verify:
        bos = BosonicLineBasis(TH, lam_b, eps=1e-10, gap=0.0)
        gw = LineGW(X, Z, qk, nk, mu0, TH, THT, bos, fz)
        gw.set_poles(e, v.astype(complex), beta=beta, thermal_tol=TTOL, thermal_floor=CZ)
        assert gw.thermal == active
        res['ver_line_pi_tol_default'] = max(rel(gw.polarization(iq, bz[bmask]), ft.pi_transition(trs[iq], bz[bmask])) for iq in range(nk))
        print(f"  line Pi at the default thermal_tol {TTOL:g} (far poles at weight 1): {res['ver_line_pi_tol_default']:.1e} (report only)", flush=True)
        gw.set_poles(e, v.astype(complex), beta=beta, thermal_tol=VTTOL, thermal_floor=CZ)
        bm_basis = gw_mask = node_floor_mask(bos.zeta, beta, TH, THT, CZ) if active else np.ones(len(bos.zeta), bool)
        zl = np.concatenate([bz[bmask], bos.zeta[bm_basis]])
        epi = ewn = 0.0; Wl = []
        for iq in range(nk):
            Pl = gw.polarization(iq, zl)
            nb1 = bmask.sum()
            Pt = ft.pi_transition(trs[iq], bz[bmask])
            epi = max(epi, rel(Pl[:nb1], Pt))
            Wd = gw.dyson_w(iq, Pl)
            ewn = max(ewn, rel(Wd[:nb1], ft.casida_w(*cas[iq], bz[bmask])))
            Wl.append(Wd[nb1:])
        res.update(ver_line_pi=epi, ver_line_w_nodes=ewn)
        print(f"  line Pi vs transition sum (unmasked test nodes, all q) {epi:.1e}; Dyson W vs Casida {ewn:.1e}  [{time.time() - T0:.0f}s]", flush=True)
        # Sigma machinery with W residues from the exact W on ALL basis nodes
        Wx = [ft.casida_w(*cas[iq], bos.zeta) for iq in range(nk)]
        wx = [bos.fit(bos.zeta, Wx[iq], W_minus=None if qminus[iq] == iq else Wx[qminus[iq]]) for iq in range(nk)]
        zs = np.concatenate([fz[fmask], iw[iwmask]])
        Sl = gw.sigma(k0, wx, zs, qminus=qminus)
        res['ver_line_sigma_fullW'] = rel(Sl, np.concatenate([Sz[0][fmask], Siw[0][iwmask]]))
        print(f"  line Sigma (W fitted on all nodes of the exact W) vs Eq. fT_sigma: {res['ver_line_sigma_fullW']:.1e}  [{time.time() - T0:.0f}s]", flush=True)
        del wx, Wx
        # design path: residues from the line Pi at the unmasked basis nodes only (report only)
        zb = bos.zeta[bm_basis]
        wd = [bos.fit(zb, Wl[iq], W_minus=None if qminus[iq] == iq else Wl[qminus[iq]]) for iq in range(nk)]
        inu = 2j * np.pi * nus / beta; big = np.abs(inu) >= zT if active else np.ones(len(inu), bool)
        res['design_w_inu_masked'] = max(rel(bos.eval(wd[iq], inu[big], w_minus=wd[qminus[iq]]), ft.casida_w(*cas[iq], inu[big])) for iq in range(nk))
        Sd = gw.sigma(k0, wd, zs, qminus=qminus)
        res['design_sigma_masked'] = rel(Sd, np.concatenate([Sz[0][fmask], Siw[0][iwmask]]))
        print(f"  DESIGN PATH (bosonic fit on the unmasked nodes): W(i nu_n >= zeta_T) {res['design_w_inu_masked']:.1e}; Sigma {res['design_sigma_masked']:.1e}  [{time.time() - T0:.0f}s]", flush=True)
        del wd, Wl
        for k, g in GATES.items():
            if res[k] > g:
                print(f"  GATE FAILED: {k} = {res[k]:.1e} > {g:g}", flush=True)
                sys.exit(2)
    data = dict(f=f, window=win.astype(np.int8), window_counts=win.sum(1), D_exact_diag=f, D_lists_diag=Dl,
                bos_mask=bmask.astype(np.int8), Pi_probe=Pi_probe, ndeg=ndeg, dPi_nu0_probe=dPi_p, W_zeta_probe=W_zeta,
                nu_n=nus, W_inu_probe=W_inu, W0_an_probe=W0an, W0_mats_probe=W0m, Sigma_zeta=Sz, ferm_mask=fmask.astype(np.int8),
                w_n=wn, iw_mask=iwmask.astype(np.int8), Sigma_iw=Siw, nu0_term=nu0)
    if beta == betas[0]:
        n1 = np.where(bmask)[0][0]; qf = [0, qs2[1]]
        data.update(Pi_full=np.array([ft.pi_transition(trs[q], [bz[n1]])[0] for q in qf]), Pi_full_q=np.array(qf), Pi_full_node=np.array([n1, n1]))
    groups[beta] = (res, data)
    del trs, cas

with h5py.File(out, 'w') as F:
    F.attrs.update(fixture=fx, thc_dir=f'{fx}_thc', generator='coqui/cayley/scripts/gen_finiteT_ref.py', cayley_head=head,
                   date=datetime.date.today().isoformat(), theta_deg=20.0, theta_t_deg=10.0, rho=RHO, thermal_tol=TTOL,
                   thermal_floor=CZ, wp_floor=WPF, deg_tol=DEG, nk=nk, nb=nb, Np=Np, nelec=nelec, nprobe=4,
                   betas=np.array(betas), sigma_k=np.array(sigma_k), lam_b_verify=lam_b, ver_thermal_tol=VTTOL)
    wr(F, 'eig', eig); wr(F, 'qk_to_k2', qk); wr(F, 'qminus', qminus); wr(F, 'probe', U); wr(F, 'bos_zeta', bz); wr(F, 'ferm_zeta', fz)
    for beta, (res, data) in groups.items():
        g = F.create_group(f'beta_{int(beta)}')
        g.attrs.update(res)
        for k, a in data.items(): wr(g, k, a)
print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
