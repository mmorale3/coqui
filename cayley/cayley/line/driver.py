"""Self-consistent GW on the tilted line (option D), numpy prototype. One object = one system; `iterate()` does one loop.

Loop (pole data in -> pole data out), all energies mu-relative, mu absolute in self.mu:
  1. W(q): Pi(q) on the bosonic nodes from the current G poles (time ray), Dyson, symmetric real-pole refit -> wres[q]
  2. Sigma^>(k), Sigma^<(k) on the fermionic nodes (per-sector time rays), linear mixing with the previous iteration
  3. per k: real-pole fit of each sector -> Cayley moments (total measure) -> block upfolding -> upfolded Hamiltonian with
     H_stat = H0 + F[Dm] - mu -> Lehmann G (positive) ; new mu from T=0 filling ; re-centre ; compressed per-sector fit of G
  4. convergence: max |Delta Sigma| at the nodes, |Delta mu|, gap
Diagnostics kept per iteration in self.history.

Finite temperature (S8b, notes section 11; beta given): the thermal loop _iterate_thermal(), line-only (no oracle input):
  1. W: LineGW.w_step on the bosonic data set D (unmasked line nodes of the GAPLESS bosonic line basis self.bosT, wedge band,
     i nu_n from the guarded ray products; nu_0 from the tau leg), Dyson on D, basis selected on D, joint odd/even pair fit
  2. TOTAL Sigma_c at the fermionic nodes (thermal G lists, Bose-weighted W(t) of Eq. fT_W), linear mixing
  3. per k: fit of the total Sigma_c at the nodes rho beta |zeta| >= c_f on the two-sided gapless basis self.bt, Cayley moments at
     omega_p = max(wp, wp_floor zeta_T), upfolding, Lehmann G (g_repr = "lehmann": poles and vectors kept as they are)
  4. mu by rule mu_rule ("auto": gap midpoint if both edges are outside the window, else N(mu) = N_el; "number"; "gap"),
     re-centre, thermal sector lists of the new poles, D = density of the thermal hole list, F = V_H + Sigma_x.
  If the window is empty at an iteration (no pole within E_T of mu) the same loop runs with the T = 0 kernels (gapless
  bosonic basis fitted on all its nodes, all fermionic nodes) - T = 0 equivalent, not the T = 0 driver above.
  scf_density = "matsubara" (S8b.3 hybrid, notes section 11.6 "Hybrid"; default "closure" = the loop above, unchanged):
  2b. Sigma_c(i w_n) on the dense set n = 0..N-1 (w_N >= hyb_wmax) from the tau leg LineGW.sigma_tau_leg (same W residues,
      ALL G poles, exact f / n weights), mixed like the node Sigma_c, with its end-point moments S1, S2;
  4'. Matsubara Dyson G(i w_n; mu) = [i w_n + dmu - (H0 + F - mu) - Sigma_c(i w_n)]^-1, D by the Matsubara sum (subtracted
      free reference + analytic tail, finite_t.density_matsubara), mu by N(mu) = N_el (bisection, Sigma_c fixed): mu, D (-> F)
      and N come from here; the closure (step 3) supplies only the next propagator poles, re-centred at the new mu. Its own
      density / count / mu choice are logged next to the hybrid ones (D_cl_err = max|D_closure - D|, N_closure, dN_closure,
      dmu_closure + rule). Initial state: mu_0 by N(mu_0) = N_el, D = f(H) exactly (Sigma_c = 0)."""
import numpy as np, time
from .line_dlr import LineBasis, BosonicLineBasis
from .thc_gw import LineGW
from .closure import fit_sigma_sectors, lehmann_from_sigma, chemical_potential, compress_sectors
from .closure import fit_sigma_total, wp_thermal, chemical_potential_auto, chemical_potential_T, electron_count_T
from .thc_gw import node_floor_mask, thermal_rho
from .. import finite_t as _ft


class LineSCGW:
    def __init__(self, X, Z, qk_to_k2, nk, nelec, H0, mu, theta=np.deg2rad(20), eps=1e-8, lam=6.0, bos_lam=4.0, bos_gap=0.02,
                 sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=24, tol_gram=1e-10, mixing=0.5, verbose=True, k_weight=None,
                 nodes_per_ray=120, node_range=(1e-3, 60.0), beta=None, thermal_tol=1e-8, thermal_floor=30.0, thermal_floor_f=None,
                 wp_floor=15.0, mu_rule='auto', bos_eps_T=1e-10, qminus=None, wstep=None, mu_dn_max=0.1, mu_th_factor=10.0,
                 scf_density='closure', hyb_wmax=100.0, hyb_tau_kw=None, theta_t_frac=0.5):
        """lam: real-pole range (Ha) of the fermionic bases (must cover the support of Sigma_c and of G: band edges + plasmon,
        ~5 Ha for Si); bos_lam: bosonic (W) range; the fermionic data live on a dense log grid of nodes_per_ray points per ray
        over node_range (Ha) — dense nodes are what pins the weight distribution of the far poles (dev/tune_sigma_fit.txt).
        Finite T (beta not None; module docstring): thermal_tol / thermal_floor (c_zeta) / thermal_floor_f (c_f, default c_zeta) /
        wp_floor / mu_rule as notes section 11; bos_eps_T: eps of the gapless bosonic LINE basis whose unmasked nodes enter D;
        qminus: index of -q per q (default qk_to_k2[:, 0], k index 0 = Gamma); wstep: LineGW W-step parameters.
        scf_density (finite T only): "closure" (default) or "matsubara" (S8b.3 hybrid, module docstring); hyb_wmax: w_max (Ha)
        of the dense Matsubara set; hyb_tau_kw: (nn, per_efold, x0) of the Sigma tau leg (default thc_gw.HYBRID_DEFAULTS).
        theta_t_frac: ray angle theta_t = theta_t_frac * theta (0.5 = theta/2, bitwise the previous rule; metals: 0.25)."""
        self.X, self.Z, self.qk, self.nk, self.nelec, self.H0 = X, Z, qk_to_k2, nk, nelec, H0
        self.nb = X.shape[2]; self.mu = mu; self.theta, self.eps, self.lam = theta, eps, lam
        self.theta_t = theta * theta_t_frac
        self.wp, self.K, self.mixing, self.verbose = wp, K, mixing, verbose
        self.tol_gram = tol_gram if tol_gram is not None else 1e-10
        self.k_weight = k_weight
        tmax = node_range[1]
        self.bos = BosonicLineBasis(theta, lam=bos_lam, eps=eps, gap=bos_gap)
        self.bp = LineBasis(theta, lam=lam, eps=eps, gap=(lam, sig_gap[1]), tmax=tmax); self.bh = LineBasis(theta, lam=lam, eps=eps, gap=(sig_gap[0], lam), tmax=tmax)
        self.gp = LineBasis(theta, lam=lam, eps=eps, gap=(lam, g_gap[1]), tmax=tmax); self.gh = LineBasis(theta, lam=lam, eps=eps, gap=(g_gap[0], lam), tmax=tmax)
        t = np.exp(np.linspace(np.log(node_range[0]), np.log(node_range[1]), nodes_per_ray))
        self.fz = np.concatenate([t * np.exp(1j * theta), t * np.exp(1j * (np.pi - theta))])
        self.gw = LineGW(X, Z, qk_to_k2, nk, mu, theta, self.theta_t, self.bos, self.fz)
        self.Sig_prev = None; self.history = []
        if verbose: print(f"LineSCGW: {self.bos}\n  Sigma bases {self.bp.r}+{self.bh.r}, G bases {self.gp.r}+{self.gh.r}, fermionic nodes {len(self.fz)}", flush=True)
        self.beta = beta
        assert scf_density in ('closure', 'matsubara') and (beta is not None or scf_density == 'closure')
        self.scf_density = scf_density
        if beta is not None:
            self.thermal_tol, self.thermal_floor, self.wp_floor, self.mu_rule = thermal_tol, thermal_floor, wp_floor, mu_rule
            self.mu_dn_max, self.mu_th_factor = mu_dn_max, mu_th_factor           # rule "auto" (notes Eq. fT_nth)
            self.thermal_floor_f = thermal_floor if thermal_floor_f is None else thermal_floor_f
            self.qminus = np.asarray(qk_to_k2[:, 0] if qminus is None else qminus)
            self.bosT = BosonicLineBasis(theta, lam=bos_lam, eps=bos_eps_T, gap=0.0)
            self.bt = LineBasis(theta, lam=lam, eps=eps, gap=(0.0, 0.0), tmax=tmax)
            self.gw = LineGW(X, Z, qk_to_k2, nk, mu, theta, self.theta_t, self.bosT, self.fz, wstep=wstep)
            self.rho = thermal_rho(theta, self.theta_t)
            self.zeta_T = thermal_floor / (self.rho * beta)
            self.fmask = node_floor_mask(self.fz, beta, theta, self.theta_t, self.thermal_floor_f)
            if scf_density == 'matsubara':
                self.hyb_tau_kw, self.Siw_prev = hyb_tau_kw, None
                self.iw_h = _ft.matsubara_set(beta, hyb_wmax)[1]
                if verbose: print(f"  scf_density matsubara: dense Matsubara set N = {len(self.iw_h)} (w_max {hyb_wmax:g} Ha)", flush=True)
            if verbose: print(f"  finite T: beta {beta:g}, thermal_tol {thermal_tol:g}, c_zeta {thermal_floor:g}, c_f {self.thermal_floor_f:g}, "
                              f"zeta_T {self.zeta_T:.4f}, omega_p {wp_thermal(wp, self.zeta_T, wp_floor):.4f}, gapless bosonic line basis "
                              f"{self.bosT.r} ({len(self.bosT.zeta)} nodes), two-sided Sigma basis {self.bt.r}, fermionic nodes kept "
                              f"{int(self.fmask.sum())}/{len(self.fz)}, mu_rule {mu_rule}", flush=True)

    def start_from_hamiltonian(self, H):
        """Initial poles from a one-body Hamiltonian H (nk, nb, nb) (e.g. KS or H0 + F).
        Finite T: mu re-centred by mu_rule on these poles (KS: notes section 11.6 initial state), thermal lists."""
        e, v = LineGW.poles_from_hamiltonian(H, self.mu)
        if self.beta is not None and self.scf_density == 'matsubara':
            dmu, _ = chemical_potential_T(e, v, self.nk, self.nelec, self.beta, self.k_weight)
            self.mu_rule_used = 'matsubara'
            self.mu += dmu; self.gw.mu = self.mu; e = e - dmu
            self._set_poles_T(e, v)
            f = _ft.fermi(self.e_leh, self.beta)                              # Sigma_c = 0: D = f(H) exactly (all poles)
            self.F = self.gw.hartree_exchange(np.einsum('km,kim,kjm->kij', f, self.v_leh, self.v_leh.conj()))
            return
        if self.beta is not None:
            dmu, self.mu_rule_used = self._mu_shift(e, v)
            self.mu += dmu; self.gw.mu = self.mu; e = e - dmu
            self._set_poles_T(e, v)
            self.F = self.gw.hartree_exchange(self.gw.density_matrix())
            return
        self.gw.set_poles(e, v)
        self.F = self.gw.hartree_exchange(self.gw.density_matrix())

    # ---------------------------------------------------------------- finite temperature (S8b)
    def _mu_shift(self, e, v):
        """(mu shift, rule used) by self.mu_rule for mu-relative Lehmann poles e (nk, M), v (nk, nb, M)."""
        if self.mu_rule == 'number':
            return chemical_potential_T(e, v, self.nk, self.nelec, self.beta, self.k_weight)[0], 'number'
        if self.mu_rule == 'gap':
            return chemical_potential(e, v, self.nk, self.nelec, self.k_weight)[0], 'gap'
        dmu, rule, _ = chemical_potential_auto(e, v, self.nk, self.nelec, self.beta, self.thermal_tol, self.k_weight,
                                               mu_dn_max=self.mu_dn_max, mu_th_factor=self.mu_th_factor)
        return dmu, rule

    def _set_poles_T(self, e, v):
        self.e_leh, self.v_leh = np.asarray(e), np.asarray(v)
        self.gw.set_poles(self.e_leh, self.v_leh, beta=self.beta, thermal_tol=self.thermal_tol, thermal_floor=self.thermal_floor)

    def _iterate_thermal(self):
        t0 = time.time(); gw = self.gw; nk, nb = self.nk, self.nb
        thermal = gw.thermal
        # 1. screened interaction on the data set D (thermal) or on all gapless line nodes (empty window)
        wres, _ = gw.w_step(self.qminus)
        tW = time.time() - t0
        # 2. total Sigma_c at all fermionic nodes, linear mixing
        Sig = [gw.sigma(ik, wres, self.fz, qminus=self.qminus) for ik in range(nk)]
        if self.Sig_prev is not None:
            Sig = [self.mixing * s + (1 - self.mixing) * p for s, p in zip(Sig, self.Sig_prev)]
        dS = 0.0 if self.Sig_prev is None else max(np.abs(s - p)[self.fmask].max() for s, p in zip(Sig, self.Sig_prev))
        self.Sig_prev = Sig
        if self.scf_density == 'matsubara':                                  # 2b. tau-leg Sigma_c(i w_n), same mixing
            r = gw.sigma_tau_leg(wres, self.iw_h, qminus=self.qminus, tau_kw=self.hyb_tau_kw)
            Siw = [r['sigma'], r['S1'], r['S2']]
            if self.Siw_prev is not None:
                Siw = [self.mixing * a + (1 - self.mixing) * b for a, b in zip(Siw, self.Siw_prev)]
            self.Siw_prev = Siw
        tS = time.time() - t0 - tW
        # 3. closure: total fit on the two-sided gapless basis at the unmasked nodes, moments at the floored omega_p, Lehmann G
        wp = wp_thermal(self.wp, self.zeta_T, self.wp_floor) if thermal else self.wp
        mask = self.fmask if thermal else None
        Hstat = self.H0 + self.F - self.mu * np.eye(nb)[None]
        e_all, v_all, info_all = [], [], []
        for ik in range(nk):
            w, g = fit_sigma_total(self.bt, self.fz, Sig[ik], mask)
            e, v, info = lehmann_from_sigma(Hstat[ik], w, g, wp, self.K, tol_gram=self.tol_gram)
            e_all.append(e); v_all.append(v); info_all.append(info)
        M = max(len(e) for e in e_all)
        e_arr = np.array([np.pad(e, (0, M - len(e)), constant_values=1e6) for e in e_all])
        v_arr = np.array([np.pad(v, ((0, 0), (0, M - v.shape[1]))) for v in v_all])
        if self.scf_density == 'matsubara':
            return self._finish_matsubara(t0, tW, tS, dS, wp, Hstat, e_arr, v_arr, info_all, thermal)
        # 4. chemical potential, re-centring, thermal lists, static part
        dmu, rule = self._mu_shift(e_arr, v_arr)
        self.mu += dmu; e_arr = e_arr - dmu; gw.mu = self.mu
        N_mu = electron_count_T(e_arr, v_arr, self.beta, 0.0, self.k_weight)
        self._set_poles_T(e_arr, v_arr)
        Dm = gw.density_matrix(); nel = 2 * np.einsum('kii->', Dm).real / nk
        self.F = gw.hartree_exchange(Dm)
        _, e_homo, e_lumo, _ = chemical_potential(e_arr, v_arr, nk, self.nelec, self.k_weight)
        rec = dict(dSigma=dS, mu=self.mu, dmu=dmu, mu_rule=rule, N_mu=N_mu, nelec=nel, gap_eV=(e_lumo - e_homo) * 27.211386,
                   thermal=thermal, window=gw.window_counts().tolist(), wp=wp, nD=len(gw.bos_nodes()),
                   rank_b=(gw.bos_w.r if thermal else gw.bos.r), npoles=[i['npoles'] for i in info_all],
                   heldout=[i['heldout_err'] for i in info_all], tW=tW, tS=tS, ttot=time.time() - t0)
        self.history.append(rec)
        if self.verbose:
            print(f"iter {len(self.history)} (beta {self.beta:g}, {'thermal' if thermal else 'empty window'}): dSigma {dS:.2e}  mu {self.mu:.9f} "
                  f"(dmu {dmu*27.2114:+.4f} eV, {rule})  N(mu) {N_mu:.12f}  nelec(D) {nel:.8f}  gap {rec['gap_eV']:.4f} eV  window/k {rec['window']}  "
                  f"|D| {rec['nD']} rank_b {rec['rank_b']}  wp {wp:.3f}  npoles {min(rec['npoles'])}-{max(rec['npoles'])}  "
                  f"held-out {max(rec['heldout']):.1e}  [W {tW:.0f}s, Sigma {tS:.0f}s, total {rec['ttot']:.0f}s]", flush=True)
        return rec

    def _finish_matsubara(self, t0, tW, tS, dS, wp, Hstat, e_arr, v_arr, info_all, thermal):
        """Step 4' of scf_density = "matsubara": mu, D, N from the Matsubara Dyson equation with the tau-leg Sigma_c(i w_n); the
        closure's Lehmann poles re-centred at the new mu; the closure's own D / N / mu choice logged."""
        gw, nk = self.gw, self.nk
        t1 = time.time()
        Siw, S1, S2 = self.Siw_prev
        dmu, D, N_h, hinfo = _ft.density_matsubara(Hstat, Siw, self.iw_h, self.beta, S1, S2, nelec=self.nelec, k_weight=self.k_weight)
        dmu_cl, rule_cl = self._mu_shift(e_arr, v_arr)                        # what the closure alone would have chosen
        self.mu += dmu; e_arr = e_arr - dmu; gw.mu = self.mu
        N_cl = electron_count_T(e_arr, v_arr, self.beta, 0.0, self.k_weight)   # closure count at the hybrid mu
        self._set_poles_T(e_arr, v_arr)
        Dcl = gw.density_matrix()
        wk = np.full(nk, 1.0 / nk) if self.k_weight is None else np.asarray(self.k_weight) / np.sum(self.k_weight)
        nel_cl = float(2 * np.sum(wk * np.einsum('kii->k', Dcl).real))
        self.F = gw.hartree_exchange(D)
        _, e_homo, e_lumo, _ = chemical_potential(e_arr, v_arr, nk, self.nelec, self.k_weight)
        rec = dict(dSigma=dS, mu=self.mu, dmu=dmu, mu_rule='matsubara', N_mu=N_h, N_trace=hinfo['N_trace'],
                   nelec=float(2 * np.sum(wk * np.einsum('kii->k', D).real)), gap_eV=(e_lumo - e_homo) * 27.211386,
                   thermal=thermal, window=gw.window_counts().tolist(), wp=wp, nD=len(gw.bos_nodes()),
                   rank_b=(gw.bos_w.r if thermal else gw.bos.r), npoles=[i['npoles'] for i in info_all],
                   heldout=[i['heldout_err'] for i in info_all], nfreq=len(self.iw_h),
                   D_cl_err=float(np.abs(Dcl - D).max()), N_closure=N_cl, dN_closure=N_cl - self.nelec, nelec_closure_D=nel_cl,
                   dmu_closure=dmu_cl - dmu, rule_closure=rule_cl, tail_max=hinfo['tail_max'],
                   tW=tW, tS=tS, tH=time.time() - t1, ttot=time.time() - t0)
        self.history.append(rec)
        if self.verbose:
            print(f"iter {len(self.history)} (beta {self.beta:g}, matsubara, {'thermal' if thermal else 'empty window'}): dSigma {dS:.2e}  "
                  f"mu {self.mu:.9f} (dmu {dmu*27.2114:+.4f} eV)  N(mu) {N_h:.12f}  | closure: N {N_cl:.8f} (dN {N_cl - self.nelec:+.1e}), "
                  f"max|D_cl - D| {rec['D_cl_err']:.1e}, own mu {dmu_cl - dmu:+.2e} Ha ({rule_cl})  | gap {rec['gap_eV']:.4f} eV  "
                  f"window/k {rec['window']}  |D| {rec['nD']} rank_b {rec['rank_b']}  wp {wp:.3f}  held-out {max(rec['heldout']):.1e}  "
                  f"[W {tW:.0f}s, Sigma {tS:.0f}s, hybrid {rec['tH']:.0f}s, total {rec['ttot']:.0f}s]", flush=True)
        return rec

    def sigma_sectors(self, ik, wres):
        """(Sigma^>, Sigma^<) at the fermionic nodes for k."""
        gw, bos, fz, nk, Np, nb = self.gw, self.bos, self.fz, self.nk, self.gw.Np, self.nb
        out = []
        Xk = self.X[ik]
        for sector, ray in (('>', gw.ray_p), ('<', gw.ray_h)):
            F = ray.transform_matrix(fz); S = np.zeros((len(fz), nb, nb), complex)
            for i0 in range(0, len(ray), gw.t_chunk):
                t = ray.t[i0:i0 + gw.t_chunk]; acc = np.zeros((len(t), Np, Np), complex); Ew = bos.time_exponentials(t, sector)
                for iq in range(nk):
                    wq = wres[iq] if sector == '>' else np.transpose(wres[iq], (0, 2, 1))
                    acc += gw.gtilde(self.qk[iq, ik], t, sector) * (Ew @ wq.reshape(wq.shape[0], -1)).reshape(Ew.shape[0], wq.shape[1], wq.shape[2])
                acc *= (1.0 if sector == '>' else -1.0) / nk
                S += (F[:, i0:i0 + gw.t_chunk] @ ((Xk.conj().T @ acc) @ Xk).reshape(acc.shape[0], -1)).reshape(-1, Xk.shape[1], Xk.shape[1])
            out.append(S)
        return out

    def iterate(self):
        if self.beta is not None:
            return self._iterate_thermal()
        t0 = time.time(); gw = self.gw; nk, nb = self.nk, self.nb
        # 1. screened interaction
        wres = []
        for iq in range(nk):
            w, _ = gw.screened_interaction(iq); wres.append(w)
        tW = time.time() - t0
        # 2. self-energy per k and sector, mixing
        Sig = [self.sigma_sectors(ik, wres) for ik in range(nk)]
        if self.Sig_prev is not None:
            Sig = [[self.mixing * s + (1 - self.mixing) * p for s, p in zip(Sk, Pk)] for Sk, Pk in zip(Sig, self.Sig_prev)]
        dS = 0.0 if self.Sig_prev is None else max(np.abs(s[0] + s[1] - p[0] - p[1]).max() for s, p in zip(Sig, self.Sig_prev))
        self.Sig_prev = Sig
        tS = time.time() - t0 - tW
        # 3. closure: moments -> upfold -> Lehmann G per k (mu-relative static Hamiltonian H0 + F - mu)
        Hstat = self.H0 + self.F - self.mu * np.eye(nb)[None]
        e_all, v_all, info_all = [], [], []
        for ik in range(nk):
            w, g = fit_sigma_sectors(self.bp, self.bh, self.fz, Sig[ik][0], Sig[ik][1])
            e, v, info = lehmann_from_sigma(Hstat[ik], w, g, self.wp, self.K, tol_gram=self.tol_gram)
            e_all.append(e); v_all.append(v); info_all.append(info)
        # new mu (T=0 filling) and re-centring
        M = max(len(e) for e in e_all)
        e_arr = np.array([np.pad(e, (0, M - len(e)), constant_values=1e6) for e in e_all])
        v_arr = np.array([np.pad(v, ((0, 0), (0, M - v.shape[1]))) for v in v_all])
        dmu, e_homo, e_lumo, nfill = chemical_potential(e_arr, v_arr, nk, self.nelec, self.k_weight)
        self.mu += dmu; e_arr = e_arr - dmu; self.gw.mu = self.mu
        wk_ = np.full(nk, 1.0 / nk) if self.k_weight is None else np.asarray(self.k_weight) / np.sum(self.k_weight)
        nel_exact = float(sum(2.0 * wk_[k] * (np.abs(v_arr[k][:, e_arr[k] < 0]) ** 2).sum() for k in range(nk)))
        # compressed per-sector G and the new static part
        w_g, c_g, dropped = [], [], 0.0
        for ik in range(nk):
            wk, ck_, dr = compress_sectors(self.gp, self.gh, self.fz, e_arr[ik], v_arr[ik]); w_g.append(wk); c_g.append(ck_); dropped = max(dropped, dr)
        gw.set_poles(np.array(w_g), coef=np.array(c_g))
        Dm = gw.density_matrix(); nel = 2 * np.einsum('kii->', Dm).real / nk
        self.F = gw.hartree_exchange(Dm)
        rec = dict(dSigma=dS, mu=self.mu, dmu=dmu, gap_eV=(e_lumo - e_homo) * 27.211386, nelec=nel, nelec_exact=nel_exact, dropped_weight=dropped,
                   npoles=[i['npoles'] for i in info_all], heldout=[i['heldout_err'] for i in info_all], tW=tW, tS=tS, ttot=time.time() - t0)
        self.history.append(rec)
        if self.verbose:
            print(f"iter {len(self.history)}: dSigma {dS:.2e}  mu {self.mu:.6f} (dmu {dmu*27.2114:+.4f} eV)  QP gap {rec['gap_eV']:.4f} eV  "
                  f"nelec {nel:.6f} (Lehmann {nel_exact:.6f}, dropped weight {dropped:.1e})  npoles {min(rec['npoles'])}-{max(rec['npoles'])}  "
                  f"held-out {max(rec['heldout']):.1e}  [W {tW:.0f}s, Sigma {tS:.0f}s, total {rec['ttot']:.0f}s]", flush=True)
        return rec

    def sigma_total(self, ik):
        if self.beta is not None:
            return self.Sig_prev[ik]
        return self.Sig_prev[ik][0] + self.Sig_prev[ik][1]

    # ---------------------------------------------------------------- checkpoint / resume (preemptible queues)
    def save_state(self, fname):
        e, coef = self.gw.poles
        np.savez(fname, e=e, coef=coef, F=self.F, mu=self.mu, niter=len(self.history),
                 Sig_p=np.array([s[0] for s in self.Sig_prev]) if self.Sig_prev is not None else np.zeros(0),
                 Sig_h=np.array([s[1] for s in self.Sig_prev]) if self.Sig_prev is not None else np.zeros(0),
                 history=np.array(self.history, dtype=object), fz=self.fz)

    def load_state(self, fname):
        z = np.load(fname, allow_pickle=True)
        assert np.allclose(z['fz'], self.fz), "line nodes differ from the saved state (theta/eps/gaps changed)"
        self.mu = float(z['mu']); self.gw.mu = self.mu
        self.gw.set_poles(z['e'], coef=z['coef']); self.F = z['F']
        self.history = list(z['history'])
        self.Sig_prev = [[z['Sig_p'][k], z['Sig_h'][k]] for k in range(self.nk)] if z['Sig_p'].size else None
        if self.verbose: print(f"resumed from {fname}: {len(self.history)} iterations done, mu {self.mu:.6f}", flush=True)
