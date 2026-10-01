"""Self-consistent GW on the tilted line (option D), numpy prototype. One object = one system; `iterate()` does one loop.

Loop (pole data in -> pole data out), all energies mu-relative, mu absolute in self.mu:
  1. W(q): Pi(q) on the bosonic nodes from the current G poles (time ray), Dyson, symmetric real-pole refit -> wres[q]
  2. Sigma^>(k), Sigma^<(k) on the fermionic nodes (per-sector time rays), linear mixing with the previous iteration
  3. per k: real-pole fit of each sector -> Cayley moments (total measure) -> block upfolding -> upfolded Hamiltonian with
     H_stat = H0 + F[Dm] - mu -> Lehmann G (positive) ; new mu from T=0 filling ; re-centre ; compressed per-sector fit of G
  4. convergence: max |Delta Sigma| at the nodes, |Delta mu|, gap
Diagnostics kept per iteration in self.history."""
import numpy as np, time
from .line_dlr import LineBasis, BosonicLineBasis
from .thc_gw import LineGW
from .closure import fit_sigma_sectors, lehmann_from_sigma, chemical_potential, compress_sectors


class LineSCGW:
    def __init__(self, X, Z, qk_to_k2, nk, nelec, H0, mu, theta=np.deg2rad(20), eps=1e-8, lam=3.0, bos_gap=0.02,
                 sig_gap=(0.03, 0.03), g_gap=(0.01, 0.01), wp=0.11, K=16, tol_gram=None, mixing=0.5, verbose=True, k_weight=None):
        self.X, self.Z, self.qk, self.nk, self.nelec, self.H0 = X, Z, qk_to_k2, nk, nelec, H0
        self.nb = X.shape[2]; self.mu = mu; self.theta, self.eps, self.lam = theta, eps, lam
        self.wp, self.K, self.mixing, self.verbose = wp, K, mixing, verbose
        self.tol_gram = tol_gram if tol_gram is not None else 10 * eps
        self.k_weight = k_weight
        self.bos = BosonicLineBasis(theta, lam=lam, eps=eps, gap=bos_gap)
        self.bp = LineBasis(theta, lam=lam, eps=eps, gap=(lam, sig_gap[1])); self.bh = LineBasis(theta, lam=lam, eps=eps, gap=(sig_gap[0], lam))
        self.gp = LineBasis(theta, lam=lam, eps=eps, gap=(lam, g_gap[1])); self.gh = LineBasis(theta, lam=lam, eps=eps, gap=(g_gap[0], lam))
        self.fz = np.unique(np.concatenate([self.bp.zeta, self.bh.zeta, self.gp.zeta, self.gh.zeta]))
        self.gw = LineGW(X, Z, qk_to_k2, nk, mu, theta, theta / 2, self.bos, self.fz)
        self.Sig_prev = None; self.history = []
        if verbose: print(f"LineSCGW: {self.bos}\n  Sigma bases {self.bp.r}+{self.bh.r}, G bases {self.gp.r}+{self.gh.r}, fermionic nodes {len(self.fz)}", flush=True)

    def start_from_hamiltonian(self, H):
        """Initial poles from a one-body Hamiltonian H (nk, nb, nb) (e.g. KS or H0 + F)."""
        e, v = LineGW.poles_from_hamiltonian(H, self.mu)
        self.gw.set_poles(e, v)
        self.F = self.gw.hartree_exchange(self.gw.density_matrix())

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
                    acc += gw.gtilde(self.qk[iq, ik], t, sector) * np.einsum('tj,jpq->tpq', Ew, wq)
                acc *= (1.0 if sector == '>' else -1.0) / nk
                S += np.einsum('zt,tab->zab', F[:, i0:i0 + gw.t_chunk], (Xk.conj().T @ acc) @ Xk)
            out.append(S)
        return out

    def iterate(self):
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
        # compressed per-sector G and the new static part
        w_g, c_g = [], []
        for ik in range(nk):
            wk, ck_ = compress_sectors(self.gp, self.gh, self.fz, e_arr[ik], v_arr[ik]); w_g.append(wk); c_g.append(ck_)
        gw.set_poles(np.array(w_g), coef=np.array(c_g))
        Dm = gw.density_matrix(); nel = 2 * np.einsum('kii->', Dm).real / nk
        self.F = gw.hartree_exchange(Dm)
        rec = dict(dSigma=dS, mu=self.mu, dmu=dmu, gap_eV=(e_lumo - e_homo) * 27.211386, nelec=nel, npoles=[i['npoles'] for i in info_all],
                   heldout=[i['heldout_err'] for i in info_all], tW=tW, tS=tS, ttot=time.time() - t0)
        self.history.append(rec)
        if self.verbose:
            print(f"iter {len(self.history)}: dSigma {dS:.2e}  mu {self.mu:.6f} (dmu {dmu*27.2114:+.4f} eV)  gap(poles) {rec['gap_eV']:.4f} eV  "
                  f"nelec {nel:.6f}  npoles {min(rec['npoles'])}-{max(rec['npoles'])}  held-out {max(rec['heldout']):.1e}  [W {tW:.0f}s, Sigma {tS:.0f}s, total {rec['ttot']:.0f}s]", flush=True)
        return rec

    def sigma_total(self, ik):
        return self.Sig_prev[ik][0] + self.Sig_prev[ik][1]
