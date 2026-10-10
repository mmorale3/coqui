"""GW kernels on the tilted line in the THC/ISDF basis (numpy prototype of option D).

All energies are measured from the line centre mu. Pole data for G: per k, energies e_m (mu-relative) and vectors v_m
(nb,) with G(k, zeta) = sum_m v_m v_m^dagger /(zeta - e_m); sector by the sign of e_m.

Time ray (particle sector): t = s e^{-i theta_t};  G~^>(k,t)_PQ = [X(k) (sum_{m>} v_m v_m^dag e^{-i e_m t}) X(k)^dag]_PQ.
Polarization (sign/normalization validated against the Casida transition sum, V1):
    Pi^>(q,t)_PQ = (2/Nk) sum_k G~^>(k,t)_PQ * conj(G~^<(k-q,t))_PQ,   G~^<(k,t) = X(k)(sum_{m<} v_m v_m^dag e^{-i e_m t})X(k)^dag
    Pi^>(q,zeta) = -i int_0^inf dt e^{i zeta t} Pi^>(q,t);   Pi(zeta) = Pi^>(zeta) + conj(Pi^>(-conj(zeta)))  (elementwise)
Screened interaction: W(q,zeta) = ([1 - Z Pi(zeta)]^-1 - 1) Z at the bosonic line nodes, refit with BosonicLineBasis
    W_PQ(q,zeta) = sum_j [ w_j(q)_PQ/(zeta - nu_j) - w_j(-q)_QP/(zeta + nu_j) ],
    W^>(q,t) = sum_j w_j(q) e^{-i nu_j t},  W^<(q,t) = -sum_j w_j(-q)^T e^{+i nu_j t}
    (notes section 3.3, corrected 2026-10-04: W(q,-zeta) = W(-q,zeta)^T; -q = qminus[q]). NOTE: screened_interaction(iq)
    without W_minus and sigma(...) without qminus use the per-q form w_j(q)^T, valid ONLY for self-inverse q (q = -q mod G,
    every q of the 2x2x2 / 2x1x1 meshes this prototype was run on); for q != -q pass W_minus = W(-q) and qminus.
Self-energy per sector (contracted to orbitals immediately):
    Sigma~^>(k,t) = (1/Nk) sum_q G~^>(k-q,t) * W^>(q,t);   Sigma^>_ab(k,zeta) = -i int dt e^{i zeta t} [X(k)^dag Sigma~^>(k,t) X(k)]_ab
    Sigma~^<(k,t) = (1/Nk) sum_q G~^<(k-q,t) * W^<(q,t)  on the hole ray t = s e^{+i theta_t}.
Static part: F = V_H[Dm] + Sigma_x[Dm] with the THC Z (ignore_g0 heads are inside Z), spin-restricted (Dm per spin).

Finite temperature (S8b, notes section 11; set_poles(..., beta=B)): energies stay mu-relative, f(e) = 1/(e^{B e}+1),
n(nu) = 1/(e^{B nu}-1), window E_T = ln(1/thermal_tol)/B. Thermal sector lists (Eq. fT_sectors): particle = {e > E_T, weight 1}
+ {|e| <= E_T, weight 1-f}, hole = {e < -E_T, weight 1} + {|e| <= E_T, weight f} (residues scaled, so a window pole enters
BOTH sectors); the rays are truncated at the beta guard S_T = B/sin(theta_t) (Eq. fT_guard) and the GL sum on [0, S_T] is the
truncated transform T_S (Eq. fT_TS). Pi and Sigma use the same formulas with the lists; W(t) carries Bose weights n_j on the
window poles nu_j <= E_T (Eq. fT_W, both terms, both legs); the bosonic fit uses only the nodes with rho B |zeta| >= c_zeta
(Eq. fT_floor, rho = sin(theta - theta_t)/sin(theta_t)). Thermal mode is active iff some window is non-empty; otherwise
(beta None or an empty window) every method runs the T = 0 code unchanged (bitwise).
"""
import numpy as np
from .timeray import TimeRay
from .line_dlr import LineBasis, BosonicLineBasis


def fermi(e, beta):
    """f(e) = 1/(e^{beta e} + 1), overflow-free."""
    return 0.5 * (1.0 - np.tanh(0.5 * beta * np.asarray(e, float)))


def bose(nu, beta):
    """n(nu) = 1/(e^{beta nu} - 1) for nu > 0, overflow-free (0 where beta nu > 700)."""
    x = beta * np.asarray(nu, float)
    with np.errstate(over='ignore'):
        return np.where(x > 700.0, 0.0, 1.0 / np.expm1(np.minimum(x, 700.0)))


def thermal_window(beta, thermal_tol=1e-8):
    """E_T = ln(1/thermal_tol)/beta (Ha)."""
    return np.log(1.0 / thermal_tol) / beta


def thermal_rho(theta, theta_t):
    """rho = sin(theta - theta_t)/sin(theta_t): |e^{i zeta t_S}| = e^{-rho beta |zeta|} at the guard (Eq. fT_floor)."""
    return np.sin(theta - theta_t) / np.sin(theta_t)


def node_floor_mask(zeta, beta, theta, theta_t, c_zeta=30.0):
    """True for the line nodes kept in the fits at finite T: rho beta |zeta| >= c_zeta (zeta mu-relative)."""
    return thermal_rho(theta, theta_t) * beta * np.abs(np.asarray(zeta)) >= c_zeta


class LineGW:
    def __init__(self, X, Z, qk_to_k2, nk, mu, theta, theta_t, bos_basis, ferm_zeta, t_chunk=8, ray_decades=36.0, ray_kw=None):
        """X: (nk, Np, nb) THC collocation; Z: (nq, Np, Np); qk_to_k2[iq, ik] = index of k - q; mu: absolute centre.
        bos_basis: BosonicLineBasis (mu-relative); ferm_zeta: mu-relative fermionic line nodes for Sigma/G.
        The time rays are built per call from the current pole spectrum (smallest |e_m| sets s_max)."""
        self.X, self.Z, self.qk, self.nk, self.mu = X, Z, qk_to_k2, nk, mu
        self.Np, self.nb = X.shape[1], X.shape[2]
        self.theta, self.theta_t = theta, theta_t
        self.bos, self.fz = bos_basis, np.asarray(ferm_zeta, complex)
        self.t_chunk, self.ray_decades, self.ray_kw = t_chunk, ray_decades, (ray_kw or {})
        self.poles = None

    # ---------------------------------------------------------------- pole data
    def set_poles(self, e, v=None, coef=None, beta=None, thermal_tol=1e-8, thermal_floor=30.0, ray_kw_T=None):
        """G(k, zeta) = sum_m coef_m /(zeta - e_m) with real mu-relative energies e (nk, M). Either v (nk, nb, M) column
        vectors (coef_m = v_m v_m^dagger, Lehmann form) or general matrix coefficients coef (nk, M, nb, nb) (compressed
        real-pole fit per sector). Sector = sign of e_m.
        beta (S8b): None = T = 0. Otherwise the thermal sector lists of Eq. fT_sectors with the window E_T =
        ln(1/thermal_tol)/beta; thermal mode (self.thermal) only if some |e_m| <= E_T, else the T = 0 path below runs
        unchanged. thermal_floor = c_zeta of the node floor; ray_kw_T: TimeRay options of the guarded rays (default
        self.ray_kw with hmax = pi/(E_T cos theta_t) = one period of the fastest window-window pair, |E| = 2 E_T, per panel)."""
        e = np.asarray(e)
        if coef is None:
            v = np.asarray(v)
            coef = np.einsum('kim,kjm->kmij', v, v.conj())
        self.poles = (e, np.asarray(coef))
        self.beta, self.thermal_tol, self.thermal_floor, self.thermal = beta, thermal_tol, thermal_floor, False
        if beta is not None:
            self.E_T = thermal_window(beta, thermal_tol)
            self.win = np.abs(e) <= self.E_T
            self.thermal = bool(self.win.any())
        if self.thermal:
            self._set_thermal_lists(ray_kw_T)
            return
        emin = min(np.abs(e[e > 0]).min(), np.abs(e[e < 0]).min())
        self.ray_p = TimeRay.for_spectrum(self.theta_t, emin, decades=self.ray_decades, sector='>', **self.ray_kw)
        self.ray_h = TimeRay.for_spectrum(self.theta_t, emin, decades=self.ray_decades, sector='<', **self.ray_kw)

    def _set_thermal_lists(self, ray_kw_T=None):
        """Eq. fT_sectors per k: self.lists['>' / '<'][ik] = (energies, scaled coefficients); guarded rays at S_T."""
        e, coef = self.poles
        beta, E_T = self.beta, self.E_T
        self.lists = {'>': [], '<': []}
        for ik in range(e.shape[0]):
            ek, w = e[ik], self.win[ik]
            f = fermi(ek, beta)
            mp = (ek > E_T) | w
            mh = (ek < -E_T) | w
            wp = np.where(w, 1.0 - f, 1.0)[mp]
            wh = np.where(w, f, 1.0)[mh]
            self.lists['>'].append((ek[mp], coef[ik][mp] * wp[:, None, None]))
            self.lists['<'].append((ek[mh], coef[ik][mh] * wh[:, None, None]))
        self.S_T = beta / np.sin(self.theta_t)
        self.zeta_T = self.thermal_floor / (thermal_rho(self.theta, self.theta_t) * beta)
        kw = dict(self.ray_kw)
        kw.setdefault('hmax', np.pi / (self.E_T * np.cos(self.theta_t)))
        kw.update(ray_kw_T or {})
        self.ray_p = TimeRay.guarded(self.theta_t, beta, sector='>', **kw)
        self.ray_h = TimeRay.guarded(self.theta_t, beta, sector='<', **kw)

    def window_counts(self):
        """Number of window poles per k (thermal mode), zeros otherwise."""
        return self.win.sum(1) if self.thermal else np.zeros(self.poles[0].shape[0], int)

    def bos_nodes(self):
        """Bosonic nodes used for Dyson + fit: all basis nodes at T = 0, the unmasked ones (rho beta |zeta| >= c_zeta) in
        thermal mode."""
        z = self.bos.zeta
        return z if not self.thermal else z[node_floor_mask(z, self.beta, self.theta, self.theta_t, self.thermal_floor)]

    def bose_weights(self, nu):
        """n_j on the window (nu_j <= E_T), 0 beyond (n < thermal_tol there) and at T = 0."""
        nu = np.asarray(nu, float)
        if not self.thermal: return np.zeros_like(nu)
        return np.where(nu <= self.E_T, bose(nu, self.beta), 0.0)

    @classmethod
    def poles_from_hamiltonian(cls, H, mu):
        """Diagonalize H (nk, nb, nb) -> (e - mu, v)."""
        e, v = np.linalg.eigh(H)
        return e - mu, v

    def gtilde(self, ik, t, sector):
        """G~(k, t) (nt, Np, Np) for the given sector on complex times t (nt,): X(k) [sum_m coef_m e^{-i e_m t}] X(k)^dag."""
        if self.thermal:
            el, cl = self.lists[sector][ik]
            ph = np.exp(-1j * el[None, :] * t[:, None])
            Gt = (ph @ cl.reshape(ph.shape[1], -1)).reshape(ph.shape[0], self.nb, self.nb)
        else:
            e, coef = self.poles
            m = e[ik] > 0 if sector == '>' else e[ik] < 0
            ph = np.exp(-1j * e[ik][m][None, :] * t[:, None])              # (nt, M)
            Gt = (ph @ coef[ik][m].reshape(ph.shape[1], -1)).reshape(ph.shape[0], self.nb, self.nb)                  # (nt, nb, nb)
        Xk = self.X[ik]
        return (Xk @ Gt) @ Xk.conj().T                                  # (nt, Np, Np): two GEMMs per t

    def g_line(self, ik, zeta, sector=None):
        """G(k, zeta) (nz, nb, nb) from the pole data (any complex zeta off the real axis)."""
        if self.thermal and sector is not None:
            el, cl = self.lists[sector][ik]
            return np.einsum('zm,mij->zij', 1.0 / (np.asarray(zeta, complex)[:, None] - el[None, :]), cl)
        e, coef = self.poles
        m = np.ones(e.shape[1], bool) if sector is None else (e[ik] > 0 if sector == '>' else e[ik] < 0)
        K = 1.0 / (np.asarray(zeta, complex)[:, None] - e[ik][m][None, :])
        return np.einsum('zm,mij->zij', K, coef[ik][m])

    # ---------------------------------------------------------------- polarization and W
    def polarization(self, iq, zeta=None):
        """Pi(q, zeta) (nz, Np, Np) at mu-relative points zeta (default: bosonic nodes).
        Both sectors are built explicitly on their rays (no q -> -q symmetry assumed), matching the Casida transition sum
        Pi = sum_t sg_t S_t S_t^H/(zeta - E_t) with S_t = sqrt(2/Nk) X(k,n) conj(X(k-q,m)):
          Pi^>(q,t)_PQ = +(2/Nk) sum_k [sum_{n occ(k)}   R~_n,PQ(k) e^{+i e_n t}] [sum_{m unocc(k-q)} conj(R~_m,PQ(k-q)) e^{-i e_m t}]   (particle ray)
          Pi^<(q,t)_PQ = -(2/Nk) sum_k [sum_{n unocc(k)} R~_n,PQ(k) e^{+i e_n t}] [sum_{m occ(k-q)}   conj(R~_m,PQ(k-q)) e^{-i e_m t}]   (hole ray)
        with sum_m R~_m,PQ e^{+i e_m t} = conj(G~(k, conj t))_QP and conj(R~_m,PQ) e^{-i e_m t} = G~(k, t)_QP (Hermitian residues),
        and Pi^{>/<}(zeta) = -i int_0^inf dt e^{i zeta t} Pi^{>/<}(t).
        Thermal mode: the same products with the thermal lists on the guarded rays (Eq. fT_pi up to the end-point terms of
        Eq. fT_floor); default nodes = the unmasked bosonic nodes."""
        zeta = self.bos_nodes() if zeta is None else np.asarray(zeta, complex)
        out = np.zeros((len(zeta), self.Np, self.Np), complex)
        for sector, ray, sign in (('>', self.ray_p, 1.0), ('<', self.ray_h, -1.0)):
            s_k, s_kmq = ('<', '>') if sector == '>' else ('>', '<')        # occupation of the state at k / at k-q
            F = ray.transform_matrix(zeta)
            for i0 in range(0, len(ray), self.t_chunk):
                t = ray.t[i0:i0 + self.t_chunk]
                acc = np.zeros((len(t), self.Np, self.Np), complex)
                for ik in range(self.nk):
                    A = np.conj(self.gtilde(ik, np.conj(t), s_k))                  # (nt, Np, Np): entries QP of sum R~ e^{+i e t}
                    B = self.gtilde(self.qk[iq, ik], t, s_kmq)                    # entries QP of sum conj(R~) e^{-i e t}
                    acc += np.transpose(A * B, (0, 2, 1))
                acc *= sign * 2.0 / self.nk
                out += (F[:, i0:i0 + self.t_chunk] @ acc.reshape(acc.shape[0], -1)).reshape(-1, acc.shape[1], acc.shape[2])
        return out

    def dyson_w(self, iq, Pi):
        """W(q, zeta_i) = ([1 - Z Pi]^-1 - 1) Z for each node; (nz, Np, Np)."""
        Z = self.Z[iq]; I = np.eye(self.Np)
        return np.array([np.linalg.solve(I - Z @ P, Z) - Z for P in Pi])

    def screened_interaction(self, iq, Pi=None, W_minus=None):
        """Residues w_j(q) (r, Np, Np) of the symmetric real-pole fit of W(q) on the bosonic nodes; also returns W at the nodes.
        W_minus: W(-q) at the nodes (required for q != -q, see the module docstring); None = self-inverse q.
        Thermal mode: Dyson and fit on the unmasked nodes bos_nodes() only (Pi, W_minus given there)."""
        z = self.bos_nodes()
        if Pi is None: Pi = self.polarization(iq, z)
        W = self.dyson_w(iq, Pi)
        return self.bos.fit(z, W, W_minus=W_minus), W

    # ---------------------------------------------------------------- self-energy
    def sigma(self, ik, wres, zeta=None, qminus=None, nu=None):
        """Sigma_c(k, zeta)_ab (nz, nb, nb), zeta mu-relative (default fermionic nodes); wres: list over q of residues (r, Np, Np).
        qminus: index of -q per q (the hole sector uses w(-q)^T); None = every q self-inverse.
        nu: optional list over q of the positive pole energies of wres[q] (default: self.bos.nu for every q).
        Thermal mode: Eq. fT_W on the guarded rays,
          W^>(q,t) = sum_j (1+n_j) w_j(q) e^{-i nu_j t} + sum_win n_j w_j(-q)^T e^{+i nu_j t},
          W^<(q,t) = -sum_j (1+n_j) w_j(-q)^T e^{+i nu_j t} - sum_win n_j w_j(q) e^{-i nu_j t},
        with the thermal G lists (Eq. fT_sigma for the analytic result)."""
        if self.thermal or nu is not None:
            return self._sigma_general(ik, wres, zeta, qminus, nu)
        zeta = self.fz if zeta is None else np.asarray(zeta, complex)
        out = np.zeros((len(zeta), self.nb, self.nb), complex)
        Xk = self.X[ik]
        for sector, ray in (('>', self.ray_p), ('<', self.ray_h)):
            F = ray.transform_matrix(zeta)
            for i0 in range(0, len(ray), self.t_chunk):
                t = ray.t[i0:i0 + self.t_chunk]
                acc = np.zeros((len(t), self.Np, self.Np), complex)
                Ew = self.bos.time_exponentials(t, sector)              # (nt, r)
                for iq in range(self.nk):
                    w = wres[iq] if sector == '>' else np.transpose(wres[iq if qminus is None else qminus[iq]], (0, 2, 1))
                    Wt = (Ew @ w.reshape(w.shape[0], -1)).reshape(Ew.shape[0], w.shape[1], w.shape[2])
                    acc += self.gtilde(self.qk[iq, ik], t, sector) * Wt
                acc *= (1.0 if sector == '>' else -1.0) / self.nk              # T=0 factor [theta(nu) - theta(-eps)] = -1 in the hole sector
                S_ab = (Xk.conj().T @ acc) @ Xk                           # (nt, nb, nb)
                out += (F[:, i0:i0 + self.t_chunk] @ S_ab.reshape(S_ab.shape[0], -1)).reshape(-1, S_ab.shape[1], S_ab.shape[2])
        return out

    def _sigma_general(self, ik, wres, zeta, qminus, nu):
        """sigma() with per-q pole energies and/or the thermal W(t) of Eq. fT_W (n_j = 0 at T = 0)."""
        zeta = self.fz if zeta is None else np.asarray(zeta, complex)
        out = np.zeros((len(zeta), self.nb, self.nb), complex)
        Xk = self.X[ik]
        qm = (lambda iq: iq) if qminus is None else (lambda iq: qminus[iq])
        nu_of = (lambda iq: self.bos.nu) if nu is None else (lambda iq: np.asarray(nu[iq], float))
        def wt(w, E):                                                    # sum_j E_j(t) w_j  -> (nt, Np, Np)
            return (E @ w.reshape(w.shape[0], -1)).reshape(E.shape[0], w.shape[1], w.shape[2])
        for sector, ray in (('>', self.ray_p), ('<', self.ray_h)):
            F = ray.transform_matrix(zeta)
            for i0 in range(0, len(ray), self.t_chunk):
                t = ray.t[i0:i0 + self.t_chunk]
                acc = np.zeros((len(t), self.Np, self.Np), complex)
                for iq in range(self.nk):
                    wq, wmT = wres[iq], np.transpose(wres[qm(iq)], (0, 2, 1))
                    nq, nmq = nu_of(iq), nu_of(qm(iq))
                    bq, bmq = self.bose_weights(nq), self.bose_weights(nmq)
                    if sector == '>':
                        Wt = wt(wq, (1.0 + bq)[None, :] * np.exp(-1j * nq[None, :] * t[:, None]))
                        if bmq.any():
                            j = bmq > 0
                            Wt += wt(wmT[j], bmq[j][None, :] * np.exp(1j * nmq[j][None, :] * t[:, None]))
                    else:
                        Wt = wt(wmT, -(1.0 + bmq)[None, :] * np.exp(1j * nmq[None, :] * t[:, None]))
                        if bq.any():
                            j = bq > 0
                            Wt += wt(wq[j], -bq[j][None, :] * np.exp(-1j * nq[j][None, :] * t[:, None]))
                    acc += self.gtilde(self.qk[iq, ik], t, sector) * Wt
                acc *= (1.0 if sector == '>' else -1.0) / self.nk
                S_ab = (Xk.conj().T @ acc) @ Xk
                out += (F[:, i0:i0 + self.t_chunk] @ S_ab.reshape(S_ab.shape[0], -1)).reshape(-1, S_ab.shape[1], S_ab.shape[2])
        return out

    # ---------------------------------------------------------------- static part
    def density_matrix(self):
        """Dm(k) = sum_{m<} coef_m (per spin, T=0); thermal mode: the density of the thermal hole list, sum_m f_m coef_m
        (far holes with weight 1, error <= thermal_tol per pole)."""
        if self.thermal:
            return np.array([self.lists['<'][ik][1].sum(0) for ik in range(self.nk)])
        e, coef = self.poles
        return np.array([coef[ik][e[ik] < 0].sum(0) for ik in range(self.nk)])

    def hartree_exchange(self, Dm, iq0=0):
        """F = V_H + Sigma_x (nk, nb, nb) in the THC basis; Dm per spin (closed shell: total density = 2 Dm).
        V_H,ab(k) = (2/Nk) sum_k' X_P(k,a)^* X_P(k,b) Z_PQ(0) [X(k') Dm(k') X(k')^dag]_QQ ;
        Sigma_x,ab(k) = -(1/Nk) sum_q X_P(k,a)^* [X(k-q) Dm(k-q) X(k-q)^dag]_PQ Z_PQ(q) X_Q(k,b)."""
        Dt = np.array([self.X[k] @ Dm[k] @ self.X[k].conj().T for k in range(self.nk)])        # (nk, Np, Np)
        rho = np.einsum('kqq->q', Dt) * (2.0 / self.nk)                                           # aux density (total)
        vh = self.Z[iq0] @ rho                                                                   # (Np,)
        F = np.zeros((self.nk, self.nb, self.nb), complex)
        for ik in range(self.nk):
            Xk = self.X[ik]
            F[ik] += (Xk.conj() * vh[:, None]).T @ Xk
            Sx = np.zeros((self.Np, self.Np), complex)
            for iq in range(self.nk):
                Sx += Dt[self.qk[iq, ik]] * self.Z[iq]
            F[ik] -= (Xk.conj().T @ Sx @ Xk) / self.nk
        return F
