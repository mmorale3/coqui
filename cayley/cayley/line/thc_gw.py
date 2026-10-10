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
W step at finite T (S8b redesign, notes section 11.5 sec:fT_W; parameters in self.wstep, defaults WSTEP_DEFAULTS):
  bosonic data set D = bos_data() = unmasked line nodes of self.bos (kind 0) U wedge band (kind 1: band_heights x band_x points,
  heights v = y cos(theta_t) - |x| sin(theta_t) log-spaced in [d0, band_top zeta_T sin(theta)], d0 = c_band sin(theta_t)/beta,
  c_band = thermal_floor) U {i nu_n, n = 1..ceil(mats_factor zeta_T beta/2 pi)} (kind 2) U {nu_0 = 0} (kind 3). Pi at kinds 0-2
  from the guarded ray products (extra transform columns), Pi(q, 0) in the DYNAMIC convention from the tau leg pi_tau_leg()
  (imaginary-time products with all poles weighted f e^{e tau} / (1-f) e^{-e tau}, bounded on [0, beta], minus the exactly
  degenerate pairs). Basis self.bos_w = BosonicLineBasis.from_data(D) (eps_b), W by Dyson at every point of D, residues by the
  decoupled odd/even fit (fit_split, cut_odd / cut_even); w(q) and w(-q) from ONE joint solve (w_step()). sigma() in thermal
  mode uses the poles of self.bos_w unless nu is given.
"""
import numpy as np
from .timeray import TimeRay, tau_grid
from .line_dlr import LineBasis, BosonicLineBasis

WSTEP_DEFAULTS = dict(
    band_heights=8, band_x=21,     # wedge band B: heights x points per height (168 points)
    band_c=None,                   # d0 = band_c sin(theta_t)/beta at the band bottom; None = thermal_floor (c_zeta)
    band_top=4.0,                  # top height / x extent: band_top * zeta_T * sin(theta)
    mats_factor=4.0,               # N_M = ceil(mats_factor zeta_T beta/(2 pi)) Matsubara points i nu_n, n >= 1
    lam_b=None,                    # candidate pole range [1e-4 lam_b, lam_b] of the data-selected basis; None = self.bos.lam
    eps_b=1e-12, npole_b=800,      # pivoted-QR tolerance / candidate count of BosonicLineBasis.from_data
    cut_odd=1e-13, cut_even=1e-10, # relative SVD cutoffs of the odd / even sectors of fit_split
    deg_tol=1e-8,                  # |e_n(k) - e_m(k-q)| below which a pair is degenerate (excluded from the dynamic Pi(q, 0))
    tau_nn=12, tau_per_efold=2.0, tau_x0=0.02,   # tau-leg grid: composite GL, nn points per panel, log panels from x0/E_max
)


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
    def __init__(self, X, Z, qk_to_k2, nk, mu, theta, theta_t, bos_basis, ferm_zeta, t_chunk=8, ray_decades=36.0, ray_kw=None,
                 wstep=None):
        """X: (nk, Np, nb) THC collocation; Z: (nq, Np, Np); qk_to_k2[iq, ik] = index of k - q; mu: absolute centre.
        bos_basis: BosonicLineBasis (mu-relative); ferm_zeta: mu-relative fermionic line nodes for Sigma/G.
        The time rays are built per call from the current pole spectrum (smallest |e_m| sets s_max)."""
        self.X, self.Z, self.qk, self.nk, self.mu = X, Z, qk_to_k2, nk, mu
        self.Np, self.nb = X.shape[1], X.shape[2]
        self.theta, self.theta_t = theta, theta_t
        self.bos, self.fz = bos_basis, np.asarray(ferm_zeta, complex)
        self.t_chunk, self.ray_decades, self.ray_kw = t_chunk, ray_decades, (ray_kw or {})
        self.poles = None
        self.wstep = dict(WSTEP_DEFAULTS, **(wstep or {}))               # finite-T W step parameters (module docstring)
        self.bos_w, self._wkey = None, None

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
        # bosonic data set D and the basis selected on it: depend on beta / floor / line nodes / wstep only (not on the poles)
        key = (beta, self.thermal_floor, tuple(sorted(self.wstep.items())), id(self.bos), len(self.bos.zeta))
        if key != self._wkey:
            self.zD, self.kD = self.bos_data()
            ws = self.wstep
            self.bos_w = BosonicLineBasis.from_data(self.theta, self.zD, self.bos.lam if ws['lam_b'] is None else ws['lam_b'],
                                                    eps=ws['eps_b'], npole=ws['npole_b'])
            self._wkey = key
        # tau-leg grid (pi_tau_leg): graded composite GL on [0, beta], decay scale 1/E_max of the current poles
        emax = float(np.abs(e[np.abs(e) < 1e5]).max())
        self.tau, self.wtau = tau_grid(beta, emax, nn=self.wstep['tau_nn'], per_efold=self.wstep['tau_per_efold'],
                                       x0=self.wstep['tau_x0'])

    def window_counts(self):
        """Number of window poles per k (thermal mode), zeros otherwise."""
        return self.win.sum(1) if self.thermal else np.zeros(self.poles[0].shape[0], int)

    def bos_nodes(self):
        """Bosonic points used for Dyson + fit: all basis nodes at T = 0; in thermal mode the data set D of bos_data()
        (unmasked line nodes, wedge band, i nu_n, nu_0 = 0)."""
        return self.bos.zeta if not self.thermal else self.zD

    def bos_data(self):
        """Bosonic data set D of the finite-T W step (notes section 11.5(a)) as (z (nD,), kind (nD,) int8):
        kind 0 = line nodes of self.bos with rho beta |zeta| >= c_zeta; 1 = wedge band (heights v = y cos(theta_t) - |x| sin(theta_t)
        log-spaced in [d0, vtop], d0 = c_band sin(theta_t)/beta, vtop = band_top zeta_T sin(theta); at each height band_x values of
        x in [-xm, xm], xm = |vtop cos(theta_t) - v|/sin(theta_t), y = (v + |x| sin(theta_t))/cos(theta_t); symmetric under
        z -> -conj(z)); 2 = i nu_n, n = 1..N_M; 3 = nu_0 (z = 0, from the tau leg)."""
        ws, beta, tht = self.wstep, self.beta, self.theta_t
        zT = self.thermal_floor / (thermal_rho(self.theta, tht) * beta)
        zl = self.bos.zeta[node_floor_mask(self.bos.zeta, beta, self.theta, tht, self.thermal_floor)]
        cb = self.thermal_floor if ws['band_c'] is None else ws['band_c']
        d0 = cb * np.sin(tht) / beta; vtop = ws['band_top'] * zT * np.sin(self.theta)
        band = []
        for v in np.exp(np.linspace(np.log(d0), np.log(vtop), ws['band_heights'])):
            xm = abs(vtop * np.cos(tht) - v) / np.sin(tht)
            xs = np.linspace(-xm, xm, ws['band_x']); xs = 0.5 * (xs - xs[::-1])          # exactly symmetric: z -> -conj(z)
            for x in xs:
                band.append(x + 1j * (v + abs(x) * np.sin(tht)) / np.cos(tht))
        nm = int(np.ceil(ws['mats_factor'] * zT * beta / (2 * np.pi)))
        zm = 2j * np.pi * np.arange(1, nm + 1) / beta
        z = np.concatenate([zl, np.array(band), zm, [0j]])
        kind = np.concatenate([np.zeros(len(zl)), np.ones(len(band)), np.full(nm, 2), [3]]).astype(np.int8)
        return z, kind

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
        Eq. fT_floor; exact on the wedge Eq. fT_wedge); default points = the data set D; points zeta == 0 (nu_0, outside the
        wedge) are taken from the tau leg in the dynamic convention (pi_tau_leg)."""
        zeta = self.bos_nodes() if zeta is None else np.asarray(zeta, complex)
        if self.thermal and np.any(zeta == 0):
            out = np.zeros((len(zeta), self.Np, self.Np), complex)
            z0 = zeta == 0
            if (~z0).any(): out[~z0] = self.polarization(iq, zeta[~z0])
            PiM, dPi = self.pi_tau_leg(iq, [0])
            out[z0] = PiM[0] - dPi
            return out
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

    def gtilde_tau(self, ik, tau, kind):
        """Imaginary-time factors of the tau leg with ALL poles (no window truncation; every factor bounded by 1 on [0, beta]):
        kind 'h': X(k) [sum_m f_m e^{+e_m tau} coef_m] X(k)^dag  (= G~(k, -tau));
        kind 'p': X(k) [sum_m (1 - f_m) e^{-e_m tau} coef_m] X(k)^dag  (= -G~(k, tau)); (ntau, Np, Np). Weights in log form."""
        e, coef = self.poles
        ek = np.asarray(e[ik], float); tau = np.asarray(tau, float)
        if kind == 'h':
            lw = -np.logaddexp(0.0, self.beta * ek)[None, :] + ek[None, :] * tau[:, None]
        else:
            lw = -np.logaddexp(0.0, -self.beta * ek)[None, :] - ek[None, :] * tau[:, None]
        ph = np.exp(lw)
        Gt = (ph @ coef[ik].reshape(ph.shape[1], -1)).reshape(ph.shape[0], self.nb, self.nb)
        Xk = self.X[ik]
        return (Xk @ Gt) @ Xk.conj().T

    def pi_tau_dpi(self, iq):
        """Degenerate-pair term of the tau leg at nu_0: the tau integral of the pairs (n at k, m at k-q) with |e_n - e_m| <
        deg_tol, -(2/Nk) sum f_n (1 - f_m) phi_nm [X c_n X^dag]_PQ [X c_m X^dag]_QP, phi = int_0^beta e^{(e_n - e_m) tau} dtau
        (= beta for an exact degeneracy). Pi^Mats(q, i nu_0) - this = Pi^an(q, 0), the dynamic (retarded) convention of the
        line (exactly degenerate pairs carry no weight there; finite_t.pi_nu0_extra is the KS oracle of this term)."""
        e, coef = self.poles; beta, tol = self.beta, self.wstep['deg_tol']
        out = np.zeros((self.Np, self.Np), complex)
        for ik in range(self.nk):
            ikmq = self.qk[iq, ik]
            en, em = np.asarray(e[ik], float), np.asarray(e[ikmq], float)
            Ed = en[:, None] - em[None, :]
            nn_, mm_ = np.nonzero(np.abs(Ed) < tol)
            for n, m in zip(nn_, mm_):
                x = Ed[n, m]
                phi = beta if x == 0.0 else np.expm1(beta * x) / x
                w = np.exp(-np.logaddexp(0.0, beta * en[n]) - np.logaddexp(0.0, -beta * em[m])) * phi
                if w == 0.0: continue
                A = self.X[ik] @ coef[ik][n] @ self.X[ik].conj().T
                B = self.X[ikmq] @ coef[ikmq][m] @ self.X[ikmq].conj().T
                out += w * (A * B.T)
        return -(2.0 / self.nk) * out

    def pi_tau_leg(self, iq, n_list=(0,), chunk=32):
        """tau leg of the finite-T W step (notes section 11.5(a)): Pi(q, i nu_n) in the MATSUBARA convention by the tau quadrature
        of Pi(q, tau)_PQ = -(2/Nk) sum_k [gtilde_tau(k, 'h')]_PQ [gtilde_tau(k-q, 'p')]_QP on the grid (self.tau, self.wtau)
        (bounded products, KMS), and the degenerate-pair term dPi of nu_0 (pi_tau_dpi). Returns (PiM (len(n_list), Np, Np), dPi);
        the dynamic Pi(q, 0) of the data set is PiM[n = 0] - dPi. Oracle: finite_t.pi_matsubara_tau / pi_nu0_extra."""
        nus = 2 * np.pi * np.asarray(n_list, float) / self.beta
        out = np.zeros((len(nus), self.Np, self.Np), complex)
        for i0 in range(0, len(self.tau), chunk):
            tt, wt = self.tau[i0:i0 + chunk], self.wtau[i0:i0 + chunk]
            acc = np.zeros((len(tt), self.Np, self.Np), complex)
            for ik in range(self.nk):
                acc += self.gtilde_tau(ik, tt, 'h') * np.transpose(self.gtilde_tau(self.qk[iq, ik], tt, 'p'), (0, 2, 1))
            acc *= -2.0 / self.nk
            ph = np.exp(1j * nus[:, None] * tt[None, :]) * wt[None, :]
            out += (ph @ acc.reshape(len(tt), -1)).reshape(len(nus), self.Np, self.Np)
        return out, self.pi_tau_dpi(iq)

    def dyson_w(self, iq, Pi):
        """W(q, zeta_i) = ([1 - Z Pi]^-1 - 1) Z for each node; (nz, Np, Np)."""
        Z = self.Z[iq]; I = np.eye(self.Np)
        return np.array([np.linalg.solve(I - Z @ P, Z) - Z for P in Pi])

    def screened_interaction(self, iq, Pi=None, W_minus=None):
        """Residues w_j(q) (r, Np, Np) of the symmetric real-pole fit of W(q) on the bosonic nodes; also returns W at the nodes.
        W_minus: W(-q) at the nodes (required for q != -q, see the module docstring); None = self-inverse q.
        Thermal mode: Dyson at every point of the data set D (bos_nodes(); Pi, W_minus given there), residues on the D-selected
        basis self.bos_w by the decoupled odd/even fit (w(q) of the joint solve; w_step() keeps w(-q) of the same solve)."""
        z = self.bos_nodes()
        if Pi is None: Pi = self.polarization(iq, z)
        W = self.dyson_w(iq, Pi)
        if self.thermal:
            ws = self.wstep
            return self.bos_w.fit_split(z, W, W_minus=W_minus, cut_odd=ws['cut_odd'], cut_even=ws['cut_even'])[0], W
        return self.bos.fit(z, W, W_minus=W_minus), W

    def w_step(self, qminus, Pi=None):
        """Residues of W for every q (list of (r, Np, Np)) and W at the bosonic points (list), Pi optional (list over q at
        bos_nodes()). Thermal mode: Pi on D (rays + tau leg), Dyson on D, fit_split on self.bos_w with w(q) and w(-q) taken from
        the ONE joint solve of each pair (q, -q). T = 0: screened_interaction per q (W_minus for q != -q)."""
        z = self.bos_nodes()
        if Pi is None: Pi = [self.polarization(iq, z) for iq in range(self.nk)]
        W = [self.dyson_w(iq, Pi[iq]) for iq in range(self.nk)]
        wres = [None] * self.nk
        ws = self.wstep
        for iq in range(self.nk):
            if wres[iq] is not None: continue
            qm = qminus[iq]
            if not self.thermal:
                wres[iq] = self.bos.fit(z, W[iq], W_minus=None if qm == iq else W[qm])
                continue
            w, wm = self.bos_w.fit_split(z, W[iq], W_minus=None if qm == iq else W[qm], cut_odd=ws['cut_odd'], cut_even=ws['cut_even'])
            wres[iq] = w
            if qm != iq: wres[qm] = wm
        return wres, W

    # ---------------------------------------------------------------- self-energy
    def sigma(self, ik, wres, zeta=None, qminus=None, nu=None):
        """Sigma_c(k, zeta)_ab (nz, nb, nb), zeta mu-relative (default fermionic nodes); wres: list over q of residues (r, Np, Np).
        qminus: index of -q per q (the hole sector uses w(-q)^T); None = every q self-inverse.
        nu: optional list over q of the positive pole energies of wres[q] (default: self.bos.nu for every q at T = 0, the
        D-selected basis self.bos_w.nu in thermal mode).
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
        nu0 = (self.bos_w if self.thermal else self.bos).nu
        nu_of = (lambda iq: nu0) if nu is None else (lambda iq: np.asarray(nu[iq], float))
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
