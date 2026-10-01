"""GW kernels on the tilted line in the THC/ISDF basis (numpy prototype of option D).

All energies are measured from the line centre mu. Pole data for G: per k, energies e_m (mu-relative) and vectors v_m
(nb,) with G(k, zeta) = sum_m v_m v_m^dagger /(zeta - e_m); sector by the sign of e_m.

Time ray (particle sector): t = s e^{-i theta_t};  G~^>(k,t)_PQ = [X(k) (sum_{m>} v_m v_m^dag e^{-i e_m t}) X(k)^dag]_PQ.
Polarization (sign/normalization validated against the Casida transition sum, V1):
    Pi^>(q,t)_PQ = (2/Nk) sum_k G~^>(k,t)_PQ * conj(G~^<(k-q,t))_PQ,   G~^<(k,t) = X(k)(sum_{m<} v_m v_m^dag e^{-i e_m t})X(k)^dag
    Pi^>(q,zeta) = -i int_0^inf dt e^{i zeta t} Pi^>(q,t);   Pi(zeta) = Pi^>(zeta) + conj(Pi^>(-conj(zeta)))  (elementwise)
Screened interaction: W(q,zeta) = ([1 - Z Pi(zeta)]^-1 - 1) Z at the bosonic line nodes, refit with BosonicLineBasis
    W_PQ(zeta) = sum_j [ w_j,PQ/(zeta - nu_j) - w_j,QP/(zeta + nu_j) ],   W^>(t) = sum_j w_j e^{-i nu_j t},  W^<(t) = -sum_j w_j^T e^{+i nu_j t}.
Self-energy per sector (contracted to orbitals immediately):
    Sigma~^>(k,t) = (1/Nk) sum_q G~^>(k-q,t) * W^>(q,t);   Sigma^>_ab(k,zeta) = -i int dt e^{i zeta t} [X(k)^dag Sigma~^>(k,t) X(k)]_ab
    Sigma~^<(k,t) = (1/Nk) sum_q G~^<(k-q,t) * W^<(q,t)  on the hole ray t = s e^{+i theta_t}.
Static part: F = V_H[Dm] + Sigma_x[Dm] with the THC Z (ignore_g0 heads are inside Z), spin-restricted (Dm per spin).
"""
import numpy as np
from .timeray import TimeRay
from .line_dlr import LineBasis, BosonicLineBasis


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
    def set_poles(self, e, v=None, coef=None):
        """G(k, zeta) = sum_m coef_m /(zeta - e_m) with real mu-relative energies e (nk, M). Either v (nk, nb, M) column
        vectors (coef_m = v_m v_m^dagger, Lehmann form) or general matrix coefficients coef (nk, M, nb, nb) (compressed
        real-pole fit per sector). Sector = sign of e_m."""
        e = np.asarray(e)
        if coef is None:
            v = np.asarray(v)
            coef = np.einsum('kim,kjm->kmij', v, v.conj())
        self.poles = (e, np.asarray(coef))
        emin = min(np.abs(e[e > 0]).min(), np.abs(e[e < 0]).min())
        self.ray_p = TimeRay.for_spectrum(self.theta_t, emin, decades=self.ray_decades, sector='>', **self.ray_kw)
        self.ray_h = TimeRay.for_spectrum(self.theta_t, emin, decades=self.ray_decades, sector='<', **self.ray_kw)

    @classmethod
    def poles_from_hamiltonian(cls, H, mu):
        """Diagonalize H (nk, nb, nb) -> (e - mu, v)."""
        e, v = np.linalg.eigh(H)
        return e - mu, v

    def gtilde(self, ik, t, sector):
        """G~(k, t) (nt, Np, Np) for the given sector on complex times t (nt,): X(k) [sum_m coef_m e^{-i e_m t}] X(k)^dag."""
        e, coef = self.poles
        m = e[ik] > 0 if sector == '>' else e[ik] < 0
        ph = np.exp(-1j * e[ik][m][None, :] * t[:, None])              # (nt, M)
        Gt = np.einsum('tm,mij->tij', ph, coef[ik][m])                  # (nt, nb, nb)
        Xk = self.X[ik]
        return (Xk @ Gt) @ Xk.conj().T                                  # (nt, Np, Np): two GEMMs per t

    def g_line(self, ik, zeta, sector=None):
        """G(k, zeta) (nz, nb, nb) from the pole data (any complex zeta off the real axis)."""
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
        and Pi^{>/<}(zeta) = -i int_0^inf dt e^{i zeta t} Pi^{>/<}(t)."""
        zeta = self.bos.zeta if zeta is None else np.asarray(zeta, complex)
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
                out += np.einsum('zt,tpq->zpq', F[:, i0:i0 + self.t_chunk], acc)
        return out

    def dyson_w(self, iq, Pi):
        """W(q, zeta_i) = ([1 - Z Pi]^-1 - 1) Z for each node; (nz, Np, Np)."""
        Z = self.Z[iq]; I = np.eye(self.Np)
        return np.array([np.linalg.solve(I - Z @ P, Z) - Z for P in Pi])

    def screened_interaction(self, iq, Pi=None):
        """Residues w_j(q) (r, Np, Np) of the symmetric real-pole fit of W(q) on the bosonic nodes; also returns W at the nodes."""
        if Pi is None: Pi = self.polarization(iq, self.bos.zeta)
        W = self.dyson_w(iq, Pi)
        return self.bos.fit(self.bos.zeta, W), W

    # ---------------------------------------------------------------- self-energy
    def sigma(self, ik, wres, zeta=None):
        """Sigma_c(k, zeta)_ab (nz, nb, nb), zeta mu-relative (default fermionic nodes); wres: list over q of residues (r, Np, Np)."""
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
                    w = wres[iq] if sector == '>' else np.transpose(wres[iq], (0, 2, 1))
                    Wt = np.einsum('tj,jpq->tpq', Ew, w)
                    acc += self.gtilde(self.qk[iq, ik], t, sector) * Wt
                acc *= (1.0 if sector == '>' else -1.0) / self.nk              # T=0 factor [theta(nu) - theta(-eps)] = -1 in the hole sector
                S_ab = (Xk.conj().T @ acc) @ Xk                           # (nt, nb, nb)
                out += np.einsum('zt,tab->zab', F[:, i0:i0 + self.t_chunk], S_ab)
        return out

    # ---------------------------------------------------------------- static part
    def density_matrix(self):
        """Dm(k) = sum_{m<} coef_m (per spin, T=0)."""
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
