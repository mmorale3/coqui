"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Reconstructs the auxiliary-basis polarization Pi_PQ(q_IBZ, Omega) from a
real-axis chkpt file (which stores the orbital-basis spectral function A)
plus a separately-supplied THC h5 file (which stores the X collocation
matrix and the BZ map).

This is a **Python re-implementation** of the C++ kernel in
`src/methods/GW_real_axis/real_axis_pi.hpp::accumulate_ImPi_one_kq`,
intended for plotting / diagnostics. It is not bit-identical with the
C++ run (FINUFFT vs cuFINUFFT, plan caching) but agrees within
NUFFT eps on small fixtures.

Recipe (per IBZ q, per (P, Q) pair):

  A_aux_PQ(s, k_FBZ, omega) = sum_{mu, nu}
      conj(X(s, k_FBZ, P, mu)) * A_phys(s, k_IBZ(k), mu, nu, omega) * X(s, k_FBZ, Q, nu)

  where A_phys = 0.5 * (A + A^H) is the matrix-hermitian projection of
  the stored componentwise A = -(i/pi) G^R.

  Im Pi_PQ(q, Omega) = -pi * (1/Nk) * sum_{s, k_FBZ} accumulate(...)

  with the kernel below building four weighted spectra at k and k+q,
  forward-NUFFTing each to t-space, combining
  Hhat = conj(F_less)*G_gtr - conj(F_gtr)*G_less, and inverse-NUFFTing
  to the bosonic Omega grid.

  Re Pi(Omega) = Hilbert transform of Im Pi on the bosonic grid.

User must provide the THC h5 file path. The THC h5 is produced when the
toml has `save = "<path>"` set in the [interaction.thc] block; the
default `storage = "incore"` without `save` leaves nothing on disk and
this reconstructor will fail to load.

Usage::

    from coqui.post_proc import RealAxisChkpt, RealAxisPiReconstructor
    ck = RealAxisChkpt("si_scgw.mbpt.h5")
    pi = RealAxisPiReconstructor(ck, thc_h5_path="si.thc.h5")
    Im, Re = pi.reconstruct_Pi_PQ(q_IBZ=0, P=0, Q=0, hilbert=True)
    pi.plot_Pi(q_IBZ=0, pairs=[(0, 0), (1, 1)])
==========================================================================
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from .real_axis_chkpt import RealAxisChkpt, _read_real, _read_complex
except ImportError:
    # Fallback for direct-script usage (no parent package).
    from real_axis_chkpt import RealAxisChkpt, _read_real, _read_complex


_HARTREE_TO_EV = 27.211386245988


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    w = np.empty(N, dtype=float)
    if N == 1:
        w[:] = 0.0
        return w
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if N > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _fermi(omega: np.ndarray, mu: float, beta: float) -> np.ndarray:
    x = beta * (omega - mu)
    return np.where(x >= 0.0, np.exp(-x) / (1.0 + np.exp(-x)),
                              1.0 / (1.0 + np.exp(x)))


def kernel_ImPi_one_kq(
    A_PQ_k: np.ndarray,
    A_PQ_kq: np.ndarray,
    omega: np.ndarray,
    Omega: np.ndarray,
    t: np.ndarray,
    *,
    beta: float,
    k_weight: float = 1.0,
    eps_nufft: float = 1e-10,
) -> np.ndarray:
    """Standalone kernel: accumulate one (k, k+q) Pi contribution.

    Mirrors C++ accumulate_ImPi_one_kq exactly, in pure Python with finufft.
    Use this for independent testing of the kernel arithmetic without
    needing a chkpt + THC h5 path.

    Parameters
    ----------
    A_PQ_k, A_PQ_kq : (NP, NQ, Nw) complex
    omega : (Nw,) — fermionic ω grid (relative to μ_chem)
    Omega : (NΩ,) — bosonic Ω grid (positive half)
    t : (Nt,) — uniform time grid
    beta, k_weight : scalars
    eps_nufft : FINUFFT accuracy

    Returns
    -------
    delta_ImPi : (NP, NQ, NΩ) real — to be ACCUMULATED by caller.
    """
    from finufft import Plan

    NP, NQ, Nw = A_PQ_k.shape
    assert A_PQ_kq.shape == A_PQ_k.shape
    assert omega.shape[0] == Nw
    NO = Omega.shape[0]
    Nt = t.shape[0]
    B = NP * NQ

    dt = float(t[1] - t[0])
    nufft_scale = dt / (2.0 * np.pi)
    x_w = omega * dt
    x_O = Omega * dt

    f = _fermi(omega, mu=0.0, beta=beta)
    fb = 1.0 - f
    wq = _trapz_weights(omega)

    # Build 4 weighted spectra.
    A_k_flat = A_PQ_k.reshape(B, Nw).astype(np.complex128)
    A_kq_flat = A_PQ_kq.reshape(B, Nw).astype(np.complex128)
    Aless_k = A_k_flat * (f * wq)[None, :]
    Agtr_k = A_k_flat * (fb * wq)[None, :]
    Akq_conj = np.conj(A_kq_flat)
    Aless_kq = Akq_conj * (f * wq)[None, :]
    Agtr_kq = Akq_conj * (fb * wq)[None, :]

    # Forward NUFFT (ω → t).
    plan_w_fwd = Plan(1, (Nt,), n_trans=B, isign=+1, eps=eps_nufft)
    plan_w_fwd.setpts(x_w.astype(np.float64))
    Fless = plan_w_fwd.execute(Aless_k)
    Fgtr = plan_w_fwd.execute(Agtr_k)
    Gless = plan_w_fwd.execute(Aless_kq)
    Ggtr = plan_w_fwd.execute(Agtr_kq)

    # Hadamard combine.
    Hhat = np.conj(Fless) * Ggtr - np.conj(Fgtr) * Gless

    # Inverse / type-2 (t → Ω) on bosonic grid.
    plan_O_inv = Plan(2, (Nt,), n_trans=B, isign=+1, eps=eps_nufft)
    plan_O_inv.setpts(x_O.astype(np.float64))
    Hraw = plan_O_inv.execute(Hhat)

    s_pi = -np.pi * k_weight * nufft_scale
    return (s_pi * Hraw.real).reshape(NP, NQ, NO)


class RealAxisPiReconstructor:
    """Reconstruct Pi_PQ(q, Omega) from a real-axis chkpt + THC h5.

    Parameters
    ----------
    chkpt : RealAxisChkpt
        The chkpt providing A_wskij at orbital basis. Must be in scGW mode
        (so A_wskij is present).
    thc_h5_path : str
        Path to the THC h5 file containing collocation_matrix (X factor),
        kpts, qpts, kp_to_ibz, etc.
    eps_nufft : float
        FINUFFT accuracy tolerance. Default 1e-10 (matches the C++ default).
    """

    def __init__(
        self,
        chkpt: RealAxisChkpt,
        thc_h5_path: str,
        eps_nufft: float = 1e-10,
    ):
        import h5py

        self._ck = chkpt
        self._thc_path = thc_h5_path
        self._eps = float(eps_nufft)
        if not chkpt.is_scgw():
            raise RuntimeError(
                "RealAxisPiReconstructor requires a scGW chkpt with A_wskij; "
                "this chkpt is QSGW (no orbital A stored)."
            )

        # Open THC h5 and load X + BZ data eagerly (X is moderate size; the
        # alternative would be a per-call h5 read which is slower).
        self._thc = h5py.File(thc_h5_path, "r")
        # X factor: (ns_in_basis * npol_in_basis, nkpts_FBZ, Np, nbnd_X)
        # stored as complex (..., 2). Keep as h5 dataset for lazy slicing.
        if "collocation_matrix" not in self._thc:
            raise RuntimeError(
                f"{thc_h5_path}: no /collocation_matrix dataset. "
                "Was this THC h5 written with `save=\"...\"` set?"
            )
        self._X_ds = self._thc["collocation_matrix"]
        # Sanity-check shape vs chkpt.
        ck_ns = chkpt.ns
        ck_Nk = chkpt.Nk
        ck_nbnd = chkpt.nbnd
        Xshape = self._X_ds.shape
        if len(Xshape) not in (4, 5):
            raise RuntimeError(
                f"X dataset has unexpected rank {len(Xshape)}; expected 4 or 5"
            )
        # X may be stored as (..., 2) for complex.
        if len(Xshape) == 5:
            assert Xshape[-1] == 2, "complex axis on X must be size 2"
            self._X_complex_axis = True
            self._Np = int(Xshape[2])
            self._X_nbnd = int(Xshape[3])
        else:
            self._X_complex_axis = False
            self._Np = int(Xshape[2])
            self._X_nbnd = int(Xshape[3])
        if Xshape[1] != ck_Nk:
            raise RuntimeError(
                f"X.shape[1]={Xshape[1]} (FBZ k) != chkpt.Nk={ck_Nk}"
            )
        if self._X_nbnd != ck_nbnd:
            raise RuntimeError(
                f"X.shape[3]={self._X_nbnd} (X nbnd) != chkpt.nbnd={ck_nbnd}; "
                "the THC X_orbital_range and chkpt bands must agree."
            )
        # Some THC files have an X_orbital_range; we assume it spans full nbnd.

        # BZ data — try THC h5 first, fall back to chkpt.
        self._kpts = self._thc["kpts"][:] if "kpts" in self._thc else chkpt.kpts
        self._qpts = self._thc["qpts"][:] if "qpts" in self._thc else None
        self._kp_to_ibz = (
            self._thc["kp_to_ibz"][:].astype(np.int64)
            if "kp_to_ibz" in self._thc
            else chkpt._sys["kp_to_ibz"][:].astype(np.int64)
        )
        # qk_to_k2 / qminus from chkpt's /system/ group (THC h5 usually has these too).
        self._qk_to_k2 = chkpt._sys["qk_to_k2"][:].astype(np.int64)
        self._qminus = chkpt._sys["qminus"][:].astype(np.int64)
        # Optional fields for non-inversion fixtures — default None if absent.
        self._kp_trev = (
            chkpt._sys["k_trev"][:].astype(np.int64)
            if "k_trev" in chkpt._sys
            else np.zeros(ck_Nk, dtype=np.int64)
        )
        self._Nk_FBZ = ck_Nk
        self._Nq_FBZ = self._qpts.shape[0] if self._qpts is not None else None

        # Map IBZ q-index -> FBZ q-index by matching Qpts_ibz against qpts.
        # The chkpt's mean_field group doesn't carry Qpts_ibz directly — we
        # rely on the convention that IBZ q-index 0 is Gamma (FBZ q=0) and
        # other IBZ q's correspond to the q-stars listed in qpts.
        # For a robust mapping, look for Qpts_ibz in the THC h5 if present.
        if "Qpts_ibz" in self._thc:
            qibz = self._thc["Qpts_ibz"][:]
            self._qibz_to_qfbz = self._match_q_to_fbz(qibz)
        else:
            # Fallback: assume IBZ q indexing is [0, 1, ..., Nq_ibz-1] mapping
            # to the first Nq_ibz entries of qpts. Will warn if used at q!=0.
            self._qibz_to_qfbz = None

        # Grid arrays from chkpt.
        self._omega = chkpt.omega          # (Nw,) — relative to mu_chem
        self._Omega = chkpt.Omega          # (NΩ,) — positive bosonic
        self._t = chkpt.t                  # (Nt,) — symmetric uniform time
        self._beta = chkpt.beta
        self._mu_chem = chkpt.mu_chem
        self._Nw = self._omega.shape[0]
        self._NO = self._Omega.shape[0]
        self._Nt = self._t.shape[0]
        # dt and the derived NUFFT scale factor (mirrors conv_t::nufft_scale).
        self._dt = self._t[1] - self._t[0]
        self._nufft_scale = self._dt / (2.0 * np.pi)
        # x = ω * dt and x = Ω * dt — scaled coordinates for FINUFFT in [-π, π].
        self._x_w = self._omega * self._dt
        self._x_O = self._Omega * self._dt
        # Trapezoidal weights for ω quadrature (used in Aless/Agtr build).
        self._w_weights = self._make_trapz_weights(self._omega)
        self._O_weights = self._make_trapz_weights(self._Omega)
        # Fermi factor on the relative ω grid (mu = 0 since ω is relative).
        self._fermi_w = self._fermi(self._omega, mu=0.0, beta=self._beta)

    # ---------------------------------------------------------------- helpers
    # Use module-level helpers _trapz_weights and _fermi (also re-exported as
    # _make_trapz_weights / _fermi for backward compatibility).
    _make_trapz_weights = staticmethod(_trapz_weights)
    _fermi = staticmethod(_fermi)

    def _match_q_to_fbz(self, qibz: np.ndarray) -> np.ndarray:
        """Return mapping IBZ q-index -> FBZ q-index by minimum-distance match."""
        Nq_ibz = qibz.shape[0]
        out = np.empty(Nq_ibz, dtype=np.int64)
        for i in range(Nq_ibz):
            d = np.sum((self._qpts - qibz[i])**2, axis=1)
            out[i] = int(np.argmin(d))
        return out

    def _read_X(self, s: int, k_FBZ: int) -> np.ndarray:
        """Read X(s, k_FBZ, :, :) returning (Np, nbnd) complex."""
        # X dataset shape: (ns_in_basis*npol_in_basis, nkpts_FBZ, Np, nbnd, 2)
        # s axis is collapsed when ns_in_basis == 1; we always slice [s, k_FBZ, :, :].
        # For npol == 1 this is just `[s, k_FBZ, :, :, :]`.
        if self._X_complex_axis:
            Xre = np.asarray(self._X_ds[s, k_FBZ, :, :, 0], dtype=float)
            Xim = np.asarray(self._X_ds[s, k_FBZ, :, :, 1], dtype=float)
            return Xre + 1j * Xim
        return np.asarray(self._X_ds[s, k_FBZ, :, :], dtype=np.complex128)

    # ---------------------------------------------------------------- API
    def project_orbital_to_aux(
        self,
        s: int,
        k_FBZ: int,
        P_idx: Sequence[int],
        Q_idx: Sequence[int],
    ) -> np.ndarray:
        """Compute A_aux_{P_idx, Q_idx}(omega) at one (s, k_FBZ).

        Returns
        -------
        A_aux : ndarray, shape (NP, NQ, Nw), complex
            Where NP = len(P_idx), NQ = len(Q_idx).

        Notes
        -----
        Includes the matrix-hermitian symmetrization 0.5*(A + A^H) and the
        IBZ k orbital read via kp_to_ibz. TR-pair fix-up: if k_FBZ has
        kp_trev=1 we conj the result on the (P, Q) block (mirrors the
        C++ projection's TR-pair pass).
        """
        kibz = int(self._kp_to_ibz[k_FBZ])
        # Read full A at IBZ k (Nw, nbnd, nbnd) complex; symmetrize.
        A = _read_complex(
            self._ck._iter_grp["A_wskij"],
            slice(None), s, kibz, slice(None), slice(None),
        )
        # A_phys = 0.5 (A + A^H) — Hermitian projection on (i, j).
        A_phys = 0.5 * (A + np.conj(A.swapaxes(-1, -2)))

        X = self._read_X(s, k_FBZ)  # (Np, nbnd) cplx
        Xp = X[np.asarray(P_idx, dtype=int), :]  # (NP, nbnd)
        Xq = X[np.asarray(Q_idx, dtype=int), :]  # (NQ, nbnd)

        # A_aux_{P, Q, w} = sum_{mu, nu} conj(Xp[P, mu]) * A_phys[w, mu, nu] * Xq[Q, nu]
        # Use einsum for clarity; small dims so cost is fine.
        A_aux = np.einsum("Pm,wmn,Qn->PQw",
                          np.conj(Xp), A_phys, Xq, optimize=True)

        # TR-pair fix-up.
        if self._kp_trev[k_FBZ] != 0:
            A_aux = np.conj(A_aux)
        return A_aux

    def _kernel_one_kq(
        self,
        A_PQ_k: np.ndarray,
        A_PQ_kq: np.ndarray,
        plan_w_fwd,
        plan_O_inv,
        k_weight: float,
    ) -> np.ndarray:
        """Single (s, k_FBZ) kernel call. Mirrors accumulate_ImPi_one_kq.

        A_PQ_k, A_PQ_kq : (NP, NQ, Nw) complex
        Returns delta-Pi for this (k, k+q): (NP, NQ, NΩ) real.
        """
        NP, NQ, Nw = A_PQ_k.shape
        assert Nw == self._Nw and A_PQ_kq.shape == A_PQ_k.shape
        B = NP * NQ
        # Build 4 weighted spectra, shape (B, Nw).
        f = self._fermi_w
        fb = 1.0 - f
        wq = self._w_weights
        A_k_flat = A_PQ_k.reshape(B, Nw)
        A_kq_flat = A_PQ_kq.reshape(B, Nw)
        Aless_k = A_k_flat * (f * wq)[None, :]
        Agtr_k = A_k_flat * (fb * wq)[None, :]
        # Second leg: conj of LOCAL block per aux hermiticity (commit 40d91a2 in C++).
        Akq_conj = np.conj(A_kq_flat)
        Aless_kq = Akq_conj * (f * wq)[None, :]
        Agtr_kq = Akq_conj * (fb * wq)[None, :]
        # Forward NUFFT (ω → t) for all four. Plan is cached and pre-set on x_w.
        # finufft Plan.execute does ntrans transforms in one call; we run
        # one batch of B at a time.
        Fless = plan_w_fwd.execute(Aless_k.astype(np.complex128))
        Fgtr = plan_w_fwd.execute(Agtr_k.astype(np.complex128))
        Gless = plan_w_fwd.execute(Aless_kq.astype(np.complex128))
        Ggtr = plan_w_fwd.execute(Agtr_kq.astype(np.complex128))
        # Hadamard combine: Hhat = conj(F_less) G_gtr - conj(F_gtr) G_less.
        Hhat = np.conj(Fless) * Ggtr - np.conj(Fgtr) * Gless
        # Inverse / type-2 (t → Ω) on bosonic grid — done via the inv plan.
        Hraw = plan_O_inv.execute(Hhat)  # shape (B, NΩ) cplx
        # Accumulate factor: ImPi += -π · k_weight · s_nufft · Re(Hraw).
        # The result of the cross-correlation is real for symmetric A (the
        # imag part is round-off); we take the real part and apply the scale.
        s_pi = -np.pi * k_weight * self._nufft_scale
        delta = s_pi * Hraw.real
        return delta.reshape(NP, NQ, self._NO)

    def reconstruct_Pi_PQ(
        self,
        q_IBZ: int,
        P: int,
        Q: int,
        *,
        hilbert: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Reconstruct Im Pi_PQ(q_IBZ, Omega) for a single (P, Q) pair.

        Returns
        -------
        ImPi : ndarray, shape (NΩ,) real
        RePi : ndarray, shape (NΩ,) real -- only if hilbert=True, else None
        """
        Im, Re = self.reconstruct_Pi_PQ_block(
            q_IBZ, [P], [Q], hilbert=hilbert,
        )
        ImPi = Im[0, 0, :]
        RePi = Re[0, 0, :] if Re is not None else None
        return ImPi, RePi

    def reconstruct_Pi_PQ_block(
        self,
        q_IBZ: int,
        P_idx: Sequence[int],
        Q_idx: Sequence[int],
        *,
        hilbert: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Vectorized: reconstruct Im Pi_{P, Q}(q_IBZ, Omega) over a (P, Q) grid.

        For per-element diagonal Pi_PP, pass `P_idx=[p1, p2, ...]` and
        `Q_idx=[p1, p2, ...]` (same list).

        Returns
        -------
        ImPi : ndarray, shape (NP, NQ, NΩ) real
        RePi : ndarray, shape (NP, NQ, NΩ) real, or None
        """
        from finufft import Plan

        # Resolve q_IBZ -> q_FBZ.
        if self._qibz_to_qfbz is not None:
            q_FBZ = int(self._qibz_to_qfbz[q_IBZ])
        else:
            # Fallback: assume Gamma is at q=0 in both indexings.
            if q_IBZ == 0:
                q_FBZ = 0
            else:
                raise RuntimeError(
                    "THC h5 has no Qpts_ibz; only q_IBZ=0 (Gamma) is supported "
                    "without it."
                )

        P_arr = np.asarray(P_idx, dtype=int)
        Q_arr = np.asarray(Q_idx, dtype=int)
        NP = P_arr.shape[0]
        NQ = Q_arr.shape[0]
        B = NP * NQ
        Nw = self._Nw
        NO = self._NO
        Nt = self._Nt

        # FINUFFT plans:
        #   forward: type-1, ω → t with isign=+1, n_modes=Nt, n_pts=Nw, ntrans=B
        #   inverse: type-2, t → Ω with isign=+1, n_modes=Nt, n_pts=NΩ, ntrans=B
        # (matching the C++ conv plans: both with iflag=+1.)
        plan_w_fwd = Plan(1, (Nt,), n_trans=B, isign=+1, eps=self._eps)
        plan_w_fwd.setpts(self._x_w.astype(np.float64))
        plan_O_inv = Plan(2, (Nt,), n_trans=B, isign=+1, eps=self._eps)
        plan_O_inv.setpts(self._x_O.astype(np.float64))

        ns = self._ck.ns
        Nk_FBZ = self._Nk_FBZ
        k_weight = 1.0 / float(Nk_FBZ)
        # qminus → "q-" used in qk_to_k2 lookup of k+q.
        qminus_q = int(self._qminus[q_FBZ])

        # Loop over (s, k_FBZ) FBZ pairs and accumulate.
        ImPi_qPQ = np.zeros((NP, NQ, NO), dtype=float)
        for s in range(ns):
            for ik in range(Nk_FBZ):
                ikq = int(self._qk_to_k2[qminus_q, ik])
                A_k = self.project_orbital_to_aux(s, ik, P_arr, Q_arr)
                A_kq = self.project_orbital_to_aux(s, ikq, P_arr, Q_arr)
                ImPi_qPQ += self._kernel_one_kq(
                    A_k, A_kq, plan_w_fwd, plan_O_inv, k_weight,
                )
        RePi_qPQ = None
        if hilbert:
            RePi_qPQ = self._hilbert_block(ImPi_qPQ)
        return ImPi_qPQ, RePi_qPQ

    def _hilbert_block(self, ImPi: np.ndarray) -> np.ndarray:
        """Hilbert transform of ImPi(Ω) over the bosonic grid → RePi(Ω).

        Mirrors RePi_from_ImPi in real_axis_pi.hpp:
          F(t) = ∫ dΩ e^{+iΩt} ImPi(Ω) w_Ω(Ω)
          K(t) = i*sgn(t) * F(t)        (Hilbert kernel)
          RePi(Ω) = ∫ dt e^{+iΩt} K(t) * (dt/2π)
        """
        from finufft import Plan

        NP, NQ, NO = ImPi.shape
        B = NP * NQ
        Nt = self._Nt
        # Forward (Ω → t) and inverse (t → Ω) on the bosonic grid.
        plan_O_fwd = Plan(1, (Nt,), n_trans=B, isign=+1, eps=self._eps)
        plan_O_fwd.setpts(self._x_O.astype(np.float64))
        plan_O_inv = Plan(2, (Nt,), n_trans=B, isign=+1, eps=self._eps)
        plan_O_inv.setpts(self._x_O.astype(np.float64))

        # Build i*sgn(t) on the uniform time grid.
        sgn = np.zeros(Nt, dtype=float)
        mid = Nt // 2
        sgn[:mid] = -1.0
        sgn[mid + 1:] = +1.0
        kernel = 1j * sgn  # (Nt,) complex

        # Apply quadrature weights to ImPi over Ω.
        wq = self._O_weights
        ImPi_w = ImPi.reshape(B, NO) * wq[None, :]
        F = plan_O_fwd.execute(ImPi_w.astype(np.complex128))   # (B, Nt) cplx
        F *= kernel[None, :]
        Re_raw = plan_O_inv.execute(F)                          # (B, NO) cplx
        # The forward direction here is the same iflag=+1; we follow the
        # convention used in conv.hilbert: result *= dt / (2π) — i.e., the
        # nufft_scale applied once.
        return (Re_raw.real * self._nufft_scale).reshape(NP, NQ, NO)

    def reconstruct_ImPi_diag(
        self,
        q_IBZ: int,
        P_list: Sequence[int],
        *,
        hilbert: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Im Pi_{PP}(q_IBZ, Omega) for a list of P values (diagonal only).

        Returns
        -------
        ImPi : (NP, NΩ) real
        RePi : (NP, NΩ) real or None
        """
        Im, Re = self.reconstruct_Pi_PQ_block(
            q_IBZ, P_list, P_list, hilbert=hilbert,
        )
        # Take the diagonal P=Q.
        idx = np.arange(len(P_list))
        ImD = Im[idx, idx, :]
        ReD = Re[idx, idx, :] if Re is not None else None
        return ImD, ReD

    # ---------------------------------------------------------------- plot
    def plot_Pi(
        self,
        q_IBZ: int,
        pairs: Sequence[tuple[int, int]],
        ax=None,
        hilbert: bool = False,
        in_eV: bool = True,
        **kwargs,
    ):
        """Plot Im Pi_PQ(Omega) (and optionally Re Pi) for a few (P, Q).

        Parameters
        ----------
        q_IBZ : int
            IBZ q-index.
        pairs : list[tuple[int, int]]
            List of (P, Q) pairs to plot.
        hilbert : bool
            If True, also plot Re Pi on a twin axis.
        in_eV : bool
            Convert the Omega axis to eV.
        ax : matplotlib axes or None
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        scale = _HARTREE_TO_EV if in_eV else 1.0
        x = self._Omega * scale
        ax.set_xlabel("Ω [eV]" if in_eV else "Ω [Ha]")
        ax.set_ylabel(r"Im $\Pi_{PQ}(\Omega)$")
        ax2 = ax.twinx() if hilbert else None
        if hilbert:
            ax2.set_ylabel(r"Re $\Pi_{PQ}(\Omega)$")

        for (P, Q) in pairs:
            ImPi, RePi = self.reconstruct_Pi_PQ(q_IBZ, P, Q, hilbert=hilbert)
            ax.plot(x, ImPi, label=f"P={P} Q={Q} (Im)", **kwargs)
            if hilbert:
                ax2.plot(x, RePi, ls="--", alpha=0.6,
                         label=f"P={P} Q={Q} (Re)", **kwargs)

        ax.set_title(f"Polarization Π (q_IBZ={q_IBZ}, iter={self._ck.iter})")
        ax.legend(fontsize=8, loc="upper right")
        if hilbert:
            ax2.legend(fontsize=8, loc="lower right")
        return (ax, ax2) if hilbert else ax

    # ---------------------------------------------------------------- cleanup
    def close(self) -> None:
        self._thc.close()

    def __enter__(self) -> "RealAxisPiReconstructor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
