"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

Reader + plotter for real-axis GW checkpoint files (`{output}.mbpt.h5`)
produced by `[real_axis_qpgw]` (Phase 1) and `[real_axis_gw]` (Phase 2)
when `write_chkpt = true`.

Layout summary (real-axis Dyson scGW):
    /system/                MF metadata (kpoints, k_weight, H0, S, ...)
    /mean_field/eigvals
    /real_frequency_grid/   omega, Omega, t, beta, mu_chem, ...
    /scf/final_iter
    /scf/iterN/A_wskij              (Nw, ns, Nk_ibz, nbnd, nbnd) cplx
    /scf/iterN/ImSigma_wskij        same shape
    /scf/iterN/ReSigma_wskij        same shape
    /scf/iterN/Sigma_x_skij         (ns, Nk_ibz, nbnd, nbnd) cplx
    /scf/iterN/Dm_skij              same shape
    /scf/iterN/mu                   double
    /scf/iterN/axis = "real"

Layout summary (real-axis QSGW):
    /system/, /mean_field/, /real_frequency_grid/  (same as scGW)
    /scf/iterN/{Dm_skij, Heff_skij, MO_skia, E_ska, mu}

Spectral function A is stored componentwise as -(i/π) G^R; the matrix-
hermitian DOS-like quantity is `0.5 * (A + A^H)`. For diagonal bands
the imag part of the off-diagonal is zero and `A_nn(ω).real` is the
direct DOS contribution.
==========================================================================
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


_HARTREE_TO_EV = 27.211386245988


def _read_real(ds, *idx) -> np.ndarray:
    """Read the real component of an nda complex array stored as
    (..., 2) double on disk. Last axis: 0=real, 1=imag.

    `idx` is the slice into the leading dimensions (excluding the
    trailing real/imag axis).
    """
    return np.asarray(ds[(*idx, 0)], dtype=float)


def _read_imag(ds, *idx) -> np.ndarray:
    """Read the imag component of an nda complex array."""
    return np.asarray(ds[(*idx, 1)], dtype=float)


def _read_complex(ds, *idx) -> np.ndarray:
    """Read both components and combine into a complex array."""
    re = ds[(*idx, 0)]
    im = ds[(*idx, 1)]
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


class RealAxisChkpt:
    """Lazy reader for a real-axis GW chkpt h5 file.

    Use it like::

        ck = RealAxisChkpt("si_scgw.mbpt.h5")
        omega = ck.omega_abs           # absolute ω (Ha), shifted by mu_chem
        A = ck.A_diag(s=0, k=0, n=4)   # spectral function for one band
        ck.plot_DOS()                  # k-summed total DOS

    Notes
    -----
    - All data is fetched via lazy h5py reads; no full A/Σ array is loaded
      until you ask for it.
    - For QSGW chkpt files (which only store H_eff/MO/E/Dm), the A_*
      methods raise; only `is_qsgw()` and the metadata accessors work.
    """

    def __init__(self, path: str, iter: Optional[int] = None):
        import h5py

        self._path = path
        self._h5 = h5py.File(path, "r")
        # Validate top-level structure.
        if "scf" not in self._h5:
            raise ValueError(f"{path}: no /scf group; not a CoQui chkpt")
        if "real_frequency_grid" not in self._h5:
            raise ValueError(
                f"{path}: no /real_frequency_grid group; "
                "this is not a real-axis chkpt (try the imag-axis reader)"
            )

        self._scf = self._h5["scf"]
        self._rfg = self._h5["real_frequency_grid"]
        self._sys = self._h5["system"]

        # Resolve the iter to read.
        final_iter = int(self._scf["final_iter"][()])
        if iter is None:
            iter = final_iter
        if iter < 0:
            iter = final_iter + 1 + iter   # -1 → final_iter
        self._iter = int(iter)
        grp_name = f"iter{self._iter}"
        if grp_name not in self._scf:
            avail = sorted(
                int(k.removeprefix("iter"))
                for k in self._scf.keys()
                if k.startswith("iter")
            )
            raise ValueError(
                f"{path}: /scf/{grp_name} not present. "
                f"Available iters: {avail}"
            )
        self._iter_grp = self._scf[grp_name]

        # Detect mode (real-axis Dyson scGW writes axis="real" + A_wskij;
        # real-axis QSGW reuses the imag-axis QSGW dump_scf which writes
        # only H_eff / MO / E / Dm — no A, no axis marker).
        self._is_scgw = "A_wskij" in self._iter_grp

    # ----------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    @property
    def iter(self) -> int:
        return self._iter

    @property
    def final_iter(self) -> int:
        return int(self._scf["final_iter"][()])

    def iter_list(self) -> list[int]:
        """Available iter indices in /scf/."""
        return sorted(
            int(k.removeprefix("iter"))
            for k in self._scf.keys()
            if k.startswith("iter")
        )

    def is_scgw(self) -> bool:
        """True if /scf/iterN/A_wskij is present (Dyson scGW chkpt)."""
        return self._is_scgw

    def is_qsgw(self) -> bool:
        """True if /scf/iterN/Heff_skij is present (QP-SCF chkpt)."""
        return "Heff_skij" in self._iter_grp

    # Real-frequency grid -----------------------------------------------------
    @property
    def omega(self) -> np.ndarray:
        """Fermionic ω grid relative to mu_chem (Ha). Shape (Nw,)."""
        return self._rfg["omega"][:]

    @property
    def Omega(self) -> np.ndarray:
        """Bosonic Ω grid (Ha, positive half). Shape (NΩ,)."""
        return self._rfg["Omega"][:]

    @property
    def t(self) -> np.ndarray:
        """Conjugate uniform time grid. Shape (N_t,)."""
        return self._rfg["t"][:]

    @property
    def beta(self) -> float:
        return float(self._rfg["beta"][()])

    @property
    def mu_chem(self) -> float:
        """Initial mu_chem stored on the grid (the SCF iter mu is in `mu`)."""
        return float(self._rfg["mu_chem"][()])

    @property
    def mu(self) -> float:
        """SCF mu at this iter."""
        return float(self._iter_grp["mu"][()])

    @property
    def omega_abs(self) -> np.ndarray:
        """Absolute frequency axis = omega + mu_chem (Ha)."""
        return self.omega + self.mu_chem

    @property
    def omega_abs_eV(self) -> np.ndarray:
        return self.omega_abs * _HARTREE_TO_EV

    # System metadata --------------------------------------------------------
    @property
    def ns(self) -> int:
        return int(self._sys["number_of_spins"][()])

    @property
    def nbnd(self) -> int:
        return int(self._sys["number_of_orbitals"][()])

    @property
    def Nk(self) -> int:
        return int(self._sys["number_of_kpoints"][()])

    @property
    def Nk_ibz(self) -> int:
        return int(self._sys["number_of_kpoints_ibz"][()])

    @property
    def k_weight(self) -> np.ndarray:
        """IBZ k-weights, sum to 1."""
        return self._sys["k_weight"][:]

    @property
    def kpts(self) -> np.ndarray:
        """k-points (ibz?). Cartesian coordinates."""
        return self._sys["kpoints"][:]

    @property
    def kpts_crystal(self) -> np.ndarray:
        return self._sys["kpoints_crys"][:]

    # ----------------------------------------------------------------
    # Spectral function accessors (scGW only)
    # ----------------------------------------------------------------
    def _check_scgw(self) -> None:
        if not self._is_scgw:
            raise RuntimeError(
                f"{self._path}: /scf/iter{self._iter}/A_wskij not present. "
                "This is a QSGW chkpt — A is reconstructible from MO/E "
                "via Lorentzian projection (not implemented here)."
            )

    def A_diag(self, s: int, k: int, n: int) -> np.ndarray:
        """A_nn(ω) — diagonal spectral function for band n at (s, k_ibz).

        Returns the real part (the imaginary part is zero by construction
        for the diagonal of A = -(i/π) G^R).
        Shape: (Nw,) real.
        """
        self._check_scgw()
        return _read_real(self._iter_grp["A_wskij"],
                          slice(None), s, k, n, n)

    def A_trace(self, s: int, k: int) -> np.ndarray:
        """tr_band A(ω, s, k) — k-resolved DOS contribution. Shape (Nw,)."""
        self._check_scgw()
        ds = self._iter_grp["A_wskij"]
        # Read real (Nw, nbnd, nbnd) at this (s, k); take diagonal trace.
        a_kn = _read_real(ds, slice(None), s, k, slice(None), slice(None))
        return np.asarray(np.trace(a_kn, axis1=1, axis2=2), dtype=float)

    def A_orbital(self, s: int, k: int, i: int, j: int) -> np.ndarray:
        """A_ij(ω) — full orbital element (complex)."""
        self._check_scgw()
        return _read_complex(self._iter_grp["A_wskij"],
                             slice(None), s, k, i, j)

    def A_kresolved(self, s: int, k: int) -> np.ndarray:
        """A(ω, i, j) at fixed (s, k). Shape (Nw, nbnd, nbnd) complex."""
        self._check_scgw()
        return _read_complex(self._iter_grp["A_wskij"],
                             slice(None), s, k,
                             slice(None), slice(None))

    def DOS(self, s: int = 0) -> np.ndarray:
        """k-summed total DOS = Σ_k w_k tr_band A(ω, s, k). Shape (Nw,)."""
        self._check_scgw()
        kw = self.k_weight
        Nw = self.omega.shape[0]
        out = np.zeros(Nw, dtype=float)
        ds = self._iter_grp["A_wskij"]
        for ik in range(self.Nk_ibz):
            block = _read_real(ds, slice(None), s, ik,
                               slice(None), slice(None))   # (Nw, nbnd, nbnd)
            out += float(kw[ik]) * np.trace(block, axis1=1, axis2=2)
        return out

    # ----------------------------------------------------------------
    # Self-energy accessors
    # ----------------------------------------------------------------
    def Sigma_c_diag(
        self, s: int, k: int, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(Im Σ_c, Re Σ_c) diagonal at band n. Each shape (Nw,) real.

        ImSigma_wskij and ReSigma_wskij are each stored as ComplexType
        nda arrays where the .real component carries the physical value
        of Im Σ_c and Re Σ_c respectively (the .imag component is zero).
        """
        self._check_scgw()
        ims = _read_real(self._iter_grp["ImSigma_wskij"],
                         slice(None), s, k, n, n)
        res = _read_real(self._iter_grp["ReSigma_wskij"],
                         slice(None), s, k, n, n)
        return ims, res

    def Sigma_x_diag(self, s: int, k: int, n: int) -> complex:
        """Static-exchange diagonal at band n (complex)."""
        if "Sigma_x_skij" not in self._iter_grp:
            raise RuntimeError("Sigma_x_skij not in this iter group")
        ds = self._iter_grp["Sigma_x_skij"]
        return complex(float(ds[s, k, n, n, 0]), float(ds[s, k, n, n, 1]))

    # ----------------------------------------------------------------
    # Convergence trajectory across iters (k, n) fixed
    # ----------------------------------------------------------------
    def trajectory_A_diag(
        self, s: int, k: int, n: int, iters: Optional[Sequence[int]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """A_nn(ω) at band n across iters. Shape (Nstored, Nw).

        Returns (iter_labels, A) where iter_labels has shape (Nstored,) and
        A has shape (Nstored, Nw).
        """
        if iters is None:
            iters = self.iter_list()
        labels = []
        rows = []
        for it in iters:
            grp = self._scf[f"iter{it}"]
            if "A_wskij" not in grp:
                continue
            labels.append(it)
            rows.append(_read_real(grp["A_wskij"], slice(None), s, k, n, n))
        return np.asarray(labels), np.stack(rows, axis=0)

    # ----------------------------------------------------------------
    # Plotters
    # ----------------------------------------------------------------
    def plot_A(
        self,
        s: int = 0,
        k: int = 0,
        bands: Optional[Sequence[int]] = None,
        ax=None,
        abs_omega: bool = True,
        in_eV: bool = True,
        **kwargs,
    ):
        """Plot A_nn(ω) for selected bands at (s, k).

        Parameters
        ----------
        bands : list[int] or None
            Band indices to plot. None -> all bands. Pass a small list for
            production-scale data (nbnd=256).
        abs_omega : bool
            If True, x-axis is absolute frequency (ω + μ_chem). Default True.
        in_eV : bool
            Convert x-axis to eV. Default True.
        ax : matplotlib axes or None
        kwargs : forwarded to ax.plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if bands is None:
            bands = list(range(self.nbnd))

        x = self.omega_abs if abs_omega else self.omega
        if in_eV:
            x = x * _HARTREE_TO_EV
        for n in bands:
            ax.plot(x, self.A_diag(s, k, n), label=f"n={n}", **kwargs)

        ax.set_xlabel("ω [eV]" if in_eV else "ω [Ha]")
        ax.set_ylabel(r"$A_{nn}(\omega)$")
        ax.set_title(f"Spectral function (s={s}, k={k}, iter={self._iter})")
        # Mark mu (in chosen units).
        mu = self.mu if abs_omega else (self.mu - self.mu_chem)
        if in_eV:
            mu *= _HARTREE_TO_EV
        ax.axvline(mu, ls="--", color="k", lw=0.8, alpha=0.5, label=r"$\mu$")
        if len(bands) < 12:
            ax.legend(fontsize=8)
        return ax

    def plot_DOS(self, s: int = 0, ax=None, in_eV: bool = True, **kwargs):
        """k-summed total DOS."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        x = self.omega_abs * (_HARTREE_TO_EV if in_eV else 1.0)
        ax.plot(x, self.DOS(s=s), **kwargs)
        ax.set_xlabel("ω [eV]" if in_eV else "ω [Ha]")
        ax.set_ylabel(r"$\sum_k w_k\,\mathrm{tr}_n\,A(\omega, k)$")
        ax.set_title(f"DOS (s={s}, iter={self._iter})")
        mu = self.mu * (_HARTREE_TO_EV if in_eV else 1.0)
        ax.axvline(mu, ls="--", color="k", lw=0.8, alpha=0.5)
        return ax

    def plot_Sigma(
        self, s: int, k: int, n: int, ax=None, in_eV: bool = True, **kwargs
    ):
        """Plot Im Σ_c(ω) on left axis, Re Σ_c(ω) on right twin axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ims, res = self.Sigma_c_diag(s, k, n)
        x = self.omega_abs * (_HARTREE_TO_EV if in_eV else 1.0)
        scale = _HARTREE_TO_EV if in_eV else 1.0
        ax.plot(x, ims * scale, color="C0", label=r"$\mathrm{Im}\,\Sigma_c$", **kwargs)
        ax2 = ax.twinx()
        ax2.plot(x, res * scale, color="C3", label=r"$\mathrm{Re}\,\Sigma_c$", **kwargs)
        ax.set_xlabel("ω [eV]" if in_eV else "ω [Ha]")
        ax.set_ylabel(r"Im $\Sigma_c$ [eV]" if in_eV else r"Im $\Sigma_c$ [Ha]")
        ax2.set_ylabel(r"Re $\Sigma_c$ [eV]" if in_eV else r"Re $\Sigma_c$ [Ha]")
        ax.set_title(
            f"Σ_c (s={s}, k={k}, n={n}, iter={self._iter})"
        )
        return ax, ax2

    def plot_iter_trajectory(
        self, s: int, k: int, n: int, ax=None, in_eV: bool = True, **kwargs
    ):
        """Overlay A_nn(ω) across all stored iters to visualize convergence."""
        import matplotlib.pyplot as plt
        from matplotlib import cm

        if ax is None:
            _, ax = plt.subplots()
        labels, A = self.trajectory_A_diag(s, k, n)
        x = self.omega_abs * (_HARTREE_TO_EV if in_eV else 1.0)
        cmap = cm.get_cmap("viridis", len(labels))
        for i, (it, row) in enumerate(zip(labels, A)):
            ax.plot(x, row, color=cmap(i), label=f"iter {it}",
                    alpha=0.85, **kwargs)
        ax.set_xlabel("ω [eV]" if in_eV else "ω [Ha]")
        ax.set_ylabel(r"$A_{nn}(\omega)$")
        ax.set_title(f"Convergence trajectory (s={s}, k={k}, n={n})")
        if len(labels) < 16:
            ax.legend(fontsize=7, ncol=2)
        return ax

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "RealAxisChkpt":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
