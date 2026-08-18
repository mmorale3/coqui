"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

"""
Retardation handling for the EDMFT impurity problem, plus the Q4-c causality
monitor for the bosonic Weiss field / local interaction U(iν).

This module is deliberately **numpy-only** at import time: it carries the two
Q4 diagnostics (``casula_werner_zb`` and ``u_causality_metrics``) that must be
unit-checkable on a host where neither ``triqs`` nor the compiled ``coqui``
package is importable (gate Q4-p2, notes/q4_edmft_skeleton_spec.md §3 P1).
Only ``app_log`` is imported from CoQuí, and that import is guarded.
"""
import numpy as np

try:  # pragma: no cover - the fallback only fires on a CoQuí-less host
    from coqui import app_log
except ImportError:  # pragma: no cover
    def app_log(level, msg, *args, **kwargs):
        """Minimal stand-in so the numpy-only unit checks can run standalone."""
        pass


# --------------------------------------------------------------------------
# Basis reductions (local copies of the weiss.py conventions; weiss.py cannot
# be imported here because it pulls in h5/TRIQS).
# --------------------------------------------------------------------------

def _product_basis_to_density_density(A_abcd):
    """A_{...abcd} (product basis) -> A_{...ab} = A_{...aabb}. Mirrors
    ``coqui.dmft.weiss.product_basis_to_density_density``."""
    n1 = A_abcd.shape[-1]
    lead = A_abcd.shape[:-4]
    A_ab = np.zeros(lead + (n1, n1), dtype=A_abcd.dtype)
    for a in range(n1):
        for b in range(n1):
            A_ab[..., a, b] = A_abcd[..., a, a, b, b]
    return A_ab


def to_density_density(U_w):
    """
    Reduce an interaction array to its density-density block ``U_{ab}(iν)``.

    Parameters
    ----------
    U_w : ndarray, shape (nw,), (nw, norb, norb) or (nw, norb, norb, norb, norb)
        Interaction on a bosonic Matsubara mesh. A scalar-per-frequency array is
        promoted to a 1x1 matrix; a product-basis (4-index) array is contracted
        as ``U_{ab} = U_{aabb}``; a 2-index array is taken as already
        density-density.

    Returns
    -------
    ndarray, shape (nw, norb, norb)
    """
    U_w = np.asarray(U_w)
    if U_w.ndim == 1:
        return U_w.reshape(-1, 1, 1)
    if U_w.ndim == 3:
        return U_w
    if U_w.ndim == 5:
        return _product_basis_to_density_density(U_w)
    raise ValueError(
        "to_density_density: U_w must have 1, 3 or 5 dimensions "
        f"(got ndim = {U_w.ndim})."
    )


def total_density_channel(U_w):
    r"""
    Project an interaction onto the **total-density** (uniform) bosonic channel.

    The retarded density-density action is
    :math:`\frac12\sum_{ab} U_{ab}(i\nu)\, n_a n_b`; the mode that couples to the
    total charge :math:`n=\sum_a n_a` (the mode whose screening renormalises the
    bandwidth) carries the coupling
    :math:`\bar U(i\nu) = N^{-2}\sum_{ab} U_{ab}(i\nu)`, i.e. the plain mean of
    the density-density matrix.

    Parameters
    ----------
    U_w : ndarray
        Any shape accepted by :func:`to_density_density`.

    Returns
    -------
    ndarray, shape (nw,), real
        :math:`\bar U(i\nu_n)`. The imaginary part is discarded (a
        density-density U on a ph-symmetric bosonic mesh is real).
    """
    U_ab = to_density_density(U_w)
    return np.real(np.mean(U_ab, axis=(-2, -1)))


# --------------------------------------------------------------------------
# Casula-Werner bandwidth renormalisation Z_B  (ruling R-Q4-5, gate Q4-p2)
# --------------------------------------------------------------------------

def casula_werner_exponent(U_w, nu_mesh, *, tail=True, clamp=True, name=""):
    r"""
    Casula–Werner exponent :math:`S = \sum_l \lambda_l^2/\omega_l^2` of the
    retarded part of ``U(iν)``, evaluated directly on the Matsubara axis.

    **Derivation (no pole fit is performed).** Writing the retarded interaction
    in its bosonic-bath form,

    .. math::
        U(i\nu) = U_\infty - \sum_l \frac{2\lambda_l^2\,\omega_l}{\nu^2+\omega_l^2},

    a one-line partial-fraction identity gives, for every single mode,

    .. math::
        \frac{U(i\nu)-U(0)}{\nu^2}
          = \frac{2\lambda_l^2}{\omega_l\,(\nu^2+\omega_l^2)},
        \qquad
        \int_0^\infty\! d\nu\;\frac{U(i\nu)-U(0)}{\nu^2}
          = \pi\,\frac{\lambda_l^2}{\omega_l^2},

    so that **exactly**

    .. math::
        S \;=\; \sum_l \frac{\lambda_l^2}{\omega_l^2}
          \;=\; \frac{1}{\pi}\int_0^\infty\! d\nu\;
                \frac{U(i\nu)-U(i\nu=0)}{\nu^2}.

    This is the Matsubara-axis twin of the real-axis Casula form
    :math:`S = -\pi^{-1}\int_0^\infty d\omega\,\mathrm{Im}\,U(\omega)/\omega^2`
    (Casula *et al.*, PRB **85**, 035115 (2012)), and it needs neither
    :math:`U_\infty` nor an analytic continuation.

    Properties used as gate Q4-p2:

    * ``U(iν)`` ν-independent  ⇒  every integrand sample is identically zero ⇒
      ``S = 0`` **exactly** (bitwise) ⇒ ``Z_B = 1``.
    * ``U(iν) ≥ U(0)`` for all ν (bosonic causality: screening lifts U from its
      static value towards the bare one) ⇒ the integrand is non-negative ⇒
      ``S ≥ 0`` ⇒ ``0 < Z_B ≤ 1``.

    Parameters
    ----------
    U_w : ndarray
        ``U(iν_n)`` on the **non-negative** bosonic Matsubara mesh. Any shape
        accepted by :func:`to_density_density`; it is reduced to the
        total-density channel by :func:`total_density_channel`.
    nu_mesh : ndarray, shape (nw,)
        Bosonic Matsubara frequencies :math:`\nu_n = 2\pi n/\beta \ge 0`,
        ascending, with ``nu_mesh[0] == 0``.
    tail : bool, default True
        Add the analytic :math:`\int_{\nu_N}^\infty (U(i\nu_N)-U(0))/\nu^2\,d\nu
        = (U(i\nu_N)-U(0))/\nu_N` tail beyond the last mesh point.
    clamp : bool, default True
        Clamp a (numerically) negative ``S`` to zero and warn, so the
        ``0 < Z_B ≤ 1`` contract of R-Q4-5 always holds.
    name : str, optional
        Label used in warnings.

    Returns
    -------
    float
        ``S``. ``Z_B = exp(-S)``.

    Notes
    -----
    The ν-integral is evaluated by the trapezoid rule **on the mesh supplied**.
    On the sparse IR/DLR bosonic meshes used by CoQuí this is a quadrature
    approximation, not an exact sum; its accuracy is set by the mesh, and it is
    exact (identically zero) in the flat-U limit that gate Q4-p2 pins.
    """
    nu = np.asarray(nu_mesh, dtype=float).reshape(-1)
    u = total_density_channel(U_w)
    if u.shape[0] != nu.shape[0]:
        raise ValueError(
            f"casula_werner_exponent: U_w has {u.shape[0]} frequencies but "
            f"nu_mesh has {nu.shape[0]}."
        )
    if nu.shape[0] < 3:
        raise ValueError("casula_werner_exponent: need at least 3 bosonic frequencies.")
    if not np.all(np.diff(nu) > 0.0):
        raise ValueError("casula_werner_exponent: nu_mesh must be strictly ascending.")
    if nu[0] != 0.0:
        raise ValueError(
            "casula_werner_exponent: nu_mesh must start at nu = 0 (pass the "
            "positive-only bosonic mesh)."
        )

    du = u - u[0]
    integrand = np.zeros_like(nu)
    integrand[1:] = du[1:] / nu[1:] ** 2

    # g(nu) is smooth and even in nu, so g(0) follows from a linear fit in nu^2
    # through the two lowest non-zero nodes. (Flat U -> both nodes are 0 -> g(0) = 0.)
    x1, x2 = nu[1] ** 2, nu[2] ** 2
    integrand[0] = integrand[1] + (integrand[2] - integrand[1]) * (0.0 - x1) / (x2 - x1)

    S = float(np.trapezoid(integrand, nu)) if hasattr(np, "trapezoid") \
        else float(np.trapz(integrand, nu))
    if tail:
        S += float(du[-1] / nu[-1])
    S /= np.pi

    if S < 0.0:
        if abs(S) > 1e-12:
            app_log(1, f"WARNING: Casula-Werner exponent S = {S:.6e} < 0 for {name} "
                       f"-- U(inu) is not bosonic-causal (U(inu) < U(0) somewhere). "
                       f"{'Clamping to 0.' if clamp else 'NOT clamped.'}")
        if clamp:
            S = 0.0
    return S


def casula_werner_zb(U_w, nu_mesh, *, tail=True, clamp=True, name=""):
    r"""
    Casula–Werner bandwidth renormalisation factor
    :math:`Z_B = \exp[-\sum_l \lambda_l^2/\omega_l^2]` of ``U(iν)``.

    See :func:`casula_werner_exponent` for the exponent, its exact
    Matsubara-axis representation, and the gate Q4-p2 properties
    (flat ``U`` ⇒ ``Z_B == 1`` exactly; ``0 < Z_B ≤ 1`` always).

    Parameters
    ----------
    U_w, nu_mesh, tail, clamp, name
        As in :func:`casula_werner_exponent`.

    Returns
    -------
    float
        ``Z_B`` in ``(0, 1]``.
    """
    return float(np.exp(-casula_werner_exponent(
        U_w, nu_mesh, tail=tail, clamp=clamp, name=name)))


def apply_impurity_retardation_mode(delta_iw, u_weiss_iw, Vloc, nu_mesh,
                                    mode="dynamic", *, static_u_source="u0", name=""):
    r"""
    Apply the impurity retardation policy of ruling R-Q4-5 to the solver inputs.

    Parameters
    ----------
    delta_iw : ndarray, shape (nw_f, nspin, norb, norb)
        Hybridization function.
    u_weiss_iw : ndarray, shape (nw_b_half, norb, norb, norb, norb)
        Bosonic Weiss field in the product basis (the **retarded** part; the
        full local interaction is ``U(iν) = Vloc + u_weiss_iw(iν)``).
    Vloc : ndarray, shape (norb, norb, norb, norb)
        Bare local interaction in the product basis. Because
        ``u_weiss_iw(iν → ∞) → 0`` this is exactly ``U(iν → ∞)``.
    nu_mesh : ndarray, shape (nw_b_half,)
        Non-negative bosonic Matsubara frequencies matching ``u_weiss_iw``.
    mode : {"dynamic", "static_u_zb"}, default "dynamic"
        - ``"dynamic"``: pass-through. The solver sees the full retarded
          ``U(iν)``. This is the pre-Q4 behaviour and the default.
        - ``"static_u_zb"``: **impurity mode (a)**. The retarded part is set to
          zero, the solver's static interaction is chosen by
          ``static_u_source``, and the hybridization is renormalised as
          ``Δ → Z_B Δ`` with the Casula–Werner factor of
          :func:`casula_werner_zb`.
    static_u_source : {"u0", "u_inf"}, default "u0"
        Which column of ``U(iν)`` becomes the solver's static interaction
        (mode ``"static_u_zb"`` only).

        - ``"u0"`` (**default**): the *screened* static value
          ``U(iν = 0) = Vloc + u_weiss_iw[0]``. This is the Casula–Werner
          standard.
        - ``"u_inf"``: the *unscreened* high-frequency limit
          ``U(iν → ∞) = Vloc``, i.e. the PDF §3.3 literal, kept selectable.

        ⚠ **The default deliberately contradicts the PDF §3.3 text as
        written** (R-Q4-5 AMENDMENT, ``notes/q4_edmft_skeleton_spec.md`` §2).
        ``Z_B`` is derived by integrating out the *screening* bosons, so the
        static interaction that survives that construction is the screened
        ``U(0)``; pairing the bare ``U(iν → ∞)`` with ``Z_B < 1`` double-counts
        screening. If ``U(iν → ∞)`` was intended, set
        ``static_u_source="u_inf"`` explicitly.
    name : str, optional
        Label used in log lines.

    Returns
    -------
    (delta_out, Vloc_out, u_weiss_out, z_b)
        ``z_b`` is ``None`` in ``"dynamic"`` mode, where all three arrays are
        the **same objects** that were passed in (no copy, so the default path
        is bit-identical to the pre-Q4 workflow).

    Notes
    -----
    ``Z_B`` itself is computed from the full ``U(iν)`` in both cases: it depends
    only on the retarded part, not on which static column is handed over.

    The caller must keep using the **unmodified** ``Vloc`` and ``u_weiss_iw``
    for the double-counting terms and for the checkpointed solver inputs; only
    the objects handed to the impurity solver are transformed here.
    """
    if mode == "dynamic":
        return delta_iw, Vloc, u_weiss_iw, None
    if mode != "static_u_zb":
        raise ValueError(
            f"apply_impurity_retardation_mode: unknown mode {mode!r}. "
            f"Valid options: \"dynamic\", \"static_u_zb\"."
        )
    if static_u_source not in {"u0", "u_inf"}:
        raise ValueError(
            f"apply_impurity_retardation_mode: unknown static_u_source "
            f"{static_u_source!r}. Valid options: \"u0\", \"u_inf\"."
        )

    U_w = combine_static_and_retarded_u(Vloc, u_weiss_iw)
    z_b = casula_werner_zb(U_w, nu_mesh, name=name or "U(inu)")

    if static_u_source == "u0":
        if U_w.shape[1:] != np.shape(Vloc):
            raise ValueError(
                f"apply_impurity_retardation_mode: static_u_source=\"u0\" needs "
                f"u_weiss_iw in the same basis as Vloc, got u_weiss_iw"
                f"{np.shape(u_weiss_iw)} and Vloc{np.shape(Vloc)}."
            )
        Vloc_out = U_w[0]
        static_label = "U(inu = 0) [screened, Casula-Werner standard]"
    else:
        Vloc_out = Vloc
        static_label = "U(inu -> infty) = Vloc [unscreened, PDF section 3.3 literal]"

    u_bar = total_density_channel(U_w)
    app_log(1, "Impurity retardation mode (a): static U + Casula-Werner bandwidth factor")
    app_log(1, "------------------------------------------------------------------------")
    app_log(1, f"  static_u_source            = {static_u_source}  ->  {static_label}")
    app_log(1, f"  Ubar(inu = 0)              = {u_bar[0]:.8f}")
    app_log(1, f"  Ubar(inu_max)              = {u_bar[-1]:.8f}")
    app_log(1, f"  Z_B                        = {z_b:.8f}")
    app_log(1, f"  Casula-Werner exponent S   = {-np.log(z_b):.8e}\n")
    if static_u_source == "u_inf":
        app_log(1, "WARNING: static_u_source = \"u_inf\" pairs the UNSCREENED interaction "
                   "with Z_B < 1, which double-counts screening (R-Q4-5 AMENDMENT). "
                   "The Casula-Werner standard is static_u_source = \"u0\".")

    # ---- domain-of-validity meters (diagnostic only; nothing below changes the result) ---
    # R-Q4-5 guarantees 0 < Z_B <= 1, and casula_werner_zb clamps S < 0. Neither guards the
    # OTHER end: an overscreened U(inu) makes S large and Z_B underflow, and Delta -> Z_B
    # Delta then annihilates the hybridization -- the impurity silently decouples from the
    # bath and the QMC solves an atomic-limit problem that looks perfectly healthy (sign
    # ~ 1). Measured on SVO kp444 beta=1000 with the qpGW lattice stage, 2026-08-17:
    # Ubar(0) = -0.196 Ha, S = 97.0, Z_B = 8e-43.
    if z_b < 1e-3:
        app_log(1, f"WARNING: Z_B = {z_b:.3e} for {name or 'this impurity'} is effectively "
                   f"ZERO (Casula-Werner exponent S = {-np.log(z_b):.4g}). Mode (a) hands the "
                   f"solver Delta -> Z_B Delta, so the impurity is DECOUPLED from the bath and "
                   f"the impurity solution is an atomic-limit artefact, not an EDMFT solution. "
                   f"This means U(inu) is far outside the weakly-retarded regime mode (a) "
                   f"assumes; use retardation=\"dynamic\", or fix the screening first (see the "
                   f"Q4-c causality monitor above).")
    if np.real(u_bar[0]) <= 0.0 and static_u_source == "u0":
        app_log(1, f"WARNING: Ubar(inu = 0) = {np.real(u_bar[0]):.6f} a.u. <= 0 for "
                   f"{name or 'this impurity'}: the SCREENED static interaction handed to the "
                   f"solver by static_u_source = \"u0\" is ATTRACTIVE. That is an overscreening "
                   f"artefact of W, not a physical Hubbard U; the mode (a) result is not "
                   f"trustworthy until the screening is fixed.")

    return delta_iw * z_b, Vloc_out, np.zeros_like(u_weiss_iw), z_b


# --------------------------------------------------------------------------
# Gate Q4-c: causality monitor for U(iν)
# --------------------------------------------------------------------------

CAUSALITY_TRAIL_LABELS = (
    "hermiticity_max",          # (i)   max_nu ||U(nu) - U(nu)^dag||_max
    "dd_monotonicity_flips",    # (ii)  max_a #sign changes of dU_aa/dnu
    "u0_minus_umax_min_eig",    # (iii) min eig of herm[U(0) - U(nu_max)]
    "u0_minus_umax_max_eig",    # (iii) max eig of herm[U(0) - U(nu_max)]  (<= 0 if causal)
)


def u_causality_metrics(U_w, nu_mesh, *, monotonicity_tol=0.0, psd_rtol=1e-10):
    r"""
    Q4-c causality diagnostics for the local interaction ``U(iν)``.

    All three meters of notes/q4_edmft_skeleton_spec.md §3 P1 are **non-fatal**;
    the caller logs them and stores them in the DMFT checkpoint trail.

    (i)   **Hermiticity in the product basis.** ``U(iν)`` reshaped to the
          ``(norb², norb²)`` pair matrix must be hermitian at every ν;
          ``hermiticity_max`` is ``max_ν ‖U(ν) − U(ν)†‖_max``.

    (ii)  **Monotonicity of the density-density diagonal.** A bosonic-causal
          ``U_aa(iν)`` rises monotonically from its screened static value
          ``U_aa(0)`` towards the bare ``U_aa(∞)``, so the discrete derivative
          ``dU_aa/dν`` keeps one sign. ``dd_monotonicity_flips`` is the largest
          number of sign changes over the orbitals; ``> 1`` is flagged.

    (iii) **Static-vs-asymptotic spread.** ``U(0) − U(iν_max)`` is negative
          semi-definite for a causally screened interaction, so its **largest**
          eigenvalue should be ``≤ 0``. Both the min and max eigenvalue of the
          hermitian part are returned.

    Parameters
    ----------
    U_w : ndarray, shape (nw, norb, norb, norb, norb) or (nw, n, n)
        Local interaction on the non-negative bosonic mesh (product basis
        preferred; a 2-index array is treated as already pair-shaped).
    nu_mesh : ndarray, shape (nw,)
        Non-negative bosonic Matsubara frequencies, ascending.
    monotonicity_tol : float, default 0.0
        Increments of ``|dU_aa/dν|`` at or below this magnitude are treated as
        flat and ignored by the sign-change count.
    psd_rtol : float, default 1e-10
        Relative tolerance for meter (iii). The pair matrix carries structurally
        **zero** rows (every pair index outside the density-density set that the
        density-only impurity solver populates), so its largest eigenvalue is 0
        up to eigensolver roundoff; only ``max_eig > psd_rtol * ‖·‖`` counts as
        a violation. ``u0_minus_umax_max_eig`` itself is reported raw.

    Returns
    -------
    dict
        Keys: ``hermiticity_max``, ``dd_monotonicity_flips``,
        ``dd_monotonicity_flagged`` (bool), ``u0_minus_umax_min_eig``,
        ``u0_minus_umax_max_eig``, ``u0_minus_umax_psd_violation`` (float, the
        amount by which the max eigenvalue exceeds the tolerance; ``0.0`` when
        clean), ``causality_violated`` (bool).
    """
    U_w = np.asarray(U_w)
    nu = np.asarray(nu_mesh, dtype=float).reshape(-1)
    if U_w.shape[0] != nu.shape[0]:
        raise ValueError(
            f"u_causality_metrics: U_w has {U_w.shape[0]} frequencies but "
            f"nu_mesh has {nu.shape[0]}."
        )

    # ---- pair-space (product-basis) matrix, and the density-density block ----
    if U_w.ndim == 5:
        norb = U_w.shape[-1]
        U_pair = U_w.reshape(U_w.shape[0], norb * norb, norb * norb)
        U_dd = _product_basis_to_density_density(U_w)
    elif U_w.ndim == 3:
        U_pair = U_w
        U_dd = U_w
    else:
        raise ValueError(
            f"u_causality_metrics: U_w must have 3 or 5 dimensions (got {U_w.ndim})."
        )

    # (i) hermiticity
    herm_dev = U_pair - np.conjugate(np.swapaxes(U_pair, -1, -2))
    herm_max = float(np.max(np.abs(herm_dev))) if herm_dev.size else 0.0

    # (ii) monotonicity of the density-density diagonal along nu
    diag = np.real(np.diagonal(U_dd, axis1=-2, axis2=-1))  # (nw, norb)
    flips = 0
    for a in range(diag.shape[1]):
        d = np.diff(diag[:, a])
        s = np.sign(d)
        s = s[np.abs(d) > monotonicity_tol]
        if s.size > 1:
            flips = max(flips, int(np.count_nonzero(s[1:] != s[:-1])))

    # (iii) eigenvalues of the hermitian part of U(0) - U(nu_max)
    D = U_pair[0] - U_pair[-1]
    D = 0.5 * (D + np.conjugate(D.T))
    eig = np.linalg.eigvalsh(D)
    min_eig, max_eig = float(eig[0]), float(eig[-1])
    psd_thresh = psd_rtol * max(abs(min_eig), abs(max_eig), 1.0)
    psd_violation = max(0.0, max_eig - psd_thresh)

    return {
        "hermiticity_max": herm_max,
        "dd_monotonicity_flips": flips,
        "dd_monotonicity_flagged": bool(flips > 1),
        "u0_minus_umax_min_eig": min_eig,
        "u0_minus_umax_max_eig": max_eig,
        "u0_minus_umax_psd_violation": psd_violation,
        "causality_violated": bool(flips > 1 or psd_violation > 0.0),
    }


def combine_static_and_retarded_u(Vloc, u_weiss_iw):
    """
    Assemble ``U(iν) = Vloc + u_weiss_iw(iν)``, reconciling the two shape
    conventions in use: ``u_weiss_iw`` is normally in the product basis
    ``(nw, norb, norb, norb, norb)`` matching ``Vloc``, but the density-density
    form ``(nw, norb, norb)`` also occurs (cf. the ``ndim == 3`` branches of
    ``scf_driver._edmft_convergence_check``), in which case ``Vloc`` is reduced
    to its density-density block first.
    """
    V = np.asarray(Vloc)
    U = np.asarray(u_weiss_iw)
    if U.ndim == V.ndim + 1:
        return U + V[np.newaxis, ...]
    if U.ndim == 3 and V.ndim == 4:
        return U + _product_basis_to_density_density(V)[np.newaxis, ...]
    raise ValueError(
        f"combine_static_and_retarded_u: incompatible shapes "
        f"u_weiss_iw{U.shape} and Vloc{V.shape}."
    )


def causality_trail(metrics):
    """Pack :func:`u_causality_metrics` output into the float array stored in
    the DMFT checkpoint (``dmft/iter{n}/impurity_{i}/results/causality``), in
    the order of :data:`CAUSALITY_TRAIL_LABELS`."""
    return np.array([float(metrics[k]) for k in CAUSALITY_TRAIL_LABELS])


def monitor_u_causality(Vloc, u_weiss_iw, nu_mesh, *, imp_index=None, verbose=True):
    """
    Run the Q4-c causality monitor on ``U(iν) = Vloc + u_weiss_iw(iν)`` and log it.

    Parameters
    ----------
    Vloc : ndarray, shape (norb, norb, norb, norb)
        Bare local interaction in the product basis.
    u_weiss_iw : ndarray, shape (nw, norb, norb, norb, norb)
        Retarded part of the local interaction (bosonic Weiss field) on the
        non-negative bosonic Matsubara mesh.
    nu_mesh : ndarray, shape (nw,)
        Non-negative bosonic Matsubara frequencies, ascending.
    imp_index : int, optional
        Impurity label used in the log lines.
    verbose : bool, default True
        Emit the log block (warnings are emitted regardless).

    Returns
    -------
    dict
        The :func:`u_causality_metrics` dictionary.
    """
    metrics = u_causality_metrics(combine_static_and_retarded_u(Vloc, u_weiss_iw), nu_mesh)

    tag = "" if imp_index is None else f" for impurity {imp_index}"
    if verbose:
        app_log(1, f"Causality monitor for U(inu){tag} (Q4-c, non-fatal)")
        app_log(1, "-------------------------------------------------------")
        app_log(1, f"  max_nu |U(nu) - U(nu)^dag|            = {metrics['hermiticity_max']:.6e}")
        app_log(1, f"  max_a #sign changes of dU_aa/dnu      = {metrics['dd_monotonicity_flips']}")
        app_log(1, f"  min eig[U(0) - U(inu_max)]            = {metrics['u0_minus_umax_min_eig']:.6e}")
        app_log(1, f"  max eig[U(0) - U(inu_max)]            = {metrics['u0_minus_umax_max_eig']:.6e}\n")

    if metrics["dd_monotonicity_flagged"]:
        app_log(1, f"WARNING: U_aa(inu){tag} changes slope more than once "
                   f"({metrics['dd_monotonicity_flips']} sign changes of dU_aa/dnu). "
                   f"This is not a bosonic-causal screened interaction.")
    if metrics["u0_minus_umax_psd_violation"] > 0.0:
        app_log(1, f"WARNING: U(0) - U(inu_max){tag} is not negative semi-definite "
                   f"(max eigenvalue = {metrics['u0_minus_umax_max_eig']:.6e}). "
                   f"Screening is anti-causal in at least one pair channel.")

    return metrics
