"""
xc_functionals.py -- LDA (PW92) and GGA (PBE) exchange-correlation potentials
on a periodic FFT grid, for abinit2coqui's real vxc/vxc_with_nlcc export
(consumed by CoQui's add_vxc for the G0W0 Sigma - Vxc subtraction).

Spin-unpolarized only (nsppol=1); callers must fail loudly otherwise.

Design: the energy density e(rho, sigma) [Ha/bohr^3, sigma = |grad rho|^2]
is coded analytically from the published functional forms (PW92: Perdew &
Wang PRB 45, 13244 (1992); PBE: Perdew, Burke, Ernzerhof PRL 77, 3865
(1996), same constants as QE's internal 'pbe'); the potential derivatives
    vrho = de/drho,  vsigma = de/dsigma
are taken by high-order central finite differences (relative step 1e-6,
error ~1e-12 relative -- far below grid/quadrature error), and the GGA
divergence term is assembled spectrally:
    vxc(r) = vrho(r) - 2 div[ vsigma(r) grad rho(r) ],
with grad and div evaluated by FFT (iG), matching QE's spectral gradients.

Validation: validate_b2.py rebuilds rho from the wavefunctions of a QE scf
and reproduces pw2coqui's vxc dataset (QE v_xc output) on the same grid.
"""

import numpy as np

# --- constants (QE values) -------------------------------------------------
_PBE_KAPPA = 0.804
_PBE_MU = 0.2195149727645171
_PBE_BETA = 0.06672455060314922
_PBE_GAMMA = 0.031090690869654895034     # (1 - ln 2)/pi^2
# PW92 unpolarized eps_c(rs) parameters
_PW92 = dict(A=0.031091, a1=0.21370, b1=7.5957, b2=3.5876, b3=1.6382, b4=0.49294)

_RHO_FLOOR = 1e-14


def _eps_c_pw92(rs):
    """PW92 unpolarized correlation energy per electron (Ha)."""
    p = _PW92
    srs = np.sqrt(rs)
    den = 2.0 * p["A"] * (p["b1"] * srs + p["b2"] * rs + p["b3"] * rs * srs
                          + p["b4"] * rs * rs)
    return -2.0 * p["A"] * (1.0 + p["a1"] * rs) * np.log1p(1.0 / den)


def exc_density(rho, sigma, functional):
    """XC energy per volume e(rho, sigma) in Ha/bohr^3 (e = rho * eps_xc).

    functional: 'lda_pw' (PW92, sigma ignored) or 'pbe'.
    """
    rho = np.maximum(rho, _RHO_FLOOR)
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    eps_x = -3.0 / (4.0 * np.pi) * kf
    eps_c = _eps_c_pw92(rs)
    if functional == "lda_pw":
        return rho * (eps_x + eps_c)
    if functional != "pbe":
        raise ValueError("unsupported functional '%s' (lda_pw | pbe)" % functional)

    sigma = np.maximum(sigma, 0.0)
    # exchange enhancement F(s), s = |grad rho| / (2 kf rho)
    s2 = sigma / (4.0 * kf ** 2 * rho ** 2)
    Fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    # correlation H(t), t = |grad rho| / (2 ks rho), ks = sqrt(4 kf / pi)
    ks2 = 4.0 * kf / np.pi
    t2 = sigma / (4.0 * ks2 * rho ** 2)
    g, b = _PBE_GAMMA, _PBE_BETA
    expo = np.exp(-eps_c / g)
    A = (b / g) / np.maximum(expo - 1.0, 1e-300)
    num = 1.0 + A * t2
    H = g * np.log1p((b / g) * t2 * num / (num + A * A * t2 * t2))
    return rho * (eps_x * Fx + eps_c + H)


def _fd_derivs(rho, sigma, functional):
    """(vrho, vsigma) = (de/drho, de/dsigma) by central finite differences."""
    hr = 1e-6 * np.maximum(rho, 1e-8)
    vrho = (exc_density(rho + hr, sigma, functional)
            - exc_density(rho - hr, sigma, functional)) / (2.0 * hr)
    if functional == "lda_pw":
        return vrho, np.zeros_like(rho)
    # sigma step: relative where sigma is finite, absolute floor tied to the
    # natural scale (2 kf rho)^2 where sigma ~ 0 (vsigma stays finite there).
    rho_s = np.maximum(rho, _RHO_FLOOR)
    kf2 = (3.0 * np.pi ** 2 * rho_s) ** (2.0 / 3.0)
    hs = 1e-6 * sigma + 1e-12 * (4.0 * kf2 * rho_s ** 2)
    sp = sigma + hs
    sm = np.maximum(sigma - hs, 0.0)
    vsigma = (exc_density(rho, sp, functional)
              - exc_density(rho, sm, functional)) / np.maximum(sp - sm, 1e-300)
    return vrho, vsigma


def vxc_grid(rho_r, recv, functional="pbe"):
    """XC potential vxc(r) on a periodic FFT grid, in HARTREE.

    rho_r : (n1,n2,n3) real electron density (bohr^-3);
    recv  : (3,3) reciprocal vectors, rows b_n with a.b = 2*pi.
    GGA: vxc = vrho - 2 div(vsigma grad rho), gradients spectral (iG).
    """
    rho_r = np.asarray(rho_r, float)
    mesh = rho_r.shape
    if functional == "lda_pw":
        vrho, _ = _fd_derivs(rho_r, np.zeros_like(rho_r), functional)
        return vrho

    # G vectors of the FFT box (numpy fft frequency order)
    freqs = [np.fft.fftfreq(n, d=1.0 / n) for n in mesh]   # integer millers
    M = np.stack(np.meshgrid(*freqs, indexing="ij"), axis=-1)  # (n1,n2,n3,3)
    G = M @ np.asarray(recv, float)                            # cartesian

    rho_g = np.fft.fftn(rho_r)
    grad = np.empty(mesh + (3,), float)
    for d in range(3):
        grad[..., d] = np.real(np.fft.ifftn(1j * G[..., d] * rho_g))
    sigma = np.einsum("...d,...d->...", grad, grad)

    vrho, vsigma = _fd_derivs(rho_r, sigma, functional)

    div = np.zeros(mesh, float)
    for d in range(3):
        fg = np.fft.fftn(vsigma * grad[..., d])
        div += np.real(np.fft.ifftn(1j * G[..., d] * fg))
    return vrho - 2.0 * div


# ABINIT ixc -> functional name (native + libxc negative encodings)
_IXC_MAP = {
    7: "lda_pw",              # native Perdew-Wang 92
    11: "pbe",                # native PBE
    -1012: "lda_pw",          # libxc LDA_X + LDA_C_PW
    -101130: "pbe",           # libxc GGA_X_PBE + GGA_C_PBE
    -130101: "pbe",
}


def functional_from_ixc(ixc):
    """Map an ABINIT ixc to a supported functional name, or raise."""
    try:
        return _IXC_MAP[int(ixc)]
    except KeyError:
        raise NotImplementedError(
            "abinit2coqui vxc: ixc=%s is not supported (supported: %s). "
            "Pass --xc {pbe,lda_pw} explicitly if this ixc is equivalent."
            % (ixc, sorted(set(_IXC_MAP.values()))))


if __name__ == "__main__":
    # LDA sanity: vxc for uniform rho must equal d(rho eps)/drho analytically
    rho0 = 0.03
    v = vxc_grid(np.full((6, 6, 6), rho0), 2 * np.pi * np.eye(3) / 10.0, "lda_pw")
    rs = (3 / (4 * np.pi * rho0)) ** (1 / 3.)
    h = 1e-7 * rho0
    vref = (exc_density(np.array([rho0 + h]), 0, "lda_pw")[0]
            - exc_density(np.array([rho0 - h]), 0, "lda_pw")[0]) / (2 * h)
    print("uniform LDA: vxc=%.10f ref=%.10f (rs=%.3f)" % (v[0, 0, 0], vref, rs))
    # PBE reduces to LDA at zero gradient
    vp = vxc_grid(np.full((6, 6, 6), rho0), 2 * np.pi * np.eye(3) / 10.0, "pbe")
    print("uniform PBE == LDA: diff=%.2e" % abs(vp[0, 0, 0] - v[0, 0, 0]))
