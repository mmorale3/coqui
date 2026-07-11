"""
paw_radial.py -- PAW radial front-end: turn partial waves into the CoQuI/QE
augmentation quantities (channel maps, multipole moments, compensation qfuncl,
augmentation overlaps).  Convention-agnostic: given a normalized radial dict it
produces exactly the QE-schema arrays, so both a QE-h5 adapter (for validation)
and the ABINIT PAW-XML path share this code.

All conventions verified against tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5:
  * channel enumeration indv/nhtol/nhtolm   (QE real-harmonic order)
  * ijtoh upper-triangle packing            (exact)
  * moment q_ij^L = INT (pfunc-ptfunc) r^L dr = augmom   (1e-11)
  * qqq = augmom[L=0];  qq_nt(ih,jh)=qqq(indv,indv) gated by nhtolm equality
Radial functions are handled in QE's u = r*R (times-r) form: pfunc = u_i u_j.
ABINIT PAW-XML stores partial waves as R, so multiply by r before calling here.
"""

import numpy as np
from paw_qvan import qe_simpson_batch


# ---------------------------------------------------------------------------
# Projector (nh) enumeration from the beta channels (angular momenta lll)
# ---------------------------------------------------------------------------
def enumerate_channels(lll):
    """From per-beta angular momenta lll (length nbeta) build the QE projector maps.

    Returns dict with (all 1-based where QE is 1-based):
      nh      : total projectors = sum_b (2 l_b + 1)
      indv    : (nh,) 1-based beta channel of each projector
      nhtol   : (nh,) angular momentum l of each projector (0-based l)
      nhtolm  : (nh,) 1-based combined harmonic index lm = l^2 + 1 + m  (QE Ylm order)
      ijtoh   : (nh,nh) 1-based upper-triangle packed pair index (symmetric)
      nij_beta: nbeta(nbeta+1)/2
    """
    lll = np.asarray(lll, dtype=int)
    nbeta = lll.size
    indv, nhtol, nhtolm = [], [], []
    for b in range(nbeta):
        l = lll[b]
        for m in range(2 * l + 1):
            indv.append(b + 1)              # 1-based beta
            nhtol.append(l)
            nhtolm.append(l * l + 1 + m)     # 1-based combined lm
    nh = len(indv)
    ijtoh = np.zeros((nh, nh), dtype=int)
    c = 0
    for ih in range(nh):
        for jh in range(ih, nh):
            c += 1
            ijtoh[ih, jh] = c
            ijtoh[jh, ih] = c
    return dict(nh=nh, indv=np.array(indv), nhtol=np.array(nhtol),
                nhtolm=np.array(nhtolm), ijtoh=ijtoh,
                nij_beta=nbeta * (nbeta + 1) // 2, nbeta=nbeta, lll=lll)


# ---------------------------------------------------------------------------
# Partial-wave products pfunc (u_i u_j) and multipole moments
# ---------------------------------------------------------------------------
def pair_products(u):
    """pfunc[nb,mb,r] = u_nb(r) * u_mb(r)  (u = r*R, times-r form). u:(nbeta,nr)."""
    return np.einsum("ir,jr->ijr", u, u)


def beta_pair_index(nb, mb):
    """0-based triangular index ijv for beta pair (1-based nb,mb), mb>=nb."""
    if mb < nb:
        nb, mb = mb, nb
    return (mb * (mb - 1)) // 2 + nb - 1


def multipole_moments(pfunc, ptfunc, r, rab, lll, nqlc, mesh):
    """q_ij^L = INT (pfunc - ptfunc)_{nb mb}(r) r^L dr, packed (nqlc, nij_beta).

    Returns moments with the QE selection rule |li-lj|<=L<=li+lj, L+li+lj even.
    """
    nbeta = pfunc.shape[0]
    nij = nbeta * (nbeta + 1) // 2
    mom = np.zeros((nqlc, nij))
    diff = pfunc - ptfunc                       # (nbeta,nbeta,nr)
    for mb in range(1, nbeta + 1):
        for nb in range(1, mb + 1):
            ijv = beta_pair_index(nb, mb)
            li, lj = lll[nb - 1], lll[mb - 1]
            for L in range(nqlc):
                if L < abs(li - lj) or L > li + lj or (L + li + lj) % 2:
                    continue
                integrand = diff[nb - 1, mb - 1] * r ** L
                mom[L, ijv] = qe_simpson_batch(integrand[None, :], rab, mesh)[0]
    return mom


# ---------------------------------------------------------------------------
# Compensation charge qfuncl_L(r) = moment_L * normalized shape g_L(r)
# ---------------------------------------------------------------------------
def normalized_shape(shape_L, r, rab, L, mesh):
    """Normalize a radial shape so that INT g_L(r) r^L dr = 1 (u/times-r convention,
    i.e. g_L already carries the r^2 -- shape_L passed in is g_L*r^2 form)."""
    norm = qe_simpson_batch((shape_L * r ** L)[None, :], rab, mesh)[0]
    return shape_L / norm


def build_qfuncl(moments, shape_by_L, r, rab, lll, nqlc, mesh):
    """qfuncl[L, ijv, r] = moment_L[ijv] * g_L(r), g_L normalized to unit L-moment.

    shape_by_L : (nqlc, nr) array of the raw (unnormalized) shape g_L*r^2 per L,
                 or a single (nr,) array reused for all L.
    """
    nij = moments.shape[1]
    nr = r.size
    qfuncl = np.zeros((nqlc, nij, nr))
    shape_by_L = np.asarray(shape_by_L)
    for L in range(nqlc):
        gL = shape_by_L[L] if shape_by_L.ndim == 2 else shape_by_L
        gLn = normalized_shape(gL, r, rab, L, mesh)
        for ijv in range(nij):
            if moments[L, ijv] != 0.0:
                qfuncl[L, ijv] = moments[L, ijv] * gLn
    return qfuncl


# ---------------------------------------------------------------------------
# Augmentation overlaps qqq (beta space) and qq_nt (projector space)
# ---------------------------------------------------------------------------
def compute_qqq(moments, nbeta):
    """qqq(nb,mb) = L=0 multipole moment of the AE-PS pair density (beta space)."""
    qqq = np.zeros((nbeta, nbeta))
    for mb in range(1, nbeta + 1):
        for nb in range(1, mb + 1):
            ijv = beta_pair_index(nb, mb)
            qqq[nb - 1, mb - 1] = moments[0, ijv]
            qqq[mb - 1, nb - 1] = moments[0, ijv]
    return qqq


def compute_qq_nt(qqq, indv, nhtolm):
    """qq_nt(ih,jh) = qqq(indv(ih),indv(jh)) if nhtolm(ih)==nhtolm(jh) else 0."""
    nh = indv.size
    qq = np.zeros((nh, nh))
    for ih in range(nh):
        for jh in range(nh):
            if nhtolm[ih] == nhtolm[jh]:
                qq[ih, jh] = qqq[indv[ih] - 1, indv[jh] - 1]
    return qq


# ---------------------------------------------------------------------------
# Analytic compensation shapes (ABINIT PAW-XML shape_function types).
# Returned in the g_L*r^2 (times-r^2) convention expected by build_qfuncl.
# Normalization is fixed later per-L by normalized_shape, so only the shape
# (not the scale) matters here.
# ---------------------------------------------------------------------------
def shape_gauss(r, rc):
    """Gaussian exp(-(r/rc)^2), density convention -> return *r^2."""
    return np.exp(-(r / rc) ** 2) * r ** 2


def shape_sinc(r, rc):
    """[sin(pi r/rc)/(pi r/rc)]^2 inside rc, density convention -> *r^2."""
    x = np.pi * r / rc
    s = np.where(r < rc, np.sinc(x / np.pi) ** 2, 0.0)   # np.sinc(y)=sin(pi y)/(pi y)
    return s * r ** 2


def shape_bessel(r, rc, L):
    """ABINIT/atompaw 'bessel' compensation shape for channel L.

    g_L(r) = a1 j_L(q1 r) + a2 j_L(q2 r) for r<rc, else 0, with q_i = x_{L,i}/rc
    (x_{L,i} = i-th positive zero of j_L, so g_L(rc)=0 automatically) and a2/a1 set
    by the smoothness condition g_L'(rc)=0.  Overall scale is irrelevant (build_qfuncl
    renormalizes to unit L-moment).  Returned in the *r^2 (density->u) convention.
    """
    from paw_qvan import sph_jn_all
    x1, x2 = _BESSEL_ZEROS2[L]
    q1, q2 = x1 / rc, x2 / rc
    # d/dr j_L(q r)|_{rc} = q j_L'(q rc); at a zero x of j_L, j_L'(x) = -j_{L+1}(x)
    jp = sph_jn_all(L + 1, np.array([x1, x2]))[L + 1]
    d1 = -q1 * jp[0]
    d2 = -q2 * jp[1]
    a1, a2 = 1.0, -d1 / d2
    jl1 = sph_jn_all(L, q1 * r)[L]
    jl2 = sph_jn_all(L, q2 * r)[L]
    g = np.where(r < rc, a1 * jl1 + a2 * jl2, 0.0)
    return g * r ** 2


# first two positive zeros of the spherical Bessel function j_L
_BESSEL_ZEROS2 = {
    0: (3.14159265359, 6.28318530718),
    1: (4.49340945790, 7.72525183694),
    2: (5.76345919689, 9.09501133048),
    3: (6.98793200050, 10.4171185475),
    4: (8.18256145257, 11.7049071546),
}
