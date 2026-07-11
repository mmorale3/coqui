"""
paw_qvan.py -- portable port of QE's augmentation Fourier transform Q^IJ(G).

Reproduces, in pure numpy, the pieces of Quantum ESPRESSO's upflib needed to
build the `augmentation_function_isp{nt}` block that CoQuí's PAW/USPP reader
consumes:

  * ylmr2      -- real spherical harmonics, QE ordering/normalization
                  (upflib/ylmr2.f90)
  * real_gaunt -- the ap / lpl / lpx expansion coefficients of a product of two
                  real harmonics (upflib/uspp.f90 `aainit`); computed by the same
                  direction-sampling + matrix-inversion recipe QE uses.  The
                  resulting Gaunt numbers are independent of the (random) sample,
                  so no QE RNG replica is needed.
  * qrad       -- radial Hankel transform  (4pi/Omega) * INT qfuncl_L(r) j_L(|G| r) dr
                  (upflib/qrad_mod.f90 `init_tab_qrad`); here evaluated directly at
                  each needed |G| instead of via QE's dq spline table.
  * qvan2      -- assembly  q(G,ij) = sum_lm (-i)^L ap(lp,ivl,jvl) Y_lp(G) qrad(|G|,ijv,L)
                  (upflib/qvan2.f90)

The G moduli use cartesian |G| in Bohr^-1 (G_cart = miller @ reciprocal_vectors),
so QE's `tpiba` never appears explicitly.

Input radial data (qfuncl, r, rab, indv, nhtolm, lll, kkbeta) can come from either
a QE UPF/pw2coqui h5 (for validation) or be constructed from ABINIT PAW-XML
multipole moments times the compensation shape (production).  All radial integrals
use QE's `simpson` on the tabulated log grid with the `rab = dr/di` measure.

Conventions verified bit-for-bit against
  tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5
(see validate_qvan.py).
"""

import numpy as np

FPI = 4.0 * np.pi


# ---------------------------------------------------------------------------
# Real spherical harmonics -- exact port of upflib/ylmr2.f90
# ---------------------------------------------------------------------------
def ylmr2(lmax2, g):
    """Real spherical harmonics ylm(G) for the first `lmax2` = (lmax+1)^2 harmonics.

    Parameters
    ----------
    lmax2 : int   total number of harmonics, must be a perfect square (lmax+1)^2
    g     : (ng,3) array of vectors (need not be normalized; direction is used)

    Returns
    -------
    ylm : (ng, lmax2) real array, QE ordering  lm = l^2 + 1 + {2m-1 for cos, 2m for sin},
          m=0 stored at l^2+1, normalized so INT |Y|^2 dOmega = 1.
    """
    g = np.asarray(g, dtype=float)
    ng = g.shape[0]
    gg = np.einsum("ij,ij->i", g, g)
    lmax = int(round(np.sqrt(lmax2))) - 1
    assert (lmax + 1) ** 2 == lmax2, "lmax2 must be (lmax+1)^2"
    ylm = np.zeros((ng, lmax2))
    eps = 1.0e-9
    if lmax == 0:
        ylm[:, 0] = np.sqrt(1.0 / FPI)
        return ylm

    gmod = np.sqrt(gg)
    cost = np.where(gmod < eps, 0.0, g[:, 2] / np.where(gmod < eps, 1.0, gmod))
    sent = np.sqrt(np.maximum(0.0, 1.0 - cost * cost))

    # Q(:,l,m) = sqrt((l-m)!/(l+m)!) P_lm, stored (unnormalized) at lm = l^2+1+2m
    Q = np.zeros((ng, lmax2))
    Q[:, 0] = 1.0                       # l=0,m=0
    Q[:, 1] = cost                      # l=1,m=0  (index 1 = 1^2+1+0)
    Q[:, 3] = -sent / np.sqrt(2.0)      # l=1,m=1  (index 3 = 1^2+1+2)
    # NB: QE's Fortran storage index lm = l^2 + 1 + 2m is 1-based; here 0-based -> l^2 + 2m.
    for l in range(2, lmax + 1):
        for m in range(0, l - 1):
            lm = l * l + 2 * m
            lm1 = (l - 1) ** 2 + 2 * m
            lm2 = (l - 2) ** 2 + 2 * m
            Q[:, lm] = (cost * (2 * l - 1) / np.sqrt(l * l - m * m) * Q[:, lm1]
                        - np.sqrt((l - 1) ** 2 - m * m) / np.sqrt(l * l - m * m) * Q[:, lm2])
        lm = l * l + 2 * l
        lm1 = l * l + 2 * (l - 1)
        lm2 = (l - 1) ** 2 + 2 * (l - 1)
        Q[:, lm1] = cost * np.sqrt(2 * l - 1.0) * Q[:, lm2]
        Q[:, lm] = -np.sqrt((2 * l - 1.0) / (2 * l)) * sent * Q[:, lm2]

    # azimuthal angle (QE's branch, defined modulo pi then fixed by sign of g_y)
    gx, gy = g[:, 0], g[:, 1]
    phi = np.where(gx > eps, np.arctan(gy / np.where(np.abs(gx) < eps, 1.0, gx)),
                   np.where(gx < -eps, np.arctan(gy / np.where(np.abs(gx) < eps, 1.0, gx)) + np.pi,
                            np.sign(gy) * (np.pi / 2.0)))

    ylm[:, 0] = Q[:, 0] / np.sqrt(FPI)
    for l in range(1, lmax + 1):
        c = np.sqrt((2 * l + 1) / FPI)
        lm0 = l * l                                 # 0-based index of the m=0 harmonic
        ylm[:, lm0] = c * Q[:, lm0]                 # m=0
        for m in range(1, l + 1):
            lmc = lm0 + 2 * m - 1                    # cos (real, +m)
            lms = lm0 + 2 * m                        # sin (real, -m); also holds Q(l,m)
            ylm[:, lmc] = c * np.sqrt(2.0) * Q[:, lms] * np.cos(m * phi)
            ylm[:, lms] = c * np.sqrt(2.0) * Q[:, lms] * np.sin(m * phi)
    return ylm


# ---------------------------------------------------------------------------
# Product-of-harmonics coefficients ap / lpl / lpx -- port of `aainit`
# ---------------------------------------------------------------------------
def real_gaunt(lmax_in, seed=1234):
    """Coefficients of  Y_li Y_lj = sum_lp ap(lp,li,lj) Y_lp  for real harmonics.

    lmax_in : maximum angular momentum of the *input* harmonics (Y_li, Y_lj).
    Returns (ap, lpx, lpl) with
      ap  : (llx, nin, nin)   llx=(2*lmax_in+1)^2, nin=(lmax_in+1)^2
      lpx : (nin, nin)        number of nonzero lp for each (li,lj)
      lpl : (nin, nin, mx)    the nonzero lp indices (0-based)
    All li,lj,lp indices here are 0-based (QE uses 1-based).
    """
    lli = lmax_in + 1
    nin = lli * lli                       # number of input harmonics
    llx = (2 * lli - 1) ** 2              # harmonics needed to close the product
    rng = np.random.default_rng(seed)
    # random unit vectors (llx of them); regenerate if ylm matrix is singular
    for _ in range(20):
        cost = 2.0 * rng.random(llx) - 1.0
        sint = np.sqrt(1.0 - cost * cost)
        phi = 2.0 * np.pi * rng.random(llx)
        r = np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1)
        ylm = ylmr2(llx, r)               # (llx, llx)
        if abs(np.linalg.det(ylm)) > 1e-6:
            break
    mly = np.linalg.inv(ylm)              # (llx, llx), mly[lp, ir]
    # ap(lp, li, lj) = sum_ir mly[lp,ir] ylm[ir,li] ylm[ir,lj]
    ap = np.einsum("pr,ri,rj->pij", mly, ylm[:, :nin], ylm[:, :nin])
    ap[np.abs(ap) < 1e-3] = 0.0
    lpx = np.zeros((nin, nin), dtype=int)
    mx = 0
    nz = [[np.nonzero(np.abs(ap[:, i, j]) > 1e-3)[0] for j in range(nin)] for i in range(nin)]
    for i in range(nin):
        for j in range(nin):
            lpx[i, j] = nz[i][j].size
            mx = max(mx, lpx[i, j])
    lpl = np.full((nin, nin, max(mx, 1)), -1, dtype=int)
    for i in range(nin):
        for j in range(nin):
            lpl[i, j, :lpx[i, j]] = nz[i][j]
    return ap, lpx, lpl


# ---------------------------------------------------------------------------
# Spherical Bessel j_l -- stable downward (Miller) recurrence, vectorized in x
# ---------------------------------------------------------------------------
def _sph_jn_series(l, x):
    """Small-argument power series j_l(x) = x^l/(2l+1)!! * (1 - x^2/(2(2l+3)) + ...)."""
    dfact = 1.0
    for k in range(1, 2 * l + 2, 2):
        dfact *= k                                  # (2l+1)!!
    t = x ** l / dfact
    s = np.array(t, dtype=float)
    x2 = x * x
    for k in range(1, 8):
        t = t * (-x2 / 2.0) / (k * (2 * l + 2 * k + 1))
        s = s + t
    return s


def sph_jn_all(lmax, x):
    """Spherical Bessel functions j_0..j_lmax for a 1-D array x >= 0.

    Hybrid: upward recurrence where it is stable (x >~ l), power series in the
    small-x region (x < l+0.5) to avoid subtractive cancellation.  No overflow.
    Matches scipy.special.spherical_jn to ~1e-13 across the range used here.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros((lmax + 1, x.size))
    xs = np.where(x < 1e-30, 1.0, x)                # guard div-by-zero at x=0
    out[0] = np.where(x < 1e-12, 1.0, np.sin(xs) / xs)
    if lmax >= 1:
        out[1] = np.where(x < 1e-12, 0.0, np.sin(xs) / xs ** 2 - np.cos(xs) / xs)
    for l in range(2, lmax + 1):
        out[l] = (2 * l - 1) / xs * out[l - 1] - out[l - 2]   # upward (stable for x>~l)
    # replace the cancellation-prone small-x tail per l with the series
    for l in range(2, lmax + 1):
        m = x < (l + 0.5)
        if np.any(m):
            out[l, m] = _sph_jn_series(l, x[m])
    return out


# ---------------------------------------------------------------------------
# QE simpson on the log grid (upflib/simpson.f90)
# ---------------------------------------------------------------------------
def qe_simpson(func, rab, mesh):
    """QE `simpson`: composite Simpson with the rab (=dr/di) measure, indices 1..mesh."""
    r12 = 1.0 / 3.0
    fr = func[:mesh] * rab[:mesh] * r12
    asum = 0.0
    # loop i = 2 .. mesh-1 step 2  (1-based) -> pairs of intervals
    for i in range(1, mesh - 1, 2):
        asum += fr[i - 1] + 4.0 * fr[i] + fr[i + 1]
    return asum


def qe_simpson_batch(func2d, rab, mesh):
    """Vectorized qe_simpson over leading axis of func2d (..., nr)."""
    r12 = 1.0 / 3.0
    fr = func2d[..., :mesh] * (rab[:mesh] * r12)
    asum = np.zeros(func2d.shape[:-1])
    for i in range(1, mesh - 1, 2):
        asum += fr[..., i - 1] + 4.0 * fr[..., i] + fr[..., i + 1]
    return asum


# ---------------------------------------------------------------------------
# Radial Hankel transform qrad(|G|, ijv, L)  -- upflib/qrad_mod init_tab_qrad
# ---------------------------------------------------------------------------
def qrad_direct(qfuncl, r, rab, kkbeta, nqlc, lll, omega, gmods):
    """qrad(iG, ijv, L) = (4pi/Omega) INT qfuncl_L,ijv(r) j_L(|G_iG| r) dr.

    qfuncl : (nqlc, nij_beta, nr)  QE-convention augmentation radial functions
             (already carry the r^2 measure; integrate with rab only).
    lll    : (nbeta,) angular momentum per beta channel (for the L-selection).
    gmods  : (nG,) cartesian |G| in Bohr^-1 (unique values recommended).
    Returns qrad (nG, nij_beta, nqlc).
    """
    nbeta = len(lll)
    nij_beta = qfuncl.shape[1]
    nG = gmods.size
    # unique |G| to avoid recomputing Bessel integrals for repeated moduli
    gu, inv = np.unique(np.round(gmods, 12), return_inverse=True)
    # j_L(|G| r) for all needed L, for each unique |G|, on the radial grid
    qrad_u = np.zeros((gu.size, nij_beta, nqlc))
    rr = r[:kkbeta]
    # beta-pair -> (nb,mb) 1-based decode for L-selection
    def pair_ls(ijv):  # ijv 0-based
        # invert ijv = nb(nb-1)/2 + mb (nb>=mb, 1-based) - 1
        k = ijv + 1
        nb = int((np.sqrt(8 * k - 7) + 1) // 2)
        while nb * (nb - 1) // 2 >= k:
            nb -= 1
        while (nb + 1) * nb // 2 < k:
            nb += 1
        mb = k - nb * (nb - 1) // 2
        return lll[nb - 1], lll[mb - 1]
    for iu, q in enumerate(gu):
        jl = sph_jn_all(nqlc - 1, q * rr)              # (nqlc, kkbeta)
        for ijv in range(nij_beta):
            li, lj = pair_ls(ijv)
            for L in range(nqlc):
                if L < abs(li - lj) or L > li + lj or (L + li + lj) % 2 != 0:
                    continue
                integrand = qfuncl[L, ijv, :kkbeta] * jl[L]
                qrad_u[iu, ijv, L] = qe_simpson(integrand, rab, kkbeta)
    qrad_u *= FPI / omega
    return qrad_u[inv]


# ---------------------------------------------------------------------------
# Full augmentation block  augmentation_function_isp{nt}  (nij_proj, nG) complex
# ---------------------------------------------------------------------------
def augmentation_function(qfuncl, r, rab, kkbeta, nqlc, lll,
                          indv, nhtolm, nh, omega, miller_g, recip):
    """Build Q^IJ(G) for one species, ordered as QE's upper-triangle (ih,jh) packing.

    indv, nhtolm : 1-based per-projector maps (length nh).
    miller_g     : (nG,3) int Miller indices of the dense grid.
    recip        : (3,3) reciprocal lattice, rows b_n (a.b = 2pi), Bohr^-1.
    Returns qgm (nij_proj = nh(nh+1)/2, nG) complex128.
    """
    Gcart = miller_g @ recip                            # (nG,3) Bohr^-1
    gmods = np.sqrt(np.einsum("ij,ij->i", Gcart, Gcart))
    nG = gmods.size
    lmaxq = nqlc                                        # (2 lmax_beta + 1)
    ylm = ylmr2(lmaxq * lmaxq, Gcart)                   # (nG, lmaxq^2)
    lmax_in = int(max(nhtolm))                          # nhtolm up to (lmax+1)^2
    lmax_beta = int(max(lll))
    ap, lpx, lpl = real_gaunt(lmax_beta)
    qrad = qrad_direct(qfuncl, r, rab, kkbeta, nqlc, lll, omega, gmods)  # (nG,nij_beta,nqlc)

    def beta_ijv(nb, mb):                               # 1-based nb,mb -> 0-based ijv
        if nb >= mb:
            return nb * (nb - 1) // 2 + mb - 1
        return mb * (mb - 1) // 2 + nb - 1

    nij_proj = nh * (nh + 1) // 2
    qgm = np.zeros((nij_proj, nG), dtype=np.complex128)
    ijh = 0
    for ih in range(nh):
        for jh in range(ih, nh):
            nb, mb = indv[ih], indv[jh]                 # 1-based beta channels
            ivl, jvl = nhtolm[ih] - 1, nhtolm[jh] - 1   # 0-based harmonic indices
            ijv = beta_ijv(nb, mb)
            acc = np.zeros(nG, dtype=np.complex128)
            for k in range(lpx[ivl, jvl]):
                lp = lpl[ivl, jvl, k]                   # 0-based combined LM
                L = int(np.floor(np.sqrt(lp)))          # angular momentum of lp
                acc += ((-1j) ** L) * ap[lp, ivl, jvl] * ylm[:, lp] * qrad[:, ijv, L]
            qgm[ijh] = acc
            ijh += 1
    return qgm
