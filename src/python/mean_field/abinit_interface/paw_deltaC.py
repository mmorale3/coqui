"""
paw_deltaC.py -- one-center PAW exact-exchange kernel deltaC(nh,nh,nh,nh).

Port of QE's paw_exx.f90 `PAW_init_fock_kernel` / `PAW_fock_onecenter`:

    deltaC[i,j,o,u] = ( K_ae[i,j,o,u] - K_ps[i,j,o,u] ) / e2
    K[i,j,o,u]      = sum_LM  INT  V_LM^{ij}(r) rho_LM^{ou}(r) dr

with
  * rho_LM^{ij}(r) = sum_lm  ap(lm, li, lj) [ pfunc_{nb mb}(r)         (AE)
                                            | ptfunc_{nb mb}(r) + qfuncl_{nmb,L}(r) (PS) ]
    (PAW_rho_lm_ij; ap = real-Gaunt coefficients, same as qvan2)
  * V_LM(r) = (4pi/(2L+1)) [ r^{-(L+1)} INT_0^r rho_LM(r') r'^L dr'
                           +  r^L        INT_r^inf rho_LM(r') r'^{-(L+1)} dr' ]
    the multipole radial Hartree (QE uses an ODE solver `hartree`; here the
    equivalent direct double-integral -- rho_LM already carries the r^2 measure
    because pfunc = aewfc_i*aewfc_j is stored in u=r*R form).

e2 = 2 (Rydberg atomic units); QE stores paw_fockrnl = e2*K and pw2coqui writes
deltaC = ke/e2^2, i.e. deltaC = (K_ae - K_ps)/e2 in Hartree.

The same routine serves ABINIT: feed pfunc/ptfunc built from the PAW-XML partial
waves and the compensation charge and it yields a CoQuI-compatible deltaC directly
(no need to reverse-engineer ABINIT's exact_exchange_X_matrix; that becomes a
cross-check via the exchange contraction).

Validated against Onecenter/deltaC in the QE reference h5 (see validate_deltaC.py).
"""

import numpy as np
from paw_qvan import real_gaunt, FPI

E2 = 2.0


def simpson_weights(rab, mesh):
    """Weight vector w so that INT f dr(rab) ~ sum_r f[r] w[r], matching QE `simpson`."""
    w = np.zeros(mesh)
    r12 = 1.0 / 3.0
    # QE loop: do i=2,mesh-1,2 :  asum += fr[i-1]+4 fr[i]+fr[i+1], fr=f*rab/3
    for i in range(1, mesh - 1, 2):
        w[i - 1] += rab[i - 1] * r12
        w[i] += 4.0 * rab[i] * r12
        w[i + 1] += rab[i + 1] * r12
    return w


def _radial_hartree(rho_lm, r, rab, Loflm):
    """V_LM(r) for every lm component of one pair density.

    rho_lm : (nlm, nr)   pair density components (carry r^2)
    Returns v_lm : (nlm, nr).
    """
    nlm, nr = rho_lm.shape
    v = np.zeros((nlm, nr))
    dr = rab  # dr/di measure on the log grid
    # r=0 guard: ABINIT log grid has r[0]=0 exactly (QE starts >0).  All pair
    # densities carry r^2 so rho[...,0]=0 and contribute nothing; only protect the
    # 1/r^{L+1} denominators from a divide-by-zero at r[0].
    rsafe = r.copy()
    if rsafe[0] == 0.0:
        rsafe[0] = rsafe[1] * 1e-6
    for lm in range(nlm):
        L = Loflm[lm]
        rho = rho_lm[lm]
        if not np.any(rho):
            continue
        rL = rsafe ** L
        rLp1 = rsafe ** (L + 1)
        # inner: A(r_i) = INT_0^{r_i} rho r'^L dr'   (cumulative, trapezoid on log grid)
        fin = rho * rL * dr
        A = np.cumsum(fin) - 0.5 * fin           # midpoint-ish: exclude half of own cell
        # outer: B(r_i) = INT_{r_i}^inf rho r'^{-(L+1)} dr'
        fout = rho / rLp1 * dr
        B = np.cumsum(fout[::-1])[::-1] - 0.5 * fout
        v[lm] = (FPI / (2 * L + 1)) * (A / rLp1 + B * rL)
    return v


def _pair_densities(pfunc, qfuncl, indv, nhtolm, nh, ap, lpx, lpl, nlm, is_ps):
    """Build rho_LM for every projector pair (ih,jh), shape (nh*nh, nlm, nr).

    pfunc  : (nbeta,nbeta,nr) AE (pfunc) or PS (ptfunc) partial-wave products.
    qfuncl : (nqlc, nij_beta, nr) compensation (only used when is_ps).
    """
    nr = pfunc.shape[-1]
    rho = np.zeros((nh * nh, nlm, nr))
    Loflm = np.array([int(np.floor(np.sqrt(lm))) for lm in range(nlm)])
    for ih in range(nh):
        for jh in range(nh):
            # order so that mb >= nb (nmb triangular index; symmetric otherwise)
            nb_, mb_ = indv[ih], indv[jh]              # 1-based beta channels
            if mb_ >= nb_:
                iih, jjh, nb, mb = ih, jh, nb_, mb_
            else:
                iih, jjh, nb, mb = jh, ih, mb_, nb_
            nmb = (mb * (mb - 1)) // 2 + nb            # 1-based triangular pair index
            ivl = nhtolm[iih] - 1                      # 0-based input-harmonic indices
            jvl = nhtolm[jjh] - 1
            out = rho[ih * nh + jh]
            for k in range(lpx[jvl, ivl]):
                lm = lpl[jvl, ivl, k]                  # 0-based combined LM
                pref = ap[lm, ivl, jvl]
                out[lm] += pref * pfunc[nb - 1, mb - 1]
                if is_ps:
                    L = Loflm[lm]
                    out[lm] += pref * qfuncl[L, nmb - 1]
    return rho, Loflm


def compute_deltaC(pfunc, ptfunc, qfuncl, r, rab, indv, nhtolm, lll, nh,
                   lmax_rho, kkbeta=None):
    """Return deltaC (nh,nh,nh,nh) in Hartree.

    indv, nhtolm : 1-based per-projector maps (length nh).
    lll          : angular momentum per beta channel (for real_gaunt sizing).
    lmax_rho     : max L in the augmentation (= 2*lmax_beta); nlm = (lmax_rho+1)^2.
    """
    nr = pfunc.shape[-1]
    mesh = nr if kkbeta is None else kkbeta
    lmax_beta = int(max(lll))
    ap, lpx, lpl = real_gaunt(lmax_beta)
    nlm = (lmax_rho + 1) ** 2

    rho_ae, Loflm = _pair_densities(pfunc, qfuncl, indv, nhtolm, nh, ap, lpx, lpl, nlm, False)
    rho_ps, _ = _pair_densities(ptfunc, qfuncl, indv, nhtolm, nh, ap, lpx, lpl, nlm, True)

    # radial Hartree potentials for each pair
    v_ae = np.stack([_radial_hartree(rho_ae[p], r, rab, Loflm) for p in range(nh * nh)])
    v_ps = np.stack([_radial_hartree(rho_ps[p], r, rab, Loflm) for p in range(nh * nh)])

    w = simpson_weights(rab, mesh)                     # (mesh,)
    # K[p_ij, q_ou] = sum_lm sum_r V_ij[lm,r] rho_ou[lm,r] w[r]
    rho_ae_w = rho_ae[:, :, :mesh] * w
    rho_ps_w = rho_ps[:, :, :mesh] * w
    K_ae = np.einsum("plr,qlr->pq", v_ae[:, :, :mesh], rho_ae_w, optimize=True)
    K_ps = np.einsum("plr,qlr->pq", v_ps[:, :, :mesh], rho_ps_w, optimize=True)

    # QE: paw_fockrnl = e2*kexx and v_lm already carries e2 (pref = e2*4pi/(2L+1)),
    # so ke = k_ae-k_ps = e2^2 * K and deltaC = ke/e2^2 = K_ae - K_ps (my V0 has no e2).
    deltaC = (K_ae - K_ps)
    return deltaC.reshape(nh, nh, nh, nh)
