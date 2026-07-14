"""
abinit_paw_hamiltonian.py -- assemble + emit the PAW augmentation of /Hamiltonian.

Ties the validated kernels together:
  paw_radial  -> channel maps, multipole moments, compensation qfuncl, qq_nt
  paw_qvan    -> augmentation_function_isp{nt} = Q^IJ(G)
  paw_deltaC  -> Onecenter/deltaC (nh^4 AE-PS one-center exchange)

Two layers:
  * build_paw_augmentation(species, miller_g, recip, omega)
        pure computation -> dict of arrays (ijtoh, qq_nt, Q^IJ(G) per nt, deltaC per nt,
        + the Species radial block).  Locally validated vs the QE reference h5.
  * write_paw_block(...)  writes those into an open h5 with the CoQuI schema
        (pp_type="paw"; /Hamiltonian/paw augmentation datasets + /Hamiltonian/Species/nt{i}).

The shared NC-like datasets (miller_g, scf_local_potential, pp_local_component,
per-k projectors, dion) come from the WFK/POT via abinit_hamiltonian and are passed
through unchanged; only the augmentation + Species radial data is PAW-specific.

`species` is a normalized per-species dict (produced by a QE-h5 adapter for testing
or by the ABINIT PAW-XML adapter in production):
    r, rab, mesh, kkbeta, nqlc, lmax_rho, lll (nbeta,),
    u_ae, u_ps (nbeta,nr)  partial waves in u=r*R form,
    pfunc, ptfunc (nbeta,nbeta,nr) [optional; else built from u_ae/u_ps],
    qfuncl (nqlc,nij_beta,nr) [optional; else built from moments*shape_by_L],
    shape_by_L (nqlc,nr) or (nr,) [needed only if qfuncl absent],
    zp, symbol, n_qn (nbeta,) [n quantum numbers, optional].
"""

import numpy as np
import paw_radial as pr
from paw_qvan import augmentation_function
from paw_deltaC import compute_deltaC


def _radial_hartree(nc, r):
    """Spherical Hartree potential of a radial density nc(r):
      V_H(r) = 4*pi [ (1/r) int_0^r nc r'^2 dr' + int_r^inf nc r' dr' ]  ->  Q/r at large r.
    """
    dr = np.diff(r)
    q_in = np.concatenate([[0.0], np.cumsum(0.5 * (nc[1:]*r[1:]**2 + nc[:-1]*r[:-1]**2) * dr)])
    seg = 0.5 * (nc[1:]*r[1:] + nc[:-1]*r[:-1]) * dr
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    rsafe = np.where(r < 1e-12, 1.0, r)
    return 4.0 * np.pi * (q_in / rsafe + tail)


def _poisson_over_r(nwk, r):
    """V_H(r) from nwk = 4*pi*r^2*rho (ABINIT poisson/r convention):
      V_H(r) = (1/r) int_0^r nwk dr' + int_r^inf nwk/r' dr'.
    """
    dr = np.diff(r)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (nwk[1:] + nwk[:-1]) * dr)])
    f = nwk / np.where(r < 1e-12, 1.0, r)
    seg = 0.5 * (f[1:] + f[:-1]) * dr
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return cum / np.where(r < 1e-12, 1.0, r) + tail


def reconstruct_vhtnzc(parse, shape_by_L):
    """Reconstruct the PS ionic HARTREE potential vhtnzc = v_H[tilde-n_Zc] from
    PAW-XML data (no instrumented ABINIT dump needed).  Follows the electrostatic
    part of ABINIT m_pawpsp.F90 (vlocopt==0 'Vbare' path):
      vhtnzc = zero_potential + vh,
      vh     = poisson[ tncore*4*pi*r^2 + g0*r^2*(qcore - Z) ] / r,
      qcore  = int (ncore - tncore) * 4*pi*r^2 dr    (AE - PS core charge),
      g0     = normalized l=0 compensation shape (int g0 r^2 = 1).
    The compensation charge (qcore - Z) makes the enclosed charge -> -Zval, so
    vhtnzc -> -Zval/r.

    This is Hartree-only ON PURPOSE: ABINIT's stored vlocr subtracts a frozen
    atomic XC potential (vlocr = vbare + vh - vxc1) for its DFT SCF; CoQui's frozen
    D^0 must be XC-free (GW/HF baseline), so vxc1 is excluded.  Verified: the tail
    matches ABINIT's dumped vhtnzc exactly (-Zval/r); the interior difference is
    exactly vxc1 (peaked at the nucleus, 0 beyond the augmentation radius).
    Returns None if any input is missing.
    """
    zpot = parse.get("zero_potential")
    if (zpot is None or parse.get("ae_core") is None or parse.get("ps_core") is None
            or parse.get("znucl") is None):
        return None
    r = parse["r"]
    Z = float(parse["znucl"])
    ncore = np.asarray(parse["ae_core"], float) / np.sqrt(4.0 * np.pi)
    tncore = np.asarray(parse["ps_core"], float) / np.sqrt(4.0 * np.pi)
    qcore = np.trapezoid((ncore - tncore) * 4.0 * np.pi * r**2, r)   # AE - PS core charge
    G = np.asarray(shape_by_L[0], float)
    G = G / np.trapezoid(G, r)                                       # g0*r^2, unit monopole
    nwk = tncore * 4.0 * np.pi * r**2 + G * (qcore - Z)
    return np.asarray(zpot, float) + _poisson_over_r(nwk, r)


def assemble_dij0(parse, u_ae, u_ps, shape_by_L, kkbeta):
    """Frozen PAW D^0 (kinetic + ionic Hartree + compensation).

    ABINIT PAW-XML stores ONLY the kinetic_energy_differences (kij) in
    <kinetic_energy_differences>; the frozen D^0 needs the ionic part added
    (see [[reference_coqui_paw_hf_deeq_architecture]]).  This reproduces ABINIT's
    atompaw_dij0 (shared/common/src/39_libpaw/m_paw_atom.F90, opt_init==0):

      D0[i,j] = kij + <u_ae,i u_ae,j | vhnzc> - <u_ps,i u_ps,j | vhtnzc>
                    - intvh * <u_ae,i u_ae,j - u_ps,i u_ps,j>       (same-l pairs)
      vhnzc  = v_H[ae_core] - Z/r        (exact; poisson(ncore)-Z, /r -> -Zval/r)
      vhtnzc = v_H[tilde-n_Zc]           (the PS ionic Hartree, -> -Zval/r)
      intvh  = <vhtnzc * g0 * r^2>,  g0 = l=0 compensation shape (unit monopole)
    All radial integrals run over dr up to the augmentation radius (kkbeta); the
    AE and PS partial waves match beyond it.  Verified to reproduce ABINIT's dumped
    D^0 to ~4 digits for Si (s2s2 1193.74 vs 1193.70, p1p1 0.137, p2p2 2.638).

    XC-FREE: vhtnzc here is the PS ionic HARTREE potential v_H[tilde-n_Zc] only
    (reconstruct_vhtnzc = zero_potential + v_H[tncore + compensation]).  ABINIT's
    stored local potential additionally subtracts a frozen atomic XC term
    (vlocr = vbare + vh - vxc1, m_pawpsp.F90) for its DFT SCF bookkeeping; CoQui's
    frozen D^0 is XC-free by design (it is a self-consistent GW/HF static baseline,
    no DFT XC -- exchange/correlation come from e_hf + e_rpa), so vxc1 is
    deliberately EXCLUDED.  Including it would double-count XC against the exact
    exchange.  This also matches QE's dion convention (XC-free).  If zero_potential
    or the core densities are missing, we fall back to kinetic-only D^0.
    """
    ked = np.asarray(parse["dij0"], float)
    ns = len(parse["states"])
    ked = ked.reshape(ns, ns) if ked.size == ns * ns else ked
    vhtnzc = reconstruct_vhtnzc(parse, shape_by_L)     # XC-free PS ionic Hartree (from XML)
    if vhtnzc is None or parse.get("znucl") is None or parse.get("ae_core") is None:
        return ked                                    # kinetic-only fallback
    r = parse["r"]
    Z = float(parse["znucl"])
    lll = np.array([s["l"] for s in parse["states"]], int)
    rsafe = np.where(r < 1e-12, 1.0, r)
    ncore = np.asarray(parse["ae_core"], float) / np.sqrt(4.0 * np.pi)
    vhnzc = _radial_hartree(ncore, r) - Z / rsafe                        # -> -Zval/r
    vhnzc[r < 1e-12] = 0.0                             # integrand ~ u^2/r -> 0 at origin
    vhtnzc = np.asarray(vhtnzc, float)[:len(r)]
    g0r2 = np.asarray(shape_by_L[0], float)            # g0(r)*r^2 (r^2 convention)
    g0r2 = g0r2 / np.trapezoid(g0r2, r)                # unit monopole: int g0 r^2 dr = 1
    k = int(kkbeta)                                    # cut at augmentation radius
    intvh = np.trapezoid((vhtnzc * g0r2)[:k], r[:k])
    D = np.array(ked, float).copy()
    for i in range(ns):
        for j in range(ns):
            if lll[i] != lll[j]:
                continue
            aa = u_ae[i] * u_ae[j]
            pp = u_ps[i] * u_ps[j]
            D[i, j] += np.trapezoid((aa * vhnzc)[:k], r[:k])
            D[i, j] -= np.trapezoid((pp * vhtnzc)[:k], r[:k])
            D[i, j] -= intvh * np.trapezoid((aa - pp)[:k], r[:k])
    return D


def abinit_species_adapter(parse):
    """Normalize an abinit_pawxml.parse_pawxml() dict into a species dict.

    ABINIT PAW-XML stores partial waves as R (radial function); QE/CoQuI use the
    u = r*R (times-r) form, so multiply by r.  The compensation shape is the
    analytic atompaw 'bessel' shape at shape_rc (paw_radial.shape_bessel); moments
    come from the AE-PS partial-wave difference and are exact regardless of shape.
    """
    r = parse["r"]
    a, d = parse["a"], parse["d"]
    rab = a * d * np.exp(d * np.arange(parse["nr"]))     # dr/di on r=a(exp(di)-1)
    lll = np.array([s["l"] for s in parse["states"]], dtype=int)
    lmax = int(max(lll))
    nqlc = 2 * lmax + 1
    lmax_rho = 2 * lmax
    kkbeta = int(np.searchsorted(r, parse["paw_radius"])) + 1   # inclusive of rc
    u_ae = parse["phi_ae"] * r                            # R -> u = r*R
    u_ps = parse["phi_ps"] * r
    # per-L compensation shape (bessel), in the *r^2 convention
    if parse["shape_type"] == "bessel":
        shape_by_L = np.stack([pr.shape_bessel(r, parse["shape_rc"], L) for L in range(nqlc)])
    elif parse["shape_type"] in ("gauss", "gaussian"):
        shape_by_L = np.stack([pr.shape_gauss(r, parse["shape_rc"])] * nqlc)
    elif parse["shape_type"] in ("sinc", "sinc2"):
        shape_by_L = np.stack([pr.shape_sinc(r, parse["shape_rc"])] * nqlc)
    else:
        raise NotImplementedError("shape_type=%s" % parse["shape_type"])
    zp = float(sum(s.get("f", 0.0) for s in parse["states"]))
    dij0 = assemble_dij0(parse, u_ae, u_ps, shape_by_L, kkbeta)  # kinetic + ionic Hartree (frozen D^0)
    # Frozen core number densities n_core(r) (= L=0 moment / sqrt(4pi)); feed the
    # core-valence Hartree (CoQui rho_core_AE / rho_core_PS, LM=00 channel).
    s4pi = np.sqrt(4.0 * np.pi)
    ae_rho_atc = np.asarray(parse["ae_core"], float) / s4pi if parse.get("ae_core") is not None else None
    rho_atc_ps = np.asarray(parse["ps_core"], float) / s4pi if parse.get("ps_core") is not None else None
    d = dict(r=r, rab=rab, mesh=parse["nr"], kkbeta=kkbeta, nqlc=nqlc,
             lmax_rho=lmax_rho, lll=lll, u_ae=u_ae, u_ps=u_ps,
             proj=parse["proj"], shape_by_L=shape_by_L, dij0=dij0,
             exx_X=parse.get("exx_X"), zp=zp,
             n_qn=[s.get("n") for s in parse["states"]])
    if ae_rho_atc is not None: d["ae_rho_atc"] = ae_rho_atc
    if rho_atc_ps is not None: d["rho_atc_ps"] = rho_atc_ps
    return d


def build_paw_augmentation(species, miller_g, recip, omega):
    """Compute all PAW augmentation arrays for a list of species dicts."""
    nsp = len(species)
    per_sp = []
    nhm = 0
    for nt, sp in enumerate(species):
        r, rab = sp["r"], sp["rab"]
        mesh = sp["mesh"]; nqlc = sp["nqlc"]; lll = np.asarray(sp["lll"], int)
        ch = pr.enumerate_channels(lll)
        nh = ch["nh"]
        nhm = max(nhm, nh)

        pfunc = sp.get("pfunc")
        ptfunc = sp.get("ptfunc")
        if pfunc is None:
            pfunc = pr.pair_products(sp["u_ae"])
        if ptfunc is None:
            ptfunc = pr.pair_products(sp["u_ps"])

        mom = pr.multipole_moments(pfunc, ptfunc, r, rab, lll, nqlc, mesh)
        qfuncl = sp.get("qfuncl")
        if qfuncl is None:
            qfuncl = pr.build_qfuncl(mom, sp["shape_by_L"], r, rab, lll, nqlc, mesh)

        qqq = pr.compute_qqq(mom, len(lll))
        qq_nt = pr.compute_qq_nt(qqq, ch["indv"], ch["nhtolm"])

        qgm = augmentation_function(qfuncl, r, rab, sp["kkbeta"], nqlc, lll,
                                    ch["indv"], ch["nhtolm"], nh, omega, miller_g, recip)
        deltaC = compute_deltaC(pfunc, ptfunc, qfuncl, r, rab, ch["indv"], ch["nhtolm"],
                                lll, nh, sp["lmax_rho"])
        per_sp.append(dict(ch=ch, nh=nh, qqq=qqq, qq_nt=qq_nt, moments=mom,
                           qfuncl=qfuncl, pfunc=pfunc, ptfunc=ptfunc,
                           qgm=qgm, deltaC=deltaC))
    return dict(nsp=nsp, nhm=nhm, per_sp=per_sp)


# --- h5 writer helpers (nda complex convention) ---------------------------
def _mk_writers(h5py):
    VLEN = h5py.special_dtype(vlen=str)

    def wreal(g, k, a, dt=None):
        a = np.ascontiguousarray(a)
        g.create_dataset(k, data=a if dt is None else a.astype(dt))

    def wcplx(g, k, a):
        a = np.ascontiguousarray(a).astype(np.complex128)
        ds = g.create_dataset(k, data=np.ascontiguousarray(np.stack([a.real, a.imag], -1)))
        ds.attrs.create("__complex__", "1", dtype=VLEN)
    return VLEN, wreal, wcplx


def write_species_block(h5root, species, aug, h5py):
    """Write /Hamiltonian/Species/nt{i} radial + Onecenter/deltaC (+ paw/ subgroup)."""
    VLEN, wreal, wcplx = _mk_writers(h5py)
    H = h5root["Hamiltonian"]
    Sg = H.require_group("Species")
    Sg.attrs.create("number_of_species", np.int32(len(species)))
    for nt, sp in enumerate(species):
        g = Sg.create_group("nt%d" % nt)
        g.attrs.create("species_kind", "paw", dtype=VLEN)
        ch = aug["per_sp"][nt]["ch"]
        nbeta = len(sp["lll"])
        for k, v in [("mesh", sp["mesh"]), ("kkbeta", sp["kkbeta"]), ("nbeta", nbeta),
                     ("nh", ch["nh"]), ("lmax", int(max(sp["lll"]))),
                     ("lmax_rho", sp["lmax_rho"]), ("nqlc", sp["nqlc"]),
                     ("q_with_l", 1), ("nqf", 0)]:
            g.attrs.create(k, np.int32(v))
        g.attrs.create("zp", np.float64(sp.get("zp", 0.0)))
        wreal(g, "r", sp["r"], np.float64)
        wreal(g, "rab", sp["rab"], np.float64)
        wreal(g, "lll", np.asarray(sp["lll"], np.int32), np.int32)
        wreal(g, "indv", ch["indv"].astype(np.int32), np.int32)
        wreal(g, "nhtol", ch["nhtol"].astype(np.int32), np.int32)
        wreal(g, "nhtolm", ch["nhtolm"].astype(np.int32), np.int32)
        wreal(g, "kbeta", np.full(nbeta, sp["kkbeta"], np.int32), np.int32)
        wreal(g, "aewfc", sp["u_ae"], np.float64)
        wreal(g, "pswfc", sp["u_ps"], np.float64)
        wreal(g, "qqq", aug["per_sp"][nt]["qqq"], np.float64)
        wreal(g, "qfuncl", aug["per_sp"][nt]["qfuncl"], np.float64)
        if "beta" in sp:
            wreal(g, "beta", sp["beta"], np.float64)
        if "dij0" in sp:
            wreal(g, "dion", sp["dij0"], np.float64)
        # paw/ subgroup (pair densities + moments; radial local pots are optional)
        pg = g.create_group("paw")
        # mandatory augmentation attrs (pseudopot.cpp read_vnl_h5:765-767):
        #   lmax_aug = lmax_rho ; raug = -1.0 (sentinel -> use iraug) ; iraug = kkbeta
        pg.attrs.create("lmax_aug", np.int32(sp["lmax_rho"]))
        pg.attrs.create("raug", np.float64(-1.0))
        pg.attrs.create("iraug", np.int32(sp["kkbeta"]))
        wreal(pg, "pfunc", aug["per_sp"][nt]["pfunc"], np.float64)
        wreal(pg, "ptfunc", aug["per_sp"][nt]["ptfunc"], np.float64)
        augmom = np.zeros((sp["nqlc"], nbeta, nbeta))
        mom = aug["per_sp"][nt]["moments"]
        for mb in range(1, nbeta + 1):
            for nb in range(1, mb + 1):
                ijv = pr.beta_pair_index(nb, mb)
                augmom[:, nb - 1, mb - 1] = mom[:, ijv]
                augmom[:, mb - 1, nb - 1] = mom[:, ijv]
        wreal(pg, "augmom", augmom, np.float64)
        for key in ("ae_vloc", "vloc_ps", "ae_rho_atc", "rho_atc_ps", "oc"):
            if key in sp:
                wreal(pg, key, sp[key], np.float64)
        # Onecenter/deltaC
        og = g.create_group("Onecenter")
        wreal(og, "deltaC", aug["per_sp"][nt]["deltaC"], np.float64)
        # Onecenter/ex_cvij: frozen core-valence exact-exchange kernel. Expand the
        # ln-basis <exact_exchange_X_matrix> to the nh basis (nonzero only for the
        # same lm, exactly ABINIT's `if(ilm==jlm) ex_cvij=...`); contracts linearly
        # with becsum in CoQui (factor 1). ns×ns exx_X is indexed by beta channel.
        exxln = sp.get("exx_X")
        if exxln is not None:
            ch = aug["per_sp"][nt]["ch"]
            indv = np.asarray(ch["indv"], int); nhtolm = np.asarray(ch["nhtolm"], int)
            nh_a = int(ch["nh"]); exxln = np.asarray(exxln, float)
            ex_cv = np.zeros((nh_a, nh_a))
            for I in range(nh_a):
                for J in range(nh_a):
                    if nhtolm[I] == nhtolm[J]:
                        ex_cv[I, J] = exxln[indv[I] - 1, indv[J] - 1]
            wreal(og, "ex_cvij", ex_cv, np.float64)


def _paw_channels(sp):
    """psp-like `channels` list for reuse of abinit_hamiltonian.build_beta_k.
    One channel per beta: chi = r*proj (u-form) so the existing r^1 radial
    transform yields INT proj(r) j_l(qr) r^2 dr (the QE beta(k+G) convention)."""
    r = sp["r"]
    proj = sp["proj"]                                  # (nbeta, nr) R-form projectors
    lll = np.asarray(sp["lll"], int)
    chans = []
    for b in range(len(lll)):
        chans.append(dict(l=int(lll[b]), r=r, chi=(r * proj[b])[:, None],
                          ekb=[float(np.diag(sp["dij0"])[b])]))
    return chans


def write_hamiltonian_paw(h5root, w, vtrial, paw_parses, verbose=True):
    """Emit /Hamiltonian (pp_type='paw') from ABINIT WFK + POT + PAW-XML parses.

    Reuses abinit_hamiltonian for the shared block (dense grid, scf_local_potential,
    per-k PAW projectors beta(k+G) via r*proj), and the builders here for the
    augmentation (ijtoh, qq_nt, Q^IJ(G)) + Species radial + Onecenter/deltaC.

    Conventions (validated by reproducing ABINIT's KS eigenvalues to ~1e-5 Ha via
    the generalized eigenproblem, see validate_h0_paw.py): PAW projectors use the
    r*proj (=> r^1) radial transform; potentials and dion are in Rydberg. The
    static `dion` is ABINIT's frozen Dij0 (kinetic + ionic, XC-free); `deeq` is left
    to CoQuI's native reconstruction from the Species block.
    """
    import h5py
    import abinit_hamiltonian as ah
    VLEN, wreal, wcplx = _mk_writers(h5py)

    recv, rprimd = w["recv"], w["rprimd"]
    vol = abs(np.linalg.det(rprimd))
    tau = w["xred"] @ rprimd
    typat = w["typat"]; nat = len(typat)
    nsp = len(paw_parses); nspin = w["nsppol"]; npol = 1; nk = w["nkpt"]

    species = [abinit_species_adapter(p) for p in paw_parses]

    # dense grid + KS potential (from POT vtrial), same as the norm-conserving path
    ngfft = tuple(int(x) for x in np.array(vtrial).shape[1:4])
    miller_g, (I1, I2, I3) = ah.dense_grid(ngfft)
    ngm = miller_g.shape[0]
    vloc = np.array(vtrial)[0, :, :, :, 0].astype(float)
    Vg = np.fft.fftn(vloc) / vloc.size
    vks_flat = Vg[I1, I2, I3]
    Gcart = miller_g @ recv
    Gn = np.linalg.norm(Gcart, axis=1)

    # projector bookkeeping (one channel per beta -> nh in beta order)
    nh_atom = np.array([sum(2 * int(l) + 1 for l in species[typat[a] - 1]["lll"])
                        for a in range(nat)], dtype=np.int32)
    proj_off = np.concatenate([[0], np.cumsum(nh_atom)[:-1]]).astype(np.int32)
    nkb = int(nh_atom.sum())
    nhm = int(max(sum(2 * int(l) + 1 for l in s["lll"]) for s in species))

    # ---- local ionic potential pp_local_component (QE vltot convention) ----
    # blochl_local_ionic_potential is Bloechl's v_H[n~_Zc]; it is NOT short-ranged:
    # it carries the bare -Z/r nuclear Coulomb tail.  The valence electrons see the
    # frozen ion of charge +Zval, so the local ionic potential must go as -Zval/r.
    # We (1) add the frozen-core Hartree V_H[n_core] (+Zc/r screening), then (2) do
    # the standard Coulomb split so the sin-transform integrand is short-ranged and
    # the -Q/r long range is analytic: V(G!=0)=FT[v+Q/r]-4*pi*Q/(vol*G^2), and the
    # G=0 term is the finite "alpha" reference.  The previous code FFT'd the bare
    # -Z/r tail over the whole radial grid, giving a huge divergent V(G=0) that made
    # the CoQui one-electron energy pathological (~-37000 Ha, ~1/Omega swing).
    vloc_tot = np.zeros(ngm, dtype=complex)
    small = Gn < 1e-8
    Gs = np.where(small, 1.0, Gn)
    for a in range(nat):
        p = paw_parses[typat[a] - 1]
        zval = float(sum(s.get("f", 0.0) for s in p["states"]))
        r = p["r"]; vbar = p["vbar"]
        nn = min(len(r), len(vbar), len(p["ae_core"]))
        r = r[:nn]; vion = vbar[:nn].astype(float)
        # add frozen-core Hartree screening: -Z/r  ->  -Zval/r
        ncore = p["ae_core"][:nn] / np.sqrt(4.0 * np.pi)   # L=0 moment -> number density
        vion = vion + _radial_hartree(ncore, r)
        # asymptotic ionic charge from the r*vion plateau (== Zval up to dataset noise)
        Qtail = -float((r * vion)[-1])
        rvsr = r * vion + Qtail                            # short-ranged: -> 0 at large r
        sr = (4 * np.pi / vol) * np.trapezoid(
            rvsr[None, :] * np.sin(np.outer(Gs, r)) / Gs[:, None], r, axis=1)
        sr -= np.where(small, 0.0, 4.0 * np.pi * Qtail / (vol * Gs**2))
        sr[small] = (4 * np.pi / vol) * np.trapezoid(rvsr * r, r)   # finite alpha
        vloc_tot += np.exp(-1j * (Gcart @ tau[a])) * sr

    # augmentation (validated builders)
    atomic_id = (typat - 1).astype(np.int32)
    aug = build_paw_augmentation(species, miller_g, recv, vol)

    # ---- write /Hamiltonian/paw ----
    H = h5root.create_group("Hamiltonian")
    H.attrs.create("pp_type", "paw", dtype=VLEN)
    g = H.create_group("paw")
    for k, val in [("number_of_nspins", nspin), ("number_of_polarizations", npol),
                   ("number_of_kpoints", nk), ("max_npw", int(w["npw"][:nk].max())),
                   ("number_of_atoms", nat), ("number_of_species", nsp),
                   ("total_num_of_proj", nkb), ("max_proj_per_atom", nhm),
                   ("ngm", ngm), ("lspinorbit_nl", 0), ("lspinorbit_loc", 0)]:
        g.attrs.create(k, np.int32(val))
    wreal(g, "miller_g", miller_g, np.int32)
    wcplx(g, "pp_local_component", 2.0 * vloc_tot)              # Rydberg
    wcplx(g, "scf_local_potential", 2.0 * vks_flat.reshape(nspin, npol * npol, ngm))
    # vxc / vxc_with_nlcc: written as zeros. These are only consumed by the G0W0
    # Sigma - Vxc subtraction; a real vxc (from ABINIT DEN + libxc) is needed there.
    zc = np.zeros((nspin, npol * npol, ngm), complex)
    wcplx(g, "vxc", zc); wcplx(g, "vxc_with_nlcc", zc)
    wreal(g, "proj_per_atom", nh_atom, np.int32)
    wreal(g, "projector_offset", proj_off, np.int32)
    wreal(g, "npw", np.array(w["npw"][:nk]), np.int32)
    wreal(g, "atomic_id", atomic_id, np.int32)
    # dion (nsp,nhm,nhm): D^0 from dij0, m-diagonal, Rydberg
    dion = np.zeros((nsp, nhm, nhm), np.float64)
    for isp, s in enumerate(species):
        ch = aug["per_sp"][isp]["ch"]; nh = ch["nh"]
        for I in range(nh):
            for J in range(nh):
                if ch["nhtolm"][I] == ch["nhtolm"][J]:
                    dion[isp, I, J] = 2.0 * s["dij0"][ch["indv"][I] - 1, ch["indv"][J] - 1]
    wreal(g, "dion", dion, np.float64)
    # per-k PAW projectors beta(k+G) = r*proj transform
    for ik in range(nk):
        n = int(w["npw"][ik]); mill_k = w["kg"][ik][:n]
        kcart = w["kpts_crys"][ik] @ recv
        cols = []
        for a in range(nat):
            s = species[typat[a] - 1]
            psp = dict(channels=_paw_channels(s))
            beta, _ = ah.build_beta_k(psp, mill_k, kcart, recv, tau[a], vol)
            cols.append(beta)
        beta_all = np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), complex)
        wreal(g, "miller_k" + str(ik), mill_k.astype(np.int32), np.int32)
        wcplx(g, "projector_k" + str(ik), beta_all.T)          # (nkb, npw)

    # augmentation + Species blocks (validated)
    write_paw_augmentation(h5root, aug, atomic_id, miller_g, ngm, h5py)
    write_species_block(h5root, species, aug, h5py)
    if verbose:
        print("#   /Hamiltonian(paw): ngm=%d nkb=%d nhm=%d ngfft=%s nsp=%d"
              % (ngm, nkb, nhm, ngfft, nsp))
    return dict(ngm=ngm, nkb=nkb, nhm=nhm, ngfft=ngfft)


def write_paw_augmentation(h5root, aug, atomic_id, miller_g, ngm, h5py):
    """Write the augmentation-only datasets into /Hamiltonian/paw:
    ijtoh, qq_nt, augmentation_function_isp{nt}.  (The NC-like datasets --
    potentials, projectors, dion -- are written separately by the NC emitter
    into the same 'paw' group.)"""
    VLEN, wreal, wcplx = _mk_writers(h5py)
    H = h5root["Hamiltonian"]
    g = H["paw"] if "paw" in H else H.create_group("paw")
    nsp = aug["nsp"]; nhm = aug["nhm"]
    # ijtoh (nsp,nhm,nhm), qq_nt (nsp,nhm,nhm) padded to nhm
    ijtoh = np.zeros((nsp, nhm, nhm), np.int32)
    qq_nt = np.zeros((nsp, nhm, nhm), np.float64)
    for nt in range(nsp):
        nh = aug["per_sp"][nt]["nh"]
        ijtoh[nt, :nh, :nh] = aug["per_sp"][nt]["ch"]["ijtoh"]
        qq_nt[nt, :nh, :nh] = aug["per_sp"][nt]["qq_nt"]
    wreal(g, "ijtoh", ijtoh, np.int32)
    wreal(g, "qq_nt", qq_nt, np.float64)
    for nt in range(nsp):
        # QE names the dataset with the 0-based species index
        wcplx(g, "augmentation_function_isp%d" % nt, aug["per_sp"][nt]["qgm"])
