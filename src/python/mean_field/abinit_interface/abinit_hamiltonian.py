"""
abinit2coqui -- build the norm-conserving `/Hamiltonian` block from an
ABINIT WFK + its psp8 pseudopotential(s).

Physics recomputed here (QE init_us_2 / vloc_of_g analogs):
  * V_loc(G)              -- local pseudopotential on the dense G-grid
  * beta_i(k+G)           -- KB nonlocal projectors in G-space, per k
  * dion                  -- KB coefficients (diagonal for NC), from psp8 ekb

CONVENTIONS (must be confirmed against a QE `vkb`/`projector_k` reference for the
same system -- see STEP2_hamiltonian_design.md; knobs exposed as module constants):
  * real spherical harmonics, QE `ylmr2` lm-ordering
  * projector phase factor (-i)^l  (QE `init_us_2`: `pref=(0,-1)**l`)
  * normalization 4*pi/sqrt(Omega)
  * psp8 chi_ln(r) treated as the bare radial projector; transform uses r^2 dr
Because these are the exact points that differ between codes, they are isolated in
`_PROJ_PHASE`, `_R_POWER`, and `real_ylm` so validation can adjust one at a time.
"""

import numpy as np
from scipy.special import spherical_jn

# --- conventions VALIDATED to 5.6e-9 Ha vs ABINIT eigenvalues (validate_h0.py, Si Gamma) ---
# psp8 chi_ln(r) is ALREADY r-weighted (chi = r * beta_true, like QE UPF beta), so the radial
# transform uses r^1: integral chi(r) j_l(qr) r dr == integral beta_true j_l r^2 dr (physical).
_R_POWER = 1          # radial transform weight (VALIDATED: 1, not 2)
_USE_MINUS_I = True   # projector phase (-i)^l (QE init_us_2)  [VALIDATED]
# normalization pref = 4*pi/sqrt(Omega), dion = psp8 ekb (Ha) [VALIDATED]


# --------------------------------------------------------------------------
# Real spherical harmonics in QE ylmr2 ordering AND sign convention
# lm index within an l-shell: m = 0, +1, -1, +2, -2, ... QE builds the real
# pair from the (-1)^m (Condon-Shortley) complex harmonics, so every ODD-m
# component carries an extra minus sign relative to plain real harmonics
# (l=1: -x, -y; l=2: -xz, -yz; m=+/-2 unchanged). CoQui contracts the
# stored projector_k with Q_IJ(G) built from its OWN QE-convention
# aainit/ylm tables (and this converter's qfuncl already follows QE), so
# beta MUST use these signs: the plain-harmonics version sign-flips the
# odd-m projector rows, which cancels out of everything channel-diagonal
# or site-symmetric (dion, Hartree becsum, onsite) but decoheres the
# off-diagonal k-pair augmentation in exchange (measured -51.7 mHa on
# b_tests si222, 2026-07-25 pin-down).
# --------------------------------------------------------------------------
def real_ylm(l, ghat):
    """Real spherical harmonics Y_lm(ghat) for one l, all m, QE ylmr2
    ordering and signs. ghat: (n,3) unit vectors. Returns (n, 2l+1)."""
    x, y, z = ghat[:, 0], ghat[:, 1], ghat[:, 2]
    n = ghat.shape[0]
    out = np.empty((n, 2 * l + 1), dtype=float)
    if l == 0:
        out[:, 0] = 0.5 * np.sqrt(1.0 / np.pi)
    elif l == 1:
        c = np.sqrt(3.0 / (4.0 * np.pi))
        out[:, 0] = c * z            # m=0  (pz)
        out[:, 1] = -c * x           # m=+1 (QE: -px)
        out[:, 2] = -c * y           # m=-1 (QE: -py)
    elif l == 2:
        c0 = np.sqrt(5.0 / (16.0 * np.pi))
        c1 = np.sqrt(15.0 / (4.0 * np.pi))
        c2 = np.sqrt(15.0 / (16.0 * np.pi))
        out[:, 0] = c0 * (3.0 * z * z - 1.0)   # m=0   (3z^2-r^2)
        out[:, 1] = -c1 * x * z                # m=+1  (QE: -xz)
        out[:, 2] = -c1 * y * z                # m=-1  (QE: -yz)
        out[:, 3] = c2 * (x * x - y * y)       # m=+2  (x^2-y^2)
        out[:, 4] = c1 * x * y                 # m=-2  (xy)
    else:
        raise NotImplementedError("real_ylm: l>2 not implemented (Si needs l<=2).")
    return out


def radial_transform(chi, r, l, qgrid):
    """F_l(q) = integral_0^inf chi(r) j_l(q r) r^_R_POWER dr, for an array of q.
    Vectorized trapezoid on the psp8 radial grid. Returns (len(qgrid),)."""
    q = np.asarray(qgrid)
    rw = (r ** _R_POWER)
    jl = spherical_jn(l, np.outer(q, r))               # (nq, nr)
    return np.trapezoid(chi[None, :] * jl * rw[None, :], r, axis=1)


def vloc_of_g(psp, Gnorm, vol, zion):
    """V_loc(G) on the dense grid (QE vloc_of_g analog).
    psp8 Vloc(r); subtract the -zion/r Coulomb tail, transform the short-ranged
    remainder, add back the analytic -4*pi*zion/G^2 (Ha, e^2=1).  Vectorized.
    Gnorm: (ngm,) |G| bohr^-1 (G=0 handled). Returns (ngm,) real."""
    r, vloc = psp["r"], psp["vloc"]
    rvsr = r * vloc + zion                              # short-ranged, -> 0 at large r
    fourpi_over_vol = 4.0 * np.pi / vol
    G = np.asarray(Gnorm, dtype=float)
    small = G < 1e-8
    Gs = np.where(small, 1.0, G)
    sr = fourpi_over_vol * np.trapezoid(
        rvsr[None, :] * np.sin(np.outer(Gs, r)) / Gs[:, None], r, axis=1)
    out = sr - fourpi_over_vol * zion / (Gs * Gs)
    out[small] = fourpi_over_vol * np.trapezoid(rvsr * r, r)   # G=0
    return out


def build_beta_k(psp, mill_k, kpt_cart, recv, tau_cart, vol):
    """KB projectors beta_i(k+G) for one k-point, one atom.
    mill_k: (npw,3) integer Miller of the projector grid (= wfc per-k grid).
    kpt_cart: (3,) cartesian k (bohr^-1).  tau_cart: (3,) atom position (bohr).
    Returns (npw, nh_atom) complex, and the per-projector (l, ekb) list.
    """
    kpg = kpt_cart[None, :] + mill_k @ recv           # (npw,3) cartesian k+G
    q = np.linalg.norm(kpg, axis=1)                    # (npw,)
    qsafe = np.where(q < 1e-12, 1e-12, q)
    ghat = kpg / qsafe[:, None]
    npw = mill_k.shape[0]
    pref = 4.0 * np.pi / np.sqrt(vol)
    sk = np.exp(-1j * (kpg @ tau_cart))                # structure factor e^{-i(k+G).tau}
    cols = []
    meta = []
    for ch in psp["channels"]:
        l = ch["l"]
        ylm = real_ylm(l, ghat)                        # (npw, 2l+1)
        phase = ((-1j) ** l) if _USE_MINUS_I else ((1j) ** l)
        for n in range(ch["chi"].shape[1]):
            Fq = radial_transform(ch["chi"][:, n], ch["r"], l, q)   # (npw,)
            for m in range(2 * l + 1):
                col = pref * phase * ylm[:, m] * Fq * sk            # (npw,)
                cols.append(col)
                meta.append((l, ch["ekb"][n]))
    beta = np.stack(cols, axis=1) if cols else np.zeros((npw, 0), dtype=complex)
    return beta, meta


def dense_grid(ngfft):
    """Miller indices of the full FFT box (dfftp/aug grid), C-order flattened.
    Returns miller_g (ngm,3) int and the (i1,i2,i3) fft-bin index arrays."""
    n1, n2, n3 = ngfft
    def freqs(n):
        i = np.arange(n)
        return np.where(i <= n // 2, i, i - n)
    m1, m2, m3 = freqs(n1), freqs(n2), freqs(n3)
    I1, I2, I3 = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing="ij")
    miller = np.stack([M1.ravel(), M2.ravel(), M3.ravel()], axis=1).astype(np.int32)
    return miller, (I1.ravel(), I2.ravel(), I3.ravel())


def dense_sphere(ngfft, rprimd, recv):
    """G-sphere miller list (plan B4): the largest sphere inscribed in the
    FFT box [-n/2, n/2). With ABINIT's boxcut >= 2 this contains the full
    ecutrho = 4*ecut sphere, and matches the QE miller_g convention (an
    ecutrho G-sphere, never the raw box). Returns (miller (ngm,3),
    (i1,i2,i3) fft-bin indices of the kept points)."""
    miller, (I1, I2, I3) = dense_grid(ngfft)
    Gn = np.linalg.norm(miller @ np.asarray(recv, float), axis=1)
    # distance from origin to box face d: (n_d/2) * 2*pi/|a_d|
    gmax = min((ngfft[d] / 2.0) * 2.0 * np.pi / np.linalg.norm(rprimd[d])
               for d in range(3))
    keep = Gn < gmax * (1.0 - 1e-9)
    return (np.ascontiguousarray(miller[keep]),
            (I1[keep], I2[keep], I3[keep]))


def write_hamiltonian_nc(h5root, w, vtrial, psps, rho_den=None, xc_name="pbe"):
    """Write /Hamiltonian (norm-conserving) into an open h5py root group.
    w: WFK dict (kg, npw, kpts_crys, recv, xred, typat, rprimd, nkpt=nkpts_ibz,
       nsppol, nspinor). psps: list of parsed psp8 dicts, indexed by (species-1).
    rho_den: real-space density (n1,n2,n3) from the ABINIT DEN (optional) --
    enables real vxc/vxc_with_nlcc export (plan B2) with functional xc_name.
    Conventions validated to 5.6e-9 Ha (validate_h0.py).
    """
    import h5py
    VLEN = h5py.special_dtype(vlen=str)
    def wattr_i(g, k, v): g.attrs.create(k, np.int32(v))
    def wreal(g, k, a, dt=None):
        a = np.ascontiguousarray(a); g.create_dataset(k, data=a if dt is None else a.astype(dt))
    def wcplx(g, k, a):
        a = np.ascontiguousarray(a).astype(np.complex128)
        ds = g.create_dataset(k, data=np.ascontiguousarray(np.stack([a.real, a.imag], -1)))
        ds.attrs.create("__complex__", "1", dtype=VLEN)

    recv, rprimd = w["recv"], w["rprimd"]
    vol = abs(np.linalg.det(rprimd))
    tau = w["xred"] @ rprimd                         # cartesian atom positions (bohr)
    typat = w["typat"]                                # 1-based species per atom
    nat = len(typat)
    nsp = len(psps)
    nspin = w["nsppol"]; npol = 1
    nk = w["nkpt"]                                    # nkpts_ibz (nosym: all k)

    # projector bookkeeping. proj_per_atom is per-SPECIES (length nsp, QE
    # nh(1:nsp) convention, plan B2 / F10a); projector_offset stays per-atom.
    nh_sp = np.array([p["nh"] for p in psps], dtype=np.int32)
    nh_atom = np.array([nh_sp[typat[a] - 1] for a in range(nat)], dtype=np.int32)
    proj_off = np.concatenate([[0], np.cumsum(nh_atom)[:-1]]).astype(np.int32)
    nkb = int(nh_atom.sum())
    nhm = int(nh_sp.max())

    # dense grid + local potentials (miller_g = ecutrho-style sphere, plan B4)
    ngfft = tuple(int(x) for x in np.array(vtrial).shape[1:4])
    miller_g, (I1, I2, I3) = dense_sphere(ngfft, rprimd, recv)
    ngm = miller_g.shape[0]
    v = np.array(vtrial)[0, :, :, :, 0].astype(float)
    Vg = np.fft.fftn(v) / v.size                      # <G|V_KS|G'> = (1/N) sum_r ...
    vks_flat = Vg[I1, I2, I3]                          # (ngm,) complex, matches miller_g
    Gcart = miller_g @ recv
    Gn = np.linalg.norm(Gcart, axis=1)
    # total ionic local potential Sum_a e^{-iG.tau_a} vloc_species(G)  (provisional; CoQuí-read pins use)
    vloc_tot = np.zeros(ngm, dtype=complex)
    vloc_sp = {}
    for isp in range(nsp):
        vloc_sp[isp] = vloc_of_g(psps[isp], Gn, vol, psps[isp]["zion"])
    for a in range(nat):
        isp = typat[a] - 1
        vloc_tot += np.exp(-1j * (Gcart @ tau[a])) * vloc_sp[isp]

    # ---- write ----
    H = h5root.create_group("Hamiltonian")
    # schema_version 2: HARTREE on disk for all energy-valued datasets
    # (contract: notes/paw_implementation_plan.md, plan B4). ABINIT is
    # natively Ha, so no conversion factors anywhere below.
    H.attrs.create("schema_version", np.int32(2))
    H.attrs.create("pp_type", "ncpp", dtype=VLEN)
    g = H.create_group("ncpp")
    for k, val in [("number_of_nspins", nspin), ("number_of_polarizations", npol),
                   ("number_of_kpoints", nk), ("max_npw", int(w["npw"][:nk].max())),
                   ("number_of_atoms", nat), ("number_of_species", nsp),
                   ("total_num_of_proj", nkb), ("max_proj_per_atom", nhm),
                   ("ngm", ngm), ("lspinorbit_nl", 0), ("lspinorbit_loc", 0)]:
        wattr_i(g, k, val)
    wreal(g, "miller_g", miller_g, np.int32)
    # Hartree on disk (schema_version 2, plan B4): readers skip the legacy
    # 0.5 Ry->Ha scale for version >= 2, so ABINIT's native-Ha values are
    # written unmodified.
    wcplx(g, "pp_local_component", vloc_tot)              # non-SO NC path: complex (ngm)
    wcplx(g, "pp_local_component_nc",                     # SO / newer builds: (nspin,npol^2,ngm)
          np.tile(vloc_tot, (nspin, npol * npol, 1)))
    wcplx(g, "scf_local_potential",
          vks_flat.reshape(nspin, npol * npol, ngm))
    # vxc / vxc_with_nlcc (Ry on disk, add_vxc scales x0.5): real values when
    # a DEN was provided (plan B2). vxc = f(rho_DEN); vxc_with_nlcc adds the
    # psp8 model core charge (NLCC) when the pseudo carries one.
    if rho_den is not None:
        import xc_functionals as xcf
        rho_r = np.asarray(rho_den, float)
        if rho_r.ndim != 3:
            raise NotImplementedError(
                "abinit2coqui NC vxc: nsppol=1 only (got shape %s)" % (rho_r.shape,))
        if tuple(rho_r.shape) != tuple(ngfft):
            raise RuntimeError("abinit2coqui: DEN grid %s != POT grid %s"
                               % (rho_r.shape, ngfft))
        vxc_r = xcf.vxc_grid(rho_r, recv, xc_name)
        core_g = np.zeros(ngm, complex)
        Gs = np.where(Gn < 1e-8, 1.0, Gn)
        for a in range(nat):
            p = psps[typat[a] - 1]
            if p.get("rhoc") is None:
                continue
            rr = p["r"]
            nc = p["rhoc"] / (4.0 * np.pi)      # psp8 stores 4*pi*n_c(r)
            ft = (4 * np.pi / vol) * np.trapezoid(
                nc[None, :] * np.sin(np.outer(Gs, rr)) * rr[None, :]
                / Gs[:, None], rr, axis=1)
            ft[Gn < 1e-8] = (4 * np.pi / vol) * np.trapezoid(nc * rr * rr, rr)
            core_g += np.exp(-1j * (Gcart @ tau[a])) * ft
        if np.any(core_g != 0.0):
            box = np.zeros(tuple(int(n) for n in ngfft), complex)
            idx = [np.mod(miller_g[:, d], ngfft[d]) for d in range(3)]
            box[idx[0], idx[1], idx[2]] = core_g
            core_r = np.real(np.fft.ifftn(box)) * rho_r.size
            vxc_nlcc_r = xcf.vxc_grid(rho_r + core_r, recv, xc_name)
        else:
            vxc_nlcc_r = vxc_r
        vg = np.fft.fftn(vxc_r) / vxc_r.size
        vgn = np.fft.fftn(vxc_nlcc_r) / vxc_nlcc_r.size
        wcplx(g, "vxc", vg[I1, I2, I3].reshape(nspin, npol * npol, ngm))
        wcplx(g, "vxc_with_nlcc",
              vgn[I1, I2, I3].reshape(nspin, npol * npol, ngm))
    wreal(g, "proj_per_atom", nh_sp, np.int32)
    wreal(g, "projector_offset", proj_off, np.int32)
    wreal(g, "npw", np.array(w["npw"][:nk]), np.int32)
    wreal(g, "atomic_id", (typat - 1).astype(np.int32), np.int32)
    # dion (nsp, nhm, nhm) diagonal, Hartree on disk (schema_version 2)
    dion = np.zeros((nsp, nhm, nhm), dtype=np.float64)
    for isp in range(nsp):
        _, meta = build_beta_k(psps[isp], np.zeros((1, 3), int), np.zeros(3), recv,
                               np.zeros(3), vol)
        dd = np.array([m[1] for m in meta])
        dion[isp, :len(dd), :len(dd)] = np.diag(dd)         # psp8 ekb are Ha
    wreal(g, "dion", dion, np.float64)
    # per-k projectors
    for ik in range(nk):
        n = int(w["npw"][ik])
        mill_k = w["kg"][ik][:n]
        kcart = w["kpts_crys"][ik] @ recv
        cols = []
        for a in range(nat):
            beta, _ = build_beta_k(psps[typat[a] - 1], mill_k, kcart, recv, tau[a], vol)
            cols.append(beta)
        beta_all = np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), complex)
        wreal(g, "miller_k" + str(ik), mill_k.astype(np.int32), np.int32)
        # CoQui reads projector_k{ik} with hyperslab {row ib, cols npw} -> shape (nkb, npw):
        # projectors as ROWS. beta_all is (npw, nkb), so transpose.
        wcplx(g, "projector_k" + str(ik), beta_all.T)
    return dict(ngm=ngm, nkb=nkb, nhm=nhm, ngfft=ngfft)


if __name__ == "__main__":
    # smoke test of the transforms on the Si psp8 (no CoQuí needed)
    import sys
    from abinit_psp8 import parse_psp8
    psp = parse_psp8(sys.argv[1])
    vol = 10.26 ** 3 / 4.0
    recv = 2 * np.pi * np.linalg.inv(
        np.array([[0, 5.13, 5.13], [5.13, 0, 5.13], [5.13, 5.13, 0]])).T
    mill = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
    beta, meta = build_beta_k(psp, mill, np.zeros(3), recv, np.zeros(3), vol)
    print("beta_k shape:", beta.shape, "(npw, nh) -- nh should be 18 for Si")
    print("per-column (l,ekb):", meta[:6], "...")
    Gn = np.linalg.norm(mill @ recv, axis=1)
    vg = vloc_of_g(psp, Gn, vol, psp["zion"])
    print("Vloc(G) sample:", np.round(vg, 4))
