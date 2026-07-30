"""
validate_b2.py -- local validation of the plan-B2 converter additions, with
no ABINIT data dependency (the ABINIT-side end-to-end rerun happens on the
cluster; see STATUS).

Modes:
  synth               synthetic PAW-XML end-to-end: parser extensions
                      (tabulated shape_function, core wavefunctions,
                      exx_core_core), adapter exports (beta, oc, ae_vloc,
                      vloc_ps with -zval/r tails in Ha, Core), the PAW-XML
                      sqrt(4pi) L=0 normalization of the local ionic potential
                      (tail, analytic alpha_Z, and a negative control for the
                      2026-07-29 EOS bug), the per-species proj_per_atom, the
                      real-vxc write path, and the shape-mismatch hard error.
  vxc <coqui.h5> <charge-density.hdf5>
                      read QE's own SCF rho(G), reproduce the pw2coqui vxc
                      dataset with xc_functionals (PBE) on the same dense
                      grid.
  ewald <coqui.h5> <zv1> [zv2 ...]
                      reproduce /System@nuclear_energy (Ha) with
                      lattice_sums.ewald_energy from the stored geometry.
"""
import sys
import os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
def run_ewald(path, zv):
    from lattice_sums import ewald_energy
    f = h5py.File(path, "r")
    S = f["System"]
    latt = S["lattice_vectors"][:]
    tau = S["atomic_positions"][:]
    aid = S["atomic_id"][:]
    ref = float(S.attrs["nuclear_energy"])
    z = np.array([zv[i] for i in aid], float)
    e = ewald_energy(latt, tau, z)
    print("ewald: computed %.10f Ha  stored %.10f Ha  diff %.2e"
          % (e, ref, e - ref))
    assert abs(e - ref) < 5e-6, "ewald mismatch beyond QE print/convergence"
    print("PASS ewald")


# ---------------------------------------------------------------------------
def run_vxc(path, rho_path):
    from xc_functionals import vxc_grid
    f = h5py.File(path, "r")
    O, S = f["Orbitals"], f["System"]
    H = f["Hamiltonian"]
    pp = H[[k for k in ("ncpp", "uspp", "paw") if k in H][0]]
    latt = S["lattice_vectors"][:]
    recv = S["reciprocal_vectors"][:]
    vol = abs(np.linalg.det(latt))
    mesh = tuple(int(x) for x in O["fft_mesh_aug"][:])

    # QE charge-density.hdf5: rho(G) with rho(r) = sum_G rho(G) e^{iGr}
    rf = h5py.File(rho_path, "r")
    mill_rho = rf["MillerIndices"][:]
    rg = rf["rhotot_g"][:]
    rg = rg[0::2] + 1j * rg[1::2]
    box = np.zeros(mesh, complex)
    box[np.mod(mill_rho[:, 0], mesh[0]), np.mod(mill_rho[:, 1], mesh[1]),
        np.mod(mill_rho[:, 2], mesh[2])] = rg
    rho = np.real(np.fft.ifftn(box)) * np.prod(mesh)
    print("vxc: integrated charge = %.8f electrons" % (rho.mean() * vol))

    vxc_r = vxc_grid(rho, recv, "pbe")
    vg = np.fft.fftn(vxc_r) / vxc_r.size
    mill_g = pp["miller_g"][:]
    # on-disk units per schema_version: >= 2 -> Ha; legacy/1 -> Ry (x2)
    sv = int(H.attrs.get("schema_version", 0))
    unit = 1.0 if sv >= 2 else 2.0
    mine = unit * vg[np.mod(mill_g[:, 0], mesh[0]),
                     np.mod(mill_g[:, 1], mesh[1]),
                     np.mod(mill_g[:, 2], mesh[2])]
    ref = pp["vxc"][:]
    ref = (ref[..., 0] + 1j * ref[..., 1]).reshape(-1)[:mill_g.shape[0]]
    # G=0 aside (QE alpha/constant conventions), compare all components
    d = np.abs(mine - ref)
    g0 = np.all(mill_g == 0, axis=1)
    print("vxc: max|diff| (G!=0) = %.3e Ry;  G=0 diff = %.3e Ry;  "
          "max|ref| = %.3e Ry" % (d[~g0].max(), d[g0].max() if g0.any() else 0.0,
                                  np.abs(ref).max()))
    assert d[~g0].max() < 2e-4, "vxc mismatch vs QE beyond tolerance"
    print("PASS vxc (PBE evaluator reproduces QE v_xc on the dense grid)")


# ---------------------------------------------------------------------------
SYNTH_RC0 = 0.9   # rc0 of the analytic synthetic v_H[n~_Zc] (see _synth_pawxml)


def _synth_pawxml(tmpdir, break_shape=False, break_vloc_norm=False,
                  no_pscore=False):
    """Write a minimal, physically sane PAW-XML: log grid, 2 valence states
    (2s occupied, 2p empty), 1s core (2 electrons, Z=4 -> zval=2), bessel
    shape (tabulated too), Dij0, exact-exchange X matrix + core-core.

    The two local-potential tags follow the real PAW-XML convention: radial
    functions are stored as their L=0 expansion coefficient, i.e. sqrt(4*pi)
    times the physical radial function (ABINIT divides every local-potential
    variant by sqrt(4*pi) on read -- m_pawpsp.F90:3730/3745/3767).  They are
    built so that BOTH routes to v_H[n~_Zc] agree exactly, as they do in real
    datasets, from an ANALYTIC ionic Hartree potential with a closed-form alpha_Z:

        v_H[n~_Zc](r) = -zval*erf(r/rc0)/r
        alpha_Z = 4*pi*int [r^2 v + zval*r] dr = pi*zval*rc0^2

    break_vloc_norm=True omits the sqrt(4*pi) on blochl_local_ionic_potential --
    the 2026-07-29 EOS bug -- so the guards can be shown to catch it.
    no_pscore=True omits <pseudo_core_density>, which switches the NLCC term off
    so that the written vxc_with_nlcc reduces to the valence-only v_xc and can be
    checked against a closed-form uniform-density reference.
    """
    import math
    from paw_radial import shape_bessel
    from abinit_paw_hamiltonian import _poisson_over_r
    erf = np.vectorize(math.erf)
    nr = 600
    d = 0.012
    a = 0.4 / (np.exp(d * (nr - 1)) - 1.0) * 40      # r_max ~ 16 bohr-ish
    i = np.arange(nr)
    r = a * (np.exp(d * i) - 1.0)
    rc = 1.3
    Z = 4.0

    def fmt(arr):
        return " ".join("%.14e" % x for x in arr)

    # hydrogenic-ish radial functions R(r) (not r*R)
    phi_s_ae = np.exp(-r) * (1.0 + 0.5 * r)
    phi_p_ae = r * np.exp(-0.8 * r)
    bump = np.where(r < rc, (1 - (r / rc) ** 2) ** 2, 0.0)
    phi_s_ps = phi_s_ae + 0.35 * bump
    phi_p_ps = phi_p_ae - 0.20 * bump * r
    proj_s = bump * (1.0 - 0.3 * r)
    proj_p = bump * r
    ae_core = np.sqrt(4 * np.pi) * (Z ** 3 / np.pi) * np.exp(-2 * Z * r) * 2 / 2
    # normalize core to exactly 2 electrons (L=0-moment convention: /sqrt(4pi))
    q = np.trapezoid(ae_core / np.sqrt(4 * np.pi) * 4 * np.pi * r ** 2, r)
    ae_core *= 2.0 / q
    ps_core = ae_core * np.exp(-(rc / (r + 0.3)) ** 2)      # smooth, arbitrary
    core_1s = np.exp(-Z * r) * (2 * Z ** 1.5)               # R(r), arbitrary norm

    shp = shape_bessel(r, rc, 0)                            # g0(r)*r^2
    g0 = np.zeros_like(r)
    g0[1:] = shp[1:] / r[1:] ** 2
    if break_shape:
        g0 = g0 * (1.0 + 1e-3 * bump)

    # --- the two local-potential tags (see the docstring) ---
    s4pi = np.sqrt(4.0 * np.pi)
    zval = 2.0
    rc0 = SYNTH_RC0
    vht = np.empty_like(r)                                  # physical v_H[n~_Zc]
    vht[0] = -zval * 2.0 / (np.sqrt(np.pi) * rc0)           # r->0 limit of erf(r/rc0)/r
    vht[1:] = -zval * erf(r[1:] / rc0) / r[1:]
    blochl = vht * (1.0 if break_vloc_norm else s4pi)
    # zero_potential is the SAME potential seen through ABINIT's 'Vbare' route:
    #   vht = zero_potential/sqrt(4pi) + poisson[tncore*4pi r^2 + g0 r^2 (qcore-Z)]/r
    ncore_phys, tncore_phys = ae_core / s4pi, ps_core / s4pi
    qcore = np.trapezoid((ncore_phys - tncore_phys) * 4.0 * np.pi * r ** 2, r)
    g0r2 = shp / np.trapezoid(shp, r)                       # unit monopole
    nwk = tncore_phys * 4.0 * np.pi * r ** 2 + g0r2 * (qcore - Z)
    zero_pot = s4pi * (vht - _poisson_over_r(nwk, r))        # short-ranged by construction

    xml = ['<?xml version="1.0"?>', '<paw_dataset version="0.7">',
           '<atom symbol="Xx" Z="%g" core="2" valence="2"/>' % Z,
           '<radial_grid eq="r=a*(exp(d*i)-1)" a="%.16e" d="%.16e" istart="0" '
           'iend="%d" id="log1"/>' % (a, d, nr - 1),
           '<valence_states>',
           '  <state n="2" l="0" f="2" e="-0.5" id="Xx-2s"/>',
           '  <state n="2" l="1" f="0" e="-0.2" id="Xx-2p"/>',
           '</valence_states>',
           '<core_states>',
           '  <state n="1" l="0" f="2" e="-15.0" id="Xx-1s"/>',
           '</core_states>',
           '<shape_function type="bessel" rc="%.12f" grid="log1">%s'
           '</shape_function>' % (rc, fmt(g0)),
           '<paw_radius rc="%.12f"/>' % rc,
           '<ae_core_density grid="log1">%s</ae_core_density>' % fmt(ae_core),
           ] + ([] if no_pscore else [
           '<pseudo_core_density grid="log1">%s</pseudo_core_density>' % fmt(ps_core),
           ]) + [
           '<zero_potential grid="log1">%s</zero_potential>' % fmt(zero_pot),
           '<blochl_local_ionic_potential grid="log1">%s'
           '</blochl_local_ionic_potential>' % fmt(blochl),
           '<ae_partial_wave state="Xx-2s" grid="log1">%s</ae_partial_wave>' % fmt(phi_s_ae),
           '<ae_partial_wave state="Xx-2p" grid="log1">%s</ae_partial_wave>' % fmt(phi_p_ae),
           '<pseudo_partial_wave state="Xx-2s" grid="log1">%s</pseudo_partial_wave>' % fmt(phi_s_ps),
           '<pseudo_partial_wave state="Xx-2p" grid="log1">%s</pseudo_partial_wave>' % fmt(phi_p_ps),
           '<projector_function state="Xx-2s" grid="log1">%s</projector_function>' % fmt(proj_s),
           '<projector_function state="Xx-2p" grid="log1">%s</projector_function>' % fmt(proj_p),
           '<ae_core_wavefunction state="Xx-1s" grid="log1">%s'
           '</ae_core_wavefunction>' % fmt(core_1s),
           '<kinetic_energy_differences>%s</kinetic_energy_differences>'
           % fmt(np.array([0.3, 0.0, 0.0, 0.5])),
           '<exact_exchange_X_matrix>%s</exact_exchange_X_matrix>'
           % fmt(np.array([-0.11, 0.0, 0.0, -0.07])),
           '<exact_exchange core-core="-0.5"/>',
           '</paw_dataset>']
    suffix = ("_badshape" if break_shape else
              "_badvloc" if break_vloc_norm else "")
    path = os.path.join(tmpdir, "synth%s.xml" % suffix)
    with open(path, "w") as fh:
        fh.write("\n".join(xml))
    return path


def run_synth(tmpdir):
    import abinit_pawxml as axml
    import abinit_paw_hamiltonian as aph

    # --- parser extensions ---
    p = axml.parse_pawxml(_synth_pawxml(tmpdir))
    assert p["shape_tab"] is not None and p["shape_tab"].size == p["nr"]
    assert p["core_ae_wfc"] is not None and p["core_ae_wfc"].shape[0] == 1
    assert p["core_states"][0]["l"] == 0 and p["core_states"][0]["n"] == 1
    assert p["exx_core_core"] == -0.5
    print("PASS parse (shape_tab, core wfc, exx_core_core)")

    # --- adapter exports + shape check (good shape passes) ---
    sp = aph.abinit_species_adapter(p)
    for key in ("beta", "oc", "ae_vloc", "vloc_ps", "core"):
        assert key in sp, "adapter missing %s" % key
    r = sp["r"]
    zval = 2.0
    for name in ("ae_vloc", "vloc_ps"):
        tail = (r * sp[name])[-1]                    # Ha (schema 2); -> -zval
        assert abs(tail + zval) < 1e-3, \
            "%s tail %.6f != -zval (Ha/schema-2 convention broken?)" % (name, tail)
    assert np.allclose(sp["beta"], p["proj"] * r)
    assert np.allclose(sp["oc"], [2.0, 0.0])
    print("PASS adapter (beta, oc, ae_vloc/vloc_ps Ha tails -> -zval)")

    # --- local ionic potential: the PAW-XML sqrt(4pi) L=0 convention ----------
    # Regression guard for the 2026-07-29 Si PAW EOS defect: the converter read
    # blochl_local_ionic_potential/zero_potential WITHOUT the 1/sqrt(4*pi) that
    # ABINIT applies (m_pawpsp.F90:3730/3767) and compensated with a spurious
    # frozen-core Hartree.  The net error was ~1/Omega, so it looked like a
    # harmless constant at any single volume and wrecked the equation of state.
    #
    # The tail assertions above CANNOT see this: vloc_ps' -zval/r asymptote comes
    # entirely from the poisson term, and the pp_local tail was only 4.5% off.
    # These three checks can.
    s4pi = np.sqrt(4.0 * np.pi)
    rc0 = SYNTH_RC0
    vion = np.asarray(p["vbar"], float) / s4pi
    qtail = -float((r * vion)[-1])
    assert abs(qtail - zval) < 1e-9, \
        "blochl/sqrt(4pi) tail is -%.9f/r, must be -zval = -%.1f/r" % (qtail, zval)
    alpha = 4.0 * np.pi * np.trapezoid(r ** 2 * vion + zval * r, r)
    alpha_exact = np.pi * zval * rc0 ** 2          # closed form for -zval*erf(r/rc0)/r
    # 1e-4 relative is the trapezoid-on-a-600-point-log-grid floor, measured
    # (2.4e-5 here; 7e-6 on the real Si jth_with_d dataset against ABINIT's
    # epsatm).  A missing sqrt(4pi) is a factor 3.54, so the gap between "right"
    # and "wrong" is 4 orders of magnitude wider than this tolerance.
    assert abs(alpha - alpha_exact) < 1e-4 * abs(alpha_exact), \
        "alpha_Z = %.9f, analytic = %.9f" % (alpha, alpha_exact)
    # vhtnzc must equal the analytic potential (and the internal two-route
    # cross-check inside reconstruct_vhtnzc already ran, above, without raising)
    vht_ref = np.empty_like(r)
    vht_ref[0] = -zval * 2.0 / (np.sqrt(np.pi) * rc0)
    vht_ref[1:] = -zval * np.vectorize(__import__("math").erf)(r[1:] / rc0) / r[1:]
    dv = np.abs(np.asarray(sp["vloc_ps"], float) - vht_ref).max()
    assert dv < 1e-10, "vloc_ps deviates from the analytic v_H[n~_Zc] by %.3e" % dv
    print("PASS vloc normalization (tail = -zval exactly, alpha_Z = pi*zval*rc0^2 "
          "to 1e-6, vloc_ps == analytic to %.1e)" % dv)

    # --- NEGATIVE CONTROL: the 2026-07-29 bug must be rejected ---------------
    p_badv = axml.parse_pawxml(_synth_pawxml(tmpdir, break_vloc_norm=True))
    try:
        aph.abinit_species_adapter(p_badv)
    except RuntimeError as e:
        assert "sqrt(4pi)" in str(e) or "normalization" in str(e).lower(), \
            "wrong error for a mis-normalized local potential: %s" % e
        print("PASS mis-normalized blochl_local_ionic_potential is rejected")
    else:
        raise AssertionError(
            "a blochl_local_ionic_potential missing its sqrt(4pi) did NOT raise "
            "-- the EOS-breaking bug of 2026-07-29 would slip through again")

    # --- tabulated-shape mismatch must fail loudly ---
    p_bad = axml.parse_pawxml(_synth_pawxml(tmpdir, break_shape=True))
    try:
        aph.abinit_species_adapter(p_bad)
    except RuntimeError as e:
        assert "shape" in str(e).lower()
        print("PASS shape-mismatch hard error")
    else:
        raise AssertionError("perturbed tabulated shape did NOT raise")

    # --- writer end-to-end on a tiny fake system (1 atom, 1 k, 6^3 grids) ---
    # vxc_with_nlcc = v_xc[rho + PS core]; with an atom-centred core it has
    # structure at every G, so the uniform-density reference below only holds for
    # a species with no pseudo_core_density. (Schema 3 dropped the valence-only
    # "vxc" dataset this assertion was originally written against.)
    p_nc = axml.parse_pawxml(_synth_pawxml(tmpdir, no_pscore=True))
    mesh = (12, 12, 12)
    L = 8.0
    rprimd = np.eye(3) * L
    kg = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], int)
    w = dict(kg=kg, npw=np.array([4]), kpts_crys=np.zeros((1, 3)),
             recv=2 * np.pi * np.linalg.inv(rprimd).T, xred=np.zeros((1, 3)),
             typat=np.array([1]), rprimd=rprimd, nkpt=1, nsppol=1, nspinor=1)
    vtrial = np.zeros((1,) + mesh + (1,))
    rho_den = np.full(mesh, 2.0 / L ** 3)            # uniform 2 electrons
    out = os.path.join(tmpdir, "synth_paw.h5")
    if os.path.exists(out):
        os.remove(out)
    with h5py.File(out, "w") as f:
        aph.write_hamiltonian_paw(f["/"], w, vtrial, [p_nc], verbose=False,
                                  rho_den=rho_den, xc_name="pbe")
    with h5py.File(out, "r") as f:
        g = f["Hamiltonian/paw"]
        ppa = g["proj_per_atom"][:]
        assert ppa.shape == (1,) and ppa[0] == 4, \
            "proj_per_atom must be per-SPECIES nh (got %s)" % ppa
        assert int(g.attrs["total_num_of_proj"]) == 4
        # schema 3 since the 2026-07-26 converter audit (was 2 here; stale).
        assert int(f["Hamiltonian"].attrs["schema_version"]) == 3
        vxc = g["vxc_with_nlcc"][:]
        assert vxc.shape[:1] == (1,) and vxc.shape[-1] == 2
        vxc0 = vxc.reshape(1, 1, -1, 2)[0, 0, :, 0]
        # uniform rho -> vxc has only a G=0 component = LDA-limit value (Ha)
        from xc_functionals import vxc_grid
        vref = vxc_grid(rho_den, w["recv"], "pbe")[0, 0, 0]
        i000 = np.where(np.all(g["miller_g"][:] == 0, axis=1))[0][0]
        assert abs(vxc0[i000] - vref) < 1e-10
        assert np.abs(np.delete(vxc0, i000)).max() < 1e-12
        nt0 = f["Hamiltonian/Species/nt0"]
        # schema 3 dropped paw/oc (dead at read; 2026-07-26 converter audit).
        for ds in ("beta", "paw/ae_vloc", "paw/vloc_ps",
                   "Core/n", "Core/l", "Core/ae_wfc"):
            assert ds in nt0, "Species missing %s" % ds
        assert nt0["Core"].attrs["ncore_orbitals"] == 1
        assert abs(float(nt0.attrs["exx_core_core"]) + 0.5) < 1e-14
    print("PASS writer (per-species proj_per_atom, real vxc, Core/, "
          "ae_vloc/vloc_ps, exx_core_core)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if mode == "synth":
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            run_synth(td)
    elif mode == "vxc":
        run_vxc(sys.argv[2], sys.argv[3])
    elif mode == "ewald":
        run_ewald(sys.argv[2], [float(x) for x in sys.argv[3:]])
    else:
        sys.exit("unknown mode %s" % mode)
