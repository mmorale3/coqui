"""
validate_b2.py -- local validation of the plan-B2 converter additions, with
no ABINIT data dependency (the ABINIT-side end-to-end rerun happens on the
cluster; see STATUS).

Modes:
  synth               synthetic PAW-XML end-to-end: parser extensions
                      (tabulated shape_function, core wavefunctions,
                      exx_core_core), adapter exports (beta, oc, ae_vloc,
                      vloc_ps in Ry with -2*zval/r tails, Core), the
                      per-species proj_per_atom, the real-vxc write path, and
                      the shape-mismatch hard error.
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
    mine = 2.0 * vg[np.mod(mill_g[:, 0], mesh[0]),
                    np.mod(mill_g[:, 1], mesh[1]),
                    np.mod(mill_g[:, 2], mesh[2])]          # Ry, pw2coqui units
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
def _synth_pawxml(tmpdir, break_shape=False):
    """Write a minimal, physically sane PAW-XML: log grid, 2 valence states
    (2s occupied, 2p empty), 1s core (2 electrons, Z=4 -> zval=2), bessel
    shape (tabulated too), Dij0, exact-exchange X matrix + core-core."""
    from paw_radial import shape_bessel
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
    zero_pot = -0.5 * bump
    core_1s = np.exp(-Z * r) * (2 * Z ** 1.5)               # R(r), arbitrary norm

    shp = shape_bessel(r, rc, 0)                            # g0(r)*r^2
    g0 = np.zeros_like(r)
    g0[1:] = shp[1:] / r[1:] ** 2
    if break_shape:
        g0 = g0 * (1.0 + 1e-3 * bump)

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
           '<pseudo_core_density grid="log1">%s</pseudo_core_density>' % fmt(ps_core),
           '<zero_potential grid="log1">%s</zero_potential>' % fmt(zero_pot),
           '<blochl_local_ionic_potential grid="log1">%s'
           '</blochl_local_ionic_potential>' % fmt(zero_pot - 2.0 / (r + 0.05)),
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
    path = os.path.join(tmpdir, "synth%s.xml" % ("_bad" if break_shape else ""))
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
        tail = (r * sp[name] / 2.0)[-1]              # Ry/2 = Ha; -> -zval
        assert abs(tail + zval) < 1e-3, \
            "%s tail %.6f != -zval (Ry convention broken?)" % (name, tail)
    assert np.allclose(sp["beta"], p["proj"] * r)
    assert np.allclose(sp["oc"], [2.0, 0.0])
    print("PASS adapter (beta, oc, ae_vloc/vloc_ps Ry tails -> -zval)")

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
        aph.write_hamiltonian_paw(f["/"], w, vtrial, [p], verbose=False,
                                  rho_den=rho_den, xc_name="pbe")
    with h5py.File(out, "r") as f:
        g = f["Hamiltonian/paw"]
        ppa = g["proj_per_atom"][:]
        assert ppa.shape == (1,) and ppa[0] == 4, \
            "proj_per_atom must be per-SPECIES nh (got %s)" % ppa
        assert int(g.attrs["total_num_of_proj"]) == 4
        vxc = g["vxc"][:]
        assert vxc.shape[:1] == (1,) and vxc.shape[-1] == 2
        vxc0 = vxc.reshape(1, 1, -1, 2)[0, 0, :, 0]
        # uniform rho -> vxc has only a G=0 component = LDA-limit value (Ry)
        from xc_functionals import vxc_grid
        vref = 2.0 * vxc_grid(rho_den, w["recv"], "pbe")[0, 0, 0]
        i000 = np.where(np.all(g["miller_g"][:] == 0, axis=1))[0][0]
        assert abs(vxc0[i000] - vref) < 1e-10
        assert np.abs(np.delete(vxc0, i000)).max() < 1e-12
        nt0 = f["Hamiltonian/Species/nt0"]
        for ds in ("beta", "paw/ae_vloc", "paw/vloc_ps", "paw/oc",
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
