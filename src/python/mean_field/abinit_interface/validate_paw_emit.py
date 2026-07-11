"""
validate_paw_emit.py -- end-to-end local test of the PAW augmentation writer.

Builds a normalized species dict from the QE reference h5 (QE-adapter), runs
build_paw_augmentation + write_paw_augmentation + write_species_block into a
scratch h5, reads it back, and compares to the QE reference's own /Hamiltonian/paw
augmentation datasets and Species/Onecenter/deltaC.  Validates the h5 encoding
(names, shapes, nda-complex layout, 0-based species index) together with the
compute kernels -- with no ABINIT dependency.
"""
import os
import numpy as np
import h5py
import abinit_paw_hamiltonian as aph

REF = "../../../../tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5"
OUT = "/tmp/paw_emit_test.h5"


def qe_species_adapter(f, nt):
    """Normalized species dict from QE reference Species/nt{nt} (u=r*R form)."""
    sp = f["Hamiltonian/Species/nt%d" % nt]
    return dict(
        r=sp["r"][:], rab=sp["rab"][:], mesh=int(sp.attrs["mesh"]),
        kkbeta=int(sp.attrs["kkbeta"]), nqlc=int(sp.attrs["nqlc"]),
        lmax_rho=int(sp.attrs["lmax_rho"]), lll=sp["lll"][:].astype(int),
        u_ae=sp["aewfc"][:], u_ps=sp["pswfc"][:],
        pfunc=sp["paw/pfunc"][:], ptfunc=sp["paw/ptfunc"][:],
        qfuncl=sp["qfuncl"][:],                 # QE stores qfuncl directly
        dij0=sp["dion"][:], beta=sp["beta"][:],
        zp=float(sp.attrs["zp"]),
    )


def main():
    f = h5py.File(REF, "r")
    recip = f["System"]["reciprocal_vectors"][:]
    lat = f["System"]["lattice_vectors"][:]
    omega = abs(np.linalg.det(lat))
    miller_g = f["Hamiltonian/paw/miller_g"][:]
    ngm = miller_g.shape[0]
    atomic_id = f["Hamiltonian/paw/atomic_id"][:]
    nsp = int(f["Hamiltonian/paw"].attrs["number_of_species"])
    species = [qe_species_adapter(f, nt) for nt in range(nsp)]

    aug = aph.build_paw_augmentation(species, miller_g, recip, omega)

    if os.path.exists(OUT):
        os.remove(OUT)
    with h5py.File(OUT, "w") as o:
        H = o.create_group("Hamiltonian")
        H.attrs.create("pp_type", "paw", dtype=h5py.special_dtype(vlen=str))
        g = H.create_group("paw")
        g.attrs.create("number_of_species", np.int32(nsp))
        g.attrs.create("ngm", np.int32(ngm))
        g.create_dataset("miller_g", data=miller_g.astype(np.int32))
        aph.write_paw_augmentation(o, aug, atomic_id, miller_g, ngm, h5py)
        aph.write_species_block(o, species, aug, h5py)

    # ---- read back + compare to reference ----
    o = h5py.File(OUT, "r")
    def cplx(ds):
        a = ds[:]; return a[..., 0] + 1j * a[..., 1]

    checks = []
    # ijtoh
    checks.append(("ijtoh", np.abs(o["Hamiltonian/paw/ijtoh"][:] -
                                   f["Hamiltonian/paw/ijtoh"][:]).max()))
    # qq_nt
    checks.append(("qq_nt", np.abs(o["Hamiltonian/paw/qq_nt"][:] -
                                   f["Hamiltonian/paw/qq_nt"][:]).max()))
    # augmentation_function per species (relative)
    for nt in range(nsp):
        a = cplx(o["Hamiltonian/paw/augmentation_function_isp%d" % nt])
        b = cplx(f["Hamiltonian/paw/augmentation_function_isp%d" % nt])
        checks.append(("aug_isp%d (rel)" % nt, np.abs(a - b).max() / np.abs(b).max()))
    # Onecenter/deltaC per species (relative)
    for nt in range(nsp):
        a = o["Hamiltonian/Species/nt%d/Onecenter/deltaC" % nt][:]
        b = f["Hamiltonian/Species/nt%d/Onecenter/deltaC" % nt][:]
        checks.append(("deltaC nt%d (rel)" % nt, np.abs(a - b).max() / np.abs(b).max()))
    # verify complex encoding is readable (trailing size-2 + __complex__)
    ds = o["Hamiltonian/paw/augmentation_function_isp0"]
    enc_ok = (ds.shape[-1] == 2 and ds.attrs.get("__complex__") == "1")
    checks.append(("complex-encoding ok", 0.0 if enc_ok else 1.0))

    print("writer round-trip (built from QE radial -> h5 -> reread vs QE ref):")
    worst = 0.0
    for name, e in checks:
        print("  %-24s %.3e" % (name, e))
        worst = max(worst, e)
    o.close(); f.close()
    print("\nEMIT", "PASS" if worst < 1e-4 else "CHECK", " (worst %.2e)" % worst)


if __name__ == "__main__":
    main()
