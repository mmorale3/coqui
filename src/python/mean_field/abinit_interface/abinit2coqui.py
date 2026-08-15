"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

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

abinit2coqui  --  ABINIT -> CoQuí mean-field converter

Reads an ABINIT WFK netCDF file (produced with `prtwf 1`, `iomode 3`) and writes a
single self-contained HDF5 file in the CoQuí "bdft" backend schema, so that CoQuí can
consume an ABINIT SCF as its mean field via  `mf_source = bdft`  --  NO QE, NO
pw2coqui, NO QE-native wfc files, and NO modification of ABINIT or CoQuí.

Supports norm-conserving (--psp8) and PAW (--pawxml) pseudopotentials, with or
without symmetry reduction of the k-mesh.  Which of the two is written is decided
automatically from the WFK: a k-list shorter than the file's own Monkhorst-Pack
grid means the run was symmetry-reduced.

    istwfk  *1         # store the full G-sphere at every k (no TR half-packing)
    prtwf   1
    iomode  3          # netCDF WFK/GSR/DEN

  nosym (every k explicit; the full bz_symm block is written out):

    kptopt  3
    nsym    1

  symmetry-reduced (IBZ + operations; CoQuí rebuilds the maps):

    kptopt     4       # spatial symmetries only -- NOT 1.  abinit2coqui writes
                       # use_trev=0, and an IBZ reduced with time reversal does
                       # not tile the grid under the stored operations (the
                       # star check in build_bz_minimal refuses it).  Pass
                       # --use-trev only if that is deliberately changed.
    symmorphi  0       # REQUIRED for PAW: CoQuí's PAW symmetry
                       # (hamilt::paw::build_atom_permutation_inverse) rejects
                       # non-symmorphic ft, which is also why the QE path runs
                       # force_symmorphic.  Diamond is non-symmorphic in the
                       # 2-atom cell, so Si/C drop from 48 to 24 operations.

Validated by converting the same ABINIT run both ways: the CoQuí total energy
difference falls monotonically with the ISDF threshold (2.8e-3 Ha at 1e-4 down to
7.8e-6 Ha at 1e-8 on Si 2x2x2), i.e. it is THC rank error, not a symmetry defect.

The bdft schema reproduced here (verified against src/mean_field/bdft/bdft_system.hpp,
bdft_readonly.hpp and src/mean_field/symmetry/bz_symmetry.hpp):

  /System   attrs: number_of_atoms, number_of_species, number_of_dimensions,
                   number_of_spins, number_of_spins_in_basis,
                   number_of_polarizations, number_of_polarizations_in_basis,
                   number_of_bands, number_of_elec, madelung_constant,
                   nuclear_energy, fermi_energy
            data : species (nspecies vlen str), atomic_id (natoms, 0-based species
                   index), atomic_positions (natoms,3 Bohr), lattice_vectors (3,3 Bohr,
                   rows a_n), reciprocal_vectors (3,3, rows b_n with a.b=2pi),
                   kpoint_weights (nkpts_ibz, star multiplicities normalized to 1)
  /System/BZ  nosym: the full bz_symm block (see write_bz) incl. qk_to_k2,
              qminus, qsymms, nq_per_s, ks_to_k, Qs, qs_to_q, Symmetries/s{i}/{R,ft}
              symmetric: the minimal block (see write_bz_minimal) -- kp_grid,
              kpoints_ibz (cartesian), Symmetries/s{i}/{R,ft} and the attrs
              no_q_sym/use_trev.  bdft_system rebuilds every map from these with
              the bz_symm constructor, the same code the QE reader goes through.
              NOTE "kpoints_ibz", not "kpoints": in the full block "kpoints" is
              the FULL BZ list, so the two layouts must not share a name.
  /Orbitals  attrs: number_of_spins(_in_basis), number_of_polarizations(_in_basis),
                    number_of_kpoints, number_of_kpoints_ibz, number_of_bands,
                    number_of_aux_bands, ecutrho
             data : fft_mesh (3), fft_mesh_aug (3), eigval (nspin,nk,nbnd Ha),
                    occ (nspin,nk,nbnd), wfc_ecut, wfc_fft_grid (3), wfc_ngm,
                    miller_wfc (ngm,3), psi_s{is}_k{ik} (nbnd,ngm complex)

Complex arrays follow the nda/TRIQS convention: stored as float64 with a trailing
axis of size 2 and a scalar string attribute  __complex__ = "1".

Usage:
    python abinit2coqui.py  path/to/o_DS1_WFK.nc  --outdir ./  --prefix abinit
"""

import os
import sys
import argparse
import numpy as np

try:
    import h5py
    _VLEN_STR = h5py.special_dtype(vlen=str)
except ImportError:            # pure-numpy map logic still importable/testable
    h5py = None
    _VLEN_STR = None

try:
    from netCDF4 import Dataset as _NCDataset
except ImportError:
    _NCDataset = None


def _require_h5py():
    if h5py is None:
        sys.exit("abinit2coqui: h5py is required for writing the output (pip install h5py).")


def _require_netcdf():
    if _NCDataset is None:
        sys.exit("abinit2coqui: netCDF4 is required for reading ABINIT WFK "
                 "(pip install netCDF4). ABINIT `iomode 3` files are classic NetCDF-3.")


def _w_attr_int(grp, name, val):
    grp.attrs.create(name, np.int32(val))


def _w_attr_dbl(grp, name, val):
    grp.attrs.create(name, np.float64(val))


def _w_real(grp, name, arr, dtype=None):
    a = np.ascontiguousarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    grp.create_dataset(name, data=a)


def _w_bool(grp, name, arr):
    # nda stores bool as an HDF5 enum over H5T_NATIVE_CHAR {FALSE=0,TRUE=1};
    # h5py writes a numpy bool array as exactly that enum.
    grp.create_dataset(name, data=np.ascontiguousarray(arr).astype(bool))


def _w_complex(grp, name, arr):
    """Write a complex array in nda layout: float64 (..., 2) + __complex__='1'."""
    a = np.ascontiguousarray(arr).astype(np.complex128)
    real_view = np.stack([a.real, a.imag], axis=-1).astype(np.float64)
    ds = grp.create_dataset(name, data=np.ascontiguousarray(real_view))
    ds.attrs.create("__complex__", "1", dtype=_VLEN_STR)


def _w_strings(grp, name, strings):
    data = np.array([str(s) for s in strings], dtype=object)
    grp.create_dataset(name, data=data, dtype=_VLEN_STR)


# --------------------------------------------------------------------------
# ABINIT WFK (ETSF-IO netCDF) reader
# --------------------------------------------------------------------------
def read_wfk(path):
    """Read an ABINIT WFK netCDF file into a plain dict of numpy arrays.

    ABINIT `iomode 3` writes classic NetCDF-3 (or NetCDF-4/HDF5 depending on the
    netcdf build); netCDF4 reads both.  Variable names/dimension-order verified
    against a real ABINIT 10.x WFK (ETSF-IO, arXiv:0805.0192).
    """
    _require_netcdf()
    ds = _NCDataset(path, "r")
    V = ds.variables

    def A(name):
        return np.array(V[name][:])

    def has(name):
        return name in V

    # --- structure (all lengths in Bohr; ABINIT native) ---
    rprimd = A("primitive_vectors")            # (3,3) rows a_n, Bohr
    xred = A("reduced_atom_positions")         # (natom,3) crystal
    typat = A("atom_species").astype(int)      # (natom,) 1-based species index
    znucl = A("atomic_numbers") if has("atomic_numbers") else A("znuclpsp")
    natom = xred.shape[0]
    nspecies = znucl.shape[0]

    # --- k-points ---
    kpts_crys = A("reduced_coordinates_of_kpoints")   # (nkpt,3) crystal
    wtk = A("kpoint_weights")                          # (nkpt,)
    nkpt = kpts_crys.shape[0]
    kp_grid = A("monkhorst_pack_folding").astype(int) \
        if has("monkhorst_pack_folding") else None

    # --- bands / spins ---
    eig = A("eigenvalues")            # (nsppol, nkpt, mband) Ha
    occ = A("occupations")            # (nsppol, nkpt, mband)
    nsppol = eig.shape[0]
    mband = eig.shape[2]

    # --- plane-wave basis (per-k Miller indices + coefficients) ---
    # reduced_coordinates_of_plane_waves: (nkpt, mpw, 3) per-k Miller indices.
    # number_of_coefficients: (nkpt,) true npw per k (rest of mpw is padding).
    kg = A("reduced_coordinates_of_plane_waves").astype(int)
    npwarr = A("number_of_coefficients").astype(int) \
        if has("number_of_coefficients") else None
    # coefficients_of_wavefunctions: (nsppol,nkpt,mband,nspinor,mpw,2)
    cg = A("coefficients_of_wavefunctions")
    nspinor = cg.shape[3]

    # --- symmetry (kept for reference / future symmetric path) ---
    symrel = A("reduced_symmetry_matrices").astype(int) \
        if has("reduced_symmetry_matrices") else np.eye(3, dtype=int)[None]
    tnons = A("reduced_symmetry_translations") \
        if has("reduced_symmetry_translations") else np.zeros((1, 3))

    # --- scalars ---
    if has("nelect"):
        nelec = float(np.array(V["nelect"][:]).ravel()[0])
    elif has("number_of_electrons"):
        nelec = float(np.array(V["number_of_electrons"][:]).ravel()[0])
    else:
        nelec = float(round(occ[0].sum() / nkpt))
    efermi = float(np.array(V["fermi_energy"][:]).ravel()[0]) if has("fermi_energy") else 0.0
    usepaw = int(np.array(V["usepaw"][:]).ravel()[0]) if has("usepaw") else 0
    ixc = int(np.array(V["ixc"][:]).ravel()[0]) if has("ixc") else None

    ds.close()

    # reciprocal vectors: rows b_n with a_i . b_j = 2*pi
    recv = 2.0 * np.pi * np.linalg.inv(rprimd).T   # (3,3) rows b_n

    return dict(rprimd=rprimd, recv=recv, xred=xred, typat=typat, znucl=znucl,
                natom=natom, nspecies=nspecies, kpts_crys=kpts_crys, wtk=wtk,
                nkpt=nkpt, kp_grid=kp_grid, eig=eig, occ=occ, nsppol=nsppol,
                mband=mband, nspinor=nspinor, kg=kg, npwarr=npwarr, cg=cg,
                symrel=symrel, tnons=tnons, nelec=nelec, efermi=efermi,
                usepaw=usepaw, ixc=ixc, ngfft=None)


def read_den(path):
    """Read the real-space density from an ABINIT DEN netCDF (prtden 1,
    iomode 3). Returns (n1,n2,n3) float64 (electrons/bohr^3; for PAW this is
    ABINIT's smooth rhor = tilde-n + nhat). nsppol=1 only."""
    _require_netcdf()
    ds = _NCDataset(path, "r")
    name = next((n for n in ("density", "rhor") if n in ds.variables), None)
    if name is None:
        sys.exit("abinit2coqui: no 'density' variable in %s (need an ABINIT "
                 "DEN netCDF, prtden 1 + iomode 3)." % path)
    rho = np.array(ds.variables[name][:])
    ds.close()
    # layout mirrors the POT vtrial: (nspin, n1, n2, n3[, ncomp])
    if rho.ndim == 5:
        rho = rho[..., 0]
    if rho.ndim != 4:
        sys.exit("abinit2coqui: unexpected DEN 'density' shape %s" % (rho.shape,))
    if rho.shape[0] != 1:
        raise NotImplementedError(
            "abinit2coqui: nsppol=%d DEN not supported (vxc export is "
            "spin-unpolarized only in v1)." % rho.shape[0])
    return rho[0].astype(float)


# --------------------------------------------------------------------------
# Shared truncated G-grid (union of per-k Miller lists)
# --------------------------------------------------------------------------
def build_shared_grid(w, symm_R=None):
    """Build the shared wfc grid: union of all per-k Miller triples.

    CoQuí's bdft backend uses ONE truncated_g_grid for all IBZ k-points; the
    coefficient array psi_s_k(nbnd,ngm) is ordered by this shared grid, with 0
    where a given k has no plane wave.  We take the union of ABINIT's per-k
    G-vectors (which ARE the G Miller indices of e^{i(k+G).r}).

    symm_R: when the run is symmetry-reduced, the rotations of the group.  The
    union then covers only the IBZ, and bdft_readonly::make_swfc_maps rotates
    Miller indices into it (w2g), so it MUST additionally be closed under the
    group or those lookups fall off the grid.  Closing needs a single pass: the
    union of the group-orbits of a set is already group-closed.  The G added
    this way carry zero coefficients at every stored k, which is exactly the
    sparse-union semantics the format already uses.

    Returns (miller_wfc (ngm,3) int, kg_index list per k, wfc_ecut, wfc_mesh).
    """
    nkpt = w["nkpt"]
    kg = w["kg"]
    npwarr = w["npwarr"]

    def kglist(ik):
        n = npwarr[ik] if npwarr is not None else kg.shape[-2]
        g = kg[ik] if kg.ndim == 3 else kg
        return g[:n]

    seen = {}
    order = []
    for ik in range(nkpt):
        for g in kglist(ik):
            key = (int(g[0]), int(g[1]), int(g[2]))
            if key not in seen:
                seen[key] = len(order)
                order.append(key)
    n_union = len(order)
    if symm_R is not None:
        # G' = G . Rinv, matching utils::transform_miller_indices' index order
        # (G'_j = sum_i G_i Rinv(i,j)) so the closure is the same set CoQuí
        # will look up.
        Rinvs = [np.linalg.inv(np.asarray(S, dtype=float)) for S in symm_R]
        base = list(order)
        for Ri in Rinvs:
            Gs = np.array(base, dtype=float) @ Ri
            Gr = np.rint(Gs)
            if np.abs(Gs - Gr).max() > 1e-6:
                raise ValueError(
                    "abinit2coqui: a symmetry rotation maps a Miller index to a "
                    "non-integer triple (max deviation %.2e); the symmetry "
                    "operations are not compatible with the reciprocal lattice."
                    % np.abs(Gs - Gr).max())
            for g in Gr.astype(np.int64):
                key = (int(g[0]), int(g[1]), int(g[2]))
                if key not in seen:
                    seen[key] = len(order)
                    order.append(key)

    miller = np.array(order, dtype=np.int32)          # (ngm,3)
    ngm = miller.shape[0]
    if symm_R is not None and ngm != n_union:
        print("#   shared G grid closed under symmetry: %d -> %d vectors"
              % (n_union, ngm))

    # per-k index map: position in `miller` of each of k's plane waves
    kg_idx = []
    for ik in range(nkpt):
        idx = np.array([seen[(int(g[0]), int(g[1]), int(g[2]))]
                        for g in kglist(ik)], dtype=np.int64)
        kg_idx.append(idx)

    # wfc_ecut must satisfy |G|^2/2 <= wfc_ecut for every G in the union
    Gcart = miller @ w["recv"]                          # (ngm,3) cartesian
    g2 = np.einsum("ij,ij->i", Gcart, Gcart)
    wfc_ecut = 0.5 * g2.max() * (1.0 + 1e-8)

    # FFT mesh large enough to hold all Miller indices without aliasing:
    #   mesh_d > 2*max|mill_d|  =>  take 2*max+2 (even)
    mmax = np.abs(miller).max(axis=0)
    mesh = (2 * mmax + 2).astype(np.int32)
    if w["ngfft"] is not None:
        mesh = np.maximum(mesh, w["ngfft"].astype(np.int32))
    return miller, kg_idx, float(wfc_ecut), mesh


def scatter_coeffs(w, kg_idx, ngm, is_, ik):
    """Return psi(nbnd, ngm) complex for (spin is_, kpoint ik) on the shared grid."""
    nband = w["mband"]
    cg = w["cg"]
    npwarr = w["npwarr"]
    n = npwarr[ik] if npwarr is not None else cg.shape[-2]
    # cg: (nsppol,nkpt,mband,nspinor,mpw,2) -> take nspinor=0 (NC, npol=1)
    block = cg[is_, ik, :, 0, :n, :]                   # (nband, n, 2)
    psik = np.zeros((nband, ngm), dtype=np.complex128)
    coeff = block[..., 0] + 1j * block[..., 1]          # (nband, n)
    psik[:, kg_idx[ik]] = coeff
    return psik


# --------------------------------------------------------------------------
# Symmetry orientation
# --------------------------------------------------------------------------
# CoQuí's symm_op.R is the rotation acting on reduced (crystal) REAL-space
# coordinates: utils::transform_miller_indices computes
#     G'_j = sum_i G_i Rinv(i,j),
# i.e. G' = R^{-T} G, which is the reciprocal-space action induced by
# r -> R r + ft.  ABINIT's symrel is that same real-space reduced rotation, so
# R = symrel and ft = tnons.
#
# What is NOT fixed by that derivation is the index order coming out of the
# netCDF file: ETSF declares reduced_symmetry_matrices with the Fortran dims
# reversed, so python[isym] may be symrel(:,:,isym) or its transpose depending
# on the writer.  The two differ only by a transpose, and the pairing of a
# rotation with ITS OWN tnons settles it: {R, ft} is a space-group operation
# only if it maps every atom onto an atom of the same species.  R^T paired with
# the same ft fails that unless R is symmetric, so the test below is decisive
# per operation rather than merely set-wise.
def _atoms_are_permuted(R, ft, xred, typat, tol=1e-5):
    """True if r -> R r + ft maps every atom onto a same-species atom (mod 1)."""
    for a in range(xred.shape[0]):
        y = R @ xred[a] + ft
        hit = False
        for b in range(xred.shape[0]):
            if typat[b] != typat[a]:
                continue
            d = y - xred[b]
            if np.all(np.abs(d - np.round(d)) < tol):
                hit = True
                break
        if not hit:
            return False
    return True


def orient_symmetries(symrel_nc, tnons, xred, typat):
    """Return (R_list, ft_list) with R in CoQuí's convention.

    symrel_nc is the array as read from the WFK, shape (nsym,3,3).  Tries the
    array as-is and transposed, and keeps whichever makes every {R,ft} a real
    space-group operation of the given structure.
    """
    symrel_nc = np.asarray(symrel_nc, dtype=float).reshape(-1, 3, 3)
    tnons = np.asarray(tnons, dtype=float).reshape(-1, 3)
    nsym = symrel_nc.shape[0]
    if tnons.shape[0] != nsym:
        raise ValueError("abinit2coqui: %d symmetry matrices but %d translations"
                         % (nsym, tnons.shape[0]))

    cands = {"as-read": symrel_nc,
             "transposed": np.transpose(symrel_nc, (0, 2, 1))}
    ok = {}
    for name, arr in cands.items():
        ok[name] = all(_atoms_are_permuted(arr[i], tnons[i], xred, typat)
                       for i in range(nsym))

    good = [n for n, v in ok.items() if v]
    if not good:
        raise ValueError(
            "abinit2coqui: neither orientation of reduced_symmetry_matrices maps "
            "the atoms onto themselves; the symmetry data or the structure in the "
            "WFK are inconsistent (nsym=%d)." % nsym)
    if len(good) == 2:
        # Only possible when every symrel is symmetric, in which case the two
        # orientations are the same data and the choice does not matter.
        diff = max(np.abs(cands["as-read"][i] - cands["transposed"][i]).max()
                   for i in range(nsym))
        if diff > 1e-12:
            raise ValueError(
                "abinit2coqui: both orientations of reduced_symmetry_matrices pass "
                "the atom-permutation test but differ; cannot pin the convention.")
    choice = good[0]
    return cands[choice].copy(), tnons.copy(), choice


# --------------------------------------------------------------------------
# Nosym Brillouin-zone maps (full bz_symm block for identity-only symmetry)
# --------------------------------------------------------------------------
def _mp_grid_dims(kpts_crys):
    """Infer Monkhorst-Pack dims n1,n2,n3 from a regular Gamma-centered set."""
    dims = []
    for d in range(3):
        vals = np.mod(np.round(kpts_crys[:, d] * 1e6) / 1e6, 1.0)
        uniq = np.unique(np.round(vals, 6))
        dims.append(len(uniq))
    return np.array(dims, dtype=np.int32)


def build_bz_nosym(kpts_crys, recv, kp_grid=None):
    """Construct all bz_symm arrays for the no-symmetry (identity-only) case.

    Mirrors src/mean_field/symmetry/bz_symmetry.hpp for nsym==1, use_trev=False,
    Gamma-centered grid (Qpts == kpts).  All k are their own IBZ image.
    """
    nk = kpts_crys.shape[0]
    if kp_grid is None:
        kp_grid = _mp_grid_dims(kpts_crys)
    kp_grid = np.asarray(kp_grid, dtype=np.int32)
    assert int(np.prod(kp_grid)) == nk, \
        "k-grid is not a complete regular grid (got %s for nk=%d)" % (kp_grid, nk)

    # Keep ABINIT's own k-point representatives (do NOT fold): the orbital and
    # projector G-lists are indexed relative to these exact k, so folding k here
    # (e.g. 0.5 -> -0.5) without shifting every G would make k inconsistent with
    # its plane-wave basis (breaks kinetic/projectors at k!=0).  Momentum matching
    # for the q-maps uses a modular-canonical key instead.
    kc = np.asarray(kpts_crys, dtype=float)

    def canon(v):                       # fractional part in [0,1), robust to noise
        return tuple(np.round(np.mod(np.round(v, 8), 1.0), 6))

    keymap = {canon(kc[i]): i for i in range(nk)}

    def find(v):
        return keymap[canon(v)]

    # --- k-point maps (identity symmetry) ---
    kp_symm = np.zeros(nk, dtype=np.int32)
    kp_trev = np.zeros(nk, dtype=bool)
    kp_trev_pair = -np.ones(nk, dtype=np.int32)
    kp_to_ibz = np.arange(nk, dtype=np.int32)

    # --- q maps: Qpts == kpts (Gamma centered) ---
    Qpts_crys = kc.copy()
    nq = nk
    qk_to_k2 = np.zeros((nq, nk), dtype=np.int32)   # kpts[b] = kpts[a] - Qpts[q]
    for q in range(nq):
        for a in range(nk):
            qk_to_k2[q, a] = find(kc[a] - Qpts_crys[q])
    qminus = np.zeros(nq, dtype=np.int32)           # Qpts[n] = -Qpts[q]
    for q in range(nq):
        qminus[q] = find(-Qpts_crys[q])

    qp_symm = np.zeros(nq, dtype=np.int32)
    qp_trev = np.zeros(nq, dtype=bool)
    qp_to_ibz = np.arange(nq, dtype=np.int32)

    # identity-only symmetry reconstruction maps
    qsymms = np.array([0], dtype=np.int32)
    nq_per_s = np.array([nq], dtype=np.int32)
    ks_to_k = np.arange(nk, dtype=np.int32).reshape(1, nk)      # (nsym,nk)
    qs_to_q = np.arange(nq, dtype=np.int32).reshape(1, nq)      # (nsym,nqibz)
    Qs = np.zeros((1, 2 * nq), dtype=np.int32)                  # (nsym,2*nqibz)
    Qs[0, :nq] = np.arange(nq, dtype=np.int32)

    # cartesian k / Q
    kpts_cart = kc @ recv
    Qpts_cart = Qpts_crys @ recv

    return dict(nkpts=nk, nkpts_ibz=nk, nkpts_trev_pairs=0, nqpts=nq, nqpts_ibz=nq,
                kp_grid=kp_grid, kpts=kpts_cart, kpts_crys=kc,
                kp_symm=kp_symm, kp_trev=kp_trev, kp_trev_pair=kp_trev_pair,
                kp_to_ibz=kp_to_ibz, Qpts=Qpts_cart, qk_to_k2=qk_to_k2,
                qminus=qminus, qp_symm=qp_symm, qp_trev=qp_trev,
                qp_to_ibz=qp_to_ibz, qsymms=qsymms, nq_per_s=nq_per_s,
                ks_to_k=ks_to_k, Qs=Qs, qs_to_q=qs_to_q)


def write_bz(sgrp, bz):
    g = sgrp.create_group("BZ")
    _w_attr_int(g, "number_of_kpoints", bz["nkpts"])
    _w_attr_int(g, "number_of_kpoints_ibz", bz["nkpts_ibz"])
    _w_attr_int(g, "number_of_trev_kpoint_pairs", bz["nkpts_trev_pairs"])
    _w_real(g, "kp_grid", bz["kp_grid"], np.int32)
    _w_real(g, "kpoints", bz["kpts"], np.float64)
    _w_real(g, "kpoints_crystal", bz["kpts_crys"], np.float64)
    _w_real(g, "kp_symm", bz["kp_symm"], np.int32)
    _w_real(g, "kp_to_ibz", bz["kp_to_ibz"], np.int32)
    _w_bool(g, "kp_trev", bz["kp_trev"])                    # nda bool enum
    _w_real(g, "kp_trev_pair", bz["kp_trev_pair"], np.int32)

    sym = g.create_group("Symmetries")
    _w_attr_int(sym, "number_of_symmetries", 1)
    s0 = sym.create_group("s0")
    _w_real(s0, "R", np.eye(3, dtype=np.float64))
    _w_real(s0, "ft", np.zeros(3, dtype=np.float64))

    _w_attr_int(g, "number_of_qpoints", bz["nqpts"])
    _w_attr_int(g, "number_of_qpoints_ibz", bz["nqpts_ibz"])
    _w_real(g, "qpoints", bz["Qpts"], np.float64)
    _w_real(g, "qk_to_k2", bz["qk_to_k2"], np.int32)
    _w_real(g, "qminus", bz["qminus"], np.int32)
    _w_real(g, "qp_symm", bz["qp_symm"], np.int32)
    _w_bool(g, "qp_trev", bz["qp_trev"])
    _w_real(g, "qp_to_ibz", bz["qp_to_ibz"], np.int32)
    _w_real(g, "qsymms", bz["qsymms"], np.int32)
    _w_real(g, "nq_per_s", bz["nq_per_s"], np.int32)
    _w_real(g, "ks_to_k", bz["ks_to_k"], np.int32)
    _w_real(g, "Qs", bz["Qs"], np.int32)
    _w_real(g, "qs_to_q", bz["qs_to_q"], np.int32)


# --------------------------------------------------------------------------
# Minimal Brillouin-zone block (symmetry-reduced runs)
# --------------------------------------------------------------------------
def build_bz_minimal(kpts_crys_ibz, recv, symrel, tnons, kp_grid,
                     no_q_sym=False, use_trev=False):
    """Assemble the minimal /System/BZ payload for a symmetry-reduced run.

    Writes only the MP grid, the IBZ k-points and the symmetry operations;
    bz_symm's constructor rebuilds every k/q map on the CoQuí side (the same
    code path the QE reader uses), so the map conventions are not duplicated
    here.

    Guard: the star of the IBZ under the given operations must reproduce the
    full MP grid exactly.  That catches a wrong symmetry orientation, an IBZ
    that was reduced with time reversal while use_trev is off, and a k-set that
    is not a complete regular grid -- all of which would otherwise surface much
    later as a confusing failure inside the constructor.
    """
    kc = np.asarray(kpts_crys_ibz, dtype=float)
    kp_grid = np.asarray(kp_grid, dtype=np.int64).reshape(3)
    nk_full = int(np.prod(kp_grid))
    R = np.asarray(symrel, dtype=float).reshape(-1, 3, 3)
    ft = np.asarray(tnons, dtype=float).reshape(-1, 3)
    nsym = R.shape[0]

    def canon(v):
        return tuple(np.round(np.mod(np.round(v, 8), 1.0), 6))

    # reciprocal reduced coords transform as k -> R^{-T} k
    star = set()
    for S in R:
        M = np.linalg.inv(S).T
        for k in kc:
            star.add(canon(M @ k))
    if use_trev:
        for S in R:
            M = np.linalg.inv(S).T
            for k in kc:
                star.add(canon(-(M @ k)))

    if len(star) != nk_full:
        raise ValueError(
            "abinit2coqui: the %d IBZ k-points under %d symmetry operation(s) "
            "generate %d distinct points, but the Monkhorst-Pack grid %s has %d. "
            "The IBZ and the symmetry list are inconsistent (if ABINIT reduced "
            "with time reversal, e.g. kptopt 1, either rerun with kptopt 4 or "
            "convert with use_trev enabled)."
            % (kc.shape[0], nsym, len(star), tuple(int(x) for x in kp_grid), nk_full))

    # and the generated points must BE the MP grid, not merely as many points
    grid = set()
    for i in range(kp_grid[0]):
        for j in range(kp_grid[1]):
            for k in range(kp_grid[2]):
                grid.add(canon(np.array([i / kp_grid[0], j / kp_grid[1],
                                         k / kp_grid[2]])))
    if star != grid:
        raise ValueError(
            "abinit2coqui: the symmetry star of the IBZ has the right count but "
            "does not coincide with the Gamma-centered Monkhorst-Pack grid %s; "
            "shifted grids (shiftk /= 0) are not supported."
            % (tuple(int(x) for x in kp_grid),))

    return dict(kp_grid=kp_grid.astype(np.float64),
                kpts=kc @ recv, kpts_crys=kc,
                R=R, ft=ft, nsym=nsym,
                no_q_sym=bool(no_q_sym), use_trev=bool(use_trev))


def write_bz_minimal(sgrp, bz):
    g = sgrp.create_group("BZ")
    _w_real(g, "kp_grid", bz["kp_grid"], np.float64)
    # "kpoints_ibz", NOT "kpoints": in the full BZ layout "kpoints" holds the
    # FULL BZ list while bz_symm's constructor takes the IBZ list, so the two
    # layouts must not share a name.  Cartesian, as the constructor expects.
    _w_real(g, "kpoints_ibz", bz["kpts"], np.float64)
    _w_real(g, "kpoints_ibz_crystal", bz["kpts_crys"], np.float64)  # informational
    _w_attr_int(g, "no_q_sym", 1 if bz["no_q_sym"] else 0)
    _w_attr_int(g, "use_trev", 1 if bz["use_trev"] else 0)

    sym = g.create_group("Symmetries")
    _w_attr_int(sym, "number_of_symmetries", bz["nsym"])
    for i in range(bz["nsym"]):
        si = sym.create_group("s%d" % i)
        _w_real(si, "R", bz["R"][i], np.float64)
        _w_real(si, "ft", bz["ft"][i], np.float64)


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------
def convert(wfk_path, outdir="./", prefix="abinit", nbnd_out=None, verbose=True,
            pot=None, psp8=None, pawxml=None, den=None, xc=None, corewf=None,
            no_q_sym=False, use_trev=False):
    _require_h5py()
    w = read_wfk(wfk_path)
    if w["nspinor"] != 1:
        raise NotImplementedError("nspinor>1 (spin-orbit) not supported in v1.")
    if w.get("usepaw", 0) != 0 and pawxml is None:
        raise NotImplementedError(
            "usepaw=1 requires --pawxml (one PAW-XML dataset per species in znucl "
            "order) plus --pot, so the PAW /Hamiltonian augmentation can be emitted.")

    # --- pseudopotential parses up front: needed for the /System lattice
    # sums (ionic charges) as well as the /Hamiltonian blocks ---
    paw_parses = psp_parses = None
    zion_sp = None                       # per-species ionic (valence) charge
    if pawxml is not None:
        import abinit_pawxml as axml
        xml_paths = [pawxml] if isinstance(pawxml, str) else list(pawxml)
        paw_parses = [axml.parse_pawxml(p) for p in xml_paths]
        if corewf is not None:
            cw_paths = [corewf] if isinstance(corewf, str) else list(corewf)
            if len(cw_paths) != len(xml_paths):
                sys.exit("abinit2coqui: --corewf needs one file per --pawxml "
                         "(got %d vs %d)." % (len(cw_paths), len(xml_paths)))
            for p, cw in zip(paw_parses, cw_paths):
                axml.attach_corewf(p, cw)
        zion_sp = []
        for p in paw_parses:
            ncore_el = np.trapezoid(
                np.asarray(p["ae_core"], float) / np.sqrt(4.0 * np.pi)
                * 4.0 * np.pi * p["r"] ** 2, p["r"]) if p.get("ae_core") is not None else 0.0
            zval = round(float(p["znucl"]) - ncore_el)
            f_sum = sum(s.get("f", 0.0) for s in p["states"])
            if f_sum > 0 and abs(zval - f_sum) > 1e-6:
                print("# WARNING abinit2coqui: species Z=%g zval from core "
                      "integral (%g) != sum of state occupations (%g); using %g"
                      % (p["znucl"], zval, f_sum, zval))
            zion_sp.append(float(zval))
    elif psp8 is not None:
        from abinit_psp8 import parse_psp8
        psp_paths = [psp8] if isinstance(psp8, str) else list(psp8)
        psp_parses = [parse_psp8(p) for p in psp_paths]
        zion_sp = [float(p["zion"]) for p in psp_parses]

    # --- density (for real vxc) + functional ---
    rho_den = read_den(den) if den is not None else None
    xc_name = None
    if rho_den is not None:
        if xc is not None:
            xc_name = xc
        elif w.get("ixc") is not None:
            from xc_functionals import functional_from_ixc
            xc_name = functional_from_ixc(w["ixc"])
        else:
            sys.exit("abinit2coqui: --den given but no functional: the WFK "
                     "carries no ixc; pass --xc {pbe,lda_pw}.")

    nspin = w["nsppol"]
    npol = 1
    nk = w["nkpt"]
    nbnd = w["mband"] if nbnd_out is None else min(nbnd_out, w["mband"])

    # species table: unique in first-appearance order (ABINIT typat is 1-based)
    znucl = w["znucl"]
    typat = w["typat"]                             # 1-based species index per atom
    Z_TO_SYM = _z_to_symbol()
    species = [Z_TO_SYM.get(int(round(znucl[s])), "X%d" % int(round(znucl[s])))
               for s in range(w["nspecies"])]
    atomic_id = (typat - 1).astype(np.int32)       # 0-based species index
    at_pos_bohr = w["xred"] @ w["rprimd"]          # crystal -> cartesian (Bohr)

    # BZ + shared grid.  A WFK whose k-list is shorter than its own MP grid came
    # from a symmetry-reduced run (kptopt /= 3): emit the minimal BZ block and let
    # CoQuí rebuild the maps.  Otherwise every k is explicit and the nosym block
    # is written as before.
    kp_grid_in = w.get("kp_grid")
    have_grid = (kp_grid_in is not None and int(np.prod(kp_grid_in)) > 0)
    nk_full = int(np.prod(kp_grid_in)) if have_grid else w["nkpt"]
    symmetric = have_grid and (w["nkpt"] < nk_full)

    if symmetric:
        R_sym, ft_sym, orient = orient_symmetries(
            w["symrel"], w["tnons"], w["xred"], w["typat"])
        bz = build_bz_minimal(w["kpts_crys"], w["recv"], R_sym, ft_sym, kp_grid_in,
                              no_q_sym=no_q_sym, use_trev=use_trev)
        if verbose:
            print("#   symmetry-reduced: %d IBZ k-points -> %d in the full grid, "
                  "%d operations (%s), no_q_sym=%d use_trev=%d"
                  % (w["nkpt"], nk_full, bz["nsym"], orient,
                     int(no_q_sym), int(use_trev)))
    else:
        bz = build_bz_nosym(w["kpts_crys"], w["recv"], kp_grid=kp_grid_in)
    miller, kg_idx, wfc_ecut, wfc_mesh = build_shared_grid(
        w, symm_R=(bz["R"] if symmetric else None))
    ngm = miller.shape[0]

    # eigenvalues / occupations (Ha). CoQuí occ in [0,1] per spin channel;
    # ABINIT NC (nsppol=1) occ are in [0,2] -> divide by 2.
    eig = w["eig"][:, :, :nbnd].astype(np.float64)          # (nspin,nk,nbnd)
    occ = w["occ"][:, :, :nbnd].astype(np.float64)
    if nspin == 1:
        occ = occ / 2.0

    fft_mesh = wfc_mesh.astype(np.int32)                    # smooth grid (holds wfc sphere)
    # dense/augmentation grid: with a POT file, use ABINIT's own FFT grid (ngfft) so
    # /Orbitals/fft_mesh_aug matches the /Hamiltonian dense grid (miller_g) that
    # read_vnl_h5 maps onto fft_grid_dim_aug.  Without POT, fall back to the wfc mesh.
    fft_mesh_aug = fft_mesh.copy()
    if pot is not None:
        _ds = _NCDataset(pot, "r")
        _sh = _ds.variables["vtrial"].shape                 # (nspin, n1, n2, n3, ncomp)
        _ds.close()
        fft_mesh_aug = np.array(_sh[1:4], dtype=np.int32)
    # /Orbitals@ecutrho (HARTREE, schema-3 pin): the exact inscribed-sphere
    # cutoff of the dense/aug grid, i.e. the sphere dense_sphere() keeps for
    # miller_g when --pot is given. Replaces the 2x/4x wfc_ecut heuristic.
    from abinit_hamiltonian import dense_sphere_ecut
    ecutrho = float(dense_sphere_ecut(fft_mesh_aug, w["rprimd"]))

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, prefix + ".h5")
    if verbose:
        print("# abinit2coqui: %s -> %s" % (wfk_path, filename))
        print("#   nspin=%d nkpt=%d nbnd=%d ngm(shared)=%d wfc_ecut=%.3f Ha mesh=%s"
              % (nspin, nk, nbnd, ngm, wfc_ecut, tuple(int(x) for x in fft_mesh)))

    with h5py.File(filename, "w") as f:
        root = f["/"]
        # ---- /System ----
        S = root.create_group("System")
        _w_attr_int(S, "number_of_atoms", w["natom"])
        _w_attr_int(S, "number_of_species", w["nspecies"])
        _w_attr_int(S, "number_of_dimensions", 3)
        _w_attr_int(S, "number_of_spins", nspin)
        _w_attr_int(S, "number_of_spins_in_basis", nspin)
        _w_attr_int(S, "number_of_polarizations", npol)
        _w_attr_int(S, "number_of_polarizations_in_basis", npol)
        _w_attr_int(S, "number_of_bands", nbnd)
        _w_attr_dbl(S, "number_of_elec", w["nelec"])
        # madelung constant (Ha): same value bdft_system would compute from
        # cell + k-mesh when the stored one is 0 (its compute-when-0 guard
        # stays as a fallback); -2 * utils::madelung port.
        from lattice_sums import madelung_bdft, ewald_energy
        madelung = -2.0 * madelung_bdft(w["rprimd"], w["recv"],
                                        np.asarray(bz["kp_grid"], dtype=np.int32),
                                        fft_mesh, 1e-10)
        _w_attr_dbl(S, "madelung_constant", madelung)
        # nuclear (Ewald) energy in Ha, ions = valence point charges in a
        # neutralizing background (QE nuclear_energy convention). Needs the
        # ionic charges, i.e. a --psp8/--pawxml parse; else 0 + warning.
        if zion_sp is not None:
            zion_at = np.array([zion_sp[t - 1] for t in typat], float)
            enuc = ewald_energy(w["rprimd"], at_pos_bohr, zion_at)
        else:
            enuc = 0.0
            print("# WARNING abinit2coqui: no --psp8/--pawxml -> ionic charges "
                  "unknown; writing nuclear_energy = 0.")
        _w_attr_dbl(S, "nuclear_energy", enuc)
        # frozen core-core exact-exchange total (Ha, additive constant):
        # sum over atoms of the species' PAW-XML core-core energy.
        if paw_parses is not None:
            exx_cc = sum(paw_parses[t - 1].get("exx_core_core", 0.0) or 0.0
                         for t in typat)
            if exx_cc != 0.0:
                _w_attr_dbl(S, "exx_core_core", exx_cc)
        _w_attr_dbl(S, "fermi_energy", w["efermi"])
        _w_strings(S, "species", species)
        _w_real(S, "atomic_id", atomic_id, np.int32)
        _w_real(S, "atomic_positions", at_pos_bohr, np.float64)
        _w_real(S, "lattice_vectors", w["rprimd"], np.float64)
        _w_real(S, "reciprocal_vectors", w["recv"], np.float64)
        # kpoint_weights is indexed by the IBZ and carries the star
        # multiplicities, normalized to 1 (the lih_kp222_nbnd16_sym checkpoint
        # stores 1/8, 4/8, 3/8 on its three IBZ points).  ABINIT's wtk is
        # already exactly that for any kptopt, and for a nosym grid it
        # degenerates to the uniform 1/nk.  Writing uniform FULL-BZ weights
        # instead makes the weights sum to nkpts_ibz/nkpts, which starves the
        # electron count and sends the chemical-potential search off to
        # infinity.
        _w_real(S, "kpoint_weights", w["wtk"] / w["wtk"].sum(), np.float64)
        if symmetric:
            write_bz_minimal(S, bz)
        else:
            write_bz(S, bz)

        # ---- /Orbitals ----
        O = root.create_group("Orbitals")
        _w_attr_int(O, "number_of_spins", nspin)
        _w_attr_int(O, "number_of_spins_in_basis", nspin)
        _w_attr_int(O, "number_of_polarizations", npol)
        _w_attr_int(O, "number_of_polarizations_in_basis", npol)
        # number_of_kpoints counts the FULL BZ, number_of_kpoints_ibz the
        # irreducible set the orbitals are actually stored on.  eigval/occ stay
        # IBZ-sized: bdft_system reads them with extent >= nkpts_ibz and fills
        # the remaining k from kp_to_ibz itself, so the full-BZ ordering (which
        # only CoQuí knows, since it rebuilds the maps) never has to be
        # reproduced here.
        _w_attr_int(O, "number_of_kpoints", nk_full if symmetric else nk)
        _w_attr_int(O, "number_of_kpoints_ibz", nk)
        _w_attr_int(O, "number_of_bands", nbnd)
        _w_attr_int(O, "number_of_aux_bands", 0)
        _w_attr_dbl(O, "ecutrho", ecutrho)
        _w_real(O, "fft_mesh", fft_mesh, np.int32)
        _w_real(O, "fft_mesh_aug", fft_mesh_aug, np.int32)
        _w_real(O, "eigval", eig, np.float64)
        _w_real(O, "occ", occ, np.float64)
        # shared wfc grid
        _w_real(O, "miller_wfc", miller, np.int32)
        O.create_dataset("wfc_ecut", data=np.float64(wfc_ecut))
        _w_real(O, "wfc_fft_grid", wfc_mesh, np.int32)
        O.create_dataset("wfc_ngm", data=np.int32(ngm))
        # coefficients, per (spin,k) on the shared grid
        for ik in range(nk):
            for is_ in range(nspin):
                psik = scatter_coeffs(w, kg_idx, ngm, is_, ik)[:nbnd]
                _w_complex(O, "psi_s%d_k%d" % (is_, ik), psik)

        # /Hamiltonian: --pot + --pawxml -> augmented paw block; --pot + --psp8 ->
        # norm-conserving block; otherwise an empty group.
        ham_w = dict(kg=w["kg"], npw=w["npwarr"], kpts_crys=w["kpts_crys"],
                     recv=w["recv"], xred=w["xred"], typat=w["typat"],
                     rprimd=w["rprimd"], nkpt=nk, nsppol=nspin, nspinor=1)
        if pot is not None and paw_parses is not None:
            import abinit_paw_hamiltonian as aph
            vtrial = np.array(_NCDataset(pot, "r").variables["vtrial"][:])
            info = aph.write_hamiltonian_paw(root, ham_w, vtrial, paw_parses,
                                             verbose=verbose, rho_den=rho_den,
                                             xc_name=xc_name)
        elif pot is not None and psp_parses is not None:
            import abinit_hamiltonian as ah
            vtrial = np.array(_NCDataset(pot, "r").variables["vtrial"][:])
            info = ah.write_hamiltonian_nc(root, ham_w, vtrial, psp_parses,
                                           rho_den=rho_den, xc_name=xc_name)
            if verbose:
                print("#   /Hamiltonian(ncpp): ngm=%d nkb=%d nhm=%d ngfft=%s"
                      % (info["ngm"], info["nkb"], info["nhm"], info["ngfft"]))
        else:
            root.create_group("Hamiltonian")

    if verbose:
        print("# done. Read in CoQuí with: mf_source = bdft, outdir/prefix pointing at %s"
              % filename)
    return filename


def _z_to_symbol():
    syms = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
            "Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
            "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
            "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
    return {i: s for i, s in enumerate(syms)}


def main():
    ap = argparse.ArgumentParser(description="Convert an ABINIT WFK to a CoQuí bdft h5.")
    ap.add_argument("wfk", help="ABINIT WFK netCDF file (prtwf 1, iomode 3, istwfk *1)")
    ap.add_argument("--outdir", default="./")
    ap.add_argument("--prefix", default="abinit")
    ap.add_argument("--nbnd", type=int, default=None, help="truncate to this many bands")
    ap.add_argument("--pot", default=None, help="ABINIT POT netCDF (prtpot 1) for /Hamiltonian")
    ap.add_argument("--psp8", nargs="+", default=None,
                    help="psp8 pseudopotential(s), one per species in znucl order (NC)")
    ap.add_argument("--pawxml", nargs="+", default=None,
                    help="PAW-XML dataset(s), one per species in znucl order (PAW)")
    ap.add_argument("--den", default=None,
                    help="ABINIT DEN netCDF (prtden 1) -> real vxc/vxc_with_nlcc")
    ap.add_argument("--xc", default=None, choices=["pbe", "lda_pw"],
                    help="XC functional for --den (default: from the WFK ixc)")
    ap.add_argument("--corewf", nargs="+", default=None,
                    help="atompaw .corewf.xml companion(s), one per --pawxml "
                         "-> Species/Core AE core orbitals (native ex_cvij)")
    ap.add_argument("--no-q-sym", action="store_true",
                    help="symmetry-reduced input only: do not symmetry-reduce the "
                         "q-points (k-points are still reduced)")
    ap.add_argument("--use-trev", action="store_true",
                    help="symmetry-reduced input only: the IBZ was reduced with "
                         "time reversal as well (ABINIT kptopt 1/2)")
    args = ap.parse_args()
    convert(args.wfk, args.outdir, args.prefix, args.nbnd, pot=args.pot,
            psp8=args.psp8, pawxml=args.pawxml, den=args.den, xc=args.xc,
            corewf=args.corewf, no_q_sym=args.no_q_sym, use_trev=args.use_trev)


if __name__ == "__main__":
    main()
