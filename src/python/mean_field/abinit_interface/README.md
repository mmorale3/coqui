# abinit2coqui — ABINIT → CoQuí mean-field converter

Drive CoQuí (RPA / GW / correlated methods) from an **ABINIT** SCF instead of Quantum
ESPRESSO, by converting an ABINIT `WFK` netCDF file into a single self-contained HDF5
file in CoQuí's **`bdft`** backend schema. No QE, no `pw2coqui.x`, no QE-native `wfc`
files, and no modification of ABINIT or CoQuí.

Motivation: QE's augmented (PAW/USPP) exact-exchange is the weak link for hybrid/RPA
equations of state; ABINIT is a mature PAW code whose PAW-XML datasets carry the
one-center exact-exchange kernel directly. This converter is the first step toward that
path — and it is already useful on its own for **norm-conserving** ABINIT users who want
CoQuí's many-body methods.

## Scope

**No symmetry reduction** of the k-mesh (each k is its own IBZ point).

- `/System`, `/System/BZ` (full nosym maps) and `/Orbitals` (KS orbitals on a shared
  truncated G-grid, eigenvalues, occupations).
- `/Hamiltonian` for **norm-conserving** (`--psp8`): β(k+G) projectors, `V_loc`,
  `scf_local_potential`, `dion`.
- `/Hamiltonian` for **PAW** (`--pawxml`): the norm-conserving block plus the
  augmentation — compensation charges `Q^IJ(G)`, `qq_nt`, `ijtoh`, the per-species
  radial block, and the one-center exact-exchange kernel `Onecenter/deltaC` recomputed
  from the AE/PS partial waves.

Plan-B2 additions (2026-07-24):

- `proj_per_atom` is per-**species** (QE `nh(1:nsp)` convention) in both writers.
- `Species/nt*/paw/{ae_vloc, vloc_ps}` — frozen ionic **Hartree** potentials
  (XC-free by design, Ry/UPF on-disk convention), the same pair `dion` is
  assembled from; plus `beta` (u = r·proj) and `paw/oc`.
- `Species/nt*/Core/` (AE core orbitals, pw2coqui GIPAW schema) when the XML
  provides them; `exx_core_core` per-species attribute and a summed
  `/System@exx_core_core` total-energy attribute.
- Real `/System@nuclear_energy` (Ewald, Ha; validated vs QE to 2e-8 Ha) and
  `/System@madelung_constant` (exact port of `utils::madelung`).
- Real `vxc`/`vxc_with_nlcc` from an ABINIT DEN (`--den <run>o_DEN.nc`,
  functional from the WFK `ixc` or `--xc {pbe,lda_pw}`); the PBE evaluator
  reproduces QE `v_xc` to ~3e-11 Ry (validate_b2.py). NOTE: the DEN-side
  wiring (netCDF layout, PAW smooth-ρ semantics, psp8 NLCC block) has not yet
  been exercised against real ABINIT files on this machine — recheck on the
  first cluster campaign.
- The analytic compensation shape is cross-checked against a tabulated
  `<shape_function>` when the XML carries one — hard error on mismatch.

## Required ABINIT run settings

Produce the WFK with the full G-sphere at every k and no symmetry reduction:

```
nsym    1        # no symmetry reduction (each k is its own IBZ point)
kptopt  3        # all k in the full Monkhorst-Pack grid, explicit
istwfk  *1       # store the full G-sphere (no time-reversal half-packing)
prtwf   1
prtpot  1        # KS potential (needed for the /Hamiltonian block)
iomode  3        # netCDF WFK/GSR/DEN/POT
```

Everything else (ecut, ngkpt, nband, pseudos) as usual.

## Usage

Orbitals only (`/System` + `/Orbitals`):

```
python abinit2coqui.py  <run>o_WFK.nc  --outdir ./  --prefix abinit  [--nbnd N]
```

With the `/Hamiltonian` block, pass the POT file and the pseudopotential(s):

```
# norm-conserving
python abinit2coqui.py  <run>o_WFK.nc  --pot <run>o_POT.nc  --psp8   Si.psp8   ...
# PAW
python abinit2coqui.py  <run>o_WFK.nc  --pot <run>o_POT.nc  --pawxml Si.xml    ...
# + real vxc/vxc_with_nlcc (add prtden 1 to the ABINIT run)
python abinit2coqui.py  <run>o_WFK.nc  --pot <run>o_POT.nc  --den <run>o_DEN.nc  --pawxml Si.xml ...
```

Writes `./<prefix>.h5`. Point CoQuí at it with `mf_source = bdft`, `outdir`/`prefix`
matching the file (`<outdir>/<prefix>.h5`).

Requires: `numpy`, `h5py`. (ABINIT `iomode 3` files are NetCDF-4 = HDF5, read directly
by `h5py` — no netCDF4/abipy dependency.)

## Conventions reproduced (verified against CoQuí source)

- `bdft_system.hpp` (`/System` + `/Orbitals` attrs/datasets),
  `bdft_readonly.hpp` (`/Orbitals/psi_s{is}_k{ik}` on the shared `miller_wfc` grid),
  `bz_symmetry.hpp` (full `/System/BZ` block).
- Units: lattice/positions in **Bohr**, eigenvalues in **Hartree** (ABINIT native).
- `reciprocal_vectors` rows are bₙ with aᵢ·bⱼ = 2π; `atomic_id` is the **0-based species
  index**; `species` is `nspecies` variable-length strings.
- Complex arrays: float64 with a trailing size-2 axis + scalar string attribute
  `__complex__ = "1"` (nda/TRIQS layout).
- Occupations: CoQuí uses [0,1] per spin channel; ABINIT NC (`nsppol=1`) occ in [0,2] are
  halved.
- Shared wfc grid = **union** of ABINIT's per-k `reduced_coordinates_of_plane_waves`;
  `wfc_ecut` set so |G|²/2 ≤ wfc_ecut for every G in the union.

## Validation

Standalone scripts (`validate_*.py`) check the emitted h5 against ABINIT / QE references:

- `validate_h0.py`, `validate_h0_from_h5.py` — rebuild `H_KS` from the emitted h5 and
  reproduce the ABINIT eigenvalues (norm-conserving path).
- `validate_h0_paw.py` — PAW generalized eigenproblem `H c = ε S c`; reproduces ABINIT
  eigenvalues to ~1e-5 Ha.
- `validate_qvan.py`, `validate_deltaC.py`, `validate_paw_frontend.py`,
  `validate_paw_emit.py`, `validate_paw_overlap.py` — the augmentation kernels
  (`Q^IJ(G)`, `deltaC`, channel maps, overlap) against a QE-PAW reference to
  (near-)machine precision.
- `eos_rpa_fit.py` — assembles an RPA@PBE total energy from an ABINIT EOS series and
  fits the lattice constant.
