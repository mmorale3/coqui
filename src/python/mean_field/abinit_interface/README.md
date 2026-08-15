# abinit2coqui — ABINIT → CoQuí mean-field converter

Drive CoQuí (RPA / GW / correlated methods) from an **ABINIT** SCF instead of Quantum
ESPRESSO, by converting an ABINIT `WFK` netCDF file into a single self-contained HDF5
file in CoQuí's **`bdft`** backend schema. No QE, no `pw2coqui.x`, no QE-native `wfc`
files, and no modification of ABINIT or CoQuí.

ABINIT's PAW-XML data sets carry the one-center exact-exchange kernel directly, which
makes this path convenient for augmented (PAW/USPP) hybrid and RPA work. The converter
is equally usable on its own for **norm-conserving** ABINIT users who want CoQuí's
many-body methods.

## Scope

The converter writes:

- `/System`, `/System/BZ` and `/Orbitals` (KS orbitals on a shared truncated G-grid,
  eigenvalues, occupations).
- `/Hamiltonian` for **norm-conserving** (`--psp8`): β(k+G) projectors, `V_loc`,
  `scf_local_potential`, `dion`.
- `/Hamiltonian` for **PAW** (`--pawxml`): the norm-conserving block plus the
  augmentation — compensation charges `Q^IJ(G)`, `qq_nt`, `ijtoh`, the per-species
  radial block, and the one-center exact-exchange kernel `Onecenter/deltaC` recomputed
  from the AE/PS partial waves.
- Frozen ionic Hartree potentials `Species/nt*/paw/{ae_vloc, vloc_ps}` (XC-free by
  design), `beta` (u = r·proj), and `Species/nt*/Core/` (AE core orbitals) when the
  data set provides them, together with the per-species `exx_core_core` attribute.
- `/System@nuclear_energy` (Ewald) and `/System@madelung_constant`.
- `vxc` / `vxc_with_nlcc` when an ABINIT DEN file is supplied (`--den`); the exchange
  correlation functional is taken from the WFK `ixc` or forced with `--xc {pbe,lda_pw}`.

Both full and symmetry-reduced k-meshes are supported; see the run settings below.

Energy-valued datasets are written in **Hartree** (ABINIT native) and tagged with
`/Hamiltonian@schema_version`. CoQuí readers apply a Ry→Ha conversion only to files
written before that tag was introduced.

## Required ABINIT run settings

Write the WFK with the full G-sphere at every k:

```
istwfk  *1       # store the full G-sphere (no time-reversal half-packing)
prtwf   1
prtpot  1        # KS potential (needed for the /Hamiltonian block)
iomode  3        # netCDF WFK/GSR/DEN/POT
prtden  1        # only if you want vxc / vxc_with_nlcc
```

For a full (unreduced) mesh, each k being its own IBZ point:

```
kptopt  3        # all k in the full Monkhorst-Pack grid, explicit
nsym    1        # no symmetry reduction
```

For a symmetry-reduced mesh:

```
kptopt     4     # spatial symmetries only
symmorphi  0     # required for PAW: the PAW symmetry path assumes symmorphic
                 # operations, so fractional translations must be disabled
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
# + vxc / vxc_with_nlcc
python abinit2coqui.py  <run>o_WFK.nc  --pot <run>o_POT.nc  --den <run>o_DEN.nc  --pawxml Si.xml ...
```

Writes `./<prefix>.h5`. Point CoQuí at it with `mf_source = bdft`, and `outdir` /
`prefix` matching the file (`<outdir>/<prefix>.h5`).

Requires `numpy` and `h5py`. ABINIT `iomode 3` files are NetCDF-4 (i.e. HDF5) and are
read directly by `h5py`, so neither netCDF4 nor abipy is needed.

## Conventions

- Units: lattice and positions in **Bohr**, eigenvalues in **Hartree**.
- `reciprocal_vectors` rows are bₙ with aᵢ·bⱼ = 2π; `atomic_id` is the **0-based
  species index**; `species` is `nspecies` variable-length strings.
- Complex arrays: float64 with a trailing size-2 axis plus a scalar string attribute
  `__complex__ = "1"` (nda/TRIQS layout).
- Occupations: CoQuí uses [0,1] per spin channel, so ABINIT `nsppol=1` occupations
  in [0,2] are halved.
- The shared wavefunction grid is the **union** of ABINIT's per-k
  `reduced_coordinates_of_plane_waves`; `wfc_ecut` is set so that |G|²/2 ≤ `wfc_ecut`
  for every G in the union.

These match `bdft_system.hpp` (`/System` and `/Orbitals`), `bdft_readonly.hpp`
(`/Orbitals/psi_s{is}_k{ik}` on the shared `miller_wfc` grid) and `bz_symmetry.hpp`
(the `/System/BZ` block).

## Validation

Standalone scripts check the emitted h5 against ABINIT or QE references:

- `validate_h0.py`, `validate_h0_from_h5.py` — rebuild `H_KS` from the emitted h5 and
  reproduce the ABINIT eigenvalues (norm-conserving path).
- `validate_h0_paw.py` — PAW generalized eigenproblem `H c = ε S c`; reproduces ABINIT
  eigenvalues to ~1e-5 Ha.
- `validate_qvan.py`, `validate_deltaC.py`, `validate_ex_cvij.py`,
  `validate_paw_frontend.py`, `validate_paw_emit.py`, `validate_paw_overlap.py` — the
  augmentation kernels (`Q^IJ(G)`, `deltaC`, the core-valence exchange kernel, channel
  maps and overlap) against a QE-PAW reference to near-machine precision.
