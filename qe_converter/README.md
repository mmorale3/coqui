Quantum ESPRESSO Converter for CoQuí
------------------------------------
**Last Updated:** Sep. 1, 2025

This directory provides the Fortran source code `pw2coqui.f90` required to 
interface **Quantum ESPRESSO (QE)** with **CoQuí**. The converter extracts 
the necessary data from QE calculations and makes it available for use in 
CoQuí workflows.

## 📦 Installation

The converter is distributed with the CoQuí source code, but it is **not 
compiled as part of CoQuí itself**. Instead, you need to integrate it into 
your local QE source tree and compile it together with QE.

### Step 1: Locate the converter source files
Copy `pw2coqui.f90` into the **`PP/src/` folder** (the *PostProc* package in the
QE suite) of your QE source tree:  
```bash
cp -r pw2coqui.f90 /path/to/qe-source/PP/src/
```

### Step 2: Modify QE’s CMakeLists.txt
Update the QE build system so that the converter is compiled. Inside the
`PostProc` package folder within your QE source tree (e.g. `/path/to/qe-source/PP/`):
1. Open the `CMakeLists.txt` file.
2. Insert the following block: 
   ```cmake
   ###########################################################
   # pw2coqui.x
   ###########################################################
   set(src_pw2coqui_x src/pw2coqui.f90)
   qe_add_executable(qe_pp_pw2coqui_exe ${src_pw2coqui_x})
   set_target_properties(qe_pp_pw2coqui_exe PROPERTIES OUTPUT_NAME pw2coqui.x)
   target_link_libraries(qe_pp_pw2coqui_exe
       PRIVATE
           qe_pw
           qe_modules
           qe_pp
           qe_upflib
           qe_fftx
           qe_mpi_fortran
           qe_xclib)
   ```
   above the section where `PP_EXE_TARGETS` is defined: 
   ```cmake 
   set(PP_EXE_TARGETS
       ...
   ) 
   ```
3. Add the new executable target to the list of `PP_EXE_TARGETS`:
   ```cmake
   set(PP_EXE_TARGETS
       ...
       qe_pp_pw2coqui_exe
       ...
   )
   ```

### Step 3: Recompile QE
Rebuild QE with the modified source and CMake configuration.
After successful compilation, the executable `pw2coqui.x` will be available
in your QE build directory (typically `bin/` inside the build tree).


## Usage
Once QE has been recompiled with the converter, you can proceed with the 
CoQuí [Quickstart tutorials](https://github.com/AbInitioQHub/coqui-tutorial/blob/main/quickstart/01s_dft_to_coqui_converter.ipynb) 
to generate CoQuí inputs from QE.

## Output schema

The converter writes `<prefix>.coqui.h5` with:

```
/System/                    # cell, atoms, k-mesh, symmetries, eigenvalues
/Orbitals/                  # mesh sizes, eigenvalues, occupations
/Hamiltonian/
  schema_version            # attribute: 2 = deeq-free + HARTREE on disk for all
                            # energy-valued datasets; 1 = deeq-free, Ry; absent =
                            # legacy export (readers scale x0.5 only for < 2).
                            # Contract: notes/paw_implementation_plan.md.
  pp_type                   # attribute: "ncpp" | "uspp" | "paw"
  {ncpp|uspp|paw}/          # plane-wave PP data: dion[_so], qq_so, qq_nt,
                            # vkb projectors, miller indices, local + scf
                            # potentials, vxc, augmentation_function_isp{nt}
  Species/{nt}/             # per-species pseudopotential data (Phase 0+)
    species_kind            # attribute: "ncpp" | "uspp" | "paw"
    r, rab, beta, dion
    lll, kbeta, indv, nhtol, nhtolm, nhtoj
    qfunc[l], qqq           # USPP/PAW augmentation
    aewfc, pswfc            # AE/pseudo partial waves (PAW)
    jjj                     # SOC: j_b per projector
    paw/                    # PAW-only subgroup
      pfunc, ptfunc         # AE / PS one-center pair densities (radial)
      augmom                # multipole moments
      ae_vloc, ae_rho_atc, oc
      pfunc_rel, aewfc_rel  # SOC: small relativistic component
      raug, iraug, lmax_aug, augshape   (attributes)
    Onecenter/              # PAW-only
      deltaC                # K_AE - K_PS, the .tex's ΔC_{αβγδ}
    Core/                   # PAW + GIPAW only (see below)
      n, l, ae_wfc          # AE core orbital quantum numbers + wavefunctions
      ncore_orbitals        (attribute)
```

### `--with-gipaw` requirement (core-valence ERIs)

Explicit core-valence and core-core exchange contributions require the AE
core wavefunctions, which are populated only when the PAW pseudopotential
is generated with the `--with-gipaw` option (`upf%has_gipaw == .true.`).
Without GIPAW data, the `Species/{nt}/Core/` group is omitted; CoQui will
fall back to a core-density-only treatment where applicable. Generate
GIPAW-enabled UPF files via QE's `atomic` package or download datasets
that explicitly support GIPAW.

### Spin-orbit / non-collinear

When the QE calculation has `lspinorb = .true.`, the converter additionally
exports `dion_so`, `qq_so`, `jjj`, `nhtoj`, and the small relativistic
components `paw/pfunc_rel`, `paw/aewfc_rel`. The remainder of the schema
is unchanged.