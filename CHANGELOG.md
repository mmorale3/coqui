
# Changelog

## Unreleased — dynamic vertex

### Added

- Dressed-leg Γ₁ readout of the dynamic-rung ladder: the dynamic rung is contracted with legs dressed by the static
  resolvent through closed-form Matsubara Grams of the pole families, exact to round-off and without an L0 application on
  the frequency-dependent intermediate. Used for the polarization (Γ₁ and the one-bare-rung column) and for the Σ-side
  deposits; host and device paths. Input `pol_vertex_dyn_dressed = "auto" | "on" | "off"`.
- Device engine for the dynamic vertex (CUDA): the unit (static resolvent LU, L0, dense τ rungs, DLR refit) is resident on
  the GPU; mirror-paired τ rungs share one gemm; optional two-input rung pass; streaming THC rung and frequency-factorized
  rung variants; device Σ deposits.
- Partial rung residency: when the dense τ rungs do not fit on the device, the remaining ones are rebuilt on the device or
  copied from pinned host memory per application; the split is chosen at run time by timing the candidate splits
  (no per-device tuning). Input `pol_vertex_dyn_device_memory = "auto" | "rebuild" | "host" | "resident"`.
- Hybrid / frequency-factorized dynamic rung builds, threaded serial sections of the rung application, the static
  resolvent and the static-ladder readout.
- Dynamic pair Σ vertex in Wannier mode; host parking of W during the vertex drivers to free device memory.
- In-run self-consistency of the polarization vertex on a coarse k sublattice (`pol_vertex_sc = "hybrid" | "direct" |
  "interp"`, `pol_vertex_sc_mesh`, `pol_vertex_sc_mix`, restart through `pol_vertex_sc_restart_file`): the vertex
  polarization is solved on the sublattice transfers and interpolated in the Wannier frame elsewhere, on IBZ meshes.
  Frequency sampling of the vertex polarization (`pol_vertex_sc_nu_nodes`) and of the dynamic Σ vertex against an in-run
  reference pass (`pol_vertex_sigma_dyn_fit_file = "self"`, `pol_vertex_sigma_dyn_auto_nodes`, `_fit_rank`), with optional
  reference renewal (`pol_vertex_sigma_dyn_ref_period`, `_ref_adaptive`) and a full final pass
  (`pol_vertex_sigma_dyn_final_full`).
- Causal projection of Σ_c onto positive-semidefinite spectral weights on every SCF iteration (host and CUDA paths;
  `vertex_debug scf_causal_proj*` keys), with causality meters for G and Σ_c.
- Frequency-factorized dynamic rung: the per-τ rungs are replaced by R time functions K_r (`pol_vertex_dyn_rung_rank` or
  `pol_vertex_dyn_rung_tol`), applied either reconstructed per particle-hole representative (default) or factorized
  (`pol_vertex_dyn_rung_apply`), cutting the rung memory from 46 per-τ matrices to R.
- Node-shared W̄ cache for the vertex (`pol_vertex_wcache = "shared"`) and a ladder conditioning meter.
- Projected Wannier functions with a frozen energy window (`frozen_energy_max`): every band below the given energy is kept
  exactly at each k, on top of `frozen_range`; per-k Gram diagnostics.
- Tie-robust ISDF pivot selection (`chol_tie_rtol`, default 1e-6): the interpolating points no longer depend on the
  reduction order, i.e. on the rank count or the GPU model; 0 restores the previous selection.
- Small Si unit-test fixtures (2×2×2, 3×3×3, 4×4×4; symmetric, time-reversal-only and unreduced meshes) for the IBZ and
  dynamic-vertex symmetry tests.

### Improved

- Si kp444, 12-band window, ν = 0 dressed Γ₁ (2 ranks): CPU 1776 s → 1308 s (hybrid rung build: 937 s); A100
  552 s → 349 s; H100 108 s; dynamic Σ vertex on H100 975 s → 352 s. Results unchanged to ≤ 1e-13 relative.

### Changed (robustness — no silent workflow changes)

- A device stage that does not fit aborts with the reason; `pol_vertex_dyn_device_fallback = true` (default false) allows
  moving it to the CPU with a WARNING.
- New inputs, all default false, each re-allowing a behaviour that can change the answer and is otherwise an abort:
  `vertex_allow_missing_head`, `pol_vertex_allow_bare_rung`, `pol_vertex_allow_unprojected`,
  `vertex_allow_unchecked_reflection`.
- `vertex_debug` keys are validated against a known-key list (an unknown key aborts) and parsed strictly.
- A Σ^C rung that violates the pair symmetry W_PQ(q) = W_QP(−q) is reported with a prominent warning.

### Known limitations

- The auxiliary DLR pole fit of the vertex kernels is a regularized least-squares map (stable in scGW + vertex); it
  satisfies the Φ-derivability identity Tr[Σ^C G] + Tr[P^C W] = 0 only approximately (relative 2–5e-4 on LiH 2×2×2,
  2–4e-2 with the gygi q → 0 head).
- The vertex supports collinear spin only (`npol = 1`).

## CoQui v0.3.0 [2026-06-03]

### Added

- DIIS for QP-SCF via new `qp_com_diis_residual` and `vspace_heff` classes, with unit tests covering both Dyson- and QP-SCF paths.
- New double-bisection chemical-potential search algorithm robust for insulators (finds the middle of the acceptable μ window rather than the first crossing), and a midpoint algorithm for QP-SCF. The algorithm is externally selectable via `mu_update_alg`.
- Unified analytic-continuation interface (`ac`) with five backends:
  - CoQui Pade (C++ implementation).
  - TRIQS Pade.
  - AAA algorithm via the adapol library.
  - Minipole via the mini_pole library.
  - `maxent_triqs` analytic-continuation via TRIQS MaxEnt.
- Python wrapper around the C++ logger.
- EDMFT convergence metrics (Weiss field, `|Gloc - Gimp|`, `|Wloc - Wimp|`) are stored in the EDMFT checkpoint, with utility functions to visualize them.

### Improved

- DIIS is now the default iterative solver for both Dyson-SCF and QP-SCF.
- Shared chemical-potential search implementation between Dyson-SCF and QP-SCF, with improved log output.
- `simple_dyson::compute_eigenspectra` refactored to exploit Hermitian symmetry.
- GW+EDMFT driver refactor: docstrings and sensible defaults across `gw_edmft` routines.
- Hermitian symmetry enforced on upfolded self-energies, impurity solutions, and `Delta_tau` / `D0_tau` preparations.
- EDMFT convergence checks moved fully to the imaginary-time axis.
- Improved GW+EDMFT logging: prints `|Gloc - Gimp|`, `|Wloc - Wimp|`, A(ω=0) proxy via `Gloc(β/2)`, and tail-fit setup.
- Suppressed ct-seg solver output during `mu_imp` search; improved `mu_imp` log output.
- `band_plot` and `spectral_plot` are importable directly from the `post_proc` submodule.
- Extract QP energies from Dyson-SCF checkpoint as a post-processing utility.
- Docstring pass for the Python IAFT module.
- `read_last_iter.py` retains only the last iteration.
- MBPT and Wannier90 routines now support explicit `outdir` parameter for consistent checkpoint and output file path handling.
  - MBPT: Simplified path resolution logic. Uses legacy `output` key if present and non-empty, otherwise constructs path as `outdir + "/" + prefix` with `outdir` default of `"./"`.
  - Wannier90: Added explicit `outdir` parameter to all public entrypoints (`to_wannier90`, `wannier90_library_mode`, `wannier90_library_mode_from_nnkp`, `append_wannier90_win`, `mlwf_h5_from_wannier90_output`). Output files are consistently placed at `outdir + "/" + prefix` with `outdir` default of `"./"`.
  - Updated all corresponding Python wrapper docstrings and examples to document the new `outdir` parameter and expected file locations.
- TRIQS compatibility: `triqs.gf` → `triqs.gfs`, updated function names from `triqs.modest`. `c2py` pinned to a known-good commit while upstream stabilizes.
- Refactored the TRIQS MaxEnt analytic-continuation interface in `post_proc/analytic_cont.py`.
- Imaginary-part tolerance warnings for `Nelec` and the correlation energy now scale with the IAFT `eps` rather than using fixed thresholds.
- Regenerated Python binding (`.wrap`) files that are compatible with c2py (2f7f21dd56ff2161299194313c6f1fb600d52a7c).

### Fixed

- Iterative-solver path when disabled in `dmft_embed`.
- Compilation error when BUILD_UNIT_TESTS is disabled.

### API Updates

- Python IAFT: renamed `ir_notation` to `phys_notation` (now a keyword-only argument).
- GW+EDMFT: `wmax` and `eps` inputs for the EDMFT subspace are consolidated under the `iaft` dictionary.
- GW+EDMFT: restart now requires an existing checkpoint.
- ct-seg interface updated for the upstream TRIQS rename of `nn_nu` to `nn_nu_dlr` (`measure_nn_nu` → `measure_nn_nu_dlr`).

### Default Value Updates

- Updated IAFT defaults:
  - default basis changed to `dlr` (from `ir`) for both `iaft.basis` and legacy `iaft_basis` interfaces.
  - default `prec` changed to `medium` (from `high`) when `eps` is not provided.
- Updated THC defaults:
  - default `ecut` changed to `1.4 * mf->ecutwfc()` if wavefunction fft grid is available, otherwise to `0.4 * mf->ecutrho()`.
  - default `thresh` changed to `1e-5` (from `1e-10`).
  - when `nIpts` is set and `thresh` is not explicitly provided, `thresh` is auto-resolved to `1e-13` to avoid premature termination before reaching requested `nIpts`.

## CoQui v0.2.0 [2026-02-04]

### Added

- Fully self-consistent GW+EDMFT solver and interface, including the new `dmft` Python submodule (`src/python/dmft/`), drivers, utilities (bath fitting, chemical potential, checkpoints, SCF driver, etc.)
- TRIQS CT-SEG interface (planned to be moved to the TRIQS library in the future).
- New examples for cRPA and GW+EDMFT.
- Unit test for Wannier90 MLWF workflow using SrVO3.
- `mlwf_h5_from_wannier90_output` utility to read Wannier90 output files and write MLWF data into CoQui h5 format.

### Improved

- Major refactor for downfolding routines to improve the API, enhance modularity, and increase code readability.
- Downfolded Coulomb interactions are now stored under `downfolded_model` h5 group within the input h5 group, e.g. `scf/iter{}` or `embed/iter{}`.
- `IAFT` initialization from CoQui checkpoint h5.
- Improved TRIQS ModEST interface for quantum embedding.
- Improved SCF runtime controls: added `iter_alg.enable` to optionally disable iterative mixing/DIIS paths, added `mu_tolerance` for explicit chemical-potential convergence control, and added stricter checkpoint/input dataset validation for `greens_func_source`/`greens_func_iteration` workflows.
- Examples for MLWF library and standalone modes.
- Examples for Fourier transform on imaginary time/frequency grids on Python.
- Chol-GW for molecules refactoring.
- Improved divergence treatment for dielectric function, and a customized option for metallic systems. The previous default `gygi` is now an alias for `gygi_smallest_q` for backward compatibility.

### Fixed

- Wannier90 compilation issue due to recent updates in Wannier90 codebase. 

### API Updates

- Renamed MBPT/downfolding input keys from `input_type` and `input_iter` to `greens_func_source` and `greens_func_iteration`.
- Renamed `downfold_local_coulomb` to `downfold_coulomb` to reflect support for non-local downfolded interactions.

## CoQui v0.1.0 [2026-01-23]

### Added

- New Python API example scripts for mean-field, MBPT, downfolding, interaction, and Wannier workflows under `examples/python_interface/`.
- `TEST_INPUT_DIR` environment variable to point to the test input files.
- Feature to write screened interaction in ISDF basis to HDF5 in MBPT with `dump_w_to_h5`.
- New QE and PySCF converter examples and input files for solids and molecules.
- ISDF for generating the auxiliary basis in the python interface (`coqui.run_isdf`).
- SrVO3 unit test files for Wannier and QE workflows under `tests/unit_test_files/`.

### Improved

- Refactored DIIS routines and improved the restart logic. Rename `diis_start` to `diis_warmup`.
- Improved SCF convergence criteria.
- ISDF auxiliary basis can now be written on reduced or full FFT mesh via `write_zeta_on_fft_mesh` parameter.
- Refactored Python interfaces for interaction and mean-field modules.
- Updated `README.md`, example documentation and docstring.

### Fixed

- Various bug fixes in PySCF interface and import logic.

### Removed

- Deprecated and redundant example input files and pseudopotentials.

### Notes

- This is the first formal changelog entry, summarizing all changes on `develop` since the last stable `main` release.
