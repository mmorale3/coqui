# C: TOML input / energy drivers

## 1. TOML input plumbing

### Parse chain
- `src/main/main.cpp:53` `main()` — cxxopts CLI (`--compute`, `--verbosity`, `--debug`, `--stacktrace`, positional `filenames`). Only `inputs[0]` used (main.cpp:145). Builds `InputParser parser; parser.read(myinput)` (main.cpp:146-148); parse failure -> `app_error("Error parsing input file. Check format.")` + exit(1).
- `main.cpp:156-172` dispatch on `--compute` into `run<MEM>(world, parser)` template (main.cpp:180).
- `src/IO/ptree/InputParser.hpp:34` class `InputParser`; `read()` (:56) picks parser by file extension; `parse()` (:68) supports json/xml/toml. TOML -> `io::read_toml` (InputParser.hpp:79).
- `src/IO/ptree/toml_utilities.hpp:106` `io::read_toml(istream, ptree&)`: splits the file into "sections" with `split_into_sections` (:58) so repeated top-level tables (e.g. two `[interaction.thc]`) are kept as separate ptree children; each section is `toml::parse`d (tomlplusplus), echoed to the log ("Input Parameters" header, :114-121), converted to JSON via `toml::json_formatter` (:124) and read into a boost `ptree` with `read_json`, then `main_pt.add_child(it.first, it.second)` (:129).
- NET EFFECT: everything downstream sees a **boost::property_tree::ptree**, not toml nodes. All option reads go through `src/IO/ptree/ptree_utilities.hpp` helpers: `io::get_value<T>` (:145, aborts if missing), `io::get_value_with_default<T>` (:184), `io::get_array<T>` (:160), `io::get_array_with_default` (:201), `io::check_exists` (:122), `io::find_child` (:134), `io::get_compute_space` (:252, reads `compute` = default/cpu/host/gpu/device/unified).

### Top-level block dispatch (main.cpp:192-524, loop over `parser.get_root()`)
`mean_field` (:195) | `interaction` (:207 -> sub-types `thc`/`cholesky`/`hamilt`) | `isdf` (:226) | `orbitals` (:232) | `mp2` (:238, not implemented) | `hf|qphf|rpa|gw|qpgw|gw_dca|evgw|gf2` (:243, all routed to `methods::mbpt`) | `downfold_1e` (:367) | `downfold_2e` (:373) | `hf_downfold` (:383) | `gw_downfold` (:392) | `dmft_embed` (:401) | pproc: `ac|unfold_bz|band_interpolation|spectral_interpolation|local_dos|dump_vxc|dump_hartree` (:407) | `unfold_wfc` (:415) | `hamiltonian` (:422) | `wavefunction` (:452) | `wannier90` (:479).
Unknown TOP-LEVEL block name -> `app_error("unknown calculation type: {}")` + exit(1) (main.cpp:517-522). Objects are kept in `mf_list`/`thc_list`/`chol_list`/`hamilt_list` maps keyed by name so later blocks can refer to them by name (main.cpp:185-188).

### [interaction.thc] -> THC/ISDF construction path
- `main.cpp:216-217` `int_type=="thc"` -> `methods::add_thc(mpi_context, int_pt, mf_list, thc_list)`.
- `src/methods/ERI/eri_utils.hpp:62` `add_thc()`: reads `name` (default auto `"thc_AaBbCcDd_<n>"`), checks name uniqueness, resolves `mean_field` via `mf::get_mf`, then `make_thc(mf, pt)` -> stores `thc_reader_t` in `thc_list`.
- `src/methods/ERI/eri_utils.cpp:27` `make_thc()`: pre-validates `nIpts`/`thresh`, `storage`, decides build-vs-read (`save` file exists -> read via `thc_reader_t(mf,storage,save,init)`, else build via `thc_reader_t(mf,pt,false,init)`).
- `src/methods/ERI/thc_reader_t.hpp:82` `thc_reader_t(MF, ptree, isdf_only, initialize)` — main option-consuming ctor; it also constructs the low-level builder `thc(...)` (`src/methods/ERI/thc.cpp:105`) with the SAME ptree, which reads the numeric/perf knobs.
- `[isdf]` top-level block (main.cpp:226-230) -> `methods::make_isdf` (eri_utils.cpp:55) -> same `thc_reader_t` ctor with `isdf_only=true` -> `build_isdf_only(check_accuracy, write_zeta_on_fft_mesh)` (thc_reader_t.hpp:181).

### FULL list of accepted options in [interaction.thc] / [isdf] (name : default : parse site)
Selection / identity
- `name` : auto `thc_AaBbCcDd_<n>` : eri_utils.hpp:68
- `mean_field` : (required, name of a [mean_field] block) : resolved by `mf::get_mf` (eri_utils.hpp:70)
- `type` : only when nested under a generic `interaction` tag; must equal "thc" : eri_utils.hpp:98 (`io::get_value`, ABORTS if missing)
- `compute` : "default" (default/cpu/host/gpu/device/unified) -> `_MEM_EVAL` : thc_reader_t.hpp:86 via io::get_compute_space
Core THC/ISDF controls
- `nIpts` : 0 -> `_Np` (Np = nIpts * nbnd) : thc_reader_t.hpp:93 (also pre-checked eri_utils.cpp:30)
- `thresh` : sentinel -1.0; auto-resolved to 1e-13 if nIpts>0 else 1e-5 -> pivoted-Cholesky threshold : thc_reader_t.hpp:108-112; the builder re-reads `thresh` with default 1e-5 at thc.cpp:120
- REQUIRED: `nIpts>0` or `thresh>0` else abort (thc_reader_t.hpp:113, eri_utils.cpp:32)
- `ecut` : `1.4*ecut_wfc` if wfc grid else `0.4*ecutrho` -> `thc::ecut` (Coulomb/rho FFT grid) : thc.cpp:113-114
- `use_least_squares` : false (LS-THC path) : thc.cpp:124
- `X_orbital_range` : `[0,nbnd]` (2-int array -> nda::range) : thc_reader_t.hpp:98
- `Y_orbital_range` : same as X_orbital_range : thc_reader_t.hpp:99
Storage / IO
- `storage` : "incore" ("incore"|"outcore"; validated eri_utils.cpp:41) : thc_reader_t.hpp:87
- `save` : "" incore / "./thc.eri.h5" outcore -> `_eri_file`; if the file exists the ERI is READ instead of built : eri_utils.cpp:43-44, thc_reader_t.hpp:88,174
- `format` : "bdft" -> `_format` : thc_reader_t.hpp:89
- `cd_dir` : "" (dir with precomputed Cholesky/CD data) : thc_reader_t.hpp:90
- `init` : true (defer initialization) : eri_utils.cpp:37
Performance / parallel
- `matrix_block_size` : 1024 -> `default_block_size` (SLATE block size, must be >0) : thc.cpp:118,129
- `chol_block_size` : 8 -> `default_cholesky_block_size` (must be >0) : thc.cpp:119,130
- `r_blk` : 1 -> `nnr_blk` (real-space grid batching) : thc.cpp:121
- `distr_tol` : 0.2 -> `distr_tol` (processor-grid shape; larger => more procs on k/Q axis) : thc.cpp:122
- `memory_frac` : 0.75 -> clamped to [0.25,0.90] : thc.cpp:123,132
ISDF-only block extras ([isdf])
- `check_accuracy` : false : thc_reader_t.hpp:181
- `write_zeta_on_fft_mesh` : false : thc_reader_t.hpp:182
Nested `[interaction.thc.potential]` sub-table -> `thc::vG` (thc.cpp:117, `io::find_child`), parsed in `src/potentials/potentials.hpp:43`
- `type` : "coulomb"; `ndim` : 3; `cutoff` : 1e-8; `screen_type` : "none"; `screen_length` : 1.0 (src/potentials/coulomb.hpp:69-72)
PAW/USPP-only (read only when `mf->pp_type()` is PAW or USPP, thc_reader_t.hpp:130)
- `paw_aug` : true (:131); `paw_onsite` : true for PAW / false for USPP (:132); `paw_vgl` : true (:136); `paw_vll` : true (:137); `paw_isdf_tol` : 1e-12 (:138); `paw_isdf_cache_h5` : "" (:139); `paw_aug_ecut` : 0.0 = off (:160); `paw_isdf_metric` : "coulomb" ("coulomb"|"l2") (:162)
- shared PAW-exx surface via `parse_paw_exx_options` (`src/methods/ERI/paw_exx_options_parse.hpp:38`, called thc_reader_t.hpp:168): `vv_compensation` : "moment" ("moment"|"shape", validated :42); `aug_lmax` : -1 (:49); `qfac_cache_mb` : 256 (:51)

### End-to-end example for ONE option: `distr_tol`
TOML `[interaction.thc] distr_tol = 0.4` (examples/toml_input_interface/mbpt/rpa.toml)
-> `io::read_toml` puts it in ptree node `interaction.thc.distr_tol` (toml_utilities.hpp:106-130)
-> main.cpp:212-217 iterates `interaction` children, `int_pt` = the thc subtree, calls `methods::add_thc`
-> eri_utils.hpp:72 `make_thc` -> eri_utils.cpp:50 `thc_reader_t(mf, pt, ...)` -> thc_reader_t.hpp:92 constructs `thc(_MF.get(), *_mpi, pt, false)`
-> parse site `src/methods/ERI/thc.cpp:122` `distr_tol( io::get_value_with_default<double>(pt,"distr_tol",0.2) )` -> member `thc::distr_tol` (declared `src/methods/ERI/thc.h`)
-> use site: passed to the processor-grid heuristics in `src/methods/ERI/thc_aux.icc` / `thc.icc` (grep `distr_tol`) to choose the {nqpools,...} processor grid; echoed by `thc::print_metadata` (thc.cpp:156).

### Unknown / misspelled options: SILENTLY IGNORED
There is NO schema/whitelist validation of keys. Reads are pull-based (`io::get_value_with_default`), so a key never queried is simply never seen; a typo (e.g. `distr_tolerance`, `nipts`) silently falls back to the default. Keys ARE case-sensitive (boost ptree paths). The only hard errors are: (a) unknown TOP-LEVEL block name (main.cpp:517), (b) unknown `interaction` sub-type (main.cpp:223), (c) missing REQUIRED keys read with `io::get_value` (aborts), (d) explicit `utils::check` value validation (storage, vv_compensation, ranges, positive block sizes). Mitigation for the user: the whole parsed TOML is echoed back to the log at startup (toml_utilities.hpp:114-121), so a typo is visible in the input echo but not flagged.

## 2. Energy drivers

### End-to-end flow of a run
`coqui <input.toml>` -> `main()` (src/main/main.cpp:53) -> InputParser/read_toml -> `run<MEM>` (main.cpp:180) loops over top-level TOML tables in file order, building objects into name-keyed maps:
1. `[mean_field.qe]` -> `mf::add_mf` (`src/mean_field/mf_utils.hpp:123`), QE reader options parsed at mf_utils.hpp:89-93: `prefix` (REQUIRED, io::get_value), `outdir` ("./"), `ecut` (0.0), `nbnd` (-1), `filetype` ("h5"), `name` (auto `mf_AaBbCcDd_<n>`).
2. `[interaction.thc]` (or `.cholesky` / `.hamilt`) -> `methods::add_thc` etc. (main.cpp:216-221) -> thc_reader_t built/read.
3. Method table `[hf] [qphf] [rpa] [gw] [qpgw] [evgw] [gw_dca] [gf2]` (main.cpp:243) -> `methods::get_eri_block` resolves `interaction = "<name>"` (plus optional `interaction_hf`, `interaction_hartree`, `interaction_exchange` slots for mixed J/K/corr ERIs) -> builds `methods::mb_eri_t` -> `methods::mbpt(cname, mb_eri, pt)`.
4. `methods::mbpt` (`src/methods/MBPT_drivers.cpp:121`) reads the common method options and dispatches on `solver_type`:
   - "rpa" -> `rpa_loop` (MBPT_drivers.cpp:175-181)
   - "hf" -> `scf_loop` with `mb_solver_t(&hf)` (:182-193)
   - "gw" -> `scf_loop` with `mb_solver_t(&hf,&gw,&scr_eri)` (:195-247)
   - "gf2" -> `scf_loop` with gf2_t (:249)
   - "gw_dca" (:282), "qphf" -> `qp_scf_loop` (:294), "evgw" (:308), "qpgw" (:330)
   - unknown -> `APP_ABORT("mbpt: Unknown solver type: {}")` (:354)

### Common options of a method block (parsed in MBPT_drivers.cpp:129-143)
`div_treatment` "gygi" | `hf_div_treatment` "gygi" | `niter` 1 | `conv_thr` 1e-8 | `const_mu` false | `mu_tolerance` 1e-9 | `mu_update_alg` "midpoint" | `restart` false | `greens_func_source` "scf" ("mf"/"scf"/"embed") | `greens_func_iteration` -1 | `output` (legacy) else `outdir` "./" + `prefix` (REQUIRED) -> stem for `<stem>.mbpt.h5` (`resolve_mbpt_output_stem`, MBPT_drivers.cpp:54).
IAFT grid options read from the same table (`src/numerics/imag_axes_ft/IAFT.hpp:82-92`): `beta` 1000.0, `iaft_wmax` (default from mf), `iaft_basis` "dlr", `iaft_prec` "medium" ("high"/"medium"/"low"), `iaft_eps`; or a nested `[<method>.iaft]` sub-table with keys `wmax`/`basis`/`prec`/`eps` (IAFT.hpp:64-77).
Mixing sub-table `[<method>.iter_alg]` (`src/numerics/iter_scf/iter_scf_utils.hpp:50-65`): `enable` true (MBPT_drivers.cpp:185), `alg` (REQUIRED: "damping"/"diis"), `mixing` 0.7, `max_subsp_size` 5, `diis_warmup` 3, `diis_start` -1, `residual_type`.
Solver-specific: gw `screen_type` "rpa", `dump_w_to_h5` false, `wannier_file`/`translate_home_cell` for gw_edmft; gf2 `gf2_direct_type`, `gf2_exchange_alg`, `gf2_exchange_type`, `gf2_save_C`, `gf2_sosex_save_memory`, `t_prescreen_thresh`; qpgw/evgw `qp_type`, `ac_alg` "pade", `eta` pi/beta, `Nfit` 18, `off_diag_mode` "fermi", `keep_scr_coulomb_fixed`.

### (a) HF / exchange
- Solver class `solvers::hf_t` (`src/methods/HF/hf_t.h`), ctor takes `hf_div_treatment` (hf_t.cpp:53-73 prints divergence-treatment choice). `hf_t::evaluate(sF_skij, Dm_skij, eri, S_skij, hartree=true, exchange=true)` — the two bool flags select J only, K only, or both. Backends: `thc_hf.cpp:42`/`thc_hf.icc` (THC), `cholesky_hf.cpp:37`/`cholesky_hf.icc` (Cholesky, `add_J`:119 / `add_K`:275), `hamilt_hf.cpp`/`hamilt_hf.icc:36` (direct route via `hamilt::Vhartree`/`hamilt::Vexchange`, hamilt_hf.icc:92).
- HF energy from the Fock matrix: `eval_hf_energy(sDm_skij, sF_skij, sH0_skij, k_weight, ...)` -> `(e_1e, e_hf)` (`src/methods/SCF/scf_common.cpp:64`; imaginary-part warnings :96-100).
- HF SCF driver: `scf_loop` (`src/methods/SCF/scf_driver.cpp`, declared scf_driver.hpp:47); energies computed at scf_driver.cpp:191-194 and PRINTED at :197-207 ("Energy contributions / non-interacting (H0) / Hartree-Fock / correlation / total energy" + per-term "energy difference"). Returns `(e_1e+e_hf, e_corr)` (:231). Per-iteration state written by `chkpt::dump_scf` (:214) — G/Sigma/F/Dm/mu, NOT energies.

### (b) RPA correlation energy
- `rpa_loop` (`src/methods/SCF/rpa.cpp:38`, decl scf_driver.hpp:68) — single-shot (no SCF): builds G from the mean-field, evaluates F via `hf_t::evaluate`, then `e_rpa = mb_solver.corr->rpa_energy(G_tskij, corr_eri)` (rpa.cpp:98).
- THC RPA kernel: `gw_t::rpa_energy` (`src/methods/GW/thc_rpa.cpp:43`, impl `src/methods/GW/thc_rpa.icc`): builds Pi(q,tau) via `scr_coulomb_t::eval_Pi_qdep`, tau->w, then `thc_rpa_energy_all_impl` (thc_rpa.icc:72) which accumulates Tr(Pi*Z) + ln|det(I-Pi*Z)| per (w,q) (diagnostics printed at thc_rpa.icc:284,309; unphysical-polarizability warning :195). Cholesky counterpart `chol_rpa_energy_impl` (`src/methods/GW/cholesky_rpa.icc`).
- PRINTS (rpa.cpp:104-107): "One-electron energy", "Hartree-Fock energy", "RPA energy", "Total energy" (all app_log level 2).
- STORES (rpa.cpp:127-136): h5 group `RPA` in `<prefix>.mbpt.h5` with datasets `1e_energy`, `hf_energy`, `rpa_energy`. This is the ONLY driver that writes energies to h5.
- In an SCF (gw/gf2) run the correlation energy instead comes from the Galitskii-Migdal-type `eval_corr_energy(comm, FT, sG_tskij, sSigma_tskij, k_weight)` (`src/methods/SCF/scf_common.cpp:107`).

### (c) Hartree / Coulomb energy
- No standalone "Hartree energy" driver: J is a component of the Fock matrix (`hf_t::evaluate(..., hartree=true, exchange=false)`).
- rpa.cpp:112-119 re-evaluates F with exchange ONLY, then prints "Exchange energy: ..." (:117) and "Hartree energy: ... (= E_HF - E_x)" (:118) — i.e. Hartree is obtained by subtraction, for cross-code comparison.
- One-electron decomposition (kinetic / local / dion / core-valence exchange / int_VQ) printed by `print_e1_decomposition` (`src/methods/SCF/energy_decomposition.hpp:68`, prints at :143-155), called from rpa.cpp:124.
- Hartree POTENTIAL (not energy) driver: `[dump_hartree]` pproc block (main.cpp:407-413) -> `methods::post_processing` (`src/methods/pproc/pproc_drivers.hpp:228-246`) -> `hamilt::dump_hartree` (`src/hamiltonian/one_body_hamiltonian.hpp:550`) which calls `hamilt::Vhartree` (:368/:489). Example input: examples/toml_input_interface/pproc/vhartree.toml. `[dump_vxc]` is the analogous Vxc dump.

### Grep handles for energy prints
`rg 'Hartree-Fock energy|RPA energy|Exchange energy|Hartree energy|Energy contributions|non-interacting \(H0\)' src/methods` hits only `src/methods/SCF/rpa.cpp` (104-119) and `src/methods/SCF/scf_driver.cpp` (197-207, 380-387 for the QP-SCF variant).

### COMPLETE example TOML — RPA run (examples/toml_input_interface/mbpt/rpa.toml, verbatim)
```toml
[mean_field.qe]
name     = "mf_qe"
prefix   = "pwscf"
outdir   = "qe_output_dir/OUT/"
filetype = "h5"

[interaction.thc]
name        = "eri"
mean_field  = "mf_qe"
storage     = "incore"
thresh      = 1e-6
chol_block_size = 8
r_blk       = 20
distr_tol   = 0.4

[rpa]
interaction = "eri"
beta      = 2000
lambda    = 1200.0
iaft_prec = "high"
outdir    = "./"
prefix    = "rpa"
```
NOTE: `lambda = 1200.0` in this shipped example is NOT read by any parse site (IAFT reads `beta`, `iaft_wmax`/`wmax`, `iaft_prec`, `iaft_basis`, `iaft_eps`) — a live demonstration that unknown keys are silently ignored.
HF variant (mbpt/hf.toml) adds `niter=12`, `restart=false`, and a `[hf.iter_alg]` sub-table (`alg="damping"`, `mixing=0.7`). GW variant (mbpt/gw.toml) adds `conv_thr`, `div_treatment` ("ignore_g0"/"gygi"/"gygi_extrplt"/"gygi_extrplt_2d") and `[gw.iter_alg]`.
Other example inputs: examples/toml_input_interface/{interaction/{isdf_thc_eri,ls_thc_eri,chol_eri}.toml, mbpt/{hf,gw,rpa,qpg0w0,qphf,diis,mix_eri}.toml, mean_field/{qe,pyscf,bdft}_mf.toml, pproc/*.toml, downfolding/**}. Python-API equivalents in examples/python_interface/.
