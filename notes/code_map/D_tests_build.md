# D: tests / fixtures / build

## 0. Repo root layout (/Users/mmorales/Projects/ISDF_metric/coqui)
- `src/` — all C++ sources (arch, grids, hamiltonian, IO, main, mean_field, methods, numerics, orbitals, potentials, python, scripts, utilities, wannier)
- `cmake/` — `unit_test.cmake` (ADD_UNIT_TEST/ADD_MPI_UNIT_TEST), `FindFFTW.cmake`, `FFTW3/`
- `tests/unit_test_files/` — ALL test fixtures (qe/, bdft/, pyscf/); no top-level `docs/` dir exists
- `qe_converter/` — standalone QE->CoQuI converter (has own tests/CMakeLists.txt)
- `extern/`, `examples/`, `build/` (existing build tree), `.omc/`
- Root markdown: README.md, CHANGELOG.md, CONTRIBUTING.md, CONTRIBUTORS.md, REFERENCES.md, LICENSE
- **NO `docs/` directory at all** — so no `docs/lapw_thc/`, no PLAN.md/decisions.md convention present in this checkout.

## 1. Test registration mechanism
- `cmake/unit_test.cmake:2` `ADD_UNIT_TEST(TESTNAME TEST_BINARY [args...])` — plain add_test, sets `OMP_NUM_THREADS=1`, LABEL "unit".
- `cmake/unit_test.cmake:9` `ADD_MPI_UNIT_TEST(TESTNAME TEST_BINARY PROC_COUNT [args...])` — wraps in `${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} <n>`, same env/label.
- `CMakeLists.txt:81` `SET(PROJECT_UNIT_TEST_DIR ${PROJECT_BINARY_DIR}/tests/bin)` — all test binaries land here.
- `CMakeLists.txt:239-246` `CTEST_NPROC` cache var (default 1, clamped to `MPIEXEC_MAX_NUMPROCS`) — this is the MPI rank count passed to ADD_MPI_UNIT_TEST calls.
- `CMakeLists.txt:116` `option(BUILD_UNIT_TESTS ON)`; `CMakeLists.txt:435-449` fetches **Catch2 v2.13.10** (note: v2, not v3) via FetchContent.

## 2. Fixture path resolution at runtime — `src/utilities/test_input_paths.hpp`
- Central helper. Two overloads of `utils::utest_filename(...)`:
  - `utest_filename(mf::mf_source_e src)` (line ~108) — default per-source fixture; honors globals `qe_prefix/qe_outdir`, `bdft_prefix/bdft_outdir`, `pyscf_prefix/pyscf_outdir` if those files exist on disk, otherwise falls back to a hardcoded default fixture.
    - qe default: `tests/unit_test_files/qe/lih_kp222_nbnd16/` prefix `pwscf`
    - bdft default: `tests/unit_test_files/bdft/lih_kp222_nbnd16/` prefix `bdft`
    - pyscf default: `tests/unit_test_files/pyscf/si_kp222_krhf/` prefix `pyscf`
    - model: `tests/unit_test_files/model/nb2_chol_gamma/` prefix `model`
  - `utest_filename(std::string src)` — string-keyed registry of named fixtures (the "which fixture" knob used by most method tests).
- Path base is the **cmake-configured macro `PROJECT_SOURCE_DIR`** (from `src/config.h.cmake.in` -> `configuration.hpp`), NOT an env var and NOT a relative path. So fixtures are read from the source tree, not the build tree.
- **Documented way to point a test at a different fixture**: set the `qe_outdir`/`qe_prefix` (resp. bdft_/pyscf_) globals — these are declared `extern` here and are set from test `main()` command-line args (see Catch2 main below). Otherwise, add a new key to the string overload.

### Named fixture keys registered in test_input_paths.hpp (string overload)
model_chol, qe_si211, qe_si111, qe_si222_so, qe_si222_ncpp, qe_si222_uspp, qe_si222_paw,
qe_si222_paw_sym, qe_lih222, qe_lih222_sym, qe_lih222_paw, qe_lih222_paw_sym, qe_lih222_uspp,
qe_lih223, qe_lih223_inv, qe_lih223_sym, qe_lih222_hf, qe_lih222_paw_hf, qe_lih222_uspp_hf,
qe_GaAs222_hf, qe_GaAs222_so_hf, qe_GaAs222_so, qe_svo222_sym, bdft_lih222, bdft_lih222_sym,
bdft_si222_paw_ab, pyscf_si222, pyscf_h2_222, pyscf_li_222u, pyscf_h2o_mol.
(`bdft_si222` -> `bdft/si_kp222_krhf/` is present but COMMENTED OUT.)
Note `bdft_si222_paw_ab` carries an in-source comment: ABINIT-sourced Si PAW (abinit2coqui WFK+POT+DEN+pawxml+corewf, LDA-PW 12-electron, 2x2x2 full-BZ nosym), added because "the real_ylm converter bug was invisible to QE-only route tests"; generation input is `PROVENANCE.abi` in that fixture dir.

## 3. Test utilities
- `src/utilities/catch_main.cpp` (103 lines) — the `catch_main` library target. Catch2 **v2** style (`CATCH_CONFIG_RUNNER`), owns `main()`, inits `boost::mpi3::environment`, defines the globals `qe_prefix/qe_outdir/bdft_prefix/bdft_outdir/pyscf_prefix/pyscf_outdir` (lines 37-39) and registers CLI opts `--qe_prefix --qe_outdir --pyscf_prefix --pyscf_outdir --bdft_prefix --bdft_outdir` (lines 72-89) via `Catch::Session` + `Catch::clara::Opt`.
  - **=> This IS the documented way to run a unit test against a different fixture**: `./test_methods_eri "<testcase>" --qe_outdir <dir> --qe_prefix <prefix>`. It only affects the `utest_filename(mf_source_e)` overload (the "default_MF(mpi, mf::qe_source)" path), NOT the string-keyed overload.
- `src/utilities/test_common.hpp:50` `utils::make_unit_test_mpi_context()` — lazily creates and caches a single `mpi_context_t<mpi3::communicator>` shared_ptr for the whole test binary. Every TEST_CASE starts with `auto& mpi = utils::make_unit_test_mpi_context();`.
- `src/utilities/test_common.hpp:61,69,80,91` `utils::VALUE_EQUAL(A,B,m=1e-8,eps=1e-8)` (scalar + complex overloads) and `:102` `utils::ARRAY_EQUAL(A,B,m,eps)`. Implemented as `REQUIRE_THAT(x, WithinRel(y,eps) || WithinAbs(y,m))` — so **the "tolerance" arg is `m` (absolute) with `eps` (relative) defaulting to 1e-8**. This is the dominant assertion style across all method tests.
- `src/mean_field/default_MF.hpp` (228 lines) — **the helper that builds a `mf::MF` from a fixture name**. Three overloads:
  - `mf::default_MF(comm, mf_source_e src, outdir, prefix, ftype=xml_input_type)` -> `mf::make_MF(...)`
  - `mf::default_MF(comm, mf_source_e src, ftype)` -> resolves via `utils::utest_filename(src)`
  - `mf::default_MF(comm, std::string src, ftype=xml_input_type)` (line 59) -> big if/else that maps each fixture key to `utest_filename(key)` + the correct source enum + correct default input file type (`xml_input_type` vs `h5_input_type`). Tests almost always use this string form.
- Adding a new fixture requires editing BOTH `src/utilities/test_input_paths.hpp` (path+prefix) and `src/mean_field/default_MF.hpp` (source enum + file type).

## 4. Build targets
- Test executables are `test_<TEST_ID>`; every methods test dir follows the identical 4-line recipe:
  `add_executable(test_${TEST_ID} <srcs>)` / `target_link_libraries(... catch_main ...)` /
  `add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})` /
  `set_tests_properties(... WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})`.
  **The ctest test name == the executable name** (there is NO `catch_discover_tests` anywhere), so ctest sees ONE test per binary and Catch2 TEST_CASE names are only selectable by running the binary manually.
- ctest names relevant here: `test_methods_eri`, `test_methods_hf`, `test_methods_gw`, `test_methods_gf2`, `test_methods_scf`, `test_methods_embed`, `test_methods_mbpt`, plus `test_mf`/`test_qe`/`test_bdft`/`test_pyscf`/`test_model`, `test_hamilt*`, numerics/utils/wannier/orbitals tests.
- All test binaries output to `${PROJECT_BINARY_DIR}/tests/bin` (`PROJECT_UNIT_TEST_DIR`), which is also the CWD for the ctest run (tests write/`remove()` scratch h5 files like `./bdft.mbpt.h5` there).
- **THC code lives in library target `eri_lib`** (`src/methods/ERI/CMakeLists.txt:22`): sources `cholesky.cpp thc.cpp mb_eri_context.h eri_utils.cpp`; links PUBLIC `utils numerics meanfield hamilt nda_c h5_c slate`. `thc.icc`/`thc_aux.icc`/`thc.h`/`thc_reader_t.hpp` are header-included into `thc.cpp`, so they are NOT separate TUs.
- `methods_lib` aggregates `meanfield methods_tools_lib eri_lib hf_lib scr_coulomb_lib gw_lib gf2_lib scf_lib embed_lib pproc_lib`.
- MPI launcher: `ADD_MPI_UNIT_TEST` builds `${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${CTEST_NPROC} ${MPIEXEC_PREFLAGS} <binary>`. `CTEST_NPROC` defaults to 1 -> by default the whole suite runs single-rank under mpiexec.

## 5. THC file sizes (src/methods/ERI/)
- `thc.cpp` 680 lines (29 KB) — the only compiled TU
- `thc.h` 642 lines (28.5 KB)
- `thc.icc` 1983 lines (88.7 KB)
- `thc_aux.icc` 2412 lines (102.3 KB)
- (context) `thc_reader_t.hpp` 2432 lines (117 KB), `cholesky.icc` 1170, `eri_utils.hpp` 418

## 6. Fixture inventory — `tests/unit_test_files/` (total ~724 MB, committed in-tree)
Three source families: `qe/`, `bdft/`, `pyscf/`. (A `model/` family is referenced by `test_input_paths.hpp` for `model_chol` -> `model/nb2_chol_gamma/` but that directory does NOT exist on disk.)

### QE fixture anatomy (typical)
`scf.inp` (+`scf.out`), `pwscf.xml` (QE XML), `pwscf.save/` (QE native save dir), the pseudopotential `.UPF/.upf`, `pw2coqui.inp` (+`.out`), and **`pwscf.coqui.h5`** — the converted CoQuI h5 produced by the `qe_converter`. Some older ones also carry `VKB`/`VLTOT`/`VSC` binary dumps and `pw2bgw.inp`/`pp_*.inp`.
- `mf::default_MF(comm, "qe_*")` chooses `xml_input_type` or `h5_input_type` per key in `src/mean_field/default_MF.hpp` — i.e. some QE fixtures are read through the XML+save path and others through the converted `.coqui.h5`.

### ALL silicon (si_*) fixtures
| key | dir | k-mesh | sym | pseudo | size | contents |
|---|---|---|---|---|---|---|
| `qe_si111` | `qe/si_kp111_nbnd8` | 1x1x1 | nosym+noinv | ONCV NC (`Si_ONCV_PBE_sr.upf`), ecutwfc 50, nbnd 8 | 4.6M | scf.inp, pwscf.xml (35 KB), pwscf.save/, pwscf.coqui.h5 (1.5 MB), VKB/VLTOT/VSC, pp_*/pw2bgw inp |
| `qe_si211` | `qe/si_kp211_ndnb8` (note typo "ndnb") | 2x1x1 | (no scf.inp kept) | — | 5.5M | basis.txt, pwscf.xml (38 KB), pwscf.save/, pwscf.coqui.h5 (1.95 MB), VKB/VLTOT/VSC |
| `qe_si222_so` | `qe/si_kp222_nbnd8_so` | 2x2x2 | nosym+noinv | ONCV fully-relativistic (`Si_ONCV_PBE_FR-1.1.upf`), nbnd 24, spin-orbit | 19M | scf.inp, pwscf.xml (50 KB), pwscf.save/, pwscf.coqui.h5 (8.9 MB) |
| `qe_si222_ncpp` | `qe/si_kp222_ncpp` | 2x2x2 | nosym,noinv,no_t_rev,force_symmorphic | ONCV NC, nbnd 16, ecutwfc 50 / ecutrho 200 | 8.5M | scf.inp+out, pwscf.xml, pwscf.save/, pwscf.coqui.h5 (4.5 MB), pw2coqui.inp+out |
| `qe_si222_uspp` | `qe/si_kp222_uspp` | 2x2x2 | nosym (full BZ) | USPP (`Si.pbe-n-rrkjus_psl.1.0.0.UPF`), nbnd 16 | 50M | same layout, pwscf.coqui.h5 = 45 MB |
| `qe_si222_paw` | `qe/si_kp222_paw` | 2x2x2 | nosym (full BZ) | PAW (`Si.pbe-n-kjpaw_psl.1.0.0.UPF`), nbnd 16 | 51M | same layout, pwscf.coqui.h5 = 46.6 MB |
| `qe_si222_paw_sym` | `qe/si_kp222_paw_sym` | 2x2x2 | **symmetry ON** (IBZ) | PAW, nbnd 16 | 45M | same layout, pwscf.coqui.h5 = 41.9 MB |
| `pyscf_si222` | `pyscf/si_kp222_krhf` | 2x2x2 KRHF | — | GTO/pyscf | 18M | `pyscf.h5` (118 KB, the mf), `Orb_fft/` (orbitals on FFT grid), `fftdf_eri.h5` (3.1 MB reference ERI), `scf.py`, `gen_gw_gt.py`, and **reference G(w)/G(t) h5**: `hf_Gw_Gt_beta1000_wmax1.2_{medium,high}.h5`, `gw_Gw_Gt_beta1000_wmax1.2_{medium,high}.h5` |
| `bdft_si222_paw_ab` | `bdft/si_kp222_paw_abinit` | 2x2x2 full-BZ nosym | ABINIT PAW, LDA-PW 12-el | 27M | `bdft.h5` (28.3 MB), `PROVENANCE.abi` (generation input), `probe_hartree.py` |
| (disabled) `bdft_si222` | `bdft/si_kp222_krhf` | 2x2x2 KRHF | — | — | 2.3M | `bdft.h5` (2.36 MB), `basis.txt` — **commented out** in test_input_paths.hpp |

### Non-Si fixtures (sizes)
qe: lih_kp222_nbnd16 13M, _hf 12M, _sym 7.2M, _paw 21M, _paw_hf 25M, _paw_sym 17M, _uspp 20M, _uspp_hf 24M; lih_kp223_nbnd16 18M, _inv_only 13M, _sym 14M; GaAs_kp222_hf 23M, GaAs_kp222_so 44M, GaAs_kp222_so_hf 44M; svo_kp222_nbnd40 49M (prefix `svo`, in an `out/` subdir).
bdft: lih_kp222_nbnd16 5.9M, lih_kp222_nbnd16_sym 2.6M, `make_qe_unit_test_files` (script dir).
pyscf: si_kp222_krhf 18M, h2_kp222_krhf 28M, li_kp222_kuhf 104M, h2o_mol 10M.

### Which fixtures the ERI/THC tests default to
- `src/methods/ERI/tests/test_thc.cpp` mostly uses `mf::default_MF(mpi, mf::qe_source)` -> falls through `utest_filename(qe_source)` to **`qe/lih_kp222_nbnd16` prefix `pwscf`** (LiH 2x2x2, NOT silicon). Explicit-key cases in that file use `qe_lih222_sym`, `qe_GaAs222_so`, and a pyscf case.
- `src/methods/HF/tests/test_thc_hf.cpp` uses `qe_lih222`, `qe_lih222_sym`, `qe_lih223`, `qe_lih223_sym`, `qe_lih223_inv`, `qe_GaAs222_so`, `pyscf_si222`, `pyscf_h2o_mol`.
- Silicon fixtures are therefore mainly exercised by the mean_field / hamiltonian / pyscf-route tests, not by the default THC/ERI path.

## 7. Unit tests covering THC / ERI / HF / RPA / GW
All in `src/methods/*/tests/`. Recall: ctest name = binary name, Catch2 TEST_CASE names are sub-selectors.

### ctest `test_methods_eri` (`src/methods/ERI/tests/CMakeLists.txt`, srcs test_thc.cpp test_cholesky.cpp test_thc_reader.cpp test_chol_reader.cpp; links `catch_main utils numerics meanfield eri_lib nda_c h5_c slate`)
- `test_thc.cpp` TEST_CASEs (all tag `[methods]`): thc_intpts_ibz(165, `qe_lih222_sym`), thc_intpts(199, default qe=lih222), thc_intpts_so(221, `qe_GaAs222_so` h5), thc_rotated_basis(241, default qe), thc_intpts_pyscf(335), thc(345), thc_so(414), thc_ranges(554), thc_nnr_blk(648), thc_io(714), thc_coul_metric(781, default qe). **thc_chol_ov(801) and thc_chol_ls(878) are inside a `/* ... */` block — DEAD/disabled** (they still reference an old `methods::thc` ctor signature and `mpi.comm` typos).
- Assertion style in test_thc.cpp is mostly *smoke/shape* (`REQUIRE(V_.global_shape()[1] == npts)` at 660-661, `ARRAY_EQUAL` cross-checks between HOST/DEVICE/UNIFIED memory paths) — **no pinned absolute energies in the ERI-level THC test**.
- `test_thc_reader.cpp`: thc_incore(49, pyscf default), thc_incore_device(64, qe), make_thc(83, qe), thc_outcore(105, pyscf), thc_ls(121, `pyscf_h2o_mol`).
- `test_cholesky.cpp`: cholesky_seq / _seq_ecut / _blocked / _blocked_ecut / _diagkk / _range / _EHF / _io (all `default_MF(mpi, mf::qe_source)` = lih222), cholesky_pyscf(299).
- `test_chol_reader.cpp`: chol_reader(45, pyscf), chol_reader_single_write(70, qe), make_cholesky(92, qe).

### ctest `test_methods_hf` (test_thc_hf.cpp test_chol_hf.cpp test_hamilt_hf.cpp; links `... eri_lib hf_lib scf_lib`)
- `test_thc_hf.cpp`: thc_hf_qe_components(53), thc_hf_qe(116), thc_qphf_qe(187), thc_hf_pyscf(250), thc_hf_mol(360), thc_hf_dlr_vs_ir(393).
  - **PINNED EXAMPLE 1** `src/methods/HF/tests/test_thc_hf.cpp:143` `VALUE_EQUAL(e_hf, e0, 1e-5)` with per-SECTION `e0`: `qe_lih222` & `qe_lih222_sym` -> **-4.2818278244126935**; `qe_lih223`, `qe_lih223_sym`, `qe_lih223_inv` -> **-4.287485045424232**; `ls_thc_nosym` (`qe_lih222`, cd_dir="./") -> -4.2818278244126935. In-source comment: references come from chol-HF at Cholesky tol 1e-10; accuracy ~1e-5 at alpha=20 (THC rank = nbnd*20).
- `test_chol_hf.cpp`: chol_hf_qe(52, `qe_lih222`), chol_hf_pyscf(84, `pyscf_si222`).
- `test_hamilt_hf.cpp` (direct/hamiltonian route vs THC route): hamilt_hf_route_equivalence(131), hamilt_hf_sym_vs_nosym(183), hamilt_hf_gygi_parity(210, `qe_lih222_paw`), hamilt_hf_sym_hartree(236, `qe_lih222_paw_sym`), hamilt_exx_options_sharing(281, `qe_lih222_paw`).
  - **PINNED EXAMPLE 2 (relative, not absolute)** `test_hamilt_hf.cpp:167-172`: `VALUE_EQUAL(ham.e_hf, ref.e_hf, tol)` + `ARRAY_EQUAL(ham.F_first/F_final, ref.*, tol)` where the reference is the all-THC run of the SAME fixture. SECTIONS at 175-182: `qe_lih222_paw`, `qe_lih222_uspp`, `qe_lih222` (ncpp), `qe_lih222_paw_sym`, all tol **1e-4**. Also `test_hamilt_hf.cpp:208` `VALUE_EQUAL(e_sym, e_nosym, 1e-5)` comparing `qe_lih222_paw` vs `qe_lih222_paw_sym`.

### ctest `test_methods_gw` (test_chol_gw.cpp test_thc_gw.cpp test_hamilt_gw.cpp; links `... eri_lib hf_lib gw_lib scf_lib`) — this is where RPA lives too
- `test_thc_gw.cpp`: thc_g0w0_qe_bdft(46), thc_gw_qe(121), **thc_rpa_qe(194)**, thc_gw_pyscf(235), **thc_rpa_pyscf(292)**, thc_gw_mol(328), thc_gw_dlr_vs_ir(365).
  - **PINNED EXAMPLE 3** `src/methods/GW/tests/test_thc_gw.cpp:83-94` — G0W0 quasiparticle energies pinned at tol 1e-5 for SECTIONS `qe_lih222`(104), `qe_lih222_sym`(108), `bdft_lih222`(112), `bdft_lih222_sym`(116): k=0 homo-1 **-1.959166853350**, homo **-0.343590135344**, lumo **0.769452793794**, lumo+1 **0.819356108320**; k=1 -1.949608656698 / -0.234561625134 / 0.332168314756 / 0.691491471197.
  - scGW: `test_thc_gw.cpp:161-162` `e_hf = -4.224737908935479`, `e_corr = -0.11256940748889475`, tol 1e-5, fixtures `qe_lih222` / `qe_lih222_sym`; repeated at 393-394 for the DLR-vs-IR test on `qe_lih222_sym` / `qe_lih223_sym`.
  - RPA: `test_thc_gw.cpp:213` `e_rpa = -0.07295472568310496` tol 1e-5 (`qe_lih222`, `qe_lih222_sym`); `:308,:319` `e_rpa = -0.06481111309877628` tol 1e-6 (pyscf default = `pyscf/si_kp222_krhf`).
  - pyscf GW: `:261-262,:278-279` `e_hf = 0.9096946909052888`, `e_corr = -0.11439719195215467`, tol 1e-6, fixture **`pyscf_si222`** (via `default_MF(mpi, mf::pyscf_source)`).
  - molecular GW: `:351-352` `e_hf = -84.66602711500559`, `e_corr = -0.41696395032933564`, tol 1e-4, `pyscf_h2o_mol`.
- `test_chol_gw.cpp`: chol_g0w0_qe(46), chol_gw_qe(116), chol_rpa_qe(147), chol_gw_pyscf(176), chol_rpa_pyscf(211), chol_gw_mol(242).
- `test_hamilt_gw.cpp`: hamilt_gw_hf_slot(65).

### Other related ctest binaries
- `test_methods_gf2` (test_thc_gf2.cpp, test_chol_gf2.cpp), `test_methods_scf` (test_simple_dyson.cpp, test_scf_common.cpp), `test_methods_embed` (test_embed.cpp), `test_methods_mbpt` (test_mbpt.cpp, links `methods_lib`).

## 8. Actual registered ctest names (from configured build tree `coqui/build/CTestTestfile.cmake`)
Current build cache has `CTEST_NPROC=2`, `MPIEXEC_EXECUTABLE=/opt/homebrew/bin/mpiexec`, `MPIEXEC_PREFLAGS=--oversubscribe`, `CMAKE_BUILD_TYPE=Release`. Every MPI test therefore launches as `mpiexec -n 2 --oversubscribe <binary>`.
- methods: `test_methods_eri`, `test_methods_hf`, `test_methods_gw`, `test_methods_gf2`, `test_methods_scf`, `test_methods_embed`, `test_methods_mbpt`
- mean_field: `test_mean_field`, `test_mean_field_qe`, `test_mean_field_bdft`, `test_mean_field_model`, `test_mean_field_pyscf` (pyscf one is NON-MPI / serial)
- other: `test_hamiltonian`, **`test_hamiltonian_np2_shm`**, `test_utilities`, `test_ac_pade`, `test_math_distributed_nda`, `test_math_shared_nda`, `test_math_slate`, `test_math_fft_nda`, `test_iaft`, `test_iaft_dlr`, `test_iaft_ir`, `test_nda_functions`, `test_sparse`, `test_wannier`, `test_orbitals`
- **`qe_converter_schema_<fixture>`** — 21 auto-generated bash tests, one per `tests/unit_test_files/qe/*/pwscf.coqui.h5` (incl. all 7 si_* fixtures). Registered by `qe_converter/tests/CMakeLists.txt` which GLOBs fixtures at configure time and filters via `h5dump -n` for a `/Hamiltonian/Species` group. LABELS `schema;qe_converter`, `SKIP_RETURN_CODE 77` (skips if h5dump absent). The converter binary itself is NOT built by CoQuI's cmake.
- `build/ctest_eri_baseline.log` records a prior run: `test_methods_eri` Passed in **279.40 sec** at np=2 — useful runtime baseline.

## 9. Unusual bits in test registration
- `src/hamiltonian/tests/CMakeLists.txt:33-40` — the ONLY place that passes a **Catch2 tag as an extra ctest arg**: `add_mpi_unit_test(test_hamiltonian_np2_shm "<bin>" 2 "[shm_h0]")` with `TIMEOUT 900`, hard-coded at 2 ranks (guarded by `MPIEXEC_MAX_NUMPROCS GREATER 1`). The in-file comment says this exists because a stray global-communicator collective in `set_H0`/`compute_int_VQ` DEADLOCKS at np>1 and is invisible at the `CTEST_NPROC=1` default. **This is the template to copy for registering a tag-selected subset as its own ctest.**
- `add_mpi_unit_test`'s `${ARGN}` is the general mechanism for passing Catch2 tag/test-name filters AND the `--qe_outdir/--qe_prefix` style fixture overrides at registration time.
- No `catch_discover_tests` -> ctest granularity is one test per binary (except the np2_shm case).
- Tests run with CWD `${PROJECT_BINARY_DIR}/tests/bin` and several of them `remove()` scratch files (`./bdft.mbpt.h5`, `chol_info.h5`, `Vq*.h5`) from that CWD on the root rank.

## 10. Build targets
- **Only one executable**: `coqui` (`src/main/CMakeLists.txt:8`, from `src/main/main.cpp`), links `orbit wannier_lib methods_lib`, installed to `bin/`. `compare_eri`, `check_eri_symmetry`, `time_cholesky` are all commented out; `src/numerics/sandbox/` builds two extra unnamed sandbox executables.
- Library targets: `utils`, `utils_f`, `catch_main`, `numerics` (INTERFACE), `ac_lib` (INTERFACE), `device_kernels` (INTERFACE), `iaft_utils_lib`, `fft_lib`, `sparse_lib`, `cuda_kernels`, `io_lib`, `arch_lib`, `orbit`, `hamilt`, `meanfield`, `wannier_lib`, and the methods family: **`eri_lib`** (THC lives here), `hf_lib`, `gw_lib`, `gf2_lib`, `scf_lib`, `scr_coulomb_lib`, `embed_lib`, `pproc_lib`, `methods_tools_lib`, `mb_state_lib`, `methods_lib`.
- README build recipe (`README.md:89-99`): `cmake -DCTEST_NPROC=[NCORES] ...` then `make -j && ctest && make install`.

## 11. Docs convention — ANSWER
- **There is no `docs/` directory anywhere in this repo, on any branch** (`git ls-tree -d` over origin/main, develop, isdf_vertex, isdf_vertex_leanW, uspp-paw-isdf, unit_tests, paw, isdf_metric shows only `cmake examples extern qe_converter src tests`, plus `notes` on two branches).
- **There is no `docs/lapw_thc/` and no `PLAN.md`/`decisions.md` files anywhere in git history** (grep over `git log --all --name-only` finds zero `PLAN.md` or `decisions.md`).
- The de-facto convention is instead a top-level **`notes/`** directory, present on `origin/unit_tests` and `origin/paw` but NOT on the checked-out `isdf_metric` branch. Naming pattern is `<topic>_implementation_plan.md` / `<topic>_plan.md` / `<topic>_reference_benchmarks.md` / LaTeX+PDF pairs, e.g.:
  - origin/paw: `notes/abinit2coqui_converter_plan.md`, `notes/converter_h5_contract.md`, `notes/converter_h5_inventory_{abinit2coqui,pw2coqui}.md`, `notes/paw_article_results/` (CSV/PDF/PNG figure data + sbatch scripts + harvest/fit .py)
  - origin/unit_tests: `notes/paw_implementation_plan.md`, `notes/paw_gw_reference_benchmarks.md`, `notes/paw_rpa_reference_benchmarks.md`, `notes/paw_isdf_thc_prb.tex/.pdf`, `notes/paw_separability_note.tex/.pdf`
  - other history: `notes/scgwt_implementation_plan.md`, `notes/paw_dmatrix_cleanup_plan.md`, `notes/si_gw_paw_dataset_plan.md`, `notes/static_route_selection_plan.md`, `perf_report/gpu_fix_plan.md`
- Repo-level markdown: README.md (build/run), CHANGELOG.md, CONTRIBUTING.md, CONTRIBUTORS.md, REFERENCES.md, plus per-example READMEs under `examples/` and `qe_converter/README.md`, `src/python/mean_field/abinit_interface/README.md`.
- NOTE: the working notes for the current task live OUTSIDE the repo at `/Users/mmorales/Projects/ISDF_metric/notes/` (sibling of `coqui/` and `nda/`), containing `isdf_metric_knobs_spec.md` and `code_map/{A_selection,B_orbitals_vG,C_input_drivers,D_tests_build}.md`.
