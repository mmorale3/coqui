# PAW-ISDF-THC external-reference benchmarks

Test scaffolding for Phase 4 / Phase 5 of `notes/paw_implementation_plan.md`.
Compares CoQui's PAW-ISDF-THC ERIs against externally-generated references
(VASP, PySCF GTH, or direct PAW Coulomb integration). The references are not
in the repository — provisioning a benchmark case means running an external
code, then dropping its output here.

## Layout

```
tests/paw_benchmark/
  run_benchmark.sh                     # harness, wired into CTest
  README.md                            # this file
  bench_<name>/                        # one directory per case
    coqui_input.toml                   # input to drive `coqui`
    eri_reference.h5                   # reference ERI tensor under /eri
    metadata.json                      # { "tol": 1e-3, "method": "VASP", ... }
    coqui_run.h5                       # produced by harness, gitignored
    coqui_run.log                      # produced by harness, gitignored
```

The harness:
* `SKIP`s (exit 77) when any input is missing or the `coqui` binary isn't built.
* `RUN`s and asserts max-abs `|ERI(coqui) − ERI(reference)| < tol` otherwise.

## Adding a benchmark case

1. Pick a small system (a few atoms, single k-point or 2×2×2). `bench_si_paw_g`
   would be a typical first case.
2. Generate the reference with VASP / PySCF / direct PAW Coulomb.
   Save the ERI tensor (any indexing convention — matched on numeric values
   only) to `eri_reference.h5` under dataset `/eri`.
3. Author a CoQui TOML input that produces `coqui_run.h5` containing the
   same ERI tensor at dataset `/eri`. Include `mean_field`, `interaction =
   thc`, `eri = "store"` (or equivalent) to materialize the tensor.
4. Drop `metadata.json` with `tol` and provenance notes.

Recommended starting tolerances:
* Si NCPP-vs-PAW (smooth-only): 1e-4
* TMO PAW-ISDF: 1e-3 (the `α = N_μ/N_orb` rank you set governs this)

## Running by hand

```
cd tests/paw_benchmark
./run_benchmark.sh bench_<name>          # one case
./run_benchmark.sh --all                 # every bench_*
```

## CTest integration

Wired in `tests/paw_benchmark/CMakeLists.txt` (registered from the
project's top-level via `add_subdirectory`). Each `bench_<name>` directory
becomes a separate CTest with `LABELS "paw_benchmark;external"`. They
SKIP at runtime if the reference data isn't on disk; that keeps `ctest -L
paw_benchmark` honest without pre-failing the suite on a fresh clone.
