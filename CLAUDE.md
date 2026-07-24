# CoQui PAW/USPP working agreement

Project: PAW/USPP support in CoQui for a self-consistent GW framework.
Canonical corrective plan: `notes/paw_dmatrix_cleanup_plan.tex/.pdf` — its §1
invariants (I1–I8) are the contract for ALL pseudopot/converter/matrix-element/
THC changes. Live checklist: STATUS section of
`notes/paw_dmatrix_cleanup_plan.md` (injected at session start by a hook).

## Session protocol

- Orient from the STATUS checklist plus the plan section for the item at hand.
  Do NOT re-audit pseudopot/converters/notes — the 2026-07-24 audit findings
  (with file:line anchors) are in the plan.
- One plan item per session where practical; end sessions at commit
  checkpoints. Update the STATUS checklist when an item lands.
- At session start, state whether the task is mechanical or derivation-heavy;
  for mechanical work suggest the user switch `/effort` to medium (keep xhigh
  for physics derivations and design).
- Delegate mechanical subtasks (plumbing edits, test runs, log summarization)
  to cheaper-model subagents; keep physics reasoning in the main loop. No
  broad multi-agent audits — that knowledge is already in the plan.

## Commit policy (user-authorized 2026-07-24)

- After a plan item's tests pass, commit WITHOUT asking: one plan item per
  commit, descriptive message. Never bundle unrelated changes; never commit
  with the fast PAW suite red.

## Navigation & output hygiene

- Grep-first navigation; read line ranges, not whole files. Hot large files:
  `src/hamiltonian/pseudo/pseudopot.cpp`, `src/methods/ERI/thc_reader_t.hpp`,
  `src/hamiltonian/paw/v_x_paw.hpp`.
- Filter build/test output (grep for errors/summary); never dump full logs.
- `git diff --stat` first, then targeted hunks.

## Physics rules (non-negotiable)

- Derive conventions/prefactors before coding; "makes the test pass" is not
  evidence (see the rad_fac and −1/(4N_k) history).
- Validate values against references (QE/ABINIT/VASP, nosym variant, NCPP
  analogue) — finite-but-wrong is the dominant failure mode here.
- No DFT XC in the D matrix; static/dynamic split per plan I2/I3; density
  terms in add_vpp_impl only when nii/nij passed (I5); both compensation
  modes (I4); ERI-vs-direct route equivalence (I7); AE basis with identity
  overlap (I8).

## Build & test

- Build dir: `build/cpu` (driven by `build/cpu/build.bash`); rebuild one
  target with `make -j 4 <target>` from inside it.
- Test binary: `build/cpu/tests/bin/test_hamiltonian` (Catch2 v2). Go-to
  pre-commit filter: `"[paw]~[slow]"`.
- Always `OMP_NUM_THREADS=1` for `mpirun` of CoQui/QE binaries on this host.
- Since 2026-07: `KMP_DUPLICATE_LIB_OK=TRUE` is required to run the test
  binaries (homebrew openblas loads `libomp`-formula libomp while the binary
  links llvm's — duplicate-runtime abort). Harmless at OMP_NUM_THREADS=1;
  proper fix (single libomp in the link) is pending.
- MPI-collective helpers must never early-return on a per-rank fast path
  (deadlocks the other ranks) — participate with zero counts.
- Canonical QE converter source: `qe_converter/pw2coqui.f90` (never the
  QE-bundled copy).
