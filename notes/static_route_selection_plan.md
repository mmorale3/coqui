# Static-route selection: ERI vs hamiltonian for Hartree/exchange in SCF

Decided 2026-07-26 (mechanism option 1 of 3, user-selected): the direct
(`hamiltonian`) route becomes a **new interaction type** filling the existing
static ERI slots. This is the "future `set_H` route-selection project" of
`paw_dmatrix_cleanup_plan.tex` (I5/I7, §workstream D preamble); the
hamiltonian-side groundwork (`add_Vpp(nij, add_hartree, add_exchange)`,
`gen_Vhartree`, `gen_Vexchange`, route equivalence I7) is complete — this
plan is the SCF-side wiring.

## Mechanism (decided)

A lightweight `hamilt_eval_t` object is admissible wherever a static ERI
object is admissible today — the `interaction_hf`, `interaction_hartree`,
`interaction_exchange` slots of `mb_eri_t` (methods/ERI/mb_eri_context.h:30) —
and is rejected for `interaction` (dynamic terms always need factorized
ERIs). The single SCF seam
(`mb_solver.hf->evaluate(sF, Dm, slot, S, hartree, exchange)`,
scf_driver.cpp:130/306) is untouched; `hf_t` gains one overload. HF, GW,
GF2, qp loops, and the RPA `interaction_hf` energy path are all served by
the same seam. Per-term mixing (e.g. Hartree THC + exchange direct) falls
out of the existing slot structure.

## Phase 0 — derivations & decisions (before code)

- **0.1 div-treatment parity (derivation, not assumption).** Direct `v_x`
  zeroes G+Δk=0 (ignore_g0). Hypothesis: gygi ≡ ignore_g0 +
  `hf_t::HF_K_correction` (madelung·S·Dm·S, PRB 80,085114) applied to the
  direct K. Derive; short note in notes/; acceptance = matched-div exchange
  ERI-vs-direct on the Si fixture to THC tolerance. **The correct gygi
  implementation for the direct route is a DELIVERABLE of this plan**
  (lands in phase 1.2 + parity test 3.6).
  **RESOLVED 2026-07-26** (`static_route_gygi_note.md`): gygi in the ERI
  route is applied entirely at the operator level by the shared
  `hf_t::HF_K_correction` (−madelung·S·Dm·S, div-gated, route-free);
  ERI heads are ignore_g0 by convention and I7 certifies head parity.
  ⇒ the direct route calls the SAME correction unmodified — gygi works
  from day one, no interim gygi guard needed.
- **0.2 pseudopot sharing.** `simple_dyson` owns a `pseudopot`;
  `thc_reader_t` builds one too. `hamilt_eval_t` must share (shared_ptr)
  rather than rebuild — duplicate PAW runtime caches are expensive and the
  exx options must have a single source (`pseudopot::set_exx_options`).
  Decide the passing pattern by inspecting how thc_reader/dyson obtain
  theirs today.
  **RESOLVED 2026-07-26**: `hamilt::make_pseudopot(*MF)` is already the
  lazy shared-acquisition path (MF::get/set_pseudopot, MF.hpp:378;
  simple_dyson.h:66 and thc_reader_t.hpp:268–278 both use it).
  `hamilt_eval_t` uses the same call; on exx options it must CHECK
  against already-set options on the shared pseudopot and error on
  conflict (no last-writer-wins).

## Phase 1 — `hamilt_eval_t` + `hf_t` overload

- **1.1** `methods/ERI/hamilt_eval_t.hpp`: holds mpi context, `mf::MF&`,
  shared `pseudopot`, `div_treatment_e`, `paw_exx_options`; lazily reads and
  **caches** the distributed orbital set (one read, reused every iteration).
  Guards at use: nosym mesh (nk_ibz == nk) for BOTH exchange(nij) and
  Hartree(nij) — interim until phase 4 lifts it — npol == 1, host memory.
- **1.2** `hf_t::evaluate(sF, Dm, Hamilt_ERI auto&&, S, hartree, exchange)`:
  shm Dm_skij → local nij (convention A, Dm_ab = ⟨ψ_a|γ|ψ_b⟩; mind the
  v_h(nij) transpose history), then `gen_Vhartree` / `gen_Vexchange`
  accumulation into sF, div correction per 0.1. Normalization (ns_scl, N_k)
  is internal to the gen_* routines — verify against the I7 tests, do not
  re-scale at the call site.
- **1.3** Factor the toml→`paw_exx_options` parsing out of `thc_reader_t`
  into a shared helper usable by both `[interaction.thc]` and
  `[interaction.hamilt]`.

## Phase 2 — toml + dispatch

- **2.1** main.cpp (~:220): accept `[interaction.hamilt]` blocks — fields:
  `name`, `mean_field`, `div_treatment`, paw exx options (`vv_compensation`,
  `aug_lmax`, `qfac_cache_mb`, …).
- **2.2** `get_eri_block`: allow type "hamilt" for the three static slots;
  clear error if named in `interaction`.
- **2.3** Extend the mb_eri_t construction ladder (main.cpp:258–325) and the
  explicit-instantiation macros (scf_driver.cpp:404–500) for the
  constructible combos only: hf ∈ {thc,chol,hamilt} × corr ∈ {thc,chol};
  (hartree,exchange) ∈ {thc,chol,hamilt}² × corr ∈ {thc,chol}. Do not
  expand the pre-existing missing chol³ case — out of scope.
- **2.4** Python bindings (src/python/mbpt/mbpt_module.cpp): deferred;
  guard with a clear error.

## Phase 3 — tests (I7 in-SCF acceptance, Catch2 [paw], fast suite green)

- **3.1** HF-SCF route equivalence on the PAW fixtures (nosym): F_skij and
  total energy, ERI-static vs hamilt-static, matched div (ignore_g0),
  tol ~ THC thresh; first iteration AND converged.
- **3.2** Same on USPP and NCPP fixtures.
- **3.3** GW: one Dyson iteration, hf slot hamilt vs thc — Σ_static equal,
  dynamics bit-identical.
- **3.4** Per-term mixing: hartree=thc + exchange=hamilt, and the inverse.
- **3.5** Guard tests: symmetric-mesh error (interim, retired by phase 4);
  corr-slot rejection; exx-options conflict rejection (0.2).
- **3.6** gygi-parity test (required deliverable, closes 0.1): direct-route
  gygi vs ERI-route gygi, F_skij and energies to THC tolerance.

## Phase 4 — symmetry in the nij path (in scope)

Lift the nosym restriction on the direct route:

- **4.1** Symmetry-correct nij becsum (the open remainder of cleanup-plan
  A3/F4: full-BZ becsum landed for the diagonal path only): implement the
  symmetrized/full-BZ becsum for a general nij density matrix in the direct
  Hartree path.
- **4.2** `v_x(nij)` symmetry lift: reconstruct full-BZ pair densities from
  IBZ nij. Use G-space rotations (immune to the truncated-basis band-space
  rotation inexactness for topmost states — see
  project_paw_symmetry_rotation_truncation); mirror the existing
  hamiltonian/v_x G-space rotation machinery and the Pskna full-BZ lift
  (atom-perm + Wigner-D + Bloch phase) for the augmentation terms.
- **4.3** Tests: (a) sym-vs-nosym invariance — same physical system, same
  grid, symmetry on/off, F_skij and energies must agree; (b) in-SCF route
  equivalence (3.1/3.3 battery) repeated on a symmetric mesh; (c) retire
  the 3.5 symmetric-mesh guard.

## Follow-ups (out of scope, separate items)

npol>1, device memory, python bindings, per-iteration perf (Qfac cache
sizing).

## Commit discipline

One commit per phase, [paw]~[slow] green before each; STATUS checklist
entry added when phase 1 lands.
