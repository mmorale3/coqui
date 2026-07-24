# PAW/USPP consolidation plan — pointer + live status

The canonical plan is the LaTeX/PDF document (content lives there, not here):

- `notes/paw_dmatrix_cleanup_plan.tex`
- `notes/paw_dmatrix_cleanup_plan.pdf`

This file holds only the live STATUS checklist below (cheap to read/update;
printed into context at session start by a SessionStart hook).

## STATUS

Last updated: 2026-07-24 — A1 done. A1 notes: QE fixtures carry no ex_cvij, so
the whole QE test suite is value-identical; the H0 flip (QE deeq → static-only)
pins NO stored test reference (only NCPP fixtures have H0 refs) — production
e_1e re-baselining still lands with A2 per plan. Pre-existing failures found
while validating (NOT from A1; bit-identical on the pre-A1 tree):
dft_eigenvalues USPP/PAW sections (max_err 0.749/0.790 Ha at Li 1s semicore —
broke sometime before A1, needs its own bisect/session; the A-tests item's
QE-eigenvalue diagnostic rework subsumes it) and vx_sensitivity_ncpp (hidden
[!benchmark] test, hard-codes ~/ceph data absent on this host).
Note: test binaries now need KMP_DUPLICATE_LIB_OK=TRUE (homebrew dual-libomp,
see CLAUDE.md); fast PAW suite green at 13441 assertions / 31 cases.

Workstream A — pseudopot D-matrix refactor
- [x] A0 stabilize working tree (4 coherent commits; .swp gone; .DS_Store gitignored) — 2026-07-24
- [x] A1 two-tensor model: Dnn_atom_static = dion + ex_cvij (eager, ctor); remove QE deeq read; compute_deeq_scf stops mutating (thin wrapper, returns by value; non-mutation REQUIREd in test) — 2026-07-24
- [ ] A2 align add_vpp paths with I5: no-density = static-only; nii/nij identical native build; add_hartree/add_exchange bools
- [ ] A3 symmetry-correct nij becsum (full-BZ lift) + Hermitian pair symmetrization + nosym guards until it lands
- [ ] A4 hoist per-call statics (aainit, qrad dq=0.01, Pskna lift, Δk-keyed Qfac); parallelize ∫V·Q loop
- [ ] A5 provenance checks at read time (hard errors, no silent fallbacks)
- [ ] A-tests: nii≡nij≡no-density+Hartree; sym≡nosym; ex_cvij factor-1 e_1e; QE-eigenvalue diagnostic

Workstream B — converter parity
- [ ] B1 QE: delete deeq/deeq_nc export; schema_version attribute
- [ ] B2 ABINIT: ae_vloc/vloc_ps export; per-species proj_per_atom; real vxc; Ewald/madelung; beta + Core/; shape_function check
- [ ] B3 native ex_cvij builder from Core/ae_wfc (Slater R^L + Gaunt²), validated vs ABINIT-XML ex_cvij
- [ ] B4 schema standardization (Ha on disk, miller_g sphere, ijtoh shape verified)
- [ ] B-tests: same-system Si PAW via both converters — dataset diff + e_1e/e_hf/e_rpa parity (closes ABINIT-mf anomaly)

Workstream C — augmentation-density modes
- [ ] C1 single mode flag (drop paw_exx_shape_restored bool); deltaC inclusion derived from mode
- [ ] C2 dense-grid THC augmentation for shape mode; lift the in_thc abort; Si a=10.20 acceptance
- [ ] C3 unify qrad dq + shared caches; Δk Qfac cache for production direct v_x
- [ ] C4 physics validation: −1.316447 operator identity (GW vs HF); match both modes vs ABINIT

Workstream D — ERI/THC route equivalence (Eq. path-equiv)
- [ ] D1 audit thc.h/thc.icc/thc_reader_t vs A conventions (AE basis, identity overlap)
- [ ] D2 route-equivalence matrix-element tests (THC vs hamiltonian), both modes, NCPP/USPP/PAW × sym/nosym
- [ ] D3 THC OOM at N_aux≳10k + synthetic l=3 augmentation unit test
- [ ] D4 Cholesky+USPP/PAW hard-abort (no augmentation yet)

Workstream E — notes/documentation
- [ ] E1 author canonical D-matrix doc (notes/paw_dmatrix_scgw.tex)
- [ ] E2 corrections to stale notes (converter plan exx_X claim, GW-vs-HF reconciliation, k-weight line, STEP3, LaNiO3 retest-pending)
