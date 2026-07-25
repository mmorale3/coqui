# PAW/USPP consolidation plan — pointer + live status

The canonical plan is the LaTeX/PDF document (content lives there, not here):

- `notes/paw_dmatrix_cleanup_plan.tex`
- `notes/paw_dmatrix_cleanup_plan.pdf`

This file holds only the live STATUS checklist below (cheap to read/update;
printed into context at session start by a SessionStart hook).

## STATUS

Last updated: 2026-07-24 — A5 done. A5 notes (read_vnl_h5 hardening):
proj_per_atom length==nsp (ABINIT-fix message) + max≤nhm + Σ_atoms
nh(ityp)==total_num_of_proj; dion/dion_so shape checks + per-species
Hermiticity (≤1e-8 rel, active nh(s)·npol block only — padding
unconstrained) + scale (max|dion|≤1e3 Ha, >0 for USPP/PAW); Species sweep:
per-species nt{} group + species 'nh'==proj_per_atom + PAW 'paw' subgroup
required; require_read hard errors for aewfc/pswfc (PAW), qfuncl +
lll/nhtol/nhtolm/indv (USPP+PAW) with length checks (lll==nbeta, channel
maps==nh). DELIBERATE DEVIATION from the plan-A5 list: ae_vloc/vloc_ps are
WARNED-optional, not hard — their only consumer compute_paw_static_D is
unused in production since A1 (dion already carries the frozen D⁰ V_loc
baseline) and lih222_paw_hf predates the PS-side export (both species warn);
promote to required with B1/B4. ae_rho_atc/rho_atc_ps stay silent-optional
(absence can be physical — no NLCC; else-zero documented). Validated:
[paw]~[slow] 13479/33 green (value-identical); [hamilt]~[slow]~[thc]~[dft]
~[paw] 97442/9 green (NCPP + GaAs SOC exercise the dion_so Hermiticity
convention); [pseudo]~[slow] green; dft_eigenvalues re-confirmed at its
documented pre-existing failure (USPP/PAW 0.749/0.790 Ha, NCPP 5.8e-07 ok).
Test-hygiene finding: isdf_threshold_convergence ([!benchmark], ceph-path
data) HARD-ABORTS the whole binary via utils::check when its data is absent
— a bare "~[slow]" sweep dies there silently mid-run and skips everything
declared after it (invalid as a gate; use tag-positive filters).
A4 notes (kept): new
src/hamiltonian/paw/paw_runtime_caches.hpp — paw::runtime_caches held by
shared_ptr on pseudopot (mutable, shared across copies; every entry is keyed
on immutable state + explicit args, so never stale). Accessors: paw_aatab()
(aainit, lli = 1+max l); paw_qrad_tabs(Kmax, shape_restored) — qrad dq
UNIFIED to 0.01 project-wide (v_x had a local 0.05; strictly finer, suite
value-identical within tolerances), keyed (Kmax, mode, aug_lmax), larger-Kmax
tables reused for smaller requests; Pskna_full_bz() — cached View-2 lift,
MPI-COLLECTIVE on psp's own communicator at first call (all consumers are
psp-context collectives). Δk-keyed Qfac cache for direct v_x
(get_or_build_qfac_pair_factor in v_x_paw.hpp): key = quantized k_p−k_q
(exact — build adds Δk to every G), first-come-stays under 256 MB/rank
budget (knob deferred to C3), context (mesh, Gcut, mode) clears on change,
hits/builds/uncached logged at verbosity 3. becsum symm helpers slimmed to
(psp, n, kp_to_ibz, kp_trev, npol). ∫V·Q̂ loop in
compute_paw_deeq_from_becsum parallelized: root FFT + bcast V(G), G strided
over ALL comm ranks, native flat-double all_reduce (was root-serial).
NOT switched: thc_reader_t.hpp:572 lift site keeps its explicit-table build
(its _mpi is not provably psp's communicator; a collective cache on the
wrong comm deadlocks) — revisit in D1. Fast suite green, value-identical:
13479 assertions / 33 cases.
A3 notes (kept): compute_becsum_full_symm added
(v_h_paw.hpp) — full-BZ Pskna lift shared with the diagonal route via a new
compute_Pskna_full_bz(psp, …, lattv, recv, symm_list, …) convenience overload
in paw_symmetry.hpp (builds atom-perm + Wigner-D internally; caching = A4);
same IBZ band matrix at rotated points, complex-CONJUGATED at trev points
(γ_K = Σ ψ_K n* ψ_K† under time reversal). compute_becsum_full now does
Hermitian pair symmetrization ½(b_IJ+b*_JI) storing Re (exact for all
consumers: the antisymmetric Im part is inert against symmetric real Q/radial
kernels) with a HARD check on the anti-Hermitian residual ≤1e-8 (input-nij
Hermiticity contract; old warn-and-drop removed). Rerouted: v_h(nij) +
compute_paw_deeq(nij) (→ compute_deeq_scf transitively); A2 nosym guard in
add_vpp_impl removed. v_x_paw(nij) guard intentionally KEPT — its full-BZ
need is the exchange kernel itself (band-space NO route), not becsum.
Tests: new TEST_CASE becsum_full_symm (diag-nij≡diagonal on sym meshes
~1e-15; nosym reduction ≡ plain, exact incl. complex Hermitian nij);
vhartree_nij_vs_nii + add_vpp_i5_alignment extended with qe_lih222_paw_sym /
qe_si222_paw_sym sections (nk_ibz=3 of 8; H(nij)≡H(nii) ≤4.5e-14). GAP: no
USPP/PAW fixture populates kp_trev (0 trev points on all), so the trev conj
branch is unexercised — needs an A-tests fixture (e.g. PAW analogue of
lih223_inv). Fast suite green, 13479 assertions / 33 cases.
A2 notes (kept): nij add_Vpp now builds the same
native compute_paw_deeq(n, V_loc+V_H, include_static=true) as nii (F2 closed;
H(nij)≡H(nii) at ≤1e-15 on NCPP/USPP/PAW LiH _hf fixtures, new TEST_CASE
add_vpp_i5_alignment); add_hartree/add_exchange bools threaded through
add_vpp_impl → public add_Vpp → gen_H0 → hamilt::H (defaults true/false keep
all callers unchanged; flags-off ≡ H0 bit-identical; add_exchange ≡ H+K at
1e-15, host-only, device aborts). Missing qq_nt / augmentation_function_isp*
/ Hamiltonian/Species now hard
errors naming the converter rerun (part of A5 done early). SCF-driver audit
clean: simple_dyson/scf_driver/qp_scf_common/downfold_1e/pproc all take H0 via
no-density set_H0 (static-only) + ERI J/K — no double-count/omission, and the
density overloads have NO production callers, so no reference re-baselining
was triggered (fast suite value-identical, now 13456 assertions / 32 cases).
CAUTION for A-tests: the plan identity "nii≡nij≡no-density+Hartree" does NOT
hold as literally stated — the density path's KS-like D uses ∫(V_loc+V_H)Q̂
(plan I3) while H0+add_Hartree gives static + ∫V_H·Q̂ only; they differ by the
static ∫V_loc·Q̂·becpair term. Whether that term belongs in H0 (and hence in
the ERI-route Fock build, I7) must be settled there against Eq. d0's
−⟨Q̂|v_H[ñ_Zc]⟩ content of dion before writing the test.
A1 notes (kept): QE fixtures carry no ex_cvij → QE suite value-identical;
pre-existing failures (NOT from A1/A2): dft_eigenvalues USPP/PAW sections
(max_err 0.749/0.790 Ha at Li 1s semicore, needs own bisect; subsumed by
A-tests QE-eigenvalue diagnostic rework) and vx_sensitivity_ncpp (hidden
[!benchmark], hard-codes ~/ceph data absent here).
Note: test binaries need KMP_DUPLICATE_LIB_OK=TRUE (homebrew dual-libomp,
see CLAUDE.md).

Workstream A — pseudopot D-matrix refactor
- [x] A0 stabilize working tree (4 coherent commits; .swp gone; .DS_Store gitignored) — 2026-07-24
- [x] A1 two-tensor model: Dnn_atom_static = dion + ex_cvij (eager, ctor); remove QE deeq read; compute_deeq_scf stops mutating (thin wrapper, returns by value; non-mutation REQUIREd in test) — 2026-07-24
- [x] A2 align add_vpp paths with I5: no-density = static-only; nii/nij identical native build; add_hartree/add_exchange bools — 2026-07-24
- [x] A3 symmetry-correct nij becsum (full-BZ lift via compute_becsum_full_symm) + Hermitian pair symmetrization w/ hard residual check; add_vpp nosym guard removed (v_x(nij) guard kept, different scope) — 2026-07-24
- [x] A4 hoist per-call statics onto pseudopot (paw_runtime_caches.hpp: aainit, qrad @ unified dq=0.01, Pskna lift, Δk-keyed Qfac w/ 256 MB budget); ∫V·Q̂ loop parallelized over G + all_reduce; THC lift site deferred to D1 — 2026-07-24
- [x] A5 provenance checks at read time — dion Hermiticity+scale+shape, proj_per_atom length+Σ==nkb, per-species group/dataset sweep w/ length checks (ae_vloc/vloc_ps warned-optional until B1/B4: unused since A1, _hf fixture predates export) — 2026-07-24
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
