# PAW/USPP consolidation plan — pointer + live status

The canonical plan is the LaTeX/PDF document (content lives there, not here):

- `notes/paw_dmatrix_cleanup_plan.tex`
- `notes/paw_dmatrix_cleanup_plan.pdf`

This file holds only the live STATUS checklist below (cheap to read/update;
printed into context at session start by a SessionStart hook).

## STATUS

Last updated: 2026-07-24 (late) — Workstream B COMPLETE (B1–B4, four
commits, one per item). B1: pw2coqui drops deeq/deeq_nc, stamps
schema_version; verified end-to-end on a fresh Si-NCPP conversion (attr
present, no deeq, check_schema green). B2: all six abinit2coqui items —
Species/paw/{ae_vloc,vloc_ps} from the same XC-free ionic-Hartree pair
assemble_dij0 integrates (tails → −zval checked); per-species proj_per_atom
in both writers (F10a, matches the A5 reader); real vxc/vxc_with_nlcc from
an ABINIT DEN via new xc_functionals.py (PW92+PBE, FD derivatives, spectral
grad/div — reproduces QE v_xc from QE's own ρ(G) to 3.4e-11 Ry); real
nuclear_energy + madelung_constant via new lattice_sums.py (NaCl Madelung 8
digits, QE ewald 2e-8 Ha α-independent, exact utils::madelung port to 10
digits); Species/beta + paw/oc + Core/{n,l,ae_wfc} + exx_core_core attrs;
tabulated-vs-analytic <shape_function> hard check. CAVEAT: DEN/NLCC wiring
unexercised against real ABINIT files on this host (no WFK/POT/DEN assets)
— recheck in the B-tests cluster campaign. B3: native ex_cvij
(paw_onecenter.hpp compute_ex_cvij_from_core): −δ_lm Σ_c Σ_L (2l_c+1)·
3j000²·R^L, cumulative-Simpson-in-index quadrature (trapezoid loses 2.5e-5
rel — measured); sign/factor pinned on ABINIT m_pawdij (dijfock_cv =
ex_cvij, factor 1); validated vs a real atompaw Al-stringent
<exact_exchange_X_matrix> (~1e-7 rel, validate_ex_cvij.py, local
abinit-10.6.7 tree) + hydrogenic K(1s,2s)=16Z/729, K(1s,2p)=112Z/6561
(~3e-11) + new TEST_CASE ex_cvij_native; wired into read_vnl_h5 after the
Core/ read (detection: h5 ex_cvij → native-from-Core → none+warn);
abinit2coqui --corewf added (atompaw companion XMLs; ids matched by name or
document order). B4: schema_version=2 = HARTREE on disk (contract written
into notes/paw_implementation_plan.md §Schema contract): pw2coqui ÷e2 on
dion[_so]/Species dion/ae_vloc/vloc_ps/pp_local/scf_local/vxc[_with_nlcc];
abinit writers drop their ×2 and emit native Ha + miller_g as a G-SPHERE
(box-inscribed ⊇ ecutrho at boxcut≥2) instead of the full FFT box; readers
scale ×0.5 ONLY for schema_version<2 (new pp_schema.hpp::h5_pp_ry2ha; sites:
read_vnl_h5 dion + svsc/svloc, add_vxc) — in-memory convention is now
Hartree everywhere incl. ae_vloc/vloc_ps (compute_paw_static_D ×½ dropped);
ae_vloc/vloc_ps promoted to REQUIRED for schema≥2 (A5 deviation closed);
ijtoh verified at read per species against the QE init_us_1 sequential
upper-triangle packing (QE 7.4.1 source-verified; padding unconstrained: QE
−1, abinit 0). VALIDATION: fresh v2 Si conversion vs a synthesized legacy
twin (same data ×2 + schema 1) → one-body energy Tr[γ(T+Vpp)] BIT-IDENTICAL
(2.4215026009313685 Ha) through the full reader; writer-level v1 = 2×v2
bit-exact on dion/Species-dion/pp_local; pw2bgw/VSC/VLTOT plot-file paths
stay unconditionally ×0.5 (non-h5, always legacy — pseudopot_to_h5 emits
unstamped legacy files, documented). Fast suite green pre-B4 (13524/35 with
ex_cvij_native) and post-B4 (value-identical expected — see commit note).
Python deps installed on this host for the converter work: h5py, scipy,
netCDF4 (homebrew python 3.14). Remaining Workstream B: B-tests (same-system
Si PAW via both converters — needs ABINIT data → cluster campaign; closes
the [[project_si_exx_rpa_abinit_mf_anomaly]]).

Previous update: ∫V_loc·Q̂ SETTLED (user directive + derivation;
post-A5). The term is the frozen one-body ELECTROSTATIC coupling ∫n̂·V_loc
— neither exchange nor correlation — and is ALWAYS included. It is NOT in
dion: Eq. d0's −⟨Q̂|v_H[ñ_Zc]⟩ is the opposite-sign ONE-CENTER descreening
reference baked in at dataset generation precisely so the solid re-adds the
full periodic integral (standard USPP descreening; corroborated: the
density path, which adds ∫(V_loc+V_H)Q̂ on top of dion, matches QE deeq at
~1e-6 — impossible if dion contained the smooth-grid piece). Placement:
Eq. (h0) static D = static_h0_D() = Dnn_atom_static + ∫V_loc·Q̂ (lazily
cached in runtime_caches, MPI-collective first call; the ∫V·Q̂ block is
factored out of compute_paw_deeq_from_becsum as compute_int_VQ). Density
path unchanged (integrates V_loc+V_H itself; no double count) ⇒ the plan
identity H(n) ≡ H0 + Vhartree(n) now holds EXACTLY — new TEST_CASE
h0_plus_hartree_identity (nii+nij × NCPP/USPP/PAW/PAW-sym×2, ≤2.5e-13,
FFT-linearity residual; A-tests item i done; flags-off ≡ H0 still
bit-identical). CONSEQUENCE: the ERI-route USPP/PAW Fock (H0 + J − K, I7)
gains this previously-MISSING frozen term — post-A2 H0-based USPP/PAW total
energies move by it (the QE-deeq-era H0 carried it inside ∫V_eff·Q̂);
re-baseline any post-A2 USPP/PAW reference numbers on next campaign.
dft_eigenvalues unchanged (its assembly uses the density path, which always
had the term; USPP/PAW 0.749/0.790 Ha semicore failure still pending the
A-tests diagnostic rework). Fast suite green: 13499 assertions / 34 cases.
A5 notes (kept, read_vnl_h5 hardening):
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
(The former CAUTION about the ∫V_loc·Q̂ mismatch between the density path
and H0+add_Hartree is RESOLVED — see the settlement note at the top: the
term now lives in Eq. (h0)'s static_h0_D and the identity holds exactly.)
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
- [ ] A-tests: (i) nii≡nij≡no-density+Hartree DONE 2026-07-24 (h0_plus_hartree_identity + add_vpp_i5_alignment, after ∫V_loc·Q̂ settlement); (ii) sym≡nosym DONE via A3 (becsum_full_symm + sym fixture sections); remaining: ex_cvij factor-1 e_1e (vs ABINIT −0.521220 Ha si222); QE-eigenvalue diagnostic rework (USPP/PAW 0.749/0.790 Ha semicore)

Workstream B — converter parity
- [x] B1 QE: delete deeq/deeq_nc export; schema_version attribute — 2026-07-24 (079783b)
- [x] B2 ABINIT: ae_vloc/vloc_ps export; per-species proj_per_atom; real vxc; Ewald/madelung; beta + Core/; shape_function check — 2026-07-24 (445a0b3; DEN/NLCC wiring pending on-cluster recheck)
- [x] B3 native ex_cvij builder from Core/ae_wfc (Slater R^L + Gaunt²), validated vs ABINIT-XML ex_cvij (~1e-7 rel) + hydrogenic analytics — 2026-07-24 (f451083)
- [x] B4 schema standardization (schema_version=2 Ha on disk both converters; miller_g sphere on ABINIT side; ijtoh packing verified at read; ae_vloc/vloc_ps required for v2; contract in notes/paw_implementation_plan.md) — 2026-07-24
- [ ] B-tests: same-system Si PAW via both converters — dataset diff + e_1e/e_hf/e_rpa parity (closes ABINIT-mf anomaly); needs ABINIT WFK/POT/DEN assets (cluster); also recheck B2 DEN layout + psp8 NLCC-block convention there

Workstream C — augmentation-density modes
- [x] C1 single mode flag: pseudopot bool dropped — paw_shape_restored() derives from _exx_opts.vv_compensation (single source; setter kept for tests, delegating); deltaC/K_a inclusion derived from mode in both routes (direct v_x skip-on-shape; THC `_paw_onsite && !shape`); _paw_onsite documented DIAGNOSTIC-ONLY at both sites — 2026-07-25
- [x] C2 dense-sphere THC LL block (aug-aug Coulomb sum on the fft_grid_dim_aug inscribed-Gcut sphere, G-chunked, whenever rho_g doesn't cover it — ζ blocks stay on rho_g, exactly band-limited there; PSD Gram preserved; default configs bit-identical, branch self-disables); in_thc shape abort lifted; new TEST_CASE thc_shape_mode_vs_direct (THC-vs-direct shape V_x 7.6e-5 default / 6.8e-5 half-ecut; mode-difference cross-check 2.8e-6/6.3e-7; dense branch log-confirmed firing) — 2026-07-25. Si a=10.20 acceptance vs direct −1.6863 deferred to the cluster campaign (a10.20 mf not on this host)
- [x] C3 shared caches + Qfac knob: THC augment now uses psp.paw_aatab() + psp.paw_qrad_tabs(K_max, mode) (same dq=0.01/aug_lmax/per-species selection as its local build — one table set shared with direct v_x; THC's larger Kmax means direct reuses without rebuild); new paw_exx_options::qfac_cache_mb (toml `qfac_cache_mb`, default 256, 0=off) read live in get_or_build_qfac_pair_factor. dq unification itself landed in A4. Value-neutral (C4 mode energies bit-identical across the change) — 2026-07-25
- [ ] C4 physics validation: −1.316447 operator identity (GW vs HF); match both modes vs ABINIT

Workstream D — ERI/THC route equivalence (Eq. path-equiv)
- [ ] D1 audit thc.h/thc.icc/thc_reader_t vs A conventions (AE basis, identity overlap)
- [ ] D2 route-equivalence matrix-element tests (THC vs hamiltonian), both modes, NCPP/USPP/PAW × sym/nosym
- [ ] D3 THC OOM at N_aux≳10k + synthetic l=3 augmentation unit test
- [ ] D4 Cholesky+USPP/PAW hard-abort (no augmentation yet)

Workstream E — notes/documentation
- [ ] E1 author canonical D-matrix doc (notes/paw_dmatrix_scgw.tex)
- [ ] E2 corrections to stale notes (converter plan exx_X claim, GW-vs-HF reconciliation, k-weight line, STEP3, LaNiO3 retest-pending)
