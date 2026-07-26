# PAW/USPP consolidation plan — pointer + live status

The canonical plan is the LaTeX/PDF document (content lives there, not here):

- `notes/paw_dmatrix_cleanup_plan.tex`
- `notes/paw_dmatrix_cleanup_plan.pdf`

This file holds only the live STATUS checklist below (cheap to read/update;
printed into context at session start by a SessionStart hook).

## STATUS

REMAINING TO COMPLETE THE PHASE (2026-07-25 late, post-D session):
1. **AB direct-Hartree defect — RESOLVED (2026-07-25 late session)**: the
   +19.98 Ha V_H trace excess was the frozen-core density
   (ρ_core_AE/ρ_core_PS) injected into the DYNAMIC one-center radial
   Hartree (compute_paw_hartree_atom lp=0 core block) — a double count of
   the core–valence electrostatics already inside the static D⁰/dion (I2).
   Invisible on every QE fixture (empty core fields); activated on the AB
   semicore fixture by its exported core wfc. Root-caused by a trace
   decomposition against the numpy probe bilinears (smooth-bra 3.7615 ✓,
   dyn ∫V·Q 25.4505 ✓, radial oc 31.77 ✗ vs 11.793 = +19.98). Fix: core
   densities removed from the dynamic Hartree driver entirely (valence-only
   by contract; compute_paw_core_density deleted). Post-fix direct trace
   +41.0055 vs target +41.005; AB VH_VX strict both V_H (2e-3 abs = 1e-4
   rel, THC-truncation scale) and V_x; regression guard
   ab_direct_vh_trace_split pins all four pieces. Full [paw]~[slow] suite
   green (13552 assertions / 38 cases). NOTE: plan tex I3's parenthetical
   "frozen-core AE/PS core densities included" is SUPERSEDED by this
   resolution — amend the tex on next edit (I2 owns core electrostatics).
2. Cluster follow-through (one short session, mechanical):
   (a) properly reconvert mf_abinit.h5 / mf_v2.h5 (+ nocv) with the fixed
       converter (WFK + --pot --den --pawxml --corewf), verify ≡ the
       *_ylmfix.h5 row-flip stopgaps, retire the stopgaps; regenerate any
       pre-3956b45 ABINIT-sourced paw_aug numbers (eos_conv500-era exchange);
   (b) C2 acceptance closure: rerun the DIRECT-route exchange on the a10.20
       mf with the CURRENT binary at matched settings and compare to
       THC-shape −1.7349632 — the recorded "direct −1.6863" is not a usable
       reference (old binary; C1 static-D moved e_1e −73 mHa on that system).
3. A-tests residual: QE-eigenvalue diagnostic rework (USPP/PAW 0.749/0.790 Ha
   semicore) — last unchecked A sub-item.
4. Hardening debt exposed this week (add as explicit items):
   (a) np>1 regression test for the shm-builder path (set_H0 had zero
       multi-rank coverage — how the compute_int_VQ deadlock shipped);
   (b) bdft reader: clear error on orbital-less (pw2coqui) files and on
       missing Core attrs instead of the HDF5 crash cascade; document
       pw2coqui = qe-reader companion (NOT a standalone bdft file).
5. Workstream E: E1 canonical doc; E2 stale-notes sweep now also includes
   retracting the C4 "−47.6 mHa operator difference" anywhere quoted and
   the pw2coqui-role documentation.
6. Asset-gated: psp8-NLCC block-convention recheck (needs an NC psp8+NLCC
   asset; none on rusty or host).

Last updated: 2026-07-25 (late) — Workstream D LANDED (D1–D4, host session).
D1 audit (notes/paw_thc_d1_audit.md): normalization chain / q=0 kernel-zero +
analytic madelung·S·D·S / K_a placement / Hermiticity / AE-basis X rows all
verified consistent with anchors; three fixes: F1 smooth-only q=0 HEAD
VECTORS (_Chi_head/_Chi_bar_head never extended by augmentation — OOB reads
in GW Sigma_div_correction / g0_div_utils / embed_eri whenever PAW aug +
gygi; aug rows now carry conj(Ω·η^q(G=0)), χ̄ zero-padded = the still-valid
smooth LS representation of the G=0 plane wave); F2 augmented-ERI file
read-back aborted on rp!=Np (allow rp<=Np, recover the smooth/aug split,
hard-check head columns); F3 A4-deferred Pskna lift site now uses the shared
psp cache (contexts provably identical via make_pseudopot; explicit build
kept as fallback). D3: gather buffers in the aug q-loop G-chunked (8192 —
the N_aux≳10k OOM driver was (tile rows)×ngm); aug-stage per-rank memory
estimator + freemem warning (thc.icc's predates augmentation); synthetic
l=3 TEST_CASE paw_q_eval_synthetic_l3 (test_paw_radial.cpp): QE-ylmr2
recursion at L≤6 vs boost 3e-15, aainit ap vs quadrature Gaunt 5e-14, qrad
interp 4e-13, full evaluate_Q_IJ_at_K s⊗s/s⊗f/f⊗f vs independent assembly
6e-18 — the LaNiO3 f-channel chain is validated. D4: cholesky ctor +
read-only chol_reader_t hard-abort on USPP/PAW (smooth-only ERIs never ship
silently); diagnostic opt-in allow_smooth_only_aug_pp=true used by the
paw_aug exchange test. D2: NEW ABINIT-SOURCED FIXTURE
tests/unit_test_files/bdft/si_kp222_paw_abinit (Si_paw_pw_12el.xml+corewf,
LDA-PW semicore, 2x2x2 full-BZ nosym, generated with the host abinit build,
PROVENANCE.abi included; wired as bdft_si222_paw_ab). AB sections added to
thc_vs_direct_VH_VX (V_x STRICT-PASSES 3.7e-5 — the real_ylm bug class is
now gated), thc_shape_mode_vs_direct, paw_aug_q_eval_at_q0 (stored qgm ≡
runtime 2e-11), paw_onecenter_dDeeq_H_matches_deltaC_contraction (1e-7),
hartree_thc_paw_aug (THC ≡ ABINIT+one-center AE target to 4.9e-4 Ha after
fixing a TEST-HELPER bug: hartree/exchange energy helpers passed the SMOOTH
fft mesh while swfc_to_rho maps to the AUG mesh — silent coefficient drops
+ miller aliasing, hidden by every equal-mesh QE fixture). D2 CATCH: the
production direct-route V_H matrix is WRONG on this mf (see item 1 above)
— exactly the class of defect the AB fixture was mandated to expose.
Deferred: RPA/HF energy regression vs rusty Si_conv (cluster); sym×AB
fixture (QE covers sym; AB adds the converter dimension nosym-first).

Previous update: 2026-07-25 (pin-down, commit 3956b45) — the B-tests 51.8 mHa
exchange gap is ROOT-CAUSED AND FIXED: abinit2coqui real_ylm wrote plain
real harmonics into projector_k while QE ylmr2 / CoQui aainit / the
converter's own qfuncl carry the (−1)^m Condon–Shortley sign on odd-m
components (l=1 −x,−y; l=2 −xz,−yz) — the l=1 m=±1 projector rows were
exactly −1×. Channel-diagonal + T_d-site-symmetric quantities are immune
(dion, Hartree becsum, onsite, same-k nonlocal — precisely the µHa set);
off-diagonal (k,k−q) pair-density augmentation decoheres (aug E_X term
+19.2 vs +69.8 mHa). Established by elimination: noaug collapse
51.8→1.2 mHa; matched 48³ aug grids move 51 µHa; +0.5-k-list rep test
3.4 µHa; stored-projector ratio ±1 per nhtolm row. POST-FIX PARITY
(row-flip validation): ΔE_X 87 µHa, ΔRPA 23 µHa, ΔHF 128 µHa; remaining
Δe_1e = 4 occ × 35.71 mHa uniform vloc-G=0 gauge constant (bands agree
~3 µHa beyond it) — B-tests CLOSED. C4 REVISED on fixed mf:
E_X(shape) = −1.3175731 vs converged GW Σx −1.31747 → **0.10 mHa** (the
earlier −47.6 mHa "operator difference" was this bug); moment −1.3178007
(shape−moment now −0.23 mHa); Δe_1e(cv) 4.8 µHa unchanged. WARNING: every
pre-3956b45 ABINIT-sourced paw_aug result is contaminated — reconvert
(mf_v2/mf_abinit row-flip copies *_ylmfix.h5 are stopgaps; regenerate
with the fixed converter before publication-grade use).

Previous update: 2026-07-25 (cluster harvest) — campaign landed after fixing a
np>1 deadlock (fff6d4e): static_h0_D's lazy build ran compute_int_VQ's
collectives on the GLOBAL comm while shm builders (set_H0) call on node
roots only; compute_int_VQ is now purely rank-local (contract documented in
paw_onecenter.hpp/pseudopot.h). Campaign results, all runs <2 min on ccq:
- C4 CLOSED: Δe_1e(cv) = −0.5212248 vs ABINIT cv −0.521220 (4.8 µHa);
  E_X(shape) −1.36508 vs converged GW Σx (Arnaud, ISZ off, pawoptosc 1)
  −1.31747 → measured Fock-vs-Arnaud operator difference −47.6 mHa at
  matched aug cutoff. Mode invariants verified on-cluster: e_1e identical
  moment/shape; E_X/RPA identical moment/nocv (cv strictly in e_1e).
- C2 acceptance number: THC-shape E_X(vv) −1.7349632 (a10.20, 16 kpt);
  shape−moment +3.19 mHa (si222: +3.38 mHa, consistent). OPEN: reconcile
  vs the recorded direct −1.6863 (operator content of that reference; note
  old→new binary moved e_1e −73 mHa on this system via C1 static-D).
- B-tests level b/c (both sides through THC/RPA, nosym, nbnd 20/20;
  QE via qe reader + si222.coqui.h5, AB via bdft): Δ(QE−AB) e_1e
  +141.73 mHa, E_X +51.77 mHa, RPA +12.27 mHa, BUT Hartree+nuc (HF−E_X)
  +41 µHa. Split runs (nocv/no-onsite, both sides) ATTRIBUTE these:
  cv agrees to 39 µHa (B3 native ex_cvij ≡ atompaw X-matrix end-to-end);
  onsite E_X agrees to 7.5 µHa; Δe_1e is entirely the cv-free
  one-electron gauge residual (Δv̄ beyond-projector part, ≫ the ~mHa
  level-a guess); ΔE_X/ΔRPA live in the SMOOTH/aug pair-density exchange
  (occupied wfcs/aug fns agree 1e-8, E_H exact ⇒ suspect off-diagonal
  k-pair augmentation: k-representative −0.5 vs +0.5 conventions /
  qe-reader vs bdft BZ maps). This narrows the ABINIT-mf anomaly; NOT yet
  closed.
- Reader/schema findings: pw2coqui h5 = qe-reader COMPANION (no orbitals;
  bdft cannot read it — needs a clear error, today an HDF5 crash cascade);
  qe reader consumes a stored BZ whenever can_init_from_h5 passes (a
  grafted BZ SEGV'd read_vnl_h5 — conventions must be the reader's own);
  b_tests QE h5 patched with Core@ncore_orbitals + exx_core_core attrs
  (earlier Core injection was attr-less).

Previous update: 2026-07-25 — Workstream C COMPLETE (C1–C4 host-side, four
commits, one per item; see the C checklist below for details). C1: single
mode flag (vv_compensation the only source; deltaC/K_a strictly
mode-derived; _paw_onsite diagnostic-only). C2: THC LL block on the dense
augmentation sphere whenever rho_g doesn't cover it → shape mode usable in
THC, in_thc abort lifted; TEST_CASE thc_shape_mode_vs_direct (mode-difference
THC≡direct at 2.8e-6/6.3e-7). C3: THC shares psp.paw_aatab()/paw_qrad_tabs()
with direct v_x; qfac_cache_mb knob live. C4: −1.316447 = GW/Arnaud settled,
HF side matched (kernel+energy, icutcoul µHa); current-code mode-energy
baselines in TEST_CASE vexchange_mode_energies; shape-vs-GW number = cluster
campaign at matched aug cutoff. Fast suite 13532/36 green (value-identical +
the new C2 test). DEFERRED to cluster (with B-tests): Si a=10.20 THC-shape
acceptance vs direct −1.6863; ABINIT GW-side regeneration on the cmp mf.

Previous update: 2026-07-24 (late) — Workstream B COMPLETE (B1–B4, four
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
- [ ] A-tests: (i) nii≡nij≡no-density+Hartree DONE 2026-07-24 (h0_plus_hartree_identity + add_vpp_i5_alignment, after ∫V_loc·Q̂ settlement); (ii) sym≡nosym DONE via A3 (becsum_full_symm + sym fixture sections); ex_cvij factor-1 e_1e DONE 2026-07-25 (cluster c4 nocv: Δe_1e(cv) −0.5212248 vs ABINIT −0.521220 → 4.8 µHa); remaining: QE-eigenvalue diagnostic rework (USPP/PAW 0.749/0.790 Ha semicore)

Workstream B — converter parity
- [x] B1 QE: delete deeq/deeq_nc export; schema_version attribute — 2026-07-24 (079783b)
- [x] B2 ABINIT: ae_vloc/vloc_ps export; per-species proj_per_atom; real vxc; Ewald/madelung; beta + Core/; shape_function check — 2026-07-24 (445a0b3; DEN/NLCC wiring pending on-cluster recheck)
- [x] B3 native ex_cvij builder from Core/ae_wfc (Slater R^L + Gaunt²), validated vs ABINIT-XML ex_cvij (~1e-7 rel) + hydrogenic analytics — 2026-07-24 (f451083)
- [x] B4 schema standardization (schema_version=2 Ha on disk both converters; miller_g sphere on ABINIT side; ijtoh packing verified at read; ae_vloc/vloc_ps required for v2; contract in notes/paw_implementation_plan.md) — 2026-07-24
- [ ] B-tests: same-system Si PAW via both converters — dataset diff + e_1e/e_hf/e_rpa parity (closes ABINIT-mf anomaly); needs ABINIT WFK/POT/DEN assets (cluster); also recheck B2 DEN layout + psp8 NLCC-block convention there. 2026-07-25 cluster: level a+b+c DONE (numbers + nocv/no-onsite attribution in STATUS header); DEN wiring exercised on a real DEN; psp8-NLCC still unchecked (no NC asset). CLOSED 2026-07-25 (pin-down): ΔE_X = real_ylm odd-m sign bug (3956b45, post-fix 87 µHa); Δe_1e = benign vloc-G=0 gauge constant (35.71 mHa/band uniform, bands agree ~3 µHa beyond it)

Workstream C — augmentation-density modes
- [x] C1 single mode flag: pseudopot bool dropped — paw_shape_restored() derives from _exx_opts.vv_compensation (single source; setter kept for tests, delegating); deltaC/K_a inclusion derived from mode in both routes (direct v_x skip-on-shape; THC `_paw_onsite && !shape`); _paw_onsite documented DIAGNOSTIC-ONLY at both sites — 2026-07-25
- [x] C2 dense-sphere THC LL block (aug-aug Coulomb sum on the fft_grid_dim_aug inscribed-Gcut sphere, G-chunked, whenever rho_g doesn't cover it — ζ blocks stay on rho_g, exactly band-limited there; PSD Gram preserved; default configs bit-identical, branch self-disables); in_thc shape abort lifted; new TEST_CASE thc_shape_mode_vs_direct (THC-vs-direct shape V_x 7.6e-5 default / 6.8e-5 half-ecut; mode-difference cross-check 2.8e-6/6.3e-7; dense branch log-confirmed firing) — 2026-07-25. Si a=10.20 acceptance vs direct −1.6863 deferred to the cluster campaign (a10.20 mf not on this host). 2026-07-25 cluster: THC-shape E_X(vv) −1.7349632 at 16 kpt (shape−moment +3.19 mHa, matches si222 +3.38 mHa); reconciliation vs the −1.6863 direct reference still open (operator content of that number)
- [x] C3 shared caches + Qfac knob: THC augment now uses psp.paw_aatab() + psp.paw_qrad_tabs(K_max, mode) (same dq=0.01/aug_lmax/per-species selection as its local build — one table set shared with direct v_x; THC's larger Kmax means direct reuses without rebuild); new paw_exx_options::qfac_cache_mb (toml `qfac_cache_mb`, default 256, 0=off) read live in get_or_build_qfac_pair_factor. dq unification itself landed in A4. Value-neutral (C4 mode energies bit-identical across the change) — 2026-07-25
- [x] C4 physics validation (host-side portion): operator identity SETTLED — −1.316447 is the GW Sigma_x (Arnaud) operator, NOT HF Fock (2026-07-21 instrumented-ABINIT work, notes/paw_article_results/abinit_exchange_gw_vs_hybrid.md, updated). HF-side match DONE: deltaC ≡ ABINIT eijkl 5.5e-5 rel + ex_cvij machine-identical (kernels), onsite vv+cv energies identical, smooth residual closed via fock_icutcoul=3 (µHa). Current-code baselines recorded via new [slow] TEST_CASE vexchange_mode_energies (direct dense-grid, ignore_g0): lih222_paw_hf −1.64406506/−1.64395244 (moment+deltaC / shape, split +1.13e-4); local si222_paw −1.31194760/−1.31187642 (split +7.1e-5 — this fixture has fft_mesh_aug==fft_mesh=36³: a coarse aug sphere truncates both modes equally, AND it is a different cell from the rusty cmp mf, so NOT comparable to −1.316447). CLUSTER PORTION DONE 2026-07-25: ABINIT GW Σx regenerated (pawoptosc=1, ISZ off; ecutsigx 25→−1.316447 exact provenance match, converged −1.31747); Δe_1e(cv) 4.8 µHa vs ABINIT −0.521220. REVISED post-3956b45 (the mf carried the real_ylm bug): E_X(shape) = −1.3175731 on the fixed mf → agrees with converged GW Σx to 0.10 mHa (the earlier −47.6 mHa was the converter bug, NOT an operator difference); moment −1.3178007, shape−moment −0.23 mHa. C4 CLOSED — 2026-07-25

Workstream D — ERI/THC route equivalence (Eq. path-equiv)
- [x] D1 audit thc.h/thc.icc/thc_reader_t vs A conventions — verified-consistent ledger + 3 fixes (aug q=0 head vectors; augmented-ERI read-back; Pskna lift → shared cache) in notes/paw_thc_d1_audit.md — 2026-07-25
- [x] D2 route-equivalence matrix-element tests: AB fixture bdft_si222_paw_ab added + AB sections in VH_VX/shape-mode/q_eval/dDeeq_H/hartree-energy tests; V_x route-equivalence STRICT on QE+AB; energy-helper aug-mesh bug fixed. CAUGHT an open direct-route V_H defect on the AB mf (STATUS item 1; strict_VH off there until root-caused). Deferred: Si_conv RPA/HF regression (cluster), sym×AB — 2026-07-25
- [x] D3 aug-build gather buffers G-chunked (the N_aux≳10k OOM driver) + aug-stage memory estimator + synthetic l=3 unit test (paw_q_eval_synthetic_l3, machine-precision) — 2026-07-25
- [x] D4 Cholesky+USPP/PAW hard-abort (builder ctor + read-only reader; diagnostic opt-in allow_smooth_only_aug_pp for the smooth-reference test) — 2026-07-25

Workstream E — notes/documentation
- [ ] E1 author canonical D-matrix doc (notes/paw_dmatrix_scgw.tex)
- [ ] E2 corrections to stale notes (converter plan exx_X claim, GW-vs-HF reconciliation, k-weight line, STEP3, LaNiO3 retest-pending; 2026-07-25 additions: retract the C4 "−47.6 mHa Fock-vs-Arnaud operator difference" wherever quoted — it was the real_ylm bug, true agreement 0.10 mHa; document pw2coqui = qe-reader companion file, not standalone bdft)
