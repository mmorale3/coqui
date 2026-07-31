# PAW/USPP consolidation plan — pointer + live status

The canonical plan is the LaTeX/PDF document (content lives there, not here):

- `notes/paw_dmatrix_cleanup_plan.tex`
- `notes/paw_dmatrix_cleanup_plan.pdf`

This file holds only the live STATUS checklist below (cheap to read/update;
printed into context at session start by a SessionStart hook).

## OPEN ITEMS

Physics / correctness
1. ~~THC-route K_a deficit~~ **RESOLVED 2026-07-30 — there was no deficit.**
   See the STATUS block below and `eos_exchange_ledger.md` §3h: the reference
   was 12% too large, not CoQui 12% short.
1b. ~~volume-dependent ISDF systematic~~ **RETRACTED — it was an artefact of
   the probe.** The nbnd scan that produced it ran at `beta = 100 /
   iaft_prec = low`, settings that put E_x 11.5 mHa from the exact answer, i.e.
   19x the 0.61 mHa effect being measured; differences between nbnd values
   inside that regime say nothing about production. Checked against the exact
   direct route on the production mf: **the EOS THC exchange (nbnd 500,
   beta 1000) is exact to 0.58 uHa**, and a0 = 10.2780 stands with no re-run
   needed. The direct route also confirms K_a = -6.61389 mHa against the
   corrected ABINIT reference -6.59720 (ratio 1.0025) with an exact method, and
   shape vs moment agree to 35 uHa at the production geometry.
   Lesson kept in eos_exchange_ledger.md §3h: vary one thing at a time from the
   PRODUCTION point, never from a cheap surrogate.
3. psp8-NLCC end-to-end exercise remains asset-gated (the convention risk it
   guarded is closed by source inspection).

Test / infrastructure debt
4. **The external one-centre anchor does not run pre-commit.**
   `vexchange_mode_energies` is `[slow]` (~8 min for the 12-electron section,
   three full direct Vexchange builds), so `[paw]~[slow]` does not cover it.
   Either promote a cheaper variant or add it to a nightly/CI target — as it
   stands the guard against the exact class of bug this campaign chased is
   opt-in.
5. **Regenerate `tests/unit_test_files/bdft/si_kp222_paw_abinit`.** It was
   built with the pre-sqrt(4pi) converter, so its `pp_local` / `dion` /
   `vloc_ps` are mis-normalized; re-baseline any pinned `e_1e`. `E_H`/`E_x`
   tests (including the new anchor) are unaffected — they use none of those.
6. ~~The anchor constant is not re-derivable~~ DONE — `emul_pawdijfock.py`
   reproduces ABINIT's `efockdc` from the dump to <=2e-15 and prints the
   physical value, i.e. the test's `-6.658384` constant.

Campaign / data hygiene
7. The rusty tree is behind this branch; sync before the next production run.
8. The `exx_split` runs used `beta=100 / iaft_prec=low` to make the RPA cheap,
   which moves the KS density matrix and shifts absolute E_x by +11.5 mHa.
   Their on/off DIFFERENCES are valid; their absolute values are not. Rerun
   with the EOS settings if absolute smooth+aug numbers are wanted.
9. Regenerate any pre-`3956b45` ABINIT-sourced `paw_aug` numbers before
   quoting them (publication-gated).

## STATUS

**EXCHANGE ROW FULLY CLOSED (2026-07-30, second half).** The 0.83 mHa left by
the `nsppol=1` finding below was ALSO the reference, not CoQui — a *second*
ABINIT defect in the same routine. `pawinit` (`m_paw_init.F90:610`, `k1min=klmn`)
fills only the `klmn<=klmn1` triangle of `eijkl`; `pawdijfock`
(`m_pawdij.F90:1223`) then indexes it as `eijkl(pack(i,l), pack(k,j))` without
ordering the two pair indices, so every term that lands in the unfilled half
reads back as a structural zero.

The THC route was cleared first, three ways: K_a is **thresh-independent to
1e-10 Ha** (−0.0065871560 at thresh 1e-5 / 4928 interpolating points vs
−0.0065871559 at 1e-4 / 4300 — E_x itself moves 0.74 µHa for a 15% rank change),
the one-centre ISDF block is uncompressed at `paw_isdf_tol = 1e-8`
("kept nlambda=324 (full-rank cap=324)"), and the interpolating-point set is
identical with K_a on and off. So the on/off differencing was clean and the
deficit had to be in the reference.

Emulating `pawdijfock`'s exact indexing in python reproduces ABINIT's printed
`efockdc` to <=2e-15 on six runs, which is what licenses comparing it against
the symmetrised tensor. The loss depends on how many off-diagonal `rho_ij` are
populated, NOT on the dataset alone: **0% at Γ, 0.6% for the s,p 12-electron
dataset, 12% for `jth_with_d`** (the d channels populate exactly the pairs whose
partners fall in the unfilled half).

CoQui matches the corrected reference: direct-route K_a = −6.59819 mHa against
−6.57272 mHa on a local `jth_with_d` 2x2x2 state converted from that very ABINIT
run (**0.4%**), and THC K_a = −6.5872 mHa against −6.5976 mHa at the production
geometry (**0.15%**). The triangle factor was measured at a = 10.05/10.25/10.55
(0.87601/0.88538/0.89943) on local 4x4x4 reruns that reproduce the production
ledger's `efockdc` exactly. With it the exchange row becomes a **constant
−0.021 ± 0.002 mHa with slope +0.008 mHa/Bohr — a0 impact −0.0001 Bohr**. A
constant cannot move an EOS: the row is closed and `a0 = 10.2780` stands.

The +0.028 Bohr this campaign started from is now fully accounted for: ~76% the
`nsppol=1` spin double count, the rest the `eijkl` triangle, both ABINIT-side.

**EXCHANGE-ROW DISCREPANCY RESOLVED — IT WAS ABINIT (2026-07-30).** The last
CoQui-vs-ABINIT gap (the ~8 mHa exchange row with a −2.14 mHa/Bohr slope, held
responsible for +0.028 Bohr of a0) is **an ABINIT `nsppol=1` double count, not a
CoQui error. No CoQui physics changed; the published EOS a0 = 10.2780,
B0 = 100.3 GPa, B' = 4.17 stands unaltered.**

Split of the exchange row (`eos_exchange_ledger.md` §3g), all measured:
core-valence agrees to **6 µHa at all six volumes**; smooth+compensation agrees
with ABINIT's `fock0` to **0.9 mHa**; the entire residual is the one-centre
valence-valence term. `efock` is NOT the cv term — `m_paw_denpot.F90:1094-1097`
makes `efockdc` the one-centre vv exchange and `efock − efockdc` the cv part, and
`fock0` already contains n̂ (`m_fock_getghc.F90:663`).

The one-centre term was then cleared on both halves: CoQui's `deltaC` ≡ ABINIT's
`pawtab%eijkl` to **1e-5** (compared via rotation invariants — the two codes'
real-Ylm conventions differ, so element-wise comparison is meaningless), and
`v_x`'s one-centre block reproduces its own closed form
`−Σ_a Σ_IJKL D_IL D_KJ ΔC` to **ratio 1.000000**. What was left was a clean
factor of 2 in the density matrix: ABINIT's `rhoij` = 2 × CoQui's `becsum`, and
`rhoij` is provably the spin-summed matrix (`½Σρρ K` = `eh2/2`, the value `epaw`
uses, ratio 0.50000).

**ARBITER — the same non-magnetic state run `nsppol=1` vs `nsppol=2`:**
`e1t10`, `eh2`, `fock0` and the core-valence term are identical to 1e-9; only
`efockdc` changes, by **exactly 2.000000** (−13.394629907 → −6.697314956). Only
the term quadratic in ρ doubles ⇒ `pawdijfock` (`m_pawdij.F90:1223`,
`nsp = pawrhoij%nsppol`) contracts the spin-summed `rhoij` twice at `nsppol=1`.
CoQui's K_a = −6.658166 vs the nsppol=2 value −6.697315 → 0.6%; that last 0.6%
is the `eijkl` triangle (this fixture's loss), and against the fully corrected
−6.658384 CoQui is exact to 3e-5. Report upstream.

Consequences: the pre-registered §3c prediction a0 = 10.2501 is **superseded**
(it used the doubled exchange). The 0.83 mHa that remained at this point was
initially attributed to a THC-route K_a deficit; the block above supersedes that
— it is the `eijkl` triangle, also ABINIT-side, and the exchange row closes to a
constant −0.021 mHa.

Testing invariants this exposed:
- **The one-centre exchange is the only genuine rank-4 PAW contraction, and it
  had no external anchor.** Every previously validated on-site quantity
  (`ex_cvij`, `dij0`, one-centre Hartree) contracts the density matrix with a
  matrix that is spherical within each (l,n) shell — rank-2 and invariant under
  both the real-Ylm convention and the rank-4 structure. `vx_onecenter_vs_thc_Ka`
  feeds the SAME `deltaC` to both sides, so it is blind to magnitude by
  construction. Now anchored: `vexchange_mode_energies` pins K_a to instrumented
  ABINIT (nsppol=2) at 1%.
- **Cross-code references need an internal-consistency arbiter.** Reading the
  reference code's source was not enough to settle the spin convention; running
  the SAME physical state two ways (`nsppol` 1 vs 2) settled it in one shot, by
  showing which terms moved and which did not.
- New diagnostics kept: `pseudopot::set_paw_onsite_diag` (direct-route twin of
  the THC `paw_onsite`, so K_a = E_x(on) − E_x(off) is measurable),
  `notes/paw_article_results/cmp_onecenter_kernel.py`, and the
  `ABI_DUMP_PAWKERNEL` site in `abinit_ene_instr.py` (dumps `eijkl`, `ex_cvij`,
  `indlmn`, `rhoij`).

**EOS ONE-BODY DEFECT RESOLVED (2026-07-29, later session).** The Si PAW EXX+RPA
EOS had no minimum even after the `V_LL` fix. It was **not** exchange:
`abinit2coqui` omitted the PAW-XML L=0 `1/sqrt(4pi)` on
`blochl_local_ionic_potential` and `zero_potential` (ABINIT applies it at
`m_pawpsp.F90:3730/3767`) and compensated with a spurious frozen-core Hartree, so
`alpha_Z` was 25.132 Ha.Bohr^3/atom against ABINIT's `epsatm` = 8.858 and `dij0`
was off by up to 12 Ha. Post-fix `alpha_Z` = 8.858488 vs 8.858424 (5 ppm),
`Qtail` = 4.000000 exactly. Full write-up + reusable cross-code ledger:
`notes/paw_article_results/eos_exchange_ledger.md`. **EOS REFIT (all 6 volumes):
a0 = 10.2780 Bohr, B0 = 100.3 GPa, B' = 4.17, BM resid 0.001 mHa** — against NC
through the same pipeline 10.2259/101.1/4.08 and VASP RPA@PBE 10.244/98, versus NO
MINIMUM AT ALL before the fix. Judge it on B0/B' (within 0.8 GPa of NC, 2.3 of
VASP); the +0.034 Bohr a0 offset vs VASP is quantitatively accounted for by the
residual 2.4 mHa/Bohr CoQui-vs-ABINIT exchange-row slope (predicts +0.031), which
is the ISDF/onsite-exchange difference and the next open item — NOT this defect.

Three testing invariants this exposed:
- **A tail check cannot validate `vhtnzc`.** Its `-zval/r` asymptote comes entirely
  from the poisson term, so it is insensitive to a `sqrt(4pi)` error inside the
  augmentation sphere (131% wrong there while the tail read correct). The B2
  validation checked exactly that tail.
- **"Benign constant" is the wrong verdict for anything going as 1/Omega.** This was
  recorded as a benign vloc-G=0 gauge constant because it is constant at fixed
  volume — which is precisely a slope error across a volume series. Validate an
  EOS-relevant quantity at >= 2 volumes against an external reference.
- **Cross-code term-by-term ledgers need a control that is known-good.** The NC
  series (psp8 reader, immune to this bug) agrees with instrumented ABINIT to
  ~5 uHa per term at every volume; running it first proved the mapping, converter
  and both divergence conventions before the PAW numbers were trusted.

Guards added: hard error if the local-potential tail != zval; hard error if the two
independent `v_H[n~_Zc]` routes disagree; regression test + negative control in
`validate_b2.py synth`. OWED: regenerate the checked-in AB fixture
`tests/unit_test_files/bdft/si_kp222_paw_abinit` (built with the buggy converter,
so its `pp_local`/`dion`/`vloc_ps` are mis-normalized) and re-baseline any pinned
`e_1e` there; `E_H`/`E_x` AB tests are unaffected (they use neither).

**RPA INSTABILITY RESOLVED (2026-07-29).** The Si PAW RPA blow-up at large band
count was `V_LL` conjugating the FIRST index of Z instead of the second, storing
its transpose (commit 86ace47). EOS spread 107.16 -> 3.75 mHa against ABINIT's
3.57. Follow-up 44c79e9 replaced the q=0-only augmentation-channel ranking with
a max-over-q-mesh criterion and made `paw_isdf_tol` RELATIVE rather than
absolute. Details: notes/paw_article_results/rpa_instability_localization.md
§11-§15.

Two testing invariants this exposed, which apply to ANY future change to the
ERI representation:
- **Off-diagonal ERIs must be tested.** The entire suite probed only diagonal
  (vc|cv); the diagonal of a Hermitian matrix is real, so a transposed block is
  invisible there, to a Hermiticity check, and to an oscillator comparison.
  `Tr(Pi*Z)` uses only the diagonal; `ln|det(I-Pi*Z)|` uses the whole matrix.
- **Validate an EOS on B0/B', not a0.** Pre-fix a0 was 10.2293 Bohr — within
  0.015 of the reference — while B0 was 45 GPa against 98 and B' 1.63 against
  ~4, at a 0.002 mHa Birch-Murnaghan residual. A smooth-in-volume error shifts
  the curve far more than it tilts it, and goodness of fit says nothing.
- **LiH fixtures cannot see this class of bug** (eta is nearly real there); it
  needs a multi-atom d-channel system such as Si jth_with_d.

REMAINING: harvest + fit the 6-volume Si EXX+RPA EOS
(`~/ceph/CoQui/abinit/eos_exxrpa`, tooling in notes/paw_article_results/
eos_exxrpa_{harvest.sh,fit.py}); a10.05 done at E_c = -0.437371. Regenerate any
EOS/GW number produced before 86ace47. Sync the rusty tree past 44c79e9 once
the series finishes (not mid-campaign — it would make volumes inconsistent,
though at paw_isdf_tol=1e-8 the selection change is a no-op).

REMAINING TO COMPLETE THE PHASE (2026-07-26 session):
- **Static-route selection COMPLETE (2026-07-26, phases 0-4)**: the
  "future set_H route-selection project" (I5/I7, workstream-D preamble) is
  done — new `[interaction.hamilt]` interaction type fills the static ERI
  slots (interaction_hf/hartree/exchange; hard-rejected in the dynamic
  slot); hf_t direct-route overload via hamilt::Vhartree/Vexchange with
  the shared route-free gygi HF_K_correction; in-SCF I7 acceptance at THC
  scale on PAW/USPP/NCPP + per-term mixing + GW hf-slot (dynamics
  bit-identical); View-2 general-nij symmetry lift in v_x(nij) (derivation
  notes/static_route_nij_symmetry_note.md) — sym meshes fully supported in
  the direct route (the "nij becsum still open" note was stale:
  compute_becsum_full_symm already handled general nij; tests now certify
  it). Plan: notes/static_route_selection_plan.md; commits b966bae,
  1692c98, 9ce5018, +phase-4.
- **Converter-audit implementation LANDED (2026-07-26, schema 3)**: both
  converters stop writing dead-at-read data (species kbeta/qqq/dion/qfunc/
  q_with_l/nqf/nqlc/lmax/lmax_rho/zp/jjj/nhtoj attrs+datasets, paw pfunc/
  ptfunc/augmom/oc/augshape/pfunc_rel/aewfc_rel, core-zeroed vxc; `beta`
  KEPT as the native-projector enabler); /Orbitals@ecutrho PINNED to
  HARTREE at schema 3 (pw2coqui ÷e2; abinit2coqui writes the exact
  inscribed-sphere cutoff replacing the 2×/4× heuristic; qe read_h5 scales
  ×0.5 below v3 — the raw-Ry attr silently inflated the h5-init dense
  sphere 2×); AB PAW path OMITS vxc_with_nlcc without --den (was silent
  zeros); pw2coqui lspinorbit_loc moved inside the ionode guard; stale "Ry
  on disk" AB comments fixed. VALIDATION: AB reconversion from the D2
  WFK/POT/DEN ≡ old fixture on every kept dataset (deltaC 1e-13 jitter;
  exactly the 15 intended removals; ecutrho 68.255→121.509 = 36³ inscribed
  sphere) and the checked-in bdft fixture REFRESHED to schema 3; fresh
  pw2coqui.x LiH-PAW conversion verified (schema 3, ecutrho 400 Ry→200 Ha
  ≡ XML, dead data absent, kept-set exact). notes/converter_h5_contract.md
  + the two write inventories committed as the normative contract.
- **STATUS item 4 hardening LANDED (2026-07-26)**: (a) TEST_CASE
  set_h0_shm ([shm_h0]) — shm set_H0 ≡ distributed H0 on
  NCPP/USPP/PAW/AB-split-mesh, machine precision at np=1/2, plus a
  DEDICATED np=2 ctest entry test_hamiltonian_np2_shm (TIMEOUT 900) so the
  compute_int_VQ deadlock class fails instead of hanging and never again
  ships unexercised (CTEST_NPROC defaults to 1 — that was the hole);
  (b) bdft reader names orbital-less pw2coqui companion files with a clear
  error; Core/ without ncore_orbitals attr errors with the converter fix
  named; add_vxc errors clearly when vxc_with_nlcc is absent (schema-3 AB
  no-DEN files omit it); miller_g out-of-box silent `continue` in
  compute_rho_aug_density_r / compute_int_VQ is now a HARD ABORT (dropped
  Fourier coefficients = the D2 energy-helper bug class).
- **A-tests residual CLOSED (2026-07-26)**: dft_eigenvalues reworked per
  plan A-tests(iv) — the diagnostic now assembles D_stat + D^H[n_QE] +
  ∫V_xc·Q̂ explicitly (read_vxc_h5 aug-mesh V_xc → compute_int_VQ →
  Pskna contraction, test-side only). USPP 0.749 Ha → **2.9e-6 STRICT
  PASS** (operator complete — closes the semicore mystery); PAW 0.790 Ha
  → 4.13e-2, concentrated on Li 1s = the radial one-center XC of QE's
  ddd_paw that CoQui deliberately never assembles (no DFT XC in D);
  pinned TWO-SIDED [3.5e-2, 4.8e-2] so silently gaining one-center XC (an
  I2/I3 violation) fails too. NCPP untouched 5.8e-7. Workstream A fully
  checked.
- **Workstream E LANDED (2026-07-26)**: E1 canonical doc authored
  (notes/paw_dmatrix_scgw.tex/.pdf — invariants with derivations: becsum
  normalization, D⁰ per backend, ∫V_loc·Q̂ descreening settlement,
  ex_cvij factor-1, valence-only dynamic Hartree incl. the D2 contract,
  moment/shape operator identities with the Fock-vs-Arnaud measurements,
  units/schema ledger, validation anchors). E2 sweep: plan tex I3
  parenthetical amended (valence-only supersedes "core densities
  included"); −47.6 mHa retracted at every quote (plan.md historical
  entry, gw_vs_hybrid addendum, onsite-analysis addendum); converter-plan
  exx_X=deltaC claim corrected (it is ex_cvij; banner + inline);
  STEP3 schema block marked superseded by the contract note;
  deeq-scaling k-weight limitation marked resolved-by-A3;
  LaNiO3 note marked RETEST-PENDING with the post-June fix list;
  pw2coqui role documented in the source header + contract + bdft error.

REMAINING (older numbering, 2026-07-25 post-D session):
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
2. Cluster follow-through:
   (a) **RECONVERSIONS + RETIREMENT DONE 2026-07-26**: mf_abinit.h5 and
       mf_v2.h5 (+ nocv) reconverted on rusty with the schema-3 converter
       (sources/XMLs fingerprint-identified: b_tests = pair_gen/vB pair +
       corewf; c4 = Psdj_paw_pbe_std/Si.xml; see
       ~/ceph/CoQui/abinit/cmp/si222/RECONVERT_2026-07-26.md). VERIFIED
       dataset-identical to the *_ylmfix row-flip stopgaps (161/157 shared
       paths; only schema_version 2→3 + ecutrho 115.354→218.566 differ;
       old nocv originals confirmed still bug-carrying). Stopgaps retired
       (.retired), pre-fix mfs kept as .pre3956b45_bak; canonical names
       now = schema-3 files. **VALIDATION RERUNS DONE (chain 6676841-44,
       binary @02c1fbe)**: b_tests parity on the reconverted mfs IMPROVED
       — Δ(QE−AB) E_X +39.7 µHa (was 87), RPA −1.6 µHa (was 23), HF
       +44.9 µHa (was 128), Δe_1e +141.74 mHa = the known benign
       vloc-G=0 gauge constant unchanged (QE side now reads the correct
       115.354 Ha ecutrho through the schema gate; AB side on the
       218.566 Ha inscribed sphere). c4 modes on reconverted mf_v2
       REPRODUCE the ylmfix-run values to ~1e-10: E_X(shape) −1.3175731
       (0.10 mHa vs converged GW Σx), moment −1.3178007, nocv ≡ moment
       to 5e-12 — the stopgap equivalence is confirmed at the ENERGY
       level too. Remaining: regeneration of OTHER pre-3956b45
       ABINIT-sourced paw_aug numbers before quoting (eos_conv500-era
       exchange — publication-gated).
   (b) **C2 ACCEPTANCE CLOSED 2026-07-26** (job 6676844, direct
       dense-grid vexchange_mode_energies on the a10.20 mf, np=16,
       ignore_g0 = the same div convention as the THC run):
       E_X(shape, direct) = −1.73509011 vs THC-shape −1.7349632 →
       **Δ = 0.127 mHa = the THC/ISDF-truncation scale** (thresh 1e-5;
       same order as the D2 route-equivalence tolerances) — Eq.
       (path-equiv) PASSES on the C2 system. Also: E_X(moment, direct)
       −1.73825524 (vs Jul-17 THC-moment −1.7381526: 0.10 mHa), direct
       mode split +3.165 mHa ≡ THC +3.19 mHa. The recorded "direct
       −1.6863" is RETIRED as a reference: no current-operator run
       reproduces it (old-binary artifact; pre-C1 static-D era).
3. ~~A-tests residual~~ DONE 2026-07-26 (see the session block above).
4. ~~Hardening debt (a)+(b)~~ DONE 2026-07-26 (see above).
5. ~~Workstream E (E1+E2)~~ DONE 2026-07-26 (see above).
6. psp8-NLCC block convention: **SOURCE-VERIFIED 2026-07-26** against
   ABINIT's own reader (m_psp8.F90::psp8cc rescales by 1/(4π): "The input
   functions contain the 4pi factor" ⇒ file = 4π·n_c(r) = exactly the
   abinit_psp8.py assumption; comment updated). Only the end-to-end
   numeric exercise on a real NLCC psp8 remains asset-gated — the
   convention risk it guarded is closed.

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
  matched aug cutoff. [RETRACTED next day — that mf carried the real_ylm
  odd-m sign bug (3956b45); true shape-vs-GW agreement is 0.10 mHa, see
  the pin-down entry above.] Mode invariants verified on-cluster: e_1e identical
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
- [x] A-tests: (i) nii≡nij≡no-density+Hartree DONE 2026-07-24 (h0_plus_hartree_identity + add_vpp_i5_alignment, after ∫V_loc·Q̂ settlement); (ii) sym≡nosym DONE via A3 (becsum_full_symm + sym fixture sections); ex_cvij factor-1 e_1e DONE 2026-07-25 (cluster c4 nocv: Δe_1e(cv) −0.5212248 vs ABINIT −0.521220 → 4.8 µHa); (iv) QE-eigenvalue diagnostic rework DONE 2026-07-26 — explicit ∫V_xc·Q̂ assembly: USPP 0.749 Ha → 2.9e-6 STRICT, PAW 0.790 → 4.13e-2 = pure one-center XC, pinned two-sided [3.5e-2, 4.8e-2] — WORKSTREAM A COMPLETE

Workstream B — converter parity
- [x] B1 QE: delete deeq/deeq_nc export; schema_version attribute — 2026-07-24 (079783b)
- [x] B2 ABINIT: ae_vloc/vloc_ps export; per-species proj_per_atom; real vxc; Ewald/madelung; beta + Core/; shape_function check — 2026-07-24 (445a0b3; DEN/NLCC wiring pending on-cluster recheck)
- [x] B3 native ex_cvij builder from Core/ae_wfc (Slater R^L + Gaunt²), validated vs ABINIT-XML ex_cvij (~1e-7 rel) + hydrogenic analytics — 2026-07-24 (f451083)
- [x] B4 schema standardization (schema_version=2 Ha on disk both converters; miller_g sphere on ABINIT side; ijtoh packing verified at read; ae_vloc/vloc_ps required for v2; contract in notes/paw_implementation_plan.md) — 2026-07-24
- [ ] B-tests: same-system Si PAW via both converters — dataset diff + e_1e/e_hf/e_rpa parity (closes ABINIT-mf anomaly); needs ABINIT WFK/POT/DEN assets (cluster); also recheck B2 DEN layout + psp8 NLCC-block convention there. 2026-07-25 cluster: level a+b+c DONE (numbers + nocv/no-onsite attribution in STATUS header); DEN wiring exercised on a real DEN; psp8-NLCC still unchecked (no NC asset). CLOSED 2026-07-25 (pin-down): ΔE_X = real_ylm odd-m sign bug (3956b45, post-fix 87 µHa); Δe_1e = benign vloc-G=0 gauge constant (35.71 mHa/band uniform, bands agree ~3 µHa beyond it)

Workstream C — augmentation-density modes
- [x] C1 single mode flag: pseudopot bool dropped — paw_shape_restored() derives from _exx_opts.vv_compensation (single source; setter kept for tests, delegating); deltaC/K_a inclusion derived from mode in both routes (direct v_x skip-on-shape; THC `_paw_onsite && !shape`); _paw_onsite documented DIAGNOSTIC-ONLY at both sites — 2026-07-25
- [x] C2 dense-sphere THC LL block (aug-aug Coulomb sum on the fft_grid_dim_aug inscribed-Gcut sphere, G-chunked, whenever rho_g doesn't cover it — ζ blocks stay on rho_g, exactly band-limited there; PSD Gram preserved; default configs bit-identical, branch self-disables); in_thc shape abort lifted; new TEST_CASE thc_shape_mode_vs_direct (THC-vs-direct shape V_x 7.6e-5 default / 6.8e-5 half-ecut; mode-difference cross-check 2.8e-6/6.3e-7; dense branch log-confirmed firing) — 2026-07-25. Si a=10.20 acceptance vs direct −1.6863 deferred to the cluster campaign (a10.20 mf not on this host). 2026-07-25 cluster: THC-shape E_X(vv) −1.7349632 at 16 kpt (shape−moment +3.19 mHa, matches si222 +3.38 mHa). CLOSED 2026-07-26: direct dense-grid E_X(shape) −1.73509011 at matched ignore_g0 settings → THC ≡ direct at 0.127 mHa (ISDF-truncation scale); the "−1.6863" was an old-binary artifact, retired
- [x] C3 shared caches + Qfac knob: THC augment now uses psp.paw_aatab() + psp.paw_qrad_tabs(K_max, mode) (same dq=0.01/aug_lmax/per-species selection as its local build — one table set shared with direct v_x; THC's larger Kmax means direct reuses without rebuild); new paw_exx_options::qfac_cache_mb (toml `qfac_cache_mb`, default 256, 0=off) read live in get_or_build_qfac_pair_factor. dq unification itself landed in A4. Value-neutral (C4 mode energies bit-identical across the change) — 2026-07-25
- [x] C4 physics validation (host-side portion): operator identity SETTLED — −1.316447 is the GW Sigma_x (Arnaud) operator, NOT HF Fock (2026-07-21 instrumented-ABINIT work, notes/paw_article_results/abinit_exchange_gw_vs_hybrid.md, updated). HF-side match DONE: deltaC ≡ ABINIT eijkl 5.5e-5 rel + ex_cvij machine-identical (kernels), onsite vv+cv energies identical, smooth residual closed via fock_icutcoul=3 (µHa). Current-code baselines recorded via new [slow] TEST_CASE vexchange_mode_energies (direct dense-grid, ignore_g0): lih222_paw_hf −1.64406506/−1.64395244 (moment+deltaC / shape, split +1.13e-4); local si222_paw −1.31194760/−1.31187642 (split +7.1e-5 — this fixture has fft_mesh_aug==fft_mesh=36³: a coarse aug sphere truncates both modes equally, AND it is a different cell from the rusty cmp mf, so NOT comparable to −1.316447). CLUSTER PORTION DONE 2026-07-25: ABINIT GW Σx regenerated (pawoptosc=1, ISZ off; ecutsigx 25→−1.316447 exact provenance match, converged −1.31747); Δe_1e(cv) 4.8 µHa vs ABINIT −0.521220. REVISED post-3956b45 (the mf carried the real_ylm bug): E_X(shape) = −1.3175731 on the fixed mf → agrees with converged GW Σx to 0.10 mHa (the earlier −47.6 mHa was the converter bug, NOT an operator difference); moment −1.3178007, shape−moment −0.23 mHa. C4 CLOSED — 2026-07-25

Workstream D — ERI/THC route equivalence (Eq. path-equiv)
- [x] D1 audit thc.h/thc.icc/thc_reader_t vs A conventions — verified-consistent ledger + 3 fixes (aug q=0 head vectors; augmented-ERI read-back; Pskna lift → shared cache) in notes/paw_thc_d1_audit.md — 2026-07-25
- [x] D2 route-equivalence matrix-element tests: AB fixture bdft_si222_paw_ab added + AB sections in VH_VX/shape-mode/q_eval/dDeeq_H/hartree-energy tests; V_x route-equivalence STRICT on QE+AB; energy-helper aug-mesh bug fixed. CAUGHT an open direct-route V_H defect on the AB mf (STATUS item 1; strict_VH off there until root-caused). Deferred: Si_conv RPA/HF regression (cluster), sym×AB — 2026-07-25
- [x] D3 aug-build gather buffers G-chunked (the N_aux≳10k OOM driver) + aug-stage memory estimator + synthetic l=3 unit test (paw_q_eval_synthetic_l3, machine-precision) — 2026-07-25
- [x] D4 Cholesky+USPP/PAW hard-abort (builder ctor + read-only reader; diagnostic opt-in allow_smooth_only_aug_pp for the smooth-reference test) — 2026-07-25

Workstream E — notes/documentation
- [x] E1 author canonical D-matrix doc (notes/paw_dmatrix_scgw.tex/.pdf) — 2026-07-26
- [x] E2 corrections to stale notes (converter plan exx_X claim corrected — it is ex_cvij; GW-vs-HF reconciliation addendum in paw_onsite_exchange_analysis.md; deeq-scaling k-weight line marked resolved-by-A3; STEP3 schema block superseded-marked; LaNiO3 marked retest-pending with fix list; −47.6 mHa retracted at every quote; pw2coqui-role documented in source header + contract + bdft error; plan tex I3 parenthetical amended to valence-only) — 2026-07-26
