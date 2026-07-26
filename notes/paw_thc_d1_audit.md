# D1 audit — THC ERI route vs Workstream-A conventions (2026-07-25)

Scope (plan §D1): `methods/ERI/thc.h`, `thc.icc`, `thc_reader_t.hpp` audited
against the matrix-element conventions fixed in Workstream A: becsum
definition, n_s/N_k normalization, K_a placement and prefactor, Hermiticity
at every q, q=0 divergence treatment with compensation-neutral pair
densities, AE basis with identity overlap (I8). Companion code changes
landed with this note; anchors below refer to the post-change tree.

## Verified consistent (no change needed)

1. **Normalization chain (n_s/N_k, Ω).** `thc::intvec_impl` builds the
   smooth Coulomb matrix as `Z ṽ Z† / (Ω·N_k)` (`thc.icc`, the
   `scale(1/volume/nkpts)` after the VCoul multiply); the reader promotes it
   to `1/Ω` with an unconditional `scale(nkpts)` (`thc_reader_t.hpp`,
   build()) — applied for ALL pseudo types before augmentation, so PAW and
   NCPP ERIs are served at the same normalization. The aug blocks are built
   directly at `1/Ω` with the ζ-unit object `χ_Λ = Ω·η_Λ`
   (V_GL = (1/Ω)·Σ_g ζ·(4π/K²)·conj(Ω·η)). n_s enters only through the
   density-matrix contractions downstream, identically for both routes.

2. **Kernel sign convention.** `vG.evaluate(..., v_zero, Q(q))` evaluates the
   smooth kernel at |G − Q|; the aug builder uses `q_cart = −Q(iq)`
   (`augment_thc_with_paw` q-loop; documented in `paw_aug_thc.hpp`). Same
   −q convention in ζ and η.

3. **Hermiticity at every q.** V_GG Hermitian by construction (real ṽ);
   V_GL/V_LG built as an explicit conjugate pair; V_LL from real weights;
   K_a real symmetric. Enforced by `thc_paw_hermiticity(_si)`
   (V(q,P,Q) = conj(V(−q,Q,P)) at 1e-6).

4. **q=0 divergence, compensation-neutral treatment.** The raw Coulomb
   kernel zeroes |K| < cutoff (`potentials/coulomb.hpp`), and the aug blocks
   use the same `w(0)=0` convention (`coulomb_weights_on_rho_g_at_q`,
   dense-LL branch). The divergence correction is applied analytically at
   assembly: `HF_K_correction` adds −madelung·S·D·S with S the AE identity
   (I8) — consistent with the FULL (smooth+aug) pair-density monopole δ_ij,
   independent of the fitted G=0 components. No kernel-level regularized
   v(0) anywhere, so no smooth-vs-aug asymmetry.

5. **K_a placement and prefactor.** K_a (from `compute_K_a(isdf, deltaC)`)
   is injected into the LL tile at every q with factor 1
   (`add_K_a_to_tile`), gated `_paw_onsite && !shape` (C1). In the exchange
   contraction Σ_q pairs k with k−q at weight −1/N_k, reproducing the direct
   one-center prefactor −1/N_k (plan/notes
   `paw_onecenter_exchange_prefactor`); pinned by
   `thc_Ka_onecenter_vs_deltaC` and `vx_onecenter_vs_thc_Ka`.

6. **AE basis / identity overlap (I8).** The X rows are smooth collocation
   plus Y = U·P aug rows built from the full-BZ Pskna lift
   (`fill_Y_rows_for_sk`), i.e. the composite features represent AE-orbital
   pair densities; SCF/Dyson algebra uses S = 1 (`ovlp()` identity
   short-circuit, 1c5738e).

7. **Block-conjugation consistency (notational caveat).** The LL block is
   written `Ω²·Σ_g conj(η_P)·w·η_Q` while GG reads `Σ_g Z_P·ṽ·conj(Z_Q)`.
   The apparent index-transpose pairs with the mirrored conjugation in the
   ζ/η definitions at −q; cross-block consistency is pinned at the
   matrix-element level by `thc_vs_direct_VH_VX(_nonqe)`,
   `thc_vs_direct_nij` and `thc_shape_mode_vs_direct` (V_x elements at
   7.6e-5 / mode-difference 2.8e-6 on the 2-atom si222 fixtures, where a
   conj-flipped inter-atom LL block would fail at ~1e-2). D2 extends this
   matrix to an ABINIT-sourced mf.

## Findings fixed with this audit

F1. **Smooth-only q=0 head vectors (real bug).** `_Chi_head`/`_Chi_bar_head`
    were never extended by augmentation, while every consumer pairs
    `basis_head()` with `thc.Np()` (= augmented N_total):
    `thc_gw.icc Sigma_div_correction` (P-loop to NP), `g0_div_utils.hpp`
    `eval_eps_inv_q`/`head_from_prod_basis`, `embed_eri_t.cpp` — an
    out-of-bounds read whenever PAW/USPP aug + gygi-family div treatment,
    and physically the compensation-charge monopole was missing from the
    q→0 head (under I8 the head reconstructs the AE δ_ij only as
    smooth+aug). FIX: `augment_thc_with_paw` step 7 appends
    `conj(Ω·η^q_Λ(G=0))` rows to `_Chi_head` (same conj/−q convention as
    the smooth `conj(ζ(G=0))`); `_Chi_bar_head` aug rows are zero-padded —
    the G=0 plane wave is exactly band-limited to the smooth sphere, so the
    smooth-only LS representation remains valid in the enlarged basis
    (exact up to the smooth ISDF fit residual of the constant).

F2. **Augmented ERI file read-back aborted.** The reader checked
    `_rp.shape(0) == _Np` on read, but augmented saves store Np = N_total
    while `_rp` is a smooth-grid object — any saved PAW/USPP THC ERI file
    failed to re-load. FIX: allow `_rp.shape(0) <= _Np`, recover
    `_Np_smooth`/`_N_aug` from the difference, and hard-check the head
    vectors have Np columns (pre-F1 files are told to regenerate).

F3. **A4-deferred Pskna lift site resolved.** `thc_reader_t.hpp`
    (augment step 1) now uses the shared shm-backed `psp.Pskna_full_bz()`
    cache when `_psp->get_mpi_context() == _mpi` — always true for a psp
    from `make_pseudopot(*_MF)` since pseudopot stores `mf.mpi()` — with the
    explicit build kept as a fallback for a foreign-context psp. One lift
    shared between THC, direct v_x and becsum consumers.

## Related D3 changes (same files, landed together)

- Gather buffers in the augmentation q-loop are G-chunked (8192, same as
  the dense-LL branch): per-rank targets are (tile rows)×gchunk instead of
  (tile rows)×ngm — the N_aux ≳ 10k OOM driver.
- Augmentation-stage per-rank memory estimate + free-memory warning
  (the smooth-stage estimator in `thc.icc` predates augmentation).

## Cross-reference: D2 outcome on the audited conventions

The D2 ABINIT-sourced fixture (bdft_si222_paw_ab) subsequently CONFIRMED the
THC side of every audited convention at production level: V_x matrix
elements route-equivalent at 3.7e-5, THC Hartree ≡ ABINIT smooth + deltaC
one-center to 4.9e-4 Ha (energy) / 1e-4 (trace). The direct-route V_H
matrix, by contrast, is wrong on that mf (+19.98 Ha trace excess) — an OPEN
defect in add_Hartree_impl recorded in the plan STATUS, item 1.

## Known gaps (out of D1 scope, recorded)

- `build_from_CD` (Cholesky→THC fit) leaves `_Chi_head`/`_Chi_bar_head`
  zero (pre-existing TODO CNY); it is NCPP-only territory now that D4
  hard-aborts Cholesky+USPP/PAW.
- The q-pool ζ slab (`zeta_dist`, rows×ngm per q-pool rank) is inherent to
  the row-split design; it is included in the new memory estimate rather
  than chunked.
