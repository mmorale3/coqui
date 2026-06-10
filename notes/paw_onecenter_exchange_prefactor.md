# PAW one-center (on-site) Fock exchange prefactor — derivation and resolution

**Date:** 2026-06-09 · **Branch:** `paw` · **Context:** determining the correct
prefactor `scl_oc` of the deltaC-mediated one-center exchange in
`src/hamiltonian/paw/v_x_paw.hpp`, which had an empirically-fit value
`scl_oc = −1/(2·n_s·N_k)` (the "½·n_s" factor) flagged as not first-principles.

## TL;DR

The theoretically correct, THC-consistent one-center exchange prefactor is

```
  scl_oc = −1/N_k          (no spin factor, no ½)
```

identical in convention to the smooth+augmentation exchange prefactor
`scl = −1/(N_k·Ω)` already in the same routine (the Ω is only the smooth
G-space Coulomb measure; the one-center deltaC integral is already in energy
units). This is confirmed four independent ways:

1. From-scratch Hartree–Fock derivation (below).
2. Quantum ESPRESSO's own source (`paw_exx.f90`, `exx_std.f90`), Eq. 32/33 of
   Paier et al. JCP **122**, 234102 (2005), unit-converted to CoQui's Hartree
   convention.
3. VASP's PAW-EXX (Paier/Marsman/Kresse), the reference implementation, which
   includes the **full** on-site AE−PS Fock exchange.
4. THC's K_a exchange (`compute_K_a`), to machine precision.

The previously "validated" `−1/(2·n_s·N_k)` (= `−1/(4 N_k)` for n_s=2) was
**4× too small**, and it appeared to match the QE HF eigenvalue of the
`lih_kp222_nbnd16_paw_hf` fixture only because that fixture's `deltaC` is
itself **4× too large** (a stale pw2coqui export, see §5). The two errors —
prefactor ÷4 and deltaC ×4 — cancelled. This is the classic "two wrongs"
empirical fit that [[feedback_derive_before_fit]] warns against.

**Nobody's physics is wrong.** QE, VASP, the derivation and THC all agree on
`−1/N_k`. The bug is a stale test fixture.

---

## 1. Conventions

Spin-restricted (`nspin=1`, `npol=1`); generalization noted at the end.
- `f^σ_{nk}` — per-spin occupation ∈ [0,1]; CoQui stores `occ = f^σ` (verified:
  the LiH fixture has `occ = 1` on the 2 occupied bands; `v_h.hpp` applies the
  spin-degeneracy `ns_scl = 2` to the *density*, confirming `occ` is per-spin).
- k-weight `w_k = 1/N_k` (fixture `kpoint_weights = 1/8` for 8 k-points).
- `P_{n,I}^k ≡ ⟨β_I|ψ̃_{nk}⟩` projector overlaps (`Pskna`).
- `Ω` cell volume. Hartree atomic units throughout (e² = 1).

## 2. Hartree–Fock Fock matrix element (no ½ in the potential)

The Fock exchange energy (closed shell)
```
  E_x = −½ Σ_σ Σ_{nm} f^σ_n f^σ_m ∫∫ ψ*_n(r)ψ_m(r) v(r,r') ψ*_m(r')ψ_n(r') .
```
Its functional derivative — the exchange **potential** (Fock operator) matrix
element — carries **no** ½ (the ½ is the energy/potential factor of a quadratic
functional and is consumed by the derivative):
```
  K_{ab}(k) = ⟨a|V_x|b⟩ = −(1/N_k) Σ_{k',m} f^σ_{mk'} ( a k, m k' | m k', b k ) ,
            = −(1/N_k) Σ_{m} f^σ_m (a m | m b)               [schematically]
```
with `(am|mb) = ∫∫ ψ*_a(r)ψ_m(r) v ψ*_m(r')ψ_b(r')`. **No n_s, no ½.** With
`f^σ = occ = nii`, the prefactor is `−1/N_k`. This is exactly the convention
the smooth+aug term in `v_x_paw.hpp` already uses (`scl = −1/(N_k Ω)`, with the
Ω as the smooth Coulomb measure), and it reproduces QE's NCPP/USPP HF
eigenvalues to ≤8e-4 (`hf_eigenvalues`).

## 3. PAW one-center (on-site) decomposition

Each pair density `ρ_{ab}(r)=ψ*_a ψ_b` is split (Blöchl 1994; Kresse–Joubert
1999; Paier 2005):
```
  ρ_{ab} = ρ̃_{ab} + Σ_atom ( ρ¹_{ab} − ρ̃¹_{ab} ),
  ρ¹_{ab}(r) = Σ_{IJ} conj(P_{a,I}) P_{b,J} φ_I(r) φ_J(r)   (AE on-site),
```
and `ρ̃¹` likewise with pseudo partial waves φ̃. The exact exchange integral
becomes the smooth+compensation FFT part plus the on-site **AE−PS** correction
```
  ΔC^a(I,J,K,L) = ⟨φ_I φ_J | v | φ_K φ_L⟩^AE
                − ⟨φ̃_I φ̃_J + Q̂ | v | φ̃_K φ̃_L + Q̂⟩^PS    (proper Ha).
```
Substituting into the Fock element of §2, the one-center contribution is
```
  K^oc_{ab}(k) = −(1/N_k) Σ_m f^σ_m Σ_a Σ_{IJKL}
                  conj(P_{a,I}^k) P_{m,J}^k  ΔC^a(I,J,K,L)  conj(P_{m,K}^k) P_{b,L}^k .
```
⇒ **`scl_oc = −1/N_k`.** This is exactly the contraction in `v_x_paw.hpp`
(`U(I,L)=Σ_{JK} ΔC(I,J,K,L) P_{m,J} conj(P_{m,K})`, then
`K_{ab} += scl_oc f conj(P_{a,I}) U(I,L) P_{b,L}`).

## 4. Quantum ESPRESSO's own source says the same

`deltaC` is **QE's own PAW Fock kernel**: pw2coqui writes
`deltaC = ke%k / e2²`, where `ke%k` comes from `PAW_init_fock_kernel`
(`PW/src/paw_exx.f90`):
```
  paw_fockrnl(ih,jh,oh,uh) = e2 * kexx        ! Eq. 33 Ref.[Paier2005]   (one e2 here)
  ke%k = k_AE − k_PS                          ! AE − PS  (the ΔC tensor)
```
and `kexx` already carries one more `e2` from the radial Hartree machinery, so
`ke%k = e2²·(Ha integral)` ⇒ `deltaC = ke%k/e2²` is the proper-Ha ΔC. (`e2 = 2`,
Ry units.)

QE's on-site Fock **potential** (`PAW_newdxx`, Eq. 35+32):
```
  deexx(ikb) += weight · 0.5 · ke%k(ih,jh,oh,uh)
                          · becphi(jkb) · conj(becphi(ukb)) · becpsi(okb)
```
called (`PW/src/exx_std.f90`) with `weight = x_occupation/nqs = occ/N_k`, then
applied as `hpsi −= exxalfa · Σ deexx·|β⟩` (`add_nlxx_pot`). The kernel index
`(oh,uh)` is symmetric (it is the pseudo-pair density), matching CoQui's
`ΔC(I,J,K,L)` contraction.

Unit-converting QE's element to CoQui Hartree (Ry → Ha is `/e2`):
```
  ⟨a|V_x^oc|b⟩[Ha] = (1/e2) · 0.5 · (e2² deltaC) · (occ/N_k) · [contr]
                   = (1/N_k) · occ · [contr] · ( 0.5 · e2² / e2 )
                   = (1/N_k) · occ · [contr]              (since 0.5·e2 = 1).
```
i.e. QE's **full** on-site Fock matrix element = `−1/N_k · occ · [ΔC contraction]`
= CoQui with `scl_oc = −1/N_k`. The QE `0.5` (Eq. 32) is **not** a physical ½;
it exactly cancels the extra `e2` carried by `ke%k` relative to the smooth
Coulomb. VASP/Paier 2005 likewise include the full on-site exchange.

## 5. Why the QE HF eigenvalue *seemed* to want ¼

`hf_eigenvalues` (PAW section) compares CoQui's direct Fock to the QE HF
eigenvalues of `tests/unit_test_files/qe/lih_kp222_nbnd16_paw_hf`. At
`scl_oc = −1/N_k` it is off by 0.176 Ha on the occupied n=0 band; the old
`−1/(4 N_k)` matched. Root cause:

- The HF fixture's `deltaC(0,0,0,0) = 0.227928`.
- The (independently validated) non-HF fixture `qe_lih222_paw` has
  `deltaC(0,0,0,0) = 0.0569819`. **Ratio = 4.0000, uniform across all
  elements.**
- `test_paw_onecenter` (`paw_onecenter_dDeeq_H_matches_deltaC_contraction`)
  confirms the **non-HF** `deltaC` equals the independent radial Poisson
  `compute_paw_hartree_atom` to ratio 1.0000 (1.6e-6 Ha). So the **non-HF
  deltaC is correct proper-Ha**, and the **HF deltaC is exactly e²=4× too
  large** = raw QE `ke%k` with the `/e2²` division missing.
- All other pseudo-only fields (`qqq`, `dion`, `qfuncl`) are byte-identical
  between the two fixtures ⇒ only `deltaC` is stale. The HF fixture was
  exported with a pw2coqui predating commit `e42fe6a` ("store deltaC in proper
  Ha (drop e2² = 4 factor)"), see [[project_paw_deltaC_e2_squared]] and
  [[feedback_pw2coqui_use_coqui_shipped]].

So QE's run (which *did* execute `PAW_newdxx`, 409600 calls) computed the full
on-site exchange = `−1/N_k`. CoQui, fed a `deltaC` 4× too large and a prefactor
`−1/(4 N_k)` 4× too small, also landed on the right number — by cancellation.
At the correct prefactor `−1/N_k`, CoQui over-shoots by 4× **only because the
fixture's deltaC is 4× too big**.

## 6. Resolution / actions

1. **Code:** `scl_oc = −1/N_k` in both `v_x` overloads of `v_x_paw.hpp`
   (diagonal and `nij`). Done.
2. **Regression test:** `vx_onecenter_vs_thc_Ka` asserts the direct one-center
   exchange ≡ THC K_a exchange at the production prefactor (machine precision).
   Note this is a *consistency* test (both sides use the same `deltaC`), so it
   is insensitive to the deltaC magnitude — it locks direct↔THC, not the
   absolute scale.
3. **Fixture:** regenerate `lih_kp222_nbnd16_paw_hf` `deltaC` with the
   canonical (post-`e42fe6a`) CoQui pw2coqui — or, equivalently and exactly,
   divide the on-disk `deltaC` by e²=4. After this, `hf_eigenvalues` (PAW) and
   the absolute QE-HF comparison pass at `scl_oc = −1/N_k`.
4. Audit other `*_paw_hf` fixtures (e.g. Si) for the same stale `deltaC`;
   USPP has no one-center term so is unaffected.

## 7. Spin generalization

`scl_oc = −1/N_k` carries **no** explicit `n_s`, exactly like the smooth
`scl = −1/(N_k Ω)`. For `nspin=2`, each spin channel s sums its own
`f^σ = occ(s,·)` over the same-spin occupied bands — the formula is unchanged;
an `nspin=2` fixture remains the open verification.

## References

- P. E. Blöchl, *Projector augmented-wave method*, PRB **50**, 17953 (1994).
- G. Kresse, D. Joubert, PRB **59**, 1758 (1999).
- J. Paier, R. Hirschl, M. Marsman, G. Kresse, JCP **122**, 234102 (2005)
  — Eqs. 31–35 (PAW Fock kernel; the `0.5` of Eq. 32).
- M. Marsman, J. Paier, A. Stroppa, G. Kresse, JPCM **20**, 064201 (2008).
- P. Giannozzi et al., *Advanced capabilities for materials modelling with
  Quantum ESPRESSO*, JPCM **29**, 465901 (2017) — EXX+USPP/PAW "following the
  method of Paier 2005".
- QE source: `PW/src/paw_exx.f90` (`PAW_newdxx`, `PAW_xx_energy`,
  `PAW_init_fock_kernel`, `PAW_fock_onecenter`), `PW/src/exx_std.f90`
  (`PAW_newdxx` call, `weight = x_occupation/nqs`).
- See also [[project_paw_vh_vx_thc_consistency]],
  [[project_paw_deeq_self_consistency_todo]], [[feedback_paw_validate_value]].
