# PAW on-site valence exact exchange: CoQuí vs. the correct (all-electron) expression

**Author's note (2026-07-16).** This documents (i) exactly what CoQuí currently
computes for the PAW on-site valence exchange, term by term, from the source; (ii)
what ABINIT computes and why it equals the true all‑electron (AE) exchange; and
(iii) a derivation of the difference between the two, which is a single, physically
identifiable term that CoQuí neglects. The analysis is backed by the instrumented
Si `a=10.20` comparison (both codes, finite‑size correction off, on‑site term
toggled): CoQuí on‑site exchange `+0.0414 Ha` vs. ABINIT `+0.0925 Ha` (converged),
a `0.052 Ha` gap that the derived missing term accounts for.

Source of record:
- CoQuí smooth + augmentation + one‑center: `src/hamiltonian/paw/v_x_paw.hpp`
- CoQuí `deltaC` kernel: `src/python/mean_field/abinit_interface/paw_deltaC.py`
  (identical definition to QE's `pw2coqui`; stored in the mean‑field h5 and read
  into `species_paw_t::deltaC`)
- ABINIT on‑site oscillator: `abinit-10.6.7/src/65_paw/m_pawpwij.F90`,
  used by `calc_sigx_me` (`src/70_gw/m_sigx.F90`)

---

## 0. Notation

Per k‑point pair `(m, k_q)`, `(n, k_p)` the **pair (transition) density** is
`ρ_{mn}(r) = ψ*_{m k_q}(r) ψ_{n k_p}(r)`. The exact‑exchange energy is the bilinear

$$
E_x \;=\; -\tfrac12 \sum_{s}\sum_{k_q k_p}\sum_{mn}
       \frac{f_{m k_q}\, f_{n k_p}}{N_k}\;
       \big\langle \rho_{mn}\,\big|\,v_C\,\big|\,\rho_{mn}\big\rangle ,
\qquad
\big\langle \rho\,|\,v_C\,|\,\rho'\big\rangle
   \equiv \tfrac{4\pi}{\Omega}\sum_{G}\frac{\rho^*(K)\,\rho'(K)}{|K|^2},
\;\; K = G + k_p - k_q .
$$

The matrix‑element form actually built in the code is
`K_{ij}(s,k_p) = -\frac{1}{N_k}\sum_{k_q}\sum_n f_{n k_q}\,\langle \rho_{n i}|v_C|\rho_{n j}\rangle`,
with `E_x = \tfrac12\sum f_i K_{ii}`.

**PAW decomposition of the pair density.** With smooth orbitals `ψ̃`, projectors
`p_I^a`, and AE/PS partial waves `φ_I^a, φ̃_I^a`:

$$
\rho_{mn}(r) \;=\; \underbrace{\tilde\rho_{mn}(r)}_{\text{smooth (grid)}}
   \;+\; \sum_a \underbrace{\Big[\rho^{1}_{mn,a}(r) - \tilde\rho^{1}_{mn,a}(r)\Big]}_{\displaystyle D_{mn,a}(r)\ \text{(on‑site, radial)}} ,
$$

$$
\rho^{1}_{mn,a} = \sum_{IJ} P^{a*}_{mI}P^{a}_{nJ}\,\phi_I^a\phi_J^a,\qquad
\tilde\rho^{1}_{mn,a} = \sum_{IJ} P^{a*}_{mI}P^{a}_{nJ}\,\tilde\phi_I^a\tilde\phi_J^a,
\qquad P^{a}_{nJ} \equiv \langle p_J^a | \tilde\psi_n\rangle .
$$

The **compensation charge** `n̂_{mn,a}` reproduces the *multipole moments* of
`D_{mn,a}` but not its shape:

$$
\hat n_{mn,a}(r) = \sum_{IJ} P^{a*}_{mI}P^{a}_{nJ}\sum_{L} q^{L}_{IJ}\,g_L(r)\,\{Y_{LM}\},
\qquad
q^{L}_{IJ} = \int \big(\phi_I\phi_J - \tilde\phi_I\tilde\phi_J\big)\,r^{L}\,r^2 dr ,
$$

with `g_L` the (Bessel/Gaussian) shape function normalised to unit `L`‑moment. By
construction `∫ (D_{mn,a}-\hat n_{mn,a})\, r^L Y_{LM}\, d^3r = 0` for all `LM`.

Two identities used below (Coulomb bilinearity + the moment property):
- `⟨D|v_C|D⟩ - ⟨ρ¹|v_C|ρ¹⟩ + ⟨ρ̃¹|v_C|ρ̃¹⟩ = -2⟨ρ̃¹|v_C|D⟩`  (expand `D=ρ¹-ρ̃¹`).

---

## 1. What CoQuí currently implements

CoQuí builds `K_{ij}` as **three** additive pieces (`v_x_paw.hpp`, function `v_x`).

### 1a. Smooth pair density (grid)
Lines 442–446. For occupied `n` at `k_q` and every `a` at `k_p`:
$$
\tilde\rho_{na}(G) = \mathrm{FFT}\big[\, \tilde u^*_{n}(k_q,r)\,\tilde u_{a}(k_p,r)\,\big](G).
$$

### 1b. Compensation‑charge augmentation (grid), lines 448–485
$$
\boxed{\;
\rho^{\text{full}}_{na}(G) = \tilde\rho_{na}(G)
   \;+\; \Omega\sum_a\sum_{IJ} P^{a*}_{nI}(k_q)\,P^{a}_{aJ}(k_p)\;
          Q^{IJ}_a(K)\,e^{-iK\cdot\tau_a}
\;}
$$
where `Q^{IJ}_a(K)` is the Fourier transform of the **compensation charge**
`n̂_{IJ}` (moments × shape function), carrying the `4π/Ω` prefactor
(`evaluate_Q_IJ_at_K_fast`, l.229; the explicit `Ω` on l.461 restores the density
normalisation). Crucially this is `\hat n`, i.e. **moments only** — it is *not* the
AE−PS pair density. So on the grid CoQuí forms `ρ^{full} = ρ̃ + \hat n`.

### 1c. Coulomb contraction (smooth + augmentation), lines 487–497
$$
K^{\text{sm+aug}}_{ij}(s,k_p) = -\frac{1}{N_k\,\Omega}\sum_{k_q}\sum_n f_{n k_q}
   \sum_G v_C(K)\,\rho^{\text{full}*}_{ni}(G)\,\rho^{\text{full}}_{nj}(G),
\qquad \Big[\;\mathrm{scl}=-\tfrac{1}{N_k\Omega}\;\Big].
$$
In bilinear form this is `-\tfrac{1}{N_k}\sum_{k_q}\sum_n f_n\,\langle \tilde\rho+\hat n\,|\,v_C\,|\,\tilde\rho+\hat n\rangle`.

### 1d. One‑center `deltaC` correction, lines 503–590
$$
\Delta K^{\text{oc}}_{ij}(s,k_p) = -\frac{1}{N_k}\sum_{k_q}\sum_n f_{n k_q}
   \sum_a\!\!\sum_{IJKL} P^{a*}_{iI}(k_p)\,P^{a}_{nJ}(k_q)\,
   \Delta C^{a}(I,J,K,L)\,P^{a*}_{nK}(k_q)\,P^{a}_{jL}(k_p),
\qquad \Big[\;\mathrm{scl_{oc}}=-\tfrac{1}{N_k}\;\Big],
$$
with (`paw_deltaC.py`, l.111–142; QE/`pw2coqui` identical):
$$
\boxed{\;
\Delta C^{a}(I,J,K,L) = K^{\text{AE}} - K^{\text{PS}},\qquad
K = \sum_{LM}\int V^{IJ}_{LM}(r)\,\rho^{KL}_{LM}(r)\,dr,
\;}
$$
$$
\rho^{IJ,\text{AE}}_{LM} = \sum_{lm}\mathrm{ap}(lm,l_i,l_j)\,\underbrace{\phi_I\phi_J}_{\text{pfunc}},
\qquad
\rho^{IJ,\text{PS}}_{LM} = \sum_{lm}\mathrm{ap}(lm,l_i,l_j)\,\big[\underbrace{\tilde\phi_I\tilde\phi_J}_{\text{ptfunc}} + \underbrace{q^{L}_{IJ}g_L(r)}_{\text{qfuncl}=\hat n}\big].
$$

**Read literally:** the AE side is the pure product `φ_Iφ_J`; the PS side is
`φ̃_Iφ̃_J + n̂_{IJ}` (note the `+ qfuncl` on l.107/126 of `paw_deltaC.py` and
l.215 of `paw_onecenter.hpp`). Hence, per atom,
$$
\Delta C \;=\; \big\langle \phi\phi \,|\,v_C\,|\,\phi\phi\big\rangle^{\text{AE}}
   \;-\; \big\langle \tilde\phi\tilde\phi + \hat n \,\big|\,v_C\,\big|\, \tilde\phi\tilde\phi + \hat n\big\rangle^{\text{PS}}
   \;=\; \langle\rho^1|v_C|\rho^1\rangle - \langle\tilde\rho^1+\hat n|v_C|\tilde\rho^1+\hat n\rangle .
$$

### 1e. CoQuí total (compensated PAW)
Collecting 1c + 1d and cancelling the shared `⟨\hat n|v_C|\hat n⟩`:
$$
\boxed{\;
E^{\text{CoQuí}}_x = -\tfrac12\sum \frac{f f}{N_k}\Big[
   \langle\tilde\rho|v_C|\tilde\rho\rangle
 + 2\langle\tilde\rho|v_C|\hat n\rangle
 - 2\langle\tilde\rho^1|v_C|\hat n\rangle
 + \langle\rho^1|v_C|\rho^1\rangle
 - \langle\tilde\rho^1|v_C|\tilde\rho^1\rangle
\Big]
\;}
$$
This is the standard **compensated (moment‑matched) PAW** exchange: the smooth
density interacts with the *multipoles* `\hat n` of the on‑site AE−PS difference,
plus an exact on‑site AE−(PS+comp) Fock correction.

*(Prefactor note: `scl = -1/(N_k Ω)` for the grid Coulomb measure and
`scl_oc = -1/N_k` for the already‑integrated radial `deltaC` are internally
consistent and validated to machine precision against CoQuí's own THC `K_a`
(`test vx_onecenter_vs_thc_Ka`). The prefactors are **not** the issue here.)*

---

## 2. What ABINIT computes (= the correct AE exchange)

ABINIT builds the **oscillator** of the *full AE pair density* directly
(`m_pawpwij.F90`). The on‑site addition to `⟨ψ̃_m|e^{-iKr}|ψ̃_n⟩` is
(l.479, 527, 561, 859–918):
$$
\boxed{\;
M^{\text{onsite}}_{mn}(K) = \sum_a\sum_{IJ} P^{a*}_{mI}P^{a}_{nJ}\;
   q^{a}_{IJ}(K),\qquad
q^{a}_{IJ}(K) = 4\pi\sum_{LM}(-i)^{L} Y_{LM}(\hat K)\,
   G^{LM}_{l_i m_i,\,l_j m_j}\; \mathrm{ff}^{aL}_{IJ}(|K|)
\;}
$$
$$
\boxed{\;
\mathrm{ff}^{aL}_{IJ}(K) = \int_0^{r_a} j_L(Kr)\,
   \big[\,\phi_I\phi_J - \tilde\phi_I\tilde\phi_J\,\big](r)\; dr
\;}
$$
i.e. the transform of the **full AE−PS on‑site pair density `D`, with the true
radial shape and NO compensation charge** (`phiphj - tphitphj`, l.561; the
`use_pawnhat0 = 0` default in `m_sigx.F90:155`). The exchange is then
$$
E^{\text{ABINIT}}_x = -\tfrac12\sum \frac{f f}{N_k}\,
   \big\langle \tilde\rho + D \,\big|\, v_C\,\big|\, \tilde\rho + D\big\rangle
 = -\tfrac12\sum \frac{f f}{N_k}\Big[
   \langle\tilde\rho|v_C|\tilde\rho\rangle
 + 2\langle\tilde\rho|v_C|D\rangle
 + \langle D|v_C|D\rangle \Big].
$$
Because `ρ̃ + D` **is** the exact AE pair density (up to the `ecutsigx` truncation
of the sharp `D`, which we verified converged: on‑site moves `<1 mHa` from
`ecutsigx = 25→50`), this equals the true AE exchange. This is the
Arnaud–Alouani / Shishkin–Kresse PAW‑GW oscillator.

---

## 3. The difference (derivation)

Subtract §1e from §2, per pair, using
`⟨D|v|D⟩ - ⟨ρ¹|v|ρ¹⟩ + ⟨ρ̃¹|v|ρ̃¹⟩ = -2⟨ρ̃¹|v|D⟩`:

$$
E^{\text{ABINIT}}_x - E^{\text{CoQuí}}_x
= -\tfrac12\sum\frac{ff}{N_k}\Big[
   2\langle\tilde\rho|v_C|D\rangle - 2\langle\tilde\rho|v_C|\hat n\rangle
 + 2\langle\tilde\rho^1|v_C|\hat n\rangle - 2\langle\tilde\rho^1|v_C|D\rangle \Big]
$$
$$
\boxed{\;
E^{\text{ABINIT}}_x - E^{\text{CoQuí}}_x
= -\sum \frac{f_m f_n}{N_k}\;
   \big\langle\, \tilde\rho_{mn} - \tilde\rho^{1}_{mn}\ \big|\ v_C\ \big|\ D_{mn} - \hat n_{mn}\,\big\rangle
\;}
$$

This is the term CoQuí drops. Its two factors are each localised in/near the
augmentation sphere:

- **`\tilde\rho - \tilde\rho^1`** — the smooth pair density minus its on‑site PS
  partial‑wave expansion. Vanishes only if the PS partial waves are *complete*
  inside the sphere (the usual PAW assumption `\tilde\rho|_{\rm sphere} = \tilde\rho^1`).
- **`D - \hat n`** — the AE−PS on‑site difference minus its multipole moments: the
  **shape of `D` beyond its moments**. Zero moments by construction, nonzero shape.

The compensated PAW total‑energy formalism sets this cross term to zero. That is
an excellent approximation for the *Hartree/total energy* (long‑ranged, moment
dominated) but **not** for **exchange**, which is short‑ranged and sensitive to the
pair‑density shape inside the core. For Si this neglected term is
`E^{ABINIT}-E^{CoQuí} = -( -1.6863 - (-1.7382)) `… numerically:

| quantity (Si a=10.20, 4×4×4, finite‑size off) | value (Ha) |
|---|---|
| smooth only, both codes (identical to 6 µHa) | −1.77964 |
| ABINIT on‑site contribution `2⟨ρ̃|v|D⟩+⟨D|v|D⟩` | **+0.0925** |
| CoQuí `⟨ρ̃|v|\hat n⟩`‑type augmentation (1b/1c extra) | +0.0481 |
| CoQuí `deltaC` one‑center (1d) | −0.0067 |
| CoQuí on‑site total | **+0.0414** |
| **missing cross term** `-⟨ρ̃-ρ̃¹|v|D-\hat n⟩` | **≈ +0.0511** |

The sign is consistent: CoQuí's on‑site is too small (recovers ~44 %), so its total
exchange is too *negative* by `0.052 Ha` (the missing term is repulsive/less‑binding).

---

## 4. What I believe is the correct expression

Two mathematically equivalent routes recover the missing term. Both are "correct";
they differ in numerical convenience.

### Route A — shape‑restored oscillator (ABINIT/Arnaud–Alouani; recommended)
Replace CoQuí's compensation‑charge augmentation (§1b) with the **full AE−PS
pair‑density form factor**, and then **drop `deltaC` entirely**:
$$
\rho^{\text{full}}_{na}(G) = \tilde\rho_{na}(G)
   + \sum_a\sum_{IJ} P^{a*}_{nI}(k_q)P^{a}_{aJ}(k_p)\,
     \mathcal{Q}^{IJ}_a(K)\,e^{-iK\tau_a},
\qquad
\mathcal{Q}^{IJ}_a(K) = \!\int_0^{r_a}\! j_L(Kr)\,[\phi_I\phi_J - \tilde\phi_I\tilde\phi_J]\,dr \times (\text{angular}) ,
$$
i.e. use `pfunc − ptfunc` (AE−PS product) in the radial Hankel transform *instead of*
`qfuncl` (the compensation moments). Structurally this is a one‑line change to
`build_qrad_tab` / `evaluate_Q_IJ_at_K`: feed `pfunc − ptfunc` rather than the
moment×shape `qfuncl`. Then `K^{sm+aug}` alone equals the exact AE exchange and the
`deltaC` block (§1d) must be removed (it would double count).
- **Pro:** exact, matches ABINIT/VASP, no separate one‑center kernel.
- **Con:** `D` is sharp; the augmentation FFT mesh (`fft_mesh_aug`) must resolve
  `ff^L_{IJ}(K)` out to the same effective cutoff ABINIT uses (`ecutsigx`; here
  converged by ~50 Ha). This is the reason the compensated route was chosen
  originally — `\hat n` is smooth and grid‑friendly.

### Route B — keep compensation, add the missing cross term to `deltaC`
Keep §1b/1c as is, but redefine the one‑center kernel so that it also supplies
`-⟨ρ̃-ρ̃¹|v|D-\hat n⟩`. In practice this means the on‑site Fock must be evaluated
between the **smooth‑minus‑onsite** density and the **shape residual** `D-\hat n`,
not only the pure on‑site AE/PS products. This is harder to cast as a frozen
`nh⁴` matrix because `ρ̃-ρ̃¹` depends on the actual smooth orbitals, not just the
projector amplitudes — so Route A is cleaner.

### Recommendation
Implement **Route A**: swap the augmentation form factor from the compensation
charge (`qfuncl`, moments) to the full AE−PS partial‑wave product
(`pfunc − ptfunc`, shape), on a mesh converged like ABINIT's `ecutsigx`, and drop
`deltaC` from the exchange. Validate against the numbers here (Si on‑site must go
`+0.0414 → +0.0925 Ha`; total vv exchange `-1.7382 → -1.6863 Ha`; per‑band `Σ_x`
must match ABINIT `iszoff` band‑for‑band). Keep the compensation‑charge path for
the **Hartree** deeq (where it is correct and grid‑friendly).

**Caveat / next check.** The identity in §3 assumes the exchange is the clean
bilinear `⟨ρ_{mn}|v|ρ_{mn}⟩` and that CoQuí's terms map exactly as written above;
the `0.052 Ha` numerical closure supports this. Before implementing, reproduce the
decomposition at a second system (e.g. LiH or a 3d oxide, where on‑site exchange is
larger) to confirm the missing‑term identity holds beyond Si sp.

---

## Addendum (2026-07-26, plan E2): GW-vs-HF reconciliation — how this note resolved

The "correct (all-electron)" ABINIT on-site target used throughout this note
(`+0.0925 Ha`, and the si222 `-1.316447 Ha` family) came from ABINIT's **GW
Sigma_x** pipeline (`calc_sigx_me` / `m_pawpwij` Arnaud–Alouani oscillators)
— which is a DIFFERENT operator from ABINIT's hybrid-DFT Fock
(`pawdijfock`: exact one-center `eijkl` + `ex_cvij`). The reconciliation
(plan C4, 2026-07-21→25; see
`notes/paw_article_results/abinit_exchange_gw_vs_hybrid.md`):

- CoQuí **moment+deltaC** mode ≡ ABINIT **HF Fock** side: deltaC ≡ `eijkl`
  to 5.5e-5 rel, `ex_cvij` machine-identical, on-site vv+cv energies
  identical; the smooth residual closed via bare-Coulomb
  (`fock_icutcoul=3`) at the µHa level.
- CoQuí **shape** mode (full AE−PS form factors,
  `build_qrad_tab_full_aeps` — i.e. exactly the "missing term" this note
  derived, landed as Option A) ≡ ABINIT **GW Sigma_x**: on the corrected
  ABINIT-converted mf, E_X(shape) = −1.3175731 vs converged GW Σx −1.31747
  → **0.10 mHa**.
- The interim "−47.6 mHa Fock-vs-Arnaud operator difference" reported from
  the first cluster campaign is **RETRACTED** — it was the abinit2coqui
  `real_ylm` odd-m Condon–Shortley sign bug (fixed 3956b45), not physics.

So the `0.052 Ha` gap analyzed here was real, was the shape-restored
pair-density term, and is now a selectable mode validated against the
matching ABINIT operator on both sides.
