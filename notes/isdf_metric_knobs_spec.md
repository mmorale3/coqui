# Adapting the fit metric in ISDF: reducing the number of interpolation points

**Implementation spec.** Goal: lower `c_mu = N_mu / N_orb_eff` — the number of interpolation points per orbital needed to reach a fixed accuracy — by changing *which* points get selected, not how many.

This is application-agnostic. ISDF appears as a Coulomb-tensor factorization in Hartree-Fock exchange, MP2/RPA, coupled cluster, response-function methods, and quantum-algorithm resource estimation. In all of them the interpolation basis carries the auxiliary index: objects stored there cost `O(N_mu^2)` and any factorization or inversion there costs `O(N_mu^3)`. So `c_mu: 10 -> 6` is `2.8x` on storage and `4.6x` on dense linear algebra in the auxiliary space, everywhere, at once.

Nothing here changes the ISDF ansatz, the structure of the `zeta` solve, or any downstream code. Three independent, composable knobs, all reusing the existing pivoted-Cholesky driver.

---

## 0. TL;DR

| Knob | What it changes | Cost | Risk | What it buys |
|---|---|---|---|---|
| **1. Pair weighting** | Gram becomes `C^w = sum_p c_p A^(p) o B^(p)*` for an application-supplied, separable pair weight | `N_L ~ 4x` on Gram builds | Low (PSD preserved) | Fit targets the quantity you actually need, not raw pair densities |
| **2. Orbital filtering** | Multiply `c_n(G)` by `exp(-alpha G^2/Gc^2)` before the grid FFT | ~free (one FFT sweep) | Low (surrogate only) | Cheap proxy for the Coulomb metric |
| **3. Coulomb re-ranking** | Two-pass: `L^2` Cholesky to rank `s*N_mu`, then exact `v`-metric Cholesky on the pool | `~s x` (s = 1.5-3) | Medium (new driver) | Exact Coulomb metric within the pool |

Do them in order 2 -> 1 -> 3. Knob 2 is an afternoon and tells you immediately whether the metric matters at all for your systems.

Knobs 2 and 3 concern the **grid** metric (which errors in `r` you care about). Knob 1 concerns the **pair** metric (which pair densities you care about). They are independent axes and compose.

---

## 1. Setup, and the one result that shapes the design

### 1.1 Notation

| Symbol | Meaning |
|---|---|
| `N_g` | real-space grid points |
| `N_G` | plane waves retained for the metric (`\|G\| <= Gc`) |
| `N_orb` | orbitals whose pair products are being fitted (all bands, or a bipartite occ/vir split) |
| `N_mu = c_mu * N_orb_eff` | interpolation points |
| `u_n(r_g)` | orbital on the grid (cell-periodic part, for periodic systems) |
| `rho_p(r) = u_i*(r) u_a(r)` | pair density, `p = (i, a)` |
| `Z` | `N_g x N_pair`, `Z_{g,p} = rho_p(r_g)` |
| `Theta = Z[I, :]` | `N_mu x N_pair`, rows at the interpolation points |
| `zeta` | `N_g x N_mu` interpolation vectors |
| `v` | grid-space metric (Coulomb kernel as a convolution) |
| `W = diag(w_p)` | pair-space weights |

The ansatz is `rho_p(r) ~ sum_mu zeta_mu(r) rho_p(r_mu)`, i.e. `Z ~ zeta Theta`.

### 1.2 The grid metric cancels from the `zeta` solve — exactly

Minimize `E = Tr[ (Z - zeta Theta) W (Z - zeta Theta)^H v ]`:

```
dE/dzeta*  =  v (Z - zeta Theta) W Theta^H  =  0
```

`v` is invertible, so

```
zeta  =  Z W Theta^H  ( Theta W Theta^H )^{-1}
```

**`v` cancels identically.** `W` does not.

The reason is structural: `zeta` carries a free index at every grid point, so the residual is driven orthogonal to `row(Theta)` at each `r` independently, and no reweighting *across* `r` can improve on that.

Three consequences that define this whole document:

1. **The fit you already have is the `v`-optimal fit for the points you gave it.** There is nothing to fix in the solve.
2. There is **no quartic step and no ALS** anywhere in what follows. The solve stays a single linear system with a Hadamard-factorized Gram.
3. The metric can only enter through **(a) which points get selected** and **(b) the pair weights `W`**. Both are cheap. That is exactly knobs 1-3.

### 1.3 The pivot objective

With the metric-free `zeta`, the residual is `Delta = Z Q_perp` with `Q_perp` a projector in pair space, and the error in a grid metric `v` as a function of the point set `I` is

```
E_v(I) = Tr[ Q_perp M ],     M = Z^H v Z   (the ERI matrix, M = L^H L)
```

Greedy minimization of this is pivoted Cholesky on

```
K = Z M Z^H = C v C ,        C = Z Z^H  (what you pivot on now)
```

Standard selection uses `M -> 1`, which is the wrong weighting whenever the target is Coulomb integrals: it spends pivots resolving short-wavelength pair structure that `4 pi / G^2` discounts.

**Why the naive route is quartic.** A column of `K` needs `C` applied to a general vector:

```
(C w)_g = sum_{ia} u_i(r_g) u_a*(r_g) [ sum_g' u_i*(r_g') u_a(r_g') w_g' ]     ->  O(N_g N_i N_a)
```

`v` is a convolution and does not commute with the Hadamard structure, so the second `C`-apply costs `N_i N_a` rather than `N_orb`. Total `O(N_mu N_g N_i N_a) ~ N_g N_orb^3`. Knob 3 removes this by replacing the exact `C`-apply with a low-rank surrogate you already compute.

### 1.4 What the current selection does

```
C_{gg'} = A_{gg'} * conj(B_{gg'})
A_{gg'} = sum_i u_i(r_g) conj(u_i(r_g'))        (left orbital set)
B_{gg'} = sum_a u_a(r_g) conj(u_a(r_g'))        (right orbital set)
```

For a single-set (all-pairs) fit, `A = B = P`. For periodic systems with a bipartite split the k/q sums factorize — `k` and `k+q` each run over the full BZ independently — so `A` and `B` are each single BZ sums and the structure above is unchanged.

- **Diagonal:** `C_gg = rho_A(r_g) * rho_B(r_g)`, real, positive, `O(N_g N_orb)` for the whole vector.
- **Column:** `C_{:,g} = A_{:,g} o conj(B_{:,g})`, each a GEMV of the grid-orbital array against `conj(u(r_g))`. `O(N_g N_orb)`.
- **Total selection cost:** `O(N_mu N_g N_orb)`.

Pivoted Cholesky maintains `L` (`N_g x rank`), the diagonal `d`, picks `g* = argmax d`, downdates, updates `d`.

**Property used repeatedly below:** after eliminating pivot set `P`, the Schur complement satisfies `E[P, :] = 0` exactly. Hence

```
C[P, :] = L[P, :] L^H            exactly, no approximation
```

---

## 2. Knob 1 — pair-space weighting

### 2.1 Rationale

`w_p = 1` is the current implicit choice, and it is rarely what any application wants: it treats a pair density that contributes at the `1e-6` level exactly like one that dominates. Every ISDF consumer has a natural weight — how much error in `rho_p` propagates into the target quantity. Supplying it costs almost nothing and redirects the fit's resolution to where it pays.

The weight is **application-supplied**. The constraint the algorithm imposes is only that it be separable, or a short sum of separable terms.

### 2.2 The separability constraint

A general `w_ia` destroys the Hadamard factorization and makes the Gram unaffordable. But

```
w_ia = u_i * u_a           (separable)
```

preserves it exactly. And any weight admitting a short separable expansion of rank `N_L` works at `N_L x` cost:

```
w_ia = sum_{p=1..N_L} c_p * f_p(i) * g_p(a)

C^w = sum_{p=1..N_L} c_p * ( A^(p) o conj(B^(p)) )

A^(p)_{gg'} = sum_i f_p(i) u_i(r_g) conj(u_i(r_g'))
B^(p)_{gg'} = sum_a g_p(a) u_a(r_g) conj(u_a(r_g'))
```

Weights that fit this shape:

| Weight | Rank | Where it comes from |
|---|---|---|
| `w = 1` | 1 | current behaviour |
| `w_ia = f(eps_i) g(eps_a)` | 1 | any product of one-body functions — occupancies, orbital-energy tapers, spatial-locality tags |
| `w_ia = exp(-Delta_ia * t)` | 1 | exponential in an energy difference; separable outright |
| `w_ia = 1/(Delta_ia + eta)` | `N_L ~ 4` | energy denominators — see 2.4 |

Weights that do **not** fit: anything depending on `rho_p` itself (`1/\|\|rho_p\|\|`, norm-adaptive schemes) or on a non-factorizable pair label. If a design step seems to need one, that is the signal to re-expand, not to reach for an iterative solver.

### 2.3 PSD guarantee

Each `A^(p)` is a Gram of scaled orbitals -> PSD (requires `f_p >= 0`). Each `conj(B^(p))` is PSD. Hadamard product of PSD matrices is PSD (Schur product theorem). With `c_p > 0`, **`C^w` is PSD and pivoted Cholesky is as robust as it is now.** No new failure mode. Enforce `f_p, g_p, c_p >= 0` when generating expansions.

### 2.4 Energy denominators via Laplace expansion

The one non-trivial case worth building infrastructure for, because it covers every method with an `1/(eps_a - eps_i)` structure:

```
1/(Delta_ia + eta) = sum_{p=1..N_L} c_p * exp(t_p (eps_i - mu)) * exp(-t_p (eps_a - mu))
```

`eta` regularizes small or vanishing denominators; it is folded into the expansion range at no cost. **Both exponents are <= 0** when energies are referenced to `mu`. No overflow, ever.

Coefficients: `1/x` on `[1, R]` with `R = (Delta_max + eta)/(Delta_min + eta)`, then rescale `c_p = tilde_c_p / (Delta_min + eta)`, `t_p = tilde_t_p / (Delta_min + eta)`. If a minimax generator is already available in the code, reuse it. Fallback: nonnegative least squares for `c_p >= 0` on 200 log-spaced `x` in `[1, R]` with `t_p` geometrically spaced over `[0.05, 20]`. For *selection* this is plenty — you are making a discrete choice, not fitting a number. `N_L = 4` gives ~1% on `1/x` over four decades, far more than the pivot ordering can resolve.

### 2.5 Implementation

The only change to the existing kernel is a per-orbital scalar. Define scaled orbitals

```
u^(p)_n(r_g) = sqrt(f_p(n)) * u_n(r_g)
```

Then `A^(p)` is a plain Gram of the scaled set and the existing column builder is reused verbatim. Better: build all `N_L` at once as a single GEMM.

```
# per pivot candidate g:
t[n]        = u_n(r_g)                                  # length N_orb, gathered
Wt[n, p]    = weight_n(p) * conj(t[n])                  # N_orb x N_L, local
Acols[:, p] = U_left  * Wt_left[:, p]                   # N_g x N_L, one ZGEMM
Bcols[:, p] = U_right * Wt_right[:, p]
Ccol[:]     = sum_p c_p * Acols[:, p] * conj(Bcols[:, p])
```

- No extra communication: the `t` gather is the same one you do now.
- No extra memory: weights are applied on the fly.
- **Orbital truncation:** for large `t_p`, `exp(-t_p (eps_a - mu))` underflows. Skip orbitals with weight `< 1e-8`. Effective cost is well below `N_L x`.

Same substitution applies to the diagonal `d_g = sum_p c_p A^(p)_gg conj(B^(p)_gg)`.

### 2.6 Scope: selection only, by default

Apply `W` to point selection but leave the `zeta` solve unweighted (`isdf_weight_solve = false`). Reason: a single fit usually serves several consumers with different natural weights, and a weighted solve is optimal for one of them at the expense of the others. Point selection is a coarser, more forgiving decision.

If you do enable it, the solve Gram is

```
(Theta W Theta^H)_{mu nu} = sum_p c_p A^(p)_{mu nu} conj(B^(p)_{mu nu})
(Z W Theta^H)_{g mu}      = sum_p c_p A^(p)_{g mu}  conj(B^(p)_{g mu})
```

— same Hadamard structure, same code path, `N_L x` cost. Non-uniform `W` can degrade the conditioning of `Theta W Theta^H`; keep the truncated-pinv threshold rather than switching to a plain Cholesky.

### 2.7 Caveat: rotation invariance

Unweighted selection depends only on the *span* of the orbital set, not the basis: the pair-product space is invariant under unitary mixing within the set, so the pivots and `zeta` are unchanged by any such rotation. That is useful — it means the point set can be cached across any procedure that rotates orbitals within a fixed space.

Weights that depend on individual orbital labels (energies, occupancies) break this. In practice pivots are a discrete choice and small parameter shifts do not move them. Recommendation: **select once with an initial set of weights and freeze the point set** (`isdf_freeze_points = true`), logging the parameters used so the choice is reproducible. Re-select only if the weights move by more than a threshold (default: never).

---

## 3. Knob 2 — filtered-orbital surrogate

### 3.1 What it is

Replace `u_n` by a `G`-space attenuated version *for point selection only*:

```
F(k+G) = exp( -alpha * |k+G|^2 / Gc^2 )        Gc = wavefunction cutoff wavevector
```

Build `tilde_C = tilde_A o conj(tilde_B)` exactly as now, pivot on it, then solve `zeta` with the **unfiltered** orbitals. Only the point set carries over.

### 3.2 Why it is a surrogate, not the metric

Filtering each factor is not the same as multiplying `rho(G)` by `sqrt(v(G))` — the pair density's `G` content is a convolution, so filtering both factors at width `omega` attenuates the product at roughly `omega/sqrt(2)`, with a Gaussian rather than `1/G` profile. It has no rigorous connection to `E_v`. It is here because it costs nothing and suppresses exactly the high-`G` pair content the Coulomb kernel discounts. If it recovers most of the `c_mu` reduction that knob 3 gives, ship it and skip knob 3.

### 3.3 Implementation — zero extra memory

Grid orbitals are produced by inverse FFT from `c_n(G)`. Multiply the coefficients by `F` inside that FFT:

```
build_grid_orbitals(filter_alpha):
    for each orbital n (and k):
        cf(G) = c_n(G) * exp(-alpha * |k+G|^2 / Gc^2)
        u_n(r) = inverse_fft(cf)
```

Sequence: build filtered -> select points -> **rebuild unfiltered** -> solve `zeta`. One extra FFT sweep at setup, no second copy of the grid-orbital array.

### 3.4 Diagnostic

Filtered orbitals are non-orthonormal. Monitor the smallest eigenvalue of their overlap; if `< 1e-6`, `alpha` is too large and the pivoting will degenerate (repeated near-duplicate points). Log the Cholesky rank reached at the standard threshold — a sharp drop signals over-filtering.

Scan `alpha in {0, 0.25, 0.5, 1.0, 2.0}`. `alpha = 0` reproduces current behaviour exactly and is the regression test.

---

## 4. Knob 3 — exact Coulomb-metric re-ranking

### 4.1 The idea

Do not try to pivot on `K = C v C` over the full grid. Instead:

1. Run the **existing** `L^2` pivoted Cholesky to an inflated rank `N_1 = ceil(s * N_mu)`, `s = 1.5-3`. This produces a pool `P` of `N_1` candidate points and the factor `L`.
2. Build `K` **exactly** on that pool, using `C[P, :] = L[P, :] L^H` (exact, see 1.4).
3. Pivoted Cholesky on the dense `N_1 x N_1` matrix `K` down to `N_mu`.

The `v`-optimum is searched only within the `L^2`-good pool. That is a genuine restriction, and it is what makes the whole thing affordable — you are **re-ranking, not re-searching**. Good candidates in one metric are almost always good candidates in the other; what differs is their ordering and which ones become redundant.

### 4.2 Exactness

With `ell = L[P, :]` (`N_1 x N_1`, lower triangular because `L[P_m, j] = 0` for `j > m`):

```
C[:, P] = L ell^H                                    exact
K_PP    = C[P, :] v C[:, P] = ell (L^H v L) ell^H    exact
L^H v L = R^H R    in G-space,   R[G, j] = sqrt(vbar(G)) * Lhat[G, j]
=>  K_PP = Y^H Y,   Y = R ell^H                      (N_G x N_1)
```

`Y^H Y` is manifestly PSD, so pivoted Cholesky on `K` is safe. The only approximations are (i) `G`-truncation at `Gc_metric`, (ii) the pool restriction. Both are converged by scanning one parameter each.

### 4.3 Algorithm

```
Input:  grid orbitals (optionally filtered/weighted), vbar(G), N_mu, s, Gc_metric

1.  N1 = ceil(s * N_mu)
    (P, L) = pivoted_cholesky_on_C(rank = N1)          # existing routine, larger rank
                                                        # L: N_g x N1, distributed over g

2.  ell = L[P, :]                                       # N1 x N1, gather pivot rows -> small

3.  for j in 0..N1-1:
        Lhat[:, j] = fft(L[:, j])                       # distributed FFT
        R[:, j]    = sqrt(vbar(G)) * Lhat[|G| <= Gc_metric, j]
    free(L)                                             # not needed past this point

4.  Y = R @ ell^H                                       # N_G x N1     [P]ZGEMM
5.  K = Y^H @ Y                                         # N1 x N1      [P]ZHERK

6.  (m_1..m_Nmu) = pivoted_cholesky_dense(K, rank = N_mu)
7.  interpolation_points = P[m_1 .. m_Nmu]

8.  solve zeta with unweighted, unfiltered orbitals at those points
```

### 4.4 Cost

| Step | Flops | Relative to current selection |
|---|---|---|
| 1. Pass-1 Cholesky | `s * N_mu * N_g * N_orb` | `s x` |
| 3. FFTs | `N_1 * N_g * log N_g` | negligible |
| 4-5. `Y`, `K` | `2 * N_G * N_1^2` | negligible once `N_G << N_g` |
| 6. Dense Cholesky | `N_1 * N_mu^2` | negligible |

**Total overhead: `~s x`, i.e. 1.5-3x on point selection, with no change in scaling.** Truncating the metric to `|G| <= Gc_metric` is what keeps steps 4-5 off the `N_g` grid; without it they dominate.

Selection is a small fraction of any ISDF-based calculation, and (knob-1 caveat aside) the point set can be cached across orbital rotations, so this is comfortably affordable.

### 4.5 Memory

| Object | Size | Notes |
|---|---|---|
| `L` | `N_g x N_1` | `s x` the current Cholesky factor. **The binding constraint.** Distribute over `g` exactly as now. Freed after step 3. |
| `R`, `Y` | `N_G x N_1` | small once truncated |
| `ell`, `K` | `N_1 x N_1` | e.g. `N_1 = 3e4` -> 14 GB complex; use ScaLAPACK block-cyclic |

If `L` at `s = 2` does not fit, reduce `s` before reducing `Gc_metric` — the pool size is what limits how much the metric can do.

### 4.6 The `vbar(G)` multiplier

Make it pluggable. `vbar` enters only as a diagonal in `G`; **any overall constant is irrelevant** (pivoting is scale-invariant), so ignore FFT normalization constants.

| Option | `vbar(G)` | When |
|---|---|---|
| `bare` | `4 pi / \|G\|^2` | reference |
| `attenuated` (**default**) | `(4 pi / \|G\|^2) * exp(-\|G\|^2 / (4 omega^2))`, `omega^-1 ~ cell dimension | well-conditioned; stops the long-wavelength tail from dominating a weight whose job is discrimination |
| `screened` | diagonal of a static screened interaction, if the application has one available | correct weight when the fit feeds a screened quantity rather than bare integrals |

**Periodic systems:** the point set is shared across `q`, so use the `q`-averaged multiplier

```
vbar(G) = (1/N_q) * sum_q v(q + G)
```

**`G = 0`:** for a bipartite fit over orthogonal orbital sets, `rho_ia(G=0) = <i|a> = 0` at `q = 0`, so the divergence is harmless there. Otherwise set `vbar(0)` to an existing auxiliary-function value or simply `max_{G != 0} vbar(G)`; for selection the latter is fine and robust. A general non-diagonal PSD metric also works (apply `v^{1/2}` as an `N_G x N_G` matrix, `N_G^2 N_1` flops) if the application's natural metric is not diagonal in `G`.

**`Gc_metric` default:** the density cutoff (`2x` the wavefunction `G`). Scan it once per system class and fix it.

---

## 5. Composition

The knobs stack. Full pipeline with everything on:

```
orbitals            -> filtered (knob 2, alpha)
Gram for pass 1     -> C^w      (knob 1, N_L separable terms)
pass 1              -> pool P, factor L at rank s*N_mu
pass 2              -> K = Y^H Y in metric vbar (knob 3) -> final N_mu points
zeta solve          -> unfiltered orbitals, W = 1 (unless isdf_weight_solve)
```

Knob 3's exactness argument is unaffected by knobs 1 and 2 — it only requires that `L` is a pivoted-Cholesky factor of *whatever* Gram pass 1 used.

### Configuration

| Parameter | Default | Scan range | Notes |
|---|---|---|---|
| `isdf_pair_weight` | `none` | `none`, `inv_gap`, user-supplied | knob 1 |
| `isdf_laplace_terms` | 4 | 3-6 | knob 1, energy-denominator weights |
| `isdf_eta` | small, system-dependent | scan | knob 1; raise when denominators approach zero |
| `isdf_weight_solve` | `false` | — | knob 1, solve-side |
| `isdf_freeze_points` | `true` | — | knob 1 + orbital updates |
| `isdf_filter_alpha` | 0.0 | 0-2 | knob 2; 0 = current behaviour |
| `isdf_metric` | `l2` | `l2`, `bare`, `attenuated`, `screened` | knob 3; `l2` = current behaviour |
| `isdf_pool_factor` | 2.0 | 1.5-4 | knob 3 |
| `isdf_metric_gcut` | density cutoff | scan | knob 3 |

Both `isdf_filter_alpha = 0` and `isdf_metric = l2` must reproduce current results **bitwise**. That is the first regression test.

---

## 6. Parallel layout

Constraint: no replication of large objects.

| Object | Layout | Communication |
|---|---|---|
| grid orbitals | as now (over `g`, and/or orbitals/k) | — |
| `t[n] = u_n(r_g)` per pivot | gathered, length `N_orb` | one Allgather per pivot; **unchanged by knob 1** (weights applied locally to an `N_orb x N_L` block) |
| `L` | `N_g x N_1`, distributed over `g` (same as current factor) | rank-1 downdates local; `argmax` over `d` is one Allreduce per pivot |
| FFT of `L[:, j]` | existing distributed FFT | one per column, `N_1` total |
| `R`, `Y` | `N_G x N_1`, block-cyclic over `G` | — |
| `K = Y^H Y` | `N_1 x N_1` block-cyclic | PZHERK internal |
| `ell = L[P, :]` | gather pivot rows | one gather of `N_1 x N_1` |
| pass-2 Cholesky | dense `K`, ScaLAPACK | `N_1` small collectives (Allreduce for `argmax`) — latency-bound; use a blocked/panel variant, or replicate `K` when it fits |

Knob 1 adds **no** communication. Knob 2 adds one FFT sweep at setup. Knob 3's only new collective pattern is the pass-2 dense Cholesky.

---

## 7. Validation

### 7.1 Fit error in the target metric (stochastic, exact estimator)

```
||Delta rho||_v^2 = E_omega [ || Delta^H u ||^2 ],   u = v^{1/2} omega,  omega = random +/-1 on the grid
```

Per probe:

```
u          = ifft( sqrt(vbar(G)) * fft(omega) )
a_i        = u * psi_i                                     # N_g x N_i, elementwise
first_p    = <psi_a | a_i>                                 # N_i N_a, one ZGEMM, O(N_g N_i N_a)
z_mu       = sum_g conj(zeta_mu(r_g)) u_g                  # N_mu
second_p   = sum_mu z_mu conj(rho_p(r_mu))                 # cheap
err       += || first_p - second_p ||^2
```

20 probes, `O(N_g N_i N_a)` each — a couple of GEMMs, minutes. This is the ground-truth number the knobs are trying to reduce. Run it with `w = 1` and with the application's `w` to see both.

Also log the pass-2 Cholesky residual trace (free), but treat it as a diagnostic, not truth — it is the surrogate's own error estimate.

### 7.2 Direct ERI check

On a system small enough to build `(pq\|rs)` exactly, compare against the ISDF-reconstructed tensor: max absolute error, RMS error, and error on the largest-magnitude 1% of integrals. This validates the estimator in 7.1 and catches sign or normalization bugs in `vbar`.

### 7.3 Application-level target

Pick one scalar from a downstream consumer and track it — exchange energy, an MP2 or RPA correlation energy, whatever is cheapest to run end-to-end. The point is a sanity check that improvements in `‖Delta rho‖_v` actually translate; the fit-error curve is what you optimize against.

### 7.4 Protocol

For each configuration, sweep `c_mu in {4, 6, 8, 10, 12}` and record error vs `c_mu`. Report `c_mu*` = smallest `c_mu` meeting a fixed target. **`c_mu*` is the deliverable**, not the error at fixed `c_mu`.

Test set:

| System | Stresses |
|---|---|
| Well-gapped semiconductor, small cell | baseline, fast turnaround |
| Small-gap / localized-orbital system | the hard case for pair-density rank |
| Metal | stresses the `eta` regularization in energy-denominator weights |
| Isolated molecule in a box | large vacuum regions; very different pivot distribution |

Run the metal and the molecule-in-a-box early. Vacuum regions in particular are where `L^2` pivoting is known to waste points, and where a Coulomb-weighted objective should behave differently.

---

## 8. Milestones

**M0 — instrumentation.** Expose `c_mu` at runtime; implement the 7.1 estimator and the 7.2 check; produce baseline error-vs-`c_mu` curves for all four test systems. *Accept: baselines reproducible; estimator agrees with the brute-force ERI check on the small system.*

**M1 — knob 2.** Filter multiplier in `build_grid_orbitals`; rebuild unfiltered before the solve. *Accept: `alpha = 0` bitwise-identical to baseline; `alpha` scan produces a `c_mu*` curve.*

**M2 — knob 1.** Separable-weight interface (application supplies `f_p`, `g_p`, `c_p`); Laplace generator for energy denominators; `N_L`-term weighted Gram in the pivot column builder and diagonal; optional solve-side weighting behind a flag. *Accept: `isdf_pair_weight = none` bitwise-identical; `N_L` convergence flat by 5-6; PSD never violated (log the minimum Cholesky pivot).*

**M3 — knob 3.** Inflated-rank pass 1; FFT/truncate/scale to `R`; `Y`, `K`; dense pass-2 Cholesky. *Accept: `isdf_metric = l2` bitwise-identical; `s = 1` reproduces baseline exactly (no-op check); `Gc_metric` and `s` scans converge.*

**M4 — composition and defaults.** Joint scan, pick defaults per system class, document.

Go/no-go after M4: if the best configuration does not move `c_mu*` by roughly 30% on the hard cases, the metric is not the limiting factor — the pair-density rank is genuinely what it is — and the effort should move to compressing the interpolation basis downstream rather than shrinking it at selection time.

---

## 9. Pitfalls

**Do not** put the grid metric into the `zeta` solve. Section 1.2 proves it is a no-op. If someone "implements the Coulomb metric" and sees a change in `zeta` at fixed points, it is a bug.

**Do not** reach for ALS. Every fit here is a single linear least-squares problem with a Hadamard-factorized Gram. If a design step seems to need alternating optimization, the weighting has been made non-separable somewhere — go back and re-expand it.

**Do not** form `C` or `K` densely over the full grid, and do not implement the exact `C`-apply of section 1.3. That is the quartic step knob 3 exists to avoid.

**`s = 1` is a no-op**, by construction: the pool equals the answer. If knob 3 shows no gain, raise `s` to 3-4 before concluding the metric does not matter.

**Degenerate pivots.** Filtering (knob 2) at large `alpha` and aggressive weighting (knob 1) both increase the chance of near-duplicate candidate points. Keep the existing small-pivot threshold and log the rank actually reached.

**Weight generality.** Enforce `f_p, g_p, c_p >= 0` in the weight interface. A weight that is separable but sign-indefinite breaks the PSD guarantee and the pivoted Cholesky with it.

**Freezing.** Points are rotation-invariant only for unweighted selection. With knob 1 on, select once and freeze; log the parameters used.

**One fit, many consumers.** If a single ISDF fit serves several parts of a calculation with different natural weights, leave the solve unweighted (the default) or blend: `w = 1 + lambda * w_app`. Optimizing the solve for one consumer degrades the others.
