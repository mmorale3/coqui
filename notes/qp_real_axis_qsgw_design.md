# Real-axis QSGW with QP-form Green's function and contour deformation

**Status:** design (2026-05-10). Implementation to follow in `src/methods/GW_real_axis_qp/`.

## Motivation

The existing real-axis QSGW (`src/methods/GW_real_axis/real_axis_qp_scf_driver.hpp`) keeps the Green's function as a **numerical spectral function** `A_wskij` sampled on a real-frequency grid:
```
A_wskij(ω) = (-1/π) Im G^R(ω) componentwise
G^R(ω) = A built from QP poles via Lorentzian projection with finite η broadening
```
This forces η > 0 to keep A finite and the kernels NUFFT-friendly. At the production scale (Si kp444 nbnd=256) the η broadening (~1.4 eV with η=0.05 Ha) dominates the QP peak widths and dictates how dense the ω grid must be near μ.

A **QP-form QSGW** keeps G in analytic quasiparticle form throughout:
```
G^R_{ij}(s, k, z) = sum_n MO_{in}(s,k) * MO_{jn}*(s,k) / (z - ε_n^QP(s,k))
```
with `z` complex. No spectral broadening; no dense ω grid for A; SCF iterates the QP energies/orbitals directly. This is the textbook Faleev–Schilfgaarde–Kotani QSGW formulation.

The bottleneck moves to evaluating Σ_c(ω = ε_n^QP) accurately. The proven numerical technique is **contour deformation (CD)**: deform the ω integration contour from the real axis to a piece running along the imaginary axis plus a residue sum from QP poles enclosed by the contour. This:
- Stays compatible with ISDF/THC: W needed only on the imaginary axis where the existing imag-axis machinery applies directly.
- Avoids the η broadening that biased the numerical-A version.
- Evaluates Σ_c only at the small set of QP energies, not on a full ω grid.

## Mathematical framework

### Σ_c via contour deformation

The standard expression (e.g. Govoni & Galli 2015, Ren et al. 2012):
```
Σ_c_{ij}(s, k, ω) = (i/2π) ∮ dω' G_{ij}(s, k, ω+ω') W_c(ω')
                 = ∮ dω' G_{ij}(s, k, ω+ω') W_c(ω') / (2π i)        (1)
```
where `W_c = W - V` is the dynamic part. Deform the contour to a closed loop consisting of:
1. **Real-axis segment** from -R to +R
2. **Quarter-arcs** at infinity (vanishing contribution since W_c decays at large |ω|)
3. **Imaginary-axis segment** from +iR down to -iR

The integral along the imaginary axis is what we KEEP (cheap, well-behaved). The deformation picks up residues at poles of G that lie INSIDE the contour. For the upper half plane (ω+ω' near the QP energy ε_n^QP):

```
Σ_c_{ij}(s,k,ω) = sum_{n: f(ω, ε_n^QP)=±1} MO_{in} MO_{jn}* W_c(ε_n^QP - ω + i*η_pole) * f_residue
                + (1/2π) ∫_{-∞}^{+∞} dω' G_{ij}(s,k, ω + iω') W_c(iω')         (2)
```

The first term is the **pole residue sum** — counts QP poles of G enclosed by the contour between ω and the imaginary axis (depends on whether ω is above or below μ_chem). For ω above μ: occupied poles contribute (with appropriate sign); for ω below μ: unoccupied poles contribute.

The second term is the **imaginary-axis integral** — well-behaved, W_c smooth on iω axis, G also smooth there (no poles since ε_n^QP is real).

### The two pieces

#### (A) Residue sum
For each (s, k_FBZ) and a query ω = ε_m^QP(s, k_ibz):
```
Σ_c^{residue}_{ij}(s, k_ibz, ω) =
    sum_{n at FBZ k_FBZ' = k_FBZ + q}
        sign_n * MO_{i,n}(s,k_FBZ') MO*_{j,n}(s,k_FBZ')
                * W_c_{aux→primary}(q, ε_n^QP - ω)
```
where:
- `sign_n = -1` if ε_n^QP < μ AND ω > μ (occupied pole below ω → enclosed)
- `sign_n = +1` if ε_n^QP > μ AND ω < μ (unoccupied pole above ω → enclosed)
- Else 0

`W_c(Ω)` evaluated on the real axis at `Ω = ε_n^QP - ω`. This is a **real-axis W evaluation at a specific real frequency Ω**. We get it from:
- Either the imag-axis W via analytic continuation (Pade fit), OR
- A real-axis Dyson solve of W = (I - V·Π)⁻¹ V at the specific Ω we need

Choice for the first cut: **Pade continuation from the imag-axis W**. Reuse the existing imag-axis machinery; CD's analytic-continuation step is well-understood.

#### (B) Imaginary-axis integral
Standard convolution on iω axis:
```
Σ_c^{iaxis}_{ij}(s, k_ibz, ω) = (1/2π) ∫ dω' G_{ij}(s, k_ibz, ω + iω') W_c(iω')
```
With G in QP form,
```
G_{ij}(s,k, z) = sum_n MO_{in} MO*_{jn} / (z - ε_n^QP)
```
so the iω integral becomes:
```
Σ_c^{iaxis}_{ij}(s,k,ω) = (1/2π) sum_n MO_{in} MO*_{jn}
                       * ∫ dω' W_c(iω') / (ω - ε_n^QP + iω')
```

The integral `∫ dω' W_c(iω') / (ω - ε_n^QP + iω')` is a 1D principal-value-like integral over a smooth W_c on iω with a simple pole at iω' = ε_n^QP - ω (located on the imaginary axis at fixed real height). Evaluate with the existing IAFT (imag-axis Fourier transform) on a Gauss-Legendre or Chebyshev mesh — this is exactly the same operation as the imag-axis GW Σ kernel.

In aux basis:
```
Σ_c_aux^{iaxis}_{PQ}(s,k,ω) = sum_n MO_{P,n}^aux MO*_{Q,n}^aux * J_n(ω)
J_n(ω) = (1/2π) ∫ dω' W_c_aux(iω') / (ω - ε_n^QP + iω')
```
Here `MO^aux_{P,n} = sum_μ X_{P,μ}(s, k) MO_{μn}(s,k)` is the QP-aux overlap (no orbital sum left). For each (n, ω) the integral J_n(ω) is one IAFT lookup.

### Reuse of the imag-axis machinery

The imag-axis CoQui code already computes W(iω') on a Matsubara-like mesh from THC ERIs (the `bdft` IAFT path). For CD, we want **NOT W on iωn fermionic** but **W on iν boson + general iω on the chosen contour**. The existing imag-axis GW does this. We thread it in:

1. Compute W(iω') on the IAFT bosonic mesh using the existing `methods/scr_coulomb/scr_coulomb_t::update_w` (imag-axis variant).
2. From W(iω'), extract W_c(iω') = W - V.
3. For each (s, k, n, m) needed by the QSGW iter, evaluate Σ_c^{iaxis}_{ij}(s,k, ε_m^QP) via the convolution above.

For the residue part, W_c(Ω real) is needed. Get it from Pade extrapolation of W_c(iω') — accurate near the real axis for plasmon poles and well below the Matsubara-cutoff frequency.

## Algorithm

```
QSGW iteration (QP-form, contour deformation):
1. Diagonalize H_eff:    H_eff · MO = MO · diag(ε^QP)
2. Update mu:            mu from Fermi count on ε^QP
3. Update Dm:            Dm = MO · diag(f(ε^QP - mu)) · MO^H
4. Compute Σ_x(Dm):      static exchange via thc_hf (already exists, reuse)
5. Compute imag-axis W:  using imag-axis scr_coulomb_t::update_w with the THC factors
                         (NO real-axis Π here)
6. For each (s, k_ibz, n, m):
   Σ_c_{nm}(s, k, ε_n^QP) = residue_sum(ε_n^QP - ω, MO, ε^QP, mu)
                          + iaxis_integral(ε_n^QP, MO, ε^QP, W_c)
7. V_corr_{ij} = 0.5 (Σ_c_{ij}(ε_i^QP) + Σ_c_{ij}(ε_j^QP)) + h.c. trick
8. H_eff_new = H_0 + Σ_x + V_corr
9. Mix (DIIS), check residual, repeat
```

Step 5 is the heavy lift; everything else is one-shot per iter.

## Memory architecture

User constraint: large tensors distributed, medium shared, small task-local.

| object | shape | dtype | scope |
|---|---|---|---|
| W_aux(iω') = ImW + ReW on iω mesh | (Nq_ibz, Naux, Naux, N_iω) | complex | **distributed** over (P, Q) — same proc grid as imag-axis production |
| W_c_aux(iω') = W - V | same | complex | distributed |
| V_aux(q) | (Nq_ibz, Naux, Naux) | complex | **distributed** (already so via THC reader) |
| MO_{μn}(s, k_ibz) | (ns, Nk_ibz, nbnd, nbnd) | complex | **shared per node** (sArray, mirrors existing pattern) |
| MO_aux_{P,n}(s, k_ibz) = X · MO | (ns, Nk_ibz, Naux, nbnd) | complex | **shared per node** (~ Naux·nbnd matrix, medium) |
| THC factor X | (ns, Nk, Naux, nbnd) | complex | shared per node (already so) |
| ε^QP_{ska} | (ns, Nk_ibz, nbnd) | real | shared / local (small) |
| H_eff_{skij} | (ns, Nk_ibz, nbnd, nbnd) | complex | shared per node |
| Σ_c_{nm}(ε_n^QP) for one (s, k) | (nbnd, nbnd) | complex | **task-local** scratch |
| J_n(ω) iaxis integrand evaluations | (nbnd, NQuad) | complex | task-local during the per-(s,k) loop |
| residue partial sums | (nbnd, nbnd) | complex | task-local |
| Pade-fit coefficients of W_c | (Nq_ibz, Naux, Naux, N_pade) | complex | **distributed** over (P, Q) — same grid as W_c |

The largest object is the imag-axis W_aux + W_c_aux + Pade coefficients — each ~5-50 GB at production. These MUST stay distributed.

The MO_aux is the medium-size shared object: at Naux=2566, nbnd=256, Nk_ibz=13, ns=1: 1·13·2566·256·16 B = 137 MB — shared per node, ~comfortable.

Per-task scratch: nbnd² complex matrices (~1 MB) for Σ_c assembly. Trivially local.

## File structure (new module)

```
src/methods/GW_real_axis_qp/
├── real_axis_qp_state.hpp           # state container (mirrors real_axis_mb_state but no A_wskij)
├── real_axis_qp_cd_t.h              # CD Σ_c evaluator (residue + iaxis)
├── real_axis_qp_residue.hpp         # residue-sum kernel
├── real_axis_qp_iaxis_integral.hpp  # iaxis-integral kernel
├── real_axis_qp_pade.hpp            # Pade fit of W_c(iω') for real-axis W_c(Ω)
├── real_axis_qp_scf_driver_cd.hpp   # SCF loop driver
├── real_axis_qp_dispatcher.hpp      # toml dispatch (selected via mode = "qsgw_qp_cd")
├── CMakeLists.txt
└── tests/
    ├── test_qp_cd_smoke.cpp         # one-iter smoke test on LiH222
    ├── test_qp_cd_residue.cpp       # check residue sum on a closed-form analytic case
    ├── test_qp_cd_iaxis.cpp         # check iaxis integral on a Drude-pole W_c
    └── test_qp_cd_lih222_vs_numerical_A.cpp   # compare QP-form vs numerical-A QSGW
```

### Toml interface

```toml
[real_axis_qpgw_cd]
mode             = "qsgw_cd"   # qsgw_qp_cd / evgw_qp_cd
beta             = 1000
niter            = 24
conv_thr         = 1e-3
alpha_mix        = 0.5
diis_window      = 6
mix_kind         = "diis"
# CD-specific:
N_iomega         = 32                  # imag-axis Matsubara points used by W_c
iomega_cutoff    = 50.0                # Matsubara cutoff (Ha) — must exceed widest QP-pole sep.
pade_npts        = 16                  # Pade fit order
pade_min_omega   = 1e-3                # smallest Matsubara |ω'| for fit stability
# Reused QSGW:
qp_type          = "bisection"
off_diag_mode    = "qp_energy"
div_treatment    = "ignore_g0"
hf_div_treatment = "ignore_g0"
align_mo         = true
dE_cluster_align = 1e-3
tol_max_de       = 1e-3
tol_dDm          = 1e-3
write_chkpt      = true
output           = "si_qsgw_qp_cd"
# THC standard:
[interaction.thc]
ecut             = 120
thresh           = 1e-5
storage          = "incore"
```

## Mathematical details to nail down before coding

1. **Residue sign convention.** I wrote the deformation with the upper half plane closure; double-check signs against Govoni-Galli Appendix A. Specifically: which way does the contour close for ω above vs below μ?

2. **Quadrature scheme for the iω integral.** Gauss-Legendre on a finite [-iω_max, +iω_max] segment, OR the IAFT mesh that the existing imag-axis production uses (high-accuracy log-spaced Matsubara). Reuse choice: existing IAFT mesh, integration weights already in CoQui (`numerics/imag_axes_ft/IAFT.hpp`).

3. **Pole vs branch-cut handling of W_c on the real axis.** When the residue argument `Ω = ε_n^QP - ω` lies near the plasmon pole of W (Ω ≈ Ω_pl), Pade extrapolation diverges. Mitigate by:
   - Adding a small imaginary part `+i*η_eval` (e.g. 1e-3) to the residue argument
   - Falling back to a real-axis Dyson solve of W at that Ω if Pade fails

4. **Handling of (k, q) pair construction.** For each (s, k_ibz, n, m, ω) the residue sum needs:
   - Loop over q_FBZ
   - For each q: look up the FBZ k' = k_FBZ + q, find the band indices n' there
   - The aux-basis W_c at q with appropriate (P, Q) entries
   - Pade-evaluate W_c(Ω) where Ω = ε_{n'}^QP(s, k') - ω
   - Multiply by X(s, k', P, μ) * MO(s, k', μ, n') * MO*(s, k', ν, n') * X(s, k', Q, ν) ?
   - Sum into Σ_c_{ij}(s, k_ibz, n, m)
   
   This is the orbital basis form. Reusing the symmetry-adapted ISDF pattern from `methods/GW/thc_gw.icc::eval_Sigma_all_kspace`: the orbital sum is folded into the X projection at FBZ k.

5. **DIIS variable.** Same as numerical-A QSGW: H_eff is the iteration variable. The CD framework provides Σ_c(ε^QP); V_corr = hermitization step; H_eff_new = H_0 + Σ_x + V_corr.

## Implementation phases

### Phase A — kernel building blocks (no SCF yet)
- residue-sum kernel: aux-basis input MO, ε^QP, μ, W_c(Ω), q-grid → orbital-basis Σ_c contribution
- iaxis-integral kernel: aux-basis MO, ε^QP, W_c(iω'), IAFT mesh → orbital-basis Σ_c contribution
- Pade fit + evaluation of W_c on the real axis

### Phase B — Σ_c(ω) at one frequency
Test against the numerical-A QSGW's Σ_c at one (s, k, n, m, ω) for LiH222. Should agree to a few percent (some η broadening expected for the numerical version).

### Phase C — SCF loop
Wrap into the QSGW iter (mirror real_axis_qp_scf_driver.hpp's structure). Use DIIS on H_eff.

### Phase D — production
Si kp444 nbnd=256 comparison: QP-form QSGW vs numerical-A QSGW (commit f1b46e1) on the SAME fixture (same MF, ecut, thresh). Compare:
- Indirect gap
- μ
- HOMO/LUMO QP energies
- Per-iter wallclock
- Memory footprint

### Tests
- `test_qp_cd_smoke`: 1-iter LiH222, sanity on Σ_c finite + hermitian.
- `test_qp_cd_residue`: closed-form analytic case — single bosonic pole (Drude-like W_c), check residue sum against analytic integral.
- `test_qp_cd_iaxis`: closed-form analytic case — same Drude W_c, check iaxis integral analytically.
- `test_qp_cd_lih222_vs_numerical_A`: full SCF comparison on LiH222 against the existing QSGW.

## Estimated effort

| phase | LOC | wall-time estimate |
|---|---|---|
| A — kernels | ~600 (C++) + 200 (tests) | 1-2 weeks |
| B — Σ_c(ω) validation | ~150 (tests) | a few days |
| C — SCF driver | ~400 | a few days |
| D — production comparison | ~tests + analysis | 1 week |
| **Total** | ~1500 LOC | 4-6 weeks |

This is a major piece of new physics code. The phase A kernels are the heaviest because of the residue-sum bookkeeping (which (n, k', q) pairs contribute for each (n, k, m, ω) target). The IAFT integration reuses the existing imag-axis primitives.

## Risks / open questions

1. **Pade is fragile.** For Si scGW the plasmon-pole structure of W is sharp; Pade can introduce spurious poles. The fallback (real-axis Dyson W) is more reliable but expensive. Need a hybrid strategy.
2. **k-point sums in the residue term** can blow up if naive loop. The full BZ sum at q-resolved level is what the imag-axis GW already does; reuse that loop structure.
3. **Comparison with numerical-A QSGW** may show systematic differences if η in the latter biases QP energies more than expected. Document this clearly in the test.
4. **THC accuracy at finite ω.** The THC factorization is exact for ground-state ERIs; for finite-ω response functions like W_c(Ω real) there can be ISDF interpolation error. Mitigate by using the same Naux and thresh as the imag-axis production runs.

## Next actions

1. Land this design as a tracked memory entry (`project_qp_real_axis_qsgw.md`).
2. Start Phase A: residue-sum kernel + Pade fit + iaxis integral.
3. After Phase A, write the validation tests (Phase B) BEFORE assembling the SCF loop.
