# Symmetry lift of the full density matrix in the direct v_x(nij) / becsum(nij)

Static-route plan phase 4 (notes/static_route_selection_plan.md). Derivation
first (working-agreement rule); the implementation is a one-line band-matrix
lift in each nij kernel.

## Setup

The direct exchange `hamilt::v_x(nij)` / `hamilt::paw::v_x(nij)` and the
direct Hartree's `compute_becsum_full_symm` need the one-particle density
matrix γ_K at every **full-BZ** point K, while storage is IBZ-only:
`nij(s, k̃, a, b)` in the canonical-band basis at k̃ ∈ IBZ. Both v_x(nij)
kernels already build the *states* at the full BZ by the exact View-2
G-space lift (`transform_k2g` + conj at time-reversal points):

- K = S·k̃, no TR:  ψ_{K,a} := R_S ψ_{k̃,a}
- TR point:        ψ_{K,a} := (R_S ψ_{k̃,a})^*

and the projectors by the cached `Pskna_full_bz` lift (atom permutation +
Wigner-D + Bloch phase). The only open question is which **band matrix**
multiplies these lifted states.

## Claim

In the lifted basis the full-BZ band matrix is the *unchanged* IBZ matrix,
complex-conjugated at time-reversal points:

    n^(K) = nij(s, k̃)        (K = S·k̃, no TR)
    n^(K) = nij(s, k̃)^*      (TR point)

— identical to the rule already derived and shipping in
`compute_becsum_full_symm` (v_h_paw.hpp).

## Derivation

Assume the SCF state does not break the crystal symmetry (the same
assumption that justifies IBZ-only storage everywhere, including the THC
route): γ(Sr, Sr') = γ(r, r') for every space-group operation S, and (npol=1,
no SOC) time-reversal symmetry γ = γ^*.

Bloch-decompose γ = Σ_K γ_K. Space-group invariance gives γ_{S k̃} =
R_S γ_{k̃} R_S†. Substituting the spectral form of the IBZ block,
γ_{k̃} = Σ_ab ψ_{k̃,a} n_ab(k̃) ψ†_{k̃,b}:

    γ_K = Σ_ab (R_S ψ_{k̃,a}) n_ab(k̃) (R_S ψ_{k̃,b})†
        = Σ_ab ψ_{K,a} n_ab(k̃) ψ†_{K,b}                (K = S·k̃)

so the band matrix is unchanged when the lifted states are used. At a
time-reversal point, γ_{-k} = γ_k^* gives

    γ_K = Σ_ab (Rψ_a)^* n_ab(k̃)^* (Rψ_b)^T
        = Σ_ab ψ_{K,a} n_ab(k̃)^* ψ†_{K,b}              (ψ_K = (Rψ)^*)

— the conjugated band matrix. ∎

## Compatibility with the natural-orbital decomposition

The nij kernels evaluate K[γ_K] = Σ_p w_p K[|χ_p⟩⟨χ_p|] by diagonalizing the
band matrix (`natural_occupations`). Conjugation preserves Hermiticity and
the (real) spectrum {w_p}; the eigenvector matrix maps U → U^*, so the
natural orbitals at a TR point are χ_p^(K) = Σ_a U^*_ap ψ_{K,a} — i.e. the
decomposition **commutes with the lift**. Feeding n^(K) into
`natural_occupations` and contracting against the already-lifted
ψ_r_full / Pskna_full_bz is therefore exact for:

- the smooth pair densities  conj(χ_p(r))·ψ_a(r),
- the Q-augmentation cross terms  P*_{nat}·P_{canonical},
- the deltaC one-center contraction (P_nat bilinears),

since each is bilinear in (lifted inner state) × (lifted inner state)†, and
any residual per-state U(1)/Bloch-phase gauge of the lift cancels within
those bilinears (same cancellation argument as the becsum docstring).

## Consequences

- becsum(nij): `compute_becsum_full_symm` already implements exactly this —
  plan item 4.1 required **no new code**, only the certifying tests (the
  cleanup-plan note "landed for the diagonal path only" was stale).
- v_x(nij) (smooth v_x.hpp and PAW v_x_paw.hpp): replace the direct
  `nij(s, kq)` indexing by `nij(s, kp_to_ibz(kq))` with conj at
  `kp_trev(kq)`, and drop the nk_ibz==nk guards (kernels + hamilt_hf.icc).
- Acceptance (plan 4.3): (a) sym-vs-nosym invariance of the direct-route
  HF-SCF (non-diagonal Dm from iteration ≥ 2 exercises the general-nij
  lift in BOTH becsum and v_x); (b) the phase-3 route-equivalence battery
  repeated on the symmetric mesh; (c) the interim sym guard retired.
