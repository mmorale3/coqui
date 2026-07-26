# Phase 0.1: gygi divergence treatment for the direct (hamiltonian) route

Resolves item 0.1 of `static_route_selection_plan.md`. Conclusion: **the
direct route obtains the correct gygi treatment by calling the existing
`hf_t::HF_K_correction` unmodified after `gen_Vexchange`** — no new
correction code, and no interim gygi-rejection guard is needed.

## Anatomy of gygi in the ERI route (established by inspection)

1. The ERI objects themselves are built with the `ignore_g0` head
   convention: only the q=0, G=0 element of the Coulomb kernel is zeroed
   (e.g. the aug-side Qfac kernel `thc_reader_t.hpp:1022`,
   `w = K²>1e-14 ? 4π/(ΩK²) : 0`). The `div_treatment` toml option on the
   ERI blocks does not alter the stored ERI head used by HF.
2. gygi is applied ENTIRELY at the operator level, in one shared routine:
   `hf_t::HF_K_correction(sF, Dm, S, madelung)` (`hf_t.cpp:66`):
   - no-op when `_div_treatment == ignore_g0`;
   - otherwise adds Δ_ij(s,k) = −madelung · (S·Dm·S)_ij at every IBZ k.
   Both the THC and Cholesky HF paths call it identically after K
   (`thc_hf.icc:304/538`, `cholesky_hf.icc:215`), with
   `madelung = MF->madelung()` (bdft computes it when absent, commit
   0758481; abinit2coqui emits/repairs it since the madelung-EXX fix).
   `hf_t` coerces `gygi_extrplt*` to `gygi` with a warning (`hf_t.cpp:53`).

This is the PRB 80, 085114 (2009) finite-size correction: the q→0
4π/q² head excluded by the zeroed-element convention contributes, for a
smooth Σ_x(q), exactly −ξ (the madelung/Gygi-Baldereschi constant) per
occupied orbital; in operator form −ξ·S·γ·S with γ the density matrix.

## Route-independence argument

Δ depends only on (Dm, S, madelung) — it contains NO ERI (or route)
content. The route-dependent object is K under the ignore_g0 convention,
and there invariant I7 (workstream D acceptance: the D1 audit explicitly
covered the q=0 divergence treatment; C2/C4 numerical closure at the
0.10–0.13 mHa ISDF scale; the I7 matrix-element tests strict on the AB
fixture) certifies

    K_direct[ignore_g0] ≡ K_ERI[ignore_g0]   (to route-equivalence tol).

Hence

    gygi(direct) := K_direct[ignore_g0] + Δ ≡ K_ERI[ignore_g0] + Δ
                  = gygi(ERI)

to the same tolerance, with Δ literally the same code. The plan's test
3.6 enforces this end-to-end in SCF.

## Notes

- PAW/qe/bdft band basis: S = I (AE-basis overlap identity), so
  S·Dm·S = Dm; keep the S form for generality — it matches the ERI call
  sites verbatim.
- Hartree needs no divergence correction in either route: the zeroed
  G=0 Hartree component is the standard charge-neutrality convention,
  identical on both sides (I7 Hartree parity).
- Implementation corollary for phase 1.2: the `hf_t::evaluate(Hamilt_ERI)`
  overload ends with the same
  `if (compute_exchange) HF_K_correction(sF, Dm, S, MF->madelung());`
  as the THC/Cholesky overloads. Plan items updated: 1.1/3.5 interim
  gygi-rejection guard dropped; 3.6 kept as the acceptance test.

## Phase 0.2 (pseudopot sharing) — resolved alongside

`hamilt::make_pseudopot(*MF)` already implements lazy shared acquisition
through the MF (`MF::get_pseudopot/set_pseudopot`, MF.hpp:378–382);
`simple_dyson` (simple_dyson.h:66) and `thc_reader_t` (:268–278) both use
it, and thc_reader propagates `paw_exx_options` to the shared instance.
`hamilt_eval_t` does the same. Consistency rule: if both a THC block and
the hamilt block set exx options on the shared pseudopot, they must agree
— `hamilt_eval_t` checks the already-set options and errors on conflict
(last-writer-wins is not acceptable for physics options).
