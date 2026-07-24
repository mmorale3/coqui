# Making CoQui's K_a (PAW one-center exact exchange) VASP-compatible

Date 2026-06-26. Compares the Paier-2006 / Kresse-Joubert-1999 / Rostgaard (arXiv:0910.1921)
one-center exact-exchange formulation against CoQui's current K_a, and specifies changes.
Companion to [reference_paw_exx_onecenter_vasp] and notes/paw_article_results/vasp_paw_exx_literature.md.

## 1. The correct formula (Rostgaard eqs. 49-50, 74, 97-104; = Paier/Kresse-Joubert)

PAW exact-exchange energy splits as
    E_xx = Ẽ_xx[pseudo plane-wave, incl. compensation] + Σ_a (E¹ₓ,AE − E¹ₓ,PS)
One-center (per atom a, valence):
    E¹ₓ = −½ Σ_{ijkl} D_{ik} D_{jl} (φ_i φ_j | v | φ_l φ_k)     [AE], minus same with φ̃ (PS)
    D^a_{i1 i2} = Σ_n f_n P*_{n,i1} P_{n,i2}   (one-center density matrix; same-spin only δ_{σn,σm})
Four-index integral → Gaunt² × radial Slater:
    (φ_i φ_j | v | φ_k φ_l) = Σ_L [G^L_{l_i m_i, l_k m_k} · G^L_{l_j m_j, l_l m_l}] · R^L(i,k,j,l)
    R^L(i,k,j,l) = ∫∫ (r_<^L / r_>^{L+1}) u_i(r)u_k(r) u_j(r')u_l(r') dr dr'   (u = r·radial part)
L-TRUNCATION (the key detail):
    - AE one-center: L exact up to 2·l_max (Gaunt triangle kills L>2·l_max automatically).
    - PS one-center + plane-wave/compensation: truncated at LMAXFOCK (lower), CONSISTENT
      with what the smooth grid + compensation charge represent. AE-pair-density "shape
      restoration" (NMAXFOCKAE, ~150-400 eV) makes the smooth PS term accurate so the
      one-center is a small correction.
Core: core-valence exchange included via a valence-core tensor X^a (Rostgaard eq.104):
    K_cv(i,j) = Σ_{core c} Σ_L G^L G^L R^L(i, c, j, c)  (frozen-core orbitals as extra channels);
    core-core exchange is a volume-independent constant (droppable for EOS).
Screened HSE: ONLY the radial kernel R^L changes (erfc(μr)/r_> instead of 1/r_>); Gaunt + L-trunc unchanged.

## 2. CoQui's current K_a (what it actually does)

- deltaC = QE's `ke(nt)%k / e2²` = (k_AE − k_PS) from QE `PAW_init_fock_kernel` (paw_exx.f90),
  IMPORTED via pw2coqui. CoQui does NOT compute the kernel itself.
- compute_K_a (local_isdf.hpp): K_{λξ} = sign(λ)sign(ξ)·deltaC[i(λ),j(λ),i(ξ),j(ξ)] in the
  ISDF symmetric-pair basis; contracted with the one-center density matrix; scl_oc = −1/N_k.
- QE's PAW_fock_onecenter L-truncation = lmax_rho (= 2·l_max for Si), SAME L for AE and PS;
  the 4-term kernel symmetrization is COMMENTED OUT (un-symmetrized kernel).
- Smooth+aug exchange (v_x_paw.hpp): q_lm compensation pair density (qgm = smooth Q̃ = qfuncl,
  multipoles only) — the QE addusxx analog. NO AE-pair-density shape restoration.
- Core-valence/core-core: Kcv/Kcc NOT consumed (pseudopot.cpp:795 comment only); deltaC is
  valence-valence only. Si kjpaw has has_gipaw=false.
- CoQui HAS the radial data to do it natively: aewfc, pswfc, qfuncl, core_aewfc (pseudopot.h:132-136);
  pfunc=aewfc·aewfc, ptfunc=pswfc·pswfc.

## 3. Differences (CoQui/QE vs VASP) — three, in priority order

(D1) **L-truncation symmetry.** CoQui inherits QE's AE=PS=lmax_rho. VASP: AE to 2·l_max but
     PS to LMAXFOCK (lower), matched to the smooth representation. Symmetric truncation can leave
     a PS-vs-smooth inconsistency → residual exchange error. [Likely contributor.]
(D2) **AE-pair-density shape restoration in the SMOOTH part.** VASP restores the AE pair-density
     shape on the PW grid (NMAXFOCKAE); CoQui's smooth term carries only compensation multipoles.
     This is what made VASP-RPA Si right WITHOUT one-center (pre-6.6.0). [Likely DOMINANT for sp Si.]
(D3) **Core-valence exchange.** VASP includes it; CoQui does not (has core_aewfc but unused).
     Frozen-core, volume-dependent; non-negligible for Si (1s2s2p core). [Secondary but real.]
Plus: QE kernel un-symmetrized vs VASP symmetrized ½(AE−PS) — affects Hermiticity, small.

## 4. Changes — recommended path: compute K_a NATIVELY in CoQui

CoQui has aewfc/pswfc/qfuncl/core_aewfc, so build ΔC_a internally (decouple from QE's kernel):
1. Radial Slater R^L(i,k,j,l) on the species radial grid (g(nt)%r, rab) from u=r·aewfc/pswfc;
   standard inward/outward Hartree (already have radial_hartree_multipole machinery in
   paw_onecenter.hpp — reuse for the r_<^L/r_>^{L+1} kernel).
2. Gaunt coefficients G^L from real spherical harmonics (l,m of each projector channel; nhtolm).
   QE exposes ap/lpl/lpx (Clebsch-Gordan) — mirror or recompute.
3. ΔC_a(I,J,K,L) = Σ_L G^L_{IK} G^L_{JL} [ R^L_AE(I,K,J,L) − R^L_PS(I,K,J,L) ], with
   AE summed to L=2·l_max, PS to LMAXFOCK_equiv (new knob; start = l_max, test up).
   Symmetrize over the 4 index permutations (enable the term QE comments out).
4. Add core-valence: K_cv(I,J) = Σ_c Σ_L G²·R^L(I,c,J,c) using core_aewfc; add to the
   one-center Fock contraction (new term, gated on core_aewfc present).
5. Full HF kernel = 1/r_>; keep an erfc(μ) variant for screened hybrids.
6. Feed ΔC_a into the EXISTING compute_K_a / ISDF path (drop-in replacement for the imported deltaC).

Lighter alternative (if native is too much first): keep imported deltaC but (a) enable
symmetrization, (b) expose/raise PS L-truncation knob in pw2coqui, (c) add the D3 core-valence
term natively from core_aewfc. Does not fix D2 (shape restoration).

## 5. Validate / find the dominant term FIRST (cheap, before the rewrite)

Numerical experiments on the qe-7.5 Si runs (runs_qe75) to rank D1/D2/D3 before investing:
- K_a on vs off: how much does the existing one-center move PAW exchange/EOS? (small ⇒ smooth-part D2 dominates).
- Compare CoQui per-volume PAW exchange e_hf vs QE EXX (already ~0.025 Ha too weak) with/without
  each candidate term.
- Vary the imported-deltaC L-truncation (regenerate via pw2coqui with higher/lower L) to test D1.
- Add core-valence (D3) from core_aewfc and measure the EOS shift.
Target: PAW & USPP RPA@PBE → ONCV's 10.26 (PP-consistency), and per-volume exchange match QE/AE.

## 6. Caveats
- VASP "Si 10.27" is HSE (screened), not full HF — the robust target is RPA@PBE PP-consistency
  (ONCV 10.26 vs augmented 10.62), not an absolute full-HF a0.
- For sp Si the one-center is SMALL; D2 (smooth shape restoration) may dominate, in which case
  the native-K_a rewrite helps 3d/TMO more than Si. Sequence the diagnostics (§5) accordingly.
- Equation sources: Rostgaard arXiv:0910.1921 (accessible, eqs 47,49-50,74,97-104); Paier JCP
  124,154709 (2006) + 122,234102 (2005); Kresse-Joubert PRB 59,1758 (1999); Blöchl PRB 50,17953.
