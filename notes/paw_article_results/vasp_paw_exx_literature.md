# How VASP/PAW handles exact-exchange EOS — literature synthesis (2026-06-24)

Deep-research result (19 sources, 25 claims verified, 23 confirmed). Motivation: CoQui/QE
augmented HF gives Si a0 ~10.7 (USPP)/11.0 (PAW) vs NCPP 10.35; need the correct one-center
exact-exchange formulation for CoQui's K_a.

## Bottom line
- **VASP implements RIGOROUS one-center (on-site) exact exchange for HF/hybrids**: Fock
  integral decomposed (Kresse–Joubert) into smooth plane-wave part + AE-onsite and PS-onsite
  partial-wave exchange, evaluated SEPARATELY with asymmetric L-truncation (PS up to LMAXFOCK,
  AE up to max-L = 2×ℓ_max). VASP wiki: "For exchange, the exact one-center terms are also
  implemented." → VASP-PAW HSE Si a0 = 5.435 Å (HSE06)/5.415 (HSEsol); no augmentation blow-up.
- **QE treats USPP and PAW identically** via augmentation-charge (q_lm) integrals (Paier 2005),
  NO distinct one-center exchange operator → QE-PAW-HF ≈ QE-USPP-HF. Confirms our diagnosis.
- **K_a must be on-site AE−PS four-partial-wave exchange**, NOT a density-augmented pair density.

## Key references (for the paper + K_a implementation)
- Kresse & Joubert, PRB 59, 1758 (1999) — US↔PAW; AE/PS one-center decomposition (US linearizes
  away the one-center terms PAW keeps → q_lm cannot carry one-center exchange).
- Paier, Hummer, Marsman, Kresse et al., JCP 122, 234102 (2005) — PAW EXX (augmentation method).
- Paier, Marsman, Hummer, Kresse, Gerber, Ángyán, JCP 124, 154709 (2006) + erratum 125, 249901
  (2006) — screened-hybrid PAW EXX one-center integrals as implemented in VASP. [EQUATION SOURCE
  for K_a — not yet equation-mined.]
- Harl & Kresse, PRL 103, 056401 (2009); Harl, Schimka & Kresse, PRB 81, 115126 (2010) — ACFDT/RPA
  EOS in PAW (E_RPA = E_c + E_EXX, E_EXX = full HF total energy incl Hartree/kinetic/Ewald). Study
  frozen-core effect. [NB: 3 authors; "Harl & Kresse" attribution is wrong for the 2010 paper.]
- Schimka, Harl & Kresse, JCP 134, 024116 (2011) — HSEsol; Si EOS benchmark table.
- arXiv:0902.0889 = Nguyen & de Gironcoli (EXX/RPA in plane-wave QE, NO PAW) — do NOT cite as Harl–Kresse.

## Critical nuance for RPA (reframes the Si case)
VASP did NOT implement one-center exchange for RPA/GW until v6.6.0 (LFOCKSTD flag, ~2026; strongly
recommended for GW/RPA). Pre-6.6.0 RPA used plane-wave "shape restoration" of the AE pair density
(accurate ~150 eV at NMAXFOCKAE=1, ~400 eV at =2; errors "worst for 3d"). Yet Harl-Schimka-Kresse
2010 got Si RPA a0 right pre-6.6.0 → for light sp Si, resolving the AE pair density on the grid was
already enough; rigorous one-center matters most for 3d. ⇒ CoQui PAW/USPP at ~10.6 is worse than
BOTH VASP routes → CoQui's augmented exchange isn't capturing the AE pair density adequately. Fix
paths: (A) rigorous one-center AE−PS exchange (K_a; essential for 3d/TMO); (B) verify CoQui's THC-aug
exchange ERI resolves the AE pair density (may suffice for Si alone).

## VASP knobs (map to CoQui choices)
- LMAXFOCK: max L for augmentation of charge densities in HF routines.
- NMAXFOCKAE / LMAXFOCKAE: AE pair-density "shape restoration" on PW grid; LMAXFOCKAE = 2 (sp),
  4 (d), test 6 (f) = 2×ℓ_max in POTCAR; NMAXFOCKAE=2 → ~400 eV restoration accuracy.
- LFOCKSTD (≥6.6.0): include exact one-center EXX in RPA/GW total energies+forces.

## Caveats
- Strong EOS evidence is SCREENED hybrids (HSE, 25% SR), not full unscreened HF — proves VASP
  one-center machinery is accurate, not a like-for-like full-HF comparison. No source quantifies
  QE's augmented-HF Si error directly (corroborated indirectly).
- QE "no one-center term" is single-source (2-1) — confirm vs latest QE.
- VASP one-center exchange integral equations (AE-AE, PS-PS on-site) not extracted here — in
  Paier 2006/2005 + Kresse-Joubert 1999.

## Open questions
1. Explicit equation form of VASP's one-center exchange integrals (Paier JCP 124/122) — the K_a formula.
2. A published full-HF (unscreened) NC-vs-USPP-vs-PAW Si EOS quantifying the augmentation/one-center error.
3. Si-specific magnitude of the frozen-core EXX effect (Harl-Schimka-Kresse studied it; magnitude not extracted).
4. Does the latest QE still treat PAW EXX = USPP via q_lm with no one-center operator?
