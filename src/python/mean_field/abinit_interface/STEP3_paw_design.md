# abinit2coqui Step 3 — PAW `/Hamiltonian` design spec

The payoff step: emit an augmented PAW `/Hamiltonian` so CoQuí does RPA/GW on an ABINIT PAW
mean field — bringing in ABINIT's **correct one-center exact exchange** (`exact_exchange_X_matrix`
= `deltaC`/K_a), which QE mishandles (see project_abinit_paw_hybrid_eos: QE PAW-EXX gives Si a0
+0.15–0.18 too large; ABINIT is VASP-correct).

## ACTUAL target schema (from a real QE-PAW pw2coqui.h5: ~/ceph/CoQui/paw/LiH/OUT/pwscf.coqui.h5)

The reference LiH PAW h5 has ONLY `/Hamiltonian/paw` (NO `/Species`, NO `/Onecenter`, NO `qq_nt`,
NO `deeq` — CoQuí recomputes deeq/qq_nt and falls back if Species is absent). So the minimal
PAW `/Hamiltonian/paw` to write (LiH: 2 species, nhm=8, nkb=10, ngm dense=15593):
  atomic_id(nat), proj_per_atom(nat), projector_offset(nat), npw(nk),
  miller_g(ngm,3), miller_k{ik}(npw_k,3), projector_k{ik}(nkb,npw_k) CPLX,
  dion(nsp,nhm,nhm), ijtoh(nsp,nhm,nhm) int,
  pp_local_component(ngm) CPLX, scf_local_potential(nspin,npol^2,ngm) CPLX,
  vxc(nspin,npol^2,ngm) CPLX, vxc_with_nlcc(nspin,npol^2,ngm) CPLX,
  augmentation_function_isp{nt}(nij, ngm) CPLX   [nij = nh*(nh+1)/2; the Q^IJ(G)],
  paw_one_center_coulomb_matrix_isp{nt}(nh^2, nh^2)  [4-index AE−PS Coulomb, NOT ABINIT's ns×ns
      exact_exchange_X_matrix — must COMPUTE from AE/PS partial waves via radial Slater+Gaunt;
      cross-check its exchange contraction vs ABINIT exx_X],
  paw_t(nsp) int, uspp_t(nsp) int  [both =1 for PAW].
All potentials + dion in RYDBERG (×2). projector_k = (nkb, npw). (These match the NC findings.)

## (older) What CoQuí's read_vnl_h5 needs for `pp_type="paw"` (running paw_tests build)

Everything the NC path needs (attrs, `miller_g`, `scf_local_potential`, `pp_local_component`,
`dion`, proj metadata, per-k `miller_k`/`projector_k`; potentials+dion in **Rydberg**, projector_k
shape `(nkb,npw)`) — PLUS the augmentation:
- `deeq` (nspin, nat, nhm, nhm) — SCF D matrix, **Rydberg** (CoQuí ×0.5).
- `qq_nt` (nsp, nhm, nhm) — per-species ⟨β_i|Q|β_j⟩ augmentation overlap (real, non-SO).
- `ijtoh` (nsp, nhm, nhm) int, 1-based — (ih,jh)→packed-ij map.
- `augmentation_function_isp{nt}` — Q^IJ(G) on the DENSE grid, complex (the qgm; qvan2 analog).
- `/Hamiltonian/Species/{nt}` — per-species radial PAW data (attrs mesh/kkbeta/lmax/nbeta/nh/zp;
  r, rab, lll, beta, dion, qqq, qfuncl, nhtolm/nhtol/indv, aewfc, pswfc; `/paw` pfunc/ptfunc/
  augmom/ae_vloc/...; `/Onecenter/deltaC(nh,nh,nh,nh)`; `/Core`).

## What the ABINIT PAW-XML provides (abinit_pawxml.py — DONE, validated)

`r` (log grid 2001), `states` (n,l,f,e; Si: 2×s + 2×p, l_max=1, nh=8), `phi_ae`/`phi_ps`/`proj`
(4×nr; phi_ae==phi_ps beyond paw_radius ✓), `shape_function` (bessel, rc), `paw_radius`,
`vbar` (blochl local ionic pot), core densities, `dij0` (kinetic_energy_differences 4×4),
**`exx_X`** (exact_exchange_X_matrix 4×4 = deltaC/K_a) + `exx_core_core` (−19.11 Ha).

## Recompute (the real work; ABINIT WFK+POT ready at converter_test_paw/)

| CoQuí field | from | method |
|---|---|---|
| β(k+G), miller_k | proj radial + WFK | init_us_2 analog (Step-2 build_beta_k, r-weight TBD for PAW proj) |
| qq_nt ⟨β|Q|β⟩ | phi_ae,phi_ps | ∫(φ_iφ_j − φ̃_iφ̃_j) r²dr per (l); moments = augmom |
| Q^IJ(G) qgm | qijl radial + shape_fn | qvan2 analog: compensation charge n̂_ij(r)=Σ q_ij^L g_L(r)Y_L → FFT to dense G |
| dion D^0_ij | dij0 + vbar + core | D^0 = kinetic_diff + ∫(v̄+v_H[ñ_Zc]) (φφ−φ̃φ̃) — check ABINIT Dij0 convention (may already be full) |
| deeq | dion + SCF | D^0 + ∫ V_eff(r) Q_ij(r) dr (V_eff from POT); = CoQuí compute_deeq_scf analog |
| /Onecenter/deltaC | exx_X | map exact_exchange_X_matrix (ns×ns, per-(l,n)) → deltaC(nh,nh,nh,nh) lm-expanded; sign/2 conv TBD |
| ijtoh, nhtol(m), indv | states | standard (l,n,m) enumeration |
| Species r,rab,beta,aewfc,pswfc,qfuncl | XML radial | resample/copy; rab = dr/di = a·d·exp(d·i) |

## Build order
1. PAW-XML parser (DONE), + emit /Hamiltonian/Species/{nt} radial block + β/dion/potentials.
2. qq_nt + Q^IJ(G) augmentation + ijtoh → CoQuí reads /Hamiltonian/paw without abort.
3. deeq (SCF from POT V_eff). Validate: overlap ⟨ψ̃|S|ψ̃⟩=1 (PAW pseudo orbitals are NOT
   PW-orthonormal — S restores it), ⟨ψ|H|ψ⟩≈eig.
4. /Onecenter/deltaC ← exx_X. Run CoQuí RPA/HF; confirm augmented exact exchange is ABINIT-correct
   (Si a0 ~10.3, not QE's 10.44) — THE payoff.

## CONVENTIONS PINNED (2026-07-10, empirical from Si.xml)
- **PAW-XML partial waves are stored as R(r)** (the radial function; full wave = R(r)·Y_lm),
  NOT u=r·R. Proof: phi_ae[3s](r→0)=8.08 finite, phi_ae[3p](r→0)=0 (~r^l); and
  **∫ R² r² dr = 1.0000** for the occupied 3s/3p (AE valence norms). So ALL radial integrals
  use the **r² dr measure**.
- Multipole moments (radial part, before Gaunt): **q_ij^L = ∫ (R_i R_j − R̃_i R̃_j) r^(L+2) dr**.
  Computed for Si (sensible): q0[3s,3s]=−0.049, q0[3p,3p]=−2.385, unbound-pair moments large
  (weighted by small occupancy). L allowed by |l_i−l_j|≤L≤l_i+l_j, L+l_i+l_j even.
- Full compensation charge: n̂_ij(r) = Σ_L q_ij^L Gaunt(l_i m_i, l_j m_j, L M) g_L(r) Y_LM(r̂),
  with g_L the (bessel-type here) shape function normalized ∫ g_L(r) r^L r² dr = 1.
  Q^IJ(G) = FT of n̂ onto the dense grid (qvan2 analog) — NEXT to implement.
- radial grid rab = dr/di = a·d·exp(d·i) (a,d from radial_grid; but trapz on r works directly).
- proj (projector_function) is ALSO R-like (p̃(r)); for β(k+G) transform, since Step-2 psp8
  used r^1 (psp8 chi was r-weighted), PAW proj=R needs r^2 in the bessel transform — VERIFY
  vs a QE-PAW β/vkb or the H0/S check.

## Key conventions to pin (like Step-2, against the running build)
- PAW proj radial: is it r-weighted (chi=r·p) like psp8? → sets the β radial-transform r-power.
- ABINIT `kinetic_energy_differences` = full D^0_ij or kinetic-only? (QE dion is full dion0.)
- deltaC sign/factor: exact_exchange_X_matrix vs CoQuí deltaC (deltaC stored in Ha, e2² audit —
  see project_paw_deltaC_e2_squared: pw2coqui divides by e2²=4).
- augmentation Q normalization + the dense-grid qvan2 (moments q_ij^L, shape g_L).
Validate each against a QE-PAW pw2coqui.h5 for the same Si (cross-code convention map).

## STATUS 2026-07-10 (evening): LOCAL EMIT LAYER COMPLETE + VALIDATED vs QE ref

Validation oracle: `tests/unit_test_files/qe/si_kp222_paw/pwscf.coqui.h5` (in-repo QE-PAW ref,
kjpaw Si) has BOTH radial inputs (Species/nt0) AND QE outputs (augmentation_function_isp0,
Onecenter/deltaC) -> reproduce QE arrays from same inputs to (near-)machine precision, no rusty.

Modules (numpy-only kernels; validate_*.py need h5py, local venv /tmp/pawenv):
- `paw_qvan.py`  Q^IJ(G): ylmr2 + real_gaunt(ap) + qrad Hankel + qvan2.  validate_qvan.py = **6.9e-11**.
- `paw_deltaC.py` one-center nh^4 AE-PS Fock kernel.  validate_deltaC.py = **4.35e-5** (direct
  multipole integral vs QE ODE hartree; corr=1.0). deltaC = K_ae-K_ps (no /e2; v_lm carries e2).
- `paw_radial.py` channel maps(indv/nhtol/nhtolm/ijtoh), moments q_ij^L=INT(pfunc-ptfunc)r^L dr=augmom,
  qqq=augmom[0], qq_nt(gated by nhtolm), build_qfuncl(moment*shape).  validate_paw_frontend.py PASS.
- `abinit_paw_hamiltonian.py` build_paw_augmentation + write_paw_augmentation + write_species_block.
  validate_paw_emit.py round-trip (QE radial -> h5 -> reread vs QE) PASS worst 4.35e-5.

SCHEMA (confirmed from ref h5): /Hamiltonian pp_type="paw"; /Hamiltonian/paw has the SAME NC-like
datasets (miller_g, scf_local_potential, pp_local_component, per-k miller_k/projector_k, npw,
proj_per_atom, projector_offset, atomic_id, dion) PLUS ijtoh, qq_nt, deeq(nspin,nat,nhm,nhm),
augmentation_function_isp{nt}(nij,ngm)CPLX, vxc, vxc_with_nlcc. NO paw_t/uspp_t datasets (old design
was wrong). One-center = /Hamiltonian/Species/nt{i}/Onecenter/deltaC(nh,nh,nh,nh) (NOT
paw_one_center_coulomb_matrix). augmentation_function name uses 0-based species index.

## REMAINING (rusty, needs real Si.xml + WFK + POT) -- task 4
1. ABINIT adapter: abinit_pawxml parse -> normalized species dict. Build u_ae=r*phi_ae, u_ps=r*phi_ps
   (XML stores R). kkbeta = index of paw_radius. **shape_by_L from Si.xml `bessel` shape -- VERIFY
   the exact q1,q2 two-Bessel combo vs the tabulated shape (paw_radial.shape_bessel is a placeholder)**.
2. Shared NC-like block from WFK/POT (reuse abinit_hamiltonian pattern): **PAW beta(k+G) radial r-power
   (psp8 used r^1 since chi=r*beta; PAW proj is R -> maybe r^2) -- verify vs H0/S or a QE-PAW vkb**;
   scf_local_potential + pp_local_component from POT; dion = D^0 (check ABINIT dij0 = kinetic-only vs full).
3. deeq: read from POT V_eff (INT V_eff Q_ij + D^0) OR rely on CoQui-native compute_deeq_scf (Species
   block present). deeq is OPTIONAL in reader (falls back to dvan, "won't match QE") -> want it for eigs.
4. vxc, vxc_with_nlcc from ABINIT DEN+libxc or prtvxc.
5. Wire usepaw=1 into abinit2coqui.convert(); iterate CoQui read (like NC's 4 fixes); RPA payoff a0~10.3.
