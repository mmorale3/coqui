# ABINIT exchange potential: GW Sigma_x vs hybrid-DFT Fock — they are DIFFERENT

Date: 2026-07-21. ABINIT 10.6.7 source at `~/abinit_build/abinit-10.6.7` (rusty).

Context: we are reproducing ABINIT's **pure exchange potential** (100% Fock, NO DFT
XC contribution) in CoQuí. ABINIT computes the on-site (one-center) exact exchange in
**two structurally different ways** depending on whether it is the GW self-energy
Sigma_x or the hybrid-DFT / HF Fock operator. This note records both and the mapping to
CoQuí's paths.

The **smooth (plane-wave) Fock** term — Poisson solve of the pseudo pair density
psi~_i psi~_j (+ compensation) — is the *same* in both paths. The **on-site** term is
where they differ.

--------------------------------------------------------------------------------
## Path 1 — GW Sigma_x  (m_sigma_driver → m_sigx → paw_rho_tw_g → m_pawpwij)

On-site added as a **plane-wave oscillator** on the G-sphere:

    rhotwg(G) += Sum_atm e^{-i(q+G).x_atm} Sum_{ij} P*_{n,i} P_{m,j} mqpgij(G, ij)

with the form factor (m_pawpwij.F90, `pawpwij_t%mqpgij(2,npw,lmn2)`)

    mqpgij(G,ij) = <phi_i|e^{-i(q+G)r}|phi_j> - <phi~_i|e^{-i(q+G)r}|phi~_j>

built in `paw_mkrhox` from the radial integrals `pwff_spl` (paw_mkrhox_spl) x realYlm(q+G).

Two methods (`m_sigma_driver.F90:495`, selected by `dtset%pawoptosc`):
- **method 1 = Arnaud-Alouani (DEFAULT in sigma)**:
      ff^L_ij(q) = INT_0^ra j_L(2*pi*q*r) [phi_i.phi_j - phi~_i.phi~_j](r) dr
  = the FULL AE-PS pair density transform.
  ABINIT's own comment: *"does not describe correctly the multipoles of the AE charge
  density if low cutoff on G"* — i.e. CUTOFF-SENSITIVE.
- **method 2 = Shishkin-Kresse**:
      ff^L_ij(q) = q_ij^{LM} INT_0^ra j_L(2*pi*q*r) g_L(r) r^2 dr
  = multipole moment x shape-function transform = the COMPENSATION-charge form.
  "Better description of multipoles of AE charge."

Content: **VALENCE-VALENCE only** (built from the valence Cprj). No core-valence.

CoQuí analogue: the **Q-augmentation** in v_x (`build_paw_aug_pair_factor` +
`evaluate_Q_IJ_at_K_fast`), with
- `build_qrad_tab_full_aeps`  ==  ABINIT Arnaud (method 1)  == "shape-restored"
- `build_qrad_tab` (qfuncl)   ==  ABINIT Shishkin (method 2) == "compensated"

--------------------------------------------------------------------------------
## Path 2 — Hybrid-DFT / HF Fock  (m_fock / fock_getghc + m_paw_denpot → pawdijfock)

On-site added as a **real-space Dij Fock matrix** (libpaw m_pawdij.F90 `pawdijfock`):

    Dijfock_vv(i,j) = - Sum_{k,l} rho_kl * eijkl(il, kj)          (valence-valence)
    Dijfock_cv(i,j) = core-valence term from pawtab%ex_cvij       (core-valence)
    paw_ij%dijfock  = Dijfock_vv + Dijfock_cv                     (both included)

`eijkl = pawtab%eijkl` (or `eijkl_sr` for HSE range-separation) are the EXACT one-center
exchange Coulomb integrals, built in `m_paw_init.F90` (pawinit) as

    eijkl_ijkl = vh1_ijkl - Vhat_ijkl - B_ijkl - C_ijkl
               = <phi_i phi_j | v | phi_k phi_l>^AE
                 - <phi~_i phi~_j + nhat | v | phi~_k phi~_l + nhat>^PS

via a real-space radial Poisson solve (`poisson(phiphj,...)`, `poisson(tphitphj,...)`),
Gaunt-coupled, with the compensation-charge (q_ijL, vhatijl, intvhatl) terms removing
the nhat double-count. EXACT (no plane-wave cutoff).

`ex_cvij` (core-valence) is read from the PAW-XML `<exact_exchange_X_matrix>` /
frozen-core data — it has NO analogue in the GW Sigma_x path.

hyb_mixing = 1 for pure exchange (HF); the eijkl contraction carries the mixing factor.

CoQuí analogue: the **deltaC one-center** term in v_x (`v_x_paw.hpp:586+`),
`deltaC(I,J,K,L) = <phi_I phi_J|v|phi_K phi_L>^{AE-PS}` == ABINIT eijkl,
PLUS `ex_cvij` core-valence (commit b9e9a54).

--------------------------------------------------------------------------------
## THE DIFFERENCE (answer to "identify the differences")

| | GW Sigma_x | Hybrid / HF Fock |
|---|---|---|
| on-site representation | plane-wave oscillator mqpgij(G) on G-sphere | real-space Dij via exact radial Coulomb eijkl |
| radial form | INT [phiphi - phi~phi~] j_L(Kr) dr (Arnaud) | <phiphi|v|phiphi>^AE-PS (Poisson) |
| cutoff behavior | CUTOFF-SENSITIVE (ABINIT's own warning) | EXACT (no G cutoff) |
| core-valence | NONE (valence Cprj only) | INCLUDED (ex_cvij) |
| ABINIT code | m_pawpwij / paw_rho_tw_g | m_pawdij pawdijfock / m_fock |
| CoQuí analogue | Q-augmentation (build_qrad_tab*) | deltaC + ex_cvij |

Consequence: the two ABINIT exchange operators are NOT numerically identical for PAW.
The **hybrid/HF Fock on-site (eijkl) is the rigorous full one-center exact exchange**
(includes core-valence, no cutoff); the GW on-site (mqpgij) is a plane-wave approximation
that is valence-only and cutoff-limited.

## Mapping for the numerical comparison

To reproduce ABINIT's **pure-Fock exchange potential** (this task), CoQuí must use:
    smooth-Fock(psi~psi~ + compensation)  +  deltaC one-center  +  ex_cvij core-valence
i.e. the **COMPENSATED** v_x pass (Q-aug=compensated + deltaC), NOT the shape-restored
(Arnaud) pass. The shape-restored/Arnaud pass corresponds to ABINIT's **GW Sigma_x**,
which is a different (cutoff-limited, valence-only) operator.

Earlier si222 numbers (bare, finite-size off, 100% exchange):
- CoQuí compensated (Q-comp + deltaC): -1.36866 a.u.   <-- compare to ABINIT HF Fock
- CoQuí shape-restored (Q-Arnaud):     -1.36528 a.u.   <-- compare to ABINIT GW Sigma_x
- prior "ABINIT full onsite" target:   -1.316447 a.u.  (need to confirm which operator it came from)

RESOLVED (2026-07-24/25, plan C4): -1.316447 was determined to be the **GW
Sigma_x (Arnaud) operator** number, not HF Fock. The HF-side match is settled by
the kernel + energy accounting below (deltaC == eijkl to 5.5e-5 rel, ex_cvij
machine-identical, onsite vv+cv energies identical; the smooth residual closed
separately via fock_icutcoul=3 — bare-Coulomb smooth exchange matches ABINIT to
~uHa). REMAINING (cluster campaign, with the B-tests assets): regenerate the
ABINIT GW Sigma_x (pawoptosc=1/Arnaud, iszoff, ecutsigx recorded) on the cmp
si222 mf and compare against CoQui shape mode AT THE SAME augmentation G-cutoff
(the Arnaud oscillator is cutoff-sensitive; CoQui uses the fft_mesh_aug
inscribed sphere -- match ecutsigx to it or truncate to ecutsigx).
Current-code CoQui baselines (2026-07-25, direct dense-grid v_x, ignore_g0,
TEST_CASE vexchange_mode_energies): qe_lih222_paw_hf E_X(moment+deltaC)
= -1.64406506 / E_X(shape) = -1.64395244 (split +1.13e-4); local qe_si222_paw
fixture -1.31194760 / -1.31187642 (split +7.1e-5 -- small because this fixture
has fft_mesh_aug == fft_mesh = 36^3, i.e. a coarse augmentation sphere truncates
both modes equally; the mode split only develops on a genuinely dense aug mesh;
also a different cell from the rusty cmp mf, so NOT comparable to -1.316447).

--------------------------------------------------------------------------------
## DIRECT KERNEL COMPARISON (2026-07-21) — apples-to-apples, exact

Instrumented ABINIT (m_paw_init.F90 pawinit, env ABI_DUMP_FOCKKERNEL) to dump pawtab%eijkl
(vv) + pawtab%ex_cvij (cv) + indlmn/indklmn maps. Ran pure HF (ixc 40, alpha=1, "no xc
applied") on si222. Compared element-by-element to CoQuí Onecenter/deltaC (8,8,8,8) +
ex_cvij (8,8) from mf.h5. Basis map is the IDENTITY: CoQuI ih (0-based) == ABINIT
ilmn-1 (both order s1,s2,p(m=-1,0,1)x2). ABINIT eijkl stored upper-triangle only
(klmn_ij<=klmn_kl). Scripts: /tmp/patch_pawinit.py, /tmp/compare_kernels.py; ABINIT dump
~/ceph/CoQui/abinit/cmp/si222/hf/abi_fockkernel_type1.dat; HF binary rebuilt at
~/abinit_build/abinit-10.6.7/build (m_paw_init.F90.bak = original).

RESULT:
- **CV (ex_cvij): IDENTICAL to machine precision** — ratio 1.000000000 every element,
  max|diff| = 8.9e-16. (Same XML <exact_exchange_X_matrix> source; converter transcribes
  exactly.)
- **VV (deltaC vs eijkl): agree to 5.5e-5 relative** over all 4096 elements (mean|abs
  diff| 2.8e-5), ratio ~1.00000 (NO normalization/sign/convention factor). The ~1e-5 is
  pure radial quadrature (CoQuI compute_deltaC cumulative-trapezoid double integral vs
  ABINIT poisson ODE + simp_gen); same integral <phiphi|v|phiphi>^AE - <tphitphi+nhat|v|
  ..>^PS, same Hartree units.

CONCLUSION: CoQuI's one-center exact-exchange POTENTIAL == ABINIT's hybrid/HF Fock
one-center matrices (cv exact, vv quadrature-identical). The one-center exchange operator
is reproduced. Remaining Efock-vs-CoQuI total gap (-0.811 Ha) is finite-size + smooth-Fock
bundling + the cv energy term, NOT the one-center kernels. ABINIT pure HF Efock = -2.179474
Ha (total). CoQuI vv exchange (no cv, no finite-size, ignore_g0) = -1.368604 Ha.

--------------------------------------------------------------------------------
## EXACT ENERGY ACCOUNTING (2026-07-21) -- Si 2x2x2, size effects off, IDENTICAL orbitals

Verified orbitals identical: CoQuI /Orbitals/eigval(Gamma) = ABINIT hf.abi DS1 PBE
eigenvalues to 5 digits ([-0.23437, 0.21250x3, 0.30272x3, 0.34412]). So density matrix
is the same (also confirmed: onsite cv energy from CoQuI ex_cvij x ABINIT pawrhoij =
ABINIT's exactly, 0.0).

KEY correction: ABINIT printed "Efock" = energies%e_fock0 = 1/2 sum_occ f <psi|V_Fock|psi>
is the SMOOTH plane-wave Fock ONLY (onsite Dij applied separately). Instrumented ABINIT:
env ABI_FOCK_NOSINGULAR (barevcoul: zero q+G=0 term = ignore_g0), ABI_FOCK_NONHAT
(m_fock_getghc: zero nhat rho12 in pair density), ABI_DUMP_FOCKENE (print efock/efockdc).

Term-by-term (finite-size off, Ha):
  onsite cv  (ex_cvij)     ABINIT -0.521220  CoQuI -0.521220  diff 0        (identical kernel)
  onsite vv  (eijkl/deltaC)ABINIT -0.015211  CoQuI -0.015211  diff 0        (kernel 5e-5)
  compensation nhat/Q-aug  ABINIT +0.037673  CoQuI +0.032693  diff -0.00498 (aug grid repr.)
  smooth E_x[psi~psi~]     ABINIT -1.464536  CoQuI -1.386097  diff -0.07844 (smooth FFT grid/cutoff)
  finite-size (q+G=0)      ABINIT -0.752611  CoQuI  0 (ignore_g0)           (matched off)
  (ABINIT nhat = Efock(nhat)-Efock(nonhat) = -1.426863-(-1.464536); Efock=smooth only.)

RESULT: the exact one-center exchange (vv+cv) is IDENTICAL in matrix AND energy. The only
residuals (78 mHa + 5 mHa) are in the SMOOTH plane-wave part -- the psi~psi~ pair-density
exchange grid/cutoff (ABINIT uses dense pawecutdg=50; CoQuI its smooth mesh) + compensation
representation. NOT the one-center exchange. Orbitals byte-identical so the 78 mHa is grid,
not physics. Scripts /tmp/patch_{pawinit,fock_energy,rhoij,nonhat,fockband}.py, compute_onsite_energy.py.
ABINIT backups: m_paw_init.F90.bak, m_barevcoul.F90.bak, m_fock_getghc.F90.bak (m_paw_denpot
patched, no .bak -- 3 env-gated blocks).

PER-BAND PIN-DOWN (2026-07-21): instrumented ABINIT eigen_ikpt dump (m_fock_getghc:1398,
env ABI_DUMP_FOCKBAND) -> per-band smooth Fock at Gamma vs CoQuI Per-state Sigma_x^vv.
Ratios CoQuI/ABINIT are BAND-DEPENDENT (0.918, 0.945, 0.924, 0.947 for bound bands 0-3;
0.993, 0.966 for higher bands 4-5), NOT a uniform normalization. Worse for more-bound
(localized, high-G) bands, near-perfect for higher bands = classic high-G/grid-resolution
signature. ABINIT consistently MORE negative (captures more high-G). CoQuI smooth Sigma_x
here = THC path on 18^3 smooth grid (+ compression) -> under-resolves the high-G pair-density
tail. => the 78 mHa is smooth-pair-density GRID CONVERGENCE (closes on CoQuI's dense 48^3
path), not the exact-exchange operator. ABINIT per-band (Ha, Gamma, nonhat=raw psi~psi~):
b0 -0.46734, b1-3 -0.34071, b4-5 -0.19095. CoQuI (Ha): b0 -0.42885, b1 -0.32193, b2 -0.31468,
b3 -0.32270, b4 -0.18968, b5 -0.18442.

---

UPDATE (2026-07-25/26, C4 closure + E2 retraction): the cluster regeneration
was done. ABINIT GW Sigma_x regenerated on the cmp mf (pawoptosc=1, ISZ off):
ecutsigx=25 reproduces -1.316447 exactly (provenance confirmed), converged
value -1.31747. First comparison on the then-current mf gave a "-47.6 mHa
Fock-vs-Arnaud operator difference" at matched aug cutoff — that number is
RETRACTED: the mf carried the abinit2coqui real_ylm odd-m Condon-Shortley
sign bug (fixed 3956b45). On the corrected mf: CoQui E_X(shape) = -1.3175731
vs converged GW Sigma_x -1.31747 → agreement to 0.10 mHa; E_X(moment+deltaC)
= -1.3178007 (shape-moment split -0.23 mHa on this system). Both operator
identities (moment ≡ pawdijfock Fock, shape ≡ Arnaud GW Sigma_x) are
therefore confirmed numerically on matching sides. C4 CLOSED.
