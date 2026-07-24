# abinit2coqui converter — design plan

Goal: use ABINIT (mature PAW code) as CoQui's base DFT instead of QE, by writing a converter
that produces the SAME data CoQui's mean-field reader consumes — ideally without modifying
ABINIT source, and (per user) producing an h5 with the identical structure to pw2coqui.x's.
Motivation: QE's PAW-EXX is the weak link (see project_paw_augmented_e1e_eos_bug etc.).

## 1. What CoQui actually reads (the real contract — 3 inputs, not 1)

CoQui's QE MF backend (src/mean_field/qe/{qe_interface.cpp, qe_readonly.cpp, qe_system.hpp})
reads THREE things from a QE run:
  (i)   `data-file-schema.xml`  → system metadata (npwx, ngm, ngms, nelec, nspin, npol,
        spinorbit, alat, ...) via qe_interface.cpp (`qes:espresso.output...`).
  (ii)  `prefix.save/wfc<ik>.hdf5` (or wfcup/wfcdw) → KS orbital coefficients AND the
        per-k G-vector list `/MillerIndices`. This is QE-NATIVE format (not pw2coqui).
  (iii) `prefix.coqui.h5` (pw2coqui output) → Hamiltonian / pseudopotential data.
So a converter must supply the equivalent of ALL THREE. NB pw2coqui does NOT write orbitals
(the `add_orbs` flag is vestigial/never acted on); orbitals come from QE-native wfc files.

## 2. Full pw2coqui.h5 schema (the exact target for input iii)

/System  [attrs: number_of_atoms, number_of_species, number_of_spins, number_of_polarizations,
    number_of_elec, noinv, lspinorbit, nuclear_energy(Ewald/e2), qe_ehart, qe_etxc, qe_vtxc, qe_epaw]
  lattice_vectors(3,3 Bohr), reciprocal_vectors(3,3), atomic_id(nat,0-based), atomic_positions(3,nat Bohr),
  species(nsp strings), kpoint_weights(nk)
  /System/BZ  [attrs number_of_kpoints, number_of_kpoints_ibz]  kp_grid(3), kpoints(3,nk cart·tpiba)
    /System/BZ/Symmetries [attr number_of_symmetries]  s{i}: R(3,3), ft(3)
/Orbitals [attrs number_of_spins, npol, number_of_kpoints(_ibz), npwx, number_of_bands, ecutrho]
  npw(nk), fft_mesh(3 smooth), fft_mesh_aug(3 dense), eigval(nbnd,nk,ns Ha), occ(nbnd,nk,ns)
  [orbital coeffs NOT here currently — would be the add_orbs extension point]
/Hamiltonian [attr pp_type=paw|uspp|ncpp]
  /Hamiltonian/{pp_type} [attrs nspins, npol, nk, nat, nsp, total_num_of_proj(nkb),
      max_proj_per_atom(nhm), ngm, lspinorbit_nl, max_npw, lspinorbit_loc]
    proj_per_atom(nh), projector_offset(ofsbeta), ijtoh, atomic_id, npw
    dion(nhm,nhm,nsp)[dion_so], qq_nt(nhm,nhm,nsp)[qq_so], deeq(nhm,nhm,nat,nspin)[deeq_nc]
    augmentation_function_isp{nt}: Q^IJ(G) qgm(ngm_g,nij)   ← qvan2 on dense G-grid
    miller_g(3,ngm_g), scf_local_potential(ngm,npol²,ns), pp_local_component(ngm)=Vloc,
    vxc_with_nlcc(ngm,..), vxc(ngm,..)
    miller_k{ik}(3,npw_g[ik]) ← per-k wfc G-grid; projector_k{ik}(npw_g[ik],nkb) ← β_i(G) (vkb, init_us_2)
  /Hamiltonian/Species/{nt} [attrs species_kind, mesh, kkbeta, lmax, lmax_rho, nbeta, nh, zp]
    r, rab, lll, kbeta, beta(radial projectors), dion(nbeta²), nhtolm, nhtol, indv, [nhtoj]
    qqq(nbeta²), q_with_l, nqf, nqlc, qfuncl, [qfunc], aewfc, pswfc   ← AE/PS partial waves
    /paw: raug, iraug, lmax_aug, augshape; pfunc, ptfunc, augmom, ae_vloc, ae_rho_atc, oc,
          vloc_ps, rho_atc_ps, [pfunc_rel, aewfc_rel]
    /Onecenter/deltaC(nh,nh,nh,nh) ← AE−PS Fock kernel K_a (= ABINIT's <exact_exchange_X_matrix>!)
    /Core (GIPAW): ncore_orbitals; n, l, ae_wfc(mesh,ncore)

## 3. Preliminary QE→ABINIT quantity mapping (ABINIT side pending research agent)

Directly in ABINIT files:
  - orbitals ψ_nk(G), G-vectors, eigenvalues, occ, k-points, weights, symmetry, structure → WFK netcdf (abipy).
  - PAW radial data (partial waves φ/φ̃, Dij0, Qij/qijl, core density, EXACT-EXCHANGE X matrix
    = deltaC/K_a!) → PAW .xml dataset (parse directly). ABINIT's <exact_exchange_X_matrix> IS
    the on-site AE−PS exchange kernel CoQui wants — a DIRECT match, precomputed in the dataset.
  - Vloc, Vxc, Vscf, density → POT/DEN netcdf (or recompute vxc from density + libxc).
Likely need recomputation in the converter (from radial data + wavefunctions):
  - projector_k β_i(G) per k (QE's vkb/init_us_2) — build from PAW projector radial fns + Ylm.
  - deeq (SCF D matrix), and cprj ⟨p_i|ψ̃⟩ (becsum ingredient) — ABINIT computes cprj internally;
    OPEN: can ABINIT print cprj without source mod (prtcprj?), else recompute from β(G)·ψ(G).
  - qgm Q^IJ(G) on the dense grid — build from qijl (radial) via the qvan2 analog.

## 4. Architecture options
(A) Full mimic: converter writes QE-format data-file-schema.xml + wfc<ik>.hdf5 + pw2coqui.h5,
    so the EXISTING CoQui qe backend reads it unchanged. No CoQui or ABINIT source changes.
    Heaviest (must reproduce QE's XML + wfc h5 layout exactly) but zero code changes elsewhere.
(B) New CoQui `mf_abinit` backend (src/mean_field/abinit/) reading ABINIT WFK + a pp-only
    abinit2coqui.h5. Cleaner; needs a CoQui backend (CoQui's MF layer is already pluggable:
    qe/pyscf/model). Converter only emits the pp h5 (+ ABINIT gives orbitals natively).
(C) Single self-contained coqui.h5: complete the `add_orbs` path so ONE h5 holds system+orbitals+
    Hamiltonian; converter emits that; CoQui reads orbitals from /Orbitals. Needs a small CoQui
    reader extension (read orbitals from coqui.h5) + the converter. Most portable long-term.

## 5. Open questions (for the ABINIT research agent)
- WFK netcdf: exact variable names for coeffs, kg_k (G-vectors), eig, occ; abipy accessors.
- cprj ⟨p|ψ̃⟩: printable without source mod? (prtcprj / a netcdf / abipy) — else recompute.
- Dij (SCF) and the compensation charge on the grid: exposed, or recompute?
- G-vector ordering / PW normalization conventions vs QE (Miller indices, 1/√Ω factors).
- Existing converter to model: a2y (Yambo), abi2bgw (BerkeleyGW) — do they support PAW?

## 6. ABINIT-side mapping — RESOLVED (research agent, 2026-07-09)

VERDICT: ABINIT exposes EVERYTHING needed via netCDF/ETSF-IO + PAW-XML, readable with abipy,
with NO ABINIT source modification. The only recomputed quantities (cprj, SCF Dij, n̂) are
exactly the ones pw2coqui already recomputes from QE. No fundamental showstopper.

| CoQui/QE quantity | ABINIT source | how |
|---|---|---|
| ψ_nk(G) coeffs | WFK netCDF `coefficients_of_wavefunctions` (nsppol,nkpt,mband,nspinor,mpw,2) | abipy WfkFile.get_wave / WFK_Reader.read_ug |
| per-k G-vectors (Miller) | WFK `reduced_coordinates_of_plane_waves` (=kg_k); npw=`number_of_coefficients` | abipy read_gvecs_istwfk |
| eigenvalues, occ, kpts, weights, spin | WFK/GSR (ElectronBands) | abipy .ebands |
| symmetry (R=symrel, ft=tnons, AFM) | WFK/GSR header `reduced_symmetry_matrices/translations` | abipy AbinitSpaceGroup |
| structure (lattice, atoms) | WFK/GSR | abipy .structure |
| Vloc, Vxc, Vscf, density | POT/VXC/DEN netCDF (or recompute vxc from DEN+libxc) | abipy / ncreader |
| φ_i, φ̃_i, p̃_i, Dij0, qijl/shape, AE&PS core dens, v̄ | PAW-XML dataset (ESL v0.7) — plain XML, tabulated radial | parse XML directly |
| ONE-CENTER EXACT EXCHANGE (core-core, core-valence) = deltaC/K_a | PAW-XML dataset (spec includes exact-exchange 1-center integrals) | parse XML — DIRECT, correct, free |
| cprj ⟨p̃_i|ψ̃_nk⟩ (becp/becsum) | NOT on disk (no prtcprj) | RECOMPUTE from WFK cg + projectors (= pw2coqui's becp path; ABINIT's own PAW-GW does this) |
| SCF Dij / rhoij | printed via pawprtvol (not clean netCDF) | RECOMPUTE from density+dataset (= CoQui compute_deeq_scf) |
| n̂ compensation charge on grid | rebuild from rhoij+qijl, or PAWAVES.nc (pawprtwf=1) | recompute |
| β_i(G) per-k projectors (vkb/projector_k) | build from PAW-XML projectors p̃_i + Ylm (init_us_2 analog) | recompute in converter |

ABINIT run flags for the converter: prtwf=1, iomode=3 (netCDF WFK/DEN/POT), istwfk 1 (avoid the
half-sphere time-reversal packing — a real gotcha; abipy WFK_Reader has the unpack reference).
CONVENTIONS to match: PW normalization Σ_G|u_k(G)|²=1 (pseudo, PAW norm restored on-site); G-sphere
|k+G|²/2≤ecut ordered by kinetic energy; ETSF-IO spec arXiv:0805.0192; PAW-XML ESL v0.7 (Hartree units).
Convention differences ABINIT-PAW-XML vs QE-UPF-PAW = the real converter engineering (all tabulated,
no reconstruction). KSS file is a DEAD END for PAW (outkss rejects multi-projector) — target WFK.

MODELS to copy: abipy WFK_Reader (abipy/waves/wfkfile.py) = exact netCDF var names + istwfk unpack;
Yambo a2y netCDF path = read-WFK structure (NC-only, no PAW); BerkeleyGW abi2bgw = read cg+den+vxc,
rewrite in target G-convention (NC-only). BEST reference = ABINIT's OWN internal PAW-GW driver
(optdriver 3/4, m_bethe_salpeter, gw_sigxcore): reads WFK, recomputes cprj from dataset projectors,
applies PAW on-site — precisely the CoQui pattern. NB no external ABINIT-PAW→MBPT converter exists
(Yambo/BGW/West are all NC-only) — we write it; ABINIT's internal GW proves the data suffices.

## 7. CONCRETE PLAN (recommended)

Language: PYTHON converter using abipy (reads netCDF WFK/GSR/DEN/POT) + an XML parser for the
PAW-XML dataset. No compiled code, no ABINIT/CoQui source changes for a first version.

ARCHITECTURE (revised per user 2026-07-09): use the H5-ONLY path via the existing `bdft` backend.
CoQui reads EITHER the pw2coqui h5 OR the QE xml (+pp.x/pw2bgw.x), NOT both — use the h5. And the
`bdft` mean-field backend (mf_source=bdft, src/mean_field/bdft/) reads ORBITALS from an h5, so no
QE-native wfc.hdf5 mimicry is needed. CRUCIALLY bdft is a FULL self-contained backend: bdft_readonly
calls hamilt::pseudopot_to_h5 to populate /Hamiltonian, and the pseudopot loader read_vnl_h5 reads
/Hamiltonian (PAW incl.) generically for any MF backend. So:

  → The converter emits ONE self-contained h5 `prefix.h5` and CoQui reads it entirely via bdft.

The h5 has three parts (all CoQui-native — NO external format to mimic):
  /System   — bdft schema (bdft_system.hpp save()): number_of_atoms/species/spins/dimensions,
              nelec, species, at_ids, at_pos, latt, recv, madelung, enuc, efermi, k_weight, +BZ
              symmetry (bz_symm.save: rotations, translations, kpts, kp_grid).
  /Orbitals — bdft schema (bdft_readonly make_wfc): wfc_ecut, wfc_fft_grid, wfc_ngm, miller_wfc
              (G-vectors), eigval[nspin,k,npol*band], occ, and the orbital coefficients (G-space,
              orb_on_fft_grid flag). fft_mesh, fft_mesh_aug, ecutrho.
  /Hamiltonian — pw2coqui schema (§2): dion,deeq,qq,qgm,projector_k,miller_k/g,vloc,vxc +
              Species/{nt}(radial pp, partial waves, /paw, /Onecenter/deltaC ← PAW-XML
              exact_exchange, /Core). Written to match what read_vnl_h5 + pseudopot expect.

This is Option B/C realized via an EXISTING backend — no QE-XML shim, no wfc.hdf5, no CoQui source
changes (bdft + pseudopot h5 path already exist). The converter is pure Python/abipy writing one h5.
KEY: match the bdft /System + /Orbitals schema (read bdft_system.hpp::save + bdft_readonly::make_wfc
for exact dataset names) and the pw2coqui /Hamiltonian schema. Confirm bdft handles PAW /Hamiltonian
(pseudopot_to_h5 h5-path + read_vnl_h5 are generic; validate on a PAW case).

Build order (incremental, testable against a QE run of the same system):
  Step 1: NCPP Si — converter reads ABINIT NC WFK, emits the 3 inputs, CoQui reads them, compare
          eigenvalues/orbitals to a QE NC run (validates G-ordering, istwfk unpack, normalization).
  Step 2: add the pp/Hamiltonian h5 (dion, projectors β(G) recomputed, vloc, vxc) — validate S/H0.
  Step 3: PAW — parse PAW-XML (partial waves, Dij0, qijl, core, exact_exchange→deltaC); recompute
          cprj/becsum, deeq, qgm; emit PAW coqui.h5; validate V_H/V_x/RPA vs the QE-PAW path.
  Step 4: run the RPA/EOS through CoQui on ABINIT-PAW and confirm the augmented EOS is correct
          (the payoff — ABINIT's one-center exchange is right where QE's is suspect).

Biggest wins: (a) ABINIT's PAW-XML gives the one-center exact-exchange kernel (deltaC/K_a)
DIRECTLY and correctly — the exact quantity QE mishandles; (b) no source changes to either code
for v1; (c) abipy does all the heavy netCDF lifting. Biggest efforts: istwfk unpacking + G-order/
normalization matching, and the ABINIT-vs-QE PAW convention mapping (qijl↔qfuncl, Dij, augmentation).
