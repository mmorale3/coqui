/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "hamiltonian/pseudo/pseudopot_type.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/mf_source.hpp"
#include "utilities/symmetry.hpp"

namespace hamilt
{

/**
 * Core-valence exchange treatment. This is NOT a user/toml option: it is
 * AUTO-DETECTED per PAW species from the mean-field h5 content in the pseudopot
 * constructor (stored in species_paw_t::cv_exchange) and then controls behaviour
 * during the calculation. Detection order: ex_cvij present -> fock; else a DFT
 * core-valence XC dataset present -> dft_xc (dataset TBD); else none (warned).
 */
enum class cv_exchange_e {
  fock,     ///< frozen core-valence exact-exchange (h5 ex_cvij found)
  dft_xc,   ///< DFT-based core-valence exchange-correlation (h5 dataset TBD; NOT YET IMPLEMENTED)
  none      ///< no core-valence exchange dataset found (contribution omitted; warned loudly)
};
enum class vv_compensation_e {
  moment,   ///< moment-restored compensation charge (Shishkin-Kresse; build_qrad_tab) (default)
  shape     ///< shape-restored full AE-PS partial-wave pair density (Arnaud; build_qrad_tab_full_aeps)
};

/**
 * PAW valence-valence exchange options (set from the toml). Either compensation
 * mode is computable from the SAME PAW h5 (qfuncl for 'moment'; aewfc+pswfc for
 * 'shape'), so this is a genuine methodological choice and is toml-controlled.
 * Scaffolding: choices not yet implemented ABORT in validate() with a clear
 * message; defaults reproduce the current validated behaviour. (Core-valence
 * exchange is NOT here — it is auto-detected from the h5; see cv_exchange_e.)
 */
struct paw_exx_options {
  /// Valence-valence on-site compensation charge. Default: moment-restored.
  vv_compensation_e vv_compensation = vv_compensation_e::moment;
  /// Maximum angular momentum L of the compensation-charge multipole expansion.
  /// Default -1 = full 2*lmax per species (lmax = highest partial-wave l), i.e.
  /// the complete augmentation with no truncation (cf. VASP LMAXPAW = 2*lmax).
  /// A value >= 0 caps the multipoles at L <= aug_lmax, trading accuracy for
  /// speed (cf. VASP LMAXFOCK for the Fock-exchange augmentation).
  int aug_lmax = -1;

  /**
   * Abort with a clear message for options declared here but not yet implemented.
   * `in_thc` selects the THC-ERI restriction: shape-restored compensation is not
   * yet available in the THC construction (its augmentation lives on the smooth
   * collocation grid, too coarse for the sharp AE-PS pair density); the direct
   * v_x path supports both compensation modes.
   */
  void validate(bool in_thc, std::string const& ctx = "paw_exx_options") const {
    if (in_thc)
      utils::check(vv_compensation != vv_compensation_e::shape,
        "{}: vv_compensation = 'shape' (shape-restored full AE-PS pair density) "
        "is NOT YET IMPLEMENTED for the THC ERI construction: the THC "
        "augmentation is built on the smooth collocation grid, which is too "
        "coarse to resolve the sharp AE-PS pair density (only a few percent of "
        "the on-site term is captured). Use vv_compensation = 'moment' (default) "
        "for THC. TODO: dense-grid THC augmentation + unit test. "
        "(The direct v_x path already supports 'shape'.)", ctx);
    utils::check(aug_lmax >= -1,
      "{}: aug_lmax = {} is invalid (must be >= 0, or -1 for no cap).",
      ctx, aug_lmax);
  }
};

/**
 * @class pseudopot
 * @brief Handler for the pseudopotential of a given mean-field object
 *
 * This class is responsible for computing and storing the pseudopotential for a
 * specified physical system. The system, including `nbnd`, `nkpt`, Brillouin zone info,
 * and "pseudopotential", are defined upon construction through a mean-field handler.
 *
 * Contributions of the pseudopotential can be evaluated by calling
 *
 *     pseudopot.add_Vpp(..., hpsi, Hij);
 *
 * where
 *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
 *   2. Contributions from the non-local potentials are added to the Hamiltonian
 *      in second quantization "Hij"
 *
 * In addition, this class also handle the Hartree potential for a given density matrix `nij`
 *
 *     pseudopot.add_Hartree(..., nij, ..., hpsi);
 *
 * in which the Hartree potential will be added to `hpsi`.
 *
 * @tparam MF_t - Type parameter for the mean-field object
 */
class pseudopot
{
  template<typename Arr>
  using sarray_t = typename math::shm::shared_array<Arr>;

  public:

  using mpi_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;

  template<typename MF_t>
  pseudopot(MF_t &mf, std::string const filename = "");

  ~pseudopot() {}

  pseudopot(pseudopot const&) = default;
  pseudopot(pseudopot &&) = default;
  pseudopot& operator=(pseudopot const&) = default;
  pseudopot& operator=(pseudopot &&) = default;

  // accessor functions
  pp_type_e pp_type() const { return ptype; }
  auto get_input_file_type() const { return input_file_type; }
  auto get_input_file_name() const { return input_file_name; } 

  // Read-only accessors for the host-resident augmentation data used by
  // v_h_paw and similar PAW/USPP utilities. The full state remains private
  // to keep the data model encapsulated.
  auto Pskna_view() const { return Pskna.local(); }
  auto Qij_view() const { return qq_nt_data.local(); }
  auto qgm_view() const { return qgm.local(); }
  // Static per-atom non-local D (nat, nhm*npol, nhm*npol): dion replica +
  // alpha_x * ex_cvij, built eagerly in read_vnl_h5 for USPP/PAW (plan I2).
  // Exposed read-only for tests/validation. Dummy (1,1,1) for NCPP.
  auto Dnn_atom_view() const { return Dnn_atom_static.local(); }
  nda::array<int,3> const& ijtoh_view() const { return ijtoh; }
  nda::array<int,1> const& ityp_view() const { return ityp; }
  nda::array<int,1> const& nh_view() const { return nh; }
  nda::array<int,1> const& ofs_view() const { return ofs; }
  nda::array<int,2> const& miller_g_dense_view() const { return miller_g_dense; }
  nda::array<double,2> const& atom_pos_cart_view() const { return atom_pos_cart; }
  long ngm_dense_get() const { return ngm_dense; }

  // Per-species PAW data (qfuncl, deltaC, partial waves, projector metadata,
  // GIPAW core orbitals). Populated from /Hamiltonian/Species/{nt}/ in
  // `read_vnl_h5`; entries for non-PAW species are mostly empty and should
  // be guarded with `sp.is_paw` (or `sp.is_uspp` for USPP-like data).
  struct species_paw_t {
    bool is_paw = false;
    bool is_uspp = false;
    int  mesh = 0;
    int  nbeta = 0;
    int  kkbeta = 0;
    int  nh = 0;
    int  lmax_aug = 0;
    double raug = 0.0;
    int  iraug = 0;
    nda::array<double,1> r;            // (mesh)
    nda::array<double,1> rab;          // (mesh)
    // Heavy radial tables — stored as views into per-node SHM owned by
    // pseudopot::paw_species_shm (one copy per node, not per rank). For
    // a 96-core node this saves ~3 GB/species over the previous per-rank
    // replication. See species_paw_shm_t below for the backing storage.
    nda::array_view<double,2> aewfc;     // (nbeta, mesh)
    nda::array_view<double,2> pswfc;     // (nbeta, mesh)
    nda::array_view<double,3> qfuncl;    // (2*lmax+1, nbeta(nbeta+1)/2, mesh)
    nda::array_view<double,4> deltaC;    // (nh, nh, nh, nh) — raw ke%k from QE
    nda::array_view<double,2> ex_cvij;   // (nh, nh) — frozen core-valence exact-
                                         //   exchange kernel (ABINIT ex_cvij =
                                         //   Σ_c⟨φ_I c|c φ_J⟩, from PAW-XML
                                         //   <exact_exchange_X_matrix>). Contracts
                                         //   LINEARLY with becsum (factor 1, no ½);
                                         //   one-center, frozen. Empty ⇒ omitted.
    // Core-valence exchange treatment for this species, AUTO-DETECTED from the
    // h5 content in the pseudopot constructor (fock if ex_cvij present, else
    // dft_xc if a DFT c-v XC dataset is present [TBD], else none). Controls the
    // frozen c-v exchange insertion into H0. See cv_exchange_e.
    cv_exchange_e cv_exchange = cv_exchange_e::none;
    nda::array_view<double,2> core_aewfc;// (ncore, mesh)
    // Angular momentum metadata (per-species slices of QE's global uspp
    // tables; ih ∈ [0, nh), nbeta ∈ [0, nbeta)).
    nda::array<int,1> lll;             // (nbeta) — l per beta projector
    nda::array<int,1> nhtol;           // (nh)    — l per ih
    nda::array<int,1> nhtolm;          // (nh)    — lm = l*(l+1)+m+1 (1-based) per ih
    nda::array<int,1> indv;            // (nh)    — beta-channel index per ih (1-based)
    // pfunc, ptfunc, augmom were previously loaded from QE's `paw` subgroup
    // but are derivable from aewfc/pswfc and qfuncl on demand:
    //   pfunc(I,J,r)  = aewfc(I,r) * aewfc(J,r)
    //   ptfunc(I,J,r) = pswfc(I,r) * pswfc(J,r)
    //   augmom(L,I,J) = ∫ qfuncl(L, ij(I,J), r) * r^(L+2) dr
    // No consumer ever reads them in steady state, so they are dropped
    // from the in-memory struct. Re-derive at the call site if needed
    // (e.g. by paw_onecenter when the deeq SCF path lands).
    nda::array<double,1> ae_vloc;      // (mesh) AE radial local potential
                                       // (= -Z/r screened by frozen AE core
                                       //  Hartree). Diagonal of T+V_AE_static.
    nda::array<double,1> ae_rho_atc;   // (mesh) AE atomic core density ρ_core_AE
    nda::array<double,1> vloc_ps;      // (mesh) PS radial local pseudopotential
                                       // — paired with ae_vloc, needed for the
                                       //   PAW static one-center D matrix
                                       //   (kinetic + ionic, frozen-core).
    nda::array<double,1> rho_atc_ps;   // (mesh) PS atomic core density ρ_core_PS
                                       //   (NLCC). Required for the dynamic
                                       //   one-center Hartree update
                                       //   (compute_deeq_scf).
    // GIPAW core orbitals (only when species was generated --with-gipaw)
    int  ncore_orbitals = 0;
    nda::array<double,1> core_n;       // principal qno (ncore)
    nda::array<double,1> core_l;       // l qno (ncore)  (real for QE convention)
  };

  // Per-node SHM storage backing the heavy `species_paw_t` fields. One
  // entry per species; aligned with `paw_species`. Each member is loaded
  // on global root, broadcast across nodes via internode_comm, and shared
  // within a node via the shared_array's shared-memory window. After
  // population the corresponding `species_paw_t` view fields are bound
  // to the `.local()` of the appropriate sarray_t below.
  //
  // Construction takes an mpi_t because `sarray_t` has no default
  // constructor; emplace each instance with a 1-element dummy size and
  // re-assign at load time once shapes are known (same pattern used for
  // Pskna/qq_nt_data/Dnn_atom_static in the pseudopot ctor init list).
  struct species_paw_shm_t {
    sarray_t<nda::array_view<double,2>> aewfc;
    sarray_t<nda::array_view<double,2>> pswfc;
    sarray_t<nda::array_view<double,3>> qfuncl;
    sarray_t<nda::array_view<double,4>> deltaC;
    sarray_t<nda::array_view<double,2>> ex_cvij;
    sarray_t<nda::array_view<double,2>> core_aewfc;

    explicit species_paw_shm_t(mpi_t& m)
      : aewfc(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1})),
        pswfc(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1})),
        qfuncl(math::shm::make_shared_array<nda::array_view<double,3>>(m, {1,1,1})),
        deltaC(math::shm::make_shared_array<nda::array_view<double,4>>(m, {1,1,1,1})),
        ex_cvij(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1})),
        core_aewfc(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1}))
    {}
  };

  // Read-only access to per-species PAW data. Empty entry for non-PAW
  // species — check `sp.is_paw` before consuming PAW-only fields.
  auto const& paw_species_view() const { return paw_species; }
  // wfc-G → dense-FFT-linear-index mapping (REMAPPED to dense mesh, NOT
  // the raw wfc_g.gv_to_fft() which is encoded on the wfc mesh).
  auto swfc_to_rho_view() const { return swfc_to_rho.local(); }

  void save(std::string fname, bool append = true);
  void save(h5::group& grp);

  std::shared_ptr<mpi_t> get_mpi_context() { return mpi; }

  /**
   * Add the contributions of norm-conserving pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{h} * psi where h is an arbitrary local operator
   *                  [output] (\hat{h} + vloc) * psi
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   */
  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  /**
   * Density-dependent overloads (plan I5): on top of the static assembly
   * above, the requested density-dependent terms are added under separate
   * boolean control, so callers (e.g. a future set_H) can mix direct and
   * ERI routes per term:
   *   - add_hartree: smooth V_H[n] applied to hpsi, plus (USPP/PAW) the
   *     one-center D built from V_loc+V_H (∫V·Q̂ + radial AE−PS Hartree)
   *     contracted into Hij. With add_hartree=false the static-only
   *     operator of the no-density overload is produced.
   *   - add_exchange: the band-pair-augmented exact-exchange matrix K
   *     (direct v_x route, SIGNED: F = H + K with K negative) accumulated
   *     into Hij. Host memory only.
   * The nii and nij overloads produce the identical operator for the same
   * physical density.
   */
  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& nii,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               bool add_hartree = true, bool add_exchange = false);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<4> auto const& nij,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               bool add_hartree = true, bool add_exchange = false);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<3> auto const& nii,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   math::nda::DistributedArrayOfRank<4> auto & Vij,
                   bool symmetrize=false);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nij     - [input] Density matrix (s, k, a, b)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<4> auto const& nij,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   math::nda::DistributedArrayOfRank<4> auto & Vij,
                   bool symmetrize=false);

  /**
   * Build the matrix elements of the exact-exchange operator K_{ij}(s, k_p)
   * for the diagonal-occupation density matrix.
   *
   * Unlike add_Hartree (which outputs an hpsi correction), exchange has no
   * representation as a local potential acting on a single orbital — the
   * augmentation cross terms involve band-pair-dependent projector overlaps.
   * add_exchange therefore writes the full K matrix directly.
   *
   * Sign convention: K is the SIGNED contribution to the Fock matrix
   * (F = H_core + V_H + K, with K already negative), matching CoQui's
   * existing set_fock / test_dft_eigenvalues convention.
   *
   * @param k_range - [input] k-point range (must cover the full IBZ)
   * @param nii     - [input] occupations (s, k_ibz, n), spin-summed for nspin=1
   * @param psi     - [input] orbitals on wfc-G grid (s, k_ibz, n, g)
   * @param Kij     - [output] K_{ij}(s, k_ibz, i, j) — written (overwritten)
   */
  void add_exchange(nda::range k_range,
                    nda::ArrayOfRank<3> auto const& nii,
                    math::nda::DistributedArrayOfRank<4> auto const& psi,
                    math::nda::DistributedArrayOfRank<4> auto & Kij);

  /**
   * Exact-exchange K_{ij}(s, k_p) from a FULL density matrix nij(s,k,a,b).
   * Generalizes add_exchange(nii) via the natural-orbital decomposition of
   * nij (see hamilt::paw::v_x / hamilt::v_x nij overloads). Same sign/scale
   * convention as the diagonal version. Currently requires a no-symmetry mesh
   * (nk_ibz == nk).
   *
   * @param nij - [input] density matrix (s, k_ibz, a, b)
   */
  void add_exchange(nda::range k_range,
                    nda::ArrayOfRank<4> auto const& nij,
                    math::nda::DistributedArrayOfRank<4> auto const& psi,
                    math::nda::DistributedArrayOfRank<4> auto & Kij);

  /**
   * Option A ("shape-restored" on-site exact exchange). When true, add_exchange
   * (both the nii and nij overloads) builds the PAW augmentation from the FULL
   * AE−PS partial-wave pair density (φφ − φ̃φ̃, via build_qrad_tab_full_aeps)
   * instead of the compensation charge, and drops the deltaC one-center
   * correction — reproducing ABINIT's phiphj−tphitphj oscillator, i.e. the
   * exact all-electron on-site exchange (fixes the ~0.05 Ha/Si compensated-PAW
   * under-count; see notes/paw_onsite_exchange_analysis.{md,pdf}). PAW species
   * only; USPP is unaffected. Default false = compensation charge + deltaC.
   */
  bool paw_exx_shape_restored = false;
  void set_paw_exx_shape_restored(bool b) {
    paw_exx_shape_restored = b;
    _exx_opts.vv_compensation = b ? vv_compensation_e::shape
                                  : vv_compensation_e::moment;
  }

  /**
   * PAW exact-exchange options (toml-controlled). Governs the core-valence
   * exchange source, the valence-valence compensation-charge model, and the
   * augmentation angular-momentum cutoff for the direct v_x / add_exchange path.
   * Set via set_exx_options() from the toml-parsing constructor (thc_reader_t).
   */
  paw_exx_options _exx_opts;
  paw_exx_options const& exx_options() const { return _exx_opts; }
  int aug_lmax() const { return _exx_opts.aug_lmax; }
  void set_exx_options(paw_exx_options const& o) {
    o.validate(/*in_thc=*/false, "pseudopot::set_exx_options");
    _exx_opts = o;
    paw_exx_shape_restored = (o.vv_compensation == vv_compensation_e::shape);
  }

  private:

  // mpi communicators
  std::shared_ptr<mpi_t> mpi;

  // pseudo type, default to ncpp and update in constructor
  pp_type_e ptype = pp_ncpp_t;

  // input type, needed for save
  mf::mf_input_file_type_e input_file_type = mf::xml_input_type;

  // input file, needed for save
  std::string input_file_name = "";

  // basic system info — dense (dfftp / augmentation) FFT grid.
  // Pseudopotential data (V_loc, V_eff, augmentation Q, dense miller_g)
  // all live on this grid; for NCPP it coincides with the smooth grid.
  nda::stack_array<int,3> fft_mesh_aug;
  long nnr_aug = 0;
 
  // reciprocal lattice vectors
  nda::stack_array<double,3,3> recv;

  // reciprocal lattice vectors
  nda::stack_array<double,3,3> lattv;

  // spin-orbit
  bool spinorbit_loc = false;
  bool spinorbit_nl = false;

  // number of spins 
  int nspin = 1;

  // number of polarizations
  int npol = 1;

  /* kpoints and symmetry properties */
  long nkpts = 0;
  long nkpts_ibz = 0;
  nda::array<double, 2> kpts;      // in cartesian coordinates
  nda::array<double, 2> kpts_crys; // in crystal coordinates
  nda::array<int, 1> kp_to_ibz;
  nda::array<bool, 1> kp_trev; // symmetry operations
  std::vector<utils::symm_op> symm_list; // symmetry operations
  nda::array<int, 1> kp_symm;   // index of symmetry operation that connects kpts/kpts_crys to IRBZ

  // type of pseudo for each atom
  nda::array<int,1> ityp;

  // number of projectors for each pseudo typle
  nda::array<int,1> nh;

  // index of first projector for each atom 
  nda::array<int,1> ofs;

  // qq
  memory::unified_array<ComplexType,1> qq;

  // Matrix elements between projectors and basis orbitals (in mf)
  sarray_t<nda::array_view<ComplexType,4>> Pskna;

  // D matrix for local projectors. Species-resolved (nsp, nhm*npol, nhm*npol)
  // for NCPP. For USPP/PAW the per-atom static D is held separately in
  // Dnn_atom_static (below); add_Vpp dispatches between the two.
  //memory::unified_array<ComplexType,3> Dnn;
  sarray_t<nda::array_view<ComplexType,3>> Dnn;

  // Static (frozen) per-atom non-local D for USPP/PAW (plan I2):
  //   D_static(a, I, J) = dion(type(a), I, J) + alpha_x * ex_cvij(type(a), I, J)
  // shape (nat, nhm*npol, nhm*npol). Built EAGERLY in read_vnl_h5 (dion is the
  // full frozen D^0 per QE convention; ex_cvij is the frozen core-valence
  // exact exchange, contracted with factor 1). Never mutated afterwards —
  // this is the only persistent per-atom D. Density-dependent (Hartree-only)
  // terms are rebuilt per evaluation via compute_paw_deeq (plan I3). Dummy
  // (1,1,1) for NCPP (species-resolved Dnn is used instead).
  sarray_t<nda::array_view<ComplexType,3>> Dnn_atom_static;

  // Per-species frozen-core radial densities (rho_core_AE, rho_core_PS),
  // SCF-invariant under the frozen-core approximation. Indexed by species
  // type. Populated lazily by `build_paw_scf_caches` on first use;
  // empty until then.
  std::vector<std::pair<nda::array<double,1>, nda::array<double,1>>>
      paw_core_density;

  // aainit `lli` parameter (= 1 + max_l over all PAW betas), needed to size
  // the angular-momentum coupling tables. Populated lazily alongside the
  // SCF caches above.
  int paw_aainit_lli = 0;

  // True after `build_paw_scf_caches` has built paw_core_density and
  // paw_aainit_lli. Acts as a memoization flag so the SCF entry point can
  // skip rebuilding the static caches each iteration.
  bool paw_scf_caches_built = false;

  // mapping from wfc_g grid to rho grid. 
  // hard coding ecut in mf now, allow for a custom cutoff later on
  sarray_t<nda::array_view<long,1>> swfc_to_rho;

  // local pseudopotential
  sarray_t<nda::array_view<ComplexType,3>> svloc;

  // scf local potential
  sarray_t<nda::array_view<ComplexType,3>> svsc;

  // qgm: Q^IJ(G) augmentation in reciprocal space, structure-factor free
  // shape: (nsp, nij_max, ngm_dense), where nij_max = max_t nh(t)*(nh(t)+1)/2
  // For species nt, valid pair indices are ij ∈ [0, nh(nt)*(nh(nt)+1)/2 ).
  // Non-USPP/PAW species rows are zero.
  sarray_t<nda::array_view<ComplexType,3>> qgm;

  // composite (ih,jh) -> ij index map for augmentation, shape (nsp, nhm, nhm)
  // 1-based indices coming from QE; subtract 1 before indexing into qgm.
  // Allocated empty for ncpp.
  nda::array<int,3> ijtoh;

  // augmentation overlap S = 1 + Σ_a Σ_IJ |β_aI⟩ q_{IJ} ⟨β_aJ|
  // shape: (nsp, nhm*npol, nhm*npol). Real-only convention until SOC ops land;
  // for non-SO USPP/PAW the imag part is zero.
  sarray_t<nda::array_view<ComplexType,3>> qq_nt_data;

  // Per-species PAW data (definition hoisted to public scope, see above).
  // Populated from /Hamiltonian/Species/{nt}/ in read_vnl_h5.
  std::vector<species_paw_t> paw_species;

  // Per-node SHM storage backing the heavy paw_species fields (aewfc,
  // pswfc, qfuncl, deltaC, core_aewfc). Index aligned with paw_species.
  // The `nda::array_view` fields in species_paw_t are bound to the
  // `.local()` of the corresponding sarray here, so consumers see
  // node-shared memory instead of per-rank replication.
  std::vector<species_paw_shm_t> paw_species_shm;

  // dense-grid G-vector count (read from /Hamiltonian/{type}/ngm attribute);
  // needed to size qgm and to map Q-index space.
  long ngm_dense = 0;

  // Miller indices for the dense G-grid that qgm lives on, shape (ngm_dense, 3).
  // Allocated only for USPP/PAW; needed by v_h_paw to build structure factors
  // and cartesian G-vectors when injecting the augmentation Q^IJ(G) e^{-iG·τ_a}.
  nda::array<int,2> miller_g_dense;

  // Cartesian atom positions, shape (nat, 3). Cached from mf at construct time
  // so the Hartree augmentation step doesn't need to thread mf through.
  nda::array<double,2> atom_pos_cart;

  template<typename MF_t>
  void read_vnl_pw2bgw(MF_t &mf, std::string outdir);

  template<typename MF_t>
  void read_vnl_h5(MF_t &mf, h5::group& grp); 

  void add_vnl_impl(nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& Dion,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  // Public PAW/USPP utilities exposed to callers (test code, paw_aug_thc,
  // v_h_paw, etc.). Re-open public scope here.
public:

  /**
   * Return the self-consistent per-atom D for a current density matrix `nij`
   * under the frozen-core no-XC scGW framework:
   *
   *   D_eff^a(ρ) = D_static^a + ΔD_H^a(becsum^a(ρ), ρ_core_AE, ρ_core_PS)
   *
   * Thin non-mutating wrapper over `compute_paw_deeq(nij, empty_Vloc,
   * include_static=true)` (plan A1): the static tensor `Dnn_atom_static`
   * (dion + ex_cvij, built in the ctor) is the baseline and is never
   * modified; the dynamic piece is the Hartree-only radial one-center term
   * from the current becsum. No G-space ∫V·Q term (pass a potential to
   * `compute_paw_deeq` directly for the full KS-like operator).
   *
   * Restrictions: npol = 1 (no SOC), non-magnetic (nspin = 1 path uses
   * full nij; LSDA averaging is the same path on the spin-summed becsum).
   * Requires populated `species_paw_t` fields (aewfc, pswfc, ae_vloc,
   * vloc_ps, qfuncl); throws otherwise.
   *
   * Implementation lives out-of-class in `hamiltonian/paw/paw_onecenter.hpp`
   * — include that header at the call site.
   */
  template<typename nij_t>
  nda::array<ComplexType,3> compute_deeq_scf(nij_t const& nij);

  /**
   * Unified native PAW non-local D builder (the single source of truth — no
   * QE deeq, no XC). Returns a per-atom Dion array (nat, nhm*npol, nhm*npol)
   * assembled from up to three terms:
   *   1. static     : dvan (h5 `dion` in Ha; already includes the AE−PS
   *                   V_loc baseline per QE convention), included iff
   *                   `include_static`.
   *   2. radial     : AE−PS one-center Hartree, `compute_paw_hartree_atom`
   *                   from `becsum = ns_scl × compute_becsum_{diagonal,full}`.
   *                   No additional multiplier — `dDeeq_H` evaluated on the
   *                   spin-summed becsum equals QE's `ddd_paw` per-channel
   *                   (Ry/Ha cancellation across QE's `e2=2` convention and
   *                   CoQui's spin doubling — see
   *                   `notes/paw_deeq_scaling_derivation.md`).
   *   3. G-space    : `∫ Vloc_r(r) Q^IJ_a(r) dr`, included iff `Vloc_r` is
   *                   non-empty (caller passes V_H for the Hartree matrix,
   *                   or V_eff = V_loc+V_H for the full deeq used in
   *                   `add_vpp_impl`).
   * Callers build Dion then feed it to `add_vnl_impl` — the single code path
   * that touches Hij/Vij. Two overloads accept the density either as the
   * diagonal `nii(s,k,n)` or the full density matrix `nij(s,k,a,b)`; both
   * funnel through `compute_paw_deeq_from_becsum` so the static / radial /
   * G-space assembly (and the rad_fac = 1 convention) lives in one place.
   *
   * Implementation lives out-of-class in `hamiltonian/paw/paw_onecenter.hpp`.
   */
  template<typename Pot_t>
  nda::array<ComplexType,3> compute_paw_deeq(nda::ArrayOfRank<3> auto const& nii,
                                             Pot_t const& Vloc_r,
                                             bool include_static);
  template<typename Pot_t>
  nda::array<ComplexType,3> compute_paw_deeq(nda::ArrayOfRank<4> auto const& nij,
                                             Pot_t const& Vloc_r,
                                             bool include_static);
  // Shared core: assemble Dion from an already spin-scaled per-atom becsum
  // (nat, nhm, nhm). The two compute_paw_deeq overloads build becsum from
  // nii / nij respectively and delegate here.
  template<typename Pot_t>
  nda::array<ComplexType,3> compute_paw_deeq_from_becsum(
                                             nda::array<double,3> const& becsum,
                                             Pot_t const& Vloc_r,
                                             bool include_static);

  /**
   * Build the SCF-invariant caches consumed by the PAW deeq builders:
   * `paw_core_density`, `paw_aainit_lli`. (The static per-atom tensor
   * `Dnn_atom_static` is NOT built here — it is assembled eagerly in
   * `read_vnl_h5`.) Idempotent; subsequent calls return immediately.
   * Called automatically from `compute_paw_deeq_from_becsum`; exposed
   * publicly so callers can pre-build the caches (e.g., during driver
   * setup, outside the SCF hot loop).
   *
   * Implementation lives out-of-class in `hamiltonian/paw/paw_onecenter.hpp`.
   */
  void build_paw_scf_caches();

  /**
   * Add the smooth-grid USPP/PAW augmentation Σ_a Σ_IJ becsum_aIJ Q^IJ_nt(G) e^{-iG·τ_a}
   * to a pair density rhoG already on the dense G grid. NCPP species are
   * skipped. The caller is responsible for building becsum_aIJ from Pskna
   * and the gvec_phase structure-factor table.
   */
  void add_augmentation_to_pairdensity(
      nda::ArrayOfRank<2> auto const& becsum_aIJ,
      nda::ArrayOfRank<2> auto const& gvec_phase,
      nda::ArrayOfRank<1> auto       & rhoG) const;

  /**
   * Add the augmentation contribution to a (s,k,a,b) orbital-basis overlap:
   *   S_ab(s,k) += Σ_atom Σ_IJ conj(P_aI(s,k,a)) * q_IJ(type(atom)) * P_aJ(s,k,b)
   *
   * Combined with the identity, this yields the full ultrasoft/PAW S overlap
   * S = 1 + Σ_a Σ_IJ |β_aI⟩ q_{IJ} ⟨β_aJ|
   * For NCPP species (qq_nt = 0) the call is a no-op. For SOC the diagonal
   * spinor block is used (qq_so support is a TODO).
   *
   * Sij must already hold the bare overlap (typically the identity) on entry;
   * this method *adds* the augmentation correction. Pattern matches
   * add_vnl_impl so consumers can dispatch the same way they do for V_NL.
   */
  void add_S(nda::range k_range, nda::range b_range,
             math::nda::DistributedArrayOfRank<4> auto & Sij)
  {
    using nda::range;
    decltype(range::all) all;
    constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(Sij.local())>>();

    if (ptype == pp_ncpp_t) return;

    auto k_range_loc = Sij.local_range(1) + k_range.first();
    auto b_range_loc = Sij.local_range(2) + b_range.first();
    long nkb  = Pskna.shape()[2]/npol;
    if (nkb == 0) return;

    auto Qloc  = qq_nt_data.local();
    auto Sloc  = Sij.local();
    auto Ploc  = Pskna.local();

    if constexpr (MEM == HOST_MEMORY) {
      long nbnd = b_range_loc.size();
      memory::array<MEM, ComplexType, 2> T(nbnd, nkb*npol);
      memory::array<MEM, ComplexType, 3> Qfull;
      int nhm = Qloc.shape()[1];
      Qfull = memory::array<MEM, ComplexType, 3>(Qloc.shape()[0], nhm*npol, nhm*npol);
      Qfull() = ComplexType(0.0);
      for (int s=0; s<int(Qloc.shape()[0]); ++s)
        for (int p=0; p<npol; ++p)
          for (int n=0; n<nhm; ++n)
            for (int m=0; m<nhm; ++m)
              Qfull(s, n*npol+p, m*npol+p) = Qloc(s, n, m);
      for (auto [is,s] : itertools::enumerate(Sij.local_range(0)))
        for (auto [ik,k] : itertools::enumerate(k_range_loc)) {
          for (auto [ia,nt] : itertools::enumerate(ityp)) {
            if (nh(nt) == 0) continue;
            nda::blas::gemm(ComplexType(1.0),
              nda::dagger(Ploc(s, k,
                    range(ofs(ia)*npol, (ofs(ia)+nh(nt))*npol), b_range_loc)),
              Qfull(nt, range(nh(nt)*npol), range(nh(nt)*npol)),
              ComplexType(0.0),
              T(all, range(ofs(ia)*npol, (ofs(ia)+nh(nt))*npol)));
          }
          nda::blas::gemm(ComplexType(1.0), T,
                          Ploc(s, k, all, b_range),
                          ComplexType(1.0), Sloc(is, ik, all, all));
        }
    } else {
      static_assert(MEM == HOST_MEMORY,
          "pseudopot::add_S device path not yet implemented");
    }
  }

  /**
   * Add the contributions of a generic pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @tparam Arr3   - Array type of nii
   * @tparam Arr4   - Array type of nij
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + Vpp_loc) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b)
   * @param add_hartree  - [input] with a density: add smooth V_H to hpsi and
   *                       the V_loc+V_H one-center D to Hij (plan I5). false
   *                       reduces to the static-only no-density assembly.
   * @param add_exchange - [input] with a density: accumulate the direct-route
   *                       exact-exchange K into Hij (host only).
   */
  template< nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_vpp_impl(boost::mpi3::communicator& comm,
               nda::range k_range, nda::range b_range,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               const Arr3 * nii, const Arr4 * nij,
               bool add_hartree = true, bool add_exchange = false);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @tparam Arr3   - Array type of "nii" array
   * @tparam Arr4   - Array type of "nij" array
   * @param k_range - [input] Range of k-point indices
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b). Note that either "nii" or "nij"
   *                          should be provided, not both.
   */
  template<nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_Hartree_impl(nda::range k_range, nda::range b_range,
                        math::nda::DistributedArrayOfRank<4> auto const& psi,
                        math::nda::DistributedArrayOfRank<4> auto & hpsi,
                        math::nda::DistributedArrayOfRank<4> auto & Vij,
                        const Arr3 *nii, const Arr4 *nij, bool symmetrize=false);


};

// if mf.get_pseudopot() returns a valid shared pointer, return it.
// otherwise, construct a new object managed by a shared pointer,
// store the pointer in mf and return it.
template<typename MF_t>
std::shared_ptr<pseudopot> make_pseudopot(MF_t &mf)
{
  // sync for safety for now, this routine is blocking
  auto mpi = mf.mpi();
  mpi->comm.barrier();
  if(mf.get_pseudopot()) { return mf.get_pseudopot(); }
  else {
    // Construct object, attach lazy reference to mf, return
    auto psp = std::make_shared<pseudopot>(mf);
    mf.set_pseudopot(psp);
    if( not mf.get_pseudopot() )
      APP_ABORT("Error in make_pseudopot. Logic problem.");
    return psp;
  }
}

}

