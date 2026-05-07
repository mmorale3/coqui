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
  // Pskna/qq_nt_data/Dnn_atom in the pseudopot ctor init list).
  struct species_paw_shm_t {
    sarray_t<nda::array_view<double,2>> aewfc;
    sarray_t<nda::array_view<double,2>> pswfc;
    sarray_t<nda::array_view<double,3>> qfuncl;
    sarray_t<nda::array_view<double,4>> deltaC;
    sarray_t<nda::array_view<double,2>> core_aewfc;

    explicit species_paw_shm_t(mpi_t& m)
      : aewfc(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1})),
        pswfc(math::shm::make_shared_array<nda::array_view<double,2>>(m, {1,1})),
        qfuncl(math::shm::make_shared_array<nda::array_view<double,3>>(m, {1,1,1})),
        deltaC(math::shm::make_shared_array<nda::array_view<double,4>>(m, {1,1,1,1})),
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

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& nii,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<4> auto const& nij,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

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
                   bool symmetrize=false);

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
  // for NCPP. For USPP/PAW the SCF-dependent deeq correction is per-atom and
  // is held separately in Dnn_atom (below); add_Vpp dispatches between the two.
  //memory::unified_array<ComplexType,3> Dnn;
  sarray_t<nda::array_view<ComplexType,3>> Dnn;

  // Atom-resolved effective non-local D for USPP/PAW:
  //   D_eff(a, I, J) = dvan(type(a), I, J) + deeq(I, J, a, spin=0)
  // shape (nat, nhm*npol, nhm*npol). Empty for NCPP. Spin-polarized
  // calculations would need (nspin, nat, ...); collinear non-magnetic
  // (nspin=1 or 2 with deeq same on both spins) is supported here. Magnetic
  // PAW with deeq(s=1)≠deeq(s=2) is a documented TODO.
  sarray_t<nda::array_view<ComplexType,3>> Dnn_atom;

  // SCF-invariant ("static") baseline for the per-atom non-local D matrix
  // under the no-XC frozen-core scGW design:
  //   D_static^a = dvan(type(a)) + paw_init_keeq(type(a))
  // where paw_init_keeq carries the kinetic and bare-ionic AE−PS one-center
  // matrix elements (`hamilt::paw::compute_paw_static_D`, in Ha). For NCPP
  // species this collapses to the dvan replica.
  // Same shape and SHM layout as Dnn_atom. Populated lazily on the first
  // call to `hamilt::paw::compute_deeq_scf`; empty until then.
  sarray_t<nda::array_view<ComplexType,3>> Dnn_atom_static_paw;

  // Per-species frozen-core radial densities (rho_core_AE, rho_core_PS),
  // SCF-invariant under the frozen-core approximation. Indexed by species
  // type. Populated lazily by `hamilt::paw::compute_deeq_scf` on first call;
  // empty until then.
  std::vector<std::pair<nda::array<double,1>, nda::array<double,1>>>
      paw_core_density;

  // aainit `lli` parameter (= 1 + max_l over all PAW betas), needed to size
  // the angular-momentum coupling tables. Populated lazily alongside the
  // SCF caches above.
  int paw_aainit_lli = 0;

  // True after `hamilt::paw::compute_deeq_scf` has built Dnn_atom_static_paw,
  // paw_core_density, and paw_aainit_lli. Acts as a memoization flag so the
  // SCF entry point can skip rebuilding the static caches each iteration.
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
   * Recompute `Dnn_atom` from a current density matrix `nij` for the
   * frozen-core no-XC scGW framework:
   *
   *   D_eff^a(ρ) = D_static^a + ΔD_H^a(becsum^a(ρ), ρ_core_AE, ρ_core_PS)
   *
   * Static piece (SCF-invariant): kinetic + bare-ionic AE−PS one-center
   * matrix elements, computed once from the canonical UPF inputs (no QE
   * runtime dependency, no frozen-XC residual). Cached in
   * `Dnn_atom_static_paw`.
   *
   * Dynamic piece (Hartree only): per-atom radial Hartree solver via
   * `compute_paw_hartree_atom` from the current becsum.
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
  void compute_deeq_scf(nij_t const& nij);

  /**
   * Build the SCF-invariant caches consumed by `compute_deeq_scf`:
   * `Dnn_atom_static_paw`, `paw_core_density`, `paw_aainit_lli`. Idempotent;
   * subsequent calls return immediately. Called automatically from
   * `compute_deeq_scf`; exposed publicly so callers can pre-build the
   * caches (e.g., during driver setup, outside the SCF hot loop).
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
   */
  template< nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_vpp_impl(boost::mpi3::communicator& comm,
               nda::range k_range, nda::range b_range, 
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               const Arr3 * nii, const Arr4 * nij);

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
  void add_Hartree_impl(nda::range k_range,
                        math::nda::DistributedArrayOfRank<4> auto const& psi,
                        math::nda::DistributedArrayOfRank<4> auto & hpsi,
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

