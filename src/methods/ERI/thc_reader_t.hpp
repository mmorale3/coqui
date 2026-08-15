/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
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


#ifndef COQUI_THC_READER_T_HPP
#define COQUI_THC_READER_T_HPP

#include <string>
#include <optional>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/freemem.h"
#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"

#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/eri_storage_e.hpp"
#include "methods/ERI/paw_exx_options_parse.hpp"
#include "methods/ERI/thc.h"
#include "methods/ERI/chol_reader_t.hpp"
#include "mean_field/MF.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/paw/paw_symmetry.hpp"
#include "hamiltonian/paw/local_isdf.hpp"
#include "hamiltonian/paw/local_isdf_compress.hpp"
#include "hamiltonian/paw/local_isdf_h5.hpp"
#include "hamiltonian/paw/paw_aug_thc.hpp"

namespace methods {

  namespace mpi3 = boost::mpi3;

  /**
   * Read-only interface for thc-ERIs, computed on the fly
   *   a) only supports q-independent interpolating points (i.e. q-independent X) at the moment
   *
   * Usage:
   *   thc_reader_t thc(...);
   *   auto Xk = thc.X(is, ik); // return nda::array_view
   *   auto Yk = thc.Y(is, ik); // return nda::array_view
   *   auto Zq = thc.Z(iq);
   */
  class thc_reader_t {
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using Array_t = memory::array<MEM, ComplexType, R>;
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using Array_view_t = memory::array_view<MEM, ComplexType, R>;
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using dArray_t = memory::darray_t<memory::array<MEM, ComplexType, R>, mpi3::communicator>;
    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;
  public:
    thc_reader_t(std::shared_ptr<mf::MF> MF,
                 ptree const& pt, 
                 bool isdf_only = false, bool intialize = true):
      _MF(std::move(MF)), _mpi(_MF->mpi()),
      _MEM_EVAL( io::get_compute_space(pt,"compute") ),
      _storage(string_to_eri_storage_enum(io::tolower_copy(io::get_value_with_default<std::string>(pt,"storage","incore")))), 
      _eri_file( io::get_value_with_default<std::string>(pt,"save","") ), 
      _format( io::get_value_with_default<std::string>(pt,"format","bdft") ),
      _cd_dir( io::get_value_with_default<std::string>(pt,"cd_dir","") ), 
      _X_type("q_indep"),
      _thc_builder_opt(thc(_MF.get(), *_mpi, pt, false)),
      _Np( int(io::get_value_with_default<int>(pt,"nIpts",0)) ), 
      _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()),
      _nqpts(_MF->nqpts()), _nqpts_ibz(_MF->nqpts_ibz()),
      _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()), _nbnd(_MF->nbnd()), 
      _npol(_MF->npol()), _npol_in_basis(_MF->npol_in_basis()),
      x_range( io::get_value_with_default<nda::range>(pt,"X_orbital_range",nda::range(_nbnd)) ), 
      y_range( io::get_value_with_default<nda::range>(pt,"Y_orbital_range",x_range) ), 
      _dZ(math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {_mpi->comm.size(), 1, 1}, {_mpi->comm.size(), 1, 1})),
      _X_shm(math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(*_mpi,{1, 1, 1, 1})),
      _Y_shm{std::nullopt},
      _Timer() {
      utils::check(x_range.first() >= 0 and x_range.last() <= _nbnd,
                   "X orbitals out of range: ({},{}), nbnd:{}",x_range.first(),x_range.last(),_nbnd);
      utils::check(y_range.first() >= 0 and y_range.last() <= _nbnd,
                   "Y orbitals out of range: ({},{}), nbnd:{}",y_range.first(),y_range.last(),_nbnd);
      auto thresh = io::get_value_with_default<double>(pt,"thresh",-1.0);
      // Auto-resolve thresh sentinel: when nIpts is set, use 1e-13 (numerical floor); otherwise use 1e-5
      if (thresh < 0.0) {
        thresh = (_Np > 0) ? 1e-13 : 1e-5;
      }
      utils::check( _Np>0 or thresh>0.0, "Error in thc_reader_t: Must set nIpts and/or thresh");

      // PAW augmentation options. Defaults: include every term that is
      // compatible with each species' pseudopotential type:
      //   - NCPP species  : skipped entirely (nothing to augment)
      //   - USPP species  : compensation-charge augmentation (V_GL/V_LL),
      //                     no on-site kernel (USPP has no K_a)
      //   - PAW species   : compensation-charge augmentation + on-site K_a
      // The per-species discrimination is enforced in make_paw_aug_layout
      // (NCPP entries get 0 ISDF rows) and in add_K_a_to_LL (skips non-PAW).
      // `paw_onsite` is a DIAGNOSTIC-ONLY knob: it lets one disable
      // just the K_a contribution for PAW species while keeping their
      // compensation augmentation, to isolate the on-site term in studies.
      // Whether K_a belongs at all is NOT this knob's job — it is derived
      // from the vv_compensation mode (moment => include, shape => drop,
      // since the shape-restored eta already carries the full AE-PS pair
      // density). Production runs leave paw_onsite at its default (true).
      if(_MF->pp_type() == hamilt::pp_paw_t or _MF->pp_type() == hamilt::pp_uspp_t) {
        _paw_aug = io::get_value_with_default<bool>(pt, "paw_aug", true);
        _paw_onsite = io::get_value_with_default<bool>(pt, "paw_onsite", _MF->pp_type() == hamilt::pp_paw_t);
        // Diagnostic gates (default true) to isolate the V_GL (smooth-aug
        // cross) vs V_LL (aug-aug) compensation blocks (used by test
        // thc_vgl_vll_split). Production runs leave both true.
        _paw_vgl = io::get_value_with_default<bool>(pt, "paw_vgl", true);
        _paw_vll = io::get_value_with_default<bool>(pt, "paw_vll", true);
        _paw_isdf_tol = io::get_value_with_default<double>(pt, "paw_isdf_tol", 1e-12);
        _paw_isdf_cache_h5 = io::get_value_with_default<std::string>(pt, "paw_isdf_cache_h5", "");
        // Cutoff (Ha, on |q+G|^2/2) applied to the augmentation channels eta.
        // The one-center oscillator Q_ij(q+G) that eta carries is built on the
        // frozen partial-wave basis, so it is only as good as the on-site
        // completeness of that basis, |psi~> ~ sum_i |phi~_i><p_i|psi~>. That
        // holds where the projectors were fitted and degrades at high |q+G|,
        // where Q_ij does not decay but the true AE oscillator does — so the
        // augmentation there is spurious and grows as |<p_i|psi~_n>|^2 with
        // band index. Every PAW response code carries the same control
        // (ABINIT `ecuteps`, VASP ENCUTGW); the THC/ISDF basis has no implicit
        // one, hence this knob.
        //
        // The cutoff is applied to eta ITSELF (not to the Coulomb weights), so
        // the assembled Z stays the exact Coulomb Gram matrix of the function
        // set {zeta (band-limited to rho_g), eta (cut at this Gcut)} and is
        // therefore still PSD. Cutting the kernel instead would leave the GG
        // block on the full v while GL/LL saw a truncated one, which is not a
        // Gram matrix of anything.
        //
        // 0 (default) = off, i.e. eta carried to the full grid, unchanged
        // behaviour. Note K_a is grid-free and is NOT truncated by this knob.
        _paw_aug_ecut = io::get_value_with_default<double>(pt, "paw_aug_ecut", 0.0);
        {
          std::string m = io::tolower_copy(io::get_value_with_default<std::string>(pt, "paw_isdf_metric", "coulomb"));
          _paw_isdf_metric = (m == "l2") ? hamilt::paw::isdf_metric::L2
                                         : hamilt::paw::isdf_metric::Coulomb;
        }
        // PAW valence-valence exchange options: shared toml surface with the
        // [interaction.hamilt] block (see paw_exx_options_parse.hpp).
        _exx_opts = parse_paw_exx_options(pt, /*in_thc=*/true,
                                          "thc_reader_t [interaction.thc]");
      } else {
        _paw_aug = false;
        _paw_onsite = false;
      }
      if(_storage == eri_storage_e::outcore and _eri_file == "") 
        _eri_file = "./thc.eri.h5";

// MAM: use mf.pp_type() to determine what needs to be done. If ncpp, turn off any PAW/USPP related work.   

      if (intialize) {
        if (isdf_only) {
          build_isdf_only(io::get_value_with_default<bool>(pt, "check_accuracy", false),
                          io::get_value_with_default<bool>(pt, "write_zeta_on_fft_mesh", false));
        } else {
          init(true);
        }
      }
    }

    // read existing THC integrals
    thc_reader_t(std::shared_ptr<mf::MF> MF,
                 std::string storage,
                 std::string eri_file = "",
                 bool intialize = true):
      _MF(MF), _mpi(_MF->mpi()), _MEM_EVAL( DEFAULT_MEMORY_SPACE ),
      _storage(string_to_eri_storage_enum(storage)), 
      _eri_file(eri_file), 
      _format("bdft"),
      _cd_dir(""),
      _X_type("q_indep"), 
      _thc_builder_opt{std::nullopt},
      _Np(read_Np()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()), 
      _nqpts(_MF->nqpts()), _nqpts_ibz(_MF->nqpts_ibz()),
      _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()), _nbnd(_MF->nbnd()),
      _npol(_MF->npol()),_npol_in_basis(_MF->npol_in_basis()), 
      x_range(0),  // read later
      y_range(0),  // read later
      _dZ(math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {_mpi->comm.size(), 1, 1}, {_mpi->comm.size(), 1, 1})),
      _X_shm(math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(*_mpi,{1, 1, 1, 1})),
      _Y_shm{std::nullopt},
      _Chi_head(_nqpts_ibz, _Np), _Chi_bar_head(_nqpts_ibz, _Np),
      _Timer() {
      if (intialize) init(false);
    };

    ~thc_reader_t() = default;

    thc_reader_t(thc_reader_t const&) = default;
    thc_reader_t(thc_reader_t &&) = default;
    thc_reader_t& operator=(thc_reader_t const&) = default;
    thc_reader_t& operator=(thc_reader_t &&) = default;

    void init(bool build_eri) {

      print_thc_reader_info(build_eri);

      utils::check(_X_type == "q_indep", "thc_reader_t: q-dependent Xk is not implemented yet!");

      for (auto &v: {"BUILD_TOTAL", "BUILD_THC", "BUILD_GATHER", "BUILD_WRITE",
                     "READ_X", "READ_V", "PAW_AUG",
                     "PAW_AUG.aainit", "PAW_AUG.qrad_tab", "PAW_AUG.dzeta",
                     "PAW_AUG.gather_smooth", "PAW_AUG.eta_at_q",
                     "PAW_AUG.wG_at_q", "PAW_AUG.eta_flat",
                     "PAW_AUG.V_GL", "PAW_AUG.V_LL",
                     "PAW_AUG.K_a", "PAW_AUG.stitch", "PAW_AUG.scatter",
                     "PAW_AUG.X_aug", "PAW_AUG.Pskna_lift"})
        _Timer.add(v);

      // PAW-aug: lazily acquire pseudopot through MF (shared with other
      // consumers via make_pseudopot) and build/load the compressed isdf.
      if (_paw_aug) prepare_paw_isdf();

      if (build_eri) {
        utils::check(_thc_builder_opt!=std::nullopt, "thc_builder is not initialized!");
        _thc_builder_opt.value().print_metadata();
        if (_cd_dir == "")
          build();         // build THC-ERI from ISDF
        else
          build_from_CD(); // build THC-ERI from LS-THC
      } else
        read();            // read existing THC-ERI

      _initialized = true;
      app_log(1, "####### End of THC initialization routines #######\n");
    }

  private:

    void print_thc_reader_info(bool build_eri) {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20ThcCoulomb
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ╔╦╗┬ ┬┌─┐╔═╗┌─┐┬ ┬┬  ┌─┐┌┬┐┌┐ \n"
                 "║  ║ ║║═╬╗║ ║║   ║ ├─┤│  ║  │ ││ ││  │ ││││├┴┐\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩   ╩ ┴ ┴└─┘╚═╝└─┘└─┘┴─┘└─┘┴ ┴└─┘\n");
      app_log(1, "  Algorithm                       = {}", (_cd_dir=="")? "ISDF" : "LS-THC");
      app_log(1, "  THC integrals access            = {}", eriform_enum_to_string(_storage));
      app_log(1, "  Found precomputed THC integrals = {}", !build_eri);
      if (build_eri) {
        if (_eri_file != "")
          app_log(1, "  --> CoQuí will compute THC integrals and save to: {}", _eri_file);
        else
          app_log(1, "  --> CoQuí will compute THC integrals.");
      } else
        app_log(1, "  --> Reading the precomputed THC integrals from file: {}", _eri_file);
      app_log(1, "");
    }

    // Acquire pseudopot via MF's lazy reference and build / load the
    // compressed local-ISDF for PAW augmentation. Idempotent: safe to call
    // multiple times.
    void prepare_paw_isdf() {
      _Timer.start("PAW_AUG");
      if (!_psp) _psp = hamilt::make_pseudopot(*_MF);
      utils::check(_psp != nullptr,
                   "thc_reader_t: paw_aug=true but make_pseudopot returned null.");
      // Propagate the toml PAW exact-exchange options to the shared pseudopot so
      // the direct v_x / add_exchange path uses the same settings as the THC ERI.
      _psp->set_exx_options(_exx_opts);

      // Try to load a cached compressed isdf if a path was given.
      if (!_paw_isdf_cache_h5.empty() && std::filesystem::exists(_paw_isdf_cache_h5)) {
        // A cache is only reusable if it was built under the SAME selection
        // rule. Pre-v1 caches carry no tag and were ranked at q=0 only; their
        // channel set differs from ours at the same tol, so they are rebuilt
        // rather than silently trusted.
        std::string sel_back;
        {
          h5::file f(_paw_isdf_cache_h5, 'r');
          h5::group root(f);
          if (root.has_subgroup(hamilt::paw::kLocalISDFGroup)) {
            h5::group g = root.open_group(hamilt::paw::kLocalISDFGroup);
            // Absent attribute leaves sel_back empty -- that IS the pre-v1 case.
            h5::h5_read_attribute(g, "selection", sel_back);
          }
        }
        if (sel_back != hamilt::paw::kISDFSelectionTag) {
          app_warning("paw_aug: ignoring local-ISDF cache {} -- it was built "
                      "with channel selection '{}' but this build uses '{}'. "
                      "Rebuilding. (Pre-v1 caches ranked channels at q=0 only, "
                      "which under-ranks every pair with no L=0 component.)",
                      _paw_isdf_cache_h5,
                      sel_back.empty() ? "<untagged, pre-v1>" : sel_back,
                      hamilt::paw::kISDFSelectionTag);
          _isdf.clear();
        } else {
        hamilt::paw::isdf_metric m_back = hamilt::paw::isdf_metric::Coulomb;
        double tol_back = 0.0;
        _isdf = hamilt::paw::load_compressed_local_isdf_from_h5(
            _paw_isdf_cache_h5, &m_back, &tol_back);
        if (!_isdf.empty()) {
          app_log(1, "  paw_aug: loaded compressed local-ISDF from {} "
                     "(metric={}, tol={:.1e}) [selection {}]",
                     _paw_isdf_cache_h5, hamilt::paw::metric_name(m_back), tol_back,
                     sel_back);
          _Timer.stop("PAW_AUG");
          return;
        }
        }
      }

      // Build from psp (compressed-by-norm at the requested tolerance).
      // Compressed-by-norm preserves exact reconstruction at every kept
      // pair (the scheme we use elsewhere); this is the right default.
      auto const& sps = _psp->paw_species_view();
      int nsp = (int)sps.size();
      auto recv = _MF->recv();
      double det_B =
          recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
        - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
        + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
      double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);
      // Channel selection. `build_local_isdf_compressed_by_norm` ranks (I,J)
      // pairs by their Coulomb-metric norm evaluated at **q = 0 only**, with
      // w(G) = 4pi/(Omega |G|^2). That criterion is biased: a pair containing
      // an L=0 component has Q_IJ(G) -> const as G -> 0 and collects the full
      // 1/G^2 enhancement, while a pair with l_i != l_j has Q_IJ(G) -> 0 like
      // G^L and collects none. The latter therefore look negligible and get
      // dropped -- yet on the q-mesh the smallest wavevector is |q|, not 0
      // (0.271 a.u. on the Si 4x4x4 mesh), where those channels are NOT small.
      // Measured cost of the bias (Si, d-complete data set): the q!=0
      // completeness gate
      // reads 1.021258 at paw_isdf_tol=5e-5 versus 0.999054 at full rank, and
      // E_c moves ~14 mHa between those two settings while a full decade of
      // `thresh` at fixed tol moves it 0.01 mHa. Note the gate is violated
      // ONLY at q!=0 -- it is exact at q=0 by construction, which is why every
      // q=0 test passed. On the LiH fixture the same signature shows up in the
      // OFF-DIAGONAL completeness residual max_{v!=c}|rho_vc(0)|, which
      // degrades 8.9e-16 -> 4.1e-9 between full rank and 5e-5 while the
      // DIAGONAL gate |rho_vv(0)-1| sits at 4.1e-12 either way: dropped
      // channels corrupt the off-diagonal, which is what ln|det(I-Pi*Z)| sums
      // over and what no diagonal test can see.
      //
      // The fix ranks each pair by its worst case over the q-mesh actually
      // used, which is what the ERI sums over. eta is built with the same
      // interpolated machinery the augmentation uses, so this costs ~1 s.
      _isdf.clear(); _isdf.reserve(nsp);
      for (int nt = 0; nt < nsp; ++nt) {
        bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
        if (!has_aug) {
          _isdf.emplace_back();
          continue;
        }
        // Start from FULL rank (tol=0 keeps every pair with a valid (I,J)),
        // then decide what to drop using the q-aware norm below.
        auto isdf_nt = hamilt::paw::build_local_isdf_compressed_by_norm(
            *_psp, nt, recv, omega, _paw_isdf_metric, 0.0);
        _isdf.push_back(std::move(isdf_nt));
      }
      if (_paw_isdf_tol > 0.0)
        select_aug_channels_qaware(recv, omega);
      app_log(1, "  paw_aug: built compressed local-ISDF (metric={}, tol={:.1e}, nsp={})",
              hamilt::paw::metric_name(_paw_isdf_metric), _paw_isdf_tol, nsp);
      // Per-species size breakdown — clarifies where N_aug comes from.
      // Full-rank cap is nh² per atom: nh diagonal pairs + 2 × nh*(nh-1)/2
      // off-diagonal sym/antisym rows. Compression drops parity/cutoff
      // -forbidden (ij)-pairs whose qgm metric norm is below `tol`.
      for (int nt = 0; nt < nsp; ++nt) {
        if (_isdf[nt].nlambda == 0) continue;
        int nh_nt = _isdf[nt].nh;
        long nij  = (long)nh_nt * (nh_nt + 1) / 2;
        app_log(1,
            "  paw_aug:   species nt={}: nh={}, nij_max={}, "
            "kept nlambda={} (full-rank cap={})",
            nt, nh_nt, nij, _isdf[nt].nlambda, nh_nt * nh_nt);
      }

      // Optionally save the cache.
      if (!_paw_isdf_cache_h5.empty() && _mpi->comm.root()) {
        std::vector<hamilt::paw::isdf_compression_report> empty_reps(_isdf.size());
        h5::file f(_paw_isdf_cache_h5,
                   std::filesystem::exists(_paw_isdf_cache_h5) ? 'a' : 'w');
        h5::group root(f);
        hamilt::paw::write_local_isdf_h5(root, _isdf, empty_reps,
                                          _paw_isdf_metric, _paw_isdf_tol);
        app_log(1, "  paw_aug: saved compressed local-ISDF to {}",
                _paw_isdf_cache_h5);
      }
      _mpi->comm.barrier();
      _Timer.stop("PAW_AUG");
    }

    void print_thc_summary() {
      app_log(1, "\n  Summary of THC Coulomb Integrals");
      app_log(1, "  --------------------------------");
      app_log(1, "  Number of interpolating points = {}", _Np);
      app_log(1, "  X orbital index range          = [{},{})", x_range.first(),x_range.last());
      app_log(1, "  Y orbital index range          = [{},{})\n", y_range.first(),y_range.last());
    }

    void build() {
      _Timer.start("BUILD_TOTAL");

      bool any_aug_species = false;
      if (_paw_aug && _psp != nullptr) {
        for (auto const& sp : _psp->paw_species_view()) {
          if (sp.is_paw || sp.is_uspp) { any_aug_species = true; break; }
        }
      }

      _Timer.start("BUILD_THC");
      {
        auto eval = [&]<MEMORY_SPACE MEM>() {
          auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<MEM>(0, _Np, x_range, y_range);
          _rp = std::move(ri);
          _Np = _rp.size();
          _Timer.stop("BUILD_THC");

          // The intvec_impl multiply/divide recovery of ζ_quG only has a
          // HOST_MEMORY implementation today (the DEVICE branch is gated
          // by an explicit utils::check). For non-HOST compute we fall
          // back to a separate evaluate_isdf_only call below, so we
          // request the optional only when MEM == HOST_MEMORY.
          constexpr bool host_mem = (MEM == HOST_MEMORY);
          bool ret_zeta_in_eval = host_mem && any_aug_species;

          _Timer.start("BUILD_THC");
          auto [_dZ_d, _Chi_head_d, _Chi_bar_head_d, dzeta_quG_d] =
              _thc_builder_opt.value().evaluate<MEM>(
                  _rp, dXa, dXb, ret_zeta_in_eval, x_range, y_range);
          utils::check(ret_zeta_in_eval == dzeta_quG_d.has_value(),
            "Error: Inconsistent optional return value.");
          _Timer.stop("BUILD_THC");

          _Np_smooth = (int)_Np;

          // scale by nkpts: thc::intvec_impl divides by Ω·Nk; promote to 1/Ω.
          nda::tensor::scale(ComplexType(1.0*_nkpts), _dZ_d.local());

          // copy to host memory if needed, otherwise just move
          _dZ = std::move(_dZ_d);
          _Chi_head = std::move(_Chi_head_d);
          _Chi_bar_head = std::move(_Chi_bar_head_d());

          // gather dPa to _X_shm — must run BEFORE augment_thc_with_paw,
          // which bumps _Np to N_total and reallocates _X_shm with the
          // appended PAW rows.
          _Timer.start("BUILD_GATHER");
          gather_X_shm(dXa);
          if(dXb.has_value())
            gather_Y_shm(dXb.value());
          else
            utils::check(x_range == y_range,
              "thc_reader::build: x_range != y_range with missing dXb value.");
          _Timer.stop("BUILD_GATHER");

          // PAW augmentation. For HOST_MEMORY we already have ζ_quG from
          // the evaluate() call; for DEVICE/UNIFIED we recompute it via
          // evaluate_isdf_only<MEM> until the device kernel is wired up.
          if (any_aug_species) {
            _Timer.start("PAW_AUG");
            if constexpr (host_mem) {
              augment_thc_with_paw<MEM>(*dzeta_quG_d);
            } else {
              auto dz = _thc_builder_opt.value().template evaluate_isdf_only<MEM>(
                  _rp, dXa, dXb, x_range, y_range);
              augment_thc_with_paw<MEM>(dz);
            }
            _Timer.stop("PAW_AUG");
          }
        };

        if(_MEM_EVAL == HOST_MEMORY)
          eval.operator()<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
        else if(_MEM_EVAL == DEVICE_MEMORY)
          eval.operator()<DEVICE_MEMORY>();
        else if(_MEM_EVAL == UNIFIED_MEMORY)
          eval.operator()<UNIFIED_MEMORY>();
#endif
      }

      // save if requested
      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          // MAM: write thc meta-data to into a "metadata" dataset. Useful for external codes/afqmc
          if (_format == "bdft") {
            std::vector<int> arng = {x_range.first(),x_range.last()};
            std::vector<int> brng = {y_range.first(),y_range.last()};
            h5::h5_write(grp, "Np", (int)_Np);
            h5::h5_write(grp, "number_of_bands", (int)_nbnd);
            h5::h5_write(grp, "X_orbital_range", arng);
            h5::h5_write(grp, "Y_orbital_range", brng);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            auto X_0 = _X_shm.local(); 
            nda::h5_write(grp, "collocation_matrix", X_0, false);
            if(_Y_shm.has_value()) {
              auto Y_0 = _Y_shm.value().local(); 
              nda::h5_write(grp, "Y_collocation_matrix", Y_0, false);
            }
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");

      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "  THC-READER::BUILD()");
      app_log(2, "  -------------------");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute thc-eri:             {0:.3f} sec", _Timer.elapsed("BUILD_THC"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write eri:                   {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, "      - paw augmentation total:      {0:.3f} sec", _Timer.elapsed("PAW_AUG"));
      if (_paw_aug) {
        app_log(2, "        .  X aug (Y rows x ks):      {0:.3f} sec", _Timer.elapsed("PAW_AUG.X_aug"));
        app_log(2, "        .  gather smooth GG block:   {0:.3f} sec", _Timer.elapsed("PAW_AUG.gather_smooth"));
        app_log(2, "        .  aainit (ap, lpx, lpl):    {0:.3f} sec", _Timer.elapsed("PAW_AUG.aainit"));
        app_log(2, "        .  qrad table (per species): {0:.3f} sec", _Timer.elapsed("PAW_AUG.qrad_tab"));
        app_log(2, "        .  q-loop dz gather:         {0:.3f} sec", _Timer.elapsed("PAW_AUG.dzeta"));
        app_log(2, "        .  q-loop eta at q+G:        {0:.3f} sec", _Timer.elapsed("PAW_AUG.eta_at_q"));
        app_log(2, "        .  q-loop wG at q+G:         {0:.3f} sec", _Timer.elapsed("PAW_AUG.wG_at_q"));
        app_log(2, "        .  q-loop eta flatten/conj:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.eta_flat"));
        app_log(2, "        .  q-loop V_GL contraction:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.V_GL"));
        app_log(2, "        .  q-loop V_LL contraction:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.V_LL"));
        app_log(2, "        .  q-loop K_a inject:        {0:.3f} sec", _Timer.elapsed("PAW_AUG.K_a"));
        app_log(2, "        .  q-loop stitch V_full:     {0:.3f} sec", _Timer.elapsed("PAW_AUG.stitch"));
        app_log(2, "        .  scatter V_full -> _dZ:    {0:.3f} sec", _Timer.elapsed("PAW_AUG.scatter"));
      }
      app_log(2, " ");

      print_thc_summary();
    }

    /**
     * Augment the smooth (X_shm, dZ) with PAW atom-local rows. Replaces
     * `_X_shm` and `_dZ` with their augmented counterparts and updates
     * `_Np = N_smooth + N_aug`.
     *
     * Distributed implementation. Production sizes (N_smooth ~ 4·10^4,
     * N_aug ~ 3·10^3, nkpts ~ 10^3, ngm_rho ~ 10^5–10^6) make the
     * V_full_q(_nqpts_ibz, N_total, N_total) tensor of the legacy path
     * unaffordable on a single node. This routine instead:
     *
     *   - allocates the augmented `_dZ` (3-aux distributed) up-front and
     *     fills it directly, never materialising an intermediate
     *     replicated 2-aux buffer;
     *   - copies the smooth GG block via a custom alltoallv that handles
     *     the (Np_smooth → N_total) chunk-shape change;
     *   - for each q owned by the local q-pool, builds ζ(μ, g) and η(λ, g)
     *     fully distributed across the q-pool comm (rows split, all g per
     *     rank); each rank gathers only the rows it needs to write into
     *     its own (P, Q) tile of `_dZ` for that iq (V_GL, V_LG, V_LL
     *     blocks) and does the local GEMM in-place;
     *   - K_a (q-independent on-site kernel) is added per-atom directly to
     *     each rank's local LL tile — no full N_aug×N_aug buffer.
     *
     * `dzeta_quG` is the smooth ζ_μ^q(G) on the thc rho_g grid (output of
     * `thc::evaluate_isdf_only`).
     */
    template<MEMORY_SPACE MEM>
    void augment_thc_with_paw(
        memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>
            const& dzeta_quG)
    {
      using nda::range;
      utils::check(_psp != nullptr,
                   "augment_thc_with_paw: _psp is null (call prepare_paw_isdf first).");
      auto const& thc_b = _thc_builder_opt.value();
      auto const& rho_g = thc_b.g_grid();
      double omega = thc_b.volume();
      long ngm_rho = (long)rho_g.size();

      _aug_layout = hamilt::paw::make_paw_aug_layout(*_psp, _isdf, _Np_smooth);
      _N_aug = _aug_layout.N_A;
      long N_total = (long)_Np_smooth + (long)_N_aug;
      app_log(1, "  paw_aug: N_smooth={}, N_aug={}, N_total={}",
              _Np_smooth, _N_aug, N_total);

      if (_N_aug == 0) {
        _Np = _Np_smooth;
        return;
      }

      // Augmentation-stage memory estimate. The smooth-stage
      // estimator (thc.icc get_ZquG_Cquv) predates augmentation and knows
      // nothing about the (N_total/Np_smooth)²-larger augmented _dZ, the
      // q-pool η/ζ slabs, or the gather buffers. Report the per-rank
      // footprint and warn (not abort — the estimate is an upper bound)
      // when it exceeds the free-memory snapshot.
      const long gchunk_rho = 8192;   // G-chunk width of the gather buffers
      {
        long nqp  = std::max(1l, _dZ.grid()[0]);
        long npP  = std::max(1l, _dZ.grid()[1]);
        long npQ  = std::max(1l, _dZ.grid()[2]);
        long npPQ = npP * npQ;
        auto ceild = [](long a, long b) { return (a + b - 1) / b; };
        double B  = (double)sizeof(ComplexType);
        double gB = 1.0 / (1024.0 * 1024.0 * 1024.0);
        double m_dZ_aug = (double)ceild(_nqpts_ibz, nqp) * ceild(N_total, npP)
                        * ceild(N_total, npQ) * B;
        double m_dZ_sm  = (double)ceild(_nqpts_ibz, nqp) * ceild(_Np_smooth, npP)
                        * ceild(_Np_smooth, npQ) * B;   // alive during the embed
        double m_slab   = ((double)ceild(_Np_smooth, npPQ)
                        + 2.0 * ceild(_N_aug, npPQ)) * (double)ngm_rho * B;
        double m_gath   = ((double)ceild(_Np_smooth, npP) + ceild(_Np_smooth, npQ)
                        + 3.0 * ceild(_N_aug, std::min(npP, npQ)))
                        * (double)std::min(gchunk_rho, ngm_rho) * B;
        double m_Xshm   = (double)(_ns_in_basis * _npol_in_basis) * _nkpts
                        * N_total * (double)x_range.size() * B;  // per NODE (shm)
        double peak = m_dZ_aug + m_dZ_sm + m_slab + m_gath;
        app_log(2, "  paw_aug: per-rank memory estimate: _dZ(aug) {:.2f} GB "
                   "(+{:.2f} GB smooth during embed), q-pool slabs {:.2f} GB, "
                   "gather buffers {:.2f} GB; X_shm per node {:.2f} GB",
                m_dZ_aug*gB, m_dZ_sm*gB, m_slab*gB, m_gath*gB, m_Xshm*gB);
        double have_gb = (double)utils::freemem() / 1024.0;  // freemem() is MB
        if (peak * gB > have_gb)
          app_warning("paw_aug: estimated per-rank augmentation peak "
                      "({:.2f} GB) exceeds available memory ({:.2f} GB).",
                      peak*gB, have_gb);
      }

      utils::check(dzeta_quG.global_shape()[2] == ngm_rho,
        "thc_reader::augment: dzeta_quG G-dim ({}) != rho_g.size ({}). "
        "Augmentation requires orb_on_fft_grid mode (ζ in G-space).",
        dzeta_quG.global_shape()[2], ngm_rho);

      // ----------------------------------------------------------------
      // 1) Lift the IBZ-stored Pskna to the full BZ via View-2 symmetry.
      //    Already SHM (one shared copy per node). This is a 4D quantity
      //    (nspin × nkpts × nkb × nbnd); not an aux index. Kept as-is.
      // ----------------------------------------------------------------
      _Timer.start("PAW_AUG.Pskna_lift");
      // A4 deferral resolved: _psp comes from make_pseudopot(*_MF)
      // and pseudopot stores mf.mpi(), so whenever the two context handles
      // are the same object the psp's shm-backed Pskna cache is collective
      // on exactly this communicator — use it (one shared lift for THC,
      // direct v_x and becsum). A psp injected on a different context (none
      // today) falls back to the explicit build on _mpi.
      std::optional<math::shm::shared_array<nda::array_view<ComplexType,4>>>
          Pk_fallback;
      if (_psp->get_mpi_context() != _mpi) {
        auto symm_list_local = _MF->symm_list();
        Pk_fallback.emplace(hamilt::paw::compute_Pskna_full_bz(
            *_psp, _MF->kp_to_ibz(), _MF->kp_symm(), _MF->kp_trev(),
            _MF->kpts(), _MF->lattv(), _MF->recv(), symm_list_local,
            _npol, *_mpi));
      }
      auto const& Pkfull = Pk_fallback ? *Pk_fallback : _psp->Pskna_full_bz();
      _Timer.stop("PAW_AUG.Pskna_lift");

      // ----------------------------------------------------------------
      // 2) Augment X_shm: append Y rows below smooth ζ rows.
      //    X_shm has shape (dim_s, nkpts, N_total, x_range). Only one
      //    aux index (Np), with two large outer axes. SHM is the right
      //    storage. Fill the new (s, k) × aug-rows in parallel across
      //    the node-comm so the per-node augmentation cost scales with
      //    ranks_per_node, not with one rank doing it serially.
      // ----------------------------------------------------------------
      _Timer.start("PAW_AUG.X_aug");
      {
        auto X_old = _X_shm;
        int dim_s = _ns_in_basis * _npol_in_basis;
        auto X_new = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {(long)dim_s, (long)_nkpts, N_total, (long)x_range.size()});
        X_new.win().fence();
        auto Xn = X_new.local();
        auto Xo = X_old.local();
        auto Pkfull_loc = Pkfull.local();
        long nsk = (long)dim_s * (long)_nkpts;
        auto* node_comm = X_new.node_comm();
        long nrank = (long)node_comm->size();
        long my_rank = (long)node_comm->rank();
        auto [sk_o, sk_e] = itertools::chunk_range(0, nsk, nrank, my_rank);
        nda::array<ComplexType,2> Y_buf(_N_aug,
            x_range.size() * std::max(1, _npol));
        for (long sk = sk_o; sk < sk_e; ++sk) {
          int s = (int)(sk / _nkpts);
          int k = (int)(sk % _nkpts);
          for (int mu = 0; mu < _Np_smooth; ++mu)
            Xn(s, k, mu, range::all) = Xo(s, k, mu, range::all);
          hamilt::paw::fill_Y_rows_for_sk(
              *_psp, _isdf, _aug_layout, _npol,
              s, k, Pkfull_loc, Y_buf);
          for (int la = 0; la < _N_aug; ++la)
            for (int i = 0; i < (int)x_range.size(); ++i)
              Xn(s, k, _Np_smooth + la, i) = Y_buf(la, i);
        }
        X_new.win().fence();
        _X_shm = X_new;
      }
      _Timer.stop("PAW_AUG.X_aug");

      utils::check(x_range == y_range || !_Y_shm.has_value(),
        "thc_reader::augment_thc_with_paw: x_range != y_range augmentation "
        "is not yet supported.");

      // ----------------------------------------------------------------
      // 3) Allocate augmented `_dZ` distributed (nqpts_ibz, N_total, N_total).
      //    Save the smooth one for the GG-block embed below.
      // ----------------------------------------------------------------
      auto _dZ_smooth = std::move(_dZ);
      long np = (long)_mpi->comm.size();
      long nqpools = _dZ_smooth.grid()[0]; 
      long np_P = _dZ_smooth.grid()[1]; 
      long np_Q = _dZ_smooth.grid()[2];
      long np_PQ = np_P*np_Q;
      long bsize = std::min(1024l, std::min(N_total/np_P, N_total/np_Q));
      utils::check(nqpools > 0 && np % nqpools == 0 && np == nqpools*np_PQ,
                   "thc_reader::augment: bad pgrid (np={}, nqpts_ibz={}, nqpools:{}, np_PQ:{})",
                   np, _nqpts_ibz, nqpools, np_PQ);
      _dZ = math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(
          _mpi->comm, {nqpools, np_P, np_Q},
          {(long)_nqpts_ibz, N_total, N_total},{1,bsize,bsize}); 
      _dZ.local()() = ComplexType(0.0);

      // ----------------------------------------------------------------
      // 4) Embed smooth GG block into the new _dZ. The two distributed
      //    arrays differ both in proc grid and in the per-axis chunk
      //    sizes (Np_smooth vs N_total), so we cannot reuse `redistribute`
      //    directly — implement an alltoallv block-into-subblock helper.
      // ----------------------------------------------------------------
      _Timer.start("PAW_AUG.gather_smooth");
      embed_smooth_block_into_aug_dZ(_dZ_smooth, _dZ);
      _dZ_smooth.reset();
      _Timer.stop("PAW_AUG.gather_smooth");

      // ----------------------------------------------------------------
      // 5) Build the small angular-coupling tables (replicated, ~kB) and
      //    per-species qrad table (replicated, small).
      // ----------------------------------------------------------------
      _Timer.start("PAW_AUG.aainit");
      // Shared A4 runtime cache: same lli formula (1 + max l over
      // all projectors) as the local build this replaces; built once per
      // pseudopot and reused by every consumer (direct v_x, becsum, here).
      auto const& aatab = _psp->paw_aatab();
      _Timer.stop("PAW_AUG.aainit");

      double K_max_g = 0.0;
      {
        auto const& gv = rho_g.g_vectors();
        for (long ig = 0; ig < ngm_rho; ++ig) {
          double g2 = gv(ig,0)*gv(ig,0) + gv(ig,1)*gv(ig,1) + gv(ig,2)*gv(ig,2);
          K_max_g = std::max(K_max_g, std::sqrt(g2));
        }
      }
      double q_cart_max = 0.0;
      {
        auto Qpc = _MF->Qpts();
        for (long iq = 0; iq < _nqpts_ibz; ++iq) {
          double q2 = Qpc(iq,0)*Qpc(iq,0) + Qpc(iq,1)*Qpc(iq,1) + Qpc(iq,2)*Qpc(iq,2);
          q_cart_max = std::max(q_cart_max, std::sqrt(q2));
        }
      }

      // ---- Dense-sphere G-grid for the atom-local (LL) Coulomb block.
      // The LL block integrates η·w·conj(η) over G (conjugate on the SECOND
      // index — see 6h). η (the compensation charge
      // in moment mode; the SHARP full AE−PS pair density in shape mode) is NOT
      // band-limited to the thc rho_g sphere — an ecut-reduced rho_g is a
      // legitimate cost knob for the smooth/ISDF machinery, but truncating the
      // LL sum there loses the on-site physics (measured for shape mode on
      // Si a10.20: a ~60 Ry rho_g captured only ~6% of the on-site
      // correction). The ζ-coupled blocks (GG, GL,
      // LG) stay on rho_g: ζ is built by FFT from the rho_g collocation grid
      // and is therefore EXACTLY band-limited to that sphere, so those sums are
      // already complete for the THC representation — computing every block on
      // the dense sphere would change nothing but the LL block. The composite V
      // thus remains a genuine Coulomb Gram matrix over {ζ (band-limited), η
      // (dense)} — PSD by construction. K_a is grid-free. The dense sphere is
      // the inscribed sphere of the augmentation FFT box (fft_grid_dim_aug),
      // the same Gcut bound the direct v_x path uses (v_x_paw.hpp), which
      // makes the two routes consistent. When rho_g already covers
      // it, the existing single-grid path is used unchanged.
      double Gcut_aug = 1e30;
      {
        auto mesh_aug = _MF->fft_grid_dim_aug();
        auto recv = _MF->recv();
        for (int i = 0; i < 3; ++i) {
          long Mi = mesh_aug(i) / 2;
          double brow = std::sqrt(recv(i,0)*recv(i,0) + recv(i,1)*recv(i,1)
                                + recv(i,2)*recv(i,2));
          Gcut_aug = std::min(Gcut_aug, (double)Mi * brow);
        }
      }
      double ecut_dense = 0.5 * Gcut_aug * Gcut_aug;
      bool dense_LL = ecut_dense > rho_g.ecut() * (1.0 + 1e-8);
      std::optional<grids::truncated_g_grid> dense_g_opt;
      if (dense_LL) {
        dense_g_opt.emplace(ecut_dense, _MF->fft_grid_dim_aug(), _MF->recv());
        app_log(1, "  paw_aug: LL block on the DENSE augmentation sphere "
                   "(Gcut={:.2f} a.u., ngm={}; rho_g ecut={:.2f} Ha, ngm={})",
                Gcut_aug, dense_g_opt->size(), rho_g.ecut(), ngm_rho);
      }
      // Optional |q+G| cutoff on the augmentation channels (see the
      // _paw_aug_ecut comment at the option parse). Applied to eta on every
      // grid it is built on, so {zeta, eta_cut} stays a consistent function
      // set and Z stays its Coulomb Gram matrix.
      const bool aug_cut = (_paw_aug_ecut > 0.0);
      const double aug_K2_max = aug_cut ? 2.0 * _paw_aug_ecut : 0.0;
      if (aug_cut)
        app_log(1, "  paw_aug: eta TRUNCATED at |q+G|^2/2 <= {:.3f} Ha "
                   "(|q+G| <= {:.3f} a.u.)", _paw_aug_ecut,
                std::sqrt(aug_K2_max));

      double K_max = (dense_LL ? Gcut_aug : K_max_g) + q_cart_max;

      _Timer.start("PAW_AUG.qrad_tab");
      // Compensation-density mode for the augmentation channels η, selected by
      // the toml `vv_compensation` option (see hamilt::paw_exx_options):
      //   moment (default) — moment-restored compensation charge
      //                      (build_qrad_tab) + K_a one-center kernel;
      //   shape            — full AE−PS partial-wave pair density
      //                      (build_qrad_tab_full_aeps, ABINIT's
      //                      phiphj−tphitphj oscillator), K_a dropped below
      //                      (the on-site exchange is already complete).
      // NOTE: the mode changes the SHARED THC ERI, so it affects Hartree and
      // correlation too, not just exchange. Shape mode requires the dense-
      // sphere LL treatment above whenever rho_g is ecut-reduced.
      bool paw_shape_restored =
          (_exx_opts.vv_compensation == hamilt::vv_compensation_e::shape);
      if (paw_shape_restored)
        app_log(1, "  paw_aug: SHAPE-RESTORED (Option A) on-site exchange enabled "
                   "(full AE−PS oscillator, K_a dropped).");
      // Shared A4 runtime cache: same per-species selection
      // ((shape && is_paw) -> full-AEPS table, else compensation), same
      // project-wide dq = 0.01, same aug_lmax (the reader's _exx_opts were
      // propagated to the psp in prepare_paw_isdf). Tables built to a larger
      // Kmax are reused; the direct v_x path shares this exact table set.
      auto const& qrad_tabs = _psp->paw_qrad_tabs(K_max, paw_shape_restored);
      _Timer.stop("PAW_AUG.qrad_tab");
      app_log(2, "  paw_aug: qrad tables (shared cache) for {} species, "
                 "K_max={:.2f} a.u., n_K={}", qrad_tabs.size(), K_max,
              qrad_tabs.empty() ? 0L : qrad_tabs.front().n_K);

      double omega_sq = omega * omega;

      // ----------------------------------------------------------------
      // 6) q-pool subcommunicator: all ranks at the same i_qpool collaborate
      //    on the q's owned by that pool. Inside a pool the (np_P, np_Q)
      //    proc grid splits the (P, Q) axes of `_dZ`.
      // ----------------------------------------------------------------
      int q_color = (int)_dZ.origin()[0];
      mpi3::communicator q_intra = _mpi->comm.split(q_color, _mpi->comm.rank());
      utils::check((long)q_intra.size() == np_PQ,
                   "thc_reader::augment: q-pool sub-comm size mismatch (got {}, expected {})",
                   q_intra.size(), np_PQ);

      auto Qpts_cart = _MF->Qpts();
      auto qloc_new  = _dZ.local_range(0);
      auto Ploc_new  = _dZ.local_range(1);
      auto Qloc_new  = _dZ.local_range(2);

      auto intersect = [](nda::range a, nda::range b) {
        long o = std::max(a.first(), b.first());
        long e = std::min(a.last(),  b.last());
        if (e < o) e = o;
        return nda::range(o, e);
      };

      // Sub-tile global ranges this rank is responsible for in `_dZ`:
      //   smooth μ on the P axis  /  smooth μ on the Q axis  (V_GL, V_LG, V_LG-conj)
      //   aug    λ on the P axis  /  aug    λ on the Q axis  (V_LL)
      nda::range P_smooth_rows = intersect(Ploc_new, range(0, _Np_smooth));
      nda::range Q_smooth_cols = intersect(Qloc_new, range(0, _Np_smooth));
      nda::range P_aug_rows_g  = intersect(Ploc_new, range(_Np_smooth, N_total));
      nda::range Q_aug_cols_g  = intersect(Qloc_new, range(_Np_smooth, N_total));
      nda::range P_aug_rows_la(P_aug_rows_g.first() - _Np_smooth,
                               P_aug_rows_g.last()  - _Np_smooth);
      nda::range Q_aug_cols_la(Q_aug_cols_g.first() - _Np_smooth,
                               Q_aug_cols_g.last()  - _Np_smooth);

      // Per-q distributed scratch lives only inside the q-pool comm.
      // Rows split np_PQ ways, all ngm cols on each rank — keeps the layout
      // simple for the row-gathers below. Per-rank memory:
      //   zeta_dist : (Np_smooth/np_PQ) × ngm × 16 B
      //   eta_dist  : (N_aug   /np_PQ) × ngm × 16 B
      auto zeta_dist  = math::nda::make_distributed_array<Array_t<HOST_MEMORY,2>>(
          q_intra, {np_PQ, 1L}, {(long)_Np_smooth, ngm_rho});
      auto eta_dist   = math::nda::make_distributed_array<Array_t<HOST_MEMORY,2>>(
          q_intra, {np_PQ, 1L}, {(long)_N_aug,    ngm_rho});
      auto eta_w_dist = math::nda::make_distributed_array<Array_t<HOST_MEMORY,2>>(
          q_intra, {np_PQ, 1L}, {(long)_N_aug,    ngm_rho});
//MAM: need to check above that np_PQ is smaller than _N_aug

      // Rank-local row buffers for this rank's _dZ tile sub-blocks,
      // G-CHUNKED: (P/Q chunk rows) × gchunk instead of × ngm.
      // At production sizes (tile rows ~10³–10⁴, ngm ~10⁵–10⁶) the
      // full-width buffers were the N_aug ≳ 10k OOM; the GEMMs below
      // accumulate over the g-chunks instead. Same 8192 chunk the dense-LL
      // branch uses (gchunk_rho defined with the memory estimate above).
      const long gbuf_rho = std::min(gchunk_rho, ngm_rho);
      nda::array<ComplexType,2> zeta_P_smooth_g(P_smooth_rows.size(), gbuf_rho);
      nda::array<ComplexType,2> zeta_Q_smooth_g(Q_smooth_cols.size(), gbuf_rho);
      nda::array<ComplexType,2> eta_w_Q_aug_g (Q_aug_cols_la.size(),   gbuf_rho);
      nda::array<ComplexType,2> eta_w_P_aug_g (P_aug_rows_la.size(),   gbuf_rho);
      nda::array<ComplexType,2> eta_P_aug_conj(P_aug_rows_la.size(),   gbuf_rho);
      // Accumulation tiles for this rank's _dZ sub-blocks (small: rows×cols).
      nda::array<ComplexType,2> V_GL_local(P_smooth_rows.size(), Q_aug_cols_la.size());
      nda::array<ComplexType,2> V_LG_local(P_aug_rows_la.size(), Q_smooth_cols.size());
      nda::array<ComplexType,2> V_LL_local(P_aug_rows_la.size(), Q_aug_cols_la.size());

      for (auto [iq_l, iq] : itertools::enumerate(qloc_new)) {
        std::array<double,3> q_cart = {
            -Qpts_cart(iq, 0), -Qpts_cart(iq, 1), -Qpts_cart(iq, 2)};

        // ---- 6a) Pull this q's ζ slab into zeta_dist (q-pool, row-split). ----
        _Timer.start("PAW_AUG.dzeta");
        fill_zeta_for_iq_into_qpool(dzeta_quG, (int)iq, zeta_dist, q_intra);
        _Timer.stop("PAW_AUG.dzeta");

        // ---- 6b) Build η^q for this rank's la_chunk. ----
        _Timer.start("PAW_AUG.eta_at_q");
        {
          auto la_rng = eta_dist.local_range(0);
          auto eta_loc = eta_dist.local();
          hamilt::paw::build_eta_on_rho_g_at_q_chunk(
              *_psp, _isdf, _aug_layout, rho_g, q_cart, omega,
              aatab, qrad_tabs,
              la_rng, range(0, ngm_rho), eta_loc);
          if (aug_cut) {
            auto const& gv = rho_g.g_vectors();
            for (long g = 0; g < ngm_rho; ++g) {
              double Kx = q_cart[0] + gv(g, 0);
              double Ky = q_cart[1] + gv(g, 1);
              double Kz = q_cart[2] + gv(g, 2);
              if (Kx*Kx + Ky*Ky + Kz*Kz <= aug_K2_max) continue;
              for (long la = 0; la < eta_loc.shape(0); ++la)
                eta_loc(la, g) = ComplexType(0.0);
            }
          }
        }
        _Timer.stop("PAW_AUG.eta_at_q");

        // ---- 6c) Coulomb weights wG^q (rank-local copy, ngm doubles). ----
        _Timer.start("PAW_AUG.wG_at_q");
        auto wG_q = hamilt::paw::coulomb_weights_on_rho_g_at_q(rho_g, q_cart, omega);
        _Timer.stop("PAW_AUG.wG_at_q");

        // ---- 6d) η_w = η * wG (local). ----
        _Timer.start("PAW_AUG.eta_flat");
        {
          auto src = eta_dist.local();
          auto dst = eta_w_dist.local();
          for (long la = 0; la < src.shape(0); ++la)
            for (long g = 0; g < ngm_rho; ++g)
              dst(la, g) = src(la, g) * wG_q(g);
        }
        _Timer.stop("PAW_AUG.eta_flat");

        // ---- 6e) G-chunked row-gathers + GEMM accumulation. ----
        // Row-gathers on q_intra (np_PQ-way) one g-chunk at a time; ALL
        // ranks run the same chunk loop (the gathers are collective) and
        // take the same dense_LL branch (pure function of the grids —
        // uniform).
        V_GL_local() = ComplexType(0.0);
        V_LG_local() = ComplexType(0.0);
        V_LL_local() = ComplexType(0.0);
        for (long gr0 = 0; gr0 < ngm_rho; gr0 += gchunk_rho) {
          nda::range g_rng(gr0, std::min(gr0 + gchunk_rho, ngm_rho));
          long ng_c = g_rng.size();
          auto zP  = zeta_P_smooth_g(range::all, range(0, ng_c));
          auto zQ  = zeta_Q_smooth_g(range::all, range(0, ng_c));
          auto ewQ = eta_w_Q_aug_g (range::all, range(0, ng_c));
          auto ewP = eta_w_P_aug_g (range::all, range(0, ng_c));
          auto ePc = eta_P_aug_conj(range::all, range(0, ng_c));
          gather_rows_from_dist_qpool(zeta_dist , P_smooth_rows, q_intra, zP,  g_rng);
          gather_rows_from_dist_qpool(zeta_dist , Q_smooth_cols, q_intra, zQ,  g_rng);
          gather_rows_from_dist_qpool(eta_w_dist, Q_aug_cols_la, q_intra, ewQ, g_rng);
          gather_rows_from_dist_qpool(eta_w_dist, P_aug_rows_la, q_intra, ewP, g_rng);
          if (!dense_LL) {
            // η rows for the rho_g V_LL GEMM (dense_LL builds its own tiles).
            // Kept UNCONJUGATED: the V_LL GEMM below conjugates the second
            // index via dagger(ewQ), matching the GG/GL convention. (It used
            // to be conjugated here and paired with transpose(ewQ), which
            // stored the transpose of V_LL.)
            gather_rows_from_dist_qpool(eta_dist, P_aug_rows_la, q_intra, ePc, g_rng);
          }

          // V_GL(μ, λ) = Ω · Σ_g ζ(μ, g) · conj(η_w(λ, g)).  (gate _paw_vgl)
          if (_paw_vgl && zP.shape(0) > 0 && ewQ.shape(0) > 0) {
            _Timer.start("PAW_AUG.V_GL");
            nda::blas::gemm(ComplexType(omega), zP, nda::dagger(ewQ),
                            ComplexType(1.0), V_GL_local);
            _Timer.stop("PAW_AUG.V_GL");
          }
          // V_LG(λ, μ) = conj(V_GL(μ, λ)) = Ω · Σ_g η_w(λ, g) · conj(ζ(μ, g)).
          if (_paw_vgl && ewP.shape(0) > 0 && zQ.shape(0) > 0) {
            _Timer.start("PAW_AUG.V_GL");
            nda::blas::gemm(ComplexType(omega), ewP, nda::dagger(zQ),
                            ComplexType(1.0), V_LG_local);
            _Timer.stop("PAW_AUG.V_GL");
          }
          // V_LL rho_g branch: Ω² · Σ_g η(λ, g) · w(g) · conj(η(ξ, g)).
          //
          // CONJUGATE THE SECOND INDEX. That is the convention the rest of Z
          // uses -- the smooth block is multiply(Z_quG, dagger(Z_quG)) in
          // thc.icc, i.e. Z_uv = Σ_G ζ_u v conj(ζ_v), and V_GL above follows
          // it with gemm(ω, zP, dagger(ewQ)). This block used to conjugate the
          // FIRST index instead, which (V_LL being Hermitian) silently stored
          // its TRANSPOSE: diagonal elements are real so they were unaffected,
          // and every diagonal-only test passed, but off-diagonal ERIs (ia|bj)
          // came out transposed. Tr(Pi*Z) uses only the diagonal and was fine;
          // ln|det(I-Pi*Z)| uses the whole matrix and carried the error --
          // measured at 39% on Si a=10.05 n=250, and thresh-independent
          // because V_LL contains no ζ.
          if (_paw_vll && !dense_LL && ePc.shape(0) > 0 && ewQ.shape(0) > 0) {
            _Timer.start("PAW_AUG.V_LL");
            nda::blas::gemm(ComplexType(omega_sq), ePc, nda::dagger(ewQ),
                            ComplexType(1.0), V_LL_local);
            _Timer.stop("PAW_AUG.V_LL");
          }
        }

        auto Z_loc = _dZ.local();

        // ---- 6f) Stitch V_GL: P ∈ smooth, Q ∈ aug.
        if (_paw_vgl && V_GL_local.shape(0) > 0 && V_GL_local.shape(1) > 0) {
          _Timer.start("PAW_AUG.stitch");
          for (long ir = 0; ir < V_GL_local.shape(0); ++ir) {
            long P_in_tile = P_smooth_rows.first() + ir - Ploc_new.first();
            for (long ic = 0; ic < V_GL_local.shape(1); ++ic) {
              long Q_in_tile = Q_aug_cols_g.first() + ic - Qloc_new.first();
              Z_loc(iq_l, P_in_tile, Q_in_tile) += V_GL_local(ir, ic);
            }
          }
          _Timer.stop("PAW_AUG.stitch");
        }

        // ---- 6g) V_LG block: P ∈ aug, Q ∈ smooth.
        //         V_LG(λ, μ) = conj(V_GL(μ, λ))
        //                    = Ω · Σ_g η_w(λ, g) · conj(ζ(μ, g)).
        if (_paw_vgl && V_LG_local.shape(0) > 0 && V_LG_local.shape(1) > 0) {
          _Timer.start("PAW_AUG.stitch");
          for (long ir = 0; ir < V_LG_local.shape(0); ++ir) {
            long P_in_tile = P_aug_rows_g.first() + ir - Ploc_new.first();
            for (long ic = 0; ic < V_LG_local.shape(1); ++ic) {
              long Q_in_tile = Q_smooth_cols.first() + ic - Qloc_new.first();
              Z_loc(iq_l, P_in_tile, Q_in_tile) += V_LG_local(ir, ic);
            }
          }
          _Timer.stop("PAW_AUG.stitch");
        }

        // ---- 6h) V_LL block + on-site K_a: P ∈ aug, Q ∈ aug.
        //         V_LL(λ, ξ) = Ω² · Σ_g η(λ, g) · w(g) · conj(η(ξ, g)).
        // The conjugate sits on the SECOND index, matching GG and GL. Getting
        // this backwards stores Z^T, which is invisible to every diagonal test
        // (the diagonal of a Hermitian matrix is real) and to a Hermiticity
        // check (the transposed block is itself Hermitian) — it corrupts only
        // the off-diagonal (ia|bj), i.e. only ln|det(I-Pi*Z)|. That was the
        // RPA instability. Do not "restore" the earlier ordering.
        // over the DENSE augmentation sphere when dense_LL: η is
        // not band-limited to rho_g, so the Coulomb sum must run to the
        // physical Gcut of the augmentation mesh (built in G-chunks, each
        // rank evaluating exactly the (row, col) × g tiles it needs — the
        // η builder is a cheap replicated interpolation, no comm).
        if (P_aug_rows_la.size() > 0 && Q_aug_cols_la.size() > 0) {
          _Timer.start("PAW_AUG.V_LL");
          if (_paw_vll && dense_LL) {
            auto const& dense_g = *dense_g_opt;
            long ngm_d = dense_g.size();
            auto const& gv_d = dense_g.g_vectors();
            const long gchunk = 8192;
            long gbuf = std::min(gchunk, ngm_d);
            nda::array<ComplexType,2> etaP_c (P_aug_rows_la.size(), gbuf);
            nda::array<ComplexType,2> etaQ_w (Q_aug_cols_la.size(), gbuf);
            for (long g0 = 0; g0 < ngm_d; g0 += gchunk) {
              nda::range g_rng(g0, std::min(g0 + gchunk, ngm_d));
              long ng_c = g_rng.size();
              auto etaP_v = etaP_c(nda::range::all, nda::range(0, ng_c));
              auto etaQ_v = etaQ_w(nda::range::all, nda::range(0, ng_c));
              hamilt::paw::build_eta_on_rho_g_at_q_chunk(
                  *_psp, _isdf, _aug_layout, dense_g, q_cart, omega,
                  aatab, qrad_tabs, P_aug_rows_la, g_rng, etaP_v);
              hamilt::paw::build_eta_on_rho_g_at_q_chunk(
                  *_psp, _isdf, _aug_layout, dense_g, q_cart, omega,
                  aatab, qrad_tabs, Q_aug_cols_la, g_rng, etaQ_v);
              // η_Q × w(q+G) with w = 4π/(Ω|q+G|²), w(0)=0 (same singular-term
              // convention as the rho_g path). η_P is NOT conjugated here —
              // the dagger below puts the conjugate on the second index.
              for (long ig = 0; ig < ng_c; ++ig) {
                long g = g_rng.first() + ig;
                double Kx = q_cart[0] + gv_d(g, 0);
                double Ky = q_cart[1] + gv_d(g, 1);
                double Kz = q_cart[2] + gv_d(g, 2);
                double K2 = Kx*Kx + Ky*Ky + Kz*Kz;
                double w = (K2 > 1e-14) ? (4.0*M_PI/(omega*K2)) : 0.0;
                // Same eta truncation as the rho_g branch above: zero the
                // channel itself, not the kernel, so both tiles describe the
                // identical cut function set.
                if (aug_cut and K2 > aug_K2_max) w = 0.0;
                // etaP stays UNCONJUGATED; the GEMM conjugates the second
                // index via dagger(etaQ_v), matching the GG/GL convention.
                if (aug_cut and K2 > aug_K2_max)
                  for (long la = 0; la < etaP_v.shape(0); ++la)
                    etaP_v(la, ig) = ComplexType(0.0);
                for (long la = 0; la < etaQ_v.shape(0); ++la)
                  etaQ_v(la, ig) *= w;
              }
              // V_LL(λ,ξ) = Ω² Σ_g η(λ,g) w(g) conj(η(ξ,g)) -- see the rho_g
              // branch above for why the conjugation sits on the SECOND index.
              nda::blas::gemm(ComplexType(omega_sq), etaP_v,
                              nda::dagger(etaQ_v),
                              ComplexType(1.0), V_LL_local);
            }
          }
          // (rho_g V_LL branch already accumulated in the 6e chunk loop.)
          _Timer.stop("PAW_AUG.V_LL");

          if (_paw_onsite && !paw_shape_restored) {
            _Timer.start("PAW_AUG.K_a");
            hamilt::paw::add_K_a_to_tile(
                *_psp, _isdf, _aug_layout,
                P_aug_rows_la, Q_aug_cols_la, V_LL_local);
            _Timer.stop("PAW_AUG.K_a");
          }

          _Timer.start("PAW_AUG.stitch");
          for (long ir = 0; ir < V_LL_local.shape(0); ++ir) {
            long P_in_tile = P_aug_rows_g.first() + ir - Ploc_new.first();
            for (long ic = 0; ic < V_LL_local.shape(1); ++ic) {
              long Q_in_tile = Q_aug_cols_g.first() + ic - Qloc_new.first();
              Z_loc(iq_l, P_in_tile, Q_in_tile) += V_LL_local(ir, ic);
            }
          }
          _Timer.stop("PAW_AUG.stitch");
        }
      }

      // ----------------------------------------------------------------
      // 7) Extend the G=0 head vectors with the augmentation rows (plan
      //    D1). The smooth _Chi_head(q,u) stores conj(ζ^q_u(G=0)) (see
      //    thc.icc intvec_impl); the matching aug entry is
      //    conj(Ω·η^q_Λ(G=0)) — Ω·η is the aug object in ζ units
      //    (V_GL = (1/Ω)·Σ_g ζ·v·conj(Ω·η), same −q convention).
      //    Without these rows every consumer that pairs basis_head() with
      //    Np() (GW Sigma_div_correction, embed_eri/g0_div head builds)
      //    indexes past the smooth-only array once augmentation bumps Np,
      //    and the pair-density monopole of the compensation charges is
      //    dropped from the q→0 head (under I8 the head reconstructs the
      //    AE δ_ij only as smooth+aug).
      //    _Chi_bar_head (dual of the FIT basis): aug rows stay ZERO — the
      //    G=0 plane wave is exactly band-limited to the smooth sphere, so
      //    its smooth-only LS representation remains a valid representation
      //    in the enlarged basis (exact up to the smooth ISDF fit residual);
      //    zero aug coefficients preserve it.
      // ----------------------------------------------------------------
      {
        auto const& gv_to_fft = rho_g.gv_to_fft();
        long g0_idx = -1;
        for (long ig = 0; ig < ngm_rho; ++ig)
          if (gv_to_fft(ig) == 0) { g0_idx = ig; break; }
        utils::check(g0_idx >= 0,
                     "thc_reader::augment: G=0 not found on the rho_g grid.");
        auto g0_vec = rho_g.g_vectors(g0_idx);
        utils::check(g0_vec(0) == 0.0 and g0_vec(1) == 0.0 and g0_vec(2) == 0.0,
                     "thc_reader::augment: gv_to_fft==0 entry is not G=(0,0,0).");

        nda::array<ComplexType,2> Chi_new(_nqpts_ibz, N_total);
        nda::array<ComplexType,2> Chi_bar_new(_nqpts_ibz, N_total);
        Chi_new() = ComplexType(0.0);
        Chi_bar_new() = ComplexType(0.0);
        Chi_new(range::all, range(0, _Np_smooth)) = _Chi_head;
        Chi_bar_new(range::all, range(0, _Np_smooth)) = _Chi_bar_head;

        nda::array<ComplexType,2> eta_g0(_N_aug, 1);
        for (long iq = 0; iq < _nqpts_ibz; ++iq) {
          std::array<double,3> q_cart = {
              -Qpts_cart(iq, 0), -Qpts_cart(iq, 1), -Qpts_cart(iq, 2)};
          hamilt::paw::build_eta_on_rho_g_at_q_chunk(
              *_psp, _isdf, _aug_layout, rho_g, q_cart, omega,
              aatab, qrad_tabs,
              range(0, _N_aug), range(g0_idx, g0_idx + 1), eta_g0);
          for (long la = 0; la < _N_aug; ++la)
            Chi_new(iq, _Np_smooth + la) = std::conj(omega * eta_g0(la, 0));
        }
        _Chi_head = std::move(Chi_new);
        _Chi_bar_head = std::move(Chi_bar_new);
      }

      _Np = (int)N_total;
    }

    // ------------------------------------------------------------------
    // PAW augmentation helpers (private, used only by augment_thc_with_paw).
    // ------------------------------------------------------------------

    /**
     * Re-select the augmentation channels using a q-AWARE criterion.
     *
     * `_isdf` comes in at full rank. For every (I,J) pair we form
     *
     *     d_IJ = max over q on the mesh of  sum_G w_q(G) |eta_IJ(q+G)|^2,
     *     w_q(G) = 4pi / (Omega |q+G|^2),  the |q+G| -> 0 term dropped,
     *
     * and keep the pair when sqrt(d_IJ / max_IJ d_IJ) exceeds `_paw_isdf_tol`.
     *
     * NOTE this changes what `paw_isdf_tol` MEANS. The old criterion in
     * `build_local_isdf_compressed_by_norm` was ABSOLUTE (`d > tol^2` on the
     * raw q=0 norm); this one is RELATIVE to the largest pair. Absolute is not
     * transferable — d carries 4pi/Omega, so the same number means different
     * things at different cell volumes, which is precisely the wrong property
     * for a knob used across an EOS volume series. Relative is
     * system-independent. Consequence for anyone reading old inputs: a given
     * numeric tol is not equivalent between the two rules.
     *
     * Why max-over-q rather than the q=0 value: the ERI sums over the whole
     * q-mesh, and a channel that is negligible at q=0 purely because
     * Q_IJ(G) -> 0 as G -> 0 (any pair with no L=0 component) is NOT
     * negligible at finite q, where the smallest wavevector is |q| rather than
     * zero. Ranking on q=0 alone silently drops exactly those channels.
     *
     * eta is evaluated with the same interpolated tables the augmentation
     * uses, over the same rho_g sphere, so this reuses the production path
     * rather than a parallel approximation of it.
     */
    void select_aug_channels_qaware(nda::stack_array<double,3,3> const& recv,
                                    double omega)
    {
      auto const& sps = _psp->paw_species_view();
      int nsp = (int)sps.size();
      // Full-rank layout, so lambda rows map onto the pairs we are ranking.
      auto layout = hamilt::paw::make_paw_aug_layout(*_psp, _isdf, 0);
      long N_aug = layout.N_A;
      if (N_aug == 0) return;

      grids::truncated_g_grid rho_g(_MF->ecutrho(), _MF->fft_grid_dim_aug(), recv);
      long ngm = rho_g.size();
      auto const& gv = rho_g.g_vectors();
      double Gmax = 0.0;
      for (long g = 0; g < ngm; ++g)
        Gmax = std::max(Gmax, std::sqrt(gv(g,0)*gv(g,0)+gv(g,1)*gv(g,1)+gv(g,2)*gv(g,2)));

      auto Qpts = _MF->Qpts();
      long nq = Qpts.extent(0);
      double qmax = 0.0;
      for (long iq = 0; iq < nq; ++iq)
        qmax = std::max(qmax, std::sqrt(Qpts(iq,0)*Qpts(iq,0) +
                                        Qpts(iq,1)*Qpts(iq,1) +
                                        Qpts(iq,2)*Qpts(iq,2)));
      auto const& aatab = _psp->paw_aatab();
      auto const& qtabs = _psp->paw_qrad_tabs(Gmax + qmax + 1e-3, false);

      // Worst-case (over q) Coulomb norm of every lambda row. `dzero` keeps the
      // q=0 value alone so we can report how badly the old q=0-only ranking
      // would have mis-ordered the channels on THIS system.
      nda::array<double,1> dmax(N_aug), dzero(N_aug);
      dmax() = 0.0; dzero() = 0.0;
      {
        const long gch = 8192;
        nda::array<ComplexType,2> eta(N_aug, std::min(gch, ngm));
        for (long iq = 0; iq < nq; ++iq) {
          std::array<double,3> q_cart = {-Qpts(iq,0), -Qpts(iq,1), -Qpts(iq,2)};
          nda::array<double,1> acc(N_aug);
          acc() = 0.0;
          for (long g0 = 0; g0 < ngm; g0 += gch) {
            nda::range gr(g0, std::min(g0+gch, ngm));
            auto sub = eta(nda::range::all, nda::range(0, gr.size()));
            hamilt::paw::build_eta_on_rho_g_at_q_chunk(
                *_psp, _isdf, layout, rho_g, q_cart, omega, aatab, qtabs,
                nda::range(0, N_aug), gr, sub);
            for (long ig = 0; ig < gr.size(); ++ig) {
              long g = gr.first() + ig;
              double Kx = q_cart[0]+gv(g,0), Ky = q_cart[1]+gv(g,1), Kz = q_cart[2]+gv(g,2);
              double K2 = Kx*Kx + Ky*Ky + Kz*Kz;
              if (K2 <= 1e-14) continue;
              double w = 4.0*M_PI/(omega*K2);
              for (long la = 0; la < N_aug; ++la)
                acc(la) += w * std::norm(sub(la, ig));
            }
          }
          double q2 = Qpts(iq,0)*Qpts(iq,0) + Qpts(iq,1)*Qpts(iq,1)
                    + Qpts(iq,2)*Qpts(iq,2);
          for (long la = 0; la < N_aug; ++la) {
            dmax(la) = std::max(dmax(la), acc(la));
            if (q2 <= 1e-14) dzero(la) = acc(la);
          }
        }
      }

      // Collapse lambda -> (I,J) pair (the +/- rows of a pair share a norm),
      // then rebuild each species keeping the pairs above a RELATIVE tolerance.
      std::vector<hamilt::paw::species_local_isdf> out;
      out.reserve(nsp);
      long dropped_total = 0, kept_total = 0, rescued_total = 0;
      double worst_dropped_rel = 0.0, worst_rescued_ratio = 1.0;
      for (int nt = 0; nt < nsp; ++nt) {
        auto const& s = _isdf[nt];
        if (s.nlambda == 0) { out.emplace_back(); continue; }
        // First atom of this species gives the row offset.
        long row0 = -1;
        auto const& ityp = _psp->ityp_view();
        for (long ia = 0; ia < ityp.extent(0); ++ia)
          if (ityp(ia) == nt) { row0 = layout.atom_aug_offset[ia]; break; }
        utils::check(row0 >= 0, "select_aug_channels_qaware: no atom for species {}", nt);

        int nh_a = s.nh;
        long nij = (long)nh_a*(nh_a+1)/2;
        std::vector<double> dpair(nij, 0.0), zpair(nij, 0.0);
        for (int lam = 0; lam < s.nlambda; ++lam) {
          dpair[s.lambda_ij(lam)] = std::max(dpair[s.lambda_ij(lam)],
                                             dmax(row0 + lam));
          zpair[s.lambda_ij(lam)] = std::max(zpair[s.lambda_ij(lam)],
                                             dzero(row0 + lam));
        }
        double dbig = 0.0, zbig = 0.0;
        for (long ij = 0; ij < nij; ++ij) {
          dbig = std::max(dbig, dpair[ij]);
          zbig = std::max(zbig, zpair[ij]);
        }

        hamilt::paw::isdf_compression_report rep;
        rep.nij = (int)nij;
        hamilt::paw::detail::make_pair_map(nt, nh_a, _psp->ijtoh_view(),
                                           (int)nij, rep.ij_to_IJ_i, rep.ij_to_IJ_j);
        double thr2 = _paw_isdf_tol * _paw_isdf_tol * dbig;
        for (long ij = 0; ij < nij; ++ij) {
          if (rep.ij_to_IJ_i[ij] < 0) continue;      // selection-rule zero
          if (dpair[ij] > thr2) {
            rep.pair_pivot_order.push_back((int)ij); ++kept_total;
            // Would the q=0-only ranking have thrown this one away?
            if (zpair[ij] <= _paw_isdf_tol*_paw_isdf_tol*zbig) {
              ++rescued_total;
              worst_rescued_ratio = std::max(worst_rescued_ratio,
                  std::sqrt((dpair[ij]/std::max(dbig,1e-300)) /
                            std::max(zpair[ij]/std::max(zbig,1e-300), 1e-300)));
            }
          }
          else if (dpair[ij] > 0.0) {
            ++dropped_total;
            worst_dropped_rel = std::max(worst_dropped_rel,
                                         std::sqrt(dpair[ij]/std::max(dbig,1e-300)));
          }
        }
        out.push_back(hamilt::paw::build_local_isdf_compressed(
            *_psp, nt, nh_a, rep, (int)rep.pair_pivot_order.size()));
      }
      _isdf = std::move(out);
      // Never let truncation be silent: report what went and how big it was.
      app_log(1, "  paw_aug: q-aware channel selection over the full {}-point q "
                 "mesh -- kept {} (I,J) pairs, dropped {}; largest dropped is "
                 "{:.3e} of the largest kept (relative tol {:.1e}). Ranking is "
                 "max over q of the Coulomb norm, not the q=0 value: a pair "
                 "with no L=0 component has Q_IJ(G) -> 0 as G -> 0 and is "
                 "under-ranked at q=0, yet the ERI's smallest wavevector is "
                 "|q|, not 0.",
              nq, kept_total, dropped_total, worst_dropped_rel, _paw_isdf_tol);
      if (rescued_total > 0)
        app_log(1, "  paw_aug: {} of those pairs would have been DROPPED by the "
                   "old q=0-only ranking (largest under-ranked by {:.1f}x). "
                   "Those are the channels whose absence corrupts the "
                   "off-diagonal (ia|bj) that ln|det(I-Pi*Z)| sums over.",
                rescued_total, worst_rescued_ratio);
    }

    /**
     * Embed a smooth-block distributed array `dZ_NA` (shape: nq, NA, NA)
     * into the [0:NA, 0:NA] sub-block of `dZ_NB` (shape: nq, NB, NB),
     * NB ≥ NA. Both arrays live on the same global communicator but in
     * general have different proc grids and per-rank chunk shapes.
     *
     * Implementation: every rank knows its own (q, P, Q) origin/shape in
     * each array; allgather them and use one alltoallv to move data from
     * src-rank's local block to the destination ranks whose [0:NA, 0:NA]
     * subblock intersects it.
     */
    template<class DA, class DB>
    void embed_smooth_block_into_aug_dZ(DA const& dZ_NA, DB& dZ_NB)
    {
      auto* comm = dZ_NA.communicator();
      utils::check(comm == dZ_NB.communicator(),
        "embed_smooth_block_into_aug_dZ: communicator mismatch");
      long mpi_size = (long)comm->size();
      long mpi_rank = (long)comm->rank();

      long NA = dZ_NA.global_shape()[1];
      utils::check(dZ_NA.global_shape()[1] == dZ_NA.global_shape()[2] &&
                   dZ_NB.global_shape()[1] == dZ_NB.global_shape()[2] &&
                   dZ_NA.global_shape()[0] == dZ_NB.global_shape()[0] &&
                   NA <= dZ_NB.global_shape()[1],
        "embed_smooth_block_into_aug_dZ: shape mismatch");

      // Allgather (origin, shape) for both A and B from all ranks.
      // Layout: per-rank 4 rows (A.origin, A.shape, B.origin, B.shape),
      // each of length 3 → 12 longs per rank.
      nda::array<long,3> blocks(mpi_size, 4, 3);
      nda::array<long,2> mine(4, 3);
      std::copy_n(dZ_NA.origin().data(),       3, mine.data() + 0);
      std::copy_n(dZ_NA.local_shape().data(),  3, mine.data() + 3);
      std::copy_n(dZ_NB.origin().data(),       3, mine.data() + 6);
      std::copy_n(dZ_NB.local_shape().data(),  3, mine.data() + 9);
      comm->all_gather_n(mine.data(), 12, blocks.data(), 12);

      auto Aloc = dZ_NA.local();
      auto Bloc = dZ_NB.local();

      // Compute send/recv volumes. For each rank d, my contribution
      // is the intersection of my A local block with d's B local block
      // restricted to (q, [0:NA), [0:NA)).
      auto intersect = [](long a0, long aN, long b0, long bN) -> std::pair<long,long> {
        long o = std::max(a0, b0);
        long e = std::min(a0 + aN, b0 + bN);
        return {o, std::max(o, e)};
      };

      std::vector<int> send_counts(mpi_size, 0), send_displs(mpi_size, 0);
      std::vector<int> recv_counts(mpi_size, 0), recv_displs(mpi_size, 0);

      auto compute_overlap = [&](long /*Aorig*/, long /*Ashape*/,
                                 long ax_q_o, long ax_q_n,
                                 long ax_P_o, long ax_P_n,
                                 long ax_Q_o, long ax_Q_n,
                                 long bx_q_o, long bx_q_n,
                                 long bx_P_o, long bx_P_n,
                                 long bx_Q_o, long bx_Q_n,
                                 std::array<std::pair<long,long>,3>& rngs) -> long {
        auto [q_o, q_e] = intersect(ax_q_o, ax_q_n, bx_q_o, bx_q_n);
        // restrict B's P/Q range to [0, NA) before intersecting
        long bP_o = std::max(0L, bx_P_o);
        long bP_e = std::min(NA, bx_P_o + bx_P_n);
        long bQ_o = std::max(0L, bx_Q_o);
        long bQ_e = std::min(NA, bx_Q_o + bx_Q_n);
        if (bP_e <= bP_o || bQ_e <= bQ_o) return 0;
        auto [P_o, P_e] = intersect(ax_P_o, ax_P_n, bP_o, bP_e - bP_o);
        auto [Q_o, Q_e] = intersect(ax_Q_o, ax_Q_n, bQ_o, bQ_e - bQ_o);
        long n = (q_e - q_o) * (P_e - P_o) * (Q_e - Q_o);
        rngs = {std::pair{q_o, q_e}, std::pair{P_o, P_e}, std::pair{Q_o, Q_e}};
        return n;
      };

      // Send side: for each dest rank d, fill block (q, P, Q) of my A
      // into a flat send-buffer.
      std::vector<std::array<std::pair<long,long>,3>> send_rngs(mpi_size);
      std::vector<std::array<std::pair<long,long>,3>> recv_rngs(mpi_size);

      for (long d = 0; d < mpi_size; ++d) {
        long count = compute_overlap(0,0,
            blocks(mpi_rank,0,0), blocks(mpi_rank,1,0),
            blocks(mpi_rank,0,1), blocks(mpi_rank,1,1),
            blocks(mpi_rank,0,2), blocks(mpi_rank,1,2),
            blocks(d,2,0), blocks(d,3,0),
            blocks(d,2,1), blocks(d,3,1),
            blocks(d,2,2), blocks(d,3,2),
            send_rngs[d]);
        send_counts[d] = (int)count;
      }
      for (long s = 0; s < mpi_size; ++s) {
        long count = compute_overlap(0,0,
            blocks(s,0,0), blocks(s,1,0),
            blocks(s,0,1), blocks(s,1,1),
            blocks(s,0,2), blocks(s,1,2),
            blocks(mpi_rank,2,0), blocks(mpi_rank,3,0),
            blocks(mpi_rank,2,1), blocks(mpi_rank,3,1),
            blocks(mpi_rank,2,2), blocks(mpi_rank,3,2),
            recv_rngs[s]);
        recv_counts[s] = (int)count;
      }
      for (long d = 1; d < mpi_size; ++d)
        send_displs[d] = send_displs[d-1] + send_counts[d-1];
      for (long s = 1; s < mpi_size; ++s)
        recv_displs[s] = recv_displs[s-1] + recv_counts[s-1];

      long total_send = (long)send_displs.back() + (long)send_counts.back();
      long total_recv = (long)recv_displs.back() + (long)recv_counts.back();
      std::vector<ComplexType> sendbuf(total_send), recvbuf(total_recv);

      // Pack
      for (long d = 0; d < mpi_size; ++d) {
        if (send_counts[d] == 0) continue;
        auto const& r = send_rngs[d];
        long off = send_displs[d];
        long Aq0 = blocks(mpi_rank,0,0), AP0 = blocks(mpi_rank,0,1), AQ0 = blocks(mpi_rank,0,2);
        for (long q = r[0].first; q < r[0].second; ++q)
          for (long P = r[1].first; P < r[1].second; ++P)
            for (long Q = r[2].first; Q < r[2].second; ++Q)
              sendbuf[off++] = Aloc(q - Aq0, P - AP0, Q - AQ0);
      }

      comm->all_to_all_v_n(
          sendbuf.data(), send_counts.data(), send_displs.data(),
          recvbuf.data(), recv_counts.data(), recv_displs.data());

      // Unpack
      long Bq0 = blocks(mpi_rank,2,0), BP0 = blocks(mpi_rank,2,1), BQ0 = blocks(mpi_rank,2,2);
      for (long s = 0; s < mpi_size; ++s) {
        if (recv_counts[s] == 0) continue;
        auto const& r = recv_rngs[s];
        long off = recv_displs[s];
        for (long q = r[0].first; q < r[0].second; ++q)
          for (long P = r[1].first; P < r[1].second; ++P)
            for (long Q = r[2].first; Q < r[2].second; ++Q)
              Bloc(q - Bq0, P - BP0, Q - BQ0) = recvbuf[off++];
      }
      comm->barrier();
    }

    /**
     * Pull the iq-th smooth ζ slab out of `dzeta_quG` into `zeta_dist`
     * (distributed on the q-pool subcomm, grid (np_PQ, 1)).
     *
     * Assumes the `dzeta_quG` proc grid's q-axis aligns with the q-pool
     * partitioning of `_dZ` (true by construction: both come from
     * `find_proc_grid_max_npools`). Each `q_intra` member therefore
     * already owns the pool's q-slab inside its `dzeta_quG.local()`,
     * so the data move is purely intra-pool and uses the standard
     * `math::nda::redistribute` on `q_intra` (no global-comm collective).
     */
    template<class dz_dist_t, class Z_dist_t>
    void fill_zeta_for_iq_into_qpool(
        dz_dist_t const& dzeta_quG,
        int iq,
        Z_dist_t& zeta_dist,
        mpi3::communicator& q_intra)
    {
      using local2d_t = memory::array<HOST_MEMORY, ComplexType, 2>;

      long Np_smooth = zeta_dist.global_shape()[0];
      long ngm       = zeta_dist.global_shape()[1];

      auto qrng = dzeta_quG.local_range(0);
      long iq_loc = -1;
      for (auto [i, q] : itertools::enumerate(qrng))
        if ((long)q == (long)iq) { iq_loc = (long)i; break; }
      utils::check(iq_loc >= 0,
        "fill_zeta_for_iq_into_qpool: iq={} not in this rank's q-range "
        "[{},{}); dzeta_quG q-axis must be partitioned to match _dZ q-pools.",
        iq, qrng.first(), qrng.last());

      long np_u = dzeta_quG.grid()[1];
      long np_g = dzeta_quG.grid()[2];

      // Match dzeta_quG's block_size on the (mu, g) dims so the local
      // chunks line up exactly. Defaulting to bsize=1 produces a
      // *different* chunk_range partition than dzeta_quG used (which is
      // built with non-trivial block sizes by thc::evaluate).
      auto dz_bs = dzeta_quG.block_size();
      auto src_2d = math::nda::make_distributed_array<local2d_t>(
          q_intra, {np_u, np_g}, {Np_smooth, ngm},
          std::array<long,2>{dz_bs[1], dz_bs[2]});

      auto Aloc = dzeta_quG.local();
      auto Bloc = src_2d.local();
      utils::check(Aloc.shape(1) == Bloc.shape(0) && Aloc.shape(2) == Bloc.shape(1),
        "fill_zeta_for_iq_into_qpool: shape mismatch — dzeta local {}x{}x{} vs "
        "src_2d local {}x{}.",
        Aloc.shape(0), Aloc.shape(1), Aloc.shape(2),
        Bloc.shape(0), Bloc.shape(1));

      for (long i = 0; i < Bloc.shape(0); ++i)
        for (long j = 0; j < Bloc.shape(1); ++j)
          Bloc(i, j) = Aloc(iq_loc, i, j);

      math::nda::redistribute(src_2d, zeta_dist);
    }

    /**
     * Gather rows `wanted_rows` (rank-specific global range) out of a
     * 2D distributed array `A` (shape (Mtot, N), proc grid (np, 1) on
     * the q-pool comm) into the rank-local `out` buffer of shape
     * (wanted_rows.size(), N).
     *
     * Used by the per-q distributed augmentation to pull only the
     * (mu / λ) rows this rank needs to fill its own _dZ tile.
     */
    template<class A_dist_t, class Out_t>
    void gather_rows_from_dist_qpool(A_dist_t const& A,
                                     nda::range wanted_rows,
                                     mpi3::communicator& comm,
                                     Out_t&& out,
                                     nda::range g_rng)
    {
      using T = typename std::decay_t<Out_t>::value_type;
      long nproc = (long)comm.size();
      // Column slice [g_rng) of the row-distributed slab (callers
      // G-chunk so the gather target is rows × chunk, not rows × ngm).
      // ALL ranks of `comm` must pass the same g_rng (collective).
      long N = g_rng.size();
      utils::check(g_rng.first() >= 0 && g_rng.last() <= A.global_shape()[1],
        "gather_rows_from_dist_qpool: g_rng [{}, {}) outside [0, {})",
        g_rng.first(), g_rng.last(), A.global_shape()[1]);

      utils::check(out.shape(0) == wanted_rows.size() && out.shape(1) == N,
        "gather_rows_from_dist_qpool: out shape ({}, {}) != ({}, {})",
        out.shape(0), out.shape(1), wanted_rows.size(), N);
      if (out.size() > 0) out() = T(0);
      // NOTE: do NOT early-return when wanted_rows is empty — this rank
      // still has to participate in the comm.all_gather_n / all_to_all_v_n
      // / barrier collectives below. With wanted_rows.first()==last() its
      // mine(2)==mine(3), interv() returns an empty intersection for every
      // partner, recv_counts[*] stay 0, and the alltoallv is a valid no-op
      // for this rank. Skipping the collectives would deadlock the other
      // ranks of `comm` whose tiles do have rows to gather (e.g. ranks
      // with empty P_aug/Q_aug while others on the same q-pool are not).

      // metadata: (A.origin[0], A.lshape[0], wanted_rows.first, wanted_rows.last)
      nda::array<long,2> meta(nproc, 4);
      nda::array<long,1> mine(4);
      mine(0) = A.origin()[0];
      mine(1) = A.local_shape()[0];
      mine(2) = wanted_rows.first();
      mine(3) = wanted_rows.last();
      comm.all_gather_n(mine.data(), 4, meta.data(), 4);

      auto interv = [](long a0, long aN, long b0, long bE) -> std::pair<long,long> {
        long o = std::max(a0, b0);
        long e = std::min(a0 + aN, bE);
        return {o, std::max(o, e)};
      };

      std::vector<int> send_counts(nproc, 0), send_displs(nproc, 0);
      std::vector<int> recv_counts(nproc, 0), recv_displs(nproc, 0);
      std::vector<std::pair<long,long>> send_rng(nproc), recv_rng(nproc);

      long my_A_o = mine(0), my_A_n = mine(1);
      for (long d = 0; d < nproc; ++d) {
        long w0 = meta(d, 2), w1 = meta(d, 3);
        auto [o, e] = interv(my_A_o, my_A_n, w0, w1);
        if (e <= o) continue;
        send_rng[d] = {o, e};
        send_counts[d] = (int)((e - o) * N);
      }
      long my_w0 = mine(2), my_w1 = mine(3);
      for (long s = 0; s < nproc; ++s) {
        long sA_o = meta(s, 0), sA_n = meta(s, 1);
        auto [o, e] = interv(sA_o, sA_n, my_w0, my_w1);
        if (e <= o) continue;
        recv_rng[s] = {o, e};
        recv_counts[s] = (int)((e - o) * N);
      }
      for (long d = 1; d < nproc; ++d) send_displs[d] = send_displs[d-1] + send_counts[d-1];
      for (long s = 1; s < nproc; ++s) recv_displs[s] = recv_displs[s-1] + recv_counts[s-1];
      long total_send = (long)send_displs.back() + (long)send_counts.back();
      long total_recv = (long)recv_displs.back() + (long)recv_counts.back();
      std::vector<T> sbuf(total_send), rbuf(total_recv);

      auto Aloc = A.local();
      long g0 = g_rng.first();
      for (long d = 0; d < nproc; ++d) {
        if (send_counts[d] == 0) continue;
        long off = send_displs[d];
        for (long r = send_rng[d].first; r < send_rng[d].second; ++r)
          for (long g = 0; g < N; ++g)
            sbuf[off++] = Aloc(r - my_A_o, g0 + g);
      }

      comm.all_to_all_v_n(
          sbuf.data(), send_counts.data(), send_displs.data(),
          rbuf.data(), recv_counts.data(), recv_displs.data());

      for (long s = 0; s < nproc; ++s) {
        if (recv_counts[s] == 0) continue;
        long off = recv_displs[s];
        for (long r = recv_rng[s].first; r < recv_rng[s].second; ++r) {
          long out_row = r - my_w0;
          for (long g = 0; g < N; ++g)
            out(out_row, g) = rbuf[off++];
        }
      }
      comm.barrier();
    }


    void build_from_CD() {
      using math::nda::make_distributed_array;
      utils::check(x_range==y_range, "thc_reader::build_from_CD: x_range!=y_range needs testing. Disabling for now.");
      _Timer.start("BUILD_TOTAL");

      _Timer.start("BUILD_THC");
      auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<HOST_MEMORY>(0, _Np, x_range, y_range);
      _rp = std::move(ri);
      _Np = _rp.size();
      _Timer.stop("BUILD_THC");
     
      // allocate structures with dynamic _Np
      _Chi_head = nda::array<ComplexType, 2>(_nqpts_ibz, _Np);
      _Chi_bar_head = nda::array<ComplexType, 2>(_nqpts_ibz, _Np);

      _Timer.start("BUILD_THC");
      /*** Read Cholesky ERIs and fit them to THC solver ***/
      auto chol_reader = chol_reader_t(_MF, _cd_dir, "chol_info.h5", single_kpair);
      long nchol_max = chol_reader.Np();
      // MAM: need chol_reader_t with different x/y ranges!

      long np = _mpi->comm.size();
      long nkpools = utils::find_proc_grid_max_npools(np, _nkpts, 0.2);
      np /= nkpools;
      long np_i = utils::find_proc_grid_min_diff(np,1,1);
      long np_j = np / np_i;

      for (size_t q = 0; q < _nqpts_ibz; ++q) {
        // read Cholesky ERIs; nchol might be different at different q-points
        long nchol_q = chol_reader.read_Np(q);
        auto chol_rng = nda::range(0, nchol_q);

        // MAM: might be a problem if I don't propagate change to cholesky code
        auto dbuffer = make_distributed_array<Array_t<HOST_MEMORY,5>>(
            _mpi->comm, {1, 1, nkpools, np_i, np_j}, {nchol_q, _ns_in_basis*_npol_in_basis, _nkpts, x_range.size(), y_range.size()});
        auto s_rng = dbuffer.local_range(1);
        auto k_rng = dbuffer.local_range(2);
        auto i_rng = dbuffer.local_range(3);
        auto j_rng = dbuffer.local_range(4);

        auto buffer_loc = dbuffer.local();
        if(_nkpts != 1) {
          for (auto [is,s] : itertools::enumerate(s_rng)) {
            for (auto [ik, k]: itertools::enumerate(k_rng)) {
              // Lqk_Pij = (nchol_max, nbnd, nbnd)
              auto Lqk_Pij = chol_reader.V(q, s, k);
              buffer_loc(nda::range::all, is, ik, nda::ellipsis{}) = Lqk_Pij(chol_rng, i_rng, j_rng);
            }
          }
        }
        else { // molecular case
          auto sLqk_Pij = math::shm::make_shared_array<nda::array<ComplexType, 3> >(_mpi->comm, 
                             _mpi->internode_comm, _mpi->node_comm,
                             std::array<long int, 3>{nchol_max, x_range.size(), y_range.size()});
          for (auto [is,s] : itertools::enumerate(s_rng)) {
            sLqk_Pij.win().fence();
            if (_mpi->node_comm.root()) {
              sLqk_Pij.local() = chol_reader.V(0, s, 0);
            }
            sLqk_Pij.win().fence();
            buffer_loc(nda::range::all, is, 0, nda::ellipsis{}) = sLqk_Pij.local()(chol_rng, i_rng, j_rng);
          }
        }
        // solve the least-square problem
        // TODO CNY: we need to evaluate _Chi_head and _Chi_bar_head for finite-size corrections
        // TODO CNY: proper warning on the missing _Chi_head and _Chi_bar_head
        auto dZq_uv = _thc_builder_opt.value().evaluate<HOST_MEMORY>(q, _rp, dbuffer);
        auto Zq_loc = dZq_uv.local();
        if (q == 0) {
          auto pgrid = dZq_uv.grid();
          auto block_size = dZq_uv.block_size();
          auto gshape = dZq_uv.global_shape();
          // choose distribution of _dZ based on dZq_uv at q = 0
          _dZ = make_distributed_array<Array_t<HOST_MEMORY,3>>(
              _mpi->comm, {1, pgrid[0], pgrid[1]}, {_nqpts_ibz, gshape[0], gshape[1]},
              {1, block_size[0], block_size[1]});
        }
        auto Z_loc = _dZ.local();
        Z_loc(q, nda::ellipsis{}) = Zq_loc;
      }
      _Chi_head() = 0.0;
      _Chi_bar_head() = 0.0;
      _Timer.stop("BUILD_THC");

      // gather dPa to _X_shm
      _Timer.start("BUILD_GATHER");
      gather_X_shm(dXa);
      if(dXb.has_value()) 
        gather_Y_shm(dXb.value());
      else
        utils::check(x_range == y_range, "thc_reader::build: x_range != y_range with missing dXb value.");
      _Timer.stop("BUILD_GATHER");

      // save if requested
      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          if (_format == "bdft") {
            std::vector<int> arng = {x_range.first(),x_range.last()};
            std::vector<int> brng = {y_range.first(),y_range.last()};
            h5::h5_write(grp, "Np", (int)_Np);
            h5::h5_write(grp, "number_of_bands", (int)_nbnd);
            h5::h5_write(grp, "X_orbital_range", arng);
            h5::h5_write(grp, "Y_orbital_range", brng);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            auto X_0 = _X_shm.local();
            nda::h5_write(grp, "collocation_matrix", X_0, false);
            if(_Y_shm.has_value()) {
              auto Y_0 = _Y_shm.value().local();
              nda::h5_write(grp, "Y_collocation_matrix", Y_0, false);
            }
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");
      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "\n  THC-READER::BUILD_FROM_CD()");
      app_log(2, "  ---------------------------");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute thc-eri:             {0:.3f} sec", _Timer.elapsed("BUILD_THC"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write eri:                   {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, "      - paw augmentation total:      {0:.3f} sec", _Timer.elapsed("PAW_AUG"));
      if (_paw_aug) {
        app_log(2, "        .  X aug (Y rows x ks):      {0:.3f} sec", _Timer.elapsed("PAW_AUG.X_aug"));
        app_log(2, "        .  gather smooth GG block:   {0:.3f} sec", _Timer.elapsed("PAW_AUG.gather_smooth"));
        app_log(2, "        .  aainit (ap, lpx, lpl):    {0:.3f} sec", _Timer.elapsed("PAW_AUG.aainit"));
        app_log(2, "        .  qrad table (per species): {0:.3f} sec", _Timer.elapsed("PAW_AUG.qrad_tab"));
        app_log(2, "        .  q-loop dz gather:         {0:.3f} sec", _Timer.elapsed("PAW_AUG.dzeta"));
        app_log(2, "        .  q-loop eta at q+G:        {0:.3f} sec", _Timer.elapsed("PAW_AUG.eta_at_q"));
        app_log(2, "        .  q-loop wG at q+G:         {0:.3f} sec", _Timer.elapsed("PAW_AUG.wG_at_q"));
        app_log(2, "        .  q-loop eta flatten/conj:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.eta_flat"));
        app_log(2, "        .  q-loop V_GL contraction:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.V_GL"));
        app_log(2, "        .  q-loop V_LL contraction:  {0:.3f} sec", _Timer.elapsed("PAW_AUG.V_LL"));
        app_log(2, "        .  q-loop K_a inject:        {0:.3f} sec", _Timer.elapsed("PAW_AUG.K_a"));
        app_log(2, "        .  q-loop stitch V_full:     {0:.3f} sec", _Timer.elapsed("PAW_AUG.stitch"));
        app_log(2, "        .  scatter V_full -> _dZ:    {0:.3f} sec", _Timer.elapsed("PAW_AUG.scatter"));
      }
      app_log(2, " ");

      print_thc_summary();
    }

    // MAM: No PAW here yet!!!
    void build_isdf_only(bool check_accuracy=true, bool write_zeta_on_fft_mesh=false) {
      _Timer.start("BUILD_TOTAL");

      _Timer.start("BUILD_ISDF");
      auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<HOST_MEMORY>(0, _Np);
      _rp = std::move(ri);
      _Np = _rp.size();
      _Timer.stop("BUILD_ISDF");

      app_log(1, "*******************************");
      app_log(1, " ISDF-only builder: ");
      app_log(1, "*******************************");
      app_log(1, "    - Np       = {}", _Np);
      app_log(1, "    - h5 chkpt file = {}", _eri_file);

      _Timer.start("BUILD_ISDF");
      auto dzeta_qur = _thc_builder_opt.value().evaluate_isdf_only<HOST_MEMORY>(_rp,dXa,dXb);
      _Timer.stop("BUILD_ISDF");

      _Timer.start("BUILD_GATHER");
      gather_X_shm(dXa);
      _Timer.stop("BUILD_GATHER");

      _Timer.start("ISDF_CHECK");
      if (check_accuracy) isdf_check(dzeta_qur);
      _Timer.stop("ISDF_CHECK");

      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          if (_format == "bdft") {
            h5::h5_write(grp, "Np", (int)_Np);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            nda::h5_write(grp, "collocation_matrix", _X_shm.local(), false);
            _thc_builder_opt.value().save(grp, _format, _rp, dzeta_qur, write_zeta_on_fft_mesh);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, dzeta_qur, write_zeta_on_fft_mesh);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");
      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "***************************************************");
      app_log(2, "                  THC-READER::BUILD_ISDF_ONLY() ");
      app_log(2, "***************************************************");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute ISDF:                {0:.3f} sec", _Timer.elapsed("BUILD_ISDF"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      app_log(2, "      - ISDF check:                  {0:.3f} sec", _Timer.elapsed("ISDF_CHECK"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write ISDF:                  {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, "***************************************************\n");
    }

    void read() {
      _Timer.start("BUILD_TOTAL");
      // Cache precomputed THC ERIs
      h5::file file(_eri_file, 'r');
      h5::group grp(file);

      {
        std::vector<int> arng(2);  
        h5::h5_read(grp, "X_orbital_range", arng);
        utils::check(arng.size()==2 and arng[0]>=0 and arng[1]<=_nbnd,
                     "thc_reader::read(): Invalid X_orbital_range."); 
        x_range = nda::range(arng[0],arng[1]);
        h5::h5_read(grp, "Y_orbital_range", arng);
        utils::check(arng.size()==2 and arng[0]>=0 and arng[1]<=_nbnd,
                     "thc_reader::read(): Invalid Y_orbital_range."); 
        y_range = nda::range(arng[0],arng[1]);
      }
      nda::h5_read(grp, "interpolating_points", _rp);
      nda::h5_read(grp, "interpolating_vectors_G0", _Chi_head);
      nda::h5_read(grp, "dual_interpolating_vectors_G0", _Chi_bar_head);
      // PAW/USPP-augmented files carry Np = N_smooth + N_aug while the
      // interpolating points are a smooth-grid object: allow rp < Np and
      // recover the split (an equality check would abort on any
      // augmented file).
      utils::check(_rp.shape(0) <= _Np,
                   "thc_reader_t::build: rp.shape() > Np. Inconsistent dimensions from the precomputed THC-ERI.");
      _Np_smooth = (int)_rp.shape(0);
      _N_aug = (int)(_Np - _Np_smooth);
      utils::check(_Chi_head.extent(1) == _Np and _Chi_bar_head.extent(1) == _Np,
                   "thc_reader_t::build: interpolating_vectors_G0 has {} columns "
                   "but Np={}. Files written before the D1 augmented-head fix "
                   "store smooth-only head vectors — regenerate the THC ERI file.",
                   _Chi_head.extent(1), _Np);

      if(_X_shm.shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, x_range.size()}) {
        _X_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, x_range.size()});
        _mpi->node_comm.barrier(); 
      }
      _X_shm.win().fence();
      if (_mpi->node_comm.root()) {
        auto Xloc = _X_shm.local();
        nda::h5_read(grp, "collocation_matrix", Xloc);
      }
      _X_shm.win().fence();

      if(x_range != y_range) {
        if(not _Y_shm.has_value() or _Y_shm.value().shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, y_range.size()}) {
          _Y_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
              *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, y_range.size()});
          _mpi->node_comm.barrier();
        }
        _Y_shm.value().win().fence();
        if (_Y_shm.value().node_comm()->root()) {
          auto Yloc = _Y_shm.value().local();
          nda::h5_read(grp, "Y_collocation_matrix", Yloc);
        }
        _Y_shm.value().win().fence();
      }

      if (_storage == eri_storage_e::incore) {
        int np = _mpi->comm.size();
        long nqpools = utils::find_proc_grid_max_npools(np, _nqpts_ibz, 0.2);
        utils::check(nqpools > 0 and nqpools <= _nqpts_ibz, "thc_reader_t::build: nqpools <= 0 or nqpools > nqpts");
        utils::check(np % nqpools == 0, "thc_reader_t::build: comm.size() % nqpools != 0");
        int np_PQ = np / nqpools;
        int np_P = utils::find_proc_grid_min_diff(np_PQ, 1, 1);
        int np_Q = np_PQ / np_P;
        _dZ = math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {nqpools, np_P, np_Q}, {_nqpts_ibz, _Np, _Np});
        math::nda::h5_read(grp, "coulomb_matrix", _dZ);

        _mpi->comm.barrier();
      }
      _Timer.stop("BUILD_TOTAL");

      print_thc_summary();
    }

    int read_Np() {
      int Np;
      h5::file file(_eri_file, 'r');
      h5::group grp(file);
      h5::h5_read(grp, "Np", Np);
      return Np;
    }

  public:
    // The q-independent collocation matrix
    /**
     * Collocation matrix for a given spin 'is', polarization 'ip' and k-point 'ik'
     */
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto X(int is, int ip, int ik) const {
      _Timer.start("READ_X");
      utils::check(is >= 0 and is < _ns, "Error in thc::reader_t::X(is,ip,ik): is out of bounds: is:{}",is);
      utils::check(ip >= 0 and ip < _npol, "Error in thc::reader_t::X(is,ip,ik): is out of bounds: ip:{}",ip);
      int id = is*_npol_in_basis+ip;
      if(_ns_in_basis == 1) id = std::min(ip,_npol_in_basis-1);
      else if(_npol_in_basis == 1) id = is;
      auto Xsk = _X_shm.local()(id, ik, nda::ellipsis{});
      _Timer.stop("READ_X");
      if constexpr (MEM == HOST_MEMORY) {
        return std::as_const(Xsk);  // to make sure it is not modified
      } else {
        return memory::to_memory_space<MEM>(Xsk); 
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto X() const {
      if constexpr (MEM == HOST_MEMORY) {
        auto X_ = _X_shm.local();
        return std::as_const(X_);
      } else {
        return memory::to_memory_space<MEM>(_X_shm.local());
      }
    }

    // The q-independent collocation matrix
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto Y(int is, int ip, int ik) const {
      if(x_range == y_range) {
        return X<MEM>(is,ip,ik);
      } else {
        utils::check(_Y_shm.has_value(), "thc_reader::Y(is,ik): _Y_shm has no value.");
        utils::check(is >= 0 and is < _ns, "Error in thc::reader_t::Y(is,ip,ik): is out of bounds: is:{}",is);
        utils::check(ip >= 0 and ip < _npol, "Error in thc::reader_t::Y(is,ip,ik): is out of bounds: ip:{}",ip);
        _Timer.start("READ_X");
        int id = is*_npol_in_basis+ip;
        if(_ns_in_basis == 1) id = std::min(ip,_npol_in_basis-1);
        else if(_npol_in_basis == 1) id = is;
        auto Ysk = _Y_shm.value().local()(id, ik, nda::ellipsis{});
        _Timer.stop("READ_X");
        if constexpr (MEM == HOST_MEMORY) {
          return std::as_const(Ysk);  // to make sure it is not modified
        } else {
          return memory::to_memory_space<MEM>(Ysk);
        }
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto Y() const {
      if constexpr (MEM == HOST_MEMORY) {
        if(x_range == y_range) {
          auto X_ = _X_shm.local();
          return std::as_const(X_);
        } else {
          utils::check(_Y_shm.has_value(), "thc_reader::Y(): _Y_shm has no value.");
          auto Y_ = _Y_shm.value().local();
          return std::as_const(Y_);
        }
      } else {
        if(x_range == y_range) {
          return memory::to_memory_space<MEM>(_X_shm.local());
        } else {
          utils::check(_Y_shm.has_value(), "thc_reader::Y(): _Y_shm has no value.");
          return memory::to_memory_space<MEM>(_Y_shm.value().local());
        }
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z_same_q(int iq) const {
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Zq(_Np, _Np);
      if (_storage == eri_storage_e::incore) {
        math::nda::gather_sub_matrix(iq, 0, _dZ, &Zq);
        _dZ.communicator()->all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
      } else {
        if (_dZ.communicator()->rank()==0) {
          h5::file file(_eri_file, 'r');
          h5::group grp(file);
          nda::h5_read(grp, "coulomb_matrix", Zq,
                       std::tuple{iq, nda::range::all, nda::range::all});
        }
        _dZ.communicator()->all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Zq;
      } else {
        return memory::to_memory_space<MEM>(Zq);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z(int iq, bool same_q=false) const {
      if (same_q)
        return Z_same_q<MEM>(iq);
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Zq(_Np, _Np);
      if (_storage == eri_storage_e::incore) {
        int iq_at_ip = -1;
        for (int ip = 0; ip < _dZ.communicator()->size(); ++ip) {
          if (ip == _dZ.communicator()->rank()) iq_at_ip = iq;
          _dZ.communicator()->broadcast_n(&iq_at_ip, 1, ip);
          utils::check(iq_at_ip >= 0, "Error: iq_at_ip < 0");
          math::nda::gather_sub_matrix(iq_at_ip, ip, _dZ, &Zq);
        }
      } else {
        h5::file file(_eri_file, 'r');
        h5::group grp(file);
        nda::h5_read(grp, "coulomb_matrix", Zq,
                     std::tuple{iq, nda::range::all, nda::range::all});
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Zq;
      } else {
        return memory::to_memory_space<MEM>(Zq);
      }
    }

    // version that requires less communication
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z(long iq, nda::range P_rng, nda::range Q_rng,
                                 long qpool_id, long nqpool,
                                 mpi3::communicator &q_intra_comm) const {
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Z_PQ(P_rng.size(), Q_rng.size());
      Z_PQ() = 0.0;
      if (_storage == eri_storage_e::incore) {
        for (long iqpool = 0; iqpool < nqpool; ++iqpool) {
          long iqq = (qpool_id==iqpool and q_intra_comm.root())? iq : 0;
          long ip  = (qpool_id==iqpool and q_intra_comm.root())? _mpi->comm.rank() : 0;
          _mpi->comm.all_reduce_in_place_n(&iqq, 1, std::plus<>{});
          _mpi->comm.all_reduce_in_place_n(&ip, 1, std::plus<>{});

          nda::array<ComplexType, 2> Zq;
          if (qpool_id == iqpool) {
            // CNY: improve this! not all processors need the entire matrix
            Zq = nda::array<ComplexType, 2>(_Np, _Np);
            Zq() = 0.0;
            math::nda::gather_sub_matrix(iqq, ip, _dZ, &Zq);
            q_intra_comm.all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
            Z_PQ = Zq(P_rng, Q_rng);
          } else {
            math::nda::gather_sub_matrix(iqq, ip, _dZ, &Zq);
          }
        }
      } else {
        h5::file file(_eri_file, 'r');
        h5::group grp(file);
        nda::h5_read(grp, "coulomb_matrix", Z_PQ, std::tuple{iq, P_rng, Q_rng});
      }

      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Z_PQ;
      } else {
        return memory::to_memory_space<MEM>(Z_PQ);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto dZ(std::array<long, 3> pgrid, std::array<long, 3> bsize = {0, 0, 0}) const {
      _Timer.start("READ_V");
      auto dZ_qPQ = math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, pgrid, {_nqpts_ibz, _Np, _Np}, bsize);
      if (_storage == eri_storage_e::incore) {
        math::nda::redistribute(_dZ, dZ_qPQ);
      } else {
        auto q_rng = dZ_qPQ.local_range(0);
        auto P_rng = dZ_qPQ.local_range(1);
        auto Q_rng = dZ_qPQ.local_range(2);

        auto Z_loc = dZ_qPQ.local();
        for( auto [iq,q] : itertools::enumerate(q_rng) ) {
          auto Zq = Z(q);
          Z_loc(iq, nda::ellipsis{}) = Zq(P_rng, Q_rng);
        }
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return dZ_qPQ;
      } else {
        auto dZ_qPQ_d = math::nda::make_distributed_array<Array_t<MEM,3>>(_mpi->comm, pgrid, {_nqpts_ibz, _Np, _Np}, bsize);  
        dZ_qPQ_d.local() = dZ_qPQ.local();
        return dZ_qPQ_d; 
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto basis_head() const {
      if constexpr (MEM == HOST_MEMORY) {
        auto _C = _Chi_head();
        return std::as_const(_C);
      } else {
        return memory::to_memory_space<MEM>(_Chi_head);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto basis_bar_head() const
    {
      if constexpr (MEM == HOST_MEMORY) {
        auto _C = _Chi_bar_head();
        return std::as_const(_C);
      } else {
        return memory::to_memory_space<MEM>(_Chi_bar_head);
      }
    }

    bool initialized() const { return _initialized; }
    bool thc_builder_is_null() const { return _thc_builder_opt == std::nullopt; }
    int Np() const { return _Np; }
    int nkpts() const { return _nkpts; }
    int nkpts_ibz() const { return _nkpts_ibz; }
    int nqpts() const { return _nqpts; }
    int nqpts_ibz() const { return _nqpts_ibz; }
    int ns() const { return _ns; }
    int ns_in_basis() const { return _ns_in_basis; }
    int npol() const { return _npol; }
    int npol_in_basis() const { return _npol_in_basis; }
    int nbnd() const { return _nbnd; }
    int nbnd_aux() const { return 0; }
    std::string& set_X_type() { return _X_type; }
    const std::string thc_X_type() const { return _X_type; }
    std::string filename() const { return _eri_file; }
    //mpi3::communicator* comm() const { return std::addressof(_mpi->comm); }
    auto& MF() const { return _MF; }
    auto& mpi() const { return _mpi; }
    auto X_orbital_range() const { return x_range; }
    auto Y_orbital_range() const { return y_range; }

    void print_timers() const {
      app_log(2, "\n  THC-READER timers");
      app_log(2, "  -----------------");
      app_log(2, "    BUILD:                {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "    READ_X:               {0:.3f} sec", _Timer.elapsed("READ_X"));
      app_log(2, "    READ_V:               {0:.3f} sec\n", _Timer.elapsed("READ_V"));
    }

  private:
    void gather_X_shm(math::nda::DistributedArrayOfRank<4> auto &dXa) {
      using nda::range;
      int norb = x_range.size();
      if(_X_shm.shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb} ) {
        _X_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb});
      }
      auto sX_buffer = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
          *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, norb, _Np});
      math::nda::gather_to_shm(dXa, sX_buffer);

      if (_X_shm.node_comm()->root()) {
        for (int is = 0; is < _ns_in_basis*_npol_in_basis; ++is) {
          for (int ik = 0; ik < _nkpts; ++ik) {
            auto Xsk_trans = sX_buffer.local()(is, ik, range::all, range::all); // (norb, Np)
            auto Xsk = _X_shm.local()(is, ik, range::all, range::all); // (Np, norb)
            Xsk = nda::transpose(Xsk_trans);
          }
        }
      }
      _X_shm.communicator()->barrier();
    }

    void gather_Y_shm(math::nda::DistributedArrayOfRank<4> auto &dXb) {
      using nda::range;
      int norb = y_range.size();
      if( not _Y_shm.has_value() or
          _Y_shm.value().shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb} ) {
        _Y_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb});
      }
      auto sY_buffer = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, norb, _Np});
      math::nda::gather_to_shm(dXb, sY_buffer);

      if (_Y_shm.value().node_comm()->root()) {
        for (int is = 0; is < _ns_in_basis*_npol_in_basis; ++is) {
          for (int ik = 0; ik < _nkpts; ++ik) {
            auto Ysk_trans = sY_buffer.local()(is, ik, range::all, range::all); // (norb, Np)
            auto Ysk = _Y_shm.value().local()(is, ik, range::all, range::all); // (Np, norb)
            Ysk = nda::transpose(Ysk_trans);
          }
        }
      }
      _Y_shm.value().communicator()->barrier();
    }

    void isdf_check(dArray_t<HOST_MEMORY, 3> const& dzeta_qur) {

      utils::check(_nkpts==1, "isdf_check: currently only supports gamma point. ");

      auto zeta = math::nda::make_distributed_array<Array_t<HOST_MEMORY, 3>>(
          _mpi->comm, {1, 1, _mpi->comm.size()}, dzeta_qur.global_shape());
      math::nda::redistribute(dzeta_qur, zeta);

      auto r_rng = zeta.local_range(2);

      auto sphi_ir = math::shm::make_shared_array<Array_view_t<HOST_MEMORY, 2>>(
          *_mpi, {_nbnd, _MF->nnr()});
      if (_mpi->node_comm.root())
        _MF->get_orbital_set('r', 0, 0, nda::range(_nbnd), sphi_ir.local());
      _mpi->comm.barrier();

      auto zeta_loc = zeta.local();
      auto phi_ir = sphi_ir.local();
      auto phi_ui = _X_shm.local();
      nda::array<ComplexType, 1> rho_fit_r(r_rng.size());
      nda::array<RealType, 1> error_r(r_rng.size());
      double max_diff = -1;
      for (size_t i=0; i<_nbnd; ++i) {
        for (size_t j=0; j<_nbnd; ++j) {
          rho_fit_r() = 0.0;
          for (size_t u=0; u<_Np; ++u)
            rho_fit_r += std::conj(phi_ui(0,0,u,i)) * phi_ui(0,0,u,j) * zeta_loc(0,u,nda::range::all);

          for( auto [ir,r] : itertools::enumerate(r_rng) )
            error_r(ir) = std::abs( std::conj(phi_ir(i, r)) * phi_ir(j, r) - rho_fit_r(ir) );
          max_diff = std::max(max_diff, nda::max_element(error_r));
        }
      }
      double max_diff_global = _mpi->comm.max(max_diff);
      app_log(1, "\nMaximum error of ISDF: {}\n", max_diff_global);
      _mpi->comm.barrier();
    }

  private:
    std::shared_ptr<mf::MF> _MF;
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi;
    // where to perform THC evaluation
    MEMORY_SPACE _MEM_EVAL = DEFAULT_MEMORY_SPACE;
    // whether the thc integrals has been initialized
    bool _initialized = false;
    // eri storage type: incore or outcore
    eri_storage_e _storage;
    // file to store eris
    std::string _eri_file;
    // eri format to store
    std::string _format;
    std::string _cd_dir;     // directory for CD eris;
    std::string _X_type;

    std::optional<thc> _thc_builder_opt;

    int _Np;
    int _nkpts;
    int _nkpts_ibz;
    int _nqpts;
    int _nqpts_ibz;
    int _ns;
    int _ns_in_basis;
    int _nbnd;
    int _npol;
    int _npol_in_basis;
    nda::range x_range;
    nda::range y_range;

    // add option to keep data on device or on host memory with INCORE
    // keep everything as optionals to keep things simple

    dArray_t<HOST_MEMORY,3> _dZ;
    // Used if storing in HOST_MEMORY
    sArray_t<memory::array_view<HOST_MEMORY, ComplexType, 4>> _X_shm;
    std::optional<sArray_t<memory::array_view<HOST_MEMORY, ComplexType, 4>>> _Y_shm;

    memory::array<HOST_MEMORY, ComplexType, 2> _Chi_head;
    memory::array<HOST_MEMORY, ComplexType, 2> _Chi_bar_head;
    memory::array<HOST_MEMORY, long, 1> _rp;

    mutable utils::TimerManager _Timer;

    // ====== PAW augmentation state ======================================
    // _paw_aug == true means the X / V arrays in this reader are
    // pre-augmented: rows [N_smooth, N_smooth + N_aug) are atom-local
    // ISDF features rather than smooth ζ. Downstream code consumes the
    // composite (X, V) without distinguishing.
    //
    // V_GL / V_LL are filled at every q via radial-Bessel η^q(G); the K_a
    // same-atom one-center kernel is added at every q.
    bool _paw_aug = true;
    bool _paw_onsite = true;           // DIAGNOSTIC-ONLY: gate on the K_a
                                       // one-center kernel for PAW species; whether K_a
                                       // belongs is derived from vv_compensation
                                       // (moment => include, shape => drop)
    bool _paw_vgl = true;              // diagnostic: include V_GL/V_LG smooth-aug cross
    bool _paw_vll = true;              // diagnostic: include V_LL aug-aug block
    double _paw_aug_ecut = 0.0;        // Ha; |q+G|^2/2 cutoff on eta (0 = off)
    int _Np_smooth = 0;                // smooth-only block size
    int _N_aug = 0;                    // total atom-local rows
    std::shared_ptr<hamilt::pseudopot> _psp;        // lazy via make_pseudopot
    std::vector<hamilt::paw::species_local_isdf> _isdf;
    hamilt::paw::paw_aug_layout _aug_layout;
    hamilt::paw::isdf_metric _paw_isdf_metric = hamilt::paw::isdf_metric::Coulomb;
    double _paw_isdf_tol = 1e-12;
    std::string _paw_isdf_cache_h5;
    // PAW exact-exchange options parsed from [interaction.thc] (cv_exchange,
    // vv_compensation, aug_lmax); see hamilt::paw_exx_options. Propagated to the
    // shared pseudopot so the direct v_x path uses the same settings.
    hamilt::paw_exx_options _exx_opts;
  };

} // methods

#endif //COQUI_THC_READER_T_HPP
