#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/sparse/csr_blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "methods/embedding/projector_t.h"
#include "utilities/kpoint_utils.hpp"
#include "utilities/interpolation_utils.hpp"
#include "mean_field/symmetry/unfold_bz.h"
#include "pproc_t.h"

namespace methods {
  template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
  void pproc_t::read_scf_dataset(std::string dataset,
                                 memory::darray_t<local_Array_4D_t, communicator_t> &A_tski) {
    decltype(nda::range::all) all;
    int iter;
    auto [nt_loc, ns_loc, nk_loc, ni_loc] = A_tski.local_shape();
    auto s_rng = A_tski.local_range(1);
    auto k_rng = A_tski.local_range(2);
    auto i_rng = A_tski.local_range(3);
    nda::array<ComplexType, 5> buffer_tskij(nt_loc, ns_loc, nk_loc, ni_loc, ni_loc);

    _Timer.start("READ");
    std::string filename = _scf_output + ".mbpt.h5";
    h5::file file(filename, 'r');
    h5::group grp(file);

    auto scf_grp = grp.open_group("scf");
    h5::h5_read(scf_grp, "final_iter", iter);

    auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
    nda::h5_read(iter_grp, dataset, buffer_tskij,
                 std::tuple{all, s_rng, k_rng, i_rng, i_rng});
    _Timer.stop("READ");

    auto A_loc = A_tski.local();
    for (size_t i = 0; i < ni_loc; ++i) {
      // explicit hermitization
      A_loc(all, all, all, i) = 0.5 * (buffer_tskij(all, all, all, i, i) + nda::conj(buffer_tskij(all, all, all, i, i)) );
    }
  }

  template<nda::MemoryArray local_Array_t, typename communicator_t>
  void pproc_t::read_scf_dataset_full(std::string dataset, std::string group,
                                 memory::darray_t<local_Array_t, communicator_t> &A) {
    decltype(nda::range::all) all;
    int iter;
    auto A_loc = A.local();
    _Timer.start("READ");
    std::string filename = _scf_output + ".mbpt.h5";
    h5::file file(filename, 'r');
    h5::group grp(file);

    constexpr int rank = nda::get_rank<local_Array_t>;
    constexpr bool is_rank_4 = (rank == 4);
    constexpr bool is_rank_5 = (rank == 5);

    if constexpr (is_rank_5) {
        if(group == "scf") { // G or Sigma
            auto scf_grp = grp.open_group("scf");
            h5::h5_read(scf_grp, "final_iter", iter);
        
            auto s_rng = A.local_range(1);
            auto k_rng = A.local_range(2);
            auto i_rng = A.local_range(3);
            auto j_rng = A.local_range(4);
           
            auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
            nda::h5_read(iter_grp, dataset, A_loc,
                         std::tuple{all, s_rng, k_rng, i_rng, j_rng});
        }
    }
    if constexpr (is_rank_4) {
        if(group == "system") { // S or H0
            auto sys_grp = grp.open_group("system");
            
            auto s_rng = A.local_range(0);
            auto k_rng = A.local_range(1);
            auto i_rng = A.local_range(2);
            auto j_rng = A.local_range(3);
           
            nda::h5_read(sys_grp, dataset, A_loc,
                         std::tuple{s_rng, k_rng, i_rng, j_rng});
        }
    }
    _Timer.stop("READ");
  }

  template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
  void pproc_t::dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                               memory::darray_t<local_Array_4D_t, communicator_t> &dA_out,
                               nda::array<ComplexType, 1> &iw_mesh,
                               memory::darray_t<local_Array_4D_t, communicator_t> &dA_in,
                               std::string dataset, std::string grp_name, int iter) {
    _Timer.start("WRITE");
    if (dA_out.communicator()->root()) {
      std::string filename = _scf_output + ".mbpt.h5";
      h5::file file(filename, 'a');
      auto grp = h5::group(file).open_group(grp_name);
      if (iter == -1)
        h5::read(grp, "final_iter", iter);
      auto iter_grp = grp.open_group("iter"+std::to_string(iter));
      auto ac_grp = (iter_grp.has_subgroup("ac"))?
          iter_grp.open_group("ac") : iter_grp.create_group("ac");
      auto data_grp = (ac_grp.has_subgroup(dataset))?
          ac_grp.open_group(dataset) : ac_grp.create_group(dataset);

      nda::h5_write(data_grp, "w_mesh", w_mesh, false);
      nda::h5_write(data_grp, "iw_mesh", iw_mesh, false);
      math::nda::h5_write(data_grp, "output", dA_out);
      math::nda::h5_write(data_grp, "input", dA_in);
    } else {
      h5::group data_grp;
      math::nda::h5_write(data_grp, "output", dA_out);
      math::nda::h5_write(data_grp, "input", dA_in);
    }
    _Timer.stop("WRITE");
  }

  template<nda::MemoryArray local_Array_t>
  void pproc_t::dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                               local_Array_t &A_out,
                               nda::array<ComplexType, 1> &iw_mesh,
                               local_Array_t& A_in,
                               std::string dataset, std::string grp_name, int iter) {
    _Timer.start("WRITE");
    if (_context.comm.root()) {
      std::string filename = _scf_output + ".mbpt.h5";

      h5::file file(filename, 'a');
      auto grp = h5::group(file).open_group(grp_name);
      if (iter == -1)
        h5::read(grp, "final_iter", iter);
      auto iter_grp = grp.open_group("iter"+std::to_string(iter));
      auto ac_grp = (iter_grp.has_subgroup("ac"))?
                    iter_grp.open_group("ac") : iter_grp.create_group("ac");
      auto data_grp = (ac_grp.has_subgroup(dataset))?
                      ac_grp.open_group(dataset) : ac_grp.create_group(dataset);

      if(!data_grp.has_dataset("w_mesh")) nda::h5_write(data_grp, "w_mesh", w_mesh, false);
      if(!data_grp.has_dataset("iw_mesh")) nda::h5_write(data_grp, "iw_mesh", iw_mesh, false);
      if(!data_grp.has_dataset("output")) nda::h5_write(data_grp, "output", A_out, false);
      if(!data_grp.has_dataset("input"))  nda::h5_write(data_grp, "input", A_in, false);
    }
    _context.comm.barrier();
    _Timer.stop("WRITE");
  }

  // TODO CNY: refactoring required
  void pproc_t::analyt_cont(mf::MF &mf, analyt_cont::ac_context_t &ac_params, std::string dataset) {
    using math::nda::make_distributed_array;
    using local_Array_3D_t = nda::array<ComplexType, 3>;
    using local_Array_4D_t = nda::array<ComplexType, 4>;
    using local_Array_5D_t = nda::array<ComplexType, 5>;
    utils::check(ac_params.stats == imag_axes_ft::fermi,
                 "pproc_t::analyt_cont: continuation for bosonic function is not supported yet!");

    long ns = mf.nspin();
    long nkpts_ibz = mf.nkpts_ibz();
    long nbnds = mf.nbnd();

    int np = _context.comm.size();
    int nkpools = utils::find_proc_grid_max_npools(np, nkpts_ibz, 0.2);
    std::array<long, 4> pgrid = {1, 1, nkpools, np/nkpools};
    utils::check(nkpools > 0 and nkpools <= nkpts_ibz, "pproc_t::analyt_cont: nkpools = {} <= 0 or > nkpts_ibz", nkpools);
    utils::check(_context.comm.size() % nkpools == 0, "pproc_t::analyt_cont: comm.size ({}) % nkpools ({}) != 0", _context.comm.size(), nkpools);

    auto FT = imag_axes_ft::read_iaft(_scf_output+".mbpt.h5");
    size_t niw   = (ac_params.stats == imag_axes_ft::fermi)? FT.nw_f() : FT.nw_b();

    app_log(1, "  Analytical continuation from iw to w");
    app_log(1, "  ------------------------------------");
    app_log(1, "    AC algorithm              =  {}", ac_params.ac_alg);
    app_log(1, "      - Statistics            = {}", imag_axes_ft::stats_enum_to_string(ac_params.stats));
    if (ac_params.ac_alg == "pade")
      app_log(1, "      - Nfit                  = {}", ac_params.Nfit);
    app_log(1, "");
    app_log(1, "    Input / Output");
    app_log(1, "      - Checkpoint            = {}", _scf_output + ".mbpt.h5");
    app_log(1, "      - Input path            = {}\n", dataset);
    app_log(1, "    Number of spins           = {}", ns);
    app_log(1, "    Number of k-points in IBZ = {}", nkpts_ibz);
    app_log(1, "    Number of bands           = {}\n", nbnds);
    app_log(1, "    Frequency mesh");
    app_log(1, "      - Range                 = [{}, {}] (a.u.)", ac_params.w_min, ac_params.w_max);
    app_log(1, "      - Number of points      = {}", ac_params.Nw);
    app_log(1, "      - Offset                = {}\n", ac_params.eta);
    app_log(2, "    Processor grid: (w, s, k, i) = ({}, {}, {}, {})\n", pgrid[0], pgrid[1], pgrid[2], pgrid[3]);

    // Prepare input and output buffer
    auto dA_iw_ski = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {niw, ns, nkpts_ibz, nbnds}, {1, 1, 1, 1});
    if(dataset != "DOS" and dataset != "A") { // backward compatibility with "A" name for spectral function
      size_t ntau  = (ac_params.stats == imag_axes_ft::fermi)? FT.nt_f() : FT.nt_b();
      auto dA_tau_ski = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {ntau, ns, nkpts_ibz, nbnds}, {1, 1, 1, 1});
      read_scf_dataset(dataset, dA_tau_ski);
      FT.tau_to_w(dA_tau_ski.local(), dA_iw_ski.local(), ac_params.stats);
      FT.check_leakage(dA_tau_ski, ac_params.stats, dataset);
    }
    else {
      std::string dataset_G = "G_tskij";
      std::string dataset_S = "S_skij";
      size_t ntau  = (ac_params.stats == imag_axes_ft::fermi)? FT.nt_f() : FT.nt_b();
      auto dG_tau_skij = make_distributed_array<local_Array_5D_t>(_context.comm, {1, 1, nkpools, np/nkpools, 1},
                                                                  {ntau, ns, nkpts_ibz, nbnds, nbnds}, {1, 1, 1, 1, 1});
      auto dS_skij = make_distributed_array<local_Array_4D_t>(_context.comm, {1, nkpools, np/nkpools, 1},
                                                              {ns, nkpts_ibz, nbnds, nbnds}, {1, 1, 1, 1});
      read_scf_dataset_full(dataset_G, "scf", dG_tau_skij);
      read_scf_dataset_full(dataset_S, "system", dS_skij);
      auto dA_tau_ski = evaluate_GS_diag(dG_tau_skij, dS_skij);
      auto dA_beta_ski = make_distributed_array<local_Array_3D_t>(_context.comm, {1, nkpools, np/nkpools},
                                                                  {ns, nkpts_ibz, nbnds}, {1, 1, 1});
      dA_beta_ski.communicator()->barrier();
      FT.tau_to_beta(dA_tau_ski.local(), dA_beta_ski.local());
      dA_beta_ski.communicator()->barrier();
      auto A_tt_loc = dA_beta_ski.local();
      ComplexType tr = nda::sum(A_tt_loc);
      dA_beta_ski.communicator()->all_reduce_in_place_n(&tr, 1, std::plus<>{});
      app_log(2, "MY: trace = {}", tr/nkpts_ibz);

      FT.tau_to_w(dA_tau_ski.local(), dA_iw_ski.local(), ac_params.stats);
      FT.check_leakage(dA_tau_ski, ac_params.stats, dataset);
    }
    auto dA_w_ski = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid,
                                                             {ac_params.Nw, ns, nkpts_ibz, nbnds}, {1, 1, 1, 1});

    // prepare imag and real frequency grids
    auto n_to_iw = nda::map([&](int n) { return FT.omega(n); } );
    nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));
    auto w_grid = analyt_cont::AC_t::w_grid(ac_params.w_min, ac_params.w_max, ac_params.Nw, ac_params.eta);

    // analytical continuation
    analyt_cont::AC_t AC(ac_params.ac_alg);
    bool is_iw_pos_only = false;
    AC.iw_to_w(dA_iw_ski.local(), iw_mesh, dA_w_ski.local(), w_grid, is_iw_pos_only, ac_params.Nfit);
    _context.comm.barrier();

    app_log(2, "Dump AC results for {}.", dataset);
    dump_ac_output(w_grid, dA_w_ski, iw_mesh, dA_iw_ski, dataset, "scf", -1);

    if(dataset == "DOS" or dataset == "A") { // backward compatibility with "A" name for spectral function
      nda::array<ComplexType, 1> Atot_w(ac_params.Nw);
      Atot_w() = 0;
      double spin_pref = (ns == 1) ? 2.0 : 1.0;
      spin_pref /= nkpts_ibz;
      for(size_t w = 0; w < ac_params.Nw; w++) {
        auto loc_A_ski = dA_w_ski.local()(w, nda::ellipsis{});
        Atot_w(w) = spin_pref * nda::sum(loc_A_ski);
      }
      dA_w_ski.communicator()->all_reduce_in_place_n(Atot_w.data(), Atot_w.size(), std::plus<>{});

      nda::array<ComplexType, 1> Atot_iw(dA_iw_ski.global_shape()[0]);
      Atot_iw() = 0;
      for(size_t iw = 0; iw < dA_iw_ski.global_shape()[0]; iw++) {
        auto loc_A_ski = dA_iw_ski.local()(iw, nda::ellipsis{});
        Atot_iw(iw) = spin_pref * nda::sum(loc_A_ski);
      }
      dA_iw_ski.communicator()->all_reduce_in_place_n(Atot_iw.data(), Atot_iw.size(), std::plus<>{});

      // dump output
      dump_ac_output(w_grid, Atot_w, iw_mesh, Atot_iw, "Atot", "scf", -1);
    }
  }

  void pproc_t::wannier_interpolation(mf::MF &mf, ptree const& pt, std::string project_file, 
                                      std::string target, std::string grp_name,
                                      long iter, bool translate_home_cell) 
  {
    std::string filename = _scf_output + ".mbpt.h5";
    double mu;
    {
      h5::file file(filename, 'r');
      utils::check(h5::group(file).has_subgroup(grp_name),
                   "wannier_interpolation: {} does not exist in {}", grp_name, filename);
      auto grp = h5::group(file).open_group(grp_name);
      if (iter==-1)
        h5::h5_read(grp, "final_iter", iter);
      h5::h5_read(grp, "iter"+std::to_string(iter)+"/mu", mu);
    }

    // read the target k-path
    nda::array<double ,2> kpts_interpolate;
    nda::array<long, 1> kpath_label_idx;
    std::string kpath_labels;
    nda::array<long, 2> Rpts_idx;
    nda::array<long, 1> Rpts_weights;
    {
      h5::file file(project_file, 'r');
      auto dft_grp = h5::group(file).open_group("dft_input");
      if (dft_grp.has_dataset("r_vector") and dft_grp.has_dataset("r_degeneracy")) {
        nda::h5_read(dft_grp, "r_vector", Rpts_idx);
        nda::h5_read(dft_grp, "r_degeneracy", Rpts_weights);
      } else {
        std::tie(Rpts_weights,Rpts_idx) = utils::WS_rgrid(mf.lattv(),mf.kp_grid()); 
      }

      auto kpath_str = io::get_value_with_default<std::string>(pt, "kpath", "");
      if( kpath_str != "" ) {

        // interpret kpath from provided string
        // Expects a list of: "label" kx1 ky1 kz1 
        auto nr = io::get_value_with_default<int>(pt, "bands_num_npoints", 100);
        std::vector<nda::array<double,2>> pts;
        std::vector<std::string> id;
        std::istringstream iss(kpath_str);
        std::string label;    
        double x,y,z;
        while(iss>>label >>x >>y >>z ) {
          pts.emplace_back(nda::array<double,2>{{x,y,z},{0.0,0.0,0.0}});   
          id.emplace_back(std::string(label));
          id.emplace_back(std::string(label));
        };
        utils::check(pts.size() > 1, " Problems with kpath. Not enough points provided.");
        for(int i=1; i<pts.size(); i++) {
          pts[i-1](1,nda::range::all) = pts[i](0,nda::range::all);
          id[2*i-1] = id[2*i];
        }  
        // remove last one
        pts.pop_back();
        id.pop_back();
        id.pop_back();
        // assemble kpath_labels, kpath_label_idx and print summary
        app_log(3, "  kpath: ");
        for(int i=0; i<pts.size(); i++) {
          kpath_labels.push_back(id[2*i][0]);
          app_log(3, "    {} - {} : {} - {}",
                  id[2*i],pts[i](0,nda::range::all),id[2*i+1],pts[i](1,nda::range::all));
        }
        app_log(3, "");
        kpath_labels.push_back(id[2*pts.size()-1][0]);
        std::tie(kpts_interpolate, kpath_label_idx) = utils::generate_kpath(mf.recv(),pts,id,nr); 
        // shifting to be consistent with fortran/dfttools convention
        kpath_label_idx[0] += 1;

      } else {

        app_log(2, "  [WARNING]: kpath string not provided. Looking for symm_kpath in file:{}\n",
                   filename);
        auto dft_misc_grp = h5::group(file).open_group("dft_misc_input");
        utils::check(dft_misc_grp.has_subgroup("symm_kpath"),
                     "wannier_interpolation: high-symmetry kpath is not found in {}. \n"
                     "Something is wrong with your Wannier90 or Wannier90 converter! ", project_file);
        auto kpath_grp = dft_misc_grp.open_group("symm_kpath");
        nda::h5_read(kpath_grp, "kpts", kpts_interpolate);
        h5::h5_read(kpath_grp, "labels", kpath_labels);
        nda::h5_read(kpath_grp, "label_idx", kpath_label_idx);

      }

      // transform vectors to cartesian coordinates
      auto recv = mf.recv();
      nda::array<double, 1> kp_temp(3);
      for(int ik=0; ik<kpts_interpolate.shape(0); ik++) {
        nda::blas::gemv(1.0, nda::transpose(recv), kpts_interpolate(ik, nda::range::all), 0.0, kp_temp);
        kpts_interpolate(ik, nda::range::all) = kp_temp;
      }
    }

    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=CoQui%20Wannier%20Interp.
    app_log(1, "╔═╗┌─┐╔═╗ ┬ ┬┬  ╦ ╦┌─┐┌┐┌┌┐┌┬┌─┐┬─┐  ╦┌┐┌┌┬┐┌─┐┬─┐┌─┐\n"
               "║  │ │║═╬╗│ ││  ║║║├─┤│││││││├┤ ├┬┘  ║│││ │ ├┤ ├┬┘├─┘\n"
               "╚═╝└─┘╚═╝╚└─┘┴  ╚╩╝┴ ┴┘└┘┘└┘┴└─┘┴└─  ╩┘└┘ ┴ └─┘┴└─┴o \n");
    app_log(1, "  Type                                  = {}", target);
    app_log(1, "  Projection matrices                   = {}", project_file);
    app_log(1, "  Input electronic structure:");
    app_log(1, "    - Coqui h5                          = {}", _scf_output+".mbpt.h5");
    app_log(1, "    - Data group                        = {}", grp_name);
    app_log(1, "    - Iteration                         = {}", iter);
    app_log(1, "  Input k-mesh:");
    app_log(1, "    - Monkhorst-Pack mesh               =  ({},{},{})",
            mf.kp_grid()(0), mf.kp_grid()(1), mf.kp_grid()(2));
    app_log(1, "    - Number of k-points                = {} total, {} in the IBZ", mf.nkpts(), mf.nkpts_ibz());
    app_log(1, "  Output k-mesh:");
    app_log(1, "    - Kpath                             = {}.", kpath_labels);
    app_log(1, "    - Number of k-points                = {}\n", kpts_interpolate.shape(0));

    projector_t proj(mf, project_file, translate_home_cell);
    long ns = mf.nspin();
    long nkpts = mf.nkpts();
    long nRpts = Rpts_idx.shape(0);
    long nImp = proj.nImps();
    long nImpOrbs = proj.nImpOrbs();
    utils::check(nImp==1, "wannier_interpolation: nImp != 1");

    if (target == "quasiparticle") {
      // unfold qp energies to full BZ
      auto sE_ski_full = unfold_qp_energy(_context, mf, filename, iter);

      // transform to a localized basis
      auto H_skIab_full = proj.downfold_k(sE_ski_full);

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto qp_grp = (iter_grp.has_subgroup("qp_approx"))?
            iter_grp.open_group("qp_approx") : iter_grp;
        auto winter_grp = (qp_grp.has_subgroup("wannier_inter")) ?
            qp_grp.open_group("wannier_inter") : qp_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "H_skIab", H_skIab_full, false);
      }
      _context.comm.barrier();

      // Fourier transform (coarse k -> R)
      nda::array<ComplexType, 5> H_sRIab(ns, nRpts, nImp, nImpOrbs, nImpOrbs);
      {
        math::shm::shared_array <nda::array_view<ComplexType, 2>> sf_Rk(
            std::addressof(_context.comm), std::addressof(_context.internode_comm), std::addressof(_context.node_comm),
            {nRpts, nkpts});
        utils::k_to_R_coefficients(_context.comm, Rpts_idx, mf.kpts(), mf.lattv(), sf_Rk);
        auto f_Rk = sf_Rk.local();
        for (size_t is = 0; is < ns; ++is) {
          auto H_kab_2D = nda::reshape(H_skIab_full(is, nda::ellipsis{}),
                                       std::array<long, 2>{nkpts, nImp*nImpOrbs*nImpOrbs});
          auto H_Rab_2D = nda::reshape(H_sRIab(is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_Rk, H_kab_2D, H_Rab_2D);
        }

        // check the largest imaginary element
        double max_imag = -1;
        nda::for_each(H_sRIab.shape(),
                      [&H_sRIab, &max_imag](auto ...i) { max_imag = std::max(max_imag, std::abs(H_sRIab(i...).imag())); });
        app_log(2, "Explicitly set the imaginary part of the QP Hamiltonian to zero");
        app_log(2, "  -> The largest imaginary value = {}. \n", max_imag);
        nda::for_each(H_sRIab.shape(),
                      [&H_sRIab](auto... i) mutable { H_sRIab(i...) = ComplexType(H_sRIab(i...).real(), 0.0); });
      }

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto qp_grp = (iter_grp.has_subgroup("qp_approx"))?
                      iter_grp.open_group("qp_approx") : iter_grp;
        auto winter_grp = (qp_grp.has_subgroup("wannier_inter")) ?
                          qp_grp.open_group("wannier_inter") : qp_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "H_sRIab", H_sRIab, false);
      }
      _context.comm.barrier();

      // Fourier transform the target k-points
      long nkpts_interpolate = kpts_interpolate.shape(0);
      nda::array<ComplexType, 4> H_skab_inter(ns, nkpts_interpolate, nImpOrbs, nImpOrbs);
      {
        math::shm::shared_array <nda::array_view<ComplexType, 2>> sf_kR(
            std::addressof(_context.comm), std::addressof(_context.internode_comm), std::addressof(_context.node_comm),
            {nkpts_interpolate, nRpts});
        utils::R_to_k_coefficients(_context.comm, Rpts_idx, Rpts_weights, kpts_interpolate, mf.lattv(), sf_kR);
        auto f_kR = sf_kR.local();
        for (size_t is = 0; is < ns; ++is) {
          auto H_Rab_2D = nda::reshape(H_sRIab(is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          auto H_kab_2D = nda::reshape(H_skab_inter(is, nda::ellipsis{}), std::array<long, 2>{nkpts_interpolate, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_kR, H_Rab_2D, H_kab_2D);
        }
      }

      // diagonalize the Hamiltonian in the localized basis
      nda::array<RealType, 3> E_ska_inter(ns, nkpts_interpolate, nImpOrbs);
      E_ska_inter() = 0.0;
      for (size_t isk=_context.comm.rank(); isk < ns*nkpts_interpolate; isk+=_context.comm.size()) {
        size_t is = isk / nkpts_interpolate;
        size_t ik = isk % nkpts_interpolate;
        E_ska_inter(is, ik, nda::range::all) = nda::linalg::eigenvalues(H_skab_inter(is, ik, nda::ellipsis{}));
      }
      _context.comm.all_reduce_in_place_n(E_ska_inter.data(), E_ska_inter.size(), std::plus<>{});

      // estimate the band gap
      //band_gap_estimator(E_ska_inter, mu);

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto qp_grp = (iter_grp.has_subgroup("qp_approx"))?
                      iter_grp.open_group("qp_approx") : iter_grp;
        auto winter_grp = (qp_grp.has_subgroup("wannier_inter")) ?
                          qp_grp.open_group("wannier_inter") : qp_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "H_skab", H_skab_inter, false);
        nda::h5_write(winter_grp, "E_ska", E_ska_inter, false);
        nda::h5_write(winter_grp, "kpts", kpts_interpolate, false);
        h5::h5_write(winter_grp, "kpt_labels", kpath_labels);
        nda::h5_write(winter_grp, "kpt_label_idx", kpath_label_idx, false);
      }

    } else if (target == "dyson") {
      // unfold dyson solutions to the full BZ
      bool include_H0 = true;
      auto sF_skij_full = unfold_1e_hamiltonian(_context, mf, filename,
                                                grp_name+"/iter"+std::to_string(iter)+"/F_skij",
                                                include_H0);
      auto sSigma_tskij_full = unfold_dynamic_hamiltonian(_context, mf, filename,
                                                          grp_name+"/iter"+std::to_string(iter)+"/Sigma_tskij");

      // transform to localized basis
      auto F_skIab_full = proj.downfold_k(sF_skij_full);
      auto Sigma_tskIab_full = proj.downfold_k(sSigma_tskij_full);
      long nts = Sigma_tskIab_full.shape(0);

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto winter_grp = (iter_grp.has_subgroup("wannier_inter")) ?
                          iter_grp.open_group("wannier_inter") : iter_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "F_skIab", F_skIab_full, false);
      }
      _context.comm.barrier();


      // Fourier transform (coarse k -> R)
      nda::array<ComplexType, 5> F_sRIab(ns, nRpts, nImp, nImpOrbs, nImpOrbs);
      nda::array<ComplexType, 6> Sigma_tsRIab(nts, ns, nRpts, nImp, nImpOrbs, nImpOrbs);
      {
        math::shm::shared_array <nda::array_view<ComplexType, 2>> sf_Rk(
            std::addressof(_context.comm), std::addressof(_context.internode_comm), std::addressof(_context.node_comm),
            {nRpts, nkpts});
        utils::k_to_R_coefficients(_context.comm, Rpts_idx, mf.kpts(), mf.lattv(), sf_Rk);
        auto f_Rk = sf_Rk.local();
        for (size_t is=0; is<ns; ++is) {
          auto F_Rab_2D = nda::reshape(F_sRIab(is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          auto F_kab_2D = nda::reshape(F_skIab_full(is, nda::ellipsis{}),
                                       std::array<long, 2>{nkpts, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_Rk, F_kab_2D, F_Rab_2D);
        }

        for (size_t its = 0; its < nts*ns; ++its) {
          size_t it = its/ns;
          size_t is = its%ns;
          auto S_Rab_2D = nda::reshape(Sigma_tsRIab(it, is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          auto S_kab_2D = nda::reshape(Sigma_tskIab_full(it, is, nda::ellipsis{}),
                                       std::array<long, 2>{nkpts, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_Rk, S_kab_2D, S_Rab_2D);
        }

        // check the largest imaginary element
        double max_imag = -1;
        nda::for_each(F_sRIab.shape(),
                      [&F_sRIab, &max_imag](auto ...i) { max_imag = std::max(max_imag, std::abs(F_sRIab(i...).imag())); });
        app_log(2, "Explicitly set the imaginary part of the static self-energy in the R-space to zero");
        app_log(2, "  -> The largest imaginary value = {}. \n", max_imag);
        nda::for_each(F_sRIab.shape(),
                      [&F_sRIab](auto... i) mutable { F_sRIab(i...) = ComplexType(F_sRIab(i...).real(), 0.0); });

        nda::for_each(Sigma_tsRIab.shape(),
                      [&Sigma_tsRIab, &max_imag](auto ...i) { max_imag = std::max(max_imag, std::abs(Sigma_tsRIab(i...).imag())); });
        app_log(2, "Explicitly set the imaginary part of the self-energy(tau) in the R-space to zero");
        app_log(2, "  -> The largest imaginary value = {}. \n", max_imag);
        nda::for_each(Sigma_tsRIab.shape(),
                      [&Sigma_tsRIab](auto... i) mutable { Sigma_tsRIab(i...) = ComplexType(Sigma_tsRIab(i...).real(), 0.0); });
      }

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto winter_grp = (iter_grp.has_subgroup("wannier_inter")) ?
                          iter_grp.open_group("wannier_inter") : iter_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "F_sRIab", F_sRIab, false);
      }
      _context.comm.barrier();

      // Fourier transform the target k-points
      long nkpts_interpolate = kpts_interpolate.shape(0);
      nda::array<ComplexType, 4> F_skab_inter(ns, nkpts_interpolate, nImpOrbs, nImpOrbs);
      nda::array<ComplexType, 5> Sigma_tskab_inter(nts, ns, nkpts_interpolate, nImpOrbs, nImpOrbs);
      {
        math::shm::shared_array <nda::array_view<ComplexType, 2>> sf_kR(
            std::addressof(_context.comm), std::addressof(_context.internode_comm), std::addressof(_context.node_comm),
            {nkpts_interpolate, nRpts});
        utils::R_to_k_coefficients(_context.comm, Rpts_idx, Rpts_weights, kpts_interpolate, mf.lattv(), sf_kR);
        auto f_kR = sf_kR.local();
        for (size_t is = 0; is < ns; ++is) {
          auto F_Rab_2D = nda::reshape(F_sRIab(is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          auto F_kab_2D = nda::reshape(F_skab_inter(is, nda::ellipsis{}),
                                       std::array<long, 2>{nkpts_interpolate, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_kR, F_Rab_2D, F_kab_2D);
        }
        for (size_t its = 0; its < nts*ns; ++its) {
          size_t it = its/ns;
          size_t is = its%ns;
          auto S_Rab_2D = nda::reshape(Sigma_tsRIab(it, is, nda::ellipsis{}),
                                       std::array<long, 2>{nRpts, nImp*nImpOrbs*nImpOrbs});
          auto S_kab_2D = nda::reshape(Sigma_tskab_inter(it, is, nda::ellipsis{}),
                                       std::array<long, 2>{nkpts_interpolate, nImp*nImpOrbs*nImpOrbs});
          nda::blas::gemm(f_kR, S_Rab_2D, S_kab_2D);
        }
      }

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto winter_grp = (iter_grp.has_subgroup("wannier_inter")) ?
                          iter_grp.open_group("wannier_inter") : iter_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "F_skab", F_skab_inter, false);
        nda::h5_write(winter_grp, "Sigma_tskab", Sigma_tskab_inter, false);
        nda::h5_write(winter_grp, "kpts", kpts_interpolate, false);
        h5::h5_write(winter_grp, "kpt_labels", kpath_labels);
        nda::h5_write(winter_grp, "kpt_label_idx", kpath_label_idx, false);
      }
      _context.comm.barrier();

      app_log(2, "Solving the Dyson equation along the kpath within the Wannier energy window.\n");

      // Dyson equation
      auto FT = imag_axes_ft::read_iaft(filename);
      nda::array<ComplexType, 5> G_wskab_inter(FT.nw_f(), ns, nkpts_interpolate, nImpOrbs, nImpOrbs);
      nda::matrix<ComplexType> X(nImpOrbs, nImpOrbs);
      auto eye = nda::eye<ComplexType>(nImpOrbs);

      // solve the dyson equation
      FT.check_leakage(Sigma_tskab_inter, imag_axes_ft::fermi, std::addressof(_context.comm), "Self-energy in Wannier basis");
      FT.tau_to_w(Sigma_tskab_inter, G_wskab_inter, imag_axes_ft::fermi);
      for (size_t nsk=0; nsk<FT.nw_f()*ns*nkpts_interpolate; ++nsk) {
        long n = nsk / (ns * nkpts_interpolate); // nsk = n*ns*nkpts_interpolate + s*nkpts_interpolate + k
        long s = (nsk / nkpts_interpolate) % ns;
        long k = nsk % nkpts_interpolate;
        if (nsk%_context.comm.size() == _context.comm.rank()) {
          long wn = FT.wn_mesh()(n);
          ComplexType omega_mu = FT.omega(wn) + mu;
          X = omega_mu * eye - F_skab_inter(s, k, nda::ellipsis{}) - G_wskab_inter(n, s, k, nda::ellipsis{});
          G_wskab_inter(n, s, k, nda::ellipsis{}) = nda::inverse(X);
        } else {
          G_wskab_inter(n, s, k, nda::ellipsis{}) = 0.0;
        }
      }
      // split all_reduce() to avoid mpi count overflow
      for (size_t shift=0; shift<G_wskab_inter.size(); shift+=size_t(1e9)) {
        ComplexType* start = G_wskab_inter.data()+shift;
        size_t count = (shift+size_t(1e9) < G_wskab_inter.size())? size_t(1e9) : G_wskab_inter.size()-shift;
        _context.comm.all_reduce_in_place_n(start, count, std::plus<>{});
      }

      {
        FT.w_to_tau(G_wskab_inter, Sigma_tskab_inter, imag_axes_ft::fermi);
        FT.check_leakage(Sigma_tskab_inter, imag_axes_ft::fermi, std::addressof(_context.comm), "Green's function in Wannier basis");
      }

      if (_context.comm.root()) {
        h5::file file(filename, 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iter));
        auto winter_grp = (iter_grp.has_subgroup("wannier_inter")) ?
                          iter_grp.open_group("wannier_inter") : iter_grp.create_group("wannier_inter");
        nda::h5_write(winter_grp, "G_tskab", Sigma_tskab_inter, false);
        nda::h5_write(winter_grp, "G_wskab", G_wskab_inter, false);
      }
      _context.comm.barrier();
    } else {
      utils::check(false, "pproc_t::wannier_interpolation: incorrect target: quasiparticle or dyson");
    }
    app_log(1, "####### wannier interpolation routines end #######\n");
  }

  void pproc_t::spectral_interpolation(mf::MF &mf, ptree const& pt, 
                                       std::string project_file, analyt_cont::ac_context_t &ac_params,
                                       std::string grp_name, long iter, bool translate_home_cell) {
    // interpolate self-energy and fock along the k-path
    wannier_interpolation(mf, pt, project_file, "dyson", grp_name, iter, translate_home_cell);
    nda::array<ComplexType, 5> G_wskab_inter;
    std::string filename = _scf_output + ".mbpt.h5";
    auto FT = imag_axes_ft::read_iaft(filename);
    {
      h5::file file(filename, 'r');
      auto grp = h5::group(file).open_group(grp_name);
      if (iter==-1)
        h5::h5_read(grp, "final_iter", iter);
      nda::h5_read(grp, "iter"+std::to_string(iter)+"/wannier_inter/G_wskab", G_wskab_inter);
    }
    auto [niw, ns, nkpts, nbnd, nbnd2] = G_wskab_inter.shape();

    int np = _context.comm.size();
    int nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.5);
    std::array<long, 4> pgrid = {1, 1, nkpools, np/nkpools};
    utils::check(nkpools > 0 and nkpools <= nkpts, "pproc_t::spectral_interpolation: nkpools = {} <= 0 or > nkpts", nkpools);
    utils::check(_context.comm.size() % nkpools == 0, "pproc_t::spectral_interpolation: comm.size ({}) % nkpools ({}) != 0",
                 _context.comm.size(), nkpools);
    utils::check(np/nkpools <= nbnd, "pproc_t::spectral_interpolation: too many processors ({}) along band indices ({})", np/nkpools, nbnd);

    app_log(1, "  Analytical continuation from iw to w");
    app_log(1, "  ------------------------------------");
    app_log(1, "    AC algorithm         = {}", ac_params.ac_alg);
    app_log(1, "      - Statistics       = {}", imag_axes_ft::stats_enum_to_string(ac_params.stats));
    if (ac_params.ac_alg == "pade")
      app_log(1, "      - Nfit             = {}", ac_params.Nfit);
    app_log(1, "");
    app_log(1, "    Input / Output");
    app_log(1, "      - Checkpoint       = {}", _scf_output + ".mbpt.h5");
    app_log(1, "      - Input path       = {}/iter{}/wannier_inter/G_wskab", grp_name, iter);
    app_log(1, "      - Output path      = {}/iter{}/ac/G_wskab_inter\n", grp_name, iter);
    app_log(1, "    Number of spins      = {}", ns);
    app_log(1, "    Number of k-points   = {}", nkpts);
    app_log(1, "    Number of bands      = {}\n", nbnd);
    app_log(1, "    Frequency mesh");
    app_log(1, "      - Range            = [{}, {}] (a.u.)", ac_params.w_min, ac_params.w_max);
    app_log(1, "      - Number of points = {}", ac_params.Nw);
    app_log(1, "      - Offset           = {}\n", ac_params.eta);
    app_log(2, "    Processor grid (w, s, k, i) = ({}, {}, {}, {})\n", pgrid[0], pgrid[1], pgrid[2], pgrid[3]);

    // Prepare input and output buffer
    using math::nda::make_distributed_array;
    using local_Array_4D_t = nda::array<ComplexType, 4>;
    auto dG_wska = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {niw, ns, nkpts, nbnd}, {1, 1, 1, 1});
    auto w_rng = dG_wska.local_range(0);
    auto s_rng = dG_wska.local_range(1);
    auto k_rng = dG_wska.local_range(2);
    auto a_rng = dG_wska.local_range(3);
    auto G_wska_loc = dG_wska.local();
    for ( auto [i, a] : itertools::enumerate(a_rng) )
      G_wska_loc(nda::range::all, nda::range::all, nda::range::all, i) = G_wskab_inter(w_rng, s_rng, k_rng, a, a);
    auto dA_wska = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {ac_params.Nw, ns, nkpts, nbnd}, {1, 1, 1, 1});

    // prepare imag and real frequency grids
    auto n_to_iw = nda::map([&](int n) { return FT.omega(n); } );
    nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));
    auto w_grid = analyt_cont::AC_t::w_grid(ac_params.w_min, ac_params.w_max, ac_params.Nw, ac_params.eta);

    // analytical continuation
    analyt_cont::AC_t AC(ac_params.ac_alg);
    bool is_iw_pos_only = false;
    AC.iw_to_w(dG_wska.local(), iw_mesh, dA_wska.local(), w_grid, is_iw_pos_only, ac_params.Nfit);
    _context.comm.barrier();

    dump_ac_output(w_grid, dA_wska, iw_mesh, dG_wska, "G_wskab_inter", grp_name, iter);
    _context.comm.barrier();
  }

  void pproc_t::local_density_of_state(mf::MF &mf, std::string project_file,
                                       analyt_cont::ac_context_t &ac_params,
                                       std::string grp_name, long iter, bool translate_home_cell) {
    using Array_view_5D_t = nda::array_view<ComplexType, 5>;

    std::string filename = _scf_output + ".mbpt.h5";
    auto FT = imag_axes_ft::read_iaft(filename);
    projector_t proj(mf, project_file, translate_home_cell);

    // read lattice Green's function from "{grp_name}/iter{iter}"
    auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
        _context.comm, _context.internode_comm, _context.node_comm,
        {FT.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});

    if (_context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto grp = h5::group(file).open_group(grp_name);
      if (iter==-1) {
        h5::h5_read(grp, "final_iter", iter);
      }
      auto G_loc = sG_tskij.local();
      nda::h5_read(grp, "iter"+std::to_string(iter)+"/G_tskij", G_loc);
    }
    _context.node_comm.barrier();

    // downfold to Wannier basis
    auto G_tskIab = proj.downfold_k_fbz(sG_tskij);
    auto [nts, ns, nkpts, nImps, nbnd, nbnd2] = G_tskIab.shape();
    auto niw = FT.nw_f();
    auto G_tskab = nda::reshape(G_tskIab, std::array<long, 5>{nts, ns, nkpts, nbnd, nbnd});
    nda::array<ComplexType, 5> G_wskab(niw, ns, nkpts, nbnd, nbnd);
    FT.tau_to_w(G_tskab, G_wskab, imag_axes_ft::fermi);

    // analytical continuation
    int np = _context.comm.size();
    int nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.5);
    std::array<long, 4> pgrid = {1, 1, nkpools, np/nkpools};
    utils::check(nkpools > 0 and nkpools <= nkpts, "pproc_t::local_density_of_state: nkpools = {} <= 0 or > nkpts", nkpools);
    utils::check(_context.comm.size() % nkpools == 0, "pproc_t::local_density_of_state: comm.size ({}) % nkpools ({}) != 0",
                 _context.comm.size(), nkpools);
    utils::check(np/nkpools <= nbnd, "pproc_t::local_density_of_state: too many processors ({}) along band indices ({})", np/nkpools, nbnd);

    app_log(2, "Analytical continuation for orbital-resolved density of states");
    app_log(2, "------------------------------------");
    app_log(2, "projection matrices: {}", project_file);
    app_log(2, "input electronic structure:");
    app_log(2, "  - coqui h5: {}", _scf_output+".mbpt.h5");
    app_log(2, "  - data group: {}", grp_name);
    app_log(2, "  - iteration: {}", iter);
    app_log(2, "  - Monkhorst-Pack mesh:  ({},{},{})", mf.kp_grid()(0), mf.kp_grid()(1), mf.kp_grid()(2));
    app_log(2, "  - ns, nkpts, nbnd: {}, {}, {}", ns, nkpts, nbnd);
    app_log(2, "analytical continuation:");
    app_log(2, "  - ac alg:  {}", ac_params.ac_alg);
    app_log(2, "  - stats:   {}", imag_axes_ft::stats_enum_to_string(ac_params.stats));
    app_log(2, "  - w grid:  [{}, {}] (a.u.) with {} frequency points", ac_params.w_min, ac_params.w_max, ac_params.Nw);
    app_log(2, "  - eta:     {}", ac_params.eta);
    app_log(2, "  - processor grid: (w, s, k, i) = ({}, {}, {}, {})\n", pgrid[0], pgrid[1], pgrid[2], pgrid[3]);

    // Prepare input and output buffer
    using math::nda::make_distributed_array;
    using local_Array_4D_t = nda::array<ComplexType, 4>;
    auto dG_wska = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {niw, ns, nkpts, nbnd}, {1, 1, 1, 1});
    auto w_rng = dG_wska.local_range(0);
    auto s_rng = dG_wska.local_range(1);
    auto k_rng = dG_wska.local_range(2);
    auto a_rng = dG_wska.local_range(3);
    auto G_wska_loc = dG_wska.local();
    for ( auto [i, a] : itertools::enumerate(a_rng) )
      G_wska_loc(nda::range::all, nda::range::all, nda::range::all, i) = G_wskab(w_rng, s_rng, k_rng, a, a);
    auto dA_wska = make_distributed_array<local_Array_4D_t>(_context.comm, pgrid, {ac_params.Nw, ns, nkpts, nbnd}, {1, 1, 1, 1});

    // prepare imag and real frequency grids
    auto n_to_iw = nda::map([&](int n) { return FT.omega(n); } );
    nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));
    auto w_grid = analyt_cont::AC_t::w_grid(ac_params.w_min, ac_params.w_max, ac_params.Nw, ac_params.eta);

    // analytical continuation
    analyt_cont::AC_t AC(ac_params.ac_alg);
    bool is_iw_pos_only = false;
    AC.iw_to_w(dG_wska.local(), iw_mesh, dA_wska.local(), w_grid, is_iw_pos_only, ac_params.Nfit);
    _context.comm.barrier();

    nda::array<ComplexType, 3> A_wsa(ac_params.Nw, ns, nbnd);
    auto A_wska_loc = dA_wska.local();
    for ( auto [ik, k] : itertools::enumerate(k_rng) ) {
      for ( auto [i, a] : itertools::enumerate(a_rng) ) {
        A_wsa(nda::range::all, nda::range::all, a) += A_wska_loc(nda::range::all, nda::range::all, ik, i);
      }
    }
    _context.comm.all_reduce_in_place_n(A_wsa.data(), A_wsa.size(), std::plus<>{});
    A_wsa() /= nkpts;

    nda::array<ComplexType, 3> G_wsa(niw, ns, nbnd);
    for (size_t ik = 0; ik < nkpts; ++ik) {
      for (size_t a = 0; a < nbnd; ++a) {
        G_wsa(nda::range::all, nda::range::all, a) += G_wskab(nda::range::all, nda::range::all, ik, a, a);
      }
    }

    // dump output
    dump_ac_output(w_grid, dA_wska, iw_mesh, dG_wska, "G_wskab_fbz", grp_name, iter);
    _context.comm.barrier();
  }

  template<nda::ArrayOfRank<5> local_Array_5D_t, nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
  auto pproc_t::evaluate_GS_diag(memory::darray_t<local_Array_5D_t, communicator_t> & dG_tau_skij,
                                 memory::darray_t<local_Array_4D_t, communicator_t> & dS_skij)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator> {
    using local_Array_2D_t = nda::array<ComplexType, 2>;
    using math::nda::make_distributed_array;

      auto [ntau, ns, nkpts, nbnds, nbnds2] = dG_tau_skij.global_shape();

      auto buffer1 = make_distributed_array<local_Array_5D_t>(_context.comm, dG_tau_skij.grid(), dG_tau_skij.global_shape(), dG_tau_skij.block_size());

      {
          auto [tau_origin, s_origin, k_origin, i_origin, j_origin] = dG_tau_skij.origin();
          utils::check(tau_origin == 0, "pproc_t::evaluate_GS_diag: tau_origin is assumed to be 0");
          utils::check(s_origin == 0, "pproc_t::evaluate_GS_diag: s_origin is assumed to be 0");
          
          int color = k_origin;
          int key = _context.comm.rank();
          mpi3::communicator k_intra_comm = _context.comm.split(color, key);
          
          auto dG_ij = make_distributed_array<local_Array_2D_t>(k_intra_comm, {k_intra_comm.size(), 1}, {nbnds, nbnds}, {1, 1});
          auto dS_ij = make_distributed_array<local_Array_2D_t>(k_intra_comm, {k_intra_comm.size(), 1}, {nbnds, nbnds}, {1, 1});
          auto dGS_ij = make_distributed_array<local_Array_2D_t>(k_intra_comm, {k_intra_comm.size(), 1}, {nbnds, nbnds}, {1, 1});
          
          size_t nkpts_loc = dG_tau_skij.local_shape()[2]; 
          utils::check(dG_tau_skij.local_shape()[1] == dS_skij.local_shape()[0], "pproc_t::evaluate_GS_diag: different local shapes of dG and dS");
          utils::check(dG_tau_skij.local_shape()[2] == dS_skij.local_shape()[1], "pproc_t::evaluate_GS_diag: different local shapes of dG and dS");
          utils::check(dG_tau_skij.local_shape()[3] == dS_skij.local_shape()[2], "pproc_t::evaluate_GS_diag: different local shapes of dG and dS");
          utils::check(dG_tau_skij.local_shape()[4] == dS_skij.local_shape()[3], "pproc_t::evaluate_GS_diag: different local shapes of dG and dS");

          for(size_t it = 0; it < ntau; it++)
          for(size_t is = 0; is < ns; is++) 
          for(size_t ik = 0; ik < nkpts_loc; ik++) {
              auto G_loc = dG_tau_skij.local()(it, is, ik, nda::ellipsis{});
              auto S_loc = dS_skij.local()(is, ik, nda::ellipsis{});
              dG_ij.local() = G_loc;
              dS_ij.local() = S_loc;
              math::nda::slate_ops::multiply(dG_ij, dS_ij, dGS_ij);
              buffer1.local()(it, is, ik, nda::ellipsis{}) = dGS_ij.local();
          }
      }

      int np = _context.comm.size();
      int ntpools = utils::find_proc_grid_max_npools(np, ntau, 0.2);
      auto buffer2 = make_distributed_array<local_Array_5D_t>(_context.comm, {ntpools, 1, np/ntpools, 1, 1}, dG_tau_skij.global_shape(), {1,1,1,1,1});

      math::nda::redistribute(buffer1, buffer2);

      auto buffer2_diag = make_distributed_array<local_Array_4D_t>(_context.comm, {ntpools, 1, np/ntpools, 1}, {ntau, ns, nkpts, nbnds}, {1,1,1,1});

      auto [ntau_loc, ns_loc, nkpts_loc, nbnd_loc] = buffer2_diag.local_shape();
      for(size_t it = 0; it < ntau_loc; it++)
      for(size_t is = 0; is < ns; is++) 
      for(size_t ik = 0; ik < nkpts_loc; ik++) 
      for (size_t i = 0; i < nbnds; i++) {
          buffer2_diag.local()(it, is, ik, i) = 0.5 * (buffer2.local()(it, is, ik, i, i) + nda::conj(buffer2.local()(it, is, ik, i, i)));
      }
      int nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.2);
      auto GS_diag = make_distributed_array<local_Array_4D_t>(_context.comm, {1, 1, nkpools, np/nkpools}, {ntau, ns, nkpts, nbnds}, {1,1,1,1});
      math::nda::redistribute(buffer2_diag, GS_diag);
      return GS_diag;
   }

} // methods
