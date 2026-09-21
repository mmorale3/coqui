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


#ifndef COQUI_PPROC_DRIVERS_HPP
#define COQUI_PPROC_DRIVERS_HPP

#include <limits>
#include <array>
#include <format>
#include <string>
#include <vector>
#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_common.hpp"
#include "numerics/ac/ac_context.h"
#include "methods/pproc/pproc_t.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "hamiltonian/one_body_hamiltonian.hpp"

namespace methods {
  /**
   * Standalone Pade analytical continuation without MPI context dependency.
   * First dimension is assumed to be the Matsubara frequency axis.
   */
  template<nda::MemoryArrayOfRank<1> mesh_iw_t, nda::MemoryArray Array_iw_t>
  nda::array<ComplexType, nda::get_rank<Array_iw_t>> pade(Array_iw_t &&A_iw, mesh_iw_t &&iw_mesh,
                                                          double w_min, double w_max, long Nw, double eta,
                                                          bool is_iw_pos_only=false, int Nfit=-1) {
    constexpr int rank = nda::get_rank<Array_iw_t>;
    static_assert(rank >= 1, "pproc_drivers.hpp::pade: input array rank must be >= 1");

    auto A_w_shape = A_iw.shape();
    A_w_shape[0] = Nw;

    auto w_grid = analyt_cont::AC_t::w_grid(w_min, w_max, Nw, eta);

    nda::array<ComplexType, rank> A_w(A_w_shape);
    analyt_cont::AC_t AC("pade");
    AC.iw_to_w(A_iw, iw_mesh, A_w, w_grid, is_iw_pos_only, Nfit);

    return A_w;
  }

  /**
   * Post-processing routines with arguments in property tree.
   * @param [INPUT] pp_type - type of post-processing, allowed options: ac, unfold
   * @param [INPUT] mf - mean-field instance
   * Requires:
   *   - prefix: prefix to the scf output h5 file.
   * Optional arguments:
   *   - outdir: location of directory with files
   * Optional arguments for pp_type = ac (analytical continuation):
   *   - ac_alg: algorithm for analytical continuation, allowed options: pade
   *   - dataset: input dataset
   *   - stats: statistics, allowed options: fermion, boson
   *   - w_min: minimum real frequency
   *   - w_max: maximum real frequency
   *   - Nw: number of real frequency points
   *   - eta: shift above the real-frequency axis
   * Optional arguments for pp_type = unfold_bz (unfold BZ from irreducible to the 1st BZ)
   */
  void post_processing(std::string pp_type, std::shared_ptr<mf::MF> mf, ptree const& pt) {
    auto mpi = mf->mpi();
    if (mpi->comm.size()%mpi->node_comm.size()!=0) {
      APP_ABORT("pproc: number of processors on each node should be the same.");
    }

    std::string err = std::string("pproc - Incorrect input - ");
    auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
    auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

    if (pp_type == "ac") {

      pproc_t pp(*mpi, prefix, outdir);
      auto dataset = io::get_value_with_default<std::string>(pt,"dataset","G_tskij");

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto stats   = io::get_value_with_default<std::string>(pt,"stats","fermion");
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-10.0);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",10.0);
      auto Nw      = io::get_value_with_default<int>(pt,"Nw",5000);
      auto eta     = io::get_value_with_default<double>(pt,"eta",0.01);
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", -1);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::string_to_stats_enum(stats), Nfit, eta, w_min, w_max, Nw);
      pp.analyt_cont(*mf, ac_context, dataset);

    } else if (pp_type == "unfold_bz") {

      std::string scf_output = outdir+"/"+prefix;
      unfold_bz(*mpi, *mf, scf_output);

    } else if (pp_type == "band_interpolation") {

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<long>(pt, "iteration", -1);

      std::string scf_output = outdir+"/"+prefix;

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(iteration==0 or iteration==-1,
                     "band_interpolation: iteration = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     iteration, scf_output);

        // dump mf data to "scf_output".mbpt.h5
        auto psp = hamilt::make_pseudopot(*mf);
        // For band interpolation, the only relevant parameter is ``beta`` which will be used to 
        // determine the Fermi level in ``write_mf_data``. 
        imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }

      // Check if a QP solution exist already. 
      // If not, compute QP energies on the IBZ k-mesh from dynamic self-energy and write to the checkpoint file.
      bool heff_exists = false;
      double beta = 0.0; 
      if (mpi->comm.root()) {
        h5::file file(scf_output+".mbpt.h5", 'r');
        auto grp = h5::group(file).open_group(grp_name);
        if (iteration == -1) h5::read(grp, "final_iter", iteration);
        auto iter_grp = grp.open_group("iter"+std::to_string(iteration));
        // Check if Heff_skij dataset exists
        if (iter_grp.has_dataset("Heff_skij") or (iter_grp.has_subgroup("qp_approx") and iter_grp.open_group("qp_approx").has_dataset("Heff_skij"))) {
          heff_exists = true;
        }
        h5::read(h5::group(file), "imaginary_fourier_transform/beta", beta);
      }
      std::array<double,3> buffer = {(heff_exists)? 1.0 : 0.0, double(iteration), beta};
      mpi->comm.broadcast_n(buffer.data(), buffer.size(), 0);
      heff_exists = (buffer[0] == 1.0)? true : false;
      iteration = static_cast<int>(buffer[1]);
      beta = buffer[2];
      if (!heff_exists) {
        // Compute QP energies from dynamic self-energy on IBZ 
        qp_params_t qp_params;
        qp_params.ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
        qp_params.qp_type = io::get_value_with_default<std::string>(pt, "qp_type", "sc");
        qp_params.eta     = io::get_value_with_default<double>(pt, "qp_eta", M_PI/beta);
        qp_params.Nfit    = io::get_value_with_default<int>(pt, "qp_Nfit", 18);
        qp_params.tol     = io::get_value_with_default<double>(pt, "qp_tol", 1e-8);

        pp.compute_qp_on_ibz_kmesh(*mf, qp_params, grp_name, iteration); 
      }

      pp.wannier_interpolation(*mf, pt, wannier_file, "quasiparticle", grp_name, iteration, trans_home_cell);

    } else if (pp_type == "qp_gaps") {
      // LFF (notes/lff_aux_plan.md): quasiparticle energies on the IBZ k-mesh from the Dyson self-energy of a scGW
      // checkpoint (Pade AC of the MO-diagonal Sigma + the QP equation, compute_qp_on_ibz_kmesh -> qp_approx/E_ska)
      // followed by the band gaps on that mesh: fundamental (min CBM - max VBM over the mesh, occupation by mu),
      // direct at Gamma, and the smallest direct gap -- the qsGW-hat benchmark's convention. Results are printed in
      // eV and written to <grp>/iter<n>/qp_approx/gaps; epsilon_inf of the same iteration is printed alongside.
      // P24 / G31 (notes/vertex_perf_plan.md): with qp_gaps_interp = true the QP Hamiltonian is in addition
      // Wannier-interpolated (pproc_t::qp_bands_on_mesh, the band_interpolation machinery) onto the uniform
      // qp_gaps_interp_mesh = [n1, n2, n3] (default: the mean-field mesh, on which the interpolation is exact),
      // and the same gaps are reported on that mesh -- the off-mesh (indirect) CBM/VBM. Needs wannier_file (the
      // projector h5 of band_interpolation; translate_home_cell as there); only the projector's band window is
      // interpolated, so the window must contain the band edges. Logged next to the mesh numbers and written to
      // the same gaps group with the suffix "_interp" (plus kpts_crys_interp / E_ska_interp / interp_mesh).
      pproc_t pp(*mpi, prefix, outdir);
      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<long>(pt, "iteration", -1);
      auto qp_gaps_interp = io::get_value_with_default<bool>(pt, "qp_gaps_interp", false);
      auto qp_gaps_interp_mesh = io::get_array_with_default<long>(pt, "qp_gaps_interp_mesh",
          std::vector<long>{mf->kp_grid()(0), mf->kp_grid()(1), mf->kp_grid()(2)});
      auto wannier_file = (qp_gaps_interp) ? io::get_value<std::string>(pt, "wannier_file", err+"wannier_file (required by qp_gaps_interp)")
                                           : std::string("");
      auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
      std::string scf_output = outdir+"/"+prefix;
      utils::check(std::filesystem::exists(scf_output+".mbpt.h5"), "qp_gaps: {}.mbpt.h5 does not exist.", scf_output);
      double beta = 0.0;
      if (mpi->comm.root()) {
        h5::file file(scf_output+".mbpt.h5", 'r');
        auto grp = h5::group(file).open_group(grp_name);
        if (iteration == -1) h5::read(grp, "final_iter", iteration);
        h5::read(h5::group(file), "imaginary_fourier_transform/beta", beta);
      }
      std::array<double,2> buffer = {double(iteration), beta};
      mpi->comm.broadcast_n(buffer.data(), buffer.size(), 0);
      iteration = long(buffer[0]); beta = buffer[1];
      qp_params_t qp_params;
      qp_params.ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      qp_params.qp_type = io::get_value_with_default<std::string>(pt, "qp_type", "sc");
      qp_params.eta     = io::get_value_with_default<double>(pt, "qp_eta", M_PI/beta);
      qp_params.Nfit    = io::get_value_with_default<int>(pt, "qp_Nfit", 18);
      qp_params.tol     = io::get_value_with_default<double>(pt, "qp_tol", 1e-8);
      pp.compute_qp_on_ibz_kmesh(*mf, qp_params, grp_name, iteration);
      mpi->comm.barrier();
      if (mpi->comm.root()) {
        const double HA = 27.211386245988;
        nda::array<double, 3> E;
        double mu = 0.0, eps_inf = -1.0;
        h5::file file(scf_output+".mbpt.h5", 'a');
        auto iter_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iteration));
        auto qp_grp = iter_grp.open_group("qp_approx");
        nda::h5_read(qp_grp, "E_ska", E);
        h5::h5_read(qp_grp, "mu", mu);
        if (iter_grp.has_dataset("epsilon_inf")) h5::h5_read(iter_grp, "epsilon_inf", eps_inf);
        const long ns = E.shape(0), nk = E.shape(1), nb = E.shape(2);
        auto kc = mf->kpts_crystal();
        utils::check(nk <= kc.shape(0), "qp_gaps: E_ska has {} k-points, the mean field {}.", nk, kc.shape(0));
        long ik_gamma = -1;
        for (long ik = 0; ik < nk and ik_gamma < 0; ++ik) {
          double d = 0.0;
          for (int i = 0; i < 3; ++i) d = std::max(d, std::abs(kc(ik, i) - std::round(kc(ik, i))));
          if (d < 1e-6) ik_gamma = ik;
        }
        app_log(1, "\n  [qp_gaps] {} iteration {}: QP energies (Pade AC of the Dyson Sigma, {} solver, Nfit {}, eta {:.3e} Ha) on {} IBZ k-points x {} bands; mu = {:.6f} Ha; epsilon_inf = {}",
                prefix, iteration, qp_params.qp_type, qp_params.Nfit, qp_params.eta, nk, nb, mu, (eps_inf > 0.0) ? std::to_string(eps_inf) : std::string("n/a"));
        auto gaps_grp = qp_grp.has_subgroup("gaps") ? qp_grp.open_group("gaps") : qp_grp.create_group("gaps");
        for (long is = 0; is < ns; ++is) {
          double vbm = -1e9, cbm = 1e9, dmin = 1e9, dgam = -1.0;
          long kv = -1, kcb = -1, kd = -1, nocc_g = -1;
          for (long ik = 0; ik < nk; ++ik) {
            double v = -1e9, c = 1e9; long nocc = 0;
            for (long a = 0; a < nb; ++a) {
              const double e = E(is, ik, a);
              if (e < mu) { ++nocc; v = std::max(v, e); } else c = std::min(c, e);
            }
            if (v > vbm) { vbm = v; kv = ik; }
            if (c < cbm) { cbm = c; kcb = ik; }
            if (c - v < dmin) { dmin = c - v; kd = ik; }
            if (ik == ik_gamma) { dgam = c - v; nocc_g = nocc; }
          }
          app_log(1, "  [qp_gaps]   spin {}: fundamental gap (mesh) = {:.4f} eV  [VBM {:.4f} eV at k {} , CBM {:.4f} eV at k {}]; direct gap at Gamma = {} eV (k {}, {} occupied QP bands); smallest direct gap = {:.4f} eV at k {}",
                  is, (cbm - vbm) * HA, vbm * HA, kv, cbm * HA, kcb, (dgam > 0.0) ? std::to_string(dgam * HA) : std::string("n/a"), ik_gamma, nocc_g, dmin * HA, kd);
          const std::string sfx = (ns > 1) ? "_s" + std::to_string(is) : "";
          h5::h5_write(gaps_grp, "fundamental_eV" + sfx, (cbm - vbm) * HA);
          h5::h5_write(gaps_grp, "direct_gamma_eV" + sfx, dgam * HA);
          h5::h5_write(gaps_grp, "direct_min_eV" + sfx, dmin * HA);
          h5::h5_write(gaps_grp, "vbm_k" + sfx, kv); h5::h5_write(gaps_grp, "cbm_k" + sfx, kcb);
        }
        h5::h5_write(gaps_grp, "epsilon_inf", eps_inf);
        h5::h5_write(gaps_grp, "mu_Ha", mu);
      }
      mpi->comm.barrier();
      if (qp_gaps_interp) {
        // P24 / G31: the same report on the Wannier-interpolated fine mesh (all ranks: the interpolation is collective)
        utils::check(qp_gaps_interp_mesh.size() == 3, "qp_gaps: qp_gaps_interp_mesh expects 3 integers (got {}).",
                     qp_gaps_interp_mesh.size());
        std::array<long, 3> mesh = {qp_gaps_interp_mesh[0], qp_gaps_interp_mesh[1], qp_gaps_interp_mesh[2]};
        auto kf_Ef = pp.qp_bands_on_mesh(*mf, wannier_file, mesh, grp_name, iteration, trans_home_cell);
        auto const& kf = std::get<0>(kf_Ef);   // (nk_fine, 3) crystal coordinates
        auto const& Ef = std::get<1>(kf_Ef);   // (ns, nk_fine, nbw) Ha
        mpi->comm.barrier();
        if (mpi->comm.root()) {
          const double HA = 27.211386245988;
          double mu = 0.0;
          h5::file file(scf_output+".mbpt.h5", 'a');
          auto qp_grp = h5::group(file).open_group(grp_name+"/iter"+std::to_string(iteration)+"/qp_approx");
          h5::h5_read(qp_grp, "mu", mu);
          auto gaps_grp = qp_grp.has_subgroup("gaps") ? qp_grp.open_group("gaps") : qp_grp.create_group("gaps");
          const long ns = Ef.shape(0), nkf = Ef.shape(1), nbw = Ef.shape(2);
          const long ik_gamma = 0;   // k index (i*n2 + j)*n3 + l: (0, 0, 0) is the first point
          app_log(1, "  [qp_gaps] {} iteration {}: Wannier-interpolated QP bands on the {}x{}x{} mesh ({} k-points x {} Wannier bands; occupation by mu = {:.6f} Ha)",
                  prefix, iteration, mesh[0], mesh[1], mesh[2], nkf, nbw, mu);
          for (long is = 0; is < ns; ++is) {
            double vbm = -1e9, cbm = 1e9, dmin = 1e9, dgam = -1.0;
            long kv = -1, kcb = -1, kd = -1, nocc_g = -1, n_bad = 0;
            for (long ik = 0; ik < nkf; ++ik) {
              double v = -1e9, c = 1e9; long nocc = 0;
              for (long a = 0; a < nbw; ++a) {
                const double e = Ef(is, ik, a);
                if (e < mu) { ++nocc; v = std::max(v, e); } else c = std::min(c, e);
              }
              if (nocc == 0 or nocc == nbw) ++n_bad;   // the window has no occupied / no empty band at this k
              if (v > vbm) { vbm = v; kv = ik; }
              if (c < cbm) { cbm = c; kcb = ik; }
              if (c - v < dmin) { dmin = c - v; kd = ik; }
              if (ik == ik_gamma) { dgam = c - v; nocc_g = nocc; }
            }
            if (n_bad > 0)
              app_warning("qp_gaps: spin {}: the Wannier window has no occupied or no empty band at {} of {} fine-mesh k-points; the interpolated gaps are not meaningful (widen the projector window).",
                          is, n_bad, nkf);
            auto kstr = [&kf](long ik) {
              return (ik < 0) ? std::string("n/a")
                              : std::format("{} = ({:.4f}, {:.4f}, {:.4f})", ik, kf(ik, 0), kf(ik, 1), kf(ik, 2));
            };
            const bool edges = (kv >= 0 and kcb >= 0 and vbm > -1e8 and cbm < 1e8);   // both band edges inside the window
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double fund = edges ? (cbm - vbm) * HA : nan, dmin_ev = edges ? dmin * HA : nan;
            if (edges)
              app_log(1, "  [qp_gaps]   spin {}: fundamental gap (interp) = {:.4f} eV  [VBM {:.4f} eV at k {} , CBM {:.4f} eV at k {}]; direct gap at Gamma = {} eV (k {}, {} occupied QP bands); smallest direct gap = {:.4f} eV at k {}",
                      is, fund, vbm * HA, kstr(kv), cbm * HA, kstr(kcb), (dgam > 0.0) ? std::to_string(dgam * HA) : std::string("n/a"), ik_gamma, nocc_g, dmin_ev, kstr(kd));
            else
              app_log(1, "  [qp_gaps]   spin {}: fundamental gap (interp) = n/a (the Wannier window holds no {} band; VBM {} eV, CBM {} eV; {} occupied QP bands at Gamma)",
                      is, (kcb < 0 or cbm >= 1e8) ? "empty" : "occupied", (kv >= 0 and vbm > -1e8) ? std::to_string(vbm * HA) : std::string("n/a"),
                      (kcb >= 0 and cbm < 1e8) ? std::to_string(cbm * HA) : std::string("n/a"), nocc_g);
            const std::string sfx = (ns > 1) ? "_s" + std::to_string(is) : "";
            h5::h5_write(gaps_grp, "fundamental_eV_interp" + sfx, fund);
            h5::h5_write(gaps_grp, "direct_gamma_eV_interp" + sfx, (dgam > 0.0) ? dgam * HA : nan);
            h5::h5_write(gaps_grp, "direct_min_eV_interp" + sfx, dmin_ev);
            h5::h5_write(gaps_grp, "vbm_eV_interp" + sfx, (kv >= 0 and vbm > -1e8) ? vbm * HA : nan);
            h5::h5_write(gaps_grp, "cbm_eV_interp" + sfx, (kcb >= 0 and cbm < 1e8) ? cbm * HA : nan);
            h5::h5_write(gaps_grp, "vbm_k_interp" + sfx, kv); h5::h5_write(gaps_grp, "cbm_k_interp" + sfx, kcb);
            h5::h5_write(gaps_grp, "direct_min_k_interp" + sfx, kd);
            nda::array<double, 1> kvec(3);
            kvec() = kf(std::max(kv, 0L), nda::range::all);  nda::h5_write(gaps_grp, "vbm_kpt_crys_interp" + sfx, kvec, false);
            kvec() = kf(std::max(kcb, 0L), nda::range::all); nda::h5_write(gaps_grp, "cbm_kpt_crys_interp" + sfx, kvec, false);
            kvec() = kf(std::max(kd, 0L), nda::range::all);  nda::h5_write(gaps_grp, "direct_min_kpt_crys_interp" + sfx, kvec, false);
          }
          nda::array<long, 1> mesh_arr(3);
          for (int d = 0; d < 3; ++d) mesh_arr(d) = mesh[d];
          nda::h5_write(gaps_grp, "interp_mesh", mesh_arr, false);
          nda::h5_write(gaps_grp, "kpts_crys_interp", kf, false);
          nda::h5_write(gaps_grp, "E_ska_interp", Ef, false);   // Ha, like qp_approx/E_ska
        }
        mpi->comm.barrier();
      }
    } else if (pp_type == "spectral_interpolation") {

      auto ft = imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false);

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", ft.nw_f()/2);
      if (Nfit <= 0 or Nfit > ft.nw_f()/2) Nfit = ft.nw_f()/2;
      // default w_min and w_max for +/- 10 eV
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-0.367);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",0.367);
      utils::check(w_min < w_max, "w_min should be smaller than w_max in spectral_interpolation.");
      // (w_max - w_min) / Nw = 0.05 => Nw = (w_max - w_min) / (0.05 / 27.2114079527)
      auto Nw      = io::get_value_with_default<int>(pt, "Nw", int((w_max - w_min) * 27.2114079527 / 0.05));
      utils::check(Nw > 1, "Nw must be > 1 in spectral_interpolation (got {}).", Nw);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermion, Nfit, eta, w_min, w_max, Nw);

      pp.spectral_interpolation(*mf, pt, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "local_dos") {

      auto ft = imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false);

      pproc_t pp(*mpi, prefix, outdir);
      // if wannier_file is not provided, calculate local DOS in the Bloch basis
      auto wannier_file = io::get_value_with_default<std::string>(pt, "wannier_file", "");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", ft.nw_f()/2);
      if (Nfit <= 0 or Nfit > ft.nw_f()/2) Nfit = ft.nw_f()/2;
      // default w_min and w_max for +/- 10 eV 
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-0.367);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",0.367);
      // default to 0.05 eV resolution => Nw = (w_max - w_min) / (0.05 / 27.2114079527)
      auto Nw      = io::get_value_with_default<int>(pt, "Nw", int((w_max - w_min) * 27.2114079527 / 0.05));
      utils::check(Nw > 1, "Nw must be > 1 in spectral_interpolation (got {}).", Nw);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermion, Nfit, eta, w_min, w_max, Nw);

      pp.local_density_of_state(*mf, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "cvv_eps") {

      // scGW-tilde increment C3: covariant-velocity dielectric readout on a stored
      // checkpoint (notes/scgwt_implementation_plan.md; the T-c probe).
      pproc_t pp(*mpi, prefix, outdir);
      auto grp_name  = io::get_value_with_default<std::string>(pt, "grp_name", "scf");
      auto iteration = io::get_value_with_default<long>(pt, "iteration", -1);
      pp.cvv_eps(*mf, pt, grp_name, iteration);

    } else if (pp_type == "dump_vxc") {

      std::string scf_output = outdir+"/"+prefix;
      hamilt::dump_vxc(*mpi, *mf, scf_output);

    } else if (pp_type == "dump_hartree") {

      auto scf_iter = io::get_value_with_default<int>(pt, "scf_iter", -1);

      std::string scf_output = outdir+"/"+prefix;
      auto psp = hamilt::make_pseudopot(*mf);

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(scf_iter==0 or scf_iter==-1,
                     "dump_hartree: scf_iter = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     scf_iter, scf_output);

        // dump mf data to "scf_output".mbpt.h5
        // 
        imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }

      hamilt::dump_hartree(*mpi, *mf, psp.get(), scf_output, scf_iter);

    } else {
      APP_ABORT("pproc: Unkonw post-processing type: {}", pp_type);
    }
  }

} // methods

#endif //COQUI_PPROC_DRIVERS_HPP
