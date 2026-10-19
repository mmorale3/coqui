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

#ifndef COQUI_PROJECTED_WANNIER_HPP
#define COQUI_PROJECTED_WANNIER_HPP

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/sparse/csr_blas.hpp"
#include "mean_field/MF.hpp"

#include "wannier/scdm.hpp"
#include "wannier/wan90_aux.hpp"

namespace wannier {
namespace detail {

/**
 * One trial orbital from its input string. Two forms (whitespace separated numbers):
 *   13 numbers: cx cy cz  l mr r  zx zy zz  xx xy xz  zona     (the wannier90 nnkp layout)
 *    7 numbers: cx cy cz  l mr r  zona                          (default axes z = (0,0,1), x = (1,0,0))
 * centre in crystal coordinates, (l, mr, r) in the wannier90 convention, zona in inverse bohr.
 */
inline projection parse_projection(std::string const& str) {
  std::vector<double> v;
  {
    std::istringstream in(str);
    double x;
    while (in >> x) v.emplace_back(x);
  }
  utils::check(v.size() == 13 or v.size() == 7,
               "wannier::projected_wannier: a projection needs 13 (cx cy cz l mr r zx zy zz xx xy xz zona) or 7 "
               "(cx cy cz l mr r zona) numbers, found {} in \"{}\".", v.size(), str);
  std::ostringstream full;
  full.precision(16);
  if (v.size() == 13) {
    for (auto x : v) full << x << " ";
  } else {
    full << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " " << v[5]
         << " 0 0 1 1 0 0 " << v[6];
  }
  std::istringstream in(full.str());
  return projection(in);
}

/**
 * U(k) = A_w(k) [A_w(k)^dag A_w(k)]^{-1/2} with A_w = diag(f(e_nk)) A(k): Loewdin orthonormalisation of the
 * energy-weighted projections. Returns the smallest eigenvalue of the Gram matrix A_w^dag A_w (the rank meter).
 */
inline double loewdin_weighted(nda::MemoryArrayOfRank<2> auto const& A,        // (nband, nwann)
                               nda::MemoryArrayOfRank<1> auto const& f,        // (nband)
                               nda::MemoryArrayOfRank<2> auto && U)            // (nband, nwann)
{
  const long nband = A.shape(0), nwann = A.shape(1);
  nda::matrix<ComplexType> Aw(nband, nwann), s(nwann, nwann), sih(nwann, nwann), tmp(nwann, nwann);
  for (long n = 0; n < nband; ++n)
    for (long m = 0; m < nwann; ++m) Aw(n, m) = f(n) * A(n, m);
  nda::blas::gemm(ComplexType(1.0), nda::dagger(Aw), Aw, ComplexType(0.0), s);
  auto [lam, V] = nda::linalg::eigenelements(s);
  utils::check(lam(0) > 1e-10,
               "wannier::projected_wannier: the weighted projection matrix is rank deficient (smallest Gram eigenvalue {}): "
               "the trial orbitals are linearly dependent on the weighted band window.", lam(0));
  for (long a = 0; a < nwann; ++a)
    for (long b = 0; b < nwann; ++b) tmp(a, b) = V(a, b) / std::sqrt(lam(b));
  nda::blas::gemm(ComplexType(1.0), tmp, nda::dagger(V), ComplexType(0.0), sih);
  nda::blas::gemm(ComplexType(1.0), Aw, sih, ComplexType(0.0), U);
  return lam(0);
}

/**
 * The same with a FROZEN set of bands kept exactly in the subspace: range(U) = span{frozen bands} + the weighted
 * projections, on the other bands, of the nwann - nF combinations of the trial orbitals that carry NO weight on the
 * frozen bands. Those combinations are the null space of the nwann x nwann Gram matrix of the trial orbitals on the
 * frozen bands (rank nF), so their number is fixed by construction; they are the part of the trial set that the frozen
 * bands do not already hold. The gauge is then fixed by projecting the trial orbitals on the subspace,
 * U = P A (A^dag P A)^{-1/2}, so the columns follow the trial orbitals as in the unfrozen case.
 * Returns {smallest non-zero eigenvalue of the frozen-band Gram matrix (how well the trial set represents the frozen
 * bands), smallest eigenvalue of the Gram matrix of the complement directions}; gram_gauge = smallest eigenvalue of
 * A^dag P A.
 */
inline std::pair<double, double> loewdin_weighted_frozen(nda::MemoryArrayOfRank<2> auto const& A,     // (nband, nwann)
                                                         nda::MemoryArrayOfRank<1> auto const& f,     // (nband)
                                                         nda::range frozen,                            // rows kept exactly
                                                         nda::MemoryArrayOfRank<2> auto && U,          // (nband, nwann)
                                                         double &gram_gauge)
{
  const long nband = A.shape(0), nwann = A.shape(1), nF = frozen.size(), nC = nwann - nF;
  // the combinations of the trial orbitals without weight on the frozen bands
  nda::matrix<ComplexType> AF(nF, nwann), gF(nwann, nwann);
  for (long j = 0; j < nF; ++j)
    for (long m = 0; m < nwann; ++m) AF(j, m) = A(frozen.first() + j, m);
  nda::blas::gemm(ComplexType(1.0), nda::dagger(AF), AF, ComplexType(0.0), gF);
  auto [lamF, VF] = nda::linalg::eigenelements(gF);              // ascending: the first nC are the null space
  const double frz_rep = lamF(nC);   // nC <= nwann - 1 or nC = 0 (frozen set = subspace)
  utils::check(frz_rep > 1e-8,
               "wannier::projected_wannier: the trial orbitals do not span the frozen bands (eigenvalue {} of their Gram matrix "
               "on the frozen bands): the frozen set cannot be separated from the rest of the trial set.", frz_rep);
  nda::matrix<ComplexType> Q(nband, nwann);
  Q() = ComplexType(0.0);
  for (long j = 0; j < nF; ++j) Q(frozen.first() + j, j) = ComplexType(1.0);
  double cgram = 1.0;   // no complement directions when the frozen set fills the subspace
  if (nC > 0) {
  // their weighted projections on the other bands, orthonormalised
  nda::matrix<ComplexType> Aw(nband, nwann), N(nwann, nC), Qc(nband, nC), s(nC, nC), sih(nC, nC), tmp(nC, nC);
  for (long n = 0; n < nband; ++n)
    for (long m = 0; m < nwann; ++m) Aw(n, m) = (n >= frozen.first() and n < frozen.last()) ? ComplexType(0.0) : f(n) * A(n, m);
  for (long a = 0; a < nwann; ++a)
    for (long j = 0; j < nC; ++j) N(a, j) = VF(a, j);
  nda::blas::gemm(ComplexType(1.0), Aw, N, ComplexType(0.0), Qc);
  nda::blas::gemm(ComplexType(1.0), nda::dagger(Qc), Qc, ComplexType(0.0), s);
  auto [lam, V] = nda::linalg::eigenelements(s);
  utils::check(lam(0) > 1e-10,
               "wannier::projected_wannier: outside the frozen bands the weighted projections of the trial orbitals span fewer "
               "than {} directions (eigenvalue {}): reduce the number of projections or widen the energy weight.", nC, lam(0));
  for (long a = 0; a < nC; ++a)
    for (long b = 0; b < nC; ++b) tmp(a, b) = V(a, b) / std::sqrt(lam(b));
  nda::blas::gemm(ComplexType(1.0), tmp, nda::dagger(V), ComplexType(0.0), sih);
  nda::matrix<ComplexType> Qo(nband, nC);
  nda::blas::gemm(ComplexType(1.0), Qc, sih, ComplexType(0.0), Qo);
  for (long n = 0; n < nband; ++n)
    for (long j = 0; j < nC; ++j) Q(n, nF + j) = Qo(n, j);
  cgram = lam(0);
  }
  // the gauge of the trial orbitals: B = Q^dag A (nwann x nwann), U = Q B (B^dag B)^{-1/2}
  nda::matrix<ComplexType> Ac(nband, nwann), B(nwann, nwann), g(nwann, nwann), gih(nwann, nwann), t2(nwann, nwann), QB(nband, nwann);
  Ac() = A;
  nda::blas::gemm(ComplexType(1.0), nda::dagger(Q), Ac, ComplexType(0.0), B);
  nda::blas::gemm(ComplexType(1.0), nda::dagger(B), B, ComplexType(0.0), g);
  auto [mu, W] = nda::linalg::eigenelements(g);
  gram_gauge = mu(0);
  utils::check(mu(0) > 1e-10,
               "wannier::projected_wannier: the trial orbitals are linearly dependent on the selected subspace (smallest Gram "
               "eigenvalue {}).", mu(0));
  for (long a = 0; a < nwann; ++a)
    for (long b = 0; b < nwann; ++b) t2(a, b) = W(a, b) / std::sqrt(mu(b));
  nda::blas::gemm(ComplexType(1.0), t2, nda::dagger(W), ComplexType(0.0), gih);
  nda::blas::gemm(ComplexType(1.0), Q, B, ComplexType(0.0), QB);
  nda::blas::gemm(ComplexType(1.0), QB, gih, ComplexType(0.0), U);
  return {frz_rep, cgram};
}

/**
 * Symmetry covariance of a projector on a symmetry-reduced mesh. For every symmetry position js >= 1 of the mean field
 * and every k of the full mesh, the rotation D(js, k) of MF::symmetry_rotation carries the bands at k to the bands at
 * ksrc; the subspace is covariant when d = U(ksrc)^dag D U(k) is unitary (the rotated subspace at k is the subspace at
 * ksrc). Returns {max leakage 1 - ||d||_F^2 / ||D U(k)||_F^2, max ||d^dag d - 1||_F}.
 */
inline std::pair<double, double> projector_covariance(mf::MF &mf, nda::range b_rng,
                                                      nda::MemoryArrayOfRank<4> auto const& U_skia)   // (ns, nk, nband, nwann)
{
  decltype(nda::range::all) all;
  const long nk = mf.nkpts(), nbnd = mf.nbnd();
  const long ns = U_skia.shape(0), nW = U_skia.shape(2), M = U_skia.shape(3);
  const long nsym = mf.qsymms().extent(0);
  auto kp_trev_pair = mf.kp_trev_pair();
  double leak_max = 0.0, unit_max = 0.0;
  nda::array<ComplexType, 2> E(nbnd, nW), Dcols(nbnd, nW), DU(nbnd, M), DUw(nW, M), d(M, M), dd(M, M);
  E() = ComplexType(0.0);
  for (long j = 0; j < nW; ++j) E(b_rng.first() + j, j) = ComplexType(1.0);
  for (long js = 1; js < nsym; ++js)
    for (long ik = 0; ik < nk; ++ik) {
      auto [cj, Dsp] = mf.symmetry_rotation(js, ik);
      math::sparse::csrmm<'N'>(ComplexType(1.0), *Dsp, E, ComplexType(0.0), Dcols);
      const long ksrc = mf.ks_to_k(int(js), int(cj ? long(kp_trev_pair(ik)) : ik));
      for (long is = 0; is < ns; ++is) {
        if (cj) {
          auto Uc = nda::make_regular(nda::conj(U_skia(is, ik, all, all)));
          nda::blas::gemm(Dcols, Uc, DU);
        } else {
          nda::blas::gemm(Dcols, U_skia(is, ik, all, all), DU);
        }
        for (long p = 0; p < nW; ++p)
          for (long a = 0; a < M; ++a) DUw(p, a) = DU(b_rng.first() + p, a);
        nda::blas::gemm(nda::dagger(U_skia(is, ksrc, all, all)), DUw, d);
        double m_all = 0.0, m_in = 0.0;
        for (long a = 0; a < M; ++a) {
          for (long p = 0; p < nbnd; ++p) m_all += std::norm(DU(p, a));
          for (long b = 0; b < M; ++b) m_in += std::norm(d(b, a));
        }
        if (m_all > 1e-24) leak_max = std::max(leak_max, std::max(0.0, 1.0 - m_in / m_all));
        nda::blas::gemm(nda::dagger(d), d, dd);
        double u2 = 0.0;
        for (long a = 0; a < M; ++a)
          for (long b = 0; b < M; ++b) u2 += std::norm(dd(a, b) - ((a == b) ? ComplexType(1.0) : ComplexType(0.0)));
        unit_max = std::max(unit_max, std::sqrt(u2));
      }
    }
  return {leak_max, unit_max};
}

/**
 * Projection-only Wannier functions: implementation of wannier::projected_wannier (see wan90.h).
 */
inline void projected_wannier_impl(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, ptree &pt)
{
  decltype(nda::range::all) all;
  const double ha_to_ev = 1.0 / 3.674932540e-2;
  const double bohr_to_ang = 0.529177249;

  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto h5_fname = io::get_value_with_default<std::string>(pt, "h5_filename", outdir + "/" + prefix + ".mlwf.h5");
  auto brange = io::get_array<int>(pt, "band_range", "wannier::projected_wannier: band_range = [first, last) is required.");
  auto proj_str = io::get_array<std::string>(pt, "projections", "wannier::projected_wannier: projections is required.");
  auto weight = io::get_value_with_default<std::string>(pt, "weight", "erfc");
  auto units = io::get_value_with_default<std::string>(pt, "units", "ev");
  auto cov_tol = io::get_value_with_default<double>(pt, "covariance_tol", 1e-6);
  auto frz = io::get_array_with_default<int>(pt, "frozen_range", std::vector<int>{});
  // frozen_energy_max (units of mu): every band below it is also kept exactly, per k (the inner window of a
  // disentanglement); the frozen set at k is [frozen_range.first, max(frozen_range.last, bands below the energy))
  const bool frz_e = pt.get_optional<double>("frozen_energy_max").has_value();
  const double frz_emax = io::get_value_with_default<double>(pt, "frozen_energy_max", 0.0) / (units == "ev" ? ha_to_ev : 1.0);

  utils::check(mf.nspin() == 1 and mf.npol() == 1, "wannier::projected_wannier: nspin > 1 or npol > 1 not implemented.");
  utils::check(brange.size() == 2 and brange[0] >= 0 and brange[1] > brange[0] and brange[1] <= mf.nbnd(),
               "wannier::projected_wannier: invalid band_range (expect [first, last) with 0 <= first < last <= nbnd = {}).",
               mf.nbnd());
  const nda::range b_rng(brange[0], brange[1]);
  const long nband = b_rng.size(), nkpts = mf.nkpts(), nspin = mf.nspin();
  nda::array<int, 1> band_list(nband);
  for (long i = 0; i < nband; ++i) band_list(i) = int(b_rng.first() + i);

  utils::check(frz.empty() or (frz.size() == 2 and frz[0] >= brange[0] and frz[1] > frz[0] and frz[1] <= brange[1]),
               "wannier::projected_wannier: invalid frozen_range (expect [first, last) inside band_range).");
  const nda::range frozen = frz.empty() ? nda::range(0, 0) : nda::range(frz[0] - brange[0], frz[1] - brange[0]);   // rows of the band range

  std::vector<projection> proj;
  for (auto const& s : proj_str) proj.emplace_back(parse_projection(s));
  const long nwann = proj.size();
  utils::check(nwann > 0 and nwann <= nband,
               "wannier::projected_wannier: {} projections for {} bands (need 0 < num_wann <= number of bands).", nwann, nband);
  utils::check(not frz_e or frozen.size() > 0,
               "wannier::projected_wannier: frozen_energy_max needs frozen_range (its first band starts the frozen set).");
  utils::check(long(frozen.size()) < nwann or frozen.size() == 0,
               "wannier::projected_wannier: frozen_range holds {} bands, not fewer than the {} projections.", frozen.size(), nwann);

  // the energy weight: the SCDM weight function on the mean-field eigenvalues (absolute, same units convention)
  ptree wpt;
  wpt.put("scdm_entanglement", weight == "none" ? std::string("isolated") : weight);
  wpt.put("scdm_units", units);
  if (weight != "none") {
    wpt.put("scdm_mu", io::get_value<double>(pt, "mu", "wannier::projected_wannier: mu is required unless weight = \"none\"."));
    wpt.put("scdm_sigma", io::get_value<double>(pt, "sigma", "wannier::projected_wannier: sigma is required unless weight = \"none\"."));
  }
  auto fw = scdm_detail::weight_function(wpt);

  app_log(1, "  Projection-only Wannier functions (energy-weighted projection + Loewdin, no localisation step)");
  app_log(1, "  ------------------------------------------------------------------");
  app_log(1, "  number of Wannier functions = {}", nwann);
  app_log(1, "  band range                  = [{}, {})", b_rng.first(), b_rng.last());
  app_log(1, "  k-points (full mesh / IBZ)  = {} / {}", nkpts, mf.nkpts_ibz());
  if (weight == "none")
    app_log(1, "  energy weight               = none");
  else
    app_log(1, "  energy weight               = {} (mu = {}, sigma = {}, units = {})", weight,
            io::get_value<double>(pt, "mu"), io::get_value<double>(pt, "sigma"), units);
  if (frozen.size() > 0)
    app_log(1, "  frozen bands (kept exactly) = [{}, {})", frz[0], frz[1]);
  if (frz_e)
    app_log(1, "  frozen energy window        = every band below {} {} is also kept exactly (per k)",
            io::get_value<double>(pt, "frozen_energy_max"), units);
  app_log(1, "  output                      = {}\n", h5_fname);

  // A(k, n, m) = <psi_nk | g_m> at every k of the full mesh, from the mean field's own orbitals at the image k-points
  auto kp_map = nda::arange<int>(nkpts);
  nda::array<double, 2> wann_kp = mf.kpts_crystal();
  auto Amn = compute_amn_projections(mpi, mf, pt, kp_map, wann_kp, band_list, proj, false, false);

  if (not mpi.comm.root()) { mpi.comm.barrier(); return; }
  auto A = Amn.local();
  auto eig = mf.eigval();

  nda::array<ComplexType, 4> U(nspin, nkpts, nband, nwann);
  nda::array<double, 1> f(nband);
  double gram_min = 1e300, wtop_max = 0.0, kept_min = 1e300, ratio_max = 1e300, frz_gap = 1e300;
  long k_min = 0, k_kept = 0, k_rep = 0, nF_min = nband, nF_max = 0;
  auto ek = [&](long ik, long n) { return eig(0, ik, b_rng.first() + n); };
  for (long ik = 0; ik < nkpts; ++ik) {
    for (long n = 0; n < nband; ++n) f(n) = fw(ek(ik, n));
    wtop_max = std::max(wtop_max, f(nband - 1));
    double g = 0.0;
    if (frozen.size() > 0) {
      long last = frozen.last();
      if (frz_e)
        while (last < nband and ek(ik, last) < frz_emax) ++last;
      utils::check(last - frozen.first() <= nwann,
                   "wannier::projected_wannier: frozen_energy_max = {} eV holds {} bands at k-point {} ({} projections): lower it "
                   "below {} eV.", frz_emax * ha_to_ev, last - frozen.first(), ik, nwann,
                   ek(ik, std::min(frozen.first() + nwann, nband - 1)) * ha_to_ev);
      nda::range frz_k(frozen.first(), last);
      nF_min = std::min(nF_min, last - frozen.first());
      nF_max = std::max(nF_max, last - frozen.first());
      auto [frz_rep, cgram] = loewdin_weighted_frozen(A(ik, all, all), f, frz_k, U(0, ik, all, all), g);
      if (cgram < kept_min) { kept_min = cgram; k_kept = ik; }
      if (frz_rep < ratio_max) { ratio_max = frz_rep; k_rep = ik; }
      // the frozen set must be separated from its neighbours in energy at every k (it must not cut a multiplet)
      if (frz_k.first() > 0) frz_gap = std::min(frz_gap, ek(ik, frz_k.first()) - ek(ik, frz_k.first() - 1));
      if (frz_k.last() < nband) frz_gap = std::min(frz_gap, ek(ik, frz_k.last()) - ek(ik, frz_k.last() - 1));
      app_log(3, "  k {:4d} ({:7.4f} {:7.4f} {:7.4f}): frozen {:2d}, frozen Gram {:.3e}, complement Gram {:.3e}, gauge Gram {:.3e}",
              ik, wann_kp(ik, 0), wann_kp(ik, 1), wann_kp(ik, 2), last - frozen.first(), frz_rep, cgram, g);
    } else {
      g = loewdin_weighted(A(ik, all, all), f, U(0, ik, all, all));
    }
    if (g < gram_min) { gram_min = g; k_min = ik; }
  }
  if (frozen.size() > 0)
    app_log(1, "  frozen set per k: {} to {} bands; smallest complement Gram at k-point {} ({:.4f} {:.4f} {:.4f}), smallest "
               "frozen Gram at k-point {}", nF_min, nF_max, k_kept, wann_kp(k_kept, 0), wann_kp(k_kept, 1), wann_kp(k_kept, 2), k_rep);
  if (frozen.size() > 0) {
    app_log(1, "  frozen bands: smallest energy gap to the neighbouring bands over k = {:.4f} eV", frz_gap * ha_to_ev);
    app_log(1, "  frozen bands: smallest Gram eigenvalue of the trial orbitals on them over k = {:.4e}; other bands: smallest Gram "
               "eigenvalue of the complement directions over k = {:.4e}", ratio_max, kept_min);
    utils::check(frz_gap * ha_to_ev > 1e-3,
                 "wannier::projected_wannier: the frozen bands are not separated from their neighbours at every k (smallest gap "
                 "{} eV): frozen_range cuts a (near-)degenerate set and the subspace is not well defined there.", frz_gap * ha_to_ev);
  }
  app_log(1, "  smallest Gram eigenvalue of the {} over k = {:.4e} (k-point {})",
          frozen.size() > 0 ? "trial orbitals on the selected subspace" : "weighted projections", gram_min, k_min);
  app_log(1, "  largest weight of the top band of the range over k         = {:.4e}", wtop_max);
  if (wtop_max > 1e-3)
    app_log(1, "  [WARNING] the top band of band_range carries a weight above 1e-3: the upper edge of the range is not\n"
               "            switched off by the energy weight, so the subspace depends on where the range cuts the bands.");

  // band-structure fidelity: eigenvalues of H_W(k) = U^dag diag(e) U against the lowest band energies of the range
  {
    nda::array<double, 1> dev(nwann);
    dev() = 0.0;
    nda::matrix<ComplexType> eU(nband, nwann), Hw(nwann, nwann);
    for (long ik = 0; ik < nkpts; ++ik) {
      for (long n = 0; n < nband; ++n)
        for (long m = 0; m < nwann; ++m) eU(n, m) = eig(0, ik, b_rng.first() + n) * U(0, ik, n, m);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(U(0, ik, all, all)), eU, ComplexType(0.0), Hw);
      auto lam = nda::linalg::eigenvalues(Hw);
      for (long m = 0; m < nwann; ++m)
        dev(m) = std::max(dev(m), std::abs(lam(m) - eig(0, ik, b_rng.first() + m)));
    }
    app_log(1, "  max over k of |eig_j(U^dag e U) - e_j| (eV), j = 0..{}:", nwann - 1);
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "   ";
    for (long m = 0; m < nwann; ++m) line << " " << dev(m) * ha_to_ev;
    app_log(1, "{}\n", line.str());
  }

  // symmetry covariance (symmetry-reduced meshes only)
  const bool sym_mesh = (mf.nkpts() != mf.nkpts_ibz()) or (mf.nqpts() != mf.nqpts_ibz());
  if (sym_mesh) {
    auto [leak, unit] = projector_covariance(mf, b_rng, U);
    app_log(1, "  symmetry covariance of the projector ({} symmetry positions): max leakage = {:.3e}, max unitarity defect "
               "||d^dag d - 1||_F = {:.3e}\n", mf.qsymms().extent(0), leak, unit);
    utils::check(cov_tol <= 0.0 or (leak <= cov_tol and unit <= cov_tol),
                 "wannier::projected_wannier: the projector is not symmetry covariant within covariance_tol = {} (leakage {}, "
                 "unitarity defect {}). The trial orbitals must form a set closed under the symmetry operations of the mean "
                 "field, and band_range / the energy weight must not cut a degenerate multiplet. Set covariance_tol <= 0 to "
                 "write the file anyway.", cov_tol, leak, unit);
  } else {
    app_log(1, "  mesh without symmetry reduction: no covariance check.\n");
  }

  // h5 output in the wannier90 converter layout: Pkwa(s, k, w, b) = U(k)(b, w); energies in eV; centres in Angstrom
  nda::array<ComplexType, 4> Pkwa(nspin, nkpts, nwann, nband);
  for (long ik = 0; ik < nkpts; ++ik) Pkwa(0, ik, all, all) = nda::transpose(U(0, ik, all, all));
  nda::array<double, 3> eigv(nspin, nkpts, nband), R_wan(nspin, nwann, 3);
  for (long ik = 0; ik < nkpts; ++ik)
    for (long n = 0; n < nband; ++n) eigv(0, ik, n) = eig(0, ik, b_rng.first() + n) * ha_to_ev;
  {
    auto lattv = mf.lattv();
    nda::stack_array<double, 3> Rn = {0.0, 0.0, 0.0};
    for (long m = 0; m < nwann; ++m) {
      nda::blas::gemv(1.0, nda::transpose(lattv), proj[m].center, 0.0, Rn);
      for (int i = 0; i < 3; ++i) R_wan(0, m, i) = Rn(i) * bohr_to_ang;
    }
  }
  if (std::filesystem::exists(h5_fname)) {
    app_log(1, "  [WARNING] {} exists and is replaced.", h5_fname);
    std::filesystem::remove(h5_fname);
  }
  ptree opt = pt;
  opt.put("h5_filename", h5_fname);
  write_wan90_h5(mf, opt, band_list, eigv, Pkwa, R_wan);
  mpi.comm.barrier();
}

} // detail
} // wannier

#endif
