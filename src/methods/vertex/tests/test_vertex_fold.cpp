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

/**
 * The star fold of the Sigma-side vertex as a term table (vertex_sym::fold_star_terms / apply_fold_terms) on the
 * symmetric unit-test mean fields, with an INDEX-ONLY sym context (vertex_sym::fill_index_maps; the C-sector rotations
 * are the 1 x 1 identity). The fold's index logic -- plain images, time-reversal images through Sigma_{-q}(p) =
 * conj Sigma_q(-p), the conjugated rotations at -k, the TRIM-point images and the -q PARTNER UNITS -- is checked against the
 * explicit star sum of a group-covariant scalar deposit
 *     F(q, k) = sum_R exp(2 pi i (q + k) . R) + 0.3 sum_R exp(2 pi i (2 q - k) . R)
 * over a star {R} of lattice vectors closed under the group: F(S q, S k) = F(q, k) and F(-q, -k) = conj F(q, k), which is
 * exactly the covariance the fold assumes of Sigma_q(k). Seconds to run.
 */

#include <cmath>
#include <complex>
#include <set>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/vertex/vertex_sym.hpp"

namespace bdft_tests {

  namespace vsym = methods::solvers::vertex_sym;
  using ComplexType = std::complex<double>;

  TEST_CASE("vertex_sym_fold_terms", "[methods][vertex][sym]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    std::vector<std::string> fxs = {"qe_si222_sym", "qe_si333_sym", "qe_si444_sym", "qe_lih222_sym", "qe_lih223_sym"};
    if (const char *f = std::getenv("COQUI_FOLD_TEST_FIXTURE")) fxs = {std::string(f)};
    for (auto const &fx : fxs) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
      vsym::sym_ctx c;
      vsym::fill_index_maps(c, *mf);
      c.active = true;
      c.ns = 1; c.nc = 1; c.naux = 0;
      c.Dc = nda::array<ComplexType, 5>(1, c.nsym, c.nk_full, 1, 1);
      c.Dc() = ComplexType(1.0);
      const long nk = c.nk_full, nq = c.nq_full;

      // crystal coordinates of k and of the transfers (q = k_0 - k_2(q, 0), as build_sym_ctx)
      auto kc = mf->kpts_crystal();
      nda::array<double, 2> qc(nq, 3);
      for (long iq = 0; iq < nq; ++iq) {
        const long k2 = mf->qk_to_k2(int(iq), 0);
        for (int a = 0; a < 3; ++a) qc(iq, a) = kc(0, a) - kc(k2, a);
      }
      // a star of lattice vectors closed under the group: the images R v of v0 = (1, 0, 0), (0, 1, 1) and their negatives.
      // The operations act on the reciprocal crystal row vectors as q -> q R (build_sym_ctx's direction check), so the
      // phase (q R) . v = q . (R v) is invariant when the star is closed under the COLUMN action v -> R v
      auto ops = mf->symm_list();
      auto qsymms = mf->qsymms();
      std::vector<std::array<double, 3>> star;
      auto add = [&](std::array<double, 3> v) {
        for (auto const &w : star) if (std::abs(w[0] - v[0]) + std::abs(w[1] - v[1]) + std::abs(w[2] - v[2]) < 1e-8) return;
        star.push_back(v);
      };
      add({1.0, 0.0, 0.0}); add({0.0, 1.0, 1.0});
      for (size_t n0 = 0; n0 != star.size();) {   // the orbit closure under the operations, their inverses and -1
        n0 = star.size();
        for (size_t r = 0; r < n0; ++r)
          for (long js = 0; js < c.nsym; ++js) {
            auto const &R = ops[size_t(qsymms(js))].R;
            std::array<double, 3> v{0.0, 0.0, 0.0}, vi{0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b) { v[a] += R(a, b) * star[r][size_t(b)]; vi[a] += ops[size_t(qsymms(js))].Rinv(a, b) * star[r][size_t(b)]; }
            add(v); add({-v[0], -v[1], -v[2]}); add(vi); add({-vi[0], -vi[1], -vi[2]});
          }
      }
      // closure check: every image of every star vector is in the star
      for (auto const &w : star)
        for (long js = 0; js < c.nsym; ++js) {
          auto const &R = ops[size_t(qsymms(js))].R;
          std::array<double, 3> v{0.0, 0.0, 0.0};
          for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) v[a] += R(a, b) * w[size_t(b)];
          bool found = false;
          for (auto const &x : star) if (std::abs(x[0] - v[0]) + std::abs(x[1] - v[1]) + std::abs(x[2] - v[2]) < 1e-8) found = true;
          REQUIRE(found);
        }
      auto F = [&](long iq, long ik) {
        ComplexType s(0.0);
        for (size_t r = 0; r < star.size(); ++r) {
          double p1 = 0.0, p2 = 0.0;
          for (int a = 0; a < 3; ++a) { p1 += (qc(iq, a) + kc(ik, a)) * star[r][size_t(a)]; p2 += (2.0 * qc(iq, a) - kc(ik, a)) * star[r][size_t(a)]; }
          s += std::exp(ComplexType(0.0, 2.0 * M_PI * p1)) + 0.3 * std::exp(ComplexType(0.0, 2.0 * M_PI * p2));
        }
        return s;
      };
      // the covariance of F itself (the premise of the fold): F(S q, S k) = F(q, k) for every image, F(-q, -k) = conj F(q, k)
      for (long iq = 0; iq < nq; ++iq)
        for (long ik = 0; ik < nk; ++ik) REQUIRE(std::abs(F(c.qminus(iq), c.kminus(ik)) - std::conj(F(iq, ik))) < 1e-10);
      for (long qp = 0; qp < nq; ++qp) {   // S_js q' = +-qs and krot(js, .) = the same operation on k
        const long js = c.q_isym(qp), qs = c.q_star(qp);
        if (js == 0) continue;
        for (long ik = 0; ik < nk; ++ik) {
          const long sk = c.krot(js, ik);
          const ComplexType ref = c.q_trev(qp) ? std::conj(F(qs, c.kminus(sk))) : F(qs, sk);
          REQUIRE(std::abs(F(qp, ik) - ref) < 1e-10);
        }
      }

      long n_terms = 0, n_partner = 0, n_trev = 0, n_cjg = 0, n_part_q = 0;
      double err_max = 0.0;
      for (long iq = 0; iq < c.nq_ibz; ++iq) {
        auto ft = vsym::fold_star_terms(c, iq, 1, 0);
        long star_size = 0;
        for (long qp = 0; qp < nq; ++qp) if (c.q_star(qp) == iq) ++star_size;
        REQUIRE(long(ft.terms.size() + ft.partner.size()) == star_size * c.nk_ibz);
        n_terms += long(ft.terms.size()); n_partner += long(ft.partner.size()); n_trev += ft.n_trev; n_cjg += ft.n_cjg;
        if (ft.needs_partner()) ++n_part_q;
        // every term reads an existing row, every IBZ external gets star_size terms
        std::vector<long> per_k(static_cast<size_t>(c.nk_ibz), 0);
        for (auto const *lst : {&ft.terms, &ft.partner})
          for (auto const &t : *lst) { REQUIRE(t.src >= 0); REQUIRE(t.src < nk); REQUIRE(t.k >= 0); REQUIRE(t.k < c.nk_ibz); ++per_k[size_t(t.k)]; }
        for (long k = 0; k < c.nk_ibz; ++k) REQUIRE(per_k[size_t(k)] == star_size);
        // the unit's scratch: Sigma_{qs}(p) = F(qs, p); the partner unit's: Sigma_{-qs}(p) = F(-qs, p)
        nda::array<ComplexType, 4> X(1, nk, 1, 1), Xm(1, nk, 1, 1), Y(1, c.nk_ibz, 1, 1);
        for (long p = 0; p < nk; ++p) { X(0, p, 0, 0) = F(iq, p); Xm(0, p, 0, 0) = F(c.qminus(iq), p); }
        Y() = ComplexType(0.0);
        vsym::apply_fold_terms(ft, X, Y);
        if (ft.needs_partner()) {
          vsym::fold_terms pt;
          pt.terms = ft.partner;
          vsym::apply_fold_terms(pt, Xm, Y);
        }
        // the explicit star sum
        for (long k = 0; k < c.nk_ibz; ++k) {
          ComplexType ref(0.0);
          for (long qp = 0; qp < nq; ++qp) if (c.q_star(qp) == iq) ref += F(qp, k);
          const double err = std::abs(Y(0, k, 0, 0) - ref);
          err_max = std::max(err_max, err);
          REQUIRE(err < 1e-9);
        }
      }
      app_log(1, "fold terms [{}]: k {} (IBZ {}), q {} (IBZ {}), {} ops, star {} vectors; terms {} (+ {} partner-unit terms "
                 "for {} IBZ transfers), time-reversal images {}, conjugated rotations {}; max |fold - star sum| = {:.3e}",
              fx, nk, c.nk_ibz, nq, c.nq_ibz, c.nsym, star.size(), n_terms, n_partner, n_part_q, n_trev, n_cjg, err_max);
    }
    mpi_context->comm.barrier();
  }

  // the UNFOLD of an IBZ object to the full mesh (vertex_sym::unfold_from_ibz), including the points reached by time reversal
  // alone: Y(k) = F(k) on the IBZ with F(k) = sum_R e^{2 pi i k . R} over a star closed under the operations but NOT under -1
  // (F complex), F(S k) = F(k), F(-k) = conj F(k); the unfold must return F at every full-mesh point
  TEST_CASE("vertex_sym_unfold_ibz", "[methods][vertex][sym]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    std::vector<std::string> fxs = {"qe_si222_sym", "qe_si333_sym", "qe_si444_sym", "qe_lih223_sym"};
    if (const char *f = std::getenv("COQUI_FOLD_TEST_FIXTURE")) fxs = {std::string(f)};
    for (auto const &fx : fxs) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
      vsym::sym_ctx c;
      vsym::fill_index_maps(c, *mf);
      c.active = true; c.ns = 1; c.nc = 1;
      c.Dc = nda::array<ComplexType, 5>(1, c.nsym, c.nk_full, 1, 1);
      c.Dc() = ComplexType(1.0);
      auto kc = mf->kpts_crystal();
      auto ops = mf->symm_list();
      auto qsymms = mf->qsymms();
      std::vector<std::array<double, 3>> star;
      auto add = [&](std::array<double, 3> v) {
        for (auto const &w : star) if (std::abs(w[0] - v[0]) + std::abs(w[1] - v[1]) + std::abs(w[2] - v[2]) < 1e-8) return;
        star.push_back(v);
      };
      add({1.0, 0.0, 0.0}); add({0.0, 1.0, 1.0}); add({1.0, 2.0, 0.0});
      for (size_t n0 = 0; n0 != star.size();) {   // closure under the operations and their inverses only
        n0 = star.size();
        for (size_t r = 0; r < n0; ++r)
          for (long js = 0; js < c.nsym; ++js) {
            std::array<double, 3> v{0.0, 0.0, 0.0}, vi{0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b) { v[a] += ops[size_t(qsymms(js))].R(a, b) * star[r][size_t(b)]; vi[a] += ops[size_t(qsymms(js))].Rinv(a, b) * star[r][size_t(b)]; }
            add(v); add(vi);
          }
      }
      auto F = [&](long ik) {
        ComplexType s(0.0);
        for (size_t r = 0; r < star.size(); ++r) {
          double ph = 0.0;
          for (int a = 0; a < 3; ++a) ph += kc(ik, a) * star[r][size_t(a)];
          s += std::exp(ComplexType(0.0, 2.0 * M_PI * ph));
        }
        return s;
      };
      nda::array<ComplexType, 4> Y(1, c.nk_ibz, 1, 1);
      for (long k = 0; k < c.nk_ibz; ++k) Y(0, k, 0, 0) = F(k);
      double cons = -1.0;
      auto X = vsym::unfold_from_ibz(c, Y, 0, &cons);
      long n_tr = 0;
      double err = 0.0, fim = 0.0;
      for (long p = 0; p < c.nk_full; ++p) {
        err = std::max(err, std::abs(X(0, p, 0, 0) - F(p)));
        fim = std::max(fim, std::abs(F(p).imag()));
        if (c.ktrev_pair(p) >= 0) ++n_tr;
      }
      app_log(1, "unfold [{}]: k {} (IBZ {}), {} time-reversal images, star {} vectors, max |Im F| {:.3e}: max |unfold - F| = {:.3e}, "
                 "little-group consistency {:.3e}", fx, c.nk_full, c.nk_ibz, n_tr, star.size(), fim, err, cons);
      REQUIRE(err < 1e-10);
      REQUIRE(cons < 1e-10);
    }
    mpi_context->comm.barrier();
  }

} // bdft_tests
