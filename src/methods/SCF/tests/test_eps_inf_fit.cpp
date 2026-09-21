/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * P25 / G32 (notes/vertex_perf_plan.md): the small-q fit of epsilon_inf, the pure
 * function alone (methods/scr_coulomb/eps_inf_fit.hpp). No fixture, no MPI: synthetic
 * eps(q) = eps_inf + A q^2 (+ B q^4) on a handful of |q| must be recovered to round-off,
 * q = 0 must be excluded, the point selection must be by ascending |q|, and the
 * degenerate inputs (one nonzero |q|, all-equal |q|) must report ok = false.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <vector>

#include "methods/scr_coulomb/eps_inf_fit.hpp"

namespace bdft_tests {

  using namespace methods::solvers;

  TEST_CASE("eps_inf_fit_quadratic", "[methods][scr_coulomb][eps_inf_fit]") {
    // eps(q) = 12 + 3 q^2 on an unsorted |q| list with a q = 0 entry (the stored head slot)
    const std::vector<double> q = {0.0, 0.3, 0.1, 0.2, 0.5, 0.4};
    std::vector<double> e;
    for (double v : q) e.push_back(12.0 + 3.0 * v * v);

    auto r = eps_fit::fit_eps_inf(q, e, 3);
    REQUIRE(r.ok);
    REQUIRE(r.degree == 1);
    REQUIRE(r.q_used.size() == 3);
    REQUIRE(r.coeffs.size() == 2);
    // the three smallest NONZERO |q|, ascending
    REQUIRE(r.q_used[0] == Approx(0.1).margin(1e-15));
    REQUIRE(r.q_used[1] == Approx(0.2).margin(1e-15));
    REQUIRE(r.q_used[2] == Approx(0.3).margin(1e-15));
    REQUIRE(std::abs(r.eps_inf - 12.0) < 1e-12);
    REQUIRE(std::abs(r.coeffs[1] - 3.0) < 1e-12);
    REQUIRE(r.residual < 1e-12);
    for (size_t i = 0; i < r.q_used.size(); ++i)
      REQUIRE(std::abs(eps_fit::eval_fit(r.coeffs, r.q_used[i]) - r.eps_used[i]) < 1e-12);

    // n_fit = 2 (the minimum) and n_fit beyond the available points (all 5 nonzero, degree 2)
    auto r2 = eps_fit::fit_eps_inf(q, e, 2);
    REQUIRE(r2.ok);
    REQUIRE(r2.q_used.size() == 2);
    REQUIRE(std::abs(r2.eps_inf - 12.0) < 1e-12);
    auto r5 = eps_fit::fit_eps_inf(q, e, 10);
    REQUIRE(r5.ok);
    REQUIRE(r5.q_used.size() == 5);
    REQUIRE(r5.degree == 2);
    REQUIRE(std::abs(r5.eps_inf - 12.0) < 1e-12);
    REQUIRE(std::abs(r5.coeffs[2]) < 1e-12);
  }

  TEST_CASE("eps_inf_fit_quartic", "[methods][scr_coulomb][eps_inf_fit]") {
    // eps(q) = 12 + 3 q^2 - 7 q^4: with 4 points the degree-2 fit in q^2 recovers all three
    // coefficients; with 3 points the degree-1 fit is biased and reports a nonzero residual.
    const std::vector<double> q = {0.0, 0.3, 0.1, 0.2, 0.5, 0.4};
    std::vector<double> e;
    for (double v : q) e.push_back(12.0 + 3.0 * v * v - 7.0 * v * v * v * v);

    auto r4 = eps_fit::fit_eps_inf(q, e, 4);
    REQUIRE(r4.ok);
    REQUIRE(r4.degree == 2);
    REQUIRE(r4.q_used.size() == 4);
    REQUIRE(r4.coeffs.size() == 3);
    REQUIRE(std::abs(r4.eps_inf - 12.0) < 1e-12);
    REQUIRE(std::abs(r4.coeffs[1] - 3.0) < 1e-11);
    REQUIRE(std::abs(r4.coeffs[2] + 7.0) < 1e-10);
    REQUIRE(r4.residual < 1e-12);

    auto r3 = eps_fit::fit_eps_inf(q, e, 3);
    REQUIRE(r3.ok);
    REQUIRE(r3.degree == 1);
    REQUIRE(r3.residual > 1e-4);                    // the q^4 term is unfit at degree 1
    REQUIRE(std::abs(r3.eps_inf - 12.0) > 1e-4);    // ... and biases the intercept
    REQUIRE(std::abs(r3.eps_inf - 12.0) < 0.1);     // ... by a small amount at these |q|

    // a production-scale check (Si-like |q| in bohr^-1, eps ~ 12): still round-off exact
    const std::vector<double> qs = {0.0, 0.2833, 0.4006, 0.4907, 0.5666};
    std::vector<double> es;
    for (double v : qs) es.push_back(11.9 + 2.5 * v * v + 0.8 * v * v * v * v);
    auto rs = eps_fit::fit_eps_inf(qs, es, 4);
    REQUIRE(rs.ok);
    REQUIRE(rs.degree == 2);
    REQUIRE(std::abs(rs.eps_inf - 11.9) < 1e-12);
    REQUIRE(std::abs(rs.coeffs[1] - 2.5) < 1e-11);
    REQUIRE(std::abs(rs.coeffs[2] - 0.8) < 1e-10);
  }

  TEST_CASE("eps_inf_fit_degenerate", "[methods][scr_coulomb][eps_inf_fit]") {
    // only q = 0 and one nonzero |q|: nothing to fit
    auto r1 = eps_fit::fit_eps_inf({0.0, 0.1}, {12.0, 12.03});
    REQUIRE_FALSE(r1.ok);
    // three nonzero points at the SAME |q|: the intercept is not separable from A
    auto r2 = eps_fit::fit_eps_inf({0.0, 0.1, 0.1, 0.1}, {12.0, 12.03, 12.03, 12.03}, 3);
    REQUIRE_FALSE(r2.ok);
    // tied |q| among 4 points: only 2 distinct |q|^2, the degree drops to 1 and the fit is exact
    auto r3 = eps_fit::fit_eps_inf({0.0, 0.1, 0.1, 0.2, 0.2}, {12.0, 12.03, 12.03, 12.12, 12.12}, 4);
    REQUIRE(r3.ok);
    REQUIRE(r3.degree == 1);
    REQUIRE(std::abs(r3.eps_inf - 12.0) < 1e-12);
    REQUIRE(std::abs(r3.coeffs[1] - 3.0) < 1e-12);
    // mismatched lengths: refused
    auto r4 = eps_fit::fit_eps_inf({0.0, 0.1, 0.2}, {12.0, 12.03});
    REQUIRE_FALSE(r4.ok);
    // the selection helper: q = 0 excluded, ascending, capped
    auto idx = eps_fit::smallest_nonzero_q({0.0, 0.3, 0.1, 0.2}, 2);
    REQUIRE(idx.size() == 2);
    REQUIRE(idx[0] == 2);
    REQUIRE(idx[1] == 3);
    REQUIRE(eps_fit::fit_degree(2) == 1);
    REQUIRE(eps_fit::fit_degree(3) == 1);
    REQUIRE(eps_fit::fit_degree(4) == 2);
  }

}  // bdft_tests
