/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * The causal projection of Sigma_c (methods/SCF/sigma_causal_projection.hpp) on synthetic input. No fixture, no MPI.
 * -Sigma(tau) = sum_p K(tau, e_p) R_p with PSD residues R_p (a dense continuum of real poles) is causal: the projection must leave it
 * unchanged up to ADMM's floor. Adding a NEGATIVE spectral weight (-eta K(tau, e_q) P, P PSD) makes it non-causal: the
 * projection must return a causal Sigma (lambda_min(-Sigma) >= 0 on the mesh), remove no more than the perturbation, and
 * land within the perturbation's norm of the causal original (the projection onto a convex set is non-expansive). A second
 * call warm-started from the first stays causal and close to the first result. "sigma_causal_projection_convergence": the
 * production mesh with a nearly causal Sigma whose exact projection is known; the solver's error at the default iteration
 * budget against plain over-relaxed ADMM's. Hidden: "[.bench]" (time per iteration at nb 60) and "[.conv]" (convergence harness,
 * options from the environment, listed at the case). CUDA builds: "sigma_causal_projection_device" (the device PSD clip and
 * the batched device solver against the host path, nb 8 and 60 on the production mesh) and the hidden "[.bench_device]".
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/sigma_causal_projection.hpp"
#include "utilities/omp_threads.hpp"

namespace bdft_tests {

  namespace {
    // e^{-w tau} / (1 + e^{-beta w}) without overflow
    double kern(double tau, double w, double beta) {
      return (w >= 0.0) ? std::exp(-w * tau) / (1.0 + std::exp(-beta * w)) : std::exp(w * (beta - tau)) / (1.0 + std::exp(beta * w));
    }
    // a PSD nb x nb matrix of rank r from a deterministic pseudo-random sequence
    nda::matrix<ComplexType> psd_matrix(long nb, long r, long seed) {
      nda::matrix<ComplexType> V(nb, r), R(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long a = 0; a < r; ++a)
          V(i, a) = ComplexType(std::sin(1.3 * double(seed + 7 * i + 3 * a) + 0.2), std::cos(0.7 * double(seed + 5 * i + 11 * a)));
      R() = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          for (long a = 0; a < r; ++a) R(i, j) += V(i, a) * std::conj(V(j, a));
      return R;
    }
    double frob3(nda::array<ComplexType, 3> const &A) {
      double s = 0.0;
      for (auto const &v : A) s += std::norm(v);
      return std::sqrt(s);
    }
    double lam_min_neg(nda::array<ComplexType, 3> const &S) {     // min over tau of lambda_min(-S^H)
      const long nt = S.shape(0), nb = S.shape(1);
      double lm = 1e300;
      nda::matrix<ComplexType> A(nb, nb);
      for (long t = 0; t < nt; ++t) {
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) A(i, j) = -0.5 * (S(t, i, j) + std::conj(S(t, j, i)));
        auto ev = nda::linalg::eigenvalues(A);
        for (auto const &v : ev) lm = std::min(lm, double(std::real(v)));
      }
      return lm;
    }
  }

  TEST_CASE("sigma_causal_projection", "[methods][scf][causal_proj]") {
    const double beta = 100.0, wmax = 10.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0), nb = 6;
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);

    // a causal Sigma_c with a dense, broad spectral weight (as a lattice Sigma_c has): 2 x 120 poles in [0.25, 5] Ha on both
    // sides of the chemical potential, Gaussian envelope, PSD residues of rank 1..3 on the first nb0 bands (the last two bands
    // carry no causal weight, so the perturbation below is non-causal at every tau). A few isolated SHARP poles between the
    // projector's nodes are not representable by PSD weights on them to better than ~1e-3: not the case tested here.
    const long nb0 = 4;
    nda::array<ComplexType, 3> S0(nt, nb, nb);
    S0() = 0.0;
    for (long p = 0; p < 240; ++p) {
      const double e = (p < 120) ? -(0.3 + 4.7 * double(p) / 119.0) : 0.25 + 4.75 * double(p - 120) / 119.0;
      const double wgt = 0.1 * std::exp(-(std::abs(e) - 1.5) * (std::abs(e) - 1.5)) / 20.0;
      auto R0 = psd_matrix(nb0, 1 + p % 3, 17 * p + 1);
      nda::matrix<ComplexType> R(nb, nb);
      R() = 0.0;
      R(nda::range(0, nb0), nda::range(0, nb0)) = R0;
      for (long t = 0; t < nt; ++t) S0(t, nda::range::all, nda::range::all) -= wgt * kern(tau(t), e, beta) * R;
    }
    REQUIRE(lam_min_neg(S0) >= -1e-14);

    methods::causal_proj::options o;
    o.maxit = 20000;
    o.tol = 1e-12;
    methods::causal_proj::projector P(beta, x, o);
    REQUIRE(P.n_nodes() > 10);
    REQUIRE(P.n_nodes() <= nt);

    SECTION("the PSD clip: a PSD matrix is kept, an indefinite one loses exactly its negative part") {
      auto A = psd_matrix(nb, 3, 5);
      nda::matrix<ComplexType> X(2, nb * nb);
      // row 0: PSD (rank 3); row 1: A - 2 v v^dag with v = e_0 + i e_1 (indefinite)
      nda::matrix<ComplexType> B = A;
      B(0, 0) -= 2.0; B(1, 1) -= 2.0; B(0, 1) -= ComplexType(0.0, -2.0); B(1, 0) -= ComplexType(0.0, 2.0);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) { X(0, i * nb + j) = A(i, j); X(1, i * nb + j) = B(i, j); }
      methods::causal_proj::projector::psd_rows(X, nb);
      double e0 = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) e0 = std::max(e0, std::abs(X(0, i * nb + j) - A(i, j)));
      REQUIRE(e0 < 1e-12);
      nda::matrix<ComplexType> Bp(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Bp(i, j) = X(1, i * nb + j);
      auto ev = nda::linalg::eigenvalues(Bp);
      for (auto const &v : ev) REQUIRE(std::real(v) > -1e-12);
      // the clip is the nearest PSD matrix: B - Bp is negative semidefinite and orthogonal to Bp
      nda::matrix<ComplexType> N = B - Bp;
      auto evn = nda::linalg::eigenvalues(N);
      for (auto const &v : evn) REQUIRE(std::real(v) < 1e-12);
      ComplexType ip = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) ip += std::conj(N(i, j)) * Bp(i, j);
      REQUIRE(std::abs(ip) < 1e-10);
    }

    SECTION("causal input is left unchanged (to the ADMM floor)") {
      nda::array<ComplexType, 3> S = S0;
      nda::matrix<ComplexType> C, U;
      auto st = P.project(S, C, U);
      const double rel = frob3(nda::array<ComplexType, 3>(S - S0)) / frob3(S0);
      INFO("removed " << st.removed / st.y_norm << ", change " << rel << ", iterations " << st.iters);
      REQUIRE(rel < 5e-7);                     // plain over-relaxed ADMM (rho 1e-6, no acceleration): 7.4e-7
      REQUIRE(st.lam_after >= -1e-12 * st.y_norm);
    }

    SECTION("a negative spectral weight is removed; the result is causal") {
      const double eta = 2e-3;
      nda::matrix<ComplexType> Pm(nb, nb);
      Pm() = 0.0;
      Pm(nda::range(nb0, nb), nda::range(nb0, nb)) = psd_matrix(nb - nb0, 1, 99);
      nda::array<ComplexType, 3> dS(nt, nb, nb);
      for (long t = 0; t < nt; ++t) dS(t, nda::range::all, nda::range::all) = eta * kern(tau(t), 0.55, beta) * Pm;   // -(-eta) K P
      nda::array<ComplexType, 3> S = S0 + dS;
      REQUIRE(lam_min_neg(S) < -1e-6);
      const double npert = frob3(dS);
      nda::matrix<ComplexType> C, U;
      auto st = P.project(S, C, U);
      const double dist = frob3(nda::array<ComplexType, 3>(S - S0));
      INFO("lambda_min " << st.lam_before << " -> " << st.lam_after << ", removed " << st.removed << ", |perturbation| " << npert
           << ", |result - causal original| " << dist << ", iterations " << st.iters);
      REQUIRE(st.lam_before < -1e-6);
      REQUIRE(lam_min_neg(S) >= -1e-12 * st.y_norm);
      REQUIRE(st.removed <= 1.01 * npert);     // the exact projection removes exactly the perturbation (plain ADMM: 1.04)
      REQUIRE(st.removed >= 0.05 * npert);
      REQUIRE(dist <= 1.05 * npert + 1e-4 * frob3(S0));

      // warm start: the same input again from the converged state
      nda::array<ComplexType, 3> S2 = S0 + dS;
      auto st2 = P.project(S2, C, U);
      INFO("warm start: iterations " << st2.iters << " (cold " << st.iters << ")");
      REQUIRE(lam_min_neg(S2) >= -1e-12 * st2.y_norm);
      REQUIRE(frob3(nda::array<ComplexType, 3>(S2 - S)) <= 0.1 * npert);
    }
  }

  // Production-like mesh (beta 1000, wmax 15: 92 tau, 79 nodes) and a nearly causal Sigma, the regime of a converging
  // Sigma-vertex SCF: an exactly representable causal part (PSD weights of mixed rank on the nodes, the last two bands
  // empty) plus a non-causal part of 1e-7 relative (a negative weight on the empty bands). The exact projection returns the
  // causal part, so |result - causal part| is the solver's error. The ADMM does not reach tol within maxit here (an
  // ill-conditioned, degenerate problem; see the header): the result must be causal, flagged as not converged, and its error
  // must be well below what plain over-relaxed ADMM (rho 1e-6, 2000 iterations) leaves at the same cost.
  TEST_CASE("sigma_causal_projection_convergence", "[methods][scf][causal_proj]") {
    const double beta = 1000.0, wmax = 15.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0), nb = 8, nb0 = nb - 2;
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);
    methods::causal_proj::options o;          // the defaults, with the nodes inside the window as scf_loop sets them
    o.wmax = wmax;
    // A call that stops short of the solver's floor, so the warm-start check below has room to act: the default maxit
    // (6000) reaches ~5e-7 in a single call here, the floor the warm-started calls cannot halve.
    o.maxit = 1800;
    methods::causal_proj::options o_old = o;  // plain over-relaxed ADMM, no Anderson
    o_old.aa_mem = 0; o_old.rho = 1e-6; o_old.relax = 1.6; o_old.maxit = 2000;
    methods::causal_proj::projector P(beta, x, o), P_old(beta, x, o_old);
    auto const &w = P.nodes();
    nda::array<ComplexType, 3> S0(nt, nb, nb), dS(nt, nb, nb);
    S0() = 0.0;
    for (long l = 0; l < w.shape(0); ++l) {
      if (std::abs(w(l)) < 0.04 or std::abs(w(l)) > 14.0 or l % 4 == 0) continue;
      const double env = std::exp(-std::pow(std::log(std::abs(w(l)) / 0.8), 2)) / 40.0;
      nda::matrix<ComplexType> R(nb, nb);
      R() = 0.0;
      R(nda::range(0, nb0), nda::range(0, nb0)) = env * psd_matrix(nb0, 1 + (l % 4) * (nb0 - 1) / 3, 13 * l + 5);
      for (long t = 0; t < nt; ++t) S0(t, nda::range::all, nda::range::all) -= kern(tau(t), w(l), beta) * R;
    }
    nda::matrix<ComplexType> Pm(nb, nb);
    Pm() = 0.0;
    Pm(nda::range(nb0, nb), nda::range(nb0, nb)) = psd_matrix(2, 1, 99);
    for (long t = 0; t < nt; ++t) dS(t, nda::range::all, nda::range::all) = kern(tau(t), 0.3, beta) * Pm;
    dS *= 1e-7 * frob3(S0) / frob3(dS);
    const nda::array<ComplexType, 3> Sin = S0 + dS;
    REQUIRE(lam_min_neg(Sin) < 0.0);
    const double ny = frob3(Sin);

    nda::array<ComplexType, 3> S = Sin, S_old = Sin;
    nda::matrix<ComplexType> C, U, C_old, U_old;
    auto st = P.project(S, C, U);
    auto st_old = P_old.project(S_old, C_old, U_old);
    const double err = frob3(nda::array<ComplexType, 3>(S - S0)) / ny, err_old = frob3(nda::array<ComplexType, 3>(S_old - S0)) / ny;
    INFO("error " << err << " after " << st.iters << " iterations (former solver " << err_old << " after " << st_old.iters
         << "), removed " << st.removed / st.y_norm << ", |non-causal part| " << frob3(dS) / ny << ", fit_res " << st.fit_res
         << ", lambda_min " << st.lam_before << " -> " << st.lam_after);
    REQUIRE(lam_min_neg(S) >= -1e-12 * st.y_norm);                  // causal at any iteration count
    REQUIRE(st.converged == (st.iters < o.maxit));                    // not converged is reported, never hidden
    REQUIRE(err < 0.5 * err_old);
    REQUIRE(err < 2e-6);
    // the warm start carries the convergence on: three more calls on the same input
    double err_w = err;
    for (int c = 0; c < 3; ++c) {
      S = Sin;
      auto st_w = P.project(S, C, U);
      REQUIRE(lam_min_neg(S) >= -1e-12 * st_w.y_norm);
      err_w = frob3(nda::array<ComplexType, 3>(S - S0)) / ny;
    }
    INFO("error after 3 warm-started calls " << err_w << " (first call " << err << ")");
    REQUIRE(err_w < 0.5 * err);
  }

  // production size (60 bands, beta 1000, wmax 15, DLR high): the time per ADMM iteration. Hidden: run with "[.bench]".
  // The thread count is the omp_threads knob, set here from COQUI_BENCH_OMP_THREADS (default 1).
  TEST_CASE("sigma_causal_projection_bench", "[.bench]") {
    if (const char *e = std::getenv("COQUI_BENCH_OMP_THREADS")) utils::set_omp_threads(std::atol(e));
    const double beta = 1000.0, wmax = 15.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0), nb = 60;
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);
    nda::array<ComplexType, 3> S(nt, nb, nb);
    S() = 0.0;
    for (long p = 0; p < 240; ++p) {
      const double e = (p < 120) ? -(0.3 + 4.7 * double(p) / 119.0) : 0.25 + 4.75 * double(p - 120) / 119.0;
      auto R = psd_matrix(nb, 2, 17 * p + 1);
      for (long t = 0; t < nt; ++t) S(t, nda::range::all, nda::range::all) -= 1e-3 * kern(tau(t), e, beta) * R;
    }
    methods::causal_proj::options o;
    o.maxit = 200;
    o.tol = 1e-16;
    if (const char *e = std::getenv("COQUI_CP_AA")) o.aa_mem = std::atol(e);
    if (const char *e = std::getenv("COQUI_CP_RHO")) o.rho = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_RELAX")) o.relax = std::atof(e);
    methods::causal_proj::projector P(beta, x, o);
    nda::matrix<ComplexType> C, U;
    auto t0 = std::chrono::steady_clock::now();
    auto st = P.project(S, C, U);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    WARN("omp_threads " << utils::omp_threads() << ", nt " << nt << ", nodes " << P.n_nodes() << ", nb " << nb << ": " << st.iters << " ADMM iterations in " << dt << " s ("
         << 1e3 * dt / double(st.iters) << " ms each), removed " << st.removed / st.y_norm);
    // the PSD step alone on the converged state (+ a small indefinite shift), and the LS gemm alone
    nda::matrix<ComplexType> X = C;
    for (long l = 0; l < X.shape(0); ++l)
      for (long i = 0; i < nb; ++i) X(l, i * nb + i) -= 1e-9;
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < 10; ++r) { nda::matrix<ComplexType> Y = X; methods::causal_proj::projector::psd_rows(Y, nb); }
    const double dpsd = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / 10.0;
    nda::matrix<ComplexType> M(X.shape(0), X.shape(0)), B(X.shape(0), nb * nb);
    M() = 1.0;
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < 10; ++r) nda::blas::gemm(ComplexType(1.0), M, X, ComplexType(0.0), B);
    const double dgemm = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count() / 10.0;
    WARN("PSD step " << 1e3 * dpsd << " ms, LS gemm " << 1e3 * dgemm << " ms per ADMM iteration");
  }

  // Production-like convergence harness (hidden, "[.conv]"): beta 1000, wmax 15 (92 tau, 79 nodes), nb from
  // COQUI_CONV_NB (12). Cases: an exactly representable causal Sigma (PSD weights of mixed rank ON the nodes, last two
  // bands empty), the same plus a negative weight on the empty bands (relative size 1e-7 and 1e-3: the exact projection
  // returns the causal part, so |result - causal| is the solver's error), and a causal continuum of 400 off-node poles.
  // Options from COQUI_CP_{AA,RHO,RELAX,TOL,AAREG,AARESTART,SELECT,NW}, the maxit values from COQUI_CONV_MAXIT (comma
  // list), threads from COQUI_BENCH_OMP_THREADS. COQUI_CONV_SCF=1: instead, an SCF-like sequence of warm-started calls.
  TEST_CASE("sigma_causal_projection_conv", "[.conv]") {
    if (const char *e = std::getenv("COQUI_BENCH_OMP_THREADS")) utils::set_omp_threads(std::atol(e));
    const double beta = 1000.0, wmax = 15.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0);
    const long nb = std::getenv("COQUI_CONV_NB") ? std::atol(std::getenv("COQUI_CONV_NB")) : 12, nb0 = nb - 2;
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);
    methods::causal_proj::options o;
    o.wmax = wmax;
    o.tol = 1e-12;
    if (const char *e = std::getenv("COQUI_CP_AA")) o.aa_mem = std::atol(e);
    if (const char *e = std::getenv("COQUI_CP_RHO")) o.rho = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_RELAX")) o.relax = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_TOL")) o.tol = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_AAREG")) o.aa_reg = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_SELECT")) o.select = std::atof(e);
    if (const char *e = std::getenv("COQUI_CP_NW")) o.nw = std::atol(e);
    if (const char *e = std::getenv("COQUI_CP_AARESTART")) o.aa_restart = std::atof(e);
    methods::causal_proj::projector P(beta, x, o);
    auto const &w = P.nodes();
    auto add_pole = [&](nda::array<ComplexType, 3> &S, double e, nda::matrix<ComplexType> const &R) {
      for (long t = 0; t < nt; ++t) S(t, nda::range::all, nda::range::all) -= kern(tau(t), e, beta) * R;
    };
    auto embed = [&](nda::matrix<ComplexType> const &R0) {
      nda::matrix<ComplexType> R(nb, nb);
      R() = 0.0;
      R(nda::range(0, R0.shape(0)), nda::range(0, R0.shape(1))) = R0;
      return R;
    };
    auto env = [](double e) { return std::exp(-std::pow(std::log(std::abs(e) / 0.8), 2)); };
    nda::array<ComplexType, 3> Sex(nt, nb, nb), Soff(nt, nb, nb), dS(nt, nb, nb);
    Sex() = 0.0; Soff() = 0.0; dS() = 0.0;
    for (long l = 0; l < w.shape(0); ++l)
      if (std::abs(w(l)) > 0.04 and std::abs(w(l)) < 14.0 and l % 4 != 0)
        add_pole(Sex, w(l), embed(env(w(l)) / 40.0 * psd_matrix(nb0, 1 + (l % 4) * (nb0 - 1) / 3, 13 * l + 5)));
    for (long p = 0; p < 400; ++p) {
      const double f = std::fmod(0.6180339887498949 * double(p + 1), 1.0);
      const double e = (p % 2 == 0 ? -1.0 : 1.0) * 0.04 * std::exp(f * std::log(14.0 / 0.04));
      add_pole(Soff, e, embed(env(e) / 400.0 * psd_matrix(nb0, 1 + p % nb0, 17 * p + 1)));
    }
    nda::matrix<ComplexType> Pm(nb, nb);
    Pm() = 0.0;
    Pm(nda::range(nb0, nb), nda::range(nb0, nb)) = psd_matrix(2, 1, 99);
    for (long t = 0; t < nt; ++t) dS(t, nda::range::all, nda::range::all) = kern(tau(t), 0.3, beta) * Pm;
    std::vector<long> maxits = {500, 1000, 2000, 4000};
    if (const char *e = std::getenv("COQUI_CONV_MAXIT")) {
      maxits.clear();
      std::string str(e);
      for (size_t a = 0; a < str.size();) {
        size_t b = str.find(',', a);
        maxits.push_back(std::atol(str.substr(a, b - a).c_str()));
        if (b == std::string::npos) break;
        a = b + 1;
      }
    }
    if (std::getenv("COQUI_CONV_SCF")) {
      // SCF-like sequence: the off-node causal Sigma plus a causal change that halves every call, plus a fixed non-causal
      // part (1e-7 of |Y| on the empty bands); warm start from call to call, maxit[0] ADMM iterations per call
      nda::array<ComplexType, 3> dC(nt, nb, nb);
      dC() = 0.0;
      for (long p = 0; p < 60; ++p) {
        const double e = (p % 2 == 0 ? -1.0 : 1.0) * (0.1 + 0.15 * double(p));
        add_pole(dC, e, embed(psd_matrix(nb0, 2, 31 * p + 7)));
      }
      const double ny0 = frob3(Soff);
      nda::array<ComplexType, 3> base = Soff + (1e-7 * ny0 / frob3(dS)) * dS;
      methods::causal_proj::options oi = o;
      oi.maxit = maxits[0];
      methods::causal_proj::projector Pi(beta, x, oi);
      nda::matrix<ComplexType> C, U;
      nda::array<ComplexType, 3> Sprev = base, Mprev = base;
      for (long k = 0; k < 14; ++k) {
        nda::array<ComplexType, 3> S = base + (1e-3 * std::pow(0.5, double(k)) * ny0 / frob3(dC)) * dC;
        nda::array<ComplexType, 3> Si = S;
        auto st = Pi.project(Si, C, U);
        std::printf("scf aa %ld rho %.0e relax %.1f maxit %ld call %2ld: it %5ld%s removed/|Y| %.3e |dIn|/|Y| %.2e |dOut|/|Y| %.2e "
                    "fit_res %.2e lam_after %.1e\n", o.aa_mem, o.rho, o.relax, oi.maxit, k, st.iters, st.converged ? " (conv)" : "",
                    st.removed / st.y_norm, frob3(nda::array<ComplexType, 3>(S - Sprev)) / ny0,
                    frob3(nda::array<ComplexType, 3>(Si - Mprev)) / ny0, st.fit_res, st.lam_after);
        std::fflush(stdout);
        Sprev = S; Mprev = Si;
      }
      return;
    }
    struct cs { const char *name; double eta; bool off; };
    for (auto c : {cs{"exact", 0.0, false}, cs{"exact+1e-7", 1e-7, false}, cs{"exact+1e-3", 1e-3, false}, cs{"off-node", 0.0, true}}) {
      nda::array<ComplexType, 3> S0 = c.off ? Soff : Sex;
      nda::array<ComplexType, 3> S = S0 + (c.eta * frob3(S0) / frob3(dS)) * dS;
      for (long mi : maxits) {
        methods::causal_proj::options oi = o;
        oi.maxit = mi;
        methods::causal_proj::projector Pi(beta, x, oi);
        nda::array<ComplexType, 3> Si = S;
        nda::matrix<ComplexType> C, U;
        auto t0 = std::chrono::steady_clock::now();
        auto st = Pi.project(Si, C, U);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-11s nb %ld nodes %ld aa %ld rho %.0e relax %.1f maxit %5ld: it %5ld%s removed/|Y| %.3e |res-causal|/|Y| %.3e "
                    "|pert|/|Y| %.1e fit_res %.2e lam %.1e -> %.1e restarts %ld %.2f s\n", c.name, nb, P.n_nodes(), o.aa_mem, o.rho,
                    o.relax, mi, st.iters, st.converged ? " (conv)" : "", st.removed / st.y_norm,
                    c.off ? -1.0 : frob3(nda::array<ComplexType, 3>(Si - S0)) / st.y_norm,
                    frob3(nda::array<ComplexType, 3>(S - S0)) / st.y_norm, st.fit_res, st.lam_before, st.lam_after, st.aa_restarts, dt);
        std::fflush(stdout);
      }
    }
  }

#if defined(ENABLE_CUDA)
  namespace {
    // the production-mesh input of "sigma_causal_projection_convergence" for nb bands: S0 exactly representable and causal
    // (PSD weights of mixed rank on the nodes, the last two bands empty), dS a negative weight on the empty bands (unit norm)
    std::pair<nda::array<ComplexType, 3>, nda::array<ComplexType, 3>> production_case(nda::array<double, 1> const &tau, double beta,
                                                                                       nda::array<double, 1> const &w, long nb) {
      const long nt = tau.shape(0), nb0 = nb - 2;
      nda::array<ComplexType, 3> S0(nt, nb, nb), dS(nt, nb, nb);
      S0() = 0.0;
      for (long l = 0; l < w.shape(0); ++l) {
        if (std::abs(w(l)) < 0.04 or std::abs(w(l)) > 14.0 or l % 4 == 0) continue;
        const double env = std::exp(-std::pow(std::log(std::abs(w(l)) / 0.8), 2)) / 40.0;
        nda::matrix<ComplexType> R(nb, nb);
        R() = 0.0;
        R(nda::range(0, nb0), nda::range(0, nb0)) = env * psd_matrix(nb0, 1 + (l % 4) * (nb0 - 1) / 3, 13 * l + 5);
        for (long t = 0; t < nt; ++t) S0(t, nda::range::all, nda::range::all) -= kern(tau(t), w(l), beta) * R;
      }
      nda::matrix<ComplexType> Pm(nb, nb);
      Pm() = 0.0;
      Pm(nda::range(nb0, nb), nda::range(nb0, nb)) = psd_matrix(2, 1, 99);
      for (long t = 0; t < nt; ++t) dS(t, nda::range::all, nda::range::all) = kern(tau(t), 0.3, beta) * Pm;
      dS /= frob3(dS);
      return {S0, dS};
    }
    // a causal continuum of off-node poles (n_p poles, PSD residues on the first nb - 2 bands)
    nda::array<ComplexType, 3> continuum_case(nda::array<double, 1> const &tau, double beta, long nb, long n_p, long seed) {
      const long nt = tau.shape(0), nb0 = nb - 2;
      nda::array<ComplexType, 3> S(nt, nb, nb);
      S() = 0.0;
      for (long p = 0; p < n_p; ++p) {
        const double f = std::fmod(0.6180339887498949 * double(p + 1 + seed), 1.0);
        const double e = (p % 2 == 0 ? -1.0 : 1.0) * 0.04 * std::exp(f * std::log(14.0 / 0.04));
        const double env = std::exp(-std::pow(std::log(std::abs(e) / 0.8), 2)) / double(n_p);
        nda::matrix<ComplexType> R(nb, nb);
        R() = 0.0;
        R(nda::range(0, nb0), nda::range(0, nb0)) = env * psd_matrix(nb0, 1 + p % 3, 17 * p + 1 + seed);
        for (long t = 0; t < nt; ++t) S(t, nda::range::all, nda::range::all) -= kern(tau(t), e, beta) * R;
      }
      return S;
    }
    double rel_diff(nda::array<ComplexType, 3> const &a, nda::array<ComplexType, 3> const &b) {
      return frob3(nda::array<ComplexType, 3>(a - b)) / std::max(frob3(b), 1e-300);
    }
    double rel_diff_m(nda::matrix<ComplexType> const &a, nda::matrix<ComplexType> const &b) {
      double s = 0.0, n = 0.0;
      for (long i = 0; i < a.shape(0); ++i)
        for (long j = 0; j < a.shape(1); ++j) { s += std::norm(a(i, j) - b(i, j)); n += std::norm(b(i, j)); }
      return std::sqrt(s / std::max(n, 1e-300));
    }

    // host (one block after the other) and device (one batched call) on the same blocks, `calls` warm-started calls each;
    // per call and block: |S_dev - S_host| / |S_host|, the warm-start state, and the stats. Returns the largest relative
    // difference of the projected Sigma over the calls and blocks.
    struct cmp_result { double ds_max = 0.0, dstate_max = 0.0, t_host = 0.0, t_dev = 0.0; long it_dmax = 0; };
    cmp_result compare_paths(char const *name, methods::causal_proj::projector const &P, std::vector<nda::array<ComplexType, 3>> const &in,
                             int calls) {
      const size_t nblk = in.size();
      std::vector<nda::matrix<ComplexType>> Ch(nblk), Uh(nblk), Cd(nblk), Ud(nblk);
      cmp_result r;
      for (int c = 0; c < calls; ++c) {
        std::vector<nda::array<ComplexType, 3>> Sh(in.begin(), in.end()), Sd(in.begin(), in.end());
        std::vector<methods::causal_proj::stats> sth(nblk);
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < nblk; ++i) sth[i] = P.project(Sh[i], Ch[i], Uh[i]);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<nda::array<ComplexType, 3> *> pS;
        std::vector<nda::matrix<ComplexType> *> pC, pU;
        for (size_t i = 0; i < nblk; ++i) { pS.push_back(&Sd[i]); pC.push_back(&Cd[i]); pU.push_back(&Ud[i]); }
        methods::causal_proj_cuda::call_info ci;
        auto std_ = P.project_device(pS, pC, pU, &ci);
        auto t2 = std::chrono::steady_clock::now();
        r.t_host += std::chrono::duration<double>(t1 - t0).count();
        r.t_dev += std::chrono::duration<double>(t2 - t1).count();
        for (size_t i = 0; i < nblk; ++i) {
          const double ds = rel_diff(Sd[i], Sh[i]), dc = rel_diff_m(Cd[i], Ch[i]), du = rel_diff_m(Ud[i], Uh[i]);
          r.ds_max = std::max(r.ds_max, ds);
          r.dstate_max = std::max(r.dstate_max, dc);
          r.it_dmax = std::max(r.it_dmax, std::abs(std_[i].iters - sth[i].iters));
          std::printf("  %-22s call %d block %zu: |S_dev - S_host|/|S| %.2e |C_dev - C_host|/|C| %.2e |U_dev - U_host|/|U| %.2e; "
                        "removed/|Y| host %.6e dev %.6e; lambda_min after host %.2e dev %.2e; iters host %ld dev %ld (conv %d/%d); "
                        "fit_res host %.3e dev %.3e; restarts %ld/%ld\n",
                        name, c, i, ds, dc, du, sth[i].removed / sth[i].y_norm, std_[i].removed / std_[i].y_norm, sth[i].lam_after,
                        std_[i].lam_after, sth[i].iters, std_[i].iters, int(sth[i].converged), int(std_[i].converged), sth[i].fit_res,
                        std_[i].fit_res, sth[i].aa_restarts, std_[i].aa_restarts);
          // the host's own lambda_min / removed definitions on the device output
          REQUIRE(lam_min_neg(Sd[i]) >= -1e-12 * std_[i].y_norm);
          REQUIRE(std::abs(std_[i].y_norm - sth[i].y_norm) <= 1e-13 * sth[i].y_norm);
          REQUIRE(std::abs(std_[i].lam_before - sth[i].lam_before) <= 1e-12 * sth[i].y_norm);
        }
        std::printf("  %-22s call %d: host %.2f s, device %.2f s (batches %ld of %ld, Jacobi sweeps <= %ld warm / %ld cold, %.0f clips, "
                      "%.1f %% kept, mean sweeps %.2f, entry %.3f loop %.3f exit %.3f s, %s memory, %.0f MB)\n",
                      name, c, std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(), ci.nbatch,
                      ci.batch, ci.sweeps_max, ci.sweeps_max_cold, ci.psd_calls, 100.0 * ci.psd_kept / std::max(ci.psd_calls, 1.0),
                      ci.sweeps_mean, ci.t_entry, ci.t_loop, ci.t_exit, ci.shared_mem ? "shared" : "global", ci.device_mb);
        std::fflush(stdout);
      }
      return r;
    }
    // the host path's own sensitivity: the same blocks with a relative perturbation of 1e-15 (a fixed pattern), host vs host
    double host_floor(methods::causal_proj::projector const &P, std::vector<nda::array<ComplexType, 3>> const &in) {
      double dmax = 0.0;
      for (auto const &s : in) {
        nda::array<ComplexType, 3> a = s, b = s;
        long k = 0;
        for (auto &v : b) v *= (1.0 + 1e-15 * std::sin(0.37 * double(k++)));
        nda::matrix<ComplexType> C1, U1, C2, U2;
        P.project(a, C1, U1);
        P.project(b, C2, U2);
        dmax = std::max(dmax, rel_diff(b, a));
      }
      return dmax;
    }
  } // namespace

  // The device path (causal_proj_cuda.cu) against the host path. (1) The PSD clip alone: random, PSD, indefinite, zero and
  // degenerate matrices, nb from 6 to 100 (shared-memory and global-memory Jacobi), agreement with projector::psd_rows to
  // ~1e-13. (2) The batched solver on the production mesh (beta 1000, wmax 15: 92 tau, 79 nodes), nb 8 and 60: the
  // nearly causal input of "sigma_causal_projection_convergence", the same with a 1e-3 non-causal part, an exactly causal
  // input and an off-node continuum, in ONE device call against the host one block after the other, three warm-started
  // calls. After a few iterations the two agree to roundoff; after the full default count of Anderson iterations the difference is
  // compared with the host path's own response to a 1e-15 perturbation of its input (the Anderson solve amplifies any
  // roundoff), and with the solver's distance from the exact projection.
  TEST_CASE("sigma_causal_projection_device", "[methods][scf][causal_proj][device]") {
    if (const char *e = std::getenv("COQUI_BENCH_OMP_THREADS")) utils::set_omp_threads(std::atol(e));
    const double beta = 1000.0, wmax = 15.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0);
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);

    SECTION("the PSD clip") {
      for (long nb : {6l, 8l, 33l, 60l, 61l, 100l}) {
        for (bool force_global : {false, true}) {
          if (force_global and nb != 60) continue;
          const long rows = 6;
          nda::matrix<ComplexType> X(rows, nb * nb);
          auto A = psd_matrix(nb, std::max(1l, nb / 3), 5 + nb);
          double amax = 0.0;
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) {
              X(0, i * nb + j) = A(i, j);                                                    // PSD, rank nb / 3
              X(1, i * nb + j) = A(i, j) - (i == j ? 0.3 * std::real(A(0, 0)) : 0.0);        // indefinite
              X(2, i * nb + j) = ComplexType(std::sin(0.3 * double(i * nb + j)), std::cos(0.7 * double(i + 3 * j)));   // non-Hermitian
              X(3, i * nb + j) = 0.0;                                                        // zero
              X(4, i * nb + j) = A(i, j) - (i == j ? 1e-12 * std::real(A(0, 0)) : 0.0);      // degenerate cluster just below 0
              X(5, i * nb + j) = (i == j) ? ComplexType(double(i % 3) - 1.0, 0.0) : 0.0;     // diagonal, degenerate +-1, 0
              amax = std::max(amax, std::abs(A(i, j)));
            }
          nda::matrix<ComplexType> Xh = X, Xd = X;
          methods::causal_proj::projector::psd_rows(Xh, nb);
          long sw = 0, kept = 0;
          methods::causal_proj_cuda::psd_rows(Xd.data(), rows, nb, force_global, &sw, &kept);
          for (long r = 0; r < rows; ++r) {
            double d = 0.0, n = 0.0;
            for (long e = 0; e < nb * nb; ++e) { d = std::max(d, std::abs(Xd(r, e) - Xh(r, e))); n = std::max(n, std::abs(X(r, e))); }
            std::printf("  PSD clip nb %3ld %s row %ld: max |dev - host| / max |X| = %.2e\n", nb, force_global ? "global" : "auto  ", r,
                        d / std::max(n, 1e-300));
            REQUIRE(d <= 1e-12 * std::max(n, 1e-300) + 1e-300);
          }
          std::printf("  PSD clip nb %3ld: Jacobi sweeps <= %ld (cold), %ld of %ld rows kept by the semidefinite test\n", nb, sw, kept, rows);
        }
      }
    }

    SECTION("the batched solver against the host path") {
      for (long nb : {8l, 60l}) {
        methods::causal_proj::options o;
        o.wmax = wmax;
        methods::causal_proj::projector P(beta, x, o);
        auto [S0, dS] = production_case(tau, beta, P.nodes(), nb);
        const double n0 = frob3(S0);
        std::vector<nda::array<ComplexType, 3>> in;
        in.emplace_back(S0 + (1e-7 * n0) * dS);                                 // nearly causal (the production regime)
        in.emplace_back(S0 + (1e-3 * n0) * dS);                                 // a sizeable non-causal part
        in.emplace_back(S0);                                                    // exactly causal
        in.emplace_back(continuum_case(tau, beta, nb, 200, 3));                 // off-node causal continuum
        // (a) 20 iterations without Anderson (plain ADMM): roundoff agreement, two warm-started calls
        {
          methods::causal_proj::options os = o;
          os.maxit = 20;
          os.aa_mem = 0;
          methods::causal_proj::projector Ps(beta, x, os);
          auto r = compare_paths(nb == 8 ? "nb 8, plain, maxit 20" : "nb 60, plain, maxit 20", Ps, in, 2);
          const double fl = host_floor(Ps, in);
          std::printf("nb %ld plain ADMM maxit 20: max |S_dev - S_host|/|S| = %.2e (host vs host, 1e-15 input perturbation: %.2e), "
                      "max |C_dev - C_host|/|C| = %.2e, max |iters diff| %ld\n", nb, r.ds_max, fl, r.dstate_max, r.it_dmax);
          // the least-squares step amplifies roundoff by up to cond(Kt^T Kt + rho) ~ 1 / options::rho = 3e6
          CHECK(r.ds_max < std::max(10.0 * fl, 1e-12));
          CHECK(r.it_dmax == 0);
        }
        // (a') 20 iterations with Anderson: the same against the host path's own response to a 1e-15 input perturbation
        {
          methods::causal_proj::options os = o;
          os.maxit = 20;
          methods::causal_proj::projector Ps(beta, x, os);
          auto r = compare_paths(nb == 8 ? "nb 8, Anderson, maxit 20" : "nb 60, Anderson, maxit 20", Ps, in, 1);
          const double fl = host_floor(Ps, in);
          std::printf("nb %ld Anderson maxit 20: max |S_dev - S_host|/|S| = %.2e (host vs host, 1e-15 input perturbation: %.2e), "
                      "max |iters diff| %ld\n", nb, r.ds_max, fl, r.it_dmax);
          CHECK(r.it_dmax == 0);
        }
        // (b) the default number of Anderson iterations (options::maxit), three warm-started calls
        auto r = compare_paths(nb == 8 ? "nb 8, defaults" : "nb 60, defaults", P, in, 3);
        const double floor = host_floor(P, in);
        // the solver's distance from the exact projection on the nearly causal block (host and device)
        nda::array<ComplexType, 3> Sh = in[0], Sd = in[0];
        nda::matrix<ComplexType> Ch, Uh, Cd, Ud;
        P.project(Sh, Ch, Uh);
        std::vector<nda::array<ComplexType, 3> *> pS = {&Sd};
        std::vector<nda::matrix<ComplexType> *> pC = {&Cd}, pU = {&Ud};
        P.project_device(pS, pC, pU);
        const double err_h = rel_diff(Sh, S0), err_d = rel_diff(Sd, S0);
        std::printf("nb %ld defaults: max |S_dev - S_host|/|S| = %.2e over 3 calls (host vs host with a 1e-15 input perturbation: %.2e); "
                    "|result - exact|/|S| host %.3e device %.3e; wall host %.2f s (omp_threads %ld), device %.2f s\n",
                    nb, r.ds_max, floor, err_h, err_d, r.t_host, long(utils::omp_threads()), r.t_dev);
        std::fflush(stdout);
        // the device result is as good as the host one (the two trajectories differ as the host's own perturbed one does)
        CHECK(err_d < 1.5 * err_h + 1e-9);
        CHECK(r.it_dmax <= 1);
      }
    }
  }

  // Production-size timing (hidden, "[.bench_device]"): nb from COQUI_BENCH_NB (60), COQUI_BENCH_NBLK blocks (17: the
  // (s, k) blocks per rank of an 8^3 Si calculation) of nearly causal input, maxit from COQUI_BENCH_MAXIT (default: options::maxit); the host time for one
  // block at omp_threads (COQUI_BENCH_OMP_THREADS) scaled to the call.
  TEST_CASE("sigma_causal_projection_bench_device", "[.bench_device]") {
    if (const char *e = std::getenv("COQUI_BENCH_OMP_THREADS")) utils::set_omp_threads(std::atol(e));
    const long nb = std::getenv("COQUI_BENCH_NB") ? std::atol(std::getenv("COQUI_BENCH_NB")) : 60;
    const long nblk = std::getenv("COQUI_BENCH_NBLK") ? std::atol(std::getenv("COQUI_BENCH_NBLK")) : 17;
    const double beta = 1000.0, wmax = 15.0;
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = x.shape(0);
    nda::array<double, 1> tau(nt);
    for (long t = 0; t < nt; ++t) tau(t) = 0.5 * beta * (double(x(t)) + 1.0);
    methods::causal_proj::options o;
    o.wmax = wmax;
    if (const char *e = std::getenv("COQUI_BENCH_MAXIT")) o.maxit = std::atol(e);
    if (const char *e = std::getenv("COQUI_CP_JTOL")) o.jacobi_tol = std::atof(e);
    o.device_profile = std::getenv("COQUI_CP_PROFILE") != nullptr;
    methods::causal_proj::projector P(beta, x, o);
    auto [S0, dS] = production_case(tau, beta, P.nodes(), nb);
    const double n0 = frob3(S0);
    std::vector<nda::array<ComplexType, 3>> in;
    for (long b = 0; b < nblk; ++b) in.emplace_back(S0 * (1.0 + 0.01 * double(b)) + (1e-7 * n0) * dS + 1e-4 * continuum_case(tau, beta, nb, 40, b));
    std::vector<nda::array<ComplexType, 3> *> pS;
    std::vector<nda::matrix<ComplexType>> C(static_cast<size_t>(nblk)), U(static_cast<size_t>(nblk));
    std::vector<nda::matrix<ComplexType> *> pC, pU;
    for (long b = 0; b < nblk; ++b) { pS.push_back(&in[size_t(b)]); pC.push_back(&C[size_t(b)]); pU.push_back(&U[size_t(b)]); }
    for (int c = 0; c < 2; ++c) {
      auto in_c = in;
      std::vector<nda::array<ComplexType, 3> *> pSc;
      for (auto &s : in_c) pSc.push_back(&s);
      methods::causal_proj_cuda::call_info ci;
      auto t0 = std::chrono::steady_clock::now();
      auto st = P.project_device(pSc, pC, pU, &ci);
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double rem = 0.0;
      for (auto const &s : st) rem = std::max(rem, s.removed / s.y_norm);
      std::printf("device call %d: %ld blocks nb %ld, %ld iterations: %.2f s (entry %.3f, loop %.3f = %.3f ms per ADMM iteration, exit %.3f); "
                  "batches %ld of %ld, Jacobi sweeps <= %ld warm (mean %.2f) / %ld cold, %.1f %% kept, %s memory, %.0f MB; jacobi_tol %.0e; "
                  "max removed/|Y| %.3e\n",
                  c, nblk, nb, st[0].iters, dt, ci.t_entry, ci.t_loop, 1e3 * ci.t_loop / double(std::max(st[0].iters, 1l)), ci.t_exit,
                  ci.nbatch, ci.batch, ci.sweeps_max, ci.sweeps_mean, ci.sweeps_max_cold, 100.0 * ci.psd_kept / std::max(ci.psd_calls, 1.0),
                  ci.shared_mem ? "shared" : "global", ci.device_mb, o.jacobi_tol, rem);
      if (o.device_profile)
        std::printf("  profile: PSD kernel %.3f s, Minv gemm %.3f s, elementwise / Anderson / control %.3f s; diagonalised clips' cycles: "
                    "load + test %.1f %%, transform %.1f %%, Jacobi %.1f %%, re-orthonormalisation + rebuild %.1f %%\n", ci.t_kernels[0],
                    ci.t_kernels[1], ci.t_kernels[2], 100 * ci.psd_phase[0], 100 * ci.psd_phase[1], 100 * ci.psd_phase[2], 100 * ci.psd_phase[3]);
      std::fflush(stdout);
    }
    {
      nda::array<ComplexType, 3> S = in[0];
      nda::matrix<ComplexType> Ch, Uh;
      auto t0 = std::chrono::steady_clock::now();
      auto st = P.project(S, Ch, Uh);
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("host: one block %.2f s at omp_threads %ld (%ld iterations, %.2f ms each) -> %.1f s for %ld blocks\n", dt,
                  long(utils::omp_threads()), st.iters, 1e3 * dt / double(st.iters), dt * double(nblk), nblk);
    }
  }

  // Real input (hidden, "[.ckpt_device]"): Sigma_c(tau, s, k) of a CoQuí checkpoint (COQUI_CKPT_FILE, iteration
  // COQUI_CKPT_ITER or the final one; beta / wmax of its IAFT from COQUI_CKPT_BETA / COQUI_CKPT_WMAX, 1000 / 15), every
  // (s, k) block projected on the device in one call and the first COQUI_CKPT_NHOST (all) on the host, same options as
  // scf_loop (the nodes inside wmax): the same input to both paths, so the comparison isolates the projection.
  TEST_CASE("sigma_causal_projection_ckpt_device", "[.ckpt_device]") {
    if (const char *e = std::getenv("COQUI_BENCH_OMP_THREADS")) utils::set_omp_threads(std::atol(e));
    const char *fname = std::getenv("COQUI_CKPT_FILE");
    REQUIRE(fname != nullptr);
    const double beta = std::getenv("COQUI_CKPT_BETA") ? std::atof(std::getenv("COQUI_CKPT_BETA")) : 1000.0;
    const double wmax = std::getenv("COQUI_CKPT_WMAX") ? std::atof(std::getenv("COQUI_CKPT_WMAX")) : 15.0;
    nda::array<ComplexType, 5> Sig;
    long iter = std::getenv("COQUI_CKPT_ITER") ? std::atol(std::getenv("COQUI_CKPT_ITER")) : -1;
    {
      h5::file file(fname, 'r');
      auto scf_grp = h5::group(file).open_group("scf");
      if (iter < 0) h5::h5_read(scf_grp, "final_iter", iter);
      auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
      nda::h5_read(iter_grp, "Sigma_tskij", Sig);
    }
    imag_axes_ft::IAFT FT(beta, wmax, imag_axes_ft::dlr_basis, "high");
    auto x = FT.tau_mesh();
    const long nt = Sig.shape(0), ns = Sig.shape(1), nk = Sig.shape(2), nb = Sig.shape(3);
    REQUIRE(nt == x.shape(0));
    methods::causal_proj::options o;
    o.wmax = FT.wmax();
    methods::causal_proj::projector P(beta, x, o);
    const long nblk = ns * nk;
    const long nhost = std::min(nblk, std::getenv("COQUI_CKPT_NHOST") ? std::atol(std::getenv("COQUI_CKPT_NHOST")) : nblk);
    std::printf("checkpoint %s iteration %ld: %ld tau, %ld spins, %ld k, %ld bands; %ld nodes; %ld blocks on the device, %ld on the host\n",
                fname, iter, nt, ns, nk, nb, P.n_nodes(), nblk, nhost);
    std::vector<nda::array<ComplexType, 3>> in, Sd;
    for (long p = 0; p < nblk; ++p) in.emplace_back(Sig(nda::range::all, p / nk, p % nk, nda::range::all, nda::range::all));
    Sd = in;
    std::vector<nda::matrix<ComplexType>> Cd(static_cast<size_t>(nblk)), Ud(static_cast<size_t>(nblk));
    std::vector<nda::array<ComplexType, 3> *> pS;
    std::vector<nda::matrix<ComplexType> *> pC, pU;
    for (long p = 0; p < nblk; ++p) { pS.push_back(&Sd[size_t(p)]); pC.push_back(&Cd[size_t(p)]); pU.push_back(&Ud[size_t(p)]); }
    methods::causal_proj_cuda::call_info ci;
    auto t0 = std::chrono::steady_clock::now();
    auto std_ = P.project_device(pS, pC, pU, &ci);
    const double t_dev = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("device: %ld blocks in %.2f s (entry %.2f, loop %.2f, exit %.2f; batches %ld of %ld; Jacobi sweeps <= %ld warm (mean %.2f) "
                "/ %ld cold; %.1f %% of %.0f clips kept; %.0f MB)\n", nblk, t_dev, ci.t_entry, ci.t_loop, ci.t_exit, ci.nbatch, ci.batch,
                ci.sweeps_max, ci.sweeps_mean, ci.sweeps_max_cold, 100.0 * ci.psd_kept / std::max(ci.psd_calls, 1.0), ci.psd_calls, ci.device_mb);
    double t_host = 0.0, y2 = 0.0, r2h = 0.0, r2d = 0.0, d2 = 0.0, s2 = 0.0, dmax = 0.0;
    for (long p = 0; p < nhost; ++p) {
      nda::array<ComplexType, 3> Sh = in[size_t(p)];
      nda::matrix<ComplexType> Ch, Uh;
      auto t1 = std::chrono::steady_clock::now();
      auto sth = P.project(Sh, Ch, Uh);
      t_host += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      auto const &sd = std_[size_t(p)];
      const double dd = frob3(nda::array<ComplexType, 3>(Sd[size_t(p)] - Sh)), sh = frob3(Sh);
      y2 += sth.y_norm * sth.y_norm; r2h += sth.removed * sth.removed; r2d += sd.removed * sd.removed; d2 += dd * dd; s2 += sh * sh;
      dmax = std::max(dmax, dd / sh);
      std::printf("  block %3ld: |S_dev - S_host|/|S| %.2e; removed/|Y| host %.4e dev %.4e; lambda_min %.3e -> host %.2e dev %.2e; "
                  "iters %ld / %ld; fit_res %.2e / %.2e\n", p, dd / sh, sth.removed / sth.y_norm, sd.removed / sd.y_norm, sth.lam_before,
                  sth.lam_after, sd.lam_after, sth.iters, sd.iters, sth.fit_res, sd.fit_res);
      std::fflush(stdout);
      REQUIRE(lam_min_neg(Sd[size_t(p)]) >= -1e-12 * sd.y_norm);
    }
    std::printf("host blocks %ld: |S_dev - S_host|/|S| = %.2e (worst block %.2e); removed/|Y| host %.4e dev %.4e; wall host %.1f s "
                "(omp_threads %ld) -> %.1f s for all %ld blocks; device %.2f s for all\n", nhost, std::sqrt(d2 / std::max(s2, 1e-300)), dmax,
                std::sqrt(r2h / std::max(y2, 1e-300)), std::sqrt(r2d / std::max(y2, 1e-300)), t_host, long(utils::omp_threads()),
                t_host * double(nblk) / double(std::max(nhost, 1l)), nblk, t_dev);
  }
#endif

} // namespace bdft_tests
