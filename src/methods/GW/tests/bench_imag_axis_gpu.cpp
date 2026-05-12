/**
 * Synthetic-data performance benches for imag-axis GW kernels (host vs device).
 *
 * Each TEST_CASE is tagged "[.bench]" — Catch2 hides "." tagged cases by
 * default. Run explicitly:
 *
 *     ./test_methods_gw "[.bench]"            # all benches
 *     ./test_methods_gw "bench_iaft"          # only IAFT bench
 *
 * Benches use random synthetic data of realistic shapes (LiH222-scale up to
 * a Si-phase1-scale extrapolation). They isolate each phase of the GW
 * pipeline so we can profile what wins / loses on A100 vs Rusty CPU as
 * problem size grows.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"
#include "arch/arch.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/nda_functions.hpp"

#include <nda/blas.hpp>
#include <nda/tensor.hpp>

#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace bdft_tests {

  // --- helpers ----------------------------------------------------------

  template<typename T>
  void fill_random(nda::array<T,2>& a, unsigned seed=42) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    if constexpr (nda::is_complex_v<T>) {
      for (long i = 0; i < a.shape(0); ++i)
        for (long j = 0; j < a.shape(1); ++j)
          a(i,j) = T(u(g), u(g));
    } else {
      for (long i = 0; i < a.shape(0); ++i)
        for (long j = 0; j < a.shape(1); ++j)
          a(i,j) = T(u(g));
    }
  }

  template<typename T>
  void fill_random(nda::array<T,3>& a, unsigned seed=42) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    if constexpr (nda::is_complex_v<T>) {
      for (long i = 0; i < a.size(); ++i) a.data()[i] = T(u(g), u(g));
    } else {
      for (long i = 0; i < a.size(); ++i) a.data()[i] = T(u(g));
    }
  }

  inline void dev_sync() {
    arch::synchronize();   // no-op when CUDA is disabled
  }

  struct bench_result {
    double total_ms;
    double per_call_ms;
    long nruns;
  };

  template<typename F>
  bench_result time_it(F&& fn, int nruns, int nwarmup=2) {
    for (int i = 0; i < nwarmup; ++i) fn();
    dev_sync();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nruns; ++i) fn();
    dev_sync();
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total_ms, total_ms / nruns, nruns};
  }

  inline void print_row(std::string const& label, long N,
                        bench_result const& h, bench_result const& d) {
    std::cout << std::left << std::setw(28) << label
              << std::right << std::setw(10) << N
              << "  host: " << std::setw(9) << std::fixed
              << std::setprecision(3) << h.per_call_ms << " ms"
              << "  dev: "  << std::setw(9) << d.per_call_ms << " ms"
              << "  speedup: " << std::setw(6) << std::setprecision(2)
              << (h.per_call_ms / std::max(d.per_call_ms, 1e-6)) << "x"
              << std::endl;
  }

  // --- IAFT benches -----------------------------------------------------

  template<MEMORY_SPACE MEM>
  bench_result bench_tau_to_w_PHsym(imag_axes_ft::IAFT const& ft,
                                    long dim1, int nruns) {
    size_t nt_half = (ft.nt_b() % 2 == 0) ? ft.nt_b()/2 : ft.nt_b()/2 + 1;
    size_t nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b()/2 : ft.nw_b()/2 + 1;

    nda::array<ComplexType, 2> h_in(nt_half, dim1);
    nda::array<ComplexType, 2> h_out(nw_half, dim1);
    fill_random(h_in);

    auto X_in  = memory::to_memory_space<MEM>(h_in);
    auto X_out = memory::to_memory_space<MEM>(h_out);

    return time_it([&]() {
      ft.tau_to_w_PHsym(X_in, X_out);
    }, nruns);
  }

  template<MEMORY_SPACE MEM>
  bench_result bench_w_to_tau_PHsym(imag_axes_ft::IAFT const& ft,
                                    long dim1, int nruns) {
    size_t nt_half = (ft.nt_b() % 2 == 0) ? ft.nt_b()/2 : ft.nt_b()/2 + 1;
    size_t nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b()/2 : ft.nw_b()/2 + 1;

    nda::array<ComplexType, 2> h_in(nw_half, dim1);
    nda::array<ComplexType, 2> h_out(nt_half, dim1);
    fill_random(h_in);

    auto X_in  = memory::to_memory_space<MEM>(h_in);
    auto X_out = memory::to_memory_space<MEM>(h_out);

    return time_it([&]() {
      ft.w_to_tau_PHsym(X_in, X_out);
    }, nruns);
  }

  template<MEMORY_SPACE MEM>
  bench_result bench_tau_to_w_fermi(imag_axes_ft::IAFT const& ft,
                                    long dim1, int nruns) {
    size_t nt = ft.nt_f();
    size_t nw = ft.nw_f();

    nda::array<ComplexType, 2> h_in(nt, dim1);
    nda::array<ComplexType, 2> h_out(nw, dim1);
    fill_random(h_in);

    auto X_in  = memory::to_memory_space<MEM>(h_in);
    auto X_out = memory::to_memory_space<MEM>(h_out);

    return time_it([&]() {
      ft.tau_to_w(X_in, X_out, imag_axes_ft::fermi);
    }, nruns);
  }

  TEST_CASE("bench_iaft", "[.bench][iaft]") {
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_source);
    std::cout << "\n--- IAFT benches ---\n";
    std::cout << "nt_f=" << ft.nt_f() << " nw_f=" << ft.nw_f()
              << " nt_b=" << ft.nt_b() << " nw_b=" << ft.nw_b()
              << " (beta=1000 wmax=1.2)\n";
    std::cout << "Each row times one transform with kernel mirror cost included.\n";

    for (long dim : {128L, 1024L, 8192L, 65536L, 262144L}) {
      auto h_pHs = bench_tau_to_w_PHsym<HOST_MEMORY>(ft, dim, 5);
#if defined(ENABLE_DEVICE)
      auto d_pHs = bench_tau_to_w_PHsym<DEVICE_MEMORY>(ft, dim, 5);
#else
      auto d_pHs = h_pHs;
#endif
      print_row("tau_to_w_PHsym", dim, h_pHs, d_pHs);

      auto h_wpHs = bench_w_to_tau_PHsym<HOST_MEMORY>(ft, dim, 5);
#if defined(ENABLE_DEVICE)
      auto d_wpHs = bench_w_to_tau_PHsym<DEVICE_MEMORY>(ft, dim, 5);
#else
      auto d_wpHs = h_wpHs;
#endif
      print_row("w_to_tau_PHsym", dim, h_wpHs, d_wpHs);

      auto h_f = bench_tau_to_w_fermi<HOST_MEMORY>(ft, dim, 5);
#if defined(ENABLE_DEVICE)
      auto d_f = bench_tau_to_w_fermi<DEVICE_MEMORY>(ft, dim, 5);
#else
      auto d_f = h_f;
#endif
      print_row("tau_to_w (fermi)", dim, h_f, d_f);
    }
  }

  // --- Hadamard (Pi inner loop) ----------------------------------------

  template<MEMORY_SPACE MEM>
  bench_result bench_hadamard(long nt, long P, long Q, int nruns) {
    nda::array<ComplexType, 3> h_A(nt, P, Q);
    nda::array<ComplexType, 3> h_B(nt, P, Q);
    fill_random(h_A);
    fill_random(h_B, 43);

    auto A = memory::to_memory_space<MEM>(h_A);
    auto B = memory::to_memory_space<MEM>(h_B);

    return time_it([&]() {
      // B = A * B (in-place Hadamard MUL on (t,P,Q)).
      nda::tensor::elementwise(ComplexType(1.0), A, "tPQ",
                               ComplexType(1.0), B, "tPQ",
                               nda::tensor::op::MUL);
    }, nruns);
  }

  TEST_CASE("bench_hadamard", "[.bench][hadamard]") {
    std::cout << "\n--- Hadamard (tensor::elementwise MUL on (nt,P,Q)) ---\n";
    long nt = 104;  // typical IR fermionic grid

    for (long P : {32L, 128L, 320L, 1024L}) {
      long Q = P;
      auto h = bench_hadamard<HOST_MEMORY>(nt, P, Q, 3);
#if defined(ENABLE_DEVICE)
      auto d = bench_hadamard<DEVICE_MEMORY>(nt, P, Q, 3);
#else
      auto d = h;
#endif
      print_row("Hadamard (nt,P,P)", P, h, d);
    }
  }

  // --- THC-style batched gemm ------------------------------------------

  template<MEMORY_SPACE MEM>
  bench_result bench_thc_gemm(long P, long Q, long M, int nruns) {
    // Models the per-(s,k) gemm pair in thc_solver_comm aux<->primary:
    // O(a,Q) = X(P,a)^H * A(P,Q), then A'(P,Q) = X(P,a) * O(a,Q).
    nda::array<ComplexType, 2> h_X(P, M);
    nda::array<ComplexType, 2> h_A(P, Q);
    nda::array<ComplexType, 2> h_O(M, Q);
    nda::array<ComplexType, 2> h_B(P, Q);
    fill_random(h_X);
    fill_random(h_A, 43);

    auto X = memory::to_memory_space<MEM>(h_X);
    auto A = memory::to_memory_space<MEM>(h_A);
    auto O = memory::to_memory_space<MEM>(h_O);
    auto B = memory::to_memory_space<MEM>(h_B);

    return time_it([&]() {
      nda::blas::gemm(ComplexType(1.0), nda::dagger(X), A,
                      ComplexType(0.0), O);
      nda::blas::gemm(ComplexType(1.0), X, O,
                      ComplexType(0.0), B);
    }, nruns);
  }

  TEST_CASE("bench_thc_gemm", "[.bench][gemm]") {
    std::cout << "\n--- THC aux<->primary gemm pair ---\n";
    long M = 16;  // ~nbnd
    for (long P : {32L, 128L, 320L, 1024L}) {
      long Q = P;
      auto h = bench_thc_gemm<HOST_MEMORY>(P, Q, M, 3);
#if defined(ENABLE_DEVICE)
      auto d = bench_thc_gemm<DEVICE_MEMORY>(P, Q, M, 3);
#else
      auto d = h;
#endif
      print_row("THC gemm pair (P=Q,M)", P, h, d);
    }
  }

  // --- Tensor contraction (cuTENSOR path; DEVICE only because we build
  //     with TBLIS=OFF and nda::tensor::contract on host needs TBLIS) -----

#if defined(ENABLE_DEVICE)
  bench_result bench_contract_device(long nt, long P, long Q, long R, int nruns) {
    nda::array<ComplexType, 3> h_X(R, P, nt);
    nda::array<ComplexType, 3> h_Y(R, nt, Q);
    nda::array<ComplexType, 3> h_Z(R, P, Q);
    fill_random(h_X);
    fill_random(h_Y, 43);

    auto X = memory::to_memory_space<DEVICE_MEMORY>(h_X);
    auto Y = memory::to_memory_space<DEVICE_MEMORY>(h_Y);
    auto Z = memory::to_memory_space<DEVICE_MEMORY>(h_Z);

    return time_it([&]() {
      // Z(R,P,Q) = sum_t X(R,P,t) * Y(R,t,Q)
      nda::tensor::contract(ComplexType(1.0), X, "Rpt", Y, "RtQ",
                            ComplexType(0.0), Z, "RPQ");
    }, nruns);
  }

  TEST_CASE("bench_contract", "[.bench][contract]") {
    std::cout << "\n--- nda::tensor::contract (DEVICE; cuTENSOR) ---\n";
    std::cout << "Host counterpart skipped: build has -DENABLE_TBLIS=OFF.\n";
    long nt = 104, R = 4;
    for (long P : {32L, 128L, 320L, 1024L}) {
      long Q = P;
      auto d = bench_contract_device(nt, P, Q, R, 3);
      std::cout << std::left << std::setw(28) << "contract (R,P=Q,t)"
                << std::right << std::setw(10) << P
                << "  dev: " << std::setw(9) << std::fixed
                << std::setprecision(3) << d.per_call_ms << " ms" << std::endl;
    }
  }
#endif

} // namespace bdft_tests
