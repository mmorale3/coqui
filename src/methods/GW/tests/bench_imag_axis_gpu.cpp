/**
 * Synthetic-data performance benches for imag-axis GW kernels (host vs device).
 *
 * Sizes target *production* GW workloads, not the LiH222 correctness fixture.
 * LiH222 (Naux=128, Nk=8, nbnd=16) is too small to amortize cuFINUFFT plan
 * setup, cuTENSOR planner cost, or per-call host->device transfers; use it
 * only for correctness checks. These benches sweep over P/Q sizes from
 * 128 up to 4096 and IAFT contracted dims up to ~16M to expose the
 * asymptotic device wins and the crossover points.
 *
 * Each TEST_CASE is tagged "[.bench]" (Catch2 hides "." cases by default).
 * Run explicitly:
 *
 *     ./test_methods_gw "[.bench]"            # all benches
 *     ./test_methods_gw "bench_iaft"          # IAFT only
 *     ./test_methods_gw "[hadamard]"          # Hadamard only
 *
 * The benches emphasise two architectural points for the GPU port:
 *   (1) memory transfers between CPU and GPU should be minimised
 *       (see bench_iaft_cold_vs_warm: cached device kernel vs per-call
 *       mirror), and
 *   (2) concurrency wins come from recasting many small ops as one
 *       large gemm/contraction (see bench_batched_vs_loop_gemm).
 *
 * Memory bookkeeping: each case prints the working-set bytes so we can
 * see what fits on a 40GB A100. Complex double is 16 bytes.
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

  inline std::string fmt_mb(size_t bytes) {
    char buf[32];
    if (bytes > (size_t)(1024*1024*1024))
      std::snprintf(buf, sizeof(buf), "%.1f GB", bytes / (1024.0*1024.0*1024.0));
    else
      std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0*1024.0));
    return buf;
  }

  inline void print_row(std::string const& label, long N,
                        bench_result const& h, bench_result const& d,
                        std::string const& bytes_str = "") {
    std::cout << std::left << std::setw(28) << label
              << std::right << std::setw(8) << N
              << "  host: " << std::setw(10) << std::fixed
              << std::setprecision(3) << h.per_call_ms << " ms"
              << "  dev: "  << std::setw(10) << d.per_call_ms << " ms"
              << "  speedup: " << std::setw(7) << std::setprecision(2)
              << (h.per_call_ms / std::max(d.per_call_ms, 1e-6)) << "x";
    if (!bytes_str.empty()) std::cout << "   " << bytes_str;
    std::cout << std::endl;
  }

  inline void print_row_dev_only(std::string const& label, long N,
                                 bench_result const& d,
                                 std::string const& bytes_str = "") {
    std::cout << std::left << std::setw(28) << label
              << std::right << std::setw(8) << N
              << "  dev: " << std::setw(10) << std::fixed
              << std::setprecision(3) << d.per_call_ms << " ms";
    if (!bytes_str.empty()) std::cout << "   " << bytes_str;
    std::cout << std::endl;
  }

  // --- IAFT benches -----------------------------------------------------
  //
  // "Cold" includes the per-call host->device kernel mirror inside
  // IAFT::tau_to_w_PHsym (current production code).
  // "Warm" pre-mirrors the kernel once and only calls gemm — what we'd
  // see if IAFT cached its device kernel across SCF iters.

  template<MEMORY_SPACE MEM>
  bench_result bench_iaft_pHs_cold(imag_axes_ft::IAFT const& ft,
                                   long dim1, int nruns) {
    size_t nt_half = (ft.nt_b() % 2 == 0) ? ft.nt_b()/2 : ft.nt_b()/2 + 1;
    size_t nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b()/2 : ft.nw_b()/2 + 1;
    nda::array<ComplexType, 2> h_in(nt_half, dim1);
    nda::array<ComplexType, 2> h_out(nw_half, dim1);
    fill_random(h_in);
    auto X_in  = memory::to_memory_space<MEM>(h_in);
    auto X_out = memory::to_memory_space<MEM>(h_out);
    return time_it([&]() { ft.tau_to_w_PHsym(X_in, X_out); }, nruns);
  }

  template<MEMORY_SPACE MEM>
  bench_result bench_iaft_pHs_warm(imag_axes_ft::IAFT const& ft,
                                   long dim1, int nruns) {
    // Build Twt_pos on host (once), mirror to MEM once, then time only gemm.
    long nt = (ft.nt_b() % 2 == 0) ? ft.nt_b()/2 : ft.nt_b()/2 + 1;
    long nw = (ft.nw_b() % 2 == 0) ? ft.nw_b()/2 : ft.nw_b()/2 + 1;
    auto Twt = ft.Twt_bb();
    nda::matrix<ComplexType> Twt_pos(nw, nt);
    for (long n = 0; n < nw; ++n) {
      long iw = ft.nw_b()/2 + n;
      for (long it = 0; it < nt; ++it) {
        long imt = ft.nt_b() - it - 1;
        Twt_pos(n, it) = (it == imt)? Twt(iw, it) : Twt(iw, it) + Twt(iw, imt);
      }
    }
    auto Twt_pos_mem = memory::to_memory_space<MEM>(Twt_pos);

    nda::array<ComplexType, 2> h_in(nt, dim1);
    nda::array<ComplexType, 2> h_out(nw, dim1);
    fill_random(h_in);
    auto X_in  = memory::to_memory_space<MEM>(h_in);
    auto X_out = memory::to_memory_space<MEM>(h_out);
    return time_it([&]() {
      nda::blas::gemm(ComplexType(1.0), Twt_pos_mem, X_in,
                      ComplexType(0.0), X_out);
    }, nruns);
  }

  TEST_CASE("bench_iaft", "[.bench][iaft]") {
    // beta=1000, wmax=10 gives a larger (nw,nt) — production-scale grid.
    imag_axes_ft::IAFT ft(1000, 10.0, imag_axes_ft::ir_source);
    std::cout << "\n--- IAFT tau_to_w_PHsym (cold = per-call mirror) ---\n";
    std::cout << "nt_b=" << ft.nt_b() << " nw_b=" << ft.nw_b()
              << " (beta=1000 wmax=10.0, IR)\n";

    for (long dim : {4096L, 65536L, 262144L, 1048576L, 4194304L, 16777216L}) {
      auto bytes = fmt_mb(2L * dim * ft.nt_b() * sizeof(ComplexType));
      auto h = bench_iaft_pHs_cold<HOST_MEMORY>(ft, dim, 5);
#if defined(ENABLE_DEVICE)
      auto d = bench_iaft_pHs_cold<DEVICE_MEMORY>(ft, dim, 5);
#else
      auto d = h;
#endif
      print_row("tau_to_w_PHsym (cold)", dim, h, d, bytes);
    }
  }

  TEST_CASE("bench_iaft_cold_vs_warm", "[.bench][iaft][cache]") {
    // Same fixture as bench_iaft, but compare cold (per-call kernel mirror)
    // to warm (kernel pre-mirrored once, just the gemm timed). The diff is
    // exactly what an IAFT device-kernel cache would buy us per call.
    imag_axes_ft::IAFT ft(1000, 10.0, imag_axes_ft::ir_source);
    std::cout << "\n--- IAFT cold vs warm (per-call mirror cost) ---\n";

    for (long dim : {65536L, 262144L, 1048576L, 4194304L}) {
#if defined(ENABLE_DEVICE)
      auto cold = bench_iaft_pHs_cold<DEVICE_MEMORY>(ft, dim, 5);
      auto warm = bench_iaft_pHs_warm<DEVICE_MEMORY>(ft, dim, 5);
      std::cout << std::left << std::setw(28) << "tau_to_w_PHsym (dim)"
                << std::right << std::setw(8) << dim
                << "  cold: " << std::setw(8) << std::fixed
                << std::setprecision(3) << cold.per_call_ms << " ms"
                << "  warm: " << std::setw(8) << warm.per_call_ms << " ms"
                << "  mirror_cost: " << std::setw(8)
                << (cold.per_call_ms - warm.per_call_ms) << " ms"
                << "  warm/cold: " << std::setw(5) << std::setprecision(2)
                << (warm.per_call_ms / std::max(cold.per_call_ms, 1e-6))
                << "x" << std::endl;
#endif
    }
  }

  // --- Hadamard (Pi/Sigma inner loop) ----------------------------------
  //
  // tensor::elementwise(MUL) is bandwidth-bound. On host, MKL serial is
  // slow on the simple 3-tensor loop; on device the operation is memory
  // bandwidth-bound and lines up with HBM BW (~1500 GB/s for A100).

  template<MEMORY_SPACE MEM>
  bench_result bench_hadamard(long nt, long P, long Q, int nruns) {
    nda::array<ComplexType, 3> h_A(nt, P, Q);
    nda::array<ComplexType, 3> h_B(nt, P, Q);
    fill_random(h_A);
    fill_random(h_B, 43);
    auto A = memory::to_memory_space<MEM>(h_A);
    auto B = memory::to_memory_space<MEM>(h_B);
    return time_it([&]() {
      nda::tensor::elementwise(ComplexType(1.0), A, "tPQ",
                               ComplexType(1.0), B, "tPQ",
                               nda::tensor::op::MUL);
    }, nruns);
  }

  TEST_CASE("bench_hadamard", "[.bench][hadamard]") {
    std::cout << "\n--- Hadamard tensor::elementwise(MUL) on (nt,P,Q) ---\n";
    long nt = 200;  // production-ish IR/DLR fermion count
    for (long P : {256L, 512L, 1024L, 2048L, 4096L}) {
      long Q = P;
      auto bytes = fmt_mb(3L * nt * P * Q * sizeof(ComplexType));
      auto h = bench_hadamard<HOST_MEMORY>(nt, P, Q, 3);
#if defined(ENABLE_DEVICE)
      auto d = bench_hadamard<DEVICE_MEMORY>(nt, P, Q, 3);
#else
      auto d = h;
#endif
      print_row("Hadamard (nt=200,P=Q)", P, h, d, bytes);
    }
  }

  // --- THC aux<->primary gemm pair --------------------------------------
  //
  // Models the per-(s,k) two-gemm sequence in thc_solver_comm:
  //   O(M,Q) = X(P,M)^H * A(P,Q)
  //   B(P,Q) = X(P,M)   * O(M,Q)

  template<MEMORY_SPACE MEM>
  bench_result bench_thc_gemm(long P, long Q, long M, int nruns) {
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
    long M = 64;  // production-ish nbnd
    for (long P : {256L, 512L, 1024L, 2048L, 4096L}) {
      long Q = P;
      auto bytes = fmt_mb((long)(P*M + P*Q + M*Q + P*Q) * sizeof(ComplexType));
      auto h = bench_thc_gemm<HOST_MEMORY>(P, Q, M, 3);
#if defined(ENABLE_DEVICE)
      auto d = bench_thc_gemm<DEVICE_MEMORY>(P, Q, M, 3);
#else
      auto d = h;
#endif
      print_row("THC gemm pair (P=Q,M=64)", P, h, d, bytes);
    }
  }

  // --- Batched vs loop-of-gemms ----------------------------------------
  //
  // Demonstrates the "recast small ops as one big op" lesson. We have N
  // per-step (P,P)*(P,M) gemms (the per-(s,k) THC pattern). The "loop"
  // form launches one cuBLAS call per step; the "batched" form launches
  // a single gemm on a flattened (N*P, P)*(P, M) shape with the same
  // total FLOP count but only one kernel launch and one wave of HBM
  // traffic.

  template<MEMORY_SPACE MEM>
  bench_result bench_loop_gemm(long N, long P, long M, int nruns) {
    nda::array<ComplexType, 3> h_X(N, P, P);
    nda::array<ComplexType, 3> h_A(N, P, M);
    nda::array<ComplexType, 3> h_Y(N, P, M);
    fill_random(h_X);
    fill_random(h_A, 43);
    auto X = memory::to_memory_space<MEM>(h_X);
    auto A = memory::to_memory_space<MEM>(h_A);
    auto Y = memory::to_memory_space<MEM>(h_Y);
    return time_it([&]() {
      for (long n = 0; n < N; ++n) {
        nda::blas::gemm(ComplexType(1.0),
                        X(n, nda::range::all, nda::range::all),
                        A(n, nda::range::all, nda::range::all),
                        ComplexType(0.0),
                        Y(n, nda::range::all, nda::range::all));
      }
    }, nruns, /*nwarmup=*/1);
  }

  template<MEMORY_SPACE MEM>
  bench_result bench_batched_gemm(long N, long P, long M, int nruns) {
    // Same total FLOP, but the N slices laid out as a single (N*P, P)
    // matrix multiplied by the same P*M matrix? No — that gives wrong
    // semantics. Instead model the *block-diagonal recast*: put all
    // X(n) into a single (N*P, P) and A(n) into (N*P, M), then a single
    // big gemm over a reshape gives the same per-slice product when
    // the underlying memory is contiguous and we feed a 3D contract
    // through cuTENSOR or by reshape+gemm.
    //
    // Simplest implementation that's apples-to-apples: build the (N*P,P)
    // flattened X and (P,M) shared A (broadcast over n) and time one
    // big gemm. This isn't byte-identical to per-slice gemm but matches
    // the FLOP count, so the timing comparison is meaningful for the
    // launch-overhead vs throughput question we care about.
    nda::array<ComplexType, 2> h_X(N*P, P);   // stack N (P,P) along rows
    nda::array<ComplexType, 2> h_A(P, M);
    nda::array<ComplexType, 2> h_Y(N*P, M);
    fill_random(h_X);
    fill_random(h_A, 43);
    auto X = memory::to_memory_space<MEM>(h_X);
    auto A = memory::to_memory_space<MEM>(h_A);
    auto Y = memory::to_memory_space<MEM>(h_Y);
    return time_it([&]() {
      nda::blas::gemm(ComplexType(1.0), X, A, ComplexType(0.0), Y);
    }, nruns, /*nwarmup=*/1);
  }

  TEST_CASE("bench_batched_vs_loop_gemm", "[.bench][batched]") {
    // N small per-(s,k) gemms vs one big batched gemm of same FLOP count.
    std::cout << "\n--- Sequential N gemms (loop) vs one big batched gemm ---\n";
    long M = 64;
    long P = 1024;
    std::cout << "Fixed (P=1024, M=64), sweep N (number of (s,k) batches).\n";
    for (long N : {8L, 16L, 32L, 64L, 128L}) {
#if defined(ENABLE_DEVICE)
      auto loop = bench_loop_gemm<DEVICE_MEMORY>(N, P, M, 3);
      auto batch = bench_batched_gemm<DEVICE_MEMORY>(N, P, M, 3);
      std::cout << std::left << std::setw(28) << "N x (P,P)*(P,M)"
                << std::right << std::setw(8) << N
                << "  loop: " << std::setw(8) << std::fixed
                << std::setprecision(3) << loop.per_call_ms << " ms"
                << "  batched: " << std::setw(8)
                << batch.per_call_ms << " ms"
                << "  speedup: " << std::setw(6) << std::setprecision(2)
                << (loop.per_call_ms / std::max(batch.per_call_ms, 1e-6))
                << "x" << std::endl;
#endif
    }
  }

  // --- cuTENSOR contraction --------------------------------------------
  //
  // Host counterpart skipped (build has TBLIS=OFF). The contraction
  // shape models the Sigma R-space step Z(R,P,Q) = sum_t X(R,P,t)*Y(R,t,Q).

  template<MEMORY_SPACE MEM>
  bench_result bench_contract(long nt, long P, long Q, long R, int nruns) {
    nda::array<ComplexType, 3> h_X(R, P, nt);
    nda::array<ComplexType, 3> h_Y(R, nt, Q);
    nda::array<ComplexType, 3> h_Z(R, P, Q);
    fill_random(h_X);
    fill_random(h_Y, 43);
    auto X = memory::to_memory_space<MEM>(h_X);
    auto Y = memory::to_memory_space<MEM>(h_Y);
    auto Z = memory::to_memory_space<MEM>(h_Z);
    return time_it([&]() {
      nda::tensor::contract(ComplexType(1.0), X, "RPt", Y, "RtQ",
                            ComplexType(0.0), Z, "RPQ");
    }, nruns);
  }

#if defined(ENABLE_DEVICE)
  TEST_CASE("bench_contract", "[.bench][contract]") {
    std::cout << "\n--- cuTENSOR tensor::contract Z(R,P,Q)=sum_t X(R,P,t)*Y(R,t,Q) ---\n";
    long nt = 200, R = 8;
    for (long P : {256L, 512L, 1024L, 2048L, 4096L}) {
      long Q = P;
      auto bytes = fmt_mb((long)R * (P*nt + nt*Q + P*Q) * sizeof(ComplexType));
      auto d = bench_contract<DEVICE_MEMORY>(nt, P, Q, R, 3);
      print_row_dev_only("contract (R=8,P=Q,t=200)", P, d, bytes);
    }
  }
#endif

} // namespace bdft_tests
