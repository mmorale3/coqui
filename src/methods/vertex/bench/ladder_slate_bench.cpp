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
 * ladder_slate_bench -- the SLATE distributed dense pipeline of the pair-space ladder,
 * on SYNTHETIC data (notes/ladder_opt_spec.md increment B0).
 *
 * WHY A MINI-APP. The ladder's cost is one LAPACK call: at kp666 a single (q, nu)
 * resolvent is D = 26 136 and takes 2 684 s on one rank -- 88.6% of the ladder, which is
 * itself 93% of the qpGW run (notes/ladder_profiling_results.md sections 1.1a, 1.3). The
 * fix is to distribute that solve, and the design space (ranks per solve grid, tile size,
 * concurrent grids, ranks per node) is far cheaper to scan without coqui state in the way.
 * This target carries NO physics: it reproduces the SHAPES, the BLOCK STRUCTURE, the
 * arithmetic pattern and the conditioning of the real problem, nothing else.
 *
 * WHAT IS FAITHFUL TO THE REAL KERNEL
 *  - D = nk * nc2 with the (nc2 x nc2) block tiling over (ikp, ik) pairs; presets are the
 *    measured meshes (kp444: nk 64, nc2 121, N_m 243; kp666: nk 216, nc2 121, N_m 245).
 *  - the rung block is the real two-gemm leg contraction wb = U1(ik)^T . W . U2(ikp) with
 *    (N_m x nc2) legs and an (N_m x N_m) rung, then the Xh row factor Cb(ikp) . wb -- so
 *    the build's flop count and its per-block memory pattern are the production ones, and
 *    each tile owner RECOMPUTES its legs instead of communicating them (spec B.3).
 *  - the resolvent is I - XK, conditioned to sigma_max(XK) = 0.5 (the measured lambda_max
 *    at kp444 iteration 1 is 0.50), so the solve is as well-posed as the real one.
 *  - the RHS is the (D x N_m) block the production path solves, and the projections are
 *    the same two gemms.
 *  - the lambda_max watchdog is the same 20-step power iteration with the same break test.
 * The synthetic legs come from a deterministic hash of the global indices, so EVERY
 * distribution (any g, any tile size, serial reference) builds bitwise the same matrix --
 * which is what makes the exactness leg meaningful.
 *
 * PHASES (each timed separately, min/mean/max over the world):
 *   build   distributed build of the owned tiles (legs + two gemms + the Xh factor)
 *   lam     20-step power iteration on -XK (distributed gemv via SLATE multiply)
 *   rhs     XKXD = (-XK) . XD, one (D,D)x(D,N_m) distributed gemm
 *   solve   slate lu_solve: LU factor + multi-RHS solve, THE phase
 *   proj    Pl = -spin * Dstack^dag . Z, one (N_m,D)x(D,N_m) distributed gemm
 *   resid   optional ||(I-XK)Z - XKXD|| / ||XKXD|| (needs a copy of the matrix)
 *
 * Usage (mpirun -np NP ladder_slate_bench [options]):
 *   --preset tiny|kp444|kp666   shape preset            (default kp444)
 *   --nk / --nc2 / --Nm         override the preset shapes
 *   --g N                       ranks per solve grid; world splits into NP/g grids
 *   --nb N                      SLATE tile size (default nc2)
 *   --nsolve N                  solves per grid (default 1)
 *   --check                     serial LAPACK reference on grid 0 (small D only)
 *   --residual                  report ||(I-XK)Z - RHS||/||RHS|| (keeps a second matrix)
 *   --verbose                   per-grid lines
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/resource.h>

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/slate_ops.hpp"

namespace mpi3 = boost::mpi3;

namespace {

  using math::nda::make_distributed_array;
  template<int N> using shape_t = std::array<long, N>;
  // SLATE needs Fortran ordering for the LU operand (slate_ops::lu_solve static_asserts it)
  using dmat_t = nda::array<ComplexType, 2, nda::F_layout>;

  double rss_gb() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return double(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
    return double(ru.ru_maxrss) / (1024.0 * 1024.0);
#endif
  }

  struct watch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double lap() {
      auto t1 = std::chrono::steady_clock::now();
      double d = std::chrono::duration<double>(t1 - t0).count();
      t0 = t1;
      return d;
    }
  };

  /** splitmix64: deterministic, index-addressable, no state -- the same value on every
   *  rank for the same global index, which is what makes any distribution reproduce the
   *  same matrix (and the serial reference comparable). */
  inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  inline double unit(uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t h = mix(a * 0x100000001b3ULL + mix(b * 0x9e3779b1ULL + mix(c)));
    return 2.0 * (double(h >> 11) / double(1ULL << 53)) - 1.0;   // (-1, 1)
  }
  inline ComplexType cunit(uint64_t a, uint64_t b, uint64_t c) {
    return ComplexType(unit(a, b, c), unit(a, b, c + 0x5bf03635ULL));
  }

  struct opts {
    long nk = 64, nc2 = 121, Nm = 243;      // kp444 preset
    long g = 1, nb = -1, nsolve = 1;
    bool check = false, residual = false, verbose = false;
    std::string preset = "kp444";
    long D() const { return nk * nc2; }
  };

  /** The synthetic legs of ONE k point, (N_m x nc2), and the shared (N_m x N_m) rung.
   *  Recomputed by whichever rank owns the tile (spec B.3: legs are small, communication
   *  is not worth it). */
  void fill_leg(nda::array<ComplexType, 2> &U, long ik, long tag) {
    const long Nm = U.shape(0), nc2 = U.shape(1);
    const double s = 1.0 / std::sqrt(double(Nm));
    for (long P = 0; P < Nm; ++P)
      for (long p = 0; p < nc2; ++p)
        U(P, p) = s * cunit(uint64_t(ik), uint64_t(P * nc2 + p), uint64_t(tag));
  }
  void fill_rung(nda::array<ComplexType, 2> &W) {
    const long Nm = W.shape(0);
    const double s = 1.0 / std::sqrt(double(Nm));
    for (long P = 0; P < Nm; ++P)
      for (long Q = 0; Q < Nm; ++Q) W(P, Q) = s * cunit(0x9111ULL, uint64_t(P), uint64_t(Q));
  }
  void fill_chi0(nda::array<ComplexType, 2> &C, long ikp) {
    const long nc2 = C.shape(0);
    const double s = 1.0 / std::sqrt(double(nc2));
    for (long r = 0; r < nc2; ++r)
      for (long c = 0; c < nc2; ++c)
        C(r, c) = s * cunit(uint64_t(ikp), uint64_t(r * nc2 + c), 0x7ab1ULL);
  }

  /** Build this rank's OWNED part of XK (not I - XK: the caller conditions and adds I).
   *  Loops the (ikp, ik) rung-block pairs that intersect the local row/column ranges,
   *  computes the full (nc2 x nc2) block exactly as the production kernel does, and copies
   *  the intersection. Returns the number of blocks touched. */
  long build_owned(nda::MemoryArrayOfRank<2> auto &&loc, long org_r, long org_c,
                   long nc2, long Nm) {
    const long lm = loc.shape(0), ln = loc.shape(1);
    const long ikp0 = org_r / nc2, ikp1 = (org_r + lm - 1) / nc2;
    const long ik0 = org_c / nc2, ik1 = (org_c + ln - 1) / nc2;
    nda::array<ComplexType, 2> W(Nm, Nm), U1(Nm, nc2), U2(Nm, nc2);
    nda::array<ComplexType, 2> WU2(Nm, nc2), wb(nc2, nc2), Cb(nc2, nc2), blk(nc2, nc2);
    fill_rung(W);
    // legs of the columns this rank touches, cached across the row loop
    std::vector<nda::array<ComplexType, 2>> Ucol;
    Ucol.reserve(size_t(ik1 - ik0 + 1));
    for (long ik = ik0; ik <= ik1; ++ik) {
      nda::array<ComplexType, 2> U(Nm, nc2);
      fill_leg(U, ik, 1);
      Ucol.push_back(std::move(U));
    }
    long nblk = 0;
    for (long ikp = ikp0; ikp <= ikp1; ++ikp) {
      fill_leg(U2, ikp, 2);
      fill_chi0(Cb, ikp);
      nda::blas::gemm(ComplexType(1.0), W, U2, ComplexType(0.0), WU2);                                  // (Nm x nc2)
      for (long ik = ik0; ik <= ik1; ++ik) {
        U1 = Ucol[size_t(ik - ik0)];
        nda::blas::gemm(ComplexType(1.0), nda::transpose(U1), WU2,
                        ComplexType(0.0), wb);                // (nc2 x nc2) rung
        nda::blas::gemm(ComplexType(1.0), Cb, wb, ComplexType(0.0), blk);                                // the Xh row factor
        ++nblk;
        // copy the part of this block that lives in the local window
        const long r0 = std::max(org_r, ikp * nc2), r1 = std::min(org_r + lm, (ikp + 1) * nc2);
        const long c0 = std::max(org_c, ik * nc2), c1 = std::min(org_c + ln, (ik + 1) * nc2);
        for (long i = r0; i < r1; ++i)
          for (long j = c0; j < c1; ++j)
            loc(i - org_r, j - org_c) = blk(i - ikp * nc2, j - ik * nc2);
      }
    }
    return nblk;
  }

}  // namespace

int main(int argc, char **argv) {
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  setup_loggers(world.root(), 2, 0);

  opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--preset") o.preset = next();
    else if (a == "--nk") o.nk = std::stol(next());
    else if (a == "--nc2") o.nc2 = std::stol(next());
    else if (a == "--Nm") o.Nm = std::stol(next());
    else if (a == "--g") o.g = std::stol(next());
    else if (a == "--nb") o.nb = std::stol(next());
    else if (a == "--nsolve") o.nsolve = std::stol(next());
    else if (a == "--check") o.check = true;
    else if (a == "--residual") o.residual = true;
    else if (a == "--verbose") o.verbose = true;
    else utils::check(false, "ladder_slate_bench: unknown option {}", a);
  }
  if (o.preset == "tiny")       { o.nk = 4;   o.nc2 = 121; o.Nm = 60;  }
  else if (o.preset == "kp444") { o.nk = 64;  o.nc2 = 121; o.Nm = 243; }
  else if (o.preset == "kp666") { o.nk = 216; o.nc2 = 121; o.Nm = 245; }
  else utils::check(o.preset == "custom", "ladder_slate_bench: unknown preset {}", o.preset);
  const long D = o.D(), nc2 = o.nc2, Nm = o.Nm;
  if (o.nb <= 0) o.nb = nc2;

  utils::check(world.size() % o.g == 0,
               "ladder_slate_bench: world size {} is not a multiple of g = {}.",
               world.size(), o.g);
  const long ngrid = world.size() / o.g;

  // one solve grid per contiguous block of g ranks (contiguous => same node under the
  // usual block placement, which is what we want for the intra-node cooperative solve)
  const int gid = int(world.rank() / o.g), grank = int(world.rank() % o.g);
  auto gcomm = world.split(gid, grank);
  const long px = utils::find_proc_grid_min_diff(o.g, D, D), py = o.g / px;

  app_log(1, "\n=== ladder_slate_bench (notes/ladder_opt_spec.md B0) ===");
  app_log(1, "  preset {} : nk = {}, nc2 = {}, D = nk*nc2 = {}, N_m = {}", o.preset, o.nk,
          nc2, D, Nm);
  app_log(1, "  world = {} ranks; g = {} ranks/solve-grid => {} concurrent grids; proc "
             "grid {} x {}; tile nb = {}; solves per grid = {}",
          world.size(), o.g, ngrid, px, py, o.nb, o.nsolve);
  const double gb = 16.0 / (1024.0 * 1024.0 * 1024.0);
  app_log(1, "  per-rank dense footprint: matrix (D*D/g) = {:.3f} GB, RHS+solution "
             "(2*D*N_m/g) = {:.3f} GB{}",
          double(D) * double(D) * gb / double(o.g),
          2.0 * double(D) * double(Nm) * gb / double(o.g),
          o.residual ? ", plus a second matrix for the residual leg" : "");

  // Tile sizes. The contracted dimensions must carry the SAME tile size in every operand,
  // so all (D, N_m)-class matrices share (nb x ncb) and the (N_m, N_m) output is (ncb x
  // ncb). ncb is chosen small enough that make_distributed_array's cap (shape/grid) never
  // shrinks it, which would silently break the tiling match.
  const long ncb = std::max(1l, Nm / std::max(px, py));
  auto A = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, D},
                                          shape_t<2>{o.nb, o.nb}, true);
  auto B = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, Nm},
                                          shape_t<2>{o.nb, ncb}, false);
  auto XD = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, Nm},
                                           shape_t<2>{o.nb, ncb}, false);
  auto Ds = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, Nm},
                                           shape_t<2>{o.nb, ncb}, false);
  auto Pl = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{Nm, Nm},
                                           shape_t<2>{ncb, ncb}, false);
  // the power-iteration iterate. A (D, 1) array cannot live on a 2D grid (the factory
  // requires shape >= grid on every axis), so it carries py IDENTICAL columns: the
  // arithmetic is one gemv per column, the norm of one column is the total over py, and
  // the extra columns cost 20 * 2 * D^2 * (py - 1) flops -- per mille of the D^3 solve.
  auto V1 = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, py},
                                           shape_t<2>{o.nb, 1}, false);
  auto V2 = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, py},
                                           shape_t<2>{o.nb, 1}, false);
  // the residual leg keeps a second (D,D) and a copy of the RHS -- off by default
  std::optional<decltype(A)> Acopy;
  std::optional<decltype(B)> Bsave;

  const long org_r = A.origin()[0], org_c = A.origin()[1];
  double t_build = 0, t_lam = 0, t_rhs = 0, t_solve = 0, t_proj = 0, t_resid = 0;
  double lam_last = 0.0, resid_last = -1.0;
  const double rss0 = rss_gb();

  for (long is = 0; is < o.nsolve; ++is) {
    watch w;
    // ---- build the owned tiles of XK ------------------------------------------------
    {
      auto loc = A.local();
      build_owned(loc, org_r, org_c, nc2, Nm);
    }
    gcomm.barrier();
    t_build += w.lap();

    // ---- condition: scale so sigma_max(XK) = 0.5, measured by the power iteration -----
    // (synthetic-data bookkeeping, not part of the production pipeline -- timed apart)
    // ---- lambda_max: the 20-step power iteration, distributed ------------------------
    {
      const double seed = 1.0 / std::sqrt(double(D));
      auto v1 = V1.local();
      v1() = ComplexType(seed, 0.0);
      double lam = 0.0, lam_prev = -1.0;
      for (int it = 0; it < 20; ++it) {
        math::nda::slate_ops::multiply(A, V1, V2);
        double n2 = 0.0;
        for (auto const &z : V2.local()) n2 += std::norm(z);
        n2 = gcomm.all_reduce_value(n2, std::plus<>{});
        // every rank of a row of the grid holds the same rows: correct for the replication
        n2 /= double(py);
        lam = std::sqrt(n2);
        if (lam <= 0.0) break;
        auto l2 = V2.local();
        for (auto &z : l2) z = z / lam;
        V1.local() = l2;
        if (lam_prev > 0.0 and std::abs(lam - lam_prev) <= 1e-3 * lam) break;
        lam_prev = lam;
      }
      lam_last = lam;
      gcomm.barrier();
      t_lam += w.lap();
      // condition the synthetic operator onto the measured lambda_max = 0.5
      const double sc = (lam > 0.0) ? 0.5 / lam : 1.0;
      auto la = A.local();
      for (auto &z : la) z = -z * sc;                 // Rm <- -XK, the production sign
      w.lap();
    }

    // ---- RHS: XKXD = (-XK) . XD, and the +I that completes the resolvent -------------
    {
      auto xd = XD.local();
      for (long i = 0; i < xd.shape(0); ++i)
        for (long j = 0; j < xd.shape(1); ++j)
          xd(i, j) = cunit(uint64_t(XD.origin()[0] + i), uint64_t(XD.origin()[1] + j), 0x31ULL);
      Ds.local() = xd;
      math::nda::slate_ops::multiply(A, XD, B);       // B = (-XK) . XD
      // resolvent: Rm = I - XK (A already holds -XK)
      auto la = A.local();
      for (long i = 0; i < la.shape(0); ++i) {
        const long gi = org_r + i;
        if (gi >= org_c and gi < org_c + la.shape(1)) la(i, gi - org_c) += ComplexType(1.0);
      }
      if (o.residual) {
        if (not Acopy.has_value()) {
          Acopy.emplace(make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py},
                                                       shape_t<2>{D, D},
                                                       shape_t<2>{o.nb, o.nb}, true));
          Bsave.emplace(make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py},
                                                       shape_t<2>{D, Nm},
                                                       shape_t<2>{o.nb, ncb}, false));
        }
        Acopy.value().local() = la;          // I - XK, before the LU destroys it
        Bsave.value().local() = B.local();   // the RHS, before it becomes the solution
      }
      gcomm.barrier();
      t_rhs += w.lap();
    }

    // ---- THE solve: distributed LU factor + multi-RHS solve (never an inverse) --------
    {
      long info = math::nda::slate_ops::lu_solve(A, B);
      utils::check(info == 0, "ladder_slate_bench: lu_solve info = {}", info);
      gcomm.barrier();
      t_solve += w.lap();
    }

    // ---- projection to (N_m, N_m) ----------------------------------------------------
    {
      math::nda::slate_ops::multiply(ComplexType(-2.0), math::nda::transpose(Ds), B,
                                     ComplexType(0.0), Pl);
      gcomm.barrier();
      t_proj += w.lap();
    }

    // ---- optional residual ||(I-XK)Z - RHS|| / ||RHS|| -------------------------------
    if (o.residual) {
      // ||(I - XK) Z - RHS|| / ||RHS||, with both operands kept from before the solve
      auto R = make_distributed_array<dmat_t>(gcomm, shape_t<2>{px, py}, shape_t<2>{D, Nm},
                                              shape_t<2>{o.nb, ncb}, false);
      math::nda::slate_ops::multiply(Acopy.value(), B, R);
      double num = 0.0, den = 0.0;
      auto rl = R.local();
      auto bl = Bsave.value().local();
      for (long i = 0; i < rl.shape(0); ++i)
        for (long j = 0; j < rl.shape(1); ++j) {
          num += std::norm(rl(i, j) - bl(i, j));
          den += std::norm(bl(i, j));
        }
      num = gcomm.all_reduce_value(num, std::plus<>{});
      den = gcomm.all_reduce_value(den, std::plus<>{});
      resid_last = std::sqrt(num / std::max(den, 1e-300));
      gcomm.barrier();
      t_resid += w.lap();
    }
  }

  const double rss1 = rss_gb();
  // ---- report: min/mean/max over the world ------------------------------------------
  auto stat = [&](double v, const char *nm) {
    const double vmin = world.all_reduce_value(v, boost::mpi3::min<>{});
    const double vmax = world.all_reduce_value(v, boost::mpi3::max<>{});
    const double vsum = world.all_reduce_value(v, std::plus<>{});
    app_log(1, "  [bench] {:<10} min/mean/max = {:12.4f} / {:12.4f} / {:12.4f} s", nm,
            vmin, vsum / double(world.size()), vmax);
    return vsum / double(world.size());
  };
  const double n = double(o.nsolve);
  app_log(1, "  --- phases, cumulative over {} solve(s) per grid ---", o.nsolve);
  stat(t_build, "build");
  stat(t_lam, "lam");
  stat(t_rhs, "rhs");
  const double solve_mean = stat(t_solve, "solve");
  stat(t_proj, "proj");
  if (o.residual) stat(t_resid, "resid");
  const double rmax = world.all_reduce_value(rss1, boost::mpi3::max<>{});
  const double rmin = world.all_reduce_value(rss0, boost::mpi3::min<>{});
  const double tsolve = solve_mean / std::max(n, 1.0);
  // complex LU + solve flops: (8/3)D^3 for getrf, 8*D^2*nrhs for getrs (real-equivalent)
  const double flops = (8.0 / 3.0) * std::pow(double(D), 3.0)
                     + 8.0 * double(D) * double(D) * double(Nm);
  app_log(1, "  [bench] MaxRSS GB: entry(min) {:.3f} -> exit(max) {:.3f}", rmin, rmax);
  app_log(1, "  [bench] time per solve = {:.4f} s ; grid Gflop/s = {:.2f} ; per-rank "
             "Gflop/s = {:.3f} ; sigma_max(XK) as built = {:.6f} (then scaled to 0.5, the "
             "measured conditioning)",
          tsolve, flops / std::max(tsolve, 1e-30) / 1e9,
          flops / std::max(tsolve, 1e-30) / 1e9 / double(o.g), lam_last);
  // ONE machine-parseable row per design point -- this is what the scan table is built
  // from (scalars only: rusty's bundled fmt cannot format containers)
  app_log(1, "  [bench] SCAN {} np {} g {} grids {} nb {} D {} Nm {} nsolve {} build "
             "{:.4f} lam {:.4f} rhs {:.4f} solve {:.4f} proj {:.4f} tsolve {:.4f} "
             "gbrank {:.3f} gfrank {:.3f} solveshr {:.2f}",
          o.preset, world.size(), o.g, ngrid, o.nb, D, Nm, o.nsolve,
          t_build / n, t_lam / n, t_rhs / n, solve_mean / n, t_proj / n, tsolve, rmax,
          flops / std::max(tsolve, 1e-30) / 1e9 / double(o.g),
          3600.0 * double(ngrid * o.nsolve) / std::max(solve_mean, 1e-30));
  app_log(1, "  [bench] throughput: {} concurrent grids x {} solves = {} solves in "
             "{:.4f} s => {:.2f} solves/hour on {} ranks",
          ngrid, o.nsolve, ngrid * o.nsolve, solve_mean,
          3600.0 * double(ngrid * o.nsolve) / std::max(solve_mean, 1e-30), world.size());
  if (o.residual)
    app_log(1, "  [bench] residual ||(I-XK)Z - RHS||/||RHS|| = {:.3e}", resid_last);

  // ---- exactness leg: serial LAPACK on grid 0 (small D only) -------------------------
  if (o.check) {
    utils::check(D <= 4000, "ladder_slate_bench: --check gathers the full matrix; keep "
                            "D <= 4000 (D = {}).", D);
    // rebuild the SAME operator serially and solve with getrf/getrs, then compare
    nda::array<ComplexType, 2> Af(D, D), Bf(D, Nm);
    {
      dmat_t full(D, D);
      full() = ComplexType(0.0);
      build_owned(full, 0, 0, nc2, Nm);
      // the same conditioning the distributed path applied
      double lam = 0.0, lam_prev = -1.0;
      nda::array<ComplexType, 1> pv(D), pw(D);
      pv() = ComplexType(1.0 / std::sqrt(double(D)), 0.0);
      for (int it = 0; it < 20; ++it) {
        nda::blas::gemv(ComplexType(1.0), full, pv, ComplexType(0.0), pw);
        double n2 = 0.0;
        for (long i = 0; i < D; ++i) n2 += std::norm(pw(i));
        lam = std::sqrt(n2);
        if (lam <= 0.0) break;
        for (long i = 0; i < D; ++i) pv(i) = pw(i) / lam;
        if (lam_prev > 0.0 and std::abs(lam - lam_prev) <= 1e-3 * lam) break;
        lam_prev = lam;
      }
      const double sc = (lam > 0.0) ? 0.5 / lam : 1.0;
      nda::array<ComplexType, 2> XKm(D, D);
      for (long i = 0; i < D; ++i)
        for (long j = 0; j < D; ++j) XKm(i, j) = -full(i, j) * sc;   // -XK
      nda::array<ComplexType, 2> rhs(D, Nm);
      for (long i = 0; i < D; ++i)
        for (long j = 0; j < Nm; ++j) rhs(i, j) = cunit(uint64_t(i), uint64_t(j), 0x31ULL);
      nda::blas::gemm(ComplexType(1.0), XKm, rhs, ComplexType(0.0), Bf);                                  // (-XK).XD
      Af = XKm;
      for (long i = 0; i < D; ++i) Af(i, i) += ComplexType(1.0);      // I - XK
    }
    nda::matrix<ComplexType> Am(D, D);
    Am() = Af;
    nda::array<int, 1> ipiv(D);
    int info = nda::lapack::getrf(Am, ipiv);
    utils::check(info == 0, "ladder_slate_bench: serial getrf info = {}", info);
    nda::basic_array<ComplexType, 2, nda::F_layout, 'A', nda::heap<>> Bser(D, Nm);
    Bser() = Bf;
    info = nda::lapack::getrs(Am, Bser, ipiv);
    utils::check(info == 0, "ladder_slate_bench: serial getrs info = {}", info);
    // compare against the distributed solution held in B
    double dmax = 0.0, bmax = 0.0;
    auto bl = B.local();
    for (long i = 0; i < bl.shape(0); ++i)
      for (long j = 0; j < bl.shape(1); ++j) {
        const ComplexType s = Bser(B.origin()[0] + i, B.origin()[1] + j);
        dmax = std::max(dmax, std::abs(bl(i, j) - s));
        bmax = std::max(bmax, std::abs(s));
      }
    dmax = world.all_reduce_value(dmax, boost::mpi3::max<>{});
    bmax = world.all_reduce_value(bmax, boost::mpi3::max<>{});
    app_log(1, "  [bench] EXACTNESS vs serial LAPACK getrf/getrs: max |dist - serial| = "
               "{:.3e} , max |serial| = {:.3e} , relative = {:.3e}",
            dmax, bmax, dmax / std::max(bmax, 1e-300));
  }
  app_log(1, "=== ladder_slate_bench done ===\n");
  return 0;
}
