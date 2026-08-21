/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * RW-2 GATE RW-2-a (notes/rw_real_axis_w_spec.md): the TOY gate of the
 * spectral-quadrature W^c pole representation.
 *
 *   (i)  the quadrature of the exact Im W^c of an analytic two-pole Drude model
 *        reproduces W^c(z) at OFF-NODE test points to quadrature-error class,
 *        AND its residue matrices are Hermitian positive semi-definite at the
 *        +Omega poles. This is what PINS the +-Omega / residue sign convention
 *        derived in wc_spectral.hpp: the wrong overall sign is a 200 % error and
 *        the wrong pair assignment flips the definiteness.
 *
 *   (ii) on a METALLIC toy -- no particle-hole gap, Drude-like low-frequency
 *        weight -- the production least-squares pole fit reproduces the
 *        cancellation catastrophe measured on SVO (notes/qpgw_metal_mode_m0.md
 *        section 8b: Sabs/|Sigma^c| = 1e4-1e5) while the spectral quadrature
 *        gives Sabs/|Sigma^c| = O(1-10).
 *
 * Nothing here needs FINUFFT or a mean-field fixture: both cases feed the pure
 * quadrature math of wc_spectral.hpp from a closed-form W^c, which is the point
 * -- the sign convention is pinned against ANALYTIC values, not against another
 * numerical chain.
 *
 * ---------------------------------------------------------------------------
 * THE MODEL. Take R_m, m = 1..M, real symmetric positive semi-definite (Naux x
 * Naux) and Lorentzian-broadened bosonic poles at +-Omega_m with width gamma:
 *
 *   Im W^c_PQ(Omega) = -sum_m R_m,PQ [ L(Omega - Omega_m) - L(Omega + Omega_m) ],
 *                      L(x) = gamma / (x^2 + gamma^2)                        (odd in Omega)
 *
 * The Hilbert transform of a Lorentzian has a closed form: for Im z > 0,
 *   (1/pi) int dOmega' L(Omega' - a) / (Omega' - z) = -1 / (z - a + i gamma)
 * [residues: writing L(x)/pi = (1/2 pi i)[1/(x - i gamma) - 1/(x + i gamma)],
 *  the first term's contour integral over the upper half plane vanishes because
 *  the pole at a + i gamma and the pole at z contribute equal and opposite
 *  residues, and the second leaves 2 pi i / (z - a + i gamma)].
 * Hence, from W^c(z) = (1/pi) int dOmega' Im W^c(Omega')/(Omega' - z),
 *
 *   W^c(z) = sum_m R_m [ 1/(z - Omega_m + i gamma) - 1/(z + Omega_m + i gamma) ].
 *
 * At z = i nu that is negative definite (screening lowers W below v), and at
 * gamma -> 0 it collapses onto the exact pole pair with residues (+R_m, -R_m) --
 * which is what the quadrature must reproduce.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
#include "methods/SCF/wc_spectral.hpp"
#include "methods/SCF/sigma_route_b.hpp"

namespace bdft_tests {

  using namespace methods;
  using cval_t = std::complex<double>;

  namespace rw2toy {

    constexpr long   NAUX  = 3;
    constexpr double GAMMA = 0.01;              // Lorentzian width, a.u. (gate (i))

    struct drude_t {
      std::vector<double> Om;                              // pole energies > 0
      std::vector<nda::array<cval_t, 2>> R;                // psd HERMITIAN residue matrices
      double gam = GAMMA;                                  // Lorentzian width
    };

    /**
     * A psd HERMITIAN matrix v v^dag + 0.35 w w^dag, deterministic. `im` scales an imaginary
     * part into the generating vectors: im = 0 gives a real symmetric matrix (Im W^c comes
     * out (P,Q)-SYMMETRIC, the qe_lih222 regime), im != 0 gives a genuinely complex Hermitian
     * one, whose antisymmetric imaginary part makes Im W^c NON-symmetric -- the SVO regime,
     * where the -Omega residue must be the TRANSPOSE.
     *
     * The model stays a legal retarded response for ANY Hermitian R: with
     * W(z) = sum_m R_m [1/(z - Om_m + i g) - 1/(z + Om_m + i g)],
     * W_PQ(-Omega) = conj(W_QP(Omega)) holds iff conj(R_PQ) = R_QP, i.e. iff R is Hermitian.
     */
    inline nda::array<cval_t, 2> psd(double a, double b, double c, double im = 0.0) {
      nda::array<cval_t, 2> M(NAUX, NAUX);
      const cval_t v[NAUX] = {cval_t(a, 0.0), cval_t(b, 0.4 * im), cval_t(c, -0.7 * im)};
      const cval_t w[NAUX] = {cval_t(b, -0.3 * im), cval_t(-c, 0.0), cval_t(a, 0.5 * im)};
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q)
          M(P, Q) = v[P] * std::conj(v[Q]) + 0.35 * w[P] * std::conj(w[Q]);
      return M;
    }

    /** The exact W^c_PQ(z), Im z >= 0 (declared first: Im W^c is read off it). */
    inline cval_t W_exact_fwd(drude_t const &d, long P, long Q, cval_t z) {
      cval_t acc(0.0, 0.0);
      const cval_t ig(0.0, d.gam);
      for (size_t m = 0; m < d.Om.size(); ++m)
        acc += d.R[m](P, Q) * (1.0 / (z - d.Om[m] + ig) - 1.0 / (z + d.Om[m] + ig));
      return acc;
    }

    /** Im W^c_PQ(Omega) on the real axis -- NOT (P,Q)-symmetric unless R is real. */
    inline double ImW(drude_t const &d, long P, long Q, double O) {
      return W_exact_fwd(d, P, Q, cval_t(O, 0.0)).imag();
    }

    inline cval_t W_exact(drude_t const &d, long P, long Q, cval_t z) {
      return W_exact_fwd(d, P, Q, z);
    }

    /** Uniform bosonic grid Omega_l = (l+1) h with real_freq_grid_t's own trapezoid
     *  weights, then the production quadrature grid (virtual node included). */
    inline wc_spectral::quad_grid_t make_grid(double Omega_max, long N) {
      nda::array<double, 1> Om(N), Ow(N);
      const double h = Omega_max / double(N);
      for (long l = 0; l < N; ++l) Om(l) = h * double(l + 1);
      Ow(0) = 0.5 * (Om(1) - Om(0));
      Ow(N - 1) = 0.5 * (Om(N - 1) - Om(N - 2));
      for (long i = 1; i < N - 1; ++i) Ow(i) = 0.5 * (Om(i + 1) - Om(i - 1));
      return wc_spectral::make_quad_grid(Om, Ow);
    }

    /**
     * Build the spectral pole representation of the model on a given grid, with optional
     * coarsening. Returns the pole list and the residue slabs R(p, P, Q).
     */
    inline void build_rep(drude_t const &d, wc_spectral::quad_grid_t const &g,
                          nda::array<double, 1> const &Om_data, long ntarget,
                          nda::array<double, 1> &om,
                          nda::array<double, 3> &Rp,
                          wc_spectral::bin_plan_t &plan) {
      const long N = g.Om.shape(0);
      nda::array<double, 1> tj(N);
      for (long j = 0; j < N; ++j) {
        double tr = 0.0;
        for (long P = 0; P < NAUX; ++P) tr += ImW(d, P, P, Om_data(g.src(j)));
        tj(j) = -wc_spectral::inv_pi * g.Ow(j) * tr;
      }
      plan = wc_spectral::build_bins(g.Om, tj, ntarget);
      om = wc_spectral::pole_list(plan);
      Rp = nda::array<double, 3>(2 * plan.nbin, NAUX, NAUX);
      for (long b = 0; b < plan.nbin; ++b)
        for (long P = 0; P < NAUX; ++P)
          for (long Q = 0; Q < NAUX; ++Q) {
            double acc = 0.0;
            for (long j = plan.lo(b); j < plan.hi(b); ++j)
              acc += g.Ow(j) * ImW(d, P, Q, Om_data(g.src(j)));
            const double A = -wc_spectral::inv_pi * acc;
            Rp(b, P, Q) = A;
            Rp(plan.nbin + b, Q, P) = -A;      // eq. (B): the -Omega residue is -A^T
          }
    }

  } // namespace rw2toy

  // =========================================================================
  // RW-2-a (i): the sign / residue convention, pinned analytically.
  //
  // Run TWICE: once on a model whose Im W^c is (P,Q)-SYMMETRIC (im = 0, the qe_lih222
  // regime, measured symmetry 1.1e-13) and once where it is NOT (im != 0, the SVO regime,
  // measured 5.3e-01). Only the second discriminates the TRANSPOSE in eq. (B): with -A_j
  // instead of -A_j^T at the -Omega pole the forward map is a 100 %-class error there and
  // an exact identity on the symmetric model.
  // =========================================================================
  static void run_drude_sign(double im, char const *tag) {
    using namespace rw2toy;

    drude_t d;
    d.Om = {0.05, 0.35, 1.20};
    d.R.push_back(psd(0.9, -0.4, 0.2, im));
    d.R.push_back(psd(0.3,  0.7, -0.5, im));
    d.R.push_back(psd(-0.6, 0.2, 0.8, im));

    // dOmega <= gamma/2 (the RW-1 rule at eta -> gamma) and a tail out to 60 a.u.: the
    // Lorentzian tails fall as 1/Omega^2, so truncation is the dominant error here.
    const double Omega_max = 60.0;
    const long   N_O = long(std::ceil(Omega_max / (0.5 * d.gam)));
    auto g = make_grid(Omega_max, N_O);
    nda::array<double, 1> Om_data(N_O);
    { const double h = Omega_max / double(N_O);
      for (long l = 0; l < N_O; ++l) Om_data(l) = h * double(l + 1); }

    // how asymmetric this model's Im W^c actually is -- the quantity the production path
    // logs as "SPECTRAL Im W^c symmetry"
    double sym = 0.0, symamp = 0.0;
    for (long l = 0; l < N_O; l += 7)
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          sym = std::max(sym, std::abs(ImW(d, P, Q, Om_data(l)) - ImW(d, Q, P, Om_data(l))));
          symamp = std::max(symamp, std::abs(ImW(d, P, Q, Om_data(l))));
        }

    nda::array<double, 1> om;
    nda::array<double, 3> Rp;
    wc_spectral::bin_plan_t plan;
    build_rep(d, g, Om_data, /*ntarget*/ 0, om, Rp, plan);      // no coarsening
    REQUIRE(plan.nbin == N_O + 1);
    app_log(2, "[rw2-a {}] Drude model: {} Omega nodes on (0, {}] (+1 virtual), dOmega = "
               "{:.4e}, gamma = {:.4e};  max|ImW_PQ - ImW_QP| / max|ImW| = {:.3e}",
            tag, N_O, Omega_max, Om_data(1) - Om_data(0), GAMMA,
            (symamp > 0.0 ? sym / symamp : 0.0));

    // ---- (a) DEFINITENESS: the SYMMETRIC PART of A_j is psd (eq. C) -------------------
    double worst_neg = 0.0, scale = 0.0;
    for (long b = 0; b < plan.nbin; ++b) {
      nda::matrix<double> M(NAUX, NAUX);
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) M(P, Q) = 0.5 * (Rp(b, P, Q) + Rp(b, Q, P));
      auto [ev, evec] = nda::linalg::eigenelements(M);
      for (long i = 0; i < NAUX; ++i) {
        worst_neg = std::min(worst_neg, ev(i));
        scale = std::max(scale, std::abs(ev(i)));
      }
    }
    app_log(2, "[rw2-a {}] definiteness of sym(A_j) at the +Omega poles: min eigenvalue = "
               "{:.4e}, max |eigenvalue| = {:.4e}, ratio = {:.4e}",
            tag, worst_neg, scale, (scale > 0.0 ? worst_neg / scale : 0.0));
    REQUIRE(worst_neg / scale > -1e-10);        // psd to round-off

    // ---- (b) the forward map, at OFF-NODE test points --------------------------------
    // Imaginary axis (this is the production meter's evaluation set) and two complex
    // points at Im z = 4 gamma, none of which is a quadrature node.
    // z = 0 -- the STATIC Matsubara node -- is handled separately below: it sits ON the
    // real axis, where the dispersion relation carries an extra boundary term i Im W^c(0).
    // Im W^c(0) is antisymmetric in (P,Q) by the reflection relation of (2), so it vanishes
    // identically on the symmetric model and does NOT on the asymmetric one; no
    // real-residue pole set can carry it (it is a logarithm, not a pole). Measured and
    // reported; the production Lehmann meter sees exactly this at its own nu = 0 node.
    std::vector<cval_t> zs;
    for (double nu : {0.013, 0.077, 0.31, 1.7, 9.3}) zs.push_back(cval_t(0.0, nu));
    zs.push_back(cval_t(0.173, 4.0 * GAMMA));
    zs.push_back(cval_t(-0.612, 4.0 * GAMMA));

    double worst = 0.0, worst_flip = 0.0, worst_notr = 0.0, amp = 0.0;
    for (auto z : zs)
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          cval_t rec(0.0, 0.0), rec_notr(0.0, 0.0);
          for (long p = 0; p < om.shape(0); ++p) rec += Rp(p, P, Q) / (z - om(p));
          // the WRONG convention: -A_j (not -A_j^T) at the -Omega member of every pair
          for (long b = 0; b < plan.nbin; ++b) {
            rec_notr += Rp(b, P, Q) / (z - om(b));
            rec_notr -= Rp(b, P, Q) / (z - om(plan.nbin + b));
          }
          const cval_t ex = W_exact(d, P, Q, z);
          worst = std::max(worst, std::abs(rec - ex));
          worst_flip = std::max(worst_flip, std::abs(-rec - ex));   // the wrong overall sign
          worst_notr = std::max(worst_notr, std::abs(rec_notr - ex));
          amp = std::max(amp, std::abs(ex));
        }
    app_log(2, "[rw2-a {}] forward map over {} off-node z x {} elements: worst |quad - exact| "
               "= {:.4e}, max|exact| = {:.4e} -> relative {:.4e}   [SIGN-FLIPPED rep: {:.4e} "
               "-> {:.4e};  NO-TRANSPOSE rep (-A instead of -A^T): {:.4e} -> {:.4e}]",
            tag, zs.size(), NAUX * NAUX, worst, amp, worst / amp, worst_flip,
            worst_flip / amp, worst_notr, worst_notr / amp);
    // Quadrature-error class. On the symmetric model (Im W^c(0) = 0) the trapezoid is
    // SECOND order in dOmega -- measured 7.7e-04 at dOmega = gamma/2 and 9.8e-07 at
    // gamma/20. On the asymmetric one the antisymmetric part of Im W^c does not vanish at
    // Omega -> 0, so the [0, Omega_0] segment (where the virtual node models Im W^c as
    // linear) drops the order to FIRST -- measured 4.9e-02, 5.4e-03, 5.4e-04 as dOmega
    // falls by 10 each time. Both are honest quadrature statements, and the SIGN is pinned
    // by the factor between the columns, not by the absolute value.
    REQUIRE(worst / amp < (sym / symamp > 1e-6 ? 1e-1 : 1e-3));
    REQUIRE(worst_flip / amp > 1.0);
    // The transpose only bites when Im W^c is asymmetric; on the symmetric model the two
    // conventions are the SAME representation and the check would be vacuous.
    if (sym / symamp > 1e-6) REQUIRE(worst_notr > 3.0 * worst);
    else                     REQUIRE(std::abs(worst_notr - worst) < 1e-14 * amp);

    // the static node, reported
    {
      double d0 = 0.0, a0 = 0.0, imex = 0.0;
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          cval_t rec(0.0, 0.0);
          for (long p = 0; p < om.shape(0); ++p) rec += Rp(p, P, Q) / (cval_t(0.0, 0.0) - om(p));
          const cval_t ex = W_exact(d, P, Q, cval_t(0.0, 0.0));
          d0 = std::max(d0, std::abs(rec - ex));
          a0 = std::max(a0, std::abs(ex));
          imex = std::max(imex, std::abs(ex.imag()));
        }
      app_log(2, "[rw2-a {}] STATIC node z = 0: relative deviation = {:.4e}; the exact "
                 "W^c(0) carries |Im| up to {:.4e} (= the boundary term i Im W^c(0), which "
                 "is antisymmetric and outside the real-residue pole class)",
              tag, d0 / a0, imex);
    }

    // ---- (c) the representation must be NEGATIVE definite and HERMITIAN at i nu --------
    {
      const cval_t znu(0.0, 0.013);
      nda::matrix<cval_t> W0(NAUX, NAUX);
      double anti = 0.0, w0amp = 0.0;
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          cval_t rec(0.0, 0.0);
          for (long p = 0; p < om.shape(0); ++p) rec += Rp(p, P, Q) / (znu - om(p));
          W0(P, Q) = rec;
        }
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          anti = std::max(anti, std::abs(W0(P, Q) - std::conj(W0(Q, P))));
          w0amp = std::max(w0amp, std::abs(W0(P, Q)));
        }
      auto [ev, evec] = nda::linalg::eigenelements(W0);
      double emax = -1e300;
      for (long i = 0; i < NAUX; ++i) emax = std::max(emax, ev(i));
      app_log(2, "[rw2-a {}] W^c(i nu = 0.013): largest eigenvalue = {:.6e} (screening must "
                 "lower W below the bare v, so this is < 0);  max|W - W^dag| / max|W| = "
                 "{:.3e} -- Hermiticity is what the TRANSPOSE in eq. (B) buys",
              tag, emax, anti / w0amp);
      REQUIRE(emax < 0.0);
      REQUIRE(anti / w0amp < 1e-12);       // the transpose is what makes this Hermitian
    }

    // ---- (d) coarsening keeps both properties ----------------------------------------
    {
      nda::array<double, 1> om_c;
      nda::array<double, 3> Rc;
      wc_spectral::bin_plan_t pc;
      build_rep(d, g, Om_data, /*ntarget*/ 64, om_c, Rc, pc);
      double w2 = 0.0, wneg = 0.0, sc = 0.0;
      for (auto z : zs)
        for (long P = 0; P < NAUX; ++P)
          for (long Q = 0; Q < NAUX; ++Q) {
            cval_t rec(0.0, 0.0);
            for (long p = 0; p < om_c.shape(0); ++p) rec += Rc(p, P, Q) / (z - om_c(p));
            w2 = std::max(w2, std::abs(rec - W_exact(d, P, Q, z)));
          }
      for (long b = 0; b < pc.nbin; ++b) {
        nda::matrix<double> M(NAUX, NAUX);
        for (long P = 0; P < NAUX; ++P)
          for (long Q = 0; Q < NAUX; ++Q) M(P, Q) = 0.5 * (Rc(b, P, Q) + Rc(b, Q, P));
        auto [ev, evec] = nda::linalg::eigenelements(M);
        for (long i = 0; i < NAUX; ++i) {
          wneg = std::min(wneg, ev(i)); sc = std::max(sc, std::abs(ev(i)));
        }
      }
      app_log(2, "[rw2-a {}] coarsened to {} bins (target 64, worst relative bin width "
                 "{:.3e}): forward-map relative dev = {:.4e}; min eigenvalue / max = {:.4e}",
              tag, pc.nbin, pc.width_worst, w2 / amp, wneg / sc);
      REQUIRE(pc.nbin <= 64);
      REQUIRE(wneg / sc > -1e-10);       // psd survives the merge, by construction
    }
  }

  TEST_CASE("wc_spectral_drude_sign", "[methods][qpgw][rw2][spectral]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() != 1) return;
    run_drude_sign(0.0, "symmetric ");     // Im W^c symmetric  -- the qe_lih222 regime
    run_drude_sign(0.6, "asymmetric");     // Im W^c asymmetric -- the SVO regime
  }

  // =========================================================================
  // RW-2-a (ii): the cancellation meter on a metallic toy.
  // =========================================================================
  TEST_CASE("wc_spectral_metal_cancellation", "[methods][qpgw][rw2][spectral]") {
    using namespace rw2toy;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() != 1) return;

    constexpr double HA2EV = 27.211386245988;
    const double beta = 1000.0, wmax = 3.0, mu = 0.0;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");

    // --- the metallic W^c: Drude-like weight at 0.02 a.u. (0.54 eV) plus a plasmon ------
    // gamma is taken WELL BELOW the evaluation offset pi/beta = 3.14e-3 so that the
    // gamma -> 0 limit -- the EXACT 6-pole representation (+-Omega_m, +-R_m) -- is a valid
    // reference at the probe energies. That reference is what makes this meter comparable
    // to the M-0b measurement, which quotes Sabs against the PHYSICAL |Sigma^c| and not
    // against route B's own (wrong, far too large) value.
    drude_t d;
    d.gam = 0.001;
    d.Om = {0.02, 0.09, 0.85};
    d.R.push_back(psd(0.7, 0.25, -0.15));
    d.R.push_back(psd(0.4, -0.5,  0.30));
    d.R.push_back(psd(0.2,  0.6,  0.55));

    // --- the metallic internal spectrum: a band straddling mu with NO gap ---------------
    const long nJ = 60;
    nda::array<double, 1> epsJ(nJ), fJ(nJ);
    nda::array<double, 2> Apair(nJ, NAUX);
    for (long J = 0; J < nJ; ++J) {
      epsJ(J) = -0.30 + 0.60 * double(J) / double(nJ - 1);          // uniform, crosses mu
      fJ(J) = 1.0 / (1.0 + std::exp(beta * (epsJ(J) - mu)));
      const double t = double(J) / double(nJ - 1);
      Apair(J, 0) = 0.9 - 0.4 * t;
      Apair(J, 1) = 0.2 + 0.6 * t;
      Apair(J, 2) = -0.3 + 0.8 * t * t;
    }
    // E_PH of this spectrum is the level spacing -- the metallic regime the M-0b note
    // identified as the one where the support constraint is set by the mesh, not physics.
    double E_PH = 1e300;
    for (long J = 0; J < nJ; ++J)
      for (long K = 0; K < nJ; ++K)
        if (fJ(J) >= 0.5 and fJ(K) < 0.5) E_PH = std::min(E_PH, epsJ(K) - epsJ(J));
    app_log(2, "[rw2-a-ii] metallic toy: nJ = {}, E_PH = {:.6e} a.u. ({:.4g} eV), "
               "10 pi/beta = {:.6e} a.u.", nJ, E_PH, E_PH * HA2EV, 10.0 * M_PI / beta);

    // --- the two representations --------------------------------------------------------
    // (1) the production LS route: W^c on the bosonic Matsubara mesh -> masked_pole_fit.
    imag_axes_ft::dlr_pole_fit pf(ft);
    const long nwb = ft.nw_b();
    nda::array<ComplexType, 1> zb(nwb);
    { auto wnb = ft.wn_mesh_b();
      for (long m = 0; m < nwb; ++m) zb(m) = ft.omega(wnb(m)); }
    // W^c(i nu) is EVEN in nu; the closed form above is the retarded function, valid in the
    // UPPER half plane only, so the negative half of the bosonic mesh is read at i |nu|.
    // (Getting this wrong makes the reference non-even and every representation of it looks
    // 50 % off -- it did, before this line.)
    nda::array<ComplexType, 2> Ww(nwb, NAUX * NAUX);
    for (long m = 0; m < nwb; ++m)
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q)
          Ww(m, P * NAUX + Q) = W_exact(d, P, Q, cval_t(0.0, std::abs(zb(m).imag())));

    auto ls = imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, E_PH, -1.0);
    auto cls = ls.coeffs(Ww);
    for (long p = 0; p < ls.nkeep; ++p)
      for (long j = 0; j < NAUX * NAUX; ++j) cls(p, j) *= ls.residue_scale(p);
    double ls_rec = 0.0, ls_den = 0.0;
    for (long m = 0; m < nwb; ++m)
      for (long j = 0; j < NAUX * NAUX; ++j) {
        ComplexType rec(0.0);
        for (long p = 0; p < ls.nkeep; ++p) rec += cls(p, j) / (zb(m) - ls.om(p));
        ls_rec = std::max(ls_rec, std::abs(rec - Ww(m, j)));
        ls_den = std::max(ls_den, std::abs(Ww(m, j)));
      }
    app_log(2, "[rw2-a-ii] LS route: {} of {} auxiliary nodes retained (support |eps| >= "
               "{:.4e}), rank {}, bosonic-mesh reconstruction = {:.4e}",
            ls.nkeep, ls.np_all, E_PH, ls.n_kept, ls_rec / ls_den);

    // (2) the spectral quadrature of the SAME W^c.
    const double Omega_max = 6.0;
    const long   N_O = long(std::ceil(Omega_max / (0.5 * d.gam)));
    auto g = make_grid(Omega_max, N_O);
    nda::array<double, 1> Om_data(N_O);
    { const double h = Omega_max / double(N_O);
      for (long l = 0; l < N_O; ++l) Om_data(l) = h * double(l + 1); }
    nda::array<double, 1> om_s;
    nda::array<double, 3> Rs;
    wc_spectral::bin_plan_t plan;
    build_rep(d, g, Om_data, /*ntarget*/ 64, om_s, Rs, plan);
    double sp_rec = 0.0;
    for (long m = 0; m < nwb; ++m)
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          ComplexType rec(0.0);
          for (long p = 0; p < om_s.shape(0); ++p) rec += Rs(p, P, Q) / (zb(m) - om_s(p));
          sp_rec = std::max(sp_rec, std::abs(rec - Ww(m, P * NAUX + Q)));
        }
    REQUIRE(sp_rec / ls_den < 1.0e-2);
    app_log(2, "[rw2-a-ii] spectral route: {} bins -> {} poles, bosonic-mesh reconstruction "
               "= {:.4e}", plan.nbin, om_s.shape(0), sp_rec / ls_den);

    // --- the cancellation meter, identical formula for both -----------------------------
    // Sigma^c(z) = sum_{J,p} M_{Jp} (n_B(om_p) + f(eps_J)) / (z - (eps_J - om_p)),
    // M_{Jp} = sum_PQ A_P(J) R_p(P,Q) conj(A_Q(J)),
    // Sabs   = sum_{J,p} |M_{Jp}| |n_B + f| / |z - (eps_J - om_p)|.
    auto meter = [&](nda::array<double, 1> const &om, auto const &Rget, long npole,
                     cval_t z, double &sabs, double &neg_share) {
      ComplexType S(0.0, 0.0);
      sabs = 0.0;
      double npos = 0.0, nneg = 0.0;
      for (long J = 0; J < nJ; ++J)
        for (long p = 0; p < npole; ++p) {
          ComplexType M(0.0, 0.0);
          for (long P = 0; P < NAUX; ++P)
            for (long Q = 0; Q < NAUX; ++Q) M += Apair(J, P) * Rget(p, P, Q) * Apair(J, Q);
          const double w = sigma_route_b::stable_nB(beta, om(p)) + fJ(J);
          const ComplexType den = z - (epsJ(J) - om(p));
          S += M * w / den;
          sabs += std::abs(M) * std::abs(w) / std::abs(den);
          const double sg = M.real() * w;
          if (sg >= 0.0) npos += std::abs(sg); else nneg += std::abs(sg);
        }
      neg_share = (npos + nneg > 0.0) ? nneg / (npos + nneg) : 0.0;
      return S;
    };

    auto Rls = [&](long p, long P, long Q) { return cls(p, P * NAUX + Q); };
    auto Rsp = [&](long p, long P, long Q) { return ComplexType(Rs(p, P, Q), 0.0); };

    // The EXACT representation in the gamma -> 0 limit: poles at +-Omega_m with residues
    // +-R_m. This is the physical answer the two numerical representations approximate, and
    // its OWN Sabs/|Sigma| is the intrinsic principal-value balance of the model -- the floor
    // no representation can beat.
    // (the -Omega member is -R_m^dag = -R_m, R_m being Hermitian: for the SHARP model the
    //  transpose of eq. (B) is the conjugate transpose of the residue matrix.)
    nda::array<double, 1> om_ex(2 * long(d.Om.size()));
    nda::array<ComplexType, 3> Rex(2 * long(d.Om.size()), NAUX, NAUX);
    for (size_t m = 0; m < d.Om.size(); ++m) {
      om_ex(long(m)) = d.Om[m];
      om_ex(long(d.Om.size() + m)) = -d.Om[m];
      for (long P = 0; P < NAUX; ++P)
        for (long Q = 0; Q < NAUX; ++Q) {
          Rex(long(m), P, Q) = d.R[m](P, Q);
          Rex(long(d.Om.size() + m), P, Q) = -std::conj(d.R[m](Q, P));
        }
    }
    auto Rref = [&](long p, long P, long Q) { return Rex(p, P, Q); };

    double ls_worst = 0.0, sp_worst = 0.0, ref_worst = 0.0;
    double sp_over_ref = 0.0, sp_in_band = 0.0;
    app_log(2, "[rw2-a-ii]  {:>12} {:>6} | {:>12} {:>12} {:>6} | {:>12} {:>12} {:>6} | "
               "{:>12} {:>12}", "eps-mu (eV)", "eta", "Sabs LS", "Sabs/|Sig_ex|", "neg %",
            "Sabs SPEC", "Sabs/|Sig_ex|", "neg %", "Sabs EXACT", "Sabs/|Sig_ex|");
    const double etas[3] = {0.0, M_PI / beta, 10.0 * M_PI / beta};
    for (long i : {29L, 30L}) {                      // the gap-window states, straddling mu
      for (int e = 0; e < 3; ++e) {
        const cval_t z(epsJ(i), etas[e]);
        double sa_ls = 0.0, sa_sp = 0.0, sa_ex = 0.0, ng_ls = 0.0, ng_sp = 0.0, ng_ex = 0.0;
        const ComplexType S_ls = meter(ls.om, Rls, ls.nkeep, z, sa_ls, ng_ls);
        const ComplexType S_sp = meter(om_s, Rsp, om_s.shape(0), z, sa_sp, ng_sp);
        const ComplexType S_ex = meter(om_ex, Rref, om_ex.shape(0), z, sa_ex, ng_ex);
        const double ref = std::abs(S_ex);
        ls_worst  = std::max(ls_worst,  sa_ls / ref);
        sp_worst  = std::max(sp_worst,  sa_sp / ref);
        ref_worst = std::max(ref_worst, sa_ex / ref);
        sp_over_ref = std::max(sp_over_ref, sa_sp / sa_ex);
        // where the model's OWN balance is inside the spec's O(1-10) band, the spectral
        // representation must be too.
        if (sa_ex / ref <= 10.0) sp_in_band = std::max(sp_in_band, sa_sp / ref);
        app_log(2, "[rw2-a-ii]  {:>12.4f} {:>6.3f} | {:>12.4e} {:>13.4e} {:>6.2f} | "
                   "{:>12.4e} {:>13.4e} {:>6.2f} | {:>12.4e} {:>13.4e}",
                (epsJ(i) - mu) * HA2EV, etas[e] * 1e3, sa_ls, sa_ls / ref, 100.0 * ng_ls,
                sa_sp, sa_sp / ref, 100.0 * ng_sp, sa_ex, sa_ex / ref);
        app_log(3, "[rw2-a-ii]      |Sigma|: LS = {:.6e}, SPEC = {:.6e}, EXACT = {:.6e}",
                std::abs(S_ls), std::abs(S_sp), ref);
      }
    }
    app_log(2, "[rw2-a-ii] GATE RW-2-a(ii), worst over the gap-window states x eta: "
               "Sabs/|Sigma^c_exact| = {:.4e} (LS, the SVO disease) vs {:.4e} (spectral "
               "quadrature) -- a factor {:.4g}. The EXACT 6-pole representation of the same "
               "model reads {:.4e}, which is the intrinsic principal-value balance and the "
               "floor for any representation.",
            ls_worst, sp_worst, ls_worst / sp_worst, ref_worst);

    // The precondition: the toy must actually reproduce the disease, otherwise it does not
    // discriminate. (SVO reads 1e4-1e5; a 3-aux toy on a 74-node DLR grid reads less, but
    // the two-orders separation is what the gate is about.)
    REQUIRE(ls_worst > 1.0e2);

    // THE GATE. The spec asks for Sabs/|Sigma^c| = O(1-10) from the quadrature. Measured on
    // this toy, the EXACT 6-pole representation of the same model -- which has no
    // representation error whatsoever -- already reads up to ref_worst on the same probe,
    // because Sigma^c has a zero crossing just above mu and the particle/hole halves of the
    // principal-value sum nearly cancel there. That value is a property of the PHYSICS, not
    // of any pole representation, so the gate is stated in the two ways that are actually
    // attainable and both are stricter than a bare threshold would be:
    //   (a) the quadrature must sit ON the exact representation's own balance;
    //   (b) wherever that balance is inside the spec's O(1-10) band, so is the quadrature.
    // [FLAGGED: the literal reading of the spec is "<= 10 everywhere"; on this toy that is
    //  unattainable by ANY representation, exact included. Numbers in notes/rw2_report.md.]
    app_log(2, "[rw2-a-ii] GATE: spectral / exact balance = {:.4f} (must be ~1); worst "
               "spectral value where the exact balance is inside O(1-10) = {:.4e}",
            sp_over_ref, sp_in_band);
    REQUIRE(sp_over_ref <= 1.1);
    REQUIRE(sp_in_band <= 1.0e1);
  }

} // namespace bdft_tests
