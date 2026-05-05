/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * HDF5 I/O for cached PAW local-ISDF factorizations.
 *
 *   /PAW_LocalISDF/
 *       attribute  metric          : "L2" | "Coulomb"
 *       attribute  tol             : pivoted-Cholesky tolerance used at build
 *       attribute  nsp             : number of species
 *       Species/{nt}/
 *           attribute  is_present  : 1 if species was compressed, 0 otherwise
 *           attribute  nh          : projector count for this species
 *           attribute  nlambda     : number of λ rows
 *           attribute  n_pairs_kept: number of (αβ) pivot pairs kept
 *           dataset    U           : (nlambda, nh)  real
 *           dataset    lambda_i    : (nlambda)      int
 *           dataset    lambda_j    : (nlambda)      int
 *           dataset    lambda_sign : (nlambda)      real
 *           dataset    lambda_ij   : (nlambda)      int
 *           dataset    eta_qg_q0   : (nlambda, ngm) complex
 *           dataset    pair_pivot_order : (n_pairs_kept) int
 *           dataset    error_at_step    : (n_pairs_kept+1) real
 *
 * The file format is intentionally self-contained — given an .h5 produced
 * here, a downstream calculation can rebuild a `species_local_isdf` with
 * no further reference to the source pseudopotential. (Validation that the
 * eta_qg_q0 grid still matches the active dense G-grid is the caller's
 * responsibility; we record `ngm` as a sanity check via dataset shape.)
 *
 * Bulk write/read functions take an h5::group as anchor so callers can
 * compose with their own h5 layout (e.g. attach to an existing
 * /Hamiltonian or /System group).
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_LOCAL_ISDF_H5_HPP
#define HAMILTONIAN_PAW_LOCAL_ISDF_H5_HPP

#include <string>
#include <vector>

#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "hamiltonian/paw/local_isdf.hpp"
#include "hamiltonian/paw/local_isdf_compress.hpp"

namespace hamilt::paw {

inline char const* kLocalISDFGroup = "PAW_LocalISDF";

namespace detail {

inline h5::group ensure_group(h5::group& parent, std::string const& name)
{
    if (parent.has_subgroup(name))
        return parent.open_group(name);
    return parent.create_group(name);
}

} // namespace detail

/**
 * Write all per-species compressed local-ISDF data and the corresponding
 * compression reports under `parent / PAW_LocalISDF /`.
 *
 * Vectors `isdf` and `report` must be the same length (one entry per
 * species, indexed by nt; non-PAW species can have empty isdf with
 * nlambda == 0 and empty report — they are recorded with is_present = 0).
 */
inline void write_local_isdf_h5(
    h5::group parent,
    std::vector<species_local_isdf> const& isdf,
    std::vector<isdf_compression_report> const& report,
    isdf_metric metric,
    double tol)
{
    utils::check(isdf.size() == report.size(),
        "write_local_isdf_h5: isdf/report size mismatch ({} vs {})",
        isdf.size(), report.size());

    h5::group root = detail::ensure_group(parent, kLocalISDFGroup);
    h5::h5_write_attribute(root, "metric", std::string(metric_name(metric)));
    h5::h5_write_attribute(root, "tol",    tol);
    int nsp = (int)isdf.size();
    h5::h5_write_attribute(root, "nsp",    nsp);

    h5::group sgrp = detail::ensure_group(root, "Species");
    for (int nt = 0; nt < nsp; ++nt) {
        std::string nt_name = std::to_string(nt);
        h5::group spnt = detail::ensure_group(sgrp, nt_name);
        auto const& s = isdf[nt];
        auto const& rep = report[nt];
        int present = (s.nlambda > 0) ? 1 : 0;
        h5::h5_write_attribute(spnt, "is_present", present);
        h5::h5_write_attribute(spnt, "nh",         s.nh);
        h5::h5_write_attribute(spnt, "nlambda",    s.nlambda);
        h5::h5_write_attribute(spnt, "n_pairs_kept",
                               (int)rep.pair_pivot_order.size());
        if (!present) continue;

        nda::h5_write(spnt, "U",           s.U);
        nda::h5_write(spnt, "lambda_i",    s.lambda_i);
        nda::h5_write(spnt, "lambda_j",    s.lambda_j);
        nda::h5_write(spnt, "lambda_sign", s.lambda_sign);
        nda::h5_write(spnt, "lambda_ij",   s.lambda_ij);
        nda::h5_write(spnt, "eta_qg_q0",   s.eta_qg_q0);

        // Report — written as plain arrays for portability
        if (!rep.pair_pivot_order.empty()) {
            nda::array<int,1>    pp((long)rep.pair_pivot_order.size());
            for (long k = 0; k < pp.extent(0); ++k)
                pp(k) = rep.pair_pivot_order[k];
            nda::h5_write(spnt, "pair_pivot_order", pp);
        }
        if (!rep.error_at_step.empty()) {
            nda::array<double,1> es((long)rep.error_at_step.size());
            for (long k = 0; k < es.extent(0); ++k)
                es(k) = rep.error_at_step[k];
            nda::h5_write(spnt, "error_at_step", es);
        }
    }
}

/**
 * Read all per-species compressed local-ISDF data from
 * `parent / PAW_LocalISDF /`. Returns a vector indexed by species.
 *
 * If the group does not exist, returns an empty vector.
 *
 * Output `metric_out` and `tol_out` (when non-null) receive the recorded
 * compression parameters; useful when reproducing the run.
 */
inline std::vector<species_local_isdf> read_local_isdf_h5(
    h5::group parent,
    isdf_metric* metric_out = nullptr,
    double* tol_out = nullptr)
{
    std::vector<species_local_isdf> out;
    if (!parent.has_subgroup(kLocalISDFGroup)) return out;
    h5::group root = parent.open_group(kLocalISDFGroup);
    int nsp = 0;
    h5::h5_read_attribute(root, "nsp", nsp);
    if (nsp <= 0) return out;
    if (metric_out) {
        std::string mn;
        h5::h5_read_attribute(root, "metric", mn);
        *metric_out = (mn == "Coulomb") ? isdf_metric::Coulomb : isdf_metric::L2;
    }
    if (tol_out) h5::h5_read_attribute(root, "tol", *tol_out);

    if (!root.has_subgroup("Species")) return out;
    h5::group sgrp = root.open_group("Species");
    out.resize(nsp);
    for (int nt = 0; nt < nsp; ++nt) {
        std::string nt_name = std::to_string(nt);
        if (!sgrp.has_subgroup(nt_name)) continue;
        h5::group spnt = sgrp.open_group(nt_name);
        int present = 0;
        h5::h5_read_attribute(spnt, "is_present", present);
        if (!present) continue;
        auto& s = out[nt];
        h5::h5_read_attribute(spnt, "nh",      s.nh);
        h5::h5_read_attribute(spnt, "nlambda", s.nlambda);
        nda::h5_read(spnt, "U",           s.U);
        nda::h5_read(spnt, "lambda_i",    s.lambda_i);
        nda::h5_read(spnt, "lambda_j",    s.lambda_j);
        nda::h5_read(spnt, "lambda_sign", s.lambda_sign);
        nda::h5_read(spnt, "lambda_ij",   s.lambda_ij);
        nda::h5_read(spnt, "eta_qg_q0",   s.eta_qg_q0);
    }
    return out;
}

/**
 * Convenience: open a file path, run pivoted-Cholesky compression for
 * every species, and write the result. Will create the parent group if
 * absent. Existing PAW_LocalISDF/ payload is overwritten.
 *
 * The resulting cache lets a downstream calculation avoid the (small but
 * non-trivial) Gram + pivoted-Cholesky cost; the eta_qg_q0 binary is
 * already in the right shape for paw_aug_thc / thc_reader_t consumption.
 */
inline void cache_compressed_local_isdf_to_h5(
    std::string const& h5_path,
    pseudopot const& psp,
    nda::stack_array<double,3,3> const& recv,
    double omega,
    isdf_metric metric,
    double tol)
{
    auto const& sps  = psp.paw_species_view();
    int nsp = (int)sps.size();
    std::vector<species_local_isdf> isdfs;
    std::vector<isdf_compression_report> reps;
    isdfs.reserve(nsp); reps.reserve(nsp);

    for (int nt = 0; nt < nsp; ++nt) {
        bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
        if (!has_aug) {
            isdfs.emplace_back();              // empty
            reps.emplace_back();               // empty
            continue;
        }
        auto [s, r] = compress_local_isdf_species(psp, nt, recv, omega,
                                                   metric, tol);
        isdfs.push_back(std::move(s));
        reps.push_back(std::move(r));
    }

    // Open in r/w if exists, create otherwise.
    h5::file f(h5_path,
               std::filesystem::exists(h5_path) ? 'a' : 'w');
    h5::group root(f);
    write_local_isdf_h5(root, isdfs, reps, metric, tol);
}

/**
 * Convenience reader: open a file and load compressed local-ISDF.
 * Returns empty vector if the file or group is missing.
 */
inline std::vector<species_local_isdf> load_compressed_local_isdf_from_h5(
    std::string const& h5_path,
    isdf_metric* metric_out = nullptr,
    double* tol_out = nullptr)
{
    if (!std::filesystem::exists(h5_path)) return {};
    h5::file f(h5_path, 'r');
    h5::group root(f);
    return read_local_isdf_h5(root, metric_out, tol_out);
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_LOCAL_ISDF_H5_HPP
