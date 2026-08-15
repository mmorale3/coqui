#!/usr/bin/env bash
# Schema validator for pw2coqui output.
#
# Usage: check_schema.sh <pwscf.coqui.h5>
#
# Exits 0 if the HDF5 file matches the pseudopotential schema, nonzero
# otherwise. Uses h5dump only — no Python / h5py dependency.
#
# What's checked:
#   * Mandatory groups: /System, /Orbitals, /Hamiltonian, /Hamiltonian/Species
#   * /Hamiltonian/pp_type attribute is one of {ncpp, uspp, paw}
#   * For each /Hamiltonian/Species/nt{N}: species_kind attribute exists, and
#     the radial datasets r, rab, beta, dion, lll, kbeta, indv, nhtol, nhtolm
#     are present.
#   * If any species_kind == "paw": Onecenter/deltaC and paw/{pfunc, ptfunc,
#     augmom} groups/datasets are present.
#   * If pp_type in {uspp, paw}: qfuncl and qqq present per species.
#
# Not checked here: exact dataset shapes (size mismatches are caught later
# by the CoQui reader), multipole-moment reproduction, and Onecenter
# symmetry.

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <pwscf.coqui.h5>" >&2
    exit 64
fi

h5file="$1"
if [[ ! -r "$h5file" ]]; then
    echo "FAIL: cannot read $h5file" >&2
    exit 1
fi

if ! command -v h5dump >/dev/null 2>&1; then
    echo "SKIP: h5dump not on PATH" >&2
    exit 77
fi

contents=$(h5dump -n "$h5file" 2>/dev/null) || {
    echo "FAIL: h5dump errored on $h5file" >&2
    exit 1
}

fail() { echo "FAIL: $1" >&2; exit 1; }
have_path() { grep -qE "^[[:space:]]+(group|dataset)[[:space:]]+$1\$" <<<"$contents"; }

# Mandatory top-level groups
for g in /System /Orbitals /Hamiltonian /Hamiltonian/Species; do
    have_path "$g" || fail "missing mandatory group: $g"
done

# Detect pp_type from /Hamiltonian attribute
pp_type=$(h5dump -a /Hamiltonian/pp_type "$h5file" 2>/dev/null \
            | awk -F'"' '/\(0\):/{print $2; exit}')
case "$pp_type" in
    ncpp|uspp|paw) ;;
    *) fail "/Hamiltonian/pp_type missing or invalid: '$pp_type'";;
esac

# Enumerate species
species_groups=$(grep -oE '/Hamiltonian/Species/nt[0-9]+' <<<"$contents" \
                   | sort -u)
[[ -z "$species_groups" ]] && fail "no /Hamiltonian/Species/nt{N} groups"

found_paw=0
for sp in $species_groups; do
    # Required radial + bookkeeping datasets per species
    for d in r rab beta dion lll kbeta indv nhtol nhtolm; do
        have_path "$sp/$d" || fail "missing $sp/$d"
    done
    # species_kind attribute
    kind=$(h5dump -a "$sp/species_kind" "$h5file" 2>/dev/null \
            | awk -F'"' '/\(0\):/{print $2; exit}')
    case "$kind" in
        ncpp|uspp|paw) ;;
        *) fail "$sp/species_kind missing or invalid: '$kind'";;
    esac
    # mesh attribute
    h5dump -a "$sp/mesh" "$h5file" >/dev/null 2>&1 \
        || fail "$sp/mesh attribute missing"
    if [[ "$kind" == "uspp" || "$kind" == "paw" ]]; then
        for d in qfuncl qqq; do
            have_path "$sp/$d" || fail "missing $sp/$d (kind=$kind)"
        done
    fi
    if [[ "$kind" == "paw" ]]; then
        found_paw=1
        for d in pfunc ptfunc augmom; do
            have_path "$sp/paw/$d" || fail "missing $sp/paw/$d"
        done
        have_path "$sp/Onecenter/deltaC" \
            || fail "missing $sp/Onecenter/deltaC"
    fi
done

# Cross-check: pp_type=paw should produce at least one PAW species
if [[ "$pp_type" == "paw" && "$found_paw" -eq 0 ]]; then
    fail "pp_type='paw' but no species had species_kind='paw'"
fi

echo "OK: $h5file matches the schema (pp_type=$pp_type, $(wc -w <<<"$species_groups") species)"
