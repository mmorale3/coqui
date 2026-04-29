#!/usr/bin/env bash
# Phase 4 external-reference benchmark harness.
#
# Compares CoQui's PAW-ISDF-THC ERIs against a reference (VASP, PySCF GTH, or
# direct PAW Coulomb integration) for a small bench input. The reference is
# *external* — generating it requires running another code, which can't be
# done from CI. So this script:
#
#   * SKIPs (exit 77) if the reference fixture is missing.
#   * RUNs and asserts when the fixture is present.
#
# Reference fixture layout (one directory per case under tests/paw_benchmark/):
#   bench_<name>/
#     coqui_input.toml         -- input that drives CoQui
#     eri_reference.h5          -- reference ERI tensor in /eri (ngrid axes)
#     metadata.json             -- {"tol": 1e-3, "method": "VASP", "notes": "..."}
#
# Usage:
#   run_benchmark.sh <bench_dir>
#   run_benchmark.sh --all
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COQUI_BIN_DEFAULT="${COQUI_BIN:-${SCRIPT_DIR}/../../build/cpu/bin/coqui}"

SKIP_RC=77
FAIL_RC=1
PASS_RC=0

usage() {
    grep -E '^#( |!|$)' "$0" | sed -E 's/^# ?//' >&2
    exit 64
}

run_case() {
    local case_dir="$1"
    local name="$(basename "$case_dir")"
    if [[ ! -f "$case_dir/coqui_input.toml" ]]; then
        echo "SKIP $name: coqui_input.toml missing"
        return $SKIP_RC
    fi
    if [[ ! -f "$case_dir/eri_reference.h5" ]]; then
        echo "SKIP $name: eri_reference.h5 missing (provide an external reference)"
        return $SKIP_RC
    fi
    if [[ ! -x "$COQUI_BIN_DEFAULT" ]]; then
        echo "SKIP $name: coqui binary not found at $COQUI_BIN_DEFAULT"
        return $SKIP_RC
    fi
    if ! command -v h5dump >/dev/null 2>&1; then
        echo "SKIP $name: h5dump not on PATH"
        return $SKIP_RC
    fi

    local tol=1e-3
    if [[ -f "$case_dir/metadata.json" ]]; then
        tol=$(python3 -c "import json,sys; d=json.load(open('$case_dir/metadata.json')); print(d.get('tol', 1e-3))" 2>/dev/null || echo 1e-3)
    fi

    echo "RUN  $name (tol=$tol)"
    pushd "$case_dir" >/dev/null

    local coqui_out="coqui_run.h5"
    rm -f "$coqui_out"
    if ! "$COQUI_BIN_DEFAULT" coqui_input.toml >"coqui_run.log" 2>&1; then
        echo "FAIL $name: coqui exited nonzero (see $case_dir/coqui_run.log)"
        popd >/dev/null
        return $FAIL_RC
    fi

    if [[ ! -f "$coqui_out" ]]; then
        # Many CoQui workflows write to a custom output filename specified in
        # the toml; the harness as-is requires the input to set it to
        # coqui_run.h5. If your workflow uses a different filename, adapt the
        # toml or extend this script.
        echo "FAIL $name: coqui_run.h5 not produced (check coqui_input.toml output config)"
        popd >/dev/null
        return $FAIL_RC
    fi

    # Numeric diff: read /eri from both files, max-abs-diff against tol.
    # Pure shell + h5dump → awk; avoids Python dependency.
    local diff
    diff=$(
        h5dump -d /eri -w 0 "$coqui_out"           > /tmp/coqui.eri 2>/dev/null
        h5dump -d /eri -w 0 "eri_reference.h5"     > /tmp/ref.eri   2>/dev/null
        # extract numeric tokens, pair-wise diff, max-abs
        paste <(grep -oE '\-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?' /tmp/coqui.eri) \
              <(grep -oE '\-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?' /tmp/ref.eri) | \
            awk 'BEGIN{m=0} { d=$1-$2; if (d<0) d=-d; if (d>m) m=d } END{print m}'
        rm -f /tmp/coqui.eri /tmp/ref.eri
    )

    popd >/dev/null

    awk -v d="$diff" -v t="$tol" 'BEGIN { exit !(d+0 < t+0) }'
    if [[ $? -eq 0 ]]; then
        echo "PASS $name: max|ERI diff|=$diff < $tol"
        return $PASS_RC
    else
        echo "FAIL $name: max|ERI diff|=$diff >= $tol"
        return $FAIL_RC
    fi
}

if [[ $# -eq 0 ]]; then usage; fi

if [[ "$1" == "--all" ]]; then
    overall=0
    any_run=0
    for d in "$SCRIPT_DIR"/bench_*; do
        [[ -d "$d" ]] || continue
        any_run=1
        run_case "$d"
        rc=$?
        case $rc in
            $PASS_RC) ;;
            $SKIP_RC) ;;
            *) overall=1;;
        esac
    done
    if [[ $any_run -eq 0 ]]; then
        echo "SKIP: no bench_* directories under $SCRIPT_DIR"
        exit $SKIP_RC
    fi
    exit $overall
else
    run_case "$1"
    exit $?
fi
