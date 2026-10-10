#!/usr/bin/env python3
"""Consistency check of src/methods/GW_line/README.md against the C++ sources (plan S8d).

Checks, and exits with status 1 on any mismatch:
  1. every [gw_line] key read by gw_line_params_t::from_ptree (driver.cpp) and optics_params_t::from_ptree (optics.hpp)
     appears in the README key table (rows between <!-- keys:begin --> and <!-- keys:end -->), and every key of the table is
     read by the code (plus `interaction`, which main.cpp / MBPT_drivers.cpp read for the gw_line block);
  2. the default printed in the table equals the default in the code for the keys whose default is a literal of the
     struct initializer (gw_line_params_t, optics_params_t, spectra_params_t, mixing_params_t) -- reported, not fatal, when the
     code default is computed (auto rules);
  3. every COQUI_GWLINE_* environment variable used under src/methods/GW_line and src/numerics/line_dlr (tests included)
     appears in the README environment tables (between <!-- env:begin --> and <!-- env:end -->), and vice versa.
Informational: keys read by the code but not mentioned in the driver.hpp header comment (the in-source key list).

Usage (from anywhere):  python3 coqui/cayley/scripts/check_gw_line_readme.py [--coqui <path to the coqui checkout>]
"""
import argparse
import os
import re
import sys


def read(path):
    with open(path) as f:
        return f.read()


def body(src, start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


KEY_PATTERNS = [
    r'get_value_with_default<[^>]+>\(\s*pt\s*,\s*"([^"]+)"',
    r'get_array_with_default<[^>]+>\(\s*pt\s*,\s*"([^"]+)"',
    r'pt\.get_optional<[^>]+>\(\s*"([^"]+)"',
    r'opt_list\(\s*pt\s*,\s*"([^"]+)"',
]


def keys_read(text):
    ks = set()
    for p in KEY_PATTERNS:
        ks.update(re.findall(p, text))
    return ks


def struct_defaults(text, struct):
    """name -> literal default of `struct NAME { ... }` member initializers (scalars and simple strings)."""
    s = body(text, 'struct ' + struct + ' {', '\n};')
    out = {}
    for line in s.splitlines():
        line = line.split('//')[0]
        m = re.match(r'\s*(?:double|long|bool|std::string|int)\s+(.*);', line)
        if not m:
            continue
        for part in re.split(r',(?![^"]*"\s*[,;])', m.group(1)):
            mm = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*$', part)
            if mm:
                out[mm.group(1)] = mm.group(2).strip()
    return out


def norm(v):
    v = v.strip().strip('`').strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    try:
        return repr(float(v))
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--coqui', default=os.path.normpath(os.path.join(here, '..', '..')))
    a = ap.parse_args()
    gl = os.path.join(a.coqui, 'src', 'methods', 'GW_line')
    nl = os.path.join(a.coqui, 'src', 'numerics', 'line_dlr')
    drv_cpp, drv_hpp = read(os.path.join(gl, 'driver.cpp')), read(os.path.join(gl, 'driver.hpp'))
    opt_hpp = read(os.path.join(gl, 'optics.hpp'))
    readme = read(os.path.join(gl, 'README.md'))
    ok = True

    # ---- 1. keys
    code = keys_read(body(drv_cpp, 'gw_line_params_t gw_line_params_t::from_ptree', 'void gw_line_params_t::log'))
    code |= keys_read(body(opt_hpp, 'static optics_params_t from_ptree', '\n  }\n'))
    code.add('interaction')   # main.cpp get_eri_block(pt, "interaction") for the [gw_line] block
    rows = body(readme, '<!-- keys:begin -->', '<!-- keys:end -->')
    table = {}
    for line in rows.splitlines():
        if not line.startswith('|') or line.startswith('|---') or line.startswith('| key'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        for k in re.findall(r'`([A-Za-z0-9_.]+)`', cells[0]):
            table[k] = cells
    doc = set(table)
    miss, extra = sorted(code - doc), sorted(doc - code)
    print(f'[keys] read by the code: {len(code)}; in the README table: {len(doc)}')
    if miss:
        ok = False
        print('  MISSING in the README table:', ', '.join(miss))
    if extra:
        ok = False
        print('  in the README table but not read by the code:', ', '.join(extra))
    if not miss and not extra:
        print('  key sets identical')

    # ---- 2. literal defaults
    sd = {}
    sd.update({k: v for k, v in struct_defaults(drv_hpp, 'gw_line_params_t').items()})
    mix = struct_defaults(read(os.path.join(gl, 'scf_mixing.hpp')), 'mixing_params_t')
    for k, v in mix.items():
        sd[{'alg': 'mixing_alg', 'hist': 'diis_hist', 'start': 'diis_start', 'beta': 'diis_beta', 'reg': 'diis_reg',
            'cmax': 'diis_cmax', 'grow': 'diis_grow', 'mix_F': 'diis_mix_F', 'wF': 'diis_wF'}.get(k, k)] = v
    for k, v in struct_defaults(opt_hpp, 'optics_params_t').items():
        sd['optics.' + k] = v
    for k, v in struct_defaults(read(os.path.join(gl, 'spectra.hpp')), 'spectra_params_t').items():
        sd['spectra.' + k] = v
    sd.update({'coarse.niter': sd.pop('coarse_niter', '0'), 'coarse.eps': sd.pop('coarse_eps', '1e-8'),
               'coarse.K': sd.pop('coarse_K', '16'), 'coarse.nodes_per_ray': sd.pop('coarse_nodes_per_ray', '80'),
               'optics.poles': sd.pop('optics_poles', '"final"'), 'spectra.enable': 'true'})
    ndef, nbad = 0, 0
    for k, cells in sorted(table.items()):
        if k not in sd or len(cells) < 3:
            continue
        dv = cells[2].split('(')[0].split(';')[0].strip()
        if not dv.startswith('`'):
            continue   # computed / described defaults ("auto: ...")
        ndef += 1
        if norm(dv) != norm(sd[k]):
            nbad += 1
            ok = False
            print(f'  DEFAULT MISMATCH {k}: README {dv} vs code {sd[k]}')
    print(f'[defaults] literal defaults compared: {ndef}, mismatches: {nbad}')

    # ---- 3. environment variables
    env_code = set()
    for root in (gl, nl):
        for dp, _, fs in os.walk(root):
            for fn in fs:
                if fn.endswith(('.hpp', '.cpp', '.cu', '.cuh', '.h')):
                    env_code.update(re.findall(r'COQUI_GWLINE_[A-Z0-9_]+', read(os.path.join(dp, fn))))
    env_doc = set(re.findall(r'`(COQUI_GWLINE_[A-Z0-9_]+)`', body(readme, '<!-- env:begin -->', '<!-- env:end -->')))
    em, ee = sorted(env_code - env_doc), sorted(env_doc - env_code)
    print(f'[env] in the sources: {len(env_code)}; in the README tables: {len(env_doc)}')
    if em:
        ok = False
        print('  MISSING in the README:', ', '.join(em))
    if ee:
        ok = False
        print('  in the README but not in the sources:', ', '.join(ee))
    if not em and not ee:
        print('  sets identical')

    # ---- informational: driver.hpp header comment
    hdr = drv_hpp[:drv_hpp.index('#include')]
    nohdr = sorted(k for k in code if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(k.split('.')[-1]) + r'(?![A-Za-z0-9_])', hdr))
    print('[info] keys read but not named in the driver.hpp header comment:', ', '.join(nohdr) if nohdr else 'none')

    print('RESULT:', 'consistent' if ok else 'INCONSISTENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
