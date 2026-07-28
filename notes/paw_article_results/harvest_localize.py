#!/usr/bin/env python3
"""Run ON rusty. Harvest the CoQui localization probe at a=10.05, PAW jth_with_d.

Diagnostic = the n=250 -> n=500 increment in E_c.  A variant that collapses it from
the baseline -153.5 mHa toward ABINIT's -6.0 mHa identifies the defective term.
"""
import os, re

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
WORK = f"{ROOT}/rpa_localize_jthd"
BASE = f"{ROOT}/rpa_eos_jthd_coqui_nb/a10.05_n250"      # baseline n=250
BASE500 = f"{ROOT}/eos_jthd_coqui/a10.05"               # baseline n=500
VARIANTS = ["aug_off", "onsite_off", "both_off", "tight"]

ABINIT_INCREMENT = -6.0   # mHa, n=250 -> n=500 on the identical mean field


def ec(d):
    p = os.path.join(d, "rpa.out")
    if not os.path.exists(p):
        return None
    v = None
    for line in open(p):
        m = re.match(r"\s*RPA energy:\s*([-\d.Ee+]+)", line)
        if m:
            v = float(m.group(1))
    return v


def status(d):
    p = os.path.join(d, "rpa.done")
    return open(p).read().strip() if os.path.exists(p) else "running"


rows = [("baseline", ec(BASE), ec(BASE500), "done")]
for v in VARIANTS:
    d250, d500 = f"{WORK}/{v}_n250", f"{WORK}/{v}_n500"
    rows.append((v, ec(d250), ec(d500), f"{status(d250)}/{status(d500)}"))

print(f"# CoQui localization probe, a=10.05, PAW jth_with_d")
print(f"# ABINIT on the identical mean field: n=250 -> n=500 increment = {ABINIT_INCREMENT:.1f} mHa\n")
print(f"{'variant':>12} {'E_c(250)':>11} {'E_c(500)':>11} {'increment':>12}   verdict")
for name, a, b, st in rows:
    f = lambda x: f"{x:11.5f}" if x is not None else f"{'--':>11}"
    if a is not None and b is not None:
        inc = 1000 * (b - a)
        tag = ("CULPRIT (increment collapsed)" if abs(inc) < 20
               else "not it (still blows up)")
        print(f"{name:>12} {f(a)} {f(b)} {inc:9.1f} mHa   {tag}")
    else:
        print(f"{name:>12} {f(a)} {f(b)} {'--':>9}       [{st}]")
