#!/usr/bin/env python3
"""Run ON rusty. Round-2 localization harvest: split the augmentation's cancellation
of the smooth-side band runaway. Diagnostic = the n=250 -> n=500 increment."""
import os, re

ROOT = "/mnt/home/mmorales/ceph/CoQui/abinit"
W1, W2 = f"{ROOT}/rpa_localize_jthd", f"{ROOT}/rpa_localize2_jthd"
BASE250 = f"{ROOT}/rpa_eos_jthd_coqui_nb/a10.05_n250"
BASE500 = f"{ROOT}/eos_jthd_coqui/a10.05"


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


def st(d):
    p = os.path.join(d, "rpa.done")
    return open(p).read().strip() if os.path.exists(p) else "running"


rows = [
    ("baseline (all on)", ec(BASE250), ec(BASE500), "done"),
    ("smooth only (aug off)", ec(f"{W1}/aug_off_n250"), ec(f"{W1}/aug_off_n500"), "done"),
]
for v, lab in (("shape", "shape compensation"), ("vgl_off", "no smooth<->aug (V_GL)"),
               ("vll_off", "no aug<->aug (V_LL)")):
    rows.append((lab, ec(f"{W2}/{v}_n250"), ec(f"{W2}/{v}_n500"),
                 f"{st(f'{W2}/{v}_n250')}/{st(f'{W2}/{v}_n500')}"))

print("# CoQui round-2 localization, a=10.05, PAW jth_with_d")
print("# ABINIT on the identical mean field: increment = -6.0 mHa\n")
print(f"{'variant':>24} {'E_c(250)':>11} {'E_c(500)':>11} {'increment':>13}")
for name, a, b, s in rows:
    f = lambda x: f"{x:11.5f}" if x is not None else f"{'--':>11}"
    if a is not None and b is not None:
        inc = 1000 * (b - a)
        tag = "  <-- COLLAPSED (culprit)" if abs(inc) < 20 else ""
        print(f"{name:>24} {f(a)} {f(b)} {inc:10.1f} mHa{tag}")
    else:
        print(f"{name:>24} {f(a)} {f(b)} {'[' + s + ']':>13}")
