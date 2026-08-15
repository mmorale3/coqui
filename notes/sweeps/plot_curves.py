#!/usr/bin/env python3
"""Error-vs-c_mu curves + c_mu* table from collect.py's results.csv.

Usage: python3 plot_curves.py --csv RUNDIR/results.csv --out FIGDIR [--target 1e-4]

Produces, per fixture: <fixture>_err_vs_cmu.pdf (three panels: |dE_x|, |dE_H|,
|dE_RPA| vs c) and prints/writes cmu_star.csv: smallest c meeting --target (Ha)
per (fixture, config, observable), linearly interpolated in log-error.
"""
import argparse, csv, math, os
from collections import defaultdict

OBS = [("de_x", r"$|\Delta E_x|$"), ("de_hartree", r"$|\Delta E_H|$"), ("de_rpa", r"$|\Delta E_{RPA}|$")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--target", type=float, default=1e-4)
    args = p.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.csv))
            if r["status"] == "ok" and r["c"] and r.get("de_rpa") not in (None, "",)]
    by_fix = defaultdict(lambda: defaultdict(list))  # fixture -> config -> [(c, {obs: err})]
    for r in rows:
        errs = {k: abs(float(r[k])) for k, _ in OBS if r.get(k)}
        by_fix[r["fixture"]][r["config"]].append((int(r["c"]), errs))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    star_rows = []
    for fix, cfgs in sorted(by_fix.items()):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        for cfg, pts in sorted(cfgs.items()):
            pts.sort()
            cs = [c for c, _ in pts]
            for ax, (key, label) in zip(axes, OBS):
                ys = [e.get(key, float("nan")) for _, e in pts]
                ax.semilogy(cs, ys, "o-", label=cfg, lw=1.2, ms=3.5)
            # c_mu*: first c where err <= target, log-interpolated between brackets
            for key, _ in OBS:
                ys = [e.get(key, float("nan")) for _, e in pts]
                star = ""
                for i, y in enumerate(ys):
                    if y <= args.target:
                        if i == 0:
                            star = cs[0]
                        else:
                            y0, y1 = ys[i-1], y
                            f = (math.log(args.target) - math.log(y0)) / (math.log(y1) - math.log(y0))
                            star = round(cs[i-1] + f*(cs[i]-cs[i-1]), 2)
                        break
                star_rows.append({"fixture": fix, "config": cfg, "obs": key,
                                  "cmu_star": star, "target": args.target})
        for ax, (_, label) in zip(axes, OBS):
            ax.axhline(args.target, color="gray", ls=":", lw=0.8)
            ax.set_xlabel(r"$c_\mu = N_\mu/N_{orb}$")
            ax.set_title(label)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("error (Ha)")
        axes[-1].legend(fontsize=7, ncol=1, frameon=False)
        fig.suptitle(f"Si {fix}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{fix}_err_vs_cmu.pdf"))
        plt.close(fig)

    with open(os.path.join(out, "cmu_star.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fixture", "config", "obs", "cmu_star", "target"])
        w.writeheader()
        w.writerows(star_rows)
    print(f"wrote {out}/<fixture>_err_vs_cmu.pdf and cmu_star.csv")
    for r in star_rows:
        if r["cmu_star"] != "":
            print(f'{r["fixture"]:5s} {r["config"]:12s} {r["obs"]:10s} c_mu* = {r["cmu_star"]}')


if __name__ == "__main__":
    main()
