# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""E4 analysis: does removing per-step sampling noise (noise_scale=0) reduce evaluation variance?

Paired per design: within-sequence SD of CA RMSD and pLDDT under baseline vs noise-free
sampling, same four seeds (0-3). Cluster bootstrap over designs for the log-SD ratio.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
E4 = E1
SEEDS = [0, 1, 2, 3]


def load(path):
    g = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if r["status"] != "ok" or int(r["seed"]) not in SEEDS:
            continue
        g[r["h_id"]][int(r["seed"])] = (float(r["rmsd"]), float(r["plddt"]))
    return g


def stats(vals):
    arr = np.array(vals)
    return arr.mean(), arr.std(ddof=1) if len(arr) > 1 else np.nan


def qualified(rmsd, plddt):
    return rmsd < 2 and plddt >= 80


def main():
    base = load(E1 / "e1_metrics.csv")
    e4 = load(E4 / "e4_metrics.csv")
    rows = []
    for h in base:
        if h not in e4:
            continue
        b = [base[h][s] for s in SEEDS if s in base[h]]
        n = [e4[h][s] for s in SEEDS if s in e4[h]]
        if len(b) < 2 or len(n) < 2:
            continue
        b_rmsd_sd = stats([x[0] for x in b])[1]
        n_rmsd_sd = stats([x[0] for x in n])[1]
        b_plddt_sd = stats([x[1] for x in b])[1]
        n_plddt_sd = stats([x[1] for x in n])[1]
        b_f = len({qualified(*x) for x in b}) > 1
        n_f = len({qualified(*x) for x in n}) > 1
        rows.append(dict(h_id=h, length=None,
                         base_rmsd_sd=b_rmsd_sd, e4_rmsd_sd=n_rmsd_sd,
                         base_plddt_sd=b_plddt_sd, e4_plddt_sd=n_plddt_sd,
                         base_flip=b_f, e4_flip=n_f,
                         base_mean_rmsd=stats([x[0] for x in b])[0],
                         e4_mean_rmsd=stats([x[0] for x in n])[0]))

    ratios = []
    for r in rows:
        if r["base_rmsd_sd"] > 0 and r["e4_rmsd_sd"] > 0:
            ratios.append(np.log(r["e4_rmsd_sd"] / r["base_rmsd_sd"]))
    rng = np.random.default_rng(5)
    ratios = np.array(ratios)
    boot = [ratios[rng.integers(0, len(ratios), len(ratios))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = dict(
        n=len(rows),
        base_flips=sum(r["base_flip"] for r in rows),
        e4_flips=sum(r["e4_flip"] for r in rows),
        mean_log_sd_ratio=float(ratios.mean()),
        median_sd_ratio=float(np.exp(np.median(ratios))),
        ci_log_ratio=(float(lo), float(hi)),
        per_design=rows,
    )
    (OUT / "e4_results.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"designs={len(rows)}")
    print(f"flips: baseline {out['base_flips']} vs noise-free {out['e4_flips']}")
    print(f"median SD ratio (noise-free / baseline) = {out['median_sd_ratio']:.2f} "
          f"(geometric-mean SD ratio interval, not median CI: {np.exp(lo):.2f}-{np.exp(hi):.2f})")
    print(f"{'design':24s} {'base_sd':>8s} {'no_noise_sd':>11s} {'ratio':>6s} {'flip':>10s}")
    for r in sorted(rows, key=lambda x: -x["base_rmsd_sd"])[:12]:
        ratio = r["e4_rmsd_sd"] / r["base_rmsd_sd"] if r["base_rmsd_sd"] > 0 else float("nan")
        print(f"{r['h_id']:24s} {r['base_rmsd_sd']:8.2f} {r['e4_rmsd_sd']:11.2f} "
              f"{ratio:6.2f} {int(r['base_flip'])}->{int(r['e4_flip'])}")


if __name__ == "__main__":
    main()
