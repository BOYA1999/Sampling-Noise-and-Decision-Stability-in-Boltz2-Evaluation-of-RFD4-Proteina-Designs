# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""E5 analysis: within-design variance, flip rate and mean shift as functions of noise scale.

Scales: 0 (E4), 0.25/0.5/0.75 (E5), 1.003 (baseline E1). Seeds 0-2 are used for every
condition so that the comparison is paired.
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
E5 = E1
SEEDS = [0, 1, 2]
SCALES = [0.0, 0.25, 0.5, 0.75, 1.003]


def load(path, scale_key="scale", base_scale=None):
    g = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(path.open()):
        if r["status"] != "ok" or int(r["seed"]) not in SEEDS:
            continue
        scale = base_scale if base_scale is not None else float(r[scale_key])
        g[float(scale)][r["h_id"]][int(r["seed"])] = (float(r["rmsd"]), float(r["plddt"]))
    return g


def main():
    data = {}
    base = load(E1 / "e1_metrics.csv", base_scale=1.003)
    zero = load(E1 / "e4_metrics.csv", base_scale=0.0)
    grad = load(E5 / "e5_metrics.csv")
    for d in (base, zero, grad):
        for s, v in d.items():
            data.setdefault(s, {}).update(v)

    out = {}
    for scale in SCALES:
        if scale not in data:
            continue
        per = []
        for h, seeds in data[scale].items():
            vals = [seeds[s][0] for s in SEEDS if s in seeds]
            pld = [seeds[s][1] for s in SEEDS if s in seeds]
            if len(vals) < 2:
                continue
            lab = [int(r < 2 and p >= 80) for r, p in zip(vals, pld)]
            per.append(dict(h_id=h, n=len(vals), mean=float(np.mean(vals)),
                            sd=float(np.std(vals, ddof=1)),
                            plddt_mean=float(np.mean(pld)),
                            flip=len(set(lab)) > 1))
        sds = np.array([p["sd"] for p in per])
        means = np.array([p["mean"] for p in per])
        out[str(scale)] = dict(n_designs=len(per),
                               sd_median=float(np.median(sds)),
                               sd_q1=float(np.percentile(sds, 25)),
                               sd_q3=float(np.percentile(sds, 75)),
                               sd_mean=float(sds.mean()),
                               sd_frac_gt_02=float(np.mean(sds > 0.2)),
                               sd_frac_gt_1=float(np.mean(sds > 1.0)),
                               flip_count=int(sum(p["flip"] for p in per)),
                               flip_rate=float(np.mean([p["flip"] for p in per])),
                               mean_rmsd_median=float(np.median(means)),
                               mean_rmsd_mean=float(means.mean()),
                               plddt_median=float(np.median([p["plddt_mean"] for p in per])),
                               per_design=per)
    # paired comparison against baseline for designs present in all scales
    common = None
    for scale in SCALES:
        if scale in data:
            ids = set(data[scale])
            common = ids if common is None else (common & ids)
    common = sorted(common or [])
    out["common_designs"] = len(common)
    if common:
        med = {}
        for scale in SCALES:
            sds = [np.std([data[scale][h][s][0] for s in SEEDS if s in data[scale][h]], ddof=1)
                   for h in common]
            med[str(scale)] = float(np.median(sds))
        out["median_sd_common"] = med

    (OUT / "e5_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{'scale':>6s} {'n':>3s} {'SD median':>10s} {'SD mean':>8s} {'flips':>5s} {'meanRMSD':>9s} {'pLDDT':>6s}")
    for scale in SCALES:
        if str(scale) not in out:
            continue
        v = out[str(scale)]
        print(f"{scale:6.2f} {v['n_designs']:3d} {v['sd_median']:10.2f} {v['sd_mean']:8.2f} "
              f"{v['flip_count']:5d} {v['mean_rmsd_median']:9.2f} {v['plddt_median']:6.1f}")
    print("common designs:", out["common_designs"], out.get("median_sd_common"))


if __name__ == "__main__":
    main()
