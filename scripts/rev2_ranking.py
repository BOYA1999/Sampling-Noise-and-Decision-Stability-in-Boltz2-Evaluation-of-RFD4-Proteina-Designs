# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""Revision 2: ranking sensitivity — median/IQR/trimmed mean and leave-one-design-out."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    rows = [r for r in csv.DictReader((E1 / "e1_metrics.csv").open()) if r["status"] == "ok"]
    by = defaultdict(list)
    meta = {}
    for r in rows:
        by[r["h_id"]].append(float(r["rmsd"]))
        meta[r["h_id"]] = (int(r["length"]), r["method"])
    design = {h: dict(mean=float(np.mean(v)), length=meta[h][0], method=meta[h][1]) for h, v in by.items()}
    out = {}
    for L in (100, 150, 250):
        ds = [h for h in design if design[h]["length"] == L]
        methods = sorted({design[h]["method"] for h in ds})
        desc = {}
        for m in methods:
            vals = np.array([design[h]["mean"] for h in ds if design[h]["method"] == m])
            if len(vals) >= 3:
                s = np.sort(vals)
                trim = float(np.mean(s[1:-1]))
            else:
                trim = None
            desc[m] = dict(n=int(len(vals)), mean=float(vals.mean()),
                           median=float(np.median(vals)),
                           q1=float(np.percentile(vals, 25)), q3=float(np.percentile(vals, 75)),
                           trimmed_mean=trim, vals=[float(x) for x in vals])
        # leave-one-design-out ranking: drop one design, recompute means, top-1
        loo = []
        for h in ds:
            rest = [x for x in ds if x != h]
            means = {}
            ok = True
            for m in methods:
                v = [design[x]["mean"] for x in rest if design[x]["method"] == m]
                if not v:
                    ok = False
                    break
                means[m] = float(np.mean(v))
            if ok:
                loo.append(dict(dropped=h, top1=min(means, key=means.get), means=means))
        top_counts = defaultdict(int)
        for r in loo:
            top_counts[r["top1"]] += 1
        out[str(L)] = dict(n_designs=len(ds), methods=methods, descriptive=desc,
                           loo_n=len(loo), loo_top1=dict(top_counts))
    (OUT / "revision2_ranking.json").write_text(json.dumps(out, indent=2, default=float))
    for L, v in out.items():
        print(L, "n=", v["n_designs"], "methods=", v["methods"])
        for m, d in v["descriptive"].items():
            print("  ", m, "n=", d["n"], "mean=", round(d["mean"], 2), "median=", round(d["median"], 2),
                  "IQR=", [round(d["q1"], 2), round(d["q3"], 2)], "trim=",
                  None if d["trimmed_mean"] is None else round(d["trimmed_mean"], 2))
        print("   LOO top1 counts:", v["loo_top1"], "of", v["loo_n"])


if __name__ == "__main__":
    main()
