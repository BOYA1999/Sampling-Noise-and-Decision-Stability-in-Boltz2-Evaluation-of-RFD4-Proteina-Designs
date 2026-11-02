# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""Recompute flip statistics with the stated definition: single draw vs majority of the
remaining draws, ties resolved to fail (leave-one-out)."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def lab(r):
    return int(float(r["rmsd"]) < 2 and float(r["plddt"]) >= 80)


def main():
    rows = [r for r in csv.DictReader((E1 / "e1_metrics.csv").open()) if r["status"] == "ok"]
    g = defaultdict(list)
    for r in rows:
        g[r["h_id"]].append(r)
    flip_designs = []
    detail = {}
    for h, rs in g.items():
        labels = [lab(r) for r in rs]
        n = len(labels)
        dis = 0
        for i in range(n):
            others = labels[:i] + labels[i + 1:]
            s = sum(others)
            ref = 1 if (2 * s > len(others)) else 0  # tie -> fail
            if labels[i] != ref:
                dis += 1
        detail[h] = dict(n=n, disagreements=dis, rate=dis / n,
                         means=float(np.mean([float(r["rmsd"]) for r in rs])),
                         dist=abs(float(np.mean([float(r["rmsd"]) for r in rs])) - 2))
        if dis > 0:
            flip_designs.append(h)
    print("designs with >=1 leave-one-out disagreement:", len(flip_designs), "/", len(g))
    for h in flip_designs:
        d = detail[h]
        print(f"  {h} n={d['n']} dis={d['disagreements']} rate={d['rate']:.2f} dist={d['dist']:.2f}")
    # near-threshold six: rates by distance
    print("\nnear-threshold designs (n>=12):")
    for h, d in sorted(detail.items(), key=lambda kv: kv[1]["dist"]):
        if d["n"] >= 12:
            print(f"  dist={d['dist']:.2f} n={d['n']} dis={d['disagreements']} rate={d['rate']:.2f}")
    out = dict(flip_designs=flip_designs, detail=detail,
               flip_rate=len(flip_designs) / len(g))
    (OUT / "revision_flip_defs.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
