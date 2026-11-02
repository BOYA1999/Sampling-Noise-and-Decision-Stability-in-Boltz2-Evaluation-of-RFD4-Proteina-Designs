# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""Revision 2: threshold sweep under unified replay definitions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(51)
THR = [0.0, 0.25, 0.5, 0.62, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def lab(rmsd, plddt):
    return int(rmsd < 2 and plddt >= 80)


def major(vals, tie="fail"):
    n = len(vals)
    s = sum(vals)
    if 2 * s == n:
        return 0 if tie == "fail" else (1 if tie == "pass" else vals[0])
    return int(2 * s > n)


def main():
    rows = [r for r in csv.DictReader((E1 / "e1_metrics.csv").open()) if r["status"] == "ok"]
    g = defaultdict(list)
    for r in rows:
        g[r["h_id"]].append(dict(seed=int(r["seed"]), rmsd=float(r["rmsd"]), plddt=float(r["plddt"])))
    for h in g:
        g[h].sort(key=lambda x: x["seed"])
    # pre-generate splits once
    splits = []
    for _ in range(400):
        sp = {}
        for h, evs in g.items():
            idx = RNG.permutation(len(evs))
            k = max(2, len(evs) // 2)
            sp[h] = (idx[:k], idx[k:])
        splits.append(sp)

    out = []
    for thr in THR:
        errs_maj, errs_mean, costs = [], [], []
        for sp in splits:
            for h, (ev_idx, ref_idx) in sp.items():
                evs = g[h]
                ev = [evs[i] for i in ev_idx]
                ref = [evs[i] for i in ref_idx]
                ref_maj = major([lab(x["rmsd"], x["plddt"]) for x in ref], "fail")
                ref_mean = lab(float(np.mean([x["rmsd"] for x in ref])),
                               float(np.mean([x["plddt"] for x in ref])))
                trig = abs(ev[0]["rmsd"] - 2) <= thr
                if trig and len(ev) >= 2:
                    r = float(np.mean([ev[0]["rmsd"], ev[1]["rmsd"]]))
                    p = float(np.mean([ev[0]["plddt"], ev[1]["plddt"]]))
                    pred = lab(r, p)
                    cost = 2
                else:
                    pred = lab(ev[0]["rmsd"], ev[0]["plddt"])
                    cost = 1
                errs_maj.append(pred != ref_maj)
                errs_mean.append(pred != ref_mean)
                costs.append(cost)
        out.append(dict(threshold=thr, error_majority=float(np.mean(errs_maj)),
                        error_mean=float(np.mean(errs_mean)), cost=float(np.mean(costs))))

    (OUT / "revision2_sweep.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(f"{r['threshold']:.2f}  err_maj={r['error_majority']*100:.2f}%  "
              f"err_mean={r['error_mean']*100:.2f}%  cost={r['cost']:.2f}")


if __name__ == "__main__":
    main()
