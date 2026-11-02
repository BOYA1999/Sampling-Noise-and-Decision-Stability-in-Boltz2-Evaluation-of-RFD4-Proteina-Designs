# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""Revision 2: AF2 reference variants, confusion matrices, tie rules, paired bootstrap."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_metrics():
    rows = list(csv.DictReader((E1 / "e3b_metrics.csv").open()))
    out = defaultdict(dict)
    for r in rows:
        out[r["name"]][int(r["seed"])] = dict(rmsd=float(r["rmsd"]), plddt=float(r["plddt"]),
                                              af2_rmsd=float(r["af2_rmsd"]),
                                              af2_plddt=float(r["af2_plddt"]))
    return out


def lab(rmsd, plddt):
    return int(rmsd < 2 and plddt >= 80)


def majority(vals, tie="fail"):
    n = len(vals)
    s = sum(vals)
    if 2 * s == n:
        return {"fail": 0, "pass": 1, "keep": vals[0]}[tie]
    return int(2 * s > n)


def aggregate(vals, tie="fail"):
    rmsd = float(np.mean([v["rmsd"] for v in vals]))
    plddt = float(np.mean([v["plddt"] for v in vals]))
    return lab(rmsd, plddt)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def main():
    data = load_metrics()
    names = sorted(data)
    missing_af2 = [n for n in names if math.isnan(data[n][0]["af2_rmsd"])]
    print("names", len(names), "missing AF2", len(missing_af2))

    refs = {
        "af2_rmsd_only": {n: int(data[n][0]["af2_rmsd"] < 2) for n in names},
        "af2_rmsd_plddt": {n: lab(data[n][0]["af2_rmsd"], data[n][0]["af2_plddt"]) for n in names},
    }
    print("AF2 positives RMSD-only:", sum(refs["af2_rmsd_only"].values()),
          "RMSD+pLDDT:", sum(refs["af2_rmsd_plddt"].values()))

    policies = {}
    for n in names:
        vals = [data[n][s] for s in sorted(data[n])]
        labels = [lab(v["rmsd"], v["plddt"]) for v in vals]
        policies.setdefault("single", {})[n] = labels[0]
        for k in (2, 3, 5):
            if len(vals) >= k:
                policies.setdefault(f"mean{k}", {})[n] = aggregate(vals[:k])
                policies.setdefault(f"maj{k}_fail", {})[n] = majority(labels[:k], "fail")
        policies.setdefault("maj2_pass", {})[n] = majority(labels[:2], "pass")
        policies.setdefault("maj2_keep", {})[n] = majority(labels[:2], "keep")
        trig = abs(vals[0]["rmsd"] - 2) <= 0.62
        policies.setdefault("d90_mean2", {})[n] = aggregate(vals[:2]) if trig else labels[0]
        policies.setdefault("d90_maj2_fail", {})[n] = majority(labels[:2], "fail") if trig else labels[0]
        policies.setdefault("d90_maj2_pass", {})[n] = majority(labels[:2], "pass") if trig else labels[0]
        policies.setdefault("d90_maj2_keep", {})[n] = majority(labels[:2], "keep") if trig else labels[0]
        trig2 = trig or abs(vals[0]["plddt"] - 80) <= 5
        policies.setdefault("d90_plus_plddt", {})[n] = aggregate(vals[:2]) if trig2 else labels[0]

    rng = np.random.default_rng(31)
    res = {}
    for pname, pv in policies.items():
        res[pname] = {}
        for rname, rv in refs.items():
            common = [n for n in names if n in rv and n in pv]
            tp = sum(1 for n in common if pv[n] == 1 and rv[n] == 1)
            fp = sum(1 for n in common if pv[n] == 1 and rv[n] == 0)
            fn = sum(1 for n in common if pv[n] == 0 and rv[n] == 1)
            tn = sum(1 for n in common if pv[n] == 0 and rv[n] == 0)
            agree = (tp + tn) / len(common)
            # paired bootstrap over designs
            a = np.array([1 if pv[n] == rv[n] else 0 for n in common])
            boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(4000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[pname][rname] = dict(n=len(common), tp=tp, fp=fp, fn=fn, tn=tn,
                                     agreement=float(agree), ci=[float(lo), float(hi)],
                                     sens=float(tp / (tp + fn)) if tp + fn else None,
                                     spec=float(tn / (tn + fp)) if tn + fp else None)
    # paired differences vs single
    diffs = {}
    for pname in policies:
        for rname in refs:
            common = [n for n in names if n in refs[rname] and n in policies[pname]]
            d = np.array([(1 if policies[pname][n] == refs[rname][n] else 0) -
                          (1 if policies["single"][n] == refs[rname][n] else 0) for n in common])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            diffs[f"{pname}_vs_single__{rname}"] = dict(delta=float(d.mean()),
                                                        ci=[float(lo), float(hi)])
    out = dict(refs=refs, policies={k: dict(v) for k, v in policies.items()},
               results=res, diffs=diffs)
    (OUT / "revision2_policies.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"{'policy':18s} {'AF2 RMSD only':>28s} {'AF2 RMSD+pLDDT':>28s}")
    for p in policies:
        cells = []
        for r in ("af2_rmsd_only", "af2_rmsd_plddt"):
            v = res[p][r]
            cells.append(f"{v['agreement']*100:5.1f}% [{v['ci'][0]*100:4.1f},{v['ci'][1]*100:4.1f}]")
        print(f"{p:18s} " + "  ".join(cells))
    print("\nconfusion (AF2 RMSD+pLDDT):")
    for p in policies:
        v = res[p]["af2_rmsd_plddt"]
        print(f"{p:18s} tp={v['tp']:2d} fp={v['fp']:2d} fn={v['fn']:2d} tn={v['tn']:2d}")


if __name__ == "__main__":
    main()
