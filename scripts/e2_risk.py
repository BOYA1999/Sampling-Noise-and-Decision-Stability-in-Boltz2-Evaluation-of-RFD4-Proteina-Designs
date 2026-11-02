# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""E2 risk analysis: flip probability vs distance to the 2 A qualification threshold.

For each design: mean RMSD over all evaluations, d = |mean - 2|, flip = an evaluation that
disagrees with the design's majority qualification label. Fits a ridge-regularised binomial
logistic P(flip | d) and reports d90 (P = 0.1) with a conditional sequence-level bootstrap percentile interval.
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


def load():
    rows = [r for r in csv.DictReader((E1 / "e1_metrics.csv").open()) if r["status"] == "ok"]
    g = defaultdict(list)
    for r in rows:
        g[r["h_id"]].append(r)
    out = []
    for h, rs in g.items():
        rmsd = np.array([float(x["rmsd"]) for x in rs])
        plddt = np.array([float(x["plddt"]) for x in rs])
        lab = (rmsd < 2) & (plddt >= 80)
        maj = lab.mean() >= 0.5
        flips = int((lab != maj).sum())
        out.append(dict(h_id=h, length=int(rs[0]["length"]), n=len(rs),
                        mean=float(rmsd.mean()), d=abs(float(rmsd.mean()) - 2),
                        flips=flips, rate=flips / len(rs), majority=bool(maj)))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fit_logistic(x, k, n, lam=2.0, iters=100):
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(n * p * (1 - p), 1e-8, None)
        grad = X.T @ (k - n * p) - lam * np.r_[0.0, beta[1]]
        H = X.T @ (X * w[:, None]) + lam * np.diag([0.0, 1.0])
        beta += np.linalg.solve(H, grad)
    return beta


def d90(beta):
    target = np.log(0.1 / 0.9)
    if beta[1] >= 0:
        return None
    return float((target - beta[0]) / beta[1])


def main():
    data = load()
    data.sort(key=lambda r: r["d"])
    print(f"{'h_id':24s} {'len':>4s} {'n':>3s} {'mean':>6s} {'d':>5s} {'flips':>6s} {'rate':>6s}")
    for r in data:
        print(f"{r['h_id']:24s} {r['length']:4d} {r['n']:3d} {r['mean']:6.2f} {r['d']:5.2f} "
              f"{r['flips']:6d} {r['rate']:6.2f}")

    x = np.array([r["d"] for r in data])
    k = np.array([r["flips"] for r in data], dtype=float)
    n = np.array([r["n"] for r in data], dtype=float)
    beta = fit_logistic(x, k, n)
    est = d90(beta)
    print(f"\nlogistic: beta0={beta[0]:.3f} beta1={beta[1]:.3f}  d90={est if est is not None else 'n/a'}")

    rng = np.random.default_rng(7)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(data), len(data))
        b = fit_logistic(x[idx], k[idx], n[idx])
        v = d90(b)
        if v is not None and np.isfinite(v) and v > 0:
            boot.append(v)
    ci = None
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci = (float(lo), float(hi))
        print(f"conditional bootstrap d90 percentile interval: [{lo:.2f}, {hi:.2f}]  (valid={len(boot)}/2000)")

    flips = [r for r in data if r["flips"] > 0]
    stable = [r for r in data if r["flips"] == 0]
    far_flip = max(r["d"] for r in flips)
    near_stable = min(r["d"] for r in stable)
    print(f"\nfurthest flipping design: d={far_flip:.2f}; nearest stable design: d={near_stable:.2f}")

    out = dict(per_sequence=data, beta=[float(b) for b in beta], d90=est,
               bootstrap_ci=ci, furthest_flip=far_flip, nearest_stable=near_stable)
    (OUT / "e2_risk.json").write_text(json.dumps(out, indent=2, default=float))
    print("wrote e2_risk.json")


if __name__ == "__main__":
    main()
