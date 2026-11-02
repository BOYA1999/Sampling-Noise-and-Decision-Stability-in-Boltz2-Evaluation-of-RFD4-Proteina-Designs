# Path adaptation for portable saved-metrics replay; see README.
#!/usr/bin/env python3
"""E1 summary: within-sequence evaluation noise, between-sequence spread, ICC, qualification flips.

Run with Python and NumPy as described in README.md.
"""
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


def load_ok():
    rows = []
    with (E1 / "e1_metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            for k in ("length", "seed"):
                r[k] = int(r[k])
            for k in ("plddt", "ptm", "rmsd"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def icc(values_by_group):
    """One-way random effects ICC(1) from nested samples."""
    groups = [np.asarray(v, dtype=float) for v in values_by_group if len(v) > 1]
    if len(groups) < 2:
        return None, None, None
    k = np.mean([len(g) for g in groups])
    grand = np.mean([g.mean() for g in groups])
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb = len(groups) - 1
    dfw = sum(len(g) - 1 for g in groups)
    msb = ssb / dfb
    msw = ssw / dfw
    var_b = max(0.0, (msb - msw) / k)
    var_w = msw
    icc_val = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return float(icc_val), float(np.sqrt(var_b)), float(np.sqrt(var_w))


def main():
    rows = load_ok()
    by_seq = defaultdict(lambda: {"plddt": [], "rmsd": [], "ptm": []})
    for r in rows:
        by_seq[r["h_id"]]["plddt"].append(r["plddt"])
        by_seq[r["h_id"]]["rmsd"].append(r["rmsd"])
        by_seq[r["h_id"]]["ptm"].append(r["ptm"])

    out = {"n_runs": len(rows), "n_sequences": len(by_seq), "per_sequence": {}}
    icc_p = icc([v["plddt"] for v in by_seq.values()])
    icc_r = icc([v["rmsd"] for v in by_seq.values()])
    out["icc"] = {"plddt": icc_p, "rmsd": icc_r}

    flips = 0
    for h, v in by_seq.items():
        q = [1 if (rm < 2.0 and pl >= 80) else 0 for rm, pl in zip(v["rmsd"], v["plddt"])]
        flip = max(q) != min(q)
        flips += flip
        out["per_sequence"][h] = {
            "n": len(v["plddt"]),
            "plddt_mean": float(np.mean(v["plddt"])),
            "plddt_sd": float(np.std(v["plddt"], ddof=1)) if len(v["plddt"]) > 1 else None,
            "rmsd_mean": float(np.mean(v["rmsd"])),
            "rmsd_sd": float(np.std(v["rmsd"], ddof=1)) if len(v["rmsd"]) > 1 else None,
            "ptm_mean": float(np.mean(v["ptm"])),
            "qualified_flips": bool(flip),
        }
    out["qualification_flips"] = int(flips)

    (OUT / "e1_summary.json").write_text(json.dumps(out, indent=2))

    print(f"runs={len(rows)} sequences={len(by_seq)} qualification_flips={flips}")
    print(f"ICC pLDDT={icc_p[0]:.3f} (between_sd={icc_p[1]:.3f}, within_sd={icc_p[2]:.3f})")
    print(f"ICC RMSD ={icc_r[0]:.3f} (between_sd={icc_r[1]:.3f}, within_sd={icc_r[2]:.3f})")
    print()
    print("| h_id | n | pLDDT mean±sd | RMSD mean±sd | flip |")
    print("|---|---|---|---|---|")
    for h, v in sorted(out["per_sequence"].items(), key=lambda kv: -(kv[1]["rmsd_sd"] or 0)):
        ps = f"{v['plddt_mean']:.2f}±{v['plddt_sd']:.2f}" if v["plddt_sd"] is not None else f"{v['plddt_mean']:.2f}"
        rs = f"{v['rmsd_mean']:.2f}±{v['rmsd_sd']:.2f}" if v["rmsd_sd"] is not None else f"{v['rmsd_mean']:.2f}"
        print(f"| {h} | {v['n']} | {ps} | {rs} | {'Y' if v['qualified_flips'] else ''} |")


if __name__ == "__main__":
    main()
