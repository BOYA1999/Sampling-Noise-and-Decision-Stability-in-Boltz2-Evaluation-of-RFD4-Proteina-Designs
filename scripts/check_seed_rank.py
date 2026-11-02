# Path adaptation for portable saved-metrics replay; see README.
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
rows = [r for r in csv.DictReader((E1 / "e1_metrics.csv").open()) if r["status"] == "ok"]
g = defaultdict(dict); meta = {}; by = defaultdict(list)
for r in rows:
    h = r["h_id"]; s = int(r["seed"])
    g[h][s] = float(r["rmsd"]); by[h].append(float(r["rmsd"]))
    meta[h] = (int(r["length"]), r["method"])
design = {h: dict(mean=float(np.mean(v)), length=meta[h][0], method=meta[h][1]) for h, v in by.items()}

for L in (150, 250):
    ds = [h for h in design if design[h]["length"] == L]
    methods = sorted({design[h]["method"] for h in ds})
    common = sorted(set.intersection(*[set(g[h]) for h in ds]))
    full = {m: np.mean([design[h]["mean"] for h in ds if design[h]["method"] == m]) for m in methods}
    full_top = min(full, key=full.get)
    flips = 0
    for s in common:
        means = {m: np.mean([g[h][s] for h in ds if design[h]["method"] == m]) for m in methods}
        top = min(means, key=means.get)
        if top != full_top:
            flips += 1
    print(f"L{L}: common seeds={common} full_top={full_top} flips={flips}/{len(common)}")

