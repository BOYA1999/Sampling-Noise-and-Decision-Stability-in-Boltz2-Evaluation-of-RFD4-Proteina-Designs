import csv
import json
import importlib.util
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
project = root / 'data'
outdir = root / 'outputs'
outdir.mkdir(exist_ok=True)
rows = [r for r in csv.DictReader((project / 'e1_metrics.csv').open()) if r['status'] == 'ok']
groups = defaultdict(list)
for r in rows:
    groups[r['h_id']].append((int(r['seed']), float(r['rmsd']), float(r['plddt'])))
for v in groups.values():
    v.sort()
assert len(groups) == 29 and len(rows) == 196

def label(r, p):
    return int(r < 2 and p >= 80)

def majority(vals):
    return int(2 * sum(vals) > len(vals))

rng = np.random.default_rng(41)
pol = defaultdict(list)
for _ in range(400):
    for h, evs in groups.items():
        idx = rng.permutation(len(evs))
        k = max(2, len(evs) // 2)
        ev, ref = [evs[i] for i in idx[:k]], [evs[i] for i in idx[k:]]
        refs = {'maj': majority([label(x[1], x[2]) for x in ref]),
                'mean': label(np.mean([x[1] for x in ref]), np.mean([x[2] for x in ref]))}
        votes = [label(x[1], x[2]) for x in ev]
        preds = {'single': votes[0], 'mean': label(np.mean([x[1] for x in ev]), np.mean([x[2] for x in ev])),
                 'majority': majority(votes)}
        preds['d90'] = label(np.mean([x[1] for x in ev[:2]]), np.mean([x[2] for x in ev[:2]])) if abs(ev[0][1] - 2) <= .62 else votes[0]
        for rn, rv in refs.items():
            for name, pred in preds.items():
                pol[f'{name}_{rn}'].append(int(pred != rv))
old = json.loads((project / 'replay_point_estimates.json').read_text())
for k, v in pol.items():
    assert abs(np.mean(v) - old[k]['disagreement']) < 1e-12, (k, np.mean(v), old[k])
boot_rng = np.random.default_rng(20261102)
indices = boot_rng.integers(0, len(groups), (4000, len(groups)))
replay = {}
for ref in ('maj', 'mean'):
    for name in ('mean', 'majority', 'd90'):
        diff = (np.array(pol[f'{name}_{ref}']) - np.array(pol[f'single_{ref}'])).reshape(400, 29).mean(axis=0)
        draws = diff[indices].mean(axis=1)
        replay[f'{name}_vs_single_{ref}'] = {'delta': float(diff.mean()), 'ci_95': np.percentile(draws, [2.5, 97.5]).tolist()}

e4 = json.loads((outdir / 'e4_results.json').read_text())
ratios = np.array([r['e4_rmsd_sd'] / r['base_rmsd_sd'] for r in e4['per_design'] if r['base_rmsd_sd'] > 0])
assert len(ratios) == 29
sd_index = np.random.default_rng(20261103).integers(0, len(ratios), (4000, len(ratios)))
sd_ratio = {'n': len(ratios), 'median': float(np.median(ratios)),
            'percentile_ci_95': np.percentile(np.median(ratios[sd_index], axis=1), [2.5, 97.5]).tolist()}

spec = importlib.util.spec_from_file_location('risk', root / 'scripts' / 'e2_risk.py')
risk_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_module)
oldrisk = json.loads((outdir / 'e2_risk.json').read_text())
riskdata = oldrisk['per_sequence']
x = np.array([r['d'] for r in riskdata]); k = np.array([r['flips'] for r in riskdata], float); n = np.array([r['n'] for r in riskdata], float)
rng = np.random.default_rng(7)
valid = []; invalid = {'nonnegative_slope': 0, 'nonpositive_or_nonfinite': 0, 'numerical_error': 0}
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        try:
            b = risk_module.fit_logistic(x[idx], k[idx], n[idx])
            v = risk_module.d90(b)
            if v is None:
                invalid['nonnegative_slope'] += 1
            elif not np.isfinite(v) or v <= 0:
                invalid['nonpositive_or_nonfinite'] += 1
            else:
                valid.append(v)
        except (FloatingPointError, np.linalg.LinAlgError):
            invalid['numerical_error'] += 1
risk = {'original_d90': oldrisk['d90'], 'valid': len(valid), 'total': 2000,
        'invalid_breakdown': invalid, 'conditional_percentile_interval': np.percentile(valid, [2.5, 97.5]).tolist()}
assert len(valid) + sum(invalid.values()) == 2000
rev = json.loads((outdir / 'revision_flip_defs.json').read_text())['detail']
risk['definition_mismatches'] = [r['h_id'] for r in riskdata if r['flips'] != rev[r['h_id']]['disagreements']]
result = {'replay_disagreement': {k: float(np.mean(v)) for k, v in pol.items()}, 'replay': replay, 'sd_ratio': sd_ratio, 'risk': risk,
          'method': {'replay_bootstrap': '4000 design resamples; each design contributes the mean paired difference over its 400 original random splits; seed 20261102',
                     'sd_bootstrap': '4000 design resamples of per-design SD ratios; bootstrap median; seed 20261103',
                     'risk_bootstrap': '2000 resamples of designs; reproduce original seed 7, fit, exclusions, and percentile calculation'}}
(outdir / 'stats_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
