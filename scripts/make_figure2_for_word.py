from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / 'data'
out = root / 'outputs' / 'figures'
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.grid': False, 'savefig.bbox': None})
rows = [r for r in csv.DictReader((base / 'e1_metrics.csv').open(encoding='utf-8')) if r['status'] == 'ok']
by = defaultdict(dict)
meta = {}
for r in rows:
    h, seed = r['h_id'], int(r['seed'])
    assert seed not in by[h]
    by[h][seed] = float(r['rmsd'])
    fields = (int(r['length']), r['method'], r['budget'], int(r['block']))
    assert h not in meta or fields == meta[h]
    meta[h] = fields
assert len(rows) == 196 and len(by) == 29
order = sorted(by, key=lambda h: np.mean(list(by[h].values())))
fig = plt.figure(figsize=(6.15, 5.75), layout='constrained')
grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
ax = fig.add_subplot(grid[0, :])
for i, h in enumerate(order):
    vals = np.array(list(by[h].values()))
    ax.scatter(np.full(len(vals), i), vals, s=12, alpha=.65,
               color='#b26658' if vals.min() < 2 < vals.max() else '#536c78', edgecolors='none')
ax.axhline(2, color='#a0794e', linestyle='--', linewidth=1, label='2 Å RMSD threshold')
ax.set_yscale('log')
ax.set_xticks(range(len(order)), [h[4:10] for h in order], rotation=90, fontsize=7.5)
ax.set_xlabel('Design ID (truncated; ordered by mean RMSD)')
ax.set_ylabel('Cα RMSD to design model (Å)')
ax.set_title('A  Evaluation draws by design', loc='left', weight='bold')
ax.legend(loc='upper left', frameon=False)
colors = {'bon': '#536c78', 'random': '#ad806a', 'online': '#6a887b'}
markers = {'bon': 'o', 'random': 's', 'online': '^'}
names = {'bon': 'Best-of-N', 'random': 'Random', 'online': 'Online search'}
rank_out = {}
for j, length in enumerate((150, 250)):
    ax = fig.add_subplot(grid[1, j])
    ids = [h for h in by if meta[h][0] == length]
    methods = sorted({meta[h][1] for h in ids})
    common = sorted(set.intersection(*(set(by[h]) for h in ids)))
    assert common == (list(range(5)) if length == 150 else list(range(6)))
    assert set(methods) == set(colors)
    ranks = defaultdict(list)
    top = []
    for seed in common:
        means = {m: np.mean([by[h][seed] for h in ids if meta[h][1] == m]) for m in methods}
        ranked = sorted(methods, key=means.get)
        top.append(ranked[0])
        for rank, m in enumerate(ranked, 1):
            ranks[m].append(rank)
    full = {m: np.mean([np.mean(list(by[h].values())) for h in ids if meta[h][1] == m]) for m in methods}
    assert all(m == min(full, key=full.get) for m in top)
    rank_out[str(length)] = {'seeds': common, 'ranks': dict(ranks), 'baseline_top': min(full, key=full.get)}
    for m in ('bon', 'random', 'online'):
        ax.plot(common, ranks[m], marker=markers[m], markersize=4.5, color=colors[m],
                linewidth=1.1, label=names[m])
    ax.set_xticks(common)
    ax.set_ylim(3.35, .65)
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel('Evaluator seed')
    ax.set_ylabel('Rank' if j == 0 else '')
    ax.set_title(('B  ' if j == 0 else '') + f'{length} residues', loc='left', weight='bold')
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    if j == 1:
        ax.legend(loc='lower left', bbox_to_anchor=(-.05, -0.64), frameon=False, ncol=3,
                  columnspacing=1, handletextpad=.4)
expected = json.loads((root / 'data' / 'figure2_reference.json').read_text(encoding='utf-8'))['ranking']
assert rank_out == expected
path = out / 'figure2_for_word.png'
fig.savefig(path, dpi=300)
plt.close(fig)
print('png:', path, 'pixels:', (1845, 1725))
print('source SHA256:', hashlib.sha256((base / 'e1_metrics.csv').read_bytes()).hexdigest())
print('data:', len(rows), 'evaluations,', len(by), 'designs; ranks match figure_checks.json exactly')
print('top-1: 150/250 aa all 11 seeds match frozen figure')
