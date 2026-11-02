from pathlib import Path
import csv
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

root = Path(__file__).resolve().parents[1]
base = root / 'data'
out = root / 'outputs' / 'figures'
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.grid': False})

def save(fig, name):
    fig.savefig(out / (name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)

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

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 12)
ax.set_ylim(-.2, 4)
ax.axis('off')
def box(x, y, w, h, title, body, face):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.08',
                           edgecolor='#53636a', facecolor=face, linewidth=0.8)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*0.70, title, ha='center', va='center', weight='bold', fontsize=9)
    ax.text(x+w/2, y+h*0.32, body, ha='center', va='center', fontsize=8)
box(.2, 1.40, 2.55, 1.05, 'Fixed design', 'sequence + design model', '#eef2f2')
box(3.5, 1.40, 2.55, 1.05, 'Evaluator samples', 'Boltz2; varying seed', '#e8efee')
for y, title, body in [(3.08, 'Confidence', 'complex pLDDT'), (2.18, 'Structure', 'Cα RMSD to design model'),
                       (1.28, 'Qualification', 'RMSD < 2 Å and pLDDT ≥ 80'), (0.38, 'Method ranking', 'cross-design mean RMSD by length')]:
    box(8.25, y-0.30, 3.35, 0.70, title, body, '#f3f3f1')
ax.add_patch(FancyArrowPatch((2.87, 1.92), (3.35, 1.92), arrowstyle='-|>', mutation_scale=10, linewidth=.9, color='#53636a'))
for y in (3.13, 2.23, 1.33, .43):
    ax.add_patch(FancyArrowPatch((6.18, 1.92), (8.08, y), arrowstyle='-|>', mutation_scale=9, linewidth=.8, color='#788689'))
save(fig, 'figure1_layers')

order = sorted(by, key=lambda h: np.mean(list(by[h].values())))
fig = plt.figure(figsize=(10.2, 4.7), layout='constrained')
grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
ax = fig.add_subplot(grid[0])
for i, h in enumerate(order):
    vals = np.array(list(by[h].values()))
    ax.scatter(np.full(len(vals), i), vals, s=12, alpha=.65,
               color='#b26658' if vals.min() < 2 < vals.max() else '#536c78', edgecolors='none')
ax.axhline(2, color='#a0794e', linestyle='--', linewidth=1, label='2 Å RMSD threshold')
ax.set_yscale('log')
ax.set_xticks(range(len(order)), [h[4:10] for h in order], rotation=90, fontsize=6)
ax.set_xlabel('Design ID (truncated; ordered by mean RMSD)')
ax.set_ylabel('Cα RMSD to design model (Å)')
ax.set_title('A  Evaluation draws by design', loc='left', weight='bold')
ax.legend(loc='upper left', frameon=False)

colors = {'bon': '#536c78', 'random': '#ad806a', 'online': '#6a887b'}
markers = {'bon': 'o', 'random': 's', 'online': '^'}
names = {'bon': 'Best-of-N', 'random': 'Random', 'online': 'Online search'}
rank_out = {}
for j, length in enumerate((150, 250), start=1):
    ax = fig.add_subplot(grid[j])
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
    ax.set_ylabel('Rank' if j == 1 else '')
    ax.set_title(('B  ' if j == 1 else '') + f'{length} residues', loc='left', weight='bold')
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    if j == 2:
        ax.legend(loc='upper right', bbox_to_anchor=(1.04, -.22), frameon=False)
save(fig, 'figure2_sampling_and_ranking')

d = json.loads((root / 'outputs' / 'revision2_sweep.json').read_text(encoding='utf-8'))
fig, ax = plt.subplots(figsize=(5.6, 3.4), layout='constrained')
ax.plot([r['threshold'] for r in d], [100*r['error_majority'] for r in d], 'o-', color='#ad806a', label='Majority reference', markersize=4)
ax.plot([r['threshold'] for r in d], [100*r['error_mean'] for r in d], 's-', color='#536c78', label='Mean reference', markersize=4)
ax.axvline(.62, color='#7c8884', linestyle='--', linewidth=.9, label='Exploratory $d_{90}$ = 0.62 Å')
ax.set_xlabel('Trigger distance from 2 Å threshold (Å)')
ax.set_ylabel('Disagreement with reference (%)')
ax.set_title('Replay disagreement by trigger distance', loc='left', weight='bold')
ax.legend(frameon=False)
save(fig, 'figure4_replay_sweep')

d = json.loads((root / 'outputs' / 'e4_results.json').read_text(encoding='utf-8'))
rs = d['per_design']
b = np.array([r['base_rmsd_sd'] for r in rs])
n = np.array([r['e4_rmsd_sd'] for r in rs])
assert len(rs) == 29 and min(b.min(), n.min()) > 0
lo, hi = min(b.min(), n.min()) / 1.6, max(b.max(), n.max()) * 1.6
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), layout='constrained', gridspec_kw={'width_ratios': [1.5, 1]})
ax = axes[0]
ax.scatter(b, n, s=22, color='#536c78', alpha=.85)
ax.plot([lo, hi], [lo, hi], '--', color='#a0794e', linewidth=1, label='Equal SD')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('Baseline within-design RMSD SD (Å)')
ax.set_ylabel('Zero-noise within-design RMSD SD (Å)')
ax.set_title('A  Paired designs (n = 29)', loc='left', weight='bold')
ax.legend(frameon=False)
ax = axes[1]
flips = [d['base_flips'], d['e4_flips']]
bars = ax.bar(['Baseline', 'Zero noise'], flips, color=['#8b9997', '#ad806a'], width=.54)
for bar, count in zip(bars, flips):
    ax.text(bar.get_x()+bar.get_width()/2, count+.07, str(count), ha='center', va='bottom')
ax.set_ylim(0, max(flips)+.8)
ax.set_yticks(range(max(flips)+1))
ax.set_ylabel('Designs with qualification flips')
ax.set_title('B  Four seeds per condition', loc='left', weight='bold')
save(fig, 'figure5_noise_intervention')

