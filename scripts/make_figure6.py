from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'outputs' / 'figures'
out.mkdir(parents=True, exist_ok=True)
data = json.loads((root / 'outputs' / 'e5_results.json').read_text(encoding='utf-8'))
keys = ['0.0', '0.25', '0.5', '0.75', '1.003']
xs = np.array([float(k) for k in keys])
design_ids = {p['h_id'] for p in data[keys[0]]['per_design']}
for k in keys:
    rows = data[k]['per_design']
    assert len(rows) == 29 and {p['h_id'] for p in rows} == design_ids
    assert all(p['n'] == 3 and p['sd'] > 0 for p in rows)
    sds = np.array([p['sd'] for p in rows])
    means = np.array([p['mean'] for p in rows])
    assert np.isclose(np.median(sds), data[k]['sd_median'])
    assert np.isclose(np.mean(sds > .2), data[k]['sd_frac_gt_02'])
    assert np.isclose(np.median(means), data[k]['mean_rmsd_median'])

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.grid': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), layout='constrained',
                         gridspec_kw={'wspace':.3})
ax = axes[0]
rng = np.random.default_rng(0)
for x, k in zip(xs, keys):
    vals = np.array([p['sd'] for p in data[k]['per_design']])
    ax.scatter(x + rng.uniform(-.022, .022, len(vals)), vals, s=11, color='#536c78',
               alpha=.5, edgecolors='none', zorder=2)
med = np.array([data[k]['sd_median'] for k in keys])
q1 = np.array([data[k]['sd_q1'] for k in keys])
q3 = np.array([data[k]['sd_q3'] for k in keys])
band = ax.fill_between(xs, q1, q3, color='#ad806a', alpha=.2, label='Interquartile range', zorder=1)
line, = ax.plot(xs, med, 'o-', color='#ad806a', linewidth=1.3, markersize=4, label='Median SD', zorder=3)
points = ax.scatter([], [], s=20, color='#536c78', alpha=.6, label='Individual designs')
ax.set_yscale('log')
ax.set_xlabel('Per-step noise scale')
ax.set_ylabel('Within-design RMSD SD (Å)')
ax.set_xticks(xs, ['0', '0.25', '0.5', '0.75', '1.003'])
ax.set_title('A  Within-design RMSD spread', loc='left', weight='bold')
ax.legend(handles=[line, band, points], loc='upper left', frameon=False)

ax = axes[1]
tail = np.array([100*data[k]['sd_frac_gt_02'] for k in keys])
means = np.array([data[k]['mean_rmsd_median'] for k in keys])
tail_line, = ax.plot(xs, tail, 's-', color='#ad806a', linewidth=1.4, markersize=4,
                     label='SD > 0.2 Å (left axis)')
ax.set_ylim(0, 35)
ax.set_yticks([0, 10, 20, 30])
ax.set_ylabel('Designs with SD > 0.2 Å (%)', color='#8c604f')
ax.tick_params(axis='y', colors='#8c604f')
ax.set_xlabel('Per-step noise scale')
ax.set_xticks(xs, ['0', '0.25', '0.5', '0.75', '1.003'])
ax.set_title('B  Tail fraction and mean RMSD', loc='left', weight='bold')
ax2 = ax.twinx()
mean_line, = ax2.plot(xs, means, 'o--', color='#536c78', linewidth=1.3, markersize=4,
                      label='Median of design means (right axis)')
ax2.set_ylim(0, .6)
ax2.set_yticks([0, .2, .4, .6])
ax2.set_ylabel('Median of design mean RMSD (Å)', color='#536c78')
ax2.tick_params(axis='y', colors='#536c78')
ax2.spines['right'].set_visible(True)
ax.legend(handles=[tail_line, mean_line], loc='upper left', frameon=False, fontsize=7)

for suffix, kwargs in (('png', {'dpi':300}), ('pdf', {})):
    fig.savefig(out / f'figure6_gradient.{suffix}', bbox_inches='tight', **kwargs)
plt.close(fig)
print('All scales: 29 matched designs, seeds 0–2; median SD:', ', '.join(f'{x:.5f}' for x in med))
print('Tail fraction (%):', ', '.join(f'{x:.2f}' for x in tail))
print('Median of design mean RMSD:', ', '.join(f'{x:.3f}' for x in means))
print('Minimum plotted SD:', min(p['sd'] for k in keys for p in data[k]['per_design']))
