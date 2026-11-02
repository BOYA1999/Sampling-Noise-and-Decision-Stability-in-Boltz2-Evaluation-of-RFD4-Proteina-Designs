from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / 'outputs'
out = base / 'figures'
out.mkdir(parents=True, exist_ok=True)
data = json.loads((base / 'e2_risk.json').read_text(encoding='utf-8'))
stats = json.loads((base / 'stats_results.json').read_text(encoding='utf-8'))['risk']
rows = data['per_sequence']
assert len(rows) == 29 and stats['valid'] == 1610 and stats['total'] == 2000
assert np.allclose(data['bootstrap_ci'], stats['conditional_percentile_interval'])
assert np.isclose(data['d90'], stats['original_d90'])
x = np.array([r['d'] for r in rows])
y = np.array([r['rate'] for r in rows])
n = np.array([r['n'] for r in rows])
assert ((x > 3.2) & (x < 3.35)).sum() == 0

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.grid': False, 'pdf.fonttype': 42})
fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True,
                                  gridspec_kw={'width_ratios':[4.2, 1], 'wspace': .06})
colors = {'points':'#536c78', 'curve':'#ad806a', 'threshold':'#6a887b'}
lo, hi = stats['conditional_percentile_interval']
for ax, bounds in ((left, (0, 3.2)), (right, (3.35, 43.0))):
    keep = (x >= bounds[0]) & (x <= bounds[1])
    ax.scatter(x[keep], y[keep], s=18 + 4*n[keep], color=colors['points'], alpha=.78, zorder=3)
    grid = np.linspace(*bounds, 300)
    eta = data['beta'][0] + data['beta'][1] * grid
    ax.plot(grid, 1 / (1 + np.exp(-eta)), color=colors['curve'], linewidth=1.5, zorder=2)
    ax.axhline(.1, color='#9a9a95', linewidth=.8, linestyle=':')
    ax.set_xlim(*bounds)
    ax.set_ylim(-.02, .40)
    ax.set_yticks([0, .1, .2, .3, .4])
left.axvspan(lo, hi, color='#6a887b', alpha=.12, zorder=0)
left.axvline(data['d90'], color=colors['threshold'], linewidth=1.0, linestyle='--')
left.set_ylabel('Fraction of evaluations that disagree')
fig.supxlabel('Distance of design mean RMSD from 2 Å threshold (Å)', y=.04, fontsize=9)
left.set_title('Disagreement by distance to threshold', loc='left', fontsize=10, weight='bold')
left.spines['right'].set_visible(False)
right.spines['left'].set_visible(False)
right.tick_params(axis='y', which='both', left=False, labelleft=False)
right.set_xticks([5, 20, 40])
for ax, xpos in ((left, 1), (right, 0)):
    ax.plot((xpos-.014, xpos+.014), (-.015, .015), transform=ax.transAxes,
            color='#6b7280', linewidth=1, clip_on=False)
left.scatter([], [], s=42, color=colors['points'], label='Design (size ∝ evaluations)')
left.plot([], [], color=colors['curve'], linewidth=1.5, label='Ridge logistic fit')
left.axvline(-1, color=colors['threshold'], linewidth=1.0, linestyle='--', label='$d_{90}$ = 0.62 Å')
left.legend(loc='upper right', fontsize=8, frameon=False)
fig.text(.11, .10, 'Conditional 2.5–97.5 percentile interval: 0.10–0.84 Å; '
         '1,610 / 2,000 eligible design resamples', fontsize=8, color='#4b5563')
fig.subplots_adjust(left=.11, right=.97, bottom=.24, top=.88)
for suffix, kwargs in (('png', {'dpi':300}), ('pdf', {})):
    fig.savefig(out / f'figure3_risk.{suffix}', bbox_inches='tight', **kwargs)
plt.close(fig)
print(f'{len(rows)} designs, {int((x<=3.2).sum())} left, {int((x>=3.35).sum())} right; '
      f'{stats["valid"]}/{stats["total"]} eligible; interval {lo:.5f}–{hi:.5f}')
