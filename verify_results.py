from pathlib import Path
import csv, hashlib, json, math

root = Path(__file__).resolve().parent
data = root / 'data'
out = root / 'outputs'
for line in (data / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
for name, rows in [('e1_metrics.csv', 196), ('e3b_metrics.csv', 250), ('e4_metrics.csv', 116), ('e5_metrics.csv', 261)]:
    with (data / name).open(newline='', encoding='utf-8') as f:
        assert sum(1 for _ in csv.DictReader(f)) == rows, name

def read(name):
    return json.loads((out / name).read_text(encoding='utf-8'))
def near(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)

e1 = read('e1_summary.json')
assert (e1['n_runs'], e1['n_sequences'], e1['qualification_flips']) == (196, 29, 3)
stats = read('stats_results.json')
near(stats['sd_ratio']['median'], 0.03847026031361092)
near(stats['sd_ratio']['percentile_ci_95'][0], 0.024558715793588114)
near(stats['sd_ratio']['percentile_ci_95'][1], 0.06436932339116096)
assert (stats['risk']['valid'], stats['risk']['total']) == (1610, 2000)
near(stats['risk']['original_d90'], 0.6192696498608674)
near(stats['replay']['d90_vs_single_maj']['delta'], 0.002327586206896552)
p = read('revision2_policies.json')
assert p['refs']['af2_rmsd_only'] == p['refs']['af2_rmsd_plddt']
assert p['policies']['single'] == p['policies']['d90_maj2_keep']
assert p['diffs']['d90_maj2_keep_vs_single__af2_rmsd_only'] == {'delta': 0.0, 'ci': [0.0, 0.0]}
assert p['results']['single']['af2_rmsd_only']['ci'] == [0.66, 0.88]
near(p['results']['mean5']['af2_rmsd_only']['agreement'], 0.86)
for name in ['figure1_layers.png', 'figure2_sampling_and_ranking.png', 'figure2_for_word.png', 'figure3_risk.png', 'figure4_replay_sweep.png', 'figure5_noise_intervention.png', 'figure6_gradient.png']:
    assert (out / 'figures' / name).is_file(), name
print('PASS: input hashes, row counts, selected manuscript values, AF2 reference policy, and generated figures.')
