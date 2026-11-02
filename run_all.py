from pathlib import Path
import subprocess, sys
root = Path(__file__).resolve().parent
scripts = ['e1_summary.py', 'rev_defs.py', 'e2_risk.py', 'rev2_policies.py', 'rev2_ranking.py', 'rev2_sweep.py', 'e4_analyze.py', 'e5_analyze.py', 'check_seed_rank.py', 'recalculate.py', 'make_figures.py', 'make_figure3.py', 'make_figure6.py', 'make_figure2_for_word.py']
for script in scripts:
    print(f'\n=== {script} ===', flush=True)
    subprocess.run([sys.executable, str(root / 'scripts' / script)], cwd=root, check=True)
print('\nCompleted saved-metrics analysis; results are in outputs/.')
