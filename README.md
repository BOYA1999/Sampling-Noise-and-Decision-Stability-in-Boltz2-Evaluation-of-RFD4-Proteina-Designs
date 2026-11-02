# Sampling Noise and Decision Stability in Boltz2 Evaluation of RFD4-Proteina Designs

This repository provides scripts and processed numerical metrics for **saved-metrics analysis replay** of the associated manuscript. It contains the metrics from which the reported comparisons, selected uncertainty estimates, ranking sensitivities, and plots can be recomputed. It does not rerun protein generation, structure prediction, alignment, or experimental validation.

## Quick start

The analysis was checked locally using Python 3.12, NumPy 1.26.4, and Matplotlib 3.10.8. From the repository root:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install -r requirements.txt
python run_all.py
python verify_results.py
```

Scripts read only the files in `data/` and create the `outputs/` directory when it does not exist. Results and plots are written to `outputs/` and never overwrite the inputs. Run the commands from the repository root. The plot styling can differ by operating system, installed fonts, or library version.

## Inputs and analytical scope

| Input | Contents | Rows |
|---|---|---:|
| `data/e1_metrics.csv` | Repeated evaluation metrics for 29 designs | 196 |
| `data/e3b_metrics.csv` | Five evaluations for 50 further candidates and their stored AF2 reference metrics | 250 |
| `data/e4_metrics.csv` | Zero per-step-noise evaluations for 29 designs and four seeds | 116 |
| `data/e5_metrics.csv` | Three intermediate noise scales for 29 designs and three seeds | 261 |
| `data/replay_point_estimates.json` | Frozen replay point estimates used to check the reimplementation | 8 entries |
| `data/figure2_reference.json` | Frozen seed-ranking values used to check the alternate Figure 2 layout | 2 lengths |

The recorded AF2 outputs are a computational reference, not experimentally measured structure or biological success. The 50-candidate sample was assembled in two stages, with 10 followed by 40 candidates, as described in the manuscript. The analysis does not establish a prospective clinical or experimental validation. Reusing these saved metrics is not an independent verification of the upstream model calls or of the source structures.

## What the scripts produce

- `e1_summary.py` and `rev_defs.py`: within-design summaries, intraclass correlations, qualification changes and leave-one-out disagreement counts.
- `e2_risk.py`: the original ridge logistic risk calculation and its conditional bootstrap percentile interval. `recalculate.py` reports how many resamples yielded an eligible boundary. The interval is conditional on eligibility, not an unconditional 95% confidence interval.
- `rev2_policies.py`: AF2-reference agreement, confusion matrices and paired-bootstrap calculations for the 50-candidate set. Its two AF2 reference columns produce identical labels on this dataset, but their confidence intervals are calculated from separate random resamples and may differ slightly. Manuscript Table 3B uses the `af2_rmsd_only` column throughout; `d90_maj2_keep` matches `single` on all 50 candidates and its paired difference interval is [0, 0].
- `rev2_ranking.py`, `check_seed_rank.py` and `rev2_sweep.py`: ranking sensitivity, eligible-seed comparison and replay threshold sweep. The 100-residue group has one best-of-N design and does not support a method-ranking claim.
- `e4_analyze.py` and `e5_analyze.py`: four-seed intervention summaries and three-seed noise-scale summaries. `e4_analyze.py` preserves the historical log-mean bootstrap for audit and labels it as the geometric-mean ratio interval; it is **not** the median ratio interval.
- `recalculate.py`: corrected Table 3A paired intervals, resampling designs after averaging their repeated splits; the median SD-ratio interval; and the count of eligible risk-curve bootstrap samples. The original replay point estimates are checked against the stored reference. It does not use the superseded bootstrap that treated 11,600 design–split pairs as independent.
- `make_figures.py`, `make_figure3.py`, `make_figure6.py`, and `make_figure2_for_word.py`: recompute the plotted quantities and regenerate Figures 1–6, including the alternate layout of Figure 2, from the saved metrics and analysis outputs.

The three-seed noise-scale summaries are descriptive; seed repetition does not create additional independent protein designs. Confidence intervals are conditional on the data and methods provided. The threshold sweep uses random splits distinct from the Table 3A replay.

## Historical model wrappers

`historical/run_boltz_nonoise.py` and `historical/run_boltz_scale.py` are retained as research-code records of how the per-step noise parameter was changed. They require a compatible external Boltz2 installation, model weights, input structures or sequences, PyTorch, CUDA, and a supported GPU. These prerequisites are not provided here, and the wrappers were **not rerun or validated end to end** for this release. Exact model checkpoints, source CIFs, per-run predictions and environments are not included. Do not interpret analysis replay as recreation of those model outputs.

## Provenance and limitations

The numerical CSV inputs are copied from the study's stored per-evaluation metric tables without changing their content. Analysis scripts are minimally adapted from the study scripts to use repository-relative input paths and write into `outputs/`; the source calculation and random seeds are retained unless stated. File hashes are recorded in `data/SHA256SUMS`. `verify_results.py` checks input integrity and selected reported numerical values against the frozen results.

The original CIF/structure files, protein sequences, full model weights, candidate inventory, raw predictor outputs, generator search logs, AF2 structures, and underlying upstream data licences are not bundled. Consequently, this repository cannot independently reproduce the generation and refolding pipeline, validate training-data provenance, re-extract RMSD from structures, or verify claims requiring those omitted artifacts. Frozen Python and model environments for the original GPU runs were not reconstructed. Please consult upstream RFD4-Proteina and Boltz2 sources for their terms and dependencies; the existence of this repository does not grant rights to redistribute upstream weights or data.

The repository's existing `LICENSE` governs the repository as stated there. No additional licence is asserted here for upstream material.
