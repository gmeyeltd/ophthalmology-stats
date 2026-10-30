# ophthalmology-stats

Compare prediction models evaluated on eyes (AI models, formulas, devices) when some patients contribute both
eyes, using every eye with the **patient as the unit of analysis**.

Tests that count every eye as independent declare too many differences significant, and keeping one eye per
patient wastes patients and makes the result depend on which eye was kept. ophthalmology-stats resamples
patients, keeping fellow eyes together, for 12 accuracy metrics. It extends
[biometry-stats](https://github.com/gmeyeltd/biometry-stats) (intraocular lens formulas) and gives identical
results for the metrics they share.

## Install

```bash
pip install -r requirements.txt
```

## Input

A CSV file or pandas DataFrame with **one row per eye**:

* a patient ID column (found automatically from names such as `patient_id`, `id` or `subject`, or set with
  `--patient`);
* **continuous outputs**: one column of prediction errors (prediction minus reference) per model, or the
  predictions plus the reference value named with `--truth`;
* **classifiers**: one column of predicted probabilities per model and the 0/1 diagnosis (`--truth`).

| patient_id | eye | axial_length | model_A | model_B | model_C |
|---|---|---|---|---|---|
| Q001 | R | 24.84 | 0.10 | 0.37 | -0.07 |
| Q002 | L | 23.90 | 0.67 | 0.15 | 0.67 |

Other columns (`eye`, `age`, `axial_length`, ...) are ignored unless used as a subgroup. Name the columns with
`--models`, `--truth` and `--patient` to avoid automatic detection.

## Use

```bash
python ophthalmology_stats.py examples/example_continuous.csv --subgroup eye --out results.xlsx
python ophthalmology_stats.py examples/example_classifier.csv --truth truth --subgroup eye --out results.xlsx
```

```python
import pandas as pd
import ophthalmology_stats as osx

res = osx.analyze(pd.read_csv("examples/example_continuous.csv"), subgroup="eye")
osx.write_results_to_excel(res, "results.xlsx")
```

The workbook has five sheets: Results (one row per model pair), Statistical Tests, Model Performance, Estimates
(one row per pair and metric) and Sample. `examples/` holds the example data and their results.

## Metrics and tests

| Output | Metric | Test with the patient as the unit |
|---|---|---|
| Continuous | Mean error, MAE, RMSE | Patient bootstrap-t, cluster-robust (CR1) standard error; RMSE tested on the MSE |
| | Median absolute error | Percentile patient bootstrap |
| | SD of the error | Morgan-Pitman test, wild cluster bootstrap |
| | Proportion within a tolerance | Clustered McNemar test (Yang, Sun and Hardin 2010) |
| Classifier | Brier score, log-loss | Patient bootstrap-t, CR1 standard error |
| | AUC | Obuchowski (1997); DeLong when every patient has one eye |
| | Sensitivity, specificity, accuracy | Clustered McNemar test |

Each model's performance is given with a patient-level 95% confidence interval (CI), and each difference between
two models with a 95% CI, a p-value and an adjusted p-value. The Estimates sheet also gives `rho_delta`, the
fellow-eye correlation of the per-eye difference between the two models, and `one_eye_factor`, how many times
more patients a study keeping one eye per patient would need for the same precision.

## Multiple comparisons

P-values are adjusted for the model pairs of each metric: Romano-Wolf step-down for the bootstrap tests and
Hommel for the others. An adjusted CI is given only where it matches the adjusted p-value: Romano-Wolf
simultaneous intervals for the mean error, MAE, median absolute error, Brier score and log-loss. With
`fwer_method="bonferroni"`, every metric gets Bonferroni p-values and matching intervals. The columns
`p_adj_method` and `ci_adj_method` name the adjustment of each row.

## Subgroups

`--subgroup COLUMN` adds results for each level of a column (e.g. `eye`, `site`). A numeric column can be split
at cut points or into equal-count groups: `axial_length:cuts=22,24.5` or `mean_k:bins=5`. Repeat `--subgroup`
for several columns, each analyzed separately. `--cross ROWS COLUMNS` crosses two subgroups (off by default) and
adds a Grid sheet for heat maps; cells with fewer than 20 patients get point estimates only. Each group is
adjusted for its own model pairs, so subgroup results are exploratory.

## Settings

Passed as `settings={...}` to `analyze`.

| Setting | Default | Meaning |
|---|---|---|
| `bootstrap_iterations` | 4999 | patient resamples |
| `seed` | 42 | random seed |
| `fwer_method` | `romano_wolf` | or `bonferroni`, `holm`, `hommel`, `fdr_bh`, `none` |
| `weighting` | `eye` | mean-loss metrics: `eye`, `patient` (each patient once) or `optimal` (minimum variance) |
| `threshold` | 0.5 | classifiers: probability threshold |
| `within` | 0.5 | continuous: tolerance for the proportion of eyes with abs(error) <= within |
| `one_eye` | False | keep one random eye per patient |
| `unit` | `patient` | `eye` treats eyes as independent (for comparison only) |
| `metrics` | all | a subset of the metrics |

## Validation

`python tests/test_ophthalmology_stats.py` runs 66 checks. These include that the continuous-output results are
identical to version 1.2 and biometry-stats, that the clustered AUC test equals DeLong's when every patient has
one eye, and that the Bonferroni intervals equal the unadjusted intervals at level 1 - 0.05/m.

`paper/` holds the code behind the accompanying paper, with its random seeds: the simulation, the literature
survey, and the worked example on PAPILA (Kovalyk et al, Sci Data 2022;9:291). PAPILA is not included; download it
from figshare (doi:10.6084/m9.figshare.14798004).

## Citation

Moussa G, et al. Keeping both eyes when comparing ophthalmic artificial intelligence models: survey, simulation,
and open-source code. (in preparation)

## License

MIT. See LICENSE.
