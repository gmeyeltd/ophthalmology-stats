# ophthalmology-stats

Compare the accuracy of prediction models evaluated on eyes (artificial intelligence models, formulas, devices)
when some patients contribute both eyes, with the **patient as the unit of analysis**.

Fellow eyes are correlated, so standard tests that count every eye as an independent observation give too many
false-positive differences, and keeping one eye per patient throws away data and makes the result depend on
which eye was kept. ophthalmology-stats keeps every eye and resamples patients, for every common accuracy metric,
and tells you how much the second eyes were worth.

It extends [biometry-stats](https://github.com/gmeyeltd/biometry-stats) (intraocular lens formulas):
`biometry_stats_advanced` is included unchanged, so `analyze_prediction_errors` gives exactly the same results,
and the same resamples, tests, p-value rules, confidence intervals, Romano-Wolf adjustment and Excel output are
used for the new metrics.

## Input

A CSV or pandas DataFrame with **one row per eye**.

**Continuous outputs** (a refraction, axial length, thickness, ...): a patient ID column and **one column of
prediction errors per model** (prediction minus reference). This is the primary input:

| patient_id | eye | axial_length | model_A | model_B | model_C |
|---|---|---|---|---|---|
| Q001 | R | 24.84 | 0.10 | 0.37 | -0.07 |
| Q002 | L | 23.90 | 0.67 | 0.15 | 0.67 |

You can instead give one column of predictions per model plus the reference value, and name it with `truth=` (CLI
`--truth`); the errors are then prediction - truth, with identical results.

**Classifiers**: a patient ID column, the **0/1 diagnosis** (reference standard) and one column of predicted
probabilities per model. The diagnosis is required, because AUC, sensitivity and specificity are defined by it.

| patient_id | eye | truth | model_A | model_B | model_C |
|---|---|---|---|---|---|
| P001 | R | 1 | 0.69 | 0.70 | 0.88 |
| P001 | L | 0 | 0.26 | 0.18 | 0.82 |

* `patient_id`: eyes sharing an ID are kept together (detected automatically from common names).
* With `task="auto"` (the default), a column with a common reference name (`truth`, `reference`, `diagnosis`,
  `measured`, `label`, `y`, ...; see `TRUTH_VARIANTS`) is used as the reference if you do not name one, and the task
  is binary when that column is 0/1 and every model value lies in [0, 1]. With no such column the model columns are
  taken as prediction errors. Name the columns with `models=`, `truth=` and `patient=` (`--models`, `--truth`,
  `--patient`) to avoid guessing.
* Model columns: every numeric column other than the patient, reference and subgroup columns and common metadata
  (`eye`, `age`, `axial_length`, ...; see `GENERIC_IGNORE`). Other columns are ignored unless used as a subgroup.

## Quick start

```bash
pip install -r requirements.txt
# continuous output, prediction errors, results also by axial length
python ophthalmology_stats.py examples/example_continuous_errors.csv --subgroup axial_length --cuts 22 24.5 --out results.xlsx
# continuous output, predictions plus the measured value
python ophthalmology_stats.py examples/example_continuous.csv --truth truth --subgroup eye --out results.xlsx
# classifier
python ophthalmology_stats.py examples/example_classifier.csv --task binary --truth truth --subgroup eye --out results.xlsx
# several subgroups, each with its own split, and axial length crossed with keratometry (Grid sheet for heat maps)
python ophthalmology_stats.py examples/example_continuous_errors.csv --subgroup axial_length:cuts=22,24.5,26 --subgroup mean_k:bins=5 --cross axial_length:cuts=22.5,24.5 mean_k:bins=3 --out results.xlsx
```

```python
import pandas as pd
import ophthalmology_stats as osx

res = osx.analyze(pd.read_csv("examples/example_continuous_errors.csv"), subgroup="axial_length",
                  settings={"subgroup_cuts": [22, 24.5]})
osx.write_results_to_excel(res, "results.xlsx")
# sheets: Results, Statistical Tests, Model Performance, Estimates, Sample (and Grid with crossed subgroups)
```

`examples/run_examples.py` makes the four example result files in `examples/` (with 4,999 resamples).

## Subgroups

`subgroup=` (CLI `--subgroup`) adds results for each subgroup of a column of any type:

* **nominal** (text, e.g. `eye`, `site`, `device`): one subgroup per level, e.g. `eye = L` and `eye = R`, a simple
  check that a model trained on one laterality (for example after flipping left-eye images) performs similarly on
  both;
* **ordinal** (ordered categories, or a numeric column with at most 10 distinct values): one subgroup per level, in
  order;
* **continuous** (e.g. axial length): `subgroup_cuts=[22, 24.5]` (`--cuts 22 24.5`) gives `<22`, `22 to <24.5` and
  `>=24.5` (each interval includes its lower cut point); `subgroup_bins=3` (`--bins 3`) gives equal-count quantile
  groups (tertiles), labeled with the range of values each holds. A numeric column with more than 10 distinct
  values and neither option is split into tertiles, and the output says so.

The subgroup column is never taken as a model: the models are found once, on all eyes, and used for every group.
The Sample sheet gives each group's definition, e.g. `axial_length 22 to <24.5: 238 eyes, 154 patients`. Each
subgroup is analyzed separately, so p-values and intervals are adjusted for the model pairs within the group, not
across groups, and subgroup results are exploratory. A patient's two eyes can fall in different subgroups of a
continuous column; within each subgroup the patient is still the unit. Eyes with no value are in All only.

### Several subgroups

Give a list (CLI: repeat `--subgroup`). Each column is analyzed separately (not crossed) and can have its own split,
written into the definition:

```python
res = osx.analyze(df, subgroup=["axial_length:cuts=22,24.5,26", "mean_k:bins=5", "eye"])
# equivalently: ("mean_k", {"bins": 5}) or {"column": "mean_k", "bins": 5}
```

```bash
python ophthalmology_stats.py errors.csv --subgroup axial_length:cuts=22,24.5,26 --subgroup mean_k:bins=5 --subgroup eye
```

This gives All, four axial length groups, five keratometry quintiles (each using all eyes, whatever their axial
length) and right and left eyes. A `grouping` column in the Estimates, Model Performance and Sample sheets says which
subgroup each row belongs to. `--cuts` and `--bins` (and `subgroup_cuts`, `subgroup_bins`) still work with a single
`--subgroup COLUMN`.

### Crossed subgroups (off by default)

`cross=[rows, columns]` (CLI `--cross ROWS COLUMNS`) also crosses two subgroups, for example axial length with
keratometry, to show where a model does poorly:

```bash
python ophthalmology_stats.py errors.csv --cross axial_length:cuts=22.5,24.5 mean_k:bins=3
```

* Cut points come from all eyes, so a row or column covers the same range in every cell (for example, keratometry
  tertile 1 means the same range of K in short and long eyes).
* Cells with fewer than 20 patients (`cross_min_patients`, CLI `--cross-min-patients`) get point estimates only,
  without intervals or tests; cells with fewer than 5 eyes are not analyzed. Keep crossings coarse (3 x 3 rather than
  5 x 5): axial length and keratometry are correlated, so the corner cells are nearly empty.
* The **Grid** sheet gives each model's performance in each cell: one block per model and metric, with a color scale
  shared by the models of each metric (white best, red worst; for the mean error, red is furthest from 0), ready to
  read as a heat map. Small cells are shown in gray italics and left out of the color scale.

## Metrics and tests

| Output | Metric | Test (patient as the unit) |
|---|---|---|
| Continuous | Mean error, MAE, RMSE | Patient bootstrap-t, cluster-robust (CR1) standard error; RMSE via the mean squared error |
| | Median absolute error | Percentile patient bootstrap (ties counted in both tails) |
| | SD of the error | Morgan-Pitman test, wild cluster bootstrap |
| | Proportion within a tolerance | Clustered McNemar test (Yang, Sun & Hardin 2010); McNemar with continuity correction when no patient has both eyes |
| Classifier | Brier score, log-loss | Patient bootstrap-t, CR1 standard error |
| | AUC | Obuchowski (1997) clustered ROC analysis (DeLong when every patient has one eye) |
| | Sensitivity, specificity, accuracy | Clustered McNemar test (Yang, Sun & Hardin 2010); McNemar with continuity correction when no patient has both eyes |

Each model's own performance on every metric is given with a patient-level 95% confidence interval (percentile
bootstrap resampling patients, so that fellow eyes stay together; not adjusted, as it describes one model). Every
difference between two models comes with a 95% confidence interval, a p-value and an adjusted p-value, and an
adjusted 95% confidence interval where one matches the adjusted p-value (see below). For each metric
the output also reports:

* **ρΔ** (column `rho_delta`): the fellow-eye correlation of the per-eye difference in loss between the two
  models. This, not the correlation of the outcome itself, decides how far eye-level tests are misled and how much
  a second eye adds.
* **one_eye_factor**: how many times more patients a study keeping one eye per patient would need for the same
  precision (design effect for unequal cluster sizes).

## Adjusted p-values and confidence intervals

Adjustment is over the m model pairs of each metric within a group. An adjusted interval is given only where it
matches the adjusted p-value, so the two cannot point different ways (columns ending `CI adj low` / `CI adj high` in
the Results sheet; `ci_adj_low` / `ci_adj_high` in the Estimates sheet). The Estimates sheet states the adjustment
of every row in `p_adj_method` and `ci_adj_method`, and the Statistical Tests sheet summarizes them. With the
default `fwer_method="romano_wolf"`:

| Metric | Adjusted p-value (`p_adj_method`) | Adjusted interval (`ci_adj_method`) |
|---|---|---|
| Mean error, MAE, MedAE, Brier score, log-loss | Romano-Wolf step-down | Romano-Wolf simultaneous (max-\|t\|): difference ± c × SE, c the 95% quantile of the largest bootstrap \|t\| across the pairs |
| RMSE, SD | Romano-Wolf step-down (on the MSE; Morgan-Pitman) | none: the p-value comes from a different statistic, so the CI is unadjusted |
| Proportion within a tolerance, sensitivity, specificity, accuracy, AUC | Hommel | none: Hommel has no matching interval, so the CI is unadjusted |

With `fwer_method="bonferroni"` every metric gets Bonferroni p-values and matching Bonferroni intervals (AUC:
difference ± z(1 - 0.05/2m) × SE; RMSE, SD and proportions: percentile interval of the same patient resamples at
level 1 - 0.05/m; bootstrap-t metrics: c the 1 - 0.05/m quantile of each pair's bootstrap \|t\|). `"holm"` and
`"hommel"` adjust the p-values only, as they have no matching intervals. `"fdr_bh"` gives Benjamini-Yekutieli
false-coverage-rate intervals (level 1 - Rα/m, R the number of significant pairs). `ci_adj_method` says why a row
has no adjusted interval. With a single model pair no adjustment is needed: the adjusted p-value and interval equal
the unadjusted ones, and the output says so. Bonferroni-type intervals use the far tail of the bootstrap
distribution, so use more resamples (e.g. 9,999) with them, especially for many models.

## Options (`settings=`)

| Setting | Default | Meaning |
|---|---|---|
| `bootstrap_iterations` | 4999 | resamples; p-values fall on a grid of 1/5000 |
| `seed` | 42 | fixed for reproducible results |
| `fwer_method` | `romano_wolf` | or `bonferroni`, `holm`, `hommel`, `fdr_bh`, `none` |
| `weighting` | `eye` | mean-loss metrics: `eye` (every eye once), `patient` (every patient once), `optimal` (minimum variance) |
| `threshold` | 0.5 | classifiers: probability threshold for sensitivity, specificity and accuracy |
| `within` | 0.5 | continuous: tolerance for the proportion of eyes with abs(error) <= within |
| `subgroup_cuts` | None | continuous subgroup: cut points, e.g. `[22, 24.5]` (CLI `--cuts 22 24.5`) |
| `subgroup_bins` | None | continuous subgroup: number of equal-count quantile groups, e.g. `3` (CLI `--bins 3`) |
| `one_eye` | False | keep one randomly chosen eye per patient |
| `unit` | `patient` | `eye` treats every eye as independent (for comparison only; not valid with both eyes) |
| `metrics` | all | a subset of the metrics for the task |

## Validation

`tests/test_ophthalmology_stats.py` (run `python tests/test_ophthalmology_stats.py`) checks, among other things,
that continuous-output results are identical to biometry_stats_advanced, that prediction errors give the same
results as predictions plus a truth column, that the clustered AUC equals DeLong's when every patient has one eye,
and that the Bonferroni intervals equal the unadjusted intervals at level 1 - 0.05/m. The Monte Carlo
simulation that validates the type I error, power and precision of every test is in `paper/simulation`, with its
random seeds. `paper/realdata/papila_example.py` repeats the comparison on real data: two image models and the
cup-to-disc ratio on both eyes of the 244 patients in PAPILA (Kovalyk et al, Sci Data 2022;9:291), and
`papila_figure.py` draws its figure. The images are
not included; the script reads PAPILA.zip as downloaded from figshare (doi:10.6084/m9.figshare.14798004).

## Citation

Moussa G, et al. Keeping both eyes when comparing ophthalmic artificial intelligence models: open-source code and simulation study. (in preparation)

## License

MIT. See LICENSE.
