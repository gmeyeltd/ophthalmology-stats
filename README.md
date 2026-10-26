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
bootstrap resampling patients, so that fellow eyes stay together). Every difference between two models comes with
a 95% confidence interval, a p-value and an adjusted p-value (Romano-Wolf step-down for the bootstrap tests,
Hommel otherwise). The unadjusted interval can exclude zero next to a non-significant adjusted p-value, so MPE, MAE,
MedAE, Brier score and log-loss differences also get an adjusted 95% interval, in the columns ending `CI adj low` /
`CI adj high` (`ci_adj_low` / `ci_adj_high` in the Estimates sheet): the difference ± c × its standard error, with c
matching the `fwer_method` chosen for the adjusted p-values. Romano-Wolf (default) gives simultaneous max-|t|
intervals (c is the quantile of the largest bootstrap |t| across the comparisons in the group). Bonferroni gives
Bonferroni intervals. Holm and Hommel also get Bonferroni intervals, because these stepwise methods have no
compatible finite intervals, so an interval can include 0 next to a significant adjusted p-value. Benjamini-Hochberg
gives Benjamini-Yekutieli false-coverage-rate intervals. Other methods, RMSE (tested on the MSE scale), SD, AUC and
proportions have no adjusted interval; with a single comparison the adjusted interval equals the unadjusted one.
Bonferroni-type intervals use the far tail of the bootstrap distribution, so use more resamples (e.g. 9,999) with them. For each metric the
output also reports:

* **ρΔ** (column `rho_delta`): the fellow-eye correlation of the per-eye difference in loss between the two
  models. This, not the correlation of the outcome itself, decides how far eye-level tests are misled and how much
  a second eye adds.
* **one_eye_factor**: how many times more patients a study keeping one eye per patient would need for the same
  precision (design effect for unequal cluster sizes).

## Input

A CSV or pandas DataFrame with **one row per eye**:

| patient_id | eye | truth | model_A | model_B | ... |
|---|---|---|---|---|---|
| P001 | R | 1 | 0.69 | 0.70 | |
| P001 | L | 0 | 0.26 | 0.18 | |
| P002 | R | 0 | 0.01 | 0.00 | |

* `patient_id`: eyes sharing an ID are kept together (detected automatically from common names).
* `truth`: the 0/1 reference standard (classifiers) or the measured value (continuous outputs). For continuous
  outputs the model columns may instead hold the prediction errors, with no truth column.
* One column per model: a predicted probability (classifiers) or a prediction (continuous outputs).
* Other columns (e.g. `eye`, `age`) are ignored unless used as a subgroup.

## Quick start

```bash
pip install -r requirements.txt
python ophthalmology_stats.py examples/example_classifier.csv --task binary --subgroup eye --out results.xlsx
```

```python
import pandas as pd
import ophthalmology_stats as osx

res = osx.analyze(pd.read_csv("examples/example_classifier.csv"), task="binary", subgroup="eye")
osx.write_results_to_excel(res, "results.xlsx")
# sheets: Results, Statistical Tests, Model Performance, Estimates, Sample
```

`subgroup="eye"` adds results for right and left eyes separately, a simple check that a model trained on one
laterality (for example after flipping left-eye images) performs similarly on both.

## Options (`settings=`)

| Setting | Default | Meaning |
|---|---|---|
| `bootstrap_iterations` | 4999 | resamples; p-values fall on a grid of 1/5000 |
| `seed` | 42 | fixed for reproducible results |
| `fwer_method` | `romano_wolf` | or `holm`, `hommel`, `fdr_bh`, `none` |
| `weighting` | `eye` | mean-loss metrics: `eye` (every eye once), `patient` (every patient once), `optimal` (minimum variance) |
| `threshold` | 0.5 | classifiers: probability threshold for sensitivity, specificity and accuracy |
| `within` | 0.5 | continuous: tolerance for the proportion of eyes with abs(error) <= within |
| `one_eye` | False | keep one randomly chosen eye per patient |
| `unit` | `patient` | `eye` treats every eye as independent (for comparison only; not valid with both eyes) |
| `metrics` | all | a subset of the metrics for the task |

## Validation

`tests/test_ophthalmology_stats.py` checks, among other things, that continuous-output results are identical to
biometry_stats_advanced and that the clustered AUC equals DeLong's when every patient has one eye. The Monte Carlo
simulation that validates the type I error, power and precision of every test is in `paper/simulation`, with its
random seeds. `paper/realdata/papila_example.py` repeats the comparison on real data: two image models and the
cup-to-disc ratio on both eyes of the 244 patients in PAPILA (Kovalyk et al, Sci Data 2022;9:291). The images are
not included; the script reads PAPILA.zip as downloaded from figshare (doi:10.6084/m9.figshare.14798004).

## Citation

Moussa G, et al. Keeping both eyes when comparing ophthalmic artificial intelligence models: open-source code and simulation study. (in preparation)

## Licence

MIT. See LICENSE.
