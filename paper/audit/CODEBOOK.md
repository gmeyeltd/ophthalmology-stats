# Audit codebook (manual coding)

The manual coding uses the same definitions as the rules in `audit_02_extract.py`. Code from the
full text (Methods, Results, table and figure legends; Discussion only for item 6). Code what the
authors **report**, not what you suspect. If something is not stated, code it as not reported.

## 1. eligible (yes / no)
**Yes** if all of the following hold:
- primary research (not a review, editorial, protocol or commentary);
- it develops or evaluates an AI or machine-learning model;
- the model's input includes data from human eyes (images or measurements);
- it reports a quantitative performance measure (AUC, sensitivity, accuracy, MAE, Dice, kappa, …).

**No** for:
- animal studies;
- eye-tracking or gaze studies;
- non-ocular inputs;
- studies with no performance measure.

## 2. unit: the eyes in the evaluation (test or validation) data
If not stated for the test set, code the whole dataset.
- **both_eyes**: some or all patients contribute both eyes. This includes "X eyes of Y patients" with X > Y, or a stated inclusion of both eyes.
- **one_eye**: exactly one eye per patient (stated, or X eyes of X patients).
- **multiple_images_unclear**: several images per eye or patient, and whether both eyes are included cannot be determined.
- **not_reported**: no information on eyes per patient.
- **patient_level_model**: the model takes both eyes and gives one output per patient, so the clustering problem does not arise.

## 3. split: how the data were divided into training and test sets
- **patient_level**: patients (all their eyes and images) kept in one partition.
- **image_or_eye_level**: images or eyes assigned to partitions independently of patient, or stated to be split at image or eye level.
- **external_only**: model evaluated only on a separate external dataset (no internal split).
- **not_reported**.

## 4. uncertainty: how CIs or variances of performance were obtained
- **clustered**: a method that treats the patient as the unit, such as:
  - a patient or cluster bootstrap;
  - Obuchowski's method;
  - GEE or mixed models;
  - cluster-robust variance;
  - clustered Wilcoxon tests.
- **independent_data**: CIs or tests that assume independent eyes or images, such as:
  - DeLong;
  - Clopper–Pearson or Wilson;
  - an eye- or image-level bootstrap;
  - an unspecified bootstrap or CI method.
- **none_reported**: no CIs, standard errors or variance for performance.

## 5. comparison: comparison of two or more models (or a model and clinicians)
- **statistical_test**: a named test or a p-value for the difference (DeLong, McNemar, t test, …).
- **descriptive_only**: models compared by point estimates only.
- **none**: a single model with no comparison.

## 6. correlation_ack (yes / no)
**Yes** if the paper mentions the correlation or non-independence of fellow eyes, or of several images from one patient, or says the analysis accounted for clustering. This can be anywhere, including the Discussion.

## Decision rules for common borderline cases
Agreed during development (22 Oct 2026), before validation coding. The coder applies them.
- **Eligibility**
  - Not eligible: AI used only as a measuring or alignment tool in an association study; non-ocular inputs (blood, questionnaires, brain MRI or fMRI, dental images); large language models answering questions; psychophysics.
  - Eligible: tear fluid, orbital CT, eyelid photographs, surgical video and ocular examination findings all count as ocular inputs.
- **Unit**
  - Several images or videos per patient, with eyes not stated: multiple_images_unclear.
  - One output per patient from both eyes: patient_level_model.
- **Split**
  - Partition counts by patient that add up to the total imply a patient-disjoint split: patient_level.
  - With one eye or image per patient, any split by eye or image is patient_level.
  - "Images were randomly split": image_or_eye_level. A random division reported only in image counts, with the unit not stated: not_reported.
  - Eyes allocated to partitions with patients not kept together: image_or_eye_level.
  - A model trained elsewhere and evaluated only on external data: external_only.
- **Uncertainty**
  - A standard deviation across cross-validation folds only: none_reported.
  - A CI whose method is not stated, or a bootstrap whose resampling unit is not stated: independent_data.
- **Comparison**
  - P values for a correlation (not for a difference between models) do not count as a test.
  - "No significant difference" with no named test or P value: descriptive_only.
- **Correlation acknowledgement**
  - Counts: any statement that fellow eyes or images of one patient are correlated or not independent. This includes "bilaterality bias" and a sensitivity analysis restricted to one eye.

### Clarifications added during validation coding (before any scoring)
Recorded when the case first arose; earlier rows affected were recoded for consistency.
- **Uncertainty**: a test of a performance difference that treats eyes or images as independent (DeLong, McNemar, Wilcoxon, t test, chi-square, Friedman) with no CI: independent_data (item 4 lists such tests). Added at article 113; 6 earlier rows were recoded.
- **Eligibility**: studies of humans using AI (reader or education studies) whose outcome is human accuracy are not eligible.
- **Correlation acknowledgement**: mixed models or GEE used only for clinical repeated measures, without any statement about correlated eyes or images, do not count; statements about leakage or the data split alone do not count.

## Primary outcome
Among eligible studies with unit = both_eyes: the proportion with uncertainty = clustered.
