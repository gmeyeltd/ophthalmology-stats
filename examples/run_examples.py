"""Analyze the example datasets and write Excel results. Run from this folder: python run_examples.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ophthalmology_stats as osx  # noqa: E402

# continuous output, primary input: one column of prediction errors (prediction minus measured value) per model;
# results for all eyes and for three axial length groups (a continuous subgroup split at cut points)
res = osx.analyze(pd.read_csv("example_continuous_errors.csv"), subgroup="axial_length",
                  settings={"within": 0.5, "subgroup_cuts": [22, 24.5]})
osx.write_results_to_excel(res, "example_continuous_errors_results.xlsx")

# the same data with several subgroups, each with its own split (axial length at cut points, keratometry in
# quintiles, right and left eyes), and axial length crossed with keratometry (3 x 3 cells; Grid sheet for heat maps)
res = osx.analyze(pd.read_csv("example_continuous_errors.csv"),
                  subgroup=["axial_length:cuts=22,24.5,26", "mean_k:bins=5", "eye"],
                  cross=["axial_length:cuts=22.5,24.5", "mean_k:bins=3"], settings={"within": 0.5})
osx.write_results_to_excel(res, "example_continuous_errors_subgroups_results.xlsx")

# continuous output given as predictions and the measured value (errors are prediction - truth);
# results for all eyes and for right and left eyes
res = osx.analyze(pd.read_csv("example_continuous.csv"), task="continuous", truth="truth", subgroup="eye",
                  settings={"within": 0.5})
osx.write_results_to_excel(res, "example_continuous_results.xlsx")

# classifier: probabilities and the 0/1 diagnosis; results for all eyes and for right and left eyes
res = osx.analyze(pd.read_csv("example_classifier.csv"), task="binary", subgroup="eye")
osx.write_results_to_excel(res, "example_classifier_results.xlsx")
print("written example_continuous_errors_results.xlsx, example_continuous_errors_subgroups_results.xlsx, "
      "example_continuous_results.xlsx and example_classifier_results.xlsx")
