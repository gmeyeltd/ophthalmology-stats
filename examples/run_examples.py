"""Analyse the two example datasets and write Excel results. Run from this folder: python run_examples.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ophthalmology_stats as osx  # noqa: E402

# classifier: probabilities and a 0/1 reference standard; results for all eyes and for right and left eyes
res = osx.analyze(pd.read_csv("example_classifier.csv"), task="binary", subgroup="eye")
osx.write_results_to_excel(res, "example_classifier_results.xlsx")

# continuous output: predictions and the measured value (errors are prediction - truth)
res = osx.analyze(pd.read_csv("example_continuous.csv"), task="continuous", truth="truth", subgroup="eye",
                  settings={"within": 0.5})
osx.write_results_to_excel(res, "example_continuous_results.xlsx")
print("written example_classifier_results.xlsx and example_continuous_results.xlsx")
