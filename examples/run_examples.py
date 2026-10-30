"""Analyze the example data and write the results. Run from this folder: python run_examples.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ophthalmology_stats as osx  # noqa: E402

# continuous output (prediction errors), all eyes and three axial length groups
res = osx.analyze(pd.read_csv("example_continuous.csv"), subgroup="axial_length:cuts=22,24.5",
                  settings={"within": 0.5})
osx.write_results_to_excel(res, "example_continuous_results.xlsx")

# classifier (probabilities and the 0/1 diagnosis), all eyes and right and left eyes
res = osx.analyze(pd.read_csv("example_classifier.csv"), task="binary", subgroup="eye")
osx.write_results_to_excel(res, "example_classifier_results.xlsx")
print("written example_continuous_results.xlsx and example_classifier_results.xlsx")
