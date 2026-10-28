"""Copy the simulation, audit and real-data code used in the paper into this folder before publishing the repository.

Run from AI_Paper/github/ophthalmology-stats/paper:  python copy_paper_code.py
"""
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
AI = os.path.join(HERE, "..", "..", "..")
for sub, patterns in (("simulation", ["simulate.py", "dgm.py", "analyse_simulation.py", "make_figures.py"]),
                      ("audit", ["audit_01_fetch.py", "audit_02_extract.py", "audit_03_validation.py", "survey_summary.py",
                                 "CODEBOOK.md", "development_set.csv"]),
                     ("realdata", ["papila_example.py", "papila_figure.py"])):
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    for p in patterns:
        for f in glob.glob(os.path.join(AI, sub, p)):
            shutil.copy2(f, os.path.join(HERE, sub, os.path.basename(f)))
            print("copied", sub + "/" + os.path.basename(f))
