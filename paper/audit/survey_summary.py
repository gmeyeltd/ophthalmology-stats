"""Survey results from the simple random sample of coded articles (the paper's primary estimates), and a count of
the patient-level analyses found in the flagged (enriched) articles.

Used by manuscript/paper_numbers.py and simulation/make_figures.py so that the text and Figure 2 agree.
Codes come from SURVEY_CODES if set (a .tsv/.csv, or the completed coding workbook .xlsx), otherwise from the
reference coding audit_data/validation_codes_reference.tsv.
"""
import importlib.util
import os

import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "audit_data")
DEFAULT_CODES = os.path.join(DATA, "validation_codes_reference.tsv")


def _read_codes(path):
    spec = importlib.util.spec_from_file_location("audit_03_validation", os.path.join(HERE, "audit_03_validation.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.read_codes(path, "")


def exact_ci(k, n):
    """Clopper-Pearson 95% interval for k of n."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def summary(codes_path=None):
    path = codes_path or os.environ.get("SURVEY_CODES") or DEFAULT_CODES
    codes = _read_codes(path) if path.endswith(".xlsx") else pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    codes.columns = [c.replace("m1_", "") for c in codes.columns]
    val = pd.read_csv(os.path.join(DATA, "validation_sample.csv"))
    val = val[val.include]
    rs = codes[codes.pmcid.isin(val[val["sample"] == "random"].pmcid)]
    enr = codes[codes.pmcid.isin(val[val["sample"] != "random"].pmcid)]
    el = rs[rs.eligible == "yes"]
    both = el[el.unit == "both_eyes"]
    S = {"codes_file": os.path.basename(path), "n": len(rs), "eligible": len(el), "both": len(both),
         "enriched_n": len(enr)}
    groups = {   # eyes per patient among eligible studies
        "unit_both": el.unit == "both_eyes",
        "unit_one": el.unit.isin(["one_eye", "patient_level_model"]),
        "unit_unclear": el.unit.isin(["multiple_images_unclear", "not_reported"]),
    }
    for k, m in groups.items():
        S[k] = int(m.sum())
    items = {    # among eligible studies with both eyes of some patients
        "none": both.uncertainty == "none_reported",
        "indep": both.uncertainty == "independent_data",
        "clustered": both.uncertainty == "clustered",
        "split_patient": both.split == "patient_level",
        "split_image": both.split == "image_or_eye_level",
        "split_external": both.split == "external_only",
        "split_nr": ~both.split.isin(["patient_level", "image_or_eye_level", "external_only"]),
        "ack": both.correlation_ack == "yes",
        "test": both.comparison == "statistical_test",
    }
    for k, m in items.items():
        S[k] = int(m.sum())
    S["test_clustered"] = int(((both.comparison == "statistical_test") & (both.uncertainty == "clustered")).sum())
    for k, n in [("eligible", S["n"]), ("unit_both", S["eligible"]), ("unit_one", S["eligible"]),
                 ("unit_unclear", S["eligible"])] + [(k, S["both"]) for k in items]:
        k_ = S[k]
        S[f"{k}_p"] = k_ / n if n else float("nan")
        S[f"{k}_lo"], S[f"{k}_hi"] = exact_ci(k_, n)
        S[f"{k}_den"] = n
    eb = enr[(enr.eligible == "yes") & (enr.unit == "both_eyes")]
    S["enriched_both"] = len(eb)
    S["enriched_clustered"] = int((eb.uncertainty == "clustered").sum())
    return S


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")
