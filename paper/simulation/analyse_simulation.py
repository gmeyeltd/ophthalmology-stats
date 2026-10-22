"""
Summarise the Monte Carlo simulation (run after simulate.py has finished).

For every grid cell, analysis strategy and metric it reports:
  scen       H0 (models equally accurate) or H1 (model B less accurate)
  reject     proportion of studies with P < 0.05 (type I error under H0, power under H1)
  mcse       its Monte Carlo standard error
  var_diff   empirical variance of the estimated difference between the models across studies
  rel_var    var_diff relative to the eye-weighted cluster analysis of the same studies
             (for one eye per patient this is the factor by which a one-eye study would need more patients)
  coverage   proportion of cluster 95% CIs containing the true difference (0 under the null; under the
             alternative the Monte Carlo mean of the estimates)
  rho_delta  mean fellow-eye intraclass correlation of the per-eye loss difference (cluster analysis)
and the values predicted from the design effect for unequal cluster sizes (Kerry & Bland 2001;
Eldridge et al. 2006), using the measured rho_delta and the observed proportion of bilateral patients f:
  one eye vs eye-weighted         (1 + f)^2 / (1 + f + 2 f rho)
  patient-averaged vs eye-weighted ((1 - f) + f (1 + rho) / 2) (1 + f)^2 / (1 + f + 2 f rho)
  minimum-variance vs eye-weighted (1 + f)^2 / ((1 + f + 2 f rho) ((1 - f) + 2 f / (1 + rho)))
  naive (eyes as independent) type I error   2 (1 - Phi(1.96 / sqrt(DE))),  DE = 1 + 2 f rho / (1 + f)

Usage (from this folder):  python analyse_simulation.py
Reads ./output_suite/{main,sweep,lottery}/ and writes ./results/summary_main.csv, summary_sweep.csv,
summary_lottery.csv. Takes about a minute.
"""
import glob
import os
import re
from math import erf, sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_suite")     # simulation run with ophthalmology_stats
RES = os.path.join(HERE, "results")
N_PATIENTS_MAIN = 250
ALPHA = 0.05


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def parse_cell(cid):
    d = {"cell": cid}
    d["scen"] = {"null": "H0", "alt": "H1"}[cid.rsplit("_", 1)[1]]     # 'null' would be read back as missing
    d["task"] = "continuous" if cid.startswith("cont") else "binary"
    m = re.search(r"_f([\d.]+)_", cid)
    d["frac"] = float(m.group(1))
    m = re.search(r"psi([\d.]+)", cid)
    d["psi"] = float(m.group(1)) if m else 0.5
    m = re.search(r"_n(\d+)_", cid)
    d["n_patients"] = int(m.group(1)) if m else N_PATIENTS_MAIN
    m = re.search(r"cont_([a-z]+)_psi", cid)
    d["shape"] = m.group(1) if m else ("normal" if d["task"] == "continuous" else "")
    m = re.search(r"bin_p([\d.]+)_", cid)
    d["prevalence"] = float(m.group(1)) if m else (0.3 if d["task"] == "binary" else np.nan)
    return d


def predicted(f, rho):
    de_num = 1 + f + 2 * f * rho
    return {"pred_rel_var_one_eye": (1 + f) ** 2 / de_num,
            "pred_rel_var_patient": ((1 - f) + f * (1 + rho) / 2) * (1 + f) ** 2 / de_num,
            "pred_rel_var_optimal": (1 + f) ** 2 / (de_num * ((1 - f) + 2 * f / (1 + rho))),
            "pred_naive_type1": 2 * (1 - phi(1.959964 / sqrt(1 + 2 * f * rho / (1 + f))))}


def summarise_cell(path):
    d = pd.read_csv(path)
    cid = d["cell"].iloc[0]
    meta = parse_cell(cid)
    f_obs = float((d["n_bilateral"] / meta["n_patients"]).mean())
    cols = [c for c in d.columns if c.count("|") == 2]
    pairs = sorted({tuple(c.split("|")[:2]) for c in cols})
    rows = []
    for strategy, metric in pairs:
        p = d.get(f"{strategy}|{metric}|p")
        diff = d.get(f"{strategy}|{metric}|diff")
        if p is None or p.notna().sum() == 0:
            continue
        ok = p.notna()
        n = int(ok.sum())
        rej = float((p[ok] < ALPHA).mean())
        row = {**meta, "f_obs": f_obs, "strategy": strategy, "metric": metric, "n_rep": n, "reject": rej,
               "mcse": sqrt(rej * (1 - rej) / n), "mean_diff": float(diff.mean()), "var_diff": float(diff.var())}
        rows.append(row)
    out = pd.DataFrame(rows)
    # cluster-only fields, and relative variance against the cluster analysis of the same metric
    for metric in out.metric.unique():
        rd = d.get(f"cluster|{metric}|rho_delta")
        lo, hi, dd = (d.get(f"cluster|{metric}|{k}") for k in ("ci_low", "ci_high", "diff"))
        sel = out.metric == metric
        if rd is not None:
            rho = float(rd.mean())
            out.loc[sel, "rho_delta"] = rho
            out.loc[sel, "rho_delta_sd"] = float(rd.std())
            for k, v in predicted(f_obs, max(rho, 0.0)).items():
                out.loc[sel, k] = v
        if lo is not None and dd is not None:
            truth = 0.0 if meta["scen"] == "H0" else float(dd.mean())
            out.loc[sel & (out.strategy == "cluster"), "coverage"] = float(((lo <= truth) & (truth <= hi)).mean())
        base = out.loc[sel & (out.strategy == "cluster"), "var_diff"]
        if len(base):
            out.loc[sel, "rel_var"] = out.loc[sel, "var_diff"] / float(base.iloc[0])
    return out


def summarise_lottery(path):
    d = pd.read_csv(path)
    task = "continuous" if "continuous" in d["cell"].iloc[0] else "binary"
    return {"task": task, "n_studies": len(d),
            "cluster_top_is_best": d["cluster_top_is_best"].mean(),
            "cluster_power": d["cluster_sig"].mean(),
            "mean_one_eye_frac_top_is_best": d["one_eye_frac_top_is_best"].mean(),
            "studies_where_eye_choice_changes_top_model": (d["one_eye_n_distinct_top"] > 1).mean(),
            "mean_distinct_top_models": d["one_eye_n_distinct_top"].mean(),
            "mean_one_eye_power": d["one_eye_frac_sig"].mean(),
            "studies_where_eye_choice_flips_significance":
                ((d["one_eye_frac_sig"] > 0) & (d["one_eye_frac_sig"] < 1)).mean(),
            "studies_sig_in_10_to_90pct_of_draws":
                ((d["one_eye_frac_sig"] >= 0.1) & (d["one_eye_frac_sig"] <= 0.9)).mean()}


def main():
    os.makedirs(RES, exist_ok=True)
    for part in ("main", "sweep"):
        files = sorted(f for f in glob.glob(os.path.join(OUT, part, "*.csv.gz")) if not f.endswith(".orig.csv.gz"))
        if not files:
            print(f"no files in output/{part}")
            continue
        s = pd.concat([summarise_cell(f) for f in files], ignore_index=True)
        s.to_csv(os.path.join(RES, f"summary_{part}.csv"), index=False)
        print(f"summary_{part}.csv: {len(files)} cells, {len(s)} rows")
    files = sorted(glob.glob(os.path.join(OUT, "lottery", "*.csv.gz")))
    if files:
        pd.DataFrame([summarise_lottery(f) for f in files]).to_csv(os.path.join(RES, "summary_lottery.csv"),
                                                                    index=False)
        print(f"summary_lottery.csv: {len(files)} tasks")


if __name__ == "__main__":
    main()
