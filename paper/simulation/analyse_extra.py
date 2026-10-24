"""
Summarise the additional simulations (run after simulate_extra.py has finished, and after analyse_simulation.py).

Writes to ./results/
  summary_extra_gee_main.csv   GEE type I error for every null setting of the main grid, beside the type I error of
                               the code's patient-level test and the measured rho_delta of the same studies
  summary_extra_gee_sweep.csv  GEE type I error and power across study size, beside the code's
  summary_extra_small.csv      20 and 30 patients: type I error of the code's tests, GEE and GEE with the
                               Mancl-DeRouen correction
  summary_extra_sens.csv       within-eye correlation 0.5 and 0.95: type I error, relative variance of one eye per
                               patient and the design-effect predictions, as summary_main.csv
and prints a short overview. Usage (from this folder):  python analyse_extra.py   (--pilot reads the pilot run)
"""
import glob
import os
import sys

import pandas as pd

from analyse_simulation import RES, summarise_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_suite_pilot" if "--pilot" in sys.argv else "output_suite")


def summarise(part):
    files = sorted(glob.glob(os.path.join(OUT, part, "*.csv.gz")))
    if not files:
        print(f"no files in output_suite/{part}")
        return None
    s = pd.concat([summarise_cell(f) for f in files], ignore_index=True)
    m = s.cell.str.extract(r"_w([\d.]+)_")[0]
    if m.notna().any():
        s.insert(s.columns.get_loc("psi") + 1, "within_eye_corr", m.astype(float))
    return s


def beside_main(gee, part):
    """Add the code's (cluster) type I error / power and rho_delta for the same cells and metrics."""
    path = os.path.join(RES, f"summary_{part}.csv")
    if not os.path.exists(path):
        return gee
    ref = pd.read_csv(path)
    ref = ref[ref.strategy == "cluster"][["cell", "metric", "reject", "rho_delta"]]
    ref = ref.rename(columns={"reject": "reject_code"})
    gee = gee.drop(columns=[c for c in ("rho_delta", "rho_delta_sd") if c in gee.columns])
    return gee.merge(ref, on=["cell", "metric"], how="left")


def main():
    os.makedirs(RES, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    g = summarise("gee_main")
    if g is not None:
        g = beside_main(g, "main")
        g.to_csv(os.path.join(RES, "summary_extra_gee_main.csv"), index=False)
        print("\nGEE, 250 patients, null: type I error by metric (range across settings)")
        print(g.groupby("metric").agg(gee_min=("reject", "min"), gee_max=("reject", "max"),
                                      code_min=("reject_code", "min"), code_max=("reject_code", "max"),
                                      min_fits=("n_rep", "min")).round(3).to_string())

    g = summarise("gee_sweep")
    if g is not None:
        g = beside_main(g, "sweep")
        g.to_csv(os.path.join(RES, "summary_extra_gee_sweep.csv"), index=False)
        print("\nGEE across study size (psi 0.5): type I error (H0) and power (H1), beside the code's")
        print(g.pivot_table(index=["task", "metric", "frac", "scen"], columns="n_patients",
                            values="reject").round(3).to_string())

    s = summarise("small")
    if s is not None:
        s.to_csv(os.path.join(RES, "summary_extra_small.csv"), index=False)
        print("\n20 and 30 patients, null: type I error (cluster = the code; gee_br = Mancl-DeRouen)")
        print(s.pivot_table(index=["task", "metric"], columns=["n_patients", "frac", "strategy"],
                            values="reject").round(3).to_string())

    s = summarise("sens")
    if s is not None:
        s.to_csv(os.path.join(RES, "summary_extra_sens.csv"), index=False)
        h0 = s[s.scen == "H0"]
        print("\nWithin-eye correlation sensitivity (f 0.8): rho_delta, naive type I error observed vs predicted,")
        print("one eye per patient relative variance observed vs predicted (MAE)")
        mae = h0[h0.metric == "mae"]
        tab = mae.pivot_table(index=["within_eye_corr", "psi"], columns="strategy",
                              values=["reject", "rel_var"]).round(3)
        tab["rho_delta"] = mae[mae.strategy == "cluster"].set_index(["within_eye_corr", "psi"]).rho_delta.round(3)
        tab["pred_naive"] = mae[mae.strategy == "cluster"].set_index(["within_eye_corr", "psi"]).pred_naive_type1.round(3)
        tab["pred_one_eye"] = mae[mae.strategy == "cluster"].set_index(["within_eye_corr", "psi"]).pred_rel_var_one_eye.round(3)
        print(tab.to_string())


if __name__ == "__main__":
    main()
