"""
Additional simulations (added after review of the first draft). Uses the same data-generating mechanisms,
seeds and study definitions as simulate.py, so the GEE analyses below are run on exactly the same simulated
studies as the main results.

Parts
  gee_main   generalised estimating equations (GEE) on both eyes, every null setting of the main grid
             (type I error across fellow-eye correlation, error shape, prevalence and proportion bilateral)
  gee_sweep  GEE on the study-size sweep (50 to 1000 patients; null and alternative: type I error and power)
  small      very small studies (20 and 30 patients; null): GEE (usual sandwich and Mancl-DeRouen corrected
             standard errors) and the patient-level bootstrap tests of ophthalmology-stats, to see which keeps the
             type I error when patients are few
  sens       sensitivity analysis of the within-eye correlation of the two models' errors (0.5 and 0.95 instead
             of 0.8): continuous outputs, normal errors, 80% bilateral, every psi; analysed with the patient as
             the unit, with eyes treated as independent and with one eye per patient, as in simulate.py

GEE as commonly applied: per-eye losses in long format (one row per eye and model), the model as the only
covariate, the patient as the cluster, an exchangeable working correlation and robust (sandwich) standard errors
with no small-sample correction (the statsmodels default), Wald test of the model coefficient. Gaussian identity
link for the signed error (mean error), absolute error (MAE), squared error (RMSE), Brier score and log-loss
contributions; logit link for 0/1 outcomes (within tolerance; correct classification among diseased eyes for
sensitivity, among non-diseased eyes for specificity, and among all eyes for accuracy). The AUC, median absolute
error and standard deviation have no standard GEE analysis and are not included.

Needs statsmodels (pip install statsmodels) in addition to the packages simulate.py uses.

Usage (from this folder)
  python simulate_extra.py --pilot                 # small run to check everything works and time it
  python simulate_extra.py --part all --cores 30   # everything (about 30-40 minutes on 30 threads)
  python analyse_extra.py                          # summaries -> ./results/summary_extra_*.csv
Results go to ./output_suite/<part>/ (./output_suite_pilot/ for the pilot). Re-running skips complete cells.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulate as sim  # noqa: E402  (data generation, seeds, cells, suite and runner are shared)
import dgm  # noqa: E402

SMALL_N = (20, 30)
SENS_SIGMA_C2 = (0.5, 0.95)          # within-eye correlation of the two models' errors (0.8 in the main run)
SENS_FRAC = 0.8
GEE_METRICS = {"continuous": ("bias", "mae", "rmse", "within"),
               "binary": ("brier", "logloss", "sensitivity", "specificity", "accuracy")}
WITHIN = 0.5                         # tolerance for 'within' (the code's default)
THRESHOLD = 0.5                      # classification threshold (the code's default)


# ----------------------------------------------------------------------------------------------
# GEE
# ----------------------------------------------------------------------------------------------
def gee_p(loss_a, loss_b, groups, binary, cov_type="robust"):
    """Wald P value and coefficient for 'model B' in a GEE of per-eye losses (long format, exchangeable).
    cov_type 'robust' is the usual sandwich estimator; 'bias_reduced' is the Mancl-DeRouen small-sample correction."""
    import statsmodels.api as sm
    n = len(loss_a)
    if n < 4 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    y = np.concatenate([loss_a, loss_b]).astype(float)
    if binary and (y.min() == y.max()):
        return np.nan, np.nan
    g = np.concatenate([groups, groups])
    X = np.column_stack([np.ones(2 * n), np.r_[np.zeros(n), np.ones(n)]])
    fam = sm.families.Binomial() if binary else sm.families.Gaussian()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sm.GEE(y, X, groups=g, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=100,
                                                                                           cov_type=cov_type)
        p, b = float(r.pvalues[1]), float(r.params[1])
        return (p, b) if np.isfinite(p) else (np.nan, np.nan)
    except Exception:
        return np.nan, np.nan


def gee_all(task, idx, X, y, cov_type="robust"):
    """GEE for every metric it can handle; X holds errors (continuous) or probabilities (binary)."""
    out = {}
    if task == "continuous":
        e_a, e_b = X[:, 0], X[:, 1]
        losses = {"bias": (e_a, e_b, False), "mae": (np.abs(e_a), np.abs(e_b), False),
                  "rmse": (e_a ** 2, e_b ** 2, False),
                  "within": ((np.abs(e_a) <= WITHIN) * 1.0, (np.abs(e_b) <= WITHIN) * 1.0, True)}
        for m, (la, lb, binary) in losses.items():
            out[m] = gee_p(la, lb, idx, binary, cov_type)
    else:
        p_a, p_b = np.clip(X[:, 0], 1e-15, 1 - 1e-15), np.clip(X[:, 1], 1e-15, 1 - 1e-15)   # as the code's log-loss
        c_a, c_b = ((X[:, 0] >= THRESHOLD) == (y == 1)) * 1.0, ((X[:, 1] >= THRESHOLD) == (y == 1)) * 1.0
        out["brier"] = gee_p((X[:, 0] - y) ** 2, (X[:, 1] - y) ** 2, idx, False, cov_type)
        out["logloss"] = gee_p(-(y * np.log(p_a) + (1 - y) * np.log(1 - p_a)),
                               -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b)), idx, False, cov_type)
        d, h = y == 1, y == 0
        out["sensitivity"] = gee_p(c_a[d], c_b[d], idx[d], True, cov_type)
        out["specificity"] = gee_p(c_a[h], c_b[h], idx[h], True, cov_type)
        out["accuracy"] = gee_p(c_a, c_b, idx, True, cov_type)
    return out


def _lambdas(cell):
    lam = sim.LAMBDA_CONT if cell["task"] == "continuous" else sim.LAMBDA_BIN
    return (1.0, 1.0) if cell["scen"] == "null" else (1.0, lam)


def run_gee(args):
    """GEE on exactly the study simulate.py generated for this cell and replicate (same seed, same draws)."""
    cell, rep, _ = args
    rng = np.random.default_rng(sim.seed_for(cell["id"], rep))
    idx, X, y = sim.generate(cell, rng, _lambdas(cell))
    row = {"cell": cell["id"], "rep": rep, "n_eyes": len(idx), "n_bilateral": int(np.sum(np.bincount(idx) == 2))}
    for m, (p, b) in gee_all(cell["task"], idx, X, y).items():
        row[f"gee|{m}|p"], row[f"gee|{m}|diff"] = p, b
    return cell["id"], row


def run_small(args):
    """Very small studies: GEE and the code's patient-level tests on the same simulated study."""
    cell, rep, n_boot = args
    rng = np.random.default_rng(sim.seed_for(cell["id"], rep))
    idx, X, y = sim.generate(cell, rng, _lambdas(cell))
    row = {"cell": cell["id"], "rep": rep, "n_eyes": len(idx), "n_bilateral": int(np.sum(np.bincount(idx) == 2))}
    for m, (p, b) in gee_all(cell["task"], idx, X, y).items():
        row[f"gee|{m}|p"], row[f"gee|{m}|diff"] = p, b
    for m, (p, b) in gee_all(cell["task"], idx, X, y, "bias_reduced").items():
        row[f"gee_br|{m}|p"], row[f"gee_br|{m}|diff"] = p, b
    df = pd.DataFrame({"pid": idx, "A": X[:, 0], "B": X[:, 1]})
    truth = None
    if y is not None:
        df["y"] = y
        truth = "y"
    st = {"bootstrap_iterations": n_boot, "seed": int(rng.integers(0, 2 ** 31)), "verbose": False}
    try:
        long = sim.osx.compare_models(df, task=cell["task"], models=["A", "B"], patient="pid", truth=truth,
                                      settings=st)[1]
        sim._store(row, long, "cluster")
    except Exception:
        pass
    return cell["id"], row


def run_sens(args):
    """As simulate.run_study (patient as unit, eyes independent, one eye), with another within-eye correlation."""
    cell, rep, n_boot = args
    rng = np.random.default_rng(sim.seed_for(cell["id"], rep))
    idx, E = dgm.continuous_study(cell["n_pat"], cell["frac"], cell["psi"], cell["psi"], cell["shape"],
                                  _lambdas(cell), rng, sigma_c2=cell["sigma_c2"])
    df = pd.DataFrame({"pid": idx, "A": E[:, 0], "B": E[:, 1]})
    row = {"cell": cell["id"], "rep": rep, "n_eyes": len(df), "n_bilateral": int(np.sum(np.bincount(idx) == 2))}
    seeds = rng.integers(0, 2 ** 31, size=3)

    def run(data, seed, **extra):
        st = {"bootstrap_iterations": n_boot, "seed": int(seed), "verbose": False, **extra}
        return sim.osx.compare_models(data, task="continuous", models=["A", "B"], patient="pid", settings=st)[1]

    sim._store(row, run(df, seeds[0]), "cluster", cluster_fields=True)
    sim._store(row, run(sim.one_eye_per_patient(df, "pid", rng), seeds[1]), "one_eye")
    sim._store(row, run(df, seeds[2], unit="eye"), "naive")
    return cell["id"], row


# ----------------------------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------------------------
def gee_main_cells(n_null):
    return [dict(c, reps=n_null) for c in sim.main_cells(n_null, 0) if c["scen"] == "null"]


def gee_sweep_cells(reps):
    return sim.sweep_cells(reps)


def small_cells(reps):
    cells = []
    for f in sim.FRACS:
        for n in SMALL_N:
            cells.append(dict(part="small", task="continuous", shape="normal", psi=sim.SWEEP_PSI, frac=f, n_pat=n,
                              scen="null", reps=reps, id=f"cont_n{n}_f{f}_null"))
            cells.append(dict(part="small", task="binary", prev=0.3, psi=sim.SWEEP_PSI, frac=f, n_pat=n,
                              scen="null", reps=reps, id=f"bin_n{n}_f{f}_null"))
    return cells


def sens_cells(n_null, n_alt):
    cells = []
    for w in SENS_SIGMA_C2:
        for k in sim.PSIS:
            for scen, reps in (("null", n_null), ("alt", n_alt)):
                cells.append(dict(part="sens", task="continuous", shape="normal", psi=k, frac=SENS_FRAC,
                                  n_pat=sim.N_PATIENTS, scen=scen, reps=reps, sigma_c2=w,
                                  id=f"cont_normal_psi{k}_f{SENS_FRAC}_w{w}_{scen}"))
    return cells


def run_part(cells, out_dir, cores, n_boot, fn):
    """As simulate.run_part, with the per-study function passed in."""
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for c in cells if not os.path.exists(os.path.join(out_dir, c["id"] + ".csv.gz"))]
    if not todo:
        print(f"  all {len(cells)} cells already complete in {out_dir}")
        return 0.0
    tasks = [(c, r, n_boot) for c in todo for r in range(c["reps"])]
    need = {c["id"]: c["reps"] for c in todo}
    buf = {c["id"]: [] for c in todo}
    t0, done = time.time(), 0
    chunk = max(1, min(20, len(tasks) // (cores * 8) or 1))
    with Pool(processes=cores) as pool:
        for cid, row in pool.imap_unordered(fn, tasks, chunksize=chunk):
            buf[cid].append(row)
            done += 1
            if len(buf[cid]) == need[cid]:
                pd.DataFrame(buf.pop(cid)).sort_values("rep").to_csv(
                    os.path.join(out_dir, cid + ".csv.gz"), index=False, compression="gzip")
            if done % 500 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} studies  elapsed {el / 60:.1f} min  "
                      f"remaining ~{el / done * (len(tasks) - done) / 60:.1f} min", flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["gee_main", "gee_sweep", "small", "sens", "all"])
    ap.add_argument("--cores", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        out = os.path.join(HERE, "output_suite_pilot")
        n_null, n_alt, sweep_reps, n_boot = 10, 5, 5, 199
    else:
        out = os.path.join(HERE, "output_suite")
        n_null, n_alt, sweep_reps, n_boot = sim.N_NULL, sim.N_ALT, sim.SWEEP_REPS, sim.N_BOOT
    parts = ["gee_main", "gee_sweep", "small", "sens"] if a.part == "all" else [a.part]
    print(f"Output: {out}   cores: {a.cores}   bootstrap resamples: {n_boot}")
    for part in parts:
        print(f"\nPart: {part}")
        if part == "gee_main":
            run_part(gee_main_cells(n_null), os.path.join(out, part), a.cores, n_boot, run_gee)
        elif part == "gee_sweep":
            run_part(gee_sweep_cells(sweep_reps), os.path.join(out, part), a.cores, n_boot, run_gee)
        elif part == "small":
            run_part(small_cells(n_null), os.path.join(out, part), a.cores, n_boot, run_small)
        else:
            run_part(sens_cells(n_null, n_alt), os.path.join(out, part), a.cores, n_boot, run_sens)


if __name__ == "__main__":
    main()
