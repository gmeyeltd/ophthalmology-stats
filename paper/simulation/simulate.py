"""
Monte Carlo simulation: comparing two prediction models when some patients contribute both eyes.

Parts
  main     type I error, power and precision for every metric, three analysis strategies, across the grid
           (fellow-eye correlation psi x proportion of bilateral patients x error shape / prevalence).
           psi is the fellow-eye correlation of every error component (shared and model-specific);
           the resulting rho_delta is measured in every study and reported.
  sweep    study size from 50 to 1000 patients (patients needed for 80% power; small-sample type I error)
  lottery  four models of graded accuracy: how often a one-eye analysis changes the ranking or the
           significance of a comparison depending on which eye is kept

Strategies analysed in every simulated study (all with ophthalmology_stats, the published suite)
  cluster    both eyes, patient as the unit, eye-weighted (recommended default)
  patient    both eyes, patient as the unit, each patient's eyes averaged (mean-loss metrics only)
  optimal    both eyes, patient as the unit, minimum-variance weights (mean-loss metrics only)
  one_eye    one randomly chosen eye per patient, standard (independent-data) tests
  naive      both eyes, every eye treated as an independent patient (eye-level bootstrap, DeLong, ...)

Usage (from this folder)
  python simulate.py --pilot                 # small run to check everything works and time it
  python simulate.py --part main             # full run of one part
  python simulate.py --part all --cores 15   # everything; default cores = all but one
Results go to ./output_suite/<part>/ (./output_suite_pilot/ for the pilot), one compressed CSV per grid cell.
Re-running skips cells that are already complete, so an interrupted run can simply be restarted.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # one thread per process; parallelism is across studies

import argparse
import sys
import time
import zlib
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))                             # repository layout (paper/simulation)
sys.path.insert(0, os.path.join(HERE, "..", "github", "ophthalmology-stats"))  # AI_Paper layout
import io as _io  # noqa: E402
import contextlib as _ctx  # noqa: E402
with _ctx.redirect_stdout(_io.StringIO()):               # the library prints its seed on import
    import ophthalmology_stats as osx  # noqa: E402
import dgm  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Design (ADEMP). Change here only, and record any change.
# ----------------------------------------------------------------------------------------------
MASTER_SEED = 20261022
N_PATIENTS = 250
N_BOOT = 1999
FRACS = (0.4, 0.8)                         # proportion of patients contributing both eyes
PSIS = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98)  # fellow-eye correlation of every error component (sets rho_delta)
SHAPES = ("normal", "heavy", "skewed", "discrete")
LAMBDA_CONT = 1.4                          # alternative: model B's own error 1.4 x larger (SD ratio ~1.09;
                                           #   ~45% power with one eye per patient at 250 patients)
PREVALENCES = (0.1, 0.3)
TAU = 0.8                                  # model noise: AUC ~0.88 for the better model
LAMBDA_BIN = 1.4                           # alternative: model B noisier (AUC ~0.82; ~40% one-eye power)
N_NULL, N_ALT = 2000, 1000

SWEEP_N = (50, 100, 150, 200, 300, 400, 600, 800, 1000)
SWEEP_PSI, SWEEP_REPS = 0.5, 1000        # null and alternative each, per study size and bilateral fraction

LOTTERY_STUDIES, LOTTERY_DRAWS, LOTTERY_FRAC, LOTTERY_PSI = 200, 200, 0.8, 0.5
LOTTERY_LAMBDAS_CONT = (1.0, 1.2, 1.4, 1.6)   # four models of graded accuracy (model 1 best)
LOTTERY_LAMBDAS_BIN = (1.0, 1.2, 1.4, 1.6)
LOTTERY_TEST = (0, 2)                          # pair tested for significance (same gap as the main alternative)

MEAN_METRICS = {"continuous": ["bias", "mae", "rmse"], "binary": ["brier", "logloss"]}
FIELDS_ALL = ("p", "diff")
FIELDS_CLUSTER = ("est_a", "est_b", "rho_delta", "ci_low", "ci_high")


# ----------------------------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------------------------
def seed_for(cell_id, rep):
    return np.random.SeedSequence([MASTER_SEED, zlib.crc32(cell_id.encode()), rep])


def main_cells(n_null, n_alt):
    cells = []
    for f in FRACS:
        for k in PSIS:
            for shape in SHAPES:
                for scen, n_rep in (("null", n_null), ("alt", n_alt)):
                    cells.append(dict(part="main", task="continuous", shape=shape, psi=k, frac=f,
                                      n_pat=N_PATIENTS, scen=scen, reps=n_rep,
                                      id=f"cont_{shape}_psi{k}_f{f}_{scen}"))
            for prev in PREVALENCES:
                for scen, n_rep in (("null", n_null), ("alt", n_alt)):
                    cells.append(dict(part="main", task="binary", prev=prev, psi=k, frac=f,
                                      n_pat=N_PATIENTS, scen=scen, reps=n_rep,
                                      id=f"bin_p{prev}_psi{k}_f{f}_{scen}"))
    return cells


def sweep_cells(reps):
    cells = []
    for f in FRACS:
        for n in SWEEP_N:
            for scen in ("null", "alt"):
                cells.append(dict(part="sweep", task="continuous", shape="normal", psi=SWEEP_PSI, frac=f,
                                  n_pat=n, scen=scen, reps=reps, id=f"cont_n{n}_f{f}_{scen}"))
                cells.append(dict(part="sweep", task="binary", prev=0.3, psi=SWEEP_PSI, frac=f,
                                  n_pat=n, scen=scen, reps=reps, id=f"bin_n{n}_f{f}_{scen}"))
    return cells


def lottery_cells(studies):
    return [dict(part="lottery", task=t, psi=LOTTERY_PSI, frac=LOTTERY_FRAC, n_pat=N_PATIENTS,
                 scen="alt", reps=studies, id=f"lottery_{t}", **({"shape": "normal"} if t == "continuous" else {"prev": 0.3}))
            for t in ("continuous", "binary")]


_MAPS = {}


def maps_for(prev, lambdas):
    key = (prev, tuple(lambdas))
    if key not in _MAPS:
        _MAPS[key] = [dgm.calibration_map(prev, TAU, lam) for lam in lambdas]
    return _MAPS[key]


# ----------------------------------------------------------------------------------------------
# One simulated study
# ----------------------------------------------------------------------------------------------
def generate(cell, rng, lambdas):
    if cell["task"] == "continuous":
        idx, E = dgm.continuous_study(cell["n_pat"], cell["frac"], cell["psi"], cell["psi"], cell["shape"], lambdas, rng)
        return idx, E, None
    idx, y, P = dgm.binary_study(cell["n_pat"], cell["frac"], cell["psi"], cell["psi"], cell["prev"], TAU, lambdas,
                                 maps_for(cell["prev"], lambdas), rng)
    return idx, P, y


def _store(row, out, strategy, cluster_fields=False):
    for r in out.itertuples(index=False):
        for fld in FIELDS_ALL:
            row[f"{strategy}|{r.metric}|{fld}"] = getattr(r, fld)
        if cluster_fields:
            for fld in FIELDS_CLUSTER:
                row[f"{strategy}|{r.metric}|{fld}"] = getattr(r, fld)


def run_study(args):
    cell, rep, n_boot = args
    rng = np.random.default_rng(seed_for(cell["id"], rep))
    lam = LAMBDA_CONT if cell["task"] == "continuous" else LAMBDA_BIN
    lambdas = (1.0, 1.0) if cell["scen"] == "null" else (1.0, lam)
    idx, X, y = generate(cell, rng, lambdas)
    df = pd.DataFrame({"pid": idx, "A": X[:, 0], "B": X[:, 1]})
    truth = None
    if y is not None:
        df["y"] = y
        truth = "y"
    task = cell["task"]
    row = {"cell": cell["id"], "rep": rep, "n_eyes": len(df), "n_bilateral": int(np.sum(np.bincount(idx) == 2))}
    seeds = rng.integers(0, 2 ** 31, size=5)

    def run(data, seed, **extra):
        st = {"bootstrap_iterations": n_boot, "seed": int(seed), "verbose": False, **extra}
        return osx.compare_models(data, task=task, models=["A", "B"], patient="pid", truth=truth, settings=st)[1]

    _store(row, run(df, seeds[0]), "cluster", cluster_fields=True)
    for j, w in enumerate(("patient", "optimal")):
        _store(row, run(df, seeds[1 + j], weighting=w, metrics=MEAN_METRICS[task]), w)
    _store(row, run(one_eye_per_patient(df, "pid", rng), seeds[3]), "one_eye")
    _store(row, run(df, seeds[4], unit="eye"), "naive")
    return cell["id"], row


def one_eye_per_patient(df, patient, rng):
    """Keep one randomly chosen eye per patient (the usual alternative to a clustered analysis)."""
    keys = rng.random(len(df))
    keep = pd.Series(keys, index=df.index).groupby(df[patient].values).idxmin().values
    return df.loc[np.sort(keep)]


def run_lottery(args):
    """One study of four models; the eye kept from each bilateral patient is redrawn many times."""
    cell, rep, n_boot, draws = args
    rng = np.random.default_rng(seed_for(cell["id"], rep))
    task = cell["task"]
    lambdas = LOTTERY_LAMBDAS_CONT if task == "continuous" else LOTTERY_LAMBDAS_BIN
    idx, X, y = generate(cell, rng, lambdas)
    a, b = LOTTERY_TEST
    K = idx.max() + 1

    def ranking_and_test(sel, unit_idx, boot_seed):
        """Top-ranked model (MAE or AUC) and the suite's test of models a and b on the selected eyes."""
        data = pd.DataFrame({"pid": unit_idx, "M0": X[sel, a], "M1": X[sel, b]})
        if task == "continuous":
            top = int(np.argmin(np.abs(X[sel]).mean(axis=0)))
            metric = "mae"
        else:
            yy = y[sel]
            data["y"] = yy
            auc = [osx._placements(X[sel, r][yy == 1], X[sel, r][yy == 0])[0].mean() for r in range(X.shape[1])]
            top = int(np.argmax(auc))
            metric = "auc"
        st = {"bootstrap_iterations": n_boot, "seed": int(boot_seed), "verbose": False, "metrics": [metric]}
        long = osx.compare_models(data, task=task, models=["M0", "M1"], patient="pid",
                                  truth=None if task == "continuous" else "y", settings=st)[1]
        return top, float(long.p.iloc[0])

    all_eyes = np.arange(len(idx))
    top_c, p_c = ranking_and_test(all_eyes, idx, rng.integers(2 ** 31))
    first = np.unique(idx, return_index=True)[1]
    sizes = np.bincount(idx)
    tops, ps = [], []
    for _ in range(draws):
        sel = first + (rng.random(K) < 0.5) * (sizes == 2)
        t, p = ranking_and_test(sel, np.arange(K), rng.integers(2 ** 31))
        tops.append(t); ps.append(p)
    tops, ps = np.array(tops), np.array(ps)
    row = {"cell": cell["id"], "rep": rep,
           "cluster_top_is_best": top_c == 0, "cluster_sig": p_c < 0.05, "cluster_p": p_c,
           "one_eye_frac_top_is_best": float(np.mean(tops == 0)),
           "one_eye_n_distinct_top": int(len(np.unique(tops))),
           "one_eye_frac_sig": float(np.mean(ps < 0.05)),
           "one_eye_median_p": float(np.median(ps))}
    return cell["id"], row


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------
def run_part(cells, out_dir, cores, n_boot, draws=None):
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for c in cells if not os.path.exists(os.path.join(out_dir, c["id"] + ".csv.gz"))]
    if not todo:
        print(f"  all {len(cells)} cells already complete in {out_dir}")
        return 0.0
    if draws is None:
        tasks = [(c, r, n_boot) for c in todo for r in range(c["reps"])]
        fn = run_study
    else:
        tasks = [(c, r, n_boot, draws) for c in todo for r in range(c["reps"])]
        fn = run_lottery
    need = {c["id"]: c["reps"] for c in todo}
    buf = {c["id"]: [] for c in todo}
    t0 = time.time()
    done = 0
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
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} studies  elapsed {el / 60:.1f} min  remaining ~{eta / 60:.1f} min",
                      flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["main", "sweep", "lottery", "all"])
    ap.add_argument("--cores", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--pilot", action="store_true", help="small run for checking and timing")
    args = ap.parse_args()

    if args.pilot:
        out = os.path.join(HERE, "output_suite_pilot")
        n_null, n_alt, sweep_reps, lot_studies, lot_draws, n_boot = 20, 10, 10, 4, 20, 499
    else:
        out = os.path.join(HERE, "output_suite")
        n_null, n_alt, sweep_reps, lot_studies, lot_draws, n_boot = (N_NULL, N_ALT, SWEEP_REPS, LOTTERY_STUDIES,
                                                                     LOTTERY_DRAWS, N_BOOT)
    parts = ["main", "sweep", "lottery"] if args.part == "all" else [args.part]
    print(f"Output: {out}   cores: {args.cores}   bootstrap resamples: {n_boot}")
    timing = {}
    for part in parts:
        print(f"\nPart: {part}")
        if part == "main":
            cells = main_cells(n_null, n_alt)
            timing[part] = (run_part(cells, os.path.join(out, part), args.cores, n_boot), cells)
        elif part == "sweep":
            cells = sweep_cells(sweep_reps)
            timing[part] = (run_part(cells, os.path.join(out, part), args.cores, n_boot), cells)
        else:
            cells = lottery_cells(lot_studies)
            timing[part] = (run_part(cells, os.path.join(out, part), args.cores, n_boot, draws=lot_draws), cells)

    if args.pilot:
        print("\nEstimated full run time with", args.cores, "cores (from the pilot):")
        scale = {"main": ((N_NULL + N_ALT) / (n_null + n_alt)) * (N_BOOT / n_boot),
                 "sweep": (SWEEP_REPS / sweep_reps) * (N_BOOT / n_boot),
                 "lottery": (LOTTERY_STUDIES * LOTTERY_DRAWS) / (lot_studies * lot_draws)}
        total = 0
        for part, (sec, _) in timing.items():
            est = sec * scale[part]
            total += est
            print(f"  {part:8s} ~{est / 3600:.1f} h")
        print(f"  total    ~{total / 3600:.1f} h (bootstrap time scales roughly with the number of resamples)")


if __name__ == "__main__":
    main()
