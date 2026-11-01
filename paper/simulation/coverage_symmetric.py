"""
Coverage of the patient-level 95% CIs under the null after ophthalmology-stats 1.4.0, which made the CIs for the
mean error, MAE, Brier score and log-loss symmetric bootstrap-t intervals (difference +/- q * SE, where q is the 95th
percentile of |t*|), matching the symmetric P value. Other CIs, and every P value, are unchanged.

No bootstrap needs repeating for most studies. With B resamples, P = (1 + m) / (B + 1), where m is the number of
|t*| at or above the observed |t|, and the symmetric CI contains 0 exactly when |t| <= q. With q interpolated
between the sorted |t*| at positions i and i + 1 (i = floor(0.95 (B - 1))):
    m >= B - i       the CI contains 0
    m <= B - i - 2   it does not
    m == B - i - 1   undetermined from P: the study is regenerated from its seed and reanalysed
(B = 1999: P >= 0.051 contains 0, P <= 0.050 does not, P = 0.0505 is reanalysed.) Each reanalysed study must
reproduce its stored P values exactly, which proves that it was regenerated correctly.

Usage (from this folder, after simulate.py; uses the library in ../github/ophthalmology-stats, version 1.4.0):
    python coverage_symmetric.py [--cores 30]
    python analyse_simulation.py          # picks up results/coverage_symmetric.csv
    python make_figures.py                # rewrites results/table_s_type1_by_metric.csv
--check N reanalyses the first N studies of every null cell whatever their P value, as a test of the rule.
"""
import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import simulate as sim

METRICS = {"continuous": ("bias", "mae"), "binary": ("brier", "logloss")}
ALPHA = 0.05


def rule(p, B=sim.N_BOOT, level=0.95):
    """1 = CI contains 0, 0 = it does not, -1 = undetermined (reanalyse), -2 = no P value."""
    p = np.asarray(p, float)
    m = np.rint(np.nan_to_num(p) * (B + 1)).astype(int) - 1
    i = int(np.floor(level * (B - 1)))
    return np.where(np.isnan(p), -2, np.where(m >= B - i, 1, np.where(m <= B - i - 2, 0, -1)))


def reanalyse(args):
    """Repeat simulate.run_study up to the patient-level analysis, with the same seeds."""
    cell, rep = args
    rng = np.random.default_rng(sim.seed_for(cell["id"], rep))
    lam = sim.LAMBDA_CONT if cell["task"] == "continuous" else sim.LAMBDA_BIN
    lambdas = (1.0, 1.0) if cell["scen"] == "null" else (1.0, lam)
    idx, X, y = sim.generate(cell, rng, lambdas)
    df = pd.DataFrame({"pid": idx, "A": X[:, 0], "B": X[:, 1]})
    truth = None
    if y is not None:
        df["y"] = y
        truth = "y"
    seeds = rng.integers(0, 2 ** 31, size=5)
    st = {"bootstrap_iterations": sim.N_BOOT, "seed": int(seeds[0]), "verbose": False}
    long = sim.osx.compare_models(df, task=cell["task"], models=["A", "B"], patient="pid", truth=truth,
                                  settings=st)[1]
    long = long[long.metric.isin(METRICS[cell["task"]])]
    return cell["id"], rep, {r.metric: (r.p, r.ci_low, r.ci_high) for r in long.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(sim.HERE, "output_suite", "main"))
    ap.add_argument("--out", default=os.path.join(sim.HERE, "results", "coverage_symmetric.csv"))
    a = ap.parse_args()
    print("ophthalmology-stats", sim.osx.__version__)
    if tuple(int(x) for x in sim.osx.__version__.split(".")[:2]) < (1, 4):
        raise SystemExit("needs ophthalmology-stats 1.4.0 or later in ../github/ophthalmology-stats")
    cells = {c["id"]: c for c in sim.main_cells(sim.N_NULL, sim.N_ALT) if c["scen"] == "null"}
    data, todo = {}, []
    for cid, cell in sorted(cells.items()):
        path = os.path.join(a.outdir, cid + ".csv.gz")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        data[cid] = d
        for m in METRICS[cell["task"]]:
            d[f"{m}|contains0"] = rule(d[f"cluster|{m}|p"])
        need = np.zeros(len(d), bool)
        for m in METRICS[cell["task"]]:
            need |= d[f"{m}|contains0"].values == -1
        need[:a.check] = True
        todo += [(cell, int(r)) for r in d.loc[need, "rep"]]
    print(f"{len(data)} of {len(cells)} null cells found; {len(todo)} studies to reanalyse")
    with Pool(a.cores) as pool:
        redone = pool.map(reanalyse, todo, chunksize=4)
    agree = 0
    for cid, rep, res in redone:
        d = data[cid]
        k = int(np.flatnonzero(d["rep"].values == rep)[0])
        for m, (p, lo, hi) in res.items():
            if not np.isclose(p, d.at[k, f"cluster|{m}|p"], rtol=0, atol=1e-12):
                raise SystemExit(f"{cid} rep {rep} {m}: P {p} differs from the stored {d.at[k, f'cluster|{m}|p']}")
            c = int(lo <= 0 <= hi)
            if d.at[k, f"{m}|contains0"] != -1:
                if d.at[k, f"{m}|contains0"] != c:
                    raise SystemExit(f"{cid} rep {rep} {m}: the rule and the reanalysis disagree")
                agree += 1
            d.at[k, f"{m}|contains0"] = c
    if a.check:
        print(f"check: the rule agreed with the reanalysis in all {agree} determined cases")
    rows = []
    for cid, d in data.items():
        for m in METRICS[cells[cid]["task"]]:
            c = d[f"{m}|contains0"]
            ok = c >= 0
            rows.append({"cell": cid, "metric": m, "n_rep": int(ok.sum()), "coverage": float(c[ok].mean()),
                         "reject": float((d.loc[ok, f"cluster|{m}|p"] < ALPHA).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"written {a.out}: coverage {100 * out.coverage.min():.1f}%-{100 * out.coverage.max():.1f}%")
    print("by metric:", {m: f"{100 * g.coverage.min():.1f}-{100 * g.coverage.max():.1f}"
                         for m, g in out.groupby("metric")})


if __name__ == "__main__":
    main()
