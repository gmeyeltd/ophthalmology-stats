"""
Checks for ophthalmology_stats.   Run:  python test_ophthalmology_stats.py   (or pytest)

The key check is the first group: for continuous outputs, ophthalmology_stats gives exactly the same differences,
confidence intervals, p-values and Romano-Wolf adjusted p-values as biometry_stats_advanced on the same data and
seed, so it is an extension of that library, not a different method.
"""
import os
import sys
import tempfile
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import ophthalmology_stats as osx  # noqa: E402

FOUNDATION = os.path.join(HERE, "reference", "biometry_stats_advanced.py")   # biometry-stats, for the identity check
RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("[PASS] " if cond else "[FAIL] ") + name)


def iol_like_data(n_pat=180, frac=0.5, J=4, seed=1):
    rng = np.random.default_rng(seed)
    sizes = np.where(rng.random(n_pat) < frac, 2, 1)
    pid = np.repeat(np.arange(n_pat), sizes)
    n = len(pid)
    shared = rng.normal(0, 0.35, n_pat)[pid] * 0.5 + rng.normal(0, 0.35, n)
    E = np.column_stack([shared + rng.normal(0, 0.12 * (1 + 0.1 * j), n) for j in range(J)])
    E = np.round(E, 2)
    df = pd.DataFrame(E, columns=[f"F{j}" for j in range(J)])
    df.insert(0, "patient_id", [f"P{p}" for p in pid])
    df["eye"] = np.where(pd.Series(pid).duplicated().values, "L", "R")
    return df


def binary_data(n_pat=220, frac=0.6, J=3, seed=2, one_eye=False):
    rng = np.random.default_rng(seed)
    sizes = np.ones(n_pat, int) if one_eye else np.where(rng.random(n_pat) < frac, 2, 1)
    pid = np.repeat(np.arange(n_pat), sizes)
    n = len(pid)
    liab = 0.6 * rng.normal(size=n_pat)[pid] + 0.8 * rng.normal(size=n)
    y = (liab > 0.5).astype(int)
    P = np.column_stack([1 / (1 + np.exp(-(1.5 * liab + rng.normal(0, 0.8 + 0.2 * j, n)))) for j in range(J)])
    df = pd.DataFrame(P, columns=[f"M{j}" for j in range(J)])
    df.insert(0, "patient_id", pid)
    df["truth"] = y
    df["eye"] = np.where(pd.Series(pid).duplicated().values, "L", "R")
    return df


def test_foundation_unchanged():
    src = open(FOUNDATION).read()
    body = src[src.index('"""', 3) + 3:]
    mine = open(os.path.join(ROOT, "ophthalmology_stats.py")).read()
    check("biometry_stats_advanced is included unchanged", body.strip() in mine)


def test_identity_with_foundation():
    spec = importlib.util.spec_from_file_location("bsa", FOUNDATION)
    bsa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsa)
    df = iol_like_data()
    B = 999
    fs = {"bootstrap_iterations": B, "report_ci": True}
    res_f, rw_f, _ = bsa.run_biometry_analysis(df.drop(columns=["eye"]), "All", settings=fs)
    res_f = bsa.adjust_pvalues(res_f, rw_stats=rw_f, method="romano_wolf")
    wide, long, info = osx.compare_models(df.drop(columns=["eye"]), task="continuous",
                                          settings={"bootstrap_iterations": B, "invalid_threshold": 4.0,
                                                    "verbose": False})
    common = [c for c in res_f.columns if c in wide.columns and c not in ("Group", "N")]
    worst = 0.0
    for c in common:
        a, b = res_f[c].values.astype(float), wide[c].values.astype(float)
        worst = max(worst, float(np.nanmax(np.abs(a - b))))
    check(f"continuous results identical to biometry_stats_advanced ({len(common)} columns, max |diff| {worst:.1e})",
          len(common) >= 25 and worst < 1e-12)


def test_loss_engine_matches_foundation():
    df = iol_like_data(seed=3)
    E = df[[c for c in df.columns if c.startswith("F")]].values
    _, idx = np.unique(df.patient_id.astype(str).values, return_inverse=True)
    K = idx.max() + 1
    rng = np.random.default_rng(7)
    b = rng.choice(K, size=(499, K))
    f = osx.all_pairs_bootstrap(E, idx, K, b, ci_level=0.95)
    g, _ = osx.all_pairs_loss_bootstrap({"MAE": np.abs(E), "MPE": E}, idx, K, b, ci_level=0.95)
    ok = all(abs(f[k]["P_MAE"] - g[k]["P_MAE"]) < 1e-12 and abs(f[k]["P_MPE"] - g[k]["P_MPE"]) < 1e-12
             and np.allclose(f[k]["CI_MAE"], g[k]["CI_MAE"]) for k in f)
    check("generic loss engine reproduces the foundation bootstrap-t (MAE, MPE, CIs)", ok)


def _delong(P, y, j, k):
    pos, neg = y == 1, y == 0
    v10, v01 = [], []
    for c in (j, k):
        a, b = osx._placements(P[pos, c], P[neg, c])
        v10.append(a); v01.append(b)
    s10 = np.cov(np.vstack(v10)); s01 = np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / pos.sum() + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / neg.sum()
    return np.mean(v10[0]) - np.mean(v10[1]), np.sqrt(var)


def test_auc_reduces_to_delong():
    df = binary_data(one_eye=True)
    P, y = df[["M0", "M1", "M2"]].values, df.truth.values
    K = len(df)
    out = osx.auc_all_pairs(P, y, np.arange(K), K)
    d, se = _delong(P, y, 0, 1)
    check("clustered AUC equals DeLong when every patient has one eye",
          abs(out[(0, 1)][2] - d) < 1e-12 and abs(out[(0, 1)][6] - se) / se < 1e-9)


def test_binary_run():
    df = binary_data()
    wide, long, info = osx.compare_models(df, task="binary", settings={"bootstrap_iterations": 499, "verbose": False})
    ok = (set(long.metric) == set(osx.BINARY_METRICS) and long.p.between(0, 1).all()
          and (wide.filter(like="p_adj").values >= wide[[c.replace("_adj", "") for c in wide.filter(like="p_adj")]]
               .values - 1e-12).all() and info["clustered"])
    check("binary task: all metrics, valid p-values, adjusted >= raw", ok)
    auto = osx.compare_models(df.drop(columns=["eye"]), task="auto", settings={"bootstrap_iterations": 99,
                                                                               "verbose": False})[2]["task"]
    check("task='auto' detects a classifier", auto == "binary")


def test_weighting_and_one_eye():
    df = binary_data(one_eye=True)
    s = {"bootstrap_iterations": 299, "verbose": False, "metrics": ["brier"]}
    a = osx.compare_models(df, task="binary", settings=s)[1]
    b = osx.compare_models(df, task="binary", settings={**s, "weighting": "patient"})[1]
    check("patient weights equal eye weights when every patient has one eye", np.allclose(a.p, b.p))
    df2 = binary_data()
    info = osx.compare_models(df2, task="binary", settings={**s, "one_eye": True})[2]
    check("one_eye keeps one eye per patient", info["n_eyes"] == info["n_patients"] and not info["clustered"])
    opt = osx.compare_models(df2, task="binary", settings={**s, "weighting": "optimal"})[1]
    check("minimum-variance weighting runs", opt.p.between(0, 1).all())


def test_subgroups_and_excel():
    df = binary_data()
    res = osx.analyze(df, task="binary", subgroup="eye", settings={"bootstrap_iterations": 199, "verbose": False})
    check("laterality subgroups reported", set(res.Group) == {"All", "eye = L", "eye = R"})
    path = os.path.join(tempfile.mkdtemp(), "r.xlsx")
    osx.write_results_to_excel(res, path)
    sheets = pd.ExcelFile(path).sheet_names
    check("Excel output has Results, Statistical Tests, Estimates and Sample sheets",
          {"Results", "Statistical Tests", "Estimates", "Sample"} <= set(sheets))


def test_ties():
    d = np.zeros(999)
    check("identical models give p = 1 (ties counted in both tails)", osx._calculate_percentile_bootstrap_p(d) == 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok} of {len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
