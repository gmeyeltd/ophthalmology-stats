"""
Checks for ophthalmology_stats.   Run:  python test_ophthalmology_stats.py   (or pytest)

The key check is the first group: for continuous outputs, ophthalmology_stats gives exactly the same differences,
confidence intervals, p-values and Romano-Wolf adjusted p-values as biometry_stats_advanced on the same data and
seed, so it is an extension of that library, not a different method.
"""
import os
import sys
import tempfile
import warnings
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
    check("Excel output has Results, Statistical Tests, Model Performance, Estimates and Sample sheets",
          {"Results", "Statistical Tests", "Model Performance", "Estimates", "Sample"} <= set(sheets))


def test_model_performance():
    df = binary_data()
    s = {"bootstrap_iterations": 499, "verbose": False}
    w, l, info = osx.compare_models(df, task="binary", settings=s)
    mp = pd.DataFrame(info["model_performance"])
    est = l.drop_duplicates(["model_a", "metric"]).set_index(["model_a", "metric"]).est_a
    same = all(abs(mp.set_index(["model", "metric"]).estimate[k] - v) < 1e-12 for k, v in est.items())
    check("each model's estimates match the comparison table", same)
    check("each model's patient-level CI contains its estimate",
          bool(((mp.ci_low <= mp.estimate) & (mp.estimate <= mp.ci_high)).all()))
    eye = pd.DataFrame(osx.compare_models(df, task="binary", settings={**s, "unit": "eye"})[2]["model_performance"])
    wp = (mp.ci_high - mp.ci_low)[mp.metric == "auc"].mean()
    we = (eye.ci_high - eye.ci_low)[eye.metric == "auc"].mean()
    check("patient-level CIs are wider than eye-level CIs when fellow eyes are alike", wp > we)


def test_adjusted_cis():
    lbl = "95% CI"
    df = iol_like_data(seed=5).drop(columns=["eye"])
    wide, long, _ = osx.compare_models(df, task="continuous", settings={"bootstrap_iterations": 999, "verbose": False})
    ok_contain, ok_wider, ok_consistent = True, True, True
    for m in ("MPE", "MAE", "MedAE"):
        d, lo, hi = wide[f"{m} Diff"], wide[f"{m} Diff {lbl} adj low"], wide[f"{m} Diff {lbl} adj high"]
        ok_contain &= bool(((lo <= d) & (d <= hi)).all())
        ok_wider &= bool(((hi - lo) >= (wide[f"{m} Diff {lbl} high"] - wide[f"{m} Diff {lbl} low"])).mean() >= 0.8)
        excl = (lo > 0) | (hi < 0)                     # single-step interval excludes 0 -> step-down p_adj below 0.05
        ok_consistent &= bool((wide.loc[excl, f"Boot ({m}) p_adj"] <= 0.05 + 1e-12).all())
    check("adjusted CIs contain the difference (MPE, MAE, MedAE)", ok_contain)
    check("adjusted CIs are wider than unadjusted ones", ok_wider)
    check("an adjusted CI excluding 0 always has Romano-Wolf p_adj <= 0.05", ok_consistent)
    check("RMSE (tested on the MSE scale) gets the Bonferroni percentile interval, not a max-|t| one",
          bool((long[long.metric == "rmse"].ci_adj_method == "Bonferroni (percentile bootstrap)").all()))
    lmae = long[long.metric == "mae"]
    check("long table carries the adjusted CIs", np.allclose(lmae["ci_adj_low"].values, wide[f"MAE Diff {lbl} adj low"].values))
    w1, _, _ = osx.compare_models(df[["patient_id", "F0", "F1"]], task="continuous",
                                  settings={"bootstrap_iterations": 499, "verbose": False})
    check("single comparison: adjusted CI equals the unadjusted CI",
          np.allclose(w1[f"MAE Diff {lbl} adj low"], w1[f"MAE Diff {lbl} low"]) and
          np.allclose(w1[f"MAE Diff {lbl} adj high"], w1[f"MAE Diff {lbl} high"]))
    w0, _, _ = osx.compare_models(df, task="continuous", settings={"bootstrap_iterations": 499, "report_ci": False, "verbose": False})
    check("no adjusted CIs when report_ci is off", not any(" adj " in c for c in w0.columns))
    wb, _, _ = osx.compare_models(binary_data(), task="binary", truth="truth",
                                  settings={"bootstrap_iterations": 499, "verbose": False})
    check("binary: adjusted CIs for Brier and log-loss", all(f"{m} Diff {lbl} adj low" in wb.columns for m in ("Brier", "LogLoss")))


def test_adjusted_cis_follow_fwer_method():
    lbl = "95% CI"
    df = iol_like_data(seed=5).drop(columns=["eye"])
    widths, ok = {}, True
    for method in ("romano_wolf", "bonferroni", "holm", "hommel", "fdr_bh"):
        w, _, info = osx.compare_models(df, task="continuous", settings={"bootstrap_iterations": 1999, "fwer_method": method,
                                                                         "verbose": False})
        for m in ("MPE", "MAE", "MedAE"):
            lo, hi, pa = w[f"{m} Diff {lbl} adj low"], w[f"{m} Diff {lbl} adj high"], w[f"Boot ({m}) p_adj"]
            ok &= bool((pa[(lo > 0) | (hi < 0)] <= 0.05 + 1e-9).all())   # excludes 0 -> significant under that method
        widths[method] = float((w[f"MAE Diff {lbl} adj high"] - w[f"MAE Diff {lbl} adj low"]).mean())
        ok &= method in info["methodology"]["Adjusted Confidence Intervals"].lower().replace("-", "_") or method != "romano_wolf"
    check("adjusted CIs follow fwer_method: an interval excluding 0 is significant under that method", ok)
    check("Holm and Hommel use the Bonferroni intervals", widths["holm"] == widths["bonferroni"] == widths["hommel"])
    check("Bonferroni intervals are wider than the max-|t| (Romano-Wolf) intervals", widths["bonferroni"] > widths["romano_wolf"])
    w, _, info = osx.compare_models(df, task="continuous", settings={"bootstrap_iterations": 499, "fwer_method": "fdr_by",
                                                                     "verbose": False})
    check("no adjusted CIs for other methods, and the methodology says so",
          not any(" adj " in c for c in w.columns) and info["methodology"]["Adjusted Confidence Intervals"].startswith("None"))


def with_axial_length(n_pat=240, seed=8):
    """iol_like_data with a numeric axial length per eye (fellow eyes alike) and a nominal site per patient."""
    df = iol_like_data(n_pat=n_pat, seed=seed)
    rng = np.random.default_rng(seed)
    pid = df.patient_id.str[1:].astype(int).values
    df["axial_length_mm"] = np.round(rng.normal(23.5, 1.2, n_pat)[pid] + rng.normal(0, 0.1, len(df)), 2)
    df["site"] = rng.choice(["north", "south", "east"], n_pat)[pid]
    return df


def _same_frames(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            if not np.array_equal(x.values.astype(float), y.values.astype(float), equal_nan=True):
                return False
        elif list(x.astype(str)) != list(y.astype(str)):
            return False
    return True


def test_prediction_errors_are_the_primary_input():
    pred = pd.read_csv(os.path.join(ROOT, "examples", "example_continuous.csv"))
    models = ["model_A", "model_B", "model_C"]
    pe = pred.drop(columns=["truth"])
    for m in models:
        pe[m] = pred[m] - pred["truth"]                     # prediction errors: prediction minus reference
    s = {"bootstrap_iterations": 499, "verbose": False}
    w1, l1, i1 = osx.compare_models(pe, task="auto", settings=s)
    w2, l2, i2 = osx.compare_models(pred, task="continuous", truth="truth", settings=s)
    check("task='auto' with no reference column: continuous, the model columns taken as prediction errors",
          i1["task"] == "continuous" and i1["truth"] is None and i1["models"] == models)
    check("prediction errors give exactly the same results as predictions plus a truth column",
          _same_frames(w1, w2) and _same_frames(l1, l2) and i1["model_performance"] == i2["model_performance"])
    a1 = osx.analyze(pe, subgroup="eye", settings=s)
    a2 = osx.analyze(pred, truth="truth", subgroup="eye", settings=s)
    check("analyze: prediction errors and predictions plus truth agree in every group",
          _same_frames(a1, a2) and _same_frames(pd.DataFrame(a1.attrs["long"]), pd.DataFrame(a2.attrs["long"])))
    check("task='auto' still uses a column named like a reference (truth) as the reference",
          osx.compare_models(pred, task="auto", settings=s)[2]["truth"] == "truth")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        osx._detect_columns(pred, "continuous", None, None, None)
    check("task='continuous' without truth warns that a column named 'truth' is analyzed as a model",
          any("truth" in str(x.message) for x in caught))


def test_nominal_subgroup():
    df = with_axial_length().drop(columns=["axial_length_mm"])
    s = {"bootstrap_iterations": 299, "verbose": False}
    res = osx.analyze(df, task="continuous", subgroup="site", settings=s)
    check("nominal subgroup: one group per level", set(res.Group) == {"All", "site = east", "site = north", "site = south"})
    sub = df[df.site == "north"]
    w, _, info = osx.compare_models(sub, task="continuous", models=["F0", "F1", "F2", "F3"], settings=s,
                                    group_name="site = north")
    mine = res[res.Group == "site = north"].reset_index(drop=True)
    check("a subgroup's results equal compare_models on its eyes", _same_frames(mine, w))
    d = res.attrs["group_info"]["site = north"]["definition"]
    check("group_info records the definition with eyes and patients",
          d == f"site = north: {info['n_eyes']} eyes, {info['n_patients']} patients")


def test_continuous_subgroup():
    df = with_axial_length()
    models = ["F0", "F1", "F2", "F3"]
    s = {"bootstrap_iterations": 299, "verbose": False}
    check("without the subgroup argument a numeric column such as axial_length_mm would be taken as a model",
          "axial_length_mm" in osx._detect_columns(df, "auto", None, None, None)[1])
    res = osx.analyze(df, subgroup="axial_length_mm", settings={**s, "subgroup_cuts": [24.5, 22]})
    names = ["All", "axial_length_mm <22", "axial_length_mm 22 to <24.5", "axial_length_mm >=24.5"]
    check("cut points give <22, 22 to <24.5 and >=24.5", list(dict.fromkeys(res.Group)) == names)
    gi = res.attrs["group_info"]
    long = pd.DataFrame(res.attrs["long"])
    check("the subgroup column is never a model", all(gi[g]["models"] == models for g in names)
          and "axial_length_mm" not in set(long.model_a) | set(long.model_b))
    al = df.axial_length_mm
    n_mid = int(((al >= 22) & (al < 24.5)).sum())
    k_mid = df[(al >= 22) & (al < 24.5)].patient_id.nunique()
    check("each cut-point group holds the right eyes, and its definition says so",
          gi["axial_length_mm 22 to <24.5"]["definition"] == f"axial_length_mm 22 to <24.5: {n_mid} eyes, {k_mid} patients"
          and gi["axial_length_mm <22"]["n_eyes"] == int((al < 22).sum()))
    w, _, _ = osx.compare_models(df[al >= 24.5], task="continuous", models=models, settings=s,
                                 group_name="axial_length_mm >=24.5")
    check("a cut-point group's results equal compare_models on its eyes",
          _same_frames(res[res.Group == "axial_length_mm >=24.5"].reset_index(drop=True), w))
    rb = osx.analyze(df, subgroup="axial_length_mm", settings={**s, "subgroup_bins": 3})
    gb = [g for g in rb.attrs["group_info"] if g != "All"]
    sizes = [rb.attrs["group_info"][g]["n_eyes"] for g in gb]
    check("bins=3 gives tertiles with (up to ties) equal numbers of eyes, labeled with their ranges",
          len(gb) == 3 and all("tertile" in g and " to " in g for g in gb)
          and max(sizes) - min(sizes) <= 0.03 * len(df) and sum(sizes) == len(df))
    rd = osx.analyze(df, subgroup="axial_length_mm", settings=s)
    sub_row = [r for r in rd.attrs["methodology_summary"] if r["Test Category"] == "Subgroups"]
    check("a numeric column with more than 10 values and no cuts or bins: tertiles by default, and the output says so",
          [g for g in rd.attrs["group_info"] if g != "All"] == gb and sub_row and "by default" in sub_row[0]["Methodology Applied"])
    small = with_axial_length(n_pat=60, seed=9)[["patient_id", "F0", "F1", "axial_length_mm"]]
    tmp = tempfile.mkdtemp()
    small.to_csv(os.path.join(tmp, "e.csv"), index=False)
    cli = osx._command_line([os.path.join(tmp, "e.csv"), "--subgroup", "axial_length_mm", "--bins", "2",
                             "--out", os.path.join(tmp, "e.xlsx")])
    check("command line: --subgroup COLUMN --bins Q", set(cli.attrs["group_info"]) == {"All", *[g for g in cli.Group if g != "All"]}
          and len(set(cli.Group)) == 3 and os.path.exists(os.path.join(tmp, "e.xlsx")))


def test_adjusted_cis_for_every_metric():
    s = {"bootstrap_iterations": 999, "verbose": False}
    bd = binary_data()                                            # 3 models, 3 pairs
    w, _, _ = osx.compare_models(bd, task="binary", settings=s)
    q3 = 1 - 0.05 / 3
    w3, _, _ = osx.compare_models(bd, task="binary", settings={**s, "ci_level": q3})
    ok_present, ok_wider, ok_bonf = True, True, True
    for m in ("AUC", "Sensitivity", "Specificity", "Accuracy"):
        lo, hi = w.get(f"{m} Diff 95% CI adj low"), w.get(f"{m} Diff 95% CI adj high")
        ok_present &= lo is not None and hi is not None
        if lo is None:
            continue
        width, width0 = hi - lo, w[f"{m} Diff 95% CI high"] - w[f"{m} Diff 95% CI low"]
        ok_wider &= bool((width >= width0 - 1e-12).all() and (width > width0).any())
        ok_bonf &= np.allclose(lo, w3[f"{m} Diff {q3 * 100:g}% CI low"], rtol=1e-9, atol=1e-12) and \
            np.allclose(hi, w3[f"{m} Diff {q3 * 100:g}% CI high"], rtol=1e-9, atol=1e-12)
    check("adjusted CIs for AUC, sensitivity, specificity and accuracy", ok_present)
    check("with 3 models they are wider than the unadjusted CIs", ok_wider)
    check("they are Bonferroni intervals: equal to the unadjusted CIs at level 1 - 0.05/3", ok_bonf)
    ratio = (w["AUC Diff 95% CI adj high"] - w["AUC Diff 95% CI adj low"]) / (w["AUC Diff 95% CI high"] - w["AUC Diff 95% CI low"])
    check("AUC: difference ± z(1 - α/2m) × SE", np.allclose(ratio, osx.stats.norm.ppf(1 - 0.05 / 6) / osx.stats.norm.ppf(0.975)))
    df = iol_like_data(seed=5).drop(columns=["eye"])              # 4 models, 6 pairs
    q6 = 1 - 0.05 / 6
    ok = True
    for weighting in ("eye", "patient"):
        st = {**s, "weighting": weighting}
        wc, _, _ = osx.compare_models(df, task="continuous", settings=st)
        wc6, _, _ = osx.compare_models(df, task="continuous", settings={**st, "ci_level": q6})
        for m in ("RMSE", "SD", "McNemar"):
            lo, hi = wc[f"{m} Diff 95% CI adj low"], wc[f"{m} Diff 95% CI adj high"]
            ok &= bool(((hi - lo) > (wc[f"{m} Diff 95% CI high"] - wc[f"{m} Diff 95% CI low"])).all())
            ok &= np.allclose(lo, wc6[f"{m} Diff {q6 * 100:g}% CI low"], rtol=1e-9, atol=1e-12) and \
                np.allclose(hi, wc6[f"{m} Diff {q6 * 100:g}% CI high"], rtol=1e-9, atol=1e-12)
    check("RMSE, SD and proportion within: Bonferroni percentile intervals, wider than unadjusted (eye and patient weights)", ok)
    w1, _, _ = osx.compare_models(bd[["patient_id", "truth", "M0", "M1"]], task="binary", settings=s)
    wc1, _, _ = osx.compare_models(df[["patient_id", "F0", "F1"]], task="continuous", settings=s)
    same = all(np.allclose(ww[f"{m} Diff 95% CI adj {e}"], ww[f"{m} Diff 95% CI {e}"]) for ww, ms in
               ((w1, ("AUC", "Sensitivity", "Specificity", "Accuracy")), (wc1, ("RMSE", "SD", "McNemar")))
               for m in ms for e in ("low", "high"))
    check("single pair: the new adjusted CIs equal the unadjusted ones", same)


def test_adjustment_method_columns():
    s = {"bootstrap_iterations": 499, "verbose": False}
    df = iol_like_data(seed=5).drop(columns=["eye"])
    _, lc, info = osx.compare_models(df, task="continuous", settings=s)
    pm = lc.drop_duplicates("metric").set_index("metric")
    rw, bt, pct = "Romano-Wolf step-down", "Romano-Wolf simultaneous (max-|t|)", "Bonferroni (percentile bootstrap)"
    check("continuous, Romano-Wolf: p_adj_method and ci_adj_method name the adjustment of every row",
          lc.p_adj_method.notna().all() and lc.ci_adj_method.notna().all()
          and all(pm.p_adj_method[m] == rw for m in ("bias", "mae", "rmse", "medae", "sd"))
          and pm.p_adj_method["within"] == "Hommel"
          and all(pm.ci_adj_method[m] == bt for m in ("bias", "mae", "medae"))
          and all(pm.ci_adj_method[m] == pct for m in ("rmse", "sd", "within")))
    _, lb, _ = osx.compare_models(binary_data(), task="binary", settings=s)
    pb = lb.drop_duplicates("metric").set_index("metric")
    check("binary, Romano-Wolf: AUC Hommel / Bonferroni (normal theory), proportions Hommel / Bonferroni percentile",
          pb.p_adj_method["auc"] == "Hommel" and pb.ci_adj_method["auc"].startswith("Bonferroni (normal theory")
          and all(pb.p_adj_method[m] == "Hommel" and pb.ci_adj_method[m] == pct for m in ("sensitivity", "specificity", "accuracy"))
          and all(pb.p_adj_method[m] == rw and pb.ci_adj_method[m] == bt for m in ("brier", "logloss")))
    _, lh, _ = osx.compare_models(df, task="continuous", settings={**s, "fwer_method": "hommel"})
    _, lf, _ = osx.compare_models(df, task="continuous", settings={**s, "fwer_method": "fdr_bh"})
    _, ly, _ = osx.compare_models(df, task="continuous", settings={**s, "fwer_method": "fdr_by"})
    _, l0, _ = osx.compare_models(df, task="continuous", settings={**s, "report_ci": False})
    check("other methods are labeled: Hommel / Bonferroni, BH / Benjamini-Yekutieli, fdr_by and report_ci off not available",
          (lh.p_adj_method == "Hommel").all() and lh.ci_adj_method.str.startswith("Bonferroni").all()
          and (lf.p_adj_method == "Benjamini-Hochberg (FDR)").all()
          and lf.ci_adj_method.str.startswith("Benjamini-Yekutieli").all()
          and ly.ci_adj_method.str.startswith("not available").all()
          and l0.ci_adj_method.str.startswith("not available").all())
    _, l1, i1 = osx.compare_models(df[["patient_id", "F0", "F1"]], task="continuous", settings=s)
    check("single pair: adjusted equals unadjusted, and the output says so",
          (l1.p_adj_method == osx.SINGLE_P).all() and (l1.ci_adj_method == osx.SINGLE_CI).all()
          and np.allclose(l1.p_adj, l1.p) and "single model pair" in i1["methodology"]["Adjusted p-values (p_adj)"].lower())
    res = osx.analyze(binary_data(), task="binary", subgroup="eye", settings=s)
    path = os.path.join(tempfile.mkdtemp(), "r.xlsx")
    osx.write_results_to_excel(res, path)
    est = pd.read_excel(path, sheet_name="Estimates")
    tests = pd.read_excel(path, sheet_name="Statistical Tests").set_index("Test Category")["Methodology Applied"]
    sample = pd.read_excel(path, sheet_name="Sample")
    check("Excel: method columns in Estimates, adjustments summarized in Statistical Tests, definitions in Sample",
          {"p_adj_method", "ci_adj_method", "ci_adj_low", "ci_adj_high"} <= set(est.columns)
          and est.ci_adj_low[est.metric == "auc"].notna().all()
          and "Hommel: AUC" in tests["Adjusted p-values (p_adj)"] and "AUC" in tests["Adjusted Confidence Intervals"]
          and "Subgroups" in tests.index and sample.definition.str.contains("eyes").all())


def test_several_and_crossed_subgroups():
    s = {"bootstrap_iterations": 199, "verbose": False}
    df = with_axial_length()
    rng = np.random.default_rng(11)
    pid = df.patient_id.str[1:].astype(int).values
    df["mean_k"] = np.round(43.6 - 0.9 * (df.axial_length_mm - 23.5) + rng.normal(0, 1, pid.max() + 1)[pid], 2)
    models = osx.analyze(df[["patient_id", "F0", "F1", "F2", "F3", "mean_k"]], settings=s).attrs["group_info"]["All"]["models"]
    check("a mean K column is not taken as a model", "mean_k" not in models)
    forms = [osx.parse_subgroup_spec(x) for x in ("mean_k:bins=5", ("mean_k", {"bins": 5}), {"column": "mean_k", "bins": 5})]
    check("subgroup definitions: string, pair and dict forms agree", all(f == forms[0] for f in forms)
          and osx.parse_subgroup_spec("axial_length_mm:cuts=24.5,22") ==
          {"column": "axial_length_mm", "cuts": [22.0, 24.5], "bins": None})
    bad = 0
    for x in ("mean_k:bins=1", "mean_k:bins=2.5", {"column": "mean_k", "bins": 3, "cuts": [43]}, {"bins": 3}):
        try:
            osx.parse_subgroup_spec(x)
        except ValueError:
            bad += 1
    check("invalid subgroup definitions are rejected", bad == 4)
    res = osx.analyze(df, subgroup=["axial_length_mm:cuts=22,24.5", "mean_k:bins=5", "site"], settings=s)
    gi = res.attrs["group_info"]
    by = pd.Series({g: v["grouping"] for g, v in gi.items()})
    check("several subgroups: each analyzed separately, with its own split",
          (by == "axial_length_mm").sum() == 3 and (by == "mean_k").sum() == 5 and (by == "site").sum() == 3)
    one = osx.analyze(df, subgroup="axial_length_mm", settings={**s, "subgroup_cuts": [22, 24.5]})
    a = res[res.Group.isin(one.Group)].reset_index(drop=True)
    check("a subgroup's results do not depend on the other subgroups given", _same_frames(a, one.reset_index(drop=True)))
    try:
        osx.analyze(df, subgroup=["axial_length_mm", "mean_k"], settings={**s, "subgroup_cuts": [22]})
        ok = False
    except ValueError:
        ok = True
    check("settings subgroup_cuts with several subgroups is rejected", ok)
    cr = osx.analyze(df, cross=["axial_length_mm:cuts=23,24.5", "mean_k:bins=3"], settings=s)
    grid = cr.attrs["grid"]
    cells = grid["cells"]
    check("crossed subgroups: one cell per row and column", len(cells) == 9 and len(grid["rows"]) == 3
          and len(grid["cols"]) == 3 and sum(c["n_eyes"] for c in cells.values()) == len(df))
    small = [g for g, c in cells.items() if c["n_eyes"] >= 5 and c["n_patients"] < 20]
    big = [g for g, c in cells.items() if c["n_patients"] >= 20]
    est = pd.DataFrame(cr.attrs["long"])
    check("small cells: estimates kept, intervals and tests removed",
          len(small) == 0 or (est[est.group.isin(small)].p.isna().all() and est[est.group.isin(small)]["diff"].notna().all()))
    g = big[0]
    edges_r = [23.0, 24.5]
    q = np.quantile(df.mean_k, [1 / 3, 2 / 3])
    row_i = grid["rows"].index(cells[g]["row"])
    col_i = grid["cols"].index(cells[g]["col"])
    mask = ((np.searchsorted(edges_r, df.axial_length_mm, side="right") == row_i)
            & (np.searchsorted(q, df.mean_k, side="right") == col_i))
    w, _, _ = osx.compare_models(df[mask], task="continuous", models=models, settings=s, group_name=g)
    check("a tested cell equals compare_models on its eyes (cut points from all eyes)",
          _same_frames(cr[cr.Group == g].reset_index(drop=True), w.reset_index(drop=True)))
    path = os.path.join(tempfile.mkdtemp(), "g.xlsx")
    osx.write_results_to_excel(cr, path)
    x = pd.ExcelFile(path)
    tests = pd.read_excel(path, sheet_name="Statistical Tests").set_index("Test Category")["Methodology Applied"]
    check("Excel: Grid sheet, grouping column, crossed-subgroup methods described",
          "Grid" in x.sheet_names and "grouping" in pd.read_excel(path, sheet_name="Estimates").columns
          and "Crossed subgroups" in tests.index and "exploratory" in tests["Subgroup adjustment"])
    tmp = tempfile.mkdtemp()
    df.to_csv(os.path.join(tmp, "k.csv"), index=False)
    cli = osx._command_line([os.path.join(tmp, "k.csv"), "--subgroup", "axial_length_mm:cuts=22,24.5", "--subgroup",
                             "mean_k:bins=4", "--cross", "axial_length_mm:cuts=23", "mean_k:bins=2",
                             "--out", os.path.join(tmp, "k.xlsx")])
    gb = pd.Series({k: v["grouping"] for k, v in cli.attrs["group_info"].items()})
    check("command line: repeated --subgroup with inline cuts and bins, and --cross",
          (gb == "axial_length_mm").sum() == 3 and (gb == "mean_k").sum() == 4
          and (gb == "axial_length_mm × mean_k").sum() == 4)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()):
                osx._command_line([os.path.join(tmp, "k.csv"), "--subgroup", "axial_length_mm", "--subgroup", "mean_k",
                                   "--cuts", "22", "--out", os.path.join(tmp, "k2.xlsx")])
        ok = False
    except SystemExit:
        ok = True
    check("command line: --cuts with several subgroups is refused", ok)


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
