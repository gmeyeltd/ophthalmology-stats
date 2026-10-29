"""
ophthalmology_stats: compare the accuracy of prediction models evaluated on eyes (artificial intelligence models,
formulas, devices), with one or both eyes per patient and the patient as the unit of analysis.

Input: one row per eye, with a patient ID column (eyes sharing an ID are resampled together) and
  continuous outputs  one column of prediction errors (prediction minus reference) per model. This is the primary
                      input. Alternatively, one column of predictions per model plus the reference value
                      (truth=..., --truth); the errors are then prediction - truth.
  classifiers         one column of predicted probabilities per model plus the 0/1 diagnosis (truth=..., --truth).
                      The diagnosis is required: AUC, sensitivity and specificity are defined by it.
With task='auto' (the default of analyze and the command line), a column with a common reference name (truth,
reference, diagnosis, measured, ...; see TRUTH_VARIANTS) is used as the reference when truth is not given; without
one, the model columns are taken as prediction errors. Other columns (eye, age, ...) are ignored unless named as a
subgroup, which can be nominal (e.g. eye), ordinal, or continuous (e.g. axial length, split at cut points or into
equal-count quantile groups).

It extends biometry_stats_advanced (biometry-stats, for intraocular lens formulas), which is included unchanged
in the first part of this file, so analyze_prediction_errors gives exactly the same results as before. The second
part generalizes it to any model and to classifiers:

  output       metric                                   test (both eyes, patient as the unit)
  -----------  ---------------------------------------  ------------------------------------------------------
  continuous   mean error (MPE/bias), MAE, RMSE          patient bootstrap-t with CR1 standard error (RMSE via MSE)
               median absolute error (MedAE)             percentile patient bootstrap (ties counted in both tails)
               SD of the error                           Morgan-Pitman, wild cluster bootstrap (Rademacher, CR0)
               proportion within a tolerance             clustered McNemar (Yang, Sun & Hardin 2010)
  binary       Brier score, log-loss                     patient bootstrap-t with CR1 standard error
               AUC                                       Obuchowski (1997) clustered ROC; DeLong (1988) with one eye
               sensitivity, specificity, accuracy        clustered McNemar (Yang, Sun & Hardin 2010)

Every difference comes with a 95% confidence interval, a p-value, an adjusted p-value (Romano-Wolf step-down
for the bootstrap tests, Hommel otherwise) and an adjusted 95% interval. The bootstrap-t and MedAE intervals match
the chosen method (max-|t| simultaneous intervals for Romano-Wolf, Bonferroni for Bonferroni/Holm/Hommel,
false-coverage-rate intervals for Benjamini-Hochberg); the AUC (normal theory, difference ± z(1 - α/2m) × SE) and
the RMSE, SD and proportion intervals (percentile bootstrap at level 1 - α/m) get Bonferroni intervals, m being the
number of model pairs. The columns p_adj_method and ci_adj_method say which adjustment each row got. For each
metric the output also gives ρΔ (column rho_delta), the fellow-eye correlation of the per-eye difference in loss
between the two models, and the one-eye factor: how many times more patients a study keeping one eye per patient
would need for the same precision (design effect for unequal cluster sizes; Kerry & Bland 2001). Mean-loss metrics
can weight every eye equally (default), every patient equally, or use minimum-variance weights.

Quick start
    import pandas as pd, ophthalmology_stats as osx
    # continuous output: patient_id plus one column of prediction errors per model
    res = osx.analyze(pd.read_csv("errors.csv"), subgroup="axial_length", settings={"subgroup_cuts": [22, 24.5]})
    # classifier: patient_id, the 0/1 diagnosis and one column of probabilities per model
    res = osx.analyze(pd.read_csv("probabilities.csv"), task="binary", truth="diagnosis", subgroup="eye")
    osx.write_results_to_excel(res, "results.xlsx")
or from the command line:
    python ophthalmology_stats.py errors.csv --subgroup axial_length --cuts 22 24.5
    python ophthalmology_stats.py predictions.csv --truth measured            (predictions plus the reference)
    python ophthalmology_stats.py probabilities.csv --task binary --truth diagnosis --subgroup eye

https://github.com/gmeyeltd/ophthalmology-stats (MIT license)
"""
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
from concurrent.futures import ThreadPoolExecutor

# Defaults. Most can be overridden through the settings dict passed to analyze_prediction_errors.
BOOTSTRAP_ITERATIONS = 4999   # with p = (1 + b)/(B + 1), p-values fall on a grid of 1/5000

# Fixed seed so results are reproducible; set to None for a random seed
BOOTSTRAP_SEED = 42

if BOOTSTRAP_SEED is None:
    BOOTSTRAP_SEED = int(np.random.default_rng().integers(0, 2**32))
    print(f"--- [Biometry Stats] Random Bootstrap Seed Generated: {BOOTSTRAP_SEED} ---")
else:
    print(f"--- [Biometry Stats] Using Fixed Bootstrap Seed: {BOOTSTRAP_SEED} ---")

# Used when the settings dict does not supply a value
YUEN_TRIM_PROPORTION = 0.2
MCNEMAR_THRESHOLD = 0.50
MULTIPLE_TEST_METHOD = 'romano_wolf'
FALLBACK_TEST_METHOD = 'hommel'
INVALID_THRESHOLD = 4.0
AXL_SHORT_CUTOFF = 22.0
AXL_LONG_CUTOFF = 26.0
MIN_GROUP_SIZE = 5      # a group (All, Short, Medium, Long) with fewer valid eyes than this is not tested
MEDIAN_WINDOW = 5       # bootstrap medians: half-width of the search window, in sqrt(n) eyes (speed only; results are exact)
RUN_GROUPS_IN_PARALLEL = True   # test All / Short / Medium / Long side by side (speed only; results are identical)

# Which tests to run
USE_STUDENTIZED_BS = True
ENFORCE_INDEPENDENCE = False
RUN_YUEN_TEST = False
RUN_SD_TEST = True
REPORT_CI = False    # report each difference between formulas with a bootstrap confidence interval (unadjusted and adjusted)
CI_LEVEL = 0.95

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def _calculate_percentile_bootstrap_p(boot_diffs):
    """Two-sided percentile bootstrap p-value (MedAE, or any metric when studentising is off)."""
    valid_diffs = boot_diffs[~np.isnan(boot_diffs)]
    if len(valid_diffs) == 0:
        return 1.0
    # ties (difference exactly 0) count in both tails, as in _bootstrap_p and the percentile CI;
    # leaving them out made p too small when medians tie (rounded errors, or identical formulas)
    tail = min(np.sum(valid_diffs >= 0), np.sum(valid_diffs <= 0))
    p = 2 * (1 + tail) / (len(valid_diffs) + 1)
    return min(p, 1.0)

def _bootstrap_p(null_stats, obs_stat):
    """Bootstrap p-value (1 + #{null >= obs}) / (B + 1); never exactly 0 (Davison & Hinkley 1997)."""
    return (1 + np.sum(null_stats >= obs_stat)) / (len(null_stats) + 1)

def mcnemar_test(mask1, mask2):
    """McNemar test with continuity correction (one eye per patient)."""
    b = np.sum(mask1 & ~mask2)
    c = np.sum(~mask1 & mask2)
    n = b + c
    if b == c: return 1.0
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    p_val = stats.chi2.sf(chi2_stat, 1)
    return min(1.0, float(p_val))

def clustered_mcnemar_test(mask1, mask2, cluster_ids):
    """Yang, Sun & Hardin (2010) modified Obuchowski test for bilateral (clustered) data.

    Per cluster k: D_k = b_k - c_k (discordant-pair difference), n_k eyes.
    The variance averages the centred (Obuchowski) and null-centred (D_k^2) forms:
        X2 = ((K-1)/K) * (sum D_k)^2 / (0.5 * sum[(D_k - n_k*(p1 - p2))^2 + D_k^2])
    referred to chi-square with 1 df (no continuity correction).
    """
    d = mask1.astype(int) - mask2.astype(int)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    if K < 2: return 1.0

    D_k = np.bincount(cluster_idx, weights=d)
    n_k = np.bincount(cluster_idx)
    sum_D = np.sum(D_k)
    if sum_D == 0: return 1.0

    p_diff = sum_D / np.sum(n_k)
    denom = 0.5 * np.sum((D_k - n_k * p_diff)**2 + D_k**2)
    if denom == 0: return 1.0

    chi2_stat = ((K - 1) / K) * sum_D**2 / denom
    return min(1.0, float(stats.chi2.sf(chi2_stat, 1)))

def wild_cluster_sd_all_pairs(X, cluster_idx, K, W_g, pairs=None, chunk=500):
    """Morgan-Pitman test of equal SDs with a wild cluster bootstrap-t, for every formula pair at once.

    For each pair, the standardized difference and sum of the two prediction errors are regressed on each other;
    the slope is tested with a wild cluster bootstrap restricted to the null (Cameron, Gelbach & Miller 2008),
    Rademacher weights W_g (B x K) and a CR0 standard error. With weights of +/-1, every bootstrap slope and
    standard error is a function of cluster sums, so all resamples and all pairs come from one matrix product.
    Returns {(j, k): (p, |t_obs|, |t*|)}.
    """
    X = np.asarray(X, float)
    n, J = X.shape
    B = W_g.shape[0]
    pairs = pairs or list(combinations(range(J), 2))
    out, cols, meta = {}, [], {}
    for (j, k) in pairs:
        diff, summ = X[:, j] - X[:, k], X[:, j] + X[:, k]
        var_d, var_s = np.var(diff, ddof=1), np.var(summ, ddof=1)
        if var_d == 0 or var_s == 0 or K < 3:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        x = (diff - np.mean(diff)) / np.sqrt(var_d)
        y = (summ - np.mean(summ)) / np.sqrt(var_s)
        x_bar, y_bar = np.mean(x), np.mean(y)
        dx = x - x_bar
        sxx = np.sum(dx ** 2)
        if sxx == 0:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        beta_hat = np.sum(dx * y) / sxx
        eps_hat = y - (y_bar + beta_hat * dx)
        var_b = np.sum(np.bincount(cluster_idx, weights=dx * eps_hat, minlength=K) ** 2) / sxx ** 2
        if var_b == 0:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        t_obs = beta_hat / np.sqrt(var_b)
        e_t = y - y_bar                                                     # null-restricted residuals
        A = np.bincount(cluster_idx, weights=dx * e_t, minlength=K)        # per-cluster sums
        E = np.bincount(cluster_idx, weights=e_t, minlength=K)
        D = np.bincount(cluster_idx, weights=dx, minlength=K)
        Q = np.bincount(cluster_idx, weights=dx ** 2, minlength=K)
        meta[(j, k)] = dict(c=len(cols), t_obs=t_obs, y_bar=y_bar, sxx=sxx, sdx=np.sum(dx),
                            DD=np.sum(D * D), AA=np.sum(A * A), QQ=np.sum(Q * Q), DQ=np.sum(D * Q))
        cols += [A, E, D * A, A * Q]
    if cols:
        V = np.column_stack(cols)
        G = np.empty((B, V.shape[1]))
        for i in range(0, B, chunk):
            G[i:i + chunk] = W_g[i:i + chunk].astype(float) @ V
        for key, m in meta.items():
            c = m["c"]
            wA, wE, wDA, wAQ = G[:, c], G[:, c + 1], G[:, c + 2], G[:, c + 3]
            beta_s = (m["y_bar"] * m["sdx"] + wA) / m["sxx"]                # bootstrap slope
            a = -wE / n                                                      # y_bar - bootstrap mean of y
            ss = (a ** 2 * m["DD"] + m["AA"] + beta_s ** 2 * m["QQ"]
                  + 2 * a * wDA - 2 * a * beta_s * m["DQ"] - 2 * beta_s * wAQ)  # sum of squared cluster scores
            var_s = np.maximum(ss, 0) / m["sxx"] ** 2
            t_s = np.zeros(B); ok = var_s > 0
            t_s[ok] = beta_s[ok] / np.sqrt(var_s[ok])
            out[key] = (float(_bootstrap_p(np.abs(t_s), np.abs(m["t_obs"]))), np.abs(m["t_obs"]), np.abs(t_s))
    return out

def comdvar_wild_cluster(x, y, cluster_idx, K, W_g):
    """Morgan-Pitman test of equal SDs with a wild cluster bootstrap p-value (one pair)."""
    return wild_cluster_sd_all_pairs(np.column_stack((x, y)), cluster_idx, K, W_g)[(0, 1)]

def olshc4(x, y):
    """OLS slope test with HC4 standard errors. Adapted from Wilcox's olshc4 (Rallfun v45)."""
    n = len(y)
    if n < 3: return 1.0

    X = np.column_stack((np.ones(n), x))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 1.0

    h = np.sum((X @ XtX_inv) * X, axis=1)
    beta = XtX_inv @ X.T @ y
    res = y - X @ beta

    sum_h = np.sum(h)
    d = (n * h) / sum_h if sum_h != 0 else np.zeros_like(h)
    d = np.minimum(4, d)

    denom = np.power(1 - h, d)
    denom[denom < 1e-10] = 1e-10
    omega = (res**2) / denom

    middle = (X.T * omega) @ X
    Sigma = XtX_inv @ middle @ XtX_inv

    se_slope = np.sqrt(Sigma[1, 1])
    if se_slope == 0: return 1.0

    t_stat = beta[1] / se_slope
    df = n - 2
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

def comdvar(x, y):
    """Morgan-Pitman test of equal SDs with HC4 standard errors (one eye per patient).
    Adapted from Wilcox's comdvar (Rallfun v45)."""
    diff = x - y
    summ = x + y
    var_x, var_y = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_x == 0 or var_y == 0: return 1.0
    z1 = (diff - np.mean(diff)) / np.sqrt(var_x)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_y)
    return olshc4(z1, z2)

def _resample_counts(idx_chunk, K):
    """Count matrix: how often each cluster (patient or eye) appears in each resample of the block."""
    c = idx_chunk.shape[0]
    flat = (idx_chunk + (np.arange(c)[:, None] * K)).ravel()
    return np.bincount(flat, minlength=c * K).reshape(c, K).astype(np.float64)


def all_pairs_bootstrap(X, cluster_idx, K, b_indices, use_studentized=True, pairs=None, chunk=None, ci_level=None, prop_threshold=None):
    """Bootstrap tests for MPE, MAE, RMSE and MedAE for every formula pair at once.

    X is an n x J matrix of prediction errors (one column per formula). Returns {(j, k): result} for j < k,
    each result having the same keys as perform_advanced_bootstrap. Resamples, statistics and p-value rules are
    the same as a pair-by-pair calculation; only the arithmetic is organized differently. Each block of resamples becomes a count matrix W (resamples x clusters), so every
    bootstrap sum the bootstrap-t needs is one matrix product W @ V, and each formula's bootstrap median is read
    from running totals of W over that formula's sorted errors (one median per formula, not per pair).

    If ci_level is given (e.g. 0.95), each result also holds confidence intervals for the difference between the two
    formulas, from the same resamples: bootstrap-t for MPE and MAE (as their tests), percentile for RMSE, MedAE, SD and,
    when prop_threshold is given, the proportion of eyes with |PE| <= prop_threshold. They are not adjusted for
    multiple comparisons, and the p-values are unaffected. Each result also holds the observed difference and its
    scale (RW_*_diff, RW_*_scale), from which adjust_pvalues builds the adjusted (simultaneous) intervals.
    """
    X = np.asarray(X, float)
    n, J = X.shape
    B = b_indices.shape[0]
    pairs = pairs or list(combinations(range(J), 2))
    N_g = np.bincount(cluster_idx, minlength=K).astype(float)
    N_tot = N_g.sum()

    # per-cluster sums for each formula and each family (abs error, squared error, signed error)
    fam = {"MAE": np.abs(X), "MSE": X ** 2, "MPE": X}
    S = {f: np.stack([np.bincount(cluster_idx, weights=v[:, j], minlength=K) for j in range(J)], axis=1) for f, v in fam.items()}
    means = {f: S[f].sum(0) / N_tot for f in fam}                      # observed per-formula means
    C = {f: S[f] - np.outer(N_g, means[f]) for f in fam}               # centred per-cluster sums (better precision)

    # columns of V: for each family, centred sums (J), and cross-products for every pair incl. squares
    tri = [(a, b) for a in range(J) for b in range(a, J)]
    cols, index = [N_g], {}
    for f in fam:
        index[(f, "s")] = len(cols); cols += [C[f][:, j] for j in range(J)]
        index[(f, "x")] = len(cols); cols += [C[f][:, a] * C[f][:, b] for a, b in tri]
        index[(f, "n")] = len(cols); cols += [C[f][:, j] * N_g for j in range(J)]
    index["nn"] = len(cols); cols.append(N_g ** 2)
    index[("MSE", "raw")] = len(cols); cols += [S["MSE"][:, j] for j in range(J)]   # for RMSE differences
    if ci_level:
        index["SQ"] = len(cols)                                                    # sums of squared deviations (SD)
        cols += [np.bincount(cluster_idx, weights=(X[:, j] - means["MPE"][j]) ** 2, minlength=K) for j in range(J)]
        if prop_threshold is not None:
            index["PROP"] = len(cols)                                              # eyes with |PE| <= threshold
            cols += [np.bincount(cluster_idx, weights=(np.abs(X[:, j]) <= prop_threshold).astype(float), minlength=K) for j in range(J)]
    V = np.column_stack(cols)
    tpos = {t: i for i, t in enumerate(tri)}

    # observed statistics, computed exactly as the original engine does
    def cr1_se(Svec, mean_val):
        res = Svec - mean_val * N_g
        return np.sqrt(max(0.0, (K / (K - 1)) * np.sum(res ** 2) / N_tot ** 2)) if K > 1 else 0.0

    # medians: one per formula per resample, read from running totals of the counts over each formula's sorted values
    absX = np.abs(X)
    order = [np.argsort(absX[:, j], kind="stable") for j in range(J)]
    sorted_vals = [absX[order[j], j] for j in range(J)]
    eye_cluster = [cluster_idx[order[j]] for j in range(J)]
    max_c = int(N_g.max())

    # A resample's median almost always lies close to the middle of the original sorted errors, so the running
    # totals are only needed over a window of sorted positions around the middle (+/- 5 sqrt(n) + 16 eyes): the eyes
    # drawn below the window come from one product with per-cluster counts. A resample whose middle rank falls outside
    # the window is redone over all the values, so every median is exactly the one the full calculation gives.
    n_eyes = len(cluster_idx)
    half = int(np.ceil(MEDIAN_WINDOW * np.sqrt(n_eyes))) + 16
    w_start, w_end = max(0, n_eyes // 2 - half), min(n_eyes, n_eyes // 2 + half + 1)
    windowed = (w_end - w_start) < n_eyes
    if windowed:
        below = [np.bincount(eye_cluster[j][:w_start], minlength=K).astype(float) for j in range(J)]
        in_window = [eye_cluster[j][w_start:w_end] for j in range(J)]

    def _positions(Wj, lo, hi, base=None):
        """Sorted positions of the lo-th and hi-th drawn eye (running totals over the columns of Wj)."""
        c = Wj.shape[0]
        cum = np.cumsum(Wj, axis=1)                                      # eyes drawn up to each sorted value
        if base is not None:
            cum += base[:, None]
        big = int(max(cum[:, -1].max(), hi.max())) + 1
        offs = np.arange(c) * big
        flat = (cum + offs[:, None]).ravel()
        start = np.arange(c) * cum.shape[1]
        pos_lo = np.searchsorted(flat, lo + offs, side="left") - start
        pos_hi = np.searchsorted(flat, hi + offs, side="left") - start
        return pos_lo, pos_hi

    def _medians(W):
        c = W.shape[0]
        out = np.empty((c, J))
        nb = W @ N_g                                                     # eyes in each resample
        lo = np.floor((nb + 1) / 2).astype(np.int64)                     # ranks of the middle value(s), 1-based
        hi = np.floor(nb / 2).astype(np.int64) + 1
        for j in range(J):
            if windowed:
                base = W @ below[j]                                      # eyes drawn below the window
                Wj = W[:, in_window[j]]
                pos_lo, pos_hi = _positions(Wj, lo, hi, base)
                inside = (base < lo) & (base + Wj.sum(axis=1) >= hi)
                pos_lo = np.clip(pos_lo, 0, w_end - w_start - 1) + w_start
                pos_hi = np.clip(pos_hi, 0, w_end - w_start - 1) + w_start
                if not inside.all():                                     # rare: redo these resamples in full
                    redo = ~inside
                    pos_lo[redo], pos_hi[redo] = _positions(W[redo][:, eye_cluster[j]], lo[redo], hi[redo])
            else:
                pos_lo, pos_hi = _positions(W[:, eye_cluster[j]], lo, hi)
            out[:, j] = (sorted_vals[j][pos_lo] + sorted_vals[j][pos_hi]) / 2
        return out

    chunk = chunk or max(1, min(500, int(2e7 // max(1, int(N_tot)))))
    sums = np.empty((B, V.shape[1]))
    med = np.empty((B, J))
    for i in range(0, B, chunk):
        idx = b_indices[i:i + chunk]
        W = _resample_counts(idx, K)
        sums[i:i + chunk] = W @ V
        med[i:i + chunk] = _medians(W)

    bN = sums[:, 0]
    a_ci = (1 - ci_level) / 2 if ci_level else None
    qci = lambda v: tuple(float(x) for x in np.quantile(v, [a_ci, 1 - a_ci]))
    out = {}
    for (j, k) in pairs:
        r = {}
        for f, key in (("MPE", "MPE"), ("MAE", "MAE"), ("MSE", "RMSE")):
            d_mean_obs = means[f][j] - means[f][k]
            Sd = S[f][:, j] - S[f][:, k]
            se_obs = cr1_se(Sd, d_mean_obs)
            t_obs = d_mean_obs / se_obs if se_obs > 0 else 0.0
            # bootstrap: centred difference sums and CR1 variance from the matrix products
            s = index[(f, "s")]; x = index[(f, "x")]; nn_ = index[(f, "n")]
            sd = sums[:, s + j] - sums[:, s + k]                                    # sum_g w (Cj - Ck)
            sdd = sums[:, x + tpos[(j, j)]] + sums[:, x + tpos[(k, k)]] - 2 * sums[:, x + tpos[(min(j, k), max(j, k))]]
            sdn = sums[:, nn_ + j] - sums[:, nn_ + k]
            snn = sums[:, index["nn"]]
            m = sd / bN                                                              # boot mean minus observed mean
            ssq = sdd - 2 * m * sdn + m ** 2 * snn                                   # sum_g w (S'_g - m N_g)^2
            var = (K / (K - 1)) * np.maximum(ssq, 0) / bN ** 2 if K > 1 else np.zeros(B)
            se = np.sqrt(var)
            if ci_level:
                if key == "RMSE":                                                    # percentile, on the RMSE scale
                    q = index[("MSE", "raw")]
                    r[f"CI_{key}"] = qci(np.sqrt(np.maximum(0, sums[:, q + j] / bN)) - np.sqrt(np.maximum(0, sums[:, q + k] / bN)))
                elif use_studentized:                                                # bootstrap-t
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ts = np.where(se > 0, m / se, 0.0)
                    lo_q, hi_q = qci(ts)
                    r[f"CI_{key}"] = (d_mean_obs - hi_q * se_obs, d_mean_obs - lo_q * se_obs)
                else:
                    r[f"CI_{key}"] = qci(m + d_mean_obs)
            if use_studentized:
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_star = np.where(se > 0, m / se, 0.0)
                r[f"P_{key}"] = _bootstrap_p(np.abs(t_star), np.abs(t_obs))
                r[f"RW_{key}_obs"] = np.abs(t_obs); r[f"RW_{key}_null"] = np.abs(t_star)
                # RMSE is tested on the MSE scale, so its statistic gives no interval for the RMSE difference
                r[f"RW_{key}_diff"] = np.nan if key == "RMSE" else d_mean_obs
                r[f"RW_{key}_scale"] = np.nan if key == "RMSE" else se_obs
            else:
                if key == "RMSE":
                    q = index[("MSE", "raw")]
                    boot = np.sqrt(np.maximum(0, sums[:, q + j] / bN)) - np.sqrt(np.maximum(0, sums[:, q + k] / bN))
                    obs = np.sqrt(means["MSE"][j]) - np.sqrt(means["MSE"][k])
                else:
                    boot, obs = m + d_mean_obs, d_mean_obs
                r[f"P_{key}"] = _calculate_percentile_bootstrap_p(boot)
                r[f"RW_{key}_obs"] = abs(obs); r[f"RW_{key}_null"] = np.abs(boot - obs)
                r[f"RW_{key}_diff"] = obs; r[f"RW_{key}_scale"] = 1.0
        boot_medae = med[:, j] - med[:, k]
        obs_medae = np.median(absX[:, j]) - np.median(absX[:, k])
        r["P_MedAE"] = _calculate_percentile_bootstrap_p(boot_medae)
        if ci_level:
            r["CI_MedAE"] = qci(boot_medae)
            s0, sq = index[("MPE", "s")], index["SQ"]
            with np.errstate(divide="ignore", invalid="ignore"):
                sd_j = np.sqrt(np.maximum(0, (sums[:, sq + j] - sums[:, s0 + j] ** 2 / bN) / (bN - 1)))
                sd_k = np.sqrt(np.maximum(0, (sums[:, sq + k] - sums[:, s0 + k] ** 2 / bN) / (bN - 1)))
            r["CI_SD"] = qci(sd_j - sd_k)
            if prop_threshold is not None:
                pc = index["PROP"]
                r["CI_PROP"] = qci((sums[:, pc + j] - sums[:, pc + k]) / bN)
        if use_studentized:
            se_m = max(np.std(boot_medae, ddof=1), 1e-6)
            r["RW_MedAE_obs"] = abs(obs_medae) / se_m; r["RW_MedAE_null"] = np.abs(boot_medae - obs_medae) / se_m
            r["RW_MedAE_scale"] = se_m
        else:
            r["RW_MedAE_obs"] = abs(obs_medae); r["RW_MedAE_null"] = np.abs(boot_medae - obs_medae)
            r["RW_MedAE_scale"] = 1.0
        r["RW_MedAE_diff"] = obs_medae
        out[(j, k)] = r
    return out


def perform_advanced_bootstrap(e1, e2, cluster_idx, K, b_indices, W_g=None, use_studentized=True):
    """Patient (cluster) bootstrap for MPE, MAE, RMSE and MedAE for one formula pair.

    MPE, MAE and RMSE use the bootstrap-t with a CR1 standard error (RMSE is tested through the MSE);
    MedAE uses the percentile bootstrap. Also returns the statistics the Romano-Wolf correction needs.
    """
    return all_pairs_bootstrap(np.column_stack((e1, e2)), cluster_idx, K, b_indices, use_studentized=use_studentized)[(0, 1)]



def yuen_t_test(x, y, tr=0):
    """Yuen's paired test on trimmed means (optional, off by default)."""
    x, y = np.array(x), np.array(y)
    d = x - y
    n = len(d)

    g = int(tr * n)
    d_sort = np.sort(d)
    d_win = d_sort.copy()
    if g > 0:
        d_win[:g] = d_win[g]
        d_win[-g:] = d_win[-g-1]

    win_var = np.var(d_win, ddof=1)
    h = n - 2 * g
    se_tr = np.sqrt((n - 1) * win_var / (h * (h - 1)))

    trimmed_mean_diff = np.mean(d_sort[g:-g]) if g > 0 else np.mean(d)
    t_stat = trimmed_mean_diff / se_tr if se_tr != 0 else 0
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), h - 1)), trimmed_mean_diff

def romano_wolf_correction(obs_stats, boot_null_stats):
    """Romano-Wolf step-down adjusted p-values from the bootstrap null distributions of |t|."""
    M = len(obs_stats)
    if M == 0: return np.array([])

    adj_p = np.zeros(M)
    order = np.argsort(obs_stats)[::-1]
    sorted_nulls = boot_null_stats[:, order]

    max_nulls = np.maximum.accumulate(sorted_nulls[:, ::-1], axis=1)[:, ::-1]

    for i in range(M):
        p_val = _bootstrap_p(max_nulls[:, i], obs_stats[order[i]])
        adj_p[order[i]] = p_val

    for i in range(1, M):
        adj_p[order[i]] = max(adj_p[order[i]], adj_p[order[i-1]])

    return adj_p

# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------


def adjusted_ci_critical(boot_null_stats, level, method, n_reject=None):
    """Critical value for each comparison's adjusted interval (difference +/- c * scale), matching the method chosen
    for the adjusted p-values. boot_null_stats is B x M (bootstrap |t| per comparison, as for Romano-Wolf).

    romano_wolf     one c for all: level-quantile of the largest |t| across comparisons (single-step max-|t|)
    bonferroni      each comparison's own 1 - (1 - level)/M quantile of |t| (agrees with Bonferroni p-values)
    holm, hommel    Bonferroni intervals: these stepwise methods have no compatible finite intervals
    fdr_bh          Benjamini-Yekutieli false-coverage-rate intervals at level 1 - R(1 - level)/M, R = number of
                    comparisons with adjusted p <= 1 - level (R = 0 gives the Bonferroni level)
    Returns an array of M critical values, or None for any other method (no adjusted interval).
    """
    N = np.asarray(boot_null_stats, float)
    M, alpha = N.shape[1], 1 - level
    if method == 'romano_wolf':
        return np.full(M, np.quantile(N.max(axis=1), level))
    if method in ('bonferroni', 'holm', 'hommel'):
        q = 1 - alpha / M
    elif method == 'fdr_bh':
        q = 1 - max(int(n_reject or 0), 1) * alpha / M
    else:
        return None
    return np.quantile(N, q, axis=0)


def adjusted_ci_description(method, level):
    """One-line description of the adjusted intervals for the methodology sheet."""
    L = f"{level * 100:g}%"
    if method == 'romano_wolf':
        return (f"{L} simultaneous intervals: difference \u00b1 c \u00d7 SE, c = {L} quantile of the largest bootstrap |t| across "
                "comparisons (single-step max-|t|, the interval counterpart of Romano-Wolf)")
    if method == 'bonferroni':
        return (f"{L} Bonferroni intervals: difference \u00b1 c \u00d7 SE, c = 1 - {1 - level:g}/m quantile of each comparison's "
                "bootstrap |t| (m comparisons); they agree with the Bonferroni-adjusted p-values")
    if method in ('holm', 'hommel'):
        return (f"{L} Bonferroni intervals (c = 1 - {1 - level:g}/m quantile of each comparison's bootstrap |t|): "
                f"{method.capitalize()} has no compatible finite intervals, so these are conservative and an interval can "
                "include 0 next to a significant adjusted p-value")
    if method == 'fdr_bh':
        return (f"Benjamini-Yekutieli false-coverage-rate intervals at level 1 - R \u00d7 {1 - level:g}/m (R = comparisons "
                f"with adjusted p \u2264 {1 - level:g}); they control the false coverage rate among the significant "
                "comparisons, not simultaneous coverage")
    return "None: adjusted intervals are given for Romano-Wolf, Bonferroni, Holm, Hommel and Benjamini-Hochberg only"

def run_biometry_analysis(df_input, group_name="All", settings=None):
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    bs_iterations = settings.get('bootstrap_iterations', BOOTSTRAP_ITERATIONS)
    invalid_threshold = settings.get('invalid_threshold', INVALID_THRESHOLD)
    mcnemar_threshold = settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD)
    report_ci = settings.get('report_ci', REPORT_CI)
    ci_level = float(settings.get('ci_level', CI_LEVEL)) if report_ci else None

    numeric_df = df_input.select_dtypes(include=[np.number])
    formulas = []

    patient_id_variants = ['patient_id', 'id_patient', 'patientid', 'pt_id', 'id', 'hosp_id', 'hospid', 'ptid']

    # biometry columns that are not prediction errors
    ignore_variants = ['axl', 'al', 'acd', 'acdepth', 'k1', 'k2', 'meank', 'age', 'gender', 'gender_num',
                       'eye', 'side', 'lt', 'lensthickness', 'cct', 'wtw', 'hcd', 'iol_implanted',
                       'postoperative_refraction', 'corneal_refractive_index']

    patient_col = None
    for c in df_input.columns:
        c_lower = str(c).lower()
        if c_lower in patient_id_variants:
            patient_col = c
            continue

        # skip metadata, predicted refraction and lens-constant columns
        if c_lower in ignore_variants or c_lower.startswith('predict_') or c_lower.endswith('_const'):
            continue

        if c in numeric_df.columns:
            formulas.append(c)

    if len(formulas) < 2: return pd.DataFrame(), {}, {}

    valid_mask = pd.Series(True, index=numeric_df.index)
    for f in formulas:
        valid_mask &= ~np.isnan(numeric_df[f])
        valid_mask &= np.abs(numeric_df[f]) <= invalid_threshold

    if patient_col:
        valid_mask &= df_input[patient_col].notna()
        if ENFORCE_INDEPENDENCE:
            rng_indep = np.random.default_rng(BOOTSTRAP_SEED)
            valid_patients = df_input.loc[valid_mask, patient_col]
            shuffled_idx = rng_indep.permutation(valid_patients.index)
            selected_idx = valid_patients.loc[shuffled_idx].drop_duplicates().index
            new_mask = pd.Series(False, index=numeric_df.index)
            new_mask.loc[selected_idx] = True
            valid_mask &= new_mask
            print(f"[{group_name}] ENFORCE_INDEPENDENCE Active: Dropped to 1 eye per patient. Analyzing {len(selected_idx)} independent rows.")
        else: print(f"[{group_name}] Cluster Bootstrap Active: Resampling by '{patient_col}'.")
    else: print(f"[{group_name}] Standard Bootstrap Active: Resampling independently.")

    valid_df = numeric_df[valid_mask]
    n_global = len(valid_df)

    min_group_size = max(2, int(settings.get('min_group_size', MIN_GROUP_SIZE)))
    if n_global < min_group_size:
        print(f"[{group_name}] Not tested: {n_global} valid eye(s), fewer than the minimum of {min_group_size}.")
        return pd.DataFrame(), {}, {}

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # IDs as strings so that mixed types sort
    cluster_ids = df_input.loc[valid_mask, patient_col].astype(str).values if patient_col else np.arange(n_global)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    has_clusters = K < n_global

    # resample indices and wild-bootstrap weights, generated in blocks and stored compactly (int32 / int8);
    # the values are identical to drawing them in one call
    global_b_indices = np.empty((bs_iterations, K), dtype=np.int32)
    for i in range(0, bs_iterations, 500):
        global_b_indices[i:i + 500] = rng.choice(K, size=(min(500, bs_iterations - i), K), replace=True)
    global_W_g = np.empty((bs_iterations, K), dtype=np.int8)
    for i in range(0, bs_iterations, 500):
        global_W_g[i:i + 500] = rng.choice([-1, 1], size=(min(500, bs_iterations - i), K))

    needs_bootstrap_nulls = fwer_method == 'romano_wolf'
    needs_wcb = RUN_SD_TEST and (has_clusters or needs_bootstrap_nulls)

    pairs = list(combinations(formulas, 2))

    bs_type = "Studentized (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized" if USE_STUDENTIZED_BS else "Percentile"
    print(f"[{group_name}] Processing {len(pairs)} comparisons ({bs_type}, N={n_global}, Iterations={bs_iterations})...")

    # bootstrap tests for every formula pair at once (matrix form; identical to pair-by-pair results)
    f_index = {f: i for i, f in enumerate(formulas)}
    boot_all = all_pairs_bootstrap(valid_df[formulas].values, cluster_idx, K, global_b_indices, use_studentized=USE_STUDENTIZED_BS,
                                   ci_level=ci_level, prop_threshold=mcnemar_threshold)
    sd_all = wild_cluster_sd_all_pairs(valid_df[formulas].values, cluster_idx, K, global_W_g) if (RUN_SD_TEST and needs_wcb) else None

    # remaining per-pair tests (SD, McNemar) run in threads
    def process_pair_task(pair):
        f1, f2 = pair
        e1, e2 = valid_df[f1].values, valid_df[f2].values

        boot_res = boot_all[(f_index[f1], f_index[f2])]

        if RUN_SD_TEST:
            if needs_wcb: p_sd, obs_sd, null_sd = sd_all[(f_index[f1], f_index[f2])]
            else: p_sd = comdvar(e1, e2); obs_sd = 0.0; null_sd = np.zeros(bs_iterations)
        else: p_sd = np.nan; obs_sd = 0.0; null_sd = np.zeros(bs_iterations)

        mask1, mask2 = np.abs(e1) <= mcnemar_threshold, np.abs(e2) <= mcnemar_threshold
        if has_clusters: p_mcnemar = clustered_mcnemar_test(mask1, mask2, cluster_ids)
        else: p_mcnemar = mcnemar_test(mask1, mask2)

        if RUN_YUEN_TEST:
            _, diff_mae = yuen_t_test(np.abs(e1), np.abs(e2), tr=0)
            yuen_mae_p, trim_mae_diff = yuen_t_test(np.abs(e1), np.abs(e2), tr=YUEN_TRIM_PROPORTION)
            yuen_rmse_p, trim_sq_diff = yuen_t_test(e1**2, e2**2, tr=YUEN_TRIM_PROPORTION)
        else:
            diff_mae = np.mean(np.abs(e1)) - np.mean(np.abs(e2))

        row = {
            'Group': group_name, 'Formula A': f1, 'Formula B': f2, 'N': n_global,
            'MPE Diff': np.mean(e1) - np.mean(e2), 'Boot (MPE) p': boot_res['P_MPE'],
            'MAE Diff': diff_mae, 'Boot (MAE) p': boot_res['P_MAE'],
            'MedAE Diff': np.median(np.abs(e1)) - np.median(np.abs(e2)), 'Boot (MedAE) p': boot_res['P_MedAE'],
            'RMSE Diff': np.sqrt(np.mean(e1**2)) - np.sqrt(np.mean(e2**2)), 'Boot (RMSE) p': boot_res['P_RMSE'],
            'SD Diff': np.std(e1, ddof=1) - np.std(e2, ddof=1),
            'Morgan-Pitman (SD) p': p_sd,
            'McNemar Diff': np.mean(mask1) - np.mean(mask2),
            'McNemar p': p_mcnemar
        }
        if ci_level:
            # confidence interval columns straight after each difference
            lbl = f"{ci_level * 100:g}% CI"
            ci_key = {'MPE Diff': 'CI_MPE', 'MAE Diff': 'CI_MAE', 'MedAE Diff': 'CI_MedAE', 'RMSE Diff': 'CI_RMSE',
                      'SD Diff': 'CI_SD', 'McNemar Diff': 'CI_PROP'}
            new_row = {}
            for col, val in row.items():
                new_row[col] = val
                if col in ci_key and ci_key[col] in boot_res:
                    new_row[f'{col} {lbl} low'], new_row[f'{col} {lbl} high'] = boot_res[ci_key[col]]
            row = new_row
        if RUN_YUEN_TEST:
            row['Trimmed MAE Diff'] = trim_mae_diff; row['Yuen (MAE) p'] = yuen_mae_p; row['Yuen (RMSE) p'] = yuen_rmse_p

        return {
            'row': row,
            'rw_mpe_obs': boot_res['RW_MPE_obs'], 'rw_mpe_null': boot_res['RW_MPE_null'],
            'rw_mae_obs': boot_res['RW_MAE_obs'], 'rw_mae_null': boot_res['RW_MAE_null'],
            'rw_rmse_obs': boot_res['RW_RMSE_obs'], 'rw_rmse_null': boot_res['RW_RMSE_null'],
            'rw_medae_obs': boot_res['RW_MedAE_obs'], 'rw_medae_null': boot_res['RW_MedAE_null'],
            **{f'rw_{m.lower()}_{part}': boot_res[f'RW_{m}_{part}'] for m in ['MPE', 'MAE', 'RMSE', 'MedAE'] for part in ('diff', 'scale')},
            'sd_obs': obs_sd, 'sd_null': null_sd
        }

    results = []
    rw_stats = {m: {'obs': [], 'null': [], 'diff': [], 'scale': []} for m in ['MPE', 'MAE', 'RMSE', 'MedAE', 'SD']}

    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair_task, pairs):
            results.append(res['row'])
            for m in ['MPE', 'MAE', 'RMSE', 'MedAE']:
                for part in ('obs', 'null', 'diff', 'scale'):
                    rw_stats[m][part].append(res[f'rw_{m.lower()}_{part}'])
            rw_stats['SD']['obs'].append(res['sd_obs'])
            rw_stats['SD']['null'].append(res['sd_null'])

    methodology = {
        'Dependence Strategy': f"Bilateral Clustering (n={bs_iterations} Resamples)" if has_clusters else "Independent Eyes",
        'Precision Test (SD)': "Morgan-Pitman (Wild Cluster Bootstrap)" if needs_wcb else "Morgan-Pitman (HC4 Asymptotic)",
        'Accuracy Test (McNemar)': "Yang's Mod. Obuchowski (Cluster-Robust)" if has_clusters else "Asymptotic McNemar",
        'MPE Test': "Studentized Cluster Bootstrap (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
        'MAE Test': "Studentized Cluster Bootstrap (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
        'MedAE Test': "Cluster Percentile Bootstrap" if has_clusters else "Percentile Bootstrap",
        'RMSE Test': "Studentized Cluster Bootstrap on MSE (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
    }
    if ci_level:
        methodology['Confidence Intervals'] = (f"{ci_level * 100:g}% bootstrap intervals for each difference: bootstrap-t (MPE, MAE), "
                                               "percentile (RMSE, MedAE, SD, proportion); not adjusted for multiple comparisons")
        methodology['Adjusted Confidence Intervals'] = (
            ("Columns 'CI adj', for MPE, MAE and MedAE, adjusted for all comparisons in the group as the p-values are: "
             + adjusted_ci_description(fwer_method, ci_level)
             + ". RMSE is tested on the MSE scale, so it has none under the bootstrap-t")
            if fwer_method in ('romano_wolf', 'bonferroni', 'holm', 'hommel', 'fdr_bh')
            else adjusted_ci_description(fwer_method, ci_level))

    df_results = pd.DataFrame(results)
    # eyes and patients actually tested in this group (used to label figures)
    df_results.attrs['group_info'] = {'n_eyes': int(n_global), 'n_patients': int(K), 'clustered': bool(has_clusters),
                                      'patient_column': patient_col}
    return df_results, rw_stats, methodology

def adjust_pvalues(df_results, rw_stats=None, method='hommel'):
    """Add adjusted p-value columns.

    Romano-Wolf is used where bootstrap null distributions exist; other columns use
    FALLBACK_TEST_METHOD (Hommel by default).
    """
    if df_results.empty: return df_results

    rw_metric_mapping = {
        'Boot (MPE) p': 'MPE', 'Boot (MAE) p': 'MAE',
        'Boot (MedAE) p': 'MedAE', 'Boot (RMSE) p': 'RMSE',
        'Morgan-Pitman (SD) p': 'SD'
    }

    p_cols = [c for c in df_results.columns if ' p' in c and 'adj' not in c]

    for col in p_cols:
        pvals = df_results[col].values
        pvals_adj = np.full(len(pvals), np.nan)
        mask = ~np.isnan(pvals)

        # a single comparison needs no adjustment
        if np.sum(mask) <= 1:
            pvals_adj[mask] = pvals[mask]
            use_fallback = False
        else:
            use_fallback = True

        # Romano-Wolf
        if method == 'romano_wolf' and col in rw_metric_mapping and rw_stats is not None and use_fallback:
            metric = rw_metric_mapping[col]
            obs = np.array(rw_stats[metric]['obs'])
            null_list = rw_stats[metric]['null']

            if len(obs) > 1 and len(null_list) > 0:
                null_matrix = np.column_stack(null_list)
                if np.sum(np.abs(null_matrix)) > 0:
                    pvals_adj_rw = romano_wolf_correction(obs, null_matrix)
                    pvals_adj[mask] = pvals_adj_rw[mask]
                    use_fallback = False

        # otherwise Hommel, Holm, etc.
        if use_fallback:
            current_method = FALLBACK_TEST_METHOD if method == 'romano_wolf' else method
            if np.sum(mask) > 0:
                _, adj, _, _ = multipletests(pvals[mask], method=current_method)
                pvals_adj[mask] = adj

        # an adjusted p-value is never smaller than the raw one
        if np.sum(mask) > 0:
            pvals_adj[mask] = np.maximum(pvals_adj[mask], pvals[mask])

        try:
            idx = df_results.columns.get_loc(col) + 1
            new_col_name = col.replace(' p', ' p_adj')
            if new_col_name in df_results.columns:
                  df_results[new_col_name] = pvals_adj
            else:
                df_results.insert(idx, new_col_name, pvals_adj)
        except Exception:
            pass

    return add_adjusted_cis(df_results, rw_stats, method=method)


def add_adjusted_cis(df_results, rw_stats=None, metrics=('MPE', 'MAE', 'MedAE', 'RMSE'), method='romano_wolf'):
    """Add adjusted ('CI adj') intervals next to each unadjusted confidence interval (MPE, MAE, MedAE; RMSE too under
    the percentile bootstrap), matching the multiple-comparison method (see adjusted_ci_critical). Needs report_ci;
    with a single comparison the adjusted interval is the unadjusted one."""
    if df_results.empty or method not in ('romano_wolf', 'bonferroni', 'holm', 'hommel', 'fdr_bh'): return df_results
    for metric in metrics:
        dcol = f'{metric} Diff'
        lows = [c for c in df_results.columns if c.startswith(f'{dcol} ') and c.endswith('% CI low') and 'adj' not in c]
        if not lows: continue
        lbl = lows[0][len(dcol) + 1:-len(' low')]                       # e.g. '95% CI'
        level = float(lbl.split('%')[0]) / 100
        low_col, high_col = f'{dcol} {lbl} low', f'{dcol} {lbl} high'
        if len(df_results) == 1:
            adj_lo, adj_hi = df_results[low_col].values.astype(float), df_results[high_col].values.astype(float)
        else:
            st = (rw_stats or {}).get(metric, {})
            diffs, scales = np.array(st.get('diff', []), float), np.array(st.get('scale', []), float)
            if len(diffs) != len(df_results) or not (np.isfinite(diffs).all() and np.isfinite(scales).all()):
                continue                                                # no statistic to adjust with (e.g. RMSE, bootstrap-t)
            pcol = f'Boot ({metric}) p_adj'
            n_rej = int(np.sum(df_results[pcol].values <= 1 - level + 1e-12)) if pcol in df_results.columns else None
            c = adjusted_ci_critical(np.column_stack(st['null']), level, method, n_rej)
            adj_lo, adj_hi = diffs - c * scales, diffs + c * scales
        idx = df_results.columns.get_loc(high_col) + 1
        for name, vals in ((f'{dcol} {lbl} adj low', adj_lo), (f'{dcol} {lbl} adj high', adj_hi)):
            if name in df_results.columns: df_results[name] = vals
            else: df_results.insert(idx, name, vals); idx += 1
    return df_results

def analyze_prediction_errors(df_input, axl_data=None, settings=None):
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    axl_short = settings.get('axl_short', AXL_SHORT_CUTOFF)
    axl_long = settings.get('axl_long', AXL_LONG_CUTOFF)

    def _process_group(df_sub, name):
        if df_sub.empty: return None
        res_df, rw_stats, methodology = run_biometry_analysis(
            df_sub,
            group_name=name,
            settings=settings
        )
        if res_df.empty: return None
        info = dict(res_df.attrs.get('group_info', {}), methodology=methodology)
        res_df = adjust_pvalues(res_df, rw_stats=rw_stats, method=fwer_method)
        return res_df, info

    groups = [("All", df_input)]

    if axl_data is not None:
        if isinstance(df_input, pd.DataFrame) and hasattr(axl_data, 'index'):
            common_idx = df_input.index.intersection(axl_data.index)
            df_aligned = df_input.loc[common_idx]
            axl_aligned = axl_data.loc[common_idx].iloc[:, 0] if isinstance(axl_data, pd.DataFrame) else axl_data.loc[common_idx]
        else:
            df_aligned, axl_aligned = df_input, axl_data

        axl_vals = np.array(axl_aligned)

        mask_short = axl_vals <= axl_short
        mask_long = axl_vals >= axl_long
        mask_medium = (axl_vals > axl_short) & (axl_vals < axl_long)

        if np.sum(mask_short) > 0:
            groups.append(("Short", df_aligned.iloc[mask_short]))

        if np.sum(mask_medium) > 0:
            groups.append(("Medium", df_aligned.iloc[mask_medium]))

        if np.sum(mask_long) > 0:
            groups.append(("Long", df_aligned.iloc[mask_long]))

    # The groups are independent (each draws its own resamples from BOOTSTRAP_SEED), so they run side by side;
    # results are collected in the order above, identical to running them one after another.
    print(f"--- Running Analysis: {', '.join(name for name, _ in groups)} ---")
    if RUN_GROUPS_IN_PARALLEL and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            outcomes = list(executor.map(lambda g: _process_group(g[1], g[0]), groups))
    else:
        outcomes = [_process_group(df_sub, name) for name, df_sub in groups]

    all_results_frames, group_info = [], {}
    for (name, _), outcome in zip(groups, outcomes):
        if outcome is not None:
            all_results_frames.append(outcome[0])
            group_info[name] = outcome[1]

    if not all_results_frames: return pd.DataFrame()

    final_df = pd.concat(all_results_frames, ignore_index=True)

    # The methods table describes the first group tested (All eyes); where a subgroup used a different method
    # (e.g. it has no bilateral pairs), that is added to the entry.
    base_name = next(iter(group_info))
    global_methodology = dict(group_info[base_name]['methodology'])
    for name, info in group_info.items():
        if name == base_name: continue
        for k, v in info['methodology'].items():
            if k not in global_methodology:
                global_methodology[k] = f"{name} eyes: {v}"
            elif v != group_info[base_name]['methodology'].get(k):
                global_methodology[k] += f"; {name} eyes: {v}"

    # describe the multiplicity correction in the methodology table
    if global_methodology:
        if fwer_method == 'romano_wolf': fwer_name = 'Romano-Wolf'
        elif fwer_method == 'fdr_bh': fwer_name = 'FDR-BH'
        else: fwer_name = fwer_method.capitalize()

        if fwer_method == 'romano_wolf':
            global_methodology['FWER Correction'] = f"{fwer_name} (Bootstrapped Tests) / Hommel (Asymptotic Fallback)"
        else:
            global_methodology['FWER Correction'] = fwer_name

        # put the thresholds into the labels
        mc_t = float(settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD))
        inv_t = float(settings.get('invalid_threshold', INVALID_THRESHOLD))
        labelled = {}
        for k, v in global_methodology.items():
            if k == 'Accuracy Test (McNemar)':
                labelled[f'Accuracy Test (McNemar, |PE| ≤ {mc_t:.2f} D)'] = f"{v} – proportion of eyes with |PE| ≤ {mc_t:.2f} D"
            else:
                labelled[k] = v
        labelled['Invalid PE Exclusion'] = f"Eyes with |PE| > {inv_t:.2f} D excluded from the pairwise dependence tests"
        global_methodology = labelled

        summary_data = [
            {'Test Category': k, 'Methodology Applied': v}
            for k, v in global_methodology.items()
        ]
        # kept as a list of dicts (a DataFrame in attrs breaks pd.concat)
        final_df.attrs['methodology_summary'] = summary_data
    final_df.attrs['mcnemar_threshold'] = float(settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD))
    # per group: eyes and patients tested, whether bilateral clustering was used, and the tests applied
    final_df.attrs['group_info'] = group_info
    final_df.attrs['fwer_method'] = fwer_method

    return final_df

# ------------------------------------------------------------------------------
# Excel output
# ------------------------------------------------------------------------------
def _set_column_widths(writer, sheet_name, widths):
    """Column widths for an openpyxl or an xlsxwriter ExcelWriter sheet."""
    try:
        ws = writer.sheets[sheet_name]
        if hasattr(ws, 'set_column'):                                   # xlsxwriter
            for idx, w in enumerate(widths):
                ws.set_column(idx, idx, w)
        else:                                                           # openpyxl
            from openpyxl.utils import get_column_letter
            for idx, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(idx)].width = w
    except (ImportError, AttributeError, KeyError):
        pass

def write_advanced_stats_to_excel(df_results, writer, sheet_name='Advanced_Stats'):
    """Write the results to an Excel sheet and the tests used to a 'Statistical Tests' sheet."""
    if df_results.empty:
        print("Warning: DataFrame is empty. Nothing to write.")
        return

    cols_to_drop = ['Precision Test (SD)', 'Accuracy Test (McNemar)', 'MAE Test', 'Dependence Strategy', 'FWER Correction']
    df_clean = df_results.drop(columns=[c for c in cols_to_drop if c in df_results.columns], errors='ignore')

    # e.g. "McNemar p_adj" -> "McNemar (≤0.50D) p_adj" so the tested threshold travels with the numbers
    mc_t = df_results.attrs.get('mcnemar_threshold') if hasattr(df_results, 'attrs') else None
    if mc_t is not None:
        df_clean = df_clean.rename(columns={c: c.replace('McNemar', f'McNemar (≤{float(mc_t):.2f}D)')
                                            for c in df_clean.columns if 'McNemar' in c})

    df_clean.to_excel(writer, sheet_name=sheet_name, index=False)

    _set_column_widths(writer, sheet_name, [15 if ' p' in col else 22 for col in df_clean.columns])

    if hasattr(df_results, 'attrs') and 'methodology_summary' in df_results.attrs:
        summary_data = df_results.attrs['methodology_summary']
        if isinstance(summary_data, list):
            meth_df = pd.DataFrame(summary_data)
        elif isinstance(summary_data, pd.DataFrame):
            meth_df = summary_data
        else:
            meth_df = pd.DataFrame()

        if not meth_df.empty:
            meth_sheet = 'Statistical Tests'
            meth_df.to_excel(writer, sheet_name=meth_sheet, index=False)

            _set_column_widths(writer, meth_sheet, [35 if idx == 0 else 75 for idx in range(len(meth_df.columns))])


# ==============================================================================
# ophthalmology_stats: extensions to biometry_stats_advanced
# ==============================================================================
# Everything above this line is biometry_stats_advanced, unchanged, so IOL users keep analyze_prediction_errors
# and its output exactly. Everything below generalizes it to any prediction model evaluated on eyes (AI models,
# formulas, devices), for continuous outputs and classifiers, using the same resamples, tests, p-value rules,
# confidence intervals, Romano-Wolf adjustment and Excel output.

CONTINUOUS_METRICS = ["bias", "mae", "rmse", "medae", "sd", "within"]
BINARY_METRICS = ["auc", "brier", "logloss", "sensitivity", "specificity", "accuracy"]
MEAN_LOSS_METRICS = {"bias", "mae", "rmse", "brier", "logloss"}

# Column names taken as the reference (truth) when truth is not given and task is 'auto' or 'binary'. With
# task='auto' and none of these present, the model columns are taken as prediction errors (continuous output).
TRUTH_VARIANTS = ['truth', 'reference', 'reference_standard', 'ground_truth', 'groundtruth', 'label', 'y', 'y_true',
                  'target', 'diagnosis', 'outcome', 'actual', 'observed', 'measured', 'gold_standard']
# Numeric columns never auto-detected as models (the subgroup column is excluded as well)
GENERIC_IGNORE = ['axl', 'al', 'axial_length', 'acd', 'acdepth', 'k1', 'k2', 'meank', 'lt', 'lensthickness', 'cct',
                  'wtw', 'hcd', 'iol_implanted', 'postoperative_refraction', 'corneal_refractive_index', 'eye', 'side',
                  'laterality', 'od_os', 're_le', 'age', 'gender', 'gender_num', 'sex', 'visit',
                  'site', 'centre', 'center', 'device', 'image_id', 'imageid']
SUBGROUP_MAX_LEVELS = 10   # a numeric subgroup column with more distinct values is split into tertiles by default
ADJUSTED_CI_METHODS = ('romano_wolf', 'bonferroni', 'holm', 'hommel', 'fdr_bh')   # methods with adjusted intervals
P_ADJ_LABELS = {'romano_wolf': 'Romano-Wolf step-down', 'bonferroni': 'Bonferroni', 'holm': 'Holm',
                'hommel': 'Hommel', 'fdr_bh': 'Benjamini-Hochberg (FDR)', 'fdr_by': 'Benjamini-Yekutieli (FDR)',
                'sidak': 'Sidak', 'holm-sidak': 'Holm-Sidak', 'simes-hochberg': 'Hochberg',
                'fdr_tsbh': 'two-stage Benjamini-Hochberg (FDR)',
                'fdr_tsbky': 'two-stage Benjamini-Krieger-Yekutieli (FDR)'}
SINGLE_P = "single comparison (p_adj = p)"
SINGLE_CI = "single comparison (CI adj = CI)"

DEFAULT_SETTINGS = {
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,  # resamples (p-values on a grid of 1/(B+1))
    "seed": None,                  # None: BOOTSTRAP_SEED
    "fwer_method": MULTIPLE_TEST_METHOD,   # 'romano_wolf' (Hommel for tests without bootstrap nulls), 'holm', ...
    "report_ci": True,             # 95% confidence interval for every difference (unadjusted and adjusted)
    "ci_level": CI_LEVEL,
    "within": MCNEMAR_THRESHOLD,   # continuous: tolerance for the proportion of eyes with |error| <= within
    "threshold": 0.5,              # binary: probability threshold for sensitivity, specificity and accuracy
    "invalid_threshold": None,     # continuous: eyes with |error| above this are excluded (IOL default 4.0 D)
    "weighting": "eye",            # mean-loss metrics: 'eye', 'patient' or 'optimal' (minimum variance)
    "unit": "patient",             # 'patient' (recommended) or 'eye' (every eye independent; for comparison)
    "one_eye": False,              # keep one randomly chosen eye per patient (as ENFORCE_INDEPENDENCE)
    "clip": 1e-15,                 # probabilities clipped to [clip, 1 - clip] for log-loss
    "metrics": None,               # subset of the metrics for the task (default: all)
    "min_group_size": MIN_GROUP_SIZE,
    "subgroup_cuts": None,         # numeric subgroup column: cut points, e.g. [22, 24.5] -> <22, 22 to <24.5, >=24.5
    "subgroup_bins": None,         # numeric subgroup column: number of equal-count quantile groups, e.g. 3 (tertiles)
    "verbose": True,
}


def _settings(settings):
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    if s["seed"] is None:
        s["seed"] = BOOTSTRAP_SEED
    if s["weighting"] not in ("eye", "patient", "optimal"):
        raise ValueError("weighting must be 'eye', 'patient' or 'optimal'")
    if s["subgroup_cuts"] is not None and s["subgroup_bins"] is not None:
        raise ValueError("give subgroup_cuts or subgroup_bins, not both")
    if s["subgroup_cuts"] is not None:
        cuts = sorted({float(c) for c in np.atleast_1d(s["subgroup_cuts"])})
        if not cuts or not np.all(np.isfinite(cuts)):
            raise ValueError("subgroup_cuts must be one or more finite numbers")
        s["subgroup_cuts"] = cuts
    if s["subgroup_bins"] is not None:
        if int(s["subgroup_bins"]) != s["subgroup_bins"] or int(s["subgroup_bins"]) < 2:
            raise ValueError("subgroup_bins must be an integer of at least 2")
        s["subgroup_bins"] = int(s["subgroup_bins"])
    return s


# ------------------------------------------------------------------------------
# Fellow-eye correlation and the efficiency of keeping both eyes
# ------------------------------------------------------------------------------
def fellow_eye_correlation(values, cluster_idx):
    """Pairwise intraclass correlation of a per-eye quantity between eyes of the same patient (rho).

    Applied to the per-eye difference in loss between two models it gives ρΔ, the quantity that decides
    how much eye-level tests are misled and how much a second eye adds (Kerry & Bland 2001; Eldridge 2006)."""
    values = np.asarray(values, float)
    m = np.bincount(cluster_idx).astype(float)
    a = np.bincount(cluster_idx, weights=values)
    q = np.bincount(cluster_idx, weights=values ** 2)
    N, NP = m.sum(), np.sum(m * (m - 1))
    if NP == 0 or N == 0:
        return np.nan
    mu = a.sum() / N
    s2 = q.sum() / N - mu ** 2
    if s2 <= 0:
        return np.nan
    return float((np.sum(a ** 2 - q) / NP - mu ** 2) / s2)


def one_eye_factor(sizes, rho, weighting="eye"):
    """How many times more patients one eye per patient needs for the same precision as the analysis of all
    eyes with the given weighting (design effect for unequal cluster sizes; Kerry & Bland 2001)."""
    m = np.asarray(sizes, float)
    if not np.isfinite(rho) or len(m) == 0:
        return np.nan
    r = float(np.clip(rho, 0.0, 0.999))
    K = len(m)
    var = {"eye": np.sum(m * (1 + (m - 1) * r)) / np.sum(m) ** 2,
           "patient": np.sum((1 + (m - 1) * r) / m) / K ** 2,
           "optimal": 1.0 / np.sum(m / (1 + (m - 1) * r))}[weighting]
    return float((1.0 / K) / var)


def eye_weights(cluster_idx, weighting, rho=0.0):
    """Per-eye weights: 'eye' 1; 'patient' 1/m (each patient counts once); 'optimal' 1/(1 + (m - 1) rho)."""
    m = np.bincount(cluster_idx).astype(float)[cluster_idx]
    if weighting == "eye":
        return np.ones(len(cluster_idx))
    if weighting == "patient":
        return 1.0 / m
    r = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.999))
    return 1.0 / (1.0 + (m - 1.0) * r)


# ------------------------------------------------------------------------------
# Engine: (weighted) mean of any per-eye loss, every pair at once
# ------------------------------------------------------------------------------
def all_pairs_loss_bootstrap(L, cluster_idx, K, b_indices, weights=None, pairs=None, ci_level=None,
                             sqrt_keys=(), extra=None, chunk=None):
    """Patient bootstrap-t (CR1) for differences in the mean of per-eye losses, for every pair of models.

    The generalization of all_pairs_bootstrap's MPE/MAE/RMSE engine to any loss (Brier score, log-loss, ...)
    and to weighted means: L maps a name to an n x J matrix of per-eye losses (one column per model); weights
    (one per eye) default to 1, in which case the results equal all_pairs_bootstrap's for the same losses and
    resamples. Keys in sqrt_keys are tested on the mean loss but report a percentile CI on the root scale
    (RMSE from squared errors); their bootstrap draws on the root scale are returned too (DRAWS_<key>, for the
    adjusted interval). extra maps names to per-cluster vectors whose bootstrap sums are returned
    (used for the proportion CIs). Returns ({(j, k): result}, {name: B-vector of bootstrap sums}).
    """
    names = list(L)
    n, J = L[names[0]].shape
    B = b_indices.shape[0]
    pairs = pairs or list(combinations(range(J), 2))
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    N_g = np.bincount(cluster_idx, weights=w, minlength=K).astype(float)
    N_tot = N_g.sum()
    S = {f: np.stack([np.bincount(cluster_idx, weights=w * L[f][:, j], minlength=K) for j in range(J)], axis=1)
         for f in names}
    means = {f: S[f].sum(0) / N_tot for f in names}
    C = {f: S[f] - np.outer(N_g, means[f]) for f in names}
    tri = [(a, b) for a in range(J) for b in range(a, J)]
    cols, index = [N_g], {}
    for f in names:
        index[(f, "s")] = len(cols); cols += [C[f][:, j] for j in range(J)]
        index[(f, "x")] = len(cols); cols += [C[f][:, a] * C[f][:, b] for a, b in tri]
        index[(f, "n")] = len(cols); cols += [C[f][:, j] * N_g for j in range(J)]
    index["nn"] = len(cols); cols.append(N_g ** 2)
    for f in sqrt_keys:
        index[(f, "raw")] = len(cols); cols += [S[f][:, j] for j in range(J)]
    extra = extra or {}
    for name, v in extra.items():
        index[("extra", name)] = len(cols); cols.append(np.asarray(v, float))
    V = np.column_stack(cols)
    tpos = {t: i for i, t in enumerate(tri)}

    chunk = chunk or max(1, min(500, int(2e7 // max(1, n))))
    sums = np.empty((B, V.shape[1]))
    for i in range(0, B, chunk):
        sums[i:i + chunk] = _resample_counts(b_indices[i:i + chunk], K) @ V
    bN = sums[:, 0]
    a_ci = (1 - ci_level) / 2 if ci_level else None
    qci = lambda v: tuple(float(x) for x in np.quantile(v[np.isfinite(v)], [a_ci, 1 - a_ci]))

    def cr1_se(Svec, mean_val):
        res = Svec - mean_val * N_g
        return np.sqrt(max(0.0, (K / (K - 1)) * np.sum(res ** 2) / N_tot ** 2)) if K > 1 else 0.0

    out = {}
    for (j, k) in pairs:
        r = {}
        for f in names:
            d_mean_obs = means[f][j] - means[f][k]
            se_obs = cr1_se(S[f][:, j] - S[f][:, k], d_mean_obs)
            t_obs = d_mean_obs / se_obs if se_obs > 0 else 0.0
            s = index[(f, "s")]; x = index[(f, "x")]; nn_ = index[(f, "n")]
            sd = sums[:, s + j] - sums[:, s + k]
            sdd = sums[:, x + tpos[(j, j)]] + sums[:, x + tpos[(k, k)]] - 2 * sums[:, x + tpos[(min(j, k), max(j, k))]]
            sdn = sums[:, nn_ + j] - sums[:, nn_ + k]
            snn = sums[:, index["nn"]]
            m = sd / bN
            ssq = sdd - 2 * m * sdn + m ** 2 * snn
            var = (K / (K - 1)) * np.maximum(ssq, 0) / bN ** 2 if K > 1 else np.zeros(B)
            se = np.sqrt(var)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_star = np.where(se > 0, m / se, 0.0)
            r[f"P_{f}"] = _bootstrap_p(np.abs(t_star), np.abs(t_obs)) if se_obs > 0 else 1.0
            r[f"RW_{f}_obs"] = np.abs(t_obs); r[f"RW_{f}_null"] = np.abs(t_star)
            # root-scale keys (RMSE from squared errors) are tested on the mean loss: no adjusted interval on the root scale
            r[f"RW_{f}_diff"] = np.nan if f in sqrt_keys else d_mean_obs
            r[f"RW_{f}_scale"] = np.nan if f in sqrt_keys else se_obs
            r[f"DIFF_{f}"] = d_mean_obs
            r[f"EST_{f}"] = (means[f][j], means[f][k])
            if ci_level:
                if f in sqrt_keys:
                    q = index[(f, "raw")]
                    draws = np.sqrt(np.maximum(0, sums[:, q + j] / bN)) - np.sqrt(np.maximum(0, sums[:, q + k] / bN))
                    r[f"CI_{f}"] = qci(draws)
                    r[f"DRAWS_{f}"] = draws
                else:
                    lo_q, hi_q = qci(t_star)
                    r[f"CI_{f}"] = (d_mean_obs - hi_q * se_obs, d_mean_obs - lo_q * se_obs)
        out[(j, k)] = r
    extra_sums = {name: sums[:, index[("extra", name)]] for name in extra}
    return out, extra_sums


# ------------------------------------------------------------------------------
# Engine: AUC (Obuchowski 1997; DeLong et al. 1988 when every patient has one eye)
# ------------------------------------------------------------------------------
def _placements(x_pos, y_neg):
    ys, xs = np.sort(y_neg), np.sort(x_pos)
    lt = np.searchsorted(ys, x_pos, side="left"); le = np.searchsorted(ys, x_pos, side="right")
    v10 = (lt + 0.5 * (le - lt)) / len(y_neg)
    gt = len(x_pos) - np.searchsorted(xs, y_neg, side="right"); ge = len(x_pos) - np.searchsorted(xs, y_neg, side="left")
    v01 = (gt + 0.5 * (ge - gt)) / len(x_pos)
    return v10, v01


def auc_all_pairs(P, y, cluster_idx, K, pairs=None, ci_level=0.95):
    """Nonparametric clustered AUC and its comparison between models (Obuchowski 1997, equations 3-5).

    P: n x J predicted probabilities (or any scores); y: 0/1 reference standard per eye. The covariance between
    two models' AUCs uses the patient as the unit, so both eyes can be kept; with one eye per patient the method
    reduces to DeLong et al. (1988). Returns {(j, k): (auc_j, auc_k, diff, ci_low, ci_high, p, se)}."""
    y = np.asarray(y).astype(int)
    n, J = P.shape
    pos, neg = y == 1, y == 0
    mi = np.bincount(cluster_idx, weights=pos, minlength=K)
    ni = np.bincount(cluster_idx, weights=neg, minlength=K)
    M, Nn = mi.sum(), ni.sum()
    I10, I01 = int(np.sum(mi > 0)), int(np.sum(ni > 0))
    pairs = pairs or list(combinations(range(J), 2))
    out = {}
    if M == 0 or Nn == 0 or I10 < 2 or I01 < 2:
        return {pr: (np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan) for pr in pairs}
    th, r10, r01 = [], [], []
    for j in range(J):
        v10, v01 = _placements(P[pos, j], P[neg, j])
        t = float(np.mean(v10))
        th.append(t)
        r10.append(np.bincount(cluster_idx[pos], weights=v10, minlength=K) - mi * t)
        r01.append(np.bincount(cluster_idx[neg], weights=v01, minlength=K) - ni * t)
    S10 = lambda a, b: I10 / ((I10 - 1) * M) * np.sum(r10[a] * r10[b])
    S01 = lambda a, b: I01 / ((I01 - 1) * Nn) * np.sum(r01[a] * r01[b])
    S11 = lambda a, b: K / (K - 1) * np.sum(r10[a] * r01[b])
    z = stats.norm.ppf(1 - (1 - (ci_level or 0.95)) / 2)
    for (j, k) in pairs:
        var = [S10(r, r) / M + S01(r, r) / Nn + 2 * S11(r, r) / (M * Nn) for r in (j, k)]
        cov = S10(j, k) / M + S01(j, k) / Nn + (S11(j, k) + S11(k, j)) / (M * Nn)
        se = float(np.sqrt(max(var[0] + var[1] - 2 * cov, 0.0)))
        diff = th[j] - th[k]
        p = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else 1.0
        out[(j, k)] = (th[j], th[k], diff, diff - z * se, diff + z * se, p, se)
    return out


# ------------------------------------------------------------------------------
# Bootstrap draws behind the percentile intervals of all_pairs_bootstrap (for the adjusted intervals)
# ------------------------------------------------------------------------------
def _resampled_cluster_sums(cols, b_indices, K, chunk=500):
    """B x C matrix: the sum of each per-cluster vector in cols over the patients drawn in each resample."""
    V = np.column_stack(cols)
    out = np.empty((b_indices.shape[0], V.shape[1]))
    for i in range(0, b_indices.shape[0], chunk):
        out[i:i + chunk] = _resample_counts(b_indices[i:i + chunk], K) @ V
    return out


def continuous_percentile_draws(E, cluster_idx, K, b_indices, within, pairs=None):
    """Bootstrap draws of the RMSE, SD and proportion-within-tolerance differences for every pair of models, from the
    same resamples and formulas as the percentile intervals of all_pairs_bootstrap (which does not return them).
    Returns {'rmse' | 'sd' | 'within': {(j, k): B-vector}}; quantiles of these at level 1 - alpha/m give the
    Bonferroni-adjusted intervals."""
    E = np.asarray(E, float)
    n, J = E.shape
    pairs = pairs or list(combinations(range(J), 2))
    N_g = np.bincount(cluster_idx, minlength=K).astype(float)
    S = np.stack([np.bincount(cluster_idx, weights=E[:, j], minlength=K) for j in range(J)], axis=1)
    mean = S.sum(0) / N_g.sum()
    C = S - np.outer(N_g, mean)                                       # centered sums, as all_pairs_bootstrap
    sq = [np.bincount(cluster_idx, weights=(E[:, j] - mean[j]) ** 2, minlength=K) for j in range(J)]
    raw = [np.bincount(cluster_idx, weights=(E ** 2)[:, j], minlength=K) for j in range(J)]
    prop = [np.bincount(cluster_idx, weights=(np.abs(E[:, j]) <= within).astype(float), minlength=K) for j in range(J)]
    sums = _resampled_cluster_sums([N_g] + [C[:, j] for j in range(J)] + sq + raw + prop, b_indices, K)
    bN = sums[:, :1]
    c0, s0, r0, p0 = 1, 1 + J, 1 + 2 * J, 1 + 3 * J
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(np.maximum(0, (sums[:, s0:s0 + J] - sums[:, c0:c0 + J] ** 2 / bN) / (bN - 1)))
    rmse = np.sqrt(np.maximum(0, sums[:, r0:r0 + J] / bN))
    pw = sums[:, p0:p0 + J] / bN
    return {"rmse": {(j, k): rmse[:, j] - rmse[:, k] for j, k in pairs},
            "sd": {(j, k): sd[:, j] - sd[:, k] for j, k in pairs},
            "within": {(j, k): pw[:, j] - pw[:, k] for j, k in pairs}}


# ------------------------------------------------------------------------------
# Comparing models: one group of eyes
# ------------------------------------------------------------------------------
def _detect_columns(df, task, models, patient, truth, extra_ignore=()):
    """Resolve the task and the patient, reference (truth) and model columns.

    truth not given: with task 'auto' or 'binary' a column named as in TRUTH_VARIANTS is used if present; with task
    'continuous' none is looked for, so the model columns are prediction errors. models not given: every numeric
    column other than the patient, reference, GENERIC_IGNORE and extra_ignore (e.g. the subgroup) columns. task
    'auto' becomes 'binary' when the reference is 0/1 and every model value lies in [0, 1], else 'continuous'
    (prediction errors when there is no reference column)."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    if patient is None:
        patient = next((cols_lower[v] for v in ['patient_id', 'id_patient', 'patientid', 'pt_id', 'id', 'hosp_id',
                                                 'hospid', 'ptid', 'patient', 'subject', 'subject_id']
                        if v in cols_lower), None)
    if truth is None and task != "continuous":
        truth = next((cols_lower[v] for v in TRUTH_VARIANTS if v in cols_lower), None)
    if models is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        skip = {str(patient).lower(), str(truth).lower()} | set(GENERIC_IGNORE) | {str(c).lower() for c in extra_ignore}
        models = [c for c in numeric if str(c).lower() not in skip and not str(c).lower().startswith('predict_')
                  and not str(c).lower().endswith('_const')]
        named_like_truth = [c for c in models if str(c).lower() in TRUTH_VARIANTS]
        if task == "continuous" and truth is None and named_like_truth:
            warnings.warn(f"column(s) {named_like_truth} are analyzed as prediction errors of a model; if one is the "
                          f"reference value, pass truth='{named_like_truth[0]}' (--truth) or use task='auto'",
                          stacklevel=3)
    if task == "auto":
        task = "continuous"
        if truth is not None:
            yv = pd.to_numeric(df[truth], errors="coerce").dropna()
            pv = df[models].stack()
            if set(np.unique(yv)) <= {0, 1} and pv.min() >= 0 and pv.max() <= 1:
                task = "binary"
    return task, list(models), patient, truth


def model_performance(P, y, task, models, metrics, cluster_idx, K, b_idx, s, has_clusters, group_name="All"):
    """Each model's own performance, every metric, with a percentile bootstrap CI that resamples patients (the same
    resamples as the comparisons), so fellow eyes stay together. Eyes are weighted equally."""
    n = P.shape[0]
    ci_level = float(s["ci_level"])
    a = (1 - ci_level) / 2
    method = "patient bootstrap, percentile" if has_clusters else "bootstrap, percentile"
    stats = {}                                   # metric -> (point estimates per model, list of bootstrap chunks)
    if task == "continuous":
        E = P if y is None else P - y[:, None]
        absE = np.abs(E)
        within = (absE <= s["within"]).astype(float)
        order = np.argsort(absE, axis=0)
    else:
        clip = float(s["clip"])
        Pc = np.clip(P, clip, 1 - clip)
        brier = (P - y[:, None]) ** 2
        logloss = -(y[:, None] * np.log(Pc) + (1 - y[:, None]) * np.log(1 - Pc))
        correct = ((P >= s["threshold"]) == (y[:, None] == 1)).astype(float)
        pos, neg = (y == 1).astype(float), (y == 0).astype(float)
        auc_prep = []
        for j in range(P.shape[1]):
            o = np.argsort(P[:, j], kind="mergesort")
            sv = P[o, j]
            starts = np.r_[0, np.flatnonzero(np.diff(sv)) + 1]
            auc_prep.append((o, starts))
    point = {}
    for metric in metrics:
        if task == "continuous":
            point[metric] = {"bias": E.mean(0), "mae": absE.mean(0), "rmse": np.sqrt((E ** 2).mean(0)),
                             "medae": np.median(absE, 0), "sd": E.std(0, ddof=1) if n > 1 else np.full(E.shape[1], np.nan),
                             "within": within.mean(0)}.get(metric)
        else:
            if metric == "auc":
                pt = []
                for j in range(P.shape[1]):
                    xp, xn = P[y == 1, j], P[y == 0, j]
                    pt.append(_placements(xp, xn)[0].mean() if len(xp) and len(xn) else np.nan)
                point[metric] = np.array(pt)
            else:
                point[metric] = {"brier": brier.mean(0), "logloss": logloss.mean(0),
                                 "sensitivity": correct[y == 1].mean(0) if pos.sum() else np.full(P.shape[1], np.nan),
                                 "specificity": correct[y == 0].mean(0) if neg.sum() else np.full(P.shape[1], np.nan),
                                 "accuracy": correct.mean(0)}.get(metric)
        stats[metric] = []
    B = b_idx.shape[0]
    for c0 in range(0, B, 250):
        cnt = np.zeros((min(250, B - c0), K))
        for r, row in enumerate(b_idx[c0:c0 + 250]):
            cnt[r] = np.bincount(row, minlength=K)
        W = cnt[:, cluster_idx]                     # resample x eye weights (copies of each patient)
        tot = W.sum(1, keepdims=True)
        for metric in metrics:
            if task == "continuous":
                if metric == "bias":
                    v = (W @ E) / tot
                elif metric == "mae":
                    v = (W @ absE) / tot
                elif metric == "rmse":
                    v = np.sqrt((W @ (E ** 2)) / tot)
                elif metric == "within":
                    v = (W @ within) / tot
                elif metric == "sd":
                    m1 = (W @ E) / tot
                    v = np.sqrt(np.maximum((W @ (E ** 2)) - tot * m1 ** 2, 0) / np.maximum(tot - 1, 1))
                elif metric == "medae":
                    v = np.empty((W.shape[0], E.shape[1]))
                    for j in range(E.shape[1]):
                        cw = np.cumsum(W[:, order[:, j]], axis=1)
                        k = np.argmax(cw >= cw[:, -1:] / 2, axis=1)
                        v[:, j] = absE[order[:, j], j][k]
                else:
                    continue
            else:
                if metric == "brier":
                    v = (W @ brier) / tot
                elif metric == "logloss":
                    v = (W @ logloss) / tot
                elif metric in ("sensitivity", "specificity"):
                    sub = pos if metric == "sensitivity" else neg
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v = (W @ (correct * sub[:, None])) / (W @ sub)[:, None]
                elif metric == "accuracy":
                    v = (W @ correct) / tot
                elif metric == "auc":
                    v = np.empty((W.shape[0], P.shape[1]))
                    for j, (o, starts) in enumerate(auc_prep):
                        wp = np.add.reduceat(W[:, o] * pos[o], starts, axis=1)
                        wn = np.add.reduceat(W[:, o] * neg[o], starts, axis=1)
                        below = np.cumsum(wn, axis=1) - wn
                        with np.errstate(divide="ignore", invalid="ignore"):
                            v[:, j] = (wp * (below + 0.5 * wn)).sum(1) / (wp.sum(1) * wn.sum(1))
                else:
                    continue
            stats[metric].append(v)
    rows = []
    for metric in metrics:
        if not stats[metric] or point[metric] is None:
            continue
        bs = np.vstack(stats[metric])
        for j, m in enumerate(models):
            col = bs[:, j]
            col = col[np.isfinite(col)]
            lo, hi = (np.quantile(col, [a, 1 - a]) if len(col) else (np.nan, np.nan))
            rows.append({"group": group_name, "model": m, "metric": metric, "estimate": float(point[metric][j]),
                         "ci_low": float(lo), "ci_high": float(hi), "n_eyes": int(n), "n_patients": int(K),
                         "method": method})
    return rows


def compare_models(df, task="continuous", models=None, patient=None, truth=None, settings=None, group_name="All"):
    """Compare every pair of models on the same eyes, with the patient as the unit of analysis.

    df       one row per eye: a patient ID column and one column per model
    task     'continuous': each model column holds the prediction errors (prediction minus reference), the
                           primary input; or, if truth is given, the predictions, and the errors are
                           prediction - truth
             'binary':     each model column holds a predicted probability, and truth the 0/1 diagnosis (reference
                           standard), which is required: AUC, sensitivity and specificity are defined by it
             'auto':       'binary' when the reference is 0/1 and every model value lies in [0, 1], else
                           'continuous'. When truth is not given, a column named as in TRUTH_VARIANTS (truth,
                           reference, diagnosis, measured, ...) is used as the reference if present; without one
                           the model columns are prediction errors
    models   columns holding each model's output (auto-detected if None: numeric columns other than the patient,
             reference and metadata columns; see GENERIC_IGNORE)
    patient  patient identifier column (auto-detected); eyes sharing an ID are resampled together
    truth    reference value (optional for continuous outputs) or 0/1 diagnosis (required for classifiers)
    settings dict overriding DEFAULT_SETTINGS

    Returns (wide, long, info): wide has one row per model pair with differences, CIs, adjusted CIs ('CI adj'),
    p-values and adjusted p-values in the biometry_stats_advanced layout; long has one row per pair and metric,
    with the estimates for each model, the fellow-eye correlation of the loss difference (ρΔ, column rho_delta),
    how many times more patients one eye per patient would need (one_eye_factor), the adjusted p-value and CI, and
    the adjustment each got (p_adj_method, ci_adj_method); info describes the sample and the tests used, and
    info["model_performance"] gives each model's own estimate of every metric with its patient-level 95% CI.

    Adjusted intervals are given for every metric when report_ci is on and fwer_method is one of
    ADJUSTED_CI_METHODS: the bootstrap-t (MPE, MAE, Brier, log-loss) and MedAE intervals by
    adjusted_ci_critical (max-|t| for Romano-Wolf); AUC as difference ± z(1 - α/2m) × SE; RMSE, SD and proportions
    as the percentile interval at level 1 - α/m from the same resamples (Bonferroni; Benjamini-Yekutieli
    1 - Rα/m for fdr_bh), m being the number of model pairs. With a single pair they equal the unadjusted ones.
    """
    s = _settings(settings)
    task, models, patient, truth = _detect_columns(df, task, models, patient, truth)
    if len(models) < 2:
        raise ValueError("need at least two model columns")
    data = df.copy()
    keep = data[models].notna().all(axis=1)
    if truth is not None:
        keep &= data[truth].notna()
    if task == "continuous" and s["invalid_threshold"] is not None:
        err = data[models].values - (data[[truth]].values if truth is not None else 0)
        keep &= (np.abs(err) <= s["invalid_threshold"]).all(axis=1)
    if patient is not None:
        keep &= data[patient].notna()
    data = data[keep].reset_index(drop=True)
    if s["one_eye"] and patient is not None:
        rng_indep = np.random.default_rng(s["seed"])
        order = rng_indep.permutation(len(data))
        data = data.iloc[order].drop_duplicates(subset=[patient]).sort_index().reset_index(drop=True)
    n = len(data)
    if n < max(2, int(s["min_group_size"])):
        if s["verbose"]:
            print(f"[{group_name}] Not tested: {n} valid eye(s).")
        return pd.DataFrame(), pd.DataFrame(), {}

    use_patient = patient is not None and s["unit"] == "patient"
    cluster_ids = data[patient].astype(str).values if use_patient else np.arange(n)
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = int(cluster_idx.max()) + 1
    sizes = np.bincount(cluster_idx, minlength=K)
    has_clusters = K < n
    B = int(s["bootstrap_iterations"])
    ci_level = float(s["ci_level"]) if s["report_ci"] else None

    # resamples and wild-bootstrap weights drawn exactly as in run_biometry_analysis
    rng = np.random.default_rng(s["seed"])
    b_idx = np.empty((B, K), dtype=np.int32)
    for i in range(0, B, 500):
        b_idx[i:i + 500] = rng.choice(K, size=(min(500, B - i), K), replace=True)
    W_g = np.empty((B, K), dtype=np.int8)
    for i in range(0, B, 500):
        W_g[i:i + 500] = rng.choice([-1, 1], size=(min(500, B - i), K))

    pairs = list(combinations(range(len(models)), 2))
    P = data[models].values.astype(float)
    y = None if truth is None else data[truth].values.astype(float)
    metrics = s["metrics"] or (CONTINUOUS_METRICS if task == "continuous" else BINARY_METRICS)
    long_rows, rw_stats = [], {}
    # sources of the Bonferroni-type adjusted intervals: bootstrap draws (percentile CIs) and AUC standard errors
    want_adj = bool(ci_level) and s["fwer_method"] in ADJUSTED_CI_METHODS and len(pairs) > 1
    pct_draws, auc_se = {}, {}
    if s["verbose"]:
        print(f"[{group_name}] {task}: {len(models)} models, {n} eyes of {K} {'patients' if use_patient else 'units'}"
              f" ({int(np.sum(sizes >= 2))} with both eyes), {B} resamples, weighting={s['weighting']}")

    def add(metric, j, k, est_a, est_b, diff, ci, p, rho, test, factor_sizes=sizes):
        long_rows.append({"group": group_name, "metric": metric, "model_a": models[j], "model_b": models[k],
                          "est_a": est_a, "est_b": est_b, "diff": diff,
                          "ci_low": ci[0] if ci else np.nan, "ci_high": ci[1] if ci else np.nan, "p": p,
                          "rho_delta": rho,
                          "one_eye_factor": one_eye_factor(factor_sizes, rho, s["weighting"])
                          if metric in MEAN_LOSS_METRICS else one_eye_factor(factor_sizes, rho, "eye"),
                          "test": test})

    def weighted_engine(Lmat_by_metric, sqrt_keys=()):
        """Mean-loss tests with patient or minimum-variance weights, pair by pair (weights depend on rho)."""
        res = {}
        for (j, k) in pairs:
            res[(j, k)] = {}
            for f, Lm in Lmat_by_metric.items():
                rho = fellow_eye_correlation(Lm[:, j] - Lm[:, k], cluster_idx)
                w = eye_weights(cluster_idx, s["weighting"], rho)
                r, _ = all_pairs_loss_bootstrap({f: Lm[:, [j, k]]}, cluster_idx, K, b_idx, weights=w,
                                                pairs=[(0, 1)], ci_level=ci_level,
                                                sqrt_keys=tuple(x for x in sqrt_keys if x == f))
                res[(j, k)].update(r[(0, 1)])
        return res

    if task == "continuous":
        E = P if y is None else P - y[:, None]
        absE = np.abs(E)
        boot = all_pairs_bootstrap(E, cluster_idx, K, b_idx, use_studentized=USE_STUDENTIZED_BS, ci_level=ci_level,
                                   prop_threshold=s["within"])
        if s["weighting"] != "eye":
            wres = weighted_engine({"MPE": E, "MAE": absE, "MSE": E ** 2}, sqrt_keys=("MSE",))
        needs_wcb = RUN_SD_TEST and (has_clusters or s["fwer_method"] == "romano_wolf")
        sd_all = wild_cluster_sd_all_pairs(E, cluster_idx, K, W_g) if (RUN_SD_TEST and needs_wcb) else None
        if want_adj and {"rmse", "sd", "within"} & set(metrics):
            pct_draws = continuous_percentile_draws(E, cluster_idx, K, b_idx, s["within"], pairs)
            if s["weighting"] != "eye":                     # the RMSE interval then comes from the weighted engine
                pct_draws["rmse"] = {pr: wres[pr].get("DRAWS_MSE") for pr in pairs}
        for (j, k) in pairs:
            ea, eb = E[:, j], E[:, k]
            r = boot[(j, k)]
            fam = {"bias": ("MPE", ea, eb, ea - eb), "mae": ("MAE", np.abs(ea), np.abs(eb), np.abs(ea) - np.abs(eb)),
                   "rmse": ("RMSE", ea ** 2, eb ** 2, ea ** 2 - eb ** 2)}
            for metric, (key, la, lb, dl) in fam.items():
                if metric not in metrics:
                    continue
                rho = fellow_eye_correlation(dl, cluster_idx) if has_clusters else np.nan
                if s["weighting"] == "eye":
                    est = (np.mean(ea), np.mean(eb)) if metric == "bias" else (np.mean(la), np.mean(lb))
                    if metric == "rmse":
                        est = (np.sqrt(np.mean(la)), np.sqrt(np.mean(lb)))
                    p, ci = r[f"P_{key}"], r.get(f"CI_{key}")
                    _rw_push(rw_stats, key, r, key)
                    test = "patient bootstrap-t (CR1)" if has_clusters else "bootstrap-t"
                else:
                    wk = {"bias": "MPE", "mae": "MAE", "rmse": "MSE"}[metric]
                    wr = wres[(j, k)]
                    ea_, eb_ = wr[f"EST_{wk}"]
                    est = (np.sqrt(ea_), np.sqrt(eb_)) if metric == "rmse" else (ea_, eb_)
                    p, ci = wr[f"P_{wk}"], wr.get(f"CI_{wk}")
                    _rw_push(rw_stats, key, wr, wk)
                    test = f"patient bootstrap-t (CR1), {s['weighting']} weights"
                add(metric, j, k, est[0], est[1], est[0] - est[1], ci, p, rho, test)
            if "medae" in metrics:
                _rw_push(rw_stats, "MedAE", r, "MedAE")
                add("medae", j, k, np.median(np.abs(ea)), np.median(np.abs(eb)),
                    np.median(np.abs(ea)) - np.median(np.abs(eb)), r.get("CI_MedAE"), r["P_MedAE"], np.nan,
                    "percentile patient bootstrap" if has_clusters else "percentile bootstrap")
            if "sd" in metrics and RUN_SD_TEST:
                if needs_wcb:
                    p_sd, obs_sd, null_sd = sd_all[(j, k)]
                    test = "Morgan-Pitman, wild cluster bootstrap"
                else:
                    p_sd, obs_sd, null_sd = comdvar(ea, eb), 0.0, np.zeros(B)
                    test = "Morgan-Pitman, HC4"
                rw_stats.setdefault("SD", {"obs": [], "null": []})
                rw_stats["SD"]["obs"].append(obs_sd); rw_stats["SD"]["null"].append(null_sd)
                add("sd", j, k, np.std(ea, ddof=1), np.std(eb, ddof=1), np.std(ea, ddof=1) - np.std(eb, ddof=1),
                    r.get("CI_SD"), p_sd, np.nan, test)
            if "within" in metrics:
                ma, mb = np.abs(ea) <= s["within"], np.abs(eb) <= s["within"]
                p_w = clustered_mcnemar_test(ma, mb, cluster_ids) if has_clusters else mcnemar_test(ma, mb)
                rho = fellow_eye_correlation(ma.astype(float) - mb.astype(float), cluster_idx) if has_clusters else np.nan
                add("within", j, k, np.mean(ma), np.mean(mb), np.mean(ma) - np.mean(mb), r.get("CI_PROP"), p_w, rho,
                    "clustered McNemar (Yang 2010)" if has_clusters else "McNemar")

    else:
        if y is None:
            raise ValueError("binary task needs a 0/1 reference standard column (truth)")
        clip = float(s["clip"])
        Pc = np.clip(P, clip, 1 - clip)
        L = {"Brier": (P - y[:, None]) ** 2,
             "LogLoss": -(y[:, None] * np.log(Pc) + (1 - y[:, None]) * np.log(1 - Pc))}
        lab = P >= s["threshold"]
        correct = lab == (y[:, None] == 1)
        subsets = {"sensitivity": y == 1, "specificity": y == 0, "accuracy": np.ones(n, bool)}
        extra = {}
        for metric, sub in subsets.items():
            extra[f"{metric}|n"] = np.bincount(cluster_idx, weights=sub.astype(float), minlength=K)
            for j in range(len(models)):
                extra[f"{metric}|{j}"] = np.bincount(cluster_idx, weights=(correct[:, j] & sub).astype(float),
                                                     minlength=K)
        loss_res, extra_sums = all_pairs_loss_bootstrap(L, cluster_idx, K, b_idx, pairs=pairs, ci_level=ci_level,
                                                        extra=extra)
        if s["weighting"] != "eye":
            wres = weighted_engine(L)
        auc = auc_all_pairs(P, y, cluster_idx, K, pairs, ci_level) if "auc" in metrics else {}
        a_ci = (1 - ci_level) / 2 if ci_level else None
        for (j, k) in pairs:
            if "auc" in metrics:
                th_j, th_k, d, lo, hi, p, se = auc[(j, k)]
                auc_se[(j, k)] = (d, se)
                add("auc", j, k, th_j, th_k, d, (lo, hi) if ci_level else None, p, np.nan,
                    "Obuchowski (1997) clustered AUC" if has_clusters else "DeLong (1988)")
            for metric, key in (("brier", "Brier"), ("logloss", "LogLoss")):
                if metric not in metrics:
                    continue
                r = loss_res[(j, k)] if s["weighting"] == "eye" else wres[(j, k)]
                rho = fellow_eye_correlation(L[key][:, j] - L[key][:, k], cluster_idx) if has_clusters else np.nan
                _rw_push(rw_stats, key, r, key)
                ea_, eb_ = r[f"EST_{key}"]
                add(metric, j, k, ea_, eb_, ea_ - eb_, r.get(f"CI_{key}"), r[f"P_{key}"], rho,
                    "patient bootstrap-t (CR1)" + ("" if s["weighting"] == "eye" else f", {s['weighting']} weights"))
            for metric, sub in subsets.items():
                if metric not in metrics:
                    continue
                ma, mb = correct[sub, j], correct[sub, k]
                ids_sub = cluster_ids[sub]
                if len(ma) < 2:
                    continue
                p_w = clustered_mcnemar_test(ma, mb, ids_sub) if has_clusters else mcnemar_test(ma, mb)
                rho = (fellow_eye_correlation(ma.astype(float) - mb.astype(float), np.unique(ids_sub, return_inverse=True)[1])
                       if has_clusters else np.nan)
                ci = None
                if ci_level:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        bd = (extra_sums[f"{metric}|{j}"] - extra_sums[f"{metric}|{k}"]) / extra_sums[f"{metric}|n"]
                    bd = bd[np.isfinite(bd)]
                    ci = tuple(float(x) for x in np.quantile(bd, [a_ci, 1 - a_ci])) if len(bd) else None
                    pct_draws.setdefault(metric, {})[(j, k)] = bd
                add(metric, j, k, np.mean(ma), np.mean(mb), np.mean(ma) - np.mean(mb), ci, p_w, rho,
                    "clustered McNemar (Yang 2010)" if has_clusters else "McNemar",
                    factor_sizes=np.bincount(np.unique(ids_sub, return_inverse=True)[1]))

    long = pd.DataFrame(long_rows)
    wide = _to_wide(long, task, n, K, group_name, s)
    p_methods = {}
    wide = _adjust(wide, rw_stats, s["fwer_method"], p_methods)
    wide = _add_bonferroni_type_cis(wide, task, models, pct_draws, auc_se, s)
    if not long.empty:
        long["p_adj"] = long.apply(lambda r: _lookup_adj(wide, r), axis=1)
        long["p_adj_method"] = [p_methods.get(WIDE_NAMES[m][1], "not available") for m in long["metric"]]
        if s["report_ci"]:
            lbl = f"{float(s['ci_level']) * 100:g}% CI"
            for side in ("low", "high"):
                long[f"ci_adj_{side}"] = long.apply(lambda r: _lookup_col(wide, r, f"{WIDE_NAMES[r['metric']][0]} {lbl} adj {side}"), axis=1)
        long["ci_adj_method"] = [_ci_adj_label(m, wide, s, has_clusters) for m in long["metric"]]
    info = {"task": task, "models": models, "patient_column": patient if use_patient else None, "truth": truth,
            "n_eyes": int(n), "n_patients": int(K), "n_bilateral": int(np.sum(sizes >= 2)),
            "bilateral_fraction": float(np.mean(sizes >= 2)), "clustered": bool(has_clusters),
            "bootstrap_iterations": B, "seed": s["seed"], "weighting": s["weighting"], "unit": s["unit"],
            "one_eye": bool(s["one_eye"]), "fwer_method": s["fwer_method"],
            "methodology": _methodology(long, task, has_clusters, s, n_pairs=len(wide)),
            "model_performance": (model_performance(P, y, task, models, metrics, cluster_idx, K, b_idx, s,
                                                    has_clusters, group_name) if s["report_ci"] else [])}
    return wide, long, info


WIDE_NAMES = {
    "bias": ("MPE Diff", "Boot (MPE) p"), "mae": ("MAE Diff", "Boot (MAE) p"),
    "medae": ("MedAE Diff", "Boot (MedAE) p"), "rmse": ("RMSE Diff", "Boot (RMSE) p"),
    "sd": ("SD Diff", "Morgan-Pitman (SD) p"), "within": ("McNemar Diff", "McNemar p"),
    "auc": ("AUC Diff", "Obuchowski (AUC) p"), "brier": ("Brier Diff", "Boot (Brier) p"),
    "logloss": ("LogLoss Diff", "Boot (LogLoss) p"), "sensitivity": ("Sensitivity Diff", "Sensitivity p"),
    "specificity": ("Specificity Diff", "Specificity p"), "accuracy": ("Accuracy Diff", "Accuracy p"),
}
RW_MAP = {"Boot (MPE) p": "MPE", "Boot (MAE) p": "MAE", "Boot (MedAE) p": "MedAE", "Boot (RMSE) p": "RMSE",
          "Morgan-Pitman (SD) p": "SD", "Boot (Brier) p": "Brier", "Boot (LogLoss) p": "LogLoss"}


def _to_wide(long, task, n, K, group_name, s):
    if long.empty:
        return pd.DataFrame()
    rows = []
    order = CONTINUOUS_METRICS if task == "continuous" else BINARY_METRICS
    for (a, b), g in long.groupby(["model_a", "model_b"], sort=False):
        row = {"Group": group_name, "Model A": a, "Model B": b, "N": n, "Patients": K}
        gm = g.set_index("metric")
        for metric in order:
            if metric not in gm.index:
                continue
            r = gm.loc[metric]
            dname, pname = WIDE_NAMES[metric]
            row[dname] = r["diff"]
            if s["report_ci"]:
                lbl = f"{float(s['ci_level']) * 100:g}% CI"
                row[f"{dname} {lbl} low"], row[f"{dname} {lbl} high"] = r["ci_low"], r["ci_high"]
            row[pname] = r["p"]
        rows.append(row)
    return pd.DataFrame(rows)


def _adjust(wide, rw_stats, method, methods_out=None):
    """adjust_pvalues with the Romano-Wolf mapping extended to the new bootstrap metrics. methods_out (a dict), if
    given, receives the adjustment applied to each p-value column (e.g. 'Romano-Wolf step-down', 'Hommel')."""
    if wide.empty:
        return wide
    methods_out = {} if methods_out is None else methods_out
    for col in [c for c in wide.columns if c.endswith(" p")]:
        pvals = wide[col].values.astype(float)
        adj = np.full(len(pvals), np.nan)
        mask = ~np.isnan(pvals)
        use_fallback = np.sum(mask) > 1
        if not use_fallback:
            adj[mask] = pvals[mask]
            methods_out[col] = SINGLE_P
        if method == "romano_wolf" and col in RW_MAP and use_fallback and RW_MAP[col] in rw_stats:
            obs = np.array(rw_stats[RW_MAP[col]]["obs"])
            nulls = rw_stats[RW_MAP[col]]["null"]
            if len(obs) > 1 and len(nulls) > 0:
                null_matrix = np.column_stack(nulls)
                if np.sum(np.abs(null_matrix)) > 0:
                    adj_rw = romano_wolf_correction(obs, null_matrix)
                    adj[mask] = adj_rw[mask]
                    use_fallback = False
                    methods_out[col] = P_ADJ_LABELS["romano_wolf"]
        if use_fallback and method not in (None, "none"):
            current = FALLBACK_TEST_METHOD if method == "romano_wolf" else method
            _, a, _, _ = multipletests(pvals[mask], method=current)
            adj[mask] = a
            methods_out[col] = P_ADJ_LABELS.get(current, str(current))
        elif use_fallback:
            adj[mask] = pvals[mask]
            methods_out[col] = "none (p_adj = p; fwer_method='none')"
        adj[mask] = np.maximum(adj[mask], pvals[mask])
        wide.insert(wide.columns.get_loc(col) + 1, col.replace(" p", " p_adj"), adj)
    # adjusted intervals next to each CI, matching the method used for the p-values (see adjusted_ci_critical)
    return add_adjusted_cis(wide, rw_stats, metrics=("MPE", "MAE", "MedAE", "RMSE", "Brier", "LogLoss"), method=method)


BOOTSTRAP_T_ADJ = {"bias", "mae", "medae", "brier", "logloss"}   # adjusted intervals from add_adjusted_cis
PERCENTILE_ADJ = {"rmse", "sd", "within", "sensitivity", "specificity", "accuracy"}   # Bonferroni percentile


def _adjusted_coverage(method, level, m, n_reject=0):
    """Per-comparison coverage of a Bonferroni-type adjusted interval, as in adjusted_ci_critical: 1 - α/m for
    Romano-Wolf (these metrics have no max-|t| statistic; their p-values use Hommel or a different statistic),
    Bonferroni, Holm and Hommel; Benjamini-Yekutieli 1 - Rα/m for fdr_bh (R = comparisons with p_adj <= α)."""
    alpha = 1 - level
    if method in ("romano_wolf", "bonferroni", "holm", "hommel"):
        return 1 - alpha / m
    if method == "fdr_bh":
        return 1 - max(int(n_reject), 1) * alpha / m
    return None


def _add_bonferroni_type_cis(wide, task, models, pct_draws, auc_se, s):
    """Adjusted ('CI adj') intervals for the metrics whose interval is not a bootstrap-t: AUC, difference ± z × SE
    with z the 1 - (1 - q)/2 normal quantile; RMSE, SD and the proportions, the percentile interval at level q from
    the same bootstrap draws as the unadjusted interval. q = 1 - α/m (Bonferroni; 1 - Rα/m for fdr_bh), m being the
    number of model pairs with a p-value for that metric. With a single pair they equal the unadjusted intervals."""
    if wide.empty or not s["report_ci"] or s["fwer_method"] not in ADJUSTED_CI_METHODS:
        return wide
    level = float(s["ci_level"])
    lbl = f"{level * 100:g}% CI"
    keys = [(models.index(a), models.index(b)) for a, b in zip(wide["Model A"], wide["Model B"])]
    for metric in (CONTINUOUS_METRICS if task == "continuous" else BINARY_METRICS):
        if metric not in PERCENTILE_ADJ and metric != "auc":
            continue
        dname, pname = WIDE_NAMES[metric]
        low_col, high_col = f"{dname} {lbl} low", f"{dname} {lbl} high"
        if low_col not in wide.columns or f"{dname} {lbl} adj low" in wide.columns:
            continue                                     # (add_adjusted_cis copies RMSE's CI for a single pair)
        if len(wide) == 1:
            adj_lo, adj_hi = wide[low_col].values.astype(float), wide[high_col].values.astype(float)
        else:
            src = auc_se if metric == "auc" else pct_draws.get(metric, {})
            m = int(np.sum(np.isfinite(wide[pname].values.astype(float))))
            n_rej = int(np.sum(wide[pname.replace(" p", " p_adj")].values <= 1 - level + 1e-12))
            q = _adjusted_coverage(s["fwer_method"], level, max(m, 1), n_rej)
            a = (1 - q) / 2
            adj_lo, adj_hi = np.full(len(wide), np.nan), np.full(len(wide), np.nan)
            for i, key in enumerate(keys):
                if key not in src or src[key] is None or not np.isfinite(wide[low_col].iloc[i]):
                    continue
                if metric == "auc":
                    d, se = src[key]
                    z = stats.norm.ppf(1 - a)
                    adj_lo[i], adj_hi[i] = d - z * se, d + z * se
                else:
                    v = np.asarray(src[key], float)
                    v = v[np.isfinite(v)]
                    if len(v):
                        adj_lo[i], adj_hi[i] = np.quantile(v, [a, 1 - a])
        idx = wide.columns.get_loc(high_col) + 1
        for name, vals in ((f"{dname} {lbl} adj low", adj_lo), (f"{dname} {lbl} adj high", adj_hi)):
            wide.insert(idx, name, vals)
            idx += 1
    return wide


def _ci_adj_label(metric, wide, s, has_clusters=True):
    """How the adjusted interval of a metric was made (column ci_adj_method), or why there is none."""
    fm = s["fwer_method"]
    if not s["report_ci"]:
        return "not available: no confidence intervals (report_ci=False)"
    if fm in (None, "none"):
        return "not available: no multiplicity adjustment (fwer_method='none')"
    if fm not in ADJUSTED_CI_METHODS:
        return f"not available: {P_ADJ_LABELS.get(fm, fm)} has no interval counterpart"
    if f"{WIDE_NAMES[metric][0]} {float(s['ci_level']) * 100:g}% CI adj low" not in wide.columns:
        return "not available: no bootstrap statistic for this metric in this group"
    if len(wide) == 1:
        return SINGLE_CI
    if fm == "fdr_bh":
        head = "Benjamini-Yekutieli false-coverage-rate"
    elif metric in BOOTSTRAP_T_ADJ and fm == "romano_wolf":
        return "Romano-Wolf simultaneous (max-|t|)"
    else:
        head = "Bonferroni"
    if metric in BOOTSTRAP_T_ADJ:
        return f"{head} (bootstrap |t|)" if metric != "medae" else f"{head} (bootstrap, studentized by its SD)"
    if metric == "auc":
        return f"{head} (normal theory, {'Obuchowski' if has_clusters else 'DeLong'} SE)"
    return f"{head} (percentile bootstrap)"


def _lookup_adj(wide, r):
    return _lookup_col(wide, r, WIDE_NAMES[r["metric"]][1].replace(" p", " p_adj"))


def _lookup_col(wide, r, col):
    sel = wide[(wide["Model A"] == r["model_a"]) & (wide["Model B"] == r["model_b"])]
    return float(sel[col].iloc[0]) if col in sel and len(sel) else np.nan


def _rw_push(rw_stats, key, r, rkey):
    """Collect one comparison's Romano-Wolf statistics: observed |t|, bootstrap null, difference and its scale."""
    st = rw_stats.setdefault(key, {"obs": [], "null": [], "diff": [], "scale": []})
    st["obs"].append(r[f"RW_{rkey}_obs"]); st["null"].append(r[f"RW_{rkey}_null"])
    st["diff"].append(r.get(f"RW_{rkey}_diff", np.nan)); st["scale"].append(r.get(f"RW_{rkey}_scale", np.nan))


METRIC_LABELS = {"bias": "Mean Error (MPE)", "mae": "MAE", "rmse": "RMSE", "medae": "MedAE", "sd": "SD",
                 "within": "Proportion Within Tolerance", "auc": "AUC", "brier": "Brier Score", "logloss": "Log-Loss",
                 "sensitivity": "Sensitivity", "specificity": "Specificity", "accuracy": "Accuracy"}


def _by_method(long, col):
    """'method: metric, metric; method: metric' from one row per metric of the long table."""
    first = long.drop_duplicates("metric")
    groups = {}
    for metric, method in zip(first["metric"], first[col]):
        groups.setdefault(method, []).append(METRIC_LABELS.get(metric, metric))
    return "; ".join(f"{k}: {', '.join(v)}" for k, v in groups.items())


def _methodology(long, task, has_clusters, s, n_pairs=None):
    if long.empty:
        return {}
    m = {"Dependence Strategy": ("Patient as the unit (cluster bootstrap / cluster-robust tests)" if has_clusters
                                 else "Independent eyes (one eye per patient or no patient identifier)"),
         "Weighting of Eyes (mean-loss metrics)": {"eye": "Every eye counts once",
                                                   "patient": "Every patient counts once",
                                                   "optimal": "Minimum-variance weights 1/(1 + (m - 1) ρΔ)"}
         [s["weighting"]]}
    for metric, g in long.groupby("metric", sort=False):
        m[f"{METRIC_LABELS.get(metric, metric)} Test"] = g["test"].iloc[0]
    if task == "binary":
        m["Classification Threshold"] = f"predicted probability >= {s['threshold']}"
    else:
        m["Tolerance (proportion within)"] = f"|error| <= {s['within']}"
    m["FWER Correction"] = ("Romano-Wolf (bootstrap tests) / Hommel (other tests)" if s["fwer_method"] == "romano_wolf"
                            else str(s["fwer_method"]))
    n_pairs = n_pairs if n_pairs is not None else long[["model_a", "model_b"]].drop_duplicates().shape[0]
    single = n_pairs == 1
    if "p_adj_method" in long:
        m["Adjusted p-values (p_adj)"] = (
            "A single model pair: no adjustment is needed, p_adj = p" if single else
            f"Adjusted for the m = {n_pairs} model pairs of each metric within the group (column p_adj_method). "
            + _by_method(long, "p_adj_method"))
    if s["report_ci"]:
        fm, level = s["fwer_method"], float(s["ci_level"])
        alpha = f"{1 - level:g}"
        if fm in ADJUSTED_CI_METHODS and single:
            text = ("A single model pair: no adjustment is needed, so the 'CI adj' columns (ci_adj_low, ci_adj_high) "
                    "equal the unadjusted intervals")
        elif fm in ADJUSTED_CI_METHODS:
            q = f"1 - R × {alpha}/m (Benjamini-Yekutieli; R = comparisons with p_adj ≤ {alpha})" \
                if fm == "fdr_bh" else f"1 - {alpha}/m (Bonferroni)"
            boot_t, pct = (("MPE, MAE and MedAE", "RMSE, SD and the proportion within tolerance") if task == "continuous"
                           else ("Brier score and log-loss", "sensitivity, specificity and accuracy"))
            text = (f"Columns 'CI adj' (Estimates: ci_adj_low, ci_adj_high; method in ci_adj_method), adjusted for the "
                    f"m = {n_pairs} model pairs of each metric within the group. "
                    + (_by_method(long, "ci_adj_method") + ". " if "ci_adj_method" in long else "")
                    + f"{boot_t}: " + adjusted_ci_description(fm, level) + ". "
                    + (f"AUC: difference ± z × SE, z the normal quantile for coverage {q}. "
                       if task == "binary" else "")
                    + f"{pct[0].upper() + pct[1:]}: percentile interval of the same patient resamples at coverage {q}. "
                    "These Bonferroni-type intervals need not agree with p_adj when p_adj comes from another "
                    "statistic (Romano-Wolf, Hommel); use more resamples (e.g. 9,999) for their far tails")
        else:
            text = adjusted_ci_description(fm, level)
        m["Adjusted Confidence Intervals"] = text
    m["Fellow-Eye Correlation"] = ("ρΔ (rho_delta): intraclass correlation of the per-eye difference in loss "
                                   "between the two models; one_eye_factor: patients a one-eye study needs, "
                                   "relative to this analysis (design effect)")
    return m


# ------------------------------------------------------------------------------
# Full analysis: all eyes and optional subgroups (e.g. laterality, device, center, axial length)
# ------------------------------------------------------------------------------
QUANTILE_NAMES = {2: "half", 3: "tertile", 4: "quartile", 5: "quintile", 10: "decile"}


def _num(v):
    return f"{float(v):g}"


def _bound(edges, i):
    """Interval i of the cut points edges: '<a', 'a to <b' or '>=b'."""
    if i == 0:
        return f"<{_num(edges[0])}"
    if i == len(edges):
        return f">={_num(edges[-1])}"
    return f"{_num(edges[i - 1])} to <{_num(edges[i])}"


def define_subgroups(df, column, cuts=None, bins=None, max_levels=SUBGROUP_MAX_LEVELS):
    """Split the eyes into subgroups by one column of any type.

    nominal (strings, booleans, unordered categories)  one subgroup per level, 'eye = L'
    ordinal (ordered categories; numeric columns with at most max_levels distinct values)  one per level, in order
    continuous (numeric)  cuts: cut points, e.g. [22, 24.5] gives '<22', '22 to <24.5' and '>=24.5' (each interval
                          includes its lower cut point); bins: that many equal-count quantile groups (equal numbers of
                          eyes, cut at the quantiles), labeled with the range of values they hold. A numeric column with
                          more than max_levels distinct values and neither cuts nor bins is split into tertiles.
    Eyes with no value are in no subgroup. Returns (groups, description): groups is a list of (name, boolean mask
    over the rows of df, definition); description has the column, its type ('kind'), a 'summary' line for the
    methodology, whether tertiles were used by default ('default_bins'), and the number of eyes with no value.
    """
    if column not in df.columns:
        raise KeyError(f"subgroup column {column!r} is not in the data")
    col = df[column]
    present = np.array(col.notna(), dtype=bool)
    is_cat = isinstance(col.dtype, pd.CategoricalDtype)
    numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col) and not is_cat
    if (cuts is not None or bins is not None) and not numeric:
        x = pd.to_numeric(col.astype(object), errors="coerce")
        if x.notna().sum() < present.sum():
            raise ValueError(f"subgroup_cuts and subgroup_bins need a numeric subgroup column; {column!r} is not")
        col, numeric = x, True
    groups, default_bins = [], False
    if numeric:
        x = col.astype("float64").to_numpy()
        present = present & np.isfinite(x)
        levels = np.unique(x[present])
        if cuts is None and bins is None and len(levels) > max_levels:
            bins, default_bins = 3, True
        if cuts is None and bins is None:
            kind = "ordinal (numeric levels)"
            for v in levels:
                name = f"{column} = {_num(v)}"
                groups.append((name, present & (x == v), name))
            summary = f"{column}, {kind}: one subgroup per value ({', '.join(_num(v) for v in levels)})"
        elif cuts is not None:
            edges = sorted({float(c) for c in np.atleast_1d(cuts)})
            kind = "continuous, cut points"
            idx = np.searchsorted(edges, x, side="right")
            for i in range(len(edges) + 1):
                name = f"{column} {_bound(edges, i)}"
                groups.append((name, present & (idx == i), name))
            summary = (f"{column}, continuous, cut points {', '.join(_num(c) for c in edges)}: "
                       f"{', '.join(_bound(edges, i) for i in range(len(edges) + 1))} (each interval includes its "
                       "lower cut point)")
        else:
            q = int(bins)
            edges = list(np.unique(np.quantile(x[present], np.arange(1, q) / q))) if present.any() else []
            idx = np.searchsorted(edges, x, side="right")
            filled = [i for i in range(len(edges) + 1) if np.any(present & (idx == i))]
            word = QUANTILE_NAMES.get(q) if len(filled) == q else None
            kind = "continuous, equal-count quantile groups"
            for r, i in enumerate(filled, 1):
                mask = present & (idx == i)
                name = (f"{column} {word or 'quantile group'} {r} of {len(filled)} "
                        f"({_num(x[mask].min())} to {_num(x[mask].max())})")
                groups.append((name, mask, f"{name}, i.e. {column} {_bound(edges, i)}"))
            summary = (f"{column}, continuous: {len(filled)} equal-count quantile groups"
                       + (f" ({word}s)" if word else "")
                       + (f" cut at {', '.join(_num(e) for e in edges)}" if edges else "")
                       + (f"; tertiles by default, as the column has more than {max_levels} distinct values "
                          "(set subgroup_cuts or subgroup_bins, CLI --cuts or --bins, to choose)" if default_bins else "")
                       + ("" if len(filled) == q else f"; fewer than the {q} groups asked for, because of tied values"))
    else:
        if is_cat:
            levels = [c for c in col.cat.categories if (col == c).any()]
            kind = "ordinal (ordered categories)" if col.cat.ordered else "nominal"
        else:
            levels = sorted(col.dropna().unique(), key=str)
            kind = "nominal"
        for lvl in levels:
            name = f"{column} = {lvl}"
            groups.append((name, present & (col == lvl).values, name))
        summary = f"{column}, {kind}: one subgroup per level ({', '.join(str(v) for v in levels)})"
    n_missing = int(len(df) - present.sum())
    if n_missing:
        summary += f"; {n_missing} eye(s) with no {column} value are in All only"
    groups = [g for g in groups if g[1].any()]
    return groups, {"column": column, "kind": kind, "summary": summary, "default_bins": default_bins,
                    "n_missing": n_missing}


def analyze(df, task="auto", models=None, patient=None, truth=None, subgroup=None, settings=None):
    """compare_models for all eyes and, if subgroup names a column, for each of its subgroups. Returns the wide table
    (one row per group and model pair) with the long table, each model's performance, the methodology and the
    sample (group_info) in .attrs, ready for write_results_to_excel.

    df        one row per eye: a patient ID column and, for continuous outputs, one column of prediction errors
              (prediction minus reference) per model; or predictions with truth naming the reference column. For
              classifiers, one column of predicted probabilities per model and truth, the 0/1 diagnosis (required:
              AUC, sensitivity and specificity are defined by it). See compare_models for task='auto'.
    subgroup  a column of any type: nominal (e.g. 'eye' for right and left eyes), ordinal, or continuous (e.g.
              axial length), split by settings subgroup_cuts (cut points) or subgroup_bins (equal-count quantile
              groups); tertiles by default for a numeric column with more than 10 distinct values (see
              define_subgroups). The subgroup column is never taken as a model.

    The task and the model, patient and reference columns are found once, on all eyes, and used for every group.
    Each group is analyzed separately: p-values and intervals are adjusted for the model pairs within the group,
    not across groups. group_info records each group's definition, e.g. 'axial_length 22 to <24.5: 412 eyes, 230
    patients'."""
    s = _settings(settings)
    if subgroup is not None and models is not None and subgroup in list(models):
        raise ValueError(f"the subgroup column {subgroup!r} cannot also be a model")
    task, models, patient, truth = _detect_columns(df, task, models, patient, truth,
                                                   extra_ignore=() if subgroup is None else (subgroup,))
    groups, sub_desc = [("All", np.ones(len(df), bool), "All eyes")], None
    if subgroup is not None:
        sub_groups, sub_desc = define_subgroups(df, subgroup, s["subgroup_cuts"], s["subgroup_bins"])
        groups += sub_groups
        if s["verbose"] and sub_desc["default_bins"]:
            print(f"[subgroups] {subgroup} has more than {SUBGROUP_MAX_LEVELS} distinct values: split into tertiles "
                  "(equal numbers of eyes) by default; set subgroup_cuts or subgroup_bins (--cuts, --bins) to choose.")
    wides, longs, infos, sample = [], [], {}, {}
    for name, mask, definition in groups:
        sub = df[mask]
        w, l, info = compare_models(sub, task=task, models=models, patient=patient, truth=truth, settings=s,
                                    group_name=name)
        if w.empty:
            sample[name] = {"definition": f"{definition}: {len(sub)} eye(s), not tested (fewer than "
                                          f"{max(2, int(s['min_group_size']))} valid eyes)"}
            continue
        units = (f"{info['n_patients']} patients" if info["patient_column"] is not None
                 else "eyes analyzed as independent units")
        wides.append(w); longs.append(l)
        infos[name] = info
        sample[name] = {"definition": f"{definition}: {info['n_eyes']} eyes, {units}",
                        **{k: v for k, v in info.items() if k not in ("methodology", "model_performance")}}
    if not wides:
        return pd.DataFrame()
    final = pd.concat(wides, ignore_index=True)
    long_all = pd.concat(longs, ignore_index=True)
    first = next(iter(infos.values()))
    methodology = dict(first["methodology"])
    if sub_desc is not None:
        methodology["Subgroups"] = (sub_desc["summary"] + ". Each subgroup is analyzed separately with the same models, "
                                    "tests and settings; p-values and intervals are adjusted for the model pairs "
                                    "within each group, not across groups")
    final.attrs["methodology_summary"] = [{"Test Category": k, "Methodology Applied": v} for k, v in methodology.items()]
    final.attrs["long"] = long_all.to_dict("records")
    final.attrs["model_performance"] = [r for v in infos.values() for r in v.get("model_performance", [])]
    final.attrs["group_info"] = sample
    return final


def write_results_to_excel(results, path):
    """Write the comparisons (sheet 'Results'), the tests and adjustments used ('Statistical Tests'), each model's
    performance with its patient-level CI ('Model Performance'), every comparison with its CI, adjusted CI, ρΔ
    (rho_delta), one-eye factor and the adjustment applied (p_adj_method, ci_adj_method; 'Estimates'), and the
    sample with each group's definition ('Sample')."""
    with pd.ExcelWriter(path) as writer:
        write_advanced_stats_to_excel(results, writer, sheet_name="Results")
        if results.attrs.get("model_performance"):
            mp = pd.DataFrame(results.attrs["model_performance"])
            mp.to_excel(writer, sheet_name="Model Performance", index=False)
            _set_column_widths(writer, "Model Performance", [18] * len(mp.columns))
        if "long" in results.attrs:
            est = pd.DataFrame(results.attrs["long"])
            est.to_excel(writer, sheet_name="Estimates", index=False)
            _set_column_widths(writer, "Estimates", [36 if c in ("test", "p_adj_method", "ci_adj_method") else 18
                                                     for c in est.columns])
        if "group_info" in results.attrs:
            gi = pd.DataFrame(results.attrs["group_info"]).T.reset_index().rename(columns={"index": "Group"})
            gi.to_excel(writer, sheet_name="Sample", index=False)
            _set_column_widths(writer, "Sample", [30, 60] + [16] * (len(gi.columns) - 2))


def _command_line(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Compare prediction models evaluated on eyes (one or both eyes per patient), with the patient as "
                    "the unit of analysis. Continuous outputs: give one column of prediction errors (prediction minus "
                    "reference) per model, or predictions with --truth. Classifiers: one column of predicted "
                    "probabilities per model and the 0/1 diagnosis (--truth), which AUC, sensitivity and specificity "
                    "need.")
    ap.add_argument("csv", help="one row per eye: a patient ID column and one column of prediction errors per model "
                                "(continuous outputs); or predictions plus a reference column (--truth); or, for "
                                "classifiers, predicted probabilities plus the 0/1 diagnosis (--truth)")
    ap.add_argument("--task", default="auto", choices=["auto", "continuous", "binary"],
                    help="auto (default): binary when the reference is 0/1 and every model value lies in [0, 1], "
                         "else continuous (the model columns are prediction errors when there is no reference)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model columns (default: every numeric column other than the patient, reference, subgroup "
                         "and common metadata columns)")
    ap.add_argument("--patient", default=None, help="patient ID column (default: patient_id, id, subject, ...)")
    ap.add_argument("--truth", default=None,
                    help="reference column: optional for continuous outputs (the model columns are then predictions "
                         "and the errors prediction - truth), required for classifiers (the 0/1 diagnosis). With "
                         "--task auto or binary a column named truth, reference, diagnosis, measured, ... is used "
                         "if present")
    ap.add_argument("--subgroup", default=None, metavar="COLUMN",
                    help="also give results for each subgroup of this column: nominal (e.g. eye), ordinal, or "
                         "continuous (e.g. axial_length; see --cuts and --bins)")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--cuts", nargs="+", type=float, default=None, metavar="CUT",
                       help="continuous subgroup: cut points, e.g. --cuts 22 24.5 gives <22, 22 to <24.5 and >=24.5")
    split.add_argument("--bins", type=int, default=None, metavar="Q",
                       help="continuous subgroup: Q equal-count quantile groups (e.g. 3 for tertiles). A numeric "
                            "column with more than 10 distinct values is split into tertiles if neither is given")
    ap.add_argument("--weighting", default="eye", choices=["eye", "patient", "optimal"])
    ap.add_argument("--threshold", type=float, default=0.5, help="classifiers: probability threshold (default 0.5)")
    ap.add_argument("--within", type=float, default=MCNEMAR_THRESHOLD,
                    help="continuous outputs: tolerance for the proportion of eyes with |error| <= within")
    ap.add_argument("--out", default="ophthalmology_stats_results.xlsx")
    a = ap.parse_args(argv)
    if (a.cuts is not None or a.bins is not None) and a.subgroup is None:
        ap.error("--cuts and --bins need --subgroup")
    res = analyze(pd.read_csv(a.csv), task=a.task, models=a.models, patient=a.patient, truth=a.truth,
                  subgroup=a.subgroup, settings={"weighting": a.weighting, "threshold": a.threshold,
                                                 "within": a.within, "subgroup_cuts": a.cuts,
                                                 "subgroup_bins": a.bins})
    write_results_to_excel(res, a.out)
    print(f"Results written to {a.out}")
    return res


if __name__ == "__main__":
    _command_line()
