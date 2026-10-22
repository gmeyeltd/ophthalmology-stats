"""
Data-generating mechanisms for the simulation (one row per eye).

Continuous (scalar output). For eye j of patient i, the prediction error of model r is

    e_r,ij = sigma_c * c_ij + lambda_r * sigma_m * m_r,ij
    c_ij   = sqrt(rho_c) * u_i + sqrt(1 - rho_c) * eps_ij            part shared by every model
    m_r,ij = sqrt(kappa) * a_r,i + sqrt(1 - kappa) * alpha_r,ij      part specific to model r

  rho_c   fellow-eye correlation of the shared part (what every model gets wrong in a patient)
  kappa   fellow-eye correlation of each model's own part: kappa is the fellow-eye correlation of the
          difference in signed error between two models, and the main lever on rho_delta
  sigma_c^2 = 0.8, sigma_m^2 = 0.2: errors of two models correlate 0.8 within an eye
  lambda_r = 1 for equally accurate models; > 1 makes model r less accurate (larger SD, MAE, RMSE)
  shape   distribution of every standardised component u, eps, a, alpha:
            "normal"; "heavy" (t, 4 df, scaled to variance 1); "skewed" (chi-square 2 df, centred
            and scaled); "discrete" (normal, final errors rounded to 0.1, i.e. about 1/10 of an SD)

Binary (classifier). Disease liability L_ij = sqrt(rho_y) u_i + sqrt(1 - rho_y) eps_ij; an eye has the
disease if L_ij exceeds the (1 - prevalence) quantile of N(0,1). Model r's score is
    s_r,ij = L_ij + tau * lambda_r * m_r,ij      (m_r as above, normal)
and its predicted probability is logistic(g0 + g1 * s), with (g0, g1) fitted once on a large sample so
that each model is calibrated. lambda_r > 1 makes a model noisier (lower AUC, worse Brier and log-loss).
"""
import numpy as np
from scipy import stats

SIGMA_C2, SIGMA_M2 = 0.8, 0.2


def _std_draw(rng, shape, size):
    if shape in ("normal", "discrete"):
        return rng.standard_normal(size)
    if shape == "heavy":
        return rng.standard_t(4, size) / np.sqrt(2.0)               # var of t4 is 2
    if shape == "skewed":
        return (rng.chisquare(2, size) - 2.0) / 2.0                 # mean 2, var 4
    raise ValueError(shape)


def patient_sizes(n_pat, frac_bilateral, rng):
    return np.where(rng.random(n_pat) < frac_bilateral, 2, 1)


def continuous_study(n_pat, frac_bilateral, kappa, rho_c, shape, lambdas, rng):
    """Returns (patient index per eye, errors as an n_eyes x n_models array)."""
    sizes = patient_sizes(n_pat, frac_bilateral, rng)
    idx = np.repeat(np.arange(n_pat), sizes)
    n = len(idx)
    c = np.sqrt(rho_c) * _std_draw(rng, shape, n_pat)[idx] + np.sqrt(1 - rho_c) * _std_draw(rng, shape, n)
    E = np.empty((n, len(lambdas)))
    for r, lam in enumerate(lambdas):
        m = np.sqrt(kappa) * _std_draw(rng, shape, n_pat)[idx] + np.sqrt(1 - kappa) * _std_draw(rng, shape, n)
        E[:, r] = np.sqrt(SIGMA_C2) * c + lam * np.sqrt(SIGMA_M2) * m
    if shape == "discrete":
        E = np.round(E, 1)
    return idx, E


def _liability(n_pat, frac_bilateral, rho_y, rng):
    sizes = patient_sizes(n_pat, frac_bilateral, rng)
    idx = np.repeat(np.arange(n_pat), sizes)
    L = np.sqrt(rho_y) * rng.standard_normal(n_pat)[idx] + np.sqrt(1 - rho_y) * rng.standard_normal(len(idx))
    return idx, L


def _scores(L, idx, n_pat, kappa, tau, lam, rng):
    m = np.sqrt(kappa) * rng.standard_normal(n_pat)[idx] + np.sqrt(1 - kappa) * rng.standard_normal(len(idx))
    return L + tau * lam * m


def calibration_map(prevalence, tau, lam, n=400_000, seed=0):
    """(g0, g1) for logistic(g0 + g1 * s): maximum likelihood on a large independent sample."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal(n)
    y = (L > stats.norm.ppf(1 - prevalence)).astype(float)
    s = L + tau * lam * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), s])
    beta = np.zeros(2)
    for _ in range(50):                                             # Newton-Raphson
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return tuple(beta)


def binary_study(n_pat, frac_bilateral, kappa, rho_y, prevalence, tau, lambdas, maps, rng):
    """Returns (patient index per eye, 0/1 disease per eye, probabilities n_eyes x n_models)."""
    idx, L = _liability(n_pat, frac_bilateral, rho_y, rng)
    y = (L > stats.norm.ppf(1 - prevalence)).astype(int)
    P = np.empty((len(idx), len(lambdas)))
    for r, lam in enumerate(lambdas):
        s = _scores(L, idx, n_pat, kappa, tau, lam, rng)
        g0, g1 = maps[r]
        P[:, r] = 1 / (1 + np.exp(-(g0 + g1 * s)))
    return idx, y, P
