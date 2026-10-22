"""Make the two synthetic example datasets (one row per eye). Run once; the CSVs are included in the repository."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)


def patients(n_pat, frac_bilateral):
    both = rng.random(n_pat) < frac_bilateral
    pid = np.repeat(np.arange(1, n_pat + 1), np.where(both, 2, 1))
    first_eye = rng.choice(["R", "L"], n_pat)
    eye = []
    for p, b, f in zip(range(n_pat), both, first_eye):
        eye += ["R", "L"] if b else [f]
    return pid, np.array(eye)


# classifier: referable disease from a fundus photograph, three models giving a probability per eye
pid, eye = patients(300, 0.7)
n = len(pid)
patient_effect = rng.normal(size=pid.max() + 1)[pid]
liability = 0.7 * patient_effect + 0.7 * rng.normal(size=n)
truth = (liability > 0.6).astype(int)
noise_shared = 0.6 * rng.normal(size=pid.max() + 1)[pid]
models = {}
for name, sd in (("model_A", 0.7), ("model_B", 0.9), ("model_C", 1.2)):
    score = 1.8 * liability + noise_shared + rng.normal(0, sd, n)
    models[name] = np.round(1 / (1 + np.exp(-(score - 1.0))), 4)
pd.DataFrame({"patient_id": [f"P{p:03d}" for p in pid], "eye": eye, "truth": truth, **models}).to_csv(
    "example_classifier.csv", index=False)

# continuous output: a measurement predicted by three models (e.g. a refractive or biometric value)
pid, eye = patients(250, 0.6)
n = len(pid)
measured = np.round(rng.normal(0, 2, n), 2)
shared = 0.3 * rng.normal(size=pid.max() + 1)[pid] + 0.3 * rng.normal(size=n)
cont = {}
for name, sd in (("model_A", 0.25), ("model_B", 0.30), ("model_C", 0.40)):
    cont[name] = np.round(measured + shared + rng.normal(0, sd, n), 2)
pd.DataFrame({"patient_id": [f"Q{p:03d}" for p in pid], "eye": eye, "truth": measured, **cont}).to_csv(
    "example_continuous.csv", index=False)
print("written example_classifier.csv and example_continuous.csv")
