"""Make the synthetic example data (one row per eye): example_classifier.csv and example_continuous.csv."""
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

# continuous output: prediction errors of three models for a measured value (e.g. refraction)
pid, eye = patients(250, 0.6)
n = len(pid)
measured = np.round(rng.normal(0, 2, n), 2)
shared = 0.3 * rng.normal(size=pid.max() + 1)[pid] + 0.3 * rng.normal(size=n)
cont = {}
for name, sd in (("model_A", 0.25), ("model_B", 0.30), ("model_C", 0.40)):
    cont[name] = np.round(measured + shared + rng.normal(0, sd, n), 2)
# axial length and mean keratometry per eye (fellow eyes alike), for subgroups
axial_length = np.round(rng.normal(23.6, 1.3, size=pid.max() + 1)[pid] + rng.normal(0, 0.12, n), 2)
mean_k = np.round(43.6 - 0.9 * (axial_length - 23.6) + rng.normal(0, 1.0, size=pid.max() + 1)[pid]
                  + rng.normal(0, 0.15, n), 2)
errors = {name: np.round(pred - measured, 2) for name, pred in cont.items()}
pd.DataFrame({"patient_id": [f"Q{p:03d}" for p in pid], "eye": eye, "axial_length": axial_length, "mean_k": mean_k,
              **errors}).to_csv("example_continuous.csv", index=False)
print("written example_classifier.csv and example_continuous.csv")
