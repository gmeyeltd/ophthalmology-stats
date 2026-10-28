"""
Worked example on real data: PAPILA (Kovalyk et al, Sci Data 2022;9:291), fundus photographs and clinical data of
both eyes of 244 patients (figshare doi:10.6084/m9.figshare.14798004, GPL-3.0+). The data are not redistributed:
download PAPILA.zip from figshare and point --data at it (or at the extracted folder).

Two AI models are built from openly available image networks, each with a simple prespecified head:
  ResNet-50  ImageNet-trained convolutional network (timm resnet50.a1_in1k)
  DINOv2     self-supervised vision transformer (timm vit_base_patch14_dinov2.lvd142m)
The networks are frozen; each image is cropped to its central square (the photographs are centred on the optic
disc), resized to 224 x 224, and passed through the network once. Heads are fitted by 5-fold cross-validation
split by patient (both eyes of a patient in the same fold), repeated 5 times with different splits and averaged,
so every eye is scored by models that never saw that patient.

  Classifier   glaucoma or glaucoma suspect vs healthy (per-eye diagnosis): standardise, 50 principal components,
               logistic regression. Clinical comparator: logistic regression on the vertical cup-to-disc ratio
               measured from the 2 experts' disc and cup outlines supplied with PAPILA (mean of the 2 experts).
  Continuous   axial length (mm) from the photograph: standardise, 50 principal components, ridge regression.

Each task is then analysed with ophthalmology-stats with the patient as the unit (default), with eyes treated as
independent, and with one random eye per patient; the one-eye analysis is repeated over 200 random choices of eye
(AUC and Brier score of the classifiers, MAE of axial length).

Usage (from this folder; Python 3.10+):
  pip install torch torchvision timm scikit-learn pillow openpyxl
  python papila_example.py --data "C:/path/to/PAPILA.zip" --pilot          # quick check (about 1 minute)
  python papila_example.py --data "C:/path/to/PAPILA.zip"                  # full run (about 5-10 minutes on CPU)
  python papila_example.py --from-predictions                              # statistics only, from saved predictions
The first run downloads the 2 network weights from Hugging Face (about 450 MB, cached afterwards).
Outputs in ./results/: papila_summary.json (the numbers for the paper), papila_*_results.xlsx (full output of
ophthalmology-stats), papila_lottery.csv, and papila_predictions.csv (per-eye predictions; derived from PAPILA, so
keep them local).
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "github", "ophthalmology-stats"))   # AI_Paper layout
sys.path.insert(0, os.path.join(HERE, "..", ".."))                              # repository layout (paper/realdata)
with contextlib.redirect_stdout(io.StringIO()):
    import ophthalmology_stats as osx  # noqa: E402

SEED = 20261026
NETWORKS = {"ResNet-50": "resnet50.a1_in1k", "DINOv2": "vit_base_patch14_dinov2.lvd142m"}
SIZE = 224
N_PC = 50
FOLDS, REPEATS = 5, 5
LOTTERY = 200
LOTTERY_METRICS = {"binary": ["auc", "brier"], "continuous": ["mae"]}   # metrics re-tested for each choice of eye


# ----------------------------------------------------------------------------------------------- data
def locate(data):
    """Return the folder that holds ClinicalData, FundusImages and ExpertsSegmentations (extracting a zip once)."""
    if data.lower().endswith(".zip"):
        dest = os.path.splitext(data)[0]
        if not os.path.isdir(dest):
            print(f"extracting {data} ...")
            with zipfile.ZipFile(data) as z:
                z.extractall(dest)
        data = dest
    for root, dirs, _ in os.walk(data):
        if "ClinicalData" in dirs and "FundusImages" in dirs:
            return root
    raise SystemExit(f"could not find ClinicalData and FundusImages under {data}")


def clinical(root):
    """One row per eye from patient_data_od.xlsx and patient_data_os.xlsx (3 header rows; columns by position)."""
    cols = ["id", "age", "sex", "diagnosis", "sphere", "cylinder", "axis", "pseudophakic", "iop_pneumatic",
            "iop_perkins", "pachymetry", "axial_length", "vf_md"]
    rows = []
    for eye, f in (("R", "patient_data_od.xlsx"), ("L", "patient_data_os.xlsx")):
        raw = pd.read_excel(os.path.join(root, "ClinicalData", f), header=None)
        head = " ".join(str(x) for x in raw.iloc[:3].values.ravel())
        for need in ("Diagnosis", "Axial_Length", "Phakic"):
            if need not in head:
                raise SystemExit(f"unexpected layout in {f}: '{need}' not found in the header rows")
        d = raw.iloc[3:, :13].copy()
        d.columns = cols
        d = d[d.id.notna()]
        d["eye"] = eye
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["patient_id"] = d.id.astype(str).str.extract(r"(\d+)")[0].astype(int)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["image"] = [os.path.join(root, "FundusImages", f"RET{p:03d}{'OD' if e == 'R' else 'OS'}.jpg")
                  for p, e in zip(d.patient_id, d.eye)]
    d = d[[os.path.exists(p) for p in d.image]].copy()
    d["truth"] = d.diagnosis.isin([1, 2]).astype(int)          # glaucoma or suspect vs healthy
    d["cdr"] = [vertical_cdr(root, p, e) for p, e in zip(d.patient_id, d.eye)]
    return d.sort_values(["patient_id", "eye"]).reset_index(drop=True)


def _contour(path):
    xy = np.loadtxt(path)
    return xy[:, 1].max() - xy[:, 1].min()                      # vertical extent (second column = y)


def vertical_cdr(root, pid, eye):
    """Vertical cup-to-disc ratio from the experts' outlines, averaged over the experts available."""
    folder = os.path.join(root, "ExpertsSegmentations", "Contours")
    tag = f"RET{pid:03d}{'OD' if eye == 'R' else 'OS'}"
    vals = []
    for ex in (1, 2):
        cup, disc = (os.path.join(folder, f"{tag}_{s}_exp{ex}.txt") for s in ("cup", "disc"))
        if os.path.exists(cup) and os.path.exists(disc):
            vals.append(_contour(cup) / _contour(disc))
    return float(np.mean(vals)) if vals else np.nan


# ----------------------------------------------------------------------------------------------- features
def features(name, model_id, paths, threads, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == [os.path.basename(p) for p in paths]:
            return z["X"]
    import torch
    import timm
    from PIL import Image
    torch.set_num_threads(threads)
    kw = {"img_size": SIZE} if "dinov2" in model_id else {}
    m = timm.create_model(model_id, pretrained=True, num_classes=0, **kw).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean, std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
    out, t0 = [], time.time()
    for i in range(0, len(paths), 16):
        batch = []
        for p in paths[i:i + 16]:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE),
                                                                                              Image.BICUBIC)
            batch.append(((np.asarray(im, np.float32) / 255 - mean) / std).transpose(2, 0, 1))
        with torch.no_grad():
            out.append(m(torch.from_numpy(np.stack(batch))).numpy())
    X = np.vstack(out)
    print(f"  {name}: {X.shape[1]} features for {len(paths)} images in {time.time() - t0:.0f} s")
    np.savez(cache, X=X, paths=np.array([os.path.basename(p) for p in paths]))
    return X


# ----------------------------------------------------------------------------------------------- models
def cross_fitted(X, y, groups, kind, repeats, seed):
    """Out-of-fold predictions from patient-split cross-validation, averaged over repeated splits."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make(n_train):
        k = min(N_PC, X.shape[1], n_train - 1)
        if kind == "classifier":
            return make_pipeline(StandardScaler(), PCA(k, random_state=0),
                                 LogisticRegression(C=0.5, max_iter=5000))
        if kind == "cdr":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        return make_pipeline(StandardScaler(), PCA(k, random_state=0),
                             RidgeCV(alphas=np.logspace(-1, 5, 31)))
    strat = y if kind != "regression" else pd.qcut(y, 5, labels=False, duplicates="drop")
    pred = np.zeros((repeats, len(y)))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, strat, groups):
            m = make(len(tr)).fit(X[tr], y[tr])
            pred[r, te] = m.predict_proba(X[te])[:, 1] if kind != "regression" else m.predict(X[te])
    return pred.mean(0)


# ----------------------------------------------------------------------------------------------- analyses
def compare(df, task, models, B, **extra):
    st = {"bootstrap_iterations": B, "verbose": False, "seed": SEED, **extra}
    w, l, info = osx.compare_models(df, task=task, models=models, patient="patient_id", truth="truth", settings=st)
    return l, info


def lottery(df, task, models, metrics, draws, B, seed):
    """Keep one random eye per bilateral patient, many times: how often does the conclusion change?"""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(draws):
        keep = df.assign(_r=rng.random(len(df))).sort_values("_r").drop_duplicates("patient_id").drop(columns="_r")
        st = {"bootstrap_iterations": B, "verbose": False, "seed": int(rng.integers(2 ** 31)), "metrics": list(metrics),
              "report_ci": False}
        l = osx.compare_models(keep, task=task, models=models, patient="patient_id", truth="truth", settings=st)[1]
        for r in l.itertuples():
            rows.append({"task": task, "draw": k, "metric": r.metric, "model_a": r.model_a, "model_b": r.model_b,
                         "est_a": r.est_a, "est_b": r.est_b, "diff": r.diff, "p": r.p})
    return pd.DataFrame(rows)


def summarise(task, models, l_pat, info, l_eye, l_one, lot, metric_main):
    out = {"n_eyes": info["n_eyes"], "n_patients": info["n_patients"], "n_bilateral": info["n_bilateral"],
           "models": {}, "pairs": []}
    for r in info["model_performance"]:
        out["models"].setdefault(r["model"], {})[r["metric"]] = [r["estimate"], r["ci_low"], r["ci_high"]]
    key = ["model_a", "model_b", "metric"]
    e = l_eye.set_index(key)
    o = l_one.set_index(key)
    for r in l_pat.itertuples():
        k = (r.model_a, r.model_b, r.metric)
        lt = lot[(lot.model_a == r.model_a) & (lot.model_b == r.model_b) & (lot.metric == r.metric)]
        out["pairs"].append({
            "metric": r.metric, "model_a": r.model_a, "model_b": r.model_b, "diff": r.diff,
            "ci": [r.ci_low, r.ci_high], "p_patient": r.p, "p_adj_patient": r.p_adj, "rho_delta": r.rho_delta,
            "ci_adj": [getattr(r, "ci_adj_low", np.nan), getattr(r, "ci_adj_high", np.nan)],
            "one_eye_factor": r.one_eye_factor,
            "ci_eyes_independent": [e.loc[k, "ci_low"], e.loc[k, "ci_high"]] if k in e.index else None,
            "p_eyes_independent": float(e.loc[k, "p"]) if k in e.index else None,
            "p_adj_eyes_independent": float(e.loc[k, "p_adj"]) if k in e.index else None,
            "p_one_eye": float(o.loc[k, "p"]) if k in o.index else None,
            "ci_one_eye": [o.loc[k, "ci_low"], o.loc[k, "ci_high"]] if k in o.index else None,
            "lottery_frac_significant": float((lt.p < 0.05).mean()) if len(lt) else None,
            "lottery_p_range": [float(lt.p.min()), float(lt.p.max())] if len(lt) else None})
    if len(lot):
        best = pd.Series([_top(g, metric_main) for _, g in lot[lot.metric == metric_main].groupby("draw")])
        out["lottery_top_model_counts"] = {k: int(v) for k, v in best.value_counts().items()}
    return out


def _top(g, metric):
    est = {}
    for r in g.itertuples():
        est[r.model_a], est[r.model_b] = r.est_a, r.est_b
    return (max if metric == "auc" else min)(est, key=est.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="PAPILA.zip from figshare, or the extracted folder")
    ap.add_argument("--from-predictions", action="store_true",
                    help="re-run only the statistics from results/papila_predictions.csv (no images needed)")
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pilot", action="store_true", help="30 patients, few resamples: checks that everything runs")
    a = ap.parse_args()
    res = os.path.join(HERE, "results_pilot" if a.pilot else "results")
    os.makedirs(res, exist_ok=True)
    if a.from_predictions:
        return analyse_predictions(res)
    if not a.data:
        ap.error("--data is required (or use --from-predictions)")
    root = locate(a.data)
    d = clinical(root)
    if a.pilot:                                                 # 30 random patients, half with glaucoma or suspect
        pt = d.groupby("patient_id").truth.max()
        rng = np.random.default_rng(SEED)
        pick = np.r_[rng.choice(pt.index[pt == 1], 15, replace=False), rng.choice(pt.index[pt == 0], 15, replace=False)]
        d = d[d.patient_id.isin(pick)].reset_index(drop=True)
    B, draws, repeats = (199, 10, 1) if a.pilot else (4999, LOTTERY, REPEATS)
    print(f"PAPILA: {d.patient_id.nunique()} patients, {len(d)} eyes; glaucoma or suspect {int(d.truth.sum())} eyes")

    feats = {nm: features(nm, mid, list(d.image), a.cores, os.path.join(res, f"features_{nm}.npz"))
             for nm, mid in NETWORKS.items()}
    g = d.patient_id.values

    # classifier: glaucoma or suspect vs healthy
    cls = d[["patient_id", "eye", "truth"]].copy()
    for nm, X in feats.items():
        cls[nm] = cross_fitted(X, d.truth.values, g, "classifier", repeats, SEED)
    ok = d.cdr.notna().values
    cls.loc[ok, "Cup-to-disc ratio"] = cross_fitted(d.loc[ok, ["cdr"]].values, d.truth.values[ok], g[ok], "cdr",
                                                    repeats, SEED)
    cls = cls.dropna().reset_index(drop=True)
    m_cls = list(NETWORKS) + ["Cup-to-disc ratio"]

    # continuous: axial length from the photograph
    al = d[d.axial_length.notna()].reset_index(drop=True)
    reg = al[["patient_id", "eye"]].assign(truth=al.axial_length.values)
    for nm, X in feats.items():
        reg[nm] = cross_fitted(X[d.axial_length.notna().values], al.axial_length.values, al.patient_id.values,
                               "regression", repeats, SEED)
    m_reg = list(NETWORKS)

    cls.merge(reg, on=["patient_id", "eye"], how="outer", suffixes=("_glaucoma", "_axial_length")).to_csv(
        os.path.join(res, "papila_predictions.csv"), index=False)
    summary = {"dataset": {"patients": int(d.patient_id.nunique()), "eyes": int(len(d)),
                           "glaucoma_eyes": int((d.diagnosis == 1).sum()), "suspect_eyes": int((d.diagnosis == 2).sum()),
                           "healthy_eyes": int((d.diagnosis == 0).sum()),
                           "patients_discordant_diagnosis": int(d.groupby("patient_id").truth.nunique().gt(1).sum()),
                           "fellow_eye_r_axial_length": float(al.pivot(index="patient_id", columns="eye",
                                                                       values="axial_length").corr().iloc[0, 1])}}
    analyse(cls, m_cls, reg, m_reg, summary, res, B, draws)


def analyse_predictions(res):
    """Statistics only, from the saved out-of-fold predictions (the dataset summary is kept from the full run)."""
    pr = pd.read_csv(os.path.join(res, "papila_predictions.csv"))
    nets = list(NETWORKS)
    cls = pr[["patient_id", "eye", "truth_glaucoma"] + [f"{m}_glaucoma" for m in nets] + ["Cup-to-disc ratio"]].dropna()
    cls = cls.rename(columns={"truth_glaucoma": "truth", **{f"{m}_glaucoma": m for m in nets}}).reset_index(drop=True)
    reg = pr[["patient_id", "eye", "truth_axial_length"] + [f"{m}_axial_length" for m in nets]].dropna()
    reg = reg.rename(columns={"truth_axial_length": "truth", **{f"{m}_axial_length": m for m in nets}}).reset_index(drop=True)
    with open(os.path.join(res, "papila_summary.json")) as f:
        summary = {"dataset": json.load(f)["dataset"]}
    analyse(cls, nets + ["Cup-to-disc ratio"], reg, nets, summary, res, 4999, LOTTERY)


def analyse(cls, m_cls, reg, m_reg, summary, res, B, draws):
    for task, df, models, main_metric in (("binary", cls, m_cls, "auc"), ("continuous", reg, m_reg, "mae")):
        name = "glaucoma" if task == "binary" else "axial_length"
        print(f"\n{name}: {df.patient_id.nunique()} patients, {len(df)} eyes")
        l_pat, info = compare(df, task, models, B)
        l_eye, _ = compare(df, task, models, B, unit="eye")
        l_one, _ = compare(df, task, models, B, one_eye=True)
        lot = lottery(df, task, models, LOTTERY_METRICS[task], draws, min(B, 999), SEED + 1)
        lot.to_csv(os.path.join(res, f"papila_lottery_{name}.csv"), index=False)
        summary[name] = summarise(task, models, l_pat, info, l_eye, l_one, lot, main_metric)
        r = osx.analyze(df, task=task, models=models, patient="patient_id", truth="truth", subgroup="eye",
                        settings={"bootstrap_iterations": B, "verbose": False, "seed": SEED})
        osx.write_results_to_excel(r, os.path.join(res, f"papila_{name}_results.xlsx"))
        for p in summary[name]["pairs"]:
            if p["metric"] == main_metric:
                print(f"  {p['model_a']} vs {p['model_b']}: diff {p['diff']:.3f}; P patient {p['p_patient']:.4f}, "
                      f"eyes independent {p['p_eyes_independent']:.4f}, one eye {p['p_one_eye']:.4f}; "
                      f"one-eye draws significant {p['lottery_frac_significant']:.0%}")
    with open(os.path.join(res, "papila_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nwritten to {res}")


if __name__ == "__main__":
    main()
