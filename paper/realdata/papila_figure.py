"""Figure 4: the PAPILA comparisons analysed with the patient as the unit, with eyes treated as
independent, and with one eye per patient over 200 random choices of eye.

Run after papila_example.py:  python papila_figure.py [--results results]
Writes figures/Figure4.tiff (600 dpi, LZW, RGB) and figures/Figure4.png.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"cluster": "#2a78d6", "naive": "#eb6834", "one_eye": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False})
FLOOR = 1e-3                          # smallest P shown (bootstrap P values with 999 resamples start at 0.001)
NAMES = {"Cup-to-disc ratio": "CDR"}


def rows(res):
    s = json.load(open(os.path.join(res, "papila_summary.json")))
    out = []
    for task, metric, lab in (("glaucoma", "auc", "AUC"), ("glaucoma", "brier", "Brier score"),
                              ("axial_length", "mae", "Axial length, MAE")):
        lot = pd.read_csv(os.path.join(res, f"papila_lottery_{task}.csv"))
        if metric not in set(lot.metric):
            continue
        for p in s[task]["pairs"]:
            if p["metric"] != metric:
                continue
            lt = lot[(lot.model_a == p["model_a"]) & (lot.model_b == p["model_b"]) & (lot.metric == metric)]
            a, b = (NAMES.get(m, m) for m in (p["model_a"], p["model_b"]))
            out.append({"label": f"{a} vs {b}\n{lab}", "one_eye": lt.p.values, "cluster": p["p_patient"],
                        "naive": p["p_eyes_independent"], "frac": (lt.p < 0.05).mean()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    R = rows(a.results)
    fig, ax = plt.subplots(figsize=(7.0, 0.62 * len(R) + 0.9))
    rng = np.random.default_rng(1)
    for i, r in enumerate(R):
        y = len(R) - 1 - i
        x = np.clip(r["one_eye"], FLOOR, 1)
        ax.scatter(x, y + rng.uniform(-0.17, 0.17, len(x)), s=7, color=COL["one_eye"], alpha=0.55, lw=0,
                   zorder=2, label="One eye per patient (each of 200 random choices)" if i == 0 else None)
        ax.scatter(max(r["naive"], FLOOR), y, marker="s", s=34, color=COL["naive"], edgecolor="white", lw=0.8,
                   zorder=4, label="Eyes treated as independent" if i == 0 else None)
        ax.scatter(max(r["cluster"], FLOOR), y, marker="D", s=34, color=COL["cluster"], edgecolor="white", lw=0.8,
                   zorder=5, label="Patient as the unit" if i == 0 else None)
        ax.text(1.6, y, f"{100 * r['frac']:.0f}%", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.text(1.6, len(R) - 0.45, "One eye:\nP < 0.05", va="bottom", ha="left", fontsize=7, color=INK2)
    ax.axvline(0.05, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.7, 1.15)
    ax.set_xticks([1e-3, 1e-2, 0.05, 0.2, 1])
    ax.set_xticklabels(["\u22640.001", "0.01", "0.05", "0.2", "1"])
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([r["label"] for r in R][::-1])
    ax.set_ylim(-0.6, len(R) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("P value for the difference between the 2 models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.08 - 0.5 / len(R)), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "Figure4.png"), dpi=200, bbox_inches="tight")
    tif = os.path.join(a.out, "Figure4.tiff")
    fig.savefig(tif, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    from PIL import Image
    Image.open(tif).convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
    print("wrote", os.path.join(a.out, "Figure4.tiff"))


if __name__ == "__main__":
    main()
