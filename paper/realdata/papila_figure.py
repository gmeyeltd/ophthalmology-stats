"""Figure 5: the PAPILA worked example.

Panel A: P values for each pair of models analysed with the patient as the unit and with fellow eyes treated as
independent. Panel B: P values with one randomly chosen eye per patient, over 200 random choices of eye.

Run after papila_example.py:  python papila_figure.py [--results results]
Writes figures/Figure5.tiff (600 dpi, LZW, RGB) and figures/Figure5.png.
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
COL = {"cluster": "#000000", "naive": "#B2182B", "one_eye": "#2166AC"}
INK, INK2, GRID = "#000000", "#404040", "#e3e3e3"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.color": INK,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.facecolor": "white"})
FLOOR = 1e-3                          # smallest P shown (bootstrap P values with 999 resamples start at 0.001)
NAMES = {"Cup-to-disc ratio": "CDR"}
METRICS = (("auc", "Area under the ROC curve"), ("brier", "Brier score"))


def rows(res):
    s = json.load(open(os.path.join(res, "papila_summary.json")))
    lot = pd.read_csv(os.path.join(res, "papila_lottery_glaucoma.csv"))
    out = []
    for metric, title in METRICS:
        out.append({"header": title})
        for p in s["glaucoma"]["pairs"]:
            if p["metric"] != metric:
                continue
            lt = lot[(lot.model_a == p["model_a"]) & (lot.model_b == p["model_b"]) & (lot.metric == metric)]
            a, b = (NAMES.get(m, m) for m in (p["model_a"], p["model_b"]))
            rho = p.get("rho_delta")
            lab = f"{a} vs {b}"
            if rho is not None and np.isfinite(rho):
                lab += f"  ($\\rho_\\Delta$ {max(rho, 0):.2f})"
            out.append({"label": lab, "one_eye": lt.p.values, "cluster": p["p_patient"],
                        "naive": p["p_eyes_independent"], "frac": (lt.p < 0.05).mean()})
    return out


def _axis(ax):
    ax.axvline(0.05, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.7, 1.6)
    ax.set_xticks([1e-3, 1e-2, 0.05, 0.2, 1])
    ax.set_xticklabels(["≤0.001", "0.01", "0.05", "0.2", "1"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("P Value for the Difference Between Two Classifiers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    R = rows(a.results)
    n = len(R)
    ys = np.arange(n)[::-1].astype(float)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(1)
    for y, r in zip(ys, R):
        if "header" in r:
            continue
        pc, pn = max(r["cluster"], FLOOR), max(r["naive"], FLOOR)
        # A: the same comparison, analysed correctly and with fellow eyes treated as independent
        if pn < pc * 0.9:
            axA.annotate("", xy=(pn * 1.08, y), xytext=(pc, y),
                         arrowprops=dict(arrowstyle="-|>", color=COL["naive"], lw=0.9, mutation_scale=7,
                                         shrinkA=3, shrinkB=0))
        axA.scatter(pn, y, marker="s", s=30, color=COL["naive"], edgecolor="white", lw=0.6, zorder=4)
        axA.scatter(pc, y, marker="o", s=30, color=COL["cluster"], edgecolor="white", lw=0.6, zorder=5)
        # B: one random eye per patient, 200 times
        x = np.clip(r["one_eye"], FLOOR, 1)
        axB.scatter(x, y + rng.uniform(-0.2, 0.2, len(x)), s=6, color=COL["one_eye"], alpha=0.5, lw=0, zorder=2)
        axB.scatter(pc, y, marker="o", s=30, color=COL["cluster"], edgecolor="white", lw=0.6, zorder=5)
        axB.text(2.0, y, f"{100 * r['frac']:.0f}%", va="center", ha="left", fontsize=7.5)
    for ax in (axA, axB):
        _axis(ax)
    axA.set_yticks([y for y, r in zip(ys, R) if "header" not in r])
    axA.set_yticklabels([r["label"] for r in R if "header" not in r])
    for y, r in zip(ys, R):
        if "header" in r:
            axA.text(-0.02, y, r["header"], transform=axA.get_yaxis_transform(), ha="right", va="center",
                     fontsize=7.6, fontweight="bold")
    axA.set_ylim(-0.6, n - 0.4)
    yb = next(y for y, r in zip(ys, R) if r.get("header") == "Brier score")
    axA.text(0.0011, yb, "The gap widens as fellow eyes\nbecome more alike (higher $\\rho_\\Delta$)",
             ha="left", va="center", fontsize=7, color=COL["naive"])
    axB.text(2.0, n - 0.75, "Eye choices\nwith P < 0.05", va="center", ha="left", fontsize=7, fontweight="bold")
    axA.set_title("Treating fellow eyes as independent\nmakes P values too small", loc="left", fontweight="bold")
    axB.set_title("With one eye per patient, the P value\ndepends on which eye was kept", loc="left",
                  fontweight="bold")
    for ax, lab in ((axA, "A"), (axB, "B")):
        ax.text(-0.02 if ax is axB else -0.62, 1.03, lab, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="right" if ax is axB else "left")
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], marker="o", ls="", ms=5, color=COL["cluster"]),
            Line2D([], [], marker="s", ls="", ms=5, color=COL["naive"]),
            Line2D([], [], marker="o", ls="", ms=3, color=COL["one_eye"], alpha=0.6)]
    fig.legend(hand, ["Both eyes, patient as the unit (valid analysis)",
                      "Both eyes treated as independent (P falsely small)",
                      "One random eye per patient (each of 200 choices)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.45, -0.17), handletextpad=0.3,
               columnspacing=1.5)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "Figure5.png"), dpi=200, bbox_inches="tight")
    tif = os.path.join(a.out, "Figure5.tiff")
    fig.savefig(tif, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    from PIL import Image
    Image.open(tif).convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
    print("wrote", os.path.join(a.out, "Figure5.tiff"))


if __name__ == "__main__":
    main()
