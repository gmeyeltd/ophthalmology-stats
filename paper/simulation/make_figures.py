"""
Figures and table data for the manuscript (run after analyse_simulation.py and, for Figure 4,
audit/audit_03_validation.py --score).

  Figure 1  Flowchart: how the second eye enters a study (study type, laterality design, analysis), with the
            headline simulation results.
  Figure 2  Headline: type I error of an eye-level analysis and extra patients needed with one eye, for any
            rho_delta and proportion bilateral (design effect), with the simulation's observed-vs-predicted check.
  Figure S3 Type I error against rho_delta by strategy and metric (all simulated settings).
  Figure 3  Power against the number of patients (sweep), and the 'one-eye lottery'.
  Figure 4  Survey of published studies: weighted estimates with 95% intervals.
  Figure S2 Variance of the estimated difference relative to the eye-weighted cluster analysis, with the
            analytic curves (one eye per patient, patient-averaged, minimum-variance weights).

Writes figures/Figure1-3.tiff and FigureS2-S6.tiff (600 dpi, LZW) with .png previews (Figure 4 and Supplementary
Figure S1 come from realdata/papila_figure.py and manuscript/build_supplement.py), and results/table1_simulation.csv,
results/table_s_type1_by_metric.csv.

Usage (from this folder):  python make_figures.py
"""
import os
from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
AUDIT = os.path.join(HERE, "..", "audit", "audit_data", "validation_results.csv")
LOTTERY = os.path.join(HERE, "output_suite", "lottery")
PAPILA = os.path.join(HERE, "..", "realdata", "results", "papila_summary.json")   # worked example (papila_example.py)

# colour follows the strategy in every figure (categorical slots validated for these combinations)
COL = {"cluster": "#2a78d6", "naive": "#eb6834", "one_eye": "#1baf7a", "patient": "#eda100", "optimal": "#6250d6"}
MARK = {"cluster": "o", "naive": "s", "one_eye": "^", "patient": "D", "optimal": "v"}
LAB = {"cluster": "Patient as the unit", "naive": "Eyes treated as independent",
       "one_eye": "One eye per patient", "patient": "Patient-averaged", "optimal": "Minimum-variance weights"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
MEAN_METRICS = ["bias", "mae", "rmse", "brier", "logloss"]
PROP_METRICS = ["within", "sensitivity", "specificity", "accuracy"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def naive_pred(f, rho):
    return 2 * (1 - phi(1.959964 / sqrt(1 + 2 * f * rho / (1 + f))))


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200, bbox_inches="tight")
    tif = os.path.join(FIG, f"{name}.tiff")
    fig.savefig(tif, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    from PIL import Image                     # journals want RGB TIFF without an alpha channel
    Image.open(tif).convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
    print("wrote", name)


def load_main():
    s = pd.read_csv(os.path.join(RES, "summary_main.csv"))
    s["shape"] = s["shape"].fillna("")
    return s


def figure_flow(s):
    """Figure 1: where clustering of fellow eyes arises, and what to do about it (after T. Julian's distinction
    between AI used as a measuring tool and AI models compared for accuracy)."""
    from matplotlib.patches import FancyBboxPatch
    h = s[(s.scen == "H0") & (s.metric != "medae")]
    naive_max = h[h.strategy == "naive"].reject.max()
    sw = pd.read_csv(os.path.join(RES, "summary_sweep.csv"))
    extra = []
    for task, met in (("continuous", "mae"), ("binary", "auc")):
        d = sw[(sw.task == task) & (sw.metric == met) & (sw.scen == "H1") & (sw.frac == 0.8)]
        n80 = {}
        for st in ("cluster", "one_eye"):
            x = d[d.strategy == st].sort_values("n_patients")
            nn, pw = x.n_patients.values, x.reject.values
            i = int(np.argmax(pw >= 0.8))
            n80[st] = nn[i - 1] + (0.8 - pw[i - 1]) * (nn[i] - nn[i - 1]) / (pw[i] - pw[i - 1])
        extra.append(n80["one_eye"] / n80["cluster"] - 1)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, hgt, head=None, body="", edge=INK2, lw=0.8, fs=7.0):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - hgt / 2), w, hgt, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc="white", ec=edge, lw=lw))
        if head:
            ax.text(x, y + hgt / 2 - 0.022, head, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
            ax.text(x, y + hgt / 2 - 0.022 - 0.034 * (head.count("\n") + 1), body, ha="center", va="top",
                    fontsize=fs - 0.3, color=INK, linespacing=1.3)
        else:
            ax.text(x, y, body, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color=INK2, dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=0,
                                    linestyle=(0, (3, 2)) if dashed else "-"))

    xl, xs, wcol = 0.135, (0.40, 0.637, 0.874), 0.228
    box(0.5, 0.955, 0.62, 0.06, body="AI applied to eye-level data, with both eyes of some patients", lw=1.0, fs=8)
    box(xl, 0.80, 0.25, 0.13, head="AI as a measuring tool",
        body="segmentation, feature or\nrepresentation, then related\nto an exposure or outcome")
    box(0.635, 0.80, 0.675, 0.13, head="Comparing the accuracy of models",
        body="classifiers or continuous outputs;\nlaterality is often fixed by the model's design")
    arrow(0.30, 0.925, xl, 0.865); arrow(0.62, 0.925, 0.635, 0.865)
    box(xl, 0.50, 0.25, 0.30, head="Association analysis",
        body="features from both eyes:\nregression with the patient\nas the unit (e.g. GEE);\n"
             "laterality-agnostic features\nlet both eyes be used")
    arrow(xl, 0.735, xl, 0.65)
    heads = ("One output per patient\nfrom both eyes", "One eye per patient\n(by design, or discarded)",
             "Each eye scored separately\n(left eyes often flipped\nto right orientation)")
    for x, t in zip(xs, heads):
        box(x, 0.615, wcol, 0.11, body=t)
        arrow(0.635, 0.735, x, 0.67)
    res = [("No clustering of fellow\neyes in the test set:\nstandard paired tests", INK2),
           (f"Valid, but needs more\npatients (+{100 * min(extra):.0f}% to +{100 * max(extra):.0f}% for\n80% "
            "power); the result\ncan depend on the eye kept", COL["one_eye"]),
           ("Fellow eyes correlated:\npatient as the unit\n(ophthalmology-stats);\nreport $\\rho_\\Delta$; compare\n"
            "right and left eyes", COL["cluster"])]
    for x, (t, c) in zip(xs, res):
        box(x, 0.385, wcol, 0.20, body=t, edge=c, lw=1.4)
        arrow(x, 0.56, x, 0.485)
    box(xs[2], 0.135, wcol, 0.15, body=f"Eyes treated as\nindependent (e.g. DeLong,\neye-level bootstrap):\n"
        f"type I error up to {100 * naive_max:.0f}%", edge=COL["naive"], lw=1.4)
    arrow(xs[2], 0.285, xs[2], 0.21, color=COL["naive"], dashed=True)
    ax.text(xs[2] + 0.012, 0.248, "if ignored", fontsize=6.5, color=COL["naive"], va="center")
    ax.text(0.02, 0.03, "AI = artificial intelligence; GEE = generalised estimating equations; $\\rho_\\Delta$ = correlation "
            "between fellow eyes of the difference in error between 2 models.", fontsize=6.3, color=INK2, ha="left")
    save(fig, "Figure1")


def figure1(s):
    h = s[(s.scen == "H0") & s.rho_delta.notna() & (s.metric != "medae")].copy()
    h["rho"] = h.rho_delta.clip(lower=0)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, sharey=True)
    mc = 1.959964 * sqrt(0.05 * 0.95 / 2000)
    rr = np.linspace(0, 1, 101)
    for i, task in enumerate(("continuous", "binary")):
        for j, f in enumerate((0.4, 0.8)):
            ax = axes[i, j]
            d = h[(h.task == task) & (h.frac == f)]
            ax.axhspan(100 * (0.05 - mc), 100 * (0.05 + mc), color=BAND, lw=0, zorder=0)
            ax.axhline(5, color=INK2, lw=0.6, zorder=1)
            ax.plot(rr, [100 * naive_pred(f, r) for r in rr], color=INK2, lw=1.2, zorder=2,
                    label="Predicted from the design effect, eyes treated as independent")
            for st in ("naive", "one_eye", "cluster"):
                x = d[d.strategy == st]
                prop = x.metric.isin(PROP_METRICS)
                ax.scatter(x.rho[~prop], 100 * x.reject[~prop], s=14, marker=MARK[st], color=COL[st], edgecolor="white",
                           linewidth=0.5, zorder=3, label=LAB[st])
                ax.scatter(x.rho[prop], 100 * x.reject[prop], s=14, marker=MARK[st], facecolor="none", edgecolor=COL[st],
                           linewidth=0.8, zorder=3)
            grid(ax)
            ax.set_ylim(0, 20)
            ax.set_xlim(-0.02, 1.0)
            title = "Continuous outputs" if task == "continuous" else "Classifiers"
            ax.set_title(f"{title}, {int(f * 100)}% of patients bilateral", loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("Type I Error (%)")
            if i == 1:
                ax.set_xlabel(r"Fellow-Eye Correlation of the Difference in Error, $\rho_\Delta$")
            panel_label(ax, "ABCD"[2 * i + j])
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    h_.append(plt.Line2D([], [], ls="none", marker="o", markersize=4, markerfacecolor="none", markeredgecolor=INK2))
    l_.append("Open symbols: proportions (single-eye analyses use McNemar's test with continuity correction)")
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05), handletextpad=0.4,
               markerscale=1.6)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "FigureS2")



def figure_headline(s):
    """Figure 2: the cost of the 2 shortcuts for any study, from the design effect (Kerry & Bland; Eldridge),
    with the simulated settings and the observed-versus-predicted check."""
    from matplotlib.colors import LinearSegmentedColormap
    rr = np.linspace(0, 1, 301)
    ff = np.linspace(0, 1, 301)
    R, F = np.meshgrid(rr, ff)
    T1 = np.vectorize(naive_pred)(F, R) * 100
    EX = ((1 + F) ** 2 / (1 + F + 2 * F * R) - 1) * 100
    h = s[(s.scen == "H0") & s.metric.isin(MEAN_METRICS) & s.rho_delta.notna()].copy()
    h["rho"] = h.rho_delta.clip(lower=0)
    pts = h[h.strategy == "cluster"].drop_duplicates(["cell", "metric"])

    cmap_a = LinearSegmentedColormap.from_list("naive", ["#fdf0ea", COL["naive"], "#7a2a0b"])
    cmap_b = LinearSegmentedColormap.from_list("oneeye", ["#e8f7f0", COL["one_eye"], "#0a5139"])
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = fig.add_gridspec(1, 2, wspace=0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    im = axA.pcolormesh(R, F * 100, T1, cmap=cmap_a, vmin=5, vmax=17, shading="auto", rasterized=True)
    from scipy.stats import norm as _norm
    lev_a = [6, 8, 10, 12, 14]
    cs = axA.contour(R, F * 100, T1, levels=lev_a, colors=INK, linewidths=0.5)
    pos = []
    for L, f in zip(lev_a, (0.30, 0.45, 0.55, 0.65, 0.68)):     # label each line inside the panel
        de = (1.959964 / _norm.ppf(1 - L / 200)) ** 2
        pos.append(((de - 1) * (1 + f) / (2 * f), f * 100))
    axA.clabel(cs, fmt="%d%%", fontsize=6.5, inline_spacing=2, manual=pos)
    cb = fig.colorbar(im, ax=axA, fraction=0.05, pad=0.03)
    cb.set_label("Type I Error (%)")
    cb.outline.set_linewidth(0.4)
    axA.set_title("Eyes treated as independent", loc="left", color=INK)

    EXm = np.ma.masked_less(EX, 0)
    axB.pcolormesh(R, F * 100, np.where(EX < 0, 1, np.nan), cmap=LinearSegmentedColormap.from_list("g", [BAND, BAND]),
                   shading="auto", rasterized=True)
    im = axB.pcolormesh(R, F * 100, EXm, cmap=cmap_b, vmin=0, vmax=100, shading="auto", rasterized=True)
    lev_b = [10, 25, 50, 75]
    cs = axB.contour(R, F * 100, EX, levels=lev_b, colors=INK, linewidths=0.5)
    pos = []
    for E_, f in zip(lev_b, (0.50, 0.60, 0.72, 0.93)):
        pos.append((((1 + f) ** 2 / (1 + E_ / 100) - (1 + f)) / (2 * f), f * 100))
    axB.clabel(cs, fmt="+%d%%", fontsize=6.5, inline_spacing=2, manual=pos)
    axB.contour(R, F * 100, EX, levels=[0], colors=INK2, linewidths=0.6, linestyles="--")
    cb = fig.colorbar(im, ax=axB, fraction=0.05, pad=0.03)
    cb.set_label("Extra Patients Needed (%)")
    cb.outline.set_linewidth(0.4)
    axB.text(0.98, 0.03, "Grey: eye weighting\nless precise than\none eye per patient", ha="right", va="bottom",
             fontsize=6.3, color=INK2, transform=axB.transAxes)
    axB.set_title("One eye per patient", loc="left", color=INK)

    rd = papila_points()
    for ax in (axA, axB):
        ax.scatter(pts.rho, pts.f_obs * 100, s=7, color="white", edgecolor=INK, linewidth=0.5, zorder=3,
                   label="Simulated settings")
        if len(rd):
            ax.scatter(rd.rho, rd.f * 100, s=42, marker="*", color=INK, edgecolor="white", linewidth=0.4, zorder=5,
                       clip_on=False, label="PAPILA comparisons")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel(r"Fellow-Eye Correlation of the Difference, $\rho_\Delta$")
        ax.set_ylabel("Patients With Both Eyes (%)")

    for ax, lab in zip((axA, axB), "AB"):
        panel_label(ax, lab)
    if len(rd):
        h_, l_ = axA.get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=7)
    save(fig, "Figure3")


def papila_points():
    """Comparisons in the worked example with a defined rho_delta (negative values shown at 0, as for the simulation)."""
    import json
    if not os.path.exists(PAPILA):
        return pd.DataFrame(columns=["rho", "f"])
    d = json.load(open(PAPILA))
    rows = []
    for task in ("glaucoma", "axial_length"):
        g = d[task]
        f = g["n_bilateral"] / g["n_patients"]
        for p in g["pairs"]:
            if p["rho_delta"] is not None and p["rho_delta"] == p["rho_delta"]:
                rows.append({"rho": max(p["rho_delta"], 0.0), "f": f, "metric": p["metric"]})
    return pd.DataFrame(rows)

def figure_check(s):
    """Supplementary Figure S2: observed against predicted type I error and one-eye variance in the simulation."""
    h = s[(s.scen == "H0") & s.metric.isin(MEAN_METRICS) & s.rho_delta.notna()].copy()
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, wspace=0.38)
    axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    n = h[h.strategy == "naive"]
    c = h[h.strategy == "cluster"]
    lim = (0, 18)
    axC.plot(lim, lim, color=INK2, lw=0.6)
    axC.axhspan(4.0, 6.0, color=BAND, lw=0, zorder=0)
    axC.scatter(n.pred_naive_type1 * 100, n.reject * 100, s=12, marker=MARK["naive"], color=COL["naive"],
                edgecolor="white", linewidth=0.4, zorder=3, label="Eyes treated as independent")
    axC.scatter(c.pred_naive_type1 * 100, c.reject * 100, s=12, marker=MARK["cluster"], color=COL["cluster"],
                edgecolor="white", linewidth=0.4, zorder=3, label="Patient as the unit")
    axC.set_xlim(lim)
    axC.set_ylim(lim)
    axC.set_xlabel("Predicted Type I Error, Eyes Independent (%)")
    axC.set_ylabel("Observed Type I Error (%)")
    axC.legend(loc="upper left", frameon=False, handletextpad=0.3, markerscale=1.2)
    axC.set_title("Simulation check: type I error", loc="left", color=INK)

    o = h[h.strategy == "one_eye"]
    lim = (0.7, 1.95)
    axD.plot(lim, lim, color=INK2, lw=0.6)
    axD.scatter(o.pred_rel_var_one_eye, o.rel_var, s=12, marker=MARK["one_eye"], color=COL["one_eye"],
                edgecolor="white", linewidth=0.4, zorder=3)
    axD.set_xlim(lim)
    axD.set_ylim(lim)
    axD.set_xlabel("Predicted Variance Ratio, One Eye / Both Eyes")
    axD.set_ylabel("Observed Variance Ratio")
    axD.set_title("Simulation check: precision", loc="left", color=INK)
    for ax in (axC, axD):
        grid(ax, "both")
    for ax, lab in zip((axC, axD), "AB"):
        panel_label(ax, lab)
    save(fig, "FigureS3")


def figure2(s):
    h = s[(s.scen == "H0") & s.metric.isin(MEAN_METRICS) & s.rho_delta.notna()].copy()
    h["rho"] = h.rho_delta.clip(lower=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    rr = np.linspace(0, 1, 101)
    for j, f in enumerate((0.4, 0.8)):
        ax = axes[j]
        d = h[h.frac == f]
        de = 1 + f + 2 * f * rr
        curves = {"one_eye": (1 + f) ** 2 / de,
                  "patient": ((1 - f) + f * (1 + rr) / 2) * (1 + f) ** 2 / de,
                  "optimal": (1 + f) ** 2 / (de * ((1 - f) + 2 * f / (1 + rr)))}
        ax.axhline(1, color=INK2, lw=0.6)
        for st in ("one_eye", "patient", "optimal"):
            ax.plot(rr, curves[st], color=COL[st], lw=1.2, zorder=2)
            x = d[d.strategy == st]
            ax.scatter(x.rho, x.rel_var, s=13, marker=MARK[st], color=COL[st], edgecolor="white", linewidth=0.5,
                       zorder=3, label=LAB[st])
        grid(ax)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(0.6, 2.0 if f == 0.8 else 1.6)
        ax.set_title(f"{int(f * 100)}% of patients bilateral", loc="left")
        ax.set_xlabel(r"Fellow-Eye Correlation of the Difference in Error, $\rho_\Delta$")
        panel_label(ax, "AB"[j])
    axes[0].set_ylim(0.6, 2.0)
    axes[0].set_ylabel("Variance Relative to Patient as the\nUnit, Equal Eye Weights (1 = Equal)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), handletextpad=0.3,
               markerscale=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "FigureS4")


def figure3():
    """Figure 3: power against study size (patient as the unit vs one eye per patient)."""
    sw = pd.read_csv(os.path.join(RES, "summary_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for j, (task, metric, title) in enumerate((("continuous", "mae", "Mean absolute error"),
                                               ("binary", "auc", "Area under the curve"))):
        ax = axes[j]
        d = sw[(sw.task == task) & (sw.metric == metric) & (sw.scen == "H1") & (sw.frac == 0.8)]
        for st in ("cluster", "one_eye"):
            x = d[d.strategy == st].sort_values("n_patients")
            ax.plot(x.n_patients, 100 * x.reject, color=COL[st], lw=1.5, marker=MARK[st], ms=4, mec="white",
                    mew=0.5, label=LAB[st])
        ax.axhline(80, color=INK2, lw=0.6)
        grid(ax)
        ax.set_ylim(0, 102)
        ax.set_xlim(0, 1020)
        ax.set_xlabel("Patients per Study")
        if j == 0:
            ax.set_ylabel("Power (%)")
        ax.set_title(title, loc="left")
        panel_label(ax, "AB"[j])
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "FigureS5")


def figure_lottery():
    """Supplementary Figure S4: how often a one-eye study's significance depends on the eye kept."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for j, task in enumerate(("continuous", "binary")):
        ax = axes[j]
        d = pd.read_csv(os.path.join(LOTTERY, f"lottery_{task}.csv.gz"))
        bins = np.linspace(0, 1, 11)
        cnt, _ = np.histogram(d.one_eye_frac_sig, bins=bins)
        ax.bar(100 * (bins[:-1] + 0.05), 100 * cnt / len(d), width=10 - 1.2, color=COL["one_eye"], zorder=2)
        grid(ax)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(45, 100 * (cnt / len(d)).max() + 5))
        ax.set_xlabel("Choices of Eye Giving P < 0.05 (%)")
        if j == 0:
            ax.set_ylabel("Simulated Studies (%)")
        mid = ((d.one_eye_frac_sig >= 0.1) & (d.one_eye_frac_sig <= 0.9)).mean()
        chg = (d.one_eye_n_distinct_top > 1).mean()
        ax.text(0.03, 0.97, f"Significant with 10%-90% of choices: {mid:.0%} of studies\n"
                f"Top-ranked model changes with the eye: {chg:.0%}\n"
                f"Patient as the unit, significant: {d.cluster_sig.mean():.0%}",
                transform=ax.transAxes, va="top", fontsize=7, color=INK2)
        ax.set_title("Continuous outputs" if task == "continuous" else "Classifiers", loc="left")
        panel_label(ax, "AB"[j])
    fig.tight_layout()
    save(fig, "FigureS6")


AUDIT_ROWS = [
    ("Both eyes in the evaluation data (of eligible)", "Both eyes of some patients\n(of eligible studies)"),
    ("Patient-level split (of both-eyes studies)", "Patient-level data split"),
    ("Image- or eye-level split (of both-eyes studies)", "Image- or eye-level split"),
    ("No uncertainty reported (of both-eyes studies)", "No CI or SE for performance"),
    ("Uncertainty assuming independence (of both-eyes studies)", "CI or test assuming independent eyes"),
    ("Clustered uncertainty (of both-eyes studies) [primary]", "Patient-level CI or test"),
    ("Correlation acknowledged (of both-eyes studies)", "Fellow-eye correlation acknowledged"),
    ("Statistical test of models (of both-eyes studies)", "Statistical test between models"),
]


ICONS = [  # (item in validation_results.csv, label, colour); the first 3 split the both-eyes studies
    ("No uncertainty reported (of both-eyes studies)", "Both eyes: no CI for performance", INK2),
    ("Uncertainty assuming independence (of both-eyes studies)", "Both eyes: CI or test assuming independent eyes",
     COL["naive"]),
    ("Clustered uncertainty (of both-eyes studies) [primary]", "Both eyes: patient-level CI or test", COL["cluster"]),
    ("One eye per patient (of eligible)", "One eye, or one output, per patient", "#a9a8a3"),
    ("Several images, eyes unclear (of eligible)", "Eyes per patient unclear or not reported", "#dcdbd6"),
]


def _largest_remainder(x, total):
    x = np.asarray(x, float) * total / np.sum(x)
    n = np.floor(x).astype(int)
    n[np.argsort(-(x - n))[:total - n.sum()]] += 1
    return n


def survey_icons(r):
    """Counts per 100 eligible studies: both eyes split by analysis; the rest by what they reported."""
    e = r["estimate"]
    both = e["Both eyes in the evaluation data (of eligible)"]
    parts = [both * e[k] for k, _, _ in ICONS[:3]]
    parts.append(e["One eye per patient (of eligible)"] + e["Patient-level model (of eligible)"])
    parts.append(e["Several images, eyes unclear (of eligible)"] + e["Eyes per patient not reported (of eligible)"])
    n_both = int(round(100 * both))
    top = _largest_remainder(parts[:3], n_both)
    rest = _largest_remainder(parts[3:], 100 - n_both)
    return [int(v) for v in top] + [int(v) for v in rest]


def figure_survey():
    """Figure 2: what 100 published ophthalmic AI studies did with the second eye (A), and the estimates with their
    95% CIs (B); the top row of B has all eligible studies as denominator, the others the studies with both eyes."""
    r = pd.read_csv(AUDIT)
    r = r[r.analysis == "corpus estimate (two-phase)"].set_index("item")
    counts = survey_icons(r)
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1.05], width_ratios=[1.0, 0.72], hspace=0.22, wspace=0.02)
    axA = fig.add_subplot(gs[0, 0])
    cols = np.concatenate([[c] * n for (_, _, c), n in zip(ICONS, counts)])
    k = np.arange(100)
    x, y = k % 20, 4 - k // 20                                       # 5 rows of 20, filled from the top left
    axA.scatter(x, y, s=95, marker="s", c=list(cols), edgecolor="white", linewidth=0.8)
    axA.set_xlim(-0.8, 19.8)
    axA.set_ylim(-0.8, 4.8)
    axA.set_aspect("equal")
    axA.axis("off")
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], marker="s", ls="", ms=8, color=c, mec="white") for _, _, c in ICONS]
    labs = [f"{n}  {lab}" for (_, lab, _), n in zip(ICONS, counts)]
    axL = fig.add_subplot(gs[0, 1])
    axL.axis("off")
    axL.legend(hand, labs, loc="center left", frameon=False, fontsize=7.3, handletextpad=0.4, labelspacing=0.7)
    axA.set_title("Of every 100 eligible ophthalmic AI studies", loc="left", color=INK)
    axA.text(-0.05, 1.04, "A", transform=axA.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")

    ax = fig.add_subplot(gs[1, :])
    n = len(AUDIT_ROWS)
    yy_ = np.r_[n - 0.3, np.arange(n - 1)[::-1]].astype(float)      # gap between the 2 denominators
    for yy, (key, lab) in zip(yy_, AUDIT_ROWS):
        e = r.loc[key]
        ax.plot([e.estimate_lo, e.estimate_hi], [yy, yy], color=COL["cluster"], lw=1.6, solid_capstyle="round")
        ax.scatter(e.estimate, yy, s=30, color=COL["cluster"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(1.005, yy, f"{e.estimate:.0%} ({e.estimate_lo:.0%}-{e.estimate_hi:.0%})", va="center", fontsize=7,
                color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(yy_)
    ax.set_yticklabels([lab.replace("\n(of eligible studies)", "") for _, lab in AUDIT_ROWS])
    for yy, txt in ((n + 0.45, "Of eligible studies"), (n - 1.3, "Of eligible studies with both eyes of some patients")):
        ax.text(-0.005, yy, txt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.3,
                fontweight="bold", color=INK)
    ax.axhline(n - 0.85, color=GRID, lw=0.8)
    ax.set_ylim(-0.7, n + 0.9)
    ax.set_xlim(0, 0.8)
    ax.text(1.005, n + 0.45, "Estimate\n(95% CI)", va="center", fontsize=7, color=INK2,
            transform=ax.get_yaxis_transform())
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    grid(ax, "x")
    ax.set_xlabel("Estimated Percentage of Open-Access Studies")
    ax.tick_params(axis="y", length=0)
    ax.text(-0.75, 1.04, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    save(fig, "Figure2")


def tables(s):
    h = s[s.metric != "medae"].copy()
    fam = {"bias": "Mean error (bias)", "mae": "Mean absolute error", "rmse": "Root mean squared error",
           "within": "Proportion within a tolerance", "sd": "Standard deviation of error", "auc": "AUC",
           "accuracy": "Accuracy", "sensitivity": "Sensitivity", "specificity": "Specificity",
           "brier": "Brier score", "logloss": "Log-loss"}
    rows = []
    for (task, metric), g in h.groupby(["task", "metric"]):
        row = {"task": task, "metric": fam.get(metric, metric)}
        for st in ("naive", "one_eye", "cluster", "patient", "optimal"):
            x = g[(g.strategy == st) & (g.scen == "H0")].reject
            row[f"type1_{st}"] = f"{100 * x.min():.1f}-{100 * x.max():.1f}" if len(x) else ""
            p = g[(g.strategy == st) & (g.scen == "H1")].reject
            row[f"power_{st}"] = f"{100 * p.min():.0f}-{100 * p.max():.0f}" if len(p) else ""
        c = g[(g.strategy == "cluster") & (g.scen == "H0")].coverage
        row["cluster_coverage"] = f"{100 * c.min():.1f}-{100 * c.max():.1f}" if c.notna().any() else ""
        rd = g[(g.strategy == "cluster") & (g.scen == "H0")].rho_delta
        row["rho_delta"] = f"{max(rd.min(), 0):.2f}-{rd.max():.2f}" if rd.notna().any() else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "table_s_type1_by_metric.csv"), index=False)
    # Table 1: by strategy, across all cells and metrics with a design-effect prediction
    t = []
    hh = h[h.scen == "H0"]
    for st in ("naive", "one_eye", "cluster", "patient", "optimal"):
        x = hh[hh.strategy == st]
        hi = hh[(hh.strategy == st) & (hh.psi >= 0.9)]
        rv = h[(h.strategy == st) & (h.scen == "H0") & h.metric.isin(MEAN_METRICS)].rel_var
        t.append({"strategy": LAB[st], "n_cells": len(x), "type1_median": x.reject.median(),
                  "type1_min": x.reject.min(), "type1_max": x.reject.max(),
                  "type1_max_psi_ge_0.9": hi.reject.max() if len(hi) else np.nan,
                  "share_cells_outside_mc_band": float(((x.reject < 0.0404) | (x.reject > 0.0596)).mean()),
                  "rel_var_min": rv.min() if len(rv) else np.nan, "rel_var_max": rv.max() if len(rv) else np.nan})
    pd.DataFrame(t).to_csv(os.path.join(RES, "table1_simulation.csv"), index=False)
    nv = hh[(hh.strategy == "naive") & hh.pred_naive_type1.notna()]
    dev = nv.reject - nv.pred_naive_type1
    print(f"naive observed - predicted type I error: mean {dev.mean():.4f}, range {dev.min():.4f} to {dev.max():.4f}, "
          f"r = {np.corrcoef(nv.reject, nv.pred_naive_type1)[0, 1]:.3f} ({len(nv)} cell-metrics)")
    print("wrote table1_simulation.csv, table_s_type1_by_metric.csv")


if __name__ == "__main__":
    s = load_main()
    figure_flow(s)
    figure_headline(s)
    figure_check(s)
    figure1(s)
    figure2(s)
    figure3()
    figure_lottery()
    if os.path.exists(AUDIT):
        figure_survey()
    tables(s)
