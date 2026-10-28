"""
Figures and table data for the manuscript (run after analyse_simulation.py and audit/audit_03_validation.py --score).

  Figure 1   Study designs: where the second eye enters a study that applies AI to eyes.
  Figure 2   Survey of published studies (random sample): what studies did with the second eye.
  Figure 3   False-positive rate (type I error) against rho_delta, 80% of patients bilateral.
  Figure 4   The cost of keeping one eye per patient: power against study size for 6 metrics, and how often a
             one-eye conclusion depends on the eye kept.
  Figure S2  Design effect: false-positive rate of an eye-level analysis and extra patients needed with one eye.
  Figure S3  Simulation against the design-effect predictions.
  Figure S4  False-positive rate against rho_delta with 40% and 80% of patients bilateral (all settings).
  (Figure 5 comes from realdata/papila_figure.py; Supplementary Figure S1 from manuscript/build_supplement.py.)

Writes figures/*.tiff (600 dpi, LZW, RGB) with .png previews, and results/table1_simulation.csv,
results/table_s_type1_by_metric.csv.

Usage (from this folder):  python make_figures.py
"""
import importlib.util
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

# neutral journal palette: black for the analysis we recommend, dark red for eyes treated as independent, dark blue
# for one eye per patient, greys for everything else; white background throughout
COL = {"cluster": "#000000", "naive": "#B2182B", "one_eye": "#2166AC"}
MARK = {"cluster": "o", "naive": "s", "one_eye": "^"}
LAB = {"cluster": "Both eyes, patient as the unit", "naive": "Both eyes treated as independent",
       "one_eye": "One eye per patient"}
TABLE_LAB = dict(LAB, patient="Both eyes, each patient weighted equally",
                 optimal="Both eyes, weights chosen for the smallest variance")
INK, INK2, GRID, BAND = "#000000", "#404040", "#e3e3e3", "#ececec"
MEAN_METRICS = ["bias", "mae", "rmse", "brier", "logloss"]
PROP_METRICS = ["within", "sensitivity", "specificity", "accuracy"]
RHO_LABEL = r"Fellow-Eye Correlation of the Difference Between Models ($\rho_\Delta$)"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.color": INK,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
    "axes.facecolor": "white", "figure.facecolor": "white"})


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def naive_pred(f, rho):
    return 2 * (1 - phi(1.959964 / sqrt(1 + 2 * f * rho / (1 + f))))


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, s, x=-0.14):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


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


def n80(d):
    """Patients for 80% power by linear interpolation of the sweep."""
    d = d.sort_values("n_patients")
    x, y = d.n_patients.values, d.reject.values
    i = int(np.argmax(y >= 0.8))
    return x[i - 1] + (0.8 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


# ---------------------------------------------------------------------------------------------- Figure 1
def figure_designs():
    """Figure 1: a categorisation of study designs that apply AI to eyes, showing where the second eye matters."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0.16, 1.0); ax.axis("off")
    LH = 0.0275                                   # one line of 6.8 pt text in axes units

    def box(x, ytop, w, h, head=None, body="", lw=0.7, fs=6.6):
        ax.add_patch(Rectangle((x - w / 2, ytop - h), w, h, fc="white", ec=INK, lw=lw))
        if head is None:
            ax.text(x, ytop - h / 2, body, ha="center", va="center", fontsize=fs + 0.4, linespacing=1.3)
            return
        ax.text(x, ytop - 0.016, head, ha="center", va="top", fontsize=fs + 0.2, fontweight="bold")
        ax.text(x, ytop - 0.016 - 1.35 * LH, body, ha="center", va="top", fontsize=fs, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))

    def h(lines, head=True):
        return 0.016 + (1.35 * LH if head else 0) + lines * LH * 1.08 + 0.016

    xl, wl = 0.115, 0.22
    xr, wr = 0.6125, 0.745
    xs, w = (0.36, 0.6125, 0.865), 0.24
    box(0.5, 0.99, 0.62, 0.075, body="AI applied to images or measurements of eyes,\nin data where some patients "
        "contribute both eyes")
    arrow(0.28, 0.915, xl, 0.84); arrow(0.6125, 0.915, xr, 0.84)
    box(xl, 0.84, wl, h(4), head="AI as a measuring tool",
        body="The model measures a\nfeature, which is then\nrelated to an exposure\nor an outcome")
    box(xr, 0.84, wr, h(2), head="Comparing the accuracy of AI models",
        body="Each model's output is compared with a reference standard\n(a diagnosis or a measured value)")
    y2 = 0.84 - h(2)
    y3 = 0.655
    for x in xs:
        arrow(x, y2, x, y3)
    heads = ("One output per patient", "One eye per patient", "A result for each eye")
    bodies = ("The model reads both eyes\nand gives a single result\nfor the patient",
              "One eye is chosen by\ndesign, or the second eye\nis discarded",
              "Both eyes of some patients\nare in the test set, each\nwith its own result")
    results = ("One result per patient.\nStandard tests apply.",
               "Standard tests apply, but\nless data are used, and\nthe result can depend on\nwhich eye was kept.",
               "Fellow eyes are correlated,\nso analyse with the patient\nas the unit. Treating eyes\nas independent "
               "overstates\nprecision.")
    y4 = 0.43
    for i, (x, hd, bd, rs) in enumerate(zip(xs, heads, bodies, results)):
        box(x, y3, w, h(3), head=hd, body=bd)
        arrow(x, y3 - h(3), x, y4)
        box(x, y4, w, h(5, head=False), body=rs, lw=1.8 if i == 2 else 0.7)
    arrow(xl, 0.84 - h(4), xl, y4)
    box(xl, y4, wl, h(5), head="Association analysis",
        body="Both eyes enter a\nregression with the\npatient as the unit, for\nexample generalised\nestimating equations")
    ax.text(xs[2], y4 - h(5, head=False) - 0.012, "Focus of this study", ha="center", va="top", fontsize=7,
            fontstyle="italic")
    save(fig, "Figure1")


# ---------------------------------------------------------------------------------------------- Figure 2
def _survey():
    spec = importlib.util.spec_from_file_location("survey_summary", os.path.join(HERE, "..", "audit",
                                                                                   "survey_summary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.summary()


def _largest_remainder(x, total):
    x = np.asarray(x, float) * total / max(np.sum(x), 1e-12)
    n = np.floor(x).astype(int)
    n[np.argsort(-(x - n))[:total - n.sum()]] += 1
    return n


ICONS = [("none", "Both eyes, no confidence interval for accuracy", "#6e6e6e"),
         ("indep", "Both eyes, interval or test assuming independent eyes", COL["naive"]),
         ("clustered", "Both eyes, interval or test with the patient as the unit", INK),
         ("unit_one", "One eye, or one output, per patient", "#c4c4c4"),
         ("unit_unclear", "Number of eyes per patient unclear", "#ededed")]


def survey_icons(S):
    """Counts per 100 eligible studies (random sample): both eyes split by analysis, then the rest."""
    n_both = int(round(100 * S["unit_both_p"]))
    top = _largest_remainder([S["none"], S["indep"], S["clustered"]], n_both) if S["both"] else [0, 0, 0]
    rest = _largest_remainder([S["unit_one"], S["unit_unclear"]], 100 - n_both)
    return [int(v) for v in top] + [int(v) for v in rest]


def figure_survey():
    """Figure 2: what published ophthalmic AI studies did with the second eye (random sample of articles)."""
    S = _survey()
    counts = survey_icons(S)
    fig = plt.figure(figsize=(7.0, 5.9))
    gsA = fig.add_gridspec(1, 2, left=0.04, right=0.99, top=0.95, bottom=0.78, width_ratios=[1.0, 1.0], wspace=0.04)
    gsB = fig.add_gridspec(1, 1, left=0.43, right=0.83, top=0.64, bottom=0.07)
    axA = fig.add_subplot(gsA[0, 0])
    cols = np.concatenate([[c] * n for (_, _, c), n in zip(ICONS, counts)])
    k = np.arange(100)
    axA.scatter(k % 20, 4 - k // 20, s=95, marker="s", c=list(cols), edgecolor="#b0b0b0", linewidth=0.4)
    axA.set_xlim(-0.8, 19.8); axA.set_ylim(-0.8, 4.8)
    axA.set_aspect("equal"); axA.set_anchor("W"); axA.axis("off")
    axA.set_title("Of every 100 eligible studies", loc="left")
    from matplotlib.lines import Line2D
    axL = fig.add_subplot(gsA[0, 1]); axL.axis("off")
    hand = [Line2D([], [], marker="s", ls="", ms=8, color=c, mec="#b0b0b0", mew=0.4) for _, _, c in ICONS]
    axL.legend(hand, [f"{n:>2}  {lab}" for (_, lab, _), n in zip(ICONS, counts)], loc="center left", frameon=False,
               fontsize=7.2, handletextpad=0.4, labelspacing=0.7)

    ax = fig.add_subplot(gsB[0, 0])
    e, b = S["eligible"], S["both"]
    rows = [(f"§Eyes per patient ({e} eligible studies)", None),
            ("Both eyes of some patients", "unit_both"),
            ("One eye, or one output, per patient", "unit_one"),
            ("Unclear", "unit_unclear"),
            (f"§Analysis of accuracy ({b} studies using both eyes)", None),
            ("No confidence interval reported", "none"),
            ("Interval or test assuming independent eyes", "indep"),
            ("Interval or test with the patient as the unit", "clustered"),
            (f"§Data splitting ({b} studies using both eyes)", None),
            ("By patient, preventing leakage (good practice)", "split_patient"),
            ("By eye or image (images of one patient can leak)", "split_image"),
            ("Separate external test set only", "split_external"),
            ("Not reported", "split_nr"),
            (f"§Other reporting ({b} studies; not mutually exclusive)", None),
            ("Correlation between fellow eyes mentioned", "ack"),
            ("Models compared with a statistical test", "test")]
    y = np.arange(len(rows))[::-1].astype(float)
    labels, ticks = [], []
    for yy, (lab, key) in zip(y, rows):
        if key is None:
            ax.text(-0.01, yy, lab[1:], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.4,
                    fontweight="bold")
            continue
        p, lo, hi = S[f"{key}_p"], S[f"{key}_lo"], S[f"{key}_hi"]
        ax.plot([lo, hi], [yy, yy], color=INK, lw=1.2, solid_capstyle="butt")
        ax.scatter(p, yy, s=22, color=INK, zorder=3, clip_on=False)
        ax.text(1.02, yy, f"{p:.0%} ({lo:.0%}-{hi:.0%})", transform=ax.get_yaxis_transform(), va="center", fontsize=7)
        labels.append(lab); ticks.append(yy)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    grid(ax, "x")
    ax.set_xlabel("Percentage of Studies (95% Confidence Interval)")
    ax.text(1.02, len(rows) - 0.55, "Estimate\n(95% CI)", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
            fontweight="bold")
    ax.set_title("Percentage of studies with each characteristic", loc="left", x=-0.68, pad=14)
    fig.canvas.draw()
    for a_, lab in ((axA, "A"), (ax, "B")):
        fig.text(0.0, a_.get_position().y1 + 0.015, lab, fontsize=10, fontweight="bold", va="bottom", ha="left")
    save(fig, "Figure2")


# ---------------------------------------------------------------------------------------------- Figure 3 / S4
def _type1_panel(ax, d, f, annotate=False):
    mc = 1.959964 * sqrt(0.05 * 0.95 / 2000)
    rr = np.linspace(0, 1, 101)
    ax.axhspan(100 * (0.05 - mc), 100 * (0.05 + mc), color=BAND, lw=0, zorder=0,
               label="Expected range for a valid test (about 5%)")
    ax.plot(rr, [100 * naive_pred(f, r) for r in rr], color=COL["naive"], lw=1.0, ls=(0, (4, 2)), zorder=2,
            label="Predicted for both eyes treated as independent")
    for st in ("naive", "one_eye", "cluster"):
        x = d[d.strategy == st]
        ax.scatter(x.rho, 100 * x.reject, s=12, marker=MARK[st], color=COL[st], edgecolor="white", linewidth=0.3,
                   zorder=3, label=LAB[st])
    grid(ax)
    ax.set_ylim(0, 20)
    ax.set_xlim(-0.02, 1.0)
    if annotate:
        ax.text(0.03, 0.97, "Eyes treated as independent: false positives\nrise as fellow eyes become more alike",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COL["naive"])
        ax.text(0.03, 0.82, "Patient as the unit: close to the intended\n5% whatever the correlation",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK)


def _type1_legend(fig, ax, y):
    h_, l_ = ax.get_legend_handles_labels()
    want = [LAB["cluster"], LAB["naive"], LAB["one_eye"], "Predicted for both eyes treated as independent",
            "Expected range for a valid test (about 5%)"]
    hd = dict(zip(l_, h_))
    fig.legend([hd[k] for k in want], want, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y),
               handletextpad=0.4, markerscale=1.5)


def figure_type1(s):
    h = s[(s.scen == "H0") & s.rho_delta.notna() & (s.metric != "medae")].copy()
    h["rho"] = h.rho_delta.clip(lower=0)
    # Figure 3: 80% of patients bilateral
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for j, task in enumerate(("continuous", "binary")):
        ax = axes[j]
        _type1_panel(ax, h[(h.task == task) & (h.frac == 0.8)], 0.8, annotate=True)
        ax.set_title("Continuous outputs" if task == "continuous" else "Classifiers", loc="left")
        ax.set_xlabel(RHO_LABEL)
        if j == 0:
            ax.set_ylabel("False-Positive Rate (Type I Error, %)")
        panel_label(ax, "AB"[j])
    _type1_legend(fig, axes[0], -0.07)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "Figure3")
    # Figure S4: both proportions bilateral
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, sharey=True)
    for i, task in enumerate(("continuous", "binary")):
        for j, f in enumerate((0.4, 0.8)):
            ax = axes[i, j]
            _type1_panel(ax, h[(h.task == task) & (h.frac == f)], f)
            title = "Continuous outputs" if task == "continuous" else "Classifiers"
            ax.set_title(f"{title}, {int(f * 100)}% of patients with both eyes", loc="left")
            if j == 0:
                ax.set_ylabel("False-Positive Rate (%)")
            if i == 1:
                ax.set_xlabel(RHO_LABEL)
            panel_label(ax, "ABCD"[2 * i + j])
    _type1_legend(fig, axes[0, 0], -0.04)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "FigureS4")


# ---------------------------------------------------------------------------------------------- Figure 4
POWER_PANELS = [("continuous", "mae", "Mean absolute error"), ("continuous", "rmse", "Root mean squared error"),
                ("continuous", "sd", "SD of the error"), ("binary", "auc", "Area under the ROC curve"),
                ("binary", "brier", "Brier score"), ("binary", "sensitivity", "Sensitivity")]
SAME = "#bdbdbd"


def power_needed():
    """Patients for 80% power with both eyes (patient as the unit) and with one eye, 80% of patients bilateral."""
    sw = pd.read_csv(os.path.join(RES, "summary_sweep.csv"))
    out = {}
    for task, metric, _ in POWER_PANELS:
        d = sw[(sw.task == task) & (sw.metric == metric) & (sw.scen == "H1") & (sw.frac == 0.8)]
        out[metric] = {st: n80(d[d.strategy == st]) for st in ("cluster", "one_eye")}
    return out


def figure_one_eye_cost():
    """Figure 4: what keeping one eye per patient costs: power (A-F) and dependence on the eye kept (G, H)."""
    sw = pd.read_csv(os.path.join(RES, "summary_sweep.csv"))
    fig = plt.figure(figsize=(7.0, 7.9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.62, wspace=0.30, top=0.93, bottom=0.11)
    for i, (task, metric, title) in enumerate(POWER_PANELS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        d = sw[(sw.task == task) & (sw.metric == metric) & (sw.scen == "H1") & (sw.frac == 0.8)]
        need = {}
        for st in ("cluster", "one_eye"):
            x = d[d.strategy == st].sort_values("n_patients")
            ax.plot(x.n_patients, 100 * x.reject, color=COL[st], lw=1.4, marker=MARK[st], ms=3.5, mec="white",
                    mew=0.4, label=LAB[st])
            need[st] = int(round(n80(x)))
        ax.axhline(80, color=INK2, lw=0.5, ls=(0, (3, 2)))
        extra = need["one_eye"] / need["cluster"] - 1
        ax.text(0.97, 0.05, f"Patients for 80% power\nboth eyes {need['cluster']}, one eye {need['one_eye']}"
                f"\n(one eye needs {extra:.0%} more)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
                linespacing=1.25)
        grid(ax)
        ax.set_ylim(0, 102); ax.set_xlim(0, 1020)
        ax.set_xticks([0, 250, 500, 750, 1000])
        if i % 3 == 0:
            ax.set_ylabel("Power (%)")
        if i >= 3:
            ax.set_xlabel("Patients per Study")
        ax.set_title(title, loc="left", fontsize=8)
        panel_label(ax, "ABCDEF"[i], x=-0.2)
    hp, lp = fig.axes[0].get_legend_handles_labels()
    fig.canvas.draw()
    for r, lab in ((0, "Continuous outputs: power to detect a true difference between two models"),
                   (3, "Classifiers: power to detect a true difference between two models")):
        fig.text(0.06, fig.axes[r].get_position().y1 + 0.038, lab, fontsize=8.5, fontweight="bold", ha="left")
    gs2 = gs[2, :].subgridspec(1, 2, wspace=0.22)
    axes2 = []
    for j, (task, title) in enumerate((("continuous", "Continuous outputs (mean absolute error)"),
                                       ("binary", "Classifiers (area under the ROC curve)"))):
        ax = fig.add_subplot(gs2[0, j]); axes2.append(ax)
        d = pd.read_csv(os.path.join(LOTTERY, f"lottery_{task}.csv.gz"))
        bins = np.linspace(0, 1, 11)
        cnt, _ = np.histogram(d.one_eye_frac_sig, bins=bins)
        share = 100 * cnt / len(d)
        colors = [COL["one_eye"] if 0 < b < 9 else SAME for b in range(10)]
        ax.bar(100 * (bins[:-1] + 0.05), share, width=9, color=colors, zorder=2)
        grid(ax)
        ax.set_xlim(0, 100); ax.set_ylim(0, 48)
        ax.set_xlabel("Random Choices of Eye Giving P < 0.05 (%)")
        if j == 0:
            ax.set_ylabel("Simulated Studies (%)")
        mid = ((d.one_eye_frac_sig >= 0.1) & (d.one_eye_frac_sig <= 0.9)).mean()
        ax.text(60, 31, f"Conclusion depends on the eye kept\n(10%-90% of choices significant): "
                f"{mid:.0%}", ha="center", va="bottom", fontsize=6.6, color=COL["one_eye"])
        ax.text(1, share[0] + 1, "Rarely or never\nsignificant\n(difference\nmissed)", ha="left", va="bottom",
                fontsize=6.0, color=INK2)
        ax.text(99.5, share[9] + 1, "Almost always\nsignificant", ha="right", va="bottom", fontsize=6.0, color=INK2)
        ax.text(0.01, 0.985, f"Both eyes, patient as the unit:\nsignificant in {d.cluster_sig.mean():.0%} of "
                "the same studies", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK)
        ax.set_title(title, loc="left", fontsize=8)
        panel_label(ax, "GH"[j], x=-0.13)
    fig.canvas.draw()
    fig.text(0.06, axes2[0].get_position().y1 + 0.038,
             "With one random eye per patient, does the conclusion depend on which eye was kept?", fontsize=8.5,
             fontweight="bold", ha="left")
    fig.legend(hp, lp, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), markerscale=1.3)
    save(fig, "Figure4")


# ---------------------------------------------------------------------------------------------- Figure S2 / S3
def figure_map(s):
    """Supplementary Figure S2: false-positive rate of an eye-level analysis, and extra patients needed with one
    eye per patient, for any rho_delta and proportion of patients with both eyes (design effect)."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import norm as _norm
    rr = np.linspace(0, 1, 301)
    ff = np.linspace(0, 1, 301)
    R, F = np.meshgrid(rr, ff)
    T1 = np.vectorize(naive_pred)(F, R) * 100
    EX_raw = ((1 + F) ** 2 / (1 + F + 2 * F * R) - 1) * 100
    EX = np.clip(EX_raw, 0, None)
    cmap_a = LinearSegmentedColormap.from_list("reds", ["#ffffff", "#f4a582", COL["naive"], "#67001f"])
    cmap_b = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#92c5de", COL["one_eye"], "#053061"])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"wspace": 0.45})
    im = axA.pcolormesh(R, F * 100, T1, cmap=cmap_a, vmin=5, vmax=17, shading="auto", rasterized=True)
    lev_a = [6, 8, 10, 12, 14]
    cs = axA.contour(R, F * 100, T1, levels=lev_a, colors=INK, linewidths=0.5)
    pos = []
    for L, f in zip(lev_a, (0.30, 0.45, 0.55, 0.65, 0.68)):
        de = (1.959964 / _norm.ppf(1 - L / 200)) ** 2
        pos.append(((de - 1) * (1 + f) / (2 * f), f * 100))
    axA.clabel(cs, fmt="%d%%", fontsize=6.5, inline_spacing=2, manual=pos)
    cb = fig.colorbar(im, ax=axA, fraction=0.05, pad=0.03); cb.set_label("False-Positive Rate (%)")
    cb.outline.set_linewidth(0.4)
    axA.set_title("Both eyes treated as independent:\nfalse-positive rate (intended 5%)", loc="left")
    im = axB.pcolormesh(R, F * 100, EX, cmap=cmap_b, vmin=0, vmax=100, shading="auto", rasterized=True)
    lev_b = [10, 25, 50, 75]
    cs = axB.contour(R, F * 100, EX, levels=lev_b, colors=INK, linewidths=0.5)
    pos = [(((1 + f) ** 2 / (1 + E_ / 100) - (1 + f)) / (2 * f), f * 100)
           for E_, f in zip(lev_b, (0.50, 0.60, 0.72, 0.93))]
    axB.clabel(cs, fmt="+%d%%", fontsize=6.5, inline_spacing=2, manual=pos)
    with plt.rc_context({"hatch.linewidth": 0.35, "hatch.color": "#8c8c8c"}):
        axB.contourf(R, F * 100, EX_raw, levels=[-100, 0], colors="none", hatches=["//////"], zorder=2)
    axB.contour(R, F * 100, EX_raw, levels=[0], colors=INK, linewidths=0.6, linestyles="dashed")
    axB.text(0.97, 0.06, "Hatched: one eye slightly\nmore precise (at most 11%)", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=6.3, bbox=dict(fc="white", ec="none", pad=1))
    cb = fig.colorbar(im, ax=axB, fraction=0.05, pad=0.03); cb.set_label("Extra Patients Needed (%)")
    cb.outline.set_linewidth(0.4)
    axB.set_title("One eye per patient: extra patients needed\nfor the same precision as both eyes", loc="left")
    for ax, lab in ((axA, "A"), (axB, "B")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 100)
        ax.set_xlabel(r"Fellow-Eye Correlation of the Difference ($\rho_\Delta$)")
        ax.set_ylabel("Patients With Both Eyes (%)")
        panel_label(ax, lab)
    save(fig, "FigureS2")


def figure_check(s):
    """Supplementary Figure S3: the simulation against the design-effect predictions."""
    h = s[(s.scen == "H0") & s.metric.isin(MEAN_METRICS) & s.rho_delta.notna()].copy()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.38})
    n = h[h.strategy == "naive"]
    lim = (0, 18)
    axA.plot(lim, lim, color=INK2, lw=0.6)
    axA.scatter(n.pred_naive_type1 * 100, n.reject * 100, s=12, marker=MARK["naive"], color=COL["naive"],
                edgecolor="white", linewidth=0.3, zorder=3)
    axA.set_xlim(lim); axA.set_ylim(lim)
    axA.set_xlabel("Predicted False-Positive Rate (%)")
    axA.set_ylabel("Simulated False-Positive Rate (%)")
    axA.set_title("Both eyes treated as independent:\nfalse-positive rate", loc="left")
    axA.text(1.2, 1.7, "Perfect agreement", rotation=45, rotation_mode="anchor", ha="left", va="bottom",
             fontsize=6.5, color=INK2, transform_rotates_text=True)
    o = h[h.strategy == "one_eye"]
    lim = (-15, 100)
    axB.plot(lim, lim, color=INK2, lw=0.6)
    axB.axhline(0, color=INK2, lw=0.4, ls=(0, (3, 2))); axB.axvline(0, color=INK2, lw=0.4, ls=(0, (3, 2)))
    axB.scatter(100 * (o.pred_rel_var_one_eye - 1), 100 * (o.rel_var - 1), s=12, marker=MARK["one_eye"],
                color=COL["one_eye"], edgecolor="white", linewidth=0.3, zorder=3)
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel("Predicted Extra Patients Needed (%)")
    axB.set_ylabel("Simulated Extra Patients Needed (%)")
    axB.set_title("One eye per patient: extra patients needed\nfor the same precision as both eyes", loc="left")
    for ax, lab in ((axA, "A"), (axB, "B")):
        grid(ax, "both")
        panel_label(ax, lab)
    save(fig, "FigureS3")


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
        t.append({"strategy": TABLE_LAB[st], "n_cells": len(x), "type1_median": x.reject.median(),
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
    figure_designs()
    figure_survey()
    figure_type1(s)
    figure_one_eye_cost()
    figure_map(s)
    figure_check(s)
    tables(s)
