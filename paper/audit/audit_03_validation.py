"""
Audit step 3: validation of the rule-based classification against manual coding.

  python audit_03_validation.py --make
      Draws the validation sample from audit_data/extracted.csv, excluding the development set
      (development_set.csv: the articles used to revise the rules, which must not be used to validate them):
        - 200 articles at random from the remaining open-access full texts (estimates accuracy and prevalence);
        - plus up to 40 more drawn from the remaining articles the rules classify as 'clustered' (the rare
          class), so its positive predictive value can also be checked. These are marked sample = "enriched"
          and are not used for overall accuracy.
      Writes audit_data/validation_sample.csv (manual columns blank, with a link to each article) and
      audit_data/validation_second_coder.csv: 25 random articles from the random part, with no codes
      shown, for the second coder.

  python audit_03_validation.py --extend
      Fixes the final design (run once, after --make): a random core of 120 (the first 120 random articles
      in the coding order), the 40 enriched 'clustered' articles, and 40 new enriched articles that the
      rules classify as both eyes but not clustered. Adds coding_order and include columns to
      validation_sample.csv and writes validation_human2.csv (50 included articles for a second human coder).

  python audit_03_validation.py --score [--codes FILE] [--codes2 FILE]
      Merges the reference (manual) codes for the 200 included articles (default
      audit_data/validation_codes_reference.tsv; a completed coding workbook .xlsx also works) and reports:
        1. agreement and Cohen's kappa (bootstrap 95% CIs) between the rules and the reference coding on the
           random core of 120;
        2. predictive values of the two enriched rule cells (Wilson 95% CIs);
        3. corpus estimates from the two-phase stratified (double-sampling) estimator: phase 1 = the rules
           applied to every open-access full text outside the development set, which defines three cells
           (C1 eligible and clustered; C2 eligible, both eyes, not clustered; C3 the rest); phase 2 = the
           reference codes within each cell. Proportions are ratios of estimated totals. 95% intervals use a
           Jeffreys (Dirichlet) posterior within each cell, which stays valid when a cell has no events; the
           percentile bootstrap within cells is given as a check. The rules-only estimate is shown alongside;
        4. agreement between two manual coders, if --codes2 is given.
      Writes audit_data/validation_results.csv and audit_data/validation_confusion.txt.

  python audit_03_validation.py --dev
      Agreement between the current rules and the hand-coded part of the development set
      (audit_data/manual_codes_dev_batch1.tsv). This is development performance, reported only as such.

Manual codes follow CODEBOOK.md. Columns m1_* are the reference coding, m2_* an optional second coder.
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA = "audit_data"
HERE = os.path.dirname(os.path.abspath(__file__))
DEV_LIST = os.path.join(HERE, "development_set.csv")
DEV_CODES = os.path.join(DATA, "manual_codes_dev_batch1.tsv")
SEED = 20261022
N_RANDOM, N_ENRICH, N_SECOND = 200, 40, 25
ITEMS = ["eligible", "unit", "split", "uncertainty", "comparison", "correlation_ack"]
AUTO = {"eligible": "eligible_auto", "unit": "unit", "split": "split", "uncertainty": "uncertainty",
        "comparison": "comparison", "correlation_ack": "correlation_ack"}


def make():
    df = pd.read_csv(os.path.join(DATA, "extracted.csv"))
    dev = set(pd.read_csv(DEV_LIST, dtype=str).pmcid)
    pool = df[(df.fulltext_source == "xml") & ~df.pmcid.isin(dev)].copy()
    rng = np.random.default_rng(SEED)
    rand = pool.sample(n=min(N_RANDOM, len(pool)), random_state=rng.integers(2 ** 31))
    rand["sample"] = "random"
    rest = pool[~pool.id.isin(rand.id) & (pool.eligible_auto == True) & (pool.uncertainty == "clustered")]  # noqa: E712
    enr = rest.sample(n=min(N_ENRICH, len(rest)), random_state=rng.integers(2 ** 31))
    enr["sample"] = "enriched"
    val = pd.concat([rand, enr])
    val["link"] = "https://europepmc.org/article/PMC/" + val.pmcid.astype(str)
    keep = ["sample", "id", "pmcid", "doi", "journal", "pub_year", "title", "link"] + [AUTO[i] for i in ITEMS]
    val = val[keep].rename(columns={AUTO[i]: f"auto_{i}" for i in ITEMS})
    for i in ITEMS:
        val[f"m1_{i}"] = ""
    val["m1_notes"] = ""
    val.to_csv(os.path.join(DATA, "validation_sample.csv"), index=False)
    second = val[val["sample"] == "random"].sample(n=N_SECOND, random_state=SEED)
    second = second[["id", "pmcid", "doi", "journal", "pub_year", "title", "link"]].copy()
    for i in ITEMS:
        second[f"m2_{i}"] = ""
    second["m2_notes"] = ""
    second.to_csv(os.path.join(DATA, "validation_second_coder.csv"), index=False)
    print(f"Pool: {len(pool)} open-access full texts after excluding {len(dev)} development articles")
    print(f"validation_sample.csv: {len(rand)} random + {len(enr)} enriched; "
          f"validation_second_coder.csv: {len(second)} articles")


N_CORE, N_BOTH_UNCL, N_HUMAN2 = 120, 40, 50
ORDER_SEED = 2310          # coding order of the first draw (fixed before any validation coding started)


def extend():
    """Final validation design (fixed 22 Oct 2026, before any scoring):
      - random core: the first N_CORE random articles in the coding order (a random subset of the 200);
      - enriched 'clustered': the 40 drawn by --make;
      - enriched 'both eyes, not clustered': N_BOTH_UNCL new articles that the rules classify as eligible,
        both eyes and not clustered, drawn from the pool outside the development set and the first draw.
    Adds columns coding_order and include, and draws N_HUMAN2 included articles for a second human coder."""
    import random
    path = os.path.join(DATA, "validation_sample.csv")
    val = pd.read_csv(path)
    if "include" in val.columns:
        print("validation_sample.csv is already extended; nothing changed")
        return
    df = pd.read_csv(os.path.join(DATA, "extracted.csv"), low_memory=False)
    dev = set(pd.read_csv(DEV_LIST, dtype=str).pmcid)
    pool = df[(df.fulltext_source == "xml") & ~df.pmcid.isin(dev) & ~df.pmcid.isin(val.pmcid)]
    cand = pool[(pool.eligible_auto == True) & (pool.unit == "both_eyes") & (pool.uncertainty != "clustered")]  # noqa: E712
    add = cand.sample(n=min(N_BOTH_UNCL, len(cand)), random_state=SEED + 1).copy()
    add["sample"] = "enriched_both_unclustered"
    add["link"] = "https://europepmc.org/article/PMC/" + add.pmcid.astype(str)
    add = add[["sample", "id", "pmcid", "doi", "journal", "pub_year", "title", "link"] + [AUTO[i] for i in ITEMS]]
    add = add.rename(columns={AUTO[i]: f"auto_{i}" for i in ITEMS})
    for c in val.columns:
        if c not in add.columns:
            add[c] = ""
    # coding order: the first draw in the order fixed for coder 1 (positions 1-40 were coded before this
    # design was set); the rest of the first draw and the new articles are shuffled together after them
    first = list(val.pmcid)
    random.Random(ORDER_SEED).shuffle(first)
    rest = first[40:] + list(add.pmcid)
    random.Random(ORDER_SEED + 1).shuffle(rest)
    order = {p: i + 1 for i, p in enumerate(first[:40] + rest)}
    val = pd.concat([val, add[val.columns]], ignore_index=True)
    val["coding_order"] = val.pmcid.map(order)
    val = val.sort_values("coding_order").reset_index(drop=True)
    rnd = val["sample"] == "random"
    val["include"] = ~rnd | (rnd.cumsum() <= N_CORE)
    val.loc[rnd & ~val.include, "sample"] = "random_reserve"
    val.to_csv(path, index=False)
    inc = val[val.include]
    # second human coder: a random subset of the included articles, in proportion to the three groups
    h2 = inc.groupby("sample", group_keys=False).apply(
        lambda g: g.sample(n=round(N_HUMAN2 * len(g) / len(inc)), random_state=SEED + 2))
    h2 = h2.sort_values("coding_order")[["coding_order", "pmcid"]]
    h2.to_csv(os.path.join(DATA, "validation_human2.csv"), index=False)
    print(f"Added {len(add)} enriched both-eyes/not-clustered articles. Included: "
          + ", ".join(f"{k} {v}" for k, v in inc["sample"].value_counts().items())
          + f"; total {len(inc)}. Second human coder: {len(h2)} articles (validation_human2.csv).")


def kappa(a, b):
    a, b = pd.Series(a).astype(str).values, pd.Series(b).astype(str).values
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def norm(s):
    return pd.Series(s).astype(str).str.strip().str.lower().replace({"true": "yes", "false": "no", "nan": ""})


def read_codes(path, prefix):
    """Manual codes from a .tsv/.csv (columns pmcid + ITEMS) or a completed coding workbook (.xlsx, 'Coding' tab)."""
    if path.endswith(".xlsx"):
        c = pd.read_excel(path, sheet_name="Coding", dtype=str)
    else:
        c = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", dtype=str)
    c = c[["pmcid"] + ITEMS].copy()
    for i in ITEMS:
        c[i] = norm(c[i])
    c = c[c.eligible.isin(["yes", "no"])]
    return c.rename(columns={i: f"{prefix}_{i}" for i in ITEMS})


def cells(df, pre):
    """Rule-defined sampling cells: C1 eligible and 'clustered'; C2 eligible, both eyes, not clustered; C3 the rest."""
    el = norm(df[f"{pre}eligible"]) == "yes"
    cl = norm(df[f"{pre}uncertainty"]) == "clustered"
    both = norm(df[f"{pre}unit"]) == "both_eyes"
    return np.where(el & cl, "C1", np.where(el & both & ~cl, "C2", "C3"))


def flags(df, pre):
    """Binary indicators used for the corpus estimates (pre = 'm1_' reference coding, 'auto_' rules)."""
    g = {i: norm(df[f"{pre}{i}"]).values for i in ITEMS}
    E = g["eligible"] == "yes"
    B = E & (g["unit"] == "both_eyes")
    T = g["comparison"] == "statistical_test"
    C = g["uncertainty"] == "clustered"
    f = {"eligible": E, "both_eyes": B, "both_clustered": B & C, "both_patient_split": B & (g["split"] == "patient_level"),
         "both_image_split": B & (g["split"] == "image_or_eye_level"), "both_test": B & T, "both_test_clustered": B & T & C,
         "both_ack": B & (g["correlation_ack"] == "yes"), "both_independent": B & (g["uncertainty"] == "independent_data"),
         "both_none": B & (g["uncertainty"] == "none_reported"), "test": E & T, "test_clustered": E & T & C,
         "clustered": E & C, "ack": E & (g["correlation_ack"] == "yes")}
    for u in ("both_eyes", "one_eye", "multiple_images_unclear", "not_reported", "patient_level_model"):
        f[f"unit_{u}"] = E & (g["unit"] == u)
    return pd.DataFrame(f).astype(float)


# estimands: (label, numerator flag, denominator flag); 'all' = every article in the frame
ESTIMANDS = [
    ("Eligible (of all open-access full texts)", "eligible", "all"),
    ("Both eyes in the evaluation data (of eligible)", "both_eyes", "eligible"),
    ("One eye per patient (of eligible)", "unit_one_eye", "eligible"),
    ("Several images, eyes unclear (of eligible)", "unit_multiple_images_unclear", "eligible"),
    ("Eyes per patient not reported (of eligible)", "unit_not_reported", "eligible"),
    ("Patient-level model (of eligible)", "unit_patient_level_model", "eligible"),
    ("Clustered uncertainty (of both-eyes studies) [primary]", "both_clustered", "both_eyes"),
    ("Uncertainty assuming independence (of both-eyes studies)", "both_independent", "both_eyes"),
    ("No uncertainty reported (of both-eyes studies)", "both_none", "both_eyes"),
    ("Patient-level split (of both-eyes studies)", "both_patient_split", "both_eyes"),
    ("Image- or eye-level split (of both-eyes studies)", "both_image_split", "both_eyes"),
    ("Correlation acknowledged (of both-eyes studies)", "both_ack", "both_eyes"),
    ("Statistical test of models (of both-eyes studies)", "both_test", "both_eyes"),
    ("Clustered method (of both-eyes studies with a test)", "both_test_clustered", "both_test"),
    ("Clustered method (of all eligible studies with a test)", "test_clustered", "test"),
]


def two_phase(coded, frame_cells, rng, n_boot=2000):
    """Two-phase (double-sampling) stratified estimator: phase 1 = rules on every article in the frame, giving
    the cell sizes N_h; phase 2 = manual codes on a sample within each cell. Total = sum_h N_h * mean_h(flag);
    proportions are ratios of totals. 95% intervals: Jeffreys-prior posterior simulation within cells (primary) and
    percentile bootstrap resampling coded articles within cells (check)."""
    N = pd.Series(frame_cells).value_counts()
    F = flags(coded, "m1_")
    F["all"] = 1.0
    h = coded["cell"].values
    idx = {c: np.where(h == c)[0] for c in N.index}

    def totals(rows):
        return sum(N[c] * F.iloc[rows[c]].mean() for c in N.index)

    def ratios(t):
        return np.array([t[a] / t[b] if t[b] > 0 else np.nan for _, a, b in ESTIMANDS])

    t0 = totals(idx)
    est = ratios(t0)
    boot = np.array([ratios(totals({c: rng.choice(ix, size=len(ix), replace=True) for c, ix in idx.items()}))
                     for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    # Jeffreys-prior interval: within each cell, the three categories (in numerator; in denominator only; neither)
    # get a Dirichlet(n + 1/2) posterior, the multinomial form of the Jeffreys interval for a proportion
    # (Brown, Cai & DasGupta 2001). Unlike the bootstrap it does not collapse when a cell has no events.
    jl, jh = [], []
    for _, a, b in ESTIMANDS:
        num, den = np.zeros(10000), np.zeros(10000)
        for c in N.index:
            fa, fb = F[a].values[idx[c]], F[b].values[idx[c]]
            n = np.array([fa.sum(), (fb - fa).sum(), (1 - fb).sum()]) + 0.5
            p = rng.dirichlet(n, size=10000)
            num += N[c] * p[:, 0]
            den += N[c] * (p[:, 0] + p[:, 1])
        r = num / den
        jl.append(np.percentile(r, 2.5))
        jh.append(np.percentile(r, 97.5))
    return est, lo, hi, np.array(jl), np.array(jh), t0


def score(codes_path, codes2_path=None, n_boot=2000):
    rng = np.random.default_rng(SEED)
    val = pd.read_csv(os.path.join(DATA, "validation_sample.csv"), dtype=str).fillna("")
    val = val[val.include.str.lower() == "true"].drop(columns=[c for c in val.columns if c.startswith("m1_")])
    val = val.merge(read_codes(codes_path, "m1"), on="pmcid", how="inner")
    if len(val) < 200:
        print(f"Warning: only {len(val)} of 200 included articles have reference codes")
    val["cell"] = cells(val, "auto_")
    out, conf = [], []

    # 1. accuracy of the rules on the random core (items other than eligibility among reference-eligible articles)
    core = val[val["sample"] == "random"]
    for i in ITEMS:
        v = core if i == "eligible" else core[core.m1_eligible == "yes"]
        a, m = norm(v[f"auto_{i}"]).values, v[f"m1_{i}"].values
        po, k = kappa(a, m)
        bs = []
        for _ in range(n_boot):
            j = rng.integers(0, len(v), len(v))
            bs.append(kappa(a[j], m[j]))
        bs = np.array(bs, dtype=float)
        out.append({"analysis": "rules vs reference, random core", "item": i, "n": len(v), "agreement": po,
                    "agreement_lo": np.nanpercentile(bs[:, 0], 2.5), "agreement_hi": np.nanpercentile(bs[:, 0], 97.5),
                    "kappa": k, "kappa_lo": np.nanpercentile(bs[:, 1], 2.5), "kappa_hi": np.nanpercentile(bs[:, 1], 97.5)})
        tab = pd.crosstab(pd.Series(a, name="rules"), pd.Series(m, name="reference"))
        conf.append((i, tab))

    # 2. predictive values of the two enriched rule cells (all coded articles in the cell)
    ref = flags(val, "m1_")
    for c, lab, num in (("C1", "C1 (rules: eligible, clustered): reference eligible and clustered", "clustered"),
                        ("C1", "C1: reference eligible, both eyes and clustered", "both_clustered"),
                        ("C2", "C2 (rules: eligible, both eyes, not clustered): reference both eyes, not clustered",
                         None),
                        ("C2", "C2: reference both eyes", "both_eyes")):
        sel = (val.cell == c).values
        y = (ref["both_eyes"] - ref["both_clustered"]).values[sel] if num is None else ref[num].values[sel]
        n, k = int(sel.sum()), int(y.sum())
        lo, hi = wilson(k, n)
        out.append({"analysis": "predictive value of rule cell", "item": lab, "n": n, "agreement": k / n,
                    "agreement_lo": lo, "agreement_hi": hi})

    # 3. two-phase corpus estimates, with the rules-only estimate alongside
    ext = pd.read_csv(os.path.join(DATA, "extracted.csv"), low_memory=False)
    dev = set(pd.read_csv(DEV_LIST, dtype=str).pmcid)
    frame = ext[(ext.fulltext_source == "xml") & ~ext.pmcid.isin(dev)].copy()
    frame = frame.rename(columns={AUTO[i]: f"auto_{i}" for i in ITEMS})
    frame_cells = cells(frame, "auto_")
    est, lo, hi, jlo, jhi, t0 = two_phase(val, frame_cells, rng, n_boot)
    R = flags(frame, "auto_")
    R["all"] = 1.0
    rules_only = [R[a].sum() / R[b].sum() if R[b].sum() > 0 else np.nan for _, a, b in ESTIMANDS]
    core_ref = flags(core, "m1_")
    core_ref["all"] = 1.0
    srs = [(core_ref[a].sum(), core_ref[b].sum()) for _, a, b in ESTIMANDS]
    for (lab, a, b), e, l_, h_, jl, jh, r, (ka, kb) in zip(ESTIMANDS, est, lo, hi, jlo, jhi, rules_only, srs):
        out.append({"analysis": "corpus estimate (two-phase)", "item": lab, "n": len(frame), "estimate": e,
                    "estimate_lo": jl, "estimate_hi": jh, "boot_lo": l_, "boot_hi": h_,
                    "est_numerator_total": t0[a], "est_denominator_total": t0[b],
                    "rules_only": r, "random_core_k": int(ka), "random_core_n": int(kb)})
    cell_n = pd.Series(frame_cells).value_counts().sort_index()
    coded_n = val.cell.value_counts().sort_index()
    for c in cell_n.index:
        out.append({"analysis": "cell sizes", "item": c, "n": int(cell_n[c]), "coded": int(coded_n.get(c, 0))})

    # 4. agreement between two manual coders, if a second coding is supplied
    if codes2_path:
        m = val.merge(read_codes(codes2_path, "m2"), on="pmcid")
        for i in ITEMS:
            mm = m if i == "eligible" else m[(m.m1_eligible == "yes") & (m.m2_eligible == "yes")]
            po, k = kappa(mm[f"m1_{i}"], mm[f"m2_{i}"])
            out.append({"analysis": "coder 1 vs coder 2", "item": i, "n": len(mm), "agreement": po, "kappa": k})

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(DATA, "validation_results.csv"), index=False)
    with open(os.path.join(DATA, "validation_confusion.txt"), "w") as fh:
        for i, tab in conf:
            fh.write(f"{i}: rules (rows) vs reference (columns), random core\n{tab.to_string()}\n\n")
    pd.set_option("display.width", 200)
    for a, g in res.groupby("analysis", sort=False):
        print(f"\n== {a}")
        print(g.dropna(axis=1, how="all").drop(columns="analysis").to_string(index=False,
                                                                              float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {DATA}/validation_results.csv and validation_confusion.txt")


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def dev():
    codes = pd.read_csv(DEV_CODES, sep="\t", dtype=str).fillna("")
    df = pd.read_csv(os.path.join(DATA, "extracted.csv"), dtype=str)
    m = codes.merge(df[["pmcid"] + [AUTO[i] for i in ITEMS]].rename(columns={AUTO[i]: f"auto_{i}" for i in ITEMS}),
                    on="pmcid", how="left")
    res = pd.DataFrame(agreement(m, "rules vs coder 1 (development set)"))
    res.to_csv(os.path.join(DATA, "development_results.csv"), index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--dev", action="store_true")
    g.add_argument("--extend", action="store_true")
    ap.add_argument("--codes", default=os.path.join(DATA, "validation_codes_reference.tsv"),
                    help="reference (coder 1) codes: .tsv/.csv or the completed coding workbook (.xlsx)")
    ap.add_argument("--codes2", default=None, help="optional second coder's codes, for inter-coder agreement")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.make:
        make()
    elif a.extend:
        extend()
    elif a.score:
        score(a.codes, a.codes2, a.boot)
    else:
        dev()
