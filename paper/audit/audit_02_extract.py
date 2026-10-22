"""
Audit step 2: rule-based extraction and classification of every full text.

For each article it finds the sentences that describe (1) how many eyes and patients were analysed,
(2) how the data were split, (3) how uncertainty was quantified, (4) how models were compared and
(5) whether inter-eye correlation or clustering was acknowledged, and classifies each item.
Rules are regular expressions (listed below) applied to the Methods and Results text; no machine
learning is used, so the classification is deterministic and reproducible.

Rules version 2 (frozen 22 Oct 2026). Version 1 was revised on a development set of 276 articles:
40 coded blind by hand, plus the remaining articles of the first validation draw and the pilot, on
which every change in classification was checked by reading the matched sentences. The development
set is listed in development_set.csv; it is excluded from the validation sample (audit_03_validation.py).

Inputs   audit_data/records.jsonl, audit_data/fulltext/*.xml  (from audit_01_fetch.py)
         audit_data/pdf/*.pdf (optional: full texts obtained another way; named by DOI with / -> _)
Outputs  audit_data/extracted.csv          one row per article: flags, counts and classifications
         audit_data/snippets.jsonl         the matched sentences for each article (for checking)

Usage    python audit_02_extract.py
Requires Python 3.8+, pandas; pypdf only if PDFs are present.
"""
import csv
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

DATA = "audit_data"
RULES_VERSION = 2

# ----------------------------------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------------------------------
SEC_METHODS = re.compile(r"method|material|patient|participant|subject|dataset|data set|data collection|"
                         r"statistic|analys|experiment|evaluation|study design|study population|"
                         r"implementation|model|training|validation|ground truth|annotation|image acquisition", re.I)
SEC_RESULTS = re.compile(r"result|performance|finding|outcome", re.I)
SEC_SKIP = re.compile(r"reference|acknowledg|funding|conflict|disclosure|competing|author contrib|"
                      r"supplementary|abbreviation|data availability|ethic", re.I)
SEC_INTRO = re.compile(r"introduction|background|related work|literature|preliminar", re.I)
SEC_DISC = re.compile(r"discussion|conclusion|limitation|future", re.I)


def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _local(tag):
    return tag.split("}")[-1]


def parse_xml(path):
    """Return dict: article_type, title, abstract, sections [(section title, text)], captions."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    art = root if _local(root.tag) == "article" else root.find(".//article")
    art = art if art is not None else root
    out = {"article_type": art.get("article-type", ""), "title": "", "abstract": "", "sections": [],
           "captions": []}
    t = root.find(".//article-meta/title-group/article-title")
    out["title"] = _text(t) if t is not None else ""
    ab = root.find(".//article-meta/abstract")
    out["abstract"] = _text(ab) if ab is not None else ""
    body = root.find(".//body")
    if body is None:
        return out

    def walk(sec, path):
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        here = path + [title] if title else path
        for child in sec:
            tag = _local(child.tag)
            if tag == "sec":
                walk(child, here)
            elif tag == "p":
                out["sections"].append((" > ".join(here), _text(child)))
            elif tag in ("table-wrap", "fig"):
                cap = child.find(".//caption")
                if cap is not None:
                    out["captions"].append(_text(cap))

    walk(body, [])
    for tw in body.iter():
        if _local(tw.tag) in ("table-wrap", "fig"):
            cap = tw.find(".//caption")
            if cap is not None:
                out["captions"].append(_text(cap))
    out["captions"] = list(dict.fromkeys(out["captions"]))
    return out


def parse_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("PDFs found: install pypdf (python -m pip install pypdf) or remove the pdf folder")
    txt = " ".join((p.extract_text() or "") for p in PdfReader(path).pages)
    txt = re.sub(r"\s+", " ", txt)
    cut = re.search(r"\bReferences\b|\bREFERENCES\b", txt)
    if cut and cut.start() > len(txt) * 0.5:
        txt = txt[:cut.start()]
    return {"article_type": "", "title": "", "abstract": "", "sections": [("full text", txt)], "captions": []}


SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\[])")


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 3]


# ----------------------------------------------------------------------------------------------
# Rules. Each rule is a named regular expression applied sentence by sentence.
# ----------------------------------------------------------------------------------------------
NUM = r"(\d{1,3}(?:[,\s]\d{3})+|\d+)"
PEOPLE = r"(?:patients?|participants?|subjects?|individuals?|persons?|people|children|infants|adults|volunteers|cases|eyes? of \w+)"
RULES = {
    # --- eligibility
    "animal": r"\b(canine|dogs?|cats?|feline|mice|mouse|murine|rats?|rabbits?|porcine|pigs?|bovine|equine|horses?|"
              r"zebrafish|veterinary|primates?|macaques?|monkeys?)\b",
    "eye_tracking": r"\b(eye[- ]tracking|gaze (?:estimation|tracking)|eye movements?|saccad\w*|fixation tracking)\b",
    "test_set": r"\b(test(?:ing)? (?:set|cohort|dataset|data|sample)|held[- ]out|hold[- ]out|validation (?:set|cohort|dataset)|"
                r"external(?:ly)? validat\w*|cross[- ]validat\w*|independent (?:test|validation)|internal validation|"
                r"validation (?:study|cohort|data)|prospective(?:ly)? (?:validat\w*|evaluat\w*|stud\w*|trial)|"
                r"testing (?:phase|stage|images|data)|unseen (?:data|images|test))\b",
    "performance": r"\b(AUC|AUROC|area under|sensitivity|specificity|accuracy|F1|mean absolute error|MAE|RMSE|"
                   r"Dice|kappa|precision|recall|R\^?2|correlation coefficient|Brier)\b",
    "ocular": r"\b(eyes?|ocular|retina(?!net)\w*|fundus|fundi|OCT|optical coherence|cornea\w*|ophthalm\w*|glaucoma\w*|"
              r"macula\w*|keratoconus|cataract\w*|myopi\w*|uveitis|intraocular|visual acuity|optic (?:disc|disk|nerve)|"
              r"choroid\w*|vitre\w*|retinopath\w*|lens|iris|conjunctiv\w*|meibo\w*|pupil\w*|sclera\w*|electroretino\w*|ERG|"
              r"visual fields?|perimetr\w*|eyelids?|periocular|orbits?|orbital|lacrimal|strabism\w*|amblyop\w*|"
              r"blephar\w*|ptosis|pterygi\w*|retinoblastoma|keratit\w*|ROP)\b",   # retina(?!net): not the RetinaNet detector
    "review_title": r"\b((?:systematic|scoping|narrative|umbrella|literature|rapid|critical) review|meta-analy\w+|"
                    r"review protocol|study protocol|protocol for|review of the literature)\b|:\s*a review\b",
    "non_ocular_title": r"\b(fMRI|functional (?:magnetic resonance|MRI)|resting[- ]state|dermal|dermatolog\w*|cutaneous|skin)\b",
    # --- unit of analysis: counts of eyes and patients in the same sentence
    "eyes_of_patients": rf"{NUM}\s+(?:\w+\s+){{0,2}}eyes\s+(?:\w+\s+){{0,2}}(?:of|from|in)\s+{NUM}\s+(?:\w+\s+){{0,2}}{PEOPLE}",
    "patients_eyes": rf"{NUM}\s+(?:\w+\s+){{0,2}}{PEOPLE}\s*\(\s*{NUM}\s+eyes",
    "people_with_eyes": rf"{NUM}\s+(?:\w+\s+){{0,2}}{PEOPLE}\s*(?:\([^)]*\)\s*)?,?\s*(?:with|having|contributing|comprising)\s+"
                        rf"{NUM}\s+(?:\w+\s+){{0,2}}eyes",
    "eyes_paren_patients": rf"{NUM}\s+(?:\w+\s+){{0,2}}eyes\s*\(\s*{NUM}\s+(?:\w+\s+){{0,2}}{PEOPLE}\s*\)",
    "eyes_sum_patients": rf"{NUM}\s+(?:\w+\s+){{0,2}}and\s+{NUM}\s+(?:\w+\s+){{0,2}}eyes\s+(?:of|from)\s+{NUM}\s+"
                         rf"(?:\w+\s+){{0,2}}and\s+{NUM}\s+(?:\w+\s+){{0,2}}{PEOPLE}",      # 'a normal and b glaucoma eyes of c and d patients'
    "images_of": rf"{NUM}\s+(?:\w+\s+){{0,3}}(?:images|scans|photographs|photos|volumes|B-scans|signals|recordings|videos)\s+"
                 rf"(?:\w+\s+){{0,2}}(?:of|from)\s+(?:a total of\s+)?{NUM}\s+(?:\w+\s+){{0,2}}(?:eyes|{PEOPLE})",
    # unit of analysis: statements. Fellow eyes are counted as included only from inclusion statements
    # (not from conditional ones such as 'if both eyes were eligible' or 'one or both eyes', which are excluded below)
    "both_eyes": r"\b((?:images?|photographs?|photos|scans|B-scans|volumes|data|measurements|recordings|videos?|OCT\w*|"
                 r"fundus \w+) (?:\w+ ){0,3}(?:from|of|for) (?:the )?(?:\w+ )?(?:both|two|fellow|left and right|right and left) eyes|"
                 r"(?:of|from|on) both eyes\b[^.]{0,60}\b(?:were|was) (?:\w+ ){0,2}(?:recorded|acquired|obtained|captured|"
                 r"collected|included|used|analy[sz]ed|retrieved)|"
                 r"(?<!in )both eyes (?:\w+ ){0,3}(?:were|was|are) (?:\w+ ){0,2}(?:included|enrolled|collected|used|"
                 r"analy[sz]ed|imaged|captured|recruited|scanned|photographed|considered)|"
                 r"(?:included|enrolled|used|analy[sz]ed|collected|imaged|scanned|contribut\w*|captur\w*|acquir\w*|"
                 r"record\w*|obtain\w*) (?:\w+ ){0,3}both eyes|"
                 r"both (?:the )?(?:left and right|right and left) eyes|"
                 r"(?:both|two) eyes (?:of|from) (?:the same |each |every |all |the )?(?:\w+ )?(?:patients?|participants?|subjects?|individuals?|"
                 r"persons?|child(?:ren)?)|eyes (?:\w+ ){0,2}(?:from|of) the same (?:patient|participant|subject|individual|"
                 r"person)|bilateral(?:ly)? (?:eyes|images|data|scans|photographs)|"
                 r"fellow eyes? (?:\w+ ){0,3}(?:were|was) (?:also )?(?:included|enrolled|used|analy[sz]ed|retrieved|collected|"
                 r"imaged|obtained))\b",
    "one_eye": r"\b(one eye (?:per|of each|from each|for each|of every) (?:patient|participant|subject|individual|person|"
               r"child|case)|single eye (?:per|from each|of each)|"
               r"(?:one|a single) eye (?:\w+ ){0,4}(?:was|were) (?:\w+ ){0,2}(?:selected|chosen|picked)|"
               r"only (?:one|a single) eye (?:per|of each|from each|for each|(?:\w+ ){0,3}(?:was|were) (?:\w+ )?"
               r"(?:selected|chosen|included|used|analy[sz]ed|enrolled|kept|retained))|"
               r"only (?:the )?(?:right|left|better|worse|study|first|operated|affected) eyes?(?!['’]s)|(?<!and )(?<!or )(?:right|left) eyes? only|"
               r"random(?:ly)? (?:selected|chose\w*|chosen|picked|selection of) (?:one |a single |either |the )?eye|"
               r"(?<!and )(?<!or )(?<!both )(?:the )?(?:right|left|better|worse|study|first|operated|more severely affected|better-seeing|worse-seeing) "
               r"eyes? (?:\w+ ){0,3}(?:was|were) (?:\w+ ){0,2}(?:used|selected|chosen|included|analy[sz]ed)|"
               r"(?:used|selected|chose|chosen|included|analy[sz]ed|took) (?:\w+ ){0,4}(?<!and )(?<!or )(?:the )?(?:right|left) eyes?\b(?! and| or)|"
               r"(?:if|when|where) both eyes (?:\w+ ){0,6}(?:eligible|met|qualified|fulfil\w*|satisf\w*|included|available|"
               r"affected)[^.]{0,120}\b(?:one|right|left|better|worse|first|random\w*|only|higher|lower|more)\b|"
               r"one eye (?:\w+ ){0,3}(?:to (?:avoid|ensure|preserve|maintain)|because of|due to) (?:\w+ ){0,4}"
               r"(?:correlat\w*|independen\w*|clustering))",
    # --- data splitting (leakage)
    "split_patient": r"\b((?:split|partition|divid|separat|assign|allocat|fold|cross[- ]validation|leak|overlap)\w*[^.]{0,80}(?:patient|subject|participant|individual)[- ](?:level|wise|based|disjoint)|"
                     r"(?:patient|subject|participant|individual)[- ](?:level|wise|based|disjoint)[^.]{0,40}(?:split|partition|divi|separat|assign|fold|cross[- ]validation)\w*|"
                     r"(?:split|divided|partitioned|separated|stratified|assigned|allocated|randomi[sz]ed)\w* (?:\w+ ){0,4}(?:by|at the level of|on the basis of|according to|based on) (?:the )?(?:patients?|subjects?|participants?|individuals?)|"
                     r"no (?:patient|subject|participant|individual)s? (?:\w+ ){0,3}(?:overlap\w*|appeared|was included|were included) (?:\w+ ){0,4}(?:both|more than one|multiple|different)|"
                     r"(?:same|single) (?:patient|subject|participant|individual)s? (?:\w+ ){0,6}(?:not|never|only)\w* (?:\w+ ){0,4}(?:both|more than one|multiple|different|same) (?:sets?|partitions?|subsets?|groups?|folds?)|"
                     r"(?:split|divided|partitioned|separated|assigned|allocated|selected)\w* (?:\w+ ){0,4}on (?:a|the) (?:patient|subject|participant|individual)[- ]?(?:basis|level)|"
                     r"(?:selected|split|divided|assigned|allocated|partitioned|separated)\w*[^.]{0,30}\(?based on (?:different|individual|unique|separate) (?:patients|subjects|participants|individuals)|"
                     r"(?:same|one|each|a single|a given) (?:patient|subject|participant|individual)s?\b[^.]{0,80}\b(?:only|exclusively|same|single|one)\b[^.]{0,30}\b(?:sets?|folds?|subsets?|partitions?|splits?)\b|"
                     r"group(?:ed)? k[- ]?fold|GroupKFold|patient overlap)\b",
    "split_image": r"\b((?:image|eye|sample|scan|slice)[- ](?:level|wise|based) (?:split|partition|division)\w*|"
                   r"(?:split|partition|divi)\w*[^.]{0,60}(?:at|on) the (?:image|eye|sample|scan)[- ]level|"
                   r"randomly (?:split|divided|partitioned) (?:\w+ ){0,3}images|images were randomly (?:split|divided|assigned|partitioned))\b",
    # --- uncertainty
    "ci_reported": r"\b(95\s?%\s?(?:CI|confidence interval)|confidence intervals?|\bCI\b)\b",
    "delong": r"\bDe ?Long\b",
    "bootstrap": r"\bbootstrap\w*\b",
    "boot_patient": r"\bbootstrap\w*\b[^.]{0,120}\b(patients?|subjects?|participants?|individuals?|clusters?|cluster[- ]level|by patient|at the patient)\b|"
                    r"\b(cluster\w*|patient[- ]level|subject[- ]level|participant[- ]level|block) bootstrap\w*",
    "exact_binomial": r"\b(Clopper[- ]Pearson|Wilson|exact binomial|binomial exact|Agresti)\b",
    "clustered_method": r"\b(Obuchowski|generali[sz]ed estimating equations?|GEE\b|mixed[- ]effects?|random[- ]effects?|"
                        r"multilevel|hierarchical (?:model|logistic)|cluster[- ]robust|robust (?:standard errors?|variance)|"
                        r"sandwich|cluster(?:ed)? (?:data|bootstrap\w*|ROC|AUC|analysis)|clustering (?:effect|within|by)|"
                        r"Rosner|Datta|marginal model)\b",
    "correlation_ack": r"\b(inter[- ]?eye correlation|intereye|correlation between (?:the )?(?:two |both |fellow )?eyes|"
                       r"(?:non-?|not )independen\w* (?:of |between )?(?:the )?(?:two )?eyes|fellow[- ]eye correlation|"
                       r"(?:within|intra)[- ](?:patient|subject|individual|cluster) correlation|account(?:ed|ing)? for (?:the )?"
                       r"(?:correlation|clustering|non-?independence|dependence)|correlated eyes?|between[- ]eye correlation|"
                       r"inter-?ocular correlation|(?:preserv|ensur|maintain|satisf|guarantee|avoid)\w*\s+(?:\w+\s+){0,4}"
                       r"(?:statistical\s+)?(?:in|non-?)dependence|eyes?\b[^.]{0,100}\b(?:regarded|considered|assumed|treated) "
                       r"(?:as )?(?:highly |strongly )?correlated|random (?:participant|patient|subject|individual)[- ]?(?:level )?"
                       r"(?:intercepts?|effects?)|(?:multiple|several|repeated|more than one) (?:\w+ ){0,2}(?:of|from|per) "
                       r"(?:the same|each|a single) (?:eye|patient|participant|subject|individual)s?\b[^.]{0,120}\b"
                       r"(?:cluster\w*|correlat\w*|independen\w*|mixed|GEE))\b",
    # --- model comparison
    "compare_test": r"\b(DeLong(?:'s)? test|McNemar\w*|paired t[- ]?test|Wilcoxon|Mann[- ]Whitney|chi[- ]?square\w*|"
                    r"permutation test|Friedman|Kruskal|ANOVA|Student'?s t[- ]?test|t[- ]tests?|z[- ]test|"
                    r"Fisher'?s exact|Cochran'?s? Q|heteroscedastic test)\b",
    "compare_models": r"\b(compar\w+ (?:\w+ ){0,6}(?:models?|algorithms?|networks?|architectures?|methods?|systems?)|"
                      r"(?:models?|algorithms?|networks?|architectures?) (?:\w+ ){0,4}compar\w+|"
                      r"(?:significantly|statistically) (?:higher|lower|better|worse|different|outperform\w*)|"
                      r"outperform\w*|superior to|best[- ]perform\w*|(?:the )?best (?:\w+ ){0,2}(?:model|algorithm|network|classifier))\b",
    "p_value": r"\b[Pp]\s*(?:[<=≤]|values?\s*(?:[<=≤]|of))\s*0?\.\d",
}
CRE = {k: re.compile(v, re.I if k not in ("performance",) else 0) for k, v in RULES.items()}
CRE["performance"] = re.compile(RULES["performance"])          # case-sensitive (AUC, MAE ...)

# Sentence-level context. A sentence counts for a rule only if it does not also match EXCLUDE[rule] and,
# where REQUIRE[rule] is given, also matches it.
NEG = (r"\b(?:did|do|does|was|were|could|can|is|are) not\b|\bnot (?:be )?(?:computed|calculated|performed|used|"
       r"estimated|reported)\b")
MODEL_CTX = (r"\b(models?|algorithms?|networks?|classifiers?|architectures?|AI|CNNs?|deep[- ]learning|machine[- ]learning|"
             r"AUCs?|AUROCs?|ROC|area under|formulas?|performance|accurac(?:y|ies)|ACCs?|LLMs?|large language models?|"
             r"ChatGPT|GPT-?\w*|Gemini|chatbots?|DeLong|McNemar\w*)\b")
EXCLUDE = {
    "bootstrap": rf"bootstrap\w* aggregat|\bbagg(?:ing|ed)\b|{NEG}",                     # bagging is not a CI method
    "boot_patient": rf"bootstrap\w* aggregat|\bbagg(?:ing|ed)\b|{NEG}",
    "ci_reported": rf"\b(?:ICC|intra-?class|inter-?(?:rater|grader|observer|reader)|intra-?(?:rater|grader|observer|reader))\b|"
                   rf"between (?:the )?(?:two |three )?(?:graders|ophthalmologists|observers|raters|readers)|{NEG}",
    "compare_test": r"\bfeature\w* (?:selection|filter\w*|ranking|screening)|\bselect\w* (?:\w+ ){0,2}features|redundan\w*",
    "split_image": r"\b(?:graders?|readers?|annotators?|raters?|observers?|ophthalmologists)\b",   # assigned to graders
    "both_eyes": r"\b(?:if|when|whether|unless)\b[^.]{0,40}\bboth eyes|\bone or both\b|\beither eye|\bboth eyes or\b",
}
REQUIRE = {"compare_test": MODEL_CTX}                 # a test counts as a model comparison only in a model context
CEXCLUDE = {k: re.compile(v, re.I) for k, v in EXCLUDE.items()}
CREQUIRE = {k: re.compile(v, re.I) for k, v in REQUIRE.items()}


def counts_for(rule, s):
    return (bool(CRE[rule].search(s)) and not (rule in CEXCLUDE and CEXCLUDE[rule].search(s))
            and (rule not in CREQUIRE or bool(CREQUIRE[rule].search(s))))


def _int(s):
    return int(re.sub(r"[,\s]", "", s))


def analyse(doc):
    """Apply every rule; return (flags dict, snippets dict)."""
    sec_text = [(t, s) for t, s in doc["sections"] if not SEC_SKIP.search(t or "")]
    core = [(t, s) for t, s in sec_text if not SEC_INTRO.search(t or "") and not SEC_DISC.search(t or "")]
    disc = [s for t, s in sec_text if SEC_DISC.search(t or "")]
    if not core:                                                     # no recognisable headings: use everything
        core = sec_text
    sents_all = [x for _, p in core for x in sentences(p)] + doc["captions"]   # methods, results, captions
    sents_meth = [x for t, p in core if not SEC_RESULTS.search(t or "") for x in sentences(p)]
    sents_disc = [x for p in disc for x in sentences(p)]
    abstract = doc["abstract"]
    head = doc["title"] + " " + abstract
    snips = defaultdict(list)
    flags = {}

    def hits(rule, sents, keep=4):
        found = [s for s in sents if counts_for(rule, s)]
        if found:
            snips[rule].extend(found[:keep])
        return len(found)

    flags["n_words"] = sum(len(p.split()) for _, p in sec_text)
    body = [x for _, p in sec_text for x in sentences(p)] + doc["captions"]      # all text except references etc.
    flags["n_ocular_terms"] = sum(len(CRE["ocular"].findall(x)) for x in body)
    flags["n_ocular_title_abstract"] = len(CRE["ocular"].findall(head))
    flags["review_title"] = bool(CRE["review_title"].search(doc["title"]))
    flags["non_ocular_title"] = bool(CRE["non_ocular_title"].search(doc["title"]))
    flags["animal_title_abstract"] = bool(CRE["animal"].search(head))
    flags["eye_tracking_title_abstract"] = bool(CRE["eye_tracking"].search(head))
    flags["test_set"] = hits("test_set", sents_all, 2) > 0
    flags["performance"] = hits("performance", sents_all, 2) > 0

    # unit of analysis: counts of eyes vs patients
    pairs = []
    for s in sents_all + sentences(abstract):
        for m in CRE["eyes_of_patients"].finditer(s):
            pairs.append((_int(m.group(1)), _int(m.group(2)), s))
        for m in CRE["eyes_paren_patients"].finditer(s):
            pairs.append((_int(m.group(1)), _int(m.group(2)), s))
        for rule in ("patients_eyes", "people_with_eyes"):
            for m in CRE[rule].finditer(s):
                pairs.append((_int(m.group(2)), _int(m.group(1)), s))
        for m in CRE["eyes_sum_patients"].finditer(s):
            pairs.append((_int(m.group(1)) + _int(m.group(2)), _int(m.group(3)) + _int(m.group(4)), s))
    for e, p, s in pairs[:6]:
        snips["eyes_of_patients"].append(s)
    flags["n_eye_patient_pairs"] = len(pairs)
    flags["eyes_gt_patients"] = any(e > p for e, p, _ in pairs)
    flags["eyes_eq_patients"] = any(e == p for e, p, _ in pairs) and not flags["eyes_gt_patients"]
    img = []
    for s in sents_all + sentences(abstract):
        for m in CRE["images_of"].finditer(s):
            img.append((_int(m.group(1)), _int(m.group(2)), s))
    flags["images_gt_units"] = any(i > u for i, u, _ in img)
    for i, u, s in img[:3]:
        snips["images_of"].append(s)
    for rule in ("both_eyes", "one_eye"):
        flags[rule] = hits(rule, sents_meth) > 0                         # Methods only (Results describe findings)
    for rule in ("split_patient", "split_image"):
        flags[rule] = hits(rule, sents_all) > 0
    for rule in ("ci_reported", "delong", "bootstrap", "boot_patient", "exact_binomial", "clustered_method",
                 "compare_test", "compare_models"):
        flags[rule] = hits(rule, sents_all) > 0
    # a stated significant difference between models with a P value also counts as a statistical comparison
    sig = [s for s in sents_all if counts_for("compare_models", s) and CRE["p_value"].search(s)
           and CREQUIRE["compare_test"].search(s)]
    if sig:
        snips["compare_test"].extend(sig[:2])
        flags["compare_test"] = True
    flags["correlation_ack"] = hits("correlation_ack", sents_all + sents_disc) > 0      # Discussion counts here
    flags["split_patient_disc"] = hits("split_patient", sents_disc, 2) > 0              # e.g. limitations
    return flags, dict(snips)


# ----------------------------------------------------------------------------------------------
# Classification (pre-specified; each class is a simple combination of the flags above)
# ----------------------------------------------------------------------------------------------
MIN_OCULAR_TERMS = 5          # ocular terms in the full text (references excluded)
MIN_OCULAR_HEAD = 2           # ocular terms in the title and abstract


def classify(f, article_type):
    c = {}
    c["eligible_auto"] = (article_type in ("research-article", "") and not f["review_title"]
                          and not f["non_ocular_title"] and not f["animal_title_abstract"]
                          and not f["eye_tracking_title_abstract"] and f["performance"]
                          and f["n_ocular_title_abstract"] >= MIN_OCULAR_HEAD and f["n_ocular_terms"] >= MIN_OCULAR_TERMS)
    # unit of analysis, in order: (1) counts of eyes and patients; (2) an explicit one-eye selection statement;
    # (3) an explicit statement that both eyes were included; (4) more images than eyes or patients
    if f["eyes_gt_patients"]:
        c["unit"] = "both_eyes"
    elif f["eyes_eq_patients"] or f["one_eye"]:
        c["unit"] = "one_eye"
    elif f["both_eyes"]:
        c["unit"] = "both_eyes"
    elif f["images_gt_units"]:
        c["unit"] = "multiple_images_unclear"
    else:
        c["unit"] = "not_reported"
    c["split"] = ("image_or_eye_level" if f["split_image"] else
                  "patient_level" if f["split_patient"] else "not_reported")
    c["uncertainty"] = ("clustered" if (f["clustered_method"] or f["boot_patient"]) else
                        "independent_data" if (f["delong"] or f["bootstrap"] or f["exact_binomial"] or f["ci_reported"])
                        else "none_reported")
    c["comparison"] = ("statistical_test" if f["compare_test"] else
                       "descriptive_only" if f["compare_models"] else "none")
    c["clustering_handled"] = c["uncertainty"] == "clustered"
    return c


def doi_key(doi):
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi)


def main():
    records = [json.loads(l) for l in open(os.path.join(DATA, "records.jsonl"), encoding="utf-8")]
    pdf_dir = os.path.join(DATA, "pdf")
    pdfs = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(pdf_dir, "*.pdf"))}
    rows, n_done = [], 0
    with open(os.path.join(DATA, "snippets.jsonl"), "w", encoding="utf-8") as fs:
        for r in records:
            pmcid, doi = r.get("pmcid", ""), r.get("doi", "")
            xml_path = os.path.join(DATA, "fulltext", f"{pmcid}.xml") if pmcid else ""
            source, doc = "", None
            if xml_path and os.path.exists(xml_path):
                doc, source = parse_xml(xml_path), "xml"
            elif doi and doi_key(doi) in pdfs:
                doc, source = parse_pdf(pdfs[doi_key(doi)]), "pdf"
            base = {"id": r.get("id"), "pmid": r.get("pmid", ""), "pmcid": pmcid, "doi": doi,
                    "journal": r.get("journalInfo", {}).get("journal", {}).get("medlineAbbreviation", ""),
                    "pub_year": r.get("pubYear", ""), "license": r.get("license", ""),
                    "title": (r.get("title") or "").strip(), "fulltext_source": source or "none",
                    "rules_version": RULES_VERSION}
            if doc is None:
                rows.append(base)
                continue
            flags, snips = analyse(doc)
            cls = classify(flags, doc["article_type"])
            rows.append({**base, "article_type": doc["article_type"], **cls, **flags})
            fs.write(json.dumps({"id": r.get("id"), "pmcid": pmcid, "doi": doi, "snippets": snips},
                                ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 250 == 0:
                print(f"  {n_done} full texts processed", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "extracted.csv"), index=False, encoding="utf-8")

    # summary for the console
    print(f"\nRules version {RULES_VERSION}. Records: {len(df)}; full text processed: {n_done} "
          f"(xml {int((df.fulltext_source == 'xml').sum())}, pdf {int((df.fulltext_source == 'pdf').sum())})")
    el = df[df.get("eligible_auto", False) == True]  # noqa: E712
    print(f"Eligible by the automated screen: {len(el)}")
    if len(el):
        for col in ("unit", "split", "uncertainty", "comparison"):
            print(f"\n{col}:\n" + el[col].value_counts().to_string())
        both = el[el.unit == "both_eyes"]
        if len(both):
            print(f"\nOf {len(both)} eligible studies with both eyes of some patients: "
                  f"clustering handled in {int(both.clustering_handled.sum())} "
                  f"({100 * both.clustering_handled.mean():.1f}%); inter-eye correlation mentioned in "
                  f"{int(both.correlation_ack.sum())} ({100 * both.correlation_ack.mean():.1f}%)")


if __name__ == "__main__":
    main()
