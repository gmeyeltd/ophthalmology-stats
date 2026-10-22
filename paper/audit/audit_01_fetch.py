"""
Audit step 1: identify ophthalmic AI evaluation papers in Europe PMC and download open-access full texts.

What it does
  1. Runs one fixed Europe PMC query (title/abstract terms; research articles indexed in MEDLINE;
     publication years set below) and saves every record found.
  2. Downloads the full-text XML of every open-access record that has one (resumable: files
     already downloaded are skipped). If Europe PMC cannot serve an article, the same open-access
     article is taken from NCBI's PubMed Central instead.
  3. Writes a list of DOIs for the paywalled records, ready to import into Zotero.

Outputs (in ./audit_data/)
  query.txt               the exact query, run date and counts (for the Methods section)
  records.jsonl           one Europe PMC record per line (includes the abstract)
  records.csv             key fields per record
  fulltext/PMCxxxx.xml    open-access full texts
  retrieval_log.csv       status of every download
  paywalled_dois.txt      DOIs of records without a retrieved full text (paywalled, or failed downloads)

Usage
  python audit_01_fetch.py              # full run (about 3,000 downloads; roughly 30-60 min)
  python audit_01_fetch.py --max 20     # quick test on the first 20 records

Requires Python 3.8+ and no extra packages.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------------------------------
# Query (fixed; record any change in query.txt)
# ----------------------------------------------------------------------------------------------
YEAR_FROM, YEAR_TO = 2020, 2026

AI_TERMS = ('TITLE_ABS:("deep learning" OR "machine learning" OR "artificial intelligence" OR '
            '"neural network" OR "convolutional" OR "transformer" OR "foundation model")')
EYE_TERMS = ('TITLE_ABS:(ophthalm* OR retina* OR retinopathy OR glaucoma OR macula* OR '
             '"optical coherence tomography" OR fundus OR cornea* OR keratoconus OR cataract OR '
             'myopia OR uveitis OR "intraocular" OR "visual acuity")')
PERFORMANCE_TERMS = ('TITLE_ABS:("area under" OR AUC OR AUROC OR "mean absolute error" OR '
                     'sensitivity OR specificity OR accuracy OR "F1")')
EXCLUDE = ('NOT PUB_TYPE:"review" NOT PUB_TYPE:"systematic-review" NOT PUB_TYPE:"meta-analysis" '
           'NOT PUB_TYPE:"preprint" AND SRC:MED')
QUERY = (f'{AI_TERMS} AND {EYE_TERMS} AND {PERFORMANCE_TERMS} {EXCLUDE} '
         f'AND PUB_YEAR:[{YEAR_FROM} TO {YEAR_TO}]')

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUT = "audit_data"
WORKERS = 4            # parallel downloads; Europe PMC asks users to be considerate
PAUSE = 0.25           # seconds between requests per worker
USER_AGENT = "both-eyes-ai-audit/1.0 (academic research; contact via GitHub gmeyeltd)"

CSV_FIELDS = ["id", "source", "pmid", "pmcid", "doi", "title", "journal", "pub_year",
              "pub_types", "is_open_access", "license", "in_epmc", "has_fulltext_xml", "cited_by"]


def get(url, tries=5):
    """GET with retries and exponential back-off. Returns bytes."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            wait = 2 ** k
        except Exception:
            wait = 2 ** k
        time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def search_all(max_records=None):
    """Page through the query with cursorMark. Returns a list of 'core' records."""
    records, cursor = [], "*"
    while True:
        params = {"query": QUERY, "format": "json", "resultType": "core",
                  "pageSize": 1000, "cursorMark": cursor}
        data = json.loads(get(f"{API}/search?" + urllib.parse.urlencode(params)))
        page = data.get("resultList", {}).get("result", [])
        records.extend(page)
        print(f"  {len(records)} / {data.get('hitCount')} records", flush=True)
        nxt = data.get("nextCursorMark")
        if not page or nxt == cursor or (max_records and len(records) >= max_records):
            break
        cursor = nxt
        time.sleep(PAUSE)
    return records[:max_records] if max_records else records, data.get("hitCount")


def flatten(r):
    pub_types = r.get("pubTypeList", {}).get("pubType", [])
    return {
        "id": r.get("id"), "source": r.get("source"), "pmid": r.get("pmid", ""),
        "pmcid": r.get("pmcid", ""), "doi": r.get("doi", ""),
        "title": (r.get("title") or "").strip(),
        "journal": r.get("journalInfo", {}).get("journal", {}).get("medlineAbbreviation", ""),
        "pub_year": r.get("pubYear", ""),
        "pub_types": "; ".join(pub_types if isinstance(pub_types, list) else [pub_types]),
        "is_open_access": r.get("isOpenAccess", ""), "license": r.get("license", ""),
        "in_epmc": r.get("inEPMC", ""),
        "has_fulltext_xml": "Y" if (r.get("pmcid") and r.get("isOpenAccess") == "Y") else "N",
        "cited_by": r.get("citedByCount", ""),
    }


def fetch_xml(pmcid, folder):
    path = os.path.join(folder, f"{pmcid}.xml")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return pmcid, "already_downloaded", os.path.getsize(path)
    try:
        xml = get(f"{API}/{pmcid}/fullTextXML", tries=3)
        status = "downloaded"
    except Exception as e:
        # fallback: the same open-access article from NCBI's PubMed Central (E-utilities)
        try:
            xml = get(f"{NCBI}?db=pmc&id={pmcid.replace('PMC', '')}", tries=3)
            if b"<body" not in xml:
                raise RuntimeError("no body in NCBI record")
            status = "downloaded_ncbi"
        except Exception as e2:
            return pmcid, f"error: Europe PMC {e}; NCBI {e2}", 0
    with open(path, "wb") as f:
        f.write(xml)
    time.sleep(PAUSE)
    return pmcid, status, len(xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="only the first N records (testing)")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "fulltext"), exist_ok=True)
    run_date = dt.date.today().isoformat()

    print("Searching Europe PMC ...")
    records, hit_count = search_all(args.max)

    with open(os.path.join(OUT, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [flatten(r) for r in records]
    with open(os.path.join(OUT, "records.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    to_get = [r["pmcid"] for r in rows if r["has_fulltext_xml"] == "Y"]
    print(f"Downloading {len(to_get)} open-access full texts ...")
    log = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_xml, p, os.path.join(OUT, "fulltext")) for p in to_get]
        for i, fu in enumerate(as_completed(futs), 1):
            log.append(fu.result())
            if i % 100 == 0 or i == len(futs):
                print(f"  {i} / {len(futs)}", flush=True)
    with open(os.path.join(OUT, "retrieval_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "status", "bytes"])
        w.writerows(sorted(log))

    failed = {p for p, st, _ in log if st not in ("downloaded", "downloaded_ncbi", "already_downloaded")}
    paywalled = [r["doi"] for r in rows if (r["has_fulltext_xml"] == "N" or r["pmcid"] in failed) and r["doi"]]
    with open(os.path.join(OUT, "paywalled_dois.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paywalled) + "\n")

    ok = sum(1 for _, s, _ in log if s in ("downloaded", "downloaded_ncbi", "already_downloaded"))
    summary = (f"Run date: {run_date}\nQuery:\n{QUERY}\n\n"
               f"Records reported by Europe PMC: {hit_count}\nRecords saved: {len(rows)}\n"
               f"Open-access full text available: {len(to_get)}\nFull texts retrieved: {ok}\n"
               f"Failed downloads: {len(to_get) - ok}\n"
               f"No full text retrieved (DOIs listed for Zotero): {len(paywalled)} "
               f"(+{sum(1 for r in rows if r['has_fulltext_xml'] == 'N' and not r['doi'])} without DOI)\n")
    with open(os.path.join(OUT, "query.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    sys.exit(main())
