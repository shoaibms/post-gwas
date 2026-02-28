"""
Fetch JASPAR Plants CORE motif metadata (family, class) via the JASPAR REST API.

Outputs a TSV with matrix_id, name, family, class, collection, base_id, version.
"""

import concurrent.futures as cf
from pathlib import Path
import requests, pandas as pd

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUTDIR = BASE / "data" / "motifs" / "jaspar_plants"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "jaspar_plants_metadata.tsv"

API = "https://jaspar.elixir.no/api/v1/matrix/"
PARAMS = {
    "collection": "CORE",
    "tax_group": "Plants",
    "version": "latest",
    "page_size": 1000,   # big pages to cut roundtrips
}
HEADERS = {
    "User-Agent": "gwas-postgwas/1.0"
}

def make_session():
    s = requests.Session()
    # Robust retries for transient errors / 429s
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    retry = Retry(
        total=6, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.headers.update(HEADERS)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64))
    return s

def paged_list(session):
    url = API
    params = PARAMS.copy()
    while url:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        yield from data["results"]
        url = data.get("next")
        params = None  # 'next' already carries the params

def fetch_detail(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()
    return {
        "matrix_id": d.get("matrix_id"),
        "name": d.get("name"),
        "family": d.get("family"),
        "class": d.get("class"),
        "collection": d.get("collection"),
        "base_id": d.get("base_id"),
        "version": d.get("version"),
    }

def main():
    session = make_session()
    print("Fetching JASPAR Plants CORE (large pages, parallel details)...")
    items = list(paged_list(session))  # typically a few thousand at most

    # Try to use fields from list payload if present (often already includes family/class)
    rows = []
    missing = []
    for rec in items:
        row = {
            "matrix_id": rec.get("matrix_id"),
            "name": rec.get("name"),
            "family": rec.get("family"),
            "class": rec.get("class"),
            "collection": rec.get("collection"),
            "base_id": rec.get("base_id"),
            "version": rec.get("version"),
        }
        if row["family"] is None or row["class"] is None:
            # fall back to detail endpoint only when needed
            if "url" in rec:
                missing.append(rec["url"])
        else:
            rows.append(row)

    if missing:
        # Use moderate parallelism to be polite to the public API
        max_workers = 12  # tune if needed; keep reasonable for a public service
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_detail, session, u) for u in missing]
            for fut in cf.as_completed(futures):
                rows.append(fut.result())

    df = (pd.DataFrame(rows)
            .dropna(subset=["matrix_id"])
            .drop_duplicates(subset=["matrix_id"])
            .sort_values("name"))
    df.to_csv(OUT, sep="\t", index=False)
    print(f"Saved: {OUT}  ({len(df)} motifs)")

if __name__ == "__main__":
    main()
