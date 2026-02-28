#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attach JASPAR TF family/class to motif delta-LLR outputs.
- Primary: join by JASPAR matrix_id from metadata TSV.
- Fallback: regex-based family mapping on tf_name (WRKY, bZIP, ERF/AP2, DOF, NAC, MYB, bHLH, GATA, TCP, HD-ZIP, MADS/AGL, HSF, ARF, BES1/BZR).
Inputs:
  output/week6_ieqtl/motif_delta_llr_all.csv
  output/week6_ieqtl/motif_delta_llr_top_per_snp.csv
  data/motifs/jaspar_plants/JASPAR_CORE_plants_metadata.tsv
Outputs:
  motif_delta_llr_all_annot.csv
  motif_delta_llr_top_per_snp_annot.csv
"""

import re
import pandas as pd
from pathlib import Path
import ast
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT  = BASE / "output" / "week6_ieqtl"
META_DIR = BASE / "data" / "motifs" / "jaspar_plants"

ALL_IN   = OUT / "motif_delta_llr_all.csv"
TOP_IN   = OUT / "motif_delta_llr_top_per_snp.csv"
META_IN  = META_DIR / "JASPAR_CORE_plants_metadata.tsv"

ALL_OUT  = OUT / "motif_delta_llr_all_annot.csv"
TOP_OUT  = OUT / "motif_delta_llr_top_per_snp_annot.csv"
ENR_OUT  = OUT / "motif_family_enrichment_top_vs_all.csv"

print("="*60)
print("ANNOTATING MOTIF RESULTS WITH JASPAR FAMILIES")
print("="*60)

# ---------- load results ----------
def load_csv(path):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    df = pd.read_csv(path)
    return df

all_scores = load_csv(ALL_IN)
top_scores = load_csv(TOP_IN)

# Expect columns from your motif script:
# ['gene','snp','chr','pos','ref','alt','matrix_id','tf_name','ref_llr','alt_llr','delta_llr','abs_delta_llr']
for need in ["matrix_id","tf_name"]:
    if need not in all_scores.columns:
        print(f"WARNING: '{need}' not found in all-scores. "
              f"Will attempt regex-only family mapping.")

# ---------- load metadata (preferred) ----------
def load_meta(path):
    if not path.exists():
        print(f"WARNING: Metadata TSV not found: {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    # normalize column names that vary across versions
    cols = {c.lower():c for c in df.columns}
    # common keys
    id_col = None
    for k in ["matrix_id","base_id","id"]:
        if k in cols: id_col = cols[k]; break
    name_col = None
    for k in ["name","tf_name","matrix_name"]:
        if k in cols: name_col = cols[k]; break
    fam_col = None
    for k in ["family","tf_family","family_name"]:
        if k in cols: fam_col = cols[k]; break
    cls_col = None
    for k in ["class","tf_class","class_name"]:
        if k in cols: cls_col = cols[k]; break

    keep = {}
    if id_col:  keep["matrix_id"] = df[id_col].astype(str)
    if name_col:keep["meta_name"] = df[name_col].astype(str)
    if fam_col: keep["meta_family"] = df[fam_col].astype(str)
    if cls_col: keep["meta_class"] = df[cls_col].astype(str)
    m = pd.DataFrame(keep)
    if "matrix_id" in m.columns:
        # make sure the IDs match form 'MA####.#'
        m["matrix_id"] = m["matrix_id"].str.extract(r'(MA\d{4}\.\d)')[0].fillna(m["matrix_id"])
    return m

meta = load_meta(META_IN)

# ---------- fallback: regex family mapper ----------
FAMILY_PATTERNS = [
    ("WRKY",   r"\bWRKY\b|\bWRKY\d+"),
    ("bZIP",   r"\bbZIP\b|\bbZIP\d+"),
    ("ERF",    r"\bERF\b|\bAP2\b|\bDREB\b"),
    ("DOF",    r"\bDOF\b|\bDOF\d+"),
    ("NAC",    r"\bNAC\b|\bANAC?\d*"),
    ("MYB",    r"\bMYB\b|\bMYB\d+"),
    ("bHLH",   r"\bbHLH\b|\bHLH\b"),
    ("GATA",   r"\bGATA\b"),
    ("TCP",    r"\bTCP\b"),
    ("HD-ZIP", r"HD[-\s]?ZIP"),
    ("MADS",   r"\bMADS\b|\bAGL\b|\bAG\b"),
    ("HSF",    r"\bHSF\b|\bHsf\b"),
    ("ARF",    r"\bARF\b"),
    ("BES1/BZR", r"\bBES1\b|\bBZR\b"),
]

def regex_family(name):
    if not isinstance(name, str):
        return "Other"
    for fam, pat in FAMILY_PATTERNS:
        if re.search(pat, name, flags=re.IGNORECASE):
            return fam
    return "Other"

def normalize_family(val: str) -> str:
    if pd.isna(val): return "Other"
    s = str(val).strip()
    try:
        if s.startswith('['):
            lst = [x for x in ast.literal_eval(s) if str(x).strip()]
            return lst[0] if lst else "Other"
    except Exception:
        pass
    return s if s and s != "[]" else "Other"

# ---------- attach meta families ----------
def enrich(df):
    out = df.copy()
    # Ensure matrix_id text form
    if "matrix_id" in out.columns:
        out["matrix_id"] = out["matrix_id"].astype(str).str.extract(r'(MA\d{4}\.\d)')[0].fillna(out["matrix_id"].astype(str))
    else:
        out["matrix_id"] = None

    if meta is not None and "matrix_id" in out.columns and "matrix_id" in meta.columns:
        out = out.merge(meta, on="matrix_id", how="left")
        # prefer metadata family/class
        out["tf_family"] = out["meta_family"]
        out["tf_class"]  = out["meta_class"]
        # fallback to regex on tf_name if missing
        miss = out["tf_family"].isna() | (out["tf_family"].astype(str).str.strip()=="")
        if "tf_name" in out.columns:
            out.loc[miss, "tf_family"] = out.loc[miss, "tf_name"].apply(regex_family)
        out["tf_class"] = out["tf_class"].fillna("NA")
    else:
        # no metadata; regex only
        if "tf_name" in out.columns:
            out["tf_family"] = out["tf_name"].apply(regex_family)
        else:
            out["tf_family"] = "Other"
        out["tf_class"] = "NA"

    # tidy columns
    order_cols = [c for c in [
        "gene","snp","chr","pos","ref","alt",
        "matrix_id","tf_name","tf_family","tf_class",
        "ref_llr","alt_llr","delta_llr","abs_delta_llr"
    ] if c in out.columns] + [c for c in out.columns if c not in {
        "gene","snp","chr","pos","ref","alt",
        "matrix_id","tf_name","tf_family","tf_class",
        "ref_llr","alt_llr","delta_llr","abs_delta_llr"}]
    return out[order_cols]

all_annot = enrich(all_scores)
top_annot = enrich(top_scores)

for df in (top_annot, all_annot):
    if "tf_family" in df.columns:
        df["tf_family"] = df["tf_family"].apply(normalize_family)

# ---------- save ----------
all_annot.to_csv(ALL_OUT, index=False)
top_annot.to_csv(TOP_OUT, index=False)

# ---------- report ----------
def summarize(df, title):
    print("\n" + title)
    fam = df["tf_family"].value_counts(dropna=False).rename_axis("family").reset_index(name="n")
    print(fam.to_string(index=False))

summarize(top_annot, "Top-per-SNP families")
summarize(all_annot, "All-hits families (top 20)")
print("\nSaved:")
print(f"  {ALL_OUT}")
print(f"  {TOP_OUT}")


# --- Enrichment of TF families in top-per-SNP vs all-hits --------------------
print("\n" + "="*60)
print("RUNNING TF FAMILY ENRICHMENT (TOP-PER-SNP VS. ALL-HITS)")
print("="*60)

# Use dataframes already in memory
top = top_annot
allhits = all_annot

# Optionally exclude 'Other' from tests (kept in totals)
TEST_FAMS = (
    top["tf_family"]
    .value_counts()
    .loc[lambda s: (s.index != "Other") & (s.values >= 1)]
    .index.tolist()
)

# Totals
n_top = len(top)
n_all = len(allhits)

# To avoid double-counting, remove top rows from the all-hits background if they are present
# (small effect, but precise)
bg = allhits.copy()
if {"snp", "matrix_id"}.issubset(top.columns) and {"snp", "matrix_id"}.issubset(bg.columns):
    keycols = ["snp", "matrix_id"]
    bg = bg.merge(top[keycols].assign(_in_top=1), on=keycols, how="left")
    bg = bg[bg["_in_top"].isna()].drop(columns=["_in_top"])
n_bg = len(bg)

records = []
for fam in TEST_FAMS:
    k_top = int((top["tf_family"] == fam).sum())
    k_bg = int((bg["tf_family"] == fam).sum())

    # Build 2x2: [ [k_top, n_top - k_top], [k_bg, n_bg - k_bg] ]
    a = k_top
    b = n_top - k_top
    c = k_bg
    d = n_bg - k_bg

    # Fisher exact (one-sided, enrichment in TOP set)
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

    # Haldane-Anscombe correction for OR
    or_hat = ((a + 0.5) / (b + 0.5)) / ((c + 0.5) / (d + 0.5))

    records.append({
        "tf_family": fam,
        "k_top": a,
        "n_top": n_top,
        "k_bg": c,
        "n_bg": n_bg,
        "odds_ratio": or_hat,
        "p_one_sided": p
    })

if records:
    enr = pd.DataFrame.from_records(records).sort_values("p_one_sided")
    if len(enr) > 0:
        enr["q_bh"] = multipletests(enr["p_one_sided"], method="fdr_bh")[1]
        # Nice, compact text for the paper
        enr["statement"] = (
            enr["tf_family"] + ": OR=" +
            enr["odds_ratio"].round(2).astype(str) +
            ", p=" + enr["p_one_sided"].apply(lambda x: f"{x:.3g}") +
            ", q=" + enr["q_bh"].apply(lambda x: f"{x:.3g}")
        )
        enr.to_csv(ENR_OUT, index=False)
        print("\nFamily enrichment (top-per-SNP vs all-hits):")
        print(enr[["tf_family","k_top","n_top","k_bg","n_bg","odds_ratio","p_one_sided","q_bh"]].head(10).to_string(index=False))
        print(f"\nSaved enrichment results to: {ENR_OUT}")
else:
    print("\nNo families to test for enrichment (after filtering).")
