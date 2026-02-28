#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Maize GWAS/eQTL Inputs (G, T, E, P)
-----------------------------------------

This script audits and summarizes input files for Maize GWAS/eQTL analyses,
producing a markdown report and a sample coverage matrix.

It inspects the following file types:
- Genotypes: VCF format (e.g., zea_mays_miss0.6_maf0.05.recode.vcf.gz)
- Transcriptomes: FPKM matrices (e.g., WW_209-Uniq_FPKM.txt.gz)
- Metadata: CRA manifest for sample/cultivar mapping (e.g., CRA002002.xlsx)
- Phenotypes: Supplemental data from related studies (e.g., 13059_2021_2481_MOESM2_ESM.xlsx)

The script is designed to run on Windows but paths can be overridden via
command-line arguments.

Usage:
  python audit_gwas_maize.py \\
    --vcf "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\zea_mays_miss0.6_maf0.05.recode.vcf.gz" \\
    --ww  "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\WW_209-Uniq_FPKM.txt.gz" \\
    --ws1 "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\WS1_208-uniq_FPKM.txt.gz" \\
    --ws2 "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\WS2_210-uniq_FPKM.txt.gz" \\
    --cra "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\CRA002002.xlsx" \\
    --supp2021 "C:\\Users\\ms\\Desktop\\gwas\\data\\maize\\13059_2021_2481_MOESM2_ESM.xlsx" \\
    --outdir "C:\\Users\\ms\\Desktop\\gwas\\output\\inspect"
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# For type hinting optional dependencies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

# Optional heavy dependencies are imported conditionally.
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, we provide a dummy function.
    tqdm = None


# --------------------------- Utilities ---------------------------

def ensure_dir(p: Path) -> None:
    """Ensures a directory exists."""
    p.mkdir(parents=True, exist_ok=True)

def file_info(p: Path) -> Dict[str, str]:
    """Returns a dictionary with file information."""
    if not p.exists():
        return {"exists": "N", "path": str(p)}
    size = p.stat().st_size
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
    return {"exists": "Y", "path": str(p), "size_bytes": str(size), "mtime": mtime, "gz": "Y" if p.suffix == ".gz" else "N"}

def canonicalize_name(x: str) -> str:
    """Strips, replaces separators with underscore, removes parentheses, and uppercases."""
    if not isinstance(x, str):
        return ""
    s = x.strip()
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s.upper()

def detect_delim(header_line: str) -> Optional[str]:
    """Detects delimiter (tab, comma) in a header line."""
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None  # Fallback for splitting on any whitespace

def preview_list(lst: List[str], n: int = 8) -> str:
    """Returns a truncated, comma-separated string representation of a list."""
    if not lst:
        return ""
    if len(lst) <= n:
        return ", ".join(lst)
    return f"{', '.join(lst[:n])}, … (+{len(lst) - n} more)"

def write_md(path: Path, content: str) -> None:
    """Writes string content to a file."""
    with path.open("w", encoding="utf-8") as f:
        f.write(content)

# --------------------------- VCF Parsing ---------------------------

def parse_vcf_header(vcf_path: Path) -> Dict:
    """
    Parses the header of a VCF file to extract sample names, contigs, and INFO/FORMAT tags.
    Returns a dictionary with the parsed information.
    """
    samples: List[str] = []
    contigs: List[str] = []
    info_ids: Set[str] = set()
    format_ids: Set[str] = set()
    gt_present = False

    if not vcf_path.exists():
        return {"exists": False}

    opener = gzip.open if vcf_path.suffix == ".gz" else open
    with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("##"):
                if line.startswith("#CHROM"):
                    parts = line.rstrip("\n").split("\t")
                    samples = parts[9:]
                break  # Header parsing is done.

            if line.startswith("##contig="):
                m = re.search(r"ID=([^,>\s]+)", line)
                if m: contigs.append(m.group(1))
            elif line.startswith("##INFO="):
                m = re.search(r"ID=([^,>\s]+)", line)
                if m: info_ids.add(m.group(1))
            elif line.startswith("##FORMAT="):
                m = re.search(r"ID=([^,>\s]+)", line)
                if m:
                    fmt = m.group(1)
                    format_ids.add(fmt)
                    if fmt == "GT":
                        gt_present = True
    return {
        "exists": True,
        "n_samples": len(samples),
        "samples": samples,
        "contigs": contigs,
        "info_ids": sorted(info_ids),
        "format_ids": sorted(format_ids),
        "gt_present": gt_present,
    }

# --------------------------- FPKM Parsing ---------------------------

def parse_fpkm_matrix(path: Path) -> Dict:
    """
    Parses a FPKM matrix file to extract sample/accession IDs and gene IDs.
    Returns a dictionary with parsed information.
    """
    if not path.exists():
        return {"exists": False}

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().rstrip("\n")
        delim = detect_delim(header_line)
        if delim is None:
            cols = re.split(r"\s+", header_line.strip())
        else:
            cols = header_line.split(delim)

        # First column is the row/sample/accession identifier
        row_id_col = cols[0] if cols else "SAMPLE"
        gene_cols = cols[1:]
        n_genes = len(gene_cols)
        gene_ids: Set[str] = set(gene_cols)

        # Collect sample/accession IDs from the first field of each subsequent row
        n_samples = 0
        sample_rows: List[str] = []
        for line in fh:
            if not line.strip():
                continue
            n_samples += 1
            if delim is None:
                rid = re.split(r"\s+", line.strip(), maxsplit=1)[0]
            else:
                rid = line.split(delim, 1)[0]
            sample_rows.append(rid)

    return {
        "exists": True,
        "delim": "\\t" if delim == "\t" else ("," if delim == "," else "whitespace"),
        "row_id_col": row_id_col,
        "n_samples": n_samples,
        "sample_rows": sample_rows,
        "n_genes": n_genes,
        "gene_ids": gene_ids,
    }


# --------------------------- Excel Extraction ---------------------------

def require_pandas():
    if pd is None:
        sys.stderr.write(
            "[ERROR] pandas is required to read Excel files. Please install it, e.g., 'pip install pandas openpyxl'\n"
        )
        sys.exit(2)

def find_sheet_with_headers(
    xlsx_path: Path, required_cols: List[str], case_insensitive: bool = True
) -> Tuple[str, int, List[str]]:
    """
    Scans all sheets and first ~50 rows to find a header row containing all required columns.
    Returns (sheet_name, header_row_index, resolved_colnames).
    """
    require_pandas()
    xl = pd.ExcelFile(xlsx_path)
    req = [c.lower() for c in required_cols] if case_insensitive else required_cols

    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None, nrows=60)
        # search each row for a potential header
        for i in range(len(df)):
            row = df.iloc[i].astype(str).tolist()
            # build normalized names
            norm = [c.strip().lower() for c in row]
            # drop NaN-like
            norm = [c for c in norm if c and c != "nan"]
            if not norm:
                continue
            ok = all(any(rc in c for c in norm) for rc in req)
            if ok:
                # read again with this as header
                df2 = xl.parse(sheet, header=i)
                cols = [str(c) for c in df2.columns]
                return sheet, i, cols
    raise RuntimeError(f"Could not find required cols {required_cols} in {xlsx_path.name}")

def read_cra_manifest(cra_xlsx: Path) -> "pd.DataFrame":
    """
    Reads the CRA manifest file and returns a tidy table with standardized column names.
    """
    require_pandas()
    # Find the sheet that has Sample name + Cultivar (handles prefaces)
    sheet, hdr, _ = find_sheet_with_headers(cra_xlsx, ["Sample name", "Cultivar"])
    df = pd.read_excel(cra_xlsx, sheet_name=sheet, header=hdr)

    # Normalize column names (case/spacing-insensitive)
    rename_map = {
        "sample name": "sample_name",
        "cultivar": "cultivar",
        "biosample accession": "biosample",
        "biosample id": "biosample",
        "biosample": "biosample",
        "library layout": "layout",
        "layout": "layout",
        "read length": "read_len",
        "avg read length": "read_len",
        "readlength": "read_len",
    }
    df = df.rename(columns=lambda c: rename_map.get(str(c).strip().lower(), c))

    # Required minimal fields
    if "sample_name" not in df.columns or "cultivar" not in df.columns:
        raise RuntimeError("CRA manifest must contain 'Sample name' and 'Cultivar' columns.")

    # Parse treatment from the sample name (WW/WS1/WS2)
    def parse_treat(x: str):
        if not isinstance(x, str): return None
        m = re.search(r"(WW|WS1|WS2)\\b", x, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    df["treatment"] = df["sample_name"].astype(str).apply(parse_treat)
    df["accession_name"] = df["cultivar"].astype(str).map(canonicalize_name)
    df["sample_name_clean"] = df["sample_name"].astype(str).map(canonicalize_name)

    # Ensure optional columns exist (fill with NA if absent)
    for opt in ("biosample", "layout", "read_len"):
        if opt not in df.columns:
            df[opt] = pd.NA

    return df[[
        "sample_name", "sample_name_clean", "accession_name", "treatment",
        "biosample", "layout", "read_len"
    ]]

def read_2021_phenotypes(supp_xlsx: Path) -> "pd.DataFrame":
    """
    Extracts Table S1-like content: accession Name, Survival rate (%).
    Returns a tidy table with standardized column names.
    """
    require_pandas()
    sheet, hdr, _ = find_sheet_with_headers(supp_xlsx, ["Name", "Survival rate"])
    df = pd.read_excel(supp_xlsx, sheet_name=sheet, header=hdr)

    # normalize columns
    rename_map = {
        "name": "Name",
        "rna-seq": "RNA-seq",
        "rnaseq": "RNA-seq",
        "rna seq": "RNA-seq",
    }
    # Find survival rate column more robustly
    survival_col = next((c for c in df.columns if "survival" in str(c).lower()), None)
    if survival_col:
        rename_map[survival_col] = "Survival rate (%)"

    df = df.rename(columns=lambda c: rename_map.get(str(c).strip().lower(), c))

    if "Name" not in df.columns:
        raise RuntimeError("Phenotype sheet lacks 'Name' column.")

    df = df.dropna(subset=["Name"]).copy()
    df["accession_name"] = df["Name"].astype(str).map(canonicalize_name)

    # Flags
    df["has_P"] = df.get("Survival rate (%)").notna()
    # Table S1 defines the metabolomics cohort in 2021
    df["has_M"] = True

    keep_cols = ["accession_name", "has_P", "has_M"]
    if "Survival rate (%)" in df.columns:
        keep_cols.insert(1, "Survival rate (%)")
    if "RNA-seq" in df.columns:
        keep_cols.append("RNA-seq")

    return df[keep_cols]


def read_2021_met_feature_sets(supp_xlsx: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns three sets of metabolite IDs: (all_features, WW_features, DS_features)
    - 'all_features' from Table S2 (metabolite metadata).
    - 'WW_features' and 'DS_features' from Table S3 (WW vs DS summary), if detectable.
      If WW/DS columns cannot be confidently identified, both sets fall back to the S3 ID set.
    """
    require_pandas()
    xl = pd.ExcelFile(supp_xlsx)

    def find_header_row(df: "pd.DataFrame") -> Optional[int]:
        # choose the first row with >=5 non-empty cells as the header
        for i in range(min(60, len(df))):
            row = df.iloc[i].astype(str).tolist()
            if sum(1 for c in row if c and c != "nan") >= 5:
                return i
        return None

    # ---------- Table S2: all features ----------
    all_set: Set[str] = set()
    s2_found = False
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, nrows=60)
        hdr = find_header_row(raw)
        if hdr is None:
            continue
        df = xl.parse(sheet, header=hdr)
        cols_lc = [str(c).strip().lower() for c in df.columns]

        # look for hallmark S2 columns
        s2_tokens = ("m/z", "mz", "retention", "rt", "vip", "log2fc", "fdr", "class")
        if sum(any(tok in c for tok in s2_tokens) for c in cols_lc) >= 2:
            # choose an ID column
            id_candidates = ["feature id", "feature_id", "id", "peak id", "peak_id", "metabolite", "name", "compound"]
            id_col = None
            for cand in id_candidates:
                for c in df.columns:
                    if cand == str(c).strip().lower():
                        id_col = c
                        break
                if id_col is not None:
                    break
            if id_col is None:
                id_col = df.columns[0]  # fallback

            ids = df[id_col].astype(str).str.strip()
            all_set = {i for i in ids if i and i.lower() != "nan"}
            s2_found = True
            break  # take the first plausible S2

    # ---------- Table S3: WW vs DS ----------
    ww_set: Set[str] = set()
    ds_set: Set[str] = set()
    s3_found = False
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, nrows=60)
        hdr = find_header_row(raw)
        if hdr is None:
            continue
        df = xl.parse(sheet, header=hdr)
        cols_lc = [str(c).strip().lower() for c in df.columns]

        # look for S3-like table with both WW and DS tokens
        has_ww = any(("ww" in c or "well-watered" in c) for c in cols_lc)
        has_ds = any(("ds" in c or "drought" in c) for c in cols_lc)
        if has_ww and has_ds:
            # pick an ID column again
            id_candidates = ["feature id", "feature_id", "id", "peak id", "peak_id", "metabolite", "name", "compound"]
            id_col = None
            for cand in id_candidates:
                for c in df.columns:
                    if cand == str(c).strip().lower():
                        id_col = c
                        break
                if id_col is not None:
                    break
            if id_col is None:
                id_col = df.columns[0]

            # columns that imply WW / DS presence
            ww_cols = [c for c in df.columns if "ww" in str(c).lower() or "well-watered" in str(c).lower()]
            ds_cols = [c for c in df.columns if str(c).lower().startswith("ds") or "drought" in str(c).lower()]

            def present_rows(subcols: List[str]) -> Set[str]:
                if not subcols:
                    return set()
                mask = df[subcols].notna().any(axis=1)
                return set(df.loc[mask, id_col].astype(str).str.strip())

            ww_set = present_rows(ww_cols)
            ds_set = present_rows(ds_cols)

            # fallback if we couldn't detect per-condition presence
            if not ww_set and not ds_set:
                ids = df[id_col].astype(str).str.strip()
                ww_set = {i for i in ids if i and i.lower() != "nan"}
                ds_set = ww_set.copy()
            s3_found = True
            break

    # Final fallbacks
    if not s2_found and s3_found:
        all_set = ww_set | ds_set
    if not s3_found:
        # cannot separate WW/DS; treat both as the 'all' set (if available)
        ww_set = all_set.copy()
        ds_set = all_set.copy()

    return all_set, ww_set, ds_set


# --------------------------- Coverage Matrix ---------------------------

def build_coverage_matrix(
    vcf_samples: List[str],
    fpkm_samples_by_t: Dict[str, List[str]],
    cra_crosswalk: Optional["pd.DataFrame"],
    phenos: Optional["pd.DataFrame"],
) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    """
    Builds a coverage matrix and identifies mismatches.
    Returns:
      - coverage rows (list of dicts)
      - mismatches dict with keys: unmatched_fpkm, unmatched_vcf, accessions_without_T, accessions_without_P
    """
    # Canonical sets
    vcf_acc = {canonicalize_name(s) for s in vcf_samples}

    # Map FPKM sample names -> accession via CRA if available; else infer
    def infer_accession_from_sample(s: str) -> str:
        s_clean = canonicalize_name(s)
        # heuristic: accession before the first underscore
        return s_clean.split("_")[0]

    fpkm_accessions_by_t: Dict[str, Set[str]] = {"WW": set(), "WS1": set(), "WS2": set()}
    unmatched_fpkm: List[str] = []

    # Build CRA lookup if provided
    cra_lookup = {}
    if cra_crosswalk is not None and not cra_crosswalk.empty:
        for _, r in cra_crosswalk.iterrows():
            cra_lookup[str(r["sample_name_clean"])] = (str(r["accession_name"]), str(r["treatment"]) if pd.notna(r["treatment"]) else None)

    for t, samples in fpkm_samples_by_t.items():
        for s in samples:
            s_clean = canonicalize_name(s)
            acc = None
            # try CRA
            if s_clean in cra_lookup:
                acc, t_cra = cra_lookup[s_clean]
                # sanity: make sure treatment matches if both present
                if t_cra and t and t_cra != t:
                    # keep anyway, but note mismatch
                    pass
            else:
                # fall back to inference
                acc = infer_accession_from_sample(s)
            if acc:
                fpkm_accessions_by_t[t].add(acc)
            else:
                unmatched_fpkm.append(s)

    # Phenotype & metabolite cohorts
    phenotype_acc = set()
    metabolome_acc = set()
    if phenos is not None and not phenos.empty:
        if "has_P" in phenos.columns:
            phenotype_acc = set(phenos.loc[phenos["has_P"] == True, "accession_name"].astype(str))
        else:
            phenotype_acc = set(phenos["accession_name"].astype(str))
        if "has_M" in phenos.columns:
            metabolome_acc = set(phenos.loc[phenos["has_M"] == True, "accession_name"].astype(str))
        else:
            metabolome_acc = set()  # absent → no M

    # Universe
    universe = set()
    universe |= vcf_acc
    universe |= metabolome_acc
    for t in ("WW", "WS1", "WS2"):
        universe |= fpkm_accessions_by_t[t]
    universe |= phenotype_acc

    # Coverage rows (+ has_M)
    rows: List[Dict[str, str]] = []
    for acc in sorted(universe):
        row = {
            "accession": acc,
            "has_G": "Y" if acc in vcf_acc else "N",
            "has_T_WW": "Y" if acc in fpkm_accessions_by_t["WW"] else "N",
            "has_T_WS1": "Y" if acc in fpkm_accessions_by_t["WS1"] else "N",
            "has_T_WS2": "Y" if acc in fpkm_accessions_by_t["WS2"] else "N",
            "has_P": "Y" if acc in phenotype_acc else "N",
            "has_M": "Y" if acc in metabolome_acc else "N",   # NEW
        }
        rows.append(row)

    # Mismatch summaries (unchanged; optional to extend for M)
    accessions_with_any_T = fpkm_accessions_by_t["WW"] | fpkm_accessions_by_t["WS1"] | fpkm_accessions_by_t["WS2"]
    accessions_without_T = sorted((phenotype_acc | vcf_acc) - accessions_with_any_T)
    accessions_without_P = sorted(accessions_with_any_T - phenotype_acc)
    unmatched_vcf = sorted([s for s in vcf_samples if canonicalize_name(s) not in universe])

    mismatches = {
        "unmatched_fpkm": unmatched_fpkm,
        "unmatched_vcf": unmatched_vcf,
        "accessions_without_T": accessions_without_T,
        "accessions_without_P": accessions_without_P,
    }
    return rows, mismatches

# --------------------------- Report Writer ---------------------------

def write_coverage_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    """Writes the coverage matrix to a CSV file."""
    if not rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            # Write header even for empty file
            header = "accession,has_G,has_T_WW,has_T_WS1,has_T_WS2,has_P,has_M\n"
            f.write(header)
        return
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def md_table_from_kv_list(kv: List[Tuple[str, str]]) -> str:
    """Renders a tiny 2-column Markdown table."""
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in kv:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

def make_report(
    out_md: Path,
    vcf_meta: Dict,
    ww_meta: Dict,
    ws1_meta: Dict,
    ws2_meta: Dict,
    cra_info: Dict[str, str],
    supp_info: Dict[str, str],
    coverage_rows: List[Dict[str, str]],
    mismatches: Dict[str, List[str]],
    met_counts: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    def section(title: str) -> str:
        return f"\n\n## {title}\n"

    c_total = sum(1 for r in coverage_rows if r["has_G"]=="Y" and r["has_P"]=="Y" and (r["has_T_WW"]=="Y" or r["has_T_WS1"]=="Y" or r["has_T_WS2"]=="Y"))
    c_2treat = sum(1 for r in coverage_rows if sum([r["has_T_WW"]=="Y", r["has_T_WS1"]=="Y", r["has_T_WS2"]=="Y"]) >= 2)

    content = ["# Maize Data Audit (G/T/E/P)\n"]

    # Inventory
    content.append(section("Inventory"))
    inv_lines = []
    for label, info in [("VCF", vcf_meta), ("WW FPKM", ww_meta), ("WS1 FPKM", ws1_meta), ("WS2 FPKM", ws2_meta)]:
        exists = "Y" if info.get("exists") else "N"
        path = info.get("path", "-")
        size = info.get("size_bytes", "-")
        mtime = info.get("mtime", "-")
        inv_lines.append((label, f"exists={exists}; size={size}; mtime={mtime}; path={path}"))
    inv_lines.append(("CRA002002.xlsx", f"exists={cra_info.get('exists','N')}; path={cra_info.get('path','-')}"))
    inv_lines.append(("2021 Supplement", f"exists={supp_info.get('exists','N')}; path={supp_info.get('path','-')}"))
    content.append(md_table_from_kv_list(inv_lines))

    # VCF
    content.append(section("VCF (G)"))
    if vcf_meta.get("exists"):
        content.append(f"- **Samples (n={vcf_meta['n_samples']}):** {preview_list(vcf_meta['samples'], 12)}")
        content.append(f"- **Contigs (n={len(vcf_meta['contigs'])}):** {preview_list(vcf_meta['contigs'], 12)}")
        content.append(f"- **FORMAT tags:** {preview_list(vcf_meta['format_ids'], 12)}; GT present={vcf_meta['gt_present']}")
        content.append(f"- **INFO tags:** {preview_list(vcf_meta['info_ids'], 12)}")
    else:
        content.append("_VCF not found._")

    # FPKM
    content.append(section("Transcriptomes (T)"))
    for label, meta in [("WW", ww_meta), ("WS1", ws1_meta), ("WS2", ws2_meta)]:
        if meta.get("exists"):
            content.append(
                f"- **{label}:** {meta['n_genes']:,} genes, {meta['n_samples']} samples, "
                f"row_id_col='{meta.get('row_id_col', '(first)')}', delimiter={meta['delim']}"
            )
            content.append(f"  - Sample rows (accessions): {preview_list(list(meta.get('sample_rows', [])), 10)}")
        else:
            content.append(f"- **{label}:** _file not found_")

    # Gene-overlap table
    g_ww  = ww_meta.get("gene_ids") or set()
    g_ws1 = ws1_meta.get("gene_ids") or set()
    g_ws2 = ws2_meta.get("gene_ids") or set()
    g_union = g_ww | g_ws1 | g_ws2
    g_inter = g_ww & g_ws1 & g_ws2

    content.append("\n**Gene Set Overlap (T)**")
    content.append(
        md_table_from_kv_list([
            ("Union (WW ∪ WS1 ∪ WS2)", f"{len(g_union):,}"),
            ("Intersection (WW ∩ WS1 ∩ WS2)", f"{len(g_inter):,}"),
        ])
    )

    # Metabolite overlap
    mt_total, mt_union, mt_inter = met_counts
    if mt_total or mt_union or mt_inter:
        content.append("\n**Metabolite Feature Overlap (M)**")
        content.append(
            md_table_from_kv_list([
                ("Total features (Table S2)", f"{mt_total:,}"),
                ("Union (WW ∪ DS)", f"{mt_union:,}"),
                ("Intersection (WW ∩ DS)", f"{mt_inter:,}"),
            ])
        )

    # Coverage matrix summary
    content.append(section("Coverage Matrix & Gates"))
    n_rows = len(coverage_rows)
    content.append(f"- **Accessions in matrix:** {n_rows}")
    content.append(f"- **Primary Gate (G ∩ P ∩ any T):** {c_total} accessions")
    content.append(f"- **Multi-condition T (≥2 treatments):** {c_2treat} accessions")

    # metabolomics summary
    c_gptm = sum(1 for r in coverage_rows if r.get("has_M") == "Y" and r["has_G"]=="Y" and r["has_P"]=="Y" and (r["has_T_WW"]=="Y" or r["has_T_WS1"]=="Y" or r["has_T_WS2"]=="Y"))
    n_met_cohort = sum(1 for r in coverage_rows if r.get("has_M") == "Y")
    content.append(f"- **Metabolomics Cohort (M):** {n_met_cohort} accessions")
    content.append(f"- **Quad-overlap (G ∩ P ∩ any T ∩ M):** {c_gptm} accessions")
    content.append("_Note: The metabolomics study (M) contrasts WW vs DS, which may not directly correspond to the WS1/WS2 transcriptomic conditions._")

    # Mismatches
    content.append(section("ID Coherence & Mismatches"))
    for key, title in [
        ("unmatched_fpkm", "FPKM samples without CRA match/inference"),
        ("unmatched_vcf", "VCF sample names not aligned to accessions"),
        ("accessions_without_T", "Accessions with G or P but missing any T"),
        ("accessions_without_P", "Accessions with T but missing P"),
    ]:
        vals = mismatches.get(key, [])
        content.append(f"- **{title} (n={len(vals)}):** {preview_list(vals, 15) if vals else '—'}")

    write_md(out_md, "\n".join(content))

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit maize GWAS/eQTL inputs (G,T,E,P).")
    parser.add_argument("--vcf", default=r"C:\Users\ms\Desktop\gwas\data\maize\zea_mays_miss0.6_maf0.05.recode.vcf.gz")
    parser.add_argument("--ww",  default=r"C:\Users\ms\Desktop\gwas\data\maize\WW_209-Uniq_FPKM.txt.gz")
    parser.add_argument("--ws1", default=r"C:\Users\ms\Desktop\gwas\data\maize\WS1_208-uniq_FPKM.txt.gz")
    parser.add_argument("--ws2", default=r"C:\Users\ms\Desktop\gwas\data\maize\WS2_210-uniq_FPKM.txt.gz")
    parser.add_argument("--cra", default=r"C:\Users\ms\Desktop\gwas\data\maize\CRA002002.xlsx")
    parser.add_argument("--supp2021", default=r"C:\Users\ms\Desktop\gwas\data\maize\13059_2021_2481_MOESM2_ESM.xlsx")
    parser.add_argument("--outdir", default=r"C:\Users\ms\Desktop\gwas\output\inspect")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use for parallel processing.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # File infos
    vcf_path = Path(args.vcf)
    ww_path, ws1_path, ws2_path = Path(args.ww), Path(args.ws1), Path(args.ws2)
    cra_path, supp_path = Path(args.cra), Path(args.supp2021)

    vcf_info_d = file_info(vcf_path)
    ww_info_d, ws1_info_d, ws2_info_d = file_info(ww_path), file_info(ws1_path), file_info(ws2_path)
    cra_info = {"exists": "Y" if cra_path.exists() else "N", "path": str(cra_path)}
    supp_info = {"exists": "Y" if supp_path.exists() else "N", "path": str(supp_path)}

    print(f"Starting parallel data parsing on {args.cores} cores...")

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = {}
        # Core tasks
        futures[executor.submit(parse_vcf_header, vcf_path)] = "vcf_meta"
        futures[executor.submit(parse_fpkm_matrix, ww_path)] = "ww_meta"
        futures[executor.submit(parse_fpkm_matrix, ws1_path)] = "ws1_meta"
        futures[executor.submit(parse_fpkm_matrix, ws2_path)] = "ws2_meta"

        # Optional tasks
        if cra_path.exists() and pd:
            futures[executor.submit(read_cra_manifest, cra_path)] = "cra_df"
        elif cra_path.exists():
            sys.stderr.write("[WARN] pandas not available; skipping CRA manifest join.\n")

        if supp_path.exists() and pd:
            futures[executor.submit(read_2021_phenotypes, supp_path)] = "phenos_df"
            futures[executor.submit(read_2021_met_feature_sets, supp_path)] = "met_sets"
        elif supp_path.exists():
            sys.stderr.write("[WARN] pandas not available; skipping phenotype/metabolite extraction.\n")

        results = {}
        tasks_iterator = as_completed(futures)
        if tqdm:
            tasks_iterator = tqdm(tasks_iterator, total=len(futures), desc="Parsing files")
        else:
            print(f"Processing {len(futures)} files...")

        for future in tasks_iterator:
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                sys.stderr.write(f"[WARN] Task '{task_name}' failed: {e}\n")
                results[task_name] = None

    # Unpack results with sensible defaults
    vcf_meta = results.get("vcf_meta") or {"exists": False}
    ww_meta = results.get("ww_meta") or {"exists": False}
    ws1_meta = results.get("ws1_meta") or {"exists": False}
    ws2_meta = results.get("ws2_meta") or {"exists": False}
    cra_df = results.get("cra_df")
    phenos_df = results.get("phenos_df")

    met_total, met_union, met_inter = 0, 0, 0
    met_sets_result = results.get("met_sets")
    if met_sets_result:
        m_all, m_WW, m_DS = met_sets_result
        met_total = len(m_all)
        met_union = len(m_WW | m_DS)
        met_inter = len(m_WW & m_DS)


    # FPKM samples by treatment (use row accessions)
    fpkm_samples_by_t = {
        "WW": ww_meta.get("sample_rows", []),
        "WS1": ws1_meta.get("sample_rows", []),
        "WS2": ws2_meta.get("sample_rows", []),
    }

    # Build coverage matrix
    vcf_samples = vcf_meta.get("samples", [])
    rows, mismatches = build_coverage_matrix(vcf_samples, fpkm_samples_by_t, cra_df, phenos_df)

    # Write outputs
    out_csv = outdir / "coverage_matrix.csv"
    write_coverage_csv(rows, out_csv)

    out_md = outdir / "audit_report.md"
    make_report(
        out_md=out_md,
        vcf_meta={**vcf_meta, **vcf_info_d},
        ww_meta={**ww_meta, **ww_info_d},
        ws1_meta={**ws1_meta, **ws1_info_d},
        ws2_meta={**ws2_meta, **ws2_info_d},
        cra_info=cra_info,
        supp_info=supp_info,
        coverage_rows=rows,
        mismatches=mismatches,
        met_counts=(met_total, met_union, met_inter),
    )

    print(f"Successfully wrote audit report to: {out_md}")
    print(f"Successfully wrote coverage matrix to: {out_csv}")


if __name__ == "__main__":
    main()
