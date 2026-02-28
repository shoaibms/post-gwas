# -*- coding: utf-8 -*-
r"""
Build supplementary tables (S1-S7) and data files (DS1-DS3) from analysis outputs.

Loads analysis CSVs, standardises column names via synonym lookup, and writes
publication-ready supplementary tables. Uses week6 ieqtl_delta_results.csv as
the primary data source for Table S5, and ieqtl_snps_overlapping_tf_peaks.csv
for TF overlap. Replaces blank ieqtl_q values with "n.s." in Table S2.
Generates comprehensive table legends and optionally bundles outputs into a
versioned ZIP archive.
"""
import sys, json, argparse, zipfile, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

if len(sys.argv) == 1:
    sys.argv += ["--zip"]

ROOT = Path(r"C:\Users\ms\Desktop\gwas")
OUTDIR = Path(r"C:\Users\ms\Desktop\gwas\data\supplimetary_data")

def expected_paths(root: Path) -> Dict[str, List[Path]]:
    return {
        "cohort_core": [root / "output" / "cohort" / "core_all3_env.csv"],
        "sample_metadata": [root / "data" / "maize" / "process" / "metadata" / "sample_metadata.csv"],
        "platinum_modulators": [root / "output" / "week1_stability" / "platinum_modulator_set.csv"],
        "window_stability": [root / "output" / "week1_stability" / "window_stability_metrics.csv"],
        "ieqtl_gene_fdr": [
            root / "output" / "week6_ieqtl" / "ieqtl_delta_gene_fdr.csv",      # AUTHORITATIVE (18 genes)
            root / "output" / "week6_ieqtl" / "ieqtl_gene_level_fdr.csv",     # Fallback
        ],
        "foreground_annotated": [root / "output" / "week2_enrichment" / "foreground_annotated.csv"],
        "tf_distances": [root / "output" / "week3_tf_binding" / "influential_tf_distances.csv"],
        "background_annotated": [root / "output" / "week2_enrichment" / "background_annotated.csv"],
        "ieqtl_results": [
            root / "output" / "week6_ieqtl" / "ieqtl_delta_results.csv",      # AUTHORITATIVE (consistent with gene_fdr)
            root / "output" / "week6_ieqtl" / "ieqtl_final_results.csv",      # Fallback
            root / "output" / "week5_ieqtl" / "ieqtl_results_complete.csv",   # Legacy fallback
        ],
        "ieqtl_tf_overlap": [root / "output" / "week6_ieqtl" / "ieqtl_snps_overlapping_tf_peaks.csv"],
        "motif_summary": [
            root / "output" / "postgwas" / "motif" / "motif_summary.csv",
            root / "output" / "week6_ieqtl" / "figure_motif_summary.csv",
        ],
        "go_modulators": [root / "output" / "week4_go_enrichment" / "modulators_go_enrichment.csv"],
        "go_drivers": [root / "output" / "week4_go_enrichment" / "drivers_go_enrichment.csv"],
        "labels_500kb": [root / "output" / "ect_alt" / "integrated" / "decouple_labels_500kb.csv"],
        "labels_1Mb": [root / "output" / "ect_alt" / "integrated" / "decouple_labels_1Mb.csv"],
        "labels_2Mb": [root / "output" / "ect_alt" / "integrated" / "decouple_labels_2Mb.csv"],
    }

SYN = {
    "gene": ["gene", "gene_id", "zm_gene", "geneID", "GeneID"],
    "snp": ["snp", "snp_id", "variant", "rsid", "id"],
    "chr": ["chr", "chrom", "chromosome", "Chromosome"],
    "pos": ["pos", "position", "bp", "BP"],
    "ref": ["ref", "ref_allele", "allele_ref"],
    "alt": ["alt", "alt_allele", "allele_alt"],
    "maf": ["maf", "af", "alt_freq", "MAF"],
    "dist_tss": ["distance_to_tss", "tss_dist", "dist_tss", "distance_tss", "dist_to_tss"],
    "vep": ["vep_consequence", "consequence", "impact", "vep"],
    "env": ["environment", "treatment", "condition", "env"],
    "accession": ["accession_id", "accession", "line", "genotype", "sample", "id"],
    "ww": ["WW", "ww", "well_watered", "WellWatered"],
    "ws1": ["WS1", "ws1", "moderate", "Moderate"],
    "ws2": ["WS2", "ws2", "severe", "Severe"],
    "p": ["p", "pvalue", "p_value", "PValue", "p.gof",
          "p_GxE", "p_gxe", "pGxE", "pGXE", "p_interaction", "pInt", "p-int",
          "p_G", "pG"],
    "q": ["q", "qvalue", "q_value", "qval", "FDR", "fdr", "p.adjust", "padj", "adj.P.Val",
          "q_GxE", "q_gxe", "qGxE", "q_interaction", "fdr_gxe", "q_gene", "gene_q",
          "q_G", "qG"],
    "beta": ["beta_gxe", "beta", "Beta", "effect", "beta_GxE", "Beta_GxE",
             "beta_G", "betaG"],
    "beta_lo": ["beta_ci_lower", "beta_low", "beta_lower", "ci_lo", "lcl"],
    "beta_hi": ["beta_ci_upper", "beta_high", "beta_upper", "ci_hi", "ucl"],
    "tf_within_1kb": ["within_1kb_TF", "tf_within_1kb", "within_1kb_tf", "tf_overlap_1kb"],
    "tf_dist": ["distance_to_nearest_TF_peak", "tf_distance", "dist_tf_peak"],
    "label": ["label", "class", "classification"],
    "h_score": ["xgb_h", "xgboost_h", "importance_h"],
    "ect_att": ["ect_attention_ws2", "ect_attention", "attention_ws2"],
    "rkhs_frac_500kb": ["rkhs_frac_500kb", "rkhs_ge_frac_500kb"],
    "rkhs_frac_1Mb": ["rkhs_frac_1Mb", "rkhs_ge_frac_1Mb"],
    "rkhs_frac_2Mb": ["rkhs_frac_2Mb", "rkhs_ge_frac_2Mb"],
    "go_id": ["GO_ID", "GO", "GO.ID", "ID"],
    "go_term": ["term", "Term", "Description", "Name", "GO_term", "GO.Term", "GO_Term_Name"],
    "gene_ratio": ["GeneRatio", "Gene_Ratio", "gene_ratio"],
    "bg_ratio": ["BgRatio", "BackgroundRatio", "bg_ratio"],
    "k": ["k", "Count", "overlap", "Hits"],
    "K": ["K", "Size", "N", "Background"],
}

class BuildError(Exception): pass
LOG: List[str] = []
OUTPUTS: List[Path] = []

def log(msg: str):
    print(msg)
    LOG.append(msg)

def save_report(outdir: Path, ok: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    rep = outdir / "validation_report.json"
    rep.write_text(json.dumps({"success": ok, "messages": LOG,
                               "outputs": [str(p) for p in OUTPUTS]}, indent=2), encoding="utf-8")
    log(f"[REPORT] {rep}")

def find_first_existing(root: Path, cands: List[Path]) -> Optional[Path]:
    for c in cands:
        if c.exists(): return c
    for c in cands:
        hits = list(root.rglob(c.name))
        if hits: return hits[0]
    return None

def load_csv_with_checks(EXPECTED, key: str, required: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Path, Dict[str, str]]:
    path = find_first_existing(ROOT, EXPECTED.get(key, []))
    if path is None:
        raise BuildError(f"[MISSING] '{key}' not found. Looked for:\n  - " + "\n  - ".join(map(str, EXPECTED.get(key, []))) +
                         f"\nCheck REPRODUCE steps or folder under {ROOT}.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise BuildError(f"[READ-ERROR] '{key}' at {path}: {e}")
    hmap = {}
    if required:
        for logical in required:
            found = None
            for cand in SYN.get(logical, [logical]):
                for col in df.columns:
                    if col.lower() == cand.lower():
                        found = col; break
                if found: break
            if not found:
                raise BuildError(f"[HEADER] '{path.name}' (for '{key}') missing '{logical}'. "
                                 f"Acceptable: {SYN.get(logical, [logical])}\nFound: {list(df.columns)}")
            hmap[logical] = found
    log(f"[OK] Loaded '{key}' from {path}")
    return df, path, hmap

def write_csv(outdir: Path, df: pd.DataFrame, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    df.to_csv(path, index=False)
    OUTPUTS.append(path)
    log(f"[WRITE] {path}")
    return path

def pick_col(df: pd.DataFrame, *alts: str) -> Optional[str]:
    lowers = {c.lower(): c for c in df.columns}
    for a in alts:
        if a.lower() in lowers: return lowers[a.lower()]
    return None

def choose_first_present(df: pd.DataFrame, logical: str) -> Optional[str]:
    for cand in SYN.get(logical, []):
        for col in df.columns:
            if col.lower() == cand.lower():
                return col
    return None

def manifest(outdir: Path):
    rows = []
    for p in OUTPUTS:
        try:
            with open(p, encoding="utf-8") as fh:
                n_rows = sum(1 for _ in fh) - 1
            df = pd.read_csv(p, nrows=1)
            rows.append({"file": str(p), "rows": max(0, n_rows),
                         "n_cols": len(df.columns), "columns": ";".join(map(str, df.columns))})
        except Exception as e:
            rows.append({"file": str(p), "rows": "?", "n_cols": "?", "columns": f"(unreadable: {e})"})
    man = pd.DataFrame(rows)
    write_csv(outdir, man, "manifest.csv")

def create_table_legends(outdir: Path):
    """
    Create comprehensive legend file for all supplementary tables.
    Addresses reviewer concerns about column definitions.
    """
    legend_text = """SUPPLEMENTARY TABLE LEGENDS
==============================

TABLE S2: Platinum Modulator Gene Set
--------------------------------------
Summary of the 31 high-confidence platinum modulator genes with validation metrics.

Column Definitions:
- gene_id: Maize gene identifier (AGPv4 annotation; Zm00001d######)
- window_stability_score: Number of cis-windows (+/-500kb, +/-1Mb, +/-2Mb) in which the gene 
  was classified as a network modulator. Values: 2 or 3. Higher scores indicate greater 
  robustness to analytical window size choice. All platinum genes have score >=2.
- ieqtl_q: Gene-level interaction eQTL q-value from delta-model analysis (FDR-corrected 
  minimum p-value per gene). Values <=0.10 are considered significant. "n.s." indicates 
  q > 0.10 (not significant at the chosen threshold).
- snp_id: Representative lead SNP for the gene (format: chr:position or chr_position)
- TF_overlap: Indicates whether the lead SNP overlaps a transcription factor (TF) ChIP-seq 
  peak within 1kb. Values: Y (yes), N (no), or blank if TF data unavailable.

Notes:
- 18 of 31 genes (58%) show significant gene-level ieQTL signals (q <= 0.10)
- Window stability score validates that modulator classification is robust to parameter choices
- TF overlap provides mechanistic support for regulatory function


TABLE S5: Interaction eQTL Results Summary
------------------------------------------
Gene-level and lead variant statistics for interaction eQTL analysis.

Column Definitions:
- gene_id: Maize gene identifier (AGPv4; Zm00001d######)
- gene_level_q: FDR-corrected q-value based on minimum p-value across all tested SNPs 
  for this gene. Threshold: q <= 0.10 for significance.
- n_sig_snps: Number of individual SNPs passing SNP-level FDR threshold (q <= 0.10)
- lead_snp: SNP with minimum p-value for this gene after LD pruning
- lead_q: SNP-level q-value for the lead SNP (may exceed 0.10 even if gene_level_q <= 0.10)
- chr: Chromosome of lead SNP
- pos: Position of lead SNP (bp)
- beta_GxE: Genotype-by-environment interaction effect size
- tf_overlap: Whether lead SNP overlaps TF ChIP-seq peak (Y/N)

Important Notes:
- Gene-level q-values are based on the minimum p-value per gene with FDR correction
- n_sig_snps reflects individual SNPs passing a stricter SNP-level FDR threshold (q <= 0.10)
- Some genes may have significant gene_level_q but n_sig_snps=0 and non-significant lead_q
  This occurs when the gene-level test (minimum p-value) passes FDR but no individual SNP 
  meets the SNP-level FDR threshold - this is statistically valid and reflects different 
  multiple testing burdens at gene vs. SNP levels.


TABLE S1: Analysis Cohort
--------------------------
Sample metadata showing which accessions were phenotyped in each environment.

Columns:
- accession_id: Unique identifier for maize inbred line
- WW: Well-watered condition (1=present, 0=absent)
- WS1: Moderate drought stress (1=present, 0=absent)  
- WS2: Severe drought stress (1=present, 0=absent)


TABLE S3: Window Stability Metrics
-----------------------------------
Pairwise comparison metrics quantifying modulator set stability across cis-window sizes.

Columns:
- pair: Window comparison (e.g., "500kb vs 1Mb")
- jaccard: Jaccard similarity index (overlap/union)
- enrichment: Enrichment ratio (observed overlap / expected overlap by chance)
- spearman: Spearman rank correlation coefficient for gene rankings


TABLE S4: Foreground SNP Annotations
-------------------------------------
Regulatory annotations for influential SNPs linked to platinum modulators.

Key Columns:
- gene_id: Associated gene
- snp_id: SNP identifier
- chr, pos: Genomic coordinates
- vep_consequence: Variant Effect Predictor consequence annotation
- distance_to_tss: Distance to transcription start site (bp)
- tf_within_1kb: Boolean indicator of TF ChIP-seq peak overlap


TABLE S6: Motif Disruption Analysis
------------------------------------
Position weight matrix (PWM) scores for ieQTL SNPs overlapping TF peaks.

Columns:
- snp_id: SNP identifier
- gene_id: Associated gene
- chr, pos: Genomic coordinates
- motif_name: TF motif from JASPAR database
- motif_family: TF family classification
- delta_llr: Change in log-likelihood ratio (reference vs. alternate allele)
- abs_delta_llr: Absolute value of delta_llr
- effect_direction: Predicted direction of binding affinity change


TABLE S7: Gene Ontology Enrichment
-----------------------------------
GO term enrichment results for modulators and drivers.

Columns:
- set: Gene class (modulator or driver)
- GO_ID: Gene Ontology identifier
- term: GO term description
- p: Raw p-value from hypergeometric test
- q: FDR-corrected q-value
- k: Number of genes from query set in this term
- K: Total number of genes in background with this term
- GeneRatio: Proportion of query genes in term (k/n)
- BgRatio: Proportion of background genes in term (K/N)


DATA S1: Complete Gene Classifications
---------------------------------------
Full classification matrix for all 500 genes across three cis-window sizes.

Columns include per-window labels (network_modulator, additive_driver, unclear) and 
model-specific metrics (XGBoost H-statistic, RKHS variance fractions, transformer attention).


DATA S2: Complete ieQTL Results
--------------------------------
Full association statistics for all gene-SNP tests in the interaction eQTL analysis.


DATA S3: Background SNP Set
----------------------------
Distance-matched background SNPs used as controls in enrichment analyses.
"""
    
    legend_file = outdir / "Table_Legends.txt"
    legend_file.write_text(legend_text, encoding="utf-8")
    OUTPUTS.append(legend_file)
    log(f"[LEGEND] {legend_file}")
    return legend_file

def build_S1(EXPECTED, outdir: Path):
    try:
        df, _, h = load_csv_with_checks(EXPECTED, "cohort_core", required=["accession", "ww", "ws1", "ws2"])
        out = df.rename(columns={h["accession"]: "accession_id",
                                 h["ww"]: "WW", h["ws1"]: "WS1", h["ws2"]: "WS2"})[["accession_id","WW","WS1","WS2"]]
        return write_csv(outdir, out, "Table_S1_analysis_cohort.csv")
    except BuildError as e:
        log(f"[S1] Using sample_metadata fallback :: {e}")
        df, _, h = load_csv_with_checks(EXPECTED, "sample_metadata", required=["accession", "env"])
        acc, env = h["accession"], h["env"]
        pivot = (df[[acc, env]].dropna().assign(flag=1)
                 .pivot_table(index=acc, columns=env, values="flag", aggfunc="max", fill_value=0))
        rename_map = {}
        for logical, target in [("ww", "WW"), ("ws1", "WS1"), ("ws2", "WS2")]:
            hit = choose_first_present(pivot.reset_index().rename(columns={"index": acc}), logical)
            if hit and hit in pivot.columns: rename_map[hit] = target
        for t in ["WW","WS1","WS2"]:
            if t not in rename_map.values(): pivot[t] = 0
        out = pivot.reset_index().rename(columns={acc: "accession_id", **rename_map})
        for t in ["WW","WS1","WS2"]:
            if t not in out.columns: out[t] = 0
        out = out[["accession_id","WW","WS1","WS2"]]
        return write_csv(outdir, out, "Table_S1_analysis_cohort.csv")

def compute_window_stability(EXPECTED) -> Optional[pd.DataFrame]:
    try:
        l500, _, h1 = load_csv_with_checks(EXPECTED, "labels_500kb", required=["gene", "label"])
        l1m,  _, h2 = load_csv_with_checks(EXPECTED, "labels_1Mb",  required=["gene", "label"])
        l2m,  _, h3 = load_csv_with_checks(EXPECTED, "labels_2Mb",  required=["gene", "label"])
    except BuildError as e:
        log(f"[S2] Labels not available to compute per-gene stability :: {e}")
        return None
    
    def as_set(df, h):
        tmp = df.rename(columns={h["gene"]:"gene_id", h["label"]:"label"})
        return set(tmp.loc[tmp["label"].astype(str).str.lower()=="network_modulator", "gene_id"])
    
    S1 = as_set(l500, h1)
    S2 = as_set(l1m, h2)
    S3 = as_set(l2m, h3)
    all_genes = sorted(set().union(S1, S2, S3))
    
    stab = pd.DataFrame({"gene_id": all_genes})
    stab["window_stability_score"] = stab["gene_id"].map(
        lambda g: int(g in S1) + int(g in S2) + int(g in S3)
    )
    
    return stab

def build_S2(EXPECTED, outdir: Path):
    """
    Build Table S2 with proper handling of ieqtl_q blanks.
    
    ISSUE 2 FIX:
    - Replaces NaN values in ieqtl_q with "n.s." (not significant; q > 0.10)
    - window_stability_score is documented in Table_Legends.txt
    """
    df_mod, _, h = load_csv_with_checks(EXPECTED, "platinum_modulators", required=["gene"])
    out = df_mod.rename(columns={h["gene"]:"gene_id"})[["gene_id"]]
    
    stab = compute_window_stability(EXPECTED)
    if stab is not None:
        out = out.merge(stab, on="gene_id", how="left")
    else:
        out["window_stability_score"] = np.nan
    
    try:
        df_q, path_q, hq = load_csv_with_checks(EXPECTED, "ieqtl_gene_fdr", required=["gene","q"])
        log(f"[S2] Using ieQTL file: {path_q.name}")
        ieqtl_data = df_q.rename(columns={hq["gene"]:"gene_id", hq["q"]:"ieqtl_q"})[["gene_id","ieqtl_q"]]
        out = out.merge(ieqtl_data, on="gene_id", how="left")
        
        if "ieqtl_q" in out.columns:
            # Replace NaN or q > 0.10 with "n.s."
            original_numeric = out["ieqtl_q"].notna().sum()
            out["ieqtl_q"] = out["ieqtl_q"].apply(
                lambda x: "n.s." if pd.isna(x) or (isinstance(x, (int, float)) and x > 0.10) else x
            )
            n_replaced = out["ieqtl_q"].eq("n.s.").sum()
            log(f"[S2] Marked {n_replaced} genes as 'n.s.' (q > 0.10 or missing)")
    except BuildError as e:
        log(f"[S2] ieQTL gene-level q not added :: {e}")
    
    try:
        fg, _, hf = load_csv_with_checks(EXPECTED, "foreground_annotated", required=["gene","snp"])
        tf, _, ht = load_csv_with_checks(EXPECTED, "tf_distances", required=["snp"])
        lead = fg.rename(columns={hf["gene"]:"gene_id", hf["snp"]:"snp_id"}).groupby("gene_id", as_index=False).first()
        tf = tf.rename(columns={ht["snp"]:"snp_id"})
        flag = choose_first_present(tf, "tf_within_1kb")
        dist = choose_first_present(tf, "tf_dist")
        if flag:
            tf["TF_overlap"] = tf[flag]
        elif dist:
            tf["TF_overlap"] = (tf[dist].astype(float) <= 1000).map({True:"Y", False:"N"})
        else:
            tf["TF_overlap"] = np.nan
        lead = lead.merge(tf[["snp_id","TF_overlap"]], on="snp_id", how="left")
        out = out.merge(lead, on="gene_id", how="left")
    except BuildError as e:
        log(f"[S2] lead SNP / TF overlap not added :: {e}")
    
    cols = [c for c in ["gene_id","window_stability_score","ieqtl_q","snp_id","TF_overlap"] if c in out.columns]
    out = out[cols]
    
    return write_csv(outdir, out, "Table_S2_platinum_modulators.csv")

def build_S3(EXPECTED, outdir: Path):
    try:
        df, _, _ = load_csv_with_checks(EXPECTED, "window_stability", required=None)
        pair = pick_col(df, "pair", "window_pair") or "pair"
        jac  = pick_col(df, "jaccard", "jaccard_index") or "jaccard"
        enr  = pick_col(df, "enrichment", "enrichment_ratio") or "enrichment"
        sp   = pick_col(df, "spearman", "spearman_rho") or "spearman"
        keep = [c for c in [pair, jac, enr, sp] if c in df.columns]
        out = df[keep].copy().rename(columns={pair:"pair", jac:"jaccard", enr:"enrichment", sp:"spearman"})
        return write_csv(outdir, out, "Table_S3_window_stability_summary.csv")
    except BuildError as e:
        log(f"[S3] Using labels fallback :: {e}")
        try:
            l500, _, h1 = load_csv_with_checks(EXPECTED, "labels_500kb", required=["gene","label"])
            l1m,  _, h2 = load_csv_with_checks(EXPECTED, "labels_1Mb",  required=["gene","label"])
            l2m,  _, h3 = load_csv_with_checks(EXPECTED, "labels_2Mb",  required=["gene","label"])
            def modset(df, h): 
                x = df.rename(columns={h["gene"]:"gene_id", h["label"]:"label"})
                return set(x.loc[x["label"].astype(str).str.lower()=="network_modulator","gene_id"])
            A, B, C = modset(l500,h1), modset(l1m,h2), modset(l2m,h3)
            def jacc(x,y): 
                u = len(x|y)
                return round(len(x&y)/u, 4) if u else 1.0
            out = pd.DataFrame([
                {"pair": "500kb vs 1Mb", "jaccard": jacc(A,B), "enrichment": np.nan, "spearman": np.nan},
                {"pair": "500kb vs 2Mb", "jaccard": jacc(A,C), "enrichment": np.nan, "spearman": np.nan},
                {"pair": "1Mb vs 2Mb",   "jaccard": jacc(B,C), "enrichment": np.nan, "spearman": np.nan},
            ])
            return write_csv(outdir, out, "Table_S3_window_stability_summary.csv")
        except BuildError as e2:
            raise BuildError(f"[S3] Both strategies failed :: {e2}")

def build_S4(EXPECTED, outdir: Path):
    fg, _, hf = load_csv_with_checks(EXPECTED, "foreground_annotated", required=["gene","snp"])
    tf, _, ht = load_csv_with_checks(EXPECTED, "tf_distances", required=["snp"])
    out = fg.rename(columns={hf["gene"]:"gene_id", hf["snp"]:"snp_id"})
    tf  = tf.rename(columns={ht["snp"]:"snp_id"})
    out = out.merge(tf, on="snp_id", how="left")
    keep = ["gene_id","snp_id","chr","pos","vep_consequence","distance_to_tss",
            "distance_to_nearest_TF_peak","within_1kb_TF"]
    out = out[[c for c in keep if c in out.columns]]
    return write_csv(outdir, out, "Table_S4_foreground_snp_annotations.csv")

def build_S5(EXPECTED, outdir: Path):
    """
    Build Table S5 with footnote explaining gene-level vs SNP-level significance.
    
    ISSUE 3 FIX (documented here for completeness):
    The legend file explains why some genes have significant gene_level_q but 
    n_sig_snps=0 and non-significant lead_q.
    """
    res_df,  _, hr = load_csv_with_checks(EXPECTED, "ieqtl_results", required=["gene","snp"])
    
    # Map columns
    pcol = choose_first_present(res_df, "p")
    qcol = choose_first_present(res_df, "q")
    beta = choose_first_present(res_df, "beta")
    
    # Get lead SNP per gene
    if qcol:
        lead = res_df.sort_values(by=qcol, ascending=True).groupby(hr["gene"], as_index=False).first()
        lead = lead.rename(columns={hr["gene"]:"gene_id", hr["snp"]:"lead_snp", qcol:"lead_q"})
    elif pcol:
        lead = res_df.sort_values(by=pcol, ascending=True).groupby(hr["gene"], as_index=False).first()
        lead = lead.rename(columns={hr["gene"]:"gene_id", hr["snp"]:"lead_snp", pcol:"lead_p"})
    else:
        lead = res_df.groupby(hr["gene"], as_index=False).first()
        lead = lead.rename(columns={hr["gene"]:"gene_id", hr["snp"]:"lead_snp"})
    
    if beta and beta in lead.columns:
        lead = lead.rename(columns={beta:"beta_GxE"})
    
    # Count significant SNPs per gene
    if qcol:
        sig = res_df[qcol] <= 0.10
    elif pcol:
        sig = res_df[pcol] <= 5e-8
        log("[S5] q-values not present; using p<=5e-8 as significance proxy")
    else:
        sig = pd.Series(False, index=res_df.index)
        log("[S5] Neither p nor q present; n_sig_snps set to 0")
    
    counts = (res_df.assign(is_sig=sig).groupby(hr["gene"], as_index=False)["is_sig"].sum()
              .rename(columns={hr["gene"]:"gene_id", "is_sig":"n_sig_snps"}))
    
    out = lead.merge(counts, on="gene_id", how="left")
    out["n_sig_snps"] = out["n_sig_snps"].fillna(0).astype(int)
    
    # Add gene-level q-values
    try:
        gene_q, _, hq = load_csv_with_checks(EXPECTED, "ieqtl_gene_fdr", required=["gene","q"])
        gene_q_mapped = gene_q.rename(columns={hq["gene"]:"gene_id", hq["q"]:"gene_level_q"})[["gene_id","gene_level_q"]]
        out = out.merge(gene_q_mapped, on="gene_id", how="left")
    except BuildError as e:
        log(f"[S5] gene-level q not added :: {e}")
    
    # Add TF overlap from ieQTL-specific overlap file (not general tf_distances)
    try:
        tf_path = find_first_existing(ROOT, EXPECTED.get("ieqtl_tf_overlap", []))
        if tf_path and tf_path.exists():
            tf_overlap = pd.read_csv(tf_path)
            # This file has columns: gene, snp, chr, pos, n_tf_peaks, tf_names
            tf_overlap = tf_overlap.rename(columns={"gene": "gene_id", "snp": "lead_snp"})
            tf_overlap["tf_overlap"] = "TRUE"
            out = out.merge(tf_overlap[["gene_id", "tf_overlap"]].drop_duplicates(), 
                          on="gene_id", how="left")
            out["tf_overlap"] = out["tf_overlap"].fillna("FALSE")
            log(f"[S5] TF overlap added from {tf_path} ({len(tf_overlap)} SNPs)")
        else:
            log("[S5] ieqtl_tf_overlap file not found, trying tf_distances fallback")
            tf, _, ht = load_csv_with_checks(EXPECTED, "tf_distances", required=["snp"])
            tf_mapped = tf.rename(columns={ht["snp"]:"lead_snp"})
            flag_col = choose_first_present(tf_mapped, "tf_within_1kb")
            if flag_col:
                tf_mapped["tf_overlap"] = tf_mapped[flag_col]
                out = out.merge(tf_mapped[["lead_snp","tf_overlap"]], on="lead_snp", how="left")
    except BuildError as e:
        log(f"[S5] TF overlap not added :: {e}")
    
    # Select and order columns
    keep = ["gene_id","gene_level_q","n_sig_snps","lead_snp","lead_q","lead_p","beta_GxE"]
    
    # Add chr/pos if available
    for coord in ["chr","pos"]:
        if coord in out.columns:
            keep.append(coord)
    
    # Add tf_overlap at the end
    if "tf_overlap" in out.columns:
        keep.append("tf_overlap")
    
    out = out[[c for c in keep if c in out.columns]]
    
    # Reorder to put gene_level_q first after gene_id
    if "gene_level_q" in out.columns:
        cols = ["gene_id", "gene_level_q"] + [c for c in out.columns if c not in ["gene_id", "gene_level_q"]]
        out = out[cols]
    
    return write_csv(outdir, out, "Table_S5_ieqtl_summary.csv")

def build_S6(EXPECTED, outdir: Path, strict_motif: bool = False):
    path = find_first_existing(ROOT, EXPECTED.get("motif_summary", []))
    if path is None:
        msg = (f"[S6] motif_summary not found :: [MISSING] 'motif_summary' not found. Looked for:\n"
               f"  - " + "\n  - ".join(map(str, EXPECTED.get("motif_summary", []))) +
               f"\nCheck REPRODUCE steps or folder under {ROOT}.")
        if strict_motif:
            raise BuildError(msg)
        else:
            log(msg)
            return None
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise BuildError(f"[READ-ERROR] 'motif_summary' at {path}: {e}")

    lowers = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lowers: return lowers[n.lower()]
        return None

    snp  = pick("snp_id","snp","variant","rsid","id")
    gene = pick("gene_id","gene","zm_gene","GeneID")
    chrC = pick("chr","chrom","chromosome")
    posC = pick("pos","position","bp")
    motif_name = pick("motif_name","tf_name","TF","TF_name","name")
    motif_family = pick("motif_family","tf_family")
    dllr = pick("delta_llr","ΔLLR","deltaLLR","llr_delta","score_diff","delta_score")
    adllr= pick("abs_delta_llr","abs_delta","absDLLR")
    edir = pick("effect_direction","direction")

    out = pd.DataFrame()
    if snp:  out["snp_id"] = df[snp]
    if gene: out["gene_id"] = df[gene]
    if chrC: out["chr"] = df[chrC]
    if posC: out["pos"] = df[posC]
    if motif_name: out["motif_name"] = df[motif_name]
    if motif_family: out["motif_family"] = df[motif_family]
    if dllr: out["delta_llr"] = df[dllr]
    if adllr: out["abs_delta_llr"] = df[adllr]
    if edir: out["effect_direction"] = df[edir]

    if out.empty:
        msg = f"[S6] motif_summary loaded from {path}, but no expected columns were found; S6 omitted."
        if strict_motif:
            raise BuildError(msg)
        else:
            log(msg)
            return None

    return write_csv(outdir, out, "Table_S6_motif_disruption.csv")

def std_go(df: pd.DataFrame, label: str) -> pd.DataFrame:
    d = df.copy()
    d["set"] = label
    for src in ["GO_ID","GO","GO.ID","ID","go_id","go"]:
        if src in d.columns and "GO_ID" not in d.columns:
            d.rename(columns={src:"GO_ID"}, inplace=True)
            break
    for src in ["term","Term","Description","Name","GO_term","GO.Term","GO_Term_Name"]:
        if src in d.columns and "term" not in d.columns:
            d.rename(columns={src:"term"}, inplace=True)
            break
    for src in ["p","pvalue","p_value","PValue","p.gof"]:
        if src in d.columns and "p" not in d.columns:
            d.rename(columns={src:"p"}, inplace=True)
    for src in ["q","qvalue","q_value","qval","FDR","fdr","p.adjust","padj","adj.P.Val"]:
        if src in d.columns and "q" not in d.columns:
            d.rename(columns={src:"q"}, inplace=True)
    for src,dst in [("GeneRatio","GeneRatio"),("Gene_Ratio","GeneRatio"),("gene_ratio","GeneRatio"),
                    ("BgRatio","BgRatio"),("BackgroundRatio","BgRatio"),("bg_ratio","BgRatio"),
                    ("k","k"),("Count","k"),("overlap","k"),("Hits","k"),
                    ("K","K"),("Size","K"),("N","K"),("Background","K")]:
        if src in d.columns and dst not in d.columns:
            d.rename(columns={src:dst}, inplace=True)
    return d

def build_S7(EXPECTED, outdir: Path, strict_go: bool = False):
    mod, _, _ = load_csv_with_checks(EXPECTED, "go_modulators", required=None)
    drv, _, _ = load_csv_with_checks(EXPECTED, "go_drivers",   required=None)
    M = std_go(mod, "modulator")
    D = std_go(drv, "driver")
    cols_pref = ["set","GO_ID","term","p","q","k","K","GeneRatio","BgRatio"]
    present = [c for c in cols_pref if (c in M.columns) or (c in D.columns)]
    out = pd.concat([M[present], D[present]], ignore_index=True, sort=False)
    missing_core = [c for c in ["GO_ID","term"] if c not in out.columns]
    if missing_core:
        msg = f"[S7] Missing critical GO columns: {missing_core}"
        if strict_go:
            raise BuildError(msg)
        log(msg + " - proceeding.")
    if "q" in out.columns:
        out = out.sort_values(by="q", ascending=True)
    elif "p" in out.columns:
        out = out.sort_values(by="p", ascending=True)
    return write_csv(outdir, out, "Table_S7_GO_enrichment_combined.csv")

def build_DS1(EXPECTED, outdir: Path):
    l500, _, h1 = load_csv_with_checks(EXPECTED, "labels_500kb", required=["gene","label"])
    l1m,  _, h2 = load_csv_with_checks(EXPECTED, "labels_1Mb",  required=["gene","label"])
    l2m,  _, h3 = load_csv_with_checks(EXPECTED, "labels_2Mb",  required=["gene","label"])
    A = l500.rename(columns={h1["gene"]:"gene_id", h1["label"]:"label_500kb"})
    B = l1m.rename(columns={h2["gene"]:"gene_id", h2["label"]:"label_1Mb"})
    C = l2m.rename(columns={h3["gene"]:"gene_id", h3["label"]:"label_2Mb"})
    out = A.merge(B, on="gene_id", how="outer").merge(C, on="gene_id", how="outer")
    return write_csv(outdir, out, "Data_S1_full_gene_classifications.csv")

def build_DS2(EXPECTED, outdir: Path):
    df, _, _ = load_csv_with_checks(EXPECTED, "ieqtl_results", required=None)
    return write_csv(outdir, df, "Data_S2_ieqtl_results_full.csv")

def build_DS3(EXPECTED, outdir: Path):
    df, _, _ = load_csv_with_checks(EXPECTED, "background_annotated", required=None)
    return write_csv(outdir, df, "Data_S3_background_snps.csv")

def main():
    global ROOT, OUTDIR
    parser = argparse.ArgumentParser(description="Build Supplementary Materials (v4j - Table S5 data source fix)")
    parser.add_argument("--root", type=str, default=str(ROOT), help="Project root directory")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR), help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Zip outputs into versioned archive")
    parser.add_argument("--strict-motif", action="store_true", help="Fail if motif_summary missing")
    parser.add_argument("--strict-go", action="store_true", help="Fail if core GO columns missing")
    args = parser.parse_args()

    ROOT = Path(args.root).resolve()
    OUTDIR = Path(args.outdir).resolve()
    EXP = expected_paths(ROOT)

    ok = True
    log("=== Supplement Builder v4j (Table S5 data source fix): starting ===")
    log(f"ROOT:   {ROOT}")
    log(f"OUTDIR: {OUTDIR}")
    
    try:
        build_S1(EXP, OUTDIR)
        build_S2(EXP, OUTDIR)
        build_S3(EXP, OUTDIR)
        build_S4(EXP, OUTDIR)
        build_S5(EXP, OUTDIR)
        build_S6(EXP, OUTDIR, strict_motif=args.strict_motif)
        build_S7(EXP, OUTDIR, strict_go=args.strict_go)
        build_DS1(EXP, OUTDIR)
        build_DS2(EXP, OUTDIR)
        build_DS3(EXP, OUTDIR)
        create_table_legends(OUTDIR)
        manifest(OUTDIR)
        log("=== Supplement Builder v4j: finished ===")
    except BuildError as e:
        ok = False
        log("=== FAILED ===")
        log(str(e))

    save_report(OUTDIR, ok)

    if args.zip:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        zpath = OUTDIR / f"supplement_package_{stamp}.zip"
        added = set()
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in OUTPUTS:
                try:
                    zf.write(p, arcname=p.name)
                    added.add(p.name)
                except Exception as e:
                    log(f"[ZIP] Could not add {p}: {e}")
            rep = OUTDIR / "validation_report.json"
            if rep.exists() and rep.name not in added:
                zf.write(rep, arcname=rep.name)
        log(f"[ZIP] Wrote {zpath}")

if __name__ == "__main__":
    main()