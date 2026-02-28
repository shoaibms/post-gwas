#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Clean Data Preparation Script for Environment-Conditional eQTL Transformers
===========================================================================

This script prepares model-ready data for analyzing environment-conditional eQTLs.
Combines two major workflows:
1. Expression/Phenotype Data Prep: Processes FPKM, phenotype, and GFF3 files.
2. Genotype Data Prep: Automates PLINK2 pipeline for VCF processing, LD pruning, PCA.

Usage Examples:
--------------
# Both workflows with core cohort (185 accessions)
python clean_prep_data.py --cohort-csv output/cohort/core_all3_env.csv

# Only expression workflow
python clean_prep_data.py --cohort-csv output/cohort/core_all3_env.csv --run-expression

# Different cohort (221 accessions with any transcriptome data)
python clean_prep_data.py --cohort-csv output/cohort/G_and_anyT.csv
"""

from __future__ import annotations
import argparse
import gzip
import logging
import re
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

# ----------------- Setup Logging -----------------

log = logging.getLogger(__name__)

def setup_logging():
    """Configure logging to print to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

# ----------------- General Utility Functions -----------------

def ensure_dir(p: Path) -> None:
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def canonicalize_name(x: str) -> str:
    if x is None: return ""
    s = str(x).strip().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"__+", "_", s)
    return s.upper()

# ----------------- Expression/Phenotype Workflow Functions -----------------

def detect_delim(first_line: str) -> str:
    if "\t" in first_line: return "\t"
    if "," in first_line: return ","
    return r"\s+"

def read_first_line(path: Path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.readline().rstrip("\n")

def read_fpkm_matrix_row_oriented(path: Path) -> pd.DataFrame:
    log.info(f"Reading FPKM matrix: {path.name}...")
    first = read_first_line(path)
    sep = detect_delim(first)
    df = pd.read_csv(path, sep=sep, compression="infer", low_memory=False)
    acc_col = df.columns[0]
    df.rename(columns={acc_col: "accession_raw"}, inplace=True)
    df["accession"] = df["accession_raw"].astype(str).map(canonicalize_name)
    df = df.drop(columns=["accession_raw"]).set_index("accession")
    num_cols = [c for c in df.columns if c is not None]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    log.info(f"Finished reading {path.name} - shape: {df.shape}")
    return df

def intersect_gene_sets(dfs: List[pd.DataFrame]) -> List[str]:
    sets = [set(d.columns) for d in dfs]
    common = set.intersection(*sets)
    ww_cols = list(dfs[0].columns)
    return [g for g in ww_cols if g in common]

def find_sheet_with_headers(xlsx: Path, required_cols: List[str]) -> Tuple[str, int]:
    xl = pd.ExcelFile(xlsx)
    req = [c.lower() for c in required_cols]
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None, nrows=60)
        for i in range(len(df)):
            row = df.iloc[i].astype(str).tolist()
            norm = [c.strip().lower() for c in row if str(c) != "nan" and c.strip()]
            if norm and all(any(rc == c or rc in c for c in norm) for rc in req):
                return sheet, i
    raise RuntimeError(f"Could not find {required_cols} in {xlsx.name}")

def read_P_from_2021(supp_xlsx: Path) -> pd.DataFrame:
    sheet, hdr = find_sheet_with_headers(supp_xlsx, ["Name", "Survival rate"])
    df = pd.read_excel(supp_xlsx, sheet_name=sheet, header=hdr)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "name": rename[c] = "Name"
        elif "survival" in cl: rename[c] = "Survival rate (%)"
    df = df.rename(columns=rename).dropna(subset=["Name"]).copy()
    df["accession"] = df["Name"].astype(str).map(canonicalize_name)
    out = df[["accession", "Survival rate (%)"]].copy()
    return out.rename(columns={"Survival rate (%)": "survival_rate"})

def parse_gff3_chunk(lines: List[str]) -> List[Tuple]:
    rows = []
    for line in lines:
        if not line or line.startswith("#"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9: continue
        chrom, _, ftype, start, end, _, strand, _, attrs = parts
        if ftype.lower() != "gene": continue
        ad: Dict[str, str] = dict(kv.split("=", 1) for kv in attrs.split(";") if "=" in kv)
        gid_raw = ad.get("ID", "")
        gid = gid_raw.split(":")[-1].split(".")[0]
        gname = ad.get("Name", ad.get("gene_name", ""))
        rows.append((gid, gname, chrom, int(start), int(end), strand))
    return rows

def parse_gff3_gene_map_parallel(gff3_path: Path, num_workers: int) -> pd.DataFrame:
    log.info("Reading GFF3 file into memory...")
    opener = gzip.open if str(gff3_path).endswith(".gz") else open
    with opener(gff3_path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if not lines: return pd.DataFrame(columns=["gene_id","gene_name","chr","start","end","strand"])
    
    line_chunks = np.array_split(lines, num_workers)
    log.info(f"Parsing GFF3 in parallel across {num_workers} cores...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(parse_gff3_chunk, line_chunks)
    
    all_rows = [row for chunk in results for row in chunk]
    if not all_rows: return pd.DataFrame(columns=["gene_id","gene_name","chr","start","end","strand"])

    df = pd.DataFrame(all_rows, columns=["gene_id","gene_name","chr","start","end","strand"])
    return df.drop_duplicates(subset=["gene_id"])

def subset_env_parallel(df: pd.DataFrame, cohort_set: set, genes: List[str]) -> pd.DataFrame:
    df2 = df.loc[df.index.intersection(cohort_set), genes].copy()
    return df2.reset_index().rename(columns={"index": "accession"})

def melt_and_tag(df: pd.DataFrame, env: str) -> pd.DataFrame:
    m = df.melt(id_vars=["accession"], var_name="gene_id", value_name="fpkm")
    m["env"] = env
    return m

def write_csv_parallel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log.info(f"Wrote {path}")

def run_expression_workflow(args):
    """Main function to execute the expression and phenotype data preparation."""
    outdir = Path(args.expression_outdir)
    ensure_dir(outdir)
    num_workers = args.cores

    # 1) Load cohort
    log.info(f"Loading cohort from: {args.cohort_csv}")
    cohort = pd.read_csv(args.cohort_csv)
    # Use first column as accession, regardless of column name
    acc_col = cohort.columns[0]
    cohort["accession"] = cohort[acc_col].astype(str).map(canonicalize_name)
    cohort_set = set(cohort["accession"])
    log.info(f"Cohort size: {len(cohort_set)}")
    log.info(f"Cohort preview: {list(cohort['accession'].head(10))}")

    # Parallel processing block
    with Pool(processes=num_workers) as pool:
        # 2) Read FPKM matrices in parallel
        fpkm_paths = [Path(args.ww), Path(args.ws1), Path(args.ws2)]
        fpkm_dfs = pool.map(read_fpkm_matrix_row_oriented, fpkm_paths)
        ww, ws1, ws2 = fpkm_dfs
        log.info(f"WW shape {ww.shape}; WS1 {ws1.shape}; WS2 {ws2.shape}")

        # 3) Gene intersection
        genes = intersect_gene_sets(fpkm_dfs)
        log.info(f"Gene intersection: {len(genes):,} genes")

        # 4) Subset to cohort & gene intersection in parallel
        p_subset_env = partial(subset_env_parallel, cohort_set=cohort_set, genes=genes)
        subset_dfs = pool.map(p_subset_env, fpkm_dfs)
        ww_sub, ws1_sub, ws2_sub = subset_dfs
        log.info(f"Subsets: WW {ww_sub.shape}, WS1 {ws1_sub.shape}, WS2 {ws2_sub.shape}")

        # 5) Write long table if requested
        if args.to_parquet:
            log.info("Creating long format table...")
            melt_tasks = [(ww_sub, "WW"), (ws1_sub, "WS1"), (ws2_sub, "WS2")]
            t_long_list = pool.starmap(melt_and_tag, melt_tasks)
            T_long = pd.concat(t_long_list, ignore_index=True)[["accession","env","gene_id","fpkm"]]
            try:
                T_long.to_parquet(outdir / "T_long.parquet", index=False)
                log.info(f"Wrote {outdir / 'T_long.parquet'} ({len(T_long):,} rows)")
            except Exception as e:
                log.warning(f"Parquet write failed: {e}")

        # 6) Write wide CSVs if requested
        if args.write_wide_csv:
            log.info("Writing wide format CSVs...")
            write_tasks = [(ww_sub, outdir / "T_WW.csv"), (ws1_sub, outdir / "T_WS1.csv"), (ws2_sub, outdir / "T_WS2.csv")]
            pool.starmap(write_csv_parallel, write_tasks)

    # 7) Phenotype P
    log.info("Processing phenotype data...")
    P = read_P_from_2021(Path(args.supp2021))
    P = P[P["accession"].isin(cohort_set)].copy().sort_values("accession")
    P.to_csv(outdir / "P.csv", index=False)
    log.info(f"Wrote {outdir / 'P.csv'} (n={len(P)})")

    # 8) Gene map
    log.info("Processing gene annotations...")
    gene_map = parse_gff3_gene_map_parallel(Path(args.gff3), num_workers)
    if not gene_map.empty:
        gene_map.to_csv(outdir / "gene_map.csv", index=False)
        log.info(f"Wrote {outdir / 'gene_map.csv'} (n={len(gene_map):,})")
    else:
        log.warning("gene_map is empty")

# ----------------- Genotype Workflow Functions -----------------

def run_command(cmd: list[str], step_name: str) -> None:
    """Executes a command, prints its progress, and handles errors."""
    log.info(f"--- [Step {step_name}] ---")
    log.info(f"CMD: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True, encoding='utf-8', errors='replace')
        log.info(f"Step '{step_name}' completed successfully.")
    except FileNotFoundError:
        log.error(f"Command '{cmd[0]}' not found. Is PLINK2 in your PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError:
        log.error(f"Step '{step_name}' failed. Check PLINK2 logs for details.")
        sys.exit(1)

def create_keep_file(cohort_csv: Path, out_path: Path) -> int:
    """Creates the sample keep-list for PLINK."""
    log.info("--- [Step 0: Create Sample Keep-List] ---")
    df = pd.read_csv(cohort_csv)
    # Use first column as accession, regardless of name
    accession_col = df.columns[0]
    accessions = df[accession_col]
    accessions.to_csv(out_path, index=False, header=False)
    log.info(f"Wrote {len(accessions)} accessions to '{out_path}'")
    return len(accessions)

def run_genotype_workflow(args):
    """Main function to execute the genotype data preparation using PLINK2."""
    outdir = Path(args.geno_outdir)
    ensure_dir(outdir)
    plink2 = args.plink2_path

    # Step 0: Create keep file
    keep_file = outdir / "sample_keep.txt"
    create_keep_file(args.cohort_csv, keep_file)

    # Step 1: VCF -> PGEN
    cohort_pgen_prefix = outdir / "cohort"
    cmd_vcf_to_pgen = [
        plink2,
        "--vcf", str(args.vcf),
        "--keep", str(keep_file),
        "--set-all-var-ids", "'@:#$r$a'",
        "--make-pgen",
        "--out", str(cohort_pgen_prefix)
    ]
    run_command(cmd_vcf_to_pgen, "1) VCF to PGEN")

    # Step 2: LD-pruning
    prune_out_prefix = outdir / "cohort_prune"
    run_command([plink2, "--pfile", str(cohort_pgen_prefix), "--indep-pairwise", args.ld_window, args.ld_step, args.ld_r2, "--out", str(prune_out_prefix)], "2a) LD Pruning List")
    
    pruned_pgen_prefix = outdir / "cohort_pruned"
    snps_to_extract_file = prune_out_prefix.with_suffix(".prune.in")
    run_command([plink2, "--pfile", str(cohort_pgen_prefix), "--extract", str(snps_to_extract_file), "--make-pgen", "--out", str(pruned_pgen_prefix)], "2b) Extract Pruned SNPs")

    # Step 3: PCA
    pcs_out_prefix = outdir / "pcs"
    run_command([plink2, "--pfile", str(pruned_pgen_prefix), "--pca", args.pca_comps, "approx", "--out", str(pcs_out_prefix)], "3) PCA")

    # Step 4: Export TRAW
    traw_out_prefix = outdir / "G_traw"
    run_command([plink2, "--pfile", str(pruned_pgen_prefix), "--export", "A-transpose", "--out", str(traw_out_prefix)], "4) Export TRAW")

    log.info(f"\n[SUCCESS] Genotype preparation workflow completed in: {outdir}")

# ----------------- Main Controller -----------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Data Preparation Script for Environment-Conditional eQTL Analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Workflow Selection ---
    ap.add_argument("--run-expression", action="store_true", help="Run only the expression/phenotype workflow.")
    ap.add_argument("--run-genotype", action="store_true", help="Run only the genotype workflow using PLINK2.")

    # --- REQUIRED: Cohort CSV (no default!) ---
    ap.add_argument("--cohort-csv", required=True, type=Path, 
                    help="REQUIRED: Path to cohort CSV file (e.g., output/cohort/core_all3_env.csv)")

    # --- Shared Arguments ---
    ap.add_argument("--cores", type=int, default=cpu_count(), 
                    help=f"Number of CPU cores for parallel tasks (default: {cpu_count()})")

    # --- Expression Workflow Arguments ---
    exp_group = ap.add_argument_group('Expression Workflow Arguments')
    exp_group.add_argument("--expression-outdir", default="output/data", type=Path, 
                          help="Output directory for expression/phenotype files.")
    exp_group.add_argument("--ww", default="data/maize/WW_209-Uniq_FPKM.txt.gz", type=Path)
    exp_group.add_argument("--ws1", default="data/maize/WS1_208-uniq_FPKM.txt.gz", type=Path)
    exp_group.add_argument("--ws2", default="data/maize/WS2_210-uniq_FPKM.txt.gz", type=Path)
    exp_group.add_argument("--supp2021", default="data/maize/13059_2021_2481_MOESM2_ESM.xlsx", type=Path)
    exp_group.add_argument("--gff3", default="data/maize/Zea_mays.B73_RefGen_v4.gff3.gz", type=Path)
    exp_group.add_argument("--no-write-wide-csv", action="store_false", dest="write_wide_csv", 
                          help="Do NOT write the wide T_*.csv files.")
    exp_group.add_argument("--no-parquet", action="store_false", dest="to_parquet", 
                          help="Do NOT write the T_long.parquet file.")

    # --- Genotype Workflow Arguments ---
    geno_group = ap.add_argument_group('Genotype Workflow Arguments')
    geno_group.add_argument("--geno-outdir", default="output/geno", type=Path, 
                           help="Output directory for genotype files.")
    geno_group.add_argument("--plink2-path", default=None, 
                           help="Path to PLINK2 executable. Auto-detects if not specified.")
    geno_group.add_argument("--vcf", default="data/maize/zea_mays_miss0.6_maf0.05.recode.vcf.gz", type=Path)
    geno_group.add_argument("--ld-window", default="50", help="LD pruning: window size (kb).")
    geno_group.add_argument("--ld-step", default="5", help="LD pruning: variant step count.")
    geno_group.add_argument("--ld-r2", default="0.2", help="LD pruning: r-squared threshold.")
    geno_group.add_argument("--pca-comps", default="20", help="Number of principal components.")

    return ap.parse_args()

def resolve_paths(args: argparse.Namespace) -> None:
    """Ensure all relevant file paths in args are absolute."""
    base_dir = Path.cwd()
    path_attrs = [
        'cohort_csv', 'ww', 'ws1', 'ws2', 'supp2021', 'gff3', 'vcf',
        'expression_outdir', 'geno_outdir'
    ]
    for attr in path_attrs:
        path_val = getattr(args, attr, None)
        if path_val and not Path(path_val).is_absolute():
            setattr(args, attr, base_dir / path_val)

def find_plink2(args: argparse.Namespace) -> str:
    """Find the PLINK2 executable."""
    if args.plink2_path:
        return args.plink2_path

    # Check for plink2.exe next to the script
    script_dir = Path(__file__).parent.resolve()
    potential_path = script_dir / "plink2.exe"
    if potential_path.exists():
        log.info(f"Auto-detected plink2.exe at: {potential_path}")
        return str(potential_path)

    # Check system PATH
    plink_in_path = shutil.which("plink2")
    if plink_in_path:
        log.info(f"Found plink2 in PATH: {plink_in_path}")
        return plink_in_path

    log.warning("Could not find plink2. Assuming it's in the PATH as 'plink2'.")
    return "plink2"

def main():
    """Main controller for the data preparation script."""
    setup_logging()
    args = get_args()
    resolve_paths(args)
    args.plink2_path = find_plink2(args)

    # Validation
    if not args.cohort_csv.exists():
        log.error(f"Cohort file not found: {args.cohort_csv}")
        sys.exit(1)

    log.info(f"Using cohort file: {args.cohort_csv}")
    log.info(f"Working directory: {Path.cwd()}")

    # If no specific workflow is chosen, run both
    run_all = not args.run_expression and not args.run_genotype

    if run_all or args.run_expression:
        log.info("="*80)
        log.info(" " * 25 + "STARTING EXPRESSION WORKFLOW")
        log.info("="*80)
        run_expression_workflow(args)

    if run_all or args.run_genotype:
        log.info("="*80)
        log.info(" " * 27 + "STARTING GENOTYPE WORKFLOW")
        log.info("="*80)
        run_genotype_workflow(args)

    log.info("--- All requested workflows complete ---")

if __name__ == "__main__":
    main()