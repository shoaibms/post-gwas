#!/usr/bin/env python3
"""
Build a confirmatory gene set of top G×E candidates from robust LMM results.
Outputs a ranked gene list and a detailed manifest for downstream validation.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

BASE = Path(r"C:\Users\ms\Desktop\gwas")
LMM = BASE / r"output\robust_lmm_analysis\tables\robust_lmm_comprehensive_results.csv"
OUT_LIST = BASE / r"data\maize\process\lists\gxe_confirmatory_21.csv"
OUT_DETAILS = BASE / r"output\final_analysis_reports\gene_modules\gxe_confirmatory_21_details.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmm", default=str(LMM))
    ap.add_argument("--out_list", default=str(OUT_LIST))
    ap.add_argument("--out_details", default=str(OUT_DETAILS))
    ap.add_argument("--n", type=int, default=21)
    args = ap.parse_args()

    df = pd.read_csv(args.lmm)
    # Prefer AGPv4 IDs from 'gene_name' if present; else use 'gene_id'
    gid_col = "gene_name" if "gene_name" in df.columns else "gene_id"
    df[gid_col] = df[gid_col].astype(str).str.strip()

    # G×E score = max transfer; fallback to |delta_r2|
    transfer_cols = [c for c in df.columns if c.startswith("transfer_")]
    if transfer_cols:
        df["gxe_score"] = df[transfer_cols].max(axis=1)
    else:
        df["gxe_score"] = df["delta_r2"].abs()

    # Rank by G×E, then stronger signal by r2_full
    sort_cols = ["gxe_score"] + (["r2_full"] if "r2_full" in df.columns else [])
    top = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).head(args.n).copy()

    # Save list (schema expected by downstream scripts)
    out = top[[gid_col]].rename(columns={gid_col: "gene_id"})
    Path(args.out_list).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_list, index=False)

    # Save transparent manifest
    keep = [gid_col, "gxe_score", "r2_full", "r2_null", "delta_r2"] + transfer_cols
    keep = [c for c in keep if c in top.columns]
    det = top[keep].rename(columns={gid_col: "gene_id"})
    Path(args.out_details).parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(args.out_details, index=False)

    print(f"[OK] Wrote list: {args.out_list}")
    print(f"[OK] Wrote details: {args.out_details}")

if __name__ == "__main__":
    main()
