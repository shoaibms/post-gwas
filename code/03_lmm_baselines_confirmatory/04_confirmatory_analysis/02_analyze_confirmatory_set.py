#!/usr/bin/env python3
"""
Confirmatory Analysis of Top G×E Genes
----------------------------------------
Merges a confirmatory gene list with ECT out-of-fold R² and cis-attention
results, then computes delta-R² statistics and generates diagnostic plots.
"""
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE = Path(r"C:\Users\ms\Desktop\gwas\output")
# Use the CORRECTED ECT files with real gene IDs
ECT_DIR = BASE / r"ect_v3m_drought_full_100"
LIST = Path(r"C:\Users\ms\Desktop\gwas\data\maize\process\lists\gxe_confirmatory_21.csv")
OUT = BASE / r"final_analysis_reports\confirmatory_21"
FIG = OUT / "figures"

# --- UTILITY FUNCTIONS ---

def _normalize_gene_id(series: pd.Series) -> pd.Series:
    """Aggressively cleans gene IDs to ensure robust merging."""
    return series.astype(str).str.strip().str.upper()

def load_list(p: Path) -> pd.Series:
    df = pd.read_csv(p)
    col = "gene_id" if "gene_id" in df.columns else df.columns[0]
    return _normalize_gene_id(df[col])

def load_r2(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.rename(columns={"gene_name": "gene_id"}) # Handle potential column name variations
    df["gene_id"] = _normalize_gene_id(df["gene_id"])
    if "delta" not in df.columns and {"r2_null", "r2_full"}.issubset(df.columns):
        df["delta"] = df["r2_full"] - df["r2_null"]
    return df[["gene_id", "r2_null", "r2_full", "delta"]]

def load_cis(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.rename(columns={"gene_name": "gene_id"})
    df["gene_id"] = _normalize_gene_id(df["gene_id"])
    return df

def ecdf(arr):
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def sign_test_counts(d):
    pos = int((d > 1e-6).sum())
    neg = int((d < -1e-6).sum())
    zero = len(d) - pos - neg
    n_nonzero = pos + neg
    p_two = 1.0
    if n_nonzero > 0:
        k = min(pos, neg)
        p_two = min(1.0, 2.0 * sum(math.comb(n_nonzero, i) * (0.5 ** n_nonzero) for i in range(k + 1)))
    return pos, neg, zero, n_nonzero, p_two

# --- ANALYSIS FUNCTIONS ---

def analyze_delta_distribution(df, out_dir, fig_dir, seed):
    d = df["delta"].to_numpy()
    pos, neg, zero, n_nonzero, p_two = sign_test_counts(d)
    
    # Bootstrap median CI
    rng = np.random.default_rng(seed)
    n = len(d)
    boot_medians = [np.median(d[rng.integers(0, n, size=n)]) for _ in range(5000)]
    ci_lo = float(np.quantile(boot_medians, 0.025))
    ci_hi = float(np.quantile(boot_medians, 0.975))
    
    summary = pd.DataFrame([{
        "N": n, "median_delta": float(np.median(d)),
        "median_delta_boot_CI95_low": ci_lo, "median_delta_boot_CI95_high": ci_hi,
        "n_pos": pos, "n_neg": neg, "n_zero": zero, "sign_test_p_two_sided": p_two
    }])
    summary.to_csv(out_dir / "confirmatory_21_delta_summary.csv", index=False)
    
    # ECDF plot
    X, Y = ecdf(d)
    plt.figure(figsize=(5, 4))
    plt.plot(X, Y, drawstyle="steps-post")
    plt.axvline(0, ls="--", color="grey")
    plt.xlabel("ΔR² (ECT Full − Null)")
    plt.ylabel("ECDF")
    plt.title(f"Confirmatory {n} Genes: ECDF of ΔR²")
    plt.tight_layout()
    plt.savefig(fig_dir / "ecdf_delta_confirmatory_21.png", dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default=str(LIST))
    parser.add_argument("--ect_r2", default=str(ECT_DIR / "ect_oof_r2_by_gene_corrected.csv"))
    parser.add_argument("--ect_cis", default=str(ECT_DIR / "ect_cis_mass_by_env_corrected.csv"))
    parser.add_argument("--out_dir", default=str(OUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    g21 = load_list(Path(args.list))
    r2 = load_r2(Path(args.ect_r2))
    cis = load_cis(Path(args.ect_cis))

    df = pd.merge(r2, cis, on="gene_id", how="inner")
    df_subset = df[df["gene_id"].isin(g21)]
    
    # --- ADDED VALIDATION STEP ---
    if df_subset.empty:
        print("[ERROR] Merge failed: No common gene IDs found between the list and the results.")
        print(f"  - Example IDs from your list file ({args.list}):")
        print(f"    {g21.head().tolist()}")
        print(f"  - Example IDs from your R2 results file ({args.ect_r2}):")
        print(f"    {r2['gene_id'].head().tolist()}")
        print("\nPlease check for formatting differences or typos in your gene lists.")
        return

    n = len(df_subset)
    print(f"[OK] Confirmatory set merged successfully: N={n}")
    df_subset.to_csv(out / "confirmatory_21_joined_data.csv", index=False)

    # Run the main analysis on the subset
    analyze_delta_distribution(df_subset, out, fig, args.seed)
    
    print(f"\n[SUCCESS] Confirmatory analysis for {n} genes complete.")
    print(f"Results saved to: {out}")

if __name__ == "__main__":
    main()