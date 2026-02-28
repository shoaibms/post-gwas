"""
Generate motif disruption summary and TF family bar plot from annotated results.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT  = BASE / "output" / "week6_ieqtl"
top  = pd.read_csv(OUT / "motif_delta_llr_top_per_snp_annot.csv")
ie   = pd.read_csv(OUT / "ieqtl_final_results.csv")

# keep the 9 TF-peak SNPs only (present in motif file)
cols = ["gene","snp","chr","pos","tf_name","tf_family","delta_llr","abs_delta_llr"]
summ = top[cols].copy()

# add effect_direction from ieQTL if present
if "effect_direction" in ie.columns:
    ie_small = ie[["gene","snp","effect_direction"]].drop_duplicates()
    summ = summ.merge(ie_small, on=["gene","snp"], how="left")

summ = summ.sort_values("abs_delta_llr", ascending=False)
summ_file = OUT / "figure_motif_summary.csv"
summ.to_csv(summ_file, index=False)
print(f"Saved: {summ_file}")

# simple family bar (top-per-SNP)
ax = summ["tf_family"].value_counts().plot(kind="bar")
ax.set_xlabel("TF family"); ax.set_ylabel("Count (top per SNP)")
ax.set_title("TF families among top per-SNP disruptions")
plt.tight_layout()
plt.savefig(OUT / "figure_motif_families_bar.png", dpi=300)
print(f"Saved: {OUT / 'figure_motif_families_bar.png'}")
