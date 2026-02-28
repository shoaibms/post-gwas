# REPRODUCE_03_lmm — LMM, Baselines, and Confirmatory Analysis

## Objective
This guide details the execution of the manuscript's core analytical workflows. It begins by establishing performance baselines with standard eQTL methods, proceeds to the primary Genotype-by-Environment (G×E) discovery using our robust Linear Mixed Model (LMM), and concludes with a targeted validation of the top G×E candidate genes.

---

## Prerequisites

**Data** (final outputs from Preprocessing and Transformer stages):
```
output/data/T_long.parquet
output/geno/G_traw.traw
output/geno/pcs.eigenvec
output/data/gene_map.csv
output/ect/bundles/transformer_data_win1Mb.pt
```

**Software**
- Python ≥ 3.9
- R ≥ 4.1

**Dependencies** (run from project root: `C:\Users\ms\Desktop\gwas\`)
```bash
# For Python
pip install -r requirements.txt

# For R, ensure required packages are installed, for example:
# install.packages(c("data.table", "MatrixEQTL", "arrow", "jsonlite"))
```
> Note: The R scripts in this section contain internal `library()` calls. Ensure all listed packages are installed in your R environment before execution.

---

## Workflow Execution
Run all commands from the project root:
```
C:\Users\ms\Desktop\gwas\
```

### Step 1 — Run Baseline eQTL Analyses (R)
These scripts establish performance baselines for comparison against the primary LMM.

#### 1a — Standard Per-Environment eQTL Discovery
**Script:** `code/03_baseline_and_comparative_models/01_run_baseline_matrix_eqtl.R`
```bash
Rscript code/03_baseline_and_comparative_models/01_run_baseline_matrix_eqtl.R
```
**Key Outputs** (created under `output/baselines_eqtl/`):
```
WW/cis_eqtls_significant.txt
WS1/cis_eqtls_significant.txt
WS2/cis_eqtls_significant.txt
analysis_metadata.json
```

#### 1b — Cross-Environment eQTL Scoring (CEES)
**Script:** `code/03_baseline_and_comparative_models/02_run_cees_prediction.R`
```bash
Rscript code/03_baseline_and_comparative_models/02_run_cees_prediction.R
```
**Key Outputs** (created under `output/cees_analysis/`):
```
cees_detailed_results.csv
cees_summary_statistics.csv
```

#### 1c — Build Per-Environment Gene Panels for Stability Analysis
**Script:** `code/03_baseline_and_comparative_models/03_build_environment_gene_panels.R`
```bash
Rscript code/03_baseline_and_comparative_models/03_build_environment_gene_panels.R
```
**Key Outputs** (created under `output/data/`):
```
gene_map_cisQTL_WW.csv   (and corresponding files for WS1/WS2)
eqtl_panel_paths.rds
```

---

### Step 2 — Execute & Validate the Primary G×E LMM
This is the core discovery engine of the manuscript.

#### 2a — Run Robust LMM for G×E Discovery
**Script:** `code/02_primary_gxe_lmm_analysis/01_run_robust_lmm.py`
```bash
python code/02_primary_gxe_lmm_analysis/01_run_robust_lmm.py ^
  --data-file  output\ect\bundles\transformer_data_win1Mb.pt ^
  --output-dir output\robust_lmm_analysis ^
  --per-env-lmm
```
**Key Outputs** (created under `output\robust_lmm_analysis\`):
```
tables/robust_lmm_comprehensive_results.csv
tables/robust_lmm_stress_triggered_hits.csv
logs/convergence_log.csv
```

#### 2b — Validate LMM Results and Generate Diagnostics
**Script:** `code/02_primary_gxe_lmm_analysis/02_validate_lmm_results.py`
```bash
python code/02_primary_gxe_lmm_analysis/02_validate_lmm_results.py ^
  --results_path   output\robust_lmm_analysis\tables\robust_lmm_comprehensive_results.csv ^
  --top_genes_path output\robust_lmm_analysis\tables\robust_lmm_stress_triggered_hits.csv ^
  --output_dir     output\robust_lmm_analysis\diagnostics
```
**Key Outputs** (created under `output\robust_lmm_analysis\diagnostics\`):
```
lmm_diagnostics.png
lmm_diagnostics.pdf
```

---

### Step 3 — Run Confirmatory Analysis
This final analysis validates the top G×E hits from the LMM using a different analytical framework. Note that this step requires correcting gene IDs in older ECT output files first.

#### 3a — Utility: Correct Gene IDs in ECT Result Files
**Script:** `code/99_utilities/correct_gene_identifiers.py`
```bash
python code/99_utilities/correct_gene_identifiers.py
```
**Key Outputs** (e.g., under `output\ect_v3m_drought_full_100\`):
```
ect_oof_r2_by_gene_corrected.csv
ect_cis_mass_by_env_corrected.csv
```

#### 3b — Build the 21-Gene Confirmatory Set
**Script:** `code/04_confirmatory_analysis/01_build_confirmatory_gene_set.py`
```bash
python code/04_confirmatory_analysis/01_build_confirmatory_gene_set.py ^
  --lmm       output\robust_lmm_analysis\tables\robust_lmm_comprehensive_results.csv ^
  --out_list  data\maize\process\lists\gxe_confirmatory_21.csv ^
  --out_details output\final_analysis_reports\gene_modules\gxe_confirmatory_21_details.csv
```
**Key Outputs:**
```
data/maize/process/lists/gxe_confirmatory_21.csv
output/final_analysis_reports/gene_modules/gxe_confirmatory_21_details.csv
```

#### 3c — Analyze the Confirmatory Gene Set
**Script:** `code/04_confirmatory_analysis/02_analyze_confirmatory_set.py`
```bash
python code/04_confirmatory_analysis/02_analyze_confirmatory_set.py ^
  --list    data\maize\process\lists\gxe_confirmatory_21.csv ^
  --ect_r2  output\ect_v3m_drought_full_100\ect_oof_r2_by_gene_corrected.csv ^
  --out_dir output\final_analysis_reports\confirmatory_21
```
**Key Outputs** (created under `output\final_analysis_reports\confirmatory_21\`):
```
confirmatory_21_delta_summary.csv
confirmatory_21_joined_data.csv
figures/ecdf_delta_confirmatory_21.png
```

---

## Final Outputs for Manuscript
The primary results for figures and tables are generated from the following files:
- **Primary LMM Results:** `output/robust_lmm_analysis/tables/robust_lmm_comprehensive_results.csv`
- **LMM Diagnostics:** `output/robust_lmm_analysis/diagnostics/lmm_diagnostics.png`
- **CEES Comparison:** `output/cees_analysis/cees_summary_statistics.csv`
- **Confirmatory Set Validation:** `output/final_analysis_reports/confirmatory_21/confirmatory_21_delta_summary.csv`

---

**End of REPRODUCE_03_lmm**
