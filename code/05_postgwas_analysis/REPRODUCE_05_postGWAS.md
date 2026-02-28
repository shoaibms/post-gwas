# Reproducibility Guide: 05 – Post‑GWAS Analysis

## Objective
Translate primary associations into mechanistic insights by: defining a stable “platinum modulator” gene set; selecting and characterising influential SNPs; testing functional decoupling (GO); evaluating predictive performance (NRS); and running mechanistic validation (ieQTL + motif disruption).

## Prerequisites
- **Data & models:** canonical outputs from prior stages (see `REPRODUCE_01_preprocessing.md`, `REPRODUCE_02_transformer.md`, `REPRODUCE_03_lmm.md`), including:
  - `output/data/T_long.parquet`, `output/data/P.csv`, `output/data/gene_map.csv`
  - `output/geno/cohort_pruned.pgen/.pvar/.psam`, `output/geno/pcs.eigenvec`
  - ECT outputs as applicable (e.g., `output/ect/runs/<run_id>/ect_oof_r2_by_gene.csv`)
- **Software:** Python 3.9+ and packages in `requirements.txt`.
- **Run location:** project root, e.g. `C:\Users\ms\Desktop\gwas\`.

---

## Workflow execution
Run the following steps **in order** from the project root.

### 1) Platinum set definition and SNP selection
**Scripts:**  
`code/05_postgwas_analysis/01_platinum_set_definition/01_run_window_stability.py`  
`code/05_postgwas_analysis/01_platinum_set_definition/02_select_influential_snps.py`

```bash
python code/05_postgwas_analysis/01_platinum_set_definition/01_run_window_stability.py ^
  --gene-map output\data\gene_map.csv ^
  --ect     output\ect\runs\ect_win1Mb\ect_oof_r2_by_gene.csv ^
  --windows 500,1000,2000 ^
  --out-dir output\postgwas\platinum

python code/05_postgwas_analysis/01_platinum_set_definition/02_select_influential_snps.py ^
  --platinum output\postgwas\platinum\platinum_modulator_set.csv ^
  --pvar     output\geno\cohort_pruned.pvar ^
  --gene-map output\data\gene_map.csv ^
  --out-dir  output\postgwas\platinum
```
**Outputs:** `output/postgwas/platinum/platinum_modulator_set.csv`, `influential_snps.csv`, `window_stability_metrics.csv`

---

### 2) SNP characterisation and enrichment
**Scripts:**  
`code/05_postgwas_analysis/02_snp_characterization/01_generate_matched_background.py`  
`code/05_postgwas_analysis/02_snp_characterization/02_run_consequence_enrichment.py`  
`code/05_postgwas_analysis/02_snp_characterization/03_run_tf_proximity_analysis.py`

```bash
python code/05_postgwas_analysis/02_snp_characterization/01_generate_matched_background.py ^
  --index-snps output\postgwas\platinum\influential_snps.csv ^
  --pvar       output\geno\cohort_pruned.pvar ^
  --ratio      10 ^
  --out-dir    output\postgwas\snp_characterization

python code/05_postgwas_analysis/02_snp_characterization/02_run_consequence_enrichment.py ^
  --index-snps      output\postgwas\snp_characterization\influential_snps.csv ^
  --background-snps output\postgwas\snp_characterization\background_snps.csv ^
  --out-dir         output\postgwas\snp_characterization

# Assumes curated/exported TF BEDs are available (see TF prep in figures or utilities).
python code/05_postgwas_analysis/02_snp_characterization/03_run_tf_proximity_analysis.py ^
  --index-snps output\postgwas\snp_characterization\influential_snps.csv ^
  --tf-bed-dir output\postgwas\tf_proximity\bed_export ^
  --out-dir    output\postgwas\snp_characterization
```
**Outputs:** `output/postgwas/snp_characterization/background_snps.csv`, `consequence_enrichment.csv`, `tf_proximity_summary.csv`

---

### 3) Functional decoupling (GO)
**Scripts:**  
`code/05_postgwas_analysis/03_functional_decoupling/01_download_go_annotations.py`  
`code/05_postgwas_analysis/03_functional_decoupling/02_run_go_enrichment.py`

```bash
python code/05_postgwas_analysis/03_functional_decoupling/01_download_go_annotations.py ^
  --out-dir output\postgwas\go

python code/05_postgwas_analysis/03_functional_decoupling/02_run_go_enrichment.py ^
  --genes   output\postgwas\platinum\platinum_modulator_set.csv ^
  --go-root output\postgwas\go ^
  --out-dir output\postgwas\go
```
**Outputs:** `output/postgwas/go/go_annotations_agpv4.parquet`, `go_enrichment_results.csv`

---

### 4) Predictive modelling (NRS)
**Scripts:**  
`code/05_postgwas_analysis/04_predictive_modeling/01_prepare_nrs_data.py`  
`code/05_postgwas_analysis/04_predictive_modeling/02_run_nrs_prediction.py`

```bash
python code/05_postgwas_analysis/04_predictive_modeling/01_prepare_nrs_data.py ^
  --t-long output\data\T_long.parquet ^
  --out-dir output\postgwas\nrs

python code/05_postgwas_analysis/04_predictive_modeling/02_run_nrs_prediction.py ^
  --in-dir  output\postgwas\nrs ^
  --out-dir output\postgwas\nrs
```
**Outputs:** `output/postgwas/nrs/nrs_prediction_summary.csv` (plus per‑panel CSVs if written)

---

### 5) Mechanistic validation (ieQTL + motif)
**Scripts:**  
`code/05_postgwas_analysis/05_mechanistic_validation/01_run_ieqtl_discovery.py`  
`code/05_postgwas_analysis/05_mechanistic_validation/02_finalize_ieqtl_results.py`  
`code/05_postgwas_analysis/05_mechanistic_validation/03_download_motif_data.py`  
`code/05_postgwas_analysis/05_mechanistic_validation/04_run_motif_disruption.py`  
`code/05_postgwas_analysis/05_mechanistic_validation/05_attach_tf_families.py`  
`code/05_postgwas_analysis/05_mechanistic_validation/06_generate_motif_summary.py`

```bash
python code/05_postgwas_analysis/05_mechanistic_validation/01_run_ieqtl_discovery.py ^
  --t-long output\data\T_long.parquet ^
  --pvar   output\geno\cohort_pruned.pvar ^
  --pcs    output\geno\pcs.eigenvec ^
  --out-dir output\postgwas\ieqtl

python code/05_postgwas_analysis/05_mechanistic_validation/02_finalize_ieqtl_results.py ^
  --in-dir  output\postgwas\ieqtl ^
  --out-dir output\postgwas\ieqtl

python code/05_postgwas_analysis/05_mechanistic_validation/03_download_motif_data.py ^
  --out-dir output\postgwas\motif

python code/05_postgwas_analysis/05_mechanistic_validation/04_run_motif_disruption.py ^
  --index-snps output\postgwas\snp_characterization\influential_snps.csv ^
  --motif-root output\postgwas\motif ^
  --out-dir    output\postgwas\motif

python code/05_postgwas_analysis/05_mechanistic_validation/05_attach_tf_families.py ^
  --delta-pwm  output\postgwas\motif\delta_pwm_results.csv ^
  --jaspar-tsv output\postgwas\motif\jaspar_plants_metadata.tsv ^
  --out-dir    output\postgwas\motif

python code/05_postgwas_analysis/05_mechanistic_validation/06_generate_motif_summary.py ^
  --in-dir  output\postgwas\motif ^
  --out-dir output\postgwas\motif
```
**Outputs:** `output/postgwas/ieqtl/ieqtl_calls.csv`, `output/postgwas/motif/delta_pwm_results.csv`, `tf_family_annotations.csv`, `motif_summary.csv`

---

## Final outputs referenced by the manuscript
- `output/postgwas/platinum/platinum_modulator_set.csv`
- `output/postgwas/platinum/influential_snps.csv`
- `output/postgwas/snp_characterization/consequence_enrichment.csv`
- `output/postgwas/snp_characterization/tf_proximity_summary.csv`
- `output/postgwas/go/go_enrichment_results.csv`
- `output/postgwas/nrs/nrs_prediction_summary.csv`
- `output/postgwas/ieqtl/ieqtl_calls.csv`
- `output/postgwas/motif/motif_summary.csv`

---

## Notes on reproducibility
- Steps are deterministic given fixed inputs; where resampling is used, seeds are exposed as script flags.
- Paths can be overridden with `--out-dir` to reproduce figures in alternate locations.