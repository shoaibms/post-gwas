# Data Note: Source Data, Preprocessing Pipeline, and Canonical Artifacts
**Version:** 4.0  
**Last Updated:** 2025-11-15  
**Project:** Proximal cis-regulatory variants organise maize drought G×E architecture

---

## 1. Scope and Analysis Cohort

This document catalogues the authoritative source data and processing workflow for the maize drought genotype-by-environment (G×E) study. The preprocessing pipeline transforms raw multi-omics data from 224 accessions (Liu et al., 2020) into quality-controlled, analysis-ready datasets for a **198-accession cohort** with complete genotype, phenotype, and three-environment expression data.

**Analysis Design:**
- **Cohort:** 198 maize inbred lines with matched G×T×E data
- **Conditions:** Well-watered (WW), moderate drought (WS1), severe drought (WS2)
- **Focus:** 500 differentially expressed genes for G×E architecture analysis
- **Genome:** B73 RefGen_v4 (AGPv4; Zm00001d###### identifiers)

Preprocessing details are documented in `REPRODUCE_01_preprocessing.md`.

---

## 2. Authoritative Source Data (External Inputs)

These files are immutable inputs from public repositories required for full reproduction:

### Expression Data (RNA-seq, FPKM)
- `data/maize/WW_209-Uniq_FPKM.txt.gz` — Well-watered condition (209 libraries)
- `data/maize/WS1_208-uniq_FPKM.txt.gz` — Moderate drought stress (208 libraries)
- `data/maize/WS2_210-uniq_FPKM.txt.gz` — Severe drought stress (210 libraries)

**Source:** Liu et al. (2020), *Genome Biology* 21:163. DOI: 10.1186/s13059-020-02069-1  
**Repository:** NGDC GSA (CRA002002) / NCBI SRA (PRJNA637522)

### Genotype Data
- `data/maize/zea_mays_miss0.6_maf0.05.recode.vcf.gz` — SNP genotypes (MAF ≥ 0.05, missingness ≤ 60%)

**Source:** Liu et al. (2020), processed VCF  
**Repository:** NGDC Genome Variation Map (GVM000048)

### Metadata and Phenotypes
- `data/maize/CRA002002.xlsx` — Sample manifest linking libraries to accessions and conditions
- `data/maize/13059_2021_2481_MOESM2_ESM.xlsx` — Supplementary phenotype data

**Sources:** Liu et al. (2020); Zhang et al. (2021), *Genome Biology* 22:260

### Reference Genome Annotation
- `data/maize/Zea_mays.B73_RefGen_v4.gff3.gz` — Gene models, coordinates (AGPv4)

**Source:** Ensembl Plants / MaizeGDB (B73 RefGen_v4 assembly)

---

## 3. Preprocessing Workflow

Scripts are executed sequentially from `<project-root>/code/01_data_preprocessing/`. All paths are relative to project root.

### Step 1: Input Audit
**Script:** `00_audit_inputs/audit_raw_inputs.py`  
**Purpose:** Validate integrity, dimensions, and sample overlap across VCF, FPKM, and metadata  
**Outputs:** `output/inspect/audit_report.md`, `output/inspect/coverage_matrix.csv`

### Step 2: AGPv4 Gene Filtering
**Script:** `01_build_analysis_data/01_filter_expression_to_agpv4.py`  
**Purpose:** Retain only genes with valid AGPv4 identifiers (Zm00001d######)  
**Inputs:** Raw FPKM files, GFF3 annotation  
**Outputs:** `output/data_filtered/WW_209-Uniq_FPKM.agpv4.txt.gz` (and WS1, WS2 equivalents)

### Step 3: Integrated Data Preparation
**Script:** `01_build_analysis_data/02_prepare_model_inputs.py`  
**Purpose:** Integrate expression, genotypes, phenotypes; execute PLINK2 pipeline  
**Key Operations:**
- Subset to 198-accession analysis cohort
- Generate long-format expression table
- LD pruning (r² < 0.2, 50 kb windows, 5-variant step)
- Population structure PCA (10 components)
- Produce gene coordinate map from GFF3

**Outputs:**
- `output/data/T_long.parquet` — Expression (accession × environment × gene_id × fpkm)
- `output/data/P.csv` — Phenotype table (198 accessions)
- `output/data/gene_map.csv` — Gene-to-coordinate map
- `output/geno/cohort_pruned.{pgen,pvar,psam}` — LD-pruned genotypes (PLINK2)
- `output/geno/pcs.eigenvec` — Population structure PCs
- `output/geno/G_traw.traw` — Transposed genotype matrix

### Step 4: Quality Control Gates
**Script 1:** `02_quality_control/01_check_cis_coverage.py`  
**Gate:** ≥80% of genes must have ≥1 SNP within ±1 Mb cis-window  
**Output:** `output/reports/preflight_report.md`

**Script 2:** `02_quality_control/02_verify_agpv4_ids.py`  
**Gate:** All gene IDs conform to `Zm00001d######` format  
**Output:** Console report confirming compliance

---

## 4. Key Parameters and Quality Standards

**Gene ID Format:** AGPv4 standard (Zm00001d######) enforced throughout  
**Cis-Window Definitions:** ±500 kb, ±1 Mb (baseline), ±2 Mb tested for stability  
**Genotype Filters:** MAF ≥ 0.05; LD pruning r² < 0.2  
**Sample Inclusion:** Complete data across WW, WS1, WS2 + genotypes (198 accessions)  
**Expression Filter:** 500 genes with significant differential expression across conditions

---

## 5. Canonical Analysis-Ready Artifacts

All downstream models and manuscript analyses consume data directly from these files:

### Primary Data Tables
| File | Description | Dimensions |
|------|-------------|------------|
| `output/data/T_long.parquet` | Long-format expression | ~29.7M rows (198 acc × 3 env × ~50K genes subset) |
| `output/data/P.csv` | Phenotypes | 198 accessions |
| `output/data/gene_map.csv` | Gene coordinates | ~39K genes |

### Genotype Files (PLINK2 Format)
| File | Description |
|------|-------------|
| `output/geno/cohort_pruned.pgen` | LD-pruned genotype matrix |
| `output/geno/cohort_pruned.pvar` | Variant metadata |
| `output/geno/cohort_pruned.psam` | Sample metadata |
| `output/geno/pcs.eigenvec` | Top 10 PCs for structure correction |
| `output/geno/G_traw.traw` | Transposed SNP matrix |

### QC Reports
| File | Content |
|------|---------|
| `output/inspect/audit_report.md` | Raw data integrity summary |
| `output/inspect/coverage_matrix.csv` | Per-accession modality coverage |
| `output/reports/preflight_report.md` | Cis-coverage gate validation |

---

## 6. Software Environment

**Python:** 3.9+  
**PLINK:** 2.0 (binary must be on PATH or in project root)  
**Required Packages:** Install from project root
```bash
pip install -r requirements.txt
```
Core dependencies: `pandas`, `numpy`, `pyarrow`, `scipy`, `statsmodels`, `scikit-learn`, `openpyxl`, `tqdm`

**Execution Platform:** Scripts tested on Windows (PowerShell); paths adapt to Linux/macOS

---

## 7. Reproducibility Quick-Start

From project root (`C:\Users\ms\Desktop\gwas\` or adapted path):
```powershell
# Step 1: Audit raw inputs
python code/01_data_preprocessing/00_audit_inputs/audit_raw_inputs.py

# Step 2: Filter to AGPv4 genes
python code/01_data_preprocessing/01_build_analysis_data/01_filter_expression_to_agpv4.py

# Step 3: Prepare integrated datasets
python code/01_data_preprocessing/01_build_analysis_data/02_prepare_model_inputs.py `
  --cohort-csv output/cohort/core_all3_env.csv `
  --ww output/data_filtered/WW_209-Uniq_FPKM.agpv4.txt.gz `
  --ws1 output/data_filtered/WS1_208-uniq_FPKM.agpv4.txt.gz `
  --ws2 output/data_filtered/WS2_210-uniq_FPKM.agpv4.txt.gz `
  --gff3 data/maize/Zea_mays.B73_RefGen_v4.gff3.gz `
  --vcf data/maize/zea_mays_miss0.6_maf0.05.recode.vcf.gz

# Step 4: Validate cis-coverage
python code/01_data_preprocessing/02_quality_control/01_check_cis_coverage.py

# Step 5: Verify AGPv4 compliance
python code/01_data_preprocessing/02_quality_control/02_verify_agpv4_ids.py
```

Detailed CLI arguments documented in `REPRODUCE_01_preprocessing.md` and script docstrings.

---

## 8. Data Provenance and Availability

**Checksums:** SHA256 manifest for all source files will be included at publication  
**Code Repository:** GitHub (URL to be assigned upon acceptance)  
**Archived Outputs:** Zenodo (DOI to be assigned upon acceptance)  

All preprocessing scripts, configuration files, and canonical artifacts listed in Section 5 will be publicly deposited to ensure exact reproducibility.

---

## 9. Key Citations

- **Liu, S. et al. (2020).** Mapping regulatory variants controlling gene expression in drought response and tolerance in maize. *Genome Biology* 21:163. https://doi.org/10.1186/s13059-020-02069-1

- **Zhang, F. et al. (2021).** Genomic basis underlying the metabolome-mediated drought adaptation of maize. *Genome Biology* 22:260. https://doi.org/10.1186/s13059-021-02481-1

- **Ensembl Plants:** B73 RefGen_v4 (AGPv4). https://plants.ensembl.org/Zea_mays/

---

**End of Data Note v4.0**