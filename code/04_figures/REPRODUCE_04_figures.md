# Reproducibility Guide: 04 – Figures

## Objective
Generate all main and supplementary figures from the processed datasets and model outputs. This stage relies on the canonical outputs from Stages 01–03 and renders publication-quality panels in both PNG and PDF formats.

## Prerequisites
- **Data & Models:** Outputs from prior stages (see `REPRODUCE_01_preprocessing.md`, `REPRODUCE_02_transformer.md`, `REPRODUCE_03_lmm.md`), including:
  - `output/data/T_long.parquet`, `output/data/P.csv`, `output/data/gene_map.csv`
  - `output/geno/cohort_pruned.pgen/.pvar/.psam`, `output/geno/pcs.eigenvec`
  - `output/ect/runs/<run_id>/ect_oof_r2_by_gene.csv` and related diagnostics (if required by a figure)
  - LMM outputs as defined in `REPRODUCE_03_lmm.md` (if required by a figure)
- **Software:** Python 3.9+ and packages in `requirements.txt`.
- **Run location:** project root, e.g. `C:\Users\ms\Desktop\gwas\`.

> Note: Figure scripts import local helpers under `code/04_figures/infrastructure/` (e.g., `colour_config.py`, `stat_utils.py`). `figure_02.py` additionally uses `data_loader_gwas.py` for curated inputs.

---

## Workflow Execution
All commands below are intended to be run from the project root.

### Main figures
**Figure 1**
```bash
python code/04_figures/main/figure_01.py
```
**Figure 2**
```bash
python code/04_figures/main/figure_02.py
```
**Figure 3**
```bash
python code/04_figures/main/figure_03.py
```
**Figure 4**
```bash
python code/04_figures/main/figure_04.py
```

### Supplementary figures
```bash
python code/04_figures/supplementary/figure_s1.py
python code/04_figures/supplementary/figure_s2.py
python code/04_figures/supplementary/figure_s3.py
python code/04_figures/supplementary/figure_s4.py
python code/04_figures/supplementary/figure_s5.py
python code/04_figures/supplementary/figure_s6.py
```

### Outputs
Each script saves bitmap and vector versions to:
- `output/figures/main/` for Figures 1–4
- `output/figures/supplementary/` for Figures S1–S6

Filenames are created by the scripts (e.g., `fig01.png`, `fig01.pdf`); if an `--out-dir` argument is supported, it overrides the default.

---

## Notes on Reproducibility
- Scripts are pure-Python and deterministic given fixed inputs; no random seeds are used unless stated in the script header.
- Colour palettes and fonts are centralized in `code/04_figures/infrastructure/colour_config.py` to ensure consistent styling.
- If paths differ from defaults, edit the small path block at the top of each figure script (or pass an `--out-dir` if available).