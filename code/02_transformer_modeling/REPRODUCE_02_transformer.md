# Reproducibility Guide: 02 – Transformer Modeling

## Objective
Train the Environment‑Conditional Transformer (ECT) to dissect genotype‑by‑environment (G×E) signal from the preprocessed cohort data. This stage builds a compact `.pt` bundle from canonical inputs and runs cross‑validated training to produce model‑ready diagnostics.

## Prerequisites
- **Data:** Outputs from Stage 01 (see `REPRODUCE_01_preprocessing.md`):
  - `output/data/T_long.parquet`
  - `output/data/P.csv`
  - `output/data/gene_map.csv`
  - `output/geno/cohort_pruned.pgen/.pvar/.psam`
  - `output/geno/pcs.eigenvec`
- **Software:** Python 3.9+. PyTorch (CPU or CUDA build appropriate for your GPU/OS) and packages listed in `requirements.txt`.
- **Python Dependencies:** From the project root (`C:\Users\ms\Desktop\gwas\`):
  ```bash
  pip install -r requirements.txt
  ```

> Note: Install PyTorch following the official instructions for your platform (Windows/RTX‑2060 users should select the correct CUDA/CPU wheel from pytorch.org).

---

## Workflow Execution
Run all commands from the **project root**: `C:\Users\ms\Desktop\gwas\`.

### Step 1 — Build the ECT data bundle
This step assembles model inputs (genotypes, expression, environment labels, PCs, gene–SNP window mask) into a single `.pt` file with a manifest for provenance.

**Script:** `code/02_transformer_modeling/01_bundle/build_transformer_bundle.py`
```bash
python code/02_transformer_modeling/01_bundle/build_transformer_bundle.py ^
  --output-file output\ect\bundles\transformer_data_win1Mb.pt ^
  --cis-window-kb 1000
```
**Key Outputs (created under `output\ect\bundles\`):**
- `transformer_data_win1Mb.pt`
- `preparation_manifest.json`
- `environment_label_map.json`

---

### Step 2 — Train the Environment‑Conditional Transformer (ECT)
Run cross‑validated training with fixed seeds and record summary diagnostics. The minimal invocation uses the bundle from Step 1; optional flags can export attention maps or fold sentinels.

**Script:** `code/02_transformer_modeling/02_train_ect/train_env_conditional_transformer.py`
```bash
python code/02_transformer_modeling/02_train_ect/train_env_conditional_transformer.py ^
  --data-file output\ect\bundles\transformer_data_win1Mb.pt ^
  --out-dir  output\ect\runs\ect_win1Mb ^
  --kfolds   5 ^
  --n-pcs    20 ^
  --seed     42 ^
  --save-attention ^
  --dump-fold-sentinels
```
**Key Outputs (created under `output\ect\runs\ect_win1Mb\`):**
- `ect_oof_r2_by_gene.csv` — cross‑validated out‑of‑fold R² by gene
- `ect_cis_mass_joined.csv` — cis mass metrics joined across environments
- `ect_summary.json` — run configuration and headline metrics
- `correlation_summary.csv` — model vs baseline comparison (if enabled)
- `ect_alpha_by_gene.csv` — per‑gene attention summaries (when `--save-attention` is used)
- `trials.csv`, `trial_config.json`, `best_trial.json` — tuning/selection metadata (when tuning is enabled)
- `checkpoints\*.pt` — optional saved checkpoints (if configured)

---

## Final Outputs for Downstream Analyses
The following artifacts are consumed by the downstream statistics and figures:
- `output/ect/runs/ect_win1Mb/ect_oof_r2_by_gene.csv`
- `output/ect/runs/ect_win1Mb/ect_cis_mass_joined.csv`
- `output/ect/runs/ect_win1Mb/ect_summary.json`

(Additional files such as attention exports, correlation summaries, and checkpoints are optional and not required by the manuscript unless specified in figure methods.)

---

## Notes on Reproducibility
- Set `--seed` and `--kfolds` explicitly to reproduce splits. When `--dump-fold-sentinels` is used, fold/seed markers are written alongside run outputs.
- Window size (`--cis-window-kb`) should mirror the setting used for preprocessing QC (default ±1,000 kb).
- GPU acceleration is recommended but not required; the scripts run on CPU with longer wall‑time.