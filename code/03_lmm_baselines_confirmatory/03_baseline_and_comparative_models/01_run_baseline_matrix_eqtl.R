#!/usr/bin/env Rscript
# Matrix eQTL Baseline Analysis for Environment-Conditional eQTL Discovery
# Author: Environment-Conditional eQTL Analysis Pipeline
# Purpose: Establish cis-eQTL baselines across WW, WS1, WS2 environments
# Usage: Rscript matrix_eqtl_baseline.R


###############################################################
# run: source("code/baselines/matrix_eqtl_baseline.R")
###############################################################

suppressPackageStartupMessages({
  library(MatrixEQTL)
  library(data.table)
  library(dplyr)
  library(readr)
  library(arrow)
  library(parallel)
  library(tibble)
  library(tidyr)
  library(jsonlite)
})

# --- Configuration parameters ---
# NOTE: These could be made command-line overridable for greater flexibility.
CIS_WINDOW_KB <- 100    # +/- 100kb window for cis analysis
FDR_THRESHOLD <- 0.05
MIN_MAF <- 0.05
N_PCS <- 20             # Number of principal components to use as covariates
N_CORES <- parallel::detectCores() / 2 # Use half the available cores, cap as needed
NORM_METHOD <- "log1p"  # Normalization method: "log1p" or "rankINT"
OUTPUT_BASE <- "output/baselines_eqtl"
DATA_DIR <- "output/data"
GENO_DIR <- "output/geno"

# Create output directories
dir.create(OUTPUT_BASE, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_BASE, "logs"), recursive = TRUE, showWarnings = FALSE)

# --- Helper Functions ---

# Logging function
log_message <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s\n", timestamp, msg))
}

# QQ plot and Genomic Inflation Factor (Lambda) calculation functions
qq_df <- function(p) {
  p <- sort(p[is.finite(p) & p > 0 & p <= 1])
  n <- length(p)
  data.frame(exp = -log10(ppoints(n)), obs = -log10(p))
}

lambda_gc <- function(p) {
  median(stats::qchisq(1 - p, 1), na.rm = TRUE) / 0.456
}

# Harmonize chromosome labels
normalize_chr <- function(x) {
  x <- gsub("^(Chr|chr)", "", x)
  x <- gsub("^0+", "", x)
  x
}

# --- Input Validation ---
validate_inputs <- function() {
  required_files <- c(
    file.path(DATA_DIR, "T_long.parquet"),
    file.path(GENO_DIR, "G_traw.traw"),
    file.path(GENO_DIR, "pcs.eigenvec"),
    file.path(DATA_DIR, "gene_map.csv")
  )
  
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0) {
    stop("Execution halted. Missing required input files: ", paste(missing_files, collapse = ", "))
  }
  log_message("All required input files have been validated.")
}

# --- Pipeline Execution Start ---
log_message("Starting Matrix eQTL baseline analysis")
validate_inputs() # Validate inputs early

# --- Data Preparation Functions ---

# Function to load and prepare expression data for specific environment
prepare_expression_data <- function(env_name) {
  log_message(sprintf("Loading and normalizing expression data for %s using '%s'", env_name, NORM_METHOD))
  
  expr_long <- read_parquet(file.path(DATA_DIR, "T_long.parquet"))
  
  expr_env <- expr_long %>%
    filter(env == env_name) %>%
    select(accession, gene_id, fpkm) %>%
    pivot_wider(names_from = accession, values_from = fpkm) %>%
    column_to_rownames("gene_id")
  
  expr_matrix <- as.matrix(expr_env)
  expr_matrix[is.na(expr_matrix)] <- 0
  
  # Apply chosen normalization method
  if (NORM_METHOD == "log1p") {
    expr_matrix <- log1p(expr_matrix)
  } else if (NORM_METHOD == "rankINT") {
    expr_matrix <- t(apply(expr_matrix, 1, function(x) {
      r <- rank(x, na.last = "keep", ties.method = "average")
      # Subtract 0.5 for Blom transformation
      qnorm((r - 0.5) / sum(!is.na(r)))
    }))
  }
  
  gene_vars <- apply(expr_matrix, 1, var, na.rm = TRUE)
  expr_matrix <- expr_matrix[gene_vars > 0, ]
  
  log_message(sprintf("Expression matrix for %s: %d genes x %d samples", 
                      env_name, nrow(expr_matrix), ncol(expr_matrix)))
  
  return(expr_matrix)
}

# Function to prepare genotype data
prepare_genotype_data <- function(expr_samples) {
  log_message("Loading and preparing genotype data")
  
  geno_file <- file.path(GENO_DIR, "G_traw.traw")
  geno_data <- fread(geno_file)
  
  meta_candidates <- c("CHR","SNP","(C)M","CM","POS","COUNTED","ALT")
  meta_cols <- intersect(names(geno_data), meta_candidates)
  sample_cols <- setdiff(names(geno_data), meta_cols)
  
  clean_sample_names <- gsub("^0_", "", sample_cols)
  sample_mapping <- setNames(sample_cols, clean_sample_names)
  
  available_samples <- intersect(clean_sample_names, expr_samples)
  
  geno_subset <- geno_data[, c(meta_cols, sample_mapping[available_samples]), with = FALSE]
  
  old_names <- sample_mapping[available_samples]
  setnames(geno_subset, old_names, available_samples)
  
  snp_pos <- geno_subset[, .(snp_id = SNP, chr = normalize_chr(CHR), pos = POS)]
  
  # Filter SNPs to chromosomes 1-10 for symmetry with gene filtering
  snp_pos[, chr_num := suppressWarnings(as.integer(chr))]
  keep_chr <- snp_pos$chr_num %in% 1:10 & !is.na(snp_pos$chr_num)
  
  geno_matrix <- as.matrix(geno_subset[, available_samples, with = FALSE])
  rownames(geno_matrix) <- geno_subset$SNP
  
  geno_matrix <- geno_matrix[keep_chr, , drop = FALSE]
  snp_pos     <- snp_pos[keep_chr, .(snp_id, chr = as.character(chr_num), pos)]
  
  mafs <- apply(geno_matrix, 1, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(0)
    p <- mean(x) / 2
    return(min(p, 1 - p))
  })
  
  keep_snps <- mafs >= MIN_MAF
  geno_matrix <- geno_matrix[keep_snps, ]
  snp_pos <- snp_pos[keep_snps, ]
  
  log_message(sprintf("Genotype matrix: %d SNPs x %d samples (MAF >= %.3f)", 
                      nrow(geno_matrix), ncol(geno_matrix), MIN_MAF))
  
  return(list(matrix = geno_matrix, positions = snp_pos))
}

# Function to prepare covariates (PCs) - More robust version
prepare_covariates <- function(expr_samples) {
  log_message("Loading principal components as covariates")
  pc_file <- file.path(GENO_DIR, "pcs.eigenvec")
  pcs <- data.table::fread(pc_file)
  
  id_col <- intersect(c("#IID", "IID", "IID2", "IID1"), names(pcs))[1]
  stopifnot(!is.na(id_col))
  
  pcs[, clean_id := gsub("^0_", "", get(id_col))]
  
  pc_cols <- grep("^PC\\d+$", names(pcs), value = TRUE)
  pc_cols <- pc_cols[seq_len(min(length(pc_cols), N_PCS))]
  
  if (length(pc_cols) == 0) {
    log_message("No principal components found or used.")
    return(matrix(numeric(0), nrow = 0, ncol = length(expr_samples), dimnames = list(NULL, expr_samples)))
  }
  
  pcs_subset <- pcs[clean_id %in% expr_samples, c("clean_id", pc_cols), with = FALSE]
  
  cov_matrix <- as.matrix(pcs_subset[, ..pc_cols])
  rownames(cov_matrix) <- pcs_subset$clean_id
  cov_matrix <- t(cov_matrix)
  
  # Defensively drop zero-variance PCs
  if (nrow(cov_matrix) > 0) {
    keep <- apply(cov_matrix, 1, sd, na.rm = TRUE) > 0
    cov_matrix <- cov_matrix[keep, , drop = FALSE]
  }
  
  log_message(sprintf("Covariates matrix: %d PCs x %d samples", nrow(cov_matrix), ncol(cov_matrix)))
  return(cov_matrix)
}

# Function to create gene position lookup for cis analysis
prepare_gene_positions <- function() {
  log_message("Loading gene positions for cis analysis")
  
  gene_map <- fread(file.path(DATA_DIR, "gene_map.csv"))
  
  gene_pos <- gene_map[, .(
    geneid = gene_id,
    chr = normalize_chr(chr),
    left = start,
    right = end
  )]
  
  gene_pos[, chr_num := suppressWarnings(as.integer(chr))]
  gene_pos <- gene_pos[chr_num %in% 1:10 & !is.na(chr_num)]
  
  log_message(sprintf("Gene positions: %d genes on chromosomes 1-10", nrow(gene_pos)))
  
  return(gene_pos)
}

# --- Main Analysis Function ---

run_matrix_eqtl_analysis <- function(env_name) {
  log_message(sprintf("Starting Matrix eQTL analysis for %s environment", env_name))
  output_dir <- file.path(OUTPUT_BASE, env_name)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  expr_matrix <- prepare_expression_data(env_name)
  expr_samples <- colnames(expr_matrix)
  
  geno_data <- prepare_genotype_data(expr_samples)
  geno_matrix <- geno_data$matrix
  snp_pos <- geno_data$positions
  
  write_json(list(env = env_name,
                  n_snps = nrow(geno_matrix),
                  maf_threshold = MIN_MAF),
             file.path(output_dir, "snp_filter_summary.json"),
             auto_unbox = TRUE, pretty = TRUE)
  
  common_samples <- intersect(expr_samples, colnames(geno_matrix))
  expr_matrix <- expr_matrix[, common_samples]
  geno_matrix <- geno_matrix[, common_samples]
  
  log_message(sprintf("Analysis will use %d common samples", length(common_samples)))
  writeLines(common_samples, file.path(output_dir, "samples_used.txt"))
  
  cov_matrix <- prepare_covariates(common_samples)
  cov_matrix <- cov_matrix[, common_samples, drop = FALSE]
  
  gene_pos <- prepare_gene_positions()
  
  before_filter <- nrow(expr_matrix)
  expr_genes <- intersect(rownames(expr_matrix), gene_pos$geneid)
  log_message(sprintf("Genes with coordinates on 1..10: %d / %d", length(expr_genes), before_filter))
  expr_matrix <- expr_matrix[expr_genes, ]
  gene_pos <- gene_pos[geneid %in% expr_genes]
  
  write_json(list(env = env_name, genes_before_coord_filter = before_filter, genes_after_coord_filter = length(expr_genes)),
             file.path(output_dir, "gene_filter_summary.json"), auto_unbox = TRUE, pretty = TRUE)
  
  snps <- SlicedData$new(); snps$CreateFromMatrix(geno_matrix)
  gene <- SlicedData$new(); gene$CreateFromMatrix(expr_matrix)
  cvrt <- SlicedData$new(); if (nrow(cov_matrix) > 0) cvrt$CreateFromMatrix(cov_matrix)
  
  snp_pos_df <- data.frame(snp = snp_pos$snp_id, chr = snp_pos$chr, pos = snp_pos$pos)
  gene_pos_df <- data.frame(geneid = gene_pos$geneid, chr = gene_pos$chr, left = gene_pos$left, right = gene_pos$right)
  
  output_file_cis <- tempfile()
  
  me_result <- Matrix_eQTL_main(
    snps = snps, gene = gene, cvrt = cvrt, output_file_name = "", output_file_name.cis = output_file_cis,
    pvOutputThreshold = 0, pvOutputThreshold.cis = 1, snpspos = snp_pos_df, genepos = gene_pos_df,
    cisDist = CIS_WINDOW_KB * 1000, useModel = modelLINEAR, verbose = TRUE, pvalue.hist = "qqplot"
  )
  
  if (file.exists(output_file_cis)) file.remove(output_file_cis)
  
  if (me_result$cis$neqtls > 0) {
    cis_results <- as.data.table(me_result$cis$eqtls)
    fwrite(cis_results, file.path(output_dir, "cis_eqtls_all.txt.gz"), sep = "\t")
    
    cis_results[, FDR := p.adjust(pvalue, "fdr")]
    sig_results <- cis_results[FDR < FDR_THRESHOLD]
    write_tsv(sig_results, file.path(output_dir, "cis_eqtls_significant.txt"))
    
    # Always save the eGene list, even if empty, to prevent downstream script failures.
    fwrite(unique(sig_results[, .(gene)]), file.path(output_dir, "egene_list.txt"), col.names = FALSE)
    
    if (nrow(sig_results) > 0) {
      # This block is for actions that only make sense when there are significant results.
    }
    
    # Clamp p-values to an open interval (0, 1) to keep QQ/lambda finite
    all_p <- pmin(pmax(cis_results$pvalue, 1e-300), 1 - 1e-16)
    lam <- lambda_gc(all_p)
    qq <- qq_df(all_p)
    
    png(file.path(output_dir, "qq.png"), width = 480, height = 480)
    plot(qq$exp, qq$obs, pch = 16, cex = 0.5, main = paste0("QQ ", env_name, " (λ=", round(lam, 2), ")"),
         xlab = "Expected -log10(p)", ylab = "Observed -log10(p)"); abline(0, 1, col = "gray"); dev.off()
    
    summary_stats <- list(
      environment = env_name,
      n_samples = length(common_samples),
      n_genes_tested = nrow(expr_matrix),
      n_snps_tested = nrow(geno_matrix),
      total_tests = me_result$cis$ntests,
      significant_eqtls = nrow(sig_results),
      genes_with_eqtl = uniqueN(sig_results$gene),
      lambda_gc = lam,
      analysis_date = Sys.time()
    )
    
    write_json(summary_stats, file.path(output_dir, "analysis_summary.json"), pretty = TRUE, auto_unbox = TRUE)
    log_message(sprintf("Analysis complete for %s: %d significant eQTLs (FDR < %.3f)", env_name, nrow(sig_results), FDR_THRESHOLD))
    return(summary_stats)
  } else {
    log_message(sprintf("No eQTLs found for %s environment", env_name))
    return(NULL)
  }
}

# --- Environment Loop & Final Comparison ---
# NOTE on BLAS/LAPACK: For maximum performance on multi-core systems, ensure R
# is linked against an optimized BLAS/LAPACK library (e.g., OpenBLAS, Intel MKL).
# This can dramatically speed up the matrix operations within MatrixEQTL.

environments <- c("WW", "WS1", "WS2")
# Use single core on Windows where forking is unsupported
used_cores <- if (.Platform$OS.type == "windows") 1 else max(1, as.integer(N_CORES))
log_message(sprintf("Starting parallel analysis for %d environments using %d cores", length(environments), used_cores))

results_list <- mclapply(environments, function(env) {
  tryCatch({
    run_matrix_eqtl_analysis(env)
  }, error = function(e) {
    log_message(sprintf("Error in %s analysis: %s", env, e$message))
    # Return NULL for this environment on error
    return(NULL)
  })
}, mc.cores = used_cores)

# The results_list is a list; convert it to a named list for consistency
results_summary <- setNames(results_list, environments)
# Filter out any environments that failed
results_summary <- results_summary[!sapply(results_summary, is.null)]


log_message("Generating cross-environment comparison")
all_sig_results_list <- lapply(names(results_summary), function(env) {
  sig_file <- file.path(OUTPUT_BASE, env, "cis_eqtls_significant.txt")
  if (file.exists(sig_file) && file.info(sig_file)$size > 0) {
    read_tsv(sig_file, show_col_types = FALSE) %>% mutate(environment = env)
  }
})

# Harden NULL handling by explicitly filtering out NULLs before binding rows
all_sig_results_list <- Filter(Negate(is.null), all_sig_results_list)
combined_results <- bind_rows(all_sig_results_list)

# Remove any rows with non-finite effect sizes or p-values
combined_results <- combined_results %>%
  filter(is.finite(beta), is.finite(pvalue))

if (nrow(combined_results) > 0) {
  combined_results$pair <- paste(combined_results$snps, combined_results$gene, sep=":")
  
  sig_tbl <- combined_results %>%
    group_by(pair) %>%
    summarise(n_env = n_distinct(environment), sign_concordant = (n_distinct(sign(beta)) == 1), .groups = "drop")
  
  combined_results <- left_join(combined_results, sig_tbl, by = "pair") %>%
    mutate(eqtl_type = case_when(
      n_env == 1 ~ "Environment-specific",
      n_env >= 2 & sign_concordant ~ "Shared-concordant",
      TRUE ~ "Shared-discordant"
    ))
  
  write_tsv(combined_results, file.path(OUTPUT_BASE, "combined_significant_eqtls.txt"))
  
  comparison_summary <- combined_results %>%
    group_by(environment, eqtl_type) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = eqtl_type, values_from = count, values_fill = 0)
  
  fwrite(comparison_summary, file.path(OUTPUT_BASE, "environment_comparison.tsv"), sep = "\t")
  log_message("Cross-environment analysis complete")
} else {
  log_message("No significant eQTLs found across any environment to compare.")
}

# --- Final Metadata ---
analysis_metadata <- list(
  analysis_type = "Matrix eQTL baseline", cis_window_kb = CIS_WINDOW_KB, fdr_threshold = FDR_THRESHOLD,
  min_maf = MIN_MAF, n_pcs = N_PCS, norm_method = NORM_METHOD, environments_analyzed = names(results_summary),
  results_summary = results_summary, analysis_completed = Sys.time()
)
write_json(analysis_metadata, file.path(OUTPUT_BASE, "analysis_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

readme_content <- paste(
  "Matrix eQTL Baseline Analysis", "============================", paste("Run Date:", Sys.time()),
  "Parameters:", paste("  - NORM_METHOD:", NORM_METHOD), paste("  - N_PCS:", N_PCS), 
  paste("  - FDR_THRESHOLD:", FDR_THRESHOLD), paste("  - MIN_MAF:", MIN_MAF),
  paste("Output generated in:", normalizePath(OUTPUT_BASE)), sep = "\n"
)
writeLines(readme_content, file.path(OUTPUT_BASE, "analysis_readme.txt"))

# Append session info for reproducibility
sink(file.path(OUTPUT_BASE, "analysis_readme.txt"), append = TRUE)
cat("\n\nSessionInfo:\n")
print(sessionInfo())
sink()

log_message("Matrix eQTL baseline analysis pipeline completed")
log_message(sprintf("Results saved to: %s", OUTPUT_BASE))