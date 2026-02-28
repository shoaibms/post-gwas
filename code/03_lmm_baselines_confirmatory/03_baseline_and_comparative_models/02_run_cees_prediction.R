#!/usr/bin/env Rscript
# =============================================================================
# Cross-Environment eQTL Scoring (CEES) Implementation
# =============================================================================
# Purpose: Test environment-conditional regulatory transfer using eQTL betas
#          from one environment to predict expression in another environment
# Author: Environment-Conditional eQTL Analysis Pipeline
# Date: 2025-01-09
# Usage: Rscript cees_implementation.R
# =============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(Matrix)
  library(parallel)
  library(jsonlite)
  library(effsize)
  library(metafor)
})

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
BASE_DIR <- "C:/Users/ms/Desktop/gwas"
OUTPUT_DIR <- file.path(BASE_DIR, "output/cees_analysis")
EQTL_DIR <- file.path(BASE_DIR, "output/baselines_eqtl")
DATA_DIR <- file.path(BASE_DIR, "output/data")
GENO_DIR <- file.path(BASE_DIR, "output/geno")

# Analysis parameters
TOP_K_SNPS <- 5          # Number of top SNPs per gene to use
MIN_SAMPLES <- 10        # Minimum samples required for correlation
N_CORES <- 4             # Number of cores for parallel processing
SEED <- 42               # Random seed for reproducibility

# Create output directory
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Set working directory
setwd(BASE_DIR)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Log message with timestamp
#' @param msg Character string message to log
log_message <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s\n", timestamp, msg))
}

#' Check data structure of eQTL files
#' @param env_name Environment name to check
check_eqtl_structure <- function(env_name) {
  eqtl_file <- file.path(EQTL_DIR, env_name, "cis_eqtls_significant.txt")
  if (file.exists(eqtl_file)) {
    dt <- fread(eqtl_file, nrows = 5)
    log_message(sprintf("Structure of %s eQTL file:", env_name))
    log_message(sprintf("  Columns: %s", paste(colnames(dt), collapse = ", ")))
    log_message(sprintf("  First few rows:"))
    print(head(dt, 2))
  } else {
    log_message(sprintf("  File not found: %s", eqtl_file))
  }
}

#' Load and prepare eQTL results from Matrix eQTL output
#' @param env_name Environment name (WW, WS1, or WS2)
#' @param top_k Number of top SNPs per gene to retain
#' @return data.table with columns: gene, snp, beta, pvalue
load_eqtl_results <- function(env_name, top_k = TOP_K_SNPS, geno_matrix = NULL, ld_threshold = 0.2) {
  log_message(sprintf("Loading eQTL results for environment: %s", env_name))
  
  eqtl_file <- file.path(EQTL_DIR, env_name, "cis_eqtls_significant.txt")
  if (!file.exists(eqtl_file)) {
    stop(sprintf("eQTL file not found: %s", eqtl_file))
  }
  
  dt <- fread(eqtl_file)
  
  # The actual columns from your Matrix eQTL output are: snps, gene, statistic, pvalue, FDR, beta
  # Rename to standardized names for consistency
  if ("snps" %in% colnames(dt)) {
    setnames(dt, "snps", "snp")
  }
  
  # Clean SNP IDs - remove quotes if present
  dt[, snp := gsub("'", "", snp)]
  
  # Perform LD-clumping to select top_k independent SNPs per gene
  if (!is.null(geno_matrix)) {
    log_message(sprintf("  Performing LD-clumping with r2 threshold %.2f", ld_threshold))
    
    # Optimization: Pre-filter to top 50 SNPs per gene before clumping
    dt_candidates <- dt[order(pvalue), .SD[1:min(.N, 50)], by = gene]
    
    # Function to perform clumping for one gene
    clump_gene <- function(gene_snps) {
      available_snps <- intersect(gene_snps, colnames(geno_matrix))
      # Early exit if clumping isn't needed
      if (length(available_snps) <= top_k) return(available_snps)
      
      # Pre-filter for monomorphic SNPs to avoid warnings
      if (length(available_snps) > 1) {
        vars <- apply(geno_matrix[, available_snps, drop=FALSE], 2, sd, na.rm=TRUE)
        available_snps <- names(vars[vars > 1e-6])
      }
      if (length(available_snps) <= 1) return(available_snps)
      
      # Robustly handle potential NAs in genotype data
      ld_matrix <- cor(geno_matrix[, available_snps, drop = FALSE], use = "pairwise.complete.obs")^2
      
      selected_snps <- character(0)
      while (length(selected_snps) < top_k && length(available_snps) > 0) {
        # available_snps is already p-value ordered
        lead_snp <- available_snps[1]
        selected_snps <- c(selected_snps, lead_snp)
        
        # Remove lead SNP and all SNPs in high LD with it
        high_ld_snps <- names(which(ld_matrix[lead_snp, ] > ld_threshold))
        available_snps <- setdiff(available_snps, high_ld_snps)
      }
      return(selected_snps)
    }
    
    # Apply clumping to each gene and join back to preserve all columns
    selected_by_gene <- dt_candidates[, .(selected = list(clump_gene(snp))), by = gene]
    sel <- selected_by_gene[, .(snp = unlist(selected)), by = gene]
    dt_top <- dt[sel, on = .(gene, snp), nomatch = 0L]
    
    # Audit/provenance: per-gene candidate vs selected counts
    cand_counts <- dt_candidates[, .(n_candidates = .N), by = gene]
    sel_counts  <- dt_top[,        .(n_selected   = .N), by = gene]
    per_gene_counts <- cand_counts[sel_counts, on = "gene"]
    fwrite(per_gene_counts, file.path(OUTPUT_DIR, sprintf("ld_clumping_summary_%s.csv", env_name)))
    med_cand <- median(per_gene_counts$n_candidates, na.rm = TRUE)
    med_sel  <- median(per_gene_counts$n_selected,  na.rm = TRUE)
    log_message(sprintf("  LD-clumping stats (%s): median candidates %.1f -> selected %.1f",
                        env_name, med_cand, med_sel))
  } else {
    # Select top k SNPs per gene based on p-value (no clumping)
    dt_top <- dt[order(pvalue), .SD[1:min(.N, top_k)], by = gene]
  }
  
  log_message(sprintf("  Loaded %d eQTLs for %d genes", nrow(dt_top), uniqueN(dt_top$gene)))
  
  return(dt_top[, .(gene, snp, beta, pvalue)])
}

#' Load genotype matrix in dosage format with sample ID mapping
#' @return matrix with samples as rows, SNPs as columns
load_genotype_matrix <- function() {
  log_message("Loading genotype matrix...")
  
  geno_file <- file.path(GENO_DIR, "G_traw.traw")
  if (!file.exists(geno_file)) {
    stop(sprintf("Genotype file not found: %s", geno_file))
  }
  
  # Read traw format (transposed raw genotype format from PLINK)
  geno_dt <- fread(geno_file)
  
  # Extract SNP IDs and sample IDs
  snp_ids <- geno_dt$SNP
  geno_sample_ids <- colnames(geno_dt)[7:ncol(geno_dt)]  # Skip first 6 metadata columns
  
  # CRITICAL FIX: Clean SNP IDs to match the eQTL file processing
  snp_ids <- gsub("'", "", snp_ids)
  
  # Convert to matrix (samples x SNPs)
  geno_matrix <- t(as.matrix(geno_dt[, 7:ncol(geno_dt)]))
  colnames(geno_matrix) <- snp_ids
  
  # CRITICAL: Map genotype sample IDs to expression sample IDs
  # The genotype IDs have prefixes like "0_" that need to be removed
  # Example: "0_05W002" -> "05W002", "0_CML134" -> "CML134"
  
  # Clean sample IDs by removing leading "0_" prefix
  clean_sample_ids <- gsub("^0_", "", geno_sample_ids)
  
  # Some samples might have additional numeric prefixes, let's check
  log_message(sprintf("  Original sample IDs (first 5): %s", 
                      paste(head(geno_sample_ids, 5), collapse = ", ")))
  log_message(sprintf("  Cleaned sample IDs (first 5): %s", 
                      paste(head(clean_sample_ids, 5), collapse = ", ")))
  
  # Try to load the cohort file which should have the correct mapping
  cohort_file <- file.path(BASE_DIR, "output/cohort/core_all3_env.csv")
  if (file.exists(cohort_file)) {
    cohort_df <- fread(cohort_file)
    log_message(sprintf("  Cohort file loaded with %d accessions", nrow(cohort_df)))
    
    # Check if cohort accessions match cleaned IDs
    cohort_accessions <- cohort_df$accession
    
    # Try different matching strategies
    # Strategy 1: Direct match after cleaning
    matches1 <- sum(clean_sample_ids %in% cohort_accessions)
    
    # Strategy 2: Match original IDs
    matches2 <- sum(geno_sample_ids %in% cohort_accessions)
    
    # Strategy 3: More aggressive cleaning (remove all prefixes before first letter)
    ultra_clean_ids <- gsub("^[0-9_]+([A-Z])", "\\1", geno_sample_ids)
    matches3 <- sum(ultra_clean_ids %in% cohort_accessions)
    
    log_message(sprintf("  Matching strategies - Clean: %d, Original: %d, Ultra-clean: %d", 
                        matches1, matches2, matches3))
    
    # Use the best matching strategy
    if (matches3 > matches1 && matches3 > matches2) {
      rownames(geno_matrix) <- ultra_clean_ids
      log_message("  Using ultra-clean IDs (removed numeric prefixes)")
    } else if (matches1 > matches2) {
      rownames(geno_matrix) <- clean_sample_ids
      log_message("  Using cleaned IDs (removed 0_ prefix)")
    } else {
      rownames(geno_matrix) <- geno_sample_ids
      log_message("  Using original IDs")
    }
  } else {
    # Fallback: just clean the 0_ prefix
    rownames(geno_matrix) <- clean_sample_ids
    log_message("  Warning: Cohort file not found, using simple cleaning")
  }
  
  log_message(sprintf("  Loaded genotype matrix: %d samples x %d SNPs", 
                      nrow(geno_matrix), ncol(geno_matrix)))
  
  return(geno_matrix)
}

#' Load expression data for specific environment
#' @param env_name Environment name (WW, WS1, or WS2)
#' @return matrix with samples as rows, genes as columns
load_expression_matrix <- function(env_name) {
  log_message(sprintf("Loading expression data for environment: %s", env_name))
  
  expr_file <- file.path(DATA_DIR, "T_long.parquet")
  if (!file.exists(expr_file)) {
    stop(sprintf("Expression file not found: %s", expr_file))
  }
  
  # Read parquet file
  expr_long <- read_parquet(expr_file)
  
  # Filter for specific environment
  expr_env <- expr_long[expr_long$env == env_name, ]
  
  # Convert to wide format (samples x genes)
  expr_wide <- dcast(as.data.table(expr_env), accession ~ gene_id, value.var = "fpkm")
  
  # Convert to matrix
  sample_ids <- expr_wide$accession
  expr_matrix <- as.matrix(expr_wide[, -1])
  rownames(expr_matrix) <- sample_ids
  
  # Log-transform expression values
  expr_matrix <- log1p(expr_matrix)
  
  log_message(sprintf("  Loaded expression matrix: %d samples x %d genes", 
                      nrow(expr_matrix), ncol(expr_matrix)))
  
  return(expr_matrix)
}

#' Cross-environment prediction function
#' @param eqtl_train eQTL results from training environment
#' @param geno_matrix Genotype matrix (samples x SNPs)
#' @param expr_test Expression matrix from test environment (samples x genes)
#' @param env_train Name of training environment
#' @param env_test Name of test environment
#' @return data.table with prediction results
predict_cross_environment <- function(eqtl_train, geno_matrix, expr_test, 
                                      env_train, env_test) {
  
  log_message(sprintf("Predicting %s -> %s", env_train, env_test))
  
  # Find common samples between genotype and expression data
  common_samples <- intersect(rownames(geno_matrix), rownames(expr_test))
  if (length(common_samples) < MIN_SAMPLES) {
    warning(sprintf("Insufficient common samples: %d", length(common_samples)))
    return(NULL)
  }
  
  # Subset to common samples
  G <- geno_matrix[common_samples, , drop = FALSE]
  Y <- expr_test[common_samples, , drop = FALSE]
  
  # Find genes present in both eQTL results and expression data
  common_genes <- intersect(unique(eqtl_train$gene), colnames(Y))
  log_message(sprintf("  Testing %d genes with eQTLs", length(common_genes)))
  
  # Initialize results list
  results_list <- list()
  
  # Process each gene
  pb <- txtProgressBar(min = 0, max = length(common_genes), style = 3)
  
  for (i in seq_along(common_genes)) {
    g <- common_genes[i]
    setTxtProgressBar(pb, i)
    
    # Get eQTL SNPs and betas for this gene
    gene_eqtls <- eqtl_train[gene == g]
    
    # Find SNPs present in genotype matrix
    available_snps <- intersect(gene_eqtls$snp, colnames(G))
    
    if (length(available_snps) == 0) {
      next
    }
    
    # Extract genotype matrix for these SNPs
    X <- G[, available_snps, drop = FALSE]
    
    # Standardize genotypes using training environment statistics
    # This is critical for proper cross-environment transfer
    mu <- colMeans(X, na.rm = TRUE)
    sigma <- apply(X, 2, sd, na.rm = TRUE)
    sigma[sigma == 0 | is.na(sigma)] <- 1  # Avoid division by zero
    
    Z <- sweep(sweep(X, 2, mu, "-"), 2, sigma, "/")
    Z[is.na(Z)] <- 0  # Mean imputation for missing values
    
    # Get betas in correct order
    beta_vec <- gene_eqtls$beta[match(available_snps, gene_eqtls$snp)]
    
    # Predict expression
    y_pred <- as.numeric(Z %*% beta_vec)
    
    # Get observed expression
    y_obs <- Y[, g]
    
    # Calculate performance metrics (only for non-missing observations)
    valid_idx <- !is.na(y_obs)
    n_valid <- sum(valid_idx)
    
    if (n_valid < MIN_SAMPLES) {
      next
    }
    
    # Calculate R-squared and correlation
    pred_sd <- sd(y_pred[valid_idx], na.rm = TRUE)
    obs_sd <- sd(y_obs[valid_idx], na.rm = TRUE)
    
    if (is.na(pred_sd) || is.na(obs_sd) || pred_sd < 1e-6 || obs_sd < 1e-6) {
      cor_val <- 0
    } else {
      cor_val <- cor(y_pred[valid_idx], y_obs[valid_idx], use = "complete.obs")
    }
    r2 <- cor_val^2
    
    # Calculate RMSE
    rmse <- sqrt(mean((y_pred[valid_idx] - y_obs[valid_idx])^2, na.rm = TRUE))
    
    # Store results
    results_list[[length(results_list) + 1]] <- data.table(
      gene = g,
      env_train = env_train,
      env_test = env_test,
      n_snps = length(available_snps),
      n_samples = n_valid,
      correlation = cor_val,
      r_squared = r2,
      rmse = rmse
    )
  }
  
  close(pb)
  
  # Combine results
  if (length(results_list) > 0) {
    results_dt <- rbindlist(results_list)
    log_message(sprintf("  Completed predictions for %d genes", nrow(results_dt)))
    return(results_dt)
  } else {
    log_message("  No valid predictions generated")
    return(NULL)
  }
}

#' Generate comparison with single-SNP baseline
#' @param eqtl_train eQTL results from training environment
#' @param geno_matrix Genotype matrix
#' @param expr_test Expression matrix from test environment
#' @param env_train Training environment name
#' @param env_test Test environment name
#' @return data.table with single-SNP prediction results
predict_single_snp_baseline <- function(eqtl_train, geno_matrix, expr_test, 
                                        env_train, env_test) {
  
  log_message(sprintf("Computing single-SNP baseline: %s -> %s", env_train, env_test))
  
  # Select only top SNP per gene
  eqtl_top1 <- eqtl_train[order(pvalue), .SD[1], by = gene]
  
  # Run prediction with single SNP
  results <- predict_cross_environment(eqtl_top1, geno_matrix, expr_test, 
                                       env_train, env_test)
  
  if (!is.null(results)) {
    results[, model := "single_snp"]
  }
  
  return(results)
}

#' Check and diagnose sample ID alignment issues
check_sample_alignment <- function() {
  log_message("Diagnosing sample ID alignment between genotype and expression data...")
  
  # Load genotype sample IDs
  geno_file <- file.path(GENO_DIR, "G_traw.traw")
  geno_dt <- fread(geno_file, nrows = 1)
  geno_samples <- colnames(geno_dt)[7:ncol(geno_dt)]
  
  # Load expression sample IDs
  expr_file <- file.path(DATA_DIR, "T_long.parquet")
  expr_df <- read_parquet(expr_file, n_max = 1000)
  expr_samples <- unique(expr_df$accession)
  
  # Load cohort file
  cohort_file <- file.path(BASE_DIR, "output/cohort/core_all3_env.csv")
  cohort_samples <- NULL
  if (file.exists(cohort_file)) {
    cohort_df <- fread(cohort_file)
    cohort_samples <- cohort_df$accession
  }
  
  log_message(sprintf("\nSample counts:"))
  log_message(sprintf("  Genotype samples: %d", length(geno_samples)))
  log_message(sprintf("  Expression samples: %d", length(expr_samples)))
  if (!is.null(cohort_samples)) {
    log_message(sprintf("  Cohort samples: %d", length(cohort_samples)))
  }
  
  # Test different cleaning strategies
  log_message("\nTesting sample ID mapping strategies:")
  
  # Strategy 1: Remove "0_" prefix
  clean1 <- gsub("^0_", "", geno_samples)
  matches1 <- sum(clean1 %in% expr_samples)
  log_message(sprintf("  Strategy 1 (remove '0_'): %d matches", matches1))
  if (matches1 > 0) {
    log_message(sprintf("    Example matches: %s", 
                        paste(head(clean1[clean1 %in% expr_samples], 3), collapse = ", ")))
  }
  
  # Strategy 2: Remove all numeric prefixes
  clean2 <- gsub("^[0-9_]+([A-Z])", "\\1", geno_samples)
  matches2 <- sum(clean2 %in% expr_samples)
  log_message(sprintf("  Strategy 2 (remove numeric prefix): %d matches", matches2))
  if (matches2 > 0) {
    log_message(sprintf("    Example matches: %s", 
                        paste(head(clean2[clean2 %in% expr_samples], 3), collapse = ", ")))
  }
  
  # Strategy 3: Check if expression samples have prefix in genotype
  expr_with_prefix <- paste0("0_", expr_samples)
  matches3 <- sum(geno_samples %in% expr_with_prefix)
  log_message(sprintf("  Strategy 3 (add '0_' to expression): %d matches", matches3))
  
  # Best strategy recommendation
  best_matches <- max(matches1, matches2, matches3)
  log_message(sprintf("\nCONCLUSION: Best alignment strategy finds %d matching samples.", best_matches))
  
  return(best_matches)
}

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

main_analysis <- function() {
  
  set.seed(SEED)
  log_message("Starting Cross-Environment eQTL Scoring (CEES) Analysis")
  log_message(sprintf("Output directory: %s", OUTPUT_DIR))
  
  # Check sample alignment first
  log_message("\nChecking data compatibility...")
  n_matches <- check_sample_alignment()
  
  if (n_matches < 100) {
    log_message("\nERROR: Insufficient sample alignment detected!")
    log_message("Please check the sample ID mapping between genotype and expression data.")
    log_message("You may need to create a sample mapping file.")
    
    if (interactive()) {
      response <- readline(prompt = "\nProceed anyway? (y/n): ")
      if (tolower(response) != "y") {
        log_message("Analysis cancelled due to sample alignment issues")
        return(NULL)
      }
    }
  }
  
  # Load genotype matrix (shared across environments)
  geno_matrix <- load_genotype_matrix()
  
  # Load all eQTL results with LD-clumping
  eqtl_ww  <- load_eqtl_results("WW",  TOP_K_SNPS, geno_matrix)
  eqtl_ws1 <- load_eqtl_results("WS1", TOP_K_SNPS, geno_matrix)
  eqtl_ws2 <- load_eqtl_results("WS2", TOP_K_SNPS, geno_matrix)
  
  # Load expression matrices for each environment
  expr_ww <- load_expression_matrix("WW")
  expr_ws1 <- load_expression_matrix("WS1")
  expr_ws2 <- load_expression_matrix("WS2")
  
  # Define all cross-environment combinations
  combinations <- list(
    list(train = "WW",  test = "WS1", eqtl = eqtl_ww,  expr = expr_ws1),
    list(train = "WW",  test = "WS2", eqtl = eqtl_ww,  expr = expr_ws2),
    list(train = "WS1", test = "WW",  eqtl = eqtl_ws1, expr = expr_ww),
    list(train = "WS1", test = "WS2", eqtl = eqtl_ws1, expr = expr_ws2),
    list(train = "WS2", test = "WW",  eqtl = eqtl_ws2, expr = expr_ww),
    list(train = "WS2", test = "WS1", eqtl = eqtl_ws2, expr = expr_ws1),
    # Add within-environment combinations for robust specificity analysis
    list(train = "WW",  test = "WW",  eqtl = eqtl_ww,  expr = expr_ww),
    list(train = "WS1", test = "WS1", eqtl = eqtl_ws1, expr = expr_ws1),
    list(train = "WS2", test = "WS2", eqtl = eqtl_ws2, expr = expr_ws2)
  )
  
  # Run CEES for all combinations
  log_message(sprintf("\nRunning CEES with top-%d SNPs per gene", TOP_K_SNPS))
  cees_results_list <- list()
  
  for (combo in combinations) {
    result <- predict_cross_environment(
      combo$eqtl, geno_matrix, combo$expr,
      combo$train, combo$test
    )
    
    if (!is.null(result)) {
      result[, model := sprintf("top_%d_snps", TOP_K_SNPS)]
      cees_results_list[[length(cees_results_list) + 1]] <- result
    }
  }
  
  # Combine all CEES results
  cees_results <- rbindlist(cees_results_list)
  
  # Run single-SNP baseline for comparison
  log_message("\nRunning single-SNP baseline analysis")
  baseline_results_list <- list()
  
  for (combo in combinations) {
    result <- predict_single_snp_baseline(
      combo$eqtl, geno_matrix, combo$expr,
      combo$train, combo$test
    )
    
    if (!is.null(result)) {
      baseline_results_list[[length(baseline_results_list) + 1]] <- result
    }
  }
  
  # Combine baseline results
  baseline_results <- rbindlist(baseline_results_list)
  
  # Combine all results
  all_results <- rbind(cees_results, baseline_results, fill = TRUE)
  
  # Calculate summary statistics
  summary_stats <- all_results[, .(
    median_r2 = median(r_squared, na.rm = TRUE),
    mean_r2 = mean(r_squared, na.rm = TRUE),
    sd_r2 = sd(r_squared, na.rm = TRUE),
    q25_r2 = quantile(r_squared, 0.25, na.rm = TRUE),
    q75_r2 = quantile(r_squared, 0.75, na.rm = TRUE),
    n_genes = .N,
    n_genes_r2_gt_0.1 = sum(r_squared > 0.1, na.rm = TRUE),
    n_genes_r2_gt_0.2 = sum(r_squared > 0.2, na.rm = TRUE)
  ), by = .(env_train, env_test, model)]
  
  # Save results
  log_message("\nSaving results...")
  
  # Save detailed results
  fwrite(all_results, file.path(OUTPUT_DIR, "cees_detailed_results.csv"))
  
  # Save summary statistics
  fwrite(summary_stats, file.path(OUTPUT_DIR, "cees_summary_statistics.csv"))
  
  # Identify genes with stable cross-environment predictions
  stable_genes <- all_results[model == sprintf("top_%d_snps", TOP_K_SNPS)][
    , .(mean_r2 = mean(r_squared, na.rm = TRUE),
        sd_r2 = sd(r_squared, na.rm = TRUE),
        n_pairs = .N), 
    by = gene][
      n_pairs >= 4  # Require predictions in at least 4 env pairs
    ][order(-mean_r2)]
  
  fwrite(stable_genes, file.path(OUTPUT_DIR, "stable_genes_across_environments.csv"))
  
  # Identify environment-specific genes
  top_model <- sprintf("top_%d_snps", TOP_K_SNPS)
  
  within <- cees_results[model == top_model & env_train == env_test,
                         .(within_env_r2 = mean(r_squared, na.rm = TRUE)), by = gene]
  
  cross  <- cees_results[model == top_model & env_train != env_test,
                         .(cross_env_r2  = mean(r_squared, na.rm = TRUE)), by = gene]
  
  env_specific <- merge(within, cross, by = "gene", all = TRUE)[
    , delta_r2 := within_env_r2 - cross_env_r2][order(-delta_r2)]
  
  fwrite(env_specific, file.path(OUTPUT_DIR, "environment_specific_genes.csv"))
  
  # Generate analysis report
  report <- list(
    analysis_date = Sys.time(),
    parameters = list(
      top_k_snps = TOP_K_SNPS,
      min_samples = MIN_SAMPLES,
      n_cores = N_CORES
    ),
    data_summary = list(
      n_samples_genotype = nrow(geno_matrix),
      n_snps = ncol(geno_matrix),
      n_samples_expr_ww = nrow(expr_ww),
      n_samples_expr_ws1 = nrow(expr_ws1),
      n_samples_expr_ws2 = nrow(expr_ws2),
      n_genes_tested = uniqueN(all_results$gene)
    ),
    results_summary = list(
      total_predictions = nrow(all_results),
      median_r2_cees = median(cees_results$r_squared, na.rm = TRUE),
      median_r2_baseline = median(baseline_results$r_squared, na.rm = TRUE),
      improvement_over_baseline = median(cees_results[env_train!=env_test]$r_squared) -
        median(baseline_results[env_train!=env_test]$r_squared),
      n_stable_genes = nrow(stable_genes[mean_r2 > 0.1]),
      n_env_specific_genes = nrow(env_specific[!is.na(delta_r2) & delta_r2 > 0.05])
    )
  )
  
  write_json(report, file.path(OUTPUT_DIR, "cees_analysis_report.json"), 
             pretty = TRUE, auto_unbox = TRUE)
  
  # Print summary to console
  log_message(paste(rep("=", 70), collapse = ""))
  log_message("ANALYSIS COMPLETE - SUMMARY")
  log_message(paste(rep("=", 70), collapse = ""))
  
  print(summary_stats)
  
  log_message(sprintf("\nMedian R² (CEES): %.4f", report$results_summary$median_r2_cees))
  log_message(sprintf("Median R² (Single-SNP): %.4f", report$results_summary$median_r2_baseline))
  log_message(sprintf("Improvement: %.4f", report$results_summary$improvement_over_baseline))
  log_message(sprintf("Stable genes (R² > 0.1): %d", report$results_summary$n_stable_genes))
  log_message(sprintf("Environment-specific genes: %d", report$results_summary$n_env_specific_genes))
  
  log_message(sprintf("\nResults saved to: %s", OUTPUT_DIR))
  
  return(list(
    detailed = all_results,
    summary = summary_stats,
    stable_genes = stable_genes,
    env_specific = env_specific
  ))
}

# =============================================================================
# EXECUTE ANALYSIS
# =============================================================================

# Quick diagnostic function to check data compatibility
run_diagnostics <- function() {
  log_message("Running CEES data diagnostics...")
  
  # Check eQTL files
  for (env in c("WW", "WS1", "WS2")) {
    check_eqtl_structure(env)
  }
  
  # Check genotype file
  geno_file <- file.path(GENO_DIR, "G_traw.traw")
  if (file.exists(geno_file)) {
    geno_header <- fread(geno_file, nrows = 1)
    log_message(sprintf("\nGenotype file structure:"))
    log_message(sprintf("  Dimensions: %d columns", ncol(geno_header)))
    log_message(sprintf("  Sample IDs start at column 7: %s", 
                        paste(colnames(geno_header)[7:min(12, ncol(geno_header))], collapse = ", ")))
  }
  
  # Check expression file
  expr_file <- file.path(DATA_DIR, "T_long.parquet")
  if (file.exists(expr_file)) {
    expr_sample <- read_parquet(expr_file, n_max = 100)
    log_message(sprintf("\nExpression file structure:"))
    log_message(sprintf("  Columns: %s", paste(colnames(expr_sample), collapse = ", ")))
    log_message(sprintf("  Environments: %s", paste(unique(expr_sample$env), collapse = ", ")))
    log_message(sprintf("  Sample accessions example: %s", 
                        paste(head(unique(expr_sample$accession), 5), collapse = ", ")))
  }
  
  # Check sample ID alignment
  log_message("\nChecking sample ID alignment...")
  check_sample_alignment()
}

if (!interactive()) {
  # Run main analysis
  results <- main_analysis()
} else {
  log_message("Script loaded in interactive mode.")
  log_message("Run run_diagnostics() to check data structure")
  log_message("Run check_sample_alignment() to diagnose sample ID issues")
  log_message("Run main_analysis() to execute full CEES analysis")
}