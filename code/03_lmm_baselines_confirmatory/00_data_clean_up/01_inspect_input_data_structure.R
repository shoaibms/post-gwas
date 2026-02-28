#!/usr/bin/env Rscript
# Data Structure Inspector for Ridge/Elastic Net Baselines
# Purpose: Inspect all input files used by ridge_elastic_baselines2.R
# Output: Concise summary of actual data structure and headers
# Usage: Rscript data_structure_inspector.R

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(jsonlite)
})

# File paths (adjust if needed)
PROJECT_ROOT <- getwd()
DATA_BASE <- file.path(PROJECT_ROOT, "output", "data")
GENO_BASE <- file.path(PROJECT_ROOT, "output", "geno")

# Expected input files for ridge_elastic_baselines2.R
files_to_check <- list(
  expr_parquet = file.path(DATA_BASE, "T_long.parquet"),
  gene_map_csv = file.path(DATA_BASE, "gene_map.csv"),
  pvar_file = file.path(GENO_BASE, "cohort_pruned.pvar"),
  traw_file = file.path(GENO_BASE, "G_traw.traw"),
  pcs_file = file.path(GENO_BASE, "pcs.eigenvec")
)

output_file <- file.path(PROJECT_ROOT, "data_structure_summary.txt")

# Initialize output
output_lines <- character()
add_line <- function(text) {
  output_lines <<- c(output_lines, text)
  cat(text, "\n")
}

add_line("=============================================================================")
add_line("DATA STRUCTURE INSPECTION FOR RIDGE_ELASTIC_BASELINES2.R")
add_line("=============================================================================")
add_line(sprintf("Inspection Date: %s", Sys.time()))
add_line(sprintf("Project Root: %s", PROJECT_ROOT))
add_line("")

# Function to inspect file structure
inspect_file <- function(file_path, file_type) {
  add_line(sprintf(">>> FILE: %s", basename(file_path)))
  add_line(sprintf("Path: %s", file_path))
  
  if (!file.exists(file_path)) {
    add_line("STATUS: FILE NOT FOUND")
    add_line("")
    return()
  }
  
  file_info <- file.info(file_path)
  add_line(sprintf("Size: %.2f MB", file_info$size / 1024^2))
  add_line("STATUS: FILE EXISTS")
  
  tryCatch({
    if (file_type == "parquet") {
      # Inspect parquet file
      df <- read_parquet(file_path)
      add_line(sprintf("Dimensions: %d rows x %d columns", nrow(df), ncol(df)))
      add_line(sprintf("Columns: %s", paste(names(df), collapse = ", ")))
      add_line("Column Types:")
      for (col in names(df)) {
        add_line(sprintf("  %s: %s", col, class(df[[col]])[1]))
      }
      
      add_line("Sample Data (first 3 rows):")
      sample_data <- head(df, 3)
      for (i in 1:min(3, nrow(sample_data))) {
        row_data <- sapply(sample_data[i, ], function(x) {
          if (is.numeric(x)) sprintf("%.3f", x)
          else as.character(x)
        })
        add_line(sprintf("  Row %d: %s", i, paste(row_data, collapse = " | ")))
      }
      
      # Check unique values for categorical columns
      if ("env" %in% names(df)) {
        envs <- unique(df$env)
        add_line(sprintf("Unique environments: %s", paste(envs, collapse = ", ")))
      }
      
    } else if (file_type == "csv") {
      # Inspect CSV file
      df <- fread(file_path, nrows = 5)
      add_line(sprintf("Columns: %s", paste(names(df), collapse = ", ")))
      add_line("Column Types:")
      for (col in names(df)) {
        add_line(sprintf("  %s: %s", col, class(df[[col]])[1]))
      }
      
      add_line("Sample Data (first 3 rows):")
      for (i in 1:min(3, nrow(df))) {
        row_data <- sapply(df[i, ], function(x) {
          if (is.numeric(x)) sprintf("%.3f", x)
          else as.character(x)
        })
        add_line(sprintf("  Row %d: %s", i, paste(row_data, collapse = " | ")))
      }
      
    } else if (file_type == "pvar") {
      # Inspect PLINK pvar file
      df <- fread(file_path, nrows = 5)
      add_line(sprintf("Columns: %s", paste(names(df), collapse = ", ")))
      add_line("Column Types:")
      for (col in names(df)) {
        add_line(sprintf("  %s: %s", col, class(df[[col]])[1]))
      }
      
      add_line("Sample Data (first 3 rows):")
      for (i in 1:min(3, nrow(df))) {
        row_data <- sapply(df[i, ], function(x) as.character(x))
        add_line(sprintf("  Row %d: %s", i, paste(row_data, collapse = " | ")))
      }
      
      # Check chromosome format
      if ("#CHROM" %in% names(df)) {
        chr_col <- "#CHROM"
      } else if ("CHROM" %in% names(df)) {
        chr_col <- "CHROM"
      } else if ("CHR" %in% names(df)) {
        chr_col <- "CHR"
      } else {
        chr_col <- NULL
      }
      
      if (!is.null(chr_col)) {
        full_df <- fread(file_path)
        unique_chrs <- unique(full_df[[chr_col]])
        add_line(sprintf("Unique chromosomes: %s", paste(head(unique_chrs, 10), collapse = ", ")))
        add_line(sprintf("Total SNPs: %d", nrow(full_df)))
      }
      
    } else if (file_type == "traw") {
      # Inspect TRAW file (large, so just header + sample info)
      df <- fread(file_path, nrows = 3)
      add_line(sprintf("Columns: %s", paste(names(df), collapse = ", ")))
      
      # Identify metadata vs sample columns
      meta_cols <- c("CHR", "SNP", "CM", "POS", "COUNTED", "ALT", "(C)M")
      sample_cols <- setdiff(names(df), meta_cols)
      
      add_line(sprintf("Metadata columns: %s", paste(intersect(names(df), meta_cols), collapse = ", ")))
      add_line(sprintf("Sample columns (first 10): %s", paste(head(sample_cols, 10), collapse = ", ")))
      add_line(sprintf("Total sample columns: %d", length(sample_cols)))
      
      # Check sample naming pattern
      if (length(sample_cols) > 0) {
        sample_pattern <- ifelse(grepl("^0_", sample_cols[1]), "Prefixed with '0_'", "No prefix")
        add_line(sprintf("Sample naming pattern: %s", sample_pattern))
      }
      
      # Check genotype values
      if (length(sample_cols) > 0) {
        sample_values <- df[[sample_cols[1]]]
        unique_geno <- unique(sample_values[!is.na(sample_values)])
        add_line(sprintf("Genotype values in %s: %s", sample_cols[1], paste(unique_geno, collapse = ", ")))
      }
      
    } else if (file_type == "eigenvec") {
      # Inspect PCA eigenvector file
      df <- fread(file_path, nrows = 5)
      add_line(sprintf("Columns: %s", paste(names(df), collapse = ", ")))
      add_line(sprintf("Total columns: %d", ncol(df)))
      
      # Infer structure
      if (ncol(df) >= 3) {
        pc_cols <- ncol(df) - 2  # Assuming FID, IID, then PCs
        add_line(sprintf("Inferred structure: FID, IID, %d PC columns", pc_cols))
      }
      
      add_line("Sample Data (first 3 rows):")
      for (i in 1:min(3, nrow(df))) {
        row_data <- sapply(df[i, ], function(x) {
          if (is.numeric(x)) sprintf("%.4f", x)
          else as.character(x)
        })
        add_line(sprintf("  Row %d: %s", i, paste(row_data, collapse = " | ")))
      }
      
      # Check sample naming pattern
      if (ncol(df) >= 2) {
        iid_col <- names(df)[2]
        sample_values <- df[[iid_col]]
        sample_pattern <- ifelse(any(grepl("^0_", sample_values)), "Some prefixed with '0_'", "No prefix pattern")
        add_line(sprintf("Sample ID pattern: %s", sample_pattern))
      }
    }
    
  }, error = function(e) {
    add_line(sprintf("ERROR reading file: %s", e$message))
  })
  
  add_line("")
}

# Inspect each file
for (file_name in names(files_to_check)) {
  file_path <- files_to_check[[file_name]]
  
  # Determine file type from extension
  if (grepl("\\.parquet$", file_path)) {
    file_type <- "parquet"
  } else if (grepl("\\.csv$", file_path)) {
    file_type <- "csv"
  } else if (grepl("\\.pvar$", file_path)) {
    file_type <- "pvar"
  } else if (grepl("\\.traw$", file_path)) {
    file_type <- "traw"
  } else if (grepl("\\.eigenvec$", file_path)) {
    file_type <- "eigenvec"
  } else {
    file_type <- "unknown"
  }
  
  inspect_file(file_path, file_type)
}

# Additional cross-file compatibility checks
add_line("=============================================================================")
add_line("CROSS-FILE COMPATIBILITY CHECKS")
add_line("=============================================================================")

# Check if files can be loaded together
all_exist <- all(sapply(files_to_check, file.exists))

if (all_exist) {
  tryCatch({
    # Quick sample alignment check
    add_line("Checking sample alignment across files...")
    
    # Load sample IDs from each file
    expr_samples <- NULL
    traw_samples <- NULL
    pcs_samples <- NULL
    
    # Expression samples
    if (file.exists(files_to_check$expr_parquet)) {
      expr_df <- read_parquet(files_to_check$expr_parquet)
      if ("accession" %in% names(expr_df)) {
        expr_samples <- unique(expr_df$accession)
        add_line(sprintf("Expression samples: %d unique accessions", length(expr_samples)))
      }
    }
    
    # TRAW samples
    if (file.exists(files_to_check$traw_file)) {
      traw_header <- fread(files_to_check$traw_file, nrows = 1)
      meta_cols <- c("CHR", "SNP", "CM", "POS", "COUNTED", "ALT", "(C)M")
      traw_sample_cols <- setdiff(names(traw_header), meta_cols)
      traw_samples <- gsub("^0_", "", traw_sample_cols)
      add_line(sprintf("TRAW samples: %d sample columns", length(traw_samples)))
    }
    
    # PCs samples
    if (file.exists(files_to_check$pcs_file)) {
      pcs_df <- fread(files_to_check$pcs_file, nrows = 10)
      if (ncol(pcs_df) >= 2) {
        pcs_samples <- gsub("^0_", "", as.character(pcs_df[[2]]))
        add_line(sprintf("PCS samples: %d individuals", length(pcs_samples)))
      }
    }
    
    # Sample overlap analysis
    if (!is.null(expr_samples) && !is.null(traw_samples)) {
      common_expr_traw <- intersect(expr_samples, traw_samples)
      add_line(sprintf("Expression-TRAW overlap: %d samples", length(common_expr_traw)))
    }
    
    if (!is.null(expr_samples) && !is.null(pcs_samples)) {
      common_expr_pcs <- intersect(expr_samples, pcs_samples)
      add_line(sprintf("Expression-PCS overlap: %d samples", length(common_expr_pcs)))
    }
    
    if (!is.null(traw_samples) && !is.null(pcs_samples)) {
      common_traw_pcs <- intersect(traw_samples, pcs_samples)
      add_line(sprintf("TRAW-PCS overlap: %d samples", length(common_traw_pcs)))
    }
    
    # Three-way overlap
    if (!is.null(expr_samples) && !is.null(traw_samples) && !is.null(pcs_samples)) {
      three_way_common <- Reduce(intersect, list(expr_samples, traw_samples, pcs_samples))
      add_line(sprintf("Three-way sample overlap: %d samples", length(three_way_common)))
    }
    
  }, error = function(e) {
    add_line(sprintf("Error in compatibility check: %s", e$message))
  })
  
} else {
  missing_files <- names(files_to_check)[!sapply(files_to_check, file.exists)]
  add_line(sprintf("Cannot perform compatibility checks - missing files: %s", 
                   paste(missing_files, collapse = ", ")))
}

add_line("")
add_line("=============================================================================")
add_line("RECOMMENDATIONS FOR RIDGE_ELASTIC_BASELINES2.R")
add_line("=============================================================================")

# Check gene map structure specifically
if (file.exists(files_to_check$gene_map_csv)) {
  gene_map <- fread(files_to_check$gene_map_csv, nrows = 5)
  required_cols <- c("gene_id", "chr", "start", "end")
  has_required <- required_cols %in% names(gene_map)
  
  add_line("Gene map requirements:")
  for (i in seq_along(required_cols)) {
    status <- ifelse(has_required[i], "OK", "MISSING")
    add_line(sprintf("  %s %s", status, required_cols[i]))
  }
  
  if (!all(has_required)) {
    add_line("ISSUE: Gene map missing required columns")
    add_line(sprintf("Available columns: %s", paste(names(gene_map), collapse = ", ")))
  }
}

# Check expression data structure
if (file.exists(files_to_check$expr_parquet)) {
  expr_df <- read_parquet(files_to_check$expr_parquet)
  required_expr_cols <- c("accession", "env", "gene_id", "fpkm")
  has_expr_cols <- required_expr_cols %in% names(expr_df)
  
  add_line("Expression data requirements:")
  for (i in seq_along(required_expr_cols)) {
    status <- ifelse(has_expr_cols[i], "OK", "MISSING")
    add_line(sprintf("  %s %s", status, required_expr_cols[i]))
  }
  
  if ("env" %in% names(expr_df)) {
    envs <- unique(expr_df$env)
    add_line(sprintf("Available environments: %s", paste(envs, collapse = ", ")))
  }
}

add_line("")
add_line("=============================================================================")
add_line("SUMMARY COMPLETE")
add_line("=============================================================================")

# Write output to file
writeLines(output_lines, output_file)
cat(sprintf("\nData structure summary saved to: %s\n", output_file))