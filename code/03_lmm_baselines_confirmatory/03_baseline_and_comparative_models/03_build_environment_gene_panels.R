# === Build per-environment gene panels (eQTL if found; otherwise ridge-proxy) ===
suppressPackageStartupMessages(library(data.table))

envs <- c("WW","WS1","WS2")
dir.create("output/data", showWarnings = FALSE, recursive = TRUE)

# 1) Load gene map
gm_path <- "output/data/gene_map.csv"
stopifnot(file.exists(gm_path))
gm <- fread(gm_path)
if ("gene" %in% names(gm) && !"gene_id" %in% names(gm)) setnames(gm, "gene", "gene_id")
req <- c("gene_id","chr","start","end")
if (!all(req %in% names(gm))) stop("gene_map.csv needs columns: ", paste(req, collapse=", "))
if (!"strand" %in% names(gm)) gm[, strand := "+"]
setkey(gm, gene_id)

# 2) Helper: try to locate a 'significant' eQTL file for each env anywhere under project root
locate_eqtl <- function(env) {
  pats <- paste0("(?i)", env, ".*(signific|signif|cis).*\\.(tsv|csv|txt)$")
  hits <- list.files(".", pattern = pats, recursive = TRUE, full.names = TRUE)
  hits[1]  # first match or NA
}

# 3) Build a panel for each env.
#    Prefer eQTL (FDR<0.05). If not found, fallback: best-per-gene ridge with oof_R2>0.05 (or top 200).
panel_files <- list()
for (E in envs) {
  message("\n[", E, "] building panel...")
  
  eqtl_file <- locate_eqtl(E)
  
  if (!is.na(eqtl_file) && file.exists(eqtl_file)) {
    message("  using eQTL file: ", eqtl_file)
    dt <- fread(eqtl_file)
    if ("gene" %in% names(dt) && !"gene_id" %in% names(dt)) setnames(dt, "gene", "gene_id")
    if (!"gene_id" %in% names(dt)) stop("eQTL file missing 'gene_id' (or 'gene'): ", eqtl_file)
    
    # Ensure FDR
    if (!"FDR" %in% names(dt)) {
      pcol <- intersect(c("p","pvalue","P","p.value","pv","PV"), names(dt))
      if (length(pcol) != 1L) stop("No FDR and ambiguous/missing p-value column in: ", eqtl_file)
      dt[, FDR := p.adjust(get(pcol), method = "BH")]
    }
    genes <- unique(dt[FDR < 0.05, gene_id])
    
    if (!length(genes)) {
      message("  (no FDR<0.05 genes—falling back to ridge-proxy)")
      eqtl_file <- NA
    } else {
      out <- gm[gene_id %in% genes]
      of <- sprintf("output/data/gene_map_cisQTL_%s.csv", E)
      fwrite(out, of)
      message(sprintf("  saved %d genes -> %s", nrow(out), of))
      panel_files[[E]] <- of
      next
    }
  }
  
  # Fallback: ridge-proxy panel
  sum_path <- file.path("output/baselines_linear", E, sprintf("%s_model_summary.tsv", E))
  if (!file.exists(sum_path)) stop("Missing ridge summary for fallback: ", sum_path)
  rs <- fread(sum_path)
  
  # best alpha per gene
  best <- rs[is.finite(oof_R2)][order(-oof_R2), .SD[1], by = gene_id]
  
  # pick genes with oof_R2 > 0.05; if too few, take top 200
  keep <- best[oof_R2 > 0.05, gene_id]
  if (length(keep) < 100) keep <- head(best$gene_id, 200)
  
  out <- gm[gene_id %in% keep]
  of <- sprintf("output/data/gene_map_proxy_%s.csv", E)
  fwrite(out, of)
  message(sprintf("  ridge-proxy: %d genes -> %s", nrow(out), of))
  panel_files[[E]] <- of
}

# 4) Save panel map for the runner
saveRDS(panel_files, "output/data/eqtl_panel_paths.rds")
print(panel_files)

cat("\n[OK] Panels ready. In your runner set:\n",
    "  use_eqtl_panels <- TRUE\n",
    "  panel_map <- readRDS('output/data/eqtl_panel_paths.rds')\n", sep = "")

