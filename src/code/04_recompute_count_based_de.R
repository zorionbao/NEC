suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript 04_recompute_count_based_de.R <revision_root>")
}

revision_root <- normalizePath(args[[1]], mustWork = TRUE)
single_cell_dir <- file.path(
  revision_root, "04_corrected_results", "01_single_cell_corrected"
)
snapshot_results <- file.path(
  revision_root, "04_corrected_results", "00_remote_snapshot", "results"
)
output_dir <- file.path(
  revision_root, "04_corrected_results", "02_statistical_reanalysis"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

target_genes <- unique(c(
  "SPIB", "GP2", "CCL20", "CXCL16", "TNFRSF11A", "SOX8",
  "PROX1", "PDPN", "FLT4", "LYVE1", "CCL21", "ACKR4",
  "FOXP3", "IL2RA", "CTLA4", "TIGIT", "CXCR6", "CCR7",
  "IL22", "IL22RA1", "EFNA1", "EPHA2", "VEGFC",
  "TNF", "IL1B", "NLRP3", "CXCL8", "ICAM1", "S100A8", "S100A9",
  "EPCAM", "MUC2", "REG3A", "OLFM4", "LGR5"
))

programs <- list(
  antigen_sampling = c(
    "SPIB", "GP2", "CCL20", "CXCL16", "TNFRSF11A", "SOX8"
  ),
  lymphatic_support = c(
    "PROX1", "PDPN", "FLT4", "LYVE1", "CCL21", "ACKR4"
  ),
  immune_residency = c(
    "FOXP3", "IL2RA", "CTLA4", "TIGIT", "CXCR6", "CCR7"
  ),
  epithelial_repair = c(
    "IL22RA1", "EPCAM", "MUC2", "REG3A", "OLFM4", "LGR5"
  ),
  inflammatory_injury = c(
    "TNF", "IL1B", "NLRP3", "CXCL8", "ICAM1", "S100A8", "S100A9"
  )
)

clean_gene_matrix <- function(frame) {
  gene <- as.character(frame[[1]])
  matrix <- as.matrix(frame[, -1, drop = FALSE])
  storage.mode(matrix) <- "numeric"
  rownames(matrix) <- gene
  keep <- !is.na(gene) & nzchar(gene)
  matrix <- matrix[keep, , drop = FALSE]
  matrix <- avereps(matrix, ID = rownames(matrix))
  matrix
}

run_edger <- function(count_path, state_name, metadata) {
  status <- data.frame(
    analysis = paste0("single_cell_pseudobulk_", state_name),
    status = "started",
    detail = "",
    stringsAsFactors = FALSE
  )
  if (!file.exists(count_path)) {
    status$status <- "skipped_missing_input"
    status$detail <- count_path
    return(status)
  }

  frame <- read.csv(count_path, check.names = FALSE)
  counts <- clean_gene_matrix(frame)
  sample_ids <- intersect(colnames(counts), metadata$pseudobulk_id)
  meta <- metadata[match(sample_ids, metadata$pseudobulk_id), , drop = FALSE]
  counts <- counts[, sample_ids, drop = FALSE]
  group <- factor(meta$condition, levels = c("control", "nec"))
  group_counts <- table(group)
  if (length(group_counts) < 2 || any(group_counts < 3)) {
    status$status <- "skipped_insufficient_independent_donors"
    status$detail <- paste(names(group_counts), group_counts, collapse = ";")
    return(status)
  }

  dge <- DGEList(counts = counts, group = group)
  keep <- filterByExpr(dge, group = group)
  dge <- dge[keep, , keep.lib.sizes = FALSE]
  dge <- calcNormFactors(dge, method = "TMM")
  design <- model.matrix(~0 + group)
  colnames(design) <- levels(group)
  dge <- estimateDisp(dge, design, robust = TRUE)
  fit <- glmQLFit(dge, design, robust = TRUE)
  contrast <- makeContrasts(nec_vs_control = nec - control, levels = design)
  test <- glmQLFTest(fit, contrast = contrast)
  table <- topTags(test, n = Inf, sort.by = "PValue")$table
  table$gene <- rownames(table)
  table$cell_state <- state_name
  table$n_control_donors <- as.integer(group_counts[["control"]])
  table$n_nec_donors <- as.integer(group_counts[["nec"]])
  table <- table[, c(
    "cell_state", "gene", "logFC", "logCPM", "F", "PValue", "FDR",
    "n_control_donors", "n_nec_donors"
  )]
  output_path <- file.path(
    output_dir, paste0("single_cell_pseudobulk_edger_", state_name, ".csv")
  )
  write.csv(table, output_path, row.names = FALSE)

  target <- table[table$gene %in% target_genes, , drop = FALSE]
  write.csv(
    target,
    file.path(
      output_dir,
      paste0("single_cell_pseudobulk_target_genes_", state_name, ".csv")
    ),
    row.names = FALSE
  )

  log_cpm <- cpm(dge, log = TRUE, prior.count = 2)
  target_cpm <- log_cpm[rownames(log_cpm) %in% target_genes, , drop = FALSE]
  target_cpm <- data.frame(gene = rownames(target_cpm), target_cpm, check.names = FALSE)
  write.csv(
    target_cpm,
    file.path(
      output_dir,
      paste0("single_cell_pseudobulk_target_logcpm_", state_name, ".csv")
    ),
    row.names = FALSE
  )

  status$status <- "completed"
  status$detail <- paste0(
    "genes=", nrow(table),
    ";control=", group_counts[["control"]],
    ";nec=", group_counts[["nec"]]
  )
  status
}

metadata_path <- file.path(single_cell_dir, "pseudobulk_metadata.csv")
if (!file.exists(metadata_path)) {
  stop("Missing corrected pseudobulk metadata: ", metadata_path)
}
pseudobulk_metadata <- read.csv(metadata_path, stringsAsFactors = FALSE)

status_rows <- list()
status_rows[[length(status_rows) + 1]] <- run_edger(
  file.path(single_cell_dir, "pseudobulk_counts_all_cells.csv.gz"),
  "all_cells",
  pseudobulk_metadata[pseudobulk_metadata$cell_state == "all_cells", , drop = FALSE]
)

state_dir <- file.path(single_cell_dir, "pseudobulk_counts_by_cell_state")
for (state_name in c(
  "M_cell", "LEC", "T_reg", "T_NK", "epithelial",
  "epithelial_stem", "myeloid", "vascular_endothelial", "fibroblast"
)) {
  status_rows[[length(status_rows) + 1]] <- run_edger(
    file.path(state_dir, paste0("pseudobulk_counts_", state_name, ".csv.gz")),
    state_name,
    pseudobulk_metadata[
      pseudobulk_metadata$cell_state == state_name,
      ,
      drop = FALSE
    ]
  )
}

compute_module_scores <- function(expr, meta, accession) {
  rows <- list()
  score_matrix <- matrix(
    NA_real_,
    nrow = length(programs),
    ncol = ncol(expr),
    dimnames = list(names(programs), colnames(expr))
  )
  for (program_name in names(programs)) {
    genes <- intersect(programs[[program_name]], rownames(expr))
    if (length(genes) < 2) {
      next
    }
    gene_values <- expr[genes, , drop = FALSE]
    gene_z <- t(scale(t(gene_values)))
    gene_z[!is.finite(gene_z)] <- NA_real_
    score_matrix[program_name, ] <- colMeans(gene_z, na.rm = TRUE)
    for (sample_name in colnames(expr)) {
      rows[[length(rows) + 1]] <- data.frame(
        accession = accession,
        sample = sample_name,
        condition = as.character(meta[sample_name, "condition"]),
        program = program_name,
        score = score_matrix[program_name, sample_name],
        genes_available = paste(genes, collapse = ";"),
        n_genes_available = length(genes),
        stringsAsFactors = FALSE
      )
    }
  }
  list(long = do.call(rbind, rows), matrix = score_matrix)
}

run_bulk_limma <- function(accession, contrasts) {
  input_dir <- file.path(snapshot_results, "bulk", accession)
  expr_path <- file.path(input_dir, "expression_gene_matrix.csv")
  meta_path <- file.path(input_dir, "metadata_harmonized.csv")
  status <- data.frame(
    analysis = paste0("bulk_", accession),
    status = "started",
    detail = "",
    stringsAsFactors = FALSE
  )
  if (!file.exists(expr_path) || !file.exists(meta_path)) {
    status$status <- "skipped_missing_input"
    status$detail <- input_dir
    return(status)
  }

  expr_frame <- read.csv(expr_path, check.names = FALSE)
  expr <- clean_gene_matrix(expr_frame)
  meta <- read.csv(meta_path, check.names = FALSE, stringsAsFactors = FALSE)
  sample_col <- if ("geo_accession" %in% colnames(meta)) {
    "geo_accession"
  } else {
    colnames(meta)[1]
  }
  rownames(meta) <- as.character(meta[[sample_col]])
  sample_ids <- intersect(colnames(expr), rownames(meta))
  expr <- expr[, sample_ids, drop = FALSE]
  meta <- meta[sample_ids, , drop = FALSE]
  condition <- factor(meta$condition)
  design <- model.matrix(~0 + condition)
  colnames(design) <- levels(condition)
  fit <- lmFit(expr, design)

  de_outputs <- list()
  for (contrast_name in names(contrasts)) {
    contrast_expression <- contrasts[[contrast_name]]
    needed <- unlist(strsplit(gsub(" ", "", contrast_expression), "-", fixed = TRUE))
    if (!all(needed %in% colnames(design))) {
      next
    }
    contrast_matrix <- makeContrasts(
      contrasts = contrast_expression,
      levels = design
    )
    fit2 <- eBayes(contrasts.fit(fit, contrast_matrix), trend = TRUE, robust = TRUE)
    table <- topTable(fit2, number = Inf, sort.by = "P")
    table$gene <- rownames(table)
    names(table)[names(table) == "logFC"] <- "logfc"
    names(table)[names(table) == "P.Value"] <- "pvalue"
    names(table)[names(table) == "adj.P.Val"] <- "fdr_bh"
    table$accession <- accession
    table$contrast <- contrast_name
    table <- table[, c(
      "accession", "contrast", "gene", "logfc", "AveExpr", "t",
      "pvalue", "fdr_bh", "B"
    )]
    de_outputs[[contrast_name]] <- table
    write.csv(
      table,
      file.path(
        output_dir,
        paste0("bulk_limma_", accession, "_", contrast_name, ".csv")
      ),
      row.names = FALSE
    )
    write.csv(
      table[table$gene %in% target_genes, , drop = FALSE],
      file.path(
        output_dir,
        paste0("bulk_target_genes_", accession, "_", contrast_name, ".csv")
      ),
      row.names = FALSE
    )
  }

  module <- compute_module_scores(expr, meta, accession)
  write.csv(
    module$long,
    file.path(output_dir, paste0("bulk_module_scores_", accession, ".csv")),
    row.names = FALSE
  )

  module_de <- list()
  for (contrast_name in names(contrasts)) {
    contrast_expression <- contrasts[[contrast_name]]
    needed <- unlist(strsplit(gsub(" ", "", contrast_expression), "-", fixed = TRUE))
    if (!all(needed %in% colnames(design))) {
      next
    }
    available <- rownames(module$matrix)[
      rowSums(is.finite(module$matrix)) == ncol(module$matrix)
    ]
    if (length(available) == 0) {
      next
    }
    module_fit <- lmFit(module$matrix[available, , drop = FALSE], design)
    contrast_matrix <- makeContrasts(
      contrasts = contrast_expression,
      levels = design
    )
    module_fit2 <- eBayes(
      contrasts.fit(module_fit, contrast_matrix),
      trend = FALSE,
      robust = TRUE
    )
    table <- topTable(module_fit2, number = Inf, sort.by = "P")
    table$program <- rownames(table)
    names(table)[names(table) == "logFC"] <- "score_delta"
    names(table)[names(table) == "P.Value"] <- "pvalue"
    names(table)[names(table) == "adj.P.Val"] <- "fdr_bh"
    table$accession <- accession
    table$contrast <- contrast_name
    module_de[[contrast_name]] <- table[, c(
      "accession", "contrast", "program", "score_delta", "AveExpr",
      "t", "pvalue", "fdr_bh", "B"
    )]
  }
  if (length(module_de) > 0) {
    write.csv(
      do.call(rbind, module_de),
      file.path(output_dir, paste0("bulk_module_effects_", accession, ".csv")),
      row.names = FALSE
    )
  }

  status$status <- "completed"
  status$detail <- paste0(
    "genes=", nrow(expr),
    ";samples=", ncol(expr),
    ";contrasts=", paste(names(de_outputs), collapse = ",")
  )
  status
}

status_rows[[length(status_rows) + 1]] <- run_bulk_limma(
  "GSE46619",
  list(nec_vs_control = "nec-control", nec_vs_sip = "nec-sip")
)
status_rows[[length(status_rows) + 1]] <- run_bulk_limma(
  "GSE64801",
  list(nec_vs_control = "nec-control")
)

status_table <- do.call(rbind, status_rows)
write.csv(
  status_table,
  file.path(output_dir, "count_based_de_status.csv"),
  row.names = FALSE
)
print(status_table)
