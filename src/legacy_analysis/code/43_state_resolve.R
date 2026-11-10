suppressPackageStartupMessages({
  library(limma)
  library(jsonlite)
})

options(stringsAsFactors = FALSE)

arguments <- commandArgs(trailingOnly = FALSE)
script_argument <- grep("^--file=", arguments, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1]]), winslash = "/")
code_root <- dirname(script_path)
analysis_root <- dirname(code_root)
package_root <- dirname(analysis_root)
workspace_root <- dirname(dirname(package_root))
local_revision_root <- dirname(analysis_root)
package_output <- file.path(
  analysis_root, "results", "story_chain_upgrade", "state_resolve"
)
local_output <- file.path(
  local_revision_root, "12_story_chain_upgrade", "state_resolve"
)
dir.create(package_output, recursive = TRUE, showWarnings = FALSE)
dir.create(local_output, recursive = TRUE, showWarnings = FALSE)

coder_root <- file.path(analysis_root, "results", "coder_nec")
bulk_root <- file.path(
  local_revision_root,
  "04_corrected_results",
  "00_remote_snapshot",
  "results",
  "bulk"
)
module_path <- file.path(
  local_revision_root,
  "04_corrected_results",
  "02_statistical_reanalysis",
  "single_cell_module_marker_coverage.csv"
)

write_dual <- function(frame, filename) {
  write.csv(frame, file.path(package_output, filename), row.names = FALSE, na = "")
  write.csv(frame, file.path(local_output, filename), row.names = FALSE, na = "")
}

write_json_dual <- function(payload, filename) {
  text <- toJSON(payload, pretty = TRUE, auto_unbox = TRUE, na = "null")
  writeLines(text, file.path(package_output, filename), useBytes = TRUE)
  writeLines(text, file.path(local_output, filename), useBytes = TRUE)
}

read_expression <- function(accession) {
  path <- file.path(coder_root, paste0("coder_", accession, "_gene_expression.csv"))
  frame <- read.csv(path, check.names = FALSE)
  genes <- toupper(as.character(frame[[1]]))
  matrix <- as.matrix(frame[, -1, drop = FALSE])
  storage.mode(matrix) <- "double"
  rownames(matrix) <- genes
  matrix <- avereps(matrix, ID = rownames(matrix))
  matrix
}

read_metadata <- function(accession, samples) {
  frame <- read.csv(
    file.path(bulk_root, accession, "metadata_harmonized.csv"),
    check.names = FALSE
  )
  rownames(frame) <- as.character(frame$geo_accession)
  frame[samples, , drop = FALSE]
}

fractions_to_pcs <- function(frame, samples) {
  states <- sort(unique(frame$cell_state))
  wide <- matrix(NA_real_, nrow = length(samples), ncol = length(states))
  rownames(wide) <- samples
  colnames(wide) <- states
  for (state in states) {
    subset <- frame[frame$cell_state == state, , drop = FALSE]
    wide[as.character(subset$sample), state] <- subset$relative_fraction
  }
  if (any(!is.finite(wide))) {
    stop("Missing CoDeR fractions after sample alignment")
  }
  wide <- pmax(wide, 1e-6)
  wide <- wide / rowSums(wide)
  clr <- log(wide)
  clr <- clr - rowMeans(clr)
  pca <- prcomp(clr, center = TRUE, scale. = FALSE)
  scores <- pca$x[, seq_len(min(2, ncol(pca$x))), drop = FALSE]
  loadings <- pca$rotation[, seq_len(min(2, ncol(pca$rotation))), drop = FALSE]
  variance <- pca$sdev^2 / sum(pca$sdev^2)
  list(
    fractions = wide,
    clr = clr,
    scores = scores,
    loadings = loadings,
    variance = variance
  )
}

fit_limma <- function(expression, condition, scores, contrast_specs, model_name) {
  condition_levels <- unique(c("control", "sip", "nec")[c(
    "control", "sip", "nec"
  ) %in% unique(condition)])
  condition_factor <- factor(condition, levels = condition_levels)
  if (model_name == "Diagnosis only") {
    design <- model.matrix(~0 + condition_factor)
  } else {
    design <- model.matrix(~0 + condition_factor + scores)
  }
  colnames(design)[seq_along(condition_levels)] <- condition_levels
  fit <- lmFit(expression, design)
  output <- list()
  for (contrast_name in names(contrast_specs)) {
    contrast_string <- contrast_specs[[contrast_name]]
    contrast <- makeContrasts(contrasts = contrast_string, levels = design)
    fit2 <- eBayes(contrasts.fit(fit, contrast), robust = TRUE, trend = TRUE)
    table <- topTable(fit2, number = Inf, sort.by = "none")
    table$gene <- rownames(table)
    table$standard_error <- abs(table$logFC / table$t)
    table$contrast <- contrast_name
    table$model <- model_name
    output[[contrast_name]] <- table[, c(
      "gene", "logFC", "standard_error", "AveExpr", "t",
      "P.Value", "adj.P.Val", "B", "contrast", "model"
    )]
  }
  do.call(rbind, output)
}

module_frame <- read.csv(module_path, check.names = FALSE)
module_genes <- unique(toupper(unlist(strsplit(
  paste(module_frame$markers_available, collapse = ";"),
  ";",
  fixed = TRUE
))))

fraction_frame <- read.csv(
  file.path(coder_root, "coder_bulk_relative_fractions.csv"),
  check.names = FALSE
)

all_effects <- list()
target_effects <- list()
pc_scores <- list()
pc_loadings <- list()
reference_sensitivity <- list()
dataset_summaries <- list()

dataset_specs <- list(
  GSE46619 = c(
    nec_vs_control = "nec-control",
    nec_vs_sip = "nec-sip"
  ),
  GSE64801 = c(nec_vs_control = "nec-control")
)

for (accession in names(dataset_specs)) {
  expression <- read_expression(accession)
  metadata <- read_metadata(accession, colnames(expression))
  condition <- tolower(as.character(metadata$condition))
  fractions <- fraction_frame[
    fraction_frame$accession == accession,
    ,
    drop = FALSE
  ]
  pca <- fractions_to_pcs(fractions, colnames(expression))
  unadjusted <- fit_limma(
    expression,
    condition,
    pca$scores,
    dataset_specs[[accession]],
    "Diagnosis only"
  )
  adjusted <- fit_limma(
    expression,
    condition,
    pca$scores,
    dataset_specs[[accession]],
    "Diagnosis plus composition PC1 and PC2"
  )
  effects <- rbind(unadjusted, adjusted)
  effects$accession <- accession
  effects$gene <- toupper(as.character(effects$gene))
  all_effects[[accession]] <- effects
  target_effects[[accession]] <- effects[
    effects$gene %in% module_genes,
    ,
    drop = FALSE
  ]

  scores_frame <- data.frame(
    sample = rownames(pca$scores),
    accession = accession,
    condition = condition,
    PC1 = pca$scores[, 1],
    PC2 = if (ncol(pca$scores) >= 2) pca$scores[, 2] else 0,
    stringsAsFactors = FALSE
  )
  pc_scores[[accession]] <- scores_frame
  loadings_frame <- data.frame(
    cell_state = rownames(pca$loadings),
    accession = accession,
    PC1_loading = pca$loadings[, 1],
    PC2_loading = if (ncol(pca$loadings) >= 2) pca$loadings[, 2] else 0,
    stringsAsFactors = FALSE
  )
  pc_loadings[[accession]] <- loadings_frame

  ensemble_path <- file.path(
    coder_root,
    paste0("coder_", accession, "_reference_ensemble.csv")
  )
  ensemble <- read.csv(ensemble_path, check.names = FALSE)
  iterations <- unique(as.character(ensemble$reference_iteration))
  iteration_rows <- list()
  target_expression <- expression[
    rownames(expression) %in% module_genes,
    ,
    drop = FALSE
  ]
  for (iteration in iterations) {
    iteration_fractions <- ensemble[
      ensemble$reference_iteration == iteration,
      ,
      drop = FALSE
    ]
    iteration_pca <- fractions_to_pcs(
      iteration_fractions,
      colnames(expression)
    )
    iteration_fit <- fit_limma(
      target_expression,
      condition,
      iteration_pca$scores,
      dataset_specs[[accession]],
      "Diagnosis plus composition PC1 and PC2"
    )
    iteration_fit$accession <- accession
    iteration_fit$reference_iteration <- iteration
    iteration_rows[[iteration]] <- iteration_fit
  }
  reference_sensitivity[[accession]] <- do.call(rbind, iteration_rows)

  dataset_summaries[[accession]] <- list(
    samples = ncol(expression),
    genes = nrow(expression),
    condition_counts = as.list(table(condition)),
    pc1_variance_fraction = unname(pca$variance[[1]]),
    pc2_variance_fraction = unname(pca$variance[[2]]),
    reference_iterations = length(iterations)
  )
}

all_effects_frame <- do.call(rbind, all_effects)
rownames(all_effects_frame) <- NULL
target_effects_frame <- do.call(rbind, target_effects)
rownames(target_effects_frame) <- NULL
scores_frame <- do.call(rbind, pc_scores)
loadings_frame <- do.call(rbind, pc_loadings)
reference_frame <- do.call(rbind, reference_sensitivity)
rownames(reference_frame) <- NULL

base_reference <- reference_frame[
  reference_frame$reference_iteration == "all_control_donors",
  c("accession", "contrast", "gene", "logFC"),
  drop = FALSE
]
names(base_reference)[names(base_reference) == "logFC"] <- "base_logFC"
reference_frame <- merge(
  reference_frame,
  base_reference,
  by = c("accession", "contrast", "gene"),
  all.x = TRUE,
  sort = FALSE
)
reference_frame$direction_matches_base <- sign(reference_frame$logFC) ==
  sign(reference_frame$base_logFC)

reference_summary <- aggregate(
  direction_matches_base ~ accession + contrast + gene,
  reference_frame,
  mean
)
names(reference_summary)[4] <- "reference_leave_one_donor_sign_stability"
range_low <- aggregate(
  logFC ~ accession + contrast + gene,
  reference_frame,
  min
)
names(range_low)[4] <- "reference_effect_min"
range_high <- aggregate(
  logFC ~ accession + contrast + gene,
  reference_frame,
  max
)
names(range_high)[4] <- "reference_effect_max"
reference_summary <- merge(
  reference_summary,
  range_low,
  by = c("accession", "contrast", "gene")
)
reference_summary <- merge(
  reference_summary,
  range_high,
  by = c("accession", "contrast", "gene")
)

write_dual(all_effects_frame, "state_resolve_all_gene_effects.csv")
write_dual(target_effects_frame, "state_resolve_target_gene_effects.csv")
write_dual(scores_frame, "state_resolve_composition_pc_scores.csv")
write_dual(loadings_frame, "state_resolve_composition_pc_loadings.csv")
write_dual(reference_frame, "state_resolve_reference_iteration_effects.csv")
write_dual(reference_summary, "state_resolve_reference_sensitivity_summary.csv")

adjusted_targets <- target_effects_frame[
  target_effects_frame$model == "Diagnosis plus composition PC1 and PC2",
  ,
  drop = FALSE
]
summary_payload <- list(
  analysis = "STATE-RESOLVE",
  status = "complete",
  datasets = dataset_summaries,
  full_gene_effect_rows = nrow(all_effects_frame),
  target_gene_effect_rows = nrow(target_effects_frame),
  reference_sensitivity_rows = nrow(reference_frame),
  minimum_adjusted_target_fdr = min(
    adjusted_targets$adj.P.Val,
    na.rm = TRUE
  ),
  adjusted_target_genes_fdr_below_0_05 = sum(
    adjusted_targets$adj.P.Val < 0.05,
    na.rm = TRUE
  ),
  method_notes = c(
    "Expression matrices retain their deposited log2-scale processing.",
    "CoDeR fractions were closed, CLR transformed and summarized by two principal components.",
    "Limma empirical-Bayes models were fitted with diagnosis alone and with composition PC1 and PC2.",
    "Reference sensitivity repeated the adjusted model for the full control reference and five leave-one-reference-donor signatures."
  )
)
write_json_dual(summary_payload, "state_resolve_summary.json")
cat(toJSON(summary_payload, pretty = TRUE, auto_unbox = TRUE), "\n")
