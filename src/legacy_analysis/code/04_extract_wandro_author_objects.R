args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: 04_extract_wandro_author_objects.R <input.RData> <output_dir>")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

load(input_path)
required <- c("infant.file", "species.file", "otu.file", "met.file")
missing_objects <- required[!vapply(required, exists, logical(1))]
if (length(missing_objects) > 0) {
  stop(paste("Missing author objects:", paste(missing_objects, collapse = ", ")))
}

write.csv(
  infant.file,
  file.path(output_dir, "wandro_author_infant_metadata.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  species.file,
  file.path(output_dir, "wandro_author_genus_relative_abundance.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  otu.file,
  file.path(output_dir, "wandro_author_otu_relative_abundance.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  met.file,
  file.path(output_dir, "wandro_author_metabolite_intensity.csv"),
  row.names = FALSE,
  quote = TRUE
)

cat(
  sprintf(
    paste0(
      "{\"status\":\"PASS\",\"metadata_samples\":%d,",
      "\"genus_samples\":%d,\"otu_samples\":%d,",
      "\"metabolite_samples\":%d,\"cloud_files_written\":0}\n"
    ),
    nrow(infant.file),
    nrow(species.file),
    nrow(otu.file),
    nrow(met.file)
  )
)
