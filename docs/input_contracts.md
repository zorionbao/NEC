# Input contracts and analysis layout

All paths below are relative to an **external analysis workspace** supplied with `--workspace`. They are not directories to create inside the GitHub repository. The launcher installs only source code. Public files and derived intermediates must be acquired or produced separately; this repository is not a data deposit.

## Core layout

```text
analysis-workspace/
  data/raw/HCA_NEC/Cell_ranger_output/
  inputs/
  04_corrected_results/
    00_remote_snapshot/results/
    01_single_cell_corrected/
    02_statistical_reanalysis/
  legacy_analysis/
    data/public/
    data/processed/
    results/
  new_results/
  publication_figures/
  logs/
```

`00_remote_snapshot` is a historical relative directory name for source-study preprocessing checkpoints. It does not imply a remote connection. No script in this release contains server credentials.

## Raw atlas and donor metadata

- `data/raw/HCA_NEC/Cell_ranger_output/<source-run>/raw_feature_bc_matrix/`: the Zenodo deposit's Matrix Market counts, barcodes and features files, with original run directory names. The raw-atlas script resolves duplicate runs before combining capture units, uses integer counts, and retains source selection labels.
- `04_corrected_results/00_remote_snapshot/results/single_cell/cell_annotations.csv`: the source annotation table with `subject`, `depletion_type`, `orig.ident`, `cell_barcode` and cell annotation fields read by `curated_annotation_table()`. Preserve original barcode/run associations. Any source-study QC table in the same snapshot is an input provenance check, not a substitute for counting the matrices.
- `inputs/donor_metadata.csv`: one row per biological donor with `donor_id` (string), `condition` (`control` or `nec`), `gestational_age_weeks` and `age_at_surgery_days`. These covariates are transcribed from the source clinical supplementary table. `age_at_surgery_days` is postnatal age at surgery. The historical output column `postconception_age_weeks` stores gestational age plus postnatal days divided by seven and is interpreted as **postmenstrual age** in the manuscript.
- `04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad`: output of raw reconstruction, or the corresponding verified count checkpoint. `layers['counts']` contains nonnegative integer counts; `obs` includes donor, condition, selection, sample and cell-state labels. `var_names` are gene symbols. The input must not be an already log-transformed matrix masquerading as counts.

Native models use lineage inputs created by `reference-prepare` under `legacy_analysis/data/processed/dr_cmap`. `reference-train` writes mapped AnnData objects and model folders under `legacy_analysis/results/dr_cmap`. `native-infer` reads those checkpoints and writes `inputs/native_posteriors.csv.gz`. This table includes `cell_id_original`, `donor_id`, `sample_id`, `condition`, `selection`, `scanvi_label`, `lineage_model`, and one `probability__<label>` field per native class. Probabilities are normalized within each lineage. Exact saved-model inference and new model training are distinct reproduction routes.

## Communication resource and imaging

`inputs/liana_native_rank_aggregate.csv` supplies `receptor_complex` for the matched-receptor null universe. The full source-study communication output is also read from `04_corrected_results/00_remote_snapshot/results/communication_strict/` by the broader host-statistics script, including `liana_native_rank_aggregate.csv` and `nichenet_native_ligand_target_links.csv`. Preserve the LIANA/NicheNet resource versions and source/target names; the two files describe different statistical objects.

The source-study imaging count table belongs at `04_corrected_results/00_remote_snapshot/results/strict_mainline/spatial/tunel_lyve1_double_positive_counts.csv`. Its essential fields are `id_number`, `condition` and `double_positive_cells_per_hpf`, with one row per counted field. The analysis groups fields by subject before inference. The underlying source images and count table are not included in this repository.

## Independent bulk-expression checkpoints

For each of `GSE46619` and `GSE64801`, place source-derived preprocessing files under `04_corrected_results/00_remote_snapshot/results/bulk/<accession>/`:

- `expression_gene_matrix.csv`: first field `gene`, remaining columns keyed by sample identifiers, in the expression scale used in the source preprocessing and documented in the manuscript.
- `metadata_harmonized.csv`: sample identifiers and source diagnostic groups. Preserve NEC, spontaneous intestinal perforation and other surgical-control groups. Do not pool diagnostic comparators simply to fit the interface.
- The corresponding platform/probe mapping inputs used by the deconvolution script, where applicable.

`bulk-deconvolution` prepares reference-based composition estimates. `bulk-adjustment` then fits expression models with those estimates. Reference signatures, age-adjusted count models and bulk residual comparisons are kept as separate calculations. Check the source parsing functions before using a differently processed GEO release.

## Stool microbial and metabolite checkpoints

The statistical code starts at the read-profile and author-table layer. It does not run raw FASTQ quality control, MetaPhlAn or HUMAnN. Recreate those profiles using the source methods and the manuscript's declared sampling/profile versions, or use their verified analysis checkpoints in the external workspace.

Under `04_corrected_results/00_remote_snapshot/results/microbiome_strict/`:

- `metaphlan_native_subset/metaphlan_species_abundance_matrix.csv`: rows keyed by `run`, a `condition` column and species percentage columns.
- `metaphlan_native_subset/selected_runs.csv`: source run-to-infant and phase metadata. Repeated runs must map to the same infant identifier; they are not independent cases.
- `humann_native_subset/humann_pathabundance_matrix.csv`: run-level pathways in the source abundance representation.
- `humann_native_subset/samples/`: original stratified HUMAnN pathway profiles for pathway-coverage analyses.
- The original run manifest read by the pathway script. Keep the metadata as deposited even when source publications describe the assay differently.

Under `04_corrected_results/00_remote_snapshot/results/microbiome_metabolome/`:

- `infant_metadata_harmonized.csv`: Wandro sample field `sample`, infant field `Patient`, source grouping field `Group` and sampling covariates used by the parsing functions.
- `metabolite_matrix.csv`: first field is feature identity, subsequent columns are sample identifiers matching the metadata; nonnegative intensities. The `mTIC` total-ion-current QC quantity is removed from biological testing, while all remaining eligible metabolites stay in the multiplicity family.
- `taxon_abundance.csv`: author taxon table with matching sample identifiers for within-cohort taxon-metabolite associations.

For `longitudinal`, place the author repository `.RData` at `legacy_analysis/data/public/preterm_infant_analysis/.RData`. Its required objects are `infant.file`, `species.file`, `otu.file` and `met.file`. The supplied R extraction script converts those objects into source tables. Temporal alignment, repeated infant sampling and follow-up summaries are handled in the longitudinal analysis. Cross-cohort participant joins are not performed.

## Organoid and methylation inputs

The reference downloader obtains `GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz` under `legacy_analysis/data/public/dr_cmap/`. `organoid-trends` computes the gene trends used by `transport`. Stage indices follow the source sample naming and range from zero to three; they describe cell maturation classes.

The methylation downloader handles GSE212913, GSE212997 and GSE213704 and writes `eclipse_sample_metadata.csv` and source coverage files under `legacy_analysis/data/public/eclipse_nec/`. The bin stage aggregates methylated/unmethylated coverage at fixed genomic windows and generates `legacy_analysis/data/processed/eclipse_nec/bin_1kb/`. `methylation-models` recomputes tissue-window tests, selected-window empirical probabilities, covariate-adjusted stool projections, cross-tissue models and promoter-signal collapsing. Complete test families remain in its external result outputs. Tissue-specific controls and repeated/deposited subject identifiers must be retained.

## Output families and figures

Stages write complete result families to `04_corrected_results/02_statistical_reanalysis`, `new_results` and `legacy_analysis/results`. The publication input-preparation script derives display tables from those full families and preserves the original corrected probabilities. It writes a manifest for the renderer; no data are embedded in the figure code.

All inferential units, adjustments and display selections are described in the manuscript and figure legends. A compact supplementary workbook is a reporting selection; it is not a replacement for the complete output families generated by these scripts. Retain those complete outputs and execution logs in the analysis workspace when reproducing the study.


## Atlas diagnostic panels

The `atlas-diagnostics` stage generates all three compact display tables from the current reconstructed atlas and existing statistics, then renders Supplementary Figure 1. It needs the atlas H5AD, `corrected_single_cell_metadata_scores_umap.csv.gz`, `single_cell_rebuild_status.json`, `cell_state_proportions_capture_unit.csv`, the two module/host summaries, donor CLR composition and `new_results/donor_covariates.csv`. These outputs come from `raw-atlas`, `host-statistics` and the donor metadata preparation. All generated tables and figures stay in the external workspace.

Panel C displays the maximum class probability from the broad classifier fitted to all labeled atlas cells. Panel D uses the separate `label_transfer.leave_one_donor_out` confusion matrix in the rebuild status. It is row normalized for display. No confidence threshold filters the map or marker panel. The paired-capture display retains all four paired controls and within-feature standardization across both captures. Panel F retains 12 features and three HC3 models; its dot area reports nominal P values for sensitivity visualization.
