# Neonatal intestinal endothelial and immune-state analyses

This folder contains the research analysis and publication-figure code for the manuscript. It is prepared for upload as a GitHub repository. It contains **code and documentation only**: no participant-level records, expression matrices, processed results, images, model weights, access credentials or copies of public datasets.

Public input sources are listed in [docs/data_sources.md](docs/data_sources.md). The expected file layout and fields are documented in [docs/input_contracts.md](docs/input_contracts.md). Place downloaded data and generated results in an external analysis workspace. The launcher rejects a workspace inside this repository, and `.gitignore` excludes common scientific data formats.

## Analysis scope

The code covers raw 10x reconstruction; developmental-reference preparation and scVI/scANVI mapping; donor-level abundance and marker-program comparisons; age-adjusted edgeR analyses; posterior and marker sensitivities; imaging subject aggregation; independent bulk-expression comparisons; infant-level microbial and metabolite analyses; longitudinal summaries; exploratory methylation models; organoid-state transport; and the four main publication figures and revised supplementary panels. Separate scripts retain their statistical comparison families, seeds and missingness rules.

The primary donor-level contrast and exploratory gene-wide analyses have different inferential roles. Functions calculate results from their input files; the repository contains no stored findings. Historical internal file and function names are retained where they form dependencies. Their biological interpretation is defined in the revised manuscript.

## Start here

Use the dependency groups in [environments/README.md](environments/README.md). Keep the statistical, native mapping and raw-rebuild environments separate when reproducing their recorded software versions. Install Liberation Sans for all publication plots.

```bash
python run.py list
python run.py install --workspace /path/to/nec-analysis
python run.py primary --workspace /path/to/nec-analysis --check
python run.py primary --workspace /path/to/nec-analysis
```

`--check` prints the command plan and checks prerequisite paths without writing anything or launching an analysis. `install` copies the scripts into the external workspace with their required relative relationships. Existing different scripts are protected. Analysis logs and code hashes are written into that workspace. Set `--python`, `--native-python` and `--rscript` when the interpreters differ from the current environment.

```bash
python run.py native-infer --workspace /path/to/nec-analysis --native-python /path/to/native-env/bin/python
python run.py adjusted-de --workspace /path/to/nec-analysis --rscript /path/to/Rscript
python run.py main-figures --workspace /path/to/nec-analysis
```

The stages do not automatically run dependencies or download all datasets. `reference-download` and `methylation-download` are explicit download stages. Other stages consume inputs already prepared in the external workspace. Large raw-read profiling, the original source-study metadata harmonization and saved model checkpoints are not bundled. See the input contract for those starting points. Re-training a neural model can produce results that differ from saved-model inference even with fixed seeds; these operations have separate stage names.

## Recommended order

1. Obtain the public sources and the metadata/annotation inputs described in the input contract. Run `raw-atlas` in the raw-rebuild environment.
2. Run `reference-download`, `reference-prepare`, `reference-train`, then `native-infer`; alternatively supply compatible mapped objects and saved models for inference.
3. Run `pseudobulk`, `unadjusted-de`, `adjusted-de`, `host-statistics`, `primary`, `specificity` and `posterior-assignment`.
4. With the independent source-study tables prepared, run `microbial-statistics`, `classification`, `pathway-coverage`, `longitudinal`, `bulk-deconvolution`, `bulk-adjustment` and `cross-source`.
5. Run `organoid-trends` then `transport`. Run `methylation-download`, `methylation-bins` and `methylation-models` for the exploratory genomic layer.
6. Run `main-figures`, `atlas-diagnostics` and `supplementary-figures` after their source statistics are complete. Figure inputs and output hashes are generated in the external workspace. The supplementary renderer covers the panels regenerated for the revision; source-study image panels are not redistributed here.

## Reproducibility and provenance

[docs/source_provenance.json](docs/source_provenance.json) identifies the source scripts and their original hashes. Portability edits replace machine-specific paths and interpreter locations; the statistical definitions are preserved. `prepare_supplementary1.py` preserves the original marker, paired-capture and HC3 covariate calculations; its companion renderer reads the current UMAP and distinguishes full-fit confidence from held-out concordance. `prepare_organoid_trends.py` isolates the original gene-trend calculation needed by the transport model. `rerun_eclipse_full.py` checks a vectorized Welch calculation against SciPy before recomputing the complete methylation models.