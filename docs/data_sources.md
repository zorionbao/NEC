# Public data sources

This is an accession and download-location guide, not a redistribution of data. The manuscript and Supplementary Table 1 provide the study-level eligibility definitions and source citations.

| Resource | Public source | Role |
|---|---|---|
| Human neonatal NEC intestinal atlas | [Zenodo version 7780794](https://doi.org/10.5281/zenodo.7780794) and Egozi et al., PLOS Biology source supplementary material | Raw 10x count files, source annotations and clinical covariates |
| Space-Time Gut Cell Atlas | [Wellcome Sanger Institute portal](https://www.gutcellatlas.org/) | Developmental reference lineages |
| Gut reference epithelium | [epi_raw_counts02_v2.h5ad](https://cellgeni.cog.sanger.ac.uk/gutcellatlas/epi_raw_counts02_v2.h5ad) | Raw reference counts |
| Gut reference endothelium | [endothelium_raw_counts02_v2.h5ad](https://cellgeni.cog.sanger.ac.uk/gutcellatlas/endothelium_raw_counts02_v2.h5ad) | Raw reference counts |
| Gut reference T/NK cells | [Tcell_raw_counts02_v2.h5ad](https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Tcell_raw_counts02_v2.h5ad) | Raw reference counts |
| Independent bowel bulk expression | [GSE46619](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46619), [GSE64801](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64801) | Source-specific expression comparisons |
| Stool microbial sequencing | [PRJNA1258245](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1258245) | Read-derived microbial profiles; retain source run metadata |
| Wandro preterm-infant cohort | [Author repository](https://github.com/swandro/preterm_infant_analysis), [source article](https://doi.org/10.1128/mSphere.00104-18) | Longitudinal genus/OTU and metabolite objects and infant metadata |
| Human M-cell differentiation | [GSE275771](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275771), [GSE275772](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275772) | Organoid maturation counts and independent single-cell reference |

Methylation series accessions and source files are defined explicitly in `src/legacy_analysis/code/14_download_eclipse_nec.py`. The included downloader writes its own source metadata and download manifest into the external workspace. GENCODE annotation release 46 is used by the bin-preparation script. The LIANA tested receptor universe is an input to the expression/detection-matched ligand-receptor sensitivity analysis; its resource and version must match the preceding communication analysis.

For Wandro data, use the author repository's `.RData` and supplied tables without renaming biological groups. For PRJNA1258245, retain the source run annotations alongside the publication metadata; the revised manuscript uses the assay-neutral term "read-derived microbial profiles". Separate cohorts are not joined at participant level.
