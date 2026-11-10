#!/usr/bin/env python3
"""Prepare public developmental references and the strict NEC atlas for DR-CMap.

This script reads the downloaded Gut Cell Atlas
and GSE275772 resources, retains every strict NEC query cell in the three target
compartments, and performs deterministic donor-by-state balancing only in the
large healthy reference atlases.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 20260726
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public" / "dr_cmap"
PROCESSED = ROOT / "data" / "processed" / "dr_cmap"
RESULTS = ROOT / "results" / "dr_cmap"
MANIFESTS = ROOT / "manifests"
PROJECT_ROOT = ROOT.parent
STRICT_H5AD = (
    PROJECT_ROOT
    / "04_corrected_results"
    / "01_single_cell_corrected"
    / "corrected_hca_nec_atlas.h5ad"
)

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)


LINEAGES = {
    "epithelial": {
        "reference_file": "epi_raw_counts02_v2.h5ad",
        "query_states": {"M_cell", "epithelial", "epithelial_stem"},
        "target_labels": {"M cell"},
        "markers": {
            "M cell": ["SPIB", "GP2", "SOX8", "TNFRSF11A", "CCL20", "MARCKSL1"],
            "Enterocyte": ["ALPI", "FABP1", "APOA4", "KRT20"],
            "Stem and progenitor": ["LGR5", "OLFM4", "SMOC2", "MKI67"],
            "Goblet": ["MUC2", "SPINK4", "CLCA1", "AGR2"],
            "Paneth": ["DEFA5", "DEFA6", "LYZ", "REG3A"],
            "Tuft": ["POU2F3", "TRPM5", "AVIL", "SOX9"],
            "Enteroendocrine": ["CHGA", "CHGB", "NEUROD1", "PAX6"],
        },
    },
    "endothelial": {
        "reference_file": "endothelium_raw_counts02_v2.h5ad",
        "query_states": {"LEC", "vascular_endothelial"},
        "target_labels": {"LEC"},
        "markers": {
            "LEC": ["PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "ACKR4"],
            "Arterial EC": ["EFNB2", "SOX17", "GJA5", "CXCL12"],
            "Venous EC": ["ACKR1", "NR2F2", "VWF", "EPAS1"],
            "Cycling EC": ["MKI67", "TOP2A", "CENPF", "UBE2C"],
        },
    },
    "immune": {
        "reference_file": "Tcell_raw_counts02_v2.h5ad",
        "query_states": {"T_NK", "T_reg"},
        "target_labels": {"ILC3", "Resident-like T", "Treg"},
        "markers": {
            "ILC3": ["RORC", "IL7R", "KIT", "AHR", "IL22", "NCR1"],
            "Resident-like T": ["CD69", "ITGAE", "CXCR6", "ZNF683", "RUNX3", "CCL5"],
            "Treg": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "CCR7"],
            "Naive T": ["SELL", "CCR7", "TCF7", "LEF1"],
            "Effector T": ["NKG7", "GZMK", "GZMB", "IFNG"],
            "NK and NKT": ["KLRD1", "GNLY", "TYROBP", "TRAC"],
        },
    },
}


def canonical_gene(value: str) -> str:
    return re.sub(r"__chr(?:[0-9XYM]+)$", "", str(value)).upper()


def parse_age_pc_weeks(value: object) -> float:
    """Convert source age labels to approximate post-conception weeks."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    fetal = re.fullmatch(r"([0-9.]+)Wk", text, flags=re.IGNORECASE)
    if fetal:
        return float(fetal.group(1))
    adult_range = re.fullmatch(r"([0-9.]+)-([0-9.]+)", text)
    if adult_range:
        years = (float(adult_range.group(1)) + float(adult_range.group(2))) / 2
        return 40.0 + 52.1775 * years
    if re.fullmatch(r"[0-9.]+", text):
        return 40.0 + 52.1775 * float(text)
    return np.nan


def age_stage(age_weeks: float) -> str:
    if not np.isfinite(age_weeks):
        return "unknown"
    if age_weeks < 37:
        return "fetal"
    if age_weeks <= 52:
        return "neonatal"
    if age_weeks < 40 + 18 * 52.1775:
        return "pediatric"
    return "adult"


def epithelial_label(annotation: str) -> str:
    value = str(annotation)
    if value == "Microfold cell":
        return "M cell"
    if value in {"Enterocyte", "Colonocyte", "BEST4+ epithelial", "CLDN10+ cells"}:
        return "Enterocyte"
    if value in {"TA", "Proximal progenitor", "Distal progenitor", "Stem cells"}:
        return "Stem and progenitor"
    if value in {"Goblet cell", "BEST2+ Goblet cell"}:
        return "Goblet"
    if value == "Paneth":
        return "Paneth"
    if value == "Tuft":
        return "Tuft"
    if value in {
        "EECs",
        "EC cells (TAC1+)",
        "Progenitor (NEUROG3+)",
        "M/X cells (MLN/GHRL+)",
        "I cells (CCK+)",
        "D cells (SST+)",
        "L cells (PYY+)",
        "K cells (GIP+)",
        "EC cells (NPW+)",
        "N cells (NTS+)",
        "β cells (INS+)",
    }:
        return "Enteroendocrine"
    return "Other epithelial"


def endothelial_label(annotation: str) -> str:
    value = str(annotation)
    if value.startswith("LEC"):
        return "LEC"
    if "arterial" in value.lower():
        return "Arterial EC"
    if "venous" in value.lower():
        return "Venous EC"
    if value == "cycling EC":
        return "Cycling EC"
    return "Other EC"


def immune_label(annotation: str) -> str:
    value = str(annotation)
    if value in {"ILC3", "LTi-like NCR- ILC3", "LTi-like NCR+ ILC3"}:
        return "ILC3"
    if value in {"ILCP", "ILC2"}:
        return "Other ILC"
    if value == "Treg":
        return "Treg"
    if value in {"CD8 Tmem", "CX3CR1+ CD8 Tmem"}:
        return "Resident-like T"
    if value in {"SELL+ CD4 T", "SELL+ CD8 T"}:
        return "Naive T"
    if value in {"Activated CD4 T", "Activated CD8 T", "Tfh", "Th17", "Th1"}:
        return "Effector T"
    if value in {
        "gdT",
        "TRGV2 gdT",
        "TRGV5/7 gdT",
        "TRGV4 gdT",
        "TRDV2/TRGV9 gdT",
        "MAIT cell",
    }:
        return "Unconventional T"
    if value in {"NK cell", "NK T cell"}:
        return "NK and NKT"
    if value == "fetal T cell":
        return "Fetal T"
    return "Other immune"


LABEL_FUNCTIONS = {
    "epithelial": epithelial_label,
    "endothelial": endothelial_label,
    "immune": immune_label,
}


def read_gse275772() -> tuple[ad.AnnData, pd.DataFrame]:
    tar_path = PUBLIC / "GSE275772_RAW.tar"
    matrices: list[pd.DataFrame] = []
    plate_ids: list[str] = []
    with tarfile.open(tar_path) as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not member.name.endswith(".tsv.gz"):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with gzip.GzipFile(fileobj=source) as gz:
                frame = pd.read_csv(io.BytesIO(gz.read()), sep="\t", index_col=0)
            frame.index = [canonical_gene(item) for item in frame.index]
            frame = frame.groupby(level=0, sort=False).sum()
            plate = member.name.split("_", 1)[0]
            frame.columns = [f"{plate}_{column}" for column in frame.columns]
            matrices.append(frame)
            plate_ids.extend([plate] * frame.shape[1])
    if not matrices:
        raise RuntimeError("No GSE275772 count matrices were found in the public TAR.")
    merged = pd.concat(matrices, axis=1, join="outer").fillna(0)
    values = sparse.csr_matrix(merged.T.to_numpy(dtype=np.float32))
    adata = ad.AnnData(values)
    adata.obs_names = merged.columns.astype(str)
    adata.var_names = merged.index.astype(str)
    adata.var_names_make_unique()
    total = np.asarray(values.sum(axis=1)).ravel()
    detected = np.asarray((values > 0).sum(axis=1)).ravel()
    mito_mask = adata.var_names.str.startswith("MT-")
    mito = np.asarray(values[:, mito_mask].sum(axis=1)).ravel()
    pct_mito = np.divide(mito, total, out=np.zeros_like(mito), where=total > 0) * 100
    qc = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "plate": plate_ids,
            "total_counts": total,
            "n_genes_by_counts": detected,
            "pct_counts_mt": pct_mito,
        }
    )
    qc["retained"] = (
        (qc["total_counts"] >= 1_000)
        & (qc["n_genes_by_counts"] >= 300)
        & (qc["pct_counts_mt"] <= 50)
    )
    adata = adata[qc["retained"].to_numpy()].copy()
    retained = qc.loc[qc["retained"]].set_index("cell_id")
    adata.obs["source"] = "GSE275772"
    adata.obs["query_kind"] = "external_validation"
    adata.obs["donor_id"] = retained.loc[adata.obs_names, "plate"].astype(str).to_numpy()
    adata.obs["sample_id"] = adata.obs["donor_id"].to_numpy()
    adata.obs["condition"] = "external_organoid"
    adata.obs["selection"] = "FACS_SORTseq"
    adata.obs["batch"] = "GSE275772_" + adata.obs["donor_id"].astype(str)
    adata.obs["protocol"] = "SORT-seq"
    adata.obs["region_group"] = "intestinal_organoid"
    adata.obs["age_pc_weeks"] = np.nan
    adata.obs["development_stage"] = "unknown_organoid"
    adata.obs["reference_label"] = "Unknown"
    adata.obs["original_annotation"] = "Unlabeled organoid cell"
    adata.obs["original_cell_state"] = "Unlabeled organoid cell"
    return adata, qc


def select_reference_indices(
    obs: pd.DataFrame,
    target_labels: set[str],
    per_donor_target: int = 500,
    per_donor_other: int = 160,
    per_label_cap: int = 2_500,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Balance the reference only; retain every rare target up to donor cap."""
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for (label, donor), block in obs.groupby(["reference_label", "donor_id"], observed=True):
        cap = per_donor_target if label in target_labels else per_donor_other
        take = min(len(block), cap)
        chosen = (
            block.index.to_numpy()
            if take == len(block)
            else RNG.choice(block.index.to_numpy(), size=take, replace=False)
        )
        selected.extend(chosen.tolist())
        rows.append(
            {
                "reference_label": label,
                "donor_id": donor,
                "available_cells": len(block),
                "retained_cells_before_label_cap": take,
                "target_label": label in target_labels,
            }
        )
    selected_index = pd.Index(selected)
    provisional = obs.loc[selected_index]
    final: list[str] = []
    for label, block in provisional.groupby("reference_label", observed=True):
        cap = per_label_cap * (2 if label in target_labels else 1)
        take = min(len(block), cap)
        chosen = (
            block.index.to_numpy()
            if take == len(block)
            else RNG.choice(block.index.to_numpy(), size=take, replace=False)
        )
        final.extend(chosen.tolist())
    final_index = pd.Index(final)
    manifest = pd.DataFrame(rows)
    final_counts = obs.loc[final_index].groupby(
        ["reference_label", "donor_id"], observed=True
    ).size()
    manifest["retained_cells"] = [
        int(final_counts.get((row.reference_label, row.donor_id), 0))
        for row in manifest.itertuples()
    ]
    manifest["excluded_cells"] = (
        manifest["available_cells"] - manifest["retained_cells"]
    )
    return obs.index.get_indexer(final_index), manifest


def reference_obs(backed: ad.AnnData, lineage: str) -> pd.DataFrame:
    obs = backed.obs.copy()
    healthy = obs["Diagnosis"].astype(str).isin(
        {"fetal", "Healthy adult", "Pediatric healthy"}
    )
    obs = obs.loc[healthy].copy()
    obs["reference_label"] = [
        LABEL_FUNCTIONS[lineage](item) for item in obs["annotation"].astype(str)
    ]
    obs["donor_id"] = obs["Sample name"].astype(str).str.strip()
    obs["sample_id"] = obs["sample name"].astype(str).str.strip()
    obs["source"] = "Gut Cell Atlas"
    obs["query_kind"] = "reference"
    obs["condition"] = "healthy_reference"
    obs["selection"] = obs["Fraction"].astype(str)
    obs["batch"] = "GCA_" + obs["donor_id"].astype(str)
    obs["protocol"] = "10x_" + obs["10X"].astype(str)
    obs["region_group"] = obs["Region"].astype(str)
    obs["age_pc_weeks"] = [parse_age_pc_weeks(item) for item in obs["Age"]]
    obs["development_stage"] = [age_stage(item) for item in obs["age_pc_weeks"]]
    obs["original_annotation"] = obs["annotation"].astype(str)
    obs["original_cell_state"] = obs["annotation"].astype(str)
    return obs


def strict_query(lineage: str, shared_genes: list[str]) -> ad.AnnData:
    backed = ad.read_h5ad(STRICT_H5AD, backed="r")
    state_mask = backed.obs["cell_state"].astype(str).isin(
        LINEAGES[lineage]["query_states"]
    )
    positions = np.flatnonzero(state_mask.to_numpy())
    strict_gene_map = {
        str(gene).upper(): str(gene) for gene in backed.var_names.astype(str)
    }
    strict_original_genes = [strict_gene_map[gene] for gene in shared_genes]
    query = backed[positions, strict_original_genes].to_memory()
    backed.file.close()
    query.X = query.layers["counts"].copy()
    query.layers.clear()
    query.obs["source"] = "HCA NEC atlas"
    query.obs["query_kind"] = "strict_nec_query"
    query.obs["sample_id"] = query.obs["sample"].astype(str)
    query.obs["condition"] = query.obs["condition"].astype(str).str.lower()
    query.obs["selection"] = query.obs["selection"].astype(str)
    query.obs["batch"] = "NEC_" + query.obs["donor_id"].astype(str)
    query.obs["protocol"] = "10x_protocol_unreported"
    query.obs["region_group"] = "intestinal_resection"
    query.obs["age_pc_weeks"] = 40.0
    query.obs["development_stage"] = "neonatal"
    query.obs["reference_label"] = "Unknown"
    query.obs["original_annotation"] = query.obs["cell_state"].astype(str)
    query.obs["original_cell_state"] = query.obs["cell_state"].astype(str)
    query.var_names = pd.Index(shared_genes)
    query.var_names_make_unique()
    return query


def prepare_lineage(
    lineage: str,
    external_epithelial: ad.AnnData | None,
) -> tuple[ad.AnnData, pd.DataFrame]:
    cfg = LINEAGES[lineage]
    ref_path = PUBLIC / cfg["reference_file"]
    backed = ad.read_h5ad(ref_path, backed="r")
    obs = reference_obs(backed, lineage)
    ref_positions, sampling = select_reference_indices(obs, cfg["target_labels"])
    ref_positions = np.sort(ref_positions)
    original_positions = backed.obs_names.get_indexer(obs.index[ref_positions])
    strict_backed = ad.read_h5ad(STRICT_H5AD, backed="r")
    reference_gene_map = {
        str(gene).upper(): str(gene) for gene in backed.var_names.astype(str)
    }
    strict_gene_map = {
        str(gene).upper(): str(gene) for gene in strict_backed.var_names.astype(str)
    }
    shared = sorted(
        set(reference_gene_map)
        & set(strict_gene_map)
    )
    strict_backed.file.close()
    reference_original_genes = [reference_gene_map[gene] for gene in shared]
    reference = backed[original_positions, reference_original_genes].to_memory()
    reference.obs = obs.iloc[ref_positions].copy()
    backed.file.close()
    reference.var_names = pd.Index(shared)
    reference.var_names_make_unique()
    reference.X = sparse.csr_matrix(reference.X)
    query = strict_query(lineage, shared)

    components: list[ad.AnnData] = [reference, query]
    if lineage == "epithelial" and external_epithelial is not None:
        shared = sorted(set(shared) & set(external_epithelial.var_names))
        reference = reference[:, shared].copy()
        query = query[:, shared].copy()
        external = external_epithelial[:, shared].copy()
        components = [reference, query, external]

    sc.pp.highly_variable_genes(
        reference,
        n_top_genes=min(2_400, reference.n_vars),
        flavor="seurat_v3",
        subset=False,
    )
    marker_genes = {
        gene
        for genes in cfg["markers"].values()
        for gene in genes
        if gene in reference.var_names
    }
    selected_genes = sorted(
        set(reference.var_names[reference.var["highly_variable"]]) | marker_genes
    )
    components = [item[:, selected_genes].copy() for item in components]
    combined = ad.concat(
        components,
        axis=0,
        join="inner",
        merge="same",
        index_unique="__",
        fill_value=0,
    )
    combined.X = sparse.csr_matrix(combined.X, dtype=np.float32)
    keep_obs = [
        "source",
        "query_kind",
        "donor_id",
        "sample_id",
        "condition",
        "selection",
        "batch",
        "protocol",
        "region_group",
        "age_pc_weeks",
        "development_stage",
        "reference_label",
        "original_annotation",
        "original_cell_state",
    ]
    combined.obs = combined.obs[keep_obs].copy()
    for column in keep_obs:
        if column != "age_pc_weeks":
            combined.obs[column] = combined.obs[column].astype(str)
    age = combined.obs["age_pc_weeks"].astype(float)
    median_age = float(np.nanmedian(age))
    combined.obs["age_pc_weeks_imputed"] = age.fillna(median_age)
    combined.obs["age_log_scaled"] = np.log1p(
        combined.obs["age_pc_weeks_imputed"].astype(float)
    )
    combined.obs["age_log_scaled"] = (
        combined.obs["age_log_scaled"] - combined.obs["age_log_scaled"].mean()
    ) / combined.obs["age_log_scaled"].std(ddof=0)
    combined.obs["cell_id_original"] = [
        name.rsplit("__", 1)[0] for name in combined.obs_names
    ]
    for column in [
        "source",
        "query_kind",
        "donor_id",
        "sample_id",
        "condition",
        "selection",
        "batch",
        "protocol",
        "region_group",
        "development_stage",
        "reference_label",
    ]:
        combined.obs[column] = pd.Categorical(combined.obs[column])
    combined.write_h5ad(PROCESSED / f"{lineage}_drcmap_input.h5ad", compression="gzip")
    sampling.insert(0, "lineage", lineage)
    return combined, sampling


def main() -> None:
    if not STRICT_H5AD.exists():
        raise FileNotFoundError(STRICT_H5AD)
    external, external_qc = read_gse275772()
    external.write_h5ad(PROCESSED / "gse275772_qc_counts.h5ad", compression="gzip")
    external_qc.to_csv(RESULTS / "gse275772_cell_qc.csv", index=False)

    sampling_tables: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}
    for lineage in LINEAGES:
        combined, sampling = prepare_lineage(
            lineage, external if lineage == "epithelial" else None
        )
        sampling_tables.append(sampling)
        summaries[lineage] = {
            "cells": int(combined.n_obs),
            "genes": int(combined.n_vars),
            "cells_by_query_kind": combined.obs["query_kind"]
            .value_counts()
            .astype(int)
            .to_dict(),
            "reference_labels": combined.obs.loc[
                combined.obs["query_kind"] == "reference", "reference_label"
            ]
            .value_counts()
            .astype(int)
            .to_dict(),
            "strict_query_cells_retained": int(
                (combined.obs["query_kind"] == "strict_nec_query").sum()
            ),
        }
    sampling_manifest = pd.concat(sampling_tables, ignore_index=True)
    sampling_manifest.to_csv(
        MANIFESTS / "dr_cmap_reference_sampling_manifest.csv", index=False
    )
    report = {
        "status": "PASS",
        "analysis": "DR-CMap input preparation",
        "random_seed": SEED,
        "strict_atlas_path": str(STRICT_H5AD),
        "strict_atlas_cells": 25_562,
        "gse275772_cells_total": int(len(external_qc)),
        "gse275772_cells_retained": int(external_qc["retained"].sum()),
        "lineages": summaries,
        "reference_sampling_only": True,
        "all_strict_query_cells_retained_within_target_compartments": True,
        "cloud_files_written": 0,
    }
    (MANIFESTS / "dr_cmap_input_preparation_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
