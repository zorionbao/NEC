from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from revision_stats import (
    RANDOM_SEED,
    bh_fdr,
    effect_row,
    exact_group_permutation,
    paired_sign_flip,
    safe_zscore,
)


CELL_STATE_ORDER = [
    "M_cell",
    "LEC",
    "T_reg",
    "T_NK",
    "epithelial",
    "epithelial_stem",
    "myeloid",
    "vascular_endothelial",
    "fibroblast",
]

MODULES = {
    "Antigen sampling": {
        "genes": ["SPIB", "GP2", "CCL20", "CXCL16", "TNFRSF11A", "SOX8"],
        "states": ["M_cell", "epithelial", "epithelial_stem"],
    },
    "Lymphatic support": {
        "genes": ["PROX1", "PDPN", "FLT4", "LYVE1", "CCL21", "ACKR4"],
        "states": ["LEC", "vascular_endothelial", "fibroblast"],
    },
    "Immune residency": {
        "genes": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "CXCR6", "CCR7"],
        "states": ["T_reg", "T_NK"],
    },
    "Epithelial repair": {
        "genes": ["IL22RA1", "EPCAM", "MUC2", "REG3A", "OLFM4", "LGR5"],
        "states": ["M_cell", "epithelial", "epithelial_stem"],
    },
    "Inflammatory injury": {
        "genes": ["TNF", "IL1B", "NLRP3", "CXCL8", "ICAM1", "S100A8", "S100A9"],
        "states": ["M_cell", "epithelial", "epithelial_stem", "myeloid"],
    },
}

MECHANISTIC_AXES = [
    {
        "axis": "CXCL16-CXCR6",
        "class": "ligand-receptor",
        "source_state": "M_cell",
        "source_gene": "CXCL16",
        "target_state": "T_reg",
        "target_gene": "CXCR6",
    },
    {
        "axis": "CCL21-CCR7",
        "class": "ligand-receptor",
        "source_state": "LEC",
        "source_gene": "CCL21",
        "target_state": "T_reg",
        "target_gene": "CCR7",
    },
    {
        "axis": "VEGFC-FLT4",
        "class": "ligand-receptor",
        "source_state": "fibroblast",
        "source_gene": "VEGFC",
        "target_state": "LEC",
        "target_gene": "FLT4",
    },
    {
        "axis": "EFNA1-EPHA2",
        "class": "ligand-receptor",
        "source_state": "LEC",
        "source_gene": "EFNA1",
        "target_state": "epithelial",
        "target_gene": "EPHA2",
    },
    {
        "axis": "TNF-ICAM1",
        "class": "ligand-target",
        "source_state": "myeloid",
        "source_gene": "TNF",
        "target_state": "epithelial",
        "target_gene": "ICAM1",
    },
    {
        "axis": "IL22-IL22RA1",
        "class": "ligand-receptor",
        "source_state": "T_NK",
        "source_gene": "IL22",
        "target_state": "epithelial",
        "target_gene": "IL22RA1",
    },
]

TARGET_GENES = sorted(
    {
        gene
        for axis in MECHANISTIC_AXES
        for gene in (axis["source_gene"], axis["target_gene"])
    }
    | {
        "NLRP3",
        "CXCL8",
        "S100A8",
        "S100A9",
        "FOXP3",
        "CTLA4",
        "TIGIT",
        "LYVE1",
        "PROX1",
        "PDPN",
        "SPIB",
        "GP2",
        "CCL20",
        "LGR5",
        "OLFM4",
        "MUC2",
        "REG3A",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_safe),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_fdr(frame: pd.DataFrame, column: str = "p_value") -> pd.DataFrame:
    output = frame.copy()
    output["fdr_bh"] = bh_fdr(output[column])
    return output


def expression_mean_by_group(
    adata: ad.AnnData,
    gene: str,
    mask: np.ndarray,
    group_columns: list[str],
    minimum_cells: int = 10,
) -> pd.DataFrame:
    if gene not in adata.var_names:
        return pd.DataFrame()
    gene_index = int(adata.var_names.get_loc(gene))
    expression = adata.X[:, gene_index]
    if sparse.issparse(expression):
        expression = np.asarray(expression.toarray()).ravel()
    else:
        expression = np.asarray(expression).ravel()
    table = adata.obs[group_columns].copy()
    table["expression"] = expression
    table = table.loc[mask].copy()
    summary = (
        table.groupby(group_columns, observed=True, as_index=False)
        .agg(
            mean_log_expression=("expression", "mean"),
            fraction_expressing=("expression", lambda value: float((value > 0).mean())),
            n_cells=("expression", "size"),
        )
    )
    return summary.loc[summary["n_cells"] >= minimum_cells].copy()


def compute_module_scores(
    adata: ad.AnnData,
    output_dir: Path,
    minimum_cells: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = adata.obs.copy()
    obs["cell_state"] = obs["cell_state"].astype(str)
    obs["selection"] = obs["selection"].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    module_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    available = set(adata.var_names.astype(str))
    for module_index, (module_name, definition) in enumerate(MODULES.items()):
        genes = [gene for gene in definition["genes"] if gene in available]
        states = definition["states"]
        coverage_rows.append(
            {
                "module": module_name,
                "n_markers_defined": len(definition["genes"]),
                "n_markers_available": len(genes),
                "markers_available": ";".join(genes),
                "states": ";".join(states),
            }
        )
        if not genes:
            continue
        gene_indices = [int(adata.var_names.get_loc(gene)) for gene in genes]
        matrix = adata.X[:, gene_indices]
        if sparse.issparse(matrix):
            score = np.asarray(matrix.mean(axis=1)).ravel()
        else:
            score = np.asarray(matrix).mean(axis=1)
        state_mask = obs["cell_state"].isin(states).to_numpy()
        table = obs[
            ["donor_id", "condition", "selection", "run_id", "cell_state"]
        ].copy()
        table["module_score"] = score
        table = table.loc[state_mask].copy()
        summary = (
            table.groupby(
                ["donor_id", "condition", "selection"],
                observed=True,
                as_index=False,
            )
            .agg(
                module_score=("module_score", "mean"),
                n_cells=("module_score", "size"),
            )
        )
        summary = summary.loc[summary["n_cells"] >= minimum_cells].copy()
        summary["module"] = module_name
        summary["markers_available"] = ";".join(genes)
        module_rows.append(summary)

    capture = pd.concat(module_rows, ignore_index=True)
    capture.to_csv(output_dir / "single_cell_module_scores_capture_unit.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "single_cell_module_marker_coverage.csv",
        index=False,
    )
    unselected = capture.loc[capture["selection"].eq("unselected")].copy()
    donor = (
        unselected.groupby(
            ["donor_id", "condition", "module"],
            observed=True,
            as_index=False,
        )
        .agg(module_score=("module_score", "mean"), n_cells=("n_cells", "sum"))
    )
    donor.to_csv(output_dir / "single_cell_module_scores_unselected_donor.csv", index=False)

    effects = []
    for module_index, (module_name, sub) in enumerate(donor.groupby("module")):
        effects.append(
            effect_row(
                sub["module_score"],
                sub["condition"],
                feature=module_name,
                family="single_cell_module",
                n_bootstrap=5000,
                random_seed=RANDOM_SEED + module_index,
            )
        )
    effect_frame = add_fdr(pd.DataFrame(effects))
    effect_frame.to_csv(output_dir / "single_cell_module_effects.csv", index=False)
    return capture, donor, effect_frame


def compute_host_state_map(
    donor_modules: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    wide = donor_modules.pivot_table(
        index=["donor_id", "condition"],
        columns="module",
        values="module_score",
        aggfunc="mean",
    ).reset_index()
    module_columns = [column for column in MODULES if column in wide.columns]
    for column in module_columns:
        wide[f"{column}_z"] = safe_zscore(wide[column])
    surveillance_columns = [
        f"{column}_z"
        for column in [
            "Antigen sampling",
            "Lymphatic support",
            "Immune residency",
            "Epithelial repair",
        ]
        if f"{column}_z" in wide.columns
    ]
    wide["surveillance_state"] = wide[surveillance_columns].mean(axis=1)
    wide["injury_state"] = wide["Inflammatory injury_z"]
    wide["injury_surveillance_imbalance"] = (
        wide["injury_state"] - wide["surveillance_state"]
    )
    wide.to_csv(output_dir / "single_cell_host_state_map.csv", index=False)
    effects = []
    for index, column in enumerate(
        ["surveillance_state", "injury_state", "injury_surveillance_imbalance"]
    ):
        effects.append(
            effect_row(
                wide[column],
                wide["condition"],
                feature=column,
                family="single_cell_host_state",
                n_bootstrap=6000,
                random_seed=RANDOM_SEED + 100 + index,
            )
        )
    add_fdr(pd.DataFrame(effects)).to_csv(
        output_dir / "single_cell_host_state_effects.csv",
        index=False,
    )
    return wide


def permanova_statistic(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    grand = matrix.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in unique:
        group = matrix[labels == label]
        centroid = group.mean(axis=0)
        between += len(group) * float(np.square(centroid - grand).sum())
        within += float(np.square(group - centroid).sum())
    numerator_df = len(unique) - 1
    denominator_df = len(matrix) - len(unique)
    if within <= 0 or denominator_df <= 0:
        return float("nan")
    return (between / numerator_df) / (within / denominator_df)


def exact_permanova(
    matrix: np.ndarray,
    labels: np.ndarray,
    case_label: str = "nec",
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=object)
    observed = permanova_statistic(matrix, labels)
    n_case = int((labels == case_label).sum())
    statistics = []
    for assignment in itertools.combinations(range(len(labels)), n_case):
        permuted = np.full(len(labels), "control", dtype=object)
        permuted[list(assignment)] = case_label
        statistics.append(permanova_statistic(matrix, permuted))
    statistics_array = np.asarray(statistics, dtype=float)
    p_value = float(
        (statistics_array >= observed - 1e-12).sum() / len(statistics_array)
    )
    return {
        "pseudo_f": observed,
        "p_value": p_value,
        "n_permutations": len(statistics),
        "method": "exact_aitchison_permanova",
    }


def compute_composition_statistics(
    single_cell_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proportions = pd.read_csv(
        single_cell_dir / "cell_state_proportions_unselected_donor.csv"
    )
    counts = proportions.pivot_table(
        index=["donor_id", "condition"],
        columns="cell_state",
        values="n_cells",
        fill_value=0,
        aggfunc="sum",
    )
    counts = counts.reindex(columns=CELL_STATE_ORDER, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    closed = (counts + 0.5).div((counts + 0.5).sum(axis=1), axis=0)
    log_closed = np.log(closed)
    clr = log_closed.sub(log_closed.mean(axis=1), axis=0)

    long_rows = []
    for donor_key in counts.index:
        donor_id, condition = donor_key
        for state in CELL_STATE_ORDER:
            long_rows.append(
                {
                    "donor_id": donor_id,
                    "condition": condition,
                    "cell_state": state,
                    "n_cells": int(counts.loc[donor_key, state]),
                    "fraction": float(fractions.loc[donor_key, state]),
                    "clr_abundance": float(clr.loc[donor_key, state]),
                }
            )
    composition_long = pd.DataFrame(long_rows)
    composition_long.to_csv(
        output_dir / "single_cell_donor_composition.csv",
        index=False,
    )
    clr_reset = clr.reset_index()
    clr_reset.to_csv(output_dir / "single_cell_donor_composition_clr_wide.csv", index=False)

    model = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = model.fit_transform(clr.to_numpy())
    pca_table = counts.index.to_frame(index=False)
    pca_table["aitchison_pc1"] = coordinates[:, 0]
    pca_table["aitchison_pc2"] = coordinates[:, 1]
    pca_table.to_csv(output_dir / "single_cell_aitchison_pca.csv", index=False)
    loading_table = pd.DataFrame(
        {
            "cell_state": clr.columns,
            "pc1_loading": model.components_[0],
            "pc2_loading": model.components_[1],
        }
    )
    loading_table.to_csv(
        output_dir / "single_cell_aitchison_pca_loadings.csv",
        index=False,
    )
    permanova = exact_permanova(
        clr.to_numpy(),
        counts.index.get_level_values("condition").to_numpy(),
    )
    permanova["pc1_variance_fraction"] = float(model.explained_variance_ratio_[0])
    permanova["pc2_variance_fraction"] = float(model.explained_variance_ratio_[1])
    write_json(output_dir / "single_cell_aitchison_permanova.json", permanova)

    effects = []
    for state_index, state in enumerate(CELL_STATE_ORDER):
        sub = composition_long.loc[composition_long["cell_state"].eq(state)]
        for metric_index, metric in enumerate(["fraction", "clr_abundance"]):
            effects.append(
                effect_row(
                    sub[metric],
                    sub["condition"],
                    feature=state,
                    family=f"single_cell_composition_{metric}",
                    n_bootstrap=5000,
                    random_seed=RANDOM_SEED + state_index * 10 + metric_index,
                )
            )
    effect_frame = pd.DataFrame(effects)
    effect_frame["fdr_bh"] = (
        effect_frame.groupby("family", group_keys=False)["p_value"]
        .transform(lambda values: bh_fdr(values))
    )
    effect_frame.to_csv(
        output_dir / "single_cell_composition_effects.csv",
        index=False,
    )
    return composition_long, pca_table, effect_frame


def compute_capture_mode_sensitivity(
    single_cell_dir: Path,
    module_capture: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    proportions = pd.read_csv(single_cell_dir / "cell_state_proportions_capture_unit.csv")
    rows = []
    for state, sub in proportions.groupby("cell_state"):
        pivot = sub.pivot_table(
            index=["donor_id", "condition"],
            columns="selection",
            values="fraction",
            aggfunc="mean",
        )
        pivot = pivot.dropna(subset=["CD45_depleted", "unselected"])
        delta = pivot["CD45_depleted"] - pivot["unselected"]
        result = paired_sign_flip(delta)
        rows.append(
            {
                "family": "cell_state_fraction",
                "feature": state,
                **result,
                "deltas": ";".join(f"{value:.8g}" for value in delta),
            }
        )
    for module, sub in module_capture.groupby("module"):
        pivot = sub.pivot_table(
            index=["donor_id", "condition"],
            columns="selection",
            values="module_score",
            aggfunc="mean",
        )
        pivot = pivot.dropna(subset=["CD45_depleted", "unselected"])
        delta = pivot["CD45_depleted"] - pivot["unselected"]
        result = paired_sign_flip(delta)
        rows.append(
            {
                "family": "module_score",
                "feature": module,
                **result,
                "deltas": ";".join(f"{value:.8g}" for value in delta),
            }
        )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = (
        frame.groupby("family", group_keys=False)["p_value"]
        .transform(lambda values: bh_fdr(values))
    )
    frame.to_csv(output_dir / "capture_mode_paired_sensitivity.csv", index=False)
    return frame


def compute_mechanistic_axis_scores(
    adata: ad.AnnData,
    output_dir: Path,
    minimum_cells: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = adata.obs.copy()
    obs["cell_state"] = obs["cell_state"].astype(str)
    obs["selection"] = obs["selection"].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    group_columns = ["donor_id", "condition", "selection"]
    component_rows = []
    score_rows = []
    for axis_index, axis in enumerate(MECHANISTIC_AXES):
        source_mask = (
            obs["selection"].eq("unselected")
            & obs["cell_state"].eq(axis["source_state"])
        ).to_numpy()
        target_mask = (
            obs["selection"].eq("unselected")
            & obs["cell_state"].eq(axis["target_state"])
        ).to_numpy()
        source = expression_mean_by_group(
            adata,
            axis["source_gene"],
            source_mask,
            group_columns,
            minimum_cells=minimum_cells,
        )
        target = expression_mean_by_group(
            adata,
            axis["target_gene"],
            target_mask,
            group_columns,
            minimum_cells=minimum_cells,
        )
        if source.empty or target.empty:
            continue
        source = source.rename(
            columns={
                "mean_log_expression": "source_expression",
                "fraction_expressing": "source_fraction_expressing",
                "n_cells": "source_n_cells",
            }
        )
        target = target.rename(
            columns={
                "mean_log_expression": "target_expression",
                "fraction_expressing": "target_fraction_expressing",
                "n_cells": "target_n_cells",
            }
        )
        joined = source.merge(target, on=group_columns, how="inner")
        if joined.empty:
            continue
        joined["axis"] = axis["axis"]
        joined["axis_class"] = axis["class"]
        joined["source_state"] = axis["source_state"]
        joined["source_gene"] = axis["source_gene"]
        joined["target_state"] = axis["target_state"]
        joined["target_gene"] = axis["target_gene"]
        joined["source_expression_z"] = safe_zscore(joined["source_expression"])
        joined["target_expression_z"] = safe_zscore(joined["target_expression"])
        joined["axis_score"] = (
            joined["source_expression_z"] + joined["target_expression_z"]
        ) / 2.0
        component_rows.append(joined)
        effect = effect_row(
            joined["axis_score"],
            joined["condition"],
            feature=axis["axis"],
            family="sample_level_mechanistic_axis",
            n_bootstrap=5000,
            random_seed=RANDOM_SEED + 300 + axis_index,
        )
        effect.update(
            {
                "axis_class": axis["class"],
                "source_state": axis["source_state"],
                "source_gene": axis["source_gene"],
                "target_state": axis["target_state"],
                "target_gene": axis["target_gene"],
            }
        )
        score_rows.append(effect)
    components = pd.concat(component_rows, ignore_index=True)
    effects = add_fdr(pd.DataFrame(score_rows))
    components.to_csv(
        output_dir / "sample_level_mechanistic_axis_scores.csv",
        index=False,
    )
    effects.to_csv(
        output_dir / "sample_level_mechanistic_axis_effects.csv",
        index=False,
    )
    coverage = pd.DataFrame(MECHANISTIC_AXES)
    coverage["source_gene_available"] = coverage["source_gene"].isin(adata.var_names)
    coverage["target_gene_available"] = coverage["target_gene"].isin(adata.var_names)
    coverage["axis_scored"] = coverage["axis"].isin(effects["feature"])
    coverage.to_csv(output_dir / "mechanistic_axis_coverage.csv", index=False)
    return components, effects, coverage


def compute_tissue_statistics(
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        snapshot_results
        / "strict_mainline"
        / "spatial"
        / "tunel_lyve1_double_positive_counts.csv"
    )
    hpf = pd.read_csv(path)
    subject = (
        hpf.groupby(["id_number", "condition"], as_index=False)
        .agg(
            mean_double_positive_cells_per_hpf=(
                "double_positive_cells_per_hpf",
                "mean",
            ),
            total_double_positive_cells=("double_positive_cells_per_hpf", "sum"),
            n_hpf=("double_positive_cells_per_hpf", "size"),
        )
    )
    subject.to_csv(output_dir / "tissue_tunel_lyve1_subject_level.csv", index=False)
    hpf.to_csv(output_dir / "tissue_tunel_lyve1_hpf_source.csv", index=False)
    effects = []
    for index, metric in enumerate(
        ["mean_double_positive_cells_per_hpf", "total_double_positive_cells"]
    ):
        effects.append(
            effect_row(
                subject[metric],
                subject["condition"],
                feature=metric,
                family="tissue_tunel_lyve1",
                n_bootstrap=7000,
                random_seed=RANDOM_SEED + 400 + index,
            )
        )
    effect_frame = add_fdr(pd.DataFrame(effects))
    effect_frame.to_csv(output_dir / "tissue_tunel_lyve1_effects.csv", index=False)
    return subject, effect_frame


def read_bulk_expression(
    snapshot_results: Path,
    accession: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = snapshot_results / "bulk" / accession
    expression = pd.read_csv(directory / "expression_gene_matrix.csv")
    expression = expression.rename(columns={expression.columns[0]: "gene"})
    expression["gene"] = expression["gene"].astype(str)
    numeric_columns = expression.columns[1:]
    expression[numeric_columns] = expression[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    expression = expression.groupby("gene", as_index=False)[numeric_columns].mean()
    metadata = pd.read_csv(directory / "metadata_harmonized.csv")
    sample_column = (
        "geo_accession"
        if "geo_accession" in metadata.columns
        else metadata.columns[0]
    )
    metadata[sample_column] = metadata[sample_column].astype(str)
    metadata = metadata.set_index(sample_column, drop=False)
    return expression, metadata


def compute_bulk_target_effects(
    snapshot_results: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    contrasts = {
        "GSE46619": [("nec_vs_control", "nec", "control"), ("nec_vs_sip", "nec", "sip")],
        "GSE64801": [("nec_vs_control", "nec", "control")],
    }
    for accession, accession_contrasts in contrasts.items():
        expression, metadata = read_bulk_expression(snapshot_results, accession)
        samples = [sample for sample in expression.columns[1:] if sample in metadata.index]
        for gene_index, gene in enumerate(TARGET_GENES):
            hit = expression.loc[expression["gene"].eq(gene)]
            if hit.empty:
                continue
            values = hit.iloc[0].reindex(samples).astype(float)
            condition = metadata.loc[samples, "condition"].astype(str)
            for contrast, case, control in accession_contrasts:
                keep = condition.isin([case, control])
                if keep.sum() < 4:
                    continue
                labels = np.where(condition.loc[keep].eq(case), "nec", "control")
                row = effect_row(
                    values.loc[keep],
                    labels,
                    feature=gene,
                    family=f"bulk_{accession}_{contrast}",
                    n_bootstrap=3000,
                    random_seed=RANDOM_SEED + gene_index,
                )
                row.update(
                    {
                        "accession": accession,
                        "contrast": contrast,
                        "case_group": case,
                        "control_group": control,
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["fdr_bh_effect_family"] = (
        frame.groupby("family", group_keys=False)["p_value"]
        .transform(lambda values: bh_fdr(values))
    )
    frame.to_csv(output_dir / "bulk_target_gene_sample_effects.csv", index=False)
    return frame


def token_contains(value: str, gene: str) -> bool:
    tokens = re.split(r"[_+&:;|]", str(value).upper())
    return gene.upper() in tokens


def audit_liana_output(
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = snapshot_results / "communication_strict" / "liana_native_rank_aggregate.csv"
    columns = [
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "lr_means",
        "expr_prod",
        "lrscore",
        "specificity_rank",
        "magnitude_rank",
    ]
    targeted_rows = []
    total_rows = 0
    same_condition_rows = 0
    cross_condition_rows = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        total_rows += len(chunk)
        source_condition = chunk["source"].astype(str).str.split("|").str[0]
        target_condition = chunk["target"].astype(str).str.split("|").str[0]
        same = source_condition.eq(target_condition)
        same_condition_rows += int(same.sum())
        cross_condition_rows += int((~same).sum())
        same_chunk = chunk.loc[same].copy()
        same_chunk["condition"] = source_condition.loc[same].to_numpy()
        same_chunk["source_state"] = (
            same_chunk["source"].astype(str).str.split("|").str[1]
        )
        same_chunk["target_state"] = (
            same_chunk["target"].astype(str).str.split("|").str[1]
        )
        for axis in MECHANISTIC_AXES:
            ligand_mask = same_chunk["ligand_complex"].map(
                lambda value: token_contains(value, axis["source_gene"])
            )
            receptor_mask = same_chunk["receptor_complex"].map(
                lambda value: token_contains(value, axis["target_gene"])
            )
            hit = same_chunk.loc[ligand_mask & receptor_mask].copy()
            if not hit.empty:
                hit["axis"] = axis["axis"]
                targeted_rows.append(hit)
    targeted = (
        pd.concat(targeted_rows, ignore_index=True)
        if targeted_rows
        else pd.DataFrame(columns=columns + ["condition", "source_state", "target_state", "axis"])
    )
    if not targeted.empty:
        targeted = targeted.sort_values(
            ["axis", "condition", "lr_means"],
            ascending=[True, True, False],
        )
    targeted.to_csv(
        output_dir / "liana_same_condition_targeted_interactions.csv",
        index=False,
    )
    audit = {
        "source_file": str(path),
        "total_rows": total_rows,
        "same_condition_rows": same_condition_rows,
        "cross_condition_rows_removed": cross_condition_rows,
        "targeted_rows": int(len(targeted)),
        "axes_detected": sorted(targeted["axis"].unique().tolist())
        if not targeted.empty
        else [],
        "rank_sort_rule": "lower magnitude_rank and specificity_rank are stronger; higher lr_means and lrscore are stronger",
        "legacy_top_sort_bug": "The legacy fallback sorted lr_means ascending when consensus_rank was absent.",
    }
    write_json(output_dir / "liana_integrity_audit.json", audit)
    return targeted, audit


def extract_nichenet_targets(
    snapshot_results: Path,
    output_dir: Path,
) -> pd.DataFrame:
    directory = snapshot_results / "communication_strict"
    links = pd.read_csv(directory / "nichenet_native_ligand_target_links.csv")
    ligand_genes = sorted({axis["source_gene"] for axis in MECHANISTIC_AXES})
    target_genes = sorted(
        {axis["target_gene"] for axis in MECHANISTIC_AXES}
        | {"NLRP3", "CXCL8", "S100A8", "S100A9", "FOXP3", "CCL20", "MUC2", "REG3A"}
    )
    selected = links.loc[
        links["ligand"].astype(str).isin(ligand_genes)
        & links["target"].astype(str).isin(target_genes)
    ].copy()
    selected = selected.sort_values("weight", ascending=False)
    selected.to_csv(output_dir / "nichenet_targeted_ligand_target_weights.csv", index=False)
    return selected


def build_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "statistical_reanalysis_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "statistical_reanalysis_manifest.csv", index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    single_cell_dir = revision_root / "04_corrected_results" / "01_single_cell_corrected"
    snapshot_results = (
        revision_root / "04_corrected_results" / "00_remote_snapshot" / "results"
    )
    output_dir = (
        revision_root / "04_corrected_results" / "02_statistical_reanalysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "started",
        "generated_utc": utc_now(),
        "revision_root": str(revision_root),
        "software": {"python": sys.version, "anndata": ad.__version__},
    }
    write_json(output_dir / "host_mechanism_statistics_status.json", status)

    atlas_path = single_cell_dir / "corrected_hca_nec_atlas.h5ad"
    atlas = ad.read_h5ad(atlas_path)
    module_capture, donor_modules, module_effects = compute_module_scores(
        atlas,
        output_dir,
    )
    host_state = compute_host_state_map(donor_modules, output_dir)
    composition, pca, composition_effects = compute_composition_statistics(
        single_cell_dir,
        output_dir,
    )
    capture_sensitivity = compute_capture_mode_sensitivity(
        single_cell_dir,
        module_capture,
        output_dir,
    )
    axis_scores, axis_effects, axis_coverage = compute_mechanistic_axis_scores(
        atlas,
        output_dir,
    )
    tissue_subject, tissue_effects = compute_tissue_statistics(
        snapshot_results,
        output_dir,
    )
    bulk_effects = compute_bulk_target_effects(snapshot_results, output_dir)
    liana_targeted, liana_audit = audit_liana_output(
        snapshot_results,
        output_dir,
    )
    nichenet_selected = extract_nichenet_targets(snapshot_results, output_dir)

    status.update(
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "single_cell": {
                "cells": int(atlas.n_obs),
                "genes": int(atlas.n_vars),
                "donors": int(atlas.obs["donor_id"].nunique()),
                "runs": int(atlas.obs["run_id"].nunique()),
                "unselected_donors": int(host_state["donor_id"].nunique()),
                "module_effect_rows": int(len(module_effects)),
                "composition_effect_rows": int(len(composition_effects)),
                "mechanistic_axes_scored": axis_effects["feature"].tolist(),
                "capture_sensitivity_rows": int(len(capture_sensitivity)),
            },
            "tissue": {
                "subjects": int(len(tissue_subject)),
                "hpf_total": int(tissue_subject["n_hpf"].sum()),
            },
            "bulk_target_effect_rows": int(len(bulk_effects)),
            "liana": liana_audit,
            "nichenet_targeted_rows": int(len(nichenet_selected)),
        }
    )
    write_json(output_dir / "host_mechanism_statistics_status.json", status)
    build_manifest(output_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
