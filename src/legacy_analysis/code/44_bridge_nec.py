from __future__ import annotations

import json
import itertools
import math
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


common = import_module("40_story_chain_common")

LOCAL = common.LOCAL_REVISION_ROOT
ANALYSIS = common.ANALYSIS_ROOT
MODULE = "bridge_nec"

SPECIFIC_ROOT = ANALYSIS / "results" / "story_chain_upgrade" / "specific_nec"
UPLIFT_ROOT = ANALYSIS / "results" / "story_chain_upgrade" / "uplift_nec"
STATE_ROOT = ANALYSIS / "results" / "story_chain_upgrade" / "state_resolve"
CODER_ROOT = ANALYSIS / "results" / "coder_nec"
BULK_ROOT = (
    LOCAL
    / "04_corrected_results"
    / "00_remote_snapshot"
    / "results"
    / "bulk"
)
SC_ROOT = LOCAL / "04_corrected_results" / "02_statistical_reanalysis"

MODULE_TO_STATE = {
    "Antigen sampling": "M cell",
    "Lymphatic support": "LEC",
    "Immune residency": "Treg",
    "Epithelial repair": "epithelial",
    "Inflammatory injury": "myeloid",
}
EXPECTED_DIRECTION = {
    "Antigen sampling": -1.0,
    "Lymphatic support": -1.0,
    "Immune residency": -1.0,
    "Epithelial repair": -1.0,
    "Inflammatory injury": 1.0,
}
UPLIFT_STATES = {"M cell", "LEC", "Treg"}


def hedges_variance(g_value: float, n_case: int, n_control: int) -> float:
    if n_case < 2 or n_control < 2 or not np.isfinite(g_value):
        return np.nan
    total = n_case + n_control
    return float(
        total / (n_case * n_control)
        + (g_value**2) / (2.0 * max(total - 2, 1))
    )


def parse_modules() -> dict[str, list[str]]:
    frame = pd.read_csv(
        SC_ROOT / "single_cell_module_marker_coverage.csv"
    )
    return {
        row.module: str(row.markers_available).upper().split(";")
        for row in frame.itertuples()
    }


def donor_conditions() -> pd.DataFrame:
    frame = pd.read_csv(
        LOCAL
        / "10_reviewer_revision"
        / "02_corrected_analysis"
        / "single_cell_donor_characteristics.csv"
    )
    frame["donor_id"] = frame["donor_id"].astype(str)
    return frame[["donor_id", "condition"]]


def read_broad_state_logcpm(state: str) -> pd.DataFrame:
    path = SC_ROOT / f"single_cell_pseudobulk_target_logcpm_{state}.csv"
    frame = pd.read_csv(path)
    long = frame.melt(id_vars="gene", var_name="donor_state", value_name="log2_cpm")
    long["donor_id"] = (
        long["donor_state"].str.extract(r"D(\d+)")[0].astype(str)
    )
    long["gene"] = long["gene"].astype(str).str.upper()
    return long


def single_cell_gene_values(
    modules: dict[str, list[str]],
) -> pd.DataFrame:
    conditions = donor_conditions()
    uplift = pd.read_csv(
        UPLIFT_ROOT / "uplift_target_gene_donor_logcpm.csv"
    )
    uplift = uplift.loc[
        uplift["method"].eq("Posterior expectation")
    ].copy()
    uplift["donor_id"] = uplift["donor_id"].astype(str)
    uplift["gene"] = uplift["gene"].astype(str).str.upper()
    rows = []
    for module, genes in modules.items():
        state = MODULE_TO_STATE[module]
        if state in UPLIFT_STATES:
            values = uplift.loc[
                uplift["target_state"].eq(state)
                & uplift["gene"].isin(genes),
                ["donor_id", "condition", "gene", "log2_cpm"],
            ].copy()
        else:
            values = read_broad_state_logcpm(state)
            values = values.loc[values["gene"].isin(genes)].merge(
                conditions,
                on="donor_id",
                how="left",
            )
        values["module"] = module
        values["source"] = "HCA NEC atlas"
        values["contrast"] = "NEC versus surgical control"
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def bulk_expression(accession: str) -> pd.DataFrame:
    frame = pd.read_csv(
        CODER_ROOT / f"coder_{accession}_gene_expression.csv"
    )
    frame = frame.rename(columns={frame.columns[0]: "gene"})
    frame["gene"] = frame["gene"].astype(str).str.upper()
    return frame.drop_duplicates("gene").set_index("gene")


def bulk_conditions(accession: str) -> pd.Series:
    frame = pd.read_csv(BULK_ROOT / accession / "metadata_harmonized.csv")
    return frame.set_index("geo_accession")["condition"].astype(str).str.lower()


def bulk_residual_gene_values(
    accession: str,
    modules: dict[str, list[str]],
    comparator: str,
) -> pd.DataFrame:
    expression = bulk_expression(accession)
    conditions = bulk_conditions(accession)
    scores = pd.read_csv(
        STATE_ROOT / "state_resolve_composition_pc_scores.csv"
    )
    scores = scores.loc[scores["accession"].eq(accession)].set_index("sample")
    samples = [
        sample
        for sample in expression.columns
        if sample in conditions.index and sample in scores.index
    ]
    rows = []
    for module, genes in modules.items():
        for gene in genes:
            if gene not in expression.index:
                continue
            values = expression.loc[gene, samples].to_numpy(dtype=float)
            for sample, value in zip(samples, values, strict=True):
                condition = conditions.loc[sample]
                if condition not in {"nec", comparator}:
                    continue
                rows.append(
                    {
                        "sample_id": sample,
                        "condition": condition,
                        "gene": gene,
                        "log2_cpm": float(value),
                        "module": module,
                        "source": accession,
                        "contrast": (
                            "NEC versus SIP"
                            if comparator == "sip"
                            else "NEC versus control"
                        ),
                        "PC1": float(scores.loc[sample, "PC1"]),
                        "PC2": float(scores.loc[sample, "PC2"]),
                    }
                )
    return pd.DataFrame(rows)


def module_scores(gene_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["source", "contrast", "module"]
    for keys, group in gene_values.groupby(group_columns, sort=False):
        source, contrast, module = keys
        id_column = (
            "donor_id"
            if "donor_id" in group and group["donor_id"].notna().any()
            else "sample_id"
        )
        pivot = group.pivot_table(
            index=[id_column, "condition"],
            columns="gene",
            values="log2_cpm",
            aggfunc="mean",
        )
        standard_deviation = pivot.std(axis=0, ddof=0).replace(0.0, np.nan)
        z = ((pivot - pivot.mean(axis=0)) / standard_deviation).fillna(0.0)
        aligned = z.mean(axis=1) * EXPECTED_DIRECTION[module]
        for (sample_id, condition), value in aligned.items():
            sample_rows = group.loc[
                group[id_column].astype(str).eq(str(sample_id))
            ]
            rows.append(
                {
                    "source": source,
                    "contrast": contrast,
                    "module": module,
                    "sample_id": str(sample_id),
                    "condition": condition,
                    "aligned_nec_direction_score": float(value),
                    "genes_available": int(z.shape[1]),
                    "PC1": float(sample_rows["PC1"].iloc[0])
                    if "PC1" in sample_rows
                    and sample_rows["PC1"].notna().any()
                    else np.nan,
                    "PC2": float(sample_rows["PC2"].iloc[0])
                    if "PC2" in sample_rows
                    and sample_rows["PC2"].notna().any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def adjusted_standardized_effect(
    values: np.ndarray,
    labels: np.ndarray,
    covariates: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    covariates = np.asarray(covariates, dtype=float)
    finite = np.isfinite(y) & np.isfinite(covariates).all(axis=1)
    y = y[finite]
    labels = labels[finite]
    covariates = covariates[finite]
    y_sd = np.std(y, ddof=0)
    if y_sd <= 0 or labels.sum() < 2 or (~labels).sum() < 2:
        return {
            "effect": np.nan,
            "standard_error": np.nan,
            "p_value": np.nan,
            "assignments": 0,
        }
    y = (y - np.mean(y)) / y_sd
    covariates = np.column_stack(
        [
            (covariates[:, index] - np.mean(covariates[:, index]))
            / np.std(covariates[:, index], ddof=0)
            if np.std(covariates[:, index], ddof=0) > 0
            else np.zeros(len(covariates))
            for index in range(covariates.shape[1])
        ]
    )

    def coefficient(permuted_labels: np.ndarray) -> tuple[float, float]:
        design = np.column_stack(
            [np.ones(len(y)), permuted_labels.astype(float), covariates]
        )
        inverse = np.linalg.pinv(design.T @ design)
        beta = inverse @ design.T @ y
        residual = y - design @ beta
        leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
        adjusted_residual = residual / np.clip(1.0 - leverage, 1e-8, None)
        meat = design.T @ np.diag(adjusted_residual**2) @ design
        covariance = inverse @ meat @ inverse
        return float(beta[1]), float(math.sqrt(max(covariance[1, 1], 0.0)))

    observed, standard_error = coefficient(labels)
    null = []
    n_case = int(labels.sum())
    for selected in itertools.combinations(range(len(y)), n_case):
        permuted = np.zeros(len(y), dtype=bool)
        permuted[list(selected)] = True
        null.append(coefficient(permuted)[0])
    null = np.asarray(null, dtype=float)
    return {
        "effect": observed,
        "standard_error": standard_error,
        "p_value": float(
            np.mean(np.abs(null) >= abs(observed) - np.finfo(float).eps)
        ),
        "assignments": int(len(null)),
    }


def source_effects(
    gene_values: pd.DataFrame,
    score_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_rows = []
    for keys, group in gene_values.groupby(
        ["source", "contrast", "module", "gene"],
        sort=False,
    ):
        source, contrast, module, gene = keys
        case = group.loc[group["condition"].eq("nec"), "log2_cpm"].dropna()
        control = group.loc[group["condition"].ne("nec"), "log2_cpm"].dropna()
        if group["PC1"].notna().any() if "PC1" in group else False:
            adjusted = adjusted_standardized_effect(
                group["log2_cpm"].to_numpy(dtype=float),
                group["condition"].eq("nec").to_numpy(),
                group[["PC1", "PC2"]].to_numpy(dtype=float),
            )
            raw_effect = adjusted["effect"]
            aligned_effect = raw_effect * EXPECTED_DIRECTION[module]
            standard_error = adjusted["standard_error"]
            permutation_pvalue = adjusted["p_value"]
            assignments = adjusted["assignments"]
            effect_method = "Composition-adjusted standardized coefficient"
        else:
            raw_effect = common.hedges_g(case, control)
            aligned_effect = raw_effect * EXPECTED_DIRECTION[module]
            exact = common.exact_label_permutation(
                group["log2_cpm"].to_numpy(dtype=float)
                * EXPECTED_DIRECTION[module],
                group["condition"].eq("nec").to_numpy(),
            )
            variance = hedges_variance(aligned_effect, len(case), len(control))
            standard_error = (
                math.sqrt(variance) if np.isfinite(variance) else np.nan
            )
            permutation_pvalue = exact["p_value"]
            assignments = exact["assignments"]
            effect_method = "Hedges g"
        gene_rows.append(
            {
                "source": source,
                "contrast": contrast,
                "module": module,
                "gene": gene,
                "n_nec": len(case),
                "n_comparator": len(control),
                "raw_standardized_effect_nec_vs_comparator": raw_effect,
                "aligned_nec_direction_standardized_effect": aligned_effect,
                "standard_error": standard_error,
                "permutation_pvalue": permutation_pvalue,
                "permutation_assignments": assignments,
                "effect_method": effect_method,
            }
        )
    module_rows = []
    for keys, group in score_values.groupby(
        ["source", "contrast", "module"],
        sort=False,
    ):
        source, contrast, module = keys
        case = group.loc[
            group["condition"].eq("nec"), "aligned_nec_direction_score"
        ].dropna()
        control = group.loc[
            group["condition"].ne("nec"), "aligned_nec_direction_score"
        ].dropna()
        if group["PC1"].notna().any():
            adjusted = adjusted_standardized_effect(
                group["aligned_nec_direction_score"].to_numpy(dtype=float),
                group["condition"].eq("nec").to_numpy(),
                group[["PC1", "PC2"]].to_numpy(dtype=float),
            )
            effect = adjusted["effect"]
            standard_error = adjusted["standard_error"]
            permutation_pvalue = adjusted["p_value"]
            assignments = adjusted["assignments"]
            effect_method = "Composition-adjusted standardized coefficient"
        else:
            effect = common.hedges_g(case, control)
            exact = common.exact_label_permutation(
                group["aligned_nec_direction_score"].to_numpy(dtype=float),
                group["condition"].eq("nec").to_numpy(),
            )
            variance = hedges_variance(effect, len(case), len(control))
            standard_error = (
                math.sqrt(variance) if np.isfinite(variance) else np.nan
            )
            permutation_pvalue = exact["p_value"]
            assignments = exact["assignments"]
            effect_method = "Hedges g"
        module_rows.append(
            {
                "source": source,
                "contrast": contrast,
                "module": module,
                "n_nec": len(case),
                "n_comparator": len(control),
                "genes_available": int(group["genes_available"].max()),
                "aligned_nec_direction_standardized_effect": effect,
                "standard_error": standard_error,
                "permutation_pvalue": permutation_pvalue,
                "permutation_assignments": assignments,
                "effect_method": effect_method,
            }
        )
    return pd.DataFrame(gene_rows), pd.DataFrame(module_rows)


def meta_analyze(
    source_effects_frame: pd.DataFrame,
    grouping_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    loo_rows = []
    for feature, group in source_effects_frame.groupby(grouping_column, sort=False):
        meta = common.random_effects_meta(
            group["aligned_nec_direction_standardized_effect"],
            group["standard_error"],
        )
        signs = []
        for omitted_source in group["source"].unique():
            keep = group.loc[group["source"].ne(omitted_source)]
            leave_out = common.random_effects_meta(
                keep["aligned_nec_direction_standardized_effect"],
                keep["standard_error"],
            )
            signs.append(np.sign(leave_out["effect"]))
            loo_rows.append(
                {
                    grouping_column: feature,
                    "omitted_source": omitted_source,
                    "sources_retained": int(keep["source"].nunique()),
                    "pooled_aligned_effect": leave_out["effect"],
                    "ci_low": leave_out["ci_low"],
                    "ci_high": leave_out["ci_high"],
                    "p_value": leave_out["p_value"],
                }
            )
        observed_sign = np.sign(meta["effect"])
        rows.append(
            {
                grouping_column: feature,
                **meta,
                "sources": int(group["source"].nunique()),
                "source_names": ";".join(group["source"].unique()),
                "source_sign_concordance": float(
                    np.mean(
                        np.sign(group["aligned_nec_direction_standardized_effect"])
                        == observed_sign
                    )
                ),
                "leave_one_source_out_sign_stability": float(
                    np.mean(np.asarray(signs) == observed_sign)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["fdr_bh"] = common.bh_fdr(summary["p_value"])
    return summary, pd.DataFrame(loo_rows)


def source_concordance(gene_effects: pd.DataFrame) -> pd.DataFrame:
    matrix = gene_effects.pivot_table(
        index="gene",
        columns="source",
        values="aligned_nec_direction_standardized_effect",
        aggfunc="first",
    )
    rows = []
    sources = matrix.columns.tolist()
    for index, source_a in enumerate(sources):
        for source_b in sources[index + 1 :]:
            pair = matrix[[source_a, source_b]].dropna()
            rho, p_value = stats.spearmanr(pair[source_a], pair[source_b])
            rows.append(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "genes_compared": len(pair),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "direction_concordance": float(
                        np.mean(np.sign(pair[source_a]) == np.sign(pair[source_b]))
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = common.bh_fdr(frame["p_value"])
    return frame


def main() -> None:
    modules = parse_modules()
    single_cell = single_cell_gene_values(modules)
    gse46619_sip = bulk_residual_gene_values(
        "GSE46619",
        modules,
        comparator="sip",
    )
    gse64801 = bulk_residual_gene_values(
        "GSE64801",
        modules,
        comparator="control",
    )
    primary_values = pd.concat(
        [single_cell, gse46619_sip, gse64801],
        ignore_index=True,
        sort=False,
    )
    score_values = module_scores(primary_values)
    gene_effects, module_effects = source_effects(primary_values, score_values)
    gene_meta, gene_loo = meta_analyze(gene_effects, "gene")
    module_meta, module_loo = meta_analyze(module_effects, "module")
    concordance = source_concordance(gene_effects)

    gse46619_control = bulk_residual_gene_values(
        "GSE46619",
        modules,
        comparator="control",
    )
    sensitivity_scores = module_scores(gse46619_control)
    sensitivity_gene_effects, sensitivity_module_effects = source_effects(
        gse46619_control,
        sensitivity_scores,
    )

    common.write_csv(
        primary_values,
        MODULE,
        "bridge_primary_sample_gene_values.csv",
    )
    common.write_csv(
        score_values,
        MODULE,
        "bridge_primary_sample_module_scores.csv",
    )
    common.write_csv(
        gene_effects,
        MODULE,
        "bridge_primary_gene_source_effects.csv",
    )
    common.write_csv(
        module_effects,
        MODULE,
        "bridge_primary_module_source_effects.csv",
    )
    common.write_csv(gene_meta, MODULE, "bridge_gene_random_effects_meta.csv")
    common.write_csv(module_meta, MODULE, "bridge_module_random_effects_meta.csv")
    common.write_csv(gene_loo, MODULE, "bridge_gene_leave_one_source_out.csv")
    common.write_csv(module_loo, MODULE, "bridge_module_leave_one_source_out.csv")
    common.write_csv(concordance, MODULE, "bridge_source_concordance.csv")
    common.write_csv(
        sensitivity_gene_effects,
        MODULE,
        "bridge_GSE46619_nec_control_gene_sensitivity.csv",
    )
    common.write_csv(
        sensitivity_module_effects,
        MODULE,
        "bridge_GSE46619_nec_control_module_sensitivity.csv",
    )

    summary = {
        "analysis": "BRIDGE-NEC",
        "status": "complete",
        "primary_independent_sources": gene_effects["source"].unique().tolist(),
        "primary_contrasts": gene_effects["contrast"].unique().tolist(),
        "genes_meta_analyzed": int(len(gene_meta)),
        "modules_meta_analyzed": int(len(module_meta)),
        "gene_meta_fdr_below_0_05": int(gene_meta["fdr_bh"].lt(0.05).sum()),
        "module_meta_fdr_below_0_05": int(module_meta["fdr_bh"].lt(0.05).sum()),
        "genes_with_complete_source_sign_concordance": int(
            gene_meta["source_sign_concordance"].eq(1.0).sum()
        ),
        "genes_with_complete_leave_one_source_out_sign_stability": int(
            gene_meta["leave_one_source_out_sign_stability"].eq(1.0).sum()
        ),
        "method_notes": [
            "Primary sources are independent: 11 HCA donors, GSE46619 NEC versus SIP, and GSE64801 NEC versus control.",
            "Bulk standardized coefficients include CoDeR CLR composition PC1 and PC2 as covariates.",
            "Module scores average within-source standardized expression after alignment to the prespecified NEC-associated direction.",
            "Random-effects models combine standardized donor effects and their sampling uncertainty without pooling biological units.",
            "GSE46619 NEC versus non-inflammatory control is retained as a sensitivity contrast and is not double-counted in the primary meta-analysis.",
        ],
    }
    common.write_json(summary, MODULE, "bridge_nec_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
