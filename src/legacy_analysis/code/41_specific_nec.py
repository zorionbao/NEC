from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module


common = import_module("40_story_chain_common")

LOCAL = common.LOCAL_REVISION_ROOT
ANALYSIS = common.ANALYSIS_ROOT
MODULE = "specific_nec"
RNG = np.random.default_rng(common.SEED)

DONOR_PATH = (
    LOCAL
    / "10_reviewer_revision"
    / "02_corrected_analysis"
    / "single_cell_donor_characteristics.csv"
)
HOST_PATH = (
    LOCAL
    / "06_source_data"
    / "01_Figure_1"
    / "Figure_1d_host_state_map.csv"
)
CLR_PATH = (
    LOCAL
    / "04_corrected_results"
    / "02_statistical_reanalysis"
    / "single_cell_donor_composition_clr_wide.csv"
)
MODULE_PATH = (
    LOCAL
    / "04_corrected_results"
    / "02_statistical_reanalysis"
    / "single_cell_module_marker_coverage.csv"
)
ALL_CELL_DE_PATH = (
    LOCAL
    / "04_corrected_results"
    / "02_statistical_reanalysis"
    / "single_cell_pseudobulk_edger_all_cells.csv"
)
BULK_ROOT = (
    LOCAL
    / "04_corrected_results"
    / "00_remote_snapshot"
    / "results"
    / "bulk"
)


FEATURE_MAP = {
    "Regulatory T-cell abundance": "T_reg",
    "T and NK-cell abundance": "T_NK",
    "M-cell abundance": "M_cell",
    "Lymphatic endothelial abundance": "LEC",
    "Antigen sampling": "Antigen sampling",
    "Lymphatic support": "Lymphatic support",
    "Immune residency": "Immune residency",
    "Epithelial repair": "Epithelial repair",
    "Inflammatory injury": "Inflammatory injury",
    "Surveillance state": "surveillance_state",
    "Injury state": "injury_state",
    "Injury-surveillance imbalance": "injury_surveillance_imbalance",
}

EXPECTED_DIRECTION = {
    "Antigen sampling": -1.0,
    "Lymphatic support": -1.0,
    "Immune residency": -1.0,
    "Epithelial repair": -1.0,
    "Inflammatory injury": 1.0,
}


def build_donor_outcomes() -> pd.DataFrame:
    donor = pd.read_csv(DONOR_PATH)
    host = pd.read_csv(HOST_PATH)
    clr = pd.read_csv(CLR_PATH)
    frame = donor.merge(host, on=["donor_id", "condition"], how="inner")
    frame = frame.merge(clr, on=["donor_id", "condition"], how="left")
    frame["postconception_age_weeks"] = (
        frame["gestational_age_weeks"] + frame["age_at_surgery_days"] / 7.0
    )
    return frame


def overlap_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for covariate in [
        "gestational_age_weeks",
        "age_at_surgery_days",
        "postconception_age_weeks",
    ]:
        control = frame.loc[frame["condition"].eq("control"), covariate].dropna()
        nec = frame.loc[frame["condition"].eq("nec"), covariate].dropna()
        lower = max(float(control.min()), float(nec.min()))
        upper = min(float(control.max()), float(nec.max()))
        in_support = frame[covariate].between(lower, upper, inclusive="both")
        rows.append(
            {
                "covariate": covariate,
                "control_min": float(control.min()),
                "control_max": float(control.max()),
                "nec_min": float(nec.min()),
                "nec_max": float(nec.max()),
                "common_support_lower": lower,
                "common_support_upper": upper,
                "common_support_exists": bool(lower <= upper),
                "control_in_common_support": int(
                    (in_support & frame["condition"].eq("control")).sum()
                ),
                "nec_in_common_support": int(
                    (in_support & frame["condition"].eq("nec")).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def specificity_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjustment_sets = {
        "Unadjusted": [],
        "Gestational age residualized": ["gestational_age_weeks"],
        "Surgery age residualized": ["age_at_surgery_days"],
        "Postconception age residualized": ["postconception_age_weeks"],
    }
    rows = []
    loo_rows = []
    for feature, column in FEATURE_MAP.items():
        for adjustment, covariates in adjustment_sets.items():
            work = frame.dropna(subset=[column, "condition", *covariates]).copy()
            values = work[column].to_numpy(dtype=float)
            if covariates:
                values = common.residualize(
                    values,
                    work[covariates].to_numpy(dtype=float),
                )
            exact = common.exact_label_permutation(
                values,
                work["condition"].eq("nec").to_numpy(),
            )
            loo = common.leave_one_out_effects(
                frame.dropna(subset=[column, "condition", *covariates]),
                column,
                covariates,
            )
            observed_sign = np.sign(exact["effect"])
            stable = (
                float(np.mean(np.sign(loo["effect"]) == observed_sign))
                if observed_sign != 0
                else np.nan
            )
            rows.append(
                {
                    "feature": feature,
                    "source_column": column,
                    "adjustment": adjustment,
                    "n_donors": len(work),
                    "n_nec": int(work["condition"].eq("nec").sum()),
                    "n_control": int(work["condition"].eq("control").sum()),
                    "hedges_g_nec_vs_control": exact["effect"],
                    "exact_permutation_pvalue": exact["p_value"],
                    "exact_assignments": exact["assignments"],
                    "leave_one_donor_out_sign_stability": stable,
                    "leave_one_donor_out_effect_min": float(loo["effect"].min()),
                    "leave_one_donor_out_effect_max": float(loo["effect"].max()),
                }
            )
            loo["feature"] = feature
            loo["adjustment"] = adjustment
            loo_rows.append(loo)
    effects = pd.DataFrame(rows)
    effects["fdr_bh_within_adjustment"] = effects.groupby(
        "adjustment", sort=False
    )["exact_permutation_pvalue"].transform(common.bh_fdr)
    return effects, pd.concat(loo_rows, ignore_index=True)


def parse_module_genes() -> dict[str, list[str]]:
    coverage = pd.read_csv(MODULE_PATH)
    return {
        row.module: str(row.markers_available).split(";")
        for row in coverage.itertuples()
    }


def matched_null_specificity(
    module_genes: dict[str, list[str]],
    draws: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de = pd.read_csv(ALL_CELL_DE_PATH)
    de["gene"] = de["gene"].astype(str).str.upper()
    de = de.dropna(subset=["gene", "logFC", "logCPM"]).drop_duplicates("gene")
    de["expression_bin"] = pd.qcut(
        de["logCPM"],
        q=min(10, de["logCPM"].nunique()),
        labels=False,
        duplicates="drop",
    )
    by_bin = {
        int(bin_id): genes["gene"].tolist()
        for bin_id, genes in de.groupby("expression_bin")
    }
    indexed = de.set_index("gene")
    summary_rows = []
    null_rows = []
    for module, genes in module_genes.items():
        available = [gene for gene in genes if gene in indexed.index]
        direction = EXPECTED_DIRECTION[module]
        observed = float(
            np.mean(indexed.loc[available, "logFC"].to_numpy(dtype=float) * direction)
        )
        null = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled: list[str] = []
            for gene in available:
                bin_id = int(indexed.loc[gene, "expression_bin"])
                candidates = [
                    item for item in by_bin[bin_id] if item not in available
                ]
                sampled.append(str(RNG.choice(candidates)))
            null[draw] = float(
                np.mean(indexed.loc[sampled, "logFC"].to_numpy(dtype=float) * direction)
            )
        p_value = float((1 + np.sum(null >= observed)) / (draws + 1))
        percentile = float(np.mean(null <= observed))
        summary_rows.append(
            {
                "module": module,
                "expected_nec_direction": "increase" if direction > 0 else "decrease",
                "genes_defined": len(genes),
                "genes_available": len(available),
                "genes": ";".join(available),
                "observed_aligned_mean_log2_fold_change": observed,
                "matched_null_mean": float(np.mean(null)),
                "matched_null_standard_deviation": float(np.std(null, ddof=1)),
                "matched_null_percentile": percentile,
                "one_sided_matched_null_pvalue": p_value,
                "null_draws": draws,
            }
        )
        quantiles = np.quantile(null, np.linspace(0.0, 1.0, 101))
        null_rows.extend(
            {
                "module": module,
                "quantile": index / 100.0,
                "aligned_mean_log2_fold_change": float(value),
            }
            for index, value in enumerate(quantiles)
        )
    summary = pd.DataFrame(summary_rows)
    summary["fdr_bh"] = common.bh_fdr(summary["one_sided_matched_null_pvalue"])
    return summary, pd.DataFrame(null_rows)


def standardize_de(path: Path, dataset: str, contrast: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "logfc": "log2_fold_change",
        "logFC": "log2_fold_change",
        "pvalue": "p_value",
        "PValue": "p_value",
        "fdr_bh": "fdr_bh",
        "FDR": "fdr_bh",
    }
    frame = frame.rename(columns=rename)
    frame["gene"] = frame["gene"].astype(str).str.upper()
    frame["dataset"] = dataset
    frame["contrast"] = contrast
    return frame[
        [
            "dataset",
            "contrast",
            "gene",
            "log2_fold_change",
            "p_value",
            "fdr_bh",
        ]
    ]


def disease_control_specificity(
    module_genes: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [
        standardize_de(
            BULK_ROOT / "GSE46619" / "differential_expression.csv",
            "GSE46619",
            "NEC versus non-inflammatory surgical control",
        ),
        standardize_de(
            BULK_ROOT / "GSE46619" / "differential_expression_nec_vs_sip.csv",
            "GSE46619",
            "NEC versus spontaneous intestinal perforation",
        ),
        standardize_de(
            BULK_ROOT / "GSE64801" / "differential_expression.csv",
            "GSE64801",
            "NEC versus non-NEC preterm control",
        ),
    ]
    all_de = pd.concat(tables, ignore_index=True)
    gene_to_module = {
        gene: module
        for module, genes in module_genes.items()
        for gene in genes
    }
    target = all_de.loc[all_de["gene"].isin(gene_to_module)].copy()
    target["module"] = target["gene"].map(gene_to_module)
    target["expected_direction_multiplier"] = target["module"].map(
        EXPECTED_DIRECTION
    )
    target["aligned_log2_fold_change"] = (
        target["log2_fold_change"] * target["expected_direction_multiplier"]
    )
    target["direction_matches_module"] = target["aligned_log2_fold_change"].gt(0)
    rows = []
    for (dataset, contrast, module), group in target.groupby(
        ["dataset", "contrast", "module"],
        sort=False,
    ):
        rows.append(
            {
                "dataset": dataset,
                "contrast": contrast,
                "module": module,
                "genes_available": len(group),
                "mean_aligned_log2_fold_change": float(
                    group["aligned_log2_fold_change"].mean()
                ),
                "median_aligned_log2_fold_change": float(
                    group["aligned_log2_fold_change"].median()
                ),
                "direction_concordance": float(
                    group["direction_matches_module"].mean()
                ),
                "minimum_gene_pvalue": float(group["p_value"].min()),
                "minimum_gene_fdr": float(group["fdr_bh"].min()),
                "genes_with_fdr_below_0_05": int(group["fdr_bh"].lt(0.05).sum()),
            }
        )
    return target.sort_values(
        ["dataset", "contrast", "module", "p_value"]
    ), pd.DataFrame(rows)


def main() -> None:
    frame = build_donor_outcomes()
    overlap = overlap_diagnostics(frame)
    effects, loo = specificity_effects(frame)
    module_genes = parse_module_genes()
    null_summary, null_quantiles = matched_null_specificity(module_genes)
    bulk_genes, bulk_modules = disease_control_specificity(module_genes)

    common.write_csv(frame, MODULE, "specific_nec_donor_analysis_table.csv")
    common.write_csv(overlap, MODULE, "specific_nec_age_overlap_diagnostics.csv")
    common.write_csv(effects, MODULE, "specific_nec_donor_effects.csv")
    common.write_csv(loo, MODULE, "specific_nec_leave_one_donor_out.csv")
    common.write_csv(
        null_summary,
        MODULE,
        "specific_nec_matched_null_module_summary.csv",
    )
    common.write_csv(
        null_quantiles,
        MODULE,
        "specific_nec_matched_null_quantiles.csv",
    )
    common.write_csv(
        bulk_genes,
        MODULE,
        "specific_nec_bulk_disease_control_gene_effects.csv",
    )
    common.write_csv(
        bulk_modules,
        MODULE,
        "specific_nec_bulk_disease_control_module_summary.csv",
    )

    input_paths = [
        DONOR_PATH,
        HOST_PATH,
        CLR_PATH,
        MODULE_PATH,
        ALL_CELL_DE_PATH,
        BULK_ROOT / "GSE46619" / "differential_expression.csv",
        BULK_ROOT / "GSE46619" / "differential_expression_nec_vs_sip.csv",
        BULK_ROOT / "GSE64801" / "differential_expression.csv",
    ]
    summary = {
        "analysis": "SPECIFIC-NEC",
        "status": "complete",
        "independent_donors": int(frame["donor_id"].nunique()),
        "nec_donors": int(frame["condition"].eq("nec").sum()),
        "control_donors": int(frame["condition"].eq("control").sum()),
        "exact_label_assignments": 462,
        "common_support": overlap.to_dict(orient="records"),
        "minimum_unadjusted_fdr": float(
            effects.loc[
                effects["adjustment"].eq("Unadjusted"),
                "fdr_bh_within_adjustment",
            ].min()
        ),
        "minimum_age_residualized_fdr": float(
            effects.loc[
                effects["adjustment"].ne("Unadjusted"),
                "fdr_bh_within_adjustment",
            ].min()
        ),
        "matched_null_modules_fdr_below_0_05": int(
            null_summary["fdr_bh"].lt(0.05).sum()
        ),
        "bulk_target_gene_rows": int(len(bulk_genes)),
        "input_sha256": {
            str(path.relative_to(common.WORKSPACE_ROOT)): common.sha256(path)
            for path in input_paths
        },
        "method_notes": [
            "All host effects use independent donors and exhaustive label permutations.",
            "Age sensitivity residualizes the outcome before the same exhaustive permutation test.",
            "Leave-one-donor-out stability reports directional sensitivity rather than pseudo-replication.",
            "Matched null modules preserve all-cell mean-expression deciles.",
            "GSE46619 includes both NEC-control and NEC-SIP contrasts.",
        ],
    }
    common.write_json(summary, MODULE, "specific_nec_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
