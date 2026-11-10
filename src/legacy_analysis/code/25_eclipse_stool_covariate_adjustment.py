#!/usr/bin/env python3
"""Replace the ECLIPSE stool analysis with covariate-adjusted inference.

Corrected gestational age, postnatal day, sex and race are available for all
16 stool samples and differ between primary NEC and control groups. This
script fits feature-wise OLS models on M-values, rebuilds a nested
leave-one-subject-out NEC axis without age leakage and refreshes all
downstream cross-tissue promoter and single-cell integration results.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ECLIPSE_CODE = ROOT / "code" / "16_eclipse_nec.py"
PUBLIC_ROOT = ROOT / "data" / "public" / "eclipse_nec"
PROCESSED_ROOT = ROOT / "data" / "processed" / "eclipse_nec"
RESULT_ROOT = ROOT / "results" / "eclipse_nec"
MANIFEST_ROOT = ROOT / "manifests"
TOP_WINDOWS = 500


def ensure_local(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(
            f"Refusing write outside local analysis root: {resolved}"
        )
    return resolved


def load_eclipse_module():
    specification = importlib.util.spec_from_file_location(
        "eclipse_covariate_source", ECLIPSE_CODE
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_corrected_gestational_age(value: object) -> float:
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def prepare_stool_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    stool = metadata[metadata["tissue"].eq("stool")].copy()
    stool["gsm"] = stool["gsm"].astype(str)
    stool["cga_weeks"] = stool[
        "corrected_gestational_age_at_sampling"
    ].map(parse_corrected_gestational_age)
    stool["postnatal_day"] = pd.to_numeric(
        stool["days_of_life_at_sampling"], errors="raise"
    )
    stool["female"] = stool["sex"].astype(str).str.casefold().eq("female").astype(float)
    stool["race_aa"] = stool["race"].astype(str).str.upper().eq("AA").astype(float)
    required = [
        "cga_weeks",
        "postnatal_day",
        "female",
        "race_aa",
    ]
    if stool[required].isna().any().any():
        raise RuntimeError("Stool covariates are incomplete")
    return stool


def reduced_design(
    metadata: pd.DataFrame,
    reference_positions: np.ndarray,
) -> np.ndarray:
    continuous = metadata[["cga_weeks", "postnatal_day"]].to_numpy(
        dtype=float
    )
    reference = continuous[reference_positions]
    mean = reference.mean(axis=0)
    standard_deviation = reference.std(axis=0, ddof=1)
    if np.any(~np.isfinite(standard_deviation)) or np.any(
        standard_deviation <= 0
    ):
        raise RuntimeError("Continuous stool covariate is constant")
    scaled = (continuous - mean) / standard_deviation
    return np.column_stack(
        [
            np.ones(len(metadata)),
            scaled,
            metadata["female"].to_numpy(dtype=float),
            metadata["race_aa"].to_numpy(dtype=float),
        ]
    )


def full_design(
    metadata: pd.DataFrame,
    reference_positions: np.ndarray,
) -> np.ndarray:
    reduced = reduced_design(metadata, reference_positions)
    condition = metadata["analysis_condition"].eq("nec").to_numpy(dtype=float)
    return np.column_stack([reduced[:, 0], condition, reduced[:, 1:]])


def featurewise_condition_ols(
    values: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit condition coefficient for rows with arbitrary missingness."""
    if values.shape[1] != design.shape[0]:
        raise ValueError("Value and design sample dimensions differ")
    row_count, sample_count = values.shape
    coefficient = np.full(row_count, np.nan, dtype=float)
    pvalue = np.ones(row_count, dtype=float)
    residual_sd = np.full(row_count, np.nan, dtype=float)
    observed_samples = np.zeros(row_count, dtype=int)

    valid = np.isfinite(values)
    powers = (2 ** np.arange(sample_count, dtype=np.int64))[None, :]
    missingness_code = np.sum(valid.astype(np.int64) * powers, axis=1)
    for code in np.unique(missingness_code):
        rows = np.flatnonzero(missingness_code == code)
        observed = valid[rows[0]]
        x = design[observed]
        rank = int(np.linalg.matrix_rank(x))
        degrees_freedom = int(x.shape[0] - rank)
        if degrees_freedom <= 0 or rank < 2:
            continue
        y = values[np.ix_(rows, observed)].T
        gram_inverse = np.linalg.pinv(x.T @ x)
        beta = gram_inverse @ x.T @ y
        residual = y - x @ beta
        variance = np.sum(residual**2, axis=0) / degrees_freedom
        standard_error = np.sqrt(
            np.maximum(variance * gram_inverse[1, 1], 0.0)
        )
        estimable = standard_error > 0
        statistic = np.divide(
            beta[1],
            standard_error,
            out=np.zeros(len(rows), dtype=float),
            where=estimable,
        )
        coefficient[rows] = beta[1]
        pvalue[rows] = np.where(
            estimable,
            2 * stats.t.sf(np.abs(statistic), degrees_freedom),
            1.0,
        )
        residual_sd[rows] = np.sqrt(variance)
        observed_samples[rows] = x.shape[0]
    return coefficient, pvalue, residual_sd, observed_samples


def inverse_m_value(values: np.ndarray) -> np.ndarray:
    ratio = np.power(2.0, values)
    beta = (1.01 * ratio - 0.01) / (1.0 + ratio)
    return np.clip(beta, 0.0, 1.0)


def residualize_selected(
    values_all: np.ndarray,
    design_all: np.ndarray,
    training_positions: np.ndarray,
) -> np.ndarray:
    """Residualize selected M-value features using training covariates."""
    result = np.full_like(values_all, np.nan, dtype=float)
    x_train_all = design_all[training_positions]
    for feature in range(values_all.shape[1]):
        y_train = values_all[training_positions, feature]
        observed_train = np.isfinite(y_train)
        x_train = x_train_all[observed_train]
        if x_train.shape[0] <= np.linalg.matrix_rank(x_train):
            continue
        coefficient = np.linalg.lstsq(
            x_train, y_train[observed_train], rcond=None
        )[0]
        observed_all = np.isfinite(values_all[:, feature])
        prediction = design_all[observed_all] @ coefficient
        result[observed_all, feature] = (
            values_all[observed_all, feature] - prediction
        )
    train = result[training_positions]
    mean = np.nanmean(train, axis=0)
    standard_deviation = np.nanstd(train, axis=0, ddof=1)
    standard_deviation[
        ~np.isfinite(standard_deviation) | (standard_deviation <= 0)
    ] = np.nan
    standardized = (result - mean) / standard_deviation
    return np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def adjusted_stool_effects(
    eclipse,
    stool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta, coverage, cpgs = eclipse.build_tissue_matrix(stool)
    primary = stool[
        stool["analysis_condition"].isin(["nec", "control"])
    ].copy()
    primary_samples = primary["gsm"].tolist()
    case_samples = primary.loc[
        primary["analysis_condition"].eq("nec"), "gsm"
    ].tolist()
    control_samples = primary.loc[
        primary["analysis_condition"].eq("control"), "gsm"
    ].tolist()
    minimum_case = max(
        3,
        int(math.ceil(eclipse.MIN_GROUP_FRACTION * len(case_samples))),
    )
    minimum_control = max(
        3,
        int(math.ceil(eclipse.MIN_GROUP_FRACTION * len(control_samples))),
    )
    keep = (
        beta[case_samples].notna().sum(axis=1) >= minimum_case
    ) & (
        beta[control_samples].notna().sum(axis=1) >= minimum_control
    )
    beta = beta.loc[keep]
    coverage = coverage.loc[keep]
    cpgs = cpgs.loc[keep]
    beta_primary = beta[primary_samples]
    beta_values = beta_primary.to_numpy(dtype=float)
    m_values = np.log2(
        (np.clip(beta_values, 0, 1) + 0.01)
        / (1 - np.clip(beta_values, 0, 1) + 0.01)
    )
    primary = primary.set_index("gsm").loc[primary_samples].reset_index()
    reference_positions = np.arange(len(primary))
    design = full_design(primary, reference_positions)
    coefficient, pvalue, residual_sd, observed = featurewise_condition_ols(
        m_values, design
    )

    old = pd.read_parquet(
        RESULT_ROOT / "eclipse_stool_window_effects.parquet"
    )
    old = old.set_index(["chromosome", "bin_start"]).reindex(beta.index)
    if old.isna().all(axis=1).any() or len(old) != len(beta):
        raise RuntimeError(
            "Adjusted stool window set does not match baseline analysis"
        )
    old = old.reset_index()
    if "unadjusted_delta_beta" not in old.columns:
        old["unadjusted_delta_beta"] = old["delta_beta"]
    if "unadjusted_welch_pvalue" not in old.columns:
        old["unadjusted_welch_pvalue"] = old["welch_pvalue"]
    if "unadjusted_welch_fdr" not in old.columns:
        old["unadjusted_welch_fdr"] = old["welch_fdr"]
    old["condition_coefficient_m_value"] = coefficient
    old["condition_residual_sd"] = residual_sd
    old["model_observed_samples"] = observed
    control_mean = beta_primary[control_samples].mean(axis=1).to_numpy(
        dtype=float
    )
    baseline_m = np.log2(
        (np.clip(control_mean, 0, 1) + 0.01)
        / (1 - np.clip(control_mean, 0, 1) + 0.01)
    )
    adjusted_case_beta = inverse_m_value(baseline_m + coefficient)
    old["delta_beta"] = adjusted_case_beta - control_mean
    old["welch_pvalue"] = pvalue
    old["welch_fdr"] = eclipse.bh_adjust(pvalue)
    old["empirical_pvalue"] = np.nan
    old["inference_model"] = (
        "OLS M-value: NEC + corrected gestational age + postnatal day "
        "+ sex + race"
    )
    old["mean_coverage"] = np.nanmean(
        coverage[primary_samples].to_numpy(dtype=float), axis=1
    )
    old["median_cpgs"] = np.nanmedian(
        cpgs[primary_samples].to_numpy(dtype=float), axis=1
    )
    old.sort_values(
        ["welch_pvalue", "chromosome", "bin_start"], inplace=True
    )
    return old, beta, primary


def nested_stool_projection(
    effects: pd.DataFrame,
    beta: pd.DataFrame,
    stool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    primary = stool[
        stool["analysis_condition"].isin(["nec", "control"])
    ].copy()
    primary_samples = primary["gsm"].tolist()
    all_samples = stool["gsm"].tolist()
    primary_all_positions = np.asarray(
        [all_samples.index(sample) for sample in primary_samples],
        dtype=int,
    )
    beta_primary = beta[primary_samples]
    m_primary = np.log2(
        (np.clip(beta_primary.to_numpy(dtype=float), 0, 1) + 0.01)
        / (1 - np.clip(beta_primary.to_numpy(dtype=float), 0, 1) + 0.01)
    )
    condition = primary["analysis_condition"].to_numpy()
    loo_scores: dict[str, float] = {}
    for held_index, held_sample in enumerate(primary_samples):
        training_positions = np.asarray(
            [
                index
                for index in range(len(primary_samples))
                if index != held_index
            ],
            dtype=int,
        )
        training_all_positions = primary_all_positions[
            training_positions
        ]
        training_metadata = primary.iloc[training_positions].reset_index(
            drop=True
        )
        training_design = full_design(
            training_metadata,
            np.arange(len(training_metadata)),
        )
        _, fold_pvalue, _, _ = featurewise_condition_ols(
            m_primary[:, training_positions],
            training_design,
        )
        selected_positions = np.argsort(fold_pvalue)[:TOP_WINDOWS]

        selected_beta = beta.iloc[selected_positions][all_samples].T
        selected_m = np.log2(
            (np.clip(selected_beta.to_numpy(dtype=float), 0, 1) + 0.01)
            / (1 - np.clip(selected_beta.to_numpy(dtype=float), 0, 1) + 0.01)
        )
        all_design = reduced_design(stool, training_all_positions)
        standardized = residualize_selected(
            selected_m,
            all_design,
            training_all_positions,
        )
        training_condition = condition[training_positions]
        axis = (
            standardized[
                training_all_positions[training_condition == "nec"]
            ].mean(axis=0)
            - standardized[
                training_all_positions[training_condition == "control"]
            ].mean(axis=0)
        )
        norm = float(np.linalg.norm(axis))
        if norm <= 0 or not np.isfinite(norm):
            loo_scores[held_sample] = np.nan
            continue
        axis /= norm
        held_all_position = all_samples.index(held_sample)
        loo_scores[held_sample] = float(
            standardized[held_all_position] @ axis
        )

    selected_effects = effects.nsmallest(TOP_WINDOWS, "welch_pvalue")
    selected_index = pd.MultiIndex.from_frame(
        selected_effects[["chromosome", "bin_start"]]
    )
    selected_beta = beta.reindex(selected_index)[all_samples].T
    selected_m = np.log2(
        (np.clip(selected_beta.to_numpy(dtype=float), 0, 1) + 0.01)
        / (1 - np.clip(selected_beta.to_numpy(dtype=float), 0, 1) + 0.01)
    )
    all_design = reduced_design(stool, primary_all_positions)
    standardized = residualize_selected(
        selected_m,
        all_design,
        primary_all_positions,
    )
    axis = (
        standardized[
            primary_all_positions[condition == "nec"]
        ].mean(axis=0)
        - standardized[
            primary_all_positions[condition == "control"]
        ].mean(axis=0)
    )
    axis /= np.linalg.norm(axis)
    scores = standardized @ axis

    score_frame = stool.copy()
    score_frame["eclipse_nec_axis_score"] = scores
    score_frame["leave_one_subject_out_score"] = score_frame["gsm"].map(
        loo_scores
    )
    sampling = pd.to_numeric(
        score_frame["days_of_life_at_sampling"], errors="coerce"
    )
    onset = pd.to_numeric(
        score_frame["days_of_life_at_nec_onset"], errors="coerce"
    )
    score_frame["days_relative_to_nec_onset"] = sampling - onset
    validation = score_frame[
        score_frame["analysis_condition"].isin(["nec", "control"])
    ].dropna(subset=["leave_one_subject_out_score"])
    labels = validation["analysis_condition"].eq("nec").astype(int)
    auc = float(
        roc_auc_score(labels, validation["leave_one_subject_out_score"])
    )

    pca = PCA(n_components=2, random_state=20260726)
    pca_scores = pca.fit_transform(standardized)
    pca_frame = score_frame[
        [
            "gsm",
            "subject_id",
            "analysis_condition",
            "days_relative_to_nec_onset",
            "cga_weeks",
            "postnatal_day",
        ]
    ].copy()
    pca_frame["PC1"] = pca_scores[:, 0]
    pca_frame["PC2"] = pca_scores[:, 1]
    pca_frame["PC1_variance"] = float(
        pca.explained_variance_ratio_[0]
    )
    pca_frame["PC2_variance"] = float(
        pca.explained_variance_ratio_[1]
    )
    summary = {
        "selected_windows": TOP_WINDOWS,
        "primary_samples": len(primary_samples),
        "future_nec_samples": int(
            stool["analysis_condition"].eq("future_nec").sum()
        ),
        "future_nec_subjects": int(
            stool.loc[
                stool["analysis_condition"].eq("future_nec"),
                "subject_id",
            ].nunique()
        ),
        "nested_covariate_adjusted_leave_one_subject_out_auc": auc,
        "PC1_variance": float(pca.explained_variance_ratio_[0]),
        "PC2_variance": float(pca.explained_variance_ratio_[1]),
    }
    return score_frame, pca_frame, summary


def refresh_downstream(
    eclipse,
    stool_effects: pd.DataFrame,
    projection_summary: dict[str, object],
) -> dict[str, object]:
    effect_tables = {
        cohort: pd.read_parquet(
            RESULT_ROOT / f"eclipse_{cohort}_window_effects.parquet"
        )
        for cohort in eclipse.COHORTS
    }
    effect_tables["stool"] = stool_effects
    window_meta = eclipse.cross_tissue_meta(effect_tables)
    genes = pd.read_parquet(
        PROCESSED_ROOT / "gencode_v46_gene_promoters.parquet"
    )
    stool_promoter = eclipse.promoter_effects(
        "stool", stool_effects, genes
    )
    promoter_tables = {}
    for cohort in eclipse.COHORTS:
        if cohort == "stool":
            promoter_tables[cohort] = stool_promoter
        else:
            promoter_tables[cohort] = pd.read_parquet(
                RESULT_ROOT
                / f"eclipse_{cohort}_promoter_effects.parquet"
            )
    promoter_meta = eclipse.cross_tissue_promoter_meta(promoter_tables)
    integrated, module_summary, key_loci = eclipse.single_cell_integration(
        promoter_meta
    )
    return {
        "cross_tissue_windows": int(len(window_meta)),
        "cross_tissue_fdr_0_05_windows": int(
            window_meta["meta_fdr"].lt(0.05).sum()
        ),
        "promoter_genes": int(promoter_meta["gene"].nunique()),
        "promoter_fdr_0_05_genes": int(
            promoter_meta["meta_fdr"].lt(0.05).sum()
        ),
        "single_cell_gene_state_pairs": int(len(integrated)),
        "module_state_pairs": int(len(module_summary)),
        "key_locus_state_pairs": int(len(key_loci)),
        "stool_projection": projection_summary,
    }


def main() -> None:
    eclipse = load_eclipse_module()
    metadata = pd.read_csv(
        PUBLIC_ROOT / "eclipse_sample_metadata.csv",
        low_memory=False,
    )
    stool = prepare_stool_metadata(metadata)
    effects, beta, primary = adjusted_stool_effects(
        eclipse, stool
    )
    effect_path = ensure_local(
        RESULT_ROOT / "eclipse_stool_window_effects.parquet"
    )
    effects.to_parquet(effect_path, index=False, compression="zstd")
    effects.head(2_000).to_csv(
        ensure_local(
            RESULT_ROOT / "eclipse_stool_top_window_effects.csv"
        ),
        index=False,
    )

    score_frame, pca_frame, projection_summary = nested_stool_projection(
        effects, beta, stool
    )
    score_frame.to_csv(
        ensure_local(
            RESULT_ROOT / "eclipse_stool_nec_axis_projection.csv"
        ),
        index=False,
    )
    pca_frame.to_csv(
        ensure_local(
            RESULT_ROOT / "eclipse_stool_methylation_pca.csv"
        ),
        index=False,
    )
    refresh = refresh_downstream(
        eclipse, effects, projection_summary
    )

    baseline_age = (
        primary.groupby("analysis_condition")[
            ["cga_weeks", "postnatal_day"]
        ]
        .mean()
        .reset_index()
    )
    baseline_path = ensure_local(
        RESULT_ROOT / "eclipse_stool_covariate_balance.csv"
    )
    baseline_age.to_csv(baseline_path, index=False)
    summary = {
        "status": "PASS",
        "analysis": "ECLIPSE stool covariate adjustment",
        "primary_model": (
            "feature-wise OLS on M-values with NEC condition, corrected "
            "gestational age, postnatal day, sex and race"
        ),
        "primary_samples": int(len(primary)),
        "retained_windows": int(len(effects)),
        "adjusted_fdr_0_05_windows": int(
            effects["welch_fdr"].lt(0.05).sum()
        ),
        "unadjusted_fdr_0_05_windows": int(
            effects["unadjusted_welch_fdr"].lt(0.05).sum()
        ),
        "minimum_adjusted_pvalue": float(
            effects["welch_pvalue"].min()
        ),
        "projection": projection_summary,
        "downstream_refresh": refresh,
        "cloud_files_written": 0,
    }
    summary_path = ensure_local(
        RESULT_ROOT / "eclipse_stool_covariate_adjustment_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    manifest_path = ensure_local(
        MANIFEST_ROOT / "eclipse_stool_covariate_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    main_manifest_path = MANIFEST_ROOT / "eclipse_analysis_manifest.json"
    if main_manifest_path.exists():
        main_manifest = json.loads(
            main_manifest_path.read_text(encoding="utf-8")
        )
    else:
        cohort_summaries = []
        for cohort in eclipse.COHORTS:
            cohort_effects = pd.read_parquet(
                RESULT_ROOT
                / f"eclipse_{cohort}_window_effects.parquet"
            )
            cohort_summaries.append(
                {
                    "cohort": cohort,
                    "case_condition": str(
                        cohort_effects["case_condition"].iloc[0]
                    ),
                    "comparator_condition": str(
                        cohort_effects["comparator_condition"].iloc[0]
                    ),
                    "case_samples": int(
                        cohort_effects["case_samples"].iloc[0]
                    ),
                    "comparator_samples": int(
                        cohort_effects["comparator_samples"].iloc[0]
                    ),
                    "retained_windows": int(len(cohort_effects)),
                    "fdr_0_05_windows": int(
                        cohort_effects["welch_fdr"].lt(0.05).sum()
                    ),
                    "absolute_delta_0_10_windows": int(
                        cohort_effects["delta_beta"].abs().ge(0.10).sum()
                    ),
                }
            )
        main_manifest = {
            "status": "PASS",
            "analysis": "ECLIPSE-NEC",
            "algorithm": (
                "Full-depth 1-kb Bismark count aggregation; "
                "tissue-specific sample-level inference; weighted "
                "Stouffer cross-tissue meta-analysis; promoter ACAT; "
                "single-cell expression-methylation concordance"
            ),
            "statistical_unit": "biological sample or subject",
            "cohorts": cohort_summaries,
            "cloud_files_written": 0,
        }
    for cohort_summary in main_manifest.get("cohorts", []):
        if cohort_summary.get("cohort") == "stool":
            cohort_summary["fdr_0_05_windows"] = summary[
                "adjusted_fdr_0_05_windows"
            ]
            cohort_summary["retained_windows"] = summary[
                "retained_windows"
            ]
            cohort_summary["inference_model"] = summary[
                "primary_model"
            ]
            cohort_summary["empirical_method"] = (
                "not applied to covariate-adjusted OLS"
            )
    main_manifest.update(refresh)
    main_manifest["stool_covariate_adjustment"] = summary
    main_manifest["algorithm"] = (
        main_manifest.get("algorithm", "")
        + "; stool feature-wise covariate-adjusted OLS and nested "
        "leave-one-subject-out residualized projection"
    )
    main_manifest_path.write_text(
        json.dumps(main_manifest, indent=2),
        encoding="utf-8",
    )
    print(
        "ECLIPSE_STOOL_COVARIATE_COMPLETE "
        f"windows={len(effects)} "
        f"fdr={summary['adjusted_fdr_0_05_windows']} "
        f"auc={projection_summary['nested_covariate_adjusted_leave_one_subject_out_auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
