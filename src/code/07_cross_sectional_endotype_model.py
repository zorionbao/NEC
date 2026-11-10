from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from revision_stats import RANDOM_SEED, safe_zscore


FEATURE_BUDGETS = {
    "Species logistic": 3,
    "Pathway logistic": 3,
    "Combined XGBoost": 5,
}
CV_REPEATS = 20
PERMUTATION_COUNTS = {
    "Species logistic": 200,
    "Pathway logistic": 500,
}
SENSITIVITY_FEATURE_BUDGETS = (2, 3, 4, 5, 6, 8)


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


def make_logistic_pipeline(n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=n_features)),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_xgboost_pipeline(n_features: int, random_seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=n_features)),
            (
                "model",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=80,
                    max_depth=1,
                    learning_rate=0.05,
                    min_child_weight=1,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_lambda=2.0,
                    reg_alpha=0.15,
                    n_jobs=1,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def repeated_oof_predictions(
    model: Pipeline,
    matrix: pd.DataFrame,
    outcome: np.ndarray,
    n_splits: int = 3,
    n_repeats: int = CV_REPEATS,
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
    )
    rows = []
    for split_index, (train, test) in enumerate(splitter.split(matrix, outcome)):
        repeat = split_index // n_splits
        fold = split_index % n_splits
        fitted = clone(model)
        fitted.fit(matrix.iloc[train], outcome[train])
        probability = fitted.predict_proba(matrix.iloc[test])[:, 1]
        for sample_index, prediction in zip(test, probability):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "observed": int(outcome[sample_index]),
                    "predicted_probability": float(prediction),
                }
            )
    return pd.DataFrame(rows)


def leave_one_infant_out_predictions(
    model: Pipeline,
    matrix: pd.DataFrame,
    outcome: np.ndarray,
) -> pd.DataFrame:
    rows = []
    splitter = LeaveOneOut()
    for fold, (train, test) in enumerate(splitter.split(matrix, outcome)):
        fitted = clone(model)
        fitted.fit(matrix.iloc[train], outcome[train])
        probability = fitted.predict_proba(matrix.iloc[test])[:, 1]
        rows.append(
            {
                "sample_index": int(test[0]),
                "fold": int(fold),
                "observed": int(outcome[test[0]]),
                "predicted_probability": float(probability[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_index")


def stratified_bootstrap_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int = 6000,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    positive = np.flatnonzero(outcome == 1)
    negative = np.flatnonzero(outcome == 0)
    auc_values = np.empty(n_bootstrap, dtype=float)
    ap_values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        auc_values[index] = roc_auc_score(outcome[sampled], probability[sampled])
        ap_values[index] = average_precision_score(
            outcome[sampled],
            probability[sampled],
        )
    return {
        "roc_auc_ci_low": float(np.quantile(auc_values, 0.025)),
        "roc_auc_ci_high": float(np.quantile(auc_values, 0.975)),
        "average_precision_ci_low": float(np.quantile(ap_values, 0.025)),
        "average_precision_ci_high": float(np.quantile(ap_values, 0.975)),
    }


def summarize_model(
    model_name: str,
    infant_table: pd.DataFrame,
    outcome: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability = infant_table["predicted_probability"].to_numpy(float)
    roc_auc = float(roc_auc_score(outcome, probability))
    average_precision = float(average_precision_score(outcome, probability))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(outcome, probability)
    precision, recall, pr_thresholds = precision_recall_curve(outcome, probability)
    finite_thresholds = np.isfinite(thresholds)
    youden = true_positive_rate - false_positive_rate
    valid_indices = np.flatnonzero(finite_thresholds)
    threshold_index = valid_indices[np.argmax(youden[finite_thresholds])]
    selected_threshold = float(thresholds[threshold_index])
    predicted = (probability >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(outcome, predicted, labels=[0, 1]).ravel()
    metric = {
        "model": model_name,
        "n_samples": int(len(outcome)),
        "n_infants": int(len(outcome)),
        "n_nec": int(outcome.sum()),
        "n_control": int((outcome == 0).sum()),
        "analysis_unit": "infant",
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(outcome, probability)),
        "youden_threshold": selected_threshold,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else float("nan"),
        **stratified_bootstrap_metrics(outcome, probability),
    }
    roc_table = pd.DataFrame(
        {
            "model": model_name,
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )
    pr_table = pd.DataFrame(
        {
            "model": model_name,
            "recall": recall,
            "precision": precision,
            "threshold": np.r_[pr_thresholds, np.nan],
        }
    )
    try:
        observed, predicted_probability = calibration_curve(
            outcome,
            probability,
            n_bins=3,
            strategy="quantile",
        )
        calibration = pd.DataFrame(
            {
                "model": model_name,
                "mean_predicted_probability": predicted_probability,
                "observed_nec_fraction": observed,
            }
        )
    except ValueError:
        calibration = pd.DataFrame(
            columns=["model", "mean_predicted_probability", "observed_nec_fraction"]
        )
    return metric, roc_table, pr_table, calibration


def permutation_auc_test(
    matrix: pd.DataFrame,
    outcome: np.ndarray,
    observed_auc: float,
    model_name: str,
    n_features: int,
    n_permutations: int = 1000,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + 900)
    auc_values = np.empty(n_permutations, dtype=float)
    for permutation_index in range(n_permutations):
        permuted = rng.permutation(outcome)
        splitter = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_SEED + permutation_index,
        )
        probability = np.zeros(len(outcome), dtype=float)
        for train, test in splitter.split(matrix, permuted):
            if model_name in {"Species logistic", "Pathway logistic"}:
                model = make_logistic_pipeline(min(n_features, matrix.shape[1]))
            else:
                model = make_xgboost_pipeline(
                    n_features=min(n_features, matrix.shape[1]),
                    random_seed=RANDOM_SEED + permutation_index,
                )
            model.fit(matrix.iloc[train], permuted[train])
            probability[test] = model.predict_proba(matrix.iloc[test])[:, 1]
        auc_values[permutation_index] = roc_auc_score(permuted, probability)
    p_value = float((1 + (auc_values >= observed_auc - 1e-12).sum()) / (n_permutations + 1))
    return p_value, auc_values


def final_xgboost_explanations(
    matrix: pd.DataFrame,
    outcome: np.ndarray,
    infant_ids: pd.Series,
    conditions: pd.Series,
    output_dir: Path,
    n_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    model = make_xgboost_pipeline(min(n_features, matrix.shape[1]))
    model.fit(matrix, outcome)
    variance_values = model.named_steps["variance"].transform(matrix)
    variance_features = matrix.columns[
        model.named_steps["variance"].get_support()
    ].tolist()
    scaled = model.named_steps["scale"].transform(variance_values)
    selected_values = model.named_steps["select"].transform(scaled)
    support = model.named_steps["select"].get_support()
    selected_features = [
        feature for feature, keep in zip(variance_features, support) if keep
    ]
    booster = model.named_steps["model"].get_booster()
    contributions = booster.predict(
        xgb.DMatrix(selected_values, feature_names=selected_features),
        pred_contribs=True,
    )
    shap_values = contributions[:, :-1]
    shap_long = pd.DataFrame(shap_values, columns=selected_features)
    shap_long.insert(0, "condition", conditions.to_numpy())
    shap_long.insert(0, "infant_id", infant_ids.to_numpy())
    shap_long = shap_long.melt(
        id_vars=["infant_id", "condition"],
        var_name="feature",
        value_name="shap_value",
    )
    mean_absolute = (
        shap_long.groupby("feature", as_index=False)
        .agg(mean_absolute_shap=("shap_value", lambda value: float(np.mean(np.abs(value)))))
        .sort_values("mean_absolute_shap", ascending=False)
    )
    shap_long = shap_long.merge(mean_absolute, on="feature", how="left")
    shap_long.to_csv(output_dir / "xgboost_shap_values.csv", index=False)
    mean_absolute.to_csv(output_dir / "xgboost_shap_importance.csv", index=False)

    selected_standardized = pd.DataFrame(
        selected_values,
        index=matrix.index,
        columns=selected_features,
    )
    top_features = mean_absolute.head(14)["feature"].tolist()
    heatmap = selected_standardized[top_features].copy()
    for column in heatmap:
        heatmap[column] = safe_zscore(heatmap[column])
    heatmap.insert(0, "condition", conditions.to_numpy())
    heatmap.insert(0, "infant_id", infant_ids.to_numpy())
    heatmap_long = heatmap.melt(
        id_vars=["infant_id", "condition"],
        var_name="feature",
        value_name="feature_zscore",
    )
    heatmap_long.to_csv(output_dir / "xgboost_endotype_heatmap_long.csv", index=False)
    return shap_long, heatmap_long, model


def pathway_logistic_stability(
    matrix: pd.DataFrame,
    outcome: np.ndarray,
    infant_ids: pd.Series,
    conditions: pd.Series,
    output_dir: Path,
    n_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )
    rows = []
    n_fits = 0
    for split_index, (train, _) in enumerate(splitter.split(matrix, outcome)):
        model = make_logistic_pipeline(min(n_features, matrix.shape[1]))
        model.fit(matrix.iloc[train], outcome[train])
        variance_features = matrix.columns[
            model.named_steps["variance"].get_support()
        ].tolist()
        selected = model.named_steps["select"].get_support()
        selected_features = [
            feature for feature, keep in zip(variance_features, selected) if keep
        ]
        coefficients = model.named_steps["model"].coef_[0]
        for feature, coefficient in zip(selected_features, coefficients):
            rows.append(
                {
                    "split_index": split_index,
                    "repeat": split_index // 3,
                    "fold": split_index % 3,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                }
            )
        n_fits += 1
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "pathway_logistic_fold_coefficients.csv", index=False)
    stability = (
        raw.groupby("feature", as_index=False)
        .agg(
            selected_folds=("standardized_coefficient", "size"),
            median_coefficient=("standardized_coefficient", "median"),
            coefficient_q25=("standardized_coefficient", lambda value: float(np.quantile(value, 0.25))),
            coefficient_q75=("standardized_coefficient", lambda value: float(np.quantile(value, 0.75))),
            positive_sign_fraction=("standardized_coefficient", lambda value: float((value > 0).mean())),
        )
    )
    stability["selection_frequency"] = stability["selected_folds"] / n_fits
    stability["stability_score"] = (
        stability["selection_frequency"] * stability["median_coefficient"].abs()
    )
    stability = stability.sort_values("stability_score", ascending=False)
    stability.to_csv(
        output_dir / "pathway_logistic_coefficient_stability.csv",
        index=False,
    )
    top_features = stability.head(14)["feature"].tolist()
    heatmap = matrix[top_features].copy()
    for column in heatmap:
        heatmap[column] = safe_zscore(heatmap[column])
    heatmap.insert(0, "condition", conditions.to_numpy())
    heatmap.insert(0, "infant_id", infant_ids.to_numpy())
    heatmap_long = heatmap.melt(
        id_vars=["infant_id", "condition"],
        var_name="feature",
        value_name="feature_zscore",
    )
    heatmap_long.to_csv(
        output_dir / "pathway_logistic_endotype_heatmap_long.csv",
        index=False,
    )
    return stability, heatmap_long


def feature_budget_sensitivity(
    matrices: dict[str, pd.DataFrame],
    outcome: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_name, matrix in matrices.items():
        budgets = (
            (3, 5, 8)
            if model_name == "Combined XGBoost"
            else SENSITIVITY_FEATURE_BUDGETS
        )
        n_repeats = 2 if model_name == "Combined XGBoost" else 5
        for budget in budgets:
            n_features = min(budget, matrix.shape[1])
            if model_name == "Combined XGBoost":
                model = make_xgboost_pipeline(n_features)
            else:
                model = make_logistic_pipeline(n_features)
            raw = repeated_oof_predictions(
                model,
                matrix,
                outcome,
                n_splits=3,
                n_repeats=n_repeats,
            )
            aggregate = (
                raw.groupby("sample_index", as_index=False)
                .agg(
                    observed=("observed", "first"),
                    predicted_probability=("predicted_probability", "mean"),
                )
                .sort_values("sample_index")
            )
            rows.append(
                {
                    "model": model_name,
                    "feature_budget": n_features,
                    "roc_auc": float(
                        roc_auc_score(
                            aggregate["observed"],
                            aggregate["predicted_probability"],
                        )
                    ),
                    "average_precision": float(
                        average_precision_score(
                            aggregate["observed"],
                            aggregate["predicted_probability"],
                        )
                    ),
                    "brier_score": float(
                        brier_score_loss(
                            aggregate["observed"],
                            aggregate["predicted_probability"],
                        )
                    ),
                    "n_repeats": n_repeats,
                    "analysis_unit": "infant",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["model", "feature_budget"])


def decision_curve_analysis(
    outcome: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    thresholds = np.arange(0.05, 0.91, 0.05)
    prevalence = float(outcome.mean())
    rng = np.random.default_rng(RANDOM_SEED + 1200)
    positive = np.flatnonzero(outcome == 1)
    negative = np.flatnonzero(outcome == 0)
    rows = []
    for threshold in thresholds:
        odds = threshold / (1.0 - threshold)
        predicted = probability >= threshold
        true_positive = int(((outcome == 1) & predicted).sum())
        false_positive = int(((outcome == 0) & predicted).sum())
        model_net_benefit = (
            true_positive / len(outcome) - false_positive / len(outcome) * odds
        )
        bootstrap = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            sampled = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            sampled_outcome = outcome[sampled]
            sampled_prediction = probability[sampled] >= threshold
            sampled_tp = int(
                ((sampled_outcome == 1) & sampled_prediction).sum()
            )
            sampled_fp = int(
                ((sampled_outcome == 0) & sampled_prediction).sum()
            )
            bootstrap[index] = (
                sampled_tp / len(sampled)
                - sampled_fp / len(sampled) * odds
            )
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        rows.extend(
            [
                {
                    "strategy": "Pathway logistic",
                    "threshold_probability": threshold,
                    "net_benefit": model_net_benefit,
                    "ci_low": float(low),
                    "ci_high": float(high),
                },
                {
                    "strategy": "Treat all",
                    "threshold_probability": threshold,
                    "net_benefit": prevalence - (1.0 - prevalence) * odds,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                },
                {
                    "strategy": "Treat none",
                    "threshold_probability": threshold,
                    "net_benefit": 0.0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def choose_trigger_features(
    reanalysis_dir: Path,
) -> tuple[list[str], list[str]]:
    species_effects = pd.read_csv(
        reanalysis_dir / "metaphlan_species_effects_corrected.csv"
    )
    preferred_species = [
        "Klebsiella_pneumoniae",
        "Klebsiella_aerogenes",
        "Clostridium_neonatale",
        "Escherichia_coli",
    ]
    selected_species = [
        f"species::{feature}"
        for feature in preferred_species
        if feature in set(species_effects["species"])
    ]
    pathway_effects = pd.read_csv(reanalysis_dir / "humann_pathway_selected_panel.csv")
    positive_targeted = pathway_effects.loc[
        pathway_effects["selection_reason"].eq("targeted_metabolic_axis")
        & pathway_effects["hedges_g"].gt(0)
    ].sort_values("hedges_g", ascending=False)
    selected_pathways = [
        f"pathway::{feature}" for feature in positive_targeted.head(4)["pathway"]
    ]
    return selected_species, selected_pathways


def build_endotype_map(
    feature_matrix: pd.DataFrame,
    probability: pd.Series,
    reanalysis_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    species_features, pathway_features = choose_trigger_features(reanalysis_dir)
    selected_species = [feature for feature in species_features if feature in feature_matrix]
    selected_pathways = [feature for feature in pathway_features if feature in feature_matrix]
    species_z = pd.DataFrame(
        {feature: safe_zscore(feature_matrix[feature]) for feature in selected_species},
        index=feature_matrix.index,
    )
    pathway_z = pd.DataFrame(
        {feature: safe_zscore(feature_matrix[feature]) for feature in selected_pathways},
        index=feature_matrix.index,
    )
    endotype = pd.DataFrame(index=feature_matrix.index)
    endotype["species_trigger_score"] = species_z.mean(axis=1)
    endotype["pathway_trigger_score"] = pathway_z.mean(axis=1)
    endotype["cross_validated_nec_probability"] = probability.reindex(
        feature_matrix.index
    )
    endotype["species_features"] = ";".join(selected_species)
    endotype["pathway_features"] = ";".join(selected_pathways)
    endotype.to_csv(output_dir / "microbial_functional_endotype_map.csv")
    return endotype


def select_effect(
    frame: pd.DataFrame,
    mask: pd.Series,
    domain: str,
    label: str,
    source: str,
    effect_column: str = "hedges_g",
    low_column: str = "hedges_g_ci_low",
    high_column: str = "hedges_g_ci_high",
    p_column: str = "p_value",
    fdr_column: str = "fdr_bh",
) -> dict[str, Any] | None:
    hit = frame.loc[mask]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "domain": domain,
        "result": label,
        "source": source,
        "hedges_g": row.get(effect_column, np.nan),
        "hedges_g_ci_low": row.get(low_column, np.nan),
        "hedges_g_ci_high": row.get(high_column, np.nan),
        "p_value": row.get(p_column, np.nan),
        "fdr_bh": row.get(fdr_column, np.nan),
    }


def build_cross_domain_effect_summary(
    reanalysis_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    host = pd.read_csv(reanalysis_dir / "single_cell_host_state_effects.csv")
    composition = pd.read_csv(reanalysis_dir / "single_cell_composition_effects.csv")
    axes = pd.read_csv(reanalysis_dir / "sample_level_mechanistic_axis_effects.csv")
    tissue = pd.read_csv(reanalysis_dir / "tissue_tunel_lyve1_effects.csv")
    bulk = pd.read_csv(reanalysis_dir / "bulk_target_gene_sample_effects.csv")
    species = pd.read_csv(reanalysis_dir / "metaphlan_species_effects_corrected.csv")
    pathways = pd.read_csv(reanalysis_dir / "humann_pathway_effects_corrected.csv")
    metabolites = pd.read_csv(reanalysis_dir / "target_metabolite_patient_effects.csv")

    selections = [
        select_effect(
            host,
            host["feature"].eq("surveillance_state"),
            "Host",
            "Surveillance state",
            "HCA NEC atlas, unselected donors",
        ),
        select_effect(
            host,
            host["feature"].eq("injury_state"),
            "Host",
            "Inflammatory injury state",
            "HCA NEC atlas, unselected donors",
        ),
        select_effect(
            composition,
            composition["family"].eq("single_cell_composition_clr_abundance")
            & composition["feature"].eq("T_reg"),
            "Host",
            "Treg relative abundance",
            "HCA NEC atlas, unselected donors",
        ),
        select_effect(
            axes,
            axes["feature"].eq("TNF-ICAM1"),
            "Circuit",
            "TNF-ICAM1 axis",
            "HCA NEC atlas, sample-level coexpression",
        ),
        select_effect(
            axes,
            axes["feature"].eq("CCL21-CCR7"),
            "Circuit",
            "CCL21-CCR7 axis",
            "HCA NEC atlas, sample-level coexpression",
        ),
        select_effect(
            tissue,
            tissue["feature"].eq("mean_double_positive_cells_per_hpf"),
            "Tissue",
            "TUNEL+ LYVE1+ injury",
            "HCA tissue imaging, subject-level",
        ),
    ]
    for gene in ["ICAM1", "NLRP3", "IL22RA1", "EFNA1"]:
        selections.append(
            select_effect(
                bulk,
                bulk["accession"].eq("GSE46619")
                & bulk["contrast"].eq("nec_vs_control")
                & bulk["feature"].eq(gene),
                "Bulk tissue",
                gene,
                "GSE46619 NEC versus control",
                fdr_column="fdr_bh_effect_family",
            )
        )
    for feature, label in [
        ("Klebsiella_pneumoniae", "Klebsiella pneumoniae"),
        ("Clostridium_neonatale", "Clostridium neonatale"),
        ("Clostridium_perfringens", "Clostridium perfringens"),
        ("Staphylococcus_epidermidis", "Staphylococcus epidermidis"),
    ]:
        selections.append(
            select_effect(
                species,
                species["species"].eq(feature),
                "Microbiome",
                label,
                "MetaPhlAn, 13 infants from a 19-run bounded subset",
            )
        )
    for category in ["Glyoxylate metabolism", "TCA cycle", "Pyruvate metabolism"]:
        category_rows = pathways.loc[pathways["pathway_category"].eq(category)].copy()
        if not category_rows.empty:
            category_rows = category_rows.reindex(
                category_rows["hedges_g"].abs().sort_values(ascending=False).index
            )
            selected_pathway = category_rows.iloc[[0]]
            selections.append(
                select_effect(
                    selected_pathway,
                    pd.Series(True, index=selected_pathway.index),
                    "Microbial function",
                    category,
                    "HUMAnN, 13 infants from a 19-run bounded subset",
                )
            )
    for canonical in ["Succinate", "Inosine", "Nicotinic acid"]:
        selections.append(
            select_effect(
                metabolites,
                metabolites["canonical_metabolite"].eq(canonical),
                "Metabolome",
                canonical,
                "Wandro cohort, patient-level",
            )
        )
    rows = [row for row in selections if row is not None]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cross_domain_effect_summary.csv", index=False)
    return summary


def build_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "endotype_model_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "endotype_model_manifest.csv", index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    reanalysis_dir = (
        revision_root / "04_corrected_results" / "02_statistical_reanalysis"
    )
    output_dir = reanalysis_dir
    status = {
        "status": "started",
        "generated_utc": utc_now(),
        "software": {
            "python": sys.version,
            "xgboost": xgb.__version__,
        },
    }
    write_json(output_dir / "cross_sectional_endotype_model_status.json", status)

    feature_table = pd.read_csv(
        reanalysis_dir / "sra_infant_model_feature_matrix.csv.gz"
    )
    metadata_columns = ["infant_id", "condition"]
    feature_columns = [
        column for column in feature_table.columns if column not in metadata_columns
    ]
    feature_matrix = feature_table.set_index("infant_id")[feature_columns].astype(float)
    conditions = (
        feature_table.set_index("infant_id")
        .loc[feature_matrix.index, "condition"]
        .astype(str)
    )
    outcome = conditions.eq("nec").astype(int).to_numpy()
    species_columns = [column for column in feature_columns if column.startswith("species::")]
    pathway_columns = [column for column in feature_columns if column.startswith("pathway::")]
    matrices = {
        "Species logistic": feature_matrix[species_columns],
        "Pathway logistic": feature_matrix[pathway_columns],
        "Combined XGBoost": feature_matrix,
    }
    model_specs = {
        "Species logistic": (
            matrices["Species logistic"],
            make_logistic_pipeline(
                min(FEATURE_BUDGETS["Species logistic"], len(species_columns))
            ),
        ),
        "Pathway logistic": (
            matrices["Pathway logistic"],
            make_logistic_pipeline(
                min(FEATURE_BUDGETS["Pathway logistic"], len(pathway_columns))
            ),
        ),
        "Combined XGBoost": (
            matrices["Combined XGBoost"],
            make_xgboost_pipeline(
                min(FEATURE_BUDGETS["Combined XGBoost"], len(feature_columns))
            ),
        ),
    }

    metrics = []
    roc_rows = []
    pr_rows = []
    calibration_rows = []
    prediction_rows = []
    aggregated_predictions: dict[str, pd.Series] = {}
    leave_one_out_metrics = []
    leave_one_out_roc_rows = []
    leave_one_out_pr_rows = []
    leave_one_out_calibration_rows = []
    leave_one_out_prediction_rows = []
    for model_name, (matrix, model) in model_specs.items():
        raw_predictions = repeated_oof_predictions(model, matrix, outcome)
        raw_predictions["model"] = model_name
        raw_predictions["infant_id"] = raw_predictions["sample_index"].map(
            dict(enumerate(matrix.index))
        )
        prediction_rows.append(raw_predictions)
        aggregated = (
            raw_predictions.groupby("sample_index", as_index=False)
            .agg(
                observed=("observed", "first"),
                predicted_probability=("predicted_probability", "mean"),
                prediction_sd=("predicted_probability", "std"),
                n_repeats=("predicted_probability", "size"),
            )
            .sort_values("sample_index")
        )
        aggregated["infant_id"] = aggregated["sample_index"].map(
            dict(enumerate(matrix.index))
        )
        aggregated["condition"] = aggregated["infant_id"].map(conditions)
        aggregated["model"] = model_name
        prediction_rows.append(aggregated.assign(row_type="infant_aggregate"))
        aggregated_probability = aggregated.set_index("infant_id")[
            "predicted_probability"
        ].reindex(matrix.index)
        aggregated_predictions[model_name] = aggregated_probability
        metric, roc_table, pr_table, calibration = summarize_model(
            model_name,
            aggregated,
            outcome,
        )
        metric["cross_validation"] = (
            f"{CV_REPEATS} repeats of stratified 3-fold CV"
        )
        metric["feature_selection"] = "SelectKBest fitted within each training fold"
        metric["fixed_feature_budget"] = FEATURE_BUDGETS[model_name]
        metrics.append(metric)
        roc_rows.append(roc_table)
        pr_rows.append(pr_table)
        calibration_rows.append(calibration)

        leave_one_out = leave_one_infant_out_predictions(
            model,
            matrix,
            outcome,
        )
        leave_one_out["model"] = model_name
        leave_one_out["infant_id"] = leave_one_out["sample_index"].map(
            dict(enumerate(matrix.index))
        )
        leave_one_out["condition"] = leave_one_out["infant_id"].map(conditions)
        leave_one_out_prediction_rows.append(leave_one_out)
        (
            leave_one_out_metric,
            leave_one_out_roc,
            leave_one_out_pr,
            leave_one_out_calibration,
        ) = summarize_model(model_name, leave_one_out, outcome)
        leave_one_out_metric["cross_validation"] = "leave-one-infant-out"
        leave_one_out_metric["feature_selection"] = (
            "SelectKBest fitted within each training fold"
        )
        leave_one_out_metric["fixed_feature_budget"] = FEATURE_BUDGETS[model_name]
        leave_one_out_metrics.append(leave_one_out_metric)
        leave_one_out_roc_rows.append(leave_one_out_roc)
        leave_one_out_pr_rows.append(leave_one_out_pr)
        leave_one_out_calibration_rows.append(leave_one_out_calibration)

    metric_frame = pd.DataFrame(metrics)
    leave_one_out_metric_frame = pd.DataFrame(leave_one_out_metrics)
    permutation_rows = []
    permutation_p_values: dict[str, float] = {}
    for model_name, n_permutations in PERMUTATION_COUNTS.items():
        matrix = matrices[model_name]
        observed_auc = float(
            metric_frame.loc[
                metric_frame["model"].eq(model_name),
                "roc_auc",
            ].iloc[0]
        )
        permutation_p, permutation_auc = permutation_auc_test(
            matrix,
            outcome,
            observed_auc,
            model_name=model_name,
            n_features=FEATURE_BUDGETS[model_name],
            n_permutations=n_permutations,
        )
        permutation_p_values[model_name] = permutation_p
        metric_frame.loc[
            metric_frame["model"].eq(model_name),
            "permutation_p_value",
        ] = permutation_p
        permutation_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "permutation_index": np.arange(1, len(permutation_auc) + 1),
                    "permuted_roc_auc": permutation_auc,
                    "observed_roc_auc": observed_auc,
                }
            )
        )
    permutation_frame = pd.concat(permutation_rows, ignore_index=True)
    metric_frame.to_csv(output_dir / "cross_sectional_model_metrics.csv", index=False)
    leave_one_out_metric_frame.to_csv(
        output_dir / "cross_sectional_model_leave_one_infant_out_metrics.csv",
        index=False,
    )
    permutation_frame.to_csv(
        output_dir / "cross_sectional_model_auc_permutation_distribution.csv",
        index=False,
    )
    permutation_frame.loc[
        permutation_frame["model"].eq("Pathway logistic")
    ].to_csv(
        output_dir / "pathway_logistic_auc_permutation_distribution.csv",
        index=False,
    )
    pd.concat(roc_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_roc_curves.csv",
        index=False,
    )
    pd.concat(pr_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_pr_curves.csv",
        index=False,
    )
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_calibration.csv",
        index=False,
    )
    pd.concat(leave_one_out_roc_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_leave_one_infant_out_roc_curves.csv",
        index=False,
    )
    pd.concat(leave_one_out_pr_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_leave_one_infant_out_pr_curves.csv",
        index=False,
    )
    pd.concat(leave_one_out_calibration_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_leave_one_infant_out_calibration.csv",
        index=False,
    )
    pd.concat(leave_one_out_prediction_rows, ignore_index=True).to_csv(
        output_dir / "cross_sectional_model_leave_one_infant_out_predictions.csv",
        index=False,
    )
    all_predictions = pd.concat(prediction_rows, ignore_index=True, sort=False)
    all_predictions.to_csv(
        output_dir / "cross_sectional_model_repeated_predictions.csv",
        index=False,
    )
    aggregate_wide = pd.DataFrame(index=feature_matrix.index)
    aggregate_wide["condition"] = conditions
    for model_name, probability in aggregated_predictions.items():
        aggregate_wide[model_name] = probability
    aggregate_wide.reset_index().to_csv(
        output_dir / "cross_sectional_model_infant_predictions.csv",
        index=False,
    )
    aggregate_wide.reset_index().to_csv(
        output_dir / "cross_sectional_model_sample_predictions.csv",
        index=False,
    )

    shap_long, heatmap_long, final_model = final_xgboost_explanations(
        feature_matrix,
        outcome,
        feature_matrix.index.to_series(),
        conditions,
        output_dir,
        n_features=FEATURE_BUDGETS["Combined XGBoost"],
    )
    stability, pathway_heatmap = pathway_logistic_stability(
        feature_matrix[pathway_columns],
        outcome,
        feature_matrix.index.to_series(),
        conditions,
        output_dir,
        n_features=FEATURE_BUDGETS["Pathway logistic"],
    )
    budget_sensitivity = feature_budget_sensitivity(matrices, outcome)
    budget_sensitivity.to_csv(
        output_dir / "cross_sectional_model_feature_budget_sensitivity.csv",
        index=False,
    )
    endotype_probability_model = "Species logistic"
    endotype = build_endotype_map(
        feature_matrix,
        aggregated_predictions[endotype_probability_model],
        reanalysis_dir,
        output_dir,
    )
    endotype.insert(0, "condition", conditions)
    endotype.to_csv(output_dir / "microbial_functional_endotype_map.csv")
    cross_domain = build_cross_domain_effect_summary(reanalysis_dir, output_dir)

    status.update(
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "analysis_label": "infant-level cross-sectional NEC endotype classification",
            "n_samples": int(len(feature_matrix)),
            "n_infants": int(len(feature_matrix)),
            "n_nec": int(outcome.sum()),
            "n_control": int((outcome == 0).sum()),
            "feature_count": int(feature_matrix.shape[1]),
            "fixed_feature_budgets": FEATURE_BUDGETS,
            "cross_validation_repeats": CV_REPEATS,
            "permutation_counts": PERMUTATION_COUNTS,
            "models": metric_frame.to_dict(orient="records"),
            "leave_one_infant_out_models": leave_one_out_metric_frame.to_dict(
                orient="records"
            ),
            "endotype_probability_model": endotype_probability_model,
            "model_permutation_p_values": permutation_p_values,
            "endotype_species_features": endotype["species_features"].iloc[0],
            "endotype_pathway_features": endotype["pathway_features"].iloc[0],
            "cross_domain_results": int(len(cross_domain)),
            "pathway_logistic_stable_features": int(len(stability)),
            "feature_budget_sensitivity_rows": int(len(budget_sensitivity)),
            "analysis_unit": "13 independent infants; repeated runs averaged within infant",
        }
    )
    write_json(output_dir / "cross_sectional_endotype_model_status.json", status)
    build_manifest(output_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
