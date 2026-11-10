#!/usr/bin/env python3
"""Run native scVI/scANVI DR-CMap with donor-held-out conformal calibration."""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi
import sklearn
import torch
from scipy import sparse, stats
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


SEED = 20260726
ALPHA = 0.10
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "dr_cmap"
RESULTS = ROOT / "results" / "dr_cmap"
MODELS = RESULTS / "models"
MANIFESTS = ROOT / "manifests"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(min(12, os.cpu_count() or 1))
torch.set_num_interop_threads(2)
torch.set_float32_matmul_precision("high")
scvi.settings.seed = SEED
scvi.settings.num_threads = min(12, os.cpu_count() or 1)

CUDA_AVAILABLE = torch.cuda.is_available()
ACCELERATOR = "gpu" if CUDA_AVAILABLE else "cpu"
TRAINING_PRECISION = "16-mixed" if CUDA_AVAILABLE else "32-true"
TRAINING_BATCH_SIZE = 128 if CUDA_AVAILABLE else 256


TARGETS = {
    "epithelial": ["M cell"],
    "endothelial": ["LEC"],
    "immune": ["ILC3", "Resident-like T", "Treg"],
}

MARKERS = {
    "M cell": ["SPIB", "GP2", "SOX8", "TNFRSF11A", "CCL20", "MARCKSL1"],
    "LEC": ["PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "ACKR4"],
    "ILC3": ["RORC", "IL7R", "KIT", "AHR", "IL22", "NCR1"],
    "Resident-like T": ["CD69", "ITGAE", "CXCR6", "ZNF683", "RUNX3", "CCL5"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "CCR7"],
}

TRAINING = {
    "epithelial": {"scvi_epochs": 22, "scanvi_epochs": 12},
    "endothelial": {"scvi_epochs": 22, "scanvi_epochs": 12},
    "immune": {"scvi_epochs": 20, "scanvi_epochs": 10},
}


def align_probabilities(
    predicted: np.ndarray,
    predicted_classes: np.ndarray,
    all_classes: list[str],
) -> np.ndarray:
    aligned = np.zeros((predicted.shape[0], len(all_classes)), dtype=np.float32)
    lookup = {label: idx for idx, label in enumerate(all_classes)}
    for source_idx, label in enumerate(predicted_classes):
        aligned[:, lookup[str(label)]] = predicted[:, source_idx]
    return aligned


def donor_held_out_knn(
    latent: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    classes: list[str],
    neighbors: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(labels), len(classes)), dtype=np.float32)
    predictions = np.empty(len(labels), dtype=object)
    for donor in sorted(pd.unique(donors)):
        test = donors == donor
        train = ~test
        k = min(neighbors, int(train.sum()))
        model = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )
        model.fit(latent[train], labels[train])
        fold_probability = model.predict_proba(latent[test])
        probabilities[test] = align_probabilities(
            fold_probability, model.classes_, classes
        )
        predictions[test] = np.asarray(classes)[
            probabilities[test].argmax(axis=1)
        ]
    return probabilities, predictions.astype(str)


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 1.0
    level = min(1.0, math.ceil((scores.size + 1) * (1 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def conformal_sets(
    probabilities: np.ndarray,
    classes: list[str],
    qhat: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    set_strings: list[str] = []
    set_sizes = np.zeros(probabilities.shape[0], dtype=int)
    singleton = np.full(probabilities.shape[0], "Rejected", dtype=object)
    for row_idx, row in enumerate(probabilities):
        members = [
            label
            for class_idx, label in enumerate(classes)
            if 1.0 - float(row[class_idx]) <= qhat[label]
        ]
        set_sizes[row_idx] = len(members)
        set_strings.append("|".join(members) if members else "Empty")
        if len(members) == 1:
            singleton[row_idx] = members[0]
    return set_strings, set_sizes, singleton.astype(str)


def exact_label_permutation(
    values: np.ndarray,
    conditions: np.ndarray,
    statistic: str = "difference",
) -> float:
    values = np.asarray(values, dtype=float)
    conditions = np.asarray(conditions, dtype=str)
    keep = (
        np.all(np.isfinite(values), axis=1)
        if values.ndim == 2
        else np.isfinite(values)
    )
    values = values[keep]
    conditions = conditions[keep]
    nec_count = int(np.sum(conditions == "nec"))
    if nec_count == 0 or nec_count == len(values):
        return np.nan
    if statistic == "distance":
        observed = float(
            np.linalg.norm(
                values[conditions == "nec"].mean(axis=0)
                - values[conditions == "control"].mean(axis=0)
            )
        )
    else:
        observed = abs(
            float(
                values[conditions == "nec"].mean()
                - values[conditions == "control"].mean()
            )
        )
    indices = np.arange(len(values))
    null: list[float] = []
    for nec_indices in itertools.combinations(indices, nec_count):
        is_nec = np.zeros(len(values), dtype=bool)
        is_nec[list(nec_indices)] = True
        if statistic == "distance":
            current = float(
                np.linalg.norm(values[is_nec].mean(axis=0) - values[~is_nec].mean(axis=0))
            )
        else:
            current = abs(float(values[is_nec].mean() - values[~is_nec].mean()))
        null.append(current)
    null_array = np.asarray(null)
    return float((np.sum(null_array >= observed - 1e-12) + 1) / (len(null_array) + 1))


def bootstrap_difference(
    values: np.ndarray,
    conditions: np.ndarray,
    iterations: int = 5_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    nec = np.asarray(values, dtype=float)[np.asarray(conditions) == "nec"]
    control = np.asarray(values, dtype=float)[np.asarray(conditions) == "control"]
    if len(nec) < 2 or len(control) < 2:
        return np.nan, np.nan
    draws = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        draws[idx] = (
            rng.choice(nec, size=len(nec), replace=True).mean()
            - rng.choice(control, size=len(control), replace=True).mean()
        )
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def hedges_g(values: np.ndarray, conditions: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    conditions = np.asarray(conditions, dtype=str)
    nec = values[conditions == "nec"]
    control = values[conditions == "control"]
    if len(nec) < 2 or len(control) < 2:
        return np.nan
    pooled_variance = (
        (len(nec) - 1) * np.var(nec, ddof=1)
        + (len(control) - 1) * np.var(control, ddof=1)
    ) / (len(nec) + len(control) - 2)
    if pooled_variance <= 0:
        return np.nan
    d = (nec.mean() - control.mean()) / np.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (len(nec) + len(control)) - 9)
    return float(d * correction)


def developmental_direction(
    latent: np.ndarray,
    obs: pd.DataFrame,
    reference_mask: np.ndarray,
) -> np.ndarray:
    ref_obs = obs.loc[reference_mask].copy()
    ref_latent = latent[reference_mask]
    nuisance = ref_obs[
        ["reference_label", "region_group", "protocol", "development_stage"]
    ].astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    nuisance_matrix = encoder.fit_transform(nuisance)
    nuisance_model = Ridge(alpha=10.0, solver="lsqr")
    nuisance_model.fit(nuisance_matrix, ref_latent)
    residual = ref_latent - nuisance_model.predict(nuisance_matrix)
    age = ref_obs["age_log_scaled"].astype(float).to_numpy()[:, None]
    age_model = Ridge(alpha=10.0)
    age_model.fit(age, residual)
    direction = np.asarray(age_model.coef_, dtype=float).reshape(-1)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("The healthy reference developmental direction was not estimable.")
    return direction / norm


def normalized_expression(adata: ad.AnnData, mask: np.ndarray) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(adata.X[mask], dtype=np.float32)
    library = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        10_000.0,
        library,
        out=np.zeros_like(library, dtype=np.float32),
        where=library > 0,
    )
    normalized = sparse.diags(scale).dot(matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def marker_scores(
    adata: ad.AnnData,
    expression: sparse.csr_matrix,
    target: str,
) -> tuple[np.ndarray, list[str]]:
    genes = [gene for gene in MARKERS[target] if gene in adata.var_names]
    if not genes:
        return np.full(expression.shape[0], np.nan), []
    positions = adata.var_names.get_indexer(genes)
    score = np.asarray(expression[:, positions].mean(axis=1)).ravel()
    return score, genes


def donor_analysis(
    lineage: str,
    obs: pd.DataFrame,
    latent: np.ndarray,
    probability_matrix: np.ndarray,
    classes: list[str],
    singleton_labels: np.ndarray,
    adata: ad.AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict_mask = obs["query_kind"].astype(str).to_numpy() == "strict_nec_query"
    strict_obs = obs.loc[strict_mask].copy()
    strict_latent = latent[strict_mask]
    strict_probability = probability_matrix[strict_mask]
    strict_singleton = singleton_labels[strict_mask]
    expression = normalized_expression(adata, strict_mask)
    class_index = {label: idx for idx, label in enumerate(classes)}
    development_vector = developmental_direction(
        latent,
        obs,
        obs["query_kind"].astype(str).to_numpy() == "reference",
    )
    development_projection = strict_latent @ development_vector
    orthogonal_latent = strict_latent - np.outer(
        development_projection, development_vector
    )

    metrics_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    dot_rows: list[dict[str, object]] = []

    for target in TARGETS[lineage]:
        target_probability = strict_probability[:, class_index[target]]
        target_marker_score, marker_genes = marker_scores(adata, expression, target)
        for gene in marker_genes:
            gene_position = adata.var_names.get_loc(gene)
            gene_values = np.asarray(expression[:, gene_position].toarray()).ravel()
            weights = target_probability
            dot_rows.append(
                {
                    "lineage": lineage,
                    "target_state": target,
                    "gene": gene,
                    "weighted_mean_log_expression": float(
                        np.average(gene_values, weights=weights)
                    ),
                    "weighted_fraction_expressing": float(
                        np.average(gene_values > 0, weights=weights)
                    ),
                    "accepted_cells": int(np.sum(strict_singleton == target)),
                }
            )

        for analysis_set, selection_mask in {
            "primary_unselected": strict_obs["selection"].astype(str).to_numpy()
            == "unselected",
            "sensitivity_all_selections": np.ones(len(strict_obs), dtype=bool),
        }.items():
            donor_rows: list[dict[str, object]] = []
            for donor, donor_frame in strict_obs.loc[selection_mask].groupby(
                "donor_id", observed=True
            ):
                positions = strict_obs.index.get_indexer(donor_frame.index)
                weights = target_probability[positions]
                weight_sum = float(weights.sum())
                if weight_sum <= 0:
                    continue
                centroid = np.average(
                    orthogonal_latent[positions], axis=0, weights=weights
                )
                donor_rows.append(
                    {
                        "lineage": lineage,
                        "analysis_set": analysis_set,
                        "target_state": target,
                        "donor_id": str(donor),
                        "condition": str(donor_frame["condition"].iloc[0]),
                        "n_compartment_cells": int(len(positions)),
                        "target_probability_fraction": float(weights.mean()),
                        "conformal_singleton_fraction": float(
                            np.mean(strict_singleton[positions] == target)
                        ),
                        "mean_target_probability": float(weights.mean()),
                        "marker_program_score": float(
                            np.average(target_marker_score[positions], weights=weights)
                        ),
                        "development_projection": float(
                            np.average(development_projection[positions], weights=weights)
                        ),
                        "_centroid": centroid,
                    }
                )
            if not donor_rows:
                continue
            donor_table = pd.DataFrame(donor_rows)
            centroids = np.vstack(donor_table.pop("_centroid"))
            conditions = donor_table["condition"].to_numpy()
            nec_mean = centroids[conditions == "nec"].mean(axis=0)
            control_mean = centroids[conditions == "control"].mean(axis=0)
            disease_vector = nec_mean - control_mean
            disease_distance = float(np.linalg.norm(disease_vector))
            if disease_distance > 0:
                disease_unit = disease_vector / disease_distance
                donor_table["orthogonal_disease_score"] = centroids @ disease_unit
            else:
                donor_table["orthogonal_disease_score"] = 0.0
            metrics_rows.extend(donor_table.to_dict("records"))

            proportion = donor_table["target_probability_fraction"].to_numpy()
            proportion_difference = float(
                proportion[conditions == "nec"].mean()
                - proportion[conditions == "control"].mean()
            )
            proportion_ci = bootstrap_difference(proportion, conditions)
            development_values = donor_table["development_projection"].to_numpy()
            development_difference = float(
                development_values[conditions == "nec"].mean()
                - development_values[conditions == "control"].mean()
            )
            disease_scores = donor_table["orthogonal_disease_score"].to_numpy()
            disease_score_difference = float(
                disease_scores[conditions == "nec"].mean()
                - disease_scores[conditions == "control"].mean()
            )
            disease_score_ci = bootstrap_difference(disease_scores, conditions)
            effect_rows.append(
                {
                    "lineage": lineage,
                    "analysis_set": analysis_set,
                    "target_state": target,
                    "nec_donors": int(np.sum(conditions == "nec")),
                    "control_donors": int(np.sum(conditions == "control")),
                    "probability_fraction_difference": proportion_difference,
                    "probability_fraction_ci_low": proportion_ci[0],
                    "probability_fraction_ci_high": proportion_ci[1],
                    "probability_fraction_exact_p": exact_label_permutation(
                        proportion, conditions
                    ),
                    "probability_fraction_hedges_g": hedges_g(proportion, conditions),
                    "development_projection_difference": development_difference,
                    "development_projection_exact_p": exact_label_permutation(
                        development_values, conditions
                    ),
                    "orthogonal_disease_distance": disease_distance,
                    "orthogonal_disease_distance_exact_p": exact_label_permutation(
                        centroids, conditions, statistic="distance"
                    ),
                    "orthogonal_disease_score_difference": disease_score_difference,
                    "orthogonal_disease_score_ci_low": disease_score_ci[0],
                    "orthogonal_disease_score_ci_high": disease_score_ci[1],
                }
            )
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(effect_rows),
        pd.DataFrame(dot_rows),
    )


def run_lineage(lineage: str) -> dict[str, object]:
    start = time.time()
    adata = ad.read_h5ad(PROCESSED / f"{lineage}_drcmap_input.h5ad")
    for column in [
        "batch",
        "reference_label",
        "protocol",
        "region_group",
        "development_stage",
        "selection",
    ]:
        adata.obs[column] = pd.Categorical(adata.obs[column].astype(str))

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="reference_label",
        categorical_covariate_keys=[
            "protocol",
            "region_group",
            "development_stage",
            "selection",
        ],
        continuous_covariate_keys=["age_log_scaled"],
    )
    vae = scvi.model.SCVI(
        adata,
        n_hidden=96,
        n_latent=16,
        n_layers=2,
        dropout_rate=0.10,
        dispersion="gene",
        gene_likelihood="nb",
    )
    vae.train(
        max_epochs=TRAINING[lineage]["scvi_epochs"],
        train_size=0.90,
        validation_size=0.10,
        batch_size=TRAINING_BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=5,
        check_val_every_n_epoch=1,
        accelerator=ACCELERATOR,
        devices=1,
        precision=TRAINING_PRECISION,
        enable_progress_bar=False,
    )
    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
    )
    scanvi.train(
        max_epochs=TRAINING[lineage]["scanvi_epochs"],
        train_size=0.90,
        validation_size=0.10,
        batch_size=TRAINING_BATCH_SIZE,
        early_stopping=True,
        early_stopping_patience=4,
        check_val_every_n_epoch=1,
        accelerator=ACCELERATOR,
        devices=1,
        precision=TRAINING_PRECISION,
        enable_progress_bar=False,
    )
    model_dir = MODELS / f"{lineage}_scanvi"
    scanvi.save(model_dir, overwrite=True, save_anndata=False)

    latent = scanvi.get_latent_representation()
    soft = scanvi.predict(soft=True)
    if not isinstance(soft, pd.DataFrame):
        soft = pd.DataFrame(soft, index=adata.obs_names)
    soft.columns = soft.columns.astype(str)
    scanvi_classes = soft.columns.tolist()
    scanvi_probability = soft.to_numpy(dtype=np.float32)
    scanvi_prediction = np.asarray(scanvi_classes)[
        scanvi_probability.argmax(axis=1)
    ]

    reference_mask = adata.obs["query_kind"].astype(str).to_numpy() == "reference"
    reference_labels = adata.obs.loc[reference_mask, "reference_label"].astype(str).to_numpy()
    reference_donors = adata.obs.loc[reference_mask, "donor_id"].astype(str).to_numpy()
    classes = sorted(pd.unique(reference_labels).tolist())
    cv_probability, cv_prediction = donor_held_out_knn(
        latent[reference_mask],
        reference_labels,
        reference_donors,
        classes,
    )
    qhat: dict[str, float] = {}
    for class_idx, label in enumerate(classes):
        truth = reference_labels == label
        qhat[label] = conformal_quantile(1.0 - cv_probability[truth, class_idx])

    knn = KNeighborsClassifier(
        n_neighbors=min(31, int(reference_mask.sum())),
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(latent[reference_mask], reference_labels)
    full_probability = align_probabilities(
        knn.predict_proba(latent), knn.classes_, classes
    )
    full_prediction = np.asarray(classes)[full_probability.argmax(axis=1)]
    set_strings, set_sizes, singleton = conformal_sets(
        full_probability, classes, qhat
    )

    cv_sets, cv_set_sizes, cv_singleton = conformal_sets(
        cv_probability, classes, qhat
    )
    calibration_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for label in classes:
        mask = reference_labels == label
        class_index = classes.index(label)
        coverage = np.mean(
            [
                label in value.split("|")
                for value in np.asarray(cv_sets, dtype=str)[mask]
            ]
        )
        calibration_rows.append(
            {
                "lineage": lineage,
                "reference_label": label,
                "reference_cells": int(mask.sum()),
                "reference_donors": int(pd.unique(reference_donors[mask]).size),
                "qhat_alpha_0_10": qhat[label],
                "donor_held_out_coverage": float(coverage),
                "donor_held_out_singleton_rate": float(
                    np.mean(cv_set_sizes[mask] == 1)
                ),
                "donor_held_out_accuracy": float(
                    np.mean(cv_prediction[mask] == reference_labels[mask])
                ),
            }
        )
        for predicted_label in classes:
            count = int(np.sum(mask & (cv_prediction == predicted_label)))
            confusion_rows.append(
                {
                    "lineage": lineage,
                    "true_label": label,
                    "predicted_label": predicted_label,
                    "cell_count": count,
                    "row_fraction": count / int(mask.sum()),
                }
            )

    adata.obsm["X_scANVI"] = latent
    sc.pp.neighbors(
        adata,
        n_neighbors=20,
        use_rep="X_scANVI",
        metric="cosine",
        random_state=SEED,
    )
    sc.tl.umap(adata, random_state=SEED, min_dist=0.35)

    mapping = adata.obs.copy()
    mapping.insert(0, "lineage", lineage)
    mapping["umap_1"] = adata.obsm["X_umap"][:, 0]
    mapping["umap_2"] = adata.obsm["X_umap"][:, 1]
    mapping["scanvi_label"] = scanvi_prediction
    mapping["scanvi_max_probability"] = scanvi_probability.max(axis=1)
    mapping["knn_label"] = full_prediction
    mapping["knn_max_probability"] = full_probability.max(axis=1)
    mapping["conformal_set"] = set_strings
    mapping["conformal_set_size"] = set_sizes
    mapping["conformal_singleton_label"] = singleton
    for target in TARGETS[lineage]:
        mapping[f"probability__{target}"] = full_probability[:, classes.index(target)]
    mapping.reset_index(names="integrated_cell_id").to_csv(
        RESULTS / f"{lineage}_cell_mapping.csv", index=False
    )
    adata.obs["scanvi_label"] = pd.Categorical(scanvi_prediction)
    adata.obs["knn_label"] = pd.Categorical(full_prediction)
    adata.obs["conformal_set"] = pd.Categorical(set_strings)
    adata.obs["conformal_singleton_label"] = pd.Categorical(singleton)
    adata.write_h5ad(
        RESULTS / f"{lineage}_drcmap_mapped.h5ad",
        compression="gzip",
    )

    donor_metrics, effects, marker_dot = donor_analysis(
        lineage,
        adata.obs,
        latent,
        full_probability,
        classes,
        singleton,
        adata,
    )
    donor_metrics.to_csv(RESULTS / f"{lineage}_donor_state_metrics.csv", index=False)
    effects.to_csv(RESULTS / f"{lineage}_disease_effects.csv", index=False)
    marker_dot.to_csv(RESULTS / f"{lineage}_marker_dotplot.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        RESULTS / f"{lineage}_conformal_calibration.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        RESULTS / f"{lineage}_donor_held_out_confusion.csv", index=False
    )

    external_summary: list[dict[str, object]] = []
    if lineage == "epithelial":
        external_mask = (
            adata.obs["query_kind"].astype(str).to_numpy() == "external_validation"
        )
        target_idx = classes.index("M cell")
        external_obs = adata.obs.loc[external_mask]
        external_probability = full_probability[external_mask, target_idx]
        external_singleton = singleton[external_mask]
        for plate, frame in external_obs.groupby("donor_id", observed=True):
            positions = external_obs.index.get_indexer(frame.index)
            external_summary.append(
                {
                    "plate": str(plate),
                    "cells": int(len(frame)),
                    "mean_m_cell_probability": float(
                        external_probability[positions].mean()
                    ),
                    "singleton_m_cell_fraction": float(
                        np.mean(external_singleton[positions] == "M cell")
                    ),
                }
            )
        pd.DataFrame(external_summary).to_csv(
            RESULTS / "gse275772_m_cell_mapping_validation.csv", index=False
        )

    elapsed = time.time() - start
    return {
        "lineage": lineage,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "reference_cells": int(reference_mask.sum()),
        "strict_query_cells": int(
            np.sum(adata.obs["query_kind"].astype(str) == "strict_nec_query")
        ),
        "external_validation_cells": int(
            np.sum(adata.obs["query_kind"].astype(str) == "external_validation")
        ),
        "classes": classes,
        "targets": TARGETS[lineage],
        "scvi_epochs_requested": TRAINING[lineage]["scvi_epochs"],
        "scanvi_epochs_requested": TRAINING[lineage]["scanvi_epochs"],
        "elapsed_seconds": elapsed,
        "model_path": str(model_dir),
        "mapped_h5ad": str(RESULTS / f"{lineage}_drcmap_mapped.h5ad"),
    }


def main() -> None:
    summaries: list[dict[str, object]] = []
    for lineage in ["epithelial", "endothelial", "immune"]:
        print(f"DR-CMap native training: {lineage}", flush=True)
        summaries.append(run_lineage(lineage))

    donor_metrics = pd.concat(
        [
            pd.read_csv(RESULTS / f"{lineage}_donor_state_metrics.csv")
            for lineage in TARGETS
        ],
        ignore_index=True,
    )
    effects = pd.concat(
        [
            pd.read_csv(RESULTS / f"{lineage}_disease_effects.csv")
            for lineage in TARGETS
        ],
        ignore_index=True,
    )
    marker_dot = pd.concat(
        [
            pd.read_csv(RESULTS / f"{lineage}_marker_dotplot.csv")
            for lineage in TARGETS
        ],
        ignore_index=True,
    )
    calibration = pd.concat(
        [
            pd.read_csv(RESULTS / f"{lineage}_conformal_calibration.csv")
            for lineage in TARGETS
        ],
        ignore_index=True,
    )
    confusion = pd.concat(
        [
            pd.read_csv(RESULTS / f"{lineage}_donor_held_out_confusion.csv")
            for lineage in TARGETS
        ],
        ignore_index=True,
    )
    donor_metrics.to_csv(RESULTS / "drcmap_donor_state_metrics.csv", index=False)
    effects.to_csv(RESULTS / "drcmap_disease_effects.csv", index=False)
    marker_dot.to_csv(RESULTS / "drcmap_marker_dotplot.csv", index=False)
    calibration.to_csv(RESULTS / "drcmap_conformal_calibration.csv", index=False)
    confusion.to_csv(RESULTS / "drcmap_donor_held_out_confusion.csv", index=False)

    manifest = {
        "status": "PASS",
        "analysis": "DR-CMap",
        "algorithm": "native scVI/scANVI with donor-held-out KNN conformal calibration",
        "native_scvi": True,
        "native_scanvi": True,
        "proxy_analysis": False,
        "conformal_alpha": ALPHA,
        "primary_statistical_unit": "donor",
        "primary_selection": "unselected",
        "sensitivity_selection": "all selections",
        "random_seed": SEED,
        "training_backend": {
            "accelerator": ACCELERATOR,
            "precision": TRAINING_PRECISION,
            "batch_size": TRAINING_BATCH_SIZE,
            "cuda_available": CUDA_AVAILABLE,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
            ),
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "scvi_tools": scvi.__version__,
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "lineages": summaries,
        "cloud_files_written": 0,
    }
    (MANIFESTS / "dr_cmap_native_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
