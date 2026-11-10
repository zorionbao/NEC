from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import rankdata


SEED = 20260726
BOOTSTRAPS = 500
COUPLING_BOOTSTRAPS = 500
COUPLING_PERMUTATIONS = 2000
LAMBDA_GRID = np.asarray([0.01, 0.1, 1.0, 10.0, 100.0])
EVENT_GRID = np.linspace(-20.0, 30.0, 101)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = ANALYSIS_ROOT / "code"
PUBLIC_ROOT = ANALYSIS_ROOT / "data" / "public" / "preterm_infant_analysis"
SYNC_ROOT = ANALYSIS_ROOT / "data" / "synced" / "tempo_nec"
RESULT_ROOT = ANALYSIS_ROOT / "results" / "tempo_nec"
SOURCE_ROOT = (
    ANALYSIS_ROOT
    / "source_data"
    / "Extended_Data_Figure_6_TEMPO_NEC"
)
MANIFEST_ROOT = ANALYSIS_ROOT / "manifests"

RDATA_PATH = PUBLIC_ROOT / ".RData"
import os
RSCRIPT_PATH = os.environ.get("NEC_RSCRIPT", "Rscript")
EXTRACT_SCRIPT = CODE_ROOT / "04_extract_wandro_author_objects.R"

TARGET_METABOLITES = [
    "succinic acid",
    "lactic acid",
    "inosine",
    "nicotinic acid",
    "indole-3-lactate",
    "indole-3-acetate",
    "3-hydroxybutyric acid",
    "chenodeoxycholic acid",
    "3-hydroxypropionic acid",
    "citric acid",
    "isocitric acid",
]

TARGET_GENERA = [
    "Pseudomonas",
    "Bifidobacterium",
    "Enterococcus",
    "Staphylococcus",
    "Streptococcus",
    "Klebsiella",
    "Clostridium",
    "Citrobacter",
]

FIGURE_GENERA = [
    "Clostridium",
    "Anaerococcus",
    "Staphylococcus",
    "Klebsiella",
    "Bifidobacterium",
    "Enterococcus",
    "Pseudomonas",
    "Citrobacter",
]

FIGURE_METABOLITES = [
    "succinic acid",
    "lactitol",
    "adenosine-5-monophosphate",
    "citramalic acid",
    "indole-3-acetate",
    "inosine",
    "nicotinic acid",
    "indole-3-lactate",
]

FIGURE_CURVES = [
    "Clostridium",
    "Anaerococcus",
    "Staphylococcus",
    "succinic acid",
    "lactitol",
    "adenosine-5-monophosphate",
]

METRIC_NAMES = [
    "delta_at_event",
    "peri_event_auc",
    "delta_at_day_14",
    "nec_pre_to_event",
    "nec_event_to_recovery",
    "trajectory_distance",
]


@dataclass
class ModalityResult:
    modality: str
    features: list[str]
    display_features: list[str]
    sample_metadata: pd.DataFrame
    transformed: np.ndarray
    age_residual: np.ndarray
    within_deviation: np.ndarray
    metrics: pd.DataFrame
    curves: pd.DataFrame
    windows: pd.DataFrame
    loo: pd.DataFrame
    feeding: pd.DataFrame
    gi_sensitivity: pd.DataFrame
    selected_indices: np.ndarray
    preprocessing: dict[str, object]


def ensure_local_write(path: Path) -> Path:
    resolved = path.resolve()
    root = ANALYSIS_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside local analysis root: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[valid] = restored
    return output


def run_author_extraction() -> None:
    expected = [
        SYNC_ROOT / "wandro_author_infant_metadata.csv",
        SYNC_ROOT / "wandro_author_genus_relative_abundance.csv",
        SYNC_ROOT / "wandro_author_otu_relative_abundance.csv",
        SYNC_ROOT / "wandro_author_metabolite_intensity.csv",
    ]
    ensure_local_write(SYNC_ROOT).mkdir(parents=True, exist_ok=True)
    if all(path.exists() and path.stat().st_size > 0 for path in expected):
        return
    if not RDATA_PATH.exists():
        raise FileNotFoundError(RDATA_PATH)
    command = [
        str(RSCRIPT_PATH),
        str(EXTRACT_SCRIPT),
        str(RDATA_PATH),
        str(SYNC_ROOT),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if '"status":"PASS"' not in completed.stdout:
        raise RuntimeError(f"Unexpected author extraction output: {completed.stdout}")


def clean_genus_label(taxonomy: str) -> str:
    parts = [part.strip() for part in str(taxonomy).split(";")]
    genus = next(
        (
            part.replace("g__", "").strip()
            for part in reversed(parts)
            if part.startswith("g__") and part.replace("g__", "").strip()
        ),
        "",
    )
    if genus:
        return genus
    family = next(
        (
            part.replace("f__", "").strip()
            for part in reversed(parts)
            if part.startswith("f__") and part.replace("f__", "").strip()
        ),
        "",
    )
    if family:
        return f"Unclassified {family}"
    return str(taxonomy)


def make_unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        output.append(
            label if counts[label] == 1 else f"{label} {counts[label]}"
        )
    return output


def load_author_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    run_author_extraction()
    metadata = pd.read_csv(SYNC_ROOT / "wandro_author_infant_metadata.csv")
    genus = pd.read_csv(
        SYNC_ROOT / "wandro_author_genus_relative_abundance.csv"
    )
    metabolite = pd.read_csv(
        SYNC_ROOT / "wandro_author_metabolite_intensity.csv"
    )
    for frame in [metadata, genus, metabolite]:
        frame["Sample"] = pd.to_numeric(frame["Sample"], errors="raise").astype(int)
    metadata["Patient"] = pd.to_numeric(
        metadata["Patient"],
        errors="raise",
    ).astype(int)
    metadata["analysis_group"] = metadata["Group"].replace(
        {"septic": "sepsis"}
    )
    if metadata["Sample"].nunique() != 77:
        raise RuntimeError("Author metadata does not contain 77 unique samples")
    if set(genus["Sample"]) != set(metadata["Sample"]):
        raise RuntimeError("Author genus matrix is not aligned to metadata")
    if not set(metabolite["Sample"]).issubset(set(metadata["Sample"])):
        raise RuntimeError("Author metabolite matrix contains unmapped samples")
    expected_metabolite_missing = {35, 50}
    observed_missing = set(metadata["Sample"]) - set(metabolite["Sample"])
    if observed_missing != expected_metabolite_missing:
        raise RuntimeError(
            f"Unexpected metabolomics missing samples: {observed_missing}"
        )
    audit = {
        "repository_commit": subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=*",
                "-C",
                str(PUBLIC_ROOT),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "rdata_sha256": sha256(RDATA_PATH),
        "metadata_samples": int(metadata["Sample"].nunique()),
        "metadata_infants": int(metadata["Patient"].nunique()),
        "primary_control_samples": int(
            metadata["analysis_group"].eq("control").sum()
        ),
        "primary_control_infants": int(
            metadata.loc[
                metadata["analysis_group"].eq("control"),
                "Patient",
            ].nunique()
        ),
        "gi_samples_excluded_primary": int(
            metadata["analysis_group"].eq("GI").sum()
        ),
        "nec_samples": int(metadata["analysis_group"].eq("nec").sum()),
        "nec_infants": int(
            metadata.loc[metadata["analysis_group"].eq("nec"), "Patient"].nunique()
        ),
        "sepsis_samples": int(metadata["analysis_group"].eq("sepsis").sum()),
        "sepsis_infants": int(
            metadata.loc[
                metadata["analysis_group"].eq("sepsis"),
                "Patient",
            ].nunique()
        ),
        "genus_samples": int(genus["Sample"].nunique()),
        "metabolite_samples": int(metabolite["Sample"].nunique()),
        "metabolite_missing_samples": sorted(observed_missing),
        "legacy_direct_otu_columns": 79,
        "legacy_direct_otu_alignment_status": (
            "rejected_in_favor_of_author_RData_objects"
        ),
    }
    return metadata, genus, metabolite, audit


def transform_genus(
    genus: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    features = [column for column in genus.columns if column != "Sample"]
    values = genus[features].apply(pd.to_numeric, errors="raise").to_numpy()
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=2e-6):
        raise RuntimeError("Author genus matrix does not close to one")
    positive = values[values > 0]
    pseudocount = max(float(np.min(positive) / 2.0), 1e-8)
    logged = np.log(values + pseudocount)
    clr = logged - logged.mean(axis=1, keepdims=True)
    display = make_unique_labels([clean_genus_label(value) for value in features])
    transformed = pd.DataFrame(clr, columns=features)
    transformed.insert(0, "Sample", genus["Sample"].to_numpy())
    details = {
        "scale": "centered log ratio",
        "pseudocount": pseudocount,
        "features": len(features),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }
    return transformed, features, display, details


def transform_metabolites(
    metabolite: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    features = [
        column
        for column in metabolite.columns
        if column not in {"Sample", "mTIC"}
    ]
    raw = metabolite[features].apply(pd.to_numeric, errors="coerce")
    keep = raw.notna().all(axis=0) & raw.std(axis=0, ddof=1).gt(0)
    raw = raw.loc[:, keep]
    features = raw.columns.tolist()
    logged = np.log1p(raw.to_numpy(dtype=float))
    mean = logged.mean(axis=0, keepdims=True)
    standard_deviation = logged.std(axis=0, ddof=1, keepdims=True)
    standardized = (logged - mean) / standard_deviation
    transformed = pd.DataFrame(standardized, columns=features)
    transformed.insert(0, "Sample", metabolite["Sample"].to_numpy())
    details = {
        "scale": "z score of log1p untargeted GC-MS intensity",
        "features": len(features),
        "excluded_total_ion_current": "mTIC",
        "dropped_incomplete_or_constant_features": int((~keep).sum()),
    }
    return transformed, features, features.copy(), details


def spline_basis(
    values: np.ndarray,
    degrees_of_freedom: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    expression = (
        "bs(x, "
        f"df={degrees_of_freedom}, "
        "degree=3, "
        "include_intercept=False, "
        f"lower_bound={lower_bound}, "
        f"upper_bound={upper_bound}) - 1"
    )
    return np.asarray(
        dmatrix(
            expression,
            {"x": np.asarray(values, dtype=float)},
            return_type="dataframe",
        ),
        dtype=float,
    )


def equal_cluster_weights(clusters: np.ndarray) -> np.ndarray:
    clusters = np.asarray(clusters)
    unique, counts = np.unique(clusters, return_counts=True)
    count_lookup = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / count_lookup[value] for value in clusters],
        dtype=float,
    )
    return weights * len(weights) / weights.sum()


def fit_ridge_by_lambda(
    design: np.ndarray,
    response: np.ndarray,
    clusters: np.ndarray,
    lambdas: np.ndarray,
    penalty: np.ndarray,
) -> np.ndarray:
    weights = equal_cluster_weights(clusters)
    weighted_design = design * weights[:, None]
    cross_product = design.T @ weighted_design
    right_hand = design.T @ (response * weights[:, None])
    coefficients = np.zeros((design.shape[1], response.shape[1]))
    for value in np.unique(lambdas):
        columns = np.flatnonzero(np.isclose(lambdas, value))
        matrix = cross_product + float(value) * np.diag(penalty)
        coefficients[:, columns] = np.linalg.pinv(matrix) @ right_hand[:, columns]
    return coefficients


def select_lambda_leave_one_cluster_out(
    design: np.ndarray,
    response: np.ndarray,
    clusters: np.ndarray,
    penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_clusters = np.unique(clusters)
    scores = np.zeros((len(LAMBDA_GRID), response.shape[1]))
    for held_out in unique_clusters:
        test = clusters == held_out
        train = ~test
        for index, value in enumerate(LAMBDA_GRID):
            coefficients = fit_ridge_by_lambda(
                design[train],
                response[train],
                clusters[train],
                np.full(response.shape[1], value),
                penalty,
            )
            residual = response[test] - design[test] @ coefficients
            scores[index] += np.mean(residual**2, axis=0)
    scores /= len(unique_clusters)
    best_index = np.argmin(scores, axis=0)
    return LAMBDA_GRID[best_index], scores


def within_cluster_center(
    response: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.zeros_like(response)
    means = np.zeros_like(response)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        value = response[mask].mean(axis=0, keepdims=True)
        centered[mask] = response[mask] - value
        means[mask] = value
    return centered, means


def event_design(
    event_day: np.ndarray,
    group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = spline_basis(event_day, 4, -24.0, 43.0)
    group_indicator = (np.asarray(group) == "nec").astype(float)[:, None]
    design = np.column_stack(
        [
            np.ones(len(event_day)),
            group_indicator,
            basis,
            group_indicator * basis,
        ]
    )
    penalty = np.asarray(
        [0.0, 0.0] + [1.0] * basis.shape[1] + [1.0] * basis.shape[1]
    )
    return design, penalty


def event_prediction_design(
    event_grid: np.ndarray,
    group: str,
) -> np.ndarray:
    basis = spline_basis(event_grid, 4, -24.0, 43.0)
    indicator = np.ones((len(event_grid), 1)) if group == "nec" else np.zeros(
        (len(event_grid), 1)
    )
    return np.column_stack(
        [
            np.ones(len(event_grid)),
            indicator,
            basis,
            indicator * basis,
        ]
    )


def trajectory_metrics(
    prediction_nec: np.ndarray,
    prediction_sepsis: np.ndarray,
) -> np.ndarray:
    difference = prediction_nec - prediction_sepsis
    event_index = int(np.argmin(np.abs(EVENT_GRID - 0.0)))
    day_14_index = int(np.argmin(np.abs(EVENT_GRID - 14.0)))
    pre_14_index = int(np.argmin(np.abs(EVENT_GRID + 14.0)))
    peri = (EVENT_GRID >= -7.0) & (EVENT_GRID <= 7.0)
    return np.vstack(
        [
            difference[event_index],
            difference[peri].mean(axis=0),
            difference[day_14_index],
            prediction_nec[event_index] - prediction_nec[pre_14_index],
            prediction_nec[day_14_index] - prediction_nec[event_index],
            np.sqrt(np.mean(difference**2, axis=0)),
        ]
    ).T


def window_summary(
    prediction_nec: np.ndarray,
    prediction_sepsis: np.ndarray,
) -> pd.DataFrame:
    windows = {
        "pre_illness": (-14.0, -1.0),
        "acute": (0.0, 7.0),
        "recovery": (8.0, 21.0),
    }
    rows = []
    for name, (lower, upper) in windows.items():
        mask = (EVENT_GRID >= lower) & (EVENT_GRID <= upper)
        nec = prediction_nec[mask].mean(axis=0)
        sepsis = prediction_sepsis[mask].mean(axis=0)
        for index in range(prediction_nec.shape[1]):
            rows.append(
                {
                    "window": name,
                    "event_day_lower": lower,
                    "event_day_upper": upper,
                    "feature_index": index,
                    "nec_estimate": float(nec[index]),
                    "sepsis_estimate": float(sepsis[index]),
                    "nec_minus_sepsis": float(nec[index] - sepsis[index]),
                }
            )
    return pd.DataFrame(rows)


def select_curve_features(
    modality: str,
    metrics: np.ndarray,
    display_features: list[str],
) -> np.ndarray:
    score = np.abs(metrics[:, 1])
    target_names = TARGET_GENERA if modality == "microbiome" else TARGET_METABOLITES
    selected: list[int] = []
    for target in target_names:
        matches = [
            index
            for index, feature in enumerate(display_features)
            if feature.casefold() == target.casefold()
        ]
        selected.extend(matches)
    ranked = np.argsort(score)[::-1]
    selected.extend(ranked[:8].tolist())
    limit = 16 if modality == "metabolome" else 14
    output = []
    for index in selected:
        if index not in output:
            output.append(index)
        if len(output) >= limit:
            break
    return np.asarray(output, dtype=int)


def resample_clusters(
    patients: np.ndarray,
    groups: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[int] = []
    bootstrap_ids: list[str] = []
    strata = ["all"] if groups is None else sorted(np.unique(groups).tolist())
    for stratum in strata:
        eligible_rows = np.arange(len(patients))
        if groups is not None:
            eligible_rows = eligible_rows[groups == stratum]
        eligible_patients = np.unique(patients[eligible_rows])
        draws = rng.choice(
            eligible_patients,
            size=len(eligible_patients),
            replace=True,
        )
        for draw_index, patient in enumerate(draws):
            patient_rows = eligible_rows[patients[eligible_rows] == patient]
            rows.extend(patient_rows.tolist())
            bootstrap_ids.extend(
                [f"{stratum}:{draw_index}:{patient}"] * len(patient_rows)
            )
    return np.asarray(rows, dtype=int), np.asarray(bootstrap_ids)


def fit_age_baseline(
    metadata: pd.DataFrame,
    response: np.ndarray,
    include_gi: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control_group = {"control", "GI"} if include_gi else {"control"}
    control_mask = metadata["analysis_group"].isin(control_group).to_numpy()
    age = pd.to_numeric(
        metadata["Age_at_sample_collection"],
        errors="raise",
    ).to_numpy(dtype=float)
    basis = spline_basis(age, 4, 7.0, 75.0)
    design = np.column_stack([np.ones(len(age)), basis])
    penalty = np.asarray([0.0] + [1.0] * basis.shape[1])
    clusters = metadata["Patient"].to_numpy()
    lambdas, cv_scores = select_lambda_leave_one_cluster_out(
        design[control_mask],
        response[control_mask],
        clusters[control_mask],
        penalty,
    )
    coefficients = fit_ridge_by_lambda(
        design[control_mask],
        response[control_mask],
        clusters[control_mask],
        lambdas,
        penalty,
    )
    predictions = design @ coefficients
    return predictions, lambdas, cv_scores, design, penalty


def point_event_fit(
    metadata: pd.DataFrame,
    response: np.ndarray,
    age_predictions: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sick_mask = metadata["analysis_group"].isin({"nec", "sepsis"}).to_numpy()
    sick_mask &= metadata["day_relative_sick"].notna().to_numpy()
    sick_metadata = metadata.loc[sick_mask].reset_index(drop=True)
    residual = response[sick_mask] - age_predictions[sick_mask]
    patients = sick_metadata["Patient"].to_numpy()
    within, between = within_cluster_center(residual, patients)
    design, penalty = event_design(
        sick_metadata["day_relative_sick"].to_numpy(dtype=float),
        sick_metadata["analysis_group"].to_numpy(),
    )
    lambdas, cv_scores = select_lambda_leave_one_cluster_out(
        design,
        within,
        patients,
        penalty,
    )
    coefficients = fit_ridge_by_lambda(
        design,
        within,
        patients,
        lambdas,
        penalty,
    )
    prediction_nec = event_prediction_design(EVENT_GRID, "nec") @ coefficients
    prediction_sepsis = (
        event_prediction_design(EVENT_GRID, "sepsis") @ coefficients
    )
    return (
        sick_mask,
        within,
        between,
        lambdas,
        cv_scores,
        coefficients,
        prediction_nec,
        prediction_sepsis,
    )


def bootstrap_trajectories(
    metadata: pd.DataFrame,
    response: np.ndarray,
    age_design: np.ndarray,
    age_penalty: np.ndarray,
    age_lambdas: np.ndarray,
    sick_mask: np.ndarray,
    event_lambdas: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    control_mask = metadata["analysis_group"].eq("control").to_numpy()
    control_rows = np.flatnonzero(control_mask)
    sick_rows_global = np.flatnonzero(sick_mask)
    control_patients = metadata.loc[control_mask, "Patient"].to_numpy()
    sick_metadata = metadata.loc[sick_mask].reset_index(drop=True)
    sick_patients = sick_metadata["Patient"].to_numpy()
    sick_groups = sick_metadata["analysis_group"].to_numpy()
    event_matrix, event_penalty = event_design(
        sick_metadata["day_relative_sick"].to_numpy(dtype=float),
        sick_groups,
    )
    prediction_design_nec = event_prediction_design(EVENT_GRID, "nec")
    prediction_design_sepsis = event_prediction_design(EVENT_GRID, "sepsis")

    metric_bootstrap = np.full(
        (BOOTSTRAPS, response.shape[1], len(METRIC_NAMES)),
        np.nan,
    )
    curve_nec = np.full(
        (BOOTSTRAPS, len(EVENT_GRID), len(selected_indices)),
        np.nan,
    )
    curve_sepsis = np.full_like(curve_nec, np.nan)
    for bootstrap in range(BOOTSTRAPS):
        sampled_control_rows, control_ids = resample_clusters(
            control_patients,
            None,
            rng,
        )
        sampled_control_global = control_rows[sampled_control_rows]
        age_coefficients = fit_ridge_by_lambda(
            age_design[sampled_control_global],
            response[sampled_control_global],
            control_ids,
            age_lambdas,
            age_penalty,
        )
        sick_age_prediction = age_design[sick_rows_global] @ age_coefficients
        sick_residual = response[sick_rows_global] - sick_age_prediction
        sick_within, _ = within_cluster_center(sick_residual, sick_patients)
        sampled_sick_rows, sick_ids = resample_clusters(
            sick_patients,
            sick_groups,
            rng,
        )
        event_coefficients = fit_ridge_by_lambda(
            event_matrix[sampled_sick_rows],
            sick_within[sampled_sick_rows],
            sick_ids,
            event_lambdas,
            event_penalty,
        )
        predicted_nec = prediction_design_nec @ event_coefficients
        predicted_sepsis = prediction_design_sepsis @ event_coefficients
        metric_bootstrap[bootstrap] = trajectory_metrics(
            predicted_nec,
            predicted_sepsis,
        )
        curve_nec[bootstrap] = predicted_nec[:, selected_indices]
        curve_sepsis[bootstrap] = predicted_sepsis[:, selected_indices]
    return metric_bootstrap, curve_nec, curve_sepsis


def leave_one_sick_infant_out(
    metadata: pd.DataFrame,
    sick_mask: np.ndarray,
    within: np.ndarray,
    event_lambdas: np.ndarray,
    full_metrics: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    sick_metadata = metadata.loc[sick_mask].reset_index(drop=True)
    patients = sick_metadata["Patient"].to_numpy()
    groups = sick_metadata["analysis_group"].to_numpy()
    design, penalty = event_design(
        sick_metadata["day_relative_sick"].to_numpy(dtype=float),
        groups,
    )
    prediction_design_nec = event_prediction_design(EVENT_GRID, "nec")
    prediction_design_sepsis = event_prediction_design(EVENT_GRID, "sepsis")
    rows = []
    metric_stack = []
    for patient in np.unique(patients):
        keep = patients != patient
        coefficients = fit_ridge_by_lambda(
            design[keep],
            within[keep],
            patients[keep],
            event_lambdas,
            penalty,
        )
        metrics = trajectory_metrics(
            prediction_design_nec @ coefficients,
            prediction_design_sepsis @ coefficients,
        )
        metric_stack.append(metrics)
        for feature_index in range(within.shape[1]):
            rows.append(
                {
                    "held_out_patient": int(patient),
                    "held_out_group": str(
                        sick_metadata.loc[
                            sick_metadata["Patient"].eq(patient),
                            "analysis_group",
                        ].iloc[0]
                    ),
                    "feature_index": feature_index,
                    "delta_at_event": float(metrics[feature_index, 0]),
                    "peri_event_auc": float(metrics[feature_index, 1]),
                    "nec_event_to_recovery": float(metrics[feature_index, 4]),
                }
            )
    stacked = np.stack(metric_stack, axis=0)
    stability = np.mean(
        np.sign(stacked) == np.sign(full_metrics)[None, :, :],
        axis=0,
    )
    return pd.DataFrame(rows), stability


def exact_group_label_permutation(
    metadata: pd.DataFrame,
    sick_mask: np.ndarray,
    within: np.ndarray,
    observed_metrics: np.ndarray,
) -> tuple[np.ndarray, int]:
    sick_metadata = metadata.loc[sick_mask].reset_index(drop=True)
    patients = sick_metadata["Patient"].to_numpy()
    event_day = sick_metadata["day_relative_sick"].to_numpy(dtype=float)
    observed_groups = sick_metadata["analysis_group"].to_numpy()
    unique_patients = np.unique(patients)
    nec_patients = np.unique(patients[observed_groups == "nec"])
    assignments = list(
        itertools.combinations(unique_patients.tolist(), len(nec_patients))
    )
    permutation_metrics = np.full(
        (len(assignments), within.shape[1], 3),
        np.nan,
    )
    prediction_design_nec = event_prediction_design(EVENT_GRID, "nec")
    prediction_design_sepsis = event_prediction_design(EVENT_GRID, "sepsis")
    for assignment_index, assignment in enumerate(assignments):
        permuted_group = np.where(
            np.isin(patients, np.asarray(assignment)),
            "nec",
            "sepsis",
        )
        design, penalty = event_design(event_day, permuted_group)
        lambdas, _ = select_lambda_leave_one_cluster_out(
            design,
            within,
            patients,
            penalty,
        )
        coefficients = fit_ridge_by_lambda(
            design,
            within,
            patients,
            lambdas,
            penalty,
        )
        metrics = trajectory_metrics(
            prediction_design_nec @ coefficients,
            prediction_design_sepsis @ coefficients,
        )
        permutation_metrics[assignment_index] = metrics[:, :3]
    pvalues = np.mean(
        np.abs(permutation_metrics)
        >= np.abs(observed_metrics[None, :, :3]) - 1e-12,
        axis=0,
    )
    return pvalues, len(assignments)


def feeding_sensitivity(
    metadata: pd.DataFrame,
    sick_mask: np.ndarray,
    within: np.ndarray,
    event_lambdas: np.ndarray,
) -> pd.DataFrame:
    sick_metadata = metadata.loc[sick_mask].reset_index(drop=True)
    design, penalty = event_design(
        sick_metadata["day_relative_sick"].to_numpy(dtype=float),
        sick_metadata["analysis_group"].to_numpy(),
    )
    prediction_design_nec = event_prediction_design(EVENT_GRID, "nec")
    prediction_design_sepsis = event_prediction_design(EVENT_GRID, "sepsis")
    rows = []
    for feeding in sorted(sick_metadata["feeding_type"].dropna().unique()):
        mask = sick_metadata["feeding_type"].eq(feeding).to_numpy()
        patient_counts = (
            sick_metadata.loc[mask]
            .groupby("analysis_group", observed=True)["Patient"]
            .nunique()
            .to_dict()
        )
        if patient_counts.get("nec", 0) < 2 or patient_counts.get("sepsis", 0) < 2:
            rows.append(
                {
                    "feeding_type": feeding,
                    "status": "not_estimable_fewer_than_two_infants_per_group",
                    "nec_infants": patient_counts.get("nec", 0),
                    "sepsis_infants": patient_counts.get("sepsis", 0),
                    "feature_index": np.nan,
                }
            )
            continue
        coefficients = fit_ridge_by_lambda(
            design[mask],
            within[mask],
            sick_metadata.loc[mask, "Patient"].to_numpy(),
            event_lambdas,
            penalty,
        )
        metrics = trajectory_metrics(
            prediction_design_nec @ coefficients,
            prediction_design_sepsis @ coefficients,
        )
        for feature_index in range(within.shape[1]):
            rows.append(
                {
                    "feeding_type": feeding,
                    "status": "estimated",
                    "nec_infants": patient_counts["nec"],
                    "sepsis_infants": patient_counts["sepsis"],
                    "feature_index": feature_index,
                    "delta_at_event": float(metrics[feature_index, 0]),
                    "peri_event_auc": float(metrics[feature_index, 1]),
                    "nec_event_to_recovery": float(metrics[feature_index, 4]),
                }
            )
    return pd.DataFrame(rows)


def analyze_modality(
    metadata: pd.DataFrame,
    transformed: pd.DataFrame,
    features: list[str],
    display_features: list[str],
    modality: str,
    preprocessing: dict[str, object],
) -> ModalityResult:
    sample_metadata = metadata[
        metadata["Sample"].isin(transformed["Sample"])
    ].copy()
    sample_metadata = sample_metadata.sort_values("Sample").reset_index(drop=True)
    aligned = transformed.set_index("Sample").loc[
        sample_metadata["Sample"],
        features,
    ]
    response = aligned.to_numpy(dtype=float)
    age_predictions, age_lambdas, _, age_design, age_penalty = fit_age_baseline(
        sample_metadata,
        response,
        include_gi=False,
    )
    (
        sick_mask,
        within,
        between,
        event_lambdas,
        _,
        coefficients,
        prediction_nec,
        prediction_sepsis,
    ) = point_event_fit(sample_metadata, response, age_predictions)
    point_metrics = trajectory_metrics(prediction_nec, prediction_sepsis)
    selected_indices = select_curve_features(
        modality,
        point_metrics,
        display_features,
    )
    bootstrap_metrics, bootstrap_nec, bootstrap_sepsis = bootstrap_trajectories(
        sample_metadata,
        response,
        age_design,
        age_penalty,
        age_lambdas,
        sick_mask,
        event_lambdas,
        selected_indices,
    )
    loo_long, loo_stability = leave_one_sick_infant_out(
        sample_metadata,
        sick_mask,
        within,
        event_lambdas,
        point_metrics,
    )
    group_pvalues, group_permutations = exact_group_label_permutation(
        sample_metadata,
        sick_mask,
        within,
        point_metrics,
    )

    metrics = pd.DataFrame(
        {
            "modality": modality,
            "feature": features,
            "display_feature": display_features,
            "age_lambda": age_lambdas,
            "event_lambda": event_lambdas,
        }
    )
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        values = bootstrap_metrics[:, :, metric_index]
        metrics[metric_name] = point_metrics[:, metric_index]
        metrics[f"{metric_name}_ci_low"] = np.nanquantile(values, 0.025, axis=0)
        metrics[f"{metric_name}_ci_high"] = np.nanquantile(values, 0.975, axis=0)
        if metric_index < 3:
            pvalue = group_pvalues[:, metric_index]
            metrics[f"{metric_name}_pvalue"] = pvalue
            metrics[f"{metric_name}_fdr"] = bh_adjust(pvalue)
            metrics[f"{metric_name}_inference"] = (
                f"exact infant-label permutation; {group_permutations} assignments"
            )
        elif metric_index < 5:
            metrics[f"{metric_name}_inference"] = (
                "infant bootstrap confidence interval and leave-one-infant-out "
                "directional stability"
            )
        metrics[f"{metric_name}_loo_sign_stability"] = loo_stability[
            :,
            metric_index,
        ]

    curve_rows = []
    selected_features = [features[index] for index in selected_indices]
    selected_display = [display_features[index] for index in selected_indices]
    for selected_position, feature_index in enumerate(selected_indices):
        for group, estimate, bootstrap_curve in [
            (
                "NEC",
                prediction_nec[:, feature_index],
                bootstrap_nec[:, :, selected_position],
            ),
            (
                "Sepsis",
                prediction_sepsis[:, feature_index],
                bootstrap_sepsis[:, :, selected_position],
            ),
        ]:
            lower = np.nanquantile(bootstrap_curve, 0.025, axis=0)
            upper = np.nanquantile(bootstrap_curve, 0.975, axis=0)
            for event_day, point, low, high in zip(
                EVENT_GRID,
                estimate,
                lower,
                upper,
            ):
                curve_rows.append(
                    {
                        "modality": modality,
                        "feature": features[feature_index],
                        "display_feature": display_features[feature_index],
                        "group": group,
                        "event_day": float(event_day),
                        "estimate": float(point),
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
    curves = pd.DataFrame(curve_rows)

    windows = window_summary(prediction_nec, prediction_sepsis)
    windows["modality"] = modality
    windows["feature"] = windows["feature_index"].map(dict(enumerate(features)))
    windows["display_feature"] = windows["feature_index"].map(
        dict(enumerate(display_features))
    )

    loo_long["modality"] = modality
    loo_long["feature"] = loo_long["feature_index"].map(dict(enumerate(features)))
    loo_long["display_feature"] = loo_long["feature_index"].map(
        dict(enumerate(display_features))
    )

    feeding = feeding_sensitivity(
        sample_metadata,
        sick_mask,
        within,
        event_lambdas,
    )
    if "feature_index" in feeding.columns:
        feeding["modality"] = modality
        feeding["feature"] = feeding["feature_index"].map(
            dict(enumerate(features))
        )
        feeding["display_feature"] = feeding["feature_index"].map(
            dict(enumerate(display_features))
        )

    gi_age_predictions, _, _, _, _ = fit_age_baseline(
        sample_metadata,
        response,
        include_gi=True,
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        gi_nec,
        gi_sepsis,
    ) = point_event_fit(sample_metadata, response, gi_age_predictions)
    gi_metrics = trajectory_metrics(gi_nec, gi_sepsis)
    gi_sensitivity = pd.DataFrame(
        {
            "modality": modality,
            "feature": features,
            "display_feature": display_features,
            "primary_delta_at_event": point_metrics[:, 0],
            "gi_inclusive_delta_at_event": gi_metrics[:, 0],
            "absolute_change": np.abs(gi_metrics[:, 0] - point_metrics[:, 0]),
            "direction_concordant": (
                np.sign(gi_metrics[:, 0]) == np.sign(point_metrics[:, 0])
            ),
        }
    )

    full_age_residual = response - age_predictions
    full_within = np.full_like(response, np.nan)
    full_within[sick_mask] = within
    return ModalityResult(
        modality=modality,
        features=features,
        display_features=display_features,
        sample_metadata=sample_metadata,
        transformed=response,
        age_residual=full_age_residual,
        within_deviation=full_within,
        metrics=metrics,
        curves=curves,
        windows=windows,
        loo=loo_long,
        feeding=feeding,
        gi_sensitivity=gi_sensitivity,
        selected_indices=selected_indices,
        preprocessing={
            **preprocessing,
            "selected_curve_features": selected_features,
            "selected_curve_display_features": selected_display,
            "bootstrap_replicates": BOOTSTRAPS,
            "group_permutation_assignments": group_permutations,
            "primary_controls": "Group == control; GI perforation excluded",
            "event_groups": ["nec", "sepsis"],
            "antibiotic_sensitivity": "not_estimable_no_variation_among_sick_samples",
        },
    )


def weighted_partial_spearman_matrix(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    x_rank = np.column_stack([rankdata(x[:, column]) for column in range(x.shape[1])])
    y_rank = np.column_stack([rankdata(y[:, column]) for column in range(y.shape[1])])
    design = np.column_stack(
        [np.ones(len(group)), (np.asarray(group) == "nec").astype(float)]
    )
    weighted = design * weights[:, None]
    inverse = np.linalg.pinv(design.T @ weighted)
    x_residual = x_rank - design @ (
        inverse @ (design.T @ (x_rank * weights[:, None]))
    )
    y_residual = y_rank - design @ (
        inverse @ (design.T @ (y_rank * weights[:, None]))
    )
    x_mean = np.sum(x_residual * weights[:, None], axis=0) / weights.sum()
    y_mean = np.sum(y_residual * weights[:, None], axis=0) / weights.sum()
    x_centered = x_residual - x_mean
    y_centered = y_residual - y_mean
    covariance = x_centered.T @ (y_centered * weights[:, None]) / weights.sum()
    x_scale = np.sqrt(
        np.sum((x_centered**2) * weights[:, None], axis=0) / weights.sum()
    )
    y_scale = np.sqrt(
        np.sum((y_centered**2) * weights[:, None], axis=0) / weights.sum()
    )
    denominator = x_scale[:, None] * y_scale[None, :]
    return np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )


def dynamic_coupling(
    microbiome: ModalityResult,
    metabolome: ModalityResult,
) -> pd.DataFrame:
    micro_map = {
        value.casefold(): index
        for index, value in enumerate(microbiome.display_features)
    }
    micro_indices = [
        micro_map[value.casefold()]
        for value in TARGET_GENERA
        if value.casefold() in micro_map
    ]
    top_micro = (
        microbiome.metrics.assign(
            _score=lambda frame: frame["peri_event_auc"].abs()
        )
        .sort_values("_score", ascending=False)
        .index.tolist()
    )
    for index in top_micro:
        if index not in micro_indices:
            micro_indices.append(index)
        if len(micro_indices) >= 8:
            break
    metabolite_map = {
        value.casefold(): index
        for index, value in enumerate(metabolome.display_features)
    }
    metabolite_indices = [
        metabolite_map[value.casefold()]
        for value in TARGET_METABOLITES
        if value.casefold() in metabolite_map
    ]

    common_samples = sorted(
        set(microbiome.sample_metadata["Sample"])
        & set(metabolome.sample_metadata["Sample"])
    )
    micro_positions = (
        microbiome.sample_metadata.reset_index()
        .set_index("Sample")
        .loc[common_samples, "index"]
        .to_numpy()
    )
    metabolite_positions = (
        metabolome.sample_metadata.reset_index()
        .set_index("Sample")
        .loc[common_samples, "index"]
        .to_numpy()
    )
    metadata = (
        microbiome.sample_metadata.set_index("Sample")
        .loc[common_samples]
        .reset_index()
    )
    sick = metadata["analysis_group"].isin({"nec", "sepsis"}).to_numpy()
    metadata = metadata.loc[sick].reset_index(drop=True)
    micro_positions = micro_positions[sick]
    metabolite_positions = metabolite_positions[sick]
    x = microbiome.within_deviation[
        np.ix_(micro_positions, micro_indices)
    ]
    y = metabolome.within_deviation[
        np.ix_(metabolite_positions, metabolite_indices)
    ]
    finite_rows = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    metadata = metadata.loc[finite_rows].reset_index(drop=True)
    x = x[finite_rows]
    y = y[finite_rows]

    windows = {
        "pre_illness": (-24.0, -1.0),
        "acute": (0.0, 7.0),
        "recovery": (8.0, 30.0),
    }
    window_data: dict[
        str,
        tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ] = {}
    point: dict[str, np.ndarray] = {}
    bootstrap: dict[str, np.ndarray] = {
        name: np.full(
            (COUPLING_BOOTSTRAPS, len(micro_indices), len(metabolite_indices)),
            np.nan,
        )
        for name in windows
    }
    permutation: dict[str, np.ndarray] = {
        name: np.full(
            (COUPLING_PERMUTATIONS, len(micro_indices), len(metabolite_indices)),
            np.nan,
        )
        for name in windows
    }
    for name, (lower, upper) in windows.items():
        mask = metadata["day_relative_sick"].between(lower, upper).to_numpy()
        window_metadata = metadata.loc[
            mask,
            ["Patient", "analysis_group"],
        ].reset_index(drop=True)
        window_x = x[mask]
        window_y = y[mask]
        patient_rows = []
        patient_x = []
        patient_y = []
        for patient in sorted(window_metadata["Patient"].unique()):
            patient_mask = window_metadata["Patient"].eq(patient).to_numpy()
            patient_rows.append(
                {
                    "Patient": int(patient),
                    "analysis_group": str(
                        window_metadata.loc[
                            patient_mask,
                            "analysis_group",
                        ].iloc[0]
                    ),
                }
            )
            patient_x.append(window_x[patient_mask].mean(axis=0))
            patient_y.append(window_y[patient_mask].mean(axis=0))
        patient_metadata = pd.DataFrame(patient_rows)
        patient_x_array = np.vstack(patient_x)
        patient_y_array = np.vstack(patient_y)
        window_data[name] = (
            patient_metadata,
            patient_x_array,
            patient_y_array,
        )
        point[name] = weighted_partial_spearman_matrix(
            patient_x_array,
            patient_y_array,
            np.ones(len(patient_metadata)),
            patient_metadata["analysis_group"].to_numpy(),
        )

    rng = np.random.default_rng(SEED + 1)
    for name, (
        patient_metadata,
        patient_x,
        patient_y,
    ) in window_data.items():
        patients = patient_metadata["Patient"].to_numpy()
        groups = patient_metadata["analysis_group"].to_numpy()
        for replicate in range(COUPLING_BOOTSTRAPS):
            sampled_rows, bootstrap_ids = resample_clusters(
                patients,
                groups,
                rng,
            )
            bootstrap[name][replicate] = weighted_partial_spearman_matrix(
                patient_x[sampled_rows],
                patient_y[sampled_rows],
                equal_cluster_weights(bootstrap_ids),
                groups[sampled_rows],
            )
        for replicate in range(COUPLING_PERMUTATIONS):
            permuted_rows = np.arange(len(patient_metadata))
            for group in np.unique(groups):
                group_rows = np.flatnonzero(groups == group)
                permuted_rows[group_rows] = rng.permutation(group_rows)
            permutation[name][replicate] = weighted_partial_spearman_matrix(
                patient_x,
                patient_y[permuted_rows],
                np.ones(len(patient_metadata)),
                groups,
            )

    rows = []
    micro_labels = [microbiome.display_features[index] for index in micro_indices]
    metabolite_labels = [
        metabolome.display_features[index] for index in metabolite_indices
    ]
    for window in windows:
        boot = bootstrap[window]
        permuted = permutation[window]
        pvalue = (
            1.0
            + np.sum(
                np.abs(permuted)
                >= np.abs(point[window])[None, :, :] - 1e-12,
                axis=0,
            )
        ) / (COUPLING_PERMUTATIONS + 1.0)
        patient_metadata, _, _ = window_data[window]
        for micro_index, micro_label in enumerate(micro_labels):
            for metabolite_index, metabolite_label in enumerate(
                metabolite_labels
            ):
                rows.append(
                    {
                        "window": window,
                        "taxon": micro_label,
                        "metabolite": metabolite_label,
                        "partial_spearman_rho": float(
                            point[window][micro_index, metabolite_index]
                        ),
                        "ci_low": float(
                            np.nanquantile(
                                boot[:, micro_index, metabolite_index],
                                0.025,
                            )
                        ),
                        "ci_high": float(
                            np.nanquantile(
                                boot[:, micro_index, metabolite_index],
                                0.975,
                            )
                        ),
                        "pvalue": float(
                            pvalue[micro_index, metabolite_index]
                        ),
                        "samples": int(
                            metadata["day_relative_sick"]
                            .between(*windows[window])
                            .sum()
                        ),
                        "infants": int(
                            patient_metadata["Patient"].nunique()
                        ),
                        "inference": (
                            "patient-level within-group metabolite-label "
                            f"permutation; {COUPLING_PERMUTATIONS} replicates"
                        ),
                    }
                )
    static_result = pd.DataFrame(rows)
    static_result["fdr"] = bh_adjust(static_result["pvalue"].to_numpy())

    acute_minus_pre = bootstrap["acute"] - bootstrap["pre_illness"]
    recovery_minus_acute = bootstrap["recovery"] - bootstrap["acute"]
    change_rows = []
    for micro_index, micro_label in enumerate(micro_labels):
        for metabolite_index, metabolite_label in enumerate(metabolite_labels):
            for name, point_value, values in [
                (
                    "acute_minus_pre",
                    point["acute"][micro_index, metabolite_index]
                    - point["pre_illness"][micro_index, metabolite_index],
                    acute_minus_pre[:, micro_index, metabolite_index],
                ),
                (
                    "recovery_minus_acute",
                    point["recovery"][micro_index, metabolite_index]
                    - point["acute"][micro_index, metabolite_index],
                    recovery_minus_acute[:, micro_index, metabolite_index],
                ),
            ]:
                change_rows.append(
                    {
                        "window": name,
                        "taxon": micro_label,
                        "metabolite": metabolite_label,
                        "partial_spearman_rho": float(point_value),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "pvalue": np.nan,
                        "samples": np.nan,
                        "infants": int(
                            len(
                                set(window_data["pre_illness"][0]["Patient"])
                                | set(window_data["acute"][0]["Patient"])
                                | set(window_data["recovery"][0]["Patient"])
                            )
                        ),
                        "inference": (
                            "patient bootstrap contrast; confidence interval only"
                        ),
                    }
                )
    changes = pd.DataFrame(change_rows)
    changes["fdr"] = np.nan
    return pd.concat([static_result, changes], ignore_index=True)


def build_decomposition_table(
    result: ModalityResult,
) -> pd.DataFrame:
    selected = result.selected_indices
    rows = []
    for position, metadata_row in result.sample_metadata.iterrows():
        for feature_index in selected:
            rows.append(
                {
                    "modality": result.modality,
                    "sample": int(metadata_row["Sample"]),
                    "patient": int(metadata_row["Patient"]),
                    "analysis_group": metadata_row["analysis_group"],
                    "age_at_sample_collection": metadata_row[
                        "Age_at_sample_collection"
                    ],
                    "day_relative_sick": metadata_row["day_relative_sick"],
                    "feeding_type": metadata_row["feeding_type"],
                    "feature": result.features[feature_index],
                    "display_feature": result.display_features[feature_index],
                    "transformed_value": float(
                        result.transformed[position, feature_index]
                    ),
                    "control_age_residual": float(
                        result.age_residual[position, feature_index]
                    ),
                    "within_infant_deviation": float(
                        result.within_deviation[position, feature_index]
                    )
                    if np.isfinite(
                        result.within_deviation[position, feature_index]
                    )
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def select_figure_source_data(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    decomposition: pd.DataFrame,
    coupling: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ranked = metrics.assign(
        ranking_score=lambda frame: (
            frame["peri_event_auc"].abs()
            * frame["peri_event_auc_loo_sign_stability"]
        )
    )
    selected_rows = []
    for modality, requested in [
        ("microbiome", FIGURE_GENERA),
        ("metabolome", FIGURE_METABOLITES),
    ]:
        group = ranked[ranked["modality"].eq(modality)].set_index(
            "display_feature"
        )
        missing = [feature for feature in requested if feature not in group.index]
        if missing:
            raise RuntimeError(
                f"Missing requested TEMPO-NEC figure features: {missing}"
            )
        selected_rows.append(group.loc[requested].reset_index())
    selected_metrics = pd.concat(selected_rows, ignore_index=True)
    selected_metrics["figure_order"] = np.arange(len(selected_metrics))

    panel_a = curves[curves["display_feature"].isin(FIGURE_CURVES)].copy()
    panel_a["figure_order"] = panel_a["display_feature"].map(
        {feature: index for index, feature in enumerate(FIGURE_CURVES)}
    )
    panel_a = panel_a.sort_values(
        ["figure_order", "group", "event_day"]
    ).reset_index(drop=True)
    panel_b = selected_metrics.copy()
    panel_d = decomposition[
        decomposition["display_feature"].isin(FIGURE_CURVES)
        & decomposition["analysis_group"].isin(["nec", "sepsis"])
    ].copy()
    panel_d["figure_order"] = panel_d["display_feature"].map(
        {feature: index for index, feature in enumerate(FIGURE_CURVES)}
    )
    panel_c = coupling[coupling["window"].eq("acute_minus_pre")].copy()
    panel_c["absolute_change"] = panel_c["partial_spearman_rho"].abs()
    return {
        "selected_metrics": selected_metrics,
        "panel_a_curves": panel_a,
        "panel_b_metric_matrix": panel_b,
        "panel_c_dynamic_coupling": panel_c,
        "panel_d_sample_decomposition": panel_d,
    }


def main() -> None:
    for directory in [RESULT_ROOT, SOURCE_ROOT, MANIFEST_ROOT]:
        ensure_local_write(directory).mkdir(parents=True, exist_ok=True)
    metadata, genus, metabolite, input_audit = load_author_inputs()
    genus_transformed, genus_features, genus_display, genus_preprocessing = (
        transform_genus(genus)
    )
    (
        metabolite_transformed,
        metabolite_features,
        metabolite_display,
        metabolite_preprocessing,
    ) = transform_metabolites(metabolite)

    microbiome = analyze_modality(
        metadata,
        genus_transformed,
        genus_features,
        genus_display,
        "microbiome",
        genus_preprocessing,
    )
    metabolome = analyze_modality(
        metadata,
        metabolite_transformed,
        metabolite_features,
        metabolite_display,
        "metabolome",
        metabolite_preprocessing,
    )
    coupling = dynamic_coupling(microbiome, metabolome)

    metrics = pd.concat(
        [microbiome.metrics, metabolome.metrics],
        ignore_index=True,
    )
    curves = pd.concat(
        [microbiome.curves, metabolome.curves],
        ignore_index=True,
    )
    windows = pd.concat(
        [microbiome.windows, metabolome.windows],
        ignore_index=True,
    )
    loo = pd.concat(
        [microbiome.loo, metabolome.loo],
        ignore_index=True,
    )
    feeding = pd.concat(
        [microbiome.feeding, metabolome.feeding],
        ignore_index=True,
    )
    gi_sensitivity = pd.concat(
        [microbiome.gi_sensitivity, metabolome.gi_sensitivity],
        ignore_index=True,
    )
    decomposition = pd.concat(
        [
            build_decomposition_table(microbiome),
            build_decomposition_table(metabolome),
        ],
        ignore_index=True,
    )

    output_tables = {
        "tempo_feature_trajectory_metrics.csv": metrics,
        "tempo_prediction_curves.csv": curves,
        "tempo_window_metrics.csv": windows,
        "tempo_leave_one_infant_out.csv": loo,
        "tempo_feeding_sensitivity.csv": feeding,
        "tempo_gi_control_sensitivity.csv": gi_sensitivity,
        "tempo_sample_decomposition.csv": decomposition,
        "tempo_dynamic_taxon_metabolite_coupling.csv": coupling,
    }
    for filename, frame in output_tables.items():
        frame.to_csv(ensure_local_write(RESULT_ROOT / filename), index=False)

    figure_data = select_figure_source_data(
        metrics,
        curves,
        decomposition,
        coupling,
    )
    source_filenames = {
        "selected_metrics": "ED_Figure_6_selected_features.csv",
        "panel_a_curves": "ED_Figure_6a_event_aligned_curves.csv",
        "panel_b_metric_matrix": "ED_Figure_6b_trajectory_metrics.csv",
        "panel_c_dynamic_coupling": (
            "ED_Figure_6c_dynamic_taxon_metabolite_coupling.csv"
        ),
        "panel_d_sample_decomposition": (
            "ED_Figure_6d_within_infant_deviations.csv"
        ),
    }
    for key, filename in source_filenames.items():
        figure_data[key].to_csv(
            ensure_local_write(SOURCE_ROOT / filename),
            index=False,
        )

    audit_rows = [
        {"check": key, "value": json.dumps(value) if isinstance(value, list) else value}
        for key, value in input_audit.items()
    ]
    pd.DataFrame(audit_rows).to_csv(
        ensure_local_write(RESULT_ROOT / "tempo_input_alignment_audit.csv"),
        index=False,
    )
    summary = {
        "status": "PASS",
        "analysis": "TEMPO-NEC",
        "analysis_version": "2026-07-26",
        "statistical_unit": "infant",
        "metadata_samples": input_audit["metadata_samples"],
        "primary_control_samples": input_audit["primary_control_samples"],
        "primary_control_infants": input_audit["primary_control_infants"],
        "nec_samples": input_audit["nec_samples"],
        "nec_infants": input_audit["nec_infants"],
        "sepsis_samples": input_audit["sepsis_samples"],
        "sepsis_infants": input_audit["sepsis_infants"],
        "microbiome_features": len(genus_features),
        "metabolome_features": len(metabolite_features),
        "bootstrap_replicates": BOOTSTRAPS,
        "coupling_bootstrap_replicates": COUPLING_BOOTSTRAPS,
        "coupling_permutation_replicates": COUPLING_PERMUTATIONS,
        "group_contrast_inference": (
            "exact infant-label permutation over all 120 assignments"
        ),
        "within_nec_inference": (
            "infant bootstrap confidence interval and leave-one-infant-out "
            "directional stability"
        ),
        "coupling_inference": (
            "patient-level within-group metabolite-label permutation"
        ),
        "microbiome_preprocessing": microbiome.preprocessing,
        "metabolome_preprocessing": metabolome.preprocessing,
        "minimum_peri_event_fdr": float(
            metrics["peri_event_auc_fdr"].min()
        ),
        "minimum_event_day_fdr": float(
            metrics["delta_at_event_fdr"].min()
        ),
        "high_stability_features": int(
            metrics["peri_event_auc_loo_sign_stability"].ge(0.8).sum()
        ),
        "dynamic_coupling_rows": int(coupling.shape[0]),
        "minimum_dynamic_coupling_fdr": float(coupling["fdr"].min()),
        "author_repository_commit": input_audit["repository_commit"],
        "legacy_direct_otu_alignment_status": input_audit[
            "legacy_direct_otu_alignment_status"
        ],
        "cloud_files_written": 0,
    }
    (RESULT_ROOT / "tempo_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "source_repository": "https://github.com/swandro/preterm_infant_analysis",
        "repository_commit": input_audit["repository_commit"],
        "files": [
            {
                "path": str(path.relative_to(ANALYSIS_ROOT)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(
                [
                    RDATA_PATH,
                    SYNC_ROOT / "wandro_author_infant_metadata.csv",
                    SYNC_ROOT / "wandro_author_genus_relative_abundance.csv",
                    SYNC_ROOT / "wandro_author_metabolite_intensity.csv",
                ]
            )
        ],
        "cloud_files_written": 0,
    }
    (MANIFEST_ROOT / "tempo_nec_input_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
